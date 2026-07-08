## Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation

Huajie Tan1,2,∗,†, Sixiang Chen1,2,∗, Yijie Xu2,3,∗, Zixiao Wang1,2, Yuheng Ji2,4, Cheng Chi2, Yaoxu Lyu1,2, Zhongxia Zhao1,2, Xiansheng Chen2, Peterson Co1,2,

Shaoxuan Xie2, Guocai Yao2, Pengwei Wang2,†, Zhongyuan Wang2, Shanghang Zhang1,2,

1 State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University 2 Beijing Academy of Artificial Intelligence, 3 University of Sydney, 4 Institute of Automation, Chinese Academy of Sciences

# arXiv:2512.23703v1[cs.RO]29Dec2025

Project website: https://robo-dopamine.github.io

Behavior Clone RL + Sparse Reward Dopamine-RL (Dense Reward)

Reward Modeling

Regress Hop Progress Hop

100

Task Success Rate (%)

###### + 8.6

-48% +64%

+ 38.3

Previous Progress

+ 55.0

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

70

+ 68.2

Pair-wise Progress

40

###### Potential Progress

10

[Figure 8]

Insert-Squares Stack-Three-Cubes Fold-the-Towels

Simulations

VLAC GVL

GRM (Ours)

Generate

General Reward Model (GRM) Construction

###### Dopamine-Reward

STEP1: Flatten the pants STEP2: Fold pants in half STEP3: Adjust STEP4: Fold pants in half again

N States N States N States N States

[Figure 9]

Observation

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Multi-Perspective Progress Fusion

[Figure 16]

[Figure 17]

[Figure 18]

Chunk-level Reward

[Figure 19]

[Figure 20]

[Figure 21]

|∆𝒕𝟏|
|---|

[Figure 22]

-0.07

+0.13

[Figure 23]

|∆𝒕𝟐|
|---|

∆𝒕𝟐

Reward Modeling

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Good Bad

[Figure 29]

[Figure 30]

| |
|---|

[Figure 31]

[Figure 32]

35M

|∆𝒕𝟏|
|---|

GRM One-Shot Task Adaptation

Policy

[Figure 33]

[Figure 34]

Act

Model

[Figure 35]

[Figure 36]

Policy-Invariant Reward Shaping

###### Obs

Adaptation and Shaping

Online Policy Learning

Figure 1. Overview of Robo-Dopamine. Robo-Dopamine integrates large-scale reward modeling with a robust policy learning algorithm. (Left) We construct a General Reward Model (GRM) trained on a large and diverse 35M-sample dataset spanning real-world, simulation, and human-centric videos with our Dopamine-Reward, a step-aware fine-grained reward modeling method. This GRM learns to predict fine-grained, relative progress between states to accurately assess task progression. (Bottom Right) The pre-trained GRM is adapted to new tasks and provides dense reward signals to our Dopamine-RL framework. By using a theoretically-sound Policy-Invariant Reward Shaping method, Dopamine-RL efficiently guides the policy during online interactions without misaligning the task objective. (Top Right) Our integrated approach establishes a new state-of-the-art in reward accuracy (radar chart) and demonstrates high training efficiency, significantly boosting policy success rates in both simulation and the real world (bar chart).

### Abstract

ward Models (PRMs) are a promising direction, they are often hindered by two fundamental limitations: their reward models lack step-aware understanding and rely on single-view perception, leading to unreliable assessments of fine-grained manipulation progress; and their reward shaping procedures are theoretically unsound, often inducing a semantic trap that misguides policy optimization. To address these, we introduce Dopamine-Reward,

The primary obstacle for applying reinforcement learning (RL) to real-world robotics is the design of effective reward functions. While recently learning-based Process Re-

∗ Equal contribution. † Project leaders.

Corresponding author: shanghang@pku.edu.cn

a novel reward modeling method for learning a generalpurpose, step-aware process reward model from multi-view inputs. At its core is our General Reward Model (GRM), trained on a vast 3,400+ hour dataset, which leverages Step-wise Reward Discretization for structural understanding and Multi-Perspective Reward Fusion to overcome perceptual limitations. Building upon Dopamine-Reward, we propose Dopamine-RL, a robust policy learning framework that employs a theoretically-sound Policy-Invariant Reward Shaping method, which enables the agent to leverage dense rewards for efficient self-improvement without altering the optimal policy, thereby fundamentally avoiding the semantic trap. Extensive experiments across diverse simulated and real-world tasks validate our approach. GRM achieves state-of-the-art accuracy in reward assessment, and Dopamine-RL built on GRM significantly improves policy learning efficiency. For instance, after GRM is adapted to a new task in a one-shot manner from a single expert trajectory, the resulting reward model enables Dopamine-RL to improve the policy from near-zero to 95% success with only 150 online rollouts (approximately 1 hour of real robot interaction), while retaining strong generalization across tasks. Project website: Robo-Dopamine.

### 1. Introduction

While large-scale imitation learning (IL) has substantially advanced embodied intelligence [1, 4, 5, 15, 21, 25], its reliance on static, expert-curated datasets imposes fundamental limitations [7, 24, 43, 52, 64, 65], which exhibits sub-optimal sample efficiency, poor generalization to outof-distribution (OOD) scenarios, and also struggles to acquire precise and contact-rich manipulation skills [23, 63]. In contrast, reinforcement learning (RL) offers a compelling alternative [11, 29, 33, 35, 56, 58, 60]. Through continuous environmental interaction, RL enables agents to transcend the limitations of static expert data, facilitating superior generalization and the mastery of high-precision tasks.

However, the primary obstacle for applying RL to realworld robotics is the design of effective reward functions. Conventional approaches falter at two extremes: sparse, binary outcome rewards [11, 33, 35, 56, 60] make exploration in long-horizon, contact-rich tasks prohibitively difficult, while handcrafted dense rewards [16, 44, 54, 55] require significant domain expertise, limiting scalability and general applicability. This dichotomy has motivated the shift towards learning-based Process Reward Models (PRMs) [2, 8, 36, 37, 59]. Despite their promise, current PRMs are hindered by two fundamental limitations. First, the underlying reward models often exhibit critical deficiencies: their task-specific design [8, 18] inherently limits generalization; uniform reward distributions [37, 59] fail to capture the varying salience of crucial sub-steps; and a reliance on

single-view observations [2, 8, 36, 37, 59] fails in manipulation scenes where occlusions obscure fine-grained progress only visible from wrist-level views. Second, the reward shaping algorithms utilizing these dense signals are often theoretically flawed. Naively incorporating dense rewards can induce a semantic trap [41] that misguides policy optimization by inadvertently altering the optimal policy, causing the agent to prioritize high proxy rewards from intermediate steps over the true task objective.

To address these, we introduce Dopamine-Reward, a novel dense reward modeling method for learning a generalpurpose, step-aware process reward from multi-view inputs. Dopamine-Reward directly tackles the first limitation by leveraging two key techniques: Hop-based Step-wise General Reward Model (GRM) Construction for a fine-grained, structural understanding of task progression from various viewpoints, and Multi-Perspective Reward Fusion via GRM to integrate bidirectional global reward and state-wise incremental reward for more precise reward estimation, which are made possible by a meticulous annotation pipeline encompassing over 3,400 hours of data, 100K trajectories, and more than 350 daily tasks, offering broad coverage, finegrained labels, and well-balanced distributions across real robots, simulations, and egocentric human videos.

Building upon GRM via Dopamine-Reward, we propose a robust and unified policy learning framework DopamineRL to resolve the second limitation. Dopamine-RL employs a theoretically-sound Policy-Invariant Reward Shaping method, which enables the agent to leverage the dense rewards from our GRM for highly efficient selfimprovement without altering the underlying optimal policy, thereby fundamentally avoiding the semantic trap. Extensive experiments on over 10 simulation and 8 real-world tasks demonstrate the superiority of our methods: (1) Stateof-the-art Reward Accuracy. The GRM achieves over 92.8% accuracy in progress assessment, with a Value-Order Consistency (VOC) score of 0.953 on rank-correlation benchmarks, outperforming established baselines. (2) High Training Efficiency. After GRM is adapted to a new task in a one-shot manner from a single expert demonstration, the resulting reward model enables Dopamine-RL to improve a policy from near-zero to 95% success rate within approximately 150 online rollouts (about one hour of real robot interaction), with some tasks reaching 100% success rate. (3) Improved Generalization. By combining step-wise structural modeling, reward fusion, and multi-view perception for robust estimation under occlusion and fine-grained state changes, our GRM provides more reliable learning signals, enabling Dopamine-RL to generalize more effectively to unseen layouts, backgrounds, and object variations.

An overview of Dopamine-Reward and Dopamine-RL, together with our empirical gains in reward accuracy and policy performance, is shown in Figure 1. In summary, our

main contributions are as follows:

- • We propose Dopamine-Reward, a novel reward modeling method built around a General Reward Model (GRM) that provides step-aware, fine-grained, and occlusion-resilient process rewards for precise robotic manipulation.
- • We introduce Dopamine-RL, a robust policy learning framework with a theoretically grounded Policy-Invariant Reward Shaping scheme, which effectively exploits dense GRM rewards to accelerate policy optimization while avoiding the semantic trap.
- • We curate a large-scale, 3,400-hour multi-view dataset with over 100K trajectories and 350 daily manipulation tasks across real robots, simulation, and egocentric human videos, offering broad coverage, fine-grained annotations, and balanced supervision for training GRM.
- • Extensive experiments validate our framework as follow: GRM achieves state-of-the-art reward assessment (over 92.8% progress accuracy and a 0.953 Value-Order Consistency score), while on 10 simulated and 8 realworld tasks, Dopamine-RL, after one-shot GRM adaptation, raises policy success from near-zero to 95% within 150 online rollouts (about one hour of robot interaction), with some tasks reaching 100% success and generalizing to unseen layouts, backgrounds, and object variations.

### 2. Related Work

Reinforcement Learning for Robotic Skills. Reinforcement Learning (RL) has demonstrated the potential to create policies that surpass the capabilities of imitation learning [11, 28, 29, 31, 33, 35, 56, 58, 60], enabling the discovery of novel and robust strategies for complex, contact-rich and dexterous tasks. Research in this area has progressed along two principal directions. The first direction investigates various policy optimization strategies, including offline RL [20, 34, 35], online RL [13, 29, 31, 32, 58], and mixed variants [11, 28, 38]. The second direction explores the efficient application of RL to different model architectures, such as small fully-connected models [35, 38], autoregressive models [11, 33, 56], and diffusion/flow-based models [31, 58, 62]. Independent of the chosen optimization algorithm or policy architecture, a more fundamental bottleneck is to design a reward function that is effective and scalable in real-world RL, which has driven a broad shift away from manual reward engineering [16, 44, 50, 54, 55] toward learning-based reward models.

Learned Process Reward Models. In real-world RL, a common practice is to train a success classifier as Outcome Reward Models (ORMs) to provide a binary reward signal [11, 35], which renders exploration prohibitively difficult in complex, long-horizon tasks. To mitigate sparsity, recent work leverages vision–language models (VLMs) as Process Reward Models (PRMs) [2, 8, 36, 37, 59], providing denser feedback by, for example, predicting progress

deltas between paired observations [59] or assigning perframe progress scores with respect to a language goal [37]. While several methods introduce additional structure by decomposing tasks into steps [8, 18], some open challenges remain (Section 1). First, task-specific designs may limit generalization across diverse activities [8, 18]. Second, many approaches adopt nearly uniform reward allocations, which may underweight the salience of critical sub-steps [37, 59]. In addition, current PRMs typically rely on single-view observations [2, 8, 36, 37, 59], which can impede multi-perspective state estimation and increase sensitivity to occlusions. In contrast, our method, Dopamine-Reward, aims to address these issues by learning a general-purpose, step-aware reward model that explicitly fuses multi-view inputs, enabling a more robust and finegrained reward estimation.

### 3. Method

Our approach is designed to address the core challenges in real-world robotic learning by introducing two synergistic components. First, we develop Dopamine-Reward that learns a general-purpose, step-aware process reward from multi-view inputs (Section 3.1). Second, we propose Dopamine-RL, a robust policy learning framework built upon Dopamine-Reward, resolving the theoretical flaws in conventional reward shaping (Section 3.2).

#### 3.1. Dopamine-Reward Modeling Method

###### 3.1.1. General Reward Model (GRM) Construction

The core of our modeling method is to build the GRM, a vision-language model designed to estimate precise task progress. To ensure the model generalizes across diverse embodiments and tasks, we construct a large-scale dataset structured around relative temporal transitions. This section details the three-stage GRM training data construction pipeline, from raw video segmentation to a scientifically rigorous hop-based labeling strategy as follows:

Step-wise task progress discretization. We treat task progress itself as the supervision signal. Given raw multiview video trajectories, we first segment each expert trajectory into sub-tasks using human-annotated multi-view keyframes {K0,K1,...,KN}, where K0 is the initial observation, KN is the final success observation, and each Kj is a set of synchronized multi-view keyframes. To obtain dense supervision, we perform adaptive sampling within each segment. For a trajectory with L frames per view, we set a chunk size C to determine the total number of sampled points and distribute them uniformly across the N segments. The number of intermediate points m within segment [Kj,Kj+1] is:

m =

1 N

L C

. (1)

[Figure 37]

One-Shot GRM Adaption

General Reward Model (GRM) Construction

[Figure 38]

Task- Prompt

[Figure 39]

[Figure 40]

You are a rigorous, impartial vision evaluator for robot task progress. … The task instruction is “Fold the plans”. Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing …

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

###### GRM

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

STATE_GOAL

BEFORE AFTER

STATE_INIT

###### Policy-Invariant Reward Shaping

Vision Encoder Tokenizer

return

[Figure 56]

return

LLM Decoder

[Figure 57]

[Figure 58]

time

time

Multi-Perspective Progress Fusion ProgressHoporRegressHop

###### Universal RL-Algorithm Compatibility

PPO

Online/Offline RL Offline-to-Online RL

Value-Based RL Gradient-Based RL

Fused Reward Prediction

Incremental

Prev. State As “BEFORE”

ReinFlow

Cal-QL ···

|∆𝒕𝟏|
|---|

-0.07

Forward-Anchored

INIT State As “BEFORE”

+0.13

[Figure 59]

[Figure 60]

|∆𝒕𝟐|
|---|

Buffer Policy

GOAL State As “BEFORE” Backward-Anchored

Good Bad

[Figure 61]

(a) Dopamine-Reward Modeling Method (b) Dopamine-RL Training Framework

- Figure 2. The overview of our method. Our framework is composed of two core components: (a) Dopamine-Reward Modeling Method and (b) Dopamine-RL Training Framework. (a) At the heart of our reward modeling is to build the General Reward Model (GRM), a vision-language model that is prompted with a task description and conditioned on multi-view images of initial, goal, “before,” and “after” states to predict a relative progress or regress hop. To ensure a stable and accurate signal, we employ Multi-Perspective Progress Fusion, which combines incremental, forward-anchored, and backward-anchored predictions into a final fused reward. (b) The Dopamine-RL framework first adapts the pre-trained GRM to a novel task using a single demonstration (One-Shot GRM Adaptation). Subsequently, it uses a theoretically-sound Policy-Invariant Reward Shaping method to convert the GRM’s dense output into a reward signal that accelerates learning without altering the optimal policy. This approach is universally compatible with a wide range of RL algorithms.

ready covered from the initial state. A key theoretical advantage is that, when global progress is reconstructed by iteratively applying predicted hops, the resulting Φ⋆(s) is guaranteed to remain strictly within [0,1]. A detailed proof is provided in Appendix A.1.

This yields a sequence of states S = {s0,s1,...,sM}, where each state si is a set of synchronous multi-view visual observations. We then define the ground-truth global progress as Φ(si) = i/M.

Hop-based relative progress normalization. A naive choice is to regress the progress gain Φδ(sp,sq) = Φ(sq)− Φ(sp) between two states, but iterating such predictions accumulates error and can push the reconstructed Φ⋆(s) outside [0,1]. Instead, we introduce a hop-based formulation that learns relative-relative progress. Each training sample is a tuple D containing a task description dtask, the initial state s0, the goal state sM, a “BEFORE” state sp, an “AFTER” state sq, and a hop label H(sp,sq) that normalizes the progress from sp to sq relative to the full task span from s0 to sM. Given Φ(sp) and Φ(sq), we define:

Sampling strategy and data balancing. For each trajectory, we construct a balanced set of hop-based training samples. Continuous hop values are first discretized into Nhop hop bins. The temporal distance between the “BEFORE” state sp and “AFTER” state sq in each pair is then chosen from Ndis distance bins within each hop bin, yielding in total Nhop × Ndis non-trivial transitions. To reduce bias toward static segments, we further introduce an additional fraction α of samples explicitly labeled as zero-hop (i.e., H(sp,sq) = 0), constructed by selecting pairs (sp,sq) whose progress change is below a small threshold ϵ:

 

Φ(sq) − Φ(sp) Φ(sM) − Φ(sp)

if q ≥ p (PROGRESS)

|Φ(sq) − Φ(sp)| ≤ ϵ. (3)

H(sp,sq) =

Applying this three-stage pipeline yields a dataset of 35M samples from about 3,400 hours of video and over 100K trajectories (see Appendix B). We train the GRM on this corpus to estimate hop-based relative progress between arbitrary state pairs, conditioned on the initial state, goal state, and task description.

###### Φ(sq) − Φ(sp) Φ(sp) − Φ(s0)



if q < p (REGRESS).

(2) This dynamically scales the supervision into [−1,1]: for forward progress, the change is normalized by the remaining distance to the goal; for regression, by the distance al-

###### 3.1.2. Multi-Perspective Progress Fusion from GRM

To mitigate error accumulation and ensure consistent accuracy, we fuse predictions based on GRM from three complementary perspectives: incremental prediction, forwardanchored prediction, and backward-anchored prediction.

Incremental Prediction first offers a fine-grained, stepby-step assessment. Refer to Equation (2), the predicted global progress Φ⋆I(st) is recursively computed from the preceding state’s progress Φ⋆(st−1) and the predicted hop H⋆(st−1,st). Let ∆Φ⋆t−1,t be the estimated progress hop:

[1 − Φ⋆(st−1)] · H⋆ if H⋆ ≥ 0 Φ⋆(st−1) · H⋆ if H⋆ < 0.

∆Φ⋆t−1,t =

(4)

The incremental progress is then calculated as follow:

Φ⋆I(st) = Φ⋆(st−1) + ∆Φ⋆t−1,t, (5)

where Φ⋆I(st) is be accumulated along the trajectory, initialized with Φ⋆(s0) = 0. While this method excels at capturing local dynamics, it is susceptible to the accumulation of prediction errors over long trajectories. To counteract this drift, we introduce extra two global perspectives. ForwardAnchored Prediction provides a stable global reference by anchoring to the initial state sinit, where progress is zero:

Φ⋆F(st) = H⋆(sinit,st). (6) Conversely, Backward-Anchored Prediction is anchored to the goal state sgoal, where progress is one. This approach offers high sensitivity near task completion:

Φ⋆B(st) = 1 + H⋆(sgoal,st). (7)

These three methods offer complementary strengths: local precision (incremental), initial stability (forward), and goal sensitivity (backward). We fuse them via averaging to obtain a robust final progress estimate:

1 3

Φ⋆(st) =

(Φ⋆I(st) + Φ⋆F(st) + Φ⋆B(st)). (8)

This fusion yields a more accurate and drift-resistant signal, which is critical for the subsequent reward shaping.

###### 3.1.3. Progress Consistency Checking (Optional)

While the multi-perspective fusion via averaging (Equation (8)) serves as a baseline, its naive application in online RL faces the risk of Out-of-Distribution (OOD) hallucination. Due to the inherent limitations of data coverage, it is impossible for the training set to encompass every corner of the state space. During RL, the policy inevitably explores unseen regions where the reward model may yield spurious high signals, leading to “reward hacking.” To address these, we propose a bi-directional consistency checking strategy that leverages consistency as a proxy for reliability, which is motivated by the observation that forward

Φ∗F and backward Φ∗B predictions tend to exhibit significant divergence in OOD scenarios or observations, whereas they remain consistent in familiar states.

###### Consistency-Aware Weighting. We first define the mean

estimated progress Φ¯∗(st) = (Φ∗F(st) + Φ∗B(st))/2. To quantify uncertainty, we calculate a normalized discrepancy metric:

∆norm(st) = |Φ∗B(st) − Φ∗F(st)| Φ¯∗(st) + ϵ

, (9)

where ϵ is a small constant for numerical stability. Normalization by Φ¯∗ ensures that discrepancies are penalized more heavily during the early stages (where Φ is small), as precise guidance is critical initially. We then derive a confidence weight wt ∈ (0,1] using a Gaussian kernel with sensitivity α:

wt = exp −α · (∆norm(st))2 . (10)

Conservative State Update. To prevent the policy from exploiting erroneous estimates in OOD scenarios, we employ a conservative update rule for the maintained progress state Φ∗(st) instead of Equation (8):

wt 2 · Φ ¯∗(st) − Φ∗(st−1) + ∆Φ⋆t−1,t .

Φ∗(st) = Φ∗(st−1)+

(11) This mechanism acts as a semantic filter: it ignores uncertain updates when wt → 0 (retaining Φ∗(st−1)) and fully trusts the estimate when consistency is high (wt → 1).

#### 3.2. Dopamine-RL Framework

Building upon Dopamine-Reward with GRM, we further introduce the Dopamine-RL framework, a reinforcement learning pipeline producing high-performance policy stimulated by Dopamine-Reward, featuring three key critical attributes: minimal downstream task effort for rapid progress alignment (Section 3.2.1), fast convergence with policyinvariant guarantees (Section 3.2.2) and seamless integration with diverse RL paradigms (Section 3.2.3).

###### 3.2.1. One-shot GRM Adaptation

Dopamine-RL requires only one single human demonstration Dhuman to adapt the pre-trained GRM to novel or highprecision tasks, since the pre-trained GRM has already possessed a broad prior for assessing progress. Given a new task, we minimize the Mean Squared Error (MSE) between its predicted hop value, Hω⋆, and the ground-truth, Hgt:

p,sq)∼Dhuman∥Hω⋆ − Hgt∥22, (12)

LGRM(ω) = E(s

where ω represents the GRM’s parameters, initialized by pre-trained GRMω

. After SFT, we obtain a task-adapted GRMω

0

, poised for efficient reinforcement learning.

⋆

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Start Insert incorrectly Position too low Misaligned Success

1.0

0.8

0.6

0.4 0.2

Reference Reward

VLAC (Baseline)

GRM (Ours)

0

0 20 40 60 80 100 120 140 160 180 200 220 Step #

- Figure 3. Reward profiles on a challenging real-world rollout. We plot the reference reward from human annotations, the VLAC baseline, and our GRM along the same trajectory. Our GRM tracks the reference signal more faithfully, sharply penalizing incorrect insertions, low positions, and misalignments, and only assigning high reward near successful task completion.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Real-World Tasks Hardware System

Insert Square Complete Circuit Fold Towel Cap the Pen

Pick and Place Arrange Flowers Build Blocks Zip the Bag

[Figure 78]

[Figure 79]

- Figure 4. Real-world tasks and hardware setup. Left: eight representative long-horizon manipulation tasks used to evaluate DopamineReward and Dopamine-RL, including insertion, circuit completion, folding, pick-and-place, and assembly tasks. Right: our multi-view hardware platform with the Pika teleoperation system and calibrated ZED cameras, providing synchronized wrist and third-person observations for GRM training and policy learning.

###### 3.2.2. Policy-Invariant Reward Shaping

- • Discount consistency: rGRM must be compatible with the standard exponentially discounted return and TD or Bellman updates with factor γ under a memoryless (Markov) reward assumption (see Appendix A.3).
- • Locality. At any step t, rGRM is efficiently computable from the single transition (st,at,st+1).

A straightforward approach to defining the dense process reward function for policy learning is to use the direct increment of this progress: r(st,at,st+1) = Φ⋆(st+1) − Φ⋆(st). However, optimizing the standard discounted return, J(π) = Eπ[ ∞t=0 γtr(st,at,st+1)], with this reward is mathematically equivalent to maximizing a different objective: J′(π) ∝ Eπ[ ∞t=1 γt−1Φ⋆(st) | s0], as detailed in Appendix A.2. This transformed objective creates a perverse incentive: it encourages the agent not to complete the task, but rather to seek and maintain states with high progress values. Consequently, the resulting policy is rewarded for stagnation, preferring a safe, suboptimal state over potentially risky trajectories that lead to true task completion. To resolve the misalignment, we formulate our GRM reward rGRM that adheres to three desiderata:

Adherence to these desiderata uniquely determines the reward structure, we derive the reward from the continuoustime “discounted potential” e−λtΦ⋆(st). As detailed in Appendix A.4, the natural discrete-time, single-step increment that is consistent with this continuous form is:

F(st,st+1) = γΦ⋆(st+1) − Φ⋆(st), (13)

where γ = e−λh. To enable autonomous learning on real robots without the need for continuous human monitoring, we automate the determination of the sparse outcome reward rgold. Specifically, we consider the task completed when the estimated progress falls within a close margin of the target (i.e., Φ⋆(st+1) ≥ 1 − δ, with δ = 0.05). Thus, rgold = 1 if the completion threshold is met, and 0 other-

• Optimal policy invariance. The optimal policy learned with rGRM must coincide with that under the sparse gold reward rgold (1 at task completion, 0 otherwise), so shaping guides exploration without changing task objective.

wise. We add the shaping term F to this automated goldstandard reward to define our final reward function:

rGRM(st,at,st+1) = rgold + γΦ⋆(st+1) − Φ⋆(st). (14)

This form guarantees policy invariance: the cumulative discounted shaping term F forms a telescoping sum that collapses to a constant boundary term depending only on the initial state s0. Appendix A.5 shows that the discrete-time sum and the continuous-time integral of the discounted potential’s derivative converge to the same constant. :

∞

###### = −Φ⋆(s0)

γt(γΦ⋆(st+1) − Φ⋆(st))

t=0

Boundary Term

Discrete PBRS Sum



∆t→0 ∞

d dt

e−λtΦ⋆(st) dt

###### = −Φ⋆(s0)

.

0

Boundary Term

Continuous Integral

(15)

Since the shaping term telescopes to a state-dependent constant that is independent of the subsequent policy π, the shaped Q-function is simply a state-wise shift of the original one:

QπGRM(s,a) = Qπgold(s,a) − Φ⋆(s). (16)

The shift −Φ⋆(s) is identical for all actions a in a given state s, so the optimal action remains unchanged:

Q∗GRM(s,a) = arg max

Q∗gold(s,a). (17)

arg max

a

a

This matches the standard Potential-Based Reward Shaping (PBRS) framework [41], with the GRM progress Φ⋆ serving as the potential function.

###### 3.2.3. Universal RL-Algorithm Compatibility

Dopamine-RL exhibits strong universality, seamlessly integrating with any RL algorithm, encompassing online RL, offline RL, and offline-to-online RL paradigms. It adapts effectively to both value-based methods and gradient-based approaches. By reshaping targeted reward functions to guide agent learning, Dopamine-RL is inherently agnostic to the specific RL algorithm employed. Experimental results confirm this flexibility. In simulations, we deploy under two settings: PPO[46] (Proximal Policy Optimization) algorithm and OpenVLA-OFT[26] model, and ReinFlow[61] algorithm with π0[6] model. exhibits excellent performance under both settings. In real-world settings, we combine with Cal-QL[39] (a offline-to-online Qlearning based RL algorithm) and it also delivers exceptional outcomes. Further details are shown in Appendix C.

### 4. Experiments

We evaluate Dopamine-Reward with GRM and DopamineRL on both simulation and real-world robotic platforms, covering a broad range of manipulation skills and deployment scenarios. This section summarizes our empirical findings and is organized around four questions:

- • RQ1: How accurate is the GRM at perceiving task progress compared to VLMs and existing reward models?
- • RQ2: How does Dopamine-RL perform in success rate, sample efficiency, and generalization against strong BC and RL baselines?
- • RQ3: How critical is Multi-Perspective Progress Fusion for final performance?
- • RQ4: How important is the Dopamine-RL framework for turning reward modeling into practical policy learning?

#### 4.1. Accurate Task Progress Perception (RQ1)

In this part, We assess GRM’s ability to estimate task progress using two complementary protocols: video frame rank-correlation and task completion judgment.

###### 4.1.1. Video Frame Rank-Correlation

To quantitatively assess task progress perception, we follow the evaluation methodology of GVL [37] and measure the Value-Order Correlation (VOC) between the GRM’s predicted progress and the ground-truth chronological order of shuffled frames. A higher VOC score ([-1, 1]) indicates a better understanding of temporal progress. We evaluate on a diverse suite of eight datasets spanning real-world robotics (DROID [24], AGIBOT-World [7], RoboBrain-X [17]), simulation (Libero [30], RoboCasa [40], RoboTwin2.0 [9]), and egocentric human manipulation (EgoDex [19]). To test robustness to temporal granularity, we test under three distinct sampling strategies: Sparse (S) using only major keyframes, Medium (M) using uniform samples between keyframes, and Dense (D) using uniform samples across the entire trajectory. We compare our multi-view and single-view GRM against four state-of-the-art reward models: GVL [37], and VLAC [59]. As shown in Table 1, our multi-view GRM consistently achieves the highest VOC scores across all seven datasets and all sampling strategies. The performance of baseline models tends to degrade as sampling becomes denser, indicating a struggle with finegrained temporal distinctions. In contrast, our model maintains exceptionally high performance, highlighting the robustness of our hop-based learning formulation and multiperspective fusion. The performance gap is most significant in complex, long-horizon tasks (e.g., LIBERO [30], RoboBrain-X [17]), where our model’s ability to accurately contextualize progress is paramount.

- Table 1. Video Frame Rank-Correlation (VOC) on Diverse Datasets. We evaluate reward models under three temporal sampling strategies: Sparse (S), Medium (M), and Dense (D). Our GRM variants (Ours-3B and Ours-8B) consistently outperform prior work. Notably, the Ours-8B (Multi-View) model sets a new state-of-the-art across all benchmarks and sampling densities, showcasing exceptional robustness and progress understanding.

Dataset

GVL [37] VLAC-2B [59] Ours-3B (Single-View) Ours-3B (Multi-View) Ours-8B (Single-View) Ours-8B (Multi-View) S M D S M D S M D S M D S M D S M D

Real.

DROID [24] 0.01 -0.30 0.07 0.66 0.69 0.50 0.96 0.95 0.94 0.99 0.98 0.97 0.97 0.96 0.95 0.99 0.99 0.98 AGIBOT-World [7] 0.24 0.17 0.12 0.29 0.40 0.41 0.90 0.89 0.87 0.97 0.96 0.95 0.92 0.91 0.89 0.97 0.97 0.96 RoboBrain-X [17] 0.32 0.38 0.33 0.18 0.13 0.17 0.85 0.83 0.81 0.91 0.89 0.88 0.87 0.87 0.83 0.92 0.91 0.89

Sim.

LIBERO [30] 0.43 0.37 0.38 0.19 0.28 0.41 0.90 0.86 0.85 0.95 0.91 0.92 0.90 0.88 0.86 0.94 0.93 0.92 RoboCasa [40] 0.06 -0.04 0.02 0.00 0.11 0.32 0.95 0.94 0.93 0.98 0.97 0.96 0.97 0.96 0.95 0.99 0.98 0.97 RoboTwin2.0 [9] 0.28 0.19 0.10 0.26 0.35 0.32 0.90 0.89 0.87 0.95 0.94 0.93 0.92 0.91 0.89 0.96 0.95 0.94

Hum. EgoDex [19] 0.05 0.04 -0.10 0.09 0.10 0.18 0.88 0.85 0.81 0.88 0.85 0.81 0.88 0.86 0.83 0.88 0.86 0.83 Average 0.20 0.12 0.13 0.24 0.29 0.33 0.91 0.89 0.87 0.96 0.94 0.93 0.92 0.91 0.89 0.96 0.96 0.94

- Table 2. Task Completion Classification Accuracy (as successes out of 60). Our GRM more accurately classifies the final outcomes of robot rollouts compared to both specialized reward models and large generalist models.

(e.g., hand covering the block during stacking), leading to noisy progress curves that fail the stability check in Equation (58). (3) Impact of Multi-View Fusion: The gap between our Single-View (83.9%) and Multi-View (92.8%) variants highlights the critical role of perceptual robustness. The multi-view fusion ensures that if one view is occluded, the model can still verify progress via alternative angles, correctly distinguishing between a completed task (SE) and a stalled one (PSE/FE).

Method Stacking Folding Clearing Average

###### Gemini-2.5-Pro [14] 50/60 45/60 51/60 81.1% GPT-5 [42] 51/60 48/60 52/60 83.9% Qwen3-VL [45] 43/60 41/60 43/60 76.7% RoboBrain 2.0 [51] 38/60 35/60 41/60 61.7%

VLMs

GVL [37] 25/60 27/60 15/60 37.2% VLAC-2B [59] 19/60 21/60 21/60 33.9%

RMs

Table 3. Policy Performance and Sample Efficiency. DopamineRL achieves significantly higher performance with fewer human demonstrations. Sample efficiency is measured by episodes needed to reach 80% of the final success rate (lower is better).

Ours-8B (Single-View) 50/60 50/60 51/60 83.9% Ours-8B (Multi-View) 56/60 54/60 57/60 92.8%

###### 4.1.2. Task Completion Judgment

To assess the GRM’s ability to make high-level judgments about task outcomes, we follow the protocol from SARM [8]. We collect 60 real-world rollouts for each of three tasks (stacking blocks, folding T-shirt, clearing desktop), with 20 successful (SE), 20 partially successful (PSE), and 20 failed (FE) episodes. We evaluate classification accuracy against reward model baselines (GVL [37], VLAC [59]) and generalist vision-language models (Gemini-2.5-Pro [14], GPT-5 [42], Qwen3-VL8B [45], RoboBrain 2.0-8B [22, 51]). More evaluation settings are shown in Appendix C.1. Results in Table 2 shows: (1) Superiority over Generalist VLMs: While large models like GPT-5 [42] and Gemini-2.5-Pro [14] achieve respectable accuracy (∼83%), they often struggle with spatial precision—misclassifying “near-misses” (PSE) as successes. Our GRM-8B (Multi-View) significantly outperforms them (+9% vs GPT-5), demonstrating that domainspecific training on progress data is more effective than zero-shot reasoning. (2) Failure of Single-View PRMs: Existing reward models like GVL [37] and VLAC [59] perform poorly (accuracy < 40%). This is primarily because single-view models lose track of objects during occlusion

Simulation (10 Tasks) Real-World (8 Tasks) SR (%) Rollout (#) SR (%) Rollout (#)

Method

BC (50 demos) 31.5 – 9.8 – RL + Sparse 79.9 560 68.0 183 Dopamine-RL 81.0 395 95.2 150

#### 4.2. Performance, Efficiency, Generalization (RQ2)

We now evaluate the Dopamine-RL framework across 10 simulation tasks (from LIBERO [30] and RoboTwin2.0 [9]) and 8 real-world tasks, whose task setups and hardware platform are illustrated in Figure 4. In our simulation experiments, Dopamine-RL is evaluated under two distinct configurations: one leveraging the PPO [46] (Proximal Policy Optimization) algorithm alongside the OpenVLA-OFT [26] model, and the other integrating the ReinFlow [61] algorithm with the π0 [6] model. For real-world implementations, we pair Dopamine-RL with Cal-QL [39] and we employ a Human-in-the-Loop setup where we use just one single human demonstrations to adapt the GRM. We compare against strong baselines: Behavioral Cloning (BC) on 50 demos, and Proximal Policy Optimization [46] (PPO) using a sparse reward in simulation and ConRFT [12] for real-world settings.

Table 4. Generalization Performance Breakdown: ID vs. OOD. The table compares success counts (out of 20 trials) for Behavioral Cloning (BC) and our framework (Ours). The final row, Avg. Relative Drop (∆), quantifies the average relative success rate drop from ID performance when tested on OOD settings.

Insert Square Circuit Cap Pen BC Ours BC Ours BC Ours

Condition

Original (ID) ↑ 7/20 19/20 5/20 20/20 8/20 19/20

Object ↑ 4/20 15/20 3/20 17/20 5/20 17/20 Layout ↑ 2/20 15/20 1/20 19/20 3/20 15/20 Background ↑ 3/20 16/20 2/20 19/20 4/20 16/20

OOD

Avg. Drop (∆ %) ↓ 57.1 19.3 60.0 8.3 50.0 15.8

As shown in Table 3, Dopamine-RL significantly outperforms all baselines in both final success rate and sample efficiency. The dense and accurate rewards from our GRM enables rapid and stable learning, achieving high performance with far fewer environment interactions. To test generalization, we evaluate the final policies under In-Distribution (ID) and Out-of-Distribution (OOD) conditions, where OOD settings include changes to object properties (Object Change), workspace layout (Layout Change), and background visuals (Background Change). Table 4 presents a detailed breakdown of this analysis. The results show that while both methods experience a performance drop when faced with distribution shifts, the Average Relative Drop (∆) is significantly more pronounced for the BC baseline (50-60% degradation). In contrast, our DopamineRL framework maintains a much higher proportion of its original performance, with a relative drop of only 8-20%. This quantitatively demonstrates that our policy has learned a more robust and generalizable understanding of the task semantics, successfully mitigating the overfitting to superficial visual features that severely impacts the BC baseline.

#### 4.3. Ablation Studies (RQ3 & RQ4)

Finally, we conduct a series of ablation studies on a representative subset of three real-world tasks to validate the key design choices in the Dopamine-Reward framework. The results in Table 5 confirm our hypotheses.

- For RQ3, we ablate the Multi-Perspective Progress Fu-

sion in Dopamine-Reward. Removing fusion and relying on a single progress estimator consistently hurts performance: the incremental-only, forward-anchored-only, and backward-anchored-only variants incur 15.0%, 19.3%, and 22.5% absolute drops, respectively. The incremental-only variant is particularly vulnerable to error drift over long horizons, confirming the importance of combining local and global progress perspectives.

- For RQ4, the importance of the Dopamine-RL is evident.

Removing policy-invariant reward shaping leads to a massive performance drop of 43.7%. The agent learns to reach

Table 5. Ablation Study Results (Average Success Rate %). Each component of the Dopamine-Reward framework is shown to be critical for achieving maximum performance.

Method Variation Success Rate ∆ from Full Full Framework (Dopamine-RL) 85.0 –

- Ablations for RQ3 (Core Components) w/o Fusion (Incremental Only) 70.0 -15.0 w/o Fusion (Forward-Anchored Only) 65.7 -19.3 w/o Fusion (Backward-Anchored Only) 62.5 -22.5

- Ablations for RQ4 (Dopamine-RL Framework) w/o Policy-Invariant Shaping 41.3 -43.7 w/o One-shot adaption 63.2 -21.8

“good-enough” states and stagnates, failing to complete the tasks, which confirms the “semantic trap” discussed in Section 3. Besides, relying solely on zero-shot GRM, it occasionally provides incorrect rewards for corner cases in outof-distribution (OOD) tasks, such as assigning positive rewards to poor actions and negative rewards to good ones. This hinders the convergence of the policy, resulting in a 21.8% drop in success rate.

- 5. Conclusion

In this work, as named Robo-Dopamine, we tackled the critical challenges of reward design in real-world robotics by introducing Dopamine-Reward, a novel approach for learning a general-purpose, step-aware reward model from multiview inputs. Our core contribution, the General Reward Model (GRM), is trained on over 3,400 hours of diverse data processed via Dopamine-Reward and leverages MultiPerspective Progress Fusion to overcome perceptual limitations like occlusion. Building upon this, our DopamineRL framework employs a theoretically-grounded, PolicyInvariant Reward Shaping method, which provides dense guidance to accelerate learning without altering the optimal policy, thereby systematically avoiding the common “semantic trap”. Extensive experiments on diverse tasks validate our approach, demonstrating state-of-the-art reward accuracy and remarkable sample efficiency, with policies improving success rates from nearly-zero to ∼95% in an average of only ∼150 interaction rollouts while exhibiting strong generalization. By combining a robust multi-view reward model with a principled RL framework, our work presents a scalable recipe for enabling embodied agents to achieve continuous self-improvement and master complex manipulation tasks far beyond their initial demonstrations. In the future, we plan to expand our work in four potential directions, detailed in Appendix. E.

### References

- [1] Abbas Abdolmaleki, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Ashwin Balakrishna, Nathan Batchelor, Alex Bewley, Jeff Bingham, Michael Bloesch, et al. Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342, 2025. 2
- [2] Minttu Alakuijala, Reginald McLean, Isaac Woungang, Nariman Farsad, Samuel Kaski, Pekka Marttinen, and Kai Yuan. Video-language critic: Transferable reward functions for language-conditioned robotics. arXiv preprint arXiv:2405.19988, 2024. 2, 3
- [3] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025. 23
- [4] Shuanghao Bai, Wenxuan Song, Jiayi Chen, Yuheng Ji, Zhide Zhong, Jin Yang, Han Zhao, Wanqi Zhou, Wei Zhao, Zhe Li, et al. Towards a unified understanding of robot manipulation: A comprehensive survey. arXiv preprint arXiv:2510.10903, 2025. 2
- [5] Johan Bjorck, Fernando Casta˜neda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 2
- [6] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint ARXIV.2410.24164. 7, 8, 20
- [7] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025. 2, 7, 8, 16, 18
- [8] Qianzhong Chen, Justin Yu, Mac Schwager, Pieter Abbeel, Fred Shentu, and Philipp Wu. Sarm: Stage-aware reward modeling for long horizon robot manipulation. arXiv preprint arXiv:2509.25358, 2025. 2, 3, 8, 20
- [9] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025. 7, 8, 17, 18
- [10] Yuhui Chen, Haoran Li, and Dongbin Zhao. Boosting continuous control with consistency policy. arXiv preprint arXiv:2310.06343, 2023. 21
- [11] Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450, 2025. 2, 3

- [12] Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450, 2025. 8, 21
- [13] Zengjue Chen, Runliang Niu, He Kong, Qi Wang, Qianli Xing, and Zipei Fan. Tgrpo: Fine-tuning vision-languageaction model via trajectory-wise group relative policy optimization. arXiv preprint arXiv:2506.08440, 2025. 3
- [14] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 8, 18
- [15] Figure AI. Helix: A vision-language-action model for generalist humanoid control. https://www.figure.ai/ news/helix, 2025. Accessed: 2025-04-18. 2
- [16] Chelsea Finn, Sergey Levine, and Pieter Abbeel. Guided cost learning: Deep inverse optimal control via policy optimization. In International conference on machine learning, pages 49–58. PMLR, 2016. 2, 3
- [17] FlagOpen. Robobrain-x0. https://github.com/ FlagOpen/RoboBrain-X0, 2025. GitHub repository, accessed 2025-11-08. 7, 8, 16, 18
- [18] Seyed Kamyar Seyed Ghasemipour, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, and Igor Mordatch. Self-improving embodied foundation models. arXiv preprint arXiv:2509.15155, 2025. 2, 3
- [19] Ryan Hoque, Peide Huang, David J Yoon, Mouli Sivapurapu, and Jian Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025. 7, 8, 17, 18
- [20] Zheyuan Hu, Aaron Rovinsky, Jianlan Luo, Vikash Kumar, Abhishek Gupta, and Sergey Levine. Reboot: Reuse data for bootstrapping efficient real-world dexterous manipulation. arXiv preprint arXiv:2309.03322, 2023. 3
- [21] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025. 2
- [22] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1724–1734, 2025. 8
- [23] Zhenyu Jiang, Yuqi Xie, Kevin Lin, Zhenjia Xu, Weikang Wan, Ajay Mandlekar, Linxi Jim Fan, and Yuke Zhu. Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 16923–16930. IEEE, 2025. 2
- [24] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale

- in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. 2, 7, 8, 16, 18
- [25] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025. 2
- [26] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025. 7, 8, 20
- [27] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 19
- [28] Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, and Huazhe Xu. Rl-100: Performant robotic manipulation with real-world reinforcement learning. arXiv preprint arXiv:2510.14830, 2025. 3
- [29] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, et al. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint arXiv:2509.09674,

2025. 2, 3

- [30] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023. 7, 8, 17, 18, 20
- [31] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl, 2025. URL https://arxiv. org/abs/2505.05470. 3
- [32] Jijia Liu, Feng Gao, Bingwen Wei, Xinlei Chen, Qingmin Liao, Yi Wu, Chao Yu, and Yu Wang. What can rl bring to vla generalization? an empirical study. arXiv preprint arXiv:2505.19789, 2025. 3
- [33] Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. arXiv preprint arXiv:2505.18719, 2025. 2, 3
- [34] Jianlan Luo, Charles Xu, Xinyang Geng, Gilbert Feng, Kuan Fang, Liam Tan, Stefan Schaal, and Sergey Levine. Multistage cable routing through hierarchical imitation learning. IEEE Transactions on Robotics, 40:1476–1491, 2024. 3
- [35] Jianlan Luo, Charles Xu, Jeffrey Wu, and Sergey Levine. Precise and dexterous robotic manipulation via human-inthe-loop reinforcement learning. Science Robotics, 10(105): eads5033, 2025. 2, 3
- [36] Yecheng Jason Ma, Vikash Kumar, Amy Zhang, Osbert Bastani, and Dinesh Jayaraman. Liv: Language-image representations and rewards for robotic control. In International Conference on Machine Learning, pages 23301–23320. PMLR,

2023. 2, 3

- [37] Yecheng Jason Ma, Joey Hejna, Chuyuan Fu, Dhruv Shah, Jacky Liang, Zhuo Xu, Sean Kirmani, Peng Xu, Danny

- Driess, Ted Xiao, et al. Vision language models are incontext value learners. In The Thirteenth International Conference on Learning Representations, 2024. 2, 3, 7, 8, 19
- [38] Max Sobol Mark, Tian Gao, Georgia Gabriela Sampaio, Mohan Kumar Srirama, Archit Sharma, Chelsea Finn, and Aviral Kumar. Policy agnostic rl: Offline rl and online rl fine-tuning of any class and backbone. arXiv preprint arXiv:2412.06685, 2024. 3
- [39] Mitsuhiko Nakamoto, Simon Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, and Sergey Levine. Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. Advances in Neural Information Processing Systems, 36:62244–62269, 2023. 7, 8, 21
- [40] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024. 7, 8, 17, 18
- [41] Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Icml, pages 278–287. Citeseer,

1999. 2, 7

- [42] OpenAI. Gpt-5 system card. System card, OpenAI, 2025. Canonical PDF version dated August 13, 2025. 8
- [43] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024. 2
- [44] Gang Peng, Jin Yang, Xinde Li, and Mohammad Omar Khyam. Deep reinforcement learning with a stage incentive mechanism of dense reward for robotic trajectory planning. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 53(6):3566–3573, 2022. 2, 3
- [45] QwenLM. Qwen3-vl. https://github.com/ QwenLM/Qwen3-VL, 2025. GitHub repository, accessed 2025-11-09. 8
- [46] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 7, 8, 20
- [47] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatronlm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019. 19
- [48] Huajie Tan, Cheng Chi, Xiansheng Chen, Yuheng Ji, Zhongxia Zhao, Xiaoshuai Hao, Yaoxu Lyu, Mingyu Cao, Junkai Zhao, Huaihai Lyu, et al. Roboos-next: A unified memory-based framework for lifelong, scalable, and robust multi-robot collaboration. arXiv preprint arXiv:2510.26536,

2025. 16

- [49] Huajie Tan, Xiaoshuai Hao, Cheng Chi, Minglan Lin, Yaoxu Lyu, Mingyu Cao, Dong Liang, Zhuo Chen, Mengsi Lyu, Cheng Peng, et al. Roboos: A hierarchical embodied framework for cross-embodiment and multi-agent collaboration. arXiv preprint arXiv:2505.03673, 2025. 16

- [50] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Xiansheng Chen, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning of vision language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 3
- [51] BAAI RoboBrain Team, Mingyu Cao, Huajie Tan, Yuheng Ji, Xiansheng Chen, Minglan Lin, Zhiyu Li, Zhou Cao, Pengwei Wang, Enshen Zhou, et al. Robobrain 2.0 technical report. arXiv preprint arXiv:2507.02029, 2025. 8, 19
- [52] Generalist AI Team. Gen-0: Embodied foundation models that scale with physical interaction. Generalist AI Blog,

2025. https://generalistai.com/blog/preview-uqlxvb-bb.html. 2

- [53] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 21
- [54] Bart van Marum, Aayam Shrestha, Helei Duan, Pranay Dugar, Jeremy Dao, and Alan Fern. Revisiting reward design and evaluation for robust humanoid standing and walking. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 11256–11263. IEEE,

2024. 2, 3

- [55] Zheng Wu, Wenzhao Lian, Vaibhav Unhelkar, Masayoshi Tomizuka, and Stefan Schaal. Learning dense rewards for contact-rich manipulation tasks. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 6214–6221. IEEE, 2021. 2, 3
- [56] Charles Xu, Qiyang Li, Jianlan Luo, and Sergey Levine. Rldg: Robotic generalist policy distillation via reinforcement learning. arXiv preprint arXiv:2412.09858, 2024. 2, 3
- [57] Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. Streamingvlm: Real-time understanding for infinite video streams. arXiv preprint arXiv:2510.09608, 2025. 23
- [58] Hongzhi Zang, Mingjie Wei, Si Xu, Yongji Wu, Zhen Guo, Yuanqing Wang, Hao Lin, Liangzhi Shi, Yuqing Xie, Zhexuan Xu, et al. Rlinf-vla: A unified and efficient framework for vla+ rl training. arXiv preprint arXiv:2510.06710, 2025. 2, 3, 20
- [59] Shaopeng Zhai, Qi Zhang, Tianyi Zhang, Fuxian Huang, Haoran Zhang, Ming Zhou, Shengzhe Zhang, Litao Liu, Sixu Lin, and Jiangmiao Pang. A vision-language-action-critic model for robotic real-world reinforcement learning. arXiv preprint arXiv:2509.15937, 2025. 2, 3, 7, 8, 19
- [60] Hongyin Zhang, Zifeng Zhuang, Han Zhao, Pengxiang Ding, Hongchao Lu, and Donglin Wang. Reinbot: Amplifying robot visual-language manipulation with reinforcement learning. arXiv preprint arXiv:2505.07395, 2025. 2, 3
- [61] Tonghe Zhang, Chao Yu, Sichang Su, and Yu Wang. Reinflow: Fine-tuning flow matching policy with online reinforcement learning. arXiv preprint arXiv:2505.22094, 2025. 7, 8, 20
- [62] Tonghe Zhang, Chao Yu, Sichang Su, and Yu Wang. Reinflow: Fine-tuning flow matching policy with online reinforcement learning. arXiv preprint arXiv:2505.22094, 2025. 3

- [63] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274, 2025. 2
- [64] Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, et al. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. arXiv preprint arXiv:2506.04308, 2025. 2
- [65] Enshen Zhou, Cheng Chi, Yibo Li, Jingkun An, Jiayuan Zhang, Shanyu Rong, Yi Han, Yuheng Ji, Mengzhen Liu, Pengwei Wang, et al. Robotracer: Mastering spatial trace with reasoning in vision-language models for robotics. arXiv preprint arXiv:2512.13660, 2025. 2

## Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation Supplementary Material

### Appendix

This supplementary material provides comprehensive details regarding the proposed method, Dopamine-Reward, and the policy learning framework, Dopamine-RL, along with additional experimental results omitted from the main manuscript due to space constraints. Section A provides rigorous theoretical proofs for the bounded global progress, the existence of the “semantic trap,” and the optimal policy invariance, etc. Section B elaborates on the composition, statistics, and sampling strategies of our 35M-sample training dataset. Section C outlines detailed experimental setups for GRM, simulation and real-world evaluations. Section D presents additional qualitative visualizations to demonstrate the effectiveness of our General Reward Model (GRM). Finally, Section E discusses limitations and potential future research directions.

### A. Proof

#### A.1. Proof of Bounded Global Progress (Proof 1)

In this subsection, we provide a formal proof that iteratively applying the predicted relative progress hops guarantees that the reconstructed global progress Φ⋆(s) remains strictly within the bounds [0,1], provided that the initial state is bounded and the model predictions lie within [−1,1].

First, we define the general recursive update rule. Based on the definition of the hop label H(sp,sq) in Equation 2, we derive the recursive update rule for estimating the global progress of the next state Φ⋆(st) given the current state Φ⋆(st−1) and the predicted hop H = H(st−1,st). We assume the normalization where Φ(s0) = 0 and Φ(sM) = 1. Rearranging the equation, the update rule is:

Φ⋆(st−1) + H · [1 − Φ⋆(st−1)] if H ≥ 0 Φ⋆(st−1) + H · Φ⋆(st−1) if H < 0

Φ⋆(st) =

(18)

Given that the initial progress Φ⋆(s0) = 0 and the predicted hop H ∈ [−1,1], the reconstructed global progress Φ⋆(st) satisfies Φ⋆(st) ∈ [0,1] for all steps t.

We proceed by mathematical induction as follow: (1) Base Case (t = 0): By definition, Φ⋆(s0) = 0, which satisfies 0 ∈ [0,1]. (2) Inductive Step: Assume that for step t − 1, the hypothesis holds: 0 ≤ Φ⋆(st−1) ≤ 1. Let G = Φ⋆(st−1) for brevity, where G ∈ [0,1]. We analyze the next state Φ⋆(st) under two cases (i.e., Positive Hop and Negative Hop) based on the sign of the predicted hop H.

- • Case 1: Positive Hop (Progress), 0 ≤ H ≤ 1. From Equation (18), the update is written as:

- Φ⋆(st) = G + H(1 − G) (19)

Rearranging terms to view this as a convex combination:

- Φ⋆(st) = H + G(1 − H) (20)

Lower Bound: Since G ≥ 0, H ≥ 0, and (1 − H) ≥ 0, it follows that Φ⋆(st) ≥ 0. Upper Bound: Since G ≤ 1, we substitute the maximum value of G:

Φ⋆(st) = H + G(1 − H) ≤ H + 1 · (1 − H)

= H + 1 − H

= 1

Thus, 0 ≤ Φ⋆(st) ≤ 1 when H ≥ 0.

- • Case 2: Negative Hop (Regress), −1 ≤ H < 0. From Equation (18), the update is:

Φ⋆(st) = G + H · G = G(1 + H) (21)

Lower Bound: Since H ∈ [−1,0), the term (1+H) ≥ 0. Since G ≥ 0, the product G(1 + H) ≥ 0. Upper Bound: Since H < 0, the term (1 + H) < 1. Combining this with G ≤ 1:

Φ⋆(st) = G(1 + H) ≤ 1 · (1) = 1 Thus, 0 ≤ Φ⋆(st) ≤ 1 when H < 0.

Conclusion. Since the property holds for the base case and is preserved in both update scenarios during the inductive step, we conclude that Φ⋆(st) ∈ [0,1] for all t.

#### A.2. Proof of the Semantic Trap (Proof 2)

In this subsection, we provide a theoretical derivation demonstrating why a naive dense reward formulation, defined as the direct increment of progress r(st,at,st+1) = Φ(st+1) − Φ(st), fundamentally alters the reinforcement learning objective, leading to the “Semantic Trap” described in the main text.

Consider the standard objective in reinforcement learning, which is to maximize the expected discounted return with a discount factor γ ∈ (0,1):

∞

γtr(st,at,st+1) s0 . (22)

J(π) = Eπ

t=0

Substituting the naive progress-difference reward r(st,at,st+1) = Φ(st+1) − Φ(st) into the cumulative return for a finite horizon T:

GT =

T−1

γt[Φ(st+1) − Φ(st)]. (23)

t=0

We can split the summation into two distinct terms:

GT =

T−1

γtΦ(st+1) −

t=0

T−1

γtΦ(st). (24)

t=0

By applying a variable substitution k = t + 1 to the first term, we rewrite it as Tk=1 γk−1Φ(sk). The second term can be expanded as Φ(s0) + Tt=1−1 γtΦ(st). Substituting these back into the expression for GT yields:

GT =

T−1

γt−1Φ(st) + γT−1Φ(sT) −

t=1

T−1

γtΦ(st) .

Φ(s0) +

t=1

(25)

We now group the terms for each time step t ∈ [1,T − 1]:

T−1

(γt−1 − γt)Φ(st) + γT−1Φ(sT)

GT = −Φ(s0) +

t=1

(26)

= −Φ(s0) + (1 − γ)

T−1

γt−1Φ(st) + γT−1Φ(sT).

t=1

(27)

Since the progress metric Φ(s) is bounded within [0,1] and γ ∈ (0,1), the term γT−1Φ(sT) vanishes as T → ∞. The infinite horizon return converges to:

GT = −Φ(s0) + (1 − γ)

G = lim

T→∞

∞

γt−1Φ(st). (28)

t=1

Because Φ(s0) is a constant with respect to the policy π, maximizing the expected return is equivalent to:

Eπ

J(π) ∝ arg max

arg max

π

π

∞

γt−1Φ(st) s0 . (29)

t=1

Conclusion: The optimization objective implicitly shifts from maximizing the change in progress to maximizing the accumulated value of progress states over time. This creates a perverse incentive where the agent is encouraged to reach a high-progress state quickly and stagnate there to accumulate rewards at each step, rather than completing the task. This theoretical result confirms the existence of the “Semantic Trap.”

#### A.3. Derivation of Exponential Discounting from Time-Consistency (Proof 3)

In this subsection, we justify the use of the exponential discount factor γ = e−λh. We start from the principle of timeconsistency (or memorylessness). Let D(t) : R≥0 → (0,1] be a discount function. The memoryless property implies that the relative discount factor for an additional delay ∆ should not depend on how much time τ has already passed:

D(τ + ∆) D(τ)

= D(∆), ∀τ,∆ ≥ 0. (30)

Setting τ = t and ∆ = s, this yields the Cauchy functional equation:

D(t + s) = D(t)D(s). (31)

Transforming to the logarithmic domain with ϕ(t) = lnD(t), we obtain the linear equation ϕ(t + s) = ϕ(t) + ϕ(s), the unique continuous solution of which is ϕ(t) = −λt for some constant λ ≥ 0. Thus, the discount function must take the exponential form:

D(t) = e−λt. (32)

In the discrete setting with time step h, this corresponds to the discount factor γ = D(h) = e−λh. This theoretical foundation ensures that our reward formulation is robust to variations in control frequency.

#### A.4. Consistency of Reward Shaping in Continuous and Discrete Time (Proof 4)

In this subsection, we derive the continuous-time counterpart of our discrete potential-based reward shaping term γΦ(st+1) − Φ(st). We demonstrate that the discrete shaping term is mathematically equivalent to the first-order Euler discretization of a specific differential equation. Furthermore, we show that its cumulative sum converges to a boundary term that ensures policy invariance.

###### A.4.1. Notation and Definitions

Let h = ∆t denote the discretization time step. We map the discrete time steps t,t + 1 to continuous time moments t,t + h. Let s(t) denote the state trajectory in continuous time.

- • The relationship between the continuous discount rate λ > 0 and the discrete discount factor γ is defined as:

γ = exp(−λh). (33)

- • Let Φ(t) := Φ(s(t)) denote the potential value along the trajectory. Its total time derivative is:

Φ(˙ t) =

d dt

Φ(s(t)). (34)

###### A.4.2. First-Order Taylor Expansion

We perform a first-order Taylor expansion for both the potential function Φ(s(t + h)) and the discount factor γ with respect to the step size h:

Φ(s(t + h)) = Φ(s(t)) + hΦ(˙ t) + O(h2), (35) γ = e−λh = 1 − λh + O(h2). (36)

###### A.4.3. Derivation of the Instantaneous Shaping Term Substituting the expansions into the discrete shaping term γΦ(st+1) − Φ(st) (where st+1 ≡ s(t + h)):

γΦ(s(t + h)) − Φ(s(t))

= (1 − λh)(Φ(t) + hΦ(˙ t)) − Φ(t) + O(h2)

= Φ(t) + hΦ(˙ t) − λhΦ(t) − λh2Φ(˙ t) − Φ(t) + O(h2)

= h Φ( ˙ t) − λΦ(t) + O(h2).

(37)

Dividing by h and taking the limit as h → 0 yields the instantaneous density of the reward shaping:

γΦ(s(t + h)) − Φ(s(t)) h

= Φ(˙ t) − λΦ(t). (38)

lim

h→0

This result indicates that the single-step potential-based reward is, to the first order, the rectangular integration of Φ(˙ t) − λΦ(t) over the interval [t,t + h].

###### A.4.4. From Cumulative Sum to Integral Form We now consider the cumulative discounted sum of the

shaping reward over a horizon N. Let tk = k · h. The discount factor at step k is γk = (e−λh)k = e−λt

k. The discrete cumulative sum is:

N−1

γk [γΦ(sk+1) − Φ(sk)]. (39)

k=0

Substituting the first-order approximation derived above:

N−1

h(Φ(˙ tk) − λΦ(tk)) + O(h2)

e−λt

k

k=0

N−1

(Φ(˙ tk) − λΦ(tk)) + O(h).

h · e−λt

=

k

k=0

(40)

As h → 0, this Riemann sum converges to the definite integral:

T

−−−→h→0

e−λt Φ( ˙ t) − λΦ(t) dt. (41)

0

###### A.4.5. Boundary Terms and Consistency

Using the product rule for differentiation, we observe that the integrand is exactly the total derivative of the discounted potential:

- d

dt

e−λtΦ(t) = e−λtΦ(˙ t)−λe−λtΦ(t) = e−λt(Φ(˙ t)−λΦ(t)).

(42) Thus, the integral can be evaluated analytically:

T

0

d dt

e−λtΦ(t) dt = e−λtΦ(s(t)) T0

= e−λTΦ(s(T)) − Φ(s(0)).

(43)

Assuming Φ(s) is bounded, as T → ∞, the term

- e−λTΦ(s(T)) vanishes. The cumulative shaping reward simplifies to a constant boundary term:

T

e−λt(Φ˙ − λΦ)dt = −Φ(s(0)). (44)

lim

T→∞

0

This confirms that in continuous time, the total shaping reward depends only on the initial state, preserving policy invariance.

###### A.4.6. The ODE / Euler Method Perspective

Finally, we provide an intuitive interpretation using Ordinary Differential Equations (ODEs). Let us define the discounted potential as a state variable y(t):

y(t) := e−λtΦ(s(t)). (45) The dynamics of y(t) are governed by:

dy dt

= e−λt(Φ(˙ t) − λΦ(t)). (46)

If we apply the Forward Euler method to solve this ODE numerically at discrete steps tk with step size h:

dy dt t

y(tk+1) ≈ y(tk) + h ·

Substituting the definitions back:

. (47)

k

e−λ(t

k+h)Φ(sk+1) ≈e−λt

Φ(sk)

k

(48)

(Φ(˙ tk) − λΦ(tk)).

+ h · e−λt

k

k (noting that e−λh = γ):

Multiplying both sides by eλt

γΦ(sk+1) ≈ Φ(sk) + h(Φ(˙ tk) − λΦ(tk)). (49) Rearranging terms yields:

γΦ(sk+1) − Φ(sk) ≈ h(Φ(˙ tk) − λΦ(tk)). (50) Conclusion: The discrete potential-based reward shaping term γΦnext − Φcurr is exactly the update step of the Forward Euler method applied to the differential equation of the discounted potential. This proves that our method is structurally consistent with the underlying continuous-time physics of the task.

#### A.5. Policy Invariance under GRM (Proof 5)

In this subsection, we prove that adding the shaping term derived above preserves the optimal policy. We show this by demonstrating that the cumulative shaped reward telescopes to a boundary term that is independent of the policy’s actions.

Let the shaped reward be rGRM = renv + F, where F(st,st+1) = γΦ(st+1) − Φ(st). The shaping component of the Q-function, Sπ(s,a), is the expected sum of discounted shaping rewards:

Sπ(s,a) = Eπ

∞

γt(γΦ(st+1) − Φ(st)) . (51)

t=0

We analyze the finite horizon sum GFT :

GFT =

=

T−1

T−1

γt+1Φ(st+1) −

γtΦ(st)

t=0

t=0

T−1

T

γtΦ(st) .

γkΦ(sk) − Φ(s0) +

t=1

k=1

(52)

This is a telescoping sum. The intermediate terms cancel out, leaving only the boundary terms:

GFT = γTΦ(sT) − Φ(s0). (53)

Assuming Φ ∈ [0,1] and γ < 1, taking the limit T → ∞ yields limT→∞ γTΦ(sT) = 0. Thus:

∞

γtF(st,st+1) = −Φ(s0). (54)

t=0

This matches the integral of the continuous form derived in Proof 4: 0 ∞ dtd (e−λtΦ)dt = [e−λtΦ]∞0 = −Φ(0). Since the shaping term evaluates to −Φ(s) (where s is the state at t = 0) regardless of the future trajectory, the Q-values are shifted by a state-dependent constant:

QπGRM(s,a) = Qπenv(s,a) − Φ(s). (55) This shift preserves the ordering of actions:

QπGRM(s,a1) ≥ QπGRM(s,a2)

⇕ (56) Qπenv(s,a1) ≥ Qπenv(s,a2).

Therefore, the optimal policy remains: πGRM∗ = πenv∗ .

### B. Details of GRM Training Data

In this section, we present a comprehensive breakdown of the training data used to construct our General Reward Model (GRM). To ensure the model possesses robust generalizability across diverse embodiments, environments, and

semantic tasks, we curated a massive corpus comprising over 35 million training samples derived from approximately 3,400 hours of raw video footage. This dataset aggregates diverse sources spanning real-world robotics, highfidelity simulation, and human-centric interactions. We begin by categorizing the constituent datasets (Section B.1), followed by an analysis of the embodiment diversity and task statistics (Section B.2). Finally, we detail the rigorous data processing pipeline, specifically our Stratified Relative Progress Sampling strategy (Section B.3), which transforms raw trajectories into balanced and high-quality supervision signals for reward learning.

#### B.1. Data Sources

Our training data is composed of the following established datasets and self-collected supplements:

###### B.1.1. Real-World Datasets

- • AGIBot-World [7]: A large-scale bimanual manipulation dataset collected on the AGIBot-A2D humanoid platform. It provides approximately 3.4M samples focusing on high-dimensional, dual-arm coordination tasks and contact-rich interactions, serving as a critical source for humanoid embodiment generalization.
- • DROID [24]: A distributed robot interaction dataset collected across multiple institutions. It contributes 8.98M samples featuring the Franka Emika Panda robot. DROID is characterized by its extreme diversity in background scenes, lighting conditions, and object instances, providing robust priors for visual perception.
- • RoboBrain-X [17]: A comprehensive skill-learning dataset covering a wide range of everyday manipulation skills. We utilize subsets totaling ∼3.0M samples, collected on agile platforms (e.g., Agilex Piper, Galaxea R1, AGIBot-A2D). This data enriches the GRM with finegrained motion primitives.
- • Self-Collected (Real): To bridge the domain gap between open datasets and our specific experimental setups, we collected an additional 1.1M samples with the RoboOS [48, 49]. These include specific long-horizon tasks (e.g., folding, assembly) and corner cases to enhance model robustness.

Collectively, these real-world datasets ground our GRM in physical reality, effectively mitigating the “sim-to-real” gap often observed in reward modeling. The combination of DROID’s extreme visual diversity with the complex morphologies present in AGIBot-World and RoboBrain-X ensures that the model learns robust, embodiment-invariant representations. This allows the GRM to remain stable across varying lighting conditions, textures, and kinematic structures, which is essential for reliable deployment in unstructured environments.

[Figure 80]

#Episode

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

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

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

35M

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

- Figure 5. Overview of GRM training data. (Left) The hierarchical composition of our 35M-sample training corpus. The dataset is derived from episodes spanning Real-World Robotics, Simulation, and Human-Centric domains, and is further expanded via multi-view augmentation. (Right) The long-tail distribution of task categories sorted by episode count (log scale). The dataset covers a broad spectrum of manipulation skills, ranging from atomic primitives (e.g., pick, push) to complex, multi-stage horizons (e.g., assemble, fold).

• Self-Collected (Human): We supplemented the human data with 0.57M samples of domain-specific demonstrations, ensuring coverage of tasks analogous to our robot evaluation protocols.

###### B.1.2. Simulation Datasets

- • LIBERO [30]: A benchmark originally designed for lifelong robot learning. We incorporate 1.33M samples from its task suites (Spatial, Object, Goal, 100), which provide procedurally generated tasks with language instructions, aiding the model in aligning visual progress with semantic goals.
- • RoboCasa [40]: A large-scale simulation framework focused on everyday kitchen activities. It leverages Generative AI to create diverse assets and layouts. We use 0.52M samples to capture realistic object interactions and complex scene semantics in a controlled environment.
- • RoboTwin [9]: A high-fidelity digital twin dataset designed for bimanual manipulation. We utilize a paired dual-view configuration (yielding 1.68M samples from 839k trajectories), which helps the model learn geometryaware representations crucial for depth disambiguation.

Integrating human video data is pivotal for scaling general manipulation intelligence beyond the limits of available robot demonstrations. By leveraging the massive scale of EgoDex, our GRM acquires universal object affordance priors, learning “how” objects should be manipulated regardless of the specific actuator. This cross-embodiment transfer is critical for evaluating progress in novel tasks where robotspecific data may be scarce, enabling the reward model to generalize purely based on the observed state changes of the objects themselves.

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

#### B.2. Statistics and Embodiment Diversity

[Figure 238]

Our compiled dataset represents one of the most comprehensive collections for robotic manipulation to date. As detailed in Table 6, the final training corpus of 35 million samples is strategically balanced to maximize generalization. Real-world interaction data constitutes the majority (∼60%) to ensure the model is grounded in physical reality, while high-fidelity simulation (∼13%) and large-scale human videos (∼26%) provide necessary semantic breadth and affordance priors that are difficult to obtain from robots.

While real-world data offers physical fidelity, these simulation environments provide a controlled testbed for learning precise semantic and geometric alignments. The procedural generation inherent in LIBERO and RoboCasa exposes the model to a vast combinatorial space of task instructions and object layouts, fostering strong instruction-following capabilities. Furthermore, the clean, occlusion-free labels and paired views from RoboTwin allow the GRM to learn finegrained geometric correspondences that are often noisy or unavailable in real-world data.

[Figure 240]

[Figure 241]

Embodiment Agnosticism. A core design principle of our GRM is robustness to morphological variations. As illustrated in Table 7, the dataset covers a wide spectrum of robot embodiments, preventing the model from overfitting to specific kinematics or camera calibrations. By training on this heterogeneous mixture, the GRM learns to focus on the state changes of the objects rather than the robot’s motion, enabling zero-shot transfer to unseen robot configurations.

###### B.1.3. Human-Centric Datasets

• EgoDex [19]: A large-scale egocentric video dataset capturing dexterous human hand-object interactions. With 6.61M samples, this dataset provides strong priors for understanding hand-object affordances and manipulation logic independent of robot morphology.

###### Task and Instruction Diversity. Figure 5 (Right) high-

- Table 6. Statistics of Raw Data Sources. The table lists the raw frame counts (in thousands, ‘k’) before augmentation. The final training set is expanded to 35M via multi-view expansion and augmentation strategies.

Domain Dataset Source Raw Samples

Real-World

AGIBot-World [7] 3,400k DROID [24] 8,983k RoboBrain-X [17] 3,025k Self-Collected (Real) 1,107k

Subtotal 16,515k

Simulation

LIBERO [30] 1,330k RoboTwin [9] 1,678k RoboCasa [40] 523k

Subtotal 3,531k

Human

EgoDex [19] 6,610k Self-Collected (Human) 574k

Subtotal 7,184k Total Raw Corpus 27,230k

- Table 7. Distribution by Robot Embodiment. Our dataset spans diverse robot hardware, from single-arm industrial robots to bimanual humanoids, ensuring strong physical generalization.

Robot Embodiment Primary Data Sources Samples Franka Emika Panda DROID, LIBERO, RoboCasa 8,983k AGIBot-A2D AGIBot-World, RoboBrain-X 3,400k Agilex Piper RoboBrain-X, Self-Collected 3,552k ARX-X5 RoboTwin 882k Galaxea RoboBrain-X, Self-Collected 695k UR5 Self-Collected 433k

lights the semantic diversity of the dataset. The task distribution follows a natural long-tail pattern, spanning from high-frequency atomic primitives (e.g., pick, place, push) that build a solid generalist foundation, to complex, multi-stage horizons (e.g., fold, assemble, pour) that challenge the model’s ability to track long-term progress. To further enhance semantic robustness, we employed Gemini 2.5 Pro [14] to re-annotate task instructions, thereby introducing linguistic diversity and reducing overfitting to rigid template prompts.

#### B.3. Sampling Strategy and Data Balancing

To train the General Reward Model (GRM) effectively, it is crucial to generate training pairs that cover the full spectrum of task progress dynamics. Naively sampling random pairs from a trajectory typically results in a long-tail distribution dominated by small, positive progress steps, leading to model bias. To address this, we use the Stratified Relative Progress Sampling strategy combined with an ag-

gressive augmentation pipeline, as described in main paper. More details are as follow:

- B.3.1. Hop-based Relative Progress Formulation Consistent with the main paper, we define the ground-

truth relative progress label H(sp,sq) using a relativerelative formulation. Let a trajectory consist of M steps {s0,...,sM} with linear global progress Φ(si) = i/M. Without loss of generality, we normalize the global progress such that the initial state has Φ(s0) = 0 and the goal state has Φ(sM) = 1. For any training pair (sp,sq):

H(sp,sq) =

 



Φ(sq) − Φ(sp) 1 − Φ(sp)

if q ≥ p (PROGRESS)

Φ(sq) − Φ(sp) Φ(sp)

if q < p (REGRESS).

(57)

This formula ensures that forward progress is normalized by the remaining distance (1 − Φ(sp)), making late-stage steps statistically significant, while regression is normalized by the accumulated distance (Φ(sp)). The continuous output H ∈ [−1,1] is quantized into integer percentage tokens for VLM training. See Figure 8 for the whole prompt.

- B.3.2. Hierarchical Score-Gap Stratified Sampling

To prevent the model from overfitting to specific step sizes, we implement a two-stage sampling mechanism:

- • Score Balancing: We discretize the progress score range [−100%,+100%] into Nscore uniform bins (e.g., Nscore = 25). We generate a large candidate pool of random pairs and fill these bins to enforce a uniform distribution of reward values. This ensures the model encounters rare events, such as significant regression (errors) or large forward jumps, as frequently as common small steps.
- • Temporal Gap Diversification: Within each score bin, we further categorize samples based on their temporal

distance ∆t = |tpost −tpre| into Ngap bins (e.g., Ngap = 4). This method explicitly forces the model to distinguish between fast progress (large score change in short time) and slow progress (large score change over long time), effectively decoupling visual state change from the temporal duration of trajectories.

- B.3.3. Zero-Hop Anchoring

Standard comparative ranking often struggles with static or near-static pairs, leading to hallucinated progress values. To mitigate this, we explicitly inject a fixed ratio α (e.g., α = 5%) of Zero-Score Samples. These samples are constructed by selecting pairs (t,t+δ) where the temporal delta δ is negligible (randomly sampled within ±10% of the local window). These pairs are labeled with a strict score of

Table 8. GRM Training Hyperparameters.

Task- Prompt

[Figure 242]

[Figure 243]

You are a rigorous, impartial vision evaluator for robot task progress. … The task instruction is “Fold the plans”. Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing …

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Hyperparameter GRM-3B GRM-8B Global Batch Size 512 256

STATE_GOAL

BEFORE AFTER

STATE_INIT

Vision Encoder

LR: {ψvViT,ϕLLMv } {5e−6,1e−5} {5e−6,1e−5} LR Scheduler Cosine Decay Cosine Decay Warmup Ratio 0.03 0.03 Optimizer AdamW AdamW Weight Decay 0.1 0.1 Max Sequence Length 8192 8192 Tensor Parallelism (TP) 1 2 Pipeline Parallelism (PP) 1 2 GPU nodes 16×8 H100 16×8 H100

Tokenizer

Projector

…

[Figure 258]

LLM Decoder

…

Progress Hop or Regress Hop

- Figure 6. Overview of GRM model structure. The GRM is built upon the Qwen2.5-VL architecture. It processes a multimodal interleaved input sequence consisting of task text instructions and multi-view images: the initial state (sinit), the goal state (sgoal), and the paired “BEFORE” (spre) and “AFTER” (spost) observation sets. The visual signals are processed by a shared Vision Encoder and Projector, then fed into the LLM Decoder, which autoregressively predicts a quantized relative progress token (e.g., <score>+15%</score>) or a regress token (e.g., <score>-4%</score>).

Training Duration ∼ 8 Days ∼ 14 Days

ity. For inputs, the model accepts a multimodal prompt sequence. Visual inputs include the Initial State Iinit, Goal State Igoal, and two sets of multi-view observations for the transition being evaluated: Spre = {Iprev }Vv=1 and Spost = {Ipostv }Vv=1, where V is the number of available camera views (e.g., one front, two wrists). Textual inputs include the system prompt defining the rigorous evaluator persona and the specific task instruction dtask. For outputs, the model outputs a discrete token representing the relative progress hop H(Spre,Spost) ∈ [−1,1].

0%. This creates a “semantic anchor” for the model, teaching it that visual similarity implies zero progress, thereby reducing false positives in stable robotic states.

Training Infrastructure. We conducted large-scale training on a high-performance cluster consisting of 128 × NVIDIA H100 (80GB) GPUs. The training framework utilizes the Megatron-LM architecture [47] for efficient distributed training. We employed Tensor Parallelism (TP) and Pipeline Parallelism (PP) to maximize throughput. Detailed hyperparameters are provided in Table 8. The 3B model was trained for approximately 8 days, while the 8B model required 14 days to converge on the 35M-sample dataset.

- B.3.4. Data Augmentation and Robustness

Finally, to bridge the gap between the raw data count and our effective training scale, and to ensure robustness against perceptual failures, we employ an aggressive data augmentation pipeline:

- • Multi-View Expansion: We maximize the utility of multi-camera setups (e.g., permuting available wrist and third-person views) by treating distinct viewpoints as complementary training signals. This strategy expands the dataset size from 27.2M raw samples to 35M training samples, improving the model’s geometric consistency.
- • Perceptual Robustness Training: To prevent the model from over-relying on any single input modality, we introduce Random Viewpoint Dropout (randomly masking specific camera feeds) and Context Dropout (randomly

masking sinit or sgoal tokens) during training. This forces the GRM to infer progress from partial or occluded observations, significantly enhancing resilience in realworld deployments.

- C. Experiment Details

Evaluation Protocols. We evaluate the GRM on two distinct tasks to assess both its fine-grained temporal understanding and its high-level task success judgment via vLLM engine [27]. For Video Frame Rank-Correlation, we use Value-Order Consistency (VOC) [37] as main metric, which assesses whether the reward model correctly orders states based on temporal progress. We measure VOC on unseen test data, where details are as follow:

- • Data Selection: For each dataset (e.g., AgiBot-World, DROID), we randomly sample 10 trajectories per subclass in each dataset from the hold-out validation set.
- • Metric: Given two frames tA and tB where tA < tB, a

correct prediction requires R(st

A

) < R(st

B

). The VOC score is the correlation coefficient [−1,1].

- • Baselines: We compare with VLAC [59] and GVL [37] using their official codebase or recommended prompts.

#### C.1. GRM Training and Evaluation

Model Details. Our GRM leverages the RoboBrain 2.0 [51] architecture, as shown in Figure 6, exploring two parameter scales: a lightweight 3B variant for efficient inference and a powerful 8B variant for maximum reasoning capabil-

For Task Completion Judgment, to verify if the GRM can serve as a reliable success detector, we collected 60 real-

https://github.com/InternRobotics/VLAC

world rollouts for each of three challenging tasks: Stacking Blocks, Folding T-shirt / Pants, and Clearing Desktop. These rollouts are uniformly distributed into 20 Successful (SE), 20 Partially Successful (PSE), and 20 Failed (FE) episodes. We define the automated judgment logic following SARM [8]. Let Pt ∈ [0,1] be the accumulated global progress at step t predicted by the GRM. The trajectory classification label is determined by:

 

T

3 T

SE, if Pfinal > 0.8 ∧

Pt > 0.6,

Label =

t=2T/3

PSE, if T1 Tt=1 Pt ≥ ξ, FE, otherwise.



(58) where ξ = 0.4 is a threshold for partial progress.

#### C.2. Simulation Experiments

This section details the simulation experiments: the benchmark and two different “RL algorithm + VLA model” settings we employ to validate our methods.

###### C.2.1. Benchmark

We evaluate our framework on the LIBERO-Goal suite [30], a challenging subset of the LIBERO benchmark consisting of 10 distinct long-horizon manipulation tasks. These tasks require the agent to reason about specific goal configurations and perform precise object interactions, serving as a rigorous testbed for progress-based reward modeling.

###### C.2.2. Setting 1: PPO + OpenVLA-OFT

In this setting, we combine PPO (Proximal Policy Optimization) [46], a stable and sample-efficient on-policy RL algorithm, with OpenVLA-OFT [26], an optimized visionlanguage-action (VLA) model, to build the reinforcement learning pipeline, implementing it via the RLinf codebase [58]. OpenVLA-OFT acts as the base policy model, refined from the OpenVLA backbone through an Optimized Fine-Tuning (OFT) recipe that integrates parallel decoding with action chunking (for high-throughput action generation), continuous action representation (avoiding lossy discretization), and an L1 regression objective (simplifying training while preserving precision). It maps multimodal inputs to continuous robot actions in one forward pass. For LIBERO-Goal’s long-horizon tasks, we set an action chunk size of 8 to capture temporal dependencies and reduce compounding errors.

PPO optimizes the OpenVLA-OFT policy by confining updates to a trust region, using a clipped surrogate objective to balance exploration and exploitation. Its core optimization objective is defined as:

JPPO(θ) = Et min ρt(θ)Aˆt, clip (ρt(θ), 1 − ϵ, 1 + ϵ) Aˆt .

(59)

[Figure 259]

Figure 7. Training Curve on LIBERO-Goal. The success rate of our ReinFlow agent fine-tuned with GRM-based reward shaping. The agent demonstrates stable convergence and achieves an average success rate of over 80% across the benchmark tasks.

θ(at|ot)

Here, ρt(θ) = π

πθold(at|ot) is the ratio between the current policy πθ and the fixed rollout policy πθ

, Aˆt is the estimated advantage at timestep t, and ϵ is the clipping parameter that prevents excessive updates. We follow RLinfVLA [58]’s PPO best practices: using action-level value estimation (superior for chunked policies) and partial reset (resetting sub-environments post-completion to boost sample efficiency).

old

###### C.2.3. Setting 2: ReinFlow + π0

In this setting, we combine ReinFlow [61], a flow-matching based reinforcement learning algorithm, and π0 [6], a flowmatching based vision-language-action model to build the reinforcement learning pipeline and implement it using the RLinf [58] codebase. The training is conducted on a compute node equipped with 8 NVIDIA H100 GPUs. The total training time is approximately 50 hours, achieving an average success rate of 81% across the LIBERO-Goal tasks (Figure 7). ReinFlow facilitates stable fine-tuning by injecting learnable noise into the flow matching policy’s deterministic path, converting it into a discrete-time Markov process. Specifically, the denoising process is modified as follows:

ak+1 ∼ N ak + vθ(tk,ak,o)∆tk, σθ2′(tk,ak,o) (60)

where vθ is the velocity network and σθ′ is a learnable noise network. This formulation allows for an exact computation of the transition probability:

πθ(ak+1|ak,o) = lnN ak+1 ak

+vθ(tk,ak,o)∆tk, σθ2′(tk,ak,o)

(61)

https://github.com/RLinf/RLinf

The policy is then optimized using a PPO-style objective that jointly updates the velocity and noise networks:

B

−Aθ¯

θ,θ′ =arg min

old

θ,θ′

i=1

+ α · R(ai,oi)]

K−1

(oi,ai)

k=0

lnπθ¯(aki+1|aki ,oi)

(62)

where A is the advantage function, and R is a regularization term (e.g., entropy). Furthermore, detailed hyperparameters for the ReinFlow training are provided in Table 9.

Table 9. Hyperparameters for LIBERO-Goal Experiments.

###### Hyperparameter Value

Algorithm ReinFlow Consistency Sensitivity α 20 Discount Factor γ 0.99 Batch Size 1792 Learning Rate 5e-6 Compute Resources 8 × H100

#### C.3. Real-World Experiments

This section details the real-world robotic experiments: the algorithm we use and 8 representative tasks we conduct.

###### C.3.1. Algorithm

We adopt the Consistency-based Reinforced Fine-Tuning (ConRFT) algorithm [12] in our real-world experiments. ConRFT is an Cal-QL [39] algorithm variant designed as an offline-to-online reinforcement learning (RL) method rooted in Q-learning for fine-tuning pre-trained VisionLanguage-Action (VLA) models. ConRFT extends CalQL’s framework by adding a consistency policy and a behavior cloning loss, enabling efficient and safe adaptation for real-world robotic manipulation tasks. Key components are as follows:

Algorithm Overview. ConRFT operates in two sequential phases: offline initialization and online adaptation. It maintains two data buffers: a demonstration buffer (D) and a replay buffer (R). The offline phase leverages a small set of human demonstrations (20-30 trajectories) stored in D to initialize a policy via behavior cloning (BC) and calibrated Q-learning (Cal-QL) [39], ensuring stable and safe initial behavior without requiring real-world exploration. The online phase then refines this policy through interaction with the physical world, storing transitions in R. A human-inthe-loop (HIL) component intervenes to correct unsafe actions during online deployment, with these corrected trajectories added back to D to guide subsequent policy updates. Symmetric sampling from the combined dataset D ∪ R is employed to mitigate distribution shift between the offline and online stages.

Policy model. The policy model is built on the OctoSmall model [53], a lightweight VLA backbone selected for its balance of performance and inference efficiency. The model is trained with visual encoders and transformer backbone and its original action head is replaced with a consistency policy [10], a diffusion-based module that maps Gaussian-sampled random actions to expert-like behaviors, conditioned on the observation embeddings.

Offline Stage. The policy is initialized exclusively using data from the demonstration buffer D, with a diffusionbased consistency policy (parameterized by ψ) serving as the action head. The unified training objective integrates BC and Q-learning to balance expert imitation and reward alignment, defined as

Lofflineπ (ψ) = βLBCπ + ηLQπ, (63)

where β and η are hyperparameters that balance the two loss terms. The BC loss minimizes the Euclidean distance between actions generated by the consistency policy and expert demonstrations from D, formulated as

LBCπ = E (s,a)∼D

∥fψ(a + kmz,km | Eϕ(s)) − a∥2 ,

m∼U[1,M−1]

(64) in which (s,a) ∼ D denotes sampling state-action pairs from the demonstration buffer, m ∼ U[1,M − 1] indicates random sampling of a diffusion sub-interval, km is the diffusion step corresponding to sub-interval m, z ∼ N(0,I) is Gaussian noise following a standard normal distribution, Eϕ(s) is the observation embedding output by the frozen Octo-Small backbone, fψ is the consistency policy module parameterized by ψ, and ∥ · ∥2 denotes the Euclidean distance. The Q loss maximizes the action-value estimates from a learned critic network Qθ (where θ are the critic’s parameters), given by

LQπ = −Es∼D,a∼π

[Qθ(s,a)], (65)

ψ

with a ∼ πψ indicating actions sampled from the consistency policy. The critic Qθ is updated using the Cal-QL objective:

LofflineQ (θ) = α (Es∼D,a∼π [max(Qθ(s,a),V µ(s))] − E(s,a)∼D [Qθ(s,a)]

- 1

- 2

E(s,a,s′)∼D Qθ(s,a) − BπQ¯θ¯(s,a) 2 . (66)

+

In this formula, α is the conservative penalty hyperparameter, V µ(s) is the value of a reference policy µ that stabilizes estimates for out-of-distribution actions, (s,a,s′) ∼ D samples transition triples (state, action, next state) from D, Bπ is the Bellman backup operator defined as BπQ¯(s,a) =

r(s,a) + γEa′∼πQ¯(s′,a′), and Q¯θ¯ is a delayed target Qnetwork whose parameters θ¯ are fixed during critic updates to ensure training stability.

Online Stage. The online phase refines the policy using combined data from the demonstration buffer D which is augmented with corrected trajectories from HIL interventions and the replay buffer R that stores online interaction transitions. The policy update objective retains the same structure as the offline stage but with adjusted weights to prioritize reward-driven exploration over expert imitation:

Lonlineπ (ψ) = β′LBCπ + η′LQπ, (67) where β′ is reduced and η′ is increased, shifting the focus toward learning from real-world feedback. The BC and Q losses follow the same formulations as in the offline phase, with the only difference being that data sampling is performed from D ∪ R instead of D alone, and all symbols retain their previously defined meanings. The critic Qθ is updated via a standard Bellman error loss without the conservative penalty (as online data reduces distribution shift and mitigates overestimation risks), given by

LonlineQ (θ) = E(s,a,s′)∼D∪R Qθ(s,a) − BπQ¯θ¯(s,a) 2 ,

(68) ensuring accurate value estimation as the policy adapts to real-world dynamics.

Hyperparameters Detailed hyperparameters for realworld experiments are provided in Table 10.

###### C.3.2. Real-World Tasks

In real-world experiments, we selected 8 representative tasks to verify the promotional effect of our reward model on physical robotic reinforcement learning. These tasks cover single-arm tasks, dual-arm tasks, fine-grained tasks, and long-horizon tasks, with detailed descriptions of each task provided as follows.

Insert Square: The end-effector of the robotic arm grasps a square block with four holes. On the table, there is a target board designed for block insertion, which is equipped with four upright pegs corresponding to the four holes on the block. The robot is required to adjust the position and orientation of the block to insert it onto the pegs of the board. This task demands millimeter-level precision tolerance, categorizing it as a single-arm fine-grained task.

Pick and Place: A white gasket and a toy are placed on the table. The robotic arm needs to first grasp the toy and then place it accurately on the gasket. The initial positions of both the gasket and the toy are randomly generated within a predefined range. This task is defined as a single-arm finegrained long-horizon task.

Complete Circuit: A disconnected circuit is placed on the table, and the right arm of the dual-arm robotic system holds a battery. The robot is required to first insert the battery (held by the right arm) into the battery slot, then use the

Table 10. Real-World Experiment Hyperparameters.

Hyperparameter Symbol Value

Global Batch Size - 256 Learning Rate - 3 × 10−4 Reward Discount Factor γ 0.98 Offline BC Loss Weight β 1.0 Offline Q Loss Weight η 0.1 Conservative Penalty Coefficient α 0.1 Online BC Loss Weight β′ 0.5 Online Q Loss Weight η′ 1.0 Compute Resources - 1 × RTX4090

left arm to toggle the circuit switch, thereby lighting up the bulb in the middle of the circuit. This task is classified as a dual-arm fine-grained long-horizon task.

Arrange Flowers: A vase is placed on the table, with the left and right arms of the robotic system each holding a flower. The robot needs to sequentially insert the two flowers into the vase. This task is categorized as a dual-arm fine-grained long-horizon task.

Fold Towel: A towel is laid on the table, and the robotic arm is required to fold it neatly. This task is defined as a dual-arm fine-grained long-horizon task.

Build Blocks: Three building blocks are placed on the table. The robotic arm needs to stack them sequentially to form a castle-like structure. This task is classified as a dualarm fine-grained long-horizon task.

Cap the Pen: The dual-arm robot holds a pen in one arm and a pen cap in the other. It needs to slowly adjust the positions and orientations of the two objects to finally fit the cap onto the pen. This task is categorized as a dual-arm fine-grained task.

Zip the Bag: The left arm of the robot holds a bag, while the right arm clamps the zipper of the bag. The robot is required to coordinate the movements of both arms to zip up the bag completely. This task is defined as a dual-arm fine-grained task.

### D. More Qualitative Results

In this section, we provide additional qualitative visualizations to further substantiate the effectiveness and robustness of our method. We focus on three key aspects: the generalization of GRM across diverse semantic tasks, the temporal robustness of progress estimation under varying sampling intervals, and the trajectory visualization of real-world RL.

GRM Predictions on Diverse Tasks. Figure 9 illustrates the predicted Hop and Progress curves generated by our GRM across a variety of manipulation tasks. The results demonstrate that our model accurately captures the monotonic increase in progress for successful trajectories while effectively identifying stagnation or regression in failed attempts, validating its semantic generalization capabilities.

Robustness to Temporal Intervals. A robust reward model should remain consistent regardless of the video sampling rate. Figure 10 compares the GRM’s progress estimation when inputs are sampled at different intervals (∆t = 10,25,50,100 frames). Despite the significant variation in visual disparity between adjacent frames, our Hopbased formulation ensures that the reconstructed global progress remains consistent, highlighting the model’s ability to decouple physical progress from temporal duration.

Real-World RL Rollout Visualization. Finally, Figure 11 visualizes the robustness of the policy learned for the “Insert Block” task. The policy used in this rollout was trained for approximately 20 minutes and achieved a success rate of over 95%. To evaluate the policy’s reactivity and the reward model’s accuracy, we introduced an artificial disturbance during execution. As shown in the sequence, a human operator manually moves the target slot while the robot is in motion (a). This intervention causes the robot to miss the target and fall into misalignment (b). Crucially, the inset plots show that our Generative Reward Model (GRM) immediately reflects this setback: the estimated Progress drops sharply, correctly identifying that the state has regressed from the goal. This accurate negative feedback guides the agent to adjust its trajectory. The robot successfully recovers from the misalignment (c), repositions itself above the target (d), aligns with the slot (e), and completes the insertion (f). This demonstrates that GRM provides dense, semantically meaningful rewards that enable the agent to recover from unexpected external perturbations.

### E. Future Work

In future research, we plan to expand the capabilities and efficiency of Robo-Dopamine in four primary directions:

- • Efficient GRM Inference: The current VLM-based reward model, while accurate, incurs high computational latency which can bottleneck online RL training loops. We aim to investigate low-precision quantization techniques (e.g., INT4/INT8 quantization or KV-cache compression) to significantly accelerate GRM inference and reduce memory footprint without compromising reward accuracy for RL phase.
- • Continuous Video Stream Reasoning: We aim to further evolve the GRM from discrete frame-pair inference to continuous video stream reasoning. By incorporating historical context sequences and integrating temporal modeling architectures (e.g., Video Transformers, Temporal CNNs, or Video-LLM [3, 57]), the model will gain the ability to capture dynamic motion trends, such as inertia and trajectory, which are essential for high-dynamic tasks. Furthermore, this temporal continuity resolves state ambiguities inherent in static snapshots (e.g., distinguishing between “grasping” and “releasing” an object,

- or tracking cumulative progress in cyclic tasks like scrubbing a test tube where multiple repetitions are required), thereby significantly enhancing the robustness of reward assessment.
- • Large-Scale Generalization: We plan to scale the Dopamine-RL framework to a broader range of tasks in both simulation and the real world. Specifically, we aim to validate the framework on highly dynamic tasks (e.g., throwing, catching) and long-horizon mobile manipulation scenarios to test the limits of the policy-invariant reward shaping mechanism.
- • Multi-Modal Reward Modeling: While vision provides rich global context, fine-grained manipulation often requires non-visual cues. We intend to incorporate tactile and auditory modalities into the GRM. Tactile feedback is crucial for contact-rich tasks (e.g., sensing force during insertion), while audio can detect discrete events (e.g., the “click” of a latch), enabling more precise reward shaping for delicate operations.

##### User Prompt for General Reward Model (GRM)

""" You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task> {task_description}

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):

- - REFERENCE START — Robot Front Image (task just starting): <image>
- - REFERENCE END — Robot Front Image (task fully completed): <image> </Task>

BEFORE Robot Front Image: <image> BEFORE Robot Left Wrist Image: <image> BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image> AFTER Robot Left Wrist Image: <image> AFTER Robot Right Wrist Image: <image>

### Goal Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)

- 1) Calibrate using the references:

- - REFERENCE START = “just beginning”; REFERENCE END = “fully completed.”
- - Visually estimate how far BEFORE and AFTER are along this START→END continuum.

- 2) Direction:

- - AFTER better than BEFORE → positive score.
- - AFTER worse than BEFORE → negative score.
- - Essentially the same → 0.

- 3) Normalize to an integer percentage in [-100%, +100%]:

- - For improvements, scale the improvement relative to what remained from BEFORE to END.
- - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
- - Clip to [-100%, +100%] and round to the nearest integer percent.

###### ### Evaluation Criteria (apply across all three views)

- 1) Task Alignment: Evidence directly tied to {task_description}.
- 2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
- 3) View-Specific Evidence & Consistency:

- - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
- - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
- - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
- - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.

- 4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
- 5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged → 0%.

### Output Format (STRICT) Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign: <score>+NN%</score> or <score>-NN%</score> or <score>0%</score>

"""

Figure 8. User Prompt for General Reward Model (GRM).

- 0
- 1

[Figure 260]

[Figure 261]

Hop Value

-1

START

- 0
- 1

[Figure 262]

Global Progress

[Figure 263]

0.5

END

- (a) Fold the Pants (Real)
- (b) Clean the table (Real)
- (c) Stack three Bowls (Sim.)
- (d) Open the drawer (Sim.)

- 0
- 1

[Figure 264]

[Figure 265]

Hop Value

-1

START

- 0
- 1

[Figure 266]

Global Progress

[Figure 267]

0.5

END

- 0
- 1

[Figure 268]

[Figure 269]

Hop Value

-1

START

- 0
- 1

[Figure 270]

Global Progress

[Figure 271]

0.5

END

- 0
- 1

[Figure 272]

[Figure 273]

Hop Value

-1

START

- 0
- 1

[Figure 274]

Global Progress

[Figure 275]

0.5

END

- Figure 9. GRM Progress Predictions across Diverse Tasks. We visualize the frame-wise Hop (instantaneous change) and accumulated Progress predicted by GRM on unseen validation tasks.

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Frame-0 Frame-400 Frame-700 Frame-1000

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Frame-1700 Frame-1900 Frame-2000 Frame-2200

- 0
- 1

[Figure 284]

- 0
- 1

[Figure 285]

Hop Value

Hop Value

-1

-1

[Figure 286]

[Figure 287]

- 0
- 1

1

Global Progress

Frame-2000

Global Progress

Frame-2000

Frame-1700

Frame-1700

Frame-700

Frame-700

Frame-2200

Frame-2200 Frame-0

0.5

0.5 Frame-0

Frame-400

Frame-1900

Frame-400

Frame-1900

Frame-1000

0

Frame-1000

(a) Frame-Interval = 100 (b) Frame-Interval = 50

- 0
- 1

- 0
- 1

[Figure 288]

[Figure 289]

Hop Value

Hop Value

-1

-1

[Figure 290]

[Figure 291]

- 0
- 1

- 0
- 1

Frame-2000

Frame-2000

Global Progress

Global Progress

Frame-1700

Frame-1700

Frame-700

Frame-700

Frame-2200 Frame-0 Frame-400

Frame-2200

Frame-0

0.5

0.5

Frame-400

Frame-1900

Frame-1900

Frame-1000

Frame-1000

(c) Frame-Interval = 25 (d) Frame-Interval =10

- Figure 10. Progress Estimation Consistency across Sampling Intervals. We plot the reconstructed progress curves for the same trajectory using different frame strides (10, 25, 50, and 100 frames). The high overlap between curves demonstrates that our GRM is robust to temporal granularity and does not simply overfit to a specific frame rate.

[Figure 292]

[Figure 293]

[Figure 294]

(a) Artificial Disturbance Position (b) Fall Into Misalignment

[Figure 295]

[Figure 296]

(c) Misalignment Recovery (d) Move to the top

[Figure 297]

[Figure 298]

(e) Align with the Slot (f) Successful Insertion

- Figure 11. Robustness to Artificial Disturbance during Real-World Execution. We visualize a rollout of the converged policy (success rate > 95%) under human interference. Each sub-figure shows the third-person view, the ego-centric view, and the real-time GRM inference (Top: Hop, Bottom: Progress). (a) Artificial Disturbance Position: A human hand intervenes and shifts the target board while the robot attempts to approach. (b) Fall Into Misalignment: The robot misses the new position. Note that the GRM Progress curve drops significantly (indicated by the red dot in the bottom inset), reflecting the failure state. (c) Misalignment Recovery: The policy reacts to the visual feedback and the drop in reward, adjusting the end-effector position. (d) Move to the top: The robot realigns directly above the target slot. (e) Align with the Slot: Precise fine-tuning before insertion. (f) Successful Insertion: The task is completed, with the progress estimation reaching its peak.

