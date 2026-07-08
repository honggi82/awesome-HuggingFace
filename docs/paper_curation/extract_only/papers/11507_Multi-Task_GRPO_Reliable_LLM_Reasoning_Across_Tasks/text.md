# arXiv:2602.05547v1[cs.CL]5Feb2026

## Multi-Task GRPO: Reliable LLM Reasoning Across Tasks

#### Shyam Sundhar Ramesh∗

UCL Department of EEE UCL Centre for AI

Xiaotong Ji Huawei Noah’s Ark Lab

Matthieu Zimmer Huawei Noah’s Ark Lab

Sangwoong Yoon UNIST Graduate School of AI

Zhiyong Wang University of Edinburgh

Haitham Bou Ammar Huawei Noah’s Ark Lab UCL Centre for AI

Aurelien Lucchi† University of Basel

Ilija Bogunovic† University of Basel UCL Centre for AI

### Abstract

RL-based post-training with GRPO is widely used to improve large language models on individual reasoning tasks. However, real-world deployment requires reliable performance across diverse tasks. A straightforward multi-task adaptation of GRPO often leads to imbalanced outcomes, with some tasks dominating optimization while others stagnate. Moreover, tasks can vary widely in how frequently prompts yield zero advantages (and thus zero gradients), which further distorts their effective contribution to the optimization signal. To address these issues, we propose a novel Multi-Task GRPO (MT-GRPO) algorithm that (i) dynamically adapts task weights to explicitly optimize worst-task performance and promote balanced progress across tasks, and (ii) introduces a ratio-preserving sampler to ensure task-wise policy gradients reflect the adapted weights. Experiments on both 3-task and 9-task settings show that MT-GRPO consistently outperforms baselines in worst-task accuracy. In particular, MT-GRPO achieves 16–28% and 6% absolute improvement on worst-task performance over standard GRPO and DAPO, respectively, while maintaining competitive average accuracy. Moreover, MT-GRPO requires 50% fewer training steps to reach 50% worst-task accuracy in the 3-task setting, demonstrating substantially improved efficiency in achieving reliable performance across tasks. Code: https://github.com/rsshyam/MT-GRPO

### 1 Introduction

Recent advances in RL post-training using policy optimization methods such as Group-Relative Policy Optimization (GRPO) have produced LLMs with impressive performance on individual reasoning benchmarks, including mathematical problem solving, code generation, and structured reasoning tasks [1–3]. Despite these advances, most post-training pipelines are designed and tuned primarily for individual tasks or benchmarks, treating each as an isolated optimization target, with limited work addressing cross-benchmark trade-offs or developing principled approaches for multi-task RL post-training. This becomes increasingly problematic as LLMs are deployed in real-world as general-purpose reasoners rather than specialists for narrow benchmarks, wherein broad competence across diverse reasoning skills is essential for reliability. A model that excels at competition mathematics but struggles with basic logical inference remains unreliable despite strong benchmark performance. This raises a fundamental question that current work largely sidesteps: How should we post-train a single model to improve reasoning across tasks while ensuring that no task is left behind?

∗Correspondence to shyam.ramesh.22@ucl.ac.uk †Co-senior authors

Preprint. Under review.

GRPO

Imbalanced Performance Uniform Distribution 82%

[Figure 1]

[Figure 2]

Zero Grad Filter

[Figure 3]

[Figure 4]

Zebra

33.3% 33.3%

46%

Skewed Flow

Many zero

30.3%

Standard

Countdown

Sampler

advantage prompts

ARC

33.3%

Zebra Countdown ARC

MT-GRPO

###### Weights Accuracies

[Figure 5]

79%

[Figure 6]

[Figure 7]

[Figure 8]

61.6%

Zebra 46% Countdown 11%

59%

Aligned Flow

| |
|---|

Our Ratio Preserving Sampler

| |
|---|

Guaranteed informative prompts

43%

ARC

Zebra Countdown ARC

Zero Grad Filter

Adaptive Distribution (Accuracy, Improvement aware)

Balanced Performance

- Figure 1: GRPO assigns uniform task weights and samples without regard to task difficulty or zero-gradient rates. Consequently, easy tasks (Countdown) dominate while harder tasks (ARC, Zebra) lag, and effective gradient flow is skewed by varying zero-gradient rates ( ⊗ marks high zero-gradient rates). In contrast, MT-GRPO adapts task weights to prioritize weaker tasks and uses a ratio-preserving sampler to align effective gradient contributions with target weights, substantially improving ARC and Zebra and yielding more balanced performance.

Addressing this challenge is non-trivial. Standard multi-task post-training that optimizes for average performance often leads to imbalanced outcomes [4, 5], where strong gains on some tasks mask stagnation on others as illustrated in Figure 1. Moreover, joint post-training of tasks can introduce negative transfer and task interference, where progress on certain tasks hinders learning on others [6, 7]. Together, these issues highlight the need for principled, robustness-aware optimization strategies for multi-task post-training.

In this work, we incorporate task-wise robustness directly into the multi-task RL post-training objective to promote balanced competence across tasks. To make this objective practical and effective in modern post-training pipelines, we propose MULTI-TASK GRPO (MT-GRPO), a novel post-training algorithm with two key ideas. First, MT-GRPO employs improvement-aware task reweighting, using both task-level rewards and task-wise improvement signals to improve worst-task performance and overall multi-task robustness without sacrificing the average performance. Second, it introduces a ratio-preserving, acceptance-aware batch construction mechanism that enforces target task proportions in the training batch, ensuring that learned task weights translate into actual gradient signals (see Figure 1 for illustration).

Similar robustness-aware objectives have been studied in other learning paradigms, including distributionally robust optimization and multi-task learning [8–10, 7, 11], as well as domain reweighting for large-scale pre-training [12–16]. These methods typically operate by adapting weights over tasks or data groups based on their losses to optimize a target objective. However, in the context of RL-based LLM post-training with GRPO, prompts whose rollouts receive identical rewards yield zero advantages and contribute no gradient to the policy update. Since the prevalence of such prompts varies across tasks, effective gradient contributions are dominated by tasks with more non-zero gradient prompts, even when weaker tasks are intentionally upweighted (see Figure 1). Moreover, the GRPO loss is unreliable for task reweighting since it takes similar values when a prompt’s rollouts are all correct or all incorrect. These challenges do not arise in prior settings and require algorithmic solutions beyond existing robust optimization techniques.

We detail other related works extensively in Section 7 and summarize our main contributions below.

- (i) A robustness-aware multi-task RL post-training objective with a tunable trade-off between worst-task robustness and average performance.
- (ii) A task reweighting framework that leverages task-level rewards and task-wise improvements to encourage balanced progress across tasks.
- (iii) A ratio-preserving batch construction that aligns task weights with realized gradient contributions.
- (iv) Empirical evaluation of MT-GRPO by post-training on multi-task reasoning benchmarks that span both planning tasks (Countdown, Zebra puzzles) and inductive reasoning tasks (ARC), across both controlled and larger multi-task settings.

Across all experimental settings, we observe that MT-GRPO consistently improves worst-task accuracy compared to strong baselines while still maintaining competitive average performance, and dynamically reallocates optimization effort toward weaker or slowly improving tasks.

### 2 Problem Formulation

We consider a collection of K reasoning tasks indexed by k ∈ [K] := {1,2,...,K}, where each task k is associated with a dataset Dk and a reward function Rk. The datasets Dk consist of a disjoint set of prompts, where each prompt contains a question specific to task k. The questions admit verifiable correct answers, and the reward function Rk is designed to evaluate both the correctness and formatting of responses to questions from task k. Let πθ denote a base policy with general reasoning capabilities. Our objective is to post-train πθ jointly over these K tasks.

For each task k, we define the task-level performance metric Jk(θ), which denotes the KL-regularized expected reward attained by policy πθ on questions sampled from Dk. The KL regularizer explicitly penalizes deviations from a reference policy πref, ensuring that post-training does not excessively alter the model’s behavior. Concretely, we define

Rk(x,y) − τ KL πθ∥πref , (1)

Jk(θ) := E x∼D

k

y∼πθ(·|x)

where τ controls the strength of KL regularization. A standard RL-based post-training approach for optimizing the policy πθ over K tasks is to maximize the average KL-regularized expected reward,

1 K

max

Javg(θ) =

θ∈Θ

K

Jk(θ). (2)

k=1

Policy gradient algorithms: For the single-task case, the objective in Equation (2) is commonly optimized using policy gradient algorithms such as RLOO, PPO, GRPO, VinePPO, etc., [17, 18, 1, 19]. They apply the policy gradient theorem [20],

∇θJ(θ) = Ex,y ∇θ log πθ(y | x) A ˜(x,y) ,

θ(y|x)

where A˜(x,y) = A(x,y) − τ log π

πref(y|x) and A(x,y) denotes an advantage function measuring the relative quality of response y for prompt x in comparison to the current behavior of the policy. Among these approaches, GRPO has recently emerged as a particularly effective and widely adopted method for improving the reasoning capabilities of large language models [1, 2, 21, 22]. It avoids the need for a value function and leverages within-prompt relative comparisons to construct stable advantage estimates. In particular, GRPO constructs a prompt-level, sample-based advantage. For each prompt x, we sample a group of G responses {yi}Gi=1 from a behavior policy πθ

and define a relative advantage via within-group normalization, e.g., A(x,yi) = R(x,yi) − mean({R(x,yj)}Gj=1 /std({R(x,yj)}Gj=1).

old

Using importance weighting to correct for off-policy sampling, the resulting GRPO objective is

JGRPO(θ) = Ex E{y

i}∼πθold

G

πθ(yi | x) πθ

1 G

θ(yi|x)

(yi | x) · A(x,yi) − τ log π

πref(yi|x) . (3)

old

i=1

In practice, GRPO employs a clipped version of Equation (3), in which the importance ratio ρi = πθ(yi | x)/πθ

(yi | x) is clipped to the interval [1 − ϵ,1 + ϵ]. This clipping mechanism prevents excessively large updates caused by samples with high importance weights and improves training stability. For brevity, we provide the explicit form of the clipped objective in Section A.

old

Limitations of standard GRPO in the multitask setting: Despite its advantages, directly applying GRPO to the multitask objective in Equation (2) by averaging losses across tasks leads to two important issues.

- (i) Lack of task-wise robustness: Optimizing average reward is fundamentally misaligned with the goal of general-purpose reasoning. The mean objective permits solutions in which strong gains on a subset of tasks compensate for substantial underperformance on others, providing no guarantees of task-wise robustness. This often leads to imbalanced outcomes in heterogeneous collections of reasoning tasks [4, 5]. As a result, models post-trained to maximize average performance may lack balanced competence across diverse reasoning skills.
- (ii) Uneven zero-gradient rates across tasks: A structural limitation of extending GRPO to the multi-task setting is that different tasks exhibit widely different rates of zero-gradient prompts. Under GRPO, when all sampled rollouts for a prompt receive identical rewards, the resulting advantages are zero and the prompt contributes no

gradient to the policy update [3, 23]. Yu et al. [3] addresses this issue in the single-task setting by filtering such prompts and resampling. However, this mechanism is insufficient for multi-task training, as post-filtered batches would become biased toward tasks with fewer zero-gradient prompts, which may even hurt overall average performance. More importantly, when weaker tasks are explicitly upweighted, this causes the realized gradient contributions to deviate substantially from the intended task proportions and leaves them under-optimized.

Together, these challenges motivate the need for modified objectives and optimization strategies that preserve the practical benefits of GRPO while explicitly emphasizing task-wise robustness and ensuring appropriate gradient contributions from each task during training.

### 3 Multi-Task Post-Training Objective

In this section, we primarily tackle the task-wise robustness issue in the standard average reward RL objective (see Equation (2) and Limitation (i) in Section 2). We note that this involves designing a novel multi-task post-training objective that explicitly controls performance disparities across tasks and balances robustness and average performance. Our goal is to improve task-level rewards while ensuring robust performance across tasks. In particular, we seek a post-trained policy that satisfies the following desiderata:

- (i) High average performance: The average rewards across all tasks is maximized.
- (ii) Robust across tasks: The difference in rewards between any two tasks is bounded, ensuring that no task significantly underperforms relative to others.

Together, these criteria promote a post-trained policy that achieves balanced competence across different reasoning tasks. We formalize these two objectives using the following constrained optimization problem:

K

1 K

Jk(θ) s.t. |Jk(θ) − Jj(θ)| ≤ ε, (4)

max

θ∈Θ

k=1

where ∀1 ≤ k < j ≤ K, ε ≥ 0 controls the allowable performance disparity between tasks. Setting ε = 0 enforces strict equality of task performance, while larger values progressively relax the constraint toward average reward optimization. The constraints explicitly encourage minimizing disparities across tasks rather than allowing large gains on some tasks to compensate for only marginal gains on others.

For tractability, we work with the equivalent max–min objective obtained via a Lagrangian relaxation of

- Equation (4) (see Section B.1 for the derivation):

max

θ∈Θ

min

z∈∆K

K

k=1

zkJk(θ) − εΩ(z), (5)

where z ∈ ∆K denotes a learned distribution over tasks, and the regularizer Ω(z) = K2 z − K1 1 1 penalizes deviations from uniform weighting. Next, we develop an RL-based post-training algorithm for optimizing

- Equation (5) and analyzes its implications for task-wise robustness.

#### 3.1 Adapting GRPO for Worst-Task Reward Maximization

We begin by considering the case ε = 0, which enforces the strongest notion of task-wise robustness and yields a minimax objective for multi-task post-training. This setting has been extensively studied in the literature under the framework of distributionally robust optimization [9, 12, 13]. A common theme in such approaches to minimax or group-robust objectives is to alternate between updating the model parameters and updating the group weights z, where groups with higher loss are assigned larger weights. This inherently assumes that the loss is a reliable scalar signal that accurately reflects the group’s performance.

In the multi-task GRPO setting, however, this assumption no longer holds. The GRPO objective is based on probability-weighted advantages and can evaluate to zero both when all sampled responses are incorrect and when all sampled responses are correct. While this behavior is acceptable for updating the policy parameters θ, it introduces ambiguity when comparing across tasks for updating weights: a task on which the policy completely fails can appear indistinguishable (in terms of JGRPO) from a task on which the policy performs perfectly. In a multi-task setting, where task weights must be adapted based on task performance, such ambiguity can lead to systematically misleading updates. To our knowledge, this issue has not been discussed in prior works, as it arises specifically from the structure of GRPO-style objectives used in modern LLM post-training.

Subroutine 1 Improvement-aware Weight Update (IWU)

- 1: Input: rewards {Jk(θt)}Kk=1, improvements {Ik(t)}Kk=1, logits ξt, stepsize β, trade-off λ
- 2: zt ← Softmax(ξt)
- 3: s(kt) ← Ik(t) + λJk(θt) ∀k ∈ [K]
- 4: (gt)k ← zk,t s(kt) − Kj=1 zj,ts(jt) ∀k ∈ [K]
- 5: ξt+1 ← ξt − β gt
- 6: Return: zt+1 = Softmax(ξt+1)

To address this, our idea is to decouple task reweighting from the GRPO loss and instead use true task-level rewards Jk(θ) to update task weights. This design choice is specific to the post-training + GRPO setting and constitutes an important departure from existing robust learning methods.

Update rule: Given this observation and the subsequent design choice, to update task weights z ∈ ∆K, we define z as zt = Softmax(ξt) over logits ξ ∈ RK, and update ξ instead. This allows unconstrained optimization over ξ while ensuring zt ∈ ∆K. At iteration t, we update ξt through gradient descent

w.r.t. weighted task rewards L(ξt) = Kk=1 zk,tJk(θt) for fixed θt. The resulting task-wise gradient, (gt)k = zk,t Jk(θt) − Kj=1 zj,tJj(θt) , is negative for tasks whose rewards fall below the current weighted average, thereby increases their corresponding logits and weights. This yields the following alternating updates:

K

zkt∇θJGRPO,k(θt), (6)

θt+1 = θt + γt

k=1

  

  . (7)

  

  

∇ξz1⊤,t . ∇ξzK,t⊤

J1(θt) . JK(θt)

ξt+1 = ξt − β gt, gt =

These updates train the policy using a z-weighted GRPO loss across the K tasks while adaptively adjusting the weights z to prioritize underperforming ones, thereby encouraging more balanced performance across tasks.

Issues with strict worst-task reward maximization: While the alternating updates in Equation (6) correctly optimize the minimax objective in Equation (5) for ε = 0, they can lead to degenerate dynamics in which training is dominated by a single worst-performing task. This behavior is inherent to the inner problem in Equation (5) as for fixed θ, the inner optimization over weights z ∈ ∆K places all mass on the lowest-reward task for ε = 0.

Worst-task GRPO

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

1.0

0.8

Weight

0.6

0.4

0.2

0.0

With the softmax parameterization z = Softmax(ξ) and gradient descent on ξ, this tendency is further amplified by the exponential mapping. Tasks with lower rewards are repeatedly upweighted, causing z to rapidly collapse toward a near one-hot distribution that persists until another task becomes worse. This behavior is visible in Figure 2, where the weight assigned to the current worst task quickly spikes toward

0 100 200 300 400 500 600 700

ARC Countdown Zebra

- Figure 2: In strict worst-task optimization (ε = 0), task weights rapidly collapse to the current worst task and oscillate as the worst task shifts, resulting in near-zero weighting of Countdown.

one, while the other tasks receive near-zero weight for extended periods (e.g., Countdown is almost entirely ignored after the early steps). As a result, non-worst tasks are systematically under-optimized.

#### 3.2 Improvement-Aware Task Reweighting

Motivated by the above limitation, we consider two mechanisms to stabilize task reweighting: (i) incorporation of task-level improvement, and (ii) l2 regularization. We focus on (i) in the main text and analyze (ii) in Section E. Absolute reward does not distinguish between tasks that are improving rapidly and tasks that have stagnated during training. A task with low reward but strong improvement may require less prioritization than a task with similar reward that no longer benefits from updates. As a result, reward-based reweighting alone can leave some

Algorithm 1 MULTI-TASK GRPO (MT-GRPO)

- 1: Input: Dtrain = {D1,...,DK}, batch size B, rollouts per prompt N, reward objective J(·), initial policy parameters θ0, initial filtered ratios ρ0, initial task-weight logits ξ0 (z0 = Softmax(ξ0)), total steps T
- 2: for t = 0 to T − 1 do
- 3: Batch construction: (x,ρt+1) ∼ RP SAMPLER (zt,B,N,ρt,Dtrain)
- 4: Policy update: θt+1 ← OPTIMIZER θt,∇θJGRPO(θt;x)
- 5: Ik(t) ← JGRPO,k(θt+1) − JGRPO,k(θt) ∀k ∈ [K]
- 6: zt+1 = IWU(ξt,zt,I(t),J(θt)) (Subroutine 1)
- 7: end for
- 8: Return final policy parameters θT

tasks under-optimized. This motivates tracking how each task’s loss (i.e., −JGRPO,k(θ)) evolves over training and introducing an improvement-aware signal that captures how much each task benefits from policy updates.

Task-level improvement: Building on the notion of task-level improvement introduced in prior work Liu et al. [11], we define the per-step improvement of task k for multi-task post-training using GRPO as

Ik(t) := JGRPO,k(θt+1) − JGRPO,k(θt). (8) This quantity captures whether the task-wise GRPO loss is improving, stagnating, or degrading. Moreover, it provides a measure of how much each task benefits from the policy update θt+1 = θt+γt Kk=1 ∇θJGRPO,k(θt). Improvement-aware reweighting: Our goal is to prioritize tasks that are both underperforming in terms of Jk(θt) and under-improving in terms of Ik(t). We therefore update task-weight logits using the combined signal Ik(t) + λJk(θt), where λ controls the trade-off between task reward and task improvement. For large λ, the update approaches strict worst-task reward maximization, and for small λ, it promotes balanced improvement across tasks. We present these improvement-aware updates in Subroutine 1 (see Lines 4, 5).

Intuitively, Subroutine 1 accounts for task stagnation and deterioration when updating weights, rather than repeatedly upweighting tasks with lower rewards. This prevents collapse onto a single task and promotes balanced progress across tasks (see Section B.3 for a formal analysis).

### 4 Algorithm

We present MULTI-TASK GRPO (MT-GRPO), a novel post-training algorithm for improving reasoning across multiple tasks. Our method addresses key limitations (see Section 2) that arise when adapting GRPO-style RL post-training to the multi-task setting, and is summarized in Algorithm 1. MT-GRPO jointly learns (i) the policy parameters and (ii) a distribution over tasks that governs how prompts are sampled during training. This distribution is dynamically updated to balance robustness and avg. performance (Limitation (i)), guided by task-level rewards and task-wise improvement signals (IWU in Subroutine 1). Moreover, MT-GRPO ensures consistency between learned task weights and effective gradient contributions (Limitation (ii)), by using RATIO-PRESERVING SAMPLER (RP SAMPLER) for batch construction.

Adaptive task reweighting (IWU): At each step, MT-GRPO updates task weights zt ∈ ∆K that control how prompts are sampled across tasks. We update these weights using a λ weighted combination of task

reward Jk(θt) and task improvement Ik(t) (Subroutine 1), where λ plays a role analogous to ε in Equation (5): larger λ emphasizes worst-task robustness, while smaller λ favors average performance. This prioritizes tasks

that are underperforming or improving slowly by increasing their sampling frequency in subsequent batches. The improvement-aware update also prevents weight collapse onto a single worst task and the consequent under-optimization of other tasks. As a result, MT-GRPO achieves strong worst-task without sacrificing average performance, consistent with our objective in Equation (5).

Uneven zero-gradient rates (RP SAMPLER): In practice, sampling prompts according to task weights zt is insufficient under GRPO because many prompts yield zero gradients. Since tasks might exhibit widely different zero-gradient rates, the effective composition of the training batch can deviate from the intended task proportions, leading to a mismatch between the task weights produced by Subroutine 1 and the actual gradient contributions. To address this, MT-GRPO uses a Ratio-Preserving Sampler (RP sampler) for batch construction

(Algorithm 2). The RP sampler enforces the target task proportions in the post-filtered batch (after zero gradient prompts filtered) using oversampling and acceptance-aware resampling. We detail this procedure in Section 5.

Together, the task weights influence how data are sampled across tasks, and the resulting reward and improvement metrics in turn affect subsequent weight updates, forming an effective loop that ultimately drives MT-GRPO’s reliable performance across tasks.

### 5 Practical Findings and Solutions

While the task-weight updates in Subroutine 1 provide a principled mechanism for balancing performance across tasks, their efficacy depends on whether these weights translate into actual gradient contributions during training (see limitation (ii) in Section 2). In this section, we discuss the underlying mechanism in RP SAMPLER (Algorithm 2), a key component of Algorithm 1, that addresses this issue and ensures faithful realization of the intended task weights.

Filtered Ratios Across Tasks

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.8

0.6

0.4

0.2

0.0

0 100 200 300 400 500 600 700

ARC Countdown Zebra

Uneven Zero Gradient Rates: A structural limitation of GRPO is that when all sampled responses for a prompt receive identical rewards, the resulting advantages are zero and the gradient vanishes for that prompt [3, 23]. The prevalence of such zerogradient samples varies substantially across tasks (see

Figure 3: Ratios of zero-gradient prompts across tasks observed during training. ARC exhibits a substantially higher proportion of zero-gradient prompts than Zebra.

Figure 3). Consequently, in a multi-task setting, even if two tasks are assigned equal weight zk, the task producing informative gradients more frequently will contribute disproportionately to parameter updates. This systematically skews the training mixture away from the intended task proportions.

Fix: Enforcing target task ratios To correct this mismatch, we explicitly enforce task proportions (in the postfiltered batch) after filtering out zero-gradient prompts. Let z ∈ ∆K denote task weights from Subroutine 2 and B the target batch size. We first sample desired post-filtered counts via (n1,...,nK) ∼ Multinomial(B,z), where nk specifies the number of informative samples from task k in the final batch. We then generate and filter samples, tracking non-zero-gradient samples ck per task. We resample prompts until ck ≥ nk or a fixed regeneration budget is exhausted.

Inefficiency Under High Filtering Rates: While the above procedure ensures correctness, it can be inefficient when some tasks exhibit high filtering rates, requiring many regeneration rounds to meet post-filtered targets.

Fix: Acceptance-aware sampling. To reduce regeneration overhead, we introduce an acceptance-aware sampling strategy that anticipates task-dependent filtering. We maintain an estimate ρk of the filtering rate (fraction of generated samples with zero gradients) for each task k. During sampling, we inflate task weights as zˆk = z

k mk K

, Macc , where Macc caps the inflation factor. Tasks with higher expected filtering are oversampled during generation, increasing the likelihood that the post-filtered batch matches the desired proportions. During resampling, we similarly prioritize tasks based on their deficiency (ck relative to nk) and expected acceptance. This strategy reduces regeneration overhead while preserving consistency with the task proportions induced by Subroutine 1.

j=1 zj mj , mk = min 1− 1ρ

k

### 6 Experiments

We evaluate MT-GRPO (Algorithm 1) in a multi-task RL post-training setting on reasoning tasks spanning planning and inductive reasoning. We post-train Qwen-2.5-3B base model on three task families from Chen et al. [4].

Planning: (i) Countdown: Given 3–5 integers, the model applies arithmetic operations to reach a target value. (ii) Zebra puzzles: Logic puzzles over 3–5 entities and properties with textual constraints to infer the correct assignment. In both cases, difficulty increases with number of input values.

Inductive reasoning: Abstraction and Reasoning Corpus (ARC): Each instance contains 3 input–output examples illustrating a transformation rule, and the model must generalize to a test example. We use string-based ARC tasks of lengths 10, 20, and 30, with length determining difficulty.

###### Worst-Task Accuracy

###### Avg Accuracy

###### Avg Relative Change

70

| |Zebra Zebra 58.8|
|---|---|
| |ARC<br><br>Zebra 52.6<br><br>57.5|
| |45.8|
| |ARC<br><br>ARC 30.3|
| |24.6|

| | |
|---|---|
| |65.2<br><br>67.2 66.9|
| |61.8|
| | |
| |51.0<br><br>52.0|
| | |
| | |

| |+5.8 +5.1|
|---|---|
| |+0.0|
| | |
| |-5.5|
| | |
| | |
| |-22.3|
| |-24.6|

5

60

65

0

60

Accuracy

50

5

55

10

40

15

50

30

20

45

25

20

40

SEC_GRPO GRPO SEC_DAPO DAPO MT-GRPO_0.2 MT-GRPO_0.25

- Figure 4: Experiment 1: MT-GRPO substantially outperforms all baselines in terms of worst-task accuracy by 6% or more without conceding on average accuracy. Moreover, it achieves higher average per-task relative change, reflecting stronger improvements on weaker tasks.

Datasets. We use datasets released by Chen et al. [4] which were generated using the ReasoningGym framework Stojanovski et al. [24]. Each task family includes three difficulty levels (easy, medium, hard), with 10k training instances and 200 evaluation instances per level.

Baselines: We compare against four competitive baselines: (i) GRPO: Uniform sampling over tasks when constructing training batches. (ii) SEC-GRPO: Self-evolving curriculum [4] prioritizing tasks with larger absolute advantages. (iii) DAPO: Uniform sampling with DAPO-style clipping and dynamic sampling [3]. (iv) SEC-DAPO: SEC-style weighting combined with DAPO.

Metrics: Our primary metric is worst-task accuracy (minimum accuracy across tasks), which reflects robustness. To assess the robustness vs. overall performance trade-off, we also report average accuracy and average per-task relative change ([11, 25], which normalizes gains across heterogeneous task scales:

∆m% = K1 Kk=1 Acc

m(k)−Accb(k)

Accb(k) ×100, where b denotes the DAPO baseline. This normalization emphasizes

gains on lower-performing tasks. Full training details are provided in Section D. 6.1 Experiment 1: Controlled Three-Task Setting We evaluate MT-GRPO in a controlled multi-task setting by post-training Qwen-2.5-3B on three mediumdifficulty tasks (Countdown, Zebra, ARC) and study how adaptive task reweighting affects training dynamics. Main results: Figure 4 summarizes the performance across all methods. MT-GRPO achieves substantially higher worst-task accuracy than all baselines for both λ = 0.2 and λ = 0.25, while simultaneously improving average accuracy. These gains arise because MT-GRPO reallocates optimization effort away from the highperforming Countdown task toward weaker tasks. In addition, MT-GRPO attains the highest average per-task relative change, indicating more balanced improvements across tasks. Consistent with the design in Subroutine 1, increasing λ strengthens worst-task performance. λ = 0.25 yields higher worst-task accuracy than λ = 0.2, while λ = 0.2 achieves better average accuracy.

Weight dynamics: Figure 6 illustrates how MT-GRPO achieves these gains. Early in training, Zebra outperforms Countdown, but this reverses after approximately 50 steps. MT-GRPO responds by reallocating weight toward Zebra and reducing emphasis on Countdown, whereas DAPO and SEC-DAPO continue to prioritize Countdown even after it attains high performance. As a result, these baselines underperform on Zebra or ARC and achieve little additional improvement on Countdown, leading to poorer worst-task performance and lower average accuracy.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

30 40 50

0

200

400

600

Steps to reach (%) worst-task accuracy

GRPO

SEC_GRPO

SEC_DAPO

DAPO

MT-GRPO_0.2

MT-GRPO_0.25

- Figure 5: MT-GRPO reaches target worst-task accuracy thresholds substantially faster (50% fewer training steps) than baselines; striped bars indicate the threshold was not reached.

Ratio Preservation: Tasks with high prevalence of zero-gradient samples (e.g., ARC; Figure 3) tend to be substantially underrepresented in training batches relative to their intended weights (Figure 6). Our proposed RP SAMPLER (Algorithm 2) ensures that realized batch proportions closely track the learned task weights from Subroutine 1 and plays a critical role in the robust performance of Algorithm 1.

Faster robustness gains: Figure 5 reports the number of training steps required to reach specified worst-task accuracy thresholds. MT-GRPO consistently reaches these thresholds in fewer

###### Zebra Accuracy

###### Countdown Accuracy

###### ARC Accuracy

- 0.3

- 0.4

- 0.5

- 0.6

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.6

0.8

Accuracy

0.6

0.4

0.4

0.2

0.2

0.0

0.0

- 0.0
- 0.1
- 0.2
- 0.3
- 0.4
- 0.5

Weights/Ratio

Countdown Weights

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 100 200 300 400 500 600 700

0.0

0.2

0.4

0.6

ARC Weights

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0 100 200 300 400 500 600 700

0.0

0.2

0.4

0.6

ARC Batch Ratio

Steps

SEC_GRPO GRPO SEC_DAPO DAPO MT-GRPO_0.2 MT-GRPO_0.25

Figure 6: Experiment 1: Top plots: Task-wise accuracies. Bottom plots: MT-GRPO reallocates weights toward the under-performing tasks. In contrast, baselines continue to prioritize high-performing Countdown, leading to weaker performance on Zebra or ARC and only marginal gains on Countdown. ARC is typically underrepresented in training batches relative to its task weight (middle vs right). RP SAMPLER ensures alignment of realized batch proportions with intended task weights and facilitates higher ARC performance.

| |Zebra|
|---|---|
| |ARC Zebra<br><br>Zebra hard<br><br>Zebra hard<br><br>44.3<br><br>hard<br><br>46.3|
| |hard<br><br>40.5<br><br>hard<br><br>40.9<br><br>Zebra hard<br><br>40.4<br><br>42.1|
| |ARC hard<br><br>34.3|
| |ARC hard<br><br>30.4|
| | |

| |+0.0|
|---|---|
| |-2.7|
| |-4.1<br><br>-6.8|
| |-11.9<br><br>-9.6<br><br>-12.7|
| |-15.3|

30

35

40

45

50

Accuracy

Worst-Task Accuracy

| |68.7 69.5|
|---|---|
| |66.7 66.0<br><br>63.9|
| |59.2<br><br>61.7|
| |56.0|
| | |
| | |
| | |

45

50

55

60

65

70

Avg Accuracy

| |+0.0<br><br>+1.7|
|---|---|
| |-2.8|
| |-3.9<br><br>-5.4<br><br>-7.7|
| | |
| |-15.7|
| |-20.7|

20

15

10

5

0

Avg Relative Change

15

10

5

0

RelativeChange(%)

Avg Relative Change (Easy)

| | |
|---|---|
| |+0.0<br><br>+2.4|
| |-1.8 -2.6|
| |-4.6<br><br>-8.6|
| | |
| |-14.9|
| |-21.2|

25

20

15

10

5

0

5

Avg Relative Change (Medium)

| |+5.4|
|---|---|
| |+0.0<br><br>+1.1|
| |-2.0 -1.9|
| |-6.0|
| | |
| | |
| |-20.2|
| |-25.6|

25

20

15

10

5

0

5

Avg Relative Change (Hard)

SEC_GRPO

GRPO

SEC_DAPO

DAPO

MT-GRPO_0.1 MT-GRPO_0.3

- MT-GRPO_0.9

- MT-GRPO_1.2

Figure 7: Experiment 2: Top plots: Increasing λ improves worst-task accuracy of MT-GRPO but reduces average accuracy, highlighting a trade-off controlled by λ. λ = 0.1 yields the highest average per-task relative change, showing stronger gains on weaker tasks. Bottom plots: For smaller λ (0.1,0.3), MT-GRPO prioritizes slower-improving tasks, yielding larger gains on hard tasks while sacrificing easy ones. For larger λ, gains concentrate on the lowest-performing task, increasing worst-task accuracy but reducing average relative change.

steps than all baselines. In several cases, baselines fail to reach the target thresholds within the training budget as marked in full length striped bars. This shows that beyond improving final worst-task accuracy, MT-GRPO also accelerates progress on the weakest tasks.

- 6.2 Experiment 2: Scaling to Nine Tasks

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 100 200 300 400 500 600 700

We evaluate whether the gains from Experiment 1 (Section 6.1) persist in a larger multi-task setting with increased task diversity by post-training Qwen-2.5-3B on nine tasks corresponding to the easy, medium, and hard variants of Countdown, Zebra, and ARC.

Main results: Figure 7 summarizes aggregate performance across methods for different values of the trade-off parameter λ. Increasing λ consistently improves worst-task accuracy (by 16% over GRPO and 6% over DAPO at λ = 1.2) but reduces average accuracy, demonstrating how the trade-off encoded in our objective

(Equation (5)) is operationalized by the task-reweighting mechanism in Algorithm 1 and subroutine 1. These results confirm that λ provides effective control over this trade-off in practice.

Difficulty-wise analysis: Figure 7 (bottom plots) reports average per-task relative change by difficulty. For smaller λ, MT-GRPO exhibits negative relative change on easy tasks but positive relative change on hard tasks, indicating a reallocation of optimization effort toward more challenging tasks. This is expected because smaller λ places greater emphasis on improvement (Subroutine 1), and harder tasks typically exhibit slower learning progress. This also explains the higher overall average per-task relative change observed for smaller λ, as this metric is more sensitive to improvements on weaker tasks than on high-performing ones.

For larger values of λ, performance gains become increasingly concentrated on the worst-performing task (Zebra-hard), improving worst-task accuracy but reducing average relative change. Overall, these trends are consistent with the intended role of λ, where larger values prioritize worst-task improvement while smaller values promote more balanced progress across all tasks.

Across both experiments, MT-GRPO consistently improves worst-task performance while maintaining competitive average accuracy. In the controlled setting (Experiment 1), the analysis of weight dynamics shows that these gains arise from adaptive reallocation of optimization effort toward weaker tasks. In the larger and more diverse setting (Experiment 2), we show that MT-GRPO scales to more heterogeneous task collections and that the trade-off parameter λ provides meaningful and effective control over the balance between worst-task robustness and overall performance. Together, these results validate both the empirical effectiveness of MT-GRPO and the intended behavior of its underlying optimization mechanism.

### 7 Related Work

Recent post-training has produced strong results on individual reasoning benchmarks, but typical pipelines still optimize tasks largely in isolation, producing specialist models rather than balanced competence across heterogeneous reasoning skills. Naive multi-task post-training can further suffer from task interference and negative transfer, motivating robustness-aware and multi-objective perspectives for improving worst-task behavior without sacrificing overall performance. The related work below summarizes (i) robustness objectives and distributional robustness, (ii) multi-task optimization methods for mitigating gradient conflict, and (iii) multi-task LLM training and RL post-training (including GRPO) in both single- and multi-task settings.

Robust objectives and distributional robustness. Robust learning objectives that balance average performance with guarantees on underperforming groups or domains have a long history in distributionally robust optimization (DRO) [8, 9]. Beyond worst-case group robustness, DRO is also closely connected to risk-sensitive objectives that trade off mean performance and variability [26], and recent work has developed non-asymptotic theory for DRO in modern non-convex regimes [27] as well as generalized formulations such as kernel DRO [28]. In large-scale language model training, robustness over mixed data has been pursued via domain reweighting and mixture optimization methods [12–16, 29], as well as gradient-aware approaches for improved alignment across domains [30, 31]. These efforts primarily target pre-training or supervised objectives over heterogeneous corpora, and do not directly instantiate robustness-aware objectives within RL post-training for LLMs, where the optimization signal is mediated by task-dependent rewards and on-policy sampling.

Multi-task optimization and gradient conflict methods. A central difficulty in multi-task learning is that task gradients may conflict or exhibit large magnitude disparities, leading to updates that degrade some tasks despite improving others [7]. This has motivated recent works on multi-objective and multi-task optimization, including classical multiple-gradient descent methods [10] and their deep learning instantiations that cast MTL as multi-objective optimization [32], task balancing via gradient normalization [33], conflict-aware or constrained updates [34], gradient manipulation methods [35–38], and game-theoretic or bargaining-style formulations [25]. A complementary line of work targets negative transfer via geometry-aware gradient balancing and homogenization [39–41]. Other approaches emphasize task weighting rules (e.g., uncertainty-based loss weighting) as a lightweight mechanism for balancing objectives [42], while objectives that encourage progress on the worst-improving task provide another robustness-inspired alternative [11]. Complementary results characterize when scalarization may fail to recover the full Pareto front [43], and analyze convergence issues of stochastic multi-objective methods along with stabilizing schemes [44, 45]. However, these approaches are typically studied in supervised multi-task settings and do not address RL post-training peculiarities such as taskdependent zero-gradient rates and the limited reliability of RL losses as direct proxies for task-level performance.

Multi-task fine-tuning and RL post-training Multi-task supervised fine-tuning is widely used to share statistical strength across related tasks, particularly when some tasks are data-limited; for example, [46] studies multi-task fine-tuning for coding, and broader multi-task learning frameworks have been explored in related settings [47–55]. RL fine-tuning of LLMs has been extensively used for preference alignment [56–59], for reasoning improvements with task rewards [1], and for self-training with internal supervision [60]; more recently, GRPO-based post-training has shown strong reasoning performance [61, 2, 62–67]. A few works analyze and adapt GRPO to the multi-reward setting [68–70]. Only a few works study multi-task RL post-training directly, including mixture selection for higher average performance [5, 71], cross dataset reward normalization to balance reward scales across datasets [72], applying GRPO to a curated set of temporal tasks, text image translation skills [73, 74], sequential pipelines and task ordering to mitigate forgetting [75, 76], discussion about utility of meta-reasoning frameworks [77], and analyses of gradient imbalance and the limits of gradient-based curricula [78]. Curriculum sampling approaches primarily adjust task/difficulty sampling to improve efficiency or average outcomes [79–81, 4], while transfer analyses evaluate whether gains generalize beyond the training domain [82]. In contrast, our work incorporates task-wise robustness directly into the multi-task RL objective under GRPO, and introduces mechanisms such as adaptive reweighting and ratio-preserving batch construction to improve worst-task performance while maintaining strong overall performance.

### 8 Conclusion

We introduced MULTI-TASK GRPO (MT-GRPO), a robustness-aware post-training algorithm for improving LLM reasoning across tasks. MT-GRPO performs improvement-aware task reweighting to promote balanced progress across tasks, and ratio-preserving sampling to ensure task weights translate into actual gradient contributions. Our experiments demonstrate that MT-GRPO consistently improves worst-task performance while maintaining competitive average accuracy. These results suggest that explicitly optimizing for task-wise robustness is practical and beneficial for building general-purpose reasoning models.

### 9 Acknowledgments

Ilija Bogunovic was supported by the ESPRC New Investigator Award EP/X03917X/1. Sangwoong Yoon was supported by the Center for Advanced Computation at Korea Institute for Advanced Study; Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2020-II201336, Artificial Intelligence Graduate School Program (UNIST))

### References

- [1] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [2] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [3] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [4] Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piché, Nicolas Gontier, Yoshua Bengio, and Ehsan Kamalloo. Self-evolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970, 2025.
- [5] Syeda Nahida Akter, Shrimai Prabhumoye, Matvei Novikov, Seungju Han, Ying Lin, Evelina Bakhturina, Eric Nyberg, Yejin Choi, Mostofa Patwary, Mohammad Shoeybi, et al. Nemotron-crossthink: Scaling self-learning beyond math reasoning. arXiv preprint arXiv:2504.13941, 2025.
- [6] Sen Wu, Hongyang R Zhang, and Christopher Ré. Understanding and improving information transfer in multi-task learning. arXiv preprint arXiv:2005.00944, 2020.
- [7] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in neural information processing systems, 33:5824–5836, 2020.

- [8] Hongseok Namkoong and John C Duchi. Stochastic gradient methods for distributionally robust optimization with f-divergences. Advances in neural information processing systems, 29, 2016.
- [9] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731, 2019.
- [10] Jean-Antoine Désidéri. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. Comptes Rendus Mathematique, 350(5-6):313–318, 2012.
- [11] Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu. Famo: Fast adaptive multitask optimization. Advances in Neural Information Processing Systems, 36:57226–57243, 2023.
- [12] Yonatan Oren, Shiori Sagawa, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust language modeling. arXiv preprint arXiv:1909.02060, 2019.
- [13] Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. Doremi: Optimizing data mixtures speeds up language model pretraining. arXiv preprint arXiv:2305.10429, 2023.
- [14] Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, and Min Lin. Regmix: Data mixture as regression for language model pre-training. arXiv preprint arXiv:2407.01492, 2024.
- [15] David Grangier, Simin Fan, Skyler Seto, and Pierre Ablin. Task-adaptive pretrained language models via clustered-importance sampling. arXiv preprint arXiv:2410.03735, 2024.
- [16] Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, Zijia Chen, Peter Belcak, Yoshi Suhara, Hongxu Yin, Mostofa Patwary, Yingyan, Lin, Jan Kautz, and Pavlo Molchanov. Nemotron-climb: Clustering-based iterative data mixture bootstrapping for language model pre-training. arXiv preprint arXiv:2504.13161, 2025. URL https://arxiv.org/abs/2504.13161.
- [17] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.
- [18] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [19] Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Refining credit assignment in rl training of llms. arXiv preprint arXiv:2410.01679, 2024.
- [20] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.
- [21] Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025.
- [22] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Heung-Yeung Shum, and Xiangyu Zhang. Openreasoner-zero: An open source approach to scaling reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025. URL https://arxiv.org/abs/2503.24290.
- [23] Thomas Foster, Anya Sims, Johannes Forkel, Mattie Fellows, and Jakob Foerster. Learning to reason at the frontier of learnability. arXiv preprint arXiv:2502.12272, 2025.
- [24] Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, and Andreas Köpf. Reasoning gym: Reasoning environments for reinforcement learning with verifiable rewards. arXiv preprint arXiv:2505.24760, 2025. URL https://arxiv.org/abs/2505.24760.
- [25] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. In International Conference on Machine Learning, pages 16109–16128. PMLR, 2022.

- [26] Jun-ya Gotoh, Michael Jong Kim, and Andrew EB Lim. Robust empirical optimization is almost the same as mean–variance optimization. Operations research letters, 46(4):448–452, 2018.
- [27] Jikai Jin, Bohang Zhang, Haiyang Wang, and Liwei Wang. Non-convex distributionally robust optimization: Non-asymptotic analysis. Advances in Neural Information Processing Systems, 34:2771–2782, 2021.
- [28] Jia-Jie Zhu, Wittawat Jitkrittum, Moritz Diehl, and Bernhard Schölkopf. Kernel distributionally robust optimization: Generalized duality theorem and stochastic approximation. In International Conference on Artificial Intelligence and Statistics, pages 280–288. PMLR, 2021.
- [29] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.
- [30] Simin Fan, Matteo Pagliardini, and Martin Jaggi. Doge: Domain reweighting with generalization estimation. arXiv preprint arXiv:2310.15393, 2023.
- [31] Simin Fan, Maria Ios Glarou, and Martin Jaggi. Grape: Optimize data mixture for group robust multi-target adaptive pretraining. arXiv preprint arXiv:2505.20380, 2025.
- [32] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Advances in Neural Information Processing Systems, volume 31, 2018.
- [33] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning, pages 794–803. PMLR, 2018.
- [34] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878–18890, 2021.
- [35] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems, 33:2039–2050, 2020.
- [36] Vitaly Kurin, Alessandro De Palma, Ilya Kostrikov, Shimon Whiteson, and Pawan K Mudigonda. In defense of the unitary scalarization for deep multi-task learning. Advances in Neural Information Processing Systems, 35:12169–12183, 2022.
- [37] Shikun Liu, Stephen James, Andrew J Davison, and Edward Johns. Auto-lambda: Disentangling dynamic task relationships. arXiv preprint arXiv:2202.03091, 2022.
- [38] Shijie Zhu, Hui Zhao, Tianshu Wu, Pengjie Wang, Hongbo Deng, Jian Xu, and Bo Zheng. Gradient deconfliction via orthogonal projections onto subspaces for multi-task learning. In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining, pages 204–212, 2025.
- [39] Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In International conference on learning representations, 2021.
- [40] Adrián Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. arXiv preprint arXiv:2103.02631, 2021.
- [41] Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. arXiv preprint arXiv:2010.05874, 2020.
- [42] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491, 2018.
- [43] Yuzheng Hu, Ruicheng Xian, Qilong Wu, Qiuling Fan, Lang Yin, and Han Zhao. Revisiting scalarization in multi-task learning: A theoretical perspective. Advances in Neural Information Processing Systems, 36: 48510–48533, 2023.

- [44] Zheyuan Zhou, Guojun Li, Xuanyuan Zhang, Zhe Chen, and Yang You. On the convergence of stochastic multi-objective gradient manipulation. In International Conference on Machine Learning, pages 27192–

27214. PMLR, 2022.

- [45] Peiyao Xiao, Hao Ban, and Kaiyi Ji. Direction-oriented multi-objective learning: Simple and provable stochastic algorithms. Advances in Neural Information Processing Systems, 36:4509–4533, 2023.
- [46] Bingchang Liu, Chaoyu Chen, Zi Gong, Cong Liao, Huan Wang, Zhichao Lei, Ming Liang, Dajun Chen, Min Shen, Hailian Zhou, et al. Mftcoder: Boosting code llms with multitask fine-tuning. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 5430–5441, 2024.
- [47] Thomas T Zhang, Katie Kang, Bruce D Lee, Claire Tomlin, Sergey Levine, Stephen Tu, and Nikolai Matni. Multi-task imitation learning for linear dynamical systems. In Learning for Dynamics and Control Conference, pages 586–599. PMLR, 2023.
- [48] Simen Eide and Arnoldo Frigessi. Bora: Bayesian hierarchical low-rank adaption for multi-task large language models. arXiv preprint arXiv:2407.15857, 2024.
- [49] Zi Gong, Hang Yu, Cong Liao, Bingchang Liu, Chaoyu Chen, and Jianguo Li. Coba: convergence balancer for multitask finetuning of large language models. arXiv preprint arXiv:2410.06741, 2024.
- [50] Kaiwen Wang, Rahul Kidambi, Ryan Sullivan, Alekh Agarwal, Christoph Dann, Andrea Michi, Marco Gelmi, Yunxuan Li, Raghav Gupta, Kumar Avinava Dubey, et al. Conditional language policy: A general framework for steerable multi-objective finetuning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 2153–2186, 2024.
- [51] Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, and Hao Wang. Mixture-of-loras: An efficient multitask tuning for large language models. arXiv preprint arXiv:2403.03432, 2024.
- [52] Zhen Qi, Jiajing Chen, Shuo Wang, Bingying Liu, Hongye Zheng, and Chihang Wang. Optimizing multitask learning for enhanced performance in large language models. In 2024 4th International Conference on Electronic Information Engineering and Computer Communication (EIECC), pages 1179–1183. IEEE, 2024.
- [53] Meni Brief, Oded Ovadia, Gil Shenderovitz, Noga Ben Yoash, Rachel Lemberg, and Eitam Sheetrit. Mixing it up: The cocktail effect of multi-task fine-tuning on llm performance–a case study in finance. arXiv preprint arXiv:2410.01109, 2024.
- [54] Rongsheng Wang, Haoming Chen, Ruizhe Zhou, Yaofei Duan, Kunyan Cai, Han Ma, Jiaxi Cui, Jian Li, Patrick Cheong-Iao Pang, Yapeng Wang, et al. Aurora: Activating chinese chat capability for mixtral-8x7b sparse mixture-of-experts through instruction-tuning. arXiv preprint arXiv:2312.14557, 2023.
- [55] Tong Zhu, Daize Dong, Xiaoye Qu, Jiacheng Ruan, Wenliang Chen, and Yu Cheng. Dynamic data mixing maximizes instruction tuning for mixture-of-experts. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1663–1677, 2025.
- [56] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.
- [57] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.
- [58] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.
- [59] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

- [60] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [61] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.
- [62] Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.
- [63] Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.
- [64] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
- [65] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [66] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.
- [67] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [68] Yicheng Zhong, Peiji Yang, and Zhisheng Wang. Multi-reward grpo for stable and prosodic singlecodebook tts llms at scale. arXiv preprint arXiv:2511.21270, 2025.
- [69] Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, et al. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization. arXiv preprint arXiv:2601.05242, 2026.
- [70] Yining Lu, Zilong Wang, Shiyang Li, Xin Liu, Changlong Yu, Qingyu Yin, Zhan Shi, Zixuan Zhang, and Meng Jiang. Learning to optimize multi-objective alignment through dynamic reward weighting. arXiv preprint arXiv:2509.11452, 2025.
- [71] Yiqing Liang, Jielin Qiu, Wenhao Ding, Zuxin Liu, James Tompkin, Mengdi Xu, Mengzhou Xia, Zhengzhong Tu, Laixi Shi, and Jiacheng Zhu. Modomodo: Multi-domain data mixtures for multimodal llm reinforcement learning. arXiv preprint arXiv:2505.24871, 2025.
- [72] Yuhao Su, Anwesa Choudhuri, Zhongpai Gao, Benjamin Planche, Van Nguyen Nguyen, Meng Zheng, Yuhan Shen, Arun Innanje, Terrence Chen, Ehsan Elhamifar, et al. Medgrpo: Multi-task reinforcement learning for heterogeneous medical video understanding. arXiv preprint arXiv:2512.06581, 2025.
- [73] Tao Wu, Li Yang, Gen Zhan, Yabin Zhang, Yiting Liao, Junlin Li, Deliang Fu, Li Zhang, and Limin Wang. Tempr1: Improving temporal understanding of mllms via temporal-aware multi-task reinforcement learning. arXiv preprint arXiv:2512.03963, 2025.
- [74] Zhaopeng Feng, Yupu Liang, Shaosheng Cao, Jiayuan Su, Jiahan Ren, Zhe Xu, Yao Hu, Wenxuan Huang, Jian Wu, and Zuozhu Liu. Mt3: Scaling mllm-based text image machine translation via multi-task reinforcement learning. arXiv preprint arXiv:2505.19714, 2025. URL https://arxiv.org/abs/2505. 19714.
- [75] Bo Pang, Deqian Kong, Silvio Savarese, Caiming Xiong, and Yingbo Zhou. Reasoning curriculum: Bootstrapping broad llm reasoning from math. arXiv preprint arXiv:2510.26143, 2025.
- [76] Derek Li, Jiaming Zhou, Leo Maxime Brunswic, Abbas Ghaddar, Qianyi Sun, Liheng Ma, Yu Luo, Dong Li, Mark Coates, Jianye Hao, et al. Omni-thinker: Scaling multi-task rl in llms with hybrid reward and task scheduling. arXiv preprint arXiv:2507.14783, 2025.

- [77] Hanqi Yan, Linhai Zhang, Jiazheng Li, Zhenyi Shen, and Yulan He. Position: Llms need a bayesian meta-reasoning framework for more robust and generalizable reasoning. In 2025 International Conference on Machine Learning: ICML25, 2025.
- [78] Runzhe Wu, Ankur Samanta, Ayush Jain, Scott Fujimoto, Jeongyeol Kwon, Ben Kretzu, Youliang Yu, Kaveh Hassani, Boris Vidolov, and Yonathan Efroni. Imbalanced gradients in rl post-training of multi-task llms. arXiv preprint arXiv:2510.19178, 2025.
- [79] Zhenting Wang, Guofeng Cui, Kun Wan, and Wentian Zhao. Dump: Automated distribution-level curriculum learning for rl-based llm post-training. arXiv preprint arXiv:2504.09710, 2025.
- [80] Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, and Qianchun Lu. Learning like humans: Advancing llm reasoning capabilities via adaptive difficulty curriculum learning and expert-guided self-reformulation. arXiv preprint arXiv:2505.08364, 2025.
- [81] Shubham Parashar, Shurui Gui, Xiner Li, Hongyi Ling, Sushil Vemuri, Blake Olson, Eric Li, Yu Zhang, James Caverlee, Dileep Kalathil, et al. Curriculum reinforcement learning from easy to hard tasks improves llm reasoning. arXiv preprint arXiv:2506.06632, 2025.
- [82] Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, and Xiang Yue. Does math reasoning improve general llm capabilities? understanding transferability of llm reasoning. arXiv preprint arXiv:2507.00432, 2025.
- [83] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- [84] Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github. io/blog/qwen2.5/.
- [85] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.
- [86] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.
- [87] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

### A Background

This section provides additional background on KL-regularized policy optimization and the practical GRPO objective used in our experiments.

KL-regularized objective. Given a dataset of prompts x ∼ D, a reward function R(x,y), and a reference policy πref, the KL-regularized objective for a policy πθ is

θ(·|x)[R(x,y)] − β DKL(πθ(· | x)∥πref(· | x)) , (9) where β > 0 controls the strength of regularization. Policy gradient. The gradient of the KL-regularized objective can be written using the policy gradient theorem:

J(θ) = Ex∼D Ey∼π

θ(y|x)

θ ∇θ log πθ(y | x) R(x,y) − β log π

πref(y|x) . (10)

∇θJ(θ) = Ex,y∼π

In practice, this gradient is estimated using Monte Carlo samples from the policy, and its variance is reduced by replacing rewards with advantage estimates.

GRPO objective (prompt-level form). Group Relative Policy Optimization (GRPO) samples, for each prompt x, a group of G responses {yi}Gi=1 from the old policy πθ

. A prompt-level relative advantage is computed via within-group normalization:

old

R(x,yi) − mean({R(x,yj)}Gj=1 std({R(x,yj)}Gj=1)

A(x,yi) = A(x,yi) =

Using importance sampling, the GRPO objective to update the current policy πθ is

(11)

JGRPO(θ) = Ex E{y

i}∼πθold

G

1 G

θ(yi|x)

ρi A(x,yi) − β log π

πref(yi|x) , (12)

i=1

where ρi = πθ(yi | x)/πθ

(yi | x) is the importance ratio.

old

Token-level formulation with clipping. In practice, optimization is performed at the token level. Let yi,t denote the t-th token of response yi, and define

πθ(yi,t | x,yi,<t) πθ

.

ri,t =

(yi,t | x,yi,<t)

old

We set the token-level advantages Aˆi,t, as the same end of output normalized rewards,

R(x,yi) − mean({R(x,yj)}Gj=1 std({R(x,yj)}Gj=1)

A(x,yi) =

.

In order to avoid instability, the importance ratios are usually clipped obtaining the below clipped GRPO objective:

|yi|

G

1 G

1 |yi|

min ri,tAˆi,t, clip(ri,t,1−ε,1+ε)Aˆi,t −β ri,t fKL ππref(yi,t|x,yi,<t)

JGRPO(θ) = Ex,{y

θ(yi,t|x,yi,<t) ,

i}

t=1

i=1

(13) where ε is the PPO-style clipping threshold and

fKL(u) = u − log u − 1. (14)

- B Proofs and Theoretical Insights In this section, we detail all the main derivations and proofs used in Section 2.

#### B.1 Closed form of induced regularizer Ω(z)

In this subsection, we derive the closed form expression for the induced regularizer Ω(z) starting from the original constrained optimization problem in Equation (4).

For the purposes of analysis, we note that the absolute-value constraint is equivalent to the pair of linear constraints Jk(θ) − Jj(θ) ≤ ε and Jj(θ) − Jk(θ) ≤ ε. Hence, we start from the simplified constrained problem

K

1 K

Jk(θ) s.t. Jk(θ) − Jj(θ) ≤ ε, ∀(k,j) ∈ [K] × [K]. (15)

max

θ∈Θ

k=1

Here, the constraints with k = j are trivial, but we keep the full indexing for notational convenience. Lagrangian Reformulation: We introduce dual variables µkj ≥ 0 for each constraint Jk(θ)−Jj(θ) ≤ ε. Then the equivalent Lagrangian reformulation is

K

1 K

L(θ,µ) =

Jk(θ) +

k=1

Collecting terms with respect to Jk(θ) yields

K

K

µkj Jk(θ) − Jj(θ) − ε . (16)

j=1

k=1

K

K

1 K

L(θ,µ) =

1 +

j=1

k=1

Note, each Jk(θ) is now weighted by a factor,

µkj −

K

K

µjk Jk(θ) − ε

j=1

k=1

K

µkj. (17)

j=1

1 + Kj=1 µkj − Kj=1 µjk K

, k ∈ [K]. (18) Rewriting Equation (17) with zk(µ), we have,

z˜k(µ) :=

K

K

L(θ,µ) =

z˜k(µ)Jk(θ) − ε

k=1

k=1

K

µkj. (19)

j=1

Here, note that Kk=1 z˜k(µ) = 1 for all µ:

K

k=1

This yields the saddle form,

1 K

z˜k(µ) =

K

1 +

k=1

K

max

min

z∈∆K

θ∈Θ

k=1

where Ω is the regularizer induced by eliminating µ.

K

K

K

µkj −

j=1

k=1

k=1

K

µjk = 1. (20)

j=1

zkJk(θ) − εΩ(z), (21)

Eliminating µ and defining Ω(z). The mapping µ  → z is not injective as multiple µ values can induce the same z. Hence, k,j µkj is not uniquely determined by z. We therefore define the induced regularizer as the minimum total dual mass among all µ that induce the given z:

Ω(z) := min

µ≥0

K

K

1 K

µkj : zk =

j=1

k=1

K

µkj −

1 +

j=1

K

µjk , ∀k . (22)

j=1

Let rk := Kj=1 µkj,ck := Kj=1 µjk and substituting it in the constraints of Equation (22), we have

rk − ck = Kzk − 1 =: dk, k ∈ [K]. (23)

Moreover, note that Kk=1 dk = K k zk−K = 0, using which the objective in Equation (22) can be written as

K

K

K

rk. (24)

µkj =

j=1

k=1

k=1

Thus, Equation (22) is equivalent to the minimum-flow problem

K

rk s.t. rk − ck = dk, ∀k. (25)

Ω(z) = min

µ≥0

k=1

Interpreting µkj as flow shipped from node k to node j on a complete directed graph, dk > 0 are supplies and dk < 0 are demands. Any feasible flow requires that the total flow is equal to or greater than the total supply:

K

K

K

rk ≥

µkj =

j=1

k=1

k=1

dk. (26)

(rk − ck) =

k: dk>0

k: dk>0

Conversely, since the graph is complete, the lower bound Equation (26) is tight as given the supplies required for each node dk, one can always adapt µ to change flow from supply nodes to demand nodes such that the total shipped flow equals the total supply (e.g., via a greedy matching construction). Hence the lower bound Equation (26) is tight and

Ω(z) =

k: dk>0

k>0 dk = 21 Kk=1 |dk|, so

Using k dk = 0, we have d

K

K

- 1

- 2

K 2

- 1

- 2

|dk| =

|Kzk − 1| =

Ω(z) =

k=1

k=1

dk. (27)

K

1 K

zk −

k=1

K 2

=

1 K

z −

1

. (28)

1

Hence, the induced regularizer Ω(z) penalizes deviations from uniform task weights via an ℓ1 distance.

- B.2 Update rule for strict worst task reward maximization Here, we detail the update rule derived for strict worst-task reward maximization in Equation (6).

Specifically, we parameterize the task weights z ∈ ∆K using a softmax over logits ξ ∈ RK, zt = Softmax(ξt), and perform gradient descent on the inner objective in Equation (5) (with ε = 0) for fixed θ w.r.t. ξt. At iteration

t, the inner objective is L(ξt) = Kk=1 zk,tJk(θt). Taking gradients with respect to ξt, we obtain

gt = ∇ξt

L(ξt) =

∂zt ∂ξt

⊤

J(θt)

=⇒ (gt)k = zk,t Jk(θt) −

K

zj,tJj(θt) . (29)

j=1

where J(θt) = [J1(θt),...,JK(θt)]⊤. The logits ξt are updated using the gradient gt, which increases the weight on tasks whose rewards fall below the current weighted average and allows unconstrained optimization

over the logits ξ while ensuring task weights zt ∈ ∆K. Subsequently at iteration t, the resulting updates are given by

  

  

  ,

  

∇ξz1⊤,t . ∇ξzK,t⊤

J1(θt) . JK(θt)

K

zkt∇θJGRPO,k(θt), ξt+1 = ξt − β gt, where gt =

θt+1 = θt + γt

k=1

(30) Next, we derive the exact gradient expression in Equation (29) in detail.

ξk K

m=1 eξm . For fixed policy parameters θt, we define the inner objective L(ξ) := Kk=1 zk(ξ)Jk(θt), and denote Jk := Jk(θt) for brevity. Let J¯ := Kj=1 zj(ξ)Jj be the current weighted average reward. Softmax derivative. The Jacobian of the softmax function satisfies the standard identity

Recall that task weights are parameterized via logits ξ ∈ RK using the softmax function: zk(ξ) = e

∂zi ∂ξk

= zi(δik − zk),

where δik is the Kronecker delta with δik = 1 if i = k and δik = 0 when i ̸= k. Gradient of the inner objective. We compute the gradient of L(ξ) with respect to ξk:

K

K

∂L ∂ξk

∂zi ∂ξk

zi(δik − zk)Ji

=

Ji =

i=1

i=1

K

= zkJk − zk

ziJi

i=1

= zk Jk − J¯ . Thus, the gradient admits the closed form

(∇ξL(ξ))k = zk Jk −

K

zjJj .

j=1

#### B.3 Derivation of Improvement-Aware Task Reweighting

In this section, we concretely derive the improvement-aware update rule in Subroutine 1. Specifically, we define a per-step minimax optimization problem in terms of policy update direction θt+1 − θt and weights z and use first-order Taylor approximation to jointly derive the policy update direction with task weights.

We begin by considering a generic policy update of the form

θt+1 = θt + γtdt, (31) where dt ∈ Rm is the update direction that is dependent on ∇θJGRPO,k(θt) of all tasks k and γt > 0 is a stepsize. By first-order Taylor expansion,

Ik(t) = JGRPO,k(θt + γtdt) − JGRPO,k(θt)

= γt⟨∇θJGRPO,k(θt),dt⟩ + O(γt2∥dt∥2). (32)

Thus, for sufficiently small γt, the improvement in task k is proportional to the inner product between the task gradient k and the update direction.

Our goal, then, is to define an optimal policy update direction dt and design an improvement-aware update strategy for z such that we prioritize tasks that are underperforming or under-improving. To this end, we consider the following minimax objective at iteration t:

K

- 1

- 2∥dt∥22 + λ

1 γt

zkIk(t) −

max

min

dt∈Rm

z∈∆K

k=1

and, using the first-order approximation in Equation (32), we obtain

K

zkJk(θt), (33)

k=1

K

K

- 1

- 2∥dt∥22 + λ

zkJk(θt), (34)

zk⟨∇θJGRPO,k(θt),dt⟩ −

max

min

dt∈Rm

z∈∆K

k=1

k=1

where the quadratic penalty on ∥dt∥22 keeps the update magnitude controlled, ensuring the validity of the first-order Taylor expansion.

Equation (34) balances worst-case improvement against worst-task performance while regularizing the policy update magnitude dt to reduce the first-order Taylor approximation error from Equation (32). The maximization

over dt selects a policy update direction that best aligns with the tasks emphasized by z, while the inner minimization identifies tasks that are both underperforming or under-improving.

Note that the objective in Equation (34) is concave quadratic in dt and convex in z with z belonging to a compact set ∆K. Hence, strong duality holds (see Liu et al. [11, Proposition 3.1]) and Equation (34) is equivalent to min-max swapped,

min

max

dt∈Rm

z∈∆K

K

- 1

- 2∥dt∥22 + λ

zk⟨∇θJGRPO,k(θt),dt⟩ −

k=1

K

zk.Jk(θt)

k=1

We can now solve the inner maximization by taking the gradient with respect to dt and setting it to zero, which yields the optimal dt as,

K

d∗t(z) =

##### zk∇θJGRPO,k(θt). (35)

k=1

Note that the term λJk(θt) vanishes when differentiating with respect to dt, since it does not depend on dt. Substituting Equation (35) back into Equation (34) eliminates dt and gives an equivalent minimization problem over z:

K

K

2 2

- 1

- 2

zkJk(θt). (36)

zk∇θJGRPO,k(θt)

min

+ λ

z∈∆K

k=1

k=1

Taking the gradient of the objective in Equation (36) with respect to z, we obtain for each k ∈ [K],

K

K

K

2 2

∂ ∂zk

- 1

- 2

zj∇θJGRPO,j(θt), ∇θJGRPO,k(θt) + λJk(θt)

zj∇θJGRPO,j(θt)

zjJj(θt) =

+ λ

j=1

j=1

j=1

(37) = ⟨dt,∇θJGRPO,k(θt)⟩ + λJk(θt) (38) ≈ γtIk(t) + λJk(θt), using Equation (32). (39)

Equation (39) allows us to avoid computing inner products between per-objective gradients, which can be costly or noisy in practice. Instead, we use the per-step improvement approximation Ik(t) from Equation (32), yielding the surrogate signal

##### s(kt) := Ik(t) + λJk(θt), (40)

where we absorb the factor γt into the tunable parameter λ, as the learning rate γt is constant in our training pipeline. Rather than fully solving Equation (36) at each step, we perform a single step gradient descent on z using the surrogate signal in Equation (40).

As in Equation (6), to enforce z ∈ ∆K during optimization, we parameterize zt = Softmax(ξt), ξt ∈ RK, and perform gradient descent with respect to ξt, which yields the update

  

  

  

  . (41)

I1(t) + λJ1(θt) . IK(t) + λJK(θt)

∇ξz1⊤,t . ∇ξzK,t⊤

ξt+1 = ξt − β

Using the softmax Jacobian identity ∂zi

∂ξk = (δik − zk), one obtains the closed form

K

K

zk,ts(kt) k = zk,t s(kt) −

zj,ts(jt) . (42)

∇ξt

j=1

k=1

###### ARC Weights

###### ARC Batch Ratio

###### Resample Counts

0.8

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

3.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.6

0.6

2.5

Weight

0.4

Ratio

0.4

2.0

0.2

1.5

0.2

1.0

0.0

0.0

0 100 200 300 400 500 600

0 200 400 600

0 200 400 600

Steps

Steps

MT-GRPO_0.2 MT-GRPO_0.2_NO_RPS

MT-GRPO_0.2_NO_AAS MT-GRPO_0.2

- Figure 8: Experiment 1 (3-tasks): Left: Comparison between MT-GRPO with and without RP SAMPLER (RPS). In the absence of RPS, MT-GRPO increases the weight assigned to ARC in an attempt to compensate for the mismatch between target weights and effective batch representation. However, the effective ARC representation still remains lower than with RPS, highlighting the benefit of RP SAMPLER . Right: Comparison between MT-GRPO with and without Acceptance Aware Sampling (AAS). Removing AAS increases the average number of resampling rounds (curve smoothed for readability).

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 100 200 300 400 500 600 700

Steps

0.0

0.2

0.4

0.6

0.8

Weight

Zebra Weights

SEC_GRPO

GRPO

SEC_DAPO

DAPO

MT-GRPO_0.2

MT-GRPO_0.25

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

20 30 40

0

200

400

600

Steps to reach (%) worst-task accuracy

GRPO

SEC_GRPO

DAPO

SEC_DAPO

MT-GRPO_0.1 MT-GRPO_0.3

- MT-GRPO_0.9

- MT-GRPO_1.2

- Figure 9: Left: Task weights assigned to Zebra across methods in Experiment 1. (See Figure 6 for full task performance and weights across other tasks.) Right: Number of training steps required to reach specified worsttask accuracy thresholds in Experiment 2. Bars reaching the maximum indicate that the method did not reach the threshold within the training budget. Our method consistently reaches thresholds in fewer steps than baselines.

### C Practical Implementation and Additional Experiments

In this section, we provide the full version of RP SAMPLER introduced in Section 5, along with additional experimental results from Experiments 1 and 2 that were deferred to the appendix due to space constraints.

Figure 9 (left) shows the task weights assigned to Zebra throughout training. Our method initially assigns a lower weight to Zebra and gradually increases it as relative task performance evolves (see also Figure 6). In contrast, all baselines steadily reduce the weight assigned to Zebra over time. This adaptive reweighting explains the higher Zebra accuracy achieved by our method.

- Figure 9 (right) reports the number of training steps required to reach specified worst-task accuracy thresholds in Experiment 2. Consistent with the trends observed in Figure 5, our method reaches all thresholds in fewer training steps than the baseline methods, indicating faster progress on the weakest task. These results further show that our approach not only improves final worst-task accuracy but also accelerates learning on under-performing tasks, even in the larger 9-task setting.

We further analyze the contribution of RP SAMPLER towards the performance of MT-GRPO. Figure 8 (left) compares MT-GRPO with and without RP SAMPLER (RPS). Without RPS, the method assigns higher weights to ARC to compensate for the mismatch between target task weights and the effective task representation in the training batch. However, this compensation is insufficient as the effective ARC representation remains lower than when RPS is enabled, highlighting the benefit of RP SAMPLER .

Figure 8 (right) compares our method with and without Acceptance-Aware Sampling (AAS), a key component of RP SAMPLER (see Algorithm 2). Removing AAS leads to a higher number of resampling rounds on average (the curve is smoothed for readability), demonstrating that AAS improves sampling efficiency.

Algorithm 2 RATIO-PRESERVING SAMPLER (RP SAMPLER) Require: Task weights z ∈ ∆K, batch size B, rollouts per prompt N, oversampling factor Mos, maximum

resamples Nrs, tracked filtered ratios ρ ∈ [0,1)K, maximum acceptance inflation factor Macc

- 1: Desired counts: (n1,...,nK) ∼ Multinomial(B,z)
- 2: Inflation factors: mk ← min 1− 1ρ

k

, Macc ∀k

- 3: Recalibrated generation distribution:

zˆk ←

zk mk K j=1 zj mj

, ∀k

- 4: Sample: (ˆn1,...,nˆK) ∼ Multinomial(MosB,zˆ)
- 5: for k = 1 to K do
- 6: Sample prompts Pk from Dk with |Pk| = nˆk
- 7: Generate rollouts Rk ← {yi,j(n)}j=1:ˆnk, n=1:N
- 8: end for
- 9: X ← ZERO_GRAD_FILTER k(Pk,Rk)
- 10: Accepted counts: ck ← |{x ∈ X : task(x) = k}|
- 11: Deficiencies: defk ← max{nk − ck, 0}, ∀k
- 12: r ← 1 (resampling round counter)
- 13: while Kk=1 defk > 0 and r ≤ Nrs do
- 14: Deficiency-aware resampling distribution:

zˆk(r) ←

defk mk K j=1 defj mj

, ∀k

- 15: (a1,...,aK) ∼ Multinomial(MosB, zˆ(r))
- 16: for k = 1 to K do
- 17: if ak > 0 then
- 18: Sample ak new prompts P˜k from Dk
- 19: Generate N rollouts R˜k for each prompt
- 20: end if
- 21: end for
- 22: X˜ ← ZERO_GRAD_FILTER k(P˜k,R˜k)
- 23: X ← X ∪ X˜
- 24: Update ck and defk ← max{nk − ck, 0}
- 25: r ← r + 1
- 26: end while
- 27: if |X| ≤ B then
- 28: Return X
- 29: else
- 30: For each task k, retain at most nk samples from Xk
- 31: X⋆ ← k Xk(nk)
- 32: Sample Xr ⊆ X \ X⋆ such that |X⋆ ∪ Xr| = B
- 33: Return X⋆ ∪ Xr
- 34: end if

### D Experimental Details and Reproducibility

We use the Qwen-2.5-3B base model for all experiments [83, 84]. All methods are fine-tuned using the Volcano Engine Reinforcement Learning (verl) library [85] for 720 training steps on two NVIDIA H200 (141GB) GPUs. We use the same versions of verl and relevant dependencies as in [4] and incorporate the required DAPO modifications [3]. We use datasets released by Chen et al. [4], generated using the ReasoningGym framework [24]. The dataset contains easy, medium, and hard variants of Countdown, Zebra, and ARC. Each variant includes 1000 training instances and 200 test instances. Rewards follow the protocol of Chen et al. [4]: 1.0 for a correct answer, 0.1 for an incorrect answer with correct formatting, and 0 otherwise.

We estimate advantages using 72 rollouts for Experiment 1 and 8 rollouts for Experiment 2, generated using vLLM [86] with temperature 1.0. We set the KL-divergence coefficient to 0, following standard practice in [4, 3]. The maximum prompt length is 1024 tokens and the maximum response length is 4096 tokens.

We use the AdamW optimizer [87] as implemented in verl, with learning rate 1e − 6 and betas (0.9, 0.99). For Experiment 1, we use a global batch size of 32 and PPO minibatch size of 8 due to the large number of rollouts. For Experiment 2, we use a global batch size of 256 and a minibatch size of 64. In verl, each global step consists of multiple minibatch updates and in our setup, each step performs four policy updates.

All methods use identical training configurations unless otherwise stated. For the SEC baseline [4], we use the hyperparameters reported in the original paper. For both our method and DAPO, we apply the “clip higher” strategy from Yu et al. [3] and set the clipping upper bound to 0.28.

Metrics: In addition to the worst-task accuracy and average accuracy, we have reported the average relative change per task metric, which quantifies how a method’s performance differs from a reference baseline across tasks:

K

Accm(k) − Accb(k) Accb(k) × 100,

1 K

∆m% =

k=1

where Accm(k) denotes the accuracy of method m on task k, and b denotes the reference baseline. For our analysis, we define the reference baseline to be DAPO. Positive values indicate average improvements over the baseline, while negative values indicate degradation. This metric is commonly used in multitask learning to account for differing task difficulty and accuracy scales [11, 25]. It complements worst-task accuracy by capturing whether improvements in robustness come at the expense of broader performance degradation.

Filtering strategy: For experiment 1, we enact a strict DAPO filtering strategy where the prompts with no rollouts with correct answer or all rollouts with correct answer are filtered out. For experiment 2, due to the increased task diversity and computational constraints, we adopt a more lenient filtering strategy following Yu et al. [3] allowing prompts with non-constant rewards across rollouts (for e.g., due to correct/incorrect formatting).

Hyperparameters for MT-GRPO: For experiment 1, we use λ = 0.2,0.25 and for experiment 2, we use λ = 0.1,0.3,0.9,1.2 for the trade-off parameter in Subroutine 1. We initialize the task weights z0 uniformly and

update them according to Subroutine 1 using current batch rewards {Jk(θt)}Kk=1, and improvements {Ik(t)}Kk=1 clipped to {0.1,0.2} for stability. We implement the gradient descent update in Line-5 of Subroutine 1 using

AdamW [87] with a learning rate of 0.025 and weight decay 1e−5 for experiment 1, and 1e−4 for experiment 2. For the RP SAMPLER , we use an oversampling factor of Mos = 3, a maximum of Nrs = 10 resampling rounds for experiment 1, and Nrs = 2 for experiment 2, and inflate sampling probabilities based on filtering rates ρk up to a maximum acceptance inflation factor Macc of 5.

### E Stabilizing Task-weight Updates through Regularization

We consider two mechanisms for improving the stability and effectiveness of task reweighting: (i) regularizing the task-weight dynamics, and (ii) incorporating task-level improvement signals. Here, we discuss the analysis of (i).

Regularized task-weight update: We add a ℓ2 regularization term to the logit update for z = Softmax(ξ). This can be viewed as a practical mechanism for relaxing the strict worst-case behavior induced by the ε = 0 formulation and moving toward the smoother regime implied by ε > 0 in Equation (5).

Concretely, we update ξt by gradient descent on L(ξt) with an additional shrinkage term, yielding

ξt+1 = ξt − β gt + ηξt , (43) where gt is given by

K

zj,tJj(θt) . (44)

(gt)k = zk,t Jk(θt) −

j=1

The additional term ηξt dampens the growth of logits and mitigates weight collapse. We summarize this regularized update in Subroutine 2.

Is regularization sufficient?: The regularized task-weight updates in Equation (43) mitigate rapid collapse of the weight distribution, but they continue to rely solely on the absolute task reward Jk(θt) as the signal for reweighting. However, absolute reward does not distinguish between tasks that are improving rapidly and tasks that have stagnated over the course of training. In multi-task post-training, this distinction is critical. A task that

Subroutine 2 Regularized Task-Weight Update

- 1: Input: rewards {Jk(θt)}Kk=1, logits ξt, stepsize β, regularization λ
- 2: zt ← Softmax(ξt)
- 3: (gt)k ← zk,t Jk(θt) − Kk=1 zk,tJk(θt) ∀k ∈ [K]
- 4: ξt+1 ← ξt − β gt + ηξt
- 5: Return: zt+1 = Softmax(ξt+1)

is currently weak but improving quickly may require less prioritization than a task with a similar reward that is no longer benefiting from updates. While increasing the regularization strength λ in Subroutine 2 stabilizes the dynamics, it does so by driving the weights toward uniformity (average reward maximization), which weakens the prioritization of genuinely underperforming or slowly improving tasks [10, 7, 11]. Consequently, even when collapse is avoided, absolute-reward-based reweighting alone may be insufficient to ensure that certain tasks do not remain under-optimized. These limitations motivate measuring how the loss (i.e., −JGRPO,k(θ)) of each task evolves over training and introducing an improvement-aware task signal that explicitly captures how much each task benefits from policy updates, as done in Subroutine 1.

Experiments: We compare improvement-aware task reweighting (Subroutine 1) to regularized reward-only reweighting (Subroutine 2) in Experiments 1 and 2.

- Experiment 1 (3 tasks): With weak regularization (η = 1e − 5), the task-weight distribution collapses onto the current worst-performing task and largely ignores Countdown for most of training (see Figure 2), resulting in low worst-task and average accuracy. Stronger regularization (η = 1e − 2) stabilizes training and yields similar worst-task accuracy but higher average accuracy and average relative change (Figure 10). These gains are driven primarily by improved performance on the easiest task (Countdown), whereas improvement-aware updates retain stronger performance on the harder task (ARC), which also exhibits high zero-gradient rates. This indicates that regularization tends to smooth weights toward uniformity rather than reassigning them to underperforming or slowly improving tasks.
- Experiment 2 (9 tasks): Moderate regularization (η = 5e − 4) improves worst-task accuracy over baselines, but remains below improvement-aware updates, particularly for λ ∈ {0.9,1.2} in terms of both worst-task accuracy and average relative change on hard tasks (Figure 11). Increasing the regularization to 1e − 2 degrades worst-task accuracy below DAPO. Although average accuracy increases, it remains below DAPO, and the gain is driven primarily by improvements on easy and medium tasks rather than hard tasks (see Figure 11, bottom).

In contrast, improvement-aware updates expose a controllable trade-off. Smaller values of the trade-off parameter (e.g., λ ∈ {0.1,0.3}) prioritize average accuracy and progress on hard tasks (higher relative change) while maintaining worst-task accuracy competitive with baselines. Larger values of λ focus primarily on improving worst-task performance.

Overall, these results suggest that regularization can prevent weight collapse but is insufficient for reliably prioritizing under-optimized tasks. Incorporating improvement signals provides a more effective mechanism for allocating training emphasis to tasks that are under-performing or improving slowly.

###### Worst Task Accuracy

###### Avg Accuracy

###### Avg Relative Change

10

| |Zebra<br><br>Zebra 58.8<br><br>Zebra 58.4<br><br>ARC 58.7 Count-|
|---|---|
| |ARC<br><br>Zebra 52.6<br><br>57.5<br><br>down 51.7|
| |45.8|
| | |
| | |

| |69.7|
|---|---|
| |65.2<br><br>67.2 66.9<br><br>64.1|
| |61.8|
| |55.5|
| | |
| | |
| | |

| | |
|---|---|
| |+5.8<br><br>+5.1<br><br>+7.8|
| |+0.0<br><br>+2.3|
| | |
| |-5.5|
| |-9.2|

70

60

65

5

50

Accuracy

60

0

55

40

5

50

30

45

10

20

40

###### ARC (Medium)

###### Countdown (Medium)

###### Zebra (Medium)

70

70

| | |
|---|---|
| |64.8<br><br>61.6 60.1 58.7 59.9|
| |54.7|
| |45.8|
| | |
| | |
| | |
| | |

| |85.2<br><br>88.2<br><br>79.2 80.3<br><br>90.5|
|---|---|
| |71.0|
| |51.7|
| | |
| | |

| | |
|---|---|
| |57.5 58.8 58.4<br><br>62.7|
| |54.3 52.6<br><br>54.9|
| | |
| | |
| | |
| | |
| | |

60

60

80

50

50

60

40

40

30

30

40

20

20

20

10

10

0

0

0

SEC_DAPO

MT-GRPO_0.2

MT-GRPO_REG_1e-2 MT-GRPO_REG_5e-4

MT-GRPO_REG_1e-5

DAPO

MT-GRPO_0.25

- Figure 10: Experiment 1 (3 tasks): Comparison of improvement-aware updates (Subroutine 1) and regularized task-weight updates (Subroutine 2). With regularization 1e − 2, the regularized task-weight update achieves similar worst-task accuracy but higher average accuracy, largely driven by gains on Countdown. However, improvement-aware updates achieve stronger performance on the harder task (ARC).

| | |
|---|---|
| |Zebra|
| |Zebra<br><br>hard 46.3 Zebra|
| |Zebra hard<br><br>hard<br><br>44.3<br><br>hard<br><br>43.9|
| |Zebra hard Zebra<br><br>hard 42.1 Zebra Zebra|
| |40.9<br><br>40.4 hard hard 39.1|
| |38.8|
| | |

37.5

40.0

42.5

45.0

47.5

50.0

52.5

Accuracy

Worst-Task Accuracy

| |68.7 69.5|
|---|---|
| |66.0<br><br>63.9<br><br>67.6<br><br>64.7|
| |61.7|
| |55.5|
| | |
| | |
| | |

45

50

55

60

65

70

Avg Accuracy

| |+0.0<br><br>+1.7|
|---|---|
| |-2.8 -2.3|
| |-5.4<br><br>-7.7<br><br>-5.1|
| | |
| |-16.8|

15

10

5

0

Avg Relative Change

| |+0.0|
|---|---|
| |-2.7|
| |-6.8<br><br>-4.3<br><br>-6.8|
| |-9.6<br><br>-12.7|
| | |
| |-21.7|
| | |

25

20

15

10

5

0

RelativeChange(%)

Avg Relative Change (Easy)

| | |
|---|---|
| |+0.0<br><br>+2.4<br><br>+1.5|
| |-2.6|
| |-4.6<br><br>-8.6<br><br>-4.0|
| | |
| |-16.0|

20

15

10

5

0

5

Avg Relative Change (Medium)

| |+5.4|
|---|---|
| |+0.0<br><br>+1.1|
| |-2.0 -1.9|
| |-4.2 -4.4|
| |-12.7|
| | |

15

10

5

0

5

Avg Relative Change (Hard)

DAPO

MT-GRPO_0.1

MT-GRPO_0.3 MT-GRPO_0.9

MT-GRPO_1.2

MT-GRPO_REG_1e-2

MT-GRPO_REG_5e-4 MT-GRPO_REG_1e-5

- Figure 11: Experiment 2 (9 tasks): Comparison between improvement-aware updates (Subroutine 1) and regularized task-weight updates (Subroutine 2). Moderate regularization (η = 5e − 4) improves worst-task accuracy over baselines but underperforms improvement-aware updates on both worst-task accuracy and average relative change on hard tasks. Stronger regularization (η = 1e − 2) reduces worst-task accuracy below DAPO and yields only modest gains in average accuracy (still below DAPO), driven primarily by improvements on easy and medium tasks rather than hard tasks.

