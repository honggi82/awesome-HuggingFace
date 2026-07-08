## CoBA-RL: Capability-Oriented Budget Allocation for Reinforcement Learning in LLMs

# arXiv:2602.03048v3[cs.LG]6Feb2026

Zhiyuan Yao14† Yi-Kai Zhang24 Yuxin Chen34 Yueqing Sun4 Zishan Xu5 Yu Yang4 Tianhao Hu4 Qi Gu4 Hui Su4 Xunliang Cai4

### Abstract

- a）GRPO
- b）CoBA-RL Training

[Figure 1]

Training

Step

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a key approach for enhancing LLM reasoning. However, standard frameworks like Group Relative Policy Optimization (GRPO) typically employ a uniform rollout budget, leading to resource inefficiency. Moreover, existing adaptive methods often rely on instance-level metrics, such as task pass rates, failing to capture the model’s dynamic learning state. To address these limitations, we propose CoBA-RL, a reinforcement learning algorithm designed to adaptively allocate rollout budgets based on the model’s evolving capability. Specifically, CoBA-RL utilizes a Capability-Oriented Value function to map tasks to their potential training gains and employs a heap-based greedy strategy to efficiently self-calibrate the distribution of computational resources to samples with high training value. Extensive experiments demonstrate that our approach effectively orchestrates the trade-off between exploration and exploitation, delivering consistent generalization improvements across multiple challenging benchmarks. These findings underscore that quantifying sample training value and optimizing budget allocation are pivotal for advancing LLM post-training efficiency. Our code is available at https:

p1 p2 p3 p1 p2 p3 p1 p2 p3 p1 p2 p3

x1 x2 x3

x1 x2 x3 x1 x2 x3 x1 x2 x3

budget

[Figure 10]

Step

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

p1 p2 p3 p1 p2 p3 p1 p2 p3 p1 p2 p3

x2

x3

x2

x1

x1

x3

x2

x1 x2

x3

x3 x1

budget

Figure 1. Comparison between GRPO-based methods and CoBA-RL. (a) GRPO employs a uniform strategy independent of training progress. (b) CoBA-RL dynamically self-calibrates the allocation strategy throughout the training process. It autonomously directs the rollout budget toward instances with high training value, aligned with the model’s evolving capability. In this visualization, pi denotes the pass rate corresponding to the task instance xi.

Yang et al., 2025; Comanici et al., 2025; Bai et al., 2025). Within this landscape, Group Relative Policy Optimization (GRPO) (Shao et al., 2024) and its variants (Yu et al., 2025; Zheng et al., 2025a; Liu et al., 2026; Shrivastava et al., 2025) have risen to prominence. By assigning a uniform budget of G rollouts to every prompt to compute group-relative advantages, GRPO eliminates the need for a separate value network. However, ideally, the rollout budgets allocated to each instance should be commensurate with its training value.Intuitively, complex samples often harbor higher training value and demand extensive exploration, whereas simple instances require minimal resources.

//github.com/Within-yao/CoBA-RL.

### 1. Introduction

Reinforcement Learning with Verifiable Rewards (RLVR) (Lambert et al., 2024) has established itself as a cornerstone for elevating LLM reasoning in agentic, coding, and mathematical domains (Guo et al., 2025;

In practice, vanilla GRPO overlooks the critical impact of sample difficulty on training value and the corresponding rollout budget. While recent studies have begun to tailor budgets using instance-level metrics related to task difficulty, such as historical pass rates (Li et al., 2025b), these approaches typically rely on static value functions. They operate on the fixed assumption that harder samples inherently offer superior training value than simpler ones and that this relationship remains constant throughout the entire training

†Work done during an internship at Meituan. 1Zhejiang University 2Nanjing University 3National University of Singapore 4Meituan 5Shanghai Jiao Tong University. Correspondence to: Zhiyuan Yao, Qi Gu.

Preprint. February 9, 2026.

process. This perspective neglects a crucial reality: the true training value of a sample is inextricably linked to the policy model’s real-time capabilities (Wang et al., 2025b; Zhang et al., 2025b; Wang et al., 2025a).

As the model’s capabilities evolve during training, the set of samples holding the highest training value constantly shifts (Hu et al., 2025). To accommodate these variations, the allocation strategy must continuously calibrate the tradeoff between exploitation and exploration. Specifically, exploitation involves consolidating mastery over instances where the model already succeeds, whereas exploration requires allocating resources to sample diverse trajectories on challenging queries, thereby expanding the search space to discover potential solutions (Yang et al., 2026; Chen et al., 2025c; Hou et al., 2025; Cui et al., 2025). Consequently, it is imperative to quantify model capability to facilitate policy self-calibration, ensuring the rollout budget is continually re-aligned with the samples most suitable for the current training phase.

To tackle the aforementioned challenges, We propose CoBARL, a reinforcement learning algorithm that dynamically allocates the rollout budget in accordance with the model’s evolving capability. Specifically, we introduce a CapabilityOriented Value function, modeled as a Beta distribution, to map individual instances to their potential training value. By continuously monitoring the global failure rate of the current training batch, we quantify the model’s global capability and dynamically calibrate the shape of the value function, as shown in Figure 1. This mechanism autonomously orchestrates the exploration-exploitation trade-off: the value function shifts its high-density regions in real-time to prioritize either consolidating established knowledge or exploring high-uncertainty frontiers based on the model’s current competence. Finally, to operationalize this theoretical distribution, we present an Efficient Allocation Optimization algorithm. By formulating the allocation as a constrained maximization problem, we employ a heap-based greedy strategy that iteratively assigns budget to samples offering the highest marginal gain, thereby maximizing the aggregate value of the training batch.

To validate the efficacy of our approach, we conducted extensive experiments using Qwen2.5-7B-Base, Qwen2.5-7BInstruct, and Qwen3-1.7B/4B-Base models. Empirical results demonstrate that our method significantly outperforms strong baselines across multiple challenging mathematical benchmarks. These findings suggest that, empowered by the Capability-Oriented Value function, CoBA-RL effectively identifies samples holding the high training value at the current training step, thereby achieving a superior trade-off between exploration and exploitation.

Overall, our contributions are summarized as follows:

- • We propose a reinforcement learning algorithm that optimizes the exploration-exploitation trade-off by autonomously allocating the rollout budget consistent with the model’s evolving capability. By formulating the allocation as a constrained maximization problem, we employ a heap-based greedy strategy to efficiently direct computational resources toward tasks with the highest learning potential.
- • We design a Capability-Oriented Value function as the core allocation criterion. This mechanism dynamically quantifies the training value of each task instance conditioned on the policy model’s evolving capability.
- • Extensive experiments across multiple benchmarks validate the effectiveness of CoBA-RL. Our results demonstrate that it significantly outperforms the standard GRPO baseline, as well as static and heuristic strategies, providing a paradigm for efficient Large Language Model post-training.

### 2. Method

In this section, we present CoBA-RL, as illustrated in Figure 2. The framework is composed of two pivotal components. First, the Capability-Oriented Value Function quantifies the policy model’s capability via global failure rate statistics. This mechanism self-calibrates the function’s shape to discern the model’s evolving preferences for specific training samples. Second, the Heap-Based Greedy Budget Allocation is an efficient strategy that optimizes allocation to maximize the cumulative value of the current training batch.

###### 2.1. Preliminaries

The post-training process is formulated as a Reinforcement Learning problem utilizing Group Relative Policy Optimization (GRPO). At each training step t, considering a batch of tasks Xt = {x1,...,xM}, the policy πθ

samples a group of G outputs Oi = {oi,1,...,oi,G} for each task xi.

t

Given a binary outcome reward function R(x,o) ∈ {0,1}, the task pass rate pi within this group-based generation framework is naturally modeled as the expected probability of correctness:

θt(·|xi) [I(R(xi,o) = 1)]. (1)

pi(xi;θt) = Eo∼π

The optimization objective is to maximize the following loss:

JGRPO(θ) = Ex∼X

t

G

1 G

min ρi,kAi,k,

k=1

clip(ρi,k,1 − ϵ,1 + ϵ)Ai,k ,

(2)

a) Capability-Oriented Budget Allocation for Reinforcement Learning

[Figure 19]

Capability-Oriented Budget Allocation for Reinforcement Learning

###### KL

[Figure 20]

Reference Model

Step k budget

- 1

. . .

- 2

- 1

. . .

- 2

1

- 1

. . .

- 2

[Figure 21]

[Figure 22]

- 1

. . .

- 2

1

Policy Model

[Figure 23]

[Figure 24]

CoBA

[Figure 25]

xi

Group Computation

Budget Allocation

[Figure 26]

Reward Model

. . . 2

[Figure 27]

##### . . . 2

Question

[Figure 28]

[Figure 29]

Value Function

Budget Allocation

[Figure 30]

Globalp,k value function

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

total budget

Model Capability

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

#### . . .

Update Max-Heap

- 1 = 1

= 0.7

- 2 = 0

[Figure 41]

=Clip( ​+λ⋅ , ​, ​)

x2

xM

x1

[Figure 42]

[Figure 43]

=κ− .​

[Figure 44]

[Figure 45]

Compute the marginal gain

xi

[Figure 46]

[Figure 47]

[Figure 48]

x2

x1

. . .

ΔV( , )>   ( 2, 2) > . . . >   ( 1, 1)

[Figure 49]

#### . . .

[Figure 50]

[Figure 51]

Select xi with the maximum marginal gain !

V( , ​, ​)

xi

[Figure 52]

[Figure 53]

[Figure 54]

Capability-Oriented Value Function

[Figure 55]

x2

[Figure 56]

xM

budget ...

x1

c) Budget Allocation

b) Value Function

- Figure 2. Overview of CoBA-RL. (a) The training pipeline of CoBA-RL. (b) The adaptive value function influenced by model capability. (c) Illustration of the heap-based greedy allocation.

where ρi,k denotes the probability ratio between the current and old policies, and Ai,k represents the estimated advantage for each output.

ing potential of each task under the current policy πθ. Intuitively, as the model masters an task, its value should diminish, thereby allocating the budget to other samples.

###### 2.2. Problem Formulation and Optimization

We now formally define the budget allocation problem. The system operates under a total rollout budget constraint Btotal. For each task xi, the policy generates a set of Bi trajectories Oi = {oi,1,...,oi,B

i}, subject to Bi = Btotal.

The objective is to determine the optimal set of budgets B = {B1,...,BM} that maximizes the aggregate training gain defined by V . In particular, the function V maps each task xi and its allocated budget Bi to a scalar value representing the expected learning potential. The optimization problem is formulated as:

max

B1,...,BM

subject to

M

V (Bi,πθ,pi)

i=1

M

Bi = Btotal,

i=1

Blow ≤ Bi ≤ Bup, ∀i ∈ {1,...,M}, Bi ∈ Z+.

(3)

The primary challenge lies in accurately defining the value function V .This function must dynamically reflect the train-

###### 2.3. Capability-Oriented Value Function

To identify the training samples with high learning value during the training process,, we propose a value function that integrates global capabilities awareness.

Definition 2.1 (Global Capability). We formally define the model’s global capability at step t. The Global Success Rate St is defined as the expected pass rate averaged over the current task batch:

1 M

St =

M

pi(xi;θt). (4)

i=1

Correspondingly, the Global Failure Rate Ft represents the complementary probability: Ft = 1 − St. These metrics serve as quantitative indicators of the policy model’s capability variation; specifically, as the model’s capability strengthens, the Global Success Rate St is expected to increase.

2.3.1. CAPABILITY-INDUCED PREFERENCE DENSITY

To dynamically adjust the model’s preference for samples of varying difficulty, we formalize the Capability-Induced Preference Density, which explicitly quantifies the varying degrees of preference that different policy models exhibit

Algorithm 1 Heap-Based Greedy Budget Allocation

parameters via a linear mapping, while keeping the sum κ = αt + βt constant:

- 1: Input: Task set {xi}Mi=1, Pass rates p = {pi}, Total budget Btotal, Constraints Blow,Bup.
- 2: Output: Optimal budget allocation B = {B1,...,BM}.
- 3: Initialize Bi ← Blow for all i ∈ {1,...,M}.
- 4: R ← Btotal − Mi=1 Blow
- 5: Construct Max-Heap H:
- 6: for i = 1 to M do
- 7: if Bi < Bup then
- 8: Calculate marginal gain: ∆Vi ← V (Bi + 1,pi) − V (Bi,pi)
- 9: Push(H,(∆Vi,i))
- 10: end if
- 11: end for
- 12: while R > 0 and H is not empty do
- 13: (∆Vi∗,i∗) ← Pop(H) {Select task with max marginal gain}
- 14: Bi∗ ← Bi∗ + 1
- 15: if Bi∗ < Bup then
- 16: Update marginal gain: ∆Vnew ← V (Bi∗ + 1,pi∗) − V (Bi∗,pi∗)
- 17: Push(H,(∆Vnew,i∗))
- 18: end if
- 19: R ← R − 1
- 20: end while
- 21: return B

αt = clip αmin + λ · F˜t, αmin, αmax , βt = κ − αt.

(7)

This dynamic modeling mechanism offers significant advantages. Macroscopically, in the early training stages characterized by a high F˜t, the distribution skews towards high pass-rate samples to rapidly acquire training signals, gradually shifting towards harder samples as capability improves. Importantly, this mechanism agilely responds to capability fluctuations by automatically recalibrating the preference density, ensuring the budget is allocated to the sample interval that best consolidates the model’s current capability, rather than blindly pursuing high-difficulty tasks.

- 2.3.2. BUDGET SATURATION FACTOR

While the Capability-Induced Preference Density identifies which tasks are theoretically pivotal, the actual training gain derived from a task instance xi is intrinsically coupled with the allocated computational budget Bi. Intuitively, increasing Bi yields higher returns, but this gain follows the law of diminishing returns. To quantify this relationship, we design the Budget Saturation Factor:

η(Bi,pi) = 1 − e−

Bi

τ pi(1−pi), (8)

where τ is a temperature coefficient that controls the velocity at which the value reaches saturation.

By synthesizing the budget constraint with the CapabilityInduced Preference Density, we formulate the final Capability-Oriented Value Function as:

V (Bi,πθ,pi) = 1 − e−

Bi

τ pi(1−pi) · Density(pi;αt,βt). (9)

The Budget Saturation Factor acts as a realizability coefficient, forcing the optimization of Bi to conform to the shape of the Capability-Induced Preference Density. Consequently, as the policy model’s capability evolves, thereby altering αt and βt, the objective function V dynamically shifts its topology. This ensures that the budget allocation strategy is not static, but continuously realigns itself with the model’s changing proficiency throughout the training trajectory. Through this value function, we can adaptively map different task instances to a value based on the current model capability and the allocated rollout budget.

- 2.4. Efficient Allocation Optimization via Heap-Based Greedy Strategy

toward samples. We model this density as a Beta distribution whose parameters evolve with the training step t. We employ the Beta distribution for its flexible parameterization that adaptively shapes the preference density, facilitating a continuous, self-calibrating shift in sampling focus aligned with the model’s evolving capability:

t−1

pα

t−1 B(αt,βt)

i (1 − pi)β

Density(pi;αt,βt) =

, (5)

where αt and βt are determined by the current global capability metrics. Specifically, we first compute the moving average of the global failure rate over the past k steps, denoted as F¯t, to obtain a stable capability estimate. To enhance sensitivity to subtle capability fluctuations during low-failure stages, we apply a non-linear transformation Ψ(·) to obtain F˜t:

F ¯t, if F¯t > 0.5 σ γ · (F¯t − 0.5) , if F¯t ≤ 0.5

F˜t = Ψ(F¯t) =

(6)

where σ(·) is the Sigmoid function and γ = 10 serves as a scaling factor. Based on F˜t, we determine the shape

The optimization of the objective function defined in Eq. 3 relies on the mathematical property of the proposed value

Table 1. Performance comparison of GRPO, Knapsack-RL, and CoBA-RL across different models and benchmarks. All results are reported in percentage (%). The best results for each model are highlighted in bold. Superscripts denote the difference relative to the GRPO baseline (Blue for Knapsack-RL, Red for CoBA-RL).

Method AIME24 AIME25 AMC23 MATH500 OLYMPIAD Avg Qwen2.5-7B-Instruct

- GRPO 14.17 12.71 69.84 76.78 37.68 42.24

- Knapsack-RL 18.54

+4.37

15.21

+2.50

71.41

+1.57

80.55

+3.77

41.23

+3.55

45.39

+3.15

CoBA-RL 18.96

+4.79

18.33

+5.62

73.12

+3.28

80.30

+3.52

43.19

+5.51

46.78

+4.54

Qwen2.5-7B-Base

GRPO 15.41 13.33 75.00 77.63 37.03 43.68

- Knapsack-RL 19.58

###### 16.67

74.37−0.63

79.73

41.33

46.34

+4.17

+3.34

+2.10

+4.30

+2.66

- CoBA-RL 21.04

+5.63

16.04

+2.71

76.71

+1.71

80.23

+2.60

43.11

+6.08

47.43

+3.75

Qwen3-4B-Base GRPO 18.54 15.62 65.62 81.19 42.61 44.72 Knapsack-RL 21.88

+3.34

20.01

+4.39

65.47−0.15

80.51−0.68

45.42

+2.81

46.66

+1.94

- CoBA-RL 22.71

###### 21.16

###### 72.34

###### 84.29

###### 46.78

###### 49.46

+4.17

+5.54

+6.72

+3.10

+4.17

###### +4.74

Qwen3-1.7B-Base GRPO 8.96 5.00 44.69 69.46 29.92 31.61 Knapsack-RL 12.50

4.79−0.21

45.16

69.69

30.68

32.56

+3.54

+0.47

+0.23

+0.76

+0.95

CoBA-RL 16.25

###### 7.29

###### 49.84

###### 72.85

###### 32.52

###### 35.75

+7.29

+2.29

+5.15

+3.39

+2.60

###### +4.14

function.

Proposition 2.2. The marginal gain of the value function is strictly monotonically decreasing with respect to the allocated budget Bi. That is, defining the marginal gain as ∆V (Bi,pi) = V (Bi + 1,pi) − V (Bi,pi), the following inequality holds for all Bi ≥ 0:

∆V (Bi,pi) > ∆V (Bi + 1,pi). (10)

The proof is provided in the Appendix A. This diminishing marginal utility property guarantees that a greedy allocation strategy yields an optimal solution for the discrete resource allocation problem. Consequently, we design an efficient Heap-Based Greedy Strategy. Specifically, we maintain a max-heap of marginal gains for all tasks. In each iteration, we extract the task with the highest potential gain from the heap top, allocate a unit rollout budget, update its state, and re-insert it into the heap. The detailed procedure is outlined in Algorithm 1.

Notably, this strategy operates based on the relative magnitude of marginal gains within the current training batch. This facilitates a dual-adaptive allocation mechanism where resource distribution naturally shifts in response to the evolving shape of the value function—reflecting the shifting preferences of the current policy πθ—while simultaneously adjusting to the specific variations of the samples in the batch.

Table 2. Performance comparison of Exploration-Exploitation strategies on Qwen2.5-7B-Instruct.

Strategy AIME24 AIME25 AMC23 MATH500 OLYMPIAD Avg

Explore → Exploit 16.87 10.41 73.43 79.93 41.84 44.50 Exploit → Explore (Ours) 18.96 18.33 73.12 80.30 43.19 46.78

Such a design guarantees that the computational budget is dynamically channeled toward the instances offering the greatest learning potential.

### 3. Experiments

###### 3.1. Experimental Setup

We initialize our policy models using Qwen2.5-7BInstruct (Qwen et al., 2025), Qwen2.5-7B-Base (Qwen et al., 2025) and Qwen3-1.7B/4B-Base (Yang et al., 2025). For training, we utilize DAPO-Math-17K (Yu et al., 2025), a dataset widely adopted for mathematical reasoning tasks. For each training instance, we generate G = 16 rollout trajectories.

We evaluate performance on five challenging benchmarks: AIME24 (Zhang & Math-AI, 2024), AIME25 (Zhang & Math-AI, 2025), AMC23, MATH500 (Lightman et al., 2024), and OLYMPIAD Bench (He et al., 2024). To benchmark the efficacy of our dynamic budget allocation, we com-

###### Qwen3-1.7B-Base

###### Qwen3-4B-Base

###### Qwen2.5-7B-Base

###### Qwen2.5-7B-Instruct

0.325

0.45

0.42

0.40

0.300

0.275

0.40

0.40

0.35

Accuracy

Accuracy

Accuracy

Accuracy

0.250

0.35

0.38

0.30

0.225

GRPO

GRPO

GRPO

GRPO

0.30

0.25

0.36

0.200

Knapsack-RL

Knapsack-RL

Knapsack-RL

Knapsack-RL

CoBA-RL

CoBA-RL

CoBA-RL

CoBA-RL

0.175

0.20

0.34

0.25

0 100 200 300 400 500

0 100 200 300 400 500

0 200 400 600 800 1000

0 200 400 600 800 1000

Training Steps

Training Steps

Training Steps

Training Steps

- Figure 3. Performance comparison of different models and methods on the Olympiad benchmark (avg@16). The curves track the validation accuracy over training steps for GRPO, Knapsack, and CoBA-RL across varying model scales.

###### =7.26, =3.74

###### =4.50, =6.50

###### =2.54, =8.46

100

80

80

70

80

AllocationCount

AllocationCount

AllocationCount

60

60

60

50

40

40

40

30

20

20

20

10

0

0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Pass Rate

Pass Rate

Pass Rate

Figure 4. Visualization of different budget allocation across varying model capabilities on Qwen2.5-7B-instruct.

pare CoBA-RL against two primary baselines: GRPO (Shao

- et al., 2024) and Knapsack-RL (Li et al., 2025b). For fair comparison, all reported results are evaluated using the avg@16 metric. Comprehensive implementation details and additional experimental results are provided in Appendix C and D.

###### 3.2. Main Results

The quantitative results of our experiments are summarized in Table 1. In terms of overall performance, CoBARL achieves an average accuracy of 46.78% on Qwen2.57B-Instruct. This performance surpasses the GRPO baseline of 42.24% by a significant margin of 4.54% and consistently outperforms the 45.39% accuracy achieved by Knapsack-RL. Parallel improvements are evident on Qwen2.5-7B-Base, where CoBA-RL attains an overall average of 47.43%. Notably, on the OLYMPIAD benchmark, our method achieves an accuracy of 43.11%, surpassing the Knapsack-RL baseline of 41.33% by 1.78%. Similarly, on the Qwen3-4B-Base and Qwen3-1.7B-Base models, our method yields average improvements of 4.74% and 4.14% over GRPO, respectively, demonstrating robust scalability across different model sizes. Figure 3 illustrates the training curves of CoBA-RL on the Olympiad benchmark (Avg@16) across different models.

Notably, on the AIME25 benchmark using the Qwen2.5-7BInstruct model, CoBA-RL improves accuracy from 12.71%

to 18.33%. This marks a 5.62% increase over GRPO and further surpasses the 15.21% performance of Knapsack-RL. Furthermore, on the AMC23 dataset with Qwen3-4B-Base, our method achieves a remarkable increase of 6.72% over the GRPO baseline. These findings underscore the critical importance of dynamically discerning whether the policy requires a bias toward exploration or exploitation throughout the reinforcement learning training process.

To intuitively validate our allocation mechanism, we visualize the resulting budget distributions under different value functions in Figure 4.As depicted, the allocated budget dynamically shifts in response to the model’s evolving capability, conforming to the geometric shape of the specified value functions.

3.3. The Relationship between Exploration and Exploitation in Reinforcement Learning

We investigate the optimal scheduling of exploration and exploitation within the reinforcement learning process. Specifically, we conducted a comparative experiment on the Qwen2.5-7B-Instruct model to evaluate the performance differences between an “Explore → Exploit” strategy and our proposed “Exploit → Explore” strategy. The evolution of αt for both methods is visualized in Figure 5.

As presented in Table 2, the results demonstrate that the “Exploit → Explore” strategy yields superior overall performance. Specifically, our method achieves a substantial

Table 3. Quantitative comparison against Static and Heuristic baselines. The best results are highlighted in bold.

###### Method AIME24 AIME25 AMC23 MATH500 OLYMPIAD Avg

(α = 10.5,β = 1.5) 15.20 15.21 68.28 80.17 42.26 44.22 (α = 1.5,β = 10.5) 18.12 16.45 70.31 79.92 41.25 45.21 Linear Step Decay 17.08 16.85 70.77 79.81 42.43 45.39 CoBA-RL (Ours) 18.96 18.33 73.12 80.30 43.19 46.78

improvement on the AIME25 benchmark, boosting accuracy from 10.41% to 18.33%, and attains the highest average score of 46.78%. This empirical evidence supports our hypothesis that prioritizing the exploitation of simple samples in the early stages facilitates the rapid consolidation of foundational capabilities, whereas allocating the budget to explore difficult instances in later stages effectively expands the model’s solution space.

- 0 198 396 594 792 990 Steps

- 3

- 4

- 5

- 6

- 7

- 8

Alpha

Exploit Explore

0 198 396 594 792 990

Steps

- 3

- 4

- 5

- 6

- 7

- 8

Alpha

Explore Exploit

Figure 5. Evolution of αt on Qwen2.5-7B-Instruct. Left: The ”Exploit → Explore” strategy, where αt exhibits a fluctuating downward trend. Right: The ”Explore → Exploit” strategy, where αt shows a fluctuating upward trend.

3.4. Comparative Analysis against Static and Heuristic Baselines

Our approach explicitly accounts for the evolution of model capability throughout the reinforcement learning process, regulating resource allocation by mapping capability fluctuations to an adaptive value function. To verify the efficacy of this dynamic mechanism, we benchmark it against three baselines categorized into Static and Heuristic strategies. Specifically, we evaluate two Static Strategies employing a Fixed Value Function: one configured with (α,β) = (10.5,1.5) to prioritize exploitation, and another with (α,β) = (1.5,10.5) to prioritize exploration. Additionally, we compare against a Heuristic Strategy using Linear Step Decay, where αt decreases stepwise from 10 to

- 1 (i.e., 10 → 9 → ··· → 1) corresponding to the progression of training steps.Visualizations of these strategies are provided in Appendix D.3.

The quantitative results are presented in Table 3. As observed, our dynamic budget allocation strategy consistently outperforms both the static fixed-parameter strategies and the heuristic baseline. CoBA-RL achieves the highest average accuracy of 46.78%, surpassing the 45.39% perfor-

mance of Linear Step Decay and the 45.21% accuracy of the best Static baseline by margins of 1.39% and 1.57%, respectively. This validates that static, pre-determined strategies struggle to align with the unpredictable dynamics of real learning compared to our adaptive approach.

###### 3.5. Ablation Study

GRPO

0.48

Knapsack-RL

CoBA-RL

0.4678

0.46

0.4552 0.4539

Accuracy

0.4473

0.44

0.4278

0.42

0.4100

0.40

2048 4096

Btotal

Figure 6. Performance comparison under different exploration budget.

We evaluate the performance of Qwen2.5-7B-instruct under varying total budget constraints Btotal, as illustrated in Figure 6. The results clearly demonstrate the consistent superiority of CoBA-RL over both the baseline GRPO and Knapsack-RL across different resource constraints. Significantly, CoBA-RL exhibits exceptional data efficiency; with a restricted budget of Btotal = 2048, it attains an accuracy of 45.52%. This result not only exceeds the performance of competing methods at the same budget level but, strikingly, surpasses the 42.78% accuracy of GRPO trained with a doubled budget of Btotal = 4096. We attribute this to the inherent resource inefficiency of GRPO’s uniform allocation strategy, which indiscriminately distributes the doubled budget, thereby squandering significant computational resources on samples with negligible training potential.In contrast, our findings verify that the capability-oriented value function in CoBA-RL effectively guides the model to utilize computational resources to explore high-value instances, thereby maximizing the aggregate value of the training batch.

###### 3.6. Runtime Efficiency of Budget Allocation

To validate the computational superiority of our allocation algorithm, we compare the Heap-Based Greedy strategy

Table 4. Computational efficiency comparison between DP and our Heap-Based Greedy strategy (Btotal = 8192).

| |DP Heap-Based Greedy (Ours) Speedup|
|---|---|
|Allocation Time (s)<br><br>|115.05 0.124 ∼ 928×|

against the standard Dynamic Programming baseline in Table 4. Notably, due to the diminishing marginal returns property proven in Appendix A, both methods theoretically yield identical allocation results; thus, our comparison focuses exclusively on runtime.

Theoretical analysis reveals that the Dynamic Programming baseline suffers from a pseudo-polynomial complexity of O(M·Btotal·(Bup−Blow)). Consequently, the computational cost scales linearly with the product of the batch size, the total budget, and the task-specific allocation range. Given a large state space where the batch size M is 512 and the total budget Btotal equals 8192, the baseline method requires approximately 115.05 seconds. This latency is unacceptable for online reinforcement learning loops. In contrast, our Heap-Based Greedy strategy significantly reduces the complexity to O(Btotal log M), effectively decoupling the multiplicative dependency between budget size and batch size. Consequently, our method completes the allocation in merely 0.124 seconds. This demonstrates that our approach operates with minimal time complexity, allowing for seamless integration into large-scale training pipelines.

### 4. Related Work

###### 4.1. Reinforcement Learning for LLMs

Methods based on Reinforcement Learning with Verification and Reasoning (RLVR) have proven effective in enhancing the reasoning capabilities of LLMs (Dai et al., 2025; Trung et al., 2024; Zheng et al., 2025c;b; Jaech et al., 2024; Xie et al., 2025). Among them, GRPO (Shao et al., 2024) has been widely adopted due to its effectiveness and efficiency. Building on this, GSPO (Zheng et al., 2025a) defines sample importance based on sequence likelihood, while DAPO (Yu et al., 2025) introduces four distinct techniques to bolster reinforcement learning performance. However, these predominantly group-based mechanisms often overlook the inherent variability across different tasks, inevitably leading to a significant waste of rollout resources.

###### 4.2. Progressive Training and Resource Allocation

Progressive training strategies, particularly curriculum learning, have been widely adopted to enhance model performance by organizing training data into distinct difficulty stages (Shi et al., 2025; Wu et al., 2025; Zeng et al., 2025). These approaches typically adhere to an easy-tohard paradigm, emphasizing sample selection and curriculum design (Lee et al., 2024; Na¨ır et al., 2024; Deng et al.,

2025; Li et al., 2025a). For instance, ADCL (Zhang et al., 2025a) addresses difficulty shifts by periodically evaluating subsequent data batches, while SEC (Chen et al., 2025b) utilizes policy gradient advantages to dynamically adjust data distribution. While curriculum methods focus on selecting which samples to train, our approach focuses on adaptively allocating varying budgets.

To achieve this, we draw inspiration from budget allocation, a fundamental problem in operations research (Hussain et al., 2013). Extensive studies have explored this in domains such as online advertising and marketing (Liu et al., 2020; Chen et al., 2025a; Cai et al., 2023), often utilizing Multi-Armed Bandit frameworks to optimize resource distribution (Ge et al., 2025). Recently, these concepts have been adapted to Large Language Models (LLMs). For example, ROI-Reasoning (Zhao et al., 2026) formulates inference under limited token budgets as an ordered stochastic multiple-choice knapsack problem.

Despite these advancements, dynamically allocating rollout budgets during reinforcement learning (RL)—specifically to optimize the trade-off between exploration and exploitation—remains a significant challenge. While approaches like GVM-RAFT (Yao et al., 2025) allocate resources via rejection sampling to minimize stochastic gradient variance, and Knapsack-RL (Li et al., 2025b) employs a classic knapsack formulation (Pisinger & Toth, 1998) to maximize batch value, they often rely on static or pre-defined value functions. Consequently, they fail to effectively adapt to the model’s capabilities, which evolve dynamically throughout the training process, as they lack a mechanism to explicitly correlate the potential training value of individual samples with the model’s real-time proficiency.

### 5. Conclusion

We propose CoBA-RL, a reinforcement learning algorithm that dynamically adapts rollout budget allocation to evolving LLM capabilities by formulating the task as a constrained optimization problem. Importantly, this algorithm allows for seamless integration into existing RL pipelines. To precisely quantify the learning value of different samples during training, we introduce a Capability-Oriented Value function modeled via a dynamic Beta distribution, employing a heap-based greedy strategy to iteratively maximize marginal gains. Experiments demonstrate that CoBA-RL achieves a superior exploration-exploitation trade-off, significantly outperforming static and heuristic baselines. Our analysis further reveals that effective training dynamics typically transition from exploitation to exploration, shifting from the rapid improvement of model capabilities to the investigation of diverse trajectories within a larger search space. Looking forward, we posit that accurately defining the training potential of tasks and optimizing budget allocation represent

critical directions for the future advancement of efficient LLM post-training.

### References

Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., et al. Kimi K2: open agentic intelligence. CoRR, abs/2507.20534, 2025. doi: 10.48550/ARXIV. 2507.20534. URL https://doi.org/10.48550/ arXiv.2507.20534.

Cai, T., Jiang, J., Zhang, W., Zhou, S., Song, X., et al. Marketing budget allocation with offline constrained deep reinforcement learning. In Chua, T., Lauw, H. W., Si, L., Terzi, E., and Tsaparas, P. (eds.), Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining, WSDM 2023, Singapore, 27 February 2023 - 3 March 2023, pp. 186–194. ACM, 2023. doi: 10. 1145/3539597.3570486. URL https://doi.org/ 10.1145/3539597.3570486.

Chen, H., Chen, Y.-J., Park, S.-H., and Shin, D. Multichannel advertising: Budget allocation in the presence of spillover and carryover effects. Manufacturing & Service Operations Management, 27(3):862–880, 2025a.

Chen, X., Lu, J., Kim, M., Zhang, D., Tang, J., Pich´e, A., Gontier, N., Bengio, Y., and Kamalloo, E. Selfevolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970, 2025b.

Chen, Z., Qin, X., Wu, Y., Ling, Y., Ye, Q., Zhao, W. X., and Shi, G. Pass@ k training for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751, 2025c.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Dai, R., Song, L., Liu, H., Liang, Z., Yu, D., Mi, H., Tu, Z., Liu, R., Zheng, T., Zhu, H., et al. Cde: Curiositydriven exploration for efficient reinforcement learning in large language models. arXiv preprint arXiv:2509.09675, 2025.

Deng, H., Zou, D., Ma, R., Luo, H., Cao, Y., and Kang, Y. Boosting the generalization and reasoning of vision language models with curriculum reinforcement learning. arXiv preprint arXiv:2503.07065, 2025.

Ge, L., Xu, Y., Chu, J., Cramer, D., Li, F., Paulson, K., and Song, R. Multi-task combinatorial bandits for budget allocation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pp. 2247–2258, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z. L., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., Liu, J., Qi, L., Liu, Z., and Sun, M. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In ACL (1), pp. 3828–3850. Association for Computational Linguistics, 2024.

Hou, Z., Lv, X., Lu, R., Zhang, J., Li, Y., Yao, Z., Li, J., Tang, J., and Dong, Y. Advancing language model reasoning through reinforcement learning and inference scaling. arXiv preprint arXiv:2501.11651, 2025.

Hu, Z., Qiu, J., Bai, T., Yang, H., Yuan, B., Jing, Q., He, C., and Zhang, W. Vade: Variance-aware dynamic sampling via online sample-level difficulty estimation for multimodal rl. arXiv preprint arXiv:2511.18902, 2025.

Hussain, H., Malik, S. U. R., Hameed, A., Khan, S. U., Bickler, G., Min-Allah, N., Qureshi, M. B., Zhang, L., Yongji, W., Ghani, N., et al. A survey on resource allocation in high performance distributed computing systems. Parallel Computing, 39(11):709–736, 2013.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N., Lyu, S., et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Lee, B. W., Cho, H., and Yoo, K. M. Instruction tuning with human curriculum. In Findings of the Association for Computational Linguistics: NAACL 2024, pp. 1281– 1309, 2024.

Li, R., Huang, H., Wei, F., Xiong, F., Wang, Y., and Chu, X. Adacurl: Adaptive curriculum reinforcement learning with invalid sample mitigation and historical revisiting. arXiv preprint arXiv:2511.09478, 2025a.

Li, Z., Chen, C., Yang, T., Ding, T., Sun, R., Zhang, G., Huang, W., and Luo, Z.-Q. Knapsack rl: Unlocking exploration of llms via optimizing budget allocation. arXiv preprint arXiv:2509.25849, 2025b.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In ICLR. OpenReview.net, 2024.

Liu, M., Yue, W., Qiu, L., and Li, J. An effective budget management framework for real-time bidding in online advertising. IEEE access, 8:131107–131118, 2020.

Liu, S.-Y., Dong, X., Lu, X., et al. Gdpo: Group rewarddecoupled normalization policy optimization for multireward rl optimization. arXiv preprint arXiv:2601.05242, 2026.

Na¨ır, M., Yamani, K., Lhadj, L., and Baghdadi, R. Curriculum learning for small code language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pp. 390–401, 2024.

Pisinger, D. and Toth, P. Knapsack problems. In Handbook of Combinatorial Optimization: Volume1–3, pp. 299–428. Springer, 1998.

Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., et al. Qwen2.5 technical report, 2025. URL https: //arxiv.org/abs/2412.15115.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Shi, T., Wu, Y., Song, L., Zhou, T., and Zhao, J. Efficient reinforcement finetuning via adaptive curriculum learning. arXiv preprint arXiv:2504.05520, 2025.

Shrivastava, V., Awadallah, A., Balachandran, V., Garg, S., Behl, H., and Papailiopoulos, D. Sample more to think less: Group filtered policy optimization for concise reasoning. arXiv preprint arXiv:2508.09726, 2025.

Trung, L., Zhang, X., Jie, Z., Sun, P., Jin, X., and Li, H. Reft: Reasoning with reinforced fine-tuning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7601–7614, 2024.

Wang, Q., Ke, J., Ye, H., Lin, Y., Fu, Y., Zhang, J., Keutzer, K., Xu, C., and Chen, Y. Angles don’t lie: Unlocking training-efficient rl through the model’s own signals, 2025a. URL https://arxiv.org/abs/ 2506.02281.

Wang, Z., Cui, G., Li, Y.-J., Wan, K., and Zhao, W. Dump: Automated distribution-level curriculum learning for rlbased llm post-training, 2025b. URL https://arxiv. org/abs/2504.09710.

Wu, M., Qian, Q., Liu, W., Wang, X., Huang, Z., Liang, D., Miao, L., Dou, S., Lv, C., Wang, Z., et al. Progressive mastery: Customized curriculum learning with guided prompting for mathematical reasoning. arXiv preprint arXiv:2506.04065, 2025.

Xie, T., Gao, Z., Ren, Q., Luo, H., Hong, Y., Dai, B., Zhou, J., Qiu, K., Wu, Z., and Luo, C. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., et al. Qwen3 technical report. CoRR, abs/2505.09388, 2025. doi: 10.48550/ARXIV.2505.09388. URL https://doi. org/10.48550/arXiv.2505.09388.

Yang, F., Chen, Z., Wang, X., Lu, X., Chai, J., Yin, G., Lin, W., Ma, S., Zhuang, F., Wang, D., et al. Your group-relative advantage is biased. arXiv preprint arXiv:2601.08521, 2026.

Yao, J., Hao, Y., Zhang, H., Dong, H., Xiong, W., Jiang, N., and Zhang, T. Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and rl. arXiv preprint arXiv:2505.02391, 2025.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., et al. DAPO: an open-source LLM reinforcement learning system at scale. CoRR, abs/2503.14476, 2025. doi: 10.48550/ARXIV.2503.14476. URL https://doi.

org/10.48550/arXiv.2503.14476.

Zeng, Y., Sun, Z., Ji, B., Min, E., Cai, H., Wang, S., Yin, D., Zhang, H., Chen, X., and Wang, J. Cures: From gradient analysis to efficient curriculum learning for reasoning llms. arXiv preprint arXiv:2510.01037, 2025.

Zhang, E., Yan, X., Lin, W., Zhang, T., and Qianchun, L. Learning like humans: Advancing llm reasoning capabilities via adaptive difficulty curriculum learning and expert-guided self-reformulation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 6630–6644, 2025a.

- Zhang, X., Liangyu, X., Duan, F., Zhou, Y., Wang, S., Weng, R., Wang, J., and Cai, X. Preference curriculum: Llms should always be pretrained on their preferred data. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 21181–21198, 2025b.
- Zhang, Y. and Math-AI, T. American invitational mathematics examination (aime) 2024, 2024.

Zhang, Y. and Math-AI, T. American invitational mathematics examination (aime) 2025, 2025.

Zhao, M., Qi, Q., and Sun, H. Roi-reasoning: Rational optimization for inference via pre-computation metacognition. arXiv preprint arXiv:2601.03822, 2026.

Zheng, C., Liu, S., Li, M., Chen, X., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., Zhou, J., and Lin, J. Group sequence policy optimization. CoRR, abs/2507.18071, 2025a. doi: 10.48550/ARXIV.2507.18071. URL https: //doi.org/10.48550/arXiv.2507.18071.

Zheng, L., Yin, L., Xie, Z., Sun, C. L., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

Zheng, T., Chen, L., Han, S., McCoy, R. T., and Huang, H. Learning to reason via mixture-of-thought for logical reasoning. arXiv preprint arXiv:2505.15817, 2025b.

Zheng, T., Zhang, H., Yu, W., Wang, X., Dai, R., Liu, R., Bao, H., Huang, C., Huang, H., and Yu, D. Parallelr1: Towards parallel thinking via reinforcement learning. arXiv preprint arXiv:2509.07980, 2025c.

- A. Proof of Proposition 3.2 Recall the definition of the value function given in Eq. 9:

V (Bi,πθ,pi) = 1 − e−

Bi

τ pi(1−pi) · Density(pi;αt,βt). (11) For simplicity, let us denote the terms independent of Bi as constants C and k. Let:

C = Density(pi;αt,βt) =

pα

t−1

i (1 − pi)β

t−1 B(αt,βt)

, (12)

k =

pi(1 − pi) τ

. (13)

We consider the non-trivial case where pi ∈ (0,1) and τ > 0, implying that C > 0 and k > 0. The value function can be rewritten as:

V (Bi,πθ,pi) = C 1 − e−kB

i

. (14)

- Step 1: Deriving the Marginal Gain. The marginal gain ∆V (Bi,pi) represents the increment in value obtained by increasing the budget by one unit:

∆V (Bi,pi) = V (Bi + 1,πθ,pi) − V (Bi,πθ,pi)

= C 1 − e−k(B

i+1) − C 1 − e−kB

i

= C e−kB

i

− e−k(B

i+1)

= Ce−kB

i

1 − e−k . (15)

Let A = C(1 − e−k). Since k > 0, we have e−k < 1, thus (1 − e−k) > 0. Combined with C > 0, it follows that A > 0. The marginal gain simplifies to:

∆V (Bi,pi) = A · e−kB

i

. (16)

- Step 2: Proving Monotonicity. To prove that the marginal gain is strictly decreasing, we compare ∆V (Bi,pi) with ∆V (Bi + 1,pi). We examine the ratio between consecutive marginal gains:

∆V (Bi + 1,pi) ∆V (Bi,pi)

=

A · e−k(B

i+1)

A · e−kBi =

e−k · e−kB

i

e−kBi = e−k. (17) Since k > 0, we have:

e−k < 1. (18) Therefore:

∆V (Bi + 1,pi) ∆V (Bi,pi)

< 1 =⇒ ∆V (Bi + 1,pi) < ∆V (Bi,pi). (19) This confirms that the marginal gain ∆V (Bi,pi) is a strictly decreasing geometric sequence with respect to the budget Bi.

- B. Implementation Details of Main Training Loop

Listing 1 illustrates how the CoBA-RL BudgetAllocator is integrated into the GRPO training loop. The core logic involves calculating the specific rollout count for each prompt in the current batch via a dictionary mapping and resampling the batch accordingly before the generation phase.

- 1 # Inside the GRPO training loop

- 2 for batch_dict in dataloader:

- 3 # Extract Batch

- 4 batch = process_batch(batch_dict)

- 5 indices = batch["index"]

- 6

- 7 # Execute Budget Allocation

- 8 # Returns a dictionary: {index: count}

- 9 total_budget = batch_size * default_group_size

- 10 allocation_dict = budget_allocator.allocate(

- 11 indices=indices,

- 12 total_budget=total_budget

- 13 )

- 14

- 15 # Expand batch according to allocation counts

- 16 repeat_indices = []

- 17 for idx, count in allocation_dict.items():

- 18 if count > 0:

- 19 repeat_indices.extend([idx] * count)

- 20

- 21 batch = batch[repeat_indices]

- 22

- 23 outputs = actor.generate_sequences(batch)

- 24 # ... GRPO loss calculation and update ... Listing 1. Core logic for Budget Allocation integration in GRPO.

### C. Experiment Details

We implement our training pipeline using the Verl (Sheng et al., 2024) framework, utilizing SGLang (Zheng et al., 2024) as the inference engine. For optimization, we employ the AdamW optimizer with a learning rate of 1 × 10−6. The hyperparameters are adjusted based on model scale: for models smaller than 7B, we set the global batch size M = 512 and train for approximately 500 steps; for the 7B model, we set M = 256 and extend training to nearly 1000 steps. To accommodate complex reasoning chains, we set the maximum response length to 4096 tokens. Consistent with recent trends in reasoning alignment, we do not incorporate the KL divergence penalty (i.e., βKL = 0). Regarding the specific parameters for our CoBA-RL method, we constrain the rollout budget for each instance within the range [Blow,Bup] = [2,128]. The sensitivity analysis of the parameter κ is detailed in Appendix D.2. The detailed inference hyperparameters used for evaluation are listed in Table 5. Notably, since the official implementation of the baseline Knapsack-RL is not open-sourced, we re-implement it within the Verl.

Table 5. Hyperparameters for evaluation rollout.

Hyperparameter Value

Temperature 1.0 Top-p 0.9 Sample Count (n) 16 Do Sample True

Regarding the specific implementation of budget allocation methods, following the recommendations of DAPO (Yu et al., 2025), we adopt the Clip-higher strategy for both Knapsack-RL and CoBA-RL.

### D. Additional Results

###### D.1. Analysis of Task Difficulty Transition

To evaluate performance across different learning stages, we categorize training prompts into five difficulty levels based on initial success rate (pi): extremely-hard (pi = 0), hard (0 < pi ≤ 0.2), medium (0.2 < pi < 0.8), easy (0.8 ≤ pi < 1.0), and extremely-easy (pi = 1.0). As shown in the final column of Figure 7, CoBA-RL consistently achieves the highest conversion rates. For medium tasks, it reaches 71.2%, significantly surpassing GRPO (46.8%) and Knapsack-RL (50.0%). For hard tasks, it achieves 36.7%, nearly doubling the GRPO baseline (17.3%) and outperforming Knapsack-RL (20.4%). Furthermore, it leads in converting extremely-hard tasks (8.7% vs. 4.1%) and easy tasks (88.8% vs. 74.0%), while maintaining the highest retention for extremely-easy instances (95.2%).

These results underscore the distinct superiority of CoBA-RL: By dynamically aligning resources with model capability,

###### CoBA-RL

###### GRPO

###### Knapsack-RL

100

100

100

[Figure 57]

[Figure 58]

[Figure 59]

extremely-hard

extremely-hard

extremely-hard

65.7 10.3 13.1 2.2 8.7

68.7 9.9 11.3 6.0 4.1

56.9 10.9 19.8 6.4 5.9

80

80

80

hard

hard

hard

15.8 9.1 29.9 8.5 36.7

15.0 13.3 31.6 22.7 17.3

8.3 8.7 43.9 18.7 20.4

InitialStatus

InitialStatus

InitialStatus

60

60

60

Percentage(%)

Percentage(%)

Percentage(%)

medium

medium

medium

1.8 0.8 16.1 10.1 71.2

1.2 2.2 18.5 31.3 46.8

0.5 0.7 24.4 24.5 50.0

40

40

40

easy

easy

easy

0.0 0.1 6.5 4.6 88.8

0.2 0.1 4.5 21.3 74.0

0.5 0.0 7.9 10.4 81.1

20

20

20

extremely-easy

extremely-easy

extremely-easy

0.2 0.0 2.2 2.5 95.2

0.0 0.0 1.0 7.5 91.6

0.3 0.0 3.0 2.6 94.1

0

0

0

extremely-hard hard medium easyextremely-easy

extremely-hard hard medium easyextremely-easy

extremely-hard hard medium easyextremely-easy

Final Status

Final Status

Final Status

Figure 7. Task transition matrices for Qwen2.5-7B-Instruct during training. The cell (i, j) indicates the percentage of samples transitioning from the initial status i to the final status j.

CoBA-RL maximizes learning potential across the entire difficulty spectrum, from solving novel problems to retaining established knowledge.

###### D.2. Sensitivity Analysis of κ

|[Figure 60]<br><br>0.4540 0.4661 0.4624 0.4656| | | | |
|---|---|---|---|---|
| | | | | |

[Figure 61]

0.4650

Accuracy

0.4625

0.4600

0.4575

0.4550

7 11 15 21

Figure 8. Ablation study results illustrating the impact of different sum parameter κ ∈ {7, 11, 15, 21} on model performance.

In this section, we examine the impact of the hyperparameter κ—defined as the constant sum of the Beta distribution shape parameters (α + β)—on the model training process. This parameter controls the variance of the Beta distribution, thereby influencing the ”sharpness” of the capability-oriented value function.

We evaluate the performance across four distinct values: κ ∈ {7,11,15,21}. As illustrated in Figure 8, our method exhibits strong robustness to variations in κ, with performance fluctuations remaining minimal across different settings. Specifically, the accuracy ranges from 45.40% (at κ = 7) to a peak of 46.61% (at κ = 11).

This stability validates the reliability of our capability-driven allocation criterion, suggesting that the improvement stems from the dynamic mechanism itself rather than specific hyperparameter tuning. Based on these results, we adopt κ = 11 as the default configuration for our main experiments to achieve optimal performance.

###### D.3. Visualization of Static and Heuristic Baselines

To facilitate a deeper understanding of the comparative analysis presented in the section 3.4, we provide visual illustrations of the baseline strategies in Figure 9.

Figure 9 (Left and Middle) depicts the geometric shapes of the fixed value functions used in the Static Strategies. The configuration (α,β) = (10.5,1.5) results in a distribution heavily skewed towards high success rates p, thereby enforcing an exploitation-prioritized allocation. Conversely, the setting (α,β) = (1.5,10.5) yields a distribution peaked at low success

rates, promoting an exploration-prioritized strategy.

Figure 9 (Right) illustrates the Heuristic Strategy(Linear Step Decay), where the parameter αt is annealed stepwise from 10 to 1 over the course of training. This heuristic implements a linear, stepwise decrease in α, yet it follows a rigid, pre-defined schedule that lacks the adaptability of our capability-oriented approach.

( , ) = (10.5, 1.5)

( , ) = (1.5, 10.5)

Linear Step Decay

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 0

- 1

- 2

- 3

- 4

- 5

- 6

10

8

Value

Value

6

4

2

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0 200 400 600 800

Pass rate p

Pass rate p

Training Steps

Figure 9. Visualization of the baseline strategies. Left & Middle: The probability density functions of the static Beta distributions used for exploitation ((α, β) = (10.5, 1.5)) and exploration ((α, β) = (1.5, 10.5)). Right: The pre-defined annealing schedule for α in the Linear Step Decay heuristic baseline.

