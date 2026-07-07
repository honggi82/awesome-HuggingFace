## arXiv:2509.13232v2[cs.LG]23Sep2025

[Figure 1]

# Single-stream Policy Optimization

Zhongwen Xu1* and Zihan Ding1* 1Tencent, *Equal contribution.

Abstract: We revisit policy-gradient optimization for Large Language Models (LLMs) from a singlestream perspective. Prevailing group-based methods like GRPO reduce variance with on-the-fly baselines but suffer from critical flaws: frequent degenerate groups erase learning signals, and synchronization barriers hinder scalability. We introduce Single-stream Policy Optimization (SPO), which eliminates these issues by design. SPO replaces per-group baselines with a persistent, KL-adaptive value tracker and normalizes advantages globally across the batch, providing a stable, low-variance learning signal for every sample. Being group-free, SPO enables higher throughput and scales effectively in long-horizon or tool-integrated settings where generation times vary. Furthermore, the persistent value tracker naturally enables an adaptive curriculum via prioritized sampling. Experiments using Qwen3-8B show that SPO converges more smoothly and attains higher accuracy than GRPO, while eliminating computation wasted on degenerate groups. Ablation studies confirm that SPO’s gains stem from its principled approach to baseline estimation and advantage normalization, offering a more robust and efficient path for LLM reasoning. Across five hard math benchmarks with Qwen3-8B, SPO improves the average maj@32 by +3.4 percentage points (pp) over GRPO, driven by substantial absolute point gains on challenging datasets, including +7.3 pp on BRUMO 25, +4.4 pp on AIME 25, +3.3 pp on HMMT 25, and achieves consistent relative gain in pass@𝑘 across the evaluated 𝑘 values. SPO’s success challenges the prevailing trend of adding incidental complexity to RL algorithms, highlighting a path where fundamental principles, not architectural workarounds, drive the next wave of progress in LLM reasoning.

### 1. Introduction

Reinforcement learning (RL) [36] has become a cornerstone for advancing the reasoning capabilities of Large Language Models (LLMs), notably the Reinforcement Learning with Verifiable Reward (RLVR) paradigm [11, 21]. Methods like Group Relative Policy Optimization (GRPO) [11, 34] have achieved remarkable success by adopting a multi-outcome approach, generating a group of responses for each prompt to construct an on-the-fly baseline for variance reduction. While this “group-based” paradigm has pushed the state of the art, it suffers from fundamental inefficiencies. When all responses in a group share the same outcome (e.g., all correct or all incorrect), the relative advantage collapses to zero, yielding no learning signal. This degeneracy represents a fundamental waste of computation and data. To counteract this, a series of engineering heuristics like dynamic sampling [42] have been developed. These workarounds, while functional, add significant complexity and create a less principled, more convoluted optimization process.

Group-based architectural choice also imposes a critical synchronization barrier. In distributed training, the entire group must wait for its slowest member, a bottleneck that becomes particularly acute in complex agentic tasks requiring multi-turn tool use or long-horizon reasoning [15, 41, 45]. In these settings, interaction times are highly variable (e.g., number of interaction turns, time per interaction, etc), and a single slow-running agentic trajectory can stall its entire group, severely hindering training throughput and scalability.

Corresponding authors: zhongwenxu@tencent.com, dingzihan737@gmail.com

We advocate for returning to the classic single-stream paradigm for policy gradient optimization [36], where each training sample is a single stream of prompt-response pair. This is not a mere simplification, but a deliberate re-alignment with foundational RL principles to address the aforementioned architectural flaws. To overcome the critical challenge of high gradient variance in this setting, we introduce Singlestream Policy Optimization (SPO). SPO replaces the noisy, on-the-fly group baseline with three synergistic components for stable and efficient learning. First, it employs a lightweight Bayesian value tracker to maintain a persistent, temporally-informed estimate of the success probability for each prompt, serving as a low-variance baseline. Second, it normalizes advantages globally across the entire batch, avoiding the instability of per-group statistics. Finally, this architecture naturally enables an adaptive curriculum via prioritized sampling, focusing computational resources on the most informative prompts.

The benefits of this principled approach are clear: SPO is inherently more scalable and eliminates the computational waste of degenerate groups. Our experiments confirm these advantages, demonstrating that SPO consistently outperforms GRPO on challenging reasoning benchmarks, improving the absolute point gains on challenging datasets, including 7.3 percentage points (pp) on BRUMO 25, 4.4 pp on AIME 25, 3.3 pp on HMMT 25, and the pass@𝑘 curves of SPO are above GRPO for all 𝑘s. The scalability benefit is particularly pronounced in agentic settings. Our simulations, designed to model these variabletime scenarios, show that SPO’s group-free design can achieve a 4.35× training throughput speedup by eliminating group synchronization bottlenecks. SPO thus provides a more robust foundation for modern LLM optimization, prompting a re-evaluation of essential versus incidental complexity in the field.

### 2. Related Work

Group Relative Policy Optimization (GRPO) [34] addresses the computational overhead and training instability of PPO-style algorithms [31] by eliminating the need for a separate critic network. Instead, GRPO constructs baselines on-the-fly using multiple responses generated for each prompt. Specifically, GRPO samples a group of multiple responses for each prompt and normalizes the rewards within this group to have zero mean and unit variance, creating relative advantages for policy updates. However, this approach can be inefficient if all responses in a group receive the same reward (e.g., all incorrect or all correct), resulting in a zero-advantage for all samples and providing no learning signal. To address this, DAPO [42] enhances GRPO with engineering treatments like dynamic sampling, which continues generating responses until non-zero advantages are achieved, ensuring meaningful gradients.

Several other works have proposed improvements to group-based methods. Zheng et al. [44] introduce GRESO, an online filtering algorithm that leverages reward training dynamics to predict and skip uninformative prompts before generation. Qu et al. [28] introduce a Bayesian estimation of the prompt accuracy and use it to form a bandit strategy, significantly reducing rollout overhead. Liu et al. [25] propose “Lite PPO”, which simplifies RLVR training to only advantage normalization and token-level loss aggregation.

Other group-based approaches include RLOO [1], which returns to the simpler REINFORCE [36, 39] algorithm using a Leave-One-Out baseline that treats entire generations as single actions. Similarly, Hao et al. [17] propose On-Policy RL with Optimal Baseline (OPO), which uses a length-weighted average of rewards as an optimal simplified baseline. Despite these improvements, all group-based methods share fundamental limitations. They construct baselines from concurrently generated responses rather than persistent, historical estimates, inheriting the same core architectural constraints as GRPO: synchronization overhead and increased generation costs in distributed settings.

Moving beyond group-based methods, Brantley et al. [6] propose 𝐴*-PO, a two-stage framework that achieves single-sample efficiency through a different approach. In the first stage, 𝐴*-PO performs offline estimation to approximate the optimal value function 𝑉 * rather than the policy-specific value function 𝑉𝜋. The second stage uses this pre-computed optimal value to construct optimal advantage estimates 𝐴* for a least-squares regression objective during online training. However, 𝐴*-PO has key limitations compared to our approach. It relies on a fixed, offline-computed estimate that does not adapt as the policy evolves during training. Additionally, 𝐴*-PO is constrained by KL-regularized policy optimization, which restricts how far the optimized policy can deviate from the reference policy.

### 3. Background

Reinforcement learning (RL) algorithms have been used to align Large Language Models (LLMs) with human preferences (RLHF) and to optimize verifiable reward signals (RLVR; e.g., [21, 34]).

#### 3.1. Policy Gradient and the REINFORCE Algorithm

The foundational method for this optimization is the policy gradient theorem [36, 39]. For LLMs, a trajectory consists of generating a single response 𝑦 from a prompt 𝑥. The objective function is the expected reward:

𝐽(𝜃) = E𝑥∼𝒟,𝑦∼𝜋𝜃(·|𝑥)[𝑅(𝑥,𝑦)], (1) where 𝒟 is the prompt distribution and 𝑅(𝑥,𝑦) is the reward for generating response 𝑦 for prompt 𝑥. The gradient of this objective is given by:

∇𝜃𝐽(𝜃) = E𝑥∼𝒟,𝑦∼𝜋𝜃(·|𝑥)[𝑅(𝑥,𝑦)∇𝜃 log 𝜋𝜃(𝑦|𝑥)]. (2)

This formulation gives rise to the REINFORCE algorithm [36, 39], which updates the policy by taking a step in the direction of this estimated gradient. A significant drawback of REINFORCE is the high variance of its gradient estimator. The raw reward 𝑅(𝑥,𝑦) can fluctuate widely, leading to noisy updates and unstable training.

To mitigate high variance, a baseline 𝑏(𝑥) that is conditionally independent of the action 𝑦 can be subtracted from the reward. This results in an unbiased gradient estimator with provably lower variance [16]:

∇𝜃𝐽(𝜃) = E𝑥∼𝒟,𝑦∼𝜋𝜃(·|𝑥)[(𝑅(𝑥,𝑦) − 𝑏(𝑥))∇𝜃 log 𝜋𝜃(𝑦|𝑥)]. (3) The term 𝐴(𝑥,𝑦) = 𝑅(𝑥,𝑦) − 𝑏(𝑥) is known as the advantage. The optimal baseline that minimizes variance is the true value function 𝑉𝜋(𝑥) = E𝑦∼𝜋𝜃(·|𝑥)[𝑅(𝑥,𝑦)], which is the expected reward for a given prompt 𝑥. In practice, 𝑉𝜋(𝑥) is unknown and must be estimated. The quality of this estimation is crucial for the stability and efficiency of the RL algorithm.

#### 3.2. Variance Reduction Baselines for Large Language Models

Several strategies have been developed to estimate the baseline 𝑏(𝑥) in the context of LLM training. PPO [31] trains a parameterized critic network 𝑣𝜑. However, learning 𝑣𝜑 is notoriously unstable and resource-intensive, as 𝜑 typically matches the size of the LLM policy parameters 𝜃.

A common approach is to construct an empirical, on-the-fly baseline from multiple samples. Group Relative Policy Optimization (GRPO) [11, 34] generates a group of 𝐺 responses {𝑦1,...,𝑦𝐺} for a single prompt 𝑥, then uses the mean rewards of the group as the baseline 𝑏GRPO. Another popular baseline is the Leave-One-Out (RLOO). For a given sample 𝑦𝑖, the baseline is the average reward of the other 𝐺 − 1 samples in the group, denoted as 𝑏RLOO:

1 𝐺 ∑︁

1 𝐺 − 1 ∑︁

𝑅(𝑥,𝑦𝑗). (4)

𝑏GRPO(𝑥) =

𝑅(𝑥,𝑦𝑗), 𝑏RLOO(𝑥,𝑦𝑖) =

𝑗

𝑗̸=𝑖

The raw advantage for sample 𝑦𝑖 is then 𝐴(𝑥,𝑦𝑖) = 𝑅(𝑥,𝑦𝑖) − 𝑏GRPO(𝑥), then it is normalized with the standard deviation 𝜎𝐺. While simple to implement, this approach suffers from two key limitations. First, it is sample-inefficient, requiring 𝐺 > 1 generations per prompt for each gradient step. Second, the baseline is estimated from a very small group (𝐺), making it a high-variance estimate of the true value function, which in turn leads to noisy advantage estimates.

### 4. Method

We introduce Single-stream Policy Optimization (SPO), a method designed for policy optimization in settings with verifiable feedback (RLVR) [21]. We assume the feedback is binary1, i.e., +1 for success and 0 for failure. SPO addresses the challenge of estimating a non-stationary success probability for a policy that evolves over training iterations. It integrates a Bayesian value tracker with an adaptive memory mechanism into a policy gradient framework. The core components are: (1) a KL-adaptive tracker that provides a low-variance, single-sample estimate of the success probability; (2) a global advantage normalization scheme that ensures high sample efficiency and stable learning dynamics; and (3) prioritized sampling across training prompts to focus on prompts with high learning potential. The following subsections detail each component.

#### 4.1. A KL-Adaptive Value Tracker

The definition of a value function is the expected reward of the prompt 𝑥 under policy 𝜋, i.e., 𝑉𝜋(𝑥) = E𝑦∼𝜋(·|𝑥)[𝑅(𝑥,𝑦)]. We use 𝑣ˆ(𝑥) to denote the tracker’s running estimate of 𝑉𝜋(𝑥); that is, 𝑣ˆ(𝑥) ≈ 𝑉𝜋(𝑥). To estimate the non-stationary success probability of a prompt 𝑥, we use a Bayesian tabular tracker instead of a separate value network2. For the binary success/failure rewards common in RLVR, this is elegantly modeled using a Beta distribution, which is the conjugate prior for the Bernoulli process governing the outcomes. We therefore model the success probability 𝑣ˆ(𝑥) using a Beta distribution: 𝑣ˆ(𝑥) ∼ Beta(𝛼(𝑥),𝛽(𝑥)), where the value estimate is the posterior mean 𝑣ˆ(𝑥) = 𝛼(𝑥)/(𝛼(𝑥) + 𝛽(𝑥)).

The tracker adapts to policy changes by dynamically adjusting its memory of past rewards. When the policy changes significantly, older observations become less relevant and should be downweighted. After each new observation 𝑟(𝑥,𝑦) ∈ {0,1}, we discount the prior Beta parameters (𝛼−1,𝛽−1) by a factor 𝜌(𝑥) before incorporating the new evidence 𝑟(𝑥,𝑦):

𝛼(𝑥) 𝛼(𝑥) + 𝛽(𝑥)

. (5)

𝛼(𝑥) = 𝜌(𝑥)𝛼−1(𝑥) + 𝑟(𝑥,𝑦), 𝛽(𝑥) = 𝜌(𝑥)𝛽−1(𝑥) + (1 − 𝑟(𝑥,𝑦)), 𝑣ˆ(𝑥) =

1Generalizing to non-binary rewards is straightforward, as discussed at the end of Section 4.1. 2The development of core RL algorithms was on tabular representation [36].

The discount factor 𝜌(𝑥) = 2−𝐷(𝑥)/𝐷half is determined by the KL divergence 𝐷(𝑥) between the current policy and the last policy that acted on prompt 𝑥, causing the tracker to forget faster as the policy changes more significantly. The hyperparameter 𝐷half controls this forgetting rate 𝜌 ∈ [𝜌min,𝜌max].

Initialization. To initialize, we collect 𝑛0 samples to compute an initial value estimate 𝑣ˆ0(𝑥). To avoid transient instability, we set the initial effective sample size to its expected equilibrium, 𝑁0 = 1/(1−𝜌min), where 𝜌min is the minimum allowed forgetting factor. The initial parameters are then:

𝛼0(𝑥) = 𝑁0 · 𝑣ˆ0(𝑥), 𝛽0(𝑥) = 𝑁0 · (1 − 𝑣ˆ0(𝑥)). (6)

This Bayesian update is equivalent to an adaptive Exponential Moving Average (EMA) on the value estimate:

𝑣ˆ(𝑥) = 𝑣ˆ−1(𝑥) + 𝜂(𝑥)(𝑟(𝑥,𝑦) − 𝑣ˆ−1(𝑥)), (7)

where the learning rate 𝜂(𝑥) = (𝜌(𝑥)𝑁eff,−1(𝑥) + 1)−1 naturally adapts to both policy shifts (via 𝜌(𝑥)) and statistical confidence (via 𝑁eff = 𝛼(𝑥) + 𝛽(𝑥) + 1). This formulation highlights how our tracker balances new evidence against accumulated knowledge. For general rewards beyond binary ones, we can just use the same EMA formulation to directly track 𝑣ˆ, rather than relying on 𝛼 and 𝛽 in the binary cases.

#### 4.2. Advantage Estimation and Policy Optimization

SPO uses the tracker’s estimate 𝑣ˆ as a baseline for advantage calculation in a policy gradient algorithm. At iteration 𝑖, for a single reward 𝑟(𝑥,𝑦) obtained with policy 𝜋𝜃𝑖, the advantage is computed using the pre-update baseline (denoted with subscript −1):

𝐴(𝑥,𝑦) = 𝑟(𝑥,𝑦) − 𝑣ˆ−1(𝑥). (8) Using the baseline from the previous step ensures that it is independent of the action taken at step 𝑖, preserving the unbiasedness of the policy gradient estimate. While the reward 𝑟(𝑥,𝑦) is typically a direct outcome signal, SPO’s framework is also compatible with more sophisticated reward functions. For instance, recent work like InfAlign [4] demonstrates how to calibrate and transform the reward signal to be “inference-aware,” directly optimizing for procedures like Best-of-𝑁 sampling. Such transformed rewards can be seamlessly integrated into SPO by replacing the standard 𝑟(𝑥,𝑦) in the advantage calculation. Since 𝑣−1(𝑥) is independent of 𝑦 ∼ 𝜋𝜃𝑖(·|𝑥), E[(𝑟 − 𝑣𝑖−1(𝑥))∇𝜃 log 𝜋] = ∇𝐽(𝜃) [39]. Instead of normalizing advantages on a per-prompt basis in a group [11, 34], SPO normalizes them across an entire batch of prompts ℬ [3, 19, 23, 31]. The normalized advantage 𝐴˜(𝑥,𝑦) is computed as:

𝐴(𝑥,𝑦) − 𝜇ℬ 𝜎ℬ

, (9)

𝐴˜(𝑥,𝑦) =

where 𝜇ℬ and 𝜎ℬ are the mean and standard deviation of advantages in the batch {𝐴(𝑥,𝑦)}𝑥∈ℬ. We then apply the advantage 𝐴˜(𝑥,𝑦) to each token in the response sequence 𝑦 and update the policy parameters using a standard PPO-Clip policy loss [31]3:

, 1 − 𝜀, 1 + 𝜀)︁𝐴˜𝑡)︃]︃. (10)

𝐿CLIP(𝜃) = E𝑠,𝑡[︃min(︃ 𝜋𝜃(𝑎𝑡 | 𝑠𝑡) 𝜋𝜃old(𝑎𝑡 | 𝑠𝑡)

𝐴˜𝑡, clip(︁ 𝜋𝜃(𝑎𝑡 | 𝑠𝑡) 𝜋𝜃old(𝑎𝑡 | 𝑠𝑡)

3The term “PPO” is frequently used with ambiguity. It may denote the entire algorithm suite (e.g., clipped policy and value losses), refer narrowly to just the clipped policy objective, or describe the broader training framework, including mechanisms like mini-batch updates.

- Algorithm 1 Single-stream Policy Optimization

- 1: for iteration 𝑖 = 1,2,...,𝑇 do
- 2: For each 𝑥 ∈ 𝒳, compute sampling weight 𝑤𝑖(𝑥) according to Eqn. (11).
- 3: Sample a batch of 𝐵 prompts ℬ𝑖 ⊂ 𝒳 according to weights {𝑤𝑖(𝑥)}.
- 4: 𝒟 ← ∅
- 5: for each prompt 𝑥 ∈ ℬ𝑖 do
- 6: Sample action 𝑦 ∼ 𝜋𝜃𝑖−1(· | 𝑥) and observe reward 𝑟(𝑥,𝑦) ∈ {0,1}.
- 7: Compute raw advantage 𝐴(𝑥,𝑦) ← 𝑟(𝑥,𝑦) − 𝑣ˆ−1(𝑥).
- 8: Store (𝑥,𝑦,𝐴(𝑥,𝑦)) in 𝒟.
- 9: Update tracker 𝑣ˆ(𝑥).
- 10: Normalize advantages: 𝐴˜(𝑥,𝑦) ← (︀

𝐴(𝑥,𝑦) − 𝜇ℬ𝑖

)︀

/𝜎ℬ𝑖.

- 11: Update 𝜃𝑖−1 to 𝜃𝑖 using mini-batches with a policy gradient algorithm (e.g., PPO-Clip).

Methods like Clip-Higher [42], Clip-Cov [10] and KL-Cov [10] to retain policy entropy are applicable here. Other policy optimization algorithms like CISPO [27] (similar to vtrace [12, 40]) and GSPO [43] (use sequence-level likelihood instead of token-level) are compatible with our advantage estimator. Advanced methods to control policy behaviors like ASPO [22] can be utilized to modulate the advantage values. We note that if we use “no baseline” (i.e., 𝑣ˆ = 0), it is an extremely simple and valid algorithm but may suffer from high policy gradient variance.

#### 4.3. Prioritized Prompt Sampling

To further enhance data efficiency, SPO employs a curriculum learning strategy by prioritizing prompts with the highest learning potential [30, 36]. At each iteration, we sample a batch of prompts based on a score that emphasizes prompts with high uncertainty, while ensuring a minimum level of exploration. The sampling weight 𝑤𝑖(𝑥) for prompt 𝑥 is defined as:

𝑤𝑖(𝑥) ∝ √︁𝑣ˆ−1(𝑥)

(︀

)︀

+ 𝜖. (11)

1 − 𝑣ˆ−1(𝑥)

The first term corresponds to the estimated standard deviation of a Bernoulli outcome, which naturally allocates more weight to prompts that are neither almost always solved (𝑣ˆ ≈ 1) nor almost always failed (𝑣ˆ ≈ 0). The exploration bonus 𝜖, set to 0.05 by default, prevents curriculum collapse by ensuring that every prompt retains a non-zero probability of being sampled, thereby maintaining broad coverage of the data distribution. The complete SPO training procedure is outlined in Algorithm 1.

#### 4.4. Advantages over GRPO

Group-Free for Scalable Infrastructure. SPO’s design is inherently “group-free”, a significant advantage in distributed training frameworks for LLMs. Each sample, consisting of a single stream of (prompt, response) pair, is a self-contained data point for the policy update. GRPO, however, requires the generation and evaluation of an entire group of 𝐺 samples for a single prompt before any training signal can be computed. We provide our illustrations in Figure 1. In a distributed setting, this introduces a synchronization barrier: the processing of a given prompt is not complete until all 𝐺 responses have been generated. This is particularly problematic in the presence of long-tail generation times, where a single slow response generation can stall the processing for its entire group. For constructing a training

##### GRPO

- 𝑦1

- 𝑦2

- 𝑟1

- 𝑟2

- 𝐴1

- 𝐴2

Group Norm

𝑥 𝜋

…

…

…

𝑦𝐺

𝑟𝐺

𝐴𝐺

##### SPO

𝑥 𝜋 𝑦 𝑟 𝐴

𝑣ො

- Figure 1: Illustrations of GRPO and the proposed SPO.

batch, SPO only needs to collect 𝐵 independent (prompt, response) pairs, which is far more flexible and efficient than waiting for 𝐵 entire groups to complete. This makes SPO’s architecture significantly more infrastructure-friendly and scalable. The advantage is amplified in agentic training, especially in settings that require multi-turn interactions with tools [9, 15] or long-horizon agent rollouts [41, 45]. The scale of these interactions can be substantial: state-of-the-art open-source models (gpt-oss-120b) may average 20 search turns per task [9], with other agentic sessions reaching over 40 tool calls and generating up to 150,000 tokens of context [15].

Adaptive Curriculum. To further enhance training efficiency, SPO integrates a prioritized sampling scheme. This mechanism naturally creates an adaptive curriculum by focusing computational resources on prompts with the highest learning potential. This ensures that the model’s training is concentrated on the most informative examples at any given point in time. GRPO, in its standard formulation, typically relies on uniform sampling of prompts. This may waste computation on prompts that are already mastered or are currently too difficult to yield useful learning signals. While dynamic sampling [42] and repeat strategies [2] have been proposed to mitigate this issue, they often discard samples after generation, wasting computation. SPO’s prioritized sampling addresses the scheduling problem before response generation, leading to a more natural and efficient training process.

More discussions on the inefficiency of dynamic sampling and the variance reduction of policy gradient are outlined in Appendix C, where we provide detailed analysis.

### 5. Experiments

#### 5.1. Experimental Setup

The SPO algorithm is broadly applicable in LLM reasoning tasks [11] and Agentic training. We evaluate Tool-Integrated Reasoning (TIR) [13, 22] scenarios, where the LLMs can utilize external Python interpreter to help solve hard problems. We conduct experiments using a moderately sized LLM, Qwen38B [29]. For training data, we use the English subset from the DAPO dataset [42]. Only outcome reward is applied for RLVR, without the format rewards. We evaluate performance on the challenging math

competition benchmarks, i.e., AIME 24, AIME 25, BeyondAIME [32], BRUMO 25 [5], and HMMT 25 [5]. See Appendix D for training and evaluation details.

We distinguish our goal from that of “hill-climbing” on benchmark leaderboards. The latter often necessitates resource-intensive and highly specialized techniques, including SFT from frontier models [24], mid-training [38], multi-stage RL pipelines [8, 18, 26], curated hard datasets with intricate processing [2, 33], test-time scaling techniques [14] and extremely large generation group sizes [45]. Our work, instead, concentrates on the fundamental efficiency and scalability of the RL algorithm itself.

#### 5.2. Empirical Comparison with GRPO

Table 1: Comparison of GRPO and SPO on five benchmarks using maj@32 and avg@32. Averages are shown in the last column. Bold indicates the better-performing method for each metric.

AIME 24 AIME 25 BeyondAIME BRUMO 25 HMMT 25 Average maj@32 avg@32 maj@32 avg@32 maj@32 avg@32 maj@32 avg@32 maj@32 avg@32 maj@32 avg@32

Method

Qwen3-8B 77.8 64.4 70.5 58.4 45.2 38.0 55.1 49.4 36.8 30.3 57.1 48.1 GRPO 83.3 77.6 72.1 64.2 45.6 39.0 56.7 56.9 44.2 40.9 60.4 55.7 SPO 84.0 74.9 76.5 65.0 46.9 40.3 64.0 59.0 47.5 40.6 63.8 56.0

0.90

SPO

GRPO

0.88

pass@k

0.86

8 16 32 k

(a) AIME 24

0.80

SPO

GRPO

0.78

0.76

pass@k

0.74

0.72

0.70

8 16 32 k

(d) BRUMO 25

SPO

0.86

GRPO

pass@k

0.84

0.82

8 16 32 k

(b) AIME 25

SPO

0.56

GRPO

0.54

pass@k

0.52

0.50

8 16 32 k

(e) HMMT 25

0.66

SPO

GRPO

0.64

0.62

pass@k

0.60

0.58

0.56

8 16 32 k

(c) BeyondAIME

0.76

SPO

GRPO

0.74

pass@k

0.72

0.70

0.68

8 16 32

k

(f) Average

- Figure 2: Pass@𝑘 plots comparing GRPO and SPO across five math competition benchmarks.

Our experiments demonstrate that SPO outperforms the GRPO baseline on aggregate metrics when training the Qwen-8B model. As shown in Table 1, SPO achieves superior weighted average scores on both primary metrics. It obtains a maj@32 of 63.8 compared to GRPO’s 60.4, a significant improvement of +3.4 percentage points (pp). This aggregate strength is driven by remarkable consistency, as SPO outperforms GRPO on the maj@32 metric across all five benchmarks. The performance gap is most pronounced on BRUMO 25, where SPO achieves a substantial +7.3 pp (64.0 vs. 56.7). Further significant gains are seen on AIME 25 (+4.4 pp) and HMMT 25 (+3.3 pp points), underscoring the robustness of SPO’s improvements. Notably, these benchmarks have minimal data contamination [5], allowing them to serve as a true test of generalization. This demonstrates that our SPO method improves the model’s ability

to generalize rather than simply overfit to the training data, a risk exemplified by the DAPO dataset’s strong correlation with AIME 24. While GRPO remains competitive on the avg@32 metric in some cases, SPO’s consistent and significant advantage in maj@32 suggests it learns more robust and repeatable solutions, a key goal for reliable reasoning models.

These findings are mirrored in the pass@𝑘 performance shown in Figure 2. The weighted average curve (Figure 2f) shows a clear and consistent advantage for SPO across all values of 𝑘, translating to an average improvement of approximately 2.4 pp. While the performance on avg@32 is more competitive on a per-benchmark basis, SPO’s strong overall performance underscores the stability and effectiveness of its learning signal. We provide additional ablation studies on 𝐴*-PO, SPO with no baseline, and SPO with no offline initialization in Appendix E.

#### 5.3. Analysis of Signal Efficiency and Stability

0.8

DegenerateGroupsRatio

0.6

GRPO

0.4

- SPO ( =0.0001)

- SPO ( =0.02)

0.2

0.0

0 20 40 60 80 100 120 140

Training Steps

(a) Ineffective Gradient Ratios

0.18

AdvantageVariance

0.16

0.14

SPO

0.12

SPO No Baseline

GRPO Effective

0.10

0.08

0 20 40 60 80 100 120 140

Training Steps

(b) Advantage Variance Comparison

- Figure 3: Signal Efficiency and Stability Analysis of SPO vs. GRPO. (a) GRPO suffers from a high ratio of degenerate groups (blue), which yield no learning signal. In contrast, SPO’s rate of near-zero advantages (red/green) increases as the model learns, reflecting prediction accuracy rather than wasted computation. (b) SPO’s baseline (red) provides a stable, low-variance signal, significantly reducing the raw reward variance (green). GRPO’s effective advantage (blue), calculated only on non-degenerate samples, is highly volatile and unstable.

To empirically assess the architectural advantages of SPO, we conduct a two-part analysis of the unnormalized advantage signals produced by SPO and GRPO (Figure 3). First, we quantify complete signal loss arising from degenerate groups. Second, we measure the variance of the remaining learning signals. Together, these metrics characterize each method’s efficiency and stability.

Signal Efficiency and Information Loss. Figure 3a reports the fraction of ineffective samples. For GRPO (blue), the share of samples in degenerate groups rises from roughly 60% to over 80%, yielding zero advantage and no gradient. For SPO, we instead track the proportion of near-zero advantages under two diagnostic tolerances, |𝐴| ≤ 𝜏, with values of 𝜏 = 10−4 (red) and 𝜏 = 0.02 (green). Advantages under the tight tolerance 𝜏 = 10−4 remain rare throughout training (red line), while the |𝐴| ≤ 0.02 share (green) gradually increases as the value tracker 𝑣ˆ becomes more accurate and residuals shrink on mastered prompts. This trend is expected and desirable: it reflects accurate prediction rather than signal loss. Unlike GRPO’s degenerate groups, these SPO samples are not discarded, they still produce well-defined gradients and contribute to learning. Notably, even under the stricter 𝜏 = 0.02 tolerance,

SPO’s near-zero ratio remains far below GRPO’s degenerate rate, underscoring SPO’s efficient use of compute.

Signal Stability and Advantage Variance. Figure 3b compares advantage variance across methods. As a reference, the green line (“SPO No Baseline”) corresponds to raw rewards, i.e., the high-variance signal faced by vanilla policy gradient. SPO’s history-informed baseline (red) delivers a substantial, stable variance reduction of nearly 50%. For GRPO, computing variance only over non-degenerate samples (“GRPO Effective”, blue) reveals a highly volatile signal with the largest variance among all conditions, exceeding even “SPO No Baseline”. We conclude that SPO’s baseline is effective, yielding stable, lowvariance gradients, whereas GRPO’s on-the-fly baseline is noisy and destabilizing when it produces a signal. The apparent stability of GRPO’s overall variance is driven by the prevalence of zero-variance degenerate samples and thus reflects inefficiency rather than robustness.

#### 5.4. Agentic Training Demonstrations

We perform simulations to demonstrate the practical implications of SPO’s group-free design in agentic training scenarios, where interaction times can be highly variable. Group-based methods like GRPO suffer from a critical scalability bottleneck due to their inherent synchronization barrier, a problem that is particularly acute in agentic tasks involving multi-turn tool use or long-horizon reasoning.

- Figure 4 illustrates this fundamental issue. In an idealized low-variance setting (Figure 4a), where all agentic trajectories complete in similar times, the group-based approach is efficient. However, in a more realistic high-variance setting (Figure 4b) characterized by long-tail latencies, a single slow-running trajectory (a “straggler”) can stall the entire group. In our simulation, while most samples finish in under 133 seconds, the group must wait 508 seconds for its slowest member. This bottleneck effect forces faster samples to remain idle, severely hindering training throughput and wasting computational resources.

Group completion: 133s

500

TotalInteractionTime

400

300

Low Variance Predictable

200

133s

116s 112s

114s

112s

95s

94s

100

62s

1 2 3 4 5 6 7 8

Sample Index

(a) Low-variance Group

Group completion: 508s

508s

500

444s

TotalInteractionTime

409s

400

300

High Variance Bottleneck Effect

200

114s 117s

90s

87s

100

74s

1 2 3 4 5 6 7 8

Sample Index

(b) High-variance Group

- Figure 4: The Bottleneck Effect in Group-Based Sampling. (a) In a low-variance environment, sample completion times are predictable, and the group synchronization cost is minimal. (b) In a realistic highvariance agentic environment, three slow trajectories (444𝑠, 508𝑠, and 409𝑠) create a severe bottleneck, forcing the entire group to wait and wasting the compute used for the six faster samples.

SPO’s group-free architecture directly resolves this inefficiency. Figure 5 compares the time required to assemble a training batch of 24 samples using both strategies. The group-based approach (left), even

600

553s

519s

495s 486s

GroupCompletionTime

500

400

Unselected groups

Selected (3 fastest)

300

Completion: 486s

200

130s 133s

100

0

1 2 3 4 5 6

Group Index

(a) Group-base

500

TotalInteractionTime

400

Unselected samples

300

Selected (24 fastest)

Completion: 112s

200

100

0 10 20 30 40 50

Sample Index

(b) Group-free

486s

500

TotalCompletionTime

400

Speedup: 4.35x

300

200

112s

100

0

Group-Based Group-Free

(c) Strategy Comparison

- Figure 5: Throughput Comparison: Group-Based vs. Group-Free. (a) A group-based strategy, even when parallelized, is bottlenecked by its slowest group, taking 486𝑠 to collect a batch of 3 groups (24 samples). (b) A group-free strategy collects the 24 fastest samples from a larger pool of 48, completing the batch in just 112𝑠 by avoiding stragglers. (c) The group-free approach achieves a 4.35× speedup, demonstrating its superior efficiency for agentic training.

when optimized by running 6 groups in parallel and selecting the 3 fastest, is still constrained by the slowest trajectory within those selected groups, taking 486𝑠 to complete. In contrast, the group-free approach (middle) leverages asynchrony by starting 48 independent samples and simply collecting the first 24 to finish. In our simulated scenario, this process takes only 112𝑠, as it naturally filters out the slow outliers. As shown on the right, this architectural difference results in a significant 4.35× speedup in this realistic agentic simulation. Simulations show that SPO’s architecture can lead to significant throughput gains, making it a more scalable and robust foundation for training on complex, long-horizon agentic tasks.

### 6. Conclusions

We identified critical inefficiencies in group-based policy optimization methods for LLMs, namely computational waste from degenerate groups and scalability bottlenecks from synchronization. To address these, we proposed Single-stream Policy Optimization (SPO), a principled return to the classic single-stream paradigm. SPO replaces the noisy, per-group baseline with a persistent KL-adaptive value tracker and global advantage normalization, creating a more stable and efficient learning signal.

Our empirical results demonstrate that SPO’s design is not merely simpler, but superior. It consistently outperformed GRPO on complex reasoning tasks while eliminating the systemic flaws of its group-based counterpart. By demonstrating that a well-designed single-stream approach can surpass more complex methods, our work challenges the prevailing trend of adding incidental complexity to RL algorithms for LLMs. SPO provides a robust, scalable, and efficient foundation for future research in agentic and reasoning model training, highlighting the enduring power of foundational reinforcement learning principles. Future work can focus on refining the best practices for applying SPO and exploring its limits, pushing its effectiveness to power the next generation of reasoning and agentic LLMs.

### References

- [1] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in LLMs. arXiv preprint arXiv:2402.14740, 2024.
- [2] Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu, Mingxuan Wang, and Lingpeng Kong. POLARIS: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025. URL https://hkunlp.github. io/blog/2025/Polaris.
- [3] Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphael Marinier, Léonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, et al. What matters in onpolicy reinforcement learning? A large-scale empirical study. arXiv preprint arXiv:2006.05990, 2020.
- [4] Ananth Balashankar, Ziteng Sun, Jonathan Berant, Jacob Eisenstein, Michael Collins, Adrian Hutter, Jong Lee, Chirag Nagpal, Flavien Prost, Aradhana Sinha, et al. InfAlign: Inference-aware language model alignment. arXiv preprint arXiv:2412.19792, 2024.
- [5] Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, and Martin Vechev. MathArena: Evaluating LLMs on uncontaminated math competitions. arXiv preprint arXiv:2505.23281, 2025.
- [6] Kianté Brantley, Mingyu Chen, Zhaolin Gao, Jason D Lee, Wen Sun, Wenhao Zhan, and Xuezhou Zhang. Accelerating RL for LLM reasoning with optimal advantage regression. arXiv preprint arXiv:2505.20686, 2025.
- [7] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [8] Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acereason-Nemotron: Advancing math and code reasoning through reinforcement learning. arXiv preprint arXiv:2505.16400, 2025.
- [9] Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, et al. BrowseComp-Plus: A more fair and transparent evaluation benchmark of deep-research agent. arXiv preprint arXiv:2508.06600, 2025.
- [10] Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.
- [11] Team DeepSeek. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645:633–638, 2025.
- [12] Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures. In International conference on machine learning, pages 1407–

1416. PMLR, 2018.

- [13] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. ReTool: Reinforcement learning for strategic tool use in LLMs. arXiv preprint arXiv:2504.11536, 2025.
- [14] Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. Deep think with confidence. arXiv preprint arXiv:2508.15260, 2025.
- [15] Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and Yi Wu. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous RL. arXiv preprint arXiv:2508.07976, 2025.
- [16] Evan Greensmith, Peter L Bartlett, and Jonathan Baxter. Variance reduction techniques for gradient estimates in reinforcement learning. Journal of Machine Learning Research, 5(Nov):1471–1530, 2004.
- [17] Yaru Hao, Li Dong, Xun Wu, Shaohan Huang, Zewen Chi, and Furu Wei. On-policy RL with optimal reward baseline. arXiv preprint arXiv:2505.23585, 2025.
- [18] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. SkyWork Open Reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025.
- [19] Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. REINFORCE++: An efficient RLHF algorithm with robustness to both prompt and reward models. arXiv preprint arXiv:2501.03262, 2025.
- [20] Team Kimi. Kimi K1.5: Scaling reinforcement learning with LLMs. arXiv preprint arXiv:2501.12599, 2025.
- [21] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.
- [22] Heng Lin and Zhongwen Xu. Understanding Tool-Integrated Reasoning. arXiv preprint arXiv:2508.19201, 2025.
- [23] Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen, Diyi Yang, Wee Sun Lee, and Min Lin. GEM: A gym for generalist LLMs, 2025. URL https://axon-rl.notion.site/gem.
- [24] Zihan Liu, Zhuolin Yang, Yang Chen, Chankyu Lee, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. AceReason-Nemotron 1.1: Advancing math and code reasoning through SFT and RL synergy. arXiv preprint arXiv:2506.13284, 2025.
- [25] Zihe Liu, Jiashun Liu, Yancheng He, Weixun Wang, Jiaheng Liu, Ling Pan, Xinyu Hu, Shaopan Xiong, Ju Huang, Jian Hu, et al. Part I: Tricks or traps? A deep dive into RL for LLM reasoning. arXiv preprint arXiv:2508.08221, 2025.
- [26] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. DeepScaleR: Surpassing o1-Preview with a 1.5B model by scaling RL. https://pretty-radio-b75.notion.site/DeepScaleR-Sur passing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8c a303013a4e2, 2025. Notion Blog.

- [27] Team MiniMax. MiniMax-M1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.
- [28] Yun Qu, Qi Wang, Yixiu Mao, Vincent Tao Hu, Björn Ommer, and Xiangyang Ji. Can prompt difficulty be online predicted for accelerating RL finetuning of reasoning models? arXiv preprint arXiv:2507.04632, 2025.
- [29] Team Qwen. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [30] Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
- [31] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [32] Team Seed. Seed1.5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025.
- [33] Ning Shang, Yifei Liu, Yi Zhu, Li Lyna Zhang, Weijiang Xu, Xinyu Guan, Buze Zhang, Bingcheng Dong, Xudong Zhou, Bowen Zhang, et al. rStar2-Agent: Agentic Reasoning Technical Report. arXiv preprint arXiv:2508.20722, 2025.
- [34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [35] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A flexible and efficient RLHF framework. arXiv preprint arXiv: 2409.19256, 2024.
- [36] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.
- [37] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/forum?id=ySyClPaTKAq.
- [38] Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. OctoThinker: Mid-training incentivizes reinforcement learning scaling. arXiv preprint arXiv:2506.20512, 2025.
- [39] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.
- [40] Bo Wu, Sid Wang, Yunhao Tang, Jia Ding, E1yk Helenowski, Liang Tan, Tengyu Xu, Tushar Gowda, Zhengxing Chen, Chen Zhu, et al. LlamaRL: A distributed asynchronous reinforcement learning framework for efficient large-scale LLM training. arXiv preprint arXiv:2505.24034, 2025.
- [41] Zhongwen Xu, Xianliang Wang, Siyi Li, Tao Yu, Liang Wang, Qiang Fu, and Wei Yang. Agents play thousands of 3D video games. arXiv preprint arXiv:2503.13356, 2025.

- [42] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [43] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
- [44] Haizhong Zheng, Yang Zhou, Brian R Bartoldson, Bhavya Kailkhura, Fan Lai, Jiawei Zhao, and Beidi Chen. Act only when it pays: Efficient reinforcement learning for LLM reasoning via selective rollouts. arXiv preprint arXiv:2506.02177, 2025.
- [45] Team Zhipu. GLM-4.5: Agentic, reasoning, and coding (ARC) foundation models. 2025.

### A. SPO Initialization

We show the SPO initialization procedure in Algorithm 2. In the experiments, we use 𝑛0 = 8 to have a good estimation of initial baseline tracker. We ablate the setting where we use no offline estimation and rely on the online moving estimator in Section E.

Algorithm 2 SPO Initialization

- 1: Set initial effective sample size 𝑁0 = 1/(1 − 𝜌min).
- 2: for each prompt 𝑥 ∈ 𝒳 do
- 3: Collect 𝑛0 outcomes {𝑟(𝑘)}𝑛𝑘=10 with an initial policy 𝜋0.
- 4: Compute initial value estimate 𝑣ˆ0(𝑥) = 𝑛1

0 ∑︀𝑛

0

𝑘=1 𝑟(𝑘).

- 5: Set 𝛼0(𝑥) = 𝑁0 · 𝑣ˆ0(𝑥) and 𝛽0(𝑥) = 𝑁0 · (1 − 𝑣ˆ0(𝑥)).

Practically, one may concern about the extra cost during the offline estimation of 𝑣ˆ0. We note that we share the offline estimation for our experiments so that people could skip this process and directly load our datasets, and there are datasets like Polaris [2] that pre-compute accuracy for Deepseek-R1-DistillQwen-7B [11]. The cost can be amortized across the experiments people run themselves, and we will share more (dataset, base_model) combinations to facilitate experiment efficiency.

### B. Batch Extensions

We could adapt Single-stream Policy Optimization (SPO) into a prompt-repetition scheme4, processing each prompt 𝐺 times per batch with a shared baseline estimator 𝑣ˆ to better handle sparse rewards. Our method’s primary advantage over GRPO lies in its asynchronous nature, achieved by removing the group synchronization barrier. Treating repeated prompts as independent trajectories unlocks two key efficiency improvements. First, it enables robust handling of long-tail generation issues, as slow or problematic trajectories can be terminated early, discarded, or managed via partial rollouts [20] without delaying the entire batch. Second, it facilitates a more flexible batching strategy. By over-sampling the number of initial prompts (e.g., by 50%), a full training batch can be assembled from the first-finishing trajectories, allowing the optimization step to proceed immediately without waiting for stragglers. This design significantly reduces training latency compared to the rigid group synchronization required by GRPO. When tackling hard prompts, the batch extensions may help obtain learning signals more quickly.

### C. Comparisons against GRPO

#### C.1. Inefficiency of Dynamic Sampling

To address the information loss from degenerate sample groups (where all rewards are identical), methods like DAPO [42] employ dynamic sampling. This strategy continues generating responses for a prompt until the collected set contains at least one success and one failure, guaranteeing a non-zero advantage. While effective at ensuring a learning signal, this approach can be extremely data- and time-inefficient. Note that when people report performance with dynamic sampling, the “steps” indicate the learning steps rather than the sampling steps, where the latter is normally a multiple of the former (e.g., 5×).

4Batch SPO or BSPO

We can formalize the expected computational cost. For a prompt 𝑥 with true success probability 𝑝 = 𝑉𝜋(𝑥), let 𝑁 be the number of samples required to obtain a non-degenerate set. We have:

E[𝑁 | 𝑝] = 𝑝(︁1 + 1−1𝑝)︁ + (1 − 𝑝)(︁1 + 𝑝1)︁ =

1 𝑝(1 − 𝑝) − 1.

This cost grows hyperbolically as the policy becomes either proficient (𝑝 → 1) or incompetent (𝑝 → 0). For example, if a policy has a 10% success rate (𝑝 = 0.1), the expected number of generations needed to collect both a success and a failure is E[𝑁] ≈ 10.11. In contrast, SPO requires exactly one sample per prompt and uses its adaptive curriculum to actively de-prioritize these inefficient prompts, allocating resources to where learning is most effective. This makes SPO fundamentally more scalable and computationally efficient.

#### C.2. Variance Reduction for Policy Gradient

The per-sample policy gradient is 𝑔 = 𝐴(𝑥,𝑦)∇𝜃 log 𝜋𝜃(𝑦|𝑥), where the advantage 𝐴 is an estimate of the expected return over a baseline. The variance of this gradient, Var[𝑔], is a key driver of training efficiency. We analyze how the construction of the advantage 𝐴 leads to significant variance differences between GRPO and SPO.

GRPO’s High-Variance Group-Based Advantage: GRPO computes advantages by comparing outcomes within a small group of 𝐺 (𝐺 = 8,16,...) samples generated for the same prompt. The normalized advantage for sample 𝑥 with binary reward 𝑟 ∈ {0,1} is 𝐴˜GRPO = 𝑟−𝜇𝒢

𝜎𝒢+𝜖, where both the baseline 𝜇𝒢 (e.g., the group mean 𝐺1

∑︀

𝑗 𝑟𝑗) and the standard deviation 𝜎𝒢 are estimated from the same small group of 𝐺 samples. This coupled, small-sample estimation introduces three fundamental sources of variance:

- • Noisy Baseline (Numerator): The baseline 𝜇𝒢, estimated from only 𝐺 samples, where 𝐺 is small, is a high-variance quantity. This inflates the variance of the unnormalized advantage (𝑟 − 𝜇𝒢) by a factor of (1 + 𝐺1 ) compared to using an optimal baseline.

- • Noisy Scaling (Denominator): The standard deviation 𝜎𝒢, estimated from only 𝐺 samples, is also highly variable. Scaling the gradient by this noisy random variable further increases total variance.
- • Information Loss (Degeneracy): When all rewards in the group are identical (e.g., all 0s or all 1s), the advantage for every sample becomes zero, providing no gradient signal. This event, which occurs with probability 𝑍𝐺(𝑝) = 𝑝𝐺 + (1 − 𝑝)𝐺 where 𝑝 = 𝑉 𝜋(𝑥), effectively reduces the batch size and inflates variance by a factor of 1/(1 − 𝑍𝐺(𝑝)), an issue that is especially severe for easy (𝑝 ≈ 1) or hard (𝑝 ≈ 0) prompts.

SPO’s Low-Variance Decoupled Advantage: In contrast, SPO is designed to minimize these variance sources by decoupling the advantage calculation from the current group of samples. It uses an actionindependent baseline 𝑏 = 𝑣ˆ(𝑥) from a historical tracker, which provides a stable, low-variance estimate of the true success probability 𝑝. The advantage is simply 𝐴SPO = batch_norm(𝑟(𝑥,𝑦) − 𝑣ˆ(𝑥)). Crucially, SPO then applies global normalization [3, 25, 31], scaling all advantages in a large batch of size 𝐵 ≫ 𝐺 by a single, stable standard deviation 𝜎batch. This design avoids GRPO’s pitfalls: the baseline 𝑏 is near-optimal, the normalization scaler 𝜎 is stable, and there is no systematic information loss from group-outcome degeneracy.

Quantitative Comparison: A simplified ratio of the reward-term variance quantifies the difference:

Var[𝑔]GRPO Var[𝑔]SPO ≈

1 + 𝐺1

1

1 + 𝜓𝐺 1 + 𝜓ℬ ⏟ ⏞

. (12)

×

×

1 + 𝑁 1 ⏟ ⏞eff+1

1 − 𝑍𝐺(𝑝) ⏟ ⏞

Normalization Noise

Information Loss

Baseline Noise

Here, 𝑁eff is the effective sample count for SPO’s tracker, and 𝜓𝐺 > 0 captures the excess variance from per-group, 𝜓ℬ represents the excess variance introduced by estimating the normalization statistics (mean and standard deviation) from a large global batch of size 𝑁ℬ (𝜓ℬ ≈ 0). For a moderately difficult prompt (𝑝 = 0.5) with 𝐺 = 8, the normalization noise dominates. However, for an easy/hard prompt (𝑝 = 0.9/𝑝 = 0.1), the information loss term dominates, and the ratio swells to ≈ 1.97. While increasing 𝐺 in GRPO mitigates information loss, it does so at a multiple generation cost and cannot fix the inherent noise from its small-sample baseline and scaling. SPO achieves lower variance more efficiently by design.

### D. Training and Evaluation Details

All experiments in this paper are implemented on top of verl [35] and ReTool [13] for the tool-integrated reasoning setup. During training, we set the maximum response length to 16,384 tokens. The policy learning rate is fixed at 1 × 10−6. Following DAPO [42], we adopt the Clip-Higher mechanism, with clipping parameters 𝜀low = 0.2 and 𝜀high = 0.28, to balance exploration and exploitation. The sampling parameters are set to temperature 1.0, top-𝑝 = 1.0, and top-𝑘 = −1. The forgetting rate thresholds are chosen as 𝜌min = 0.875 and 𝜌max = 0.96, yielding window sizes 𝑊min = 1 − 𝜌 1

= 8 and 𝑊max = 25.

min

GRPO rollouts are collected with multiple responses per prompt, and training mini-batch sizes are chosen such that 8 gradient updates are performed per rollout step. For a fair comparison, the prompt batch size in SPO is set equal to the total number of responses in GRPO, as SPO generates only a single response for each prompt. Specifically, GRPO uses a prompt batch size of 256 with 8 responses per prompt and a training mini-batch size of 256, while SPO operates on 2,048 = 256 × 8 prompts. Both algorithms are set with maximum of 8 Python interpreter interaction turns.

For evaluation on hard math competition benchmarks, i.e., AIME 24, AIME 25, BeyondAIME [32], BRUMO 25 [5] and HMMT 25 [5], we set sampling parameters to temperature 0.6, top-𝑝 0.95, and top-𝑘 20, as officially recommended5. We define a binary reward function 𝑟𝑖,𝑗 such that a response receives 𝑟𝑖,𝑗 = 1 if the final answer is correct, and 𝑟𝑖,𝑗 = 0 otherwise. The same reward function is consistently used during training for policy optimization and during evaluation. We set the maximum response token to 32,768.

Given a test set with 𝑀 problems, and for each problem 𝑖 we independently sample 𝑘 responses with rewards {𝑟𝑖,1,𝑟𝑖,2,...,𝑟𝑖,𝑘}, we define:

- • avg@𝑘: the expected correctness of an individual response:

avg@𝑘 =

1 𝑀

∑︁𝑀

𝑖=1

1 𝑘

∑︁𝑘

𝑗=1

𝑟𝑖,𝑗.

- • pass@𝑘: the probability of solving a problem within 𝑘 attempts. Directly computing 1

(︀

max1≤𝑗≤𝑘 𝑟𝑖,𝑗 = 1

)︀

can lead to high variance. Following [7], we instead generate 𝑛 ≥ 𝑘 responses per problem,

5https://huggingface.co/Qwen/Qwen3-8B

count the number of correct ones 𝑐 ≤ 𝑛, and use the unbiased estimator:

]︃,

[︃1 −

(︀𝑛−𝑐

)︀ (︀𝑛

∑︁𝑀

𝑖

1 𝑀

pass@𝑘 =

𝑘

)︀

𝑘

𝑖=1

where 𝑐𝑖 denotes the number of correct responses for problem 𝑖.

• maj@𝑘: the correctness of the majority-voted answer [37]. This metric first identifies the most frequent answer among 𝑘 responses for each problem. The score is 1 if that modal answer is correct, and 0 otherwise. Let 𝑎𝑖,𝑗 be the final answer string for the 𝑗-th response to problem 𝑖, and let 𝑟(·) be the reward function for a given answer string. The metric is defined as:

1 𝑀

maj@𝑘 =

𝑟 (︁mode{𝑎𝑖,𝑗}𝑘𝑗=1)︁.

∑︁𝑀

𝑖=1

### E. Ablation Studies

We conduct a series of ablation studies to dissect the core components of SPO and validate our design choices. To facilitate efficient experimentation, these studies are performed under a streamlined setting compared to our main experiments. Specifically, we utilize a batch size of 256 prompt-response pairs, and the model is updated with 4 gradient steps for each collected batch. All ablation results are reported on the AIME 25 benchmark, using the avg@16 metric with a maximum generation length of 16,384 tokens.

0.60

0.55

0.50

avg@16

0.45

0.40

SPO

0.35

A *-PO

0.30

0 20 40 60 80 100 120 140

Training Steps

(a) SPO vs. 𝐴*-PO

0.600

0.575

0.550

0.525

avg@16

0.500

0.475

0.450

w/ Baseline

0.425

w/o Baseline

0.400

0 20 40 60 80 100 120 140

Training Steps

(b) Baseline Ablation

0.600

0.575

0.550

0.525

avg@16

0.500

0.475

0.450

w/ Offline Init

0.425

w/o Offline Init

0.400

0 20 40 60 80 100

Training Steps

(c) Offline Initialization Ablation

- Figure 6: Ablation studies evaluating the core components of SPO. (a) SPO’s adaptive baseline outperforms the static baseline of 𝐴*-PO, demonstrating the benefit of a value function that evolves with the policy. (b) Removing the value tracker (“w/o Baseline”) causes a severe performance drop, confirming its critical role in reducing gradient variance. (c) Eliminating the offline initialization step (“w/o Offline Init”) leads to initial training instability and suboptimal convergence, highlighting the importance of a warm start for the value tracker.

SPO vs. 𝐴*-PO. This experiment, presented in Figure 6a, compares our proposed SPO with 𝐴*-PO [6]. 𝐴*-PO utilizes a static baseline derived from a pre-computed optimal value function, 𝑉 *, which is tied to the KL-regularized objective with respect to an initial reference policy, 𝜋ref. While this approach is highly efficient, its central assumption may be challenged in tool-calling scenarios. In these tasks, learning involves acquiring new functional capabilities, leading to a significant policy drift where the learned policy, 𝜋𝑡, diverges substantially from 𝜋ref. Consequently, the pre-computed 𝑉 * may become a less representative baseline for the current policy’s true value function, 𝑉𝜋𝑡, potentially affecting the accuracy of the advantage estimates. In contrast, SPO’s baseline is adaptive, dynamically tracking an

estimate of 𝑉𝜋𝑡 as the policy evolves. The empirical results, which show SPO’s superior performance, suggest that this adaptability is crucial. By maintaining a baseline that remains relevant to the current policy, SPO provides a more stable and effective learning signal in environments that demand significant policy evolution. Finally, from a practical perspective, 𝜋ref computation during 𝐴*-PO policy update occupies an extra trunk of GPU memory, making it less appealing than the proposed SPO algorithm.

Baseline Ablation. Figure 6b presents a crucial ablation that validates the fundamental principle of using a baseline for variance reduction. In this experiment, we remove the value tracker component 𝑣ˆ−1(𝑥) from the advantage calculation, causing the algorithm to rely solely on the globally batch-normalized raw reward 𝑟(𝑥,𝑦) as its learning signal. However, the substantial performance degradation observed is a classic illustration of the remaining challenges. While global normalization effectively controls the overall scale of rewards, the raw reward signal is still noisy on a per-sample basis as it fails to account for prompt-specific difficulty. SPO’s history-informed baseline is designed to subtract this expected difficulty, thereby effectively reducing variance and providing a much cleaner, more reliable gradient for learning. This experiment confirms that the adaptive value tracker is the most critical component for SPO’s success, directly addressing the core challenge of variance in single-stream policy optimization.

Offline Initialization Ablation. In Figure 6c, we analyze the impact of the value tracker’s initialization phase. The standard SPO algorithm initializes the value tracker with estimates computed from a small set of 𝑛0 offline samples, giving it a “warm start”. The ablation removes this step, forcing the tracker to learn from scratch online. The results clearly demonstrate the benefit of the offline initialization. Without it, the tracker begins with a highly inaccurate baseline, leading to high-variance gradients and significant instability in the initial training phase, as evidenced by the performance dip. Although the model eventually recovers, it fails to reach the same level of performance as the properly initialized model, underscoring the importance of a good initial value estimate for stable and effective learning.

