# arXiv:2510.04996v3[cs.LG]5Dec2025

## Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives

Wei Xiong∗◦ Chenlu Ye∗◦ Baohao Liao∗■ Hanze Dong∗♢ Xinxing Xu♢ Christof Monz■ Jiang Bian♢ Nan Jiang◦ Tong Zhang◦

◦University of Illinois Urbana-Champaign ♢Microsoft Research ■University of Amsterdam

### Abstract

Reinforcement learning (RL) for large language model reasoning is frequently hindered by signal loss, a phenomenon where standard uniform sampling with small group sizes fails to uncover informative learning signals for difficult prompts. We demonstrate that this collapse is a statistical artifact of undersampling rather than an inherent model limitation. To address this systematically, we introduce a theoretical framework based on optimizing a non-linear RL objective (e.g., log-likelihood). We show that this objective naturally induces a weighted gradient estimator that prioritizes difficult prompts, which can be robustly realized through adaptive sampling. Guided by this framework, we propose Reinforce-Ada, a family of algorithms that dynamically allocates inference budgets based on prompt difficulty, effectively scaling up RL compute to where it is needed most. Unlike passive filtering methods that discard low-signal prompts, Reinforce-Ada actively invests compute to recover them. We introduce two efficient realizations: an estimation-based approach and a modelfree sequential sampling approach. Extensive experiments across multiple benchmarks show that Reinforce-Ada significantly outperforms uniform baselines like GRPO, recovering lost signals and accelerating convergence by up to 2× while maintaining the same total inference budget. Code is available at https://github.com/RLHFlow/Reinforce-Ada.

0.45

[Figure 1]

GRPO (Reward)

Reinforce-Ada-Seq-Balance (Reward)

0.5 Reward

0.40

GRPO (Entropy)

Reinforce-Ada-Seq-Balance (Entropy)

0.35

0.4

0.30

Entropy

0.25

0.3

0.20

0.2

0.15

0 100 200 300 400 500 600 700 800 Training Steps

Figure 1: Plug-and-play usage. Left: a direct replacement of the generation API in verl (generate sequences → generate multi round adaptive downsampling). Right: with no other changes, Reinforce-Ada attains faster reward growth and a higher asymptote than GRPO.

∗Equal contribution. A detailed attribution of authorship credits is provided in Appendix A. Correspondence to Hanze Dong (hanzedong@microsoft.com) and Wei Xiong (wx13@illinois.edu).

### 1 Introduction

Reinforcement learning (RL) has become a central paradigm for aligning large language models (LLMs). Conceptually, the training objective is to maximise the expected reward of model outputs under a given prompt distribution. The primary challenge is not from the formulation of the objective but from the high variance in gradient estimates, which introduces instability into the optimization process.

Formally, for a prompt x ∼ d0, the policy πθ produces a response a ∼ πθ(·|x), and a verifier yields a binary reward r⋆(x,a) ∈ {0,1}. The learning objective is

pθ(x), (1) where pθ(x) is often referred to as the pass rate, which is metric of the prompt difficulty given the model πθ. Estimating its gradient requires sampling multiple responses. With only a few samples per prompt, inference is affordable but the gradient is noisy; with many samples, the signal is clear but inference becomes prohibitively expensive. The trade-off between signal quality and cost is a central challenge in RL for LLMs. The vanilla policy gradient with small n has notoriously high variance. A standard remedy introduces a reward baseline b(x), yielding

J(θ) = Ex∼d

[Ea∼π

θ(·|x)r⋆(x,a)] =: Ex∼d

0

0

gθ(x,a) = (r⋆(x,a) − b(x)) · ∇θ log πθ(a|x), which stabilizes training while preserving unbiasedness. Group Relative Policy Optimization (GRPO) (Shao et al., 2024) extends this principle by assigning n responses per prompt and normalizing each sample’s advantage:

ri − r¯ σr + ε

, (2)

AGRPO(x,ai) =

where r¯ and σr are the mean and standard deviation of group rewards. This group-wise normalization highlights informative variations while suppressing noise, making GRPO widely adopted in practice.

Despite these benefits, GRPO fundamentally relies on a small and fixed n per prompt, creating a vulnerability to signal collapse. When all n samples for a prompt yield identical rewards, either all correct or all incorrect, the group mean r¯ equals each reward ri. Consequently, the computed advantages vanish (AGRPO = 0), resulting in a zero gradient. Such uniform-reward scenarios arise frequently: during early training phases when the model fails on all attempts for challenging prompts, and later when it consistently succeeds on trivial ones. For instance, with a group size of 8, if the pass rate p = 0.1, the probability that all 8 samples are 0 is 0.98 ≈ 19%. Empirically, even with n = 32, more than half of prompts fall into this “zero-gradient” regime as models improve (Yu et al., 2025).

Crucially, this collapse is not due to the prompts being inherently trivial or impossible, but a statistical artifact of undersampling (ˆpθ(x) = 0 or 1: estimating pθ(x) with finite samples from πθ(·|x)). Training prompts are typically filtered to ensure moderate difficulty (e.g., Yang et al. (2024) retain prompts where 2–5 out of 8 responses are correct), meaning most prompts have non-trivial success probabilities (0 < pˆθ < 1). With small n, however, random fluctuations make it likely to observe all-correct or all-incorrect groups, thereby masking the true learning signal. Larger n reliably recovers these signals, but at unsustainable inference cost (Figure 21).

To resolve this trade-off, we first introduce a general framework that provides a principled foundation for our solution. We argue that the standard RL objective in Equation (1) is the root of the problem, as it values

- 1This figure shows results on Open-R1 subset for Qwen2.5-Math-1.5B and an RL-trained checkpoint (step 400). The base

model scores 26.5% at pass@1, but its pass@256 rises to 81.3%, highlighting its potential to solve the majority of prompts. Similarly, the model exhibits 35.3% all-correct groups at n = 4, but only 10.2% at n = 256. These results demonstrate that the missing signal is often recoverable with larger n, confirming that uniform-reward collapse is a statistical artifact of undersampling rather than a model limitation.

Pass@k Curves

All k Responses Reward = 1

1.0

Qwen2.5-Math-1.5B-base

0.5

Qwen2.5-Math-1.5B-RL

0.8

0.4

Allkresponsesreward=1

Accuracy(Pass@k)

0.6

0.3

0.2

0.4

0.1

0.2

Qwen2.5-Math-1.5B-base

Qwen2.5-Math-1.5B-RL

0.0

0.0

1 2 4 8 16 32 64 128 256

1 2 4 8 16 32 64 128 256

k (Number of Attempts)

k (Number of Attempts)

- Figure 2: Pass@k curves (left) and the ratio of prompts with all-correct responses (right) for two models on a subset of the Open-R1 prompt set. The models tested are the Qwen2.5-Math-1.5B base model and an intermediate checkpoint from its RL training. The percentage of prompts yielding all-correct/all-incorrect responses is high for small k but drops significantly as k increases. This suggests that signal loss is often a statistical artifact of small sample groups.

all prompts equally regardless of their difficulty. However, from a learning dynamics perspective, solving a difficult prompt (where the model currently fails) provides significantly more information than repeatedly solving an easy one. To effectively balance cost and signal discovery, we must prioritize prompts based on their pass rates. This motivates us to consider a non-linear objective Jf(θ) = Ex[f(pθ(x))], such as the log-likelihood f(p) = log p. The gradient of this objective naturally acquires a prompt-dependent weight, ∇Jf = Ex[f′(p) · ∇p]. This weight (e.g., 1/p for the log objective) explicitly assigns more importance to difficult prompts, directly targeting the signal loss problem. This formulation unifies two implementation paths: explicitly weighting the gradient, or implicitly re-weighting by allocating more samples ni (i.e., adaptive sampling).

Building on this framework, we propose Reinforce-Ada, a family of adaptive sampling algorithms that robustly implement this implicit weighting principle. We present two efficient realizations to determine the optimal budget allocation:

- 1. Estimation-based Allocation (Reinforce-Ada-Est): This variant estimates pass rates online using a lightweight value network or Bayesian moving averages, and then explicitly allocates sampling budgets proportional to the theoretical target (e.g., ni ∝ 1/pˆi).
- 2. Implicit Allocation via Sequential Sampling (Reinforce-Ada-Seq): This variant adopts a model-free approach inspired by multi-armed bandits. It continues sampling until sufficient signal is found (e.g., until K positive responses are collected). We show that this simple stopping rule naturally achieves the desired allocation without needing explicit estimation.

Reinforce-Ada is a plug-and-play replacement for the generation step in standard RL pipelines, requiring no architectural modifications. Unlike prompt selection or curriculum methods that operate at the macro prompt level, our method performs micro-level response allocation, shaping the internal structure of each training group. Across multiple LLMs and benchmarks, it consistently improves signal quality and sample efficiency, achieving the benefits of large-n training at a fraction of its cost.

### 2 The Weighted Objective Framework

#### 2.1 Prior Approach: Passive Filtering and Large Group Size

The standard RL objective is J(θ) = Ex∼d[pθ(x)]. In practice, d is often a simple uniform distribution over the training set. The most direct, unbiased estimator for this objective is to sample a batch of prompts and a uniform group size n of responses for each.

However, this naive, uniform-sampling approach, leads to signal loss when n is small: if all responses in a group yield the same reward (all correct or all incorrect), the advantages normalize to zero, and the gradients vanish (see Equation (2)). Crucially, this “signal collapse” is not due to the prompts being inherently trivial or impossible, but is a statistical artifact of undersampling.

Prior work has observed this issue and proposed passively filtering-out groups with uniform rewards (Yu et al., 2025; Xiong et al., 2025). While this prevents wasted gradient computations, it still incurs the significant upfront cost of generating responses that are ultimately discarded. Moreover, if difficult prompts are systematically discarded due to a lack of positive signal, the model never learns to solve them, thus hindering training improvement.

With these observations, a natural alternative is to use a large n to reliably capture learning signals. Indeed, concurrent work by Hu et al. (2025) validates this intuition, showing that increasing n up to 512 recovers valid signals and improves performance. However, generating hundreds of samples for every prompt is computationally prohibitive at scale. As demonstrated by DeepSeek-R1 (DeepSeek-AI et al., 2025), we can use only n = 16 responses per prompt to get an effective gradient for model updates. This reveals a significant gap between the inference budget needed to reliably find a learning signal and the update budget required for an effective parameter update.

To bridge this gap, we need a framework that moves beyond uniform allocation. We propose an adaptive approach that smartly allocates a larger inference budget to prompts where the signal is scarce, efficiently discovering robust learning signals without the waste of the uniform, large-n approach.

#### 2.2 A Principled Solution: The Weighted Objective Framework

Beyond practical intuition, we establish a principled theoretical foundation for allocating inference budgets adaptively. We argue that the root cause of the uniform sampling inefficiency is the standard RL objective J(θ) = Ex[pθ(x)], which values all prompts equally.

Instead, we propose a general theoretical framework based on optimizing a non-linear transformation of the pass rate:

Jf(θ) = Ex∼d

f(pθ(x)), (3)

0

where f :→ R is a non-decreasing function. When f(p) = p, we recover the standard objective. We also may want to choose a concave f. In this way, we implicitly assign higher marginal utility to improvements on prompts with low pass rates (difficult prompts). The policy gradient for this general objective reveals a crucial insight. By applying the chain rule, the gradient acquires a prompt-dependent weight:

∇Jf(θ) = Ex∼d

0

θ(·|x) ∇θ log πθ(a|x) · r(x,a) .

f′(pθ(x)) Weight w(p)

·Ea∼π

A canonical choice that targets signal loss is the log-likelihood objective, f(p) = log p. This yields a weight w(x) = 1/pθ(x), which scales inversely with the model’s current performance. This explicitly instructs the optimizer to prioritize difficult prompts (where p → 0) with a potentially large weight, counteracting the vanishing gradients typically observed in these regimes.

- Example 1 (Log objective f(t) = log(t)). Here Jlog(θ) = Ex∼d

0

log(pθ(x)), and the gradient for a prompt x is

1 pθ(x) · Ea∼π

θ(·|x) r⋆(x,a) · ∇θ log πθ(a|x) .

This explicitly instructs the optimizer to prioritize difficult prompts (where p → 0) with a potentially infinite weight. We may also consider other functions such as the power function to present a softer weight.

- Example 2 (Power function f(t) = tα, α > 0). In this case, Jα(θ) = Ex∼d

[pθ(x)]α, and the gradient for prompt x is

0

αpθ(x)α−1 · Ea∼π

θ(·|x) r⋆(x,a) · ∇θ log πθ(a|x) . Setting α = 12 gives,

1 2 pθ(x) · Ea∼π

θ(·|x) r⋆(x,a) · ∇θ log πθ(a|x) .

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- f 1(t) = 1

- f 2(t) = 1/t

- f 3(t) = 1 2 t

1.0

0.5

0.0

Derivativevalue

0.5

f(t)

1.0

1.5

f(t) = t

2.0

f(t) = log(t) + 1

2.5

f(t) = t

3.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

t

t

- Figure 3: Visualization of different f(t) and f′(t). The concave functions log(t) and √t assign larger weights f′(t) to difficult prompts (t → 0).

This weight w(p) = f′(p) formalizes our intuition. We present a visualization of these function choices in Figure 3. For the standard objective f(p) = p, the weight is w(p) = 1, , confirming that it treats all prompts uniformly. In contrast, both the logarithmic and power objectives introduce prompt-dependent weights w(p) provide us with a weighted gradient, naturally leading to an adaptive treatment. In particular, for concave functions such as log t and √t (on (0,1)), we observe that their prompt-dependent weights decrease with t, assigning larger weights to more difficult prompts. We will also provide an interesting explanation of GRPO’s weight under the log-objective in Appendix B.

Implementation of Weighted Objective: Explicit vs. Implicit. This weighted gradient formulation Ex[w(p) · ∇p(x)] unifies distinct implementation strategies. We can decompose the total weight into two components, w(p) = wsample(p) · wgrad(p), allowing for a flexible spectrum of solutions:

- 1. Explicit Weighting (wsample = 1,wgrad = w): One retains uniform sampling (ni = n) but explicitly multiplies the gradient by w(ˆpi). However, this strategy fails to solve the core “signal loss” problem. Since we rely on Monte-Carlo estimation for the pass rate, a small group size n will often yield pˆi = 0 for difficult prompts, causing the gradient to vanish rather than be weighted up.
- 2. Implicit Weighting via Sampling (wsample = w,wgrad = 1): Alternatively, we can absorb the weight w(p) into the sampling distribution. We sample prompts from a reweighted distribution d′(x) ∝ d0(x) · w(p(x)) and apply the standard, unweighted gradient. In a batched setting with group sampling, this is naturally implemented by assigning a variable group size ni ∝ w(pi) to each prompt.

- 3. Hybrid Strategy: We can balance the two approaches. For instance, with the log-objective (w(p) =

1/p), a pure sampling strategy (ni ∝ 1/p) might be too aggressive, allocating excessive compute to the hardest prompts. A balanced approach is to allocate samples proportional to 1/√p and apply a residual explicit weight of 1/√p to the gradient. This hybrid approach achieves a “sweet spot” by dividing the variance into two stages.

This framework highlights that the intuitive adaptive group sizes are not just a heuristic, but a robust implementation of a non-linear objective. By allocating ni ∝ 1/pi (Implicit Weighting), we naturally direct more computational budget to difficult prompts. This explicitly solves the signal loss problem: the increased budget allows us to recover rare signals that a uniform small-n approach would miss.

Theoretical Connections with GRPO and Variance-reduction Gradient Estimation. This framework is broadly applicable and offers a unified perspective on recent advancements. As a side product, it explains why algorithms like GRPO (Shao et al., 2024) utilize prompt-dependent weights in their advantage and gradient formulations. Furthermore, we find that the log-objective is essential for deriving the optimal variance-reduction gradient estimator under a fixed inference budget. In Appendix B, we generalize the findings of Yao et al. (2025) from rejection sampling fine-tuning to the general Reinforce algorithm, proving the optimality of the log-objective for efficient training.

### 3 Reinforce-Ada: Reinforce with Adaptive Sampling

#### 3.1 Reinforce-Ada-Est: Estimation-based Allocation

Our theoretical framework suggests allocating an implicit sampling budget ni ∝ 1/pi to optimize the logobjective. The central challenge, however, is that the true pass rate pi is unknown prior to sampling.

A common approach in related work (Yao et al., 2025) follows an “explore-then-exploit” paradigm. In these methods, pass rates (ˆpi) are first estimated with a small “explore” budget, and then a larger online sampling budget is allocated proportionally. This two-stage process, however, suffers from a critical flaw. The initial estimation, based on a limited budget, has high variance. This error is most severe for difficult prompts (low pi), where a small budget will almost certainly yield zero positive responses. This leads to a catastrophic estimate of pˆi = 0, causing the algorithm to discard the very prompts that need the most attention.

Consequently, any allocation rule (like 1/√pˆi) will either fail (divide by zero) or assign zero budget, forcing the algorithm to discard these prompts. This means the two-stage approach ends up replicating the same problem as passive filtering (like DAPO (Yu et al., 2025)), failing to solve the very signal loss problem it was designed for.

Adaptive Sampling with Pass Rate Estimation. To overcome this, we propose Reinforce-Ada-Est, which estimates pass rates online using historical training data or auxiliary models, rather than a separate exploration phase. We introduce two strategies for this estimation. A natural idea is to train a value network Vϕ(x) (similar to the critic) to predict the pass rate. Unlike standard PPO which requires estimating tokenlevel values, our task is simpler: we only need the prompt-level expected success rate. Meanwhile, the estimator does not need to be perfectly accurate; it only needs to capture the relative difficulty of prompts to guide budget allocation.

The second strategy is Ada-EMA, a model-free Bayesian Moving Average approach, which is inspired by Qu et al. (2025). Suppose we encounter the prompt across different training epochs t = 1,2,.... We can therefore maintain a running statistic for each prompt to track its difficulty without an auxiliary model. Crucially, since the policy πθ is constantly updating, the pass rate is non-stationary; data collected in early steps becomes stale. To handle this, we apply an exponential decay to historical counts before integrating new data. Specifically, suppose at step t, we allocate a budget of nt samples to prompt x and observe kt

- Algorithm 1 Reinforce-Ada-Est under Log-objective (One Training Iteration)

- 1: Input: Current policy πθ, batch of prompts D, effective update size n, sampling bounds [Nmin,Nmax], total budget target Ntotal, and estimator E
- 2: Phase 1: Estimation and Budget Allocation
- 3: Initialize response set Sx ← ∅ for all x ∈ D
- 4: Get current pass rate estimates pˆx ← E(x) for all x ∈ D
- 5: Allocate budgets: Set Nx ∝ 1/√pˆx + ϵ subject to Nx ∈ [Nmin,Nmax], targeting total usage Nx ≈ Ntotal

- 6: for each prompt x ∈ D do
- 7: Sample Nx responses {aj,rj}N

x

j=1 ∼ πθ(·|x),r⋆(·,·)

- 8: Store collection: Sx ← {aj,rj}N

x

j=1

- 9: end for
- 10: Phase 2: Training Batch and Objective Construction
- 11: Initialize an empty set for the final training data: B ← ∅
- 12: for each prompt x ∈ D do ▷ Use global statistics for baseline and weight
- 13: Let Sx be the full set of collected samples for prompt x
- 14: Compute global mean (high-fidelity pass rate): r¯x ← |S1

x|

|Sx| j=1 rj

- 15: Compute explicit weight: αx ← 1/√r¯x + ϵ ▷ Apply residual weight for log-objective

- 16: Compute weighted advantage for i-th response: Ai ← (ri − r¯x) · αx
- 17: end for
- 18: Compute the policy gradient objective with batch B and update the model.
- 19: Update estimator E with newly collected statistics {Sx}x∈D.

successes. We update the running counters as follows:

Ntotal(t) ← λNtotal(t−1) + nt, Npos(t) ← λNpos(t−1) + kt, where λ ∈ (0,1) is a discount factor that weighs recent data more heavily. The pass rate is then estimated using a Bayesian approach with a prior Beta(α,β): pˆt = (Npos(t) +α)/(Ntotal(t) +α +β). This ensures that even if a prompt yields zero successes in the current batch (kt = 0), the estimate remains non-zero due to the prior and historical accumulation, preventing the algorithm from permanently discarding difficult prompts. We will mainly implement the Reinforce-Ada-Est via the hybrid strategy under the log objective, as this divide the variance into two separate stages and may be more stable. Specifically, we allocate the inference budget proportional to 1/√pi and leave another 1/√pi to the advantage. We also include an ablation in Appendix C.2.

#### 3.2 Reinforce-Ada-Seq: Implicit Allocation via Sequential Sampling

While the estimation-based strategies (Reinforce-Ada-Est) provide a direct way to allocate budgets, they introduce specific implementation requirements. For instance, training an auxiliary value network adds computational overhead and architectural complexity to the RL pipeline. Similarly, the Bayesian approach relies on maintaining persistent statistics and revisiting the same prompts multiple times, which may not suit all training setups.

Motivated by these considerations, we design an alternative strategy that achieves adaptive allocation without explicitly estimating the pass rate. We design a model-free algorithm that integrates estimation and sampling into a unified online process. Our core idea leverages a simple statistical property: if we keep sampling responses until we collect a fixed number of correct answers (say, K = 1), the expected total number of samples E[Ni] is exactly 1/pi. This provides a “built-in” mechanism to automatically achieve the ni ∝ 1/pi allocation required by our log-objective, without needing to estimate pi beforehand.

- Algorithm 2 Reinforce-Ada-Seq (One Training Iteration)

- 1: Input: Current policy πθ, batch of prompts D, effective group size for update n, number of sampling rounds N, samples per round M ≥ n, and exit condition function ExitCondition(·)
- 2: Phase 1: Adaptive Sampling Data Collection
- 3: Set all prompts x as active and initialize response set Sx ← ∅
- 4: for t = 1,...,N do ▷ Iterate through sampling rounds
- 5: for each prompt x ∈ D where active(x) is true do
- 6: Sample M responses {aj,rj}Mj=1 ∼ πθ(·|x),r⋆(·,·)
- 7: Add to collection: Sx ← Sx ∪ {aj,rj}Mj=1
- 8: if ExitCondition(Sx) is met then
- 9: Mark prompt as inactive: active(x) ← false
- 10: end if
- 11: end for
- 12: end for
- 13: Phase 2: Training Batch and Objective Construction
- 14: Initialize an empty set for the final training data: B ← ∅
- 15: for each prompt x ∈ D do ▷ Use all collected samples (“global statistics”) for normalization
- 16: Let Sx = {(aj,rj)}|Sj=1x| be the full set of collected samples for prompt x
- 17: Compute global mean: r¯x ← |S1

x|

|Sx| j=1 rj

- 18: Form update group by downsampling Sx to size n, trying to ensure ≥ n/2 size for each correct or incorrect subset (fill from the other if needed). ▷ Downsample to create the effective group
- 19: Compute advantage for i-th response of prompt x as Ai ← ri − r¯x.
- 20: end for
- 21: Compute the policy gradient objective with batch B and update the model.

We implement a more general version of this “sample-until-condition” idea, inspired by successive elimination methods in multi-armed bandit literature (Slivkins et al., 2019). We present the code of a single training step of this method in Algorithm 2. Specifically, the algorithm operates over a batch of prompts in sequential rounds:

- 1. Initialization: All prompts in the current batch begin in an active set (active arms).
- 2. Iterative Sampling: In each round, we generate M new responses (e.g., M = 16) for every prompt currently in the active set.
- 3. Elimination: At the end of each round, active prompts are checked against an exit condition, and those that satisfy it are removed from future rounds. 2

This sequential sampling process continues until all prompts are resolved or a maximum budget (e.g., Nmax =

128) is reached. Exit condition. The elimination rule plays a central role in shaping the algorithm’s behavior. We consider two primary exit conditions:

- • Positive-focused (Reinforce-Ada-Seq-pos): A prompt is deactivated once we collect at least Kpos correct responses (e.g., Kpos = 16).
- • Balanced (Reinforce-Ada-Seq-balance): A prompt is deactivated once at least Kpos correct and Kneg incorrect responses have been collected (e.g., Kpos = Kneg = 8).

2We also experimented with a more complex variant that estimates pass rates and allocates budgets proportionally within each round. However, this did not yield clear performance gains, so we use this simpler, easy-to-implement version.

Method Core Idea for Handling Zero-Variance Uniform GRPO (small-n) No recovery mechanism. σ({r}) = 0 forces advantage Ai → 0,

causing gradient vanish. Passive Filtering (Yu et al., 2025) Skips updates for groups with uniform rewards (data inefficient). Large-n GRPO Relies on large n to render Pr[σ({r}) = 0] negligible, approximating

the true distribution. N-GRPO (Nan et al., 2025) Augment the reward group with a constant (the maximum possible reward). RL-ZVP (Le et al., 2025) Assign advantages based on entropy information for prompts with uniform rewards.

Reinforce-Ada-Est (Ours) Estimates pass rate pˆ (via EMA/Value Net) to estimate difficulty-

based inference budget allocation (ni ∝ 1/√pˆi).

Reinforce-Ada-Seq (Ours) By design: always collects mixed outcomes before exit.

Table 1: Summary of methods for recovering signal under zero-variance groups.

Both strategies ensure that difficult prompts (low pi) receive more samples, thereby realizing the goal of our theoretical framework. Reinforce-Ada-Seq-balance further emphasizes collecting failure cases, which intuitively corresponds to the behavior of GRPO-type weights that emphasizes prompts with pass rates near 0 or near 1. This design encourages gathering a more diverse set of training signals before a prompt is eventually deactivated.

Practical Implementation via Static Batches. A unique challenge of this sequential approach is that the total number of samples Ni becomes a random variable, resulting in dynamic batch sizes that are incompatible with static GPU computational graphs. To make Reinforce-Ada-Seq-pos/Balance practical, we employ an Oversample-Downsample-Correct pipeline to convert these variable pools into a static training batch.

First, we leverage the variable pool of Ni responses to compute a high-fidelity global baseline, pˆi = Npos/Ni. Because we allocate larger budgets to difficult prompts, this estimate is now more likely to be non-zero and robust, far superior to statistics derived from small fixed groups.

Next, to create a static batch, we down-sample each prompt’s pool to a fixed group of n responses (e.g., n = 16). We use a balanced sampling strategy (drawing n/2 positive and n/2 negative samples) to ensure signal diversity. While this down-sampling achieves engineering efficiency, it removes the implicit weight (Ni) generated by the sequential sampling process. Fortunately, the high-fidelity baseline pˆi obtained in the first step allows us to recover the theoretical objective. Because pˆi is stable and bounded away from zero (thanks to the sequential sampling process), we can safely return to the Explicit Weighting strategy. We simply re-introduce the weight by explicitly multiplying the gradient by 1/pˆi.

The final gradient estimator for Reinforce-Ada-Seq-positive/Balance is thus:

1 pˆi Re-introduced Explicit Weight

gˆ(xi) =

·

  1

∇log π(aj) · (rj − pˆi)

n



n

j=1

Static Group Gradient

This pipeline enables the algorithm to operate with the throughput of static batches while mathematically optimizing the weighted non-linear objective.

- 3.3 Comparison with the Dynamic Sampling in DAPO

We now compare our framework with the dynamic sampling strategy proposed in DAPO (Yu et al., 2025). DAPO samples batches of prompts with a fixed group size n and discards those that lack learning signals (i.e., groups with uniform rewards). To maintain a full training batch size B, DAPO repeatedly samples new batches of prompts until enough valid groups are collected.

While this approach effectively avoids wasted gradient computations, it fundamentally differs from our objective. Because DAPO uses a uniform group size for all prompts, difficult prompts (where p(x) is low) are statistically destined to yield no signal and be discarded, and very easy prompts with p(x) close to 1 are similarly likely to produce all-success groups that are also discarded. Consequently, DAPO neither resolves the signal-loss problem for hard prompts nor preserves signal for already-solvable ones, biasing training toward a narrow band of mid-difficulty problems and potentially limiting the model’s ability to improve at the frontier of its capabilities.

In contrast, our method reallocates compute within each prompt by assigning larger inference budgets to harder prompts. This helps the model obtain meaningful learning signals even on difficult prompts with low pass rates—something DAPO is unable to do. Finally, we note that DAPO’s accumulative-prompt dynamic sampling is not conflict with our adaptive sampling scheme. The two strategies can be combined to yield more stable and effective training.

- 4 Experiment

4.1 Experimental Setup

Models and Benchmarks. We evaluate the generality of our framework across varying scales and architectures, using four foundation models: Qwen2.5-Math (1.5B and 7B), Qwen3-4B-Instruct, and Llama-

- 3.2-3B-Instruct. For evaluation, we employ a comprehensive suite of mathematical reasoning benchmarks: MATH500 (Hendrycks et al., 2021), Minerva Math (Lewkowycz et al., 2022), and OlympiadBench (He

- et al., 2024). Furthermore, to test robustness on competition-level problems, we compile an AIME-like test set consisting of 230 problems sourced from recent competition: AIME24, AIME25, HMMT24, HMMT25, BRUMO25, AMC23, and CMIMC25 (Balunovi´c et al., 2025). All evaluations report the Pass@1 accuracy averaged over 32 samples (Ave@32), generated with a temperature of 1.0 and a max token limit of 4096. We will also include the analysis of reward-entropy trade-off.

Data Curation and Verifier. Training utilizes the standard subset of OpenR1-Math-220k3. We employ the Math-Verify tool (Kydlı´ˇcek) for automatic solution correctness verification. To ensure a highquality training signal, we implement a standard preprocessing pipeline: (1) Deduplication of prompts; (2) Exclusion of prompts where reference solutions fail verification; (3) Difficulty-based Filtering: Following Yang et al. (2024), we filter prompts based on empirical difficulty estimated by sampling 16 responses from the base model. We discard trivial prompts (average reward > 0.375) and effectively impossible prompts (0 correct responses). This helps us to focus the training distribution on problems of moderate difficulty rather than trivially easy or impossibly hard ones. This choice is motivated by the observation of Hugging Face (2025) that incorporating overly simple problems into the training set can degrade model performance.

RL Training Details. All experiments are conducted based on the verl framework (Sheng et al., 2024), where a Tinker-based4 implementation is also provided. Training is configured with a prompt batch size of 512. We use the AdamW optimizer with a fixed learning rate of 1 × 10−6. To encourage exploration (with 10-step warm-up), an entropy regularization term with coefficient 1 × 10−4 is applied, while no KL penalty is introduced. Following Yu et al. (2025), we adopt the clip-higher trick by setting the clipping range to

- 3https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- 4https://github.com/thinking-machines-lab/tinker-cookbook

[0.2,0.28]. For all variants, we keep these standard RL hyperparameters (learning rate, batch size, entropy coefficient, optimizer, etc.) fixed. No tuning was performed for Reinforce-Ada. Each RL run is carried out for 600 training steps to obtain the final model.

Model Algorithm Math500 Minerva Math Olympiad Bench AIME-like Weighted Average

GRPO-n4 74.2 34.4 38.4 16.2 45.3 GRPO-n8 75.3 34.3 39.2 17.3 46.1 GRPO-n16 76.9 35.1 40.4 17.9 47.3 DAPO 75.2 34.7 38.5 17.5 45.9

Qwen2.5-Math-1.5B

Reinforce-Ada-Seq-pos 75.8 35.7 38.6 16.5 46.1 Reinforce-Ada-Seq-balance 77.4 36.5 40.5 17.5 47.6 Reinforce-Ada-Est 76.1 34.8 39.1 18.5 46.5

GRPO 82.2 44.7 45.6 23.2 53.3 Reinforce-Ada-Seq-pos 82.7 45.1 46.7 23.7 54.2 Reinforce-Ada-Seq-balance 84.0 45.2 47.1 23.7 54.6 Reinforce-Ada-Est 83.6 42.1 46.6 24.1 53.7

Qwen2.5-Math-7B

GRPO 51.7 20.5 20.4 7.2 27.9 Reinforce-Ada-Seq-pos 52.6 22.2 21.0 7.5 28.8 Reinforce-Ada-Seq-balance 53.2 22.4 21.2 8.0 29.1

Llama-3.2-3B-instruct

GRPO 90.4 51.2 64.9 38.5 66.5 Reinforce-Ada-Seq-pos 91.6 50.4 66.3 38.8 67.4 Reinforce-Ada-Seq-balance 91.7 53.0 65.7 38.8 67.6

Qwen3-4B-instruct

- Table 2: Performance comparison of GRPO and Reinforce-Ada. We report average@32 accuracy with a sampling temperature of 1.0 and a maximum generation length of 4096 tokens. The weighted average score is computed according to the number of prompts in each benchmark.

Sampling Configuration. The downsampling mechanism and random samples of Reinforce-Ada-Seq make the methods not directly comparable. We standardize the update group size (the number of samples used for the backward pass) to n = 4 for Reinforce-Ada-Seq and baseline GRPO, but we will also include the GRPO baselines with larger group size 8 and 16. We refer readers to Figure 5 to get a sense of the sampling dynamic of different methods. Specifically, we will consider the following competitors:

- • GRPO Baselines: We compare against GRPO with fixed group sizes n ∈ {4,8,16}.
- • Reinforce-Ada-Seq: The algorithm adaptively samples up to Nmax = 32 responses per prompt but downsamples the collected pool to a fixed group size of n = 4 for the gradient update. This ensures that the training cost (backward pass) remains identical to GRPO-n4, with overhead limited only to inference.
- • Reinforce-Ada-Est: The total sampling budget is allocated proportionally to 1/√pˆi. The choice of hyper-parameter 8 is to roughly match the sampling cost of the Reinforce-Ada-Seq

In general, Reinforce-Ada-Est is most directly comparable to GRPO-n8, as they consume approximately the same number of forward and backward samples. In contrast, Reinforce-Ada-Seq involves inherent stochasticity in its adaptive sampling process, so only coarse-grained comparisons are possible; nevertheless, our experiments consistently show clear and robust trends across different models.

#### 4.2 Main Result

Adaptive Allocation vs. Uniform Scaling. Table 2 summarizes the performance across all benchmarks and Figure 4 presents the training reward curves. We first notice that standard GRPO shows monotonic improvement as group size increases from n = 4 to n = 16 (e.g., 45.3% → 47.3% on Qwen2.5-Math1.5B). This confirms that signal loss is an artifact of undersampling. However, this comes at a linear cost. Reinforce-Ada demonstrates a more efficient pathway to scale up RL training. Instead of uniformly increasing the budget for all prompts, our method dynamically directs compute to the learning frontier.

Crucially, Seq-Balance successfully matches or even exceeds the performance of the expensive GRPO-n16 (e.g., 47.6% vs. 47.3% on Qwen2.5-Math-1.5B) while operating with a significantly lower inference budget (comparable to n ≈ 8 − 10) and lower training cost (n = 4). This indicates that Reinforce-Ada can capture the signal-quality benefits of large-scale sampling without incurring the full computational penalty.

Comparison with Passive Filtering (DAPO). While DAPO improves over the naive GRPO-n4 (45.9% vs 45.3%) by filtering out zero-signal groups, it generally lags behind our adaptive methods. This supports our hypothesis that passive filtering is insufficient for difficult reasoning tasks; instead, active investment of compute is required to recover the latent learning signals in hard prompts. In Appendix C.1, we will provide a more detailed analysis of the accuracies on prompts with different difficulty levels, which further validates our intuition.

Qwen2.5-Math-7B: Training Step vs Reward

Qwen2.5-Math-1.5B: Training Step vs Reward

0.50

0.7

0.45

0.6

0.40

0.35

Reward

Reward

0.5

GRPO

0.30

Reinforce-Ada-Seq-Pos

0.25

GRPO

Reinforce-Ada-Seq-Balance

0.4

Reinforce-Ada-Seq-Pos

Reinforce-Ada-Est

0.20

Reinforce-Ada-Seq-Balance

GRPO-n8

0.15

0.3

Reinforce-Ada-Est

GRPO-n16

0 50 100 150 200 250 300 350 400 Training Steps

0 50 100 150 200 250 300 350 400 Training Steps

Llama3.2-3B-it: Training Step vs Reward

Qwen3-4B: Training Step vs Reward

GRPO

Reinforce-Ada-Seq-Pos

0.6

0.5

Reinforce-Ada-Seq-Balance

Reinforce-Ada-Est

0.5

0.4

Reward

Reward

0.4

0.3

GRPO

Reinforce-Ada-Seq-Pos

0.3

0.2

Reinforce-Ada-Seq-Balance

Reinforce-Ada-Est

0.2

0.1

0 50 100 150 200 250 300 350 400 Training Steps

0 50 100 150 200 250 300 350 400 Training Steps

- Figure 4: Training reward vs. steps for GRPO and Reinforce-Ada across backbones: Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, and Llama-3.2-3B-it, Qwen3-4B. Curves are smoothed with a 20-step moving average. In all cases, Reinforce-Ada learns faster and reaches a higher reward than GRPO, with the Balance variant typically achieving the highest asymptote.

Comparison among different adaptive sampling variants. Comparing the adaptive variants, we observe that Seq-Balance consistently outperforms Seq-Pos, with the gap widening in later training stages. As the policy improves, positive responses become abundant, causing Seq-Pos to exit early and revert to a near-uniform sampling regime. However, negative responses (errors) become rare for these improved models. Seq-Balance forces the collection of these ”hard negatives,” ensuring that the variance of the gradient estimator remains non-zero. This sustained signal quality explains its higher asymptotic performance and robustness across different model families. Ada-Seq and Ada-Est are not directly comparable as their inference and training costs are not identical. We also observe a mixed result across different mdoels. For instance, with Qwen2.5-Math-1.5B, Ada-Est is the best one. But for Llama-3.2-3B-it, we notice that the Ada-Est roughly matches that of Ada-Seq-Pos but is behind the Ada-Seq-Balance.

Finally, we emphasize that the additional computational overhead of Reinforce-Ada-Seq stems only from the inference (generation) stage. Because we uniformly down-sample the collected trajectories to a fixed group size for the policy update, the backward pass (training cost) remains identical to standard GRPO-n4. In contrast, GRPO-n8, GRPO-n16, and Ada-Est will incur a much larger training cost than other methods since the train on more samples per iteration.

Sampling Strategy Comparison

8000

GRPO-8

GRPO-16

7000

Reinforce-Ada-Seq-Balance

ExtraSamplesperSteps

Reinforce-Ada-Seq-Pos

6000

DAPO

5000

4000

3000

2000

1000

0 100 200 300 400 500

Training Step

(1-2 rounds) Prompts meeting positive threshold

Prompts below positive-sample threshold

Reinforce-ada-balance

55

(1-2 rounds) Prompts meeting both thresholds

Prompts below negative-sample threshold

Reinforce-ada-pos

6000

400

50

5000

45

AdditionalSamples

350

#Samples

#Prompts

4000

40

300

35

3000

30

2000

250

25

1000

0 100 200 300 400 500 Training Steps

0 100 200 300 400 500 600 Training Steps

0 100 200 300 400 500 600 Training Steps

- Figure 5: First row: Sampling dynamics of different training strategies using the Qwen2.5-Math-1.5B model. We omit Reinforce-Ada-Est since its sampling cost matches that of GRPO-8. Second row: Sampling dynamics with the Qwen2.5-Math-1.5B model. Left: additional samples generated in later rounds compared to standard GRPO. Middle: number of prompts that remain active after multi-round adaptive sampling with the Reinforce-Ada-Seq-balance variant. Right: number of prompts that satisfy the stopping criteria within the first two rounds with the Reinforce-Ada-Seq-balance variant. All curves are smoothed using a moving average with a window size of 20.

#### 4.3 Analysis of Efficiency and Learning Dynamics

Computation vs. Performance Sweet Spot. We would like to quantify the efficiency gains and analyze the trade-off between inference cost (average samples generated per prompt) and final accuracy. Figure 5 summarizes the inference cost5 measured as the average number of generated samples per iteration. The ranking is

- • Inference Cost Ranking: GRPO-n4 < Seq-Pos < Seq-Balance < GRPO-n16.
- • Performance Ranking: GRPO-n4 < Seq-Pos < GRPO-n16 ≈ Seq-Balance < Ada-Est.

This comparison highlights the efficiency of our approach. Ada-Est achieves the high performance of the expensive GRPO-n16 baseline but requires significantly less compute (lower than GRPO-n16). By allocating

5We remark that GRPO-n16 uses more samples per iteration, resulting in a higher backward-pass cost.

the budget adaptively, we can capture the benefits of scaling up group size (recovering lost signals) without the waste of generating many samples for easy prompts.

#### Model Algorithm Avg. Step Time (s) Relative Cost

GRPO 102 1.0×

Reinforce-Ada-Seq-pos 228 2.2× Reinforce-Ada-Seq-balance 290 2.8×

Qwen2.5-Math-1.5B

Reinforce-Ada-Est 128 1.3×

GRPO 236 1.0×

Reinforce-Ada-Seq-pos 333 1.41× Reinforce-Ada-Seq-balance 375 1.59×

Qwen2.5-Math-7B

Reinforce-Ada-Est 382 1.62×

- Table 3: Average step time (wall-clock seconds per update) of GRPO vs. Reinforce-Ada on 8×H100 (1.5B) and 8×A100 (7B). Relative cost is normalized against GRPO for the same model.

Wall-Clock Efficiency. We quantify the wall-clock cost of our adaptive sampling. We report the average per-step time with verl implementation in Table 3. Here 1.5B model is with 8×NVIDIA H100, and 7B model is with 8×NVIDIA A100. Naturally, Ada-Seq incurs a higher inference cost compared to the baseline GRPO (n = 4) because it generates additional responses to hunt for sparse signals6. For the 1.5B model, Seq-pos and Seq-balance increase the step time by approximately 2.2× and 2.8×, respectively. In comparison, the relative overhead of Reinforce-Ada is smaller on the 7B model than on the 1.5B model. From Figure 4, the 7B policy exhibits a sharp early jump in training reward, which means that most prompts quickly satisfy the positive-sample stopping criterion. Consequently, adaptive sampling requires far fewer additional rollouts on 7B than on 1.5B. Second, the actor update dominates the per-step time on the 7B model (85s out of 236s), whereas it is only a small portion on the 1.5B model (15s out of 102s). As a result, the extra inference introduced by adaptive sampling has a much smaller impact on the overall step time for the 7B model.

The Sampling Dynamic of Ada-Seq. Figure 5 provides a deep dive into how Ada-Seq allocates its budget over time, where the experiments are with Qwen2.5-Math-1.5B:

- • Early Training (Exploration Phase): The cost is high as the model struggles with difficult prompts. The algorithm automatically allocates more samples to find the sparse positive solutions needed to initiate learning.
- • Mid Training (Consolidation Phase): As the model improves, positive samples become easier to find. The sampling cost for Seq-Pos drops monotonically.
- • Late Training (Refinement Phase): Crucially, Seq-Balance exhibits a distinct “U-shaped” cost curve. After ∼300 steps, the cost rises again. This phenomenon occurs because the model becomes highly proficient, making negative samples (failures) rare. To maintain a valid gradient signal (variance > 0), the algorithm actively hunts for these hard negatives. This adaptivity prevents the “all-correct” signal collapse that plagues standard GRPO in later stages.

#### 4.4 Entropy Dynamic and Pass@k

Beyond simple accuracy, we evaluate the quality of the learned policy through the lens of the Reward-Entropy trade-off. A known challenge in evaluating the reasoning ability of LLMs is on the trade-off between reward

6Currently, the multi-sample generation in Ada-Seq is implemented synchronously, meaning that each optimization step is bottlenecked by the slowest completion among the sampled responses. We expect that an asynchronous implementation would be much faster.

(accuracy) and entropy (generation diversity). Previous work has argued that post-training can trade this uncertainty for higher rewards in a predictable manner (Cui et al., 2025). Moreover, prior work shows that standard GRPO often collapses to a low-entropy policy early in training, reducing reasoning diversity thus a worse pass@k for large k (Shao et al., 2024; Yue et al., 2025).

An ideal algorithm should improve rewards without causing the policy to become overly deterministic (entropy collapse). To provide a more comprehensive evaluation, we analyze this trade-off in this part and present the results in Figure 6.

- (i) Reward-entropy frontier. While the precise entropy dynamics can vary depending on the foundation model, Reinforce-Ada consistently achieves a comparable or superior reward-entropy curve. Specifically, from (2-1) of Figure 6, we notice that increasing the group size of GRPO cannot improve the reward-entropy trade-off. In comparison, on Qwen2.5-Math-1.5B (1-1), GRPO concentrates mass early (low entropy, narrow cloud) and achieves lower reward for a given entropy. All Reinforce-Ada variants shift the frontier outward and Reinforce-Ada-Seq-balance lies furthest, at equal reward it sustains higher entropy, and at equal entropy it achieves higher reward. On Llama-3.2-3B-it (1-2), the base policy starts with a higher entropy floor, so the separation among different methods is smaller. But the Ada-Seq variants still achieve a competitive frontier than GRPO and Ada-Est performs even better. We hypothesize that this is because the adaptive sampling pushes the model to explore towards the prompts they are uncertain, thus preserving the entropy. We also observe that Reinforce-AdaSeq-balance > Reinforce-Ada-Seq-pos. We attribute this to the fact that exposure to negative signals discourages the model from becoming overconfident in a single solution path, thus preserving valuable policy diversity.
- (ii) Pass@k behavior. An related observation is that the moderate policy entropy typically converts to an improved pass@k compared to the base model for a wide range of k. Specifically, panel (2-2) shows that all RL methods dominate the base across k, but the largest, most practical gains appear at small budgets (k ≤ 8). In this regime, Reinforce-Ada yields the highest Pass@k; the advantage narrows as k grows (diminishing-returns regime), where Reinforce-Ada-Seq-balance typically remains marginally best. The pattern implies two complementary effects: (a) improved top-1 quality (higher reward at given entropy), and (b) retained diversity among high-scoring modes (shallower saturation with k). Together, adaptive sampling moves the reward–entropy curve outward and converts that shift into higher Pass@k at realistic attempt budgets, with Reinforce-Ada-Seq-balance offering the most stable trade-off.

### 5 Related Work

Data filtering and selection in online RL training for LLMs. Our work is related to the growing body of literature on data selection and filtering for online reinforcement learning with LLMs, which further dates back to the RLHF studies (Zhang et al.; Xiong et al., 2023; Dong et al., 2024; Shi et al., 2024; Feng

- et al., 2025). In the context of RLVR, some methods employ an oversample-then-downsample strategy: they first generate a large, uniform set of responses for each prompt and then select a subset based on specific criteria. Xu et al. (2025) propose downsampling to maximize reward variance within the group, while Ye et al. (2025) use process rewards to select positive samples, mitigating issues with falsely correct responses. Xue et al. (2025) study the data filtering in the context of tool-integrated reasoning, where they find that the trajectories with invalid tool calling will significantly hurt the training stability. Li et al. (2025) propose to use a process search to branch out the trajectories and collect data.

Addressing signal loss in GRPO. A central challenge in applying GRPO is the “signal loss” problem, where groups of responses with uniform rewards yield zero gradients. Prior work has identified this issue and proposed several solutions. The most direct approach is passively filtering out these prompts (Yu et al.,

Qwen2.5-Math-1.5B: Entropy vs Reward

Llama3.2-3B-it: Entropy vs Reward

0.6

0.5

GRPO

Reinforce-Ada-Seq-Pos

Reinforce-Ada-Seq-Balance

0.5

Reinforce-Ada-Est

0.4

0.4

Reward

Reward

0.3

0.3

0.2

0.2

GRPO

Reinforce-Ada-Seq-Pos

Reinforce-Ada-Seq-Balance

0.1

0.1

Reinforce-Ada-Est

0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 Entropy

0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 Entropy

Qwen2.5-Math-1.5B: Entropy vs Reward

0.80

GRPO-n4

0.5

GRPO-n16

0.75

GRPO-n8

0.70

###### Accuracy(Pass@k)

0.4

0.65

Reward

0.60

0.3

0.55

0.2

0.50

Base

Reinforce-Ada-Seq-Balance

0.45

GRPO

0.1

Reinforce-Ada-Seq-Pos

0.40

0.1 0.2 0.3 0.4 0.5 Entropy

2 4 8 16 32 64 128 256

k (Number of Attempts)

- Figure 6: Reward–entropy trade-off (left and mid) and Pass@k (right) on the test benchmarks. ReinforceAda shifts the frontier outward: higher reward at fixed entropy and higher entropy at fixed reward and converts this into stronger Pass@k at small, practical budgets (k ≤ 8), with Reinforce-Ada-Seq-balance typically best.

2025; Xiong et al., 2025). Another line of works proposes to modify the advantage computation and avoid zero gradient. To avoid discarding samples, Nan et al. (2025) propose augmenting the reward group with a constant (the maximum possible reward), ensuring the variance is not zero by introducing a bias. Le et al. (2025) propose to assign advantages based on entropy information for prompts with uniform rewards. Finally, some works propose to mitigate this issue by selectively choose the prompt batch. For instance, Qu et al. (2025) employ a Bayesian framework to predict a prompt’s pass rate and selectively sample informative prompts during online training. Our work differs by tackling the problem at the collection stage itself, ensuring a sufficient signal is gathered adaptively rather than correcting for its absence afterward. Similarly, Shi et al. (2025) propose to use an adaptive curriculum learning to select prompts with suitable difficulty during the online RL training. Zhang et al. (2025) also develop a curriculum learning methods to select training prompts of intermediate difficulty in a two-stage manner. Zheng et al. (2025b) use a dictionary-based approach to record historical reward from the last epoch and skip uninformative prompts.

Learning objective in RLVR. A key contribution of this work is the systematic formulation of non-linear RL objectives and the weighted gradient estimators that arise from them. Our theoretical framework builds directly on the foundational insights of Yao et al. (2025), which is one of the earliest works to recognize that RLVR can be viewed through a prompt-dependent weighting scheme from a theoretical perspective. In particular, Yao et al. (2025) interpret rejection sampling fine-tuning (RAFT) (Dong et al., 2023) under an Expectation-Maximization (EM) view (Singh et al., 2023; Zhong et al., 2025) and make the important observation that the true expected gradient in EM-RAFT is weighted by 1/pθ(x). They further characterize the variance structure under this weighted gradient and show that the optimal sampling budget should scale with 1/ pθ(x). Motivated by their results, our early attempts tried to extend their optimal variancereduction estimator to the standard Reinforce algorithm, but this turned out to be infeasible. This led us to

analyze the log-objective for Reinforce, which naturally recovers a variance structure similar to EM-RAFT. Our framework also captures and generalizes the result of Yao et al. (2025) (see Appendix B for details).

From an algorithmic perspective, Yao et al. (2025) propose an improved RAFT variant with variancereduced gradient estimates. However, their method follows a two-stage “explore-then-exploit” procedure: a small portion of the budget is used to estimate pass rates (often offline), and the remaining budget is allocated based on these estimates. This design has a key limitation—when the initial budget is small, the pass-rate estimate has very high variance, especially for difficult prompts (low pi), where the initial sample almost always contains zero positive responses. In contrast, we study the more general Reinforce algorithm, clarify the connection between explicit weighting and implicit weighted sampling, and provide two practical realizations—an estimation-based method and a model-free sequential sampling method—that mitigate this issue.

We also note a very recent theoretical note by Davis & Recht (2025), which appeared concurrently with our second pre-print version. Their work also observes that non-linear objectives can be linked to heuristic weighting schemes used in algorithms such as GRPO. In comparison, we not only establish the theoretical link but also explicate the duality between optimizing this objective via explicit weights versus adaptive sampling. Furthermore, we go beyond theory to translate these insights into a practical, efficient algorithm, Reinforce-Ada, and rigorously validate its effectiveness through extensive experiments.

GRPO variant designs. Our work is orthogonal to another line of research focusing on modifying the policy gradient algorithm itself, such as innovations in advantage estimation and clipping mechanisms (Hu, 2025; Zhu et al., 2025; Zheng et al., 2025a; Huang et al., 2025; Chu et al., 2025). While these methods refine the core update rule, we focus on data generation and construction pipeline. Our adaptive sampling framework is complementary to these algorithmic improvements and is possible to be combined with them.

### 6 Discussion and End Note

In this work, we presented a general framework for optimizing non-linear RL objectives, identifying adaptive sampling as a robust strategy to address the signal loss problem inherent in standard Reinforce with group sampling algorithm. Guided by this framework, we introduced Reinforce-Ada, a family of algorithms that dynamically allocate inference budgets based on prompt difficulty. By prioritizing prompts that require more exploration, our method efficiently constructs informative training groups, recovering valid learning signals that are typically lost in uniform sampling. Designed as a lightweight, drop-in replacement for the standard RL training loop, Reinforce-Ada delivers consistent and robust improvements across diverse foundation models, accelerating convergence and boosting final performance. These benefits come with only moderate computational overhead, offering a scalable and theoretically grounded alternative to the brute-force approach of uniformly large group sizes.

We view this work as part of the broader recent effort on data curation for online reinforcement learning. Recent studies have explored macro, prompt-level strategies, such as curriculum learning (Zhao et al., 2024; Shi et al., 2025; Zhang et al., 2025), to shape the distribution of training data during online learning. In contrast, our contribution operates at the response-sampling level, focusing on how to construct effective learning signals within each prompt. However, as discussed in Section 4.3, the relative difficulty of the prompt set evolves alongside model training, and this interplay critically affects both learning dynamics and final performance of our method. Moreover, while our experiments are restricted to the math domain due to resource constraints, real-world post-training systems require data curation as a holistic challenge spanning the entire pipeline. We hope that the adaptive sampling framework can serve as an effective building block in this broader ecosystem when combined with complementary approaches from the literature.

### Acknowledgment

The authors would like to thank Ziniu Li for the insightful discussions.

### References

Mislav Balunovi´c, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovi´c, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions, February 2025. URL https://matharena.ai/.

Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Damek Davis and Benjamin Recht. What is the objective of reasoning with reinforcement learning? arXiv preprint arXiv:2510.13651, 2025.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, KaShun SHUM, and Tong Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https: //openreview.net/forum?id=m7p5O7zblY.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024.

Yunzhen Feng, Ariel Kwiatkowski, Kunhao Zheng, Julia Kempe, and Yaqi Duan. Pilaf: Optimal human preference sampling for reward modeling. arXiv preprint arXiv:2502.04270, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.

Jian Hu, Mingjie Liu, Ximing Lu, Fang Wu, Zaid Harchaoui, Shizhe Diao, Yejin Choi, Pavlo Molchanov, Jun Yang, Jan Kautz, and Yi Dong. Brorl: Scaling reinforcement learning via broadened exploration, 2025. URL https://arxiv.org/abs/2510.01180.

Wenke Huang, Quan Zhang, Yiyang Fang, Jian Liang, Xuankun Rong, Huanjin Yao, Guancheng Wan, Ke Liang, Wenwen He, Mingjun Li, et al. Mapo: Mixed advantage policy optimization. arXiv preprint arXiv:2509.18849, 2025.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https://github. com/huggingface/open-r1.

Hynek Kydlı´ˇcek. Math-Verify: Math Verification Library. URL https://github.com/huggingface/ math-verify.

Thanh-Long V Le, Myeongho Jeon, Kim Vu, Viet Lai, and Eunho Yang. No prompt left behind: Exploiting zero-variance prompts in llm reinforcement learning via entropy-guided advantage shaping. arXiv preprint arXiv:2509.21880, 2025.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.

Yizhi Li, Qingshui Gu, Zhoufutu Wen, Ziniu Li, Tianshun Xing, Shuyue Guo, Tianyu Zheng, Xin Zhou, Xingwei Qu, Wangchunshu Zhou, et al. Treepo: Bridging the gap of policy optimization and efficacy and inference efficiency with heuristic tree-based modeling. arXiv preprint arXiv:2508.17445, 2025.

Gongrui Nan, Siye Chen, Jing Huang, Mengyu Lu, Dexun Wang, Chunmei Xie, Weiqi Xiong, Xianzhou Zeng, Qixuan Zhou, Yadong Li, et al. Ngrpo: Negative-enhanced group relative policy optimization. arXiv preprint arXiv:2509.18851, 2025.

Yun Qu, Qi Wang, Yixiu Mao, Vincent Tao Hu, Bj¨rn Ommer, and Xiangyang Ji. Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models? arXiv preprint arXiv:2507.04632, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Y Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Ruizhe Shi, Runlong Zhou, and Simon S Du. The crucial role of samplers in online direct preference optimization. arXiv preprint arXiv:2409.19605, 2024.

Taiwei Shi, Yiyang Wu, Linxin Song, Tianyi Zhou, and Jieyu Zhao. Efficient reinforcement finetuning via adaptive curriculum learning. arXiv preprint arXiv:2504.05520, 2025.

Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, et al. Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585, 2023.

Aleksandrs Slivkins et al. Introduction to multi-armed bandits. Foundations and Trends® in Machine Learning, 12(1-2):1–286, 2019.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. 2023.

Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025.

Yixuan Even Xu, Yash Savani, Fei Fang, and Zico Kolter. Not all rollouts are useful: Down-sampling rollouts in llm reinforcement learning. arXiv preprint arXiv:2504.13818, 2025.

Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun Ma, and Bo An. Simpletir: Endto-end reinforcement learning for multi-turn tool-integrated reasoning. arXiv preprint arXiv:2509.02479, 2025.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.

Jiarui Yao, Yifan Hao, Hanning Zhang, Hanze Dong, Wei Xiong, Nan Jiang, and Tong Zhang. Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and rl. arXiv preprint arXiv:2505.02391, 2025.

Chenlu Ye, Zhou Yu, Ziji Zhang, Hao Chen, Narayanan Sadagopan, Jing Huang, Tong Zhang, and Anurag Beniwal. Beyond correctness: Harmonizing process and outcome rewards through rl training. arXiv preprint arXiv:2509.03403, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Chuheng Zhang, Wei Shen, Li Zhao, Xuyun Zhang, Lianyong Qi, Wanchun Dou, and Jiang Bian. Policy filtration in rlhf to fine-tune llm for code generation.

Ruiqi Zhang, Daman Arora, Song Mei, and Andrea Zanette. Speed-rl: Faster training of reasoning models via online curriculum learning. arXiv preprint arXiv:2506.09016, 2025.

Zirui Zhao, Hanze Dong, Amrita Saha, Caiming Xiong, and Doyen Sahoo. Automatic curriculum expert iteration for reliable llm reasoning. arXiv preprint arXiv:2410.07627, 2024.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025a.

Haizhong Zheng, Yang Zhou, Brian R Bartoldson, Bhavya Kailkhura, Fan Lai, Jiawei Zhao, and Beidi Chen. Act only when it pays: Efficient reinforcement learning for llm reasoning via selective rollouts. arXiv preprint arXiv:2506.02177, 2025b.

Han Zhong, Yutong Yin, Shenao Zhang, Xiaojun Xu, Yuanxin Liu, Yifei Zuo, Zhihan Liu, Boyi Liu, Sirui Zheng, Hongyi Guo, et al. Brite: Bootstrapping reinforced thinking process to enhance language model reasoning. arXiv preprint arXiv:2501.18858, 2025.

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347, 2025.

### A Authorship and Credit Attribution

All authors provided valuable contributions to this project, each bringing unique expertise and insights that were crucial for its success.

WX proposed the project idea, initiated and organized the project, and developed the core sequential sampling algorithm; developed the framework of non-linear RL objective; implemented the initial codebase, processed the data, and ran proof-of-concept experiments to validate its effectiveness; drafted the initial version of the paper, with subsequent contributions and revisions from co-authors.

CY proposed the idea of global normalization; jointly developed the core sequential sampling algorithms, including the codes for Reinforce-Ada-Seq-pos, Reinforce-Ada-Seq-balance, global normalization, and verification; ran the experiments for Qwen2.5-Math-1.5B and Qwen2.5-Math-7B; contributed to the paper writing.

BL processed the data; ran the experiments for Qwen2.5-Math-7B, Llama, Qwen3-4B; provided the released version of code (including the Tinker version); and contributed to the paper proofreading.

HD initiated, coordinated, and drove the project; provided insights about the algorithm and project design; implemented the production-ready Reinforce-Ada; developed and refactored a scalable and configurable codebase; conducted key experiments for Qwen2.5-Math-1.5B; made substantial contributions to the manuscript writing.

XX, CM, JB, NJ, TZ are senior authors, supported and advised the work, provided resources, and suggested experiments and improvements to the writing.

### B Variance-Reduction Gradient Estimator under the Log-objective

In this section, we connect our general framework to the variance-reduction principle explored in Yao et al. (2025). We generalize their findings from the specific case of Rejection Sampling Fine-Tuning (RFT) to the general Reinforce algorithm with a baseline.

We recall that Yao et al. (2025) demonstrated that RFT can be viewed as maximizing the log-objective Jlog(θ) = E[log pθ(x)]. Here, we analyze the gradient estimator for this objective under the general Reinforce algorithm with baseline:

1 p(x) · Ea∼π

θ(·|x) ∇θ log πθ(a|x) · (r(x,a) − p(x)) . (4)

glog(x) =

#### B.1 Optimal Budget Allocation

Our goal is to estimate the total gradient for a batch gbatch = i glog(xi) with minimum variance, given a total budget N. Suppose we have a total of N samples to distribute across B prompts x1,...,xB. Using ni samples for prompt xi, the batch gradient estimator is

1 B

B

gˆbatch =

i=1

1 nipi

ni

∇θ log π(aij|xi)(rij − pi), (5)

j=1

where pi = pθ(xi).

Crucially, note the distinction from Section 2: In this estimator, the optimization of the log-objective is captured entirely by the explicit factor 1/pi. Since the inner sum is normalized by ni, the allocation ni does not affect the optimization objective (it is an unbiased estimator for any ni ≥ 1). Therefore, the role of ni here is purely to minimize the variance of the estimator. We seek the allocation (n1,...,nB) minimizing

Optimal Inference Budget Allocation (N=4096, B=512)

EM-RAFT: n 1p

50

Reinforce with Baseline: n 1 p p

Allocatedsamplesn_i

40

30

20

10

0

0.0 0.2 0.4 0.6 0.8 1.0

Pass rate p

- Figure 7: The theoretically optimal inference budget allocation of RAFT and Reinforce with baseline algorithms under log-objective.

the total gradient variance:

min

n1,...,nB

subject to

σg2(xi) nip2i

B

V =

, (6)

i=1

B

ni = N and ni ≥ 1.

i=1

Here σg2(xi) is the variance of the baselined Reinforce gradient ∇θ log π(aij|xi)(rij − pi). Assuming gradient norms are similar across prompts (see the derivation of Proposition 1 in Yao et al. (2025) for details), we have

p(x) (RFT without baseline), p(x) · (1 − p(x)) (Reinforce with baseline).

σg2(x) ∝

Solving Equation (6) via Lagrange multipliers gives the optimal allocation rule:

σg(xi) pi

, (7)

n⋆i ∝

which yields

, Reinforce: n⋆i ∝ 1−p

RFT: n⋆i ∝ p1

pi .

i

i

This shows that incorporating a baseline reduces gradient variance and thus changes the optimal sampling strategy. Combining these findings, we can formulate a reasonable variance-reduction algorithm.

#### Reinforce with baseline. At each iteration t:

- 1. Allocate inference budgets as ni ∝ N · 1−pθ

t(xi)

pθt(xi) .

- 2. For each prompt xi, compute the advantage:

At(xi,aij) =

1 nipi · (rij − pθ

t

(xi)) ∝

1 pθ

t

(xi) · (1 − pθ

t

(xi)) · (rij − pθ

t

(xi)).

- 3. Update the policy using the weighted gradient.

- Remark 1 (Recovering the GRPO Advantage). The derived advantage term √r−p p(1−p)

is exactly equivalent

to normalizing the reward by its standard deviation, since Std[r] = p(1 − p) for binary rewards. This result provides an interesting explanation for the heuristic used in algorithms like GRPO (Shao et al., 2024), which normalize advantages by the group standard deviation.

- Remark 2 (The Necessity of the Log-Objective). It is important to note that the log-objective factor (1/p) in Equation (5) is essential for this result. If we were optimizing the standard objective J = E[p] (where the

2 g(xi)

2 g(xi)

ni instead of σ

weight is 1), the variance term would be σ

nip2i . Minimizing that variance would lead to allocating more budget to prompts with maximum variance (i.e., p ≈ 0.5) rather than the difficult prompts (p → 0). Thus, the log-objective is a prerequisite for focusing the variance-reduction effort on difficult problems.

### C Ablation Studies

#### C.1 Adaptive Sampling Solves More Difficult Prompts

###### Model Algorithm Math500 Minerva Math Olympiad Bench AIME-like Weighted Average

GRPO-n4 74.2 34.4 38.4 16.2 45.3 Reinforce-Ada-Seq-pos 75.8 35.7 38.6 16.5 46.1 (+0.8) Reinforce-Ada-Seq-balance 77.4 36.5 40.5 17.5 47.6 (+2.3)

Qwen2.5-Math-1.5B

GRPO 71.0 31.8 34.3 13.8 41.9 Reinforce-Ada-Seq-pos 73.9 33.1 36.4 16.4 44.6 (+2.7) Reinforce-Ada-Seq-balance 74.7 33.7 38.7 17.6 45.5 (+3.6)

Qwen2.5-Math-1.5B (hard)

GRPO 82.2 44.7 45.6 23.2 53.3 Reinforce-Ada-Seq-pos 82.7 45.1 46.7 23.7 54.2 (+0.9) Reinforce-Ada-Seq-balance 84.0 45.2 47.1 23.7 54.6 (+1.3)

Qwen2.5-Math-7B

GRPO 80.7 42.8 42.9 21.8 51.3 Reinforce-Ada-Seq-pos 82.4 43.1 45.0 22.2 52.8 (+1.5) Reinforce-Ada-Seq-balance 83.1 43.4 46.4 24.9 53.9 (+2.6)

Qwen2.5-Math-7B (hard)

- Table 4: Ablation study on the impact of prompt difficulty. The gains become more pronounced on the more challenging training set, where our methods allocate more inference budget to difficult prompts to recover a valid learning signal, while GRPO samples all prompts uniformly.

The impact of prompt set difficulty After ∼200 steps (Fig. 5, right, second row), most prompts become easy—two rounds already satisfy the positive quota—so additional sampling brings diminishing returns. To stress-test our method, we construct a hard subset that keeps only prompts with 1-2 correct out of 16 initial samples. As shown in Fig. 8, adaptive sampling yields a much larger margin over GRPO on this challenging set, and the gap widens late in training. This is also reflected on the final model performance reported in Table 4.

Intuitively, harder prompts reduce all-correct saturation and keep uncertainty high; Reinforce-Ada continues to mine informative negatives and avoids premature deactivation, sustaining exploration and gradient signal.

To better understand how adaptive allocation reshapes the learning dynamics, we randomly sample a subset of 5000 prompts from the hard training set. We partition the 5000-prompt subset into four difficulty levels using the base model’s pass@16: (i) Extremely Hard (0), (ii) Hard (0–0.2], (iii) Medium (0.2–0.5], and (iv)

Qwen2.5-Math-1.5B: Training Step vs Reward

Qwen2.5-Math-1.5B-Hard: Training Step vs Reward

0.30

GRPO

0.45

Reinforce-Ada-Seq-Pos

Reinforce-Ada-Seq-Balance

0.25

0.40

Reinforce-Ada-Est

0.35

0.20

Reward

Reward

0.30

0.15

0.25

GRPO

Reinforce-Ada-Seq-Pos

0.20

0.10

Reinforce-Ada-Seq-Balance

0.15

Reinforce-Ada-Est

0.05

0 50 100 150 200 250 300 Training Steps

0 50 100 150 200 250 300 Training Steps

- Figure 8: Ablation study on the prompt set difficulty. Left: prompt set with moderate difficulty. Right: challenging prompt set. The benefit of adaptive sampling is more obvious with challenging prompt set.

Method Extremely Hard Hard Medium Easy Overall

Base Model 0.0000 0.0889 0.2950 0.6151 0.1139 GRPO 0.3414 (+34.14) 0.4639 (+37.51) 0.6497 (+35.46) 0.6865 (+7.14) 0.4749 (+36.11) DAPO 0.3487 (+34.87) 0.4648 (+37.60) 0.6521 (+35.71) 0.5580 (-5.70) 0.4757 (+36.19) Reinforce-Ada 0.3674 (+36.74) 0.4825 (+39.37) 0.6579 (+36.29) 0.7153 (+10.02) 0.4933 (+37.94)

- Table 5: Accuracy and absolute improvements over the base model across difficulty levels. Seq-Balance achieves consistent improvements across all difficulty categories.

Easy (0.5–1.0]. The dataset is dominated by difficult items where 984 (19.7%) are extremely hard and 3,082 (61.7%) are hard. Meanwhile, medium and easy prompts account for 870 (17.4%) and 63 (1.3%), respectively. As expected, the base model’s accuracy rises monotonically with difficulty, achieving 0.00%, 8.89%, 29.50%, and 61.51% on the four groups.

Table 5 summarizes different models’ accuracy across four difficulty levels. All three RL methods substantially improve over the base model, but Reinforce-Ada-Seq-Balance achieves the highest accuracy in every category. The gains are especially pronounced on the Extremely Hard and Hard prompts—where the base model succeeds rarely or not at all—improving accuracy by +36.74 and +39.37 points, respectively. Reinforce-Ada-Seq-Balance also attains the best performance on Easy prompts (71.53%), whereas DAPO even hurts the performance in this regime. A likely reason is that DAPO tends to discard high-passrate prompts during training, while our balanced adaptive sampling variant allocates sufficient inference budget to preserve learning signals even for easy items.

#### C.2 Variants of Reinforce-Ada-Est

We also compare the effect of sampling and reweighting in Reinforce-Ada-Est. The training reward averaged at the prompt level is similar for both the hybrid strategy (wsample = 1/√p,wgrad = 1/√p) and the implicit weighting via sampling (wsample = 1/p,wgrad = 1). However, the batch-level reward under implicit weighting eventually decreases. This indicates that the method is overly aggressive in sampling negative samples, leading to excessive oversampling of difficult prompts. When the task distribution contains many hard examples, this issue becomes more pronounced, since simpler samples receive too few draws.

In fact, the number of samples drawn per prompt Nx, the total sampling budget Ntotal, and the reward rx satisfy

(rbatch − rprompt)Ntotal = Cov(Nx,rx),

A large gap between prompt level reward and batch level reward implies a strong negative covariance between Nx and rx, meaning low reward prompts dominate the sampling budget. Under a finite sampling constraint, it is therefore important to ensure broad prompt coverage and use a more moderate sampling strategy to avoid such imbalance and extreme behavior.

Qwen2.5-Math-1.5B: Training Step vs Reward (Batch)

Qwen2.5-Math-1.5B: Training Step vs Reward

0.50

wsample = 1/ p, wgrad = 1/ p wsample = 1/p, wgrad = 1

0.35

0.45

0.40

0.30

0.35

Reward

0.25

0.30

0.25

0.20

0.20

wsample = 1/ p, wgrad = 1/ p wsample = 1/p, wgrad = 1

0.15

0.15

0 100 200 300 400 500 Training Steps

0 100 200 300 400 500 Training Steps

Figure 9: Comparison with different sampling/reweighting strategy.

### D Simulation of Sampling Cost of Reinforce-Ada-Seq

To better understand the sampling cost induced by our sequential stopping rules, we conduct a set of numerical simulations illustrated in Figure 10. We evaluate two variants of our method: Reinforce-AdaSeq-pos, which continues sampling until K positive responses are observed, and Reinforce-Ada-Seqbalance, which stops once both K positive and K negative responses have been collected.

For each strategy, we consider three maximum exploration budgets Nmax ∈ {32,64,128} and use a batch size of kbatch = Nmax/8. We then vary the stopping requirement K and estimate the expected number of samples required as a function of the true success probability p ∈ (0,1). Because each response is a Bernoulli trial, the total number of required samples can be simulated accurately and efficiently.

Average Sample Cost — Until-K-Positive (N_MAX=32, Batch=4)

Average Sample Cost — Until-K-Positive (N_MAX=64, Batch=8)

Average Sample Cost — Until-K-Positive (N_MAX=128, Batch=16)

Fixed N = 16

Fixed N = 32

Fixed N = 64

- Until 1 Pos

- Until 2 Pos

- Until 1 Pos

- Until 2 Pos

- Until 1 Pos

- Until 2 Pos

30

60

120

Until 4 Pos Until 8 Pos

Until 4 Pos Until 8 Pos

Until 4 Pos Until 8 Pos

25

50

100

Avg.samplesE[N]

Avg.samplesE[N]

Avg.samplesE[N]

20

40

80

15

30

60

10

20

40

5

10

20

0.0 0.2 0.4 0.6 0.8 1.0 True pass rate p

0.0 0.2 0.4 0.6 0.8 1.0 True pass rate p

0.0 0.2 0.4 0.6 0.8 1.0 True pass rate p

Average Sample Cost — Until-K-Balanced (N_MAX=32, Batch=4)

Average Sample Cost — Until-K-Balanced (N_MAX=64, Batch=8)

Average Sample Cost — Until-K-Balanced (N_MAX=128, Batch=16)

Fixed N = 16

Fixed N = 32

Fixed N = 64

- Until 1 Pos & Neg

- Until 2 Pos & Neg

- Until 1 Pos & Neg

- Until 2 Pos & Neg

- Until 1 Pos & Neg

- Until 2 Pos & Neg

120

60

30

Until 4 Pos & Neg Until 8 Pos & Neg

Until 4 Pos & Neg Until 8 Pos & Neg

Until 4 Pos & Neg Until 8 Pos & Neg

100

50

25

Avg.samplesE[N]

Avg.samplesE[N]

Avg.samplesE[N]

80

40

20

60

30

15

40

20

10

20

10

5

0.0 0.2 0.4 0.6 0.8 1.0 True pass rate p

0.0 0.2 0.4 0.6 0.8 1.0 True pass rate p

0.0 0.2 0.4 0.6 0.8 1.0 True pass rate p

Figure 10: Estimated sampling cost under sequential stopping rules. The first row shows the expected number of responses required until observing K positive samples; the second row shows the cost until collecting K positive and K negative samples. Each column corresponds to a different maximum exploration budget Nmax ∈ {32,64,128}.

