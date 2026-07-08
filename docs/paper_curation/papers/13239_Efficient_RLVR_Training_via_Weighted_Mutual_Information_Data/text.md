## Efficient RLVR Training via Weighted Mutual Information Data Selection

Xinyu Zhou1 Boyu Zhu2 Haotian Zhang3 Huiming Wang3 Zhijiang Guo14

# arXiv:2603.01907v1[cs.LG]2Mar2026

### Abstract

Reinforcement learning (RL) plays a central role in improving the reasoning and alignment of large language models, yet its efficiency critically depends on how training data are selected. Existing online selection strategies predominantly rely on difficulty-based heuristics, favouring datapoints with intermediate success rates, implicitly equating difficulty with informativeness and neglecting epistemic uncertainty arising from limited evidence. We introduce INSIGHT, an INformationguided data SamplInG metHod for RL Training, grounded in a weighted mutual information objective. By modeling data outcomes with Bayesian latent success rates, we show that expected uncertainty reduction decomposes into complementary difficulty- and evidence-dependent components, revealing a fundamental limitation of difficultyonly selection. Leveraging this observation, INSIGHT constructs a stable acquisition score based on the mean belief of datapoints’ success rather than noisy sampled outcomes, and naturally extends to multi-rollout settings common in reinforcement learning with verifiable rewards (RLVR). Extensive experiments demonstrate that INSIGHT consistently achieves state-of-the-art performance and improves training efficiency, including a +1.41 average gain on Planning & Mathmatics benchmarks, +1.01 improvement on general reasoning, and up to ∼2.2x acceleration, with negligible additional computational overhead.

### 1. Introduction

Reinforcement learning (RL) has emerged as a central paradigm for aligning Large Language Models (LLMs) with human preferences and improving their reasoning abilities (Shao et al., 2024; DeepSeek-AI et al., 2025; Zeng et al., 2025; Luo et al., 2025b). Despite its success, RL is sensitive

1HKUST(GZ) 2UCL 3Kuaishou Technology 4HKUST. Correspondence to: Zhijiang Guo <zhijiangguo@hkust-gz.edu.cn>, Huiming Wang <huiming wang@mymail.sutd.edu.sg>.

Preprint. March 3, 2026.

to the choice of training data samples (Parashar et al., 2025; Qu et al., 2025; Shen et al., 2025a;b), and is also expensive in computations and memory usage for policy evaluation and updates, since it requires massive rollouts, especially in GRPO-style training (DeepSeek-AI et al., 2025). Training on static and uniformly sampled data distributions is inherently inefficient: substantial computation is wasted on tasks that the model has already mastered or that remain intractable given its current capability (Yu et al., 2025; Bae et al., 2025). Beyond increased training cost, this mismatch degrades optimization stability by reducing the effective batch size. Consequently, a key challenge in RL is to adaptively select tasks of appropriate difficulty, ensuring that learning remains efficient as the model’s capabilities evolve.

A growing body of work has explored data selection strategies to improve the efficiency of RL training. Early approaches primarily adopt offline curricula (Parashar et al., 2025; Wen et al., 2025; Shen et al., 2025b), organizing training data along a predefined progression from easy to hard. While effective in controlled settings, such curricula are fixed prior to training and therefore do not adapt to the model’s evolving capabilities. Motivated by this limitation, several recent works propose online data selection methods (Qu et al., 2025; Shen et al., 2025a; Yu et al., 2025) that dynamically select datapoints based on the current policy.

These methods, however, face an inherent trade-off between the cost of acquiring informative signals and the accuracy with which data utility can be estimated. Oversamplingbased approaches (Yu et al., 2025; Bae et al., 2025) address uncertainty by executing enlarged rollout batches to obtain reliable performance estimates, but this strategy introduces substantial computational overhead. In contrast, non-oversampling methods typically rely on a single proxy signal. In particular, difficulty-based heuristics (Shen et al., 2025a; Qu et al., 2025) prioritize datapoints whose empirical success rates lie near a target value (e.g., 0.5), implicitly treating difficulty as a surrogate for informativeness. While such heuristics can be effective during early stages of training, their utility may diminish as evidence accumulates: data can remain difficult yet provide a limited additional learning signal once uncertainty has been sufficiently reduced. Consequently, existing approaches either incur significant evaluation costs to maintain accurate estimates or rely on simplified heuristics that do not fully reflect the evolving

[Figure 1]

- Figure 1. Overview of the INSIGHT pipeline. INSIGHT maintains a Bayesian belief over data success rates, scores candidate datapoints using Weighted Mutual Information (WMI), and selects the top-M datapoints for RL training. Observed rewards update the posterior beliefs, enabling adaptive data selection that jointly accounts for data difficulty and accumulated evidence, unlike difficulty-only heuristics.

uncertainty structure during training.

To address the limitations, we propose INSIGHT, an efficient information-aware data selection method available for all RL training, based on a weighted mutual information objective. Unlike prior online methods that rely on empirical or sampled success rates and implicitly equate difficulty with informativeness, INSIGHT jointly considers both data difficulty and mutual information as shown in Figure 1, which yields an acquisition score prioritizing datapoints that are expected to produce the greatest reduction in uncertainty. We further provide a theoretical analysis showing how information gain decays with accumulated evidence while preserving meaningful data-dependent structure. Across a range of benchmarks, INSIGHT consistently achieves stateof-the-art performance (e.g., up to +1.41 gain on Planning & Mathmatics benchmarks and +1.01 improvement on general reasoning), and improves the efficiency by ∼2.2x acceleration, establishing a principled and effective alternative to difficulty-only data selection.

### 2. Related Work

RL for LLMs. RL has become a central paradigm for aligning LLMs with desired behaviors and unlock reasoning ability. Recently, Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated strong gains in structured reasoning domains (Li et al., 2025b), where reward signals can be automatically evaluated (DeepSeek-AI et al., 2025; Qwen et al., 2025; Luo et al., 2025a; Yang et al., 2025a). Among RL algorithms, GRPO (Shao et al., 2024) has attracted increasing attention by eliminating the costly value network. Building on these foundations, recent work has focused on mitigating training instability and bias, reducing computational overhead, and improving sample efficiency (Yu et al., 2025; Yue et al., 2025; Yang et al., 2025c;b). Furthermore, many recent studies have continued to advance the performance frontier across a wide range of domains and model scales (Luo et al., 2025c; Liang et al.,

2025; Ma et al., 2025; Meng et al., 2025), while complementary efforts focus on building scalable infrastructure for RL training (Sheng et al., 2025).

Data Selection for RL. Data selection has long been recognized as a critical factor in improving training efficiency and generalization in large-scale learning. A substantial body of work studies data selection in the context of pretraining and SFT, focusing on identifying informative, or high-quality samples (Xia et al., 2024; Zhou et al., 2025; Kwon et al., 2024; Koh & Liang, 2020; Ghorbani & Zou, 2019). RL is substantially more expensive than SFT due to repeated rollouts, policy evaluation, and credit assignment, making uniform sampling highly inefficient. As a result, selecting informative prompts is crucial for both training stability and computational efficiency. Early approaches adopt offline filtering, preselecting prompts based on heuristics such as difficulty or diversity (Li et al., 2025a; Ye et al., 2025). However, these methods lack adaptivity and often incur additional overhead for prompt assessment. To address this, recent work explores online data selection strategies that adapt to the current policy. Oversampling-based methods identify suitable data by executing oversized rollout batches and discarding uninformative samples (Yu et al., 2025; Liu et al., 2025), but this substantially increases computational cost. To reduce evaluation overhead, several methods attempt to predict task success rates and select prompts accordingly. Some frame task selection as a non-stationary multi-armed bandit problem (Shen et al., 2025a; Qu et al., 2025). However, these approaches typically rely on sampled or empirical success rates and implicitly equate difficulty with informativeness, neglecting how accumulated evidence reduces epistemic uncertainty over time. In contrast, our method INSIGHT explicitly selects data by maximizing expected variance reduction. This principled formulation decouples difficulty from accumulated evidence, enabling adaptive and information-aware data selection without auxiliary rollouts or heuristic difficulty prediction.

### 3. Preliminary

Setup & Notations. A datapoint τ in the reasoning tasks can be a form of a mathematical or logical problem, which belongs to the full pool of dataset T = {τi}Ni=1. The parameters of the LLM at t-th training step is denoted

- as πθ

t

. The selected data batch at t-th training step is

TtM = {τt,i}Mi=1 ⊂ T , where M is the batch size. At t-th time step, for a datapoint τ, the policy model πθ

t

generates K independent response rollouts yτt = {yτt,j}Kj=1.

Data Difficulty Modeling. For each datapoint τ, we associate it with a success rate ϕtτ ∈ [0,1] to demonstrate its difficulty under the current policy, which is unknown and thus regarded as a latent variable (Shen et al., 2025a; Qu et al., 2025). RLVR usually leverages a binary reward r ∈ {0,1} as the training signal. Therefore, each response yτt,j is examined with the ground-truth and scored with a binary reward:

rτt,j ∈ {0,1} ∼ Bernoulli(ϕtτ) (1)

For each datapoint τ, rτt = {rτt,j}Kj=1 denotes the set of rewards for K independent generated responses, and RMt = {rτt

t,i

}Mi=1 denotes the collected rewards for the data batch

- at step t. Thus, the likelihood of observing rτt given ϕtτ is binomial:

p(rτt|ϕtτ) =

K Sτt

t

t

τ(1 − ϕtτ)K−S

(ϕtτ)S

τ (2)

where Sτt = Kj=1 rτt,j means the success counts. The entire optimization history up to step t is noted as Ht = {TiM,RMi }ti=0

Bayesian Online Data Selection. A Bayesian surrogate model is widely adopted to dynamically estimate the success rate ϕtτ, and sample the most informative datapoints without requiring additional LLM inference (Shen et al., 2025a; Qu et al., 2025). Each datapoint τ is modeled as an arm in a stochastic multi-armed bandit, with an unknown success rate ϕtτ ∈ [0,1]. Selecting an arm corresponds to querying the current policy πθ

on datapoint τ and observing binary feedback rτ ∈ {0,1} indicating success or failure. Unlike standard bandit formulations that aim to maximize cumulative reward, our objective is to adaptively select data that yield the most informative learning signals for updating the model (Bae et al., 2025; Chen et al., 2021).

t

A natural way to model the initial success rate is via a Beta prior distribution (Shen et al., 2025a; Qu et al., 2025), due to the straightforward inference and closed-form posterior updates:

ϕ0τ ∼ Beta(ατ0,βτ0) (3)

where ατ0 and βτ0 represent the accumulated counts of successes and failures, respectively, which are typically set to

(1,1) for a uniform prior. Then, the posterior distribution of ϕtτ given observations is formulated as:

p(ϕtτ|Ht) ∝ p(rτt|ϕtτ) · p(ϕtτ|Ht−1) (4)

Note that p(ϕtτ|Ht−1) ∼ Beta(ατt ,βτt) represents the conditional prior. The posterior of ϕ would also follow the Beta distribution, since the Beta distribution is conjugate to the Bernoulli likelihood in Equation 2:

ϕtτ|Ht ∼ Beta(ατt + Sτt,βτt + K − Sτt) (5)

Then ατt+1 = ατt + Sτt,βτt+1 = βτt + K − Sτt serve as the prior for the next step. The temporal discounting techniques are also applied for better stability:

ατt+1 = λ · ατt + (1 − λ) · ατ0 + Sτt, βτt+1 = λ · βτt + (1 − λ) · βτ0 + K − Sτt

(6)

Previous works (Bae et al., 2025; Qu et al., 2025; Shen et al., 2025a) all claim that the problems with mid-level difficulty (i.e., near a target value ϕ∗ ≈ 0.5) can yield the most informative gradients for RL training, from empirical experiments. As a result, those problems whose sampled success rate ϕˆtτ is closest to ϕ∗, will be ranked as top and selected for the following training.

### 4. Limitation of Difficulty-Only Data Selection

While potentially effective in early training, difficulty-only heuristics primarily target aleatoric uncertainty (i.e., the intrinsic stochasticity of reward outcomes for a given datapoint) by favoring prompts with high outcome variability, implicitly equating difficulty with informativeness, but ignore the epistemic uncertainty, which arises from limited rollouts and captures uncertainty about the data latent success rate under the current policy (Chan et al., 2024; H¨ullermeier & Waegeman, 2021). As evidence accumulates, epistemic uncertainty about the underlying success rate diminishes, and prompts may remain difficult yet yield limited gains for parameter estimation.

We posit that a more accurate estimation of ϕtτ leads to more data-efficient selection for policy optimization. In RLVR, policy gradients are driven by reward feedback from sampled trajectories. When the surrogate estimate of task success is noisy or poorly calibrated, the resulting learning signal becomes unreliable. Prioritizing datapoints that maximize expected uncertainty reduction in ϕtτ would promote the collection of reward signals that are more reliable and discriminative, yielding gradients that better reflect true task difficulty and thereby improving downstream policy learning performance. To quantify the improvement in the estimation of ϕ after observing the reward signal r, we consider the expected variance reduction for ϕ as follows:

∆V (τ) = Varprior(ϕτ) − Er[Varposterior(ϕτ|r)] (7)

[Figure 2]

- Figure 2. Expected variance reduction as a function of prior mean ϕ¯τ and accumulated evidence n. While difficulty-based heuristics focus solely on ϕ¯τ ≈ 0.5, the expected uncertainty reduction decays rapidly with evidence, revealing a fundamental limitation of difficulty-only selection.

which captures the expected decrease in uncertainty of ϕτ after observing all possible reward outcomes. Without loss of generality, we consider the binary reward signal and K = 1, which is well-suited to most common RLVR settings and used to illustrate evidence-dependent decay.

According to conditional expectation, we can expand Equation 7 and simplify it as:

αβ (α + β)2(α + β + 1)2

∆V (τ) =

ϕ¯τ · (1 − ϕ¯τ) (n + 1)2

=

(8)

where n = α+β and ϕ¯τ = α+αβ represents the mean of ϕ’s distribution. The full derivation is included in Appendix A.

Analysis. Existing difficulty-based data selection relies on a sampled success rate ϕˆtτ, and prioritize data whose sampled value lies closest to a target difficulty level ϕ∗. However, as visualized in Figure 2, expected variance reduction depends not only on ϕ¯τ but also critically on the accumulated evidence n = α + β. In particular, even when ϕ¯τ ≈ 0.5, the contribution to uncertainty reduction becomes negligible once n is large. In conclusion, while ϕˆtτ concentrates around ϕ¯τ in expectation, the difficulty heuristic ignores both sampling variability and the evidence-dependent decay, which would repeatedly select well-estimated yet mid-level difficult data, producing limited information gain in practice.

### 5. Method

The preceding analysis exposes a key limitation of difficultyonly data selection: by relying on sampled success rates, purely difficulty-based heuristics conflate outcome variability with informativeness, optimizing only a partial aspect of uncertainty reduction. Our derivation shows that the distributional mean of the prior belief and accumulated evidence governs informativeness. Motivated by the decomposition of expected variance reduction into a difficulty-dependent and another evidence-dependent component, we decouple

these two factors and design a new acquisition score:

##### A(τ) ∝ w(·)

##### · I(·)

Aleatoric Exploitation

Epistemic Exploration

(9)

In the following, we detail the two components of the proposed acquisition score.

#### 5.1. Mutual Information for Epistemic Exploration

Mutual information provides a principled measure of the uncertainty reduction, as it directly captures the expected decrease in posterior uncertainty about ϕτ induced by observations. Let Φτ denote the random variable corresponding to the latent success rate ϕτ, and let R denote the reward outcome. The mutual information between R and Φτ is defined as:

##### I(R;Φτ) = H(Φτ) − ER[H(Φτ|R = r)] (10)

This represents the expected reduction in uncertainty about Φτ caused by observing the reward, indicating “how informative will this datapoint be”, which serves the Epistemic Exploration term in Equation 9. H(·) denotes the entropy:

1

H(Φτ) = −

f(ϕτ)lnf(ϕτ)dϕτ

0

= lnB(ατ,βτ) + (ατ + βτ − 2)ψ(ατ + βτ) − (ατ − 1)ψ(ατ) − (βτ − 1)ψ(βτ)

(11) where f(·) denotes the probability density function, B(α,β) is the beta function, and ψ(·) denotes the digamma function. Full derivation is in Appendix B.

Extend to multi-rollouts. In practice, for example, using GRPO for RLVR training, each datapoint τ is typically evaluated multiple times to obtain K independent responses. To account for this, we extend the above single-observation mutual information and variance reduction derivations to the multi-rollout setting.

Let Sτ = Kk=1 Rk denote the number of successes observed, given K independent responses with the correspond-

ing rewards R1:K = (R1,...,RK) at a datapoint τ, which follows the Binomial distribution as in Equation 2. Then we further derive it as follows:

P(Sτ = s|K,ατ,βτ) =

=

1

P(Sτ = s|Φτ = ϕ,K)P(Φ = ϕ)dϕ

0

B(ατ + s,βτ + K − s) B(ατ,βτ)

K s

(12)

[Figure 3]

Algorithm 1 INSIGHT Online Data Selection

Input: Datapoint pool T = {τ}Ni=1; Prior Beta parameters α,β; Hyperparameters of weighted function (w(ϕ¯τ)) η,µ; Larger candidate batch size Mˆ ; selected batch size M; Policy model πθ

; Total training steps T. Output: Finetuned model πθ

0

T

∀τ ∈ T , initialize Beta parameters (ατ0,βτ0) ← (α,β); for t = 0 to T − 1 do

Randomly sample a larger candidate set TtMˆ ⊂ T ; for τˆt,j ∈ TtMˆ do

Figure 3. Weighted functions w(ϕ¯) under different η, µ.

Compute WMI score A(ˆτt,j) via Equation 16; end for # Rank Datapoints by WMI Score

Interpretation. This result demonstrates that the information gained from an additional observation decreases as prior evidence accumulates, reflecting the diminishing epistemic uncertainty of well-understood datapoints.

Select top datapoints set TtM ⊂ TtMˆ by Equation 17; for τt,i ∈ TtM do

# Generate Responses and Compute Rewards Generate responses yτt

; Compute rewards rτt

}Kj=1 by πθ

= {yτt,j

#### 5.3. Weighted Function for Aleatoric Exploitation

t

t,i

t,i

}Kj=1;

= {rτt,j

t,i

t,i

While mutual information provides a principled measure of epistemic uncertainty reduction, it is agnostic to task difficulty and treats all outcome regimes symmetrically. To account for this, we introduce a weighted function w(·) that modulates the epistemic acquisition term with a controlled preference for informative difficulty regimes.

end for Update θt to θt+1 with an RL algorithm; # Posterior Update for τt,i ∈ TtM do

Update (ατt+1,βτt+1) via Equation 6; end for

#### end for

Motivation. The design of it is motivated by two complementary factors. First, effective data selection should favor prompts with sufficient outcome variability, as neardeterministic regimes provide a limited learning signal. Second, learning dynamics in reinforcement learning are known to benefit from curriculum-like progression (Parashar et al., 2025; Bae et al., 2025), and trivial tasks may mislead the model to exploit shortcuts to bypass meaningful reasoning (Laidlaw et al., 2025; Parashar et al., 2025).

Therefore, the mutual information can be written as:

I(R1:K;Φτ) = H(Φτ) − ES

[H(Φτ | Sτ)]

τ

K

= H(Φτ) −

P(Sτ = s | K,ατ,βτ)H(Φτ | Sτ = s)

s=0

K

B(ατ + s,βτ + K − s) B(ατ,βτ)

K s

H(Φτ | Sτ = s)

= H(Φτ) −

s=0

Inspired from Equation 8, unlike difficulty-only methods that operate on a sampled success rate, our weighting function is defined over the mean of the current prior belief of the latent success rate: ϕ¯τ = E[ϕτ], providing an expectationbased characterization of task difficulty. As a result, we design the weighted function as (shown in Figure 3):

(13)

This expression demonstrates that the information gain from K rollouts is governed by the expected reduction in posterior uncertainty about the latent success rate, averaged over all possible success counts.

#### 5.2. Asymptotic Analysis of Mutual Information

We next analyze the asymptotic behavior of the mutual information to characterize how epistemic uncertainty about the latent success rate decays as evidence accumulates.

Proposition 5.1. Let Φτ ∼ Beta(ατ,βτ) denote the latent success rate of a datapoint τ, and R ∈ {0,1} be a Bernoulli reward conditioned on Φτ. Define nτ = ατ + βτ. Then, as nτ → ∞, we have

1 nτ

) (14)

I(R;Φτ) ≈ O(

A complete derivation is provided in Appendix C.

·exp −η ϕ ¯τ − µ 2

w(ϕ¯τ) = ϕ ¯τ 1 − ϕ¯τ

High Variance Filter

Bias to Difficulty µ

(15)

where the first term focuses on regions with high outcome variability, the second term introduces a smooth bias towards a beneficial difficulty level µ, and another hyperparameter η controls the sharpness of curriculum bias.

#### 5.4. Weighted Mutual Information–Based Selection

Combining the preceding components, we define the final acquisition score, named Weighted Mutual Information

- Table 1. Evaluation results across planning and mathematics benchmarks. Accuracy is computed as the average pass@1 over 16 independent generations per problem. Bold indicates the best results, and gray denotes the oracle baseline. ↑ (↓) indicates the improvement(degradation) compared to the RANDOM baseline.

###### MODELS METHODS COUNTDOWN AIME24 AMC23 MATH500 MINERVA. OLYMPIAD. AVG. RANDOM 71.57 10.41 43.90 71.30 21.60 34.20 42.16

QWEN3-0.6B MOPPS 76.28 10.83 44.16 70.60 21.60 34.25 42.95 INVERSE-EVIDENCE 72.13 10.60 42.50 71.80 21.70 34.80 42.25 INSIGHT (Ours) 76.70(5.13↑) 11.90(1.49↑) 44.30(0.40↑) 71.75(0.45↑) 21.94(0.34↑) 34.80(0.60↑) 43.56(1.40↑) RANDOM 84.90 51.45 78.01 91.22 41.13 56.24 67.16

QWEN3-4B MOPPS 87.80 50.00 77.41 90.82 41.59 56.45 67.35 INVERSE-EVIDENCE 83.59 52.22 77.18 91.23 41.49 56.20 66.99 INSIGHT (Ours) 87.96(3.06↑) 53.75(2.30↑) 79.00(0.99↑) 91.22(0.00↑) 41.80(0.67↑) 57.00(0.76↑) 68.46(1.30↑)

DS 83.05 50.62 80.79 90.82 38.39 53.38 66.17 RANDOM 79.02 46.25 77.90 90.50 37.98 51.80 63.90

R1-DISTILL-QWEN-7B MOPPS 81.66 47.20 77.86 90.40 38.02 52.01 64.52 INVERSE-EVIDENCE 78.32 47.00 77.70 90.40 37.93 52.07 63.90 EXPECTED-DIFFICULTY 81.20 47.50 78.10 90.43 38.39 52.25 64.64 INSIGHT (Ours) 81.73(2.71↑) 47.71(1.46↑) 78.40(0.50↑) 90.45(0.05↓) 39.06(1.08↑) 52.50(0.70↑) 64.98(1.08↑)

(WMI), for a datapoint τ as:

A(τ) = w(ϕ¯τ) · I(R1:K,Φτ) (16)

At each iteration, we will select the datapoints corresponding to the largest WMI score. Formally, given a larger

candidate set TtMˆ , we choose:

TtM = Top-M τ ∈ TtMˆ |A(τ) (17)

where Mˆ ≫ M. Then top-M selected datapoints are used for rollouts and policy training. The full algorithm of INSIGHT is described in Algorithm 1, which can be easily integrated with any RLVR algorithm.

### 6. Experiments

In this section, we conduct comprehensive experiments to evaluate various data selection approaches on RL training and validate the effectiveness of our proposed method.

#### 6.1. Experimental Setup

We evaluate INSIGHT across representative reasoning tasks: planning, mathematics, and general reasoning. To examine its effectiveness, we adopt diverse LLMs with different scales, from 0.6B to 7B in size. For RL tuning, we choose one of the most commonly used algorithms: GRPO (Shao

- et al., 2024). Test accuracy is reported as average pass@1 over 16 independent generations per problem. More implementation details are included in Appendix D.

Reasoning Tasks. (1) Planning: We adopt the Countdown task, which requires combining given numbers using basic arithmetic operations to reach a target value. Training is conducted on the subset of the Countdown dataset (Pan

- et al., 2025), and test is evaluated on the held-out split. (2) Mathematics: We train the LLMs on the DeepScaler training dataset (Luo et al., 2025b), which consists of approximately 40.3K mathematics problem-answer pairs. Then

we evaluate the final performance on common benchmarks, including AIME24, AMC23, MATH500 (Lightman et al., 2023), Minerva Math (Minerva.) (Lewkowycz et al., 2022), and OlympiadBench (Olympiad.) (He et al., 2024). (3) General Reasoning: We utilize the WebInstruct-verified dataset (Ma et al., 2025), which encompasses a broad range of domains (e.g., physics, chemistry, finance), and evaluate the performances on MMLU and GPQA-Main (Hendrycks et al., 2021; Rein et al., 2023).

Models. For Countdown and Mathematics tasks, we include three LLMs in different scales: Qwen3-0.6B, Qwen34B (Yang et al., 2025a), and DeepSeek-R1-Distil-Qwen7B; and for General Reasoning, we also use three LLMs: Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B (Yang et al., 2025a).

Baselines. (1) RANDOM, which selects datapoints uniformly at random, serving as a non-adaptive baseline; (2) MOPPS (Qu et al., 2025), a recent state-of-the-art Bayesian online sampling method, prioritizes datapoints whose estimated success rates are closest to targeted difficulty (i.e., 0.5); (3) INVERSE-EVIDENCE, which is a theory-inspired heuristic baseline derived from the expected variance reduction in Equation 8. It selects datapoints based solely on the inverse of accumulated evidence, favoring prompts with smaller nτ, under the intuition that less-observed datapoints carry higher epistemic uncertainty, while ignoring task difficulty; (4) EXPECTED-DIFFICULTY, which selects datapoints whose mean success rate is closest to the target difficulty 0.5, removing sampling variance in MOPPS but still ignoring evidence accumulation; and (5) Dynamic Sampling (DS) (Yu et al., 2025), which over-samples datapoints and filter out those without effective gradients based on their exact evaluation. It is worth noting that DS is extremely time-consuming, and we view it as an oracle baseline rather than outperforming it in test accuracy (refer to Appendix E

- Table 2. Evaluation results across general-reasoning benchmarks, via LM-Evaluation-Harness framework. Bold indicates the best results. ↑ indicates the improvement compared to the RANDOM baseline.

###### MMLU

MODELS METHODS

GPQA AVG. HUMANITIES OTHER SOCIAL-SCI. STEM

RANDOM 38.83 47.22 52.32 40.95 26.34 41.13

- QWEN3-0.6B MOPPS 39.17 47.12 51.12 38.69 27.90 40.80 INVERSE-EVIDENCE 38.55 45.19 49.76 36.95 29.46 39.98 INSIGHT (Ours) 39.21(0.38↑) 47.34(0.12↑) 52.84(0.52↑) 41.80(0.85↑) 29.50(3.16↑) 42.14(1.01↑) RANDOM 48.63 60.41 64.09 53.47 30.20 51.36

- QWEN3-1.7B MOPPS 48.63 60.32 64.38 53.35 30.10 51.35 INVERSE-EVIDENCE 48.80 60.40 64.41 54.14 29.69 51.48 INSIGHT (Ours) 49.00(0.37↑) 60.83(0.42↑) 65.00(0.91↑) 54.61(1.14↑) 30.36(0.16↑) 51.96(0.60↑)

RANDOM 59.13 71.07 77.87 68.25 34.38 62.14 MOPPS 59.62 71.00 78.00 69.01 33.26 62.17

QWEN3-4B INVERSE-EVIDENCE 59.36 71.12 78.03 68.79 33.48 62.16 EXPECTED-DIFFICULTY 59.60 71.22 77.93 68.63 33.50 62.17 INSIGHT (Ours) 59.70(0.57↑) 71.13(0.06↑) 78.10(0.23↑) 69.14(0.89↑) 35.04(0.66↑) 62.62(0.48↑)

[Figure 4]

Figure 4. Performance comparisons among different methods on the Countdown task. Our proposed INSIGHT outperforms the existing SOTA (MOPPS) and other baselines in both training efficiency and performance.

for the performance and runtime cost comparisons in detail).

#### 6.2. Main results

Planning & Mathematics. Table 1 reports results on planning and mathematical reasoning benchmarks for models from 0.6B to 7B parameters. Across all scales, INSIGHT consistently outperforms RANDOM and MOPPS, achieving the highest average accuracy in every setting. Gains are most pronounced on CountDown and AIME24, with improvements of up to +5.13 and +2.30 over RANDOM, respectively. The largest average improvement occurs on the 0.6B model (+1.40), with gains gradually diminishing as model size increases, though INSIGHT still yields a +1.08 average improvement on R1-Distill-Qwen-7B.

General Reasoning. Table 2 presents results on MMLU and GPQA. While absolute improvements are smaller than in mathematical reasoning, INSIGHT consistently achieves the strongest or near-strongest performance across all model sizes. Notable gains are observed on MMLU-STEM and GPQA, reaching +1.14 on Qwen3-1.7B and +3.16 on Qwen3-0.6B, respectively.

Discussions. First, improvements decrease with model

scale across all benchmarks, reflecting reduced headroom as larger models exhibit stronger priors and reasoning capabilities. Second, EXPECTED-DIFFICULTY consistently outperforms MOPPS, indicating that using the mean difficulty provides a more stable acquisition signal than samplingbased heuristics. In contrast, INVERSE-EVIDENCE performs comparably or even worse than RANDOM, suggesting that epistemic uncertainty alone is insufficient when decoupled from task difficulty. On general reasoning tasks, where outcome noise is higher, RANDOM occasionally surpasses MOPPS, highlighting the sensitivity of sampling-based difficulty heuristics to noisy surrogate signals.

Finally, we evaluate training efficiency on CountDown, an in-domain benchmark drawn from the same distribution as the RL training data, providing a controlled setting for comparing training efficiency. As shown in Figure 4, INSIGHT reaches competitive or superior performance with fewer training steps, yielding up to ∼2.2x, ∼1.5x, and ∼1.6x speedups on Qwen3-0.6B, Qwen3-4B, and R1Distill-Qwen-7B, respectively. Although MOPPS exhibits similar convergence behavior, INSIGHT demonstrates more consistent early-stage gains and slightly stronger final per-

- Table 3. Evaluation comparisons of WMI components across different models. Bold indicates the best results.

###### MODELS COMPONENT AIME24 AMC23 OLYMPIAD.

I(R1:K,Φτ) 9.37 43.52 34.01 QWEN3-0.6B w(ϕ¯τ) · I(R1:K,Φτ) 11.90 44.30 34.80

I(R1:K,Φτ) 51.66 76.43 56.24 QWEN3-4B w(ϕ¯τ) · I(R1:K,Φτ) 53.75 79.00 57.00

w(ϕ¯τ) 46.67 77.33 52.25 R1-DISTILL-7B I(R1:K,Φτ) 46.67 77.90 52.35 w(ϕ¯τ) · I(R1:K,Φτ) 47.71 78.40 52.50

formance, indicating that jointly modeling difficulty and epistemic uncertainty improves gradient efficiency.

### 7. Ablation Analysis

#### 7.1. Effects of WMI Components

We first analyze the contribution of each component (e.g., weighted function w(ϕ¯τ) and mutual information I(R1:K,Φτ)) in the WMI score of INSIGHT.

Results. Table 3 reports the component-wise ablation. Mutual information alone consistently underperforms the full WMI objective. While MI captures epistemic uncertainty in the latent success rate, it does not account for whether this uncertainty lies in difficulty regimes that yield informative policy gradients. Conversely, the difficulty weighting alone performs competitively but lacks explicit uncertainty awareness. Combining both components yields the most consistent performance across all benchmarks, demonstrating that jointly modeling epistemic uncertainty and task difficulty is crucial for effective data selection in RL.

#### 7.2. Expected vs. Sampled Difficulty in WMI

According to Equation 8 and Equation 15, WMI relies on the mean of the underlying success-rate distribution. In this experiment, we explore the effect of replacing ϕ¯τ with ϕˆτ ∼ Beta(ατ,βτ), which is always used in previous work. Results. As shown in Table 4, using the mean success rate ϕ¯τ consistently outperforms the sampled variant ϕˆτ across all models and benchmarks. This aligns with our theoretical analysis: expected variance reduction depends on the mean belief and accumulated evidence, whereas sampling introduces unnecessary noise that destabilizes data ranking. The results confirm that mean-based estimation yields a more reliable and effective acquisition signal than sampled success rates used in prior work.

#### 7.3. Effect of Difficulty Bias µ in WMI

The WMI objective incorporates a difficulty bias that favors tasks near a target success rate µ. We study the sensitivity of WMI to this design choice by varying µ, assessing both robustness and the trade-off between exploration and curriculum shaping.

- Table 4. Evaluation comparisons between ϕ¯τ and ϕˆτ variants in WMI across different models. Bold indicates the best results.

MODELS ϕ¯τ / ϕˆτ AIME24 AMC23 OLYMPIAD.

ϕˆτ 10.41 43.22 34.25 QWEN3-0.6B ϕ¯τ 11.90 44.30 34.80

ϕˆτ 51.45 77.48 56.33 QWEN3-4B ϕ¯τ 53.75 79.00 57.00

ϕˆτ 46.25 77.93 52.37 R1-DISTILL-7B ϕ¯τ 47.71 78.40 52.50

- Table 5. Evaluation comparisons of different µ in weighted function across different models. Bold indicates the best results.

MODELS µ AIME24 AMC23 OLYMPIAD.

0.1 10.41 44.80 34.75

QWEN3-0.6B 0.3 11.90 44.30 34.80 0.7 10.41 42.62 33.09 0.1 50.83 77.03 56.32

QWEN3-4B 0.3 53.75 79.00 57.00 0.7 52.91 77.41 56.58 0.1 46.87 78.23 52.08

R1-DISTILL-7B 0.3 47.71 78.40 52.50 0.7 47.91 78.46 52.25

Results. Table 5 shows LLMs’ behavior under the choices of the difficulty bias µ. Generally, µ = 0.3 achieves the strongest or near-strongest performance, indicating that prioritizing moderately challenging tasks provides an effective balance between exploration and curriculum shaping. In contrast, more extreme settings (µ = 0.1 or 0.7) tend to underperform, suggesting that overemphasizing very easy or very difficult tasks is less effective, and this phenomenon is more significant in smaller models (e.g., Qwen3-4B’s AIME24 score drops to 50.83 from 53.75 when µ = 0.1). The sensitivity to µ further diminishes as model size increases, with the 7B model exhibiting only minor variations, reflecting greater inherent robustness. We further analyze the impact of the candidate batch size Mˆ on model performance, with a detailed discussion provided in Appendix F.

### 8. Conclusion

We introduced INSIGHT, an information-guided data selection framework for efficient reinforcement learning of large language models. By analyzing expected variance reduction, we revealed why difficulty-only heuristics fail to capture true informativeness as evidence accumulates, and motivated a weighted mutual information objective that jointly accounts for task difficulty and epistemic uncertainty. INSIGHT operationalizes this insight using stable mean posterior beliefs and naturally extends to multi-rollout RLVR settings. Experiments across planning, mathematical, and general reasoning benchmarks show that INSIGHT consistently improves training efficiency and final performance over strong online baselines, particularly in smaller-model

and low-resource regimes. These results demonstrate that principled, information-aware data selection offers a robust alternative to heuristic difficulty-based sampling.

### Impact Statements

This paper presents work whose goal is to advance the field of machine learning by improving the efficiency of reinforcement learning through principled data selection. The proposed method focuses on algorithmic efficiency and does not introduce new model capabilities or application domains. While more efficient training may reduce computational costs and resource usage, we do not identify any specific societal or ethical risks beyond those commonly associated with large language model development.

Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen,

- Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen,
- R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye,
- S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen,

- X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang,

- X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang,

- X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang,
- Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong,

Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang,

- Z., and Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

### References

Bae, S., Hong, J., Lee, M. Y., Kim, H., Nam, J., and Kwak, D. Online difficulty filtering for reasoning oriented reinforcement learning, 2025. URL https://arxiv. org/abs/2504.03380.

Chan, M. A., Molina, M. J., and Metzler, C. A. Estimating epistemic and aleatoric uncertainty with a single model, 2024. URL https://arxiv.org/abs/ 2402.03478.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov,

- M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., HerbertVoss, A., Guss, W. H., Nichol, A., Paino, A., Tezak,
- N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/ 2107.03374.

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K.,

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602.

Ghorbani, A. and Zou, J. Data shapley: Equitable valuation of data for machine learning, 2019. URL https:// arxiv.org/abs/1904.02868.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z. L., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., Liu, J., Qi, L., Liu, Z., and Sun, M. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024. URL https://arxiv.org/abs/2402.14008.

Hendrycks, D., Burns, C., Basart, S., Critch, A., Li, J., Song, D., and Steinhardt, J. Aligning ai with shared human values. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

H¨ullermeier, E. and Waegeman, W. Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. Machine Learning, 110 (3):457–506, March 2021. ISSN 1573-0565. doi:

10.1007/s10994-021-05946-3. URL http://dx.doi. org/10.1007/s10994-021-05946-3.

Koh, P. W. and Liang, P. Understanding black-box predictions via influence functions, 2020. URL https: //arxiv.org/abs/1703.04730.

Kwon, Y., Wu, E., Wu, K., and Zou, J. Datainf: Efficiently estimating data influence in lora-tuned llms and diffusion models, 2024. URL https://arxiv.org/ abs/2310.00902.

Laidlaw, C., Singhal, S., and Dragan, A. Correlated proxies: A new definition and improved mitigation for reward hacking, 2025. URL https://arxiv.org/abs/ 2403.03185.

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T., Wu, Y., Neyshabur, B., Gur-Ari, G., and Misra, V. Solving quantitative reasoning problems with language models, 2022. URL https://arxiv.org/abs/2206.14858.

Li, X., Zou, H., and Liu, P. Limr: Less is more for rl scaling, 2025a. URL https://arxiv.org/abs/ 2502.11886.

Li, Z.-Z., Zhang, D., Zhang, M.-L., Zhang, J., Liu, Z., Yao, Y., Xu, H., Zheng, J., Wang, P.-J., Chen, X., et al. From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419, 2025b.

Liang, X., Li, Z., Gong, Y., Shen, Y., Wu, Y. N., Guo, Z., and Chen, W. Beyond pass@ 1: Self-play with variational problem synthesis sustains rlvr. arXiv preprint arXiv:2508.14029, 2025.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., and Dong, Y. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models, 2025. URL https://arxiv.org/abs/2505.

24864.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization, 2019. URL https://arxiv.org/abs/ 1711.05101.

Luo, M., Tan, S., Huang, R., Patel, A., Ariyak, A., Wu, Q., Shi, X., Xin, R., Cai, C., Weber, M., Zhang, C., Li, L. E., Popa, R. A., and Stoica, I. Deepcoder: A fully open-source 14b coder at o3-mini level, 2025a. Notion Blog.

Luo, M., Tan, S., Wong, J., Shi, X., Tang, W., Roongta, M., Cai, C., Luo, J., Zhang, T., Li, E., Popa, R. A., and Stoica, I. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025b. Notion Blog.

Luo, X., Huang, J., Zheng, W., Zhu, Q., Xu, M., Xu, Y., Fan, Y., Qin, L., and Che, W. How many code and test cases are enough? evaluating test cases generation from a binary-matrix perspective, 2025c. URL https:

//arxiv.org/abs/2510.08720.

Ma, X., Liu, Q., Jiang, D., Zhang, G., MA, Z., and Chen, W. General-Reasoner: Advancing LLM reasoning across all domains. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https: //openreview.net/forum?id=pBFVoll8Xa.

Meng, F., Du, L., Liu, Z., Zhou, Z., Lu, Q., Fu, D., Han, T., Shi, B., Wang, W., He, J., Zhang, K., Luo, P., Qiao, Y., Zhang, Q., and Shao, W. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning, 2025. URL https://arxiv.

org/abs/2503.07365.

Pan, J., Zhang, J., Wang, X., Yuan, L., Peng, H., and Suhr, A. Tinyzero. https://github.com/Jiayi-Pan/TinyZero, 2025. Accessed: 2025-01-24.

Parashar, S., Gui, S., Li, X., Ling, H., Vemuri, S., Olson, B., Li, E., Zhang, Y., Caverlee, J., Kalathil, D., and Ji, S. Curriculum reinforcement learning from easy to hard tasks improves llm reasoning, 2025. URL https:// arxiv.org/abs/2506.06632.

Qu, Y., Wang, Q., Mao, Y., Hu, V. T., Ommer, B., and Ji, X. Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models?, 2025. URL https: //arxiv.org/abs/2507.04632.

Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2025. URL https:

//arxiv.org/abs/2412.15115.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark, 2023. URL https://arxiv.org/abs/2311.12022.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo,

D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Shen, Q., Chen, D., Huang, Y., Ling, Z., Li, Y., Ding, B., and Zhou, J. Bots: A unified framework for bayesian online task selection in llm reinforcement finetuning. arXiv preprint arXiv:2510.26374, 2025a.

Shen, W., Pei, J., Peng, Y., Song, X., Liu, Y., Peng, J., Sun, H., Hao, Y., Wang, P., Zhang, J., and Zhou, Y. Skyworkr1v3 technical report, 2025b. URL https://arxiv. org/abs/2507.06167.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, pp. 1279–1297. ACM, March 2025. doi: 10.1145/ 3689031.3696075. URL http://dx.doi.org/10.

1145/3689031.3696075.

Wen, C., Guo, T., Zhao, S., Zou, W., and Li, X. Sari: Structured audio reasoning via curriculum-guided reinforcement learning, 2025. URL https://arxiv.org/ abs/2504.15900.

Xia, M., Malladi, S., Gururangan, S., Arora, S., and Chen, D. Less: Selecting influential data for targeted instruction tuning, 2024. URL https://arxiv.org/abs/ 2402.04333.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang,

- J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang,
- K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang,

- X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan,
- Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3 technical report, 2025a. URL https: //arxiv.org/abs/2505.09388.

Yang, Z., Guo, Z., Huang, Y., Liang, X., Wang, Y., and Tang, J. Treerpo: Tree relative policy optimization. arXiv preprint arXiv:2506.05183, 2025b.

Yang, Z., Guo, Z., Huang, Y., Wang, Y., Xie, D., Li, H., Wang, Y., Liang, X., and Tang, J. Depth-breadth synergy in rlvr: Unlocking llm reasoning gains with adaptive exploration. arXiv preprint arXiv:2508.13755, 2025c.

Ye, Y., Huang, Z., Xiao, Y., Chern, E., Xia, S., and Liu, P. Limo: Less is more for reasoning, 2025. URL https: //arxiv.org/abs/2502.03387.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., Liu, X., Lin, H., Lin, Z., Ma, B., Sheng, G., Tong, Y., Zhang, C., Zhang, M., Zhang, W., Zhu, H., Zhu, J., Chen, J., Chen, J., Wang, C., Yu, H., Song, Y., Wei, X., Zhou, H., Liu, J., Ma, W.Y., Zhang, Y.-Q., Yan, L., Qiao, M., Wu, Y., and Wang, M. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/ abs/2503.14476.

Yue, Y., Yuan, Y., Yu, Q., Zuo, X., Zhu, R., Xu, W., Chen, J., Wang, C., Fan, T., Du, Z., Wei, X., Yu, X., Liu, G., Liu, J., Liu, L., Lin, H., Lin, Z., Ma, B., Zhang, C., Zhang, M., Zhang, W., Zhu, H., Zhang, R., Liu, X., Wang, M., Wu, Y., and Yan, L. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks, 2025. URL https://arxiv.org/abs/2504.05118.

Zeng, W., Huang, Y., Liu, Q., Liu, W., He, K., Ma, Z., and He, J. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025. URL https://arxiv.org/abs/ 2503.18892.

Zhou, X., Fan, S., and Jaggi, M. Hyperinf: Unleashing the hyperpower of the schulz’s method for data influence estimation, 2025. URL https://arxiv.org/abs/ 2410.05090.

### A. Derivation of Expected Variance Reduction

In this section, we include the complete derivation of expected variance reduction in section 4. We know that ϕ ∼ Beta(α,β), and the variance of it is:

αβ (α + β)2(α + β + 1)

(18) For the posterior distribution of ϕ, we first apply the Bayes’ rule:

Var(ϕ) =

p(r|ϕ)p(ϕ)

p(r) ∝ p(r|ϕ)p(ϕ) (19) The likelihood for a single reward observation r ∈ {0,1} is:

p(ϕ|r) =

p(r|ϕ) = ϕr(1 − ϕ)1−r (20) Therefore, the posterior of ϕ also follows the Beta distribution:

ϕ|r ∼ Beta(α + r,β + 1 − r) (21) Now let n = α + β, plug these into Equation 7, and obtain:

Pr(r = r′)Varposterior(ϕτ|r = r′) (22)

∆V (τ) = Varprior(ϕτ) −

r′∈{0,1}

α n ·

(α + 1)β (n + 1)2(n + 2)

β n ·

α(β + 1) (n + 1)2(n + 2)

αβ n2(n + 1) − [

] (23)

+

=

αβ n2(n + 1) −

αβ n(n + 1)2

(24)

=

αβ n2(n + 1)2

(25)

=

### B. Derivation of Entropy of the Beta Distribution Let Φ ∼ Beta(α,β) be a continous random variable with density:

1 B(α,β)

ϕα−1(1 − ϕ)β−1 (26)

f(ϕ) =

where B(α,β) is the beta function. By definition of entropy, we have:

H(Φ) = −E[lnf(Φ)] (27) Then we can simplify it as:

H(Φ) = lnB(α,β) − (α − 1)E[lnΦ] − (β − 1)E[ln(1 − Φ)] (28) The expectation E[lnΦ] and E[ln(1 − Φ)] admit closed-form expressions in terms of the digamma function ψ(·):

E[lnΦ] = ψ(α) − ψ(α + β),E[ln(1 − Φ)] = ψ(β) − ψ(α + β) (29) Substituting these identities and simplifying gives the closed-form entropy:

H(Φ) = lnB(α,β) − (α − 1)(ψ(α) − ψ(α + β)) − (β − 1)(ψ(β) − ψ(α + β))

= lnB(α,β) − (α − 1)ψ(α) − (β − 1)ψ(β) + (α + β − 2)ψ(α + β)

(30)

### C. Proof of Asymptotic Scaling of Mutual Information

For a sufficiently large nτ = ατ + βτ, the Beta distribution would converge to a Gaussian distribution. The entropy of a Gaussian variable is given by:

- 1

- 2

ln(2πe · Var(Φτ)) (31) Then the Mutual Information can be approximated as the expected log-ratio of variance:

H(Φτ) ≈

I(R;Φτ) ≈

- 1

- 2

ER ln

Var(Φτ) Var(Φτ|R)

(32)

During RL training, when nτ is large, the posterior variance Var(Φτ|R) concentrates tightly around its expectation. Denoting the expected variance reduction by:

∆V (τ) = ER [Var(Φτ) − Var(Φτ|R)] (33)

For large nτ, the ratio Var∆V(Φ(τ)

τ) is small. Then apply the Taylor expansion ln(1/(1 − x)) ≈ x for x ≈ 0:

Var(Φτ) Var(Φτ|R) ≈

- 1

- 2

ER ln

I(R;Φτ) ≈

Var(Φτ) Var(Φτ) − ∆V (τ)

- 1

- 2

ln

 

 

 ln

  1

- 1

- 2

=

1 − Var∆V(Φ(ττ))

- 1

- 2

∆V (τ) Var(Φτ)

≈

(34)

Now substituting the exact expressions from Equation 18 and Equation 25 yields:

n2τ(nτ + 1) ατβτ

- 1

- 2 ·

ατβτ n2τ(nτ + 1)2 ·

I(R;Φτ) ≈

1 nτ

1 2(nτ + 1) ∼ O(

)

=

(35)

### D. Implementation Details

#### D.1. Datasets Details

CountDown. We adopt the Countdown Number Game, which requires combining given numbers using basic arithmetic operations to reach a target value. In detail, we randomly selected 2,000 samples from the CountDown-34 dataset1 for training, and evaluated performances on a 512-problem held-out test set. In CountDown-34, each problem provides either 3 or 4 source numbers, and the distribution is balanced. We set the reward function as the follows:

 

1, if response is correct, 0.1, if response is incorrect but with correct formatting, 0, otherwise.

(36)

r =



DeepScaler. We train LLMs on the training set of the DeepScaleR dataset2 (Luo et al., 2025b), which consists of approximately 40,000 unique mathematics problem-answer pairs. For evaluation, we conduct it on a suite of math benchmarks: AIME24, AMC23, MATH500 (Lightman et al., 2023), Minerva Math (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024), using the datasets provided by DeepScaler (Luo et al., 2025b). We report the average

- 1https://huggingface.co/datasets/BlackBeenie/simple-countdown
- 2https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset

pass@1 over 16 samples for each problem. Following the default setting in Sheng et al. (2025),we use a binary reward function that assigns a reward of 1 for a correct answer and 0 otherwise.

WebInstruct-verified. We train LLMs on the training set of WebInstruct-verified (Ma et al., 2025), which is a diverse and high-quality dataset to enhance robust reasoning capabilities across a broad range of domains, including physics, chemistry, and more. For evaluation, we apply the LM-Evaluation-Harness (Gao et al., 2024), a unified framework to test generative language models on different evaluation tasks. For RL training, we adopt the rule-based reward function as used in the above mathematics task.

#### D.2. Training Details

We adopt the widely used GRPO (Shao et al., 2024), implemented in the VeRL framework (Sheng et al., 2025), as our default algorithm. It is worth noting that our method INSIGHT can be integrated into any other RLVR algorithms.

We set training steps as 100 during RL training. At each training step, we set k = 8 responses per prompt for estimating advantages, using temperature 1.0 and top p = 1.0. Evaluation is based on pass@1, computed from 16 independent generations per prompt with temperature 0.6 and top p = 0.95, keeping the same setting as in Luo et al. (2025b). Training batch sizes are set to 256, while the mini-batch sizes are 128 and 64 for Mathematics&General-reasoning and Countdown, respectively. Optimization is performed by AdamW (Loshchilov & Hutter, 2019), with learning rate 1e − 6, weight decay 0.1, beta (0.9,0.999). For online data selection methods, (including MOPPS, INSIGHT, INVERSE-EVIDENCE, EXPECTEDDIFFICULTY), we set Mˆ is 16x of the selected batch size M, and λ = 1.0. For our INSIGHT, we set η = 3.0,µ = 3.0. All experiments are conducted on 8x NVIDIA H-series GPUs.

### E. Performance and Runtime Comparisons Between INSIGHT and Dynamic Sampling

In this section, we compare the performance and runtime (i.e., total training hours) between INSIGHT and Synamic Sampling (DS), where the training setting is the same as described above. INSIGHT precomputes the WMI scores for each datapoint, which is significantly faster than DS, since DS requires the LLM to over-sample the data and keep only effective datapoints after exact generation and evaluations.

Table 6. Evaluation results across mathematics benchmarks. Accuracy is computed as the average pass@1 over 16 independent generations per problem.

MODELS METHODS AIME24 AMC23 MATH500 MINERVA. OLYMPIAD. TRAINING HOURS

DS 50.62 80.79 90.82 38.39 53.38 ∼ 30.5 Hours R1-DISTILL-QWEN-7B RANDOM 46.25 77.90 90.50 37.98 51.80 ∼ 12.5 Hours INSIGHT (Ours) 47.71 78.40 90.45 39.06 52.50 ∼ 12.5 Hours

- Analysis. Table 6 shows the comparisons on R1-Distill-Qwen-7B, where we set the trainer.total training steps=100 in VeRL (Sheng et al., 2025) training for fair comparison. As expected, DS incurs substantially higher computational cost, requiring more than 2x the total training hours of INSIGHT. This overhead arises from DS’s reliance on oversampling, followed by full generation and reward evaluation to filter effective datapoints.

Despite this additional cost, DS achieves only marginal performance gains over INSIGHT and, in some benchmarks, performs comparably or even worse. In contrast, INSIGHT consistently improves over RANDOM while matching or exceeding DS on most metrics, without introducing any extra training-time overhead. These results highlight that information-aware selection based on Bayesian uncertainty can recover much of the benefit of oversampling-based methods, while being significantly more computationally efficient.

Overall, this comparison underscores the practical advantage of INSIGHT: it achieves strong performance gains comparable to expensive dynamic sampling strategies, yet preserves the runtime efficiency of standard RL training pipelines.

### F. Ablation Analysis of the Effect of Larger Candidate Batch Size Mˆ

In this section, we further study the effect of the larger candidate batch size Mˆ in Algorithm 1. We conduct experiments on Mˆ = 8x, 12x, and 16x (our original setting) to compare the performances.

Table 7. Evaluation comparisons of different Mˆ across different models, on mathematics benchmarks. Bold indicates the best results.

###### MODELS METHODS Mˆ AIME24 AMC23 MATH500 MINERVA. OLYMPIAD.

MOPPS 8x 10.20 44.42 71.46 21.64 34.07 INSIGHT 8x 10.20 44.50 72.53 21.69 35.07

QWEN3-0.6B INSIGHT 12x 10.83 43.30 70.85 22.20 34.33 INSIGHT 16x 11.90 44.30 71.75 21.94 34.80 MOPPS 8x 52.29 77.71 91.08 41.40 56.70 INSIGHT 8x 52.29 78.54 91.09 41.73 56.91

QWEN3-4B INSIGHT 12x 51.25 78.31 91.27 41.43 56.92 INSIGHT 16x 53.75 79.00 91.22 41.80 57.00 INSIGHT 8x 46.45 77.10 90.33 38.37 51.98

R1-DISTILL-7B INSIGHT 12x 46.67 78.01 90.48 38.46 52.51 INSIGHT 16x 47.71 78.40 90.45 39.06 52.50

- Analysis. Table 7 examines the effect of the larger candidate batch size Mˆ on mathematical reasoning benchmarks across different model scales. We observe a non-monotonic trend for smaller models and a more consistent preference for larger Mˆ as model size increases.

For QWEN3-0.6B, intermediate candidate sizes (8x or 12x) achieve the best performance on several benchmarks, occasionally outperforming 16x. This behavior suggests that, for smaller models, excessively large candidate pools may introduce additional variance or overly challenging samples that exceed the model’s effective learning capacity. In this regime, a moderately sized candidate set strikes a better balance between exploration and learnability, allowing INSIGHT to select informative yet tractable datapoints.

In contrast, as model capacity increases (e.g., QWEN3-4B and R1-DISTILL-7B), performance improves more consistently with larger Mˆ , and 16x becomes the strongest setting across most benchmarks. Larger models are better able to exploit increased candidate diversity and accurately estimate epistemic uncertainty, enabling more reliable ranking of highinformation datapoints from a broader pool.

Overall, these results indicate that the optimal Mˆ depends on model capacity: smaller models benefit from moderate candidate sizes that avoid over-exploration, while larger models can effectively leverage larger pools to improve selection quality. This adaptive behavior further highlights the robustness of INSIGHT across model scales and practical training settings.

