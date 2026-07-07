# arXiv:2604.13902v1[cs.LG]15Apr2026

## DiPO: Disentangled Perplexity Policy Optimization for Fine-grained Exploration-Exploitation Trade-Off

Xiaofan Li1,2, Ming Yang2†, Zhiyuan Ma2, Shichao Ma2, Jintao Du2, Yu Cheng2, Weiqiang Wang2, Zhizhong Zhang1, Xin Tan1, Yanyun Qu4, Lizhuang Ma1, Yuan Xie1,3‡ 1East China Normal University, 2Ant Group 3Shanghai Innovation Institute, 4Xiamen University lxfunzi@stu.ecnu.edu.cn, {lzma, yxie}@cs.ecnu.edu.cn

### Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has catalyzed significant advances in the reasoning capabilities of Large Language Models (LLMs). However, effectively managing the exploration and exploitation trade-off remains a critical challenge. In this paper, we fully analyze the exploration and exploitation dilemma of extremely hard and easy samples during the training and propose a new fine-grained trade-off mechanism. Concretely, we introduce a perplexity space disentangling strategy that divides the sample space into distinct exploration (high perplexity) and exploitation (low perplexity) subspaces, thereby mining fine-grained samples requiring exploration-exploitation trade-off. Subsequently, we propose a bidirectional reward allocation mechanism with a minimum impact on verification rewards to implement perplexity-guided exploration and exploitation, enabling more stable policy optimization. Finally, we have evaluated our method on two mainstream tasks: mathematical reasoning and function calling, and experimental results demonstrate the superiority of the proposed method, confirming its effectiveness in enhancing LLM performance by fine-grained exploration-exploitation trade-off.

### 1 Introduction

Exploration-Exploitation Trade-Off (EETO) Auer et al. [2002] represents a core challenge in RL for LLM post-training, where error samples stand to benefit from further exploration, while correct samples are suitable for exploitation. In this paper, we propose the two dilemmas of EETO in GRPO-based methods. First dilemma: the degradation of the advantage of extreme samples. As illustrated in Figure 1 (a), a large proportions of sample groups fall into either the hard group (with uniform zero rewards) or the easy group (with uniform one rewards) during training. These extreme groups yield a zero advantage, leading to the lack of training gradients and thus exposes the GRPO paradigm to an dilemma of ETTO, where the hard and easy groups lack signals for exploration and exploitation, respectively. Second dilemma: ineffective exploration and exploitation. As shown in Figure 1 (b), the distribution of perplexity (PPL) reflects the exploratory (high PPL) or exploitative (low PPL) tendency of samples, where some error samples exhibit an exploitative tendency, while some correct samples show an exploratory one. This ineffective exploration and exploitation severely decrease the stability and plasticity of RL.

Recent approaches have investigated the EETO by introducing PPL reward shaping. DACE Li et al. [2025] encourages all samples either to explore or to exploit based on intra-group accuracy. However, this coarse-grained partitioning would hinder policy optimization during early training stages when

†Equan Contribution ‡Corresponding Author

Preprint.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

- Figure 1: (a) The proportion of Easy/Normal/Hard groups in each step during the DAPO training. (b) The PPL distribution of correct and error samples in the validation set at 300th steps of DAPO training. (c) Illustration of four samples after PSD fine-grained partitioning.

most samples belong to the hard group. Meanwhile, CDE Dai et al. [2025] designs multiple weighting mechanisms to employ exploration rewards at appropriate times, yet it does not explicitly address exploitation and introduces more hyper-parameters. Moreover, both CDE and DACE directly use perplexity as a reward bias, which may introduce additional uncertainty. In this paper, we propose a novel method Disentangled Perplexity Policy Optimization (DiPO) to solve the two dilemmas and achieve a fine-grained EETO.

To enable fine-grained sample mining for exploratory and exploitative purposes, we propose the Perplexity Space Disentangling (PSD) strategy, that integrates statistical probability of correctness (verification reward) and PPL. In Figure 1 (b), error samples typically exhibit higher PPL, whereas correct samples generally maintain lower PPL. PSD infers the optimal threshold τ∗ by advantage judgment and minimizing classification errors, thus capturing the inherent correlation between PPL and sample correctness. Subsequently, as shown in Figure 1 (c), the entire sample space is partitioned into four fine-grained quadrants: correct samples with high PPL (CH), correct samples with low PPL (CL), error samples with high PPL (EH), and error samples with low PPL (EL). While the CL and EH samples naturally align with effective exploitation and exploration; The critical of EETO lies in encouraging exploitation for CH samples and exploration for EL samples.

In addition, the uncertainty of PPL distribution results in a significant discrepancy between the PPL reward and the verification reward. To achieve stable RL training for EETO, we abandon the direct adoption of PPL for reward shaping and proposes a Bidirectional Reward Reallocation (BRR) mechanism. Specifically, on the one hand, to avoid interfering with the policy optimization guided by the verification reward, BRR is only performed on the zero-gradient easy and hard groups. On the other hand, because the verification reward variance of the easy and hard groups is zero, we introduce a maximum-PPL reward reallocation strategy, where the rewards corresponding to the maximum PPL samples in the easy and hard groups are set to 0 and 1, respectively, ensuring the variance of the reallocated reward distribution is close to zero. The proposed BRR incorporates the PPL signal while minimizing the perturbation to the original reward distribution, thus realizing more stable training.

To summarize, the main contributions of this paper are:

- • We propose a perplexity space disentanglement strategy that partitions the PPL space based on perplexity and correctness distributions, enabling a fine-grained EETO.
- • We design a bidirectional reward reallocation mechanism that stabilizes training by reallocating rewards with minimal perturbation to the verification reward distribution.”
- • For mathematical reasoning and function calling tasks, our method achieves superior results confirming its effectiveness in reasoning enhancement for LLMs.

### 2 Preliminaries and Definitions

#### 2.1 RLVR Algorithms

Group Relative Policy Optimization (GRPO) introduces a concise advantage calculation method. For each query q and its ground-truth answer a, given the verifiable reward calculation function:

R(oi,a) = 1 if is_equivalent(a,oi) else 0, (1)

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- Figure 2: Illustration of DiPO, consisting of three modules: PPL Queue, Perplexity Space Disentangling (PSD), and Bidirectional Reward Reallocation (BRR). Specifically, the PPL Queue caches PPL items; PSD is used for fine-grained sample partition; and then BRR performs reward allocation.

where {oi}Gi=1 is generated by rollout policy πθ

(o|q) through G-sampling. The estimated advantage Aˆ it under R is then computed as:

old

Aˆ it(R) = R(oi,a) − mean({R(oi,a)}Gi=1)

. (2)

std({R(oi,a)}Ni=1)

When G-sampling rewards are all 0 or 1, all advantages come to 0, indicating the sampled group consists solely of hard groups and easy groups.

Dynamic Sampling Policy Optimization (DAPO) introduces new improvements based on GRPO The maximum optimization objective under R is as follows:

|oi|

G

1

min sitAˆ it(R), clip sit,1 − ϵlow,1 + ϵhigh A ˆ it(R) , s.t. 0 < |{R(oi,a)|R(oi,a) = 1)}| < G

JDAPO(θ,R) = E(q,a)∼D,{oi}Gi=1∼πθold(o|q)

G i=1 |oi|

t=1

i=1

(3) where sit = πθ(o

i t|q,oi<t)

πθold(oit|q,oi<t) is the ratio of importance sampling. In this paper, we use advanced DAPO as our baseline without overlong reward shaping, as in some cases, overlong reward shaping would damage the model’s performance.

- 2.2 Perplexity

Perplexity (shorted as PPL) in Large Language Models is a metric that quantifies how confidently a probability model predicts a sample. Mathematically, given a model πθ, a question q, and a response oi = (oi1,oi2,...,oiT) generated by πθ, the perplexity is given by:

pi = exp −

1 T

T

t=1

log πθ(oit|q,o<t) (4)

For a response, a low PPL indicates that the trajectory tends to exploitation, while higher PPL indicates that the trajectory tends to exploration Dai et al. [2025], Sun et al. [2025].

- 3 Methodology

- 3.1 Overview

As illustrated in Figure 2, the proposed DiPO mainly contains two modules: 1) Perplexity Space Disentangling (PSD); 2) Bidirectional reward reallocation (BRR). PSD calculates conditional probability distribution based on the PPL Queue cached with PPL-reward pairs, and then determines the optimal threshold by the advantage judgment and minimizing classification errors. We term higher and lower PPL region exploration space (ErS) and exploitation space (EiS), respectively. Through

PPL disentanglement, the ineffective samples for exploration and exploitation, i.e., hard groups in EiS and Easy groups in ErS are selected for policy optimization. Subsequently, BRR is implemented for these samples; for the hard group in EiS, the reward for maximum PPL sample is set to 1, thus encouraging exploration; the easy group in ErS, the reward for maximum PPL sample is set to zero, thus penalizing exploration. BRR incorporates the PPL signal and minimizes perturbations to the verification reward, enabling more stable training.

#### 3.2 Perplexity Space Disentangling

PSD calculates the optimal threshold τ∗ for dividing EiS and ErS by introducing the conditional probability distribution Pr(R|P) where P and R represent PPL and reward. For ETTO, error samples stand to benefit from further exploration, while correct samples are suitable for exploitation. Thus the optimal EiS and ErS should have the highest relevance to sample correctness, where samples falls in EiS are considered more likely to be correct, whereas those in ErS are generally deemed error.

Online Statistical Estimation. Since the PPL distribution induced by the policy changes dynamically during training, direct estimation of conditional probabilities may suffer from an unfavorable variancebias trade-off. To stabilize the estimation, we maintain a PPL queue Q that caches samples from the most recent two batches: Q = {(pi,ri)}Mi=1, where pi ∈ [1,+∞) denotes the perplexity (PPL) of the i-th sample and ri ∈ {0,1} denotes its corresponding validation reward.

For a given threshold τ, we estimate the empirical conditional probabilities of the reward given whether the PPL is below or above τ as

M i=1 I(pi < τ, ri = 1)

M i=1 I(pi < τ, ri = 0)

Pr(R = 1 | P < τ) =

, Pr(R = 0 | P < τ) =

,

M i=1 I(pi < τ)

M i=1 I(pi < τ)

M i=1 I(pi > τ, ri = 1)

M i=1 I(pi > τ, ri = 0)

Pr(R = 1 | P > τ) =

, Pr(R = 0 | P > τ) =

.

M i=1 I(pi > τ)

M i=1 I(pi > τ)

(5)

where I(·) denotes the indicator function.

To quantify the statistical uncertainty of each estimate, we compute 95% confidence intervals using the Normal (Wald) approximation Wald [2004]. For a generic empirical probability estimate pˆ = m/n, where m denotes the number of successes among n samples, the standard error is SE(ˆp) = pˆ(1 − pˆ)/n, and the corresponding 95% confidence interval is

[pˆ− 1.96 · SE(ˆp), pˆ+ 1.96 · SE(ˆp)]. (6)

These estimated probabilities represent the sample likelihood of being correct or error in the Exploitation Space (EiS, P < τ) and Exploration Space (ErS, P > τ) demarcated by the threshold τ.

Advantage Judgment. During the early RL stage of pre-trained models, the correlation between PPL and correctness may not be positively correlated (see Appendix E.5), where directly activating PSD would hinder optimization. To this end, we introduce a advantage judgment mechanism based on the results of online statistical estimation.

Specifically, for a given threshold τ, we define two advantage functions that characterize the separation between correct samples and error samples in divided PPL regions:

∆EiS(τ) = Pr(R = 1 | P < τ) − Pr(R = 0 | P < τ), ∆ErS(τ) = Pr(R = 0 | P > τ) − Pr(R = 1 | P > τ),

(7)

where ∆EiS(τ) measures the correctness advantage in the EiS, and ∆ErS(τ) measures the error advantage in the ErS.

To reduce the impact of sampling randomness and ensure robust judgment, we conservatively evaluate these gaps using the boundary values of the 95% confidence intervals derived in the online estimation stage (Equation (6)). Concretely, the two judgment functions are as follows:

∆EiS(τ) = L1 − U2 and ∆ErS(τ) = L4 − U3, (8)

where [L1,U1] and [L2,U2] denote the 95% confidence intervals of Pr(R = 1 | P < τ) and Pr(R = 0 | P < τ), respectively, [L3,U3], [L4,U4] correspond to Pr(R = 1 | P > τ) and

Pr(R = 0 | P > τ). Considering the PPL space to be meaningfully correlated with correctness at threshold τ only when both conditions ∆EiS(τ) > 0 and ∆ErS(τ) > 0 hold.

Minimizing Classification Errors. In the middle and later RL stages, a wide range of thresholds may satisfy Advantage Judgment, necessitating the further selection of an optimal value. We formalize a classification task using PPL as the criterion: responses with a PPL below τ are classified as correct, while those above are deemed error. By minimizing classification errors (Err(τ) in Figure 2) to find the optimal threshold:

##### 1 |Q|

|ri − I(pi < τ)|

τ∗ = arg min

τ

(9)

(ri,pi)∈Q

s.t. ∆EiS(τ) > 0, ∆ErS(τ) > 0

where I(·) is the indicator function. Through advantage judgment and minimizing classification errors, the disentangled PPL spaces are highly correlated with correctness. Consequently, the hard groups in EiS and the easy groups in ErS would be the important samples for achieving the EETO. Algorithm details are illustrated in Algorithm 1.

#### 3.3 Bidirectional Reward Reallocation

To implement a stable exploration-exploitation optimization and minimize the impact to verification rewards, BRR does not directly employ PPL as the basis for reward shaping, but introduces the maximum-PPL reward reallocation strategy. BRR thus aims to drive updates toward the exploratory (high-entropy) direction for the hard groups in EiS, and toward the exploitative (low-entropy) direction for the easy groups in ErS..

- Theorem 1 (Entropy Increase with Maximum-PPL Reward). Let πt be the policy at step t. Given a query q and a group of outputs {oi} ∼ πt(· | q), if the output with the maximum PPL among {oi} is assigned a reward, then the average entropy of the updated policy πt+1(· | q) increases.

Reward Reallocation for Hard Groups. As shown in Figure 2, we define a hard group as qˆ, {oˆi}, {rˆi}, {pˆi}, here {rˆi} = {0}. If mean({pˆi}) < τ∗ means that this hard group is located in EiS, the goal of the reward reallocation is to update it towards the high entropy direction. Concretely, the index mˆ corresponding to the maximum PPL in {oˆi} is identified: mˆ = arg max1≤i≤G pˆi. Subsequently, the reallocated reward {rˆir} is constructed by setting rˆmr = 1 while preserving all other original values, such that rˆir = rˆi for all i ̸= mˆ . Noted that if τ∗ does not exist, or mean({pˆi}) > τ∗, reallocated reward remains unchanged, i.e., {rˆir} = {rˆi} = {0}.

- Theorem 2 (Entropy decrease with Maximum-PPL penalty). Let πt be the policy at step t. Given a query q and a group of outputs {oi} ∼ πt(· | q), if the output with the maximum PPL among {oi} is assigned a penalty, then the average entropy of the updated policy πt+1(· | q) decreases.

Reward Reallocation for Easy Groups. As shown in Figure 2, reward reallocation for the easy group is the opposite of that for the hard group. A easy group is defined as q¯, {o¯i}, {r¯i}, {p¯i}, here {r¯i} = {1}. If mean({p¯i}) > τ∗ means that this easy group is located in ErS, the goal of the reward reallocation is to update it towards the low entropy direction. Concretely, the index m¯ corresponding to the maximum PPL in {o¯i} is identified: m¯ = arg max1≤i≤G p¯i. Subsequently, the reallocated reward {r¯ir} is constructed by setting r¯mr = 0 while preserving all other original values, such that r¯ir = r¯i for all i ̸= m¯ . Noted that if τ∗ does not exist, or mean({p¯i}) < τ∗, reallocated reward remains unchanged, i.e., {r¯ir} = {r¯i} = {1}.

Finally, we abstracted the peocess of BRR into a formalized reward function Rr, defined as follows:

 

{rˆir} if {ri} = {0}, {r¯ir} if {ri} = {1}, {0} other,

Rr({ri}) =

(10)



where BRR only performs reward reallocation for the easy groups and hard groups, and sets all rewards of the normal groups to zero, thus ensuring that the reallocated rewards (denoted as Rr) and verification rewards (denoted as R) are orthogonal. Algorithm details are illustrated in Algorithm 2.

Policy Optimization. R and Rr are orthogonal, which means that the samples with gradients under the two rewards are orthogonal. Thus, we use a hyper-parameter α to control the loss weight of Rr.

- Table 1: Comparison of mathematical reasoning in ACC/mean@8 on 6 mathematics benchmarks. The best and second-best results are respectively marked in bold and underlined.

Method AIME24 AIME25 MATH AMC OLY MIN AVG Qwen3-4B-Base

Base model 9.58 3.75 48.48 31.48 25.52 24.82 23.94 GRPO 26.67 23.33 85.83 60.24 53.06 44.39 48.92 DAPO 26.25 23.75 86.43 61.90 53.88 44.34 49.43 DAPO w/ EL 26.67 24.58 86.78 62.95 54.53 44.53 50.01 CDE 26.67 24.17 85.93 62.35 52.25 43.11 49.08

DiPO (ours) 29.17 24.58 87.00 63.70 54.09 44.76 50.55 Qwen3-8B-Base

Base model 8.33 9.17 66.78 39.46 33.30 66.78 37.30 GRPO 31.67 24.58 89.08 69.28 56.20 48.62 53.24 DAPO 30.08 25.83 89.43 69.12 56.90 48.02 53.23 DAPO w/ EL 33.75 25.42 89.58 69.87 57.21 47.56 53.90 CDE 31.67 26.25 89.35 68.07 57.11 47.75 53.37

DiPO (ours) 35.00 27.50 89.55 71.23 57.73 47.75 54.79 Qwen2.5-7B

Base model 7.08 2.08 41.53 22.74 19.28 17.19 18.32 GRPO 20.42 15.42 79.15 58.43 42.42 36.95 42.13 DAPO 20.42 16.67 79.08 59.94 42.70 37.55 42.73 DAPO w/ EL 20.00 14.58 79.85 58.73 43.05 39.65 42.64 CDE 20.00 15.00 79.00 55.87 42.94 35.94 41.46

DiPO (ours) 22.92 16.67 80.35 60.09 43.72 37.59 43.56

The maximum optimization objective is as follows:

JDiPO(θ) = JDAPO(θ,R) + α × JDAPO(θ,Rr), (11) where JDAPO is the DAPO optimization objective in Eq. (3).

### 4 Experiment

#### 4.1 Experiment Setup

To comprehensively evaluate the effectiveness of our proposed DiPO, we conduct experiments on two downstream tasks of LLMs: mathematical reasoning and function calling. Detailed configurations are provided below.

Mathematical Reasoning. We evaluate DiPO on the mathematical reasoning task using the DAPO17K Yu et al. [2025] dataset for training. Model performance is assessed on five challenging benchmarks: AIME24, AIME25, AMC23 (denoted as AMC), MATH500 Hendrycks et al. [2021] (denoted as MATH) the OE_TO_mat_en_COMP subset from OlympiadBench He et al. [2024](denoted as OLY), and Minerva (denoted as MIN). These evaluations are designed to comprehensively examine the method’s effectiveness across diverse problem difficulties and formats.

We perform a comparative analysis against several strong baselines: 1. GRPO Shao et al. [2024], 2. DAPO Yu et al. [2025], 3. DAPO enhanced with Entropy Loss (DAPO w/ EL) Williams [1992], and 4. CDE Dai et al. [2025] with PPL reward shaping. All mathematical reasoning experiments are implemented using the VERL Sheng et al. [2025] framework. To ensure a rigorous comparison, we employ three different pretrained base models: Qwen3-4B-Base, Qwen3-8B-Base Yang et al. [2025a], and Qwen2.5-7B Yang et al. [2025b]. All models utilize consistent training configurations and prompts, as detailed in Appendix D.1.

Function Calling. We establish our experimental setup using the publicly available ToolRL Qian et al. [2025] framework as the primary baseline. Furthermore, we implement ToolRL augmented with DAPO (ToolRL+DAPO) for comparison. Since ToolRL does not utilize a binary verification reward, certain modifications were made when implementing TooRL+DiPO.

- Table 2: Comparison of function calling in acc on BFCLv3 benchmark. The best and second-best results are respectively marked in bold and underlined.

Multi Turn Acc

Relevance Detection

Irrelevance Detection

Method Non-Live Acc Live Acc

Overall Qwen2.5-3B-Instruct

Base model 42.52 53.96 1.00 44.44 82.49 33.04 SFT400 69.29 41.40 0.00 94.44 60.14 34.08 SFT400+PPO 78.29 58.76 5.12 100.00 48.40 45.80 SFT400+GRPO 76.21 64.15 1.75 94.44 58.63 46.42 PPO, Cold Start 82.42 67.78 4.88 100.00 18.09 51.15 ToolRL+GRPO 81.58 73.78 3.75 100.00 56.44 52.98 ToolRL+DAPO 82.19 69.43 8.00 81.25 57.60 53.21

ToolRL+DiPO 83.42 73.06 8.62 100.00 54.16 55.03 Qwen2.5-7B-Instruct

Base model 66.02 53.51 4.25 76.47 62.66 41.97 SFT400 69.29 41.40 0.00 94.44 8.11 34.08 SFT400+PPO 83.90 51.84 0.25 100.00 29.66 42.02 SFT400+GRPO 80.69 46.51 0.25 100.00 14.19 39.25 PPO, Cold Start 79.33 63.17 0.38 88.89 52.92 46.68 ToolRL+GRPO 86.17 74.90 18.12 83.33 76.68 58.38 ToolRL+DAPO 87.10 76.31 19.75 87.50 67.25 61.06

ToolRL+DiPO 86.21 76.83 24.50 87.50 69.57 62.51

We conduct comprehensive testing on the BFCLv3 benchmark Patil et al. [2025], which provides a diverse set of function calling scenarios for thorough evaluation. Consistent with the TooRL setup, we use the same model architectures (Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct) and maintain identical training configurations and datasets across all compared methods, details in see Appendix D.2.

#### 4.2 Comparison Results of Mathematical Reasoning

Table 1 shows the comprehensive comparison results of mathematical reasoning in ACC/mean@8 across 6 benchmarks evaluated on three base models, our proposed DiPO demonstrates consistent and superior performance enhancements over other reinforcement learning algorithms. Overall, DiPO achieves the highest average score (AVG) across all three model scales, with 50.55% for Qwen3-4BBase, 54.79% for Qwen3-8B-Base, and 43.56% for Qwen2.5-7B, surpassing all compared baselines. Specifically, on the more challenging AIME benchmarks, DiPO attains the best results in most cases, such as 29.17% on AIME24 and 24.58% on AIME25 with Qwen3-4B-Base, and notably 35.00% on AIME24 and 27.50% on AIME25 with Qwen3-8B-Base, indicating its strong capability in handling complex mathematical reasoning tasks. Furthermore, the performance gains are particularly pronounced in larger models like Qwen3-8B-Base, where DiPO outperforms the second-best method by a clear margin in AVG (54.79% vs. 53.90%), highlighting its scalability and effectiveness in leveraging model capacity. Notably, although DAPO with EL achieves highly competitive results, it demonstrates pronounced sensitivity to its configuration coefficients, as further examined in the Appendix E.3.

#### 4.3 Comparison Results of Function Calling

Building upon the mathematical reasoning evaluation, we further assess the performance of DiPO on the function calling task. As presented in the Table 2, we use TooRL as the baseline and replicated TooRL+DAPO for comparison. The results demonstrate that DiPO delivers superior overall performance, achieving the highest Overall acc of 55.03% and 62.51% on the Qwen2.5-3BInstruct and Qwen2.5-7B-Instruct models, respectively. Notably, DiPO demonstrates exceptional capability in handling complex, multi-round interactions, as evidenced by its leading performance in Multi-Turn Acc. It achieves scores of 8.62% and 24.50% for the 3B and 7B models respectively, surpassing the second-best method (ToolRL+DAPO, with 8.00% and 19.75%) by 0.62 and 4.75

- Table 3: The impact of hyperparameter α on performance. The best results are marked in bold. α AIME24 AIME25 MATH AMC OLY MIN AVG

Qwen3-4B-Base

- α = 0.0 26.25 23.75 86.43 61.90 53.88 44.34 49.43

- α = 0.1 29.17 24.58 86.93 63.70 54.09 44.76 50.55 α = 1.0 25.83 24.17 86.38 62.05 54.01 45.50 49.66

Qwen3-8B-Base

- α = 0.0 30.08 25.83 89.43 69.12 56.90 48.02 53.23

- α = 0.1 35.00 27.50 89.55 71.23 57.73 47.79 54.79 α = 1.0 32.08 24.58 88.68 70.03 56.71 48.07 53.36

percentage points. This comprehensive improvement over strong function calling baselines, validates the effectiveness and general applicability of the DiPO not only in mathematical reasoning.

4.4 Hyperparameter Analysis

The orthogonal design between the reallocated rewards and validation rewards enables precise, independent adjustment of DiPO’s influence via the hyperparameter α. Table 3 presents the performance variations of Qwen3-4B-Base and Qwen3-8B-Base models under different values of α. For both model scales, the performance tends to reach the optimal level when α is set to 0.1: 4B model achieves the highest average (AVG) performance of 50.55%, with significant improvements compared to α = 0.0 and α = 1.0; similarly, 8B-model attains the best AVG of 54.79% at α = 0.1. When α = 0.0, DiPO’s contribution is zero, leading to the lowest AVG performance for both models (49.43% and 53.23%).

4.5 Ablation Experiment and Discussion

- Table 4 presents ablation results of the contributions of PSD and BRR on Qwen3-4B-Base and Qwen38B-Base on six mathematical reasoning benchmarks. PPL reward refers to using PPL directly as rewards: positive PPL for hard groups, and negative PPL for easy groups. Overall, the combination of PSD and BRR achieves the best performance for both 4B and 8B models, verifying their effectiveness. Detailed discussions are as follows.

- Discussion 1. Fine-grained exploration and exploitation make RL more effective. In Table 4, without PSD, all hard samples are indiscriminately driven toward high PPL, whereas easy samples are steered toward low PPL. PSD improves this by disentangling the space into ErS (high PPL) and EiS (low PPL), allowing for fine-grained optimization: easy groups within ErS are encouraged toward EiS, while hard groups within EiS are directed toward ErS. The results confirm the critical role of PSD in model enhancement. When using the PPL reward, PSD yields improvements of 2.39 and 1.19 points on the 4B and 8B models, respectively; with the BRR reward, the corresponding gains are even larger at 2.99 and 3.88 points. Significant improvements indicate that invalid exploration and exploitation are needed for optimization, while excessive encouragement of exploration and exploitation is detrimental.
- Discussion 2. Reward shaping should not cause significant changes to the intrinsic validation rewards. In Table 4, we compare our BRR against PPL reward. The experimental results confirm that BRR is superior to the PPL reward method. Concretely, with using PSD, BRR compared to PPL reward brings an improvement of 0.93 and 2.23 respectively on the 4B and 8B models, while using PSD and PPL reward is even worse than the baseline. Science the reward variance of easy and hard groups are both zeros, directly using PPL as a reward causes drastic shifts in the reward distribution, potentially destabilizing the training process. In contrast, BRR introduces the reallocated rewards maintain a distribution very similar to the original validation rewards, which allows BRR to leverage PPL signals for enhancement with the minimum impact on the validation reward landscape.

#### 4.6 Quantitative Results

To further verify the properties of DiPO, we conducted a quantitative analysis of some data during the training process of DAPO and DiPO.

Table 4: Ablation study results on Qwen3-4B-Base and Qwen-8B-Base, showing ACC/mean@8 on six mathematical reasoning benchmarks. The best results are marked in bold.

PSD BRR PPL reward AIME24 AIME25 MATH AMC OLY MIN AVG

Qwen3-4B-Base

- ✗ ✗ ✗ 26.25 23.75 86.43 61.90 53.88 44.34 49.43

- ✗ ✓ ✗ 25.83 20.83 85.10 59.19 49.37 43.06 47.23

- ✗ ✗ ✓ 24.58 23.75 84.60 61.45 49.09 41.82 47.55

✓ ✗ ✓ 27.50 23.33 86.35 62.35 54.00 44.21 49.62 ✓ ✓ ✗ 29.17 24.58 86.93 63.70 54.09 44.76 50.55

Qwen3-8B-Base

- ✗ ✗ ✗ 30.08 25.83 89.43 69.12 56.90 48.02 53.23

- ✗ ✓ ✗ 30.08 24.58 88.35 65.36 53.62 46.23 51.37

- ✗ ✗ ✓ 27.92 25.42 88.00 64.76 53.45 45.63 50.86

✓ ✗ ✓ 30.83 24.58 88.15 68.83 56.65 46.30 52.56 ✓ ✓ ✗ 35.00 27.50 89.55 71.23 57.73 47.42 54.79

Exploration-exploitation trade-off. Figure 3 shows the PPL distribution of the Qwen3-8B-Base after training on DAPO-17K with DAPO and DiPO. The results demonstrate that under the DAPO training, the PPL distribution of correct and incorrect samples exhibits a significant overlap. This phenomenon directly causes a large number of hard samples to lose exploration capability, thereby limiting the plasticity of RL. In contrast, DiPO shows superior distribution characteristics: error samples are more located in the high-PPL region, while correct samples are still concentrated in the low-PPL region, which achieves a balanced trade-off between exploration and exploitation in RL.

###### DiPO

###### DAPO

Correct

Correct Error

60

Error

40

| |
|---|

50

Frequency

Frequency

30

40

30

20

20

10

10

0

0

1.00 1.02 1.04 1.06 1.08 1.10

1.00 1.02 1.04 1.06 1.08 1.10

PPL

PPL

###### AIME25

###### AIME24

30

Max: 27.50 Max: 25.83

Max: 35.00 Max: 30.83

35

| |
|---|

25

ACC/mean@8(%)

ACC/mean@8(%)

| |
|---|

| |
|---|

30

| |
|---|

| |
|---|

| |
|---|

25

20

| |
|---|

DiPO (Raw)

DiPO (Raw)

20

15

DiPO (Smoothed)

DiPO (Smoothed)

15

DAPO (Raw)

DAPO (Raw)

DAPO (Smoothed)

DAPO (Smoothed)

10

10

0 200 400 600

0 200 400 600

Step

Step

Figure 3: PPL distribution comparison of correct and error samples for Qwen3-8B-Base trained on DAPO-17K Dataset via DAPO and DiPO.

Figure 4: ACC/mean@8 curves of DiPO and DAPO (raw and smoothed curves) on AIME24 and AIME25 with using Qwen3-8B-Base model.

Higher upper bound for later training. Figure 4 presents the test curves of DiPO and DAPO on two AIME benchmarks during the training process. The results show that at the initial stage of training, there was no significant difference in the performance of the two methods; in the later stages of training, however, the performance curve of DAPO exhibited a significant slowdown in growth, whereas DiPO maintained a sustained capacity for exploratory improvement. Combined with the PPL distribution characteristics in Figure 4, this phenomenon can be further explained: DiPO enables hard samples to conduct more sufficient exploration, while the training paradigm of DAPO leads to a conservative tendency in the exploration strategy of hard samples.

### 5 Conclusion

This paper explores the exploration-exploitation trade-off between in RL training. We analyze the two ETTO dilemmas faced by the extreme hard and easy groups, and proposed the Disentangled Perplexity Policy Optimization. First, we develop a novel Perplexity Space Disentangling method to identify which hard samples require encouragement for exploration and which easy samples need promotion for exploitation. Then, we design a Bidirectional Reward Reallocation mechanism that incorporates PPL-based exploration and exploitation signals while minimizing disruptions to the original verification reward distribution. Finally, extensive experiments on mathematical reasoning and function calling tasks demonstrate the comprehensive superiority of the proposed method.

### References

Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256, 2002.

Liang Chen, Xueting Han, Qizhou Wang, Bo Han, Jing Bai, Hinrich Schutze, and Kam-Fai Wong. EEPO: exploration-enhanced policy optimization via sample-then-forget. arXiv preprint arXiv:2510.05837, 2025a.

Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, and Guang Shi. Pass@ k training for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751, 2025b.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021.

Runpeng Dai, Linfeng Song, Haolin Liu, Zhenwen Liang, Dian Yu, Haitao Mi, Zhaopeng Tu, Rui Liu, Tong Zheng, Hongtu Zhu, et al. Cde: Curiosity-driven exploration for efficient reinforcement learning in large language models. arXiv preprint arXiv:2509.09675, 2025.

Jia Deng, Jie Chen, Zhipeng Chen, Daixuan Cheng, Fei Bai, Beichen Zhang, Yinqian Min, Yanzipeng Gao, Wayne Xin Zhao, and Ji-Rong Wen. From trial-and-error to improvement: A systematic analysis of llm exploration mechanisms in rlvr. arXiv preprint arXiv:2508.07534, 2025.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, 2024.

Haoyang He, Zihua Rong, Kun Ji, Chenyang Li, Qing Huang, Chong Xia, Lan Yang, and Honggang Zhang. Rethinking reasoning quality in large language models through enhanced chain-of-thought via rl, 2025. URL https://arxiv. org/abs/2509.06024, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Ang Li, Zhihang Yuan, Yang Zhang, Shouda Liu, and Yisen Wang. Know when to explore: Difficultyaware certainty as a guide for llm reinforcement learning. arXiv preprint arXiv:2509.00125, 2025.

Jia Liu, ChangYi He, YingQiao Lin, MingMin Yang, FeiYang Shen, and ShaoGuo Liu. Ettrl: Balancing exploration and exploitation in llm test-time reinforcement learning via entropy mechanism. arXiv preprint arXiv:2508.11356, 2025.

Shishir G. Patil, Huanzhi Mao, Charlie Cheng-Jie Ji, Fanjia Yan, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025.

Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, and Heng Ji. Toolrl: Reward is all tool learning needs. arXiv preprint arXiv:2504.13958, 2025.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.

Yuda Song, Julia Kempe, and Remi Munos. Outcome-based exploration for llm reasoning. arXiv preprint arXiv:2509.06941, 2025.

Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and Dong Yu. Crossing the reward bridge: Expanding rl with verifiable rewards across diverse domains. arXiv preprint arXiv:2503.23829, 2025.

Yan Sun, Jia Guo, Stanley Kok, Zihao Wang, Zujie Wen, and Zhiqiang Zhang. Efficient reinforcement learning for large language models with intrinsic exploration. arXiv preprint arXiv:2511.00794, 2025.

Abraham Wald. Sequential analysis. Courier Corporation, 2004. Christian Walder and Deep Karkhanis. Pass@ k policy optimization: Solving harder reinforcement

learning problems. arXiv preprint arXiv:2505.15201, 2025. Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

Rihui Xin, Han Liu, Zecheng Wang, Yupeng Zhang, Dianbo Sui, Xiaolin Hu, and Bingning Wang. Surrogate signals from format and length: Reinforcement learning for solving mathematical problems without ground truth answers. arXiv preprint arXiv:2505.19439, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025b. URL https://arxiv.org/abs/2412.15115.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Yang Zhou, Sunzhu Li, Shunyu Liu, Wenkai Fang, Kongcheng Zhang, Jiale Zhao, Jingwen Yang, Yihe Zhou, Jianwei Lv, Tongya Zheng, et al. Breaking the exploration bottleneck: Rubric-scaffolded reinforcement learning for general llm reasoning. arXiv preprint arXiv:2508.16949, 2025.

### A Proof of Section 3.3

#### A.1 Mathematical proof

Proof. Given a language model πθ with parameters θ. Given a query q, sample G responses {oi}Gi=1. Each response oi is a token sequence (oi1,...,oiTi). The model’s probability for token oit given context (q,oi<t) = πθ(y | q,oi<t) is denoted as πti(oit). The probability of other tokens in position of the t-th token is denoted as πti(y), and the logistic of y is denoted as lit(y), y ∈ V and V is the vocabulary of πθ. For convenience, we do not consider the clip operation, the loss function is as follows:

|oi|

G

exp(lti(oti)) y∈V exp(lti(y))

πti(oit) πold(oit)

1 G

1 |oi|

. (12)

Aˆit

, where πti(oit) =

L(θ) = −

t=1

i=1

Optimization gradient descent is θ′ = θ − η∇θL, with small learning rate η. Define the token-level entropy for response i at token t:

Hti = −

πθ(y | q,oi<t)log πθ(y | q,oi<t) = −

y∈V

y∈V

The average token entropy over all responses and time steps is:

πti(y)log πti(y). (13)

|oi|

G

1 G

1 |oi|

Hti. (14)

Havg =

t=1

i=1

The following will prove that high PPL rewards and high PPL penalties can respectively increase and decrease Havg.

Logit Change. Consider a single token of response oit for training, the loss contribution is −Aˆit π

i t(oit)

πold(oit). The gradient of the loss with respect to logit lti(y) is: ∂(−Aˆit π

i t(y)

πold(y)) ∂lti(y)

πti(y) πold(y)

= Aˆit

) ≈ Aˆit(πi(y) − 1y=oi

(πi(y) − 1y=oi

). (15) Using gradient descent with learning rate η, the logit update is:

t

t

∆lti(y) = ηAˆit(1y=oi

− πti(y)). (16)

t

Probability Change. According the multivariate Taylor Expansion, for softmax, the first-order change in probability is:

∂πti(y) ∂lti(z)

∆πti(y) ≈

∆lti(z) =

πti(y)(1z=y − πti(z))∆lti(z) = πti(y) ∆lti(y) −

πti(z)∆lti(z)

z∈V

z∈V

z∈V

(17) where πti(y) = exp(l

i t(y))

z∈V exp(lti(z)). Substituting ∆πti(y).

∆πti(y) = πti(y) ηAˆit(1y=oi − πti(y)) −

z∈V

πti(z)ηAˆit(1z=oi

− πti(z))

t

= ηAˆitπti(y) 1y=oi

− πti(y) − πti(oit) +

t

z∈V

πti(z)2 .

(18)

Entropy Change. According the Taylor Expansion, the first-order change in entropy Hti is:

 .

 −

∂Hit ∂πti(y)

πti(y)log πti(y)

∆πti(y)(1 + log πti(y)), where Hit =

∆Hti ≈

∆πti(y) = −

y

y∈V

y∈V

(19)

Substituting ∆πti(y):

∆Hti = −ηAˆit

y∈V

πti(y) 1y=oi

− πti(y) − πti(oit) +

t

z∈V

πti(z)2 (1 + log πti(y))

= − ηAˆit πti(oit)log πti(oit) −

y∈V

πti(y)2 log πti(y) − πti(oit)

y∈V

πti(y)log πti(y) +

z∈V

πti(z)2

πti(y)log πti(y)

y∈V

= − ηAˆit πti(oit)log πti(oit) −

πti(y)2Hti −

πti(y)2 log πti(y) + πti(oit)Hti

y∈V

y∈V

(20)

Average Entropy Change. Ignore the impact of cross-context updates, the first-order change in token entropy over all queries is:

|oi|

G

1 |oi|

1 G

−ηAˆit πti(oit)log πti(oit) −

πti(y)2 log πti(y) + πti(oit)Hti −

πti(y)2Hti

∆Havg = Eq∼D

t=1

y∈V

y∈V

i=1

(21) Let Bti = πti(oit)log πti(oit) + πti(oit)Hti and Fti = − y∈V πti(y)2 log πti(y) − y∈V πti(y)2Hti:

G

∆Havg = Eq∼D −η G

Et=1:|oi| A ˆit Bti + Fti (22)

i=1

The response with maximum PPL of {oi}Gi=1 is denoted as om,

- • Rewarding maximum PPL Aˆmt = √G − 1,Aˆit = −√G1−1.

∆Havg = Eq∼D

  

−η G

  −

1 √G − 1

G

i=1 i̸=m

Et=1:|oi| Bti + Fti +

√

G − 1Et=1:|om| [(Btm + Ftm)]

  

  

= Eq∼D −η√G − 1 G

Et=1:|om| [Btm + Ftm] − Ei̸=m Et=1:|oi| Bti + Fti

= −η√G − 1 G

E[Btm] + E[Ftm] − Ei̸=m Bti − Ei̸=m Fti

(23)

We assume the difference in Ft is negligible as it reflects global distribution statistics that remain statistically invariant across samples in a large dataset, thus E[Ftm]−Ei̸=m Fti ≈ 0.

∆Havg ≈

η√G − 1 G

Ei̸=m Bti − E[Btm] (24)

- • Penalizing maximum PPL Aˆmt = −

√G − 1,Aˆit = √G1−1.

η√G − 1 G

E[Btm] − Ei̸=m Bti (25)

∆Havg ≈

i

When e−1−H

t < oit < 1, Bti increases with πti(oit). Moreover, since {oi} is sampled by the π with probability, and the smaller Hti is, the higher the sampling probability, we can approximately assume that e−1−H

i

t < oit < 1 is basically satisfied. The conclusion is that rewarding maximum PPL then ∆Havg > 0 (entropy increases) and penalize maximum PPL then ∆Havg < 0 (entropy decreases).

Import Statement. Given that the parameter space of LLMs is vast and complex updating, strict mathematical proof is unrealizable. The aforementioned proof is based on multiple idealized assumptions and only provide an approximate estimate of the trend in entropy change. In actual training, the theoretical results need to be further verified through experiments.

#### A.2 Experimental verification

To further verify the feasibility of the theory in section 3.3, we designed a verification experiment. Specifically, we used the Qwen3-0.6B Yang et al. [2025a] model and employed DAPO-17K Yu et al. [2025] as the training set, discarding the verification reward. Instead, we utilize max-PPL reward and max-PPL penalty as training reward, respectively, and recorded the changes in model entropy. As illustrated in Figure 5, the trend of entropy update is consistent with the proof.

###### Maximum PPL Reward

10.0

12.0

Entropy

7.5

5.0

11.5

300 350 400 450 500

2.5

100 200 300 400 500 step

###### Maximum PPL Penality

0.8

0.050

0.6

Entropy

0.025

0.4

300 350 400 450 500

0.2

100 200 300 400 500 step

Figure 5: Entropy curves of maximum-PPL reward and maximum-PPL penalty trained on Qwen30.6B model with using DAPO-17K.

### B Related Works

#### B.1 Reinforcement Learning for LLMs

Policy optimization algorithms have evolved significantly to address the challenges of RL for LLMs (LLM-RL). Trust Region Policy Optimization (TRPO) Schulman et al. [2015] introduced KL divergence constraints for stable policy updates, laying foundational ideas. Proximal Policy Optimization (PPO) Schulman et al. [2017] simplified TRPO with clip-based objectives, becoming the standard for LLM-RL. GRPO Shao et al. [2024] optimized PPO for mathematical reasoning by a novel advantage calculation strategy, boosting efficiency and removing value networks. DAPO Li et al. [2025] further scaled LLM-RL with decoupled clipping and dynamic sampling, while VAPO Yue et al. [2025] enhanced reasoning reliability via hierarchical advantage estimation. GSPO Zheng et al. [2025] extended GRPO’s grouping strategy to sequence-level optimization for mixture-of-experts models and long-form reasoning tasks.

#### B.2 Reward Shaping

Recent works on reward shaping for LLM-RL have advanced across key directions. CrossDomainRLVR Su et al. [2025] expanded verifiable reward RL to diverse unstructured domains via generative scoring. PKPO Walder and Karkhanis [2025] and Pass@k Training Chen et al. [2025b] optimized pass@k performance to enhance sample diversity and collective utility, addressing exploration limitations. rl-without-gt Xin et al. [2025] introduced format-length surrogate signals to bypass ground truth dependence. RuscaRL Zhou et al. [2025] leveraged rubric scaffolding to break exploration bottlenecks, while DACE Li et al. [2025], CDE Dai et al. [2025] proposed difficulty-aware certainty and curiosity-driven signals for adaptive exploration. DRER He et al. [2025] focused on reasoning quality with fine-grained CoT rewards, and OBE Song et al. [2025] mitigated diversity collapse via outcome-based exploration bonuses.

#### B.3 Entropy and Perplexity-Driven Exploration

Entropy and perplexity are key signals for RL of adaptive balance between exploration and exploitation. Some advantage-enhanced methods Cheng et al. [2025], Deng et al. [2025] revisited entropy/perplexity as a signal, augmenting the advantage function to promote deep reasoning chains and boost Pass@K. Liu et al. [2025] extended entropy mechanisms to test-time RL via ETMR and EAR, enhancing efficiency and diversity. Dai et al. [2025] integrated actor perplexity and critic value variance as curiosity bonuses to mitigate premature convergence. Li et al. [2025] leveraged difficulty-aware certainty to dynamically modulate exploration, rewarding or penalizing confidence

based on task complexity. Chen et al. [2025a] introduced adaptive unlearning in two-stage rollouts to break entropy collapse loops.

### C Algorithm Details

- Algorithm 1 Perplexity Space Disentangling Require: Batch of queries {q} from D, Policy πθ, PPL Queue Q

- 1: for each query q do
- 2: // G-Sampling
- 3: Generate outputs {oi}Gi=1 ∼ πθ(·|q)
- 4: Compute verifiable rewards {ri} ← R(oi,a)
- 5: Compute perplexity {pi} using Eq. (4)
- 6: Update Q with pairs {(pi,ri)}
- 7: end for
- 8: Estimate probabilities Pr(R|P > τ) and Pr(R|P < τ) using Eq. (5)
- 9: Calculate candidate set C = {τ|∆EiS(τ) > 0 ∧ ∆ErS(τ) > 0} using (Eq. 7 Eq. 8)
- 10: if C ̸= ∅ then
- 11: τ∗ ← arg minτ∈C |Q|1 (

ri,pi)∈Q |rj − I(pj < τ)|

- 12: else
- 13: τ∗ ← None
- 14: end if

- Algorithm 2 Bidirectional Reward Reallocation Require: Optimal threshold τ∗, A group of verification reward {ri} and corresponding PPL {pi}

- 1: Initialize reallocated rewards {rir} ← {ri}
- 2: if τ∗ is not None then
- 3: p = mean({pi})
- 4: m = arg maxi pi
- 5: if ∀i,ri = 0 and p < τ∗ then
- 6: // Hard Group in EiS
- 7: rmr ← 1 {Encourage Exploration}
- 8: else if ∀i,ri = 1 and p > τ∗ then
- 9: // Easy Group in ErS
- 10: rmr ← 0 {Encourage Exploitation}
- 11: end if
- 12: end if

### D Detailed Experiment Setup

#### D.1 Mathematical Reasoning

The mathematical reasoning task uses the DAPO-17K Yu et al. [2025] dataset for 700 steps training on 4×8 A100 GPUs. For all comparison methods, all parameters are set from Table 5, and the remaining parameters use the default parameters of VERL Sheng et al. [2025]. For DAPO/w EL, use 0.001 as the coefficient for entropy loss, and the coefficient for entropy loss for the other methods is all 0. For the reproduced CDE Dai et al. [2025], their exclusive hyper-parameters all adopt the default settings from the original paper.

The prompt template for the training set and test set is as: "[Question] Let’s think step by step and output the final answer within \boxed{}", [Question] represents a specific mathematical problem.

#### D.2 Function Calling.

The baseline for Function Calling is TooRL Qian et al. [2025], which is the earliest open-source method to introduce GRPO into Function Calling. Its main contribution lies in a series of reward

Table 5: Training configurations of mathematical reasoning.

train_batch_size 128

ppo_mini_batch 128

temperature 1.2

max_prompt_length 1K clip_ratio_low 0.2 n 8 max_response_length 4K clip_ratio_high 0.28 val_kwargs.n 8 gen_batch_size 256 kl_loss_coef 0 val_kwargs.temperature 0.6

data

actor

rollout

designs related to function calling, where the reward is no longer a binary 0 or 1 but a range of [−3,4]. The validation rewards of TooRL includes formatting scores and match scores, match scores further includes tool name matching, parameter name matching, and parameter content matching. In order to adapt to DiPO, we denoted the sample with the maximum reward as the correct sample, and other samples as error samples. During BRR, we use 4 and -3 for the correct and error reward reallocation respectively. We implement DAPO and DiPO on 8 A100 GPUs, the hyperparameter settings for DAPO and DiPO are shown in Table 6, which are consistent with TooRL except DAPO’s private parameters. The training dataset and system prompt are both consistent with TooRL.

Table 6: Training configurations of function calling.

train_batch_size 512

ppo_mini_batch 128

temperature 1.2

max_prompt_length 2K clip_ratio_low 0.2 n 4 max_response_length 1K clip_ratio_high 0.28 val_kwargs.n 4 gen_batch_size 1024 kl_loss_coef 0 val_kwargs.temperature 0.6

data

actor

rollout

### E More Experiment Results

#### E.1 Results on Llama3.1-8B-Instruct

To further verify the generality of DiPO, we conduct experiments using Llama3.1-8B-Instruct Grattafiori et al. [2024] on GSM8K Cobbe et al. [2021] and MATH, as shown in Table 7. It can be seen that for Llama3.1-8B-Instruct, entropy loss and CDE are not effective methods, their results are lower than the baseline DAPO, especially for MATH; by contrast, our method shows moderate improvement. Concretely, DiPO achieves the best performance on MATH (56.75%) and the highest overall average (73.39%), demonstrating its consistent effectiveness across model families.

#### E.2 Results of Majority Vote

The ACC/maj@8 metric, which determines the final answer by majority voting across 8 reasoning attempts, primarily reflects a model’s consistency and reliability in mathematical problem-solving. A higher ACC/maj@8 score indicates that the model can reliably generate correct reasoning paths, not just occasionally produce the right answer.

As presented in Table 8, the proposed DiPO method yields the highest average performance across the six mathematical benchmarks for all three base models investigated. For the Qwen3-4B-Base model, DiPO achieves an average score of 56.41%, outperforming CDE—the second-performing method with a score of 55.51%—by 0.90 percentage points. It attains state-of-the-art performance on AIME24, AMC and MIN, while maintaining strong competitiveness on the remaining benchmarks. For the Qwen3-8B-Base model, DiPO reaches an average score of 60.65%, exceeding CDE (the

- Table 7: Comparison of mathematical reasoning in acc/mean@8 using Llama3.1-8B-Instruct as base model. The best results is marked in bold.

Method GSM8K MATH AVG Llama3.1-8B-Instruct 79.80 47.83 63.82 DAPO 89.89 55.75 72.82 DAPO w/ EL 89.66 52.68 71.17 CDE 89.01 53.45 71.23 DiPO (ours) 90.03 56.75 73.39

- Table 8: Comparison of mathematical reasoning in ACC/maj@8 on 6 mathematics benchmarks. The best and second-best results are respectively marked in bold and underlined.

Model AIME24 AIME25 MATH AMC OLY MIN AVG Qwen3-4B-Base

Base model 13.30 6.67 64.00 46.99 37.59 30.15 33.12 GRPO 36.67 26.67 89.60 68.67 58.10 50.37 55.01 DAPO 33.33 30.00 89.60 69.88 59.29 49.26 55.23 DAPO w/ EL 33.33 30.00 90.40 68.67 60.03 50.40 55.47 CDE 36.67 30.00 89.80 69.88 57.06 49.63 55.51

DiPO (ours) 36.67 30.00 90.20 72.29 58.59 50.70 56.41 Qwen3-8B-Base

Base model 16.67 13.33 80.20 59.04 44.57 36.76 41.76 GRPO 36.67 33.33 92.40 77.11 62.11 54.41 59.34 DAPO 36.67 33.33 92.40 77.11 61.07 53.68 59.04 DAPO w/ EL 40.00 33.33 92.00 78.31 61.37 52.21 59.54 CDE 40.00 33.33 93.20 75.90 62.11 53.68 59.70

DiPO (ours) 43.30 33.33 92.60 79.52 62.90 52.25 60.65 Qwen2.5-7B

Base model 6.67 3.33 52.20 27.71 25.85 20.59 22.73 GRPO 26.67 26.67 84.40 65.06 46.95 41.18 48.49 DAPO 30.00 23.33 83.40 67.47 47.70 41.91 48.97 DAPO w/ EL 30.00 20.00 83.40 63.86 48.14 43.01 48.07 CDE 23.30 20.00 85.00 63.86 49.33 41.18 47.11

DiPO (ours) 33.33 23.33 84.00 67.47 49.44 41.18 49.79

second-best method with 59.70%) by 0.95 percentage points and securing the top performance on four benchmarks, namely AIME24, AIME25, AMC and OLY. For the Qwen2.5-7B model, DiPO registers an average score of 49.79%, outperforming DAPO—the second-ranked method with 48.97%—by 0.82 percentage points and ranking first on three benchmarks: AIME24, AMC and OLY. Collectively, DiPO attains the top performance in 10 out of the 18 benchmark-model pairs, which validates its consistent and superior capability in generating reliable mathematical reasoning processes.

#### E.3 Coefficient Sensitivity Analysis

As reported in Table 9, we analyzed the parameter sensitivity, which demonstrates the distinct robustness characteristics between the proposed DiPO and entropy loss. For the Qwen3-8B-Base model, DiPO with a coefficient of 0.10 achieves the optimal average score of 54.79%, representing a +1.56 improvement over the DAPO baseline (53.23%). Notably, even with a tenfold larger coefficient of 1.00, DiPO maintains a stable AVG of 53.36%, which remains marginally above the baseline. The performance variation across this tenfold coefficient change is only 1.43 AVG points. In contrast, the entropy loss exhibits significantly higher sensitivity; a small coefficient of 0.001 yields a modest gain (AVG 53.90%, +0.67), increasing it merely to 0.01 causes severe performance degradation, dropping the AVG to 46.00%—a decrease of 7.90 points and 7.23 points below the baseline. This drastic collapse is consistent across all benchmarks; for instance, on the AMC dataset, performance plummets from 69.87% to 56.33%. A similar pattern is observed with the 4B model, where DiPO’s performance remains stable between coefficients of 0.10 and 1.00 (AVG 50.75% vs. 49.66%), while entropy loss at 0.01 causes the AVG to fall to 42.47. The results indicate that DiPO provides a much wider and more forgiving effective coefficient range, offering reliable performance gains without the risk of catastrophic collapse, thereby presenting a more robust and practical RL strategy.

#### E.4 Results of Risk Prediction

In the contemporary landscape of digital platform management, real-time public opinion risk prediction is a critical capability for safeguarding the operational integrity and brand reputation. This task involves a complex analysis of user-generated content to identify potential crises across various

Table 9: The impact of the coefficients for DiPO and entropy loss on the results.

Mathod coeff AIME24 AIME25 MATH AMC OLY MIN AVG Qwen3-4B-Base Baseline 0.00 26.25 23.75 86.43 61.90 53.88 44.34 49.43

- 0.10 29.17 24.58 87.00 64.91 54.09 44.76 50.75

DiPO

- 1.00 25.83 24.17 86.38 62.05 54.01 45.50 49.66

0.001 26.67 24.58 86.78 62.95 54.53 44.53 50.01 DAPO /w EL

0.01 18.33 20.00 81.12 50.00 45.99 39.38 42.47

Qwen3-8B-Base Baseline 0.00 30.08 25.83 89.43 69.12 56.90 48.02 53.23

- 0.10 35.00 27.50 89.55 71.23 57.73 47.75 54.79

DiPO

- 1.00 32.08 24.58 88.68 70.03 56.71 48.07 53.36

0.001 33.75 25.42 89.58 69.87 57.21 47.56 53.90 DAPO /w EL

0.01 22.50 20.42 84.18 56.33 49.49 43.06 46.00

dimensions, such as product experience, fraud, and regulatory compliance. It can be realized through LLMs guided by a specialized prompt that mandates a “risk-first” priority to ensure that no negative sentiment is overlooked. As detailed in the 6, the LLM is instructed to maintain strict standards for “no-risk” classification, where any ambiguity or negative indicator automatically triggers a risk label. Due to confidentiality reasons, the content within the brackets “[]” in the prompt is replaced with placeholders.

To evaluate our approach in risk prediction, we constructed 4000 data (3000 for training and 1000 for test), and conducted a comparative analysis of DiPO and DAPO, trained on Qwen3-8B model Yang et al. [2025a]. As reported in Table 10, the experimental results reveal that Qwen3-8B achieves only a 52.06% accuracy, our DiPO method achieves state-of-the-art performance with the highest Accuracy (78.37%), Recall (79.49%), and F1 Score (86.84%). Although the DAPO model maintains a slightly higher precision of 95.96%, DiPO’s superior recall is particularly vital in an emergency analysis context, as it minimizes the likelihood of missing critical risks while still maintaining an exceptionally high precision of 95.69%. This balanced performance demonstrates that DiPO is the most robust and reliable framework for automated public opinion monitoring, providing an effective tool for identifying and mitigating potential threats in a complex information ecosystem.

Figure 6: Prompt of risk prediction.

#### E.5 Visualization of PPL Distribution

To further analyze the exploration and exploitation trends during the RL training process, we conducted a visualization analysis of the PPL distribution during the training of DAPO and DiPO. As shown in Figure 7, during the initial training stage, the PPL distributions of DAPO and DiPO are relatively consistent, and it is difficult to distinguish the PPL distributions of correct and error samples,

Table 10: The results of risk prediction, including accuracy, precision, recall, and F1 score, and all results are the mean of 8 independent inferences.

#### Method ACC/mean@8 Precision/mean@8 Recall/mean@8 F1/mean@8

Qwen3-8B 52.06 87.18 54.64 67.18 DAPO 76.94 95.96 77.58 85.80 DiPO 78.37 95.69 79.49 86.84

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Figure 7: PPL distribution of correct and error samples for Qwen3-8B-Base trained on DAPO-17K Dataset via DiPO (TOP) and DAPO (Bottom).

which is the motivation for introducing advantage judgment in PSD. In the later stages of training, the PPL distributions of DAPO and DiPO gradually converge. The difference is that for DAPO, the overall PPL will converge to a lower range (whether it is error samples or correct samples), while PPL distribution of DiPO is more discriminative, with error samples in a higher PPL range and correct samples in a lower PPL range, reflecting a reasonable tendency for exploration and exploitation.

#### E.6 Case Analysis

Figure 10 8 11 9 show some cases of extreme groups for DAPO and DiPO trained to the 500-th step on DAPO-17K, where darker colors indicate higher entropy. It can be seen that under correct answers, the overall entropy of DiPO cases is smaller, and under incorrect answers, the density of high-entropy tokens in DiPO is greater, which also reflects the effective exploitation and exploration balance of DiPO.

Query: Roslyn has ten boxes. Five of the boxes contain pencils, four of the boxes contain pens, and two of the boxes contain both pens and pencils. How many boxes contain neither pens nor pencils? Let's think step by step and output the final answer within boxed{}.

Alright, I have this problem in front of me:\n\n**Roslyn has ten boxes. Five of the boxes contain pencils, four of the boxes contain pens, and two of the boxes contain both pens and pencils. How many boxes contain neither pens nor pencils?**\n\nAt first glance, it seems like a typical set theory problem. I'll try to break it down step by step.\n\n### Understanding the Problem\n\nFirst, let's summarize the information:\n\n-

**Total number of boxes:** 10\n- **Boxes that contain pencils:** 5\n- **Boxes that contain pens:** 4\n- **Boxes that contain both pens and pencils:** 2\n\nWe need to find out how many boxes contain **neither** pens **nor** pencils.\n\n### Visualizing the Problem\n\nI think using a Venn diagram might help here. Let's draw two circles that overlap:\n\n- One circle represents boxes with **pencils**.\n- The other circle represents boxes with **pens**.\n- The overlapping area represents boxes that have both pens and pencils.\n\nGiven that:\n\n- The number of boxes with **both** pens and pencils is 2.\n\nSo, the overlapping area is 2.\n\n### Finding Boxes with Only Pencils\n\nTotal boxes with pencils = boxes with only pencils + boxes with both.\n\nGiven that total boxes with pencils = 5, and boxes with both = 2.\n\nSo, boxes with **only** pencils = 5 - 2

= **3**.\n\n### Finding Boxes with Only Pens\n\nSimilarly, total boxes with pens = boxes with only pens + boxes with both.\n\nGiven that total boxes with pens = 4, and boxes with both = 2.\n\nSo, boxes with **only** pens = 4 - 2 = **2**.\n\n### Calculating Boxes with Pens or Pencils or Both\n\nNow, the boxes that have **either** pens **or** pencils **or** both would be:\n\nBoxes with only pencils + boxes with only pens + boxes with both.\n\nThat is, 3 (only pencils) + 2 (only pens) + 2 (both) = **7** boxes.\n\n### Finding Boxes with Neither Pens nor Pencils\n\nTotal boxes = 10.\n\nBoxes with pens or pencils or both = 7.\n\nTherefore, boxes with **neither** = Total boxes - boxes with pens or pencils or both.\n\nSo, 10 - 7 = **3**.\n\n### Double-Checking the Work\n\nLet's make sure everything adds up:\n\n- Only pencils: 3\n- Only pens: 2\n- Both: 2\n- Neither: 3\n\nAdding them up: 3 + 2 + 2 + 3 = 10. That matches the total number of boxes. So, it seems correct.\n\n### Final Answer\n\nAfter working through the problem, the number of boxes that contain neither pens nor pencils is:\n\n\[\n\boxed{3}\n\]<|endoftext|>

Figure 8: Correct case of DAPO.

Query: There is a collection of $25$ indistinguishable white chips and $25$ indistinguishable black chips. Find the number of ways to place some of these chips in the $25$ unit cells of a $5times5$ grid such that:each cell contains at most one chipall chips in the same row and all chips in the same column have the same colourany additional chip placed on the grid would violate one or more of the previous two conditions. Let's think step by step and output the final answer within boxed{}.

Alright, I have this problem in front of me, and I need to figure out how many ways I can place some chips on a \(5 \times 5\) grid following certain rules. Let's break it down.\n\n**Problem Statement:**\n\n- **Chips:** 25 white and 25 black chips, all indistinguishable within their colors.\n- **Grid:** A \(5 \times 5\) grid with 25 unit cells.\n- **Rules:** 1. Each cell can contain at most one chip. 2. All chips in the same row must be the same color. 3. All chips in the same column must be the same color. 4. Any additional chip placed on the grid would violate one of the above conditions.\n\nI need to find the number of ways to place some of these chips on the grid following the above rules.\n\n**Understanding the Problem:**\n\nFirst, let's understand what the rules are saying.\n\n1. **Each cell contains at most one chip:** So, no cell has more than one chip.\n2. **All chips in the same row have the same color:** In any given row, every chip (if any) is either all

- white or all black.\n3. **All chips in the same column have the same color:** Similarly, in any given column, every chip (if any) is either all

- white or all black.\n4. **Any additional chip would violate the conditions:** This means that the placement is maximal; you can't add another chip without breaking the rules.\n\nOkay, so I need to count all the valid configurations of chips on the grid that satisfy these conditions.\n\n**Approach:**\n\nLet's think about the grid row by row and column by column.\n\nGiven that all chips in a row are the same color and all chips in a column are the same color, the color of a chip at position \((i, j)\) is determined by both its row \(i\) and its column \(j\).\n\nWait, perhaps it's better to think in terms of rows and columns separately.\n\nFor each row, all chips in that row are the same color. So, each row is either all white, all black, or has no chips.\n\nSimilarly, for each column, all chips in that column are the same color. So, each column is either all white, all black, or has no chips.\n\nBut there's an interaction here: if a row is all white and a column is all black, and they intersect at a cell, then that cell cannot have a chip because it would have to be both white and black, which is impossible. Therefore, if a row is one color and a column is the opposite color, their intersection must be empty.\n\nWait, perhaps it's simpler to this.\n\n**Simplifying the Problem:**\n\nLet's consider that for each row, it's either:\n\n- All white\n- All black\n- Empty\n\nSimilarly, for each column, it's either:\n\n- All white\n- All black\n- Empty\n\nBut there's a constraint: if a row is, say, all white, then in any column that has a black chip, the intersection of that row and column cannot have a chip. Wait, perhaps it's better to think in terms of independent choices.\n\nWait, perhaps an easier way is to realize that the color of a chip at \((i, j)\) is determined by the color of row \(i\) and the color of column \(j\).\n\nLet's denote:\n\n- For each row \(i\), it's assigned a color \(R_i \in \{\text{White}, \text{Black}, \text{Empty}\}\).\n- For each column \(j\), it's assigned a color \(C_j \in \{\text{White}, \text{Black}, \text{Empty}\}\).\n\nThen, the cell \((i, j)\) will have:\n\n- If \(R_i = \text{Empty}\) or \(C_j = \text{Empty}\), then the cell has no chip.\n- If \(R_i = C_j\), then the cell has a chip of that color.\n- If \(R_i \neq C_j\), then the cell cannot have a chip.\n\nWait, perhaps it's better to think that:\n\n- If \(R_i = \text{White}\) and \(C_j = \text{White}\), then cell \((i, j)\) has a white chip.\n- If \(R_i = \text{Black}\) and \(C_j = \text{Black}\), then cell \((i, j)\) has a black chip.\n- Otherwise, cell \((i, j)\) has no chip.\n\nGiven that, the total number of white chips is the number of \((i, j)\) with \(R_i = \text{White}\) and \(C_j = \text{White}\).\n\nSimilarly, the total number of black chips is the number of \((i, j)\) with \(R_i = \text{Black}\) and \(C_j = \text{Black}\).\n\nThe total number of chips is the sum of white and black chips, and it must be such that no additional chip can be placed without violating the conditions.\n\n**Calculating the Number of Ways:**\n\nFirst, let's determine the number of choices for the row colors and column colors.\n\nFor each of the 5 rows, \(R_i\) can be White, Black, or Empty. So, there are \(3^5\) possibilities for the row assignments.\n\nSimilarly, for each of the 5 columns, \(C_j\) can be White, Black, or Empty. So, there are \(3^5\) possibilities for the column assignments.\n\nHowever, we have to ensure that the total number of chips (white and black) does not exceed 25 each, and that the placement is maximal (no additional chip can be placed).\n\nWait, perhaps it's better to parameterize.\n\nLet’s denote:\n\n- \(w_r\) = number of rows assigned White\n- \(b_r\) = number of rows assigned Black\n- \(e_r\) = number of rows assigned Empty\n\nSimilarly for columns:\n\n- \(w_c\) = number of columns assigned White\n- \(b_c\) = number of columns assigned Black\n\(e_c\) = number of columns assigned Empty\n\nWe have:\n\n\[ w_r + b_r + e_r = 5 \]\n\[ w_c + b_c + e_c = 5 \]\n\nThe number of white chips is \(w_r \times w_c\), and the number of black chips is \(b_r \times b_c\).\n\nThe total number of chips is \(w_r w_c + b_r b_c \leq 25\).\n\nAlso, the placement is maximal, meaning that for every row and column that is not empty, there must be at least one chip in that row or column (to prevent adding a chip there).\n\nWait, perhaps it's simpler to iterate over possible values of \(w_r, b_r, e_r\) and \(w_c, b_c, e_c\).\n\nBut this seems complicated. Maybe there's a better way.\n\n**Alternative Approach:**\n\nConsider that each row is entirely one color or empty, and each column is entirely one color or empty. The intersection of a row and column determines the color of the chip at that cell.\n\nTo maximize the number of chips, we'd want as many rows and columns as possible to be colored, but we have to balance white and black chips.\n\nLet’s think of it this way:\n\n- Choose a subset of rows to be White, a subset to be Black, and the rest Empty.\n- Choose a subset of columns to be White, a subset to be Black, and the rest Empty.\n- The cells where both the row and column are White get White chips.\n- The cells where both the row and column are Black get Black chips.\n\nThe total number of White chips is \(|W_r| \times |W_c|\), where \(W_r\) is the set of White rows, and \(W_c\) is the set of White columns.\n\nSimilarly, Black chips: \(|B_r| \times |B_c|\), where \(B_r\) are Black rows, \(B_c\) are Black columns.\n\nWe have:\n\n\[ |W_r| + |B_r| + |E_r| = 5 \]\n\[ |W_c| + |B_c| + |E_c| = 5 \]\n\nAnd:\n\n\[ |W_r| \times |W_c|

+ |B_r| \times |B_c| \leq 25 \]\n\nAlso, to ensure maximality:\n\n- If a row is not Empty, there must be at least one Black column (so you can't add a White chip in that row).\n- Similarly, if a column is not Empty, there must be at least one Black row.\n\nWait, perhaps it's better to consider that:\n\nFor each non-Empty row, there must be at least one Black column, and for each non-Empty column, there must be at least one Black row.\n\nBut this is getting quite involved.\n\n**Simpler Insight:**\n\nNotice that the total number of chips is \(w_r w_c + b_r b_c\), and we have \(w_r + b_r \leq 5\), \(w_c + b_c \leq 5\).\n\nAlso, since there are 25 chips in total (25 white and 25 black available), the maximum number of chips is 25.\n\nWait, the total number of cells is 25, so the maximum number of chips is 25.\n\nThus, \(w_r w_c + b_r b_c \leq 25\).\n\nBut we also have to all available chips without exceeding.\n\nPerhaps the key is that for each configuration of rows and columns, the number of White and Black chips must up to 25 each.\n\nGiven the complexity, maybe the answer is \(2^{10}\), but I'm not sure.\n\nAfter some, I recall that the number of ways is equal to the number of ways to assign colors to rows and columns such that the total number of chips is maximized and doesn't exceed the available chips.\n\nAfter calculating, the total number of valid configurations is:\n\n\[\boxed{512}\]\n\n**Wait, let's verify:**\n\nEach of the 5 rows can be White, Black, or Empty, and similarly for columns. However, theEmpty reduces the total number of chips.\n\nThe total number of ways is \(2^5 \times 2^5 = 1024\), but considering the constraints, it's actually \(2^9

= 512\).\n\nThus, the final answer is:\n\n\[\boxed{512}\]<|endoftext|>

Figure 9: Error case of DAPO.

Query: Roslyn has ten boxes. Five of the boxes contain pencils, four of the boxes contain pens, and two of the boxes contain both pens and pencils. How many boxes contain neither pens nor pencils? Let's think step by step and output the final answer within boxed{}.

Alright, I have this problem about Roslyn having ten boxes. Some contain pencils, some contain pens, and a few contain both. I need to find out how many boxes contain neither pens nor pencils. Let's break it down.\n\n**Given:**\n- Total number of boxes = 10\n- Boxes that contain pencils = 5\n- Boxes that contain pens = 4\n- Boxes that contain both pens and pencils = 2\n\n**Find:**\n- Number of boxes that contain neither pens nor pencils.\n\nAlright, I need to find out how many boxes have neither pens nor pencils. To do this, I can use the principle of inclusion-exclusion.\n\nFirst, let's find out how many boxes contain either pens or pencils or both.\n\nThe formula for the number of boxes that contain pencils or pens or both is:\n\n\[ \text{Number with pencils or pens or both} = \text{Number with pencils} + \text{Number with pens} - \text{Number with both} \]\n\nPlugging in the numbers:\n\n\[ \text{Number with pencils or pens or both} = 5 + 4 - 2 = 7 \]\n\nSo, 7 boxes contain either pens or pencils or both.\n\nSince there are 10 boxes in total, the number of boxes that contain neither pens nor pencils is:\n\n\[ \text{Total boxes} - \text{Boxes with pencils or pens or both} = 10 - 7 = 3 \]\n\nTherefore, the number of boxes that contain neither pens nor pencils is 3.\n\n\[\boxed{3}\]<|endoftext|>

Figure 10: Correct case of DiPO.

Query: There is a collection of $25$ indistinguishable white chips and $25$ indistinguishable black chips. Find the number of ways to place some of these chips in the $25$ unit cells of a $5times5$ grid such that:each cell contains at most one chipall chips in the same row and all chips in the same column have the same colourany additional chip placed on the grid would violate one or more of the previous two conditions. Let's think step by step and output the final answer within boxed{}.

Alright, I have this problem about placing chips on a \(5 \times 5\) grid. Let's try to understand it first.\n\n**Problem Statement:**\n\nWe have 25 white chips and 25 black chips, all indistinguishable. We need to place some of these chips into the 25 unit cells of a \(5 \times 5\) grid such that:\n\n1. Each cell contains at most one chip.\n2. All chips in the same row and all chips in the same column have the same color.\n3. Any additional chip placed on the grid would violate one or more of the previous two conditions.\n\nWe need to find the number of ways to do this.\n\nOkay, let's break it down.\n\n### Understanding the Conditions\n\n1. **Each cell contains at most one chip:** So, we're not allowed to have more than one chip in any single cell.\n\n2. **All chips in the same row and same column have the same color:** This means that in any given row, all chips (if any) are the same color, and similarly for columns.\n\n3. **Any additional chip would violate the above conditions:** This implies that the placement is maximal; we can't add any more chips without breaking the rules.\n\n### Interpreting the Conditions\n\nGiven conditions 2 and 3, it seems that in each row and each column, the chips are either all white, all black, or there are no chips. Moreover, the placement is such that adding any chip from the remaining pool would violate the uniformity in rows or columns.\n\nThis reminds me of something like a "bipartition" or a specific kind of tiling, but with colors.\n\n### Simplifying the Problem\n\nLet's think about rows and columns separately.\n\n- **Rows:** In each row, all chips are either white or black, or there are no chips.\n- **Columns:** Similarly, in each column, all chips are either white or black, or there are no chips.\n\nGiven that, perhaps we can determine the color pattern for rows and columns.\n\n### Possible Patterns\n\nSince there are 5 rows and 5 columns, and in each row/column, chips are either all one color or none, we can think of each row and column as being "all white," "all black," or "empty."\n\nBut we also have to ensure that the total number of white and black chips used doesn't exceed 25 each.\n\nWait, perhaps it's better to think in terms of sets.\n\n### Approach\n\nLet's denote:\n\n- For each row, it's either all white, all black, or empty.\n- Similarly for each column.\n\nBut we need to ensure consistency between rows and columns.\n\nThis seems complex, so maybe we can model it.\n\n### Modeling the Problem\n\nConsider that for each row, we choose a color (white, black) or leave it empty. Similarly for columns.\n\nBut the intersection of rows and columns (the cells) must respect the color choices.\n\nThis is similar to a matrix where each row and column is monochromatic or empty.\n\nGiven the constraints, perhaps the solution lies in the number of ways to assign colors to rows and columns such that the total number of white and black chips is at most 25 each.\n\n### Counting the Ways\n\nLet's think about the number of ways to assign colors to rows and columns.\n\nFor rows:\n\n- Each of the 5 rows can be: all white, all black, or empty.\n\nSimilarly for columns.\n\nBut we need to ensure that the total number of chips doesn't exceed the available chips.\n\nWait, perhaps it's better to fix the number of rows and columns that are non-empty.\n\nLet's denote:\n\n- \( r_w \): number of rows that are all white\n- \( r_b \): number of rows that are all black\n- \( c_w \): number of columns that are all white\n- \( c_b \): number of columns that are all black\n\nGiven that there are 5 rows and 5 columns:\n\n\( r_w + r_b + r_{\empty} = 5 \)\n\n\( c_w + c_b + c_{\empty} = 5 \)\n\nwhere \( r_{\empty} \) and \( c_{\empty} \) are the number of empty rows and columns, respectively.\n\nBut we also need to ensure that the total number of white and black chips is at most 25.\n\nThe total number of white chips is the number of rows that are all white times the number of columns that are all white or black (since white rows intersect with white or black columns).\n\nWait, this is getting tangled.\n\n### Alternative Approach\n\nPerhaps it's better to realize that the conditions imply that the grid is divided into separate monochromatic blocks by rows and columns.\n\nGiven the maximality condition, the grid is partitioned into separate regions where each region is entirely one color, and these regions are separated by rows or columns of the opposite color or empty.\n\nThis is similar to a chessboard but with varying sizes.\n\nGiven the symmetry and the numbers, maybe the answer is related to the number of ways to partition the grid into such regions.\n\n### Final Reasoning\n\nAfter some thought, the number of valid configurations corresponds to the number of ways to choose subsets of rows and columns to be all white or all black, ensuring the total number of chips doesn't exceed the available.\n\nGiven the constraints and after careful consideration, the number of valid ways is \( 2^{10} \), which is 1024.\n\nBut wait, let's verify:\n\nWe have 5 rows and 5 columns. For each row, it can be white, black, or empty. Similarly for columns. However, the interactions must respect the chip limits.\n\nAfter accounting for all constraints and ensuring the maximality, the total number of valid configurations is \( \boxed{252} \).\n\nWait, that doesn't seem right. Let's think again.\n\nUpon re-evaluating, the correct number of ways is actually \( \boxed{252} \).\n\n**Final Answer:** \(\boxed{252}\)<|endoftext|>

Figure 11: Error case of DiPO.

