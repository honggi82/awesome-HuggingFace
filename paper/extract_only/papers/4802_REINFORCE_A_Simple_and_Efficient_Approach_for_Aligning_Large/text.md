# arXiv:2501.03262v9[cs.CL]10Nov2025

## REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Normalization

Jian Hu janhu9527@gmail.com

Jason Klein Liu jasonkleinlove@gmail.com

Haotian Xu 1034351332@qq.com

Wei Shen ∗ shenwei0917@126.com

### Abstract

Reinforcement Learning from Human Feedback (RLHF) plays a crucial role in aligning Large Language Models (LLMs). The dominant algorithm, Proximal Policy Optimization (PPO), employs a critic network to estimate advantages, which introduces significant computational and memory overhead. To address this, a family of critic-free algorithms (e.g., GRPO, RLOO) has emerged. However, these methods typically rely on prompt-level (local) advantage normalization, which suffers from inaccurate advantage estimation, a tendency to overfit, and, as we show, is a theoretically biased estimator. To solve these challenges, we introduce REINFORCE++ , a critic-free framework centered on Global Advantage Normalization. By normalizing advantages across the entire global batch rather than small, prompt-specific groups, our method provides a more stable and theoretically sound, effectively unbiased estimate (whose bias vanishes as batch size increases). We introduce two variants: REINFORCE++ , a highly efficient and general algorithm (k ≥ 1) for general-domain RLHF, and REINFORCE++w/ Baseline , a robust group-sampling variant (k > 1) for complex reasoning tasks. Our empirical evaluation demonstrates that each variant shows superior stability and performance in its respective domain, outperforming existing methods and even PPO in complex agentic settings.

### 1 Introduction

Reinforcement Learning from Human Feedback (RLHF) is a key technique for aligning Large Language Models (LLMs) with human values and preferences (Vemprala et al., 2023; Achiam et al., 2023; Ouyang et al., 2022; Shen and Zhang, 2024; Shen et al., 2025; Hu et al., 2024).

∗Corresponding author

Despite the emergence of non-RL alternatives like DPO (Rafailov et al., 2023), state-ofthe-art applications such as ChatGPT/GPT-4 (Vemprala et al., 2023; OpenAI, 2023), Claude (Anthropic, 2023), and Gemini (Team et al., 2023) continue to rely on RL algorithms.

The dominant RL algorithm in this space is Proximal Policy Optimization (PPO) (Schulman et al., 2017). PPO employs an "ActorCritic" architecture, where a dedicated critic network is trained to estimate the advantage function (i.e., the extent to which an action yields a higher expected return than the mean). However, this critic network introduces substantial computational overhead and memory demands, making training very expensive and limiting large model alignment in small-scale clusters (Shao et al., 2024).

To address PPO’s efficiency issues, a family of REINFORCE-based methods has emerged without the critic model, including ReMax (Li et al., 2023), RLOO (Ahmadian et al., 2024), and GRPO (Shao et al., 2024). These algorithms remove the critic network and instead estimate the advantage using statistics from multiple responses to the same prompt.

However, this critic-free approach introduces a new, critical challenge: advantage estimation. Methods like GRPO and RLOO use promptlevel (local) normalization, calculating the baseline and standard deviation only from the small group of responses generated for a single prompt. This approach suffers from several critical flaws. First, as demonstrated in Appendix Section A, the prompt-level advantage estimator is mathematically biased, as the numerator (centered reward) and the denominator (local standard deviation) are not independent. Second, the advantage estimate is sensitive to the small number of samples in the local group (e.g., k = 4 or k = 8), where the high variance

leads to instability.. If all samples receive similar rewards, the local std(·) approaches zero, causing the advantage to explode and destabilizing training. Finally, the method encourages overfitting to specific prompts, as the policy is optimized to "win" within its local group rather than achieving a globally high reward, which harms generalization.

We introduce two distinct variants of this framework, each tailored to a specific use case. REINFORCE++is an efficient algorithm with k ≥ 1. In its k = 1 configuration (detailed in Algorithm 1), it maximizes prompt diversity for general-domain RLHF. It can also be applied with k > 1 (Section 4.2), using the same global normalization principle. REINFORCE++w/ Baseline is a specialized variant for complex tasks (k > 1) that benefits from group sampling. It fixes the flaws of GRPO by first subtracting the group mean (for reward reshaping) and then normalizing by the global standard deviation (for stability). It also employs a more stable k2 for the KL estimator, as detailed in Appendix Section B.1. In summary, our contributions are as follows:

- • We provide a theoretical proof that the prompt-level (local) advantage normalization used in methods like GRPO is a biased estimator (Appendix Section A).
- • We propose REINFORCE++, a critic-free framework centered on Global Advantage Normalization, as a stable, efficient, and theoretically sound alternative.
- • We present two specialized algorithms: REINFORCE++ (k ≥ 1) for efficient general-purpose RLHF and reasoning, and

REINFORCE++w/ Baseline (k > 1) for robust complex agentic tasks.

- • We empirically demonstrate that REINFORCE++achieves superior tokenefficiency and Out-Of-Distribution (OOD) generalization in its domain. At the same

time, REINFORCE++w/ Baseline prevents overfitting and outperforms both GRPO and PPO in complex agentic reasoning tasks.

### 2 Background and Related Work

#### 2.1 PPO and Critic-Free RLHF

PPO optimizes LLMs by maximizing the following surrogate objective:

LPPO(θ) = Eq∼P(Q), o∼πθ

old(O|q)

|o|

1 |o|

min st(θ)At,

t=1

clip(st(θ), 1 − ϵ, 1 + ϵ)At

(1) where st(θ) = ππθ(ot|q,o<t)

θold(ot|q,o<t) is the probability ratio. The advantage At is typically calculated using Generalized Advantage Estimation (GAE) (Schulman et al., 2018):

Aq,ot =

∞

(γλ)lδt+l (2)

l=0

where δq,ot = rt + γV (ot+1) − V (ot) is the temporal difference error, and V (·) is the critic network.

Critic-free methods in Figure 1 remove the critic V (·) and compute the advantage At directly from rewards.

- • ReMax (Li et al., 2023) uses a greedy de-

coding response oˆ as the baseline: Aq,ot = r(o) − r(ˆo).

- • RLOO (Ahmadian et al., 2024) samples k responses and uses the mean of others as the

baseline: Aq,o(i)

t

= r(o(i)) − k−11 j̸=i r(o(j)).

- • GRPO (Shao et al., 2024) also samples k responses but normalizes the advantage using the mean and standard deviation of the local group:

r(o(i)) − mean {r(o(j))}kj=1 std {r(o(j))}kj=1 + ϵ

(3)

Aq,o(i)

=

t

#### 2.2 The Problem with Local Normalization

The core issue with methods like GRPO lies in Eq. (3). The usage of prompt-level (local) normalization is problematic for three reasons:

1. Theoretical Bias: As formally proven in Appendix Section A, the estimator is biased. Therefore, the numerator (centered

PPO

ReMax

|𝔻KL|
|---|

Reference Model

|⨁| |
|---|---|
| | |

𝑟

𝔻KL

Reference Model

𝑜sample

𝑟sample

Policy

𝑞 𝑜 RewardModel

Policy Model

𝐴

𝑞

Model

GAE 𝐴

Reward

𝑜greedy

𝑟greedy

Model

Value Model

𝑣

GRPO

Reinforce++

𝔻KL

RLOO

Reference Model

𝑜1

𝑟1

Reference Model

|𝔻KL|
|---|

𝑜1

- 𝑞1

- 𝑞2

Group Baseline

|⨁|
|---|

𝑞 RewardModel

Policy

𝑜1

𝑟1

𝐴

𝐴 NormalizationBatch

| |𝑜2|
|---|---|
|∙∙∙| |
| |𝑜𝐵|

Policy Model

Model

|∙∙∙|
|---|

∙∙∙ RLOO

∙∙∙

Baseline

𝑜𝐺

𝑟𝐺

𝑞𝐵

Reward Model

𝑟

- Figure 1: The comparison of PPO, ReMax, GRPO, RLOO, and REINFORCE++. REINFORCE++, as shown in the figure, removes the critic model and uses global batch normalization.

REINFORCE++w/ Baseline uses group sampling, similar to GRPO/RLOO, but replaces the local group baseline with a group-mean subtraction followed by global batch normalization.

reward) and the denominator (local standard deviation std(·)) are not independent. The denominator’s value is correlated with the rewards in the small group, introducing a systematic error in the advantage estimate (Mai et al., 2025).

- 2. Practical Instability: In practice, the group size k is small (e.g., 4 or 8). If all sampled responses for a prompt happen to get similar rewards, the local std(·) approaches zero, causing the advantage to explode. It results in high variance and unstable training.
- 3. Task Overfitting: The policy is rewarded for being "better than other samples from the same prompt," not for being "globally good." It can lead to overfitting on simple prompts where it’s easy to generate diverse rewards, while failing to improve on complex prompts.

### 3 Method

Our method addresses the instability of criticfree RLHF by replacing biased local normalization with stable global normalization. We propose two variants for different use cases.

#### 3.1 Global Normalization

The primary REINFORCE++algorithm is designed for general-purpose RLHF. In its k = 1 configuration, it is designed for maximum efficiency and prompt diversity.

It optimizes the PPO objective in Eq. (1) but redefines the advantage Aq,ot. We use the

standard PPO reward formulation, which incorporates the k1-style KL penalty directly into the reward:

Aq,ot = r(o1:T,q) − β ·

T

KL(i) (4)

i=t

where KL(t) = log ππθold(ot|q,o<t)

ref(ot|q,o<t) .

The key innovation is our normalization strategy. Instead of a non-existent (for k = 1) or biased local norm, we use Global Advantage Normalization (Andrychowicz et al., 2020):

Aq,ot − mean(A|A ∈ Dbatch) std(A|A ∈ Dbatch) + ϵ

Anormq,ot =

(5)

This global normalization is the core of REINFORCE++. As the global batch size Dbatch is typically large (e.g., 1024 or more), the mean(·) and std(·) converge to stable constants. It makes the gradient estimator effectively less biased (as N → ∞) and robust to outliers, drastically improving training stability (see Appendix Section A.2). The algorithm for the efficient k = 1 case is detailed in Algorithm Algorithm 1.

#### 3.2 Local Baseline

For more complex tasks, such as multi-step reasoning, sampling multiple responses per prompt (k > 1) can be beneficial (Yue et al., 2025). For this, we introduce REINFORCE++w/ Baseline . This variant combines the benefits of groupsampling with the stability of global normalization. The advantage calculation is a two-step process:

Algorithm 1 REINFORCE++when k = 1 Require: Initial policy model πref, reward

models R, task prompts D

- 1: policy model πθ ← πref
- 2: for step = 1,...,M do
- 3: Sample a batch Dbatch from D
- 4: Update the old policy model πold ← πθ
- 5: Sample output (k = 1) o ∼ πold(· | q) for each question q ∈ Dbatch
- 6: Compute rewards ri for each sampled oi using R
- 7: Compute advantage Aq,ot via Eq. (4)
- 8: Normalize advantages globally across Dbatch to get Anormq,ot via Eq. (5)
- 9: for iteration = 1,...,k do
- 10: Update πθ by maximizing the objective in Eq. (1) using Anormq,ot
- 11: end for
- 12: end for Ensure: πθ

- • Group Mean Subtraction for Reshaping: We first subtract the group mean reward. It serves as a local baseline, reshaping rewards to be robust to different reward scales (e.g., 0/1 vs. −1/1).

A′q,ot = Rq,ot − meangroup(Rq,ot) (6)

- • Global Batch Normalization for Stability: We then normalize this initial advantage using the global batch statistics, not the unstable local group statistics.

A′q,ot − meanbatch(A′) stdbatch(A′) + ϵ

Anormq,ot =

(7)

This combination avoids the bias and instability of local standard deviation in GRPO.

Furthermore, this variant employs a separate KL loss term for regularization. We adopt the k2 estimator. As shown in Appendix Section B.1, the k2 estimator provides a stable, unbiased gradient for the Reverse KL divergence, unlike the k3 estimator (used in GRPO), which is an unstable approximation (Liu et al., 2025a). The final objective is:

L = LPPO(Anorm) − λ · Jk2 as loss(θ) (8) where Jk2 as loss(θ) = E[12(log ππθ

)2].

ref

#### 3.3 Relationship with PPO

REINFORCE++w/ Baseline can be viewed as a simplified and more stable variant of PPO. It is formally equivalent to a PPO agent where: (1) The critic network is removed; (2) The GAE parameters are set to λ = 1 and γ = 1; and (3) A two-step global batch normalization is used as the baseline instead of a learned value function.

#### 3.4 Summary

To address these challenges, we propose REINFORCE++, a method centered on a simple yet powerful idea without the critic model: Global Advantage Normalization. Instead of normalizing within a prompt’s local group, REINFORCE++normalizes the advantage function across the entire global training batch. This approach is:

- • Theoretical Stability: As the global batch size grows (e.g., N = 1024), the batch mean and standard deviation converge to constants, resulting in an effectively unbiased (bias vanishes as N → ∞) and low-variance advantage estimator (see Appendix Section A.2).
- • Training Efficiency: It retains the criticfree architecture, significantly reducing computational and memory overhead compared to PPO.
- • Strong Generalization: Using a global baseline prevents overfitting to specific prompts and encourages a more robust policy.

### 4 Experiments

We evaluate our two algorithms, REINFORCE++and REINFORCE++w/ Baseline , in their respective target domains using the OpenRLHF framework (Hu et al., 2024).

#### 4.1 General RLHF

First, we evaluate the REINFORCE++in its k = 1 configuration, focusing on generaldomain RLHF, where efficiency and prompt diversity are key.

Experimental Setup We use an instructionfollowing policy model (Llama-3-8B-SFT) and refine it using a Bradley-Terry reward model

(Bai et al., 2022) trained on ∼700K human preference pairs. The policy is trained on 20,000 diverse prompts. We compare REINFORCE++(k = 1) with other critic-free methods: GRPO (k = 4), RLOO (k = 4), and ReMax (k = 1 + 1).

Experimental Results As shown in Table 1, the single-sample REINFORCE++ (k = 1) achieves a score of 46.7, statistically tied with the group-sampling GRPO (46.8). However, it produces shorter responses (832 tokens vs. 860), resulting in a higher and more efficient per-token score (0.0561). This result highlights that for general tasks, group sampling (k > 1) is not necessary and may even be suboptimal.

Results Analysis Figure 2 shows the training dynamics. GRPO’s reward rises quickly, but its KL divergence also increases rapidly, suggesting it is "hacking" the reward model. In contrast, REINFORCE++shows a more stable reward increase with a much smaller KL divergence. It highlights the stability of global normalization and its higher "KL-to-reward" conversion efficiency, avoiding the reward hacking seen in local normalization methods, which is often linked to length exploitation (Dubois et al., 2024).

Training Reward with Smooth

KL Divergence with Smooth

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.6

2.2

0.5

2.0

0.4

1.8

0.3

1.6

1.4

0.2

1.2

0.1

1.0

0.0

0 50 100 150 200 250 300

0 50 100 150 200 250 300

REINFORCE++ GRPO

- Figure 2: Comparison of smoothed Training Reward and KL Divergence. REINFORCE++ (k = 1) achieves strong rewards with significantly more stable (lower) KL divergence, avoiding GRPO’s reward-hacking behavior.

#### 4.2 Reasoning Experiments

Next, we evaluate REINFORCE++(groupsample, k > 1) in its target domain: complex reasoning, where group sampling is standard and reward signals are often sparse (e.g., 0/1). We compare it directly with GRPO (k > 1).

- 4.2.1 Long Reasoning Task We use rule-based rewards (RLVR) in mathematical reasoning settings (Guo et al., 2025;

Seed et al., 2025).

Analysis on Small-Scale Datasets To test for overfitting, we trained on only 30 questions from AIME-24 and evaluated on AIME-25. As shown in Table 2, GRPO (local norm) achieves a near-perfect 95.0% on the training set but completely fails on the test set (0.0 Pass@1), demonstrating catastrophic overfitting. In contrast, REINFORCE++with global norm, while scoring lower on the training set (71.0), shows significantly better generalization (2.5 Pass@1, 40.0 Pass@16).

Figure 3 confirms this. GRPO (left) immediately overfits, mastering the training questions in just a few steps. REINFORCE++(right) learns more gradually, enabled by the stable global normalization signal.

Logical Reasoning (K&K Puzzles) We also tested on the Knights and Knaves (K&K) puzzles (Xie et al., 2025, 2024), where difficulty increases with the number of "people." Figure 4 shows that while GRPO is competitive on easy tasks (2-3 people), its performance collapses on harder, OOD tasks (8 people). REINFORCE++is more robust, outperforming GRPO on all tasks with four or more people and achieving a much higher average score (62.1 vs. 55.7).

RL from Zero Setting Finally, we trained a Qwen2.5-Math-Base model from zero on MATH dataset splits (Guo et al., 2025). Table 3 shows that REINFORCE++again achieves better OOD generalization on the more challenging AIME-24 and AMC-23 datasets, while remaining competitive on the in-distribution MATH500 test set.

4.3 Multi-step Reinforcement Learning Finally, to test performance in a complex, multistep environment, we utilize the zero-shot agent setup from (Mai et al., 2025), where a Qwen 2.5 Base 7B model must learn to use Python tools to solve mathematical problems. It is a challenging scenario where group sampling (k > 1) and reward reshaping are crucial (see Appendix Section 5).

Experimental Setup We adhere to the training and evaluation protocols of the ZeroTIR environment (Mai et al., 2025). The backbone model is Qwen 2.5 Base 7B, trained

Score Length Per Token REINFORCE++ (k=1) 46.7 832 0.0561

GRPO (k=4) 46.8 860 0.0544 RLOO (k=4) 44.6 866 0.0515 ReMax (k=1+1) 45.1 805 0.0560

- Table 1: Comparison on Chat-Arena-Hard (Li et al., 2024). The single-sample REINFORCE++(k = 1) achieves a top-tier score while being more token-efficient than the group-sampling GRPO.

AIME-24 (Train) AIME-25 (Test) Pass@N N = 1 N = 1 N = 16 GRPO 95.0 0.0 0.4 REINFORCE++ (k > 1) 71.0 2.5 40.0

- Table 2: Comparison on a small training dataset. GRPO (local norm) overfits completely, while REINFORCE++(global norm) generalizes.

[Figure 1]

[Figure 2]

- Figure 3: Training curves on a small prompt dataset. Left: GRPO (local norm) immediately overfits. Right: REINFORCE++(global norm) learns more stably.

AIME-24 (OOD) AMC-23 (OOD) MATH-500 (ID)

Pass@N N = 8 N = 8 N = 1 GRPO 18.96 59.22 73.00 REINFORCE++ 21.04 60.47 72.00

Table 3: Comparison on RL from Zero. REINFORCE++shows superior OOD performance.

2 3 4 5 6 7 8

Number fo People

20

30

40

50

60

70

80

Pass@1Score

REINFORCE++

GRPO

- Figure 4: Comparison on logic benchmarks. REINFORCE++’s global normalization provides better robustness to increasing task difficulty and OOD challenges (8 people).

using OpenRLHF (Hu et al., 2025a) with datasets from ORZ (Hu et al., 2025b) and DAPO (Yu et al., 2025). We assess performance on benchmarks including AIME 2024 (Jia, 2024), AIME-2025 (math ai, 2025), HMMT FEB-2024/2025, and CMIMC, using the average@32 metric.

Experimental Result As shown in Table 4, REINFORCE++w/ Baseline achieves the highest average accuracy (24.10) across all benchmarks. It significantly outperforms both GRPO (22.58) and the full-critic PPO (21.85). It demonstrates that our stable, critic-free approach is highly effective for complex agentic tasks, sur-

passing even the heavyweight PPO algorithm. The combination of group-mean reshaping, stable global normalization, and the correct ‘k2‘ KL loss proves to be a superior strategy.

### 5 Best Practices

- 5.1 General Principle This paper introduces two algorithms: REIN-

FORCE++and REINFORCE++w/ Baseline [n], with a key focus on selecting the appropriate algorithm based on task conditions.

Empirical evidence from the opensource community suggests that REINFORCE++w/ Baseline [n] is particularly effective for sample filtering or more complex scenarios. For instance, in the multiturn tool-calling setting, a high proportion of void (i.e., non-informative or incorrect) samples can destabilize the training process. In such cases, incorporating a baseline by subtracting the intra-group mean reward significantly improves training stability by effectively filtering out void samples. Moreover, this automatic reward reshaping reduces the need for designing complex reward structures. For example, REINFORCE++w/ Baseline [n] supports both 0/1 and -1/1 reward schemes, while REINFORCE++performs best with symmetric rewards, such as -1/1 in RLVR tasks.

In contrast, for general-domain tasks where prompt diversity and efficiency are paramount, or for tasks where obtaining multiple reward signals for distinct responses is challenging—such as training with Process-Supervised Reward Models (PRMs) or online realtime sampling—we recommend using REINFORCE++ (k = 1). Our experiments (Section 4.1) show that it provides superior OOD generalization in these settings.

#### 5.2 Third-Party Validation

The core principle of REINFORCE++, global advantage normalization, has been independently validated and adopted in several subsequent large-scale reasoning systems, confirming its stability and effectiveness.

LitePPO (Liu et al., 2025c), which effectively combines REINFORCE++w/ Baseline with a token-level loss, conducted experiments demonstrating

that the global standard deviation is superior to the local standard deviation used in GRPO. Their results show more stable training and better generalization, aligning perfectly with our findings.

ScaleRL (Khatri et al., 2025) performed a detailed ablation study on advantage estimation methods in large-scale (16,000 GPU-hour) experiments. They directly compared batchlevel normalization in REINFORCE++with prompt-level normalization in GRPO. Their findings concluded that batch-level normalization was "slightly superior in both compute efficiency and final performance," confirming the advantages of global normalization at scale.

DLER (Liu et al., 2025b) found that when using truncation to control the output length of an LLM, batch-wise (global) normalization remains stable while group-wise (local) normalization shows declining accuracy. The study provides further evidence that global normalization is more robust to variations in training conditions and reward landscapes.

### 6 Conclusion

In this paper, we present REINFORCE++, a critic-free RLHF framework designed to enhance training stability by addressing fundamental flaws in existing methods. We identified that prior critic-free algorithms, such as GRPO, rely on prompt-level (local) advantage normalization, which we prove to be theoretically biased (Appendix Section A) and demonstrate to be practically unstable.

Our solution, Global Advantage Normalization, normalizes advantages across the entire batch, providing a stable and effectively unbiased estimator (whose bias vanishes as batch size increases). We proposed two variants tailored for different needs:

- 1. REINFORCE++: For general-domain RLHF, we show the single-sample (k = 1) approach is highly efficient and achieves superior OOD generalization. We also show that its k > 1 application is robust for reasoning tasks (Section 4.2).
- 2. REINFORCE++w/ Baseline : For complex reasoning and agent tasks, this variant

Algorithm AIME 24 AIME 25 HMMT 2025 HMMT 2024 CMIMC Avg GRPO (local norm) 31.66 21.87 16.97 17.70 24.68 22.58 PPO (critic-based) 30.20 21.66 15.00 18.43 23.95 21.85 RF++-Baseline (global norm) 30.83 27.18 17.91 18.95 25.62 24.10

Table 4: Performance Comparison on complex tool-use benchmarks (average@32). REINFORCE++w/ Baseline outperforms both GRPO (local norm) and PPO (critic-based) models.

combines group-mean reshaping with stable global normalization and a theoretically sound k2 estimator for KL.

Our empirical results, supported by independent validation from multiple thirdparty systems, confirmed this specialized approach. REINFORCE++demonstrated stateof-the-art generalization in general RLHF. REINFORCE++w/ Baseline showed dramatic improvements in stability, preventing overfitting in low-data regimes and outperforming both GRPO and full-critic PPO in complex, long-horizon tool-use tasks. This work demonstrates that a theoretically sound and stable approach without the critic model can be more efficient and effective than traditional PPO.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet stün, and Sara Hooker. 2024. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint.

Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphael Marinier, Léonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, and 1 others. 2020. What matters in on-policy reinforcement learning? a large-scale empirical study. arXiv preprint arXiv:2006.05990.

AI Anthropic. 2023. Introducing claude.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, and 1 others. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. 2024. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Jian Hu, Xibin Wu, Wei Shen, Jason Klein Liu, Zilin Zhu, Weixun Wang, Songlin Jiang, Haoran Wang, Hao Chen, Bin Chen, Weikai Fang, Xianyu, Yu Cao, Haotian Xu, and Yiming Liu. 2025a. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. Preprint, arXiv:2405.11143.

Jian Hu, Xibin Wu, Zilin Zhu, Weixun Wang, Dehao Zhang, Yu Cao, and 1 others. 2024. Openrlhf: An easy-to-use, scalable and highperformance rlhf framework. arXiv preprint arXiv:2405.11143.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. 2025b. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. Preprint, arXiv:2503.24290.

Maxwell Jia. 2024. Aime 2024 dataset card. https://huggingface.co/datasets/ Maxwell-Jia/AIME_2024. Version 2024-03.

Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S Dhillon, David Brandfonbrener, and Rishabh Agarwal. 2025. The art of scaling reinforcement learning compute for llms. arXiv preprint arXiv:2510.13786.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. 2024. From crowdsourced data to high-quality benchmarks: Arenahard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939.

Ziniu Li, Tian Xu, Yushun Zhang, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. 2023. Remax: A simple, effective, and efficient method for aligning large language models. arXiv preprint arXiv:2310.10505.

Kezhao Liu, Jason Klein Liu, Mingtao Chen, and Yiming Liu. 2025a. Rethinking kl regularization in rlhf: From value estimation to gradient optimization. arXiv preprint arXiv:2510.01555.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Mingjie Liu, Min-Hung Chen, Hongxu Yin, YuChiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, and 1 others. 2025b. Dler: Doing length penalty right-incentivizing more intelligence per token via reinforcement learning. arXiv preprint arXiv:2510.15110.

Zihe Liu, Jiashun Liu, Yancheng He, Weixun Wang, Jiaheng Liu, Ling Pan, Xinyu Hu, Shaopan Xiong, Ju Huang, Jian Hu, and 1 others. 2025c. Part i: Tricks or traps? a deep dive into rl for llm reasoning. arXiv preprint arXiv:2508.08221.

Xinji Mai, Haotian Xu, Xing W, Weinong Wang, Jian Hu, Yingying Zhang, and Wenqiang Zhang. 2025. Agent rl scaling law: Agent rl with spontaneous code execution for mathematical problem solving. Preprint, arXiv:2505.07773.

math ai. 2025. Aime 2025 dataset card. https: //huggingface.co/datasets/math-ai/aime25. Version 2024-03.

OpenAI. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and

Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. 2018. High-dimensional continuous control using generalized advantage estimation. Preprint, arXiv:1506.02438.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, and 1 others. 2025. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Yu Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Wei Shen, Guanlin Liu, Zheng Wu, Ruofei Zhu, Qingping Yang, Chao Xin, Yu Yue, and Lin Yan. 2025. Exploring data scaling trends and effects in reinforcement learning from human feedback. arXiv preprint arXiv:2503.22230.

Wei Shen and Chuheng Zhang. 2024. Policy filtration in rlhf to fine-tune llm for code generation. arXiv preprint arXiv:2409.06957.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, and 1 others. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Sai Vemprala, Rogerio Bonatti, Arthur Bucker, and Ashish Kapoor. 2023. Chatgpt for robotics: Design principles and model abilities. Microsoft Auton. Syst. Robot. Res, 2:20.

Chulin Xie, Yangsibo Huang, Chiyuan Zhang, Da Yu, Xinyun Chen, Bill Yuchen Lin, Bo Li, Badih Ghazi, and Ravi Kumar. 2024. On memorization of large language models in logical reasoning. arXiv preprint arXiv:2410.23123.

Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. 2025. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin

Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and 16 others. 2025. Dapo: An opensource llm reinforcement learning system at scale. Preprint, arXiv:2503.14476.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, and 1 others. 2025. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118.

## Appendix

### A Proof: The GRPO Advantage Estimator is Biased

- A.1 Assumptions and Settings We observe N rewards ri for a prompt, and assume the true baseline is θ, such that

ri = θ + ϵi, ϵi ∼ N(0,σ2), i = 1,...,N, with all advantage value ϵi independent. Define

N

1 N

ϵ¯ =

ϵj, D =

j=1

N

ϵi − ϵ¯ D

1 N

(ϵj − ϵ¯)2, Ai =

.

j=1

We will prove the following: Theorem 1. For any finite N ≥ 2, the advantage estimator Ai is biased:

E[Ai | ϵi] ̸= ϵi.

Proof. We can prove our findings with the following steps: Step 1: Numerator Bias We rewrite the numerator:

1 N

1 N j̸=i

ϵi − ϵ¯ = 1 −

ϵi −

ϵj.

Since the ϵj for j ̸= i are zero-mean and independent of ϵi,

1 N

E[ϵi − ϵ¯ | ϵi] = 1 −

ϵi.

Step 2: Denominator Depends on ϵi (a) Compute E[D2 | ϵi]: By definition,

N

1 N

1 N

D2 =

(ϵj − ϵ¯)2 =

j=1

N

ϵ2j − ϵ¯2.

j=1

Since

 ϵi +

 ,

1 N

ϵj

ϵ¯ =

j̸=i

and conditioning on ϵi keeps the ϵj, j ̸= i, i.i.d. N(0,σ2), we obtain:

 

  = ϵ2i + (N − 1)σ2,

N

ϵ2j | ϵi

E

j=1

 

 

 ϵi +

 

2

1 N2

E[¯ϵ2 | ϵi] =

E

ϵi

ϵj

j̸=i





 

 

 

 

 

2 

1 N2

ϵ2i + 2ϵi · E

+E

ϵj

=

ϵj

 

 

j̸=i

j̸=i

0

ϵ2i + (N − 1)σ2 N2

=

.

Subtracting:

ϵ2i + (N − 1)σ2 N2

1 N

E[D2 | ϵi] =

(ϵ2i + (N − 1)σ2) −

(N − 1)2 N2

N − 1 N2 β

ϵ2i = α + βϵ2i. (9)

σ2

=

+

α

(b) g(ϵi) is Not Constant: Let g(ϵi) = E[1/D | ϵi] and µ(ϵi) = E[D2 | ϵi] = α + βϵ2i. Using Taylor expansion of f(x) = 1/√x around x0 = µ(ϵi):

1 √x0 −

f(x) =

Taking conditional expectation:

x − x0 x30/2

- 1

- 2

(x − x0)2 x50/2

3 8

+ O((x − x0)3)

+

1 √x ≈ f(µ(ϵi)) + f′(µ(ϵi)) · E[D2 − µ(ϵi) | ϵi] +

1 √

| ϵi = E f(D2) | ϵi , where f(x) =

g(ϵi) = E[1/D | ϵi] = E

D2

f′′(µ(ϵi)) 2 · E[(D2 − µ(ϵi))2 | ϵi]

3 8µ(ϵi)5/2 · Var(D2 | ϵi)

1 µ(ϵi) −

1 2µ(ϵi)3/2 · E[D2 − µ(ϵi) | ϵi]

+

=

= 0

Var(D2 | ϵi) µ(ϵi)5/2

1 µ(ϵi)

3 8 ·

=

+

Since µ(ϵi) = α + βϵ2i with β > 0, the first term alone shows that g(ϵi) depends on ϵ2i and hence is not constant.

#### Step 3: Putting It Together Decomposing Ai,

  1

  ·

ϵi − ϵ¯ D

1 N

ϵi D −

1 D

= 1 −

ϵj

.

Ai =

N j̸=i

For fixed ϵi, the conditional distribution of j̸=i ϵj is symmetric about zero, while 1/D is always positive. Thus:

  ·

  = 0.

 

 −1

1 D

E

ϵj

ϵi

N j̸=i

It follows that

1 N

1 D | ϵi = 1 −

1 N

E[Ai | ϵi] = 1 −

ϵi · E

ϵi · g(ϵi). Step 4: Concluding the Bias If Ai were unbiased, we would have:

N N − 1

1 N

g(ϵi) ≡ 1 ⇒ g(ϵi) ≡

1 −

,

which contradicts Step 2. Therefore, for any finite N ≥ 2,

E[Ai | ϵi] ̸= ϵi. Hence Ai is a biased estimator.

| |
|---|

#### A.2 Why Use Global Batch Normalization?

We observe that as N → ∞, the denominator of the estimator converges to the constant σ, and the bias in the numerator vanishes. This insight underlies our decision to adopt global batch normalization in both REINFORCE++ and REINFORCE++-Baseline, as the global batch size (Nglobal) is typically much larger (e.g., 1024) than the group batch size (Ngroup, e.g., 4 or 8) in practice. As Nglobal → ∞, the sample statistics µb and σb converge to true constants, making the estimator effectively unbiased.

### B Algorithm Details

- B.1 KL Penalty Design In non-critic algorithms like REINFORCE++-Baseline, a KL divergence term is often added as a

separate loss to constrain the policy πθ from deviating too far from a reference policy πref. In practice, this expectation is estimated using samples, and three common estimators are used.

Given the importance ratio δ(y) = πref(y|·)/πθ(y|·) (where πθ is the sampling policy and πref is the reference):

- • k1(y) = −log δ(y) = log ππθ(y|·)

ref(y|·)

- • k2(y) = 21(log δ(y))2 = 21 log ππref(y|·)

θ(y|·)

2

- • k3(y) = δ(y) − 1 − log δ(y)

We analyze which of these (k1, k2, or k3) is the correct choice when used as a separate loss term.

Theoretical Reverse KL Gradient We are training with samples from πθ, so we are approximating the **Reverse KL (RKL)**, DKL(πθ||πref). The practical policy gradient for RKL is (more details are in (Liu et al., 2025a)):

πθ(y|x) πref(y|x) ∇θ log πθ(y|x)

∇θJRKL(θ) = Ey∼πθ(·|x) log

- k1 Analysis We can **exclude k1** as a loss term. Its gradient does not depend on πref and thus provides no constraining effect. The loss Jk1 = Ey∼πθ[log πθ − log πref] differentiates to

∇θJk1 = Ey∼πθ[∇θ log πθ], as the πref term vanishes. (Note: k1 *is* used inside the reward for REINFORCE++, but not as a separate loss term).

- k2 Analysis The k2 estimator provides a gradient that is **equivalent to the theoretical Reverse KL gradient**.

)2 = (log ππref

The k2 loss is (noting that (log ππθ

)2):

ref

θ

Jk2as loss(θ) = Ey∼πθ(·|x)

- 1

- 2

Differentiating the term inside the expectation gives:

πθ(y|x) πref(y|x)

log

2

∇θ

- 1

- 2

πθ πref

log

2

πθ πref · ∇θ log

πθ πref

= log

πθ πref · ∇θ(log πθ − log πref)

= log

πθ πref ∇θ log πθ

= log

When we take the expectation over y ∼ πθ, the gradient of the k2 loss becomes:

πθ(y|x) πref(y|x) ∇θ log πθ(y|x)

Ey∼πθ log

This **perfectly matches** the practical RKL gradient. Therefore, k2 is the theoretically correct and stable choice.

- k3 Analysis The k3 gradient is problematic. It is theoretically an estimator for the **Forward KL** (KL(πref||πθ)), not the Reverse KL. This is a mismatch, as we are sampling from πθ, but the Forward KL gradient requires sampling from πref.

This method, used in GRPO, suffers from two major issues:

- • Extremely High Variance: When πθ(y) becomes tiny for a sample y where πref(y) is moderate, the importance weight ππref(y)

θ(y) in the gradient estimator explodes, leading to "infinite variance".

- • Numerical Instability: It requires computing p/q via exp(log p − log q), which is prone to overflow.

This explains why methods using k3 require frequent resetting of πref to prevent the policies from diverging and the k3 estimator from becoming unstable.

Conclusion Based on this analysis, the **k2 estimator is the optimal choice** for a KL loss term as it correctly and stably estimates the Reverse KL gradient. We therefore use the k2 estimator in our REINFORCE++-Baseline algorithm.

#### B.2 Implementation Tricks

Token-level Advantage The advantage Anormq,ot is computed at the token level. For t < T (where T is the sequence length), the advantage is set to 0. For the final token t = T, the

advantage is the normalized reward Anormq,oT . This is standard practice in RLHF for LLMs.

Batch Construction For REINFORCE++ (k = 1), a global batch of size N consists of N different prompts. For REINFORCE++-Baseline (k = 4), a global batch of size N = 1024 might consist of 1024/4 = 256 unique prompts. The global mean/std are computed over all N samples.

Mini-Batch Updates To enhance training efficiency, we implement mini-batch updates with the following characteristics:

- • Batch Processing: Data is processed in smaller, manageable chunks rather than full-batch updates.
- • Multiple Updates: Each mini-batch allows for multiple parameter updates, improving convergence rates.
- • Stochastic Optimization: Introduces beneficial randomness for better generalization.

Reward Normalization and Clipping We implement comprehensive reward processing to stabilize training:

- • Normalization: Standardizes rewards using z-score normalization (our global normalization) to mitigate outliers.
- • Clipping: Constrains reward values within predefined bounds to avoid instability.
- • Scaling: Applies appropriate scaling factors for numerical stability during updates.

### C Acknowledgements

- • Jian Hu: Conceived the ideas, implemented the REINFORCE++ and

REINFORCE++w/ Baseline algorithms, and contributed to the theoretical proof of the GRPO advantage estimator.

- • Jason Klein Liu: Implemented the experimental framework, fine-tuned hyperparameters, wrote the manuscript, and provided GPU resources.
- • Haotian Xu: Conducted comparative experiments between REINFORCE++w/ Baseline and complex tool-calling as well as agent-based scenarios.
- • Wei Shen: Supervised the overall project, designed the main experiments, and led the paper writing.

