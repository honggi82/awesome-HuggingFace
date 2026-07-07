# arXiv:2510.03222v2[cs.LG]7Nov2025

[Figure 1]

2025-11-10

### Low-probability Tokens Sustain Exploration in Reinforcement Learning with Verifiable Reward

Guanhua Huang1,∗, Tingqiang Xu1,2,∗†, Mingze Wang1,3,†, Qi Yi1, Xue Gong1, Siheng Li1,4,†, Ruibin Xiong1, Kejiao Li1, Yuhao Jiang1, Bo Zhou1‡

1LLM Department, Tencent 2Tsinghua University 3Peking University 4The Chinese University of Hong Kong

##### Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has propelled Large Language Models in complex reasoning, yet its scalability is often hindered by a training bottleneck where performance plateaus as policy entropy collapses, signaling a loss of exploration. Previous methods typically address this by maintaining high policy entropy, yet the precise mechanisms that govern meaningful exploration have remained underexplored. Our analysis suggests that an unselective focus on entropy risks amplifying irrelevant tokens and destabilizing training. This paper investigates the exploration dynamics within RLVR and identifies a key issue: the gradual elimination of valuable low-probability exploratory tokens, which we term reasoning sparks. We find that while abundant in pre-trained models, these sparks are systematically extinguished during RLVR due to over-penalization, leading to a degeneracy in exploration. To address this, we introduce Low-probability Regularization (Lp-Reg). Its core mechanism regularizes the policy towards a heuristic proxy distribution. This proxy is constructed by filtering out presumed noise tokens and re-normalizing the distribution over the remaining candidates. The result is a less-noisy proxy where the probability of reasoning sparks is amplified, which then serves as a soft regularization target to shield these valuable tokens from elimination via KL divergence. Experiments show that Lp-Reg enables stable on-policy RL, sustaining continuous scaling across 3,000 training steps and 81,204 GPU-hours, where baseline entropy-control methods collapse. This sustained exploration leads to state-of-the-art performance, achieving a 60.17% average accuracy on five math benchmarks, an improvement of 2.66% over prior methods. Code is available at https://github.com/CarlanLark/Lp-Reg.

##### 1 Introduction

The advent of large reasoning models has reshaped the trajectory of artificial intelligence, with paradigmatic examples including OpenAI O1 (OpenAI et al., 2024) and DeepSeek-R1 (DeepSeek-AI et al., 2025). A central technique underpinning these systems is reinforcement learning with verifiable reward (RLVR), which assigns reward to verifiable solutions through rule-based verification. These models generate extended chain-of-thought (CoT) reasoning (Wei et al., 2023) to solve challenging problems in domains like mathematical olympiads (He et al.,

- 2024b). However, a notable bottleneck emerges during RL training that limits its scalability, frequently culminating in a performance plateau and subsequent collapse. This failure is consistently accompanied by a rapid decay in policy entropy, indicating a severe loss of exploration capacity (Yu et al., 2025; Cui et al., 2025; Wang et al., 2025).

Previous approaches have recognized this declining exploration, attempting to address it through various entropy control mechanisms. Methods such as adaptive entropy regularization (He et al., 2025), high entropy change blocking (Cui et al., 2025), or selective token updates (Wang et al., 2025) aim to maintain higher entropy as a proxy for exploration. However, relying on overall entropy can be an indirect and imprecise tool. An indiscriminate focus on maximizing randomness risks amplifying noise and destabilizing training (Omer¨ Veysel C¸a˘gatan & Akg¨un,

- 2025), suggesting a deeper issue beyond simply the quantity of randomness.

Our analysis suggests the performance bottleneck may stem from the systematic elimination of valuable lowprobability exploratory tokens. We term these tokens Reasoning Sparks; they include words like “wait”, “however”, or “perhaps”, which often serve as logical connectives or expressions of uncertainty that naturally initiate diverse reasoning pathways (Figure 1a). As the aggregated violin plots in Figure 1c show, standard GRPO training suppresses the low-probability sampling of these important exploratory tokens, causing the suppression of Reasoning Sparks. Furthermore, we find that indiscriminately boosting randomness amplifies the low-probability sampling of irrelevant tokens (e.g., “cost”, “fine”), which are semantically out of context for the mathematical reasoning task. We refer to the low-probability appearance of these irrelevant tokens as noise. This amplification leads to an even faster performance collapse than the baseline, as shown in Figure 1b.

∗ Equal contribution. † Work completed during an internship at Tencent. ‡ Correspondence to Bo Zhou: chaysezhou@tencent.com.

𝑝 = 0.97

Answer : \boxed{2} , …, Answer: \boxed{3}

[Figure 2]

###### ✖

… The answer is 2.

|𝑝 = 0.03 Wait<br><br>Reasoning Spark<br><br>[Figure 3]|
|---|

[Figure 4]

###### ✔

(a) Reasoning Sparks: low-probability exploratory tokens that initiates a new reasoning path.

0.65

GRPO

%Avg.TestAccuracy()

GRPO + Entropy Loss

0.60

Lp-Reg

0.55

0.50

0.45

0 200 400 600 800 1000

Training Steps

0.6

GRPO

GRPO + Entropy Loss

0.5

Lp-Reg

TrainEntropy

0.4

0.3

0.2

0.1

0.0

0 200 400 600 800 1000

Training Steps

(b) Training Dynamics

hard to sample!

(c) Aggregated distribution of observed sampling probabilities for a class of meaningful exploratory tokens (e.g., “wait”, “however”). “n” represents sampling number.

GRPO GRPO+Entropy loss Lp-Reg

1.00

n=192 n=192 n=192 n=324 n=2092 n=230 n=334 n=974 n=240

Probability

0.75

0.50

0.25

0.00

1 128 256

Training Steps

(d) Aggregated distribution of observed sampling probabilities for a class of irrelevant tokens (e.g., “cost”, “fine”). “n” represents sampling number.

- Figure 1: Selectively preserving low-probability tokens is key to overcoming performance plateaus in reasoning RL. (a) An illustration of a reasoning spark. (b) Standard GRPO training reaches a performance plateau and collapses, accompanied by decaying entropy. An indiscriminate entropy bonus (GRPO + Entropy Loss) leads to an even faster collapse. (c) We reveal the cause: GRPO systematically suppresses the low-probability sampling of important exploratory tokens (like “wait”), and forces these tokens’ sampling distributions to collapse towards high probabilities. Entropy Loss fails to fix this. In contrast, our method, Lp-Reg, successfully preserves a healthy, wide distribution, sustaining exploration. (d) The failure of entropy bonuses is explained by amplifying the lowprobability sampling of irrelevant tokens, creating noise, and thereby degrading exploration quality. The aggregated statistics in (c) and (d) demonstrate a systemic effect beyond single-token instances. Detailed plots for individual tokens are available in Appendix B.1.

These findings present a central challenge: a successful exploration strategy should protect valuable reasoning sparks without simultaneously amplifying the destructive effects of irrelevant noise. To address this challenge, we introduce Low-probability Regularization (Lp-Reg). The primary goal of Lp-Reg is to preserve valuable low-probability tokens via regularization. To avoid amplifying noise, the method leverages a key observation: within the low-probability range, meaningful exploratory tokens (like “wait”) consistently exhibit a higher average probability than irrelevant noise (like “cost”) in the immediate next-token prediction. Based on this statistical distinction, Lp-Reg first discards low-probability tokens presumed to be noise using a probability threshold. It then redistributes the probability mass from these discarded tokens among the remaining candidates. This process constructs a less-noisy proxy distribution where valuable low-probability tokens are preserved and their relative probabilities amplified. Finally, Lp-Reg penalizes the deviation of the original policy from this proxy using a forward KL divergence, which selectively protects the low-probability tokens that were preserved in the less-noisy proxy distribution.

Our experimental evaluation demonstrates the effectiveness of Lp-Reg. Our method enables stable on-policy training for around 3,000 steps over 81,204 GPU-hours, a regime where many entropy-control methods have collapsed, resulting in better performance. On five widely used math benchmarks, this results in a 60.17% average accuracy on Qwen3-14B-Base, improving upon prior methods by 2.66%. Our contributions are summarized as follows:

- • In contrast to prior work focusing on overall policy entropy, we identify the suppression of reasoning sparks as a key issue and provide evidence that their preservation is crucial for sustained performance.
- • We introduce Low-probability Regularization (Lp-Reg), a method that creates a more stable exploratory environment by filtering out presumed meaningless noise to protect the remaining low-probability tokens.
- • We demonstrate through extensive experiments that Lp-Reg achieves state-of-the-art performance, while also enabling stable on-policy continuous scaling over extended periods where baselines collapse.
- • We provide a comprehensive analysis showing that our approach of filtering presumed meaningless noise yields superior results compared to indiscriminate entropy-control methods.

##### 2 Related Work

Reinforcement learning for LLMs Recently, reinforcement learning has become the dominant framework for enhancing the reasoning abilities of large language models (LLMs) (OpenAI et al., 2024; DeepSeek-AI et al., 2025). By leveraging automatic checkers or symbolic verification, reinforcement learning with verifiable rewards (RLVR) achieved further breakthroughs in improving the reasoning capability of LLMs (Shao et al., 2024a; Yang et al., 2025; Team et al., 2025). Based on RLVR and GRPO (Shao et al., 2024a), subsequent methods such as DAPO Yu et al. (2025), VAPO (Yue et al., 2025), and other policy optimization variants (Zhao et al., 2025; Cui et al., 2025; Zheng et al., 2025) have been proposed to improve the stability, efficiency, and scalability of RL for reasoning models.

Entropy collapse in RL training A recurring difficulty in training reasoning models with RL is the rapid collapse of policy entropy during the early stages of training. This phenomenon, which reflects excessive exploitation and insufficient exploration, has been widely recognized as a bottleneck for scaling RL in reasoning models. To mitigate collapse, researchers have explored several directions, including selectively regularizing updates at high-entropy “forking” tokens (Wang et al., 2025), amplifying advantages at exploratory positions (Cheng et al., 2025), modifying clipping strategies (Yu et al., 2025; Zhao et al., 2025; Cui et al., 2025; Zheng et al., 2025), or doing weight clipping (MiniMax et al., 2025; Su et al., 2025). While these methods primarily operate by monitoring policy entropy, which is correlational rather than causal to exploration, our analysis delves directly into the next-token prediction distribution. This allows for a more semantically grounded and causally-informed investigation of the probabilities of individual candidates and their role in exploration dynamics.

Intrinsic confidence of LLMs As the capabilities of Large Language Models (LLMs) have rapidly advanced, they have demonstrated an increasingly strong and reliable sense of intrinsic confidence (Saurav et al., 2022; Loka

- et al., 2024; Amir et al., 2025). Research investigates how these intrinsic confidence signals, often reflected in the next-token prediction distribution, can guide complex reasoning and exploration (Amirhosein et al., 2025; Xuezhi & Denny, 2024; Xuandong et al., 2025). Studies have shown that tokens with higher relative probabilities in the next-token prediction are often more contextually appropriate than their lower-probability counterparts (Nguyen
- et al., 2025; Xu et al., 2025; Fu et al., 2025). Building on this, some work has explored entropy minimization, which sharpens the model’s confidence distribution. This approach can improve inference performance by encouraging the model to commit to consistent and high-confidence solution paths (Gao et al., 2025; Agarwal et al., 2025). Our work builds upon a similar insight, leveraging the model’s intrinsic confidence to distinguish between valuable reasoning sparks and irrelevant noise within this low-probability range.

##### 3 Preliminaries

- 3.1 Reinforcement Learning with Verifiable Rewards Reinforcement learning (RL) has played a critical role in LLMs (Murphy, 2024). Formally,

JRL(θ) = E(q,a)∼D,o∼πθ(·|q) r(o, a) , (1)

where r(o, a) denotes the reward assigned to an output o given a reference answer a. In reinforcement learning with verifiable rewards (RLVR), this reward is computed through rule-based functions, such as Math-Verify1. Recent studies have demonstrated that large-scale RLVR encourages models to perform more deliberative reasoning by producing extended chains of thought prior to the final prediction, thereby substantially improving their capacity to solve complex problems (DeepSeek-AI et al., 2025). In practice, Eq. 1 is typically optimized using policy gradient methods, such as Proximal Policy Optimization (PPO) (Schulman et al., 2017) and Group Relative Policy Optimization (GRPO) (Shao et al., 2024b).

- 3.2 Group-Relative Policy Optimization

GRPO is a representative actor-only policy gradient method for optimizing LLMs. It directly estimates the advantage of each token by leveraging multiple samples drawn from the same prompt. Formally, the advantage is defined as

R(oi) − mean(G) std(G)

, (2)

Ai,t =

where {o1, . . . , oG} are independent outputs sampled from the same prompt, with group size G, G = {R(o1), . . . , R(oG)} denotes their associated rewards, and R(oi) is the reward of output oi. In this formulation, Ai,t represents the advantage of the t-th token in oi. The policy is then optimized on the basis of these advantages using the PPO

1https://github.com/huggingface/Math-Verify

surrogate objective:

JGRPO(θ) =E(q,a)∼D,{o

i}iG=1∼πθold(·|q)

(3)

|oi|

G

1 G

1 |oi|

#### ∑

#### ∑

min ri,t Ai,t, clip ri,t,1 − ϵ,1 + ϵ Ai,t − β DKL (πθ ∥ πref) ,

t=1

i=1

where β controls the strength of KL regularization between the current policy πθ and the reference policy πref. The probability ratio

πθ oi,t | q, oi,<t πθold oi,t | q, oi,<t

ri,t =

(4)

serves as the importance sampling weight for off-policy training, where πθold denotes the behavior policy. The hyperparameter ϵ specifies the clipping ratio, which constrains the updated policy from deviating excessively from

the behavior policy, thereby ensuring stability during optimization.

##### 4 Low-probability Regularization

To address the premature elimination of valuable reasoning sparks, we propose a regularization method termed Lowprobability Regularization (Lp-Reg). This method is designed to be integrated into policy gradient algorithms to create a more stable exploratory environment. The central idea is to leverage the model’s own predictive distribution to construct a less-noisy proxy for regularization, preserving low-probability tokens.

###### 4.1 Proxy Distribution πproxy

The foundation of Lp-Reg is the construction of a proxy distribution, which represents a filtered variant of the current policy πθ. It is constructed in two steps:

###### (a) Original Policy

(b) Renormalized Policy proxy

0.3

###### Probability

Renormalized sum = 1

0.2

0.1

= 0.02

0.0

0 2 4 6 8 10 12 14 16 18

0 2 4 6 8 10 12 14 16 18

Token Index

Token Index

- Figure 2: An example of probability renormalization. πproxy assigns zero probability to tokens with πθ ≤ τ and renormalizes the probability mass to tokens with πθ > τ.

- 1. Filtering Noise Tokens: We first filter out a set of low-confidence tokens, which are presumed to be noise,

defined as those whose probability πθ(o|·) is under a threshold τ. This threshold controls the filtering strategy, for which we explore two primary choices:

- • Fixed threshold: A simple approach where τ is a constant hyperparameter, e.g., τ = 0.02.
- • Min-p threshold: Following (Nguyen et al., 2025), τ is defined relative to the peak probability: τ =

κ · maxo′∈V πθ(o′|·), where κ ∈ (0,1) is a hyperparameter. This makes the filtering adaptive to the distribution’s sharpness.

Our primary experiments employ the min-p strategy for its adaptiveness, though fixed thresholds are also shown to be effective in our ablation studies.

- 2. Probability Renormalization: As shown in Figure 2, the proxy distribution πproxy assigns zero probability to tokens filtered out in the previous step and renormalizes the probability mass across the remaining tokens:

πproxy(o|·) =

πθ(o|·)

∑o′ s.t. πθ(o′|·)>τ πθ(o′|·) if πθ(o|·) > τ 0 otherwise

. (5)

This process effectively treats tokens with low relative probabilities as potential noise, while preserving all others to form a high-confidence reference.

###### 4.2 Low-probability Regularization Objective

The Low-probability Regularization (Lp-Reg) penalty is integrated into the GRPO framework as a selective regularization term. The final objective function is:

|oi|

G

1 ∑iG=1 |oi|

#### ∑

#### ∑

clip(ri,t(θ),0,U) · Ai,t

JLp-Reg(θ) =EB∼D,(q,a)∼B,{o

i}iG=1∼πθold(·|q)

t=1

i=1

− β · I πθ(oi,t|q, oi,<t) < δρB

∧ Ai,t < 0 neg. sample

∧ πproxy(oi,t|q, oi,<t) > 0

not noise

low probability

(6)

· DKL (πproxy(·|q, oi,<t)∥πθ(·|q, oi,<t))

The first term is the policy gradient objective from GRPO. We modify its clipping by removing the lower bound to avoid clipping low-probability exploratory actions and adding a large upper bound U for numerical stability.

The second term is the Lp-Reg penalty, designed to protect reasoning sparks. It is activated by the indicator function I[·] only for tokens that satisfy three conditions simultaneously: first, their sampling probability πθ is below a dynamic low-percentile threshold δρB, which is calculated as the lowest ρ-th percentile of the sampling probabilities of all tokens within the current training batch B; second, their probability in the proxy distribution πproxy is greater than zero, means oi,t is not a noise token; and third, the token receives a negative advantage signal (Ai,t < 0). This final condition ensures the regularization applies exclusively to tokens receiving a negative learning signal, preventing their potential over-penalization while leaving updates from positive experiences unaffected. We ablate all three conditions in Section 5.3.

We use the forward KL divergence, DKL(πproxy∥πθ) as the regularization function. It imposes a significant penalty when πθ(o|·) approaches zero for a token o with non-zero probability in πproxy, providing a targeted penalty against token elimination without forcing the policy to strictly match the heuristic proxy distribution.

##### 5 Experiments

- 5.1 Experimental Setup

- 5.1.1 Baselines

We compare Lp-Reg against a suite of strong baselines, including a foundational algorithm and several state-of-theart methods designed to enhance exploration through entropy control. Our primary baseline is GRPO (Shao et al., 2024a), a value-free policy optimization algorithm that employs group-relative advantage estimation, making it a common choice for RLVR. To represent classical entropy regularization methods, we implement GRPO + Entropy Loss, which directly incorporates the principles of Maximum Entropy RL by adding a policy entropy bonus to the GRPO objective function. We also compare against several advanced methods: Clip-Higher (Yu et al., 2025), a core component of DAPO that encourages higher entropy by using an asymmetric clipping range in the PPO objective; Selective High-Entropy Training (80/20) (Wang et al., 2025), a method that restricts policy gradient updates to only the top 20% of tokens with the highest generation entropy; KL-Cov (Cui et al., 2025), which prevents entropy collapse by applying a selective KL-divergence penalty to tokens with the highest covariance between their log probabilities and advantages; and GSPO (Zheng et al., 2025), which modifies the clipping mechanism to operate at the sequence level to promote higher training entropy.

- 5.1.2 Training Settings

All experiments are conducted within the verl (Sheng et al., 2024) framework to ensure a standardized and fair comparison. To mitigate unfair comparisons arising from different convergence speeds during the early stages of training, we trained models until their performance saturated. This approach ensures a more stable and equitable evaluation point. Consequently, our comparisons are based on approximately 1,000 training steps for the Qwen314B-Base model and 800 for the Qwen2.5-32B model. Each training requires about 8,000 GPU hours on 32 NVIDIA H20 GPUs for the 14B model and 16,000 GPU hours on 64 NVIDIA H20 GPUs for the 32B model. However, for models that experienced a training collapse, which we define as a performance drop greater than 10% in accuracy, we implemented early stopping to conserve computational resources. To assess whether low-probability tokens sustain exploration in RLVR, we further trained the Qwen2.5-32B model for 3,000 steps over 81,204 GPU-hours with our Lp-Reg and evaluated its training stability.

For the reinforcement learning from verifier rewards (RLVR) phase, models are trained on the Dapo-Math-17K (Yu et al., 2025) dataset with a maximum response length of 8,192 tokens. We use a global batch size of 256. For

off-policy methods, we use a mini-batch size of 32, resulting in 8 gradient updates per rollout. To ensure a fair comparison, a “step” in our experimental results consistently refers to a single rollout for all methods. Consequently, each reported step for off-policy training corresponds to 8 gradient updates. A constant learning rate of 1 × 10−6 is applied without a warmup schedule. We set the group number as 8 for all GRPO-based methods. To ensure numerical stability, we set the policy gradient’s clipping by setting the upper bound of the importance sampling ratio to U = 10. For our proposed Lp-Reg, which uses the min-p threshold, we set the probability percentile threshold ρ to 0.5% for Qwen2.5-32B-Base and 1% for Qwen3-14B-Base, the KL regularization coefficient β to 1.0, and

the min-p ratio κ to 0.02. The proxy distribution, πproxy, is constructed from the data-generating policy (πθold in the off-policy setting and the current policy πθ in the on-policy setting). For all baseline methods, we adopt the hyperparameters specified in their original public implementations to ensure a faithful reproduction. Specifically for the GRPO + Entropy Loss baseline, we set the entropy coefficient to 0.002 within the verl framework.

###### 5.1.3 Evaluation

For evaluation, we assess model performance across five diverse mathematical reasoning benchmarks: AIME24 (MAA), AIME25 (MAA), MATH-500 (Hendrycks et al., 2021), OlympiadBench (He et al., 2024a), and Minerva Math (Lewkowycz

et al., 2022). Following (Cui et al., 2025), we employ distinct decoding strategies based on the benchmark. For AIME24 and AIME25, which have smaller test sets, we use sampled decoding with a temperature of 0.6 and generate 16 independent responses per problem to obtain a robust performance estimate. For the remaining benchmarks, including MATH-500, OlympiadBench, and Minerva, we utilize greedy decoding to evaluate performance.

###### 5.2 Results

0.8

0.60

8000

0.7

0.55

###### %Avg.TestAccuracy()

7000

0.6

0.50

###### ResponseLength

6000

###### TrainEntropy

0.5

0.45

5000

0.4

0.40

4000

0.3

0.35

3000

0.2

0.30

2000

0.1

0.25

1000

Lp-Reg(on.)

Lp-Reg(on.)

Lp-Reg(on.)

0

0.0

0.20

0 500 1000 1500 2000 2500 3000

0 500 1000 1500 2000 2500 3000

0 500 1000 1500 2000 2500 3000

Training Steps

Training Steps

Training Steps

- Figure 3: Continuous scaling over 3,000 training steps, totaling 81,204 GPU-hours, for Lp-Reg (on-policy) on the Qwen2.5-32B-Base model.

As shown in Figure 3, Lp-Reg enables a continuous reinforcement learning scaling for 3,000 training steps, totaling 81,204 GPU-hours on the Qwen2.5-32B-Base model. Furthermore, Figure 4 and Table 1 reveal that Lp-Reg achieves state-of-the-art performance across five challenging mathematical reasoning benchmarks on both 14B and 32B model scales. On the Qwen3-14B model, on-policy Lp-Reg sets a new benchmark with an average accuracy of 60.17%, surpassing the next best method, 80/20, by 2.66%. Notably, Lp-Reg’s advantage is more pronounced on the newer Qwen3-14B base model compared to the older Qwen2.5-32B in the first 1,000 training steps. We hypothesize that as base models improve, their capacity for nuanced, low-probability reasoning increases, creating a richer substrate for the emergence of valuable reasoning sparks, which Lp-Reg can then effectively protect and leverage.

Our experiments consistently show the superiority of on-policy training over off-policy methods across 14B and 32B scales in Figure 4. This is due to the inherent stability of on-policy updates, which avoid distribution shifts caused by mismatched data-sampling and training policies. Off-policy methods, such as Clip-Higher, often rely on importance sampling clipping, leading to instability. While competitive on Qwen2.5-32B, Clip-Higher’s performance drops on Qwen3-14B, highlighting its fragility. In contrast, Lp-Reg’s self-contained, policy-intrinsic regularization ensures its effectiveness in both on-policy and off-policy settings, unlike competing methods that are heavily reliant on off-policy importance sampling.

Beyond raw performance, Lp-Reg demonstrates a distinct entropy signature indicative of a healthy explorationexploitation balance. As shown in Figure 5, methods like Clip-Higher induce a continuous, often artificial increase in policy entropy. Lp-Reg, however, facilitates a dynamic, multi-phase entropy trajectory: entropy initially decreases as the model learns core reasoning patterns, then gradually increases to foster exploration as performance improves, and finally stabilizes within a healthy range as accuracy converges. This adaptive behavior stems from our confidenceaware regularization, which selectively protects reasoning sparks without amplifying low-probability out-of-context irrelevant noise.

Method AIME24 AIME25 Math-500 Minerva Olympiad Bench Avg. Qwen2.5-32B-Base (800 training steps)

GRPO (Shao et al., 2024a) (off.) 30.63 22.29 88.00 41.18 54.37 47.29 GSPO (Zheng et al., 2025) (off.) 33.33 22.29 87.60 48.53 55.56 49.46 Clip-Higher (Yu et al., 2025) (off.) 38.33 29.79 87.60 45.22 56.44 51.48 KL-Cov (Cui et al., 2025) (off.) 35.62 27.50 87.40 44.49 55.11 50.02 80/20 (Wang et al., 2025) (off.) 38.12 28.75 87.00 45.22 58.37 51.49 Lp-Reg (off.) 37.71 24.58 90.20 40.81 59.70 50.60 GRPO (Shao et al., 2024a) (on.) 28.54 22.50 86.60 44.85 60.30 48.56 GRPO + Entropy Loss (on.) 3.75 1.88 60.80 27.94 22.22 23.32 80/20 (Wang et al., 2025)(on.) 32.50 28.54 89.40 45.59 57.63 50.73 Lp-Reg (on.) 38.12 27.08 90.00 46.32 61.19 52.54

Qwen3-14B-Base (1,000 training steps)

GRPO (Shao et al., 2024a) (off.) 34.38 27.08 89.20 49.26 55.70 51.13 GSPO (Zheng et al., 2025) (off.) 41.46 34.58 88.60 50.74 59.85 55.05 Clip-Higher (Yu et al., 2025) (off.) 41.67 32.71 95.00 47.43 64.00 56.16 KL-Cov (Cui et al., 2025) (off.) 49.17 34.79 93.00 47.43 62.07 57.29 80/20 (Wang et al., 2025) (off.) 43.96 34.58 91.80 48.16 60.89 55.88 Lp-Reg (off.) 46.25 34.17 92.40 48.16 64.44 57.08 GRPO (Shao et al., 2024a) (on.) 46.04 34.38 93.00 48.53 65.19 57.43 GRPO + Entropy Loss (on.) 37.29 25.21 88.20 46.32 54.96 50.40 80/20 (Wang et al., 2025) (on.) 47.29 32.50 91.60 50.37 65.78 57.51 Lp-Reg (on.) 50.83 37.92 94.40 49.26 68.44 60.17

Table 1: Main results on five mathematical reasoning benchmarks across two model scales. On-policy (on.) and off-policy (off.) training methods are highlighted with distinct colors. We report the best score among the total training for each method.

###### Off-Policy Models

On-Policy Models

0.62

0.62

| | | |GR GR<br><br>| | |PO(on.) PO + Entropy| | | |Loss(on.)| | | |80/20 Lp-Re<br><br>| | | |(on.) g(on.)| | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |

GRPO(off.) GSPO(off.)

Clip-Higher(off.)

80/20(off.)

%Avg.TestAccuracy()

%Avg.TestAccuracy()

KL-Cov(off.)

Lp-Reg(off.)

0.60

0.60

0.58

0.58

0.55

0.55

0.53

0.53

0.50

0.50

0.48

0.48

0.45

0.45

0 200 400 600 800 1000

0 200 400 600 800 1000

Training Steps

Training Steps

- Figure 4: Training dynamics on the Qwen3-14B-Base model. On-policy training exhibits better training stability and testing performance compared to off-policy training.

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0

1000

2000

3000

4000

5000

6000

7000

8000

ResponseLength

| | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0.0

0.1

0.2

0.3

0.4

0.5

0.6

TrainEntropy

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0.45

0.50

0.55

0.60

%Avg.TestAccuracy()

Clip-Higher(off.) GRPO(on.) Lp-Reg(on.)

- Figure 5: Training dynamics on the Qwen3-14B-Base model. To best illustrate the performance differences, we compare the top-performing methods. Lp-Reg demonstrates more stable performance throughout training.

###### 5.3 Ablation Study

We conduct a series of ablation studies to analyze the core components of Lp-Reg and validate our key design choices for regularization in Equation 6, including low-probability regularization, noise filtering, negative sample regularization, and forward KL.

Importance of Low-Probability Token To verify that targeting low-probability tokens is superior to the conventional wisdom of targeting high entropy, we conduct a comparison between the high-entropy token regularization (w/ highest H regularization) and the low-probability regularization (w/ lowest πθ regularization, vanilla Lp-Reg). Instead of applying Lp-Reg to the lowest 1% probability tokens, we apply an identical regularization mechanism to the tokens with the highest 1% entropy. As shown in Figure 7, this approach not only fails to improve performance

Clip-Higher(off.) 80/20(on.) Lp-Reg(on.)

8000

1.0

| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

0.54

%Avg.TestAccuracy()

7000

0.8

0.52

###### ResponseLength

6000

###### TrainEntropy

0.50

5000

0.6

0.48

4000

0.4

0.46

3000

0.44

2000

0.2

0.42

1000

0

0.0

0.40

0 200 400 600 800

0 200 400 600 800

0 200 400 600 800

Training Steps

Training Steps

Training Steps

- Figure 6: Training dynamics on the Qwen2.5-32B-Base model. To best illustrate the performance differences, we compare the top-performing methods.

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0

1000

2000

3000

4000

5000

6000

7000

8000

ResponseLength

| | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0.0

0.2

0.4

0.6

0.8

1.0

TrainEntropy

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0.45

0.48

0.50

0.53

0.55

0.58

0.60

0.62

0.65

%Avg.TestAccuracy()

No Regularization (GRPO) Highest Regularization Lowest Regularization

- Figure 7: Ablation study comparing low-probability token regularization versus high-entropy token regularization for Lp-Reg (on-policy) on the Qwen3-14B-Base model.

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0

1000

2000

3000

4000

5000

6000

7000

8000

ResponseLength

0 200 400 600 800 1000

Training Steps

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

TrainEntropy

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0.20

0.30

0.40

0.50

0.60

%Avg.TestAccuracy()

GRPO Lp-Reg w/ fixed Lp-Reg w/o Lp-Reg w/ dynamic

- Figure 8: Ablation studies for noise filtering of Lp-Reg (on-policy) on the Qwen3-14B-Base model. The results confirm that targeting our noise filtering threshold τ is critical for stable performance. The adaptiveness of the min-p threshold is also shown to be beneficial over a fixed one.

but also fails to sustain high entropy, which collapses after an initial spike. This result reinforces our claim from the Introduction: high entropy is a poor proxy for valuable exploration. As our analysis in Section 6.1 further corroborates, high-entropy tokens are often common function words or formatting characters, not the meaningful, low-probability exploratory tokens we term reasoning sparks. Regularizing them pollutes the learning signal without protecting the structured, low-probability reasoning paths necessary for progress.

Importance of Noise Filtering. Lp-Reg only protects tokens deemed meaningful by the proxy distribution (πproxy > 0). To test this, we remove the filter and fork all tokens below the noise threshold τ from contributing to gradient updating (Lp-Reg w/o τ). Figure 8 shows that this leads to a catastrophic performance collapse and entropy explosion. This confirms that filtering is critical to ignore the extreme tail of the distribution, which consists of irrelevant noise that destabilizes training if regularized. We further conduct a comparison between the dynamic min-p noise threshold (Lp-Reg w/ dynamic τ) and the fixed noise threshold (Lp-Reg w/ fixed τ) in Section 4.1. As shown in Figure 8, the fixed threshold underperforms compared to the dynamic threshold, which we adopt as the default. However, it still significantly surpasses the standard GRPO. This indicates that while the core filtering principle is effective, the dynamic nature of min-p provides a more robust estimate of the model’s confidence across different contexts, better preserving genuine reasoning sparks.

Effect of Negative Samples. To dissect the contribution of our regularization strategy, we analyze the effects of applying the Lp-Reg penalty to negative samples versus both positive and negative samples. As illustrated in Figure 9, applying regularization to low-probability tokens in all samples (Regularize Pos. & Neg.) sustains exploration and yields continuous performance growth, outperforming standard GRPO (Regularize None). This

Regularize None (GRPO) Regularize Neg. Regularize Pos. & Neg.

8000

1.0

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

0.60

%Avg.TestAccuracy()

7000

0.8

###### ResponseLength

0.55

6000

###### TrainEntropy

5000

0.50

0.6

4000

0.45

0.4

3000

0.40

2000

0.2

0.35

1000

0

0.0

0.30

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

Training Steps

Training Steps

Training Steps

- Figure 9: Ablation study comparing positive sample regularization and negative sample regularization for the Lp-Reg penalty (on-policy) on the Qwen3-14B-Base model. Negative sample regularization exhibits better performance.

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0

1000

2000

3000

4000

5000

6000

7000

8000

ResponseLength

| | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0.0

0.2

0.4

0.6

0.8

1.0

TrainEntropy

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

0 200 400 600 800 1000

Training Steps

0.30

0.35

0.40

0.45

0.50

0.55

0.60

%Avg.TestAccuracy()

KL( proxy) KL( proxy )

- Figure 10: Ablation study comparing the forward and reverse KL formulations for the Lp-Reg penalty (on-policy) on the Qwen3-14B-Base model. The results demonstrate the superiority of the forward KL, which uses the heuristic proxy distribution as a soft guide, over the reverse KL, which forces a strict imitation.

finding validates that protecting low-probability tokens is essential for successful exploration in RLVR. Furthermore, the comparison reveals that concentrating the regularization solely on negative samples leads to a faster learning rate than applying it to both sample types. We attribute this to the primary function of Lp-Reg, which is to protect valuable low-probability tokens from being excessively penalized during training. These crucial tokens are far more prevalent in negative samples. Once these tokens are protected, extending the penalty to positive samples provides little additional exploratory advantage. Instead, it appears to introduce minor noise into the gradient updates, which, while not destabilizing, slows the overall training progress. Thus, we conclude that targeting negative samples exclusively is the most efficient and effective application of our method.

Forward KL vs. Reverse KL. We compare the performance of our chosen “forward KL” formulation, which is DKL(πproxy∥πθ), against the ”reverse KL” formulation, DKL(πθ∥πproxy), in Equation 6. As shown in Figure 10, our forward KL significantly outperforms the reverse KL. This result stems from the fact that our proxy distribution, πproxy, is a heuristic reference derived from the current policy, not an ideal target distribution. The reverse KL, DKL(πθ∥πproxy), penalizes any deviation of πθ from πproxy, effectively forcing the policy to strictly imitate this non-ideal, heuristic target. This aggressive imitation constrains the protection of potentially valuable exploratory tokens. In contrast, the forward KL, DKL(πproxy∥πθ), provides a much softer regularization: it only penalizes the policy for completely discarding tokens that πproxy considers plausible, without forcing a strict match. This allows the policy to use πproxy as a stabilizing guide while retaining the freedom to explore beyond it, which empirically leads to better performance.

##### 6 Analysis

To understand the mechanisms behind Lp-Reg’s performance, we conduct a series of analyses focusing on how it overcomes the exploration bottleneck by targeting and preserving valuable reasoning tokens.

###### 6.1 Probability-Entropy Distribution of Exploratory Tokens

We begin by exploring the distinction between low-probability tokens and high-entropy tokens. Figure 11 highlights this contrast by comparing tokens from the top 1% lowest probability with those from the top 1% highest entropy. The difference is striking: low-probability tokens frequently include semantically meaningful exploratory markers such as “But”, “Wait”, “Perhaps”, and “Alternatively”, which often signal a shift in the reasoning trajectory. In contrast, high-entropy tokens are dominated by common functional terms (e.g., “sqrt”, “times”) or formatting symbols (e.g., \n), which carry little exploratory intent. This explains why entropy-based regularization often fails

High Entropy Tokens Low Probability Tokens

[Figure 5]

[Figure 6]

Figure 11: The word cloud statistics from training samples at GRPO training step 1 on Qwen3-14B-Base.

to enhance exploration: it confuses noise with exploration.

However, the set of low-probability tokens is also not uniformly useful. It also includes noisy artifacts such as spurious newline characters (\n) or formatting debris, whose regularization can destabilize training rather than enhance reasoning. To mitigate this, Lp-Reg applies a threshold τ that filters out such noise. Ablation studies in Section 5.3 confirm the necessity of this step: removing the threshold results in unstable training dynamics and degraded reasoning performance. Thus, Lp-Reg’s effectiveness stems not only from targeting low-probability tokens but also from selectively excluding irrelevant noise.

###### 6.2 Sampling Dynamics of Exploratory Tokens

###### GRPO

###### GRPO+entropy loss

Lp-Reg

- 0

- 1

- 2

- 3

- 4

- 5

n = 42023

n = 34470

n = 74079

3.5

Mean Prob: 0.9099 ± 0.2036 Mean Entropy: 0.1576 ± 0.2717

Mean Prob: 0.4565 ± 0.3172 Mean Entropy: 1.5960 ± 1.4706

Mean Prob: 0.7169 ± 0.3100 Mean Entropy: 0.6183 ± 0.5867

10

3.0

2.5

8

Entropy

Entropy

Entropy

2.0

6

1.5

4

1.0

2

0.5

0.0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

Sampled Probability

Sampled Probability

Figure 12: Probability–Entropy scatter plots of five explorative tokens from training samples at training step 256 on Qwen3-14B-Base, displaying a random sample of 5% of all data points. Detailed plots for individual tokens are analyzed in Appendix B.2

- Figure 12 shows the probability–entropy distributions of key explorative tokens (“but”, “wait”, “perhaps”, “alternatively”, and “however”) under three methods: GRPO, GRPO + Entropy Loss, and our Lp-Reg.

With the baseline GRPO, these tokens are concentrated in low-entropy, high-probability regions. In this case, tokens like “wait” tend to appear only when the model is already confident, turning them into deterministic patterns rather than initiating a new exploration path with uncertainty.

Adding an entropy loss changes this behavior, but in an uncontrolled way. Some sampled “wait” tokens appear at extremely high entropy levels (sometimes exceeding 10), which superficially boosts diversity but produces little useful exploratory signal. These scattered occurrences do not integrate meaningfully into the reasoning process.

Our Lp-Reg method yields a more balanced dynamic. Explorative tokens are sampled across a broad range of entropy values, from high probability to low probability states. This balance prevents their probabilities from collapsing under negative feedback while keeping them informative for reasoning. As a result, tokens like “wait” remain viable options throughout training, allowing the model to explore alternative reasoning paths rather than overfitting to fixed usage patterns.

- Figure 13 further compares the frequency of explorative tokens (“but”, “wait”, “perhaps”, “alternatively”, and “however”) under GRPO and Lp-Reg. Our method consistently maintains a higher fraction of these tokens, demonstrating that Lp-Reg not only broadens their probability–entropy distribution but also sustains their practical use throughout training.

GRPO Lp-Reg

###### Step 256

###### Step 512

Step 768

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.0035

0.0035

0.0035

0.0030

0.0030

0.0030

0.0025

0.0025

0.0025

Frequency

Frequency

Frequency

0.0020

0.0020

0.0020

0.0015

0.0015

0.0015

0.0010

0.0010

0.0010

0.0005

0.0005

0.0005

0.0000

0.0000

0.0000

but wait perhaps alternatively however

but wait perhaps alternatively however

but wait perhaps alternatively however

Figure 13: Frequency of explorative tokens during training.

###### 6.3 Probabilistic Distinction between Reasoning Sparks and Noise

Our introduction established a challenge for a successful exploration strategy: it must protect valuable, lowprobability reasoning sparks without simultaneously amplifying the destructive effects of irrelevant noise. This raises a critical question: is there a systemic, observable difference between these two classes of tokens within the low-probability range that our method can exploit?

To investigate this, we analyze the next-token prediction distribution throughout the training process. Due to storage limitations, we focus our analysis on the top-64 most probable tokens, but specifically examine those within a low-probability range (0 to 0.1) to isolate the phenomenon from high-probability tokens. Figure 14 plots the average probability of two distinct classes of tokens over time: a group of meaningful exploratory tokens (e.g., “wait”, “perhaps”) and a group of irrelevant tokens (e.g., “cost”, “fine”).

The results reveal a clear and consistent statistical distinction: across all training stages, the average next-token probability of meaningful exploratory tokens is persistently higher than that of irrelevant tokens. It can be attributed to the intrinsic confidence of LLMs (Nguyen et al., 2025; Xu et al., 2025; Fu et al., 2025). This persistent probabilistic gap provides the foundational justification for our Lp-Reg design. It suggests that while a perfect separation is not possible, a probability threshold τ, as defined for our proxy distribution in Section 4.1, can serve as a principled filtering mechanism. By setting such a threshold, we can effectively filter out a substantial portion of the lowest-probability irrelevant tokens, which constitute destabilizing noise, while simultaneously retaining a majority of the valuable exploratory tokens that give rise to reasoning sparks. This allows Lp-Reg to focus its regularization on tokens that are more likely to be meaningful, providing a targeted and robust approach to preserving high-quality exploration.

###### Mean Probability of Tokens in (0, 0.1)

Exploratory Token

0.008

Irrelevant Token

0.007

MeanProbability

0.006

0.005

0.004

| |
|---|

| |
|---|

0.003

| |
|---|

| |
|---|

50 100 150 200 250 300 350 400

Training Step

Figure 14: Probabilistic distinction between exploratory and irrelevant tokens across training steps in standard GRPO training.

##### 7 Conclusion

In this work, we investigated the exploration collapse in Reinforcement Learning with Verifiable Rewards. We identified a key mechanism driving this failure: the systematic elimination of a class of valuable, low-probability exploratory tokens we term reasoning sparks. To address this, we introduced Low-probability Regularization (Lp-Reg), a method designed to selectively preserve these crucial exploratory pathways. Lp-Reg is founded on the key insight that within the low-probability range, meaningful exploratory tokens consistently exhibit higher average probabilities than semantically irrelevant tokens, whose low-probability appearances act as noise. By leveraging this statistical distinction to filter out irrelevant noise and regularizing the policy towards the remainder, our method effectively protects valuable low-probability tokens from being extinguished. This focus on exploration quality over quantity enables continuous on-policy scaling for around 3,000 steps, resulting in an absolute 2.66% test accuracy improvement over baselines and underscoring the importance of preserving the reasoning sparks within the policy’s low-probability tail.

##### References

Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, and Hao Peng. The unreasonable effectiveness of entropy minimization in llm reasoning, 2025. URL https://arxiv.org/abs/2505.15134.

Taubenfeld Amir, Sheffer Tom, Ofek Eran, Feder Amir, Goldstein Ariel, Gekhman Zorik, and Yona Gal. Confidence improves self-consistency in llms. arXiv preprint arXiv:2502.06233, 2025. URL https://www.arxiv. org/abs/2502.06233.

Ghasemabadi Amirhosein, Mills Keith, G., Li Baochun, and Niu Di. Guided by gut: Efficient test-time scaling with reinforced intrinsic confidence. arXiv preprint arXiv:2505.20325v1, 2025. URL https://www.arxiv. org/abs/2505.20325v1.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective on reinforcement learning for llms, 2025. URL https: //arxiv.org/abs/2506.14758.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models, 2025. URL https: //arxiv.org/abs/2505.22617.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. Deep think with confidence, 2025. URL https: //arxiv.org/abs/2508.15260.

Zitian Gao, Lynx Chen, Haoming Luo, Joey Zhou, and Bryan Dai. One-shot entropy minimization, 2025. URL https://arxiv.org/abs/2505.20282.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, Bangkok, Thailand, August 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.211. URL https://aclanthology.org/ 2024.acl-long.211/.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024b.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner 1 technical report, 2025. URL https://arxiv.org/abs/2505.22312.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https: //openreview.net/forum?id=7Bywt2mQsCe.

Aitor Lewkowycz, Anders Johan Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=IFXTZERXdM7.

Li Loka, Chen Zhenhao, Chen Guangyi, Zhang Yixuan, Su Yusheng, Xing Eric, and Zhang Kun. Confidence matters: Revisiting intrinsic self-correction capabilities of large language models. arXiv preprint arXiv:2402.12563, 2024. URL https://www.arxiv.org/abs/2402.12563.

MAA. American invitational mathematics examination (AIME). Mathematics Competition Series, n.d. URL https://maa.org/math-competitions/aime.

MiniMax, :, Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, Chengjun Xiao, Chengyu Du, Chi Zhang, Chu Qiao, Chunhao Zhang, Chunhui Du, Congchao Guo, Da Chen, Deming Ding, Dianjun Sun, Dong Li, Enwei Jiao, Haigang Zhou, Haimo Zhang, Han Ding, Haohai Sun, Haoyu Feng, Huaiguang Cai, Haichao Zhu, Jian Sun, Jiaqi Zhuang, Jiaren Cai, Jiayuan Song, Jin Zhu, Jingyang Li, Jinhao Tian, Jinli Liu, Junhao Xu, Junjie Yan, Junteng Liu, Junxian He, Kaiyi Feng, Ke Yang, Kecheng Xiao, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Li, Lin Zheng, Linge Du, Lingyu Yang, Lunbin Zeng, Minghui Yu, Mingliang Tao, Mingyuan Chi, Mozhi Zhang, Mujie Lin, Nan Hu, Nongyu Di, Peng Gao, Pengfei Li, Pengyu Zhao, Qibing Ren, Qidi Xu, Qile Li, Qin Wang, Rong Tian, Ruitao Leng, Shaoxiang Chen, Shaoyu Chen, Shengmin Shi, Shitong Weng, Shuchang Guan, Shuqi Yu, Sichen Li, Songquan Zhu, Tengfei Li, Tianchi Cai, Tianrun Liang, Weiyu Cheng, Weize Kong, Wenkai Li, Xiancai Chen, Xiangjun Song, Xiao Luo, Xiao Su, Xiaobo Li, Xiaodong Han, Xinzhu Hou, Xuan Lu, Xun Zou, Xuyang Shen, Yan Gong, Yan Ma, Yang Wang, Yiqi Shi, Yiran Zhong, Yonghong Duan, Yongxiang Fu, Yongyi Hu, Yu Gao, Yuanxiang Fan, Yufeng Yang, Yuhao Li, Yulin Hu, Yunan Huang, Yunji Li, Yunzhi Xu, Yuxin Mao, Yuxuan Shi, Yuze Wenren, Zehan Li, Zelin Li, Zhanxu Tian, Zhengmao Zhu, Zhenhua Fan, Zhenzhen Wu, Zhichao Xu, Zhihang Yu, Zhiheng Lyu, Zhuo Jiang, Zibo Gao, Zijia Wu, Zijian Song, and Zijun Sun. Minimax-m1: Scaling test-time compute efficiently with lightning attention, 2025. URL https://arxiv.org/abs/2506.13585.

Kevin Murphy. Reinforcement learning: an overview. arXiv preprint arXiv:2412.05265, 2024. Minh Nhat Nguyen, Andrew Baker, Clement Neo, Allen Roush, Andreas Kirsch, and Ravid Shwartz-Ziv. Turning

up the heat: Min-p sampling for creative and coherent llm outputs, 2025. URL https://arxiv.org/abs/ 2407.01082.

OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Qui˜nonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz,

Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card, 2024. URL https://arxiv.org/abs/2412.16720.

Kadavath Saurav, Conerly Tom, Askell Amanda, Henighan Tom, Drain Dawn, Perez Ethan, Schiefer Nicholas, Hatfield-Dodds Zac, DasSarma Nova, Tran-Johnson Eli, Johnston Scott, El-Showk Sheer, Jones Andy, Elhage Nelson, Hume Tristan, Chen Anna, Bai Yuntao, Bowman Sam, Fort Stanislav, Ganguli Deep, Hernandez Danny, Jacobson Josh, Kernion Jackson, Kravec Shauna, Lovitt Liane, Ndousse Kamal, Olsson Catherine, Ringer Sam, Amodei Dario, Brown Tom, Clark Jack, Joseph Nicholas, Mann Ben, McCandlish Sam, Olah Chris, and Kaplan Jared. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022. URL https://www.arxiv.org/abs/2207.05221.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024a. URL https://arxiv.org/abs/2402.03300.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024b.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Zhenpeng Su, Leiyu Pan, Xue Bai, Dening Liu, Guanting Dong, Jiaming Huang, Wenping Hu, Fuzheng Zhang, Kun Gai, and Guorui Zhou. Klear-reasoner: Advancing reasoning capability via gradient-preserving clipping policy optimization, 2025. URL https://arxiv.org/abs/2508.07629.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Weixin Xu, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, Zonghan Yang, and Zongyu Lin. Kimi k1.5: Scaling reinforcement learning with llms, 2025. URL https://arxiv.org/abs/2501.12599.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning, 2025. URL https://arxiv.org/abs/2506.01939.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL https: //arxiv.org/abs/2201.11903.

Zenan Xu, Zexuan Qiu, Guanhua Huang, Kun Li, Siheng Li, Chenchen Zhang, Kejiao Li, Qi Yi, Yuhao Jiang, Bo Zhou, Fengzong Lian, and Zhanhui Kang. Adaptive termination for multi-round parallel reasoning: An universal semantic entropy-guided framework, 2025. URL https://arxiv.org/abs/2507.06829.

Zhao Xuandong, Kang Zhewei, Feng Aosong, Levine Sergey, and Song Dawn. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590v2, 2025. URL https://www.arxiv.org/abs/2505. 19590v2.

Wang Xuezhi and Zhou Denny. Chain-of-thought reasoning without prompting. arXiv preprint arXiv:2402.10200v2,

2024. URL https://www.arxiv.org/abs/2402.10200v2.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL

- https://arxiv.org/abs/2503.14476.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, Xiangyu Yu, Gaohong Liu, Juncai Liu, Lingjun Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Ru Zhang, Xin Liu, Mingxuan Wang, Yonghui Wu, and Lin Yan. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks, 2025. URL

- https://arxiv.org/abs/2504.05118.

Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, Fang Wan, and Furu Wei. Geometric-mean policy optimization, 2025. URL https://arxiv. org/abs/2507.20673.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025. URL https: //arxiv.org/abs/2507.18071.

Omer¨ Veysel C¸a˘gatan and Barı¸s Akg¨un. Failure modes of maximum entropy rlhf, 2025. URL https://arxiv. org/abs/2509.20265.

## Appendix

##### A Details of Experiments

= 0.005 = 0.010 = 0.015

8000

1.0

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.60

%Avg.TestAccuracy()

7000

0.8

ResponseLength

0.55

6000

TrainEntropy

5000

0.50

0.6

4000

0.45

0.4

3000

0.40

2000

0.2

0.35

1000

0

0.0

0.30

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

Training Steps

Training Steps

Training Steps

(a) Effect of different ρ which defined the low-probability percentile threshold δρB.

= 0.01 = 0.02 = 0.03

8000

1.0

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.60

%Avg.TestAccuracy()

7000

0.8

ResponseLength

0.55

6000

TrainEntropy

5000

0.50

0.6

4000

0.45

0.4

3000

0.40

2000

0.2

0.35

1000

0

0.0

0.30

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

Training Steps

Training Steps

Training Steps

(b) Effect of different κ which defined the noise threshold τ = κ · maxo′∈V πθ(o′|·)

Figure 15: Training dynamics of Lp-Reg method with different hyperparameters.

- A.1 Hyperparameter Sensitivity Analysis

In this section, we analyze the sensitivity of two core hyperparameters in Lp-Reg to demonstrate the robustness of our method: the low-probability percentile ρ and the min-p ratio κ. The results are presented in Figure 15.

The parameter ρ, as defined in our objective function (Equation 6), determines the percentile threshold for identifying low-probability tokens that are candidates for regularization. A higher ρ means a wider range of tokens are protected. As shown in the top panel of Figure 15, we evaluated ρ with values of 0.005, 0.010, and 0.015. The training trajectories for average test accuracy are comparable, and the final performance across all three settings is highly comparable. This indicates that Lp-Reg is not overly sensitive to the precise scope of tokens being protected within this reasonable range.

The hyperparameter κ controls the adaptiveness of the min-p filtering threshold, which defines the boundary for what is treated as noise. A smaller κ results in a more conservative filtering strategy, removing fewer tokens. Our sensitivity analysis for κ, presented in the bottom panel of Figure 15, shows a similar trend of stability. Across the tested values of 0.01, 0.02, and 0.03, the training curves and final performance remain consistently high and tightly clustered. Taken together, these results demonstrate the robustness of Lp-Reg. The method’s effectiveness is not contingent on extensive, fine-grained hyperparameter tuning, highlighting its practical applicability.

B Further Analysis

- B.1 Details of Sampling Probability Density

This section provides a detailed, token-by-token breakdown of the aggregated distributions presented in Figure 1c and Figure 1d of the main paper, reinforcing the conclusions drawn from our analysis.

Figure 17 exhibits the individual distribution of observed sampling probabilities for a class of meaningful lowprobability exploratory tokens we term reasoning sparks: “but”, “wait”, “perhaps”, “alternatively”, and “however”.

A consistent trend is observable across all five tokens, validating our claims in the introduction. With standard GRPO training, the ability to sample these tokens at low probabilities is systematically eliminated, causing their distributions to collapse and shift towards higher probabilities. The indiscriminate entropy bonus (GRPO + Entropy Loss) is largely ineffective at restoring this crucial low-probability tail. In stark contrast, our proposed method, Lp-Reg, consistently maintains a healthy, wide distribution for each of these tokens, demonstrating its effectiveness in preserving the model’s capacity for exploration.

Conversely, Figure 18 details the behavior of a class of what we term irrelevant tokens (e.g. “cost”, “fine”, “balanced”, “ere”, and “trans”). We refer to the low-probability sampling of these tokens as irrelevant noise, which can be destructive to the training process. These individual plots clearly illustrate the detrimental side effect of a simple entropy bonus. For nearly every token, the GRPO + Entropy Loss baseline significantly amplifies the sampling of this irrelevant noise, which, as shown in our main analysis, contributes to a faster performance collapse. Lp-Reg, by design, avoids this amplification and maintains a suppressed probability distribution for these tokens, comparable to or even more constrained than the standard GRPO baseline.

These detailed visualizations confirm that the elimination of reasoning sparks and the amplification of irrelevant noise are not artifacts of aggregation but are consistent patterns at the individual token level. This provides strong, granular evidence for the central challenge our paper addresses and highlights the necessity of a selective preservation mechanism like Lp-Reg.

###### B.2 Details of Probability-Entropy Distribution

To supplement the aggregated analysis presented in Figure 12 of the main text, this section provides a detailed breakdown of the probability-entropy distributions for individual meaningful exploratory tokens. Figure 19 shows a consistent pattern across all representative tokens, ranging from “but” (Figure 19a) to “however” (Figure 19e). For frequently occurring tokens such as “but”, “wait”, and “perhaps”, we randomly subsample one out of every 20 instances for visualization. Under the baseline GRPO, these sparks are consistently confined to a low-entropy, highprobability region, indicating a collapse into deterministic usage. In contrast, the addition of an entropy loss pushes these tokens into highly scattered, often excessively high-entropy states, suggesting an uncontrolled and potentially noisy form of exploration. Our method, Lp-Reg, strikes a crucial balance, maintaining a structured and broad distribution across a healthy range of entropy values. This consistent behavior demonstrates that the trends identified in the aggregated data are not artifacts of averaging. The individual plots offer strong, disaggregated evidence for our central claim: Lp-Reg effectively preserves the exploratory potential of reasoning sparks by preventing both the deterministic collapse seen in the baseline and the chaotic scattering induced by the indiscriminate entropy bonus.

###### B.3 Training Dynamics of Regularized Token To better understand how Lp-Reg operates during training, we analyze the dynamics of the probability threshold

ρB∧πproxy(o|·)>0|

δρB and the proportion of low-probability tokens subjected to regularization |πθ(o|·)<δ

|πθ(o|·)<δρB| . As shown in Figure 16, the threshold δρB gradually decreases with training steps. At the same time, the regularization ratio also declines steadily. This trend suggests that as training progresses, the extreme low-probability range becomes increasingly dominated by irrelevant tokens, constituting what we term irrelevant noise. It also explains the relatively worse performance using a fixed threshold τ = 0.02 than the dynamic min-p threshold τ in Equation 5, since a fixed threshold failed to regularize the lowest 1% tokens while the dynamic one succeeded. Furthermore, the semantically meaningful exploratory tokens are lifted into higher-probability regions, thus requiring less regularization.

###### Probability Threshold

###### Regularization Ratio

Probability Threshold

Regularization Ratio

27.5

0.08

RegularizationRatio(%)

25.0

ProbabilityThreshold

22.5

0.06

20.0

17.5

0.04

15.0

0.02

12.5

10.0

0.00

7.5

128 256 384 512 640 768 Training Steps

128 256 384 512 640 768 Training Steps

Figure 16: Training dynamics of the probability threshold and regularization ratio.

###### B.4 Case Study

To further illustrate the effect of the filter applied on low-probability tokens, Figure 20 to Figure 22 present a case study of a model-generated response, where low-probability tokens are highlighted according to whether they were preserved or filtered. Tokens with probability greater than τ are those retained by the filter, while tokens with probability smaller than τ are suppressed. The preserved tokens include meaningful exploratory markers such as ”Then”, ”Wait”, which guide the reasoning trajectory, whereas the discarded set largely consists of relatively irrelevant tokens such as ”We”, ”also”, ”that”. This qualitative evidence complements our quantitative analysis, demonstrating that Lp-Reg effectively leverages the min-p transformation to distinguish between valuable low-probability exploratory tokens (reasoning sparks) and tokens that constitute destabilizing, irrelevant noise.

n=656 n=656 n=656 n=25239 n=11311 n=28173 n=21728 n=35821

Probability

0.75

0.50

0.25

0.00

1 256 512

Training Steps

(a) Density of observed sampling probabilities for token “but”.

GRPO GRPO+Entropy loss Lp-Reg

1.00

n=31 n=31 n=31 n=11454 n=15102 n=17153 n=8388 n=17157

Probability

0.75

0.50

0.25

0.00

1 256 512

Training Steps

(b) Density of observed sampling probabilities for token “wait”.

GRPO GRPO+Entropy loss Lp-Reg

1.00

n=12 n=12 n=12 n=14965 n=4386 n=16465 n=6875 n=10859

Probability

0.75

0.50

0.25

0.00

1 256 512

Training Steps

(c) Density of observed sampling probabilities for token “perhaps”.

GRPO GRPO+Entropy loss Lp-Reg

1.00

n=14 n=14 n=14 n=4891 n=1419 n=7355 n=3125 n=6150

Probability

0.75

0.50

0.25

0.00

1 256 512

Training Steps

(d) Density of observed sampling probabilities for token “alternatively”.

GRPO GRPO+Entropy loss Lp-Reg

1.00

n=499 n=499 n=499 n=1864 n=2252 n=2200 n=1907 n=4092

Probability

0.75

0.50

0.25

0.00

1 256 512

Training Steps

(e) Density of observed sampling probabilities for token “however”.

- Figure 17: Individual density of observed sampling probabilities for meaningful exploratory tokens: “but”, “wait”, “perhaps”, “alternatively”, and “however”.

n=156 n=156 n=156 n=60 n=591 n=89 n=107 n=246 n=96

Probability

0.75

0.50

0.25

0.00

1 128 256

Training Steps

(a) Density of observed sampling probabilities for token “cost”.

GRPO GRPO+Entropy loss Lp-Reg

1.00

n=3 n=3 n=3 n=13 n=411 n=32 n=23 n=130 n=23

Probability

0.75

0.50

0.25

0.00

1 128 256

Training Steps

(b) Density of observed sampling probabilities for token “fine”.

GRPO GRPO+Entropy loss Lp-Reg

1.00

n=1 n=1 n=1 n=43 n=271 n=30 n=38 n=95 n=30

Probability

0.75

0.50

0.25

0.00

1 128 256

Training Steps

(c) Density of observed sampling probabilities for token “balanced”.

GRPO GRPO+Entropy loss Lp-Reg

1.00

n=27 n=27 n=27 n=0 n=175 n=0 n=0 n=57 n=0

Probability

0.75

0.50

0.25

0.00

1 128 256

Training Steps

(d) Density of observed sampling probabilities for token “ere”.

GRPO GRPO+Entropy loss Lp-Reg

1.00

n=5 n=5 n=5 n=208 n=644 n=79 n=166 n=446 n=91

Probability

0.75

0.50

0.25

0.00

1 128 256

Training Steps

(e) Density of observed sampling probabilities for token “trans”.

- Figure 18: Individual density of observed sampling probabilities for irrelevant tokens: “cost”, “fine”, “balanced”, “ere”, and “trans”.

###### GRPO

###### GRPO+entropy loss

Lp-Reg

- 0

- 1

- 2

- 3

- 4

- 5

n = 21728

n = 11311

n = 35821

3.5

Mean Prob: 0.9315 ± 0.1801 Mean Entropy: 0.1189 ± 0.2440

Mean Prob: 0.4321 ± 0.3277 Mean Entropy: 1.5500 ± 1.3882

Mean Prob: 0.7323 ± 0.3035 Mean Entropy: 0.5976 ± 0.5860

10

3.0

2.5

8

Entropy

Entropy

Entropy

2.0

6

1.5

4

1.0

2

0.5

0.0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

Sampled Probability

Sampled Probability

(a) Scattered probability–entropy plot of observed sampling instances for the token “but”.

###### GRPO

###### GRPO+entropy loss

Lp-Reg

| | |Me|Mean Prob: an Entropy:|n =<br><br>0.8975 ± 0.2 0.1747 ± 0.2|8388 159 816|
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

n = 15102

n = 17157

Mean Prob: 0.5244 ± 0.3036 Mean Entropy: 1.3724 ± 1.0535

Mean Prob: 0.7356 ± 0.3078 Mean Entropy: 0.5430 ± 0.5383

2.0

- 0

- 1

- 2

- 3

- 4

10

8

1.5

Entropy

Entropy

Entropy

6

1.0

4

0.5

2

0.0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

Sampled Probability

Sampled Probability

- (b) Scattered probability–entropy plot of observed sampling instances for the token “wait”.

| | |Mea|Mean Prob: n Entropy:|n =<br><br>0.8820 ± 0.2 0.2125 ± 0.2|6875 273 942|
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

0.00

0.25

0.50

0.75

1.00

1.25

1.50

1.75

2.00

Entropy

GRPO

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

0

2

4

6

8

10

Entropy

n = 4386

Mean Prob: 0.3737 ± 0.2931 Mean Entropy: 2.1152 ± 1.9655

GRPO+entropy loss

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

4.0

Entropy

n = 10859

Mean Prob: 0.7049 ± 0.3117 Mean Entropy: 0.6639 ± 0.6148

Lp-Reg

- (c) Scattered probability–entropy plot of observed sampling instances for the token “perhaps”.

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

1.6

Entropy

n = 3125

Mean Prob: 0.8641 ± 0.2389 Mean Entropy: 0.2428 ± 0.3138

GRPO

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

0

2

4

6

8

10

Entropy

n = 1419

Mean Prob: 0.3093 ± 0.2816 Mean Entropy: 2.0874 ± 2.0776

GRPO+entropy loss

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

4.0

Entropy

n = 6150

Mean Prob: 0.6740 ± 0.3159 Mean Entropy: 0.7196 ± 0.5960

Lp-Reg

- (d) Scattered probability–entropy plot of observed sampling instances for the token “alternatively”.

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

1.6

Entropy

n = 1907

Mean Prob: 0.8941 ± 0.2168 Mean Entropy: 0.1850 ± 0.2897

GRPO

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

0

2

4

6

8

10

Entropy

n = 2252

Mean Prob: 0.3781 ± 0.3130 Mean Entropy: 2.0051 ± 2.1316

GRPO+entropy loss

| | |Mea|Mean Prob: n Entropy:|n =<br><br>0.6001 ± 0.3 0.8421 ± 0.6|4092 283 114|
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.0 0.2 0.4 0.6 0.8 1.0

Sampled Probability

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

Entropy

Lp-Reg

- (e) Scattered probability–entropy plot of observed sampling instances for the token “however”.

- Figure 19: Individual scattered probability–entropy plot of observed sampling instances for meaningful tokens: “but”, “wait”, “perhaps”, “alternatively”, and “however”.

|Assistant :\n Alright , I have this problem about a square that 's divided into smaller squares , and the process keeps repeating indefinitely . Some parts are shaded , and I need to find out what fraction of the whole figure is shaded . The final answer should be in the form of a fraction \ (\ frac { m }{ n }\ ), and I need to provide the value of \( m + n \ ). Let 's break this down step by step .\n\n ### Understanding the Problem \n\n First , let 's try to understand what 's happening :\n\n 1 . ** Initial Square **: We start with one big square .\n 2 . ** First Division **: This square is divided into nine smaller squares of equal area . So , it 's like a 3 x 3 grid .\n 3 . ** Center Division **: The center square of this 3 x 3 grid is then further divided into nine even smaller squares ( 3 x 3 again ).\n 4 . ** Pattern Contin ues **: This process repeats indefinitely â each new center square is divided into nine smaller squares .\n\n From the As ym pt ote code , it seems that at each step , certain parts ( corn ers ) of the squares are shaded . Specifically , the four corner squares of each smaller subdivision are shaded .\n\n Wait , looking at the code :\n\n ``` asy fill ((- n ,n )-- (- 1 / 3 *n ,n )-- (- 1 / 3 *n , 1 / 3 *n )-- (- n , 1 / 3 *n )-- cycle );\n fill ((- n ,- n )-- (- 1 / 3 *n ,- n )-- (- 1 / 3 *n ,- 1 / 3 *n )-- (- n ,- 1 / 3 *n )-- cycle );\n fill (( n ,- n )-- ( 1 / 3<br><br>*n ,- n )-- ( 1 / 3 *n ,- 1 / 3 *n )-- ( n ,- 1 / 3 *n )-- cycle );\n fill (( n ,n )-- ( 1 / 3 *n ,n )-- ( 1 / 3 *n , 1 / 3 *n )-- ( n , 1 / 3 *n )-- cycle );\n `` `\n\n This code seems to shade the four corner squares of each subdivision at each level . So , at each step , four of the nine smaller squares are shaded .\n\n ### Visual Representation \n\n Let 's try to visualize this :\n\n 1 . ** Level 0<br><br>**: The original big square .\n 2 . ** Level 1 **: Div ided into 9 squares , and the four corners are shaded .\n 3 . ** Level 2 **: The center square of Level 1 is divided into 9 even smaller squares , and its four corners are shaded .\n 4 . ** Level 3 **: The center of that latest division is divided again , and so on , infinitely .\n\n Wait , actually , looking back at the As ym pt ote code , it seems that at every level , the four corner squares of each current smallest subdivision are shaded . But in the description , it mentions the center square is divided , not necessarily the corners . Maybe I mis interpreted .\n\n Wait , the code sh ad ings correspond to the corners of each subdivision , but the description says the center square is divided . Hmm , perhaps the shaded parts are always the four corners of the current center square .\n\n But perhaps it 's better to assign a value .\n\n ### Assign ing Areas \n\n Let 's assign an area to the original square to make calculations easier . Let 's say the area of the original square is 1<br><br><br>.\n\n - ** Level 0 **: Total area = 1 . Fraction shaded = 0 ? Wait , no , according to the As ym pt ote code , the parts shaded are the four corners of each subdivision , not clearly defined yet .\n\n Wait , perhaps it 's better to think in terms of geometric series .\n\n At each level :\n\n<br><br>- The entire figure is divided into 9 equal smaller squares .\n - Out of these , 4 are shaded ( the corners ).\n\n Wait , perhaps it 's not exactly that . Looking at the code , for each square at a certain level , its four corners in the next subdivision are shaded . But the center square continues to be subdiv ided .\n\n Wait , perhaps it 's better to think recursively .\n\n ### Recursive Thinking \n\n Let 's denote :\n\n - \( S _n \ ) as the total shaded area after \( n \ ) levels of subdivision .\n\n But since it 's infinite , we need to consider the limit as \( n \ ) approaches infinity .\n\n At each step :\n\n - The current smallest squares are divided into 9 even smaller squares .\n - Out of these 9 , 4 are shaded .\n\n Wait , perhaps it 's better to think of the entire figure and how much is shaded at each stage .\n\n Let 's consider the first level :\n\n - Original square is divided into 9 .\n - Sh aded area : the four corners , each has area \( \ frac { 1 }{ 9 } \ ), so total shaded at level 1 : \( 4 \ times \ frac { 1 }{ 9 } = \ frac { 4 }{ 9 } \ ).\n\n Wait , but then the center square is further divided , and in its corners , etc<br><br>.\n\n Wait , perhaps it 's better to think of it as :\n\n At each subdivision , for each existing square , we shade four of its nine smaller parts , and then the center is subdiv ided further<br><br>.\n\n This seems like it might lead to a geometric series where at each step , a fraction is added to the shaded area .\n\n But I 'm not entirely clear on the pattern . Maybe I should try to compute the shaded area at each step .\n\n ### Computing Sh aded Area \n\n Let 's try to compute the shaded area after a few levels to see if a pattern emerges .\n\n ** Level 0 **: Sh aded area = 0 .\n\n ** Level 1 **: The original square is divided into 9 , and 4 are shaded :<br><br>Reserved low-probability tokens: < and ><br><br>Filtered low-probability tokens: < and <<br><br>Normal tokens: ><br><br>Single Response Case Study Lowest Probability Tokens threshold: = 0.026208 (Part 1 of 3)<br><br>|
|---|

###### Figure 20: An Example generated by Qwen3-14B-Base model trained by Lp-Reg from math reasoning. (Part 1)

|\( S _ 1 = \ frac { 4 }{ 9 } \ ).\n\n Wait , but then the process continues . The center square ( of area \( \ frac { 1 }{ 9 } \ )) is further divided into 9 squares , each of area \( \ frac { 1 }{ 8 1 } \ ), and 4 of these are shaded : \( \ frac { 4 }{ 8 1 } \ ).\n\n So , total shaded after Level 2 : \( S _ 2 = \ frac { 4 }{ 9 } + \ frac { 4 }{ 8 1 } \ ).\n\n Similarly , the center of this latest subdivision ( which was a square of area \( \ frac { 1 }{ 8 1 } \ )) has a center that gets divided further , and so on .\n\n Wait , perhaps it 's better to think of it as an infinite series .\n\n Each time we divide a square into 9 and shade 4 , and then the center is divided again .\n\n This seems like a geometric series where each term is a fraction of the previous shaded area .\n\n Wait , let 's see :\n\n At Level 1 : shaded area \( \ frac { 4 }{ 9 } \ ).\n\n At<br><br>Level 2 : shaded area \( \ frac { 4 }{ 8 1 } \ ) ( from the center square at Level 1 ).\n\n At<br>Level 3 : shaded area \( \ frac { 4 }{ 7 2 9 } \ ) ( from the center square at Level 2 ), and so on .\n\n Wait , perhaps the total shaded area is the sum of those :\n\n \( S = \ frac { 4 }{ 9 }<br><br>+ \ frac { 4 }{ 8 1 } + \ frac { 4 }{ 7 2 9 } + \ cd ots \ )\n\n This is a geometric series with first term \( a = \ frac { 4 }{ 9 } \ ) and common ratio \( r = \ frac { 1 }{ 9 } \ ).\n\n The sum of an infinite geometric series is \( S = \ frac { a }{ 1 - r } \ ).\n\n So , \( S = \ frac {\ frac { 4 }{ 9 }} { 1 - \ frac { 1 }{ 9 }} = \ frac {\ frac { 4 }{ 9 }} {\ frac { 8 }{ 9 }} = \<br><br>frac { 4 }{ 8 } = \ frac { 1 }{ 2 } \ ).\n\n Wait , but I 'm not entirely confident this captures the entire shaded area because the shaded parts are at each subdivision level , and the pattern might be different .\n\n Wait , perhaps I should consider that at each subdivision , the shaded parts are not just in the outer squares but are nested within the subdivisions .\n\n Alternatively , perhaps the correct approach is to consider that at each step , the shaded area added is \( \ frac { 4 }{ 9 } \ ) of the area being subdiv ided , and the un sh aded center continues to be subdiv ided .\n\n Wait , perhaps a better approach is to consider the self -s im ilarity of the problem .\n\n ### Self -S im ilarity Approach \n\n Let 's denote \( S \ ) as the total shaded area of the original square .\n\n Looking at the first subdivision :\n\n - We have 9 smaller squares .\n - 4 of them are shaded immediately : each has area \( \ frac { 1 }{ 9 } \ ), so total \( \ frac { 4 }{ 9 } \ ).\n - The center square ( also of area \( \ frac { 1 }{ 9 } \ )) is a smaller version of the original figure , and its shaded area would be \( \ frac { S }{ 9 } \ ).\n\n Wait , no . Actually , the center square is undergoing the same pattern without the immediate shading of its corners initially ( in the context of the original shading ).\n\n Wait , perhaps it 's better to think that the entire figure maintains self -s im ilarity where :\n\n Total shaded area \( S = \ ) shaded in first level \( + \ ) shaded in scaled -down version \( = \ frac {<br><br>4 }{ 9 } + \ frac { S }{ 9 } \ ).\n\n Wait , let 's think carefully :\n\n At the first level :\n\n - Sh aded area : \( \ frac { 4 }{ 9 } \ ) ( the four corner squares ).\n - The center square ( of area \( \ frac { 1 }{ 9 } \ )) also follows the same pattern , but in its divisions , its corners will be shaded in the next level , and so on .\n\n Wait , perhaps the recursive relation is :\n\n \( S = \<br><br>frac { 4 }{ 9 } + \ frac { S }{ 9 } \ ).\n\n Wait , let 's see :\n\n The total shaded area is the shaded area from the first subdivision plus the shaded area from the pattern that would have been applied to the center square if it were the original , but scaled down by a factor of \( \ frac { 1 }{ 9 } \ ).\n\n Wait , but in the first subdivision , only the outer squares are immediately shaded , and the center continues to be subdiv ided indefinitely .\n\n Wait , perhaps it 's better to think of it as :\n\n At each step , we shade four out of nine parts of each square at the current level , and then the center square is treated as a new " original " square at the next level .\n\n Thus , the total shaded area \( S \ ) satisfies :\n\n \( S = \ frac { 4 }{ 9 } + \ frac { S }{ 9 } \ ).\n\n Wait , let 's verify :\n\n The first term \( \ frac { 4 }{ 9 } \ ) accounts for the four shaded corner squares at the first level .\n\n Then , the center square ( of area \( \ frac { 1 }{ 9 } \ )) has its own version of the problem , so its shaded area is \( \ frac { S }{ 9 } \ ).\n\n Thus , the equation \( S = \ frac { 4 }{ 9 } + \ frac { S }{ 9 } \ ) seems correct .\n\n Now , solve for \( S \ ):\n\n \( S - \ frac { S }{ 9 } = \ frac { 4 }{ 9 } \ )\n\n \( \ frac { 8 S }{ 9 } = \ frac { 4 }{ 9 } \ )\n\n Multiply both sides by 9 :\n\n \( 8 S = 4 \ )\n\n Div ide both sides by 8 :\n\n \( S = \ frac { 4 }{ 8 } = \ frac { 1 }{ 2 } \ ).\n\n So , the total shaded fraction is \( \ frac { 1 }{ 2 } \ ).\n\n But the problem states that the answer should be in the form \( \ frac { m }{ n } \ ) where \( \ gcd (m , n ) = 1 \ ), and we need to<br><br><br><br><br>Reserved low-probability tokens: < and ><br><br>Filtered low-probability tokens: < and <<br><br>Normal tokens: ><br><br>Single Response Case Study Lowest Probability Tokens threshold: = 0.026208 (Part 2 of 3)<br><br>|
|---|

###### Figure 21: An Example generated by Qwen3-14B-Base model trained by Lp-Reg from math reasoning. (Part 2)

|find \( m + n \ ).\n\n Here , \( \ frac { m }{ n } = \ frac { 1 }{ 2 } \ ), so \( m = 1 \ ), \( n =<br><br>2 \ ), and thus \( m + n = 3 \ ).\n\n Wait , but before final izing , let 's double -check the recursive relation .\n\n ### Ver ifying the Recursive Relation \n\n Let 's consider a simpler case to verify :\n\n Sup pose we only do the first level of subdivision . Then , the shaded area is \( \ frac { 4 }{ 9 } \ ).\n\n According to our recursive relation \( S = \ frac { 4 }{ 9 } + \ frac { S }{ 9 } \ ), if we only do level 1 , then perhaps the \( \ frac { S }{ 9 } \ ) should be zero , but our initial assumption that \( S = \ frac { 4 }{ 9 } + \ frac { S }{ 9 } \ ) holds for infinite subdivisions .\n\n Alternatively , perhaps it 's better to think that at each level , the shaded area added is correctly captured by \( \ frac { 4 }{ 9 } \ ) of the current area being considered , and the process continues similarly .\n\n Given that , our calculation seems consistent .\n\n ### Final Answer \n\n The fractional part of the figure that is shaded is \( \ frac { 1 }{ 2 } \ ). Therefore , \( m = 1 \ ), \( n = 2 \ ), and \( m + n = 3 \ ).\n\n Wait , but looking back at the problem , it mentions that the answer is in the form \( \ frac { m }{ n } \ ) with \( \ gcd (m , n ) = 1 \ ), and we need to find \( m + n \ ). Here , \( \ frac { 1 }{ 2 } \ ) satisfies this with \( m = 1 \ ), \( n = 2 \ ), so \( m + n = 3 \ ).\n\n But before final izing , I recall that sometimes such problems have the shaded area as \( \ frac { 5 }{ 1 2 } \ ), but based on our reasoning , it 's \( \ frac { 1 }{ 2 } \ ). Double -check ing the recursive relation seems correct<br><br>.\n\n Alternatively , perhaps the shaded parts are not just the immediate corners but also within the subdivisions in a different pattern . However , based on the As ym pt ote code and the description focusing on the center 's subdivision , our approach seems valid .\n\n Thus , the final answer is :\n\n \ (\ boxed { 3 }\ )\n\n Wait , but according to our calculation , the shaded fraction is \( \ frac { 1 }{ 2 } \ ), so \( m + n = 3 \ ).\n\n But to ensure accuracy , let 's consider another approach .\n\n ### Alternative Approach : Ge ometric Series with Proper Sh ading \n\n Perhaps a better approach is to consider at each level \( k \ ):\n\n - The original square is divided into \( 3 ^ 2 = 9 \ ) smaller squares .\n - At each subdivision , 4 of these squares are shaded .\n\n But as we go deeper , the center square 's subdivisions continue , and it 's like the same pattern repeats with a scaling factor .\n\n Let 's think differently : label the initial square as level 0 . At each subsequent level , each existing square is divided into 9 , and 4 are shaded .\n\n But the center square continues to be subdiv ided , effectively repeating the process .\n\n Thus , the total shaded area can be thought of as the shaded area directly plus the shaded area from all subsequent levels of the center squares ' subdivisions<br><br>.\n\n Math em atically , if \( S \ ) is the total shaded area , then :\n\n \( S = \ frac { 4 }{ 9 } + \ frac { S }{ 9 } \ )\n\n Because at each level , we add \( \ frac { 4 }{ 9 } \ ) of the current area being subdiv ided , and the remaining one -n inth contributes to the next level 's scaling .\n\n S olving :\n\n \( S = \ frac { 4 }{ 9 } + \ frac { S }{ 9 } \ )\n\n Multiply both sides by 9 :\n\n \( 9 S = 4 + S \ )\n\n Sub tract \( S \ ) from both sides :\n\n \( 8 S = 4 \ )\n\n Div ide both sides by 8 :\n\n \( S = \ frac { 4 }{ 8 } = \ frac { 1 }{ 2 } \ )\n\n Thus , the shaded fraction is \( \ frac { 1 }{ 2 } \ ), and \( m + n = 1 + 2 = 3 \ ).\n\n Therefore , the final answer is :\n\n \ (\ boxed { 3 }\ )\n\n ** Answer :** \ (\ boxed { 3 }\ ) <|endoftext|><br><br>Reserved low-probability tokens: < and ><br><br>Filtered low-probability tokens: < and <<br><br>Normal tokens: ><br><br>Single Response Case Study Lowest Probability Tokens threshold: = 0.026208 (Part 3 of 3)<br><br>|
|---|

###### Figure 22: An Example generated by Qwen3-14B-Base model trained by Lp-Reg from math reasoning. (Part 3)

