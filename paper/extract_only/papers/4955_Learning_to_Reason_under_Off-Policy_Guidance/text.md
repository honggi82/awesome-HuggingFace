arXiv:2504.14945v5[cs.LG]22Jun2025

## Learning to Reason under Off-Policy Guidance

##### Jianhao Yan21∗ Yafu Li1∗ Zican Hu31 Zhi Wang3 Ganqu Cui1 Xiaoye Qu1

Yu Cheng4† Yue Zhang2† 1 Shanghai AI Laboratory 2 Westlake University 3 Nanjing University 4 The Chinese University of Hong Kong Corresponding to: chengyu@cse.cuhk.edu.hk, yue.zhang@wias.org.cn

Project Page: https://github.com/ElliottYan/LUFFY

#### Abstract

Recent advances in large reasoning models (LRMs) demonstrate that sophisticated behaviors such as multi-step reasoning and self-reflection can emerge via reinforcement learning with verifiable rewards (RLVR). However, existing RLVR approaches are inherently “on-policy”, limiting learning to a model’s own outputs and failing to acquire reasoning abilities beyond its initial capabilities. To address this issue, we introduce LUFFY (Learning to reason Under oFF-policY guidance), a framework that augments RLVR with off-policy reasoning traces. LUFFY dynamically balances imitation and exploration by combining off-policy demonstrations with on-policy rollouts during training. Specifically, LUFFY combines the MixedPolicy GRPO framework, which has a theoretically guaranteed convergence rate, alongside policy shaping via regularized importance sampling to avoid superficial and rigid imitation during mixed-policy training. Compared with previous RLVR methods, LUFFY achieves an over +6.4 average gain across six math benchmarks and an advantage of over +6.2 points in out-of-distribution tasks. Most significantly, we show that LUFFY successfully trains weak models in scenarios where on-policy RLVR completely fails. These results provide compelling evidence that LUFFY transcends the fundamental limitations of on-policy RLVR and demonstrates the great potential of utilizing off-policy guidance in RLVR.

#### 1 Introduction

Recent breakthroughs in large reasoning models, including OpenAI-o1 [1], DeepSeek-R1 [2], and Kimi-1.5 [3], have demonstrated remarkable capabilities in complex reasoning tasks. These models have shown unprecedented proficiency in generating extensive Chains-of-Thought (CoT, [4]) responses and exhibiting sophisticated behaviors, such as self-reflection and self-correction. Particularly noteworthy is how these achievements have been realized through reinforcement learning with purely verifiable rewards (RLVR), as demonstrated by recent efforts including Deepseek R1 [2, 5, 6, 7]. The emergence of long CoT reasoning and self-reflection capabilities through such straightforward reward mechanisms, termed the “aha moment”, represents a significant advancement in the field.

Nevertheless, the reinforcement learning methods behind the success have a fundamental limitation worth highlighting: it is inherently on-policy, constraining learning exclusively to the model’s selfgenerated outputs through iterative trials and feedback cycles. Despite showing promising results, on-policy RL is bounded by the base LLM itself [8, 9]. In essence, reinforcement learning under this setting amplifies existing behaviors rather than introducing genuinely novel cognitive capacities. Recent study [10] corroborates this constraint, demonstrating that models like Llama 3.2 [11] quickly reach performance plateaus under RL training precisely because they lack certain foundational

∗ Equal contributions. Work was done during Jianhao Yan’s internship at Shanghai AI Laboratory. Yafu Li is the Project Lead.

† Corresponding authors.

Preprint. Under review.

cognitive behaviors necessary for further advancement. This inherent limitation provokes critical questions about the effectiveness and scope of RL for reasoning: How can we empower LLMs to acquire reasoning behaviors surpassing their initial cognitive boundaries?

In this paper, we introduce LUFFY: Learning to reason Under oFF-policY guidance, addressing this issue by introducing external guidance from a stronger policy (e.g., from DeepSeek-R1). The strong policy serves as guidance for diverging the training trajectory beyond the limitations of the model’s initial capabilities. Unlike on-policy RL, where the model can only learn from its own generations, our approach leverages off-policy learning to expose the model to reasoning patterns and cognitive structures that might otherwise remain inaccessible. This external guidance functions as a form of cognitive scaffolding, allowing the model to observe and internalize reasoning strategies from a more capable teacher model, thereby expanding its reasoning repertoire beyond what self-improvement alone could achieve.

In particular, LUFFY extends on GRPO [12] to Mixed-Policy GRPO by introducing a new offpolicy objective with importance sampling to calibrate policy gradient, and combining off-policy reasoning traces with models’ on-policy roll-outs during advantage computation, as illustrated in Figure 1. Intuitively, since off-policy traces consistently obtain positive rewards, LUFFY enables the model to selectively imitate these high-quality reasoning traces when its own roll-outs fail to achieve correctness, while preserving the capacity for self-driven exploration whenever its generated reasoning steps are successful. In this way, LUFFY achieves a dynamic and adaptive equilibrium between imitation and exploration. To avoid overly rapid convergence and entropy collapse, causing the model to latch onto superficial patterns rather than acquiring genuine reasoning capabilities, we further introduce policy shaping via regularized importance sampling, which amplifies learning signals for low-probability yet crucial actions under off-policy guidance. This mechanism encourages the model to preserve exploration throughout training, ultimately enabling it to internalize deeper and more generalizable reasoning behaviors.

LUFFY achieves significant improvements of +6.4 points on average compared with previous RLVR methods, across AIME24/25 [13], AMC [13], OlympiadBench [14], Minerva [15], and MATH500 [16] benchmarks, establishing the effectiveness of off-policy learning in RLVR paradigms. Moreover, LUFFY demonstrates superior generalization capability, i.e., an advantage of over +6.2 points on average, on out-of-distribution tasks, where other off-policy methods fall short. Critically, we show that LUFFY successfully trains weak foundation models, i.e., LLaMA3.1-8B, while onpolicy RL fails, providing evidence of LUFFY transcending the limitation of model capacity. Our in-depth analyses demonstrate that LUFFY encourages the model to imitate high-quality reasoning traces while maintaining exploration of its own sampling space, resulting in more reliable and generalizable reasoning capabilities.

Our contributions can be summarized as:

- • We introduce LUFFY, an approach that incorporates off-policy guidance into GRPO and integrates policy shaping through regularized importance sampling to address entropy collapse, effectively transcending the limitations of on-policy RL. (Sec. 3)
- • We empirically demonstrate LUFFY’s effectiveness across various foundation models, achieving an average gain of +6.4 points across six math benchmarks and +6.2 points on out-of-distribution tasks against previous RLVR methods, establishing a new state-of-the-art on RLVR with Qwen2.5-Math-7B. (Sec. 5.1)
- • We demonstrate that LUFFY successfully trains weaker foundation models where On-Policy RL fails. Specifically, while On-Policy RL can only train Llama3.1-8B on simplified datasets, LUFFY effectively trains these models across varying difficulty levels, overcoming capability-based limitations (Sec. 5.2).

#### 2 Reinforcement Learning with Verifiable Rewards

Verifiable Reward Function. The verifiable reward emphasizes the comparison between the extracted answer from the models’ output and the predefined golden answer. For instance, the model is instructed to output the final answer in a certain format, e.g., \boxed{}, and a regex function is used to extract the answer from \boxed{}. Formally, given a model’s output τ to question q, the

Solution Advantage Probability

q ModelPolicy

Verifier & Group Compute ...

... ... &

Off-policy Model

Policy Shaping

Solution Advantage Probability

q ModelPolicy

Verifier & Group Compute ...

... ... &

Off-policy Model

Policy Shaping

###### Solution Advantage Probability

|Policy Model|
|---|

|Verifier & Group Compute| |
|---|---|
| | |

q

... ... &

...

Off-policy Model

|Policy Shaping|
|---|

- Figure 1: Overview: LUFFY integrates off-policy reasoning traces into reinforcement learning by combining them with on-policy rollouts. Policy shaping emphasizes low-probability but crucial actions, enabling a balance between imitation and exploration for more generalizable reasoning. reward is defined by,

1 if τ outputs the correct final answer to q 0 otherwise

(1)

R(τ) =

This reward design avoids the risk of reward hacking [17, 18, 19] to a great extent and thus leads to successful scaling of RL training [2].

Group Relative Policy Optimization (GRPO). GRPO [12] shows exceptional performance in various tasks, especially to enable effective scaling within the RLVR paradigm [2, 20, 6]. It uses the reward scores of N sampled solutions from a query to estimate the advantage and thus remove the need for an additional value model. Formally, we denote the policy model before and after the update as πθ

and πθ, where both represent probability distributions over possible actions/tokens at each position. Given a question q, a set of sampled solutions τi generated by πθ

old

, and the reward function R(·), the advantage Ai of each in GRPO is computed by normalized rewards inside the group,

old

R(τi) − mean({R(τi) | τi ∼ πθ

###### (τ),i = 1,2,...,N}) std({R(τi) | τi ∼ πθ

, (2)

old

Ai =

(τ),i = 1,2,...,N})

old

Then, the RL objective is inherited from the clipped RL objective proposed by PPO [21],

|τi|

N

1

CLIP(ri,t(θ),Ai,ϵ) − β · DKL[πθ∥πref]. (3)

JGRPO(θ) =

N i=1 |τi|

t=1

i=1

where ri,t(θ) = πθ(τi,t|q,τi,<t)/πθ

(τi,t|q,τi,<t) is a importance sampling term to calibrate the gradient based on policy gradient theory [22], as the solutions are generated by πθ

old

instead of πθ.

old

The KL divergence DKL and the clipped surrogate objective CLIP(r,A,ϵ) = min[r · A,clip(r;1 − ϵ,1 + ϵ) · A] empirically ensures that the current policy πθ is within the trust region [23] of the old policy πθ

. We loosely categorize this method as On-Policy RL, indicating that the model is optimized using samples drawn from distributions closely aligned with its current policy. Nevertheless, recent practices [24, 25, 7] have increasingly omitted the KL divergence term, making these methods somewhat less “On-Policy”.

old

#### 3 Learning to Reason under Off-Policy Guidance

To facilitate exploration beyond the model’s own capabilities, we incorporate off-policy guidance, i.e., off-the-shelf reasoning trajectories generated by a stronger reasoning model such as Deepseek R1, into the RLVR learning. We expect the model to learn generalizable knowledge from off-policy beyond superficial imitation and maintain effective and efficient exploration as in RLVR training. In the following sections, we introduce LUFFY: mixed-policy GRPO (§ 3.1) with policy shaping (§3.2). The former one adaptively integrates off-policy trajectories into advantage estimation while the latter encourages continuous exploration throughout training.

##### 3.1 Mixed-Policy GRPO

We incorporate off-policy rollouts in GRPO by adding them directly to the group of on-policy rollouts generated by the model itself. Provided an off-policy distribution πϕ, this would affect the advantage computations in the following way,

R(τi) − mean(Gon ∪ Goff) std(Gon ∪ Goff)

Aˆi =

, (4)

(τ),i = 1,2,...,Non} and Goff = {R(τj) | τj ∼ πϕ(τ),j = 1,2,...,Noff}. As the quality of off-policy rollouts is high (yielding high rewards), this group computation naturally assigns a higher advantage to off-policy rollouts when the model struggles to generate correct solutions independently. Once the model begins producing successful reasoning traces, on-policy rollouts take precedence, thereby encouraging self-driven exploration.

where Gon = {R(τi) | τi ∼ πθ

old

Extending from the GRPO objective (Eq. 3), we introduce the off-policy objective to incorporate off-policy rollouts. Similar to GRPO and PPO [21, 12], we introduce an importance sampling term rˆj,t(θ,ϕ) = πθ(τj,t|q,τj,<t)/πϕ(τj,t|q,τj,<t) to calibrate gradient estimates [22]. We refer to this approach as Mixed-Policy GRPO:

1 Z

JMixed(θ) =

|τj|

Noff

CLIP(ˆrj,t(θ,ϕ),Aˆj,ϵ)

(

+

t=1

j=1

off-policy objective

|τi|

Non

CLIP(ri,t(θ),Aˆi,ϵ)

,

t=1

i=1

on-policy objective

(5)

where rˆj,t(θ,ϕ) = πθ(τj,t|q,τj,<t)/πϕ(τj,t|q,τj,<t) and ri,t(θ) = πθ(τi,t|q,τi,<t)/πθ

(τi,t|q,τi,<t). Z = Nj=1off |τj| + Ni=1on |τi| is the normalization factor.

old

The newly introduced importance sampling term rˆj,t contrasts with the importance sampling term ri,t used in on-policy RL (Eq. 3), where the denominator corresponds to the pre-update roll-out model

policy πθ

. Since the divergence between πθ and πθ

is typically much smaller than that between

old

old

πθ and the off-policy policy πϕ, the off-policy importance sampling ratio rˆj,t tends to be smaller, serving to calibrate gradient estimates from a distinct distribution.

Based on the theoretical analysis of stochastic gradient descent in nonconvex optimization [26], we give a convergence analysis in Theorem 1 to show that our importance-weighted policy gradient estimator in Eq. (5) stabilizes and converges to a stationary point, and the convergence rate is O(1/

√

K), where K is the total number of iterations. The proof can be found in Appendix B.1.

Theorem 1. Suppose the objective function of the policy gradient algorithm J ∈ Jn, where Jn is the class of finite-sum Lipschitz smooth functions, has σ-bounded gradients, and the importance weight

√

∗)−J(θ0)) Lσ2ww , and θ∗ is an optimal solution. Then, the iterates of our algorithm in Eq. (5) satisfy:

K where c = 2(J(θ

w = πθ/πϕ is clipped to be bounded by [w,w]. Let αk = α = c/

E[||∇J(θk)||2] ≤

min

0≤k≤K−1

2(J(θ∗) − J(θ0))Lw Kw

σ.

The importance sampling ratio in off-policy learning typically involves πϕ, representing the behavior policy’s probability in off-policy trajectories [27]. Theoretically, our derivations and guarantees

hold for any well-defined πϕ distribution. In practice, to facilitate direct integration of high-quality demonstrations from large, powerful models (e.g., DeepSeek-R1), we adopt πϕ = 1 for computational efficiency. This practical choice not only avoids the complexity caused by different tokenization between the on-policy and off-policy models. It also facilitates the easy incorporation of off-the-shelf datasets without recomputation of πϕ, as well as preserving theoretical guarantees. We omit the clip operation for the off-policy rollouts, as the clip operation will be imbalanced when πϕ = 1.

##### 3.2 Policy Shaping via Regularized Importance Sampling

While Mixed-Policy GRPO incorporates off-policy guidance successfully via group computation and importance sampling, a new practical challenge emerges: Mixed-Policy GRPO accelerates

###### Entropy during Training

###### Loss Weighting

Gradient Weighting

0.6

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.25

1.0

On-policy

Mixed-policy

0.5

0.20

0.8

Mixed-policy + Shaping

0.4

###### Entropy

Weights

Weights

0.15

0.6

0.3

0.10

0.4

0.2

0.05

0.2

0.1

On-policy/Mixed-policy Mixed-policy + Shaping

Entropy Collapse

0.00

0.0

0.0

0.0 0.2 0.4 0.6 0.8 1.0 Probability

0.0 0.2 0.4 0.6 0.8 1.0 Probability

0 100 200 300 400 500

Steps

- Figure 2: Left: generation entropy during training; Middle: shaping function value w.r.t. action probability; Right: gradient weights w.r.t. action probability.

convergence but significantly reduces exploration (Figure 2, left). Specifically, entropy collapses much faster than in on-policy RL, indicating increasingly deterministic rollouts and a diminished capacity for exploring diverse reasoning trajectories.

This originates from the “hacking” of the Mixed-Policy objective. When combining both learning off-policy and on-policy signals, the model tends to quickly converge toward reinforcing off-policy tokens that are also likely in the on-policy πθ distribution, and ignoring off-policy tokens that are deviated from the model’s original policy, i.e., low-probability tokens that may represent essential reasoning capabilities the model has yet to acquire. We empirically analyze this issue in detail in Section 5.3.

To address this issue, we introduce policy shaping via regularized importance sampling, a technique that re-weights the gradient of off-policy distributions to enhance learning from low-probability tokens. In particular, our approach replaces the importance sampling ratio rˆj,t(θ,ϕ) with f(ˆrj,t(θ,ϕ)), where f(·) represents a transformation function that alters the dynamics between off-policy and on-policy distributions, thereby increasing gradient emphasis on tokens with low probability in the model’s standard distribution. Recall that we omit the clip operation for the off-policy rollouts. The loss function with policy shaping can be written as below:

1 Z

JSHAPING(θ) =

Noff

(

j=1

|τj|

Non

f(ˆrj,t(θ,ϕ)) · Aˆj +

t=1

i=1

|τi|

CLIP(ri,t(θ),Aˆi,ϵ), (6)

t=1

To further illustrate the meaning of shaping function f, we derive the gradient of off-policy objective,

πθ πϕ ∇θ log πθ · Aˆj importance sampling

[f′(πθ)

∇θJSHAPING-OFF(θ) = Eτ∼π

.]

ϕ

(7)

We write π(τj,t|q,τj,<t) as π for simplicity. From Eq. 7, we can see f′(πθ) is a weighting function of the gradient. The vanilla mixed-policy GRPO can be regarded as using a linear shaping function,

i.e., f(π) = π, where the original importance sampling ratio πθ/πϕ is applied. We decompose the log probability and derive the gradient on each output logit (action):

∂JSHAPING-OFF(θ) ∂Mθ(τ′ j,t)

f′(πθ)πθ I(τj,t′ = τj,t) − πθ · Aˆj

= Eτ∼π

ϕ

(8)

∂JSHAPING-OFF(θ) ∂Mθ(τ′ j,t) | ≤ Eτ∼π

ϕ |f′(πθ)|πθ (1 − πθ) · |Aˆj| ,

⇒|

where τj,t′ is any possible action/token on in the action space at the j-th trajectory and t-th position, and Mθ(τ) denotes the logits of that action. The identity case represents the gradient when the action is the off-policy action τ = τ′, which elevates the probability of predicting the off-policy action. From Eq. 8, if f(πθ) = πθ and thus f′(πθ) = 1, we can see that the scale of gradient is upper-bounded by πθ(1 − πθ), leading to small values when πθ → 0 and πθ → 1. We opt for f(x) = x/(x + γ) as our shaping function (middle part of Figure 2), where γ is set as 0.1. As shown in Figure 2 (Right), the shaping function reweights the gradients to assign more importance to low-probability actions, thereby improving learning from unfamiliar yet effective decisions from the off-policy traces.

Further, we provide an informal analysis on the variance of our importance weights regularized by f(·) in Appendix B.2. With a first-order approximation and a special case derivation, we show that a smaller variance can be achieved for the sampling weights, thus proving more stable training for leveraging off-policy guidance.

#### 4 Experimental Setup

Dataset Construction. Our training set is a subset of OpenR1-Math-220k [28] 3, of which the prompts are collected from NuminaMath 1.5 [29], and the off-policy reasoning traces are generated by Deepseek-R1 [2]. We use the default subset, which contains 94k prompts, and we filter out generations that are longer than 8192 tokens and those that are verified wrong by Math-verify4, resulting in 45k prompts and off-policy reasoning traces.

RL Practice. We remove the KL loss term by setting β = 0 and set the entropy loss coefficient to 0.01. Following Dr.GRPO[6], we remove the length normalization and standard error normalization of GRPO loss (Eq. 3) for all experiments. For policy shaping, we empirically set the γ as 0.1 and study the value of γ in Appendix E.4. Our rollout batch size is 128, and the update batch size is 64. We use 8 rollouts per prompt. Specifically, for on-policy RL, we use 8 on-policy rollouts. For our methods, we use 1 off-policy and 7 on-policy rollouts to ensure fairness. We use temperature=1.0 for rollout generation. We use Math-Verify as our reward function and include no format or length reward. We use Qwen2.5-Math-7B [30] by default, following previous work [24, 5, 6]. In addition, we extend LUFFY to Qwen2.5-Math-1.5B [30] and Qwen2.5-Instruct-7B [31], and LLaMA 3.1-8B [32].

Evaluation. For evaluation, we mainly focus on six widely used math reasoning benchmarks, including AIME 2024, AIME 2025, AMC [13], Minerva [15], OlympiadBench [14], and MATH500 [16]. For AIME 2024, AIME 2025 and AMC, we report avg@32 as the test set is relatively small, and for the other three benchmarks, we report pass@1. As our RL training mainly focuses on math reasoning, we further validate the generalization capability on three out-of-distribution benchmarks, namely ARC-c [33](Open-Domain Reasoning), GPQA-diamond [34](Science Graduate Knowledge, denoted as GPQA∗), and MMLU-Pro [35](Reasoning-focused Questions from Academic Exams and Textbooks). We shuffle the multiple-choice options to avoid contamination. For testing, the temperature is set as 0.6.

Baseline Methods. For RLVR methods, we consider the following methods: (1) Simple-RL [5]: training from Qwen2.5-Math-7B using rule-based reward; (2) Oat-Zero [6]: training from Qwen2.5Math-7B and rule-based reward, proposing to remove the standard deviation in GRPO advantage computation and token-level normalization in policy loss computation; (3) PRIME-Zero [24]: using policy rollouts and outcome labels through implict process rewards; (4) OpenReasonerZero [7]: a recent open-source implementation of RLVR methods. Except RLVR approaches from previous work, we consider two kinds of baselines with our setting (1) On-Policy RL – we train on-policy RL within RLVR paradigm using Dr.GRPO with the same reward and data. (2) Alternative Methods to Incorporate Off-Policy Guidance – We consider three methods, namely SFT, we train the model with the same prompts and reasoning traces as LUFFY using SFT; RL w/ SFT Loss, using SFT loss during RL training; SFT + RL, two-stage training that continues RL training after SFT. For detailed setup for training these methods, we refer readers to Appendix C.

#### 5 Experimental Results

##### 5.1 Main Results

SOTA performance on RLVR with Qwen2.5-Math-7B. Our main results are presented in Table 1. We first compare LUFFY against other RLVR methods and our RLVR replication. All prior methods are built upon Qwen2.5-Math-7B base models, differing in dataset composition (source and difficulty) and optimization strategies, e.g., removing length and standard error normalization [6] or incorporating process-level rewards [24]. Evaluated on six challenging competition-level benchmarks, LUFFY achieves an average score of 50.1, significantly outperforming existing RLVR methods by a substantial margin of +6.4 points, establishing a new state-of-the-art. Notably, while LUFFY exhibits comparable performance in AIME 24, it demonstrates a significantly greater advantage on the newly

- 3https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- 4https://github.com/huggingface/Math-Verify

Table 1: Overall in-distribution and out-of-distribution performance based on Qwen2.5-Math-7B. We compare with the following baselines: (1) Qwen2.5-Math-7B-Instruct (Qwen-Instruct), (2) prior RLVR approaches, (3) our replication of on-policy RL, and (4) alternative off-policy learning methods. All models are evaluated under a unified setting. LUFFY† denotes training with extra steps (Table 2). Bold and underline indicate the best and second-best results, respectively. ∗ represents significantly better than baselines (p < 0.05).

In-Distribution Performance Out-of-Distribution Performance AIME 24/25 AMC MATH-500 Minerva Olympiad Avg. ARC-c GPQA∗ MMLU-Pro Avg.

Model

Qwen-Base [30] 11.5/4.9 31.3 43.6 7.4 15.6 19.0 18.2 11.1 16.9 15.4 Qwen-Instruct [30] 12.5/10.2 48.5 80.4 32.7 41.0 37.6 70.3 24.7 34.1 43.0

Previous RLVR methods

SimpleRL-Zero [5] 27.0/6.8 54.9 76.0 25.0 34.7 37.4 30.2 23.2 34.5 29.3 OpenReasoner-Zero [7] 16.5/15.0 52.1 82.4 33.1 47.1 41.0 66.2 29.8 58.7 51.6 PRIME-Zero [24] 17.0/12.8 54.0 81.4 39.0 40.3 40.7 73.3 18.2 32.7 41.4 Oat-Zero [6] 33.4/11.9 61.2 78.0 34.6 43.4 43.7 70.1 23.7 41.7 45.2

Our On-policy RLVR Replication On-Policy RL 25.1/15.3 62.0 84.4 39.3 46.8 45.5 82.3 40.4 49.3 57.3 Alternative Off-policy Learning Methods

SFT 22.2/22.3 52.8 82.6 40.8 43.7 44.1 75.2 24.7 42.7 47.5 RL w/ SFT Loss 19.5/16.4 49.7 80.4 34.9 39.4 40.1 71.2 23.7 43.2 46.0 SFT+RL 25.8/23.1 62.7 87.2 39.7 50.4 48.2 72.4 24.2 37.7 44.8

Our Methods

LUFFY 29.4/23.1 65.6 87.6 37.5 57.2 50.1* 80.5 39.9 53.0 57.8* LUFFY† 30.7/22.5 66.2 86.8 41.2 55.3 50.4* 81.8 49.0 54.7 61.8*

released AIME 25 test set (+8.1), demonstrating its generalization to internalize nuanced reasoning behaviors from off-policy traces. Compared to On-Policy RL, LUFFY improves performance by +4.6 points on average, demonstrating the benefit of integrating high-quality off-policy traces.

Regarding out-of-distribution performance, LUFFY also demonstrates strong performance gain. Over three challenging out-of-distribution benchmarks, LUFFY achieves an average score of 57.8 and outperforms the best RLVR method OpenReasoner-Zero for +6.2 points. These findings underscore the effectiveness of LUFFY in leveraging off-policy reasoning guidance for enhanced generalization across diverse, out-of-distribution tasks.

Comparing against other Off-Policy Learning Methods. Comparing LUFFY against alternative methods to incorporate Off-Policy Guidance, we can see that LUFFY beats all three offpolicy baselines in in-distribution math reasoning tasks and achieves substantial improvements over OOD tasks (+10.3 points). Compared to SFT+RL, LUFFY is advantageous in both in-distribution tasks (+1.9 points) and out-ofdistribution tasks (+16.1), with only 59% GPU hours and much less off-policy data usage (Table 2). The additional GPU hours in SFT+RL and RL w/ SFT Loss are largely attributed to excessively long generations induced by rigid imitation from SFT (Appendix F), which substantially increase the computational overhead during the RL roll-out stage. With matching GPU hours, LUFFY† further enlarges the advantage, providing a more robust and effective alternative for distilling knowledge from stronger LRMs, except for supervised fine-tuning [2, 36, 37]. In

##### Table 2: Comparison of resource requirements between LUFFY and other off-policy methods.

Model GPU Hours Data Usage (On/Off) LUFFY 77 × 8 64K × 7 / 64K LUFFY† 130 × 8 110K × 7 / 110K SFT 24 × 8 0 / 64K RL w/ SFT Loss 133 × 8 64K × 7 / 64K SFT+RL 130 × 8 64K × 8 / 135K

Instruct

SFT

On-policy RL

LUFFY

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### Qwen2.5-Math-1.5B

###### Llama-3.1-8B

Qwen2.5-7B-Instruct

40.7

38.0

40

35.7

35.2

34.4

AverageScore

31.9

30.0

29.0

30

20

17.1

13.2

9.6

10

5.9

0

Figure 3: Average performance on six mathematical reasoning benchmarks of LUFFY on different backbones (Details in Appendix E.2)

On-policy RL LUFFY Oﬀ-policy Reasoning Traces

0.8

0.6

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| |Adapt|ing to|Externa|l Guida|nce|
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| |Imi|tating|ﬀ-poli|cy Trac|es|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | |Con|tinuous|Explor|ation|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

###### OutcomeTrainingRewards

4000

0.7

0.5

###### ResponseLength

3500

0.6

0.4

Entropy

3000

0.5

0.3

2500

0.4

2000

0.2

0.3

1500

0.1

0.2

1000

0.1

0.0

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

Steps

Steps

Steps

- Figure 5: Training dynamics of LUFFY compared with on-policy RL. Left: outcome training rewards; Middle: generation length; Right: generation entropy.

particular, we notice the SFT training causes the model to learn superficial and rigid imitation of off-policy traces, and costs the model on out-of-distribution performance, while LUFFY selectively and strategically learn from off-policy traces (Sec. 5.3 and App. F) to enhance its own policy rollouts.

Extending LUFFY to More Models. We further investigate whether LUFFY can be applied to small models, instruction-tuned models, or weak models. To answer this question, we train LUFFY on three more models, i.e., Qwen2.5-Math-1.5B (small models), Qwen2.5-Instruct-7B (instruction-tuned models), and LLaMA-3.1-8B (weak models), and compare with their respective Instruct models (black bar in Figure 3). LUFFY achieves consistent and substantial improvements, surpassing both SFT and On-Policy RL for all three models, demonstrating the general applicability of LUFFY. Specifically, LUFFY improves over on-policy RL for +8.0 points on Qwen2.5-Math-1.5B, +3.6 points on Llama-3.1-8B, and +5.5 points on Qwen2.5-Instruct-7B.

##### 5.2 LUFFY Succeeds Where On-Policy Fails

Easy Training Set

TrainingRewards

0.4

More interestingly, we observe that LUFFY can successfully train models in scenarios where onpolicy RLVR fails. We conduct experiments using LLaMA-3.1-8B on two subsets of varying difficulty (Easy and Hard), with details provided in Appendix C. As shown in Figure 4, on-policy reinforcement learning performs well on the Easy subset but fails on the Hard subset, where training rewards collapse to zero, since on-policy rollouts struggle to obtain positive feedback signals. In contrast, LUFFY achieves stable reward improvements on both datasets, highlighting its robustness and its ability to overcome limitations imposed by model capacity.

0.2

Llama-On-Policy-Easy

Llama-LUFFY-Easy

0.0

Steps

Hard Training Set

TrainingRewards

0.2

0.1

Llama-On-Policy-Hard

Llama-LUFFY-Hard

0.0

0 100 200 300 400 500

Steps

Figure 4: Training rewards of LLaMA-3.1 8B on the Easy and Hard training set.

##### 5.3 Training Dynamics

Strategically Learning from Guidance. Figure 5 illustrate the training dynamics regarding training rewards, generation length and entropy for On-Policy RL and LUFFY. Initially, LUFFY primarily imitates off-policy trajectories, as indicated by the increasing generation length gradually aligning with the off-policy reasoning traces (middle part of Figure 5). At this early stage, imitation dominates, causing an initial performance dip (left part of Figure 5) as the model adjusts to external guidance and potentially sophisticated cognitive behaviors [38]. As training progresses, on-policy rollouts gradually become more prominent, fostering independent exploration within the model’s own sampling space while effectively retaining insights gained from off-policy demonstrations. This guided exploration brings growing advantages (training rewards) over On-Policy RL. uently, LUFFY achieves a dynamic balance between imitation and exploration, leading to more effective off-policy learning (Section F). These results highlight that LUFFY selectively adopts valuable reasoning patterns rather than blindly

imitating off-policy traces. Such strategic off-policy learning is further evidenced in reasoning behaviors during inference, such as generation length and exploration (Appendix F).

Maintaining Exploration. Figure 5 (Right) illustrates that LUFFY consistently sustains higher entropy compared to On-Policy RL throughout the entire training process. Specifically, the generation entropy of On-Policy RL rapidly converges to nearly zero after approximately 200 steps, indicating a highly deterministic policy with limited exploration potential. Conversely, the elevated entropy observed in LUFFY allows continuous exploration of less confident yet potentially superior policies, facilitating the discovery and learning of novel cognitive behaviors. Interestingly, we observe entropy fluctuations and even occasional increases, such as between steps 200 and 250, reflecting ongoing exploration of low-probability but crucial actions, also referred to as pivotal tokens [39, 25]. This strategic exploration enables the model to escape local optima, thus improving its convergence towards more globally optimal solutions.

##### 5.4 Policy Shaping Encourages Continuous Exploration

55

- Figure 6 illustrates the validation performance over the course of training, shedding light on the impact of policy shaping. Mixed-Policy achieves rapid early gains, significantly outperforming On-Policy RL at the start. However, its performance soon plateaus and eventually converges with On-Policy RL, whereas LUFFY continues to improve steadily. These trends are consistent with our earlier analysis of entropy collapse (Section 3.2), underscoring the role of policy shaping as an effective regularizer that prevents premature convergence and sustains performance gains in later stages of training.

50

###### ValidationAccuracy(%)

0.8

45

0.6

Entropy

40

On-Policy RL

Mixed-Policy RL

35

Mixed-Policy RL + Shaping (LUFFY)

0.4

30

0.2

25

20

0.0

0 100 200 300 400 500

Steps

#### 6 Related Work

Figure 6: Effects of policy shaping.

RL for LRMs Recent advances have demonstrated remarkable progress in enhancing LLMs’ reasoning capabilities through RL approaches [2, 1, 3, 40], including DeepSeek-R1, OpenAIo1, and Kimi-1.5. Subsequent work systematically investigate RL with purely verifiable rewards (RLVR) [5, 7, 24, 6, 41], providing insights into how this approach enables complex reasoning. Various advances have been proposed for reasoning enhancement. Test-time adaptation mechanisms [36, 42] demonstrate potential in dynamic optimization, but remain bounded by inherent model knowledge. While structured reasoning approaches [43, 37, 44, 45] have demonstrated that complex reasoning capabilities can emerge from strategically designed prompting and training strategies. The development of RL optimization techniques [46, 47, 48, 6] has contributed novel training paradigms and optimization objectives that specifically target the enhancement of reasoning capabilities. However, recent studies [8, 9] demonstrate that on-policy learning is limited by vast exploration space and primarily amplifies existing behaviors. Existing approaches optimize within model boundaries rather than expanding reasoning horizons. Our approach leverages off-policy reasoning traces to transcend these cognitive constraints while preserving self-driven exploration capabilities.

On-Policy and Off-Policy RL Reinforcement learning algorithms are fundamentally distinguished by their approach to experience utilization during policy optimization. On-policy methods (e.g., TRPO [23], A2C/A3C [49], PPO [21]) strictly update using trajectories from the current policy, ensuring training stability but potentially constraining the exploration space. In contrast, off-policy algorithms (e.g., DQN [50],TD3 [51], SAC [52]) leverage experiences from diverse policies, offering superior sample efficiency while introducing optimization challenges due to distribution shift. Extending to LLM training, on-policy methods are more commonly adopted, with approaches like GRPO [12], REINFORCE [53], and PPO [21] demonstrating strong performance through various optimization techniques. PRIME [24] and NFT [54] models the implicit policy to utilize self-generated answers. Meanwhile, off-policy approaches such as DPO [55] offer alternative optimization frameworks by reformulating preference learning as classification. To leverage advantages from both paradigms, our work bridges these approaches through policy shaping with regularized importance sampling, effectively combining on-policy optimization with off-policy guidance.

#### 7 Conclusion

We presented LUFFY, a simple yet powerful framework that integrates off-policy reasoning guidance into the RLVR paradigm. By dynamically balancing imitation and exploration, LUFFY effectively leverages external reasoning traces without sacrificing the model’s ability to discover novel solutions. Our method outperforms strong baselines across competitive math benchmarks and generalizes robustly to out-of-distribution tasks, surpassing both on-policy RLVR and off-policy baselines. These results highlight the promise of off-policy learning as a scalable and principled path toward building more general, capable, and self-improving reasoning models. Future work may focus on extending LUFFY to broader domains or modalities [56] and further refining policy shaping to maximize exploration under off-policy guidance.

#### References

- [1] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [2] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [3] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [4] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.
- [5] Weihao Zeng, Yuzhen Huang, Wei Liu, Keqing He, Qian Liu, Zejun Ma, and Junxian He. 7b model and 8k examples: Emerging reasoning with reinforcement learning is both effective and efficient. https://hkust-nlp.notion.site/simplerl-reason, 2025. Notion Blog.
- [6] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [7] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model, 2025.
- [8] Rosie Zhao, Alexandru Meterez, Sham Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach. Echo chamber: Rl post-training amplifies behaviors learned in pretraining, 2025.
- [9] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?, 2025.
- [10] Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D. Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars, 2025.
- [11] Meta AI. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models, September 2024. 15 minute read.
- [12] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.
- [13] Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q. Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. https: //huggingface.co/datasets/Numinamath, 2024. Hugging Face repository, 13:9.
- [14] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, 2024.
- [15] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.
- [16] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

- [17] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016.
- [18] Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460– 9471, 2022.
- [19] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.
- [20] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025.
- [21] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [22] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.
- [23] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.
- [24] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.
- [25] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025.
- [26] Sashank J Reddi, Ahmed Hefny, Suvrit Sra, Barnabas Poczos, and Alex Smola. Stochastic variance reduction for nonconvex optimization. In ICML, pages 314–323, 2016.
- [27] Wenjia Meng, Qian Zheng, Gang Pan, and Yilong Yin. Off-policy proximal policy optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 9162–9170,

- 2023.

- [28] Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025.
- [29] Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https://huggingface.co/AI-MO/NuminaMath-1.5](https://github.com/ project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf),

- 2024.

- [30] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement, 2024.
- [31] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [32] Meta Team. The llama 3 herd of models, 2024.
- [33] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

- [34] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.
- [35] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024.
- [36] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.
- [37] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025.
- [38] Xiao Hu, Xingyu Lu, Liyuan Mao, YiFan Zhang, Tianke Zhang, Bin Wen, Fan Yang, Tingting Gao, and Guorui Zhou. Why distillation can outperform zero-rl: The role of flexible reasoning, 2025.
- [39] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, and Yi Zhang. Phi-4 technical report, 2024.
- [40] Xiaoye Qu, Yafu Li, Zhaochen Su, Weigao Sun, Jianhao Yan, Dongrui Liu, Ganqu Cui, Daizong Liu, Shuxian Liang, Junxian He, Peng Li, Wei Wei, Jing Shao, Chaochao Lu, Yue Zhang, XianSheng Hua, Bowen Zhou, and Yu Cheng. A survey of efficient reasoning for large reasoning models: Language, multimodality, and beyond, 2025.
- [41] Yaru Hao, Li Dong, Xun Wu, Shaohan Huang, Zewen Chi, and Furu Wei. On-policy rl with optimal reward baseline, 2025.
- [42] Yuxin Zuo, Kaiyan Zhang, Shang Qu, Li Sheng, Xuekai Zhu, Biqing Qi, Youbang Sun, Ganqu Cui, Ning Ding, and Bowen Zhou. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.
- [43] Bo Pang, Hanze Dong, Jiacheng Xu, Silvio Savarese, Yingbo Zhou, and Caiming Xiong. Bolt: Bootstrap long chain-of-thought in language models without distillation. arXiv preprint arXiv:2502.03860, 2025.
- [44] Mehdi Fatemi, Banafsheh Rafiee, Mingjie Tang, and Kartik Talamadupula. Concise reasoning via reinforcement learning. arXiv preprint arXiv:2504.05185, 2025.
- [45] Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335, 2025.
- [46] Xuerui Su, Shufang Xie, Guoqing Liu, Yingce Xia, Renqian Luo, Peiran Jin, Zhiming Ma, Yue Wang, Zun Wang, and Yuting Liu. Trust region preference approximation: A simple and stable reinforcement learning algorithm for llm reasoning. arXiv preprint arXiv:2504.04524, 2025.
- [47] Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. arXiv preprint arXiv:2503.10460, 2025.
- [48] Wang Yang, Hongye Jin, Jingfeng Yang, Vipin Chaudhary, and Xiaotian Han. Thinking preference optimization. arXiv preprint arXiv:2502.13173, 2025.
- [49] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pages 1928–1937. PmLR, 2016.
- [50] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.

- [51] Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International conference on machine learning, pages 1587–1596. PMLR, 2018.
- [52] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. Pmlr, 2018.
- [53] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.
- [54] Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Yin Cui, Haotian Ye, Tsung-Yi Lin, Ming-Yu Liu, Jun Zhu, and Haoxiang Wang. Bridging supervised learning and reinforcement learning in math reasoning. arXiv preprint arXiv:2505.18116, 2025.
- [55] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.
- [56] Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444, 2025.
- [57] Tingchen Fu, Jiawei Gu, Yafu Li, Xiaoye Qu, and Yu Cheng. Scaling reasoning, losing control: Evaluating instruction following in large reasoning models. arXiv preprint arXiv:2505.14810, 2025.
- [58] Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.
- [59] Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020.
- [60] Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction. MIT Press, 2 edition, 2018.
- [61] Philipp Koehn. Statistical significance tests for machine translation evaluation. In Proceedings of the 2004 conference on empirical methods in natural language processing, pages 388–395, 2004.
- [62] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin, editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics.
- [63] Amrith Setlur, Nived Rajaraman, Sergey Levine, and Aviral Kumar. Scaling test-time compute without verification or RL is suboptimal. In ICLR 2025 Workshop: VerifAI: AI Verification in the Wild, 2025.
- [64] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training, 2025.
- [65] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. https://github.com/UCSC-VLAA/VLAA-Thinking, 2025.

# Appendix

- A Limitations 15
- B Theoretical Proof 15

- B.1 Convergence Rate of the Importance-Weighted Policy Gradient Estimator . . . . . 15
- B.2 Informal Analysis on Variance of Regularized Importance Sampling . . . . . . . . 17

- C Experimental Details 18

- C.1 Detailed Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.2 System Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.3 Significance Test . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- D Case Study 20
- E Additional Results 20

- E.1 Removing On-policy Clip . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- E.2 Extension to More Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- E.3 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- E.4 Hyperparameter Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- F Analysis 21

- F.1 LUFFY Learns Strategically from Off-Policy Traces, While SFT Imitates Rigidly . 21
- F.2 LUFFY Can Explore During Test-time While SFT Cannot. . . . . . . . . . . . . . 22

#### A Limitations

Firstly, we mainly focus on math reasoning RL training that has the golden answer and support verifiable rewards. Tasks lacking verifiable rewards are not addressed in this manuscript. Recent research [57] indicates that scaling reasoning may impair instruction-following capabilities, a challenge that can be alleviated by incorporating more comprehensive reward signals. Secondly, we focus on 7B and smaller foundation models, due to the limited computational resources. Scaling LUFFY to larger models could be an interesting topic, given scaling law [58, 59] is one of the most powerful principles in the area of large language models. Finally, as we are the first to incorporate off-policy guidance into the RLVR paradigm, we are limited to only including one off-policy trajectory per question, and find that one trajectory is already strong. However, extending off-policy guidance to multiple trajectories and multiple teachers could help the performance even further.

#### B Theoretical Proof

##### B.1 Convergence Rate of the Importance-Weighted Policy Gradient Estimator We study the nonconvex finite-sum problems of the form

1 n

max

J(θ) :=

θ∈Rd

n

Ji(θ), (9)

i=1

where both J and Ji (i ∈ [n]) may be nonconvex. We denote the class of such finite-sum Lipschitz smooth functions by J ∈ Jn. Here, we optimize functions in Jn of our importance-weighted policy gradient estimator.

The vanilla policy gradient algorithm maximizes the expected advantage function (equivalent to minimizing the negative expected advantage function) as

1 n

J(θ) = Eτ∼π

[A(τ)] ≈

max

θ

θ∈Rd

n

[A(τi)], (10)

i=1

According to the Policy Gradient Theorem [60], the vanilla policy gradient estimator has the following form:

n

1 n

[∇log πθ(τi) · A(τi)], (11)

∇J(θ) = Eτ∼π

[∇log πθ(τ) · A(τ)] ≈

θ

i=1

where we use ∇J(θ) to denote ∇θJ(θ) for simplicity. Our algorithm draws samples from another behavior policy πϕ, resulting in an importance-weighted policy gradient estimator as

∇J(θ) = Eτ∼π

ϕ

πθ(τi) πϕ(τi) · ∇log πθ(τ) · A(τ) ≈

1 n

n

[wi · ∇Ji(θ)], (12)

i=1

where wi = π

θ(τi)

πϕ(τi) is the importance weight assigned to sample i. Let αk denote the learning rate at iteration k, and wi

be the instance weight assigned to sample i by our algorithm. By stochastic gradient ascent, our algorithm has the following update rule:

k

θk+1 = θk + αkwi

(θk),i ∈ [n]. (13)

k∇Ji

k

- Definition 1. For J ∈ Jn, our algorithm takes an index i ∈ [n] and a point x ∈ Rd, and returns the pair (Ji(θ),∇Ji(θ)).
- Definition 2. We say J : Rd → R is Lipschitz smooth (L-smooth) if there is a constant L such that ||∇J(ϑ) − ∇J(θ)|| ≤ L||ϑ − θ||, ∀ϑ,θ ∈ Rd. (14)
- Definition 3. A point θ is called ϵ-accurate if ||∇J(θ)||2 ≤ ϵ. A stochastic iterative algorithm is said to achieve ϵ-accuracy in k iterations if E[||∇J(θk)||2 ≤ ϵ, where the expectation is over the stochasticity of the algorithm.
- Definition 4. We say J ∈ Jn has σ-bounded gradients if ||∇Ji(θ)|| ≤ σ for all i ∈ [n] and θ ∈ Rd.
- Definition 5. We say the positive instance weight w in our algorithm is bounded if there exist constants w and w such that w ≤ wi ≤ w for all i ∈ [n].

Theorem 1. Suppose the objective function of the policy gradient algorithm J ∈ Jn, where Jn is the class of finite-sum Lipschitz smooth functions, has σ-bounded gradients, and the importance weight

√

∗)−J(θ0)) Lσ2ww , and θ∗ is an optimal solution. Then, the iterates of our algorithm in Eq. (??) satisfy:

K where c = 2(J(θ

w = πθ/πϕ is clipped to be bounded by [w,w]. Let αk = α = c/

E[||∇J(θk)||2] ≤

min

0≤k≤K−1

2(J(θ∗) − J(θ0))Lw Kw

σ.

Proof. According to the Lipschitz continuity of ∇J, the iterates of our algorithm satisfy the following bound:

L 2 ||θk+1 − θk||2]. (15)

E[J(θk+1)] ≥ E[J(θk) + ⟨∇J(θk),θk+1 − θk⟩ −

After substituting (13) into (15), we have:

E[J(θk+1)] ≥ E[J(θk)] + αkwkE[||∇J(θk)||2] −

≥ E[J(θk)] + αkwkE[||∇J(θk)||2] −

Lαk2wk2 2

(θk)||2]

E[||∇Ji

k

Lαk2wk2 2

σ2. (16)

The first inequality follows from the unbiasedness of the stochastic gradient Ei

(θk)] = ∇J(θk). The second inequality uses the assumption on gradient boundedness in Definition 4. Re-arranging (16) we obtain

[∇Ji

t

k

1 αkwk

Lαkwk 2

σ2. (17) Summing (17) from k = 0 to K − 1 and using that αk is fixed α we obtain

E[||∇J(θk)||2] ≤

E[J(θk+1) − J(θk)] +

K−1

1 K

E[||∇J(θk)||2] ≤

E[||∇J(θk)||2]

min

t

k=0

K−1

K−1

1 K

1 αwk

1 K

Lαwk 2

E[J(θk+1) − f(θk)] +

σ2

≤

k=0

k=0

1 Kαw

Lαw 2

σ2

J(θK) − J(θ0) +

≤

1 Kαw

Lαw 2

J(θ∗) − J(θ0) +

σ2

≤

1 √

1 cw

Lcw 2

(J(θ∗) − J(θ0)) +

σ2 . (18)

≤

K

The first step holds because the minimum is less than the average. The second step is obtained from (17). The third step follows from the assumption on instance weight boundedness in Definition 5. The fourth step is obtained from the fact that J(θ∗) ≥ J(θK). The final inequality follows upon using α = c/

√

K. By setting

c =

2(J(θ0) − J(θ∗)) Lσ2ww

(19)

in the above inequality, we get the desired result. As seen in Theorem 1, our importance-weighted policy gradient estimator has a convergence rate of O(1/

| |
|---|

√

K). Equivalently, the time complexity of our algorithm to obtain an ϵ-accurate solution is O(1/ϵ2). Note that our choice of step size α requires knowing the total number of iterations K in advance. A more practical approach is to use a time-decayed step size of αk ∝ 1/

√

k or αk ∝ 1/k.

##### B.2 Informal Analysis on Variance of Regularized Importance Sampling

Importance sampling is a widely spread Monte-Carlo technique that adopts a reweighting strategy to estimate the so-called target distribution using samples from another distribution. A major drawback of vanilla importance sampling is the large variance of the weights, which is known to impact the accuracy of the estimates badly. In Sec. 3.2, we regularize the importance weights to enhance learning from low-probability tokens with the shaping function:

x x + γ

, γ ∈ [0,1], (20) where x = π

f(x) =

πϕ ∈ (0,+∞) is the original weight. We consider the first-order approximation of f(x) by Taylor expansion as

θ

∞

f(n)(u) n!

f(x) = f(u) + f′(u)(x − u) +

(x − u)n

(21)

n=2

≈ f(u) + f′(u)(x − u)

Suppose that πϕ dominates πθ, and we have E[x] = E[π

πϕ ] = 1. We consider the Taylor expansion at point u = 1 as

θ

1 1 + γ

f(x) ≈ f(1) + f′(1)(x − 1) =

γ (1 + γ)2

(x − 1) (22)

+

The variance of the first-order approximation of f(x) is

Var[f(x)] ≈ Var

γ (1 + γ)2

(x − 1) =

γ (1 + γ)2

2

Var[x] (23)

2

Since (1+ γγ)2

≪ 1, we have Var[f(x)] ≪ Var[x].

Further, we analyze the variance of the regularized importance weights f(x) = x+xγ,γ ∈ (0,1) given a special case that the original weight variable x follow a specific distribution as p(x) = e−x, x > 0.

This distribution makes sense for the importance weight x = π

θ(τ)

πϕ(τ) in our setting. First, with the distribution p(x) = e−x, the expectation of x is Ep(x)[x] = 0 ∞ e−xdx = 1. This matches the expectation of the importance weight as E[x] = E π

θ(τ)

πϕ(τ) = 1, given that πϕ(τ) dominates πθ(τ). Second, the probability density p(x) decreases as x gets larger, which matches the intuition that the importance weight x = π

θ(τ)

πϕ(τ) tends to have smaller values. πϕ(τ) is the probability of an expert trajectory under the assumed optimal policy, which is likely to produce large values as πϕ(τ) → 1. πθ(τ) is the probability of an expert trajectory under the current policy, which is usually small, especially at the early learning stage.

In this case, the variance of the original importance weight is

Var[x] = Ep(x)[x2] − Ep(x)[x]2

∞

(24)

x2e−xdx − 1

=

0

= 1.

The variance of the regularized importance weight is

Var[f(x)] = Ep(x)[f(x)2] − Ep(x)[f(x)]2

2

2

∞

∞

xe−x x + γ

x x + γ

e−x

dx −

=

dx

0

0

= 2 − 2γeγE1(γ) − 2γ2eγE3(γ) − (1 − γeγE1(γ))2

= 1 − 2γ2eγE3(γ) − γ2e2γE1(γ)2 < 1

(25)

where E1(γ)= γ ∞ e

u du > 0 and E3(γ)= γ ∞ e

−u

−u

u3 du > 0. Hence, we have Var[f(x)] < Var[x]. In summary, with the above informal analysis, we show that our regularized importance weights can achieve reduced variance, thus providing more stable training for leveraging off-policy guidance.

#### C Experimental Details

##### C.1 Detailed Setup

Easy and Hard Training Set These two datasets of different difficulties are generated from subsets of the OpenR1-MATH-220K dataset. We first filter the questions for which DeepSeek-R1 can generate a correct answer. Then, we split the data according to the length of DeepSeek-R1’s solution. We coin questions R1 can solve within 2k tokens as Easy set and those within 4k tokens as the Hard set, respectively. Intuitively, the more tokens needed for Deepseek-R1 to generate a correct answer, the more difficult the question is. Finally, the Easy dataset contains 7.3k prompts, and the Hard dataset contains 25.4k prompts.

Training. In addition to Qwen2.5-Math-7B, we extend LUFFY to Qwen2.5-Math-1.5B [30] and Qwen2.5-Instruct-7B [31], and LLaMA 3.1-8B [32]. To ensure fairness, we maintain 8 samples per prompt for all RL-trained models. The learning rate is constantly set as 1e-6. All training experiments are conducted using 8 A100 GPUs. We train 500 steps for all RL models and three epochs for SFT models. The only exception is LUFFY†, which is trained for 860 steps to match the GPU hour of SFT + RL.

Our implementation is based on verl5, which uses vLLM6 as the rollout generators. We are thankful for these open-source repositories.

Qwen2.5-Series Models. Since the context length of Qwen2.5-Math is 4096 and the generation length of off-policy samples could be lengthy, we change the rope theta from 10000 to 40000 and extend the window size to 16384. For Qwen2.5-Instruct, the context window is large enough. Hence, we do not change the model configurations. For all Qwen2.5-Series models, we use the same dataset as described in Sec. 4.

Llama-3.1-8B. For Llama3.1-8B, we follow Simple-RL-Zoo [20] and use a simplified prompt, and we do not ask the model to generate <think>/</think> tokens.

The dataset used for LLaMA3.1-8B is the subset of OpenR1-Math-220k we used with Qwen2.5Series models, selected by the length of DeepSeek-R1’s correct solution, i.e., 0-2k tokens (Easy training set described in Sec. 5.2). We find that on-policy RL fails on other subsets, such as the Hard training set (0-4k) or the same data used in Qwen2.5-Series.

SFT. For all SFT models, we train on the same DeepSeek-R1 generated traces and prompts as LUFFY. We follow the SFT setting from open-r1/OpenR1-Qwen-7B7, which reproduces the performance of Deepseek-R1’s distilled model. We train each model for 3 epochs. The train batch size is 64, and the learning rate is 5e-5. We use learning rate warmup ratio 0.1 and set the max length to 16k.

RL w/ SFT Loss. For multi-tasking RL and SFT objectives, we compute the on-policy loss on 7 on-policy samples and SFT loss on 1 off-policy sample per prompt. The other setup is the same as other RL experiments.

SFT + RL We use the same SFT model described earlier and further conduct RLVR training for 500 more steps. Following previous literature [24], we use the held-out dataset of OpenR1-Math-220k, resulting in around 49k prompts.

##### C.2 System Prompt

All our trained models, except LLaMA-3.1-8B, share the same system prompt for training and inference:

Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: “<think>\n thoughts </think>\n”. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After “</think>\n” in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \boxed{} for closed-form results like multiple choices or mathematical solutions.

User: This is the problem: {QUESTION} Assistant: <think>

For LLaMA-3.1-8B, we do not use the above system prompt as we find the model cannot follow such an instruction. Thus, we use a simplified version that only includes the CoT prompt and do not include <think> token.

User: {QUESTION} Answer: Let’s think step by step.

- 5https://github.com/volcengine/verl
- 6https://github.com/vllm-project/vllm
- 7https://huggingface.co/open-r1/OpenR1-Qwen-7B

##### C.3 Significance Test

We report the significance test results in our main results, i.e., Table 1. The significance test [61] is calculated by paired bootstrapping resampling, and the sample size is 1000 times. The null hypothesis asserts that any observed difference is merely due to random sampling variation rather than representing a genuine effect or difference between the two groups.

From our results, we can see that LUFFY and LUFFY† significantly outperform all baseline methods.

#### D Case Study

A demonstrative case study (Fig.7) comparing our proposed approach (LUFFY) against baseline methods (SFT and GPRO) in mathematical problem solving reveals distinct characteristics in reasoning patterns. SFT demonstrates redundant and circular reasoning with excessive repetition (over

- 8,129 tokens), while GPRO shows concise but unfounded deduction (1002 tokens), both leading to incorrect conclusions. In contrast, LUFFY presents a well-balanced approach (2623 tokens) that combines systematic decomposition with clear mathematical calculation. Through rigorous reasoning and proper verification steps, LUFFY successfully reaches the correct answer, demonstrating the effectiveness of our methodology in achieving both accuracy and efficiency.

#### E Additional Results

- E.1 Removing On-policy Clip

0 200 400

Steps

0.00

0.05

0.10

0.15

0.20

ClipRatio(%)

On-policy

Mixed-policy w/ Shaping

- Figure 8: Ratio of clipped signals.

0.05 0.1 0.2 0.3 0.5

γ in Policy Shaping Function

47.0

47.5

48.0

48.5

49.0

49.5

50.0

Accuracy

- Figure 9: Accuracy versus the choice of γ.

The clipping mechanism is introduced to constrain policy updates within a trust region [23], thereby ensuring stable training. However, when incorporating off-policy guidance, the target behavior may deviate significantly from the model’s current policy, especially early in training.

As shown in Figure 8, LUFFY experiences more frequent clipping compared to On-Policy RL, which can suppress learning from high-quality off-policy traces. To address this, we remove the on-policy clip to allow greater flexibility in updating toward unfamiliar yet effective actions, thereby unlocking the model’s capacity to better integrate off-policy reasoning behaviors.

- E.2 Extension to More Models

- Table 3 presents the detailed performance across six challenging competition-level benchmarks for Qwen2.5-Math1.5B, Qwen2.5-Instruct-7B, and LLaMA-3.1-8B. On all three models, LUFFY achieves consistent and substantial improvements, surpassing both SFT and On-Policy RL. On Qwen2.5-Math-1.5B, LUFFY attains an average score of 38.0, demonstrating notable gains of +6.1 and +8.0 points over SFT and On-Policy RL, respectively. Similar advantages are observed on the Qwen2.5-Instruct-7B and LLaMA-3.1-8B, where LUFFY consistently outperforms baselines across all benchmarks.

E.3 Ablation Study

In this section, we perform an ablation study to examine the contributions of LUFFY components, as summarized in

- Table 4. Shaping and NoClip both positively contribute to the final performance of Mixed-Policy training. However, applying these enhancements without off-policy guidance (On-Policy + No Clip/Shaping) does not yield improvement, underscoring the necessity of external signals to acquire nuanced and generalizable reasoning skills.

Table 3: Overall performance on six competition-level benchmark performance on Qwen2.5-Math1.5B, Qwen2.5-Instruct-7B and LLaMA-3.1-8B.

Model AIME 24 AIME 25 AMC MATH-500 Minerva Olympiad Avg. Qwen2.5-Math-1.5B

Qwen2.5-Math-1.5B-Base [30] 7.2 3.6 26.4 28.0 9.6 21.2 16.0 Qwen2.5-Math-1.5B-Instruct [30] 12.1 8.9 48.1 77.4 28.7 39.1 35.7

SFT 11.7 13.2 37.8 70.6 26.8 31.3 31.9 On-Policy RL 11.8 7.7 40.2 61.8 26.8 32.0 30.0

- LUFFY 16.0 13.1 47.1 80.2 30.5 41.0 38.0 Qwen2.5-Instruct-7B

Qwen2.5-7B-Instruct [31] 11.7 7.5 43.8 71.8 30.9 40.4 34.4 SFT 7.9 9.2 36.0 68.6 21.3 31.1 29.0 On-Policy RL 14.1 8.3 43.5 74.0 33.8 37.6 35.2

- LUFFY 17.7 14.8 50.9 82.0 31.3 47.4 40.7 LLaMA-3.1-8B

LLaMA-3.1-8B-Instruct [32] 5.1 0.4 18.6 44.6 19.5 14.1 17.1

SFT 0.5 0.1 5.4 20.2 4.0 5.3 5.9 On-Policy RL 0.3 0.5 9.4 23.4 17.6 6.1 9.6 LUFFY 1.9 0.1 13.5 39.0 15.1 9.6 13.2

Table 4: Ablation study on LUFFY components.

Model AIME 24 AIME 25 AMC MATH-500 Minerva Olympiad Avg. Mixed-Policy RL 19.4 17.7 58.9 84.6 35.7 49.9 44.4

+ Shaping 27.4 21.7 61.2 86.6 37.1 53.0 47.8 + Shaping + NoClip 29.4 23.1 65.6 87.6 37.5 57.2 50.1

On-Policy RL 25.1 15.3 62.0 84.4 39.3 46.8 45.5 + Shaping 21.3 13.6 58.0 80.6 36.8 41.8 42.0 + No Clip 21.5 17.4 61.1 83.4 36.8 49.0 44.9

##### E.4 Hyperparameter Study

In this section, we study the choice of γ in policy shaping function. The results are shown in Figure

- 9, trained from Qwen2.5-Math-7B. We choose γ value from [0.05, 0.1, 0.2, 0.3, 0.5]. When γ = 0.1, the model performs the best with 50.1 accuracy scores, and increasing or decreasing this value leads to a notable decline in model performance. Therefore, we consistently use γ = 0.1 throughout our experiments.

#### F Analysis

In this section, we analyze how LUFFY effectively leverages off-policy guidance, i.e., imitating to illuminate, to improve reasoning quality and generalization.

- F.1 LUFFY Learns Strategically from Off-Policy Traces, While SFT Imitates Rigidly

We compare the generation length distributions of LUFFY and SFT on the combined set of six mathematical reasoning benchmarks. As shown in Figure 10, LUFFY produces significantly shorter generations on average (2,832 tokens) compared to SFT (4,646 tokens), suggesting a more effective reasoning process that balances imitation and exploration. This observation helps explain the

###### Correct Solutions Length Distribution

80

LUFFY

Count

60

SFT

40

20

0

1000 2000 3000 4000 5000 6000 7000 8000

Length (# tokens)

Incorrect Solutions Length Distribution

300

| | | |
|---|---|---|
| | | |
| | | |

Count

200

100

0

1000 2000 3000 4000 5000 6000 7000 8000

Length (# tokens)

Figure 10: Generation length of correct and incorrect solutions.

excessive training costs of methods that naively combine SFT and RL, e.g., SFT+RL and RL w/ SFT Loss (Table 2), as these models spend substantially more compute on producing unnecessarily lengthy CoTs during the RL roll-out stage. In contrast, SFT often mimics the surface form of offpolicy demonstrations without genuinely engaging in problem-solving. This behavior is especially evident in incorrect outputs, where SFT frequently generates overly long and ultimately unproductive reasoning traces. These results indicate that while both methods are exposed to similar off-policy signals, LUFFY learns to selectively internalize useful reasoning patterns, whereas SFT tends to overfit to superficial features of the off-policy data.

We further analyze the generation lengths of RL w/ SFT Loss and LUFFY during training, as shown in Figure 11. RL w/ SFT Loss quickly imitates the off-policy traces, exhibiting a steep increase in generation length early in training. However, it soon becomes trapped in the superficial patterns of the demonstrations, leading to excessively long outputs that even surpass the length of the original off-policy traces. In contrast, LUFFY’s dynamic advantage balancing between on-policy and off-policy rollouts encourages more strategic learning. As a result, its generation length grows more gradually and steadily, reflecting a more selective and grounded adoption of reasoning behavior.

6000

5000

ResponseLength

4000

3000

2000

LUFFY

Oﬀ-policy Reasoning Traces

1000

RL w/ SFT Loss

Beyond generation length, imitation behavior can also be observed through the similarity between model outputs and off-policy traces. To quantify this, we compare generations from SFT, On-Policy RL, and LUFFY against those from DeepSeek-R1 on a held-out set of 1,000 samples, using BLEU [62] as the similarity metric. The resulting BLEU scores are 57.5 for SFT, 8.8 for On-Policy RL, and 44.8 for LUFFY, reflecting the strong imitation behavior of SFT and the more selective, yet substantial, imitation in LUFFY. We present a case study in Appendix D.

0 100 200 300 400 500

Steps

Figure 11: Generation length of RL w/ SFT Loss and LUFFY.

##### F.2 LUFFY Can Explore During Test-time While SFT Cannot.

We compute pass@8 accuracy on the combined AIME 2024 and AMC datasets, varying the generation temperature from 0.1 to 1.0. As shown in Figure 12, both RLbased methods (On-Policy RL and LUFFY) exhibit strong exploratory capabilities, with pass@8 improving as the temperature increases, showing potential in scaling testtime compute [63]. In contrast, although SFT performs comparably to LUFFY under near-deterministic decoding (temperature 0.1), its performance deteriorates at higher temperatures, failing to uncover additional correct reasoning paths. This highlights the fragility and limited adaptability of SFT, which aligns with prior findings [64, 65] that suggest SFT tends to memorize reasoning patterns rather than learning generalizable reasoning capability.

Performance w.r.t. Temperature

80.0

Accuracy(Pass@8)

77.5

75.0

72.5

70.0

67.5

On-policy RL

LUFFY

65.0

SFT

62.5

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Temperature

Figure 12: Pass@8 accuracy (on the merge sets of AIME 2024 and AMC) under different generation temperatures.

### Case Study

Prompt: <system prompt> This is the problem: A diagonal of a regular 2006-gon is called odd if its endpoints divide the boundary into two parts, each composed of an odd number of sides. Sides are also regarded as odd diagonals. Suppose the 2006-gon has been dissected into triangles by 2003 nonintersecting diagonals. Find the maximum possible number of isosceles triangles with two odd sides.

### SFT

### On-Policy RL

### LUFFY

##### Response:

##### Response:

##### Response: <think>

<think> [Formulaic Decomposition] To solve this problem, we need to understand ... [Pseudo-Systematic Structure] Let’s step-by-step. 1.Understanding the ... if we label the vertices of the polygon as (V_1,..., V_2006)... [Circular Property Description]

<think> [Problem Restatement Not Analysis] Okay, let’s try to tackle this problem. So, we have a regular 2006gon, and we need to find the maximum number of isosceles triangles... [Endless Definition Loop] Let me break this down step by step. First, let’s recall some basics about regular polygons and their diagonals... Hmm, maybe they mean that sides are considered as diagonals here?... Hmm, maybe they are using "diagonal" in a different sense here... Alternatively, maybe the problem is using diagonal¨ ¨to mean any chord... [Wait-Pattern Paralysis] Wait, but in the problem statement... Wait, but sides are chords... Wait, but in the problem’s definition, sides are odd diagonals... [Small-Case Trap]

[Clear Problem Definition]

To solve this problem, we need to understand ... and how it can be ... We are specifically interested in ...

[Systematic Decomposition and Analysis] First, let’s define some terms and properties...

- Step 1: Understanding Odd Diagonals...
- Step 2: Properties of Isosceles Triangles in a Regular Polygon...
- Step 3: Counting Odd Diagonals...
- Step 4: Dissecting the Polygon into Triangles... [Precise Mathematical Calculation] "For each vertex, there are 1002 vertices at an odd distance... [Logical Verification] Given that each isosceles triangle with two odd sides uses one odd diagonal and one side... [Consistency Check] "Let’s verify the logic:

Properties of Odd Diagonals... Properties of triangulation... Properties of the polygon...

[False Authority Appeal]

The number of isosceles triangles with two odd sides in a triangulation of a regular polygon is a well-known problem in combinatorial geometry...

[Unsubstantiated Conclusion]

For a 2006-gon, the maximum number of such triangles is 501.

</think> [Final Answer]

The maximum possible number of isosceles triangles with two odd sides in a triangulation of a regular 2006-gon is

- 1. Odd Diagonals ...
- 2. Dissection ..." </think>

Let’s take a regular polygon with a small even number of sides, say 4 (a square). A triangulation of a square has 2 triangles...

|501|
|---|

[Final Answer]

. Tokens Length: 1002 Correctness: False

Thus, the maximum possible number of isosceles triangles with two odd sides is

[Complete Breakdown]

|1003|
|---|

.

but diagon diagon polygon.< odd diagon in the odd of odd diagon = think... wait diagon diagon diagon diagon...

##### Tokens Length: 2623 Correctness: True

Tokens Length: > 8192 Correctness: False

Answer: "$1003$"

Figure 7: Comparison of three approaches (SFT, On-Policy RL, and LUFFY) for a geometric problem.

