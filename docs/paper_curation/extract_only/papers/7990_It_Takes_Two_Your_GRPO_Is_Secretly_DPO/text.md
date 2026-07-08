# arXiv:2510.00977v3[cs.LG]14May2026

## It Takes Two: Your GRPO Is Secretly DPO

Yihong Wu1†∗, Liheng Ma2,3∗, Lei Ding4, Muzhi Li5, Xinyu Wang2, Kejia Chen6, Zhan Su1

##### Chenyang Huang7,8, Zhanguang Zhang9, Derek Li9, Yingxue Zhang9, Jian-Yun Nie1, Mark Coates2,3

1UdeM 2McGill 3Mila 4UManitoba 5CUHK 6ZJU 7UAlberta 8Amii 9Huawei Noah’s Ark Lab

### Abstract

GRPO has emerged as a prominent reinforcement learning algorithm for posttraining LLMs. Unlike critic-based methods, GRPO computes advantages by estimating the value baselines from group-level statistics, eliminating the need for a critic network. Consequently, the prevailing view emphasizes the necessity of large group sizes, which are assumed to yield more accurate statistical estimates. In this paper, we propose a different view that the efficacy of GRPO stems from its implicit contrastive objective in the optimization, which helps reduce variance via the control variate method. This makes GRPO structurally related to preference learning methods such as DPO. This perspective motivates 2-GRPO, a minimal group-size variant that constructs contrastive signals with only two rollouts. We provide a rigorous theoretical analysis of 2-GRPO and empirically validate its effectiveness: 2-GRPO retains 97.6% of the performance of 16-GRPO, while requiring only 12.5% of the rollouts and 21% of the training time.

Gradient of GRPO

|o+j |

|o−k |

c(o−k,t|o−k,<t,q)∇θπθ(o−k,t|o−k,<t,q)

c(o+j,t|o+j,<t,q)∇θπθ(o+t |o+<t,q)

− E

−

q∼Q,o∼πθ

t

t

Positive

Negative

###### Gradient of DPO

|o−|

|o+|

c(o−t |o−<t,q)∇θπθ(o−t |o−<t,q)

c(o+t |o+<t,q)∇θπθ(o+t |o+<t,q)

###### − E

−

[(q,o+,o−)∼D]

t

t

Positive

Negative

KL Divergence

Reference Model

Policy Model

Reward Model

Advantage ( ?)

Contrastive Learning

Figure 1: (Top Left) Gradients of GRPO and DPO. The two share the same positive–negative structure and differ only in the token-level coefficient. (Top Right) Avg. Mean@32 of RL posttrained Qwen-1.5B. 2-GRPO matches the performance of 16-GRPO while substantially reducing training time. With resampling, 2-GRPO can even surpass 16-GRPO. (Bottom) A conceptual view of GRPO as contrastive (preference) learning: rollouts are split by their advantage relative to the group mean into positive (A+) and negative (A−) sets, which form implicit contrastive pairs.

∗Equal contribution. †Correspondence: yihong.wu@umontreal.edu, liheng.ma@mail.mcgill.ca.

Preprint.

### 1 Introduction

Reinforcement Learning (RL) has emerged as a central paradigm for the post-training of Large Language Models (LLMs). Two critical functions are aligning model outputs with human intent via RL with Human Feedback (RLHF) [27] and enhancing reasoning capabilities through RL with Verifiable Rewards (RLVR) [7]. Among recent advances, Group Relative Policy Optimization (GRPO) [34] is a prominent critic-free variant of Proximal Policy Optimization (PPO) [33], which effectively reduces the variance of gradient estimates by subtracting the estimated value baseline. Diverging from PPO, which relies on an auxiliary critic network for estimating the value baselines, GRPO estimates the advantage function by sampling a group of responses (rollouts) for a single prompt and normalizing their rewards based on the group statistics (mean/standard deviation). This design eliminates the memory and computational overhead of the value network while maintaining strong performance across various reasoning tasks.

Conventional intuition suggests that GRPO’s efficacy is strongly correlated with its group size, grounded in the premise that larger sample sizes yield more accurate advantage estimates and lead to stronger post-trained LLMs. However, this intuition overlooks the specific construction of the group-relative gradient estimator in GRPO. First, we demonstrate that GRPO intrinsically functions as contrastive learning [5] and the contrastive objective effectively reduces the variance of the gradient estimates as a control variate method [16]. The group of rollouts serves primarily to pair contrastive samples, rather than to estimate the value baselines. Specifically, the GRPO objective is de facto a Monte Carlo estimator to approximate the true contrastive gradients. Thus, the choices of group size primarily affects the variance of the Monte Carlo estimator, while the approximation itself remains unbiased. Therefore, in contrast to the prevailing value-baseline estimation viewpoint, GRPO with a small group size shall still work properly. Second, this perspective further reveals its close connection to the well-known Direct Preference Optimization (DPO) algorithm [29], which explicitly introduces the contrastive objective in the offline RLHF setting. The GRPO is de facto doing direct preference optimization on online RL settings with the necessary adaptations.

To evidence this hypothesis, we propose the minimal two-rollout setting (2-GRPO), a configuration previously regarded as inadequate for estimating group statistics [36], but well aligned with the contrastive learning interpretation and the DPO objectives. We provide a thorough theoretical analysis of the properties of 2-GRPO and empirically evaluate its effectiveness and efficiency across a diverse set of models and tasks. The theoretical analysis justifies the rationale behind the 2-GRPO designs. Empirically, 2-GRPO achieves performance comparable to 16-GRPO while substantially reducing training time. We further propose a resampling variant, 2-GRPO+RS, which reduces sample discard rate and achieves performance closer to 16-GRPO while being more efficient than 16-GRPO. These findings support our central hypothesis: GRPO derives its strength primarily from its contrastive formulation, rather than from accurate advantage estimation. The efficiency of 2-GRPO further highlights the promise of the contrastive policy optimization direction.

Overview

- • Theoretical Finding: GRPO reduces variance of policy gradient estimate via a Contrastive Gradient Optimization, rather than through reward shaping with value baselines as in PPO. GRPO is de facto the online RL version of DPO with corresponding adaptations.
- • Evidence: GRPO with a group size of 2 is invalid from the value estimate perspective but remains valid under the contrastiveness view. Empirically, 2-GRPO remains effective, indicating that contrastive learning provides a more principled explanation.
- • Implication: 2-GRPO matches the performance of 16-GRPO with substantially fewer rollouts, indicating that large group sizes are not essential and that advancing GRPO through contrastive learning techniques is a promising direction.

### 2 Preliminary

##### 2.1 Problem Setting and Notation

Our work focuses on RL-based post-training of LLMs for reasoning capabilities. Given an input prompt q ∈ Q, the model generates the i-th response oi = (oi,1,...,oi,T), where oi,t is the token

generated at step t ∈ [0,T] and oi,<t denotes the sequence of preceding tokens. A trajectory τ = (q,o) ∈ T is defined as a concatenation of a prompt and its corresponding generated response. In current RL post-training, the reward function r : T → R is typically defined at the trajectory level. The learning objective is to maximize the expected reward over the trajectory space:

θ(·|q)[r(τ)], (1)

J (θ) = Eq∼QEo∼π

where πθ denotes the policy model, a LLM with parameters θ; and Q is the set of prompts, each consisting of a question and necessary instructions. We mainly focus on the setting of verifiable rewards, where the responses can be verified as correct (r = 1) or incorrect (r = 0).

- 2.2 The Story of Variance Reduction: VPG, PPO, and GRPO

- As a foundational policy gradient method, Vanilla Policy Gradient (VPG) [38] optimizes the objective function using the following gradient estimator (where ri is the reward of (q,oi)):

∇θJ (θ) = E

oqi∼Q∼πθ

 ri

|oi|

t=0

∇θ log πθ(oi,t|oi,<t,q)

  . (2)

Although effective, VPG usually suffers from high variance of gradient estimates and training instability. Therefore, subsequent works [31, 33] utilize advantage estimates [3] to reduce the variance of the policy gradient estimator: Ai,t = ri − b(q), where Ai,t is token-level advantage and b(q) is the value baseline function (See Appx. B.1 for more details). An auxiliary LLM is employed as a critic to estimate this value baseline, such as in Proximal Policy Optimization (PPO) [33]:

J (θ) = E

oiq∼∼Qπθold

1 |oi|

|oi|

t=1

min{Ai,tρi,t,Ai,tCϵ(ρi,t)} , ρi,t =

πθ(oi,t | oi,<t,q) πθ

old

(oi,t | oi,<t,q)

, (3)

where πθ

old

is the policy used to generate trajectories, while πθ denotes the current policy being optimized. The ρi,t term is the introduced importance sampling technique for online (near-)on-policy RL while Cϵ(x) denotes the clipping function within the interval [1 − ϵ,1 + ϵ].

To eliminate the substantial computational overhead and memory demands of the critic network, several studies [20, 1, 34] propose estimating the baseline without a critic network. Specifically, GRPO estimates the advantage using the reward statistics from a group of generated responses:

Ai,t =

ri − mean(r) std(r) + ϵ

, (4)

where ri is the reward for response oi given query q, and r denotes the vector of rewards for G sampled responses associated with q. Therefore, it is generally believed that GRPO requires a sufficiently large group size to obtain accurate group-level statistics for advantage estimation. 3

3 A Tale of Two Algorithms: GRPO and DPO

- At first glance, the objectives of GRPO and DPO appear distinct on different RL settings. We show that they are the twin objects of the contrastive RL objective under the online/offline RL setting, which can be seen from the gradient forms of GRPO and DPO. This finding provides a new theoretical analysis (Sec. 4) and motivates a more efficient yet effective algorithm (Sec. 4.2).

- 3.1 Contrastive Objective for Sequences

Contrastive Learning [5] has been a powerful learning paradigm in (self-)supervised learning, ranging from 1-vs-1 (one positive and one negative) objectives [30] to 1-vs-N [26] and N-vs-M variants [9]. We first formalize the contrastive loss objective for sequences for further analysis.

3Due to limited space, the comprehensive related work is provided in Appx. A.

Definition 3.1 (Contrastive Loss for Sequences). Let πθ be a probabilistic model and D be a data distribution. Consider an anchor sequence x ∼ D, and let D+(· | x) and D−(· | x) denote the

conditional distributions for positive and negative samples, respectively. Let yt denote the t-th token of sequence y. A differentiable loss function L is contrastive if its gradient holds the following form:

|y−|

|y+|

c−t ∇θπθ yt−|y<t− ,x , (5)

c+t ∇θπθ(yt+|y<t+ ,x) − E

∇θL = − E

###### E

x∼D

y−∼D−

y+∼D+

t=1

t=1

where c+t and c−t are token-level coefficients depending on specific algorithm design.

We adopt token-level coefficients for generality, as sequence-level coefficients can be recovered as a special case. Furthermore, the number of positive (N) and negative (M) samples of each data point may vary depending on the specific designs, serving as a Monte Carlo estimator to approximate the true gradient in Eq. (5).

DPO is a 1-vs-1 contrastive learning Direct Preference Optimization (DPO) [29] is a dominant offline RLHF algorithms for LLMs:

πθ(o−|q) πref(o−|q)

πθ(o+|q) πref(o+|q) − β log

, (6)

###### LDPO = −E

log σ β log

(q,o+,o−)∼DDPO

where the preference pair (q,o+,o−) ∼ DDPO are from precollected human-annotated data. σ denotes the sigmoid function. It is easy to show that DPO is a 1-vs-1 contrastive learning. We provide Lemma B.1 and its proof in Appx. B.5 for reference.

##### 3.2 GRPO: N-vs-M Contrastive Learning

We demonstrate that GRPO effectively functions as a dynamic N-vs-M contrastive learning, where the group size G = N + M is fixed, but the specific values of N (positive samples) and M (negative samples) are dynamic based on the sampled responses. Let G+q and G−q denote the counts of correct and incorrect trajectories, respectively. The GRPO objective function can be formulated as:

JGRPO(θ,G) = E[q∼Q;{o+j ,o−k }Gj,k∼πθold(·|q)]

|o+j |

G−q

G+q

1 G+q

1 G−q

1 |o+j |

1 |o−k |

Cϵ+ (ρj,t)

−

VarG(q)

t=1

j=1

k=1

negative

positive

|o−k |

Cϵ− (ρk,t)

t=1

, (7)

where o+j and o−k denote rollouts with correct and incorrect outcomes, respectively. Denoting pˆθ

old,q is the empirical variance of the G sampled trajectories from the true Bernoulli(pold,q) under the RLVR setting.4 For simplicity, we denote the upper and lower clippings as Cϵ+(x) = min[x,1 + ϵ] and Cϵ−(x) = max[x,1 − ϵ], respectively.

old,q = G+q /G, the term VarG(q) = (1 − pˆθ

old,q)ˆpθ

The formulation in Eq. (7) provides the foundation for the following proposition, with a proof provided in Appx. B.3. Despite the sophisticated algorithm design of GRPO, this proposition unveils its contrastive nature.

- Proposition 3.2. The maximization of the GRPO objective is equivalent to the minimization of an N-vs-M contrastive loss estimator.

The derivation of Proposition 3.2 is based on the binary reward assumption to align with the RLVR setting. However, GRPO’s contrastive nature extends to continuous rewards inherently (as shown in Figure 1). Due to the space limit, we focus on the properties of GRPO on RLVR.

4In subsequent parts, we omit the subscript θold of p for brevity.

##### 3.3 Echoes of Contrastiveness: GRPO and DPO Summary

GRPO and DPO perform the same contrastive policy gradient – increasing preferred outputs relative to unpreferred ones – under different choices of Monte Carlo estimator. They are instantiated under different RL regimes, leading to different design choices in weighting, aggregation, and regularization.

Based on previous analysis, the differences between GRPO and DPO are merely in the coefficients of contrastive gradient:

ϵ i,t Var(q)

GRPO: c(oi,t | oi,<t,q) :=

, (8)

|oi|πθ

(oi,t | oi,<t,q)

old

β σ(ˆrθ(q,o−) − rˆθ(q,o+)) πθ(o)

πθ(y | x) πref(y | x)

DPO: c(ot | o<t,q) :=

. (9)

, rˆθ = β log

In the following, we show that the differences between GRPO and DPO are largely adaptations to their respective learning regimes: GRPO operates online with generated rollouts, whereas DPO operates offline with pre-collected preference data.

Group Size. DPO typically learns with fixed 1-vs-1 preference pairs which are collected offline in advance. By contrast, due to sampling responses online, GRPO needs to handle arbitrary N-vs-M positive–negative samples within each group. This changes only the Monte Carlo sample size used to estimate the same positive and negative contrastive gradients.

Token Aggregation. Within a sequence, GRPO averages token-level gradients, whereas DPO sums them. This is a design choice rather than a fundamental difference: e.g., SimPO [25] – a DPO variant – uses mean aggregation, while Dr. GRPO [22]– a GRPO variant – adopts sum aggregation.

Token-Level Weighting (Importance Sampling vs. Log-Likelihood). In GRPO, importancesampling coefficients correct the gradient for samples generated by the old policy, specifically for its near-on-policy online RL setting. It is typically used together with clipping for training stability. DPO, however, does not require such correction in the offline setting and therefore directly uses the log-likelihood form of πθ.

Group-Level Weighting. GRPO weights each group by Var(q), embodying its design philosophy of attending to more uncertain questions. DPO, in contrast, weights each pair by σ(ˆrθ(q,o−) − rˆθ(q,o+)), assigning higher scores to pairs where the negative sample outscores the positive one.

Reference Model. DPO is regularized toward the reference model πref through an implicit KL term. Optionally, GRPO can add a separate explicit KL penalty term w.r.t. the reference model.

In conclusion, the differences between GRPO and DPO mainly reflect adaptations to online vs. offline RL settings. The core mechanisms remain the same: both estimate a contrastive gradient that increases the likelihood of preferred outputs relative to unpreferred ones.

### 4 Why Viewing GRPO From Contrastive Learning?

Summary In this section, we address two key questions:

- • How contrastive learning reduces the variance of policy gradient estimates in GRPO?
- • Why contrastiveness is more principled than the value-baseline view?

##### 4.1 Variance Reduction via Contrastive Objective

We demonstrate that this contrastive gradient formulation functions as a control variate method, where the coefficients serve to control the variance of the estimator.

- Proposition 4.1. Let πθ denote the policy model. Let o+ ∼ πθ+(·|q) and o− ∼ πθ−(·|q) denote random variables representing a positive sample and a negative sample, respectively. Let g+ = ∇θ log πθ(o+|q), g− = ∇θ log πθ(o−|q) and ρ denote the correlation coefficient of g+ and g−. If

+,g−)

Cov(g+,g−) > 0 and 0 ≤ c ≤ 2Cov(g

Var(g−) , then Var(g+ − cg−) ≤ Var(g+). Specifically, if c = Cov(g

+,g−)

Var(g−) , then

Var(g+ − cg−) = 1 − ρ2 Var(g+), (10) where Var(·) and Cov(·,·) denotes the corresponding traces of var/cov matrices for gradient vectors. This proposition (proof in Appx. B.7) shows that, when the coefficient c lies within an appropriate range, the variance of the gradient estimator can be reduced. This result directly follows the control variate method, a variance reduction technique widely used in Monte Carlo estimation and stochastic gradient optimization [16]. The reduction of gradient variance stabilizes RL training [20].

A key implication of Proposition 4.1 is that the degree of variance reduction depends on the correlation between positive and negative samples. In LLM post-training, the positive sample o+ and the negative sample o− are generated by the same model conditioned on the same prompt q, which typically induces a nontrivial correlation between them. (See Appx. B.8 for more discussion.)

##### 4.2 GRPO with Small Group Size: It Should Fail, But Doesn’t

While both the value-baseline and contrastive perspectives account for GRPO’s variance reduction, they rest on fundamentally different assumptions. The prevailing value-baseline view holds that GRPO requires a sufficiently large group size to yield reliable group-level statistics; under this view, small-group GRPO should fail due to high-variance estimates [36]. The contrastive perspective, by contrast, treats the positive and negative samples within a group as Monte Carlo estimates of the true positive and negative gradients. Because Monte Carlo estimation is unbiased regardless of sample size, GRPO with small groups should remain effective under stochastic optimization.

To adjudicate between these two views, we introduce 2-GRPO, a variant that uses the minimal group size of 2. The value-baseline perspective predicts that this setting will fail (see Appx. B.2 for details). Empirically, however, 2-GRPO matches the performance of standard GRPO while achieving substantially higher efficiency, supporting the contrastive perspective as a more principled account of GRPO’s underlying mechanism. We describe 2-GRPO concretely in the following section.

##### 4.3 Introducing 2-GRPO

With a group size of two, the GRPO advantage reduces to a simple contrastive signal: A+ = +1 and A− = −1 when the two rollouts disagree on the reward, and both zeros otherwise. This yields an online RL counterpart of Direct Preference Optimization (DPO).

Analysis on 2-GRPO

- (Appx. C.1) Implicit Reweighting under Stochastic Optimization. At first glance, 2-GRPO uses fixed advantages regradless prompt success rates. However, under stochastic optimization, prompts are implicitly reweighted by their probability of forming a contrastive pair.
- (Appx. C.2) The Real Driver of Variance Reduction: Mini-Batch Size, Not Group Size. A common view attributes GRPO’s stability to its large group size. We argue that the key factor is instead the size of training mini-batch. Enlarging the group size is one way to increase rollouts, but enlarging the number of prompts per mini-batch achieves the same variance reduction.
- (Appx. C.3) Exploration on Hard Questions. A natural concern is that small-group GRPO may under-explore difficult prompts. We show that, for a fixed total rollout budget, the probability of 2-GRPO sampling at least one correct answer is no lower than that of 16-GRPO.

2-GRPO with Re-Sampling. Because of its binary contrastive nature, 2-GRPO discards any group whose two rollouts share the same reward. When the policy is highly accurate on the training set, this wastes a substantial fraction of generated samples and leads to sub-optimal performance. 5

5A discussion on discard rate is provided in Appx. D.3.

Table 1: 2-GRPO vs. 16-GRPO: post-trained on MATH/DAPO-Math-Sub and evaluated on five math reasoning benchmarks. G is the group size. (·) shows the gaps to 16-GRPO. All models are post-trained with 10 epochs.

Mean@32↑ G Time (h) ↓ MATH-500 AMC 2023 Minerva Math AIME 2025 Olympiad Bench Avg Post-training on MATH dataset Qwen-1.5B

w/o – 31.83 34.30 5.33 3.64 15.40 18.10 16 8.53 70.24 51.25 16.84 10.10 23.11 34.31

Ours 2 2.05(-75.96%) 69.28(-0.96) 49.53(-1.72) 16.25(-0.59) 9.48(-0.62) 22.31(-0.80) 33.37(-0.94) Ours 2+RS 3.71(-56.51%) 71.36(+1.12) 51.64(+0.39) 18.74(+1.90) 7.29(-2.81) 23.85(+0.74) 34.58(+0.27)

###### Qwen-7B

w/o – 47.16 38.36 5.99 5.00 9.83 21.27 16 9.30 75.90 61.79 22.81 13.23 25.99 39.94

Ours 2 2.43(-73.87%) 75.23(-0.67) 64.60(+2.81) 23.13(+0.32) 12.81(-0.42) 26.39(+0.40) 40.43(+0.49) Ours 2+RS 5.78(-37.85%) 76.89(+0.99) 61.64(-0.15) 24.90(+2.09) 11.67(-1.56) 25.39(-0.60) 40.10(+0.16)

Post-training on DAPO-Math-Sub dataset Qwen-1.5B

w/o – 31.83 34.30 5.33 3.64 15.40 18.10 16 13.30 70.66 56.56 18.00 9.58 24.56 35.87

Ours 2 2.12(-84.06%) 68.81(-1.85) 52.19(-4.37) 16.79(-1.21) 8.13(-1.45) 23.52(-1.04) 33.89(-1.98) Ours 2+RS 4.53(-65.93%) 71.64(+0.98) 58.59(+2.03) 20.11(+2.11) 9.79(+0.21) 24.67(+0.11) 36.96(+1.09)

###### Qwen-7B

w/o – 47.16 38.36 5.99 5.00 9.83 21.27 16 17.68 77.35 69.69 24.45 14.27 28.86 42.92

Ours 2 3.63(-79.47%) 77.43(+0.08) 64.84(-4.85) 21.95(-2.50) 14.58(+0.31) 29.86(+1.00) 41.73(-1.19) Ours 2+RS 7.55(-57.29%) 77.14(-0.21) 68.91(-0.78) 23.94(-0.51) 16.67(+2.40) 28.39(-0.47) 43.01(+0.09)

To address this, we introduce a resampling variant, denoted 2-GRPO+RS, which follows the strategy of DAPO [42]: whenever a group is discarded due to a zero advantage, it is replaced with a fresh group sampled from a new prompt. Resampling adds rollout-stage computation, but the overhead is modest and consistently lifts peak performance.

Efficiency Gains. The efficiency gains of 2-GRPO arise at two stages: rollout generation and policy optimization. For a fixed number of prompts, 2-GRPO generates only 12.5% of the rollouts required by 16-GRPO, and optimizes over the same 12.5% fraction during policy updates. 2-GRPO+RS may generate additional rollouts through resampling, but its optimization-stage cost matches standard 2-GRPO. In our experiments, we cap the rollout budget of 2-GRPO+RS at that of 16-GRPO; in practice, it typically uses fewer.

### 5 Experiments

Goal of Experiment. Building on the theoretical justification for 2-GRPO, we seek to empirically assess its validity in RLVR. We anticipate that 2-GRPO will achieve a comparable performance as the regular GRPO (16-GRPO), and exhibit better efficiency—with respect to computational resources and/or wall-clock time.

Datasets, Baselines and Hyper-parameters. We provide the details of datasets, baselines and hyper-parameter choices in Appx. D.1. For training, we adopt the verl framework [35] and utilize the built-in implementation of GRPO [34] as the baseline algorithm.

##### 5.1 Math Reasoning

Following prior studies [42], we consider mathematical tasks as representative instances of RLVR to verify our hypothesis. In the main experiment, the models are post-trained with RL techniques on MATH and DAPO-Math-Sub datasets under a fixed budget of 10 training epochs. The post-trained models are evaluated on five widely-used math reasoning benchmarks. This is an out-of-distribution evaluation setting, imposing requirements on the generalization ability of the post-trained models.

Figure 2: Pass@K (y) vs. K = 1,2,··· (x) over five math benchmarks. First/second row for models post-trained on MATH/DAPO-Math-Sub dataset.

Table 1 showcases the Mean@32 as well as the training time. The empirical results show that 2-GRPO achieves 97.6% of 16-GRPO’s average performance while using only 12.5% of its total rollouts and 21.0% of its training time. 6

These results provide strong corroboration of our theoretical finding that reducing group size preserves performance while substantially improving efficiency.

The resampling variant, 2-GRPO+RS, further improves peak performance, outperforming 16-GRPO on average while using roughly half of its training time. Although it is slower than 2-GRPO, it remains substantially more efficient than 16-GRPO, making it a practical alternative that preserves small-group efficiency while recovering the performance benefits of broader exploration.

Figure 2 shows the Pass@K over various K choices. Overall, 2-GRPO achieves Pass@K performance comparable to 16-GRPO across different choices of K. In particular, 2-GRPO even outperforms 16-GRPO on the AMC 2023 and Olympiad Bench. On AIME 2025, 2-GRPO performs better when post-trained on DAPO-Math-Sub, but worse when post-trained on MATH, likely due to the larger distribution shift between training dataset and the evaluation one.

We extend the evaluation of 2-GRPO to additional RLVR tasks beyond mathematical reasoning, including Vision Reasoning (Geometry3K) and Code Generation (Code-R1), with results reported in Figure 3. The results demonstrate that 2-GRPO remains both effective and efficient across these diverse tasks, highlighting its broader applicability beyond math reasoning. As shown in the figure, 2-GRPO converges substantially faster than 16-GRPO, owing to the reduced number of samples generated and updated per step. This phenomenon is consistent with our theoretical findings, which identify the role of the group as providing contrastive sample pairs. Reducing the group size not only preserves performance but also accelerates the learning process.

##### 5.2 Ablation Study: The Effect of Group Size

Our proposed 2-GRPO changes the group size while also adjusting the training batch size and the learning rate to account for the reduced number of rollouts per prompt (discussed in Appx. B.1). To isolate the effect of group size, we conduct an ablation study over different group sizes using the exact same configuration: 10 training epochs, a generation batch size of 512 prompts, a training batch size of 32 prompts, and a learning rate of 10−6.

It is worth noting that this setting is slightly unfavorable to smaller group sizes – the actual training mini-batch size by # rollouts is # prompts per batch multiplied by the group size. Therefore, GRPO with smaller group sizes in the ablation study suffered from higher variance of gradient estimates (see C.2 for details). Nonetheless, Figure 4 shows that the Mean@32 differences among

6Appx. D.2 discusses the relationship between the total number of rollouts and computational cost.

(a) Train Reward (y) vs. Time (x) [Geo3K] (b) Test Acc. (y) vs. Time (x) [Geo3K]

(c) Train Reward (y) vs. Time (x) [CodeR1] (d) Test Acc. (y) vs. Time (x) [CodeR1]

- Figure 3: The performance of 2-GRPO over training time (mins) on Geometry3K and Code-R1.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 4: Average performance evaluated over 5 Math benchmarks with group size G = 2,4,8,16. The results are reported in Mean@32 (left) and Pass@32 (right).

G = 2,4,8,16 are consistently small across all settings. Moreover, increasing the group size does not reliably improve Pass@32: larger groups do not consistently outperform smaller ones, and in some cases smaller groups achieve better Pass@32. The full results of the ablation study are provided in Table 4 in Appx. D.4.

### 6 Conclusion

In this work, we demonstrate that GRPO de facto functions as contrastive learning. We argue that the primary role of the group mechanism is not for accurate value-baseline estimation, as commonly assumed, but for the efficient construction of contrastive signals. Based on this insight, we reveal the fundamental connection between GRPO and DPO—they are two echoes of the same contrastive gradient optimization principle, reflected through the online and offline RL settings, respectively. To further validate this insight, we introduce 2-GRPO, a minimal variant with only two rollouts per prompt. Although this setting is degenerate from the standpoint of traditional advantage estimation, it remains theoretically well motivated under our contrastive framework. Empirically, 2-GRPO achieves performance comparable to 16-GRPO while substantially reducing the computational overhead of rollout generation and policy optimization. These results support our hypothesis and suggest a more efficient design principle for RL algorithms for LLMs. More broadly, while our analysis focuses on GRPO, the insights developed here may extend to a wider class of group-based RL algorithms.

### References

- [1] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. In Proc. Annu. Meet. Assoc. Comput. Linguist.,

- 2024.

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923,

- 2025.

- [3] III Baird, Leemon C. Advantage updating. Technical report, Wright Laboratory, 1993.
- [4] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In Proc. Int. Conf. Mach. Learn., 2020.
- [5] Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. In Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit., 2005.
- [6] Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025.
- [7] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [8] Yannis Flet-Berliac, Nathan Grinsztajn, Florian Strub, Eugene Choi, Bill Wu, Chris Cremer, Arash Ahmadian, Yash Chandak, Mohammad Gheshlaghi Azar, Olivier Pietquin, et al. Contrastive policy gradient: Aligning llms on sequence-level scores in a supervised-friendly fashion. In Proc. Conf. Empir. Methods Nat. Lang. Process., 2024.
- [9] Nicholas Frosst, Nicolas Papernot, and Geoffrey Hinton. Analyzing and improving representations with the soft nearest neighbor loss. In Proc. Int. Conf. Mach. Learn., 2019.
- [10] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.
- [11] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proc. Annu. Meet. Assoc. Comput. Linguist., 2024.
- [12] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020.
- [13] Joey Hejna, Rafael Rafailov, Harshit Sikchi, Chelsea Finn, Scott Niekum, W Bradley Knox, and Dorsa Sadigh. Contrastive preference learning: learning from human feedback without rl. arXiv preprint arXiv:2310.13639, 2023.
- [14] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Adv. Neural Inf. Process. Syst. (Track Datasets Benchmarks), 2021.
- [15] Minda Hu, Muzhi Li, Yasheng Wang, and Irwin King. Momentum contrastive pre-training for question answering. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 4324– 4330, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics.
- [16] Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In Adv. Neural Inf. Process. Syst., 2013.

- [17] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [18] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. In Adv. Neural Inf. Process. Syst., 2022.
- [19] Gang Li, Ming Lin, Tomer Galanti, Zhengzhong Tu, and Tianbao Yang. Disco: Reinforcing large reasoning models with discriminative constrained optimization. arXiv preprint arXiv:2505.12366, 2025.
- [20] Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: a simple, effective, and efficient reinforcement learning method for aligning large language models. In Proc. Int. Conf. Mach. Learn., 2024.
- [21] Jiawei Liu and Lingming Zhang. Code-r1: Reproducing r1 for code with reliable rewards. https://github.com/ganler/code-r1, 2025.
- [22] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [23] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6774–6786, 2021.
- [24] Xufei Lv, Kehai Chen, Haoyuan Sun, Xuefeng Bai, Min Zhang, and Houde Liu. The hidden link between rlhf and contrastive learning. arXiv preprint arXiv:2506.22578, 2025.
- [25] Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. In Adv. Neural Inf. Process. Syst., 2024.
- [26] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.
- [27] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Adv. Neural Inf. Process. Syst., 2022.
- [28] Lei Pang and Ruinan Jin. On the theory and practice of grpo: A trajectory-corrected approach with fast convergence. arXiv preprint arXiv:2508.02833, 2025.
- [29] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Adv. Neural Inf. Process. Syst., 2023.
- [30] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. Bpr: Bayesian personalized ranking from implicit feedback. In Proc. Conf. Uncertain. Artif. Intell., 2009.
- [31] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In Proc. Int. Conf. Mach. Learn., 2015.
- [32] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. In Proc. Int. Conf. Learn. Represent., 2016.
- [33] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [35] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proc. Eur. Conf. Comput. Syst., 2025.
- [36] Student. The probable error of a mean. Biometrika, pages 1–25, 1908.
- [37] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In Proc. Int. Conf. Mach. Learn., 2020.
- [38] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.
- [39] Yihong Wu, Liheng Ma, Muzhi Li, Jiaming Zhou, Lei Ding, Jianye Hao, Ho-fung Leung, Irwin King, Yingxue Zhang, and Jian-Yun Nie. Advancing multi-agent rag system with minimalist reinforcement learning. In Proc. Int. Conf. Auton. Agents Multi-Agent Syst., 2026.
- [40] Yihong Wu, Le Zhang, Fengran Mo, Tianyu Zhu, Weizhi Ma, and Jian-Yun Nie. Unifying graph convolution and contrastive learning in collaborative filtering. In Proc. ACM SIGKDD Conf. Knowl. Discov. Data Min., 2024.
- [41] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 Technical Report, January 2025.
- [42] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. In Adv. Neural Inf. Process. Syst., 2025.
- [43] Le Zhang, Bo Wang, Xipeng Qiu, Siva Reddy, and Aishwarya Agrawal. REARANK: Reasoning re-ranking agent via reinforcement learning. In Proc. 2025 Conf. Empir. Methods Nat. Lang. Process., 2025.
- [44] Ruiqi Zhang, Daman Arora, Song Mei, and Andrea Zanette. Speed-rl: Faster training of reasoning models via online curriculum learning. arXiv preprint arXiv:2506.09016, 2025.
- [45] Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, et al. Geometric-mean policy optimization. arXiv preprint arXiv:2507.20673, 2025.
- [46] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
- [47] Haizhong Zheng, Yang Zhou, Brian R Bartoldson, Bhavya Kailkhura, Fan Lai, Jiawei Zhao, and Beidi Chen. Act only when it pays: Efficient reinforcement learning for llm reasoning via selective rollouts. arXiv preprint arXiv:2506.02177, 2025.
- [48] Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework (github), 2025.
- [49] Linghao Zhu, Yiran Guan, Dingkang Liang, Jianzhong Ju, Zhenbo Luo, Bin Qin, Jian Luan, Yuliang Liu, and Xiang Bai. Shuffle-r1: Efficient rl framework for multimodal large language models via data-centric dynamic shuffle. arXiv preprint arXiv:2508.05612, 2025.

Appendix

- A Related Work

- A.1 Contrastive Learning and LLM Alignment

Contrastive learning is the cornerstone of self-supervised representation learning [37, 12, 4, 15, 40]. The fundamental objective is to minimize the distance between anchor and positive samples in the representation space while maximizing the distance between the anchor and negative samples. Given this contrastive nature, the framework shares structural similarity with DPO, which conducts preference learning by increasing the likelihood of preferred completions relative to dispreferred ones. While recent literature explores the theoretical connections between RLHF and contrastive learning [13, 8, 24], our work establishes a formal link between GRPO and DPO through a contrastive lens. This provides a unified analytical framework for understanding alignment. Specifically, we attribute the efficacy of GRPO to the construction of contrastive pairs, which serves as a control variate to reduce the variance of the gradient estimator. This analysis offers generalizable insights to broader alignment algorithms.

- A.2 Adaptive Rollouts in RLVR

RL post-training has demonstrated significant success in enhancing LLM performance across diverse domains [39, 43]. Unlike SFT, RL requires the model to generate online samples during training. Although modern frameworks integrate high-throughput inference engines such as vLLM and SGLang, the autoregressive nature of LLMs ensures that the generation phase remains a primary computational bottleneck. This challenge is exacerbated by the common intuition that LLM-based RL often necessitates large group sizes to achieve good performance. To mitigate this overhead, recent studies have proposed selective or adaptive sampling techniques to reduce the number of rollouts without compromising performance [47, 44, 49]. Within this context, 2-GRPO serves as a robust baseline. Furthermore, our contrastive analysis of GRPO opens a new design space for developing efficient sampling algorithms in RLVR.

- B Theorems

- B.1 Variance Reduction of Policy Gradient Estimate

Given a prompt, consider the random variable (r.v.) of the reward r (which can be replaced by the advantage a) and the r.v. of the policy gradient g (corresponding to |o1

i|

|oi| t=0 ∇θ log πθ(oi,t|oi,<t,q)

in VPG or |o1

i|

|oi| t ∇θρi,t in PPO/GRPO). Since E[g] = 0 over all potential actions, the variance

of the product of these r.v.’s can be written as: Var(r · g) = Var(g) Var(r) + (E[r])2 + Cov(r2,g2) − (Cov(r,g))2

Interaction term

.

(11)

The interaction term can be ignored when importance sampling and clipping are applied, as the gradient is bounded in a small region. Previous work [3, 31, 32, 33] shows that replacing raw rewards with advantage functions (E[a] = 0) effectively reduces variance, leading to more stable and improved RL optimization.

- B.2 Mean Estimation with Samples n = 2

The instability of normalization with extremely small samples is a well-documented phenomenon in classical statistics, dating back to the seminal work of William Sealy Gosset (published under the pen name Student) [36]. For a sample size of n = 2, the degrees of freedom df = 1 result in a normalization factor that follows a Cauchy distribution. Such small-sample estimates of variance are highly skewed, leading to normalized outputs with infinite variance and no defined mean, undermining the goal of statistical stability.

##### B.3 Reveal GRPO as Contrastive Learning

Proof of Proposition 3.2. In the RLVR setting, rewards are binary, which leads to binary advantages given a prompt. Let A+q ,A−q denote the positive and negative advantage, respectively. From Eq. (4), we can have

1 − pˆq pˆq(1 − pˆq)

1 − pˆq pˆq

A+q =

=

,

0 − pˆq pˆq(1 − pˆq)

pˆq 1 − pˆq

A−q =

= −

.

(12)

The clipping function is

 

x, |x − 1| ≤ ϵ

clip(x,1 − ϵ,1 + ϵ) =

, (13)

1 − ϵ, x < 1 − ϵ 1 + ϵ, x > 1 + ϵ



which means that x will be assigned to 1 − ϵ (1 + ϵ) if x is less (greater) than 1 − ϵ (1 + ϵ). For simplifying notation, let Cϵ+(x) = min[x,1 + ϵ] and Cϵ− = max[x,1 − ϵ]. The key derivation of rewriting GRPO objective is as follows:

JGRPO(θ)

|oi|

G

πθ(oi,t|oi,<t,q) πθ

1 G

1 |oi|

= E q∼Q

Cϵ

Ai,t ,

(oi,t|oi,<t,q)

{oi}Gi=1∼πθold(·|q)

old

t=1

i=1

= E q∼Q

- {oj}Gj=1+ ∼πθ+

old

(·|q)

- {ok}Gk=1− ∼πθ−

(·|q)

old

 

  ,

G−

G+

|oj|

|ok|

πθ(oj,t|oj,<t,q) πθ

πθ(ok,t|ok,<t,q) πθ

- 1 G

1 |oj|

1 |ok|

A−k Cϵ−

A+j Cϵ+

+

(oj,t|oj,<t,q)

(ok,t|ok,<t,q)

old

old

t=1

t=1

j=1

k=1

= E q∼Q

- {oj}Gj=1+ ∼πθ+

old

(·|q)

- {ok}Gk=1− ∼πθ−

(·|q)

old

G−

G+

|oj|

|ok|

G− G

G+ G

πθ(oj,t|oj,<t,q) πθ

πθ(ok,t|ok,<t,q) πθ

1 G+

1 G−

1 |oj|

1 |ok|

+ A−q

Cϵ−

Cϵ+

A+q

,

(oj,t|oj,<t,q)

(ok,t|ok,<t,q)

old

old

t=1

t=1

j=1

k=1

= E q∼Q

- {oj}Gj=1+ ∼πθ+

old

(·|q)

- {ok}Gk=1− ∼πθ−

(·|q)

old

  .

  1

G−

G+

|oj|

|ok|

πθ(oj,t|oj,<t,q) πθ

πθ(ok,t|ok,<t,q) πθ

1 G−

1 |oj|

1 |ok|

Cϵ−

Cϵ+

(oj,t|oj,<t,q) −

VarG(q)

G+

(ok,t|ok,<t,q)

old

old

t=1

t=1

j=1

k=1

(14) The second equation is obtained by dividing the trajectories into two groups: positive and negative. The third equation is obtained by the fact that all positive advantages are the same and that all negative

−

+

G = 1−pˆpˆpˆ = (1 − pˆ)ˆp and A− G

advantages are the same. Since A+G

G = − (1 − pˆ)ˆp, we

obtain Eq. (7). When G → ∞, we have the following facts:

G+ = ∞ , lim G→∞

lim

G→∞

G− = ∞ lim G→∞

(1 − pˆ)ˆp = (1 − p)p ,

G+

1 G+

f(oj) = Eo

j∼Oθ+f(oj) ,

lim

G+→∞

j=1

G−

1 G−

f(ok) = Eo

lim

k∼Oθ−f(ok) .

G−→∞

k=1

(15)

Then the GRPO objective has the following gradient w.r.t. parameter θ: ∇θJGRPO =

|o+j |

G−q

G+q

o−k

ϵ k,t∇θπθ(o−k,t|o−k,<t,q)

ϵ j,t∇θπθ(o+j,t|o+j,<t,q)

1 G+q

1 G−q

###### E

(o+j,t|o+j,<t,q) −

VarG(q)

|o−k |πθ

(o−k,t|o−k,<t,q)

|o+j |πθ

q∼Q

old

old

t

t

j=1

k=1

(16)

|o+j |

G−q

G+q

|o−k |

1 G−q

1 G+q

c(o−k,t|o−k,<t,q)∇θπθ(o−k,t|o−k,<t,q)

c(o+j,t|o+j,<t,q)∇θπθ(o+j,t|o+j,<t,q)

###### = E

−

q∼Q

t

t

j=1

k=1

Negative

Positive

(17) where ϵj,t is an indicator function if the token oj,t is clipped and c(oi,t|oi,<t,q) :=

√

Var(q) ϵi,t

|oi|πθold(oi,t|oi,<t,q). Compare Eq. (17) with Def. 3.1, the derivative of GRPO is a Monte Carlo estimator of contrastive derivative.

| |
|---|

##### B.4 Further Discussion on Importance Sampling and the Log-likelihood Term

Most autoregressive LLMs adopt causal probability modelling as log πθ(o|q) = log πθ(ot|o<t,q). This decomposition leads to the following trajectory-level form to describe the gradient of token probabilities:

1

πθ(ot|o<t,q)∇θπθ(ot|o<t,q). (18) DPO follows a similar structural derivation.

∇θ log πθ(o|q) =

t

It is worth mentioning that the importance sampling in PPO can be viewed as a natural extension of such gradient form for online on/off-policy RL [33]. However, the token-level importance sampling in PPO and vanilla GRPO often obscures this direct connection at the trajectory level.

Recent subsequent variants of GRPO [46, 45, 28], e.g., GSPO and TIC-GRPO, utilize sequence-level importance sampling. This formulation allows us to draw a direct connection between importance sampling and the log-likelihood terms:

πθ(o | q) πθ

πθ(o | q) πθ

1 πθ(ot | o<t,q)∇θπθ(ot | o<t,q). (19)

∇θ

=

(o | q)

(o | q) t

old

old

It is straightforward to see from the gradient form that the importance sampling term adjusts the Log-likelihood term by a coefficient π

θ(o|q)

πθold(o|q). The token-level importance sampling in PPO and GRPO behaves similarly by applying token-level correction. The clipping applied on top of importance sampling is a minor additional modification, which we do not elaborate on here.

##### B.5 Proof of Lemma B.1: DPO is 1-vs-1 contrastive learning

- Lemma B.1. The DPO loss is a 1-vs-1 contrastive loss estimator.

- Proof of Lemma B.1. The DPO loss (Eq. (6)) has the following derivatives:

σ(ˆrθ(q,o−) − rˆθ(q,o+)) ∇θ log πθ(o+|q) − ∇θ log πθ(o−|q)

∇θLDPO = −β E

[(q,o+,o−)∼DDPO]

|o−|

|o+|

c(o−t |o−<t,q)∇θπθ(o−t |o−<t,q)

c(o+t |o+<t,q)∇θπθ(o+t |o+<t,q)

= − E

−

[(q,o+,o−)∼DDPO]

t

t

Positive

Negative

(20) where rˆθ = β(x,y)log π

θ(y|x)

πref(y|x); σ denotes the sigmoid function; and c(ot|o<t,q) := βσ(ˆrθ(q,o−)−rˆθ(q,o+))

πθ(o|q) aligning with Def. 3.1.

| |
|---|

###### B.6 GRPO v.s. DPO from Contrastive Learning As shown in Appx. B.3 and B.5, GRPO and DPO admit the following gradient formulations:

|o−k |

|o+|

c(o−k,t|o−k,<t,q)∇θπθ(o−k,t|o−k,<t,q)

c(o+t |o+<t,q)∇θπθ(o+t |o+<t,q)

∇θJGRPO = E

−

q∼Q

t

t

Positive

Negative

(21)

|o−|

|o+|

c(o−t |o−<t,q)∇θπθ(o−t |o−<t,q)

c(o+t |o+<t,q)∇θπθ(o+t |o+<t,q)

∇θLDPO = − E

−

[(q,o+,o−)∼DDPO]

t

t

Positive

Negative

(22)

These expressions reveal that both maximizing the GRPO objective and minimizing the DPO loss correspond to the same underlying contrastive learning mechanism, differing only in the specific design of the coefficient c(o+t |o+<t,q). The key distinction lies in how the coefficient c(·) is instantiated under different RL settings (online vs. offline):

- • Importance sampling term (online GRPO) vs. log-likelihood term (offline DPO) (see Appx. B.4 for detailed discussion).
- • Reference-model regularization: an explicit KL term (GRPO) vs. implicit incorporation into the “advantage” (DPO).

Importantly, these differences do not alter the fundamental optimization structure, but rather reflect distinct design choices tailored to their respective RL regimes.

##### B.7 Proof of Proposition 4.1

Proof.

Var(g+ − cg−) = Var(g+) + c2Var(g−) − 2cCov(g+,g−),

Cov2(g+,g−) Var(g−)

(23)

= Var(g+) −

,

= (1 − ρ2)Var(g+).

The first equation is obtained by the definition of variance. The second equation is obtained by substituting c = Cov(g

+,g−)

√ +,g−)

Var(g−) . The third equation is hold because ρ = Cov(g

. On the other hand, consider f(c) = c2Var(g−) − 2cCov(g+,g−). If 0 ≤ c ≤ 2Cov

Var(g+)Var(g−)

2(g+,g−)

Var(g−) , then f(c) ≤ 0.

| |
|---|

##### B.8 The Correlation between The Positive and The Negative

We do not have access to the joint distribution of positive and negative gradients, so direct empirical estimation of their covariance is infeasible. Instead, we use the law of total covariance:

Cov(g+,g−) = Ep[Cov(g+,g− | p)] + Covp(E[g+|p],E[g−|p]). (24)

Under conditionally independent sampling, the first term is zero, so we estimate the second term across prompts. Statistics are computed over 100 prompts with 10 responses each on MATH with Qwen-1.5B. As high-dimensional vectors usually have small dot-products, we report a baseline as reference where pos/neg pairs are randomly permuted.

Table 2: Covariance metrics between positive and negative gradients. Metric Cov(g+,g−) Covperm Value 0.08705 0.00128

- As shown in the table, the covariance between positive and negative gradients from the same prompt is significantly larger than that between randomly paired positive and negative gradients, which confirms our assumption.

##### B.9 Proof of Proposition C.1

Proof. Case 1. Notice that σˆ = 21N 2kN=1(Xk − µˆ)2 = µˆ(1 − µˆ) and µˆ = 21N 2kN=1 Xk. Fix an index i and condition on the event {Xi = x} with x ∈ {0,1}. In this case, by the strong law of large numbers and the continuous mapping theorem, we have µˆ a.s.→ p and σˆ a.s.→ p(1 − p). Thus, it follows that

x − p p(1 − p)

E[Yi | Xi = x] =

lim

lim

.

ϵ→0

N→∞

Case 2. When Xi,1 = Xi,2, we have Xi,j = µˆi and Yi,j = 0 for any j ∈ {1,2}. When Xi,1 ̸= Xi,2, we have µˆi = 0.5, σˆi = 0.5, and Yi,j = 2X1+2i,j−ϵ 1. By the law of total expectation, it follows that

Thus, we have

, E[Yi,j | Xi,j = 0] = −p 1 + 2ϵ

1 − p 1 + 2ϵ

E[Yi,j | Xi,j = 1] =

.

E[Yi,j | Xi,j = x] = x − p.

lim

ϵ→0

| |
|---|

##### B.10 Proof of Lemma C.3

- Proof of Lemma C.3.

Var(gˆB) = Var{x

i}Bi=1

B

1 B2

=

i=1

B

1 B

##### g(xi)

i=1

Varx(g(x)) B

.

Varx

(g(xi)) =

i

(25)

where the second and third equalities are obtained by the properties of independence and identity in i.i.d. data, respectively. By the above equation, increasing B decreases Var.

| |
|---|

### C Theoretical Analysis of 2-GRPO

##### C.1 Implicit Weighting in Stochastic Optimization

At first glance, 2-GRPO appears to use only fixed advantages, A+ = 1 and A− = −1, ignoring prompt-level success rates. However, under mini-batch stochastic optimization, 2-GRPO implicitly reweights prompts through their likelihood of forming contrastive pairs.

Standard GRPO relies on the empirical success rate pˆq to estimate the true correctness probability pq for advantage assignment, relying on larger group sizes for accuracy. While this mechanism appears degenerate in 2-GRPO, we show that, through the lens of stochastic optimization, 2-GRPO implicitly estimates the advantage.

Moreover, 2-GRPO does not simply estimate the large-group GRPO gradient with fewer samples; it induces a different prompt-level weighting that prioritizes prompts likely to yield contrastive pairs.

Proposition C.1. Given a constant p ∈ (0,1) and a small positive constant ϵ, we consider two scenarios below:

- • Case 1: Consider X1,··· ,X2N i.i.d.∼ Bernoulli(p). Let Yi = X

i−µˆ

σˆ+ϵ , where µˆ = 21N 2i=1N Xi and σˆ = 21N 2i=1N (Xi − µˆ)2. Then, it follows that

lim

ϵ→0

lim

N→∞

E[Yi|Xi = x] =

x − p p(1 − p)

. (26)

- • Case 2: Consider N pairs of (Xi,1,Xi,2) with each Xi,j i.i.d.∼ Bernoulli(p). Let Yi,j = Xσi,jˆ −µˆi

i+ϵ , where µˆi = 21(Xi,1 + Xi,2) and σˆi = 12 2j=1(Xi,j − µˆi)2. Then, it follows that

E[Yi,j|Xi,j = x] = x − p. (27)

lim

lim

ϵ→0

N→∞

Term limϵ→0,N→∞ E[Yi,j|Xi,j = x] differs from limϵ→0,N→∞ E[Yi|Xi = x] by a scaling factor √ 1

.

p(1−p)

In Proposition C.1 (proof in Appx. B.9), Case 1 corresponds to regular GRPO with sufficiently large group size. In this case, E[Yi|Xi = 1] and E[Yi|Xi = 0] are, respectively, the advantage estimates of positive and negative trajectories given a prompt, dependent on the success probability pq. A large G will lead to a better estimate of the success probability pq. Case 2 corresponds to 2-GRPO, where E[Yi,j|Xi,j = 1] and E[Yi,j|Xi,j = 0] are advantage estimates, which are also dependent on the success rate pq, amortizing over multiple stochastic updates.

2-GRPO produces advantage estimates that differ from standard GRPO solely by a scaling factor; this factor is effectively a design choice. Whether such a scaling is beneficial remains an open question [19].

##### C.2 Key of Variance Reduction: the Training Batch Size, not the Group Size

Beyond the inherent variance reduction mechanisms of PPO and GRPO, it is generally understood that using a larger group of rollouts yields a lower-variance policy gradient estimate. However, this perspective overlooks the practicalities of mini-batch optimization. In this section, we analyze the practical gradient variance within a mini-batch setting. To facilitate this discussion, we focus strictly on the optimization phase and treat the sampled rollouts as fixed training data for notational simplicity.

Note that there are two notions of “batch size” in VERL: data.train_batch_size denotes the rollout-generation batch size (by # prompts), whereas actor.ppo_mini_batch_size denotes the optimization mini-batch size (by # prompts). However, the effective number of samples during optimization is actually actor.ppo_mini_batch_size * rollout.n, counted by the number of rollouts.

Definition C.2 (Variance of Gradient Estimate in Mini-Batch). Without loss of generality, let {xi}Bi=1 be a batch of B random variables (r.v.’s), where each xi is i.i.d. x ∼ D, and let g(xi) = ∇θLθ(xi)

denote the gradient of Lθ(xi) w.r.t. θ. Define the empirical batch gradient gˆB = B1 Bi=1 g(xi). Note that g(xi) and gˆB are dependent r.v.’s of xi and {xi}Bi=1, respectively. We denote the expectation of the gradient g¯ = Ex∼D[g(x)]. The variance of the gradient estimate over the batch is then defined as:

i}Bi (gˆB − g¯)2 . (28)

i}Bi (gˆB) = E{x

Var(gˆB) = Var{x

Following the definition of Variance of Gradient Estimate in Mini-batch (Def. C.2), we provide a lemma for its relationship to the mini-batch size.

i=1,{xi}B

##### Lemma C.3. Let {xi}B

i=1 be two batches of B1 and B2 r.v.’s, respectively. Let gˆB

,gˆB

2

1

1

2

denote the empirical batch gradients of these two batches, respectively. If B1 < B2, then Var[ˆgB

] > Var[ˆgB

1

].

2

While decreasing the group size in Eq. (7) appears to increase the gradient variance for each individual prompt, this conclusion overlooks the total number of rollouts optimized across all prompts in a mini-batch, which is the effective number of examples for optimization. In Lemma C.3 (proof in Appx. B.10), we show that a larger batch size B naturally leads to a lower variance of the gradient. Note that B is the number of rollouts in each mini-batch rather than the number of prompts.

The actual calculation of GRPO is:

1 QG

JGRPO(θ,G,Q) =

Q

G

AijπθGRPO(oij|qj), (29)

i=1

j=1

where

|oi|

G

πθ(oi,t|oi,<t,q) πθ

1 G

1 |oi|

πθGRPO(o|q) =

Cϵ Ai,t

(oi,t|oi,<t,q)

old

t=1

i=1

and Q is the number of prompts in the mini-batch, and the batch size w.r.t the number of rollouts is B = QG. When we decrease G, we can increase Q to compensate to retain the same B in a mini-batch. Since the total number of prompts in the dataset is fixed, increasing Q does not increase the total computational cost per training epoch.

##### C.3 Exploration on Hard Questions

A difficult question often requires many attempts to yield a correct answer, which is necessary to form a valid contrastive signal. With a smaller group, the likelihood of sampling a correct response in a single iteration may appear lower, potentially raising concerns about degraded learning.

Under a fixed computational budget, 2-GRPO and 16-GRPO explore approximately the same total number of rollouts across all training epochs – the overall probability of sampling a correct answer under 2-GRPO is not lower than 16-GRPO, according to the Proposition C.4.

Proposition C.4. Let pi ∈ [0,1] denote the probability that a single rollout under the policy πi produces a correct answer. Then:

- 1. The probability of obtaining at least one correct answer in 2m independent rollouts with policy π0 is

P2m = 1 − (1 − p0)2m. (30)

- 2. The probability of obtaining at least one correct answer when performing m consecutive trials of 2 independent rollouts each, with the corresponding policy [π0,π1,··· ,πm−1] is

(1 − pi)2 ≥ 1 − (1 − p0)2m = P2m (31)

Pm×2 = 1 −

i=0,···m−1

when we have pi ≥ p0,∀i > 0.

Note that the assumption pi ≥ p0,∀i > 0 is prevailing, as we assume that the reasoning ability of LLM can be improved by RL post-training.

Proposition C.4 suggests that for difficult questions, 2-GRPO does not degrade in effectiveness compared to 16-GRPO given the same budget of the total number of rollouts in whole training process. Notably, due to its higher frequency of policy updates, 2-GRPO may yield a higher probability of generating correct outputs for hard questions. It is also more adaptive, allowing it to capture nuanced update requirements for varying inputs. This observation also extends to PPO with the standard single-rollout implementations per epoch against multi-rollout variants.

### D Experiments

##### D.1 Experiment Details

Dataset and Baselines For math reasoning task, following prior work [6], we employ Qwen2.5Math-1.5B (Qwen-1.5B) and Qwen2.5-Math-7B (Qwen-7B) [41] as base models. Both models are post-trained via RL on the MATH [14] and DAPO-Math-17k [42] datasets, and evaluated on MATH500 [14], AMC23, Minerva Math [18], AIME-2025, and OlympiadBench [11]. For DAPO-Math-17k dataset, we randomly sample 7.5k questions from the original data to form a subset for training in order to align with the size of MATH. In addition, we assess the proposed method on DeepSeekR1-Distill-Qwen-1.5B (DS-1.5B) [7], which is post-trained on MATH. Owing to computational constraints, we do not extend its post-training to DAPO-Math-17k. All 1.5B models are trained on

- 4 GPUs with 140GB Memory. Qwen-7B is trained on 8 GPUs with 140GB Memory. We evaluate model performance using two metrics: Mean@32, the average accuracy across 32 i.i.d. samples, and Pass@32, which measures whether a problem is solved in at least one of those 32 attempts.

For visual reasoning task, we use EasyR1 [48] framework, Qwen2.5-7B [2] as the base model, and Geometric3K [23] as the dataset. For code generation task, we use Code-R1 [21] framework, Qwen2.5-7B-Instruct-1M as the base model, and code-r1-12k7 as the dataset. Both visual reasoning and code generation tasks are conducted on 8 GPU.

Hyper-parameters We mainly follow the default configuration of the verl framework. For sampling parameters in training generation, we set temperature to 1, top-p to 1 to encourage exploration, sequence length to 4096 for Qwen-series model and 8192 for DS-1.5B. For sampling parameters in test generation, we set temperature to 0.7, top-p to 0.8, top-k to 20 and sequence length to 4096 for all models. For optimization, training employs the Adam optimizer [17] with a constant learning rate and a linear warm-up over the first 10 steps. For GRPO hyper-parameters, we set the clip ratio high to 0.28 and clip ratio lower to 0.2 following DAPO [42]. All models are trained for 10 epochs. The baseline method, 16-GRPO, is trained with batch sizes of 32 (32 prompts and 16 rollouts per prompt) and a learning rate 1 × 10−6. As discussed in Appx. C.2, we trained 2-GRPO with a larger batch size of 256 (256 prompts and 2 rollouts per prompt). Both case will have 512 rollouts in each mini-batch of training. Since we have fewer update steps due to the larger batch size, we adjust the learning rate of 2-GRPO to 8 × 10−6 based on the linear relationship of learning rate and batch size [10].

##### D.2 The Connection Between Training Rollouts and Computational Cost

In Sec. 5.1, the total number of rollouts generated and utilized during training is adopted as a metric for comparing the computational cost of different methods.

The rationale for this choice is as follows. A principled measure of computational cost in the context of RL post-training is the number of floating-point operations (FLOPs) performed. Unlike wallclock time, which is susceptible to variations arising from software implementation details (e.g., optimization of training libraries) and hardware characteristics (e.g., GPU/CPU architecture, I/O throughput), FLOPs provide a more direct and stable measure of computational effort.

For a fixed base model and the same type of RL algorithm (GRPO in our case), the FLOPs required for a single forward or backward pass with one input prompt can be considered constant, for both the generation and training phases. Accordingly, the total number of rollouts executed during training is directly proportional to the FLOPs executed, thereby serving as a theoretically justified and consistent proxy for computational cost.

7https://huggingface.co/datasets/ganler/code-r1-12k

##### D.3 Sample Discard Rate of 2-GRPO

As discussed in Sec. 4.3, 2-GRPO may suffer from a high discard rate when prompts are either extremely easy or extremely difficult for the LLM.

In the RLVR setting, the average discard rate Pdiscard can be estimated from the average reward r¯ as Pdiscard = 1 − r¯2 − (1 − r¯)2.

To quantify this effect, we report the average discard rate of 2-GRPO using Qwen-7B post-trained on MATH as a representative case (shown in Table 3).

- Table 3: The discard rate vs. training steps of 2-GRPO for Qwen-7B post-trained on MATH dataset.

Step 10 40 70 100 130 Pdiscard 0.5284 0.6917 0.6763 0.7711 0.7141

##### D.4 Ablation Study on Group Size

0.80

0.75

0.70

ValidationScores

0.65

0.60

0.55

G=16

0.50

G=8 G=4 G=2

0.45

0.40

0 50 100 150 200 250

Time (mins)

(a) Training reward during training.

0.70

0.68

0.66

ValidationScores

0.64

0.62

0.60

G=16

G=8 G=4 G=2

0.58

0.56

0 50 100 150 200 250

Time (mins)

(b) Evaluation score on the Test set.

Figure 5: Reward curves and validation scores of different group size on MATH. Curves are post simple-moving-average (SMV) with window-size=4 for better visualization, respectively.

We present a comprehensive ablation study on group size in Table 4. To further illustrate this effect, Figure 5 reports the reward curves during training alongside the corresponding validation scores throughout the training process on MATH dataset.

### E Limitation

The contrastive learning nature of GRPO applies regardless of whether rewards are continuous or binary. However, the present study focuses primarily on the reasoning tasks with the RLVR setting, and we leave the empirical investigation of continuous rewards to future work due to the space limit.

#### F The Use of Large Language Models (LLMs) We used LLMs in writing, editing and formatting purposes. Our experiments also involve the LLMs.

- Table 4: Ablation study on group size G: post-trained on MATH and DAPO, respectively, and evaluated on five mathematical reasoning benchmarks. M/P@32 stands for Mean@32 and Pass@32. #P and #R denote the number of prompts and the number of rollouts in a training mini-batch.

Model G Batch (#P/#R) MATH-500 AMC 2023 Minerva Math AIME 2025 Olympiad Bench Average

Mean@32 / Pass@32 Post-training on MATH dataset

Qwen-1.5B

w/o – 31.83 / 81.92 34.30 / 79.23 5.33 / 28.91 3.64 / 22.31 15.40 / 37.16 18.10 / 49.91 2† 256/512 69.28 / 87.43 49.53 / 81.76 16.25 / 33.26 9.48 / 32.88 22.31 / 37.24 33.37 / 54.51 2 32/64 67.73 / 87.85 53.28 / 86.21 14.15 / 34.02 6.15 / 29.54 23.11 / 37.82 32.88 / 55.09 4 32/128 69.05 / 87.49 52.50 / 92.01 15.29 / 33.57 8.33 / 27.13 23.08 / 38.99 33.65 / 55.84 8 32/256 69.34 / 86.05 51.64 / 83.96 14.60 / 32.63 7.18 / 32.24 22.77 / 36.69 33.11 / 54.31 16 32/512 70.24 / 87.24 51.25 / 83.46 16.84 / 33.46 10.10 / 35.82 22.30 / 38.33 34.15 / 55.66

###### Qwen-7B

w/o – 47.16 / 85.95 38.36 / 85.29 5.99 / 31.10 5.00 / 25.17 9.83 / 34.30 21.27 / 52.36 2† 256/512 75.23 / 89.77 64.60 / 81.53 23.13 / 38.45 12.81 / 38.85 26.39 / 40.20 40.43 / 57.76 2 32/64 74.41 / 89.25 63.83 / 89.58 21.53 / 37.72 11.67 / 33.05 26.04 / 41.34 39.50 / 58.19 4 32/128 76.24 / 88.16 63.51 / 84.97 23.09 / 41.03 10.83 / 32.42 26.25 / 40.78 39.98 / 57.47 8 32/256 75.12 / 89.53 64.38 / 88.63 22.24 / 35.94 12.71 / 35.85 26.25 / 40.52 40.14 / 58.09 16 32/512 75.90 / 88.24 61.79 / 80.77 22.81 / 37.68 13.23 / 34.22 25.99 / 40.11 39.94 / 56.20

Post-training on DAPO-Math-Sub dataset Qwen-1.5B

w/o – 31.83 / 81.92 34.30 / 79.23 5.33 / 28.91 3.64 / 22.31 15.40 / 37.16 18.10 / 49.91 2† 256/512 68.81 / 87.36 52.19 / 85.77 16.79 / 33.61 8.13 / 29.33 23.52 / 39.29 33.89 / 55.07 2 32/64 67.71 / 87.68 53.82 / 88.35 16.85 / 34.83 8.12 / 32.99 23.21 / 39.26 33.94 / 56.62 4 32/128 69.14 / 87.78 54.69 / 86.88 17.53 / 35.74 8.43 / 36.18 23.30 / 39.00 34.62 / 57.12 8 32/256 70.25 / 86.84 57.57 / 81.19 17.80 / 35.08 8.54 / 29.42 24.23 / 39.95 35.68 / 54.50 16 32/256 70.66 / 87.03 56.56 / 85.53 18.00 / 34.16 9.58 / 32.31 24.55 / 39.19 35.87 / 55.64

###### Qwen-7B

w/o – 47.16 / 85.95 38.36 / 85.29 5.99 / 31.10 5.00 / 25.17 9.83 / 34.30 21.27 / 52.36 2† 256/512 77.43 / 90.51 64.84 / 91.59 21.95 / 38.05 14.58 / 33.03 29.86 / 45.24 41.73 / 59.68 2 32/64 75.24 / 89.37 66.33 / 90.56 23.49 / 39.89 15.21 / 41.71 28.60 / 43.04 41.77 / 60.91 4 32/128 76.58 / 90.62 66.33 / 95.75 23.44 / 40.06 15.21 / 40.02 27.94 / 42.24 41.90 / 61.74 8 32/256 72.43 / 87.27 71.17 / 93.08 25.03 / 37.94 17.71 / 39.67 28.74 / 41.46 43.02 / 59.88 16 32/512 77.35 / 88.79 69.69 / 87.31 24.45 / 40.04 14.27 / 33.73 28.86 / 39.84 42.92 / 57.94

†2-GRPO with larger batch size (256 prompts / 512 rollouts).

