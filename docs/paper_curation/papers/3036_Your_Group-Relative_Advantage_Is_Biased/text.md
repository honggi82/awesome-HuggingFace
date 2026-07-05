### Your Group-Relative Advantage Is Biased

Fengkai Yang1,3,4, Zherui Chen2, Xiaohan Wang4, Xiaodong Lu1,4, Jiajun Chai4, Guojun Yin4, Wei Lin4, Shuai Ma1, Fuzhen Zhuang1, Deqing Wang1, Yaodong Yang3, Jianxin Li1, Yikun Ban1* 1Beihang University 2University of California, Berkeley 3Peking University 4Meituan

# arXiv:2601.08521v2[cs.LG]22Jan2026

#### Abstract

Reinforcement Learning from Verifier Rewards (RLVR) has emerged as a widely used approach for post-training large language models on reasoning tasks, with group-based methods such as GRPO and its variants gaining broad adoption. These methods rely on group-relative advantage estimation to avoid learned critics, yet its theoretical properties remain poorly understood.

In this work, we uncover a fundamental issue of group-based RL: the group-relative advantage estimator is inherently biased relative to the true (expected) advantage. We provide the first theoretical analysis showing that it systematically underestimates advantages for hard prompts and overestimates them for easy prompts, leading to imbalanced exploration and exploitation. To address this issue, we propose History-Aware Adaptive Difficulty Weighting (HA-DW), an adaptive reweighting scheme that adjusts advantage estimates based on an evolving difficulty anchor and training dynamics. Both theoretical analysis and experiments on five mathematical reasoning benchmarks demonstrate that HA-DW consistently improves performance when integrated into GRPO and its variants. Our results suggest that correcting biased advantage estimation is critical for robust and efficient RLVR training.

#### 1 Introduction

After the success of DeepSeek-R1 (DeepSeek-AI, 2025), RLVR has rapidly emerged as a simple yet powerful paradigm for training reasoning-oriented LLMs. GRPO (Shao et al., 2024) has gained increasing popularity after PPO (Schulman et al., 2017). Numerous variants of GRPO have been proposed to improve the algorithm, with the goal of achieving better stability and performance. Common variants include GSPO (Zheng et al., 2025),

*Corresponding Author. If you have any questions, feel free to contact yikunb@buaa.edu.cn or yangfengkai@stu.pku.edu.cn

DAPO (Yu et al., 2025), Dr.GRPO (Liu et al., 2025b) and GMPO (Zhao et al., 2025).

In post-training, intra-group advantage estimation is critical to the performance of group-relative RL algorithms. Typically, for each sampled prompt, the algorithm generates only a small number of rollouts and uses the within-group average reward as a baseline to compute advantages, thereby avoiding the need for a separate critic model. While this design is appealing and has attracted broad interest in the RL community, it still lacks a detailed theoretical characterization(Xiong et al., 2025; Tan et al., 2025).

###### Your advantage estimation is biased.

In this paper, we identify a fundamental issue in group-based RL: the group-relative advantage estimator is generally biased relative to the true (expected) advantage. We provide a theoretical analysis showing that for hard prompts, the estimator tends to underestimate the expected advantages, whereas for easy prompts, it tends to overestimate the expected advantages, as presented in Section 2.2. Such systematic bias can cause the policy to under-learn from hard questions while over-exploiting easy ones, ultimately hurting both training stability and generalization. As illustrated by the representative example in Figure 1(b), grouprelative estimation can introduce substantial bias in advantage estimation for group-based RL algorithms. Our empirical results further corroborate this phenomenon, with consistent evidence reported in appendix E.1.

Motivated by these findings, we propose a novel policy optimization algorithm that adaptively reweights advantage estimates to mitigate the bias induced by group-based advantage estimation. The overall framework is depicted in Figure 3. Our main contributions are summarized as follows:

[Discovery]. We provide the first theoretical analysis revealing that group-based advantage estima-

###### (a) (b) (c)

[Figure 1]

[Figure 2]

[Figure 3]

Figure 1: (a) Comparison of the performance of RL algorithms with and without HA-DW on Qwen3-4B-Base across five mathematical reasoning benchmarks. (b) Significant biased advantage estimation on the MATH dataset under 8 and 128 rollouts. (c) Performance gain by GRPO+HA-DW on MATH500 stratified by difficulty levels.

tion in RLVR is inherently biased, systematically underestimating advantages for hard prompts and overestimating them for easy prompts.

policy πθt(· | xt). Each response yt,i receives a corresponding scalar reward rt,i ∈ {0,1}, forming the reward set {rt,i}Gi=1, where r(·) is the reward function and we denote r(yt,i) by rt,i for brevity. The group-relative policy optimization (Group-PO) objective is defined as:

[Algorithm]. Motivated by this fundamental discovery, we propose History-Aware Adaptive Difficulty Weighting (HA-DW), which dynamically adjusts advantage weights using an evolving difficulty anchor that integrates long-term reward trends and historical training information. HA-DW compensates for the bias induced by group-relative advantage estimation and enables a more principled balance between exploration and exploitation in RL training.

Å

ã

Ä

ä

G

πθ(yt,i | xt) πθold(yt,i | xt)

1 G

Aˆt,i

Jgroup(θ) =

ψ

ϕ

, (1)

i=1

where πθold denotes the reference (behavior) policy.

The group-relative advantage Aˆt,i is computed as:

[Performance]. As illustrated in Figure 1(a), we validate our approach through extensive experiments on mathematical reasoning benchmarks, demonstrating consistent performance improvements when integrated HA-DW with GRPO and its variants across model scales. Notably, even when compared with GRPO using a larger number of rollouts, our method still achieves superior results.

G

1 G

Aˆt,i = rt,i − pˆt, pˆt =

rt,i, (2)

i=1

where pˆt is the group baseline:

Here, ψ(·) denotes a function applied to the importance sampling ratio (e.g., identity, clipping, or logarithmic transformation), and ϕ(·) denotes a function applied to the advantage term, introduced to maintain generality across different grouprelative policy optimization variants.

Our goal is not to model all RLVR settings, but to expose a previously overlooked statistical bias in group-relative algorithms and demonstrate that even lightweight corrections can yield consistent gains.

Definition 1 (Expected Reward). Assume the reward function r(·) is binary, i.e., r(·) ∈ {0,1}. Given a prompt xt ∼ D and a policy πθt, let yt ∼ πθt(· | xt) be a sampled response. The expected reward of policy πθt on prompt xt is defined as:

#### 2 Why Your Advantage Estimation is Biased?

In this section, we theoretically analyze the biased estimation in group-relative algorithms. Firstly, we provide the prerequisite definitions.

pt = Eyt∼πθ

t(·|xt) r(yt) = P(r(yt) = 1 | xt,πθt).

(3)

In the RLVR setting, pt represents the expected reward under policy πθt given xt, while pˆt can be regarded as an empirical estimator of pt obtained from a finite group of sampled responses. This motivates the following definition.

###### 2.1 Definitions

At training step t, we sample a prompt xt ∼ D. Given xt, a group-relative RL algorithm samples G responses {yt,i}Gi=1 independently from the current

- Definition 2 (Expected Advantage). Given a

prompt xt ∼ D, let yt,i ∼ πθt(· | xt) be a sampled response with corresponding reward rt,i. The expected advantage is defined as:

At,i = rt,i − pt. (4)

Thus, in the RLVR setting, At,i represents the expected advantage of response yt,i under policy πθt

given xt, while Aˆt,i can be regarded as an empirical estimator of At,i obtained from a finite group of sampled responses. Most group-relative RL algorithms rely on Aˆt,i for policy updates, differing primarily in how Aˆt,i is processed or transformed within their respective optimization objectives.

2.2 Fundamental Discovery

Next, we present a formal formulation of the problem. Given a prompt xt ∼ D, let pt denote the expected reward of policy πθt on xt. We then sample G responses independently from πθt(· | xt). In RLVR, rewards are often binary, especially in mathematical and formal reasoning tasks where verifiers return pass/fail signals. Under this widely adopted setting, it is natural to model the reward associated with each response as a Bernoulli random variable:

rt,i ∼ Bernoulli(pt), ∀i ∈ [G]. (5)

Let R = Gi=1 rt,i denote the total reward within the group. The empirical group baseline is given by pˆt = R/G.

- Definition 3 (Prompt Difficulty). Given a prompt

xt, a policy πθt, and ∆ ∈ [0,1), we define the difficulty of xt as follows:

- • xt is a hard prompt if pt < 0.5 − ∆;
- • xt is a moderate prompt if 0.5 − ∆ ≤ pt ≤ 0.5 + ∆;
- • xt is a easy prompt if pt > 0.5 + ∆,

where ∆ is a user-defined threshold to customize the prompt difficulty.

In group-based policy optimization, the grouprelative advantage estimator satisfies Aˆt,i = 0 for all i ∈ [G] when either R = 0 or R = G, resulting in zero gradients and hence no parameter updates. In practice, such degenerate groups do not contribute to learning and are either explicitly discarded or implicitly ignored by GRPO-style algorithms.

Accordingly, our analysis focuses on the effective update regime, namely groups for which at least one response receives a non-zero advantage. This corresponds to the non-degenerate event

###### S := {1 ≤ R ≤ G − 1}. (6)

Importantly, conditioning on S does not alter the optimization trajectory, but isolates the subset of samples that actively drive learning, allowing us to precisely characterize the bias in advantage estimation. Next, we present the main results.

Theorem 1. Given a prompt xt ∼ D, let yt,i ∼ πθt(· | xt) denote a sampled response with reward rt,i. Suppose G ≥ 2, and condition on the event S = {1 ≤ R ≤ G − 1}. Then, for any i ∈ [G], we have:

E Aˆt,i | S < At,i, if pt < 0.5; E Aˆt,i | S > At,i, if pt > 0.5; E

î

ó

Aˆt,i | S

= At,i, if and only if pt = 0.5.

Theorem 1 shows that the expectation of the group-based advantage estimator Aˆt,i is lower than the true advantage At,i for difficult prompts, and larger than At,i for easy prompts. The estimator is unbiased only when pt = 0.5. This bias is amplified as pt deviates from 0.5 and G is smaller in Figure 2 based on Lemma 2.

However, the expectation-level result in Theorem 1 alone is insufficient to characterize the probability of overestimation or underestimation of Aˆt,i. We provide the following probabilistic result.

Theorem 2. Under the condition of Theorem 1, suppose xt is a hard prompt (pt < 0.5). Then, for any ϵ ∈ (0,E[ˆpt | S] − pt), we have:

Ä

ä

At,i − Aˆt,i > ϵ | S

P

G−1 k=⌊G(pt+ϵ)⌋+1

G k pkt (1 − pt)G−k

=

.

1 − (1 − pt)G − pGt

Simlarily, suppose xt is an easy prompt (pt > 0.5). Then, for any ϵ ∈ (0,pt − E[ˆpt | S]), we have:

Ä

ä

Aˆt,i − At,i > ϵ | S

P

⌈G(pt−ϵ)⌉−1 k=1

G k pkt (1 − pt)G−k

.

=

1 − (1 − pt)G − pGt

Theorem 2 provides a distribution-level characterization of how likely group-relative advantage estimation is to underestimate or overestimate the true advantage, depending on prompt difficulty. In contrast to expectation-level results, this theorem quantifies the exact probability mass of large estimation errors under finite group sizes.

It is well known that generating multiple rollouts per prompt is computationally expensive in practice. Consequently, existing RLVR methods typically sample only a small number of responses (e.g., G = 8) for each prompt xt to estimate pˆt (Zhang et al., 2025; Liu et al., 2025a; Shen et al., 2025) . Motivated by this practical constraint, we derive the following corollaries based on Theorem 2, which explicitly characterize the estimation behavior under small group sizes.

Corollary 1. Under the condition of Theorem 2, suppose the group size satisfies 2 ≤ G ≤ 8, and assume that pt is uniformly distributed over [0,1]. Then, for any i ∈ [G], the following inequalities hold:

P Aˆt,i < At,i | S, pt < 0.5 > 0.63, P

Ä

Aˆt,i > At,i | S, pt > 0.5 > 0.63, P Aˆt,i < At,i | S, pt < 0.25 > 0.78, P

Ä

Aˆt,i > At,i | S, pt > 0.75 > 0.78 P Aˆt,i < At,i | S, pt < 0.125 = 1.00, P

ä

Ä

Aˆt,i > At,i | S, pt > 0.875

= 1.00.

- Corollary 1 shows that, with high probability,

the group-relative advantage estimator Aˆt,i underestimates the true advantage At,i for hard prompts and overestimates At,i for easy prompts, under the practical set of G. Moreover, as the prompt difficulty becomes more extreme (i.e., as ∆ increases), this bias becomes more pronounced, which is also demonstrated in Colloary 2.

- Corollary 2. Under the condition of Corollary 1, suppose G ≥ 6. The following inequalities hold:

P Aˆt,i < At,i | S, pt <

2 G

ã

> 0.78,

P

Å

Aˆt,i > At,i | S, pt >

G − 2 G

ã

> 0.78.

- Corollary 3. Under the condition of Theorem 2, suppose G ≥ 2. Then, for any i ∈ [G], the following inequalities hold surely:

Aˆt,i < At,i, if pt < G1 , Aˆt,i > At,i, if pt > GG−1.

G=2 G=4 G=6 G=8

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.5

0.4

||AAS[|]titi,,

0.3

0.2

0.1

0.0

0.0 0.2 0.4 0.6 0.8 1.0

pt

Figure 2: Illustration of advantage bias |At,i − E[Aˆt,i | S]| as a function of pt and group size G.

Corollary 3 shows that the group-based advantage estimator Aˆt,i necessarily underestimates the true advantage At,i for extremely difficult prompts (pt < 1/G), and overestimates At,i for extremely easy prompts (pt > (G − 1)/G). Detailed derivation process is presented in appendix D.

Discovery. Group-relative advantage is provably biased except at pt = 0.5. Specifically, it systematically underestimates the true advantage for hard prompts and overestimates it for easy prompts. Moreover, this bias becomes deterministic in extreme regimes: the estimator necessarily underestimates the true advantage for extremely difficult prompts and necessarily overestimates it for extremely easy prompts.

Discussion. While the binary reward assumption covers many standard RLVR setups

— particularly those using hard verifier outcomes — we recognize that real-world reward signals can be more general. To address this, we extend our analysis in Appendix D.5 to continuous bounded reward distributions. The results suggest that the core bias phenomenon is not an artifact of the Bernoulli reward assumption but is prevalent across a broader class of bounded reward models.

#### 3 Proposed Solution

Since the group-based advantage estimator is biased, we propose an algorithm to adjust the advantage estimation accordingly. The proposed approach consists of two key components. First, we introduce a framework that incorporates crossbatch information into RL training, enabling a

Relative Difficulty Reweighting

[Figure 4]

∆∝ 𝐞𝐱𝐩(𝑫𝒕,𝒊 𝐂𝐭 )

## HA-DW

Correct(reward) Incorrect(penalty)

Introducing long-term reward signal

Advantage Advantage

reweight

Boosting Exploration(hard) Weakening Exploitation(easy)

Biased Es ma on

###### Batch Prior Ct- Observation 𝒚𝒕

more rollouts

𝚽𝒕,𝒊 𝑨𝒕,𝒊

History Batch

𝐶 = 1 − 𝜂 𝐶 + 𝜂 𝑦

C2- C3- Cn-

###### 𝑨𝒕,𝒊 ·········

Evolving Capability

C1 C2 C3 Cn-1 Cn

Step 1 Step 2 Step 3 Step n-1 Step n

Group with only 1 Correct Response

- Figure 3: HA-DW consists of two collaborative phases. In the first phase, an evolving difficulty anchor incorporates cross-batch historical information by propagating the model’s prior through a history buffer, capturing long-term reward trends. In the second phase, prompt weights are adaptively adjusted based on their estimated difficulty under the model’s evolving state, compensating for biased advantage estimates.

history-aware anchor for prompt difficulty. Second, we design an adaptive advantage reweighting algorithm to correct the induced bias.

The adaptive forgetting factor is then given by:

ηt = η · σt, (11)

where η is a task-dependent hyperparameter. Intuitively, a larger ηt is used during early training stages to capture rapid capability shifts, while a smaller ηt is adopted in later, more stable stages to preserve historical information and reduce noise.

###### 3.1 Evolving Difficulty Anchor

To track the evolving model state across batches, we propose the cross-batch difficulty anchor framework that integrates long-term reward trends and historical information. Let Bt denote the total number of responses in batch t. Model updates are guided by observations of the current batch’s prompt accuracy yt together with historical information, defined as:

Between consecutive steps, the posterior belief Ct+ serves as the prior belief for the next batch:

Ct+ → Ct−+1. (12)

Overall, Ct enables the model to aggregate information across historical batches via belief updates and to condition its training strategy on this evolving belief. This evolving belief serves as a historyaware anchor for the subsequent difficulty-adaptive reweighting strategy. We also provide an alternative, hard update variant of Ct in Appendix F.

Bt

Kt Bt

rt,i. (7)

yt =

, Kt =

i=1

We treat the model’s solving capability Ct as a latent belief state. At training step t, the observation yt is used to update the prior belief Ct− to the posterior belief Ct+ via a Kalman-style update (Battilotti et al., 2026; Zhang, 2026):

3.2 History Aware Adaptive Difficulty Weighting (HA-DW)

Ct+ = (1 − ηt)Ct− + ηt yt, ηt ∈ [0,1]. (8)

To rectify the inherent bias in group-based advantage estimation, we introduce HA-DW, which dynamically adjusts advantage weights based on the model’s evolving state while incorporating longterm reward signals. Coupled with the evolving difficulty anchor, we define the history-based prompt difficulty as:

The forgetting factor ηt controls the influence of historical information and is dynamically modulated by model stability. Specifically, we compute the average belief over the previous m batches as:

m

1 m

C¯t =

Ct−j, (9)

j=1

diffhist = pˆt − Ct−, (13)

and define the corresponding standard deviation:

Ã

where diffhist captures both the magnitude and direction of a prompt’s difficulty relative to the current model belief.

m

1 m

Ct−j − C¯t 2. (10)

σt =

j=1

To determine the direction of adjustment, we use the evolving difficulty anchor as a reference and define:

ä

Ä

ä

Ä

Aˆt,i

diffhist

, (14) where sgn(·) denotes the sign function.

· sgn

Dt,i = −sgn

Next, we quantify the magnitude of adjustment using the absolute history-based difficulty:

###### Mt = diffhist . (15)

Here, Mt measures the extent to which the prompt deviates from the model’s current capability.

We then define the history-aware reweighting factor as:

Φt,i = λscale · exp(Dt,i · Mt), (16)

where λscale is a scaling constant, and the exponential form ensures smooth and multiplicative adjustment of advantage weights. The resulting HA-DW objective is:

Å

ã

G

πθ(yt,i | xt) πθold(yt,i | xt)

1 G

·

LHA-DW(θ) =

ψ

(17)

i=1

Ä

ä

Aˆt,i

· Φt,i,

ϕ

where ψ(·) and ϕ(·) follow specific definitions in group-relative RL algorithms.

Intuitively, Φt,i amplifies the estimated advantage for difficult prompts—where group-based estimation tends to be conservative—and suppresses it for easy prompts—where overestimation is prevalent—thereby correcting systematic bias identified in our analysis. HA-DW can be seamlessly integrated as a plug-and-play module into GRPO and its variants, improving reasoning performance under fixed rollouts while effectively mitigating biased advantage estimation. Detailed instantiations for GRPO and related algorithms are provided in the appendix B

#### 4 Theoretical Analysis

In this section, we provide a theoretical analysis of the effectiveness of the proposed adjustment strategy. We begin by analyzing how reweighting the empirical baseline pˆt affects the expected bias. Lemma 1 (Baseline Rectification). Given a prompt xt ∼ D and the policy πθt, let p˜t = c · pˆt be the rectified group baseline. Assume pt ∈ [∆, 1 − ∆]

for some ∆ ∈ (0,1/2]. Given any δ ∈ (0,1), we can define that:

å

Ç

- 1

- 2G

2 δ 1 − (1 − ∆)G − ∆G

.

ϵδ :=

log

(18) Let

It := p ˆt − ϵδ, pˆt + ϵδ ∩ [∆,1 − ∆], A(p) := 1 − (1 − p)G − pG.

Fix any ϵ > 0, we define:

(p − ϵ)A(p) p(1 − pG−1)

, (19)

clow := sup p∈It

and:

(p + ϵ)A(p) p(1 − pG−1)

. (20)

chigh := inf p∈It

Then, with probability at least 1 − δ conditional on S, for any choice

c ∈ (clow, chigh), (21) we can derive that:

E[˜pt | S] ∈ (pt − ϵ, pt + ϵ).

Specifically, we consider adjusting the empirical group baseline using a reweighting factor c. From the perspective of the expected estimation bias, Lemma 1 that an appropriate choice of c can effectively reduce estimation bias. Detailed derivations are provided in Appendix D.4. Next, we now proceed to present the main theoretical result.

Theorem 3. Under the condition of Lemma 1, suppose there exists a scaling factor λscale in Equation (16) such that:

Ñ

exp(Dt,iMt) é

1 + (1−1c−highpˆ )ˆpt

1 + (1−1c−lowpˆ )ˆpt

t

t

λscale ∈

,

exp(Dt,iMt)

Å

ã

chigh exp(Dt,iMt)

clow exp(Dt,iMt)

∪

,

.

Then, HA-DW algorithm provably mitigates the bias of group-relative advantage:

î

ó

Aˆt,i · Φt,i | S

E

− At,i < E

î

ó

Aˆt,i | S

− At,i .

Algorithm MATH500 AIME25 AMC23 Minerva OlympiadBench AVG Qwen-3-4B-Base

GRPO 75.4 19.6 60.3 33.8 43.5 46.5 → + HA-DW 78.0 20.4 63.4 36.8 44.7 48.7 GSPO 75.8 20.0 62.2 35.3 42.3 47.1

- → + HA-DW 77.6 19.6 68.6 37.1 43.2 49.2 DAPO 76.8 18.3 60.0 35.7 43.2 46.8

- → + HA-DW 78.6 21.3 65.0 37.5 45.3 49.5 Qwen-3-8B-Base

GRPO 78.8 20.4 64.2 38.2 46.4 49.6 → + HA-DW 80.0 22.9 72.8 39.7 47.1 52.5 GSPO 78.6 21.7 67.0 37.9 45.9 50.2 → + HA-DW 80.2 22.1 66.5 41.9 47.6 51.7 DAPO 79.2 20.4 67.5 39.3 47.2 50.7

→ + HA-DW 82.8 23.3 70.0 40.8 50.0 53.4 LLaMA-3.2-3B-Instruct

GRPO 51.4 2.7 31.7 22.8 19.9 25.7 → + HA-DW 53.2 3.3 35.0 23.9 20.1 27.1 GSPO 48.6 1.9 30.9 23.2 19.8 24.9 → + HA-DW 50.4 2.3 32.7 22.4 21.0 25.8 DAPO 52.4 2.5 35.0 22.4 20.2 26.5 → + HA-DW 53.2 3.1 37.5 24.6 22.3 28.1

- Table 1: Overall results across models (Qwen, LLaMA) and different group-relative RL algorithms (GRPO, GSPO, DAPO). We report the performance of different base RL algorithms, and the corresponding accuracy when applied HA-DW for each model scale and family.

[Figure 5]

- Figure 4: Comparison of training dynamics under different training strategies. Average accuracy across five benchmarks, training reward and response length of Qwen3-4B-Base and Qwen3-8B-Base on different training methods.

Theorem 3 shows that, with an appropriate choice of the scaling parameter λscale, the HADW adjustment yields advantage estimates that are closer to the true advantage At,i in expectation. This theoretical result provides principled guidance for selecting λscale in practice.

- 5 Experiments

within the VeRL framework (Sheng et al., 2024) on a single node with 8 × NVIDIA A100 GPUs. More implementation details in Appendix C.

###### 5.1 Main Results

Our main results are presented in Table 1. Notably, group-based RL algorithms ( GRPO, GSPO and DAPO ) equipped with HA-DW outperforms original methods across five benchmarks. We observed clear and consistent improvements across benchmarks on different models of different scales and family. Overall, the results underscore that HADW compensates for advantage estimation bias via dynamic reweighting to fully leverage these overshadowed critical prompts, thereby unlocking the potential performance gains in RL.

Setups. We conduct our experiments on Qwen34B-Base, Qwen3-8B-Base (Team, 2025) and LLaMA-3.2-3B-Instruct on five common-used RLVR benchmarks. We apply our proposed method on top of several representative grouprelative reinforcement learning algorithms: GRPO, GSPO, and DAPO. We compare the performance of group-relative algorithms applying HA-DW to original ones, verifying the effectiveness and scalability of our method. We conduct RL training

To validate our method’s effectiveness in extending model capabilities, we divided the MATH500

Threshold MATH500 AIME25 AMC23 Minerva OlympiadBench AVG Base 75.4 19.6 60.3 33.8 43.5 46.5

- 0.4 (fixed) 77.0 18.5 63.1 37.5 44.3 48.1
- 0.5 (fixed) 76.6 20.0 62.7 35.7 44.0 47.8
- 0.6 (fixed) 76.8 21.3 61.1 36.4 44.3 48.0 Ct 78.0 20.4 63.4 36.8 44.7 48.7

- Table 2: Ablation on the effectiveness of dynamic threshold for RL training using Qwen3-4B-Base. Ct denotes the dynamic threshold.

dataset into three difficulty levels: Easy (Level 1), Mid (Levels 2-3), and Hard (Levels 4-5). We evaluated Qwen3-4B-Base trained with GRPO and GRPO+HA-DW, as shown in Figure 1(c). The performance on Easy and Mid levels was comparable for both methods, but GRPO+HA-DW outperformed GRPO by 3.4% on Hard prompts. This improvement is due to our history-based dynamic reweighting strategy, which enhances exploration on hard prompts while reducing unnecessary exploitation on easy ones. Simultaneously, it substantiates the existence of bias estimation indirectly.

Training Dynamics. Figure 4 demonstrates the temporal dynamics of average accuracy across five benchmarks, training rewards throughout the training process, and response lengths of Qwen3-4BBase and Qwen3-8B-Base. RL algorithms applied HA-DW converge to a higher performance plateau in accuracy and acquired higher reward compared to the original RL algorithms, suggesting that the application of HA-DW boosts the exploration of hard prompts and weakens the exploitation of easy ones by mitigating the biased advantage estimation. In addition, our method encourages longer reasoning, greatly improving its reasoning abilities (Jin

- et al., 2024; DeepSeek-AI, 2025). HA-DW is capable of incentivizing the model to produce more sophisticated reasoning chain of thoughts to tackle more challenging tasks.

Ablation Study on Ct. We evaluate the effectiveness of the dynamic threshold Ct by comparing it with a fixed thresholds across five benchmarks, as shown in Table 2. Experiments on Qwen3-4B-Base with GRPO-based training show that dynamic adjustment achieves the best performance. Removing Ct degrades performance, while a fixed threshold still improves over the baseline by partially mitigating biased estimation. By incorporating crossbatch information, Ct captures long-term reward signals and further enhances RL performance.

Supplementary Experiments (Appendix E) Due to space limitations, we include the following

Dataset 8 16 8+HA-DW

MATH500 75.4 76.2 78.0 AIME25 19.6 19.2 20.4 AMC23 60.3 61.6 63.4 Minerva 33.8 34.2 36.8

OlympiadBench 43.5 43.9 44.7

Table 3: Performance of Qwen3-4B-Base trained with: Rollout=8 with GRPO, Rollout=16 with GRPO and Rollout=8 with GRPO+HA-DW. Rollout=32 with GRPO is out of memory.

additional experiments in Appendix E: (1) empirical verification of advantage estimation bias, (2) an ablation study on the group size G (Table 3), and (3) an ablation study on the scaling parameter λscale.

#### 6 Related Work

GRPO and GRPO Variants. Following the success of Deepseek-R1 (DeepSeek-AI, 2025), GRPO has attracted widespread attention. To achieve better performance, numerous GRPO-based variants have been proposed. Dr.GRPO removes heuristic normalizations to obtain more stable, less biased updates. DAPO stabilizes training with decoupled clipping and dynamic sampling. GSPO uses sequence-level ratios and clipping to improve stability and efficiency, especially for large and MoE models. However, these variants adopt static prompt difficulty and suffer from insufficient exploration of model’s capability. More related work are placed in Appendix A.

#### 7 Conclusion

Our work uncovers a fundamental limitation of group-relative RL algorithms: biased advantage estimation. To address this issue, we propose HADW, which dynamically adjusts advantage weights based on the model’s evolving state. Extensive experiments demonstrate that HA-DW effectively improves reasoning performance by mitigating biased advantage estimation.

#### Acknowledgement

Z.C. acknowledges the Challenge Institute for Quantum Computation (CIQC) funded by NSF through grant number OMA-2016245.

#### Limitations

This work reveals an intrinsic limitation of grouprelative RL—namely, biased advantage estimation under non-degenerate sampling—and proposes HA-DW to effectively mitigate this issue. Our study primarily focuses on the issue of group-wise estimation bias, restricting the application of HADW to group-relative methods. Nevertheless, estimation bias is pervasive, and future work will focus on extending this concept to a broader scope.

#### References

Stefano Battilotti, Alessandro Borri, Filippo Cacace, Massimiliano D’Angelo, and Alfredo Germani. 2026. A consensus kalman filter on L2 spaces. Autom., 183:112530.

Stéphane Boucheron, Gábor Lugosi, and Pascal Massart.

2013. Concentration Inequalities - A Nonasymptotic Theory of Independence. Oxford University Press.

Weizhe Chen, Sven Koenig, and Bistra Dilkina. 2025. LSPO: length-aware dynamic sampling for policy optimization in LLM reasoning. CoRR, abs/2510.01459.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948.

Yuyang Ding, Chi Zhang, Juntao Li, Haibin Lin, Xin Liu, and Min Zhang. 2025. FAPO: flawed-aware policy optimization for efficient and reliable reasoning. CoRR, abs/2510.22543.

Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. 2018. IMPALA: scalable distributed deep-rl with importance weighted actorlearner architectures. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 1406–1415. PMLR.

Pietro Di Gianantonio and Abbas Edalat. 2025. A domain-theoretic framework for conditional probability and bayesian updating in programming. CoRR, abs/2502.00949.

Yiran Guo, Lijie Xu, Jie Liu, Dan Ye, and Shuang Qiu. 2025. Segment policy optimization: Effective segment-level credit assignment in RL for large language models. CoRR, abs/2505.23564.

Trevor Hastie, Robert Tibshirani, and Jerome H. Friedman. 2009. The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd Edition. Springer Series in Statistics. Springer.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 3828–3850. Association for Computational Linguistics.

Xinrui He, Yikun Ban, Jiaru Zou, Tianxin Wei, Curtiss Cook, and Jingrui He. 2025. Llm-forest: Ensemble learning of llms with graph-augmented prompts for data imputation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 6921– 6936.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.

Wenke Huang, Quan Zhang, Yiyang Fang, Jian Liang, Xuankun Rong, Huanjin Yao, Guancheng Wan, Ke Liang, Wenwen He, Mingjun Li, Leszek Rutkowski, Mang Ye, Bo Du, and Dacheng Tao. 2025. MAPO: mixed advantage policy optimization. CoRR, abs/2509.18849.

Zixuan Huang, Yikun Ban, Lean Fu, Xiaojie Li, Zhongxiang Dai, Jianxin Li, and 1 others. Adaptive batchwise sample scheduling for direct preference optimization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Nan Jiang and Lihong Li. 2016. Doubly robust offpolicy value evaluation for reinforcement learning. In Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, volume 48 of JMLR Workshop and Conference Proceedings, pages 652– 661. JMLR.org.

Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. 2024. The impact of reasoning step length on large language models. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 1830–1842. Association for Computational Linguistics.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe.

2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Bingshuai Liu, Ante Wang, Zijun Min, Liang Yao, Haibo Zhang, Yang Liu, Anxiang Zeng, and Jinsong Su. 2025a. Spec-rl: Accelerating on-policy reinforcement learning with speculative decoding. Preprint, arXiv:2509.23232. Rollouts generated using vLLM (rollout N=8).

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 2025b. Understanding r1-zero-like training: A critical perspective. CoRR, abs/2503.20783.

Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc G. Bellemare. 2016. Safe and efficient offpolicy reinforcement learning. In Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pages 1046–1054.

Kevin P. Murphy. 2012. Machine learning - a probabilistic perspective. Adaptive computation and machine learning series. MIT Press.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. CoRR, abs/1707.06347.

Robert J Serfling. 1978. Some elementary results on poisson approximation in a sequence of bernoulli trials. Siam review, 20(3):567–579.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300.

Yuanzhe Shen, Zisu Huang, Zhengkang Guo, Yide Liu, Guanxu Chen, Ruicheng Yin, Xiaoqing Zheng, and Xuanjing Huang. 2025. Intentionreasoner: Facilitating adaptive LLM safeguards through intent reasoning and selective query refinement. CoRR, abs/2508.20151.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256.

Wei Sun, Wen Yang, Pu Jian, Qianlong Du, Fuwei Cui, Shuo Ren, and Jiajun Zhang. 2025. KTAE: A modelfree algorithm to key-tokens advantage estimation in mathematical reasoning. CoRR, abs/2505.16826.

Zelin Tan, Hejia Geng, Mulei Zhang, Xiaohang Yu, Guancheng Wan, Yifan Zhou, Qiang He, Xiangyuan Xue, Heng Zhou, Yutao Fan, Zhongzhi Li, Zaibin Zhang, Guibin Zhang, Chen Zhang, Zhenfei Yin, and

Lei Bai. 2025. Scaling behaviors of LLM reinforcement learning post-training: An empirical study in mathematical reasoning. CoRR, abs/2509.25300.

Zichang Tan, Ajian Liu, Jun Wan, Hao Li, Zhen Lei, Guodong Guo, and Stan Z. Li. 2022. Cross-batch hard example mining with pseudo large batch for ID vs. spot face recognition. IEEE Trans. Image Process., 31:3224–3235.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Michael Tschannen, Josip Djolonga, Paul K. Rubenstein, Sylvain Gelly, and Mario Lucic. 2020. On mutual information maximization for representation learning. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Jinpeng Wang, Jieming Zhu, and Xiuqiang He. 2021. Cross-batch negative sampling for training two-tower recommenders. In SIGIR ’21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, pages 1632–1636. ACM.

Xun Wang, Haozhi Zhang, Weilin Huang, and Matthew R. Scott. 2020. Cross-batch memory for embedding learning. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 6387–6396. Computer Vision Foundation / IEEE.

Xinran Wu, Kun Yue, Huashuai Liu, and Liang Duan. 2025. Learning conditional probability distributions for robust probabilistic inference in bayesian network. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, CIKM 2025, Seoul, Republic of Korea, November 10-14, 2025, pages 3438–3447. ACM.

Xuan Xie, Xuan Wang, and Wenjie Wang. 2025. Dagrpo: Rectifying gradient conflict in reasoning via distinctiveness-aware group relative policy optimization. arXiv preprint arXiv:2512.06337.

Wei Xiong, Chenlu Ye, Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, Jiang Bian, Nan Jiang, and Tong Zhang. 2025. Reinforce-ada: An adaptive sampling framework for reinforce-style LLM training. CoRR, abs/2510.04996.

Huei-Fang Yang, Kevin Lin, and Chu-Song Chen. 2016. Cross-batch reference learning for deep classification and retrieval. In Proceedings of the 2016 ACM Conference on Multimedia Conference, MM 2016, Amsterdam, The Netherlands, October 15-19, 2016, pages 1237–1246. ACM.

Shihui Yang, Chengfeng Dou, Peidong Guo, Kai Lu, Qiang Ju, Fei Deng, and Rihui Xin. 2025. DCPO: dynamic clipping policy optimization. CoRR, abs/2509.02333.

Zhuliang Yao, Yue Cao, Shuxin Zheng, Gao Huang, and Stephen Lin. 2021. Cross-iteration batch normalization. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 12331–12340. Computer Vision Foundation / IEEE.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, and 16 others. 2025. DAPO: an open-source LLM reinforcement learning system at scale. CoRR, abs/2503.14476.

Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, and Qianchun Lu. 2025. Learning like humans: Advancing LLM reasoning capabilities via adaptive difficulty curriculum learning and expert-guided selfreformulation. CoRR, abs/2505.08364.

Qinghua Zhang. 2026. Stability analysis of the kalman filter under practical conditions. Autom., 183:112670.

Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, Fang Wan, and Furu Wei. 2025. Geometric-mean policy optimization. CoRR, abs/2507.20673.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. 2025. Group sequence policy optimization. CoRR, abs/2507.18071.

Jiaru Zou, Yikun Ban, Zihao Li, Yunzhe Qi, Ruizhong Qiu, Ling Yang, and Jingrui He. 2025. Transformer copilot: Learning from the mistake log in LLM finetuning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

#### appendix A More Related Work

Group-based RLVR. Artificial intelligence has achieved significant advances in recent years(Zou

- et al., 2025; He et al., 2025). Recent studies have proposed numerous improvements to group-based reinforcement learning algorithms. DaGRPO (Xie et al., 2025) tackles GRPO’s instability and poor sample efficiency (caused by low distinctiveness in on-policy rollouts) by introducing sequence-level gradient rectification to filter low-distinctiveness pairs and off-policy anchor augmentation to restore learning signals on hard prompts. To address the advantage reversion and advantage mirror issues of fixed advantage formulations in GRPO that fail to adapt to samples with varying trajectory certainty, MAPO (Huang et al., 2025) introduces Advantage Percent Deviation (APD) for high-certainty trajectories and Trajectory Certainty Reweight (TCR) to dynamically reweight the advantage function, enabling adaptive and reliable trajectory evaluation. LSPO (Chen et al., 2025) adopts length-aware dynamic sampling to retain shortest/longest responses, addressing the ineffectiveness of RLVR training for LLM reasoning. GMPO (Zhao et al.,

2025) uses the geometric mean of token-level rewards (replacing GRPO’s arithmetic mean) to resolve unstable policy updates from outlier importance sampling ratios. And DCPO (Yang et al., 2025) uses dynamic adaptive clipping and smooth advantage standardization to solve zero gradients, limited token exploration, and low response utilization in RLVR. FAPO (Ding et al., 2025) uses a generative reward model (GenRM) to detect flawed-positive rollouts and a parameter-free reward penalty, addressing unreliable reasoning patterns and performance limitations caused by such rollouts in RLVR. SPO (Guo et al., 2025) uses segment-level advantage estimation (with Monte Carlo sampling and flexible segmentation) to solve inaccurate advantage estimation of token-level methods and imprecise credit assignment of trajectory-level methods in LLM reinforcement learning. KTAE (Sun et al., 2025) uses statistical analysis to quantify tokens’ association with correct rollouts and combines it with rollout-level advantages, solving the coarse granularity issue of GRPO that ignores token-specific contributions.

Leverage of Cross-batch Signals. Cross-batch signals have found widespread application across

numerous domains. SamS (Huang et al.) leverages a cross-batch scheduler to identify high-quality samples for direct preference optimization. XBM (Wang et al., 2020) improves embedding learning by leveraging memory from previous batches to enhance the consistency and quality of embeddings. CBNS (Wang et al., 2021) introduces a method to improve negative sampling in embedding learning by utilizing negative samples from different batches, enhancing the model’s ability to learn more robust and generalized representations. CIBN (Yao et al., 2021) extends traditional batch normalization across iterations, rather than within a single batch, to improve model convergence and generalization. CBRL (Yang et al., 2016) utilizes reference samples from different batches during training to improve the learning of deep classification and retrieval models. CBHEM-PLB (Tan et al., 2022) combines cross-batch hard example mining with a pseudo large batch strategy to improve face recognition models.

Biased Estimation. Considerable research effort has been directed towards addressing the critical challenge of biased estimation. The Bias–Variance Tradeoff theory (Hastie et al., 2009; Murphy, 2012) suggests that as a model’s complexity increases, its bias decreases but its variance increases, and vice versa. It emphasizes that there is a balance between bias and variance that affects the overall error in model predictions, and finding the optimal model complexity is crucial to minimize both bias and variance. Retrace (Munos et al., 2016) addresses the challenge of bias estimation in off-policy reinforcement learning. It proposes a retracing technique to mitigate the bias caused by off-policy data, which can lead to inaccurate value estimates. V-trace (Espeholt et al., 2018; Boucheron et al., 2013) introduces a method for improving off-policy reinforcement learning by applying importanceweighted corrections to the value function updates in actor-critic algorithms, mitigating bias in offpolicy data. DR-OVR (Jiang and Li, 2016; Tschannen et al., 2020) combines importance sampling and regression to correct for bias in off-policy value estimation, making it more stable and accurate.

#### B Detailed Instantiations for GRPO and Related Algorithms

In this section, we present detailed instantiations of three group-relative reinforcement learning al-

gorithms: GRPO, GSPO, and DAPO. And t and τ denote training step and token index in this part.

GRPO streamlines PPO by discarding the value network without compromising stability. Instead of fitting a baseline, it derives the advantage using group-relative normalization. This groupnormalized advantage is then assigned uniformly to all tokens in the response, formulating the clipped surrogate loss:

|ot,i|

Ä

G

1 G

1 |ot,i|

rt,i,τ(θ)Aˆt,i,τ,

min

JGRPO(θ) =

τ=1

i=1

ä

clip(rt,i,τ(θ),1 − ϵ,1 + ϵ)Aˆt,i,τ

, (22) where ϵ is the clipping hyperparameter and rt,i,τ is is the importance sampling ratio comparing the new and old policy:

πθ(yt,i,τ | xt,yt,i,<τ) πθold(yt,i,τ | xt,yt,i,<τ)

. (23)

rt,i,τ(θ) =

And GRPO defines the group advantage by subtracting the average reward of the group and normalizing by its standard deviation:

{R(xt,ot,j)}Gj=1 std ä Ä

Ä

R(xt,ot,τ) − mean

Aˆt,i,τ =

ä ,

{R(xt,ot,j)}Gj=1

(24) where R(x,o) denotes the reward function.

The objective function of GRPO applied with HA-DW can be denoted as:

JGRPO+HA-DW(θ)

|ot,i|

Ä

G

1 |ot,i|

1 G

rt,i,τ(θ)Aˆt,i,τ · Φt,i,

min

=

τ=1

i=1

ä

clip(rt,i,τ(θ),1 − ϵ,1 + ϵ)Aˆt,i,τ · Φt,i

,

(25) where Φt,i is the history-aware reweighting factor defined earlier.

GSPO optimizes policy learning by defining importance ratios at the sequence level, eliminating the need for a critic model. Rather than relying on a separate value network, it computes advantages through normalized relative rewards of group responses. This sequence-level advantage is directly used for policy updates without token-level pro-

cessing, yielding the following objective function:

Å

G

1 G

rt,i(θ)Aˆt,i,

JGSPO(θ) =

min

i=1

Å

ã

ã

Aˆt,i

clip

rt,i(θ), 1 − ϵ, 1 + ϵ

,

(26) where the sequence-level importance sampling ratio ri(θ) can be denoted as:

πθ(yt,i|xt) πθold(yt,i|xt)

rt,i(θ) =

(27)

|yt,i| τ=1 πθ(yt,i,τ|xt,yt,i,<τ)

=

,

|yt,i| t=1 πθold(yt,i,τ|xt,yt,i,<τ)

where the advantage for GSPO can be denoted as:

{R(xt,ot,j)}Gj=1 std ä Ä

Ä

R(xt,ot,i) − mean

Aˆt,i =

ä

{R(xt,ot,j)}Gj=1

(28) And the objective function of GSPO+HA-DW is:

JGSPO+HA-DW(θ)

Å

G

1 G

rt,i(θ)Aˆt,i · Φt,i,

=

min

(29)

i=1

Å

ã

ã

Aˆt,i · Φt,i

clip

rt,i(θ), 1 − ϵ, 1 + ϵ

.

DAPO’s key feature is operating at the token level instead of treating full responses as single units, ensuring each token in sampled output oi contributes proportionally to gradient updates. This fine-grained optimization boosts training stability and delivers more informative feedback for LLMs. The objective function is defined as:

JDAPO(θ)

Å

|ot,i|

G

1

rt,i,τ(θ)Aˆt,i,τ,

min

=

G i=1 |ot,i|

τ=1

i=1

ã

clip rt,i,τ(θ),1 − ϵ,1 + ϵ′ A ˆt,i,τ

.

(30)

DAPO introduces two key mechanisms: decoupled clipping and dynamic sampling, to address the limitations of traditional group-based methods. Decoupled clipping refines the trust region for more stable updates, while dynamic sampling mitigates estimation bias by adaptively reweighting samples based on their distribution.

Applying HA-DW on Equation (30), and we have:

JDAPO+HA-DW(θ)

Å

|ot,i|

G

1

rt,i,τ(θ)Aˆt,i,τ · Φt,i,

=

min

G i=1 |ot,i|

τ=1

i=1

ã

clip rt,i,τ(θ),1 − ϵ,1 + ϵ′ A ˆt,i,τ · Φt,i

.

(31)

#### C Setup Details

Models & Datasets. We conduct our experiments on Qwen3-4B-Base, Qwen3-8B-Base (Team, 2025) and LLaMA-3.2-3B-Instruct to assess the mathematical reasoning performance of different algorithms across models of varying scales and family. Our training dataset is sourced from MATH dataset (Hendrycks et al., 2021; Lightman et al., 2024) which contains 7.5k questions for training. Our evaluation suite includes: MATH500 (Hendrycks et al., 2021), AMC23, AIME25, Minerva, and OlympiadBench (He et al., 2024). To mitigate high variance on small benchmark sets and obtain reliable results, we report avg@16 on AIME25 and AMC23.

Baseline. We apply our proposed method on top of several representative group-relative reinforcement learning algorithms: GRPO, GSPO, and DAPO. We compare the performance of group-relative algorithms applying HA-DW to original ones, verifying the effectiveness and scalability of our method.

Implementation Details. We conduct RL training within the VeRL framework (Sheng et al., 2024) on a single node with 8 × NVIDIA A100 GPUs. All experiments use a maximum prompt batch size of 1,024 and a maximum response length of 4,096. More hyperparameter details are provides in appendix C.

Training Hyperparameters. The detailed hyperparameters used during our training process on 6 different methods of 3 models (Qwen34B-Base, Qwen3-8B-Base and LLaMA-3.2-3BInstruct) used in our experiments are demonstrated in Table 8.

#### D Theoretical Proof

###### D.1 Proof of Theorem 1

In group-relative RL algorithms, the truncation mechanism will discard prompts with all-correct or all-incorrect responses. Under the binary reward setting, the retention condition for the total reward within the group R is given by:

###### S := {1 ≤ R ≤ G − 1}.

Under the retention condition S, E[ˆpt | S] denotes the conditional expectation of the empirical estimation pˆt = R/G. And it can be derived what the relationship is between it and the expected reward pt:

ï

ò

R G | S

E[ˆpt | S] = E

E R · 1{S} P(S)

1 G ·

=

E[R] − E R · 1{R=G} P(S)

1 G ·

(32)

=

Gpt − GP(R = G) P(S)

1 G ·

=

pt − pGt 1 − (1 − pt)G − pGt

=

,

where the indicator function 1{S} takes the value 1 if the event S occurs and 0 otherwise. Through

the conditional expectation of pˆt, we can obtain that its expected value is less than pt when pt < 12 and the baseline tends to be underestimated. Conversely, when pt > 12, the expected value exceeds pt, leading to an overestimation.

Based on Equation (2) and Equation (4), inaccurate baseline estimation will induce biased advantage estimation. From the foregoing analysis, we can derive that:

E Aˆt,i | S < At,i, if pt < 0.5; E Aˆt,i | S > At,i, if pt > 0.5; E

î

ó

Aˆt,i | S

= At,i, if and only if pt = 0.5.

(33) Lemma 2. Under the condition of Theorem 1, the bias induced by the group-relative advantage is formulated as:

î

= ó

Aˆt,i | S

At,i − E

(34)

pt(1 − pt)G + pGt +1 − pGt 1 − (1 − pt)G − pGt

.

Proof.

E[ˆpt | S] − pt

(35)

pt(1 − pt)G + pGt +1 − pGt 1 − (1 − pt)G − pGt

=

.

Replacing the baseline with the advantage completes the proof.

| |
|---|

- D.2 Proof of Theorem 2 and Corollary 1 For hard prompts, in Theorem 2, we have:

P(ˆpt − pt > ϵ | S)

G−1 k=⌊G(pt+ϵ)⌋+1

(36)

G k pkt (1 − pt)G−k

=

.

1 − (1 − pt)G − pGt

The above equation is given by the following argument: The conditioning event S restricts the sample space by excluding the outcome R ∈ {0,G} (hence under S we only keep R ∈ {1,...,G−1}). Let:

m(pt) := ⌊G(pt + ϵ)⌋ + 1. (37)

Therefore, within the event S, the deviation event A becomes

A ∩ S = {R ≥ m(pt)} ∩ {1 ≤ R ≤ G − 1}

= {m(pt) ≤ R ≤ G − 1}.

(38)

By definition of conditional probability, the numerator is the (unconditional) probability mass of all outcomes that satisfy the deviation requirement pˆ− p > ϵ and simultaneously, and satisfy the restriction imposed by S. Because R is binomial, for any integer k we have:

Ç

å

G k

pkt (1 − pt)G−k. (39)

P(R = k) =

Summing over all admissible counts k ∈ {m(pt),m(pt) + 1,...,G − 1} yields:

G−1

P(R = k) (40)

P(A ∩ S) =

k=m(pt)

Ç

å

G−1

G k

pkt (1 − pt)G−k. (41)

=

k=m(pt)

Thus, based on the formula of conditional probability (Wu et al., 2025; Gianantonio and Edalat, 2025), we can derive the conclusion of Theorem 2.

According to Theorem 2, we can formulate:

f(G,pt) := P(ˆpt − pt > ϵ | S). (42)

Assume that pt follows a uniform distribution. And we define:

pt2

1 pt2 − pt1

P(G,pt1,pt2) :=

f(G,pt)dpt

pt1

(43)

where pt1 and pt2 is the expected reward. And P(G) reflects the probability that, when G is fixed,

the baseline pˆt is overestimated of group-relative RL algorithms over a certain expected-reward interval. For hard prompts with pt ∈ (0,0.25) under different group size G, when G ∈ [2,8], we have:

###### G P(G,0,0.25)

2 0.999997499987 4 0.999995948256 6 0.827761785622 8 0.781129955681

Table 4: P(G,0,0.25) as a function of G ∈ [2,8].

Similarly, we can calculate hard prompts with pt ∈ (0,0.5) under different group size G.

###### G P(G,0,0.5)

2 0.999994999975 4 0.776965795853 6 0.689721502158 8 0.640944744224

Table 5: P(G,0,0.5) as a function of G ∈ [2,8].

We can conclude from Table 4, when 2 ≤ G ≤ 8, P(G,0,0.25) > 0.78. This reveals that for hard prompts whose pt ∈ (0,0.25) when G is limited, its baseline pˆt of group-relative RL algorithms is substantially likely to be overestimated. Similarly, due to the evident symmetry of the group-relative methods, for easy prompt with pt ∈ (0.75,1), the baseline pˆt is underestimated with the same probability distribution.

Based on the aforementioned conclusions, for group-based algorithms, when G ∈ [2,8], the probability of biased advantage estimation can be denoted as:

P Aˆt,i < At,i | S, pt < 0.25 > 0.78, P

Ä

ä

Aˆt,i > At,i | S, pt > 0.75

> 0.78.

(44)

Similarlly, Table 5 can give: P Aˆt,i < At,i | S, pt < 0.5 > 0.63, P

Ä

ä

(45)

Aˆt,i > At,i | S, pt > 0.5

> 0.63.

For pt > 0.875 and pt < 0.125, the results are the adaptation of Corollary 3.

###### D.3 Proof of Corollary 2 and Corollary 3

Let G be a large integer, for hard prompts, according to Theorem 2, we have:

P(ˆpt − pt > ϵ | S)

G−1 k=⌊G(pt+ϵ)⌋+1

G k pkt (1 − pt)G−k

=

.

1 − (1 − pt)G − pGt

And we define:

(46)

f(pt) := P(ˆpt − pt > ϵ | S). (47)

We analyze the integral in the limit of large G using the Poisson approximation (Serfling, 1978). Let us perform the change of variable xt = Gpt. The limits of integration change from [1/G,2/G] to [1,2], and dpt = dxt/G. We define the integral of interest:

G2

G G2 − G1

dxt G

P(G1,G2) =

f(xt/G)

G1

(48)

2

=

f(xt/G)dxt.

1

First, we determine the summation lower bound m(pt). For pt ∈ [1/G,2/G), we have Gpt ∈ [1,2). Consequently, ⌊Gpt⌋ = 1, which implies:

m(pt) = ⌊Gpt⌋ + 1 = 2. (49)

Next, we approximate the binomial terms. In the limit G → ∞ with Gpt = x fixed, the binomial distribution converges to the Poisson distribution with parameter xt. The denominator Z(pt) approximates to:

Z(pt) = 1 − (xt/G)G − (1 − xt/G)G −−−−→G→∞ 1 − e−xt.

(50)

The numerator is the probability that a Poisson random variable K ∼ Pois(xt) takes a value k ≥ 2 (ignoring the upper limit G − 1 as the Poisson tail vanishes exponentially):

∞

xkt e−xt k!

f(pt) = P(K ≥ 2) =

k=2

(51)

= 1 − P(K = 0) − P(K = 1)

= 1 − e−xt − xe−xt = 1 − e−xt(1 + xt).

Substituting these approximations into f(xt/G), we obtain the limiting integrand h(xt):

1 − e−xt(1 + xt) 1 − e−xt

h(xt) =

(52)

xte−xt 1 − e−xt = 1 −

xt ext − 1

= 1 −

.

Assume that pt follows a uniform distribution. Calculating Equation (52) numerically, for sufficiently large G, we can show that P(0,2) =

2

2 0 f(xt)dxt = GP(0,1)+

- 1 h(xt) dxt

- 2 ≈ 0.7818.

G 2

Next, we use numerical computation to show how large G should be, whose result can be found in the following table:

G G2 0 2/G f(pt)dpt

- 2 0.499997499987
- 3 0.749995833315
- 4 0.776965795853
- 5 0.780787089465
- 6 0.781154327380

Table 6: G2 0 2/G f(pt)dpt as a function of G ∈ [2,6].

Thus, G ≥ 6 is sufficiently large to have:

ã

Å

2 G

Aˆt,i < At,i | S, pt <

> 0.78. (53)

P

Proof of Corollary 3. On S, we have R ≥ 1, hence pˆ = R/G ≥ 1/G. Since p < 1/G, it follows that pˆ ≥ 1/G > p. On S, we have R ≤ G − 1, hence pˆ = R/G ≤ (G − 1)/G. Since p > (G−1)/G, it follows that pˆ ≤ (G−1)/G < p. This leads to the Corollary.

D.4 Proof of Lemma 1 and Theorem 3 D.4.1 Proof of Lemma 1

Before proving Lemma 1, we begin with the following auxiliary result. Lemma 3. Define the non-degenerate event S := {1 ≤ S ≤ G − 1}, and ϵ ∈ (0,|pt − pˆt|). If

Ç

(pt − ϵ) · 1 − (1 − pt)G − pGt pt(1 − pGt −1)

c ∈

,

å

(54)

(pt + ϵ) · 1 − (1 − pt)G − pGt pt(1 − pGt −1)

,

we have:

###### E[˜pt | S] ∈ (pt − ϵ, pt + ϵ). (55)

Proof. We define the adjusted factor c to compensate for the bias in the advantage estimation which applied on empirical group baseline pˆt. The globally scaled estimator can be approximated as:

p˜t(R) := cpˆt = c

R G

. (56)

We can derive the conditional expectation of p˜t on non-degenerate event S = {1 ≤ R ≤ G − 1}:

ò

ï

R G S

E[˜pt | S] = E

c

c G

E[R | S]

(57)

=

E[R · 1{S}] P(S)

c G

=

.

And we have:

G−1

k P(R = k)

E[R · 1{S}] =

(58)

k=1

= E[R] − GP(R = G).

Because the only term excluded from G k=0 kP(R = k) = E[R] is the k = G term (the k = 0 term is zero anyway). Using E[R] = Gpt and P(R = G) = pGt , we can obtain:

E[R · 1{S}] = Gpt − GpGt

= Gpt(1 − pGt −1).

Therefore:

(59)

Gpt(1 − pGt −1) 1 − (1 − pt)G − pGt

c G ·

E[˜pt | S] =

(60)

1 − pGt −1 1 − (1 − pt)G − pGt

= cpt

,

which proves the stated conditional expectation formula.

To mitigate the biased estimation, let:

E[˜pt | S] = pt. (61) And we can solve:

1 − pGt −1 1 − (1 − pt)G − pGt

= pt. (62)

cpt

The analytical solution for this equation is:

1 − (1 − pt)G − pGt 1 − pGt −1

. (63)

c =

When the adjustment coefficient c falls within a specific range of values, we will have |p˜t − pt| < |pˆt − pt|. We first let:

ϵ = |pˆt − pt|. (64) For E[˜pt | S] = pt + ϵ, solve:

1 − pGt −1 1 − (1 − pt)G − pGt

= pt + ϵ. (65)

c+ pt

And we can derive:

(pt + ϵ) · 1 − (1 − pt)G − pGt pt(1 − pGt −1)

c+ =

(66)

Å

ã

ϵ pt

=

1 +

c.

For E[˜pt | S] = pt − ϵ, solve:

1 − pGt −1 1 − (1 − pt)G − pGt

= pt − ϵ. (67)

c− pt

Thus, we have:

(pt − ϵ) · 1 − (1 − pt)G − pGt pt(1 − pGt −1)

c− =

(68)

Å

ã

ϵ pt

1 −

=

c.

We can conclude that when:

Ç

(pt − ϵ) · 1 − (1 − pt)G − pGt pt(1 − pGt −1)

c ∈

,

å

(69)

(pt + ϵ) · 1 − (1 − pt)G − pGt pt(1 − pGt −1)

,

we have

###### E[˜pt | S] ∈ (pt − ϵ, pt + ϵ). (70)

| |
|---|

Lemma 4 (pt-free concentration under S). Define the non-degenerate event S := {1 ≤ S ≤ G − 1}. Assume pt ∈ [∆, 1 − ∆] for some ∆ ∈ (0,1/2]. Then for any ζ > 0, we have:

P(|pˆt − pt| < ζ |S) ≥

1 − 2exp(−2Gζ2) − (1 − ∆)G − ∆G 1 − (1 − ∆)G − ∆G

.

(71)

Proof. Let A := {|pˆt −pt| < ζ}. By the definition of conditional probability:

P(A ∩ S) P(S)

. (72)

P(A | S) =

We lower bound the numerator. Since A ∩ S ⊇ A \ S−, we have:

###### P(A ∩ S) ≥ P(A) − P(S−). (73)

Next, note that S− = {S = 0} ∪ {S = G} and these two events are disjoint. Therefore:

P(S−) = P(S = 0) + P(S = G) = (1 − pt)G + pGt .

Moreover, we can derive that:

(74)

###### P(S) = 1 − P(S−) = 1 − (1 − pt)G − pGt . (75)

We now lower bound P(A) using Hoeffding’s inequality. Since each rt,i ∈ [0,1] almost surely and {rt,i}Gi=1 are independent with E[rt,i] = pt, Hoeffding’s inequality yields:

P(|pˆt − pt| ≥ ζ) ≤ 2exp(−2Gζ2), (76) equivalently:

P(A) = P(|pˆt − pt| < ζ) ≥ 1 − 2exp(−2Gζ2).

(77)

It remains to remove the dependence on pt in P(S). Define f(p) := pG + (1 − p)G. For G ≥ 1, f is symmetric around 1/2 and attains its maximum over [∆,1 − ∆] at the endpoints. Hence:

###### (1−pt)G+pGt = f(pt) ≤ f(∆) = (1−∆)G+∆G,

(78) which implies:

###### P(S) = 1 − f(pt) ≥ 1 − (1 − ∆)G − ∆G. (79)

Combining Equation (72) and (73) with Equation (77), (78), and (79), we can obtain that:

P(A | S) (80) ≥

P(A) − P(S−) P(S) ≥

1 − 2exp(−2Gζ2) − (1 − ∆)G + ∆G 1 − (1 − ∆)G − ∆G

, (81) which completes the proof.

| |
|---|

- Lemma 5 (Conditional pt-free concentration under S). Assume pt ∈ [∆, 1−∆] for some ∆ ∈ (0,1/2]. Then for any δ ∈ (0,1), with probability at least 1 − δ conditional on S, we have:

|pˆt−pt| <

- 1

- 2G

log

Ç

2 δ 1 − (1 − ∆)G − ∆G

å

. (82)

Proof. Now choose γ such that the right-hand side of Equation (71) is at most δ, i.e.:

2exp(−2Gγ2)

1 − (1 − ∆)G − ∆G ≤ δ. Solving for γ gives:

γ ≥

- 1

- 2G

log

Ç

2 δ 1 − (1 − ∆)G − ∆G

å

.

Therefore, for

γ⋆ :=

- 1

- 2G

log

Ç

2 δ 1 − (1 − ∆)G − ∆G

å

,

we have P(|pˆt − pt| ≥ γ⋆ | S) ≤ δ, equivalently, P(|pˆt − pt| < γ⋆ | S) ≥ 1 − δ, which proves Equation (82).

| |
|---|

Finally, combining Lemma 3, 4, and 5 gives Lemma 1. We restated it here for completeness:

- Lemma 6 (A pt-free feasible range of c expressed via pˆt). Assume the conditions of Lemma 4 and define:

Ç

å

- 1

- 2G

2 δ 1 − (1 − ∆)G − ∆G

log

.

ϵδ :=

(83) Let:

It := p ˆt − ϵδ, pˆt + ϵδ ∩ [∆,1 − ∆], A(p) := 1 − (1 − p)G − pG.

(84)

Fix any ϵ > 0, we define:

(p − ϵ)A(p) p(1 − pG−1)

, (85) and:

clow := sup p∈It

(p + ϵ)A(p) p(1 − pG−1)

. (86)

chigh := inf p∈It

Then, on the event {|pˆt − pt| < ϵδ} (which holds with probability at least 1 − δ conditional on S), any choice

###### c ∈ (clow, chigh) (87)

implies that the condition (69) holds for the true pt, and hence:

###### E[˜pt | S] ∈ (pt − ϵ, pt + ϵ).

###### D.4.2 Proof of Theorem 3

When applying adjustment on the advantage Aˆt,i, we do not consider the standard deviation here, and assume that:

Φt,iAˆt,i = rt,i − p˜t = rt,i − cpˆt. (88) It is equivalent to:

Φt,irt,i − Φt,ipˆt = rt,i − cpˆt. (89) And for correct responses with rt,i = 1: Φt,i =

1 − cpˆt 1 − pˆt

. (90)

While for incorrect responses with rt,i = 0:

Φt,i = c. (91) According to Equation (16):

Φt,i = λscale · exp(Dt,i · Mt), (92)

the adjustment of At,i can be categorized into four types. For responses in defined hard prompts with rt,i = 1, the adjusted advantage can be denoted as:

Aˆ1t,i = λscale · exp(Mt) · Aˆt,i. (93)

For hard prompts, we have c ∈ (0,1) and pˆ ∈ (0,1). Based on Equation (69) and Lemma 6, to mitigate biased estimation, λscale satisfies:

Ñ

exp(Mt) é

hard high)ˆpt

1 + (1−c

hard low )ˆpt

1 + (1−c

1−pˆt

1−pˆt

λ1scale ∈

,

.

exp(Mt)

(94) And for incorrect responses in hard prompts, we have:

λscale

exp(Mt) · Aˆt,i. (95) And we can set:

Aˆ2t,i =

Ä

ä

λ2scale ∈

chardlow · exp(Mt), chardhigh · exp(Mt)

. (96)

For easy prompts, we have c > 1 and pˆ ∈ (0,1), thus for correct answers:

å

ÇÇ

(1 − ceasyhigh)ˆpt 1 − pˆt

λ3scale ∈

· exp(Mt), Å

1 +

ã

ã

(1 − ceasylow )ˆpt 1 − pˆt

· exp(Mt)

1 +

.

(97)

And for incorrect responses:

Ç

å

ceasyhigh exp(Mt)

ceasylow exp(Mt)

λ4scale ∈

. (98)

,

In training process with HA-DW, to rectify the biased advantage estimation, there exists a specfic λscale supposing to satisfy:

λscale ∈ λ1scale ∪ λ2scale ∪ λ3scale ∪ λ4scale (99) which denotes:

Ñ

exp(Mt) é

hard low )ˆpt

hard low )ˆpt

1 + (1−c

1 + (1−c

1−pˆt

1−pˆt

λscale ∈

∪ ÇÇ

,

exp(Mt)

å

(1 − ceasyhigh)ˆpt 1 − pˆt

· exp(Mt), Å

1 +

ã

ã

(1 − ceasylow )ˆpt 1 − pˆt

∪ Ä

· exp(Mt)

1 +

ä

chardlow · exp(Mt), chardlow · exp(Mt)

∪ Ç

å

ceasyhigh exp(Mt)

ceasylow exp(Mt)

,

.

(100)

Overall, since the difficulty does not affect the expressions, we can further derive Equation (100) as follows:

Ñ

exp(Dt,i · Mt)é

1 + (1−1c−highpˆ )ˆpt

1 + (1−1c−lowpˆ )ˆpt

t

t

λscale ∈

,

exp(Dt,i · Mt)

Å

ã

chigh exp(Dt,i · Mt)

clow exp(Dt,i · Mt)

∪

,

.

(101)

When Equation (101) holds, our method HADW is efficient in compensating biased advantage estimation.

###### D.5 Non-binary Reward Analysis

In this section, we extend our analysis to continuous bounded reward distributions (e.g., Beta and truncated Gaussian scores), which better reflect the behavior of soft verifiers and learned reward models commonly used in practice. Our extended analysis demonstrates that, under these more general reward assumptions, the group-relative advantage estimator remains systematically biased in an analogous manner: it tends to underestimate the true advantage for hard prompts and overestimate the true advantage for easy prompts. Moreover, as prompt difficulty becomes more extreme (i.e., as ∆ increases), the magnitude of this bias becomes increasingly pronounced. Next, we show the main results.

Theorem 4. At training step t and let G ≥ 2, with CDF F and PDF f. Given a prompt xt ∼ D and draw G ≥ 2 i.i.d. rewards:

###### rt,1,...,rt,G i.i.d.∼ D(pt). (102)

And we extend the binary reward setting to nonbinary rewards:

rt,i ∈ {0,1} → rt,i ∈ [0,1]. (103) The group-relative advantage can be denoted as:

1 G

Aˆt,i := rt,i − pˆt, pˆt =

G

rt,i, (104)

i=1

while the expected advantage is defined as:

At,i := rt,i − pt. (105) Fix a constant σ ∈ [0,1] and define the update

event:

Sσ := ∃i ̸= j : |rt,i − rt,j| > σ

⇒ Sσc = max

rt,i − min

rt,i ≤ σ .

i

i

(106)

For u ∈ [0,1], define u+ := min{1,u + σ}, we have:

q(u) := F(u+) − F(u), (107) and:

m(u) := E[rt,1 | u ≤ rt,1 ≤ u+]

u+ u xf(x)dx

(when q(u) > 0).

=

F(u+) − F(u)

(108) Then the probability of a non-update is:

1

f(u)q(u)G−1 du, (109) and:

P(Sσc) = G

0

P(Sσ) = 1 − P(Sσc). (110) Moreover, we have:

with:

pt − E[ˆpt · 1{Sc

σ}] P(Sσ)

E[ˆpt | Sσ] =

(111)

E[ˆpt · 1{Sc

σ}]

1

u + (G − 1)m(u) f(u)q(u)G−1 du.

=

0

(112)

Finally, the conditional bias transferred to advantages satisfies, for all i, we have:

E[Aˆt,i − At,i | Sσ] = pt − E[ˆpt | Sσ]. (113)

Proof. The complement event can be denoted as:

###### Sσc = {max−min ≤ σ} (114)

For absolutely continuous i.i.d. samples, the minimum has density:

###### gmin(u) = Gf(u) 1 − F(u) G−1. (115)

Condition on min = u. The remaining G − 1 samples are i.i.d. with the original law conditioned on [u,1]; imposing max ≤ u+ is equivalent to requiring each of those samples lies in [u,u+]. Thus:

F(u+) − F(u) 1 − F(u)

G−1

P(Sσc | min = u) =

,

(116) and multiplying by gmin(u) gives:

1

f(u)q(u)G−1du. (117)

P(Sσc) = G

0

On Sσc and min = u, one sample equals the minimum and the other G−1 samples lie in [u,u+]. By symmetry, the conditional mean of each of the G − 1 non-minimum samples is m(u), hence:

G

rt,i Sσc,min = u = u + (G − 1)m(u).

E

i=1

(118)

So we can derive: E[ˆpt · 1{Sc

σ}]

1

u + (G − 1)m(u) G

dP(min ∈ du,Sσc)

=

0

1

(u + (G − 1)m(u))f(u)q(u)G−1du.

=

0

(119) Corollary 4. For Beta(α,β) reward distribution, the Beta density is:

xα−1(1 − x)β−1 B(α,β)

, (120)

f(x) =

and the CDF is:

###### F(x) = Ix(α,β) for x ∈ [0,1], (121)

where B(·,·) is the Beta function and Ix(α,β) is the regularized incomplete beta function. In particular:

α α + β

. (122) Moreover, we have:

pt = E[rt,1] =

q(u) = F(u+) − F(u)

= Iu+(α,β) − Iu(α,β),

(123)

and the conditional mean over [u,u+] admits the closed form:

u+ u xf(x)dx

m(u) =

u+ u f(x)dx

(124)

Bu+(α + 1,β) − Bu(α + 1,β) Bu+(α,β) − Bu(α,β)

=

,

where Bx(·,·) denotes the (unregularized) incomplete beta function.

Consequently, substituting F,f,q,m into conclusions obtained earlier yields explicit onedimensional integral expressions (in standard special functions) for P(Sσc) and E[ˆpt | Sσ].

Corollary 5. Let the reward Zt,1,...,Zt,G be i.i.d. N(µ,ξ2) with ξ > 0, and define rt,i to be properly truncated to [0,1], i.e. rt,i has the conditional law:

rt,i =d Zt,i (0 ≤ Zt,i ≤ 1), i = 1,...,G.

(125) Let u+ := min{1,u + c} and define, for u ∈

[0,1] with q(u) > 0, we have:

q(u) := P u ≤ rt,1 ≤ u+ , (126) and:

m(u) := E rt,1 | u ≤ rt,1 ≤ u+ . (127) Let Φ and φ be the standard normal CDF and

PDF, and set:

0 − µ ξ

1 − µ ξ

. (128)

a :=

, b :=

Then the truncated-normal density on [0,1] is:

Äx−µ

σ Φ(b) − Φ(ä a)

φ

ξ

1[0,1](x). (129)

f(x) =

Its CDF on [0,1] is:

Äx−µ

ä

− Φ(a) Φ(b) − Φ(a)

Φ

ξ

. (130)

F(x) =

The mean satisfies:

φ(a) − φ(b) Φ(b) − Φ(a)

. (131)

pt = E[rt,1] = µ + ξ

Moreover:

q(u) = F(u+) − F(u), (132)

and the conditional mean over [u,u+] has the standard truncated-normal form:

Φ ä

ä

Äu−µ

Äu+−µ

− φ

φ

ξ

ξ

ä

Äu−µ

ä. (133)

Äu+−µ

m(u) = µ + σ

− Φ

ξ

ξ

Consequently, substituting F,f,q,m to yield explicit one-dimensional integral expressions for P(Sσc) and E[ˆpt | Sσ] in terms of Φ and φ.

Remark. Theoretical and numerical evaluations under properly truncated Gaussian reward distributions indicate that the grouprelative advantage bias, At,i − E[Aˆt,i | Sσ] , tends to increase as pt deviates further from 1/2. Intuitively, near these extremes, a typical group of samples exhibits reduced dispersion. As a result, conditioning on the non-degenerate event Sσ preferentially selects groups with atypically large variability, which in turn shifts the conditional expectation of the advantage away from its true value.

Figure 5 illustrates two representative cases corresponding to group sizes G = 4 and G = 8, as predicted by Corollary 5. In both settings, the magnitude of the bias |At,i − E[Aˆt,i | Sσ]| increases as pt moves farther away from 0.5, corroborating our theoretical analysis.

pt σ = 0.10 c = 0.10

ˆ||−|A[AS]t,it,i

| |
|---|

pt

Figure 5: Illustration of advantage bias under truncated Gaussian rewards for different group sizes.

#### E Supplementary Experiments

###### E.1 Advantage Distribution

We conducted an assessment of select prompts from the widely used training dataset MATH and DAPO-Math-17k (Yu et al., 2025) on Qwen3-4BBase across different rollouts. At first, we evaluated

the model’s performance on the dataset at rollout=8. From these, we selected four groups of 50 prompts each: groups with single correct or incorrect response. We then evaluated the outcomes of these selected prompts at rollout=128 where enough rollouts can reflect intrinsic difficulty of these prompts.

For those groups with only 1 correct responses at rollout=8, the distribution of the number of correct responses within these groups is shown in Figure 6(a). For the MATH and DAPO-Math-17k datasets, 24 and 15 groups have fewer than 16 correct responses at rollout=128 respectively which suggests that the advantage of correct responses for these prompts are underestimated at rollout=8. And these distinct responses in these most challenging prompts are crucial for pushing the model’s capability frontier, requiring more exploration. Similarly, for prompts with 1 incorrect answer at rollout=8 and we find that 12 and 21 groups have less than 16 incorrect responses with 128 rollouts on MATH and DAPO-Math-17k which may lead to over-exploitation as Figure 6(b).

###### E.2 Ablation Study on G

It is a widely accepted consensus that increasing the number of rollouts effectively mitigates estimation bias (Xiong et al., 2025). As the group size grows, the empirical advantage distribution converges closer to the true advantage distribution, thereby reducing the variance and bias inherent in the advantage estimation of group-relative RL algorithms. To rigorously validate the effectiveness of our dynamic adjusting approach HA-DW in mitigating estimation bias under constrained sampling conditions, we conducted a comparative analysis of model’s training performance across varying rollout sizes. The results presented in Table 3 shows that increasing the number of rollouts can, to a certain extent, enhance model performance by providing a more stable baseline. Although scaling up the number of rollouts is a straightforward method to improve performance, its benefits are often capped by computational constraints. As the rollout size increases, the training time grows substantially and out-of-memory (OOM) will occur once it exceeds a certain threshold. Our method offers a more efficient alternative: dynamic advantage adjustment demonstrates superior efficacy even under the limited rollouts condition, and it effectively mitigates the estimation bias that typically plagues low-sample scenarios, achieving robust performance without the need for extensive

sampling. E.3 Ablation Study on λscale

As illustrated in Section D.4,there exists a specific scaling factor λscale satisfying Equation (101) to compensate biased advantage estimation. Table 7 demonstrates the performance of RL training under different values of λscale. When λscale = 1.3 or 1.5, the trained model achieves the best performance across five benchmarks. The results correspond to our analysis that there exists an optimal value that balances the adjustment across prompts of varying difficulties, thereby enhancing RL training performance.

#### F Hard Evolving Difficulty Anchor

To simplify the update process of evolving belief Ct, thereby reducing algorithmic complexity. The synchronization of the model’s state can be facilitated through a hard update mechanism, executed at every training step. Let h be a hyperparameter denoting the number of most recent training rounds considered. Let h be hyper-paramter to represent the last h training rounds. The Equation (8) can be rewritten as:

1 h Ñ

é

h−1

h − 1 h

1 h

Ct− +

Ct+ =

yt =

yt−j + yt

,

j=1

(134) which indicates that the belief update is effectuated by directly synthesizing the accuracy information derived from the preceding h batches with observations from the current iteration, and we leave the remaining update procedures intact. Although this formulation ignores short-term oscillations in belief updates, it significantly simplifies the overall algorithm.

##### G Prompt Prompt Template

{question}. Let’s think step by step and output the final answer within \\boxed{}.

#### H Case Study

This appendix demonstrates some output examples generated by policy models trained with GRPO and GRPO+HA-DW. And the results are shown in Figure 7 and Figure 8.

###### λscale MATH500 AIME25 AMC23 Minerva OlympiadBench AVG

- 0.5 75.4 18.1 61.1 34.2 43.7 46.5

- 0.8 76.8 19.2 61.3 34.9 43.7 47.2
- 1.0 76.8 18.5 61.6 36.0 44.3 47.4

- 1.3 78.0 20.4 63.4 36.8 44.7 48.7 1.5 77.8 20.8 63.1 37.1 44.0 48.6

- 1.7 76.4 20.0 63.4 36.4 44.3 48.1
- 2.0 76.8 19.0 61.9 35.3 43.5 47.3

Table 7: Performance of Qwen3-4B-Base trained with GRPO+HA-DW on different λscale.

###### (b)

###### (a)

40

40

35

35

30

27

30

PromptCount

PromptCount

25

24

25

22

25

21 17

20

20

15 16

15

12

15

9

8

10

10

4 0

5

5

0

MATH DAPO-Math-17k

MATH DAPO-Math-17k

incorrect count n n<16 n=16 n>16

correct count n n<16 n=16 n>16

Figure 6: The distribution of prompts by the number of correct and incorrect responses on the MATH dataset and DAPO-Math-17k under 8 and 128 rollouts of Qwen3-4B-Base.

Hyperparameter GRPO GRPO+HA-DW GSPO GSPO+HA-DW DAPO DAPO+HA-DW

nnode 1 1 1 1 1 1 gpus per node 8 8 8 8 8 8 use kl in reward False False False False False False use kl loss False False False False False False tensor parallel size 1 1 1 1 1 1 test frequency 5 5 5 5 5 5

General

train batch size 256 256 256 256 256 256 mini batch size 16 16 16 16 16 16 micro batch size 4 4 4 4 4 4 epoch 3 3 3 3 9 9 gradient clip 1.0 1.0 1.0 1.0 1.0 1.0 optimizer AdamW AdamW AdamW AdamW AdamW AdamW warmup steps 10 10 10 10 10 10 weight decay 0.1 0.1 0.1 0.1 0.1 0.1 learning rate 1 × 10−6 1 × 10−6 1 × 10−6 1 × 10−6 1 × 10−6 1 × 10−6

Training

clip-high 0.2 0.2 0.0004 0.0004 0.28 0.28 clip-low 0.2 0.2 0.0003 0.0003 0.2 0.2

Clipping

max prompt length 1024 1024 1024 1024 1024 1024 max response length 4096 4096 4096 4096 4096 4096 rollout.n 8 8 8 8 8 8 do sample False False False False False False filtering False False False False False False dynamic batch size True True True True True True

Rollout

Table 8: Hyperparameter settings for Group-relative methods.

Case Study Question:

In the xy-plane, a circle of radius 4 with center on the positive x-axis is tangent to the y-axis at the origin, and a circle with radius 10 with center on the positive y-axis is tangent to the x-axis at the origin. What is the slope of the line passing through the two points at which these circles intersect? The final answer can be written in the form mn , where m and n are relatively prime positive integers. What is m+n? Let’s think step by step and output the final answer within \\ boxed{}.

#### Answer: 7

###### GRPO Solution:

To solve the problem, we need to find the points of intersection of the two circles and then determine the slope of the line passing through these points.

- 1. Equation of the Circles: The first circle has a radius of 4 and is centered on the positive x-axis. Since it is tangent to the y-axis at the origin, its center is at (4, 0). The equation of this circle is: (x − 4)2 + y2 = 16. The second circle has a radius of 10 and is centered on the positive y-axis. Since it is tangent to the x-axis at the origin, its center is at (0, 10). The equation of this circle is: x2 + (y − 10)2 = 100.
- 2. Finding the Points of Intersection: We need to solve the system of equations:

 



(x − 4)2 + y2 = 16 x2 + (y − 10)2 = 100

Expanding both equations, we get:

 



x2 − 8x + 16 + y2 = 16 =⇒ x2 − 8x + y2 = 0 x2 + y2 − 20y + 100 = 100 =⇒ x2 + y2 − 20y = 0

Subtract from the second equation: (x2 + y2 − 20y) − (x2 − 8x + y2) = 0 =⇒ −20y + 8x = 0 =⇒ 8x = 20y =⇒ x = 52y.

Substitute x = 52y into the first equation : 52y 2 − 8 52y + y2 = 0 =⇒ 254y2 − 20y + y2 = 0 =⇒ 25y2+4y2

4 − 20y = 0 =⇒ 294y2 − 20y = 0 29y2 − 80y = 0 =⇒ y(29y − 80) = 0 So, y = 0 or y = 2980. If y = 0, then x = 52·0 = 0, which is the origin and not a point of intersection of the two circles. If y = 8029, then x = 5·

80 29

2 = 20029 .

So, the points of intersection are 20029 , 8029 and (0, 0), but we already know (0, 0) is not a valid intersection point for the two circles. Therefore, the only valid intersection point is 20029 , 8029 .

- 3. Finding the Slope: The other point of intersection can be found by symmetry or by solving the equations again, but we know the circles are symmetric about the line y = x. So, the other point of intersection is 8029, 20029 . The slope of

200

29 − 8029 80

120 29

the line passing through the points 20029 , 8029 and 8029, 20029 is: slope =

−12029 = −1. So, the slope is −1, and in the form mn , we have m = −1 and n = 1.

29−20029 =

###### Therefore, m + n = −1 + 1 = 0.

|−1|
|---|

The final answer is:

Figure 7: An example of GRPO

###### Case Study GRPO+HA-DW Solution:

To solve the problem, we need to find the points of intersection of the two circles and then determine the slope of the line passing through these points.

- 1. Equation of the first circle: The first circle has a radius of 4 and is centered on the positive x-axis. Since it is tangent to the y-axis at the origin, the center of the circle is at (4, 0). The equation of the first circle is: (x − 4)2 + y2 = 16.
- 2. Equation of the second circle: The second circle has a radius of 10 and is centered on the positive y-axis. Since it is tangent to the x-axis at the origin, the center of the circle is at (0, 10). The equation of the second circle is: x2 + (y − 10)2 = 100.
- 3. Find the points of intersection: We need to solve the system of equations:

 



(x − 4)2 + y2 = 16 x2 + (y − 10)2 = 100

First, expand both equations:

 



x2 − 8x + 16 + y2 = 16 =⇒ x2 − 8x + y2 = 0 x2 + y2 − 20y + 100 = 100 =⇒ x2 + y2 − 20y = 0

Subtract the first expanded equation from the second: (x2 + y2 − 20y) − (x2 − 8x + y2) = 0 − 0 −20y + 8x = 0 =⇒ 8x = 20y =⇒ x = 52y.

Substitute x = 52y into the first expanded equation: 52y 2 − 8 52y + y2 = 0. This gives us two solutions for y: y = 0 or 294 y = 20 =⇒ y = 8029.

For y = 0: x = 25(0) = 0. So, one intersection point is (0, 0), which is the origin. For y = 8029: x = 52 8029 = 20029 . So, the other intersection point is 20029 , 8029 .

- 4. Find the slope of the line passing through the points (0, 0) and 20029 , 8029 : The slope m of the line passing

through two points (x1, y1) and (x2, y2) is given by: m = xy2−y1

2−x1 Substituting the points (0, 0) and 20029 , 8029 : m =

80 29 200 29

80 29 −0

= 20080 = 25.

29 −0 =

200

Therefore, the slope of the line is 25, and m + n = 2 + 5 = 7.

|7|
|---|

The final answer is

.

Figure 8: An example of GRPO+HA-DW

