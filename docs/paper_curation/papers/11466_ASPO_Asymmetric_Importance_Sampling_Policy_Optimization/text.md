# arXiv:2510.06062v2[cs.CL]15May2026

## When Importance Sampling Misallocates Credit: Asymmetric Ratios for Outcome-Supervised RL

Jiakang Wang2∗, Runze Liu1,2,3∗, Qingpeng Cai2, Lei Lin2, Wenping Hu2, Xiu Li3, Fuzheng Zhang2, Guorui Zhou2, Kun Gai2, Ling Pan1 1The Hong Kong University of Science and Technology, 2Kuaishou Technology, 3Tsinghua University

### Abstract

Reinforcement learning (RL) has shown great promise in large language models (LLMs) post-training, which typically rely on token-level clipping to maintain stability during optimization. Despite the empirical success of GRPO-style methods, we identify a fundamental and previously overlooked challenge in this popular Outcome-Supervised RL (OSRL) paradigm. We reveal that in OSRL, where advantages are shared across tokens within a response, importance sampling (IS) ratios deviate from their traditional purpose of distribution correction as in classic RL, which become token-level weights that allocate the shared advantage signal across tokens. We show that this hidden role shift induces a critical mismatch for positive-advantage tokens, leading to unbalanced token weighting between positive and negative tokens. Specifically, it suppresses the update of underrepresented tokens that are lagging behind, while over-amplifying already high-probability tokens. This mismatch results in rich-get-richer dynamics that over-reinforce confident tokens, weaken catch-up learning that drive entropy collapse, excessive repetition, and premature convergence. To address this, we propose Asymmetric Importance Sampling Policy Optimization (ASPO), a simple yet effective strategy that reverses the ratio-induced weighting of positive-advantage tokens, while stabilizing extreme updates and maintaining gradient flow. This mismatch correction aligns their update direction with the learning dynamics of negative ones. Comprehensive experiments across math reasoning and coding benchmarks demonstrate that ASPO significantly mitigates entropy collapse, improves training stability, and enhances performance over strong GRPO-based baselines. Our analysis provides new insights into the role of token-level weighting in OSRL and highlights the critical importance of correcting ratio-induced weighting in LLM RL.

### 1 Introduction

Reinforcement Learning (RL) has achieved remarkable success across various domains, including board games [32, 31], robotic control [10, 1, 4], Large Language Model (LLM) alignment [24, 2] and reasoning [6, 23, 39]. In classical RL, an agent interacts with an environment and receives intermediate rewards at different timesteps along a trajectory, where different actions contribute unequally to the final return, leading to fine-grained credit assignment for effective policy learning.

In this regime, policy optimization methods such as Proximal Policy Optimization (PPO) [28] rely on importance sampling (IS) ratios to account for the distribution mismatch between the behavior policy that generated the data and the current policy being optimized. Together with the clipping mechanism, IS plays a principled role in reusing on-policy samples for optimization while preventing

∗ Equal contribution

Preprint.

excessively large policy changes. In standard PPO, the IS ratio is coupled with a per-step advantage At, and each action-specific ratio modulates the contribution of the corresponding action-specific advantage (Figure 1(a)). Thus, the probability ratio primarily serves as a distribution-correction factor within a fine-grained credit-assignment structure.

Outcome-Supervised Reinforcement Learning (OSRL) for LLMs departs sharply from the traditional RL paradigm [41]. Group Relative Policy Optimization (GRPO) [29] and its variants, such as DAPO [39], have become widely adopted in this setting due to their simplicity and effectiveness. They estimate a response-level advantage from group rollouts Aˆresp which is shared across all tokens in the response. Despite its empirical success, GRPO inherits optimization machinery from PPO (e.g., IS ratios) in a regime that is substantially different from the one for which PPO was originally designed. We uncover a previously underexplored consequence of this design: under the commonly-adopted shared advantage setups with Aˆresp in OSRL, the standard distribution-correction of PPO-style ratios becomes incomplete. Since Aˆresp is shared across all tokens, it no longer provides token-level specific credit assignment, and consequently, the ratio degenerates into intra-response token-level weights that determine how strongly each token contributes to the gradient update.

This role shift leads to a critical mismatch in token weighting, as shown in Figure 4. When the oldpolicy probability is fixed, for positive-advantage samples, tokens that already have higher probability under the current policy receive larger update weights, while tokens with low probability are heavily down-weighted. This creates a richer-get-rich dynamic: already confident tokens in successful responses are reinforced more aggressively, while lagging but desirable tokens are under-updated when they may need stronger learning signals to catch up, as illustrated in Figure 1(b), which has been overlooked in previous research. Existing modifications to GRPO, such as clipping-threshold adjustments in DAPO, partially relax the upper clipping bound to facilitate low-probability token learning, but does not address the core issue that low-probability positive tokens remain weakly updated. This design contradicts the desired learning dynamics, where low-probability tokens in positive trajectories should receive stronger updates to catch up, and result in a series of abnormal training dynamics. Empirically, we observe accelerated entropy collapse, increasing repetition rate, rising clipping ratios, and premature convergence (Figure 2). These phenomena indicate unstable optimization driven by self-reinforcing updates on confident tokens, rather than healthy convergence.

To address these issues, we propose Asymmetric Importance Sampling Policy Optimization (ASPO), a simple yet effective method for GRPO-style OSRL. ASPO is based on the observation that, once a shared response-level advantage is broadcast to all tokens, the ratio primarily determines how this outcome signal is distributed across tokens. For positive-advantage responses, ASPO reverses the direction of ratio-induced token weighting: low-probability positive tokens receive stronger updates, while already confident tokens are down-weighted. This directly counteracts the rich-get-richer effect of standard GRPO, as shown in Figure 1. To further stabilize training, we incorporate a soft dual-clipping mechanism [38, 3] that constrains extreme ratios without discarding gradients for positive tokens. Extensive experiments on both mathematical reasoning and coding benchmarks demonstrate that ASPO: (1) mitigates entropy collapse and overfitting, (2) yields more stable training dynamics, and (3) significantly outperforms GRPO-based baselines in final performance.

###### Per-token advantage Response-level shared advantage

🏆 🏆 𝐴 > 0

𝐴 > 0

𝐴 𝐴 𝐴

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

𝑟 𝑟 𝑟

[Figure 5]

Rich-get-richer

[Figure 6]

Overupdate

[Figure 7]

ASPO Token 𝑡 :

[Figure 8]

… …

𝑟 > 1

[Figure 9]

Token 𝑡 : 𝑟 < 1

[Figure 10]

Catch-up update

[Figure 11]

Underupdate

… … 𝑡 𝑡 𝑡

… … … …

𝑡 𝑡 𝑡 𝑡 𝑡 𝑡

(a) (b) (c)

- Figure 1: Illustration of ratio-induced token misallocation in outcome-supervised RL and the ASPO correction. For positive tokens, ASPO prevents overoptimization of high-probability tokens and facilitates low-probability token learning.

In summary, our main contributions are as follows:

- • We identify a role shift of IS ratios in widely-adopted OSRL and a fundamental misallocation, where the IS ratio acts as token-level weighting and can misallocate credit across tokens: positiveadvantage tokens are weighted in the undesired direction, with confident tokens overoptimized and lagging tokens suppressed.
- • We propose ASPO, an asymmetric ratio-based mechanism that promotes catch-up learning for underrepresented successful tokens while preserving gradient flow.
- • We provide extensive empirical evidence that correcting token-level weighting significantly improves training stability and performance across multiple math reasoning and coding benchmarks.

### 2 Related Work

RL for Large Language Models. Reinforcement Learning from Human Feedback (RLHF) [5, 16] has achieved remarkable success in aligning LLMs with human values. Recently, DeepSeek-R1 [29] and the GRPO algorithm [29] have demonstrated that RLVR effectively enhances the reasoning capabilities of LLMs. Subsequent works have extended the GRPO algorithm for RLVR [19, 39, 40, 8, 35, 17, 33, 34, 43]. Our method follows this GRPO-based line of research but introduces a new approach of the clipping and IS mechanism to address inherent limitations in GRPO.

Clipping Mechanism in RL. PPO [28] introduces clipping based on importance sampling ratios as an alternative to KL divergence to constrain policy updates relative to the reference policy, and GRPO [29] adopts this clipping loss for LLM RL. There has been a growing line of work investigating whether this clipping is the right stabilization mechanism for LLM RL. CISPO [3] observes that gradients for clipped tokens are masked and proposes preserving them based on the PPO-Clip objective [28] but without the conservative min operation. GSPO [42] argues that the optimization objective should match the sequence-level reward’s granularity, proposing sequence-level clipping and IS ratio computation. DCPO [37] employs dynamic-adaptive clipping ranges instead of fixed bounds. Our method also targets the clipping term in GRPO but differs primarily by focusing on the IS ratio, incorporating a reciprocal weight for positive tokens. DAPO [39] and DPPO [27] further relax the clipping ranges of low-probability (positive) tokens, but do not address the problem of weight mismatch of positive tokens. These studies mainly refine the gating behavior of ratio-based objectives, deciding when updates should be clipped, masked, relaxed, or computed at a different granularity to maintain training stability. In contrast, our work focuses on the overlooked phenomenon of relative weighting of updates that remain active.

### 3 Preliminaries

#### 3.1 Group Relative Policy Optimization

Group Relative Policy Optimization (GRPO) [29] samples a group of G rollouts for advantage estimation: Aˆit = R

i−mean({Ri}Gi=1)

std({Ri}Gi=1) . The loss function of GRPO is defined as:

JGRPO(θ) =Eq∼D,{oi}Gi=1∼πθold(·|q)

 , (1)

|oi|

G

1 |oi|

 1 G

min rti(θ)Aˆit,clip rti(θ),1 − ε,1 + ε A ˆit − βDKL(πθ∥πref)

t=1

i=1

i t|q,oi<t)

where rti = πθ(o

πθold(oit|q,oi<t) is the Importance Sampling (IS) ratio, and β is a weight for the KullbackLeibler (KL) divergence between the current policy πθ and the reference policy πref.

#### 3.2 PPO Clipping and the Improvements

The clipping mechanism in GRPO [29] is first introduced by PPO-Clip [28]. It serves as an simple yet effective constraint to prevent large policy updates. However, it is shown in [3] that this clipping mechanism clips the value and also masks the gradient of the clipped tokens.

### 4 The Role Shift of Importance Sampling Ratio in OSRL

#### 4.1 Theoretical Understanding: PPO-style IS Ratios as Induced Token Weights in OSRL

Importance Sampling (IS) enters policy optimization as a way to evaluate an objective under one policy using samples collected from another [26] for optimizing a πθ(a|s)Aθ

(s,a), where

old

θ and θold denote parameters of the current and old policies, respectively, with A the stepwise advantage function for the current step’s state action pair (s,a). Since trajectories are sampled

from the old policy πθ

, this expectation is rewritten using an importance sampling estimator Ea∼π

old

πθ(a|s)

(s,a) . This derivation gives the likelihood ratio rt = π

πold a clear interpretation, which corrects the distribution mismatch between the old policy that generated the data and the current policy being optimized for that action [28]. In the classic PPO surrogate in the RL literature, this IS ratio is coupled with a step-specific advantage At, which specifies the estimated credit of the same timestep t (operating at the same granularity).

πθold(a|s)Aθ

θ

θold(·|s)

old

In OSRL for LLMs, such as GRPO [29] and its variants (e.g., DAPO [39]), this granularity alignment is broken. GRPO-style methods still use token-level probability ratios rt, but the advantage is not estimated at the token level. Instead, a response-level advantage Aˆresp is estimated and then broadcast to every token in the response, i.e., Ait = Aˆi,∀t. Thus, all tokens within a response share the same advantage value, which is fundamentally different from the classical setting where At provides timestep-specific credit. Once the same Aˆresp is assigned to every token, the advantage can no longer distinguish which tokens should receive stronger or weaker updates.

The original motivation of IS is to correct the distribution mismatch of the action-specific advantage term. However, under shared advantages, there is no token-specific credit estimation, which makes it unclear what the token-level ratio is correcting when the advantage itself no longer provides token-level credit, leading to the following critical question:

If the real credit of each token is already unclear due to outcome-based advantage estimation, what is the role and effect of further adjusting the distribution using IS weights?

We first answer this question theoretically by analyzing the gradient of GRPO, which is further empirically validated in Section 4.2. The following proposition shows that it is gradient-equivalent to a weighted log-likelihood objective with ratio-induced token weights.

Proposition 4.1 (IS ratio as token-wise gradient weighting in GRPO). Consider the GRPO objective

 

1, Ait ≥ 0 and rti ≤ 1 + ϵ, 1, Ait < 0 and rti ≥ 1 − ϵ, 0, otherwise

in Eq. (1). For each token oit, define the active clipping mask mit =



and the detached token weight wti = sg(mitrti), where sg(·) denotes the stop-gradient operator. Consider the weighted log-likelihood objective in Eq. (2), then we have ∇θJGRPO(θ) = ∇θJWLL(θ).

  1

 . (2)

|oi|

G

1 |oi|

wtiAit log πθ(oit|q,oi<t) − βDKL(πθ∥πref)

JWLL(θ) = Eq∼D,{oi}Gi=1∼πθold(·|q)

G

t=1

i=1

Remark. Proposition 4.1 reveals how the IS ratio enters the GRPO update, and the proof can be found in Appendix A.1. Because GRPO computes the advantage at the response level and broadcasts it to all tokens, the token-level ratio rti is no longer paired with a token-specific credit estimate. Instead, after clipping, it becomes the main token-dependent scalar that modulates the update of each token on the shared advantage, i.e., sg(mitrti)Aˆi. Therefore, tokens within the same response can receive different update magnitudes solely because their current-to-old probability ratios differ, rather than because their individual contributions to the response reward are different.

#### 4.2 Experimental Validation

The gradient view in Appendix A.2 motivates our empirical analysis. Although IS ratios are introduced to compensate for the mismatch between the behavior policy and the current policy, GRPO-style

GRPO ASPO GRPO w/o IS GRPO w/Pos Response-Level IS Mean

0.65

0.015

0.4

0.04

0.0015

0.60

0.3

0.03

0.010

0.0010

0.02

0.2

0.55

0.005

0.0005

0.01

0.1

0.50

0.0000

0.000

0.00

0 100 200 300 400

0 100 200 300 400

0 100 200 300 400

0 100 200 300 400

0 100 200 300 400

Steps

Steps

Steps

Steps

Steps

(a) Avg Eval Score

(b) Entropy

(c) Repetition Rate

(d) Positive Token CLip Ratio

(e) KL Loss

- Figure 2: The training dynamics curves of all methods on Qwen3-4B, including (a) average evaluation accuracy, (b) entropy, (c) repetition rate, (d) positive token clip ratio, and (e) KL loss. The curves are smoothed with EMA for better visualization.

OSRL computes advantages at the response level and broadcasts them to all tokens. Thus, the IS ratio becomes the main token-specific factor in the effective update weight. We therefore ask whether IS is necessary as a distribution-correction mechanism in this setting, or whether it mainly reshapes learning dynamics through ratio-induced weighting.

Setup. To evaluate the practical effects of IS weights in OSRL, we compare two variants that are identical except for the use of rt ratios: (1) GRPO with original IS weights, and (2) GRPO without IS weights (where all IS weights are fixed to 1.0). If IS were indispensable for correcting this mismatch, removing it would be expected to noticeably degrade performance or destabilize training.

- 4.2.1 Results

- Figure 2 compares the evaluation accuracy and training dynamics of standard GRPO and GRPO without IS. The two variants achieve comparable final test accuracy. Standard GRPO reaches its peak slightly earlier and stops to increase, whereas GRPO without IS converges more smoothly with competitive performance. Its training dynamics are also more stable, with slower entropy decay and smaller increases in repetition, and KL divergence. These results suggest that removing IS does not compromise final performance, but makes optimization less aggressive.

4.2.2 Analysis To understand what the IS ratios are doing, we further examine their values during GRPO training.

- Figure 3 shows the response-level IS ratios, where the average IS ratio is typically slightly greater than 1.0 (around 1.0004), and the ratios are not uniformly distributed across samples. Specifically, the average IS ratio of positive samples is slightly higher than that of negative samples. While the numerical gap (for each update) is small, it accumulates over many tokens and repeated off-policy updates, and its cumulative effect is significant. According to Proposition 4.1, the GRPO gradient

can be written as a weighted log-likelihood update whose effective weight is proportional to ri,tAˆi, which changes the relative strength of the corresponding gradient terms. For positive responses, ratios above one amplify the reward-increasing update. For negative responses, ratios below one tend to shrink the magnitude of the penalizing update in the gradient-active region. Thus, it causes GRPO training to prioritize learning positive responses rather than suppressing negative ones, and this gap widens across training, which amplifies positive-sample learning and accelerates entropy decay.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

100 120 140 160 180 200

Training Step

0.9998

1.0000

1.0002

1.0004

1.0006

1.0008

(a) Clip Ratio

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

100 120 140 160 180 200

Training Step

1.0002

1.0004

1.0006

1.0008

1.0010

1.0012

(b) Positive Clip Ratio

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

100 120 140 160 180 200

Training Step

0.99900

0.99925

0.99950

0.99975

1.00000

1.00025

1.00050

(c) Negative Clip Ratio

- Figure 3: Curves of response-level IS ratios throughout GRPO training. The average IS ratios are shown in gray dashed lines.

What happens without IS weights? With all IS ratios set to 1.0, the update no longer contains this ratio-induced preference amplification. On the one hand, this makes the overall weights slightly lower than in standard GRPO. On the other hand, it eliminates the weight difference between positive and negative samples. While these two factors slow down the learning speed compared to standard GRPO, they do not compromise the final performance, as shown in Figure 2.

This observation supports the role-shift interpretation in Section 4.1. In GRPO-style OSRL, although IS ratios still quantify the deviation between the current policy and the behavior policy, when a response-level advantage is shared by all tokens, their dominant operational effect is to reallocate the outcome-level learning signal across samples and tokens. The more aggressive entropy decay, repetition growth, clipping, and KL increase observed in Figure 2 are consistent with this ratio-induced amplification effect, and motivate the finer-grained token-level analysis in the next section.

Based on the above analysis, we draw the following conclusions: Takeaways for Importance Sampling

- • In the shared-advantage GRPO setting, removing ratio-induced weights preserves comparable final performance, while making optimization smoother and less aggressive.
- • Although IS ratios still measure current-to-old policy deviation, their dominant operational role is to act as effective update weights under shared response-level advantages.
- • IS ratios induce an asymmetric weighting pattern: positive responses are amplified more than negative responses are suppressed.

### 5 Ratio-induced Weight Mismatch in Positive-Advantage Responses

#### 5.1 Rethinking the Role of IS-induced Token Weights

The analysis in Section 4 shows that, under shared response-level advantages, PPO-style IS ratios (after clipping) in GRPO-style OSRL primarily act as token-level training weights, which modulate how strongly each token is updated under the same response-level advantage. This motivates a different question: if the ratio is viewed as an update weight, does the PPO-Clip weighting pattern allocate learning signal to tokens in a desirable way?

The design principle of PPO-Clip is to ensure training stability by preventing tokens that already have a strong advantage in the update direction from dominating the update. This avoids overly aggressive parameter changes that could push the model too far from the old policy. An ideal weighting scheme might look like this: along the update direction of the advantage, the lower a token’s probability is relative to the old policy, the larger its training weight should be. Conversely, the higher its probability, the smaller the weight should be. There are two reasons for this: (1) Assigning higher weight to tokens that are lagging behind accelerates their learning progress. (2) Such tokens are already far from the old policy, so updates pose less risk of destabilization. To make this more intuitive, we visualize IS weights in a three-dimensional coordinate plot, where the z-axis represents the original IS ratio, as shown in Figure 4(a). It can be seen that when the current probability is low, the corresponding IS weight is also small, resulting in insufficient training for these tokens.

#### 5.2 Weight Misallocation of Positive Tokens in PPO-Clip

As shown in Figure 4(c), for negative-advantage samples, the weight distribution behaves as expected: weights decrease gradually from the top-left region to the bottom-right region. However, for positiveadvantage samples shown in Figure 4(b), the allocation is the opposite to our intuition. Tokens in the top-left region, whose probabilities under the current policy are already much larger than under the old policy, are given larger weights, while tokens in the bottom-right region, with lower current probabilities, are assigned very small weights. Therefore, PPO-Clip underweights lagging tokens and overweights already reinforced tokens within positive-advantage samples.

This mismatch causes two problems: (1) tokens in the bottom-right region, which are clearly lagging behind the old policy, are suppressed further by excessively low weights. For example, if the old policy probability is 0.9 and the current policy probability is 0.1, the assigned weight is merely 1/9,

w/ Update r = 1.0 r = 0.8

###### r = 1.2

###### r = 3.0

|w/o Update| |
|---|---|
| || |
|---|
<br><br>Original Area<br><br>|
| |A: Original<br><br>B: Lower<br><br>|
|C: Higher| |
| | |
| | |
| | |
| | |
| | |

Lower Area

Higher Area

Dual-clip Area

| |
|---|

| |
|---|

| | | | | | |
|---|---|---|---|---|---|
|Advant|ages > 0| | | | |
| | |Cur|rent pr|ob|↓|
| | |IS w|eight|↓| |
|C|urrent Pr|ob = 0.1| | | |
|C|urrent Pr|ob = 0.5| | | |
|C|urrent Pr|ob = 0.9| | | |
| | | | | | |

1.2

×

1.0

1.0

1.0

ISRatio

0.8

D: Dual

3.0

0.6

C: Higher

0.8

0.8

2.5

0.4

A: Original

IS Ratio

2.0

0.2

CurrentProb

CurrentProb

0.0

1.5

0.6

0.6

1.0

3.2

0.5

B: Lower

Advantages < 0

0.4

0.4

2.8

0.0

Current prob ↓ IS weight ↓

ISRatio

2.4

1.0

0.8

2.0

CurrentProb

0.2

0.2

0.6

1.6

0.4

1.2

0.0

0.2

0.2

0.4

0.0

0.0

0.6

0.8

0.8

0.0

OldProb

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

1.0

0.0 0.2 0.4 0.6 0.8 1.0

Old Prob

Old Prob

Old Prob

(d) 2D Visualization

(b) Advantages > 0

(c) Advantages < 0

(a) 3D Visualization

- Figure 4: Visualization of IS weights in PPO-Clip. (a) 3D visualization of IS weights, showing that for a fixed old probability, tokens with lower current probabilities receive smaller IS weights. (b) When the advantage is positive (A > 0), GRPO optimizes regions A and B. (c) When the advantage is negative (A < 0), GRPO optimizes regions A and C. (d) 2D visualization of IS weights, illustrating that along a fixed old probability (dashed line), IS weights monotonically decrease as the current probability decreases. For negative-advantage tokens, this behavior aligns with our expectations, whereas for positive-advantage tokens, it contradicts the desired optimization trend.

resulting in a weak update signal and insufficient learning. (2) for tokens in the top-left region that already have a significant advantage, they are assigned disproportionately high weights. On the one hand, the model is more likely to deviate from the old policy, undermining training stability. On the other hand, the model overfits on positive samples, further amplifying their probabilities after the update. In the next update, their weights increase even more, forming a self-reinforcing loop. This mechanism underlies the previously observed phenomena such as entropy collapse and increased output repetition. For more details on how to distinguish healthy convergence and local optima, please refer to Appendix D.

#### 5.3 Experimental Validation

Setup. To empirically validate the preceding analysis, we conduct a controlled comparison between the original GRPO baseline and a modified variant. In this variant, we replace the token-level IS ratios of positive-advantage samples with their response-level average IS ratios, while keeping the negative-advantage samples unchanged. This design isolates the impact of the mismatched IS weights identified in Section 5.2, ensuring that any observed behavioral differences arise solely from the reweighting of positive tokens. If our hypothesis is correct, the modified setup should partially alleviate the weight mismatch issue.

#### 5.3.1 Results

Improved exploration and smoother training dynamics. After replacing the positive-token IS weights with response-level means, all training curves become substantially smoother. The entropy decline slows down (Figure 2), preventing premature convergence to local optima, and encouraging more diverse and exploratory behavior of the policy. Additionally, the increases in repetition rate, positive clipping ratio, and KL loss are noticeably moderated without the accelerating trends observed in GRPO. These results provide strong empirical evidence that the original IS ratio design indeed leads to unstable optimization through excessive weighting of high-probability tokens.

Better performance with stable training. As shown in Figure 2, the modified method achieves performance improvements compared with GRPO, demonstrating that stable training does not compromise final performance.

Takeaways for IS Weights of Positive Tokens

- • The standard PPO-Clip design introduces a token-weight mismatch for positive-advantage samples, where high-probability tokens receive disproportionately large update weights.

- • This imbalance leads to entropy collapse, and increased repetition, ultimately pushing the policy toward a local optimum and limiting its capacity for continual improvement.
- • Reweighting positive samples with response-level IS ratios effectively mitigates these issues, confirming the validity of our preliminary analysis and motivating the design of our proposed method in the next section.

### 6 Asymmetric Importance Sampling Policy Optimization

Based on the analysis in Section 4 and 5, we show that token-level IS ratios in GRPO-style OSRL should be understood primarily as token-wise update weights under shared response-level advantages. In addition, the response-level mean-ratio diagnostic in Section 5 shows that removing token-level ratio variation from positive-advantage responses substantially stabilizes training. This perspective reveals an asymmetric issue: the original PPO-style ratio weighting is reasonably aligned with negative-advantage responses, but mismatched for positive-advantage responses. For Aˆ > 0, tokens with πθ ≪ πold are lagging behind the desired update direction but receive small weights, while tokens with πθ ≫ πold have already been reinforced but receive large weights.

Motivated by this, we propose Asymmetric Importance Sampling Policy Optimization (ASPO), a simple yet effective approach for clipping and IS ratio computation, following the mismatch analysis in Section 5. ASPO inverts the IS weights of positive samples, aligning their update behavior with that of negative samples. In other words, tokens whose current policy probabilities are lower than the old policy should be assigned higher learning weights, while those with higher probabilities should receive lower weights.

Specifically, the implementation can be divided into three steps:

- Step 1: Token Masking. We retain the original clipping mechanism in GRPO. The gradient of

tokens that satisfy: (1) rti(θ) < 1 − εlow (Aˆit < 0) or (2) rti(θ) > 1 + εhigh (Aˆit > 0) will be masked in a hard clipping manner.

- Step 2: Weight Flipping. For tokens with negative advantage values, the ASPO ratio is the same

as that of GRPO, i.e., rˆti = rti. For tokens with Aˆit > 0, we use the reciprocal of their IS weights and the ASPO ratio is computed as:

rˆti =

πθ

old

(oit | q,oi<t)πθ(oit | q,oi<t) sg(πθ2(oit | q,oi<t))

, (3)

where sg(·) denotes stop gradient operation. The gradient analysis in Appendix A.2 shows that the gradient of positive tokens in ASPO is positively correlated with π1

θ

, indicating that the gradient becomes larger when the probability is lower.

- Step 3: Dual Clipping. PPO-Clip usually uses a dual-clip mechanism [38] to handle cases where, for Aˆ < 0, extremely small or large ratios could lead to weight explosion, destabilizing training. Originally, for the Aˆ > 0 region, this problem was naturally avoided by the hard clipping mechanism. However, since we now invert the weights for positive samples, extreme cases shift to the right-hand side of the Aˆ > 0 region (region B in Figure 4(b)). Therefore, when using ASPO, positive-sample tokens also require dual-clip. Specifically, this dual-clip is implemented using a soft clipping manner, which only clips the values but retains the gradient.

It is important to note that tokens clipped by dual-clip are fundamentally different from tokens masked in the first step. The latter are blocked because they already have sufficient advantage in the update direction, whereas the former are tokens that lag significantly behind the old policy but need their weight magnitude constrained due to abnormal computation. We still want these tokens to participate in training, so we use the soft clipping in CISPO [3] for these dual-clipped tokens.

- Table 1: Evaluation results on mathematical benchmarks. The results of ASPO are shaded and the highest values are bolded.

Method

AIME24 AIME25 AMC23 MATH-500 Minerva Olympiad

Avg.

avg@64 pass@64 avg@64 pass@64 avg@64 pass@64 avg@4 pass@4 avg@8 pass@8 avg@4 pass@4

DeepSeek-R1-1.5B 30.6 80.0 23.5 63.3 70.7 100.0 83.6 92.4 27.6 48.2 44.6 59.4 46.8 GRPO 42.1 80.0 28.6 56.7 80.3 97.5 87.6 94.6 29.2 46.3 53.2 65.8 53.5 DeepScaleR-1.5B 42.0 83.3 29.0 63.3 81.3 100.0 87.7 93.6 30.3 51.1 50.7 61.0 53.5 Nemotron-1.5B 48.0 76.7 33.1 60.0 86.1 97.5 90.6 93.6 35.3 47.8 59.2 66.8 58.7 ASPO-Math-1.5B 49.0 80.0 35.1 70.0 87.2 95.0 90.5 94.4 35.1 50.4 58.8 66.9 59.3

Qwen3-4B 23.6 56.7 18.3 63.3 67.7 95.0 84.5 92.4 41.5 56.3 54.1 66.6 48.3 GRPO 43.4 83.3 35.5 70.0 84.3 97.5 91.7 95.8 47.2 58.5 67.4 75.8 61.6 ASPO-Math-4B 50.5 83.3 40.9 70.0 87.4 97.5 93.4 97.0 49.0 60.7 68.8 78.5 65.0

Qwen3-8B 27.0 63.3 19.1 56.7 68.9 97.5 83.6 92.4 43.9 58.1 55.7 69.6 49.7 GRPO 50.3 83.3 34.1 66.7 84.1 95.0 92.7 96.0 50.2 61.4 68.2 76.1 63.3 ASPO-Math-8B 52.4 83.3 38.9 80.0 89.1 97.5 93.6 96.4 50.6 60.7 69.8 77.9 65.7

Qwen3-30B-A3B 30.1 70.0 19.9 56.7 74.3 100.0 88.4 96.0 47.7 59.6 59.6 71.2 53.3 GRPO 59.5 80.0 43.1 73.3 91.9 97.5 95.3 97.8 51.9 61.8 71.0 79.2 68.8 ASPO-Math-30B 61.7 90.0 50.3 80.0 94.9 97.5 95.7 98.4 54.3 64.3 74.3 81.5 71.8

- Table 2: Evaluation results on code benchmarks. The results of ASPO are shaded and the highest values are bolded.

LCB v5 (2024.08.01-2025.02.01) LCB v6 (2025.02.01-2025.05.01)

Method

Avg. avg@8 pass@8 avg@16 pass@16

DeepSeek-R1-1.5B 16.7 29.0 17.2 34.4 17.0 GRPO 26.0 40.5 27.6 43.5 26.8 DeepCoder-1.5B 23.3 39.1 22.6 42.0 23.0 Nemotron-1.5B 26.1 35.5 29.5 42.8 27.8 ASPO-Code-1.5B 31.5 47.0 30.5 46.0 31.0

Qwen3-4B 23.8 35.8 24.0 35.1 23.9 GRPO 40.8 55.2 36.6 45.0 38.7 ASPO-Code-4B 44.8 58.1 38.3 47.3 41.6

### 7 Experiments

#### 7.1 Setup

Models and Baselines. We conduct experiments using DeepSeek-R1-Distill-Qwen-1.5B [6], Qwen3-4B, Qwen3-8B, and Qwen3-30B-A3B1 [36] as the base model. We compare ASPO against several representative baselines: (1) Base Model, (2) GRPO [29], (3) DeepScaleR-1.5B [19], (4) DeepCoder-1.5B [18], and (5) Nemotron-1.5B [15].

Evaluation. We evaluate models on both mathematical and coding domains. For math, we use six challenging datasets: AIME24 [21], AIME25 [22], AMC23 [20], MATH-500 [14], Minerva Math [12], and OlympiadBench [7]. For coding, we adopt LiveCodeBench v5 (2024.08.01–2025.02.01) and v6 (2025.02.01–2025.05.01) [9]. vLLM [11] is used for inference with a maximum output length of 32,768 tokens and a temperature of 0.8. We report both avg@K and pass@K for each benchmark.

Implementation Details. We implement ASPO based on GRPO [29] with verl [30]. The training batch size is set to 64, with a mini-batch size of 16. The learning rate is 1.0 × 10−6. Additional implementation details are provided in Appendix B.

#### 7.2 Main Results

The results in Table 1 and Table 2 show that, when using DeepSeek-R1-Distill-Qwen-1.5B as the base model, ASPO improves upon the base model by 12.5% and 14.0% on math and coding tasks, respectively. Moreover, ASPO consistently outperforms GRPO and several strong OSRL methods

1We use non-thinking mode for Qwen3 models.

averaged across all evaluated benchmarks, demonstrating the effectiveness of ASPO. On larger dense and MoE models, ASPO still exceeds the baselines by a large margin, demonstrating the effectiveness across different model sizes. As shown in Table 3, ASPO still outperforms the baseline method on OOD tasks.

#### 7.3 Analysis

As shown in Figure 2 and Table 1, compared with GRPO, ASPO yields trends that align with our earlier findings, but with even stronger improvements. Specifically, entropy decreases more gradually, the repetition rate and positive token clipping ratio grow more slowly, and all metrics eventually stabilize, showing characteristics of healthy convergence as discussed in Appendix D. More training dynamics results are shown in Appendix C.

More importantly, since entropy declines more smoothly and remains at a higher level, the model avoids premature collapse and continues learning effectively. As training progresses, performance steadily improves, significantly surpassing the best results achieved by GRPO-based training.

It is also noteworthy that in the early training stages, models trained with ASPO exhibit slightly slower reward improvement (Figure 5(a) and 6(a)) compared to other variants. This occurs because inverting positive-sample weights reduces the overall average IS weight, leading to slower initial fitting of positive samples. However, as training continues, the performance not only catches up but ultimately surpasses the other approaches.

### 8 Conclusion

In this paper, we identify a fundamental role shift of IS in GRPO-style OSRL: under shared response-level advantages, token-level IS ratios primarily act as token-wise update weights rather than distribution-correction terms. We further reveal a weight mismatch for positive-advantage tokens, where already-confident tokens are over-amplified while lagging tokens are under-updated, leading to entropy collapse and unstable training. To address this, we propose ASPO, which flips the IS ratios of positive tokens and incorporates a soft dual-clipping mechanism to stabilize OSRL for LLMs. Experimental results across mathematical and coding domains demonstrate that our method effectively alleviates token-level weight mismatch, mitigates entropy collapse, and improves training stability, leading to superior model performance than the baselines.

Limitations and Discussion. Our investigation primarily targets the importance sampling mechanism within the widely-adopted GRPO-based methods, which have recently emerged as a popular paradigm for efficient LLM post-training. It is a promising direction to extend our investigation to other RL paradigms, e.g., PPO with token-level IS ratios and advantages. We hope our work serves as a foundation for these broader explorations.

### References

- [1] OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, et al. Learning dexterous in-hand manipulation. The International Journal of Robotics Research, 39(1):3–20, 2020.
- [2] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.
- [3] Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.
- [4] Yuanpei Chen, Tianhao Wu, Shengjie Wang, Xidong Feng, Jiechuan Jiang, Zongqing Lu, Stephen McAleer, Hao Dong, Song-Chun Zhu, and Yaodong Yang. Towards human-level bimanual dexterous manipulation with reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 5150–5163, 2022.

- [5] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.
- [6] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.
- [7] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
- [8] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025.
- [9] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025.
- [10] Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research, 32(11):1238–1274, 2013.
- [11] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, page 611–626, New York, NY, USA, 2023. Association for Computing Machinery.
- [12] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving Quantitative Reasoning Problems with Language Models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 3843–3857. Curran Associates, Inc., 2022.
- [13] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022.
- [14] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024.
- [15] Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025.
- [16] Runze Liu, Fengshuo Bai, Yali Du, and Yaodong Yang. Meta-reward-net: Implicitly differentiable reward learning for preference-based reinforcement learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 22270–22284. Curran Associates, Inc., 2022.

- [17] Runze Liu, Jiakang Wang, Yuling Shi, Zhihui Xie, Chenxin An, Kaiyan Zhang, Jian Zhao, Xiaodong Gu, Lei Lin, Wenping Hu, Xiu Li, Fuzheng Zhang, Guorui Zhou, and Kun Gai. Attention as a compass: Efficient exploration for process-supervised rl in reasoning models. arXiv preprint arXiv:2509.26628, 2025.
- [18] Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepcoder: A fully open-source 14b coder at o3-mini level. https://pretty-radio

-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Lev el-1cf81902c14680b3bee5eb349a512a51, 2025. Notion Blog. Accessed: 2026-04-30.

- [19] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/D eepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681 902c1468005bed8ca303013a4e2, 2025. Notion Blog. Accessed: 2026-04-30.
- [20] MAA. American mathematics contest 12 (amc 12), November 2023. Accessed: 2026-04-30.
- [21] MAA. American invitational mathematics examination (aime), February 2024. Accessed: 2026-04-30.
- [22] MAA. American invitational mathematics examination (aime), February 2025. Accessed: 2026-04-30.
- [23] OpenAI. Learning to reason with llms, 2024. Accessed: 2026-04-30.
- [24] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022.
- [25] Guilherme Penedo, Anton Lozhkov, Hynek Kydlíˇcek, Loubna Ben Allal, Edward Beeching, Agustín Piqueres Lajarín, Quentin Gallouédec, Nathan Habib, Lewis Tunstall, and Leandro von Werra. Codeforces. https://huggingface.co/datasets/open-r1/codeforces, 2025. Accessed: 2026-04-30.
- [26] Doina Precup, Richard S. Sutton, and Satinder P. Singh. Eligibility traces for off-policy policy evaluation. In Proceedings of the Seventeenth International Conference on Machine Learning, ICML ’00, page 759–766, San Francisco, CA, USA, 2000. Morgan Kaufmann Publishers Inc.
- [27] Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, and Wee Sun Lee. Rethinking the trust region in llm reinforcement learning. arXiv preprint arXiv:2602.04879, 2026.
- [28] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [29] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [30] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, page 1279–1297, New York, NY, USA, 2025. Association for Computing Machinery.
- [31] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419):1140–1144, 2018.

- [32] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676):354–359, 2017.
- [33] Zhenpeng Su, Leiyu Pan, Xue Bai, Dening Liu, Guanting Dong, Jiaming Huang, Minxuan Lv, Wenping Hu, Fuzheng Zhang, Kun Gai, et al. Klear-reasoner: Advancing reasoning capability via gradient-preserving clipping policy optimization. arXiv preprint arXiv:2508.07629, 2025.
- [34] Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, and Guorui Zhou. Ce-gppo: Coordinating entropy via gradient-preserving clipping policy optimization in reinforcement learning. arXiv preprint arXiv:2509.20712, 2025.
- [35] Jiakang Wang, Runze Liu, Fuzheng Zhang, Xiu Li, and Guorui Zhou. Stabilizing knowledge, promoting reasoning: Dual-token constraints for rlvr. arXiv preprint arXiv:2507.15778, 2025.
- [36] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [37] Shihui Yang, Chengfeng Dou, Peidong Guo, Kai Lu, Qiang Ju, Fei Deng, and Rihui Xin. Dcpo: Dynamic clipping policy optimization. arXiv preprint arXiv:2509.02333, 2025.
- [38] Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng Wu, Qingwei Guo, Qiaobo Chen, Yinyuting Yin, Hao Zhang, Tengfei Shi, Liang Wang, Qiang Fu, Wei Yang, and Lanxiao Huang. Mastering complex control in moba games with deep reinforcement learning. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04):6672–6679, Apr. 2020.
- [39] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [40] Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.
- [41] Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, Yu Fu, Xingtai Lv, Yuchen Zhang, Sihang Zeng, Shang Qu, Haozhan Li, Shijie Wang, Yuru Wang, Xinwei Long, Fangfu Liu, Xiang Xu, Jiaze Ma, Xuekai Zhu, Ermo Hua, Yihao Liu, Zonglin Li, Huayu Chen, Xiaoye Qu, Yafu Li, Weize Chen, Zhenzhao Yuan, Junqi Gao, Dong Li, Zhiyuan Ma, Ganqu Cui, Zhiyuan Liu, Biqing Qi, Ning Ding, and Bowen Zhou. A survey of reinforcement learning for large reasoning models. arXiv preprint arXiv:2509.08827, 2025.
- [42] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
- [43] Xuekai Zhu, Daixuan Cheng, Dinghuai Zhang, Hengli Li, Kaiyan Zhang, Che Jiang, Youbang Sun, Ermo Hua, Yuxin Zuo, Xingtai Lv, Qizheng Zhang, Lin Chen, Fanghao Shao, Bo Xue, Yunchong Song, Zhenjie Yang, Ganqu Cui, Ning Ding, Jianfeng Gao, Xiaodong Liu, Bowen Zhou, Hongyuan Mei, and Zhouhan Lin. FlowRL: Matching reward distributions for LLM reasoning. In The Fourteenth International Conference on Learning Representations, 2026.

### A Theoretical Analysis

- A.1 Proof of Proposition 4.1 Proposition 4.1 (IS ratio as token-wise gradient weighting in GRPO). Consider the GRPO objective in

 

1, Ait ≥ 0 and rti ≤ 1 + ϵ, 1, Ait < 0 and rti ≥ 1 − ϵ, 0, otherwise

Eq. (1). For each token oit, define the active clipping mask mit =

and



the detached token weight wti = sg(mitrti), where sg(·) denotes the stop-gradient operator. Consider the weighted log-likelihood objective in Eq. (4), then we have that ∇θJGRPO(θ) = ∇θJWLL(θ).

  1

 . (4)

|oi|

G

1 |oi|

wtiAit log πθ(oit|q,oi<t) − βDKL(πθ∥πref)

JWLL(θ) = Eq∼D,{oi}Gi=1∼πθold(·|q)

G

t=1

i=1

Proof. We prove in three cases:

- Case 1: Ait ≥ 0 and rti > 1 + ϵ. We have

JWLL(θ) = Eq∼D,{oi}Gi=1∼πθold(·|q)

  1

G

G

i=1

1 |oi|

|oi|

t=1

− βDKL(πθ∥πref)

 . (5)

Also, we have

JGRPO(θ) = Eq∼D,{oi}Gi=1∼πθold(·|q)

  1

G

G

i=1

1 |oi|

|oi|

t=1

(1 + ε)Aˆit − βDKL(πθ∥πref)

 . (6)

Thus, we have ∇θJGRPO(θ) = ∇θJWLL(θ).

- Case 2: Ait < 0 and rti < 1 − ϵ. We have

JWLL(θ) = Eq∼D,{oi}Gi=1∼πθold(·|q)

  1

G

G

i=1

1 |oi|

|oi|

t=1

− βDKL(πθ∥πref)

 . (7)

Also, we have

JGRPO(θ) = Eq∼D,{oi}Gi=1∼πθold(·|q)

  1

G

G

i=1

1 |oi|

|oi|

t=1

(1 − ε)Aˆit − βDKL(πθ∥πref)

 . (8)

Thus, we have ∇θJGRPO(θ) = ∇θJWLL(θ).

- Case 3: Otherwise, we have

|oi|

G

1 |oi|

1 G

JWLL(θ) = Eq∼D,{oi}Gi=1∼πθold(·|q)

t=1

i=1

sg

− βDKL(πθ∥πref) .

πθ(oit|q,oi<t) πold(oit|q,oi<t)

Ait log πθ(oit|q,oi<t)

(9)

  1

|oi|

G

1 |oi|

JGRPO(θ) = Eq∼D,{oi}Gi=1∼πθold(·|q)

G

t=1

i=1

πθ(oit|q,oi<t) πold(oit|q,oi<t)

Aˆit − βDKL(πθ∥πref)

 .

(10)

##### Thus, we have ∇θJGRPO(θ) = ∇θJWLL(θ).

| |
|---|

- A.2 Gradient Analysis The gradient of original GRPO is as follows:

G

1 G

rti(θ)Aˆit

∇θJ (θ) = ∇θEq∼D,{oi}Gi=1∼πθold(·|q)

i=1

G

∇θπθ(oit | q,oi<t) πθ

1 G

Aˆit

(11)

= Eq∼D,{oi}Gi=1∼πθold(·|q)

(oit | q,oi<t)

old

i=1

G

πθ(oit | q,oi<t) πθ

1 G

(oit | q,oi<t)∇θ log πθ(oit | q,oi<t)Aˆit

= Eq∼D,{oi}Gi=1∼πθold(·|q)

old

i=1

##### where πθ denotes πθ(oit | q,oi<t) and πθ

(oit | q,oi<t). Then, we derive the gradient of positive tokens in ASPO as follows:

denotes πθ

old

old

G

1 G

rˆti(θ)Aˆit

∇θJ (θ) = ∇θEq∼D,{oi}Gi=1∼πθold(·|q)

i=1

G

(oit | q,oi<t)∇θπθ(oit | q,oi<t) sg(πθ2(oit | q,oi<t))

##### πθ

1 G

Aˆit

= Eq∼D,{oi}Gi=1∼πθold(·|q)

old

i=1

G

(oit | q,oi<t)πθ(oit | q,oi<t) πθ2(oit | q,oi<t) ∇θ log πθ(oit | q,oi<t)Aˆit

##### πθ

1 G

= Eq∼D,{oi}Gi=1∼πθold(·|q)

old

i=1

G

(oit | q,oi<t) πθ(oit | q,oi<t) ∇θ log πθ(oit | q,oi<t)Aˆit

##### πθ

1 G

= Eq∼D,{oi}Gi=1∼πθold(·|q)

old

i=1

(12) It is worth noting that the gradient of positive tokens in (12) differs from the original gradient of GRPO in (11) at the highlighted red term. From the above derivation, we can observe that the

gradient of ASPO is positively correlated with π1

, indicating that the gradient becomes larger when the probability of a token is lower.

θ

### B Experimental Details

Baselines. We conduct experiments using DeepSeek-R1-Distill-Qwen-1.5B [6], Qwen3-4B, Qwen3-8B, and Qwen3-30B-A3B [36] as the base model and compare ASPO with the following baselines:

- • Base Model: The original model without any RL fine-tuning.
- • GRPO [29]: A strong OSRL algorithm with token-level loss.
- • DeepScaleR-1.5B [19]: A 1.5B model trained for mathematical reasoning with iterative contextlength expansion.
- • DeepCoder-1.5B [18]: A 1.5B model trained on code datasets using similar context expansion strategies as DeepScaleR.
- • Nemotron-1.5B [15]: A strong 1.5B reasoning model with reference policy resetting.

Evaluation. We follow standard practice [15, 35] and report both avg@K and pass@K metrics. For benchmarks with fewer samples (AIME24/25 and AMC23), we set K = 64. For LiveCodeBench v6, we use K = 16; for LiveCodeBench v5 and Minerva Math, K = 8; and for MATH-500 and OlympiadBench, K = 4. To ensure fair and accurate evaluation, we adopt the official verification

functions from both DeepScaleR and Math-Verify2 for mathematical problems, following the protocol in [8].

Implementation Details. For training data, we use a mixture of DeepScaleR-Preview-Dataset [19], Skywork-OR1-RL-Data [8], and DAPO-Math-17K [39] for DeepSeek-R1-Distill-Qwen-1.5B on mathematical tasks. For Qwen3-4B, Qwen3-8B, and Qwen3-30B-A3B, we directly use DAPO-Math17K for training. For coding, we employ DeepCoder [18], CodeContests [13], and CodeForces [25] datasets. All datasets are cleaned and filtered following the preprocessing protocol of [35]. After filtering, the mathematical dataset contains 70.8k samples, while the coding dataset contains 8.9k samples. The clipping ranges of GRPO and ASPO are set to ε = 0.2. The KL divergence is incorporated as an explicit loss term (k3 estimator) with coefficient β = 0.001, and all baselines use the same KL implementation to ensure that observed differences in entropy and stability are not caused by KL. For each prompt, 16 responses are sampled with a temperature of 1.0. We use a max response length of 32,768 tokens for DeepSeek-R1-Distill-Qwen-1.5B, 8,192 for Qwen3-4B and Qwen3-30B-A3B, and 6,144 for Qwen3-8B. Experiments of 1.5B, 4B, and 8B LLMs are conducted on 8 NVIDIA H800 GPUs and Qwen3-30B-A3B is trained with 32 NVIDIA H800 GPUs.

### C Additional Experimental Results

We present the training dynamics and test performance curves on mathematical tasks in Figure 5, 6, and 7. As shown in Figures 5 and 6, although ASPO exhibits slightly slower learning during the initial training phase compared to GRPO, it maintains more stable model entropy and consistently lower repetition rate throughout training. These results demonstrate the superior training stability of ASPO.

Ablation on Dual Clipping. Weight Flipping is the core mechanism of ASPO, while Dual Clipping is a stabilizer introduced because, after inversion, extreme positive-token ratios shift to the side where their magnitude also needs to be bounded. As shown in Figure 8, without Dual Clipping, the gradient norm exhibits explosive spikes during training, while ASPO with Dual Clipping maintains stable gradient dynamics. This confirms that Dual Clipping is essential for stabilizing training in extreme cases, complementing the Weight Flipping mechanism.

Table 3: Evaluation results on OOD benchmarks. The results of ASPO are shaded and the highest values are bolded.

Method ARC-Challenge GPQA-Diamond MMLU-Pro Avg. Qwen3-4B 74.7 14.3 31.5 40.2

GRPO 90.1 29.2 49.6 56.3 ASPO-Math-4B 87.6 37.3 53.5 59.5

Qwen3-8B 52.9 8.6 29.6 30.4 GRPO 62.4 33.0 42.9 40.6 ASPO-Math-8B 88.6 56.3 57.4 59.8

Table 4: Evaluation results with Qwen3-4B. The highest values are bolded.

Method AIME24 AIME25 AMC23 MATH-500 Minerva Olympiad Avg. GRPO 43.4 35.5 84.3 91.7 47.2 67.4 61.6 GRPO w/o IS 46.4 36.3 88.9 92.9 49.3 67.2 63.5 GRPO w/ Pos. Resp. IS Mean 46.5 42.3 87.4 92.0 48.6 67.1 64.0

### D How to distinguish between “healthy convergence” and “local optima”?

To better contextualize our analysis, we introduce an important concept in RL training: how to distinguish between healthy convergence and local optima.

2https://github.com/huggingface/Math-Verify

GRPO ASPO

0.4

0.70

0.04

3500

0.65

0.3

0.03

3000

0.60

0.02

0.2

0.55

2500

0.01

0.1

0.50

2000

0.45

0.00

0 100 200 300 400

0 100 200 300 400

0 100 200 300 400

0 100 200 300 400

Steps

Steps

Steps

Steps

(a) Reward

(b) Entropy

(c) Repetition Rate

(d) Response Length

- Figure 5: The training dynamics curves of all methods on Qwen3-4B, including (a) training reward, (b) model entropy, (c) repetition rate, and (d) response length. The curves are smoothed with EMA for better visualization.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0 100 200 300 400

Steps

0.45

0.50

0.55

0.60

0.65

0.70

(a) Reward

0 100 200 300 400

Steps

0.1

0.2

0.3

0.4

(b) Entropy

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0 100 200 300 400

Steps

0.000

0.005

0.010

0.015

0.020

(c) Repetition Rate

0 100 200 300 400

Steps

3000

3500

4000

(d) Response Length

GRPO ASPO

- Figure 6: The training dynamics curves of all methods on Qwen3-8B, including (a) training reward, (b) model entropy, (c) repetition rate, and (d) response length. The curves are smoothed with EMA for better visualization.

0 100 200 300 400

Steps

0.25

0.30

0.35

0.40

0.45

0.50

(a) AIME24 (Avg@32)

0 100 200 300 400

Steps

0.20

0.25

0.30

0.35

0.40

(b) AIME25 (Avg@32)

0 100 200 300 400

Steps

0.70

0.75

0.80

0.85

(c) AMC23 (Avg@32)

0 100 200 300 400

Steps

0.84

0.86

0.88

0.90

0.92

(d) MATH-500 (Avg@4)

0 100 200 300 400

Steps

0.42

0.44

0.46

0.48

0.50

(d) Minerva (Avg@8)

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0 100 200 300 400

Steps

0.525

0.550

0.575

0.600

0.625

0.650

0.675

(e) Olympiad (Avg@4)

GRPO ASPO

- Figure 7: The test curves of all methods trained with Qwen3-4B on six mathematical benchmarks.

In RL training, healthy convergence typically exhibits the following characteristics:

- • Entropy decreases gradually from a relatively high initial value and stabilizes at a small but positive level, indicating that the policy becomes more deterministic while retaining moderate exploration.
- • The reward curve increases steadily and eventually plateaus at a stable high value.

GRPO ASPO ASPO w/o Dual Clip

15000

0.0

0.4

0.04

0.03

0.1

0.3

10000

0.02

0.2

0.2

5000

0.01

0.3

0.1

0

0.00

0 100 200 300 400

0 100 200 300 400

0 100 200 300 400

0 100 200 300 400

Steps

Steps

Steps

Steps

(a) Entropy

(b) Repetition Rate

(c) Gradient Norm

(d) Policy Loss

- Figure 8: Ablation on dual clipping. The training dynamics curves of ASPO with and without dual clipping on Qwen3-4B, including (a) entropy, (b) repetition rate, (c) gradient norm, and (d) policy loss. Without dual clipping, the gradient norm and policy loss exhibit explosive spikes during training, demonstrating that dual clipping is essential for stabilizing training in extreme cases. The curves are smoothed with EMA for better visualization.

• Clip ratios and KL divergence loss remain stable during later training stages, without drastic fluctuations.

In contrast, when the training becomes trapped in a local optimum, the model enters a self-reinforcing policy-data distribution loop, characterized by:

- • Entropy collapses rapidly toward zero.
- • The reward curve stagnates with no further improvement.
- • Persistently high clip ratios without meaningful policy updates.

When training GRPO-based approaches, we may observe this phenomenon: in late-stage training, entropy collapses, repetition rates spike, clip ratios surge, and the model’s test performance begins to degrade, indicating convergence to a local optimum.

In early training, moderate entropy reduction and a gradual increase in clip ratio are expected. If accompanied by an increasing rewards, it indicates that the policy is learning effectively. However, when later stages exhibit an abrupt entropy drop, reward stagnation, and persistently high clip ratios without further progress, it clearly signals that the model has fallen into a local optimum.

The underlying cause lies in the token-level weight mismatch for positive samples in PPO-Clip identified in Section 5, which drives the model to overfit certain high-probability tokens, eventually leading to entropy collapse and training degradation.

