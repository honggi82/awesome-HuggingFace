# arXiv:2606.10968v2[cs.LG]10Jun2026

[Figure 1]

## Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning

Renjie Mao∗ Xiangxin Zhou∗† Lvfang Tao∗† Yixin Ding Yu Shi

Yongguang Lin Yuheng Wu Honglin Zhu Qian Qiu Wenxi Zhu† Tencent Hunyuan

∗Equal contribution †Corresponding author

Abstract Reinforcement learning with verifiable rewards (RLVR) has become standard for improving LLM reasoning. However, existing PPO-style trust-region mechanisms remain position-agnostic by enforcing uniform thresholds across all tokens independently. This pointwise treatment conflicts with autoregressive generation in two critical ways. First, uniform thresholds ignore autoregressive asymmetry. Early-stage deviations produce compounding sequence-level drift, causing static thresholds to under-regulate early divergence and excessively constrain late-stage exploration. Second, evaluating token-level divergence in isolation overlooks cumulative prefix drift, granting the same divergence allowance regardless of how far the conditioning history has already deviated from the rollout policy. To address this limitation, we propose CPPO (Cumulative Prefix-divergence Policy Optimization), a tokenlevel masking rule that aligns updates with a finite-horizon policy-improvement bound via two coupled mechanisms. First, a position-weighted threshold imposes stricter limits at early positions whose effects persist longer, relaxing constraints for late-stage tokens. Second, a cumulative prefix budget tracks historical deviations, dynamically restricting further tokenlevel deviation to prevent compounding errors along the prefix. Empirically, CPPO enhances training stability and significantly improves reasoning accuracy across various model scales.

Correspondence: wenxizhu@tencent.com, ltao@pku.edu.cn, zhouxiangxin1998@gmail.com Project Page: https://hunyuan-cppo.github.io

|54.79| | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

###### CPPO

L(θ) = Ey∼µ |ty=1| Mtρt At Mt= 1 sgn( At)(ρt − 1) ≤ 0∨It

0.50

MinPRO

DPPO

0.40

TRM-Avg

PPO It=[ |ρt − 1| ≤ ] DPPO It=[ Dt ≤ δ ] CPPO It= wtDt ≤ δ∧ j≤twjDj

0.30

t|st) µ(yt|st)

ρt = π(y

TRM-Max

0.20

CISPO

###### Dt = D(π(·|st), µ(·|st))

j≤twj ≤ δb

GRPO

0.10

125 250 375 500

Training Step

Figure 1: Overview of CPPO. Left: token-level masking rules for PPO (Schulman et al., 2017), DPPO (Qi et al., 2026), and CPPO. Right: validation AIME24/25/26 Avg@16 on Qwen3-30B-A3BBase. Full validation curves for the Base-model runs are shown in Figure 4.

### 1 Introduction

Reinforcement learning has become a standard tool for LLM post-training, from preference-feedback alignment to verifier-driven reasoning (Ouyang et al., 2022; Rafailov et al., 2023; Shao et al., 2024; Guo et al., 2025; Yu et al., 2026; Liu et al., 2025). In reinforcement learning with verifiable rewards (RLVR), the policy generates responses, a verifier assigns a scalar reward, and the model is updated with a PPO/GRPO-style token-level objective (Schulman et al., 2017; Shao et al., 2024).

A practical RLVR update is off-policy. Each batch of responses is sampled from a fixed rollout policy µ and then reused for several gradient steps, so the policy being optimized, π, steadily moves away from the policy that generated the data. Unconstrained policy optimization often leads to unstable updates and degraded reasoning performance. Autoregressive generation further amplifies this divergence because early token deviations alter the conditioning of all subsequent steps.

To mitigate this drift, RLVR borrows the trust-region idea from classical policy optimization. Trust Region Policy Optimization (TRPO) constrains the divergence between successive policies and, in return, guarantees monotonic improvement (Schulman et al., 2015; Kakade and Langford, 2002; Achiam et al., 2017; Peters et al., 2010). Enforcing such a divergence exactly is expensive at LLM vocabulary scale, so PPO and GRPO instead clip the likelihood ratio of the sampled token (Schulman et al., 2017; Shao et al., 2024). For each token, this ratio is a single-sample Monte Carlo estimate of the true divergence between µ and π; built from one sampled token, it is noisy and has high variance (Wang et al., 2019b,a; Engstrom et al., 2020; Andrychowicz et al., 2021). It is also poorly calibrated across the long-tailed LLM vocabulary: a rare token can yield a large ratio and be over-penalized, while a high-probability token can move substantial probability mass under a ratio close to one and go under-penalized (Qi et al., 2026). DPPO (Qi et al., 2026) replaces this estimate with a direct measure of policy divergence. At each token it constrains the total-variation (TV) divergence Dt between the rollout policy µ and the target policy π, which reflects the actual change in the next-token distribution rather than a single sampled outcome.

While these methods refine the measurement of token-level policy divergence, they apply a uniform, position-agnostic threshold across all steps. This pointwise constraint conflicts with the autoregressive factorization of LLMs by overlooking two structural properties of the generation process: First, uniform thresholds ignore the autoregressive asymmetry in error propagation. Because early tokens condition the entire subsequent generation, an identical token-level divergence at earlier positions induces a larger sequence-level distribution shift. By assigning a static threshold across all positions, uniform constraints inherently underestimate early-stage deviations, while overly restricting late-stage exploration. Second, evaluating token-level divergence in isolation ignores cumulative prefix divergence. In LLM reinforcement learning, the state at step t is defined by the generated prefix st = (x,y<t). If individual token deviations accumulate along this prefix, the model is effectively generating from a highly off-policy state. When the conditioning prefix has already drifted significantly from the rollout policy, any further deviation at the current token carries compounding risk. A uniform threshold fails to account for this historical context, granting the exact same room for divergence regardless of how far the prefix has already drifted. Consequently, the trust region should dynamically tighten as off-policy drift accumulates along the prefix, with the permissible divergence threshold for subsequent tokens proportionally reduced to prevent sequence-level collapse.

To address this limitation, we propose CPPO (Cumulative Prefix-divergence Policy Optimization), a novel trust-region mechanism that directly aligns token-level updates with the autoregressive structure of LLMs. While recent advances focus on how token-level divergence is measured, CPPO

addresses where and how much policy deviation is permitted to accumulate along a trajectory. Specifically, rather than enforcing pointwise divergence limits in isolation, CPPO regularizes the policy update through two coupled constraints designed to mitigate the aforementioned structural mismatches. First, to account for autoregressive asymmetry, we introduce a position-weighted token-level threshold. This mechanism enforces relatively conservative divergence limits at early positions where deviations cascade into substantial sequence-level distribution shifts, while relaxing the constraints for late-stage tokens to preserve exploration. Second, to prevent compounding errors, we establish a cumulative prefix budget. By tracking and budgeting the weighted average of divergences along the generated prefix, CPPO dynamically restricts further updates once the historical context has drifted significantly from the rollout policy. Theoretically, this dual-constraint approach explicitly bounds the finite-horizon error propagation, yielding a provably tighter policy-improvement bound compared to uniform, position-agnostic thresholding. Empirically, this structured allocation of the divergence budget significantly enhances training stability and reasoning accuracy across various model scales. Figure 1 summarizes the token-level masking conditions and representative Qwen3-30B-A3B-Base validation curves.

This paper makes the following contributions:

- • We formalize how token position enters the finite-horizon error bound: early token-level policy shifts affect longer suffixes and contribute more, explaining why uniform token-level divergence thresholds are loose for long responses.
- • We introduce CPPO, a token-level mask that follows the prefix-to-suffix generation order and

- constrains token-level divergence with a position-weighted token-level constraint and a cumulative prefix budget.

- • We integrate CPPO as a drop-in token mask and evaluate it under matched RLVR settings. Across varying model sizes, CPPO obtains the best AIME24/25/26 average scores, with ablations supporting both constraints.

### 2 Preliminaries

Reinforcement learning for LLMs is a finite-horizon sequential decision problem. Given a prompt x ∼ D, a policy π generates a response y = (y1,...,yT) one token at a time. At step t the state is the prompt together with the tokens generated so far, st = (x,y<t), the action is the next token yt, and the policy defines a conditional distribution π(yt | st) over the vocabulary. The response probability factorizes autoregressively as π(y | x) = Tt=1 π(yt | st). After the full response is produced, a verifier returns a scalar reward R(x,y) with |R(x,y)| ≤ ξ, and the training objective is J(π) = Ex,y∼π[R(x,y)]. An RLVR update is off-policy: responses are drawn from a fixed rollout policy µ and reused to optimize a target policy π under common support.

Trust-region methods control how far the target policy π may move from the rollout policy µ. Trust Region Policy Optimization (TRPO) maximizes a surrogate objective subject to an explicit constraint on the policy divergence, which guarantees monotonic improvement as long as π stays close to µ (Schulman et al., 2015; Kakade and Langford, 2002; Achiam et al., 2017). The trust-region rules used in LLM RL are heuristic approximations of this constrained update: PPO and GRPO replace the divergence constraint with clipping of the sampled likelihood ratio (Schulman et al., 2017; Shao et al., 2024), DPPO replaces the sampled ratio with a direct measure of the token-level divergence (Qi et al., 2026), and TRM applies a similar divergence test at the sequence level (Li et al., 2025). This

section sets up the finite-horizon surrogate these methods share, the token-level trust-region rules they impose, and how the token-level divergence is computed in practice. Full proofs and extensions are in Appendix B.

- 2.1 Finite-horizon token-level surrogate We fix the rollout policy µ and optimize the target policy π under common support. Let

ρt :=

π(yt | st) µ(yt | st)

, ρa:b :=

b

j=a

ρj,

with ρT+1:T = 1. Following Qi et al. (2026), autoregressive factorization gives the exact finite-horizon performance difference identity (Lemma 2, proved in Appendix B)

J(π) − J(µ) = L′µ(π) − ∆(µ,π), (1) where

L′µ(π) := Eµ R(x,y)

T

t=1

(ρt − 1) ,

∆(µ,π) := Eµ R(x,y)

T

t=1

(ρt − 1)(1 − ρt+1:T) .

The factor ρt+1:T is the likelihood ratio of the future tokens yt+1:T under π relative to µ, conditioned on the prefix up to token t. Reverse telescoping gives the exact corrected objective Eµ[R(x,y) t(ρt− 1)ρt+1:T]. The token-level surrogate L′µ sets this suffix correction to one. Trust-region constraints are therefore tied to the surrogate itself because they control the approximation error |∆(µ,π)| induced by dropping the future likelihood-ratio correction.

- 2.2 From sampled-ratio clipping to token-level divergence

The clipped PPO-style objective used by PPO and GRPO is the standard practical implementation of the token-level surrogate. For token advantage Aˆt and clip range ϵ, the clipped token objective is

LPPOµ (π) = Eµ

T

min ρtAˆt, clip(ρt,1 − ϵ,1 + ϵ)Aˆt .

t=1

Equivalently, its one-sided clipping rule can be written as the sampled-ratio mask

MtPPO = 1 A ˆt(ρt − 1) ≤ 0 ∨ |ρt − 1| ≤ ϵ . (2)

This mask makes explicit the trust-region criterion implicit in the clipped token objective. The decision is made from the ratio of the sampled token; GRPO changes how the advantage is estimated, but uses the same signed clipping asymmetry.

The sampled ratio is a one-sample view of the token-level distributional change. For each state st,

DTV(µ(· | st),π(· | st)) =

- 1

- 2

Eyt∼µ(·|st)|ρt − 1|.

DPPO replaces the sampled-ratio test in Equation (2) with a direct measure of the token-level divergence Dt = D(µ(· | st),π(· | st)):

MtDPPO = 1 A ˆt(ρt − 1) ≤ 0 ∨ Dt ≤ δ . (3)

This change replaces the sampled-ratio trust-region criterion with a distributional next-token divergence criterion, but it still assigns a uniform token-level threshold δ across token positions.

#### 2.3 Token-level divergence approximation

A DPPO-style mask needs the token-level divergence Dt between the next-token distributions µ(· | st) and π(· | st), whose exact value is the total variation Dt = DTV(µ(· | st),π(· | st)). Evaluating this over the full vocabulary at every token is computationally prohibitive at LLM scale, so we follow the DPPO Top-K reduced-TV approximation (Qi et al., 2026) (K=20; construction in Appendix D). All token-level trust-region methods in our experiments use this same approximation and a per-model threshold scale, so that any difference between them reflects the trust-region rule rather than the divergence approximation. By Pinsker’s inequality DTV ≤ DKL/2, the construction and the bound in Section 3 cover a KL-based constraint as well.

#### 2.4 Limitations of uniform token-level thresholds

Although existing methods differ in their token-level divergence metrics, they universally apply a static threshold δ across all positions. This uniform constraint is globally suboptimal due to two properties of autoregressive generation.

First, a uniform threshold ignores how a token-level divergence propagates along the response. The surrogate error ∆(µ,π) that a trust region must control comes from the dropped suffix correction 1−ρt+1:T (Section 2.1), so a divergence introduced at token t enters the conditioning of every token sampled after it. Early deviations affect long suffixes, whereas late deviations have minimal downstream impact. A uniform threshold fails to account for this asymmetry.

Second, a uniform threshold ignores how divergence accumulates along the prefix. Because per-token deviations accumulate within the historical context st = (x,y<t), a sequence of locally bounded steps can still shift the sampling prefix far from the rollout policy distribution. A uniform token-level threshold provides a constant divergence budget regardless of prior prefix drift.

These two limitations motivate CPPO (Section 3). CPPO retains the standard token-level divergence but revises how the permitted divergence budget is distributed along the response via two novel mechanisms. To match the divergence allowed at a position to how far that divergence can propagate, CPPO replaces the uniform threshold with a position-weighted threshold that is tighter at early positions and looser at late ones. To bound accumulated divergence, CPPO introduces a cumulative prefix budget, which caps the weighted average of token-level divergences over every response prefix. Unlike a per-token threshold, this cap binds on the prefix as a whole, so once earlier tokens have driven the prefix average up, the divergence allowed at the next token is reduced accordingly. The trust region therefore tightens dynamically as the prefix drifts. The next section turns these two mechanisms into a concrete token-level mask.

### 3 CPPO

- Section 2.4 motivated two mechanisms for distributing the permitted divergence along a response: a position-weighted token-level threshold and a cumulative prefix budget. Before constructing the mask, we make precise the finite-horizon bound that these mechanisms are designed to improve.
- Section 3.1 quantifies how a token-level divergence propagates through the remaining response, and Section 3.2 states the resulting policy-improvement bound. Sections 3.3 and 3.4 then turn the bound into the CPPO token-level mask. Throughout, the policy loss keeps the usual token-level PPO-style ratio-advantage form with GRPO group-relative advantages in our experiments; the only change is the masking decision for update terms that move the sampled-token ratio farther from one.

#### 3.1 Autoregressive Asymmetry in Error Propagation

We first quantify the autoregressive asymmetry: how the position of a token-level divergence dictates its downstream impact on the surrogate residual ∆(µ,π) from Section 2.1. Let Dt := DTV(µ(· | st),π(· | st)) and ut := Est∼dµ

[Dt] be the expected token-level divergence at position t. Suppose a position-specific divergence limit Dt ≤ ℓt, is enforced. A maximal-coupling argument on the suffix likelihood ratio (Lemma 3) gives a bound on the surrogate residual (Proposition 4, proved in Appendix B):

t

T−1

|∆(µ,π)| ≤ 4ξ

ut

t=1

T

ℓj ≤

j=t+1

T−1

λtut, λt := 4ξℓ¯(T − t), ℓ¯:= max

ℓj, (4)

j

t=1

where ξ is the absolute bound on the reward (|R(x,y)| ≤ ξ). Equation (4) makes the first limitation from Section 2.4 quantitative. The coefficient λt = 4ξℓ¯(T − t) attached to the expected token-level divergence ut grows linearly with the remaining horizon T − t. In sequential decision making, this linear dependence formalizes the error propagation and compounding covariate shift inherent in autoregressive generation. An early policy deviation shifts the distribution of generated prefixes, skewing the conditioning context for all subsequent tokens and accumulating its surrogate residual error over the entire suffix. This reveals a fundamental asymmetry: early tokens act as critical branching points with long-term consequences, whereas late tokens have minimal downstream impact. A uniform token-level threshold Dt ≤ δ ignores this profile entirely. Consequently, it under-penalizes early deviations (which carry the largest error propagation multipliers) while overly restricting late-stage exploration, where divergence has minimal effect on future conditioning.

This autoregressive asymmetry, however, presents a structural opportunity: Because the error propagation multiplier λt scales linearly with the remaining horizon T − t, we introduce a monotonic position weight wt designed to mirror this exact trajectory. By aligning the position weight with the intrinsic error propagation profile, the divergence constraint naturally relaxes as generation progresses. This connection between the surrogate bound and the position weight is formalized below.

#### 3.2 Prefix-constrained improvement bound

CPPO constrains token-level divergence through weighted prefix averages. Let wt > 0 be a position weight, ct > 0 a weighted token-level divergence threshold, and δb > 0 a prefix-average threshold. Define

m

m

Pm :=

wjDj, Wm :=

wj.

j=1

j=1

The theorem uses the following token-level and prefix constraints along the rollout: wtDt ≤ ct (t = 1,...,T), Pm ≤ δbWm (m = 1,...,T − 1). (5)

The theorem is stated in weighted form because the implementation constrains wtDt directly. For a fixed threshold on wtDt, positions with larger wt allow a smaller value of Dt, while the prefix constraint limits the weighted average policy deviation on every prefix. Set ℓt := ct/wt and ℓ¯:= maxt ℓt.

Theorem 1 (CPPO policy-improvement bound). Suppose (5) holds µ-a.s. along the rollout. Let λt = 4ξℓ¯(T − t) and assume the ratio of the error propagation penalty to the position weight, rt := λt/wt, is non-increasing in t = 1,...,T − 1. Then

J(π) − J(µ) ≥ L′µ(π) − 2ξT(T − 1)ℓδ¯ b. (6) In particular, for a constant threshold ct ≡ δ, and a weight floor wmin where wt ∈ [wmin,1], then

δb wmin

J(π) − J(µ) ≥ L′µ(π) − 2ξT(T − 1)δ

. (7)

The proof is deferred to Appendix B; it combines the performance difference identity (Lemma 2) with the remaining-horizon residual bound (Proposition 4) and an Abel-summation step over the prefix constraints.

To facilitate comparison, we separate the current-token divergence term from the suffix-divergence terms in Equation (4). The coefficient λt = 4ξℓ¯(T −t) contains the suffix threshold ℓ¯, while ut = E[Dt] is the expected token-level divergence at position t:

|∆| ≤

λtut, λt = 4ξℓ¯(T − t).

t<T

A position-independent token-level divergence method with Dt ≤ δ controls both factors pointwise, ℓ¯≤ δ, ut ≤ δ, and therefore gives

T−1

|∆| ≤ 4ξδ2

(T − t) = 2ξT(T − 1)δ2.

t=1

CPPO keeps the same suffix-divergence factor ℓ¯. Its prefix constraints change how the expected tokenlevel divergences are bounded: instead of using the pointwise implication ut ≤ δ, Abel summation gives

λt = 2ξT(T − 1)ℓδ¯ b.

λtut ≤ δb

t<T

t<T

Thus the prefix constraints replace the pointwise token-level divergence factor δ by the prefix-average threshold δb.

Uniform token-level threshold. For the clean comparison, both methods use a uniform token-level threshold Dt ≤ δ (Corollary 6). This corresponds to setting ct = wtδ for CPPO, so ℓ¯= δ. The two residual constants are then

Cuniform = 2ξT(T − 1)δ2, CCPPO = 2ξT(T − 1)δδb,

and hence

CCPPO Cuniform

δb δ

=

.

This comparison fixes the threshold value δ across the two methods. The improvement comes from the prefix constraints, which prevent many early prefixes with large remaining-horizon coefficients from all reaching the token-level threshold. The bound improves when δb < δ.

Replacing the purely pointwise bound with the weighted prefix-average threshold δb directly mitigates worst-case error accumulation. A pointwise constraint permits the policy to saturate the divergence budget at every consecutive step. Because the state in autoregressive generation is the historical prefix st = (x,y<t), such worst-case accumulation drives the state visitation under π far from that under µ, which directly inflates the surrogate residual |∆(µ,π)|. Abel summation refactors the residual sum in terms of the prefix sums Pm. Consequently, constraining weighted prefix averages rather than pointwise divergences tightens the bound by precluding the worst-case accumulation patterns permitted by pointwise constraints. The factor δb/δ above quantifies how much of that pointwise looseness the prefix budget closes.

Position-dependent token-level threshold. The implementation in Section 3.3 constrains the weighted divergence by wtDt ≤ δ, which is equivalent to a position-dependent token-level threshold Dt ≤ δ/wt: the threshold equals δ at the beginning of the response and is relaxed toward the end (Corollary 7). Since wt ≥ wmin,

δ wt ≤

δ wmin

ℓ¯= max

. Combining this with the prefix part of the theorem gives

t

δ wmin

CCPPO ≤ 2ξT(T − 1)

δb,

CCPPO Cuniform ≤

δb δwmin

.

The factor 1/wmin is the cost of relaxing late-token thresholds. Under this implementation parameterization, the bound is tighter than the position-independent threshold Dt ≤ δ when δb < δ wmin.

From theoretical bounds to trust-region mechanisms. Theorem 1 establishes that tightening the surrogate residual requires transitioning from a loose, position-agnostic dependence on δ to a stricter bound governed by the weighted prefix-average threshold δb. This transition requires two structural modifications to the policy update. First, the remaining-horizon coefficient (λt ∝ T − t) formally captures autoregressive asymmetry. Early policy deviations propagate over longer suffixes and therefore demand tighter regulation. We address this by introducing a monotonically decreasing position weight wt, which enforces stricter limits initially and naturally relaxes them as generation progresses. Second, the analytical reliance on δb demonstrates that policy drift must be constrained cumulatively across the conditioning history rather than evaluating next-token divergences in isolation. We implement this via a dynamic cumulative prefix budget. The following subsections translate these theoretical properties into the concrete token-level masking rule implemented by CPPO.

#### 3.3 Prefix constraints

Theorem 1 requires the cumulative budget Pm ≤ δbWm to hold at every intermediate prefix (m = 1,...,T − 1), not only at the full response. Enforcing this average bound exclusively at the sequence terminal (e.g., TRM-Avg) creates a loose constraint. It permits excessive deviations at early positions, provided later tokens mathematically offset them to satisfy the overall response average. To enforce the prefix constraint at every m during training, when only the prefix produced so far is available

at token t, we turn the cumulative requirement into a single threshold on the weighted token-level divergence wtDt that depends only on the preceding tokens.

Recall the token-level divergence Dt from Section 2.3. Define the weighted prefix divergence and cumulative weight by

t

St :=

wjDj, Wt :=

j=1

t

wj, S0 = W0 := 0,

j=1

where St is the running form of the prefix sum Pt in Theorem 1. Given a token-level threshold scale δ, a prefix-average threshold δb (the maximum weighted average allowed over each prefix), and positive weights {wt}Tt=1, CPPO keeps an update term that moves the sampled-token ratio farther from one only when

wtDt ≤ cCPPOt := min{δ, δ + δbWt−1 − St−1}, t = 1,...,T. (8) This effective threshold enforces the weighted token-level condition wtDt ≤ δ together with the cumulative prefix condition St ≤ δ+δbWt−1. The extra δ term gives the first token the full token-level threshold prior to any prefix accumulation. At t = 1, W0 = S0 = 0 and the effective prefix threshold is δ. For later tokens, the effective prefix threshold is above the token-level threshold when the preceding weighted average St−1/Wt−1 is below δb, so the token-level threshold remains active. Once preceding deviations make this prefix average exceed δb, the prefix-adjusted term δ + δbWt−1 − St−1 falls below δ and reduces the allowed weighted divergence at the current token. The effective threshold therefore tightens dynamically as prefix divergence accumulates, instead of granting every token the same allowance.

The implementation uses the decreasing linear schedule

1 − wmin T − 1

(t − 1), t = 1,...,T, wt ∈ [wmin,1]. (9)

wt = 1 −

Because the constraint is applied to wtDt, the allowed token-level divergence at position t is Dt ≤ δ/wt. This threshold starts at δ at the beginning of the response and relaxes to δ/wmin near the end. The weights parameterize the token-level and prefix constraints rather than defining a new divergence measure or loss term. Early positions therefore allow smaller values of Dt, while later positions are relaxed where the remaining suffix is shorter. The schedule satisfies the monotonicity condition required by Theorem 1, namely that (T − t)/wt is non-increasing in t (Proposition 5 in Appendix B). The scalar δ controls the token-level threshold scale, and δb controls the weighted average allowed at every prefix.

- Figure 2 places this schedule next to an offline diagnostic from Qwen3-30B-A3B rollouts after 200

GRPO updates. For each token position, we compute |πθ(at | st) − µ(at | st)|, where πθ is the post-update training policy and µ is the rollout policy. The mean and top-10% tail views show that policy deviation varies with token position. The right panel shows the induced token-level threshold δ/wt used by CPPO.

- Figure 3 illustrates Equation (8) on a schematic token window. The orange curve is the prefix-

adjusted threshold δ + δbWt−1 − St−1, and the green curve is the effective threshold after taking the minimum with the token-level threshold δ. The blue shaded region marks tokens for which the token-level threshold is active; the orange shaded region marks tokens for which the prefix constraint is active. Because the implementation includes the initial prefix slack in Equation (8), the effective threshold starts at δ. Low initial weighted deviations keep the token-level threshold active, while larger accumulated deviations later reduce the effective threshold below δ. A token can then satisfy

###### Mean of | |

0.020

0.015

0.010

0 2000 4000 6000 8000

Token Position T

###### Top 10% of | |

0.20

0.15

0.10

0.05

0 2000 4000 6000 8000

Token Position T

Equivalent Threshold

| | | | | |
|---|---|---|---|---|
| |y =|y<br><br>y = 0 0.20/wt|= 0.20<br><br>.25| |
| | | | | |

δ/wmin

δ

0 2000 4000 6000 8000

Token Position T

- Figure 2: Position-conditioned policy deviation and the position-dependent token-level threshold. Left: maximum of bin-level mean absolute probability deviation across token positions. Middle: the corresponding top-10% tail view. Right: the equivalent token-level threshold δ/wt induced by the decreasing wt schedule.

the token-level threshold but exceed the effective threshold, in which case it is masked by the prefix constraint.

The initial slack is an implementation detail rather than part of the formal theoretical bound. It changes the prefix inequality from St ≤ δbWt to St ≤ δ+δbWt−1 ≤ δbWt+δ, adding only a lower-order O(Tℓδ¯ ) term to the Abel bound in Theorem 1; Proposition 8 in Appendix B gives the calculation.

0 T

Token position

δb

δ

same at t = 0

token-level threshold active preﬁx constraint active

avg. deviation < δb

token-level threshold δ

preﬁx-adjusted threshold

effective threshold

preﬁx-average threshold δb

··· ···

masked by token-level threshold

masked by preﬁx constraint

- Figure 3: Cumulative prefix constraint on a token window. Grey bars are simulated weighted

deviations wtDt. The blue shaded region denotes tokens where the token-level threshold δ is active; the orange shaded region denotes tokens where the prefix constraint determines the effective threshold. The blue bar is masked by the token-level threshold; the orange bar satisfies the token-level threshold but violates the effective threshold.

#### 3.4 Token-level mask and surrogate objective

The feasibility test of Equation (8) collects the position-dependent token-level threshold and the cumulative prefix constraint into a single per-token condition It. An update term is kept whenever it

moves the sampled-token ratio toward one, and otherwise only when It holds: MtCPPO = 1 A ˆt(ρt − 1) ≤ 0 ∨ It , It : wtDt ≤ δ ∧ St ≤ δ + δbWt−1. (10)

The first clause preserves update terms that move π closer to µ. The indicator It exclusively restricts terms that drive π further from µ, simultaneously enforcing the token-level and prefix constraints. Substituting this mask into the token-level ratio-advantage objective gives the CPPO surrogate

LCPPOµ (π) = Eµ

T

MtCPPO ρt Aˆt , (11)

t=1

with the trust-region decision carried entirely by MtCPPO (we use GRPO group-relative advantages for Aˆt in our experiments). When the prefix has not yet drifted, It reduces to the position-dependent token-level threshold; once accumulated deviation drives the prefix average toward δb, the prefix

- constraint becomes the binding term and tightens the mask.

Algorithm 1 writes the mask for one sampled response in the same order in which the tokens appear. The running sums St and Wt are ordinary prefix sums: after token t, they store the total weighted divergence and total weight up to that token. The threshold at token t uses St−1 and Wt−1, so the current decision depends only on the preceding prefix and the current weighted divergence Zt = wtDt. A batched tensor implementation computes the same quantities with a cumulative sum over the response dimension and a one-token right shift; the loop below is written for readability rather than for kernel efficiency. The soft-gate variant is deferred to Appendix C.

Algorithm 1 Token-level CPPO mask for one response Require: ratios ρ1:T, advantages Aˆ1:T, token-level divergences D1:T, thresholds δ,δb, weight floor wmin. Ensure: token mask M1:T.

- 1: Initialize S0 ← 0 and W0 ← 0.
- 2: for t = 1,...,T do
- 3: Set wt ← 1 − 1−Tw−min1 (t − 1) and Zt ← wtDt.

- 4: Compute the effective threshold ct ← min{δ,δ + δbWt−1 − St−1}.
- 5: if Aˆt(ρt − 1) ≤ 0 or Zt ≤ ct then
- 6: Mt ← 1.
- 7: else
- 8: Mt ← 0.
- 9: Update prefix sums St ← St−1 + Zt and Wt ← Wt−1 + wt.
- 10: return M1:T.

### 4 Experiments

The experiments test whether prefix-budgeted masking improves reasoning RL under matched data, rollout lengths, and validation selection windows. The baselines include ratio-based masks, prefix-ratio objectives, divergence-based token masks, and sequence-level trust-region masks. We first report results across four Qwen3 settings, then use ablations to test which CPPO constraints drive the gain.

#### 4.1 Setup

All runs train on DAPO-Math-17k (Yu et al., 2026), a set of roughly 17k verifiable mathematical reasoning prompts, using the verl (Sheng et al., 2025) GRPO/DAPO rollout stack. We evaluate four Qwen3 (Yang et al., 2025) model settings: Qwen3-1.7B-Base, Qwen3-1.7B (post-trained), Qwen3-8B-Base, and Qwen3-30B-A3B-Base. The two 1.7B settings and the 8B-Base run use Tmax = 8k with n = 8 rollouts. The 30B-A3B-Base run uses 16k with n = 16 rollouts. Validation uses Avg@16 on AIME24, AIME25, and AIME26, and the main score is their unweighted mean, AIME24/25/26 Avg@16. For each method, we report the best validation score within the matched evaluation horizon [0,Tstop] for that model, so no method is selected after receiving additional training budget. Full training details, per-model update budgets, and per-benchmark breakdowns are in Appendix D.

The baselines are grouped by the trust-region rule they use. GRPO (Shao et al., 2024) and CISPO (Chen et al., 2025) operate on sampled-token ratios, with CISPO using asymmetric clip thresholds. MinPRO (Lei et al., 2026) adds a prefix-ratio surrogate without a cumulative distributional budget. TRM-Max and TRM-Avg (Li et al., 2025) summarize each completed response with a sequence-level KL criterion. The GRPO row uses the Clip-Higher setting (ϵlow,ϵhigh) = (0.2,0.28). The other baseline hyperparameters follow their original papers. The closest baseline is DPPO (Qi et al., 2026). DPPO and CPPO use a matched Top-K (K=20) reduced-TV score and per-model token-level threshold scale. DPPO applies this scale as a uniform token-level threshold, while CPPO uses it inside the weighted and prefix constraints.

CPPO adds two hyperparameters on top of this shared threshold scale, both defined in Section 3. The token-level threshold scale δ is the per-token divergence allowed at the start of a response, exactly as in DPPO. The prefix-average threshold δb is the largest weighted average of the token-level divergences permitted over any prefix of the response (the St ≤ δbWt part of Equation (8)); it controls how much accumulated deviation a prefix may carry, not a per-token quantity. The weight floor wmin sets the smallest position weight, so the linear schedule wt ∈ [wmin,1] runs from a full-strength constraint on the first token to its loosest setting on the last; a smaller wmin relaxes late tokens more, matching the shorter remaining horizon there. We set the threshold scale to δ = 0.15 for the dense models and δ = 0.2 for the 30B-A3B MoE model. CPPO uses wmin = 0.8 throughout. For the post-trained reasoning model, the average token-level divergence remains stable, so we use a fixed δb = 0.015. In contrast, during the initial exploration phase of Base-model training, the average token-level divergence is exceptionally large before rapidly decaying as the policy stabilizes. To avoid excessively clipping these exploratory tokens before training stabilizes, we set δb adaptively based on each sequence’s divergence statistics on the Base models. Specifically, for each generated sequence, we compute the top-10% quantile (the 90th percentile) of its token-level divergences as the raw δb, and then clamp this value between a minimum threshold δbmin (listed in Table 3) and an upper bound of 2δbmin. Appendix Figure 7 reports the realized effective δb values and the fraction of masked tokens rejected by the prefix-budget condition.

#### 4.2 Main results

Table 1 reports the best validation AIME24/25/26 Avg@16 reached within the same [0,Tstop] window for each model setting. This selection rule prevents a method from benefiting from a longer training budget on the same model.

- Table 1: Best validation AIME24/25/26 Avg@16 across the four models (%, higher is better). For each method we report the highest AIME24/25/26 Avg@16 within the matched training-step window [0,Tstop]. The best score in each column is in bold and the second-best is underlined. Per-benchmark breakdowns and Tstop are in Table 4 of Appendix D.

Method 1.7B 1.7B-Base 8B-Base 30B-A3B-Base

GRPO 27.91 8.89 23.96 38.19 MinPRO 27.71 11.04 29.72 48.12 CISPO 28.82 11.87 29.58 collapse DPPO 28.19 10.90 28.89 49.23 TRM-Max 25.21 9.72 26.73 20.27 TRM-Avg 26.87 11.70 27.98 48.96

CPPO (ours) 31.88 12.78 31.11 54.79

CPPO consistently outperform all the baselines across all settings by a significant margin. Specifically, CPPO attains the best AIME24/25/26 Avg@16 in all settings, reaching 31.88, 12.78, 31.11, and 54.79 on Qwen3-1.7B, Qwen3-1.7B-Base, Qwen3-8B-Base, and Qwen3-30B-A3B-Base, respectively. The margins over the second-best method are 3.06, 0.91, 1.39, and 5.56 absolute points. The performance of the baselines varies across models. CISPO attains the second highest validation performance on the 1.7B models, while MinPRO and DPPO rank second on 8B-Base and 30B-A3B-Base respectively.

The comparison with DPPO is strictly controlled, as the two methods share the same Top-K reduced-TV score and the same per-model threshold scale δ and differ only in the weighted and prefix constraints. Under this matched setup CPPO improves on DPPO by 3.69, 1.88, 2.22, and

- 5.56 points across the four models. The gain is thus attributable to how the token-level divergence is allocated along the response, and not to a different divergence measure or a looser threshold scale.

The largest improvement, 5.56 points, occurs on Qwen3-30B-A3B-Base, the largest model and the only run with a 16k rollout horizon. This is the regime where the remaining-horizon amplification of Section 3.1 is most pronounced, since an early-token deviation there propagates through a longer suffix. The same setting separates stable from unstable trust-region rules: CISPO collapses partway through training and is omitted from this column, and the sequence-level TRM-Max degrades to 20.27, whereas CPPO trains stably to the selected checkpoint. The per-benchmark AIME24/25/26 components at the selected checkpoints are reported in Table 4.

- Figure 4 shows the validation AIME24/25/26 Avg@16 trajectories for the three Base-model runs. CPPO consistently maintains a performance advantage over the baselines throughout training, confirming that the prefix-constrained trust region yields stable and sustained performance gains. The separation from DPPO widens as training proceeds, consistent with the prefix constraint tightening only after policy deviation has accumulated over a prefix.

#### 4.3 Ablations

We isolate the contributions of CPPO’s components by incrementally modifying the matched DPPO baseline on Qwen3-1.7B and Qwen3-1.7B-Base.

1.7B-Base

8B-Base

- 0.0
- 0.1
- 0.2
- 0.3
- 0.4
- 0.5
- 0.6 A3B-Base

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.30

AIME24/25/26Avg@16

0.12

0.10

0.25

0.08

0.20

0.06

0.04

0.15

0.02

0.10

150 300 450 600

100 200 300 400

125 250 375 500

Training Step

Training Step

Training Step

CPPO (Ours) DPPO TRM-Avg TRM-Max CISPO GRPO MinPRO

Figure 4: Validation AIME24/25/26 Avg@16 curves for the three Base-model runs.

Single mechanism ablation. The left panel of Figure 5 separates the two constraints inside CPPO on Qwen3-1.7B against the matched DPPO baseline, which uses a uniform token-level threshold with no prefix constraint. CPPO w/o Position Weight adds the prefix constraint with wt ≡ 1, and CPPO w/o Prefix Budget uses wtDt ≤ δ without the prefix constraint. Both single-constraint variants outperform DPPO, while the complete CPPO mask achieves the highest validation scores. This indicates that the position weights and the cumulative prefix budget provide independent and complementary performance gains.

Position-weight ordering. The middle panel evaluates whether the performance improvements of the position-dependent threshold stem from the autoregressive order itself or from threshold heterogeneity. The shuffled variant keeps the same multiset of position-dependent token-level thresholds implied by {w1,...,wT} within each rollout but randomly reassigns them to token positions. The ordered schedule outperforms the shuffled variant, confirming that the autoregressive position order, rather than threshold heterogeneity, drives the performance gain.

Mask vs. soft gate. The right panel compares the full CPPO hard mask with a soft variant, which attenuates a token’s gradient near the constraint boundary with a non-increasing gate instead of dropping it, following SAPO (Gao et al., 2025); the gate construction is in Appendix C. The soft variant stays in the same range as the hard mask, so we keep the hard mask as the default and treat soft gating as an implementation choice.

Single Mechanism Ablation

Position-Weight Ordering

Mask vs. Soft Gate

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | |C w<br><br>|PPO w/o P / Shuffled|refix Bud Weights|get|
| | |D|PPO-TV| | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | |CPPO (Fu CPPO (S|ll) oft)|
| | | |DPPO-TV| |

0.30

0.30

0.30

AIME24/25/26Avg@16

0.25

0.25

0.25

0.20

0.20

0.20

CPPO (Full)

0.15

0.15

0.15

CPPO w/o Position Weight

CPPO w/o Prefix Budget

0.10

0.10

0.10

DPPO-TV

250 500 750 1000

250 500 750 1000

250 500 750 1000

Training Step

Training Step

Training Step

##### Figure 5: Ablations on Qwen3-1.7B.

Hyperparameter sensitivity. The left panel of Figure 6 sweeps the minimum prefix-average threshold and the weight floor around the default (δb,wmin) = (0.02,0.8) on Qwen3-1.7B-Base. The neighboring settings stay close to the default and above DPPO, so the gain is not tied to a single operating point.

KL vs. TV divergence. The middle panel replaces the Top-K reduced-TV score with a KL score while keeping the CPPO prefix constraints fixed, using the per-token and prefix-average thresholds of TRM (δ = 0.1 and δb = 0.002). The KL configuration yields performance comparable to the TV configuration and consistently outperforms DPPO. This demonstrates that the improvements of CPPO are robust to the choice of divergence metric. As a control, the TRM Max&Avg curve applies these same two thresholds through TRM’s sequence-level masks, which keep or drop an entire response by its maximum and mean token-level KL rather than by a cumulative prefix budget. This variant matches DPPO and stays below CPPO, indicating that the gain comes from enforcing the thresholds as a prefix budget rather than from the threshold values.

Binary vs. Top-K approximation. The right panel replaces the Top-K reduced-TV score with the simpler Binary-TV partition used by DPPO. Both configurations maintain similar validation scores and consistently outperform DPPO. This indicates the performance improvement is robust to the choice of divergence metric and its approximation granularity, aligning with the DPPO ablation of Qi et al. (2026) that finds Binary and Top-K estimators yield comparable results. The prefix budget, not the divergence estimator, drives the improvement.

0.14 δb, wmin Sensitivity

0.14 Binary vs. TopK

0.14 KL vs. TV

AIME24/25/26Avg@16

0.12

0.12

0.12

0.10

0.10

0.10

0.08

0.08

0.08

0.06

0.06

0.06

- CPPO (δb = 0.02, wmin = 0.8)

- CPPO (δb = 0.03, wmin = 0.8)

CPPO (TV) CPPO (KL) TRM Max&Avg

0.04

0.04

0.04

CPPO (Top-K TV)

CPPO (δb = 0.02, wmin = 0.6)

0.02

0.02

0.02

CPPO (Binary TV)

DPPO-TV

DPPO-TV

DPPO-TV

0.00

0.00

0.00

150 300 450 600

150 300 450 600

150 300 450 600

Training Step

Training Step

Training Step

Figure 6: Qwen3-1.7B-Base ablations.

### 5 Conclusion

This work revisits the token-level trust region used in reasoning RL. A uniform token-level threshold applies an identical constraint across all positions. This ignores the autoregressive nature of generation, where early deviations affect longer suffixes and per-token errors accumulate within the conditioning prefix. Starting from the finite-horizon performance difference identity, we derive a prefix-constrained policy-improvement bound that makes both effects explicit and turn it into CPPO, a token-level mask with two mechanisms: a decreasing position weight that makes the threshold tighter at early positions, and a cumulative prefix budget that reduces the divergence allowed at a later update once preceding deviations have driven the weighted prefix average up.

Both mechanisms act through the masking decision alone, so CPPO reuses the PPO/GRPO ratioadvantage objective and the same per-token divergence as DPPO, introducing no additional loss

terms. Across four Qwen3 settings spanning dense and MoE models and Base and post-trained checkpoints, CPPO attains the best validation AIME24/25/26 Avg@16, and the ablations attribute the gain to the position weight and the prefix budget rather than to the divergence estimator.

### Acknowledgement

We are grateful to all the members of the Hunyuan Multimodal Foundational RL Team for their support. We especially thank Tianyu Pang for his invaluable advice and feedback on this work.

### References

Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Rémi Munos, Nicolas Heess, and Martin A. Riedmiller. Maximum a posteriori policy optimisation. CoRR, abs/1806.06920, 2018. URL http://arxiv.org/abs/1806.06920.

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International conference on machine learning, pages 22–31. Pmlr, 2017.

Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphaël Marinier, Leonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, Sylvain Gelly, and Olivier Bachem. What matters for on-policy deep actor-critic methods? a large-scale study. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=nIAx jsniDzg.

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, Chengjun Xiao, Chengyu Du, Chi Zhang, Chu Qiao, Chunhao Zhang, Chunhui Du, Congchao Guo, Da Chen, Deming Ding, Dianjun Sun, Dong Li, Enwei Jiao, Haigang Zhou, Haimo Zhang, Han Ding, Haohai Sun, Haoyu Feng, Huaiguang Cai, Haichao Zhu, Jian Sun, Jiaqi Zhuang, Jiaren Cai, Jiayuan Song, Jin Zhu, Jingyang Li, Jinhao Tian, Jinli Liu, Junhao Xu, Junjie Yan, Junteng Liu, Junxian He, Kaiyi Feng, Ke Yang, Kecheng Xiao, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Li, Lin Zheng, Linge Du, Lingyu Yang, Lunbin Zeng, Minghui Yu, Mingliang Tao, Mingyuan Chi, Mozhi Zhang, Mujie Lin, Nan Hu, Nongyu Di, Peng Gao, Pengfei Li, Pengyu Zhao, Qibing Ren, Qidi Xu, Qile Li, Qin Wang, Rong Tian, Ruitao Leng, Shaoxiang Chen, Shaoyu Chen, Shengmin Shi, Shitong Weng, Shuchang Guan, Shuqi Yu, Sichen Li, Songquan Zhu, Tengfei Li, Tianchi Cai, Tianrun Liang, Weiyu Cheng, Weize Kong, Wenkai Li, Xiancai Chen, Xiangjun Song, Xiao Luo, Xiao Su, Xiaobo Li, Xiaodong Han, Xinzhu Hou, Xuan Lu, Xun Zou, Xuyang Shen, Yan Gong, Yan Ma, Yang Wang, Yiqi Shi, Yiran Zhong, and Yonghong Duan. Minimax-m1: Scaling test-time compute efficiently with lightning attention. CoRR, abs/2506.13585, June 2025. URL https://doi.org/10.48550/arXiv.2506.13585.

Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. CoRR, abs/2504.02546, April 2025. URL https://doi.org/10.48550/arXiv.2504.02546.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, LEI BAI, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models, 2025. URL https://openreview.net/forum?id=vXoksdcfqC.

Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry. Implementation matters in deep rl: A case study on ppo and trpo. In International Conference on Learning Representations, 2020. URL https://openreview.net/for um?id=r1etN1rtPB.

Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, and Junyang Lin. Soft adaptive policy optimization. arXiv preprint arXiv:2511.20347, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong

Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, Hao Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, Tao Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nat., 645(8081):633–638, 2025. URL https://doi.org/10.1038/s41586-025-09422-z.

Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. Reinforce++: Stabilizing critic-free policy optimization with global advantage normalization. arXiv preprint arXiv:2501.03262, 2025.

Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In

Proceedings of the nineteenth international conference on machine learning, pages 267–274, 2002. Shiye Lei, Zhihao Cheng, and Dacheng Tao. A step back: Prefix importance ratio stabilizes policy

optimization. arXiv preprint arXiv:2601.22718, 2026. Yingru Li, Jiacai Liu, Jiawei Xu, Yuxuan Tong, Ziniu Li, Qian Liu, and Baoxiang Wang. Trust region masking for long-horizon llm reinforcement learning. arXiv preprint arXiv:2512.23075, 2025.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=5PAF7PAY2Y.

Chiyu Ma, Shuo Yang, Kexin Huang, Jinda Lu, Haoming Meng, Shangshang Wang, Bolin Ding, Soroush Vosoughi, Guoyin Wang, and Jingren Zhou. Fipo: Eliciting deep reasoning with future-kl influenced policy optimization. arXiv preprint arXiv:2603.19835, 2026.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike,

and Ryan Lowe. Training language models to follow instructions with human feedback. CoRR, abs/2203.02155, 2022. URL https://doi.org/10.48550/arXiv.2203.02155.

Jan Peters, Katharina Mulling, and Yasemin Altun. Relative entropy policy search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 24, pages 1607–1612, 2010.

Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, and Wee Sun Lee. Rethinking the trust region in llm reinforcement learning. arXiv preprint arXiv:2602.04879, 2026.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. URL https://doi.org/10.48550/arXiv.2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, page 1279–1297, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400711961. doi: 10.1145/3689031.3696075. URL https://doi.org/10.1145/3689031.3696075.

H. Francis Song, Abbas Abdolmaleki, Jost Tobias Springenberg, Aidan Clark, Hubert Soyer, Jack W. Rae, Seb Noury, Arun Ahuja, Siqi Liu, Dhruva Tirumala, Nicolas Heess, Dan Belov, Martin A. Riedmiller, and Matthew M. Botvinick. V-mpo: On-policy maximum a posteriori policy optimization for discrete and continuous control. CoRR, abs/1909.12238, 2019. URL http://arxiv.org/abs/1909.12238.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. CoRR, abs/2506.01939, June 2025. URL https://doi.org/10.48550/arXiv.2506.01939.

Yuhui Wang, Hao He, and Xiaoyang Tan. Truly proximal policy optimization. In UAI, pages 113–122, 2019a. URL http://proceedings.mlr.press/v115/wang20b.html.

Yuhui Wang, Hao He, Xiaoyang Tan, and Yaozhong Gan. Trust region-guided proximal policy optimization. CoRR, abs/1901.10314, 2019b. URL http://arxiv.org/abs/1901.10314.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng,

Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. CoRR, abs/2505.09388, May 2025. URL https://doi.org/10.48550/arXiv.2505.09388.

Deokgyu Yoon, Hyungkyu Kang, Joongkyu Lee, Byeongchan Kim, Gyungin Shin, Sungrae Park, and Min-hwan Oh. Multi-step likelihood-ratio correction for reinforcement learning with verifiable rewards. arXiv preprint arXiv:2605.20865, 2026.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, YuYue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/for um?id=2a36EMSSTp.

Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, Fang Wan, and Furu Wei. Geometric-mean policy optimization. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=nCEs0tSwc2.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. CoRR, abs/2507.18071, July 2025. URL https://doi.org/10.48550/arXiv.2507.18071.

### A Related work

CPPO is a trust-region method for RLVR. We organize related work by the statistic each method uses to constrain policy movement; CPPO’s contribution is prefix-budgeted, position-aware token masking under a fixed token-level divergence statistic, not a new way to measure that divergence.

Sampled-ratio methods. PPO-style methods (Schulman et al., 2017) restrict updates through the sampled-token importance ratio ρt. GRPO (Shao et al., 2024) adapts this template to LLMs with group-relative advantages from a verifier; Dr.GRPO (Liu et al., 2025) and REINFORCE++ (Hu et al., 2025) change the advantage normalization or the policy-gradient estimator while keeping the sampled-ratio update. DAPO (Yu et al., 2026) relaxes the upper clip range to preserve exploratory high-ratio updates; CISPO (Chen et al., 2025) clips the importance-sampling weight rather than the token update itself, preserving gradients from low-probability tokens that token-level clipping would otherwise suppress. GSPO (Zheng et al., 2025) extends the importance ratio from tokens to sequences with sequence-level clipping; GMPO (Zhao et al., 2026) replaces the arithmetic mean of token ratios in the surrogate with a geometric mean, reducing the influence of outlier tokens. MinPRO (Lei et al., 2026) substitutes a non-cumulative minimum-prefix-ratio surrogate for the unstable cumulative prefix ratio. GPG (Chu et al., 2025) drops the surrogate and the ratio-clip test altogether for a direct policy-gradient estimator, outside the trust-region family. Apart from GPG, these methods control the update through one or more sampled importance ratios, which only provides a single-sample estimate of the policy shift.

Distributional-divergence trust regions. A second line of work replaces the sampled-ratio test with a token-level distributional divergence Dt. DPPO (Qi et al., 2026) uses TV or KL as Dt via Binary or Top-K reduced approximations and applies a uniform threshold across token positions. TRM (Li et al., 2025) applies a sequence-level mask: if the maximum token divergence (TRM-Max) or the response-mean divergence (TRM-Avg) exceeds a threshold, the entire response is excluded from the update. Classical trust-region methods (Kakade and Langford, 2002; Schulman et al., 2015; Achiam et al., 2017; Peters et al., 2010; Abdolmaleki et al., 2018; Song et al., 2019) constrain or regularize KL in classical RL settings, providing the trust-region/relative-entropy lineage rather than LLM token-level divergence-budget methods. Trust-region-guided PPO variants further modify the clip threshold or rollback behavior using divergence information (Wang et al., 2019b,a); separate empirical analyses document how PPO/TRPO implementation choices and clip thresholds affect performance (Engstrom et al., 2020; Andrychowicz et al., 2021). These methods change how the trust region measures policy movement; CPPO addresses a separate question of where that movement is allowed to accumulate. Given a fixed token-level divergence, CPPO weights early positions more strongly through wt and caps the weighted prefix average through δb, rather than introducing a new measurement of token-level shift. Their feasibility constraints map onto specializations of CPPO: TRM-Max to the uniform-threshold sequence-level specialization without a prefix budget (wt ≡ 1, no δb), and TRM-Avg to a full-response-average specialization of CPPO’s prefix constraints that is weaker by up to a factor 2 − 2/T at wt ≡ 1. Algorithmically, however, TRM masks whole sequences while CPPO masks individual token updates. In our experiments, DPPO and CPPO use a matched Top-K reduced-TV statistic and per-model threshold scale, so the empirical comparison isolates the prefix constraints from the divergence statistic.

Complementary directions. Several recent methods modify the policy objective in orthogonal dimensions and are complementary to CPPO. NFPO (Yoon et al., 2026) and FIPO (Ma et al., 2026) change the weight attached to each token update through future-dependent likelihood-ratio or KL signals, whereas CPPO keeps the token surrogate and changes how policy deviation may

accumulate across prefixes. Entropy-based analyses identify high-entropy minority tokens that act as reasoning forks (Wang et al., 2025) and study how policy entropy evolves to prevent collapse during RLVR training (Cui et al., 2025); in contrast, CPPO addresses the distinct issue of how the allowed deviation should be allocated across positions and prefixes. Soft-masking methods (Gao et al., 2025) replace hard violation indicators with non-increasing attenuation, and CPPO’s violation score xCPPOt admits the same soft variant (Appendix C).

- Table 2: Trust-region methods grouped by the statistic they use and where the constraint is applied. Here token-level TV/KL denotes the divergence between rollout and target next-token distributions at a sampled prefix. In the matched comparison, DPPO and CPPO use a matched Top-K reduced-TV statistic; prefix-ratio objectives such as MinPRO are not divergence-budget methods.

Method class Statistic used by the rule Constraint applied PPO/GRPO sampled-token ratio ρt clips sampled ratios in the surrogate;

no distributional divergence budget DPPO token-level divergence Dt uses one divergence threshold;

shared across token positions TRM-Max response maximum maxt Dt constrains the largest token divergence; computed over the full response TRM-Avg response average T−1 t Dt constrains the average divergence;

computed over the full response CPPO (ours) token-level divergence Dt

token-level threshold + prefix-average constraints; position weights wt vary the effective threshold

with prefix sums

### B Full Proofs and Theoretical Details

This appendix provides the full proofs of the results stated in Section 2, together with the corollaries that characterize CPPO’s relationship to position-agnostic and sequence-level methods, and the product-form suffix bound.

- B.1 Finite-horizon performance difference identity The following lemma is the exact identity stated as Equation (1) in Section 2.1.

- Lemma 2 (Performance difference identity). Under common support, let ρa:b := bj=a ρj and ρT+1:T := 1. Then

J(π) − J(µ) = L′µ(π) − ∆(µ,π),

T

L′µ(π) := Eµ R(x,y)

(ρt − 1) ,

(12)

t=1

T

∆(µ,π) := Eµ R(x,y)

(ρt − 1)(1 − ρt+1:T) .

t=1

Hence J(π) − J(µ) ≥ L′µ(π) − |∆(µ,π)|. Proof By importance sampling,

J(π) − J(µ) = Eµ[R(x,y)(ρ1:T − 1)].

The reverse telescoping identity

then gives

T

ρ1:T − 1 =

(ρt − 1)ρt+1:T

t=1

J(π) − J(µ) = Eµ R(x,y)

= Eµ R(x,y)

T

(ρt − 1)ρt+1:T

t=1

T

(ρt − 1) − Eµ R(x,y)

t=1

T

(ρt − 1)(1 − ρt+1:T) .

t=1

The two terms on the last line are exactly L′µ(π) and ∆(µ,π). The lower bound follows from −∆(µ,π) ≥ −|∆(µ,π)|.

#### B.2 Suffix TV via maximal coupling

- Lemma 3 (Suffix TV bound). If Dj ≤ ℓj pathwise for all j > t, then for any st+1,

T

T

DTV(Pµt+1:T(· | st+1),Pπt+1:T(· | st+1)) ≤ 1 −

ℓj. (13)

(1 − ℓj) ≤

j=t+1

j=t+1

Proof Construct a stepwise maximal coupling of the two suffix processes. As long as the two sampled suffixes have not disagreed, they share the same state sj. The one-step disagreement probability at that state is

DTV(µ(· | sj),π(· | sj)) = Dj ≤ ℓj. Therefore the conditional probability of no disagreement over the whole suffix is at least

T

(1 − ℓj).

j=t+1

The coupling characterization of total variation gives

DTV Pµt+1:T(· | st+1),Pπt+1:T(· | st+1) ≤ Pr Ytµ+1:T ̸= Ytπ+1:T | st+1 ≤ 1 −

T

(1 − ℓj).

j=t+1

Since total variation is at most one, we may take ℓj ∈ [0,1]. The union bound then yields

T

T

1 −

(1 − ℓj) ≤

ℓj.

j=t+1

j=t+1

#### B.3 Remaining-horizon bound on the surrogate residual

- Proposition 4 (Surrogate-residual bound under token-level divergence thresholds). A maximalcoupling argument on the suffix likelihood ratio gives the following bound on the surrogate residual. This is the first inequality of Equation (4) in Section 3.1. If Dt ≤ ℓt pathwise for all t, then

T−1

T

ℓj. (14)

|∆(µ,π)| ≤ 4ξ

ut

t=1

j=t+1

Proof Lemma 2 and |R(x,y)| ≤ ξ give

|∆(µ,π)| ≤ ξ

T

Eµ[|ρt − 1||1 − ρt+1:T|].

t=1

Fix t < T and condition on st+1. The future likelihood ratio ρt+1:T is the Radon–Nikodym derivative of the suffix law under π with respect to the suffix law under µ, so

Pπt+1:T(yt+1:T | st+1) Pµt+1:T(yt+1:T | st+1)

Pµt+1:T(yt+1:T | st+1) 1 −

Eµ[|1 − ρt+1:T| | st+1] =

yt+1:T

= 2DTV Pµt+1:T(· | st+1),Pπt+1:T(· | st+1) ≤ 2

T

ℓj,

j=t+1

where the last line uses Lemma 3. The sampled-token ratio contributes

π(yt | st) µ(yt | st) − 1

Eyt∼µ(·|st)[|ρt − 1| | st] =

µ(yt | st)

yt

= 2Dt. Taking expectation over st ∼ dµt gives Eµ|ρt − 1| = 2ut. Combining the two displayed bounds,

T

Eµ[|ρt − 1||1 − ρt+1:T|] ≤ 4ut

ℓj.

j=t+1

The term t = T is zero because ρT+1:T = 1. Summing over t = 1,...,T − 1 proves the claim.

#### B.4 CPPO policy-improvement bound (Theorem 1) Proof [Proof of Theorem 1] We prove the theorem in three steps.

- Step 1: reduce the residual to a weighted sum of token-level divergences. The constraint wtDt ≤ ct in (5) implies

ct wt

, ℓ¯:= max

Dt ≤ ℓt :=

ℓt.

t

By Proposition 4,

T−1

|∆(µ,π)| ≤ 4ξ

t=1

T

ut

j=t+1

T−1

λtut, λt := 4ξℓ¯(T − t).

ℓj ≤

t=1

- Step 2: convert the prefix constraints into prefix inequalities in expectation. Taking expectations in the prefix-budget constraints gives, for every m = 1,...,T − 1,

m

j=1

wjuj ≤ δb

m

j=1

wj.

Define the centered prefix slack

Sm :=

m

j=1

wj(uj − δb), S0 := 0.

Then Sm ≤ 0 for every m < T. Also define

∆St := St − St−1 = wt(ut − δb), rt :=

λt wt

.

- Step 3: apply Abel summation. Using the previous definitions, T−1

T−1

T−1

λtut − δb

λt(ut − δb)

λt =

t=1

t=1

t=1

T−1

=

rt ∆St

t=1

T−2

(rt − rt+1)St.

= rT−1ST−1 +

t=1

By assumption, rt is non-increasing, so rt −rt+1 ≥ 0; also rT−1 ≥ 0 and St ≤ 0. Hence the right-hand side is non-positive, and therefore

T−1

T−1

λtut ≤ δb

λt. Finally,

t=1

t=1

T−1

T−1

λt = 4ξℓ¯

(T − t) = 2ξT(T − 1)ℓ.¯

t=1

t=1

Combining this residual bound with Lemma 2 gives

J(π) − J(µ) ≥ L′µ(π) − 2ξT(T − 1)ℓδ¯ b. If ct ≡ δ and wt ∈ [wmin,1], then

δ wt ≤

δ wmin

ℓ¯= max

. Substituting this into the bound gives Equation (7).

t

#### B.5 Linear schedule satisfies the monotonicity

- Proposition 5 (Linear schedule monotonicity). For wt = 1 − 1−Tw−min1 (t − 1) and gt := (T − t)/wt, gt − gt+1 = wmin/(wtwt+1) > 0, so rt = 4ξℓg¯ t is strictly decreasing on t = 1,...,T − 1. Therefore Theorem 1 applies to the implemented schedule.

Proof Write

1 − wmin T − 1

T − t wt

, wt = 1 − c(t − 1), gt =

. Then

c :=

(T − t)wt+1 − (T − t − 1)wt wtwt+1

gt − gt+1 =

wt − c(T − t) wtwt+1

=

1 − c(T − 1) wtwt+1

=

wmin wtwt+1

=

> 0.

Thus gt is strictly decreasing, and rt = 4ξℓg¯ t is strictly decreasing as well.

#### B.6 Corollaries: uniform threshold and implementation threshold

These two corollaries supply the residual-constant comparisons used in the discussion after Theorem 1 in Section 3.2: Corollary 6 gives the uniform token-level threshold case (δb/δ ratio) and Corollary 7 the position-dependent implementation case (δb/(δwmin) ratio).

- Corollary 6 (Uniform token-level threshold). If ct = wtδ, equivalently ℓt ≡ δ, then ℓ¯= δ and

J(π) − J(µ) ≥ L′µ(π) − 2ξT(T − 1)δδb. (15) The residual constants are

CCPPO := 2ξT(T − 1)δδb, Cuniform := 2ξT(T − 1)δ2. Thus

CCPPO Cuniform

=

δb δ

. CPPO is tighter in the intended regime δb < δ ≤ 1.

- Corollary 7 (Position-dependent token-level threshold). If wtDt ≤ δ, equivalently ℓt = δ/wt, with wt ∈ [wmin,1], then ℓ¯≤ δ/wmin, hence

δ wmin

J(π) − J(µ) ≥ L′µ(π) − 2ξT(T − 1)

δb. (16)

Against the position-independent token-level divergence constant Cuniform(δ) = 2ξT(T − 1)δ2, the implemented CPPO constant is

Cimpl Cuniform(δ)

δ wmin

δb δwmin

Cimpl := 2ξT(T − 1)

δb,

=

.

It is tighter whenever δb < δ wmin. The linear schedule wt = 1 − 1−Tw−min1 (t − 1) used throughout our experiments satisfies the monotonicity assumption of Theorem 1 by Proposition 5.

#### B.7 Implementation slack from the initial prefix

Under Equation (8), the first update that moves the sampled-token ratio away from one matches the uniform token-level divergence baseline threshold. This adds a constant initial slack to the clean prefix inequality used in Theorem 1. The slack changes constants but not the leading prefix-budget term.

Proposition 8 (Initial prefix slack bound). Assume the weighted constraint gives Dt ≤ ℓ¯ and, in expectation,

m

wjuj ≤ δbWm + η (m = 1,...,T − 1)

j=1

for some constant slack η ≥ 0. Under the same monotonicity condition as Theorem 1,

4ξℓ¯(T − 1) w1

|∆(µ,π)| ≤ 2ξT(T − 1)ℓδ¯ b + η

. (17)

For the implemented linear schedule, w1 = 1. The initial prefix slack in Equation (8) has η ≤ δ, so its additional residual term is O(Tℓδ¯ ).

#### Proof Define

Am :=

m

wj(uj − δb), A0 := 0.

j=1

The assumed slack gives Am ≤ η for every m < T. As in the proof of Theorem 1, set

λt wt

λt := 4ξℓ¯(T − t), rt :=

.

Abel summation gives

T−1

λtut − δb

t=1

T−1

λt = rT−1AT−1 +

t=1

T−2

(rt − rt+1)At.

t=1

Because rt is non-increasing, all coefficients on the right are nonnegative. Therefore

rT−1AT−1 +

T−2

T−2

(rt − rt+1)At ≤ η rT−1 +

t=1

t=1

= ηr1

4ξℓ¯(T − 1) w1

.

= η

(rt − rt+1)

Combining this with Proposition 4 proves the displayed residual bound. It remains to identify the slack induced by the implemented constraint test. Equation (8) implies

Sm ≤ δ + δbWm−1 ≤ δ + δbWm. Taking expectations gives the proposition’s prefix-slack condition with η ≤ δ.

#### B.8 Sequence-level methods as special cases

Let W := t<T wt.

Uniform token-level divergence masks recover the position-agnostic branch. If wt ≡ 1 and the prefix budget is removed, only the token-level threshold Dt ≤ δ remains; this is the positionagnostic specialization underwriting the textbook bound, of which the uniform-TV mask is one implementation. TRM-Max corresponds to the sequence-level form of this uniform token-level threshold:

Dt ≤ δ =⇒ ut ≤ δ for every t. It therefore recovers the same quadratic constant

max

t

Cmax = 2ξT(T − 1)δ2.

CPPO adds the prefix conditions in (5). Under this uniform token-level threshold, the constant becomes

CCPPO Cmax

δb δ

CCPPO = 2ξT(T − 1)δδb,

.

=

TRM-Avg is a terminal-only relaxation of the prefix constraints. With wt ≡ 1 and only the final prefix constraint kept, a direct bound on the residual gives

Cterminal ≤ 4ξℓ¯(T − 1)2δb. CPPO’s Abel bound using every prefix at w ≡ 1 is

CCPPOw≡1 = 2ξT(T − 1)ℓδ¯ b. Hence

Cterminal Cw≡1

2 T

. (18)

= 2 −

CPPO

For arbitrary wt > 0, the weighted min-max inequality

λt wt ≥

W max

λt

t<T

t<T

yields

Cterminal(w) ≥ CCPPO(w),

with equality if and only if wt ∝ λt ∝ T − t. The same algebra applies to length-neutral KL or TV variants of the sequence-level comparison.

#### B.9 Product-form suffix bound

The main paper uses the linear-suffix specialization of the residual bound. Lemma 3 also gives the tighter product bound

T

T

(1 − ℓj) ≤

βt := 1 −

ℓj.

j=t+1

j=t+1

Substituting βt for the linear sum yields

T−1

|∆(µ,π)| ≤ 4ξ

ut βt,

t=1

with corresponding λβt := 4ξβt. Under the analogous monotonicity rtβ := λβt /wt non-increasing, the same Abel-summation step gives, with ℓj ≡ δ,

T−1

λβt

|∆| ≤ δb

t=1

T−1

1 − (1 − δ)T−t

= 4ξδb

t=1

= O(T δb),

i.e. a true O(T) branch (not requiring δb = O(1/T)). The monotonicity for λβt /wt is not identical to the linear-branch monotonicity for λt/wt, since βt saturates at 1 for early positions and at → 0 as t → T, while T − t does not. For small δ, βt ≈ (T − t)δ and the two monotonicity conditions coincide. We do not use this branch as the main quantitative claim in the paper.

#### B.10 Abel summation and tightness

The proof of Theorem 1 relies on prefix constraints rather than a pointwise estimate. The residual bound contributes the decreasing remaining-horizon coefficients λt ∝ T −t, while the prefix constraints supply the family of prefix inequalities Sm ≤ 0. A pointwise upper bound on t<T λtut ignores the prefix structure entirely and yields nothing better than O(T δ ℓ¯); Abel summation, the discrete analogue of integration by parts, instead rewrites the same sum as

λtut − δb

t<T

t<T

T−2

(rt − rt+1)St,

λt = rT−1ST−1 +

t=1

which expresses the residual as a sum of prefix slacks St ≤ 0 weighted by the gaps rt − rt+1. Both factors carry the correct sign as soon as rt = λt/wt is non-increasing, so every term contributes to tightening the bound. The condition rt − rt+1 ≥ 0 is therefore the formal statement that the constraint places larger weights where the remaining-horizon coefficient is larger, and the bound saturates when the two profiles are matched (wt ∝ λt ∝ T − t).

Sharpness. Among all nonnegative sequences {ut} satisfying the same prefix inequalities Sm ≤ 0 for every m, the upper bound

λtut ≤ δb

λt

t<T

t<T

is attained by the saturating sequence ut = δb whenever rt is non-increasing. The bound is therefore tight for this class of prefix constraints and for the residual bound in Proposition 4: it cannot be improved without strengthening one of them.

- B.11 Technical Lemmas Total variation and L1 distance.

- 1

- 2∥P − Q∥1.

∥P − Q∥TV =

#### Likelihood-ratio identity.

- p(X)

- q(X)

EX∼q 1 −

= |q − p| = 2DTV(p,q).

#### Coupling characterization of total variation.

DTV(P,Q) = inf Pr(X ̸= Y ), with maximal coupling attaining the infimum. Weighted averaging inequality. For at,wt > 0,

at t wt

at wt ≥ t

max

,

t

with equality if and only if at/wt is constant. Abel summation identity. For {rt}nt=1 and partial sums St = j≤t ∆j,

n−1

n

(rt − rt+1)St.

rt∆t = rnSn +

t=1

t=1

### C Soft-gate details

The main paper focuses on the hard CPPO mask. Soft trust-region gates can also attenuate gradients near the boundary of the constraint set; this appendix collects the construction, default choices, and the mixture-policy discussion deferred from Section 3.4. The soft variant is evaluated empirically in the hard-vs-soft ablation of Section 4.3 (right panel of Figure 5).

#### C.1 Gradient-scaling interpretation

The hard mask of Equation (10) tests the feasibility condition It, which holds when both the tokenlevel threshold wtDt ≤ δ and the prefix constraint St ≤ δ + δbWt−1 are met. To attenuate gradients smoothly near this boundary rather than dropping them, we measure how close a token is to violating It with the normalized violation score

wtDt δ

St δ + δbWt−1

xCPPOt := max

, (19)

,

which is at most one exactly when It holds and exceeds one in proportion to the larger of the two constraint violations.

Multiplying the loss by a soft gate g(xCPPOt ) does not by itself construct a softened policy; it scales each token’s gradient contribution. For a non-increasing gate g: [0,∞) → [0,1] with g(x) = 1 for

x ≤ 1 and xg(x) ≤ 1 for x > 1, the effective normalized violation per token, xCPPOt g(xCPPOt ), is bounded by

xCPPOt g(xCPPOt ) ≤ 1. This bounds the scaled violation that enters the gradient term. The corresponding soft CPPO mask is

1, sgn(Aˆt)(ρt − 1) ≤ 0, g(xCPPOt ), otherwise.

Mtsoft =

(20)

- C.2 Mixture-policy construction To obtain a formal guarantee that mirrors Theorem 1, define a mixture policy

πg(· | s) = (1 − gs)µ(· | s) + gs π(· | s), gs ∈ [0,1]. (21) For each state s,

πg(· | s) − µ(· | s) = gs π(· | s) − µ(· | s) . Therefore

DTV(µ(· | s),πg(· | s)) =

- 1

- 2 ∥πg(· | s) − µ(· | s)∥1

= gs DTV(µ(· | s),π(· | s)).

Applying the token-level threshold and prefix-budget constraints to πg therefore recovers an exact-TV guarantee with Dt replaced by gtDt. We do not deploy πg in our experiments; the construction is included to clarify what a soft-gate guarantee would require, and to distinguish it from the effective-gradient interpretation above.

- C.3 Default gate and SAPO compatibility

We use ginv(x) = min{1,1/x}, which trivially satisfies xg(x) ≤ 1 and g(x) = 1 for x ≤ 1. The theorem chain only requires the admissibility condition xg(x) ≤ 1, so any other admissible gate can be substituted without changing the formal guarantee. Soft gates introduced by SAPO-style schemes (Gao et al., 2025) can be plugged in by applying their attenuation function to the normalized CPPO score xCPPOt in place of a sampled-ratio score.

### D Experiment details and per-benchmark breakdown

Training stack. All runs use the verl-compatible GRPO/DAPO trainer with group-normalized advantages, the mask_std_0 filter (prompts with zero advantage variance are skipped), no entropy regularizer, and no KL-to-reference penalty in the loss. Optimizer is AdamW at a maximum learning rate of 1 × 10−6.

Rollout and update budget. At each iteration we draw a rollout batch of prompts and unroll rollout.n responses per prompt under the current policy µ, then split each rollout batch into ministep gradient minibatches of size train_bs for the policy update. Training rollouts are sampled from the actor’s untruncated softmax; validation rollouts use temperature 0.7 and top-p = 0.95 with 16 samples per prompt. Table 3 reports the per-model rollout / minibatch configuration (ministep = rollout_bs/train_bs).

- Table 3: Per-model rollout, update, and trust-region configuration. “Prompts” is the prompt count per rollout iteration, n is the number of responses per prompt, and “updates” is the number of gradient minibatches per rollout iteration. The token-level threshold scale δ is shared by DPPO and

CPPO; the listed δbmin values are the minimum values used by the Base-model warm-up calibration, whose upper bound is dynamically bounded at 2δbmin; wmin is the weight floor.

Model Prompts n Minibatch Updates Tmax δ δbmin wmin

Qwen3-1.7B 64 8 32 2 8k 0.15 0.015 0.8 Qwen3-1.7B-Base 64 8 32 2 8k 0.15 0.020 0.8 Qwen3-8B-Base 128 8 32 4 8k 0.15 0.020 0.8 Qwen3-30B-A3B-Base 256 16 32 8 16k 0.20 0.020 0.8

Evaluation. We evaluate every checkpoint on AIME24, AIME25, and AIME26. For each benchmark we report Avg@16 — the success rate computed over 16 sampled completions per prompt under the validation decoding configuration above. As a single summary score we further report AIME24/25/26 Avg@16, the unweighted mean of the three per-benchmark Avg@16 values. The best value reported in Tables 1 and 4 is the highest AIME24/25/26 Avg@16 attained on the validation curve within [0,Tstop], where Tstop is the matched evaluation horizon used for the training-curve plots; the per-benchmark numbers in Table 4 are the components of the best mean (i.e. the per-benchmark scores at the step that attained the best mean), not the per-benchmark maxima.

Baselines and divergence scores. We group the baselines by the signal used to decide whether a token update that moves the sampled-token ratio farther from one remains inside the trust region. GRPO and CISPO are sampled-ratio baselines: they operate directly on the sampled importance ratio ρt (with asymmetric clip thresholds for CISPO) and do not use a distributional divergence score. The GRPO row uses the Clip-Higher recipe, with (ϵlow,ϵhigh) = (0.2,0.28) in the asymmetric clipping interval [1 − ϵlow,1 + ϵhigh]; we denote it as GRPO in the tables for brevity. The remaining baseline hyperparameters use the values recommended in their original papers. MinPRO is a prefix-ratio baseline that weights the sampled importance ratio by a noncumulative minimum-prefix-ratio surrogate, again without a divergence score. DPPO applies a uniform reduced-TV token-level mask with the per-model token-level threshold scale δ in Table 3. CPPO uses this matched scale inside the weighted and prefix constraints. For all token-level divergence masks in the main experiments, we estimate Dt with a Top-K (K=20) reduced-TV score. Following the DPPO approximation construction, this score keeps the top-K rollout-policy tokens together with the sampled token and an “other” category, yielding a partitioned-TV lower bound on the exact full-vocabulary TV. DPPO and CPPO use a matched Top-K score and permodel threshold scale δ, so the empirical comparison isolates the prefix constraints rather than the token-level divergence approximation. This reduced-TV lower bound is used for the matched empirical mask, not as the exact-TV guarantee in Theorem 1. In Base-model runs, the listed δb is the minimum value in the warm-up calibration described in Section 4.1. Specifically, for a rollout sequence with token-level divergences D1:T, the dynamic sequence budget is computed as δbseq = clamp(Quantile(D1:T,0.9),δbmin,2δbmin), where Quantile(D1:T,0.9) represents the boundary value above which the top 10% largest divergences lie. Hence, the δb = 0.03 ablation corresponds to the setting where this quantile-based calibration is disabled and δb is fixed instead. The Binary-TV score is used only in the right panel of Figure 6. TRM-Max and TRM-Avg use KL as the token-level divergence in their original form: M(y) = 1[maxt DtKL ≤ δ] and M(y) = 1[T−1 t DtKL ≤ δavg] respectively, with DtKL = KL µ(· | st)∥π(· | st) . The comparison of Section 2 (TRM-Max ↔ CPPO’s uniform token-level threshold; TRM-Avg ↔ terminal-only specialization of CPPO’s prefix

constraints) does not depend on the choice between TV and KL and applies to either. Each baseline’s corresponding threshold follows the value recommended in its original paper.

Batched mask computation. Algorithm 1 gives the readable one-response version. The implementation applies the same recurrence to a batch tensor. For a valid-token mask M ∈ {0,1}B×T, padding positions are zeroed in the weights and in Z = w ⊙ D. A cumulative sum along the response dimension gives the tensors W and S for all positions at once, and a one-token right shift gives W− and S− so that the threshold at token t uses only the preceding prefix. The final binary mask is then multiplied into the same token-level ratio–advantage terms as the base PPO-style clipped objective.

Base-model warm-up diagnostics. Figure 7 visualizes the warm-up calibration used by the three Base-model CPPO runs. During the initial exploration phase of Base-model training, the average token-level divergence is initially large but decays rapidly after a few steps. To prevent prematurely clipping these early exploratory tokens before training stabilizes, a uniform sequence-independent threshold can be overly restrictive. By utilizing the 90th percentile of the sequence’s own divergence profile, the constraint dynamically adapts to the current sequence’s statistics:

δbseq = min 2δbmin,max δbmin,P90(D1:T) (22)

where P90 denotes the 90th percentile of the token-level divergence sequence D1:T. The top row of Figure 7 shows the mean effective prefix-average threshold after calibration; the δb values in Table 3 are the minimum values δbmin. The bottom row shows the fraction of masked tokens rejected by the prefix-budget condition. The effective values remain well below the corresponding token-level thresholds (δ = 0.15 or 0.20), and the prefix-budget mask fraction is concentrated early in training, which is the regime where the average token-level divergence is exceptionally large before rapidly decaying as the policy stabilizes.

1.7B-Base

8B-Base

A3B-Base

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0.030

EffectiveδBudget-maskshareb

0.025

0.020

1.00

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.75

0.50

0.25

0.00

150 300 450 600

100 200 300 400

125 250 375 500

Training Step

Training Step

Training Step

- Figure 7: Warm-up diagnostics for the three Base-model CPPO runs. Each column is one model.

Top: mean effective δb after the per-sequence warm-up calibration. Bottom: fraction of masked tokens rejected by the prefix-budget condition.

Full training diagnostics. Figures 8–11 provide per-model training traces behind the aggregate results in Table 1. The top row separates the three validation components, AIME24, AIME25, and AIME26, while the bottom row reports training reward, response length, and the relative log-probability error between the rollout and training log-probabilities. These plots are included as diagnostics rather than as additional selection rules: the reported scores in Tables 1 and 4 are still selected only by the best AIME24/25/26 mean within the matched evaluation horizon. The relative log-probability error panels are implementation stability checks and are not part of the training objective or the formal guarantee.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0.3

0.3

0.3

0.2

0.2

0.2

0.1

0.1

0.1

Reward

Response Length (k)

Rel. Log-Prob. Error

0.08

0.70

0.07

7.0

0.60

0.06

6.0

0.50

0.05

0.40

0.04

5.0

0.03

0.30

4.0

0.02

0.20

250 500 750 1000

250 500 750 1000

250 500 750 1000

Training Step

Training Step

Training Step

CPPO (Ours) DPPO-TV TRM-Avg TRM-Max CISPO GRPO MinPRO

##### Figure 8: Complete training diagnostics for Qwen3-1.7B (post-trained). Top: AIME24, AIME25, and AIME26 validation Avg@16. Bottom: training reward, response length, and relative log-probability error. Only reward is smoothed for readability.

###### 0.15 AIME25

###### AIME26

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.15

0.10

0.10

0.10

0.05

0.05

0.05

0.00

0.00

0.00

Reward

Response Length (k)

Rel. Log-Prob. Error

6.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.40

0.08

5.0

0.30

4.0

0.06

3.0

0.20

0.04

2.0

0.10

1.0

0.02

0.00

150 300 450 600

150 300 450 600

150 300 450 600

Training Step

Training Step

Training Step

CPPO (Ours) DPPO TRM-Avg TRM-Max CISPO GRPO MinPRO

##### Figure 9: Complete training diagnostics for Qwen3-1.7B-Base. Top: AIME24, AIME25, and AIME26 validation Avg@16. Bottom: training reward, response length, and relative log-probability error. Only reward is smoothed for readability.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0.3

0.3

0.3

0.2

0.2

0.2

0.1

0.1

0.1

Reward

Response Length (k)

0.20 Rel. Log-Prob. Error

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

8.0

0.60

0.15

6.0

0.40

0.10

4.0

0.20

0.05

2.0

0.00

100 200 300 400

100 200 300 400

100 200 300 400

Training Step

Training Step

Training Step

CPPO (Ours) DPPO TRM-Avg TRM-Max CISPO GRPO MinPRO

##### Figure 10: Complete training diagnostics for Qwen3-8B-Base. Top: AIME24, AIME25, and AIME26 validation Avg@16. Bottom: training reward, response length, and relative log-probability error. Only reward is smoothed for readability.

- 0.0
- 0.1
- 0.2
- 0.3
- 0.4
- 0.5
- 0.6
- 0.7 AIME24

###### AIME25

- 0.0

- 0.1

- 0.2

- 0.3

- 0.4

- 0.5

- 0.6 AIME26

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.4

0.3

0.2

0.1

0.0

Reward

Response Length (k)

Rel. Log-Prob. Error

0.60

15.0

0.80

0.50

12.5

0.60

0.40

10.0

0.30

0.40

7.5

0.20

5.0

0.20

0.10

2.5

0.00

0.0

0.00

125 250 375 500

125 250 375 500

125 250 375 500

Training Step

Training Step

Training Step

CPPO (Ours) DPPO TRM-Avg TRM-Max CISPO GRPO MinPRO

##### Figure 11: Complete training diagnostics for Qwen3-30B-A3B-Base. Top: AIME24, AIME25, and AIME26 validation Avg@16. Bottom: training reward, response length, and relative log-probability error. Only reward is smoothed for readability.

- Table 4: Detailed per-benchmark evaluation results. This table expands the aggregate results from Table 1 (%, Avg@16). AVG denotes the best AIME24/25/26 Avg@16 within the matched evaluation window; the per-benchmark columns report the AIME24/AIME25/AIME26 scores at this best-average checkpoint, not their individual maxima. collapse indicates training divergence, which only occurred for CISPO on Qwen3-30B-A3B-Base at step 215. Within each model block, the best score in a column is in bold and the second-best is underlined.

Method AIME24 AIME25 AIME26 AVG (↑) Qwen3-1.7B (Tstop = 1120)

GRPO 30.41 28.75 24.58 27.91 MinPRO 26.87 32.71 23.54 27.71 CISPO 30.62 28.12 27.71 28.82 DPPO 31.04 28.22 25.31 28.19 TRM-Max 26.46 22.91 26.25 25.21 TRM-Avg 29.79 25.41 25.41 26.87 CPPO (ours) 34.79 30.63 30.21 31.88

Qwen3-1.7B-Base (Tstop = 680)

GRPO 11.25 6.87 8.54 8.89 MinPRO 16.04 10.41 6.66 11.04 CISPO 15.62 12.08 7.91 11.87 DPPO 13.96 8.33 10.41 10.90 TRM-Max 14.16 7.29 7.71 9.72 TRM-Avg 15.41 10.21 9.48 11.70 CPPO (ours) 15.21 12.71 10.42 12.78

Qwen3-8B-Base (Tstop = 490)

GRPO 25.21 23.75 22.91 23.96 MinPRO 35.21 26.04 27.91 29.72 CISPO 32.91 25.41 30.41 29.58 DPPO 30.21 26.87 29.58 28.89 TRM-Max 28.33 24.37 27.50 26.73 TRM-Avg 31.46 25.00 27.50 27.98 CPPO (ours) 34.58 30.00 28.75 31.11

Qwen3-30B-A3B-Base (Tstop = 560) GRPO 43.75 32.71 38.12 38.19 MinPRO 52.91 38.54 52.91 48.12 CISPO collapse collapse collapse collapse DPPO 57.29 39.58 50.83 49.23 TRM-Max 26.46 20.41 13.96 20.27 TRM-Avg 54.37 37.08 55.41 48.96 CPPO (ours) 64.79 43.13 56.46 54.79

