# arXiv:2602.10693v3[cs.LG]8May2026

## VESPO: Variational Sequence-level Soft Policy Optimization for Off-Policy LLM Training

##### Guobin Shen Chenxiao Zhao Xiang Cheng Lei Huang Xing Yu∗

Xiaohongshu Inc.

#### Abstract

Off-policy updates are inevitable in reinforcement learning (RL) for large language models (LLMs) due to rollout staleness from asynchronous training and mismatches between training and inference engines. Naive importance sampling gives an unbiased correction but suffers from high variance, which is amplified by unbounded ratios and autoregressive generation. Prior remedies either rely on scenario-specific engineering, or trade bias for variance via token-level clipping or sequence-level normalization, yet these approaches remain largely heuristic. We propose Variational sEquence-level Soft Policy Optimization (VESPO). By explicitly incorporating variance reduction into a variational formulation, we derive a principled closed-form reshaping kernel that operates directly on sequence-level importance weights, avoids token-level approximation and length normalization, and admits an explicit variance bound for the deployed kernel. Experiments on math reasoning and code generation show that VESPO maintains stable training under severe off-policy conditions (staleness up to 64×) and delivers consistent gains across both dense and Mixture-of-Experts (MoE) models, outperforming recent reshaping baselines under matched setup. Code is available at https://github.com/FloyedShen/VESPO.

#### 1 Introduction

Reinforcement learning (RL) has become a key technique for tackling complex problem-solving tasks with large language models (LLMs), enabling capabilities such as multi-step mathematical reasoning and code generation [20, 1, 4, 30].

In practice, off-policy updates arise naturally in RL pipelines for LLMs. A common source is that systems split large rollout batches into mini-batches for sequential updates [30], causing later batches to become stale relative to the evolving policy. Asynchronous systems [6, 38, 19] amplify this by decoupling rollout from training entirely. Training-inference mismatches introduce further discrepancies, especially in MoE models where routing decisions compound through layers. To stabilize training under such distribution mismatch, existing works adopt truncated importance sampling (TIS) [16], mask out off-policy samples from training [10], or replay expert routing for target policy inference [35, 18]. While PPO [22] enforces trust region constraints via clipping, it does not directly address the variance challenge of sequence-level importance sampling (IS).

Target

Proposal *

Behavior

Figure 1: VESPO reformulates IS weight reshaping as finding a proposal Q∗ that balances proximity to µ and π under a variance constraint.

∗Correspondence to: yuanshan2@xiaohongshu.com, floyed shen@outlook.com.

Preprint.

Existing methods address this challenge through various importance weight transformations. Most operate at the token level: GRPO [23, 4] applies PPO-style clipping to per-token ratios; other methods such as SAPO [7] design heuristic transformations for off-policy importance weights [29, 36, 5, 21, 11]. However, token-level transformations are a compromise to avoid the multiplicative variance explosion of sequence-level weights, and have been shown to be merely a first-order approximation to their sequence-level counterparts [35]. Sequence-level approaches [35, 33] introduce length normalization to control variance, but this normalization makes the IS estimator biased; hard clipping is still required on top. Despite these efforts, principled guidance for designing importance weight transformations remains limited.

We propose Variational sEquence-level Soft Policy Optimization (VESPO), which takes a fundamentally different approach: rather than designing reshaping heuristics, we explicitly incorporate variance reduction for off-policy importance sampling into a variational formulation, yielding a principled closed-form solution (Figure 1). Additionally, we find that the resulting transformation is particularly friendly to sequence-level optimization: unlike previous methods that rely on length normalization to avoid variance explosion, VESPO operates directly on sequence-level importance weights without approximation or normalization.

Our contributions are summarized as follows:

- • We recast importance weight reshaping as a measure change to an implicit proposal distribution, and derive a principled closed-form kernel from a variance-constrained variational objective.
- • The resulting algorithm, VESPO, operates directly on sequence-level importance weights without length normalization, preserving inter-token dependencies, and admits an explicit variance bound that ensures bounded gradient contributions under arbitrary staleness.
- • Experiments on math reasoning and code generation demonstrate that VESPO remains stable under staleness up to 64× across dense and MoE architectures, transfers across reward modalities without retuning, and outperforms recent reshaping baselines.

#### 2 Preliminaries

Notation. We consider an autoregressive language model parameterized by θ as a policy πθ. Let x denote a query sampled from a dataset D. In off-policy settings, responses y = (y1,...,yT) are sampled from a behavior policy µ (e.g., an earlier checkpoint or a different inference engine). The likelihood of generating y given x factorizes as:

T

πθ(yt | x,y<t). (1)

πθ(y | x) =

t=1

We write τ = (x,y) for a query-response pair, and R(τ) ∈ R for the sequence-level reward assigned to the complete response.

Policy Gradient with Off-Policy Correction. The goal is to maximize the expected reward under the current policy:

R(τ) . (2)

J (θ) = Eτ∼π

θ

When samples are drawn from a behavior policy µ instead, importance sampling provides an unbiased correction. Taking the gradient yields the off-policy policy gradient:

∇θJ (θ) = Eτ∼µ [W(τ) · R(τ) · ∇θ log πθ(τ)], (3)

where W(τ) = πθ(τ)/µ(τ) is the importance weight. This classical policy gradient view [27] reveals that the importance weight W serves as a gradient weighting factor: it determines how much each sample contributes to the parameter update. More generally, any modification to the importance weight can be understood as defining a reshaping function ϕ(W) that reweights the gradient:

∇θJ˜(θ) = Eτ∼µ [ϕ(W(τ)) · R(τ) · ∇θ log πθ(τ)]. (4)

This gradient-centric view will be central to our analysis: the practical effect of any weight transformation must ultimately be understood through how it reweights the policy gradient.

The Variance Challenge of Sequence-Level IS. Expanding the importance weight in terms of token-level ratios reveals a fundamental structural tension. Define the token-level importance ratio as

θ(yt|x,y<t)

ρt = π

µ(yt|x,y<t) . The sequence-level weight is then a product: W(τ) =

T

ρt, (5) while the log-policy gradient is a sum:

t=1

T

∇θ log πθ(yt | x,y<t). (6)

∇θ log πθ(τ) =

t=1

This product-sum structure creates a tension: the gradient contribution of each token is weighted by a global factor W(τ) that compounds across all T positions. Even small per-token deviations accumulate multiplicatively, causing W(τ) to exhibit extreme values for long sequences. The variance of W grows exponentially with T, rendering naive importance sampling impractical.

To tame this variance, existing methods define specific reshaping functions ϕ that modify the gradient weighting. GRPO [23] operates at the token level with a PPO-style clipped surrogate. From the gradient perspective, the effective weight function depends on the sign of the advantage A:

 

ρt, if A > 0 and ρt ≤ 1+ε, ρt, if A < 0 and ρt ≥ 1−ε, 0, otherwise (gradient zeroed).

(7)

ϕGRPO(ρt;A) =



This breaks the product structure and treats each token update independently, yielding only a firstorder approximation [34].

GSPO [35] operates at the sequence level, defining the gradient weight as the geometric mean of token-level ratios (i.e., normalizing by sequence length):

1/T

T

T

1 T

log ρt , (8)

ϕGSPO(W) =

ρt

= exp

t=1

t=1

followed by a clipping mechanism similar to GRPO. This normalization introduces a length-dependent bias: the implicit proposal distribution varies with T, and sequences with identical per-token statistics but different lengths receive identical weights despite having different true importance weights (see Section E for a formal analysis). These approaches all define ϕ heuristically; the question of what constitutes a principled choice of ϕ motivates the variational framework we develop next.

#### 3 VESPO: Variational Sequence-level Soft Policy Optimization

We develop a principled framework for designing importance weight transformations. We first show that any reshaping function ϕ(W) implicitly defines a proposal distribution, then formulate the design of ϕ as a variational problem with variance constraints, and finally derive a closed-form solution.

##### 3.1 Weight Reshaping as Measure Change

Standard importance sampling performs an unbiased measure change µ → π, while any transformation ϕ(W) induces a different measure change µ → Q for some implicit proposal Q. We now formalize this perspective. For any function G(τ):

Eτ∼µ ϕ(W(τ)) · G(τ) = Z · Eτ∼Q G(τ) , (9) where Z = Eµ[ϕ(W)] is a normalization constant, and Q is defined by

1 Z

µ(τ) · ϕ(W(τ)). (10)

Q(τ) =

The reshaped gradient (Equation (4)) thus equals Z · Eτ∼Q R(τ)∇θ log πθ(τ) : an expectation under the proposal Q retaining πθ’s score function (Q’s own score differs unless ϕ ∝ W).

This is the key insight: any sequence-level reshaping function ϕ(W) implicitly defines a proposal distribution Q. This perspective provides a unified lens to analyze existing importance weight transformations (see Section A for detailed analysis of specific methods). Rather than handcrafting ϕ directly, we can specify desirable properties of the proposal Q and derive the corresponding ϕ. A good proposal should remain close to µ for sampling efficiency, incorporate π to guide optimization, and control variance. The following subsections formalize these desiderata as a variational objective and derive a closed-form ϕ∗.

##### 3.2 Variational Objective

KL prior with importance tilt. We seek a proposal Q that retains sample efficiency from the available µ-samples while shifting mass toward trajectories favored by the target π. We formalize this as a KL-regularized linear objective:

J (Q) = DKL(Q∥µ) − α EQ[log W], α ≥ 0. (11)

The KL term keeps Q close to the sampling distribution, ensuring that µ-samples remain informative for estimating expectations under Q. The log-importance term rewards Q for placing mass where π exceeds µ (i.e., W > 1), tilting Q toward the target policy and reducing bias in the gradient estimate. The coefficient α acts as an inverse temperature controlling the strength of this tilt: α = 0 recovers Q = µ, α = 1 yields Q = π, and α > 1 concentrates Q on increasingly high-importance trajectories. This is the KL-regularized linear form standard in maximum-entropy RL and control-asinference [15], here applied to the proposal distribution rather than the policy itself.

Variance Constraint. Proximity alone is insufficient: finite sample sizes demand variance control. In importance sampling, the variance of the estimator scales with the second moment Eµ[W2], a classical result that also underlies the effective sample size (ESS) diagnostic.

Under the measure-change view, this second moment can be related to an expectation under Q. By Equation (10), we have Q(τ) ∝ µ(τ)ϕ(W(τ)), so

Eτ∼Q[W(τ)] ∝ Eτ∼µ[ϕ(W) · W]. (12)

When ϕ(W) ≈ W (approaching unbiased IS), this recovers Eµ[W2]; for general ϕ with K := supw ϕ(w)/w, the chain Eµ[ϕ(W)2] = Z · EQ[ϕ(W)] ≤ ZK · EQ[W] (formalized in Theorem 3.1) shows that bounding EQ[W] controls the second moment for any reshaping kernel. We therefore impose:

Eτ∼Q[W(τ)] ≤ C. (13) Constrained Optimization. Combining the KL-regularized tilt objective with the variance constraint:

DKL(Q∥µ) − α EQ[log W]

min

Q

s.t. EQ[W] ≤ C, Q(τ)dτ = 1. (14)

Introducing Lagrange multipliers λ ≥ 0 for the moment constraint and γ for normalization, we obtain the Lagrangian:

L(Q,λ,γ) = DKL(Q∥µ) − α EQ[log W]

+ λ EQ[W]−C + γ Qdτ−1 . (15)

##### 3.3 Closed-Form Solution

Taking the functional derivative δL

δQ = 0 yields (see Section B):

Q∗(τ) ∝ µ(τ) · W(τ)α · exp(−λW(τ)). (16) Comparing with Equation (10), we identify the reshaping function:

ϕ(W) = Wα · exp(−λW). (17)

This kernel has two components: the power term Wα and the exponential term exp(−λW) for soft suppression. It is smooth and differentiable everywhere, avoiding the discontinuities of hard clipping.

Surrogate Objective. The reshaped gradient implicitly maximizes a smooth surrogate JVESPO(θ) = Eµ[f(W(θ))A(τ)] that saturates as W → ∞ (Figure 2); f is the lower incomplete gamma function (full derivation in Section B).

Shifted form for practice. In what follows, we use the shifted form ϕ(W) = Wc

exp(c2(1−W)) with c1 = α, c2 = λ to ensure ϕ(1) = 1, so that on-policy samples receive unit weight. This shift differs from Equation (17) only by the multiplicative constant ec

1

2, equivalent to rescaling the learning rate, so the variational derivation carries over unchanged.

SAPO VESPO Hard Clip

f(W) when A > 0

f(W) when A < 0

(W) when A > 0

(W) when A < 0

- 0

- 1

- 2

- 3

- 4

- 0

- 1

- 2

- 3

- 4

3.0

3.0

2.5

2.5

2.0

2.0

()W

()fW

1.5

1.5

1.0

1.0

0.5

0.5

0.0

0.0

0 1 2 3

0 1 2 3

0 1 2 3

0 1 2 3

W

W

W

W

- Figure 2: Surrogate objectives f(W) and gradient scaling factors ϕ(W) = W · f′(W) for positive and negative advantages. Hard clipping zeros ϕ abruptly at the boundary; VESPO peaks near W=1 and decays smoothly.

##### 3.4 Variance Bound and Staleness Robustness

The variational derivation in Section 3.3 identifies the kernel form, but the constraint EQ[W] ≤ C controls only the first moment under Q. We now establish a formal variance bound that translates this constraint into an explicit guarantee for the second moment under µ, providing a theoretical foundation for the deployed kernel.

Proposition 3.1 (Variance Bound). Let ϕ(W) = Wc

exp(c2(1−W)) with c1,c2 > 0, and define K = supw>0 ϕ(w)/w. Under the moment constraint EQ[W] ≤ C,

1

Eµ ϕ(W)2 ≤ Z · K · C, Z = Eµ[ϕ(W)]. (18) Moreover, K < ∞ if and only if c1 ≥ 1. When finite, K admits the closed form

c1−1

c1 − 1 c2

exp(c2 − c1 + 1), (19)

K =

attained at w∗ = (c1−1)/c2.

The proof, given in Section C, follows by setting g = ϕ in the measure-change identity (Equation (9)) and applying ϕ(w) ≤ Kw. The bound is non-trivial precisely when c1 ≥ 1: at our practical settings, (c1,c2) = (2,3) gives K ≈ 2.46 and (3,2) gives K = 1, while for c1 < 1 the bound becomes vacuous (K = ∞). In the unbiased-IS limit (ϕ ≈ W), the bound reduces to Eµ[W2] ≲ Z · K · C, i.e., the classical second-moment quantity controlling effective sample size (ESS) and IS variance, so Theorem 3.1 can be viewed as extending this classical control to general ϕ. The criterion c1 ≥ 1 identified by the bound coincides with the inverse-temperature regime of the variational objective (Section 3.2, α = c1 ≥ 1) and is consistent with our practical choices selected purely from empirical performance.

Uniform Boundedness. The smooth kernel is also uniformly bounded for any c1,c2 > 0 (proof and explicit ϕmax in Section C). Consequently, no single off-policy sample can produce an unbounded gradient contribution, no matter how stale (i.e., no matter how large its raw weight W). Hard-clipping methods achieve a similar property only through discontinuous truncation, while VESPO’s smooth attenuation preserves differentiability throughout.

##### 3.5 The VESPO Algorithm

We instantiate the shifted kernel from Section 3.3 with asymmetric hyperparameters (c1,c2) for positive and negative advantages, mirroring the asymmetric clipping in PPO. Recent work [28] has shown that positive and negative samples exhibit different gradient dynamics during training; accordingly, we apply stronger suppression for A < 0 with W < 1 to prevent excessive penalization of samples the policy already dislikes, as shown in Figure 2. We treat (c1,c2) as tunable hyperparameters, allowing flexibility beyond the specific values implied by the variational derivation. The asymmetric assignment is an instance of the same kernel functional family (rather than a new family), so Theorem 3.1 applies to each branch since both (c+1 ,c+2 ) and (c−1 ,c−2 ) satisfy c1,c2 > 0. Substituting the shifted kernel into Equation (4), the VESPO gradient estimator becomes:

∇JVESPO = Eτ∼µ Wc

1

exp(c2(1−W)) · A(τ) · ∇log πθ(τ) , (20)

where A(τ) = R(τ) − b is the advantage with baseline b (mean reward within each prompt group, following GRPO [23]). For general ϕ ̸= W, the baseline acts as a control variate for the deployed surrogate Eµ[ϕ(W)A(τ)] rather than preserving the unbiasedness identity that holds only for ϕ = W. The kernel adapts smoothly to the importance weight: ϕ(W) ≈ 1 near on-policy, the exponential term decays for W ≫ 1, and the power term down-weights W ≪ 1. The factorized form gives flexible control: c1 governs W < 1 behavior and c2 controls decay for W > 1.

Computational cost. VESPO incurs no additional forward passes or memory beyond GRPO/GSPO: it reuses the same per-token log-probabilities under πθ and µ, computes log W as their masked sum, and applies ϕ as an elementwise reshaping detached from the computation graph. We evaluate c1 log W + c2(1 − W) in log-space, exponentiating only at the final step to avoid overflow under extreme staleness. This makes VESPO a drop-in replacement in standard RL pipelines (full pseudocode in Section I).

#### 4 Experiments

We evaluate VESPO on mathematical reasoning (the primary setting) and code generation (crossdomain transfer; Section 4.5). For math, we examine two practical sources of off-policy distribution shift: (1) policy staleness from batched rollouts, where later mini-batches are updated using samples from an outdated policy; and (2) train-inference mismatch, where different implementations between training and inference engines produce different outputs for the same input, an effect exacerbated in MoE models due to routing inconsistencies.

##### 4.1 Experimental Setup

We train on DAPO-Math [31] (verifier reward [12]) for math reasoning across three model scales: Llama-3.2-3B-Instruct [8], Qwen3-8B-Base, and Qwen3-30B-A3B-Base [30], on 32 H20 GPUs with veRL [24]. Math evaluation uses AIME 2024/2025, AMC 2023, and MATH-500 [9] (avg@k). For cross-domain transfer (Section 4.5), we additionally train Qwen3-30B-A3B-Base on PRIMERL/Eurus-2-RL-Data [3] with execution reward and evaluate on HumanEval+, MBPP+ [17], and LiveCodeBench v6 [13] (pass@10). Best checkpoint per method is selected by average accuracy. Baselines GRPO [23], GSPO [35], SAPO [7], TOPR [21], CISPO [2], BAPO [29], etc., use their respective official hyperparameters; VESPO uses (c1,c2) = (2.0,3.0) for A>0 and (3.0,2.0) for A<0, applied identically across all settings (no retuning between math and code). Staleness is simulated by fixing mini-batch size 256 and varying global batch size; primary experiments use staleness ratio N=gbs/mbs=8, ablations span N ∈ {4,8,16,32,64}. Full per-method hyperparameters and infrastructure details are in Section D.

##### 4.2 Main Results

- Table 1 presents the main results with gbs/mbs = 8 across three model scales. VESPO achieves the best (or tied-best) average accuracy on all three models, demonstrating the generality of our approach. Notably, the improvements are most pronounced on Qwen3-30B-A3B-Base, where VESPO outperforms the best baseline by 10.9 percentage points in average accuracy (70.0 vs 59.1). This suggests that VESPO’s soft suppression of extreme importance weights is particularly beneficial for MoE architectures, where routing inconsistencies amplify distribution shift and make training stability more challenging. Given these observations, we focus our detailed analysis on Qwen3-30B-A3B-Base in the following sections.

Table 1: Mathematical reasoning accuracy (%) with gbs/mbs

= 8 across three model scales. Best results in bold.

Model Method AIME25 AIME24 AMC23 MATH500 Avg

GRPO 0.5 14.5 40.6 51.6 26.8 GSPO 0.2 14.7 43.9 47.8 26.7 SAPO 0.7 12.2 34.1 51.0 24.5 VESPO 0.6 13.9 47.3 51.3 28.3

Llama3.2-3BInstruct

GRPO 27.4 40.0 74.5 76.4 54.6 GSPO 28.8 37.7 80.8 78.4 56.4 SAPO 36.7 49.0 80.0 78.0 60.9 VESPO 33.5 49.4 82.2 78.6 60.9

Qwen38B-Base

GRPO 28.2 40.0 81.4 78.3 57.0 GSPO 25.1 43.3 83.0 84.8 59.1 SAPO 21.4 27.9 73.0 84.6 51.7 VESPO 44.3 59.6 91.4 84.8 70.0

Qwen330B-A3BBase

- Table 2: Effect of staleness ratio N = gbs/mbs on Qwen3-30B-A3B-Base (avg@k, %). N=8 is in Table 1. SAPO’s downstream accuracy collapses at N≥32.

N = 4 N = 16 Method AIME25 AIME24 AMC23 MATH500 Avg Method AIME25 AIME24 AMC23 MATH500 Avg GRPO 22.1 33.1 76.4 84.5 54.0 GRPO 20.3 31.4 71.4 81.3 51.1 GSPO 27.6 43.3 83.1 85.7 59.9 GSPO 24.3 41.6 83.1 84.2 58.3 SAPO 38.4 51.4 85.2 85.9 65.2 SAPO 19.6 26.0 72.2 84.3 50.5 VESPO 43.1 60.3 91.2 85.4 70.0 VESPO 40.2 53.2 90.8 84.8 67.3 N = 32 N = 64 Method AIME25 AIME24 AMC23 MATH500 Avg Method AIME25 AIME24 AMC23 MATH500 Avg GRPO 21.8 33.4 73.4 80.9 52.4 GRPO 14.6 28.0 69.4 82.4 48.6 GSPO 18.8 27.3 73.4 80.6 50.0 GSPO 15.4 24.1 73.9 76.8 47.6 SAPO 12.4 7.2 29.5 34.5 20.9 SAPO 3.3 7.3 23.8 31.4 16.5 VESPO 37.7 51.4 85.2 83.7 64.5 VESPO 34.2 46.2 83.6 83.3 61.8

4.3 Robustness to Policy Staleness

We examine robustness to policy staleness by varying the staleness ratio N = gbs/mbs from 4 to 64. The training-reward curves in Figure 3 show VESPO’s remarkable consistency: all five curves (N=4 to N=64) follow nearly identical trajectories, converging to similar final rewards around 0.7. In contrast, GRPO saturates early; GSPO degrades as N grows, and additionally suffers a catastrophic collapse around step 1,200 at N=4; SAPO is competitive at N=4 but its training reward becomes unstable from N=8 onward, and downstream accuracy collapses at N≥32. These differences translate to downstream performance (Table 2): VESPO achieves the best average accuracy at every N, maintaining 61.8% at N=64, while GRPO and GSPO degrade to 48.6% and 47.6% respectively, and SAPO collapses entirely.

0 250 500 750 1000 1250 1500

Gradient Steps

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

TrainingReward

GRPO

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

0 250 500 750 1000 1250 1500

Gradient Steps

GSPO

0 250 500 750 1000 1250 1500

Gradient Steps

SAPO

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

0 250 500 750 1000 1250 1500

Gradient Steps

VESPO

N=4 N=8 N=16 N=32 N=64

Figure 3: Training reward across staleness levels (N ∈ {4,8,16,32,64}) on Qwen3-30B-A3B-Base. Each panel shows one method; VESPO is the only one with stable, consistent curves across all N.

4.4 Comparison with Recent Importance Weight Methods

Recent works propose alternative importance weight transformations targeting specific instability sources. We compare VESPO directly against three representative methods on Qwen3-30B-A3BBase under identical training setup (N=8): TOPR [21] hard-clamps sequence-level ratios to [0,1] for negative rewards (and uses weight 1 for positive rewards) with length normalization; CISPO [2] clips token-level ratios as stop-gradient REINFORCE weights; BAPO [29] adjusts PPO clipping bounds dynamically with a hard cap for negative advantages.

- Table 3: Comparison with recent importance weight reshaping methods on Qwen3-30B-A3B-Base (N=8).

Table 4: Cross-domain transfer to code generation (pass@10) using the same (c1,c2).

Method AIME25 AIME24 AMC23 MATH500 Avg TOPR 16.5 29.9 66.6 81.8 48.7 CISPO 16.0 25.5 70.2 83.0 48.7 BAPO 34.2 47.9 88.9 83.2 63.6 VESPO 44.3 59.6 91.4 84.8 70.0

Method HE+ MBPP+ LCB v6 Avg GRPO 84.8 74.9 23.1 60.9 GSPO 82.9 73.0 23.1 59.7 SAPO 86.6 74.6 24.2 61.8 VESPO 88.4 75.4 25.3 63.0

- Table 3 shows VESPO substantially outperforms all three baselines: +10 on AIME25 over BAPO and +28 over TOPR/CISPO. The training dynamics (Figure 11 in Section G) reveal distinct failure modes: CISPO collapses around step 280 from insufficient sequence-level variance control; BAPO

Training Reward

###### AIME25

Response Length

KL Divergence

Entropy

PG Loss

0.8

0.6

- 100
- 101

15

15000

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | |
|---|---|
| | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

400

gbs/mbs=4

0.6

10

0.4

10000

0.4

5

200

10 1

5000

0.2

0.2

0

10 2

0

0

0.0

0.0

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0.8

0.6

4

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

6000

100

gbs/mbs=8

0.6

0

0.4

4000

2

0.4

10 1

0.2

2000

0.2

- 0 500 1000 1500

- 1

0

0.0

0.0

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0.8

0.6

6

0.50

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

6000

gbs/mbs=16

0.6

100

0.4

0.25

4

4000

0.4

0.00

2

0.2

0.2

10 1

2000

0.25

0

0.0

0.0

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0.8

0.6

7.5

0.5

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

6000

100

gbs/mbs=32

0.6

5.0

0.4

0.0

4000

0.4

0.5

10 1

2.5

0.2

2000

0.2

1.0

0.0

0

0.0

0.0

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0.8

0.6

0.5

6000

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

15

gbs/mbs=64

0.6

0.4

100

0.0

4000

10

0.4

0.2

5

0.2

0.5

2000

10 1

0

0.0

0.0

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

Steps

Steps

Steps

Steps

Steps

Steps

GRPO GSPO SAPO VESPO

- Figure 4: Training dynamics on Qwen3-30B-A3B-Base across staleness N ∈ {4,8,16,32,64} (rows: N; columns: training reward, AIME25, response length, KL, entropy, PG loss). VESPO (red) is the only method with stable training across all N; baseline failure modes (GRPO entropy collapse, GSPO length-bias amplification, SAPO under-suppressed A<0) are detailed in Section F.

exhibits entropy explosion after step 1,100 as adaptive clip bounds over-relax; TOPR converges slowly with suppressed response length. The advantages of VESPO stem not from any single component, but from the joint effect of sequence-level operation, smooth (non-clipping) kernel, and variational derivation.

##### 4.5 Code Generation

- Table 4 reports pass@10 on HumanEval+, MBPP+, and LiveCodeBench v6 after training on code

(setup in Section 4.1; same (c1,c2) as in math). VESPO is the best on all three benchmarks, demonstrating that the variational framework yields hyperparameters that generalize across reward modalities (verifier-based math vs. execution-based code). Training dynamics and detailed analysis are in Section H.

##### 4.6 Robustness to Train-Inference Mismatch

A second source of off-policy shift is traininference mismatch: different numerical implementations between training (FSDP/Megatron) and inference (vLLM/SGLang [37]) engines produce different outputs, amplified in MoE models. Engineering fixes include Truncated IS (TIS) [16] and Routing Replay (R2) [18]. Figure 5 shows vanilla GRPO plateaus at ∼0.60; adding TIS or R2 improves it. VESPO without any fixes matches GRPO+R2, and VESPO+R2 attains the highest reward and best AIME25, showing complementarity with engineering fixes.

0.8

0.5

| | | | |
|---|---|---|---|
| | |GR GR<br><br>|PO PO+TIS|
| | |GR VE VE<br><br>|PO+R2 SPO SPO+R2|
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

AIME25(avg@32)

0.4

TrainingReward

0.6

0.3

0.2

0.4

0.1

0.2

0.0

0 500 1000 1500

0 500 1000 1500

Gradient Steps

Gradient Steps

Figure 5: Train-inference mismatch on Qwen330B-A3B-Base.

##### 4.7 Ablations

Length normalization. A key design choice in VESPO is operating at the sequence level without length normalization. Methods like GSPO normalize by 1/T to reduce variance, but we hypothesize this creates length-dependent bias. We compare VESPO with two normalized variants: VESPOsqrt

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
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
| | | | | |
| | | | | |

1000

2.5

0.7

100

800

2.0

0.6

TrainingReward

GradientNorm

KLDivergence

Entropy

600

1.5

0.5

1.0

400

0.4

10 1

0.5

200

0.3

0.0

0

0.2

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

0 200 400 600 800 1000

Gradient Steps

Gradient Steps

Gradient Steps

Gradient Steps

VESPO VESPOsqrt VESPOlin

Figure 6: Ablation on length normalization. VESPO without normalization achieves stable training; adding √

T or T normalization causes instability and collapse. (normalize by √

T) and VESPOlin (normalize by T). Figure 6 reveals striking differences: VESPOlin collapses around step 350 (KL spike + gradient explosion); VESPOsqrt shows periodic gradient spikes; VESPO without normalization is stable. Length normalization causes longer sequences to dominate batch gradients, biasing toward even longer outputs until collapse, a failure mode VESPO avoids.

Asymmetric hyperparameters. VESPO uses asymmetric hyperparameters c+ = (2,3) and c− = (3,2) for positive and negative advantages. We compare against symmetric variants (Figure 7). Using (2,3) for both gives insufficient suppression for A < 0, leading to instability; using (3,2) for both over-suppresses positive samples, slowing learning. The asymmetric design balances these trade-offs. More broadly, VESPO is robust to moderate hyperparameter variations as long as sufficient down-weighting is applied to negative-advantage samples with W < 1.

0.7

0.6

TrainingReward

0.5

0.4

VESPO

0.3

All c

All c +

0.2

0 100 200 300 400 500 600 700 800

Gradient Steps

Figure 7: Asymmetric hyperparameter ablation.

#### 5 Related Work

Policy Gradient Methods for LLMs. PPO [22] stabilizes updates via a clipped surrogate objective, while value-free alternatives have gained traction for LLM fine-tuning: GRPO [23] normalizes rewards within sample groups and clips token-level ratios; GSPO [35] operates at the sequence level with geometric-mean (length) normalization; DAPO [31] introduces decoupled clipping and dynamic sampling. These methods control variance via heuristic clipping or normalization. Our measure-change view (Section 3.1) reveals each as a specific choice of implicit proposal distribution, and VESPO derives the weight function from a variational principle rather than manual design.

Importance Weight Reshaping. Recent alternatives to hard clipping can be categorized along three axes: granularity, boundary, and theoretical origin. Token-level hard-boundary methods include CISPO [2], BAPO [29], GPPO/KLEAR [26], and M2PO [36]; however, per-token reshaping does not compose into IS under any single sequence-level proposal (Section A). SAPO [7] uses token-level smooth gating, sharing VESPO’s philosophy but inheriting the token-level limitation. TOPR [21] is sequence-level but with hard clamping to [0,1] for negative rewards and length normalization, suffering from boundary discontinuity and length bias (Section E). VESPO uniquely combines sequence-level operation, smooth kernel, and variational derivation, outperforming TOPR/CISPO/BAPO under matched setup (Table 3). Engineering fixes such as Routing Replay [18] and truncated IS [16] target specific instability sources and are complementary to VESPO (Section 4.6).

#### 6 Conclusion

We presented a measure-change view of importance weight reshaping and derived VESPO, a sequencelevel smooth kernel with explicit variance guarantees. VESPO is stable under staleness up to 64×, improves consistently across dense and MoE models, and transfers to code generation without retuning. While the kernel form is principled, specific (c1,c2) values still require user choice; our experiments cover verifiable-reward settings (math and code) and models up to 30B-scale, with broader reward modalities and frontier-scale verification left as future directions. Beyond LLM RL, the measure-change formulation and variance-constrained variational design generalize naturally to other variance-bias trade-offs that arise in value learning and reward-model evaluation under distribution shift, suggesting broader applicability across off-policy correction problems.

#### References

- [1] Anthropic. Introducing Claude 4. https://www.anthropic.com/news/claude-4, 2025. Accessed: 2025-05-22.
- [2] Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention, 2025. URL https://arxiv.org/abs/2506.13585.
- [3] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, and Ning Ding. Process reinforcement through implicit rewards,

2025. URL https://arxiv.org/abs/2502.01456.

- [4] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Jun-Mei Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiaoling Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bing-Li Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dong-Li Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, JingChang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Jiong Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, M. Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, Ruiqi Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shao-Kang Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wen-Xia Yu, Wentao Zhang, Wangding Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyu Jin, Xi-Cheng Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yi Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yu-Jing Zou, Yujia He, Yunfan Xiong, Yu-Wei Luo, Yu mei You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yao Li, Yi Zheng, Yuchen Zhu, Yunxiang Ma, Ying Tang, Yukun Zha, Yuting Yan, Zehui Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhen guo Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zi-An Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645:633 – 638, 2025. URL https://api.semanticscholar.org/CorpusID:275789950.
- [5] Madeleine Dwyer, Adam Sobey, and Adriane Chapman. It’s not you, it’s clipping: A soft trust-region via probability smoothing for LLM RL, 2025. URL https://arxiv.org/abs/ 2509.21282.
- [6] Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, WANG JIASHU, Tongkai Yang, Binhang Yuan, and Yi Wu. AREAL: A large-scale asynchronous reinforcement learning system for language reasoning. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025. URL https: //openreview.net/forum?id=X9diEuva9R.
- [7] Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, and Junyang Lin. Soft adaptive policy optimization, 2025. URL https://arxiv.org/abs/2511.20347.
- [8] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela

Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzm´an, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur C¸elebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, V´ıtor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman,

James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The Llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

- [9] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [10] Jian Hu. Stabilizing MoE RL without router replay: The online IcePop solution. https: //hijkzzz.notion.site/online-ice-pop, 12 2025. Accessed: 2025-12-16.
- [11] Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. REINFORCE++: Stabilizing critic-free policy optimization with global advantage normalization, 2025. URL https://arxiv.org/ abs/2501.03262.
- [12] Hugging Face. Math-verify: A rule-based math verification library. https://github.com/ huggingface/Math-Verify, 2025. Accessed: 2025-01-26.
- [13] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code, 2024. URL https://arxiv.org/abs/ 2403.07974.
- [14] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.
- [15] Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review, 2018. URL https://arxiv.org/abs/1805.00909.

- [16] Jiacai Liu, Yingru Li, Yuqian Fu, Jiawei Wang, Qian Liu, and Yu Shen. When speed kills stability: Demystifying RL collapse from the training-inference mismatch, September 2025. URL https://richardli.xyz/rl-collapse.
- [17] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by ChatGPT really correct? rigorous evaluation of large language models for code generation,

2023. URL https://arxiv.org/abs/2305.01210.

- [18] Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song, Yudong Wang, Zhifang Sui, and Fuli Luo. Stabilizing MoE reinforcement learning by aligning training and inference routers, 2025. URL https://arxiv.org/abs/2510.11370.
- [19] Michael Noukhovitch, Shengyi Huang, Sophie Xhonneux, Arian Hosseini, Rishabh Agarwal, and Aaron Courville. Asynchronous RLHF: Faster and more efficient off-policy RL for language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=FhTAG591Ve.
- [20] OpenAI. Learning to reason with LLMs. https://openai.com/index/ learning-to-reason-with-llms/, 2024. Accessed: 2024-09-12.
- [21] Nicolas Le Roux, Marc G. Bellemare, Jonathan Lebensold, Arnaud Bergeron, Joshua Greaves, Alex Fr´echette, Carolyne Pelletier, Eric Thibodeau-Laufer, S´andor Toth, and Sam Work. Tapered off-policy REINFORCE: Stable and efficient reinforcement learning for LLMs, 2025. URL https://arxiv.org/abs/2503.14286.
- [22] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. 2017. URL https://arxiv.org/abs/1707.06347.
- [23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/ 2402.03300.
- [24] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A flexible and efficient RLHF framework. arXiv preprint arXiv:2409.19256, 2024.
- [25] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.
- [26] Zhenpeng Su, Leiyu Pan, Xue Bai, Dening Liu, Guanting Dong, Jiaming Huang, Minxuan Lv, Wenping Hu, Fuzheng Zhang, Kun Gai, and Guorui Zhou. Klear-Reasoner: Advancing reasoning capability via gradient-preserving clipping policy optimization, 2025. URL https: //arxiv.org/abs/2508.07629.
- [27] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.
- [28] Xinyu Tang, Yuliang Zhan, Zhixun Li, Wayne Xin Zhao, Zhenduo Zhang, Zujie Wen, Zhiqiang Zhang, and Jun Zhou. Rethinking sample polarity in reinforcement learning with verifiable rewards, 2025. URL https://arxiv.org/abs/2512.21625.
- [29] Zhiheng Xi, Xin Guo, Yang Nan, Enyu Zhou, Junrui Shen, Wenxiang Chen, Jiaqi Liu, Jixuan Huang, Zhihao Zhang, Honglin Guo, Xun Deng, Zhikai Lei, Miao Zheng, Guoteng Wang, Shuo Zhang, Peng Sun, Rui Zheng, Hang Yan, Tao Gui, Qi Zhang, and Xuanjing Huang. BAPO: Stabilizing off-policy reinforcement learning for LLMs via balanced policy optimization with adaptive clipping, 2025. URL https://arxiv.org/abs/2510.18927.
- [30] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei

- Zhang, Jianxin Yang, Jiaxin Yang, Jingren Zhou, Jingren Zhou, Junyan Lin, Kai Dang, Keqin Bao, Ke-Pei Yang, Le Yu, Li-Chun Deng, Mei Li, Min Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shi-Qiang Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yi-Chao Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. ArXiv, abs/2505.09388, 2025. URL https://api.semanticscholar.org/CorpusID:278602855.
- [31] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, YuYue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=2a36EMSSTp.
- [32] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. PyTorch FSDP: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.
- [33] Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, Fang Wan, and Furu Wei. Geometric-mean policy optimization,

2025. URL https://arxiv.org/abs/2507.20673.

- [34] Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li, Huiqiang Jiang, Junrong Lin, Yuqiong Liu, Hao Lin, Chencan Wu, Feng Hu, An Yang, Jingren Zhou, and Junyang Lin. Stabilizing reinforcement learning with LLMs: Formulation and practices, 2025. URL https://arxiv.org/abs/2512. 01374.
- [35] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. 2025. URL https://arxiv.org/abs/2507.18071.
- [36] Haizhong Zheng, Jiawei Zhao, and Beidi Chen. Prosperity before collapse: How far can offpolicy RL reach with stale data on LLMs? In NeurIPS 2025 Workshop on Efficient Reasoning,

2025. URL https://openreview.net/forum?id=osEqHuHQWK.

- [37] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. SGLang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.
- [38] Yinmin Zhong, Zili Zhang, Xiaoniu Song, Hanpeng Hu, Chao Jin, Bingyang Wu, Nuo Chen, Yukun Chen, Yu Zhou, Changyi Wan, Hongyu Zhou, Yimin Jiang, Yibo Zhu, and Daxin Jiang. StreamRL: Scalable, heterogeneous, and elastic RL for LLMs with disaggregated stream generation. 2025. URL https://arxiv.org/abs/2504.15930.

### VESPO: Variational Sequence-level Soft Policy Optimization Supplementary Material

##### Table of Contents

- A Implicit Proposal Distributions of Existing Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- B Derivation of the Proposal Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- C Proof of the Variance Bound and Uniform Boundedness. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- D Experimental Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- E Length Normalization Introduces Bias . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- F Additional Analysis of Baseline Failure Modes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- G Training Dynamics for Recent Baseline Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- H Code Generation Training Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- I Algorithm Pseudocode. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .21
- J Limitations and Future Directions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

#### A Implicit Proposal Distributions of Existing Methods

We analyze the implicit proposal distributions induced by existing reshaping strategies under the measure-change framework of Section 3.1.

Token-level methods (GRPO). GRPO applies clipping independently at each token position, yielding a gradient of the form

T

Eτ∼µ ϕt(ρt) · A(τ) · ∇log π(yt|x,y<t) , (21)

∇JGRPO =

t=1

where ϕt(ρt) is the clipping function applied to the t-th token ratio. The key observation is that different tokens within the same trajectory receive different weights ϕt(ρt). In contrast, sequencelevel methods assign a single weight ϕ(W) to the entire trajectory:

T

∇log π(yt|x,y<t) . (22)

∇Jseq = Eτ∼µ ϕ(W) · A(τ) ·

t=1

The token-level formulation cannot be expressed as importance sampling toward any single proposal distribution Q, because such a representation requires a uniform weight across all token gradients within each trajectory. This token-wise weighting breaks the coherence of sequence-level credit assignment: tokens in the same trajectory that share a common outcome receive inconsistent gradient signals.

Token-level IS as a First-Order Approximation. We now show that token-level IS is a first-order approximation to sequence-level IS, following Zheng et al. [34]. Consider the unclipped case where ϕt(ρt) = ρt and ϕ(W) = W. The sequence-level gradient (without advantage for clarity) is

T

T

T

∇log π(yt|·) . (23)

∇Jseq = Eτ∼µ W ·

∇log π(yt|·) = Eτ∼µ

ρs ·

t=1

s=1

t=1

When ρt ≈ 1 for all t (near on-policy), we can expand W = s ρs via Taylor series:

T

T

(ρs − 1)(ρs′ − 1) + ··· (24)

ρs ≈ 1 +

(ρs − 1) +

W =

s<s′

s=1

s=1

Substituting into the gradient and keeping only first-order terms in (ρs − 1):

T

T

∇log π(yt|·) (25)

∇Jseq ≈ Eτ∼µ 1 +

(ρs−1)

s=1

t=1

T

. (26)

= Eτ∼µ

Eτ∼µ (ρs−1)∇log π(yt|·)

∇log π(yt|·)

###### +

s,t

t=1

REINFORCE (no IS)

first-order IS correction

The token-level gradient is

T

T

T

Eτ∼µ ρt·∇log π(yt|·) = Eτ∼µ

Eτ∼µ (ρt−1)∇log π(yt|·) .

∇Jtok =

∇log π(yt|·) +

t=1

t=1

t=1

(27) Comparing, we see that token-level IS only retains the diagonal terms (s = t) of the first-order correction, discarding cross-token interactions where s ̸= t. The approximation error is

Eτ∼µ (ρs − 1)∇log π(yt|·) + O (ρ − 1)2 . (28)

∇Jseq − ∇Jtok ≈

s̸=t

This error captures the fact that changing the policy at position s affects the importance of the gradient at position t, a cross-token dependency that token-level methods ignore.

Remark A.1 (When Token-Level Approximation is Reasonable). The approximation ∇Jtok ≈ ∇Jseq is reasonable when:

- 1. Near on-policy: ρt ≈ 1 for all t, so higher-order terms and cross-token terms are small.
- 2. Short sequences: Fewer cross-token pairs (s,t) with s ̸= t means smaller accumulated error.

In practice, RL for LLMs often violates both conditions: off-policy updates are common, and reasoning tasks require long sequences. This motivates sequence-level methods that preserve the full product structure of importance weights.

Length normalization (GSPO). GSPO uses the geometric mean ϕ(W) = W1/T, which induces a proposal distribution that explicitly depends on sequence length:

###### QGSPO(τ) ∝ µ(τ)1−1/T · π(τ)1/T. (29)

As T → ∞, QGSPO → µ, meaning longer sequences receive vanishingly small corrections toward the target policy. See Section E for a detailed analysis of the length-dependent bias this introduces.

Sequence-level hard clipping. If one were to apply hard clipping at the sequence level (truncating W at threshold c), the reshaping function would be ϕ(W) = min(W,c), inducing

###### Qclip(τ) ∝ min π(τ),c · µ(τ) . (30)

This is a truncated distribution: trajectories with W > c are capped rather than weighted proportionally. The discontinuity at W = c can cause optimization difficulties when trajectories cross the boundary during training. Note that standard PPO applies clipping at the token level rather than the sequence level, combining the issues of both token-level weighting and hard truncation.

- B Derivation of the Proposal Distribution We derive the closed-form solution to the constrained optimization problem in Equation (14).

- Proposition B.1 (Solution to the Constrained Problem). The solution to

DKL(Q∥µ) − α EQ[log W]

min

Q

s.t. EQ[W] ≤ C, Q(τ)dτ = 1 (31)

is given by

1 Z

Q∗(τ) =

µ(τ)W(τ)α exp(−λW(τ)), (32)

where λ ≥ 0 is the Lagrange multiplier for the moment constraint and Z is the normalization constant.

Proof. The Lagrangian, after expanding DKL(Q∥µ) = Qlog(Q/µ)dτ, is

###### L(Q,λ,γ) = Q(τ) log Q(τ) − log µ(τ) − α log W(τ)

+ λW(τ) + γ dτ − γ − λC. (33) Taking the functional derivative with respect to Q and setting it to zero:

Solving for Q:

δL δQ

= log Q(τ) + 1 − log µ(τ) − α log W(τ) + λW(τ) + γ = 0. (34)

log Q∗(τ) = log µ(τ) + α log W(τ) − λW(τ) + const (35) Q∗(τ) ∝ µ(τ)W(τ)α exp(−λW(τ)). (36)

Comparing with Q(τ) = Z1 µ(τ)ϕ(W(τ)) from Equation (10), we identify

###### ϕ(W) = Wα · exp(−λW). (37)

| |
|---|

Equivalent dual-KL form. Using the identity DKL(Q∥π) = DKL(Q∥µ) − EQ[log W], the objective in Equation (11) can equivalently be written as

###### J (Q) = (1−α)DKL(Q∥µ) + α DKL(Q∥π), (38)

which exposes the geometric interpretation: the proposal Q interpolates between µ (α=0) and π (α=1). Both forms yield the same Q∗; we use the single-KL form in the main text because DKL(Q∥µ) retains unit weight in the deployed regime α ≥ 1.

Surrogate Objective. The reshaped gradient (Equation (4)) corresponds to maximizing an implicit surrogate. Using ∇θW = W∇θ log πθ, the estimator can be rewritten as

###### Eτ∼µ ϕ(W) · A · ∇log πθ = Eτ∼µ ϕ(WW) · A · ∇W . (39)

Defining f by f′(W) = ϕ(W)/W, this equals ∇θEτ∼µ[f(W)A(τ)], since A(τ) does not depend on θ. Hence VESPO implicitly maximizes

###### JVESPO(θ) = Eτ∼µ f(W(θ))A(τ) . (40)

For ϕ(W) = Wα exp(−λW), f′(W) = Wα−1 exp(−λW) integrates to the lower incomplete gamma function:

f(W) =

1 λα

γ(α,λW), γ(a,x) =

x

ta−1e−t dt. (41)

0

This form is smooth and infinitely differentiable, and saturates as W → ∞, providing a principled soft alternative to hard clipping.

Shifted Form. In practice, we use the shifted form ϕ(W) = Wc

exp(c2(1 − W)), which satisfies ϕ(1) = 1. This ensures that on-policy samples receive unit weight. We treat (c1,c2) as tunable hyperparameters, allowing flexibility beyond the specific values implied by the variational derivation.

1

#### C Proof of the Variance Bound and Uniform Boundedness

We prove Theorem 3.1 from Section 3.4 and the supporting boundedness result for the deployed kernel.

Proof of Theorem 3.1. By Equation (10), Q(τ) = µ(τ)ϕ(W(τ))/Z, so for any measurable g : R>0 → R:

Eµ ϕ(W) · g(W) = Z · EQ g(W) . (42)

Setting g = ϕ gives Eµ[ϕ(W)2] = Z · EQ[ϕ(W)]. Since ϕ(w) = (ϕ(w)/w) · w ≤ K · w for all w > 0 where K = supw>0 ϕ(w)/w, we have

EQ[ϕ(W)] ≤ K · EQ[W] ≤ K · C, (43) using the moment constraint EQ[W] ≤ C. Combining yields Eµ[ϕ(W)2] ≤ Z · K · C. Finiteness and closed form of K. We have ϕ(w)/w = wc

1−1 exp(c2(1 − w)). As w → 0+, this tends to ∞ when c1 < 1 (since wc

1−1 → ∞), to exp(c2) when c1 = 1, and to 0 when c1 > 1. As w → ∞, the exponential dominates and the ratio tends to 0 for any c2 > 0. Hence K < ∞ iff c1 ≥ 1.

For c1 > 1, h(w) := (c1 − 1)log w + c2(1 − w) has derivative h′(w) = (c1 − 1)/w − c2, vanishing at w∗ = (c1 − 1)/c2 > 0. The second derivative −(c1 − 1)/w2 < 0 confirms an interior maximum. Substituting:

K = (w∗)c

1−1 exp c2(1 − w∗) =

c1 − 1 c2

c1−1

exp(c2 − c1 + 1), (44)

yielding the stated formula. For c1 = 1 the supremum K = exp(c2) is approached as w → 0+.

| |
|---|

- Proposition C.1 (Uniform Boundedness of ϕ). For any c1,c2 > 0, the kernel ϕ(W) =

exp(c2(1 − W)) satisfies

Wc

1

c1

c1 c2

exp(c2 − c1), (45)

sup

ϕ(W) = ϕmax =

W>0

attained at W∗∗ = c1/c2.

Proof. log ϕ(W) = c1 log W + c2(1 − W) has derivative c1/W − c2, vanishing at W∗∗ = c1/c2. The second derivative −c1/W2 < 0 confirms a maximum. Substituting yields ϕmax = (c1/c2)c

exp(c2 − c1).

| |
|---|

1

Numerical values at practical settings. For (c1,c2) = (2,3): K = (1/3)e2 ≈ 2.46 and ϕmax = (2/3)2e ≈ 1.21. For (c1,c2) = (3,2): K = 1 and ϕmax = (3/2)3e−1 ≈ 1.24. The two propositions together establish that the deployed kernel offers (i) tight second-moment control under the moment constraint and (ii) uniformly bounded gradient contribution from any single off-policy sample, regardless of staleness.

#### D Experimental Details

Training infrastructure. All experiments run on 32 NVIDIA H20 GPUs with veRL [24], using vLLM 0.11.0 [14] for rollout inference, FSDP [32] for the dense models (Llama-3.2-3B-Instruct, Qwen3-8B-Base), and Megatron [25] for the MoE model (Qwen3-30B-A3B-Base). Each run executes 1,500 gradient steps with maximum sequence length 16,384 tokens and 8 responses per query.

Reward and evaluation protocol. Math reward is verifier-based via Math-Verify [12]; code reward is execution-based (test case pass/fail). Math evaluation reports avg@k with k=32 for AIME, k=16 for AMC, and k=4 for MATH-500; code evaluation reports pass@10. The best checkpoint per method is selected by average accuracy across the corresponding evaluation suite.

Baseline hyperparameters. We use the official hyperparameters published with each method:

- • GRPO [23]: token-level PPO clipping with asymmetric bounds (0.2,0.28) following DAPO [31].
- • GSPO [35]: sequence-level with 1/T length normalization, clipping bounds (3e-4, 4e-4).
- • SAPO [7]: token-level soft gating, τpos=1.0, τneg=1.05.

VESPO uses asymmetric (c1,c2) = (2.0,3.0) for A>0 and (3.0,2.0) for A<0, applied identically across all models and training tasks (no retuning between math and code).

Staleness simulation. We fix mini-batch size 256 and vary global batch size to obtain staleness ratio N = gbs/mbs. Each rollout batch is split into N mini-batches for sequential gradient updates, so later mini-batches use parameters increasingly stale relative to the rollout policy. Primary experiments use N=8; staleness ablations span N ∈ {4,8,16,32,64}.

#### E Length Normalization Introduces Bias

We analyze how length normalization in sequence-level importance sampling introduces lengthdependent bias that conflates distinct trajectories.

Length-Dependent Bias in GSPO. GSPO uses ϕ(W) = W1/T, which induces a proposal distribution that explicitly depends on the sequence length T:

QGSPO(τ) ∝ µ(τ)1−1/T · π(τ)1/T. (46)

As T → ∞, QGSPO → µ: the normalized weight converges to exp(E[log ρt]), a constant independent of the specific trajectory. This causes two fundamental problems:

- 1. Signal dissipation: All weights collapse toward a constant, losing discriminative power.
- 2. Conflation of distinct sequences: Sequences with identical per-token statistics but different lengths receive identical weights, despite having different true importance weights.

Proposition E.1 (Conflation under Length Normalization). For any two sequences τ1, τ2 with lengths T1 ̸= T2, if logW

T1 = logW

T2 , then GSPO assigns identical weights: ϕGSPO(W1) = ϕGSPO(W2). However, their true importance weights differ: W1 = eT

2

1

2r¯ for r¯ = logW

T1 = logW

1r¯ vs W2 = eT

T2 .

1

2

This conflation is problematic: a short sequence that is moderately off-policy and a long sequence that is severely off-policy may receive identical gradient weights, even though their contributions to the policy gradient should differ substantially.

VESPO Avoids Length-Dependent Bias. VESPO uses ϕ(W) = Wc

exp(c2(1−W)) without any length normalization. The reshaping function depends only on the sequence-level importance weight W, not on the sequence length T. Two sequences with the same average per-token log-ratio r¯ but different lengths receive different weights: the longer sequence has W = eTr¯, which is appropriately transformed by ϕ. This preserves the discriminative power of the importance weight while the soft-shaping kernel controls variance through exponential suppression of extreme weights.

1

#### F Additional Analysis of Baseline Failure Modes

This section provides supplementary visualizations comparing N=4 vs N=8 for each baseline (the comprehensive training dynamics figure across all N values is in the main text, Figure 4).

#### G Training Dynamics for Recent Baseline Comparison

Figure 11 shows the training dynamics of TOPR, CISPO, BAPO, and VESPO under the matched setup of Section 4.4. CISPO collapses around step 280 due to insufficient sequence-level variance control; BAPO exhibits entropy explosion after step 1,100 as adaptive clip bounds over-relax; TOPR converges slowly with suppressed response length. VESPO is the only method that maintains stable training across all metrics.

5000

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | |
|---|---|
| | |
| | |
| | |

0.6

100

0.004

4000

0.5

10 1

3000

0.4

0.002

2000

N=4 N=8

0.3

10 2

0.000

1000

0.2

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

- 0

200

400

600

Gradient Norm

0 500 1000 1500

Steps

0.0

0.5

PG Loss

0 500 1000 1500

Steps

0.000

0.001

0.002

0.003

Clip Fraction

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

0 500 1000 1500

Steps

0.20

0.15

0.10

0.05

Advantage

- Figure 8: GRPO: N = 4 (blue) vs N = 8 (orange). Entropy decreases more rapidly at N = 4, limiting exploration.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 500 1000 1500

0.0

0.2

0.4

0.6

Training Reward

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 500 1000 1500

0

1000

2000

3000

Response Length

| | |
|---|---|
| | |

0 500 1000 1500

10 1

100

Entropy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 500 1000 1500

0

5

KL Divergence

| || |
|---|
<br><br>Co<br><br>N=|llapse 4| | |
|---|---|---|---|---|
| |N=|8| | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 500 1000 1500

Steps

0

25000

50000

75000

100000

Gradient Norm

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 500 1000 1500

Steps

0

200

400

PG Loss

0 500 1000 1500

Steps

0.0

0.2

0.4

0.6

0.8

Clip Fraction

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 500 1000 1500

Steps

0.1

0.0

0.1

Advantage

- Figure 9: GSPO: N = 4 (blue) vs N = 8 (orange). Response length spikes above 3,000 tokens at N = 4 before collapse.

H Code Generation Training Details and Dynamics

Setup recap. We train Qwen3-30B-A3B-Base on PRIME-RL/Eurus-2-RL-Data [3] using the same VESPO hyperparameters (c1,c2) = (2,3)/(3,2) as math, without any retuning. The reward is execution-based (test case pass/fail), structurally distinct from the verifier-based math reward. We evaluate on HumanEval+, MBPP+ [17], and LiveCodeBench v6 [13], reporting pass@10 (Table 4). VESPO is the best on all three benchmarks with zero tuning, demonstrating that the variational framework yields hyperparameters that generalize across reward modalities.

Training dynamics. Figure 12 shows the training dynamics on PRIME-RL/Eurus-2-RL-Data. VESPO maintains the highest training reward with stable entropy and response length throughout

- 1,500 steps. SAPO trains well initially but collapses around step 1,250. GRPO and GSPO converge slowly to lower reward levels.

0 500 1000 1500

Steps

- 100

- 101

15

15000

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
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

0.6

10

10000

0.4

5

5000

10 1

0.2

0

0

0.0

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

0 500 1000 1500

- 0

500

1000

Gradient Norm

Collapse

N=4 N=8

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0 500 1000 1500

Steps

1.5

1.0

0.5

0.0

0.5

PG Loss

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

0 500 1000 1500

Steps

1.5

2.0

Gate Mean

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 500 1000 1500

Steps

0.1

0.0

0.1

0.2

Advantage

Figure 10: SAPO: N = 4 (blue) vs N = 8 (orange). Training reward collapses at N = 8 due to insufficient suppression for negative advantages.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 500 1000 1500

Steps

0.0

0.2

0.4

0.6

0.8

Training Reward

0 500 1000 1500

Steps

10 1

100

Entropy

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 500 1000 1500

Steps

0

2000

4000

6000

8000

10000

Response Length

0 500 1000 1500

Steps

0

25

50

75

100

125

Gradient Norm

GRPO GSPO SAPO VESPO CISPO TOPR BAPO

Figure 11: Training dynamics of recent importance weight reshaping methods on Qwen3-30B-A3BBase (N=8). CISPO collapses early; BAPO experiences entropy explosion; TOPR converges slowly with suppressed length. VESPO is the only method maintaining stable training across all metrics.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 500 1000 1500

Steps

0.0

0.1

0.2

0.3

0.4

0.5

0.6

Training Reward

| | |
|---|---|
| | |
| | |

0 500 1000 1500

Steps

10 1

100

Entropy

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 500 1000 1500

Steps

500

1000

1500

2000

2500

3000

Response Length

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

0 500 1000 1500

Steps

0

5

10

15

20

25

Gradient Norm

GRPO GSPO SAPO VESPO

Figure 12: Training dynamics on code generation (PRIME-RL/Eurus-2-RL-Data, Qwen3-30B-A3BBase, N=8). VESPO is the only method maintaining stable training; SAPO collapses around step

- 1,250.

0 500 1000 1500

Steps

#### I Algorithm Pseudocode

- 1 def compute_policy_loss_vespo(log_pi , log_mu , advantages , mask , c_pos , c_neg):

- 2 """

- 3 Args:

- 4 log_pi: log probs from current policy , shape (batch , seq_len)

- 5 log_mu: log probs from behavior policy , shape (batch , seq_len)

- 6 advantages: sequence -level advantages , shape (batch ,)

- 7 mask: response mask , shape (batch , seq_len)

- 8 c_pos: (c1, c2) for positive advantages

- 9 c_neg: (c1, c2) for negative advantages

- 10 """

- 11 # Sequence -level IS ratio in log -space (true product , no length norm )

- 12 log_ratio = log_pi - log_mu

- 13 seq_log_w = (log_ratio * mask).sum(dim=-1) # (batch ,)

- 14 W = exp(seq_log_w)

- 15

- 16 # Asymmetric hyperparameter selection

- 17 c1 = where(advantages >= 0, c_pos[0], c_neg [0])

- 18 c2 = where(advantages >= 0, c_pos[1], c_neg [1])

- 19

- 20 # VESPO kernel in log -space , normalized so phi(1) = 1

- 21 log_phi = c2 + c1 * seq_log_w - c2 * W

- 22 phi = exp(log_phi).detach() # gradient scaling only

- 23

- 24 # Policy gradient loss: grad = -phi * A * grad_log_pi

- 25 loss = -phi.unsqueeze(-1) * advantages.unsqueeze(-1) * log_pi

- 26 return aggregate(loss , mask) VESPO Policy Loss

#### J Limitations and Future Directions

Hyperparameter specification. While the variational framework provides the kernel form ϕ(W) = Wc

exp(c2(1 − W)), the specific values (c1,c2) require user choice. Our analysis identifies c1 ≥ 1 as the principled inverse-temperature regime under which the variance bound (Theorem 3.1) is non-vacuous, and we use (c+1 ,c+2 ) = (2,3) and (c−1 ,c−2 ) = (3,2) across all settings without retuning between math and code. Adaptive schemes—scheduling (c1,c2) along training, automatic per-task search, or learning the hyperparameters from data—could further improve performance and remain an open direction.

1

Reward modality scope. Our experiments cover verifiable-reward settings: math reasoning with verifier-based reward (Math-Verify) and code generation with execution-based reward (test pass/fail). Effectiveness on preference-based rewards (e.g., RLHF, RLAIF), shaped/dense rewards, or openended creative generation remains to be explored. We expect VESPO’s smooth variance control to remain beneficial in these regimes since the underlying off-policy IS challenge is the same, but direct empirical validation is needed.

Model scale. Our largest evaluated model is Qwen3-30B-A3B-Base (30B total parameters with MoE routing). Scaling to substantially larger models (>100B) is not directly evaluated due to compute constraints. The observed monotonic trend—larger gains on the MoE Qwen3-30B-A3B-Base than on the dense Qwen3-8B-Base or Llama-3.2-3B-Instruct—suggests favorable scaling behavior, but direct verification at frontier scales remains future work.

