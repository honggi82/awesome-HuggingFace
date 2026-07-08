# arXiv:2605.27079v1[cs.LG]26May2026

## Trust Region Q Adjoint Matching

Yonghoon Dong1 yonghoon.dong@kaist.ac.kr

Kyungmin Lee1 kyungmnlee@kaist.ac.kr

Changyeon Kim1 changyeon.kim@kaist.ac.kr

Jaehyuk Kim2 waewae1@snu.ac.kr

Jinwoo Shin1,3 jinwoos@kaist.ac.kr

### Abstract

Off-policy reinforcement learning of pretrained flow policies remains challenging due to the instability of optimization arising from the multi-step sampling process. Recently, Q-learning with Adjoint Matching (QAM) addressed this issue by reformulating into a memoryless stochastic optimal control (SOC) problem with a learned critic. However, QAM inherits a fundamental fragility of critic-guided improvement: small critic errors are amplified when critics are ill-conditioned, often leading to model collapse. This paper introduces Trust Region Q-Adjoint Matching (TRQAM), a stable off-policy fine-tuning algorithm that adaptively controls the path-space KL with pretrained flow policies through projected dual descent. Specifically, we optimize the trust-region parameter λ in SOC dynamics, and theoretically show that the path-space KL can be represented by a closed-form function of λ. As a result, our method can precisely control the exact deviation from pretrained flow policies, achieving stable off-policy RL. Through experiments on 50 OGBench tasks, TRQAM consistently outperforms prior arts in both offline RL and offline-to-online RL. In particular, TRQAM achieves an overall success rate of 68% in offline RL, substantially improves the strongest baseline at 46%.

### 1 Introduction

Recently, flow matching policies [2, 7, 27, 29] have emerged as a promising approach to model rich and diverse action distributions, enabling high-capacity behavior generation beyond conventional uni-modal Gaussian policies. A pretrained flow policy captures useful skills, behavioral constraints, and broad coverage over plausible actions, making it an attractive prior for downstream off-policy RL fine-tuning [4, 11, 17–20, 45, 48, 51, 57, 60].

However, as the flow policy is defined implicitly through a multi-step denoising process, gradientbased policy improvement requires differentiating through the multi-step sampling chain, making direct backpropagation expensive and unstable [39, 59]. Existing approaches sidestep this through residual-style methods that keep the pretrained policy frozen and learn an additive residual to its actions [11, 55], or noise-space RL methods that freeze the pretrained flow policy and run actor-critic over its input noise [51]. Yet, both have fundamental limits: residual methods correct only at the action level, ignoring the multi-step generative dynamics, while noise-space methods are bounded by the expressivity of the frozen flow policy.

Recently, Q-learning with Adjoint Matching (QAM) [25] addresses this by reformulating fine-tuning as a memoryless stochastic optimal control (SOC) problem: QAM uses a learned critic to control the sampling process toward higher-value actions via adjoint matching. While this resolves the multi-step sampling instability, critic-induced instability still remains. In off-policy RL, the learned critic is inevitably imperfect, thus the approximation errors compound through TD bootstrapping, where each value update depends on the critic’s noisy estimate at the next state, producing systematic

1KAIST AI 2Seoul National University 3RLWRLD Code: https://github.com/yonghdong/trqam Blog: https://yonghdong.github.io/blog/trqam/

Preprint.

[Figure 1]

Pretrained flow policy

Naive critic-guided fine-tuning causes drift

TRQAM internalizes trust-region control

CGQL-L 23 FQL 28 IFQL 35

controlled deviation

QAM 35

𝜆

QAM-E 45 DSRL 46 TRQAM 68

useful pretrained policy

adapts via dual descent

destructive drift e.g., QAM, QAM-E

(a) Higher reward, without destroying the pretrained policy (b) Offline RL success rate on OGBench

- Figure 1: TRQAM builds adaptive trust-region control into the SOC dynamics. (a): Methods whose optimum admits an exponentially-tilted form (e.g., QAM, QAM-E [25]) suffer from destructive drift, where small critic errors can be exponentially amplified into large deviations from the pretrained prior (Lemma 1). TRQAM regulates this deviation through a trust-region parameter λ internalized in the SOC sampling dynamics. (b): Offline RL success rate across 50 OGBench [38] tasks. TRQAM outperforms adjoint-matching baselines (QAM, QAM-E) and other flow-policy fine-tuning paradigms (FQL [39], IFQL [22], DSRL [51], CGQL-L [8]) which lack such convergence guarantees.

overestimation [12]. Critic-guided policy updates can then amplify these errors into large deviations from the pretrained prior (see Lemma 1). QAM [25] acknowledges this and apply gradient clipping as a partial remedy, while calling for a more principled method beyond this heuristics. We empirically observe that gradient clipping does not prevent this fragility: the adjoint loss can become unstable and collapse task performance on Robomimic [32] (Figures 2, 10, and 11). Thus, the central challenge becomes how to improve downstream performance without destructive drift from the pretrained prior, similar to the trust-region principle in on-policy RL [46, 47].

We propose Trust Region Q-Adjoint Matching (TRQAM), an algorithm for stable off-policy finetuning of pretrained flow policies. TRQAM introduces a trust-region parameter λ directly into the stochastic optimal control (SOC) sampling dynamics and adapts it via projected dual descent to enforce a prescribed KL bound between the fine-tuned and pretrained policies. Our central theoretical result, proved using Girsanov’s theorem, shows that scaling the diffusion coefficient by √

λ makes the path-space KL between the controlled and pretrained sampling processes an explicit closed-form function of λ. As a result, the dual update enforces the target KL bound directly through the sampling dynamics, rather than softly imposing the constraint through a conventional loss-level KL regularizer.

This distinction matters in practice. Conventional loss-level KL regularization only competes with critic guidance at the loss level, allowing strong critic signals to push the realized KL far beyond the target bound and leave the policy vulnerable to collapse. In contrast, TRQAM tightly tracks the target bound throughout both offline and online training (Figures 5, 14, 15, and 16). The prescribed target KL bound therefore provides a practical control over how much the fine-tuned policy can deviate from the pretrained policy, and its best setting varies systematically with task structure (Section 4.3). Through experiments on 50 OGBench tasks, TRQAM consistently outperforms prior methods in both offline RL and offline-to-online RL. In particular, TRQAM achieves an overall offline RL success rate of 68%, substantially outperforming the strongest baseline at 46% (Table 1).

Contributions. We highlight the key contributions of our paper below:

- • We identify the exponential amplification of critic errors as a fundamental fragility of fixedtemperature adjoint matching, formalized by Lemma 1 and confirmed empirically on Robomimic.
- • We prove that scaling the diffusion coefficient by √

λ makes the path-space KL an exact function of λ via Girsanov (Theorem 1), turning λ into a principled trust-region parameter.

- • We propose Trust Region Q-Adjoint Matching (TRQAM), which internalizes λ inside the SOC sampling dynamics and adapts it via projected dual descent to enforce a target KL bound at the sampling level rather than as a loss-level penalty.
- • On 50 OGBench tasks, TRQAM consistently outperforms prior arts on offline RL and offline-toonline RL. Especially, TRQAM achieves an overall success rate of 68% in offline RL, demonstrating superior performance compared to 46% of strongest baseline.

### 2 Background

Reinforcement Learning. We consider a Markov Decision Process (MDP) [49] (S,A,P,r,γ,ρ0), where S is the state space, A is the action space, r(s,a) : S × A → R is a reward function,

P(s′ |s,a) : S × A → ∆(S) is a transition function, ρ0 is the initial state distribution, and γ ∈ [0,1) is a discount factor. The objective is to learn a policy π : S → ∆(A) that maximizes the expected

0,π,P[ ∞t=0 γtr(st,at)], where t indexes environment timesteps. To this end,

discounted return Eρ

we learn an action-value function Qπ(s,a) = Eπ[ ∞k=0 γkr(st+k,at+k)|st = s,at = a] used as a critic that guides policy improvement. We majorly focus on off-policy fine-tuning of a pretrained

policy πbase(· | s), where transitions (s,a,r,s′) used for updates are drawn from a replay buffer D collected by behavior policies different from the current πθ.

Flow matching and flow policy. Flow Matching [2, 27] learns a velocity field vθ(x,τ) that transports samples from a simple source distribution p0 = N(0,Id) to a target distribution p1 through an ODE: dXτ = vθ(Xτ,τ)dτ, X0 ∼ p0, X1 ∼ p1.

A standard choice is the optimal transport (OT) path, with straight-line interpolation Xτ = (1 − τ)X0 + τX1 between endpoint pairs (X0,X1) ∼ p0 ⊗ p1 and target velocity E[X1 − X0|Xτ = x]. Then, the velocity field is trained by regressing vθ against target velocity X1 − X0, giving

0, X1 vθ(Xτ,τ) − (X1 − X0) 2 .

LFM(θ) = Eτ, X

Note that a flow policy applies flow matching to model policy: given a state s, a flow policy πθ(· | s) samples actions by integrating dXτ = vθ(Xτ,τ;s)dτ from X0 ∼ N(0,Id).

Stochastic optimal control and Q adjoint matching. Stochastic optimal control (SOC) is a framework that fine-tunes a pretrained flow policy by adding a drift perturbation u to its sampling dynamics, steering trajectories toward higher-reward regions without backpropagating through the multi-step sampling chain, a problem similar to backpropagation through time in RNNs [39, 59]. Domingo-Enrich et al. [10] showed that this SOC objective can be solved efficiently via a lean adjoint ODE, leading to the adjoint matching algorithm.

To formulate this, we first replace the deterministic flow ODE with an equivalent SDE whose distribution at every timestep τ ∈ [0,1] coincides with that of the ODE [31]. While each trajectory might differ, the random variable Xτ at each timestep has the same distribution under both processes, including the terminal X1 ∼ πbase. Following Domingo-Enrich et al. [10], we adopt the memoryless OT schedule, and apply the SOC parameterization of Domingo-Enrich et al. [9] that explicitly separates a scalar √

λ from the diffusion coefficient, giving √

λσ(τ) = 2(1 − τ)/τ. This gives the base and controlled SDEs

√

dXτbase = b(Xτbase,τ)dτ +

λσ(τ)dBτ, (1) dXτu = b(Xτu,τ) + σ(τ)u(Xτu,τ) dτ +

√

λσ(τ)dBτ, (2) where the base SDE generates the pretrained policy and the control u steers the process away from it. Given a terminal cost g : Rd → R, SOC seeks the control u by optimizing

1

- 1

- 2

u(Xτu,τ) 2 dτ − g(X1u) subject to Equation (2). (3)

EX∼Pu

min

u

0

From a reinforcement learning perspective, taking X1u as the action at state s and critic Qπ(s,·) as the terminal cost g, the control u steers the pretrained policy toward critic-preferred actions, with

the quadratic term penalizing deviation from πbase [10]. This instantiation is Q Adjoint Matching (QAM) [25], which solves (3) via adjoint matching [10]: the adjoint aτ is computed by integrating

1 τ

aτ−h = aτ + h a⊤τ ∇x 2vbase(Xτ,τ) −

Qπ(s,X1), (4)

Xτ , a1 = −∇x1

backwards in time, and the fine-tuned velocity field vθft, whose deviation from vbase parameterizes the control u, is updated by minimizing the adjoint-matching loss

2

2 σ(τ)

vθft(Xτ,τ) − vbase(Xτ,τ) + σ(τ) aτ

, (5)

LAdj-Match(θ) =

τ∈{0,...,1−h}

without backpropagating through the sampling chain. However, QAM uses standard SOC dynamics without the √

λ-scaling, corresponding to the special case λ = 1 in our framework.

TRQAM QAM QAM-E

Robomimic–can

| | | |
|---|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
| | | |

100

1020

| |
|---|

Successrate(%)

75

1015

Adjointloss

| |
|---|

1010

50

105

25

| |
|---|

100

| |
|---|

| |
|---|

| |
|---|

0

0 5 10 15

0 5 10 15

Training Steps (×105)

- Figure 2: Empirical fragility of fixed-temperature adjoint matching. On Robomimic-can, the adjoint-matching loss in QAM and QAM-E grows above 1020 even with gradient clipping (min– max across seeds), driving task success from over 80% to near zero (±1 standard deviation), while TRQAM remains stable. This collapse persists across most hyperparameter settings we tested on both Robomimic-lift and Robomimic-can; see Appendix Figures 10 and 11 for the full sweep.
- 3 Method

In this section, we introduce Trust Region Q-Adjoint Matching (TRQAM), a framework for stable off-policy fine-tuning of pretrained flow policies that adapts a trust-region parameter λ inside the stochastic optimal control sampling dynamics. By scaling the diffusion coefficient by √

λ, TRQAM makes the path-space KL between the fine-tuned and pretrained processes an exact function of λ via Girsanov, turning a prescribed KL bound εKL into a structural constraint enforced at the sampling level. We develop our core contribution through investigating the following questions:

- • Why fixed λ is fragile in off-policy RL? (Section 3.1)
- • What does λ control when internalized in the SOC sampling dynamics? (Section 3.2)
- • How do we adapt λ throughout training? (Section 3.3)
- • Why must λ be internalized in the SOC sampling dynamics rather than added as a conventional KL regularization? (Section 3.4)

#### 3.1 Why fixed λ is fragile in off-policy RL?

In off-policy RL, the learned critic is inevitably imperfect: approximation error compounds through bootstrapping and replay, and is especially severe under distributional shift [12, 23]. The central risk of critic-guided policy improvement is that small critic errors can induce large policy deviations. We formalize this in the following lemma, which applies to any updated policy that exponentially tilted from a base policy. Note that this includes QAM, since solving its memoryless SOC objective yields a terminal policy of this form [10, 25].

Lemma 1 (Exponential amplification of critic errors). Fix a state s ∈ S and let Q, Q : A → R satisfy ∥Q − Q∥∞ ≤ ε. Define the corresponding exponentially tilted distributions

πQ(a | s) ∝ πbase(a | s)eβQ(a), π Q(a | s) ∝ πbase(a | s)eβ Q(a), where β > 0 is the inverse temperature. Then the following inequalities hold:

- 1

- 2

DKL(πQ ∥π Q) ≤ 2βε, TV(πQ,π Q) ≤

e2βε − 1 .

- Proof. See Appendix D.1 for the full proof.

| |
|---|

The total-variation bound is exponential in βε: large β amplifies critic errors into large policy deviations, while small β suppresses both critic errors and useful improvement. The lemma thus formalizes a fundamental tension in critic-guided updates: no single fixed β can simultaneously exploit a reliable critic and protect against an unreliable one. We observe this empirically on Robomimic [32]: fixed-temperature QAM and QAM-E exhibit diverging adjoint loss and collapsing

task success even with gradient clipping (Figure 2). An analogous tension arises in our SOC setting, where λ governs the size of the trust region: small λ permits aggressive deviation from πbase (corresponding to large β), while large λ keeps the controlled sampler close to it. Before we can adapt λ, we need a quantitative link between λ and the deviation from πbase, which we establish next.

#### 3.2 λ as a trust-region parameter

To adapt λ in a principled way, we first establish what λ controls. Note that the diffusion coefficient is scaled by √

λ in Equation (2), following the parameterization of SOCM [9]. Its consequence under change-of-measure, however, has not been made explicit. We now derive this consequence: under this scaling, Girsanov’s theorem yields an exact identity between the quadratic control cost and the path-space KL between the fine-tuned and base trajectory distributions, in which λ appears explicitly as the inverse coefficient.

Theorem 1 (SOC control cost = path-space KL). Let Pu and Pbase denote the distributions over trajectories induced by the controlled dynamics (2) and the base dynamics (1). Then,

DKL Pu ∥Pbase = EX∼Pu

- 1

- 2λ

1

∥u(Xτ,τ)∥2 dτ . (6)

0

- Proof. See Appendix D.2 for the full proof.

| |
|---|

While Theorem 1 ties λ to the path-space KL, the quantity we ultimately care about is the deviation of the terminal action distribution πθ(· | s) from πbase(· | s). The following proposition shows that controlling the path-space KL suffices: it provides an upper bound on the terminal KL.

Proposition 1 (Terminal KL upper-bounded by path-space KL). Let Pu and Pbase denote the distributions over trajectories induced by the controlled dynamics (2) and the base dynamics (1), and let πθ(· | s) and πbase(· | s) denote the corresponding terminal action distributions at τ = 1. Then,

DKL πθ(· | s)∥πbase(· | s) ≤ DKL Pu ∥Pbase . (7)

- Proof. See Appendix D.3 for the full proof.

| |
|---|

Informally, the three results form a chain that links the trust-region parameter λ to the amplification of critic errors. Theorem 1 ties λ to the path-space KL between the controlled and base trajectories. Proposition 1 shows that this path-space KL upper-bounds the terminal-policy KL DKL(πθ∥πbase). Lemma 1 bounds this terminal-policy KL by 2βε, where β is the inverse temperature in the exponential tilting of πbase by the critic and ε is the critic approximation error.

Remark: Connection between trust-region parameter λ and inverse temperature β

≲

DKL(Pu∥Pbase) path-space KL

.

1/λ ∝

≥

DKL(πθ∥πbase) terminal KL

βε

Thm. 1

Prop. 1

Lem. 1

critic-error bound

Increasing λ shrinks terminal KL, effectively reducing the critic guidance strength β and tightening the bound on critic-error amplification. A single scalar λ therefore adaptively balances exploiting the critic and staying close to πbase.

#### 3.3 Adaptive λ via projected dual descent

To keep the realized path-space KL within a target bound εKL, we need a tractable KL estimator and a principled rule for updating λ based on it. We use the fact that the discretized memoryless

OT sampler with step size h and diffusion schedule g(τ) = 2(1 − τ)/τ is a Markov chain whose Gaussian transitions share the same covariance, so per-step KL divergences admit a closed form.

#### Algorithm 1 QAM

Algorithm 2 TRQAM (ours)

Require: vbase, training step N

Require: vbase, training step N, KL budget εKL

- 1: Init vθft ← vbase, λ = 1
- 2: for n = 0, . . . , N − 1 do
- 3: Sample X via Eq. (2) with vθft, λ
- 4: Solve adjoint ODE (4)
- 5: θ ← θ − ∇θLAdj-Match via Eq. (5) with σ
- 6:
- 7:
- 8: end for

- 1: Init vθft ← vbase, λ0, D0, dual stepsize ηλ

- 2: for n = 0, . . . , N − 1 do
- 3: Sample X via Eq. (2) with vθft, λn
- 4: Solve adjoint ODE (4)
- 5: θ ← θ − ∇θLAdj-Match via Eq. (5) with σn
- 6: Estimate Dn via Eq. (8); EMA Dn

- 7: λn+1 ← max{0, λn + ηλ(Dn − εKL)}

- 8: end for

Summing these per-step KLs along a trajectory approximately recovers the path-space KL, which we estimate via Monte Carlo over sampled trajectories (derivation in Appendix E):

Dn = EX∼Pu

K−1

2h g(τk)2

k=0

,τk) 2 , (8)

vθft(Xτ

,τk) − vbase(Xτ

k

k

To reduce variance, we smooth Dn with an exponential moving average: Dn ← (1 − ρ)Dn−1 + ρ Dn. Given this estimator, we adapt λ by interpreting it as the dual variable of a KL-constrained improvement problem maxu E[Qπ(X1u)] subject to DKL(Pu∥Pbase) ≤ εKL, and apply projected dual descent with a fixed step size ηλ > 0 (derivation in Appendix F):

λn+1 ← max 0, λn + ηλ (Dn − εKL) . (9)

When the realized KL exceeds the bound, λ rises and the controlled dynamics become more conservative; when it falls below, λ decreases and allows more aggressive improvement. Algorithm 2 summarizes the resulting TRQAM update, where the adaptive trust-region components are highlighted in blue. Full algorithm can be found in Appendix A.

- 3.4 Internal vs. external KL regularization The dual update in Equation (9) can be combined with the SOC objective in two different approaches:

LAdj-Match(θ) s.t. SDE uses √

vs. min

. (10)

LAdj-Match(θ) + λ · Dn(θ)

min

λσ(τ)

θ

θ

External: λ is a regularization weight

Internal (TRQAM): λ appears in the SOC sampling dynamics

For the internal form (right), √

λσ(τ) = 2(1 − τ)/τ is fixed by the OT schedule, thus adjusting λ changes σ(τ) and reshapes the entire controlled SDE, including its drift term b(x,τ) + σ(τ)u(x,τ) in Equation (2). Intuitively, increasing λ shrinks σ(τ), which weakens the control contribution σ(τ)u(x,τ) to the drift and pulls the controlled SDE toward the base dynamics. By Theorem 1, the realized path-space KL is an exact function of λ, and the dual update directly enforces the trust region through the sampling dynamics. For the external form (left), λ enters only as a coefficient on a loss penalty, and has limitations in enforcing the target. As such, under strong critic guidance, the realized KL can drift far from this target. We refer to Table 6 in Appendix G, where we summarize these structural differences side-by-side. Also, we validate this distinction empirically in Section 4.2.

### 4 Experiments

We evaluate TRQAM on off-policy fine-tuning of pretrained flow policies in the offline-to-online setting. We use OGBench [38] (50 tasks) for main comparison, and Robomimic [32] for ablation and mechanism studies. All methods share the same pretrained flow policy and training schedule.

Setup. OGBench [38] is an offline goal-conditioned RL benchmark spanning 10 suites, from which we evaluate on 50 tasks. While OGBench is originally designed for offline goal-conditioned RL, we use its reward-based single-task variants. Robomimic [32] is a demonstration based manipulation benchmark used to test the stability. For all manipulation tasks (e.g., OGBench’s scene, cube, puzzle suites and all Robomimic tasks), we use action-chunked policies with chunk size h = 5 [26].

Table 1: Offline RL on 50 OGBench [38] tasks at 1M training steps (8 seeds). Mean success rate (%) with ±1 standard deviation. (per-task breakdown across all 50 tasks in Table 7).

al ag hm hl scene p33 p44 c2 c3 c4 all

5 tasks 5 tasks 5 tasks 5 tasks 5 tasks 5 tasks 5 tasks 5 tasks 5 tasks 5 tasks 50 tasks

BACKPROP FQL 38±9 2±6 74±5 2±1 70±5 25±10 9±7 44±4 7±5 9±5 28 GUIDANCE CGQL-L 48±7 7±5 57±2 6±3 58±1 0±0 0±0 55±2 0±1 1±1 23

POST PROCESSING DSRL 53±2 1±1 53±10 1±1 80±0 100±0 61±8 72±4 34±6 9±3 46

IFQL 29±8 12±3 93±2 30±7 36±1 64±4 42±4 9±2 24±7 6±3 35 ADJOINT

MATCHING QAM 62±9 29±4 64±7 4±3 64±4 15±3 1±1 71±2 19±6 18±3 35 QAM-E 86±3 6±8 60±6 4±5 63±6 89±4 54±8 71±3 11±4 9±3 45

###### OURS TRQAM 89±4 41±4 84±3 36±4 79±1 100±0 99±1 81±3 50±5 19±5 68

On OGBench, we compare our method against six off-policy fine-tuning baselines: FQL [39], CGQL-Linex [8], DSRL [51], IFQL [14], and QAM / QAM-E [25]. On Robomimic, we focus on the adjoint-matching variants (i.e., QAM and QAM-E), and DSRL as a non-adjoint reference. We refer to Appendix B for details on each baseline. Since trust-region fine-tuning regulates deviation from a pretrained prior, this prior must itself encode meaningful behavior. Therefore, all methods are pretrained for 300K steps with behavior cloning, and run offline-to-online fine-tuning for each 1M step. We report average success rate (%) over 8 seeds, and the detailed hyperparameters are in Appendix C.3.

Abbreviations. We evaluate on 10 OGBench task suites and abbreviate their names in tables and figures for compactness: puzzle-4x4 (p44), cube-double (c2), cube-triple (c3), cube-quadruple (c4), scene, humanoidmaze-medium (hm), humanoidmaze-large (hl), antmaze-large (al), antmaze-giant (ag), and puzzle-3x3 (p33).

#### 4.1 Main results on offline and offline-to-online RL

Table 1 reports offline success rates after 1M training steps. TRQAM achieves 68% aggregate success across 50 tasks, improving on QAM (35%) by 33 points, on its strongest variant QAM-E (45%) by 23 points, and on the strongest non-adjoint baseline DSRL (46%) by 22 points, with the largest gains on long-horizon and combinatorial suites. This lead is sustained through the offline-to-online transition: per-task curves in Appendix Figures 17 and 18 show that TRQAM remains the strongest method through 500K steps of online fine-tuning.

A natural explanation for TRQAM’s gains is that it leverages the pretrained policy more effectively. Figure 3 tests this by running TRQAM, QAM-E, and QAM both from a pretrained flow policy (dashed) and from scratch (solid) under an identical offline-to-online protocol. The contrast is clear: TRQAM benefits substantially from the pretrained prior, reaching high success much earlier than its scratch counterpart, while QAM and QAM-E show little to no benefit from the same pretrained initialization, with their pretrained and scratch curves remaining close throughout training.

#### 4.2 What mechanism drives its gains?

To isolate what drives the asymmetry above, we compare three variants differing only in how λ is regulated: QAM (constant λ in our framework), QAM + External KL (adaptive λ as a conventional KL regularization loss), and TRQAM (adaptive λ internalized in the SOC sampling dynamics). This isolates two design axes: adaptation (constant KL vs. adaptive KL) and internalization (external loss penalty vs. internalized SOC sampling dynamics).

- 1. Adaptative KL outperforms constant KL. Both adaptive KL variants, TRQAM and QAM with external KL regularization, substantially outperform QAM on cube-triple-task1 and humanoidmaze-medium-task1 (Figure 4). The same asymmetry appears on Robomimic, where QAM exhibits the diverging adjoint loss and collapsing task success of Figure 2 while both adaptive variants remain stable (Appendix H.4). These results are consistent with Lemma 1’s exponential amplification of critic errors under fixed temperature.

TRQAM QAM-E QAM scratch pretrained

|Offlin|e RL|Online RL|
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

100

75

Successrate(%)

50

25

0

0 5 10 15

Training steps (×105)

- Figure 3: Pretraining alone is not sufficient. On humanoidmaze-medium-task1, each algorithm is run from a pretrained flow policy (dashed) and from scratch (solid); shaded regions denote standard deviation across seeds. TRQAM benefits substantially from the pretrained prior, while QAM and QAM-E show little to no benefit from the same pretrained initialization, with their pretrained and scratch curves remaining close throughout training.

0 5 10 15

0

25

50

75

100

cube–triple–task1

| |
|---|

| |
|---|

| |
|---|

0 5 10 15

humanoidmaze–medium–task1

| |
|---|

| |
|---|

| |
|---|

| |
|---|

training steps (×105)

successrate(%)

TRQAM (QAM + Internal KL) QAM + External KL QAM (Constant KL)

- Figure 4: Adaptation is necessary. Both adaptive variants (TRQAM and QAM + External KL) outperform QAM (constant λ) on cube-triple-task1 and humanoidmaze-medium-task1, consistent with Lemma 1: a fixed temperature can amplify critic errors into policy deviations, while adaptation can mitigate this. Shaded regions denote standard deviation across seeds.

- 2. Internal KL outperforms external KL regularization. Among the two adaptive variants, only TRQAM enforces the prescribed KL budget. On Robomimic-lift and Robomimic-can with εKL = 0.1, TRQAM tightly tracks the bound throughout training, while QAM + External KL lets the realized KL drift above εKL with corresponding success-rate degradation (Figure 5). By Theorem 1, only the internal parameterization ties λ to the realized KL through an exact identity, whereas an external penalty enters only as an additive loss term that strong critic guidance can override. The same pattern holds across all three Robomimic tasks and six KL budgets (Appendix Figures 14, 15, and 16). These violations are consistent with Lemma 1’s exponential amplification of critic errors.

#### 4.3 Sensitivity Analysis

Remark that εKL is the most important hyperparameter, and varying it produces predictable changes in success rate. We sweep εKL from 0.5 to 4 in steps of 0.5 on four representative OGBench tasks (humanoidmaze-medium-task1, humanoidmaze-large-task1, cube-double-task2, cube-triple-task2), with the same training setup as Section 4.1. Figure 6 shows two patterns. First, success rate changes smoothly with εKL on every task. Second, tight budgets are best across all four tasks. The similar pattern holds across most of the ten OGBench domains in Appendix H.3, with puzzle-4x4 as the exception where larger budgets monotonically improve performance, in keeping with its notably larger state space. Because TRQAM tightly enforces the chosen budget (Section 4.2), tuning εKL to task structure produces predictable, controlled behavior, so adapting TRQAM to a new environment comes down to choosing εKL based on task structure.

TRQAM (QAM + Internal KL) QAM + External KL

Robomimic–can

Robomimic–lift

- 100

- 101

- 102

- 100

- 101

- 102

100

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

Successrate(%)

Successrate(%)

| |
|---|

KLestimate

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10−1

10−1

target εKL = 0.1

target εKL = 0.1

0

0

10−2

10−2

0 10 15

0 10 15

0 10 15

0 10 15

Training steps (×105)

- Figure 5: Internalization is necessary. On Robomimic-lift & Robomimic-can with εKL = 0.1, TRQAM tightly tracks the target KL bound throughout training, while QAM + External KL lets the realized KL drift well above it, with corresponding success rate degradation. By Theorem 1, only the internal parameterization (TRQAM) ties λ to the realized KL through an exact identity; the external loss penalty (QAM + External KL) can be overridden by strong critic guidance. Shaded regions denote standard deviation across seeds.

0 10 15

0

50

100

humanoidmaze–medium–task1

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>|

0 10 15

humanoidmaze–large–task1

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

0 10 15

cube–double–task2

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

0 10 15

cube–triple–task2

training steps (×105)

successrate(%)

εKL = 0.5 1 1.5 2 2.5 3 3.5 4

- Figure 6: Sensitivity Analysis. Success-rate curves on four OGBench tasks under varying KL

budgets. Two patterns emerge. First, success rate changes smoothly with εKL on every task, making the budget a predictable knob. Second, tight budgets are best across all four tasks, with the optimum tracking task structure.

### 5 Related Work

Offline-to-online RL. Pretraining a policy and value function on offline data and then fine-tuning online is a standard recipe for sample-efficient RL [3, 15, 16, 22–24, 30, 34, 35, 43, 50]. A central challenge is the distribution shift at the transition, which destabilizes the value function and induces catastrophic forgetting [3, 35, 53], motivating trust-region-style constraints during fine-tuning.

Fine-tuning flow and diffusion policies. Flow matching and diffusion policies [7, 27] parameterize multi-modal action distributions and are increasingly pretrained at scale [4, 5, 18, 19]. A growing body of work trains such policies with RL [6, 11, 14, 28, 39, 44, 51, 52, 55, 56, 58, 59], each navigating a tradeoff between policy expressivity, computational cost, and training stability.

KL trust regions in RL. KL regularization stabilizes policy updates as a hard constraint or soft penalty [1, 41, 46, 47, 54], sometimes with strength adapted via dual updates (e.g., SAC [13] for entropy). These approaches enforce the constraint via auxiliary losses external to the policy.

### 6 Conclusion

We introduce Trust Region Q-Adjoint Matching (TRQAM), a stable off-policy fine-tuning method for pretrained flow policies. TRQAM adapts a trust-region parameter λ inside the SOC sampling dynamics: scaling the diffusion by √

λ makes the path-space KL an exact function of λ via Girsanov (Theorem 1), so dual descent on λ enforces the target bound at the sampling level rather than through a loss-level penalty. Across 50 OGBench tasks, TRQAM improves the strongest baseline by 22 points, with the largest gains on long-horizon and combinatorial domains; on Robomimic, it remains stable where fixed-temperature adjoint matching collapses. In the spirit of TRPO and PPO for on-policy RL, we hope TRQAM offers an analogous trust-region stabilization for off-policy fine-tuning of pretrained flow policies.

Limitations. Computing the adjoint matching loss requires a vector-Jacobian product (VJP) through the velocity field at each step of the backward ODE; this VJP cost scales with model size.

### Acknowledgments and Disclosure of Funding

This work was supported by Institute for Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (RS-2019-II190075, Artificial Intelligence Graduate School Program(KAIST)). We thank RLWRLD Inc. for providing compute resources to conduct experiments in this work.

### References

- [1] A. Abdolmaleki, J. T. Springenberg, Y. Tassa, R. Munos, N. Heess, and M. Riedmiller. Maximum a posteriori policy optimisation. In International Conference on Learning Representations, 2018.
- [2] M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. In Journal of Machine Learning Research, 2025.
- [3] P. J. Ball, L. Smith, I. Kostrikov, and S. Levine. Efficient online reinforcement learning with offline data. In International Conference on Machine Learning, 2023.
- [4] J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. URL https://arxiv.org/abs/2503.14734.
- [5] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch,

L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky. π0: A vision-languageaction flow model for general robot control. In Robotics: Science and Systems, 2025.

- [6] K. Chen, Z. Liu, T. Zhang, Z. Guo, S. Xu, H. Lin, H. Zang, X. Li, Q. Zhang, Z. Yu, G. Fan,

T. Huang, Y. Wang, and C. Yu. πRL: Online rl fine-tuning for flow-based vision-language-action models, 2026. URL https://arxiv.org/abs/2510.25889.

- [7] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Robotics: Science and Systems, 2023.
- [8] P. Dhariwal and A. Nichol. Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems, 2021.
- [9] C. Domingo-Enrich, J. Han, B. Amos, J. Bruna, and R. T. Q. Chen. Stochastic optimal control matching. In Advances in Neural Information Processing Systems, 2024.
- [10] C. Domingo-Enrich, M. Drozdzal, B. Karrer, and R. T. Q. Chen. Adjoint matching: Finetuning flow and diffusion generative models with memoryless stochastic optimal control. In International Conference on Learning Representations, 2025.
- [11] P. Dong, Q. Li, D. Sadigh, and C. Finn. Expo: Stable reinforcement learning with expressive policies. In International Conference on Learning Representations, 2026.
- [12] S. Fujimoto, H. van Hoof, and D. Meger. Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning, 2018.
- [13] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning, 2018.
- [14] P. Hansen-Estruch, I. Kostrikov, M. Janner, J. G. Kuba, and S. Levine. Idql: Implicit q-learning as an actor-critic method with diffusion policies, 2023. URL https://arxiv.org/abs/2304. 10573.
- [15] T. Hester, M. Vecerik, O. Pietquin, M. Lanctot, T. Schaul, B. Piot, D. Horgan, J. Quan, A. Sendonaris, G. Dulac-Arnold, I. Osband, J. Agapiou, J. Z. Leibo, and A. Gruslys. Deep q-learning from demonstrations. In AAAI Conference on Artificial Intelligence, 2018.

- [16] H. Hu, S. Mirchandani, and D. Sadigh. Imitation bootstrapped reinforcement learning. In International Conference on Learning Representations, 2024.
- [17] C.-Y. Hung, N. Majumder, H. Deng, L. Renhang, Y. Ang, A. Zadeh, C. Li, D. Herremans, Z. Wang, and S. Poria. Nora-1.5: A vision-language-action model trained using world modeland action-based preference rewards, 2025. URL https://arxiv.org/abs/2511.14659.
- [18] P. Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, D. Driess, M. Equi, A. Esmail, Y. Fang, C. Finn, C. Glossop, T. Godden, I. Goryachev, L. Groom, H. Hancock, K. Hausman, G. Hussein, B. Ichter, S. Jakubczak, R. Jen, T. Jones, B. Katz, L. Ke, C. Kuchi, M. Lamb, D. LeBlanc, S. Levine, A. Li-Bell, Y. Lu, V. Mano, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, C. Sharma, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, W. Stoeckle, A. Swerdlow, J. Tanner, M. Torne, Q. Vuong, A. Walling, H. Wang, B. Williams, S. Yoo, L. Yu, U. Zhilinsky, and Z. Zhou. π0∗.6: a vla that learns from experience, 2025. URL https://arxiv.org/abs/2511.14759.
- [19] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke,

A. Walling, H. Wang, L. Yu, and U. Zhilinsky. π0.5: a vision-language-action model with open-world generalization. In Conference on Robot Learning, 2025.

- [20] T. Jiang, T. Yuan, Y. Liu, C. Lu, J. Cui, X. Liu, S. Cheng, J. Gao, H. Xu, and H. Zhao. Galaxea open-world dataset and g0 dual-system vla model, 2025. URL https://arxiv.org/abs/ 2509.00576.
- [21] C. Kim, H. Lee, Y. Seo, K. Lee, and Y. Zhu. Deas: Detached value learning with action sequence for scalable offline rl. In International Conference on Learning Representations, 2026.
- [22] I. Kostrikov, A. Nair, and S. Levine. Offline reinforcement learning with implicit q-learning. In International Conference on Learning Representations, 2022.
- [23] A. Kumar, A. Zhou, G. Tucker, and S. Levine. Conservative q-learning for offline reinforcement learning. In Advances in Neural Information Processing Systems, 2020.
- [24] K. Lei, Z. He, C. Lu, K. Hu, Y. Gao, and H. Xu. Uni-o4: Unifying online and offline deep reinforcement learning with multi-step on-policy optimization. In International Conference on Learning Representations, 2024.
- [25] Q. Li and S. Levine. Q-learning with adjoint matching. In International Conference on Learning Representations, 2026.
- [26] Q. Li, Z. Zhou, and S. Levine. Reinforcement learning with action chunking. In Advances in Neural Information Processing Systems, 2025.
- [27] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.
- [28] J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang. Flow-grpo: Training flow matching models via online rl. In Advances in Neural Information Processing Systems, 2025.
- [29] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.
- [30] J. Luo, Z. Hu, C. Xu, Y. L. Tan, J. Berg, A. Sharma, S. Schaal, C. Finn, A. Gupta, and S. Levine. Serl: A software suite for sample-efficient robotic reinforcement learning. In IEEE International Conference on Robotics and Automation, 2024.
- [31] N. Ma, M. Goldstein, M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden, and S. Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, 2024.

- [32] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. In Conference on Robot Learning, 2021.
- [33] V. Myers, B. Zheng, B. Eysenbach, and S. Levine. Offline goal-conditioned reinforcement learning with quasimetric representations. In Advances in Neural Information Processing Systems, 2025.
- [34] A. Nair, A. Gupta, M. Dalal, and S. Levine. Awac: Accelerating online reinforcement learning with offline datasets. In International Conference on Learning Representations, 2021.
- [35] M. Nakamoto, Y. Zhai, A. Singh, M. S. Mark, Y. Ma, C. Finn, A. Kumar, and S. Levine. Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. In Advances in Neural Information Processing Systems, 2023.
- [36] N. Nüsken and L. Richter. Solving high-dimensional Hamilton–Jacobi–Bellman pdes using neural networks: perspectives from the theory of controlled diffusions and measures on path space. Partial differential equations and applications, 2:1–48, 2021.
- [37] B. Øksendal. Stochastic differential equations. Springer, 2003.
- [38] S. Park, K. Frans, B. Eysenbach, and S. Levine. Ogbench: Benchmarking offline goalconditioned rl. In International Conference on Learning Representations, 2025.
- [39] S. Park, Q. Li, and S. Levine. Flow q-learning. In International Conference on Machine Learning, 2025.
- [40] A. Parsian and S. Kirmani. Estimation under linex loss function. In Handbook of applied econometrics and statistical inference, pages 75–98. CRC Press, 2002.
- [41] X. B. Peng, A. Kumar, G. Zhang, and S. Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. In International Conference on Learning Representations, 2021.
- [42] Y. Polyanskiy and Y. Wu. Information Theory: From Coding to Learning. Cambridge University Press, 2025.
- [43] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. In Robotics: Science and Systems, 2018.
- [44] A. Z. Ren, J. Lidard, L. L. Ankile, A. Simeonov, P. Agrawal, A. Majumdar, B. Burchfiel, H. Dai, and M. Simchowitz. Diffusion policy policy optimization. In International Conference on Learning Representations, 2025.
- [45] M. Reuss, H. Zhou, M. Rühle, Ö. E. Ya˘gmurlu, F. Otto, and R. Lioutikov. FLOWER: Democratizing generalist robot policies with efficient vision-language-flow models. In Conference on Robot Learning, 2025.
- [46] J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel. Trust region policy optimization. In International Conference on Machine Learning, 2015.
- [47] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.
- [48] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, S. Alibert, M. Cord, T. Wolf, and R. Cadene. Smolvla: A vision-language-action model for affordable and efficient robotics, 2025. URL https: //arxiv.org/abs/2506.01844.
- [49] R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. MIT press, 2018.
- [50] M. Vecerik, T. Hester, J. Scholz, F. Wang, O. Pietquin, B. Piot, N. Heess, T. Rothörl, T. Lampe, and M. Riedmiller. Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards, 2018. URL https://arxiv.org/abs/1707.08817.

- [51] A. Wagenmaker, M. Nakamoto, Y. Zhang, S. Park, W. Yagoub, A. Nagabandi, A. Gupta, and S. Levine. Steering your diffusion policy with latent space reinforcement learning. In Conference on Robot Learning, 2025.
- [52] Z. Wang, J. J. Hunt, and M. Zhou. Diffusion policies as an expressive policy class for offline reinforcement learning. In International Conference on Learning Representations, 2023.
- [53] M. Wołczyk, B. Cupiał, M. Ostaszewski, M. Bortkiewicz, M. Zaja˛c, R. Pascanu, Ł. Kuci´nski, and P. Miło´s. Fine-tuning reinforcement learning models is secretly a forgetting mitigation problem. In International Conference on Machine Learning, 2024.
- [54] Y. Wu, G. Tucker, and O. Nachum. Behavior regularized offline reinforcement learning. In International Conference on Learning Representations, 2020.
- [55] W. Xiao, H. Lin, A. Peng, H. Xue, T. He, Y. Xie, F. Hu, J. Wu, Z. Luo, L. J. Fan, G. Shi, and Y. Zhu. Self-improving vision-language-action models with data generation via residual rl. In International Conference on Learning Representations, 2026.
- [56] C. Xu, J. T. Springenberg, M. Equi, A. Amin, A. Esmail, S. Levine, and L. Ke. Rl token: Bootstrapping online rl with vision-language-action models, 2026. URL https://arxiv. org/abs/2604.23073.
- [57] A. Zhai, B. Liu, B. Fang, C. Cai, E. Ma, E. Yin, H. Wang, H. Zhou, J. Wang, L. Shi, L. Liang, M. Wang, Q. Wang, R. Gan, R. Yu, S. Li, S. Liu, S. Chen, V. Chen, and Z. Xu. Igniting vlms toward the embodied space, 2025. URL https://arxiv.org/abs/2509.11766.
- [58] T. Zhang, C. Yu, S. Su, and Y. Wang. Reinflow: Fine-tuning flow matching policy with online reinforcement learning. In Advances in Neural Information Processing Systems, 2025.
- [59] Y. Zhang, S. Yu, T. Zhang, M. Guang, H. Hui, K. Long, Y. Wang, C. Yu, and W. Ding. Sac flow: Sample-efficient reinforcement learning of flow-based policies via velocity-reparameterized sequential modeling. In International Conference on Learning Representations, 2026.
- [60] J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, T. Wang, Y.-Q. Zhang, J. Liu, and X. Zhan. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. In International Conference on Learning Representations, 2026.

### Contents

- A Algorithm 15
- B Baselines 15
- C Experimental details 17

- C.1 Domains and tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.2 Compute resources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.3 Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- D Proofs 20

- D.1 Proof of Lemma 1: exponential amplification of critic errors . . . . . . . . . . . . 20
- D.2 Proof of Theorem 1: path-space KL identity . . . . . . . . . . . . . . . . . . . . . 21
- D.3 Proof of Proposition 1: Terminal KL upper-bounded by path-space KL . . . . . . . 22

- E Path-space KL surrogate under OT memoryless discretization 22
- F KL-budgeted improvement: primal–dual derivation 23
- G Internal vs. external KL regularization: detailed comparison 24
- H Additional experiments 25

- H.1 TRQAM tightly enforces the prescribed KL budget . . . . . . . . . . . . . . . . . 25
- H.2 Time-varying KL budget . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- H.3 Sensitivity Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- H.4 Stability stress test on Robomimic . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- I Broader impacts 27

### A Algorithm

Algorithm 3 Trust Region Q-Adjoint Matching (TRQAM) for fine-tuning Flow Matching policies. Blue marks TRQAM additions over QAM [25].

Input: replay buffer D; vbase: pretrained (behavior) velocity field; vθft: fine-tuned velocity field; Qϕ: critic function; step size h; KL budget εKL; dual stepsize ηλ; EMA coefficient ρ; fine-tuning iterations N.

Initialize: vθft ← vbase with parameters θ; λ0 > 0; D0 ← 0. Memoryless SDE: √λn σn(τ) = g(τ) for all n, where g(τ) := 2(1 − τ)/τ for n ∈ {0,...,N − 1} do

Sample a batch B = {(si,ai,ri,s′i)} from D Critic update: Optimize ϕ w.r.t.

1 |B|

L(ϕ) =

(s,a,r,s′)∈B

Qϕ(s,a) − r − γ Qϕ¯(s′, a′ ∼ πθ(· | s′)) 2 ▷ TD backup (11)

Policy update: For each state s ∈ B, sample a trajectory X = (Xτ)τ∈{0,h,...,1} via the memoryless Euler scheme:

Xτ+h = Xτ + h 2vθft(s,Xτ,τ) −

1 τ

Xτ +

√

ετ, ετ,X0 ∼ N(0,I). (12)

h λn σn(τ)

= g(τ)

Compute the critic’s action gradient: a˜1 ← −∇X1

Qϕ(s,X1). Solve the lean adjoint ODE backwards:

a˜τ−h = a˜τ + ha˜⊤τ∇Xτ

2vbase(s,Xτ,τ) −

Stop gradient: Xτ ← stopgrad(Xτ), a˜τ ← stopgrad(˜aτ). Optimize θ w.r.t. the adjoint matching objective:

1 τ

Xτ . (13)

1 |B| s∈B τ

LAdj-Match(θ) =

σn(τ) vθft(s,Xτ,τ) − vbase(s,Xτ,τ) + σn(τ)a˜τ

2

Trust region update: Estimate path-space KL surrogate:

2

(14)

1 |B| s∈B τ

g(τ)2 vθft(s,Xτ,τ) − vbase(s,Xτ,τ) 2 (15)

2h

Dn =

EMA smoothing: Dn ← (1 − ρ)Dn−1 + ρ Dn. Dual descent: λn+1 ← max 0, λn + ηλ(Dn − εKL) .

end for Output: fine-tuned velocity field vθft, critic Qϕ.

### B Baselines

We compare against six baselines spanning distinct fine-tuning paradigms for flow-matching policies. Throughout, (s,a,r,s′) are sampled uniformly from the replay buffer D without re-weighting; D contains the offline dataset during offline training and is augmented with online rollouts during online fine-tuning.

FQL [39]. FQL distills a multi-step flow policy into a one-step policy to avoid backpropagation through time. Conditioned on state s, the behavior-cloning rollout is given by the ODE

##### dXτ = vθ(Xτ,τ;s)dτ, X0 ∼ N(0,I), τ ∈ [0,1], (16)

and we define ODE(vθ,s,X0) := X1 as its terminal value. The one-step policy πω(s,X0) is trained jointly with vθ to maximize the critic while staying close to this rollout:

##### + α ∥πω(s,X0) − ODE(vθ,s,X0)∥22

Lonestep(ω) = EX

0∼N −Q(s,πω(s,X0))

RL maximization

BC distillation

where α controls how closely πω stays to the BC rollout. The environment policy is πω.

, (17)

CGQL-Linex. CGQL-Linex is a baseline introduced in Li and Levine [25], combining a BC velocity field with classifier-free guidance [8] from a Q-function. It trains an auxiliary intermediate critic Qψ(s,Xτ,τ) on intermediate noisy actions Xτ, used to construct a guidance velocity that steers sampling toward the entropy-regularized optimal policy π⋆(· | s) ∝ eβQ

ϕ(s,·). The guidance velocity is

(1 − τ)β ∇Xτ

Qψ(s,Xτ,τ) + Xτ τ

, (18)

vˆψ(Xτ,τ;s) :=

where Xτ = (1 − τ)X0 + τX1 with X0 ∼ N(0,I). The final sampling velocity is v = vbase + w vˆψ, where w modulates guidance strength. The intermediate critic Qψ is trained via a Linex regression [40, 33]:

##### exp β(Qϕ(s,X1) − Qψ(s,Xτ,τ)) + βQψ(s,Xτ,τ) , (19)

##### LLinex(ψ) = Eτ,X

0

while the standard critic Qϕ is trained via standard TD with target actions sampled from the full velocity v:

##### LTD(ϕ) = E Qϕ(s,a) − r − γ Qϕ¯(s′,ODE(v,s′,X0)) 2 . (20)

We follow Li and Levine [25] in using a Huber-style stabilization of the Linex loss to prevent exponential blow-up.

DSRL [51]. DSRL performs RL directly in the noise space of a frozen flow policy. It trains a onestep Gaussian noise-space policy πω(X0 | s) via SAC, maximizing a noise-space critic Qψ(s,X0) that regresses to the original action-space critic:

0∼N Qψ(s,X0) − Qϕ(s,ODE(vθ¯,s,X0)) 2 . (21) At inference, actions are obtained by sampling X0 ∼ πω(· | s) and pushing it through the flow policy: the BC velocity online (using a target network vθ¯ for stability), which yields stronger offline-to-online performance.

L(ψ) = EX

- a = ODE(vθ,s,X0). Following Li and Levine [25], we modify the original DSRL to also fine-tune

IFQL. IFQL is the flow counterpart of implicit diffusion Q-learning [14], considered as a baseline in Park et al. [39]. Value learning uses IQL-style expectile regression [22]: a value network Vξ is trained to fit an upper expectile of the critic, and the critic is bootstrapped through Vξ:

LV (ξ) = E Lκ2 Qϕ¯(s,a) − Vξ(s) , (22) LQ(ϕ) = E r + γVξ(s′) − Qϕ(s,a) 2 , (23)

where Lκ2(u) = |κ − 1(u < 0)|u2 is the expectile loss with parameter κ ∈ (0.5,1). Policy extraction uses rejection sampling: N candidate actions are drawn from a BC flow policy by sampling

X0(i) ∼ N(0,I) and computing a(i) = ODE(vθ,s,X0(i)) for i = 1,...,N, and the action with the highest Qϕ value is selected.

QAM and QAM-E [25]. QAM is the closest prior work to TRQAM: it solves a memoryless SOC problem analogous to (3) but without the √

λ scaling on the diffusion coefficient (i.e., the special case λ = 1 of our framework), with an inverse temperature β applied to the terminal reward β Qϕ(s,X1). The fine-tuned velocity vθ is trained against the BC velocity vbase through the lean adjoint matching loss

 , (24)

 

2

1

2 vθ(Xτ,τ;s) − vbase(Xτ,τ;s) σ(τ)

LAM(θ) = E

dτ

+ σ(τ) ˜aτ

0

2

where a˜τ is the lean adjoint state with terminal condition a˜1 = −β ∇X1

Qϕ(s,X1), and Xτ follows the memoryless SDE in (2) with λ = 1. Element-wise gradient clipping is applied for numerical stability. QAM-E augments QAM with an additional residual edit policy πω(∆a | s,a˜) that perturbs the QAM-generated action a˜ by at most σa in L∞ distance (enforced by a tanh-squashed Gaussian), trained via entropy-regularized SAC with automatic entropy tuning. Both QAM and QAM-E share their BC velocity, critic, and inner adjoint solver with TRQAM; the only difference from TRQAM is that β is fixed rather than adapted via projected dual descent on λ.

### C Experimental details

#### C.1 Domains and tasks

We evaluate on two benchmarks: OGBench [38] and Robomimic [32].

OGBench is a recent offline goal-conditioned RL benchmark. While OGBench originally designed for offline-goal-conditioned-RL, we use its reward based single-task variants. From OGBench, we use 10 domains spanning long-horizon navigation, multi-object manipulation, and combinatorial planning. Abbreviations: scene, puzzle-3x3 (p33), puzzle-4x4 (p44), cube-double (c2), cube-triple (c3), cube-quadruple (c4), humanoidmaze-medium (hm), humanoidmaze-large (hl), antmaze-large (al), and antmaze-giant (ag).

Robomimic is a demonstration based manipulation benchmark used as a stability stress-test. We use the lift, can, and square tasks.

The dataset size, episode length, and action dimension for each domain are reported in table 2. For each method and task, we run 8 random seeds. Unless otherwise stated, tables report mean success rate ± standard deviation across seeds, and plots show the mean with shaded regions denoting standard deviation.

Dataset structure. Unlike teleoperation-style benchmarks where each demonstration directly solves the target task, OGBench’s offline data is task-agnostic: navigate datasets capture free maze exploration and play datasets capture unstructured object manipulation. The BC-pretrained policy thus serves as a behavioral prior rather than a task-specific solution.

Dataset sources. We use the official OGBench datasets [38] for all domains except where noted below. For cube-triple-10M-* and puzzle-4x4-10M-*, we use a 10M-size subset of the official 100M release. The 100M release is split into 100 files of 1M transitions each, and we take the first 10 files sorted by name, following [21]. For antmaze-giant-10M-*, OGBench does not release a pretrained dataset at the size used in our experiments, so we generate it ourselves using the official OGBench data-generation pipeline with default settings. For Robomimic, we use the Multi-Human (MH) datasets, each consisting of 300 trajectories collected by six operators of varying proficiency (two “worse”, two “okay”, and two “better”), yielding diverse mixed-quality demonstrations [32].

#### C.2 Compute resources

Hardware. Experiments ran on an internal heterogeneous GPU cluster. The two dominant GPU types were NVIDIA GeForce RTX 2080 Ti (11GB GPU memory) and NVIDIA A100-SXM4-80GB (80GB GPU memory; ≈ 1.5TB host memory and 64 logical CPU cores per node). A small fraction of runs (notably the scene OGBench domain and parts of antmaze-giant-10M) additionally used NVIDIA RTX 3090, RTX A6000, or RTX 4090 cards as availability allowed. To improve cluster

throughput, we packed up to four runs per A100-80GB GPU concurrently, while RTX 2080 Ti runs were single-tenant.

Per-run wallclock by hardware and scale. Median per-run wallclock times, measured directly from our experiment logs, are reported separately by GPU type because A100-80GB and RTX 2080 Ti are not interchangeable in run time:

- • Robomimic (lift, can, square): RTX 2080 Ti single-tenant median ≈ 10h on lift, ≈ 12h on can, ≈ 12h on square; A100-80GB with four-run multi-tenancy median ≈ 11h on lift, ≈ 11h on can, ≈ 12h on square.
- • OGBench 1M-data domains (antmaze-large, puzzle-3x3, scene, humanoidmaze-medium, humanoidmaze-large, cube-double, cube-triple): RTX 2080 Ti single-tenant median ≈ 7h; the small fraction of these runs scheduled on A100 (four-run multi-tenancy) or RTX 3090 finished in ≈ 2–4h.
- • OGBench 10M-data domain (antmaze-giant-10M, cube-triple-10M, puzzle-4x4-10M): RTX 2080 Ti single-tenant median ≈ 19h; A100-80GB with four-run multi-tenancy median ≈ 9h.
- • OGBench 100M-data domains (cube-quadruple-100M): A100-80GB with four-run multitenancy median ≈ 7h (this domain was run only on A100).

Total compute by category. Summing run wallclocks across all logged runs that contributed to the reported results, the two main reporting categories of the project are:

- • OGBench main reproduction and supporting ablations across the 10 domains, 8 evaluation seeds, and the 7 compared methods (TRQAM, QAM, QAM-E, FQL, IFQL, CGQL-L, DSRL), including BC pretraining of the flow-matching priors: ≈ 38,400 wallclock-run-hours, of which ≈ 30,700h on RTX 2080 Ti, ≈ 5,700h on A100-80GB, and ≈ 1,900h on RTX 3090, RTX A6000, and RTX 4090 cards combined.
- • Robomimic main comparison on lift, can, and square for TRQAM, QAM, QAM-E (QAMEDIT), and DSRL with the per-method hyperparameter sweeps in Appendix C.3: ≈ 8,600 wallclock-run-hours, of which ≈ 6,000h on A100-80GB and ≈ 2,600h on RTX 2080 Ti.

The two categories together total approximately 47,000 wallclock-run-hours.

Physical GPU-hours. Accounting for the four-run multi-tenancy on A100-80GB and the singletenant policy on the remaining GPU types, the wallclock-run-hours above translate to approximately 33,400 RTX 2080 Ti GPU-hours, 2,900 A100-80GB GPU-hours, and 1,900 GPU-hours on the RTX 3090/A6000/4090 cards combined—about 38,000 physical GPU-hours for the two reporting categories. Additional preliminary, failed, and supporting compute (KL-budget sweeps used to select εKL, sensitivity analyses in Appendix H.3, and prior re-pretraining) consumed further GPU-hours that are not included in the totals above.

#### C.3 Hyperparameters

Most methods share the common hyperparameters in table 3; method-specific hyperparameters are listed in table 4.

OGBench tuning. For all baselines (FQL, DSRL, IFQL, CGQL-L, QAM, QAM-E), we adopt the per-domain values reported as optimal in QAM [25], which were selected through an extensive per-domain hyperparameter sweep covering 6–20 configurations per method across all ten domains, totaling approximately 32,000 GPU-hours. For TRQAM, we tune εKL on two tuning tasks per domain with 2 seeds per configuration (different from the evaluation seeds), sweeping over {0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0}. Following Li and Levine [25], we use task 1 (the default task) and task 4 for locomotion domains, and task 2 (the default task) and task 4 for manipulation domains, as this combination better covers the characteristics of each domain than the default task alone. We select the configuration based on the combined offline-to-online learning curve and stability across the two tuning tasks; selected values are reported in table 4. For antmaze-giant-10M-*, we use a time-varying schedule: εKL = 0.5 during offline training and 3.0 during online fine-tuning. The offline value is selected from the per-domain sweep, where it yielded the most stable offline training. However, retaining 0.5 online led to slow improvement in success rate, reflecting the larger

exploration demands of giant-scale navigation. We therefore relax the bound at the online transition, selecting 3.0 from the same sweep range as a sweet spot between exploration and stability (see section H.2 for a direct comparison of static vs. relaxed schedules). This schedule exploits TRQAM’s capability to track time-varying εKL without retraining. All main-paper results are averaged over 8 evaluation seeds.

Robomimic tuning. Since Robomimic was not evaluated in the QAM paper [25], no per-domain hyperparameters are available, so we conducted an independent sweep for each method (DSRL, QAM, QAM-E, TRQAM) on this benchmark. As Robomimic serves as our stability testbed, we run the full sweep range with 8 seeds per configuration to obtain reliable estimates of variance across hyperparameters. For QAM-E, which has two hyperparameters, we sweep the full Cartesian product (4 × 2 = 8 configurations). Robomimic consists of human teleoperation data, for which staying closer to behavior cloning is generally beneficial. We therefore adjust the QAM [25] sweep ranges toward configurations that more strongly anchor the policy to the behavior policy. The sweep ranges are listed in table 5.

#### Table 2: Domain metadata.

###### Domain Data Horizon Act. dim.

cube-double-* 1M 500 5 cube-triple-10M-* 10M 1000 5 cube-quadruple-100M-* 100M 1000 5 antmaze-large-* 1M 1000 8 antmaze-giant-10M-* 10M 1000 8 humanoidmaze-medium-* 1M 2000 21 humanoidmaze-large-* 1M 2000 21 scene-* 1M 750 5

- puzzle-3x3-* 1M 500 5
- puzzle-4x4-10M-* 10M 500 5 lift 31127 500 5 can 62756 500 5 square 80731 500 5

#### Table 3: Common hyperparameters.

###### Parameter Value

Batch size 256 Discount factor (γ) 0.995 (default), 0.999 (humanoidmaze) Optimizer Adam

Learning rate 3 × 10−4 Target network update rate 5 × 10−3 Critic ensemble size (K) 10 Critic pessimism coefficient (ρ) 0.5 (default), 0 (humanoidmaze) UTD ratio 1 Number of flow steps (T) 10 BC training steps 0.3 × 106 Offline RL steps 106 Online RL steps 0.5 × 106 Network width 512 (default), 1024 (10M/100M data) Network depth 4 hidden layers Gradient max-norm clipping False (default), 1 (QAM, QAM-E, TRQAM) Actor layer norm False (default), True (10M/100M data) Critic layer norm True

#### Table 4: Domain-specific hyperparameters.

Domain FQL DSRL IFQL CGQL-L QAM QAM-E TRQAM

α σz κ (ϑ, ϱ, τ) β (β, σa) εKL scene-* 300 0.4 0.9 (10, 0.1, 0.1) 1 (1, 0) 0.5

- puzzle-3x3-* 300 1.0 0.95 (10, 0.001, 0.1) 3 (1, 0.1) 2.0
- puzzle-4x4-10M-* 1 1.0 0.9 (10, 0.001, 1) 30 (0.1, 0.9) 4.0 cube-double-* 300 1.0 0.9 (10, 0.001, 0.01) 1 (1, 0) 0.5 cube-triple-10M-* 30 1.4 0.95 (10, 0.001, 0.1) 3 (3, 0.1) 0.5 cube-quadruple-100M-* 100 1.4 0.95 (10, 0.1, 0.01) 1 (3, 0.1) 1.0 antmaze-large-* 3 0.8 0.9 (10, 0.001, 0.1) 10 (1, 0.1) 1.0 antmaze-giant-10M-* 3 1.2 0.8 (10, 0.001, 0.1) 3 (10, 0.1) 0.5→3.0 humanoidmaze-medium-* 30 0.6 0.7 (10, 0.1, 0.1) 3 (3, 0.1) 0.5 humanoidmaze-large-* 30 0.8 0.8 (10, 0.1, 0.1) 3 (3, 0.1) 0.5 lift - 0.8 - - 1 (1, 0.03) 0.1 can - 0.6 - - 0.3 (1, 0.03) 0.1 square - 0.8 - - 0.1 (1, 0.03) 0.5

Table 5: Robomimic hyperparameter sweep ranges. Method Hyperparameter(s) Sweep Range

DSRL σz {0.03,0.1,0.2,0.4,0.6,0.8} QAM β {0.01,0.03,0.1,0.3,1,3} QAM-E (β,σa) ({0.03,0.1,0.3,1}, {0.03,0.1}) TRQAM εKL {0.01,0.03,0.1,0.5,1.0,1.5}

### D Proofs

This section provides full proofs for the three theoretical results stated in Section 3 of the main text: Theorem 1 (path-space KL identity), Proposition 1 (Terminal KL upper-bounded by path-space KL), and Lemma 1 (exponential amplification of critic errors).

We work throughout on a filtered probability space (Ω,F,(Fτ)τ∈[0,1],P) equipped with a natural filtration and an initial state distribution X0 ∼ p0. We adopt standard regularity assumptions for controlled diffusions, as in Nüsken and Richter [36]: the coefficients b and σ are sufficiently smooth,

- b has at most linear growth, σσ⊤ is uniformly positive definite, and the admissible control set U consists of progressively measurable controls with at most linear growth. Crucially, to ensure strong duality in our KL-budgeted improvement problem, we assume that the set of achievable path measures {Pu | u ∈ U} is convex. Moreover, we assume that u ∈ U satisfies Novikov’s condition,

1

- 1

- 2λ

∥u(Xτ,τ)∥2 dτ < ∞,

EX∼Pbase exp

0

which justifies the application of Girsanov’s theorem in the proof of Theorem 1.

#### D.1 Proof of Lemma 1: exponential amplification of critic errors

Lemma 1 (Exponential amplification of critic errors). Fix s ∈ S and let Q, Q : A → R satisfy ∥Q − Q∥∞ ≤ ε. Assume πbase(· | s) > 0 a.e. and define

πbase(a | s)eβQ(a) ZQ

πbase(a | s)eβ Q(a) Z Q

,

πQ(a | s) =

, π Q(a | s) =

with normalizers ZQ,Z Q ∈ (0,∞). Then

- 1

- 2

e2βε − 1 , DKL(πQ ∥π Q) ≤ 2βε.

TV(πQ,π Q) ≤

Proof. For notational simplicity, we suppress s throughout the proof. From ∥Q − Q∥∞ ≤ ε, multiplying −ε ≤ Q(a) − Q(a) ≤ ε by β and exponentiating yields, for all a,

e−βε ≤ eβ(Q(a)− Q(a)) ≤ eβε. (25)

Multiplying (25) by πbase(a)eβ Q(a) ≥ 0 and integrating gives e−βεZ Q ≤ ZQ ≤ eβεZ Q. Combining this with (25) in the likelihood ratio

Z Q ZQ

πQ(a) π Q(a)

= eβ(Q(a)− Q(a)) ·

shows that both factors lie in [e−βε,eβε], so

e−2βε ≤

πQ(a) π Q(a) ≤ e2βε a.e. (26)

The TV bound follows from (26): |πQ(a) − π Q(a)| ≤ (e2βε − 1)π Q(a) a.e., and integrating with

π Q = 1 gives TV(πQ,π Q) = 12∥πQ−π Q∥1 ≤ 12(e2βε−1). The KL bound also follows from (26): log(πQ/π Q) ≤ 2βε a.e., so

πQ(a) π Q(a)

da ≤ 2βε πQ(a)da = 2βε.

DKL(πQ ∥π Q) = πQ(a)log

| |
|---|

#### D.2 Proof of Theorem 1: path-space KL identity

Theorem 1 (Path-space KL identity with explicit λ dependence). Let Pu and Pbase denote the path measures on C([0,1];Rd) induced by the base and controlled SDEs, respectively:

√

λσ(τ)dBτbase, dXτu = b(Xτu,τ) + σ(τ)u(Xτu,τ) dτ +

dXτbase = b(Xτbase,τ)dτ +

√

λσ(τ)dBτu, with common initial distribution X0 ∼ p0. Then

1

- 1

- 2λ

∥u(Xτu,τ)∥2 dτ .

DKL Pu(X | X0)∥Pbase(X | X0) = EX∼Pu

0

Proof. Let (Fτ)τ∈[0,1] denote the natural filtration on the canonical path space C([0,1];Rd). By Girsanov’s theorem [37, Theorem 8.6.6], the Radon–Nikodym derivative of Pu with respect to Pbase restricted to Fτ is

τ

τ

dPu dPbase F

1 √

- 1

- 2λ

u(Xsu,s)⊤ dBsbase −

∥u(Xsu,s)∥2 ds , (27)

= exp

Zτ :=

λ

0

0

τ

where Bbase is a standard Brownian motion under Pbase. By the definition of KL divergence on the full path space (terminal σ-algebra F1),

DKL Pu ∥Pbase = EX∼Pu[log Z1]. Next, define

τ

1 √

Bτu := Bτbase −

u(Xsu,s)ds.

λ

0

By Girsanov’s theorem, Bu is a standard Brownian motion under Pu; in differential form, dBsbase = dBsu + √1λ u(Xsu,s)ds. Substituting this into (27) at τ = 1 gives

1

1

1 √

1 √

- 1

- 2λ

u(Xsu,s)⊤ dBsu +

u(Xsu,s)ds −

∥u(Xsu,s)∥2 ds

log Z1 =

λ

λ

0

0

1

1

1 √

- 1

- 2λ

u(Xsu,s)⊤ dBsu +

∥u(Xsu,s)∥2 ds.

=

λ

0

0

Taking expectation under Pu and using that the Itô integral 0 1 u(Xsu,s)⊤ dBsu is a zero-mean martingale yields

1

- 1

- 2λ

∥u(Xsu,s)∥2 ds ,

DKL Pu ∥Pbase = EX∼Pu

0

which is the claimed identity. Comparison to prior SOC parameterizations. The SOC parameterization used in prior work [10] takes the form

| |
|---|

dXτu = b(Xτu,τ) + σ(τ)u(Xτu,τ) dτ + σ(τ)dBτu without a √

λ scaling on the diffusion coefficient. Then Girsanov theorem yields

1

- 1

- 2

DKL(Pu∥Pbase) = EX∼Pu

∥u(Xsu,s)∥2 ds ,

0

In this form, the path-space KL coincides with the SOC quadratic cost up to a fixed constant; there is no parameter λ inside the SDE that modulates the strength of this regularization, so adapting the trust-region strength would require an external coefficient applied to the SOC objective itself. By scaling the diffusion by √

λ instead, our parameterization makes λ an intrinsic parameter of the controlled dynamics, with the 1/λ factor appearing directly in the KL identity above. This is the structural property that allows λ to serve as an adaptive dual variable for KL-budgeted improvement (Section 3): adjusting λ reshapes the SDE itself, which in turn directly modulates the path-space KL.

#### D.3 Proof of Proposition 1: Terminal KL upper-bounded by path-space KL

Proposition 1 (Terminal KL upper-bounded by path-space KL). Let Pu and Pbase denote the path measures on C([0,1];Rd) induced by the base and controlled SDEs, respectively:

√

dXτbase = b(Xτbase,τ)dτ +

λσ(τ)dBτbase, dXτu = b(Xτu,τ) + σ(τ)u(Xτu,τ) dτ +

√

λσ(τ)dBτu,

with common initial distribution X0 ∼ p0. Let πθ(· | s) and πbase(· | s) denote the corresponding terminal action distributions at τ = 1, respectively. Then

DKL πθ(· | s)∥πbase(· | s) ≤ DKL Pu(X | X0)∥Pbase(X | X0) , (28)

Proof. Let Π : C([0,1];Rd) → Rd denote the deterministic terminal projection Π(X) = X1. By construction, the terminal action distributions are pushforwards of the path measures: πθ(· | s) = Π# Pu and πbase(· | s) = Π# Pbase. Applying the data-processing inequality for KL divergence under maps [42, Corollary. 2.18] to Π yields

DKL Π# Pu ∥Π# Pbase ≤ DKL Pu ∥Pbase , which proves the claim.

| |
|---|

### E Path-space KL surrogate under OT memoryless discretization

This section derives the closed-form path-space KL estimator Dn used in Algorithm 3. Recall from Theorem 1 that the path-space KL between Pu and Pbase admits a closed-form expression in terms of the control u, and from Proposition 1 that this path-space KL upper-bounds the terminal-policy KL we ultimately wish to constrain. We thus seek a tractable estimator of the path-space KL under the discretized OT memoryless sampler.

Gaussian KL with shared covariance. For two Gaussians N(µ1,Σ) and N(µ0,Σ) with shared covariance Σ ≻ 0, specializing the multivariate Gaussian KL formula [42, Equation. 2.8] makes the log-determinant and trace terms vanish. In other words,

DKL(N(µ1,Σ)∥N(µ0,Σ)) = 21 (µ1 − µ0)⊤Σ−1(µ1 − µ0). (29)

OT memoryless Euler step KL. Under the OT memoryless Euler scheme with step size h and schedule g(τ) = 2(1 − τ)/τ, the fine-tuned and base transitions at time τk given Xτ

= x are both Gaussian with covariance Σk = hg(τk)2 I and means

k

x , µbase(x,τk) = x + h 2vbase(x,τk) − τ1

µθ(x,τk) = x + h 2vθfin(x,τk) − τ1

x .

k

k

The means differ only through the velocity fields: µθ(x,τk) − µbase(x,τk) = 2h(vθfin(x,τk) − vbase(x,τk)). Substituting into (29) with Σ−k 1 = h g(1τ

k)2 I gives the per-step KL DKL(pθ(· | x)∥pbase(· | x)) =

2h g(τk)2 ∥vθfin(x,τk) − vbase(x,τk)∥2. (30)

Chain rule and Monte Carlo estimator. Let Pu and Pbase now denote the discrete-time path measures of the Markov chains X = (Xτ)τ∈{0,h,...,1} with shared initial distribution p0 and the Gaussian transition kernels above. Under mild regularity conditions, the Markov KL chain rule gives

DKL(Pu ∥Pbase) =

τ∼Pu[DKL(pθ(· | Xτ)∥pbase(· | Xτ))]. (31)

EX

τ∈{0,h,...,1−h}

Substituting (30) and estimating the expectation by a Monte Carlo average over the trajectories in batch B yields

2h

1 |B| X∈B

g(τ)2 ∥vθfin(Xτ,τ) − vbase(Xτ,τ)∥2, (32) which is the estimator used in Algorithm 3 (optionally smoothed by EMA to reduce variance).

D =

τ∈{0,h,...,1−h}

### F KL-budgeted improvement: primal–dual derivation This section derives the projected dual update on λ used in Algorithm 3.

Primal problem and Slater’s condition. We consider the path-space KL-budgeted improvement problem

EX∼Pu[Qπ(X1)] (33) s.t. DKL(Pu ∥Pbase) ≤ εKL. (34)

max

u∈U

Viewed in the path measure space, the objective is linear in Pu and DKL(·∥Pbase) is convex; the trivial control u ≡ 0 recovers Pu = Pbase and is strictly feasible whenever εKL > 0, so Slater’s condition holds and strong duality applies.

Lagrangian and dual function. Let λ ≥ 0 be the dual variable for the KL constraint. The Lagrangian is

L(u,λ) = EX∼Pu[Qπ(X1)] + λ εKL − DKL(Pu ∥Pbase) ,

and the dual function is g(λ) := supu L(u,λ). By strong duality, minimizing g over λ ≥ 0 solves the primal. A standard subgradient of g at λ is

s(λ) = εKL − DKL(Pu

∥Pbase), (35)

λ

where uλ := arg supu L(u,λ) denotes the inner maximizer at the current dual variable. The subgradient inequality follows immediately: for any λ′ ≥ 0,

g(λ′) ≥ L(uλ,λ′) = g(λ) + (λ′ − λ) εKL − DKL(Pu

∥Pbase) .

λ

Projected dual descent. Projected subgradient descent on minλ≥0 g(λ) would require evaluating the KL at the inner maximizer uλ, which is not available in closed form. In Algorithm 3, we instead evaluate the path-space KL at the control un at iteration n via the Monte Carlo surrogate Dn from (32). The dual update

λn+1 ← max 0, λn + ηλ(Dn − εKL) (36) preserves the key sign property: when the realized KL exceeds the budget, λ rises and the controlled dynamics become more conservative; when it falls below, λ relaxes the trust region.

### G Internal vs. external KL regularization: detailed comparison

Section 3.3 contrasts two ways of pairing the dual update (9) with a KL constraint: appending Dn as an auxiliary loss term (external), or letting λ scale the diffusion coefficient inside the SOC dynamics (internal, TRQAM). The two formulations share the same dual update rule and the same KL estimator, yet they differ in how the constraint is enforced. This appendix expands on that contrast and explains why the difference is structural rather than cosmetic.

The key asymmetry is in what λ represents. In the external formulation, λ is a scalar weight on an auxiliary KL regularization term in the loss: it balances this regularization against the critic signal, but does not appear in the controlled SDE itself. The path-space KL between Pu and Pbase is therefore independent of λ, and is determined entirely by the learned control u. Adapting λ only changes the relative weight between the adjoint-matching loss LAdj-Match and the KL regularization Dn during gradient descent; whether the realized KL ends up close to εKL depends on the optimizer balancing these competing signals. Under strong reward gradients, the KL term can be effectively overridden, and the realized KL drifts above the target KL bound εKL

In the internal formulation, by contrast, λ is a parameter of the controlled SDE itself. Theorem 1 establishes that the path-space KL is exactly E[21λ ∥u∥2], so λ appears as the inverse coefficient of the KL itself. Increasing λ shrinks the path-space KL directly through the SDE’s diffusion term, rather than indirectly through the critic gradient. The dual update therefore reshapes the trajectory distribution structurally rather than competing with critic guidance. Table 6 summarizes these distinctions.

Table 6: Internal vs. external KL regularization. Although both approaches can use the same dual update rule on λ, the role of λ differs structurally. Internalizing λ inside the SDE makes the path-space KL an exact function of λ via Girsanov (Theorem 1), turning the target KL bound εKL into a structural constraint rather than a soft penalty.

External KL (auxiliary regularization) Internal KL (TRQAM) Controlled SDE dXτu = (b + σu) dτ + σ(τ) dBτ dXτu = (b + σu) dτ +

√

λ σ(τ) dBτ Path-space KL E 2 1 0 1∥u∥2 dτ

E 2 1λ 0 1∥u∥2 dτ (Theorem 1)

(λ-independent)

Role of λ Regularization weight on auxiliary loss Intrinsic SDE parameter Training loss LAdj-Match(θ) + λ Dn(θ) LAdj-Match(θ) Effect of increasing λ Competes with reward gradient at loss

Reshapes the SDE at sampling level

level

Enforcement mechanism Soft (gradient competition) Structural (via dynamics) Realized KL vs. target bound

Exceed the target bound (Figure 14, 15) Tracks target bound tightly

A useful way to see the practical consequence is that the two approaches differ in when the KL constraint takes effect. External regularization acts after trajectories are generated: the SDE produces samples freely under the current u, and the KL term enters only at the loss level, where it competes with the reward gradient during optimization. The realized path-space KL has no direct tie to λ; whether it ends up close to εKL depends on the optimizer balancing these competing signals. Internal√ regularization, by contrast, enforces the constraint during sample generation. Because

λσ(τ) = 2(1 − τ)/τ is fixed by the OT schedule, adjusting λ co-adjusts σ(τ) and reshapes the entire controlled SDE, including its drift term b(x,τ) + σ(τ)u(x,τ): increasing λ shrinks σ(τ), which weakens the control contribution σ(τ)u(x,τ) and pulls the controlled SDE toward the base dynamics. By Theorem 1, the realized path-space KL is then an exact function of λ, so the dual update directly reshapes the trajectory distribution rather than competing with critic guidance at the loss level. This distinction is what makes εKL a structural constraint in TRQAM rather than a nominal target. Appendix Figures 14 and 15 demonstrate this empirically. On Robomimic lift and Robomimic can, external regularization lets the realized KL exceed the prescribed bound across all six target values, with corresponding degradation in success rate.

εKL = 0.5 εKL = 1.0 εKL = 1.5 εKL = 2.0

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

- 0
- 1
- 2
- 3
- 4

KLestimate

0 5 10 15

Training steps (×105)

#### Figure 7: TRQAM tracks the prescribed KL budget across offline and online training. We

vary the target KL bound εKL ∈ {0.5,1.0,1.5,2.0} and plot the realized path-space KL throughout training. Larger budgets produce correspondingly larger policy deviation, and the monotonic ordering is preserved across the offline-to-online transition.

H Additional experiments

- H.1 TRQAM tightly enforces the prescribed KL budget

TRQAM faithfully enforces the prescribed KL budget across a range of target values. Figure 7 plots the realized path-space KL under four budgets εKL ∈ {0.5,1.0,1.5,2.0} throughout offline and online training. The realized KL tracks each prescribed target with a clear monotonic ordering, and this ordering is preserved across the offline-to-online transition, indicating that the dual update remains effective under distribution shift. The KL constraint in TRQAM is therefore both controllable and stable across training regimes.

εKL : 0.5 εKL : 0.5 → 3.0 εKL : 0.5 → 4.0

antmaze−giant−task1

KL estimate

- 0
- 1
- 2
- 3
- 4
- 5

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

100

Successrate(%)

KLestimate

75

50

25

0

0 5 10 15

0 5 10 15

Training steps (×105)

###### Figure 8: TRQAM tracks time-varying εKL schedules across the offline-to-online transition. We illustrate this on antmaze-giant, the domain in our benchmark with the largest state space and

accordingly the highest exploration demands during online fine-tuning. We keep εKL = 0.5 during offline training and optionally switch to a larger bound at the online transition (vertical dashed line). Right: The realized KL adapts to each new target almost immediately, showing that the dual update on λ handles schedule changes without instability. Left: The static schedule (0.5) improves more slowly online than the moderate schedule (0.5 → 3.0), which reaches ∼98% success rapidly and outperforms the more aggressive (0.5 → 4.0) alternative.

#### H.2 Time-varying KL budget

The KL bound εKL is not necessarily a fixed quantity throughout training: in offline-to-online RL, the appropriate trust region may differ between phases, since online fine-tuning often requires more exploration than the offline phase permits. TRQAM accommodates this naturally because λ is internalized in the SOC dynamics rather than entering as a soft loss penalty: the dual update on λ remains stable under time-varying εKL, allowing the bound to be re-set at any point without retraining or instability.

We illustrate this on antmaze-giant, the domain in our benchmark with the largest state space and accordingly the highest exploration demands during online fine-tuning. Figure 8 sweeps three schedules on antmaze-giant-task1, the tuning task for our per-domain εKL choice: the static schedule (0.5, matching the offline value) and two relaxations at the online transition (0.5 → 3.0, 0.5 → 4.0). Two observations follow.

First, the realized KL adapts to each new target almost immediately after the switch, confirming that TRQAM tracks time-varying bounds without loss of stability.

Second, the static schedule (0.5) improves substantially more slowly online than the moderate schedule (0.5 → 3.0). On domains with large state spaces such as antmaze-giant, online finetuning relies on exploring beyond the offline support, and a tight KL budget restricts this exploration and slows adaptation to the new state distribution. The moderate schedule reaches ∼98% success rapidly, while the more aggressive 0.5 → 4.0 introduces substantial variance, identifying 3.0 as a sweet spot.

#### H.3 Sensitivity Analysis

We extend the four-task sweep of Section 4.3 to all ten OGBench domains, sweeping εKL ∈ {0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0} on the default tuning task of each domain with all other hyperparameters fixed. Results across 8 seeds appear in Figure 9, showing two patterns. Tight budgets are best on humanoidmaze-medium, humanoidmaze-large, cube-double, cube-triple, and cube-quadruple. Larger budgets monotonically improve performance on puzzle-4x4, in keeping with its notably larger state space.

Smaller εKL also produces slower online adaptation across the sweep, consistent with the dual update tightening the trust region. We observe the same effect on antmaze-giant (Section H.2), where the static 0.5 schedule slows online adaptation in a domain whose large state space demands broader exploration. Combined with the realized-KL ordering of Figure 5, this is consistent with εKL controlling realized deviation from πbase rather than acting as a nominal target. Because TRQAM tightly enforces the chosen budget (Section 4.2), εKL remains a hyperparameter, but picking it for a new task comes down to task structure.

#### H.4 Stability stress test on Robomimic

We complement our OGBench results with Robomimic [32], a standard manipulation benchmark on which we observe fixed-temperature adjoint matching to be unstable.

Fixed-temperature methods collapse across the sweep. Sweeping QAM over six values of β ∈ {0.01,0.03,0.1,0.3,1.0,3.0} and QAM-E over eight (β,σa) configurations, the collapse pattern persists across most settings on lift and can (See Figure 10, 11), the adjoint-matching loss explodes during offline training and success rate collapses to near zero. This is consistent with the structural fragility predicted by fixed-β critic-error amplification (Lemma 1).

External KL is insufficient; TRQAM enforces the budget. External KL regularization partially mitigates the collapse, but the realized path-space KL substantially exceeds the prescribed bound across every budget, leaving the policy vulnerable to critic-error amplification. TRQAM, by contrast, tracks each budget tightly throughout offline training and across the offline-to-online transition, and remains stable across the full sweep εKL ∈ {0.01,0.03,0.1,0.5,1,1.5}. (See Figure 14, 15, 16).

### I Broader impacts

TRQAM is a methodological contribution to stable off-policy fine-tuning of pretrained flow-matching policies, evaluated entirely on simulated benchmarks (OGBench, Robomimic). On the positive side, more reliable fine-tuning of pretrained robot policies can reduce the data and compute required to specialize useful behaviors and lessen the brittleness of pretrained-prior degradation under TD bootstrapping, which is a common source of instability in real-world RL deployments. On the negative side, the same trust-region machinery makes off-policy improvement of expressive flow policies more reliable, which could in principle accelerate the deployment of autonomous systems whose downstream uses we cannot fully anticipate; in particular, when applied to settings beyond the simulated benchmarks studied here, fine-tuned policies could exhibit failure modes (e.g., physical safety violations in robotics) that an algorithmic KL bound does not directly address. We view standard mitigations—safety filters at deployment, sandboxed evaluation, and explicit reward and constraint design—as necessary complements rather than substitutes for the algorithmic guarantees provided by TRQAM. The paper does not release pretrained models or datasets that pose dual-use or high-risk concerns.

#### Table 7: Full offline results at 1M training steps (8 seeds).

FQL CGQL-L DSRL IFQL QAM QAM-E TRQAM

- task1 24±44 62±38 45±12 7±11 96±3 93±5 95±4

- task2 0±0 21±36 73±7 9±14 36±49 91±4 87±6

- task3 73±14 88±5 84±5 62±24 86±7 93±3 93±3

- task4 0±0 0±0 0±0 35±22 0±0 66±9 75±17

- task5 92±4 68±42 63±8 34±24 94±3 89±5 96±3

antmaze-large

agg. (5 tasks) 38±9 48±7 53±2 29±8 62±9 86±3 89±4

- task1 1±1 11±12 2±3 0±0 37±9 0±0 48±5

- task2 0±0 14±12 0±0 43±13 18±19 0±0 21±12

- task3 0±0 0±0 0±0 14±8 2±2 0±0 5±5

- task4 0±0 0±0 0±0 1±2 20±24 0±0 58±13

- task5 10±29 8±23 2±2 2±2 69±8 28±38 75±7

antmaze-giant-10M

agg. (5 tasks) 2±6 7±5 1±1 12±3 29±4 6±8 41±4

- task1 84±21 84±9 24±23 93±3 33±34 28±30 87±5

- task2 99±1 99±1 88±6 94±4 100±1 98±2 89±7

- task3 89±9 0±0 64±27 96±4 90±8 74±18 90±6

- task4 0±0 0±0 0±0 82±8 0±0 0±0 61±7

- task5 99±1 100±1 88±5 99±2 100±1 99±1 95±4

humanoidmaze-medium

agg. (5 tasks) 74±5 57±2 53±10 93±2 64±7 60±6 84±3

- task1 1±2 16±11 1±1 32±5 6±10 9±14 45±8

- task2 0±0 0±0 0±0 6±9 0±0 0±0 10±6

- task3 7±4 16±8 2±3 76±6 7±5 3±4 34±15

- task4 0±0 0±1 0±0 35±26 0±0 0±0 44±13

- task5 0±1 0±0 1±1 0±0 5±11 9±18 45±6

humanoidmaze-large

agg. (5 tasks) 2±1 6±3 1±1 30±7 4±3 4±5 36±4

- task1 100±0 100±0 100±0 99±1 100±0 100±0 100±0

- task2 100±1 99±1 100±0 2±2 100±0 100±0 100±0

- task3 82±5 93±7 100±1 77±7 94±4 93±4 100±1

- task4 60±22 0±0 100±0 2±2 26±21 22±30 93±6

- task5 8±7 0±0 0±0 0±0 0±0 0±0 0±1

scene

agg. (5 tasks) 70±5 58±1 80±0 36±1 64±4 63±6 79±1

- task1 87±11 0±0 100±0 100±1 75±15 99±2 100±0

- task2 37±39 0±0 100±0 23±7 0±0 100±0 100±0

- task3 0±0 0±0 100±0 82±11 0±0 76±11 100±0

- task4 0±0 0±0 100±0 29±8 0±0 87±6 100±1

- task5 3±6 0±0 100±0 86±9 0±1 84±20 100±0

- puzzle-3x3

agg. (5 tasks) 25±10 0±0 100±0 64±4 15±3 89±4 100±0

- task1 3±3 1±1 90±5 74±11 3±4 90±10 100±0

- task2 6±9 0±0 9±9 4±3 1±1 43±22 99±1

- task3 23±15 0±0 81±20 84±6 2±3 57±10 98±2

- task4 7±5 0±0 76±28 46±13 0±1 31±16 100±1

- task5 9±16 0±0 49±43 3±1 1±1 50±22 98±5

- puzzle-4x4-10M

agg. (5 tasks) 9±7 0±0 61±8 42±4 1±1 54±8 99±1

- task1 75±10 83±4 85±7 18±7 96±3 94±4 98±2

- task2 59±11 66±6 85±5 9±2 81±6 82±7 92±3

- task3 39±8 72±7 82±10 8±4 75±7 67±8 80±12

- task4 6±3 15±5 35±6 2±2 24±6 29±6 54±8

- task5 44±9 37±6 71±4 7±3 78±6 82±6 82±6

cube-double

agg. (5 tasks) 44±4 55±2 72±4 9±2 71±2 71±3 81±3

- task1 26±20 2±3 70±18 51±22 56±22 26±11 87±8

- task2 1±1 0±0 16±7 16±3 15±11 14±9 42±9

- task3 8±6 0±0 40±8 28±8 21±11 9±6 45±9

- task4 2±3 0±0 15±5 1±1 4±3 3±3 30±13

- task5 1±1 0±0 31±12 23±8 1±1 2±4 48±17

cube-triple-10M

agg. (5 tasks) 7±5 0±1 34±6 24±7 19±6 11±4 50±5

- task1 36±25 7±5 36±12 15±11 70±10 45±19 66±16

- task2 4±5 0±1 5±3 5±7 2±4 0±0 18±11

- task3 3±3 0±0 2±2 6±8 16±11 2±4 11±7

- task4 0±0 0±0 0±1 0±0 2±2 0±0 3±2

- task5 0±0 0±0 1±1 2±3 0±0 0±0 0±0

cube-quadruple-100M

agg. (5 tasks) 9±5 1±1 9±3 6±3 18±3 9±3 19±5 all agg. (50 tasks) 28 23 46 35 35 45 68

εKL = 0.5 1 1.5 2 2.5 3 3.5 4

antmaze–giant–task1

antmaze–large–task1

100

| | |
|---|---|
| | |

| |
|---|

50

| |
|---|

0

0 10 15

0 10 15

humanoidmaze–medium–task1

humanoidmaze–large–task1

100

| | |
|---|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>|

| |
|---|

50

| |
|---|

| |
|---|

0

0 10 15

0 10 15

puzzle–3x3–task4

puzzle–4x4–task4

100

| | |
|---|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

successrate(%)

50

0

0 10 15

0 10 15

cube–double–task2

cube–triple–task2

100

| | |
|---|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |

| |
|---|

| |
|---|

50

0

0 10 15

0 10 15

scene–task2

cube–quadruple–task2

100

| | |
|---|---|

50

0

0 10 15

0 10 15

training steps (×105)

Figure 9: Hyperparameter εKL sweep on OGBench [38] (8 seeds). We sweep εKL ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0} on the default tuning task of each domain, with all other hyperparameters fixed to the main-comparison setting. Each panel reports success rate over training steps, with the offline-to-online transition at 106 steps. Shaded regions denote ±1 standard deviation across seeds; evaluated tasks are listed in the panel titles. Two patterns emerge. Tight budgets are best on humanoidmaze-medium, humanoidmaze-large, cube-double, cube-triple, and cube-quadruple, while larger budgets monotonically improve performance on puzzle-4x4, consistent with its notably larger state space. Smaller εKL also produces slower online adaptation across the sweep, consistent with the dual update tightening the trust region.

QAM (β)

0.01 0.03 0.1 0.3 1 3

100

|| |
|---|
<br><br>| |
|---|---|

| | | |
|---|---|---|
| | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>| |
| | | |
| || |
|---|
<br><br>| |

1020

Successrate(%)

Adjointloss

1015

50

1010

105

100

0

0 10 15

0 10 15

QAM-E (β, σa)

(0.03, 0.03) (0.03, 0.1) (0.1, 0.03) (0.1, 0.1) (0.3, 0.03) (0.3, 0.1) (1, 0.03) (1, 0.1)

100

|| |
|---|
<br><br>| |
|---|---|

| | | |
|---|---|---|
| | || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
| | | |
| | | |

1020

Successrate(%)

Adjointloss

1015

50

1010

105

100

0

0 10 15

0 10 15

TRQAM (εKL)

0.01 0.03 0.1 0.5 1 1.5

100

| | |
|---|---|

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| || |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>|

1020

Successrate(%)

Adjointloss

1015

50

1010

105

100

0

0 10 15

0 10 15

Training Steps (×105)

Figure 10: Hyperparameter sweep on Robomimic-lift (8 seeds). Sweep ranges for QAM, QAM-E, and TRQAM follow table 5. Left column reports success rate; right column reports adjointmatching loss on log scale. Fixed-temperature methods (QAM, QAM-E) exhibit adjoint loss growth of 10 to 20 orders of magnitude across most settings, with corresponding success rate collapse. TRQAM remains stable across all six budgets. Shaded regions denote ±1 standard deviation across seeds.

QAM (β)

0.01 0.03 0.1 0.3 1 3

1025

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

| | |
|---|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
| | |
|| |
|---|
<br><br>| |
|| |
|---|
<br><br>| |
|---|
<br><br>| |

Successrate(%)

1020

Adjointloss

1015

50

1010

105

100

0

0 10 15

0 10 15

QAM-E (β, σa)

(0.03, 0.03) (0.03, 0.1) (0.1, 0.03) (0.1, 0.1) (0.3, 0.03) (0.3, 0.1) (1, 0.03) (1, 0.1)

1025

100

|| |
|---|
| |
<br><br>| |
|---|---|

Successrate(%)

1020

Adjointloss

1015

| |
|---|

| |
|---|

50

1010

| |
|---|

105

100

0

0 10 15

0 10 15

TRQAM (εKL)

0.01 0.03 0.1 0.5 1 1.5

1025

100

|| |
|---|
<br><br>| |
|---|---|

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| || |
|---|
<br><br>| |
|---|
<br><br>| |

Successrate(%)

1020

Adjointloss

1015

50

1010

105

100

0

0 10 15

0 10 15

Training Steps (×105)

Figure 11: Hyperparameter sweep on Robomimic-can (8 seeds). Same setup as fig. 10. Fixedtemperature collapse is again hyperparameter-wide: QAM and QAM-E exhibit adjoint loss growth of 10 to 25 orders of magnitude across most settings with success rate collapse, while TRQAM remains stable across all six budgets. Shaded regions denote ±1 standard deviation across seeds.

QAM (β)

0.01 0.03 0.1 0.3 1 3

1025

100

|| |
|---|
<br><br>|| |
|---|
<br><br>|
|---|---|

Successrate(%)

1020

Adjointloss

1015

50

1010

| |
|---|

| |
|---|

| |
|---|

| |
|---|

105

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

100

0

0 10 15

0 10 15

QAM-E (β, σa)

(0.03, 0.03) (0.03, 0.1) (0.1, 0.03) (0.1, 0.1) (0.3, 0.03) (0.3, 0.1) (1, 0.03) (1, 0.1)

1025

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
| |
<br><br>|
|---|---|

| | | |
|---|---|---|
| | | |
| | | |
| || |
|---|
<br><br>| |
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| |
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>|

Successrate(%)

1020

Adjointloss

1015

50

1010

105

100

0

0 10 15

0 10 15

TRQAM (εKL)

0.01 0.03 0.1 0.5 1 1.5

1025

100

|| |
|---|
<br><br>| |
|---|---|

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|

Successrate(%)

1020

Adjointloss

1015

50

1010

105

100

0

0 10 15

0 10 15

Training Steps (×105)

Figure 12: Hyperparameter sweep on Robomimic-square (8 seeds). Same setup as fig. 10. While the adjoint loss explosion is less severe than on lift and can, fixed-temperature variants still exhibit signs of instability across the sweep, with adjoint loss steadily growing during offline training. TRQAM, in contrast, maintains a bounded adjoint loss and matches or exceeds the fixed-temperature variants across the sweep. Shaded regions denote ±1 standard deviation across seeds.

TRQAM QAM QAM-E DSRL

Lift

Can

Square

100

100

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| |
<br><br>| |
|---|---|

Successrate(%)

| |
|---|

50

50

50

| |
|---|

| |
|---|

| |
|---|

0

0

0

0 10 15

0 10 15

0 10 15

Training Steps (×105)

Figure 13: Best per-method hyperparameters on Robomimic across all three tasks (8 seeds). For each method (TRQAM, QAM, QAM-E, DSRL) we select the configuration with the strongest overall offline-to-online learning curve from the sweep in table 5. Shaded regions denote ±1 standard deviation across seeds.

|| |
|---|
<br><br>| |
|---|
| |
|---|---|
| | |

Successrate(%)

| |
|---|

| |
|---|

- 100

- 101

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

10−1

10−2

target εKL

10−3

0

0 10 15

0 10 15

εKL = 0.03

- 100

- 101

- 102

100

| |
|---|

Successrate(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10−1

0.03

10−2

target εKL

10−3

0

0 10 15

0 10 15

εKL = 0.1

- 100

- 101

- 102

100

Successrate(%)

| |
|---|

| |
|---|

| |
|---|

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10−1

target εKL

10−2

0

0 10 15

0 10 15

εKL = 0.5

- 100

- 101

- 102

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| |
|---|---|

Successrate(%)

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

0.5

| |
|---|

| |
|---|

10−1

target εKL

10−2

0

0 10 15

0 10 15

εKL = 1

- 100

- 101

- 102

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|
| | |

Successrate(%)

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

target εKL

10−1

0

0 10 15

0 10 15

εKL = 1.5

- 100

- 101

- 102

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| |
|---|---|
| | |

Successrate(%)

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

1.5

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

target εKL

10−1

0

0 10 15

0 10 15

Training Steps (×105)

#### Figure 14: Internal vs. external KL regularization on Robomimic-lift across all budgets (8

Successrate(%)

- 100

- 101

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

KLestimate

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10−1

10−2

target εKL

10−3

0

0 10 15

0 10 15

εKL = 0.03

- 100

- 101

- 102

100

| | |
|---|---|
|| |
|---|
| |

Successrate(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10−1

0.03

10−2

target εKL

10−3

0

0 10 15

0 10 15

εKL = 0.1

- 100

- 101

- 102

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
|
|---|---|

Successrate(%)

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10−1

target εKL

10−2

0

0 10 15

0 10 15

εKL = 0.5

- 100

- 101

- 102

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| |
|---|---|

Successrate(%)

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

0.5

| |
|---|

10−1

target εKL

10−2

0

0 10 15

0 10 15

εKL = 1

- 100

- 101

- 102

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
|
|---|---|
| | |

Successrate(%)

KLestimate

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

target εKL

10−1

0

0 10 15

0 10 15

εKL = 1.5

- 100
- 101
- 102

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>|
|---|---|
| | |

| | | |
|---|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
|
| |target εKL|| |
|---|
<br><br>| |
|---|
<br><br>|

Successrate(%)

KLestimate

50

1.5

10−1

0

0 10 15

0 10 15

Training Steps (×105)

#### Figure 15: Internal vs. external KL regularization on Robomimic-can across all budgets (8

| | | |
|---|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| |
| | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
| | | |
| |target εKL| |

Successrate(%)

- 100
- 101

KLestimate

50

10−1

10−2

10−3

0

0 10 15

0 10 15

εKL = 0.03

- 100
- 101

100

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| |
|---|---|---|
| | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
| | | |
| | | |
| |target εKL| |

| |
|---|

Successrate(%)

KLestimate

10−1

50

0.03

| |
|---|

| |
|---|

| |
|---|

10−2

10−3

0

0 10 15

0 10 15

εKL = 0.1

- 100

- 101

- 102

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
|
|---|---|

Successrate(%)

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

10−1

target εKL

10−2

0

0 10 15

0 10 15

εKL = 0.5

- 100

- 101

- 102

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
|
|---|---|

Successrate(%)

KLestimate

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.5

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

target εKL

10−1

0

0 10 15

0 10 15

εKL = 1

- 100

- 101

- 102

100

| || |
|---|
|
|---|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |

Successrate(%)

KLestimate

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

target εKL

10−1

0

0 10 15

0 10 15

εKL = 1.5

- 100

1.5

- 101

100

| || |
|---|
|
|---|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| |

Successrate(%)

KLestimate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

target εKL

10−1

0

0 10 15

0 10 15

Training Steps (×105)

#### Figure 16: Internal vs. external KL regularization on Robomimic-square across all budgets (8 seeds). Each row plots one target KL budget εKL ∈ {0.01,0.03,0.1,0.5,1.0,1.5}. Left: success

antmaze-large (1M)

task 1

###### task 2

###### task 3

###### task 4

###### task 5

100

|| |
|---|
<br><br>| |
|---|---|

| | |
|---|---|

| | |
|---|---|

| |
|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

50

0

0 10 15

0 10 15

0 10 15

0 10 15

0 10 15

antmaze-giant (10M)

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>|
|---|---|

| || |
|---|
<br><br>|
|---|---|

| | |
|---|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

50

0

0 10 15

0 10 15

0 10 15

0 10 15

0 10 15

humanoidmaze-medium (1M)

100

successrate(%)

|| |
|---|
| |
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>|
|---|---|

|| |
|---|
<br><br>| |
|---|---|

|| |
|---|
<br><br>| |
|---|---|

| | |
|---|---|

|| |
|---|
<br><br>| |
|---|---|

50

0

0 10 15

0 10 15

0 10 15

0 10 15

0 10 15

humanoidmaze-large (1M)

100

| |
|---|

| || |
|---|
<br><br>|
|---|---|

|| |
|---|
<br><br>|
|---|

| || |
|---|
| |
<br><br>|
|---|---|

50

| |
|---|

0

0 10 15

0 10 15

0 10 15

0 10 15

0 10 15

scene (1M)

100

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |

| |
|---|
| |

50

0

0 10 15

0 10 15

0 10 15

0 10 15

0 10 15

training steps (×105)

Figure 17: Per-task offline-to-online learning curves on OGBench [38] (8 seeds) Suites: antmaze-large, antmaze-giant, humanoidmaze-medium, humanoidmaze-large, scene. Each panel reports success rate over training steps for all seven methods (TRQAM, QAM, QAM-E, FQL, IFQL, CGQL-Linex, DSRL); the offline-to-online transition occurs at 106 steps. Shaded regions denote ±1 standard deviation across seeds.

- puzzle-3x3 (1M)
- puzzle-4x4 (10M)

task 1

###### task 2

###### task 3

###### task 4

###### task 5

100

|| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

50

0

0

10 15

0 10 15

0 10 15

0 10 15

0 10 15

100

| || |
|---|
|
|---|---|

| || |
|---|
|
|---|---|

| || |
|---|
|
|---|---|

| | |
|---|---|

50

0

0 10 15

0 10 15

0 10 15

0 10 15

0 10 15

cube-double (1M)

100

successrate(%)

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|

50

0

0 10 15

0 10 15

0 10 15

0 10 15

0 10 15

cube-triple (10M)

100

|| |
|---|
<br><br>| |
|---|
| |
<br><br>| |
|---|
<br><br>| |
|---|---|

|| |
|---|
<br><br>| |
|---|---|

|| |
|---|
<br><br>| |
|---|---|

|| |
|---|
| |
<br><br>| |
|---|---|

| || |
|---|
<br><br>|
|---|---|

50

0

0 10 15

0 10 15

0 10 15

0 10 15

0 10 15

cube-quadruple (100M)

100

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|---|
|| |
|---|
<br><br>| |

| || |
|---|
<br><br>|
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| |
|---|
| |

50

0

0 10 15

0 10 15

0 10 15

0 10 15

0 10 15

training steps (×105)

Figure 18: Per-task offline-to-online learning curves on OGBench [38] (8 seeds) Suites: puzzle-3x3, puzzle-4x4, cube-double, cube-triple, cube-quadruple. Each panel reports success rate over training steps for all seven methods (TRQAM, QAM, QAM-E, FQL, IFQL, CGQLLinex, DSRL); the offline-to-online transition occurs at 106 steps. Shaded regions denote ±1 standard deviation across seeds.

