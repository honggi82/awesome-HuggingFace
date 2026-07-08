# arXiv:2505.17508v4[cs.LG]19Feb2026

### ON THE DESIGN OF KL-REGULARIZED POLICY GRADIENT ALGORITHMS FOR LLM REASONING

Yifan Zhang∗⋄,1,2 Yifeng Liu∗1 Huizhuo Yuan1 Yang Yuan Quanquan Gu1† Andrew Chi-Chih Yao3† 1University of California, Los Angeles 2Princeton University 3Tsinghua University

yifzhang@princeton.edu, liuyifeng@cs.ucla.edu qgu@cs.ucla.edu, andrewcyao@tsinghua.edu.cn

ABSTRACT

Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). KL regularization is ubiquitous, yet the design surface, choice of KL direction (forward vs. reverse), normalization (normalized vs. unnormalized), and estimator (k1/k2/k3), is scattered across the literature and often intertwined with off-policy estimation. We ask a focused question: under the off-policy setting, what weighting is required for each KL variant so that the surrogate we optimize yields the exact gradient of the intended KL-regularized objective? We answer this with a compact, unified derivation we call the Regularized Policy Gradient (RPG) view. RPG (i) unifies normalized and unnormalized KL variants and shows that the widely-used k3 penalty is exactly the unnormalized KL; (ii) specifies conditions under which REINFORCE-style losses with stop-gradient are gradient-equivalent to fully differentiable surrogates; (iii) identifies and corrects an off-policy importance-weighting mismatch in GRPO’s KL term; and (iv) introduces RPG-Style Clip, a clipped-importance-sampling step within RPG-REINFORCE that enables stable, off-policy policy-gradient training at scale. On mathematical reasoning benchmarks (AIME24, AIME25), RPG-REINFORCE with RPG-Style Clip improves accuracy by up to +6 absolute percentage points over DAPO. We extend our experiments to 8K context length, and RPG-REINFORCE with RPG-Style Clip achieves 52% accuracy on AIME25, surpassing the official Qwen3-4B-Instruct model (47%). Notably, RPG is a stable and scalable RL algorithm for LLM reasoning, realized via (a) a KL-correct objective, (b) clipped importance sampling, and (c) an iterative reference-policy update scheme. Project Page: https://github.com/complex-reasoning/RPG.

1 INTRODUCTION

Reinforcement learning (RL), particularly policy gradient (PG) methods, provides a powerful framework for solving sequential decision-making problems in complex environments. These methods have been successfully applied in diverse domains, ranging from robotics to game playing, and have recently become instrumental in fine-tuning large language models (LLMs) to align with human preferences and instructions (Ouyang et al., 2022) and enhancing the reasoning capabilities of LLMs (Shao et al., 2024; Guo et al., 2025). Classical PG algorithms like REINFORCE (Williams, 1992) optimize policies directly but often suffer from high gradient variance. Advanced methods like Proximal Policy Optimization (PPO) (Schulman et al., 2017) improve stability and sample efficiency, enabling large-scale applications, often by operating in an off-policy manner and employing techniques like training critic models for the estimation of value functions. Our theme in this paper is stability and scalability: which design choices in KL-regularized PG matter for robustness under off-policy sampling, and practical throughput on modern LLM stacks?

A crucial technique for stabilizing policy optimization, especially when deviating from strictly onpolicy updates or aiming to control policy complexity, is regularization. Kullback-Leibler (KL) divergence is a commonly used regularizer, penalizing the deviation of the learned policy πθ from

∗Equal contribution; †Corresponding authors; ⋄Work was done during a visit to UCLA.

Goal: Stable and Scalable LLM Reasoning

###### RPG Core Engine

###### Outputs

###### Inputs (Iteration t)

- 1. Construct J(θ(t)) = Eπ(t) θ

[R] − β · KL

- 2. Derive ∇θ(t)J(θ(t))
- 3. Formulate Surrogate Loss L(θ(t))
- 4. Optimize to get πθ(t+1)

Updated Policy

Current Policy πθ(t)

Reference πold(t) Rewards R(x)

πθ(t+1)

Key Design Configuration

- 3. KL Divergence:
- • Forward KL(πold∥πθ)
- • Reverse KL(πθ∥πold)

2. KL Form:

###### 1. Loss Estimator:

- • Normalized
- • Unnormalized (UKL / k3)

- • Fully Differentiable
- • REINFORCE-style

###### Update Reference: πold(t+1) ← πθ(t+1)

- Figure 1: Overview of the iterative Regularized Policy Gradient (RPG) framework proposed in this work. At each iteration t, the central RPG Core Engine processes inputs: the current policy πθ(t), a

reference policy πold(t), and associated rewards R(x). The engine’s operation encompasses four main steps: (1) constructing the KL-regularized objective J(θ(t)), which combines the expected reward

with a KL divergence term; (2) deriving the off-policy policy gradient ∇θ(t)J(θ(t)); (3) formulating a corresponding surrogate loss function L(θ(t)); and (4) optimizing the policy parameters to yield

an updated policy πθ(t+1), aimed at enhancing LLM reasoning capabilities. The specific behavior of the RPG Core Engine is configured by three key design choices: (i) the Loss Estimator type

(Fully Differentiable or REINFORCE-style with Stop-Gradient); (ii) the KL Form (Normalized or Un-normalized, e.g., using UKL / k3 estimators); and (iii) the KL Divergence Type (Forward KL(πold∥πθ) or Reverse KL(πθ∥πold)). The framework operates iteratively, with the updated policy πθ(t+1) from one iteration informing the inputs for the next, including the update of the reference policy πold(t+1), to facilitate continuous learning and performance improvement.

a reference policy πref (e.g., policy from previous iteration πθ

or a fixed prior policy πSFT). KL regularization helps prevent destructive policy updates, encourages exploration around known good policies, and can prevent catastrophic forgetting or overly confident outputs (Ouyang et al., 2022).

old

Despite the widespread use of KL regularization in methods such as PPO (often implicitly through reward penalties) and explicit formulations like GRPO (Shao et al., 2024), there exists a considerable variety in how the KL divergence is formulated and estimated. Different choices include Forward KL and Reverse KL, handling potentially unnormalized distributions (Minka et al., 2005) (leading to unnormalized KL (UKL) and unnormalized reverse KL (URKL) formulations), and the use of various estimators like the k2 and k3 estimators (Schulman, 2020) designed to potentially reduce variance or offer different properties compared to the standard log-ratio (k1) estimator. Furthermore, the interplay between the choice of KL formulation, the policy optimization setting (on-policy vs. off-policy), and the derivation of appropriate surrogate loss functions (fully differentiable vs. REINFORCE-style gradient estimators) can lead to subtle differences.

This paper provides systematic derivations and a unifying treatment of KL-regularized policy gradient methods, and revisits classical REINFORCE through the lens of clipped importance sampling. Our main contributions are summarized as follows:

- • We derive policy gradients and corresponding surrogate losses for Forward/Reverse KL, in normalized (KL) and unnormalized (UKL) forms, under off-policy sampling with importance weights.

- • We give both fully differentiable surrogates and REINFORCE-style losses (with stop-gradient) and prove their gradient-equivalence to the intended regularized objective (Proposition 4.1, Appendix L).
- • We introduce RPG-Style Clip, a clipped-importance-weighted REINFORCE estimator that substantially improves stability and variance control while preserving the RPG gradients.
- • We reveal the equality between the k3 estimator and unnormalized KL (Appendix D), and show that GRPO’s KL penalty omits an essential importance weight under off-policy sampling. We provide a corrected estimator and loss consistent with the intended objective.
- • We present an iterative training framework that periodically updates the reference model to satisfy KL constraints while allowing the policy to depart meaningfully from the initial checkpoint.
- • On math reasoning, RPG-REINFORCE (with RPG-Style Clip) yields stable and scalable training and outperforms DAPO by up to +6 absolute points on AIME24/25.
- • We extend our experiments to 8K context length and find that RPG-REINFORCE with RPG-Style Clip achieves 52% accuracy on AIME25, surpassing the official Qwen3-4B-Instruct model (47%) and outperforming strong baselines.

2 PRELIMINARIES

Policy gradient (PG) methods are a cornerstone of modern reinforcement learning (RL), optimizing parameterized policies πθ by estimating the gradient of an expected objective function J(θ) with respect to the policy parameters θ. Typically, J(θ) represents the expected cumulative discounted reward over trajectories τ = (s0,a0,r0,s1,...,sT,aT,rT) generated by the policy: J(θ) = Eτ∼π

##### [G(τ)],

θ

where G(τ) = Tt=0 γtrt is the trajectory return (with discount factor γ), and the expectation is taken over the trajectories sampled according to the policy πθ(a|s) and the environment dynamics p(s′|s,a). The Generalized Policy Gradient Theorem (GPPT) provides a foundation for deriving these gradients (see Appendix J for the proof).

- Proposition 2.1 (Generalized Policy Gradient Theorem). Let πθ(x) be a probability density or mass function parameterized by θ, representing the probability of sampling item x. Let f(x,θ) be a scalar-valued function associated with x, potentially depending on θ. Under suitable regularity

conditions, the gradient of the expectation Ex∼π

##### [f(x,θ)] with respect to θ is: ∇θEx∼π

θ

##### [f(x,θ)∇θ log πθ(x) + ∇θf(x,θ)]. (2.1)

##### [f(x,θ)] = Ex∼π

θ

θ

The term E[f∇log π] reflects how changes in θ affect the probability of sampling x, while E[∇f] reflects how changes in θ directly affect the function value f.

The classic REINFORCE algorithm (Williams, 1992) applies the GPPT to the standard RL objective J(θ) = Eτ∼π

[G(τ)]. In this case, f(τ,θ) = G(τ), the total trajectory return, which does not depend

θ

directly on θ (i.e., ∇θG(τ) = 0). The theorem simplifies, and the gradient can be expressed using per-timestep contributions (Sutton et al., 1998):

T

##### ∇θJ(θ) = Eτ∼π

Gt∇θ log πθ(at|st) ,

θ

t=0

where Gt = Tk=t γk−trk is the return-to-go from timestep t. Due to space limit, we defer the detailed introduction of REINFORCE to Appendix C.1.

- 2.1 KL REGULARIZATION IN POLICY GRADIENTS

A common technique to stabilize policy optimization, especially in off-policy settings or when fine-tuning large models, is regularization. The Kullback-Leibler (KL) divergence is frequently used to penalize the deviation of the learned policy πθ from a reference policy πref (which could be πθ

, an initial supervised fine-tuned model, or another prior). KL(P ∥Q) ≥ 0 with equality iff P = Q almost everywhere. It is asymmetric (i.e., KL(P ∥Q) ̸= KL(Q∥P)). Minimizing the forward KL KL(πref ∥πθ) encourages πθ to cover the support of πref (zero-forcing), while minimizing the reverse KL KL(πθ ∥πref) encourages πθ to be concentrated where πref has high probability mass (mode-seeking).

old

Adding a KL penalty to the RL objective, such as J(θ) = Eπ

[R] − β KL(πθ∥πref), helps control the policy update size, prevents large deviations from πref, encourages exploration near known good

θ

policies, and can mitigate issues like catastrophic forgetting or overly confident outputs, particularly relevant in LLM fine-tuning (Ouyang et al., 2022). For PPO (see Appendix C.2), this penalty can be incorporated implicitly via reward shaping: rt′ = rt − β log(πθ(at|st)/πref(at|st)). Alternatively, it can be added explicitly to the objective function, as in GRPO. The specific form of the KL divergence (forward/reverse), whether distributions are normalized (KL vs. UKL), and the choice of estimator (e.g., standard log-ratio vs. k3 estimator (Schulman, 2020)) can vary, leading to different properties (mode seeking v.s. zero-forcing) and gradient estimators, as explored later in this paper (Sections 3 and 4).

- 2.2 GROUP RELATIVE POLICY OPTIMIZATION (GRPO)

Group Relative Policy Optimization (GRPO) (Shao et al., 2024) adapts the PPO framework for training LLMs, notably by eliminating the need for a learned value function (critic). Instead of using

GAE, GRPO estimates the advantage Ai,t at token t of output oi based on the relative rewards within a group of G outputs {o1,...,oG} sampled from the old policy πθ

old

for the same prompt q.

Crucially, GRPO modifies the PPO objective by explicitly adding a KL regularization term directly to the objective function. Its objective (simplified notation) is:

JGRPO(θ) = Eq∼P(Q),{o

i}∼πold

1 G

G

i=1

1 |oi|

|oi|

t=1

Ji,tClip(θ) − β · KLest πθ(·|hi,t)∥πref(·|hi,t) ,

where hi,t = (q,oi,<t) is the history, Ji,tClip(θ) represents the PPO-Clip term from Eq. (C.3) applied using the group-relative advantage estimate Ai,t, and πref is a reference model (e.g., the initial SFT model). For the KL penalty, GRPO employs the k3 estimator form (Schulman, 2020), evaluated per token oi,t:

KLest(πθ∥πref) ≈ k3

πref(oi,t | hi,t) πθ(oi,t | hi,t)

=

πref(oi,t | hi,t) πθ(oi,t | hi,t) − log

πref(oi,t | hi,t) πθ(oi,t | hi,t) − 1.

This uses the functional form k3(y) = y − log y − 1 as discussed in Schulman (2020), applied with y = πref(oi,t|hi,t)/πθ(oi,t|hi,t). This form is related to the unnormalized reverse KL divergence, UKL(πθ∥πref) (see Section 3.2 and Appendix D for a detailed discussion). However, a key observation regarding GRPO’s KL penalty is its estimation. If the KL penalty in GRPO is intended to approximate β · UKL(πθ(·|hi,t)∥πref(·|hi,t)), its off-policy estimation (sampling oi,t from πold) would generally involve an importance weight wi,t = ππθ(oi,t|hi,t)

old(oi,t|hi,t) multiplying the k3 term. The direct subtraction without this weight means the gradient derived from GRPO’s objective does not, in general, correspond to the gradient of the intended off-policy objective JClip − β UKL(πθ∥πref). For clarity, a corrected off-policy estimator for the GRPO KL component at history hi,t is

KLGRPO-corrected(hi,t;θ) = Eo

i,t∼πold(·|hi,t) wi,t k3

πref(oi,t|hi,t) πθ(oi,t|hi,t)

,

which is consistent with URKL/UKL depending on direction (see Section 3 and Appendix D). Our results in Section 3 provide derivations for KL-regularized objectives that explicitly account for off-policy sampling via importance weights. Related work is detailed in Appendix A.

- 3 REGULARIZED POLICY GRADIENTS

In this section, we start from the KL regularized objective J(θ) = E[R] − β KL and we treat this as the exact target for training. Then we derive its true gradient under off-policy sampling. The derivation shows that we need precise importance weighting so that the gradient of the surrogate loss matches the gradient of this objective. The weights are different for Forward vs. Reverse KL, as summarized in Table 1. This viewpoint unifies many existing estimators within a single framework and clarifies why the KL term in GRPO can lead to unstable updates when its weighting is chosen improperly. In the main text, we focus on the unnormalized objectives (UFKL/URKL), while the normalized formulations (FKL/RKL) and their losses are deferred to Appendix E (see also Table 4). All proofs are provided in Appendix K.

- 3.1 UNNORMALIZED FORWARD KL REGULARIZATION

In scenarios where distributions might not be normalized (i.e., x π(x)dx ̸= 1), the standard KL divergence might not fully capture the dissimilarity. The unnormalized forward KL divergence

addresses this by adding a mass correction term. Let πold(x) be a potentially unnormalized reference measure with total mass Zold = x πold(x)dx. Let πold(x) = πold(x)/Zold be the corresponding normalized probability distribution, such that πold(x)dx = 1.

- Table 1: Summary of fully differentiable surrogate loss functions L(θ) for unnormalized KL-

regularized objectives (main text). Minimizing L(θ) corresponds to maximizing J(θ) = Eπ

θ

[R(x)]− β · Divergence. Samples x are drawn from πold = πold/Zold. These losses yield −∇θJ(θ) via differentiation. Notation: w(x) = πθ(x)/πold(x), R(x) is reward, β the regularization strength, and Zold = πold. Normalized counterparts are in Appendix E (Table 4).

Regularization (Unnormalized) Surrogate loss (expectation w.r.t. πold)

Forward (UFKL) Zold E −w(x)R(x) + β w(x) − log w(x) − 1 Reverse (URKL) Zold E −w(x)R(x) + β w(x)log w(x) − w(x)

Definition 3.1 (Unnormalized Forward KL). The unnormalized forward KL divergence (Minka et al., 2005; Zhu and Rohwer, 1995) between the measure πold and the density πθ is defined as:

UKL(πold∥πθ) =

x

πold(x)log

πold(x) πθ(x)

dx

Generalized KL

+

x

πθ(x) − πold(x) dx

Mass Correction

.

This form is particularly relevant when dealing with reference measures that may not be perfectly normalized or when connecting to certain KL estimators like k3 (see Remark 3.5).

Consider the objective using UKL regularization as follows:

JUFKL(θ) = Ex∼π

θ

[R(x)] − β UKL(πold∥πθ). (3.1)

To estimate this off-policy using samples from the normalized reference πold(x) = πold(x)/Zold, we define the importance weight w(x) = πθ(x)/πold(x) (using the unnormalized πold). The gradient and corresponding loss function, incorporating the total mass Zold of the reference measure, are given in Proposition 3.2.

Proposition 3.2 (Policy Gradient and Differentiable Loss for Unnormalized Forward KL). Consider the unnormalized KL regularized objective function in Eq. (3.1). The gradient of JUFKL(θ) is:

∇θJUFKL(θ) = ZoldEx∼ π

old

w(x)R(x) − β (w(x) − 1) ∇θ log πθ(x) .

The corresponding surrogate loss for gradient descent optimization, estimated using samples {xi} ∼ πold, is:

LUFKL(θ) = ZoldEx∼ π

old −w(x)R(x) + β w(x) − log w(x) − 1 , satisfying ∇θLUFKL(θ) = −∇θJUFKL(θ). Remark 3.3 (Interpretation of UFKL Loss and Gradient). The regularization component of the surrogate loss LUFKL(θ), specifically ZoldEx∼ π

old

[β(w(x) − log w(x) − 1)], corresponds to an off-policy estimate of the unnormalized forward KL divergence term β · UKL(πold∥πθ) present in the objective JUFKL(θ). This connection is established via the k3 estimator (see Remark 3.5 and Appendix D). Furthermore, the gradient term −β(w(x) − 1) effectively modifies the reward, guiding πθ to match not only the shape of πold but also its overall mass Zold, due to the mass correction component in UKL(πold∥πθ).

- 3.2 UNNORMALIZED REVERSE KL REGULARIZATION

Similar to the forward case, we can define an unnormalized reverse KL divergence, relaxing the normalization constraint on the reference distribution πold. Let πold(x) be a potentially unnormalized reference measure with total mass Zold = πold(x)dx. Let πold(x) = πold(x)/Zold be the corresponding normalized probability distribution.

Definition 3.4 (Unnormalized Reverse KL). The unnormalized reverse KL divergence between the density πθ and the measure πold is defined as:

πθ(x) πold(x)

UKL(πθ∥πold) =

πold(x) − πθ(x) dx

πθ(x)log

dx

##### +

##### .

x

x

Mass Correction

Generalized KL

The mass correction term simplifies to Zold − πθ(x)dx. Remark 3.5. (Equivalence to k3 estimator) The k3 estimator (Schulman, 2020), often used for its empirical properties (e.g., in GRPO (Shao et al., 2024)), is defined for a density ratio y(x) as:

k3(y) := y − 1 − log y. (3.2) As shown in Appendix D, this functional form directly relates to unnormalized KL divergences. For instance, KLk

[k3(πold(x)/πθ(x))] is equivalent to UKL(πθ∥πold). This equivalence relationship justifies the exploration of UKL/URKL formulations within our framework. Consider the objective using URKL:

##### (πθ∥πold) := Ex∼π

3

θ

[R(x)] − β UKL(πθ∥πold), (3.3) where UKL is defined above. As with UFKL, we derive the gradient and loss using expectations over the normalized reference πold and the importance weight w(x) = πθ(x)/πold(x) (with unnormalized πold). The results are summarized in Proposition 3.6.

JURKL(θ) = Ex∼π

θ

- Proposition 3.6 (Policy Gradient and Differentiable Loss for Unnormalized Reverse KL). Consider

the reverse unnormalized KL regularized objective function in Eq. (3.3). The gradient of JURKL(θ) is:

∇θJURKL(θ) = ZoldEx∼ π

##### w(x) R(x) − β log w(x) ∇θ log πθ(x) .

old

A corresponding surrogate loss for gradient descent optimization, estimated using samples {xi} ∼ πold, is:

LURKL(θ) = ZoldEx∼ π

old −w(x)R(x) + β w(x)log w(x) − w(x) ,

satisfying ∇θLURKL(θ) = −∇θJURKL(θ). The constant Zold scales the loss and gradient and may be omitted in practice.

- Remark 3.7 (URKL Loss and Mass Correction). The surrogate loss LURKL(θ) is designed such

that its gradient is −∇θJURKL(θ). Specifically, the term ZoldEx∼ π

old

[β(w(x)log w(x) − w(x))] in the loss directly relates to the off-policy estimation of the unnormalized reverse KL divergence β UKL(πθ∥πold), omitting a constant related to the total mass Zold which does not affect the gradient. The policy gradient’s effective reward scaling factor, (R(x) − β log w(x)), is simpler than its normalized RKL counterpart.

- Remark 3.8. In Appendix B, we show the connection between RPG and the Natural Policy Gradient (NPG) (Kakade, 2001; Schulman et al., 2015). In particular, the NPG update is a special case of the RPG update, which uses a linear approximation for the expected return and a quadratic approximation for the KL regularization. This transforms the problem from simple first-order gradient ascent in PG (REINFORCE) into a second-order-like update: RPG.

- 3.3 RPG-STYLE CLIP: DUAL-CLIP TRUNCATION OF IMPORTANCE RATIOS

Large importance ratios w(x) = π

θ(x)

πold(x) induce high variance and destabilize off-policy updates. Our RPG-Style Clip follows the dual-clip method implemented in Algorithm 1 in the Appendix: we clip w into [1 − ϵ1, 1 + ϵ2] and additionally impose a lower bound for negative advantages. Let A(x;θ) denote the regularized advantage analogue determined by the chosen objective (e.g., AURKL = (R−b)−β log w, ARKL = (R−b)−β(log w+1)). The loss used in our implementation is

 

max − w(x) A(x;θ), −clip(w(x), 1 − ϵ1, 1 + ϵ2) A(x;θ) , A(x;θ) ≥ 0, min max − w(x) A(x;θ), −clip(w(x), 1 − ϵ1, 1 + ϵ2) A(x;θ) , −c A(x;θ) , A(x;θ) < 0,

with

LRPG-Clip(x,θ) =



ϵ1,ϵ2 > 0 and c > 1. The choice of A for each divergence (URKL/UFKL/RKL/FKL) matches the gradients in Section 3 and is instantiated in Algorithm 1.

Case A(x) < 0 (e.g., A = −1)

Case A(x) ≥ 0 (e.g., A = 1)

0

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | |Gra Gra<br><br>|d via d = 0|w (w.r.t|w)|

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | |Gra Gra<br><br>|d = 0 d via|(w.r.t w|w)|
| | | | | | | | |

DualClipLosstermL(x, θ)

DualClipLosstermL(x, θ)

−0.5

−1

−1.5

−20.5 1 1.5 2 2.5

- 0.5 1 1.5 2 2.5

- 0.5
- 1

1.5

2

- 2.5

c

c

−1ϵ1

−1ϵ1

1+ϵ2

1+ϵ2

w(x) = πθ(x)/πold(x)

w(x) = πθ(x)/πold(x)

- Figure 2: Visualization of the Dual-Clip loss term LDualClip(x,θ) vs. importance weight w(x), as described in Section G.1 and Algorithm 1. This formulation is typically implemented as fully differentiable w.r.t θ (via w(x) and potentially A(x) if A depends on θ, e.g., via log w(x)), unlike

REINFORCE-style implementations that use SG( A) or SG(ℓi) within the loss. For visualization, A(x) is treated as constant ( A = 1 left, A = −1 right) to isolate the effect of w. Solid blue: Loss depends linearly on w, gradient ∇θL flows via w(x). Dotted magenta: Loss is constant w.r.t w, gradient ∇θL does not flow via w(x) in this segment (though it might flow via A if A depends on θ). Left: Case A < 0. Right: Case A ≥ 0.

- 4 REINFORCE-STYLE REGULARIZED POLICY GRADIENTS

- Table 2: REINFORCE-style surrogate losses L(θ) for unnormalized KL-regularized objectives using the stop-gradient operator (SG). These losses yield the target gradient via automatic differentiation. Compare with the fully differentiable losses in Table 1. Normalized versions are given in Appendix F.

Regularization (Unnormalized) REINFORCE-style loss (sampling x ∼ πold) Forward (UFKL) −E SG(Zold(w(x)R(x) − β(w(x) − 1)))log πθ(x) Reverse (URKL) −E SG(Zoldw(x)(R(x) − β log w(x)))log πθ(x)

In Section 3, we derived policy gradient estimators and corresponding fully differentiable surrogate losses L(θ) for KL-regularized objectives. Those losses were constructed such that ∇θL(θ) = −∇θJ(θ) directly, typically by setting L(θ) = −JIS(θ) (where JIS is the importance-sampled objective) up to constants. Notice that the gradients derived in Section 3 (Theorems 3.2 through 3.6) share a structural similarity with the REINFORCE estimator:

[Weight(x,θ)∇θ log πθ(x)]

∇θJ(θ) = Ex∼π

sampling

where πsampling is πold or its normalized version πold, and Weight(x,θ) encapsulates the reward and KL regularization terms, differing for each specific objective.

- Proposition 4.1 (Gradient-Equivalence of Surrogates). For each KL-regularized objective J(θ) derived in Section 3, the corresponding REINFORCE-style losses in Table 2 satisfy ∇θL(θ) = −∇θJ(θ) under the standard regularity assumptions used in the policy-gradient theorem. In particular, the stop-gradient operator ensures that dependence of the weight on θ (through importance ratios) does not leak unintended gradients. A proof sketch follows directly from the policy-gradient theorem and is completed in Appendix L.

This structural similarity motivates an alternative REINFORCE-style implementation using the stopgradient operator SG. The general form of such losses and the detailed rationale for how they yield the target gradient via automatic differentiation are presented in Appendix F.1 (see Eq. (F.1)).

We explore these REINFORCE-style estimators as part of our framework, as they offer an alternative implementation path and demonstrate stronger or competitive empirical performance (Section 5). Proofs are in Appendix L. In the main text, we tabulate the unnormalized REINFORCE-style losses; normalized counterparts are deferred to Appendix F.

- 4.1 RPG-STYLE DUAL CLIP FOR REINFORCE-STYLE ESTIMATORS

Directly optimizing the REINFORCE-style losses in Table 2 can be unstable due to high variance in the importance weights w(x). To mitigate this, we introduce RPG-Style Clip, which adapts the Dual-Clip technique (Ye et al., 2020) to our regularized setting.

The key strategy is to decompose the weight term inside the stop-gradient operator into the importance ratio w(x) and a regularized advantage Areg(x). For example, for the URKL objective, we identify Areg(x) = Zold(R(x)−β log w(x)) such that the weight is w(x) Areg(x). We then replace this linear weighting with a clipped version C(w(x),SG( Areg(x))). The resulting loss is:

LRPG-Clip(θ) = −Ex∼ π

old C w(x),SG( Areg(x)) log πθ(x) , (4.1)

where C(w,A) applies dual-clip bounds, clipping w to [1 − ϵ1,1 + ϵ2] for positive A, and enforcing a lower bound c for negative A, as detailed in Appendix G.1. This ensures stable off-policy updates while preserving the correct gradient direction derived from the KL-regularized objective.

LosscoefficientLi

###### Case ψi ≥ 0

Grad via ℓi

- 0.5 1 1.5 2 2.5

- 0.5
- 1

- 1.5

Grad = 0

c

1+ϵ2

−1ϵ1

wi = πθ(xi)/πold(xi)

###### Case ψi < 0

Grad = 0

−1

LosscoefficientLi

Grad via ℓi

−1.5

−2

−2.50.5 1 1.5 2 2.5

c

1+ϵ2

−1ϵ1

wi = πθ(xi)/πold(xi)

- Figure 3: Visualization of the loss coefficient Li vs. importance weight wi based on the specific implementation in Algorithm 2 as RPG-REINFORCE. This version swaps the main branching condition compared to previous versions (branches on ψi > 0). The plot assumes ℓi = −log πθ(xi) = 1 for visualizing the value of Li. The line styles indicate the nature of the gradient ∇θLi: Solid blue: Gradient exists, flowing only via ℓi. The coefficient multiplying ∇θℓi depends on SG(wi). Dotted magenta: Gradient is zero. This occurs when ℓi is detached via SG in the loss calculation. Left: Case ψi ≥ 0. Right: Case ψi < 0.

- 5 EXPERIMENTS

In this section, we empirically evaluate our proposed Regularized Policy Gradient (RPG) framework, including both its fully differentiable (RPG) and REINFORCE-style (RPG-REINFORCE) variants. We compare their performance against established baselines on challenging mathematical reasoning tasks using large language models, including GRPO (Shao et al., 2024) and DAPO (Yu et al., 2025). Our evaluation focuses on task-specific accuracy, training stability, and key training dynamics such as reward, policy entropy, and response length.

Base Models and Datasets. We conduct experiments using the Qwen3-4B and Qwen2.5-7B-Instruct models. For training, we utilize the DAPO-Math-17k dataset (Yu et al., 2025) (13.9k English samples). We evaluate on AIME2024, AIME2025, and AMC23, and additionally report results on MinervaMath and OlympiadBench in the Appendix. We compare against baselines including GRPO, DAPO, and REINFORCE++.

(a) AIME24 (b) AIME25 (c) AMC23

(d) Reward (Critic Score) (e) Entropy (f) Response Length

- Figure 4: Training dynamics and benchmark performance for fully differentiable Regularized Policy Gradient (RPG) and REINFORCE-Style RPG (RPG-REINFORCE) compared to baselines (GRPO and DAPO) with 8k context length.

- Table 3: Combined performance metrics with 8K context length on the AIME24, AIME25, and AMC23 mathematical reasoning benchmarks, showing “Last” and “Best” scores. The “Last” score is from the 400th training step, assuming the training process remained stable to that point. The highest score in each column is bolded, and the second highest is underlined. RPG and RPG-REINFORCE methods are highlighted with light cyan and light green backgrounds, respectively.

AIME24 AIME25 AMC23 Last Best Last Best Last Best

Method

GRPO 0.3750 0.4396 0.3354 0.4063 0.9109 0.9297 DAPO 0.5438 0.5740 0.4469 0.4740 0.9375 0.9430

RPG-UFKL 0.5938 0.6177 0.4698 0.4865 0.9492 0.9517 RPG-URKL 0.4542 0.5260 0.5261 0.4938 0.9406 0.9539

RPG-REINFORCE-UFKL 0.5906 0.5958 0.4833 0.5031 0.9453 0.9469 RPG-REINFORCE-URKL 0.5708 0.5781 0.5073 0.5208 0.9398 0.9469

Implementation and Framework. Experiments are implemented using the verl framework (Sheng et al., 2025) with the vLLM engine (Kwon et al., 2023) for efficient LLM serving and inference. For practical implementation of our RPG methods, we emphasize that the probabilities (or logprobabilities) from the last iteration’s model (πold) for the sampled data can be pre-computed and stored. This allows the KL regularization terms to be calculated without needing to keep πold in GPU memory during the training step of the current policy πθ. Consequently, only one model (πθ) needs to be actively managed in GPU memory for training, which is faster and more memory-efficient compared to approaches like GRPO that typically require access to at least two models (the current policy and a reference/sampling policy) during optimization.

Iterative reference updates. To further stabilize optimization, we adopt an iterative reference-update scheme: we periodically set πold ← πθ (every K optimizer steps, or when a moving average of token-level KL exceeds a target κ). This realizes a practical KL trust region while avoiding overregularization toward the initial checkpoint. Further implementation details and hyperparameters (learning rate, β, clipping) are provided in Appendix H.

Stabilization and Advanced RL Techniques. Our RPG implementations (both fully differentiable and REINFORCE-style) incorporate stabilization techniques like baseline subtraction and PPO-style objective clipping (specifically, Dual-Clip (Ye et al., 2020; Schulman et al., 2017)), crucial for robust off-policy learning. Detailed algorithmic descriptions are provided in Appendix G (see Algorithm 1 for RPG with Dual-Clip and Algorithm 2 for the REINFORCE-style equivalent, along with Figures 3 and 2 for visualization). Varying the clip ratios in REINFORCE-style RPG algorithms, we find that while critic scores and response lengths are similar for (ϵ1,ϵ2) = (0.1,0.1) and (0.2,0.28) (Figure 8), DAPO’s higher-and-clip-higher strategy substantially reduces actor entropy, which appears to underlie its performance gains, details can be found in Appendix I.2.1. For PPO-style clipping, we set (ϵ1,ϵ2) = (0.2,0.28) for RPG, RPG-REINFORCE and DAPO. For GRPO, we use (ϵ1,ϵ2) = (0.2,0.2). Furthermore, to enhance training efficiency and data quality, we adopted techniques introduced by DAPO (Yu et al., 2025), including a dynamic sampling strategy with a group filtering mechanism, which oversamples challenging prompts and filters out those with near-perfect or nearzero accuracy based on initial rollouts and an overlong punishment component in the reward shaping to discourage excessively verbose outputs. In addition, we enable RPG-Style Clip (Section 3.3) for the REINFORCE-style estimators, which we found to be the best variant for RL training at larger scales.

Results and Discussion. We display the curves in Figure 4 and last and best scores on AIME24 and AIME25 benchmarks in Table 3 for the experiments with 8K context length. The results also demonstrate the superiority of our algorithms over baselines, including GRPO and DAPO. With 8k context length, RPG-REINFORCE achieves 52% accuracy on AIME25, surpassing the official Qwen3-4B-Instruct baseline (47%). Tables 5 and 6 in Appendix I summarize the performance of our RPG algorithms against baselines with 4k and 2k context lengths, reporting both the last and best scores achieved during training on these benchmarks. Figures 5 and 6 in Appendix I complement these results by illustrating the evaluation scores and training dynamics for the fully differentiable RPG variants and baselines when training the Qwen-3-4B model using 4k and 2k context length, respectively. These figures display performance on the AIME24 and AIME25 benchmarks, alongside key training metrics: reward (critic score), policy entropy, and average response length. Across settings, the RPG-REINFORCE variants with RPG-Style Clip have the strongest results. Following DAPO (Yu et al., 2025; Yue et al., 2025), we report “Mean@32” (average accuracy of 32 sampled responses).

Moreover, these algorithms generally exhibit stable training progressions regarding reward (critic score) and policy entropy, as shown in subfigures (c) and (d) in Figures 5 and 6 in the Appendix, compared to some baselines like GRPO, which can show more volatility. This stability likely contributes to their robust benchmark performances (subfigures a-b). The response lengths (subfigure e) for RPG methods also appear well-controlled. These observations align with the strong final scores reported in Tables 5 and 6 for these variants.

- 6 CONCLUSION

We introduced RPG, a framework for deriving and organizing KL-regularized policy gradient algorithms for online, off-policy RL. We provided derivations for policy gradients and surrogate loss functions covering forward/reverse KL, normalized/unnormalized distributions, and both fully differentiable and REINFORCE-style estimators. Beyond derivations, we revisited the classical REINFORCE algorithm and made it viable off-policy through RPG-Style Clip and iterative reference updates. On LLM reasoning, these design choices deliver stable and scalable training with competitive and superior accuracy relative to strong baselines.

ETHICS STATEMENT

The methods developed in this paper contribute to the broader effort of enhancing the reasoning capabilities of large language models. Improved reasoning in LLMs has the potential to significantly benefit various fields, including scientific discovery, education, and complex problem-solving in engineering and medicine. By providing more stable and efficient training algorithms, our work can facilitate the development of more reliable and capable AI systems.

However, as with any advancement in AI capabilities, it is crucial to consider the ethical implications and ensure responsible development and deployment of these technologies to mitigate potential

misuse. While our framework offers a unified perspective on KL-regularized policy gradient algorithms and demonstrates strong empirical performance, it has certain limitations. RPG-Style Clip introduces a controllable bias: variance trade-off through (ϵ1,ϵ2), so developing principled schedules for clipping would be valuable. We used LLMs as assistive tools to polish part of this paper. The roles of LLMs in this work are restricted to improving readability and presentation.

REFERENCES

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12248–12267, 2024.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pages 4447–4455. PMLR, 2024.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, 2024.

Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 4299–4307, 2017.

Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. RAFT: reward ranked finetuning for generative foundation model alignment. Trans. Mach. Learn. Res., 2023, 2023.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.

Kaixuan Ji, Guanlin Liu, Ning Dai, Qingping Yang, Renjie Zheng, Zheng Wu, Chen Dun, Quanquan Gu, and Lin Yan. Enhancing multi-step reasoning abilities of language models through direct q-function optimization. arXiv preprint arXiv:2410.09302, 2024.

Sham M Kakade. A natural policy gradient. Advances in neural information processing systems, 14, 2001.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment. arXiv preprint arXiv:2410.01679, 2024.

Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free! ICLR 2019 workshop: Deep RL Meets Structured Prediction, 2019.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.

Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, 2024.

Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J. Liu, and Jialu Liu. Statistical rejection sampling improves preference optimization. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019, 2019.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a referencefree reward. Advances in Neural Information Processing Systems, 37:124198–124235, 2024.

Tom Minka et al. Divergence measures and message passing. Technical report, Microsoft Research, 2005.

Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Côme Fiegel, Andrea Michi, Marco Selvi, Sertan Girgin, Nikola Momchev, Olivier Bachem, Daniel J. Mankowitz, Doina Precup, and Bilal Piot. Nash learning from human feedback. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, 2024.

OpenAI. ChatGPT, 2022. URL https://chat.openai.com/.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

John Schulman. Approximating kl divergence. http://joschu.net/blog/kl-approx. html, March 2020. Accessed on February 20, 2026.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. In Yoshua Bengio and Yann LeCun, editors, 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 - 3 April 2025, pages 1279–1297. ACM, 2025.

Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement

learning. Machine learning, 8:229–256, 1992.

Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025, 2025.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: bridging theory and practice for rlhf under kl-constraint. In Proceedings of the 41st International Conference on Machine Learning, pages 54715–54754, 2024.

Jing Xu, Andrew Lee, Sainbayar Sukhbaatar, and Jason Weston. Some things are more cringe than others: Preference optimization with the pairwise cringe loss. arXiv preprint arXiv:2312.16682, 18, 2023.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng Wu, Qingwei Guo, et al. Mastering complex control in moba games with deep reinforcement learning. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 6672–6679, 2020.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

Yifan Zhang, Ge Zhang, Yue Wu, Kangping Xu, and Quanquan Gu. Beyond bradley-terry models: A general preference model for language model alignment. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Huaiyu Zhu and Richard Rohwer. Information geometric measurements of generalisation. Preprint, 1995.

## Appendix

- A Related Work 17
- B Connection Between Regularized Policy Gradient and Natural Policy Gradient 18
- C REINFORCE and Proximal Policy Optimization (PPO) 19

- C.1 REINFORCE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.2 Proximal Policy Optimization (PPO) . . . . . . . . . . . . . . . . . . . . . . . . . 19

- D Equivalence of k3 Estimator and Unnormalized KL Divergence 20
- E Normalized KL Regularization 20

- E.1 Forward KL Regularization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- E.2 Reverse KL Regularization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- F REINFORCE-Style Regularized Policy Gradients with Various KL Regularization Forms 21

- F.1 Rationale for REINFORCE-Style Loss Formulation . . . . . . . . . . . . . . . . . 21
- F.2 REINFORCE-Style RPG with Forward KL Regularization . . . . . . . . . . . . . 22
- F.3 REINFORCE-Style RPG with Unnormalized Forward KL Regularization . . . . . 22
- F.4 REINFORCE-Style RPG with Reverse KL Regularization . . . . . . . . . . . . . 22
- F.5 REINFORCE-Style RPG with Unnormalized Reverse KL Regularization . . . . . 22

- G More on Algorithmic Details 22

- G.1 Stabilization Techniques for Regularized Policy Gradients . . . . . . . . . . . . . 22
- G.2 Stabilization Techniques for REINFORCE-Style Regularized Policy Gradients . . 24

- H Detailed Experimental Setup 25
- I Additional Experiment Results 27

- I.1 The performance with 2k and 4k context length . . . . . . . . . . . . . . . . . . . 27
- I.2 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- I.3 Experiments on Qwen-2.5-7B-Instruct . . . . . . . . . . . . . . . . . . . . . . . . 29

- J Proof of Theorem 2.1 (Generalized Policy Gradient Theorem) 32
- K Proofs for Regularized Policy Gradients 32

- K.1 Proof of Proposition E.1 (Policy Gradient and Differentiable Loss for Normalized

- Forward KL) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

K.2 Proof of Proposition 3.2 (Policy Gradient and Differentiable Loss for Unnormalized

- Forward KL) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

- K.3 Proof of Proposition E.3 (Policy Gradient and Differentiable Loss for Normalized

- Reverse KL) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34

K.4 Proof of Proposition 3.6 (Policy Gradient and Differentiable Loss for Unnormalized

- Reverse KL) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35

#### L Proofs for REINFORCE-Style Regularized Policy Gradients 37

- L.1 Proof of Proposition F.1 (REINFORCE-style Policy Gradient for Forward KL) . . 37
- L.2 Proof of Proposition F.3 ((REINFORCE-style Policy Gradient for Unnormalized Forward KL) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
- L.3 Proof of Proposition F.4 (REINFORCE-Style Loss) . . . . . . . . . . . . . . . . . 38
- L.4 Proof of Proposition F.5 (REINFORCE-Style Loss for Unnormalized Reverse KL) 38

- A RELATED WORK

Fine-tuning large language models (LLMs) using human feedback has become a critical step in developing capable and aligned AI systems. Broadly, methods fall into two main categories: those relying on policy optimization using an explicit reward model learned from feedback, and those directly optimizing policies based on preference data.

RLHF via Policy Optimization. The classic RLHF involves training a reward model (RM) rϕ(x,y) to predict human preferences and then using reinforcement learning to optimize the language model

policy πθ to maximize the expected reward from the RM, often regularizing against deviating too far from an initial reference policy πref. This approach was pioneered by Christiano et al. (2017) and gained widespread prominence with its application to LLMs like InstructGPT (Ouyang et al., 2022) and ChatGPT (OpenAI, 2022), which utilized Proximal Policy Optimization (PPO) (Schulman et al., 2017). PPO became a workhorse due to its relative stability, achieved by constraining policy updates via a clipped surrogate objective. The standard PPO setup for RLHF involves the policy πθ, a value function Vψ, the RM rϕ, and the reference policy πref.

RLHF via Direct Preference Optimization. An alternative and increasingly popular approach bypasses explicit reward modeling by directly optimizing the policy πθ based on preference data, typically pairwise comparisons (yw,yl) indicating that response yw is preferred over yl for a given prompt x. Inspired by the Bradley-Terry model (Bradley and Terry, 1952), Direct Preference Optimization (DPO) (Rafailov et al., 2023) derived a simple loss function directly relating preference probabilities to policy likelihoods under πθ and a reference policy πref. DPO maximizes the relative likelihood of preferred responses using a logistic loss: LDPO ∝ −E[log σ(β∆logp)], where ∆logp is the difference in log-probabilities of yw and yl between πθ and πref. DPO’s simplicity and effectiveness led to its wide adoption in models like Llama-3 (Grattafiori et al., 2024), Qwen2 (Yang et al., 2024), and Phi-3 (Abdin et al., 2024). Numerous variants have followed: SLiC-HF (Zhao et al., 2023) uses a pairwise hinge loss for calibration; IPO (Azar et al., 2024) uses an identity link function; SimPO (Meng et al., 2024) offers a simpler objective focusing on the margin; KTO (Ethayarajh et al., 2024) handles binary (good/bad) feedback; DQO (Ji et al., 2024) incorporates direct Q-value modeling; RAFT (Dong et al., 2023), RSO (Liu et al., 2024) and RFT (Yuan et al., 2023) use a rejection sampling perspective. Recognizing that preferences might evolve, iterative methods like Iterative DPO (Xiong et al., 2024), PCO (Xu et al., 2023) and SPIN (Chen et al., 2024) alternate between generation/preference learning and policy updates, often using the current policy’s outputs in a self-improvement loop. Game theory offers another lens, with Nash Learning from Human Feedback (NLHF) (Munos et al., 2024) framing RLHF as finding a Nash equilibrium between policies. Self-play ideas appear in SPPO (Wu et al., 2025) and GPO (Zhang et al., 2025), where the policy generates pairs for comparison. Methods like GPM (Zhang et al., 2025) aim to handle more general preference structures efficiently using latent embeddings beyond pairwise comparisons.

RL for Enhancing LLM Reasoning. Beyond general alignment with human preferences, RL techniques are increasingly explored to specifically enhance the multi-step reasoning capabilities of LLMs in domains like mathematics, coding, and complex instruction following. In these contexts, RL optimizes the policy to generate sequences (e.g., chain-of-thought, code blocks) that lead to successful outcomes, often using rewards derived from external feedback like unit test results, execution outcomes, or correctness checks by an automated judge or specialized reward model trained on reasoning quality. For instance, the DeepSeekMath model (Shao et al., 2024) employed the GRPO algorithm, a value-free PPO variant, demonstrating significant improvements in mathematical problem-solving benchmarks through RL fine-tuning. DeepSeek-R1 (Guo et al., 2025) represents efforts in applying advanced techniques potentially involving RL for complex tasks, although specific methods might vary. Furthermore, preference-based methods like SPPO and GPO have been applied to reasoning-specialized models such as Kimi-1.5 (Team et al., 2025), and the resulting improvements observed on benchmarks involving coding and math suggest that preference-based RLHF can also contribute to refining reasoning abilities, potentially by optimizing implicit properties related to logical consistency and correctness within the preference data. The need for a value function (critic model) used in PPO incurs significant computational costs, and standard PPO can face stability challenges with sparse rewards common in LLM tasks. Addressing these issues has driven recent work. Several methods aim to improve efficiency by removing the value network: RLOO (Kool et al., 2019; Ahmadian et al., 2024) shows that drawing multiple samples per input allows for a baseline based on the average reward. ReMax (Li et al., 2024) adapts REINFORCE (Williams, 1992)

using Monte Carlo returns and normalization; GRPO (Shao et al., 2024) uses a group-average reward baseline and adds a k3-based KL penalty to the objective; and VinePPO (Kazemnejad et al., 2024) uses MC sampling from intermediate steps. Other approaches focus on stability and alternative baselines, such as RLOO (Ahmadian et al., 2024), which uses leave-one-out statistics within a group, and REINFORCE++ (Hu, 2025), which enhances REINFORCE with token-level KL penalties (using the k2 estimator) and normalization. Dr. GRPO (Liu et al., 2025) identifies and corrects a bias found in GRPO’s advantage estimators, DAPO (Yu et al., 2025) introduces strategies like Clip-Higher, reward over-sampling, and a token-level loss to handle long sequences and entropy collapse, while VAPO (Yuan et al., 2025) builds upon it with length-adaptive advantage estimation. Group Policy Gradient (GPG) (Chu et al., 2025) revisits the original REINFORCE objective, using group-normalized rewards and a debiased gradient estimator. Recently, GSPO (Zheng et al., 2025) was proposed with sequence-level rewards and used in the Qwen3 model series (Team, 2025).

Our contribution is to make the off-policy weighting and estimator equivalences explicit across normalized/unnormalized variants, to identify a bias introduced when these weights are omitted (as in the GRPO KL term), and to provide corrected surrogates that are gradient-equivalent to the intended objectives. The design-space view makes transparent how several recent algorithms arise as special cases.

- B CONNECTION BETWEEN REGULARIZED POLICY GRADIENT AND NATURAL POLICY GRADIENT

In this section, we draw the connection between RPG and Natural Policy Gradient (NPG) (Kakade, 2001; Schulman et al., 2015). Note that NPG moves along the steepest-ascent direction defined by the Riemannian geometry induced by the Fisher information matrix, rather than by the Euclidean geometry of the raw parameters. More specifically, we demonstrate that the Natural Policy Gradient (NPG) update can be recovered as a special instance of the RPG update by applying a linear approximation to the expected return and a quadratic approximation to the KL regularization term.

In detail, consider the RPG objective at iteration k:

k∥πθ), (B.1) where J(θ) is the expected return. To study the local behavior of the update, we first apply the first-order Taylor expansion to the return J(θ) around the current policy parameter θk:

JRPG(θ) = J(θ) − β KL(πθ

J(θ) ≈ J(θk) + ∇θJ(θk)⊤∆θ, (B.2) where ∆θ = θ − θk denotes a small change. Then we apply the second-order Taylor expansion of the KL divergence term at θk as follows:

k∥πθ) ⊤θ=θ

k∥πθ) ≈ KL(πθ

k∥πθ

) + ∇θ KL(πθ

KL(πθ

∆θ

k

k

- 1

- 2

∆θ⊤∇2θ KL(πθ

∆θ (B.3)

k∥πθ) θ=θ

+

k

- 1

- 2

∆θ⊤F(θk)∆θ, (B.4) where ∆θ = θ − θk, and F(θk) is the Fisher information matrix:

= 0 + 0 +

##### ∇θ log πθ(x) ⊤θ=θ

##### F(θk) = Ex∼π

##### ∇θ log πθ(x) θ=θ

##### .

θk

k

k

Note that the Fisher information matrix describes the local geometry of the parameter space of the policy family. It gives a good metric inside a small neigbourhood around θk. Now insert (B.2) and (B.4) back into the RPG objective (B.1), we obtain the following quadratic surrogate JRPG(∆θ) around θk:

β 2

JRPG(∆θ) = J(θk) + ∇θJ(θk)⊤∆θ −

∆θ⊤F(θk)∆θ. (B.5)

We now look for the best local step ∆θ∗. Take the gradient of (B.5) with respect to ∆θ and set it equal to zero, we obtain:

∇∆θ JRPG = ∇θJ(θk) − βF(θk)∆θ = 0. Solving this linear system for ∆θ gives

∆θ∗ =

1 β

F(θk)−1∇θJ(θk).

The step ∆θ∗ matches the natural policy gradient (Kakade, 2001) update direction up to the factor 1/β. This suggests that the policy gradient update of RPG in (B.1) can be approximated by NPG as follows

1 β

F(θk)−1∇θJ(θk).

θk+1 ← θk +

In other words, the maximizer of the local KL regularized RPG approximation follows the same direction as a natural policy gradient update.

- C REINFORCE AND PROXIMAL POLICY OPTIMIZATION (PPO)

- C.1 REINFORCE

REINFORCE performs Monte Carlo (MC) updates after sampling a complete trajectory, using the sampled return Gt as an unbiased estimate of the state-action value function Qπ

θ

(st,at). However, these MC estimates often exhibit high variance, leading to slow and unstable learning. To reduce variance, a state-dependent baseline b(st) (commonly an estimate of the state value function, V π

θ

(st)) is subtracted from the return-to-go:

∇θJ(θ) = Eτ∼π

θ

T

t=0

(Gt − b(st))∇θ log πθ(at|st) = Eτ∼π

θ

T

t=0

At∇θ log πθ(at|st) .

(C.1) Here, At = Gt−b(st) is an estimate of the advantage function Aπ

θ

(st,at) = Qπ

θ

(st,at)−V π

θ

(st). Subtracting a baseline that only depends on the state st does not bias the gradient estimate, since Ea

t∼πθ(·|st)[b(st)∇θ log πθ(at|st)] = b(st)∇θ at πθ(at|st) = b(st)∇θ1 = 0. REINFORCE with baseline is typically implemented by minimizing the loss:

LREINFORCE(θ) = −Eτ∼π

θ

T

t=0

SG( At)log πθ(at|st) , (C.2)

using the stop-gradient operator SG(·) to prevent gradients from flowing into the advantage estimate At. As REINFORCE uses samples collected under the current policy πθ for gradient estimation, it is an on-policy algorithm.

- C.2 PROXIMAL POLICY OPTIMIZATION (PPO)

On-policy methods like REINFORCE can be sample-inefficient, requiring new trajectories for each gradient update. Proximal Policy Optimization (PPO) (Schulman et al., 2017) improves stability and sample efficiency by enabling multiple updates using the same batch of data collected under a slightly older policy πθ

. This makes PPO effectively off-policy. PPO achieves this by optimizing a surrogate objective function that discourages large deviations between the current policy πθ and the old policy πθ

old

. The most widely used variant, PPO-Clip, employs a clipped objective:

old

##### JPPO-Clip(θ) = Et min wt(θ) At, clip(wt(θ),1 − ϵ,1 + ϵ) At , (C.3)

where the expectation Et is taken over timesteps in the collected batch sampled from πold. Here, wt(θ) = π

θ(at|st)

πold(at|st) is the importance sampling ratio. At is an advantage estimate, typically computed

using Generalized Advantage Estimation (GAE) (Schulman et al., 2016), which leverages observed rewards and a learned state-value function V (s) to reduce variance.

Notably, in many practical implementations, especially in Reinforcement Learning from Human Feedback (RLHF) for large language models (Ouyang et al., 2022), a KL divergence penalty against a reference policy πref (e.g., the initial supervised model) is often incorporated implicitly by modifying the reward signal before calculating the advantage. For example, the reward used for GAE calculation might become rt′ = rt − β log(πθ(at|st)/πref(at|st)). When this rt′ is used within GAE to compute At, the KL penalty term is effectively folded into the advantage estimate that multiplies the importance weight wt(θ) in the objective function. This approach contrasts with adding the KL penalty as a separate term to the final objective, as seen in GRPO (Section 2.2) or the formal derivations in Section 3.

The hyperparameter ϵ (e.g., 0.2) defines the clipping range [1 − ϵ,1 + ϵ] for the importance ratio wt(θ). This clipping limits the influence of potentially noisy importance weights when the policy changes significantly, preventing destructive updates and further stabilizing the off-policy training. PPO optimizes the policy πθ by maximizing JPPO-Clip(θ).

- D EQUIVALENCE OF k3 ESTIMATOR AND UNNORMALIZED KL DIVERGENCE

As mentioned in Section 3.2, the k3 estimator for KL divergence (Schulman, 2020) is equivalent to the unnormalized KL (UKL) divergence. The k3 function is defined as k3(y) = y − 1 − log y.

Forward KL-k3 and UKL(πold∥πθ): The forward KL-k3 divergence is KLk

3

(πold∥πθ) := Ex∼π

old

[k3(πθ(x)/πold(x))]. Ex∼π

old

k3

πθ(x) πold(x)

= Ex∼π

old

πθ(x) πold(x) − 1 − log

πθ(x) πold(x)

=

x

πold(x)

πθ(x) πold(x) − 1 dx −

x

πold(x)log

πθ(x) πold(x)

dx

=

x

(πθ(x) − πold(x))dx +

x

πold(x)log

πold(x) πθ(x)

dx

= UKL(πold∥πθ).

Reverse KL-k3 and UKL(πθ∥πold): The reverse KL-k3 divergence is KLk

3

(πθ∥πold) := Ex∼π

θ

[k3(πold(x)/πθ(x))]. Ex∼π

θ

k3

πold(x) πθ(x)

= Ex∼π

θ

πold(x) πθ(x) − 1 − log

πold(x) πθ(x)

=

x

πθ(x)

πold(x) πθ(x) − 1 dx −

x

πθ(x)log

πold(x) πθ(x)

dx

=

x

(πold(x) − πθ(x))dx +

x

πθ(x)log

πθ(x) πold(x)

dx

= UKL(πθ∥πold).

- E NORMALIZED KL REGULARIZATION

For completeness, we collect here the normalized KL formulations that were previously in the main text. Their proofs remain in Appendix K.

- E.1 FORWARD KL REGULARIZATION

Consider the objective function with forward KL regularization:

##### [R(x)] − β KL(πold ∥ πθ). (E.1)

##### JFKL(θ) = Ex∼π

θ

- Proposition E.1 (Policy Gradient and Differentiable Loss for Forward KL). The gradient of JFKL(θ) with respect to θ is:

##### ∇θJFKL(θ) = Ex∼π

##### w(x)R(x) + β ∇θ log πθ(x) ,

old

- Table 4: Summary of fully differentiable surrogate losses for normalized KL-regularized objectives (counterparts to Table 1). Here x ∼ πold, w(x) = πθ(x)/πold(x).

#### Regularization (Normalized KL) Surrogate loss (sampling x ∼ πold)

##### Forward KL E[−w(x)R(x) − β log πθ(x)] Reverse KL E[w(x)(−R(x) + β log w(x))]

where w(x) = πθ(x)/πold(x). A corresponding surrogate loss is: LFKL(θ) = Ex∼π

old − w(x)R(x) − β log πθ(x) , which satisfies ∇θLFKL(θ) = −∇θJFKL(θ).

- Remark E.2 (Connection to Maximum Likelihood Estimation). If R(x) = 0, maximizing JFKL(θ) reduces to minimizing β KL(πold ∥ πθ), i.e., MLE on samples from πold.

- E.2 REVERSE KL REGULARIZATION Consider the reverse KL objective:

[R(x)] − β KL(πθ ∥ πold). (E.2)

JRKL(θ) = Ex∼π

θ

- Proposition E.3 (Policy Gradient and Differentiable Loss for Reverse KL). The gradient of JRKL(θ) is:

∇θJRKL(θ) = Ex∼π

w(x) R(x) − β(log w(x) + 1) ∇θ log πθ(x) . A corresponding surrogate loss is:

old

LRKL(θ) = Ex∼π

w(x) −R(x) + β log w(x) ,

old

with ∇θLRKL(θ) = −∇θJRKL(θ). REINFORCE-style RPG with normalized KL regularizations. REINFORCE-style losses for FKL/RKL appear in Appendix F (Table analogues to Table 2).

- F REINFORCE-STYLE REGULARIZED POLICY GRADIENTS WITH VARIOUS KL REGULARIZATION FORMS

- F.1 RATIONALE FOR REINFORCE-STYLE LOSS FORMULATION

As noted in Section 4 of the main text, the derived off-policy policy gradients (Theorems E.1 through 3.6) share a structural similarity with the REINFORCE estimator:

[Weight(x,θ)∇θ log πθ(x)].

∇θJ(θ) = Ex∼π

sampling

This structure suggests an alternative way to implement the gradient update, analogous to the REINFORCE-style approach used in the on-policy setting. Specifically, one could define a surrogate loss of the form:

[SG(Weight(x,θ))log πθ(x)]. (F.1) The rationale is that applying automatic differentiation to this loss should yield:

LREINFORCE-style(θ) = −Ex∼π

sampling

∇θLREINFORCE-style(θ) Autodiff= −Ex∼π

[SG(Weight(x,θ))∇θ log πθ(x)].

sampling

When this gradient is used for optimization, the stop-gradient SG is conceptually removed, resulting in an update aligned with −∇θJ(θ). This relies on SG preventing gradients from flowing through the θdependence within Weight(x,θ) (specifically, the dependence via the importance weight w(x)). The following subsections detail these REINFORCE-style loss formulations for each KL regularization type.

- F.2 REINFORCE-STYLE RPG WITH FORWARD KL REGULARIZATION

We can convert Forward KL regularization of RPG to REINFORCE-style using the stop-gradient operator:

Proposition F.1 (REINFORCE-Style Loss for Forward KL). For the forward KL regularized objective function in Eq. (E.1), the corresponding REINFORCE-style surrogate loss function for gradient descent optimization via automatic differentiation is:

LREINFORCE-styleFKL (θ) = −Ex∼π

old

[SG(w(x)R(x) + β)log πθ(x)],

where w(x) = πθ(x)/πold(x). This loss aims to produce the gradient −∇θJFKL(θ) via automatic differentiation.

Remark F.2. This REINFORCE-style loss requires SG to prevent backpropagation through w(x) in the weight term. Baselines can be added to R(x) inside SG for variance reduction (see Appendix G). In practice we further apply RPG-Style Clip (Section 3.3) by replacing w with w¯ and, when present, log w with log w¯ inside SG(·).

- F.3 REINFORCE-STYLE RPG WITH UNNORMALIZED FORWARD KL REGULARIZATION

Similarly, we can also transform the Unnormalized Forward KL Regularization of RPG into REINFORCE-style as follows:

Proposition F.3 (REINFORCE-Style Loss for Unnormalized Forward KL). For the objective JUFKL(θ) = Eπ

θ

[R(x)] − β UKL(πold∥πθ), whose gradient (sampling from πold) is ∇θJUFKL(θ) = Ex∼ π

old

[Zold(w(x)R(x) − β(w(x) − 1))∇θ log πθ(x)] (Proposition 3.2), a corresponding REINFORCE-style surrogate loss is:

LREINFORCE-styleUFKL (θ) = −Ex∼ π

old

[SG(Zold (w(x)R(x) − β(w(x) − 1)))log πθ(x)],

where πold = πold/Zold and w(x) = πθ(x)/πold(x) (using unnormalized πold). This loss aims to produce the gradient −∇θJUFKL(θ) via automatic differentiation.

- F.4 REINFORCE-STYLE RPG WITH REVERSE KL REGULARIZATION

- Proposition F.4 (REINFORCE-Style Loss for Reverse KL). For the objective JRKL(θ) =

Eπ

θ

[R(x)]−β KL(πθ ∥ πold), whose gradient is ∇θJRKL(θ) = Ex∼π

old

[w(x)(R(x)−β(log w(x)+ 1))∇θ log πθ(x)] (Proposition E.3), a corresponding REINFORCE-style surrogate loss is:

LREINFORCE-styleRKL (θ) = −Ex∼π

old

[SG(w(x)(R(x) − β log w(x) − β))log πθ(x)], (F.2)

where w(x) = πθ(x)/πold(x). This loss aims to produce the gradient −∇θJRKL(θ) via automatic differentiation.

F.5 REINFORCE-STYLE RPG WITH UNNORMALIZED REVERSE KL REGULARIZATION

- Proposition F.5 (REINFORCE-Style Loss for Unnormalized Reverse KL). For the objective

##### [R(x)] − β UKL(πθ∥πold), whose gradient (sampling from πold) is ∇θJURKL(θ) = Ex∼ π

##### JURKL(θ) = Eπ

θ

[Zoldw(x)(R(x) − β log w(x))∇θ log πθ(x)] (Proposition 3.6), a corresponding REINFORCE-style surrogate loss is:

old

LREINFORCE-styleURKL (θ) = −Ex∼ π

##### [SG(Zoldw(x)(R(x) − β log w(x)))log πθ(x)],

old

where πold = πold/Zold and w(x) = πθ(x)/πold(x) (using unnormalized πold). This loss aims to produce the gradient −∇θJURKL(θ) via automatic differentiation.

- G MORE ON ALGORITHMIC DETAILS

- G.1 STABILIZATION TECHNIQUES FOR REGULARIZED POLICY GRADIENTS

Practical implementations of off-policy policy gradient methods often require stabilization techniques to manage variance or prevent destructively large policy updates. Common techniques include:

- • Dual-Clip Objective: This method adapts the clipping mechanism from PPO, with a modification for negative advantages proposed by Ye et al. (2020), to stabilize updates (Schulman et al., 2017).

The Dual Clip objective aims to maximize JDualClip = Ex∼π

old

[LDualClip(x,θ)], where A(x) is an estimate of the advantage analogue (e.g., R(x) − b or the full term derived from the regularized gradient), w(x) = πθ(x)/πold(x) is the importance ratio, and LDualClip(x,θ) is defined as:

- – If A(x) ≥ 0: LDualClip(x,θ) = min(w(x) A(x), clip(w(x),1 − ϵ1,1 + ϵ2) A(x)).
- – If A(x) < 0: LDualClip(x,θ) = max(min(w(x) A(x), clip(w(x),1 − ϵ1,1 + ϵ2) A(x)),c A(x)).

where ϵ1,ϵ2 > 0 are clipping parameters and c > 1 provides a lower bound for negative advantages. To use this with gradient descent (which minimizes a loss L), we minimize the negative of the Dual Clip objective term. Using −min(a,b) = max(−a,−b) and −max(a,b) = min(−a,−b), the corresponding loss term for a single sample x is:

- – If A(x) ≥ 0: LDualClip(x,θ) = max −w(x) A(x), −clip(w(x),1 − ϵ1,1 + ϵ2) A(x) .
- – If A(x) < 0: Let Lclip = max −w(x) A(x), −clip(w(x),1 − ϵ1,1 + ϵ2) A(x) . Then, LDualClip(x,θ) = min Lclip, −c A(x) .

Here, A(x) should represent the advantage or an analogous term derived from the gradient of the original (non-negated) regularized objective (e.g., Proposition E.3). The overall loss is L(θ) = Ex∼π

old

[LDualClip(x,θ)]. This loss function is differentiable with respect to θ (which appears in w(x) and potentially A(x) if it includes terms like log w(x)).

This loss formulation ensures that updates are conservative. For positive advantages, it acts like standard PPO-Clip. For negative advantages, it prevents the objective from becoming arbitrarily large (loss becoming arbitrarily small) by introducing the lower bound c A(x) on the objective (upper bound −c A(x) on the loss).

- • Baseline Subtraction: Used to define the advantage A(x) = R(x) − b(x), reducing the variance of the gradient estimates. The baseline b(x) should ideally not depend strongly on θ. A common

choice is a value function estimate V (x) or simply the batch average reward b = N1 R(xi). The definition of A(x) might also incorporate regularization terms depending on the base objective chosen (see RKL example below).

For instance, applying Dual Clip to stabilize the reverse KL objective (Proposition E.3). The gradient involves the term w(x) (R(x) − b) − β(log w(x) + 1)

∇log πθ. Using this ARKL in the Dual Clip

Analogue to ARKL(x,w;b)

loss structure LDualClipRKL (θ) = Ex∼π

[LDualClipRKL (x,θ)] where:

old

• If ARKL(x,w;b) ≥ 0:

LDualClipRKL (x,θ) = max −w(x) ARKL,−clip(w(x),1 − ϵ1,1 + ϵ2) ARKL .

• If ARKL(x,w;b) < 0: Let Lclip = max −w(x) ARKL,−clip(w(x),1 − ϵ1,1 + ϵ2) ARKL .

LDualClipRKL (x,θ) = min Lclip, −c ARKL ,

where ARKL(x,w;b) = (R(x) − b) − β(log w(x) + 1). Simpler approximations might use A(x) = R(x) − b.

Using PPO-style clipping alters the optimization objective compared to the original KL-regularized objectives, trading strict adherence for enhanced stability. The choice of base objective structure, definition of A, and stabilization techniques depends on the specific application.

- Algorithm 1 RPG with Dual-Clip Stabilization

Require: Reference policy πold, Reward function R(x), Initial policy parameters θ0 Require: Base objective structure Jchosen (implies regularization type), Regularization strength β ≥ 0 Require: Learning rate α > 0, Batch size N > 0, Number of epochs K ≥ 1 per iteration Require: Dual Clip parameters: ϵ1 > 0, ϵ2 > 0, c > 1 Require: Baseline method (e.g., batch/group average, value function Vϕ)

- 1: Initialize policy parameters θ ← θ0
- 2: Initialize value function parameters ϕ (if baseline uses Vϕ)
- 3: for each training iteration do
- 4: Sample batch D = {xi}Ni=1 ∼ πold ▷ Collect data using old policy
- 5: Compute Ri for i = 1..N
- 6: Compute baselines bi for i = 1..N (e.g., bi = N1 j Rj or bi = Vϕ(xi))

- 7: for k = 1 to K do ▷ Multiple optimization epochs on the same batch
- 8: Initialize batch loss Lbatch = 0
- 9: for i = 1 to N do
- 10: wi = ππθ(xi)

old(xi), log wi = log πθ(xi) − log πold(xi) ▷ Compute importance weight

- 11: Define Advantage analogue Ai based on Jchosen, Ri, bi, wi, β.
- 12: ▷ Ex: For RKL, Ai = (Ri − bi) − β(log wi + 1). Note: Ai depends on current θ via wi
- 13: if Dual Clip enabled then
- 14: loss_term1i = −wi × Ai ▷ Negative of unclipped term, gradient flows through wi
- 15: wi,clipped = clip(wi, 1 − ϵ1, 1 + ϵ2)
- 16: loss_term2i = −wi,clipped × Ai ▷ Negative of clipped term
- 17: Lclip(i) = max(loss_term1i, loss_term2i)
- 18: if Ai ≥ 0 then
- 19: Lterm(i) = Lclip(i)
- 20: else ▷ Ai < 0
- 21: loss_lower_boundi = −c × Ai ▷ Lower bound term
- 22: Lterm(i) = min(Lclip(i), loss_lower_boundi)
- 23: end if
- 24: else
- 25: ▷ Define base loss term (unclipped) based on chosen objective’s negative gradient structure
- 26: ▷ Ex: For RKL loss (no clip): Lterm(i) = wi(−(Ri − bi) + β log wi)
- 27: Lterm(i) = −wi × Ai
- 28: end if
- 29: Lbatch = Lbatch + Lterm(i)
- 30: end for
- 31: L(θ) = N1 Lbatch ▷ Compute final batch loss for minimization

- 32: g ← ∇θ L(θ) ▷ Compute gradient (flows through wi and Ai)
- 33: θ ← OptimizerUpdate(θ, g, α) ▷ Update policy parameters
- 34: if using a learned baseline Vϕ then
- 35: Update value function parameters ϕ (e.g., by minimizing E[(Vϕ(xi) − Ri)2] over the batch)
- 36: end if
- 37: end for
- 38: end for
- 39: return Optimized policy parameters θ

- G.2 STABILIZATION TECHNIQUES FOR REINFORCE-STYLE REGULARIZED POLICY GRADIENTS

While the REINFORCE-style losses derived in this section (Table 2) provide theoretically grounded gradient estimators for the regularized objectives, practical implementations often benefit significantly from stabilization techniques common in policy gradient methods. These techniques aim to reduce variance and control the magnitude of policy updates, which is especially crucial in the off-policy setting where importance weights w(x) and can exacerbate instability.

- • Baseline Subtraction and Regularized Advantage Definition: This is a standard variance reduction technique. Critically, when combining with stabilization like PPO clipping in this REINFORCE-style context, the term playing the role of the advantage ( At) that gets clipped should

ideally incorporate not just the baselined reward but also the regularization terms derived from the objective’s gradient. Recall the REINFORCE-style gradient structure ∇θJ(θ) = Ex∼π

[Weight(x,θ)∇θ log πθ(x)]. The PPO objective involves terms like wt At. To align these, we define the regularized advantage At such that wt At approximates the key part of Weight(x,θ). For example:

sampling

- – For RKL (Proposition F.4), WeightRKL = w(x)(R(x) − β(log w(x) + 1)). We define the regularized advantage as ARKLt = (R(x) − b(x)) − β(log w(x) + 1).
- – For URKL (Proposition F.5), WeightURKL = Zoldw(x)(R(x) − β log w(x)). Ignoring Zold, we define AURKLt = (R(x) − b(x)) − β log w(x).
- – For FKL or UFKL, the structure might not cleanly separate into w(x) × (...). In such cases,

a common simplification is to use At = R(x) − b(x) and accept that the clipping primarily stabilizes the reward term’s contribution.

This calculated At (incorporating reward, baseline, and KL terms) is then treated as constant using the stop-gradient operator, SG( At), when plugged into the clipping loss function.

- • RPG-Style Objective Clipping (Dual-Clip Variant): PPO (Schulman et al., 2017) introduces objective clipping to limit the impact of large importance ratios w(x). The Dual-Clip variant (Ye et al., 2020) refines this, particularly for negative advantages, using a lower bound parameter c > 1. When applied in the REINFORCE-style setting, the PPO Dual-Clip objective aims to maximize (simplified notation, expectation over t ∼ πold):

JDualClip(θ) = Et[LDualClipt (θ)] where At is the regularized advantage defined above (incorporating Rt,bt, and KL terms), wt(θ) =

πold(at|st), and LDualClipt (θ) is defined based on the sign of SG( At):

πθ(at|st)

- – If SG( At) ≥ 0: LDualClipt (θ) = min(wt(θ)SG( At),clip(wt(θ),1 − ϵ1,1 + ϵ2)SG( At))
- – If SG( At) < 0: LDualClipt (θ) = max(min(wt(θ)SG( At),clip(wt(θ),1 − ϵ1,1 +

ϵ2)SG( At)),cSG( At)) Here, ϵ1,ϵ2 are clipping hyperparameters, and c is the lower bound factor. Note that θ influences this objective only through wt(θ), as At is detached via SG. To implement this using gradient descent (minimizing a loss), we minimize the negative of the PPO Dual-Clip objective. The loss function becomes LDualClip(θ) = Et[LDualClipt (θ)], where LDualClipt (θ) = −LDualClipt (θ). Explicitly:

- – If SG( At) ≥ 0: LDualClipt (θ) = max(−wt(θ)SG( At),−clip(wt(θ),1−ϵ1,1+ϵ2)SG( At)).
- – If SG( At) < 0: Let Lclip = max(−wt(θ)SG( At),−clip(wt(θ),1 − ϵ1,1 + ϵ2)SG( At)). Then, LDualClipt (θ) = min(Lclip,−cSG( At)).

This PPO Dual-Clip loss function LDualClip(θ) replaces the simpler REINFORCE-style losses derived earlier (like LREINFORCE-styleRKL in Eq. (F.2)). The gradient ∇θLDualClip(θ) is computed via automatic differentiation, where the gradient flows through wt(θ) but is stopped at At. This approach uses the PPO objective structure with the appropriately defined regularized advantage for stabilization in an off-policy REINFORCE-style update. Algorithm 2 details this implementation.

- H DETAILED EXPERIMENTAL SETUP

Hyperparameters. Unless otherwise specified, all experiments use AdamW optimizer (Loshchilov and Hutter, 2019) with a learning rate of 1 × 10−6, a weight decay of 0.1, and gradient clipping at 1.0. Training proceeds for 400 steps, including an initial 10 warm-up steps, after which a constant learning rate is maintained. The global training batch size is 512. For each sample in the batch, we roll out 16 responses using a temperature of 1.0. The per-GPU mini-batch size is 32, and experiments are conducted on 8 NVIDIA H100 GPUs. The maximum training and rollout length is set to 4,096 tokens for a 2K context length and 8,192 tokens for a 4K context length, with dynamic batching enabled. The KL regularization coefficient β is set to 1 × 10−4.

Specific Clipping Parameters and Adopted Techniques. As mentioned in Section 5, we set (ϵ1,ϵ2) = (0.2,0.28) for RPG, RPG-REINFORCE and DAPO. For GRPO, we use (ϵ1,ϵ2) = (0.1,0.1).

- Algorithm 2 REINFORCE-Style RPG with Dual-Clip Stabilization

Require: Reference policy πold, Reward function R(x), Initial policy parameters θ0 Require: KL Component function Compute_KL_Component(x, θ, πold), KL Component Coefficient β Require: Learning rate α > 0, Batch size N > 0, Number of epochs K ≥ 1 per iteration Require: Dual Clip parameters: ϵ1 > 0 (low), ϵ2 > 0 (high), c > 1 Require: Baseline method (e.g., batch average, value function Vϕ)

- 1: Initialize policy parameters θ ← θ0
- 2: Initialize value function parameters ϕ (if baseline uses Vϕ)
- 3: for each training iteration do
- 4: Sample batch D = {xi}Ni=1 ∼ πold
- 5: Compute rewards Ri for i = 1..N
- 6: Compute baselines bi for i = 1..N (e.g., bi = N1 j Rj or bi = Vϕ(xi))

- 7: for k = 1 to K do ▷ Multiple optimization epochs on the same batch
- 8: Initialize batch loss Lbatch = 0
- 9: for i = 1 to N do
- 10: wi = ππθ(xi)

old(xi) ▷ Importance weight

- 11: ℓi = − log πθ(xi) ▷ Negative log probability
- 12: AR,i = Ri − bi ▷ Baseline-subtracted reward
- 13: CKL,i = β · Compute_KL_Component(xi, θ, πold(xi)) ▷ KL component
- 14: A′i = AR,i + SG(CKL,i)/ SG(wi) ▷ Effective advantage
- 15: ψi = A′i × ℓi ▷ Branching term
- 16: if ψi ≥ 0 then
- 17: whigh = 1 + ϵ2
- 18: if wi < whigh then
- 19: Li = ψi × SG(wi) ▷ Grad exists
- 20: else ▷ wi ≥ whigh
- 21: A′high = AR,i + SG(CKL,i)/ SG(whigh)
- 22: ψhigh = A′high × SG(ℓi)
- 23: Li = ψhigh × SG(whigh)
- 24: end if
- 25: else ▷ ψi ≤ 0
- 26: wlow = 1 − ϵ1
- 27: if wi ≤ wlow then
- 28: A′low = AR,i + SG(CKL,i)/ SG(wlow)
- 29: ψlow = A′low × SG(ℓi)
- 30: Li = ψlow × SG(wlow)
- 31: else if wi < c then
- 32: Li = ψi × SG(wi) ▷ Grad exists
- 33: else ▷ wi ≥ c
- 34: Li = AR,i × SG(ℓi) × c + SG(CKL,i) × SG(ℓi)
- 35: end if
- 36: end if
- 37: Lbatch = Lbatch + Li
- 38: end for
- 39: L(θ) = N1 Lbatch ▷ Compute average batch loss

- 40: g ← ∇θL(θ) ▷ Compute gradient
- 41: θ ← OptimizerUpdate(θ, g, α) ▷ Update policy parameters
- 42: if using a learned baseline Vϕ then
- 43: Update value function parameters ϕ
- 44: end if
- 45: end for
- 46: end for
- 47: return Optimized policy parameters θ

(a) AIME24 (b) AIME25 (c) AMC23

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

(d) Reward (Critic Score)

(e) Entropy (f) Response Length

- Figure 5: Performance of RPG and REINFORCE-Style Regularized Policy Gradient (RPGREINFORCE) methods compared to baselines with 4k context length.

- I ADDITIONAL EXPERIMENT RESULTS

- I.1 THE PERFORMANCE WITH 2K AND 4K CONTEXT LENGTH

The quantitative results in Table 5 demonstrate the competitive performance of the proposed RPG and REINFORCE-style RPG frameworks with 4k context length. On AIME24, RPG-REINFORCE variants lead, with RPG-REINFORCE-URKL achieving the best “Best” score (0.4531) and the best “Last” score (0.4458), while RPG-REINFORCE-UFKL attain a second best “Last” score (0.4281). For AIME25, RPG-REINFORCE-URKL still achieves the top “Best” score (0.4313) and a strong “Last” score (0.4125) and RPG-REINFORCE-UFKL is second only to that. Overall, RPG and RPG-REINFORCE methods rank at or near the top across benchmarks and metrics, while exhibiting stable training dynamics.

Similarly, Table 6 shows the experiment results with 2k context length. It can be observed that RPG and RPG-REINFORCE variants demonstrate robust performance, often competitive with or exceeding baselines. For example, RPG-REINFORCE-UFKL achieves the top “Best” scores for

- AIME24 (0.3625) and AIME25 (0.3083), and the top “Last” score of AIME25 (0.2927), while RPG-UFKL attains the top “Last” score of AIME24 (0.3427) and the second highest “Last” score of
- AIME25 (0.2833). Their training curves in Figure 6 generally indicate good stability and effective learning. The consistently high performance across various RPG formulations underscores the utility of the systematically derived KL-regularized objectives explored in this work.

- I.2 ABLATION STUDY

To further investigate our algorithms, we implement an ablation study on the clip ratio and the effect of the KL regularization coefficient.

- I.2.1 ABLATION ON CLIP RATIO

We first implement experiments with different clip ratios on REINFORCE-style RPG algorithms. We choose (0.1,0.1) and (0.2,0.28) for (ϵ1,ϵ2) since they are 2 typical choices of clip ratios (Schulman et al., 2017; Yu et al., 2025), and the performance curves as well as key training dynamics are displayed in Figure 8. It can be observed that although the critic score and response length are similar for different settings, the actor entropy shows a huge difference in trend, demonstrating that an

- Table 5: Combined performance metrics with 4k context length on the AIME24, AIME25 and AMC23 mathematical reasoning benchmarks, showing “Last” and “Best” scores. The “Last” score is from the 400th training step, assuming the training process remained stable to that point. The highest score in each column is bolded, and the second highest is underlined. RPG and RPG-REINFORCE methods are highlighted with light cyan and light green backgrounds, respectively.

AIME24 AIME25 AMC23 Last Best Last Best Last Best

Method

REINFORCE++-Baseline 0.3854 0.4302 0.3406 0.3844 0.9234 0.9328 REINFORCE++ 0.3490 0.3885 0.2822 0.3479 0.8977 0.9297 GRPO 0.3458 0.3677 0.2896 0.3042 0.9016 0.9109 DAPO 0.4063 0.4479 0.3510 0.3938 0.9297 0.9297

RPG-UFKL 0.4031 0.4396 0.3625 0.3979 0.9477 0.9500 RPG-URKL 0.3990 0.4219 0.3438 0.3792 0.9500 0.9531

RPG-REINFORCE-UFKL 0.4281 0.4375 0.3771 0.4042 0.9023 0.9133 RPG-REINFORCE-URKL 0.4458 0.4531 0.4125 0.4313 0.9313 0.9352

(a) AIME24 (b) AIME25 (c) AMC23

(d) Reward (Critic Score) (e) Entropy (f) Response Length

- Figure 6: Training dynamics and benchmark performance for fully differentiable Regularized Policy Gradient (RPG) and REINFORCE-Style RPG (RPG-REINFORCE) compared to baselines (GRPO and DAPO) with 2k context length.

adequately higher and clip-higher strategy proposed by DAPO may greatly contribute to the increase of performance by increasing the actor entropy.

- I.2.2 ABLATION ON KL REGULARIZATION COEFFICIENT

We also implement ablation studies on the effect of the KL regularization coefficient. We implement experiments with REINFORCE-style RPG-UFKL (RPG-REINFORCE-UFKL) with β = 1 × 10−3 and 1 × 10−4, and the results are shown in Figure 9. Figures 9(a) and 9(b) show that the coefficient 1×10−4 performs better than 1×10−3, and the trend in response length conforms to the performance, indicating that longer response length may help with the performance improvement.

We also dig into the effect of the iteratively updated reference model. We implement another experiment with no iteratively updated reference model, and display the performance and dynamics

- Table 6: Combined performance metrics with 2K context length on the AIME24, and AIME25 mathematical reasoning benchmarks, showing “Last” and “Best” scores. The “Last” score is from the 400th training step, assuming the training process remained stable to that point. The highest score in each column is bolded, and the second highest is underlined. RPG and RPG-REINFORCE methods are highlighted with light cyan and light green backgrounds, respectively.

Method

AIME24 AIME25 Last Best Last Best

GRPO 0.2563 0.2708 0.2323 0.2479 DAPO 0.3229 0.3281 0.2792 0.2844

RPG-UFKL 0.3427 0.3479 0.2833 0.2833 RPG-URKL 0.3260 0.3594 0.2677 0.2677

RPG-REINFORCE-UFKL 0.3396 0.3625 0.2927 0.3083 RPG-REINFORCE-URKL 0.3188 0.3417 0.2792 0.2938

- Table 7: Combined performance metrics with 4k context length on the Minerva-Math and OlympiadBench mathematical reasoning benchmarks, showing “Last” and “Best” scores. The “Last” score is from the 400th training step, assuming the training process remained stable to that point. The highest score in each column is bolded, and the second highest is underlined. RPG and RPG-REINFORCE methods are highlighted with light cyan and light green backgrounds, respectively.

Minerva-Math OlympiadBench Last Best Last Best

Method

REINFORCE++-Baseline 0.1250 0.1471 0.4926 0.5875 REINFORCE++ 0.1691 0.1728 0.4778 0.6202 GRPO 0.1029 0.1177 0.4926 0.5594 DAPO 0.0993 0.1507 0.5163 0.5727

RPG-UFKL 0.1397 0.2059 0.4688 0.5564 RPG-URKL 0.1654 0.1654 0.4837 0.5801

RPG-REINFORCE-UFKL 0.1360 0.1434 0.5074 0.5564 RPG-REINFORCE-URKL 0.1140 0.1434 0.4674 0.5816

(a) AIME24 (b) AIME25 (c) AMC23

- Figure 7: Pass@32 performances for fully differentiable Regularized Policy Gradient (RPG) and REINFORCE-Style RPG (RPG-REINFORCE) compared to baselines with 4k context length.

in Figure 9. It can be observed that the performance recovers with longer response length and much lower actor entropy, showing that longer response length can be an important factor and indicator of the performance on benchmarks.

- I.3 EXPERIMENTS ON QWEN-2.5-7B-INSTRUCT

I.3.1 REGULARIZED POLICY GRADIENT USING QWEN-2.5-7B-INSTRUCT

(a) AIME24 (b) AIME25 (c) AMC23

(d) Reward (Critic Score) (e) Entropy (f) Response Length

- Figure 8: Performance of REINFORCE-Style Regularized Policy Gradient (RPG-REINFORCE) methods with different clip ratios with 2k context length. Plots display accuracy on mathematical reasoning benchmarks (AIME24, AIME25) and key training dynamics (reward, policy entropy, response length).

(a) AIME24 (b) AIME25 (c) AMC23

(d) Reward (Critic Score) (e) Entropy (f) Response Length

- Figure 9: Performance of REINFORCE-Style Regularized Policy Gradient (RPG-REINFORCE) methods with different KL coefficients with 4K context length. Here, “noiterref” indicates the model is trained with no iteratively updated reference model. Plots display accuracy on mathematical reasoning benchmarks (AIME24, AIME25) and key training dynamics (reward, policy entropy, response length).

- Table 8: Combined performance metrics on the AMC23, AIME24, and AIME25 mathematical reasoning benchmarks with Qwen-2.5-7B-Instruct model, showing “Last” and “Best” scores. The “Last” score is from the 400th training step, assuming the training process remained stable to that point. The highest score in each column is bolded, and the second highest is underlined. RPG and RPGREINFORCE methods are highlighted with light cyan and light green backgrounds, respectively.

AMC23 AIME24 AIME25 Last Best Last Best Last Best

Method

GRPO 0.6266 0.7250 0.1094 0.1406 0.0281 0.0948 REINFORCE++ 0.7625 0.7664 0.0521 0.1177 0.0302 0.0740 REINFORCE++-Baseline 0.8711 0.8711 0.0990 0.1510 0.0656 0.0969 DAPO 0.8039 0.8734 0.0760 0.1240 0.0531 0.1063

RPG-FKL 0.8695 0.8836 0.1083 0.1490 0.0427 0.1083 RPG-RKL 0.8648 0.8672 0.1167 0.1469 0.0677 0.1240 RPG-UFKL 0.8703 0.8703 0.0885 0.1427 0.0927 0.1177 RPG-URKL 0.8258 0.8641 0.0875 0.1271 0.0677 0.0917

RPG-REINFORCE-FKL 0.8727 0.8727 0.1208 0.1667 0.0573 0.0875 RPG-REINFORCE-RKL 0.8305 0.8516 0.1125 0.1375 0.0490 0.0875 RPG-REINFORCE-UFKL 0.8391 0.8602 0.1229 0.1458 0.0740 0.0979 RPG-REINFORCE-URKL 0.8531 0.8531 0.1208 0.1500 0.0813 0.0938

(a) AMC23 (b) AIME24 (c) AIME25

(d) Reward (Critic Score) (e) Entropy (f) Response Length

- Figure 10: Performance of fully differentiable Regularized Policy Gradient (RPG) methods compared to baselines when using base model: Qwen-2.5-7B-Instruct. Plots display accuracy on mathematical reasoning benchmarks (AMC23, AIME24, AIME25) and key training dynamics (reward, policy entropy, response length).

- J PROOF OF THEOREM 2.1 (GENERALIZED POLICY GRADIENT THEOREM)

Proof. The proof relies on the log-derivative trick, ∇θπθ(x) = πθ(x)∇θ log πθ(x), and the product rule under the integral sign:

∇θEx∼π

θ

[f(x,θ)] = ∇θ πθ(x)f(x,θ)dx

= ∇θ(πθ(x)f(x,θ))dx (Swap ∇, )

= ((∇θπθ(x))f(x,θ) + πθ(x)(∇θf(x,θ)))dx

= (πθ(x)(∇θ log πθ(x))f(x,θ) + πθ(x)(∇θf(x,θ)))dx (Log-derivative)

= πθ(x)(f(x,θ)∇θ log πθ(x) + ∇θf(x,θ))dx

= Ex∼π

θ

[f(x,θ)∇θ log πθ(x) + ∇θf(x,θ)].

| |
|---|

- K PROOFS FOR REGULARIZED POLICY GRADIENTS

This section provides detailed proofs for the theorems presented in Section 3, demonstrating that the gradients of the proposed fully differentiable off-policy surrogate losses correspond to the negative gradients of the respective original objectives. The core tool used is the policy gradient theorem: ∇θEx∼π

##### [f(x,θ)∇θ log πθ(x) + ∇θf(x,θ)]. We use the notation w(x) = πθ(x)/πold(x) for the importance weight.

##### [f(x,θ)] = Ex∼π

θ

θ

- K.1 PROOF OF PROPOSITION E.1 (POLICY GRADIENT AND DIFFERENTIABLE LOSS FOR NORMALIZED FORWARD KL)

Proof. We start by rewriting the objective function JFKL(θ) using expectations with respect to the fixed reference policy πold. The first term, the expected reward under πθ, can be rewritten using importance sampling:

Ex∼π

[R(x)] = πθ(x)R(x)dx =

θ

πθ(x) πold(x)

The second term is the forward KL divergence:

πold(x)R(x)dx = Ex∼π

old

##### [w(x)R(x)].

πold(x) πθ(x)

##### KL(πold ∥πθ) = Ex∼π

log

old

= Ex∼π

[log πold(x) − log πθ(x)]

old

= Ex∼π

##### [−log πθ(x)] + Ex∼π

[log πold(x)]. Substituting these into the objective function:

old

old

##### JFKL(θ) = Ex∼π

##### [w(x)R(x)] − β (Ex∼π

##### [−log πθ(x)] + Ex∼π

[log πold(x)])

old

old

old

= Ex∼π

##### [w(x)R(x) + β log πθ(x)] − βEx∼π

[log πold(x)]. Since πold(x) does not depend on θ, the term βEx∼π

old

old

[log πold(x)] is a constant with respect to θ. Now we compute the gradient ∇θJFKL(θ). Assuming we can swap gradient and expectation (standard assumption in policy gradient methods):

old

##### ∇θJFKL(θ) = ∇θEx∼π

[w(x)R(x) + β log πθ(x)]

old

= Ex∼π

[∇θ(w(x)R(x) + β log πθ(x))]

old

= Ex∼π

##### [(∇θw(x))R(x) + β∇θ log πθ(x)].

old

We use the identity for the gradient of the importance weight:

πθ(x) πold(x)

∇θw(x) = ∇θ

1 πold(x)∇θπθ(x)

=

∇θπθ(x) πθ(x)

πθ(x) πold(x)

=

= w(x)∇θ log πθ(x). Substituting this back into the gradient expression: ∇θJFKL(θ) = Ex∼π

[w(x)(∇θ log πθ(x))R(x) + β∇θ log πθ(x)]

old

= Ex∼π

##### w(x)R(x) + β ∇θ log πθ(x) .

old

This proves the first part of the theorem. Now, consider the surrogate loss function:

##### LFKL(θ) = Ex∼π

[−w(x)R(x) − β log πθ(x)]. We compute its gradient:

old

∇θLFKL(θ) = ∇θEx∼π

[−w(x)R(x) − β log πθ(x)]

old

= Ex∼π

[∇θ(−w(x)R(x) − β log πθ(x))]

old

= Ex∼π

[−(∇θw(x))R(x) − β∇θ log πθ(x)]

old

= Ex∼π

[−w(x)(∇θ log πθ(x))R(x) − β∇θ log πθ(x)]

old

= −Ex∼π

##### w(x)R(x) + β ∇θ log πθ(x) .

old

Comparing this with the gradient of the objective function, we see that ∇θLFKL(θ) = −∇θJFKL(θ). This confirms that minimizing LFKL(θ) corresponds to maximizing JFKL(θ) using gradient-based methods.

| |
|---|

- K.2 PROOF OF PROPOSITION 3.2 (POLICY GRADIENT AND DIFFERENTIABLE LOSS FOR UNNORMALIZED FORWARD KL)

Proof. We start by expressing the components of JUFKL(θ) using expectations over the normalized reference distribution πold(x) = πold(x)/Zold. The importance weight is w(x) = πθ(x)/πold(x), which implies πθ(x) = w(x)πold(x) = w(x)Zold πold(x).

The expected reward term:

Ex∼π

[R(x)] = πθ(x)R(x)dx = w(x)πold(x)R(x)dx

θ

= w(x)Zold πold(x)R(x)dx = ZoldEx∼ π

old

##### [w(x)R(x)].

The unnormalized KL divergence term UKL(πold∥πθ) has two parts. Part 1 (Generalized KL):

πold(x) πθ(x)

πold(x) πθ(x)

πold(x)log

dx = Zold πold(x)log

dx

1 w(x)

= ZoldEx∼ π

= ZoldEx∼ π

[−log w(x)]. Part 2 (Mass Correction):

log

old

old

(πθ(x) − πold(x))dx = (w(x)πold(x) − πold(x))dx

= (w(x) − 1)πold(x)dx = (w(x) − 1)Zold πold(x)dx

= ZoldEx∼ π

##### [w(x) − 1] = ZoldEx∼ π

##### [w(x)] − Zold.

old

old

Combining these parts for the UKL term: UKL(πold∥πθ) = ZoldEx∼ π

##### [−log w(x)] + ZoldEx∼ π

##### [w(x)] − Zold.

old

old

Now, substitute everything into the objective JUFKL(θ): JUFKL(θ) = ZoldEx∼ π

##### [w(x)R(x)] − β (ZoldEx∼ π

##### [−log w(x)] + ZoldEx∼ π

[w(x)] − Zold)

old

old

old

= ZoldEx∼ π

##### [w(x)R(x) + β log w(x) − βw(x) + β].

old

To compute the gradient ∇θJUFKL(θ), we differentiate the terms inside the expectation. The constant term βZold (arising from β inside the expectation) vanishes upon differentiation.

##### ∇θJUFKL(θ) = ∇θ (ZoldEx∼ π

[w(x)R(x) + β log w(x) − βw(x)])

old

= ZoldEx∼ π

[∇θ(w(x)R(x)) + β∇θ(log w(x)) − β∇θ(w(x))]. We need the gradients of w(x) and log w(x):

old

∇θw(x) = w(x)∇θ log πθ(x) (as derived in Proposition E.1 proof) ∇θ log w(x) = ∇θ(log πθ(x) − log πold(x)) = ∇θ log πθ(x).

Substituting these into the gradient expression:

∇θJUFKL(θ) = ZoldEx∼ π

[(∇θw(x))R(x) + β∇θ log πθ(x) − β(∇θw(x))]

old

= ZoldEx∼ π

[w(x)R(x)∇θ log πθ(x) + β∇θ log πθ(x) − βw(x)∇θ log πθ(x)]

old

= ZoldEx∼ π

[(w(x)R(x) − βw(x) + β)∇θ log πθ(x)]

old

= ZoldEx∼ π

w(x)R(x) − β (w(x) − 1) ∇θ log πθ(x) . This proves the first part of the theorem. Now, consider the surrogate loss function:

old

LUFKL(θ) = ZoldEx∼ π

old −w(x)R(x) + β w(x) − log w(x) − 1 . We compute its gradient:

∇θLUFKL(θ) = ZoldEx∼ π

[∇θ(−w(x)R(x)) + β∇θ(w(x) − log w(x) − 1)]

old

= ZoldEx∼ π

[−(∇θw(x))R(x) + β(∇θw(x) − ∇θ log w(x))]

old

= ZoldEx∼ π

[−w(x)R(x)∇θ log πθ(x) + β(w(x)∇θ log πθ(x) − ∇θ log πθ(x))]

old

= ZoldEx∼ π

old −w(x)R(x) + βw(x) − β ∇θ log πθ(x)

= −ZoldEx∼ π

##### w(x)R(x) − β(w(x) − 1) ∇θ log πθ(x) .

old

Comparing this with the gradient of the objective function, we find ∇θLUFKL(θ) = −∇θJUFKL(θ). This confirms the surrogate loss function. Note that the constant −1 inside the logarithm term in

the loss LUFKL corresponds to the constant βZold in the objective JUFKL and does not affect the gradient.

| |
|---|

- K.3 PROOF OF PROPOSITION E.3 (POLICY GRADIENT AND DIFFERENTIABLE LOSS FOR NORMALIZED REVERSE KL)

Proof. We rewrite the objective function JRKL(θ) using expectations with respect to πold. The expected reward term is Ex∼π

[w(x)R(x)], as shown previously. The reverse KL divergence term is:

##### [R(x)] = Ex∼π

θ

old

πθ(x) πold(x)

##### KL(πθ ∥πold) = Ex∼π

log

θ

= Ex∼π

[log w(x)]

θ

= πθ(x)log w(x)dx

πθ(x) πold(x)

=

πold(x)log w(x)dx

= Ex∼π

##### [w(x)log w(x)].

old

Substituting these into the objective function: JRKL(θ) = Ex∼π

##### [w(x)R(x)] − βEx∼π

##### [w(x)log w(x)] = Ex∼π

[w(x)R(x) − βw(x)log w(x)]. Now we compute the gradient ∇θJRKL(θ):

old

old

old

∇θJRKL(θ) = ∇θEx∼π

[w(x)R(x) − βw(x)log w(x)]

old

= Ex∼π

[∇θ(w(x)R(x)) − β∇θ(w(x)log w(x))]. We need the gradient of w(x)log w(x):

old

∇θ(w(x)log w(x)) = (∇θw(x))log w(x) + w(x)∇θ(log w(x))

= (w(x)∇θ log πθ(x))log w(x) + w(x)(∇θ log πθ(x))

= w(x)∇θ log πθ(x)(log w(x) + 1).

Substituting this and ∇θw(x) = w(x)∇θ log πθ(x) into the gradient expression for JRKL(θ):

##### ∇θJRKL(θ) = Ex∼π

[(∇θw(x))R(x) − βw(x)∇θ log πθ(x)(log w(x) + 1)]

old

= Ex∼π

[w(x)(∇θ log πθ(x))R(x) − βw(x)(log w(x) + 1)∇θ log πθ(x)]

old

= Ex∼π

w(x) R(x) − β(log w(x) + 1) ∇θ log πθ(x) . This proves the first part of the theorem. Now, consider the surrogate loss function:

old

##### LRKL(θ) = Ex∼π

w(x) −R(x) + β log w(x) . We compute its gradient:

old

∇θLRKL(θ) = ∇θEx∼π

[−w(x)R(x) + βw(x)log w(x)]

old

= Ex∼π

[∇θ(−w(x)R(x)) + β∇θ(w(x)log w(x))]

old

= Ex∼π

[−(∇θw(x))R(x) + βw(x)∇θ log πθ(x)(log w(x) + 1)]

old

= Ex∼π

[−w(x)(∇θ log πθ(x))R(x) + βw(x)(log w(x) + 1)∇θ log πθ(x)]

old

= Ex∼π

w(x) −R(x) + β(log w(x) + 1) ∇θ log πθ(x)

old

= −Ex∼π

##### w(x) R(x) − β(log w(x) + 1) ∇θ log πθ(x) .

old

Comparing this with the gradient of the objective function, we confirm that ∇θLRKL(θ) = −∇θJRKL(θ).

| |
|---|

- K.4 PROOF OF PROPOSITION 3.6 (POLICY GRADIENT AND DIFFERENTIABLE LOSS FOR UNNORMALIZED REVERSE KL)

Proof. We again express the objective components using expectations over the normalized reference distribution πold(x) = πold(x)/Zold, with w(x) = πθ(x)/πold(x). The expected reward term: Ex∼π

##### [w(x)R(x)].

##### [R(x)] = ZoldEx∼ π

θ

old

The unnormalized reverse KL divergence UKL(πθ∥πold) has two parts. Part 1 (Generalized KL):

πθ(x) πold(x)

πθ(x)log

dx = πθ(x)log w(x)dx

= w(x)πold(x)log w(x)dx

= w(x)Zold πold(x)log w(x)dx

= ZoldEx∼ π

old

##### [w(x)log w(x)].

Part 2 (Mass Correction):

(πold(x) − πθ(x))dx = πold(x)dx − πθ(x)dx

= Zold − w(x)πold(x)dx

= Zold − w(x)Zold πold(x)dx

= Zold − ZoldEx∼ π

[w(x)]. Combining these for the UKL term:

old

##### UKL(πθ∥πold) = ZoldEx∼ π

##### [w(x)log w(x)] + Zold − ZoldEx∼ π

[w(x)].

old

old

Now, substitute into the objective JURKL(θ): JURKL(θ) = ZoldEx∼ π

##### [w(x)R(x)] − β (ZoldEx∼ π

##### [w(x)log w(x)] + Zold − ZoldEx∼ π

[w(x)])

old

old

old

= ZoldEx∼ π

##### [w(x)R(x) − βw(x)log w(x) − β + βw(x)].

old

We compute the gradient ∇θJURKL(θ). The constant term −βZold vanishes upon differentiation.

##### ∇θJURKL(θ) = ∇θ (ZoldEx∼ π

[w(x)R(x) − βw(x)log w(x) + βw(x)])

old

= ZoldEx∼ π

[∇θ(w(x)R(x)) − β∇θ(w(x)log w(x)) + β∇θw(x)].

old

Using the previously derived gradients ∇θw(x) = w(x)∇θ log πθ(x) and ∇θ(w(x)log w(x)) = w(x)∇θ log πθ(x)(log w(x) + 1):

∇θJURKL(θ) = ∇θ (ZoldEx∼ π

[w(x)R(x) − βw(x)log w(x) + βw(x)])

old

= ZoldEx∼ π

[∇θ(w(x)R(x)) − β∇θ(w(x)log w(x)) + β∇θw(x)]

old

= ZoldEx∼ π

[(∇θw(x))R(x) − βw(x)∇θ log πθ(x)(log w(x) + 1) + β(∇θw(x))]

old

= ZoldEx∼ π

[w(x)R(x)∇θ log πθ(x) − βw(x)(log w(x) + 1)∇θ log πθ(x)

old

+βw(x)∇θ log πθ(x)]

= ZoldEx∼ π

w(x)∇θ log πθ(x) R(x) − β(log w(x) + 1) + β

old

= ZoldEx∼ π

##### w(x)∇θ log πθ(x) R(x) − β log w(x)

old

##### = ZoldEx∼ π

old

##### w(x) R(x) − β log w(x) ∇θ log πθ(x) .

This proves the first part of the theorem. Now, consider the surrogate loss function:

LURKL(θ) = ZoldEx∼ π

old −w(x)R(x) + β w(x)log w(x) − w(x) . We compute its gradient:

∇θLURKL(θ) = ZoldEx∼ π

[∇θ(−w(x)R(x)) + β∇θ(w(x)log w(x) − w(x))]

old

= ZoldEx∼ π

[−(∇θw(x))R(x) + β(∇θ(w(x)log w(x)) − ∇θw(x))]

old

= ZoldEx∼ π

[−w(x)R(x)∇θ log πθ(x)

old

+β w(x)(log w(x) + 1)∇θ log πθ(x) − w(x)∇θ log πθ(x)

= ZoldEx∼ π

[−w(x)R(x)∇θ log πθ(x) + βw(x)log w(x)∇θ log πθ(x)]

old

= ZoldEx∼ π

w(x) −R(x) + β log w(x) ∇θ log πθ(x)

old

= −ZoldEx∼ π

old

##### w(x) R(x) − β log w(x) ∇θ log πθ(x) .

Comparing this with the gradient of the objective function, we confirm that ∇θLURKL(θ) = −∇θJURKL(θ). The constant term +1 (corresponding to −βZold in the objective) that appeared in the derivation in Section 3.2 does not affect the gradient and is often omitted from the final loss expression used in practice.

| |
|---|

- L PROOFS FOR REINFORCE-STYLE REGULARIZED POLICY GRADIENTS

This section provides justifications for the REINFORCE-style surrogate loss functions presented in Section 4 (Theorems F.1 to F.5). These proofs demonstrate how automatic differentiation applied to the proposed losses, utilizing the stop-gradient operator SG, yields the correct gradient direction (negative of the objective gradient derived in Section 3).

The core idea relies on the operational definition of the stop-gradient operator SG(·) within automatic differentiation frameworks: ∇θ SG(f(θ)) = 0, while the forward computation uses the value of f(θ). We use the notation w(x) = πθ(x)/πold(x).

- L.1 PROOF OF PROPOSITION F.1 (REINFORCE-STYLE POLICY GRADIENT FOR FORWARD KL)

Proof. The objective is JFKL(θ) = Eπ

θ

[R(x)]−β KL(πold ∥πθ). From Proposition E.1, its gradient is:

∇θJFKL(θ) = Ex∼π

old

   w(x)R(x) + β

WeightFKL(x,θ)

∇θ log πθ(x)

  .

The proposed REINFORCE-style surrogate loss is:

LREINFORCE-styleFKL (θ) = −Ex∼π

old

[SG(w(x)R(x) + β)log πθ(x)].

We compute the gradient of this loss as it would be computed by an automatic differentiation system. Assuming the gradient can be swapped with the expectation:

∇θLREINFORCE-styleFKL (θ) = −Ex∼π

old

[∇θ (SG(w(x)R(x) + β)log πθ(x))]

= −Ex∼π

old

  (∇θ SG(w(x)R(x) + β))

=0 by definition of SG

log πθ(x)

+SG(w(x)R(x) + β)(∇θ log πθ(x))]

= −Ex∼π

old

[SG(w(x)R(x) + β)∇θ log πθ(x)].

This gradient expression, when used in an optimization algorithm (where SG is conceptually removed), corresponds to applying updates proportional to:

−(−Ex∼π

old

[(w(x)R(x) + β)∇θ log πθ(x)]) = ∇θJFKL(θ).

Thus, minimizing LREINFORCE-styleFKL (θ) using gradient descent with automatic differentiation effectively performs gradient ascent on the original objective JFKL(θ).

| |
|---|

- L.2 PROOF OF PROPOSITION F.3 ((REINFORCE-STYLE POLICY GRADIENT FOR UNNORMALIZED FORWARD KL)

Proof. The objective is JUFKL(θ) = Eπ

[R(x)] − β UKL(πold∥πθ). From Proposition 3.2, its gradient is:

θ

  .

  Zold w(x)R(x) − β (w(x) − 1)

∇θJUFKL(θ) = Ex∼ π

∇θ log πθ(x)

old

WeightUFKL(x,θ)

The proposed REINFORCE-style surrogate loss is:

LREINFORCE-styleUFKL (θ) = −Ex∼ π

[SG(Zold (w(x)R(x) − β(w(x) − 1)))log πθ(x)].

Computing the gradient via automatic differentiation: ∇θLREINFORCE-styleUFKL (θ) = −Ex∼ π

[∇θ (SG(Zold(...))log πθ(x))]

old

 (∇θ SG(Zold(...)))

 

= −Ex∼ π

log πθ(x) + SG(Zold(...))(∇θ log πθ(x))

old

=0

= −Ex∼ π

##### [SG(Zold (w(x)R(x) − β(w(x) − 1)))∇θ log πθ(x)].

old

This gradient corresponds to the update direction −∇θJUFKL(θ) when the SG is dropped. Minimizing this loss achieves gradient ascent on JUFKL(θ). If Zold is omitted, the same argument applies to the proportionally scaled objective and loss.

| |
|---|

- L.3 PROOF OF PROPOSITION F.4 (REINFORCE-STYLE LOSS)

Proof. The objective is JRKL(θ) = Eπ

θ

[R(x)]−β KL(πθ ∥πold). From Proposition E.3, its gradient is:

∇θJRKL(θ) = Ex∼π

old

  w(x) R(x) − β(log w(x) + 1)

WeightRKL(x,θ)

∇θ log πθ(x)

  .

The proposed REINFORCE-style surrogate loss is:

LREINFORCE-styleRKL (θ) = −Ex∼π

old

[SG(w(x)(R(x) − β log w(x) − β))log πθ(x)]. Computing the gradient via automatic differentiation: ∇θLREINFORCE-styleRKL (θ) = −Ex∼π

old

[∇θ (SG(w(x)(...))log πθ(x))]

= −Ex∼π

old

 (∇θ SG(w(x)(...)))

=0

log πθ(x) + SG(w(x)(...))(∇θ log πθ(x))

 

= −Ex∼π

old

[SG(w(x)(R(x) − β log w(x) − β))∇θ log πθ(x)].

This gradient corresponds to the update direction −∇θJRKL(θ) when the SG is dropped. Minimizing this loss achieves gradient ascent on JRKL(θ).

| |
|---|

- L.4 PROOF OF PROPOSITION F.5 (REINFORCE-STYLE LOSS FOR UNNORMALIZED REVERSE KL)

Proof. The objective is JURKL(θ) = Eπ

[R(x)] − β UKL(πθ∥πold). From Proposition 3.6, its gradient is:

θ

  Zoldw(x) R(x) − β log w(x)

  .

∇θJURKL(θ) = Ex∼ π

∇θ log πθ(x)

old

WeightURKL(x,θ)

The proposed REINFORCE-style surrogate loss is:

LREINFORCE-styleURKL (θ) = −Ex∼ π

[SG(Zoldw(x)(R(x) − β log w(x)))log πθ(x)]. Computing the gradient via automatic differentiation:

old

∇θLREINFORCE-styleURKL (θ) = −Ex∼ π

[∇θ (SG(Zoldw(x)(...))log πθ(x))]

old

= −Ex∼ π

old

(∇θ SG(Zoldw(x)(...))) =0

log πθ(x)

+ SG(Zoldw(x)(...))(∇θ log πθ(x))

= −Ex∼ π

[SG(Zoldw(x)(R(x) − β log w(x)))∇θ log πθ(x)].

###### This gradient corresponds to the update direction −∇θJURKL(θ) when the SG is dropped. Minimizing this loss achieves gradient ascent on JURKL(θ). If Zold is omitted, the same argument applies to the proportionally scaled objective and loss.

| |
|---|

