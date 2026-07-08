[Figure 1]

# arXiv:2601.10201v2[cs.LG]23May2026

## Future-KL Regularized GRPO: Process-Level Credit Assignment from f-Divergence Regularization

Jiarui Yao, Ruida Wang, Hao Bai, Tong Zhang {jiarui14,tozhang}@illinois.edu

University of Illinois Urbana-Champaign

Group Relative Policy Optimization (GRPO) is widely used for critic-free Large Language Model (LLM) post-training, but its KL regularization is usually implemented as a local loss-side token penalty. We show that this misses the policy-gradient signal induced by autoregressive KL regularization. Unlike standard KL-regularized Reinforcement Learning (RL) objectives, GRPO’s group normalization induces a non-linear prompt-level utility; for binary verifier rewards, this utility is 2arcsin√p. As a result, reward and KL cannot be fused before normalization without changing the implicit objective. We derive the on-policy gradient of GRPO-style objectives with token-wise f-divergence regularization. The reward term recovers the standardized GRPO advantage, while the regularizer term includes a causal future-regularization return-to-go omitted by local KL losses. For reverse KL, this yields a simple future KL correction: add a reverse cumulative sum of per-token log ratios after advantage construction. The resulting method, Future-KL Regularized Policy Optimization (FRPO), requires no critic or extra model passes. On mathematical reasoning tasks, FRPO improves pass@16 in our main large-model setting while maintaining higher entropy and lower policy drift than conventional loss-side KL baselines.

Code: https://github.com/ScaleML/KL-in-LLM-RL

### 1 Introduction

RL has become a central component of post-training LLMs for mathematical reasoning and other verifiable tasks. A widely used paradigm is to sample multiple responses for each prompt, score them using an outcome verifier, and update the policy toward responses that outperform others in the same group. GRPO [Shao et al., 2024, Guo et al., 2025] follows this paradigm and removes the need for a learned critic by normalizing outcome rewards within each prompt group. This critic-free design has made GRPO and its variants attractive for large-scale reasoning training, where outcome rewards are often sparse, binary, and inexpensive to verify.

A key ingredient in such training pipelines is KL regularization against a reference policy. KL regularization stabilizes policy updates, prevents excessive drift from the supervised model, and preserves general capabilities while optimizing for task-specific rewards. In the original GRPO formulation, the KL penalty is added as a token-level loss-side regularizer, often using a sampled estimator such as the k3 estimator [Schulman, 2020]. This implementation is simple and empirically useful, but it obscures several distinct questions: which sampling distribution the sampled KL

[Figure 2]

- Figure 1: Left: Accumulating future KL divergence between the current policy πθ and the reference policy πref contributes to the trajectory-level regularization signal, but is not captured by an independent loss-side token penalty. Right: The dynamics of performance pass@16 for Qwen31.7B-Base and Qwen3-30B-A3B-Base under different KL integrations.

term estimates under, whether its gradient corresponds to the intended KL objective, and, more importantly, how KL regularization should assign credit across tokens in an autoregressive trajectory.

Recent analyses [Zhang et al., 2025b] of KL-regularized policy-gradient methods have clarified several implementation-level issues, including stale-rollout distribution mismatch and the gap between value-unbiased and gradient-unbiased sampled KL estimators. These observations are important, but they do not fully answer a more basic on-policy question: what token-level policy-gradient signal is induced by regularization itself when the expectation is over the current autoregressive policy?

We therefore separate two issues that are often conflated. The first is an estimator issue: rollouts may be sampled from πθold while a KL term is evaluated under πθ, requiring importance correction or fresh log-probabilities. The second is an objective issue: even when samples are on-policy, a loss-side local KL term differentiates only the sampled token penalty and ignores how the current token changes the distribution of future states. This paper focuses on the second issue and shows that it leads to a different token-level credit-assignment rule.

This issue is fundamental in autoregressive language modeling. An early token not only incurs its own local regularization cost, but also changes the future prefix distribution and hence the regularization costs of later tokens. Consequently, token-wise f-divergence regularization induces a process-level credit assignment signal. For reverse KL, the policy-gradient weight for each token should include a future-KL return-to-go term, rather than only an independent token-wise loss penalty.

Why GRPO is not the standard MaxEnt-RL setting. Existing analyses of KL-regularized policy gradients (soft-Q, MaxEnt RL, KL-in-reward PPO/GAE) target objectives that are linear in reward, Eπ[r] − β Eπ[DKL]. Linearity is what allows reward and regularizer to be folded into a single augmented reward, and the future-KL return-to-go is the textbook consequence of differentiating an autoregressive expectation in this regime. GRPO is structurally different. GRPO does not

use a value-function baseline; it constructs advantages by group-normalizing outcome rewards, which makes the effective prompt-level objective non-linear in the expected reward — for binary verifier rewards, ℓ(p) = 2arcsin√p rather than the raw pass rate p. This non-linearity has two consequences absent from prior derivations. First, reward and KL are no longer fuseable: ℓ(E[r − β DKL]) ̸= ℓ(E[r]) − β E[DKL], so naively adding a KL penalty into the reward before group normalization changes the group mean and standard deviation, distorts the implicit arcsin objective, and collapses token-level KL information into a single trajectory-level scalar shared by all tokens. Second, the on-policy gradient acquires a prompt-dependent gain ℓ′(p) on the reward term that the regularizer term does not inherit, so the natural insertion point for a token-level future-KL correction is after advantage construction, not via reward shaping.

We derive the on-policy gradient of a GRPO-style objective under this general non-linear-ℓ + token-wise-f structure, and show that the resulting token-level weight decomposes into two terms: an outcome-level group-relative advantage (the score of the non-linear utility ℓ) and a future regularization correction. This leads to a decoupled future-KL formulation for reverse KL: compute the GRPO advantage from the original verifier reward, and then add a token-dependent futureKL term without modifying the group reward statistics. This separates trajectory-level ranking from process-level KL credit assignment, and differs from loss-side KL correction, KL-in-reward normalization, and actor-critic GAE methods, which respectively address surrogate estimation, coupled reward shaping under linear reward objectives, and value-based credit assignment, rather than critic-free GRPO under a non-linear utility with token-dependent KL regularization.

In summary, this work makes the following contributions:

- 1. We identify that GRPO’s group normalization induces a non-linear prompt-level utility ℓ (the arcsin transform for binary verifier rewards), and we show that this non-linearity makes reward and KL non-fuseable: KL-in-reward modifies the implicit GRPO objective in a way that vanishes only when ℓ is linear.
- 2. Under this non-linear-ℓ + token-wise-f setting, we derive the on-policy policy gradient. The reward term inherits a prompt-dependent gain ℓ′ that the regularizer term does not, recovering the standardized GRPO advantage as the score of the arcsin utility, while the regularizer term decomposes into a local token penalty plus a causal future-regularization return-to-go. The classical loss-side surrogate misses this future term even on-policy.
- 3. The asymmetry between ℓ′-weighted reward and unweighted regularizer dictates a decoupled placement: we propose FRPO, which preserves GRPO’s outcome advantage and adds a token-level future-KL correction after advantage construction. Experiments demonstrate its effectiveness in our main mathematical-reasoning setting.

### 2 Preliminaries and Notations

Consider an autoregressive language model πθ generating a response o = (o1,o2,...,oT) given a prompt x, θ represents the model’s parameters. We write the conditional probability of generating

response o from the prompt x as πθ(o | x) = Tt=1 πθ(ot | x,o<t). Let πref denote a frozen reference policy, and πθold denote the policy at the time of rollout (i.e., the sampling policy) which is an older version of the current policy πθ.

We define the following shorthand for token-level quantities at position t: δt ≜ log πθ(ot | x,o<t) − log πref(ot | x,o<t) = log

πθ(ot|x,o<t) πref(ot|xt,o<t)

,

πθ(ot | x,o<t) πref(ot | x,o<t)

πθ(ot | x,o<t) πθold(ot | x,o<t)

αt ≜

= eδt, ρt ≜

.

For the ease of derivation, we further define ut ≜ αt−1 = πref/πθ. The true token-level reverse KL divergence at position t (given context st = (x,o<t)) is:

πθ(a | st) πref(a | st)

DKL πθ(· | st)∥πref(· | st) =

πθ(a | st)log

, (2.1)

a∈V

where V is the vocabulary. Here we slightly abuse the notations with both a and o representing the actions taken by the policy π, and denote a(j) = (a1,a2,··· ,aj) as the generated tokens so far, and a(−j) = (aj+1,aj+2,··· ,aT) the remaining tokens.

KL Estimators Schulman [2020] proposes three widely adopted KL estimators, Dˆk1 = log α, Dˆk2 = (log α)2/2, and Dˆk3 = 1/α +log α −1, for estimating reverse KL DKL(πθ∥πref), among which k3 is used most commonly due to its unbiased value and nonnegative property. We provide a thorough analysis of the expectation and gradient characteristics in Appendix B. The corresponding forward KL estimators could be defined similarly, with the ratio α being simply reversed. While these estimators motivate the main examples, our policy-gradient derivation in Section 4 applies to general token-wise f-divergence generators.

### 3 Classical GRPO and the Regularization Question

### 3.1 GRPO Objective and Group-Normalized Advantage

DeepSeek-Math [Shao et al., 2024] proposes GRPO as a critic-free policy optimization method. For each prompt x, it samples a group of G responses {oi}Gi=1 ∼ πθold(· | x), evaluates scalar outcome rewards ri = r(x,oi), and optimizes a PPO-style surrogate:

JGRPO(θ) = E

|oi|

G

1 G

1 |oi|

t=1

i=1

min ρi,tAˆi,t,clip(ρi,t,1 − ε,1 + ε)Aˆi,t − β Dˆi,t , (3.1)

where ρi,t = πθ(oi,t | x,oi,<t)/πθold(oi,t | x,oi,<t) and Dˆi,t is a sampled token-level KL penalty against the reference policy.

Unlike PPO with a learned value function, GRPO constructs advantages by normalizing rewards within the prompt group:

µG =

G

1 G

ri, σG =

i=1

G

1 G

(ri − µG)2,

i=1

and assigns every token in response i the same outcome advantage

ri − µG σG

Aˆi,t = AˆGRPOi =

, t = 1,...,|oi|. (3.2)

This design removes the value model but also means that GRPO’s reward signal is trajectory-level: before any KL term is added, all tokens in the same response receive the same credit.

### 3.2 The Implicit Non-Linear Utility from Group Normalization

Before discussing KL regularization, we record a consequence of group normalization that is the structural source of every novelty in this paper. The arcsin objective induced by standardized binary rewards has also been noted in REINFORCE-Ada [Xiong et al., 2025b]. We revisit it here not as a new policy-gradient observation, but to expose a GRPO-specific complication: in binary-reward reasoning tasks, the normalization in Eq. (3.2) is not a neutral preprocessing step; it determines a non-linear prompt-level utility followed by the population GRPO gradient. Standard derivations of KL-regularized policy gradients (soft-Q, MaxEnt RL, KL-in-reward PPO) target objectives that are linear in reward, Eπ[r] − β Eπ[DKL], where reward and regularizer fuse into a single augmented reward. GRPO breaks that linearity, and as we show below, this is what makes putting KL into the reward before normalization structurally — not just numerically — incorrect.

- Proposition 3.1 (Implicit objective of binary GRPO). Fix a prompt x with binary reward r(x,a) ∈

{0,1} and pass probability pθ(x) = Ea∼πθ(·|x)[r(x,a)]. Replacing finite-group statistics with their population values, the expected GRPO policy gradient for this prompt is

Ea∼πθ(·|x)

r(x,a) − pθ(x) pθ(x)(1 − pθ(x))∇θ log πθ(a|x) = ∇θ 2arcsin pθ(x) .

The proof is given in Appendix C. Group normalization should therefore be viewed as part of the objective, not merely as a numerical rescaling: the standardized GRPO advantage is the score of ℓ(p) = 2arcsin√p, with ℓ′(p) = 1/ p(1 − p). Because ℓ is non-linear, the operations of folding KL into the reward and applying ℓ do not commute, ℓ(E[r − β DKL]) ̸= ℓ(E[r]) − β E[DKL], and modifying the reward before normalization changes the implicit arcsin objective by an amount that depends on the per-prompt KL distribution and the curvature of ℓ.

- Proposition 3.2 (KL-in-reward distorts the implicit utility under non-linear ℓ). Let r˜i = ri −

β j δi,j and AˆKLRi = (˜ri − µ˜G)/σ˜G. In the population limit, the gradient driven by AˆKLR targets a prompt-level objective that reduces to ℓ(E[r]) − β E[DKL] only when ℓ is affine. Under GRPO’s ℓ(p) = 2arcsin√p, the discrepancy is non-vanishing whenever the group has non-trivial KL spread.

GRPO’s outcome advantage should continue to rank responses by verifier reward, while KL regularization should provide a separate token-level credit-assignment signal — this is the structural reason behind the general regularized objective in Eq. (4.1), and the predicted “KL-in-reward collapse” is observed in Section 6 (Appendix F.1 discusses the alternative placements in detail).

3.3 Loss-Side KL in the Classical Formulation

The KL term commonly used in GRPO implementations is the sampled k3 estimator [Schulman, 2020]. With ut = πref(ot | st)/πθ(ot | st) = αt−1, it is

Dˆt(k3)(πθ∥πref) = ut − log ut − 1 =

1 αt − log

1 αt − 1. (3.3)

- Proposition 3.3 (Value unbiasedness under on-policy sampling). For a fixed state st, Dˆt(k3) is an unbiased value estimator of DKL[πθ(·|st)∥πref(·|st)] when the sampled token is drawn from πθ(·|st).

This proposition is about the scalar value of a sampled KL estimator. It does not by itself imply that placing Dˆt(k3) as a local differentiable loss term gives the policy gradient of a KL-regularized autoregressive objective. Even in the on-policy case πθold = πθ, a token ai changes not only its own local KL term but also the distribution of future prefixes, and therefore the KL costs at later positions. The next section derives this missing causal term directly.

### 4 On-Policy f-Divergence Regularization

- 4.1 General Policy Gradient Consider the on-policy regularized objective

Qf(θ) = Ex ℓ Ea∼πθ(·|x)r(x,a) − β ExEa∼πθ(·|x)

T

πref(ai | si) πθ(ai | si)

f(ui), ui =

. (4.1)

i=1

Here f is a differentiable convex generator applied to the token-level likelihood ratio; KL is only one special case. The function ℓ captures the prompt-level reward objective induced by the advantage construction. For binary-reward GRPO, Proposition 3.1 gives ℓ(p) = 2arcsin√p. Thus the derivation below applies to GRPO as a special case while also covering a broader class of token-wise f-divergence regularizers between the current and reference policies. The formulation should be read as a sampled autoregressive analogue of per-state f-divergence regularization: at each generated prefix si, the penalty is evaluated on the sampled action through the ratio πref(ai|si)/πθ(ai|si), and the expectation over trajectories supplies the state distribution induced by πθ.

- Theorem 4.1 (On-policy gradient with token-wise regularization). For any baseline b(si) that does not depend on the current action ai, the gradient of Eq. (4.1) can be written as

where

∇θQf(θ) = ExEa∼πθ(·|x)

T

wi ∇θ log πθ(ai | si), wi = sg(Ai − β regi), (4.2)

i=1

Ai = ℓ′(Ea∼πθr(x,a))r(x,a) − b(si), (4.3) regi =

f(uj) − f′(ui)ui =

f(uj) + f(ui) − f′(ui)ui. (4.4)

j≥i

j>i

Here sg denotes the implementation convention that the sampled coefficient is treated as a stopgradient weight in the policy-gradient surrogate; the equality is the corresponding likelihood-ratio gradient identity. The proof is in Appendix D.3.1. Two structural features of Eq. (4.2) drive every algorithmic choice that follows. First, ℓ multiplies Ai but not regi: the reward signal carries a prompt-dependent gain ℓ′(Er), while the regularizer term is independent of ℓ. This asymmetry is invisible in linear-reward MaxEnt RL (ℓ(p) = p, ℓ′ ≡ 1) and is what blocks the standard “fold KL into reward” move under GRPO normalization: any reward shaping is rescaled by ℓ′ along with the verifier reward, whereas the genuine token-level regularizer is not. Second, the sum over j ≥ i is the gradient through the future state distribution induced by the autoregressive policy, and it is exactly what a purely local loss-side penalty misses by differentiating only f(ui) at the sampled token.

The theorem also clarifies why the issue is not specific to the reverse KL estimator. Whenever the regularizer is placed inside an expectation over trajectories, differentiating the objective produces two

contributions. The first is the local derivative of the penalty at the current action, −f′(ui)ui. The second is a score-function term for all future penalties, j≥i f(uj), because changing ai changes the distribution of all later prefixes. In bandit problems these two notions collapse, but in autoregressive language generation they are different. Therefore, any token-wise regularizer that is optimized as an on-policy trajectory objective induces a causal cost-to-go, and the right place to add this cost-to-go is on the regularizer side of the ℓ′-asymmetry — i.e., after advantage construction, as a token-level correction that is not subjected to ℓ′.

### 4.2 Specialization to GRPO and Reverse KL

Corollary 4.2 (GRPO advantage as the score of the arcsin utility). Theorem 4.1 specializes to binary-reward GRPO via ℓ(p) = 2arcsin√p, with the population GRPO advantage AGRPOi = (r(x,a) − pθ(x))/ pθ(x)(1 − pθ(x)) = ℓ′(pθ(x))(r(x,a) − pθ(x)) recovered as the score of ℓ. Since regi is unmultiplied by ℓ′, any reward-side modification is rescaled by ℓ′(p) together with r while a genuine token-level regularizer is not, so the only placement consistent with both signals is to insert the regularization term after advantage construction.

The GRPO advantage and the future-KL correction are thus objects of different scaling type under ℓ and cannot be merged into a single reward without distorting one of them. Section 5 turns this into a one-line algorithmic change.

For token-wise reverse KL, taking f(u) = −log u gives, up to action-independent constants that vanish under the score-function identity,

 AGRPOi − β

 ∇θ log πθ(ai | si). (4.5)

T

πθ(aj | sj) πref(aj | sj)

∇θQrKL(θ) = ExEa∼πθ

log

j≥i

i=1

Thus each token receives the usual outcome advantage plus a causal future-KL correction. Early tokens are penalized for the regularization cost they induce downstream, not merely for their own local log-ratio.

The same theorem also covers the k3 generator f(u) = −log u + u − 1. Substituting it into Eq. (4.4) yields the coefficient

 −log ui +

 . (4.6)

AGRPOi − β

(−log uj + uj − 1)

j>i

This expression differs from differentiating a local k3 loss term at each sampled token: the latter captures only the direct derivative of the token penalty, while Eq. (4.6) also includes the future regularization terms. The distinction matters because k3 was designed primarily as a low-variance nonnegative value estimator for KL. In our setting the regularizer also acts as a token-level credit signal, so the sign and centering of the cumulative term affect how the policy assigns blame or credit to earlier tokens.

### 4.3 What the Loss-Side Surrogate Misses

The classical loss-side implementation in Eq. (3.1) treats Dˆi,t as an additive differentiable token loss. For example,

∇θDˆt(k3) = (1 − ut)∇θ log πθ(at | st).

This is a local derivative evaluated on the sampled token and prefix. It does not differentiate the expectation over future trajectories in Eq. (4.1); consequently, it omits the j > i terms in Eq. (4.4). This is the on-policy issue studied in this paper. It is separate from the familiar stale-rollout or off-policy issue in PPO-style training.

- Proposition 4.3 (Local loss-side KL omits future state-distribution terms). Consider the on-policy objective in Eq. (4.1). A surrogate that differentiates only the sampled local penalty f(ui) at each token recovers the direct term −f′(ui)ui but omits the score-function contribution j≥i f(uj) in

- Eq. (4.4). Hence it is not the policy gradient of the trajectory-level regularized objective, except in degenerate settings where future penalties are action-independent.

For the reverse k3 estimator f(u) = −log u + u − 1, the local differentiable loss gives the coefficient 1 − ui. The on-policy trajectory objective instead gives

regi = −log ui +

(−log uj + uj − 1),

j>i

where the second term is the missing future regularization cost. For the forward k3 estimator f(u) = ulog u − u + 1, the same calculation gives

regi = 1 − ui +

(uj log uj − uj + 1),

j>i

again showing that the missing term is not an artifact of reverse KL. These expressions are the regularization analogues of reward-to-go in policy gradient: past costs can be dropped by the score-function identity, but future costs cannot.

Practical off-policy caveat. In real systems, rollouts are produced by πθold and actor updates may use mini-batches after the policy has already changed. Then even the value estimate in

- Proposition 3.3 is no longer exactly on-policy unless importance weighting or fresh log-probabilities are used. This implementation-level mismatch can matter, but it is orthogonal to our main point:

even when πθold = πθ, the correct regularized policy gradient contains a future regularization return-to-go.

This distinction is useful when comparing implementations. Importance weighting can correct a mismatch between the behavior policy and the current policy in a sampled expectation, but it does not create the missing future term. Conversely, adding the future term can improve the on-policy credit assignment even when PPO epochs are set to one and rollout staleness is small. Importantly, once the future-KL correction is folded into the advantage, the whole token weight is multiplied by the PPO importance ratio ρi,t in the clipped surrogate, just like the outcome advantage. This gives the regularization signal the same first-order off-policy correction as the policy-gradient term. By contrast, a separate additive loss-side KL penalty is not multiplied by ρi,t in the classical surrogate, so its sampled expectation remains distribution-mismatched under stale rollouts. The two issues therefore operate on different axes: distribution correction concerns which policy generated the tokens, while future KL concerns what objective those tokens estimate.

### 5 Future-KL GRPO and Optimality View

### 5.1 The Decoupled FRPO Update

Equation (4.5) suggests a lightweight modification of GRPO: compute the group-normalized outcome advantage from the original verifier reward, and then add a token-dependent future-KL correction. For response i and token t, FRPO uses

 AˆGRPOi − β

 , δi,j = log

Ti

πθ(ai,j | si,j) πref(ai,j | si,j)

AˆFRPOi,t = sg

. (5.1)

δi,j

j=t

The finite-sample implementation uses rollout log-probabilities for efficiency and then optimizes the usual clipped PPO surrogate with AˆFRPOi,t ; see Appendix E. Because the future-KL correction is part of the advantage, the PPO update uses ρi,tAˆFRPOi,t (and its clipped counterpart), so both the verifier reward signal and the regularization signal receive the same importance-ratio correction from πθold to πθ. This decoupled form preserves GRPO’s outcome-reward normalization while restoring token-level credit assignment from the regularizer.

Simple change, structural reason. Operationally, Eq. (5.1) adds about ten lines of code to a standard GRPO trainer: one reverse cumulative sum over per-token log-ratios already computed during rollout, with no extra forward passes and no learned value function. The non-trivial part is the placement, not the implementation: Corollary 4.2 shows that the GRPO advantage and the future-KL correction live on different sides of the ℓ′-asymmetry, so the only placement consistent with the on-policy gradient under non-linear ℓ is after advantage construction. The alternative placements fail by prediction: Proposition 3.2 explains the KL-in-reward collapse observed in Section 6, and

- Proposition 4.3 explains why loss-side KL omits the future return-to-go even on-policy.

We discuss alternative ways of inserting KL before GRPO normalization in Appendix F.1. Briefly, trajectory-level KL-in-reward changes the group mean and standard deviation, while step-level normalization is fragile for small groups and variable-length responses. The decoupled update therefore keeps GRPO normalization on verifier rewards and adds future KL afterward as a token correction.

- 5.2 Optimal KL-Regularized Rewards

The same future-KL structure also appears from the optimality conditions of entropy-regularized RL. For a fixed prompt, consider

Q(π) = Ea∼π(·|x)[r(x,a)] − β DKL[π(·|x)∥πref(·|x)]. (5.2)

- Theorem 5.1 (Optimal policy). The maximizer satisfies π∗(a | x) ∝ πref(a | x)exp(r(x,a)/β).

- Proposition 5.2 (Constant shifted reward). Equivalently, for each prompt x there exists a constant C(x) such that

π∗(a | x) πref(a | x)

r(x,a) − β log

= C(x). (5.3)

Because autoregressive likelihood ratios decompose over tokens, Eq. (5.3) implies the following process-reward form.

Theorem 5.3 (Future KL as process reward). At the optimal policy, the entropy-regularized process reward after prefix a(t) can be written as

T

rt(x,a(t)) = r(x,a) − β

j=t+1

π∗(aj | x,a(j−1)) πref(aj | x,a(j−1))

. (5.4)

log

Proofs are deferred to Appendix D.4. This optimality view is not needed to derive Theorem 4.1, but it gives the same interpretation: KL regularization is a process-level signal, and the relevant token weight is a future cost-to-go rather than an isolated per-token penalty.

The optimality view also explains why the future term is naturally causal. For two adjacent prefixes,

- Eq. (5.4) implies

rt(x,a(t)) − rt−1(x,a(t−1)) = β log

π∗(at | x,a(t−1)) πref(at | x,a(t−1))

.

Thus the change in process reward between consecutive steps is exactly a token-level log-ratio at the selected action. The future-KL correction in FRPO can be viewed as a practical on-policy approximation to this structure, with πθold used for rollout log-probabilities and GRPO’s groupnormalized advantage used in place of a learned process value. This gives a direct bridge between the policy-gradient derivation and the process-level interpretation of KL-regularized optimality.

6 Experiments and Results

In this section, we briefly summarize the experiments with their results. For details about hyperparameters, other configurations, and more experiment results, please refer to Appendix G.

Basic Configuration We select Qwen3-1.7B-Base and Qwen3-4B-Base [Yang et al., 2025] for dense models, and Qwen3-30B-A3B-Base for MoE models as base models. For the training dataset, we use a filtered version of DAPO-Math-17k [Yu et al., 2025], which contains about 12,000 problems. For the evaluation benchmarks, we choose MATH500 [Hendrycks et al., 2021], AIME24, AIME25, AMC23, OlympiadBench [He et al., 2024], and MinervaMath [Lewkowycz et al., 2022]. For the metric pass@16, we perform a bootstrap by sampling with replacement 1,000 times and report the mean.

- 6.1 Main Experiments

- Figure 2 demonstrates that incorporating the future-KL correction leads to higher Pass@n performance in the Qwen3-30B-A3B-Base setting, outperforming the strongest baseline in this comparison by nearly 5% absolute gain. This is accompanied by higher final entropy and lower PPO-KL, which measures policy drift from the reference policy for FRPO. The finetuned model remains close to the reference policy as measured by PPO-KL divergence, suggesting that the base model already contains useful long-Chain-of-Thought (CoT) reasoning patterns and that future-KL credit assignment helps elicit them without excessive drift. Evaluations on the AIME24 and AIME25 benchmarks also show that FRPO achieves better pass@n performance than using KL in loss directly for the large model. The Qwen3-1.7B-Base result is weaker, and we hypothesize that this model may be closer to its reasoning capacity under the current training budget and hyperparameters.

###### PPO-KL

###### Entropy

###### Response Length

MATH500 Pass@16

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

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

0.6

0.015

3000

0.95

0.010

0.4

2000

0.90

0.005

0.2

1000

0.85

0.000

0.0

0

0.80

0 100 200

0 100 200

0 100 200

0 100 200

Step

Step

Step

Step

GRPO (no KL) GRPO + KL-loss GRPO + KL-reward FRPO

Figure 2: The training dynamics of Qwen-30B-A3B-Base with different KL integrations.

- Table 1: Full evaluation results of pass@16 (%) for Qwen3-1.7-Base, Qwen3-4B-Base, and Qwen330B-A3B-Base on different benchmarks. Each entry reports the mean with the standard deviation in parentheses.

Model MATH500 AIME24 AIME25 AMC23 OlympiadBench MinervaMath

Qwen3-1.7B-Base 69.65 (2.48) 12.94 (2.87) 7.15 (2.45) 52.76 (4.01) 38.80 (0.75) 36.02 (1.16) + GRPO (no KL) 78.62 (2.01) 25.82 (2.92) 16.76 (2.72) 69.73 (2.97) 57.23 (0.58) 48.60 (0.83) + GRPO KL-loss 77.07 (1.68) 27.15 (2.65) 18.33 (3.26) 81.56 (3.23) 54.90 (0.56) 46.36 (0.90) + GRPO KL-reward collapse / / / / /

+ FRPO 85.30 (3.00) 27.00 (2.53) 20.31 (3.73) 74.47 (3.00) 59.13 (0.65) 47.15 (1.03)

Qwen3-4B-Base 70.99 (1.08) 21.08 (2.37) 22.26 (3.17) 71.39 (3.27) 52.27 (0.68) 46.98 (0.90) + GRPO (no KL) 86.15 (0.85) 29.43 (2.87) 34.16 (2.48) 85.73 (2.67) 64.28 (0.51) 55.07 (0.79) + GRPO KL-loss 91.11 (1.36) 46.53 (2.83) 50.89 (3.65) 90.10 (2.11) 72.50 (0.48) 58.64 (0.87) + GRPO KL-reward collapse / / / / /

+ FRPO 92.93 (1.70) 41.09 (3.43) 42.17 (4.03) 92.41 (2.16) 72.23 (0.51) 58.56 (0.76)

Qwen3-30B-A3B-Base 87.77 (2.08) 35.31 (3.38) 21.04 (2.39) 76.67 (2.44) 61.36 (0.76) 51.68 (0.95) + GRPO (no KL) 91.63 (0.87) 53.06 (2.99) 27.45 (2.30) 91.44 (1.31) 66.33 (0.49) 57.46 (0.68) + GRPO KL-loss 90.23 (1.17) 51.25 (4.52) 22.04 (3.05) 89.06 (2.57) 62.86 (0.49) 55.51 (0.70) + GRPO KL-reward collapse / / / / /

+ FRPO 96.07 (1.54) 63.88 (3.86) 45.23 (3.41) 94.25 (1.35) 77.71 (0.47) 58.65 (0.71)

### 6.2 Ablation Studies

##### 6.2.1 KL Direction and Estimator Ablations

Forward KL versus Reverse KL. Forward KL DKL(πref∥πθ) is mode-covering: it penalizes the current policy for assigning too little probability to regions supported by the reference policy.

Reverse KL DKL(πθ∥πref) is more mode-seeking: it discourages generations that are unlikely under the reference. Figure 3 shows that reverse KL works better in our setting, likely because forward KL keeps entropy higher than is useful for math reasoning.

Different KL Estimators. Many RL frameworks favor k3 over k1 because both are value-unbiased for reverse KL under on-policy sampling, while k3 is always nonnegative. In FRPO, however, the future-KL term is used as a signed credit-assignment signal rather than only as a scalar divergence monitor. The nonnegativity of k3 makes its cumulative future term one-sided: it always pushes the token weight in the penalty direction, even when a centered signal would be needed to distinguish relatively helpful and harmful deviations within a group. Figure 3 shows a gap between k1 and k3

- 5.0

2.5

0.0

2.5

1e 5 PPO-KL

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 100 200

Step

0.00

0.25

0.50

0.75

1.00

1.25

Entropy

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 100 200

Step

1000

1500

2000

2500

3000

3500

Response Length

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

0 100 200

Step

0.70

0.75

0.80

0.85

MATH500 Pass@16

FRPO (reverse k1) FRPO (reverse k3) FRPO (forward k1) FRPO (forward k3)

Figure 3: Reverse KL versus forward KL, with k1 and k3 as the estimators on Qwen3-1.7B-Base.

estimators, especially on forward KL, so we use k1 as the default estimator in FRPO.

- 6.2.2 Subtracting a Baseline for k3 Estimators

7.5

0 100 200

Step

Due to the nonnegativity of k3 estimators, the future KL term will always be a penalty, therefore pushing the policy towards the same direction. To avoid such a phenomenon, we implement a naive baseline, which takes the average KL divergence at the token level among all tokens in the same batch, and then scaled by the remaining length of a particular response from the current token i, i.e.,

bi = (T − i) · DKL.

In Table 2, we summarize the results of incorporating baseline subtraction into the k3 estimators for both forward and reverse KL. We observe that baseline subtraction alone does not lead to performance improvements. We hypothesize that this is because the estimated baselines are relatively coarse and do not accurately approximate the true baselines. More sophisticated baseline estimation methods may be beneficial, and we leave their investigation to future work.

- Table 2: Ablation studies with subtracting a baseline from the nonnegative k3 KL estimator. Experiments are done on Qwen3-1.7B-Base. Performances are measured by Pass@16 (%).

Method MATH500 AIME24 AIME25 AMC23 OlympiadBench MinervaMath

Forward k3 with baseline 75.21 (2.37) 20.68 (2.39) 10.45 (2.52) 72.66 (2.95) 52.50 (0.60) 46.86 (0.90) Forward k3 without baseline 75.62 (2.55) 22.32 (3.22) 18.94 (3.31) 70.55 (2.84) 53.05 (0.66) 46.37 (0.98) Reverse k3 with baseline 68.27 (0.75) 20.63 (2.55) 19.89 (2.89) 76.36 (3.23) 54.14 (0.60) 47.82 (1.01) Reverse k3 without baseline 84.82 (3.43) 32.23 (4.29) 11.76 (1.91) 79.16 (3.83) 58.57 (0.62) 48.02 (1.02)

##### 6.2.3 Ablation for Different Advantage Estimators

In existing RL frameworks such as verl [Sheng et al., 2024] and AReaL [Fu et al., 2025], KL-in-reward can be combined with the GAE advantage estimator. This folds the regularizer into token rewards before advantage estimation. In contrast, FRPO preserves the GRPO-style group-normalized outcome advantage and adds future KL afterward. Figure 4, based on Qwen3-30B-A3B-Base, suggests that retaining GRPO’s outcome-level normalization is beneficial compared with integrating KL into rewards before estimating advantages.

###### PPO-KL

###### Entropy

###### Response Length

MATH500 Pass@16

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
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

3500

0.96

0.000100

0.6

3000

0.94

0.000075

2500

0.4

0.92

0.000050

2000

0.000025

0.2

0.90

1500

0.000000

1000

0.0

0.88

0 100 200

0 100 200

0 100 200

0 100 200

Step

Step

Step

Step

GAE + KL-reward FRPO

Figure 4: Comparison between GAE and GRPO advantage estimation, both with future KL. Other ablation studies and full evaluation results could be found in Appendix G.

### 7 Conclusion and Discussion

We revisited KL regularization in GRPO from the on-policy objective rather than only from the implementation of a local loss penalty. The resulting policy-gradient formula applies to general token-wise f-divergence regularizers and shows that autoregressive regularization induces a future return-to-go term. For reverse KL, this gives the future-KL correction used by FRPO, which keeps GRPO’s group-normalized outcome advantage separate from token-level regularization credit. The same structure is supported by the optimality view of KL-regularized trajectory rewards. In mathematical-reasoning experiments, this formulation improves pass@n while limiting policy drift and entropy collapse.

### References

Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. Large language models for mathematical reasoning: Progresses and challenges. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop, pages 225–237, 2024.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.

Riccardo Della Vecchia, Alena Shilova, Philippe Preux, and Riad Akrour. Entropy regularized reinforcement learning with cascading networks. arXiv preprint arXiv:2210.08503, 2022.

Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. arXiv preprint arXiv:1802.06070, 2018.

Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, Tongkai Yang, Binhang Yuan, and Yi Wu. Areal: A large-scale asynchronous reinforcement learning system for language reasoning, 2025. URL https://arxiv.org/abs/2505. 24298.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. Reinforce++: Stabilizing critic-free policy optimization with global advantage normalization. arXiv preprint arXiv:2501.03262, 2025.

Audrey Huang, Wenhao Zhan, Tengyang Xie, Jason D Lee, Wen Sun, Akshay Krishnamurthy, and Dylan J Foster. Correcting the mythos of kl-regularization: Direct alignment without overoptimization via chi-squared preference optimization. arXiv preprint arXiv:2407.13399, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Tadashi Kozuno, Wenhao Yang, Nino Vieillard, Toshinori Kitamura, Yunhao Tang, Jincheng Mei, Pierre M´enard, Mohammad Gheshlaghi Azar, Michal Valko, R´emi Munos, et al. Kl-entropyregularized rl with a generative model is minimax optimal. arXiv preprint arXiv:2205.14211, 2022.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35: 3843–3857, 2022.

Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https: //huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/project-numina/ aimo-progress-prize/blob/main/report/numina_dataset.pdf), 2024.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023.

Jingbin Liu, Xinyang Gu, and Shuai Liu. Policy optimization reinforcement learning with entropy regularization. arXiv preprint arXiv:1912.01557, 2019.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, et al. Gdpo: Group rewarddecoupled normalization policy optimization for multi-reward rl optimization. arXiv preprint arXiv:2601.05242, 2026.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Yu Luo, Shuo Han, Yihan Hu, Dong Li, and Jianye Hao. Ratio-variance regularized policy optimization for efficient llm fine-tuning. arXiv preprint arXiv:2601.03320, 2026.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022.

Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Defeating the training-inference mismatch via fp16. arXiv preprint arXiv:2510.26788, 2025.

Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, and Wee Sun Lee. Rethinking the trust region in llm reinforcement learning. arXiv preprint arXiv:2602.04879, 2026.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

John Schulman. Approximating kl divergence. http://joschu.net/blog/kl-approx.html, 2020. Blog post.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Daniil Tiapkin, Nikita Morozov, Alexey Naumov, and Dmitry P Vetrov. Generative flow networks as entropy-regularized rl. In International Conference on Artificial Intelligence and Statistics, pages 4213–4221. PMLR, 2024.

Nino Vieillard, Tadashi Kozuno, Bruno Scherrer, Olivier Pietquin, R´emi Munos, and Matthieu Geist. Leverage the average: an analysis of kl regularization in reinforcement learning. Advances in Neural Information Processing Systems, 33:12163–12174, 2020a.

Nino Vieillard, Bruno Scherrer, Olivier Pietquin, and Matthieu Geist. Momentum in reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pages 2529–2538. PMLR, 2020b.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022a.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022b.

Muning Wen, Junwei Liao, Cheng Deng, Jun Wang, Weinan Zhang, and Ying Wen. Entropyregularized token-level policy optimization for language agent reinforcement. arXiv preprint arXiv:2402.06700, 2024.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. arXiv preprint arXiv:2312.11456, 2023.

Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025a.

Wei Xiong, Chenlu Ye, Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, Jiang Bian, Nan Jiang, and Tong Zhang. Reinforce-ada: An adaptive sampling framework under non-linear rl objectives. arXiv preprint arXiv:2510.04996, 2025b.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Fengkai Yang, Zherui Chen, Xiaohan Wang, Xiaodong Lu, Jiajun Chai, Guojun Yin, Wei Lin, Shuai Ma, Fuzhen Zhuang, Deqing Wang, et al. Your group-relative advantage is biased. arXiv preprint arXiv:2601.08521, 2026.

Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. Your efficient rl framework secretly brings you off-policy rl training, August 2025a. URL https: //fengyao.notion.site/off-policy-rl.

Jiarui Yao, Yifan Hao, Hanning Zhang, Hanze Dong, Wei Xiong, Nan Jiang, and Tong Zhang. Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and rl. arXiv preprint arXiv:2505.02391, 2025b.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Lifan Yuan, Wendi Li, Huayu Chen, Ganqu Cui, Ning Ding, Kaiyan Zhang, Bowen Zhou, Zhiyuan Liu, and Hao Peng. Free process rewards without process labels. arXiv preprint arXiv:2412.01981, 2024.

Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, and Lu Qianchun. Learning like humans: Advancing llm reasoning capabilities via adaptive difficulty curriculum learning and expert-guided self-reformulation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6630–6644, 2025a.

Yifan Zhang, Yifeng Liu, Huizhuo Yuan, Yang Yuan, Quanquan Gu, and Andrew Chi-Chih Yao. On the design of kl-regularized policy gradient algorithms for llm reasoning. arXiv preprint arXiv:2505.17508, 2025b.

Heyang Zhao, Chenlu Ye, Quanquan Gu, and Tong Zhang. Sharp analysis for kl-regularized contextual bandits and rlhf. arXiv preprint arXiv:2411.04625, 2024.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

- A Related Work

RL for LLM Policy Optimization In the classical RL domain, TRPO [Schulman et al., 2015] and PPO [Schulman et al., 2017] are generally utilized to optimize models. As LLMs become popular, RL algorithms are applied to further improve their abilities [Rafailov et al., 2023, Jaech et al., 2024, Ouyang et al., 2022, Bai et al., 2022]. Since the release of DeepSeek-R1 [Guo et al., 2025], critic-free policy optimization [Liu et al., 2025, Hu et al., 2025] has become more and more popular in optimizing LLM, as it could be formulated into a bandit problem in a vast number of scenarios, leading to simple adaptations to many training settings. Based on the initially proposed GRPO [Shao et al., 2024, Guo et al., 2025] algorithm, many follow-up variants appear to resolve the bias [Liu et al., 2025, Yang et al., 2026], mitigate the variance [Qi et al., 2026, Luo et al., 2026], improve the sample efficiency [Yao et al., 2025b, Xiong et al., 2025b], and derive simpler and more effective algorithms [Xiong et al., 2025a, Yu et al., 2025, Zhang et al., 2025a, Liu et al., 2026, Zheng et al., 2025].

Entropy-Regularized RL From the theoretical perspective, [Zhao et al., 2024] establishes a sharp convergence rate compared to normal RL without regularization. Other works [Huang et al., 2024, Xiong et al., 2023, Vieillard et al., 2020a, Liu et al., 2019, Wen et al., 2024, Eysenbach et al., 2018, Della Vecchia et al., 2022, Vieillard et al., 2020b] also demonstrate the effectiveness of regularization in RL optimization both from mathematical analysis and empirical results. Besides, entropy-regularized RL is also an important technique in training models other than LLM, e.g., Tiapkin et al. [2024], Kozuno et al. [2022], as it limits the deviation from the base model when it is strong enough and maintains the generalizability on domains not RL finetuned on.

LLM for Reasoning RL Due to the strong capabilities of pretrained LLMs, they have been used to tackle challenging scenarios requiring complex reasoning [Wei et al., 2022a, Jin et al., 2025, Kumar et al., 2024]. Among various reasoning domains, mathematical reasoning is particularly important due to its ease of verification and its foundational role for other tasks [Wei et al., 2022b, Shao et al., 2024, Yang et al., 2024, Ahn et al., 2024, Lightman et al., 2023, Ahn et al., 2024, Yuan et al., 2024, Cui et al., 2025]. The base models, evaluation metrics, and other supporting techniques are relatively mature; therefore, we also select math reasoning as the empirical task to verify the effectiveness of our proposed method. Commonly used benchmarks [Hendrycks et al., 2021, He et al., 2024, Lewkowycz et al., 2022] and open-sourced datasets [Yu et al., 2025, LI et al., 2024] are also available for conducting large-scale experiments.

- B KL Estimators

- B.1 Reverse KL Estimator Variants and Their Properties

Following Schulman [2020], we analyze four estimators. Since the original GRPO objective and popular RL frameworks like verl use reverse KL as the default option, we first summarize the expectations and gradients for reverse KL-based estimators, i.e., DKL(πθ∥πref).

##### Definition B.1 (KL Estimators).

- Dˆ(k1) = δ = log α,
- Dˆ(k2) = 21δ2 = 12(log α)2,

- Dˆ(k3) = α−1 + δ − 1 = α−1 + log α − 1,

πθ(a) πref(a)

Dˆ(full) =

πθ(a)log

.

a∈V

k1 is used in aligning InstructGPT [Ouyang et al., 2022], while k3 is used in DeepSeek-Math [Shao et al., 2024].

##### B.1.1 Expectation Properties

Under Eπθ:

- • Eπθ[Dˆ(k1)] = DKL[πθ∥πref]. Unbiased, but can be negative for individual tokens.
- • Eπθ[Dˆ(k2)] = 12Varπθ[δ] + 12(DKL)2 ̸= DKL in general. Biased as a value estimator.

- • Eπθ[Dˆ(k3)] = DKL[πθ∥πref] (Proposition 3.3). Unbiased and always ≥ 0.
- • Dˆ(full) is exact (no sampling involved).

##### B.1.2 Gradient Properties

The gradient of the expected KL loss under the sampling distribution πθold depends critically on the estimator choice. We analyze ∇θEπθ

[Dˆ] for each estimator, noting that πref is constant w.r.t. θ.

old

- k1 gradient. Since ∇θDˆ(k1) = ∇θ log πθ(ot | st), the expected gradient under πθ is:

Eπθ ∇θDˆ(k1) = Eπθ[∇θ log πθ] =

o∈V

πθ(o|s)∇θπθ(o|s) πθ(o|s)

= ∇θ

o∈V

πθ(o|s) = 0.

This is the score function identity—the expected gradient of k1 is zero under πθ, not ∇θDKL. Under πθold (without importance weighting), Eπθ

old

[∇θDˆ(k1)] gives a biased gradient estimate of ∇θDKL[πθ∥πref].

- k2 gradient. We have ∇θDˆ(k2) = δ · ∇θδ = δ · ∇θ log πθ.

Proposition B.2. Under on-policy sampling, k2 provides an unbiased gradient estimator: Eπθ ∇θDˆ(k2) = ∇θDKL[πθ∥πref].

Proof. Since ∇θδ = ∇θ log πθ (the πref term vanishes):

πθ(a) πref(a)

∇θDKL[πθ∥πref] = ∇θ

πθ(a)log

a

πθ(a) πref(a)

∇θπθ(a) · log

=

+

a

a

∇θπθ(a) πθ(a)

πθ(a) ·

πθ(a) · ∇θ log πθ(a) · δa +

∇θπθ(a)

=

a

a

= 0

= Eπθ[∇θ log πθ · δ] = Eπθ ∇θDˆ(k2) .

| |
|---|

Moreover, under πθold ≈ πθ (e.g., early stage in PPO epochs), it remains approximately unbiased. Under off-policy sampling (πθold ̸= πθ), the k2 gradient estimator also becomes biased, though the bias is generally smaller than that of k1 or k3.

##### k3 gradient. We have:

∇θDˆ(k3) = ∇θ

1 α

1 α ∇θ log πθ.

+ log α − 1 = 1 −

The expected gradient under πθ:

πref πθ ∇θ log πθ

Eπθ ∇θDˆ(k3) = Eπθ 1 −

πref(a)∇θπθ(a) πθ(a)

∇θπθ(a) −

=

a

a

= 0 − Eπref ∇θπθ πθ

= −Eπref ∇θπθ πθ

.

In general, this does not equal ∇θDKL[πθ∥πref]. Thus the expected gradient of k3 is biased.

##### Straight-through trick (verl’s k3+ [Sheng et al., 2024]). Define:

−sg D ˆ(k2) + sg D ˆ(k3) ,

Dˆ(k3+) = Dˆ(k2)

backward

where sg(·) denotes stop-gradient. This yields: Forward value: Dˆ(k3+) = Dˆ(k3),

Backward gradient: ∇θDˆ(k3+) = ∇θDˆ(k2) = δ · ∇θ log πθ. This combines the unbiased value of k3 with the unbiased gradient of k2. We briefly summarize the properties of different KL estimators in Table 3.

- Table 3: Properties of Reverse KL estimators under Eπθ. All sample-based estimators become biased

under Eπθ

when πθold ̸= πθ. Estimator Value unbiased Gradient unbiased Non-negative

old

- k1 ✓ × ×
- k2 × ✓ ✓
- k3 ✓ × ✓ k3+ ✓ ✓ ✓ full exact exact ✓

### B.2 Forward KL Estimators

For the forward KL formulation DKL(πref∥πθ), we can similarly define KL estimators by reversing the log-ratio:

Definition B.3 (Forward KL Estimators).

- Dˆforward(k1) = log α−1 = log u = −δ,
- Dˆforward(k2) =

- 1

- 2

(−δ)2 =

- 1

- 2

δ2 =

- 1

- 2

(log α)2,

- Dˆforward(k3) = α − δ − 1,

πref(a) πθ(a)

Dˆforward(full) =

.

πref(a)log

a∈V

##### B.2.1 Expectation properties.

Under Eπref:

Eπref D ˆforward(k1) = Eπref[−δ] = DKL[πref∥πθ].

Thus k1 is an unbiased value estimator for the forward KL, but it can be negative for individual tokens.

- For k2, we have

Eπref D ˆforward(k2) =

- 1

- 2

Eπref[δ2] =

- 1

- 2

Varπref[δ] +

- 1

- 2

(DKL[πref∥πθ])2 ̸= DKL[πref∥πθ] in general. Hence k2 is biased as a value estimator.

- For k3, since Eπref[α] = 1, we obtain

Eπref D ˆforward(k3) = Eπref[α − δ − 1] = Eπref[−δ] = DKL[πref∥πθ].

Moreover, by the inequality x − log x − 1 ≥ 0 for x > 0, Dˆforward(k3) ≥ 0 for every token. Therefore, k3 is an unbiased and nonnegative value estimator for the forward KL.

Finally, Dˆforward(full) is exact, since it explicitly sums over the whole vocabulary.

##### B.2.2 Gradient properties.

We now analyze

∇θEπref D ˆforward , noting that πref is constant with respect to θ.

##### k1 gradient. Since

∇θDˆforward(k1) = ∇θ(−δ) = −∇θ log πθ, we have

Eπref ∇θDˆforward(k1) = −Eπref [∇θ log πθ]. On the other hand,

πref(a) πθ(a)

∇θDKL[πref∥πθ] = ∇θ

πref(a)log

a

= −

πref(a)∇θ log πθ(a)

a

= −Eπref [∇θ log πθ]. Therefore,

Eπref ∇θDˆforward(k1) = ∇θDKL[πref∥πθ]. Thus k1 provides an unbiased gradient estimator for the forward KL under πref sampling.

###### k2 gradient. For k2, we have ∇θDˆforward(k2) = ∇θ

- 1

- 2

δ2 = δ · ∇θ log πθ. Hence

Eπref ∇θDˆforward(k2) = Eπref [δ · ∇θ log πθ]. This is not equal to

∇θDKL[πref∥πθ] = −Eπref [∇θ log πθ] in general. Therefore, although k2 is a second-order approximation to the KL value near α = 1, it is generally biased as a gradient estimator for the forward KL.

###### k3 gradient. For k3, we have

∇θDˆforward(k3) = ∇θ(α − δ − 1)

= α∇θ log πθ − ∇θ log πθ

= (α − 1)∇θ log πθ. Taking expectation under πref gives

Eπref ∇θDˆforward(k3) = Eπref [(α − 1)∇θ log πθ]

= Eπref [α∇θ log πθ] − Eπref [∇θ log πθ]

= Eπθ [∇θ log πθ] − Eπref [∇θ log πθ]

= 0 − Eπref [∇θ log πθ]

= ∇θDKL[πref∥πθ].

Thus k3 also provides an unbiased gradient estimator for the forward KL under πref sampling, while additionally being nonnegative and value-unbiased.

Under πref sampling, the forward KL estimators have the following properties in Table 4. Table 4: Properties for Forward KL Estimators

##### Estimator Value Unbiased Gradient Unbiased Nonnegative

k1 ✓ ✓ × k2 × × ✓ k3 ✓ ✓ ✓ full exact exact ✓

### C Implicit Arcsin Objective of Binary GRPO

This appendix proves Proposition 3.1. We use the same notation as the main text. Fix a prompt x, let a ∼ πθ(·|x) be a sampled response, and assume a binary verifier reward r(x,a) ∈ {0,1}. Define

pθ(x) := Ea∼πθ(·|x)[r(x,a)]. We analyze the population version of GRPO, replacing the empirical group mean and standard deviation by

µ(x) = pθ(x), σ(x) = pθ(x)(1 − pθ(x)), for pθ(x) ∈ (0,1). The corresponding population GRPO advantage is

r(x,a) − pθ(x) pθ(x)(1 − pθ(x))

AGRPO(x,a) =

.

Proof of Proposition 3.1. By the likelihood-ratio identity,

∇θpθ(x) = ∇θEa∼πθ(·|x)[r(x,a)] = Ea∼πθ(·|x) [r(x,a)∇θ log πθ(a|x)]. Also,

Ea∼πθ(·|x)[∇θ log πθ(a|x)] = 0. Therefore,

Ea∼πθ(·|x) AGRPO(x,a)∇θ log πθ(a|x)

Ea∼πθ(·|x)[(r(x,a) − pθ(x))∇θ log πθ(a|x)] pθ(x)(1 − pθ(x))

=

= ∇θpθ(x) pθ(x)(1 − pθ(x))

.

Finally,

2arcsin√p =

d dp

1 p(1 − p)

, so the last display equals

∇θ 2arcsin pθ(x) .

- Remark C.1 (Finite groups and degenerate cases). The proof uses population group statistics. In finite groups, GRPO replaces pθ(x) and pθ(x)(1 − pθ(x)) by their empirical estimates. This introduces finite-sample noise and degenerate groups when all sampled responses have the same

binary reward, in which case σG = 0 and implementations typically mask or skip the prompt. The proposition should therefore be read as the population objective followed by the GRPO estimator.

- Remark C.2 (Variance-stabilizing intuition). For a Bernoulli reward, AGRPO is the standardized reward and satisfies

E[AGRPO] = 0, E[(AGRPO)2] = 1.

Thus group normalization removes the direct pθ(x)(1−pθ(x)) reward-variance factor. The transform 2arcsin√p is the corresponding variance-stabilizing transform for Bernoulli means.

- Remark C.3 (Score-of-utility view; cf. Corollary 4.2). The proof above also establishes Corollary 4.2:

the standardized GRPO advantage is exactly ℓ′(pθ(x))(r(x,a) − pθ(x)) for ℓ(p) = 2arcsin√p. The non-linearity of ℓ is the structural reason that any reward-side modification (in particular, KL-inreward) is rescaled by ℓ′(p) together with r, while a token-level regularizer added after normalization is not — see Proposition 3.2 for the resulting non-commutativity of ℓ with KL shaping.

### D Missing Proofs

### D.1 Proof Sketch of Proposition 3.2

We give a short structural argument; the key observation is that the population gradient driven by the GRPO advantage on a KL-shaped reward targets the score of the variance-stabilizing transform of the shaped reward, not the sum of two independent gradients.

Under binary r and the population GRPO advantage from Proposition 3.1, the expected policy gradient equals the gradient of the implicit utility ℓ(pθ(x)) = 2arcsin pθ(x) (Corollary 4.2). Now replace ri by the KL-shaped trajectory reward r˜i = ri − β j δi,j, with population statistics µ˜(x) = E[˜r] = pθ(x) − β K(θ) and σ˜(x) = Std[˜r], where K(θ) = Ea∼πθ j δj abbreviates the prompt-level KL.

The standardized advantage AˆKLR(x,a) = (˜r(x,a) − µ˜(x))/σ˜(x) drives, in expectation,

1 σ˜(x) ∇θ pθ(x) − β K(θ) , by the same chain-rule identity used in Proposition 3.1. Two structural consequences follow.

Ea∼πθ A ˆKLR(x,a)∇θ log πθ(a|x) =

- (i) The factor 1/σ˜(x) multiplies both ∇θpθ and β ∇θK(θ). The desired objective ℓ(pθ(x)) − β K(θ) has its KL gradient unweighted by ℓ′, so the KL-in-reward update misweights the regularization signal by exactly ℓ′(pθ(x)) = 1/ pθ(x)(1 − pθ(x)), which varies across prompts and over training. The two gradients agree only when ℓ′ ≡ const, i.e., when ℓ is affine.

- (ii) The denominator σ˜(x) itself depends on β: for binary r, Var[˜r] = pθ(1 − pθ) − 2β Cov[r,D] + β2 Var[D],

where D = j δj, so the implicit utility carries a non-trivial β-dependence that does not factor as

“ℓ minus regularizer.” Together (i) and (ii) imply that the population objective targeted by AˆKLR reduces to ℓ(E[r]) − β E[DKL] only in the affine case ℓ(p) = ap + b, which fails for GRPO’s ℓ(p) = 2arcsin√p.

### D.2 KL Estimator Value Unbiasedness

Proof of Proposition 3.3. Fix a state s and write u(a) = πref(a | s)/πθ(a | s). Under on-policy sampling a ∼ πθ(· | s),

πref(a | s) πθ(a | s)

πθ(a | s) πref(a | s) − 1

Eπθ D ˆ(k3) =

πθ(a | s)

+ log

a∈V

πθ(a | s) πref(a | s) −

πref(a | s) +

πθ(a | s)log

πθ(a | s)

=

a∈V

a∈V

a∈V

= DKL[πθ(· | s)∥πref(· | s)].

| |
|---|

### D.3 On-Policy GRPO with KL Regularization

- D.3.1 Derivation of the Regularized Policy-Gradient Weight Proof. We derive the policy-gradient weight in Eq. (4.4). Recall the objective function,

T

Q(θ) = Exℓ(Ea∼πθr(x,a)) − β · ExEa∼πθ

f

i=1

πref(ai | si) πθ(ai | si)

.

For a fixed prompt x, denote

r¯θ(x) := Ea∼πθr(x,a), ui :=

πref(ai | si) πθ(ai | si)

,

T

πθ(a | x) =

πθ(ai | si).

i=1

The objective is

Q(θ) = Exℓ(¯rθ(x)) − βExEa∼πθ

T

f(ui).

i=1

Then

∇θQ(θ) = ∇θExℓ(¯rθ(x)) − β∇θExEa∼πθ

T

f(uj).

j=1

First, consider the reward term. By the chain rule, ∇θℓ(¯rθ(x)) = ℓ′(¯rθ(x))∇θr¯θ(x)

= ℓ′(¯rθ(x))∇θEa∼πθr(x,a).

Using the likelihood-ratio identity, ∇θEa∼πθr(x,a) = ∇θ

a

πθ(a | x)r(x,a)

=

a

∇θπθ(a | x)r(x,a)

πθ(a | x)∇θ log πθ(a | x)r(x,a)

=

a

= Ea∼πθ [r(x,a)∇θ log πθ(a | x)]. Since

log πθ(a | x) =

T

log πθ(ai | si),

i=1

we have

T

∇θEa∼πθr(x,a) = Ea∼πθ r(x,a)

∇θ log πθ(ai | si)

i=1

T

= Ea∼πθ

r(x,a)∇θ log πθ(ai | si).

i=1

Therefore,

∇θℓ(¯rθ(x)) = Ea∼πθ

T

ℓ′(¯rθ(x))r(x,a)∇θ log πθ(ai | si).

i=1

For any baseline b(si),

Eai∼πθ(·|si) [b(si)∇θ log πθ(ai | si)]

πθ(ai | si)∇θ log πθ(ai | si)

= b(si)

ai

= b(si)

ai

∇θπθ(ai | si)

= b(si)∇θ

= 0.

ai

πθ(ai | si)

Thus the reward contribution can be written as

T

∇θℓ(¯rθ(x)) = Ea∼πθ

i=1

where

Ai∇θ log πθ(ai | si),

Ai := ℓ′(¯rθ(x))r(x,a) − b(si)

= ℓ′(Ea∼πθr(x,a))r(x,a) − b(si).

Next, consider the regularization term. Define

T

G(θ;x) := Ea∼πθ

f(uj).

j=1

Then

For each j,

∇θG(θ;x) = ∇θ

a

πθ(a | x)

T

∇θπθ(a | x)

=

a

j=1

 

 

T

= Ea∼πθ

f(uj)

j=1

T

f(uj)

j=1

T

πθ(a | x)∇θ

f(uj) +

f(uj)

a

j=1

 ∇θ log πθ(a | x) +

 .

T

∇θf(uj)

j=1

πref(aj | sj) πθ(aj | sj)

uj =

.

The derivative below is the direct derivative of the sampled token penalty conditional on the realized prefix sj; the dependence of later prefixes on earlier actions is accounted for by the score-function term above. Since πref is fixed,

πref(aj | sj) πθ(aj | sj)

∇θuj = ∇θ

πref(aj | sj) πθ(aj | sj)2 ∇θπθ(aj | sj)

= −

πref(aj | sj) πθ(aj | sj) ∇θ log πθ(aj | sj)

= −

= −uj∇θ log πθ(aj | sj).

Hence

∇θf(uj) = f′(uj)∇θuj

= −f′(uj)uj∇θ log πθ(aj | sj).

Substituting this into ∇θG(θ;x) gives

 

 

 

 

T

T

T

f′(uj)uj∇θ log πθ(aj | sj)

∇θG(θ;x) = Ea∼πθ

∇θ log πθ(ai | si) −

f(uj)

j=1

i=1

j=1

 

 ∇θ log πθ(ai | si).

T

T

f(uj) − f′(ui)ui

= Ea∼πθ

i=1

j=1

Now we convert the first summation into its causal reward-to-go form. For j < i, the quantity f(uj) is measurable with respect to the history si = (x,a<i). Therefore,

Ea∼πθ [f(uj)∇θ log πθ(ai | si)]

= Esi f(uj)Eai∼πθ(·|si) [∇θ log πθ(ai | si)]

= Esi f(uj)

ai

πθ(ai | si)∇θ log πθ(ai | si)

= Esi f(uj)

ai

∇θπθ(ai | si)

= Esi f(uj)∇θ

= 0.

ai

πθ(ai | si)

Thus the past terms j < i can be dropped from the score-function coefficient, and we obtain

 

 ∇θ log πθ(ai | si).

T

f(uj) − f′(ui)ui

∇θG(θ;x) = Ea∼πθ

j≥i

i=1

Finally, since

Q(θ) = Exℓ(¯rθ(x)) − βExG(θ;x), we have

 Ai − β

 

 

 ∇θ log πθ(ai | si).

T

f(uj) − f′(ui)ui

∇θQ(θ) = ExEa∼πθ

j≥i

i=1

Therefore, defining

 Ai − β

 

 

 ,

f(uj) − f′(ui)ui

wi := stopgrad

j≥i

we get

∇θQ(θ) = ExEa∼πθ

T

wi∇θ log πθ(ai | si).

i=1

| |
|---|

### D.4 KL in Reward from the Perspective of Optimality

- D.4.1 Proof of Theorem 5.1 Proof. Consider for a fixed prompt x ∈ V∗, and our optimization objective is:

{Ea∼π(·|x)[r(x,a)] − βDKL(π∥πref)}

max

π(·|x)

π(a|x) πref(a|x)

π(a|x) · r(x,a) − β ln

= max

π(·|x)

a∈V∗

under the condition a∈V∗ π(a|x) = 1. Thus, we construct the Lagrangian of the above optimization problem,

,

L(π(·|x),λ) =

π(a|x) · r(x,a) − β ln

a∈V∗

π(a|x) πref(a|x) − λ 1 −

π(a|x) .

a∈V∗

Setting the partial derivative of L(π,λ) with respect to π equal to 0, ∂

∂π(a|x)L(π,λ) = 0, we have

π(a|x) πref(a|x) − λ 1 −

∂ ∂π(a|x) a∈V∗

π(a|x) · r(x,a) − β ln

a∈V∗

π(a|x) πref(a|x)

⇔ r(x,a) − β ln

+ 1 − λ = 0

1 β · λ − 1 = ln(π(a|x)) − ln(πref(a|x))

1 β · r(x,a) −

⇔

1 β · r(x,a) −

1 β · λ − 1

⇔ lnπ(a|x) = lnπref(a|x) +

1 β · r(x,a) −

1 β

⇔ π(a|x) = πref(a|x)exp

λ − 1 ∝ πref(a|x)exp

Therefore, we complete the proof.

π(a|x) = 0

1 β

r(x,a) .

D.4.2 Proof of Theorem 5.3 Proof. From Proposition 5.2, under optimal policy π∗ and reward r:

π∗(a|x) πref(a|x)

C = r(x,a) − β ln

.

Given the auto-regressive generation property of language modeling, we have:

π∗(a|x) πref(a|x)

ln

Therefore, we have:

T j=1 π∗(aj|x,a(j−1))

= ln

=

T j=1 πref(aj|x,a(j−1))

T

ln

j=1

π∗(aj|x,a(j−1)) πref(aj|x,a(j−1))

.

t

C = r(x,a) − β

ln

j=1

T

π∗(aj|x,a(j−1)) πref(aj|x,a(j−1)) − β

ln

j=t+1

π∗(aj|x,a(j−1)) πref(aj|x,a(j−1))

| |
|---|

###### Algorithm 1 Future-KL Regularized Policy Optimization (FRPO)

Require: Policy πθ, reference policy πref, rollout batch B, mini-batch size Bmini, group size G, KL coefficient

β, PPO clip ϵ

- 1: for each training iteration do
- 2: Set rollout policy πθ

old ← πθ

- 3: for each prompt x ∈ B do
- 4: Sample responses {ai}Gi=1 ∼ πθ

old

(·|x)

- 5: Evaluate rewards ri = r(x,ai) and token masks mi,t
- 6: Compute µG = G1 i ri and σG = G1 i(ri − µG)2

- 7: AˆGRPOi ← (ri − µG)/(σG + εstd)
- 8: for each response token t of response i do
- 9: δi,t ← log πθ(ai,t|x,ai,<t) − log πref(ai,t|x,ai,<t) ▷ Recompute the probability w.r.t. πθ
- 10: Si,t ← Tj=i t mi,jδi,j ▷ KL-to-go from current token
- 11: AˆFRPOi,t ← sg(AˆGRPOi − βSi,t)
- 12: end for
- 13: end for
- 14: Split the rollout batch into actor mini-batches of size Bmini
- 15: Update θ once per mini-batch using PPO with

LPPOi,t (θ) = min ρi,tAˆFRPOi,t ,clip(ρi,t,1 − ϵ,1 + ϵ)AˆFRPOi,t , where ρi,t = πθ(ai,t|x,ai,<t)/πθ

old

(ai,t|x,ai,<t).

- 16: end for

Therefore,

rt(x,a(t))

 r(x,a) − β

= Ea(−t)∼π∗(·|x,a(t))

 C + β

t

= Ea(−t)∼π∗(·|x,a(t))

ln

j=1

= C′,

T

π∗(aj|x,a(j−1)) πref(aj|x,a(j−1))

ln

j=t+1

 

π∗(aj|x,a(j−1)) πref(aj|x,a(j−1))

 

for some constant C′ ∈ R, where the final equation holds due to the fact that a(t) is fixed given the conditional distribution π∗(·|x,a(t)). Thus, the process reward rt(x,a(t)) should be a constant no matter what the future path is under the optimal policy π∗. Therefore, given the whole reasoning trajectory a = (a1,··· ,aT), the universal equality holds:

T

π∗(aj|x,a(j−1)) πref(aj|x,a(j−1))

rt(x,a(t)) = r(x,a) − β

ln

.

j=t+1

| |
|---|

### E Algorithmic Details for FRPO

Implementation notes. The default future term uses the sampled log-ratio (k1) estimator. A k3 variant replaces δi,j in Si,t by −log ui,j + ui,j − 1, where ui,j = πref(ai,j|si,j)/πθold(ai,j|si,j). All

sums are masked over response tokens and exclude prompt and padding tokens; the default FRPO advantage includes the current response token in the KL-to-go. With multiple PPO epochs, the future-KL term can either remain fixed from rollout log-probabilities or be recomputed under the current policy at each actor update. In our experiments, PPO epochs are set to 1: a train batch is rolled out once for efficiency and then split into actor mini-batches, each used for a single update. Refer to Algorithm 1 for details.

### F Discussion about Future KL

Table 5: Where this paper sits among KL-regularized policy-gradient analyses. Linear ℓ allows reward and regularizer to fuse into a single augmented reward; under GRPO’s non-linear ℓ(p) = 2arcsin√p they cannot, and the right insertion point for the future-KL correction is on the regularizer side of the ℓ′-asymmetry.

Setting Prompt objective Reward+KL fuseable? Future-reg in PG? Insertion point MaxEnt RL / soft-Q Eπ[r] − β DKL yes (augmented reward) yes (textbook) either PPO + KL-in-reward (+GAE) Eπ[r] − β DKL yes yes (via value) reward shaping GRPO (this paper) ℓ(Eπ[r]) − β DKL no (Prop. 3.2) yes (Thm. 4.1) after advantage

### F.1 Why Not Put KL into the Reward Before Normalization?

Section 5 uses a decoupled update: compute the GRPO outcome advantage first, then add a tokenwise future-KL correction. There are three natural ways to combine future KL with GRPO-style normalization.

- (a) Decoupled: Compute GRPO advantage from the original outcome reward, then add the future-KL correction:

Aˆi,t =

ri − µG σG − β ·

T

j=t

δi,j.

- (b) Coupled, trajectory-level: Augment the outcome reward with the full trajectory KL, then apply group normalization:

r˜i = ri − β ·

T

j=1

δi,j, A˜i =

r˜i − µ˜G σ˜G

.

This yields a uniform advantage for all tokens in response i—the per-token differentiation from future KL is lost entirely. It also changes the group mean and standard deviation, so the regularizer affects not only policy drift but also the scale and ordering used to construct the GRPO advantage.

- (c) Coupled, step-level: Use the step-dependent process reward ri,t and normalize across the group at each step:

ri,t − µG,t σG,t

1 G

Aˆi,t =

, µG,t =

G

rk,t, σG,t = stdk(rk,t).

k=1

In this strategy, ri,t uses token-level KL instead of trajectory-level KL, leading to possibly higher variance. In addition, trajectories have different lengths, so at large t only a subset of trajectories contributes, further degrading the estimates.

Analysis. Strategy (b) defeats the purpose of process rewards: by collapsing the future KL into a single trajectory-level scalar, all tokens share the same advantage, and the fine-grained credit assignment from Theorem 5.3 is discarded. Moreover, as formalized in Proposition 3.2, modifying the reward before group normalization is structurally incompatible with the non-linear utility ℓ(p) = 2arcsin√p: the modified standard deviation σ˜G ≠ pˆ(1 − pˆ) breaks the variance-stabilizing property that makes the GRPO gradient the score of ℓ, and the failure does not vanish with tuning because ℓ is not affine. Strategy (c) is closer to the process-reward interpretation, but suffers from statistical instability: with small groups, the per-step normalization constants µG,t and σG,t are noisy estimates, and variable trajectory lengths mean that for large t, very few trajectories contribute to the statistics. This can lead to erratic advantage values in the later portion of responses.

Strategy (a), used by FRPO, avoids both problems by separating the two roles of the advantage:

- • Trajectory ranking is handled by AˆGRPOi , which benefits from stable group normalization on outcome rewards. Crucially, since the original reward enters the normalization unchanged, the implicit 2arcsin(√p) objective is preserved.

- • Token differentiation is handled by the future KL term −β · Tj=t δi,j, which operates within each trajectory and does not require cross-trajectory normalization.

### G More Experiment Details

We show the experiment configurations in Table 6 and Table 7. We use vLLM [Kwon et al., 2023]

- as the inference backend to speedup rollout. Most experiments are conducted on 4×8 NVIDIA H20 GPU nodes.

Table 6: Hyperparameters for Baselines.

#### Hyperparameters Qwen3-1.7B-Base Qwen3-4B-Base Qwen3-30B-A3B-Base

Learning Rate 1e-6 1e-6 1e-6 PPO Epochs 1 1 1 Max Prompt Length 2048 2048 2048 Max Response Length 8192 8192 8192 Train Batch Size 256 256 256 PPO Mini Batch Size 64 64 64 Rollout Temperature 1.0 1.0 1.0 Group Size 5 5 5

For math reasoning, we add an additional prompt behind each problem, as shown in G.1.

Table 7: Hyperparameters for FRPO.

##### Hyperparameters Qwen3-1.7B-Base Qwen3-4B-Base Qwen3-30B-A3B-Base

Learning Rate 1e-6 1e-6 1e-6 PPO Epochs 1 1 1 Max Prompt Length 2048 2048 2048 Max Response Length 8192 8192 8192 Train Batch Size 256 256 256 PPO Mini Batch Size 64 64 64 Rollout Temperature 1.0 1.0 1.0 Group Size 5 5 5 β 0.01 0.01 0.01

G.1: Prompt for math reasoning {problem} Let’s think step by step and output the final answer within \boxed{}.

### G.1 Experiments on Qwen3-4B-Base

###### PPO-KL

###### Entropy

###### Response Length

MATH500 Pass@16

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

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

4000

0.00020

0.90

0.6

0.00015

0.85

3000

0.4

0.00010

0.80

2000

0.00005

0.75

0.2

0.00000

0.70

1000

0.0

0 100 200

0 100 200

0 100 200

0 100 200

Step

Step

Step

Step

GRPO (no KL) GRPO + KL-loss GRPO + KL-reward FRPO

Figure 5: Training dynamics for Qwen3-4B-Base model with different KL integrations.

The training dynamics of the Qwen3-4B-Base model are shown in Figure 5. The training trend is similar to that of other models like Qwen3-1.7B-Base and Qwen3-30B-A3B-Base.

### G.2 Other Evaluation Metrics

In previous sections, we mainly focused on the metric pass@n, as we hope that KL regularization could help maintain the base model’s inherent ability and alleviate entropy collapse. From Figure 2 and Figure 5, we could see that the convergence entropy of FRPO is indeed at a higher value, leading to a higher pass@n performance. Here, we also show the mean@n performance.

From Figure 6, Figure 7, and Figure 8, we can see that FRPO not only improves pass@n performance

Qwen3-1.7B-Base

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.70

Accuracy

0.65

0.60

0.55

0 50 100 150 200

Step

GRPO (no KL)

GRPO + KL-reward

GRPO + KL-loss

FRPO

- Figure 6: The training dynamics of mean@16 accuracy on MATH500 for Qwen3-1.7B-Base under different KL integrations.

but also achieves a comparable or even better mean@n accuracy than other methods that directly put KL in the loss or the reward. Especially on Qwen3-30B-A3B-Base, where off-policy issues are more pronounced than in other dense models due to its MoE structure, using future KL in FRPO helps stabilize training and yields a smoother training process.

### G.3 Full Evaluation Results

We use AIME24, AIME25, AMC23, OlympiadBench [He et al., 2024], and MinervaMath [Lewkowycz et al., 2022] for evaluation besides MATH500. The information about each benchmark can be found in Table 8

Table 8: Information about benchmarks.

Benchmark Size

- AIME24 30
- AIME25 30 AMC23 40 OlympiadBench 674 MinervaMath 272

Table 1 and Table 9 summarize the evaluation results of all models with different KL integrations on several benchmarks for pass@16 and mean@16, respectively. Note that using (reverse) KL in the reward directly leads to model collapse.

### G.4 Using Current KL versus Stale KL

To fully utilize the computation resources for the rollout engine, the rollout train batch is usually larger than the actor update mini-batch. In our experiments, the train batch size is 256 prompts and

Qwen3-4B-Base

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.85

0.80

Accuracy

0.75

0.70

0.65

0 50 100 150 200 250

Step

GRPO (no KL)

GRPO + KL-reward

GRPO + KL-loss

FRPO

- Figure 7: The training dynamics of mean@16 accuracy on MATH500 for Qwen3-4B-Base under different KL integrations.

the PPO mini-batch size is 64 prompts. Thus, a rollout stage samples responses for 256 prompts

- at once, and the actor then performs updates on four mini-batches. The PPO epoch is set to 1, so each mini-batch is used once rather than repeatedly optimized. This design improves rollout throughput, while any staleness comes from policy changes across mini-batch updates within the same rollout batch; using multiple PPO epochs would introduce an additional stale-rollout effect.

Qwen3-30B-A3B-Base

0.900

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.875

0.850

Accuracy

0.825

0.800

0.775

0.750

0 50 100 150 200 250

Step

GRPO (no KL)

GRPO + KL-reward

GRPO + KL-loss

FRPO

- Figure 8: The training dynamics of mean@16 accuracy on MATH500 for Qwen3-30B-A3B-Base under different KL integrations.

Table 9: Full evaluation results of mean@16 (%) for Qwen3-1.7-Base, Qwen3-4B-Base, and Qwen330B-A3B-Base on different benchmarks.

Model MATH500 AIME24 AIME25 AMC23 OlympiadBench MinervaMath Qwen3-1.7B-Base 56.71 2.08 0.83 13.44 11.21 9.79

+ GRPO (no KL) 69.23 8.33 4.17 40.16 33.25 28.33 + GRPO KL-loss 69.29 9.38 4.58 42.03 30.80 27.23 + GRPO KL-reward / / / / / /

+ FRPO 70.90 9.79 4.79 41.56 31.64 25.46

Qwen3-4B-Base 63.90 4.79 4.17 21.25 18.03 17.10 + GRPO (no KL) 80.41 16.25 12.29 61.41 44.95 39.48 + GRPO KL-loss 85.41 23.54 23.13 65.94 53.46 40.65 + GRPO KL-reward / / / / / /

+ FRPO 84.11 21.67 19.58 63.28 51.03 39.22

Qwen3-30B-A3B-Base 77.16 9.17 3.96 33.28 23.44 18.45 + GRPO (no KL) 87.10 24.17 17.71 75.16 51.00 44.28 + GRPO KL-loss 84.83 22.29 10.21 66.56 47.53 42.69 + GRPO KL-reward / / / / / /

###### + FRPO 88.13 32.08 24.17 75.63 58.32 43.18

### H Case Studies

In this section, we briefly show concrete cases where models finetuned with the future-KL formulation answer hard problems that the model constrained with only the current single-step KL does not solve. From H.1, we observe that the model finetuned with future KL typically induces longer responses, suggesting that FRPO can elicit long CoT behavior from the base model while maintaining the reference-policy constraint.

H.1: Qwen3-30B-A3B-Base AIME25 Problem

Find the sum of all integer bases b > 9 for which 17b is a divisor of 97b. Please reason step by step, and put your final answer within \boxed{}. Answer 70.

- Wrong answer 1 To solve the problem, we need to convert the numbers 17b and 97b from base b to base 10 and then find the integer bases b > 9 for which 17b is a divisor of 97b. First, let’s convert 17b and 97b to base 10:

17b = 1 · b + 7 = b + 7 97b = 9 · b + 7 = 9b + 7

We need to find the integer bases b > 9 such that b + 7 is a divisor of 9b + 7. This means that there exists an integer k such that:

9b + 7 = k(b + 7) Rearranging the equation, we get:

9b + 7 = kb + 7k 9b − kb = 7k − 7 b(9 − k) = 7(k − 1)

7(k − 1) 9 − k

b =

Since b is an integer, 7(9k−−k1) must be an integer. Therefore, 9−k must be a divisor of 7(k −1). Let’s find the possible values of k.

The divisors of 7 are ±1,±7. We will check each case:

- 1. 9 − k = 1 =⇒ k = 8

b =

7(8 − 1) 1

= 49

- 2. 9 − k = −1 =⇒ k = 10

7(10 − 1) −1

= −63 not valid since b > 9.

b =

- 3. 9 − k = 7 =⇒ k = 2

b =

7(2 − 1) 7

= 1 not valid since b > 9.

- 4. 9 − k = −7 =⇒ k = 16

7(16 − 1) −7

= −15 not valid since b > 9.

b =

The only valid value is b = 49. We need to check if there are any other possible values of k. Since 9 − k must be a divisor of 7, we have checked all possible divisors. Therefore, the only integer base b > 9 for which 17b is a divisor of 97b is b = 49. The sum of all such integer bases is:

|49|
|---|

- Wrong answer 2 To solve the problem, we need to interpret the numbers 17b and 97b in base b and determine the integer bases b > 9 for which 17b is a divisor of 97b. First, let’s convert 17b and 97b from base b to base 10:

- • 17b = 1 · b + 7 = b + 7
- • 97b = 9 · b + 7 = 9b + 7

We need to find the integer bases b > 9 such that b + 7 is a divisor of 9b + 7. This means that there exists an integer k such that:

9b + 7 = k(b + 7) Rearranging the equation, we get:

9b + 7 = kb + 7k 9b − kb = 7k − 7 b(9 − k) = 7(k − 1)

7(k − 1) 9 − k

b =

Since b is an integer, 7(9k−−k1) must be an integer. Therefore, 9−k must be a divisor of 7(k −1). Let’s find the possible values of k such that b > 9.

We will iterate over possible values of k and check if b is an integer greater than 9. Let’s write the code to find the valid bases b.

# Function to find the sum of all valid bases b > 9 def find_valid_bases():

valid_bases = [] # Iterate over possible values of k for k in range(1, 9):

if (9 - k) != 0 and (7 * (k - 1)) % (9 - k) == 0:

b = (7 * (k - 1)) // (9 - k) if b > 9:

valid_bases.append(b) return valid_bases

# Find the valid bases valid_bases = find_valid_bases() # Calculate the sum of valid bases sum_valid_bases = sum(valid_bases) print((valid_bases, sum_valid_bases))

([32, 57], 89)

The valid bases b > 9 for which 17b is a divisor of 97b are 32 and 57. The sum of these bases is 32 + 57 = 89. Therefore, the final answer is

|89|
|---|

.

- Correct answer 1 Alright, I have this problem:

##### Find the sum of all integer bases b > 9 for which 17b is a divisor of 97b.

First, I need to understand what 17b and 97b mean. The subscript b indicates that these numbers are in base b. So, 17b means 1 × b + 7 in base 10, and 97b means 9 × b + 7 in base 10. Let’s write that down:

17b = 1 × b + 7 = b + 7 97b = 9 × b + 7 = 9b + 7

The problem states that 17b is a divisor of 97b. In other words, b + 7 divides 9b + 7 without leaving a remainder. Mathematically, this can be expressed as:

9b + 7 ≡ 0 (mod b + 7)

This means that 9b+7 is a multiple of b+7. To find the values of b that satisfy this condition, I need to solve the congruence. Let’s denote d = b + 7. Then, b = d − 7. Substitute b in the expression 9b + 7:

9b + 7 = 9(d − 7) + 7 = 9d − 63 + 7 = 9d − 56 So, the congruence becomes:

9d − 56 ≡ 0 (mod d) Simplifying this, we get:

9d ≡ 56 (mod d) But 9d is a multiple of d, so 9d ≡ 0 (mod d). Therefore:

0 ≡ 56 (mod d)

This implies that d divides 56. So, d is a divisor of 56. Let’s list all the positive divisors of 56:

1,2,4,7,8,14,28,56 But we know that d = b + 7 and b > 9. Therefore:

d = b + 7 > 9 + 7 = 16 So, d must be a divisor of 56 that is greater than 16. Looking at the list of divisors:

28,56

So, d can be 28 or 56. Now, let’s find the corresponding b for each d:

- 1. If d = 28: b = d − 7 = 28 − 7 = 21
- 2. If d = 56: b = d − 7 = 56 − 7 = 49

So, the possible values of b are 21 and 49. Finally, the problem asks for the sum of all such b:

21 + 49 = 70 Therefore, the sum of all integer bases b > 9 for which 17b is a divisor of 97b is:

|70|
|---|

- Correct answer 2 Alright, I have this problem:

##### Find the sum of all integer bases b > 9 for which 17b is a divisor of 97b.

First, I need to understand what 17b and 97b mean. The subscript b indicates that these numbers are in base b. So, 17b and 97b are numbers written in base b.

- Step 1: Convert 17b and 97b to Base 10 To work with these numbers more easily, I’ll convert them to base 10.

- 1. Converting 17b to base 10: In base b, the number 17b represents:

1 × b1 + 7 × b0 = b + 7

- 2. Converting 97b to base 10:

Similarly, 97b represents:

9 × b1 + 7 × b0 = 9b + 7 So, in base 10, 17b is b + 7 and 97b is 9b + 7.

### Step 2: Understand the Divisibility Condition

The problem states that 17b is a divisor of 97b. In base 10 terms, this means that b+7 divides 9b + 7 without leaving a remainder. Mathematically, this can be expressed as:

9b + 7 ≡ 0 (mod b + 7) or

b + 7 | 9b + 7

### Step 3: Simplify the Divisibility Condition

To find the values of b that satisfy this condition, I’ll use the concept of modular arithmetic. We have:

9b + 7 ≡ 0 (mod b + 7) This means that 9b + 7 is a multiple of b + 7. Let’s express 9b + 7 in terms of b + 7:

9b + 7 = 9(b + 7) − 63 + 7 = 9(b + 7) − 56 So,

9b + 7 = 9(b + 7) − 56

Since b + 7 divides 9b + 7, it must also divide the remainder when 9b + 7 is divided by b + 7. In this case, the remainder is −56, but since we’re dealing with divisibility, we can consider the absolute value:

b + 7 | 56

### Step 4: Find All Divisors of 56

Now, I need to find all integers b + 7 that divide 56. First, let’s list all the positive divisors of 56:

1,2,4,7,8,14,28,56

Since b > 9, b + 7 > 16. So, we’ll only consider divisors of 56 that are greater than 16. Looking at the list:

- • 1 :≤ 16 → Skip
- • 2 :≤ 16 → Skip
- • 4 :≤ 16 → Skip
- • 7 :≤ 16 → Skip
- • 8 :≤ 16 → Skip
- • 14 :≤ 16 → Skip

- • 28 :> 16 → Keep
- • 56 :> 16 → Keep

So, the possible values for b + 7 are 28 and 56.

- Step 5: Solve for b Now, let’s find the corresponding values of b:

- 1. Case 1: b + 7 = 28 b = 28 − 7 = 21
- 2. Case 2: b + 7 = 56 b = 56 − 7 = 49

- Step 6: Verify the Solutions Before finalizing, I should verify that these values of b satisfy the original condition.

- 1. For b = 21:

- • 1721 = 21 + 7 = 28
- • 9721 = 9 × 21 + 7 = 189 + 7 = 196
- • Check if 28 divides 196: 196 ÷ 28 = 7

Yes, it does.

- 2. For b = 49:

- • 1749 = 49 + 7 = 56
- • 9749 = 9 × 49 + 7 = 441 + 7 = 448
- • Check if 56 divides 448: 448 ÷ 56 = 8

Yes, it does. Both values of b satisfy the condition.

- Step 7: Calculate the Sum of All Valid b

The problem asks for the sum of all integer bases b > 9 that satisfy the condition. We found two such values: 21 and 49.

21 + 49 = 70

### Final Answer

|70|
|---|

### I License of Assets

Table 10: License of assets used in this work.

Asset License

Qwen3 models apache-2.0 DAPO-Math-17k dataset apache-2.0 code framework (verl) apache-2.0

### J Limitations

Our theoretical analysis identifies the future-KL term in the on-policy gradient of a specified KL-regularized GRPO-style objective, and our empirical results suggest that this credit-assignment signal can improve reasoning RL in the tested setting. The conclusion may need to be more generally verified on different model architectures, scales, datasets, and KL estimator choices. Due to the computation resources budget, we only conduct experiments on the Qwen series base models with a few scales ourselves. Besides, staleness or off-policy effects from other sources, such as training-inference mismatch [Yao et al., 2025a, Qi et al., 2025], could also lead to training instability. The mutual interactions among all these factors are still underexplored. We leave these valuable directions for possible future research.

