## The Reasoning–Creativity Trade-off: Toward Creativity-Driven Problem Solving

# arXiv:2601.00747v1[cs.LG]2Jan2026

##### Max Ruiz Luyten Mihaela van der Schaar

University of Cambridge

#### Abstract

State-of-the-art large language model (LLM) pipelines rely on bootstrapped reasoning loops—sampling diverse chains of thought and reinforcing the highest-scoring ones—mainly optimizing correctness. We analyze how this design choice is sensitive to the collapse of the model’s distribution over reasoning paths, slashing semantic entropy and undermining creative problemsolving. To analyze this failure, we introduce Distributional Creative Reasoning (DCR), a unified variational objective that casts training as gradient flow through probability measures on solution traces. STaR, GRPO, and DPO, as well as entropy bonuses, and other methods, all constitute special cases of the same loss. The framework delivers three core results: (i) the diversity decay theorem, describing how correctness-based objectives lead to distinct modes of diversity decay for STaR, GRPO, and DPO; (ii) designs that ensure convergence to a stable and diverse policy, effectively preventing collapse; and (iii) simple, actionable recipes to achieve this in practice. DCR thus offers the first principled recipe for LLMs that remain both correct and creative.

#### 1 Introduction

Diversity collapse in modern training loops. A canonical post-training pipeline for training reasoning LLMs includes two main stages: after supervised fine-tuning, the focus shifts to reinforcement learning (RL), which rewards the highest-scoring traces, typically based on correctness. A recurring and detrimental side-effect of this process is creative collapse: the model’s output entropy plummets, resulting in a distribution dominated by a handful of semantic templates (Mohammadi, 2024).

Creative collapse has been extensively reported across RL from human feedback (RLHF) stages (Kirk et al., 2024), when applying GRPO for mathematical reasoning (Shao et al., 2024), and during self-consistency tuning (Wang et al., 2023). In this paper, we examine why this collapse occurs and whether we can apply design choices that prevent it without sacrificing accuracy.

Why diversity matters: Creativity as a diverse portfolio for generalization. Especially for tasks outside the training distribution (OOD), creativity in problem-solving is not just a nice-to-have but rather a core requirement for high performance. A single reasoning template will inevitably fail when under novel conditions. We therefore frame creativity as the ability to maintain a diverse portfolio of high-utility reasoning strategies. This portfolio promotes OOD generalization, robust planning, and genuine discovery (Stanley and Lehman, 2020).

The central question. Our work addresses the following question:

Can we design a framework that:

- 1. explains why diversity collapse occurs,
- 2. predicts the specific mode of collapse for different algorithms, and
- 3. provides provably effective designs that guarantee a diverse portfolio of reasoning paths?

Existing literature provides incomplete answers. KL penalties preserve diversity by constraining the policy’s proximity to a base model, limiting drift at the cost of indiscriminately penalizing diverse, high-utility distant parameterizations. Sampling-based methods like Boltzmann sampling or top-k decoding also increase diversity at the cost of quality, and, more critically, they cannot recover strategies whose probabilities have vanished during training.

Our answer: Distributional Creative Reasoning. Our primary contribution is theoretical: we provide a unified framework to analyze diversity decay and a provably sufficient remedy. Since our object of study is not an individual trace, we analyze the dynamics of the entire conditional distribution pθ(π | x) over the space of solution traces. By modeling training as a gradient flow on this probability simplex, we develop a framework, Distributional Creative Reasoning (DCR), to analyze diversity decay and uncover its various sources. The DCR objective is a core component of this framework and encompasses multiple terms for utility, regularization, and a crucial, strictly concave diversity energy:

J(p) = U[p] + λD[p] − βKL KL p∥pbase .

In particular, the diversity energy D[p] is a composite functional with two distinct roles:

D[p] = αH[p] − βQ[p].

In this equation, αH[p], the Shannon entropy, promotes undiscriminated breadth, while −βQ[p] is a kernel coverage term that penalizes concentration on semantically similar traces, thereby promoting conceptual distinctiveness. This objective can recover various existing algorithms as specific instantiations, including STaR (Zelikman et al., 2022), GRPO (Shao et al., 2024), and DPO (Rafailov et al., 2023).

DCR leads to three core theoretical insights: First, it leads to the Diversity Decay Theorem, which predicts distinct modes of collapse under scalar-only objectives for the most well-known reasoning algorithms: (i) a “winner-takes-all” fixation for STaR, (ii) a neutral drift for GRPO, and (iii) a homogenization of correct strategies for DPO.

Second, we prove that incorporating the DCR diversity energy fundamentally can alter the learning dynamics, guaranteeing convergence to a unique, stable, and diverse interior equilibrium that neutralizes these collapse modes.

Third, DCR provides a set of design levers, the specific creativity kernel k(π,π′) and the coefficients α and β. We analyze the effects of their choices, resulting in a recipe for training models that are both correct and creative.

##### Contributions.

1. Unified Dynamical Lens. We introduce a variational framework based on Shahshahani gradient flow that encompasses STaR, GRPO, and DPO. Within this framework, we derive their diversity

decay dynamics under scalar objectives and finitebatch noise. We also provide a recipe for adapting the framework to new reward designs.

- 2. A Remedy for Collapse. We prove that the DCR objective, with the diversity energy functional D[p] = αH[p] − βQ[p] guarantees convergence to a high utility and (under an appropriate design) diverse policy, preventing creative collapse.
- 3. Principled Design Space and Practical Recipes. We detail how to design the creativity kernel and provide guidance on tuning DCR’s hyperparameters. We hope this will transform diversity preservation from ad-hoc heuristics to a principled design process.

Road-map. Section 2 discusses the literature on diversity collapse and related theoretical frameworks. Section 3 formally defines the DCR objective and its associated gradient flow dynamics. Section 4 presents the Diversity Decay Theorem, analyzing the distribution modes of STaR, GRPO, and DPO under scalar objectives. Section 5 proves how the DCR diversity energy reshapes the equilibrium landscape to guarantee diverse outcomes, and Section 6 discusses the design of the creativity kernel. Finally, Section 7 concludes with key insights and future directions. We empirically validate these theoretical collapse modes in Section J.

#### 2 Related Work

From reward optimisation to reasoning monoculture. A consistent empirical observation is now widely documented in the literature: when a language model is trained to maximise a single scalar reward, its solution space contracts. Early studies of RLHF showed that the resulting policy rarely develops novel strategies; instead, it reweights the trajectories present in the SFT checkpoint, leading to higher Pass@1 accuracy while leaving the underlying portfolio unchanged (Yue et al., 2025). Controlled ablations subsequently isolated the cause to the RLHF stage. Diversity, measured by entropy, type–token ratio, and embedding spread, dropped notably after RLHF, while the preceding SFT maintained it (Kirk et al., 2024). The effect is algorithm-agnostic: PPO, Expert Iteration, and GRPO all converge to the same narrow attractors, failing “to explore significantly beyond solutions already produced by SFT models” (Havrilla et al., 2024).

Beyond reasoning-based benchmarks, creative decline has also been documented in other domains. On openended story-telling and idea-generation tasks, aligned Llama-2 variants lose 3–6× token-level entropy and cluster in a few semantic basins (Mohammadi, 2024).

Treating a set of traces as a “population,‘’ Murthy et al. (2025) quantified conceptual variance, further underscoring that RLHF results in less diversity than either instruction-tuned or human populations. The overall conclusion from these works is that performance gains come, at least partly, at the cost of reducing the space of possible explanations and expressions.

First attempts at diversity-aware objectives. Several works have sought to counter this collapse by injecting ad hoc diversity terms. Entropy-regularised PPO is the most widespread heuristic, but its effect is largely to keep stochasticity indiscriminately, leaving performance gains on the table, and it does not aim to foster qualitatively distinct ideas. Novelty search and quality-diversity algorithms from evolutionary methods have also been applied to language modelling, yet the generated solutions are typically managed separately from the model, and redistillation frequently regresses gains (Havrilla et al., 2024). At the reward level, Xiao et al. (2024) identified “preference-collapse” in RLHF and proposed a Preference-Matching regulariser that adds an entropy bonus, improving minority-preference recall but with the same drawback as discussed above, and without a principled analysis of how much diversity is sufficient. In conclusion, these works demonstrate viability but leave open a unifying view that predicts when collapse will occur and the size of the required counterforce.

Theoretical lenses on collapse. Two theoretical lines are especially relevant. First, replicator dynamics from evolutionary game theory (Hofbauer and Sigmund, 1998) have been used to model reward optimisation in large populations and already hint that pure utility maximisation drives mass toward the highestfitness type. Second, information-theoretic RL reinterprets entropy bonuses as Lagrange multipliers of a KL constraint, but offers no guarantee that entropy will capture structural novelty. While these frameworks provide valuable insights, they do not offer a comprehensive analysis of creativity in LLMs.

Distributional Creative Reasoning (DCR). Our work builds on the empirical diagnostics of collapse (Yue et al., 2025; Kirk et al., 2024; Havrilla et al., 2024; Mohammadi, 2024; Murthy et al., 2025) and the first corrective steps of PM-RLHF (Xiao et al., 2024), but provides a more fundamental and unified solution, differing in three key respects:

1. Variational Framework for Diversity. We include in DCR a single concave diversity regularizers, D[p], composed of distinct terms, like entropy (Shannon entropy H[p] weighted by α) and structured novelty promotion (through a kernel k(π,π′)

in a quadratic form Q[p] weighted by β). Properly choosing the functional form of the kernel k and the relative weights α and β for these components within D[p] ensures convergence to stable, mixed-strategy ensembles, effectively counteracting collapse.

- 2. Characterization of Diversity Dynamics. Whereas prior work largely reports collapse through empirical analyses, our framework provides a dynamical systems examination (Section 4) that demonstrates how the scalar-reward objectives for STaR, GRPO, and DPO inherently lead to distinct dynamical modes that drive the evolution and erosion of diversity. This results in a deeper, mechanistic understanding of why reasoning monocultures form.
- 3. Actionable and Principled Design. DCR characterizes how diverse training objectives and diversity-regularizing terms affect the diversity dynamics. This transforms the search for diversity from heuristics to principled design. This involves selecting the kernel function and hyperparameters for the diversity functional D[p] (i.e., α and β), which become levers to shape the policy’s distribution.

#### 3 Distributional Creative Reasoning

DCR recasts LLM training as a dynamical system within the space of probability distributions over solution traces. This perspective enables the formal definition and promotion of diversity alongside correctness. This section establishes DCR’s mathematical foundations: its variational objective, the role of the diversity component, and the resultant dynamics.

##### 3.1 The Landscape of Reasoning

For a given prompt x ∈ X, an LLM generates a trace π = (t1,...,t|π|), a sequence of tokens from a finite vocabulary V up to a maximum length T. Traces can represent chains of thought, code, or action sequences. The set of all such traces, ST, is vast but finite for any fixed T and vocabulary, justifying a finite-dimensional analysis, and the choice of the counting measure on ST. An LLM’s policy p(·|x) is a probability mass function over ST, represented as a vector p in the probability simplex ∆S−1, where S := |ST|:

S

∆S−1 = p ∈ [0,1]S |

pi = 1 .

i=1

This compact, convex polytope is our domain for policy optimization. Treating the policy as a full distri-

bution, rather than focusing on single “best” traces, is crucial for modeling its diversity.

##### 3.2 The DCR Objective

During training, we optimize an objective J(p) over p ∈ ∆S−1. In DCR, we model the objective as a term representing task performance, and others for KL and diversity regularization:

J(p) = U[p] + λD[p] − βKL KL p∥pbase . The components are:

- 1. Utility (U[p]): U[p] = π∈S pected utility (e.g., correctness) of traces, encouraging high-quality outputs.

- T
- U(π)p(π) is the ex-

- 2. Diversity Energy (D[p]): Weighted by λ ≥ 0, this functional (detailed in Section 3.3) rewards policies with diversity, countering collapse.
- 3. KL-Divergence: It penalizes divergence from a reference policy pbase (e.g., the SFT checkpoint), promoting stability.

The coefficients λ,β!KL ≥ 0 tune this balance.

##### 3.3 The Diversity Energy Functional D[p]

Clearly, the core of DCR’s creativity preservation mechanism is the diversity energy functional D[p], designed to reward both probabilistic spread and semantic variation:

D[p] = αH[p] − βQ[p],

with α,β ≥ 0. Indeed, its two components serve distinct roles:

- 1. Shannon Entropy (H[p]): Promotes breadth by rewarding probability distributed across many traces, ensuring a baseline level of diversity and exploration.
- 2. Kernel Coverage (Q[p]): Q[p] = p⊤Kp =

π,π′ k(π,π′)p(π)p(π′). Here, K is the matrix of a symmetric, positive semi-definite (PSD) creativity kernel (see Section 6) measuring trace similarity. −βQ[p] thus penalizes probability concentration on similar traces, fostering semantic distinctiveness.

While entropy provides a valuable form of regularization, entropy alone is insufficient for structured creativity, as it is blind to the content of the traces. The kernel term is essential for promoting qualitatively different reasoning strategies, and the full functional D[p] is concave, which will prove to be useful:

Proposition 3.1 (Concavity of D, cf. Section A.3). If the kernel matrix K is PSD, D[p] is concave. It is strictly concave on the affine simplex if α > 0, or if β > 0 and K is strictly positive definite on the tangent subspace.

Strict concavity ensures a well-defined optimization target. In practice, incorporating into J(p) a small entropy barrier +εH[p] (ε ∈ (0,10−4] small) ensures strict concavity and that p(π) > 0 throughout optimization, guaranteeing a unique interior maximizer (cf. Section A.4, Proposition A.1).

##### 3.4 Learning Dynamics: Gradient Flow

We model policy evolution under J(p) as a gradient flow on ∆S−1, endowed with the Shahshahani metric. For tangent vectors u,v at policy p, this metric is gp(u,v) = π u(π)v(π)/p(π), and ensures the flow remains on the simplex. The DCR gradient flow is a replicator-like ODE (cf. Section A.5, Eq. (6)):

p˙t(π) = pt(π)(Ft(π) − Ep

[Ft]), where the effective trace fitness Ft(π) = δpδJ(π) p

t

is (cf. Section A.6):

t

Ft(π) = U(π) + λ(α(−1 − log pt(π)) − 2β(Kpt)π) − βKL 1 + log

pt(π) pbase(π)

.

Under the discussed regularity assumptions (finite ST, p(π) > 0 via an entropy barrier, PSD k, and bounded U(π); cf. Section A.1, (A1)–(A7)), the flow converges:

Theorem 3.1 (Global Convergence of DCR Training, cf. Section A.6, Theorem A.1). Let J(p) = J(p) + εH[p] be strictly concave on the affine simplex (e.g. if λα + ε > 0 and K is PSD) and Assumptions (A1)–(A7) hold. For any p0 ∈ int∆S−1, the Shahshahani gradient flow p˙t = ∇Sh J(pt) has a unique global solution pt, which lies on the interior of the simplex. The objective J(pt) is strictly increasing (unless pt = p⋆), and pt → p⋆ as t → ∞, where p⋆ is the unique maximizer of J(p).

Thus, DCR training with its explicit diversity energy functional provably converges to a unique policy p⋆ that balances utility, diversity, and regularization.

##### 3.5 Parametric Realization and Scalability

Parametric Realization. In practice, LLMs are function approximators. For tractability, we represent LLMs as a parameterization over policies pθ(π) via a softmax over logits θπ, so that for any target policy p⋆ ∈ int∆S−1, there exists a unique set of

(gauge-fixed) logits θ⋆ such that pθ⋆ = p⋆, making the parametric form sufficiently expressive (cf. Section B.2, Proposition B.1). To ensure numerical stability and align with the theoretical requirement of pθ(π) > δ⋆ > 0, we assume the use of projection or clipping, which constrain policies to a trimmed simplex (cf. Section B). The properties of these parameterized policies and their gradients under stochastic optimization are detailed in Section B and underpin the analysis of noise effects in Section 4.3.

Scalability. Training is performed with stochastic gradient descent on θ. The kernel coverage term Q[pθ], even though it may be intensive to fully realize, can be efficiently managed in this setting. For a mini-batch of B sampled traces, an unbiased estimate of the gradient of Q[pθ] can be computed via a U-statistic, with a computational cost of O(B2) per step. This quadratic complexity is standard in contrastive and metric learning methods. Practical kernel design strategies, including embedding-based kernels and gating mechanisms to focus diversity on correct traces, are discussed in Section 6.

#### 4 Collapse Under Scalar Objectives

While the DCR framework (Section 3) encompasses regularization terms, a typical LLM training pipeline often defaults to simpler, scalar-driven objectives. These scenarios correspond to DCR with a negligible diversity energy coefficient (λ ≈ 0) and a purely entropic diversity term with a small weight (β = 0, small λα).

This section provides a dynamical systems analysis of these “scalar objective” cases, demonstrating how they lead to distinct and predictable modes of diversity collapse. This analysis culminates in the Diversity Decay Theorem, which formally characterizes these failure modes and motivates the necessity of the full DCR objective.

##### 4.1 Scalar-Driven Dynamics: The SRCT Framework

When diversity energy is minimal, the policy p(t) evolves according to the replicator-entropy flow (formally derived in Sections D to F):

p˙π(t) =pπ(t) ϕπ(p(t)) − ϕ¯(p(t)) (1) − εpπ(t) log pπ(t) − ⟨log p(t)⟩p(t) ,

where ϕπ(p) is the trace score derived from the utility and any KL term, ϕ¯(p) is its mean, and ε ≥ 0 is the effective entropic weight (e.g., ε = εbase + λα).

The key diagnostic for diversity dynamics is the evolution of

zij(t) = log(pi(t)/pj(t)),

the log-ratio between two traces, which follows the ODE (cf. Sections D to F):

d dt

zij(t) = (ϕi(p(t)) − ϕj(p(t))) − εzij(t). (2)

This equation reveals that diversity dynamics is driven by two competing forces: selective pressure from score differences, which can negatively impact diversity, and entropic damping, which always pushes log-ratios towards zero (equalization).

##### 4.2 Deterministic Diversity Decay (Small ε)

In the pure-selection limit where ε → 0, the raw effect of scalar rewards becomes apparent. While incorrect traces are universally suppressed due to their lower utility (cf. Sections D to F), the diversity among correct traces (π ∈ C) evolves in three distinct, algorithmspecific modes:

- • STaR: “Winner-Takes-All” Collapse. For two correct traces a,b ∈ C, the score difference is ϕa(p)− ϕb(p) = (pa − pb)/ρ(t), where ρ(t) is the total mass

on correct traces. The log-ratio dynamics become d dt log p

a

pb = (pa − pb)/ρ(t) (see Section D).

Any initial random advantage for trace a (pa(0) > pb(0)) creates a positive feedback loop, causing pa/pb → ∞ and leading to a rapid, deterministic collapse onto a single dominant correct solution.

- • GRPO: “Proportional Curation” & Drift Vulnerability. For correct traces a,b ∈ C, GRPO’s score design results in ϕa(p) − ϕb(p) = 0. The logratio dynamics become dtd log p

a

pb ≈ 0 (see Section E). This preserves the initial relative probabilities of correct traces, creating a neutrally stable manifold. However, this provides no active protection for diversity, making the policy vulnerable to stochastic drift from mini-batch sampling.

- • DPO: “Equalization” & Homogenization. For two correct traces a,b ∈ C, the score difference is

ϕa(p)−ϕb(p) = gβ(log pa)−gβ(log pb), where gβ(·) is a strictly decreasing function (see Section F). Since

d dt log p

pb has the opposite sign of log p

pb , this dynamic actively drives pa/pb → 1.

a

a

DPO thus homogenizes the probability distribution across the set of preferred traces, but it does not promote targeted semantic diversity between conceptually different solutions (thereby pushing probability mass towards longer traces).

##### 4.3 Stochastic Dynamics: Fixation Under Noise

In practice, training is stochastic. The discrete mini-batch updates converge to a Wright-Fisher-type stochastic differential equation (SDE) in the diffusion limit (formally derived in Section H, Theorem H.1):

1 √

dpi = Fi(p)dt +

B

√pi dWi − pi

k

√pk dWk ,

where Fi(p) is the deterministic drift and B is the batch size. Such a random effect from batching can result in noise-induced collapse:

- • STaR: The strong “winner-takes-all” dynamic is robust, and noise results only on minor perturbations around the deterministic collapse trajectory.
- • GRPO: The neutral stability is fragile. Stochastic fluctuations introduce random selective pressure, causing the policy to drift along the manifold of correct solutions until it fixates on a corner or a small subset, leading to diversity collapse in this algorithm.
- • DPO: While equalization is the deterministic tendency, noise can break symmetries and result in convergence to a state where a subset of solutions dominates, even if they are semantically redundant.

Although a small ε ensures the policy remains in the interior (minpi(t) > δ⋆ > 0), the SDE admits a unique invariant measure π∞ (Section H, Theorem H.3). For small ε, this measure concentrates in high-utility, lowdiversity regions, as the stationary distribution is heavily influenced by the utility landscape (Section H, Section H.7). Batch noise does not increase diversity; it often accelerates fixation.

##### 4.4 Synthesis: The Diversity Decay Theorem

The analyses of both the deterministic and the stochastic dynamics converge on the conclusion that scalar-driven objectives with minimal entropic regularization are fundamentally insufficient to maintain a creative repertoire of reasoning strategies. This leads to our main diagnostic result.

Theorem 4.1 (Diversity Decay Theorem). Under scalar-objective training (DCR with λ ≈ 0 or β = 0), policies exhibit algorithm-specific modes of diversity decay among correct traces:

(i) STaR follows a “winner-takes-all” dynamics, deterministically collapsing onto a single dominant correct trace.

- (ii) GRPO evolves on a neutrally stable manifold of correct traces, leading to stochastic drift and eventual fixation on a low-diversity subset.
- (iii) DPO actively homogenizes probabilities across high-utility traces, leading to equalization instead of structured semantic diversity.

Minimal entropy (ε ≪ 1) does not prevent these outcomes and finite-batch noise can accelerate collapse.

Scope Note: This theorem characterizes the decay modes for STaR, GRPO, and DPO; it is not a general statement about every scalar-only objective.

The defined diversity-trajectories highlight the need for a more structured lever to influence the dynamics. The failure does not lie in the optimization process itself, but rather in the objective, which lacks an explicit, strong enough force that rewards structured diversity. This motivates the introduction of the DCR objective, specifically its diversity energy functional D[p], as a mechanism to counteract these modes and actively carve a rich and creative policy landscape.

#### 5 The Diversity Energy Effect on the Equilibrium Structure

Scalar objectives, as demonstrated in Section 4, lead to a degeneration in reasoning diversity. The DCR framework provides a solution by incorporating a diversity energy functional, D[p]. It reshapes the optimization landscape, altering the learning dynamics toward different equilibria: those that contain various simultaneously correct and diverse traces. This section details how DCR’s diversity regularizer achieves this shift.

##### 5.1 From Collapse to Structured Diversity

With its full objective J(p) = U[p] + λD[p] − βKL KL(p∥pbase) and a diversity weight λ > 0, DCR leverages the diversity energy

D[p] = αH[p] − βQ[p].

##### 5.2 The Dual Levers of Diversity Energy: Shaping p⋆

The specific structure of the equilibrium p⋆ with a diversity weight is shaped by the two components of the diversity energy, λD[p] = λαH[p] − λβQeff[p]. For practical applications, the quadratic term can incorporate an effective kernel keff(π,π′) := R(π)R(π′)ksem(π,π′), which gates a semantic kernel ksem with a verifier R(π) = 1π ∈ C to focus the diversity pressure only on correct traces C (see Section I, Section 6.3).

- 1. Entropic Pressure (λαH[p]): The entropic pressure promotes probabilistic breadth. It is the simplest mechanism for encouraging the equalization of probabilities among correct traces, at the cost of also promoting incorrect ones (Section I).
- 2. Kernel-Driven Structural Diversity

(−λβQeff[p]): This term penalizes p⋆ for concentrating mass on sets of correct traces that are semantically similar (as defined by ksem). It therefore actively promotes structural or semantic diversity among distinct, valid reasoning paths (Section I). Entropy alone cannot achieve this structured outcome.

- 5.3 Balancing Correctness and Structured Diversity at Equilibrium

The DCR equilibrium p⋆ is characterized by the firstorder condition Uπ − 2λβ(Keffp⋆)π − εtotal log p⋆π ≈ Constant (ignoring KL terms and gauge constants; see Section I.2). A crucial consequence for incorrect traces i ∈ I (where (Keffp⋆)i = 0 and Ui = 0) and correct traces c ∈ C (where Uc = 1) is the exact equilibrium ratio (cf. Section I.2):

1 − 2λβ(Keffp⋆)c εtotal

p⋆i p⋆c ≈ exp −

.

This identity reveals a central trade-off. To effectively suppress incorrect traces, the exponent’s numerator, 1−2λβ(Keffp⋆)c, must be substantially positive. This provides a clear heuristic for tuning the kernel weight: the kernel penalty among correct traces should not overwhelm the unit utility gain, i.e., 2λβ(Keffp⋆)c < 1.

At the same time, while a larger εtotal (from a larger λα) aids equalization among correct traces, it also increases the denominator of the exponent, thereby weakening the suppression of incorrect traces. A careful choice of λα and λβ is therefore essential to steer this trade-off and achieve a “phase” where incorrect traces are suppressed while a rich, diverse set of correct solutions thrives.

#### 6 The Creativity Kernel

The preceding sections established that DCR’s diversity energy, D[p] = αH[p] − βQ[p], is pivotal in guiding learning towards equilibria p⋆ that are diverse and stable (Section 5). While the entropy component, αH[p], provides naive probabilistic breadth, it is intrinsically “blind” to the content and structure of reasoning traces. This section explains how to build the kernel-based component −βQ[p] to provide a plausible, grounded mechanism for developing LLMs with structured, semantic diversity.

##### 6.1 Limitations of Entropic Diversity

H[p]’s utility for promoting genuine creativity is limited because it operates solely on trace probabilities, irrespective of their content or conceptual underpinnings. It cannot, for instance, distinguish a set of solutions that are mere syntactic rephrasings of a single idea from a set representing truly distinct problemsolving strategies.

Entropy alone is insufficient for structured creativity; without a mechanism to differentiate valuable novelty from trivial variation, it also preserves probability mass on incorrect traces, hindering optimization of correctness. To generate correct, structurally varied solutions, an LLM requires a mechanism that appreciates and actively promotes semantic dissimilarity rather than merely probabilistic dispersion.

##### 6.2 Sculpting Semantic Diversity

The kernel quadratic term Q[p] =

π,π′∈ST k(π,π′)p(π)p(π′) within DCR is designed to fill this critical gap. The creativity kernel k(π,π′) is a symmetric, positive semi-definite (PSD) function that quantifies the “similarity” or “redundancy” between traces π and π′. By including −βQ[p] (for β > 0) in the diversity energy, DCR explicitly penalizes policies that concentrate probability on sets of traces deemed highly similar by k.

As explored in Section I (Section I.1), an ideally engineered kernel could, in principle, sculpt a highly specific target equilibrium p⋆. Achieving this, however, would require the kernel to satisfy stringent, globally defined, and equilibrium-dependent conditions (cf. Section I, Proposition I.1). While this idealized scenario underscores the deep, direct influence of k(π,π′) on the policy structure p⋆, its practical realization is typically infeasible. This motivates the shift towards more practical, learnable semantic kernels.

##### 6.3 Practical Design of the Semantic Kernel

A more pragmatic and powerful DCR strategy, detailed in Section I (Section I.2), must utilize a learnable semantic kernel ksem(π,π′) as its foundation. This ksem should be able to capture meaningful similarities between traces. To ensure this semantic guidance is applied judiciously, DCR adopts an effective kernel, keff(π,π′):

keff(π,π′) := R(π)R(π′)ksem(π,π′), where R(π) = 1{π ∈ C} is a binary verifier for correct traces C. The kernel coverage term thus becomes Qeff[p] = c,c′∈C pcpc′ksem(c,c′). This construction focuses the diversity-promoting penalty −λβQeff[p]

exclusively on interactions among correct traces, promoting targeted diversity: it encourages the model to find diverse valid solutions, rather than rewarding “diverse ways to be wrong,” as incorrect traces do not participate in the kernel interactions that shape diversity (recall (Keffp⋆)i = 0 for i ∈ I from Section 5.3).

Practical examples of ksem can include embeddingbased kernels, where we compute an embedding for each trace (e.g., sentence-level embeddings over the full chain of thought) and apply a standard PSD kernel on those, or domain-tailored kernels, in structured tasks like mathematics, where ksem can be learned using structural proximity (e.g., from proof-step or lemma dependency graphs), so that similarity reflects shared strategy rather than just surface-level wording.

##### 6.4 Implementation and Desiderata

The kernel term can be readily integrated into standard training loops. For SGD, the gradient of Qeff[p] can be estimated with the mini-batch of B sampled traces. The quadratic nature of Qeff[p] admits a Ustatistic estimator with O(B2) per-step cost, a manageable complexity in the context of LLM training.

The efficacy of kernel-driven diversity inherently depends on the quality of the learned ksem(π,π′). Key desiderata for its design include (cf. Section 6.3): (1) Intra-Lump Coherence or high similarity for traces belonging to the same essential category or “lump” of solutions (ignoring syntactic differences); and (2) Inter-Lump Discrimination: It must assign low similarity to traces from qualitatively different correct problem-solving approaches.

#### 7 Concluding Insights

Scalar reward maximization leads to a collapse of strategic diversity. This paper has established a principled remedy: Distributional Creative Reasoning (DCR), which recasts training as a gradient flow on the policy simplex.

Our Diversity Decay Theorem offers a precise diagnosis, predicting algorithm-specific collapse modes—winner-takes-all (STaR), neutral drift (GRPO), and homogenization (DPO). The DCR framework counteracts this decay by incorporating a diversity energy functional, D[p] = αH[p]−βQ[p]. We proved this ensures convergence to a unique, stable, and interior policy p⋆.

DCR provides concrete design levers. The creativity kernel, particularly when gated to correct traces via an effective kernel keff, actively promotes novel, valid strategies. Tuning the balance between en-

tropic breadth (α) and kernel-driven diversity (β) allows practitioners to navigate the trade-off between equalization and the suppression of incorrect traces, as quantified by our equilibrium analysis.

##### 7.1 Testable Predictions

Our theoretical framework yields a set of concrete, falsifiable predictions that align with existing empirical observations:

##### 1. Algorithm-Specific Decay Modes. Under scalar-only objectives:

- • STaR exhibits winner-takes-all fixation on a single successful strategy.
- • GRPO shows neutral drift among correct traces, leading to a stochastic erosion of diversity.
- • DPO will act as an entropy equalizer, homogenizing probabilities across preferred traces.

##### 2. Kernel Sufficiency for Structured Diversity.

- • An entropy-only approach (β = 0,α > 0) preserves indiscriminate policy breadth at the cost of correctness.
- • A kernel-inclusive approach (β > 0) can not only prevent collapse but will also measurably increase the semantic diversity among correct solutions.

Acknowledgements. The authors would like to acknowledge and thank their funders, where Max Ruiz Luyten is funded by AstraZeneca. Moreover, we would like to warmly thank all the anonymous reviewers, alongside research group members of the van der Schaar lab (www.vanderschaar-lab.com), for their valuable input, comments, and suggestions as the paper was developed. We used ChatGPT and Gemini to edit and polish the text and for coding assistance.

##### References

Alexander Havrilla, Yuqing Du, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Eric Hambro, Sainbayar Sukhbaatar, and Roberta Raileanu. Teaching large language models to reason with reinforcement learning. In AI for Math Workshop @ ICML 2024, 2024. URL https:// openreview.net/forum?id=mjqoceuMnI.

J. Hofbauer and K. Sigmund. Evolutionary Games and Population Dynamics. Cambridge University Press, Cambridge, 1998. ISBN 9780521625708. URL https://pure.iiasa.ac.at/id/eprint/5442/.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. Understanding the

effects of RLHF on LLM generalisation and diversity. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=PXD3FAVHJT.

Behnam Mohammadi. Creativity has left the chat: The price of debiasing language models, 2024. URL https://arxiv.org/abs/2406.05587.

Sonia Krishna Murthy, Tomer Ullman, and Jennifer Hu. One fish, two fish, but not the whole sea: Alignment reduces language models’ conceptual diversity. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11241– 11258, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 9798-89176-189-6. URL https://aclanthology.org/ 2025.naacl-long.561/.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https:// openreview.net/forum?id=HPuSIXJaa9.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https: //arxiv.org/abs/2402.03300.

Kenneth O. Stanley and Joel Lehman. Why greatness cannot be planned: The myth of the objective, 2020. Springer, 2015 original, updated.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview. net/forum?id=1PL1NIMMrw.

Jiancong Xiao, Ziniu Li, Xingyu Xie, Emily Getzen, Cong Fang, Qi Long, and Weijie J. Su. On the algorithmic bias of aligning large language models with rlhf: Preference collapse and matching regularization, 2024. URL https://arxiv.org/abs/2405. 16455.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?, 2025. URL https://arxiv.org/abs/2504.13837.

Ethan Zelikman, Yuhuai Wu, and Noah D. Goodman. Star: Bootstrapping reasoning with reasoning. arXiv preprint arXiv:2203.14465, 2022.

#### A Mathematical Foundations and Problem Formalism

This appendix fixes notation and geometric conventions on the simplex, records canonical inequalities and curvature facts for the objective slices (entropy/KL/kernel), develops the Shahshahani gradient representation, and derives global properties of the induced gradient flows (Lyapunov identity, log–ratio contraction, time–uniform floors/caps, and exponential convergence). It also states a generic Barrier–Dominance (BD) calculus for forward invariance of trimmed domains.

##### A.1 Preliminaries and Standing Assumptions

Scope & conventions. All logarithms are natural; 0log 0 := 0. The indicator is 1{·}, and ⟨u,v⟩ is the Euclidean inner product. We write a ≲ b to mean a ≤ C b for an absolute constant C; any parameter dependence is displayed as C(·). Sums over traces are with respect to the counting measure on the finite set ST.

Symbol Meaning x ∈ X Fixed prompt / task instance π ∈ ST Trace (finite token sequence, length ≤ T) ST Trace set up to length T; S := |ST| p(π) Policy mass on π (probability on ST) ∆S−1 Probability simplex on ST H[p] Shannon entropy, − π p(π)log p(π)

DKL(p∥q) Kullback–Leibler divergence, π p(π)log pq((ππ)) k(π,π′) Symmetric positive semidefinite kernel on ST K = [k(π,π′)] Kernel matrix in RS×S

D[p] Diversity: α H[p] − β p⊤Kp

##### Standing assumptions.

- (A1) Finite trace space. ST is finite for a fixed horizon T < ∞; policies are p ∈ ∆S−1 ⊂ RS.
- (A2) Interior vs. trimmed domain. Variational derivatives and Shahshahani gradients are taken on int∆S−1 =

{p : minπ p(π) > 0}. When a floor is operative, we work on the trimmed simplex ∆Sδ−1 := {p ∈ ∆S−1 : pi ≥ δ ∀i}, nonempty iff δ ≤ 1/S.

- (A3) Entropy/KL domains. H[p] and (when present) DKL(p∥pbase) are defined on the closed simplex; all variational derivatives are computed on int∆S−1. Adding +εH (ε ≥ 0) is permitted.
- (A4) Kernel regularity and strictness on T. K = K⊤ ⪰ 0. Write T := {1}⊥ and ΠT := I − S1 11⊤. The quadratic slice −p⊤Kp is strictly concave along feasible directions iff kerK ∩ T = {0} (equivalently, ΠTKΠT ≻ 0 on T).

- (A5) Bounded utility. |U(π)| ≤ Umax < ∞ on ST whenever U[p] = π U(π)p(π) is used.
- (A6) Nonnegative coefficients. α,β,βKL,λ,ε ≥ 0 unless noted.
- (A7) Base-policy support (for KL). If DKL(p∥pbase) is present, assume pbase(π) ≥ pbase,min > 0 for all π.

Norm conventions. For vectors: ∥·∥1, ∥·∥2, ∥·∥∞. For A ∈ RS×S: ∥A∥2→2 (spectral norm) and ∥A∥∞→∞ := maxi j |Aij|.

- A.2 Spaces and Simplex Geometry

- A.2.1 Trace space, simplex, tangent. Fix vocabulary V and horizon T ∈ N.

ST = {(t1,...,tℓ) : 1 ≤ ℓ ≤ T, ti ∈ V}, S := |ST| < ∞.

Policies are p ∈ ∆S−1 := {p ∈ [0,1]S : ⟨1,p⟩ = 1}. On int∆S−1, feasible directions lie in the affine tangent

T = Tp∆S−1 = {v ∈ RS : ⟨1,v⟩ = 0} = {1}⊥, which does not depend on p.

##### A.2.2 Floors: policy vs. effective.

A chosen floor δ ∈ (0,1/S] defines the trimmed simplex ∆Sδ−1 = {p ∈ ∆S−1 : pi ≥ δ ∀i}. Algorithmic clip–renormalize with threshold δ⋆ ∈ (0,1] induces an effective floor

δ⋆ 1 + (S − 1)δ⋆

δ⋆

∈

, δ⋆ ,

δeff(p) =

S j=1 max{pj,δ⋆}

since the denominator ranges from 1 to 1 + (S − 1)δ⋆ (max at a simplex vertex). The exact clip–renormalize map and logit lift are given in Section B.

##### A.2.3 Canonical inequalities.

- Lemma A.1 (Mean–log bounds and entropic Lipschitzness). Let p ∈ ∆S−1 and ⟨log p⟩ := i pi log pi.

- 1. (Mean–log bounds) For all p ∈ ∆S−1, −log S ≤ ⟨log p⟩ ≤ 0.
- 2. (Entropic Lipschitz on ∆Sδ−1) Fix δ ∈ (0,1/S] and Λ(δ) := 1 + log(1/δ). For all p,q ∈ ∆Sδ−1,

1 δ ∥p − q∥2, ∇H(r) = −(1 + log r), (3)

∥∇H(p) − ∇H(q)∥2 ≤

√

p ⊙ (log p − ⟨log p⟩) − q ⊙ (log q − ⟨log q⟩) 2 ≤ Λ(δ)(2 +

S)∥p − q∥2. (4)

Proof. (1) Upper bound: each log pi ≤ 0. Lower bound: H(p) is maximized at the uniform u = (1/S)1 with H(u) = log S.

(2) For (3), ∇2H(r) = −diag(1/ri) on int∆S−1 so ∥∇2H(r)∥2→2 ≤ 1/δ on ∆Sδ−1, and the mean–value theorem applies.

For (4), set E(r) := r ⊙ (log r − ⟨log r⟩) and G(r) := r ⊙ log r. Then DG(r)[h] = h ⊙ (1 + log r), hence ∥DG(r)∥2→2 ≤ Λ(δ). For B(r) := ⟨log r⟩r,

DB(r)[h] = (1 + log r) ⊙ h r + ⟨log r⟩h,

√

√

S, ∥r∥2 ≤ 1, and |⟨log r⟩| ≤ Λ(δ) − 1 on ∆Sδ−1. Therefore ∥DE(r)∥2→2 ≤ Λ(δ)(2 +

S + (Λ(δ) − 1) because ∥1 + log r∥2 ≤ Λ(δ)

so ∥DB(r)∥2→2 ≤ Λ(δ)

√

S) and the mean–value theorem yields (4).

| |
|---|

- A.3 Functionals: Entropy, KL, Kernel, and Diversity

- A.3.1 Entropy and KL calculus. On int∆S−1,

H[p] = −

DKL(p∥q) =

i

pi log pi,

i

pi qi

pi log

,

δH δpi

= −(1 + log pi), ∇2H = −diag(1/pi),

δ δpi

DKL(p∥q) = 1 + log

pi qi

, ∇2DKL(p∥q) = diag(1/pi),

with qi > 0 for KL. Both extend continuously to the closed simplex (using 0log 0 := 0).

- A.3.2 Kernel quadratic form. For K = K⊤ ⪰ 0, set Q[p] = p⊤Kp. Then

∇(−Q)(p) = −2Kp, ∇2(−Q) = −2K ⪯ 0, so −Q is concave on RS and 2∥K∥2→2-Lipschitz in gradient. Along any feasible direction v ∈ T, d

2

dt2 [−Q(p0 + tv)]|t=0 = −2v⊤Kv, hence strict concavity on feasible directions iff kerK ∩ T = {0} (equivalently ΠTKΠT ≻ 0 on T).

- A.3.3 Diversity functional. Let D[p] = αH[p]−βQ[p] with α,β ≥ 0. Writing κT := λmin (ΠTKΠT)|T ≥ 0, for all p ∈ int∆S−1 and v ∈ T,

⟨∇2D[p]v,v⟩ = α⟨∇2H[p]v,v⟩ − 2β v⊤Kv ≤ − α + 2βκT ∥v∥22.

Thus D is concave, α–strongly concave on the affine simplex if α > 0, and strictly concave along feasible directions when α = 0, β > 0, and κT > 0.

- A.4 Barriers and Interiority

- A.4.1 Entropy/KL barriers exclude boundary maximizers.

- Proposition A.1 (Interior maximizers). Let J be concave on ∆S−1.

- 1. For any ε > 0, J(p) := J(p) + εH[p] is strictly concave on int∆S−1 and attains its unique maximum at an interior point.
- 2. If pbase has full support (A7), then for any βKL > 0, J(p) − βKLDKL(p∥pbase) cannot be maximized on the boundary ∂∆S−1.

Proof. (1) On int∆S−1, ∇2H = −diag(1/p) ≺ 0, so J is strictly concave. At a boundary point with some pi = 0, the directional derivative of −pi log pi = −tlog t along ei diverges to +∞ as t ↓ 0, excluding boundary maxima.

(2) With pi = 0, for p(t) = (1 − t)p + tei, dtd tlog p t

base,i t↓0 = log t + 1 − log pbase,i → −∞, so the derivative of −βKLDKL(·∥pbase) is +∞ inward. Boundary maxima are impossible.

| |
|---|

- A.4.2 No finite–time boundary hitting under bounded fitness.

Lemma A.2 (Bounded fitness implies interiority). Consider the replicator ODE p˙i = pi Gi(p) − Ep[G] with a continuous field G satisfying supp,i |Gi(p)| ≤ M < ∞. If p(0) ∈ int∆S−1, then for all t ≥ 0 and all i,

e−2Mtpi(0) ≤ pi(t) ≤ e2Mtpi(0), in particular pi(t) > 0 for all t.

Proof. dtd log pi = Gi(p) − Ep[G] is bounded in [−2M,2M]; integrate. Remark A.1 (Applicability). For Gi(p) = U(i) − 2λβ (Kp)i, (A5) and finiteness of ∥K∥∞→∞ imply |(Kp)i| ≤ ∥K∥∞→∞ and hence a uniform M < ∞.

| |
|---|

- A.5 Shahshahani Geometry and Gradient Representation

- A.5.1 Metric and replicator form. On int∆S−1, the Shahshahani metric on T = {1}⊥ is

S

uivi pi

gp(u,v) :=

i=1

(u,v ∈ T). (5)

For J ∈ C1, the Shahshahani gradient is the unique w ∈ T with gp(w,v) = DJ[p]· v for all v ∈ T, yielding the classical replicator form

|p˙i = (∇ShJ)i = pi δp δJ<br><br>i<br><br>− Ep[δJδp] , Ep[ξ] :=<br><br>i<br><br>piξi.|
|---|

(6)

Mass is conserved ( i p˙i = 0). The dynamics are invariant under adding any scalar field a(p) to the scores δJ/δp (gauge invariance), since centering by Ep[·] removes it.

##### A.5.2 Integrability of replicator fields.

- Proposition A.2 (Integrability on the simplex). Let G ∈ C1(int∆S−1;RS) and consider p˙i = pi Gi(p)−Ep[G] . The following are equivalent; they hold iff there exists J ∈ C1 with p˙ = ∇ShJ:

(Gj − Gk) for all i,j ̸= k. (PJ) Projected–Jacobian symmetry: there exists a scalar field a(p) such that ΠTD G − a1 ΠT is symmetric on T for all p.

(Gi − Gk) = ∂p

(AC) Anchored cross–partials: for some (hence any) anchor k, ∂p

j

i

In that case, J is unique up to an additive constant and gauge a(p)1.

Proof sketch. Work on the chart q = (p1,...,pS−1), pS = 1 − Si=1−1 qi. The T-restricted 1–form is ωT = S−1 i=1 (Gi −GS)dqi. Condition (AC) is the closedness of ωT; on the simply connected domain, Poincar´e’s lemma

J = Gi − GS. Setting a(p) := GS(p) recovers the replicator field. (PJ) is the coordinate–free restatement on T.

yields exactness, giving J with ∂q

i

| |
|---|

Instantiation. For J = U + λD − βKLDKL((∥·)∥pbase) + εH, the pointwise variational derivative is

δJ δpi

= Ui − 2λβ (Kp)i − (λα + ε)(1 + log pi) − βKL 1 + log p

pbase,i , and the flow is p˙i = pi Fi(p) − Ep[F] .

Fi(p) :=

i

- A.6 Gradient–Flow Dynamics and Convergence

- A.6.1 ODEs and barrier strength.

Let

J(p) = U[p] + λD[p] − βKLDKL(p∥pbase), J(p) = J(p) + εH[p], and define the aggregate barrier strength

|A := ε + λα + βKL.|
|---|

Then the J–flow is

p˙i = pi Fi(p) − Ep[ F] , Fi(p) = Fi(p) − ε(1 + log pi), (7) with mass conservation i p˙i = 0.

- A.6.2 Lyapunov identity (with boundary continuity).

- Lemma A.3 (Strict Lyapunov identity). Along any solution t  → pt ∈ int∆S−1 of (7),

2

d dt

(pt) − Ep

pt(i) δp δ J

[δδp J ]

t ∇Sh J(pt), ∇Sh J(pt) =

≥ 0, (8)

J(pt) = gp

t

i

i

with equality iff ∇Sh J(pt) = 0. Moreover, the right–hand side extends continuously to the closed simplex: p(log p)2 → 0 as p ↓ 0 and (A7) yields the same for p log p p

2.

base

##### A.6.3 Log–ratio contraction; time–uniform floor and cap.

- Lemma A.4 (Log–ratio contraction and uniform bounds). Assume (A1), (A4), (A5), (A7) and A > 0. For

zij(t) := logp

i(t) pj(t),

z˙ij(t) = −Azij(t) + cij(pt), |cij(p)| ≤ B, (9) where

pbase,max pbase,min

B := 2Umax + 4λβ ∥K∥∞→∞ + βKL log

.

Hence |zij(t)| ≤ |zij(0)|e−At + BA(1 − e−At) ≤ M, and for all t ≥ 0 and all i,

|1 S eM ≤ pi(t) ≤<br><br>eM S<br><br>.|
|---|

(10)

Proof. Subtract the log–dynamics dtd log pi = Fi−Ep[ F] to get z˙ij = Fi− Fj. The (log p)–terms contribute −Azij, while the remaining terms are bounded by B. Solve the linear ODE and use the standard “max–coordinate”

argument to obtain (10).

| |
|---|

##### A.6.4 Global convergence with explicit rate.

- Theorem A.1 (Well–posedness, unique equilibrium, exponential rate). Assume (A1), (A4), (A5), (A7) and

- A > 0. For any p0 ∈ int∆S−1, the flow (7) admits a unique global solution staying in the compact trimmed simplex ∆Sδ−1 with δ = 1/(SeM) from Lemma A.4. On the affine simplex,

∇2 J(p) = A∇2H(p) − 2λβK = −Adiag(1/p) − 2λβK ⪯ −AI, so J is A–strongly concave and has a unique maximizer p⋆ ∈ int∆S−1. Moreover,

d dt

J(p⋆) − J(pt) ≤ −2Aδ J(p⋆) − J(pt) , and

|∥pt − p⋆∥2 ≤ A2 J(p⋆) − J(p0)<br><br>=:C<br><br>exp(−Aδ t) .|
|---|

Proof sketch. Lyapunov identity and Lemma A.4 give global existence and a uniform floor δ. Strong concavity on the affine simplex yields the Polyak– Lojasiewicz inequality ∥ΠT∇ J(p)∥22 ≥ 2A J(p⋆) − J(p) . Since gp(w,w) ≥ δ∥ΠTw∥22 on ∆Sδ−1, (8) implies exponential decay of the suboptimality gap and then of ∥pt − p⋆∥2 by strong concavity.

| |
|---|

Remarks. (i) If A = 0 (no entropy/KL barrier), the contraction term in (9) vanishes; neither the time–uniform floor/cap (10) nor exponential convergence follow by this route (uniqueness may still hold if ΠTKΠT ≻ 0). (ii) For S = 1, statements are trivial. (iii) The bound for |(Kp)i − (Kp)j| can be sharpened (e.g., by 2∥K∥2→2) without changing the argument.

- A.7 Special Case: Replicator Flow with Single–Site Scores Consider p˙i = pi Gi(pi) − Ep[G] where Gi depends only on pi.

- Proposition A.3 (Lyapunov structure). Define L(p) = Si=1 Ψi(pi) with Ψ′i(s) = Gi(s). Then d

pi Gi(pi) − Ep[G] 2 ≥ 0,

dtL(p(t)) = Varp(t) G(p(t)) =

i

with equality iff Gi(pi) is constant across the support. If, in addition, all Gi ≡ g are identical and strictly monotone, the unique interior equilibrium is uniform on its support. In general, with distinct strictly monotone Gi, the interior equilibrium need not be uniform.

- A.8 Barrier–Dominance (BD) Scope. Consider the deterministic replicator field endowed with an entropy slice

p˙i = pi ϕi(p) − ϕ¯(p) + εBD pi ⟨log p⟩ − log pi , ϕ¯(p) :=

j

pj ϕj(p), (11)

with εBD ≥ 0 and a selection score field ϕ : ∆S−1 → RS. Norms are as in §A.1.

- A.8.1 Entropy face gap LS(δ). Definition A.1 (Entropy face gap). For S ≥ 2 and δ ∈ (0,1/S],

LS(δ) := inf ⟨log p⟩ − log δ : p ∈ ∆S−1, ∃i s.t. pi = δ .

Lemma A.5 (Closed form and properties). For all S ≥ 2 and δ ∈ (0,1/S],

LS(δ) = (1 − δ) log

1 − δ (S − 1)δ

,

with LS(δ) ≥ 0 (equality iff δ = 1/S); LS is strictly decreasing in δ and, for fixed δ, strictly decreasing in S.

Proof. Fix the face {pi = δ}. Jensen for the convex x  → xlog x implies the minimum when the remaining mass 1 − δ is split equally: pj = (1 − δ)/(S − 1) for j ̸= i.

| |
|---|

Lemma A.6 (Two–sided bounds). For all S ≥ 2 and δ ∈ (0,1/S],

log

1 (S − 1)δ − 1 + log (S−11)δ δ

lower

≤ LS(δ) ≤ log

1 (S − 1)δ

upper

.

- A.8.2 Deterministic BD conditions.

Assume ϕ is bounded on the operative domain: Mϕ,∞ := supp ∥ϕ(p)∥∞ < ∞, Mϕ,2 := supp ∥ϕ(p)∥2 < ∞.

- Proposition A.4 (Forward invariance of ∆Sδ−1). For the flow (11), fix δ ∈ (0,1/S]. If either

(ℓ∞) εBD LS(δ) ≥ 2Mϕ,∞, (ℓ2) εBD LS(δ) ≥ 2Mϕ,2,

then ∆Sδ−1 is forward invariant: any solution with p(0) ∈ ∆Sδ−1 satisfies p(t) ∈ ∆Sδ−1 for all t ≥ 0.

Proof. On the face {pi = δ},

p˙i pi

= ϕi − ϕ¯

###### + εBD (⟨log p⟩ − log δ)

.

≥−2Mϕ,∞ or ≥−2Mϕ,2

≥LS(δ)

Hence the outward normal component is nonnegative on every face under either condition. By Nagumo’s tangency criterion (viability theory), ∆Sδ−1 is forward invariant.

| |
|---|

- Remark A.2 (Tightness and scaling). The factor 2 in the ℓ∞ condition is tight without further structure (place all remaining mass on a single coordinate and choose ϕ with opposite signs on the two active coordinates). For

small δ, LS(δ) ≍ log 1/((S − 1)δ) and degrades monotonically with S; at δ = 1/S, LS(δ) = 0 and the trimmed set collapses to the uniform point.

#### B Parametric (Logit-Space) Geometry and Propagation Bounds

- B.1 Introduction and Notation

This appendix records the deterministic, parametric (logit-space) geometry used throughout: the soft-max map, its Jacobian, conditioning, Lipschitz constants, the clip–renormalize/logit-lift construction, composite smoothness constants, and second-order remainders. Stochastic topics (e.g., clipping bias, mini-batch covariance) are deferred to Section H.

Notation. Let 1 := (1,...,1)⊤. The simplex and its relative interior are

∆S−1 := {p ∈ [0,1]S : ⟨1,p⟩ = 1}, ri(∆S−1) = {p ∈ ∆S−1 : pi > 0 ∀i}. The centered logit space (gauge slice) and the tangent space are

Θ := {θ ∈ RS : ⟨1,θ⟩ = 0}, T := 1⊥, ΠT := I − S1 11⊤, C := ΠT. Define the soft-max pθ := softmax(θ) := eθ/⟨1,eθ⟩ ∈ ∆S−1, and its Jacobian

Jθ := ∇θpθ = diag(pθ) − pθp⊤θ . Appendix C writes the same covariance-form matrix as S(p) := diag(p) − pp⊤; we use the identification

|Jθ = S(pθ)|
|---|

(12)

to keep notation uniform across appendices.

- B.2 Soft-max Map: Gauge, Inverse, and Log-ratio

- Lemma B.1 (Translation invariance). For any θ ∈ RS and c ∈ R, softmax(θ + c1) = softmax(θ).

- Proposition B.1 (Real-analytic diffeomorphism). The restriction softmax : Θ → ri(∆S−1) is a real-analytic diffeomorphism with inverse

G : ri(∆S−1) → Θ, G(p) := C log p = log p − S1 ⟨1,log p⟩1.

Proof. For p ∈ ri(∆S−1), writing log p := S1 ⟨1,log p⟩,

softmax(G(p))i =

exp log pi − log p j exp log pj − log p

= pi.

Conversely, for θ ∈ Θ,

G(softmax(θ))i = log

eθ

- i

- j eθj

− S1

k

log

eθ

k

j eθj

= θi.

Analyticity follows from analyticity of exp and log and linearity of C.

| |
|---|

- Corollary B.1 (Log-ratios & gauge uniqueness). If p = softmax(θ) with θ ∈ Θ, then θi − θj = log(pi/pj) for all i ̸= j. If softmax(θ) = softmax(θ′), then θ − θ′ = c1; on Θ this forces θ = θ′.

Remark B.1 (Edge case S = 1). If S = 1, then Θ = {0}, ∆0 = {1}, and softmax(0) = 1.

- B.3 Geometry and Conditioning of the Soft-max Jacobian

Basic differential. For any θ,

|Jθ = diag(pθ) − pθp⊤θ = S(pθ).|
|---|

(13)

- Lemma B.2 (Kernel, rank, variance form). Let p = pθ. Then kerJθ = span{1} and rank(Jθ) = S−1. Moreover, for v ∈ T,

v⊤Jθv =

i

pivi2 −

i

pivi

2

= 12

i,j

pipj (vi − vj)2 = Vari∼p(vi) ≥ 0,

with equality iff v = 0.

- Corollary B.2 (Loewner sandwich on T; global operator norm). If pmin := mini pθ(i) > 0, then

|pmin I ≼ Jθ |T ≼ 12 I<br><br>|
|---|

, ∥Jθ∥op ≤ 12.

Proof. Upper bound: for v ∈ T, Popoviciu’s inequality yields Varp(vi) ≤ 14(maxv − minv)2 ≤ 12∥v∥22. Lower bound: write p = pmin1 + q with q ≥ 0, i qi = 1 − Spmin. Then for v ∈ T, v⊤Jθv − pmin∥v∥22 = i qivi2 − ( i qivi)2 ≥ 0 (Cauchy–Schwarz with weights q). Since JθT ⊆ T and Jθ1 = 0, the global ∥Jθ∥op equals the supremum on T.

| |
|---|

- Remark B.2 (Tightness). The upper bound 21 is attained for S = 2 at p = (1/2,1/2); the lower bound pmin is attained at p = S1 1, where Jθ |T= (1/S)I.

Lemma B.3 (Per-coordinate bound). For every θ and k ∈ {1,...,S},

|∥∂θ<br><br>k<br><br>Jθ∥op ≤ 3√13<br><br>|
|---|

and the constant 3√13 is optimal (already for S = 2).

Proof sketch. WLOG k = 1. With a := p1 ∈ (0,1) and b ∈ RS≥−01, b = 1 − a,

∂θ

1

Jθ = aN(a,b), N(a,b) =

(1 − a)(1 − 2a) −(1 − 2a)b⊤ −(1 − 2a)b 2bb⊤ − diag(b)

.

The Rayleigh quotient in b is convex on the simplex (Hessian 4yy⊤ ⪰ 0), thus maximized at a vertex b = (1−a)ej. In the {e1,ej} subspace the spectral norm equals 2a(1 − a)|1 − 2a|, whose maximum over a ∈ [0,1] is 1/(3√3) at a = 12 ± 2√13.

| |
|---|

Theorem B.1 (Global Lipschitz continuity of θ  → Jθ). For all θ1,θ2 ∈ Θ,

|∥Jθ<br><br>2 − Jθ<br><br>1∥op ≤ 3√13∥θ2 − θ1∥1 ≤<br><br>√<br><br>S<br><br>3√3∥θ2 − θ1∥2 ≤ 3√S3∥θ2 − θ1∥∞.<br><br>|
|---|

Proof. Parameterize θ(τ) = θ1 + τ(θ2 − θ1). By the fundamental theorem of calculus and Lemma B.3,

∥Jθ

2 − Jθ

1∥op ≤

1

0

S

k=1

|∆θk|∥∂θ

k

Jθ(τ)∥op dτ ≤ 3√13∥∆θ∥1.

The ℓ2,ℓ∞ versions follow from norm monotonicity.

| |
|---|

- Remark B.3 (Dimension-free lower bounds). Along θ(t) = (t,−t,0,...,0) one has ∥dJθ(t)/dt∥op = 2/(3√3) at the extremal p while ∥θ˙(t)∥1 = 2, giving optimality in the ℓ1 domain norm. Restricting to the same two-coordinate

√2/(3√3) and L(J∞) ≥ 2/(3√3).

subspace gives L(2)J ≥

Boundary behavior. As pmin ↓ 0 (e.g., pθ → ei), Jθ = S(pθ) → 0. Then λmin(Jθ |T) ↓ 0 while λmax(Jθ |T) ≤ 12, so κ(Jθ |T) ≤ (1/2)/pmin → ∞.

- B.4 Clip–Renormalize and the Logit Lift Definition and effective floor. Fix δ⋆ ∈ (0,1). Define the clip–renormalize operator

max(p,δ⋆) ∥max(p,δ⋆)∥1

Cδ⋆

, (max(p,δ⋆))i := max{pi,δ⋆}.

(p) :=

(p), then qi ≥ δmin := δ⋆/(1 + (S − 1)δ⋆), and this lower bound is sharp whenever clipping occurs. Given δ ∈ (0,1/S),

If q = Cδ⋆

|δ⋆ =<br><br>δ 1 − (S − 1)δ<br><br>=⇒ min<br><br>i<br><br>Cδ⋆<br><br>(p) i ≥ δ ∀p.<br><br>|
|---|

##### Logit lift and normalization cancellation. Define the logit lift

P : Θ → Θ, P(θ) := C log max(pθ,δ⋆) . If p′ = max(pθ,δ⋆) and q := p′/∥p′∥1, then P(θ) = C log q and

|softmax(P(θ)) = q = Cδ⋆<br><br>(pθ).|
|---|

(14)

- Proposition B.2 (Global Lipschitz of P and softmax◦P). For all θ,ϑ ∈ Θ,

∥θ − ϑ∥2, ∥softmax(P(θ)) − softmax(P(ϑ))∥2 ≤ 41δ

∥P(θ) − P(ϑ)∥2 ≤ 21δ

∥θ − ϑ∥2.

⋆

⋆

Proof. ∥pθ − pϑ∥2 ≤ 12∥θ − ϑ∥2 (MVT + Corollary B.2); clipping is 1-Lipschitz in ℓ2; log is 1/δ⋆-Lipschitz on [δ⋆,1]; C is nonexpansive; softmax has Jacobian norm ≤ 12.

| |
|---|

##### Differentials (a.e.). Since P is piecewise C1,

|∥DP(θ)∥op ≤ 21δ<br><br>⋆<br><br>for a.e. θ, ∥D(softmax◦P)(θ)∥op ≤ 41δ<br><br>⋆<br><br>.|
|---|

(15)

0∥∞ ≤ 21ε, hence no coordinate is clipped: P(θ) = C log pθ = θ.

(i) ≥ δ⋆+ε and ∥θ−θ0∥2 ≤ ε, then ∥pθ−pθ

##### Local no-clip criterion. If mini pθ

0

##### Post-clipping deviation with a known floor. If mini pθ(i) ≥ δ > 0 and c := |{i : pθ(i) < δ⋆}|, then

|∥P(θ) − θ∥2 ≤<br><br>δ⋆ δ<br><br>√c ≤<br><br>δ⋆ δ<br><br>√<br><br>S.|
|---|

(16)

Smooth vs. hard clip; Lipschitz of DP. Let LDP denote a Lipschitz constant of θ  → DP(θ) in operator norm. Two regimes are useful:

- • Hard-clip, kink-free segment (active set fixed):

|LDP ≤<br><br>1 4δ⋆2<br><br>+<br><br>√<br><br>S 3√3 ·<br><br>1 δ⋆<br><br>.|
|---|

- (17)

• Smooth clip surrogate χτ: if 0 ≤ χ′τ ≤ 1 and Lip(χ′τ) ≤ cτ, then

|LDP ≤<br><br>1 + cτ 4δ⋆2<br><br>+<br><br>cτ 2δ⋆<br><br>+<br><br>√<br><br>S 3√3 ·<br><br>1 δ⋆<br><br>.|
|---|

- (18)

##### B.5 Composite Smoothness for Φ(θ) := J(softmax(P(θ)))

(pθ) lies in the rectangle [δmin,1]S, δmin = δ⋆/(1 + (S − 1)δ⋆). Assumption (A) (Euclidean norms throughout): for all p,q ∈ [δmin,1]S,

Domain and Assumption (A). By (14), p(θ) := softmax(P(θ)) = Cδ⋆

∥∇pJ(p) − ∇pJ(q)∥2 ≤ Lp∥p − q∥2, sup

∥∇pJ(p)∥2 ≤ Gp < ∞.

p∈[δmin,1]S

Chain pieces and uniform bounds. Let ϕ(θ) := P(θ), p(θ) := softmax(ϕ(θ)), and

B(θ) := Dθp(θ) = Jϕ(θ) DP(θ). Using (15) and Corollary B.2, uniformly in θ,

|∥DP(θ)∥op ≤ 21δ<br><br>⋆<br><br>, ∥Jϕ(θ)∥op ≤ 21, ∥B(θ)∥op ≤ 41δ<br><br>⋆<br><br>.|
|---|

- Lemma B.4 (Lipschitz of B(θ)). For all θ1,θ2 ∈ Θ,

|∥B(θ2) − B(θ1)∥op ≤<br><br>√<br><br>S 12√3 ·<br><br>1 δ⋆2<br><br>+ 12 LDP ∥θ2 − θ1∥2,<br><br>|
|---|

with LDP as in (17)–(18).

- (19)

Also, Proposition B.2 gives

|∥p(θ2) − p(θ1)∥2 ≤ 41δ<br><br>⋆<br><br>∥θ2 − θ1∥2.|
|---|

- (20)

Proof. Split B(θ2) − B(θ1) = (Jϕ

(DP(θ2) − DP(θ1)). First term: by Theorem B.1 and Proposition B.2,

2 − Jϕ

)DP(θ2) + Jϕ

1

1

√

√

1∥op ≤ 3√13∥ϕ2 − ϕ1∥1 ≤

S

S

6√3δ⋆∥∆θ∥2, then multiply by ∥DP(θ2)∥op ≤ 21δ

∥Jϕ

2 − Jϕ

3√3∥ϕ2 − ϕ1∥2 ≤

1∥op ≤ 21 and ∥DP(θ2) − DP(θ1)∥op ≤ LDP∥∆θ∥2.

. Second term: ∥Jϕ

| |
|---|

⋆

- Theorem B.2 (Composite Lipschitz constant for ∇θΦ). Under Assumption (A),

|∥∇θΦ(θ2) − ∇θΦ(θ1)∥2 ≤ Lθ ∥θ2 − θ1∥2, Lθ ≤<br><br>Lp 16δ⋆2<br><br>+ Gp<br><br>√<br><br>S 12√3δ⋆2<br><br>+ 21LDP .<br><br>|
|---|

Proof. ∇θΦ(θ) = B(θ)⊤∇pJ(p(θ)). Subtract and add:

∥∆∇θΦ∥2 ≤ ∥B2 − B1∥op ∥∇pJ(p1)∥2 + ∥B2∥op ∥∇pJ(p2) − ∇pJ(p1)∥2. Use Lemma B.4 and ∥∇pJ(p1)∥2 ≤ Gp for the first term. For the second, apply (19) and (20). Step-size guidance. A conservative choice for gradient methods on Φ is

|η ≤ 1/Lθ.|
|---|

A common heuristic (ignoring Gp-driven variation of B) is η ≈ 16δ⋆2/Lp.

##### B.6 Quadratic Approximation and Hessian Suprema Second derivatives. For i,k,ℓ ∈ {1,...,S},

|∂θ<br><br>ℓ<br><br>∂θ<br><br>k<br><br>pθ(i) = pθ(i) (δiℓ − pθ(ℓ))(δik − pθ(k)) − pθ(k) δkℓ − pθ(ℓ) .|
|---|

Let Hkℓ(θ) ∈ RS collect the components ∂θ

∂θ

ℓ

pθ(i), and H(θ)[u,v] := k,ℓ ukvℓ Hkℓ(θ).

k

| |
|---|

(21)

- Theorem B.3 (ℓ2 and ℓ1 suprema). For every S ≥ 2,

|sup<br><br>θ,k,ℓ<br><br>∥Hkℓ(θ)∥2 = √154, sup θ,k,ℓ<br><br>∥Hkℓ(θ)∥1 = 3√13.<br><br>|
|---|

Both are attained for S = 2, and are strict suprema for S > 2 (approached by concentrating residual mass).

Proof sketch. Using (21), for fixed (k,ℓ) the Rayleigh quotient in the residual mass is convex over the simplex, hence maximized at vertices (mass on one coordinate). Reducing to 2×2 or 3×3 blocks yields the stated optima,

attained at p = (12 ± 2√13, 12 ∓ 2√13,0,...). Second-order expansion and remainders. For any θ,g ∈ RS and η ≥ 0,

| |
|---|

|pθ+ηg = pθ + ηJθg + η2<br><br>1<br><br>0<br><br>(1 − τ)H(θ + τηg)[g,g] dτ.|
|---|

- (22)

Consequently,

|∥Rθ,η∥1 ≤ η<br><br>2<br><br>6√3 ∥g∥21,<br><br>∥Rθ,η∥2 ≤ η<br><br>2<br><br>2√54 ∥g∥21, ∥Rθ,η∥∞ ≤ η<br><br>2<br><br>6√3 ∥g∥21, ∥Rθ,η∥2 ≤ η<br><br>2<br><br>6√3<br><br>√s∥g∥22 (s := ∥g∥0).<br><br>|
|---|

- (23)

√s∥g∥2.

The last bound uses Theorem B.1 to control ∥∇Jθ+sg[g]∥op and ∥g∥1 ≤

δ-interior refinements. Assume the path τ  → pθ+τηg stays in the trimmed simplex

∆Sδ−1 := {p ∈ ∆S−1 : pi ≥ δ ∀i}, δ ∈ (0,1/S). For m ∈ N and M ≥ mδ, define the extremal “mass-under-a-floor” functional

|Ξm(M;δ) := max<br><br>m<br><br>j=1<br><br>x2j :<br><br>m<br><br>j=1<br><br>xj = M, xj ≥ δ = (M − (m − 1)δ)2 + (m − 1)δ2.|
|---|

(24)

Then, for k = ℓ with a = pθ(k) ∈ [δ, 1 − (S − 1)δ],

∥Hkk∥22 ≤ (a(1 − a)(1 − 2a))2 + a2(2a − 1)2 ΞS−1(1 − a;δ) =: cdiag2 (δ,S) 2, and for k ̸= ℓ with a,b ∈ [δ, 1 − (S − 1)δ], r := 1 − a − b ∈ [(S − 2)δ, 1 − 2δ],

∥Hkℓ∥22 ≤ (ab)2 (2a − 1)2 + (2b − 1)2 + 4a2b2 ΞS−2(r;δ) =: coff2 (δ,S) 2.

Define c2(δ,S) := max{cdiag2 ,coff2 } < 1/√54. An entirely analogous construction (sums of absolute values instead of squares) yields c1(δ,S) < 1/(3√3) with

|max<br><br>k,ℓ<br><br>∥Hkℓ(θ)∥2 ≤ c2(δ,S), max<br><br>k,ℓ<br><br>∥Hkℓ(θ)∥1 ≤ c1(δ,S) whenever pθ ∈ ∆Sδ−1.|
|---|

The global maximizers lie at a± = 12 ± 2√13 ≈ 0.7887, 0.2113. Thus if

|δ > δcrit := 12 − 2√13 ≈ 0.2113,<br><br>|
|---|

(25)

then c2(δ,S) < 1/√54 and c1(δ,S) < 1/(3√3) strictly. The remainder bounds (23) improve by replacing the global constants with c2(δ,S) and c1(δ,S).

##### B.7 Reference table: Parametric Constants

Spectral norms are ∥ · ∥op; vector norms are Euclidean unless labeled. Tangent space T = 1⊥, projector ΠT, centering C as above. The bridge (12) Jθ = S(pθ) is used in Section C.

##### Symbol Value / Bound (where introduced)

∥Jθ∥op ≤ 21 (global); λ(Jθ |T) ∈ [pmin, 12] (Corollary B.2) ∥Jθ

√

3√3∥∆θ∥2 ≤ 3√S3∥∆θ∥∞ (Theorem B.1) ∥P(θ) − P(ϑ)∥2 ≤ 21δ

1∥op ≤ 3√13∥∆θ∥1 ≤

S

2 − Jθ

∥θ − ϑ∥2 (Proposition B.2) ∥B(θ)∥op ≤ 41δ

⋆

(Section B.5, (19)) LDP Hard-clip kink-free: (17); smooth clip: (18) Lθ ≤

⋆

√

Lp 16δ⋆2

S 12√3δ⋆2

+ 12LDP (Theorem B.2)

+ Gp

supk,ℓ ∥Hkℓ∥2 = 1/√54 (Theorem B.3) supk,ℓ ∥Hkℓ∥1 = 1/(3√3) (Theorem B.3) c1(δ,S), c2(δ,S) ℓ1/ℓ2 Hessian suprema on ∆Sδ−1, both < global constants (§B.6)

Domain reminder for composite bounds. All composite bounds in §B.5 are evaluated on the rectangle [δmin,1]S, where δmin = δ⋆/(1 + (S − 1)δ⋆) (from clip–renormalize). Assumption (A) holds on this set.

#### C The Self-Reinforcing Correctness Training (SRCT) Framework

This appendix records the SRCT calculus used throughout the paper, with canonical constants, operator identities, and dynamical statements in a form suitable for direct citation. The development is self-contained and uses the standard Shahshahani–replicator correspondence.

- C.1 Domain, notation, and canonical constants Fix K ≥ 2 and a floor 0 < δ⋆ < 1/K. The trimmed simplex is

K

∆Kδ −1

pi = 1, pi ≥ δ⋆ ∀i , T := 1⊥ = {v ∈ RK : ⟨v,1⟩ = 0}.

:= p ∈ [0,1]K :

⋆

i=1

Euclidean inner products and norms are used throughout. Write ⟨log p⟩ := i pi log pi and H(p) := −⟨log p⟩.

|Λ := 1 + log<br><br>1 δ⋆<br><br>, CA := A(2 +<br><br>√<br><br>K)Λ, A := ε + λα + βKL ≥ 0.|
|---|

##### C.2 SRCT objective, correct variational derivative, and canonical drift

Let U ∈ RK be a bounded utility vector, K ∈ RK×K symmetric PSD, and pbase ∈ ∆K−1 with full support pbase,i > 0. Consider

J[p] =

i

Uipi + λ αH[p] − β p⊤Kp − βKLKL(p∥pbase) + εH[p].

A direct calculation gives the pointwise variational derivative

δ J δpi

= Ui − 2λβ (Kp)i + βKL log pbase,i − A 1 + log pi , A = ε + λα + βKL.

Introduce the selection covariance and entropic vector S(p) := diag(p) − pp⊤, E(p) := p ⊙ (log p − ⟨log p⟩),

and the selective score

ϕA(p) := U − 2λβ Kp + βKL log pbase. Then the Shahshahani gradient flow p˙ = ∇Sh J(p) is the SRCT ODE

|p˙ = F(p) := S(p)ϕA(p) − AE(p),<br><br>i<br><br>p˙i = 0 (tangency to T).|
|---|

##### C.3 Operator facts for S and the entropic map E

Selection covariance S(p). For all p, S(p)1 = 0, and v⊤S(p)v = Varp(V ) where V takes value vi with probability pi. By Popoviciu and (max−min)2 ≤ 2∥v∥22,

|∥S(p)∥2→2 ≤ 21 , ∥S(p) − S(q)∥2→2 ≤ 3∥p − q∥2.<br><br>|
|---|

Entropic vector E(p). For any p ∈ ∆Kδ −1

and v ∈ RK, the Jacobian is

⋆

|JE(p)v = diag 1 + log p − ⟨log p⟩ v − p⟨1 + log p, v⟩.|
|---|

Consequently, on ∆Kδ −1

,

⋆

|∥E(p) − E(q)∥2 ≤ (2 +<br><br>√<br><br>K)Λ∥p − q∥2.|
|---|

##### C.4 Global Lipschitz of the SRCT drift and Carath´eodory regularity

∥ϕA(p)∥2 < ∞ (compactness). Using §C.3 and F = SϕA − AE,

Let Lϕ := 2λβ ∥K∥2→2 and Mϕ,2 := supp∈∆K−1

δ⋆

|∥F(p) − F(q)∥2 ≤ 12 Lϕ + 3Mϕ,2 + CA ∥p − q∥2.<br><br>|
|---|

Hence F is globally Lipschitz on ∆Kδ −1

. For non-autonomous scores ϕA(t,p) that are measurable in t, locally Lipschitz in p, and locally bounded, F(t,p) satisfies Carath´eodory conditions on ri∆Kδ −1

⋆

; the ODE admits a unique local absolutely continuous solution from any interior initial condition. Tangency to T and §C.7 (BD) give global-in-time confinement.

⋆

###### C.5 Mass balance and log-ratio calculus For any absolutely continuous solution p(·) with M(t) := i pi(t),

|M˙ (t) = ϕA(t,p(t)) − A⟨log p(t)⟩ 1 − M(t) , ϕA =<br><br>i<br><br>piϕA,i.|
|---|

Thus M(0) = 1 ⇒ M(t) ≡ 1. Fix i ̸= j and let J be an interval on which pi,pj > 0. Set z(t) := log p

i(t)

pj(t) and dij(t) := Ui − Uj − 2λβ (Kp)i − (Kp)j + βKL log

- pbase,i

- pbase,j

.

Subtracting the i and j equations yields the log-ratio identity

|z˙(t) = dij(t) − Az(t) for a.e. t ∈ J, z(t) = z(t0)e−A(t−t<br><br>0) +<br><br>t<br><br>t0<br><br>e−A(t−s) dij(s)ds.|
|---|

(eq:C-VoC)

The usual time-varying and constant-box envelopes follow by comparison; if A > 0 and |dij| ≤ M on [t0,∞)∩J, then |z(t)| ≤ |z(t0)|e−A(t−t

0) + MA (1 − e−A(t−t

0)) (uniform boundedness).

- C.6 Positivity and face invariance on the closed simplex Let H(p) = −⟨log p⟩ ∈ [0,log K] and Mtraj(t) := maxk |ϕA,k − ϕA|(t,p(t)) ∈ L1loc.

- Lemma C.1 (No finite-time boundary hitting). If pi(0) > 0, then for all finite t,

|log pi(t) ≥ log pi(0) −<br><br>t<br><br>0<br><br>Mtraj(s) + AH p(s) ds, ⇒ pi(t) > 0.|
|---|

- Lemma C.2 (Face invariance at zero). If pi(0) = 0, then pi(t) ≡ 0. Sketch. With y = pi, one has y′ =

a(t)y − Ay log y with a ∈ L1loc. The Osgood modulus ω(y) = y(1 + |log y|) satisfies 0

+

dr/ω(r) = ∞, giving uniqueness of y ≡ 0 through y(0) = 0.

- C.7 Barrier–Dominance and confinement on ∆Kδ −1

⋆

On the lower face {pi = δ⋆}, using pj ≥ δ⋆ and j̸=i pj = 1 − δ⋆, the convexity of x  → xlog x yields the entropy face gap

|LK(δ⋆) := (1 − δ⋆)log<br><br>1 − δ⋆ (K − 1)δ⋆<br><br>> 0 (δ⋆ < 1/K).|
|---|

A direct computation gives the face inequality

|at pi = δ⋆ : Fi(p) ≥ δ⋆ ALK(δ⋆) − ϕA,i(p) − ϕA(p) − .<br><br>|
|---|

(eq:C-face-gap)

Define the worst outward selective pressure on the boundary

Meffface := sup

p∈∂∆Kδ⋆−1 i: pi=δ⋆

ϕA,i(p) − ϕA(p) − < ∞.

Theorem C.1 (Barrier–Dominance). If

|ALK(δ⋆) ≥ Meffface|
|---|

(eq:C-BD)

then F(p) lies in the tangent cone of ∆Kδ −1

⋆

at every boundary point; hence ∆Kδ −1

⋆

is forward invariant. If the inequality is strict, trajectories starting in ri∆Kδ −1

⋆

never hit the boundary (strict interior invariance).

Coarse sufficient BD. Since |ϕA,i − ϕA| ≤ 2∥ϕA∥∞, it suffices that

ALK(δ⋆) ≥ 2 sup

p∈∆Kδ⋆−1

∥ϕA(p)∥∞.

Degenerate floor: If δ⋆ = 1/K, then LK(δ⋆) = 0 and the simplex is a singleton.

- C.8 Existence/uniqueness on the mass hyperplane

By §C.4, F is globally Lipschitz on ∆Kδ −1

⋆

and tangent to H := {p : i pi = 1}. Kirszbraun–Valentine yields a Lipschitz extension F : H → H with the same constant; Picard–Lindel¨f gives a unique global absolutely continuous solution from any p(0) ∈ H. Under (C.1), the trajectory remains in ∆Kδ −1

⋆

.

- C.9 Single-site score fields: Lyapunov structure and convergence

Assume a separable score ϕi(p) = fi(pi) with fi ∈ C([δ,1]) ∩ C1((δ,1]), supi,s |fi′(s)| < ∞, and fi′ ≤ 0 on (δ,1]. On ∆Kδ −1

take δ = δ⋆; on the closed simplex (for A = 0) take δ = 0. Define

⋆

gi(s) := fi(s) − Alog s, Ψi(s) :=

s

gi(u)du, Lψ(p) :=

s0

K

Ψi(pi), g¯(p) :=

i=1

i

pigi(pi).

Along classical solutions,

|d dt Lψ p(t) =<br><br>K<br><br>i=1<br><br>pi(t) gi(pi(t)) − g¯(p(t)) 2 ≥ 0.|
|---|

Regime A > 0: strong concavity, KKT, convergence. On [δ⋆,1], gi′(s) = fi′(s)−A/s ≤ −A, hence on the affine simplex

D2Lψ(p) = diag(g1′ (p1),...,gK′ (pK)) ⪯ −AI, so Lψ is A-strongly concave. Maximization over ∆Kδ −1

has a unique solution p†; the KKT conditions give a scalar c† and multipliers νi† ≥ 0 such that

⋆

gi(p†i) = c† − νi†, νi†(δ⋆ − p†i) = 0,

p†i = 1.

i

Under strict BD, p† is interior and gi(p†i) ≡ c†. Since trajectories are confined and Lψ is nondecreasing and bounded above, LaSalle’s invariance principle implies global convergence to p†.

Regime A = 0: water-filling and support selection. Assume (CR+SM): each fi is continuous and strictly decreasing on [0,1], with inverse fi−1 : [fi(1),fi(0)] → [1,0]. There exists a unique pair (S⋆,c⋆) with

fi−1(c⋆), i ∈ S⋆, 0, i ∈/ S⋆,

fi−1(c⋆) = 1, p⋆i =

S⋆ = {i : fi(1) ≤ c⋆ < fi(0)}.

i∈S⋆

Moreover, Lψ is strictly concave on every face; by face invariance and monotonicity, p(t) → p⋆.

##### C.10 Safe denominators (linear-functional floor)

If ϕ contains denominators of the form a⊤p with a ∈ RK+ \ {0}, then on ∆Kδ −1

,

⋆

|a⊤p ≥ δ⋆ ∥a∥1.|
|---|

Hence such denominators are uniformly bounded away from zero.

#### D STaR through the SRCT Lens

This appendix instantiates the SRCT framework for the Self-Taught Reasoner. We specify the score field, establish norm and Lipschitz bounds (including Jacobian structure and rank), prove well-posedness and confinement (trimmed-domain barrier–dominance), and analyze log-ratio dynamics and asymptotics.

- D.1 Setting, notation, and basic aggregates Fix K ≥ 2 and the probability simplex

K

∆K−1 := p ∈ [0,1]K :

pk = 1 , int∆K−1 := {p ∈ ∆K−1 : pk > 0 ∀k}.

k=1

Split indices into correct C (size M ≥ 1) and incorrect I := {1,...,K} \ C (size L = K − M). For p ∈ ∆K−1 define

K

pc, S(2)(p) :=

p2c, ⟨log p⟩ :=

pk log pk ∈ [−log K,0].

ρ(p) :=

c∈C

c∈C

k=1

For a floor δ⋆ ∈ (0,1/K), the trimmed simplex is ∆Kδ −1

:= {p ∈ ∆K−1 : min

pk ≥ δ⋆} ⇒ ρ(p) ≥ Mδ⋆.

⋆

k

Vector norms are Euclidean; for matrices we use ∥ · ∥1 (max. column sum), ∥ · ∥∞ (max. row sum), and the spectral norm ∥ · ∥2, with ∥J∥2 ≤ ∥J∥1∥J∥∞.

##### D.2 The STaR score field: bounds, Jacobian, and Lipschitzness Definition D.1 (STaR score). On D := {p ∈ int∆K−1 : ρ(p) > 0} define ϕSTaR : D → RK by

 

pk − S(2)(p) ρ(p)

, k ∈ C,

ϕSTaRk (p) =

S(2)(p) ρ(p)



−

, k ∈ I.

For M ≥ 1 and p ∈ int∆K−1, ρ(p) > 0, hence D = int∆K−1 and ϕSTaR is C∞ on D.

##### Componentwise and norm bounds (sharp). For ρ = ρ(p) and S(2) = S(2)(p):

K

pk ϕSTaRk (p) = 0 (centering).

k=1

For c ∈ C, 0 ≤ pc ≤ ρ and S(2) ≥ ρ2/M (Cauchy–Schwarz), whence

|ϕc ∈ −ρ, 1 − Mρ , ϕi = −S<br><br>(2)<br><br>ρ ∈ [−ρ,0] (i ∈ I), ∥ϕSTaR(p)∥∞ ≤ 1.|
|---|

Moreover,

|∥ϕSTaR(p)∥22 ≤ 1 − 2ρ(p) + K ρ(p)2 ≤ K − 1, ∥ϕSTaR(p)∥2 ≤<br><br>√<br><br>K − 1.|
|---|

The quadratic upper bound is tight in the limit ρ → 1 with all correct mass on one index.

- Lemma D.1 (Jacobian, zero columns on I, and rank). Let J(p) := [∂ϕSTaRk /∂pj](p). Then Jk,j(p) = 0 for all j ∈ I. For j ∈ C,

2pjρ − S(2) ρ2

S(2) ρ

δkjρ − pk ρ2

∂ ∂pj

∂ ∂pj

pk ρ

=

,

=

,

hence

 

S(2) ρ2

δkj ρ −

pk ρ2 −

2pj ρ

, k ∈ C, j ∈ C,

+

S(2) ρ2

Jk,j(p) =

2pj ρ

, k ∈ I, j ∈ C, 0, j ∈ I.

−

+



In particular, rankJ(p) ≤ M.

- Proposition D.1 (Lipschitz bounds on ∆Kδ −1

and interior compacts). On ∆Kδ −1

one has ρ ≥ Mδ⋆. Uniformly for p ∈ ∆Kδ −1

⋆

⋆

,

⋆

|∥J(p)∥∞ ≤<br><br>2 δ⋆<br><br>+ M + 2, ∥J(p)∥1 ≤<br><br>2 Mδ⋆<br><br>+ 3K, ∥J(p)∥2 ≤ Mδ 2<br><br>⋆<br><br>+ 3K δ 2<br><br>⋆<br><br>+ M + 2 .|
|---|

If D0 ⊂ int∆K−1 is compact with ρ(p) ≥ ρmin > 0, then uniformly for p ∈ D0,

|∥J(p)∥∞ ≤<br><br>M + 1 ρmin<br><br>+ M + 2, ∥J(p)∥1 ≤<br><br>2 ρmin<br><br>+ 3K, ∥J(p)∥2 ≤ ρ 2<br><br>min<br><br>+ 3K Mρ +1<br><br>min<br><br>+ M + 2 .|
|---|

Proof sketch. Sum the absolute values of the entries in Lemma D.1 by rows/columns using ρ ≥ Mδ⋆, pj ≤ ρ, S(2) ≤ ρ2; then apply ∥J∥2 ≤ ∥J∥1∥J∥∞.

Continuity caveat (stiffness near faces). Although ϕSTaR is bounded and smooth on D, the 1/ρ2 factors in J blow up as ρ ↓ 0. Thus ϕSTaR is not globally Lipschitz on int∆K−1; quantitative Lipschitz control requires either ∆Kδ −1

or a uniform ρmin > 0.

⋆

- Proposition D.2 (Ambient spectral lower bound; dependence on M). For all p ∈ D,

|∥J(p)∥2 ≥<br><br>∥pC∥2 ρ(p)<br><br>√<br><br>K ≥<br><br>K M<br><br>.|
|---|

√

Proof. Let v = (pC/∥pC∥2, 0I). Lemma D.1 implies Jv = −(∥pC∥2/ρ)1. Taking inner product with 1/

K yields the first inequality; Cauchy–Schwarz gives ∥pC∥2 ≥ ρ/

√

M.

Corollary√ D.1 (Exact formulas when M = 1). If M = 1 with C = {c}, then J(p) = −1e⊤c , hence ∥J(p)∥2 = K. The restriction to the tangent space T = 1⊥ has operator norm ∥J|T∥2 = √K − 1; moreover ΠTJΠT ≡ 0.

- D.3 STaR as an SRCT flow: well-posedness, Lipschitz drift, and confinement Dynamics. For ε ≥ 0 (entropic weight), the SRCT ODE reads

|p˙k = pk ϕSTaRk (p) − εpk log pk − ⟨log p⟩ , k = 1,...,K.|
|---|

By centering, k p˙k = 0, so k pk(t) ≡ 1.

No finite-time boundary hitting and uniform floor. Let Yi := −log pi. Using |ϕSTaRi | ≤ 1 and −⟨log p⟩ ≤ log K,

Y˙i ≤ 1 + εlog K − εYi.

Therefore Yi(t) remains finite on any finite interval (no coordinate reaches 0 in finite time, even for ε = 0). If ε > 0, solving the linear inequality gives the uniform floor

|pi(t) ≥ min pi(0), K1 e−1/ε (∀t ≥ 0).<br><br>|
|---|

##### Global ℓ2 Lipschitz bound for the SRCT drift on ∆Kδ −1

. Write S(p) := diag(p) − pp⊤ and E(p) := p ⊙ (log p − ⟨log p⟩). Then

⋆

F(p) := p ⊙ ϕSTaR(p) − εE(p) = S(p)ϕSTaR(p) − εE(p).

On ∆Kδ −1

,

⋆

∥S(p)∥2→2 ≤ 12, ∥S(p) − S(q)∥2→2 ≤ 3∥p − q∥2, and, with Λ := 1 + log(1/δ⋆),

√

K)Λ∥p − q∥2. Combining with sup∥ϕSTaR∥2 ≤

∥E(p) − E(q)∥2 ≤ (2 +

√

∥J(r)∥2 from Proposition D.1,

K and Lϕ,2 := supr∈∆K−1

δ⋆

|∥F(p) − F(q)∥2 ≤ 12 Lϕ,2 + 3<br><br>√<br><br>K + ε(2 +<br><br>√<br><br>K)Λ ∥p − q∥2 (p,q ∈ ∆Kδ −1<br><br>⋆<br><br>).|
|---|

##### Forward invariance of a trimmed simplex (Barrier–Dominance). On the facet pi = δ⋆,

p˙i = δ⋆ ϕSTaRi (p) + ε[⟨log p⟩ − log δ⋆ ] . The entropy face gap

1 − δ (K − 1)δ

⟨log p⟩ − log δ = (1 − δ)log

LK(δ) := inf

p: pi=δ

is attained by equalizing the other K − 1 coordinates. Since ϕSTaRi ≥ −1, inf

p˙i ≥ δ⋆ −1 + εLK(δ⋆) ,

p: pi=δ⋆

so the sharp sufficient condition

|εLK(δ⋆) ≥ 1|
|---|

guarantees inward pointing drift on every facet and hence forward invariance (Nagumo). A conservative alternative, robust to mild non-centering, uses |ϕi − ϕ¯| ≤ 2∥ϕ∥2 ≤ 2

√

K to give

|εLK(δ⋆) ≥ 2<br><br>√<br><br>K .|
|---|

Uniform linear growth. Along any trajectory in int∆K−1,

||p˙i| ≤ pi|ϕi| + ε |pi log pi| + pi|⟨log p⟩| ≤ 1 + ε 1e + log K .<br><br>|
|---|

Well-posedness summary. For any p(0) ∈ int∆K−1 and ε ≥ 0 there is a unique global solution in int∆K−1 (no finite-time boundary hitting). On ∆Kδ −1

the drift is globally Lipschitz with the bound above; under either BD condition the trimmed simplex is forward invariant. For ε > 0 every coordinate satisfies the uniform floor.

⋆

##### D.4 Log-ratio dynamics and asymptotics

For k ̸= j, set zkj := log p

pj . Differentiating,

k

|z˙kj(t) = ϕSTaRk (p(t)) − ϕSTaRj (p(t)) − εzkj(t).|
|---|

Instantiating the score differences:

pa − pb ρ

ϕi − ϕj ≡ 0 (i,j ∈ I), ϕa − ϕb =

pc ρ

(a,b ∈ C), ϕc − ϕi =

(c ∈ C,i ∈ I).

Incorrect vs. incorrect (i,j ∈ I). z˙ij = −εzij ⇒ zij(t) = zij(0)e−εt: incorrect traces equalize exponentially when ε > 0.

a−pb

Within C (a,b ∈ C). z˙ab = p

ρ − εzab, p

a−pb

ρ < 1. Variation of constants yields

1 − e−εt ε

|zab(t)| ≤ |zab(0)|e−εt +

.

On ∆Kδ −1

, ρ ≥ Mδ⋆ strengthens this to

⋆

||zab(t)| ≤ |zab(0)|e−εt +<br><br>1 − Mδ⋆ ε<br><br>(1 − e−εt).|
|---|

Correct vs. incorrect (c ∈ C,i ∈ I). Let c⋆(t) ∈ arg maxc∈C pc(t) and set zic⋆ := log p

. Then

i

pc⋆

pc⋆ ρ − εzic⋆,

z˙ic⋆ = −

pc⋆ ρ ∈

1 M

, 1 ,

so

|zic⋆(t) ∈ zic⋆(0)e−εt − 1−e<br><br>−εt<br><br>ε , zic⋆(0)e−εt − 1−e<br><br>−εt<br><br>Mε , limsup<br><br>t→∞<br><br>pi(t) pc⋆(t) ≤ e−1/(Mε).<br><br>|
|---|

Asymptotics. If ε > 0 and there exists c ∈ C with pc(t) → p∞c > 0 and p

c(t)

ρ(t) → g ∈ [1/M,1], then zic(t) → −g/ε and

|pi(t) → p∞c e−g/ε ∈ p∞c e−1/ε, p∞c e−1/(Mε) .|
|---|

If ε = 0 and there exist c ∈ C, gmin > 0 with p

c(t)

ρ(t) ≥ gmin on an unbounded time set, then z˙ci ≥ gmin, hence zci(t) → +∞ and pi(t) → 0 (incorrect mass vanishes). Non-vanishing ρ alone does not imply extinction.

##### D.5 Edge cases and remarks

If M = 0 the score in Definition D.1 is undefined (ρ ≡ 0). If M = K, then ρ ≡ 1 and ϕSTaRk (p) = pk − Kj=1 p2j. The ambient lower bound in Proposition D.2 is realized in the normal direction span{1} and does not directly

lower-bound the tangent-restricted operator ΠTJΠT with T = 1⊥.

#### E GRPO through the SRCT Lens

We analyze GRPO within the SRCT framework. We prove barrier–dominance (face invariance), derive rank-one Lipschitz constants for the GRPO score, obtain two-sided cross-class envelopes, and establish exponential convergence to a unique two-level equilibrium under a slope condition.

- E.1 Setup and GRPO characteristic Domain and classes. Fix integers K ≥ 2, G ≥ 2, and a floor δ⋆ ∈ (0,1/K]. Work on the trimmed simplex

K

∆Kδ −1

:= p ∈ [0,1]K :

pk = 1, pk ≥ δ⋆ .

⋆

k=1

Partition indices into correct and incorrect sets C,I with sizes KC := |C| ≥ 0, KI := |I| ≥ 0, KC + KI = K. Write the correct mass

ρ := ρC(p) :=

pc.

c∈C

If KI ≥ 1 and p ∈ ∆Kδ −1

then ρ ∈ KCδ⋆, 1 − KIδ⋆ .

⋆

GRPO characteristic. For t ∈ (0,G] set fG(t) := (G − t)/t. With S ∼ Binom(G − 1,ρ) define

c1(ρ) 1 − ρ

c1(ρ) := E fG(1 + S) , hG(ρ) :=

(ρ ∈ (0,1)).

- Lemma E.1 (basic properties of hG). The map hG extends to C1([0,1]) with

√

|h′G(ρ)| < ∞.

G − 1, DG := sup

hG(0) = hG(1) =

ρ∈[0,1]

Moreover for all ρ ∈ [0,1],

√

1 − G1 ≤ hG(ρ) ≤

G − 1, and hG is constant when G ∈ {2,3}.

Proof sketch. c1 is a finite binomial sum of smooth terms, hence C∞([0,1]). Expansion at ρ = 1 gives c1(1) = 0 and c′1(1) = −

√G − 1, so hG extends continuously with hG(1) = √G − 1 and is C1 on [0,1]; boundedness of h′G follows by continuity on a compact interval. The lower bound follows from fG(t) ≥ (G − t)/G on t ∈ [1,G]. The upper bound follows from a binomial reweighting showing hG is an average of terms bounded by √G − 1.

| |
|---|

- Lemma E.2 (binomial-shift identities). For all ρ ∈ [0,1] with S ∼ Binom(G − 1,ρ),

(1 − ρ)hG(ρ) = E G−1+1S−S , ρhG(ρ) = E G−SS .

##### E.2 GRPO scores: envelopes and rank-one Lipschitz constants Scores and centering. The raw GRPO score is class-constant:

γkraw(p) =

Its centered version γk := γkraw − j pjγjraw equals

hG(ρ), k ∈ C, 0, k ∈ I.

γk(p) =

(1 − ρ)hG(ρ), k ∈ C, −ρhG(ρ), k ∈ I,

If KI = 0 or KC = 0 then γ ≡ 0. Pointwise envelopes. By Lemma E.2,

K

pk γk(p) = 0.

k=1

∥ γ(p)∥∞ ≤

√

G − 1,

|∥γ(p)∥2 = hG(ρ) KC(1 − ρ)2 + KIρ2 ≤<br><br>√<br><br>G − 1 max{KC,KI} .<br><br>|
|---|

If additionally KI ≥ 1 and p ∈ ∆Kδ −1

, then 1 − ρ ≥ KIδ⋆ and

⋆

√G − 1 KIδ⋆

hG(ρ) ≤

=: HG, ⇒ ∥ γ(p)∥2 ≤ HG max{KC,KI}.

##### Rank-one Jacobian and exact norms. Set

d dρ

d

dρ − ρhG(ρ) = −hG(ρ) − ρh′G(ρ). Since ∇ρC = 1C,

(1 − ρ)hG(ρ) = c′1(ρ), β(ρ) :=

α(ρ) :=

D γ(p) = α 1C, β 1I (1C)⊤ =: uv⊤ (rank one). Thus the operator norms are exact:

∥D γ(p)∥2→2 = ∥u∥2 ∥v∥2 = KC KCα2 + KIβ2 1/2,

|∥Dγ(p)∥T→2 = K<br><br>CKI K KCα2 + KIβ2 1/2 = K<br><br>I<br><br>K ∥Dγ(p)∥2→2 .|
|---|

Consequently, the sharp global Lipschitz constant on the simplex is

|Ltanγ := sup<br><br>p∈∆K−1<br><br>∥D γ(p)∥T→2 = K<br><br>CKI K sup<br><br>ρ∈[0,1]<br><br>KCα(ρ)2 + KIβ(ρ)2 1/2 .|
|---|

From |α| ≤ H⋆ + DG, |β| ≤ H⋆ + DG with H⋆ := sup|hG| = √G − 1,

Ltanγ ≤ KCKI H⋆ + DG .

- E.3 SRCT drift: global Lipschitzness and mass conservation Drift. With entropy weight ε > 0 define

Fk(p) := pk γk(p) − ε log pk − ⟨log p⟩ , ⟨log p⟩ :=

K

pi log pi.

i=1

Centeredness yields k Fk(p) = 0 (mass conservation).

##### Entropic Lipschitz bound on ∆Kδ −1

. On [δ⋆,1], h(x) := xlog x has ∥h′∥∞ ≤ Λ := 1 + log(1/δ⋆). A direct decomposition gives

⋆

√

K)∥p − q∥2, p,q ∈ ∆Kδ −1

∥Fent(p) − Fent(q)∥2 ≤ εΛ(2 +

.

⋆

##### Selection Lipschitz bound and full modulus. For Fsel(p) := p ⊙ γ(p) and p,q ∈ ∆Kδ −1

,

⋆

∥Fsel(p) − Fsel(q)∥2 ≤ ∥diag(p)∥2→2 Ltanγ + sup

∥ γ(r)∥2 ∥p − q∥2,

r∈∆Kδ⋆−1

√G − 1 max{KC,KI} or (when KI ≥ 1) the trim-aware bound HG max{KC,KI},

with ∥diag(p)∥2→2 ≤ 1 − (K − 1)δ⋆. Using either sup∥ γ∥2 ≤

|∥F(p) − F(q)∥2 ≤ (1 − (K − 1)δ⋆)Ltanγ + Mγ + εΛ(2 +<br><br>√<br><br>K) ∥p − q∥2 ,|
|---|

where Mγ denotes the chosen envelope.

- E.4 Barrier–Dominance (BD) and forward invariance Entropy face gap. For a facet pk = δ⋆ define the gap

Gapk(p) := ⟨log p⟩ − log δ⋆. The global lower benchmark (uniform-others gap) is

|LK(δ⋆) := (1 − δ⋆)log<br><br>1 − δ⋆ (K − 1)δ⋆<br><br>.|
|---|

At fixed ρ = ρC(p), the minimal face gap is attained by equalizing within blocks:

Emin(I) (ρ) = (δ⋆ − 1)log δ⋆ + 1{K

C≥1} ρlog

ρ KC

Emin(C) (ρ) = (δ⋆ − 1)log δ⋆ + 1{K

C≥2} (ρ − δ⋆)log

and minρ Emin(·) (ρ) = LK(δ⋆). Exact BD on facets. On pk = δ⋆,

1 − δ⋆ − ρ KI − 1

I≥2} (1 − δ⋆ − ρ)log

+ 1{K

,

ρ − δ⋆ KC − 1

1 − ρ KI

I≥1} (1 − ρ)log

+ 1{K

,

Fk(p) = δ⋆ γk(p) + εGapk(p) .

Correct faces: if k ∈ C and KI ≥ 1, then (1 − ρ) ≥ KIδ⋆ > 0 implies γk = (1 − ρ)hG(ρ) > 0, hence Fk(p) ≥ εδ⋆Emin(C) (ρ) ≥ 0 (automatically inward). Incorrect faces: if k ∈ I, then γk = −ρhG(ρ) ≤ 0. The facet is inward/tangent iff

|(BDexact) εEmin(I) (ρ) ≥ ρhG(ρ) ∀ ρ ∈ KCδ⋆, 1 − KIδ⋆ .|
|---|

√G − 1,

Convenient sufficient relaxations. Using Emin(I) (ρ) ≥ LK(δ⋆) and ρhG(ρ) ≤

|εLK(δ⋆) ≥<br><br>√<br><br>G − 1 =⇒ (BDexact).|
|---|

On trimmed domains with KI ≥ 1, 1 − ρ ≥ KIδ⋆ implies hG(ρ) ≤ HG = √G − 1/(KIδ⋆), hence

|εLK(δ⋆) ≥<br><br>√G − 1 KI δ⋆<br><br>=⇒ (BDexact).|
|---|

Well-posedness and invariance. Interior solutions cannot hit the boundary in finite time: writing yi := −log pi,

√

G − 1 − εyi + εlog K, so yi cannot blow up in finite time. If (BDexact) (or either sufficient relaxation) holds, every facet is inward/tangent; ∆Kδ −1

y˙i = − γi(p) − εyi − ε⟨log p⟩ ≤

is forward invariant and the drift is globally Lipschitz on a compact forward-invariant set, yielding global existence and uniqueness.

⋆

##### E.5 Log-ratio dynamics, envelopes, and scalar reduction

For i ̸= j,

pi pj

- pi

- pj

d dt

= γi(p) − γj(p) − εlog

log

.

Intra-class equalization. If i,j are in the same class then γi = γj and

|log<br><br>pi(t)<br><br>pj(t)<br><br><br>= e−εt log<br><br>pi(0)<br><br>pj(0)<br><br><br>.|
|---|

Thus within-class proportions equalize exponentially at rate ε.

Cross-class envelopes. For c ∈ C, i ∈ I let zci := log(pc/pi). Then

z˙ci(t) = hG(ρC(t)) − εzci(t). Variation of constants and Lemma E.1 give, for all t ≥ 0,

|zci(t) ∈ zci(0)e−εt + 1−<br><br>1 G<br><br>ε (1 − e−εt), zci(0)e−εt +<br><br>√G−1<br><br>ε (1 − e−εt) .|
|---|

If (BD) holds with KI ≥ 1, then hG(ρC(s)) ≤ HG along the trajectory and the upper envelope sharpens to zci(t) ≤ zci(0)e−εt +

HG ε

(1 − e−εt).

Feasibility band (under BD). Write pc = αcρ with c αc = 1 and pi = βi(1−ρ) with i βi = 1, and define

|Ψ(ρ) := log<br><br>KI KC ·<br><br>ρ 1 − ρ<br><br>, ρ(z) =<br><br>KCez KI + KCez<br><br>.|
|---|

Let

- pa(t)

- pb(t)

- pj(t)

- pk(t)

, δintra(t) := ∆C(t) + ∆I(t) = δintra(0)e−εt. Then

∆C(t) := max a,b∈C

log

, ∆I(t) := max j,k∈I

log

||zci(t) − Ψ(ρC(t))| ≤ δintra(t) and ρC(t) ∈ KCδ⋆, 1 − KIδ⋆ .|
|---|

Scalar reduction, closure error, and fixation (under BD). Define F×(z) := hG(ρ(z)) − εz. Since |ρ′(z)| ≤ 41,

4 |zci − Ψ(ρC)| ≤ D

hG(ρC) − hG(ρ(zci)) ≤ DG |ρC − ρ(zci)| ≤ D

4 δintra(t). Hence z˙ci = F×(zci) + r(t) with |r(t)| ≤ D

G

G

4 δintra(t).

G

- Theorem E.1 (fixation under a slope condition). If ε > D

4 , then F× is strictly decreasing and has a unique zero z⋆. Moreover, for all c ∈ C, i ∈ I,

G

||zci(t) − z⋆| ≤ e−(ε−<br><br>DG<br><br>4 )t |zci(0) − z⋆| + ∆C(0) + ∆I(0) .|
|---|

If z⋆ ∈ Ψ(KCδ⋆), Ψ(1 − KIδ⋆) then the limit distribution is interior and class-uniform: p⋆c =

ez

1 KCez⋆ + KI

⋆

(c ∈ C), p⋆i =

(i ∈ I). Otherwise the limit lies on the corresponding face (feasibility truncation).

KCez⋆ + KI

##### E.6 Edge cases and checks

- • Maximal trim: if δ⋆ = 1/K, then ∆Kδ −1

⋆

= {(1/K,...,1/K)}; dynamics are trivial.

- • Degenerate classes: if KI = 0 or KC = 0, then γ ≡ 0 and p˙i = −εpi(log pi −⟨log p⟩); the unique equilibrium on active coordinates is uniform.
- • Single incorrect: KI = 1 yields ρ = 1 − δ⋆ on the only incorrect face and

Emin(I) (1 − δ⋆) = (δ⋆ − 1)log δ⋆ + (1 − δ⋆)log 1−δ

⋆

KC .

The uniform sufficient BD εLK(δ⋆) ≥

√G − 1 is sharp as δ⋆ ↓ 0.

- • Two classes (K = 2): KC = KI = 1 and z = log(pc/pi) obey z˙ = hG(pc) − εz; the envelopes become equalities with ρ = pc.
- • Constant cases: for G ∈ {2,3}, hG ≡

√G − 1, so Ltanγ = √G − 1√KCKI and F×(z) = √G − 1 − εz.

#### F DPO through the SRCT Lens

This appendix develops a self-contained SRCT analysis of Direct Preference Optimisation (DPO). We define the score field, prove uniform size and Lipschitz bounds (with explicit constants), record entropy and fulldrift Lipschitz constants, establish well-posedness and Barrier–Dominance (BD) confinement (exact face test and tight templates), derive intra-class contraction with sharp thresholds, give cross-class envelopes (including trimmed sharpening and a static cap), prove eventual trimming under a slope condition, and conclude existence, uniqueness, and global convergence to a two-level equilibrium. All logarithms are natural.

Notation. Fix an integer K ≥ 2. The simplex and trimmed simplex are

∆K−1 := p ∈ [0,1]K :

K

pi = 1 , ∆Kδ −1

:= p ∈ ∆K−1 : min

pi ≥ δ⋆ ,

⋆

i

i=1

with floor 0 < δ⋆ < 1/K. For vectors, ∥·∥∞,∥·∥2 denote max/Euclidean norms; for matrices, ∥·∥2→2. We write ⟨log p⟩ := j pj log pj.

##### F.1 Setting and single-site map

Each index i ∈ {1,...,K} is labeled si ∈ {+1,−1}, with C := {i : si = +1}, I := {i : si = −1} and sizes M := |C|, N := |I|. Fix β > 0 and a reference ℓ0 ∈ R. Define

1 1 + e−z , so gβ ∈ C∞(R), 0 < gβ(ℓ) < 1, strictly decreasing, and

gβ(ℓ) := 1 − σ β(ℓ − ℓ0) , σ(z) :=

β(ℓ − ℓ0)

β 4

sech2

gβ′ (ℓ) = −

2 ∈ [−β/4,0). For u ∈ (0,1], define the raw scores and centered field

γi(u) := si gβ(log u), γ¯(p) :=

K

pjγj(pj), ϕi(p) := γi(pi) − γ¯(p).

j=1

By construction, i piϕi(p) = 0.

##### F.2 Uniform size and Lipschitz bounds for the DPO score

Let

1 δ⋆

gβ(log u) = gβ(log δ⋆) ∈ (0,1), Λ := 1 + log

Mγ,∞ := sup

.

u∈[δ⋆,1]

- Lemma F.1 (Size bounds). For every p ∈ ∆Kδ −1

, ∥ϕ(p)∥∞ ≤ 2Mγ,∞, ∥ϕ(p)∥2 ≤ 2Mγ,∞

⋆

√

K. Proof. |ϕi| ≤ |γi| + |γ¯| ≤ Mγ,∞ + j pj|γj| ≤ 2Mγ,∞, then ∥ · ∥2 ≤

√

K∥ · ∥∞.

| |
|---|

- Lemma F.2 (Lipschitz of single-site map). For fi(s) := γi(s) = sigβ(log s) on [δ⋆,1],

|fi′(s)| = |gβ′ (log s)|

s ≤

cmax δ⋆ ≤

β 4δ⋆

=: Lf,

where cmax := supℓ∈[logδ

⋆,0](−gβ′ (ℓ)) ≤ β/4; the inequality is strict if ℓ0 ∈/ [log δ⋆,0].

- Lemma F.3 (Operator-norm Lipschitz for ϕ). For all p,q ∈ ∆Kδ −1

⋆

, ∥ϕ(p) − ϕ(q)∥2 ≤ Lϕ ∥p − q∥2, Lϕ := K Mγ,∞ + (

√

K + 1)Lf. Proof. Write ϕ(p) = f(p) − 1(p⊤f(p)) with f(p) = (fi(pi))i. Then

Jϕ(p) = diag(f′(p)) − 1(f(p) + p ⊙ f′(p))⊤. On ∆Kδ −1

⋆

: ∥f(p)∥2 ≤

√

KMγ,∞, ∥p ⊙ f′(p)∥2 ≤ Lf, ∥diag(f′(p))∥2→2 ≤ Lf. Hence ∥Jϕ(p)∥2→2 ≤ Lf + ∥1∥2(∥f(p)∥2 +∥p⊙f′(p)∥2) = K Mγ,∞ +(

√

K +1)Lf, and the mean-value formula on the convex domain yields the claim.

| |
|---|

- Lemma F.4 (Mixed ℓ∞–ℓ1 bound). For all p,q ∈ ∆Kδ −1

⋆

,

∥ϕ(p) − ϕ(q)∥∞ ≤ Lf ∥p − q∥∞ + (Mγ,∞ + Lf)∥p − q∥1. F.3 Entropy map and drift Lipschitzness Define

E(p) := p ⊙ (log p − ⟨log p⟩1), F(p) := p ⊙ ϕ(p) − εE(p) (ε ≥ 0).

- Lemma F.5 (Entropy map). For all p,q ∈ ∆Kδ −1

, ∥E(p) − E(q)∥2 ≤ Clog ∥p − q∥2, Clog := (2Λ − 1) +

⋆

√

√

K Λ ≤ (2 +

K)Λ. Proof. The Jacobian is JE(p)v = diag(1+log p−⟨log p⟩)v −p⟨1+log p, v⟩. On ∆Kδ −1

, ∥ diag(·)∥2→2 ≤ 2Λ−1 and ∥p⟨1 + log p,·⟩∥2→2 ≤ ∥p∥2∥1 + log p∥2 ≤

√

⋆

K Λ. Mean-value completes the proof.

| |
|---|

- Proposition F.1 (Full drift Lipschitz). For all p,q ∈ ∆Kδ −1

⋆

,

∥F(p) − F(q)∥2 ≤ Lϕ + 2Mγ,∞ + εClog ∥p − q∥2.

Proof. Product decomposition: ∥p⊙ϕ(p)−q⊙ϕ(q)∥2 ≤ ∥ϕ(p)∥∞∥p−q∥2+∥ϕ(p)−ϕ(q)∥2 ≤ (2Mγ,∞+Lϕ)∥p−q∥2, then add the entropy term via Lemma F.5.

| |
|---|

- F.4 DPO–SRCT ODE, mass conservation, and positivity The SRCT drift is

|p˙i = pi ϕi(p) − ε log pi − ⟨log p⟩ , i = 1,...,K.|
|---|

Mass conservation holds since i piϕi(p) = 0 and i pi(log pi − ⟨log p⟩) = 0.

- Proposition F.2 (No finite-time boundary hitting). Let p(0) ∈ int∆K−1 and ε ≥ 0. Then the solution exists for all t ≥ 0 and remains in the interior for every finite t. Proof. Set yi := −log pi. Using |ϕi| ≤ 2 and

−⟨log p⟩ ≤ log K, y˙i ≤ −εyi + (2 + εlog K), whence yi(t) ≤ yi(0)e−εt + 2+εεlogK(1 − e−εt) for ε > 0, and yi(t) ≤ yi(0) + 2t for ε = 0. Thus yi(t) < ∞ for finite t.

| |
|---|

##### F.5 Barrier–Dominance (BD)

On the lower face pi = δ⋆,

p˙i = δ⋆ ϕi(p) + ε ⟨log p⟩ − log δ⋆ . By convexity of s  → slog s, the entropy face gap

|LK(δ⋆) := (1 − δ⋆)log<br><br>1 − δ⋆ (K − 1)δ⋆<br><br>> 0|
|---|

satisfies ⟨log p⟩ − log δ⋆ ≥ LK(δ⋆) on that face. Exact face test (necessary & sufficient). p˙i ≥ 0 on pi = δ⋆ iff

ϕi(p) + ε ⟨log p⟩ − log δ⋆ ≥ 0 for all p with pi = δ⋆.

Uniform sufficient templates. Using Lemma F.1:

√

εLK(δ⋆) ≥ Mϕ,∞ or εLK(δ⋆) ≥ Mϕ,2 (≤ 2

K),

√

√

where Mϕ,∞ := supp ∥ϕ(p)∥∞ ≤ 2Mγ,∞ ≤ 2 and Mϕ,2 := supp ∥ϕ(p)∥2 ≤ 2Mγ,∞

K ≤ 2

K. The first is a sharp ℓ∞ test; the second yields the tight threshold εLK(δ⋆) ≥ 2

√

√

K. Strict inequality implies strict interior invariance.

K and the convenient conservative form 4

Numerical note. As δ⋆ ↓ 0, Lf = Θ(1/δ⋆) and Clog = Θ(log(1/δ⋆)) deteriorate; discretizations should scale stepsizes accordingly.

##### F.6 Intra-class contraction

pk . Subtracting the log˙ p equations gives z˙ik = ϕi(p) − ϕk(p) − εzik = s gβ(log pi) − gβ(log pk) − εzik = sgβ′ (ξ) − ε zik,

For i,k with si = sk =: s, set zik := log p

i

for some ξ between log pi and log pk. Definition F.1 (Sharp thresholds).

β/4, ℓ0 ≤ 0, β 4 sech2 βℓ

β(ℓ − ℓ0) 2

β 4

sech2

(−gβ′ (ℓ)) =

copen := sup ℓ≤0

max

=

ℓ≤0

2 , ℓ0 > 0, and, under confinement to ∆Kδ −1

0

, cmax := sup

⋆

(−gβ′ (ℓ)) ≤ copen.

ℓ∈[log δ⋆,log(1−(K−1)δ⋆)]

- Theorem F.1 (Intra-class contraction). (i) For i,k ∈ C, |zik(t)| ≤ |zik(0)|e−εt. (ii) For i,k ∈ I, on the open simplex,

|zik(t)| ≤ |zik(0)|e−(ε−c

open)t iff ε > copen. Under confinement to ∆Kδ −1

the same holds with cmax replacing copen. Proof. For s = +1, gβ′ (ξ) ≤ 0 gives rate ε. For s = −1, dtd |zik| ≤ (c − ε)|zik| with c ∈ {copen,cmax}; Gr¨onwall gives sufficiency, and necessity follows by choosing data with −gβ′ (ξ0) ↑ c.

⋆

| |
|---|

Slope Condition (SC). We will often invoke the sufficient condition

|(SC) ε > β/4|
|---|

which implies ε > copen and hence contraction in both classes.

##### F.7 Cross-class envelopes, trimming sharpenings, and a static cap

For i ∈ C, j ∈ I, set zij := log p

pj . Then z˙ij = gβ(log pi) + gβ(log pj) − εzij =: h(t) − εzij.

i

Since gβ is decreasing and log px ≤ 0, we have gβ(log px) ≥ gβ(0) and gβ(log px) < 1. Variation of constants yields, for all t ≥ 0,

zij(t) ∈ z0e−εt + 2gβε(0)(1 − e−εt), z0e−εt + 2ε(1 − e−εt) , z0 := zij(0). (26)

If, in addition, p(t) ∈ ∆Kδ −1

, then log px ∈ [log δ⋆,0] and

⋆

2gβ(log δ⋆) ε

zij(t) ≤ z0e−εt +

(1 − e−εt). (27)

Independently, mass constraints on ∆Kδ −1

give the static cap

⋆

|zij(t) ≤ log<br><br>1 − (K − 1)δ⋆ δ⋆<br><br>(∀t ≥ 0).|
|---|

(28)

Lemma F.6 (Cap dominates a half-gap). For every K ≥ 2 and δ⋆ ∈ (0,1/K),

1 − δ⋆ (K − 1)δ⋆

1 − (K − 1)δ⋆ δ⋆

- 1

- 2

log

< log

.

2, which reduces to (K−1) 1−(K−1)δ 2−δ(1−δ) > 0 on (0,1/K); the function decreases from K − 1 at 0 to 0 at 1/K.

(K−1)δ⋆ < 1−(K−1)δ

Proof. Equivalently, 1−δ

⋆ δ⋆

⋆

| |
|---|

Compatibility under BD. Under the sharp ℓ∞ BD test εLK(δ⋆) ≥ Mϕ,∞ ≤ 2,

1 − δ⋆ (K − 1)δ⋆

1 − (K − 1)δ⋆ δ⋆

2gβ(0) ε ≤

2 ε ≤ LK(δ⋆) ≤ log

< 2 log

by Lemma F.6, so the asymptotic lower envelope in (26) lies strictly below the static cap (28). A stronger trimmed constant is available by replacing gβ(0) with g⋆ := gβ(log(1 − (K − 1)δ⋆)) in (26); a sufficient compatibility condition is

2g⋆ log1−(K−1)δ

ε ≥

.

⋆ δ⋆

##### F.8 Lyapunov structure and eventual trimming (under SC) Define

K

s

Gi(s) := si gβ(log s) − εlog s, Ψi(s) :=

Gi(u)du, L(p) :=

Ψi(pi).

δ⋆

i=1

The ODE rewrites as pure replicator: p˙i = pi Gi(pi) − G¯(p) , G¯(p) :=

j

pjGj(pj),

and satisfies the Lyapunov identity

d dtL p(t) =

K

pi Gi(pi) − G¯(p) 2 ≥ 0. (29)

i=1

Under (SC), G′i(s) = (sigβ′ (log s) − ε)/s < 0 for both classes, so each Ψi and hence L is strictly concave on the affine simplex.

- Proposition F.3 (Eventual trimming under (SC)). Assume (SC) and p(0) ∈ int∆K−1. There exist δ > 0 and T < ∞ (depending on K,M,N,β,ε,p(0)) such that p(t) ∈ ∆Kδ −1 for all t ≥ T. An explicit choice is:

ZU := max

2 ε

zij(0) , u := eZ

, max

U

i∈C,j∈I

, r := eZ

L

gβ(0) ε

, ZL :=

> 0,

and then, for some T large enough, r ≤ pi(t)/pj(t) ≤ u for all i ∈ C, j ∈ I, t ≥ T, which implies

|min<br><br>k<br><br>pk(t) ≥ δ :=<br><br>r u(N + Mr)<br><br>> 0 (∀t ≥ T).|
|---|

Sketch. Use the envelopes (26) to choose any ZL < liminf zij and ZU > supt zij(t). From pi ≤ upj and pi ≥ rpj, derive lower bounds on class masses and on the minimal coordinate (algebra as in the display).

| |
|---|

##### F.9 Two-level equilibrium: existence, uniqueness, and global convergence

A two-level equilibrium has p⋆i = LC for i ∈ C and p⋆j = LI for j ∈ I, with MLC + NLI = 1. Parameterize by the gap z := log(LC/LI) ≥ 0:

ez N + Mez

1 N + Mez

LI(z) =

, LC(z) =

.

At equilibrium, Gi(p⋆i ) ≡ const, equivalently

|gβ log LC(z) + gβ log LI(z) = εz.|
|---|

Define h(z) := gβ(log LC(z)) + gβ(log LI(z)) ∈ (0,2) and F(z) := h(z) − εz. Then F(0) = 2gβ(log(1/K)) > 0, and F(z) → −∞ as z → ∞ (since h is bounded). Differentiating,

h′(z) = gβ′ (log LC)NLI + gβ′ (log LI)(−MLC), |h′(z)| ≤ β/4, so under (SC) we have F′(z) ≤ β/4 − ε < 0 and thus:

- Lemma F.7 (Unique gap and quantitative bounds). Under (SC) there exists a unique z⋆ > 0 solving F(z) = 0.

Moreover

2 ε

h(0) ε + β/4 ≤ z⋆ ≤

h(0) ε − β/4

2gβ(0) ε ≤ z⋆ ≤

, h(0) = 2gβ log K1 .

,

- Theorem F.2 (Global convergence). Assume (SC). For any p(0) ∈ int∆K−1, the trajectory converges to the unique two-level equilibrium p⋆ with gap z⋆ from Lemma F.7. Proof. By Proposition F.3, p(t) enters and stays in a compact trimmed simplex for t ≥ T. On this compact set the drift is globally Lipschitz (Proposition F.1). The Lyapunov identity (29) and strict concavity of L under (SC) imply that the largest invariant set in {L˙ = 0} consists of equilibria, which are two-level; uniqueness of z⋆ then yields global convergence.

| |
|---|

Edge cases (no mixed preferences). If N = 0 (all si = +1), G′i(s) = (gβ′ (log s) − ε)/s ≤ −ε/s < 0 for any ε ≥ 0; the unique equilibrium is uniform and globally attractive. If M = 0 (all si = −1), uniqueness and global attraction of the uniform equilibrium hold provided ε > β/4.

Choosing a compatible floor. Given z⋆, set δ⋆ ≤ LI(z⋆) to ensure p⋆ ∈ ∆Kδ −1

. This does not obstruct BD since LK(δ⋆) → ∞ as δ⋆ ↓ 0.

⋆

#### G Dynamics on Coarse-Grained “Lumps”

Simplex, solution concept, and entropy map. Let the finite index set be S = {π1,...,πS} (S ≥ 2). The closed simplex is

∆S−1 := p ∈ [0,1]S :

π

pπ = 1 , int∆S−1 := {p ∈ ∆S−1 : min

pπ > 0}.

π

We work with Carath´eodory solutions p : [0,T] → ∆S−1 of

p˙(t) = p(t) ⊙ ϕ p(t) − εE◦ p(t) , ε ≥ 0, (SRCT) where ϕ : ∆S−1 → RS is centered, π pπϕπ(p) = 0, and

Eπ◦(p) := h(pπ) − pπ⟨log p⟩, h(x) := xlog x, ⟨log p⟩ :=

pπ log pπ.

π

- E◦ is continuous on ∆S−1; if pπ = 0, then (p ⊙ ϕ)π = Eπ◦(p) = 0, so faces are viable and the closed simplex is forward invariant.

Trim and feasibility. Fix δ⋆ ∈ (0,1/S] and the trimmed simplex ∆Sδ−1

:= {p ∈ ∆S−1 : pπ ≥ δ⋆ ∀π} (nonempty by choice of δ⋆).

⋆

##### G.1 Lumps

Let (Ck)K

k=1 be a partition of S into nonempty, disjoint lumps. For k = 1,...,KL define

L

KL

pπ log pπ, h¯ :=

mj.

qk :=

pπ, mk :=

pπ log pπ =

π

π∈Ck

π∈Ck

j=1

If qk > 0, write Ep|C

pπ log pπ so that mk = qk Ep|C

[log p] := (1/qk) π∈C

[log p].

k

k

k

- Lemma G.1 (Lump ODE). Every Carathe´odory solution of (SRCT) satisfies, for each k,

|q˙k =<br><br>π∈Ck<br><br>pπ ϕπ(p) − ε mk − qk h¯ .|
|---|

(30)

[log p] − h¯ . For qk = 0 the right-hand side vanishes by

pπ ϕπ(p) − εqk Ep|C

If qk > 0, equivalently q˙k = π∈C

k

k

continuity. Aggregation operator. Let A ∈ {0,1}K

L×S be the indicator matrix, Akπ = 1{π∈C

k}, so that q = Ap. Exact norms:

|∥A∥1→1 = 1, ∥A∥2→2 = √m∗, ∥A∥∞→∞ = m∗,<br><br>|
|---|

|Ck|. (31) In particular, aggregation is 1-Lipschitz in ℓ1: ∥Au − Av∥1 ≤ ∥u − v∥1.

m∗ := max

k

##### G.2 Technical facts used repeatedly

On ∆Sδ−1

:

⋆

##### • Mean-log bounds.

|−log S ≤ ⟨log p⟩ ≤ 1 − (S − 1)δ⋆ log 1 − (S − 1)δ⋆ + (S − 1)δ⋆ log δ⋆ ≤ 0.|
|---|

- (32)

• Entropy size. With E(p) := p ⊙ (log p − ⟨log p⟩1),

|∥E(p)∥1 ≤ 2log<br><br>1 δ⋆<br><br>.|
|---|

- (33)

• Replicator matrix bounds. Writing S(p) := diag(p) − pp⊤,

|∥S(p)∥2→2 ≤ 12, ∥S(p) − S(q)∥2→2 ≤ 3∥p − q∥2 .<br><br>|
|---|

- (34)

Centeredness gives p ⊙ ϕ = S(p)ϕ.

###### • Selection envelopes. For any domain D ⊆ ∆S−1 and lump Ck,

|π∈Ck<br><br>pπ ϕπ(p) ≤ qk Mϕ,∞(D) and ≤ qk Mϕ,2(D),|
|---|

with Mϕ,∞(D) := supp∈D ∥ϕ(p)∥∞, Mϕ,2(D) := supp∈D ∥ϕ(p)∥2.

(35)

##### G.3 Small-ε perturbation: trace and lump bounds

Assume on ∆Sδ−1

that

⋆

∥ϕ(p)∥2 ≤ Mϕ,2, ∥ϕ(p) − ϕ(q)∥2 ≤ Lϕ ∥p − q∥2. (36) By (34), for F0(p) := p ⊙ ϕ(p) = S(p)ϕ(p),

|∥F0(p) − F0(q)∥1 ≤ L(1)F ∥p − q∥1, L(1)F :=<br><br>√<br><br>S 2 1 Lϕ + 3Mϕ,2 .<br><br>|
|---|

(37)

- Theorem G.1 (Trace-level perturbation with exit-time qualification). Let pε,p0 solve p˙ε = F0(pε)−εE(pε) and

p˙0 = F0(p0) with pε(0) = p0(0) ∈ ∆Sδ−1

. Set τ∧ := inf{t > 0 : minπ pεπ(t) = δ⋆ or minπ p0π(t) = δ⋆}. Then for t ∈ [0,τ∧),

⋆

|∥pε(t) − p0(t)∥1 ≤<br><br>2ε log(1/δ⋆) L(1)F<br><br>eL<br><br>(1)<br><br>F t − 1 .|
|---|

Consequently, for any partition, ∥qε(t) − q0(t)∥1 ≤ ∥pε(t) − p0(t)∥1.

Forward-invariance templates. Let LS(δ) := (1 − δ)log(S1−−1)δ δ > 0. If on ∆Sδ−1

either

⋆

|εLS(δ⋆) ≥ 2Mϕ,∞ or εLS(δ⋆) ≥ 2Mϕ,2,|
|---|

(38)

then ∆Sδ−1

is forward invariant for (SRCT), and the bound in Theorem G.1 holds for all t ≥ 0.

⋆

##### G.4 Pure-score (ε = 0) lump dynamics

When ε = 0, Lemma G.1 reduces to q˙k = π∈C

k

pπ ϕπ(p).

##### G.4.1 STaR

Let C ⊂ S denote “correct” indices (M := |C| ≥ 1) and I := S \ C. Set ρ(p) := c∈C pc and S(2)(p) := c∈C p2c. The centered STaR field is

 

pπ − S(2)(p) ρ(p)

, π ∈ C,

ϕSTaRπ (p) =

defined when ρ(p) > 0.

S(2)(p) ρ(p)



−

, π ∈ I,

- Proposition G.1 (STaR lump ODE). For Sk,(2)C(p) := π∈C

k∩C p2π,

|q˙k =<br><br>Sk,(2)C(p) − qk S(2)(p) ρ(p)<br><br>.|
|---|

If Ci,Cj ⊂ C, then

d dt

log q

i

qj = ρ1 S

(2) i,C

qi − S

(2) j,C

qj . G.4.2 GRPO

Let√G −G 1.≥ The2 becenteredthe grouptwo-levelsize andfieldhGis: [0,1] → (0,∞) the GRPO characteristic (continuous), e.g. bounded by

ϕGRPOπ (p) =

(1 − ρ(p))hG(ρ(p)), π ∈ C, −ρ(p)hG(ρ(p)), π ∈ I.

For qk,C := π∈C

k∩C pπ define corr(Ck;p) := qk,C/qk (if qk > 0).

- Proposition G.2 (GRPO lump ODE).

|q˙k = hG ρ(p) qk corr(Ck;p) − ρ(p) .|
|---|

d dt

Hence

log q

qj = hG(ρ) corr(Ci;p) − corr(Cj;p) .

i

- G.4.3 DPO (sign-pure lumps) Fix labels sπ ∈ {±1} and a link gβ : R → (0,1) with gβ′ (ℓ) ∈ [−β/4,0) on [log δ⋆,0]. Define

γπ(p) := sπ gβ(log pπ), γ¯(p) :=

π

pπ γπ(p), ϕπ(p) := γπ(p) − γ¯(p).

Assume each lump Ck is sign-pure: sπ ≡ sk on Ck. Let

1 qk π∈C

Gk(p) :=

k

pπ gβ(log pπ), g¯(p) :=

KL

qj sj Gj(p) = γ¯(p).

j=1

Interpret qkGk := π∈C

pπ gβ(log pπ) so the right-hand side is well-defined even if qk = 0.

k

- Proposition G.3 (DPO lump ODE (sign-pure)).

|q˙k = qk sk Gk(p) − g¯(p) .|
|---|

If Ci = {πi} and Ck = {πk} with sπ

= sπ

=: s, then for zik := log(pπ

/pπ

),

i

i

k

k

|z˙ik = s gβ(log pπ<br><br>i<br><br>) − gβ(log pπ<br><br>k<br><br>) , |z˙ik| ≤ (β/4)|zik|.|
|---|

##### G.5 Entropy deviation envelopes for the lump term

For qk > 0 write wπ := pπ/qk on Ck and H(wk) := − π∈Ck wπ log wπ. Then

|mk = qk log qk + qk<br><br>π∈Ck<br><br>wπ log wπ ∈ qk log q<br><br>k<br><br>|Ck|, qk log qk ,|
|---|

- (39)

hence

||mk − qkh¯| ≤ qk max |log qk − h¯|, log q<br><br>k<br><br>|Ck| − h¯ .|
|---|

- (40)

On ∆Sδ−1

, the dimension-only bound

⋆

||mk − qkh¯| ≤ qk log<br><br>1 − (S − 1)δ⋆ δ⋆<br><br>|
|---|

is immediate from the log-domain [log δ⋆,log(1 − (S − 1)δ⋆)].

##### G.6 Open problems

(41)

Fix a partition of indices into correct C and incorrect I with sizes KC := |C| ≥ 0, KI := |I| ≥ 0 (K = KC +KI = S). For δ ∈ (0,1/K) define the trimmed simplex ∆Kδ −1 and the uniform face gap LK(δ) := (1−δ)log(K1−−1)δ δ > 0. The feasible band for ρ := c∈C pc is [KCδ, 1 − KIδ].

##### Face-wise entropy minima (at fixed ρ and pk = δ). For a fixed ρ and an incorrect face k ∈ I,

1 − δ − ρ KI − 1

ρ KC

Emin(I) (ρ) = (δ − 1)log δ + 1{K

I≥2} (1 − δ − ρ)log

C≥1} ρlog

+ 1{K

.

For a correct face k ∈ C,

ρ − δ KC − 1

1 − ρ KI

Emin(C) (ρ) = (δ − 1)log δ + 1{K

C≥2} (ρ − δ)log

I≥1} (1 − ρ)log

+ 1{K

.

In both cases Emin(·) (ρ) ≥ LK(δ) and the minima are attained by uniform allocation among active coordinates.

###### OP1 (sharp BD thresholds at trim δ). STaR. On incorrect faces, ϕk = −S(2)/ρ ≥ −ρ; inwardness at fixed ρ follows if −ρ + εEmin(I) (ρ) ≥ 0, hence

|ε(sufI)(δ;KC,KI) := max<br><br>ρ∈[KCδ, 1−KIδ]<br><br>ρ Emin(I) (ρ)<br><br>suffices.|
|---|

On correct faces, ϕk = (δ − S(2))/ρ ≥ (δ − Smax(2) (ρ,δ))/ρ with Smax(2) (ρ,δ) = δ2 + (ρ − δ)2, so

|ε(sufC)(δ;KC,KI) := max<br><br>ρ<br><br>max{0, Smax(2) (ρ,δ) − δ} ρEmin(C) (ρ)<br><br>suffices.|
|---|

The uniform sufficient threshold is εSTaRsuf := max{ε(sufI),ε(sufC)}. The above are exact in the special cases KC = 1 for incorrect faces and KC = 2 for correct faces.

GRPO. On correct faces the drift is inward for any ε ≥ 0. On incorrect faces, inwardness at fixed ρ is equivalent to −ρhG(ρ) + εEmin(I) (ρ) ≥ 0, hence the exact threshold

|εGRPOcrit (δ;KC,KI,G) = max<br><br>ρ∈[KCδ, 1−KIδ]<br><br>ρhG(ρ) Emin(I) (ρ)<br><br>.|
|---|

Iδ)√G−1

√G − 1/LK(δ) and εGRPOcrit ≤ (1−K

Useful bounds: εGRPOcrit ≤

KIδLK(δ) .

###### OP2 (DPO sensitivity to ε; gap and linear response). Assume ε > β/4. Then the SRCT flow admits a unique two-level interior equilibrium p⋆(ε) (all correct, resp. incorrect, coordinates equal). Let z⋆(ε) := log(p⋆c/p⋆i ) ≥ 0 satisfy

|h(z⋆) = εz⋆, h(z) := gβ(log LC(z)) + gβ(log LI(z)),|
|---|

with LI(z) := (KI + KCez)−1 and LC(z) := ezLI(z). Then:

|d dε<br><br>z⋆(ε) = −<br><br>z⋆(ε) ε − h′(z⋆(ε))<br><br>< 0, z⋆(ε) =<br><br>h(0) ε<br><br>+<br><br>h′(0)h(0) ε2<br><br>+ O(ε−3).|
|---|

Moreover, writing ℓπ := log p⋆π(ε) and dπ := ε − sπgβ′ (ℓπ) > 0,

|d dε<br><br>p⋆π = −p⋆π<br><br>ℓπ − a dπ<br><br>, a := ⟨p⋆,D−1ℓ⟩ ⟨p⋆,D−11⟩<br><br>, D := diag(dπ),|
|---|

and for any lump Ck,

d dε

qk⋆ = −

π∈Ck

ℓπ − a dπ

p⋆π

.

###### OP3 (DPO coarse-graining: closure errors). For a sign-pure lump Ck with weights wπ := pπ/qk, let

ℓ¯k := π∈C

###### wπ(log pπ − ℓ¯k)2, and H(wk) := − π∈Ck wπ log wπ. On ∆Sδ−1

###### wπ log pπ, σk2 := π∈C

set cmax := supℓ∈[logδ

⋆

k

k

⋆, 0](−gβ′ (ℓ)) ≤ β/4. Then

|Gk − gβ(log qk) ≤ cmax σk + cmax H(wk) (static closure error),|
|---|

and the exact log-ratio identity augments to

d dt

- qi

- qj

- qi

- qj

= siGi − sjGj − εlog

log

+ ε H(wi) − H(wj) ,

so that replacing Gk by gβ(log qk) incurs an error bounded by cmax(σi+σj +H(wi)+H(wj))+ε(H(wi)+H(wj)).

Remarks. (i) STaR requires KC ≥ 1 (else ρ ≡ 0). (ii) The BD templates (38) are sufficient (not necessary). (iii) The lump-level entropy term is not the gradient of a lump entropy; bounds (40)–(41) are the correct bridge.

All statements above are consistent with the SRCT model (SRCT), are valid on the closed simplex via E◦, and become uniform on ∆Sδ−1

under (36).

⋆

#### H Analysis of Stochasticity in SRCT

This appendix develops a concise, self–contained analysis of the stochastic dynamics induced by mini–batch sampling in SRCT. We (i) fix the domain and standing hypotheses, (ii) quantify global Lipschitz moduli and mini–batch noise statistics, (iii) derive ODE and diffusion limits under the correct scaling, (iv) analyze boundary behavior (unreflected vs. reflected models), (v) record uniform ellipticity on the tangent bundle, (vi) treat small centred bias via an exponential Lyapunov device, and (vii) provide algorithm–specific log–ratio SDEs.

- H.1 Domain, notation, and standing hypotheses Fix an integer K ≥ 2 and a design floor δ⋆ ∈ (0,1/K). The trimmed simplex is

K

∆Kδ −1

:= p ∈ [0,1]K :

pi ≥ δ⋆ .

pi = 1, min

⋆

i

i=1

All logarithms are natural; 0log 0 := 0. For x ∈ RK and a probability vector p, set ⟨x⟩p := i pixi and ⟨log p⟩ := i pi log pi. Vector norms ∥ · ∥2,∥ · ∥∞ are Euclidean and supremum norms, respectively. The tangent subspace is T := 1⊥.

##### Score field and SRCT drift. A centred score field ϕ : ∆Kδ −1

→ RK satisfies

⋆

K

pi ϕi(p) = 0 (∀p ∈ ∆Kδ −1

), (S1)

⋆

i=1

and the uniform regularity Mϕ := sup

∥ϕ(p)∥∞ < ∞, ∥ϕ(p) − ϕ(q)∥2 ≤ Lϕ ∥p − q∥2 (∀p,q ∈ ∆Kδ −1

). (S2–S3)

⋆

p

For ε ≥ 0, the SRCT drift is

Fi(p) := pi ϕi(p) − ε log pi − ⟨log p⟩ , F(p) ∈ T by (S1).

Write E(p) := p ⊙ log p − ⟨log p⟩1 and S(p) := diag(p) − pp⊤; then F(p) = S(p)ϕ(p) − εE(p).

##### H.2 Global Lipschitz moduli and envelopes

√

Define Λ(δ⋆) := 1 + log δ1

and Clog(K,δ⋆) := (2 +

K)Λ(δ⋆).

⋆

- Lemma H.1 (Entropy map modulus). For all p,q ∈ ∆Kδ −1

, ∥E(p) − E(q)∥2 ≤ Clog(K,δ⋆)∥p − q∥2.

⋆

- Lemma H.2 (Global Lipschitz drift). For all p,q ∈ ∆Kδ −1

,

⋆

∥F(p) − F(q)∥2 ≤ Lϕ + Mϕ + εClog(K,δ⋆) ∥p − q∥2.

Proofs (sketch). For Lemma H.1, write E(r) = G(r) − ⟨log r⟩r with G(r) := r ⊙ log r and use that |(xlog x)′| ≤ Λ(δ⋆) on [δ⋆,1] together with |⟨log p⟩ − ⟨log q⟩| ≤ Λ(δ⋆)∥p − q∥1 ≤ Λ(δ⋆)

√

K∥p − q∥2. Lemma H.2 follows from ∥p ⊙ (ϕ(p) − ϕ(q))∥2 ≤ Lϕ∥p − q∥2, ∥(p − q) ⊙ ϕ(q)∥2 ≤ Mϕ∥p − q∥2, and Lemma H.1.

| |
|---|

##### Size envelope. On ∆Kδ −1

one has x|log x| ≤ 1/e and −⟨log p⟩ ≤ log δ1

, hence

⋆

⋆

|Fi(p)| ≤ Mϕ + ε 1e + log δ1

⋆

(∀i). (42)

- H.3 Discrete mini–batch updates and noise statistics Given step size η > 0 and batch size B ∈ N, define

Nt B − pt ∈ T, pt+1 = pt + η F(pt) + ξt+1 ,

Nt ∼ Multinomial(B,pt), ξt+1 :=

optionally followed by Euclidean projection onto ∆Kδ −1

(which preserves mass).

⋆

- Lemma H.3 (Mini–batch noise). Conditionally on pt,

1 − ∥pt∥22 B ≤

E[ξt+1 | pt] = 0, E[∥ξt+1∥22 | pt] =

K − 1 BK

<

1 B

.

- H.4 Continuous–time limits (correct scaling) Let p(η) be the piecewise–linear interpolation. Set γη := η/B.

- Theorem H.1 (ODE and diffusion limits). Fix T > 0. As η ↓ 0 on [0,T]:

- (i) If γη → 0, then p(η) ⇒ p in C([0,T],RK), where p solves p˙ = F(p).
- (ii) If γη → γ ∈ (0,∞), then p(η) ⇒ p solving the Wright–Fisher–type SDE

dpi = Fi(p)dt + √γ √pi dWi − pi

K

√pk dWk , i = 1,...,K, (43)

k=1

with independent standard Brownian motions (Wk) and i pi(t) ≡ 1.

Sketch. Using Lemma H.3, the predictable quadratic variation of s<t/η η ξs+1 is η2E[∥ξ∥2] ∼ (η/B)t = γηt. Combine Lemma H.2 with a functional martingale CLT (Ethier–Kurtz) and Gr¨nwall–type estimates on the

compact domain ∆Kδ −1

.

| |
|---|

⋆

- H.5 Boundary behavior: entropy gap and BD conditions For y ∈ (0,1) define the face gap

K

pj log pj − log pi = (1 − y)log

Γ(y) := inf

p∈∆K−1 pi=y

j=1

1 − y (K − 1)y

. (44)

In particular LK(δ) := (1 − δ)log (K1−−1)δ δ > 0 for δ ∈ (0,1/K), and if pi = δ⋆ then ⟨log p⟩ − log pi ≥ LK(δ⋆). Barrier–Dominance (facewise). We say BD♯ holds if, for each i,

inf

p∈∆Kδ⋆−1 pi=δ⋆

ϕi(p) + ε ⟨log p⟩ − log pi > 0.

A convenient sufficient condition is

εLK(δ⋆) > Mϕ. (45)

- Proposition H.1 (Deterministic forward invariance). If BD♯ holds, then ∆Kδ −1

is forward invariant for p˙ = F(p) (Nagumo criterion). A conservative test is εLK(δ⋆) ≥ 2Mϕ.

⋆

Unreflected vs. reflected diffusions. Unreflected model. In (43), the one–dimensional marginal variance at a trimmed face pi = δ⋆ equals γ δ⋆(1 − δ⋆) > 0; hence a.s. non–attainability of the face cannot be deduced from inward drift alone. What holds are sharp high–probability non–exit bounds on finite horizons.

Reflected model. With orthogonal, mass–preserving reflection on each face of ∆Kδ −1

, solutions remain in the trim for all t by construction. On the compact domain with globally Lipschitz drift and uniformly elliptic tangent covariance, the reflected diffusion is strong Feller and irreducible, admits a unique invariant law, and exhibits exponential mixing.

⋆

- Theorem H.2 (Bandwise high–probability confinement (unreflected)). Fix a coordinate i and a band width η0 ∈ (0, 1 − Kδ⋆], and set ymax := δ⋆ + η0 and

Γband := inf

y∈[δ⋆,ymax]

Γ(y), µband := δ⋆ εΓband − Mϕ , σmax2 := γ ymax(1 − δ⋆).

If εΓband > Mϕ, then for any start Y0 = pi(0) ∈ [δ⋆,ymax], P(hit δ⋆ before ymax) ≤ exp − 2µ

band

σmax2 (Y0 − δ⋆) .

By the strong Markov property this yields an exponentially small (in η0 and γ−1) probability of ever touching the floor from any interior start.

- Theorem H.3 (Reflected diffusion: well–posedness and ergodicity). On ∆Kδ −1

with orthogonal reflection in

⋆

- H = { i pi = 1}, the SDE (43) admits a unique global strong solution, is strong Feller and irreducible, and has a unique invariant probability measure π∞ with

∥Pt(p,·) − π∞∥TV ≤ C e−κt (∀p ∈ ∆Kδ −1

, t ≥ 0).

⋆

##### H.6 Uniform ellipticity on the tangent bundle

Let Q(p) := γ(diag(p) − pp⊤) = γ S(p). For any p ∈ ∆Kδ −1

and v ∈ T,

⋆

γ 2 ∥v∥22. (46)

γ δ⋆ ∥v∥22 ≤ v⊤Q(p)v ≤

The upper bound is Popoviciu’s inequality; the lower bound uses i pivi2 ≥ δ⋆∥v∥22.

##### H.7 Gradient–field drifts and stationary laws

If ϕ = ∇Ψ and (S1) holds, π∞ (when it exists; e.g., Theorem H.3) is characterized as the unique Neumann solution of the stationary Fokker–Planck equation associated with (43). The naive Gibbs ansatz ∝ exp{−2γ−1(Ψ−εH)}

fails in general: inserting U = 2γ−1(Ψ − εH) into the reversibility identity F = 12 (divT Q) − 12 Q∇TU gives F = −2F unless F ≡ 0.

##### H.8 Small centred bias: concentration toward the fittest face

Let δ ∈ RK satisfy i δi = 0 and set δmax := maxi δi, S := {i : δi = δmax}, I := Sc, and the selection gap γδ := δmax − maxi∈I δi > 0 (if I ̸= ∅). The biased drift is

Fiδ(p) := pi ϕi(p) + δi −

pjδj − ε log pi − ⟨log p⟩ .

j

Exponential Lyapunov device (reflected model). Let m(p) := j δjpj and V (p) := j pj(δj − m(p))2 (variance of δ under p). For λ > 0 define U(p) := eλm(p).

- Lemma H.4 (Lyapunov inequality). For the reflected diffusion with generator Lδ and any p ∈ ∆Kδ −1

,

⋆

LδU(p) ≥ U(p) λV (p) − λ∥δ∥∞ Mϕ + εClog .

In particular, with λ := 2∥δ∥∞(Mϕ + εClog) −1, LδU ≥ U λV − 21 .

Proof. ∇U = λU δ, ∇2U = λ2U δδ⊤; the diffusion contribution is non–negative. For the drift, use j pjδj(δj − m) = V and the envelopes j pj|ϕj| ≤ Mϕ, j pj|log pj − ⟨log p⟩| ≤ Clog.

| |
|---|

- Theorem H.4 (Stationary concentration near the fittest face). Let π∞ be the invariant law of the reflected biased diffusion. Then

e2λ∥δ∥∞ 2λ

1 2∥δ∥∞(Mϕ + εClog)

Eπ

[V ] ≤

with λ =

.

∞

Since V (p) ≥ γδ2 L(p) 1−L(p) with L(p) := i∈I pi, this implies the symmetric band estimate, for any θ ∈ (0, 12],

π∞ θ ≤ L(p) ≤ 1 − θ ≤

e1/(M

ϕ+εClog) ∥δ∥∞(Mϕ + εClog) γδ2 θ(1 − θ)

.

Remark (no fixation under a positive floor). If δ⋆ > 0 then i∈I pi(t) ≥ |I|δ⋆ for all t; thus one has concentration toward (not fixation on) the fittest face. A bona fide fixation statement appears only in the

vanishing–floor limit δ⋆ ↓ 0.

- H.9 Log–ratio SDEs (algorithm–specific) For zij := log(pi/pj), Itˆ’s formula applied to (43) yields the exact identity

dzij = ϕi(p) − ϕj(p) dt − εzij dt −

γ 2

1 − pi pi −

1 − pj pj

dt + √γ

dWi √pi −

dWj √pj

. (47)

GRPO (within–class). If all correct traces share the same centred score, ϕi = ϕj within the class, then (47) reduces to

dzij = −εzij dt −

γ 2

1 − pi pi −

1 − pj pj

dt + √γ

dWi √pi −

dWj √pj

.

STaR (within–class). If ϕi − ϕj = (pi − pj)/ρ with ρ := c∈C pc, then

dzij =

pi − pj ρ − εzij dt −

γ 2

1 − pi pi −

1 − pj pj

dt + √γ

dWi √pi −

dWj √pj

.

On ∆Kδ −1

⋆

one has |pi − pj|/ρ ≤ 1−(K−1)δ

⋆

|C|δ⋆ |zij|.

DPO (same–sign pair). With si ∈ {±1} and ϕi(p) = si gβ(log pi) − k pkskgβ(log pk), gβ′ (x) ∈ [−β/4,0); for i,k with si = sk and pi ≈ pk,

dzik ≈ sgβ′ (ξ) − ε zik dt + (Itoˆ & noise as in (47)). Intra–class log–ratios contract if ε > sup(−gβ′ ) (e.g. ε > β/4).

- H.10 Regime dictionary (concise)

Let r := σ2/λeff with σ2 := γ the diffusion variance scale and λeff a local contraction modulus of F on T (for log–ratios, λeff ≳ ε). Under BD♯:

- • r ≪ 1 (low noise): tight interior concentration; Var(zij) = O(σ2/ε).
- • r ≍ 1 (balanced): moderate interior spread; unique invariant law.
- • r ≫ 1 (noise–dominated but interior): broad interior law; faces are still repelling.

If BD♯ fails, boundary approach and absorption may occur; interior concentration statements do not apply.

Summary. On the trimmed simplex, the SRCT drift is globally Lipschitz with an explicit modulus; mini–batch noise is centred with variance O(1/B). The correct continuous–time limits are the ODE (η/B → 0) and a Wright–Fisher–type diffusion (η/B → γ). The entropy face gap LK(δ⋆) quantifies inward normal speed; BD♯ yields ODE invariance and, for the unreflected SDE, high–probability confinement on finite horizons; the reflected diffusion is strictly invariant and exponentially ergodic. A small centred bias admits an exponential Lyapunov control that quantifies stationary concentration toward the fittest face. Exact log–ratio SDEs provide algorithm–specific envelopes (GRPO, STaR, DPO).

#### I Kernel Design Strategies for SRCT

This appendix gives a self–contained, concise treatment of kernel design and analysis for SRCT. Part §I.1 establishes an exact two–level stationarity condition, curvature (uniqueness/interiority), a tight log–ratio envelope with a dynamic floor, exponential convergence rates, a uniform suppression guarantee, and a block–constant PSD construction that realizes a prescribed class gap with controlled norms. Part §I.2 turns to practically learned kernels, including a gated effective kernel, exact suppression ratios, a support–function identity that quantifies diversity pressure, and an explicit global Lipschitz modulus for the SRCT drift.

##### Setting, notation, and standing assumptions. Let S = {π1,...,πS}, S ≥ 2, and ∆S−1 := {p ∈ [0,1]S :

- S i=1 pi = 1}. All logs are natural; 0log 0 := 0. Fix a partition S = C ∪ I with C ∩ I = ∅, sizes M := |C| ≥ 1,

N := |I| = S − M, and utilities Ui := 1{i∈C} ∈ {0,1}. Kernels are symmetric PSD: K = K⊤ ⪰ 0. Vector norms ∥ · ∥2,∥ · ∥∞; operator norms ∥A∥2→2 (spectral), ∥A∥∞→∞ := maxi j |Aij|, ∥A∥max := maxi,j |Aij|. Let

- T := 1⊥ (tangent subspace) and ΠT := I − S1 11⊤. SRCT objective, Shahshahani flow, and gauge. For λ,β ≥ 0 and entropy strength A > 0 define

S

J(p) := U⊤p − λβ p⊤Kp + AH[p], H[p] := −

pi log pi.

i=1

Variational derivative (on int∆S−1):

δ J δpi

Fi(p) =

= Ui − 2λβ (Kp)i − A(1 + log pi), F¯(p) :=

j

pjFj(p).

The Shahshahani (replicator) flow is

p˙i = pi Fi(p) − F¯(p) ,

i

p˙i = 0.

Adding a constant to F leaves the vector field invariant (gauge invariance); thus the “+1” in −A(1 + log pi) can be absorbed into the KKT multiplier at stationarity.

##### I.1 Idealized Kernel for a Two–Level Equilibrium Two–level target. Fix δ⋆ ∈ (0,1) with Nδ⋆ < 1 and set

1 − Nδ⋆ M

p⋆i := δ⋆ (i ∈ I), p⋆c =: pC :=

> 0 (c ∈ C),

and write VC := (Kp⋆)c (all c ∈ C), VI := (Kp⋆)i (all i ∈ I).

- Proposition I.1 (KKT ⇐⇒ classwise constancy + gap). Under the two–level ansatz above, p⋆ is a stationary point of the Shahshahani flow if and only if

- (i) Classwise constancy: (Kp⋆)c ≡ VC for all c ∈ C and (Kp⋆)i ≡ VI for all i ∈ I.

- (ii) Gap identity:

pC δ⋆

1 − 2λβ (VC − VI) − Alog

= 0.

Proof. Subtract the KKT equations for two indices in the same class to force classwise constancy; subtract a correct–incorrect pair and use Uc − Ui = 1 and log p⋆c − log p⋆i = log(pC/δ⋆) to obtain the gap. The converse is immediate by inspection.

| |
|---|

##### Curvature, strict concavity, uniqueness, interiority. Let κT := λmin (ΠTKΠT)|T ≥ 0. For any v ∈ T,

vi2 pi − 2λβ v⊤Kv ≤ −(A + 2λβ κT)∥v∥22.

⟨∇2 J(p)v,v⟩ = −A

i

Hence J is A–strongly concave on the affine simplex; in particular, the maximizer is unique and (by the steepness of AH[p]) interior.

##### Log–ratio dynamics, operator–norm envelope, dynamic floor. Let zij := log p

pj . Along trajectories, z˙ij = (Ui − Uj) − 2λβ (Kp)i − (Kp)j − Azij.

i

For all p ∈ ∆S−1 and i ̸= j,

(Kp)i − (Kp)j = (Ki· − Kj·)⊤p ≤ ∆K, where one may take any of the following (use the tightest available):

√

∥Ki· − Kj·∥∞ . With B♯ := |Ui − Uj| + 2λβ ∆K ≤ 1 + 2λβ ∆K, variation of constants yields

2∥K∥2→2, 2∥K∥∞→∞, 2∥K∥max, max

∆K ∈

i̸=j

B♯ A

(1 − e−At). Let

|zij(t)| ≤ |zij(0)|e−At +

B♯ A

, δ := S−1e−M

|zkℓ(0)|,

M♯ := max max

.

♯

k̸=ℓ

eM

♯

, so the ODE is globally well–posed and ∆δ := {p ∈ ∆S−1 : mini pi ≥ δ} is forward–invariant.

Then, for all t ≥ 0 and all i, δ ≤ pi(t) ≤

S

Exponential convergence. Let a(p) := F(p) − ⟨p,F(p)⟩1. Along trajectories, dtd J(pt) = i pi ai(pt)2 ≥ δ∥a(pt)∥22 on ∆δ. Since J is A–strongly concave on the affine simplex, J(p⋆) − J(p) ≤ 21A∥a(p)∥22. Therefore, for all t ≥ 0,

J(p⋆) − J(pt) ≤ J(p⋆) − J(p0) e−2Aδt, ∥pt − p⋆∥2 ≤ A2 J(p⋆) − J(p0) e−Aδt.

Moreover, since −∇2 J(p) ⪰ Adiag(1/p), J is A–strongly concave in the Shahshahani metric gp(u,u) = i u2i/pi, and the Riemannian PL inequality with the Lyapunov identity gives the δ–free rate

J(p⋆) − J(pt) ≤ J(p⋆) − J(p0) e−2At.

Stationary structure and uniform suppression. At any equilibrium p⋆, subtracting KKT equations with the same utility yields, for Ua = Ub,

- p⋆a

- p⋆b

2λβ A

(Kp⋆)a − (Kp⋆)b . For c ∈ C, i ∈ I,

= −

log

p⋆i p⋆c

1 A

1 − 2λβ (Kp⋆)c − (Kp⋆)i . A p–independent sufficient condition ensuring p⋆i < p⋆c for all such pairs is

= −

log

|2λβ ∆K < 1|
|---|

(use any ∆K bound above; the ℓ∞ row–difference is tight).

##### Block–constant kernels: PSD, norms, gap realization, low–norm choice. Consider

 

κCC, i,j ∈ C, κII, i,j ∈ I, κCI, otherwise.

Kij =



Let B := (κ

CC κCI

κCI κII ) and T : R2 → RS, T(a,b) = a1C + b1I, so K = TBT⊤ and rank(K) ≤ 2. Then K ⪰ 0 ⇐⇒ B ⪰ 0, i.e., κCC ≥ 0, κII ≥ 0, κCCκII ≥ κ2CI. Norm controls: ∥K∥2→2 ≤ max{M,N}∥B∥2→2 and ∥K∥∞→∞ = max{M|κCC| + N|κCI|, M|κCI| + N|κII|}. With the two–level p⋆,

(Kp⋆)c − (Kp⋆)i = (κCC − κCI)(1 − Nδ⋆) + (κCI − κII)Nδ⋆, so the gap identity of Proposition I.1 becomes

1 − Alog(pC/δ⋆) 2λβ

(1 − Nδ⋆)(κCC − κCI) + Nδ⋆(κCI − κII) =

=: X.

A low–norm constructive choice sets κCI = 0 and then

X + Nδ⋆ κminII 1 − Nδ⋆

X Nδ⋆

κminII = max 0, −

(N ≥ 1), κCC =

,

minimizing ∥K∥∞→∞ = max{MκCC, NκII} under PSD. Edge case N = 0: the gap is void; maximizing −λβ p⊤Kp + AH[p] yields a unique interior solution for A > 0.

##### I.2 Practical Design with a Learnable Semantic Kernel

Gated effective kernel and objective. Let ksem = ksem⊤ ⪰ 0 be a learnable semantic kernel and R ∈ {0,1}S a binary verifier with C = {i : Ri = 1}, I = {i : Ri = 0}. Define the effective kernel

Keff := Diag(R)ksem Diag(R) ⪰ 0. Consider the objective

J (p) = U⊤p + λ α H[p] − β p⊤Keffp , λ,α,β ≥ 0, and let the effective entropy coefficient be

εtot := εbase + λα, εbase > 0. The SRCT flow uses the score ϕi(p) = Ui − 2λβ (Keffp)i and reads

p˙i = pi ϕi(p) − ϕ¯(p) − εtot pi log pi − ⟨log p⟩ , ϕ¯(p) :=

pjϕj(p), ⟨log p⟩ :=

j

j

pj log pj.

Stationary points p⋆ ∈ int∆S−1 satisfy the KKT system

Ui − 2λβ (Keffp⋆)i − εtot 1 + log p⋆i = λ0, with the “+1” and λ0 eliminated by taking differences. Incorrect suppression and equalization among correct traces. Since Keff(i,·) ≡ 0 for i ∈ I, (Keffp⋆)i = 0 and, for any c ∈ C,

|p⋆i p⋆c<br><br>= exp −<br><br>1 − 2λβ (Keffp⋆)c εtot<br><br>.|
|---|

Thus strong suppression (p⋆i ≪p⋆c) is promoted by small εtot and moderate λβ (Keffp⋆)c. For a,b ∈ C,

- p⋆a

- p⋆b

= 2λβ (Keffp⋆)b − (Keffp⋆)a ,

εtot log

so larger εtot enhances equalization when the correct–side kernel averages are close.

##### Support–function identity (diversity pressure). For any A ∈ RS×S and distinct i,j,

(Ai· − Aj·)⊤p = ∥Ai· − Aj·∥∞.

(Ap)i − (Ap)j = sup

sup

p∈∆S−1

p∈∆S−1

(Proof: ∆S−1 is the convex hull of basis vectors; the support function in direction a equals maxk ak; take absolute values.)

Applying this to A = Keff shows that the maximal instantaneous disparity of kernel averages across two correct indices is exactly the ℓ∞ row–difference; when ksem is semantically coherent, this term is larger across distinct semantic lumps, enforcing diversity via the −β p⊤Keffp penalty.

##### Global Lipschitz modulus of the SRCT drift on a trimmed simplex. Let ∆Sδ−1

:= {p ∈ ∆S−1 : pi ≥ δ⋆ ∀i} and Λ(δ⋆) := 1 + log(1/δ⋆). Write S(p) := diag(p) − pp⊤ and E(p) := p ⊙ log p − ⟨log p⟩1 , so the drift is F(p) = S(p)ϕ(p) − εtotE(p) with ϕ(p) = U − 2λβ Keffp. On ∆Sδ−1

⋆

,

⋆

∥S(p)∥2→2 ≤ 21, ∥S(p) − S(q)∥2→2 ≤ 3∥p − q∥2, L(2)ϕ := 2λβ ∥Keff∥2→2, ∥ϕ(p)∥2 ≤

√

M + 2λβ ∥Keff∥2→2 =: Mϕ,2, ∥E(p) − E(q)∥2 ≤ Λ(δ⋆)(2 +

√

S)∥p − q∥2. Combining,

|∥F(p) − F(q)∥2 ≤ 12 L(2)ϕ + 3Mϕ,2 + εtot Λ(δ⋆)(2 +<br><br>√<br><br>S) ∥p − q∥2.|
|---|

Hence the ODE is globally Lipschitz on ∆Sδ−1

with an explicit modulus.

⋆

Tuning guidance (concise). Smaller εtot (i.e., smaller λα given εbase) yields exponentially stronger incorrect suppression but weaker equalization; larger εtot does the opposite. The coefficient λβ regulates semantic diversity pressure via Keff and should be chosen to spread mass across genuinely distinct correct lumps without excessively penalizing semantically coherent high–utility traces.

##### Design–to–guarantee checklist (explicit constants).

- 1. Target & gap. X =

1 − Alog(pC/δ⋆) 2λβ

with pC =

1 − Nδ⋆ M

.

- 2. Kernel. Choose symmetric PSD K realizing the gap; for block–constant K, the low–norm choice is κCI = 0

and κII = κminII , κCC =

X + Nδ⋆ κminII 1 − Nδ⋆

.

- 3. Curvature (uniqueness/interiority). Ensure A > 0 (then the maximizer is unique and interior).
- 4. Log–ratio floor. With any ∆K option above, set B♯ = 1 + 2λβ ∆K, M♯ = max{maxi̸=j |zij(0)|, B♯/A},

δ = S−1e−M

♯

; then pi(t) ∈ [δ,eM

♯

/S] for all t.

- 5. Rates. Euclidean–PL on ∆δ: ∥pt−p⋆∥2 ≤ A2 J(p⋆) − J(p0) e−Aδt; metric–PL (δ–free): J(p⋆)− J(pt) ≤ ( J(p⋆) − J(p0))e−2At.

- 6. Suppression. A uniform sufficient condition for p⋆i < p⋆c is 2λβ ∆K < 1.

Notation hygiene and edge cases. Symbol δ⋆ denotes the prescribed target floor in the two–level ansatz, while δ = S−1e−M

is the dynamic floor from the log–ratio envelope. When N = 0, the cross–class gap is void; all curvature, floor, and convergence statements remain valid with A > 0.

♯

#### J Insight Experiments

This appendix complements the main paper with simple experiments to validate parts of the theory. Unless stated otherwise: lines are means across five seeds and ribbons show ±1s.d; the vertical line at step 200 indicates the event–detection smoothing floor. Metrics used throughout are the entropy H = − i pi log pi, fixation index

Fix = i p2i, cluster Gini (inequality over masses of the three correct–strategy clusters), incorrect mass (total probability on incorrect traces), and the objective proxy

Jp := utility mass + λα H − λβ p⊤Keffp.

##### J.1 Experimental Implementation and Reproducibility

Synthetic trace universe. All experiments share the same finite “trace universe” with S = 12 traces. Eight traces are correct and partitioned into three semantic clusters (strategies) A,B,C of sizes 3,3,2; the remaining four are incorrect. Let C ⊂ {1,...,12} be the set of correct traces and I = {1,...,12}\C the incorrect traces. A policy is a probability vector p ∈ ∆S−1, with numerical clipping pi ← max(pi,10−12) before any log is evaluated. Cluster membership is used only for analysis and, in Study B, for the creativity kernel.

Verifier and rewards. Correctness is deterministic: U(i) = 1 for i ∈ C, U(i) = 0 for i ∈ I. In Study B, we additionally use base rewards r(i) = 1.0 for i ∈ C and r(i) = 0.2 for i ∈ I.

Mini-batch sampling and noise. Each update step draws a multinomial mini-batch of size B from the current policy p, yielding counts n ∼ Multinomial(B,p) and empirical frequencies pˆ = n/B. All fitness/payoff computations that require batch statistics use pˆ (not the full p) so that finite-batch noise is the only source of stochasticity.

##### Common metrics and event detection. At fixed intervals we log:

- • Entropy: H[p] = − i pi log pi.
- • Fixation index: Fix = i p2i (monoculture → 1).
- • Cluster masses: mA,mB,mC (probability within each correct cluster).
- • Cluster inequality: Gini(mA,mB,mC).
- • Incorrect mass: Minc = i∈I pi.
- • Objective proxy (Study B): Jp = i∈C pi + λα H[p] − λβ p⊤Keffp, where Keff is the gated creativity kernel described below.

Events are detected on 50-step moving averages with a 200-step floor: (i) fixation (STaR/GRPO) when maxi pi ≥ 0.75 and max{mA,mB,mC} ≥ 0.9; (ii) homogenization (DPO) when the smoothed cluster Gini ≤ 0.10 and all nonzero cluster masses ≥ 0.15. Unless noted, runs use T = 5000 steps and five seeds {101,202,303,404,505}; lines show seed means and ribbons ±1 s.d.

Theoretical (replicator) update used in Studies A and A+. All “theory” tracks use the same exponentiated-gradient (replicator) step

p˜i ← pi exp η [ϕi − εlog pi] , p ← p/˜ ∥p˜∥1, with learning rate η = 0.15 and barrier ε ∈ {0,3 × 10−4}. The method-specific fitness ϕi is:

STaR: ϕi = pˆi/ρˆ if i ∈ C, else 0, ρˆ =

pˆc;

c∈C

GRPO: ϕi = 1{i ∈ C}; DPO: ϕi = −log max(ˆpi,10−12) if i ∈ C, else 0.

Algorithm-faithful (procedural) updates used in Study A+. In parallel to the “theory” track, we run algorithm-faithful procedures on logits θ with p = softmax(θ):

- • STaR (sequential reinforcement). Sample up to L traces i.i.d.; on the first correct c apply θ ← θ + ηstar(ec − p). If none is correct, no-op that step. L ∈ {16,64} co-varies with B.

- • GRPO (group REINFORCE with baseline). Sample a group of size m; with centered advantages aj = rj − r¯, θ ← θ + ηgrpom j aj(ei

j − p); m ∈ {8,16,32} depending on B.

- • DPO (pairwise preferences, Davidson ties). For pairs (i,j) drawn from the batch, compute the Davidson

log-likelihood with tie parameter ν and take a gradient step θ ← θ + ηdpo∇θℓ. We use batched pairs and adaptive scaling to match one-step norms to the theory track.

For each method and B, ηproc (and, for DPO, pairs-per-step and ν) is calibrated on a small set of anchor states to maximize the mean cosine between one-step ∆p from the procedural and theory tracks while keeping the norm ratio close to 1.

DCR objective and kernel (Study B). Study B augments a GRPO-like base with a diversity energy λ(αH[p]− β Q[p]), and folds the entropic term into the effective barrier: ε ← εbarrier + λα with εbarrier = 10−4. The gated kernel is

Keff = R Ksem R, Rii = 1{i ∈ C}, and Ksem(i,j) = 1 if i,j are correct and in the same cluster, else 0. The fitness used in the replicator step is ϕi = r(i) − 2λβ (Keffpˆ)i,

so that the quadratic penalty −λβ p⊤Keffp discourages concentration on similar correct traces only. We sweep α ∈ {0.02,0.05,0.10}, β ∈ {0.10,0.25,0.50,0.75}, with λ = 1, B = 128, η = 0.15. Two ablations are reported: Entropy-only (β = 0) and Ungated (apply K to all traces).

Time horizons, seeds, and smoothing. Unless stated otherwise: T = 5000 steps; seeds {101,202,303,404,505}; 50-step moving averages and a 200-step event floor are used for all event times and overlaid ribbons.

We run all experiments on a single NVIDIA RTX 6000 with 49GB of VRAM.

##### J.2 Strategy–simplex overview (Fig. 1)

- Figure 1 provides a qualitative, distributional view of training on the three–strategy simplex (clusters A/B/C): STaR flows to a corner (monoculture), GRPO meanders along a neutral manifold before noise–driven fixation, DPO equalizes mass within the correct set, and DCR converges to a unique interior equilibrium with multi–strategy support. These panels summarize the high–level modes that are quantitatively confirmed in the subsequent figures.

J.3 Study A: scalar–objective dynamics (Fig. 2)

- Figure 2 aggregates the time evolution of H, Fix, cluster Gini, and incorrect mass for STaR, GRPO, and DPO. STaR collapses essentially immediately (H → 0, Fix → 1); GRPO exhibits slow, batch–size–dependent drift (median fixation ≈4.7k steps at B=16; no fixation by 5k at B=64); DPO homogenizes correct strategies early while maintaining zero incorrect mass.

##### J.4 Study B: overlays and alignment diagnostics (Figs. 3, 4, 5)

The overlays in Fig. 3 compare the replicator “theory” track and the algorithm–faithful procedural track for a common seed: STaR nearly coincides; GRPO shows small–magnitude neutral steps; DPO matches event timing but sustains higher entropy due to paired–comparison (Davidson ties) and the θ →p geometry.

Per–step alignment in Fig. 4 shows (i) high sign agreement for DPO with modest cosine (geometry mismatch), (ii) near–neutral GRPO behavior, and (iii) high STaR cosine with zero event–gap. Batch–size summaries in Fig. 5 confirm that, despite low cosines at larger B, the one–step JS divergence shrinks and event timing synchronizes.

##### J.5 Study C: DCR phase diagrams (Fig. 6) and ablations (Fig. 7)

- Figure 6 sweeps (α,β) and reports: incorrect mass, minimum cluster mass, between–seed JSD, and correct mass. A broad band achieves near–zero incorrect mass, full coverage, and negligible between–seed JSD—an empirical signature of a unique, interior, diverse equilibrium.

Cluster C

STaR

GRPO

DPO DCR

stable

Cluster A Cluster B

- Figure 1: Strategy–simplex dynamics. Representative trajectories of cluster masses (mA,mB,mC) under STaR, GRPO, DPO, and DCR. STaR collapses to a vertex; GRPO drifts along the face; DPO equalizes on the face; DCR reaches a stable interior point retaining all clusters. Early (step 200) and late (step 5000) states are marked.

- Figure 7 compares DCR, Entropy–only, and Ungated. While coverage saturates at 3 for all, DCR reduces kernel energy (structured diversity) and maintains large positive safety margins; Entropy–only lacks targeted distinctiveness; Ungated penalizes incorrect–incorrect similarity, degrading safety despite larger proxy gains.

J.6 Objective and safety trajectories (Fig 8)

- Figure 8 shows trajectories: DCR reaches a stable interior solution with safety ≳ 0.93; Entropy–only has safety fixed at 1 (no kernel); Ungated converges at much lower safety (≈ 0.48).

##### J.7 Safety–margin distribution (Fig. 9)

The histogram in Fig. 9 reports the minimum safety margin attained along training within the DCR band; all runs remain strictly positive (worst case ≈ 0.267), empirically validating the tuning rule that kernel pressure must not overwhelm the unit utility signal.

Entropy H[p] (nats)

Fixation index F

Cluster Gini

###### Incorrect mass

1.0

| | | |B=64|
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
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

10 3 200 [200 200]

B=16

200 [200 200]

200 [200 200]

200 [200 200]

0.6

1.5

0.8

10 10

0.5

STaR

10 17

1.0

0.6

0.4

10 24

0.5

0.4

0.3

10 31

0.0

0.2

Entropy H[p] (nats)

Fixation index F

Cluster Gini

###### Incorrect mass

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | |4707 [470|7<br><br>|70|7]|
| | | | | | | | | |
| | | | | | | | | |

| | | | | |4707 [470|7<br><br>|70|7]|
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

10 3 4707 [4707 4707]

1.75

0.5

4707 [4707 4707]

0.6

10 10

0.4

1.50

GRPO

10 17

0.3

1.25

0.4

10 24

0.2

1.00

10 31

0.1

0.75

0.2

###### Entropy H[p] (nats)

Fixation index F

Cluster Gini

Incorrect mass

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

200 [200 200]

2.0

200 [200 200]

10 9 200 [200 200]

0.35 200 [200 200]

200 [200 200]

0.3

10 19

0.30

1.8

###### DPO

10 29

0.25

0.2

1.6

10 39

0.20

200 [200 200]

10 49

0.15

1.4

0.1

0 1000 2000 3000 4000 5000

0 1000 2000 3000 4000 5000

0 1000 2000 3000 4000 5000

0 1000 2000 3000 4000 5000

Step

Step

Step

Step

###### Figure 2: Study A: collapse modes. Rows: STaR (top), GRPO (middle), DPO (bottom). Columns: entropy H, fixation index Fix, cluster Gini, incorrect mass (log scale). STaR deterministically fixates; GRPO drifts with speed increasing at smaller batch; DPO equalizes among correct traces while keeping incorrect mass at 0.

Entropy H[p] (nats)

Cluster Gini

| | |Theory Procedural| |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | |Mean | entropy| = 0.0<br><br>|74|
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | |Mean | Gini| = 0.008<br><br>|
| | | |
| | | |

1.50

0.6

1.25

0.5

1.00

Value

STaR

0.4

0.75

0.3

0.50

0.2

0.25

0.1

0.00

0.45

1.8

0.40

0.35

1.6

0.30

GRPO

Value

1.4

0.25

0.20

1.2

Mean | entropy| = 0.461 Correct/batch = 0.87 ± 0.11

Mean | Gini| = 0.106

0.15

1.0

0.10

0.18

2.3

0.16

0.14

2.2

Value

0.12

DPO

2.1

0.10

0.08

Mean | entropy| = 0.178

Mean | Gini| = 0.035

2.0

0.06

0 1000 2000 3000 4000 5000

0 1000 2000 3000 4000 5000

Step

Step

###### Figure 3: Theory vs. procedural overlays (single seed). Entropy and cluster–Gini trajectories for STaR, GRPO, and DPO. Procedural updates (sequential STaR, group REINFORCE, Davidson–ties DPO) track theory closely in events; instantaneous directions differ most for DPO.

DPO alignment (solid: Euclidean, dotted: Shahshahani)

Sign agreement over time

Event gap (procedural theory)

0.325

| | |B=16 B=64 B=128| |
|---|---|---|---|
| | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| |t -1<br><br>| | | | |
| | | | | | |
| | | | | | |

5000

1.0

0.300

50

t(proceduraltheory)

4000

0.8

0.275

Signagreement

JSdivergence

0

t 0

3000

0.6

0.250

Cosine

0.225

50

2000

0.4

0.200

1000

0.2

100

0.175

0

0.0

0.150

GRPO alignment (solid: Euclidean, dotted: Shahshahani)

Sign agreement over time

Event gap (procedural theory)

1.0

| | | | |
|---|---|---|---|
| | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

5000

1.0

0.16

t(proceduraltheory)

0.14

0.8

4000

0.8

Signagreement

0.12

JSdivergence

0.6

3000

0.6

Cosine

0.10

2000

0.4

0.4

0.08

0.06

1000

0.2

0.2

0.04

0

0.0

0.0

STaR alignment (solid: Euclidean, dotted: Shahshahani)

Sign agreement over time

Event gap (procedural theory)

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| |t 0<br><br>| | | | |
| | | | | | |
| | | | | | |

5000

1.0

0.25

0.04

t(proceduraltheory)

4000

0.8

Signagreement

0.02

0.20

JSdivergence

3000

0.6

Cosine

0.00

0.15

t 0

2000

0.4

0.02

0.10

1000

0.2

0.04

0.05

0

0.0

0 1000 2000 3000 4000 5000

0 1000 2000 3000 4000 5000

20 30 40 50 60

Step

Step

Batch size

###### Figure 4: Alignment vs. theory over time. For each method: cosine of ∆p (solid: Euclidean; dotted: Shahshahani), sign agreement of log–ratio slopes, and event–time gap (procedural − theory). DPO: low cosine, near–perfect signs; GRPO: near–neutral; STaR: high cosine, zero gap.

Euclidean cosine Barrier

Shahshahani cosine

p / p

JS divergence

0.0000

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.005

0.30

=0

0.0005

0.3

0.004

=0.0003

0.25

0.0010

Cosine

0.003

0.2

0.0015

0.002

0.20

0.0020

0.001

0.1

0.15

0.0025

0.000

16 64 128

16 64 128

16 64 128

16 64 128

Batch size

Batch size

Batch size

Batch size

###### Figure 5: Alignment summary vs. batch size. Euclidean/Shahshahani cosine and one–step JS divergence as functions of B (markers: mean; bars: s.d.). Cosine decreases with B for DPO while JS concurrently decreases, indicating increasingly synchronous trajectories despite metric/parameterization mismatch.

Incorrect mass

###### Min cluster mass

###### Between-seed JSD

###### Correct mass

10 3

1.000

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

|0.3|12 0.3|22 0.3|26 0.3|25|
|---|---|---|---|---|
|0.2|95 0.3|08 0.3|19 0.3|23|
|0.2|80 0.3|01 0.3|09 0.3|12|
| | | | | |

|1.14e|-05 6.46e|-05 7.55e|-05 0.000|149|
|---|---|---|---|---|
|1.08e|-05 5.52e|-05 0.000|112 0.000|177|
|5.52e|-06 3.7e|-05 4.64e|-05 0.000|188|
| | | | | |

|1.0|00 1.0|00 1.0|00 1.0|00|
|---|---|---|---|---|
|1.0|00 1.0|00 1.0|00 0.9|99|
|1.0|00 0.9|99 0.9|95 0.9|78|
| | | | | |

10 2

0.32

0.02

0.02

0.02

0.02

6.29e-78 9.31e-47 4.2e-11 1.68e-07

0.995

10 3

0.31

10 4

0.990

0.05

0.05

0.05

0.05

2.23e-07 1.64e-06 4.58e-05 0.00126

10 4

0.30

0.985

10 5

0.29

0.1

0.1

0.1

0.1

0.000332 0.000905 0.0047 0.0223

10 5

0.980

10 6

0.1 0.25 0.5 0.75

0.1 0.25 0.5 0.75

0.1 0.25 0.5 0.75

0.1 0.25 0.5 0.75

###### Figure 6: DCR phase diagrams over (α,β). From left to right: incorrect mass (log scale), minimum cluster mass, between–seed JSD, and correct mass. A contiguous band shows near–zero error, high structured diversity, and a unique terminal distribution.

Incorrect mass

Correct mass

Min cluster mass

Kernel energy

Objective

Safety margin

0.30

0.35 =+0.000 =+0.010

|=+0<br><br>| |.000 =+0<br><br>| | |.000 =+0<br><br>| | |.000| |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

=+0.000

=+0.092

1.0

1.0

=+0.000

=-0.000

=+0.000

0.30

10 81

=+0.000

=-0.030

0.30

0.25

10 89

0.8

0.8

=-0.062

0.25

0.25

10 97

=+0.035

0.20

=-0.000

0.20

=+0.008

0.6

0.6

10 105

0.20

=+0.000

Value

0.15

0.15

10 113

0.15

0.4

0.4

10 121

0.10

0.10

0.10

10 129

0.2

0.2

0.05

0.05

0.05

10 137

=-0.000

0.0

0.00

0.00

0.00

0.0

DCR Entropy-only Ungated

DCR Entropy-only Ungated

DCR Entropy-only Ungated

DCR Entropy-only Ungated

DCR Entropy-only Ungated

DCR Entropy-only Ungated

###### Figure 7: DCR vs. ablations. Bars (mean±sd) for incorrect mass (log axis), coverage, kernel energy, objective

∆Jp, and safety margin. DCR achieves the best trade–off (low error, full coverage, lower kernel energy, strong safety). Entropy–only preserves breadth without distinctiveness; Ungated reduces safety by penalizing similarity outside the correct set.

Objective proxy

Safety margin

1.00

| | |
|---|---|
| |DCR Entropy-only|
| |Ungated|
| | |
| | |
| | |
| | |

1.025

0.98

1.000

0.96

Value

0.975

0.94

0.950

0.92

0.925

0.90

0.900

Correct mass

Kernel energy

1.00

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

0.50

0.98

0.45

0.96

Value

0.94

0.40

0.92

0.35

0.90

0.30

0.88

0 1000 2000 3000 4000 5000

0 1000 2000 3000 4000 5000

Step

Step

###### Figure 8: Objective & safety (overlay). Overlay of Jp (left) and safety (right) for DCR (green), Entropy–only (gray), and Ungated (gold).

10

0.02 0.05 0.1

Safety

0.5

8

0.0

### Seeds

6

0.1 0.25 0.5

4

2

0

0.0 0.2 0.4 0.6 0.8

Minimum safety margin

###### Figure 9: Safety–margin distribution within the DCR band. Minimum safety margin per run (bars) with a scatter inset over (α,β) (green markers). All seeds stay comfortably above 0 (min ≈ 0.267).

