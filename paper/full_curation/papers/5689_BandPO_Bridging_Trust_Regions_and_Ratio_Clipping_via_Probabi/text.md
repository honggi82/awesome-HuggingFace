# arXiv:2603.04918v1[cs.LG]5Mar2026

OpenMOSS

[Figure 1]

### BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning

Yuan Li1,2 Bo Wang1 Yufei Gao1 Yuqian Yao1,2 Xinyuan Wang1 Zhangyue Yin1 Xipeng Qiu1,2,†

liyuan24@m.fudan.edu.cn, xpqiu@fudan.edu.cn 1Fudan University 2Shanghai Innovation Institute

Abstract

Proximal constraints are fundamental to the stability of the Large Language Model reinforcement learning. While the canonical clipping mechanism in PPO serves as an efficient surrogate for trust regions, we identify a critical bottleneck: fixed bounds strictly constrain the upward update margin of low-probability actions, disproportionately suppressing high-advantage tail strategies and inducing rapid entropy collapse. To address this, we introduce Band-constrained Policy Optimization (BandPO). BandPO replaces canonical clipping with Band, a unified theoretical operator that projects trust regions defined by 𝑓 -divergences into dynamic, probability-aware clipping intervals. Theoretical analysis confirms that Band effectively resolves this exploration bottleneck. We formulate this mapping as a convex optimization problem, guaranteeing a globally optimal numerical solution while deriving closed-form solutions for specific divergences. Extensive experiments across diverse models and datasets demonstrate that BandPO consistently outperforms canonical clipping and Clip-Higher, while robustly mitigating entropy collapse.

GitHub: https://github.com/OpenMOSS/BandPO.git

#### 1 Introduction

Reinforcement Learning from Human Feedback (RLHF) has established itself as the dominant paradigm for the post-training of Large Language Models (LLMs), wherein the proximal constraint on policy updates serves as a pivotal mechanism designed to balance optimization stability with effective exploration [17]. Schulman et al. [22] emulate trust-region updates via clipping the surrogate objective, circumventing the expensive Fisher Information computations required by TRPO [21]. This clipping mechanism has emerged as the default configuration for LLM RL and is extensively adopted in GRPO [25] and its variants.

While the canonical clipping mechanism ensures stability by emulating trust-region updates, Park et al. [18] argue that it implicitly inhibits exploration by imposing a detrimental bias against policy entropy. More precisely, in Reinforcement Learning with Verifiable Rewards (RLVR) scenarios, while lower-bound clip-

† Corresponding author.

Both allow (intersection)

BandPO allows, DAPO clips DAPO allows, BandPO clips Both clip

[Figure 2]

1.00

10

BandPO: rKL,δ(a;P), rKL,δ(a;P) BandPO: rTV,δ(a;P), rTV,δ(a;P) BandPO: rχ2,δ(a;P), rχ2,δ(a;P)

0.75

ProbabilityVariations

8

old

0.50

/w

6

0.25

e

n

=

0.00

4

atio

0.25

r

2

0.50

0

0.0

DAPO: rfix, rfix DCPO: rDAC, rDAC Theory: rTheor., rTheor.

0.75

1.0

0.2

0.8

0.4

1.00

0.6

0.6

0.4

0.0 0.2 0.4 0.6 0.8 1.0

old

new

0.8

0.2

old(a|s)

1.0

0.0

(a) Ratio clipping regions

(b) Bounds of Probability Variation

###### Figure 1 Comparison of clipping bounds between BandPO and baselines. (a) Comparison of ratio clipping regions:

BandPO vs. DAPO. While DAPO enforces fixed asymmetric bounds (𝜖+ = 0.28, 𝜖− = 0.2), BandPO projects a KLinduced trust region (𝛿 = 0.1) into dynamic bounds. The blue region highlights the expanded margin for low-probability, positive-advantage actions, effectively preventing premature saturation and preserving critical exploration gradients. (b) Comparison of the Bounds of Probability Variation. We visualize the bounds of variation derived from the Theoretical Simplex, DAPO, DCPO, and BandPO (ours). The symbols 𝑟¯ and 𝑟 denote the upper and lower clipping boundaries, respectively. Parameters are fixed at 𝜖+ = 0.28, 𝜖− = 0.2, and 𝛿 = 0.1. BandPO strictly adheres to physical simplex constraints while unlocking significant upward variation for low-probability actions.

ping tends to increase entropy, upper-bound clipping decreases it; with symmetric clipping thresholds, the latter effect dominates, resulting in net entropy reduction even when the algorithm is fed purely random rewards [24]. This bias effectively silences the gradient signals for low-probability yet high-advantage actions, preventing the model from reinforcing novel, superior strategies that lie in the tail of the distribution. To mitigate this, Yu et al. [32] proposes the Clip-Higher strategy to relax the upper clipping bound. While Cui et al. [5] acknowledges that it delays entropy collapse, they also highlight its instability—often leading to performance collapse after saturation. This indicates that adjusting thresholds fails to address the inherent limitations of fixed clipping bounds.

We formally characterize a critical structural bottleneck inherent in the canonical clipping mechanism, as detailed in Section 4. Specifically, constraining the probability ratio within fixed bounds enforces a linear dependence, wherein the maximum feasible probability variation scales proportionally with the old probability. Consequently, for positive-advantage actions, lower probabilities dictate vanishingly small margins for upward variation, rendering them susceptible to premature clipping and nullifying their gradient contributions. Constrained by the necessity of proximal policy updates, the margin for expanding the fixed clipping ratio range is limited, failing to reconcile the fundamental tension between optimization constraints and effective exploration.

To address this bottleneck, we propose Band-constrained Policy Optimization (BandPO). We introduce a unified theoretical operator, Band, which projects trust regions induced by general 𝑓 -divergences into dynamic, probability-aware clipping intervals. BandPO substitutes the clipping in GRPO with Band, employing a single interpretable radius parameter to enforce proximal constraints, thereby significantly streamlining hyperparameter tuning. Mathematically, we frame this mapping as a convex optimization problem, guaranteeing globally optimal numerical solutions while deriving efficient closed-form solutions for specific instances like Total Variation (TV) and Pearson 𝜒2-divergence. Crucially, we theoretically analyze the properties of Band and confirm that it naturally circumvents this bottleneck. As exemplified by the KL case in Figure 1a, the projected bounds adaptively expand the feasible upward margin for low-probability actions to

prevent premature clipping, corresponding to the expanded blue region. Empirically, compared with GRPO and GRPO with clip-higher, BandPO demonstrates consistent performance improvements across Qwen2.5 (3B, 7B) and Llama3 (8B) on multiple mathematical benchmarks, while robustly mitigating entropy collapse. Our main contributions are summarized as follows:

- • We formally characterize a critical bottleneck in canonical clipping, revealing that the feasible upward update margin scales linearly with the action probability. This tends to nullify gradients for low-probability, positive-advantage actions, inhibiting effective exploration.
- • We propose BandPO, introducing a unified Band operator to project 𝑓 -divergence-induced trust regions into dynamic clipping intervals. We formulate this as a convex optimization problem, guaranteeing globally optimal numerical solutions while deriving closed-form solutions for specific divergences, theoretically demonstrating that BandPO circumvents the bottleneck.
- • We demonstrate that BandPO achieves consistent performance gains over GRPO and Clip-Higher baselines across various models on math benchmarks, robustly mitigating entropy collapse.

#### 2 Related Work

From Trust Regions to Ratio Clipping. Proximal constraints are fundamental to stable policy optimization, ensuring that the updated policy remains within a controlled vicinity of the sampling policy. Introduced by Schulman et al. [21], the trust-region concept has been widely adopted in policy gradient optimization to impose proximal constraints. Schulman et al. [21] employs the KL-induced trust region to ensure that the new policy remains within a small neighborhood of the old policy at each update. Theoretically, such trust regions can be characterized by various distributional discrepancies, including integral probability metrics [27] and the family of 𝑓 -divergences—such as Pearson 𝜒2 and TV [4, 14]. However, the resulting constrained optimization problem involves an inequality constraint, rendering it computationally prohibitive for large-scale applications. To address this issue, Schulman et al. [22] proposed two variants—PPO-Penalty and PPO-Clip —that eliminate the need for conjugate gradient methods. The PPO-Clip mechanism and its variants have been extensively adopted across diverse domains, spanning complex strategic gaming [2, 16], physics-based character animation [19], and robotic control [15, 31]. Subsequent works have established the clipping mechanism as the dominant paradigm for LLM post-training [17, 33]. Concurrently, there is a notable shift towards critic-free paradigms to enhance computational efficiency [1, 9, 10, 25].

Adaptive Clipping Variants. The clipping mechanism is favored for its simplicity, yet it also raises ongoing questions regarding the appropriate choice of clipping bounds. To mitigate instability from negativeadvantage actions, Ye et al. [30] introduces an auxiliary lower bound, adopted in the LLM RL framework [26]. Chen et al. [3] proposes a state-wise adaptive clipping mechanism that modulates the clipping strength according to state importance, while Farsang and Szegletes [7] applies a simple time-decaying schedule to the clipping range. Despite their empirical success, these heuristics rely on auxiliary hyperparameters lacking a clear theoretical grounding, rendering them brittle and difficult to tune. Wang et al. [28] establishes a theoretical connection between KL-divergence and the clipping bounds, demonstrating performance improvements in continuous control tasks.

Clip Control in LLM. LLM RL operates in an extremely high-dimensional action space, where the combination of long reasoning horizons and extensive group sampling results in a high cumulative incidence of low-probability actions. To address this issue, DAPO [32] proposes the Clip-Higher strategy, which decouples and relaxes the clipping upper bounds. Building upon the decoupled bounds, DCPO [29] derives dynamic clipping boundary functions via inequality relaxation, which dynamically adjust according to the action probabilities.

However, compared to the maturity of continuous control, the theoretical underpinnings of clipping mechanisms in LLM RL remain under-explored. Existing methods lack a principled framework to control clipping bounds via simple, effective, and interpretable parameters, consequently struggling to balance proximal

constraints with effective exploration. To bridge this gap, we propose BandPO, which employs a unified operator that projects 𝑓 -divergences-induced trust regions into probability-aware clipping intervals, resolving the bottleneck using an interpretable parameter.

#### 3 Preliminaries

- 3.1 Notation

We formulate the RL alignment of LLMs as a discrete Markov Decision Process. Let 𝜋𝜃 denote the policy represented by the LLM. Given a prompt 𝑥 sampled from a dataset 𝒟, the policy generates a response sequence 𝑦 = (𝑎1, 𝑎2, . . . , 𝑎𝑇) by auto-regressively sampling from a vocabulary 𝒱 of size 𝑉 = |𝒱 |. Each action 𝑎𝑡 ∈ 𝒱 corresponds to a token generated by the LLM. At step 𝑡, the state 𝑠𝑡 = (𝑥, 𝑦<𝑡) comprises the prompt 𝑥 and the preceding tokens 𝑦<𝑡 = (𝑎1, . . . , 𝑎𝑡−1). The policy 𝜋𝜃 maps 𝑠𝑡 to a conditional probability distribution 𝜋𝜃(· | 𝑠𝑡) over 𝒱. We denote by ℝ and ℝ+ the sets of real and non-negative real numbers, respectively. Let Δ𝑉 ≜ {𝑝 ∈ ℝ𝑉+ | 𝑉𝑖=1 𝑝𝑖 = 1} denote the probability simplex over 𝑉 categories. The optimization objective is to maximize the expected reward: 𝐽(𝜃) = 𝔼𝑥∼𝒟,𝑦∼𝜋𝜃[𝑅(𝑥, 𝑦)], where 𝑅(𝑥, 𝑦) denotes a sparse, sequence-level scalar reward, typically derived from verification signals from verification signals.

- 3.2 Clip-Based Proximal Constraints in LLM RL

Consider an iterative optimization process where we update 𝜋𝜃 using trajectories sampled by 𝜋old. The probability ratio 𝑟𝑡(𝜃) = 𝜋𝜋𝜃(𝑎𝑡|𝑠𝑡)

old(𝑎𝑡|𝑠𝑡) serves as an importance sampling weight to correct for the distributional shift. Schulman et al. [22] proposed clipping 𝑟𝑡 to impose proximal constraints, while employing a critic model for Generalized Advantage Estimation [23]. Inheriting the clipping mechanism, Shao et al. [25] introduced Group Relative Policy Optimization (GRPO), which circumvents the computational burden of training a critic model by estimating advantages from a group of sampled responses. We denote the objective as 𝒥GRPO(𝜃), which aggregates the per-token objectives across a group of 𝐺 outputs:

𝑇𝑖

𝐺

1 𝐺

1 𝑇𝑖

𝒥GRPO(𝜃) = 𝔼𝑥∼𝒟,{𝑦

𝒥𝑡(𝜃; 𝑦𝑖) , (1)

𝑖}𝐺𝑖=1∼𝜋old

𝑡=1

𝑖=1

Specifically, we formulate 𝒥𝑡(𝜃; 𝑦𝑖) to admit asymmetric clipping boundaries: 𝒥𝑡(𝜃; 𝑦𝑖) = min 𝑟𝑡,𝑖𝐴𝑡,𝑖, clip(𝑟𝑡,𝑖, 1 − 𝜖−, 1 + 𝜖+)𝐴𝑡,𝑖 − 𝛽 𝐷KL(𝜋𝜃(·|𝑠𝑡)∥𝜋ref(·|𝑠𝑡))

. (2)

per-token KL divergence

This formulation averages the objective over tokens to accommodate varying sequence lengths 𝑇𝑖. 𝐺 denotes the group size, and 𝑟𝑡,𝑖 represents the probability ratio. The advantage 𝐴𝑡,𝑖 is derived from the sequencelevel rewards 𝑅𝑖, standardized within the group via 𝐴𝑡,𝑖 = (𝑅𝑖 − 𝜇𝑅)/𝜎𝑅, where 𝜇𝑅 and 𝜎𝑅 are the mean and standard deviation of rewards {𝑅𝑗}𝐺𝑗=1, respectively. The term 𝛽𝐷KL serves as a regularization penalty, anchoring 𝜋𝜃 to the reference policy 𝜋ref to preserve linguistic coherence. While GRPO employs symmetric bounds (𝜖+ = 𝜖−), Equation (2) generalizes this to allow asymmetric bounds (𝜖+ > 𝜖−).

#### 4 The Bottleneck in Canonical Clipping

The canonical clipping mechanism confines the probability ratio 𝑟𝑡(𝜃) to the interval [1 − 𝜖−, 1 + 𝜖+], where 𝜖−, 𝜖+ > 0 are fixed constants. This constraint is formulated as:

𝜋𝜃(𝑎 | 𝑠) 𝜋old(𝑎 | 𝑠)

1 − 𝜖− ≤

≤ 1 + 𝜖+. (3)

We define the probability variation of an action 𝑎 given state 𝑠 as: Δ𝜋(𝑎|𝑠) = 𝜋𝜃(𝑎|𝑠) − 𝜋old(𝑎|𝑠), which explicitly quantifies the deviation of the probability relative to the sampling policy. Multiplying all terms in Inequality (3) by the strictly positive 𝜋old(𝑎|𝑠) and subtracting 𝜋old(𝑎|𝑠) yields the set of feasible probability variations, denoted as 𝒞clip:

{Δ𝜋(𝑎|𝑠) | −𝜖−𝜋old(𝑎|𝑠) ≤ Δ𝜋(𝑎|𝑠) ≤ 𝜖+𝜋old(𝑎|𝑠)} . (4)

Theoretically, the constraint 𝜋𝜃(·|𝑠) ∈ Δ𝑉 allows the variation to range from −𝜋old (dropping to 0) to 1 − 𝜋old (rising to 1). We define this maximal feasible set as 𝒞simplex:

{Δ𝜋(𝑎|𝑠) | −𝜋old(𝑎|𝑠) ≤ Δ𝜋(𝑎|𝑠) ≤ 1 − 𝜋old(𝑎|𝑠)} . (5)

The derivation above establishes the relationship Δ𝜋(𝑎|𝑠) = (𝑟𝑡(𝜃) − 1)𝜋old(𝑎|𝑠). Using this relationship to map 𝒞simplex into the ratio space yields the theoretical bounds of 𝑟𝑡(𝜃):

1 𝜋old(𝑎 | 𝑠)

𝜋𝜃(𝑎 | 𝑠) 𝜋old(𝑎 | 𝑠)

0 ≤

. (6)

≤

As illustrated in Fig. 1b, analysis within 𝒞simplex reveals that fixed clipping bounds on 𝑟𝑡(𝜃) (e.g., DAPO) constrain probability variations to scale linearly with 𝜋old(𝑎|𝑠). Consequently, the feasible upward shift vanishes as the probability approaches zero, contradicting the theoretical upper bound. This discrepancy induces premature clipping for low-probability actions with positive advantages, effectively nullifying gradient contributions. Conversely, in high-probability regimes, the upper bounds of DAPO and DCPO exceed inherent simplex limits, rendering the constraint mathematically vacuous. Thus, fixed bounds fail to reconcile proximal constraints with exploration efficacy, while dynamic bounds lacking clear theoretical grounding result in partial constraint failure. These limitations necessitate a transition to theoretically grounded, dynamic clipping bounds of the probability ratio.

Consider a tail token 𝜋old = 0.08 with 𝜖+ = 0.2. The allowed update Δ𝜋 ≤ 0.016 is negligible compared to the theoretical capacity (0.92). Enabling meaningful exploration (e.g., Δ𝜋 ≈ 0.4) would require raising 𝜖+ to ≈ 5.0. However, this renders the constraint mathematically vacuous for head tokens (𝜋old = 0.8), as the permitted shift Δ𝜋 ≤ 4.0 far exceeds the physical limit of 0.2. Thus, the static clipping approach is fundamentally ill-equipped to bridge the gap between stable optimization and the acquisition of novel and effective strategies from the tail of the probability distribution.

#### 5 Method

In this section, we introduce BandPO. Central to BandPO is the Band operator, which projects trust regions defined by 𝑓 -divergences into probability-aware clipping intervals. Our analysis of Band’s theoretical properties demonstrates that it effectively resolves the exploration bottleneck in Section 4.

##### 5.1 𝑓 -Divergence-Induced Trust Regions

Consider a fixed state 𝑠𝑡. Let 𝑃(·) ≜ 𝜋old(·|𝑠𝑡) ∈ Δ𝑉 and 𝑄(·) ≜ 𝜋𝜃(·|𝑠𝑡) ∈ Δ𝑉 denote the distributions over 𝒱, respectively. We formalize the trust region using the family of 𝑓 -divergences [4]. Let 𝑓 : ℝ+ → ℝ be a strictly convex function with 𝑓 (1) = 0. The divergence 𝐷𝑓 (𝑄∥𝑃) is defined as:

𝑄(𝑎) 𝑃(𝑎)

. (7)

𝐷𝑓 (𝑄∥𝑃) ≜

𝑃(𝑎) 𝑓

𝑎∈𝒱

With 𝑃 serving as the anchor, we define the trust region 𝒯𝑓 ,𝛿(𝑃) as the convex set constraining 𝑄 within a 𝛿-neighborhood of 𝑃 on the probability simplex Δ𝑉:

𝒯𝑓,𝛿(𝑃) ≜ 𝑄 ∈ Δ𝑉 | 𝐷𝑓 (𝑄∥𝑃) ≤ 𝛿 , (8)

where 𝛿 > 0 dictates the radius of the trust region. This geometric formulation generalizes standard proximal constraints. Notably, choosing the generator function 𝑓 (𝑢) = −log 𝑢 + 𝑢 − 1 recovers the specific trust region employed in TRPO [21], i.e., 𝐷KL(𝑃∥𝑄) ≤ 𝛿.

##### 5.2 Band: An Operator for Projecting Trust Regions to Probability-Aware Clipping Bounds

We now derive the probability-aware ratio bounds induced by the geometric constraint 𝒯𝑓 ,𝛿(𝑃). To facilitate the analysis, we recast the probability ratio 𝑟𝑡(𝜃) as a function of the candidate distribution 𝑄 ∈ Δ𝑉. For any token 𝑎 ∈ 𝒱 satisfying 𝑃(𝑎) > 0, the ratio function is redefined as:

𝑄(𝑎) 𝑃(𝑎)

𝑟(𝑎;𝑄, 𝑃) ≜

. (9)

While 𝑃 remains fixed, the candidate distribution 𝑄 ∈ Δ𝑉 varies subject to the constraint 𝒯𝑓 ,𝛿(𝑃).

Optimal Dynamic Bounds. The upper bound of the probability ratio is determined by maximizing the probability 𝑄(𝑎) subject to the constraint 𝒯𝑓 ,𝛿(𝑃). Since 𝒯𝑓 ,𝛿(𝑃) forms a convex feasible set, this formulation constitutes a convex optimization problem with a linear objective function with respect to the variable 𝑄:

𝑄(𝑎) 𝑃(𝑎)

𝑟 𝑓,𝛿(𝑎;𝑃) ≜ max

. (10)

𝑄∈𝒯𝑓 ,𝛿(𝑃)

Symmetrically, the minimal admissible probability ratio is determined by solving:

𝑄(𝑎) 𝑃(𝑎)

𝑟 𝑓,𝛿(𝑎;𝑃) ≜ min

. (11)

𝑄∈𝒯𝑓 ,𝛿(𝑃)

Crucially, Problems (10) and (11) are convex programs. Under strictly convex 𝑓 and 𝑃(𝑎) ∈ (0, 1), any local optimum is the global optimum, enabling stable numerical solutions.

The Band Operator. We propose Band, a unified operator designed to supersede canonical clipping by strictly constraining the probability ratio within the feasible interval induced by the 𝑓 -divergence trust region. Given a generator function 𝑓 and radius 𝛿, for ∀𝑃(𝑎) ∈ (0, 1), solving Problems (10) and (11) yields the rigorous lower and upper bounds for the ratio. Crucially, this derivation projects the high-dimensional geometric constraint 𝒯𝑓 ,𝛿(𝑃) onto a scalar interval specific to action 𝑎, to which the ratio is strictly confined. We formulate this operation as:

Band𝑓,𝛿 𝑟; 𝑎, 𝑃 ≜ clip 𝑟, 𝑟 𝑓,𝛿(𝑎;𝑃), 𝑟 𝑓,𝛿(𝑎;𝑃) . (12)

In stark contrast to the fixed clipping interval [1 − 𝜖, 1 + 𝜖], Band yields probability-aware bounds governed solely by the single, interpretable trust-region radius 𝛿.

##### 5.3 Reduction to Univariate Optimization

Although Problems (10) and (11) are convex, the decision variable 𝑄 resides in a high-dimensional simplex (𝑉 ≈ 105), rendering direct optimization computationally prohibitive. We circumvent this by exploiting the structural symmetry of the divergence constraint to strictly reduce the problem to a univariate formulation, as established in the following lemma, proven in Appendix A.1.

Lemma 1 (Optimality of Uniform Complement Rescaling). Given a reference distribution 𝑃 ∈ Δ𝑉 with full support (i.e., 𝑃(𝑣) > 0, ∀𝑣) and an action 𝑎 ∈ 𝒱, the optimal solution 𝑄★ to the extremal Problems (10) and (11) must strictly preserve the relative probability proportions within the complement set 𝒱 \ {𝑎}. Specifically, the probability ratio is constant for all non-target actions:

𝑄★(𝑏) 𝑃(𝑏)

= 𝑐, ∀𝑏 ∈ 𝒱 \ {𝑎}, (13)

where the scaling factor 𝑐 ∈ ℝ+ is uniquely determined by the simplex normalization constraint 𝑣∈𝒱 𝑄★(𝑣) = 1. This implies that 𝑄★ is fully parameterized by the single scalar ratio 𝑟 = 𝑄★(𝑎)/𝑃(𝑎) at the target action.

Based on Lemma 1, the optimization reduces to determining the scalar ratio 𝑟 ≜ 𝑄(𝑎)/𝑃(𝑎) in the target action 𝑎. Let 𝑝 ≜ 𝑃(𝑎). The simplex constraint 𝑣 𝑄(𝑣) = 1 uniquely determines the complement scaling factor 𝑐 as a function of 𝑟, since 𝑟𝑝+𝑐(1−𝑝) = 1 =⇒ 𝑐(𝑟) = 11−−𝑟𝑝𝑝 . Substituting this mapping into Equation (7), the divergence collapses into a univariate function 𝑔𝑓 (𝑝, 𝑟):

1 − 𝑟𝑝 1 − 𝑝

𝑃(𝑏) 𝑓 (𝑐) = 𝑝 𝑓 (𝑟) + (1 − 𝑝) 𝑓

≜ 𝑔𝑓 (𝑝, 𝑟). (14)

𝐷𝑓 (𝑄∥𝑃) = 𝑃(𝑎) 𝑓 (𝑟)

+

𝑏≠𝑎

target

Equation (14) strictly reduces the high-dimensional constraint 𝐷𝑓 (𝑄∥𝑃) ≤ 𝛿 to a scalar inequality 𝑔𝑓 (𝑝, 𝑟) ≤ 𝛿. This scalarization effectively shifts the optimization domain from the high-dimensional simplex 𝑄 ∈ Δ𝑉 to the univariate scalar 𝑟. Consequently, we transform Problems (10) and (11) into a tractable root-finding problem, where the distinct roots of the equation 𝑔𝑓 (𝑝, 𝑟) = 𝛿 yield the precise values of the projection clipping bounds 𝑟 𝑓 ,𝛿 and 𝑟 𝑓 ,𝛿, as formalized in the following theorem (proof in Appendix A.2).

Theorem 1 (Exact Scalarization of Trust-Region Constraints). Assume the generator function 𝑓 : ℝ+ → ℝ is strictly convex with 𝑓 (1) = 0. For any action 𝑎 with 𝑃(𝑎) = 𝑝 ∈ (0, 1), the Problems (10) and (11) are equivalent to finding the roots of the scalar function:

1 − 𝑟𝑝 1 − 𝑝

𝑔𝑓 (𝑝, 𝑟) ≜ 𝑝 𝑓 (𝑟) + (1 − 𝑝) 𝑓

= 𝛿, (15)

subject to the feasibility constraint 𝑟 ∈ [0, 1/𝑝]. Specifically, 𝑔𝑓 (𝑝, 𝑟) is strictly convex with respect to 𝑟 and achieves its global minimum of 0 at 𝑟 = 1. Consequently, for any valid trust-region radius 𝛿 > 0, the equation 𝑔𝑓 (𝑝, 𝑟) = 𝛿 possesses exactly two unique roots, corresponding to the optimal clipping bounds:

𝑟 𝑓 ,𝛿(𝑎; 𝑃) = min 𝑟 ∈ [0, 1] | 𝑔𝑓 (𝑝, 𝑟) = 𝛿 , (16) 𝑟 𝑓 ,𝛿(𝑎; 𝑃) = max 𝑟 ∈ [1, 1/𝑝] | 𝑔𝑓 (𝑝, 𝑟) = 𝛿 . (17)

- 5.4 Properties of Band Bounds Building upon Theorem 1, we theoretically analyze the asymptotic behavior and strict monotonicity of the

bounds 𝑟 𝑓 ,𝛿(𝑝) and 𝑟 𝑓 ,𝛿(𝑝) with respect to 𝑝 ∈ (0, 1), with proofs provided in Appendix A.3 and A.4. This analysis demonstrates that Band theoretically resolves the exploration bottlenecks identified in Section 4.

- Proposition 1 (Asymptotic Behavior of Band Bounds). Given a trust-region radius 𝛿 > 0, the bounds 𝑟 𝑓 ,𝛿(𝑝) and 𝑟 𝑓 ,𝛿(𝑝) exhibit the following limiting behaviors as the probability 𝑝 ∈ (0, 1) approaches the boundaries of the simplex:

lim

𝑝→0+

𝑟 𝑓 ,𝛿(𝑝) = +∞, lim

𝑝→0+

𝑟 𝑓 ,𝛿(𝑝) = 0, lim

𝑝→1−

𝑟 𝑓 ,𝛿(𝑝) = 1. (18)

- Proposition 2 (Strict Monotonicity of Band Bounds). The clipping bounds are strictly monotonic functions of the

4.0

| | |Band<br><br>|PO: rKL,δ(a;P|), rKL,δ(a;P)|
|---|---|---|---|---|
| | |Band Band|PO: rTV,δ(a;P PO: rχ2,δ(a;P)|), rTV,δ(a;P) , rχ2,δ(a;P)<br><br>|
| | |DCP DAP<br><br>|O: rDAC, rDAC O: rfix, rfix| |
| | |Theo|ry: rTheor., rThe<br><br>|or.|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

3.5

###### ProbabilityRatio

3.0

2.5

2.0

1.5

1.0

0.5

0.0

0.0 0.2 0.4 0.6 0.8 1.0

old(a|s)

- Figure 2 Comparison of Probability Ratio Bounds. We visualize the ratio bounds derived from the Theoretical Simplex, DAPO, DCPO, and BandPO (ours). As 𝑝 → 1, BandPO bounds exhibit strict monotonicity: upper bounds decrease while lower bounds increase towards 1. Conversely, as 𝑝 → 0, the upper bounds of both DCPO and BandPO expand rapidly, effectively preventing premature clipping. Note that for TV and 𝜒2, the radius 𝛿 = 0.1 triggers the simplex saturation condition, where the lower bounds explicitly align with the theoretical limit of 0.

probability 𝑝. Specifically, given a 𝛿 > 0, the upper bound 𝑟 𝑓 ,𝛿(𝑝) is strictly decreasing with respect to 𝑝, while the lower bound 𝑟 𝑓 ,𝛿(𝑝) is strictly increasing with respect to 𝑝.

Resolving the Exploration Bottleneck. To demonstrate the constraint behaviors, Figures 1b and 2 visualize the bounds for probability variation and ratios, respectively. We contrast the Theoretical Simplex, DAPO, and DCPO against BandPO, instantiated with KL, Pearson 𝜒2, and TV divergences. As illustrated, the ratio upper bounds of probability-aware methods (DCPO and BandPO) decrease, while the lower bounds increase with respect to 𝑝. Furthermore, their upper ratio bounds expand rapidly as 𝑝 → 0, which prevents the premature suppression of low-probability actions with positive advantages, thereby preserving effective gradients and facilitating the exploration of valuable tail strategies. For Band𝑓 ,0.1, the KL variant permits a broader ratio range compared to the TV and 𝜒2 variants. Notably, TV and 𝜒2 correctly trigger the simplex saturation condition at low probabilities, where the lower bound explicitly aligns with the theoretical simplex limit of 0. Crucially, all BandPO variants maintain strict geometric consistency with the probability simplex Δ𝑉, underscoring the method’s theoretical soundness. In contrast, heuristic approaches like DCPO and DAPO violate the theoretical upper bound in high-probability regimes. Ultimately, while BandPO and DCPO successfully mitigate the exploration bottleneck, BandPO offers a more rigorous foundation, guaranteeing valid constraints governed by an interpretable parameter 𝛿.

Resolving the Exploration Bottleneck. Figures 1b and 2 compare BandPO (KL, 𝜒2, TV) against baselines. Both BandPO and DCPO expand ratio bounds as 𝑝 → 0, preventing premature suppression of tail actions to facilitate exploration. However, unlike heuristics (DCPO, DAPO) that violate theoretical limits at high probabilities, all BandPO variants maintain strict geometric consistency with the simplex Δ𝑉. Specifically, BandPOKL yields the broadest range, while TV and 𝜒2 correctly capture the simplex saturation at 0. Thus, BandPO uniquely resolves the exploration bottleneck with a rigorous foundation, guaranteeing valid constraints unlike heuristic approximations.

- 5.5 Solving Band Bounds For a chosen function 𝑓 and radius 𝛿, given the probability 𝑝 = 𝑃(𝑎), we solve Problems (16) and (17) to derive the probability ratio bounds 𝑟 𝑓 ,𝛿(𝑎; 𝑃) and 𝑟 𝑓 ,𝛿(𝑎; 𝑃).

Simplex Saturation. For a sufficiently large radius 𝛿, the divergence constraint may extend beyond the boundaries of the probability simplex Δ𝑉. This creates a conflict between the simplex boundaries and the divergence constraint, rendering Problems (16) and (17) ill-defined. We term this phenomenon Simplex Saturation. To enforce the simplex constraint, we formalize this saturation condition and align the Band bounds with the simplex limits, with details deferred to Appendix A.5.

- Proposition 3 (Constraint Saturation). Based on Inequality (6), denote simplex bounds as 𝑟max ≜ 1/𝑝 and 𝑟min ≜ 0. The optimal Band upper bound 𝑟 𝑓 ,𝛿(𝑝) is given by:

𝑟 𝑓 ,𝛿(𝑝) =

𝑟max, if 𝑔𝑓 (𝑝, 𝑟max) ≤ 𝛿, 𝑟†, otherwise,

(19)

where 𝑟† denotes the unique root of 𝑔𝑓 (𝑝, 𝑟) = 𝛿 in (1, 𝑟max), which is guaranteed to exist in the non-saturated regime. Symmetrically, 𝑟 𝑓 ,𝛿(𝑝) = 0 if 𝑔𝑓 (𝑝, 𝑟min) ≤ 𝛿; otherwise, it is the unique root of 𝑔𝑓 (𝑝, 𝑟) = 𝛿 in (0, 1).

Generic Numerical Solver. In the active regime, the optimal bounds correspond strictly to the unique roots of the binding equation 𝑔𝑓 (𝑝, 𝑟) = 𝛿. These roots are well-separated by the singularity at 𝑟 = 1. Consequently, they can be efficiently computed via standard bracketed root-finding algorithms (e.g., Bisection or Brent’s method), which are guaranteed to converge to the global optimum. We provide the rigorous convergence analysis and the standard solver formulation in Appendix A.6, along with a concrete instantiation for the KLdivergence. Closed-Form Solutions. Specific 𝑓 -divergences admit closed-form solutions to Problems (16) and (17), offering superior computational efficiency compared to numerical methods. We provide bounds for TV and Pearson 𝜒2 divergence below, with derivations detailed in Appendix A.7.

- Proposition 4 (Closed-Form Band Bounds for TV and Pearson 𝜒2). Consider a reference probability 𝑝 ∈ (0, 1)

and a trust-region radius 𝛿 > 0. Defining the generator functions as 𝑓TV(𝑢) = 12|𝑢 − 1| and 𝑓𝜒2(𝑢) = (𝑢 − 1)2, the closed-form clipping bounds are derived as follows. For TV, the bounds scale linearly with the inverse probability:

𝑟TV,𝛿(𝑝) = 1 +

𝛿 𝑝

, 𝑟TV,𝛿(𝑝) = 1 −

𝛿 𝑝

. (20)

For Pearson 𝜒2, the bounds scale with the inverse square root of the odds:

𝑟𝜒2,𝛿(𝑝) = 1 +

𝛿(1 − 𝑝) 𝑝

, 𝑟𝜒2,𝛿(𝑝) = 1 −

𝛿(1 − 𝑝) 𝑝

. (21)

##### 5.6 BandPO: Band-Constrained Policy Optimization

We propose BandPO, a policy optimization framework that substitutes the canonical clipping mechanism of GRPO with the theoretically rigorous Band operator. Formally, BandPO maximizes the objective 𝒥BandPO(𝜃):

𝑇𝑖

𝐺

1 𝐺

1 𝑇𝑖

𝒥𝑡Band(𝜃; 𝑦𝑖) . (22)

𝒥BandPO(𝜃) = 𝔼𝑥∼𝒟,{𝑦

𝑖}𝐺𝑖=1∼𝜋old

𝑡=1

𝑖=1

By employing the probability-aware clipping interval derived in Equation (12), the core per-token surrogate objective 𝒥𝑡Band(𝜃; 𝑦𝑖) is formulated as:

𝒥𝑡Band(𝜃; 𝑦𝑖) = min 𝑟𝑡,𝑖𝐴𝑡,𝑖, Band𝑓 ,𝛿 𝑟𝑡,𝑖; 𝑦𝑡,𝑖, 𝜋old(·|𝑠𝑡,𝑖) 𝐴𝑡,𝑖 − 𝛽𝐷KL(𝜋ref∥𝜋𝜃)𝑡, (23)

where 𝑟𝑡,𝑖 = 𝜋𝜋𝜃(𝑦𝑡,𝑖|𝑠𝑡,𝑖)

old(𝑦𝑡,𝑖|𝑠𝑡,𝑖) and the advantage 𝐴𝑡,𝑖 is computed as described in Section 3.2. By projecting the trust region induced by 𝑓 -divergence into a probability-aware scalar interval specific to the target action 𝑦𝑡,𝑖, the Band operator strictly confines the ratio within a theoretically feasible domain, thereby balancing

Table 1 Reasoning performance comparison across model scales (1.5B/3B/7B/8B). We report mean@32 and pass@32 (%) on AMC/AIME. Red marks the best average. BandPO consistently outperforms vanilla GRPO and heuristic variants (Clip-Higher).

AMC2023 AIME2024 AIME2025 Average

Method

mean@32/pass@32 mean@32/pass@32 mean@32/pass@32 mean@32/pass@32

###### DeepSeek-R1-Distill-Qwen-1.5B (800 steps)

GRPO 72.11 / 94.31 18.13 / 39.00 21.88 / 38.89 37.37 / 57.40 GRPO w/ Clip-Higher 77.03 / 94.98 18.23 / 41.09 23.12 / 40.16 39.46 / 58.74 GRPO w/ Relaxed BandKL,0.05 74.69 / 93.77 19.69 / 43.28 23.54 / 38.84 39.31 / 58.63 GRPO w/ BandKL,0.05 77.34 / 94.98 20.00 / 51.80 23.85 / 40.65 40.40 / 62.48

###### Qwen2.5-3B-Instruct (800 steps)

GRPO 45.94 / 77.33 3.54 / 11.68 3.23 / 8.79 17.57 / 32.60 GRPO w/ Clip-Higher 52.66 / 82.91 4.69 / 14.95 4.06 / 23.93 20.47 / 40.60 GRPO w/ Relaxed BandKL,0.05 52.97 / 87.05 4.58 / 15.11 4.06 / 21.00 20.54 / 41.05 GRPO w/ BandKL,0.03 52.81 / 87.84 4.27 / 10.00 4.06 / 22.40 20.38 / 40.08 GRPO w/ BandKL,0.10 51.41 / 84.77 3.54 / 14.31 6.04 / 20.85 20.33 / 39.98 GRPO w/ BandKL,0.05 55.17 / 87.55 4.79 / 14.21 6.04 / 24.28 22.00 / 42.01

###### DeepSeek-R1-Distill-Qwen-7B (500 steps)

GRPO 87.11 / 95.00 27.29 / 49.71 32.71 / 55.62 49.04 / 66.78 GRPO w/ Clip-Higher 87.50 / 95.00 26.77 / 48.11 30.83 / 56.96 48.37 / 66.69 GRPO w/ Relaxed BandKL,0.05 88.58 / 95.00 29.69 / 50.78 33.44 / 54.52 50.57 / 66.77 GRPO w/ BandKL,0.03 88.28 / 95.00 29.17 / 51.58 32.71 / 61.46 50.05 / 69.35 GRPO w/ BandKL,0.10 89.69 / 95.00 29.79 / 47.19 32.29 / 52.03 50.59 / 64.74 GRPO w/ BandKL,0.05 89.84 / 95.00 29.90 / 49.14 34.58 / 57.21 51.44 / 67.12

###### DeepSeek-R1-Distill-Llama-8B (500 steps)

GRPO 85.47 / 94.11 23.23 / 46.00 23.54 / 54.80 44.08 / 64.97 GRPO w/ Clip-Higher 86.79 / 94.13 24.58 / 46.49 28.33 / 61.86 46.57 / 67.49 GRPO w/ Relaxed BandKL,0.05 86.72 / 94.67 25.00 / 49.82 29.79 / 53.55 47.17 / 66.01 GRPO w/ BandKL,0.05 87.03 / 95.00 25.31 / 51.21 29.90 / 57.61 47.41 / 67.94

optimization stability with the effective exploration of tail strategies.

#### 6 Empirical Study

##### 6.1 Experimental Setup

Models and Datasets. We employ a composite training set merging DAPO [32] with MATH Levels 3–5 [8] to fine-tune four models spanning diverse architectures and scales: Qwen2.5-3B-Instruct [20] and the DeepSeekR1-Distill family (Qwen-1.5B/7B, Llama-8B) [6]. To evaluate reasoning robustness across varying difficulty levels, we validate on AMC 2023 [11], AIME 2024 [12], and AIME 2025 [13].

Implementation Details. We conduct all experiments on 8× NVIDIA H200 GPUs using the verl framework [26]. We train models for 800 steps (for 1.5B/3B) or 500 steps (for 7B/8B), utilizing a global batch size of 256 (mini-batch 64, micro-batch 8) and a learning rate of 1 × 10−6. During generation, we set both the sampling temperature and top-𝑝 to 1.0. All experiments are repeated three times to ensure statistical significance. Regarding clipping bounds, we adopt asymmetric thresholds with 𝜖+ = 0.28 and 𝜖− = 0.2. For BandPO, we enforce the KL divergence as the trust region constraint with 𝛿 = 0.05. Crucially, we implement a CUDA-accelerated parallel bisection method to efficiently solve for the Band bounds.

Baselines and Metrics. To isolate the impact of clipping mechanisms, we implement comparisons within the GRPO framework [25]. We establish two primary baselines: (1) GRPO, representing canonical symmetric clipping [22]; and (2) GRPO w/ Clip-Higher, representing the state-of-the-art asymmetric clipping introduced by DAPO [32]. For our method (BandPO), (3) GRPO w/ BandKL,𝛿, we replace the clip operator with the Band operator. Evaluation metrics include pass@32 to measure peak reasoning capability (probability of at least one correct solution) and mean@32 to quantify the expected policy robustness across 32 samples.

##### 6.2 Main Results

As presented in Table 1, BandPO consistently outperforms all baselines in mean@32 across diverse models and datasets, demonstrating superior exploitation capabilities. Specifically, BandPO improves mean@32 by at least 2.0 points over GRPO across all settings, with a notable gain of ∼10 points on the Qwen2.5-3B AMC2023 task. In contrast, baseline methods exhibit significant instability: GRPO w/ Clip-Higher suffers from performance regression on the 7B AIME benchmarks (dropping 1–2 points), and the 1.5B GRPO baseline exhibits severe instability, consistently collapsing around training step 340 (verified across multiple runs). Crucially, BandPO substantially lifts pass@32 (e.g., a 28.9% relative gain on 3B) and consistently surpasses heuristic baselines like Clip-Higher and Relaxed-Band. Overall, these results confirm that the probabilityaware constraints provided by Band offer a superior exploration-exploitation trade-off compared to heuristic clipping bounds of the probability ratio.

##### 6.3 Relaxing Band Bounds Degrades Performance

As illustrated in Figure 1a, while BandPO significantly expands the bounds for low-probability actions, it imposes tighter constraints in the high-probability regime compared to fixed asymmetric bounds (DAPO), as highlighted bythe green region. To investigate whether directly expanding the Band bounds yields further improvements, we evaluate GRPO w/ Relaxed BandKL,0.05, which heuristically aligns the high-probability bound with Clip-Higher. Results in Table 1 demonstrate that this heuristic relaxation leads to consistent overall performance degradation. For larger models (7B/8B), relaxing the bounds causes a slight decline in overall performance (∼0.5 decrease in average mean@32) and a notable drop of approximately 3 points in pass@32 on AIME 2025. This detrimental effect is exacerbated in smaller models (1.5B/3B); specifically, the 1.5B model suffers a drop of nearly 8 points in pass@32 on AIME 2024, while the 3B model decreases by roughly 3 points in mean@32 on AMC 2023. These findings underscore that directly relaxing Band to fully cover the Clip-Higher range is detrimental to performance. This suggests that clipping bounds should not be altered heuristically but should stem from rigorous theoretical derivation, underscoring the importance of BandPO’s theoretically grounded formulation.

##### 6.4 Radius 𝛿 Matters More for Smaller Backbones

While canonical clipping typically employs fixed heuristic thresholds (e.g., 𝜖− = 0.2, 𝜖+ = 0.28) across varying model scales, BandPO consolidates these constraints into a single, theoretically grounded hyperparameter: the trust region radius 𝛿. To investigate the impact of 𝛿 on performance and stability, we evaluate the KL-instantiated BandPO, varying 𝛿 ∈ {0.03, 0.05, 0.10} on Qwen2.5-3B and Qwen2.5-7B backbones. As evidenced in Table 1, setting 𝛿 = 0.05 consistently yields superior performance in terms of mean@32 for both model sizes compared to tighter (𝛿 = 0.03) or looser (𝛿 = 0.10) constraints. This finding contradicts the intuition that larger 𝛿 monotonically improves performance. Instead, it highlights a critical trade-off: while overly strict constraints limit the policy’s ability to learn from effective exploration, excessively loose constraints risk destabilizing the update, leading to suboptimal convergence. Crucially, our analysis reveals that smaller models exhibit higher sensitivity to the trust-region radius. For the 3B model, the optimal 𝛿 = 0.05 outperforms the suboptimal settings by approximately 10% in mean@32 and 5% in pass@32. In contrast, the 7B model demonstrates greater robustness, with performance fluctuations across the three 𝛿 settings confined to a narrow margin of 2–3%. This disparity suggests that larger models possess a more resilient optimization landscape, capable of tolerating rougher updates, whereas smaller models require precise trust-region management to prevent performance degradation. Based on these empirical findings, we recommend 𝛿 = 0.05 as a robust default starting point for practical implementations of BandKL,𝛿.

##### 6.5 BandPO Unlocks Exploration for Tail Actions

The theoretical analysis in Section 5.4 posits that BandPO resolves the exploration bottleneck via the Band operator. To empirically validate this mechanism, we analyze the training dynamics of the Band constraint (GRPO w/ BandKL,0.05) compared to the canonical clipping constraint (GRPO) and the Clip-Higher con-

OverallClipRate

0.0025

GRPO

GRPO w/ BandKL,0.05

0.0020

GRPO w/ Clip-Higher

0.0015

0.0010

0.0005

0.0000

0 100 200 300 400 500 600 700 800 Training Step

###### (a) Overall clip rate across training steps.

0.6

Clip-HighRate(p<0.2)

0.4

0.2

0.0

100 200 300 400 500 600 700 800 Training Step

(b) Clip-High rate for low-prob tokens (𝑝 < 0.2).

0.5

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.4

Entropy

0.3

0.2

0.1

0.0

0 100 200 300 400 500 600 700 800 Training Step

(c) Policy entropy evolution.

- Figure 3 Comparison of training dynamics. (a) Overall clip rate measuring the fraction of clipped tokens relative to total tokens per update. (b) Proportion of clip-high for low-probability tokens (𝑝 < 0.2) relative to total clipped tokens, identifying erroneous tail-action suppression. (c) Evolution of policy entropy measuring the concentration of action distributions. Our method (red) effectively prevents mode collapse by mitigating the vanishing margin issue in (b).

straint (GRPO w/ Clip-Higher) on the Qwen2.5-3B-Instruct backbone. As shown in Figure 3a, the overall clip rate of Band aligns closely with that of canonical clipping, whereas Clip-Higher exhibits a significantly lower rate (∼50% reduction). This contrast is expected; as visualized in Figure 1a, Band imposes stricter constraints than Clip-Higher (DAPO) in the high-probability regime (𝑝 > 0.8).

However, aggregate statistics mask a critical distributional shift. Figure 3b isolates the incidence of upperbound clipping (clip-high) specifically for low-probability actions (𝑝 < 0.2). A distinct contrast emerges: for fixed-bound methods (canonical clipping and Clip-Higher), the upper-bound constraint on tail tokens alone accounts for approximately 20% of the total clipping volume. In the early training phase (first 50 steps) of canonical clipping, this proportion surges to 60%, corresponding to the rapid entropy collapse observed in Figure 3c. These findings empirically confirm the bottleneck identified in Section 4: fixed bounds prematurely suppress positive-advantage updates for tail actions due to the vanishing upward margin.

In sharp contrast, Band reduces the clipping incidence for tail actions to nearly zero. By dynamically expanding the upper bound as 𝑝 → 0, Band ensures that low-probability actions with positive advantages are not prematurely discarded but are instead preserved to contribute effective gradients. Crucially, while Band maintains a high aggregate clip rate comparable to canonical clipping (Figure 3a), its entropy evolution mir-

rors the robust stability of Clip-Higher (Figure 3c). By substituting the canonical clipping operator with Band, BandPO effectively averts the early-stage entropy collapse characteristic of GRPO, eventually converging to a mean entropy an order of magnitude higher (0.2 vs. 0.02). This result yields a pivotal insight: optimization stability relies not merely on reducing the total volume of clipping, but on rigorously redistributing the clipping budget. BandPO succeeds by strategically reallocating the permissible update margin—loosening constraints on tail actions to facilitate exploration while tightening them on head actions to enforce stability.

#### 7 Discussion

In this work, we introduced BandPO, a principled framework that realigns the canonical proximal clipping mechanism with the geometric constraints of trust regions. By projecting 𝑓 -divergence constraints into dynamic, probability-aware intervals, BandPO resolves the exploration bottleneck for tail strategies while maintaining strict simplex consistency. Below, we discuss the implications of our findings, computational considerations, and future directions.

Limitations. Computational Overhead of Numerical Solvers. Unlike the computationally trivial canonical clipping mechanism, BandPO involves solving convex constraint equations, which inherently introduces numerical complexity. While our derived closed-form solutions for divergences such as TV and Pearson 𝜒2 (Prop. 4) maintain analytical efficiency, divergences lacking closed-form inverses—most notably the KLdivergence—necessitate the use of iterative root-finding algorithms. Consequently, applying BandPO with KL-divergence relies on numerical solvers, which inevitably incurs additional computational latency and minor approximation errors compared to elementary scalar operations. To mitigate this in latency-critical deploymentscenarios, wehighlight that the strictmonotonicityof Band bounds permits the pre-computation of high-precision lookup tables, effectively reducing the runtime complexity to 𝑂(1) memory access. Static Trust Region Assumptions. Our current framework enforces a global trust region radius 𝛿 across all tokens. This assumption of a uniform trust budget homogenizes the diverse landscape of language generation, overlooking the varying informational value of tokens. In practice, routine syntactic connectors and pivotal reasoning leaps likely require distinct stability margins. Consequently, a fixed 𝛿 represents a compromise: it may be inadvertently too loose for high-confidence syntax (risking instability) while simultaneously being too tight for complex reasoning steps (stifling deep exploration).

Future Work. A natural evolution of this framework is to transition from static to adaptive Band operators. We plan to investigate methods where the radius 𝛿𝑡 is dynamically modulated by token-level metrics, such as policy entropy or semantic uncertainty. By assigning tighter constraints to low-entropy syntactic transitions and relaxing bounds for high-stakes reasoning steps, we aim to further disentangle the trade-off between optimization stability and the exploration of novel reasoning paths.

#### 8 Conclusion

We introduced BandPO, a principled optimization framework that bridges the gap between computationally efficient clipping and rigorous trust region constraints. By projecting 𝑓 -divergence balls into dynamic, probability-aware bounds, BandPO effectively resolves the exploration bottleneck for high-advantage tail actions while maintaining training stability. Extensive experiments across models from 1.5B to 8B demonstrate that BandPO significantly outperforms heuristic baselines and robustly prevents entropy collapse. These results highlight the limitations of static heuristics and demonstrate the efficacy of geometrically grounded constraints in optimizing complex reasoning policies.

#### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

#### References

- [1] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms,

2024. URL https://arxiv.org/abs/2402.14740.

- [2] Bowen Baker, Ingmar Kanitscheider, Todor Markov, Yi Wu, Glenn Powell, Bob McGrew, and Igor Mordatch. Emergent tool use from multi-agent autocurricula, 2020. URL https://arxiv.org/abs/1909.07528.
- [3] Gang Chen, Yiming Peng, and Mengjie Zhang. An adaptive clipping approach for proximal policy optimization,

2018. URL https://arxiv.org/abs/1804.06461.

- [4] Imre Csiszár. On information-type measure of difference of probability distributions and indirect observations. Studia Sci. Math. Hungar., 2:299–318, 1967.

- [5] Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models, 2025. URL https://arxiv.org/abs/ 2505.22617.
- [6] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zĳia Zhu, Zĳun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.
- [7] Monika Farsang and Luca Szegletes. Decaying clipping range in proximal policy optimization. In 2021 IEEE 15th International Symposium on Applied Computational Intelligence and Informatics (SACI), page 000521–000526. IEEE, May 2021. doi: 10.1109/saci51354.2021.9465602. URL http://dx.doi.org/10.1109/SACI51354.2021.9465602.

- [8] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/ 2103.03874.
- [9] Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. Reinforce++: Stabilizing critic-free policy optimization with global advantage normalization, 2025. URL https://arxiv.org/abs/2501.03262.
- [10] Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models, 2024. URL https://arxiv.org/abs/ 2310.10505.

- [11] MAA. American mathematics contest 12 (amc 12), November 2023. URL https://artofproblemsolving.com/wiki/ index.php/AMC_12_Problems_and_Solutions.
- [12] MAA. American invitational mathematics examination (aime ), February 2024. URL https://artofproblemsolving. com/wiki/index.php/AIME_Problems_and_Solutions.
- [13] MAA. American invitational mathematics examination (aime ), February 2025. URL https://artofproblemsolving. com/wiki/index.php/AIME_Problems_and_Solutions.
- [14] Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using variational divergence minimization, 2016. URL https://arxiv.org/abs/1606.00709.
- [15] OpenAI, Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, Jonas Schneider, Szymon Sidor, Josh Tobin, Peter Welinder, Lilian Weng, and Wojciech Zaremba. Learning dexterous in-hand manipulation, 2019. URL https: //arxiv.org/abs/1808.00177.
- [16] OpenAI, Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemyslaw Debiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, Rafal Jozefowicz, Scott Gray, Catherine Olsson, Jakub Pachocki, Michael Petrov, Henrique P. d. O. Pinto, Jonathan Raiman, Tim Salimans, Jeremy Schlatter, Jonas Schneider, Szymon Sidor, Ilya Sutskever, Jie Tang, Filip Wolski, and Susan Zhang. Dota 2 with large scale deep reinforcement learning, 2019. URL https://arxiv.org/abs/1912.06680.
- [17] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. URL https://arxiv.org/abs/2203.02155.
- [18] Jaesung R. Park, Junsu Kim, Gyeongman Kim, Jinyoung Jo, Sean Choi, Jaewoong Cho, and Ernest K. Ryu. Clip-low increases entropy and clip-high decreases entropy in reinforcement learning of large language models, 2025. URL https://arxiv.org/abs/2509.26114.
- [19] Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel van de Panne. Deepmimic: example-guided deep reinforcement learning of physics-based character skills. ACM Transactions on Graphics, 37(4):1–14, July 2018. ISSN 1557-7368. doi: 10.1145/3197517.3201311. URL http://dx.doi.org/10.1145/3197517.3201311.

- [20] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.
- [21] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel. Trust region policy optimization, 2017. URL https://arxiv.org/abs/1502.05477.
- [22] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.
- [23] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation, 2018. URL https://arxiv.org/abs/1506.02438.
- [24] Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, and Luke Zettlemoyer. Spurious rewards: Rethinking training signals in rlvr, 2025. URL https://arxiv.org/abs/2506.10947.
- [25] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models,

2024. URL https://arxiv.org/abs/2402.03300.

- [26] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

- [27] Antonio Terpin, Nicolas Lanzetti, Batuhan Yardim, Florian Dörfler, and Giorgia Ramponi. Trust region policy optimization with optimal transport discrepancies: Duality and algorithm for continuous actions, 2022. URL https: //arxiv.org/abs/2210.11137.
- [28] Yuhui Wang, Hao He, Xiaoyang Tan, and Yaozhong Gan. Trust region-guided proximal policy optimization, 2019. URL https://arxiv.org/abs/1901.10314.
- [29] Shihui Yang, Chengfeng Dou, Peidong Guo, Kai Lu, Qiang Ju, Fei Deng, and Rihui Xin. Dcpo: Dynamic clipping policy optimization, 2025. URL https://arxiv.org/abs/2509.02333.
- [30] Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng Wu, Qingwei Guo, Qiaobo Chen, Yinyuting Yin, Hao Zhang, Tengfei Shi, Liang Wang, Qiang Fu, Wei Yang, and Lanxiao Huang. Mastering complex control in moba games with deep reinforcement learning, 2020. URL https://arxiv.org/abs/ 1912.09729.
- [31] Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising effectiveness of ppo in cooperative, multi-agent games, 2022. URL https://arxiv.org/abs/2103.01955.
- [32] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.
- [33] Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, Limao Xiong, Lu Chen, Zhiheng Xi, Nuo Xu, Wenbin Lai, Minghao Zhu, Cheng Chang, Zhangyue Yin, RongxiangWeng, WensenCheng, HaoranHuang, TianxiangSun, HangYan, TaoGui, QiZhang, XipengQiu, andXuanjing Huang. Secrets of rlhf in large language models part i: Ppo, 2023. URL https://arxiv.org/abs/2307.04964.

## Appendix

#### A Omitted Proofs

##### A.1 Proof of Lemma 1

Proof. Without loss of generality, we analyze the structural properties of the optimal solution for the maximization formulation in Problem (10). The derivation for the minimization case is identical due to the shared feasible set.

Let 𝑞 ≜ 𝑄(𝑎) be a fixed probability mass assigned to the target action 𝑎. To maximize the feasible range of 𝑞 under the divergence constraint, the optimal policy must distribute the remaining probability mass 1 − 𝑞 over the complement set 𝒱′ = 𝒱 \{𝑎} such that the divergence contribution is minimized. This sub-problem is formulated as:

𝑄(𝑏) 𝑃(𝑏)

min

𝑄(𝑏) = 1 − 𝑞. (24)

s.t.

𝑃(𝑏) 𝑓

{𝑄(𝑏)}𝑏∈𝒱′ 𝑏∈𝒱′

𝑏∈𝒱′

- 1. Optimality via Symmetry and Convexity. Observe that the objective function is a sum of strictly con-

vex terms, and the variables {𝑄(𝑏)}𝑏∈𝒱′ appear symmetrically in the objective (weighted by 𝑃(𝑏)) and the constraint. Due to the strict convexity of the generator function 𝑓 , the global minimum of this sub-problem must satisfy the symmetry condition where the likelihood ratio is constant across all complement actions. Formally, for any 𝑏𝑖, 𝑏𝑗 ∈ 𝒱′, optimality requires:

𝑄(𝑏𝑖) 𝑃(𝑏𝑖)

=

𝑄(𝑏𝑗) 𝑃(𝑏𝑗)

= 𝑐, (25)

for some scalar 𝑐 ≥ 0. If this were not true, one could construct a strictly better solution by averaging the ratios, thereby reducing the strictly convex objective (Jensen’s Inequality). Thus, the optimal structure is necessarily 𝑄(𝑏) = 𝑐𝑃(𝑏) for all 𝑏 ≠ 𝑎.

- 2. Determination of the Scaling Factor. Substituting 𝑄(𝑏) = 𝑐𝑃(𝑏) into the probability mass constraint 𝑏∈𝒱′ 𝑄(𝑏) = 1 − 𝑞 yields:

𝑐

𝑏∈𝒱′

𝑃(𝑏) = 1 − 𝑞 =⇒ 𝑐(1 − 𝑃(𝑎)) = 1 − 𝑞 =⇒ 𝑐 =

1 − 𝑞 1 − 𝑃(𝑎)

. (26)

Recall that 𝑞 = 𝑟𝑃(𝑎) where 𝑟 is the ratio at the target action. This recovers the scaling factor 𝑐(𝑟) = 11−−𝑟𝑃𝑃((𝑎𝑎)).

- 3. Strict Interiority. In the context of LLM post-training, the reference policy 𝜋old is derived from a Softmax distribution, ensuring 𝑃(𝑣) > 0 for all 𝑣 ∈ 𝒱. Furthermore, the trust-region radius 𝛿 is typically small (e.g.,

𝛿 ≪ 𝐷𝑓 (𝕀𝑎∥𝑃)), ensuring that the feasible set does not include the degenerate solution where 𝑄(𝑎) = 1 (which would imply an infinite or maximally large divergence). Consequently, we strictly have 𝑞 < 1, which implies

𝑐 = 1−1−𝑃(𝑞𝑎) > 0. Since 𝑃(𝑏) > 0 and 𝑐 > 0, it follows that 𝑄(𝑏) > 0 for all 𝑏. Thus, the optimal solution strictly resides in the interior of the probability simplex, justifying the validity of the derivation without requiring ad-hoc assumptions. □

##### A.2 Proof of Theorem 1

Proof. We aim to prove that the scalar function 𝑔𝑓 (𝑝, 𝑟) is strictly convex and that the roots of 𝑔𝑓 (𝑝, 𝑟) = 𝛿 uniquely determine the optimal clipping bounds.

- 1. Strict Convexity of 𝑔𝑓 (𝑝, 𝑟). Recall the definition of the scalarized divergence function:

𝑔𝑓 (𝑝, 𝑟) = 𝑝 𝑓 (𝑟) + (1 − 𝑝) 𝑓 (𝑐(𝑟)) , where 𝑐(𝑟) =

1 − 𝑟𝑝 1 − 𝑝

. (27)

We compute the first and second derivatives with respect to 𝑟. First, note that 𝑑𝑐𝑑𝑟(𝑟) = 1−−𝑝𝑝. The first derivative is:

𝜕𝑔𝑓 𝜕𝑟

= 𝑝 𝑓 ′(𝑟) + (1 − 𝑝) 𝑓 ′(𝑐(𝑟))

−𝑝 1 − 𝑝

= 𝑝 𝑓 ′(𝑟) − 𝑓 ′(𝑐(𝑟)) . (28)

The second derivative is:

𝜕2𝑔𝑓 𝜕𝑟2

= 𝑝 𝑓 ′′(𝑟) − 𝑝 𝑓 ′′(𝑐(𝑟))

−𝑝 1 − 𝑝

= 𝑝 𝑓 ′′(𝑟) +

𝑝2 1 − 𝑝

𝑓 ′′(𝑐(𝑟)). (29)

Since 𝑓 is strictly convex by definition, we have 𝑓 ′′(𝑥) > 0 for all 𝑥. Given that 𝑝 ∈ (0, 1), it follows that 𝜕2𝑔𝑓

𝜕𝑟2 > 0 for all valid 𝑟. Thus, 𝑔𝑓 (𝑝, 𝑟) is strictly convex.

- 2. Global Minimum. We examine the critical point where 𝜕𝜕𝑔𝑟𝑓 = 0. 𝑝[ 𝑓 ′(𝑟) − 𝑓 ′(𝑐(𝑟))] = 0 =⇒ 𝑓 ′(𝑟) = 𝑓 ′(𝑐(𝑟)). Since 𝑓 is strictly convex, 𝑓 ′ is strictly increasing and injective, implying 𝑟 = 𝑐(𝑟). Solving 𝑟 = 11−−𝑟𝑝𝑝

yields 𝑟(1 − 𝑝) = 1 − 𝑟𝑝 =⇒ 𝑟 = 1. At 𝑟 = 1, we have 𝑔𝑓 (𝑝, 1) = 𝑝 𝑓 (1) + (1 − 𝑝) 𝑓 (1) = 0 (since 𝑓 (1) = 0). Therefore, 𝑔𝑓 achieves its unique global minimum of 0 at 𝑟 = 1.

- 3. Existence and Uniqueness of Roots. Since 𝑔𝑓 (𝑝, 𝑟) is strictly convex and minimal at 𝑟 = 1, it is strictly decreasing on the interval [0, 1) and strictly increasing on (1, 1/𝑝]. For any valid trust region 𝛿 > 0:

- • At the minimum, 𝑔𝑓 (𝑝, 1) = 0 < 𝛿.
- • As 𝑟 moves away from 1 towards the boundaries (0 or 1/𝑝), 𝑔𝑓 increases strictly. Assuming 𝛿 is within the feasible range of the divergence (which is true for standard small 𝛿), the Intermediate Value Theorem guarantees exactly one root in [0, 1) (denoted 𝑟) and exactly one root in (1, 1/𝑝] (denoted 𝑟).

Thus, the inequality constraint 𝐷𝑓 (𝑄∥𝑃) ≤ 𝛿, which is equivalent to 𝑔𝑓 (𝑝, 𝑟) ≤ 𝛿, corresponds precisely to the interval 𝑟 ∈ [𝑟, 𝑟]. Since the objective is to maximize (or minimize) 𝑟, the solution must lie on the boundary of the feasible interval. Thus, the optimal bounds are strictly defined by the unique roots of the binding equation 𝑔𝑓 (𝑝, 𝑟) = 𝛿. □

##### A.3 Proof of Proposition 1 (Asymptotic Behavior)

Proof. We investigate the limiting behavior of the optimal bounds 𝑟 𝑓 ,𝛿(𝑝) and 𝑟 𝑓 ,𝛿(𝑝) by analyzing the feasibility set defined by 𝑔𝑓 (𝑝, 𝑟) ≤ 𝛿.

- 1. Tail Expansion (𝑝 → 0+). Recall the scalarized divergence 𝑔𝑓 (𝑝, 𝑟) = 𝑝 𝑓 (𝑟)+(1−𝑝) 𝑓 (𝑐(𝑟)) with 𝑐(𝑟) = 11−−𝑟𝑝𝑝 . As 𝑝 → 0+, for any fixed finite 𝑟, we have 𝑐(𝑟) → 1. Since 𝑓 (1) = 0, the complement term (1 − 𝑝) 𝑓 (𝑐(𝑟)) vanishes. The constraint asymptotically behaves as 𝑝 𝑓 (𝑟) ≤ 𝛿.

- • Upper Bound (𝑟): For 𝑟 > 1, convex 𝑓 grows superlinearly (or at least linearly for valid divergences). The binding condition implies 𝑓 (𝑟) ≈ 𝛿/𝑝. As 𝑝 → 0+, 𝛿/𝑝 → +∞, forcing 𝑓 (𝑟) → +∞ and consequently 𝑟 → +∞.

- • Lower Bound (𝑟): The lower bound is defined as 𝑟 = min{𝑟 ∈ [0, 1] | 𝑔𝑓 (𝑝, 𝑟) ≤ 𝛿}. As 𝑝 → 0, the maximum divergence contribution from the target action is bounded by 𝑝 𝑓 (0) (assuming 𝑟 = 0). If lim𝑝→0 𝑝 𝑓 (0) =

- 0 < 𝛿, then for sufficiently small 𝑝, the inequality 𝑔𝑓 (𝑝, 𝑟) ≤ 𝛿 holds for all 𝑟 ∈ [0, 1]. In this regime, the constraint is inactive, and the minimization is solely determined by the simplex boundary 𝑟 ≥ 0. Thus, lim𝑝→0 𝑟 = 0.

- 2. Head Constriction (𝑝 → 1−). We analyze the distinct limiting behaviors for the upper and lower bounds separately as 𝑝 → 1−.

- • Upper Bound (𝑟 → 1): By definition, 𝑄(𝑎) = 𝑟𝑃(𝑎) = 𝑟𝑝. Since 𝑄 ∈ Δ𝑉, we strictly require 𝑄(𝑎) ≤ 1, implying the feasible domain for 𝑟 is 𝑟 ∈ [0, 1/𝑝]. As 𝑝 → 1−, the feasible upper limit 1/𝑝 → 1. Since the upper bound must theoretically satisfy 𝑟 ≥ 1, the domain constraint forcefully squeezes it: 1 ≤ lim𝑝→1− 𝑟 ≤ lim𝑝→1−(1/𝑝) = 1. Thus, lim𝑝→1− 𝑟 = 1.

- • Lower Bound (𝑟 → 𝑟∗): Unlike the upper bound, the lower bound 𝑟 ≤ 1 operates within [0, 1] and is not squeezed by the simplex boundary as 𝑝 → 1−. Instead, it converges to a constant 𝑟∗ governed by

the asymptotic properties of the 𝑓 -divergence. Let 𝐶∞ ≜ lim𝑢→+∞ 𝑓(𝑢𝑢) denote the asymptotic linear growth rate. As 𝑝 → 1−, assuming 𝑟 < 1, the complement ratio 𝑢 = 11−−𝑟𝑝𝑝 → +∞. We can rewrite the complement divergence term from the binding equation as:

𝑓 (𝑢) 𝑢

𝑓 (𝑢) 𝑢

𝑝→1−

(1 − 𝑝) 𝑓 (𝑢) = (1 − 𝑝)𝑢

= 1 − 𝑝 + 𝑝(1 − 𝑟)

−−−−→ (1 − 𝑟)𝐶∞. (30)

Taking the limit 𝑝 → 1− on both sides of the scalarized binding equation 𝑔𝑓 (𝑝, 𝑟) = 𝛿, we obtain: 𝑓 (𝑟∗) + (1 − 𝑟∗)𝐶∞ = 𝛿, where 𝑟∗ = lim

𝑟 𝑓 ,𝛿(𝑝). (31)

𝑝→1−

Depending on the specific 𝑓 -divergence employed, solving this exact limit equation yields distinct properties for the lower bound:

- – Total Variation (TV): The generator 𝑓𝑇𝑉(𝑢) = 12|𝑢 − 1| yields 𝐶∞ = lim𝑢→+∞ 𝑢2−𝑢1 = 12. The limit equation becomes 21(1 − 𝑟∗) + 12(1 − 𝑟∗) = 𝛿 =⇒ 1 − 𝑟∗ = 𝛿. Thus, 𝑟∗ = max(1 − 𝛿, 0).

- – Pearson 𝜒2 Divergence: The generator 𝑓𝜒2(𝑢) = (𝑢 − 1)2 grows super-linearly, yielding 𝐶∞ = +∞. To maintain a finite budget 𝛿, the term (1 − 𝑟∗) · (+∞) forces 1 − 𝑟∗ = 0. Thus, 𝑟∗ = 1.
- – KL Divergence (TRPO/PPO): The standard generator 𝑓𝐾𝐿(𝑢) = −log 𝑢 + 𝑢 − 1 yields 𝐶∞ =

lim𝑢→+∞ −log𝑢𝑢+𝑢−1 = 1. The limit equation becomes (−log 𝑟∗ + 𝑟∗ − 1) + 1 · (1 − 𝑟∗) = 𝛿 =⇒ −log 𝑟∗ = 𝛿. Thus, 𝑟∗ = 𝑒−𝛿.

This theoretical derivation perfectly aligns with the empirical lower bounds observed at 𝑝 → 1 in Figure 2.

This completes the proof. □ □

##### A.4 Proof of Proposition 2 (Monotonicity)

Proof. We establish the strict monotonicity of the roots of 𝐹(𝑝, 𝑟) ≜ 𝑔𝑓 (𝑝, 𝑟)−𝛿 = 0 using the Implicit Function Theorem. The sensitivity of the root 𝑟 with respect to 𝑝 is given by:

𝜕𝐹/𝜕𝑝 𝜕𝐹/𝜕𝑟

𝑑𝑟 𝑑𝑝

. (32)

= −

- 1. Positivity of 𝜕𝜕𝐹𝑝 via Bregman Divergence. Differentiating 𝑔𝑓 with respect to 𝑝, and noting that 𝜕𝜕𝑝𝑐 = (11−−𝑝𝑟)2 , we derive:

𝜕𝑐 𝜕𝑝

𝜕𝐹 𝜕𝑝

= 𝑓 (𝑟) − 𝑓 (𝑐) + (1 − 𝑝) 𝑓 ′(𝑐)

1 − 𝑟 1 − 𝑝

. (33)

= 𝑓 (𝑟) − 𝑓 (𝑐) + 𝑓 ′(𝑐)

Using the algebraic identity 𝑐 − 𝑟 = 11−−𝑟𝑝𝑝 − 𝑟 = 11−−𝑝𝑟 , we substitute this into the expression:

𝜕𝐹 𝜕𝑝

= 𝑓 (𝑟) − 𝑓 (𝑐) − 𝑓 ′(𝑐)(𝑟 − 𝑐) ≡ 𝐷𝑓 (𝑟, 𝑐), (34)

where 𝐷𝑓 (𝑟, 𝑐) is the Bregman divergence generated by the strictly convex function 𝑓 . By the strict convexity of 𝑓 , the Bregman divergence is strictly positive for all 𝑟 ≠ 𝑐. Since 𝑟 ≠ 𝑐 holds strictly for any 𝑟 ≠ 1 (which

implies 𝑝 < 1 and 𝛿 > 0), we conclude that 𝜕𝜕𝐹𝑝 > 0 always.

- 2. Sign Analysis of 𝜕𝜕𝐹𝑟 . Differentiation with respect to 𝑟 yields:

𝜕𝐹 𝜕𝑟

= 𝑝 𝑓 ′(𝑟) + (1 − 𝑝) 𝑓 ′(𝑐)

−𝑝 1 − 𝑝

= 𝑝[ 𝑓 ′(𝑟) − 𝑓 ′(𝑐)]. (35)

Since 𝑓 is strictly convex, 𝑓 ′ is strictly increasing. We analyze the two roots:

- • Upper Bound (𝑟 > 1): Here 𝑟 > 1. The complement ratio satisfies 𝑐 = 11−−𝑟𝑝𝑝 < 1 (since 𝑟 > 1 =⇒ 1 − 𝑟𝑝 < 1 − 𝑝). Thus 𝑟 > 𝑐, implying 𝑓 ′(𝑟) > 𝑓 ′(𝑐). Hence, 𝜕𝜕𝐹𝑟 𝑟 > 0.

- • Lower Bound (𝑟 < 1): Here 𝑟 < 1. It follows that 𝑐 > 1. Thus 𝑟 < 𝑐, implying 𝑓 ′(𝑟) < 𝑓 ′(𝑐). Hence,

𝜕𝐹 𝜕𝑟 𝑟 < 0.

- 3. Conclusion. Applying the implicit derivative signs:

(+) (+)

𝑑𝑟 𝑑𝑝

< 0 =⇒ Strictly Decreasing, (36)

= −

𝑑𝑟 𝑑𝑝

(+) (−)

> 0 =⇒ Strictly Increasing. (37)

= −

This completes the proof. □

##### A.5 Proof of Proposition 3 (Constraint Activity and Saturation)

Proof. We analyze the existence of interior roots for the binding equation 𝑔𝑓 (𝑝, 𝑟) = 𝛿 relative to the simplex boundaries. Recall from Theorem 1 that 𝑔𝑓 (𝑝, 𝑟) is strictly convex with a global minimum 𝑔𝑓 (𝑝, 1) = 0.

Upper Bound Saturation. Consider the function 𝑔𝑓 (𝑝, 𝑟) on the interval [1, 𝑟max], where 𝑟max = 1/𝑝. Since 𝑔𝑓 is strictly increasing for 𝑟 > 1, the maximum divergence is attained at the boundary: 𝑔peak ≜ 𝑔𝑓 (𝑝, 𝑟max).

- • Inactive Regime (𝑔peak ≤ 𝛿): If the maximal divergence is within the trust region, then 𝑔𝑓 (𝑝, 𝑟) ≤ 𝑔peak ≤ 𝛿 holds for all 𝑟 ∈ [1, 𝑟max]. The constraint is inactive. Since the objective in Eq. (10) is to maximize the ratio, the optimal solution saturates at the boundary: 𝑟 𝑓 ,𝛿(𝑝) = 𝑟max. In this case, no interior root exists for 𝑔𝑓 (𝑝, 𝑟) = 𝛿 (unless 𝛿 = 𝑔peak, where the root is the boundary itself).

- • Active Regime (𝑔peak > 𝛿): Here we have 𝑔𝑓 (𝑝, 1) = 0 < 𝛿 and 𝑔𝑓 (𝑝, 𝑟max) > 𝛿. Since 𝑔𝑓 is continuous on [1, 𝑟max], the Intermediate Value Theorem guarantees the existence of a root 𝑟† ∈ (1, 𝑟max) such that 𝑔𝑓 (𝑝, 𝑟†) = 𝛿. Furthermore, due to the strict monotonicity of 𝑔𝑓 on this interval, this root is unique.

Lower Bound Saturation. Symmetrically, consider the interval [𝑟min, 1] where 𝑟min = 0. 𝑔𝑓 (𝑝, 𝑟) is strictly decreasing on this interval.

- • Inactive Regime (𝑔𝑓 (𝑝, 𝑟min) ≤ 𝛿): The constraint 𝑔𝑓 (𝑝, 𝑟) ≤ 𝛿 holds for all 𝑟 ∈ [0, 1]. Minimizing 𝑟 leads to saturation at the boundary: 𝑟 𝑓 ,𝛿(𝑝) = 0.

- • Active Regime (𝑔𝑓 (𝑝, 𝑟min) > 𝛿): Since 𝑔𝑓 (𝑝, 1) = 0 < 𝛿 and 𝑔𝑓 (𝑝, 0) > 𝛿, there exists a unique root in (0, 1).

Combining these cases yields the formulation in Proposition 3. □

##### A.6 Numerical Implementation Details

This section details the numerical procedures for solving the Band bounds in the general case where closedform solutions are unavailable (e.g., KL divergence).

###### A.6.1 Convergence Guarantees

We first establish the theoretical basis for the global convergence of bracketed root-finding methods on the scalarized constraint function.

Lemma 2 (Strict Monotonicity on Intervals). Let 𝑓 : ℝ+ → ℝ be strictly convex with a unique global minimum at 𝑓 (1) = 0. The scalarized constraint function

1 − 𝑟𝑝 1 − 𝑝

𝑔𝑓 (𝑝, 𝑟) ≜ 𝑝 𝑓 (𝑟) + (1 − 𝑝) 𝑓

(38)

is strictly convex with respect to 𝑟. Specifically, 𝑔𝑓 (𝑝, 𝑟) is strictly decreasing on the interval [𝑟min, 1] and strictly increasing on [1, 𝑟max].

Proof. The strict convexity of 𝑔𝑓 follows directly from the strict convexity of 𝑓 and the linearity of the argument transformation 𝑐(𝑟) (as shown in Theorem 1). Since 𝑔𝑓 (𝑝, 𝑟) is strictly convex and differentiable on its domain, and attains its global minimum 𝑔𝑓 (𝑝, 1) = 0, the gradient ∇𝑟𝑔𝑓 (𝑝, 𝑟) must be strictly negative for 𝑟 < 1 and strictly positive for 𝑟 > 1. This strictly implies the monotonicity properties required for root isolation. □

Theorem 2 (Global Convergence of Bisection). Consider the active regime where min(𝑔𝑓 (𝑝, 𝑟min), 𝑔𝑓 (𝑝, 𝑟max)) > 𝛿. The equation 𝑔𝑓 (𝑝, 𝑟) = 𝛿 possesses exactly two unique roots: a lower bound 𝑟 ∈ (𝑟min, 1) and an upper bound 𝑟 ∈ (1, 𝑟max). Applying the bisection method on the brackets [𝑟min, 1] and [1, 𝑟max] guarantees linear convergence to 𝑟 and 𝑟, respectively, to any arbitrary precision 𝜖.

Proof. By Lemma 2, 𝑔𝑓 (𝑝, 𝑟) is continuous and strictly monotonic on each identified bracket. In the active regime, the function values at the bracket endpoints strictly enclose the target value 𝛿 (since 𝑔𝑓 (𝑝, 1) = 0 < 𝛿 and the boundary values exceed 𝛿). The Intermediate Value Theorem (IVT) guarantees the existence of a root in each interval, and strict monotonicity guarantees its uniqueness. Standard convergence analysis of the bisection method applies directly. □

###### A.6.2 Standard Bisection Solver Algorithms

We present the standard bisection procedure to solve 𝑔𝑓 (𝑝, 𝑟) − 𝛿 = 0. Let 𝜖 be the convergence tolerance (e.g., 10−6).

Upper Bound Solver (𝑟). The objective function ℎ(𝑟) ≜ 𝑔𝑓 (𝑝, 𝑟) − 𝛿 is strictly increasing on the interval [1, 𝑟max].

- • Initialize: 𝐿 ← 1, 𝑅 ← 1/𝑝.
- • Iterate until |𝑅 − 𝐿| < 𝜖:

𝐿 + 𝑅 2

; if 𝑔𝑓 (𝑝, 𝑚) ≤ 𝛿, set 𝐿 ← 𝑚; else 𝑅 ← 𝑚. (39)

𝑚 ←

- • Return: 𝐿 (approximating 𝑟 𝑓 ,𝛿).

Lower Bound Solver (𝑟). The objective function is strictly decreasing on the interval [0, 1]. Note the reversed conditional update logic compared to the upper bound.

- • Initialize: 𝐿 ← 0, 𝑅 ← 1.
- • Iterate until |𝑅 − 𝐿| < 𝜖:

𝑚 ←

𝐿 + 𝑅 2

; if 𝑔𝑓 (𝑝, 𝑚) ≤ 𝛿, set 𝑅 ← 𝑚; else 𝐿 ← 𝑚. (40)

- • Return: 𝑅 (approximating 𝑟 𝑓 ,𝛿).

###### A.6.3 Instantiation: Forward KL Divergence

We illustrate the solver with the forward KL divergence, which is the standard metric for TRPO. This trust region is recovered by the generator function 𝑓𝐾𝐿(𝑢) = −log 𝑢 + 𝑢 − 1. Substituting 𝑓𝐾𝐿 into the scalarized form yields the binding equation:

𝑝(−log 𝑟 + 𝑟 − 1) + (1 − 𝑝) − log 𝑐(𝑟) + 𝑐(𝑟) − 1 = 𝛿, (41)

where 𝑐(𝑟) = 11−−𝑟𝑝𝑝 is the complement scaling factor. Simplifying the linear terms (which exactly cancel out to zero: 𝑝𝑟 − 𝑝 + 1 − 𝑝𝑟 − 1 + 𝑝 = 0), this reduces to a transcendental equation strictly dominated by the logarithmic terms:

1 − 𝑟𝑝 1 − 𝑝

−𝑝 log 𝑟 − (1 − 𝑝)log

= 𝛿. (42)

As this equation involves 𝑟 logarithmically both inside and outside the complement scaling argument, it admits no closed-form solution in terms of elementary functions. Therefore, the generic numerical solver described in Appendix A.6.2 is necessary to compute the exact Band bounds for KL-based trust regions.

##### A.7 Proof of Proposition

Proof. Recall from Theorem 1 that the exact bounds 𝑟 𝑓 ,𝛿(𝑝) and 𝑟 𝑓 ,𝛿(𝑝) correspond to the roots of the scalar binding equation:

1 − 𝑟𝑝 1 − 𝑝

𝑔𝑓 (𝑝, 𝑟) ≜ 𝑝 𝑓 (𝑟) + (1 − 𝑝) 𝑓

= 𝛿. (43)

We substitute the specific generator functions for Total Variation and Pearson 𝜒2 to derive the solutions.

- 1. Total Variation (TV). The generator function is defined as 𝑓TV(𝑢) = 12|𝑢 −1|. Substituting this into Eq. (43) gives:

- 1 − 𝑝

- 2

1 − 𝑟𝑝

𝑝 2|𝑟 − 1| +

1 − 𝑝 − 1 = 𝛿. (44) We simplify the complement term. Note that:

𝑔TV(𝑝, 𝑟) =

1 − 𝑟𝑝 1 − 𝑝 − 1 =

1 − 𝑟𝑝 − (1 − 𝑝) 1 − 𝑝

=

Thus, the second term in the binding equation simplifies to:

𝑝(1 − 𝑟) 1 − 𝑝

. (45)

- 1 − 𝑝

- 2

𝑝(1 − 𝑟) 1 − 𝑝

- 1 − 𝑝

- 2 ·

𝑝 1 − 𝑝 |1 − 𝑟| =

𝑝 2|𝑟 − 1|. (46)

=

Substituting this back yields a unified expression:

𝑝 2|𝑟 − 1| +

𝑔TV(𝑝, 𝑟) =

Setting 𝑔TV(𝑝, 𝑟) = 𝛿, we solve for 𝑟:

𝛿 𝑝

|𝑟 − 1| =

𝑝 2|𝑟 − 1| = 𝑝|𝑟 − 1|. (47)

=⇒ 𝑟 = 1 ±

𝛿 𝑝

. (48)

- 2. Pearson 𝜒2 Divergence. The generator function is 𝑓𝜒2(𝑢) = (𝑢 − 1)2. Substituting this into Eq. (43):

2

1 − 𝑟𝑝 1 − 𝑝 − 1

𝑔𝜒2(𝑝, 𝑟) = 𝑝(𝑟 − 1)2 + (1 − 𝑝)

= 𝛿. (49)

Using the same algebraic simplification for the complement term as in the TV case: 1 − 𝑟𝑝 1 − 𝑝 − 1

2

2

𝑝2 (1 − 𝑝)2

𝑝(1 − 𝑟) 1 − 𝑝

(𝑟 − 1)2. (50)

=

=

Substituting this back into 𝑔𝜒2(𝑝, 𝑟):

𝑝2 (1 − 𝑝)2

𝑔𝜒2(𝑝, 𝑟) = 𝑝(𝑟 − 1)2 + (1 − 𝑝)

(𝑟 − 1)2 (51)

𝑝 1 − 𝑝

= 𝑝(𝑟 − 1)2 1 +

(52)

1 − 𝑝 + 𝑝 1 − 𝑝

= 𝑝(𝑟 − 1)2

(53)

𝑝

1 − 𝑝(𝑟 − 1)2. (54) Setting 𝑔𝜒2(𝑝, 𝑟) = 𝛿 and solving for 𝑟:

=

1 − 𝑝 𝑝

(𝑟 − 1)2 = 𝛿

=⇒ 𝑟 = 1 ±

𝛿(1 − 𝑝) 𝑝

. (55)

Practical Implementation (Simplex Constraints). The theoretical bounds derived above assume the trust region lies entirely within the simplex. In practice, to ensure global feasibility (i.e., 𝑄(𝑎) ∈ [0, 1]), the effective bounds must be clamped. Applying the saturation logic from Proposition 3, the final operational bounds are:

1 𝑝

𝑟★𝑓,𝛿(𝑝) ≜ min 𝑟 𝑓,𝛿(𝑝),

, 𝑟★𝑓,𝛿(𝑝) ≜ max 𝑟 𝑓,𝛿(𝑝), 0 . (56) □

