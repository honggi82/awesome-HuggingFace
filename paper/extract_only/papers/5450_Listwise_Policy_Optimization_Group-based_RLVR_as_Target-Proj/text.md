# arXiv:2605.06139v2[cs.LG]20May2026

[Figure 1]

2026-05-21

## Listwise Policy Optimization: Group-based RLVR as Target-Projection on the LLM Response Simplex

Yun Qu1,2,∗Qi Wang1,†, Yixiu Mao1, Heming Zou1,2,∗, Yuhang Jiang1, Yingyue Li1, Wutong Xu1, Lizhou Cai1, Weijie Liu2, Clive Bai2, Kai Yang2, Yangkun Chen2, Saiyong Yang2,†, Xiangyang Ji1,†

1Department of Automation, Tsinghua University 2LLM Department, Tencent cheemswang@mail.tsinghua.edu.cn, stevesyang@tencent.com, xyji@tsinghua.edu.cn

#### Abstract

Reinforcement learning with verifiable rewards (RLVR) has become a standard approach for large language models (LLMs) post-training to incentivize reasoning capacity. Among existing recipes, group-based policy gradient is prevalent, which samples a group of responses per prompt and updates the policy via group-relative advantage signals. This work reveals that these optimization strategies share a common geometric structure: each implicitly defines a target distribution on the response simplex and projects toward it via first-order approximation. Building on this insight, we propose Listwise Policy Optimization (LPO) to explicitly conduct the target-projection, which demystifies the implicit target by restricting the proximal RL objective to the response simplex, and then projects the policy via exact divergence minimization. This framework provides (i) monotonic improvement on the listwise objective with bounded, zero-sum, and self-correcting projection gradients, and (ii) flexibility in divergence selection with distinct structural properties through the decoupled projection step. On diverse reasoning tasks and LLM backbones, LPO consistently improves training performance over typical policy gradient baselines under matched targets, while intrinsically preserving optimization stability and response diversity.

#### 1 Introduction

Recent advances have revealed the prominent potential of reinforcement learning with verifiable rewards (RLVR) for large language models (LLMs) post-training, which incentivizes reasoning capabilities on complex problem-solving tasks (Guo et al., 2025; Jaech et al., 2024; Luo et al., 2025). In particular, critic-free, group-based RL paradigms, such as group relative policy optimization (GRPO) (Shao et al., 2024), have been widely adopted for RLVR. This setup samples a group of responses, scores them with a verifier, and performs policy gradient updates using group-relative advantages. Further extensions in the literature (Liu et al., 2025b; Yu et al., 2025; Tajwar et al., 2026; Hu, 2025; Chen et al., 2025) have introduced critical refinements with special focus on advantage normalization and training stabilization.

[Figure 2]

Figure 1: LPO iteratively ascends the reward landscape via explicit targetprojection, enabling stable optimization and flexible divergence design.

Group-based policy gradients as implicit target-projections. Though these empirical refinements have proven effective, viewing them purely through the manner of advantage normalization obscures the intrinsic optimization mechanism. By defining a listwise distribution (Cao et al., 2007; Liu et al., 2025a) jointly over the sampled responses on a simplex, this work provides a unified geometric perspective on group-based RL algorithms: their advantage formulas implicitly construct a reward-weighted softmax target distribution over the responses, with the target’s sharpness configured by the normalization scheme. Then, the standard policy gradient update acts merely as a first-order approximation of a reverse Kullback-Leibler (KL) (Kullback, 1951) projection toward this implicit target. This integrated perspective not only elucidates the workings of current methods but also motivates the explicit design of the target-projection mechanism.

From implicit approximation to explicit projection. Explicit target projection has been studied in classical RL (Peters et al., 2010; Abdolmaleki et al., 2018; Peng et al., 2019). However, the existence of continuous action spaces necessitates the use of function approximation. In contrast, group-based RLVR exhibits a distinct and desirable property: the sampled responses for a prompt naturally form a finite simplex, allowing for the exact computation of both the target distribution and the projection in closed form. This makes it feasible to define clear separated goals between what distribution to target and how to project toward it, facilitating a seamless transition from implicit approximations to exact listwise optimization. Consequently, the central research question arises:

∗ Work completed during an internship at Tencent. † Corresponding Authors

What properties emerge when this target-projection is made explicit, and how does this decoupled optimization space influence RLVR of LLMs?

Listwise Policy Optimization. In response to the above research question, this work develops Listwise Policy Optimization (LPO) to enable explicit target-projection on the response simplex. Specifically, LPO (i) explicates the implicit target by constraining the proximal RL objective to the sampled responses, yielding a closed-form solution with a controllable temperature, and (ii) optimizes the policy by projecting it onto the target via divergence minimization on the response simplex. The exact projection onto the simplex results in gradients that are bounded, zero-sum, and self-correcting by design, which induces variance reduction and stable optimization. Furthermore, the decoupled structure allows for flexible projection divergences, and we implement forward and reverse KL divergence as two representative instantiations. The resulting iterative target-projection algorithm provides provable monotonic improvement of the listwise reward per iteration.

Contributions. This work aims to offer deeper insights into policy optimization in RLVR, focusing on understanding and identifying potential improvements. The main contribution is two-fold:

- 1. We provide a unifying analytical perspective, revealing that group-based policy gradient methods implicitly perform approximate target-projections on the response simplex.
- 2. We develop LPO, an explicit target-projection framework that decouples listwise target construction from divergence projection, supported by theoretical analysis that proves improvement guarantee and characterizes projections’ structural properties.

Extensive evaluations across logic, mathematics, programming, and multi-modal reasoning tasks with diverse LLM backbones demonstrate the effectiveness of LPO: (i) LPO achieves higher expected Pass@1 and Pass@k accuracy during training compared to baselines under matched implicit target constructions; (ii) decoupling the target from the projection accommodates diverse divergences, with a novel forward KL variant showing exceptional competitiveness; and (iii) LPO induces highly stable optimization trajectories while inherently preserving response diversity.

#### 2 Preliminaries

###### 2.1 Reinforcement Learning with Verifiable Rewards

RLVR has emerged as a critical post-training paradigm for incentivizing reasoning capabilities of LLMs (Shao et al., 2024; Jaech et al., 2024). Let x denote a prompt and y = (y1, . . . , y|y|) a response of length |y|, generated

autoregressively by a parameterized policy πθ(y|x) = ∏i|=y|1 πθ(yi|x, y<i). Given a reward function R(x, y) and a reference policy πref, the standard KL-regularized objective for RLVR (Shao et al., 2024) is defined as:

Jx(πθ) = Ey∼πθ(·|x)[R(x, y)] − β DKL πθ(·|x) ∥ πref(·|x) , (1)

where β ≥ 0 controls the strength of the reference constraint. Following recent advances (Yu et al., 2025; Qu et al., 2025), we primarily focus on rule-based outcome rewards, which are typically binary or sparse (R ∈ [0,1]), without an explicit reference penalty, i.e., β = 0.

###### 2.2 Group-based Policy Gradient

The dominant paradigm in RLVR is group-based policy gradient (PG), represented by Group-Relative Policy Optimization (GRPO) (Shao et al., 2024). For each prompt x, a behavior policy πb, which is typically the pre-update snapshot πθold, generates a group of K responses {y1, . . . , yK}, each assigned a reward Rk forming the reward vector R = [R1, . . . , RK]⊤. These rewards are converted into group-relative advantages, forming the advantage vector A = [A1, . . . , AK]⊤ via centering and scaling. For instance, GRPO uses Ak = Rkσ−µG

, where µG and σG are the group mean and standard deviation. Table 1 details other common normalization schemes. The policy is typically updated by maximizing a clipped surrogate objective (Schulman et al., 2017b; Shao et al., 2024):

G

|yk|

K

1 K

1 |yk|

### ∑

### ∑

min rk,i Ak, clip(rk,i, 1−ϵ, 1+ϵ) Ak , (2)

LGRPO(θ) = Ex,{y

k}kK=1∼πb(·|x)

i=1

k=1

where rk,i(θ) = ππθ(yk,i|x,yk,<i)

b(yk,i|x,yk,<i) is the importance ratio and ϵ is the clipping hyperparameter.

At the exact on-policy point (πθ = πb), the importance ratios are identically one (rk,i = 1). Consequently, for a fixed prompt x, the surrogate objective gradient reduces to the standard sequence-level group-based policy

Table 1: Advantages and implicit targets of existing group-based policy gradient methods.

Algorithm Advantage Ak Implicit target w∗ Temperature τ Dr.GRPO (Liu et al., 2025b) / RLOO (Ahmadian et al., 2024) Rk − µG softmax(R) ∼ 1 GRPO (Shao et al., 2024) / DAPO (Yu et al., 2025) (Rk − µG)/σG softmax(R/σG) σG MaxRL (Tajwar et al., 2026) (Rk − µG)/µG softmax(R/µG) µG

gradient (Sutton et al., 1999):

1 K

gPG =

K

1 |yk|

### ∑

Ak ∇θ log πθ(yk|x), where log πθ(yk|x) ≜

k=1

|yk|

### ∑

log πθ(yk,i|x, yk,<i). (3)

i=1

#### 3 Group-based Policy Gradient as Implicit Target-Projection

This section reinterprets group-based policy gradients through the lens of the listwise distribution. We aim to explore: (i) the target distribution that these updates implicitly pursue, and (ii) the impact of different advantage normalization schemes on shaping that target.

###### 3.1 Listwise Distribution on the Response Simplex

To formalize, we represent the policy’s relative preference over the K sampled responses for prompt x as a listwise distribution Pθ (Cao et al., 2007; Rafailov et al., 2024; Liu et al., 2025a):

exp(sθ,k) ∑Kj=1 exp(sθ,j)

πθ(yk|x) πb(yk|x)

= softmax(sθ)k, with sθ,k = log

, (4)

Pθ,k =

where Pθ reflects the extent to which πθ prioritizes each response relative to πb. At the on-policy point (πθ = πb), Pθ reduces to the uniform distribution 1/K. Since Pθ,k ≥ 0 and ∑k Pθ,k = 1, the vector Pθ lies on the probability simplex ∆K−1 = {p ∈ RK : pk ≥ 0, ∑k pk = 1}, which we call the response simplex.

###### 3.2 Group-based Policy Gradient as Approximate Reverse KL

With the listwise distribution, we now reveal the underlying geometric property: standard group-based policy gradients implicitly perform target-projection via reverse Kullback-Leibler (KL) (Kullback, 1951) minimization.

- Proposition 1 (Group-based policy gradient as reverse KL at on-policy). Let A ∈ RK be a zero-mean advantage

vector, i.e., ∑kK=1 Ak = 0, and let w∗ = softmax(A). At the on-policy point (πθ = πb), the policy gradient in Eq. 3 equals the negative gradient of the reverse KL divergence DKL:

1 K

gPG =

K

∑

Ak ∇θ log πθ(yk|x) = −∇θ DKL(Pθ∥w∗)

k=1

. (5)

πθ=πb

This observation identifies w∗ = softmax(A) as the implicit target on the response simplex induced by the advantage design. This equivalence is exact at the on-policy point, but the approximation error grows as the policy drifts from the sampling distribution. Concretely, the per-response coefficient discrepancy scales as O(δ¯ · (1 +

∥A∥∞)/K), where δ¯ = maxk |ππθ(yk|x)

b(yk|x) − 1| measures the degree of off-policy drift. See Appendix B.2 for detailed proof.

###### 3.3 Implicit Targets of Existing Methods

Table 1 summarizes the specific implicit targets induced by existing group-based PG algorithms. Advantages in these methods take the form Ak = (Rk − µ)/τ for various choices of centering µ and scaling τ. By the shift-invariance of softmax, the centering cancels and the target w∗ reduces to softmax(R/τ), where τ acts as a temperature. Different normalization schemes thus preserve the same reward ordering with the main difference in target sharpness, as detailed in Appendix C.3.

From approximation to exact projection. This unifying view also suggests a natural refinement. Since both the target w∗ and the listwise distribution Pθ lie on the finite response simplex, the projection can be performed in an exact manner. Moreover, it provides a new lens on algorithm design worth investigating: exact projection allows for any statistical divergence, e.g., Forward KL, that were inaccessible under the current policy gradient paradigm. Accordingly, the next section will develop a generalized framework.

#### 4 Listwise Policy Optimization

We now replace implicit policy gradient approximations with an explicit target-projection framework on the response simplex. This framework decouples each iteration into two entangled steps:

Jˆ(w)

w∗ = arg max

θ′ = argmin

D(w∗ ∥ Pθ)

(6)

w∈∆K−1

θ

(ii) Projection: how to optimize toward it

(i) Target: what distribution to aim for

where Jˆ is a proximal objective on the simplex and D is a divergence measure. Next, we will detail the optimization steps, their implementation, and the theoretical analysis.

###### 4.1 Target Induced on the Response Simplex

To demystify the principled origin of the implicit target in group-based policy gradients, we define a local proximal RL objective per prompt on the response simplex, which maximizes the expected reward subject to a trust region around the policy (Schulman et al., 2017a):

K

Jˆ(w) =

∑

max

wkRk − τ DKL(w∥Pt), (7)

w∈∆K−1

k=1

where Pt = softmax(st) is the listwise distribution induced by the pre-update policy πt, with st,k = log πt(yk|x)/πb(yk|x) . Equivalently, Pt is Pθ from Eq. 4 evaluated at θ = θt. Both Pt and st are held fixed while θ is updated. Theorem 1 (Listwise Gibbs target). The objective Jˆ(w) in Eq. (7) has a unique maximizer w∗:

Rk τ

w∗

+ st,k. (8)

k = softmax(ϕ)k, with ϕk =

- Theorem 1 indicates that the target w∗ re-weights the baseline Pt toward high-reward responses, with τ controlling

the sharpness: w∗ → argmaxk Rk as τ → 0, and w∗ → Pt as τ → ∞. Under the on-policy setup (πt = πb), Pt degenerates to a uniform distribution and w∗ = softmax(R/τ) recovers the implicit targets of existing methods (Proposition 1), with τ now an explicit design parameter with trust-region interpretation rather than a byproduct of advantage normalization. Notably, concurrent works (Kaddour, 2026; Shu et al., 2026) explore a similar paradigm by explicitly formulating the target as a reward-based Gibbs distribution with varied reference policies.

As K → ∞, the empirical response simplex approximates the full policy space, and Eq. 7 recovers the KLregularized RL objective maxw Ew[R] − τDKL(w∥πt) (Ziebart, 2010; Levine, 2018), whose solution is w∗ ∝ πt(y) exp(R(y)/τ) with an intractable partition function. Operating on a finite response simplex yields a tractable formulation and makes the implicit target explicit.

Monotonic improvement guarantee. The proximal objective Jˆ(w) serves as a surrogate to the listwise reward Rˆ(w) = ∑k wkRk, establishing a performance improvement bound:

- Theorem 2 (Performance improvement bound). Assume |Rk| ≤ Rmax. If the projection step achieves TV(Pt+1, w∗) ≤ ϵproj, then

Rˆ(Pt+1) ≥ Rˆ(Pt) + τ DKL(w∗∥Pt) + DKL(Pt∥w∗)

− 2Rmaxϵproj projection error

. (9)

target gain ≥ 0

The target gain in Theorem 2 is the Jeffreys divergence (Jeffreys, 1946). With perfect projection, i.e., ϵproj = 0, the reward strictly improves whenever Pt ̸= w∗. In the idealized full policy space, iterating the exact proximal update converges to the reward-maximizing policy, providing a limiting reference for the target-projection framework. See Appendix B.5 and Appendix B.6 for proofs.

Proposition 2 (Idealized full-space convergence). Let π0(y) > 0 for all y, and assume R(y) is bounded. Under exact proximal updates πt+1(y) ∝ πt(y) exp(R(y)/τ), the iteration satisfies πt(y) ∝ π0(y) exp(tR(y)/τ) and Eπt[R] → maxy R(y) as t → ∞.

###### 4.2 Projection for Policy Optimization

With both the target w∗ in Eq. 8 and the listwise distribution Pθ in Eq. 4 on ∆K−1, policy optimization reduces to a projection under a chosen divergence. As representative choices, we develop the forward and reverse KL versions, with full derivations in Appendix B.1.

- Example 1 (Forward KL). Minimizing the forward KL divergence DKL(w∗∥Pθ) gives:

K

### ∑

min LLPOfwd = DKL(w∗∥Pθ) ⇒ ∇θ LLPOfwd =

Pθ,k − w∗

∇θ log πθ(yk|x). (10)

k

k=1

cfwdk

Rewards

Target

###### Listwise Policy Optimization (LPO)

Responses

Verifier

Target Construction Prompt

| | |
|---|---|
| | |

...

...

Behavior Policy

Listwise Distribution

...

Policy Model

Projection

###### ... ...

Update

Rewards

Target

###### Listwise Policy Optimization (LPO)

Responses

Verifier

Target Construction Prompt

| | |
|---|---|
| | |

...

...

Behavior Policy

Listwise Distribution

Projection

...

Policy Model

...

...

Update

- Figure 2: Illustration of LPO, which performs explicit target projection on the LLM response simplex, in contrast to the implicit approximations of group-based policy gradient methods.

Algorithm 1: Listwise Policy Optimization (LPO) Require: Policy parameters θ, temperature τ > 0, batch size B, inner epochs E, step size η

- 1: for each training iteration do
- 2: Set behavior policy πb ← πθ, pre-update policy πt ← πθ
- 3: Sample a batch of prompts B = {xi}iB=1
- 4: For each x ∈ B, generate responses {yk}kK=1 ∼ πb(·|x) and compute rewards {Rk}kK=1
- 5: Compute Target: w∗(x) = softmax(ϕ(x)) via Eq. 8 for all x ∈ B
- 6: for epoch e = 1 to E do
- 7: Compute Coefficients: ck(x) via Eq. 10 (forward KL) or Eq. 11 (reverse KL)
- 8: Gradient Update: θ ← θ − η B1 ∑x∈B ∑kK=1 ck(x)∇θ log πθ(yk|x)

- 9: end for
- 10: end for

The coefficient cfwdk measures the probability gap between the current policy and the target. Although similar projection objectives exist in prior methods (Abdolmaleki et al., 2018; Peng et al., 2019; Shu et al., 2026), they

are implemented in a pointwise manner, treating each response independently without relative comparison. In contrast, LPO performs projection on the response simplex via shared normalization, which couples across responses. Furthermore, this yields the following desirable properties:

- Corollary 1 (Gradient coefficient properties). The forward KL gradient coefficients cfwdk satisfy: (a) bounded: |cfwdk | ≤ 1; (b) zero-sum: ∑k cfwdk = 0; (c) self-correcting: cfwdk → 0 as Pθ → w∗.
- Corollary 2 (Mode-Coverage). If wk∗ ≥ α and DKL(w∗∥Pθ) ≤ D, then Pθ,k > α exp (−D/α − 1).

The zero-sum property acts as a built-in control variate for variance reduction (Sutton, 1988). The bounded and self-correcting properties further improve optimization stability. Moreover, Corollary 2 provides a log-barrier against mode collapse, ensuring response diversity. Recently, Kaddour (2026) adopts a very similar listwise forward KL projection, empirically corroborating its practical efficacy.

- Example 2 (Reverse KL). Minimizing the reverse KL divergence DKL(Pθ∥w∗), with logit gap dk = sθ,k − ϕk (the

difference between the current policy and the target) and its Pθ-weighted mean d¯ = ∑j Pθ,j dj, yields the following gradient:

K

Pθ,k dk − d¯ crevk

### ∑

min LLPOrev = DKL(Pθ∥w∗) ⇒ ∇θ LLPOrev =

∇θ log πθ(yk|x). (11)

k=1

Similar to the forward KL, the gradient coefficient crevk is zero-sum and self-correcting. Minimizing reverse KL is equivalent to maximizing the proximal objective Jˆ (See Proposition 3), and it decomposes as H(Pθ) + ∑k Pθ,k ϕk: revealing an implicit entropy bonus (Appendix C.7). At the on-policy point, the gradient of this objective exactly recovers the standard policy gradient (Proposition 1).

###### 4.3 Practical Implementation

The overall LPO procedure is summarized in Algorithm 1. The training pipeline is identical to standard group-based RL algorithms, with no additional computational cost.

Temperature as an adaptive baseline. While the temperature τ could theoretically be treated as a trust-region hyperparameter, we intentionally avoid introducing new tuning burdens. Instead, we adapt τ using the group-relative

advantage normalization statistics of existing methods, e.g., τ = σG for GRPO or τ = µG for MaxRL. This allows us to isolate gains from exact listwise projection while preserving the target temperature used by prior methods.

#### 5 Main Empirical Results

###### 5.1 Experimental Setup

We evaluate LPO across four representative domains of reasoning: logic, mathematics, programming, and multi-modal geometry. To assess generality, we benchmark across a diverse set of LLM backbones spanning different model sizes (1.5B–14B) and various LLM families. We track the training performance by plotting the curves for expected Pass@1 (average accuracy over rollouts) and Pass@k (Chen et al., 2021), with the specific k configurations detailed per benchmark.

Logical Reasoning. We adopt the Countdown Game, which requires composing given numbers using basic operations to match a target value. We train on a subset of Countdown-34 dataset (Pan et al., 2025) and evaluate on both Countdown-34 and the harder Countdown-4. We primarily use Qwen3-4B-Base (Yang et al., 2025a) and further evaluate models from other families in Sec. 5.4.3.

Mathematical Reasoning. We train on the MATH dataset (Hendrycks et al., 2021) using Qwen3-1.7B-Base and Qwen3-8B-Base (Yang et al., 2025a). Evaluation is conducted on standard benchmarks following Qu et al. (2025); Gao et al. (2025): AIME24, AIME25, AMC23, MATH500 (Lightman et al., 2023), Minerva Math (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024). In Appendix E.1, we scale to Qwen3-14B-Base on the larger Polaris dataset (An et al., 2025).

Programming. We train and evaluate Qwen3-1.7B-Base on the respective training and test splits of the PRIME code dataset (Cui et al., 2025).

Multi-Modal Geometry. Geometry problems require multi-modal understanding and reasoning. We train Qwen2.5VL-3B-Instruct (Bai et al., 2025) on the training split of the Geometry3k dataset (Lu et al., 2021; Hiyouga, 2025) and evaluate it on the test split.

Baselines and LPO Variants. We compare against three representative group-based policy gradient (PG) methods with varied target temperature designs: GRPO (τ = σG), Dr.GRPO (τ = 1), and MaxRL (τ = µG). To ensure a rigorous apples-to-apples comparison, we isolate the effect of the gradient formulation from temperature scaling by implementing LPO variants for each baseline. Specifically, we develop forward (LPOfwd) and reverse KL (LPOrev) versions that use the exact same temperature τ as their corresponding PG counterpart. The paired evaluation ensures that any performance differences are attributable to explicit listwise projection rather than temperature tuning. We implement baselines and LPO with the verl framework (Sheng et al., 2024).

Additional implementation details are provided in Appendix D, together with extended experimental results in Appendix E and prompt examples in Appendix F.

###### 5.2 Training Performance

Performance gains. Under paired temperature configurations, LPO consistently outperforms group-based PG baselines. For Pass@1 accuracy in Fig. 3, both LPO variants demonstrate efficient and improved training performance, exceeding their corresponding PG baselines in nearly all settings (13/15 for LPOfwd and 13/15 for LPOrev). This advantage also extends to Pass@k evaluations in Fig. 4, where both LPO variants continue to surpass the implicit PG methods (15/15 for LPOfwd and 11/15 for LPOrev). Together, these consistent gains suggest that replacing first-order advantage approximations with exact listwise projection on the response simplex offers a promising paradigm for improving the training efficiency and performance of RLVR.

Projection divergence effects. Comparing the two variants reveals an empirical distinction: LPOfwd outperforms LPOrev in 13/15 scenarios for Pass@k. This observation aligns well with the expectation: the mode-coverage property inherent to forward-KL actively preserves reasoning diversity for a broader distribution of valid solution paths. More broadly, this highlights the flexibility of the decoupled target-projection framework, suggesting that exploring alternative projection divergences could unlock further unique optimization properties.

Robustness across temperature parameterizations. We observe that the optimal implicit temperature strategy τ is highly task-dependent, with no single design consistently dominating across all benchmarks. Despite this task-varying behavior, LPO delivers stable performance gains under all tested τ designs. This indicates that exact listwise projection provides a robust optimization mechanism, yielding benefits that are largely orthogonal to the underlying temperature heuristic.

PG (GRPO, Dr.GRPO, MaxRL) LPOfwd LPOrev

###### Countdown 4B (GRPO)

###### MATH 1.7B (GRPO)

###### MATH 8B (GRPO)

###### PRIME Code 1.7B (GRPO)

###### Geometry 3B (GRPO)

- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36

42.5

45.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
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

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

50

42.5

40.0

60

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

40.0

37.5

48

55

37.5

35.0

35.0

46

32.5

50

32.5

30.0

30.0

44

45

27.5

27.5

40

42

25.0

25.0

40 60 80 100 120

200 400 600 800

100 200 300 400 500

200 250 300 350 400 450

50 100 150 200

Step

Step

Step

Step

Step

###### Countdown 4B (Dr.GRPO)

###### MATH 1.7B (Dr.GRPO)

###### MATH 8B (Dr.GRPO)

###### PRIME Code 1.7B (Dr.GRPO)

###### Geometry 3B (Dr.GRPO)

42.5

45.0

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
| | | | | |

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
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

50

- 28
- 29
- 30
- 31
- 32
- 33

42.5

40.0

60

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

40.0

37.5

48

55

37.5

35.0

35.0

46

32.5

50

32.5

30.0

30.0

44

45

27.5

27.5

40

42

25.0

25.0

40 60 80 100 120

100 200 300 400 500

100 200 300 400 500

200 250 300 350 400 450

50 100 150 200

Step

Step

Step

Step

Step

###### Countdown 4B (MaxRL)

###### MATH 1.7B (MaxRL)

###### MATH 8B (MaxRL)

###### PRIME Code 1.7B (MaxRL)

Geometry 3B (MaxRL)

- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36

42.5

45.0

| | | | | |
|---|---|---|---|---|
| | | | | |
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

| | | | | | |
|---|---|---|---|---|---|
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
| | | | | |

50

42.5

40.0

60

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

40.0

37.5

48

55

37.5

35.0

35.0

46

32.5

50

32.5

30.0

30.0

44

45

27.5

27.5

40

42

25.0

25.0

40 60 80 100 120

100 200 300 400 500

100 200 300 400 500

200 250 300 350 400 450

50 100 150 200

Step

Step

Step

Step

Step

- Figure 3: Training curves of Pass@1 accuracy. Two LPO variants (LPOfwd, LPOrev) are evaluated against groupbased PG baselines (GRPO, Dr.GRPO, MaxRL, shown from top to bottom) across various LLM backbones and reasoning tasks with corresponding temperature designs.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

40 60 80 100 120

Step

65

70

75

80

85

Pass@kAccuracy(%)

Countdown 4B (GRPO)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

200 400 600 800

Step

- 40
- 41
- 42
- 43
- 44
- 45
- 46
- 47

Pass@kAccuracy(%)

MATH 1.7B (GRPO)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100 200 300 400 500

Step

52

54

56

58

60

Pass@kAccuracy(%)

MATH 8B (GRPO)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

200 250 300 350 400 450

Step

40

42

44

46

48

50

52

Pass@kAccuracy(%)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

PRIME Code 1.7B (GRPO)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

50 100 150 200

Step

58

60

62

64

66

68

Pass@kAccuracy(%)

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
| | | | | |

Geometry 3B (GRPO)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

40 60 80 100 120

Step

65

70

75

80

85

Pass@kAccuracy(%)

Countdown 4B (Dr.GRPO)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100 200 300 400 500

Step

- 39
- 40
- 41
- 42
- 43
- 44
- 45

Pass@kAccuracy(%)

MATH 1.7B (Dr.GRPO)

100 200 300 400 500

Step

52

54

56

58

60

Pass@kAccuracy(%)

MATH 8B (Dr.GRPO)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

200 250 300 350 400 450

Step

36

38

40

42

44

46

48

Pass@kAccuracy(%)

PRIME Code 1.7B (Dr.GRPO)

50 100 150 200

Step

58

60

62

64

66

68

Pass@kAccuracy(%)

Geometry 3B (Dr.GRPO)

40 60 80 100 120

Step

65

70

75

80

85

Pass@kAccuracy(%)

Countdown 4B (MaxRL)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100 200 300 400 500

Step

- 40
- 41
- 42
- 43
- 44
- 45
- 46
- 47

Pass@kAccuracy(%)

MATH 1.7B (MaxRL)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100 200 300 400 500

Step

54

56

58

60

62

Pass@kAccuracy(%)

MATH 8B (MaxRL)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

200 250 300 350 400 450

Step

40

42

44

46

48

Pass@kAccuracy(%)

PRIME Code 1.7B (MaxRL)

50 100 150 200

Step

58

60

62

64

66

68

70

72

Pass@kAccuracy(%)

Geometry 3B (MaxRL)

PG (GRPO, Dr.GRPO, MaxRL) LPOfwd LPOrev

- Figure 4: Pass@k training curves. LPO variants (LPOfwd, LPOrev) are evaluated against group-based PG baselines (GRPO, Dr.GRPO, MaxRL, shown from top to bottom) across various LLMs and tasks under paired temperature settings. Specific k configurations are detailed per benchmark.

PG (GRPO) LPOfwd LPOrev

###### Countdown 4B (GRPO)

###### MATH 1.7B (GRPO)

###### MATH 8B (GRPO)

###### PRIME Code 1.7B (GRPO)

###### Geometry 3B (GRPO)

0.30

0.10

0.06

0.200

0.40

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
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.35

0.175

0.25

0.05

0.08

ResponseEntropy

ResponseEntropy

ResponseEntropy

ResponseEntropy

ResponseEntropy

0.30

0.150

0.20

0.04

0.25

0.125

0.06

0.15

0.20

0.100

0.03

0.10

0.04

0.15

0.075

0.02

0.05

0.10

0.050

0.02

0.05

0.00

0.01

0.025

20 40 60 80 100 120

200 400 600 800

100 200 300 400 500

100 200 300 400

50 100 150 200

Step

Step

Step

Step

Step

0.200

0.40

- 0.40

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
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
| | | | | |
| | | | | |

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
| | | | | |

0.22

- 0 50 100 150 200 Step
- 1
- 2
- 3
- 4

0.175

0.20

0.35

0.35

0.150

0.18

GradNorm

GradNorm

GradNorm

GradNorm

GradNorm

0.30

0.30

0.125

0.16

0.25

0.100

0.14

0.25

0.20

0.075

0.12

0.20

0.050

0.15

0.10

0.025

0.08

0 25 50 75 100 125

0 200 400 600 800

0 100 200 300 400 500

0 100 200 300 400

Step

Step

Step

Step

1200

300

1300

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
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

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

750

275

1200

600

1100

700

ResponseLen

ResponseLen

ResponseLen

ResponseLen

ResponseLen

250

1100

1000

650

500

225

1000

900

600

200

900

800

400

175

800

550

700

150

700

500

300

125

600

600

0 25 50 75 100 125

0 200 400 600 800

0 100 200 300 400 500

0 100 200 300 400

0 50 100 150 200

Step

Step

Step

Step

Step

- Figure 5: Training dynamics of LPO variants and GRPO. Rows from top to bottom respectively show the curves of response entropy, gradient norms, and response lengths.

- 5.3 Training Dynamics

To better understand the underlying optimization behaviors and validate our theoretical analysis, we track key training metrics: response entropy, gradient norm, and response length.

Response entropy and exploration preservation. As shown in Fig. 5 (top), both LPO variants generally maintain higher response entropy than PG baselines. This corresponds to the projection properties: LPOrev corresponds to a maximum-entropy objective, while LPOfwd exhibits mode-covering behavior. This sustained diversity directly explains the robust Pass@k improvements, positioning listwise projection as a principled remedy for the entropy collapse in RLVR.

Gradient norms and optimization stability. Fig. 5 (middle) reveals that LPO variants exhibit lower and more stable gradient norms compared to group-based PG methods. This empirical stability is consistent with Corollary 1: LPO’s exact projection on the response simplex yields controlled gradient coefficients, leading to stable optimization dynamics.

Response length and reasoning behaviors. Fig. 5 (bottom) shows that LPO tends to generate longer responses than PG. As increased length often correlates with more detailed reasoning chains (Yu et al., 2025), this is consistent with LPO encouraging more extensive exploration. LPOfwd’s maximum length aligns with its mode-covering property, which promotes diverse reasoning paths.

- 5.4 Additional Analysis

- 5.4.1 Listwise vs. Pointwise Projection

To highlight the contribution of the listwise projection, we ablate the listwise policy distribution in Eq. 4 while keeping the target in Eq. 8 unchanged. This recovers the pointwise projection with forward KL (Peters et al., 2010;

Abdolmaleki et al., 2018; Peng et al., 2019), defined as Lpoint = − ∑k wk∗ log πθ(yk|x). As shown in Fig. 6, this pointwise variant suffers a severe performance drop. This failure stems from the lack of a coupled competitive

mechanism across responses in pointwise updates, resulting in unstable optimization. Conversely, both group-based PG and LPO intrinsically employ a built-in control variate that stabilizes training. These results suggest that our performance gains stem not merely from the target design, but from successfully marrying exact target fitting with the crucial structural variance reduction provided by the listwise projection. Detailed properties of the two projections are deferred to Appendix C.4.

- 5.4.2 Effect of Group Size K

We investigate the impact of the sampled group size K on Countdown. As shown in Fig. 7, across the tested group sizes (K ∈ {2,4,8,16,32}), both LPO variants remain highly competitive with GRPO, with advantages being

PG (GRPO) LPOfwd Pointwise

PG (GRPO) LPOfwd LPOrev

###### Math 1.7B - Pass@1

Math 1.7B - Grad Norm

###### Pass@1

Pass@64

36

0.6

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
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
| | | | | | |

90

65.0

0.5

62.5

34

85

Pass@64Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

60.0

0.4

GradNorm

32

80

57.5

0.3

55.0

30

75

0.2

52.5

28

70

0.1

50.0

47.5

26

0.0

65

100 200 300 400 500 600 700 800

0 100 200 300 400 500 600 700 800

2 4 8 16 32

2 4 8 16 32

Step

Step

Group Size

Group Size

Figure 7: Effect of varying group sizes K ∈ {2,4,8,16,32} on Countdown.

- Figure 6: Ablation comparing listwise LPO with pointwise projection and GRPO baselines on MATH (Qwen3-

- 1.7B-Base).

particularly pronounced at smaller group sizes. This suggests that explicit listwise projection improves sample efficiency, which stabilizes updates under limited samples. Furthermore, LPO variants exhibit distinct scaling behaviors that validate their theoretical properties: LPOrev achieves stronger Pass@1 performance, while LPOfwd scales exceptionally well on Pass@64, supporting its mode-coverage property which structurally preserves reasoning diversity.

###### 5.4.3 Generalization across LLM Families

To evaluate the generalizability of LPO, we conduct experiments across four prominent LLM families: Qwen, DeepSeek, Mistral, and Llama. As illustrated in Fig. 11 in Appendix E.3, LPO consistently delivers performance gains on the Countdown task, regardless of the underlying model architecture or training paradigm. This consistent improvement across diverse backbones suggests that LPO is not sensitive to a specific model architecture, but rather benefits from the fundamental robustness of the listwise projection framework. These results indicate that LPO can serve as a model-agnostic approach for improving reasoning performance in RLVR.

#### 6 Conclusion

This work introduces a unified geometric framework for deep insight into group-based RLVR of LLMs. We show that existing policy gradient methods act as approximate target-projections on the response simplex and present LPO to perform this projection explicitly. LPO benefits from directly optimizing on the simplex, which improves optimization stability and yields monotonic performance improvements. Moreover, the decoupled target-projection perspective opens up a flexible design space for developing rich and diverse optimization methods for RLVR of LLMs.

Limitations and future work. Our current formulation primarily focuses on sequence-level projection within outcome reward settings. Future research will explore step-level listwise projections and investigate broader divergences to fully unlock the potential of the decoupled framework.

#### References

Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a posteriori policy optimisation. arXiv preprint arXiv:1806.06920, 2018.

Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust¨¨ un, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12248–12267, 2024.

Shun-Ichi Amari. Natural gradient works efficiently in learning. Neural computation, 10(2):251–276, 1998. Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu,

Mingxuan Wang, and Lingpeng Kong. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025. URL https://hkunlp.github.io/blog/2025/Polaris.

GX-Chen Anthony, Jatin Prakash, Rob Fergus, and Rajesh Ranganath. Reverse-kl reinforcement learning can sample from multiple diverse modes. In First Workshop on Foundations of Reasoning in Language Models.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. Learning to rank: from pairwise approach to listwise approach. In Proceedings of the 24th international conference on Machine learning, pp. 129–136, 2007.

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.

Peter Dayan and Geoffrey E Hinton. Using expectation-maximization for reinforcement learning. Neural Computation, 9(2):271–278, 1997.

Zhaolin Gao, Joongwon Kim, Wen Sun, Thorsten Joachims, Sid Wang, Richard Yuanzhe Pang, and Liang Tan. Prompt curriculum learning for efficient llm post-training. arXiv preprint arXiv:2510.01135, 2025.

Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In International conference on machine learning, pp. 2160–2169. PMLR, 2019.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. Pmlr, 2018.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Hiyouga. Geometry3K: A large-scale multi-modal geometry reasoning dataset. https://huggingface.co/ datasets/hiyouga/geometry3k, 2025.

Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv e-prints, pp. arXiv–2501, 2025.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Harold Jeffreys. An invariant form for the prior probability in estimation problems. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 186(1007):453–461, 1946.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Jean Kaddour. Target policy optimization, 2026. URL https://arxiv.org/abs/2604.06159. Sham M Kakade. A natural policy gradient. Advances in neural information processing systems, 14, 2001. Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014.

Solomon Kullback. Kullback-leibler divergence. Encyclopedia of Machine Learning, pp. 581–583, 1951. Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint

arXiv:1805.00909, 2018.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.

Long Li, Zhijian Zhou, Jiaran Hao, Jason Klein Liu, Yanting Miao, Wei Pang, Xiaoyu Tan, Wei Chu, Zhe Wang, Shirui Pan, et al. The choice of divergence: A neglected key to mitigating diversity collapse in reinforcement learning with verifiable reward. arXiv preprint arXiv:2509.07430, 2025.

Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. arXiv preprint arXiv:2310.10505, 2023.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Tianqi Liu, Zhen Qin, Junru Wu, Jiaming Shen, Misha Khalman, Rishabh Joshi, Yao Zhao, Mohammad Saleh, Simon Baumgartner, Jialu Liu, Peter J. Liu, and Xuanhui Wang. Lipo: Listwise preference optimization through learning-to-rank, 2025a. URL https://arxiv.org/abs/2402.01878.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021), 2021.

R Duncan Luce et al. Individual choice behavior, volume 4. Wiley New York, 1959. Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin

Cai, Maurice Weber, et al. Deepcoder: A fully open-source 14b coder at o3-mini level. Notion Blog, 2025. Youssef Mroueh. Reinforcement learning with verifiable rewards: Grpo’s effective loss, dynamics, and success amplification. arXiv preprint arXiv:2503.06639, 2025. Radford M Neal and Geoffrey E Hinton. A view of the em algorithm that justifies incremental, sparse, and other variants. In Learning in graphical models, pp. 355–368. Springer, 1998.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Jiayi Pan, Junjie Zhang, Xingyao Wang, Lifan Yuan, Hao Peng, and Alane Suhr. Tinyzero. https://github.com/JiayiPan/TinyZero, 2025. Accessed: 2025-01-24.

Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

Jan Peters, Katharina Mulling, and Yasemin Altun. Relative entropy policy search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 24, pp. 1607–1612, 2010.

Robin L Plackett. The analysis of permutations. Journal of the Royal Statistical Society Series C: Applied Statistics, 24(2):193–202, 1975.

Yun Qu, Qi Wang, Yixiu Mao, Vincent Tao Hu, Bj¨orn Ommer, and Xiangyang Ji. Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models? arXiv preprint arXiv:2507.04632, 2025.

Yun Qu, Qi Wang, Yixiu Mao, Heming Zou, Yuhang Jiang, Weijie Liu, Clive Bai, Kai Yang, Yangkun Chen, Saiyong Yang, et al. Small generalizable prompt predictive models can steer efficient rl post-training of large reasoning models. arXiv preprint arXiv:2602.01970, 2026.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https://arxiv.org/ abs/2305.18290.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel. Trust region policy optimization, 2017a. URL https://arxiv.org/abs/1502.05477.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017b.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Yao Shu, Chenxing Wei, Hongbin Lin, Shuang Qiu, and Hui Xiong. Reference-sampled boltzmann projection for kl-regularized rlvr: Target-matched weighted sft, finite one-shot gaps, and policy mirror descent. arXiv preprint arXiv:2605.02469, 2026.

H Francis Song, Abbas Abdolmaleki, Jost Tobias Springenberg, Aidan Clark, Hubert Soyer, Jack W Rae, Seb Noury, Arun Ahuja, Siqi Liu, Dhruva Tirumala, et al. V-mpo: On-policy maximum a posteriori policy optimization for discrete and continuous control. arXiv preprint arXiv:1909.12238, 2019.

Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3(1):9–44, 1988. Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement

learning with function approximation. Advances in neural information processing systems, 12, 1999.

Fahim Tajwar, Guanning Zeng, Yueer Zhou, Yuda Song, Daman Arora, Yiding Jiang, Jeff Schneider, Ruslan Salakhutdinov, Haiwen Feng, and Andrea Zanette. Maximum likelihood reinforcement learning. arXiv preprint arXiv:2602.02710, 2026.

Manan Tomar, Lior Shani, Yonathan Efroni, and Mohammad Ghavamzadeh. Mirror descent policy optimization. arXiv preprint arXiv:2005.09814, 2020.

Milan Vojnovic and Se-Young Yun. What is the alignment objective of grpo? arXiv preprint arXiv:2502.18548, 2025.

Chaoqi Wang, Yibo Jiang, Chenghao Yang, Han Liu, and Yuxin Chen. Beyond reverse kl: Generalizing direct preference optimization with diverse divergence constraints. arXiv preprint arXiv:2309.16240, 2023.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Wei Yang, Jiacheng Pang, Shixuan Li, Paul Bogdan, Stephen Tu, and Jesse Thomason. Maestro: Learning to collaborate via conditional listwise policy optimization for multi-agent llms. arXiv preprint arXiv:2511.06134, 2025b.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025. URL https: //arxiv.org/abs/2507.18071.

Xuekai Zhu, Daixuan Cheng, Dinghuai Zhang, Hengli Li, Kaiyan Zhang, Che Jiang, Youbang Sun, Ermo Hua, Yuxin Zuo, Xingtai Lv, et al. Flowrl: Matching reward distributions for llm reasoning. arXiv preprint arXiv:2509.15207, 2025.

Brian D Ziebart. Modeling purposeful adaptive behavior with the principle of maximum causal entropy. Carnegie Mellon University, 2010.

#### Appendix Overview

This appendix provides supplementary proofs, conceptual discussions, and experimental details supporting the main text. It is organized as follows:

- • Appendix A (Related Works): reviews literature on RLVR, RL as probabilistic inference, and listwise formulations.
- • Appendix B (Proofs): provides detailed mathematical derivations for all theoretical claims.
- • Appendix C (Additional Discussions): expands on the framework’s conceptual and practical scope. It unifies existing group-based RLVR algorithms, compares listwise projection with pointwise and preference optimization, explores future extensions.
- • Appendix D (Implementation Details): outlines the experimental setup, including tasks, LLM backbones, and training details.
- • Appendix E (Extended Experimental Results): reports further empirical findings, including scalability validation, on-policy optimization, extended training dynamics, and generalization across diverse LLM families.
- • Appendix F (Data Examples): presents representative data examples used across the evaluated reasoning tasks.

#### A Related Works

Reinforcement learning with verifiable rewards. The alignment and reasoning capabilities of LLMs have been significantly advanced by RL (Ouyang et al., 2022; Bai et al., 2022), initially dominated by PPO (Schulman et al., 2017b) with a learned value model. The emergence of RLVR for reasoning tasks (Jaech et al., 2024; Guo et al., 2025) has driven a paradigm shift toward critic-free, group-based policy gradient methods (Shao et al., 2024; Ahmadian et al., 2024; Li et al., 2023), which sample multiple responses per prompt and derive advantages entirely from within-group reward statistics. A subsequent line of work has refined this paradigm, introducing novel advantage normalization (Liu et al., 2025b; Hu, 2025; Tajwar et al., 2026), trust-region mechanics (Yu et al., 2025; Chen et al., 2025), and sequence-level scaling (Zheng et al., 2025). Recent theoretical works have sought to uncover the underlying mechanics of these methods (Mroueh, 2025; Vojnovic & Yun, 2025). Our LPO framework provides a unifying perspective by revealing that major group-based methods share the same target-projection geometry. Concurrently, FlowRL (Zhu et al., 2025) minimizes reverse KL against a Gibbs target approximated by a learned partition function network, while contemporaneous TPO (Kaddour, 2026) similarly adopts cross-entropy on tilted simplex targets. Differently, LPO contributes a unifying analytical Target-Projection framework that recovers existing group-based methods, and admits multiple divergences with provably satisfying properties.

RL as probabilistic inference. The idea of constructing a reward-weighted target distribution and projecting the policy toward it has deep roots in the RL-as-inference literature, which casts control as inference under a KL-regularized objective (Dayan & Hinton, 1997; Ziebart, 2010; Levine, 2018; Geist et al., 2019). This perspective gives rise to a natural trust-region structure (Amari, 1998; Kakade, 2001; Schulman et al., 2017a) and underlies a wide range of practical algorithms (Peters et al., 2010; Abdolmaleki et al., 2018; Song et al., 2019; Peng et al., 2019; Haarnoja et al., 2018; Tomar et al., 2020). However, these methods typically operate in continuous action

spaces and resort to pointwise projections, i.e., − ∑k wk∗ log πθ(yk). In contrast, sampled responses in LLM form a finite response simplex, where normalization is exact and the partition function reduces to a finite sum over samples.

This structure enables listwise projection on the simplex, as exploited by LPO, which couples all responses through shared normalization and inherits satisfying gradients (Appendix C.4).

Listwise formulation. Listwise formulation has a long history in classical choice and learning-to-rank models (Luce et al., 1959; Plackett, 1975; Cao et al., 2007), where a distribution over candidate sets or permutations is modeled or optimized. Recent LLM alignment methods, such as DPO (Rafailov et al., 2024) and its extensions (Liu et al., 2025a), adopt pairwise or listwise preference structures to model relative comparisons among responses. Listwise structures have also been employed in multi-agent LLM collaboration (Yang et al., 2025b). Our approach operates in the standard RLVR setting for LLM post-training and explicitly constructs a target distribution based on verifiable rewards on the response simplex, followed by direct projection onto it.

#### B Proofs

###### B.1 KL Gradient Derivations

We derive the gradients of the forward and reverse KL divergences stated in Section 4.2. For both derivations, we recall from Eq. 4 that the logits are defined as sθ,k = log πθ(yk|x) − log πb(yk|x). Since the behavior policy πb is frozen, the gradient of the logit with respect to the parameters is simply ∇θsθ,k = ∇θ log πθ(yk|x).

Forward KL: DKL(w∗∥Pθ). By definition, DKL(w∗∥Pθ) = − ∑kK=1 wk∗ log Pθ,k − H(w∗), where H(w∗) is the entropy of w∗, which is constant with respect to θ. Using the fact that log Pθ,k = sθ,k − log ∑Kj=1 esθ,j, the Jacobian of the log-softmax is given by

K

### ∑

Pθ,j ∇θsθ,j. (12) Substituting this into the gradient of the forward KL divergence, we obtain:

∇θ log Pθ,k = ∇θsθ,k −

j=1

K

### ∑

∇θDKL(w∗∥Pθ) = −

k=1

K

### ∑

= −

k=1

w∗

k ∇θsθ,k −

K

w∗

k ∇θsθ,k +

### ∑

Pθ,j ∇θsθ,j

j=1

K

K

### ∑

### ∑

w∗

Pθ,j ∇θsθ,j.

k

j=1

k=1

Since w∗ is a valid probability distribution (∑kK=1 wk∗ = 1), the second term simplifies. Reindexing the summation and substituting ∇θsθ,k = ∇θ log πθ(yk|x), we get:

K

### ∑

∇θDKL(w∗∥Pθ) =

(Pθ,k − w∗

k) ∇θ log πθ(yk|x). (13)

k=1

Reverse KL: DKL(Pθ∥w∗). Write the reverse KL as DKL(Pθ∥w∗) = ∑kK=1 Pθ,k log(Pθ,k/wk∗). We first compute the partial derivative with respect to a single logit sθ,j. Using the standard softmax Jacobian ∂Pθ,k/∂sθ,j =

Pθ,k(δkj − Pθ,j) and applying the product rule, we have:

K

∂ ∂sθ,j

### ∑

DKL(Pθ∥w∗) =

Pθ,k(δkj − Pθ,j) log(Pθ,k/w∗

##### k) + 1

k=1

K

### ∑

Pθ,k log(Pθ,k/w∗

= Pθ,j log(Pθ,j/w∗j ) + 1 − Pθ,j

##### k) + 1

k=1

Pθ,j w∗j

+ Pθ,j − Pθ,jDKL(Pθ∥w∗) − Pθ,j(1)

= Pθ,j log

Pθ,j w∗j − DKL(Pθ∥w∗) .

= Pθ,j log

Applying the multivariate chain rule ∇θDKL = ∑Kj=1 ∂∂DsθKL,j ∇θsθ,j, we arrive at the full gradient:

K

Pθ,j w∗j − DKL(Pθ∥w∗) ∇θ log πθ(yj|x).

### ∑

∇θ DKL(Pθ∥w∗) =

Pθ,j log

| |
|---|

j=1

(14)

Logit-gap simplification for reverse KL. The per-logit coefficient for the reverse KL gradient, crevk = ∂s∂

DKL(Pθ∥w∗),

θ,k

can be elegantly simplified when written in terms of the logit gap dk = sθ,k − ϕk, where ϕk is the target logit from Eq. 8.

Express both probabilities explicitly with their partition functions: Pθ,k = exp(sθ,k)/Zs and wk∗ = exp(ϕk)/Zϕ. The log-ratio becomes:

Pθ,k wk∗

log

= (sθ,k − log Zs) − (ϕk − log Zϕ) = (sθ,k − ϕk) − (log Zs − log Zϕ) = dk − cs, (15)

where cs = log Zs − log Zϕ is strictly constant across all k. Consequently, the KL divergence can be written in terms of the expected gap:

K

Pθ,k(dk − cs) = d¯− cs, (16)

### ∑

DKL(Pθ∥w∗) =

k=1

where d¯ = ∑kK=1 Pθ,kdk. Substituting these back into the coefficient crevk , the constant cs perfectly cancels out:

crevk = Pθ,k (dk − cs) − (d¯− cs) = Pθ,k(dk − d¯). (17) This reveals that the reverse KL gradient admits a baseline-subtracted form, where the baseline corresponds to the expected logit gap under the current policy Pθ.

###### B.2 Proof of Proposition 1

- Proposition 1 (Group-based policy gradient as reverse KL at on-policy). Let A ∈ RK be a zero-mean advantage

vector, i.e., ∑kK=1 Ak = 0, and let w∗ = softmax(A). At the on-policy point (πθ = πb), the policy gradient in Eq. 3 equals the negative gradient of the reverse KL divergence DKL:

K

1 K

### ∑

Ak ∇θ log πθ(yk|x) = −∇θ DKL(Pθ∥w∗)

. (5)

gPG =

πθ=πb

k=1

Proof. By the logit-gap simplification derived in Eq. 17, the reverse KL gradient is fully characterized by the per-logit coefficients:

crevk = Pθ,k(dk − d¯), (18)

where dk = sθ,k − Ak and d¯ = ∑kK=1 Pθ,kdk. We directly evaluate this coefficient at the on-policy point πθ = πb. At the on-policy point, the logit offsets vanish (sθ,k = 0 for all k), which yields a uniform probability distribution over the generated list: Pθ,k = softmax(0)k = 1/K. Consequently, the logit gap simplifies to dk = −Ak.

Applying the zero-mean advantage assumption ∑kK=1 Ak = 0, the expected gap identically vanishes: d¯ =

K

1 K

### ∑

(−Ak) = 0. (19) Substituting these on-policy values back into the coefficient yields:

k=1

crevk

=

πθ=πb

1 K

(−Ak − 0) = −

Ak K

. (20)

Recall that the standard policy gradient in Eq. 3 can be expressed as gPG = ∑k cPGk ∇θ log πθ(yk|x) with coefficients cPGk = Ak/K. Comparing the coefficients, we immediately obtain cPGk = −crevk |πθ=πb, which proves that

gPG = −∇θDKL(Pθ∥w∗) π

θ=πb. (21) confirming that the policy gradient step is a gradient descent step on the reverse KL divergence at the on-policy point.

The centering assumption ∑k Ak = 0 is without loss of generality: by the shift-invariance of softmax, replacing A with A − A¯ does not change the target w∗ = softmax(A).

Finally, we clarify that the zero-mean assumption, i.e., ∑k Ak = 0, is not a restrictive algorithmic requirement, but rather a natural reflection of the listwise projection’s intrinsic mechanics. Due to the shift-invariance of the target softmax, any prompt-level scalar baseline applied uniformly to the group’s rewards, e.g., the greedy baseline in ReMax (Li et al., 2023), is mathematically absorbed and nullified. The zero-sum constraint of the local simplex inherently induces a dynamically weighted mean-centering control variate (dk − d¯) during the gradient computation. At the on-policy point, this natively recovers the arithmetic zero-mean counterpart (Ak − A¯). Thus, assuming centered advantages simply aligns our notation with the framework’s built-in behavior at the exact point of equivalence.

| |
|---|

Off-policy approximation error. Proposition 1 establishes exact equality at πθ = πb. We now quantify the discrepancy off-policy, incorporating the importance sampling ratio rk = πθ(yk|x)/πb(yk|x) standard in practical PG methods without clipping.

Let sθ,k = logrk, Pθ = softmax(sθ), and δ¯ = maxk |rk − 1|. Both the policy gradient and the reverse KL gradient can be written as ∑k ck∇θ log πθ(yk|x) with respective coefficients

rkAk K

, crevKLk = −Pθ,k(dk − d¯), (22) where dk = sθ,k − Ak is the logit gap from Eq. 17 and d¯ = ∑k Pθ,kdk. The per-coefficient discrepancy is

cPGk =

rkAk K

+ Pθ,k(dk − d¯), (23)

∆k = cPGk − crevKLk =

which vanishes identically at on-policy point (πθ = πb). We analyze the local regime where δ¯ < 1/2, under which we have rk ∈ [1/2,3/2], and thus ∥sθ∥∞ ≤ 2δ¯.

A first-order Taylor expansion of Pθ,k = softmax(sθ)k and rk = exp(sθ,k) around the zero vector sθ = 0 gives

+ O ∥sθ∥2∞

sθ,k − s¯θ K

1 K

, rk = 1 + sθ,k + O(∥sθ∥2∞), (24)

Pθ,k =

+

K

where s¯θ = K1 ∑k sθ,k. Using the zero-mean advantage assumption ∑k Ak = 0, the first-order expansion of d¯ = ∑k Pθ,k(sθ,k − Ak) yields

1

d¯ = s¯θ −

sθ,mAm + O(δ¯2). (25)

### K ∑

m

Collecting terms:

1 K

1 K

∑m sθ,mAm + O δ ¯2(1+K∥A∥∞) . (26) Bounding the three terms via |sθ,k − s¯θ| ≤ 2∥sθ∥∞ ≤ 4δ¯, and symmetrically for the advantage terms |s¯θAk| ≤ 2δ¯∥A∥∞ and |K1 ∑m sθ,mAm| ≤ ∥sθ∥∞∥A∥∞ ≤ 2δ¯∥A∥∞:

(sθ,k − s¯θ) + s¯θAk +

∆k =

C δ¯(1 + ∥A∥∞) K

|∆k| ≤

(27)

for a universal constant C > 0, to leading order in δ¯. By the triangle inequality, the parameter-space gradient discrepancy satisfies

∥gPG − grevKL∥ ≤ ∑k |∆k| Gmax ≤ C′δ¯(1 + ∥A∥∞) Gmax, (28)

where Gmax = maxk ∥∇θ log πθ(yk|x)∥. The error is linear in the off-policy drift δ¯ and vanishes at the on-policy point, confirming that the policy gradient approximates reverse KL projection only in a neighborhood of the sampling distribution. This rapid degradation under off-policy drift directly motivates the exact listwise projection proposed in LPO.

Remark: Connection to group-based policy gradients. Our off-policy analysis explicitly uncovers the structural relationship between exact listwise projection and practical group-based PG methods. By performing a strict first-order Taylor expansion on the exact reverse KL projection coefficient, we decouple it into three components:

sθ,k − s¯θ K Intrinsic entropy regularization

sθ,kAk K

s ¯θAk K

1 K2

Ak K

crevKLk ≈

. (29)

∑m sθ,mAm

−

−

+

+

Pointwise advantage fitting

Listwise normalization

Remarkably, the gradient coefficient of group-based policy gradients without clipping, given by cPGk = rkKAk , yields a first-order expansion cPGk ≈ AKk + sθ,KkAk . This reveals a direct mathematical connection: the pointwise IS objective utilized in group-based policy gradients formally corresponds to the first-order advantage-fitting component of the reverse KL projection on the simplex, while the exact listwise formulation explicitly retains the coupled listwise normalization and intrinsic entropy regularization.

###### B.3 Proof of Theorem 1

- Theorem 1 (Listwise Gibbs target). The objective Jˆ(w) in Eq. (7) has a unique maximizer w∗:

Rk τ

w∗

k = softmax(ϕ)k, with ϕk =

+ st,k. (8)

Proof. Consider the optimization problem

K

Jˆ(w), Jˆ(w) =

### ∑

w∗ = arg max

w∈∆K−1

k=1

where Pt = softmax(st) satisfies Pt,k > 0 for all k. Expanding the KL term gives

K

K

Jˆ(w) =

### ∑

### ∑

wkRk − τ

k=1

k=1

Introduce the Lagrangian for the simplex constraint ∑k wk = 1:

K

K

### ∑

### ∑

wk log

L(w, λ) =

wkRk − τ

k=1

k=1

wkRk − τDKL(w∥Pt), (30)

wk log

wk Pt,k

. (31)

wk Pt,k

+ λ 1 −

K

### ∑

wk . (32)

k=1

Setting the stationary condition ∂L/∂wk = 0 yields

Rk − τ(log wk − log Pt,k + 1) − λ = 0, (33) hence

Rk τ

λ τ

. (34) Exponentiating,

log wk =

+ log Pt,k − 1 −

wk = Pt,k exp(Rk/τ) · C, C = exp(−1 − λ/τ) . (35) Using ∑k wk = 1, the normalization constant is

K

C−1 =

### ∑

Pt,j exp(Rj/τ), (36)

j=1

therefore

Pt,k exp(Rk/τ) ∑Kj=1 Pt,j exp(Rj/τ)

w∗

. (37)

k =

Equivalently,

w∗ = softmax R/τ + log Pt . (38) Since log Pt = st − log ∑j est,j, softmax is shift-invariant, this yields

Rk τ

w∗ = softmax(R/τ + st) = softmax(ϕ), ϕk =

+ st,k, (39)

which is Eq. 8. Finally, Jˆ(w) is strictly concave on ∆K−1: the reward term is linear in w, while DKL(w∥Pt) is strictly convex for Pt,k > 0. Hence the maximizer is unique.

| |
|---|

###### B.4 Proximal Objective as Reverse KL

Proposition 3 (Proximal objective as reverse KL). Jˆ(Pθ) = −τDKL(Pθ∥w∗) + τ log Zˆ, so argmaxPθ Jˆ(Pθ) = argminPθ DKL(Pθ∥w∗).

Proof. From Theorem 1, wk∗ = Pt,k exp(Rk/τ)/Zˆ where Zˆ = ∑j Pt,j exp(Rj/τ). Therefore log wk∗ = Rk/τ + log Pt,k − log Zˆ. Expanding the reverse KL:

K

Pθ,k wk∗

∑

−τDKL(Pθ∥w∗) = −τ

Pθ,k log

k=1

= −τ∑

Pθ,k log Pθ,k − log w∗

k

k

Pθ,k log Pθ,k − Rk/τ − log Pt,k + log Zˆ

= −τ∑

k

Pθ,k Pt,k − τ log Zˆ.

= ∑

Pθ,kRk −τ∑

Pθ,k log

k

k

Recognizing Jˆ(Pθ) = ∑k Pθ,kRk − τDKL(Pθ∥Pt) = ∑k Pθ,kRk − τ ∑k Pθ,k log(Pθ,k/Pt,k), we obtain Jˆ(Pθ) = −τDKL(Pθ∥w∗) + τ log Zˆ.

| |
|---|

###### B.5 Proof of Theorem 2

- Theorem 2 (Performance improvement bound). Assume |Rk| ≤ Rmax. If the projection step achieves TV(Pt+1, w∗) ≤ ϵproj, then

Rˆ(Pt+1) ≥ Rˆ(Pt) + τ DKL(w∗∥Pt) + DKL(Pt∥w∗)

− 2Rmaxϵproj projection error

. (9)

target gain ≥ 0

Proof. (a) By Proposition 3, Jˆ(w) = −τDKL(w∥w∗) + τ log Zˆ. Evaluating at the anchor Pt: Rˆ(Pt) = Jˆ(Pt) = τ log Zˆ − τDKL(Pt∥w∗). (40)

Evaluating at the target w∗ (where DKL(w∗∥w∗) = 0):

Rˆ(w∗) = Jˆ(w∗) + τDKL(w∗∥Pt) = τ log Zˆ + τDKL(w∗∥Pt). (41) Subtracting: Rˆ(w∗) − Rˆ(Pt) = τ[DKL(w∗∥Pt) + DKL(Pt∥w∗)].

- (b) We bound the expected reward error using the Total Variation (TV) distance. By definition, the L1 norm relates

to the TV distance as ∥Pt+1 − w∗∥1 = 2TV(Pt+1, w∗). By applying Pinsker’s inequality, the TV distance is upper-bounded by either choice of KL projection:

TV(Pt+1, w∗) ≤

- 1

- 2

min DKL(w∗∥Pt+1), DKL(Pt+1∥w∗) . (42) Assuming the projection step achieves TV(Pt+1, w∗) ≤ ϵproj, we apply H¨older’s inequality with |Rk| ≤ Rmax:

|Rˆ(Pt+1)− Rˆ(w∗)| = ∑

k

(Pt+1,k − w∗

k)Rk ≤ Rmax∥Pt+1 − w∗∥1 = 2Rmaxϵproj. (43) Substituting this error term back into the minorization inequality from part (a) yields the final bound.

- (c) Combining (a) and (b): Rˆ(Pt+1) ≥ Rˆ(w∗) − 2Rmaxϵproj

≥ Rˆ(Pt) + τ[DKL(w∗∥Pt) + DKL(Pt∥w∗)] − 2Rmaxϵproj.

| |
|---|

###### B.6 Proof of Proposition 2

- Proposition 2 (Idealized full-space convergence). Let π0(y) > 0 for all y, and assume R(y) is bounded. Under exact proximal updates πt+1(y) ∝ πt(y) exp(R(y)/τ), the iteration satisfies πt(y) ∝ π0(y) exp(tR(y)/τ) and Eπt[R] → maxy R(y) as t → ∞.

Proof. By induction: the base case t = 0 is trivial. If πt(y) ∝ π0(y) exp(tR(y)/τ), then πt+1(y) ∝ πt(y) exp(R(y)/τ) ∝ π0(y) exp((t + 1)R(y)/τ).

For convergence, consider any two responses y1, y2 with R(y1) > R(y2):

- πt(y1)

- πt(y2)

- π0(y1)

- π0(y2)

R(y1)−R(y2)

τ → ∞. (44) Since π0(y) > 0 for all y, the mass concentrates on argmaxy R(y), giving Eπt[R] → maxy R(y).

exp t ·

=

| |
|---|

Connecting global optimality to LPO. Proposition 2 characterizes the ideal full-space proximal operator: if one could exactly apply the Gibbs update over the entire response space, the resulting iteration converges to the global RL optimum. For autoregressive LLMs, however, the required partition function is intractable over the combinatorially large sequence space. This computational barrier motivates LPO. Rather than operating in the full space, LPO restricts the same target-projection principle to the finite response simplex induced by K sampled trajectories, yielding a principled and fully tractable approximation to the ideal proximal step.

###### B.7 Proof of Corollary 1

- Corollary 1 (Gradient coefficient properties). The forward KL gradient coefficients cfwdk satisfy: (a) bounded: |cfwdk | ≤ 1; (b) zero-sum: ∑k cfwdk = 0; (c) self-correcting: cfwdk → 0 as Pθ → w∗.

Proof. Let cfwdk = Pθ,k − wk∗ where Pθ, w∗ ∈ ∆K−1.

- (a) Since Pθ,k ∈ [0,1] and wk∗ ∈ [0,1], we have cfwdk ∈ [−1,1], hence |cfwdk | ≤ 1.
- (b) Since both Pθ and w∗ are probability distributions, ∑kK=1 cfwdk = ∑k Pθ,k − ∑k wk∗ = 1 − 1 = 0. Partitioning

into positive and negative parts: ∑cfwd

k >0 cfwdk = − ∑cfwd

k <0 cfwdk . Therefore ∑k |cfwdk | = 2 ∑cfwd

k >0 cfwdk . Since each cfwdk ≤ 1 and the positive parts sum to at most 1 (because ∑cfwd

k >0 cfwdk ≤ ∑cfwd

k >0 Pθ,k ≤ 1), we obtain ∑k |cfwdk | ≤ 2.

- (c) As Pθ → w∗, cfwdk = Pθ,k − wk∗ → 0 for all k by definition.

For the parameter-space bound, ∇θLLPOfwd = ∑k cfwdk ∇θ log πθ(yk|x), so by the triangle inequality: ∥∇θLLPOfwd∥ ≤ ∑k |cfwdk | · ∥∇θ log πθ(yk|x)∥ ≤ 2Gmax.

| |
|---|

###### B.8 Proof of Corollary 2

- Corollary 2 (Mode-Coverage). If wk∗ ≥ α and DKL(w∗∥Pθ) ≤ D, then Pθ,k > α exp (−D/α − 1).

Proof. To rigorously bound Pθ,k, we construct a binary event space (whether a response is k or not k). By the Data Processing Inequality (DPI), the binary KL divergence is bounded by the full KL divergence:

wk∗ Pθ,k

1 − wk∗ 1 − Pθ,k ≤ DKL(w∗∥Pθ) ≤ D. (45)

w∗

##### + (1 − w∗

k log

k) log

Since 1 − Pθ,k ≤ 1, the term log(1/(1 − Pθ,k)) ≥ 0. Dropping this non-negative component preserves the upper bound inequality:

wk∗ Pθ,k

w∗

##### + (1 − w∗

k) log(1 − w∗

k log

k) ≤ D. (46)

Rearranging the terms to isolate Pθ,k, we obtain: log

wk∗ Pθ,k ≤

D

wk∗ −

1 − wk∗ wk∗

log(1 − w∗

k). (47)

Exponentiating both sides yields the rigorously corrected lower bound:

D wk∗

Pθ,k ≥ w∗

k exp −

(1 − w∗

k)

1−wk∗

wk∗ . (48)

Let f(x) = x exp(−D/x)(1 − x)1−xx . It can be shown that f(x) is monotonically increasing for x ∈ (0,1). Given the assumption that wk∗ ≥ α > 0, it follows that Pθ,k ≥ f(wk∗) ≥ f(α). Therefore, we conclude:

D α

D α − 1 . (49)

(1 − α)1−αα ≥ α exp −

Pθ,k ≥ α exp −

| |
|---|

#### C Additional Discussions

###### C.1 Contribution Clarification

This work mainly makes two contributions: (i) the Target-Projection (TP) framework, a unified geometric interpretation showing that dominant group-based PG methods implicitly construct the same Gibbs target family and approximate a reverse KL projection toward it; and (ii) Listwise Policy Optimization (LPO), which makes this target-projection explicit and, by decoupling the target from the projection, opens divergence selection as a new design axis inaccessible under the implicit PG paradigm, with provable theoretical guarantees and consistent empirical gains.

Several clarifications are included regarding the scope and design choices.

- 1. The TP analysis and LPO operate in the group-based regime (K ≥ 2), which covers the vast majority of contemporary RLVR practice. Single-sample methods (K=1) lack a per-prompt simplex and require a different analytical treatment, as discussed in Appendix C.2.
- 2. The specific choice of forward or reverse KL is not a core contribution of this work, with the broader design space discussed in Appendix C.6. Reverse KL is a natural choice since policy gradient implicitly performs an approximate reverse KL projection (Proposition 1). Forward KL is similarly motivated by its mode-covering geometry, whose benefit for diversity has been observed in adjacent settings (Wang et al., 2023; Li et al., 2025; Anthony et al.).
- 3. Recent engineering innovations, e.g., dynamic sampling and asymmetric clipping (Yu et al., 2025), are orthogonal to LPO and can be combined with it, as discussed in Appendix C.3. The current experiments intentionally use a minimal shared pipeline to cleanly attribute gains to the target-projection mechanism.
- 4. The experiments adopt a paired-temperature design, varing only the projection mechanism to isolate it as the sole controlled variable. While the temperature τ could be independently tuned, this is deliberately avoided to ensure a fair comparison, leaving for future work.

- 5. All theoretical results and the implementation hold for arbitrary rewards Rk ∈ R without binary assumption. The focus on binary outcome rewards reflects the dominant RLVR setting, while the programming experiments already assess a non-binary reward.
- 6. The experimental evaluation focuses on reasoning tasks with verifiable rewards, the primary application domain of group-based PG methods. The TP analysis and LPO are not inherently limited to this setting: extending empirical validation to broader RL post-training scenarios, e.g., RLHF with learned reward models, is a natural direction for future work.

###### C.2 Extensions and Future Directions

Step-level listwise projection. Real-world applications often necessitate fine-grained optimization beyond sequence-level rewards, such as multi-turn agentic RL (Jin et al., 2025) or reasoning tasks with dense step-level rewards (Lightman et al., 2023). The current sequence-level framework may extend to these scenarios: given a shared intermediate state, one can sample K candidate continuations to form the local response simplex. Crucially, deriving the target for these immediate steps requires estimating their expected final outcomes. This can be achieved by rolling out each continuation to the terminal state. Alternatively, to bypass the prohibitive cost of full rollouts, one can rely on a value network or a value-calibrated process reward model (Lightman et al., 2023) to estimate these expected future returns. In either setting, the core LPO machinery carries over naturally, shifting the primary practical challenge to the fidelity of the step-level value estimation.

Off-policy replay. Because the listwise projection operates on the response simplex for each prompt, LPO can theoretically incorporate off-policy data to improve sample efficiency and amortize the high rollout costs typical of RLVR. Specifically, by recording the behavior policy πb used to generate past responses, LPO can account for off-policy drift via importance sampling ratios πt/πb and πθ/πb. The listwise normalization implicitly acts as a self-normalized importance sampling (SNIS) estimator, inherently adapting the policy and target distribution without altering the underlying projection geometry. Despite this theoretical elegance, realizing off-policy replay introduces practical optimization hurdles. As the policy evolves, severe drift from stale checkpoints can yield extreme probability ratios, which may collapse the effective listwise distributions and destabilize the projection gradients. Developing robust staleness-filtering or trust-region buffer management strategies to stabilize off-policy LPO remains a promising direction for future work.

Beyond group-based sampling. Current LPO requires K ≥ 2 responses per prompt to form the response simplex, which precludes direct application in single-sample (K=1) pipelines. One potential resolution is to assemble virtual groups using the off-policy replay buffer, though this inherits the aforementioned stability challenges. A minimal alternative constructs a virtual response simplex by pairing the single sampled reward R with a batch-level baseline b.

This contrastive formulation yields a sigmoid-squashed gradient coefficient c = 12 − σ (R − b)/τ that preserves boundedness (|c| ≤ 1/2), though it necessarily sacrifices the zero-sum property as there is no physical group. Both relaxations remain exploratory and characterizing their practical tradeoffs is a promising avenue.

Alternative divergences and adaptive scheduling. A distinctive feature of the explicit target-projection framework is the complete decoupling of the target distribution from the projection divergence, which is a critical design axis unavailable to policy gradient methods. This separation naturally invites the exploration of entirely different statistical divergences that might induce unique and favorable optimization geometries tailored to specific reasoning tasks, as analyzed in Appendix C.6. Furthermore, this decoupling enables dynamic scheduling strategies during training. For instance, one could employ forward KL in early stages to encourage broad exploration, and subsequently switch to reverse KL for stable late-stage exploitation, or progressively anneal the temperature τ to sharpen the target as the performance improves. Systematic exploration of this expanded design space constitutes a natural next step for RL post-training.

###### C.3 Existing Group-based RLVR as Implicit Target-Projection

As revealed in Section 3, the dominant group-based RL algorithms can be unified under a shared geometric structure: each defines an implicit Gibbs target distribution w∗ and executes an approximate reverse KL projection via policy gradient. The methods differ primarily in how they normalize advantages, which determines the implicit temperature τ and thus the sharpness of w∗. Table 2 groups these methods by their implicit target family.

σG-family: GRPO (Shao et al., 2024), DAPO (Yu et al., 2025), CISPO (Chen et al., 2025), GSPO (Zheng et al.,

- 2025).

Rk − µG σG

, τ = σG = µG(1 − µG), w∗ = softmax(R/σG). (50)

Ak =

The temperature is adaptive: maximal at µG = 0.5 (balanced groups) and vanishing as µG → 0 or 1 (nearunanimous groups), coupling target sharpness to group difficulty. DAPO adds four projection-level innovations:

Table 2: Target-Projection decomposition of existing methods, grouped by implicit target family.

Target family Methods τ

softmax(R/σG) GRPO, DAPO, CISPO, GSPO σG softmax(R) Dr.GRPO, RLOO (τ≈1), ReMax 1

softmax(R/µG) MaxRL µG softmax(R/σB′) REINFORCE++w/Baseline σB′

asymmetric clipping, dynamic sampling to filter uninformative groups, token-level loss normalization, and overlong reward shaping. CISPO modifies the projection by replacing clipping with a stop-gradient on the clipped importance ratio, preserving gradient contributions from all tokens. GSPO lifts the importance ratio and clipping from the

token level to the sequence level sk = [πθ(yk|x)/πθold(yk|x)]1/|yk|, aligning the optimization unit with the reward granularity. Many of these projection-level engineering tricks are orthogonal to our target construction and can be

seamlessly integrated into the LPO framework. τ≈1 family: Dr.GRPO (Liu et al., 2025b), RLOO (Ahmadian et al., 2024), ReMax (Li et al., 2023).

Ak = Rk − µ, τ = 1, w∗ = softmax(R). (51)

Dr.GRPO removes σG normalization (fixing τ = 1) and adopts token-level loss normalization to address length bias. RLOO uses a leave-one-out baseline (τ = (K−1)/K → 1), yielding an unbiased advantage estimator with nearly

the same implicit target. ReMax uses a greedy-decode baseline Rgreedy which cancels in the softmax, recovering the same target.

MaxRL (Tajwar et al., 2026). Ak = (Rk−µG)/µG, τ = µG = n/K, w∗ = softmax(R/µG). The temperature is directly proportional to the success rate, implementing an implicit curriculum: hard prompts (low µG) receive aggressively sharp targets to encourage exploitation, while easy prompts (high µG) receive diffuse targets to maintain diversity.

REINFORCE++ (Hu, 2025). REINFORCE++ proposes two variants. The base variant uses single-stage batch normalization Ak = (Rk − µB)/σB; its primary use case is K=1 , where no per-prompt group exists and the target-projection decomposition does not apply. The w/ Baseline variant employs a two-stage process: first subtract the per-group mean to reshape rewards, A′k = Rk − µG, then normalize by the global batch statistics of these centered rewards, Anormk = (A′k − µB′)/σB′. Since both µG and µB′ are constant within a group, they cancel under softmax, yielding w∗ = softmax(R/σB′) with τ = σB′. Here σB′ is the batch-level standard deviation of the group-centered rewards.

###### C.4 Listwise vs. Pointwise Projection

An alternative to the listwise framework developed in Section 4 is standard pointwise projection, a paradigm widely used in classical RL algorithms (e.g., MPO (Abdolmaleki et al., 2018) and AWR (Peng et al., 2019)). Both paradigms

share the same target step, constructing the reward-weighted Gibbs distribution wk∗ ∝ πb(yk) exp(Rk/τ), but they diverge fundamentally in how they project the policy toward it.

Independent vs. coupled formulation. Pointwise projection minimizes a weighted negative log-likelihood:

K

### ∑

w∗

k log πθ(yk|x), (52)

Lpointwise = −

k=1

which treats each sampled response independently. The gradient coefficient for response k is simply cpointk = −wk∗. This yields a strictly one-sided update that pushes probability mass toward high-weight responses without any

coupled counterbalancing force.

In contrast, LPO with forward KL minimizes divergence DKL(w∗∥Pθ), where Pθ = softmax(sθ) is the policy’s listwise distribution. This explicit listwise formulation couples all K responses through a shared normalization

factor. Consequently, the gradient coefficient ck = Pθ,k − wk∗ is strictly two-sided: responses where the policy over-allocates probability mass (Pθ,k > wk∗) are actively suppressed, while under-allocated responses are boosted.

Structural consequences. This architectural difference in the projection space produces three structural properties that pointwise projection inherently lacks:

- 1. Zero-sum updates. For LPO, the coefficients strictly sum to zero: ∑k ck = 0, acting as a built-in control

variate for variance reduction. For pointwise projection, ∑k cpointk = − ∑k wk∗ = −1, yielding a net gradient direction that exerts a continuous, uncalibrated pull on the parameter space.

- 2. Bounded gradients. LPO coefficients satisfy ∑k |ck| ≤ 2 (Corollary 1), providing an intrinsic, rewardscale-invariant bound on the projection step. Pointwise projection lacks this relative scaling.
- 3. Self-correcting convergence. As Pθ → w∗, the LPO coefficients vanish (ck = Pθ,k − wk∗ → 0), meaning

optimization naturally terminates once the target is matched. Pointwise coefficients (cpointk = −wk∗) are constant with respect to πθ.

Origin of the difference. The pointwise objective − ∑k wk∗ log πθ(yk) mathematically corresponds to the crossentropy H(w∗, πθ), which equals DKL(w∗∥πθ) + H(w∗). Because πθ is evaluated independently per response and is not normalized over the response group, this KL divergence measures the gap between unnormalized densities. LPO, by contrast, operates on the normalized listwise distribution Pθ ∈ ∆K−1, which lives on the exact same finite probability simplex as w∗. This shared simplex geometry is what dictates the two-sided, zero-sum gradient structure.

Connection to Expectation-Maximization. The explicit target-projection procedure mirrors the structure of the Expectation-Maximization (EM) algorithm (Dayan & Hinton, 1997; Neal & Hinton, 1998): the Gibbs target construction resembles an E-step that forms a target distribution, while the divergence minimization corresponds to an M-step that fits the model to this target.

###### C.5 Connection to DPO and Preference Optimization

When K = 2, LPO reduces to a pairwise objective closely related to Direct Preference Optimization (DPO) (Rafailov et al., 2024). Consider two responses: a preferred response yw with reward Rw = 1, and a dispreferred response yl with reward Rl = 0.

For two responses, the listwise distribution becomes

exp(sw) exp(sw) + exp(sl)

= σ(sw − sl), (53)

Pw =

where sk = log(πθ(yk|x)/πb(yk|x)) and σ(·) is the sigmoid function. In on-policy setup (πt = πb), the baseline distribution is uniform, yielding

ww∗ = σ(1/τ), w∗

l = σ(−1/τ). (54) The forward-KL objective then simplifies to

LLPOfwd = −σ(1/τ) log σ(sw − sl) − σ(−1/τ) log σ(sl − sw), (55) which is a binary cross-entropy objective with temperature-controlled soft targets. By comparison, DPO uses the pairwise logistic objective

LDPO = − log σ β(sw − sl) , (56) where sk = log(πθ(yk|x)/πref(yk|x)).

Thus, both methods share the same pairwise sigmoid structure, but differ fundamentally in four aspects: (i) standard DPO operates within an offline paradigm on static datasets, whereas LPO is an online RL algorithm; (ii) DPO penalizes logits against a static reference policy πref, whereas LPO derives its target within a trust region around the pre-update policy πt; (iii) DPO is derived under a Bradley–Terry style preference model, whereas LPO arises from explicit divergence projection on the response simplex; (iv) LPO uses soft targets controlled by τ, recovering a hard preference target as τ → 0.

This view places DPO-style pairwise optimization as the foundational K = 2 baseline of the broader LPO framework, which naturally extends from pairwise preferences (K = 2) to listwise optimization (K > 2), and further to the population-level RL-as-inference limit as K → ∞.

Distinction from Listwise Preference Optimization. Recent works like LiPO (Liu et al., 2025a) extend DPO to listwise preference optimization with Plackett–Luce style ranking models. Despite the similar “listwise” terminology, these methods learn from ranked preference data y1 ≻ y2 ≻ · · · ≻ yK. In contrast, LPO is designed for online RLVR, utilizing absolute reward signals for explicit target-projection without any ranking-model assumptions. Mathematically, Plackett–Luce ranking models and our Gibbs target both take a normalized softmax form. Thus, they represent complementary ways of obtaining the same exponential-family target: one inferred from comparisons, the other directly specified by rewards.

###### C.6 Extension to General Divergences

In Section 4.2, we instantiated LPO using forward and reverse KL divergences. However, the projection framework is not specific to KL and can be applied to any differentiable divergence defined on the probability simplex, including general f-divergences such as the Jensen–Shannon divergence.

Let L = D(w∗, Pθ) be a differentiable divergence on ∆K−1. The gradient takes the form

K

∑

ck∇θ log πθ(yk | x), (57)

∇θL =

k=1

where the gradient coefficient is ck = ∂L/∂sθ,k. Applying the chain rule with the softmax Jacobian, we have

K

K

∂Pθ,j ∂sθ,k

∂L ∂Pθ,j

∂L ∂Pθ,k − Pθ,k

∂L ∂Pθ,j

∑

∑

. (58)

ck =

= Pθ,k

Pθ,j

j=1

j=1

Summing these coefficients over all K responses yields the identity

K

K

∂L ∂Pθ,k −

∑

∑

ck =

Pθ,k

k=1

k=1

K

∑

Pθ,k

k=1

=1

K

∂L ∂Pθ,j

∑

Pθ,j

j=1

= 0. (59)

This zero-sum property is a direct consequence of the softmax parameterization on the probability simplex and holds for any differentiable objective defined on Pθ. It plays a role analogous to a baseline in policy gradient methods and contributes to stabilizing the update.

While this zero-sum property is universal, other characteristics, such as coefficient boundedness or mode-seeking behavior, depend on the specific choice of divergence. KL divergences are adopted as natural default choices in LPO due to their stability and well-understood geometry.

###### C.7 Entropy Regularization and Reverse KL Diversity

Reverse KL as max-entropy RL. The objective of LPO with reverse KL is equivalent to maxθ ∑k Pθ,kϕk + H(Pθ), mirroring the maximum entropy RL objective (Ziebart, 2010). Here, the explicit entropy bonus emerges naturally from the structural formulation of the divergence. In contrast, standard policy gradient methods lose this property, as they are equivalent to a first-order approximation only at the on-policy point.

Entropy regularization as target mixing. Adding entropy bonus γH(πθ) modifies the listwise target to w˜∗ = softmax(R/(τ + γ)) in the on-policy setup, equivalent to increasing τ by γ. The entropy bonus is redundant when τ is a controllable hyperparameter.

###### C.8 Broader Societal Impacts

This work introduces LPO as a novel paradigm for RLVR. As an algorithmic contribution to policy optimization, LPO may improve the efficiency and stability of RL post-training, potentially reducing the computational cost of training strong LLMs. More broadly, improvements in reasoning capability and training efficiency may indirectly benefit downstream applications of LLMs, such as scientific problem solving, software development, and educational tools, by enabling more capable and reliable systems. On the negative side, the method inherits the general risks associated with increasingly capable LLMs, including potential dual-use concerns if deployed without appropriate safeguards. Additionally, while LPO improves optimization efficiency in RLVR, addressing the environmental and societal costs of large-scale model training remains an open challenge for the broader research community.

#### D Implementation Details

- D.1 Tasks

- D.1.1 Logical Reasoning

Training Dataset. We adopt the Countdown Number Game as the logical reasoning testbed. This task requires models to synthesize basic arithmetic operations to reach a target value using a provided set of integers. We use a subset of 2000 problems sampled from the Countdown 34 dataset (https://huggingface.co/ datasets/Jiayi-Pan/Countdown-Tasks-3to4) as training dataset, which supplies either three or four source numbers per question.

Evaluation Benchmarks. We assess model performance using two reserved evaluation sets: a split of 512 instances from Countdown 34 (CD34) and a subset of 512 instances from Countdown 4 (CD4), available at https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-4. The CD4 variant is notably more difficult because it strictly guarantees four source numbers per problem, thereby massively expanding the combinatorial search space. To evaluate performance, we generate 64 independent responses per instance to compute both the expected Pass@1 (the average correctness across all 64 samples) and the Pass@64 metrics. All reported training curves reflect the average performance across both the CD34 and CD4 benchmarks.

Reward Function. Following Pan et al. (2025), we augment the binary accuracy reward with a formatting bonus. This design explicitly incentivizes proper structural adherence alongside correct reasoning:

 

1 if the response is correct, 0.1 if the response is incorrect but properly formatted, 0 otherwise.

r =

(60)



###### D.1.2 Mathematics Reasoning

Training Dataset. Following Qu et al. (2025), we train models on the MATH dataset (Hendrycks et al., 2021), which consists of 7.5k problems from mathematics competitions. We use the public version hosted at https:// huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval. For extended validation at a larger scale, we additionally train the Qwen3-14B-Base model on the Polaris (An et al., 2025) dataset in Appendix E.1, which comprises a broader collection of roughly 53k high-quality mathematical reasoning problems. The Polaris dataset is hosted at https://huggingface.co/datasets/POLARIS-Project/ Polaris-Dataset-53K.

Evaluation Benchmarks. Following Gao et al. (2025); Qu et al. (2026), we evaluate mathematical reasoning performance on a suite of benchmarks, including AIME24, AIME25, AMC23, MATH500 (Lightman et al., 2023), Minerva Math (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024), using the datasets hosted at https://huggingface.co/datasets/math-ai. Following prior works (Gao et al., 2025; Qu et al., 2026), we sample k independent responses per problem to compute both the expected Pass@1 (defined as the average accuracy across all k samples, or avg@k) and the Pass@k metrics. The sample size k is tailored to the size and difficulty of each benchmark: we set k = 32 for competition-level suites (AIME24, AIME25, AMC23), k = 4 for Minerva Math, and k = 1 for MATH500 and OlympiadBench. Training curves report the average performance across all math benchmarks.

Reward Function. Following the default configuration in verl (Sheng et al., 2024), we use a binary reward function that assigns 1 to correct responses and 0 otherwise.

###### D.1.3 Programming

Training Dataset. To assess the training performance in code generation, following Cui et al. (2025), we adopt the code split in PRIME dataset, available at https://huggingface.co/datasets/PRIME-RL/ Eurus-2-RL-Data, which contains 25.3k problems that are mainly programming competition level.

Evaluation Benchmarks. For evaluation, we evaluate on the 1k held-out validation problems from the PRIME code dataset. For each prompt, we sample k = 8 independent Python programs to compute the expected Pass@1 (the average success rate across the 8 samples) and the pass@8 metrics.

Reward Function. Following PRIME (Cui et al., 2025), we extract the generated Python program and evaluate it against a suite of test cases. The reward is defined as the fraction of tests passed:

number of passed tests total number of tests

. (61)

r =

Compared to a strict binary reward setup, this formulation provides a denser learning signal, yielding values in [0,1] where 1 indicates a fully correct solution and 0 indicates complete failure.

###### D.1.4 Geometry

Training Dataset. We train on the 2.1k-problem training split of the Geometry3k dataset (Lu et al., 2021; Hiyouga, 2025), available at https://huggingface.co/datasets/hiyouga/geometry3k. Each problem in Geometry3k consists of a geometric diagram and an accompanying natural language question, often requiring multi-step spatial or logical reasoning.

PG (GRPO) LPOfwd LPOrev

###### Polaris (GRPO) - Pass@1

Polaris (GRPO) - Pass@k

- 55
- 56
- 57
- 58
- 59
- 60
- 61
- 62
- 63

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
| | | | | |
| | | | | |
| | | | | |

50

Pass@1Accuracy(%)

Pass@kAccuracy(%)

48

46

44

42

0 50 100 150 200

0 50 100 150 200

Step

Step

- Figure 8: Scalability validation. We compare LPO with GRPO by training Qwen3-14B-Base on the larger Polaris dataset.

Evaluation Benchmarks. We evaluate performance on the official 601-problem test split of Geometry3k. For each prompt, we generate 16 independent responses to calculate both the expected Pass@1 (the average accuracy across 16 samples) and the Pass@16 metrics.

Reward Function. Following verl (Sheng et al., 2024), we use the same reward function as in Countdown. Appendix F presents data examples from each of the training datasets.

###### D.2 Models

We evaluate eight models spanning a diverse range of types, parameter scales, and model families. All models are sourced directly from their official Hugging Face repositories and used as released:

- • Qwen3-1.7B-Base: https://huggingface.co/Qwen/Qwen3-1.7B-Base;
- • Qwen3-4B-Base: https://huggingface.co/Qwen/Qwen3-4B-Base;
- • Qwen3-8B-Base: https://huggingface.co/Qwen/Qwen3-8B-Base;
- • Qwen3-14B-Base: https://huggingface.co/Qwen/Qwen3-14B-Base;
- • Qwen2.5-VL-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct;
- • DeepSeek-R1-Distill-Qwen-1.5B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1. 5B;
- • Llama-3.1-8B-Instruct: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct;
- • Mistral-7B-Instruct-v0.1: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1;

###### D.3 Training Details

We evaluate our method against three representative group-based policy gradient baselines: GRPO (Shao et al., 2024), Dr.GRPO (Liu et al., 2025b), and MaxRL (Tajwar et al., 2026), all implemented within the verl framework (Sheng et al., 2024). Across all four reasoning scenarios, we sample a group of K = 8 responses per prompt during training to estimate advantages or construct response simplex. The generation temperature is set to 1.0 with top p = 1.0,top k = −1.0, and we disable the KL penalty by setting β = 0, consistent with Yu et al. (2025). Evaluation generations use a lower temperature of 0.6, following common practice (Qu et al., 2025).

We tailor the batch sizes and context lengths according to the complexity of the specific benchmark. For the Math and PRIME-Code tasks, we set the training batch size to 256, the mini-batch size to 128, and the maximum response length to 4096 tokens. For the Countdown and Geometry tasks, we scale down the training batch size to 128 and the mini-batch size to 64, with the maximum response length capped at 1024 tokens. This configuration performs two gradient updates per iteration, inherently introducing a mild off-policy drift. A strictly fully on-policy ablation is provided in Appendix E.4.

Optimization is uniformly performed using Adam (Kingma & Ba, 2014) with a learning rate of 1e−6 across all tasks. The optimizer parameters are set to β = (0.9,0.999) with a weight decay of 0.1. The clipping parameter is fixed at ϵ = 0.2. Given the highly non-linear parameter-space updates, we additionally apply token-level clipping (Schulman et al., 2017b). The token-level log-density ratio δk,i = log πθ(yk,i|x, yk,<i) − log πb(yk,i|x, yk,<i) is clipped to [log(1−ϵ),log(1+ϵ)] and then weighted by ck to form the final loss.

All experiments are conducted on 8 NVIDIA H20 GPUs.

PG (Dr.GRPO) LPOfwd LPOrev

###### Countdown 4B (Dr.GRPO)

###### Math 1.7B (Dr.GRPO)

###### Math 8B (Dr.GRPO)

###### Prime-Code 1.7B (Dr.GRPO)

###### Geo3k 3B (Dr.GRPO)

0.30

0.10

0.06

0.200

0.40

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
| | | | | |

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
| | | | | |

0.35

0.175

0.25

0.05

0.08

ResponseEntropy

ResponseEntropy

ResponseEntropy

ResponseEntropy

ResponseEntropy

0.30

0.150

0.20

0.04

0.25

0.125

0.06

0.15

0.20

0.100

0.03

0.10

0.04

0.15

0.075

0.02

0.05

0.10

0.050

0.02

0.05

0.00

0.01

0.025

20 40 60 80 100 120

100 200 300 400 500

100 200 300 400 500

100 200 300 400

50 100 150 200

Step

Step

Step

Step

Step

0.13

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

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.09

0.07

0.12

2.0

0.08

0.11

0.06

0.08

GradNorm

GradNorm

GradNorm

GradNorm

GradNorm

0.10

1.5

0.05

0.07

0.07

0.09

0.04

0.06

1.0

0.08

0.06

0.03

0.05

0.07

0.5

0.05

0.06

0.02

0.04

0.0

0 25 50 75 100 125

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400

0 50 100 150 200

Step

Step

Step

Step

Step

1200

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
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
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

800

600

650

300

1100

750

ResponseLen

ResponseLen

ResponseLen

ResponseLen

ResponseLen

600

1000

250

550

900

700

200

550

500

800

150

650

500

700

450

100

450

600

600

0 25 50 75 100 125

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400

0 50 100 150 200

Step

Step

Step

Step

Step

- Figure 9: Training dynamics of LPO variants and Dr.GRPO. Rows from top to bottom respectively show the curves of response entropy, gradient norms, and response lengths.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

20 40 60 80 100 120

Step

0.00

0.05

0.10

0.15

0.20

0.25

0.30

ResponseEntropy

Countdown 4B (MaxRL)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100 200 300 400 500

Step

0.02

0.04

0.06

0.08

0.10

ResponseEntropy

Math 1.7B (MaxRL)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100 200 300 400 500

Step

0.01

0.02

0.03

0.04

0.05

0.06

ResponseEntropy

Math 8B (MaxRL)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

100 200 300 400

Step

0.025

0.050

0.075

0.100

0.125

0.150

0.175

0.200

ResponseEntropy

Prime-Code 1.7B (MaxRL)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

50 100 150 200

Step

0.05

0.10

0.15

0.20

0.25

0.30

0.35

0.40

ResponseEntropy

Geo3k 3B (MaxRL)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 25 50 75 100 125

Step

0.2

0.3

0.4

0.5

GradNorm

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 100 200 300 400 500

Step

0.125

0.150

0.175

0.200

0.225

0.250

0.275

GradNorm

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 100 200 300 400 500

Step

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

GradNorm

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 100 200 300 400

Step

0.15

0.20

0.25

0.30

0.35

GradNorm

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 50 100 150 200

Step

0.5

1.0

1.5

2.0

2.5

3.0

GradNorm

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 25 50 75 100 125

Step

400

450

500

550

ResponseLen

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 100 200 300 400 500

Step

600

700

800

900

1000

1100

ResponseLen

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0 100 200 300 400 500

Step

600

800

1000

1200

ResponseLen

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 100 200 300 400

Step

500

550

600

650

700

ResponseLen

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 50 100 150 200

Step

150

200

250

300

350

ResponseLen

PG (MaxRL) LPOfwd LPOrev

- Figure 10: Training dynamics of LPO variants and MaxRL. Rows from top to bottom respectively show the curves of response entropy, gradient norms, and response lengths.

#### E Extended Experimental Results

###### E.1 Scalability Validation

To verify the scalability and extensibility of the LPO framework, we conduct additional experiments using the Qwen3-14B-Base model on the Polaris dataset, which contains approximately 53k complex reasoning problems. We compare both LPO variants with the GRPO baseline. As shown in Fig. 8, LPO-fwd exhibits remarkable sample efficiency, reaching the peak performance achieved by GRPO at 200 training steps within only 70 steps, while simultaneously providing significant improvements in both Pass@1 and Pass@k metrics. For the LPO-rev variant, although its Pass@1 accuracy is comparable to GRPO, it shows superior robustness in maintaining Pass@k, effectively preserving response diversity. These findings provide strong evidence that LPO is scalable and capable of maintaining its theoretical advantages alongside increases in model capacity and data volume.

PG (GRPO) LPOfwd LPOrev

###### Qwen3-4B-Base - Pass@1

###### DeepSeek-R1-Distill-1.5B - Pass@1

###### Llama-3.1-8B-Instruct - Pass@1

###### Mistral-7B-Instruct - Pass@1

60

26

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

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

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

50

60

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

Pass@1Accuracy(%)

24

55

40

55

22

50

30

20

50

45

18

20

45

40

16

40

10

35

40 60 80 100 120

25 50 75 100 125 150

20 40 60 80 100 120

0 50 100 150

Step

Step

Step

Step

###### Qwen3-4B-Base - Pass@64

###### DeepSeek-R1-Distill-1.5B - Pass@64

###### Llama-3.1-8B-Instruct - Pass@64

Mistral-7B-Instruct - Pass@64

80

40.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

85

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

80

75

37.5

Pass@64Accuracy(%)

Pass@64Accuracy(%)

Pass@64Accuracy(%)

Pass@64Accuracy(%)

75

70

35.0

80

65

70

32.5

75

60

30.0

65

55

27.5

60

70

50

25.0

55

45

22.5

65

40

50

20.0

40 60 80 100 120

25 50 75 100 125 150

20 40 60 80 100 120

0 50 100 150

Step

Step

Step

Step

- Figure 11: Generalization of LPO across diverse LLM families. Performance is evaluated on Countdown using Qwen, DeepSeek, Mistral, and Llama backbones.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 20 40 60 80 100 120 140

Step

0

10

20

30

40

50

60

Pass@1Accuracy(%)

Countdown 4B (GRPO) - Pass@1

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 20 40 60 80 100 120 140

Step

40

50

60

70

80

Pass@kAccuracy(%)

Countdown 4B (GRPO) - Pass@k

PG (GRPO) LPOfwd LPOrev

- Figure 12: Empirical evaluation on the Countdown task under a fully on-policy regime (one gradient update per iteration).

###### E.2 Extended Training Dynamics

To corroborate the analysis presented in Sec. 5.3, we provide the extended training dynamics of LPO compared against Dr.GRPO in Fig. 9 and MaxRL in Fig. 10.

Consistent with the observations relative to the GRPO baseline in the main text, LPO variants demonstrate superior optimization properties across these baselines. Specifically, LPO maintains higher response entropy, exhibits lower and more stable gradient norms, and encourages longer response chains. These supplementary results further support the structural advantages of listwise projection.

###### E.3 Generalization across LLM Families

To evaluate the generalizability of LPO, we conduct experiments across four prominent LLM families: Qwen, DeepSeek, Mistral, and Llama. These include models with different training paradigms such as base (only pretrained), distilled, and instruction-tuned variants. As shown in Fig. 11, across all evaluated LLMs, LPO consistently improves performance on the Countdown task over the PG baseline, with especially stable gains under Pass@64 evaluation.

###### E.4 Fully On-Policy Optimization

To empirically validate the theoretical connections established in Proposition 1, we conduct an evaluation on Countdown under a strictly fully on-policy regime, as shown in Fig. 12. By setting both the batch size and the optimization mini-batch size to 256, we ensure exactly one gradient update is performed per iteration. As predicted by our analysis, the training curves of LPOrev are practically indistinguishable from standard GRPO, supporting that the group-based PG objective mathematically collapses into the exact reverse KL projection at the on-policy

Table 3: Evaluation on mathematics benchmarks. Base denotes the backbone without RLVR. Pass@1 and Pass@k are computed by averaging benchmark-level Avg@k and Pass@k scores across benchmarks, respectively. Bold and underlined values indicate the best and second-best results for each policy gradient baseline, respectively.

MATH500 Olympiad. Minerva. AMC23 AIME24 AIME25

Backbone Method

Avg@1 Avg@1 Avg@4 pass@4 Avg@32 pass@32 Avg@32 pass@32 Avg@32 pass@32 Pass@1↑ Pass@k↑

Base 52.8 21.2 21.2 32.8 30.0 79.3 3.4 25.0 3.3 23.8 22.0 40.2 GRPO 71.4 33.5 29.8 37.2 45.4 83.7 10.8 26.5 4.2 20.9 32.5 42.1

→LPOfwd 72.0 38.1 29.9 37.5 50.1 83.4 13.2 33.7 8.6 29.6 35.3 46.1

→LPOrev 73.0 37.1 29.2 36.4 47.0 83.1 13.9 26.6 9.6 22.9 35.0 42.3 DrGRPO 69.2 36.1 29.8 36.8 43.3 76.2 8.5 25.4 6.3 30.4 32.2 42.2

Qwen3-1.7B-Base

→LPOfwd 73.8 36.5 28.3 36.5 46.2 75.9 10.3 27.1 5.3 30.4 33.4 42.5

→LPOrev 70.0 36.7 28.6 38.1 45.6 78.9 10.3 31.7 6.3 26.7 32.9 43.9 MaxRL 72.6 35.3 28.6 36.9 42.4 79.0 10.6 30.8 4.8 24.6 32.4 42.8

→LPOfwd 71.8 37.5 30.5 36.4 49.9 85.5 11.8 28.6 8.5 31.9 35.0 45.6 →LPOrev 72.6 35.8 28.8 36.3 46.1 82.6 10.7 28.3 7.7 32.6 33.6 45.0

Base 68.0 33.7 31.7 44.1 46.5 84.3 12.1 39.9 7.9 31.8 33.3 50.0 GRPO 86.2 51.9 40.4 46.1 63.8 79.1 24.0 52.1 19.5 40.7 47.6 54.5

→LPOfwd 86.4 55.8 42.3 48.3 69.1 95.1 29.3 51.0 19.1 38.7 50.3 58.3

→LPOrev 85.0 53.9 41.1 46.9 67.0 93.1 23.3 45.7 21.6 40.2 48.7 56.5 DrGRPO 85.8 54.7 42.2 48.4 67.7 89.7 24.9 56.3 19.3 47.0 49.1 60.4

Qwen3-8B-Base

→LPOfwd 87.4 51.6 42.6 48.3 70.2 91.5 25.6 59.5 19.8 38.4 49.5 59.4

→LPOrev 84.6 51.0 42.0 47.8 64.9 91.4 26.0 53.0 17.9 35.3 47.7 56.9 MaxRL 86.4 53.6 42.6 48.9 66.0 93.4 23.9 48.6 18.9 41.7 48.6 58.2

→LPOfwd 89.4 54.5 44.8 52.3 69.0 94.5 23.9 57.6 21.3 47.8 50.5 63.1 →LPOrev 87.6 55.8 45.3 52.3 70.1 92.5 22.5 52.6 22.5 43.6 50.6 60.3

point. Furthermore, evaluating LPOfwd under this identical setup reveals its distinct exploration superiority: it demonstrates higher sample efficiency in early training stages and achieves a superior Pass@k accuracy.

###### E.5 Evaluation Results

Table 3 presents the final evaluation on mathematics benchmarks, with k configurations following standard practices (Gao et al., 2025; Qu et al., 2025). Furthermore, to assess out-of-distribution (OOD) generalization, Table 4 compares LPO against counterpart PG baselines (all trained on MATH using Qwen3-8B-Base) across general reasoning tasks: MMLU-Pro Wang et al. (2024), ARC-c Clark et al. (2018), and GPQA-diamond Rein et al. (2024). While specific LPO variants can improve the overall average, OOD evaluation exhibits inherent variance, suggesting multi-domain joint training as a natural direction for future work.

Table 4: Out-of-Distribution evaluation of LPO against baseline methods trained on MATH.

ARC-c MMLU-Pro GPQA

Method

Avg@32 Avg@32 Avg@32 Avg.↑ GRPO 33.4 56.0 25.3 38.2

→LPO 38.4 53.5 23.7 38.5 Dr.GRPO 33.2 55.4 23.8 37.5 →LPO 36.4 58.5 25.6 40.2 MaxRL 22.5 49.8 18.7 30.3 →LPO 26.3 51.2 19.3 32.3

#### F Data Examples

The prompt templates for MATH and Geometry3k are adopted from the official verl framework; the template for Countdown follows the format introduced in Pan et al. (2025), and we directly use the prompts for PRIME code from Cui et al. (2025).

MATH example Prompt: The points (9, −5) and (−3, −1) are the endpoints of a diameter of a circle. What is the sum of the coordinates of the center of the circle? Let’s think step by step and output the final answer within \boxed{}. Ground-Truth Answer: 0

Countdown example Prompt: A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Using the numbers [2, 54, 17], create an equation that equals 35. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> < /think> tags. And return the final answer in <answer> < /answer> tags, for example <answer> (1 + 2)/3 </answer>. Assistant: Let me solve this step by step. <think> PRIME-Code example System Prompt: When tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process. [ASSESS] [ADVANCE] [VERIFY] [SIMPLIFY] [SYNTHESIZE] [PIVOT] [OUTPUT] You should strictly follow the format below: [ACTION NAME]

- # Your action step 1
- # Your action step 2
- # Your action step 3

... Next action: [NEXT ACTION NAME] Prompt Problem: Given a natural number N less than or equal to 12, find the smallest natural number such that the number of divisors is exactly N. Constraints

* 1 ≤ N ≤ 12 Input: One natural number N is given in one line. Output: Output the smallest natural number on a line so that the number of divisors is exactly N. Examples Input 1 Output 1 Input 2 Output 2 Input 3 Output 4 Write Python code to solve the problem. Present the code in python Your code at the end.

Geometry3k example Prompt:

[Figure 3]

Find x. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> < /think> tags. The final answer MUST BE put in \boxed{}. Ground-Truth Answer:

- 4

