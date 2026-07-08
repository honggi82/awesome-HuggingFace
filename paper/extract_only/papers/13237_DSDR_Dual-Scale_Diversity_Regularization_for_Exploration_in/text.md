## DSDR: Dual-Scale Diversity Regularization for Exploration in LLM Reasoning

Zhongwei Wan*1 Yun Shen*1 Zhihao Dou2 Donghao Zhou3 Yu Zhang4 Xin Wang1 Hui Shen5 Jing Xiong6 Chaofan Tao6 Zixuan Zhong7 Peizhou Huang5 Mi Zhang1

# arXiv:2602.19895v1[cs.LG]23Feb2026

### Abstract

Reinforcement learning with verifiers (RLVR) is a central paradigm for improving large language model (LLM) reasoning, yet existing methods often suffer from limited exploration. Policies tend to collapse onto a few reasoning patterns and prematurely stop deep exploration, while conventional entropy regularization introduces only local stochasticity and fails to induce meaningful path-level diversity, leading to weak and unstable learning signals in group-based policy optimization. We propose DSDR, a Dual-Scale Diversity Regularization reinforcement learning framework that decomposes diversity in LLM reasoning into global and coupling components. Globally, DSDR promotes diversity among correct reasoning trajectories to explore distinct solution modes. Locally, it applies a length-invariant, token-level entropy regularization restricted to correct trajectories, preventing entropy collapse within each mode while preserving correctness. The two scales are coupled through a global-tolocal allocation mechanism that emphasizes local regularization for more distinctive correct trajectories. We provide theoretical support showing that DSDR preserves optimal correctness under bounded regularization, sustains informative learning signals in group-based optimization, and yields a principled global-to-local coupling rule. Experiments on multiple reasoning benchmarks demonstrate consistent improvements in accuracy and pass@k, highlighting the importance of dual-scale diversity for deep exploration in RLVR. Code is available at DSDR.

1The Ohio State University, USA 2Case Western Reserve University, USA 3The Chinese University of Hong Kong, Hong Kong 4Macquarie University, Australia 5University of Michigan, USA 6The University of Hong Kong, Hong Kong 7University College London, UK. Correspondence to: Zhongwei Wan <wan.512@osu.edu>, Mi Zhang <mizhang.1@osu.edu>.

Preprint. February 24, 2026.

### 1. Introduction

Reinforcement learning with verified reward (RLVR) (Liu et al., 2024; Shao et al., 2024) has recently emerged as a powerful paradigm for enhancing the reasoning capabilities of LLMs. Group-based policy optimization methods, such as GRPO (Guo et al., 2025), further improve training stability by exploiting relative comparisons among sampled solutions, making RLVR practical at scale. By leveraging outcome-based supervision rather than token-level imitation, RLVR has enabled substantial improvements in math and code reasoning, logical inference, and multi-step problem solving, and has become a core component of recent advances in reasoning-oriented LLM training (Comanici et al., 2025; Singh et al., 2025; Yang et al., 2025).

Despite these successes, verified reward-maximizing RL training often exhibits limited deep exploration, even when alternative valid solution paths exist (Liu et al., 2025b; Shen, 2025; Wu et al., 2025; Chen et al., 2025a). This phenomenon is widely observed across RLVR pipelines: while models improve pass@1 accuracy, they often do so by concentrating probability mass on a small set of homogeneous reasoning patterns, leading to a collapse in solution diversity. Consequently, pass@k performance fails to improve and generalization deteriorates, especially when models are evaluated on out-of-domain or more compositional reasoning tasks (Walder & Karkhanis, 2025; Jiang et al., 2025).

A natural response is to encourage diversified exploration during training, yet existing methods remain insufficient. Entropy regularization (Shen, 2025; Chen et al., 2025c; Agarwal et al., 2025), widely used in RL and RLVR, injects token-level stochasticity but mainly induces local randomness and fails to promote distinct reasoning paths. Conversely, recent diversity-driven methods (Zhang et al., 2025; Chen et al., 2025b; Li et al., 2025; Hu et al., 2025) encourage variation among generated solutions, sometimes with quality or correctness constraints. However, these approaches are typically single-scale or weakly coupled across scales: token-level entropy or uncertainty control induces local stochasticity and rarely sustains distinct reasoning trajectories, while trajectory-level diversity alone fails to prevent intra-mode entropy collapse once a few correct templates dominate. Consequently, policies may still prematurely con-

MOSAIC Evaluation Multimodal Gate

Dynamic Weighting

Multimodal Deep Research

Itemwise Score

Weight

Judge LLM

SEM: Wsem, ACC: Wacc, VQA: Wvqa

Report

Route

Categorical Evaluators

Judge LLM

Parse

Multimodal Items Collection

Item 1 Item 2

Item n

...

...

Main Pipeline

Deep Research Agents

Multimodal Deep Research Report

###### Final Score

Multimodal Deep Research Report

TRACE Evaluation FLEA

Pairwise Score

Dynamic Weighting

Judge LLM Judge LLM

CON: Wcon

VEF: Wvef

Dynamic Weighting

COV: Wcov, FID: Wfid

Weight

Multimodal Deep Research Report

*T

Claim-URL Pairs

Extract

Claim n

Claim 2

Claim 1

...

Judge LLM

URL n

URL 2

URL 1

Pair 1 Pair 2 Pair n

d₂

###### Policy Update

r₁base r₂base r₃base

###### Correct-Only Global Diversity

z₁ z₂ ... z₁

d₁

d₂ d₃

₁

z₁ z₂

d₂ ... dᵐ

₁ dᵐ₂

⋅ λ +

₁

d₂

###### JGRPO Policy Loss

∑dᵢʲ =

...

J

i≠j

DSDR

rₙbase

dₘ₁ dₘ₂

zₘ

dₘ

...

Global-to-Local Coupling

###### Problem Input

Consider a particle starting at vertex

(0,0,0) of a unit cube. At each step, the particle moves to a randomly chosen adjacent vertex with equal

Local Pos-sample Entropy

×

Regularization

JLocal

probability (1/3). What is the

expected number of steps required for the particle to first reach the opposite vertex (1,1,1)?

Softmax

Mean Token-level Entropy

DSDR: Dual-Scale Diversity Regularization for Exploration in LLM Reasoning

d₁

Baseline Exploration

DSDR Exploration

[Figure 42]

[Figure 43]

T1 T2 T3 ... Tn

| | |
|---|---|
| | |

d₂

Diversity Exploration

[Figure 44]

[Figure 45]

[Figure 46]

T1 T2 T3 ... Tn

JDSDR

Diversity Exploration

Mean Token-level Entropy d₃

[Figure 47]

[Figure 48]

Incorrect Solution

Incorrect Solution

T1 T2 T3 ... Tn

Correct Solution

Correct Solution

[Figure 49]

T1 T2 T3 ... Tn

Different Solving Steps

Coupling Weight

Solution Space

Solution Space

- Figure 1. (Left): global-to-local coupling for enhanced exploration during RL training. (Right): baseline exploration collapses to local suboptimal solutions, while DSDR promotes diverse trajectories that escape local optima and reach the correct solution space.

d₁

[Figure 50]

[Figure 51]

T1 T2 T3 ... Tn

| | |
|---|---|
| | |

d₂

Diversified Exploration

centrate on a small set of correct reasoning modes, and in group-normalized RLVR this concentration weakens withingroup preference signals as verifier rewards become nearly constant. The core tension between deep exploration and correctness therefore remains unresolved, requiring exploration to be correctness-aligned and explicitly coordinated across both trajectory- and token-scales.

bility, highlighting the importance of principled dual-scale diversity for deep exploration in RLVR. Our main contributions are summarized as follows:

[Figure 52]

[Figure 53]

[Figure 54]

T1 T2 T3 ... Tn

JDSDR

Diversified Exploration

d₃

[Figure 55]

[Figure 56]

|Hₙ: Mean Token-level Entropy<br><br>Wₙ: Coupling Weight The dashed box represents diversified explorations of problem-solving steps.|
|---|

T1 T2 T3 ... Tn

[Figure 57]

T1 T2 T3 ... Tn

- • We introduce a dual-scale perspective on exploration in LLM reasoning, explicitly distinguishing global (intermode) and local (intra-mode) diversity and clarifying their complementary roles in RLVR.
- • We propose DSDR, a correctness-aligned dual-scale diversity regularization framework that couples global diversity with positive-only, length-invariant local entropy through a global-to-local allocation mechanism.
- • We provide theoretical support for correctness preservation and signal preservation in group-normalized RLVR, together with a principled interpretation of the global-tolocal coupling, validated by consistent empirical gains.

This motivates a dual-scale formulation of exploration for LLM reasoning. (i) At the global level, exploration requires discovering and maintaining multiple distinct reasoning modes, corresponding to different solution paths. (ii) At the local level, exploration requires preventing premature entropy collapse (Cui et al., 2025; Shen, 2025) within each mode, so that correct trajectories remain robust and expressive rather than brittle or over-confident. Crucially, these two forms of diversity are complementary and can be jointly optimized rather than treated in isolation, since not all correct modes are equally valuable for further exploration, motivating a mechanism that allocates local regularization based on global distinctiveness.

### 2. Related Work

RLVR and Exploration in LLMs. Reinforcement learning with verifiable rewards (RLVR) has become a prominent approach for improving LLM reasoning (Cobbe et al., 2021; Singh et al., 2025; Guo et al., 2025). While this training can elicit emergent reasoning behaviors such as verification and self-reflection (Gandhi et al., 2025; Wan et al., 2025), it often suffers from limited exploration, where policies converge early to a narrow set of reasoning patterns, resulting in performance plateaus. To mitigate this issue, prior work has explored a range of exploration-enhancing strategies, including increasing policy stochasticity through entropy regularization or temperature adjustment (Hou et al., 2025), modifying optimization objectives via relaxed clipping or pass@k-based rewards (Yu et al., 2025; Chen et al., 2025c), and intervening in rollout dynamics to encourage mode switching, such as the sample-then-forget mechanism (Chen et al., 2025a). While these approaches improve exploration from different angles, they either rely on unstructured randomness, objective-level relaxation, or rollout-level interventions, and do not explicitly model how exploration should be coordinated across different scales of reasoning. Instead, our work attempt to address exploration

Building on this insight, we propose DSDR, a Dual-Scale Diversity Regularization framework for RL-based LLM reasoning. DSDR integrates global diversity regularization over correct reasoning trajectories with a length-invariant, token-level entropy term applied exclusively to correct solutions. As illustrated in Figure 1, these two scales are coupled through a global-to-local allocation mechanism, which prioritizes local regularization for more distinctive correct trajectories. Additionally, this coupling prevents exploration from collapsing into narrow, locally suboptimal reasoning templates, instead promoting diverse trajectories that escape local optima and expand the correct solution space. We further provide theoretical support for DSDR, showing that bounded positive-only local entropy preserves optimal correctness, while correct-only global shaping prevents signal degeneracy in group-normalized optimization. We also justify the global-to-local softmax coupling from a principled objective view, explaining why dual-scale diversity strengthens learning signals. Extensive experiments across multiple reasoning benchmarks demonstrate that DSDR consistently improves accuracy, pass@k performance, and training sta-

###### Policy Update

[Figure 58]

r₁base r₂base r₃base

###### Correct-Only Global Diversity

z₁ z₂ ... z₁ z₁

d₁

d₂ d₃

[Figure 59]

₁

d₂ ... dᵐ ...

₁ dᵐ₂

⋅ λ +

₁

[Figure 60]

z₂

d₂

###### JGRPO Policy Loss

∑dᵢʲ =

J

i≠j

DSDR

[Figure 61]

rₙbase

dₘ₁ dₘ₂

zₘ

dₘ

...

[Figure 62]

Global-to-Local Coupling

###### Problem Input

[Figure 63]

Consider a particle starting at vertex (0,0,0) of a unit cube. At each step,

Local Pos-sample Entropy

×

the particle moves to a randomly

Regularization

chosen adjacent vertex with equal probability (1/3). What is the expected number of steps required for the particle to first reach the

JLocal

opposite vertex (1,1,1)?

Softmax

Mean Token-level Entropy

- Figure 2. DSDR training pipeline for dual-scale exploration in RL. Correct-only global diversity promotes exploration across solution modes, while a global-to-local coupling mechanism allocates length-invariant local entropy regularization to distinctive correct trajectories. Both signals are integrated into policy updates to enable deep exploration without sacrificing correctness.

limitations by structuring diversity directly within policy optimization across global-to-local scales, enabling deep exploration without modifying rollout procedures.

Diversity and Entropy Control for LLM Reasoning. Recent studies have explored promoting diversity in LLM reasoning by manipulating uncertainty at different levels of the policy. Token-level methods selectively encourage stochastic actions through entropy bonuses, clipping, or KL constraints (Cui et al., 2025; Liu et al., 2025a; Yu et al., 2025; Agarwal et al., 2025; Shen, 2025; Yao et al., 2025), which can alleviate premature collapse but primarily operate at a local action level. While effective in increasing shortterm randomness, these methods do not explicitly encourage diversity across complete reasoning trajectories. More recent approaches consider diversity at a global level. Chen et al. (2025c) and Walder & Karkhanis (2025) leverage pass@k as a training signal to encourage multiple candidate solutions, while, in a concurrent work, Cui et al. (2025) train a partitioning classifier to measure and amplify diversity in the advantage function. Closely related, some approaches (Chen et al., 2025b; Li et al., 2025; Hu et al., 2025) promotes global diversity among candidate solutions to improve deep exploration. However, above methods treat global and local diversity signals largely independently and do not specify how diversity at different scales should interact during optimization. In contrast, DSDR explicitly decomposes diversity into global and local components and couples them through a global-to-local allocation mechanism, which adaptively concentrates local entropy regularization on more distinctive correct reasoning trajectories.

### 3. Methodology

3.1. Preliminaries

We briefly review Group Relative Policy Optimization (GRPO) (Guo et al., 2025), which serves as the optimization backbone for reinforcement learning with verifiable rewards (RLVR) in reasoning tasks. Given an input prompt q, a policy πθ generates an output sequence o = (o1,...,oT) following autoregressive factorization

πθ(o | q) =

T

πθ(ot | q,o<t). (1)

t=1

A verifier provides a scalar reward r = R(q,o) for the completed sequence, which is typically binary for verifiable reasoning tasks. GRPO samples a group of G candidate outputs {oi}Gi=1 from a lagged behavior policy πθ

and

old

computes rewards {ri}Gi=1. To obtain a group-relative learning signal, rewards are normalized within each group to

form advantages. Specifically, the advantage Ai for each response is computed as

ri − mean(r1,r2,...,rG) std(r1,r2,...,rG)

, (2)

Ai =

where {ri}Gi=1 are rewards from the group and std(·) denotes the standard deviation with a small constant added for numerical stability. Optimization is performed using a PPO-style clipped surrogate objective at the token level. Let Ti denote the length of oi, and define the token-wise importance ratio ρi,t = πθ(oi,t | q,oi,<t)/πθ

##### (oi,t | q,oi,<t).

old

The GRPO objective is then given by

G

JGRPO(θ) = Eq G 1

i=1

Ti

1 Ti

min ρi,tAi,

t=1

clip(ρi,t,1 − ϵc,1 + ϵc)Ai − β DKL(πθ(·| q)∥πref(·| q)).

(3) where ϵc denotes the clipping threshold, β controls the KL regularization strength, and πref is a fixed reference policy. GRPO leverages relative comparisons within each group to stabilize optimization, but its learning signal critically depends on reward variation across sampled trajectories.

#### 3.2. DSDR: Dual-Scale Diversity Regularization

We adopt the group-based RLVR training protocol defined earlier: for each prompt q, we sample a group of G rollouts {oi}Gi=1 ∼ πθ

(· | q) and obtain verifiable rewards ri ∈ {0,1}. DSDR augments the backbone with two diversity regularizers that operate at different scales and are explicitly coupled, as shown in Figure 2. At the global (trajectory) scale, DSDR assigns extra credit to correct solutions that are more distinct within the group, which keeps the learning signal informative even when many rollouts are correct and prevents premature convergence to a single reasoning template. At the local (token) scale, DSDR encourages controlled entropy along positive trajectories to avoid the typical correct-mode collapse where the model becomes highly confident token-by-token and loses nearby correct variants. The coupling is key: global distinctiveness determines where local entropy should be strongest, so local regularization expands probability mass around unique correct paths rather than uniformly perturbing all positives.

old

- 3.2.1. GLOBAL-SCALE DIVERSITY SIGNALS

For each rollout oi in a group {o1,...,oG}, we compute a bounded per-response diversity score d(oi) ∈ [0,1]. The design goal is pragmatic: (i) it should reflect trajectory-level differences (not merely token noise), (ii) it should be cheap relative to rollout generation, and (iii) it should be wellscaled so it can be safely mixed into RLVR rewards without dominating correctness. We use two coupling signals.

Semantic Level. Let fϕ be a frozen text encoder that maps a full response oi to a vector zi ∈ Rd. We normalize embeddings so that cosine similarity becomes a stable inner product:

zi ∥zi∥2

. (4)

zi = fϕ(oi), z¯i =

Given two responses oi,oj, we define their embedding dissimilarity via cosine distance. We scale it into [0,1] to make it numerically comparable with other bounded components:

1 − z¯i⊤z¯j 2 ∈ [0,1]. (5)

d˜emb(oi,oj) =

The intuition is simple: if two responses encode similar reasoning semantics, their embeddings align and d˜emb is small; if they represent different solution directions, similarity drops and the distance increases. To turn pairwise distances into a per-response score, we use the group-average dissimilarity:

1 G − 1 j̸=i

d˜emb(oi,oj). (6)

Demb(oi) =

This aggregation matters for optimization stability. A single most different neighbor can be noisy; averaging across G−1 comparisons yields a smoother signal that is less sensitive to an outlier rollout. Computationally, Eq. (6) is efficient: with Z¯ = [¯z1,...,z¯G] ∈ RG×d, all pairwise similarities are obtained via a single matrix product Z¯Z¯⊤ followed by an elementwise transform.

Formula Level. Semantic similarity alone can miss an important axis of reasoning variation in math tasks: two solutions may appear similar at the surface level while relying on different symbolic manipulations, or vice versa. To capture this aspect, we introduce an Formula-level uniqueness signal, following the prior work (Wu et al., 2024; Chen et al., 2025b; Hu et al., 2025) while adopting a formulation aligned with our group-based setting. Let S(oi) denote the set of extracted mathematical expressions appearing in response oi. For each formula f ∈ S(oi), we define a binary indicator that measures whether f is unique relative to the rest of the group:

 f ∈/

 . (7)

Iuniq(f,oi) =

S(oj)

j̸=i

The equational diversity of response oi is then computed as the average uniqueness of its constituent formulas:

 

1 |S(oi)| f∈S(oi)

Iuniq(f,oi), |S(oi)| > 0, 0, otherwise.

Deq(oi) =



(8) This definition is intentionally conservative: responses that contain no detectable formulas contribute no equational novelty. When formulas are present, the averaging form encourages structural diversity in symbolic reasoning while remaining invariant to non-mathematical paraphrasing.

Combined Global Diversity. Both components are bounded in [0,1], so we combine them into a single global diversity score by simple averaging:

- 1

- 2

d(oi) =

Demb(oi) + Deq(oi) . (9)

This combination yields a bounded and well-scaled scalar signal for reward shaping across diverse reasoning tasks.

The embedding-based component captures trajectory-level semantic differences broadly. while the equation-based component provides a complementary, paraphrase-robust notion of novelty when symbolic manipulations are present.

Correct-Only Global Diversity Reward. We incorporate global-level diversity into RLVR only when it is consistent with the task objective. In particular, we avoid the failure case where responses are rewarded for being different despite incorrect reasoning. Accordingly, DSDR applies diversity shaping exclusively to positive rollouts and explicitly limits its influence. To prevent reward hacking (Pan et al., 2022) and avoid the diversity signal overpowering correctness, we apply a clipped technique (Sullivan et al., 2023; Li et al., 2023) only to correct rollouts and define the augmented reward as

##### r˜i = ri+λd d¯i·(r i = 1), d¯i = clip(d(oi); 0, σd). (10)

Where λd ≥ 0 controls the bonus strength and σd bounds the contribution of the diversity term. Equation (10) addresses a concrete optimization issue in group-relative methods: when many rollouts are correct, verifier rewards can become nearly constant within a group, shrinking reward variance and weakening within-group preference gradients. By introducing controlled dispersion among correct solutions, DSDR preserves a meaningful learning signal that differentiates alternative correct trajectories without creating incentives to explore incorrect ones. A formal statement and proof are provided in Appendix C.4.

- 3.2.2. GLOBAL-TO-LOCAL COUPLING OVER CORRECT TRAJECTORIES

Global diversity should not only determine which correct solutions are reinforced, but also where local entropy regularization is most effective. Intuitively, if a correct trajectory is already redundant within the group, expanding its neighborhood adds little coverage. In contrast, when a trajectory is globally distinctive, local expansion around it helps populate underexplored regions of the correct solution manifold. Let P = {i ∈ [G] | ri = 1} denote the set of correct rollouts. We allocate local regularization strength via a diversity-weighted softmax over correct responses:

 

exp(τ d¯i) j∈P exp(τ d¯j)

, i ∈ P, 0, i ∈/ P,

(11)

wi =



where τ > 0 is a temperature parameter. This construction defines a probability distribution over correct rollouts (i.e.,

i wi = 1 when P ̸= ∅). As τ increases, the allocation concentrates on the most globally distinctive correct solutions; as τ → 0, it reduces to uniform weighting across correct solutions. This allocation view unifies DSDR’s

dual-scale regularization: global diversity measures intertrajectory novelty, while local entropy concentrates exploration within trajectories where novelty is highest. Theoretically, this diversity-softmax coupling can be derived as the self-normalized policy-gradient weighting induced by a correct-only, diversity-tilted objective, as claimed in Theorem 3.1 and Appendix C.6. A further principled optimality analysis of this softmax allocation is given in Appendix C.5.

3.2.3. LOCAL POSITIVE-SAMPLE REGULARIZATION

A direct way to promote diversity is to encourage high entropy in the model’s output distribution. However, for long-form generation, response-level entropy is confounded by length: longer outputs naturally accumulate more tokenlevel uncertainty, so higher entropy may partially reflect more tokens. DSDR instead adopts token-level conditional entropy, averaged over timesteps (Agarwal et al., 2025; Cui et al., 2025), so that the objective measures per-step uncertainty rather than length accumulation. Let o = (o1,...,oT) ∼ πθ

(· | q). We start from the timeaveraged conditional entropy:

old

Jent(θ) = Eq, o∼π

θold(·|q)

T

1 T

H(πθ(· | q,o<t)) .

t=1

(12)

Using H(π) = −Ea∼π[log π(a)], each entropy term can be written as an expectation of log πθ(·). The remaining practical issue is that rollouts are sampled from πθ

, while the inner expectation is taken under πθ. We therefore reexpress the inner expectation using standard importance sampling (Precup et al., 2000; Sheng et al., 2025b; Yao et al., 2025), allowing it to be estimated from the observed tokens ot without resampling:

old

Ea∼π

θ(·|s)[g(a)] = Ea∼π

θold(·|s)

πθ(a | s) πθ

g(a) .

(a | s)

old

(13)

Applied at each timestep with s = (q,o<t) and g(a) = log πθ(a | s), this yields a tractable surrogate objective that is differentiable with respect to θ while reusing group-sampled rollouts. For each group rollout oi = (oi,1,...,oi,T

), we define the per-token importance ratio:

i

πθ(oi,t | q,oi,<t) πθ

. (14)

ρi,t =

(oi,t | q,oi,<t)

old

DSDR then defines the local objective as

Ti

G

1 Ti

Jlocal(θ) = E −

(r i = 1)wi

ρi,t g(oi,t) ,

t=1

i=1

(15) where g(a) = log πθ(a | q,o<t). The formulation reflects three deliberate choices. Time averaging removes incentives to increase length solely to amplify the regularizer,

yielding a per-decision signal. Restricting the objective to correct rollouts ensures that local entropy refines correct trajectories rather than encouraging noise on incorrect ones. The allocation weight wi couples local and global scales, so globally distinctive solutions receive stronger local regularization, focusing exploration on underrepresented reasoning modes. Furthermore, an information-theoretic decomposition and correctness-preservation guarantee under bounded local regularization are given in Appendix C.2.

- 3.2.4. DSDR OBJECTIVE

Let JGRPO(θ;r˜) denote the group-relative policy optimization objective defined earlier, computed using the augmented rewards r˜i from Eq. (10) when forming group advantages. DSDR optimizes

##### JDSDR(θ) = JGRPO(θ;r˜) + λℓ Jlocal(θ), (16)

where λℓ ≥ 0 controls the strength of local regularization. Eq. (16) makes the dual-scale structure explicit: the global term induces preferences among correct trajectories, while the local term mitigates token-level collapse around them, jointly enabling broad exploration with stable behavior within the correct set.

Theorem 3.1 (Diversity-tilted policy gradient induces DSDR global-to-local softmax coupling). Fix a prompt q. Let d¯(q,o) ∈ [0,σd] denote a bounded (clipped) global diversity score for a completed rollout o (e.g., Eq. (10)), and let R(q,o) ∈ {0,1} be the verifiable reward. For τ > 0, define the correct-only diversity-tilted objective

1 τ

log Zτ(θ;q), Zτ(θ;q) = Eo∼π

Jτ(θ;q) =

θ(·|q) exp τd¯(q,o) (R(q,o) = 1) .

(17) Assume Zτ(θ;q) > 0. Then the policy gradient of Jτ admits the form

θ(·|q) Aθτ(q,o) ∇θ log πθ(o | q) ,

∇θJτ(θ;q) = Eo∼π

(18) where the diversity-tilted advantage is

exp τd¯(q,o) (R(q,o) = 1) Zτ(θ;q) − 1 .

1 τ

Aθτ(q,o) =

(19)

Moreover, given i.i.d. rollouts {oi}Gi=1 ∼ πθ(· | q) with rewards ri = R(q,oi), the self-normalized Monte Carlo form of (18) assigns weights

exp(τd¯i) (r i = 1)

, (20)

wˆi =

G j=1 exp(τd¯j) (r j = 1)

which reduces to a diversity-softmax over correct rollouts: wˆi ∝ exp(τd¯i) on P = {i : ri = 1}, matching DSDR’s coupling rule in Eq. (11).

### 4. Experiments

#### 4.1. Experiment Settings

Backbone Models. For fair comparison, we conduct all experiments on the filtered DAPO-Math-17K (Hugging Face, 2025), which removes the duplicated samples. Training is performed on four base models with increasing capacity: Qwen2.5-Math-1.5B (Yang et al., 2024), Qwen3-1.7B and 4B (Yang et al., 2025). We adopt all-MiniLM-L6v2 (Reimers & Gurevych, 2019) as a lightweight sentence encoder for extracting response embeddings.

Training Setting. We use the same training configuration for all models. The batch size is 256, with 8 rollouts per prompt during policy optimization and a learning rate of 1e-6. The maximum response length is set to 4096 for Qwen2.5-Math-1.5B and 8192 for Qwen3-1.7B and Qwen34B in our experiment setting.

Evaluation Setting. Evaluation is conducted on a diverse set of mathematical reasoning benchmarks, including AIME2024 (Zhang & Math-AI, 2024) and AIME2025, MATH500 (Cobbe et al., 2021), Minerva Math (Lewkowycz et al., 2022), and Olympiad-level problems (He et al., 2024). For each benchmark, we report pass@1, computed from a single rollout per problem, and Avg@16, computed by averaging correctness over 16 independent rollouts. Avg@16 reflects the overall quality and stability of the model’s sampling distribution under a fixed sampling budget. In addition, we evaluate Pass@k for k ∈ {2,4,...,64} by sampling 64 independent rollouts per problem. We compare our method with GRPO (Shao et al., 2024), DAPO (Yu et al., 2025) and the corresponding base models.

#### 4.2. Main Results

Overall Performance. Our main experimental results are summarized in Table 1, covering five representative math reasoning benchmarks across three model scales. Overall, DSDR consistently outperforms all compared baselines, including Backbone, GRPO, and DAPO, demonstrating robust and scalable improvements in both Pass@1 and Avg@16 accuracy. On Qwen2.5-Math-1.5B, DSDR achieves the best average performance (25.4 / 25.6), with clear gains on challenging benchmarks such as AIME24, MATH500, and Minerva, where multiple valid reasoning paths exist. These improvements suggest that DSDR better preserves informative learning signals under group-relative optimization by differentiating correct trajectories, mitigating the reward-variance collapse that arises when many rollouts are correct. The advantage of DSDR becomes more pronounced as model scale increases. On Qwen3-1.7B and Qwen3-4B, DSDR achieves 36.8 / 36.8 and 48.0 / 46.8 average performance respectively, substantially outperforming both GRPO and DAPO at each scale, with especially large margins on AIME24 and AIME25. Importantly, DSDR consistently

Table 1. Results on different reasoning benchmarks. We report Pass@1 and Avg@16 (%) accuracy across different model scales. DSDR consistently outperforms Backbone, GRPO, and DAPO across most benchmarks and achieves the best average performance. Ablation results (w/o GD, w/o GC) show consistent performance drops, suggesting that both global diversity (GD) and global-to-local coupling (GC) regularization play important roles in DSDR.

Models AIME24 AIME25 MATH500 Minerva Olympiad Average Qwen2.5-math-1.5B, Max Response Length = 4K tokens

Backbone 10.0 / 3.0 3.3 / 3.0 27.4 / 21.0 5.9 / 3.9 13.1 / 9.9 11.9 / 8.2 GRPO 16.7 / 15.2 3.5 / 8.5 57.8 / 58.3 14.0 / 15.0 24.0 / 22.9 23.2 / 24.0 DAPO 13.3 / 14.4 6.7 / 6.3 54.8 / 55.3 15.8 / 15.2 22.7 / 21.4 22.7 / 22.5

DSDR 20.0 / 18.8 6.7 / 10.2 59.0 / 57.5 18.0 / 18.0 23.1 / 23.7 25.4 / 25.6 Qwen3-1.7B, Max Response Length = 8K tokens

Backbone 13.3 / 10.6 10.0 / 7.3 57.6 / 56.5 17.3 / 17.5 23.7 / 25.3 24.4 / 23.4 GRPO 16.7 / 21.0 20.0 / 15.8 57.4 / 60.7 18.4 / 19.9 29.5 / 31.2 28.4 / 29.7 DAPO 20.0 / 20.4 10.0 / 21.0 63.8 / 64.0 21.3 / 22.3 32.9 / 32.9 29.6 / 32.1

DSDR 36.7 / 32.1 23.2 / 27.3 64.4 / 65.4 23.3 / 23.4 36.4 / 36.0 36.8 / 36.8 w/o GD 23.3 / 20.6 13.3/ 19.0 61.6 / 63.5 20.6 / 21.4 29.8 / 32.5 29.7 / 31.4 w/o GC 30.0 / 26.5 23.3 / 24.0 63.0 / 64.6 23.5 / 23.4 35.9/ 36.1 35.1 / 34.9

Qwen3-4B, Max Response Length = 8K tokens

Backbone 13.33 / 20.63 16.67 / 16.04 64.8 / 65.75 26.10 / 24.61 32.79 / 32.69 30.74 / 31.94 GRPO 36.67 / 37.50 33.33 / 35.21 64.0 / 63.83 25.00 / 23.62 37.54 / 37.70 39.31 / 39.57 DAPO 33.33 / 38.38 40.00 / 33.13 59.0 / 62.74 25.74 / 27.37 36.65 / 39.42 38.94 / 40.21

DSDR 56.67 / 52.08 50.00 / 46.46 66.2 / 66.18 26.84 / 27.87 40.36 / 42.43 48.01 / 46.80 w/o GD 46.67 / 45.00 43.33 / 40.42 63.6 / 64.74 26.47 / 27.44 40.65 / 41.20 44.14 / 43.76 w/o GC 26.67 / 44.38 23.33 / 35.63 58.0 / 62.78 20.21 / 24.59 33.98 / 40.60 32.04 / 41.20

improves Avg@16 alongside Pass@1, indicating that the gains are not driven by occasional lucky samples but by systematically expanding the diversity of correct reasoning trajectories. Across scales, by promoting exploration exclusively within the correct solution space, DSDR enables more effective and stable exploration in RL-based LLM reasoning, leading to consistent and scalable performance gains.

Performance on Pass@k Evaluation. Figure 3 presents the pass@k performance for k ∈ {2,4,...,64} across our method, DAPO and the base model Qwen3-1.7B, qwen3-4B on five benchmarks. Overall, DSDR consistently outperforms the base model and DAPO across a wide range of k, with the most pronounced gains observed on AIME2024, AIME2025, and Olympiad. On these benchmarks, the performance gap between DSDR and the baselines continues to widen as k increases, indicating that DSDR effectively expands the set of correct reasoning trajectories rather than merely sharpening a single dominant solution. On Minerva, where baseline pass@k values are relatively low and correct solutions are sparse, the absolute gains are more modest; nevertheless, DSDR maintains a consistent advantage over both the base model and DAPO across k, suggesting improved exploration even in low-reward regimes. On MATH500, where baseline accuracy is already high, DSDR delivers stable improvements across all k without satura-

tion or degradation at large k. Importantly, DSDR does not exhibit performance drop-offs at high k, highlighting its ability to promote exploration within the correct solution space rather than drifting toward noisy or incorrect samples. These results demonstrate that DSDR yields more reliable and scalable improvements in pass@k.

#### 4.3. Ablation Study

We conduct ablation studies on Qwen3-1.7B and 4B. Table 1 reports Avg@16 and Pass@1 for variants without global diversity (GD) and without global-to-local coupling (GC). Removing global diversity causes a clear drop in average performance across benchmarks and model sizes (1.7B and 4B), indicating that trajectory-level diversity is necessary to preserve informative learning signals when many rollouts are correct and verifier rewards saturate. Removing globalto-local coupling also degrades performance, with larger drops on AIME and Olympiad, where discovering multiple valid reasoning strategies is critical. This shows that local entropy alone is insufficient; instead, local regularization must be guided by global distinctiveness to expand underexplored correct trajectories. Overall, the ablations confirm that GD and GC are complementary: GD differentiates correct solutions at the trajectory level, while GC focuses local exploration, jointly enabling the consistent gains of DSDR.

Base DAPO DSDR

###### Olympiad

###### MINERVA

###### AIME2024

###### AIME2025

###### MATH500

0.6

0.775

0.6

0.45

0.35

0.750

0.5

0.5

Qwen3-1.7B

0.725

Pass@k

0.40

0.30

0.4

0.4

0.700

0.3

0.3

0.675

0.25

0.35

0.2

0.650

0.2

0.1

0.20

0.625

0.30

21 22 23 24 25 26

21 22 23 24 25 26

21 22 23 24 25 26

21 22 23 24 25 26

21 22 23 24 25 26

0.500

0.38

0.8

0.7

0.775

0.475

0.36

0.7

0.750

0.6

0.450

0.34

0.6

Qwen3-4B

Pass@k

0.725

0.5

0.32

0.425

0.5

0.700

0.4

0.30

0.4

0.400

0.675

0.3

0.28

0.3

0.375

0.650

21 22 23 24 25 26

21 22 23 24 25 26

21 22 23 24 25 26

21 22 23 24 25 26

21 22 23 24 25 26

- Figure 3. Pass@k performance across five benchmarks for both Qwen3-1.7B and Qwen3-4B. The Base models serve as backbones. DSDR consistently outperforms both the Base models and DAPO across all values of k..

0 20 40 60 80 100

Steps

0.12

0.15

0.18

0.20

0.23

0.25

0.28

0.30

Avg@16

AIME2024 Avg@16

0 20 40 60 80 100

Steps

0.00

1.00

2.00

3.00

4.00

5.00

6.00

7.00

Entropy

Entropy

0 20 40 60 80 100

Steps

0.398

0.400

0.402

0.404

0.406

0.408

0.410

SemanticSimilarity

Semantic Level

0 20 40 60 80 100

Steps

0.70

0.72

0.74

0.76

0.78

FormulaSimilarity

Formula Level

50 60 70 80

GRPO DSDR DSDR w/o GD DSDR w/o GC DAPO

- Figure 4. Training dynamics across methods conducted on Qwen3-1.7 model. From left to right, we report AIME2024 Avg@16, policy entropy, semantic-level diversity similarity, and formula-level diversity similarity. Results are shown for GRPO, DSDR, DSDR w/o GD, DSDR w/o GC, and DAPO.

| |3.35 3.35<br><br>| |
|---|
<br><br>| |
|---|
| | | | | |
|---|---|---|---|---|---|---|
| |2.83<br><br>3.28| | | | | |
| |2.53<br><br>2.28 2.25<br><br>2.42| | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| || |
|---|
| | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| || |
|---|
| | | | | |
| | | | | | | |
| | | | | | | |

Minverva Olympiadbench AIME2025 AIME2024

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

Diversity(010)

DAPO Diversity DSDR Diversity DAPO pass@32 DSDR pass@32

0.35

0.40

0.45

0.50

0.55

0.60

pass@32

- Figure 5. We generate 32 test-time rollouts per problem on four benchmarks and evaluate response diversity using an LLM-asa-Judge (1–10 scale). The figure reports diversity scores and corresponding pass@32 for DAPO and DSDR.

AIME2024, indicating that DSDR improves performance while simultaneously enhancing exploration, preventing the policy from collapsing into a single dominant reasoning pattern. The entropy dynamics further highlight the role of DSDR’s dual-scale design: DSDR w/o GD exhibits a rapid and excessive increase in entropy, reflecting uncontrolled random exploration without global diversity guidance, while DSDR w/o GC, which removes token-level entropy regularization, shows diminishing exploration capacity in later stages as the policy becomes overly concentrated. In contrast, GRPO and DAPO maintain relatively low and flat entropy, suggesting limited exploration. By combining correct-only global diversity (GD) with globalto-local coupling (GC), DSDR achieves a balanced entropy profile that increases exploration without instability. This effect is further reflected in the semantic- and formula-level similarity curves, where DSDR maintains lower semantic similarity and sustained symbolic diversity among rollouts

#### 4.4. Training Dynamics Analysis

- Figure 4 shows the training dynamics of DSDR vs. other methods. As training progresses, DSDR achieves consistently higher Avg@16 than GRPO and DAPO on

Test Scores (DSDR w/o GD)

Test Scores (DSDR w/o GC)

0.30

d = 0.001(AIME25)

λd = 0.001 as the default setting. Overall, DSDR remains stable within a reasonable regularization range.

0.25

d = 0.01(AIME25)

| |
|---|

d = 0.1(AIME25)

0.25

d = 0.001(AIME24)

0.20

| |
|---|

d = 0.01(AIME24)

| |
|---|

d = 0.1(AIME24)

| |
|---|

| |
|---|

0.15

Avg@16

Avg@16

0.20

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- = 0.001(AIME25)

- = 0.002(AIME25)

0.10

| |
|---|

| |
|---|

| |
|---|

0.15

| |
|---|

### 5. Conclusion

= 0.01(AIME25)

| |
|---|

0.05

- = 0.001(AIME24)

- = 0.002(AIME24)

| |
|---|

0.10

| |
|---|

= 0.01(AIME24)

0.00

In this paper, we introduced DSDR, a correctness-aligned dual-scale diversity regularization framework for RLVR that improves exploration in LLM reasoning. DSDR promotes global diversity among correct trajectories to sustain multiple solution modes, and applies a local, length-invariant token-level entropy regularizer exclusively to correct trajectories to prevent intra-mode entropy collapse. A global-tolocal allocation mechanism tightly couples the two scales, focusing local regularization on globally distinctive correct trajectories. Our analysis shows that bounded local regularization preserves correctness while correct-only global shaping maintains informative learning signals under groupbased optimization. Experiments across diverse reasoning benchmarks demonstrate consistent gains in accuracy, pass@k, and training stability, underscoring the importance of coordinating trajectory- and token-level exploration in RLVR for stable and robust policy optimization.

0 20 40 60 80 100 120 Steps

0 20 40 60 80 100 Steps

- Figure 6. Hyperparameter sensitivity of DSDR on Qwen3-1.7B. Left: varying λℓ shows that overly large entropy regularization destabilizes training. Right: λd = 0.001 achieves the best and most stable Avg@16 performance on AIME2024/2025.

throughout training, indicating that the model continues to explore multiple distinct reasoning trajectories while preserving correctness. Together, these dynamics demonstrate that DSDR enables stable and targeted exploration, preventing both random drift and premature mode collapse.

#### 4.5. Diversity Analysis

- Figure 5 compares response diversity and pass@32 across four benchmarks using 32 test-time rollouts. We use GPT5.2 (Singh et al., 2025) to evaluate the diversity, which aggregates semantic, logical, and formula-level differences, scored on a 1–10 scale. The diversity judge prompt is provided in Appendix B.3. As shown in the figure, DSDR consistently produces higher diversity scores than DAPO across all datasets, indicating that its generated responses cover a broader range of reasoning strategies rather than collapsing into similar solution patterns. Importantly, these diversity gains are accompanied by higher pass@32 performance, demonstrating that increased diversity does not come at the cost of correctness. It indicates that by applying correct-only global diversity regularization, DSDR encourages distinct correct trajectories, while the globalto-local coupling further expands locally around the most distinctive solutions instead of injecting uniform randomness. As a result, DSDR is able to improve exploration as well as performance, yielding both higher-quality diversity and stronger pass@k performance compared to DAPO.

### 6. Impact Statements

This work introduces DSDR, a dual-scale diversity regularization framework for reinforcement learning with verifiable rewards (RLVR) in large language model (LLM) reasoning. By promoting correctness-aligned exploration at both trajectory and token levels, DSDR aims to improve reasoning robustness, stability, and sample efficiency. The proposed approach is intended to support the development of more reliable reasoning-oriented LLMs, with potential benefits for applications that require multi-step decision making and formal reasoning.

### References

Agarwal, S., Zhang, Z., Yuan, L., Han, J., and Peng, H. The unreasonable effectiveness of entropy minimization in llm reasoning. arXiv preprint arXiv:2505.15134, 2025.

#### 4.6. Hyperparameter Sensitivity of λℓ and λd

Boyd, S. and Vandenberghe, L. Convex optimization. Cambridge university press, 2004.

We study the sensitivity of DSDR to the local coefficient λℓ and global diversity factor λd on Qwen3-1.7B (Figure 6). For λℓ ∈ {0.001,0.002,0.01}, moderate regularization improves performance, while values larger than 0.01 cause training instability and collapse, indicating that excessive entropy-driven exploration disrupts correctness-aligned optimization. λℓ = 0.001 yields the most stable and consistently strong results and is used in our main experiments. For λd, we observe that 0.001 achieves the best average performance on AIME2024 and AIME2025 with stable training dynamics, whereas larger values introduce additional variance without consistent gains. We therefore adopt

Chen, L., Han, X., Wang, Q., Han, B., Bai, J., Schutze, H., and Wong, K.-F. Eepo: Exploration-enhanced policy optimization via sample-then-forget, 2025a. URL https://arxiv.org/abs/2510.05837.

- Chen, Y., Chakraborty, S., Wolf, L., Paschalidis, Y., and Pacchiano, A. Post-training large language models for diverse high-quality responses. arXiv preprint arXiv:2509.04784, 2025b.
- Chen, Z., Qin, X., Wu, Y., Ling, Y., Ye, Q., Zhao, W. X.,

and Shi, G. Pass@ k training for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751, 2025c.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Cover, T. M. Elements of information theory. John Wiley & Sons, 1999.

Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Gandhi, K., Chakravarthy, A., Singh, A., Lile, N., and Goodman, N. D. Cognitive behaviors that enable selfimproving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., et al. Deepseekr1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al. Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

Hou, Z., Lv, X., Lu, R., Zhang, J., Li, Y., Yao, Z., Li, J., Tang, J., and Dong, Y. Advancing language model reasoning through reinforcement learning and inference scaling. arXiv preprint arXiv:2501.11651, 2025.

Hu, Z., Zhang, S., Li, Y., Yan, J., Hu, X., Cui, L., Qu, X., Chen, C., Cheng, Y., and Wang, Z. Diversityincentivized exploration for versatile reasoning. arXiv preprint arXiv:2509.26209, 2025.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https://github. com/huggingface/open-r1.

Jaynes, E. T. Information theory and statistical mechanics. Physical review, 106(4):620, 1957.

Jiang, Y., Huang, J., Yuan, Y., Mao, X., Yue, Y., Zhao, Q., and Yan, L. Risk-sensitive rl for alleviating exploration dilemmas in large language models. arXiv preprint arXiv:2509.24261, 2025.

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T., et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.

Li, M., Zhao, X., Lee, J. H., Weber, C., and Wermter, S. Internally rewarded reinforcement learning. In International Conference on Machine Learning, pp. 20556– 20574. PMLR, 2023.

Li, T., Zhang, Y., Yu, P., Saha, S., Khashabi, D., Weston, J., Lanchantin, J., and Wang, T. Jointly reinforcing diversity and quality in language model generations, 2025. URL https://arxiv.org/abs/2509.02534.

Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Liu, J., He, C., Lin, Y., Yang, M., Shen, F., and Liu, S. Ettrl: Balancing exploration and exploitation in llm test-time reinforcement learning via entropy mechanism. arXiv preprint arXiv:2508.11356, 2025a.

Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., and Dong, Y. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025b.

Pan, A., Bhatia, K., and Steinhardt, J. The effects of reward misspecification: Mapping and mitigating misaligned models. arXiv preprint arXiv:2201.03544, 2022.

Precup, D., Sutton, R. S., and Singh, S. Eligibility traces for off-policy policy evaluation. 2000.

Reimers, N. and Gurevych, I. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shen, H. On entropy control in llm-rl algorithms. arXiv preprint arXiv:2509.03493, 2025.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279– 1297, 2025a.

Sheng, Y., Huang, Y., Liu, S., Zhang, H., and Zeng, A. Espo: Entropy importance sampling policy optimization. arXiv preprint arXiv:2512.00499, 2025b.

Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., El-Kishky, A., McLaughlin, A., Low, A., Ostrow, A., Ananthram, A., et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Sullivan, R., Kumar, A., Huang, S., Dickerson, J., and Suarez, J. Reward scale robustness for proximal policy optimization via dreamerv3 tricks. Advances in Neural Information Processing Systems, 36:1352–1362, 2023.

Yao, J., Cheng, R., Wu, X., Wu, J., and Tan, K. C. Diversityaware policy optimization for large language model reasoning. arXiv preprint arXiv:2505.23433, 2025.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint

- arXiv:2503.14476, 2025.

Zhang, Q., Wu, H., Zhang, C., Zhao, P., and Bian, Y. Right question is already half the answer: Fully unsupervised llm reasoning incentivization. arXiv preprint

- arXiv:2504.05812, 2025.

Zhang, Y. and Math-AI, T. American invitational mathematics examination (aime) 2024, 2024.

Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.

Walder, C. and Karkhanis, D. Pass@ k policy optimization: Solving harder reinforcement learning problems. arXiv preprint arXiv:2505.15201, 2025.

Wan, Z., Dou, Z., Liu, C., Zhang, Y., Cui, D., Zhao, Q., Shen, H., Xiong, J., Xin, Y., Jiang, Y., et al. Srpo: Enhancing multimodal llm reasoning via reflection-aware reinforcement learning. arXiv preprint arXiv:2506.01713, 2025.

Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

Wu, F., Xuan, W., Lu, X., Liu, M., Dong, Y., Harchaoui, Z., and Choi, Y. The invisible leash: Why rlvr may or may not escape its origin. arXiv preprint arXiv:2507.14843, 2025.

Wu, T., Li, X., and Liu, P. Progress or regress? selfimprovement reversal in post-training. arXiv preprint arXiv:2407.05013, 2024.

Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., et al. Qwen2. 5-math technical report: Toward mathematical expert model via selfimprovement. arXiv preprint arXiv:2409.12122, 2024.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

### A. Implementation Details

We provide more details for experiments in Section 4. We provide additional experimental details in Section 4. All models are trained using the VERL framework (Sheng et al., 2025a) and deployed on 8× NVIDIA A100 GPUs (40GB). Table 2 and Table 3 summarize the training and evaluation hyperparameters. Unless otherwise specified, we adopt a rollout group size of n = 8, a learning rate of 1 × 10−6, and binary verifier rewards. We conduct hyperparameter sweeps over the global diversity scaling factor λd ∈ {0.001,0.01,0.1}, the local regularization coefficient λℓ ∈ {0.001,0.002,0.01}, and the coupling temperature τ ∈ {1,5,10}. Empirically, we find that λd = 0.001, λℓ = 0.001, and τ = 5 consistently yield the best or near-best performance across benchmarks; these values are therefore used as defaults in all reported experiments.

Table 2. Summary of training details.

Training Settings Hardware 8×A100 GPUs (40GB) Base models Qwen2.5-Math-1.5B / Qwen3-1.7B, 4B Training dataset open-r1/DAPO-Math-17K-Processed Max response length 4096 / 8192 Batch / mini-batch size 256 / 16 Rollout group size n 8 Learning rate 1 × 10−6 Temperature (training) 1.0 Clip range (ϵlow,ϵhigh) (0.2, 0.28) / (0.2, 0.2) for GRPO only Reward type Binary reward Coupling temperature parameter τ {1,5,10} Reward scaling factor λd {0.001,0.01,0.1} Regularization coefficient λℓ {0.001,0.002,0.01}

Table 3. Evaluation settings.

Evaluation settings

Max response length 4096 / 8192 Top-p (eval) 0.9 Temperature (eval) 0.1 for Pass@1; 0.7 for Avg@16

### B. Additional Results

#### B.1. Additional Training Dynamics Analysis

- As shown in figure 7, DSDR steadily improves both Avg@16 and Pass@16 over training, indicating that performance gains are accompanied by sustained exploration rather than convergence to a narrow solution mode. In contrast, GRPO and DAPO display slower growth and earlier saturation, suggesting limited ability to expand the set of correct solutions. And removing global diversity (w/o GD) results in large fluctuations in the policy-gradient loss, reflecting unstable learning when trajectory-level differentiation among correct rollouts is absent. on the other hand, without global-to-local coupling (w/o GC), leads to weaker late-stage improvements and a sharp increase in the clip ratio, indicating overly aggressive updates and reduced robustness as token-level exploration diminishes. By jointly enforcing correct-only global diversity and diversity-guided local regularization, DSDR maintains controlled updates and effective exploration throughout training, which translates into more reliable and higher final performance.

#### B.2. Case Study

In this section, we present generated samples during testing. One sample with 16 test-time rollouts, DSDR achieves 7 correct solutions, compared to only 2 for DAPO. We show two example responses generated by DSDR, which arrive at the correct answer through distinct reasoning processes, illustrating its ability to explore multiple valid solution paths. In contrast, we present two responses generated by DAPO, which exhibit more limited diversity. This demonstrates that DSDR

GRPO DSDR DSDR w/o GD DSDR w/o GC DAPO

AIME2025 Avg@16

AIME2025 Pass@16

PG Loss

response_length/clip_ratio

0.300

0.50

0.25

0.50

0.250

0.23

0.40

0.45

0.20

0.200

0.30

0.40

ClipRatio

Pass@16

Avg@16

PGLoss

0.17

0.150

0.35

0.20

0.15

0.30

0.100

0.10

0.12

0.25

0.050

0.10

0.00

0.20

0.000

0.08

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

Steps

Steps

Steps

Steps

- Figure 7. Training dynamics across methods conducted on Qwen3-1.7B model. From left to right, we show Avg@16 and Pass@16 for AIME2025, PG loss, and response length/clip ratio, comparing GRPO, DSDR, DSDR w/o GD, DSDR w/o GC, and DAPO.

improves exploration without sacrificing solution fidelity: by maintaining controlled diversity across rollouts, the model avoids mode collapse while consistently discovering correct reasoning paths, leading to stronger overall performance.

- B.2.1. SAMPLES GENERATED BY DSDR

Two generated samples by DSDR are shown below, and the yellow boxes highlight the different solution strategies used to solve this problem.

- Generated Sample: Answer 1

Problem.

Let N be the greatest four-digit positive integer with the property that whenever one of its digits is changed to 1, the resulting number is divisible by 7. Let Q and R be the quotient and remainder, respectively, when N is divided by 1000. Find Q + R. Ground Truth:

|699|
|---|

#### Answer.

We are given a four-digit number N such that whenever one of its digits is changed to 1, the resulting number is divisible by 7. We are to find Q + R, where N = 1000Q + R, and Q and R are the quotient and remainder when N is divided by 1000.

- STEP 1: UNDERSTAND THE CONDITION Let’s consider all possible changes to the digits of N, and the resulting number must be divisible by 7. Let N = ABCD, where A,B,C,D are digits (each from 0 to 9, with A ̸= 0). We must ensure that for every digit position (i.e., for each digit A,B,C,D), changing that digit to 1 results in a number divisible by 7. This means:

- Changing the first digit A to 1: 1BCD must be divisible by 7. - Changing the second digit B to 1: AB1D must be divisible by 7. - Changing the third digit C to 1: ABD1 must be divisible by 7. - Changing the fourth digit D to 1: ABC1 must be divisible by 7.

So we have four divisibility conditions:

1BCD ≡ 0, AB1D ≡ 0, ABD1 ≡ 0, ABC1 ≡ 0 (mod 7).

- STEP 2: REPRESENT THE NUMBERS ALGEBRAICALLY Let N = 1000A + 100B + 10C + D.

The four conditions become:

1000 + 100B + 10C + D ≡ 0 (mod 7), 1000A + 100 + 10C + D ≡ 0 (mod 7), 1000A + 100B + 10 + D ≡ 0 (mod 7),

1000A + 100B + 10C + 1 ≡ 0 (mod 7).

- STEP 3: USE MODULAR ARITHMETIC TO SIMPLIFY Using

1000 ≡ 6, 100 ≡ 2, 10 ≡ 3 (mod 7),

we obtain four congruences, which are then manipulated to derive relations among A,B,C,D.

- STEP 4: SOLVING RELATIONS Through successive subtraction and modular inversion, we find:

##### A = 5, B = 6.

Solving the remaining congruences yields two admissible digit pairs:

##### (C,D) ∈ {(2,4),(9,4)}.

Each candidate is then explicitly checked by substituting back into all four digit-replacement conditions to verify divisibility by 7.

Both 5624 and 5694 satisfy the conditions. Since the problem asks for the greatest such number, we select

N = 5694. Thus,

|699|
|---|

Q = 5, R = 694, Q + R =

. FINAL ANSWER

|699|
|---|

- Generated Sample: Answer 2

Problem.

Let N be the greatest four-digit positive integer with the property that whenever one of its digits is changed to 1, the resulting number is divisible by 7. Let Q and R be the quotient and remainder, respectively, when N is divided by 1000. Find Q + R. Ground Truth:

|699|
|---|

Answer. We are given a four-digit number N, such that whenever one of its digits is changed to 1, the resulting number is divisible by 7. We are to find Q + R, where Q is the quotient and R is the remainder when N is divided by 1000.

- STEP 1: UNDERSTANDING THE PROBLEM We need to find the greatest four-digit number N such that any single digit change to 1 results in a number divisible by 7. Let N = ABCD, where A,B,C,D are digits with A ̸= 0. For each digit position, changing that digit to 1 must yield a number divisible by 7.

- STEP 2: FORMULATING CONDITIONS

- Case 1: Change the thousands digit A to 1: 1000 + BCD ≡ 0 (mod 7) ⇒ BCD ≡ 1 (mod 7).
- Case 2: Change the hundreds digit B to 1: 1000A + 100 + 10C + D ≡ 0 (mod 7) ⇒ 6A + 3C + D ≡ 5 (mod 7).
- Case 3: Change the tens digit C to 1: 1000A + 100B + 10 + D ≡ 0 (mod 7) ⇒ 6A + 2B + D ≡ 4 (mod 7).
- Case 4: Change the units digit D to 1: 1000A + 100B + 10C + 1 ≡ 0 (mod 7) ⇒ 6A + 2B + 3C ≡ 6 (mod 7).

- STEP 3: SUMMARIZE THE CONDITIONS Using 100 ≡ 2, 10 ≡ 3 (mod 7), we obtain:

2B + 3C + D ≡ 1 (mod 7).

Together with the other three congruences, we form a system in A,B,C,D.

- STEP 4: SOLVE THE SYSTEM OF CONGRUENCES By subtracting equations and eliminating variables, we obtain:

3A − B ≡ 2 (mod 7), 3C − D ≡ 2 (mod 7), 2B − 3C ≡ 6 (mod 7).

- STEP 5: EXPRESS VARIABLES Solving step-by-step yields:

A = 5, B = 6, and

C ∈ {2,9}, D = 4. This branching step yields two admissible candidates:

##### N ∈ {5624, 5694}.

- STEP 8: VERIFICATION All four digit-replacement conditions are explicitly checked and satisfied for

N = 5694.

- STEP 9: COMPUTE Q + R Since 5694 = 1000 · 5 + 694, we have:

|699|
|---|

. FINAL ANSWER

Q = 5, R = 694, Q + R =

|699|
|---|

- B.2.2. SAMPLES GENERATED BY DAPO

Two samples generated by DAPO are shown below, and the red boxes highlight the erroneous solution strategies that lead to failure.

- Generated Sample: Answer 1

Problem.

Let N be the greatest four-digit positive integer with the property that whenever one of its digits is changed to 1, the resulting number is divisible by 7. Let Q and R be the quotient and remainder, respectively, when N is divided by 1000. Find Q + R. Ground Truth:

|699|
|---|

Answer. We are given a four-digit number N such that any change of one digit to 1 results in a number divisible by 7. Let’s denote N as a four-digit number:

N = 1000a + 100b + 10c + d Where:

- • a,b,c,d are digits (integers from 0 to 9, with a ̸= 0 since N is a four-digit number),
- • a ∈ {1,2,...,9}, b,c,d ∈ {0,1,...,9}.

STEP 1: UNDERSTANDING THE CONDITION The problem says: Whenever one digit is changed to 1, the resulting number is divisible by 7. So, there are four different cases:

- 1. Change a to 1: 1000 + 100b + 10c + d
- 2. Change b to 1: 1000a + 100 + 10c + d
- 3. Change c to 1: 1000a + 100b + 10 + d
- 4. Change d to 1: 1000a + 100b + 10c + 1

Each must be divisible by 7, giving:

1000 + 100b + 10c + d ≡ 0 (mod 7), 1000a + 100 + 10c + d ≡ 0 (mod 7), 1000a + 100b + 10 + d ≡ 0 (mod 7),

1000a + 100b + 10c + 1 ≡ 0 (mod 7). Using

1000 ≡ 6, 100 ≡ 2, 10 ≡ 3 (mod 7), we obtain:

2b + 3c + d ≡ 1 (mod 7) (1) 6a + 3c + d ≡ 5 (mod 7) (2) 6a + 2b + d ≡ 4 (mod 7) (3)

6a + 2b + 3c ≡ 6 (mod 7) (4) We now attempt to solve this system of congruences.

Logical error starts here (unsupported assumption). Assume a = 1 (the smallest possible digit, since N is a four-digit number), then try values of b,c,d that satisfy the congruences.

From this point onward, the reasoning proceeds by ad-hoc guessing and trial-based arguments that are not logically implied by the modular system above. Eventually, the solution concludes:

|Q = 111, R = 1, Q + R = 112|
|---|

FINAL ANSWER

|112|
|---|

- Generated Sample: Answer 2

Problem.

Let N be the greatest four-digit positive integer with the property that whenever one of its digits is changed to 1, the resulting number is divisible by 7. Let Q and R be the quotient and remainder, respectively, when N is divided by 1000. Find Q + R. Ground Truth:

|699|
|---|

Answer. We are asked to find the greatest four-digit number N such that if any one of its digits is changed to 1, the resulting number is divisible by 7. Then, given N = 1000Q + R, we are to find Q + R.

STEP 1: UNDERSTAND THE CONDITION Let’s denote N as a four-digit number:

##### N = 1000a + 100b + 10c + d

where a,b,c,d ∈ {0,1,...,9}, with a ̸= 0. Changing each digit to 1 gives the conditions:

1000 + 100b + 10c + d ≡ 0 (mod 7), 1000a + 100 + 10c + d ≡ 0 (mod 7), 1000a + 100b + 10 + d ≡ 0 (mod 7),

1000a + 100b + 10c + 1 ≡ 0 (mod 7). Using

1000 ≡ 6, 100 ≡ 2, 10 ≡ 3 (mod 7), we obtain:

2b + 3c + d ≡ 1 (mod 7), 6a + 3c + d ≡ 5 (mod 7), 6a + 2b + d ≡ 4 (mod 7),

6a + 2b + 3c ≡ 6 (mod 7).

Logical error starts here (invalid constraint assumption). Let’s suppose that all four expressions are congruent to the same value x (mod 7), and in particular, assume

x = 0 (i.e., N ≡ 0 (mod 7)).

From this point onward, the reasoning proceeds by enumerating digit values under the unsupported assumption that all four congruences must equal the same residue class, which is not implied by the original system. The solution eventually concludes:

##### Q = 9, R = 888, Q + R = 897.

FINAL ANSWER

|897|
|---|

#### B.3. Prompt Template Prompt Template for Diversity Scoring

System: You are a strict but fair grader. You will be given one math problem and 32 independent rollouts (solutions). Your task is to score diversity across rollouts.

Evaluate the following three dimensions on a 0–10 scale:

- • logic diversity: differences in reasoning flow or step order

- • formula diversity: differences in formulas, identities, or techniques used

- • semantic diversity: differences in overall semantic approach or framing

Then output overall diversity as the average of the three scores (0–10, rounded to one decimal). Be objective and do not prefer any model. Output JSON only.

### C. Theoretical Analysis of DSDR

This appendix part provides formal justification for the key design choices in DSDR (Sec. 3.2): (i) the correct-only global diversity reward (Eq. (10)), (ii) the global-to-local coupling weights (Eq. (11)), and (iii) the positive-only, length-invariant token-level entropy regularizer (Eq. (15)).

#### C.1. Setup, Notation, and Standing Assumptions

We consider RLVR with verifiable reward on prompts q ∼ D. A policy π(· | q) induces an output random variable O (a sequence of tokens) and a verifier reward R(q,O) ∈ {0,1}. For group-based training (GRPO-style), for each prompt q we sample a group {oi}Gi=1 and obtain rewards {ri}Gi=1 with ri = R(q,oi).

Augmented Reward (Correct-Only Global Diversity Reward) For each rollout oi, we compute a diversity score d(oi) and use a clipped score

d¯i = clip d(oi);0,σd ∈ [0,σd]. (21) The augmented reward used for group advantage computation is

r˜i = ri + λd d¯i (r i = 1), λd ≥ 0. (22)

Group Normalization Define the empirical mean and standard deviation of {r˜i}Gi=1:

G

G

1 G

1 G

(˜ri − µr˜)2. (23)

r˜i, σr˜ =

µr˜ =

i=1

i=1

In practice we use a stabilized standard deviation σr,ε˜ > 0, e.g., σr,ε˜ = σr˜ + ε or σr,ε˜ = σr2˜ + ε2 with ε > 0. The normalized advantages are

r˜i − µr˜ σr,ε˜

A˜i =

. (24) All results below remain valid under either stabilization choice as long as ε > 0.

Entropy Conventions. The action space at each decoding step is the vocabulary V, assumed finite. All logarithms are natural logs. Standard entropy identities and bounds follow classical information theory (Cover, 1999).

Correctness Gap Assumption For Proposition C.2, we assume a strict gap

##### ∆ := JR⋆ − JR > 0, (25)

where JR⋆ is optimal correctness and JR is the best correctness strictly below optimal. This separation commonly holds when the set of achievable correctness values is discrete under a restricted policy class.

#### C.2. Inter-/Intra-mode Decomposition (Formal Complementarity)

DSDR introduces two diversity mechanisms operating at different scales: a global (sequence-level) preference among correct trajectories (Eq. (10)) and a local (token-level) positive-only entropy regularizer (Eq. (15)). This subsection provides an information-theoretic lens clarifying why these two terms are complementary rather than redundant: global shaping targets inter-mode coverage (deep exploration across distinct reasoning modes), while local entropy targets intra-mode thickening (maintaining non-collapsed variability within a mode).

We view generation as a mixture over latent “reasoning modes” Z (e.g., distinct solution strategies). For each prompt q, suppose

Z ∼ p(z | q), O ∼ p(o | z,q). (26) Lemma C.1 (Inter-/intra-mode entropy decomposition). For any fixed prompt q,

H(O | q) = I(O;Z | q) + H(O | Z,q), (27) where I(O;Z | q) is conditional mutual information.

Proof. This is a standard identity (Cover, 1999). By definition, I(O;Z | q) = H(O | q) − H(O | Z,q). Rearranging yields (27).

| |
|---|

Interpretation of DSDR. I(O;Z | q) captures inter-mode diversity (deep exploration across reasoning modes), while H(O | Z,q) captures intra-mode diversity (variation within a mode). DSDR’s global shaping primarily promotes inter-mode coverage, while local positive-only entropy discourages intra-mode collapse.

#### C.3. Correctness Preservation Under Bounded Local Regularization

The local objective in DSDR (Eq. (15)) increases token-level entropy only along correct trajectories and is length-normalized to avoid incentivizing longer outputs. A natural concern is whether adding this regularizer could sacrifice verifiable correctness. This subsection shows that, at the population level, a sufficiently small local regularization weight cannot make a lower-correctness policy optimal; it can only break ties among correctness-optimal policies.

Entropy Upper Bound for Length-invariant Token Entropy. For any categorical distribution over |V| tokens, entropy is maximized by the uniform distribution, so 0 ≤ H(π(· | s)) ≤ log |V| (Cover, 1999). Define the length-invariant per-token conditional entropy functional

T

1 T

H π(· | q,o<t) . (28)

JH(π) = E

t=1

Then

0 ≤ JH(π) ≤ Hmax := log |V|. (29)

The same upper bound also holds for the positive-only variant JH+(π) = E[(R(q,O) = 1) · T1 t H(·)] since (·) ≤ 1. Proposition C.2 (Correctness preservation under bounded λℓ). Define the population correctness objective

##### JR(π) = Eq∼D, o∼π(·|q)[R(q,o)] ∈ [0,1], (30)

and let JR⋆ = supπ JR(π). Define the best correctness among strictly suboptimal policies

JR(π), ∆ := JR⋆ − JR > 0. (31)

JR = sup

π: JR(π)<JR⋆

Let JH(π) be any regularizer satisfying 0 ≤ JH(π) ≤ Hmax (e.g., (28)). Consider the regularized objective

Jreg(π) = JR(π) + λℓJH(π), λℓ ≥ 0. (32) If λℓ < ∆/Hmax, then every maximizer πreg⋆ ∈ arg maxπ Jreg(π) is correctness-optimal:

JR(πreg⋆ ) = JR⋆. (33)

Proof. Take any correctness-suboptimal policy π with JR(π) ≤ JR. Using JH(π) ≤ Hmax,

Jreg(π) = JR(π) + λℓJH(π) ≤ JR + λℓHmax. (34) Take any correctness-optimal policy π⋆ with JR(π⋆) = JR⋆. Since JH(π⋆) ≥ 0,

Jreg(π⋆) = JR⋆ + λℓJH(π⋆) ≥ JR⋆. (35) If λℓ < ∆/Hmax, then

JR + λℓHmax < JR + ∆ = JR⋆ ≤ Jreg(π⋆). (36)

Thus every correctness-suboptimal π has strictly smaller Jreg(π) than some correctness-optimal π⋆, so no correctnesssuboptimal policy can maximize Jreg. Hence any maximizer satisfies (33).

| |
|---|

#### C.4. GRPO Signal Preservation via Correct-Only Global Diversity Reward

DSDR’s augmented reward (Eq. (10)) is motivated by a concrete optimization issue in group-relative methods: when verifier rewards become nearly constant within a group (e.g., many correct rollouts), the within-group variance shrinks and the group-normalized advantages used in GRPO can degenerate. This subsection formalizes two points aligned with Sec. 3.2: (i) verifier-only rewards yield informative groups only when the sampled group mixes successes and failures, and (ii) DSDR’s correct-only diversity bonus creates controlled dispersion among correct solutions, ensuring non-degenerate group-normalized advantages whenever diversity scores differ.

Lemma C.3 (Probability of a mixed verifier-reward group). Fix prompt q. Let r1,...,rG be i.i.d. Bernoulli(p), where

p = Pr(R(q,O) = 1). (37) Let Mix = {∃i,j : ri ̸= rj} denote the mixed-group event. Then

Pr(Mix) = Pmix(p;G) = 1 − pG − (1 − p)G. (38)

Proof. The complement of Mix is the disjoint union of “all ones” and “all zeros”, with probabilities pG and (1 − p)G, respectively. Therefore Pr(Mix) = 1 − pG − (1 − p)G.

| |
|---|

Proposition C.4 (Non-vanishing GRPO signal under correct-only diversity bonus). Let r˜i be defined in (22) and A˜i be the stabilized group-normalized advantages in (24). If there exist i ̸= j such that r˜i ̸= r˜j, then

Var(˜r1,...,r˜G) > 0 and {A˜i}Gi=1 are not all zero. (39) In particular, in a solve-all group (ri = 1 for all i),

##### r˜i = 1 + λdd¯i. (40)

If λd > 0 and {d¯i} are not all identical, then group-relative advantages remain non-degenerate even though verifier rewards are constant.

Proof. If r˜i are not all identical, then letting µr˜ be the group mean in (23), there exists an index i such that r˜i ̸= µr˜, hence

G

1 G

(˜rk − µr˜)2 > 0. (41)

Var(˜r1,...,r˜G) =

k=1

Because σr,ε˜ > 0 by construction, for that index i we have A˜i = (˜ri − µr˜)/σr,ε˜ ̸= 0, so the advantages are not all zero. The solve-all case follows by substituting (40): if λd > 0 and d¯i are not all equal, then r˜i are not all equal and the above applies.

| |
|---|

Remark (Connection to PPO/GRPO). In PPO-style clipped surrogate objectives (Schulman et al., 2017) and GRPO-style group-relative updates (e.g., (Guo et al., 2025)), the reward-driven policy-improvement term is scaled by normalized advantages. If all advantages were zero, the reward-driven gradient would vanish (leaving only KL regularization).

- Proposition C.4 guarantees non-degenerate preference signal whenever diversity induces non-constant r˜i.

C.5. Optimality of Diversity-Softmax Global-to-Local Coupling

DSDR couples global and local regularization by allocating the local entropy budget across correct trajectories according to the diversity-softmax weights (Eq. (11)). This subsection shows that the softmax allocation is not an ad-hoc heuristic: it is the unique solution of an entropy-regularized resource allocation problem. This provides a principled interpretation of the temperature τ as a concentration–exploration control knob.

- Proposition C.5 (Softmax allocation optimality). Assume P = {i : ri = 1} ̸= ∅. Consider allocating a local-entropy budget across correct rollouts with a distribution w over P:

wi = 1. (42)

wi ≥ 0,

i∈P

For τ > 0, the diversity-softmax weights

exp(τd¯i) j∈P exp(τd¯j)

wi⋆ =

(43)

are the unique maximizer of the entropy-regularized allocation problem

max

w∈∆(P)

where ∆(P) is the simplex in (42).

wi d¯i + H(w) , H(w) = −

τ

i∈P

i∈P

wi log wi, (44)

Proof. This is the classical maximum-entropy / Gibbs distribution derivation (Jaynes, 1957); we provide a convexoptimization proof via KKT conditions (Boyd & Vandenberghe, 2004).

(Uniqueness) The objective in (44) is linear in w plus entropy H(w), which is strictly concave over the simplex interior (Cover, 1999; Boyd & Vandenberghe, 2004). Hence the objective is strictly concave and the maximizer is unique.

(Stationarity) Form the Lagrangian with multiplier α for i∈P wi = 1:

L(w,α) = τ

i∈P

- At an interior optimum, stationarity gives

wid¯i −

wi log wi + α

i∈P

wi − 1 . (45)

i∈P

∂L ∂wi

= τd¯i − (1 + log wi) + α = 0 ⇒ log wi = τd¯i + α − 1. (46)

Exponentiating yields wi = C exp(τd¯i) with C = exp(α − 1). Enforcing the simplex constraint implies C =

j∈P exp(τd¯j) −1, yielding (43). By strict concavity, this stationary point is the unique global maximizer.

| |
|---|

Equivalent KL-Regularized Form. Let u be the uniform distribution on P. Using H(w) = log |P| − DKL(w∥u), the problem (44) is equivalent (up to an additive constant) to

τ Ei∼w[d¯i] − DKL(w∥u) ,

max

w∈∆(P)

highlighting the exploration–concentration tradeoff controlled by τ.

#### C.6. Proof of Theorem 3.1

Proof. Fix q and define

f(q,o) = exp τd¯(q,o) (R(q,o) = 1) ≥ 0. By definition,

1 τ

log Zτ(θ;q), Zτ(θ;q) = Eo∼π

θ(·|q)[f(q,o)]. Since Zτ(θ;q) > 0 by assumption, we can differentiate:

Jτ(θ;q) =

1 τ ·

1

Zτ(θ;q) ∇θZτ(θ;q). (47) Using the score-function (log-derivative) identity (Williams, 1992; Sutton et al., 1999),

∇θJτ(θ;q) =

θ(·|q)[f(q,o)∇θ log πθ(o | q)], (48) we obtain

∇θEo∼π

θ(·|q)[f(q,o)] = Eo∼π

θ(·|q)[f(q,o)∇θ log πθ(o | q)]. (49) Substituting (49) into (47) yields

∇θZτ(θ;q) = Eo∼π

1 τ

f(q,o)

Zτ(θ;q) ∇θ log πθ(o | q) . (50) Next, note that

∇θJτ(θ;q) = Eo∼π

θ(·|q)

Eo∼π

θ(·|q)[∇θ log πθ(o | q)] = ∇θ πθ(o | q)do = ∇θ1 = 0, so subtracting the constant baseline 1/τ does not change the expectation (?):

1 τ

f(q,o)

Zτ(θ;q) − 1 ∇θ log πθ(o | q) . (51) Expanding f(q,o) and Zτ(θ;q) gives exactly (18)–(19).

∇θJτ(θ;q) = Eo∼π

θ(·|q)

Finally, for the Monte Carlo form, let {oi}Gi=1 be i.i.d. samples from πθ(· | q) and define fi = f(q,oi) = exp(τd¯i)(r i = 1). A plug-in estimator for Zτ(θ;q) is Zˆ = (1/G) Gj=1 fj, and for the numerator ∇Z = (1/G) Gi=1 fi∇θ log πθ(oi | q). Thus the ratio form in (50) yields

∇Z Zˆ

1 τ ·

∇θJτ(θ;q) =

G

1 τ

fi

∇θ log πθ(oi | q),

=

G j=1 fj

i=1

which corresponds to weights wˆi = fi/ Gj=1 fj, proving (20). Because fi = 0 whenever ri = 0, restricting to the correct set P = {i : ri = 1} gives wˆi ∝ exp(τd¯i) on P, matching Eq. (11).

| |
|---|

