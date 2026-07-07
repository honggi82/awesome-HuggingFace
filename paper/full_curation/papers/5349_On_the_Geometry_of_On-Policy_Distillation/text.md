# arXiv:2606.07082v3[cs.LG]14Jun2026

## On the Geometry of On-Policy Distillation

Zhennan Shen1, Yanshu Li2, Qingyu Yin3, Chak Tou Leong4, Zhilin Wang5 Yanxu Chen6, Rongduo Han7, Sunbowen Lee8, Yi R. Fung1

1HKUST 2Brown University 3Zhejiang University 4Hong Kong PolyU 5USTC 6BUPT 7Nankai University 8BIT

Abstract

On-policy distillation (OPD) is increasingly used to improve large language model reasoning, but its training dynamics remain poorly understood. We characterize the trajectory of OPD updates in parameter space and compare it with supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). A suite of parameter-space diagnostics consistently places OPD in a relaxed offprincipal regime: compared with SFT, its updates affect fewer weights and avoid principal directions more strongly, while compared with RLVR, they remain less tightly constrained. Beyond this static localization, OPD exhibits subspace locking: its cumulative updates rapidly enter a narrow low-dimensional channel. Constraining training to the update subspace formed early in training preserves OPD performance but substantially degrades SFT, indicating that the locked subspace is functionally sufficient for OPD. Control experiments further show that sparsifying the update tokens and shifting rollout generation offpolicy preserve the rank dynamics, whereas mixing the OPD objective with RLVR changes them. Overall, these results suggest that OPD is not merely an intermediate point between SFT and RLVR, but induces its own update geometry in parameter space.

### 1 Introduction

Large reasoning models (LRMs) have substantially advanced complex mathematical and programming reasoning in large language models (Guo et al., 2025; Shao et al., 2024; OpenAI, 2024). Posttraining is a central driver of this progress. Beyond supervised fine-tuning (SFT) (Wei et al., 2022) on offline demonstrations and reinforcement learning with verifiable rewards (RLVR) (Shao et al., 2024; Guo et al., 2025; Yu et al., 2025) from sparse outcome signals, on-policy distillation (OPD) has recently emerged as a complementary paradigm: it trains a student on its own sampled trajectories

under dense token-level guidance from a stronger teacher (Agarwal et al., 2024; Lu and Lab, 2025).

Despite its empirical utility, the parameter-space dynamics of OPD remain poorly understood. Prior analyses show that SFT and RLVR leave distinct geometric footprints: SFT induces dense, principalaligned updates (Liu et al., 2026), whereas RLVR produces sparse, off-principal updates that better preserve pretrained spectral structure (Mukherjee et al., 2025; Zhu et al., 2025). OPD combines features of both: dense token-level distillation resembles SFT, while on-policy sampling and policygradient optimization connect it to RLVR (Agarwal et al., 2024). This makes its parameter trajectory difficult to infer from either paradigm alone. Thus, we study OPD through three research questions:

- RQ1 Where does OPD lie within the parameterspace spectrum between SFT and RLVR?
- RQ2 What intrinsic update trajectory does OPD follow during training?
- RQ3 Which component of OPD controls this trajectory?

OPD occupies a relaxed off-principal regime (§3). To answer RQ1, we locate OPD relative to SFT and RLVR using a suite of parameter-space diagnostics proposed by Zhu et al. (2025). The results show that across update sparsity, spectral drift, principal-subspace rotation, and update localization, OPD consistently falls between SFT and RLVR with a bias toward the RLVR side. We refer to this region as a relaxed off-principal regime: OPD is more selective and geometry-preserving than SFT, yet less constrained than RLVR. We further interpret this positioning through a relaxed Three-Gate view: OPD retains RLVR’s geometrypreserving update bias, but dense teacher supervision broadens the set of active directions and makes more coordinates visibly update.

OPD learns through subspace locking (§4). Answering RQ2, we move beyond endpoint local-

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- Figure 1: Optimization geometry of OPD compared with SFT and RLVR. (a) OPD occupies a relaxed off-principal regime between dense principal-aligned SFT updates and sparse off-principal RLVR updates. (b) OPD rapidly enters a locked low-dimensional subspace during training. (c) The lock is robust to token and rollout perturbations, but sensitive to objective composition.

ization and characterize how OPD updates evolve during training: we track cumulative updates ∆Wt across checkpoints using effective dimension, update scale, and spectral shape diagnostics. This trajectory-level analysis shows that OPD rapidly enters a narrow low-dimensional update band, while SFT expands and RLVR contracts. This is not a vanishing-update artifact: OPD has a substantially larger cumulative update norm than RLVR while ending with comparable stable rank. We then test whether this low-dimensional channel is stable and functional: subspace similarity shows early alignment with the final update channel, and constraining subsequent training to this early subspace preserves OPD performance while degrading SFT. These results identify OPD subspace locking: an early-emerging, persistent, and functionally sufficient low-dimensional update channel.

Objective composition controls subspace locking (§5). Answering RQ3, we test which part of OPD maintains the locked trajectory by perturbing three factors that distinguish OPD from standard SFT or RLVR: token supervision density, rollout policy, and objective composition. Token sparsification and off-policy rollouts preserve the OPD rank trajectory, changing update scale at most. In contrast, objective-level interpolation exposes a boundary of the locked regime, where the rank dynamics depart from the trajectory. These controls indicate that the lock is robust to runtime perturba-

tions but sensitive to objective composition.

Together, our findings provide a parameter-space interpretation of OPD-based post-training. OPD is not merely an endpoint interpolation between SFT and RLVR; it follows a distinct update trajectory characterized by relaxed off-principal localization, early subspace locking, and sensitivity to objective composition. We discuss how these diagnostics can guide future geometry-aware OPD algorithms in Section 6.

### 2 Related Work

Post-training for large language models. Posttraining large language models for reasoning has largely followed three paradigms. Supervised finetuning (SFT) trains on offline demonstrations with cross-entropy objectives (Howard and Ruder, 2018; Dodge et al., 2020; Wei et al., 2022). Reinforcement learning methods, including RLHF and reinforcement learning with verifiable rewards (RLVR), optimize policies using preference or outcomelevel feedback (Ziegler et al., 2020; Ouyang et al., 2022; Shao et al., 2024; Guo et al., 2025).

More recently, on-policy distillation (OPD) has emerged as a complementary route: it trains on student-generated rollouts while using a stronger teacher to provide dense token-level guidance (Lu and Lab, 2025; Song and Zheng, 2026). Recent work studies OPD from the perspectives of empirical failure modes, scaling recipes, multi-teacher

distillation, and training efficiency (Fu et al., 2026; Li et al., 2026; DeepSeek-AI, 2026; Yang et al., 2025; Cai et al., 2026). These accounts characterize when OPD works behaviorally or efficiently, but leave its position within the broader SFT–RLVR parameter-space spectrum largely unexamined.

Post-training weight geometry. Recent studies reveal a fundamental optimization dichotomy: SFT induces dense weight updates that distort the pretrained spectral structure along principal directions (Liu et al., 2026), whereas online RL targets highly localized, off-principal subnetworks (Mukherjee et al., 2025). This RL bias can be viewed as a conservative projection that limits policy-level KL drift (Shenfeld et al., 2025; Wu et al., 2026).

Prior work by Zhu et al. (2025) most directly informs our methodology: they provide a parameter space account of RLVR, Zhu et al. (2025) provide a parameter-space account of RLVR, showing that RLVR learns off the principal directions rather than merely changing fewer parameters. They attribute this behavior to a model-conditioned optimization bias and formalize it through a Three-Gate account: a KL anchor constrains the update magnitude, pretrained model geometry steers bounded steps toward low-curvature, spectrum-preserving subspaces, and finite-precision realization makes the resulting off-principal bias appear as sparse visible updates. Their diagnostics further show that RLVR preserves the pretrained spectrum, induces limited principal-subspace rotation, and differs sharply from SFT’s principal-direction update pattern.

These studies establish the SFT–RLVR geometric split and the PNT account of RLVR’s offprincipal dynamics. We build on this lens but study a different regime: where OPD lies within the SFT– RLVR parameter-space spectrum, how its updates evolve during training, and which OPD-specific factors control this trajectory.

### 3 Locating OPD in Parameter Space

In this section, we aim to clarify how on-policy distillation (OPD) relates to SFT-like distillation methods and RLVR-like online optimization approaches. To this end, we design controlled experiments to position OPD within the SFT–RLVR spectrum using parameter-space diagnostics, including update support, subspace rotation, spectral drift, and update localization. Specifically, we first describe the parameter-space diagnostics proposed by Zhu

et al. (2025), which provide a useful framework for analyzing training dynamics. We then apply them to compare OPD with SFT and RLVR, and finally interpret the observed behavior through a relaxed Three-Gate view.

Experimental setup. We analyze Qwen3-family checkpoints (Yang et al., 2025) spanning teacher– student gaps, data domains, and dense/MoE teachers. Our main comparison centers on Qwen3-8B: OPD and RLVR both initialized from the same SFT checkpoint and math-domain prompt distribution. For SFT, we analyze the update from the pretrained base model to the SFT checkpoint. Full experimental details are provided in Appendix C.

#### 3.1 Parameter-Space Diagnostics

We use the diagnostic suite of Zhu et al. (2025) for post-training weight geometry which contains four measurements to locate OPD in the SFT–RLVR parameter-space spectrum: update support, subspace rotation, spectral drift, and update localization. Let W0 denote the initialization, W+ the posttraining weight matrix, and ∆W = W+ − W0.

Update sparsity. We adopt the bf16-visible update convention used by Zhu et al. (2025), small updates are treated as invisible when they can be absorbed by bfloat16 rounding. We treat scalar weights wi,wˆi ∈ R as unchanged if

|wˆi − wi| ≤ η max(|wi|,|wˆi|), η = 10−3. The bf16-aware update sparsity is

1 n i,j

sp(W0,W+) = 1 −

1[W+,ij ̸≈η W0,ij],

(1) where n is the number of entries. Larger values indicate fewer visible weight changes.

Principal-angle rotation. For subspace rotation, we use the principal-angle diagnostic from Zhu et al. (2025). Specifically, rotation of the dominant singular subspaces is measured by the principal angles between the top-k subspaces of W0 and W+:

- cosθi(U) = σi U0⊤,kU+,k ,
- cosθi(V ) = σi V0⊤,kV+,k ,

(2)

where U·,k and V·,k denote the top-k left and right singular vectors. Smaller angles indicate stronger preservation of the pretrained dominant subspaces.

Base Model Finetuned (FT) Model Algorithm Data sparsitybf16 Controlled comparison: SFT < OPD < RLVR

Qwen3-8B-Base Qwen3-8B-SFT SFT Math 8.1% Qwen3-8B-SFT OPD-8B-T32B OPD Math 51.6% Qwen3-8B-SFT RLVR-8B GRPO Math 77.2%

OPD robustness across teacher / student / data

Qwen3-4B-SFT OPD-4B-T8B OPD Math 50.3% Qwen3-4B-SFT OPD-4B-T14B OPD Math 51.1% Qwen3-4B-SFT OPD-4B-T32B OPD Math 51.7% Qwen3-14B-SFT OPD-14B-T32B OPD Math 56.6% Qwen3-8B-SFT OPD-8B-T32B-Code OPD Code 57.1% Qwen3-8B-SFT OPD-8B-T32B-MoE OPD Math 48.6%

Published reference points

Qwen3-8B-Base Klear-Reasoner-8B-SFT SFT Math+Code 0.6% Qwen3-14B-Base UniReason-14B-think-SFT SFT Mixed† 18.8% Klear-Reasoner-8B-SFT Klear-Reasoner-8B GRPO Math+Code 69.9% Qwen3-8B-Base GT-Qwen3-8B-Base GRPO Math 79.0% Qwen3-4B Polaris-4B-Preview DAPO Math 79.9%

Table 1 bf16-aware update sparsity. OPD lies between SFT and RLVR in the controlled Qwen3-8B comparison, and remains stable across OPD variants. Published checkpoints are used only as external reference points. In OPD variant names, TxB denotes a teacher model with xB parameters, e.g., OPD-4B-T8B uses a 4B student and an 8B teacher. Mixed† denotes a mixture of math, code, STEM, logic, and instruction data.

Principal Angle (single-layer)

###### Singular Value (single-layer)

###### Max Principal Angle (all-layer)

###### Spectrum Drift (all-layer)

|Left (U)<br><br>Right (V)|
|---|

|Base<br><br>Tuned|
|---|

|Left (U)<br><br>Right (V)|
|---|

|1e 3<br><br>|
|---|

10

100

MaxAngle(°)

Singularvalue

6

3.5

Angle(°)

SFT

NSS

5

50

4

3.0

0

0

0

200 400

0 200 400

0 100 200

0 50 100 150 200

30

|Left (U)<br><br>Right (V)|
|---|

|Left (U)<br><br>Right (V)|
|---|

|Base<br><br>Tuned|
|---|

|1e 4<br><br>|
|---|

2.0

MaxAngle(°)

Singularvalue

6

1.0

Angle(°)

20

###### OPD

###### NSS

1.5

0.5

10

4

1.0

0.0

0

0 200 400

0 200 400

0 100 200

0 50 100 150 200

|Left (U)<br><br>Right (V)|
|---|

|Base<br><br>Tuned|
|---|

||Left (U<br><br>Right (V|)<br><br>)|
|---|---|
| | |
|
|---|

|1e 5<br><br>|
|---|

MaxAngle(°)

Singularvalue

0.4

6

Angle(°)

- 1
- 2

5

###### RLVR

NSS

0.2

4

0.0

0

0 200 400

0 200 400

0 100 200

0 50 100 150 200

Rank index k

Rank index k

Weight matrix index

Weight matrix index

- Figure 2: Parameter-space diagnostics. SFT induces larger subspace rotation and spectral drift, RLVR preserves the pretrained geometry most strongly, and OPD lies between them. Here, k denotes the rank index of principal angles or singular values; the all-layer panels enumerate analyzed weight matrices across layers and module types.

Spectral drift. For spectral preservation, we report the normalized spectral shift (NSS) used in Zhu et al. (2025):

NSS(W0,W+) = ∥σ(W+) − σ(W0)∥2 ∥σ(W0)∥2

, (3)

where σ(·) denotes singular values in descending order. Smaller NSS indicates stronger spectral preservation.

Update–mask overlap. For update localization, we instantiate the mask-overlap analysis of Zhu

et al. (2025) by comparing the bf16 update mask

M = {(i,j) : bf16(W+)ij ̸= bf16(W0)ij}

with two masks derived from W0. The principal mask Mprinc contains the top-α fraction of entries in the rank-k SVD reconstruction of W0, serving as a proxy for high-curvature regions (Liu et al., 2026). The low-magnitude mask Mlow contains the bottom-α fraction of entries by |W0|. For M⋆ ∈ {Mprinc,Mlow}, we report

Overlap(M⋆,M) = |M⋆ ∩ M| |M|

,

Update density

###### Cond. principal overlap

###### Cond. low-mag. overlap

0

|[Figure 80]<br><br>92.73%|
|---|

|[Figure 81]<br><br>29.66%|
|---|

|[Figure 82]<br><br>31.88%|
|---|

100.00%

1024

|[Figure 83]|
|---|

SFT

2048

3072

4096

0

|[Figure 84]<br><br>46.28%|
|---|

|[Figure 85]<br><br>27.31%|
|---|

|[Figure 86]<br><br>53.59%|
|---|

1024

Localratio/density

OPD

2048

3072

50.00%

4096

0

|[Figure 87]<br><br>27.76%|
|---|

|[Figure 88]<br><br>26.65%|
|---|

|[Figure 89]<br><br>73.48%|
|---|

1024

RLVR

2048

3072

4096

0 1024 2048 3072 4096

0 1024 2048 3072 4096

0 1024 2048 3072 4096

100

50

100

Low-M 73.48%

Upd.

Princ.

92.73%

29.66% 27.31% 26.65%

Global

0.00%

53.59%

46.28%

| |
|---|

50

25

50

| |
|---|

| |
|---|

31.88%

27.76%

0 SFT OPD RLVR

0 SFT OPD RLVR

0 SFT OPD RLVR

- Figure 3: Update-mask localization. We compare where bf16-visible updates land relative to principal and low-magnitude masks. OPD shifts updates away from principal weights and toward low-magnitude regions, while remaining less selective than RLVR.

with random baseline α. Sub-random overlap with Mprinc indicates depletion from principal weights, while super-random overlap with Mlow indicates concentration in low-magnitude regions.

- 3.2 OPD Occupies a Relaxed Off-Principal Regime

Using these PNT-style diagnostics, OPD falls in an intermediate but RLVR-biased region of the parameter-space spectrum. OPD is more selective and geometry-preserving than SFT, yet less constrained than RLVR. We call this regime relaxed off-principal.

Update sparsity. In the controlled Qwen3-8B comparison, SFT leaves only 8.1% of weights unchanged at bf16 precision, while RLVR leaves 77.2% unchanged. OPD lies between them at 51.6%. This pattern is stable across OPD variants: teacher scale, student scale, code data, and a MoE teacher keep OPD within 48.6%–57.1% sparsity. Published reference checkpoints exhibit the same SFT–RLVR separation (Table 1).

Subspace rotation and spectral drift. SFT induces the largest rotation of the pretrained singular subspaces: in the representative layer, its principal angles rise above 10◦. OPD shows smaller but nonzero rotation, with representative-layer angles around 1◦, while RLVR exhibits the smallest rotations below 0.5◦. Spectral drift follows the same

ordering: SFT is at the 10−3 level, OPD at the 10−4 level, and RLVR at the 10−5 level (Figure 2).

Update localization. Conditioned on bf16visible updates, global update density decreases from SFT to OPD to RLVR: 92.73%, 46.28%, and 27.76%. Principal-mask overlap also decreases monotonically, placing OPD and RLVR below the 30% random baseline. Low-magnitude overlap shows a stronger separation: 31.88% for SFT, 53.59% for OPD, and 73.48% for RLVR (Figure 3).

Across all three diagnostics, OPD follows the same off-principal direction as RLVR, but with weaker selectivity and larger visible support.

#### 3.3 A Relaxed Three-Gate Account of OPD

We next use the Three-Gate account of RLVR from Zhu et al. (2025) as a reference lens, and ask how this geometry changes when the sequence-level RLVR signal is replaced by dense teacher-token supervision in OPD. In this view, RLVR updates become off-principal through three filters: (1) a distributional anchor that limits update size, (2) pretrained model geometry that routes bounded updates away from dominant spectral directions, and (3) finite-precision realization that determines which coordinates become visibly changed. We argue that OPD preserves this gated structure, but relaxes it through dense token-level teacher super-

RLVR OPD SFT

(a) Stable rank

(b) Update magnitude

(c) Hill alpha

75

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | |OPD band| |
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
| | | | | |
| | | | | |
| | | | | |

4.5

StablerankofΔWt

65

4.0

100

‖ΔW‖tF

50

Hillα

3.5

35

3.0

10−1

2.5

20

10

2.0

25 50 75 100

25 50 75 100

25 50 75 100

Training progress (%)

Training progress (%)

Training progress (%)

- Figure 4: Intrinsic update geometry. We track cumulative updates ∆Wt. OPD stays in a narrow stable-rank band, whereas SFT expands and RLVR contracts. Frobenius norms rule out a small-update explanation: OPD moves more than RLVR while ending with comparable stable rank. Hill estimates provide an auxiliary spectral-shape check.

vision. Full objective forms and derivation details are provided in Appendix D.

Signal granularity. The key difference is the granularity of the training signal. Let

ϕt = ∇θ log pθ(yt | x,y<t)

denote the token score. A broad class of posttraining updates can be written as

T

atϕt. (4)

g =

t=1

The paradigms differ in the coefficients at. In RLVR, the reward signal is sequence-level, so at = A(y) is shared across tokens. In OPD, at varies across tokens according to local teacher– student disagreement. Thus, OPD retains the onpolicy form of RLVR, but replaces a scalar credit signal with dense token-level supervision.

- Gate I: distributional anchor. OPD remains anchored. Under a local quadratic budget, the update norm is bounded as

∥∆W∥F ≤

2δW µW

. (5)

The difference is how this budget is used. RLVR spends it through a sequence-level reward signal, whereas OPD uses teacher-token distributions to provide denser descent directions within the anchored region. This yields larger effective updates than RLVR while remaining far below SFT-style unconstrained rewriting.

- Gate II: model geometry. A bounded update is still shaped by pretrained model geometry. OPD therefore does not freely rewrite the dominant spectral structure as SFT does. The relaxation appears in the update covariance:

E[gg⊤] =

E atat′ϕtϕ⊤t′ . (6)

t,t′

RLVR couples all token scores through a shared scalar coefficient, whereas OPD uses heterogeneous token coefficients. This broadens the accessible directional support while keeping the update geometry-steered, matching the observed pattern that OPD preserves spectral structure more than SFT but less strictly than RLVR.

Gate III: precision realization. The bf16 realization gate is unchanged. A coordinate becomes visibly updated only when

Mij = 1 |∆Wij| ≳ 21ULPbf16(W0,ij) . (7)

Larger geometry-constrained updates allow more OPD coordinates to pass the bf16 realization threshold, lowering sparsity while preserving the offprincipal bias.

Implication. OPD occupies a relaxed offprincipal regime: geometry-steered like RLVR, but less selective under dense teacher supervision. This suggests that OPD recipes should regulate update geometry, rather than treating token-level supervision density as the primary design axis.

### 4 Subspace Locking in OPD

Section 3 located on-policy distillation (OPD) within the SFT–RLVR parameter-space spectrum. We now ask whether this positioning is only an endpoint property, or whether OPD follows a distinct update trajectory during training. We study the cumulative update ∆Wt = Wt − W0 across checkpoints and show that OPD rapidly enters a persistent low-dimensional update channel.

Similarity to final V16 subspace

1.0

0.8

Sim(t,t)16end

0.6

OPD

RLVR

0.4

SFT

0 25 50 75 100

Training progress (%)

- Figure 5: Subspace emergence. Top-16 subspace similarity to the final update shows that OPD locks onto its final update channel earlier than SFT and RLVR.

- 4.1 A Low-Dimensional Update Band We first characterize the effective dimension of ∆Wt. For each matrix, we compute the stable rank

srank(∆Wt) = ∥∆Wt∥2F ∥∆Wt∥2op

, (8)

and average over all analyzed weight matrices. This measures how many dominant singular directions carry the update energy.

- Figure 4(a) shows that OPD stays within a

narrow low-rank band throughout training. SFT progressively expands its update subspace, while RLVR contracts toward a low-dimensional endpoint. Thus, OPD is not a temporal interpolation between the two: it enters the low-dimensional regime early and remains there.

To rule out a trivial small-update explanation, we complement stable rank with update-scale and spectral-shape diagnostics. Frobenius norms in

- Figure 4(b) show that OPD accumulates a substantially larger update than RLVR while ending with comparable stable rank. Hill tail estimates in Figure 4(c) provide an auxiliary check on the singularvalue profile: OPD evolves mildly, whereas SFT increases sharply and RLVR decreases over training. Together, these diagnostics show that OPD’s low-dimensionality is not a byproduct of small parameter movement, but a bounded spectral profile of a nontrivial update. Full metric definitions are given in Appendix E.

#### 4.2 Stability and Functional Sufficiency

Early subspace emergence. Low dimensionality alone does not imply subspace locking: a trajectory may remain low rank while rotating through different low-dimensional subspaces. We therefore test when the final low-dimensional update channel emerges. Let VK(t) denote the top-K right singular subspace of ∆Wt. We compare each checkpoint

[Figure 90]

Figure 6: Rank-16 projected training. Rank-16 projection leaves OPD intact but degrades SFT.

to the final subspace VK(tend):

1 K

SimK(t,tend) =

VK(t)⊤VK(tend)

2 F

. (9)

Values near 1 indicate that the update already uses the singular directions of the final update channel.

Figure 5 shows that OPD aligns with its final V16 subspace from the first measured checkpoint, whereas SFT and RLVR converge more gradually. Thus, OPD’s low-dimensional channel emerges early rather than being assembled late.

Functional sufficiency. We next ask whether this channel is only descriptive or also sufficient for learning. Motivated by the early stable-rank scale of OPD, we use K = 16 as a stringent bottleneck: it is close to, but slightly below, the effective dimension observed near the projection point. We then constrain each gradient matrix g to the early right singular subspace:

g ← gV16V16⊤. (10)

For both OPD and SFT, V16 is extracted from the top right singular vectors of ∆Wt around 20% training progress, and training then resumes under this rank-16 constraint.

Figure 6 shows that OPD is essentially unchanged under the rank-16 bottleneck, indicating that the early low-dimensional channel is sufficient for OPD training. The same constraint degrades SFT over the matched window, confirming that this sufficiency is not a generic property of the projection dimension. The same qualitative pattern

holds across additional reasoning benchmarks (Appendix F, Figure 8).

Implication. OPD subspace locking identifies an early-emerging update channel that is both persistent and sufficient for training. Future OPD implementations should therefore monitor and exploit this channel, rather than treating low-dimensionality as a post-hoc spectral statistic only.

### 5 What Controls Subspace Locking?

Section 4 shows that on-policy distillation (OPD) learns through an early-emerging update channel sufficient for training. We now ask what maintains this channel by perturbing three candidate factors: token-level teacher supervision, rollout policy, and objective composition. We use stable rank as the primary diagnostic, with details in Appendix G.

#### 5.1 Runtime Perturbations Preserve the Lock

We first perturb runtime sources while keeping the objective fixed. Token sparsification retains either top-KL or randomly selected tokens at 25% and 50% density. Figure 7(a) shows that all sparsified variants closely track the OPD stable-rank trajectory. Even random 25% retention changes update scale more than spectral shape, indicating that the low-rank profile is not localized to a small set of high-KL tokens.

Off-policy rollout perturbations also leave the stable-rank trajectory nearly unchanged (Figure 7(b)). Although off-policy OPD produces a modestly larger update norm, its stable rank remains matched to on-policy OPD, showing robustness to rollout-policy changes.

- 5.2 Objective Mixing Changes the Rank Dynamics

We perturb objective composition by interpolating the OPD and RLVR advantage signals in our GRPO-style policy-gradient implementation:

A(iα) = αAi,OPD + (1 − α)Ai,RLVR. (11)

Unlike token sparsification or off-policy generation, this changes the gradient source rather than only the sampled data.

Figure 7(c) shows a clear split. OPD-dominant mixtures retain the OPD-like stable-rank trajectory. When the OPD component becomes weak, the trajectory no longer follows the baseline and enters

a distinct spectral regime. Objective composition therefore changes the rank dynamics in a way that runtime perturbations do not.

Mechanistic view. The controls separate sample perturbations from objective perturbations. OPD gradients can be written as token-level sums,

gOPD =

t

Jt⊤δt, g˜OPD =

t

mtJt⊤δt, (12)

where Jt is the local Jacobian, δt is the teacher– student token discrepancy, and mt is a token mask. If token gradients share a dominant update subspace, masking primarily rescales the second moment,

E[˜gg˜⊤] ≈ cE[gOPDgOPD⊤ ] + noise, (13)

so the leading spectral directions are preserved. Off-policy rollouts change the sampling distribution, but retain the same teacher-token gradient source. Objective mixing changes the source itself:

gα = αgOPD + (1 − α)gRLVR, Σα ≈ α2ΣOPD + (1 − α)2ΣRLVR

(14)

+ α(1 − α)Σcross.

Thus, weakening the OPD term can change the dominant covariance geometry, explaining why objective mixing, unlike token thinning or rollout changes, breaks the OPD-like rank trajectory.

Implication. Subspace locking is objectivesensitive rather than runtime-induced. Effective OPD recipes should regulate objective-induced update geometry, not only token coverage or rollout generation.

### 6 Discussion & Conclusion

Our analysis identifies on-policy distillation (OPD) as a distinct parameter-space regime rather than a simple endpoint interpolation between SFT and RLVR. OPD occupies a relaxed off-principal region, but its training trajectory further exhibits subspace locking: cumulative updates rapidly enter a small, persistent low-dimensional channel that is sufficient to preserve training progress.

These findings suggest a key guiding principle for future OPD algorithms: design OPD as geometry control, not merely as denser token supervision. Effective recipes should monitor the locked update channel and use objective composition as the primary lever when this geometry drifts, with token

(a) Token sparsification

(b) Rollout policy

(c) Objective mixing

29

On-policy Off-policy

OPD

OPD

α=0.25 α=0.05 α=0.01

Top-KL 50% Top-KL 25% Random 50% Random 25%

StablerankofΔWt

27

24

21

18

25 50 75 100

25 50 75 100

25 50 75 100

Training progress (%)

Training progress (%)

Training progress (%)

Figure 7: Controls on subspace locking. Runtime perturbations preserve the OPD stable-rank trajectory, whereas objective-level interpolation changes it, identifying objective composition as the sensitive control axis.

selection, rollout policy, and teacher scale tuned through their effect on update geometry. Such geometry-aware control may make OPD more stable, interpretable, and transferable by preserving the update channel that supports learning while avoiding unnecessary parameter-space drift.

### Limitations

Our analysis focuses on controlled Qwen3-family reasoning settings. While this design isolates the effect of training paradigms, the observed geometry may vary across other model families, modalities, and task distributions.

Our diagnostics characterize parameter-space trajectories from stored checkpoints. The ThreeGate account and covariance analysis should therefore be viewed as mechanistic explanations consistent with the evidence, rather than complete causal or formal theories of on-policy distillation (OPD) optimization.

### Acknowledgments

The experimental setting and analysis methodology in Section 3 are motivated by the parameter-space perspective of PNT (Zhu et al., 2025) on RLVR post-training. In particular, we adopt their diagnostic protocol to locate OPD within the SFT–RLVR spectrum and use the resulting OPD-specific findings for further analysis. Several visualization and table-layout choices are also inspired by their paper. All measurements, checkpoints, and OPD-specific analyses in this paper are our own.

### References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. 2024. On-policy distillation of language models: Learning from self-generated mistakes. Preprint, arXiv:2306.13649.

Yuchen Cai, Ding Cao, Liang Lin, Chunxi Luo, Xin Xu, Kai Yang, Weijie Liu, Saiyong Yang, Tianxiang

Zhao, Guangzhong Sun, Guiquan Liu, and Junfeng Fang. 2026. Learning to foresee: Unveiling the unlocking efficiency of on-policy distillation. Preprint, arXiv:2605.11739.

DeepSeek-AI. 2026. Deepseek-v4: Towards highly efficient million-token context intelligence.

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. 2020. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. Preprint, arXiv:2002.06305.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. 2026. Revisiting on-policy distillation: Empirical failure modes and simple fixes. Preprint, arXiv:2603.25562.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Bruce M. Hill. 1975. A simple general approach to inference about the tail of a distribution. The Annals of Statistics, 3(5):1163–1174.

Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. Preprint, arXiv:1801.06146.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando SolarLezama, Koushik Sen, and Ion Stoica. 2024. Livecodebench: Holistic and contamination free evaluation of large language models for code. Preprint, arXiv:2403.07974.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. 2026. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. Preprint, arXiv:2604.13016.

Zihang Liu, Tianyu Pang, Oleg Balabanov, Chaoqun Yang, Tianjin Huang, Lu Yin, Yaoqing Yang, and Shiwei Liu. 2026. Lift the veil for the truth: Principal weights emerge after rank reduction for reasoning-focused supervised fine-tuning. Preprint, arXiv:2506.00772.

Kevin Lu and Thinking Machines Lab. 2025. Onpolicy distillation. Thinking Machines Lab: Connectionism. Https://thinkingmachines.ai/blog/onpolicy-distillation.

Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. 2025. Deepcoder: A fully open-source 14b coder at o3-mini level. Notion Blog.

Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng. 2025. Reinforcement learning finetunes small subnetworks in large language models. Preprint, arXiv:2505.11711.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, and 49 others. 2025. Olmo 3. Preprint, arXiv:2512.13961.

OpenAI. 2024. Openai o1 system card. Preprint, arXiv:2412.16720.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. Preprint, arXiv:2203.02155.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. 2025. Rl’s razor: Why online reinforcement learning forgets less. Preprint, arXiv:2509.04259.

Mingyang Song and Mao Zheng. 2026. A survey of on-policy distillation for large language models. Preprint, arXiv:2604.00626.

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022. Finetuned language models are zero-shot learners. Preprint, arXiv:2109.01652.

Fang Wu, Weihao Xuan, Ximing Lu, Mingjie Liu, Yi Dong, Zaid Harchaoui, and Yejin Choi. 2026. The invisible leash: Why rlvr may or may not escape its origin. Preprint, arXiv:2507.14843.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,

Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and 16 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. Preprint, arXiv:2503.14476.

Hanqing Zhu, Zhenyu Zhang, Hanxian Huang, DiJia Su, Zechun Liu, Jiawei Zhao, Igor Fedorov, Hamed Pirsiavash, Zhizhou Sha, Jinwon Lee, David Z. Pan, Zhangyang Wang, Yuandong Tian, and Kai Sheng Tai. 2025. The path not taken: RLVR provably learns off the principals. arXiv preprint arXiv:2511.08567.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020. Fine-tuning language models from human preferences. Preprint, arXiv:1909.08593.

### A AI Usage

We used ChatGPT for writing assistance, including language polishing, LaTeX formatting, organization suggestions, and refinement of presentation. We also used it to assist with drafting plotting and analysis scripts. All scientific claims, experimental designs, code, figures, numerical results, and final manuscript content were reviewed, verified, and edited by the authors. No generative AI system was used as an author, to produce experimental results, or to make autonomous scientific decisions.

### B Artifact Use

We use publicly available model checkpoints, datasets, and benchmarks only for research evaluation and analysis. We cite the corresponding creators in the main text or appendix and use these artifacts consistently with their intended research use. We checked the public license or usage terms of the major artifacts where available, including Qwen3, DAPO-Math-17k, Dolci-Think SFT, DeepCoder, and LiveCodeBench. We do not redistribute the original artifacts.

### C Experimental Details C.1 Training Setup

This section summarizes the training setups used in the main parameter-space analyses. Table 2 gives

the shared setup. Tables 3, 4, and 5 report the default supervised finetuning (SFT), on-policy distillation (OPD), and reinforcement learning with verifiable rewards (RLVR) settings, respectively. OPD and RLVR are initialized from the same SFT anchor, while the SFT trajectory is measured from the pretrained base checkpoint. We use Qwen3 checkpoints (Yang et al., 2025); math-domain OPD/RLVR runs use dapo-math-17k (Yu et al., 2025), and the SFT anchor is trained on Dolci-Think SFT data (Olmo et al., 2025).

Item Shared setting

Model family Qwen3 Main backbone Qwen3-8B Evaluation AIME 2024 avg@8 Precision bf16 training Gradient accumulation fp32 Attention softmax fp32 Hardware H200 80GB GPUs Dropout 0.0

Table 2 Shared settings used in the main training runs.

###### Item SFT default setting

Initialization Qwen3-8B-Base Objective Supervised next-token prediction Loss mask Qwen3 loss mask Training data Dolci-Think SFT data Epochs 5 Final checkpoint iter_0005375 Global batch size 256

- Learning rate 10−5 Schedule Cosine decay to 10−6 Warmup 10% of training steps Optimizer Adam Weight decay 0.1 Adam betas (0.9, 0.95) Save interval 256 steps Eval interval 256 steps Role Produces the shared SFT anchor

Table 3 Default SFT setting used to produce the shared anchor checkpoint.

Endpoint selection. We choose the main analyzed endpoints according to the evaluation curves observed during training. The SFT, OPD, and RLVR runs showed little additional improvement near the selected checkpoints, corresponding approximately to 5k, 300, and 1k training steps, respectively. We use these rounded horizons as the main comparison points for interpretability and reproducibility, rather than tuning endpoints to optimize the parameter-space diagnostics.

Item OPD default setting

Initialization SFT anchor iter_0005375 Student Qwen3-8B Teacher Qwen3-32B Training data dapo-math-17k Rollout policy Student-generated, on-policy Rollout prompts 300 per step Samples per prompt 8 Rollout temperature 1.0 Training steps 299 Save interval 64 steps Objective On-policy distillation Teacher supervision Token log-probabilities OPD coefficient 1.0 Advantage estimator GRPO KL loss Disabled Entropy bonus Disabled Global batch size 64 Learning rate 10−6 Schedule Constant Optimizer Adam Weight decay 0.1 Adam betas (0.9, 0.98)

Table 4 Default OPD setting used in the main experiments.

Geometry origins. For OPD and RLVR, W0 is the shared SFT anchor iter_0005375. For SFT, W0 is the pretrained Qwen3-8B-Base checkpoint. All updates are computed as ∆W = W+ − W0 relative to the corresponding stage initialization. Thus, OPD and RLVR share a common origin in the parameter-space comparison.

#### C.2 OPD Variant Setup

All OPD variant runs follow Table 4 unless the changed factor is listed explicitly. Table 6 summarizes the student, teacher, and changed factor for each variant. Additional variant-specific notes are provided in Table 7.

The code-domain variant uses DeepCoder data (Luo et al., 2025); evaluation is conducted on LiveCodeBench v5 (Jain et al., 2024).

#### C.3 Parameter-Space Diagnostic Implementation

All diagnostics are computed offline on saved checkpoints. Each analysis loads a pair of checkpoints (W0,W+) and computes ∆W = W+ −W0. For OPD and RLVR, W0 is the shared SFT anchor; for SFT, W0 is the pretrained Qwen3-8BBase model. Metrics are computed per matrix and then averaged across matrices unless otherwise specified.

bf16-aware update sparsity. Both W0 and W+ are cast to bf16 and then back to fp32 before com-

###### Item RLVR default setting

Initialization SFT anchor iter_0005375 Student Qwen3-8B Teacher None Training data dapo-math-17k Rollout policy Student-generated, on-policy Rollout prompts 1024 per step Samples per prompt 8 Rollout temperature 1.0 Training steps 1023 Objective Verifier-reward GRPO Reward Binary DeepScaler math accuracy Advantage estimator GRPO KL loss Disabled Entropy bonus Disabled PPO clipping ϵ = 0.2, ϵhigh = 0.28 Dynamic filter Skip batches with zero reward variance Global batch size 512

- Learning rate 10−6 Schedule Constant Optimizer Adam Weight decay 0.1 Adam betas (0.9, 0.98) Adam epsilon 10−15 Table 5 Default RLVR setting used in the main experiments.

parison. A scalar entry is considered unchanged when

|W+bf16,ij − W0bf16,ij | ≤ η max |W0bf16,ij |,|W+bf16,ij | ,

η = 10−3. The bf16-aware sparsity is

1 n i,j

spbf16(W0,W+) = 1−

1[W+,ij ̸≈η W0,ij],

where n is the number of entries in the analyzed matrix. Overall sparsity is parameter-count weighted across analyzed weights.

#### Principal-angle rotation. For each matrix, let

##### W0 = U0Σ0V0⊤,W+ = U+Σ+V+⊤.

Let U0,k,V0,k and U+,k,V+,k denote the top-k left and right singular subspaces. We compute principal angles by

##### cosθiU = σi(U0⊤,kU+,k),cosθiV = σi(V0⊤,kV+,k).

We use k = 512 by default and report summary statistics of the resulting angle vectors. These diagnostics use CPU float64 SVD, since GPU SVD can produce unstable angles for near-identical matrices.

Variant Student Teacher Changed factor

Baseline Qwen3-8B Qwen3-32B Reference MoE teacher Qwen3-8B Qwen3-30B-

Teacher architecture

A3B

4B→8B Qwen3-4B Qwen3-8B Teacher scale 4B→14B Qwen3-4B Qwen3-14B Teacher scale 4B→32B Qwen3-4B Qwen3-32B Teacher

scale 14B→32B Qwen3-

Qwen3-32B Student scale

14B

Code domain Qwen3-8B Qwen3-32B Data domain Multi-seed Qwen3-8B Qwen3-32B Random

seed

Table 6 OPD variant runs used for robustness checks in Table 1.

Variant Additional setting Code domain Uses DeepCoder data and LiveCodeBench v5

evaluation MoE teacher Uses Qwen3-30B-A3B as the teacher model Multi-seed Uses the same Qwen3-8B/Qwen3-32B math

setting with different random seeds

Table 7 Additional notes for OPD variant runs.

Spectral drift. Let σ(W) denote the singularvalue vector of W in descending order. We measure normalized spectral shift as

NSS(W0,W+) = ∥σ(W+) − σ(W0)∥2 ∥σ(W0)∥2

.

NSS is computed per matrix and then averaged without parameter-count weighting.

Update–mask overlap. The visible update mask is

Mupd = {(i,j) : W+,ij ̸≈η W0,ij}. We compare it with two masks derived from W0. The principal mask Mprinc contains the top-α fraction of entries by magnitude in the rank-r SVD reconstruction of W0, and the low-magnitude mask Mlow contains the bottom-α fraction of entries by |W0|. For M⋆ ∈ {Mprinc,Mlow}, we report

Overlap(M⋆,Mupd) = |M⋆ ∩ Mupd| |Mupd|

.

We use rank r = 64 and α = 0.5 unless otherwise specified. The random baseline is α, since each reference mask contains an α fraction of entries.

Matrix coverage and averaging. For bf16aware sparsity, we compute overall sparsity over all analyzed weight parameters and report both overall

and per-type summaries. For spectral geometry and update-mask overlap, we analyze standard weight matrices across layers; QKV projections are split into Q, K, and V when needed. Metrics are first computed per matrix and then averaged across matrices. This avoids domination by a small number of large matrices while preserving layer-wise and module-wise variation.

### D Objectives and Three-Gate Extension

Original Three-Gate account. The Three-Gate account of RLVR (Zhu et al., 2025) explains visible update sparsity as the outcome of a constrained, geometry-steered, and precision-filtered optimization process.

Gate I: KL anchor. RLVR updates are locally constrained in policy space. In a KL-regularized or trust-region view, the post-update policy remains close to the current or reference policy:

DKL(πθ+ ∥πθ) ≤ K. (15)

This policy-space leash prevents each step from freely rewriting the pretrained model.

- Gate II: model geometry. The KL anchor limits

the size of the update, but not its location. The second gate attributes this location to pretrained model geometry. Given a bounded step, the structured loss landscape steers updates away from high-curvature principal directions and toward lower-curvature, spectrum-preserving regions. This explains why RLVR updates are off-principal rather than merely small.

- Gate III: precision realization. The final

gate determines which coordinates become visible in stored weights. Under bf16 precision, sub-threshold micro-updates are not realized as changed parameters. Thus the underlying routing bias appears as high bf16-aware update sparsity: preferred regions receive visible updates, while changes elsewhere remain hidden.

Training objectives. We write the three posttraining paradigms in a common notation. For an input x and sequence y = (y1,...,yT), SFT minimizes cross-entropy on offline demonstrations:

T

LSFT = −

t=1

log pθ(yt⋆ | x,y<t⋆ ). (16)

RLVR uses on-policy samples with a scalar

sequence-level advantage:

T

LRLVR = −A(y)

log pθ(yt | x,y<t)+βRKL,

t=1

(17) where

RKL = DKL(pθ(· | x)∥pref(· | x)). (18)

Here A(y) denotes a normalized sequence-level advantage and pref is the reference policy. Cliponly RL variants can be viewed as replacing the explicit KL penalty with a local trust-region effect.

OPD optimizes student-generated rollouts against a teacher distribution:

where

LOPD =

T

DKL ptθ ∥qTt , (19)

t=1

ptθ = pθ(· | x,y<t),qTt = qT(· | x,y<t).

If a forward-KL implementation is used, the KL direction is replaced accordingly; the signalgranularity argument below is unchanged.

#### Unified score-weighted form. Let

ϕt = ∇θ log pθ(yt | x,y<t) (20)

denote the token score. A broad class of stochastic post-training updates can be abstracted as

g =

T

atϕt. (21)

t=1

The distinction lies in the coefficient at. In RLVR, at = A(y) is shared across the sequence. In OPD, at varies across tokens, induced by teacher–student disagreement. Thus OPD keeps the on-policy structure of RLVR but replaces a scalar credit signal with dense token-level supervision.

OPD-specific relaxation. The relaxation appears at the level of update covariance. Let C = E[gg⊤]. For RLVR,

CRLVR = E A(y)2ss⊤ ,s =

t

ϕt. (22)

For OPD,

 

COPD = E

 . (23)

atat′ϕtϕ⊤t′

t,t′

RLVR couples all token scores through one sequence-level scalar. OPD uses heterogeneous token coefficients, expanding the accessible directional support within the same geometry-steered update family. This is the sense in which OPD is relaxed: it remains anchored and geometryconstrained, but less selective than RLVR.

Local anchor and weight bound. We now make the anchoring step explicit. For a weight block W, consider a local quadratic constraint

- 1

- 2⟨vec(∆W),SWvec(∆W)⟩ ≤ δW,SW ⪰ µWI.

(24) This implies

2δW µW

. (25)

∥∆W∥F ≤

The bound captures the anchoring effect. OPD differs from RLVR not by removing the anchor, but by using denser token-level information inside the anchored region.

Spectral stability under bounded updates. Let W+ = W0 + ∆W and γk = σk(W0) − σk+1(W0) be the singular-value gap. By Wedin-type perturbation bounds,

max{∥sinΘ(Uk(W0),Uk(W+))∥2, ∥sinΘ(Vk(W0),Vk(W+))∥2} ≤

∥∆W∥2 γk

.

Singular values satisfy

(26)

|σi(W+) − σi(W0)| ≤ ∥∆W∥2, (27) and

∥σ(W+) − σ(W0)∥2 ≤ ∥∆W∥F. (28) Therefore, updates with small operator and Frobenius norm tend to preserve the pretrained spectral structure. OPD remains in this bounded regime, but its dense token-level signal permits larger subspace rotation and spectral drift than RLVR.

Precision realization. bf16 precision determines which coordinates become visible in stored weights. A coordinate is realized as changed only if

|∆Wij| ≳ 12ULPbf16(W0,ij). (29) The corresponding visible update mask is

Mij = 1 |∆Wij| ≳ 12ULPbf16(W0,ij) . (30) Since OPD updates are larger than RLVR updates but remain geometry-constrained, more coordinates cross the realization threshold, yielding intermediate bf16-aware sparsity.

### E Trajectory Metric Definitions

For trajectory-level analysis, we study the cumulative update ∆Wt = Wt − W0 at each checkpoint. Unless otherwise specified, metrics are computed for each analyzed weight matrix and then averaged across matrices.

Stable rank. Stable rank measures the effective number of dominant singular directions that carry the update energy:

srank(∆Wt) = ∥∆Wt∥2F ∥∆Wt∥2op

. (31)

Unlike algebraic rank, stable rank is insensitive to arbitrarily small singular values and therefore provides a scale-aware measure of effective update dimensionality.

Frobenius norm. We use the Frobenius norm to measure cumulative update magnitude:

 

 

1/2

(∆Wt,ij)2

. (32)

∥∆Wt∥F =

i,j

This diagnostic rules out the possibility that low stable rank is merely caused by negligible parameter movement.

Hill tail estimate. To inspect the spectral shape of ∆Wt, we use the Hill tail estimator (Hill, 1975). Let σ1 ≥ σ2 ≥ ··· ≥ σk denote the selected top singular values. We estimate the tail exponent as

αHill =

k−1

1 k − 1

σi σk

log

i=1

−1

. (33)

We use this only as an auxiliary spectral-shape diagnostic, complementary to stable rank.

### F Additional Evaluation of Rank-Constrained Training

To test whether the functional-sufficiency result in Section 4.2 is specific to AIME 2024, we evaluate the same rank-16 projected runs on additional reasoning benchmarks, as shown in Figure 8. The projection subspace, projection start, and matched training windows are the same as in the main experiment.

The additional benchmarks show the same qualitative pattern as Figure 6. OPD remains broadly robust under the early rank-16 subspace constraint,

plain K16 shared anchor projection start

AIME 2024

###### AMC 2023

###### AIME 2025

Minerva

Beyond AIME

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

94

- 44
- 45
- 46
- 47
- 48
- 49

34

52

66

92

50

32

avg@32(%)

64

avg@4(%)

avg@8(%)

avg@4(%)

avg@4(%)

OPD

90

48

62

30

88

46

60

28

86

44

58

25 50 75 100

25 50 75 100

25 50 75 100

25 50 75 100

25 50 75 100

Training progress (%)

Training progress (%)

Training progress (%)

Training progress (%)

Training progress (%)

18

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
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

60

40.0

90

48

16

85

55

37.5

14

46

80

avg@32(%)

35.0

50

avg@4(%)

avg@8(%)

avg@4(%)

avg@4(%)

12

###### SFT

44

75

32.5

45

10

70

42

30.0

40

8

65

40

27.5

35

6

60

25.0

38

4

25 50 75 100

25 50 75 100

25 50 75 100

25 50 75 100

25 50 75 100

Training progress (%)

Training progress (%)

Training progress (%)

Training progress (%)

Training progress (%)

- Figure 8: Additional evaluation of rank-constrained training. We compare unconstrained training and rank-16 projected training across five reasoning benchmarks. Across benchmarks, OPD is consistently less affected by the rank-16 bottleneck than SFT. Green diamonds denote the shared anchor checkpoint, and dashed vertical lines denote the projection start.

whereas SFT is substantially more sensitive to the same bottleneck. This supports the conclusion that the early low-dimensional update channel is functionally sufficient for OPD beyond the primary AIME 2024 evaluation.

### G Control Experiment Details

#### G.1 Shared OPD Control Setup

All control experiments in Section 5 are based on the same OPD baseline configuration. Unless otherwise specified, each intervention changes only the stated factor and keeps the remaining OPD training pipeline fixed. Full training hyperparameters are provided in Appendix C.

For all spectral diagnostics, we analyze the same matrix set as in the main trajectory analysis: 144 matrices, corresponding to 36 layers and four attention weight types per layer. Metrics are computed per matrix and then averaged across matrices.

#### G.2 Perturbation Protocols

Token sparsification. This intervention perturbs the density of token-level teacher supervision while keeping the rollout policy and OPD objective fixed. For each sampled response token at, the implementation computes the sampled-token student–teacher log-probability gap

rt = log πθ(at | st) − log πT(at | st),

where πθ is the student policy and πT is the teacher policy. This is a sampled-token log-probability gap, not a full-vocabulary KL divergence.

Item Setting Student model Qwen3-8B Teacher model Qwen3-32B Student initialization Qwen3-8B SFT anchor,

iter_0005375 Training data dapo-math-17k Training length 300 steps Checkpoint steps 63, 127, 191, 255, 299 Baseline rollout Student-generated, 8 sam-

ples per prompt Analyzed matrices 36 layers × 4 attention

weights Primary diagnostic Stable rank Auxiliary diagnostics Frobenius norm, Hill tail

estimate

Table 8 Shared setup for OPD control experiments.

For token sparsification, we introduce a binary token mask mt ∈ {0,1}. In top-KL retention, tokens are ranked by rt and the largest ρ fraction is retained; this selects tokens where the student is most over-confident relative to the teacher. In random retention, mt = 1 for a uniformly sampled ρ fraction of response tokens. We use ρ ∈ {0.25,0.50}.

The retained-token signal is compensated by

T

rt =

mtrt.

T τ=1 mτ

The additive OPD update then uses this gap as a correction term, subtracting it from the advantage. Thus, the top-KL variant selects the tokens

with the largest student–teacher deviation, while the training signal penalizes this deviation. Masking is applied independently for each sample at the response-token level. All other settings, including teacher model, on-policy rollout generation, optimizer, batch size, learning rate, and training data, are kept identical to the OPD baseline. The sparsification runs are evaluated at checkpoints s = {63,127,191,255}.

Off-policy rollouts. This intervention perturbs the rollout policy while keeping the token-level OPD objective fixed. In the baseline, rollouts are generated by the current student policy:

y ∼ πθ(· | x).

In the off-policy variant, rollouts are generated by the teacher:

y ∼ πT(· | x),

and the student then computes log-probabilities on these teacher-generated tokens during the training forward pass. The training objective remains pure OPD KL distillation; no reward mixing is introduced. Thus, this intervention changes the sampling distribution but keeps the teacher-token gradient source fixed.

The off-policy runs use Qwen3-32B teacher rollouts served through an sglang router with two teacher replicas. The importance-ratio clipping threshold is set to ϵ = 1.0, making clipping effectively inactive for this comparison. The off-policy checkpoints are evaluated at s = {63,127,191,255,299}.

Objective interpolation. This intervention perturbs the objective composition while keeping the student rollout policy fixed. We implement this at the advantage level by mixing the OPD teachercorrection signal with the RLVR reward signal:

A(iα) = αAi,OPD + (1 − α)Ai,RLVR, (34)

where Ai,OPD = −ri and ri = log πθ(ai | si) − log πT(ai | si) is the sampled-token student– teacher log-probability gap. Thus, the OPD term encourages correction toward the teacher. Ai,RLVR is the GRPO advantage from the math accuracy reward, broadcast to response tokens. The resulting policy-gradient direction is

gα = Ei A(iα)∇θ log πθ(ai | si) . (35)

We evaluate α ∈ {0.75,0.50,0.25,0.05,0.01}, with α = 1 corresponding to the pure OPD baseline and α = 0 corresponding to the RLVR endpoint. All alpha-mixing runs use raw signal mixing without per-batch standard deviation normalization. The rollout policy remains on-policy, and the optimizer, batch size, learning rate, training data, and evaluation setup are identical to the baseline.

Controlled variables. Table 9 summarizes which factor is changed by each intervention.

#### G.3 Auxiliary Metrics

Stable rank is the primary diagnostic in Section 5 because the question is whether the locked lowdimensional update channel is preserved. We additionally track update scale and spectral shape through Frobenius norm and Hill tail estimates (Figure 9). These auxiliary diagnostics verify that runtime perturbations preserve the OPD-like spectral trajectory up to scale shifts, whereas objective interpolation changes the update geometry more substantially.

Dimension Baseline Token sparse Off-policy Alpha mix Rollout policy Student Student Teacher Student Token coverage 100% 25/50% 100% 100% Objective signal OPD OPD OPD OPD/RLVR mix Reward module OPD OPD OPD Mixed reward Other settings – Same Same Same

Table 9 Controlled variables in Section 5. Each intervention changes one target factor while keeping the remaining OPD setup fixed where possible.

28

- (a)Tokensparsification

Stable rank of ΔW

OPD

Top-KL 50% Top-KL 25% Random 50% Random 25%

0.20

0.25

0.30

0.35

0.40

0.45

‖ΔW‖F

‖ΔW‖F

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

2.3

2.4

2.5

2.6

2.7

2.8

2.9

Hillα

Hill α

18

20

22

24

26

28

Stablerank

- (b)Rolloutpolicy

On-policy Off-policy

0.25

0.30

0.35

0.40

0.45

0.50

‖ΔW‖F

2.3

2.4

2.5

2.6

2.7

2.8

2.9

Hillα

25 50 75 100

Training progress (%)

17.5

20.0

22.5

25.0

27.5

30.0

Stablerank

- (c)Objectivemixing

26

Stablerank

24

22

20

18

3.0

OPD

α=0.75 α=0.50 α=0.25 α=0.05 α=0.01

0.4

2.8

###### ‖ΔW‖F

Hillα

0.3

2.6

0.2

2.4

25 50 75 100

25 50 75 100

Training progress (%)

Training progress (%)

- Figure 9: Auxiliary metrics for control experiments. We report update scale and spectral-shape diagnostics for the same perturbations analyzed in Figure 7. Runtime perturbations preserve the OPD-like spectral profile up to modest scale changes, whereas objective interpolation induces a distinct trajectory.

