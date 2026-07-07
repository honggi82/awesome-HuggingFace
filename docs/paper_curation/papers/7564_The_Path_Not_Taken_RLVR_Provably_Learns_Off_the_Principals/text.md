# arXiv:2511.08567v1[cs.LG]11Nov2025

## The Path Not Taken: RLVR Provably Learns Off the Principals

Hanqing Zhu1,2,∗, Zhenyu Zhang2, Hanxian Huang1, DiJia Su1, Zechun Liu1, Jiawei Zhao1, Igor Fedorov1, Hamed Pirsiavash1, Zhizhou Sha2, Jinwon Lee1, David Z. Pan2, Zhangyang Wang2,†, Yuandong Tian1,†, Kai Sheng Tai1,†

1Meta AI, 2The University of Texas at Austin ∗Work done during an internship at Meta AI., †Equal advisory contribution.

Reinforcement Learning with Verifiable Rewards (RLVR) reliably improves large-language-model reasoning while apparently modifying only a small fraction of parameters. We revisit this paradox and show that sparsity is a surface artifact of a model-conditioned optimization bias: for a fixed pretrained model, updates consistently localize to model-preferred parameter regions, remain highly consistent across runs, and are largely invariant to datasets and RL recipes. We mechanistically explain these dynamics with a Three-Gate Theory: Gate I (KL Anchor) imposes a KL-constrained update; Gate II (Model Geometry) steers steps off principal directions into low-curvature, spectrumpreserving subspaces; Gate III (Precision) hides micro-updates in non-preferred regions, making the off-principal bias appear as sparsity. We then validate this theory and, to our knowledge, provide the first parameter-level characterization of RLVR’s learning dynamics: RLVR learns off-principal directions in weight space, achieving gains via minimal spectral drift, reduced principal-subspace rotation, and off-principal update alignment. In contrast, SFT targets principal weights, distorts the spectrum, and even lags RLVR.

Together, these results provide the first parameter-space account of RLVR’s training dynamics, revealing clear regularities in how parameters evolve. Crucially, we show that RL operates in a distinct optimization regime from SFT, so directly adapting SFT-era parameter-efficient fine-tuning (PEFT) methods can be flawed, as evidenced by our case studies on advanced sparse fine-tuning and LoRA variants. We hope this work charts a path toward a white-box understanding of RLVR and the design of geometry-aware, RLVR-native learning algorithms, rather than repurposed SFT-era heuristics.

Author emails:: hqzhu@utexas.edu

1 Introduction

Large Reasoning Models (LRMs), such as OpenAI-o3 (Jaech et al., 2024) and DeepSeek-R1 (Guo et al., 2025), have advanced the ability of large language models to solve complex mathematical and programming tasks. A key driver is large-scale Reinforcement Learning with Verifiable Rewards (RLVR), which uses simple, easy-to-verify rewards to incentivize complex, multi-step reasoning.

Yet, despite these advances, the mechanisms by which RL shapes model representations and behavior remain poorly understood. Given the substantial computational resources devoted to RL (xAI, 2025), especially relative to supervised fine-tuning (SFT), and the emergence of striking new behaviors, one might naturally assume that such progress arises from significant parameter changes. However, recent evidence points in the opposite direction: RL induces sparse parameter updates, whereas SFT yields dense ones (Mukherjee et al., 2025). This counterintuitive finding reveals a paradox, a high-cost, high-gain process that relies on surprisingly minimal weight modification.

Key observation. We resolve this paradox by uncovering a deeper mechanism behind the apparent sparsity: a persistent, model-conditioned optimization bias. For a fixed pretrained model, this bias concentrates visible updates into a narrow, stable subset of parameters and remains strikingly invariant across diverse algorithms

[Figure 1]

- Figure 1 SFT vs. RLVR: optimization geometry and evidence. (a) SFT follows an externally guided route and traverses high-curvature directions (“over the mountain”) to reach the target. (b) RLVR, without an explicit teacher, behaves as if steered by an implicit compass (a model-conditioned optimization bias), taking a low-curvature detour. (c) Evidence. Left: positional maps comparing the update mask (non-zero parameter updates) with the principal mask (positions aligned with top-k singular subspaces, defined by the largest-magnitude entries of the rank-k SVD reconstruction Liu et al. (2025c); details in Sec. 4.2). RLVR updates avoid principal-weight positions, whereas SFT targets them (Meng

- et al., 2024a; Liu et al., 2025c). Right: principal-angle curves of the top-k subspaces show that RLVR rotates less (spectrum preserved), while SFT rotates more.

and datasets—a model-conditioned feature. bfloat16 precision further accentuates the apparent sparsity by attenuating micro-updates in non-preferred regions. As illustrated in Fig. 1, we depict this bias as an implicit compass: unlike SFT with an explicit teacher, RLVR is subtly guided during optimization even without one.

Research Question. These phenomena raise a central question about RL’s learning dynamics:

Where does this optimization bias originate, and how does it shape parameter evolution during training?

Mechanistic explanation. We formalize the mechanism behind RLVR’s optimization dynamics through a Three-Gate Theory. Gate I (KL Anchor) enforces a KL-constrained update at each on-policy step. Gate II (Model Geometry) then steers this update off the principal directions toward low-curvature, spectrumpreserving subspaces embedded in the structured optimization landscape of a pretrained model, unlike training from a randomly initialized model. This geometry gate explains the model-conditioned nature of the bias: it arises from the pretrained landscape rather than particular datasets or RL recipes. Gate III (Precision) acts as a realization filter by hiding those micro-updates in non-preferred regions, making the off-principal bias appear sparse.

Experimental validation. We validate this theory with a comprehensive suite of experiments, uncovering striking optimization dynamics: RLVR learns off the principal directions, operating in a regime disjoint from SFT’s. We show that (i) RLVR preserves the pretrained spectral structure with , whereas SFT distorts it; (ii) RLVR avoids principal weights—the high-energy directions indicated by rank-k SVD reconstructions—whereas parameter-efficient SFT targets them (Liu et al., 2025c); and (iii) RLVR depends on the pretrained geometry: function-preserving orthogonal rotations abolish the effect of update locality overlap, consistent with a model-conditioned optimization bias.

Rethinking learning algorithms for RLVR. Beyond characterizing learning dynamics, our findings reveal that RLVR operates in a regime fundamentally distinct from SFT. Consequently, direct carry-over of SFT-era parameter-efficient fine-tuning (PEFT) methods can be flawed, especially those over-aligned with SFT’s optimization geometry. (1) Sparse fine-tuning. Restricting updates to principal weights, an SFT prior (Liu et al., 2025c), yields the weakest optimization trajectory and markedly degrades performance, as reflected by both forward-KL drift (Shenfeld et al., 2025) and accuracy. Conversely, updating non-principal, low-magnitude weights, precisely the off-principal regime predicted by our theory, closely tracks the dense RLVR trajectory, validating our parameter-level understanding. (2) LoRA variants. A recent report (Schulman & Lab, 2025) finds that low-rank LoRA (even rank-1) can match full-parameter performance in RL. However, our analysis challenges their belief that advanced LoRA variants such as PiSSA (Meng et al., 2024a) bring further gains: PiSSA explicitly targets principal weights-suitable for SFT but misaligned with RLVR’s off-principal dynamics.

Empirically, PiSSA offers no obvious gain over standard LoRA. Moreover, enforcing principal-direction updates-e.g., via learning-rate scaling, a common requirement to match full-parameter performance with low-rank adapters, often destabilizes training and precipitates early collapse.

Contributions. Our work makes the following key contributions:

- • Observation (Sec. 2). We identify a persistent, model-conditioned optimization bias in RLVR fine-tuning that is largely invariant to datasets and RL variants, yet highly consistent for a fixed pretrained model.
- • Theory (Sec. 3). We propose the Three-Gate Theory—KL Anchor, Model Geometry, and Precision—which mechanistically explains how RL updates are constrained, steered, and filtered to produce the observed optimization pattern.
- • Evidence (Sec. 4). We provide a parameter-level validation contrasting the training dynamics of RL and SFT, including reduced spectral drift, smaller principal-subspace rotation, low overlap with principal weights, and basis-rotation interventions that isolate geometry as the steering core.
- • Insight (Sec. 5). We show that SFT-era sparse and low-rank priors (e.g., principal-targeted variants) are misaligned with RLVR’s off-principal dynamics, motivating geometry-aware, RLVR-native learning algorithms.

Together, these findings provide the first parameter-space account linking RL optimization dynamics to weight evolution, complementing concurrent work that focuses primarily on policy-level or distributional effects (Wu

- et al., 2025; Shenfeld et al., 2025). Crucially, RLVR operates in a distinct, geometry-driven optimization regime from SFT, calling for the development of RL-native, geometry-aware PEFT methods (see Sec. 5) and marking a step toward a white-box understanding of RLVR training.

### 2 A Persistent, Model-Conditioned Optimization Bias in RLVR

We begin with the observation that RL induces sparse parameter updates, and go beyond quantification to ask where RL localizes these changes in order to understand the underlying mechanism. Our analysis reveals a model-conditioned optimization bias: for a fixed pretrained model, RL consistently routes visible updates to specific regions of the network, highly consistent across runs and largely invariant to datasets and RL variants. We further find that the observed sparsity is a superficial readout of this bias, amplified by bfloat16 precision, which attenuates micro-updates in non-preferred regions.

Model suite. We analyze publicly released checkpoints, as shown in Tab. 1. The suite spans multiple RLVR variants (e.g., GRPO, DAPO, Reinforcement++), diverse data domains, and several model families and types (dense and Mixture-of-Experts). We place particular emphasis on DeepSeek-R1-Distill-Qwen-1.5B (DS-Qwen-1.5B), for which a long-horizon RL checkpoint is available (Liu et al., 2025a). This model serves as a robust case study given its extensive training for over 3,000 steps on a diverse data mixture encompassing mathematics, coding, STEM, logic puzzles, and instruction-following tasks.

- 2.1 A Robust, bfloat16-aware Analysis of Update Sparsity

A bfloat16-aware probe for unchanged weights. bfloat16 (bf16) is standard in modern RL frameworks like verl (Sheng et al., 2024), to improve throughput without compromising performance. However, analyzing parameter changes under bf16 requires a careful probe. Its unique numerical format, with only 7 mantissa bits for precision, means that the smallest representable difference between two numbers scales with their magnitude. Consequently, a fixed absolute-tolerance check as used in (Mukherjee et al., 2025), is unreliable, which can over- or under-report the fraction of unchanged weights (see Appendix E.1).

To ensure a rigorous report, we adopt a numerically robust, bfloat16-aware probe to define the update sparsity sparsitybf16 as the fraction of parameters that remain unchanged.

Table 1 Update sparsity in SFT vs. RLVR. Higher sparsitybf16 indicates more weights unchanged. RLVR is consistently much sparser than SFT. † Mixed denotes a diverse data source combining math, coding, STEM, logic puzzles, and instruction-following Liu et al. (2025a).

Base Model Finetuned (FT) Model Algorithm Data sparsitybf16 Qwen-1.5B DS-R1-Distill-Qwen-1.5B SFT Mixed 2.8% DS-R1-Distill-Qwen-1.5B DeepScaleR-1.5B-Preview GRPO Math 53.8% DS-R1-Distill-Qwen-1.5B DeepCoder-1.5B-Preview GRPO Code 45.5% DS-R1-Distill-Qwen-1.5B Archer-Code-1.5B GRPO Code 52.5% DS-R1-Distill-Qwen-1.5B NV-ProRL GRPO Mixed† 38.4% DS-R1-Distill-Qwen-1.5B NV-ProRL-v2 Reinforcement++ Mixed† 36.3% Qwen3-8B-Base Klear-Reasoner-8B-SFT SFT Math+Code 0.6% Klear-Reasoner-8B-SFT Klear-Reasoner-8B GRPO Math+Code 69.5% Qwen3-8B-Base GT-Qwen3-8B-Base GRPO Math 79.9% Qwen3-8B-Base OURS DAPO Math 79.7% Qwen3-14B-Base UniReason-Qwen3-14B-think-SFT SFT Math 18.8% Qwen3-14B-Base UniReason-Qwen3-14B-RL GRPO Math 68.3% Qwen3-4B Polaris-4B-Preview DAPO Math 79.3% DS-R1-Distill-Qwen-7B Polaris-7B-Preview DAPO Math 61.7% Qwen3-30B-A3B UloRL-A3B GRPO Math 91.7%

#### Definition 2.1 (Unchanged Weight in bf16). Let wi,ŵi ∈ R be scalars stored in bf16 (finite, nonzero). We say wi is unchanged with respect to ŵi iff

∣̂wi − wi ∣ ≤ η max(∣wi∣, ∣̂wi∣), η = 10−3. (1)

Choosing η=10−3 < 2−9 makes equation 1 equivalent to bitwise equality (See Appendix E.2,).

#### Definition 2.2 (bf16-aware Update Sparsity). Write x ≈bf16η y for Def. 2.1. Define the bf16 change count ∥θ1 − θ0∥bf160,η ∶= ∣{i ∶ θi1 ≈/bf16η θi0 }∣ and the corresponding sparsity

sparsitybf16(θ0,θ1;η) ∶= 1 −∥θ1 − θ0∥bf160,η /n. (2)

where n is the total number of parameters. Values near 1 indicate few stored changes, while values near 0 indicate dense apparent change.

RLVR update sparsity results. As shown in Tab. 1, our analysis confirms that RL yields substantially higher update sparsity than SFT. Across models, SFT sparsity is consistently low (typically 0.6%–18.8%), whereas RL sparsity is an order of magnitude higher, ranging from 36% to 92%. However, absolute levels on recent checkpoints are lower than earlier reports (Mukherjee et al., 2025), underscoring the need for bf16-aware probes and re-evaluation on current models.

- 2.2 RLVR Exhibits Model-Conditioned Update Locality Magnitude alone does not reveal where changes occur, impeding deep analysis of how sparse changes arise.

We therefore examine the updated subnetwork. We use 5 independent RLVR checkpoints from the same DSQwen-1.5B in Tab. 1, trained on different datasets and RLVR algorithms. For each layer ℓ and run r, we first

Table 2 Cross-run stability for 13th block.

Layer Jaccard Overlap Random Baseline

Q 0.580 0.430 K 0.580 0.413 V 0.597 0.467 O 0.552 0.373 MLP-down 0.585 0.453 MLP-up 0.578 0.443 MLP-gate 0.575 0.437

form the bf16-aware changed mask Mℓ(r) ∶= 1[Wℓ(r) ≈/bf16η Wℓ0 ] (Def. 2.2) against the base weights Wℓ0.

Stability across runs (Is the bias persistent?) We first analyze their spatial agreement using Jaccard Overlap.

For runs r,s, let A ={(i,j)∶ Mℓ,ij(r) = 1} and B ={(i,j)∶ Mℓ,ij(s) = 1}. We report the mean off-diagonal of the

pairwise Jaccard matrix J(A,B)= ∣∣AA∩∪BB∣∣ and compare it to the independent Bernoulli baseline E[J]= p+pqq−pq. As summarized in Tab. 2, Jaccard is consistently high across runs, confirming a shared footprint when trained from the same base model, with Jaccard matrix shown in Fig. 12.

1.0

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

0.8

K proj

###### ConsensusRatio

0.6

[Figure 6]

0.4

V proj

[Figure 7]

0.2

0

Q proj

MLP down proj

O proj

- Figure 2 Consensus ratio of weight updates. Across five RLVR runs, we plot the 13th layer’s projections (Q/K/V/O) and the MLP down projection. Lighter bands mark coordinates updated in most runs, revealing a stable, stripe-like routing pattern rather than random scatter (zoom in to see fine structure). Consensus ratio (Where do updates land?) Stability alone does not indicate where updates land. We therefore

visualize and analyze the consensus ratio Cℓ,ij = R1 ∑Rr=1 Mℓ,ij(r), the fraction of runs realizing a weight update at coordinate (i,j). Values near 1 indicate that all runs consistently change that weight; values near 0 indicate that none do. As shown in Fig. 2, consensus maps reveal contiguous row/column bands, stripe-like, localized routing rather than scattered noise. Especially, there are obvious row-wise stripes in Q/K/V projections and column-wise stripes in O projections. This exposes a clear optimization bias: RLVR consistently concentrates updates in specific regions of the parameter matrices for a fixed pretrained model, even though the five runs use disjoint data and RL variants.

Temporalstability(Howdoesthebiasemergeovertime?) To examine within-run dynamics, we track the row-wise ratio ρℓ,i(t)= n1

∑i Mℓ,ij(t) across checkpoints at t steps. On DS-Qwen-1.5B (training setting in Appendix C.1), the relative profiles ρℓ,⋅(t) and κℓ,⋅(t) remain aligned while

∑j Mℓ,ij(t) and column-wise ratio κℓ,j(t)= m1

ℓ

ℓ

overall density grows as shown in Fig. 3: peaks and troughs persist. The routing bias emerges early and is reinforced over training, indicating a temporally stable phenomenon rather than a transient artifact. Moreover, the peak is consistent with the bias structure shown in Fig. 2. We also show their remaining column-wise (Q) and row-wise (O) update ratio dynamics in Fig. 14, without a clear trend, indicating the bias is indeed structured, not random.

[Figure 8]

Figure3 Temporalemergenceoftheoptimizationbias with row and column-wise update ratios for the 13th attention block across gradient update steps (t∈{240,720,1200}), smoothed with a 3-step window. The row-dominant (Q) and column-dominant (O) patterns are consistent with the bias structures in Fig. 2. We visualize the head boundaries with grey dashed lines. The bias appears not only across heads but also within heads.

Across model families:(Is the bias generic?) We observe similar stripe-structured footprints on Llama and Mistral (Fig. 13 in Appendix), suggesting the routing bias is generic to RLVR.

- 2.3 Sparsity Is a Superficial Artifact of the Optimization Bias

The stable footprint of where updates land, persisting both throughout training and in the final model, suggests the focus should move from sparsity itself to the underlying optimization bias.

We find that sparsity is actually the readout of this optimization bias, whose visibility is amplified by the precision limits of bf16 storage. Because bf16 has a limited mantissa, changes smaller than the unit-in-the-lastplace (ULP) threshold (Lemma E.2) are not representable. Therefore, if RLVR consistently routes sub-ULP updates toward a particular subset of parameters, the stored values will not change, and the result appears as sparsity.

We test this hypothesis by increasing the learning rate to scale otherwise sub-ULP updates above the representable threshold. As predicted, the apparent update sparsity largely disappears. This directly challenges the interpretation of (Mukherjee et al., 2025) that sparsity stems from zero gradients. Consistent with this view, concurrent work observes that sparsity mostly vanishes under float32 storage (Shenfeld et al., 2025) by increasing the precision, even though task performance does not improve. Hence, our results point to sparsity as a byproduct of an optimization bias interacting with finite precision.

Clarification on precision. It may be tempting to blame precision limit for sparsity. In fact, verl keeps optimizer states and gradient reductions/accumulation in float321. Thus, sparsity cannot be explained by precision alone. It requires a consistent optimization bias during RL that concentrates visible changes in specific parameter regions throughout training.

Aha Finding! RLVR exhibits a persistent, model-conditioned optimization bias in where updates land—highly consistent across runs and largely invariant to datasets and RL recipes. The observed sparsity is a superficial readout of this bias, amplified by bf16 precision.

### 3 A Mechanistic Theory of RL’s Unique Optimization Dynamics

#### In the post-training era, RL has become a key stage, albeit with intensive compute (xAI, 2025). Paradoxically (Sec. 2), these gains arise not from broad parameter changes but from selective, patterned edits that reveal a persistent optimization bias. Understanding this distinctive training behavior raises the central question:

Where does this optimization bias originate, and how does it shape parameter evolution?

We characterize these optimization dynamics with the Three-Gate Theory, KL Anchor, Model Geometry, and Precision, which mechanistically explains how on-policy RL updates are constrained via Gate I (KL Anchor; Sec. 3.1), steered via Gate II (Model Geometry; Sec. 3.2), and filtered via Gate III (Precision; Sec. 3.3) into the observed update pattern.

Notations.over possibleWeoutputconsidertokena largesequenceslanguageymodel=(y1,...,ywith parametersT)∈Y givenθ, defininga prompta conditionalx ∈X fromdistributionthe space Xπθ.(yEach∣ x) sequence y is composed of tokens from a vocabulary V of size N.

- 3.1 Gate I: On-Policy RL Imposes a One-Step KL Leash

We first show that online policy gradient updates yield a per-step policy KL bound (an anchoring effect), which in turn limits parameter movement during the RLVR update.

RLVR objective. Various RLVR algorithms including PPO, GRPO, DAPO, and REINFORCE++, learn a policy πθ by optimizing variants of a KL-regularized objective:

Ey∼πθ(⋅∣x),x∼X[R(x,y)− βKL(πθ(⋅∣ x)∥πref(⋅∣ x))]. (3)

max

θ

where πref is a fixed reference policy and β ≥ 0 controls the KL regularization (β = 0 recovers the clip-only variants such as DAPO). Rewards R(x,y) are verifiable and (after normalization) bounded (e.g., pass/fail or

θ(yt∣x,y<t) πold(yt∣x,y<t)

execution scores). Moreover, the surrogate typically uses the token-wise importance ratio wt = π

with clipping relative to πold. One-step surrogate. With equation 3, a standard sequence-level online policy-gradient surrogate is

LPG(θ)=−Ex∼X, y∼πθ(⋅∣x)[A⊥(x,y) logπθ(y ∣ x)], (4)

where A⊥ is a (normalized) advantage estimate, optionally shaped by a reference-KL log-ratio term. In practice, updates are performed over mini-batches, with a collected batch of data, not in a fully on-policy manner. But the resulting error after a small step size ∆θ is O(∥∆θ∥2) (Lemma F.1).

1verl mixed-precision settings with {reduce_type, buffer_dtype}=float32.

Implicit KL leash. The KL leash emerges as policy gradient methods can be understood as a conservative projection, keeping new policy close to its starting point while reweighting it toward higher-reward outcomes, not pulling it toward a potentially distant external distribution like SFT:

Propositionx)exp(R/β) denote3.1 (One-stepthe soft-regularizedpolicy-KL leash)improvement. Let q(⋅∣oracle.x) be aLetfull-supportθ+ be thereferenceparametricandfitletobtainedq˜β(⋅∣ x)∝byqthe(⋅∣ M-projection of q˜β onto the policy class, θ+ ∈ argminθ DKL(q˜β∥πθ). Then, for a sufficiently small one-step update,

DKL(πθ+ ∥ πθ) ≤ (1 + o(1)) DKL(q˜β ∥ πθ), (5)

where the o(1) term vanishes as DKL(q˜β∥πθ)→ 0.

Notably, even when the explicit KL term is removed (e.g., in DAPO with β = 0), the ratio clipping trick still imposes a KL bound O(ε2) in the small-step regime (Appendix. F.2.4), confirmed empirically with a bounded KL divergence change during a DAPO run (Fig. 15).

Weight update constraint. Now we show the KL leash puts a constraint on the weight update ∆W

Proposition 3.2 (Policy-KL leash ⇒ weight bound). Assume logπθ is C3 and let F(θ) denote the Fisher information. If a one-step update θ+ = θ+∆ satisfies DKL(πθ+∥πθ)≤ K and, on the update subspace, F(θ)⪰ µI for some µ > 0, then for K sufficiently small

√2K

##### √∆⊺F(θ)∆ ≤

√2K (1 + o(1)), ∥∆∥2 ≤

µ (1 + o(1)). (6)

∥∆∥F(θ) ≜

√2K/µ(1 + o(1)).

Consequently, for any weight matrix block W ⊂ θ, ∥∆W∥F ≤

See a detailed proof for Proposition 3.1 in Appendix F.2.1 and Proposition 3.2 in Appendix F.2.2.

Take-away 1: RL update imposes an implicit KL leash (anchor effect), ensuring that the per-step drift from the current policy is small. This aligns with recent work arguing that even the final policy is KL-proximal (Wu et al., 2025; Shenfeld et al., 2025). Our focus, however, is to understand how this leash affects the weight change dynamics.

- 3.2 Gate II: Model Geometry Determines Where a KL-Bounded Step Goes

From Gate I to location. Gate I supplies a one-step KL leash that bounds the move, but it does not specify where the update lands. We propose Gate II(Model Geometry), where we argue that, unlike a randomly initialized network, a well-pretrained model possesses a highly structured geometry, e.g., spectral statistics and high-curvature directions during optimization, that determines where a KL-constrained update goes.

LayerwisenormboundfromtheKLleash. Let W0 be a pretrained linear block, W+ = W0+∆W the post-step block, andδW, letthenSW(Appendix⪰ µWI be F.10a per-layer) curvature proxy. If the per-layer KL budget satisfies 12⟨vec∆W,SW vec∆W⟩≤

√

√

∥∆W∥F ≤

2δW µW , ∥∆W∥2 ≤

2δW µW . (7)

We then show that this conservative update yields three consequences, preserving the pretrained weight spectrum rather than destroying it based on weight perturbation theory (Stewart, 1998).

Limited subspace rotation. First, as shown in Theorem 3.3, the angle between the original and updated subspaces is quadratically bounded, meaning the fundamental directions are preserved.

Theoremσk(W0)− σk3.3+1(W(Constrained0) be the singularsubspacevaluerotationgap. Forwithany kWedin’swith γksin–> 0,Θ theorem (Wedin, 1972)). Let γk ∶=

√2δW/µW γk

∥∆W∥2

max(∥sinΘ(Uk(W0),Uk(W+))∥2, ∥sinΘ(Vk(W0),Vk(W+))∥2 ≤

γk ≤

. (8)

#### Singular value stability. Second, the magnitudes of the principal components themselves are preserved. The change in each singular value is bounded by the norm of the update.

Spectrum Drift (all-layer)

Principal Angle (single-layer)

Singular Value (single-layer)

Max Principal Angle (all-layer)

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

RL

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

SFT

- Figure 4 Spectral geometry under SFT vs. RLVR on Qwen3-8B (Su et al., 2025). Left: for an exemplar layer, top-k principal angles and singular-value curves. Right: across all layers, maximum principal angle and normalized spectral drift. RLVR maintains a stable top-k spectrum with minimal subspace rotation, unlike SFT. See DS-Qwen-1.5B in Fig. 16 and Qwen3-14B-Base in Fig. 17.

- Corollary 3.4 (Singular-value stability). For each k,

∣σk(W+)− σk(W0)∣ ≤ ∥∆W∥2 ≤

√2δW µW

, ∑

i

(σi(W+)− σi(W0))2 ≤ ∥∆W∥2F ≤

2δW µW

. (9)

Top-k energy preservation. Finally, these effects combine to ensure the cumulative energy of the top-k components of the weights remains stable.

- Corollary 3.5 (Top-k energy and Ky Fan norms). Let ∥⋅∥(k) ∶= ∑ki=1 σi(⋅) be the Ky Fan k-norm. Then

∣∥W+∥(k) −∥W0∥(k) ∣ ≤

k

∑

i=1

∣σi(W+)− σi(W0)∣ ≤ k ∥∆W∥2 ≤ k

√

2δW µW . (10)

See the detailed proofs in Appendix F.3 for all results presented here.

Take-away 2: Under the KL leash, RL updates tend to preserve the model’s original weight structure rather than destroy it. This naturally favors updates in low-curvature directions of the optimization landscape, which avoids dramatic changes in model behavior. Since directly quantifying curvature in LRM with long CoTs is computationally prohibitive, we instead adopt a powerful and efficient proxy, principal weights (Liu et al., 2025c), as detailed in Sec. 4.2.

3.3 Gate III: Precision Acts as a Lens Revealing the Compass

Building on the optimization bias, the bfloat16 with limited precision acts as a lens: it hides those micro-updates that occur where the RL consistently holds a weak willingness to apply large changes.

- Corollary 3.6 (Magnitude-dependent realization threshold). A stored weight Wij changes at a step iff ∣∆Wij∣≳ 12 ULPbf16(Wij). The effect of this gate has been discussed aforementioned. We would emphasize again that precision is more an amplifier for visible sparsity, not the cause of optimization bias, as optimizer states, etc., are still in float32 (See Sec. 2.3).

- 4 Theory-Guided Validation of RLVR’s Optimization Dynamics

We conduct theory-guided experiments analyzing how RLVR modifies parameters and interacts with pretrained geometry. These results validate our central prediction: the pretrained model geometry steers KL-constrained

[Figure 17]

- Figure 5 RL avoids updating principal weights. We compare the RL update mask with principal weight mask Mprinc, low magnitude mask Mlow, and the one Mprinc ∩ Mlowc . The layer-wise overlap between RL updates and principal weights is consistently sub-random, an effect more pronounced when removing its overlapped weights with Mlow, i.e., Mprinc ∩ Mlowc . updates, yielding distinct, off-principal optimization dynamics that set RLVR apart from SFT.

- 4.1 RLVR Preserves Spectral Geometry, While SFT Distorts It

We begin by probing spectral changes to test whether RL updates are steered toward low-curvature, spectrumpreserving directions. If so, RLVR should largely preserve the pretrained spectral structure, whereas SFT, lacking this steering, should significantly distort it.

Setups. We analyze checkpoints from a standard SFT→RLVR pipeline on Qwen3-8B-Base (Su et al., 2025) and a long-horizon RL run on DS-Qwen-1.5B (Liu et al., 2025a). We also consider a setting where SFT and RL are applied separately to Qwen3-14B-Base, matched on in-domain math performance (Huan et al., 2025). In all cases, we compare base weights W0 and fine-tuned weights W+.

Metrics. We compare the base weights W0 with the finetuned weights W+:

- • Subspace rotation. For the top-k left (U)/right(V ) singular subspaces, we check the rotation using principal angles via cosθi(U)∶= σi(U0⊺,kU+,k) and cosθi(V )∶= σi(V0⊺,kV+,k).
- • Spectrumdrift. Beyond showing the singular value curve, we quantify singular-value change with a normalized ℓ2 shift: NSS(W)=∥σ(W+)− σ(W0)∥2/∥σ(W0)∥2

Our findings. RLVR checkpoints exhibit a Insightably stable spectrum within the top principal components: across layers, RLVR shows consistently small principal-subspace rotation and minimal spectral drift. The singular-value profiles are even nearly identical to the base model. By contrast, SFT induces substantially larger rotations and pronounced drifts on the same metrics (Fig. 4).

- 4.2 RLVR Avoids Principal Weights, While SFT Targets Them

We now move from macro-level spectral analysis to a micro-level examination of individual weights, probing which parameters RLVR favors or avoids to update, a deeper investigation into the parameter-space dynamics.

Principal weights as a proxy for high-curvature directions. Directly identifying high-curvature directions is computationally prohibitive, especially given LRM with long CoTs. Instead, we adopt a powerful proxy from recent work Liu et al. (2025c), principal weights, which is defined as the weights with the largest magnitude after low-rank approximation, representing its most influential computational pathways. The validity of this proxy is confirmed by their perturbation studies, which show that modifying these specific weights causes sharp reasoning performance degradation. This degradation is directly linked to high-curvature regions via

a Taylor expansion of the loss. The principal mask, Mprinc(k) = Topα(s(ijk)), is defined as the top-α fraction of weights with the highest score, s(ijk) =∣W0(k)(i,j)∣, where W0k is the rank-k SVD reconstruction of W0.

Low-magnitude weights as low-resistance pathway. We further include the top-α lowest magnitude weights, as Mlow = Bottomα(∣W0∣). The magnitude is also a bias from the model geometry (distribution prior), impacting how easily the weights can be updated based on our precision gate.

Metrics. Let M be the weight update update mask from an RLVR run. We report the overlap ratio between our identified mask M● with it, defined as Overlap(M●,M) = ∣M

●∩M∣ ∣M∣ ., with a random guess baseline overlap

ratio as the density of M● itself., i.e., α.

Our findings. Fig. 5 visualizes the RL update mask M in relation to the principal mask Mprinc and the low-magnitude mask Mlow, reporting their layer-wise overlap against a random baseline as well. The results show a clear dichotomy. RL updates exhibit a sub-random overlap with principal weights, indicating a strong tendency to avoid them. Conversely, the updates show a super-random overlap with low-magnitude weights due to their low resistance to micro-updates. Besides, we found that the residual overlap between updates and principal weights is highly accounted for by weights that are both principal (defined by the rank-k approximation of W0) and low-magnitude (original W0). After excluding this intersection, i.e., Mprinc ∩ Mlowc , the overlap drops significantly.

Insight. This points to a central implication: RLVR and SFT operate in distinct optimization regions of

parameter space, even at comparable task performance. RLVR avoids high-curvature, principal regions, whereas SFT targets them. This regional mismatch helps explain the limited transferability of SFT-oriented PEFT under RL (Sec. 5).

- 4.3 RLVR Relies on Model Geometry, Disrupting Geometry Destroys the Bias

[Figure 18]

Figure 6 Overlap ratio after intervention.

Gate II posits that the pretrained model’s geometry steers RL updates. To test this causal link, we deliberately "scramble" the geometry of specific layers in a Qwen3-4B-Base model using orthogonal rotations for O/V layers (Rotate) and head permutations for all Q/K/V/O layers (Permute) (details in Appendix D) and compare the update overlap ratio Overlap(M●,M) = ∣M

●∩M∣ ∣M∣ . between

the base run with another independent run without intervention and one run with intervention.

Our Findings. We modify (i) layer 20 with Rotate+Permute, and (ii) layer 25 with Rotate. As shown in Fig. 6, the update overlap collapsed to a random level in the intervened layers, while remaining high in all untouched layers. This provides strong causal evidence that the pretrained model’s geometry is the source of the optimization bias.

- 4.4 RLVR signatures persist in agentic tasks and RLHF

Setup. We analyze additional agent and RLHF (RL with human feedback) checkpoints and apply the same weight–space diagnostics as in Sec. 4.1 and Sec. 4.2: (i) principal-subspace rotation, (ii) spectral drift, and (iii) update–principal misalignment. The extended model suite is summarized in Tab. 3. Agents. We evaluate policies from AgentFlow (Li et al., 2025) and VERL-Agent (Feng et al., 2025) on multi-turn and longhorizon tasks. We also assess tool-augmented agents from SkyRL (Cao et al., 2025) and VERL-Tool (Jiang et al., 2025) on WebSearch, DeepSearch, and SWE. RLHF. We include preference-optimized models trained with DPO (Rafailov et al., 2023a) and SimPO (Meng et al., 2024c), primarily targeting instruction following.

[Figure 19]

- Figure 7 Update–principal misalignment in RL-trained agents. For representative layers from agentflow-planner-7b (Layer 16, o_proj; top row) and SkyRL-Agent-WebResearch-8B (Layer 11, k_proj; bottom row), the left panels visualize the bf16-aware update mask Mℓ (locations that changed under RL), while the right panels show the principal

mask P(k)

ℓ (top-k singular-subspace support typically favored by SFT). Dashed red boxes highlight stripe regions where RL updates concentrate outside principal-weight bands, indicating robust off-principal routing in agent and tool-use settings.

Table 3 Model List for analyzed checkpoints for agentic tasks and RLHF algorithms.

Category Base Model FT Model Algorithm Data

Qwen3-8B SkyRL-Agent-WebResearch-8B GRPO WebResearch Qwen3-8B VT-deepsearch-8B GRPO Deepsearch Qwen3-8B VT-SWE-8B GRPO SWE Qwen2.5-7B-Instruct agentflow-planner-7b Flow-GRPO Planning Qwen2.5-7B-Instruct GiGPO-Qwen2.5-7B-Instruct-WebShop GiGPO WebShop Qwen2.5-7B-Instruct GiGPO-Qwen2.5-7B-Instruct-ALFWorld GiGPO ALFWorld

Agent

Meta-Llama-3-8B-Instruct Llama-3-Instruct-8B-DPO DPO instruction-following Meta-Llama-3-8B-Instruct Llama-3-Instruct-8B-SimPO SimPO instruction-following

RLHF

OurFindings. (i)Stablespectra,minimalrotation. Across agents and RLHF, top-k subspaces rotate only slightly, and layer spectra remain near-identical to the base model (Fig. 8; Fig. 18), matching the spectrum-preserving, off-principal regime observed earlier. (ii) Off-principal updates. Update masks in agent and RLHF checkpoints consistently avoid principal weights: the most active bands are spatially misaligned with the principal mask (Fig. 7). Takeaway. RLVR’s optimization dynamics—minimal rotation, spectrum preservation, off-principal routing—persist beyond verifiable math/code to agents and RLHF, indicating a common, model-conditioned optimization bias within a KL-anchored RL post-training game, consistent with our Three-Gate Theory.

Takeaway 3. RLVR learns off the principals: it preserves spectral geometry, avoids principal weights, and its optimization bias vanishes when pretrained geometry is disrupted.

- 5 Theory-Guided Rethinking of Learning Algorithms for RL

A good theory should not only explain observations but also inform design. Our account shows that RLVR and SFT follow disjoint optimization dynamics in parameter space, which implies that many SFT-era PEFT methods, especially those aligned with principal directions through sparse or low-rank priors, transfer poorly to RLVR. This section validates our predictions and demonstrates how they guide the redesign of learning algorithms for RL.

[Figure 20]

##### Figure 8 Spectrum under RL in agent tasks. In agent settings, including multi-turn interactions and tool use, RL leaves layer singular-value spectra nearly unchanged and induces only small rotations of the top-k singular subspaces, consistent with the spectrum-preserving, off-principal RLVR regime. Results for RL with human feedback (RLHF), which exhibit the same optimization signature, appear in Fig. 18. For consistency, we use the second block O-projection layer as an exemplar single-layer readout.

- 5.1 Probing Sparse Fine-Tuning in RL

We use sparse RL fine-tuning to probe RL’s optimization dynamics by asking which weights can be frozen without materially altering the training trajectory. We construct a parametermask directly from the pretrained model without any additional training and apply it to perform sparse RL fine-tuning. Following (Shenfeld et al., 2025), we track the token-wise forward KL divergence KL(π ∥πref) between the fine-tuned policy and the base model throughout training. This metric quantifies how closely a sparse run follows the dense baseline trajectory, if pruning certain weights impedes learning, the KL drift will slow, indicating blocked optimization progress.

0 100 200 300 Training Steps

0.0000

0.0025

0.0050

0.0075

0.0100

0.0125

KLLoss

Dense

Mprinc Mprincc

Mlow

Mlow ∪ Mprincc

rand-Mlow ∪ Mprincc

Figure 9 KL loss curves on DS-Qwen-1.5B under different masks.

Mask design. We evaluate several masks constructed directly from the pretrained model: (i) Mprinc (principal-only, top-50% principal weights), (ii) Mprincc (non-principal-only, the complementary subspace), (iii) Mlow (low-magnitude-only), (iv) Mlow ∪ Mprincc (safe mask, favoring non-principal and low-magnitude weights), , and (v) a random mask with the same layer-wise sparsity as (iv). We choose 50% for (i) as we want to isolate the effect of the number of parameters for a fair comparison to see the difference between (i) and (ii).

Our findings. (See KL in Fig. 9; accuracy in Tab. 4 and the extended 500-step results in Tab. 5.) The safe mask Mlow ∪ Mprincc most closely tracks the dense RLVR KL curve and reaches comparable final accuracy, indicating that our theory correctly identifies highly touchable weights. By contrast, the principal-only mask yields the worst optimization trajectory—its KL curve rises slowly—showing excessive intervention and degraded training dynamics. This directly confirms that principal-targeted directions favored in SFT are ineffective for RL.

Insight. Our results suggest a simple yet effective alternative: Freezing principal and large-magnitude weights while updating non-principal, low-magnitude ones closely reproduces dense RLVR behavior (KL trajectory and final accuracy) using roughly 70% the parameters This shows that our theory provides practical guidance for identifying the effective subspace of RL updates, entirely without additional training. While the masks used here are one-shot and fixed, combining this framework with dynamic mask refresh or adaptive scheduling (Zhao

- et al., 2024; Zhu et al., 2024; Liu et al., 2025c) is a promising next step.

- 5.2 Revisiting LoRA Through the Lens of Our Theory

A recent report (Schulman & Lab, 2025) finds that low-rank LoRA, even rank-1, can match full-parameter RL performance. Our theory offers an explanation: in full-parameter RL, effective updates lie off the principal directions and induce only small spectral changes. Low-rank adapters can approximate these off-principal updates, while freezing the base weights regularizes training and discourages moves toward principal directions. With an appropriately scaled learning rate, the limited adapter capacity is therefore sufficient to catch up to full-parameter performance at least in the short run.

However, the same report suggests principal-targeted variants such as PiSSA (Meng et al., 2024a) should yield further gains. Our geometry account disagrees: aligning updates to top-r principal directions enforces SFT-style behavior that is misaligned with RLVR’s off-principal bias.

Empirical test. On DS-Qwen-1.5B with DeepMath-103K (He et al., 2025), we sweep ranks {8,32,64} and learning rates {1×10−4, 5×10−5, 1×10−5} for 200 steps, and report pass@1 (mean over 16 samples) on AIME24 and AMC23 (Fig. 10). To control for model effects, we repeat on Llama-3.2-3B-Instruct with a Math corpus and report pass@1 (mean over 4) on MATH500 (Fig. 11).

[Figure 21]

- Figure 10 LoRA vs. PiSSA onDS-Qwen-1.5B(DeepMath-103K). We sweep ranks {8,32,64} and learning rates {1×10−4,5× 10−5,1×10−5} for 200 steps, reporting pass@1 (avg@16) on AIME24 (top) and AMC23 (bottom). Across settings, PiSSA (principal-targeted) provides no additional gains over LoRA and, at higher learning rates that force principal-direction updates, often collapses early; LoRA remains more stable. This supports our geometric account: forcing updates into principal directions (favored in SFT) is misaligned with RL, offering no obvious gain and leading to training collapse when scaling up learning rates.

Our findings. Across settings, the principal-targeted PiSSA provides no clear gain over LoRA. At the higher learning rates used for low-rank adapters to match full-parameter performance, PiSSA often becomes unstable and collapses earlier than LoRA. This occurs because scaling the learning rate in PiSSA enforces updates along principal directions, higher-curvature and spectrum-distorting, precisely the directions RLVR tends to avoid. The result is brittle optimization and early collapse, whereas LoRA’s off-principal updates remain better aligned with RLVR’s geometry.

Insight. These results support the geometry-based account: principal-aligned LoRA variants are over-fit to SFT’s update geometry and misaligned with RL’s training dynamics, so success in SFT does not transfer to RL.

Takeaway 4. RLVR operates in a distinct, geometry-driven optimization regime, so your old PEFT tricks may not work. Methods over-aligned with SFT’s principal-targeted dynamics (e.g., sparse or low-rank adapters) fail to transfer, calling for a new generation of RL-native, geometry-aware parameter-efficient algorithms.

- 6 Conclusion

We revisited the paradox of visible update sparsity in RLVR and showed that it is a superficial readout of a deeper, model-conditioned, geometry-aligned optimization bias that determines where updates land. We formalized this mechanism with the Three-Gate Theory: a KL anchor constrains each on-policy step; pretrained geometry steers updates off principal directions into low-curvature, spectrum-preserving subspaces; and finite precision renders the bias visible as sparsity by masking micro-updates. Empirically, RLVR preserves spectral structure and avoids principal weights, whereas SFT targets principal directions and distorts the spectrum; when the pretrained geometry is disrupted, these signatures vanish, establishing geometry as the steering core. Beyond explanation, our case studies bridge mechanism and practice: SFT-era principal-aligned PEFT (e.g., sparse/low-rank variants) often misaligns with RLVR’s off-principal regime. Taken together, these results provide the first parameter-level account of RLVR’s training dynamics, replacing a black-box view with a

MATH500

Accuracy(mean@4)

- lr=1e-4 LoRA

- lr=1e-4 PiSSA

lr=5e-5 LoRA

lr=5e-5 PiSSA

- lr=1e-5 LoRA

- lr=1e-5 PiSSA

- lr=1e-6 Full-ﬁnetune

0.5

0.4

0.3

0 50 100 150 200 Training Steps

- Figure 11 LoRA vs. PiSSA on LLaMA-3.2-3B. We sweep learning rates {1×10−4,5×10−5,1×10−5} with a fixed rank of 64 for 200 steps, reporting pass@1 (mean@4) on MATH500. Consistent with the DS-Qwen-1.5B results in Fig. 10, PiSSA provides no additional gain over LoRA and, under higher learning rates that emphasize principal-direction updates, often collapses early.

white-box understanding of how parameters evolve under RLVR, and laying the foundation for geometry-aware, RLVR-native parameter-efficient learning algorithms.

Acknowledgment

We thank Zhengqi Gao (Massachusetts Institute of Technology) for insightful discussion on the verl framework and idea discussion.

References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Shiyi Cao, Sumanth Hegde, Dacheng Li, Tyler Griggs, Shu Liu, Eric Tang, Jiayi Pan, Xingyao Wang, Akshay Malik, Kourosh Hakhamaneshi, Richard Liaw, Philipp Moritz, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Skyrl-v0: Train real-world long-horizon agents via reinforcement learning. https://novasky-ai.notion.site/skyrl-v0, 2025. NovaSky AI, Notion page. Accessed 2025-11-10.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https: //arxiv.org/abs/2501.12948.

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. arXiv preprint arXiv:2002.06305, 2020.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training. arXiv preprint arXiv:2505.10978, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, and Junxiao Song. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.07570, 2025. URL https://arxiv.org/abs/2501.07570.

Seungwook Han, Jyothish Pari, Samuel J Gershman, and Pulkit Agrawal. General reasoning requires learning to reason from the get-go. arXiv preprint arXiv:2502.19402, 2025.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob

Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. arXiv preprint

arXiv:1801.06146, 2018.

Yafei Hu, Quanting Xie, Vidhi Jain, Jonathan Francis, Jay Patrikar, Nikhil Keetha, Seungchan Kim, Yaqi Xie, Tianyi Zhang, Hao-Shu Fang, et al. Toward general-purpose robots via foundation models: A survey and meta-analysis. arXiv preprint arXiv:2312.08782, 2023.

Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, and Xiang Yue. Does math reasoning improve general llm capabilities? understanding transferability of llm reasoning. arXiv preprint arXiv:2507.00432, 2025.

Aaron Jaech et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, et al.

Verltool: Towards holistic agentic reinforcement learning with tool use. arXiv preprint arXiv:2509.01055, 2025.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al. Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision, 16(1-2):1–214, 2024.

Zhuofeng Li, Haoxiang Zhang, Seungju Han, Sheng Liu, Jianwen Xie, Yu Zhang, Yejin Choi, James Zou, and Pan Lu.

In-the-flow agentic system optimization for effective planning and tool use. arXiv preprint arXiv:2510.05592, 2025. Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman,

Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025a.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Zihang Liu, Tianyu Pang, Oleg Balabanov, Chaoqun Yang, Tianjin Huang, Lu Yin, Yaoqing Yang, and Shiwei Liu. Lift the veil for the truth: Principal weights emerge after rank reduction for reasoning-focused supervised fine-tuning. arXiv preprint arXiv:2506.00772, 2025c.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Michael Luo, Sijun Tan, Roy Huang, Xiaoxiang Shi, Rachel Xin, Colin Cai, Ameen Patel, Alpay Ariyak, Qingyang Wu, Ce Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepcoder: A fully open-source 14b coder at o3-mini level. https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level1cf81902c14680b3bee5eb349a512a51, 2025a. Notion Blog.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025b. Notion Blog.

MAA. American mathematics contest 12 (amc 12), November 2023. URL https://artofproblemsolving.com/wiki/ index.php/AMC_12_Problems_and_Solutions.

- MAA. American invitational mathematics examination (aime), February 2024. URL https://artofproblemsolving. com/wiki/index.php/AIME_Problems_and_Solutions.
- MAA. American invitational mathematics examination (aime), February 2025. URL https://artofproblemsolving. com/wiki/index.php/AIME_Problems_and_Solutions.

Fanxu Meng, Zhaohui Wang, and Muhan Zhang. Pissa: Principal singular values and singular vectors adaptation of large language models. Advances in Neural Information Processing Systems, 37:121038–121072, 2024a.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward.

- Advances in Neural Information Processing Systems, 37:124198–124235, 2024b.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward.

- Advances in Neural Information Processing Systems, 37:124198–124235, 2024c.

Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng. Reinforcement learning finetunes small subnetworks in large language models. Advances in Neural Information Processing Systems, 2025.

Long Ouyang et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pp. 27730–27744, 2022.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. arXiv preprint arXiv:2303.08774, 2018.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing

- systems, 36:53728–53741, 2023a.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing

- systems, 36:53728–53741, 2023b.

Negin Raoof, Etash Kumar Guha, Ryan Marten, Jean Mercat, Eric Frankel, Sedrick Keh, Hritik Bansal, Georgios Smyrnis, Marianna Nezhurina, Trung Vu, Zayne Rea Sprague, Mike A Merrill, Liangyu Chen, Caroline Choi, Zaid Khan, Sachin Grover, Benjamin Feuer, Ashima Suvarna, Shiye Su, Wanjia Zhao, Kartik Sharma, Charlie Cheng-Jie Ji, Kushal Arora, Jeffrey Li, Aaron Gokaslan, Sarah M Pratt, Niklas Muennighoff, Jon Saad-Falcon, John Yang, Asad Aali, Shreyas Pimpalgaonkar, Alon Albalak, Achal Dave, Hadi Pouransari, Greg Durrett, Sewoong Oh, Tatsunori Hashimoto, Vaishaal Shankar, Yejin Choi, Mohit Bansal, Chinmay Hegde, Reinhard Heckel, Jenia Jitsev, Maheswaran Sathiamoorthy, Alex Dimakis, and Ludwig Schmidt. Automatic evals for llms, 2025. URL https://github.com/mlfoundations/evalchemy.

Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627–635. JMLR Workshop and Conference Proceedings, 2011.

John Schulman and Thinking Machines Lab. Lora without regret. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20250929. https://thinkingmachines.ai/blog/lora/.

Zhihong Shao et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. Rl’s razor: Why online reinforcement learning forgets less. arXiv

preprint arXiv:2509.04259, 2025. Guangming Sheng et al. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv:2409.19256, 2024. Gilbert W Stewart. Perturbation theory for the singular value decomposition. 1998. Zhenpeng Su, Leiyu Pan, Xue Bai, Dening Liu, Guanting Dong, Jiaming Huang, Wenping Hu, and Guorui Zhou.

Klear-reasoner: Advancing reasoning capability via gradient-preserving clipping policy optimization. arXiv preprint arXiv:2508.07629, 2025.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste

Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Per-Åke Wedin. Perturbation bounds in connection with singular value decomposition. BIT Numerical Mathematics, 12(1):99–111, 1972.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.

Fang Wu, Weihao Xuan, Ximing Lu, Zaid Harchaoui, and Yejin Choi. The invisible leash: Why rlvr may not escape

its origin. arXiv preprint arXiv:2507.14843, 2025. xAI. Grok: Ai assistant, 2025. URL https://x.ai/grok. Accessed: 2025-09-24, continuously updated. Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong,

et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating

and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025a. Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and

taming zero reinforcement learning for open base models in the wild, 2025b. URL https://arxiv.org/abs/2503.18892.

Simon Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Peter Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. Advances in neural information processing systems, 37:110935–110971, 2024.

Xiaojiang Zhang et al. Srpo: A cross-domain implementation of large-scale reinforcement learning on llm. arXiv preprint arXiv:2504.14286, 2025.

Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. Galore: Memory-efficient llm training by gradient low-rank projection. arXiv preprint arXiv:2403.03507, 2024.

Hanqing Zhu, Zhenyu Zhang, Wenyan Cong, Xi Liu, Sem Park, Vikas Chandra, Bo Long, David Z Pan, Zhangyang

Wang, and Jinwon Lee. Apollo: Sgd-like memory, adamw-level performance. arXiv preprint arXiv:2412.05270, 2024. Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

- A Clarification of LLM Usage

In this work, we employ LLMs to polish the writing throughout the paper and to assist in generating code for figure plotting. Besides, we use it for drawing the teaser figure.

- B More Related works

Post-training Large-scale models pre-trained on broad domains serve as general-purpose backbones with extensive domain knowledge and notable zero-shot capabilities (Radford et al., 2021; Achiam et al., 2023; Touvron et al., 2023; Hu et al., 2023; Li et al., 2024; Radford et al., 2018; Brown et al., 2020). However, such pre-trained models often fail to meet the specific application requirements or align with domain-specific constraints. Post-training methods address this gap by adapting foundation models to downstream tasks. Common approaches include supervised fine-tuning on curated datasets (Howard & Ruder, 2018; Dodge et al., 2020; Wei et al., 2021; Chung et al., 2024), reinforcement learning from human or automated feedback (Ziegler et al., 2019; Ouyang et al., 2022; Guo et al., 2025; Zhai et al., 2024), and other recent techniques (Rafailov

- et al., 2023b).

Especially, the recent advances in LLM reasoning (DeepSeek-AI, 2025) highlight the effectiveness of Reinforcement Learning with Verifiable Rewards (RLVR), which replaces subjective human judgments with automatically verifiable signals. RLVR has been shown to significantly enhance reasoning ability using policy optimization algorithms such as PPO (Ouyang et al., 2022) and GRPO (Shao et al., 2024). Building on these successes, a growing body of work (Yu et al., 2025; Liu et al., 2025b; Luo et al., 2025a; Zhang et al., 2025; Liu

- et al., 2025a; Xiong et al., 2025) continues to refine RL methods tailored for LLM reasoning.

SFT versus RL. Prior work comparing these paradigms has largely focused on downstream performance. A foundational result shows that on-policy RL can outperform offline SFT even with the same expert data (Ross et al., 2011). Recent empirical studies consistently reinforce this, finding that RL-tuned models often generalize better out-of-distribution (Han et al., 2025; Chu et al., 2025) and transfer more effectively to new tasks (Huan et al., 2025) than their SFT counterparts.

While these studies establish a performance hierarchy, our work investigates a different dimension: how these distinct methods affect the model’s internal structure. A recent study observed that RL fine-tunes only a fraction of the network’s parameters (Mukherjee et al., 2025), but this empirical finding left the underlying mechanism unexplored and did not characterize or predict the affected subnetwork. Our work aims to bridge this gap by providing a mechanistic explanation for this phenomenon.

- C Experimental Details

- C.1 Training Settings

Models & Datasets. We run post-training experiments on three open models: DeepSeek-R1-Distill-Qwen-1.5B (Yang

et al., 2024), Qwen2.5-Math-7B (Yang et al., 2024), and Qwen3-Base (Team, 2025). The maximum context length is set to 8192 for DeepSeek-R1-Distill-Qwen-1.5B and Qwen2.5-Math-7B, and to 20480for Qwen3-4B-Base.

We evaluate primarily on mathematics using two training corpora to reduce dataset-specific confounds. (1) DAPO+MATH (DM): a union of the DAPO-Math-17k set2 and the MATH dataset (Hendrycks et al., 2021). (2) DS+SR: the 47k DeepScaler collection (Luo et al., 2025b) combined with high-difficulty (levels 3–5) problems extracted from SimpleRL (Zeng et al., 2025a).We use the version from Huan et al. (2025).

Training details. We implement RLVR on the VeRL pipeline (Sheng et al., 2024) (v4.0) and use vLLM (Kwon et al., 2023)(v8.5) for rollouts. We use FSDPv2 with the default mixed precision configuration. All experiments run on NVIDIA H200 GPUs. Unless otherwise noted, we use DAPO (Yu et al., 2025) without an explicit reference-KL penalty (ratio clipping as in DAPO), a global batch size of 256 (mini-batch 64) with 4 gradient update per step. We use DAPO primarily to eliminate the confounding effect of the KL penalty, which can otherwise obscure the intrinsic parameter update dynamics during training.

Per-model configurations without specific mention:

- • Qwen2.5-Math-7B on DM: 16 rollouts per prompt; 8 x H200 GPUs; 300 training steps.
- • DeepSeek-R1-Distill-Qwen-1.5B on DS+SR: 12 rollouts per prompt; 16 x H200 GPUs; 320 steps.
- • Qwen3-4B-Base on DS+SR: 16 rollouts per prompt; 32 x H200 GPUs; 150 steps.

For LoRA and PiSSA studies, to reduce compute cost during learning-rate sweeps, we use the same DAPO recipe with a global batch size of 128 (mini-batch 32), four gradient updates per step, and 16 rollouts per prompt. Both DeepSeek-R1-Distill-Qwen-1.5B and LLaMA-3.2-3B are trained for 200 steps.

The actor is optimized with AdamW (Loshchilov & Hutter, 2017) (constant learning rate 1×10−6, β1=0.9, β2=0.999). Rewards are verifiable: +1.0 if the extracted final answer is correct and −1.0 otherwise (no separate format score), following the verifier implementation of Su et al. (2025). We enable an over-length penalty with an additional 1024-token budget and a penalty factor of 1.0.

- C.2 Evaluation settings

We evaluate models on four widely used benchmarks: AIME24 (MAA, 2024), AIME25 (MAA, 2025), AMC23 (MAA, 2023), MATH-500 (Lightman et al., 2023), as we main train using math daastets. We used Eval-Chemy (Raoof et al., 2025) with their default temperature 0.7 and 0.8 as the top-p value. In our experiments, we used the averaged accuracy, i.e., pass@1(avg@k) for all benchmarks. to evaluate the models’ performance. Specifically, for AIME24 and AIME 25, we averaged accuracy on 64 samples, for AMC, we average accuracy on 32 samples, For MATH 500, our score is the average accuracy over 2 samples.

- D Intervention details

- Intervention 1: loss–preserving V/O rotation. Let D be the head dimension, Hq the number of query heads, Hkv the number of key/value heads, and nrep = Hq/Hkv (grouped GQA). Denote

Wv ∈ Rd

model×(HkvD), Wo ∈ Rd

model×(HqD).

2DAPO-Math-17k

Draw any orthogonal R ∈ RD×D (Haar/Hadamard) and form the block rotations

Rkv = diag(R,...,R

)∈ R(H

kvD)×(HkvD), Rq = diag(R,...,R

#### ,...)∈ R(H

qD)×(HqD).

#### , R,...,R

nrep

nrep

Hkv

We edit the weights by right–multiplication along the head axis:

= = (11)

|Wv′ WvRkv, Wo′ WoRq.|
|---|

If bv exists, reshape bv per head and set b′v = bvRkv.

- Proposition D.1 (Exact invariance). Let Ctx = Attn(Q,K,V )∈ R⋅×(H

qD). Under equation 11, out′ = Attn(Q,K,V Rkv) (WoRq)⊺ = CtxRqRq⊺Wo⊺ = CtxWo⊺ = out.

Intervention 2: head shuffle (lossless). Let Pkv be a permutation of the Hkv KV heads and Pq its grouped expansion to Hq heads. Apply

cols of (Wk,Wv)← Pkv, cols of Wq ← Pq, columns of Wo ← Pq−1. This relabels which head carries which subspace, while leaving the block function unchanged. We show that after weight intervention, the model weights update position has a sub-random overlap while those untouched weights stay a high overlap.

E Examples of why previous identified method fails

- E.1 Failures of a Fixed Absolute Tolerance Rule

- • False positives at large scale. Within [210,211)=[1024,2048), the bf16 spacing is ULPbf16 = 210−7 = 8. Numbers like 1024.001 and 1024.002 differ by 10−3>10−5, hence would be flagged as “changed” by the 10−5 rule, yet both round to the same bf16 code (1024), i.e., no storage-level change.
- • False negatives at small scale. Around 10−6≈2−20, the bf16 spacing is ULPbf16 = 2−27≈7.45×10−9. Weights w=10−6 and ŵ=2×10−6 differ by 10−6≤10−5 and would be marked “equal” by the 10−5 rule, yet they are separated by ≈ 134 ULPs and quantize to different bf16 codes.

- E.2 Justification of our probe

- Lemma E.1 (Gap between distinct bf16 representables). If x ≠ y are normalized bf16 numbers in the same binade [2e,2e+1), then

∣x − y∣ ≥ 2e−7 and ∣x − y∣

max(∣x∣,∣y∣)

> 2−8. The strict inequality also holds across the binade boundary.

- Lemma E.2 (ULP lens: magnitude-dependent threshold). For normalized bf16 values x with ∣x∣∈[2e,2e+1),

ULPbf16(x) ∣x∣

∈(2−8, 2−7] = (0.390625%, 0.78125%].

Hence the minimal realized relative update at magnitude ∣x∣ is ≳ 12 ULPbf16(x)/∣x∣∈(0.195%, 0.391%]. In particular, larger ∣x∣ requires a larger absolute step to register.

| |
|---|

- Proposition E.3 (Soundness and completeness of the probe). Let wi,ŵi be normalized bf16 values (finite, nonzero), and suppose η < 12 minx ULPbf16(x)/∣x∣= 2−9 ≈ 1.953 ⋅ 10−3. Then

∣̂wi − wi ∣≤ η max(∣wi∣,∣̂wi∣) ⇐⇒ bf16(wi)= bf16(̂wi).

Proof. (⇒)If wi ≠ ŵi, Lemma E.2 gives ∣̂wi − wi∣/max(∣wi∣,∣̂wi∣)> 2−8 > 2η, contradiction. Hence wi = ŵi as bf16 numbers. (⇐) If the stored bf16 values are equal, the difference is 0, which satisfies equation 1.

- Corollary E.4 (Choice η = 10−3 is safe). Since 10−3 < 2−9, Proposition E.3 applies: the test equation 1 passes iff the two bf16 entries are bit-wise identical (or both zero). Thus η = 10−3 yields a scale-aware probe that flags equality only when storage is unchanged.

### F Math Analysis

- F.1 Policy-Gradient Fine-Tuning (DAPO)

Assume an old policy πold that we use to sample G candidate completions y1∶G for each prompt x ∈X. For a single token yi,t (token t in completion i) we define the importance-weighted advantage

πθ(yi,t∣x,y<t) πold(yi,t∣x,y<t) importance ratio

wi,t =

Aˆi,t Iclip ∈ R, (1)

where Aˆi,t is the estimated advantage and Iclip∈{0,1} implements the usual trust-region clipping. Token-level objective. The DAPO loss can be written as a sum of weighted log-probabilities

∣yi∣

G

wi,t logπθ(yi,t ∣ x,y<it)]. (2)

JRL(θ) = Ex∼X, y1∶G∼πold[∑i1∣yi∣

∑

∑

t=1

i=1

- F.2 Proof of Gate I: On-Policy RL Implies a One-Step KL Leash

This appendix provides the standard tilting oracle and M-projection facts, local second-order expansions, and the proof of the one-step policy-KL leash (Prop. 3.1 in the main text). We keep the proof concise, otherwise too lengthy, especially for those has shown in some prior workShenfeld et al.(2025);Wu et al.(2025). Our one-step analysis is inspired by recent work Wu et al. (2025); Shenfeld et al. (2025), which uses a similar variational approach to show that even the final converged policy remains KL-proximal to the base policy. We also record a trust-region/clipping bound used when β = 0.

Throughout, x is fixed, q(⋅∣ x) has full support on Y, and πθ(⋅∣ x) is a C3 parametric family with log-density logπθ locally smooth. Expectations without explicit subscript are conditional on x.

We first show useful lemmas here.

- Lemma F.1 (Frozen-policy surrogate is second-order tight). Let f(θ)∶=LPG(θ) in equation 4 and g(θ)∶=

L̃PG(θ;θt) be the frozen-policy surrogate with Aθ

t

. Then f(θt)= g(θt) and ∇f(θt)=∇g(θt). If ∇f and ∇g are L-Lipschitz in a neighborhood of θt, then

∣f(θt + ∆θ)− g(θt + ∆θ)∣ ≤ L2 ∥∆θ∥2. Proof. At θt, both objectives evaluate to −Eπ

θt

[Aθ

t

logπθ

t

]. For the gradient, using the log-derivative trick and the centering of Aθ

t

, both yield −Eπ

θt

[Aθ

t

∇logπθ

t

]. Thus f(θt)= g(θt) and ∇f(θt)=∇g(θt). The bound is the standard second-order Taylor remainder under Lipschitz gradients.

| |
|---|

- 1: Exponential tilting and M-projection

- Lemma F.2 (Gibbs variational principle / exponential tilting). Fix β > 0 and a full-support reference q(⋅∣ x). Then

max π≪q {Ey∼π[R(x,y)]− β DKL(π∥q)}

is uniquely maximized by

q(y ∣ x)exp(R(x,y)/β) Ey∼q[exp(R(x,y)/β)]

q˜β(y ∣ x)=

.

Proof. Consider L(π,λ)= Eπ[R]− βEπ[log πq ]+ λ(∑y π(y)− 1). Stationarity in π gives log πq = R/β − λ − 1, hence π ∝ q eR/β. Strict concavity in π yields uniqueness.

| |
|---|

- Lemma F.3 (Policy Gradient Update as Parametric M-projection). For fixed q˜β,

argmin

θ

DKL(q˜β∥πθ) = argmax

θ

Ey∼q˜

β

[logπθ(y ∣ x)].

Proof. DKL(q˜β∥πθ)= Eq˜

β

[logq˜β]− Eq˜

β

[logπθ], where the first term is θ-independent. We omit the full proof here, with one can be found in Shenfeld et al. (2025).

| |
|---|

- 2: Local second-order identities

- Lemma F.4 (Local Pythagorean identity for the M-projection). Let f(θ)∶= DKL(q˜β∥πθ)= Eq˜

β

const. Assume logπθ is C3 near θ, and let θ+ ∈ argminf. Writing ∆ ∶= θ+ − θ, for ∥∆∥ small, [−logπθ]+ f(θ)− f(θ+)= 12 ∆⊺Hq˜(θ)∆ + O(∥∆∥3), Hq˜(θ)∶=−Eq˜

β

[∇2 logπθ].

Proof. Taylor-expand f at θ+: f(θ)= f(θ+)+ 12∆⊺Hq˜(θ+)∆+O(∥∆∥3) since ∇f(θ+)= 0. Local C3 smoothness implies Hq˜(θ+)= Hq˜(θ)+ O(∥∆∥), which is absorbed into the cubic remainder.

| |
|---|

- Lemma F.5 (Quadratic expansion of policy KL). Let F(θ)∶=−Eπ

θ

[∇2 logπθ] be the Fisher information. Then

DKL(πθ+∆∥πθ)= 12 ∆⊺F(θ)∆ + O(∥∆∥3). Proof. Expand log π

θ+∆

πθ = ∆⊺∇logπθ + 12∆⊺∇2 logπθ ∆+O(∥∆∥3), take expectation under πθ+∆ = πθ +O(∥∆∥), use Eπ

θ

[∇logπθ]= 0 and −Eπ

θ

[∇2 logπθ]= F(θ).

| |
|---|

3. Relating projection Hessian and Fisher under small tilt

- Lemma F.6 (Hessian–Fisher proximity). Suppose ∥∇2 logπθ(y ∣ x)∥op ≤ L uniformly near θ. Then

∥Hq˜(θ)− F(θ)∥op ≤ 2LTV(q˜β,πθ) ≤ L√2DKL(q˜β∥πθ). In particular, with κ ∶= DKL(q˜β∥πθ)→ 0, we have Hq˜(θ)=(1 + O(

√κ))F(θ) as quadratic forms.

Proof. For bounded matrix-valued h, ∥Eq˜h − Eπh∥op ≤ 2∥h∥∞ TV(q,π˜ ). Apply this with h ∶=−∇2 logπθ and Pinsker’s inequality TV(p,q)≤

√1

2DKL(p∥q).

| |
|---|

4. Remainder control

- Lemma F.7 (Cubic remainder is o(f)). If Hq˜(θ)⪰ mI on the update subspace (local strong convexity), then for ∥∆∥ small

∥∆∥2 ≤ m2 (f(θ)− f(θ+)), O(∥∆∥3)= o(f(θ)).

Proof. From Lemma F.4, f(θ)− f(θ+)≥ m2 ∥∆∥2 + O(∥∆∥3). Rearranging yields ∥∆∥2 = O(f(θ)− f(θ+)), so the cubic term is lower order.

| |
|---|

- F.2.1 Proof of Proposition 3.1

- Proof of Proposition 3.1. Let f(θ)= DKL(q˜β∥πθ) and ∆ = θ+ − θ. By Lemma F.4, f(θ)− f(θ+)= 21 ∆⊺Hq˜(θ)∆ + O(∥∆∥3).

By Lemma F.5,

DKL(πθ+∥πθ)= 12 ∆⊺F(θ)∆ + O(∥∆∥3). By Lemma F.6 with κ = f(θ), ∆⊺F∆ =(1 + O(

√κ))∆⊺Hq˜∆. Hence DKL(πθ+∥πθ)=(1 + O(

√κ))(f(θ)− f(θ+)) + O(∥∆∥3).

Since f(θ+)≥ 0, f(θ)− f(θ+)≤ f(θ)= κ. By Lemma F.7, O(∥∆∥3)= o(f(θ)). Therefore

DKL(πθ+∥πθ) ≤ (1 + o(1))f(θ) = (1 + o(1))DKL(q˜β∥πθ), which is the desired inequality.

| |
|---|

F.2.2 Proof of Proposition 3.2

- Proof of Proposition 3.2. By the quadratic expansion of policy KL (Lemma F.5),

- F.2.3 One-step KL budget (used in Gate II)

DKL(πθ+∆∥πθ)= 12 ∆⊺F(θ)∆ + R(∆), ∣R(∆)∣ ≤ C ∥∆∥3 (12)

for some local constant C > 0 (from C3 smoothness). Let a ∶= ∆⊺F(θ)∆. Using the spectral lower bound F(θ)⪰ µI on the update subspace,

∥∆∥2 ≤ µa. (13) Combining equation 12–equation 13 yields

3/2

DKL(πθ+∆∥πθ) ≥ 12 a − C (µa)

.

Since DKL(πθ+∥πθ)≤ K, we have

K ≥ 12 a − C µ−3/2a3/2. (14) For a sufficiently small (equivalently, K small), the cubic term is dominated by the linear term: choose a0 > 0 so that C µ−3/2√a ≤ 14 whenever 0 < a ≤ a0. Then from equation 14

K ≥ (12 − 14)a = 14 a ⇒ a ≤ 4K.

Substitutingo(K), so DKLa(π≤θ4+K∆∥πbackθ)=into21a +equationo(K). Hence12 refinesa = 2DtheKL(remainder:πθ+∆∥πθ)+∣Ro((K∆)≤)∣≤2CK∥∆+ o∥3(K≤)C, (i.e.a/µ)3/2 = O(K3/2)= ∆⊺F(θ)∆ ≤ 2K (1 + o(1)).

√∆⊺F(θ)∆ ≤

#### √2K (1 + o(1)). The Euclidean bound follows from equation 13:

Taking square roots gives the Fisher-norm bound in equation 6: ∥∆∥F(θ) =

√

#### √2K

∆⊺F(θ)∆

∥∆∥2 ≤

µ ≤

µ (1 + o(1)).

Finally, for any parameter block W ⊂ θ, its Frobenius change is the ℓ2-norm of the corresponding subvector of ∆; therefore ∥∆W∥F ≤∥∆∥2.

| |
|---|

- Corollary F.8 (KL budget). If DKL(πθ+∥πθ)≤ K, then

- 1

- 2 ∆⊺F(θ)∆ ≤ K (1 + o(1)).

#### Proof. Apply Lemma F.5 and Lemma F.7.

| |
|---|

- F.2.4 Trust-region / clipping bound (for β = 0)

- Lemma F.9 (Implicit KL leash from ratio clipping). Let rt = π

θ+(yt∣x,y<t)

πθ(yt∣x,y<t) and suppose clipping enforces rt ∈[1 − ε, 1 + ε] on the batch. Then

D̂KL(πθ+∥πθ) ≤ Ê[T(x)]⋅ max{−log(1 − ε), log(1 + ε)} = O(ε)⋅ Ê[T(x)], and in the small-step regime (mean-zero advantage) this tightens to O(ε2).

Proof. Autoregressive factorization gives DKL(πθ+∥πθ)= Eπ

θ+[∑t logrt]. Because logrt ∈[log(1−ε),log(1+ε)], we have ∣logrt∣ ≤ c(ε); summing over t and taking batch expectation yields the stated bound. Using log(1 ± ε)=±ε + O(ε2) and small-step arguments gives O(ε2).

| |
|---|

F.3 Proofs for Gate II (Sec. 3.2)

Setup (layer-conditioned budget). Partition θ =(vec(W),θ¬W) and let the Fisher at θ = θt be

F(θ)=[

FW,W FW,¬W F¬W,W F¬W,¬W]⪰ 0.

For a one-step update ∆θ, the global KL leash implies 12 ∆θ⊺F(θ)∆θ ≤ K. Define the layer-conditioned curvature

SW ∶= FW,W − FW,¬WF¬−W,1 ¬WF¬W,W ⪰ 0,

and the per-layer budget δW ∶= 12 vec(∆W)⊺SW vec(∆W)≤ K. Let µW ∶= λmin(SW)> 0 on the update subspace.

- Lemma F.10 (Layer-conditioned Frobenius/operator bounds). ∥∆W∥F ≤

√2δW/µW and ∥∆W∥2 ≤∥∆W∥F.

Proof. Since SW ⪰ µWI, δW ≥ 12 µW∥∆W∥2F.

- Lemma F.11 (Wedin’s sin–Θ). For W+ = W0 + ∆W, let γk ∶= σk(W0)− σk+1(W0) be the singular value gap. For any k where γk > 0, the principal subspace angles satisfy ∥sinΘ(Uk(W0),Uk(W+))∥2 ≤∥∆W∥2/γk and similarly for Vk.

| |
|---|

σLemmai(W0))2 ≤∥F.12∆W(Weyl/Mirsky∥2F. and Hoffman–Wielandt). ∣σk(W+)− σk(W0)∣ ≤ ∥∆W∥2 and ∑i(σi(W+)− Corollary F.13 (Projection stability). With the same assumptions (including γk > 0),

| |
|---|

√2δW/µW γk

∥Uk(W0)Uk(W0)⊺ − Uk(W+)Uk(W+)⊺∥2 = ∥sinΘ(Uk(W0),Uk(W+))∥2 ≤

.

√δW/µW/γk); when the gap is moderate, the rotation is small.

The analogous bound holds for the right subspaces with Vk. Interpretation. The leading invariant subspaces rotate by at most O(

| |
|---|

### G More Visualization

- G.1 Jaccard matrix

RL updates are highly consistent across independent training runs. Fig. 12 shows the pair-wise Jaccard similarity between the final update masks from five RLVR runs on different data and algorithms. The high similarity scores demonstrate that the optimization process consistently targets the same subset of parameters, providing strong evidence for a deterministic, non-random optimization bias.

- G.2 Spectrum shift for DS-1.5B and Qwen3-1 We also show the spectrum shift for DS-1.5B and Qwen3-14B here.

[Figure 22]

- Figure 12 Pair-wise Jaccard similarity of update masks from five independent RLVR runs on Layer 13 of the DS-DistillQwen-1.5B model.

- Table 4 Performance of DS-Qwen-1.5B with different masking strategies at 320 steps. Parameter counts shown are for linear layers only, excluding the embedding and head layers. Detailed evaluation settings are available in Appendix C.2. We observe that training only on principal weights Mprinc results in a clear accuracy gap compared to both the dense baseline

and its complement Mprincc . The models using the Mlow and Mprincc ∪ Mlowest masks achieve performance closest to the dense baseline.

Model Mask Math500 AMC23 AIME24 AIME25 Average #params

Dense 84.20 81.56 36.98 27.03 57.44 100%

Mprinc 83.60 77.19 30.16 24.32 53.82 50% Mprincc 82.70 78.90 34.28 25.73 55.40 50%

DS-Qwen-1.5B

Mlow 84.50 80.08 35.62 26.56 56.69 58.59% Mprincc ∪ Mlow 85.20 78.83 34.74 26.20 56.24 74.02% Random-Mprincc ∪ Mlow 84.50 77.35 34.48 25.01 55.34 74.02%

[Figure 23]

- Figure 13 Structured Update observed on Llama(Llama-3.1-8B) and Mistral (Mistral-Small-24B) models. Here we plot the weight update mask using the zero-RL checkpoints from Zeng et al. (2025b).

[Figure 24]

- Figure 14 Temporal emergence of the optimization bias with row and column-wise update ratios for the 13th attention block across gradient update steps (t∈{240,720,1200}), smoothed with a 3-step window. The column-wise (Q) and row-wise (O) update ratios show a much weaker bias.

- Table 5 Performance of DS-Qwen-1.5B with different masking strategies with a extended training window to 500 steps. Parameter counts shown are for linear layers only, excluding the embedding and head layers. Detailed evaluation settings are available in Appendix C.2. We observe that training only on principal weights Mprinc results in a clear accuracy

gap compared to both the dense baseline and its complement Mprincc . The models using the Mlow and Mprincc ∪ Mlowest masks achieve performance closest to the dense baseline.

Model Mask Math500 AMC23 AIME24 AIME25 Average #params

Dense 84.5 83.52 38.28 28.075 58.59 100%

Mprinc 83.60 78.83 34.06 25.63 55.44 50% Mprincc 84.0 77.97 38.64 27.81 56.90 50%

DS-Qwen-1.5B

Mlow 83.8 82.42 37.03 27.82 57.77 58.59% Mprincc ∪ Mlow 84.10 81.41 40.30 27.70 58.37 74.02% Random-Mprincc ∪ Mlow 84.10 81.72 34.69 27.34 56.89 74.02%

[Figure 25]

##### Figure 15 Token-wise KL loss. We show the token-wise KL loss during a DAPO run without a KL loss penalty, which shows a steadily increasing KL loss instead of being unconstrained.

Principal Angle (single-layer)

Singular Value (single-layer)

Max Principal Angle (all-layer)

Specturm Drift (all-layer)

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

RL

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

SFT

##### Figure 16 The spectrum probe results on the RL and SFT version on the DS-Distill-Qwen-1.5B Liu et al. (2025a). RLVR shows surprisingly stable top-k spectrum with minimal subspace rotation and top-k eigenvalue changes.

Principal Angle (single-layer)

Singular Value (single-layer)

Max Principal Angle (all-layer)

Specturm Drift (all-layer)

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

RL

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

SFT

##### Figure 17 The spectrum probe results on the RL and SFT version on the Qwen3-14B Huan et al. (2025). RLVR shows surprisingly stable top-k spectrum with minimal subspace rotation and top-k eigenvalue changes.

[Figure 42]

##### Figure 18 Spectral geometry under RLHF setting Meng et al. (2024b). Across RLHF checkpoints, RL training preserves layer spectra and induces only minor rotation of the top-k subspaces, consistent with the RLVR regime.

