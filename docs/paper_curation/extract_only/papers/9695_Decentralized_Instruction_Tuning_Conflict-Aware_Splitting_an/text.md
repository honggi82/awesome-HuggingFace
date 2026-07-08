## Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging

# arXiv:2606.01717v1[cs.LG]1Jun2026

Minsik Choi*12‡ Geewook Kim*13†

### Abstract

: Task Conflict

###### Centralized Training

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Instruction tuning aligns large language models, including multimodal ones, with diverse user intents, but scaling to heterogeneous mixtures is hindered by gradient interference and bandwidthheavy synchronization. We ask whether these two bottlenecks can be addressed jointly by training parts of the mixture independently and reconciling them once in parameter space. We develop a local quadratic theory inside a shared flat basin that yields three results: weight merging produces a curvature-weighted variance reduction; PCA-aligned conflict splitting maximizes this gain along high-curvature directions; and merging additionally acts as spectral filtering with implicit norm regularization. These results directly motivate MERIT, a decentralized merge-ready instruction-tuning pipeline that estimates datasetlevel gradient conflicts, partitions the mixture along the top PCA conflict axes, fine-tunes each partition independently with no inter-partition communication, and merges once via tokenweighted averaging. On Qwen2.5-VL-3B with 136 Vision-FLAN tasks, MERIT improves the 8benchmark average from 54.3 (joint training) to 57.0. The same recipe scales to a 7B model on a 1.6M-example, 176-source mixture—matching or exceeding centralized joint training with minimal cost overhead—and transfers to text-only FLAN. Our code is available at https://github.

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

Instruction Tuning

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

N × GPU servers (Requires High-Performance Cross-node Communication)

Θ

T tasks (mixed)

###### Resolve Task Conflict MERIT (Ours)

[Figure 27]

| |
|---|

###### → partition into N groups

| |
|---|

GPU₁ → Θ₁

| |
|---|

[Figure 28]

Merge

[Figure 29]

| |
|---|

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

| |
|---|

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

| |
|---|

[Figure 40]

[Figure 41]

[Figure 42]

GPU₂ → Θ₂

[Figure 43]

| |
|---|

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

| |
|---|

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

| |
|---|

[Figure 56]

| |
|---|

GPU₃ → Θ₃

| |
|---|

T tasks (mixed)

Decentralized Instruction Tuning

Θഥ∗

| |
|---|

[Figure 57]

| |
|---|

| |
|---|

GPU₄ → Θ₄

| |
|---|

(Proposed)

| |
|---|

Figure 1. Centralized training vs. MERIT. Centralized tuning synchronizes conflicting tasks across a tightly-coupled cluster (top); MERIT partitions the mixture by conflict, fine-tunes each group independently, and merges once into θ¯(bottom).

step is particularly critical for multimodal large language models (MLLMs) (Liu et al., 2024a;b), where much of the practical multimodal behavior is acquired after language pretraining through vision–language alignment and heterogeneous instruction data spanning perception, reasoning, OCR and document understanding, diagram comprehension, and safety (Xu et al., 2024b; Lauren¸con et al., 2024). As a result, practitioners keep scaling instruction mixtures to achieve robust, product-level capability (Hu et al., 2024; Wang et al., 2024; Kim & Seo, 2024; Bai et al., 2025; Li et al., 2025; Gemma Team et al., 2025; Chen et al., 2024).

com/naver-ai/merit.

However, the prevailing approach, centralized joint training on the entire mixture, faces two bottlenecks. On the optimization side, heterogeneous datasets induce conflicting gradients, leading to negative transfer and stiff dynamics that constrain the learning rate and slow progress (analyzed in Section 3). Classical multi-task corrections that manipulate task-wise gradients (Chen et al., 2018; Yu et al., 2020; Liu et al., 2021) become infeasible at the scale of over a hundred tasks and billions of parameters (Appendix E.1), leaving mixture-ratio curation (Laurenc¸on et al., 2024) as the main recourse. On the systems side, joint training relies on frequent gradient synchronization (e.g., all-reduce), effectively requiring tightly-coupled clusters with high-bandwidth interconnects; this assumption breaks in fragmented compute environments such as heterogeneous GPU pools, geo-

### 1. Introduction

The modern recipe for building capable foundation models relies on extensive post-training, in particular large-scale instruction tuning over a diverse mixture of tasks. This

*Equal contribution ‡This work was conducted during Minsik Choi’s internship at NAVER Cloud. 1 NAVER Cloud AI 2 Korea University 3 KAIST AI . Correspondence to: †Geewook Kim <gwkim.rsrch@gmail.com>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

distributed clusters, and cloud spot instances, where communication is a dominant cost or a hard constraint.

These two bottlenecks are coupled: as mixtures grow more heterogeneous, optimization becomes more sensitive to dataset interference, yet mitigating such interference during training demands fine-grained, synchronized training signals. When synchronization is unavailable, practitioners fall back to coarse mixing heuristics, leaving interference largely unaddressed.

This calls for an alternative: instead of forcing all datasets to share a single synchronized trajectory, can we train parts of the mixture independently and reconcile them once in parameter space? Weight-space averaging and model soups (Wortsman et al., 2022; Jang et al., 2024) suggest that when fine-tuning runs start from the same checkpoint and remain within a connected low-loss region, one-shot parameter averaging can yield a single model that is often better than its constituents. Crucially, this weight-space compatibility is common in post-training, where fine-tuning typically stays within a shared flat basin around a strong initialization. We term such a checkpoint a merge-ready initialization (empirically verified via linear mode connectivity, displacement, and perturbation diagnostics in Appendix B) and ask: how should we split a large instruction mixture so that independent training remains mergeable and merging systematically reduces interference? Existing soup-style methods average models trained on the same data with different randomness; our setting instead requires merging models trained on disjoint partitions of a heterogeneous mixture, where the choice of split directly determines merging quality.

We answer this question with MERIT (Merge-Ready Instruction Tuning), a decentralized instruction-tuning pipeline built on two moves at a merge-ready initialization—

split the mixture along the dominant gradient-conflict axes identified by PCA, then fine-tune each partition independently with no cross-partition communication—followed by a single token-weighted merge (Figure 1). Structurally, the split separates conflicting updates across branches; operationally, the pipeline reduces post-training to a problem that fragmented, bandwidth-constrained hardware can solve. A local quadratic analysis inside a shared flat basin (Section 3) makes this concrete: the merging gain is a curvatureweighted variance reduction, PCA-aligned splitting maximizes it, and averaging implicitly regularizes toward θ(0) while acting, conceptually, as spectral filtering.

Our main contributions are:

• Theoretical framework. A local quadratic analysis inside a shared flat basin derives three results: merging yields a curvature-weighted variance reduction; PCA-aligned splitting maximizes this gain, with the

advantage growing with the Hessian spectral gap; and weight averaging acts as implicit norm regularization and, conceptually, spectral filtering (Section 3).

- • Decentralized instruction-tuning pipeline. MERIT estimates dataset-level gradient conflicts from a small calibration set, partitions the mixture along top-r PCA conflict axes at a merge-ready initialization, trains K=2r branches fully independently, and merges once via token-weighted averaging—no cross-partition gradient communication during fine-tuning.
- • Consistent empirical gains at scale. On Qwen2.5VL-3B with 136 Vision-FLAN tasks (Xu et al., 2024b), MERIT improves the 8-benchmark average from 54.3

→ 57.0 under identical token budgets; on a Qwen2.5VL-7B build with a 1.6M-example 176-source mixture, MERIT matches or exceeds centralized joint training across three independent seeds with minimal wallclock overhead (Section 6), and the same recipe transfers to a text-only FLAN setting (Wei et al., 2022).

### 2. Background and Related Work

Model Merging and Loss-Landscape Connectivity. Weight-space model merging averages multiple checkpoints into a single model and is motivated by loss-landscape connectivity (Garipov et al., 2018; Draxler et al., 2018) and flatminima interpretations (Hochreiter & Schmidhuber, 1997; Izmailov et al., 2018). Model soups and Model Stock show that simple averaging over independently fine-tuned models can be effective (Wortsman et al., 2022; Jang et al., 2024), while follow-up work explores more robust merging rules, including curvature-aware variants and conflict-mitigating heuristics (e.g., sign alignment) (Matena & Raffel, 2022; Yadav et al., 2023; Yu et al., 2024a). Most prior work merges models trained on the same dataset/objective (different seeds or hyperparameter settings), where averaging mainly reduces run-to-run variance. To disentangle such generic averaging gains from our contributions, our experiments include soup-style baselines (uniform averaging of multiple runs on the full mixture) and random dataset partitioning followed by averaging. MERIT instead targets heterogeneous instruction mixtures and introduces an a priori conflict-aware split so that models trained on disjoint subsets remain mergeable and benefit from averaging.

Federated and Decentralized Optimization. Independent local training followed by averaging is reminiscent of FedAvg and Local SGD (McMahan et al., 2017; Stich, 2019), which periodically synchronize client models to optimize a shared global objective under siloed data and communication/privacy constraints (Kairouz et al., 2021). Recent work extends federated learning to LLM finetuning via communication-efficient strategies such as shared-

randomness gradient compression (Zelikman et al., 2023), seed-based full-parameter tuning (Qin et al., 2024), and forward-pass perturbation (Xu et al., 2024a), while lowrank (Hyeon-Woo et al., 2022) and personalization techniques (Qin et al., 2023; 2025) further reduce per-round cost. However, these methods still require iterative synchronization rounds. One-shot federated learning (Li et al., 2021; Diao et al., 2023; Li et al., 2024c; Huang & Shu, 2025) removes iterative communication by aggregating after a single local-training round, sharing MERIT’s single-merge structure. A key distinction is that FL methods typically assume fixed, pre-assigned data partitions dictated by privacy or ownership constraints, whereas MERIT assumes centrally available data and optimizes the partition itself to maximize merging quality via gradient-conflict analysis. This active partitioning, a degree of freedom absent in the FL setting, is the primary driver of MERIT’s performance gains.

Task Interference in Multi-Task and Instruction Tuning. Negative transfer from conflicting gradients is a central challenge in multi-task learning; gradient reweighting/projection methods mitigate interference but typically require synchronized access to per-task gradients during training (Chen et al., 2018; Yu et al., 2020; Liu et al., 2021). Instruction tuning is likewise sensitive to mixture composition (Longpre et al., 2023; Sanh et al., 2022). In multimodal post-training, Vision-FLAN reports conflicts between diverse short-answer tasks and verbose conversational responses (Xu et al., 2024b). More broadly, alignment datasets can encode competing objectives (e.g., helpfulness vs. harmlessness) (Askell et al., 2021). In the multimodal setting, recent studies show that vision-language adaptation can impact model safety (Lee et al., 2025a), motivating behavioralignment pipelines such as RLHF-V (Yu et al., 2024b). MERIT complements these lines by separating conflicting datasets before training and reconciling them once in parameter space, avoiding step-level interference accumulation without requiring synchronized per-step control.

Data Curation and Mixture Design for (M)LLMs. Recent (M)LLM pipelines build large instruction corpora by curating and mixing heterogeneous datasets with careful ratio tuning (Zhou et al., 2023; Liu et al., 2023; Laurenc¸on et al., 2024). As new datasets are added or priorities shift, mixture design becomes an iterative bottleneck. MERIT complements curation with a reusable decomposition primitive: it estimates dataset interactions at a shared initialization, enables communication-free parallel fine-tuning during training, and merges once at the end.

Positioning of Our Work. MERIT sits at the intersection of model merging, decentralized training, and instruction-mixture learning. Unlike post-hoc merging methods, MERIT introduces an a priori conflict-aware dataset

[Figure 58]

[Figure 59]

[Figure 60]

16.0

17.1

[Figure 61]

[Figure 62]

0.4

0.4

15.2

16.5

14.4

15.9

0.2

0.2

13.6

15.3

Direction2

Direction2

12.8

Loss

14.7

Loss

0.0

0.0

12.0

14.1

11.2

13.5

- -0.2

After llava-recipe stage2

- -0.4

- -0.2
- -0.4

10.4

12.9

9.6

12.3

-0.4

8.8

11.7

-0.4 - 0.2 0.0 0.2

-0.4 - 0.2 0.0 0.2 0.4

0.4

[Figure 63]

[Figure 64]

Direction 1

Direction 1

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

17

16 15

16

14

15

Loss

Loss

- 12 11 10

[Figure 69]

[Figure 70]

- 13

Loss

[Figure 71]

[Figure 72]

14

[Figure 73]

[Figure 74]

[Figure 75]

13

Loss

Loss

[Figure 76]

[Figure 77]

[Figure 78]

12

[Figure 79]

-0.4

-0.4

[Figure 80]

-0.4

-0.4

[Figure 81]

-0.2

-0.2

Direction2

-0.2

-0.2

Direction1

[Figure 82]

Direction1 Direction2

0.0

0.0

[Figure 83]

0.0

0.0

0.2

0.2

[Figure 84]

0.2

0.2

[Figure 85]

0.4

0.4

0.4

0.4

After llava-recipe stage1

Figure 2. Local loss surfaces before and after basin preparation (e.g., LLaVA Stage 2). The merge-ready initialization θ(0) resides in a flat, connected region (right), where independently fine-tuned checkpoints remain within the same basin. Our analysis operates within this flat-basin regime, yielding three key implications that directly motivate MERIT’s algorithm design.

split to make simple averaging effective under disjoint heterogeneous data. Unlike gradient-level multi-task methods, MERIT moves conflict handling before training and removes step-level synchronization. Unlike federated averaging/local SGD, MERIT chooses partitions to reduce dataset interference and is analyzed under a flat-basin merging theory rather than as an approximation to centralized training on a single objective.

### 3. Theoretical Framework

We present a theoretical framework explaining why weight merging improves optimization and generalization when fine-tuned checkpoints remain confined to a common flat basin (Figure 2). Our analysis yields three key implications, each experimentally verified in Sections 5–6, that directly motivate MERIT (Section 4): merging is never worse than the weighted average of individual losses in the quadratic model, with improvement governed by curvature-weighted variance; conflict-aware PCA splitting maximizes this gain over random partitioning, with the advantage growing with the curvature gap; and merging acts as implicit norm regularization and, conceptually, as curvature-weighted spectral filtering, improving both generalization and optimization dynamics. Formal assumptions and complete proofs are deferred to Appendix F.

Setting. Let θ ∈ Rd denote the model parameters and L(θ) the population loss. We consider K ≥ 2 checkpoints θ1,...,θK, each fine-tuned on a different dataset group

from a shared merge-ready initialization θ(0)—a checkpoint from which independently fine-tuned models remain in a connected low-loss region. We describe how to obtain such an initialization in Section 4. We fix a reference point θ⋆ within the basin, define displacements δi = θi − θ⋆ (for the first-order analysis below we take θ⋆ = θ(0); the gain Gvar is invariant to this choice), and let H = ∇2L(θ⋆) ⪰ 0 denote the local Hessian. The merged checkpoint is θ¯w = i wiθi with weights wi ≥ 0 summing to one, with uniform averaging recovered as a special case of the token-weighted averaging used by MERIT.

Merging yields curvature-weighted variance reduction. Under a local quadratic approximation L(θ) ≈ L(θ⋆) +

- 1

- 2(θ − θ⋆)⊤H(θ − θ⋆), weighted averaging yields a deterministic gain:

K

- 1

- 2 ℓ

wiL(θi) − L(θ¯w)

λℓ Varw u⊤ℓ δi ≥ 0

=

i=1

merging gain Gvar

(1) where λℓ,uℓ are eigenvalue–eigenvector pairs of H, and Varw(xi) := i wix2i − ( i wixi)2 denotes the wweighted variance over {xi}Ki=1.

Theorem 3.1 (Merging gain in a flat basin). Under a local quadratic model with H ⪰ 0, L(θ¯w) ≤ i wiL(θi). The inequality is strict whenever the displacements {δi} are not identical modulo the nullspace of H. The gain Gvar is maximized when checkpoint dispersion concentrates along the dominant eigendirections of H.

The gain therefore depends on where the dispersion lies: variance along high-curvature directions dominates, while variance in flat subspaces contributes little. This motivates splitting strategies that concentrate inter-group updates along curvature-bearing directions, the role of conflictaware PCA in MERIT.

Conflict-aware splitting maximizes the merging gain. The above result shows that Gvar grows with curvaturealigned dispersion. We now show that PCA-based splitting achieves this alignment.

Under a first-order fine-tuning approximation from θ(0), the displacement of branch k is δk ≈ −η g¯k, where g¯k is the mean gradient of datasets assigned to group k. The merging gain for two groups becomes

Gvar =

η2 8

(¯g1 − g¯2)⊤H (¯g1 − g¯2).

Since the gradient of dataset t at θ(0) satisfies gt = −H∆t (where ∆t is the centered per-dataset optimum), the gain is governed by H3-weighted interactions among the perdataset optima. This curvature amplification means that

PCA on gradient conflicts preferentially identifies highcurvature disagreement axes.

Proposition 3.2 (PCA-aligned splitting maximizes Gvar). In a tractable linear-quadratic model (Appendix F.4), PCAaligned partitioning selects the high-curvature disagreement axis and provably outperforms random partitioning: it attains the maximum gain among all balanced partitions in the T=4, d=2 instance (Proposition F.6), and for general T exceeds random partitioning in expectation under a spectral-concentration condition (Proposition F.11). The advantage grows with the spectral gap λ1/λ2, and the gain scales as λ31—curvature enters twice through gradient conflict (gt ∝ H∆t) and once through the Hessian-weighted gain formula. MERIT’s recursive splitting along the top-r PCA axes (K=2r) accumulates such gains across r orthogonal high-curvature directions, consistent with the monotonic 1D → 2D → 3D improvements observed in Section 5.

Although PCA is applied to dataset-level gradient similarities rather than the Hessian directly, gradient disagreement at a shared initialization provides a first-order proxy for curvature-sensitive update directions under local smoothness. Appendix F.2 formalizes the connection between cosine-similarity and raw-gradient PCA, and Appendix F.3 empirically confirms across two MLLM scales that PCA conflict directions partially align with the top Hessian eigenvectors (far above a Gaussian random baseline), supporting this proxy interpretation.

Merging as spectral filtering with implicit norm regularization. The preceding results establish that conflictaware merging reduces loss. We now show it also improves optimization dynamics and regularizes parameters.

Spectral filtering (conceptual). Joint training on a heterogeneous mixture is constrained by stability along the stiffest curvature direction: the learning rate must satisfy η < 2/λmax, forcing slow progress along flatter dimensions when the condition number κ = λmax/λmin is large. PCA-structured splitting and merging instead drive U⊤(θ¯w − θ⋆) ≈ 0, where U spans the dominant curvature subspace, suppressing high-curvature error components. This acts as approximate spectral filtering, lowering the effective condition number to κeff ≈ λ⊥,max/λmin ≪ κ (Appendix F.1.1), trading the slow, stability-constrained descent of joint training for a one-shot neutralization of high-curvature conflicts; we verify the predicted stability benefit empirically under an SGD learning-rate sweep.

Implicit norm regularization. Weight averaging additionally contracts the merged model toward the shared initialization. By convexity of the squared norm,

∥θ¯w − θ(0)∥2 ≤

i

wi∥θi − θ(0)∥2, (2)

- Table 1. Merge-readiness diagnostics (Qwen2.5-VL-3B, MERIT2D, K=4 branches). The merged model stays closer to θ(0) while carrying higher training loss, yet generalizes better.

Displacement ∥ · −θ(0)∥2 Training loss Joint Merged Ratio Joint Merged Gap

Epoch

- 0.5 13.73 5.65 2.43× 0.709 1.198 +0.489
- 1.0 19.73 7.50 2.63× 0.560 1.172 +0.611
- 2.0 28.15 10.11 2.78× 0.370 1.167 +0.797 6.0 34.61 11.87 2.92× 0.064 1.330 +1.266

with strict inequality whenever checkpoints differ. This contraction yields a smaller distance-based complexity term under PAC-Bayes generalization bounds, providing a principled account of why MERIT can improve generalization even when individual branch training losses are not lower than joint training (see Table 1).

Empirical verification of merge-readiness. The analysis above rests on two assumptions—branches remaining in a single flat basin and averaging contracting displacements along high-curvature directions—which we verify directly on Qwen2.5-VL-3B’s K=4 branches (MERIT-2D) through four diagnostics (full details in Appendix B). First, along all 6 pairwise and 4 branch-to-merged linear interpolation paths, the loss barrier maxα[L(θ(α)) − ((1−α)L(0) + αL(1))] is exactly zero: every path stays at or below linear interpolation, the strongest possible form of linear mode connectivity. Second, consistent with Eq. (2), the merged model remains 2.4–2.9× closer to θ(0) than the jointly trained model throughout training, with the ratio widening monotonically (see Table 1). Third, despite carrying a substantially higher training loss on the full mixture (+0.49 to +1.27 across epochs), the merged model achieves better held-out performance—the classic signature of implicit regularization, and precisely the behavior predicted by the norm-contraction argument above (formalized via PACBayes in Appendix F.5). Finally, isotropic Gaussian weight perturbations (σ ∈ {0.01,0.05,0.1}) produce consistently smaller loss increases on the merged model than on the jointly trained one, confirming a flatter surrounding landscape (Appendix B.2).

### 4. Proposed Method

Overview. MERIT is a decentralized instruction-tuning pipeline that turns a heterogeneous dataset mixture into K=2r conflict-aware partitions, fine-tunes each partition independently from a shared merge-ready initialization θ(0), and merges the resulting checkpoints via token-weighted averaging. The pipeline has five stages (Algorithm 1): (i) dataset-level gradient conflict estimation, (ii) PCA-based decomposition of the conflict structure, (iii) balanced partitioning along the dominant PCA axes, (iv) communication-free

Algorithm 1 MERIT: Conflict-Aware Dataset Partitioning and Weight Merging

Require: Merge-ready initialization θ(0); datasets {Dt}Tt=1 with sample counts {st}Tt=1 and token budgets {nt}Tt=1; PCA dimension r. Ensure: Merged model θ¯.

- 1: ▷ Step 1: Gradient conflict estimation at θ(0).
- 2: for t = 1,...,T do
- 3: Compute gt at θ(0) under identical training settings (backbone, trainable-parameter subset, gradientestimation budget); set g˜t ← gt/∥gt∥.
- 4: end for
- 5: Form C ∈ RT×T with Cij = ⟨g˜i,g˜j⟩.
- 6: ▷ Step 2: PCA-based conflict decomposition.
- 7: Apply (column-centered) PCA to C and obtain the topr PCA embedding zt ∈ Rr for each t.
- 8: ▷ Step 3: Balanced conflict-aware partitioning.
- 9: Recursively split {1,...,T} along the r PCA axes via sample-balanced medians (weights st) into K = 2r disjoint groups {Gk}Kk=1, balancing per-group sample

counts t∈G

k

st; let Nk = t∈G

k

nt denote the pergroup token budget (with Kk=1 Nk = Tt=1 nt).

- 10: ▷ Step 4: Communication-free group-wise training.
- 11: for k = 1,...,K do
- 12: Train θk from θ(0) on t∈G

k

Dt using budgets {nt}t∈Gk

.

- 13: end for
- 14: ▷ Step 5: Token-weighted parameter-space merging.
- 15: return θ¯ = Kk=1 wk θk with wk = Nk/ Kj=1 Nj.

branch training, and (v) weight merging.

Merge-ready initialization. A prerequisite for MERIT is a merge-ready initialization θ(0): a checkpoint from which independently fine-tuned models remain in a connected low-loss region and therefore admit effective oneshot merging. In our multimodal experiments, θ(0) is an already instruction-tuned MLLM checkpoint (e.g., released Qwen2.5-VL); in text-only settings a pretrained LLM is sufficient. Merge-readiness is an empirical property of the chosen initialization; comprehensive diagnostics (linear mode connectivity, weight displacement, and perturbation robustness) are provided in Appendix B.

#### 4.1. Dataset-Level Gradient Conflict Estimation

MERIT quantifies dataset-level interference by comparing the directions of per-dataset gradients at θ(0). For each dataset t, we estimate a representative gradient gt by averaging gradients over a small calibration set (up to 200 examples per dataset), using identical model and trainable-

parameter settings. We then normalize each gradient and construct the cosine similarity matrix,

Cij := ⟨gi,gj⟩ ∥gi∥∥gj∥

,

where high values indicate aligned updates and low or negative values indicate conflicting updates under joint optimization. Cosine-based gradient alignment is a standard tool for characterizing task interactions in multitask and continual learning (Lopez-Paz & Ranzato, 2017; Yu et al., 2020; Ilharco et al., 2023; Lei et al., 2026).

On a 100-task subset of Vision-FLAN, the mean cosine similarity between the n=200 calibration gradient and the full 1,000-sample reference gradient reaches 0.847 (std 0.106), with diminishing returns beyond n=200 (Appendix C.4); a qualitative t-SNE view of these dataset-level gradients is provided in Appendix C.1. To keep preprocessing cheap at scale, we subsample one gradient entry every s parameters (default s=5, 20% retained); the resulting cosine similarities match the full-gradient baseline with Pearson/Spearman correlations above 0.98 (Appendix E). Finally, the matrix can be reused across collections: extending an existing Tdataset collection with m new datasets needs only O(Tm) additional similarity computations rather than recomputing the full O((T+m)2) matrix, followed by a negligible PCA update.

#### 4.2. PCA-Based Conflict Decomposition

We extract the dominant conflict structure of C via PCA, yielding an r-dimensional embedding zt ∈ Rr per dataset that captures the principal patterns of agreement and disagreement among dataset-level gradients. We consider r ∈ {1,2,3}, giving the 1D, 2D, and 3D variants of MERIT (with K=2r ∈ {2,4,8} branches).

We use cosine-similarity PCA as the default: it is scaleinvariant and recovers the same leading eigenspace as rawgradient PCA under gradient-norm concentration, which we verify empirically across our dataset mixtures (Appendix F.2).

#### 4.3. Conflict-Aware and Balanced Dataset Partitioning

Using the PCA embedding, MERIT partitions datasets into disjoint groups so that datasets with similar projections are assigned to the same group while datasets with opposing projections are separated. This spreads group-wise updates across distinct, conflicting directions from θ(0).

A practical challenge is group imbalance: naive thresholding along PCA coordinates can yield groups with highly unequal data volumes. MERIT instead performs recursive 50/50 splits along PCA coordinates using sample-balanced medians, where each dataset carries weight st equal to its

sample count. This balances group sizes without sacrificing separation along dominant conflict axes. A comparison with distance-based clustering (K-means on the same gradient representation) is provided in Appendix C.3.

#### 4.4. Communication-Free Branch Training

After partitioning, MERIT fine-tunes one model per group, all initialized from θ(0) and sharing the same backbone, trainable-parameter subset, and hyperparameters; the only difference is which datasets each group sees. No crossgroup communication occurs during training, so branches can run in parallel on disjoint hardware.

For a fair comparison to centralized joint training, MERIT matches the total budget. Each dataset t is assigned to exactly one group k and contributes its full budget nt in that branch, so the per-group token count Nk := t∈G

nt satisfies k Nk = t nt, matching the joint-training baseline.

k

#### 4.5. Token-Weighted Merging

The resulting branch checkpoints θ1,...,θK are merged in a single pass via token-weighted averaging:

θ¯ =

K

Nk K j=1 Nj

wk θk, wk =

.

k=1

When per-group token budgets are exactly balanced, this reduces to uniform averaging; under our sample-balanced split the token budgets are only approximately balanced, so the merge is in general non-uniform.

### 5. Experiments

Setup. We evaluate MERIT in a controlled 3B study on Qwen2.5-VL-3B (Bai et al., 2025) with 136 Vision-FLAN tasks (Xu et al., 2024b), reporting 8 multimodal benchmarks grouped into four categories (General MCQA, User Preference & Fluency, Text-Rich VQA, Image Reasoning). Baselines include centralized joint training at 0.5/1/2 epochs, random partitioning (2/4/8 groups), conflict-induced splitting, and uniform model soups. Full experimental details (training recipe, hyperparameters, calibration set, per-benchmark protocols, and statistical significance tests) are provided in Appendix A. Large-scale 7B results and text-only transfer are reported in Section 6.

#### 5.1. Overall Performance

Table 2 summarizes the controlled 3B study. Every MERIT variant outperforms single-epoch joint training on the 8benchmark average; even random partitioning surpasses Joint 1 ep, consistent with Theorem 3.1 predicting a deterministic merging gain within a shared flat basin. The best variant, MERIT-3D, improves over the primary Joint

- Table 2. Controlled comparison of multimodal post-training strategies across diverse benchmarks. Avg. denotes the mean score over all benchmarks. Bold indicates the best result in each column, and underline indicates the second-best result in each column. Unless otherwise noted, all numbers are averaged over 3 independent runs; Joint training (1 ep) and MERIT-3D are averaged over 5 seeds as our primary comparison. Statistical significance tests for Joint training (1 ep) vs. MERIT-3D are reported in Appendix C.2.1.

Method

General MCQA User Preference & Fluency Text-Rich VQA Image Reasoning

Avg. SeedBench MMBench LLaVA-W MMVet TextVQA AI2D MathVista MMMU

Base model 66.8 79.7 53.2 34.0 61.2 63.8 29.6 41.2 53.7 Joint training (0.5 ep) 67.4 79.4 40.2 33.1 67.2 60.9 31.4 41.6 52.7

- Joint training (1 ep) 69.2 80.5 41.9 36.4 68.0 62.6 34.2 41.9 54.3
- Joint training (2 ep) 70.0 81.4 42.8 37.6 63.4 62.5 36.5 43.0 54.7

Random (2 groups) 69.4 80.1 44.5 34.7 70.4 62.7 34.0 41.2 54.6 Random (4 groups) 70.4 81.0 40.6 34.7 70.4 63.1 34.0 40.8 54.4 Random (8 groups) 69.5 79.9 42.2 35.0 73.7 61.7 33.5 40.5 54.5

Conflict-induced (2 groups) 70.7 80.6 42.6 35.4 70.0 62.9 34.4 42.3 54.9

- Uniform soup (2) 70.2 81.1 45.0 35.3 68.9 63.4 36.8 42.2 55.4

- Uniform soup (3) 70.1 81.1 42.3 36.3 68.8 63.1 35.8 42.5 55.0

- Uniform soup (4) 70.2 81.1 41.8 36.3 68.4 63.4 35.9 42.2 54.9

- MERIT (Proposed, 1D split, 2 groups) 71.0 80.0 43.1 35.0 72.4 62.1 36.5 41.4 55.2

- MERIT (Proposed, 2D split, 4 groups) 70.8 78.4 47.4 36.6 74.1 61.5 36.0 40.7 55.7

- MERIT (Proposed, 3D split, 8 groups) 70.5 80.1 52.0 37.7 75.2 62.5 35.4 42.7 57.0

- Table 3. Comparison over LLaVA-Series on diverse MLLM benchmarks under two 7B base models. We report each of the eight benchmarks together with their overall mean (Avg., computed only when all eight scores are available). For each base model, we compare further full fine-tuning via centralized Joint FFT against MERIT under a matched training budget; MERIT uses the 2D split with K=4 groups. Bold marks the better of Joint FFT vs. MERIT within the same base. For the 0.7M-base build we report the first seed, and its Joint FFT and MERIT are each validated over three independent seeds in Appendix C.2; the stronger 3.6M-base build is a single run.

General MCQA User Preference & Fluency Text-Rich VQA Image Reasoning

Model Train Data

Avg. SeedBench MMBench LLaVA-W MMVet TextVQA AI2D MathVista MMMU

LLaVA-7B 0.6M 37.0 38.7 57.2 25.5 – 48.3 25.4 34.1 – LLaVA-1.5-7B 0.7M 65.9 64.3 59.6 31.1 58.2 54.8 25.6 35.3 49.4 LLaVA-1.5-13B 0.7M 68.2 67.7 66.1 36.1 61.3 59.5 27.7 33.6 52.5 + Joint FFT (Xu et al., 2024b) 0.7M + 0.2M – 69.8 38.5 33.4 – – – 34.4 –

Base VLM 7B 0.7M 64.5 67.2 57.2 28.9 57.6 46.5 26.9 32.8 47.7 + Joint FFT 0.7M + 1.6M 70.1 76.2 58.8 32.5 73.8 58.6 32.9 36.4 54.9 + MERIT (Proposed, 2D) 0.7M + 1.6M 70.6 75.7 59.2 35.1 72.4 58.4 35.4 36.1 55.4

Scaled base VLM 7B 3.6M 69.8 74.5 67.1 34.0 72.4 69.2 39.4 45.1 58.9 + Joint FFT 3.6M + 1.6M 71.3 75.3 50.2 41.5 80.2 71.6 49.7 47.4 60.9 + MERIT (Proposed, 2D) 3.6M + 1.6M 71.5 75.8 66.2 39.1 79.8 71.9 43.0 44.8 61.5

- 1 ep baseline by +2.7 without any cross-partition gradient communication during fine-tuning. The pattern replicates at

- 7B scale across three independent seeds, confirming consistency (Table 9 in Appendix C.2).

#### 5.2. Conflict-Aware vs. Random Partitioning

A natural question is how much of MERIT’s gain comes from its conflict-aware split, rather than from splitting and merging alone. To isolate this, we compare MERIT against random partitioning under the same number of groups, the same budget, and the same one-shot merge; the only difference is how datasets are assigned.

The gap is substantial. MERIT-3D improves over Random (8 groups) by +2.5 under identical budgets and the same

one-shot merging step, attributable purely to the choice of split. Consistent with Proposition 3.2, the advantage grows monotonically with the number of PCA dimensions, while random partitioning shows no such trend. Beyond random splits, MERIT also outperforms K-means clustering on the same gradient representations, indicating that aligning the partition with conflict directions matters more than simply grouping similar datasets (Appendix C.3).

### 6. Further Analyses and Discussions

#### 6.1. Large-Scale Vision–Language Model Experiments

To test whether MERIT’s split-and-merge gains persist as we scale both model size and data, we apply it to further

- Table 4. Text-only benchmark results on Qwen2.5-3B with 66 FLAN tasks. Bold indicates the best result in each column, and underline indicates the second-best result in each column.

###### Knowledge QA Commonsense Text Inference Problem Solving

Method

Avg. MMLU GPQA HellaSwag WinoGrande BoolQ XNLI ARC-C HumanEval

Base model 65.5 29.5 74.5 70.1 77.3 42.9 55.6 37.8 56.7

- Joint training (1 ep) 65.8 30.1 74.0 69.9 83.9 41.9 55.5 39.7 57.6
- Joint training (2 ep) 66.1 30.6 76.3 69.2 82.9 41.9 56.5 42.7 58.3

Random (2 groups) 65.8 29.5 74.4 70.3 85.3 42.4 55.1 42.4 58.2 Random (4 groups) 66.0 30.3 74.7 70.3 85.7 42.5 54.9 42.1 58.3

Conflict-induced (2 groups) 65.8 28.7 74.0 69.4 85.3 42.2 53.8 42.1 57.7

- Uniform soup (2) 65.9 30.6 74.0 70.2 83.9 41.9 55.2 40.2 57.7

- Uniform soup (3) 65.7 30.8 74.0 70.2 84.2 42.1 55.7 39.0 57.7

- MERIT (Proposed, 1D split, 2 groups) 66.0 30.6 74.4 69.9 86.3 42.1 54.2 41.2 58.1

- MERIT (Proposed, 2D split, 4 groups) 66.1 30.0 74.7 70.8 86.1 42.4 55.3 41.5 58.4

full fine-tuning (FFT) of a 7B vision–language model on a

- 1.6M-example mixture drawn from 176 sources (details in Appendix A.2). We run the same comparison on top of two base models of increasing strength, holding the 1.6M FFT mixture and the training budget identical across methods. Table 3 reports the 8-benchmark average for each build.

Base VLM. The first base follows a standard LLaVAstyle recipe (feature alignment followed by a 0.7M-example instruction-tuning stage). Here MERIT improves the average over centralized Joint FFT (54.9 → 55.4) while achieving a more balanced profile: it recovers the open-ended degradation that Joint FFT suffers on User Preference & Fluency (+2.6 on MMVet) and adds a clear gain on Image Reasoning (+2.5 on MathVista), at the cost of staying within roughly a point and a half on text-rich benchmarks. Across three independent runs (Appendix C.2), MERIT outperforms Joint FFT on the 8-benchmark average in every run, confirming the gain is robust to training randomness.

Scaled base VLM. A natural concern is that MERIT’s gains might wash out once the base model is already strong. Recent recipes show that inserting a high-quality knowledge learning stage—training on large-scale high-quality image captioning data—substantially strengthens a VLM before instruction tuning (Li et al., 2024b; 2025). Following these recipes, we build a stronger base by adding this stage to the pipeline above: starting from feature alignment, we train on

- 2.9M image captioning samples (stage 1.5) and then on the 0.7M instruction-tuning data (stage 2), for 3.6M examples in total. This produces a markedly stronger starting point, on top of which we repeat the Joint FFT vs. MERIT comparison with the same 1.6M FFT mixture.

Even from this stronger initialization, MERIT again exceeds Joint FFT on the overall average (60.9 → 61.5), though the headroom over the base is naturally smaller for both methods. The gap is concentrated on open-ended generation:

Joint FFT collapses LLaVA-Wild to short, generic answers (67.1 → 50.2), whereas MERIT largely preserves the base model’s open-ended quality (66.2; cf. the short-answer collapse analysis in Appendix D.1). This collapse is not specific to this run: the same pattern appears in our primary 3B comparison (Table 2), averaged over five seeds, where Joint training drops LLaVA-Wild from the base model’s 53.2 to 41.9 while MERIT preserves it (52.0). On the more saturated perception and text-rich benchmarks the two remain comparable. On the reasoning-heavy benchmarks MERIT still improves over the base (MathVista 39.4 → 43.0) or holds it about flat (MMMU 45.1 → 44.8); Joint FFT reaches higher scores there, but in tandem with the open-ended collapse noted above, so the two differ mainly in capability profile while MERIT retains the better overall average.

Takeaway. Together, these results show that conflictaware splitting followed by one-shot merging generalizes to 7B-scale post-training under both larger mixtures and stronger initializations.

#### 6.2. Application to Text-only Instruction Tuning

We further validate MERIT on text-only instruction tuning using 66 FLAN (Wei et al., 2022) tasks with Qwen2.5-3B. Table 4 shows that MERIT-2D achieves the best average, exceeding 1-epoch joint training by +0.8 and matching or slightly exceeding 2-epoch joint training at half the budget, confirming that conflict-aware splitting generalizes beyond multimodal settings.

#### 6.3. Efficiency Analysis

MERIT is designed for bandwidth-constrained, decentralized settings, eliminating step-level synchronization by independently fine-tuning branch models and merging them once. Per-task gradient manipulation methods such as PCGrad (Yu et al., 2020) and GradNorm (Chen et al., 2018)

±

Figure 3. Post-hoc merging baselines applied to MERIT’s 7B

- 2D branches (K=4). Each operator replaces MERIT’s tokenweighted averaging as a drop-in merge step on the same four branches. Bars show the mean over 3 seeds on the 8-benchmark suite; circles mark per-seed scores and error bars denote ±1 std. Token-weighted averaging outperforms all four alternatives; we attribute this to MERIT’s branches being complementary by construction, so trimming- or orthogonalization-based operators can discard branch-specific content (Section 6.4).

are infeasible at our scale: PCGrad alone requires memory exceeding both our V100×8 and A100×8 systems at T=136 tasks on a 3B model (Appendix E.1). MERIT’s only preprocessing cost is dataset-level gradient estimation; with the stride-s subsampling introduced in Section 4, this cost is small and one-time.

At 3B scale, MERIT’s parallel branch training and one-shot merge run only ∼ 24% above single-epoch joint training on

- 8 V100 GPUs while achieving a substantially better average, and faster than two-epoch joint training; the one-time gradient-conflict preprocessing (∼2h, amortizable across runs) is reported separately. At 7B scale this recurring overhead drops to 0.8% on 8 A100 GPUs, with preprocessing an even smaller one-time fraction of total training (full breakdown in Appendix E).

#### 6.4. Post-Hoc Merging Baselines

A natural question is whether MERIT’s token-weighted averaging could be replaced with more sophisticated posthoc merging operators. We evaluate four state-of-the-art alternatives (TIES (Yadav et al., 2023), STAR (Lee et al., 2025b), TSV (Gargiulo et al., 2025), and Iso-CTS (Marczak et al., 2025)) as drop-in replacements on MERIT’s 7B 2D branches over three seeds. Token-weighted averaging consistently outperforms all four alternatives by 5–15 points on the 8-benchmark average (Figure 3). The reason is structural: these operators target redundancy among models specialized on the same task, but MERIT’s branches are complementary by construction—each carries task-specific information the others lack. Trimming or orthogonalizing such signals discards content only one branch holds, whereas token-weighted averaging preserves all branch contributions (per-seed results in Appendix C.5).

### 7. Conclusion

We introduced MERIT, a decentralized instruction-tuning pipeline grounded in a local flat-basin quadratic analysis showing that weight merging yields curvature-weighted variance reduction, that PCA-based conflict-aware splitting maximizes this gain, and that merging acts as spectral filtering with implicit norm regularization. MERIT splits datasets before fine-tuning using gradient-conflict PCA, trains group-wise branch models without cross-group synchronization, and merges them once via token-weighted averaging. Experiments on multimodal and text-only instruction mixtures (Qwen2.5-VL-3B on Vision-FLAN and a 7B-scale setting with a 1.6M-example mixture) confirm these implications and show consistent improvements on the 8-benchmark average over centralized joint training while enabling communication-free parallel training.

### Acknowledgements

We thank the anonymous reviewers and area chair for their constructive feedback. We are also grateful to our colleagues in the NAVER Cloud Hyperscale AI Vision Understanding Team for helpful discussions and support.

### Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Antonakopoulos, K., Mertikopoulos, P., Piliouras, G., and Wang, X. AdaGrad avoids saddle points. In International Conference on Machine Learning, pp. 731–771. PMLR, 2022.

Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., Jones, A., Joseph, N., Mann, B., DasSarma, N., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Kernion, J., Ndousse, K., Olsson, C., Amodei, D., Brown, T., Clark, J., McCandlish, S., Olah, C., and Kaplan, J. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu,

- Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang,
- Z., Xu, H., and Lin, J. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923, 2025.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N.,

Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra,

- V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba,
- W. Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374, 2021.

Chen, Z., Badrinarayanan, V., Lee, C.-Y., and Rabinovich, A. GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning, pp. 794–803. PMLR, 2018.

Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., Gu, L., Wang, X., Li, Q., Ren, Y., Chen, Z., Luo, J., Wang, J., Jiang, T., Wang, B., He, C., Shi, B., Zhang, X., Lv, H., Wang, Y., Shao, W., Chu, P., Tu, Z., He, T., Wu, Z., Deng, H., Ge, J., Chen, K., Zhang, K., Wang, L., Dou, M., Lu, L., Zhu, X., Lu, T., Lin, D., Qiao, Y., Dai, J., and Wang, W. Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2924–2936, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1300. URL https:// aclanthology.org/N19-1300/.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S., Schwenk, H., and Stoyanov, V. XNLI: Evaluating cross-lingual sentence representations. In Proceedings of the 2018 conference on empirical methods in natural language processing, pp. 2475–2485, 2018.

Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P. N., and Hoi, S. InstructBLIP: Towards generalpurpose vision-language models with instruction tuning.

Advances in neural information processing systems, 36: 49250–49267, 2023.

Davis, C. and Kahan, W. M. The rotation of eigenvectors by a perturbation. III. SIAM Journal on Numerical Analysis, 7(1):1–46, 1970.

Diao, Y., Li, Q., and He, B. Towards addressing label skews in one-shot federated learning. In The Eleventh International Conference on Learning Representations, 2023.

Draxler, F., Veschgini, K., Salmhofer, M., and Hamprecht, F. Essentially no barriers in neural network energy landscape. In International conference on machine learning, pp. 1309–1318. PMLR, 2018.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., Wu, Y., Ji, R., Shan, C., and He, R. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. URL https://openreview.net/forum? id=DgH9YCsqWm.

Gargiulo, A. A., Crisostomi, D., Bucarelli, M. S., Scardapane, S., Silvestri, F., and Rodola, E. Task singular vectors: Reducing task interference in model merging. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18695–18705, 2025.

Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D. P., and Wilson, A. G. Loss surfaces, mode connectivity, and fast ensembling of dnns. Advances in neural information processing systems, 31, 2018.

Gemma Team, Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ram´e, A., Rivi`ere, M., Rouillard, L., Mesnard, T., Cideron, G., bastien Grill, J., Ramos, S., Yvinec, E., Casbon, M., Pot, E., Penchev, I., Liu, G., Visin, F., Kenealy, K., Beyer, L., Zhai, X., Tsitsulin, A., Busa-Fekete, R., Feng, A., Sachdeva, N., Coleman, B., Gao, Y., Mustafa, B., Barr, I., Parisotto, E., Tian, D., Eyal, M., Cherry, C., Peter, J.-T., Sinopalnikov, D., Bhupatiraju, S., Agarwal, R., Kazemi, M., Malkin, D., Kumar, R., Vilar, D., Brusilovsky, I., Luo, J., Steiner, A., Friesen, A., Sharma, A., Sharma, A., Gilady, A. M., Goedeckemeyer, A., Saade, A., Feng, A., Kolesnikov, A., Bendebury, A., Abdagic, A., Vadi, A., Gy¨orgy, A., Pinto, A. S., Das, A., Bapna, A., Miech, A., Yang, A., Paterson, A., Shenoy, A., Chakrabarti, A., Piot, B., Wu, B., Shahriari, B., Petrini, B., Chen, C., Lan, C. L., Choquette-Choo, C. A., Carey, C., Brick, C., Deutsch, D., Eisenbud, D., Cattle, D., Cheng, D., Paparas, D., Sreepathihalli, D. S., Reid, D., Tran, D., Zelle, D., Noland, E., Huizenga, E., Kharitonov, E., Liu, F.,

Amirkhanyan, G., Cameron, G., Hashemi, H., KlimczakPluci´nska, H., Singh, H., Mehta, H., Lehri, H. T., Hazimeh, H., Ballantyne, I., Szpektor, I., Nardini, I., PougetAbadie, J., Chan, J., Stanton, J., Wieting, J., Lai, J., Orbay, J., Fernandez, J., Newlan, J., yeong Ji, J., Singh, J., Black, K., Yu, K., Hui, K., Vodrahalli, K., Greff, K., Qiu, L., Valentine, M., Coelho, M., Ritter, M., Hoffman, M., Watson, M., Chaturvedi, M., Moynihan, M., Ma,

- M., Babar, N., Noy, N., Byrd, N., Roy, N., Momchev,
- N., Chauhan, N., Sachdeva, N., Bunyan, O., Botarda, P., Caron, P., Rubenstein, P. K., Culliton, P., Schmid, P., Sessa, P. G., Xu, P., Stanczyk, P., Tafti, P., Shivanna, R., Wu, R., Pan, R., Rokni, R., Willoughby, R., Vallu, R., Mullins, R., Jerome, S., Smoot, S., Girgin, S., Iqbal, S., Reddy, S., Sheth, S., P˜oder, S., Bhatnagar, S., Panyam, S. R., Eiger, S., Zhang, S., Liu, T., Yacovone, T., Liechty, T., Kalra, U., Evci, U., Misra, V., Roseberry, V., Feinberg, V., Kolesnikov, V., Han, W., Kwon, W., Chen, X., Chow, Y., Zhu, Y., Wei, Z., Egyed, Z., Cotruta, V., Giang, M., Kirk, P., Rao, A., Black, K., Babar, N., Lo, J., Moreira, E., Martins, L. G., Sanseviero, O., Gonzalez, L., Gleicher, Z., Warkentin, T., Mirrokni, V., Senter, E., Collins, E., Barral, J., Ghahramani, Z., Hadsell, R., Matias, Y., Sculley, D., Petrov, S., Fiedel, N., Shazeer, N., Vinyals,
- O., Dean, J., Hassabis, D., Kavukcuoglu, K., Farabet, C., Buchatskaya, E., Alayrac, J.-B., Anil, R., Dmitry, Lepikhin, Borgeaud, S., Bachem, O., Joulin, A., Andreev,

- A., Hardin, C., Dadashi, R., and Hussenot, L. Gemma 3 Technical Report. arXiv preprint arXiv:2503.19786, 2025.

Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., Wang, X., Chen, L., Huang, F., Yacoob, Y., Manocha, D., and Zhou, T. HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14375–14385, June 2024.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021. URL https:// openreview.net/forum?id=d7KBjmI3GmQ.

Hochreiter, S. and Schmidhuber, J. Flat Minima. Neural Computation, 9(1):1–42, 01 1997. ISSN 0899-7667. doi: 10.1162/neco.1997.9.1.1. URL https://doi.org/ 10.1162/neco.1997.9.1.1.

Hu, A., Xu, H., Ye, J., Yan, M., Zhang, L., Zhang, B., Zhang, J., Jin, Q., Huang, F., and Zhou, J. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 3096–3120, 2024.

Huang, G. and Shu, T. Federated Oriented Learning: A Practical One-Shot Personalized Federated Learning Framework. In Forty-second International Conference on Machine Learning, 2025.

Hyeon-Woo, N., Ye-Bin, M., and Oh, T.-H. FedPara: Lowrank Hadamard Product for Communication-Efficient Federated Learning. In International Conference on Learning Representations, 2022. URL https:// openreview.net/forum?id=d71n4ftoCBy.

Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., and Farhadi, A. Editing models with task arithmetic. In International Conference on Learning Representations, 2023.

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., and Wilson, A. G. Averaging weights leads to wider optima and better generalization. In Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence (UAI), pp. 876–885. AUAI Press, 2018.

Jang, D.-H., Yun, S., and Han, D. Model stock: All we need is just a few fine-tuned models. In European Conference on Computer Vision, pp. 207–223. Springer, 2024.

Jin, C., Ge, R., Netrapalli, P., Kakade, S. M., and Jordan, M. I. How to escape saddle points efficiently. In International conference on machine learning, pp. 1724–1732. PMLR, 2017.

Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R., D’Oliveira, R. G. L., Eichner, H., Rouayheb, S. E., Evans, D., Gardner, J., Garrett, Z., Gasc´on, A., Ghazi, B., Gibbons, P. B., Gruteser, M., Harchaoui, Z., He, C., He, L., Huo, Z., Hutchinson, B., Hsu, J., Jaggi, M., Javidi, T., Joshi, G., Khodak, M., Koneˇcn´y, J., Korolova, A., Koushanfar, F., Koyejo, S., Lepoint, T., Liu, Y., Mittal, P., Mohri, M., Nock, R., Ozg¨¨ ur, A., Pagh, R., Raykova, M., Qi, H., Ramage, D., Raskar, R., Song, D., Song, W., Stich, S. U., Sun, Z., Suresh, A. T., Tram`er, F., Vepakomma, P., Wang, J., Xiong, L., Xu, Z., Yang, Q., Yu, F. X., Yu, H., and Zhao, S. Advances and open problems in federated learning. Foundations and trends® in machine learning, 14(1–2):1–210, 2021.

Kembhavi, A., Salvato, M., Kolve, E., Seo, M., Hajishirzi, H., and Farhadi, A. A diagram is worth a dozen images. In European conference on computer vision, pp. 235–251. Springer, 2016.

Kim, G. and Seo, M. On Efficient Language and Vision Assistants for Visually-Situated Natural Language Understanding: What Matters in Reading and Reasoning. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in

Natural Language Processing, pp. 16978–17000, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main. 944. URL https://aclanthology.org/2024.

emnlp-main.944/.

Lauren¸con, H., Tronchon, L., Cord, M., and Sanh, V. What matters when building vision-language models? Advances in Neural Information Processing Systems, 37: 87874–87907, 2024.

Lee, S., Kim, G., Kim, J., Lee, H., Chang, H., Park, S. H., and Seo, M. How Does Vision-Language Adaptation Impact the Safety of Vision Language Models? In The Thirteenth International Conference on Learning Representations, 2025a. URL https://openreview.

net/forum?id=eXB5TCrAu9.

Lee, Y.-A., Ko, C.-Y., Pedapati, T., Chung, I.-H., Yeh, M.-Y., and Chen, P.-Y. STAR: Spectral truncation and rescale for model merging. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pp. 496–505, 2025b.

Lei, Y., He, S., Hu, J., Zhang, D., Luo, X., Zhu, D., Feng, S., Liu, R., He, J., Sun, Y., Wu, H., and Wang, H. MoE Adapter for Large Audio Language Models: Sparsity, Disentanglement, and Gradient-Conflict-Free. arXiv preprint arXiv:2601.02967, 2026.

Li, B., Ge, Y., Ge, Y., Wang, G., Wang, R., Zhang, R., and Shan, Y. SEED-Bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13299–13308, 2024a.

Li, B., Zhang, H., Zhang, K., Guo, D., Zhang, Y., Zhang, R., Li, F., Liu, Z., and Li, C. LLaVANeXT: What Else Influences Visual Instruction Tuning Beyond Data?, May 2024b. URL https://llava-vl.github.io/blog/ 2024-05-25-llava-next-ablations/.

Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., and Li, C. LLaVA-OneVision: Easy Visual Task Transfer. Transactions on Machine Learning Research, 2025. ISSN 28358856. URL https://openreview.net/forum? id=zKv8qULV6n.

Li, Q., He, B., and Song, D. Practical One-Shot Federated Learning for Cross-Silo Setting. In Zhou, Z.-H. (ed.), Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pp. 1484– 1490. International Joint Conferences on Artificial Intelligence Organization, 8 2021. doi: 10.24963/ijcai.2021/

205. URL https://doi.org/10.24963/ijcai. 2021/205. Main Track.

Li, Q., Xie, C., Xu, X., Liu, X., Zhang, C., Li, B., He, B., and Song, D. Effective and efficient federated tree learning on hybrid data. In International Conference on Learning Representations, 2024c. URL https:// openreview.net/forum?id=py4ZV2qYQI.

Liu, B., Liu, X., Jin, X., Stone, P., and Liu, Q. Conflictaverse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878– 18890, 2021.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26296–26306, 2024a.

Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., and Lee, Y. J. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge, January 2024b. URL https://llava-vl.github.io/ blog/2024-01-30-llava-next/.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., Chen, K., and Lin, D. MMBench: Is Your Multi-modal Model an All-Around Player? In Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., and Varol, G. (eds.), Computer Vision – ECCV 2024, pp. 216–233, Cham, 2025. Springer Nature Switzerland. ISBN 978-3-031-72658-3.

Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le, Q. V., Zoph, B., Wei, J., and Roberts, A. The flan collection: Designing data and methods for effective instruction tuning. In International Conference on Machine Learning, pp. 22631–22648. PMLR, 2023.

Lopez-Paz, D. and Ranzato, M. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017.

Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=KUNzEQMWU7.

Marafioti, A., Zohar, O., Farr´e, M., noyan, M., Bakouch, E., Jim´enez, P. M. C., Zakka, C., allal, L. B., Lozhkov, A., Tazi, N., Srivastav, V., Lochner, J., Larcher, H., Morlon, M., Tunstall, L., Werra, L. V., and Wolf, T.

SmolVLM: Redefining small and efficient multimodal models. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum? id=qMUbhGUFUb.

Marczak, D., Magistri, S., Cygert, S., Twardowski, B., Bagdanov, A. D., and van de Weijer, J. No task left behind: Isotropic model merging with common and taskspecific subspaces. In International Conference on Machine Learning. PMLR, 2025.

Matena, M. S. and Raffel, C. A. Merging models with fisherweighted averaging. Advances in Neural Information Processing Systems, 35:17703–17716, 2022.

Mathew, M., Karatzas, D., and Jawahar, C. V. DocVQA: A Dataset for VQA on Document Images. In 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 2199–2208, 2021. doi: 10.1109/ WACV48630.2021.00225.

McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273–1282. PMLR, 2017.

Nesterov, Y. and Polyak, B. T. Cubic regularization of Newton method and its global performance. Mathematical programming, 108(1):177–205, 2006.

Neyshabur, B., Sedghi, H., and Zhang, C. What is being transferred in transfer learning? Advances in neural information processing systems, 33:512–523, 2020.

Qian, Y., Ye, H., Fauconnier, J.-P., Grasch, P., Yang, Y., and Gan, Z. MIA-Bench: Towards Better Instruction Following Evaluation of Multimodal LLMs. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=7EhS3YBxjY.

- Qin, Y., Qian, C., Yi, J., Chen, W., Lin, Y., Han, X., Liu, Z., Sun, M., and Zhou, J. Exploring mode connectivity for pre-trained language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 6726–6746, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.

451. URL https://aclanthology.org/2022. emnlp-main.451/.

- Qin, Z., Deng, S., Zhao, M., and Yan, X. FedAPEN: Personalized cross-silo federated learning with adaptability to statistical heterogeneity. In Proceedings of the 29th ACM SIGKDD conference on knowledge discovery and data mining, pp. 1954–1964, 2023.

Qin, Z., Chen, D., Qian, B., Ding, B., Li, Y., and Deng, S. Federated Full-Parameter Tuning of Billion-Sized Language Models with Communication Cost under 18 Kilobytes. In International Conference on Machine Learning, pp. 41473–41497. PMLR, 2024.

Qin, Z., Wu, Z., He, B., and Deng, S. Federated dataefficient instruction tuning for large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 15550–15568, 2025.

Qwen, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115, 2025.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Sakaguchi, K., Le Bras, R., Bhagavatula, C., and Choi, Y. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 8732–8740. AAAI Press, 2020. doi: 10.1609/aaai.v34i05.6399.

Sanh, V., Webson, A., Raffel, C., Bach, S., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Raja, A., Dey, M., Bari, M. S., Xu, C., Thakker, U., Sharma, S. S., Szczechla, E., Kim, T., Chhablani, G., Nayak, N., Datta, D., Chang, J., Jiang, M. T.-J., Wang, H., Manica, M., Shen, S., Yong, Z. X., Pandey, H., Bawden, R., Wang, T., Neeraj, T., Rozen, J., Sharma, A., Santilli, A., Fevry, T., Fries, J. A., Teehan, R., Scao, T. L., Biderman, S., Gao, L., Wolf, T., and Rush, A. M. Multitask Prompted Training Enables Zero-Shot Task Generalization. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=9Vrb9D0WI4.

Sch¨olkopf, B., Smola, A., and M¨uller, K.-R. Nonlinear component analysis as a kernel eigenvalue problem. Neural computation, 10(5):1299–1319, 1998.

Singh, A., Natarjan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards VQA Models That Can Read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8317– 8326, 2019.

Stich, S. U. Local SGD converges fast and communicates little. In International Conference on Learning Represen-

tations, 2019. URL https://openreview.net/ forum?id=S1g2JnRcFX.

Van der Maaten, L. and Hinton, G. Visualizing Data using tSNE. Journal of machine learning research, 9(11), 2008.

Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., and Lin, J. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Wei, J., Bosma, M., Zhao, V., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned Language Models are Zero-Shot Learners. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=gEZrGCozdqR.

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., and Schmidt, L. Model soups: averaging weights of multiple finetuned models improves accuracy without increasing inference time. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 23965–23998. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/ v162/wortsman22a.html.

Xu, M., Cai, D., Wu, Y., Li, X., and Wang, S. {FwdLLM}: Efficient federated finetuning of large language models with perturbed inferences. In 2024 USENIX Annual Technical Conference (USENIX ATC 24), pp. 579–596, 2024a.

Xu, Z., Feng, C., Shao, R., Ashby, T., Shen, Y., Jin, D., Cheng, Y., Wang, Q., and Huang, L. VisionFlan: Scaling human-labeled tasks in visual instruction tuning. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 15271–15342, Bangkok, Thailand, 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl. 905. URL https://aclanthology.org/2024.

findings-acl.905/.

Yadav, P., Tam, D., Choshen, L., Raffel, C. A., and Bansal, M. TIES-Merging: Resolving interference when merging models. Advances in Neural Information Processing Systems, 36:7093–7115, 2023.

Yu, L., Yu, B., Yu, H., Huang, F., and Li, Y. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In Forty-first International Conference on Machine Learning, 2024a.

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. Gradient surgery for multi-task learning. Advances in neural information processing systems, 33: 5824–5836, 2020.

Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., Hu, J., Liu, Z., Zheng, H.-T., and Sun, M. RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13807–13816, 2024b. doi: 10.1109/CVPR52733.2024.01310.

Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., and Wang, L. MM-Vet: Evaluating large multimodal models for integrated capabilities. In International Conference on Machine Learning. PMLR, 2024c.

Yue, X., Ni, Y., Zheng, T., Zhang, K., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., Wei, C., Yu, B., Yuan, R., Sun, R., Yin, M., Zheng, B., Yang, Z., Liu, Y., Huang, W., Sun, H., Su, Y., and Chen, W. MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9556–9567, 2024. doi: 10.1109/ CVPR52733.2024.00913.

Zelikman, E., Huang, Q., Liang, P., Haber, N., and Goodman, N. D. Just one byte (per gradient): A note on lowbandwidth decentralized language model finetuning using shared randomness. arXiv preprint arXiv:2306.10015, 2023.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791– 4800, Florence, Italy, 2019. Association for Computational Linguistics. URL https://aclanthology.

org/P19-1472/.

Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., YU, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., and Levy, O. LIMA: Less Is More for Alignment. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https:

//openreview.net/forum?id=KBMOKmX2he.

### Appendix

#### Table of Contents

- A. Experimental Details p. 15

- A.1. Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 15
- A.2. Large-Scale Vision–Language Model Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 16

- B. Merge-Readiness Diagnostics p. 17

- B.1. Linear Mode Connectivity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 17
- B.2. Weight Perturbation Robustness . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 17
- B.3. Displacement Contraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 18
- B.4. Training Loss vs. Generalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 18

- C. Additional Analyses and Ablations p. 18

- C.1. Visualization of Dataset-Level Gradients . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 18
- C.2. Robustness and Multi-Seed Reproducibility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 19
- C.3. Clustering Strategies for Dataset Partitioning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 20
- C.4. Calibration Dataset Size Sensitivity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 21
- C.5. Post-Hoc Merging Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 21

- D. Additional Qualitative Analysis p. 21

- D.1. Short-Answer Collapse on LLaVA-Wild . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 21
- D.2. Qualitative Examples on Multimodal Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 22

- E. Detailed Efficiency Analysis p. 22

- E.1. Comparison with Centralized Gradient Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 23
- F. Theoretical Analysis p. 23

- F.1. Quadratic Analysis of Merging in Flat PCA-Structured Basins . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 23

- F.2. Theoretical Connection Between Gradient PCA and Cosine-Similarity PCA . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 27
- F.3. Empirical Hessian–PCA Alignment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 29
- F.4. Formal Justification of Gradient–Curvature Alignment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 30
- F.5. Implicit Regularization via Averaging-Induced Contraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . p. 32

### A. Experimental Details

- A.1. Experimental Setup This section describes the experimental setup shared across all experiments unless otherwise specified.

- A.1.1. TRAINING DATASETS

Vision–Language Model Training Data. For vision–language model experiments, we conduct instruction tuning on Vision-FLAN (Xu et al., 2024b). For experiments at the 3B scale, we use 136 tasks out of the full set of 187 Vision-FLAN tasks for training. All comparison methods share the same dataset splits and training data.

Language Model Training Data. For text-only language model experiments, we conduct instruction tuning on the FLAN dataset (Wei et al., 2022). We use a subset of 66 instruction tasks for training in all language model experiments. All baselines and MERIT variants are trained on the same FLAN task mixture.

- A.1.2. EVALUATION BENCHMARKS

Multimodal Benchmarks. Our primary evaluation suite, used for every headline comparison in the main text (Sections 5–6), consists of eight widely used multimodal benchmarks that collectively assess general reasoning, open-ended understanding, text-centric perception, and expert-level reasoning: SeedBench (Li et al., 2024a) and MMBench (Liu et al., 2025) for general multimodal reasoning and compositional generalization; LLaVA-Wild (Liu et al., 2023) and MMVet (Yu et al., 2024c) for open-ended and fine-grained multimodal understanding; TextVQA (Singh et al., 2019) for text-centric visual understanding; AI2D (Kembhavi et al., 2016) for diagram-based reasoning; MathVista (Lu et al., 2024) for mathe-

matical reasoning; and MMMU (Yue et al., 2024) for expert-level multimodal reasoning across multiple images. For each benchmark, we report the standard metric defined by the dataset. The clustering-strategy ablation in Appendix C.3 adopts a broader 11-benchmark protocol that additionally includes MME (Fu et al., 2025), HallusionBench (Guan et al., 2024), DocVQA (Mathew et al., 2021), and MIABench (Qian et al., 2025) for robustness to evaluation protocol; LLaVA-Wild is excluded there because it is generation-only and does not fit the MCQA-centric aggregation used in that ablation.

Text-Only Benchmarks. For language models, we evaluate on a diverse set of text-only benchmarks covering reasoning, commonsense understanding, code generation, and cross-lingual generalization. These include MMLU (Hendrycks et al., 2021), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2020), ARC-C (Clark et al., 2018), HumanEval (Chen et al., 2021), BoolQ (Clark et al., 2019), GPQA (Rein et al., 2024), and XNLI (Conneau et al., 2018). We follow the standard evaluation protocols for all benchmarks.

- A.1.3. BASELINES

Joint Training. All datasets within each setting are trained jointly as a single corpus. We evaluate joint training for 0.5, 1, and 2 epochs to cover different optimization regimes. To give the joint baseline the strongest possible footing, we additionally grid-search its per-device batch size and report the best-performing configuration; MERIT branches use a single fixed batch size across all runs. Consequently, any reported gap understates MERIT’s advantage under matched tuning budgets.

Random Split. Datasets are randomly partitioned into 2, 4, or 8 groups. Each group is trained independently and the resulting checkpoints are merged via weight averaging.

Uniform Soup. We include a uniform model soup baseline following Model Soups (Wortsman et al., 2022). Multiple models are trained on the full dataset using different random data orders and merged by uniform averaging. We report soups constructed from 2, 3, and 4 models.

Conflict-Induced Split. We consider a conflict-induced split baseline based on greedy maximization of inter-group gradient disagreement. Starting from the most conflicting task pair, remaining tasks are assigned to minimize average cosine similarity within each group. Models are trained independently and merged using the same averaging procedure.

- A.1.4. IMPLEMENTATION DETAILS

Language Model Experiments For text-only experiments, we use Qwen2.5-3B (Qwen et al., 2025). Only language model parameters are trained, and all other settings follow the MERIT pipeline.

Vision–Language Model Experiments For vision–language experiments, we use Qwen2.5-VL-3B-Instruct (Bai et al., 2025), initialized from a standard two-stage LLaVA-style training recipe. During fine-tuning, the vision encoder and multimodal projector are frozen, and only the language model decoder is updated. Images are processed at a maximum resolution of 784×784 pixels. Unless otherwise specified, all methods share the same initialization and training configuration.

#### A.2. Large-Scale Vision–Language Model Experiments

This section provides details of the two base models and the data mixture used in our large-scale (7B) vision–language model experiments (Section 6.1). Unless stated otherwise, all training procedures follow the setup described in Appendix A.1.

Overview. For the 7B-scale study, we do not start from a released vision–language checkpoint; instead, we reuse the vision encoder from Qwen2.5-VL and pair it with the Qwen2.5-7B-Instruct language model, building our own merge-ready vision–language base from this combination via a LLaVA-style training recipe. The vision encoder is kept frozen throughout. The multimodal projector is tuned only while building the (scaled) base VLM; during the subsequent FFT stage (both Joint FFT and MERIT) it is frozen as well, so that this stage updates the language-model parameters alone. We consider two such base checkpoints of increasing strength, and on top of each we perform further full fine-tuning (FFT) on the same large-scale multimodal mixture of 1.6M examples spanning 176 dataset/task sources. Each source (or task-specific subset when a dataset provides multiple tasks) is treated as a dataset unit, following the definition in Section 4, and serves as the

basic unit for gradient conflict estimation and partitioning in MERIT. Both the 1.6M FFT mixture and the training budget are held identical across Joint FFT and MERIT.

Base VLM and Scaled base VLM. The Base VLM follows a standard two-stage LLaVA-style recipe: feature alignment followed by a 0.7M-example instruction-tuning stage (stage 2). The Scaled base VLM additionally inserts a high-quality knowledge-learning stage before instruction tuning, following recent recipes (Li et al., 2024b; 2025): starting from feature alignment, we train on a 2.9M-example re-captioned corpus (stage 1.5) and then on the same 0.7M instruction-tuning data (stage 2), for 3.6M examples in total. For stage 1.5 we use the publicly available LLaVA-ReCap-CC3M dataset (https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC3M). Both checkpoints are then finetuned on the same 1.6M FFT mixture described below, so that the only difference between the two builds is the strength of the initialization.

Data Sources (1.6M FFT mixture). The large-scale FFT mixture is drawn from publicly available datasets hosted on Hugging Face:

- • https://huggingface.co/datasets/X2FD/LVIS-Instruct4V
- • https://huggingface.co/datasets/Vision-Flan/vision-flan
- • https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
- • https://huggingface.co/datasets/zhiqings/LLaVA-Human-Preference-10K
- • https://huggingface.co/datasets/VictorSanh/LrvInstruction
- • https://huggingface.co/datasets/laion/gpt4v-dataset
- • https://huggingface.co/datasets/ys-zong/VLGuard

For Vision-FLAN and The Cauldron we use only a curated subset of the upstream task collection. The exact list of 176 task-unit identifiers and scripts will be released at https://github.com/naver-ai/merit.

Preprocessing and Mixture Construction. We convert all datasets into a unified multimodal instruction format compatible with our LLaVA-style training pipeline (image + instruction + response). When a dataset provides multiple task-specific subsets (as in Vision-FLAN and The Cauldron), each subset is preserved as its own task unit, yielding 176 units in total. We apply standard pre-processing and concatenate the units into a single 1.6M-example mixture used for FFT.

Reproducibility. The datasets are publicly available under their original licenses. More details, including the full base-model training configurations, will be released at https://github.com/naver-ai/merit. Multi-seed reproducibility of these 7B results is reported in Appendix C.2.

### B. Merge-Readiness Diagnostics

This section consolidates four complementary diagnostics that collectively verify the merge-ready initialization assumption underlying MERIT.

#### B.1. Linear Mode Connectivity

We evaluated loss barriers along all pairwise and branch-to-merged interpolation paths in the 3B 2D split (K=4) using 21 evenly spaced interpolation points (α ∈ [0,1], step 0.05). The barrier is defined as maxα[L(θ(α))−((1−α)L(0)+αL(1))], measuring the worst-case deviation above the linear interpolation.

#### B.2. Weight Perturbation Robustness

We applied isotropic Gaussian perturbations (σ ∈ {0.01,0.05,0.1}) to both the merged and jointly trained models at epochs 1–3. The merged model exhibits consistently lower loss sensitivity and lower flatness AUC (area under the ∆L–σ curve) at all perturbation levels and all epochs (Figure 4), confirming that the merged solution sits in a flatter region of the loss landscape.

- Table 5. Linear Mode Connectivity (2D split, K=4). All 10 barriers are exactly 0, confirming that branches remain in a shared flat basin with no loss barrier.

Path L(α=0) minα L(α) L(α=1) Barrier Branch ↔ Branch (pairwise)

- b0 ↔ b1 2.306 1.901 (α=0.70) 1.993 0.0
- b0 ↔ b2 2.306 2.244 (α=0.35) 2.468 0.0
- b0 ↔ b3 2.306 1.959 (α=0.65) 2.046 0.0

- b1 ↔ b2 1.993 1.972 (α=0.15) 2.468 0.0
- b1 ↔ b3 1.993 1.792 (α=0.45) 2.046 0.0 b2 ↔ b3 2.468 2.040 (α=0.95) 2.046 0.0 Branch → Merged

- b0 → merged 2.306 1.973 (α=1.0) 1.973 0.0
- b1 → merged 1.993 1.874 (α=0.50) 1.973 0.0
- b2 → merged 2.468 1.973 (α=1.0) 1.973 0.0
- b3 → merged 2.046 1.944 (α=0.60) 1.973 0.0

###### (a) Mean Loss (Epoch 1)

###### (b) Mean Loss (Epoch 2)

###### (c) Mean Loss (Epoch 3)

###### (d) Flatness AUC (lower = flatter)

0.36

0.36

0.36

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| || |
|---|
| | | | |
| || |
|---|
| | | | |
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
| | | | | |

0.34

0.34

0.34

3.5

0.32

0.32

0.32

| |
|---|

| |
|---|

| |
|---|

0.30

0.30

0.30

Loss

Loss

Loss

3.0

0.28

0.28

0.28

0.26

0.26

0.26

0.24

0.24

0.24

2.5

| |
|---|

| |
|---|

| |
|---|

0.22

0.22

0.22

0.20

0.20

0.20

FlatnessAUC

0 0.01 0.05 0.1 Perturbation magnitude

0 0.01 0.05 0.1 Perturbation magnitude

0 0.01 0.05 0.1 Perturbation magnitude

2.0

###### (e) DocVQA Loss (Epoch 1)

###### (f) DocVQA Loss (Epoch 2)

###### (g) DocVQA Loss (Epoch 3)

1.5

0.75

0.75

0.75

| |
|---|

0.70

0.70

0.70

0.65

0.65

0.65

1.0

Loss

Loss

Loss

0.60

0.60

0.60

0.55

0.55

0.55

0.5

0.50

0.50

0.50

| |
|---|

| |
|---|

| |
|---|

0.45

0.45

0.45

0.0

0 0.01 0.05 0.1 Perturbation magnitude

0 0.01 0.05 0.1 Perturbation magnitude

0 0.01 0.05 0.1 Perturbation magnitude

Epoch 1 Epoch 2 Epoch 3

Merged Joint

- Figure 4. Weight perturbation robustness. The merged model (blue) exhibits consistently lower loss sensitivity than the jointly trained model (red) at all perturbation levels and epochs, providing direct evidence of a flatter loss landscape.

B.3. Displacement Contraction

By convexity of the squared norm, weight averaging contracts the merged model toward the shared initialization: ∥θ¯w − θ(0)∥2 ≤ i wi∥θi − θ(0)∥2. Table 6 confirms this empirically: the merged model remains 2–3× closer to θ(0) throughout training, with the ratio widening monotonically.

B.4. Training Loss vs. Generalization

The merged model has substantially higher training loss than joint training yet achieves better held-out performance, the classic signature of implicit regularization from merging. The gap widens monotonically over training (Table 7).

C. Additional Analyses and Ablations

C.1. Visualization of Dataset-Level Gradients

- Figure 5 projects the dataset-level gradients at θ(0) to two dimensions via t-SNE. Tasks of the same type (e.g., VQA, image classification, and captioning) tend to cluster together, while different types occupy distinct regions of the gradient space.

- Table 6. Parameter displacement from shared initialization over training epochs.

Table 7. Training loss on the full mixture over epochs.

Epoch Ljoint Lmerged Gap

Epoch ∥θjoint − θ(0)∥2 ∥θ¯merge − θ(0)∥2 Ratio

0.5 0.709 1.198 +0.489 1.0 0.560 1.172 +0.611 2.0 0.370 1.167 +0.797 3.0 0.205 1.202 +0.998 6.0 0.064 1.330 +1.266

- 0.5 13.73 5.65 2.43×
- 1.0 19.73 7.50 2.63×
- 2.0 28.15 10.11 2.78× 6.0 34.61 11.87 2.92×

|LEV|R_CoG|enT+Q|uestio|n_Answ|ering|O|ffice_|31+ima|ge_cla|
|---|---|---|---|---|---|---|---|---|---|
|mult|iinstru|ct_VQ|A_colo|r| | |[Figure 86]| | |
| | | | | | |VisDA-|2017+|objec|t_class|
| | | | | | |NU|S-WID|E+anim|al_cl|
| |Core|50+Ob|ject_d|etectio|n| | | | |
|EMO|TION+|sentim|ent_de|tectio|n| | | | |
|xpw+|expre|ssion_d|etecti|on| | | | | |
| | | | | | |semart|+imag|e_desc|ription|
| | | | | | |WIT+de|tailed|_desc|ription|
|ONC|ADIA+|image_|captio|n_con|text| | | | |

A-OKVQA+visual_question_answering

0.027

ssification

0.0225

ification

0.0180

ssification

t-SNE Dimension2

Density(KDE)

0.0135

0.0090

0.0045

0.0000

t-SNE Dimension 1

- Figure 5. Two-dimensional t-distributed stochastic neighbor embedding (t-SNE) (Van der Maaten & Hinton, 2008) of dataset-level gradients at θ(0) (Qwen2.5-VL-3B, 136 Vision-FLAN tasks), overlaid with kernel density estimation (KDE) contours. Each marker is one task. Labeled examples are colored by task type: VQA (yellow), image classification (red), and description/captioning (green).

This task-type structure is what MERIT’s conflict-aware split exploits: gradients that disagree in direction fall into separable regions, so partitioning along the PCA conflict axes groups compatible datasets together.

#### C.2. Robustness and Multi-Seed Reproducibility

We assess MERIT’s reproducibility under training randomness at both scales: (i) the 3B primary comparison over five independent seeds with a paired Wilcoxon test, and (ii) a 7B replication over three independent seeds. Per-seed comparisons against alternative post-hoc merging operators on the same 7B 2D branches are provided separately in Appendix C.5, as they target a different question (aggregation rule choice rather than training-seed variability).

- C.2.1. 3B FIVE-SEED COMPARISON

We focus on the primary comparison used throughout the paper: MERIT (CosSim PCA, 3D split, 8 groups) versus Joint training (1 ep). Each method is evaluated over five independent runs with different random seeds, on the same eight benchmarks grouped into four categories (General MCQA, User Preference & Fluency, Text-Rich VQA, Image Reasoning). The reported Avg. denotes the mean score across all eight benchmarks.

Per-run results. Table 8 reports per-run results for both methods. MERIT consistently outperforms Joint training on the 8-benchmark average across all five independent runs.

Statistical Test. To further quantify the statistical significance of this improvement, we apply a paired Wilcoxon signedrank test on the per-run averaged scores. Using the five paired observations, the test yields a one-sided p-value of 0.03125,

- Table 8. Per-run results on the 3B setting comparing MERIT (CosSim PCA, 3D split, 8 groups) against Joint training (1 ep) over five independent runs. MERIT wins on the 8-benchmark average in all five runs (bold).

Run Method

General MCQA User Preference & Fluency Text-Rich VQA Image Reasoning

Avg. SeedBench MMBench LLaVA-W MMVet TextVQA AI2D MathVista MMMU

- Run 1

Joint (1 ep) 68.6 80.1 41.8 38.5 68.3 62.4 35.0 40.9 54.5 MERIT-3D 70.4 80.3 52.2 38.1 75.1 63.1 35.7 42.1 57.1

- Run 2

Joint (1 ep) 69.6 80.6 41.7 36.0 67.9 62.4 34.0 41.9 54.3 MERIT-3D 70.6 80.1 53.7 38.8 75.0 62.1 35.9 43.3 57.4

- Run 3

Joint (1 ep) 69.1 80.4 41.5 34.9 68.0 62.2 34.2 42.1 54.1

- MERIT-3D 70.5 79.5 49.9 36.5 75.1 62.8 34.3 42.8 56.4

Run 4

Joint (1 ep) 69.3 80.7 42.3 36.8 67.9 63.6 33.8 42.5 54.6

- MERIT-3D 70.5 80.2 52.7 38.0 75.3 61.5 34.7 42.3 56.9

- Run 5

Joint (1 ep) 69.5 80.5 42.1 35.8 67.7 62.6 34.0 41.9 54.3 MERIT-3D 70.3 80.5 51.7 37.1 75.4 63.2 36.6 43.2 57.3

which is statistically significant at the 5% level (p < 0.05). C.2.2. 7B THREE-SEED REPLICATION

To verify that the gains persist at scale, we repeated the full training pipeline (joint FFT and MERIT-2D) over three independent training seeds on the 7B setting. Table 9 shows that MERIT outperforms joint training in all three runs, with the same gain pattern as the 3B setting (open-ended reasoning benchmarks up, factual benchmarks within noise).

- Table 9. 7B results over three independent training seeds (Qwen2.5-VL-7B, 176 tasks). MERIT outperforms joint training in all three runs.

Run Method

General MCQA User Preference & Fluency Text-Rich VQA Image Reasoning

Avg. SeedBench MMBench LLaVA-W MMVet TextVQA AI2D MathVista MMMU

- Run 1

Joint FFT 70.1 76.2 58.8 32.5 73.8 58.6 32.9 36.4 54.9 MERIT-2D 70.6 75.7 59.2 35.1 72.4 58.4 35.4 36.1 55.4

- Run 2

Joint FFT 70.3 75.9 60.9 34.0 74.1 58.5 31.8 36.2 55.2 MERIT-2D 70.8 75.5 63.0 34.1 72.0 58.6 31.9 36.2 55.3

- Run 3

Joint FFT 70.1 75.9 60.8 34.1 77.1 58.6 31.7 35.9 55.5 MERIT-2D 70.4 75.3 60.1 35.6 76.3 58.8 33.0 36.4 55.7

C.3. Clustering Strategies for Dataset Partitioning

- Table 10. Ablation study on clustering strategies for dataset partitioning. We compare our PCA-based MERIT (Ours) against K-means clustering with different numbers of clusters.

Method MMBench MME SeedBench MathVista Hallusion AI2D MMVet MIABench MMMU TextVQA DocVQA Avg.

- Ours (MERIT, 1D) 80.0 1899.4 (82.6) 71.0 36.5 54.4 62.1 35.0 34.4 41.4 72.4 50.0 56.3

- Ours (MERIT, 2D) 78.4 1910.3 (83.0) 70.8 36.0 53.0 61.5 36.6 37.3 40.7 74.1 50.8 56.6

- Ours (MERIT, 3D) 80.1 1872.8 (81.4) 70.5 35.4 54.2 62.5 37.7 39.2 42.7 75.2 50.4 57.2

K-means (k = 2) 80.1 1834.5 (79.8) 70.2 34.3 55.1 62.1 36.9 34.4 38.6 71.2 49.3 55.5 K-means (k = 4) 79.7 1853.4 (80.6) 70.9 35.2 56.2 61.9 34.4 35.4 42.3 73.7 50.2 56.4 K-means (k = 8) 79.4 1833.1 (79.7) 70.4 35.1 53.6 62.1 35.1 39.4 41.1 75.0 50.4 56.5

- Table 10 compares PCA-based MERIT against K-means clustering (k ∈ {2,4,8}) on the same dataset-level gradient representations, with identical training and merging procedures. MERIT consistently outperforms K-means and scales monotonically with dimensionality, while K-means plateaus at a lower aggregate and exhibits non-monotonic per-benchmark patterns (e.g., MathVista, HallusionBench, MMVet). This confirms that aligning partitions with dominant conflict directions is more effective than generic distance-based clustering.

Random partitioning as a complementary baseline. Random partitioning followed by weight merging can already outperform joint training: under the flat-basin setting (Appendix F.1), any non-trivial dispersion among checkpoints yields a deterministic quadratic gain through variance cancellation, even when splits are random. This is consistent with Model Soups (Wortsman et al., 2022), which empirically observe that averaging compatible checkpoints reduces sharpness. However, random partitioning does not control where variance is introduced: the displacement covariance Σδ = K1 i(δi − δ¯)(δi − δ¯)⊤ is generally unstructured and need not align with dominant curvature directions, limiting the Hessian-weighted gain Gvar. MERIT’s PCA-based splitting explicitly concentrates dispersion along high-curvature conflict axes, yielding systematically larger gains (Tables 10 and main text Table 2).

#### C.4. Calibration Dataset Size Sensitivity

We fix calibration size to 200 samples per dataset throughout the paper. Table 11 reports calibration sensitivity across 100 Vision-FLAN tasks; we restrict this analysis to the 100 tasks that contain at least 1,000 samples, since the reference gradient requires n=1,000. At n=200, the mean cosine similarity with the full 1000-sample reference gradient is 0.847 (std 0.106), with diminishing returns beyond this point.

- Table 11. Gradient calibration sensitivity across 100 Vision-FLAN tasks. Left: cosine similarity between the calibration-subset gradient and the full 1000-sample reference, averaged over 100 tasks. Right: the same values as a function of n, with n=200 (used) marked.

n Mean cos Std

50 0.683 0.176 100 0.762 0.147 150 0.814 0.121 200 0.847 0.106 250 0.871 0.095 300 0.891 0.087 400 0.920 0.076 500 0.940 0.070 700 0.968 0.062

1000 1.000 0.000

0 200 400 600 800 1000

Calibration samples (n)

0.65

0.70

0.75

0.80

0.85

0.90

0.95

1.00

Meancosinesimilarity

n = 200 (used)

C.5. Post-Hoc Merging Baselines

This section provides per-seed results for the post-hoc merging comparison summarized in Section 6.4. Each baseline is applied as a drop-in replacement for token-weighted averaging on MERIT’s 7B 2D branches (K=4) over three independent seeds, using default hyperparameters from each method’s official repository.

- Table 12. Per-seed results for post-hoc merging baselines on MERIT’s 7B 2D branches. The Mean column matches Figure 3 in the main text.

Merging method Seed 1 Seed 2 Seed 3 Mean

Token-Weighted Avg 55.4 55.3 55.7 55.5 TIES (Yadav et al., 2023) 48.8 50.0 52.4 50.4 STAR (Lee et al., 2025b) 38.4 42.5 42.2 41.0 TSV (Gargiulo et al., 2025) 45.5 46.2 46.5 46.1 Iso-CTS (Marczak et al., 2025) 42.9 45.1 48.3 45.4

### D. Additional Qualitative Analysis

#### D.1. Short-Answer Collapse on LLaVA-Wild

Joint training on the 3B setting shows a large LLaVA-Wild degradation (53.2 → 41.9–42.8) due to short-answer collapse: the model produces overly terse responses under heterogeneous mixtures (Dai et al., 2023; Xu et al., 2024b). The same effect appears at 7B scale, where Joint FFT on the Scaled base collapses LLaVA-Wild from 67.1 to 50.2, whereas MERIT

Table 13. Additional qualitative examples from LLaVA-Wild.

Input (Image + Question) Model Responses

[Figure 87]

Q: What is the significance of this scene in the context of the movie?

###### Joint: love

- MERIT (1D): This scene is significant because it is the first time the couple is seen together.
- MERIT (2D): The scene captures a romantic moment between the two characters as they look out at the ocean together, with the sunset adding emotional tone.
- MERIT (3D): The scene depicts a key romantic moment between Jack and Rose, emphasizing their emotional connection and the beginning of their journey together, reinforced by the sunset setting.

[Figure 88]

Q: Can you explain why this is funny. Think about it step-by-step.

Joint: The joke is that the person is trying to convey that the VC is trying to get more money for the company, but the company is not stable enough to do so.

- MERIT (1D): The image is funny because it is a cartoon.
- MERIT (2D): The funny part of this is that the cartoon character is trying to explain something to the other character, but the other character is not listening.
- MERIT (3D): The funny part of this is that the cartoon character is talking about a computer science concept, but he is using a very simple and basic example. The character is talking about stacking layers, but he is only stacking two layers. This is a very basic example of a computer science concept, and it is not very interesting or complex. The fact that the character is using such a simple example makes it funny because it is not what one would expect to see in a computer science context.

[Figure 89]

Joint: This is a sketch of a website.

- MERIT (1D): The sketch is a white board with a blue line and text. The text is ”My Joke Website” and ”Funny Joke” and ”Push to reveal punchline”.
- MERIT (2D): A white board with a blue line and text.
- MERIT (3D): A white board with a blue line and the words ”My Joke Website” written on it.

Q: Describe this sketch in detail.

preserves the base model’s open-ended quality (66.2). In both cases, MERIT mitigates the collapse by separating conflicting subsets before fine-tuning, yielding more stable open-ended behavior; representative examples are shown in Table 13.

#### D.2. Qualitative Examples on Multimodal Reasoning

The examples above show that joint-trained models tend to produce terse or generic responses, while MERIT models provide more informative and context-aware answers. Higher decomposition dimensionality often (though not uniformly) yields richer and more context-aware answers, with the largest gains on responses that require multi-step reasoning or named-entity recall (e.g., Examples 1 and 2).

### E. Detailed Efficiency Analysis

Table 15 summarizes wall-clock times. Preprocessing is dominated by gradient extraction (∼1.6h for 136 datasets on V100, ∼2.0h for 176 on A100); cosine similarity computation adds ∼28–38 min and PCA takes < 1s. Table 16 confirms that uniform gradient sampling preserves cosine similarity structure with high fidelity (ρ>0.98).

Table 14. Hardware and system configuration used for efficiency analysis.

V100 System A100 System CPU

Model Intel Xeon Gold 5120 @ 2.20GHz Intel Xeon Gold 6338 @ 2.00GHz Sockets 2 2 Cores 28 (14 per socket) 64 (32 per socket) Threads 56 128

###### GPU

Model Tesla V100-SXM2-32GB NVIDIA A100-SXM4-80GB Number of GPUs 8 8 Memory per GPU 32GB 80GB Total GPU Memory 256GB 640GB

- Table 15. Wall-clock time breakdown of MERIT and joint training. Experiments are conducted with a 3B model on the V100 system (136 datasets, 3D split, K=8) and a 7B model on the A100 system (176 datasets, 3D split, K=8, measured with sequential branch execution). The 7B robustness results in Table 9 use the 2D split (K=4) configuration.

Stage V100 System A100 System Basin preparation (pre-alignment) 4h 24m / 20h 13m (LLaVA recipe stage 1/2) 39 h (Base VLM 7B) Gradient extraction 1h 38m (136 datasets) 1h 59m (176 datasets) Cosine similarity matrix computation 28m 38m PCA on cosine similarity matrix 0.7s (load) + 0.2s (compute) 0.6s (load) + 0.12s (compute)

- Joint training (1 ep) 4h 22m 43 h 18m

- Joint training (2 ep) 8h 40m – MERIT (CosSim PCA, 3D split) 5h 24m 43 h 39m

The similarity matrix is reusable: adding m new datasets to an existing T-dataset collection requires only O(Tm) crosssimilarity computations plus a negligible PCA update, avoiding full O((T+m)2) recomputation.

Counting branch training and merging (preprocessing reported separately above), MERIT-3D completes in 5h24m on V100, faster than 2-epoch joint training (8h40m) and only modestly above 1-epoch joint training (4h22m). At 7B scale on A100, MERIT adds just 21 minutes (0.8%) over 1-epoch joint training (43h39m vs. 43h18m). This one-time, amortizable preprocessing cost removes the need for synchronous gradient communication, making the approach well suited to decentralized or bandwidth-constrained environments.

#### E.1. Comparison with Centralized Gradient Methods

We provide a quantitative feasibility analysis for centralized per-step gradient conflict resolution methods at our experimental scale (T=136 tasks, 3B parameters).

PCGrad (Yu et al., 2020) requires T backward passes + T2 = 9,180 pairwise projections per step. Storing 136 full gradients requires ∼816GB (fp16), exceeding both our V100×8 (256GB) and A100×8 (640GB) systems. A conservative

wall-clock estimate is >17 days per epoch. GradNorm (Chen et al., 2018) requires all T=136 tasks to contribute individual loss terms at every step with centralized coordination. Both methods were originally evaluated on 2–10 tasks on models orders of magnitude smaller, and to our knowledge neither has been applied to >100 tasks on billion-parameter models.

### F. Theoretical Analysis

This appendix collects the formal assumptions, proofs, and extended analyses behind the three implications stated in Section 3.

#### F.1. Quadratic Analysis of Merging in Flat PCA-Structured Basins

This section makes precise the variance-reduction and spectral-filtering implications. We analyze the effect of merging through a deterministic quadratic gain from variance reduction, controlled remainder and dataset-split mismatch terms, and an optimization and regularization benefit over joint training.

- Table 16. Similarity between cosine similarity matrices computed using full gradients and uniformly sampled gradients (stride s = 5).

Metric Value

Pearson correlation (r) 0.9884 Spearman rank correlation (ρ) 0.9882 Mean absolute difference 0.0216 Root mean squared error (RMSE) 0.0355

Table 17. Computational comparison of conflict-resolution methods at T=136 tasks on a 3B model.

Method When resolved Per-step overhead Communication PCGrad Per-step O(T2)×d; ∼816GB All-reduce + T backpasses GradNorm Per-step O(T) forward All-reduce + per-task loss MERIT One-time (∼2h) Zero Zero (one-shot merge)

Setting and notation. Let θ ∈ Rd be the model parameters and L : Rd → R the true population loss. We consider K ≥ 2 checkpoints θ1,...,θK obtained by fine-tuning a common pretrained initialization θ(0) on different dataset splits via a two-phase procedure. In Phase I, pretraining and instruction tuning yield the merge-ready θ(0); in Phase II, the K branches are fine-tuned independently on their splits.

Because of the shared Phase I training, all checkpoints lie in one loss basin, a property widely observed in pretrained models and captured by linear mode connectivity (Garipov et al., 2018; Draxler et al., 2018; Qin et al., 2022). We fix a reference point θ⋆ inside this basin, a near-stationary point that serves as the center of the local analysis. For clarity we assume

∇L(θ⋆) = 0, noting that approximate stationarity ∥∇L(θ⋆)∥ ≤ ε only adds O εmaxi ∥δi∥ terms that do not change the conclusions. We write the deviations from θ⋆ and the parameter average as

1 K

δi := θi − θ⋆, i = 1,...,K, θ¯ :=

K

θi, δ¯ := θ¯− θ⋆,

i=1

and denote the Hessian of the true loss at θ⋆ by H := ∇2L(θ⋆). We present the uniform-weight case (wi = 1/K); the token-weighted case used in the algorithm is identical after replacing K1 i with i wi.

- F.1.1. TWO-PHASE BASIN CONFINEMENT AND LOCAL QUADRATICITY

The analysis rests on a local quadratic approximation of the loss, which is naturally induced by pretrained initialization and two-phase training.

#### Assumption 1 (Flat basin and local quadratic model). There exists a convex neighborhood B of θ⋆ such that:

- (a) (Basin confinement) All checkpoints and their average lie in B, i.e., θi ∈ B for all i and θ¯ ∈ B, as expected from a shared merge-ready initialization and confirmed empirically throughout training (Appendix B).
- (b) (Flatness) For all θ ∈ B, 0 ⪯ ∇2L(θ) ⪯ λmaxId for some λmax > 0, reflecting the flattening effect of pretraining on downstream landscapes (Neyshabur et al., 2020).
- (c) (Controlled remainder) The loss admits a second-order Taylor expansion around θ⋆, L(θ) = L(θ⋆) + 12(θ − θ⋆)⊤H(θ − θ⋆) + R(θ),

with |R(θ)| ≤ ρ6∥θ − θ⋆∥3 for some third-order smoothness constant ρ > 0 (Nesterov & Polyak, 2006; Jin et al., 2017; Antonakopoulos et al., 2022).

We write the quadratic surrogate of the true loss as

LQ(θ) := L(θ⋆) + 12(θ − θ⋆)⊤H(θ − θ⋆).

- STEP 1: QUADRATIC AVERAGING YIELDS A DETERMINISTIC GAIN

Under the quadratic model, averaging never hurts, and the benefit is exactly the spread of the checkpoints measured in the curvature (H) geometry.

- Theorem F.1 (Quadratic averaging bound). Under Assumption 1 and H ⪰ 0,

LQ(θ¯) ≤

1 K

K

i=1

LQ(θi),

with equality iff all displacements δi − δj lie in ker(H) (in particular, equality iff θ1 = ··· = θK when H ≻ 0). The improvement is strict whenever some pair (i,j) has (δi − δj) with a non-zero projection onto range(H) (the span of eigenvectors with positive eigenvalues), and admits the explicit form

1 K

K

i=1

LQ(θi) − LQ(θ¯) =

1 4K2

K

i=1

K

j=1

(δi − δj)⊤H(δi − δj)

=:Gvar≥0

.

The term Gvar is a Hessian-weighted variance: merging performs an explicit variance reduction, and the saved loss is larger along sharper (higher-curvature) directions (Izmailov et al., 2018). This is why where the checkpoints disagree matters more than how much.

STEP 2: ACCOUNTING FOR DATASET-SPLIT MISMATCH

The gain above is stated for the empirical loss seen during training. To extend it to the true objective we write L(θ) = L¯(θ) + ε(θ), where L¯ is the loss averaged over the dataset splits and ε is the population-level mismatch. A naive Lipschitz bound on the mismatch scales linearly (O(maxi ∥δi − δ¯∥)) and could in principle overwhelm the quadratic merging gain as checkpoints converge. We avoid this by treating the mismatch as smooth, which makes its penalty quadratic as well.

Assumption 2 (Second-order smoothness of mismatch). The mismatch ε(θ) is twice continuously differentiable in B; write Hε := ∇2ε(θ¯). With ∆split := ε(θ¯) − K1 Ki=1 ε(θi), a second-order expansion around θ¯(using i(θi − θ¯) = 0) gives

1 K

K

i=1

ε(θi) ≈ ε(θ¯) + ∇ε(θ¯)⊤ K 1

i

(θi − θ¯)

=0

+

- 1

- 2K

K

i=1

(θi − θ¯)⊤Hε(θi − θ¯),

so the penalty is bounded quadratically,

|∆split| ≲

- 1

- 2

λmax(Hε) ·

1 K

K

i=1

∥δi − δ¯∥2.

Hence both the gain Gvar and the penalty ∆split are O maxi ∥δi − δ¯∥2 , and the net effect of merging is decided by the relative spectra of H and Hε. Our PCA-structured split (Step 4) deliberately places dispersion in the curvature-bearing subspace of H, whereas the mismatch term is residual split-specific disagreement that need not align with those directions, so in this regime the gain dominates the penalty.

STEP 3: EXTENSION TO THE TRUE POPULATION LOSS

- Theorem F.2 (Merging in a flat basin (informal)). Under Assumptions 1–2 and the second-order mismatch approximation of Step 2, the merged model satisfies

L(θ¯) ≤

K

K

1 K

ρ 6 ∥δ¯∥3 +

1 K

∥δi∥3 .

##### − Gvar

+ |∆HO| + |∆split|, |∆HO| ≤

L(θi)

i=1

i=1

Quadratic Gain

Expected Individual Loss

- 0.5

0.75

- 1

[Figure 90]

- Group 1
- Group 2
- Group 3
- Group 4

[Figure 91]

Base model Merged model

0.25

0

PC2

- -0.75
- -0.5
- -0.25

-1

-1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 PC1

- Figure 6. PCA-based visualization of dataset groups in gradient space. The split groups extend along distinct, often conflicting directions from the shared initialization, while the merged model lies near the center, reflecting aggregation of complementary updates.

The gain Gvar is exact (Theorem F.1); the two remainders are the cubic Taylor residual |∆HO| and the second-order (hence approximate) split-mismatch penalty |∆split|.

Merging therefore helps whenever the true-loss curvature along the displacement direction outweighs the mismatch curvature. For small displacements confined to the basin the cubic term is negligible, and since Gvar is maximized by our PCA alignment while |∆split| stays bounded by the smaller curvature of task disagreement, the quadratic gain offsets both remainders.

- STEP 4: PCA-STRUCTURED PARTITIONING We now link the algorithmic choice of cosine-similarity PCA to the amplification of Gvar.

Assumption 3 (PCA-aligned subspace). Let U ∈ Rd×r be an orthonormal basis of the dominant curvature directions, so H ≈ UΛU⊤ + H⊥ with Λ ⪰ 0 (Marczak et al., 2025). Decompose δi = Uai + bi, and assume the split distributes checkpoints so that the mean projection onto the principal subspace is negligible, ∥a¯∥ ≪ K1 i ∥ai∥2. This assumption is justified by gradient accumulation. Under a first-order approximation of fine-tuning with a constant learning rate (here S denotes the number of SGD steps),

δi = −η

S

t=1

gi,t + ri, (3)

where ri collects higher-order effects (curvature variation, stochastic noise). Hence, up to second order, Cov(δi) ≈ η2 Cov t gi,t . Our algorithm runs PCA on the cosine-similarity matrix C (Appendix F.2) and splits along its principal axes, which enlarges the dispersion of δi along the dominant curvature directions U and thus enlarges Gvar relative to a random balanced split. We do not require the gradient-defined subspace to equal the top eigenspace of H; it suffices that the two are strongly coupled. Appendix F.4 makes this exact in a tractable linear-quadratic model where the two coincide: there, PCA-aligned splitting maximizes Gvar among all balanced partitions in the T=4, d=2 instance, and for general T it provably dominates random partitioning in expectation under a spectral-concentration condition.

Proposition F.3 (Effect of PCA-structured splitting). Under Assumption 3, PCA-structured splitting aligns the displacements δi primarily with the high-curvature subspace U, so Gvar concentrates along the largest-eigenvalue directions Λ and is maximized. In the ideal symmetric case ( i ai = 0), the high-curvature loss component is fully cancelled in the merged model, leaving only the residual loss on the flat subspace H⊥.

- STEP 5: OPTIMIZATION ADVANTAGE OVER JOINT TRAINING (CONCEPTUAL)

Beyond a lower loss, merging also eases the optimization difficulty of joint training. We give this as an interpretation of the deterministic averaging step rather than a formal guarantee; the SGD experiment in Step 6 provides the matching empirical support.

Table 18. Average performance under an SGD learning-rate sweep. MERIT (1D) stays stable across all η and MERIT (1D/2D) for η ≥ 8×10−5, whereas insufficient-epoch joint training collapses at small η; MERIT (3D) also degrades there, faster than 1D/2D, due to vanishing displacement variance (Appendix F.1, Step 6). Bold denotes the best result per column.

Learning rate η 3×10−5 4×10−5 8×10−5 2×10−4 4×10−4

Method

Joint (1 ep) 17.7 30.8 51.7 53.4 53.3 Joint (2 ep) 48.5 51.7 53.1 53.1 53.3

- MERIT (1D) 47.6 51.3 52.9 53.1 53.7
- MERIT (2D) 18.2 32.3 51.5 53.2 53.8
- MERIT (3D) 17.5 17.3 34.0 52.6 53.7

Joint training is slow for the following reason. Near θ⋆, descent on the full mixture evolves the error et := θt − θ⋆ as et+1 ≈ (I − ηH)et, so stability along the sharpest direction caps the learning rate at η < 2/λmax. When conflicting datasets keep injecting gradient fluctuations along these sharp directions, joint training must keep η small to avoid oscillation, which in turn slows progress along the flat directions (λmin ≪ λmax).

Merging sidesteps this. PCA-structured splitting followed by averaging makes the merged displacement nearly orthogonal to the sharp subspace, U⊤δ¯ ≈ 0 (Steps 1–4), so high-curvature error is suppressed in one shot instead of being slowly contracted. Conceptually, the remaining optimization is governed by an effective condition number

λ⊥,max λmin ≪ κ =

λmax λmin

κeff ≈

,

where λ⊥,max is the largest eigenvalue of H⊥. Merging thus trades the slow, stability-limited descent of joint training for an immediate cancellation of the stiffest conflicts.

- STEP 6: EMPIRICAL VERIFICATION UNDER SGD

The displacement model (3) matches SGD exactly, where the update is proportional to accumulated gradients. Adaptive optimizers such as Adam rescale per coordinate via mˆ t/(√vˆt + ϵ) and distort this covariance, so the theory of Steps 1–5 is tightest under SGD. We therefore use an SGD learning-rate sweep (Table 18) as a controlled stress-test of the mechanism; our main results use AdamW, so this study isolates the variance-reduction effect rather than reproducing those numbers. Two qualitative predictions follow from Steps 4–5 and Proposition F.3.

First, merging should help in the practical regime. For η ∈ [8×10−5, 4×10−4], displacements are large enough that PCAstructured splitting separates datasets along directions of maximal gradient divergence, generating non-trivial inter-group variance that averaging then cancels along high-curvature directions. Table 18 confirms this for 1D and 2D splits.

Second, finer splits should degrade at small η. As η → 0, Gvar ∝ η2 → 0 for any partition; in addition, 3D splits give each branch a more homogeneous subset, shrinking ∥g¯1 − g¯2∥ independently of η. Together these push Gvar ∝ η2 ∥g¯1 − g¯2∥2H to zero faster than for 1D/2D. Table 18 shows MERIT (3D) collapsing at η ≤ 4×10−5, as predicted.

These results match the two predictions above: PCA-aligned splitting concentrates inter-group variance along high-curvature directions (Proposition F.3), and the conditioning benefit of merging (Step 5) keeps MERIT (1D) stable across all η and MERIT (1D/2D) stable in the practical regime (η ≥ 8 × 10−5), even where insufficient-epoch joint training collapses.

#### F.2. Theoretical Connection Between Gradient PCA and Cosine-Similarity PCA

This section establishes when PCA on cosine-similarity conflict matrices recovers the same dominant interaction structure as PCA on raw gradients, justifying cosine-similarity PCA as a scale-invariant proxy for the gradient-space analysis above.

- F.2.1. PROBLEM SETUP AND NOTATION

Let n be the number of datasets and d the parameter dimension. For dataset i, let gi ∈ Rd be the population gradient, collected as G := [g1,...,gn]⊤ ∈ Rn×d. We compare two constructions:

• Raw gradient PCA, equivalent to the eigendecomposition of the Gram matrix (Sch¨olkopf et al., 1998) K := GG⊤ ∈

Rn×n;

Table 19. Statistics of dataset-level gradient norms across 136 Vision-FLAN datasets. We report the mean (µ), standard deviation (σ), coefficient of variation (σ/µ), and max-to-min ratio under raw and trimmed settings to assess gradient norm concentration.

Metric Raw 5% Trimmed 10% Trimmed

Number of datasets 136 124 110 Mean gradient norm (µ) 23.87 22.90 22.67 Standard deviation (σ) 9.27 5.27 3.88 Coefficient of variation (σ/µ) 0.39 0.23 0.17 Max / Min ratio 7.12 3.12 2.05

• Cosine-similarity PCA, the eigendecomposition of C ∈ Rn×n with Cij := gi⊤gj/(∥gi∥∥gj∥). Our goal is to characterize when PCA on C recovers the same dominant subspace as PCA on G.

- F.2.2. COSINE SIMILARITY AS A NORMALIZED GRAM MATRIX Lemma 1 (Normalization). With D := diag(∥g1∥,...,∥gn∥), the cosine-similarity matrix satisfies

C = D−1KD−1. (4)

Proof. Immediate from Cij = Kij/ KiiKjj. ■ Cosine similarity is thus a two-sided rescaling of the Gram matrix: it strips out per-dataset gradient magnitude while keeping directional alignment.

- F.2.3. SUBSPACE EQUIVALENCE UNDER GRADIENT NORM CONCENTRATION

Equivalence of the leading eigenspaces of K and C requires the gradient magnitudes to be roughly balanced, a mild condition in large models, where dataset-level gradients are averaged over many samples.

Assumption 4 (Gradient norm concentration). There exists ϵ ≪ 1 with ∥gi∥ = α¯(1 + δi), |δi| ≤ ϵ, where α¯ > 0 is a reference scale (e.g., the mean norm).

Empirical justification. Across 136 Vision-FLAN datasets (Table 19), raw norms vary, but removing a few extreme datasets sharply improves concentration: the coefficient of variation drops from 0.39 (raw) to 0.23 (5% trimmed) and 0.17 (10% trimmed), with the trimmed max-to-min ratio near 2–3. This supports the small-perturbation regime of Theorem F.4.

Theorem F.4 (Leading eigenspace equivalence). Let Vr(K) and Vr(C) be the spans of the top-r eigenvectors of K and C. Under Assumption 4, if K has a spectral gap λr − λr+1 ≥ γ > 0, then

dist Vr(K),Vr(C) = O

ϵ∥K∥2 γ

, (5)

where dist(·,·) is the canonical subspace distance via principal angles.

Proof Sketch. Write D = α¯(I + E) with ∥E∥2 = O(ϵ). Then

C = D−1KD−1 = α¯−2(I + E)−1K(I + E)−1 = α¯−2(K + ∆), (6)

with ∥∆∥2 = O(ϵ∥K∥2). Scalar multiplication leaves eigenvectors unchanged, so the perturbation is governed by ∆; the Davis–Kahan sinΘ theorem (Davis & Kahan, 1970) gives the bound. ■

- F.2.4. DIRECTIONAL INTERPRETATION VIA NORMALIZED GRADIENTS

With normalized gradients g˜i := gi/∥gi∥ and G˜ := [˜g1,...,g˜n]⊤, we have C = G˜G˜⊤. Thus cosine-similarity PCA is PCA on unit-sphere-projected gradients, consistent with Theorem F.4: magnitude normalization does not move the leading subspace under norm concentration.

Table 20. Direct comparison between MERIT instantiated with raw-gradient PCA and cosine-similarity PCA. Method SeedBench MMBench LLaVA-W MMVet TextVQA AI2D MathVista MMMU Avg.

- Raw PCA, 1D 70.9 79.9 41.5 35.0 72.5 62.7 36.2 41.3 55.0
- Raw PCA, 2D 71.0 81.4 44.8 35.4 73.4 62.6 36.4 42.5 55.9
- Raw PCA, 3D 70.6 81.2 40.0 35.4 75.4 62.2 33.6 41.0 54.9

- CosSim PCA, 1D 71.0 80.0 43.1 35.0 72.4 62.1 36.5 41.4 55.2
- CosSim PCA, 2D 70.8 78.4 47.4 36.6 74.1 61.5 36.0 40.7 55.7
- CosSim PCA, 3D 70.5 80.1 52.0 37.7 75.2 62.5 35.4 42.7 57.0

Table 21. Empirical Hessian–PCA alignment between top-3 PCA conflict directions and top-10 Hessian eigenvectors at θ(0). Alignment is partial in absolute terms (leading principal angles 55–70◦) but well above the Gaussian random baseline (z ≫ 1) at both scales.

Metric Qwen2.5-VL-3B (d≈3B) SmolVLM-256M (d≈256M) Raw max |cos| 0.30 0.48 Raw mean top-1 |cos| 0.136 0.332 (2.4×↑) Raw leading principal angle 69.5◦ 55.5◦ Raw z-score (vs. Gaussian) 15,386 17,973 Centered z-score 19,385 15,858

- F.2.5. EMPIRICAL AND PRACTICAL IMPLICATIONS

Empirically (Table 20), at 1D and 2D the two PCA constructions give nearly identical averages (within 0.2 points) and the same ordering (2D > 1D), consistent with the leading-subspace equivalence of Theorem F.4; the upward trend on text-heavy benchmarks (e.g., TextVQA, MMVet) appears in both. At 3D the two diverge: cosine-similarity PCA continues to improve while raw-gradient PCA does not, which motivates the scale-invariant cosine construction adopted below. Practically, cosine-similarity PCA is the convenient, scale-invariant choice: it emphasizes directional disagreement and is robust to dataset-specific gradient-norm differences, without altering the dominant interaction subspace. We therefore use it as the default instantiation of MERIT.

#### F.3. Empirical Hessian–PCA Alignment

The link between gradient-PCA and Hessian curvature (Section 3) predicts that PCA conflict directions should partially align with the top Hessian eigenvectors. We test this on two MLLMs spanning roughly an order of magnitude in trainableparameter scale: Qwen2.5-VL-3B (Bai et al., 2025) (d≈3B) and SmolVLM-256M (Marafioti et al., 2025) (d≈256M).

Setup. We use a four-stage protocol per model. (i) Task-level gradients. From Vision-FLAN tasks with ≥ 50 samples we sample 136 tasks (seed=42) and compute a per-task SFT gradient over a fixed set of 50 examples per task; the same example indices are reused in stage (iii). (ii) PCA-based task selection. We form the 136×136 cosine-similarity matrix, double-center it, embed via the top-2 eigenvectors, and pick the task farthest from the origin in each PC1/PC2-sign quadrant, giving 4 representative tasks. (ii.5) Conflict directions. On the 4×4 Gram matrix Gij=⟨gi,gj⟩ we take the top-3 eigenvectors and lift them to Rd, for both the raw Gram and its double-centered variant. (iii) Hessian top eigenvectors. On the 4×50=200 examples from the selected tasks (reusing the stage (i) indices) we run 50 Lanczos iterations with full reorthogonalization (fp32 central-difference Hessian–vector products, ε=0.1), yielding the top-10 Hessian eigenvectors at θ(0). (iv) Alignment. We report |cos(ui,hj)|, principal angles, and an alignment z-score against a Gaussian random-vector baseline, for both constructions.

Results. Table 21 reports exactly the alignment our theory predicts. The overlap between PCA conflict directions and the top-10 Hessian eigenvectors is partial in absolute terms—leading principal angles of 55–70◦ and max|cos| of 0.30–0.48matching the claim that gradient-PCA identifies high-curvature axes preferentially, not perfectly. That this partial overlap is nonetheless far from random is confirmed by the Gaussian-baseline z-scores (≥104 at both scales), which rule out chance alignment but do not by themselves imply tight alignment. SmolVLM-256M aligns more strongly than Qwen2.5-VL-3B (top-1 mean |cos| is 2.4× larger, leading principal angle 14◦ smaller), consistent with curvature concentrating along fewer directions at smaller scale. The double-centered variant tracks the raw one closely.

#### F.4. Formal Justification of Gradient–Curvature Alignment in a Tractable Setting

This section proves the splitting-optimality claim (Section 3, Proposition 3.2) in a tractable linear-quadratic model. The model is simplified but captures the essential mechanism: gradient conflict at a shared initialization is curvature-coupled, so PCA on gradient similarities naturally finds the directions where merging gains the most.

Setup. Consider T datasets with quadratic losses sharing a Hessian,

Lt(θ) = 12 (θ − θt⋆)⊤H (θ − θt⋆), t = 1,...,T,

where H = diag(λ1,...,λd), λ1 ≥ ··· ≥ λd > 0. The joint optimum is θ¯⋆ = T1 t θt⋆, and we initialize at θ(0) = θ¯⋆ so the mean gradient vanishes. The gradient of dataset t at θ(0) is

gt = ∇Lt(θ(0)) = −H ∆t, ∆t := θt⋆ − θ¯⋆, t ∆t = 0. Key identity: gradients are curvature-amplified. The gradient covariance is

1 T t

Σg :=

1 T t

gt gt⊤ = H Σθ H, Σθ :=

∆t ∆⊤t .

Remark F.5 (Curvature amplification). Since Σg = H Σθ H, gradient-PCA is steered not by the optima spread Σθ alone but by HΣθH: in the diagonal case the gradient variance along coordinate j is λ2j Var(∆t,j). Directions that are both high-curvature and high-disagreement are amplified quadratically, so PCA on gradient conflicts preferentially surfaces high-curvature axes.

Fine-tuning approximation. For a balanced split into G1,G2 (|G1|=|G2|=T/2), a single-step approximation gives θk ≈ θ(0) − η g¯k with g¯k = |G1

k| t∈Gk gt and displacement δk = −η g¯k. The merging gain (Theorem 3.1) for w1 = w2 = 1/2 is

2

Gvar = 18 (δ1 − δ2)⊤H (δ1 − δ2) = η

8 (¯g1 − g¯2)⊤H (¯g1 − g¯2). (7)

Partition encoding. Encode a balanced split by signs s ∈ {±1}T with t st = 0 (st = +1 if t ∈ G1). Then g¯1 − g¯2 = T2 t st gt and

2

Gvar(s) = η

2T2 s⊤M s, Mij := gi⊤H gj = ∆⊤i H3 ∆j, so the gain is governed by the H3-weighted gradient Gram matrix M = GHG⊤.

Concrete example (T=4, d=2). Let H = diag(λ1,λ2) with λ1 > λ2 > 0 and centered optima ∆1 = (α,β), ∆2 = (−α,β), ∆3 = (α,−β), ∆4 = (−α,−β) (α,β > 0), so gt = −H∆t. The three distinct balanced partitions give:

- • P1 (PCA-aligned, split along coord. 1): g¯1 − g¯2 = (−2λ1α,0), Gvar(1) = η

2

2 λ31α2.

- • P2 (orthogonal, split along coord. 2): g¯1 − g¯2 = (0,−2λ2β), Gvar(2) = η

2

2 λ32β2.

- • P3 (canceling): g¯1 − g¯2 = (0,0), Gvar(3) = 0.

The gradient Gram matrix has coordinate variances λ21α2 and λ22β2, so its top PCA direction is coordinate 1 exactly when λ21 α2 > λ22 β2, (8)

under which PCA selects P1. Proposition F.6 (PCA-aligned splitting maximizes Gvar). Under (8), P1 attains the maximum gain among all balanced partitions: Gvar(1) = η

2

2

2 λ32β2 = Gvar(2) > 0 = Gvar(3).

2 λ31α2 ≥ η

3 1α2

Proof. It suffices that λ31α2 ≥ λ32β2 under (8), which gives α2/β2 > (λ2/λ1)2. Then λ

λ32β2 = (λ1/λ2)3(α2/β2) > (λ1/λ2)3(λ2/λ1)2 = λ1/λ2 ≥ 1.

| |
|---|

Corollary F.7 (PCA split dominates random split in expectation). A uniformly random balanced partition has expected gain E[Gvarrand] = η

2

2

6 (λ31α2 + λ32β2) < η

2 λ31α2 = GvarPCA.

Proof. Averaging the three equally likely partitions gives 13(Gvar(1) + Gvar(2) + Gvar(3)). The strict inequality holds whenever λ31α2 > 12λ32β2, guaranteed by Proposition F.6.

| |
|---|

- Remark F.8 (Why the λ3 scaling). The PCA-aligned split concentrates variance along λ1 for gain ∝ λ31α2; the orthogonal split yields ∝ λ32β2; the canceling split yields 0. The λ3 scaling appears because curvature enters three times—twice through the gradient conflict (gt ∝ H∆t, giving H2) and once through the gain formula (Gvar ∝ δ⊤Hδ, giving H1). This is why the advantage of conflict-aware splitting grows with the curvature gap λ1/λ2.
- Remark F.9 (Gradient PCA is a safe proxy). The Gvar-optimal split is set by M = GHG⊤ (eigenvalue weight λ3j), while gradient PCA uses K = GG⊤ (weight λ2j). Since λ3 is even more biased toward high curvature than λ2, whenever gradient

PCA picks the high-curvature axis the Gvar-optimal split agrees (λ21α2 > λ22β2 ⇒ λ31α2 > λ32β2). Gradient PCA is thus a sufficient (if not strictly necessary) criterion for the gain-maximizing split, which is why MERIT can partition on gradient similarities alone.

General result for T datasets. We now move beyond T=4 to compare PCA-aligned and random partitioning for general T. Retain the quadratic setting with Mij = ∆⊤i H3 ∆j. Since t gt = 0, M1 = 0, so 0 is an eigenvalue with eigenvector 1/

##### √

T; let µ1 ≥ ··· ≥ µT−1 > 0 = µT be the eigenvalues with unit eigenvectors v1,...,vT. For a balanced split s, Gvar(s) = η

2

##### 2T2 s⊤M s.

- Proposition F.10 (Expected gain under random partitioning). For s drawn uniformly from balanced sign vectors, E[Gvarrand] = η2

2T(T−1) tr(M) = η

2

2T(T−1)

T−1 k=1 µk.

Proof. For a uniform balanced sign vector, E[s2i] = 1 and E[sisj] = −T−1 1 (i ̸= j). Hence E[s⊤Ms] = tr(M) −

1

T−1(1⊤M1 − tr(M)) = TT−1 tr(M), using M1 = 0. Dividing by 2T2/η2 gives the result.

| |
|---|

- Proposition F.11 (PCA-aligned splitting dominates random splitting). Let s∗ = sign(v1), where v1 is the leading eigenvector of M (adjusted for balance if needed1). Then

2

2

GvarPCA = η

2T2 (s∗)⊤M s∗ ≥ η

2T2 µ1 ∥v1∥21, (9) where ∥v1∥1 = t |v1,t|. Consequently GvarPCA > E[Gvarrand] whenever

µ1 ∥v1∥21 > TT−1 tr(M). (10)

Proof. Decompose s∗ = k(s∗⊤vk)vk, so (s∗)⊤Ms∗ = k µk(s∗⊤vk)2 ≥ µ1(s∗⊤v1)2. With s∗ = sign(v1), s∗⊤v1 =

t |v1,t| = ∥v1∥1, giving (9); comparing with Proposition F.10 yields (10).

| |
|---|

The dominance condition (10) asks for two things. First, spectral concentration: µ1 should be a large fraction of tr(M) =

k µk, i.e., the conflict structure has a dominant axis. This is exactly the regime where structured splitting helps, whereas isotropic conflicts leave no room over random. Second, eigenvector delocalization: ∥v1∥1 should be large (1 ≤ ∥v1∥1 ≤

##### √

T, with √

T when entries are equal-magnitude), meaning many datasets contribute to the dominant axis, a mild requirement for large mixtures. Corollary F.12 (Limiting regimes). Under Proposition F.11:

- 1. Rank-1 conflicts: if M ≈ µ1v1v1⊤ and v1 is maximally spread (|v1,t| = 1/

√

T), then GvarPCA/E[Gvarrand] ≥ T − 1: PCA captures all conflict structure while random wastes a (1 − 1/T) fraction.

- 2. Isotropic conflicts: if µ1 = ··· = µT−1 = tr(M)/(T−1), then (10) is tight and no strategy beats random—when every direction matters equally, there is no preferred axis to split along.

1When T is even and v1 has no zero entries, sign(v1) is balanced w.h.p. for generic v1; otherwise we assume balance or reassign at most one dataset, in which case (9) holds up to a single-entry correction.

Proof. (1) With tr(M) = µ1 and ∥v1∥21 = T: GvarPCA ≥ η2µ1/(2T) and E[Gvarrand] = η2µ1/(2T(T−1)), giving ratio ≥ T−1. (2) With uniform eigenvalues every balanced split has the same expected projection onto each eigenspace, so structure gives no advantage.

| |
|---|

Real multimodal conflicts are far from isotropic: the 2D PCA projection in Figure 6 shows a few principal components separating the branch-wise updates along well-defined axes. This places MERIT in the spectral-concentration regime of Corollary F.12(1), where PCA-aligned splitting helps the most.

Remark F.13 (Recursive splitting along orthogonal PCA axes). MERIT splits recursively along the top-r (orthogonal) PCA eigenvectors {uj}rj=1 of C. In the linear-quadratic model, dispersions from successive splits lie on disjoint eigendirections, so cross-axis contributions to s⊤Ms vanish and the gain decomposes additively:

Gvar(r) ∝

r

λ3j αj2,

j=1

where αj is the group-mean dispersion along uj. Each PCA dimension adds a non-negative, cubically curvature-weighted term, predicting the monotone 1D → 2D → 3D gains reported in Section 5.

#### F.5. Implicit Regularization via Averaging-Induced Contraction

This section analyzes the implicit regularization from averaging, beyond the variance reduction Gvar. Empirical validation (displacement contraction, perturbation robustness, train-loss vs. generalization gap, linear mode connectivity) is consolidated in Appendix B.

Displacement decomposition. With displacements δi := θi − θ(0) and δ¯ := θ¯− θ(0) (recall θ⋆ = θ(0) in our analysis, so these coincide with the δi of Appendix F.1), the parallel-axis identity gives, exactly,

K

1 K

1 K

∥δi∥2 = ∥δ¯∥2 +

i=1

K

∥δi − δ¯∥2. (11)

i=1

So unless all checkpoints coincide, averaging gives a strictly smaller squared displacement, ∥θ¯ − θ(0)∥ ≤

1 K i ∥θi − θ(0)∥2: merging contracts the model toward the shared initialization.

Empirical validation. Displacement contraction, perturbation robustness, and the train-loss vs. generalization gap are verified in Appendix B (Tables 6–7 and Figure 4).

One mechanism, two regularizers. Both the quadratic gain Gvar (Theorem F.2) and the contraction (11) are instances of the same identity K1 i ∥ · ∥2 = ∥mean∥2 + variance, applied under the H-induced norm (loss along high-curvature directions) and under the Euclidean norm (displacement from initialization), respectively. Averaging reduces variance under both, giving MERIT spectral filtering and norm contraction at once.

Connection to PAC-Bayes. Many PAC-Bayes bounds for deterministic predictors scale with the squared distance to a reference prior, often the pretrained initialization. Under a Gaussian prior centered at θ(0), the complexity term is KL(Q∥P) ∝ ∥θ − θ(0)∥2. Since the merged model stays strictly closer to θ(0) than the jointly trained one (Table 6), ∥θ¯− θ(0)∥ < ∥θjoint − θ(0)∥, it incurs a strictly smaller distance-based complexity term, all else equal.

