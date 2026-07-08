# arXiv:2604.19835v2[cs.LG]10May2026

## Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts

#### Chaitanya Dwivedi1,2 Binxuan Huang1,3 Himanshu Gupta1 Pratik Jayarao1,2 Neeraj Varshney1 Bing Yin1

1Amazon Stores Foundation AI 2Carnegie Mellon University 3Anthropic cdwivedi@alumni.cmu.edu

### Abstract

Mixture-of-Experts (MoE) has become the dominant architecture for scaling large language models: frontier models routinely decouple total parameters from pertoken computation through sparse expert routing. Scaling laws show that under fixed active computation, model quality scales predictably with total parameters, and MoEs realize this by increasing expert count. However, training large MoEs is expensive, as memory requirements and inter-device communication both scale with total parameter count. We propose expert upcycling, a method for progressively expanding MoE capacity by increasing the number of experts during continued pre-training (CPT). Given a trained E-expert model, the upcycling operator constructs an mE-expert model through expert duplication and router extension while holding top-K routing fixed, preserving per-token inference cost. Duplication provides a warm initialization: the expanded model inherits the source checkpoint’s learned representations, starting from a substantially lower loss than random initialization. Subsequent CPT then breaks the symmetry among duplicated experts to drive specialization. We formalize the upcycling operator and develop a theoretical framework decomposing the quality gap into a capacity term and an initialization term. We further introduce utility-based expert selection, which uses gradient-based importance scores to guide non-uniform duplication, more than tripling gap closure when CPT is limited. In our 7B→13B total parameter experiments, the upcycled model achieves lower validation loss than the fixed-size baseline while saving ∼32% of GPU hours at 50% CPT. Extending to 100% CPT saves ∼24% and closes the downstream accuracy gap to within 0.3 points (average across 11 benchmarks). Comprehensive ablations across model scales, activation ratios, MoE architectures, and training budgets yield a practical recipe for deploying expert upcycling, establishing it as a principled, compute-efficient alternative to training large MoE models from scratch1.

### 1 Introduction

Mixture-of-Experts (MoE) models have become the dominant architecture for scaling language models efficiently [48, 22, 7, 38]. By routing each token to K out of E total experts, MoEs decouple total parameters from per-token compute. Scaling-law analyses show that under fixed active computation, model quality scales predictably with total parameters [35, 1], with the activation ratio (K/E) identified as the primary driver of MoE efficiency over dense models [52]. MoEs realize this by increasing expert count at fixed K, expanding capacity without increasing inference cost. Frontier MoE models have pushed this aggressively: Qwen3 activates 22B of 235B total parameters [51],

1We release our code and training configurations at https://github.com/amazon-science/expert-upcycling.

Preprint.

[Figure 1]

Figure 1: Overview of the expert upcycling procedure. Step 1: Pre-train an E-expert MoE for τ steps. Step 2: Apply the upcycling operator Um at step τ: each expert e is replicated re ≥ 1 times (high-utility experts receive more copies, rE > ri ≥ ··· ≥ r1, s.t. re = m · E), and the router is extended with replicated slots plus bias noise. All copies are identical at τ, providing a warm initialization. Step 3: Continue pre-training on the expanded mE-expert model for T−τ steps; stochastic gradient diversity breaks symmetry among duplicates, driving specialization. Top-K routing is fixed throughout, so active parameters and per-token compute are unchanged.

DeepSeek-V3 activates 37B of 671B [7], and Kimi K2 activates 32B of 1T [38], all matching or exceeding dense models many times their active size.

Despite these favorable scaling properties, training MoEs with a large number of total parameters from scratch is expensive. All expert weights, gradients, and optimizer states must reside in accelerator memory regardless of how few experts are active per token, so memory requirements, and therefore the number of GPUs needed, scale with the total parameter count [58, 35]. Further, distributing experts across devices introduces all-to-all communication that can consume 45–50% of total training time on standard GPU clusters [41]. Both costs grow with E, making it progressively more expensive to train MoEs at the low activation ratios that scaling laws recommend.

This tension motivates expert upcycling, a capacity-expansion strategy for MoEs that obtains the quality benefits of a larger MoE without paying its full training cost from scratch. Rather than committing to the full expert count from step 0, training begins with a smaller E-expert model. At a chosen transition step τ, the upcycling operator expands the model to mE experts by duplicating existing experts and extending the router, increasing total parameters while holding active parameters and per-token FLOPs fixed. This two-phase strategy is strictly cheaper than fixed-size training: both phases process the same tokens, but the first τ steps execute on the smaller model. Unlike a randomly initialized larger MoE, the upcycled model inherits the source checkpoint’s learned representations, providing a warm initialization that starts at approximately the same loss. Continued pre-training (CPT) then breaks the symmetry among duplicated experts, driving them to specialize.

To our knowledge, expert upcycling is the first method to progressively grow MoE capacity during training while preserving inference cost by holding top-K routing fixed. This distinguishes it from dense progressive training methods [5, 9] and width-expansion approaches for MoEs [57], which increase active parameters and thus inference cost, and from sparse upcycling [26], which converts a dense model into an MoE but does not address capacity expansion within already-sparse architectures.

Our main contributions are:

- (i) We propose expert upcycling, a method for progressively expanding MoE capacity by increasing the number of experts during continued pre-training, and formalize the duplication and router-extension operator (§3). In our 7B→13B total parameter experiments, the upcycled model achieves lower validation loss than the fixed-size baseline while saving ∼32% of GPU hours at 50% CPT; extending to 100% CPT saves ∼24% and closes the downstream accuracy gap to within 0.3 points (average across 11 benchmarks). Across activation ratios, expert upcycling (MoE→MoE) consistently outperforms sparse upcycling [26] (dense→MoE) as target activation ratio decreases.
- (ii) Within expert upcycling, we introduce utility-based expert selection, a novel operator that uses gradient-based importance scores to guide non-uniform duplication. Utility-based selection consistently outperforms uniform duplication, more than tripling gap closure when CPT budget is limited (§3.3).
- (iii) We develop a theoretical framework that decomposes the quality gap into a capacity term and an initialization term, yielding testable predictions for when upcycling succeeds, and validate these predictions through comprehensive experiments across model scales (154M–7B total parameters), activation ratios (3–50%), MoE architectures, training budgets, and transition points, deriving a practical recipe for practitioners (§3, §5). Code is provided as supplemental material and will be released on GitHub upon publication (footnote 1).

Together, these results establish expert upcycling as a principled, compute-efficient paradigm for training sparse models where progressive capacity expansion is not merely a fallback for reusing existing checkpoints, but has the potential to become the recommended training strategy from the outset.

### 2 Related work

Scaling Mixture-of-Experts models. Sparsely-gated MoE layers enable high-capacity models with limited per-token computation [48], and recent work has scaled this paradigm to open-source frontier models [8, 12, 29, 22, 36, 7, 51, 38, 13]. Across these systems the trend is consistent: total parameters grow aggressively while active parameters per token remain fixed, directly instantiating the scalinglaw prediction that lower activation ratios yield better quality-per-FLOP trade-offs [52, 35, 27]. Expert upcycling exploits precisely this property: by expanding total expert count mid-training while holding top-K fixed, it captures the quality benefits of a larger MoE without paying the full training cost from scratch.

Growing model capacity during training. Progressive training grows networks mid-run to amortize compute: Net2Net introduced function-preserving width/depth transforms [5] and recent work extends these to Transformers via layer stacking [9, 2, 3]; SPARKLING [57] brings mid-training width expansion to MoE models. All of these grow depth or dense width, raising active parameters and inference cost at each step. A complementary line of work upcycles capacity from an existing checkpoint: Sparse Upcycling converts a dense checkpoint into an MoE [26], with scaling laws derived in Liew et al. [33]; follow-ups improve initialization diversity [40], explore parameter-efficient variants [59, 21], and study router design at scale [19]. These perform a dense→MoE transition. Closer to our setting, Nexus [14] expands an existing MoE by adaptively integrating new domainspecific experts. Expert upcycling instead expands expert count within the MoE regime by duplicating existing experts while holding top-K fixed, increasing total parameters without increasing active parameters or inference cost.

Load balancing and saliency metrics. Imbalanced routing causes representation collapse in MoE, starving some experts of gradient signal [6]; auxiliary balancing losses [48, 12] address this but trade off task performance, while loss-free load balancing [53] corrects imbalance via router-bias updates with no loss modification. We adopt the loss-free scheme throughout so that every duplicated replica receives differentiated gradient signal, a prerequisite for symmetry breaking after upcycling. Saliency metrics repurposed from pruning (weight magnitude [16], second-order sensitivity [28, 17], first-order Taylor approximations [37], and Fisher-based estimators [49, 31], with expert-level variants [34]) identify which experts contribute most to the loss; we invert the pruning direction and use the same

scores to choose which experts to duplicate, so pruning and upcycling act as duals (one recovers inference efficiency, the other expands training capacity).

### 3 Expert upcycling

We introduce expert upcycling, a capacity-expansion procedure that grows the number of experts in an MoE model mid-training by reusing learned parameters. Given an E-expert MoE trained for τ gradient steps, expert upcycling constructs an mE-expert model by duplicating existing experts and extending the router, formalized as the operator Um in §3.2, then continues training for the remaining T − τ steps. The conventional alternative, which we call fixed-size training, trains the mE-expert model from random initialization for all T steps. Both approaches process the same number of training tokens and produce a model with mE experts and identical per-token FLOPs since top-K is fixed.

For expert upcycling to be a viable alternative to fixed-size training, two conditions must hold: (i) it must be cheaper in total training compute, and (ii) it must close the quality gap to the from-scratch model. We address compute efficiency in §3.1 and quality gap closure in §3.2.

#### 3.1 Compute efficiency of expert upcycling

The two approaches differ in training cost. Let sE and smE denote the per-step training time of the E-expert and mE-expert models respectively. The larger model is more expensive per step due to increased memory requirements, gradient and optimizer-state updates over all expert parameters, and all-to-all communication overhead [10, 24], giving sE < smE. In our experimental setup (§4), we measure smE ≈ 1.9 × sE when doubling expert count (∼2.2s vs. ∼4.2s per step at 7B→13B model scale).

The total training cost under each approach is Cfs for fixed-size training and Cup for expert upcycling: Cfs = T × smE,

Cup = τ × sE + (T − τ) × smE. (1) Expert upcycling is strictly cheaper because the first τ steps execute on the smaller model:

Cfs − Cup = τ × (smE − sE) > 0. (2) The saving grows linearly with τ and the per-step cost gap smE − sE. In our 7B-scale experiments, we find that τ ≈ 23T is sufficient for the upcycled model to match the from-scratch validation loss (§5), which translates to ∼32% reduction in GPU hours.

Expanding existing models. When a trained E-expert checkpoint already exists, for instance from a prior training run or a public release, the compute advantage is even larger: the pre-training cost τ × sE is sunk, and expert upcycling requires only the incremental (T − τ) × smE for CPT on the expanded model. This makes expert upcycling particularly attractive for continued pre-training of publicly available MoE models. In our 7B-scale experiments, the sunk-cost setting at τ ≈ 32T reduces GPU hours by ∼67% compared to fixed-size training (∼50% at τ = 12T).

#### 3.2 Quality gap closure and the upcycling operator

A cheaper procedure is only useful if it preserves quality. Adapting the OCO regret-telescoping analysis of Bu [3] for progressive training, we decompose the gap between upcycling and fixed-size training as a sum of two interpretable terms (formal derivation in Appendix A).

- • Capacity gap (Term I): during the first τ steps, training is in the E-expert model, a strictly less expressive class than the mE-expert target. The gap between the two classes’ optima is inherited by the upcycled model in proportion to τ/T.
- • Initialization gain (Term II): at step τ, the upcycled model must be initialized in the expanded parameter space. The closer this initialization is to the mE-expert optimum (compared to fixedsize’s random initialization), the larger upcycling’s head start.

The compute saving τ(smE − sE) from §3.1 and the capacity gap (Term I) pull τ in opposite directions: larger τ saves more compute but widens Term (I), while smaller τ shrinks Term (I) but

erodes the compute advantage. We perform an ablation in §5.2.1 to identify a good operating point for τ. Term (II), in contrast, is independent of τ: it depends only on how the expanded parameters are constructed, offering a second, τ-independent lever to close the quality gap. We therefore introduce an expert upcycling operator designed to maximize the initialization gain by producing a warm start close to the mE-expert optimum.

Definition 3.1 (Expert Upcycling Operator). Fix an integer expansion factor m ≥ 2. Given a trained E-expert parameter vector θE ∈ ΘE, the operator Um: ΘE → ΘmE constructs Um(θE) as follows:

- 1. Expert replication. Assign replication counts {re}Ee=1 with re ≥ 1 and e re = mE; copy the parameters of expert e exactly re times. The canonical choice is uniform replication (re = m for all e); non-uniform allocations are developed in §3.3.
- 2. Router extension. Copy the router weight vector of source expert e to each of its re replicas. Add independent noise ϵ ∼ U(−δ,δ) (with δ ≪ 1) to the router biases of replicated experts only, leaving the source experts’ router parameters unchanged.

By construction, each new expert starts from a trained weight vector and the router approximately reproduces the pre-expansion routing distribution. All other parameters (attention layers, embeddings, and layer norms) are unchanged. The post-expansion loss therefore satisfies LmE(Um(θE)) ≈ LE(θE), with a gap below 10−2 in practice (see §5). We refer to this property as warm initialization: Um places the expanded parameters substantially closer to the mE-expert optimum than random initialization, reducing Term (II).2

#### 3.3 Utility-based upcycling

Definition 3.1 leaves the replication counts {re} unspecified; uniform replication (re = m for all e) is the natural baseline. However, MoE experts are heterogeneous in their contribution to the objective [34], so a non-uniform allocation that concentrates capacity on high-importance experts should yield a warm start closer to the mE-expert optimum, tightening Term (II). We allocate replicas using gradient-based importance scores repurposed from the structured pruning literature [28, 17, 37]. We evaluate two first-order utility scores, both computed from the gradient ge = ∇weL evaluated at transition step τ:

- • Squared gradient norm: uG(e) = ∥ge∥22, which captures how sensitive the loss is to each expert’s parameters.
- • Weight–gradient saliency: uSAL(e) = ∥we∥2 · ∥ge∥2, which combines parameter magnitude with gradient signal.

Both scores can be derived from a first-order Taylor expansion of the loss (Appendix B). Replicas are allocated greedily to the highest-scoring experts. Both scores offer similar improvements over uniform duplication, with squared gradient norm marginally but consistently outperforming weight– gradient saliency. We also tried curvature-normalized variants (∥ge∥22/He) and weight-norm-only scoring (∥we∥22); neither worked as well (Appendix B).

Beyond utility-based selection, we evaluated a broader family of diversity-inducing heuristics at the expert and router level (noise injection, orthogonalization, drop-based re-initialization, interpolation, and others; 20 variants in total, detailed in Appendix D). None exceed simple copy-paste duplication by more than 10−3 in validation loss, indicating that warm-start quality in this setting is dominated by which experts are replicated rather than how the copied parameters are perturbed.

#### 3.4 Post-expansion dynamics

Immediately after applying Um, the replicated experts are near-identical copies. Three mechanisms break this symmetry during CPT: the router bias perturbation from Um creates initial routing asymmetry; loss-free load balancing [53] ensures every replica receives gradient signal; and stochastic gradient diversity drives a self-reinforcing cycle of specialization (different parameters → different routing → different gradients).

2Exact function preservation à la Net2Net [5] is structurally unavailable here because top-K routing is discrete; expert upcycling therefore targets low initialization loss directly rather than exact preservation.

### 4 Experimental setup

Architecture and training. Our main result uses a 20-layer interleaved MoE, which alternates dense and MoE layers as in Llama 4 [36], with ∼7B→13B total and ∼1B active non-embedding parameters. We focus on the interleaved architecture for the majority of our experiments, including both recipe ablations and the 7B-scale run, as only half the layers incur all-to-all communication, substantially reducing per-step training time [10, 7] and allowing faster iteration on an NVIDIA A100 GPU cluster. Most ablations for deriving the practical upcycling recipe are conducted at the ∼1B total parameter scale on the same interleaved architecture. To assess generalizability to full MoE, we conduct an additional ablation at the ∼1B scale on a full MoE architecture with 256 experts and top-K=8, matching the routing configuration of frontier MoE models [7, 13, 38]. All models use top-K gating with K ∈ {2,8} and no shared experts, Grouped Query Attention (GQA), and RoPE positional embeddings. Full model architecture details and optimal training hyperparameters, as determined by preliminary scaling sweeps, are provided in Tables 5–6 (Appendix C). Optimization follows a Warmup–Stable–Decay (WSD) schedule with loss-free load balancing [53]. Training is performed using data parallelism and tensor parallelism.

Data. To separate the effects of pre-training from continued pre-training (CPT), we use disjoint data splits: the base models are trained on the pre-training split, and all CPT stages are performed on a separate CPT split, thereby avoiding data leakage between stages. Small-scale ablation experiments use DCLM [30]. The 7B-scale main experiment uses a curated data mixture emphasizing instruction following, logical reasoning, and math.

Evaluation protocol. For each experiment, we compare three configurations: a fixed E-expert model with no expansion, our upcycled E→mE model, and a fixed mE-expert model trained from scratch, all at matched total token budget. Each training stage concludes with a 10% annealing phase. For the 7B-total/1B-active main result, we report both downstream benchmark accuracy (11 benchmarks) and validation loss. For the ∼1B-total/∼144M-active-non-embedding ablation experiments, we report validation loss only, as most downstream benchmarks do not reliably differentiate models at this scale [55]. To compare across settings, we report upcycling efficiency over validation loss L:

L(Fixed-E) − L(Upcycled) L(Fixed-E) − L(Fixed-mE)

η =

, (3) which measures normalized gap closure; a value of 1 indicates complete gap closure.

Upcycling procedure. At transition step τ, optimizer states are reset (matching sparse upcycling [26]), and utility scores are computed from gradients averaged over 10 batches. Replicas are allocated greedily subject to a per-expert duplicate cap of n=3, preventing any single expert from monopolizing the allocation. The router-bias noise in Definition 3.1 uses δ = 10−3.

### 5 Results

We first demonstrate expert upcycling at the full 7B→13B scale (§5.1). The design choices for this experiment (operator, CPT budget, and transition timing) were informed by systematic ablations at the ∼1B scale (§5.2), which study each mechanism of upcycling and yield a practical recipe.

#### 5.1 Expert upcycling at scale

Table 1 reports validation loss and downstream benchmark accuracy across 11 tasks for Fixed-32, the upcycled 32→64 model (gradient-norm guided duplication, §5.2.2), and Fixed-64 at 50% and 100% CPT. At 50% CPT, the upcycled model surpasses Fixed-64 on validation loss (efficiency 109.7%); average downstream accuracy across 11 benchmarks is within 1.3 points of Fixed-64 (52.1 vs. 53.4), saving ∼32% of GPU-hours. Commonsense and language understanding tasks (HellaSwag, PIQA, Social IQA, OpenBookQA) match or exceed Fixed-64 at this point. The remaining gap concentrates in knowledge and reasoning tasks (MMLU, BBH, GSM8K, IFEval), which continue to improve with additional CPT. By 100% CPT, these tasks largely converge as well (e.g., BBH: 43.8 vs. 45.0; GSM8K: 48.3 vs. 49.1), bringing average accuracy to 56.4 vs. 56.7 and validation loss to 1.263 vs. 1.267, with efficiency 111.8%.

- Table 1: Expert upcycling at scale. A 20-layer interleaved MoE (∼7B total / ∼1B active nonembedding parameters) is pre-trained with 32 experts on 380B tokens, then continued for 50% (190B) or 100% (380B) of the pre-training budget. Fixed-32: 32-expert model continued without expansion. Upcycled 32→64 (Ours): expand to 64 experts via duplication, then CPT. Fixed-64: 64-expert model trained from scratch. Best value per row in bold. Accuracy higher↑; Val. Loss lower↓.

50% CPT 100% CPT Metric Fixed-32 Upcycled (Ours) Fixed-64 Fixed-32 Upcycled (Ours) Fixed-64 Val. Loss (↓) 1.339 1.305 1.308 1.301 1.263 1.267 MMLU 43.9 46.2 47.4 47.5 52.3 52.7 BBH CoT 19.4 28.0 33.8 33.4 43.8 45.0 GSM8K 28.1 36.0 40.1 39.3 48.3 49.1 IFEval 19.6 24.4 27.5 20.7 27.6 29.4 HellaSwag 62.4 65.0 63.9 65.1 67.3 66.1 ARC-Challenge 45.6 46.2 47.3 47.5 48.7 48.8 ARC-Easy 68.8 72.9 74.4 74.4 76.0 75.3 PIQA 74.2 75.9 75.5 76.6 77.4 77.5 OpenBookQA 36.0 39.4 38.4 37.8 38.8 39.8 SciQ 91.7 93.0 93.7 93.3 94.0 94.1 Social IQA 43.6 46.3 45.5 46.5 46.5 46.1 Avg (acc ↑) 48.5 52.1 53.4 52.9 56.4 56.7

Immediately after upcycling, the 64-expert model’s training loss on the CPT split is 1.38, close to the 32-expert source at 1.32 and far below the 10.5 of a randomly initialized 64-expert model; within the first 22B CPT tokens (∼6% of pre-training) the upcycled loss falls below Fixed-32 and subsequent CPT closes the gap to Fixed-64. Doubling the CPT budget from 50% to 100% raises average accuracy from 52.1 to 56.4, consistent with the capacity-gap penalty (Term I) shrinking as CPT length grows.

Generalization to full MoE. Expert upcycling also transfers beyond the interleaved architecture: on a full MoE with 256 experts and top-K=8 (∼3% activation ratio), ∥g∥2 utility-based upcycling achieves ≥93% gap-closure efficiency across 154M–1B total parameters (Appendix C.1, Table 7).

#### 5.2 Recipe ablations

Three ablations at the ∼1B scale test the two-term decomposition of §3.2 and yield the recipe used in the 7B→13B run. Training-budget allocation (§5.2.1) trades Term (I) against Term (II) via transition timing and CPT length. Operator design (§5.2.2) targets Term (II) along two axes: which experts to duplicate, and how to initialize the duplicates. The activation-ratio sweep (§5.2.3) stress-tests Term (I) by widening the source–target capacity gap, contrasting expert upcycling with sparse (dense→MoE) upcycling. All three share a 10-layer interleaved MoE (32→64 experts, ∼1B total / 144M active non-embedding; Table 5).

#### 5.2.1 Training budget allocation: pre-training, CPT, and transition timing

Term (I) predicts that gap closure is governed by how much training is spent in the expanded model relative to the smaller one. This motivates two practical questions: when training from scratch, at what point should the transition occur? And given an already pre-trained model, how much CPT is needed after upcycling? We test these with two experiments (Table 2), using the ∼1B-total/144M-active 10-layer interleaved MoE pre-trained for ∼50K steps.

Transition timing. Under a fixed total budget of 100K steps (Table 2a), upcycling early (τ/T ≤

- 0.25) achieves near-complete gap closure (94–100% efficiency). Very early transitions (τ/T = 0.05) underperform slightly, likely because the source model has seen too few tokens for experts to develop meaningful specialization, weakening the warm initialization that Term (II) relies on.

- Table 2: Training budget allocation for expert upcycling (10-layer interleaved MoE, 32→64 experts, top-K=2, pre-trained for τ=50K steps; upcycled model uses ∥g∥2 utility-based duplication). (a) When to upcycle when training from scratch (T=100K steps fixed, transition point τ varies). (b) How much CPT is needed after upcycling an already pre-trained model (τ fixed, T varies; τ/T is the fraction of total training spent in the smaller model).

(a) When to upcycle? (T=100K fixed, τ varies) (b) CPT budget sweep (τ=50K fixed, T varies) τ

steps τ/T

Fixed 32

Upcycled

Fixed 64

Eff.(%)

CPT steps τ/T

Fixed 32

Upcycled

Fixed 64

Eff.(%)

5K 0.05 2.804 2.753 2.741 81.0 5K 0.91 2.850 2.833 2.801 34.7 13K 0.13 2.805 2.754 2.754 100.0 13K 0.80 2.835 2.803 2.785 64.0 25K 0.25 2.806 2.754 2.754 100.0 25K 0.67 2.823 2.780 2.772 84.3 38K 0.38 2.806 2.757 2.757 100.0 38K 0.57 2.815 2.769 2.763 88.5 51K 0.51 2.809 2.759 2.758 98.0 51K 0.50 2.809 2.759 2.758 98.0

- Table 3: Comparison of duplication strategies on the 10-layer interleaved MoE (32→64 experts, top-K=2, ∼1B total / 144M active non-embedding parameters; upcycling happens at τ=50K steps for all rows). Rows correspond to different CPT budgets. Fixed-32 and Fixed-64 bracket the achievable range. Random initializes the new 32 experts with Kaiming initialization [20]. Uniform copies every expert once. The remaining four columns are utility-based strategies that preferentially duplicate high-importance experts: weight norm (∥w∥2), saliency (∥g∥ · ∥w∥), gradient norm (∥g∥2), and curvature-normalized (∥g∥2/H). All values are validation loss (↓).

Fixed Expert Upcycling

Non-utility Utility-based

CPT steps Fixed-32 Fixed-64 Random Uniform ∥w∥2 ∥g∥ · ∥w∥ ∥g∥2 ∥g∥2/H

13K 2.857 2.808 3.107 2.853 2.846 2.846 2.844 2.845 25K 2.835 2.785 3.010 2.809 2.807 2.809 2.804 2.805 38K 2.821 2.769 2.969 2.787 2.785 2.778 2.773 2.776 50K 2.809 2.758 2.963 2.769 2.771 2.766 2.759 2.768

CPT budget. Sweeping CPT from 10% to 100% of the pre-training budget (Table 2b), efficiency rises monotonically from 34.7% to 98.0%. At least 50% CPT is needed for strong gap closure, consistent with Term (I): duplicated experts require sufficient post-upcycling optimization to break symmetry and specialize. Together, these results confirm that CPT budget is the binding constraint: pre-training determines initialization quality, while CPT determines the extent of expert differentiation.

#### 5.2.2 Expert upcycling strategies

Which experts to duplicate? The 7B→13B result used gradient-norm guided duplication; here we study how much the choice of duplication strategy matters. Since experts in a trained MoE contribute unevenly to the objective, non-uniform duplication that concentrates capacity on high-importance experts may yield a better initialization than simply copying every expert once. We compare four utility-based expert duplication strategies from § 3.3 (weight norm uWN, gradient norm uGN, saliency uSAL, and curvature-normalized utility uCN) against two baselines: uniform duplication and random initialization [20], across CPT budgets, with experts ranked per layer and duplicated via greedy selection with replacement. Table 3 summarizes results.

Selective duplication consistently outperforms both baselines at every CPT budget. Random initialization performs far worse than even the Fixed-32 baseline (e.g., loss 3.107 vs. 2.857 at 25% CPT), confirming that warm initialization is essential. Among warm-start strategies, utility-based selection outperforms uniform duplication at every CPT budget, with the largest gains when CPT is limited, more than tripling gap closure at 25% CPT (26.5% vs. 8.2%, computed from Table 3). The advantage narrows at higher budgets as longer training partially compensates for initialization differences.

- Table 4: Effect of activation ratio and comparison with sparse upcycling (8-layer interleaved MoE, top-K=1, uniform duplication). Expert upcycling (Ours) starts from an MoE base with half the target expert count; sparse upcycling starts from a dense checkpoint. Both methods target the same mE-expert model (Fixed-mE). All values are validation loss (↓).

Target K/E

Ours (MoE→MoE)

Sparse Upc. (Dense→MoE)

Fixed-E Fixed-mE

25% 3.085 3.056 3.061 3.087 12.5% 3.056 3.018 3.025 3.086 6.25% 3.018 2.986 2.992 3.069 3.13% 2.894 2.788 2.808 3.049

Among the four utility strategies, gradient norm (∥g∥2) performs best overall; curvature-normalized (∥g∥2/H) and saliency (∥g∥ · ∥w∥) are close behind, and weight-norm-only (∥w∥2) lags slightly. In practice, ∥g∥2 is the recommended default.

Initialization diversity does not substitute for initialization quality. We also evaluated 20 diversity-inducing initialization heuristics (noise injection, drop upcycling [40], interpolation, orthogonalization, SVD-based perturbation, sparse-code mixing; 10 expert-level and 10 router-level variants); none exceed simple copy-paste duplication by more than 10−3 in validation loss. Across all n=65 utility + heuristic runs, the Spearman rank correlation between initialization and terminal loss is ρ=0.80 (val) / 0.86 (train): runs that start worse end worse.

#### 5.2.3 Effect of activation ratio and comparison with sparse upcycling

Term (I) penalizes the capacity gap L⋆E−L⋆mE between source and target. We study this on the 8-layer interleaved MoE with top-K=1 and uniform duplication, comparing expert upcycling (MoE→MoE)

against sparse upcycling [26] (dense→MoE) across target activation ratios from 25% down to 3.13% (Table 4). For each target, expert upcycling starts from an MoE base with half the target expert count, while sparse upcycling starts from a dense checkpoint.

Expert upcycling consistently produces losses close to the Fixed-mE ceiling across all activation ratios, though the residual gap grows at lower ratios (0.005 at 25% vs. 0.020 at 3.13%, computed from Table 4). Sparse upcycling, by contrast, fails to match even the Fixed-E baseline in every setting: the dense→MoE transition spans too large a capacity gap for CPT to close, confirming the Term (I) prediction. The gap between the two methods widens as the target activation ratio decreases, from

- 0.026 at 25% to 0.241 at 3.13%. Stabilized sparse-upcycling variants [40, 23] could narrow this gap and are future work.

### 6 Discussion

We introduced expert upcycling, which progressively expands MoE capacity by duplicating experts and extending the router mid-training while holding top-K routing fixed to preserve inference cost. A theoretical decomposition into a capacity gap and an initialization gain guided the operator design and ablations; in our 7B→13B experiments the upcycled model achieves lower validation loss than the fixed-size baseline while saving 24–32% of GPU hours (50–67% when the source model’s pre-training cost is sunk); at 100% CPT, the downstream accuracy gap closes to within 0.3 points (average across 11 benchmarks). The activation-ratio sweep (§5.2.3) further suggests iterated upcycling, repeatedly doubling expert count through successive steps to keep Term (I) small, as a natural extension.

Our results cover m=2 upcycling on MoE architectures up to 7B parameters, with DCLM for ablations and an English-majority mixture for the 7B run. Open directions include larger m, frontier-scale (>10B) models, and multilingual or distribution-shifted CPT.

### References

- [1] Samira Abnar, Harshay Shah, Dan Busbridge, Alaaeldin Mohamed Elnouby Ali, Josh Susskind, and Vimal Thilak. Parameters vs FLOPs: Scaling laws for optimal sparsity for mixture-ofexperts language models. arXiv preprint arXiv:2501.12370, 2025.
- [2] Naman Agarwal, Pranjal Awasthi, Satyen Kale, and Eric Zhao. Stacking as accelerated gradient descent, 2024. URL https://arxiv.org/abs/2403.04978.
- [3] Zhiqi Bu. Deep progressive training: scaling up depth capacity of zero/one-layer models, 2025. URL https://arxiv.org/abs/2511.04981.
- [4] Zhiqi Bu, Shiyun Xu, and Jialin Mao. Convex dominance in deep learning I: A scaling law of loss and learning rate. arXiv preprint arXiv:2602.07145, 2026. Accepted to ICLR 2026.
- [5] Tianqi Chen, Ian Goodfellow, and Jonathon Shlens. Net2net: Accelerating learning via knowledge transfer. In International Conference on Learning Representations (ICLR), 2016. URL https://arxiv.org/abs/1511.05641.
- [6] Zewen Chi, Li Dong, Shaohan Huang, Damai Dai, Shuming Ma, Barun Patra, Saksham Singhal, Payal Bajaj, Xia Song, Xian-Ling Mao, Heyan Huang, and Furu Wei. On the representation collapse of sparse mixture of experts. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [7] DeepSeek-AI. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [8] Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V. Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. GLaM: Efficient scaling of language models with mixture-of-experts. arXiv preprint arXiv:2112.06905, 2022.
- [9] Wenyu Du, Tongxu Luo, Zihan Qiu, Zeyu Huang, Yikang Shen, Reynold Cheng, Yike Guo, and Jie Fu. Stacking your transformers: A closer look at model growth for efficient LLM pre-training. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [10] Xianzhi Du, Tom Gunter, Xiang Kong, Mark Lee, Zirui Wang, Aonan Zhang, Nan Du, and Ruoming Pang. Revisiting MoE and dense speed-accuracy comparisons for LLM training. arXiv preprint arXiv:2405.15052, 2024.
- [11] Dongyang Fan, Bettina Messmer, and Martin Jaggi. Towards an empirical understanding of moe design choices. arXiv preprint arXiv:2402.13089, 2024.
- [12] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research (JMLR), 23(120):1–39, 2022. URL https://jmlr.org/papers/v23/21-0998.html.
- [13] GLM-4.5 Team. GLM-4.5: Agentic, reasoning, and coding (ARC) foundation models. arXiv preprint arXiv:2508.06471, 2025. URL https://arxiv.org/abs/2508.06471.
- [14] Nikolas Gritsch, Qizhen Zhang, Acyr Locatelli, Sara Hooker, and Ahmet Üstün. Nexus: Adaptive upcycling to efficiently pretrain mixture of experts. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 24364–24381. Association for Computational Linguistics, 2025.
- [15] Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats L. Richter, Quentin Anthony, Eugene Belilovsky, Irina Rish, and Timothée Lesort. Continual pre-training of large language models: How to (re)warm your model? arXiv preprint arXiv:2308.04014, 2023.
- [16] Song Han, Jeff Pool, John Tran, and William J. Dally. Learning both weights and connections for efficient neural networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 28, 2015.

- [17] Babak Hassibi and David G Stork. Second order derivatives for network pruning: Optimal brain surgeon. In Advances in Neural Information Processing Systems (NeurIPS), volume 5, pages 164–171, 1992.
- [18] Elad Hazan. Introduction to Online Convex Optimization. MIT Press, 2nd edition, 2016.
- [19] Ethan He, Abhinav Khattar, Ryan Prenger, Vijay Korthikanti, Zijie Yan, Tong Liu, Shiqing Fan, Ashwath Aithal, Mohammad Shoeybi, and Bryan Catanzaro. Upcycling large language models into mixture of experts. arXiv preprint arXiv:2410.07524, 2024.
- [20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015.
- [21] Yongqi Huang, Peng Ye, Chenyu Huang, Jianjian Cao, Lin Zhang, Baopu Li, Gang Yu, and Tao Chen. DeRS: Towards extremely efficient upcycled mixture-of-experts models. arXiv preprint arXiv:2503.01359, 2025. Accepted at CVPR 2025.
- [22] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.
- [23] Wangyi Jiang, Yaojie Lu, Hongyu Lin, Xianpei Han, and Le Sun. Improved sparse upcycling for instruction tuning. In Proceedings of the 31st International Conference on Computational Linguistics (COLING), pages 9485–9498, 2025.
- [24] Chao Jin, Ziheng Jiang, Zhihao Bai, Zheng Zhong, Juncai Liu, Xiang Li, Ningxin Zheng, Xi Wang, Cong Xie, Qi Huang, Wen Heng, Yiyuan Ma, Wenlei Bao, Size Zheng, Yanghua Peng, Haibin Lin, Xuanzhe Liu, Xin Jin, and Xin Liu. MegaScale-MoE: Large-scale communicationefficient training of mixture-of-experts models in production. arXiv preprint arXiv:2505.11432, 2025.
- [25] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526, 2017.
- [26] Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-of-experts from dense checkpoints. arXiv preprint arXiv:2212.05055, 2022.
- [27] Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pióro, Michał Krutul, Szymon Antoniak, Kamil Ciebiera, Krystian Król, Tomasz Odrzygóz´dz´, Piotr Sankowski, Marek Cygan, and Sebastian Jaszczur. Scaling laws for fine-grained mixture of experts. arXiv preprint arXiv:2402.07871, 2024.
- [28] Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. In Advances in Neural Information Processing Systems (NeurIPS), volume 2, pages 598–605, 1989.
- [29] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations (ICLR), 2021. URL https://arxiv.org/abs/2006.16668.
- [30] Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar,

- Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. Datacomp-lm: In search of the next generation of training sets for language models, 2024. URL https://arxiv.org/abs/2406.11794.
- [31] Yu Xin Li, Felix Dangel, Derek Tam, and Colin Raffel. Fishers for free? approximating the fisher information matrix by recycling the squared gradient accumulator. In Proceedings of the 42nd International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, pages 34252–34270. PMLR, 2025.
- [32] Zichong Li, Chen Liang, Zixuan Zhang, Ilgee Hong, Young Jin Kim, Weizhu Chen, and Tuo Zhao. Slimmoe: Structured compression of large moe models via expert slimming and distillation. arXiv preprint arXiv:2506.18349, 2025.
- [33] Seng Pei Liew, Takuya Kato, and Sho Takase. Scaling laws for upcycling mixture-of-experts language models. arXiv preprint arXiv:2502.03009, 2025.
- [34] Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. Not all experts are equal: Efficient expert pruning and skipping for mixture-ofexperts large language models. arXiv preprint arXiv:2402.14800, 2024.
- [35] Jan Ludziejewski, Maciej Píoro, Jakub Krajewski, Maciej Stefaniak, Michał Krutul, Jan Mała´snicki, Marek Cygan, Piotr Sankowski, Kamil Adamczewski, Piotr Miło´s, and Sebastian Jaszczur. Joint MoE scaling laws: Mixture of experts can be memory efficient. In Proceedings of the 42nd International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, pages 41056–41073. PMLR, 2025. doi: 10.48550/ARXIV.2502.

05172. URL https://proceedings.mlr.press/v267/ludziejewski25a.html. Also available as arXiv:2502.05172.

- [36] Meta AI. Llama 4: Natively multimodal foundation models. https://github.com/ meta-llama/llama-models, 2025. Model card available at https://github.com/ meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md.
- [37] Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, and Jan Kautz. Pruning convolutional neural networks for resource efficient inference. In International Conference on Learning Representations (ICLR), 2017.
- [38] Moonshot-AI. Kimi k2: Open agentic intelligence. Technical Report, 2025. URL https: //moonshotai.github.io/Kimi-K2/.
- [39] Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained language models. arXiv preprint arXiv:2305.16264, 2023.
- [40] Taishi Nakamura, Takuya Akiba, Kazuki Fujii, Yusuke Oda, Rio Yokota, and Jun Suzuki. Drop-upcycling: Training sparse mixture of experts with partial re-initialization. arXiv preprint arXiv:2502.19261, 2025.
- [41] Xiaonan Nie, Qibin Liu, Fangcheng Fu, Shenhan Zhu, Xupeng Miao, Xiaoyang Li, Yang Zhang, Shouda Liu, and Bin Cui. LSH-MoE: Communication-efficient MoE training via locality-sensitive hashing. In Advances in Neural Information Processing Systems, volume 37, 2024.
- [42] Yu Pan, Ye Yuan, Yichun Yin, Zenglin Xu, Lifeng Shang, Xin Jiang, and Qun Liu. Reusing pretrained models by multi-linear operators for efficient training. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
- [43] Abhishek Panigrahi, Nikunj Saunshi, Kaifeng Lyu, Sobhan Miryoosefi, Sashank Reddi, Satyen Kale, and Sanjiv Kumar. Efficient stagewise pretraining via progressive subnetworks. arXiv preprint arXiv:2402.05913, 2024.
- [44] Jupinder Parmar, Sanjev Satheesh, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. Reuse, don’t retrain: A recipe for continued pretraining of language models. arXiv preprint arXiv:2407.07263, 2024.

- [45] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-depths: Dynamically allocating compute in transformer-based language models. arXiv preprint arXiv:2404.02258, 2024.
- [46] Fabian Schaipp, Aaron Defazio, Harsh Mehta, Konstantin Mishchenko, and Ahmed Khaled. The surprising agreement between convex optimization theory and learning-rate scheduling for large model training. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025.
- [47] Shai Shalev-Shwartz. Online learning and online convex optimization. Foundations and Trends in Machine Learning, 4(2):107–194, 2012.
- [48] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
- [49] Alexander Soen and Ke Sun. Trade-offs of diagonal fisher information matrix estimators. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [50] Sainbayar Sukhbaatar, Olga Golovneva, Vasu Sharma, Hu Xu, Xi Victoria Lin, Baptiste Rozière, Jacob Kahn, Daniel Li, Wen-tau Yih, Jason Weston, and Xian Li. Branch-train-MiX: Mixing expert LLMs into a mixture-of-experts LLM. arXiv preprint arXiv:2403.07816, 2024.
- [51] Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [52] Changxin Tian, Kunlong Chen, Jia Liu, Ziqi Liu, Zhiqiang Zhang, and Jun Zhou. Towards greater leverage: Scaling laws for efficient mixture-of-experts language models. arXiv preprint arXiv:2507.17702, 2025.
- [53] Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. Auxiliary-loss-free load balancing strategy for mixture-of-experts. arXiv preprint arXiv:2408.15664, 2024.
- [54] Qi Wang, Hanyang Peng, and Yue Yu. Symphony-moe: Harmonizing disparate pre-trained models into a coherent mixture-of-experts. Proceedings of the AAAI Conference on Artificial Intelligence, 2026. arXiv preprint arXiv:2509.18542.
- [55] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022.
- [56] Haoyuan Wu, Haoxing Chen, Xiaodong Chen, Zhanchao Zhou, Tieyuan Chen, Yihong Zhuang, Guoshan Lu, Zenan Huang, Junbo Zhao, Lin Liu, Zhenzhong Lan, Bei Yu, and Jianguo Li. Grove MoE: Towards efficient and superior MoE LLMs with adjugate experts. arXiv preprint arXiv:2508.07785, 2025.
- [57] Qifan Yu, Xinyu Ma, Zhijian Zhuo, Minrui Wang, Deyi Liu, Shiyi Zhan, Yiyuan Ma, Liang Xiang, Xingyan Bin, and Di He. SPARKLING: Balancing signal preservation and symmetry breaking for width-progressive learning. arXiv preprint arXiv:2602.02472, 2026.
- [58] Ted Zadouri, Ahmet Üstün, Arash Ahmadian, Beyza Ermi¸s, Acyr Locatelli, and Sara Hooker. Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning. arXiv preprint arXiv:2309.05444, 2023.
- [59] Qizhen Zhang, Nikolas Gritsch, Dwaraknath Gnaneshwar, Simon Guo, David Cairuz, Bharat Venkitesh, Jakob Foerster, Phil Blunsom, Sebastian Ruder, Ahmet Ustun, and Acyr Locatelli. BAM! just like that: Simple and efficient parameter upcycling for mixture of experts. arXiv preprint arXiv:2408.08274, 2024.
- [60] Qizhen Zhang, Prajjwal Bhargava, Chloe Bi, Chris X. Cai, Jakob Nicolaus Foerster, Jeremy Fu, Punit Singh Koura, Ruan Silva, Sheng Shen, Emily Dinan, Suchin Gururangan, and Mike Lewis. BTS: Harmonizing specialized experts into a generalist LLM. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6816–6834, 2025. arXiv:2502.00075.

- [61] Xue Zhang, Yunlong Chen, Tong Liu, Cong Wang, Mo Liu, Hongji Huang, and Yihan Li. Less, but better: Efficient multilingual expansion for LLMs via layer-wise mixture-of-experts. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025. arXiv:2505.22582.
- [62] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th International Conference on Machine Learning (ICML), pages 928–936, 2003.

### Appendix

#### Contents

Regret-bound decomposition for expert upcycling A Theoretical Justification for Gradient-Based Utility Scores B Model Configurations C Heuristic Upcycling Methods and Results D Extended Related Work E

### A Regret-bound decomposition for expert upcycling

Using the online convex optimization (OCO) framework [62, 47, 18], we adapt the regret-telescoping approach of Bu [3] to formalize the two-term decomposition introduced in §3.2.

#### A.1 Setup

Notation. Let z ∼ D denote tokens from the pretraining distribution and ℓ(·) the token-level crossentropy loss. For an MoE Transformer with top-K routing, define the population objective for a model with n experts as Ln(θ) := Ez∼D[ℓ(fn(z;θ))], where θ collects all parameters and Θn denotes the parameter space. Write LE for the E-expert objective, LmE for the expanded mE-expert objective, and L⋆n = minθ∈Θ

n Ln(θ). Partition the mE-expert parameter vector as θ = (θs,θ+), where θs denotes parameters shared with the E-expert model (dense layers, embeddings, original experts) and θ+ denotes the degrees of freedom introduced by expansion (replica experts, expanded router weights). Let θmE⋆ = (θs⋆,θ+⋆ ) ∈ arg min LmE, and let θ+U and θ+0 denote the new-parameter initialization at step τ produced by the upcycling operator (Definition 3.1) and the random initialization used by the fixed-size procedure, respectively. We compare two procedures over T total gradient steps with learning-rate schedule {ηt}Tt=0−1: (i) expert upcycling: train an E-expert model for τ steps, expand to mE experts, continue for T − τ steps; (ii) fixed-size: train an mE-expert model for all T steps from random initialization.

#### Assumptions.

- Assumption A.1 (Convexity). LmE(θ) is convex in θ.
- Assumption A.2 (Bounded gradients). ∥∇Ln(θ)∥2 ≤ G for all θ and n ∈ {E,mE}.

These are standard in the OCO literature [62, 18]. They do not hold literally for deep networks; we adopt them to derive structural insights rather than tight numerical bounds. Recent work shows that convex optimization theory is surprisingly predictive of large-scale training dynamics despite non-convexity [46, 4].

#### Canonical lifting.

Definition A.1 (Canonical lifting). The canonical lifting ι : ΘE → ΘmE retains the original E experts and router unchanged, sets the extra (m−1)E expert weights to zero, and sets their router logits to a sufficiently large negative constant −M (chosen so the extra experts are never selected by top-K; because top-K is a discrete hard threshold, any finite M strictly larger than the maximum attained logit gap suffices). Since the extra experts are never selected by top-K:

- (a) LmE(ι(θE)) = LE(θE) for all θE (loss preservation),
- (b) ∇θ+LmE(ι(θE)) = 0 (zero gradient on new parameters).

Property (b) holds because zero-weight experts with zero gating contribute nothing to the forward pass. The lifting ι is a proof device: it lets us represent Phase 1 iterates in ΘmE for the telescoping argument without changing the optimization dynamics.

#### A.2 Theorem and proof

Theorem A.1 (Expert upcycling bound). Let L¯up and L¯fs denote the learning-rate-weighted average training losses of the two procedures over the shared schedule {ηt}Tt=0−1, and let Rup and Rfs denote their respective OCO regret upper bounds (defined in Step 3 of the derivation below):

L¯up ≤ Rup, L¯fs ≤ Rfs.

Suppose both procedures share the same initial θs and that the shared components of their respective optima coincide. Then under Assumptions A.1–A.2, these regret bounds differ by:

τ−1 t=0 ηt T−1 t=0 ηt

+ ∥θ+U − θ+⋆ ∥2 − ∥θ+0 − θ+⋆ ∥2 2 T−1

Rup − Rfs =

L⋆E − L⋆mE

. (4)

t=0 ηt

(I) capacity gap

(II) initialization gain

The proof proceeds in four steps.

- Step 1: one-step descent lemma. Lemma A.1 (Gradient-descent regret inequality). Let L satisfy Assumptions A.1–A.2. For any comparator θ⋆ and iterate θt+1 = θt − ηt∇L(θt):

ηt L(θt) − L(θ⋆) ≤ 12 ∥θt − θ⋆∥2 − ∥θt+1 − θ⋆∥2 + 12ηt2G2. (5) Proof. Expand the squared distance after the update:

∥θt+1 − θ⋆∥2 = ∥θt − θ⋆∥2 − 2ηt⟨∇L(θt), θt − θ⋆⟩ + ηt2∥∇L(θt)∥2. (6)

By convexity (Assumption A.1): ⟨∇L(θt), θt − θ⋆⟩ ≥ L(θt) − L(θ⋆). By the gradient bound (Assumption A.2): ∥∇L(θt)∥2 ≤ G2. Substituting into (6) and rearranging yields (5). This is the standard OCO regret inequality for online gradient descent [62, Theorem 1].

| |
|---|

- Step 2: phase-wise telescoping. Phase 1 (t = 0,...,τ − 1): training in ΘE. We represent the upcycling iterates in ΘmE via the lifting ι (Definition A.1). Write θ˜t = ι(θtup) for t ≤ τ. By

property (a) of ι, LmE(θ˜t) = LE(θtup). By property (b), the new-parameter coordinates remain at zero throughout Phase 1, so the lifted iterates follow the same trajectory as the E-expert SGD. Choose

comparator ι(θE⋆ ), where θE⋆ ∈ arg min LE. Applying Lemma A.1 at each step t = 0,...,τ − 1 and summing (intermediate distance terms telescope, leaving only boundary terms):

τ−1

t=0

ηt LE(θtup) − L⋆E ≤ 12∥θ˜0 − ι(θE⋆ )∥2 − 21∥θ˜τ − ι(θE⋆ )∥2 + G

2

2

τ−1

t=0

ηt2. (7)

Phase 2 (t = τ,...,T − 1): training in ΘmE. At step τ, the upcycling operator is applied to the Phase-1 terminal iterate to obtain θτup ∈ ΘmE. From this point, iterates live in ΘmE directly. Choose comparator θmE⋆ ∈ arg min LmE. Applying Lemma A.1 at each step t = τ,...,T −1 and summing:

T−1

t=τ

ηt LmE(θtup) − L⋆mE ≤ 21∥θτup − θmE⋆ ∥2 − 12∥θTup − θmE⋆ ∥2 + G

2

2

T−1

t=τ

ηt2. (8)

- Step 3: combining phases. Adding (7) and (8), and writing Lupt = LE(θtup) for t < τ and Lupt = LmE(θtup) for t ≥ τ, we obtain a bound on the η-weighted sum of losses. Of the two terminal distance terms on the right side, the Phase-T term −12∥θTup − θmE⋆ ∥2 ≤ 0 is dropped (yielding a

looser upper bound); the Phase-1 terminal −12∥θ˜τ − ι(θE⋆ )∥2 is retained, since its shared-parameter component will cancel a matching term at the transition in Step 4. This gives

T−1

ηt Lupt ≤

t=0

τ−1

ηt L⋆E +

t=0

T−1

T−1

2

ηt2

ηt L⋆mE + G

2

t=τ

t=0

+ 12 ∥θ˜0 − ι(θE⋆ )∥2 − ∥θ˜τ − ι(θE⋆ )∥2 + ∥θτup − θmE⋆ ∥2 . (9)

Dividing by Tt=0−1 ηt gives L¯up ≤ Rup, where L¯up := ( t ηtLupt )/ t ηt is the learning-rateweighted average loss and the upcycling regret upper bound is defined as:

τ−1 t=0 ηt L⋆E + T−1

G2 Tt=0−1 ηt2 2 T−1 t=0 ηt

t=τ ηt L⋆mE

Rup :=

+

T−1 t=0 ηt

+ ∥θ˜0 − ι(θE⋆ )∥2 − ∥θ˜τ − ι(θE⋆ )∥2 + ∥θτup − θmE⋆ ∥2 2 T−1

. (10)

t=0 ηt

For the fixed-size procedure, which trains in ΘmE for all T steps with comparator θmE⋆ , applying Lemma A.1, dropping the non-positive terminal −21∥θTfs − θmE⋆ ∥2, and dividing by Tt=0−1 ηt analogously gives L¯fs ≤ Rfs, where the fixed-size regret upper bound is defined as:

G2 Tt=0−1 ηt2 2 T−1 t=0 ηt

Rfs := L⋆mE + ∥θ0fs − θmE⋆ ∥2 2 T−1 t=0 ηt

+

. (11)

- Step 4: difference of the regret upper bounds. We now compute Rup − Rfs as an algebraic identity between the two explicit expressions (10) and (11). This is a direct computation on the bounds themselves, not a bound on the loss gap L¯up − L¯fs. Subtracting (11) from (10) gives three groups of terms: a comparator-loss difference, a distance-term difference, and a G2 contribution,

τ−1 t=0 ηt L⋆E + T−1

t=τ ηt L⋆mE

Rup − Rfs =

− L⋆mE

T−1 t=0 ηt

comparator-loss difference

+ ∥θ˜0 − ι(θE⋆ )∥2 − ∥θ˜τ − ι(θE⋆ )∥2 + ∥θτup − θmE⋆ ∥2 − ∥θ0fs − θmE⋆ ∥2 2 T−1

t=0 ηt

distance-term difference

G2 Tt=0−1 ηt2 2 T−1 t=0 ηt

G2 Tt=0−1 ηt2 2 T−1 t=0 ηt

+

. (12)

−

= 0

The G2 terms are identical in both bounds (same schedule, same G) and cancel exactly. For the comparator-loss difference, writing Tt=−τ1 ηt = T−1

t=0 ηt − τt=0−1 ηt and simplifying:

τ−1 t=0 ηt L⋆E + T−1

τ−1 t=0 ηt T−1 t=0 ηt

t=τ ηt L⋆mE

− L⋆mE =

L⋆E − L⋆mE . (13)

T−1 t=0 ηt

For the distance-term difference, decompose θ = (θs,θ+) so that ∥θ − θ⋆∥2 = ∥θs − θs⋆∥2 + ∥θ+ − θ+⋆ ∥2. Both procedures use the same initial shared parameters (θ˜0,s = θ0fs,s), and assuming the shared components of θE⋆ and θmE⋆ coincide (the same simplification used in progressive depth expansion [3]), ι(θE⋆ )s = θmE,s⋆ . Under the lifting, both θ˜0 and ι(θE⋆ ) have zero new-parameter coordinates, so

∥θ˜0 − ι(θE⋆ )∥2 = ∥θ0fs,s − θmE,s⋆ ∥2, ∥θ0fs − θmE⋆ ∥2 = ∥θ0fs,s − θmE,s⋆ ∥2 + ∥θ+0 − θ+⋆ ∥2,

whose difference leaves −∥θ+0 − θ+⋆ ∥2. At the transition, the upcycling iterate satisfies ∥θτup − θmE⋆ ∥2 = ∥θτ,sup − θmE,s⋆ ∥2 + ∥θ+U − θ+⋆ ∥2, and under the lifting ∥θ˜τ − ι(θE⋆ )∥2 = ∥θτ,sup − θmE,s⋆ ∥2 (new coordinates still zero). The retained Phase-1 terminal −∥θ˜τ − ι(θE⋆ )∥2 exactly cancels the shared-parameter piece of +∥θτup −θmE⋆ ∥2, leaving +∥θ+U −θ+⋆ ∥2. Thus the distance-term difference reduces to

∥θ+U − θ+⋆ ∥2 − ∥θ+0 − θ+⋆ ∥2 2 T−1 t=0 ηt

. (14)

Substituting (13) and (14) into (12) yields Rup − Rfs = (I) + (II), which is Eq. (4) and establishes Theorem A.1.

| |
|---|

#### A.3 Discussion of assumptions

Convexity. The convex assumption does not hold for deep networks. However, a growing body of evidence shows that convex optimization theory provides qualitatively accurate predictions for large-scale training: Schaipp et al. [46] demonstrate that convex SGD bounds closely predict optimal learning-rate schedules for LLM pre-training; Bu et al. [4] show that optimization dynamics are “empirically convex-like” across diverse tasks and models. We adopt the convex framework for structural insight, not tight numerical bounds.

Parameter decomposition. The assumption that the shared components of θE⋆ and θmE⋆ coincide is a simplification: adding experts may shift the optimal dense-layer parameters. This is the same

assumption used in progressive depth expansion [3] and enables a clean decomposition. For expertcount expansion with m = 2, a natural choice is to designate the first copy of each expert as “shared” and the second as “new.”

### B Theoretical justification for gradient-based utility scores

We derive the two utility scores used in Section 3.3 from first principles, starting from a local approximation of the loss at transition time τ.

Setup. Let θEτ ∈ ΘE denote the trained E-expert checkpoint at transition time τ. Partition the parameters as θ = (w1,...,wE,ϕ), where we ∈ Rd

e are the parameters of expert e and ϕ collects all remaining parameters (dense layers, embeddings, router). Write L(θ) = Ez∼D[ℓ(f(z;θ))] for the population loss. Let ge = ∇weL(θEτ ) denote the gradient with respect to expert e’s parameters, evaluated at transition time τ.

First-order loss expansion. Consider perturbing expert e’s parameters by ∆we while holding all other parameters fixed. A first-order Taylor expansion around θEτ gives:

L(θEτ + ∆we1e) = L(θEτ ) + ge⊤∆we + O(∥∆we∥22), (15) where 1e denotes the indicator that only expert e’s block is perturbed.

Worst-case sensitivity and Utility 1: uG(e) = ∥ge∥22. By the Cauchy–Schwarz inequality, |ge⊤∆we| ≤ ∥ge∥2 · ∥∆we∥2, with equality when ∆we ∝ ge. For a unit perturbation, the worst-case first-order loss change is ∥ge∥2. Ranking experts by ∥ge∥2 thus ranks them by how much the loss can change per unit perturbation. We use the square for convenience:

uG(e) := ∥ge∥22. (16)

An expert with large uG(e) is one where the loss landscape is steep: the objective is actively sensitive to this expert’s parameters under the current data distribution and routing at time τ. Replicating such an expert introduces new degrees of freedom precisely where the loss is most responsive, giving CPT the greatest opportunity to reduce the initialization gap term in Eq. (4).

Scale-aware sensitivity and Utility 2: ∥we∥2 · ∥ge∥2. The gradient norm ∥ge∥2 is not scaleinvariant: if expert e’s parameters are uniformly rescaled by α > 0, the gradient scales as ge → ge/α, so ∥ge∥2 decreases even though the expert’s functional contribution is unchanged. This can cause uG to systematically underrank large-norm experts that are functionally important but whose gradients have been reduced by scale.

To correct for this, consider perturbing expert e’s parameters proportionally to their current magnitude, ∆we = ϵ · we:

L(θEτ + ϵwe1e) − L(θEτ ) ≈ ϵ · ge⊤we.

The absolute value is bounded by |ge⊤we| ≤ ∥ge∥2 · ∥we∥2, tight when ge ∝ we. The product ∥we∥2 · ∥ge∥2 captures worst-case loss sensitivity under proportional perturbations. We define:

uSAL(e) := ∥we∥2 · ∥ge∥2. (17) This is the weight-space analogue of the Taylor saliency criterion of Molchanov et al. [37], who derive ΘTE(hi) = |∂h∂C

hi| in activation space. The weight-space version |gijwij| has been used as a structured pruning criterion in recent work [32]. Our criterion operates at the expert block level rather than individual scalars.

i

In practice, uG and uSAL perform similarly to each other and both significantly outperform uniform copy-paste, suggesting that any gradient-based importance signal is more informative than treating all experts as equally valuable replication targets.

Why not second-order? The second-order term in Eq. (15) involves the Hessian block He = ∇2weL(θEτ ). Curvature-normalized scores such as ge⊤He−1ge [17] are theoretically more precise but require estimating He, which is expensive and noisy in practice. Diagonal Fisher approximations introduce significant bias [49], and in our experiments curvature-normalized variants did not outperform the first-order scores. We therefore use uG and uSAL as our primary utilities.

### C Model configurations

This appendix provides the complete model configurations used across all experiments (see § 4 for the experimental setup). Table 5 details the interleaved MoE architecture configuration, with corresponding parameter counts and compute statistics in Tables 5a–5b. Tables 6–6c provide the full (non-interleaved) MoE architecture (Table 6a), parameter counts (Table 6b), and training configurations.

- Table 5: Interleaved MoE model configurations and statistics. All models share: Context Length = 8192, Vocab Size = 200704, Grouped Query Attention (GQA), loss-free load balancing, 32 routed experts, and 0 shared experts.

- (a) Architecture configuration. Layers are interleaved dense and MoE; Act. FFN = top-K × FFN/Expert; Total FFN = 32 × FFN/Expert.

Model Layers Hidden FFN Attn Attn MoE FFN/ top-K Act. FFN Total FFN Dim Dim Heads Groups Layers Expert (MoE) (MoE)

20-layer 20-layer (baseline) 20 2048 7168 16 16 10 3072 2 6144 98304

10-layer 10-layer (top2) 10 1024 3584 8 8 5 1536 2 3072 49152

8-layer 8-layer (top1) 8 768 2688 6 6 4 1152 1 1152 36864

- (b) Parameter counts (in millions) and compute. Train Tokens (B) are the actual pre-training token budgets used in experiments.

Model Router Dense MoE Act. MoE Total Embed Act. Params Total Params Train Total Train (M) Layer (M) Layer (M) Layer (M) (M) w/ Emb (M) w/ Emb (M) Tokens (B) FLOPs

20-layer 20-layer (baseline) 0.07 60.8 54.6 620.8 822.1 1976.3 7638.6 383.5 3.43 × 1021

10-layer 10-layer (top2) 0.03 15.2 13.7 155.2 411.0 555.4 1263.2 92.6 1.27 × 1020

8-layer 8-layer (top1) 0.02 8.6 5.0 87.3 308.3 362.6 691.8 59.5 3.36 × 1019

#### C.1 Full MoE generalization results

To verify that expert upcycling transfers beyond the interleaved architecture, we evaluate on a full MoE with 256 experts and top-K=8 (∼3% activation ratio), consistent with frontier MoE models [7, 13, 38]. At the ∼1B total parameter scale with ∥g∥2 utility-based upcycling, the full MoE achieves strong gap closure across all model sizes tested (Table 7). Both interleaved and full MoE architectures show strong gap closure, confirming that expert upcycling is effective across MoE families and activation ratios.

- Table 6: Full (non-interleaved) MoE model configurations and statistics. All models share: Context Length = 8192, Vocab Size = 200704, Grouped Query Attention (GQA), loss-free load balancing, 256 routed experts with Top-8 routing, and 0 shared experts.

- (a) Architecture configuration. Each configuration has two dense layers (prefix and suffix); the remaining layers are MoE. Exp FFN = FFN dimension per expert.

General Config GQA MoE Config Layers Hidden FFN Dim Vocab Heads Groups MoE L Exp FFN Experts Act Exp

4 256 896 200K 2 2 2 384 256 8 6 256 896 200K 2 2 4 384 256 8 8 384 1,344 200K 3 3 6 576 256 8

10 512 1,792 200K 4 4 8 768 256 8

- (b) Parameter counts and compute. MoE Act/Total = activated/total parameters per MoE layer. FLOPs/Token counts non-embedding forward-pass FLOPs.

Layer Parameters Total Parameters Compute Router Dense Layer MoE Act MoE Total Activated Total FLOPs/Token

65.5K 0.95M 2.69M 75.8M 110M 256M 9.50e7 65.5K 0.95M 2.69M 75.8M 115M 408M 1.54e8 98.3K 2.14M 6.00M 171M 194M 1.18B 3.97e8 131K 3.80M 10.6M 303M 298M 2.64B 8.15e8

- (c) Training configuration. Token budgets and learning rates determined by scaling laws.

Tokens Batch Size Steps LR Total FLOPs

2.20B 16 16,790 1.69e-2 2.09e17

- 2.31B 16 17,611 1.16e-2 3.54e17
- 3.89B 16 29,663 5.95e-3 1.54e18 5.96B 32 22,741 3.75e-3 4.86e18

- Table 7: Full MoE upcycling results (256→512 experts, top-K=8, ∥g∥2 utility-based duplication) across model sizes from 154M to 1B total parameters. All values are validation loss (↓).

Active (M) Total (M) Fixed-256 Upcycled 512 Fixed-512 Eff. (%)

7 154 3.564 3.519 3.516 93.8 13 305 3.153 3.071 3.067 95.3 40 1028 2.819 2.767 2.763 92.9

- Table 8: Heuristic upcycling: best variant per method category (10-layer, 32→64 experts). No heuristic meaningfully outperforms copy-paste (≤ 10−3 loss); several degrade performance. Full results in Appendix Tables 11 and 12.

Category Best Variant Val Loss Expert Initialization (baseline = 2.76858)

Copy (baseline) – 2.76858 Scaled Copy s=0.90 2.76815 Copy + Noise λ=0.01 2.76859 Interpolate α=0.2 2.76888 Sparse Code Mix dict=1024 2.76938 SVD Perturb values only 2.76938 Drop Upcycle drop=0.3 (Xavier) 2.77043 SVD Mix ratio=0.3 2.77472 Orthogonal standard 2.77487

Router Initialization (baseline = 2.76858)

Copy (baseline) – 2.76858 Interpolate heavy 2.76776 SVD Perturb moderate 2.76816 Copy + Noise very light 2.76819 Bias Disc Dup all layers 2.76827 Temp. Scaled sharp 2.76847 Adversarial light 2.76848 Bias Enc Dup all layers 2.77831

### D Heuristic upcycling methods and results

This appendix provides the full description and experimental evaluation of heuristic expert and router upcycling methods referenced in § 5.2.2. We evaluated 10 expert-level and 10 router-level initialization heuristics designed to seed diversity among duplicates while retaining inherited capability. As reported in the main text, none of these heuristics meaningfully outperform simple copy-paste duplication.

#### D.1 Summary of results

- Table 8 summarizes the best variant per method category on the 10-layer, 32→64 expert interleaved MoE model. Across all expert and router heuristics, validation losses lie in a narrow band around the copy-paste baseline, with improvements of at most ∼10−3. Several more aggressive methods (SVD mixing, orthogonalization) slightly degrade performance. These results indicate that maintaining a low initial loss at the upcycling boundary is more important than introducing artificial diversity: perturbations can disrupt the pre-upcycling solution and force CPT to allocate capacity to recovery rather than specialization. In contrast, simple duplication provides a warm initialization, and loss-free load balancing ensures that all experts receive gradient signal, allowing training dynamics to drive expert differentiation naturally.

D.2 Expert upcycling heuristics

- Table 9 summarizes the 10 expert-level initialization heuristics evaluated. All methods upcycle from 32→64 experts on the 10-layer interleaved MoE.

D.3 Router upcycling heuristics

- Table 10 summarizes the 10 router-level initialization heuristics evaluated on the same 10-layer model.

- Table 9: Expert upcycling heuristic descriptions and key hyperparameters. Method Description Key Hyperparameters

Copy (baseline) Duplicate each expert exactly, producing identical twins. – Copy + Noise Add calibrated Gaussian noise to duplicated expert

weights.

λ ∈ {0.01, 0.02, 0.05} Drop Upcycle [40] Re-initialize a fraction of weight-matrix columns while

keeping the remainder from the original expert.

drop ∈ {0.3, 0.5, 0.7}; init: Xavier, Kaiming, Normal

Shuffle Columns Randomly permute columns of weight matrices, preserving marginal statistics but changing connectivity.

– Interpolate Create new experts by interpolating between adjacent experts: winew ← αwi + (1−α)wi+1.

α ∈ {0.2, 0.5, 0.7} Orthogonal Gram–Schmidt orthogonalization to make duplicates or-

thogonal (in parameter space) to originals.

ϵ = 10−6 Scaled Copy Scale duplicated weights by a constant factor, changing

magnitude while preserving direction.

s ∈ {0.90, 0.95, 1.05} SVD Perturb Compute W = USV ⊤ and perturb selected components

(singular values, vectors, and/or drop small components).

σv ∈ [0.05, 0.2], σu ∈ [0.02, 0.1], drop∈[0, 0.3]

SVD Mix Compute SVDs for multiple experts and create hybrids by mixing singular vectors.

mix ratio ∈ {0.2, 0.3, 0.5, 0.7}

Sparse Code Mix Approximate W ≈DC via ISTA-style sparse coding, then mix sparse codes between experts.

dict ∈ {256, 512, 1024}; sparsity ∈ {0.05, 0.2}

- Table 10: Router upcycling heuristic descriptions. Method Description

Copy (baseline) Duplicate router weights exactly. Copy + Noise Add Gaussian noise to duplicated router weights to seed routing diversity.

Interpolate Interpolate router weights with neighbors: ridup ← αri + (1−α)ri+1. Bias Only Keep router weights identical but modify only biases to shift routing preferences.

Scaled Copy Scale duplicate router weights to adjust routing sharpness/entropy. Perturb New Only Freeze original router weights and perturb only duplicates. Orthogonal Construct duplicate router weights orthogonal to originals. Adversarial Push duplicate router weights in the opposite direction of originals. Temperature Scaled Apply temperature scaling to duplicate router logits (pre-softmax) to control entropy. SVD Perturb SVD-based perturbations to router weights to preserve coarse routing structure while

varying details.

- D.4 Expert heuristic results

Table 11 reports validation loss for all expert upcycling heuristics on the 10-layer, 32-expert interleaved MoE model.

- D.5 Router heuristic results

- Table 12 reports validation loss for all router upcycling heuristics on the same 10-layer model.

### E Extended related work

This appendix provides detailed comparisons between expert upcycling and each line of related work, organized by category. For each cited paper, we explain the relationship to our contributions and highlight key differences.

#### E.1 MoE foundations and scaling laws

Sparsely-Gated MoE [48]. Introduced the sparsely-gated MoE layer with top-K routing and load-balancing losses. Our work builds directly on this architecture: expert upcycling preserves the top-K routing mechanism while expanding the expert pool, keeping per-token FLOPs constant.

- Table 11: Expert heuristic upcycling results (interleaved MoE, 10-layer, 32 experts). All methods upcycle from 32→64 experts and train with identical CPT budget. Baseline copy-paste = 2.76858. Method descriptions and hyperparameters in Table 9.

Method Qualifier / Variant Key Param Val Loss Copy-based

Copy (baseline) – – 2.76858 Copy + Noise conservative λ=0.01 2.76859 Copy + Noise moderate λ=0.02 2.76860 Copy + Noise aggressive λ=0.05 2.76895 Scaled Copy slight reduction s=0.95 2.76846 Scaled Copy moderate reduction s=0.90 2.76815 Scaled Copy slight amplification s=1.05 2.76896

Drop upcycling

Drop Upcycle conservative (Xavier) drop=0.3 2.77043 Drop Upcycle moderate (Xavier) drop=0.5 2.77125 Drop Upcycle aggressive (Xavier) drop=0.7 2.77304 Drop Upcycle moderate (Kaiming) drop=0.5 2.77337 Drop Upcycle moderate (Normal) drop=0.5 2.77203

Interpolation

Interpolate slight α=0.2 2.76888 Interpolate balanced α=0.5 2.76953 Interpolate heavy α=0.7 2.76966

Orthogonal Orthogonal standard ϵ=1e-6 2.77487 SVD-based

SVD Perturb conservative σv=0.05, σu=0.02 2.77476 SVD Perturb moderate σv=0.1, σu=0.05 2.77483 SVD Perturb aggressive σv=0.2, σu=0.1 2.77472 SVD Perturb drop heavy drop=0.3 2.77441 SVD Perturb values only σv=0.15 2.76938 SVD Perturb vectors only σu=0.1 2.77468 SVD Mix light ratio=0.2 2.77483 SVD Mix moderate ratio=0.3 2.77472 SVD Mix heavy ratio=0.5 2.77472 SVD Mix aggressive ratio=0.7 2.77475

Sparse Code Mix

Sparse Code Mix small dict dict=256 2.77034 Sparse Code Mix standard dict=512 2.76955 Sparse Code Mix large dict dict=1024 2.76938 Sparse Code Mix high sparsity sparsity=0.2 2.76981 Sparse Code Mix low sparsity sparsity=0.05 2.76951 Sparse Code Mix heavy mixing mix=0.5 2.76964 Sparse Code Mix more iterations n_iter=200 2.76945

- Table 12: Router heuristic upcycling results (interleaved MoE, 10-layer, 32 experts). All methods upcycle routers from 32→64 expert slots with identical CPT budget. Baseline router copy = 2.76858. Methods are described in Appendix D.

Method Variant Val Loss Copy-based

Copy (baseline) – 2.76858 Copy + Noise very light 2.76819 Copy + Noise light 2.76844 Copy + Noise moderate 2.76843 Copy + Noise aggressive 2.76856

Interpolation

Interpolate light 2.76861 Interpolate balanced 2.76877 Interpolate heavy 2.76776

Bias-based

Bias Only all layers 2.76836 Bias + Noise Only – 2.76845 Bias Enc Dup – 2.77827 Bias Enc Dup all layers 2.77831 Bias Disc Dup – 2.76827 Bias Disc Dup all layers 2.76827

Scaling & Temperature

Scaled Copy very soft 2.76874 Scaled Copy soft 2.76878 Scaled Copy sharp 2.76860 Temperature Scaled soft 2.76877 Temperature Scaled very soft 2.76883 Temperature Scaled sharp 2.76847

Perturbation

Perturb New Only light 2.76843 Perturb New Only moderate 2.76860 Perturb New Only aggressive 2.76858 Orthogonal – 2.76867 Adversarial light 2.76848 Adversarial strong 2.76990

SVD-based

SVD Perturb conservative 2.76852 SVD Perturb moderate 2.76816 SVD Perturb aggressive 2.76826

GLaM [8]. Demonstrated MoE scaling to 1.2T parameters with favorable quality-per-FLOP tradeoffs. GLaM trains from scratch at the target scale; expert upcycling achieves similar capacity expansion goals but avoids the full from-scratch cost by growing an existing smaller MoE checkpoint.

Switch Transformers [12] and GShard [29]. Simplified MoE routing (top-1) and scaled expert parallelism across thousands of devices. These works focus on training infrastructure and routing simplification for from-scratch training. Our method is complementary: it provides an alternative path to large expert counts by growing mid-training rather than starting large.

Joint MoE Scaling Laws [35]. Derived scaling laws jointly over active parameters, total parameters, and training tokens for MoE models, showing that MoE can be memory-efficient. These scaling laws directly motivate expert upcycling: they predict that increasing expert count at fixed active compute improves quality, and our method operationalizes this prediction without restarting training.

Fine-Grained MoE Scaling Laws [27]. Extended scaling analysis to fine-grained (many small) experts. Our experiments span activation ratios from 3% to 50%, covering both coarse and finegrained regimes, and we show that upcycling efficiency is robust across this range.

Greater Leverage MoE Scaling Laws [52]. Identified sparsity as the most effective lever for improving MoE performance among MoE hyperparameters. This directly supports our approach: expert upcycling decreases the activation ratio (active-to-total parameter ratio) by adding experts while holding top-K fixed.

Optimal Sparsity for MoE [1]. Derived compute-optimal sparsity schedules showing that the optimal number of experts depends on the compute budget. Our work complements this by providing a mechanism to change the activation ratio mid-training as the compute budget evolves, rather than committing to a fixed activation ratio from the start.

Scaling Data-Constrained LMs [39]. Showed that repeated data yields diminishing returns beyond ∼4 epochs, proposing scaling laws for data-constrained regimes. This is relevant to expert upcycling because our continued pre-training phase uses additional tokens; their findings inform how much additional data is needed for effective upcycling versus when returns diminish.

#### E.2 Growing network size during training

Net2Net [5]. Introduced function-preserving transforms (Net2WiderNet, Net2DeeperNet) for accelerating training via knowledge transfer. Expert upcycling is inspired by the function-preserving philosophy of Net2Net: our duplication + router expansion produces a warm initialization whose initial loss matches the parent model’s loss (see § 3.2). The key difference is that Net2Net grows dense width or depth, while we grow MoE expert count, a fundamentally different axis that exploits sparse activation.

Stacking Your Transformers [9]. Systematically studied model growth operators (depth stacking, width expansion) for efficient LLM pre-training, showing that stacking can save 50%+ of training compute. Our work addresses a complementary growth axis: rather than stacking layers (depth), we duplicate experts (MoE width). The two approaches could potentially be composed for compound savings.

Stacking as Accelerated Gradient Descent [2]. Provided optimization-theoretic justification for why layer stacking accelerates training, connecting it to accelerated gradient descent. Our regret-bound decomposition (Appendix A) serves an analogous role for expert duplication, but the mechanisms differ: stacking exploits depth-wise structure, while our analysis addresses the capacity gap and initialization quality introduced by expert replication.

Deep Progressive Training [3]. Analyzed progressive depth expansion through optimization theory and feature learning, establishing conditions for function-preserving growth to maintain convergence. Our work extends this progressive training philosophy to the MoE expert-count dimension, with an analogous theoretical framework analyzing warm initialization and symmetry breaking in the sparse routing setting.

RAPTR: Progressive Subnetwork Training [43]. Proposed training random subnetworks (depthwise, width-wise) and progressively increasing subnetwork size, showing this generalizes and fixes issues with layer dropping. RAPTR operates within a fixed architecture by training subsets; expert upcycling instead expands the architecture by adding new experts. The approaches are complementary: RAPTR could be applied during the continued pre-training phase after upcycling.

Multi-linear Operators for Model Reuse [42]. Proposed correlating each weight of a target model to all weights of a pretrained model via multi-linear operators, capturing inter- and intra-weight interactions. This addresses dense model growth with full weight correlation. Expert upcycling takes a simpler approach (exact duplication) that achieves warm initialization by construction, avoiding the computational overhead of learning cross-weight mappings, and operates in the MoE setting where sparse routing provides a natural growth axis.

#### E.3 Upcycling from dense checkpoints

Sparse Upcycling [26]. The foundational work on converting dense checkpoints to MoE by replicating the FFN into multiple experts. The critical distinction from our work: Sparse Upcycling performs a dense → MoE transition, while expert upcycling performs MoE → larger MoE expansion. This difference has significant implications: (i) our starting point already has trained routing and specialized experts, (ii) we preserve the existing sparse computation pattern, and (iii) our method enables iterative capacity expansion without architectural regime changes.

Drop-Upcycling [40]. Improves upon Sparse Upcycling by partially re-initializing expert weights during the dense-to-MoE conversion, promoting diversity and maintaining a learning curve slope similar to from-scratch MoE training. Our experiments with heuristic initialization methods (Appendix D) include analogous drop-based strategies along with nine other approaches (noise injection, SVD perturbation, orthogonalization, interpolation, sparse code mixing, and others); we find that all heuristic methods yield negligible gains (<1e-3 validation loss) over simple copy-paste for MoE-to-MoE upcycling. This suggests that in the already-sparse setting, the trained router provides sufficiently strong symmetry-breaking signals during CPT, making elaborate initialization diversity unnecessary.

Scaling Laws for Upcycling MoE [33]. Derived scaling laws specifically for the dense-to-MoE upcycling transition, revealing a critical interaction term: upcycling efficiency decreases with longer dense pre-training because the sunk dense tokens slow subsequent MoE training progress. Our MoE-to-MoE setting exhibits the opposite behavior: upcycling efficiency increases with pre-training duration (Table 2a, transition-timing sweep), because the base MoE already possesses sparse routing structure and specialized experts that transfer directly to the expanded architecture. This qualitative reversal underscores that dense-to-MoE and MoE-to-MoE upcycling are governed by fundamentally different dynamics.

BAM! Parameter Upcycling [59]. Explored simple and efficient parameter upcycling strategies for creating MoE from dense models, finding that straightforward approaches can be surprisingly effective. Our findings align: simple expert duplication (COPY) is competitive with more elaborate heuristics for MoE-to-MoE growth. However, BAM focuses on the dense-to-MoE transition while we address the already-sparse setting.

Upcycling LLMs into MoE [19]. Studied upcycling at scale (NVIDIA), examining router design choices and expert granularity for converting dense LLMs to MoE. Their work provides practical recipes for the dense-to-MoE transition at production scale. Our contribution is orthogonal: we address the next step, growing an already-sparse model to have more experts.

DeRS: Efficient Upcycled MoE [21]. Proposed decomposing upcycled MoE experts into shared base weights and lightweight delta weights for parameter efficiency, applicable to both training and compression. DeRS addresses parameter redundancy within upcycled experts; our work addresses how to create more experts from existing ones. The two approaches could be combined: one could apply DeRS compression after expert upcycling to reduce the parameter overhead of the expanded model.

Nexus: Adaptive Upcycling [14]. Introduced an adaptive router with domain embeddings that can incrementally integrate new experts trained on new domains. Nexus and expert upcycling share the goal of expanding MoE capacity post-training, but differ in mechanism: Nexus adds independently trained domain-specific experts with a specialized router, while we duplicate existing experts and rely on continued pre-training for specialization. Our approach does not require domain-specific expert training and works with standard MoE routers.

LayerMoE [61]. Expands multilingual MoE capacity by adding new experts to an existing MoE backbone, with a layer-wise allocation algorithm that decides how many new experts each layer needs based on language-specific representation characteristics. Tokens from old languages are routed to the original experts while new-language tokens use the added experts, avoiding catastrophic forgetting. LayerMoE adds new, independently-trained experts targeted at new signal (new languages); expert upcycling instead duplicates existing experts and relies on continued pre-training for specialization on the same data distribution.

Branch-Train-Stitch [60]. Combines independently trained, domain-specialized LLM experts into a generalist via lightweight “stitch layers” inserted between frozen experts and a seed LLM. New experts can be added with minimal retraining of the stitch layers, supporting flexible capacity expansion. BTS adds new, externally-trained experts that bring new domain signal; expert upcycling reuses existing experts in place and does not require parallel domain-specific training runs.

Branch-Train-MiX [50]. Trains domain-specialized expert LLMs in embarrassingly parallel fashion, then merges them into an MoE with learned routing. BTX creates expert diversity through independent training on different data domains; expert upcycling creates diversity through duplication followed by symmetry breaking during joint continued pre-training. BTX requires parallel training infrastructure for each expert branch, while our method operates on a single training run.

Symphony-MoE [54]. Constructs an MoE from multiple disparate pretrained models (e.g., Qwen2 + Qwen2.5-Coder) via functional alignment using neuron permutation and SLERP-based parameter merging. Symphony-MoE maximizes initial expert diversity by leveraging independently trained models, but requires access to multiple compatible architectures and a training-free harmonization stage to resolve parameter space misalignment. Expert upcycling requires only a single MoE checkpoint and achieves diversity through continued training rather than multi-source assembly, making it applicable in settings where diverse pretrained models are unavailable.

Reuse, Don’t Retrain [44]. Provided practical guidelines for continued pre-training of language models, covering data distribution design and learning rate schedules. These recipes are complementary to expert upcycling: they inform how to design the continued pre-training phase after expert expansion. Our work focuses on the architectural growth step itself rather than the training recipe.

#### E.4 Expert specialization, diversity, and routing

Representation Collapse in MoE [6]. Identified and addressed the representation collapse problem where MoE routing encourages token clustering around expert centroids, proposing hyperspherical routing as a solution. This is relevant to expert upcycling because duplication initially creates identical expert representations; our theoretical analysis of symmetry breaking explains how continued training escapes this degenerate state.

Grove MoE [56]. Introduced a new MoE architecture with heterogeneous expert sizes (adjugate experts) and a dynamic per-token activation mechanism; the paper’s primary contribution is architectural. Their 33B-parameter instantiation (GroveMoE-Base) is produced by applying an upcycling strategy to the Qwen3-30B-A3B-Base MoE checkpoint during mid-training, but upcycling is a training strategy in their setup rather than the main contribution. Grove MoE therefore operates on a different axis than expert upcycling: it modifies expert sizes and activation dynamics, while expert upcycling modifies expert count with homogeneous experts and fixed top-K routing. The two are orthogonal and could be combined.

MoE Design Choices [11]. Systematically ablated MoE design choices, finding that routing granularity (token-level vs. sequence-level) has the largest impact on performance, while expert

collapse does not necessarily hurt validation perplexity. Their finding that top-2 token-level routing is the only configuration surpassing the dense baseline with equivalent total parameters motivates our choice of top-K values across experiments (top-K=2 for the interleaved MoE main result and top-K=8 for the full MoE result, matching frontier configurations).

#### E.5 MoE deployment and saliency metrics

Expert Pruning and Skipping [34]. Developed methods for reducing MoE inference cost by pruning or dynamically skipping experts. Expert upcycling and expert pruning are natural duals: upcycling adds capacity for quality improvement, pruning removes capacity for efficiency. Our utility-based expert selection identifies the most important experts to duplicate, concentrating added capacity where the loss is most sensitive.

Saliency metrics [16, 28, 17, 37, 49, 31]. The pruning literature provides a rich toolkit of saliency metrics. We repurpose these tools for capacity expansion: gradient-based importance scores identify which experts contribute most to the loss landscape, and we preferentially duplicate these high-utility experts.

#### E.6 Conditional compute and dynamic routing

Mixture-of-Depths [45]. Mixture-of-Depths (MoD) routes tokens to skip transformer layers entirely, varying the compute depth per token rather than routing to different expert FFNs. Like MoE, MoD decouples total model capacity from per-token compute, but through a depth-routing mechanism rather than expert selection. Expert upcycling is orthogonal: we expand the number of experts in an MoE model, not the depth of computation per token. The two approaches could in principle be combined.

#### E.7 Continual and lifelong learning

Continual pre-training and plasticity. Continual learning studies how models can acquire new knowledge without forgetting previously learned representations, with plasticity (the ability to continue learning) as a central concern [25, 15]. Expert upcycling involves a form of continued pre-training on new data after an architectural change, which raises related questions: do duplicated experts retain the source model’s representations while also specializing on new data? Our experiments show that the upcycled model maintains most of the from-scratch baseline’s capability across 11 downstream benchmarks (within 0.3 points on average at 100% CPT), suggesting that catastrophic forgetting is not a significant concern in our setting. We attribute this to the warm initialization: duplicated experts start from trained weights, so the model does not need to relearn existing knowledge from scratch.

