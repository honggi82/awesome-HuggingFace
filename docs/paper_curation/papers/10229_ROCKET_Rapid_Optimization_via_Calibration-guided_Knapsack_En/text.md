## ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression

# arXiv:2602.11008v1[cs.LG]11Feb2026

Ammar Ali*12 Baher Mohammad*12 Denis Makhov2 Dmitriy Shopkhoev2 Magauiya Zhussip2 Stamatios Lefkimmiatis2

### Abstract

We present ROCKET, a training-free model compression method that achieves state-of-theart performance in comparison with factorization, structured-sparsification and dynamic compression baselines. Operating under a global compression budget, ROCKET comprises two key innovations: First, it formulates layer-wise compression allocation as a multi-choice knapsack problem, selecting the optimal compression level for each layer to minimize total reconstruction error while adhering to a target model size. Second, it introduces a single-step sparse matrix factorization inspired by dictionary learning: using only a small calibration set, it sparsifies weight coefficients based on activation-weights sensitivity and then updates the dictionary in closed form via least squares bypassing iterative optimization, sparse coding, or backpropagation entirely. ROCKET consistently outperforms existing compression approaches across different model architectures at 20–50% compression rates. Notably, it retains over 90% of the original model’s performance at 30% compression without any finetuning. Moreover, when applying a light finetuning phase, recovery is substantially enhanced: for instance, compressing Qwen3-14B to an 8Bparameter model and healing it with just 30 million tokens yields performance nearly on par with the original Qwen3-8B. The code implementing ROCKET.

### 1. Introduction

In recent years, transformers have achieved unprecedented success across a wide range of tasks in both computer vision

*Equal contribution 1Department of Computer Science, ITMO University, Saint-Petersburg, Russia 2MWS AI, Moscow, Russia. Correspondence to: Ammar Ali <ammarali32@itmo.ru>, Baher Mohammad <b.mohammad@mts.ai>.

Preprint. February 12, 2026.

and natural language processing. Modern large language models (LLMs) typically scale up to billions of parameters, significantly increasing the computational and memory requirements for both training and inference stages. This substantial resource demand poses a critical challenge for their wider practical deployment, especially on edge devices or in latency-sensitive applications.

Due to the excessive size of modern LLMs, there has been significant research effort to make such models more efficient and accessible under constrained hardware budgets. Such efforts primarily focus on three key strategies: quantization (Hassibi & Stork, 1992), distillation(Hinton et al., 2015), and weight compression via matrix factorization (Wang et al., 2020). Among these, post-training weight factorization has emerged as a particularly promising direction, enabling substantial parameter reduction without the need for costly retraining or fine-tuning. A dominant paradigm in this area is low-rank approximation using truncated Singular Value Decomposition (SVD), which approximates each weight matrix as the product of two smaller dense matrices. However, this strategy imposes a rigid structural constraint forcing all columns of the weight matrix to lie in a single shared low-dimensional subspace. This often limits the representational capacity and leads to significant performance degradation under moderate to high compression.

This limitation has spurred the development of methods that go beyond a single shared subspace representation , adopting instead a union-of-subspaces framework akin to dictionary learning. In such models, a weight matrix is expressed as a combination of a subset of basis matrices (Zhussip et al., 2025), or alternatively, its individual columns are represented as sparse linear combinations of atoms from a shared dictionary (Shopkhoev et al., 2025). These formulations provide greater flexibility by capturing the heterogeneous local structures present within the weight matrix. Despite their theoretical appeal, practical adoption of these methods faces severe computational challenges: conventional dictionary learning algorithms rely on iterative alternating minimization between sparse coding and dictionary update steps, which is prohibitively expensive for large-scale LLM weight matrices (Aharon et al., 2006).

In this work, we propose ROCKET, a fast, training-free compression method that overcomes the representational rigidity of low-rank factorization while avoiding the computational burden of iterative dictionary learning. Our approach introduces two key innovations. First, ROCKET compresses weight matrices via a single-step structured sparsification of a low-rank basis. This yields a factorization that inherits the expressive power of union-of-subspaces models yet operates orders of magnitude faster than alternating minimization schemes. Second, rather than applying uniform compression or relying on heuristic layer-wise sensitivity estimates, ROCKET formulates global compression allocation as a multi-choice knapsack problem. For each layer, it selects the optimal compression configuration from a set of precomputed candidates to minimize total weight reconstruction error under a target model size constraint. Together, these components enable ROCKET to produce compact models that achieve substantially higher accuracy compared to existing post-training compression methods.

The contributions of this work are summarized as follows: (1) We propose ROCKET, an efficient, training-free LLM compression method that factorizes weight matrices into a sparse dictionary representation computable in a single step, eliminating the need for iterative optimization; (2) We introduce a calibration-guided criterion for sparsifying the coefficient matrix, operating effectively in both the original and whitened weight spaces to preserve salient directional information; (3) We formulate layer-wise compression allocation as a multi-choice knapsack problem, enabling dynamic, performance-aware distribution of the global compression budget across layers; (4) Through extensive experiments, we demonstrate that ROCKET consistently outperforms state-of-the-art compression methods including structured sparsification, low-rank factorization, and adaptive budget allocation techniques across multiple modalities (text, vision, and audio).

### 2. Related Work

This work intersects three primary research directions in model compression: (1) dynamic per-layer allocation of compression budgets, (2) structured matrix factorization for weight approximation, and (3) sparsification techniques. We review recent advances in each area, with emphasis on methods most relevant to our training-free, reconstructionaware compression framework.

Structured Matrix Factorization for Weight Approximation Early approaches employed truncated SVD to obtain low-rank approximations of transformer weights. However, several studies (Yuan et al., 2023; Wang et al., 2025b; Chen et al., 2021) demonstrated that weight matrices themselves are not inherently low-rank; instead, activations exhibit lowrank structure. These works proposed data-aware low-rank approximations using a whitening transform estimated from

a small calibration dataset, yielding significantly more effective compression.

A more general representation was recently introduced in (Shopkhoev et al., 2025), where weights are expressed as sparse linear combinations of dictionary atoms in a whitened space, computed via K-SVD and Orthogonal Matching Pursuit (OMP) updates. This approach overcomes the limitation of fixed, layer-invariant bases in low-rank methods since it allows each column of the weight matrix to reside in a different low-dimensional subspace, effectively promoting a more flexible union-of-subspaces modeling strategy.

Our approach extends this line of work by replacing the computationally intensive iterative K-SVD/OMP optimization with a novel single-step greedy algorithm. This eliminates alternating minimization while achieving higher reconstruction accuracy and orders-of-magnitude faster compression critical for scaling to billion-parameter models.

Budget Allocation and Layer Importance Many early compression methods apply uniform compression across all layers, implicitly assuming equal layer importance. Recent work challenges this assumption. LLM-Pruner (Ma et al., 2023b) estimates the importance of coupled layer groups using gradient and Hessian-based metrics, pruning less critical groups. ARS (Gao et al., 2024) proposes an adaptive rank selection mechanism using differentiable binary masks, regularized to respect the ordering of singular values from SVD, thereby allocating more capacity to important layers. DobiSVD (Wang et al., 2025a) introduces a learnable truncation threshold k per weight matrix, optimized during training via a multi-objective loss balancing task performance and global compression ratio. Similarly, ARA (Xv et al., 2025) dynamically assigns ranks to linear modules by learning a monotonic probabilistic mask over singular values, guided by a loss that accounts for cases where full-rank retention is more efficient than decomposition.

In contrast to these training-based approaches, our method performs budget allocation in a purely post-training setting. We formulate the problem as a multi-choice knapsack optimization, where each layer is associated with a discrete set of feasible compression configurations. Using dynamic programming, we select the globally optimal combination that minimizes total weight reconstruction error while enforcing a per-layer upper bound on relative reconstruction error, ensuring both global efficiency and local fidelity.

Sparsification Methods Unstructured pruning has demonstrated strong efficacy in compressing large language models. While magnitude-based pruning (Han et al., 2015) remains a simple baseline, Frantar et al. (Frantar & Alistarh, 2023) showed its inadequacy for LLMs and proposed SparseGPT, a Hessian-aware, layer-wise pruning method that reconstructs output errors via an efficient approximate

solver. Alternative importance metrics have also been explored. WANDA (Sun et al., 2024) computes a saliency score as the product of weight magnitude and the L2 norm of corresponding input activations (estimated from calibration data), pruning the lowest-scoring weights per output neuron. Bonsai (Kolawole et al., 2024) formulates module importance as an underdetermined regression problem, estimating importance using only forward passes to enable efficient structured pruning. Although sparsification achieves high compression ratios, it often yields irregular memory access patterns that hinder inference acceleration on modern hardware. Our method produces structured sparse-factorizations that are compatible with standard dense linear algebra operations, potentially offering a practical balance between compression efficiency, reconstruction quality, and hardware compatibility.

### 3. Method

Recent training-free compression methods for LLMs exhibit a fundamental trade-off between computational efficiency and representational flexibility. On one end, truncated SVD (SVD-LLM) (Wang et al., 2025b) enforces a rigid, globally shared low-rank subspace, enabling fast compression but severely limiting reconstruction fidelity under aggressive ratios. On the other, CoSpaDi (Shopkhoev et al., 2025) employs conventional sparse dictionary learning (K-SVD + OMP) to realize a union-of-subspaces model, at the cost of increased runtime and poor scalability for multi-billion models.

ROCKET bridges this gap by introducing a calibrationaware, single-step structured sparsification strategy grounded in eigen decomposition. Given a calibration dataset X ∈ RN×d

1×d2, we seek an approximation W that minimizes the activationaware reconstruction error:

1 and a weight matrix W ∈ Rd

arg min

W

XW − X W

F

subject to W ∈ C, (1)

where C imposes structural constraints on W.

Following established data-aware compression strategies, we operate in the whitened activation space. Let L be the Cholesky factor of the Gram matrix A = X⊤X, and define the decorrelated input Y = XL−1, which satisfies Y⊤Y = Id

. It can then be shown that the objective simplifies to:

1

min

W

XW − X W

= min

F

W

Y(LW − L W)

F

, (2)

LW − L W

= min

F

W

which is attributed to Y having orthogonal columns. If we denote the whitened weight as WL := LW, the problem reduces to minimizing WL − W

, with W = L W.

F

We compute the top-r eigenvectors of WLWL⊤:

WLWL⊤ ≈ BΛrB⊤, where B ∈ Rd

1×r is column-orthogonal (B⊤B = Ir) and Λr = diag(λ1,...,λr). The coefficient matrix is obtained via orthogonal projection: C := B⊤WL ∈ Rr×d

2 and a low-rank weight factorization is computed as WL ≈ BC. This formulation enables structured sparsification: rather than pruning W, we note that XW ≈ (YB)C and operate on C. Because B is semi-orthogonal, zeroing cij deactivates the i-th latent direction for the j-th output dimension. Since sparsity is applied independently per column, each output may activate a distinct subset of basis vectors realizing a union-of-subspaces model similar to dictionary learning.

However, the ultimate goal is to reconstruct W, not WL. The inverse whitening transform L−1 maps errors back to the original space, but it is generally non-orthogonal. Consequently, two coefficients with identical magnitudes in the whitened space may contribute very differently to the final reconstruction error, depending on how their corresponding basis vectors bi are scaled by L−1. Specifically, an error along bi incurs a cost proportional to ∥L−1bi∥2.

To account for this directional sensitivity, we define two complementary importance measures. The whitened-space importance impwhiteij reflects local optimality under orthonormality:

impwhiteij = |cij|. (3)

The original-space importance imporigij quantifies the actual impact on the reconstructed weight:

imporigij = |cij| · L−1bi 2 . (4) We fuse these importance scores via a scale-invariant geometric interpolation. For a balance parameter λ ∈ [0,1], the combined importance is:

λ

impij = impwhiteij 1−λ imporigij

= |cij| · L−1bi λ2 .

(5) This can be interpreted as solving a weighted sparse approximation problem in the whitened space, where weights encode the distortion induced by L−1. We set λ = 0.5 (geometric mean), which empirically balances activation fidelity and weight-space stability.

Let ν ∈ Rr with νi = L−1bi λ2. The full importance matrix is:

#### Imp = |C| ⊙ ν1⊤d

, (6)

2

applying row-wise scaling consistent with the per-direction nature of the metric distortion.

We then apply a two-stage sparsification strategy. First, we perform column-wise hard thresholding on C, retaining the top-s entries per column according to impij. To

Passing Optimal

Budget Budget Allocation

###### Compressed LLM

Optimal Path Graph Edges Start Node Compression Options End Node

Step 1 Step 2 Step L

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

EVD

Edge to End Node

exists if and only if cr at this node is >= crtarget

Figure 1. Overview of the proposed method. Left: Budget allocation formulated as a shortest-path problem on a directed graph, where nodes represent compression options and edges encode cost (reconstruction error), solved via DP algorithm to find the optimal sequence of operations. Right: The selected optimal path determines per-layer compression parameters (rank Ki and sparsity Si), which are then applied to each layer via Eigen value decomposition (EVD) followed by structured hard thresholding sparsification (T (.)) of coefficients.

allow global flexibility beyond fixed per-column budgets, we initially over-sparsify C to a ratio of cr + β where β is relatively small (we set β = 5e−3 in all experiments), then reactivate the most important masked coefficients across the entire matrix until the exact target compression ratio cr is reached. This yields a sparse coefficient matrix Csparse.

Crucially, after sparsification, there is no longer a need to enforce orthonormality on the basis. The initial eigenbasis B was used only to derive an expressive, data-adaptive representation; once Csparse is fixed, we can optimize the left factor freely to minimize reconstruction error in the whitened space. We therefore compute the final dictionary Dfinal (we use different notation to highlight that orthonormality for the left matrix factor is not required) by solving a ridge-regularized least-squares problem:

∥WL − DCsparse∥2F + µ∥D∥2F (7)

Dfinal = arg min

D

which admits a closed-form Cholesky-based solution. This step relaxes the semi-orthogonality constraint, yielding a better fit without increasing inference cost. The final compressed weight is recovered as:

W = L−1DfinalCsparse, (8) stored as two factors U = L−1Dfinal and V = Csparse.

Thus, ROCKET unifies three perspectives: (i) a closedform surrogate to iterative dictionary learning, replacing alternating updates with eigen decomposition and optimal thresholding; (ii) a generalization of SVD: when no sparsity is applied (s = r), it recovers standard low-rank SVD; (iii) a structured sparsification method, preserving the UV product for seamless merging during inference.

Layer Profiling. To enable optimal global compression under a fixed parameter budget, we first perform a

lightweight layer profiling pass. For each compressible layer, we evaluate a predefined set of (rank, sparsity) configurations. For each candidate, we: (i) compute the eigendecomposition of WLWL⊤, (ii) determine rank k and sparsity level s, (iii) sparsify C using fused importance scores, (iv) compute Dfinal via least squares, and (v) record the actual parameter count (cost), ks ratio, and relative reconstruction error (eℓ,i ≤ 1) in the original space for layer ℓ and option i given as eℓ,i = ∥Wℓ− Wℓi∥F

∥Wℓ∥F . This yields, per layer ℓ, a discrete set of feasible options Oℓ = {(cℓ,i,ksℓ,i,eℓ,i)}.

Constrained Multi-Choice Knapsack Formulation. Let there be L compressible layers. For layer ℓ ∈ {1,...,L}, let Oℓ = {(cℓ,i,ksℓ,i,eℓ,i)}K

i=1 denote its feasible compression options obtained during profiling, where cℓ,i ∈ R≥0 is the parameter count, ksℓ,i is the sparsity to truncation ratio, and eℓ,i ≥ 0 the Frobenius reconstruction error. Let Ctotal be the global parameter budget (e.g., for target cr% compression). The optimal allocation is traditionally cast as a multi-choice knapsack problem (MCKP):

ℓ

L

min

xℓ,i∈0,1

ℓ=1

L

s.t.

ℓ=1

Kℓ

eℓ,i,xℓ,i (9)

i=1

Kℓ

Kℓ

cℓ,i,xℓ,i ≤ Ctotal,

xℓ,i = 1, ∀ℓ.

i=1

i=1

To prevent degradation below a uniform-compression baseline, we introduce an additional hard constraint. Let e¯ref be the average reconstruction error across all layers when each is compressed at a fixed reference ratio (e.g., ρref). We then require: K

i=1 eℓ,i xℓ,i ≤ α.e¯ref ∀ℓ ∈ {1,...,L}, where, α is a tunable hyperparameter. In all experiments, we set α = αmin := inf {α′ > 0 | a feasible solution exists for α′}. Further per-

ℓ

model tuning of α may yield improved performance. This constraint eliminates pathological solutions that achieve low global error by severely damaging a few layers while overpreserving others. The problem remains MCKP with layerwise error caps, ensuring both global optimality and local robustness. We reformulate the problem using graph theory in Appendix C, which clarifies the dynamic programming solution that follows.

Dynamic Programming for Allocation. We solve the constrained MCKP via bottom-up dynamic programming. Let DPℓ[k] denote the minimal error after processing the first ℓ layers with discretized kept parameter count k. The recurrence is:

DPℓ+1[k + ⌊βκℓ+1,i⌋] = min

DPℓ[k] + εℓ+1,i , (10)

i

where β = param precision/Ptotal, κℓ,i is the kept count, and εℓ,i the error. After each layer, we prune dominated states (k1 < k2 and DP[k1] ≥ DP[k2]), keeping the state space small in practice. The algorithm runs in O(LMB¯) time and O(B¯) space, outperforming Dijkstrabased approaches in both speed and memory while yielding the same globally optimal solution.

In summary, ROCKET is a fully training-free pipeline that: (1) constructs a data-adaptive basis via eigen decomposition in the whitened activation space; (2) performs importanceweighted structured sparsification with global refinement; (3) relaxes orthogonality post-sparsification via a closedform least-squares update; and (4) allocates a global parameter budget through a constrained knapsack solver with perlayer robustness guarantees. This combination achieves the expressivity of sparse dictionary learning and the efficiency of spectral methods, enabling high-fidelity compression of billion-parameter LLMs with minimal overhead.

### 4. Experiments

In this section, we describe our experimental setup and compare ROCKET against recent compression methods. We focus on low-rank and dictionary-based approaches, specifically SVD-LLM (Wang et al., 2025b) and CoSpaDi (Shopkhoev et al., 2025), as well as budget allocation based methods including ARS (Gao et al., 2024), Dobi-SVD (Wang et al., 2025a), and ARA (Xv et al., 2025). We also provide Comparisons with sparsification and width-pruning methods such as LLM-Pruner (Ma et al., 2023a), SliceGPT (Ashkboos et al., 2024), Bonsai (Kolawole et al., 2024), and Wanda (Sun et al., 2024). We also conduct ablations to isolate the contribution of each design choice.

#### 4.1. Experimental Setup

We evaluate our method in a per-layer setting using LLaMA and Qwen models. All evaluations are performed in a zeroshot setting on the following benchmarks: PIQA (Bisk

et al., 2019), HellaSwag (Zellers et al., 2019), OpenAI LAMBADA (Paperno et al., 2016), ARC-Easy and ARCChallenge (Clark et al., 2018), SciQ (Welbl et al., 2017), RACE (Lai et al., 2017), and MMLU (Hendrycks et al., 2021a). In addition, we report perplexity on WikiText (Merity et al., 2016) and LAMBADA-OpenAI.

We apply compression at compression weight ratios ranging from 0.2 to 0.5, in steps of 0.1. For methods that need calibration data, we use 256 randomly sampled sequences from the RefinedWeb dataset (Penedo et al., 2023) (fixed across all experiments). We also test how the choice of calibration dataset affects results in the appendix. Unless otherwise noted, we compress all dense linear layers in the self-attention blocks (Q, K, V, and O projections) and the feed-forward network (gate, up, and down projections). Embedding layers and the lm head are not compressed following other works.

Comparison with SVD-LLM and CoSpaDi To contextualize ROCKET’s performance, we directly compare it against SVD-LLM and CoSpaDi, the two most closely related training-free compression methods. All methods are evaluated in a strictly training-free setting, no fine-tuning, healing, or data augmentation is applied post-compression.

As shown in Table 1 and Fig 2, ROCKET consistently outperforms both baselines by a significant margin across multiple architectures (Qwen3-8B, Llama3-8B and Llama3.2-1B) and compression ratios (20%–50%), in terms of both zeroshot accuracy and perplexity (check Appendix E.1 for more detailed results). Notably, ROCKET exhibits superior scalability under aggressive compression and increasing model scales (detailed in Appendix E.2): while baseline methods suffer severe degradation beyond 30% compression, ROCKET retains more robust performance even at 50% compression (e.g., 51.3 average accuracy on Qwen3-8B vs. 38.1 for SVD-LLM and 42.0 for CoSpaDi). This demonstrates that ROCKET’s combination of calibration-guided factorization, sparsification and optimal layer-wise budget allocation effectively preserves model fidelity under strict parameter constraints.

Comparison against other budget allocation methods To evaluate the effectiveness of ROCKET’s layer-wise budget allocation, we compare it against four established parameter allocation strategies: SVD-LLM (Wang et al., 2025b), which applies uniform compression across layers; Adaptive Rank Selection (ARS) (Gao et al., 2024); Dobi-SVD (Wang et al., 2025a); and Adaptive Rank Allocation (ARA) (Xv et al., 2025). Figure 3 shows normalized performance across three model-compressions configurations Qwen3-8B at 20% and 40% compression, and LLaMA2-7B at 40% compression with all scores scaled to their respective dense baselines. ROCKET consistently outperforms all baselines, demonstrating that globally optimizing the allocation of parameters

- Table 1. Performance comparison of ROCKET vs SOTA SVD-LLM and CoSpaDi methods on Qwen3-8B at different compression ratios (CR). Best results are highlighted with bold.

Method CR Accuracy↑ Perplexity↓ PIQA HellaSwag LAMBADA ARC-e ARC-c SciQ Race MMLU Avg. WikiText LAMBADA

Qwen3 8B – 77.7 74.9 64.1 80.7 56.7 95.7 40.9 73.0 70.5 1.2E+01 4.6E+00 SVD-LLM 73.8 63.9 62.2 68.7 45.7 90.1 40.5 54.7 62.5 2.1E+01 6.4E+00

0.2

CoSpaDi 76.5 68.0 65.6 72.2 48.9 93.2 40.7 60.8 65.7 1.8E+01 4.9E+00 ROCKET

###### 77.6 72.9 66.0 75.8 53.9 94.5 41.4 67.2 68.7 1.5E+01 4.7E+00

SVD-LLM 70.4 55.2 53.8 59.3 37.1 87.2 38.4 44.8 55.8 2.7E+01 1.1E+01 CoSpaDi 72.4 60.5 62.6 63.9 41.2 88.4 39.5 51.3 60.0 2.3E+01 6.3E+00 ROCKET

0.3

###### 75.1 67.2 68.0 72.1 47.7 92.9 41.4 62.3 65.8 1.8E+01 4.4E+00

SVD-LLM 66.3 44.6 37.9 45.0 28.1 77.3 35.3 29.1 45.4 4.3E+01 3.6E+01 CoSpaDi 68.9 49.0 49.9 49.4 29.9 82.0 36.8 36.6 50.3 3.6E+01 1.5E+01 ROCKET

0.4

###### 71.0 58.7 63.7 65.7 40.4 86.1 41.9 52.3 60.0 2.4E+01 5.9E+00

SVD-LLM 61.5 34.9 25.1 37.4 25.3 65.1 31.6 24.0 38.1 7.6E+01 8.8E+01 CoSpaDi 63.8 39.7 32.4 41.2 26.8 70.4 33.2 28.1 42.0 5.9E+01 4.1E+01 ROCKET

0.5

###### 68.8 48.4 47.5 54.8 33.0 81.6 38.5 37.9 51.3 3.5E+01 2.4E+01

Llama3.2-1B

Llama3-8B

- 0.8
- 1

- 0.8
- 1

80

60

60

PPL↑1/ln()

Orig Acc SVD-LLM Acc

AvgAcc↑

0.6

0.6

40

CoSpaDi Acc Ours Acc

40

Orig 1/ ln(P) SVD-LLM 1/ ln(P)

0.4

0.4

CoSpaDi 1/ ln(P) Ours 1/ ln(P)

20

20

0.2

0.2

0.2 0.3 0.4 0.5

0.2 0.3 0.4 0.5

Compression Ratio (CR)

Compression Ratio (CR)

Figure 2. Comparison of Accuracy and Inverse Log Perplexity for Llama3-8B and Llama3.2-1B.

via constrained knapsack selection preserves significantly more of the original model’s capabilities than uniform or trainable strategies, especially under aggressive compression and across different architectures.

Comparison with Depth and Sparsity Pruning Methods To demonstrate the effectiveness of our method, we compare it against several approaches that address model pruning from different perspectives. These include depth pruning (e.g., SliceGPT), combined depth/width pruning (e.g., LLM-Pruner), structured sparsification (e.g., Wanda, Bonsai), and adaptive low-rank decomposition with quantization (e.g., Dobi-SVD). As shown in Table 2, ROCKET achieves strong performance, outperforming all other baselines at a 60% compression ratio (0.56 average accuracy). Since Dobi-SVD uses quantization, we additionally evaluate ROCKET with post-compression quantization to ensure a fair comparison. Under this setting, ROCKET surpasses Dobi-SVD at a 40% compression ratio (0.65 vs. 0.63). At a 60% compression ratio, ROCKET again leads with an average accuracy of 0.60 compared to Dobi-SVD’s 0.52. This demonstrates that ROCKET’s training-free pipeline is not only simple and fast but also highly effective, matching or exceeding other pruning methods.

#### 4.2. Post-Compression Healing

To evaluate the potential for a lightweight recovery, we apply a simple healing step, fine-tuning on a small amount of data to the ROCKET-compressed Qwen3-14B model, which was reduced to 8B parameters (40% compression). During healing, we fine-tune both the unmasked entries in the factorized weights and the associated dictionary, while keeping the sparsity pattern fixed. Training is performed on 30 million tokens of high-quality text sampled from the AllenAI C4 dataset.

As shown in Table 3, the healed model, named ROCKETQwen3-8B (healed) achieves an average accuracy of 67.96, substantially improving over the training-free compressed version (63.56) and approaching the performance of the native Qwen3-8B (70.46), even surpassing it on several benchmarks (e.g., PIQA, Lambada). For per-benchmark results, please refer to appendix E.4. This demonstrates that ROCKET not only excels in the training-free regime but also provides a high-quality initialization that enables effective recovery with minimal data and compute.

Critically, this result represents a practical step forward for model development: rather than training multiple models of different sizes from scratch, one can train a single large

###### Method Comparison per Model (Normalized to Model's Dense Baseline) C4 PPL inverted ( ); all values in [0, 1]

Uniform

ARS

Qwen3-8B (20%)

Qwen3-8B (40%)

LLaMA2-7B (40%)

Dobi-SVD ARA ROCKET

C4 PPL

C4 PPL

C4 PPL

1.00

|PIQA|ARC-e<br><br>|
|---|---|
|Wino|Hella|

|PIQA|ARC-e<br><br>|
|---|---|
|Wino|Hella<br><br>|

PIQA

ARC-e

0.75

0.50

0.25

MathQA

ARC-c

MathQA

ARC-c

MathQA

ARC-c

Wino

Hella

OBQA

OBQA

OBQA

Figure 3. Comparison of ROCKET with alternative budget allocation methods (Uniform, ARS(Gao et al., 2024), Dobi-SVD(Wang et al., 2025a), and ARA(Xv et al., 2025)) on three model configurations: Qwen3-8B at 20% and 40% pruning, and LLaMA2-7B at 40% pruning. Subplots show normalized performance on eight benchmarks (C4 perplexity inverted so higher is better), scaled to each model’s dense baseline (value=1.0). ROCKET consistently retains the most performance under the same parameter constraints.

- Table 2. Performance comparison of ROCKET against depth- and sparsity-based pruning methods on Llama3.1-8B across compression levels and benchmarks. ”Avg” denotes the average accuracy across all benchmarks, ”Drop” indicates the relative accuracy drop percentage compared to the dense (uncompressed) model, and ”Quant.” indicates whether post-compression quantization is applied.

Models CR Method Quant. PIQA HellaSwag WinoGrande ARC e ARC c Avg. (↑) Drop (↓)

LLaMA-3.1-8b

– Baseline ✗ 0.80 0.59 0.74 0.81 0.51 0.69 0%

0.4

LLM-Pruner ✗ 0.66 0.32 0.54 0.58 0.23 0.46 33.3% SliceGPT ✗ 0.62 0.40 0.53 0.49 0.25 0.46 33.3% Bonsai ✗ 0.59 0.29 0.49 0.47 0.18 0.41 40.6% Wanda-sp ✗ 0.57 0.28 0.50 0.44 0.17 0.39 43.5% Dobi-SVD ✓ 0.76 0.52 0.72 0.73 0.39 0.63 8.70% ROCKET ✗ 0.72 0.43 0.66 0.64 0.33 0.56 18.84%

ROCKET ✓ 0.78 0.54 0.72 0.76 0.42 0.65 5.79% 0.6 Dobi-SVD ✓ 0.68 0.41 0.66 0.58 0.27 0.52 24.6% ROCKET ✓ 0.75 0.49 0.68 0.72 0.38 0.60 13.0%

- Table 3. Healing results of Qwen3-14B model after compressing by 40% resultsing in an 8B version.

Method perplexity Avg. Acc.

Qwen3-14B (dense) 1.1E+01 73.32 Qwen3-8B (dense) 1.2E+01 70.46 ROCKET-Qwen3-8B (training-free) 2.4E+01 63.56 ROCKET-Qwen3-8B (healed) 1.3E+01 67.96

model and compress it to any desired size using ROCKET, leveraging the resulting sparsity for faster inference (see Appendix D), and optionally applying light healing to recover performance. With cleaner data and longer training, the healed model has the potential to match or even surpass a dense counterpart of equal size, offering a flexible, efficient, and scalable alternative to traditional multi-size training pipelines.

- 4.3. Generalization to Other Modalities

Table 4. Average accuracy of Qwen3-4B-VL before and after ROCKET compression (20%).

###### Method MMB MMMU MMS OCR RWQA

dense 83.76 49.44 61.85 81.70 71.50 ROCKET 78.95 44.44 54.85 74.50 65.75

and validate speech results on different samples from mls eng 10k (Pratap et al., 2020). For Qwen3-4B-VL, we construct a multimodal calibration set using 256 samples from the MathVista portion of the MathVerse dataset and evaluate on MMBench-en-dev (MMB) (Liu et al., 2023a), MMMU-val (MMMU) (Yue et al., 2024), MMStar (MMS) (Chen et al., 2024), OCRBench (Liu et al., 2023b), and RealWorldQA (RWQA). In both cases, we compress the models to 20% of their original size without any fine-tuning.

As shown in Table 4, ROCKET preserves strong performance on Qwen3-4B-VL, achieving 65.75 average accuracy (over 90% of the original model’s performance). On VibeVoice, Table 5 shows near-identical speech quality: WER remains stable (0.149 vs. 0.148), and UTMOS drops only slightly (3.43 vs. 3.52), staying close to the groundtruth reference (3.73). These results demonstrate that ROCKET generalizes effectively across modalities.

To assess the generality of ROCKET beyond languageonly models, we apply it to two transformer-based architectures from different modalities: the vision-language model Qwen3-4B-VL and the speech generation model VibeVoice (Peng et al., 2025). For VibeVoice, we use 256 transcriptions-only from mls eng 10k dataset

- Table 5. ROCKET applied to VibeVoice (speech generation model) at 20% compression.

Method WER ↓ UTMOS ↑

Ground Truth 0.04 3.73 VibeVoice (dense) 0.148 3.52 ROCKET 0.149 3.43

5. Ablations

To evaluate the contribution of each key component in ROCKET, we conduct a series of ablation studies using the Llama3-1B model. All experiments follow the evaluation protocol in Section 3.1, ensuring a fair and controlled comparison. We report average accuracy across the benchmarks along with word-level WikiText perplexity. In this section, we focus on two central ablations: (1) the choice of reconstruction error metric used during layer profiling, (2) the individual contributions of ROCKET’s two core components a) structured sparsification and b) dynamic per-layer budget allocation, In the appendix F we also provide ablations on the effect of calibration data, and alternative sparsification strategies. All variants are evaluated at a fixed global compression ratio of 20%, with all other design choices held constant, enabling precise attribution of performance differences to specific methodological choices within the ROCKET framework.

5.1. Ablation on Reconstruction Error Metric

In ROCKET, the layer profiling stage enumerates a set of candidate compression configurations per layer, where each candidate is defined by a pair (cr,ks), namely, the compression ratio and the sparsity ratio applied to the coefficient matrix. Given the original layer weights W, each candidate (cr,ks) induces a reconstructed weight matrix W˜ . For each candidate, we compute a reconstruction error, which serves as the estimated cost in the constrained multi-choice knapsack problem. The optimizer then selects one candidate per layer to minimize the sum of estimated error while satisfying the global parameter budget.

We evaluate four variants for this per-candidate error estimate: (1) relative Frobenius error ∥W − W˜ ∥F/∥W∥F

(our default), (2) ℓ1 distance ∥W − W˜ ∥1, (3) mean cosine distance across columns, and (4) spectral distance

- Table 6. Ablation on reconstruction error metric for layer profiling in ROCKET for Llama3-1B at 20% compression.

Error Metric Avg. Acc. ↑ Perplexity ↓

None (Baseline) 57.6 1.2E+01 L1 Distance 35.2 1.8E+02 Mean Cos Columns 51.1 1.9E+01 Spectral Distance 51.3 1.8E+01 Frobenius (ours) 52.4 1.8E+01

Table 7. Ablation study on ROCKET’s core components for Llama3-1B at 20% compression. ROCKET† uses uniform budget allocation across layers. “Budg. Alloc.” means Budget Allocation.

Method Sparse Budg. Alloc. Avg. Acc. ↑ Perplexity ↓

None — — 57.6 1.2E+01 SVD-LLM ✗ ✗ 37.6 1.7E+02 CoSpaDi ✓ ✗ 42.7 6.4E+01 ROCKET† ✓ ✗ 45.4 2.7E+01 ROCKET ✓ ✓ 52.4 1.8E+01

∥W − W˜ ∥2. All variants run under the same configurations. As shown in Table 6, the relative Frobenius error yields the best downstream performance, while ℓ1-based estimates lead to significant degradation, highlighting the effectiveness of this metric for effective budget allocation.

#### 5.2. Ablation on core components

To assess the contribution of ROCKET’s key design choices, we compare three variants: (1) SVD-LLM which uses neither sparsification nor dynamic budget allocation ;(2) CoSpaDi, which uses K-SVD-based sparse dictionary learning; (3) ROCKET with uniform compression across layers; and (4) full ROCKET, which further incorporates optimal knapsack-based budget allocation. As shown in Table 7, we first notice that using sparsification improves both average accuracy and perplexity. Moreover, replacing CoSpaDi’s iterative sparsification with our closed-form, activation-aware approach is not only much faster, but also improves average accuracy from 42.7 to 45.4 and reduces perplexity from 64 to 27. Adding optimal budget allocation yields a further significant gain, reaching 52.4 average accuracy and 18 perplexity, demonstrating that both our sparsification strategy and global parameter allocation are critical to ROCKET’s performance.

### 6. Conclusion and Limitations

ROCKET introduces a fast, training-free LLM compression method that combines calibration-guided structured weight factorization with optimal layer-wise budget allocation via a knapsack formulation. It achieves state-of-the-art performance retaining over 90% of original accuracy at 30% compression without any fine-tuning. Neverrtheless, it is important to mention that the dynamic programming solution, while efficient for standard dense models, is hard to scale to architectures with a very large number of compressible components such as modern Mixture-of-Experts (MoE) models with 128 or more experts per block. This is due to the combinatorial growth in compression options and scalable alternatives are left for future work. Moreover, our healing experiments assume a fixed sparsity pattern determined during the training-free compression phase, which is sub-optimal. Jointly learning adaptive sparsity patterns during fine-tuning may yield further improvements and is a direction we intend to explore.

### 7. Ethical Statement and Broader Impact

ROCKET is a training-free compression method designed to improve the efficiency and accessibility of large language models without requiring additional data or extensive computational resources for fine-tuning. By enabling highfidelity model compression with minimal environmental and economic cost, it supports more sustainable deployment of AI systems, particularly in resource constrained settings. The method does not introduce new data collection, human annotation, or model behaviors beyond those already present in the original pretrained model; thus, it neither amplifies nor mitigates existing biases in the base model. Users should remain vigilant about the ethical implications of the underlying model’s outputs, as ROCKET preserves its functional characteristics including potential biases or safety limitations. We encourage responsible deployment, including thorough evaluation and alignment measures when compressed models are used in real-world applications.

### References

Aharon, M., Elad, M., and Bruckstein, A. K-svd: An algorithm for designing overcomplete dictionaries for sparse representation. IEEE Transactions on Signal Processing, 54(11):4311–4322, 2006. doi: 10.1109/TSP.2006. 881199.

Ashkboos, S., Croci, M. L., Nascimento, M. G. D., Hoefler, T., and Hensman, J. Slicegpt: Compress large language models by deleting rows and columns. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/ forum?id=vXxardq6db.

Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y. Piqa: Reasoning about physical commonsense in natural language, 2019.

Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.

Chen, P. H., Yu, H.-f., Dhillon, I. S., and Hsieh, C.-j. Drone: data-aware low-rank compression for large nlp models. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA, 2021. Curran Associates Inc. ISBN 9781713845393.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018.

Frantar, E. and Alistarh, D. Sparsegpt: Massive language models can be accurately pruned in one-shot. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 10323–10337. PMLR, 2023.

Gao, S., Hua, T., Hsu, Y.-C., Shen, Y., and Jin, H. Adaptive rank selections for low-rank approximation of language models. In Duh, K., Gomez, H., and Bethard, S. (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 227–241, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.13. URL https:// aclanthology.org/2024.naacl-long.13/.

Han, S., Pool, J., Tran, J., and Dally, W. J. Learning both weights and connections for efficient neural network. In Cortes, C., Lawrence, N. D., Lee, D. D., Sugiyama, M., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pp. 1135–1143, 2015.

Hassibi, B. and Stork, D. Second order derivatives for network pruning: Optimal brain surgeon. In Hanson, S., Cowan, J., and Giles, C. (eds.), Advances in Neural Information Processing Systems, volume 5. MorganKaufmann, 1992.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding, 2021a.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the MATH dataset. In Vanschoren, J. and Yeung, S. (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021b.

Hinton, G. E., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. CoRR, abs/1503.02531, 2015. URL http://arxiv.org/abs/1503.02531.

Kolawole, S., Dery, L., Kagy, J.-F., Smith, V., Neubig, G., and Talwalkar, A. Everybody prune now: Structured pruning of llms with only forward passes. arXiv preprint arXiv:2402.05406, 2024.

Lai, G., Xie, Q., Liu, H., Yang, Y., and Hovy, E. Race: Large-scale reading comprehension dataset from examinations, 2017.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhnag, S., Zhao, W.,

- Yuan, Y., Wang, J., Liu, C. H. Z., Chen, K., and Lin,

- D. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281, 2023a.

Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X., lin Liu, C., Jin, L., and Bai, X. Ocrbench: On the hidden mystery of ocr in large multimodal models. 2023b. doi: 10.1007/s11432-024-4235-6.

Ma, X., Fang, G., and Wang, X. Llm-pruner: On the structural pruning of large language models. Advances in neural information processing systems, 36:21702–21720, 2023a.

Ma, X., Fang, G., and Wang, X. Llm-pruner: On the structural pruning of large language models. In Advances in Neural Information Processing Systems, 2023b.

Macko, V. and Boˇza, V. Macko: Sparse matrix-vector multiplication for low sparsity. arXiv preprint arXiv: 2511.13061, 2025.

Marcus, M. P., Santorini, B., and Marcinkiewicz, M. A. Building a large annotated corpus of English: The Penn Treebank. Computational Linguistics, 19(2): 313–330, 1993. URL https://www.aclweb.org/ anthology/J93-2004.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models, 2016.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q. N., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fern´andez, R. The lambada dataset: Word prediction requiring a broad discourse context, 2016.

Penedo, G., Malartic, Q., Hesslow, D., Cojocaru, R., Cappelli, A., Alobeidli, H., Pannier, B., Almazrouei, E., and Launay, J. The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023.

Peng, Z., Yu, J., Wang, W., Chang, Y., Sun, Y., Dong, L., Zhu, Y., Xu, W., Bao, H., Wang, Z., Huang, S., Xia, Y., and Wei, F. Vibevoice technical report. CoRR, abs/2508.19205, 2025. doi: 10.48550/ARXIV. 2508.19205. URL https://doi.org/10.48550/ arXiv.2508.19205.

Pratap, V., Xu, Q., Sriram, A., Synnaeve, G., and Collobert, R. Mls: A large-scale multilingual dataset for speech research. ArXiv, abs/2012.03411, 2020.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. GPQA: A graduate-level google-proof q&a benchmark. CoRR, abs/2311.12022, 2023. doi: 10.48550/ARXIV.

2311.12022. URL https://doi.org/10.48550/ arXiv.2311.12022.

Shopkhoev, D., Zhussip, M., Makhov, D., Ali, A., and Lefkimmiatis, S. Cospadi: Compressing llms via calibration-guided sparse dictionary learning. 2025. URL https://openreview.net/forum? id=oLBIcEHhxs.

Sprague, Z., Ye, X., Bostrom, K., Chaudhuri, S., and Durrett, G. Musr: Testing the limits of chain-of-thought with multistep soft reasoning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum? id=jenyYQzue1.

Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https: //openreview.net/forum?id=PxoFut3dWW.

Suzgun, M., Scales, N., Sch¨arli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., and Wei, J. Challenging big-bench tasks and whether chain-of-thought can solve them. In Rogers, A., Boyd-Graber, J. L., and Okazaki, N. (eds.), Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pp. 13003–13051. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-ACL. 824. URL https://doi.org/10.18653/v1/ 2023.findings-acl.824.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca, 2023.

Wang, Q., Ke, J., Tomizuka, M., Keutzer, K., and Xu, C. Dobi-svd: Differentiable SVD for LLM compression and some new perspectives. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025a. URL https://openreview.net/forum? id=kws76i5XB8.

- Wang, X., Zheng, Y., Wan, Z., and Zhang, M. Svd-llm: Truncation-aware singular value decomposition for large language model compression. In International Conference on Learning Representations, 2025b.
- Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., Li, T., Ku,

M., Wang, K., Zhuang, A., Fan, R., Yue, X., and Chen, W. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In Globersons, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J. M., and Zhang, C. (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

Wang, Z., Wohlwend, J., and Lei, T. Structured pruning of large language models. In Webber, B., Cohn, T., He, Y., and Liu, Y. (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pp. 6151–6162. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020. EMNLP-MAIN.496. URL https://doi.org/10.

18653/v1/2020.emnlp-main.496. Welbl, J., Liu, N. F., and Gardner, M. Crowdsourcing multiple choice science questions, 2017.

Xv, L., Gao, J., Gao, X., Liu, T., and Fu, Y. ARA: adaptive rank allocation for efficient large language model SVD compression. CoRR, abs/2510.19389, 2025. doi: 10.48550/ARXIV.2510.19389. URL https://doi.

org/10.48550/arXiv.2510.19389.

- Yuan, Z., Shang, Y., Song, Y., Wu, Q., Yan, Y., and Sun, G. ASVD: activation-aware singular value decomposition for compressing large language models. CoRR, abs/2312.05821, 2023. doi: 10.48550/ARXIV. 2312.05821. URL https://doi.org/10.48550/ arXiv.2312.05821.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., Wei, C., Yu, B., Yuan, R., Sun, R., Yin, M., Zheng, B., Yang, Z., Liu, Y., Huang, W., Sun, H., Su, Y., and Chen, W. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence?, 2019.

Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., and Hou, L. Instruction-following evaluation for large language models. CoRR, abs/2311.07911, 2023. doi: 10.48550/ARXIV.2311.07911. URL https:// doi.org/10.48550/arXiv.2311.07911.

Zhussip, M., Shopkhoev, D., Ali, A., and Lefkimmiatis, S. Share your attention: Transformer weight sharing via matrix-based dictionary learning. arXiv preprint arXiv:2508.04581, 2025.

### A. Proposed Algorithm

In Algo. 1, we present the proposed method in the form of a formal algorithm to facilitate a clear and systematic understanding of the method’s complete pipeline construction and implementation.

Algorithm 1 ROCKET: Training-Free Heterogeneous transformer Compression Require: Pre-trained transformer with linear layers {W(ℓ)}Lℓ=1, calibration data X ∈ RN×d

1, global parameter budget Ctotal

Ensure: Compressed model with factorization {B(ℓ),C(ℓ)}Lℓ=1

- 1: Compute whitening transform: Lt ← chol(X⊤X)
- 2: for each layer ℓ = 1 to L do
- 3: WL(ℓ) ← LtW(ℓ)
- 4: for each candidate rank r ∈ R and sparsity ratio s ∈ S do
- 5: Apply Eigen Value Decomposition to find top-r eigenvectors B: WL WL⊤ ≈ B ΛrB⊤,
- 6: Form coefficient matrix: C ← BTWL
- 7: Compute importance scores: impij ← |cij| · ∥L−1bi∥λ2 {λ = 0.5}
- 8: Over-sparsify C to ratio s + β, then reactivate top entries globally to reach exact sparsity s
- 9: Solve for optimal left factor: D ← arg minB ∥WL(ℓ) − BCsparse∥2F
- 10: Reconstruct weight: W(ℓ) ← Lt−1DCsparse
- 11: Record option: cost cℓ,i ← nnz(D) + nnz(Csparse), error eℓ,i
- 12: Store (cℓ,i, eℓ,i) in Oℓ
- 13: end for
- 14: end for
- 15: Compute reference error e¯ref from uniform compression baseline and minα that admits a solution
- 16: Initialize DP table: DP0[0] ← 0, others ← ∞
- 17: for ℓ = 1 to L do
- 18: for each state k in DPℓ−1 do
- 19: for each option i ∈ Oℓ do
- 20: if eℓ,i ≤ α · e¯ref then
- 21: k′ ← k + cℓ,i
- 22: DPℓ[k′] ← min DPℓ[k′], DPℓ−1[k] + eℓ,i
- 23: end if
- 24: end for
- 25: end for
- 26: Prune dominated states in DPℓ: remove (k1,err1) if ∃(k2,err2) with k2 ≥ k1 and err2 ≤ err1
- 27: end for
- 28: Find k∗ = arg mink≥C

total

DPL[k]

- 29: Backtrack to recover optimal per-layer choices {i∗ℓ}Lℓ=1
- 30: for ℓ = 1 to L do
- 31: Assign D(finalℓ) ,C(finalℓ) from option i∗ℓ
- 32: end for

### B. Norm-Preserving Properties of Low-Rank Approximation and Sparsification

Let W ∈ Rd

1×d2 with singular value decomposition (SVD)

#### W = UΣV⊤,

where U ∈ Rd

1×r, V ∈ Rd

2×r have orthonormal columns, Σ = diag(σ1,...,σr), σ1 ≥ ··· ≥ σr > 0, and r = rank(W).

#### B.1. Relative Error of Rank-k Truncated SVD The optimal rank-k approximation (1 ≤ k ≤ r) is

Wk = UkΣkVk⊤, with error given by the Eckart–Young–Mirsky theorem:

r

∥W − Wk∥2F =

σi2.

i=k+1

Since ∥W∥2F = ri=1 σi2, it follows that

with equality iff k = 0.

∥W − Wk∥F ∥W∥F

=

r i=k+1 σi2

r i=1 σi2 ≤ 1,

#### B.2. Sparsification of the Coefficients

Let T : Rk×d

2 be an entrywise sparsification operator satisfying

→ Rk×d

2

∥T (ΣkVk⊤)∥F ≤ ∥ΣkVk⊤∥F Define the sparsified reconstruction as

W = BkT (ΣkVk⊤). Since T is norm-non-increasing,

∥T (ΣkVk⊤)∥F ≤ ∥ΣkVk⊤∥F = ∥Wk∥F ≤ ∥W∥F. Thus,

∥ W∥F = ∥T (ΣkVk⊤)∥F ≤ ∥W∥F. The worst-case relative error occurs when W = 0, yielding

∥W − W∥F ∥W∥F

= 1.

For any non-zero sparsified reconstruction derived from the SVD basis of W, the error is strictly less than or equal to this maximum. Therefore,

∥W − W∥F ∥W∥F

≤ 1.

#### B.3. Equivalence of Eigenvalue-Based Basis Construction and SVD in ROCKET

1×d2 is generally rectangular and non-symmetric, so eigenvalue decomposition (EVD) cannot be applied directly to W. Instead, compression operates on the whitened weight WL = LW, where L = chol(X⊤X) whitens the input activations using a small calibration set X ∈ RN×d

In ROCKET, the weight matrix W ∈ Rd

1. The method then computes the top-k eigenvectors of the symmetric positive semi-definite matrix WLWL⊤. This EVD yields

WLWL⊤ = BkΛkB⊤k , where Bk ∈ Rd

1×k has orthonormal columns (B⊤k Bk = Ik) and Λk = diag(λ1,...,λk) with λi ≥ 0. Let the compact singular value decomposition of WL be

#### WL = UΣV⊤.

Then

#### WLWL⊤ = UΣ2U⊤,

which is precisely the eigenvalue decomposition of WLWL⊤. Therefore, the eigenbasis coincides exactly with the left singular vectors: Bk = Uk and λi = σi2. The coefficient matrix is obtained via orthogonal projection:

Ck = B⊤k WL = U⊤k WL = ΣkVk⊤, which matches the right factor in the SVD. Hence, the reconstruction

#### WL = BkCk

is identical to the rank-k truncated SVD of WL. Sparsification is applied to Ck using an operator T : Rk×d

2 satisfying ∥T (Ck)∥F ≤ ∥Ck∥F.

→ Rk×d

2

The sparsified approximation in the whitened space is defined as

WL = Bk T (Ck).

Since Bk has orthonormal columns, the Frobenius norm is preserved under multiplication:

∥ WL∥F = ∥T (Ck)∥F ≤ ∥Ck∥F = ∥WL∥F. Consequently, the relative reconstruction error in the whitened space satisfies

∥WL − WL∥F ∥WL∥F

≤ 1,

with equality only in the degenerate case WL = 0. Because the EVD-derived basis Bk is mathematically identical to the left singular vectors of WL, all norm-preserving properties and error bounds established for truncated SVD carry over verbatim.

Finally, the compressed weight in the original space is recovered as

W = L−1 WL. Although L−1 is not orthogonal, the theoretical guarantees in the whitened space where the core approximation occurs remain intact. Thus, despite using EVD of WLWL⊤ for computational efficiency, ROCKET achieves the same norm constraints and relative error bound (≤ 1) as classical SVD-based low-rank approximation followed by norm-non-increasing sparsification.

### C. Mapping MCKP to Graph Theory.

We reformulate the constrained multi-choice knapsack problem (MCKP) as a shortest-path problem on a directed acyclic graph (DAG), where the key to enforcing the global compression ratio lies in a clever target-aware sink connectivity rule. Each node is labeled (ℓ,p), with ℓ ∈ {0,...,L} denoting the layer index and p ∈ Z≥0 representing the scaled cumulative number of parameters pruned up to layer ℓ, discretized via p = scale · ℓℓ′=1 cℓ′,i using scale = param precision/Ctotal. For each feasible compression option i in layer ℓ, we add an edge from (ℓ−1,pin) to (ℓ,pout) with pout = pin + ⌊scale · cℓ,i⌋ and edge cost ⌊error scale factor · eℓ,i⌋, while discarding any option violating the per-layer error cap eℓ,i ≤ αe¯ref. Crucially, we connect a terminal node (L,p) to the sink if and only if p ≥ pmin = ⌊scale · (1 − ρtarget) · Ctotal⌋, where ρtarget is the desired compression ratio. This structural embedding of the global budget constraint via node naming and selective sink connectivity ensures that any path reaching the sink automatically satisfies the target compression, transforming the constrained combinatorial optimization into an unconstrained shortest-path search solvable exactly (up to discretization) by Dijkstra’s algorithm. The graph thus encodes both layer-wise flexibility and global feasibility, with complexity rendered tractable by limiting the candidate space per layer to a modest grid of compression ratios and ks sparsity levels.

This graph formulation underlies the dynamic programming procedure in Algorithm 1, where states correspond to nodes and transitions to edges.

p = pmin

- (2, p(1)2 )

- (2, p(2)2 )

- (2, p(3)2 )

- (2, p(4)2 )

- (L, p(1)L )

- (L, p(2)L )

- (L, p(3)L )

- (1, p(1)1 )

- (1, p(2)1 )

- (1, p(3)1 )

Feasible sink

Start (0 layers) (0, 0)

(L, p ≥ pmin)

State (i, p): i = number of layers processed p = scaled total parameters kept

After layer 0

After layer L−1

Transition: pick option for layer i Sink: only if p ≥ pmin

After layer 1

- Figure 4. Exact state-space graph matching. Each state (i, p) represents having processed the first i layers with p scaled parameters retained. From (i, p), the algorithm branches to all options for layer i, producing states (i + 1, p + ∆p). The sink is reachable only from states with p ≥ pmin, enforcing the global compression ratio by construction.

D. Inference Optimization

To accelerate inference, we leverage MACKO (Macko & Boˇza, 2025), a specialized sparse matrix–vector multiplication kernel that outperforms PyTorch’s built-in implementation for structured, column-wise sparse matrices. Because ROCKET employs a calibration-guided, adaptive compression strategy, the sparsity level quantified by the ratio k/s of dictionary atoms to nonzeros per coefficient column varies across layers. This heterogeneity arises from the solution to a constrained multi-choice knapsack problem, which allocates parameters to layers based on their marginal contribution to reconstruction fidelity under a global budget.

Empirically, we observe that MLP layers (gate and up projections) are assigned significantly higher sparsity and lower rank than attention projections (query, key, value, out). This behavior is theoretically justified: MLP weight matrices are substantially larger (dmodel × 4dmodel) than attention matrices (dmodel × dmodel), and their calibration-aware reconstruction error increases more slowly with sparsification. Consequently, the optimizer preferentially compresses MLP layers to maximize parameter savings while preserving overall model accuracy.

- Figure 5 compares the runtime of MLP projections in Qwen3-8B using either PyTorch’s default sparse kernels or MACKO. For large, moderately sparse coefficient matrices (e.g., gate and up projections), MACKO provides consistent speedups; for smaller or denser matrices (e.g., down projections), both implementations perform similarly. In attention layers, where weight matrices are small and less aggressively compressed, we retain PyTorch’s native implementation, as it proves faster in practice. Additionally, we fuse dictionaries and coefficients for layers sharing the same input namely, {query, key, value} in attention and {gate, up} in the MLP before applying MACKO.

In terms of theoretical floating-point operations (FLOPs), ROCKET is structurally analogous to CoSpaDi: both represent a weight matrix W ∈ Rd

2 is column-wise sparse. Assuming optimal reuse of the intermediate product XD, the FLOP count is Nd1Kactive + Nsd2, where Kactive is the number of distinct active atoms. Under a fixed global parameter budget, the total FLOP count may be similar between the two methods.

##### 1×d2 as W ≈ BC, where B ∈ Rd

1×k is a dense dictionary and C ∈ Rk×d

However, a key distinction lies in budget allocation: CoSpaDi typically enforces uniform sparsity across layers, whereas ROCKET dynamically assigns heterogeneous (kℓ,sℓ) pairs per layer based on reconstruction sensitivity. This results in lower FLOPs in large, robust layers (e.g., MLP) and higher FLOPs in small, sensitive layers (e.g., attention) even though the latter contribute minimally to total computation due to their size. The net effect is a more balanced per-layer runtime profile and better utilization of sparse kernels, which explains the consistent throughput advantage of ROCKET over CoSpaDi (Table 8) despite comparable theoretical operation counts.

Table 8. Throughput (tokens/s) for Qwen3-8B with batch size = 1 and context length = 256.

Method 20% 30% 40%

SVD-LLM 24.36 24.31 24.72 CoSpaDi 25.45 25.76 25.62 ROCKET 26.74 26.60 26.36

Latency per Layer for ROCKET Representation of MLP Projections

Gate Projection

Up Projection

Down Projection

built-in torch implementation

0.250

0.12

0.12

Macko

0.225

0.11

0.11

0.200

RunningTime(s)

RunningTime(s)

RunningTime(s)

0.175

0.10

0.10

0.150

0.125

0.09

0.09

0.100

built-in torch implementation

built-in torch implementation

Macko

Macko

0.08

0.08

0.075

0 5 10 15 20 25 30 35

0 5 10 15 20 25 30 35

0 5 10 15 20 25 30 35

Projection Index

Projection Index

Projection Index

Figure 5. Comparison between PyTorch’s built-in sparse-matrix vector multiplication and MACKO across MLP layers in Qwen3-8B. MACKO shows consistently better running time for large coefficients (gate and up projections), while being on par with PyTorch for smaller sparse matrices (down projection).

#### D.1. Environmental Impact

As shown in Table 9, the ROCKET method not only achieves superior compression performance but also offers dramatic environmental benefits. Compared to COSPADI, ROCKET consumes over 100 times less energy, completes compression 96 times faster, and produces 23 times lower CO2 emissions. These results highlight ROCKET as both a high-performance and environmentally sustainable solution.

- Table 9. Comparison of energy consumption, runtime, and CO2 emissions for COSPADI and ROCKET using the Llama3-1b model.

Method Model GPU CPU Energy Consumed (kWh) Duration (s) CO2 Emissions (kg eq) CoSpaDi Llama3-1b 1 × NVIDIA A100-SXM4-40GB AMD EPYC 7742 64-Core Processor 7.88 90080.97 0.782 ROCKET Llama3-1b 1 × NVIDIA A100-SXM4-40GB AMD EPYC 7742 64-Core Processor 0.0765 930 0.0337

E. Further results

E.1. Detailed Comparison with CoSpaDi and SVD-LLM

We begin this section by providing the detailed per-benchmark results from Figure 2 in Tables 10, 11. As shown, ROCKET outperforms CosPaDi and SVD-LLM by a large margin across different benchmarks and compression ratios.

- Table 10. ROCKET comparison vs low-rank and SDL counterparts in data-aware scenarios on Llama3.2-1B at different compression ratios (CR). Best results are provided in bold All experiments are in training-free setup.

Accuracy↑ Perplexity↓ Method CR

|PIQA Hella Swag LAMBADA ARC-e ARC-c SciQ Race MMLU Avg.<br><br>|Wiki Text LAMBADA|
|---|---|
|Llama3.2 1B – 74.5 63.7 63.0 60.5 36.2 88.3 37.8 37.0 57.6<br><br>|1.2E+01 5.7E+00|
|SVD-LLM 62.1 36.4 24.4 36.0 25.1 64.9 29.0 23.0 37.6 CoSpaDi 66.1 42.9 38.4 39.9 26.0 71.6 31.7 24.8 42.7 ROCKET<br><br>0.2<br><br>71.9 56.7 46 56.7 32.4 88.6 36.4 30.8 52.4<br><br>|1.7E+02 1.7E+02 6.4E+01 3.5E+01<br>1.8E+01 1.3E+01<br>|
|SVD-LLM 55.7 30.1 9.1 30.5 21.5 45.9 25.8 23.2 30.2 CoSpaDi 56.9 32.4 18.2 31.9 22.1 56.7 28.0 23.1 33.7 ROCKET<br><br>0.3<br><br>66.3 46.8 38.0 47.9 27.4 79.6 34.1 27.4 45.9<br><br>|5.9E+02 2.5E+03<br><br>2.9E+02 6.6E+02<br>3.5E+01 2.6E+01<br>|
|SVD-LLM 51.8 27.3 1.3 26.9 22.9 32.3 24.4 23.0 26.2 CoSpaDi 53.5 28.2 3.8 27.8 23.0 36.9 24.0 23.1 27.5 ROCKET<br><br>0.4<br><br>63.9 39.4 23.8 41.0 25.7 72.1 30.9 23.0 39.8|1.6E+03 3.3E+04 8.0E+02 9.2E+03 8.8E+01 1.3E+02<br><br>|
|SVD-LLM 51.1 26.6 0.0 26.1 25.9 26.1 23.9 23.0 25.4 CoSpaDi 51.7 27.0 0.3 26.3 24.0 29.5 24.2 23.3 25.8 ROCKET<br><br>0.5<br><br>57.3 31.6 9.4 34.9 22.6 50 26.1 22.9 31.9<br><br>|3.1E+03 1.0E+05 1.8E+03 7.3E+04 3.3E+02 2.2E+03|

- Table 11. Performance comparison of ROCKET vs SOTA SVD-LLM methods on Llama3-8B at different compression ratios (CR). Best results are highlighted with bold.

Method CR Accuracy↑ Perplexity↓ PIQA HellaSwag LAMBADA ARC-e ARC-c SciQ Race MMLU Avg. WikiText LAMBADA

Llama3 8B – 80.7 79.1 75.6 77.7 53.5 93.9 40.3 62.2 70.4 7.3E+00 3.1E+00 SVD-LLM 71.1 58.4 59.3 55.5 34.0 86.4 35.5 32.6 54.1 4.1E+01 1.1E+01

0.2

CoSpaDi 75.2 66.5 73.8 66.5 41.6 89.5 38.2 42.8 61.8 2.0E+01 4.3E+00 ROCKET

###### 76.9 74.8 73.6 73.7 47.1 92.7 40.7 54.9 66.8 1.1E+01 3.8E+00

SVD-LLM 65.8 46.4 38.1 41.9 27.7 70.0 31.8 27.2 43.6 1.5E+02 6.1E+01 CoSpaDi 70.5 56.2 61.3 54.2 33.5 85.7 36.2 32.2 53.7 4.5E+01 9.2E+00 ROCKET

0.3

###### 76.8 69.5 70.4 70.5 42.3 90.9 39.7 47.8 63.5 1.5E+01 4.4E+00

SVD-LLM 60.3 34.5 11.4 32.4 24.5 44.2 25.7 23.1 32.0 5.5E+02 1.3E+03 CoSpaDi 63.7 41.4 30.3 39.1 26.6 68.5 30.5 25.4 40.7 1.8E+02 1.2E+02 ROCKET

0.4

###### 71.9 60.4 59.3 64.4 36.4 88.1 36.8 39.7 57.1 2.9E+01 8.1E+00

SVD-LLM 55.1 24.7 1.8 26.0 23.1 30.5 22.4 21.0 25.6 1.8E+03 6.5E+03 CoSpaDi 58.4 31.8 13.6 30.7 24.8 46.2 25.8 23.4 31.8 7.4E+02 5.2E+02 ROCKET

0.5

#### E.2. Scaling Behavior Across Model Sizes

Average Accuracy vs. Model Size

AvgAcc(%ofUncompressedModel)↑

100

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

90

80

70

60

50

0 5 10 15 20 25 30 35

Model Size (Billion Parameters)

CR = 0.2 CR = 0.4

Figure 6. Average accuracy (relative to uncompressed model) as a function of model size for two compression ratios.

In Figure 6, we present the results of compressing Qwen models of varying sizes from 0.6B to 32B parameters at two compression ratios (20% and 40%). The results show that larger models retain a higher fraction of their original (uncompressed) performance after compression. This suggests that larger models may still be significantly underfitted relative to their capacity.

#### E.3. Evaluation on Advanced Benchmarks

Table 12. Performance comparison of ROCKET against CoSpaDi and SVD-LLM on a new set of benchamrks.

Method CR Accuracy↑

IfeVal BBH MATH GPQA MUSR MMLU-Pro Qwen3 8B — 39.21 60.86 52.57 36.16 43.12 47.72

- SVD-LLM 0.2 25.54 41.00 1.06 28.36 39.81 26.30

- CoSpaDi 0.2 28.90 45.25 1.96 28.61 42.06 31.46

- ROCKET 0.2 31.89 50.33 11.10 31.45 39.68 36.65

SVD-LLM 0.3 22.90 34.42 0.98 25.59 41.40 18.82 CoSpaDi 0.3 25.18 38.22 0.98 24.75 38.36 22.81

- ROCKET 0.3 25.54 47.54 2.64 29.19 39.94 32.27

SVD-LLM 0.4 22.66 30.24 0.83 23.07 37.70 11.55 CoSpaDi 0.4 26.14 32.72 0.76 26.01 38.10 16.74

- ROCKET 0.4 23.99 40.06 1.28 28.85 41.00 25.53

- In Table 12 we compare against other methods on a new set of benchmarks. These modern benchmarks IFEVal (Zhou et al., 2023), BBH (Suzgun et al., 2023), MATH(Hendrycks et al., 2021b), GPQA(Rein et al., 2023), MuSR(Sprague et al., 2024), and MMLU-Pro(Wang et al., 2024) represent an evolution in the evaluation of large language models (LLMs), targeting more rigorous, diverse, and realistic capabilities. IfeVal (Instruction-Following Evaluation) assesses a model’s ability to follow precise, verifiable natural language instructions. BBH (Big-Bench Hard) isolates 23 of the most challenging tasks from the original BIG-Bench suite to probe complex reasoning. MATH evaluates advanced mathematical problem-solving across algebra, geometry, and other domains. GPQA tests graduate-level scientific knowledge with high difficulty and minimal data contamination risk. MuSR (Multi-step Reasoning) focuses on multi-hop and long-context reasoning, while MMLU-Pro enhances the original MMLU by increasing answer choices (from 4 to 10) and reducing ambiguity, thereby offering a cleaner, more demanding assessment of expert knowledge. Unfortunately other papers are still following the old benchmarks therefore for fair comparison we stick with them for comparison in the main paper.

E.4. Post-Compression Healing

- In Table 13, we provide the per-benchmark results corresponding to the summary in Table 3. As noted, we are approaching the performance of models trained from scratch and in some cases nearly matching or surpassing them despite using a compressed model. Specifically, we compressed Qwen-14B to an 8B model and applied a very limited fine-tuning phase (using only approximately 30 million tokens) while keeping the sparsity pattern fixed an approach that is known to be suboptimal. Nevertheless, the resulting model achieves performance comparable to the original Qwen3-8B trained from scratch. We expect that fine-tuning with higher quality, carefully curated data would further improve results. Moreover, as previously mentioned, enabling trainable sparsity patterns remains a direction for future work.

Table 13. Post-compression healing results on Qwen3-8b models.

Method PIQA HellaSwag Lambada ARC-e ARC-c SciQ Race MMLU WikiText Preplexity Avg. Acc.

Qwen3-14B (dense) 79.86 78.85 67.88 82.82 60.23 96.50 43.25 77.20 1.1E+01 3.7E+00 73.32 Qwen3-8B (dense) 77.70 74.90 64.10 80.70 56.70 95.70 40.90 73.00 1.2E+01 4.6E+00 70.46 ROCKET-Qwen3-8B (training-free) 72.68 62.63 70.26 67.76 44.19 91.20 39.80 59.99 2.5E+01 3.8E+00 63.56 ROCKET-Qwen3-8B (healed) 78.51 74.67 65.55 75.29 53.07 93.50 37.89 65.23 1.3E+01 4.7E+00 67.96

### F. Further Ablations

#### F.1. Ablation on calibration data

- Table 14. Ablation on calibration data for Llama3-1B at 20% compression (CR = 0.2). The first row shows the uncompressed baseline. Higher accuracy and lower perplexity are better.

Calibration Data CR Accuracy↑ Perplexity↓ PIQA HellaSwag LAMBADA ARC-e ARC-c SciQ Race MMLU Avg. Acc. WikiTextword LAMBADAPPL

None (Baseline) 0 74.5 63.7 63.0 60.5 36.2 88.3 37.8 37.0 57.6 1.2E+01 5.7E+00 RefinedWeb 0.2 71.9 56.7 46.0 56.7 32.4 88.6 36.4 30.8 52.4 1.8E+01 1.3E+01 PTB 0.2 70.3 54.9 47.3 55.2 31.9 87.8 34.5 26.7 51.1 2.1E+01 1.3E+01 WikiText 0.2 70.7 56.3 50.9 57.9 33.6 88.1 35.5 28.3 52.7 1.7E+01 1.1E+01 Alpaca 0.2 73.7 57.2 53.2 57.9 34.9 87.5 35.4 30.5 53.8 2.0E+01 9.7E+00

We evaluate the sensitivity of ROCKET to the choice of calibration dataset by comparing four sources: RefinedWeb (Penedo et al., 2023), PTB (Marcus et al., 1993), WikiText (Merity et al., 2016), and Alpaca (Taori et al., 2023). As shown in Table 14, while instruction-tuned data such as Alpaca yield slightly higher average accuracy (53.8 vs. 52.4), the differences across datasets are relatively small, confirming that ROCKET is robust to the calibration data choice. Nevertheless, to ensure a fair comparison with CoSpaDi, which uses RefinedWeb as its default calibration data, we adopt RefinedWeb for all primary experiments reported in this paper.

F.2. Ablation on the Sparsification Strategy

- Table 15. Ablation on sparsification strategies for Llama3-1B at 20% compression (CR = 0.2). All methods use the same calibration data and global parameter budget.

Sparsification Strategy CR Accuracy↑ Perplexity↓ PIQA HellaSwag LAMBADA ARC-e ARC-c SciQ Race MMLU Avg. Acc. WikiTextword LAMBADAPPL

None (Baseline) 0 74.5 63.7 63.0 60.5 36.2 88.3 37.8 37.0 57.6 1.2E+01 5.7E+00 ROCKET 0.2 71.9 56.7 46.0 56.7 32.4 88.6 36.4 30.8 52.4 1.8E+01 1.3E+01 Per-Row Sparsification 0.2 67.8 48.1 32.5 48.6 29.4 79.9 31.6 26.7 45.6 3.2E+01 3.8E+01 Global Importance Sparsification 0.2 69.5 54.7 42.7 53.5 30.7 85.7 35.5 28.1 50.1 1.9E+01 1.7E+01 Whitened-Space Importance Only 0.2 71.5 55.7 47.3 56.3 32.9 87.9 35.4 28.9 52.0 1.9E+01 1.3E+01

To evaluate the impact of our sparsification strategy, we compare ROCKET against three alternative approaches for pruning the coefficient matrix C = ΣV⊤: (1) Per-Row Sparsification, where importance scores are computed identically but sparsity is enforced independently per row (breaking column-wise structure); (2) Global Importance Sparsification, which ignores any structural constraints and simply zeros out the globally least-important entries based on the full importance matrix; and (3) Whitened-Space Importance Only, which disables the original-space fidelity term by setting λ = 0 in Eq. (8) (Theoretically optimal with respect to the whitened space). Table 15 show that ROCKET’s column-aware, activation-and-weight-balanced sparsification outperforms alternatives, demonstrating the efficiency of both structural awareness and dual-space importance scoring.

