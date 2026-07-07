# arXiv:2511.11062v1[cs.CV]14Nov2025

## LiteAttention: A Temporal Sparse Attention for Diffusion Transformers

#### Dor Shmilovich Tony Wu Aviad Dahan Yuval Domb MoonMath.ai

research@moonmath.ai

https://github.com/moonmath-ai/LiteAttention

### Abstract

Diffusion Transformers, particularly for video generation, achieve remarkable quality but suffer from quadratic attention complexity, leading to prohibitive latency. Existing acceleration methods face a fundamental trade-off: dynamically estimating sparse attention patterns at each denoising step incurs high computational overhead and estimation errors, while static sparsity patterns remain fixed and often suboptimal throughout denoising. We identify a key structural property of diffusion attention, namely, its sparsity patterns exhibit strong temporal coherence across denoising steps. Tiles deemed non-essential at step t typically remain so at step t + δ. Leveraging this observation, we introduce LiteAttention, a method that exploits temporal coherence to enable evolutionary computation skips across the denoising sequence. By marking non-essential tiles early and propagating skip decisions forward, LiteAttention eliminates redundant attention computations without repeated profiling overheads, combining the adaptivity of dynamic methods with the efficiency of static ones. We implement a highly optimized LiteAttention kernel on top of FlashAttention and demonstrate substantial speedups on production video diffusion models, with no degradation in quality.

### 1 Introduction

Video generation via Diffusion Transformers (DiT) has reached a remarkable inflection point: models now produce compelling, high-fidelity content that rivals professional production quality. Yet this capability conceals a fundamental inefficiency. Despite their impressive generative performance, these models demand extraordinary computational resources, generating a single 5-second video can take up to 30 minutes even on state-of-the-art GPUs. The primary culprit is the computationally quadratic attention Vaswani et al. (2017) mechanism: in some video diffusion architectures, the attention mechanism alone accounts for up to 80% of the total inference latency Chen et al. (2024a).

The computational burden of diffusion models has motivated extensive research into efficient inference strategies Ma et al. (2024b); Liu et al. (2025b); Ma et al. (2024a); Tang et al. (2024). Two main directions have emerged: (1) Dynamic methods, which exploit sparsity within each denoising step Xi et al. (2025); Li et al. (2025); Zhang et al. (2025a,b); and (2) Static and caching methods, which exploit redundancy across denoising steps Zou et al. (2025); Bu et al. (2025); Lv et al. (2025); Liu et al. (2025a); Chu et al. (2025). Dynamic methods repeatedly determine which computations to skip, incurring overhead and estimation noise, while static methods may misalign with evolving attention patterns. Critically, none exploit the temporal persistence of sparsity patterns across denoising steps. Our central observation is that tiles deemed non-essential at denoising step t tend to remain non-essential at step t + δ. This temporal coherence of sparsity enables a fundamentally different strategy: identify skippable tiles once during early denoising, and propagate these skip decisions forward through the entire trajectory.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

We present LiteAttention, which leverages temporal sparsity coherence to propagate computation skips through the denoising process. By determining skip patterns early and reusing them throughout the denoising processs, LiteAttention achieves three key advantages simultaneously: (1) the content adaptivity of dynamic sparsity (patterns are derived from actual attention statistics), (2) the efficiency of static sparsity (no per-step re-evaluation overhead), and (3) the completeness of full computation elimination. Together, these properties yield substantial acceleration while preserving the generative fidelity of DiT.

#### 1.1 Evolutionary Computation Skips

LiteAttention performs evolutionary skips, completely eliminating attention computation for tiles marked as skippable. Once a tile is skipped, the entire attention iteration for that tile is bypassed across subsequent timesteps where the skip decision remains valid. This full-iteration elimination distinguishes LiteAttention from prior sparse attention approaches that skip only partial attention computations, where major bottlenecks such as softmax evaluation and memory transfers continue to dominate runtime despite partial sparsification.

#### 1.2 Implementation and Robustness

LiteAttention integrates seamlessly into modern CUDA-accelerated attention kernels via FlashAttention3, maintaining full compatibility while introducing only moderate memory overhead for storing metadata. The method is production-ready and does not require model retraining or architectural modifications.

To ensure robustness when skip decisions persist across timesteps with varying denoising conditions, LiteAttention incorporates a lightweight calibration mechanism that weights approximation errors by their layer-dependent impact. This calibration acts as a supporting component to maintain accuracy, while the core contribution lies in the evolutionary skip mechanism.

Our key contributions are summarized as follows:

- • Evolutionary skip framework. We introduce a mechanism that exploits the temporal coherence of sparsity patterns to eliminate full attention computations for marked tiles across denoising timesteps.
- • Amortized sparsity profiling. We determine which tiles can be skipped early in the denoising process and reuse these skip decisions for all subsequent timesteps, eliminating the need for repeated sparsity profiling.
- • Efficient GPU implementation. We develop optimized CUDA kernels leveraging evolved skip masks with moderate memory overhead, achieving efficient runtime performance.

### 2 Related Work

Accelerating diffusion models requires addressing their substantial computational demands. Prior work has approached this challenge along two orthogonal directions: reducing attention within individual denoising steps, or exploiting redundancy across the denoising sequence.

#### 2.1 Per Timestep Sparse Attention

Recent work has observed that attention patterns exhibit significant sparsity within individual denoising steps. Methods in this category determine sparse patterns either statically or dynamically, but crucially, these determinations are made independently at each step.

Sparse VideoGen (SVG) Xi et al. (2025); Yang et al. (2025) presented dynamic sparse attention for video DiTs by classifying attention heads into spatial and temporal categories and profiling sparse patterns within each step. Sparse-vDiT Chen et al. (2024a) complements this with architectural insights, showing that attention patterns follow recurring structures: diagonal blocks for self-frame interactions, multi-diagonal blocks for cross-frame consistency, and vertical stripes for global tokens. These patterns are largely input-invariant and intrinsic to the model architecture, yet their stability across timesteps is not exploited.

Radial Attention Li et al. (2025) formalizes Spatiotemporal Energy Decay, proposing static O(nlog n) masks. While theoretically efficient, static patterns sacrifice adaptivity. SpargeAttention Zhang et al. (2025a) presents a sparse attention framework that predicts low-attention blocks via two-stage online filtering which requires profiling at every step. Sliding Tile Attention (STA) Zhang et al. (2025b) takes advantage of the concentration of attention at the tile-level through sliding windows in granularity of the tile.

In every one of these approaches, sparsity patterns are determined independently at each denoising step, either via dynamic recomputation or static commitment, without leveraging their persistence across steps.

#### 2.2 Cross-Sequence Redundancy Exploitation

Another line of research leverages structure across denoising timesteps, observing that different phases exhibit varying computational requirements.

TGATE Liu et al. (2025b) noted that cross-attention outputs converge in early denoising steps, enabling phase-based computation where patterns are cached and reused later. DeepCache Ma et al.

- (2024b) demonstrates that high-level transformer features remain similar across adjacent timesteps, allowing feature caching without retraining. Learning-to-Cache (L2C) Ma et al. (2024a) learns timestep-dependent layer-level routing strategies, identifying which layers benefit from caching throughout the denoising sequence. Token-wise Feature Caching Zou et al. (2025) captures finegrained temporal redundancy at the token level using layer-specific caching ratios. AdaDiff Tang et al. (2024) implements dynamic early exit based on timestep-aware uncertainty, allocating the computation proportional to the importance of each phase. While these approaches exploit crosssequence structure in features or layer utilization, they do not consider temporal structure in attention sparsity patterns. Additionally, they retain or approximate intermediate values, incurring nontrivial estimation errors and significant memory overheads.

#### 2.3 Sparsity Stability Across Denoising: A New Paradigm

Unlike per-step sparse attention methods, which recompute patterns at each denoising step, or crosssequence methods that exploit feature redundancy, LiteAttention is based on a fundamentally different principle: attention sparsity patterns remain stable throughout the denoising sequence.

The key observation is that attention sparsity is temporally coherent rather than random. By exploiting this persistence, LiteAttention determines skip patterns once and reuses them across the entire trajectory, achieving full elimination of attention computation for marked tiles without repeated profiling. LiteAttention thus enables a new class of accelerators that combine adaptive pertimestep sparsity with zero repeated profiling. Concurrent work, SparseD Wang et al. (2025), observes similar temporal sparsity stability in diffusion language models, applying cross-step sparsity in a different domain.

### 3 Preliminaries

#### 3.1 Attention Mechanisms

FlashAttention. FlashAttention Dao (2024) is an efficient attention algorithm that significantly reduces the memory bandwidth requirements of standard attention implementations by employing a tiling-based memory strategy. It is specifically optimized for NVIDIA GPUs, though its principles naturally extend to other parallel-processing architectures.

Given the standard attention formulation

√

S = QK⊤/

d, (1) P = σ(S), (2) O = PV, (3)

where Q ∈ Rn×d

k are the query, key, and value matrices, n denotes the sequence length, and dq,dk are the hidden dimensions (for simplicity we set d := dq = dk). The softmax

q and K,V ∈ Rn×d

operator σ(·) is applied row-wise such that pij := exp(sij)/ k exp(sik) where pij and sij are elements of P and S, respectively.

In FlashAttention, the matrices Q, K, and V are partitioned along the sequence dimension into tile sets {Qi}, {Kj}, {Vj} where the tiles are of sizes hq × w, hk × w, and hk × w, respectively. These tiles are processed sequentially using the online softmax algorithm Milakov and Gimelshein (2018), which maintains numerical stability and enables incremental accumulation of partial results. For each query tile Qi, the computation proceeds iteratively over the key-value tile pairs {(Kj,Vj)}:

√

Sij = QiKj⊤/

d, (4) ( Pij,mij) = σ˜(Sij,mi,j−1), (5) Oij = PijVj, (6)

where Oij is a partial output, Pij := exp(Sij − mij), and mij := rowmax(mi,j−1,rowmax(Sij)), where rowmax(·) operates on a matrix and outputs a vector of the maximums of all rows. The value

mij is updated cumulatively across the tile index j. For brevity, the remainder of the online softmax procedure is omitted.

SpargeAttention. The sparse online softmax, introduced in SpargeAttention Zhang et al. (2025a), extends FlashAttention with a dynamic pruning mechanism that skips partial computation of tiles whose contribution to the output is negligible. Specifically, when the local maximum mlocal := rowmax(Sij) is significantly smaller than the cumulative maximum mij, the corresponding tile has exponentially suppressed weights. The computation of tile (Kj,Vj) is partially skipped when

max(mlocal − mij) ≤ −ε, (7) for a chosen threshold ε > 0 (note that we are required to take a maximum since m is a vector of row maximums). In this case, max( Pij) ≤ e−ε, implying that the term PijVj contributes negligibly to the final output and can be safely omitted. With an appropriately chosen threshold, SpargeAttention preserves the guarantees of the online softmax while eliminating redundant computation and improving efficiency.

- 3.2 Flow, Diffusion Transformers, and Caching Flow. We begin by defining an Ordinary Differential Equation (ODE) as

dXt = u(Xt,t)dt, (8) X0 = x0, (9)

where t ∈ [0,1], u(x,t) is a vector field, and x0 is an initial condition. A solution to this ODE is termed a trajectory, and the collection of all trajectories arising from all possible initial conditions

constitutes a flow. When Xt is regarded as a random process, the flow induces a mapping between two distributions, p(X0) and p(X1). An alternative formulation, the Stochastic Differential Equation (SDE),1 gives rise to a diffusion process instead of a deterministic flow. In this work, we use the terms flow and diffusion interchangeably.

Consider X0 ∼ pinit(x) and X1 ∼ pdata(x), where pinit(x) is a simple noise distribution independent of the data distribution pdata(x). A trajectory under this formulation maps a noise sample into a meaningful data sample. A flow model (also known as a flow matching) Lai et al. (2025) parameterizes the vector field uθ(x,t) with learnable parameters θ, and is trained to reverse the forward process

##### Xt = √αtX1 + √1 − αtX0, (10)

where 0 ≤ αt ≤ 1 is the noise schedule. In this work, we consider an extended form of the vector field,

##### uθ(x,t,c), (11)

where c denotes a conditioning signal that guides trajectories toward the conditional data distribution pdata(x | c).

1An SDE takes the form dXt = u(Xt, t) dt + σt dWt where Wt is Brownian motion.

Diffusion Transformers. A DiT Peebles and Xie (2023) is a feedforward architecture composed of M bidirectional transformer blocks (or layers). Each block typically consists of a sequence of submodules: self-attention, cross-attention, and a multilayer perceptron (MLP). See Peebles and Xie (2023) and Chen et al. (2024b) for common variants of transformer block architectures. Let Ti denote the i-th transformer block in a DiT; its output is defined as the composition

yti = Ti ◦ Ti−1 ◦ ··· ◦ T1(xt,t,c), (12)

where (t,c) serves as conditioning input to all transformer blocks and the final DiT output is yt := ytM. Our learned diffusion vector field uθ(x,t,c) is implemented using this DiT backbone.

Caching. Caching exploits the slow temporal evolution of DiT outputs across diffusion timesteps to improve computational efficiency. Following Bu et al. (2025), define the residual at layer i and timestep t as

rti := yti − xt. (13)

In particular, the final residual rt := rtM often changes slowly between adjacent timesteps, i.e. ∥rt − rt−1∥ is small. This property can be exploited by caching and reusing rt−1 instead of recomputing the full DiT output at each step.

Furthermore, it has been observed in Bu et al. (2025) that a small relative change between successive intermediate representations,

γti := ∥yti − yti−1∥ ∥yti∥

, (14)

correlates with a small relative change between successive outputs γt := γtM. This insight underpins timestep-level caching strategies that leverage temporal smoothness to accelerate diffusion inference, a concept that is directly linked to our findings.

### 4 Method

##### 4.1 LiteAttention

Our initial approach aimed to optimize the runtime performance of self-attention by incorporating the sparse online softmax technique from SpargeAttention. As discussed in Section 3.1, whenever the condition (7) is satisfied, the corresponding tile can be safely skipped. This algorithm, which we refer to as PV-Skip, terminates the tile-processing iteration early once this condition is met. In practice, this requires computing the QK product and its row-wise maximum, but allows us to omit both the element-wise exponentiation and the subsequent PV product, as illustrated in Algorithm 1.2 Overall, this approach can reduce the computational cost of a skipped iteration by roughly half, provided that the skipping mechanism is effectively leveraged.

- Algorithm 1 SpargeAttention PV-Skip Zhang et al. (2025a)

Require: Qi, {Kj}, {Vj} while j do

√

Sij ← QiKj⊤/

d mlocal ← rowmax(Sij) mij ← rowmax(mi,j−1,mlocal) if max(mlocal − mij) ≤ −ε then

continue end if Pij ← exp(Sij − mij) Oij ← PijVj

#### ... end while

2The epilogue of the online softmax is omitted since it is not pertinent to our discussion.

We later observed that when the skipping condition was satisfied at timestep t, it tended to remain valid in subsequent timesteps as well. Further investigation revealed that this temporal consistency persisted across timesteps for transformer blocks within the same layer. Within each transformer block, computations are further partitioned by attention head. Interestingly, we also found that the skipped tiles exhibited strong correlations across conditioning batches, suggesting that skip patterns could be inferred across batches at the same timestep. A similar observation was reported by Lv et al.

- (2025).

|do|skip|do| |skip|do|skip|skip|do|do|
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

| | |
|---|---|
| | |

True

|do|skip|skip|skip|do|skip|skip|do|do|
|---|---|---|---|---|---|---|---|---|

Figure 1: A graphical depiction of the Skip-Mask update step in Algorithm 2.

These observations led to the development of LiteAttention’s QK-Skip algorithm. As shown in both Algorithm 2 and Figure 1, the method maintains a Skip-Mask that is updated at each timestep. As the diffusion process progresses, the number of tiles marked for skipping gradually increases. The PV-Skip mechanism can be optionally integrated into this algorithm; however, for a sufficient number of timesteps, its additional benefit becomes marginal.

- Algorithm 2 LiteAttention QK-Skip

Require: Qi, {Kj}, {Vj}, SkipMask while j do if SkipMask(i,j) then

continue end if Sij ← QiKj⊤/

√

d mlocal ← rowmax(Sij) mij ← rowmax(mi,j−1,mlocal) if max(mlocal − mij) ≤ −ε then SkipMask(i,j) ← True continue

end if Pij ← exp(Sij − mij) Oij ← PijVj

#### ... end while

Finally, we empirically observe that LiteAttention exhibits sub-quadratic complexity. This claim is supported by a toy experiment, with results presented in Figure 2. We evaluated both FlashAttention and LiteAttention within a video diffusion model across a varying number of video frames. Assuming FlashAttention scales quadratically, the observed trend suggests that LiteAttention operates with lower effective complexity - otherwise, the skip percentage would remain approximately constant rather than increasing with sequence length.

[Figure 1]

- Figure 2: Toy run of FlashAttention vs. LiteAttention within a video diffusion model for a varying number of video frames. Left: provides the runtimes. Right: provides LiteAttention’s sparsity. If we assume that FlashAttention is of quadratic complexity, then this suggests that LiteAttention is of lower complexity, otherwise we would expect the sparsity percentage to be constant and not increasing.

#### 4.2 From Caching to Skipping (Informal)

Building on prior work on caching schemes, we can draw an informal connection between those observations and our findings. Specifically, we argue that the slow evolution of transformer outputs across timesteps is closely related to the gradual evolution of the transformer’s transition matrix P.

Our analysis is qualitative and omits certain architectural details. In particular, we consider a singleheaded transformer whose output is the self-attention output, ignoring contributions from the MLP layer. Let us denote the transformer output as

yt = ptVt, (15)

where pt is a row of the transition matrix P and yt is an output token. We will argue that small changes in p imply small changes in y and vise-versa.

A direct computation of the output difference gives

∆y = yt − yt−δ (16) = ptVt − pt−δVt−δ (17) = (pt − pt−δ)Vt + pt−δ(Vt − Vt−δ) (18) = ∆pVt + pt−δ∆V. (19)

An upper bound on the output difference follows

∥∆y∥ ≤ ∥∆p∥∥Vt∥ + ∥∆V ∥, (20)

where all norms are Euclidean (Frobenius for matrices). This inequality follows from the triangle and Cauchy-Schwarz inequalities and the fact that ∥pt−δ∥ ≤ 1 since its entries are non-negative and sum to 1.

Conversely, the transition difference satisfies

∆p = (∆y − pt−δ∆V )Vt†, (21)

where Vt† is the Moore-Penrose pseudoinverse of Vt. The transition difference can be bounded as

∥∆y∥ + ∥∆V ∥ σmin(Vt)

, (22)

∥∆p∥ ≤

where σmin(Vt) is the smallest singular value of Vt.

The forward relation (20) suggests that small perturbations in the transition matrix induce small perturbations in the transformer’s output. The reverse relation is less clear unless Vt is well-conditioned, i.e., its smallest singular value is not too small. Nevertheless, our skipping scheme only relies on the slow-evolution assumption to bind the tiles eligible for skipping. Tiles that are not skipped are recomputed from scratch and may undergo large changes.

Finally, we note that an additional advantage of transition-based skipping over transformer output caching is the significantly reduced intermediate memory requirements.

#### 4.3 Skipping Condition

The skipping condition (7) is local in nature, as it evaluates each tile’s individual contribution to the output while disregarding interactions across multiple tiles. In contrast, a global condition would typically assess the entire row of tiles and omit those whose cumulative contribution falls below a specified threshold (for example, removing the weakest tiles such that their cumulative sum does not exceed a given bound). Naturally, the local condition is somewhat more conservative than the global one, as it must account for cases where several tiles might be simultaneously omitted, even though in practice only a few are. Nevertheless, our experiments indicate that the proposed local criterion, when properly calibrated, performs similarly to several global conditions we evaluated.

In addition to the locality of the condition (7), its precision is further limited by the use of the cumulative row maximum mij instead of the true global row maximum. This limitation arises from the causal nature of the online softmax algorithm. Motivated by Li et al. (2025), we explored alternative orderings of the j-loop within the attention kernel. We observed that radial-centric ordering improved performance by reaching the global maximum faster.

#### 4.4 Accumulated-Error Calibration

Diffusion models usually require multiple timesteps (Wan et al., 2025) to progressively denoise an entire data sample. We note that attention errors brought about by sparsity at different timesteps have varying impacts on the final attention output (i.e., at the last timestep): the earlier the timestep, the greater its influence on the final output error. As shown in Table 1, we analyze this effect on the Wan2.1 (Wan et al., 2025) model and find that, under the same attention error magnitude, earlier timesteps contribute more significantly to the final attention error. Based on this observation, we propose assigning different error bounds to different timesteps and searching for the optimal PV-threshold for each timestep.

Specifically, we divide the timesteps evenly into three segments. For these three segments, we set the error bounds to ξ − τ, ξ, and ξ + τ, respectively. Based on these error bounds, we search for the optimal PV-threshold per each timestep. Here, the error is defined as the relative L1 error ηt = |Ots − Ot|/|Ot|, where | · | denotes the matrix L1 norm, Ots represents the output of the sparse attention, and Ot represents the output of full attention, all at timestep t.

- Table 1: Analysis of the final attention error η49 for a fixed PV-threshold at different intermediate timesteps.

#### Timestep 0 16 32 48

η49 0.392 0.375 0.325 0.318

|GEMM1|GEMM0|Softmax + SkipLogic-1|
|---|---|---|

- Warpgroup 1

|GEMM0|Softmax + SkipLogic-1|GEMM1|GEMM0|Softmax + SkipLogic-1|
|---|---|---|---|---|

- Warpgroup 2

|GEMM0|Softmax + SkipLogic-2|SkipLogic-3|GEMM1|GEMM0|Softmax + SkipLogic-2|SkipLogic-3|GEMM1|GEMM0|Softmax + SkipLogic-2|SkipLogic-3|
|---|---|---|---|---|---|---|---|---|---|---|

time

- Figure 3: LiteAttention’s pipeline for the two warpgroup H100 configuration (based on FA3). In SkipLogic-1, a skip bit is computed per each warp in the warpgroup. In SkipLogic-2, a skip bit is again computed per warp and the result is combined with the bitmap of warpgroup 1. In SkipLogic-3 the warp-level skip bitmap is reduced to a single skip bit per the complete tile.

### 5 Implementation

##### 5.1 LiteAttention

LiteAttention is implemented atop FlashAttention3 (FA3) Dao (2024). Rather than reimplementing the kernel from scratch, we extend the FA3 API with an additional parameter, a configurable skip threshold that governs sparsity selection. LiteAttention maintains a persistent Skip-Mask that records and reuses tile-level skip decisions across diffusion timesteps. These lightweight extensions preserve full compatibility with FA3 while introducing evolutionary sparsity, enabling skip patterns to propagate across successive transformer timesteps.

FA3 computes attention in fixed-size tiles along both the query and key/value dimensions. The tile geometry depends on the data type and the available shared-memory capacity. In our setup, LiteAttention targets the NVIDIA H100 (Hopper) GPU under CUDA 12.8, adopting the same configuration as the official FA3 Hopper implementation. Specifically, for BF16, each tile spans 128 × 176 elements per head for a head dimension of 128, maximizing occupancy under Hopper’s 228KB per-SM shared-memory limit. Each CUDA thread block comprises three warpgroups: one producer and two consumers operating in a pipelined fashion. The producer asynchronously streams Kj and Vj tiles from global to shared memory, while the consumer warpgroups process distinct query tiles Qi in parallel, performing the matrix multiplication QiKj⊤/

√

d, executing the online softmax reduction for numerical stability, and multiplying by Vj to accumulate partial outputs. This producer–consumer pipeline achieves near-complete overlap between memory transfers and computation. LiteAttention’s sparsity mechanism operates at the same tile granularity as FA3, allowing its skip logic to integrate seamlessly into the existing producer–consumer pipeline with minimal kernel modifications. The key additions occur within both the consumer and producer warpgroups: the consumer warpgroups evaluate the skip condition concurrently with the online softmax computation, while the producer warpgroup consults the Skip-List to stream only the relevant Kj and Vj tiles.

#### 5.2 Skip-Mask Evaluation Mechanism

For each QK tile, LiteAttention evaluates the skip condition (7), which determines whether the tile’s contribution to the output is negligible. Each per-tile decision is written to a global SkipMask data structure that persists across denoising timesteps, as depicted in Figure 4. Although this predicate could, in principle, be used to immediately bypass the subsequent PV computation, doing so would introduce additional latency on Hopper due to synchronization dependencies within the consumer warpgroups. On the H100, each matrix multiply–accumulate (MMA) operation is executed by a warpgroup of four warps, and enforcing cross-warp agreement on the skip predicate before continuation would stall otherwise overlapping execution. To avoid this, LiteAttention records partial skip results independently for each of the four warps. Intra-warp reductions are implemented efficiently using warp-synchronization primitives, while cross-warp reductions, which require barriers, are deferred to the kernel epilogue, executed once by the final warpgroup after completing all softmax operations for the current tile. Figure 3 depicts LiteAttention’s pipeline, constructed on top of FA3 (see Shah et al. (2024) - Figure 1).

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

(a) Block 15 - Head 1 (b) Block 30 - Head 2

- Figure 4: The evolving Skip-Mask across diffusion timesteps for LTX-13B HaCohen et al. (2024) over two block/head sets. The top and bottom are the start and end masks, respectively. Dark purple means skipped.

- Table 2: Comparison of LiteAttention’s video quality (VBench), attention sparsity (Sps), and runtime (Run) compared with FlashAttention3 (FA3), SparseVideoGen, and RadialAttention over Wan2.1-14B and Wan2.2-14B. Best results are in bold and second best in italic.

Method AQ↑ BC↑ DD↑ IQ↑ SC↑ TF↑ TS↑ Sps[%]↑ Run[sec]↓

- Wan2.1-14B FA3 0.676 0.977 0.417 68.74 0.965 0.962 0.137 0 1707

SVG 0.665 0.971 0.500 68.58 0.962 0.959 0.066 66 1019 Radial 0.660 0.970 0.417 64.73 0.964 0.972 0.061 74 1192 Lite 0.677 0.975 0.500 66.76 0.963 0.962 0.142 42 902

- Wan2.2-14B FA3 0.693 0.977 0.583 72.73 0.970 0.953 0.133 0 1473

SVG 0.689 0.962 0.417 72.24 0.961 0.952 0.061 66 1022 Radial 0.682 0.974 0.500 72.73 0.967 0.947 0.061 66 1207 Lite 0.698 0.977 0.500 71.44 0.969 0.953 0.135 32 893

- Table 3: Ablation study of LiteAttention’s self-attention runtime (SR), its relative improvement (dSR), and video quality (VBench) over different levels of induced sparsity (Sps).

#### Sps[%] SR[sec]↑ dSR[%]↑ AQ↑ BC↑ DD↑ IQ↑ SC↑ TF↑ TS↑

0 695 0 0.702 0.978 0 77.100 0.994 0.978 0.092 21 573 18 0.692 0.977 0 76.860 0.993 0.978 0.098 42 418 40 0.690 0.964 0 76.086 0.987 0.978 0.096 57 308 56 0.672 0.962 0 76.393 0.969 0.976 0.094 77 163 77 0.630 0.964 0 77.061 0.962 0.978 0.075

#### 5.3 Producer and Skip-List Optimization

The resulting per-tile skip flags are consumed by the producer warpgroup in the subsequent diffusion timestep. Because the producer is lightweight and inherently synchronized to wait for the consumer’s completion, it can query the skip decision without impacting throughput. If a tile is marked as skippable, the producer omits loading the corresponding Kj and Vj tiles from global memory, allowing the consumer to bypass all computation for the entire j’th iteration. Initially, the skip mask was implemented as a simple bitmask, allocating one bit per Sij tile. However, as sparsity increased, we observed that a compressed representation offered superior efficiency. We therefore adopted a Skip-List structure inspired by run-length encoding, where consecutive non-skipped ranges are represented as (start, end) pairs. This compact format enables the producer to skip entire contiguous sequences of K tiles with a single conditional check, improving both memory efficiency and kernel throughput under high sparsity conditions.

### 6 Experiments

#### 6.1 Setup

Models, Dataset, and Baselines. We evaluate LiteAttention using the 12 prompts dataset from OpenSora1.0 Zheng et al. (2024). For video generation models, we consider Wan2.1-14B and Wan2.214B (Wan et al., 2025). FlashAttention3 (FA3) (Shah et al., 2024), SparseVideoGen (SVG) Xi et al. (2025), and RadialAttention (Radial) Li et al. (2025) comparison baseline.

Metrics. Generated video quality is evaluated using VBench Huang et al. (2024) across the metrics Aesthetic Quality (AQ), Background Consistency (BC), Dynamic Degree (DD), Imaging Quality (IQ), Subject Consistency (SC), Temporal Flickering (TF), and Temporal Style (TS). All values are averaged over the dataset. For LiteAttention, sparsity (Sps) denotes the fraction of computations

skipped relative to full attention, averaged over the generation process. For all other methods, we report the sparsity values they report.

Settings. Throughout our experiments, LiteAttention was applied exclusively to accelerate the self-attention primitive and was run using the standard (suboptimal) linear ordering for the j-loop (see Section 4.3).

- The results in Table 2 were obtained using calibrated PV-thresholds with τ = 0.01 and ξ = 0.075.
- The results in Table 3 were generated for Wan2.1 without calibration. Instead, the PV-threshold was set to −8 for the first 20 timesteps, and a grid search was used to select thresholds for the last 30 timesteps to achieve the desired sparsity level. Experiments were conducted on NVIDIA’s H200 GPU.

#### 6.2 Results

Effectiveness. Table 3 examines the impact of increasing sparsity on video quality. At 77% sparsity, we observed visible distortion in the generated video, which is empirically reflected in the TS metric. Comparing this metric in Table 2 indicates that LiteAttention preserves visual quality comparable to full attention (FA3), while SVG and Radial show marked degradation.

Efficiency. Although SVG and Radial report higher nominal sparsity in Table 2, LiteAttention achieves at least 10% greater runtime improvement. Combined with their stronger quality drop, this highlights LiteAttention’s superior trade-off between efficiency and fidelity. We expect an additional 10–20% gain in sparsity without quality loss once optimized j-loop ordering is applied. Notably, Table 3 shows that runtime reduction scales nearly one-to-one with skipped computation.

Ablation Study. Table 3 further examines quality and runtime under varying sparsity levels using an uncalibrated setup, where the PV-threshold was increased over the last 30 of 50 timesteps. Video quality degrades sharply beyond 70% sparsity. Comparing the 42% entry in Table 3 with the corresponding entry in Table 2 for Wan2.1-14B shows that calibrated runs achieve substantially higher quality at the same sparsity. This suggests that with proper calibration and improved j-loop ordering, sparsity, and thus runtime, can reach around 70% without visible quality loss.

### 7 Conclusions

We presented LiteAttention, a method that exploits the temporal coherence of sparsity in diffusion transformer attention to accelerate video generation. By identifying and propagating skippable tiles across timesteps, LiteAttention combines the adaptivity of dynamic sparsity with the efficiency of static approaches, achieving substantial runtime reductions without compromising video fidelity. The method integrates seamlessly into existing CUDA-accelerated attention kernels, requires no model retraining, and demonstrates that evolutionary computation skips can unlock practical, highperformance DiT inference at scale.

### References

Jiazi Bu, Pengyang Ling, Yujie Zhou, Yibin Wang, Yuhang Zang, Dahua Lin, and Jiaqi Wang. Dicache: Let diffusion model determine its own cache, 2025.

Pengtao Chen, Xianfang Zeng, Maosen Zhao, Peng Ye, Mingzhu Shen, Wei Cheng, Gang Yu, and Tao Chen. Sparse-vdit: Unleashing the power of sparse attention to accelerate video diffusion transformers. In European Conference on Computer Vision (ECCV), 2024a.

Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. Gentron: Diffusion transformers for image and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6441–6451, 2024b.

Huanpeng Chu, Wei Wu, Guanyu Fen, and Yutao Zhang. Omnicache: A trajectory-oriented global perspective on training-free cache reuse for diffusion transformer models, 2025.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, 2024.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

Chieh-Hsin Lai, Yang Song, Dongjun Kim, Yuki Mitsufuji, and Stefano Ermon. The principles of diffusion models, 2025.

Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: O(n log n) sparse attention with energy decay for long video generation. In Conference on Neural Information Processing Systems (NeurIPS), 2025.

Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model, 2025a.

Haozhe Liu, Wentian Zhang, Jinheng Xie, Francesco Faccio, Mengmeng Xu, Tao Xiang, Mike Zheng Shou, Juan-Manuel Perez-Rua, and Jürgen Schmidhuber. Faster diffusion via temporal attention decomposition. Transactions on Machine Learning Research, 2025b.

Zhengyao Lv, Chenyang Si, Junhao Song, Zhenyu Yang, Yu Qiao, Ziwei Liu, and Kwan-Yee K. Wong. Fastercache: Training-free video diffusion model acceleration with high quality, 2025.

Xinyin Ma, Gongfan Fang, Michael Bi Mi, and Xinchao Wang. Learning-to-cache: Accelerating diffusion transformer via layer caching. In Conference on Neural Information Processing Systems (NeurIPS), 2024a.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024b.

Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. arXiv preprint arXiv:1805.02867, 2018.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37:68658–68685, 2024.

Shengkun Tang et al. Adadiff: Adaptive timestep scheduling for diffusion models. In European Conference on Computer Vision (ECCV), 2024.

Ashish Vaswani, Noam Shazeer, Nir Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Conference on Neural Information Processing Systems (NeurIPS), 2017.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Zeqing Wang, Gongfan Fang, Xinyin Ma, Xingyi Yang, and Xinchao Wang. Sparsed: Sparse attention for diffusion language models. arXiv preprint arXiv:2509.24014, 2025.

Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. In International Conference on Machine Learning (ICML), 2025.

Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. arXiv preprint arXiv:2505.18875, 2025.

Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattn: Accurate sparse attention accelerating any model inference. In International Conference on Machine Learning (ICML), 2025a.

Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhenghong Liu, and Hao Zhang. Fast video generation with sliding tile attention. In International Conference on Machine Learning (ICML), 2025b.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. URL https://github. com/hpcaitech/Open-Sora, 2024.

Yajing Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with token-wise feature caching. In International Conference on Learning Representations (ICLR), 2025.

