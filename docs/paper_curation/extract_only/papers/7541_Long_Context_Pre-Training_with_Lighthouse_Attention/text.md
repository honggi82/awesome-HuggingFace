# arXiv:2605.06554v1[cs.CL]7May2026

## Long Context Pre-Training with Lighthouse Attention

Bowen Peng* Nous Research bloc@nousresearch.com

Subho Ghosh* Nous Research subho@nousresearch.com

Jeffrey Quesnelle Nous Research emozilla@nousresearch.com

### Abstract

Training causal transformers at extreme sequence lengths is bottlenecked by the quadratic time and memory of scaled dot-product attention (SDPA). In this work, we propose Lighthouse Attention, a training-only symmetrical selection-based hierarchical attention algorithm that wraps around ordinary SDPA and can be easily removed towards the end of the training. Our hierarchical selection is also gradient-free, which exempts us from dealing with a complicated and potentially inefficient backward pass kernel. Our contribution is three-fold: (i) A subquadratic hierarchical pre- and post-processing step that does adaptive compression and decompression of the sequence. (ii) A symmetrical compression strategy that pools queries, keys and values at the same time, while preserving left-to-right causality, which greatly improves parallelism. (iii) A two stage training approach which we pre-train for the majority of the time with Lighthouse Attention and recover a full attention model at the end with a short training. We run preliminary small scale LLM pre-training experiments that show the effectiveness of our method compared to full attention training with all other settings matched, where we achieve a faster total training time and lower final loss after the recovery phase.

Full code is available at: https://github.com/ighoshsubho/lighthouse-attention.

### 1 Introduction

The frontier of language modeling has moved toward contexts of 128K, 1M, and longer, pushed by agentic multi-step reasoning, long-document understanding, and interleaved multimodal inputs [25, 1, 11, 22, 27, 8, 23]. Training at these scales is the dominant hardware bottleneck: scaled dot-product attention has Θ(N2) compute and memory, a wall that FlashAttention [29] pushes back but does not remove.

A growing body of work replaces dense attention with selection: each query attends only to a small subset of keys. Block-level methods such as MoBA [20] and Native Sparse Attention [36] select contiguous blocks, while token-level methods such as DeepSeek Sparse Attention (DSA; 9) score every past token via a learned indexer and forward the top-k into a sparse attention operator; HISA [40] adds a hierarchical indexer to keep scoring from becoming the new bottleneck. These methods produce meaningful inference speedups but inherit two design decisions that fit long-context pretraining poorly. (i) Asymmetry: queries stay at full resolution while keys and values are pooled, so the hierarchy serves only as a compressed addressable memory rather than a multi-scale representation. (ii) Architectural entanglement: selection lives inside the attention kernel, so the carefully optimized dense-attention kernels that modern tensor-core GPUs accelerate cannot be reused; every sparse method ships its own kernel.

There is also a concern specific to training. An inference-time sparse method [40, 28, 31, 38, 32] is by construction as good as its dense backbone, since the sparse substitution is evaluated only against

*Equal contribution.

Preprint.

the dense forward. A training-time sparse method must survive a harder test: once training is done, will the resulting model still be a competent dense-attention model?

We take this last question as our central correctness criterion. We introduce Lighthouse Attention: a selection-based hierarchical attention that pools Q,K,V symmetrically across a multi-level pyramid, scores every pyramid entry bidirectionally with a parameter-free scorer, and selects the top-K entries with a fused chunked-bitonic kernel. The selected entries form a dense, causally consistent sub-sequence attended to with stock FlashAttention; outputs are scattered back through a deterministic kernel. The top-K step is non-differentiable, with no straight-through estimator: gradients flow through scatter, FlashAttention, and gather into WQ,WK,WV , which learn to produce values that are useful when selected. No auxiliary parameters or losses are added. Two consequences follow: the symmetric pyramid is a full multi-scale representation rather than a compressed context, and because selection sits outside the attention path, the expensive step is stock FlashAttention on a sub-sequence of size O(LpK + N/pL−1), which reduces to O(N log N) at L = logp(N/K).

Our central empirical finding addresses the training-correctness concern directly: after a brief denseSDPA resumption, Lighthouse-trained models match or beat a fully dense-SDPA baseline trained from scratch on the same token budget. The hierarchical training signal does not hollow out the model’s ability to use full attention at inference, a property inference-only sparse methods cannot claim because they never touch the training loop. We summarize our contributions:

- • A selection-based hierarchical attention designed for long-context pretraining with symmetric Q/K/V pooling, bidirectional top-K selection, and stock FlashAttention on the gathered sub-sequence, keeping sparse logic entirely outside the attention kernel.
- • Fused GPU kernels (chunked-bitonic top-K and a custom scatter-back) that make this design fast at very large contexts.
- • The strongest empirical criterion for a training-time hierarchical method to our knowledge: dense-SDPA resumption after Lighthouse pretraining matches a dense-from-scratch baseline on training loss.

### 2 Related works

Compression and pruning. A first response to quadratic attention abandons softmax for a bounded-size state: linear attention [? 4], state-space and gated variants [12, 6, 34, 30], and loglinear attention [13]: which gives strong asymptotics but compresses the entire past and limits long-range recall [2]. A second keeps softmax and prunes at block granularity, either trainingfree (MInference, FlexPrefill, XAttention, SpargeAttention [15, 16, 33, 37]) or end-to-end (MoBA, NSA [20, 36]); these map cleanly onto tiled matmul but force a single retain/discard decision per block and pool only the key–value side. A third prunes at token granularity, mostly at inference for KV-cache eviction (H2O, TOVA, SnapKV, LazyLLM, Quest, SparQ [39, 26, 17, 10, 31, 28]), or via a learned indexer trained end-to-end (DSA [9]). The defining property of this family is that once selection is identified it is welded into the attention operator as a custom sparse matmul or per-query gather, foreclosing reuse of stock dense kernels.

Hierarchies and training-time correctness. Multi-resolution attention [35] has returned to sparse LLM attention in two flavors. NSA [36], InfLLM-V2 [41], Twilight [18], and DoubleP [24] build hierarchies that the attention itself reads from compression branches, centroid summaries, or quantized proxies. HISA [40] is a training-free, plug-in replacement for DSA’s indexer that runs a block-totoken two-stage score and forwards the selected tokens unchanged to the same Sparse MLA operator DSA already uses. In every case the hierarchy applies only to keys and values, and the selection that emerges still feeds a custom sparse attention kernel. Lighthouse differs on three axes: it pools queries symmetrically with keys and values into coherent multi-resolution (Q(ℓ),K(ℓ),V (ℓ)) triples; the pyramid is used purely to rank and select, so the attention that follows is stock FlashAttention on a dense sub-sequence with no sparse indexing inside the kernel; and it is trained end-to-end through a non-differentiable top-k wrapped by a differentiable scatter, with no auxiliary loss or straightthrough estimator. Inference-only sparse methods (including HISA) inherit a correctness floor from their underlying dense model, but training-time sparse methods (MoBA, NSA) must answer whether the weights they produce remain competent dense models. We take a brief dense-SDPA resumption recovering the quality of a dense-from-scratch baseline as our central correctness criterion.

###### Hierarchical Selector

Score Top-K

I

Q, K, V O

Projections

Dense Gather

Pyramid Pool

Scatterback

SDPA

Ht

Ot

WQ, WK, WV

- Figure 1: Lighthouse Attention architecture. Forward (black): Ht is projected to Q,K,V , passed through the symmetric Pyramid Pool on the trunk, and guided by indices I from the Hierarchical Selector, is fed to a dense gather which topographically sorts the gathered hierarchies into a sin-

gle contiguous and causal sequence, then stock SDPA, and scatter-back to produce Ot. Selection (green): the selector taps the pooled summaries off the trunk; the parameter-free scorer ranks them

by ℓ2 norm and a top-K kernel keeps the largest entries, emitting integer indices I that merge back into the trunk at the dense gather. Gradient (red, dashed): ∇L travels along the trunk (Ot →scatter →FA→gather→pyramid pool→ WQ,K,V → Ht); the selector branch is non-differentiable and is bypassed.

### 3 Method

We present Lighthouse Attention, a selection-based hierarchical attention mechanism for longcontext pretraining. Lighthouse replaces a standard Transformer attention layer with a four-stage pipeline that surrounds, but does not modify, the attention kernel: a pre-attention selection stage drives a contiguous gather, stock FlashAttention [5] runs on the gathered sub-sequence, and a postattention scatter writes the result back to the original positions. Selection is driven by a parameterfree scoring functional over a multi-resolution pyramid of the layer’s own queries, keys, and values, so Lighthouse introduces no new learnable parameters beyond those of the underlying attention block.

#### 3.1 Preliminaries

Let X ∈ RN×d

model be the input, WQ,WK,WV ∈ Rd

model×d projection matrices for one head, and

- M ∈ RN×N a causal mask. Standard scaled dot-product attention [7] is

Q = XWQ, K = XWK, V = XWV , Attn(Q,K,V ) = softmax QK

⊤

√

d + M V, (1) with both time and memory cost Θ(N2d). FlashAttention reduces constants but not asymptotics; at

- N ≥ 105 this term dominates. Lighthouse replaces Eq. (1) with: (i) symmetric average-pooling of Q,K,V into an L-level pyramid (factor p); (ii) parameter-free scoring and a fused chunked-bitonic top-k selection over all levels jointly; (iii) stock FlashAttention on a contiguous sub-sequence of S ≪ N selected entries; (iv) a scatter-back that distributes each output to the pℓ base positions it represents. Stages (ii) and (iv) are custom kernels (Sec. 5); stage (iii) is the same FlashAttention call as the dense baseline. The top-k is treated as discrete and non-differentiable: indices carry

no gradient and the scoring functional is not trained. Gradients reach WQ,WK,WV only through stages (iv), (iii), and the gather: the projections learn to produce values that are useful when selected rather than scores that are good at selecting, sidestepping the optimization fragility of learnable selectors.

#### 3.2 Overview

A Lighthouse attention layer replaces standard scaled dot-product attention (Eq. (1)) with a fourstage pipeline that surrounds, but does not modify, the attention kernel. Let Q,K,V ∈ RN×d be the per-head projections from the layer’s own WQ,WK,WV (Sec. 3.1).

- (i) Pyramid. Average-pool Q,K,V symmetrically into an L-level pyramid with pooling factor p, producing coherent triples (Q(ℓ),K(ℓ),V (ℓ)) for ℓ = 0,...,L − 1.

###### Hierarchical Selector

|pyramid tokens| |
|---|---|
| | |

| | |
|---|---|
| | |

###### 1. Pyramid Pool

###### 2. Norm Score

###### 3. Chunked Bitonic Top-K

ℓ=2

- ℓ=0
- ℓ=1
- ℓ=2

ℓ=2

scoresqk scoreskq

ℓ=1

I

###### Q, K, V

ℓ=1

[B, H, K]

[B, S, H, D]

ℓ=0

ℓ=0

parent kept bitonic argsort + tree-prune

mean-pool by pℓ

ℓ2 norm + max-pool

- Figure 2: Pyramid Pool and the Hierarchical Selector. The Pyramid Pool is a fixed pre-selection stage that lives outside the selector. (1) Pyramid Pool mean-pools Q,K,V by pℓ; lines show which tokens feed each summary. The pooled tokens enter the selector, where (2) Norm Score

computes parameter-free ℓ2 norms ∥Q(ℓ)∥2,∥K(ℓ)∥2 (coarser levels reuse finer norms via maxpool) and (3) Chunked Bitonic Top-K keeps top-K parents (dark) that descend, while rejectedkept entries (light) emit to I without descending. Pruning is implicit: children of non-parents are never expanded. Per-tile bitonic argsort runs in-register; uint64 packed keys let a single torch.sort produce I.

- (ii) Score and top-k. Assign each pyramid entry parameter-free query and key scores and select the k entries with the highest combined relevance across all levels via a fused chunked-bitonic top-k kernel.
- (iii) Dense sub-sequence attention. Gather the selected triples into a contiguous sub-sequence of length S and compute softmax attention over it with stock FlashAttention.
- (iv) Scatter-back. Distribute each entry’s output to the pℓ base positions it represents via a deterministic integer-atomic scatter kernel.

Stages (ii) and (iv) are custom kernel (Sec. 5); stage (iii) is the same FlashAttention call used by the dense baseline. Lighthouse adds no learnable parameters or losses: the pyramid is a fixed pooling, the scorer is parameter-free, and gather/scatter are data-flow primitives. Gradients flow from the loss through stages (iv) and (iii) into the gathered Q,K,V and on into WQ,WK,WV ; the top-k step is discrete and non-differentiable, so its indices carry no gradient and we use no straight-through estimator. The projections therefore learn to be useful when selected, not to score well at selecting.

#### 3.3 Pyramid Construction

Given Q,K,V ∈ RN×d, Lighthouse Attention constructs an L-level pyramid whose ℓ-th level is a non-overlapping window pooling of the previous level. For ℓ = 0,...,L−1, define the i-th window at level ℓ as

Wi(ℓ) = ipℓ, (i + 1)pℓ − 1 , i = 0,..., pN

ℓ − 1, (2)

where p is the pooling factor. The pyramid entries are then Q(iℓ) = Poolµ Qj | j ∈ Wi(ℓ) , Ki(ℓ) = Poolµ Kj | j ∈ Wi(ℓ) , Vi(ℓ) = Poolµ Vj | j ∈ Wi(ℓ) ,

(3) with Poolµ denoting mean pooling over the window. Level 0 is the original full-resolution sequence (Wi(0) = {i}), and each subsequent level summarizes p consecutive entries of the level below. We require pL−1 | N. Unlike prior hierarchical sparse designs (NSA, HISA, InfLLM-V2), which pool only the context side, Lighthouse applies Poolµ symmetrically to all three projections. Symmetry buys two properties used in subsequent stages: a pooled query Q(iℓ) and a pooled key Kj(ℓ) live in the same representation space, and each pyramid entry is a coherent (Q,K,V ) triple summarizing the same pℓ-token span. The total number of pyramid entries is Lℓ=0−1 N/pℓ ≤ N · p/(p − 1), so pyramid construction costs Θ(N) time and memory.

#### 3.4 Scoring and Selection

Each pyramid entry receives two scalar scores — one as a query, one as a key. At level 0 we use per-head ℓ2 norms,

sQK0,i = ∥Qi∥2, sKQ0,i = ∥Ki∥2, i = 0,...,N − 1, (4)

and at coarser levels we max-pool from level 0 rather than recomputing from pooled projections,

ℓ+j, sKQℓ,i = max

sKQ0, ip

sQKℓ,i = max

sQK0, ip

ℓ+j. (5)

0≤j<pℓ

0≤j<pℓ

Max-pooling lets a coarse span inherit the importance of its strongest token. Selection runs jointly over the concatenated QK and KQ streams across all levels via the chunked-bitonic kernel of Sec. D.2:

I = TopK sQKℓ,i , sKQℓ,i : (ℓ,i) ∈ P , k , (6) where P is the full set of pyramid indices. An entry chosen via its KQ score still enters the gather

- as its own (Q(iℓ),Ki(ℓ),Vi(ℓ)) triple. The coarsest level is always retained in full — it is cheap and guarantees at least one contributor at every base position; the remaining budget is spent on finer levels.

- 3.5 Gathered-Sequence Attention Given I, Lighthouse assembles a contiguous sub-sequence

Qm = Q(iℓm)

m

, Km = Ki(ℓm)

m

, Vm = Vi(ℓm)

m

, (ℓm,im) ∈ I, m = 1,...,S, (7) of length

S =

N pL−1 + (L − 1)pk, (8)

because the coarsest level contributes all N/pL−1 entries while each of the remaining L − 1 levels contributes at most pk (the factor of p is causal-boundary bookkeeping; Sec. D.2). At N = 106,L = 4,p = 4,k = 4096, S ≈ 6.5 × 104 ≪ N. The sub-sequence is then attended to via stock SDPA or FlashAttention,

O = Attn( Q, K, V ; M), (9)

where Attn(·) is standard masked softmax attention. The causal mask M derives from the pyramid coordinates (ℓm,im) so each entry attends only to entries whose base positions are no greater than its own; the gather is topologically sorted, so M reduces to a standard S×S causal mask and Eq. (9) contains no sparse indexing.

Due to the hierarchical decomposition, this gathering process guaranties that there are no ”holes” or empty spaces in the sequence, which is especially important as we also compress queries Q; a hole could cause training instabilities as those missing tokens would be cut out during the forward pass and have no gradients during the backward pass. This is unlike asymmetrical methods that do not compress queries.

- 3.6 Scatter-Back Reconstruction

The attention output O ∈ RS×d is redistributed to the full N-token output O. A selected entry at level ℓ, position i summarized window Wi(ℓ) during pooling but its output is written to a shifted range

##### R(ℓ,i) = ipℓ + pℓ − 1, ipℓ + 2pℓ − 2 , (10)

that starts at the last summarized token. The shift of pℓ − 1 preserves causality: a base position j never receives a summary that contains its own future. Within a level, consecutive windows write to disjoint adjacent ranges; across levels, contributions are summed,

Om, (11)

Oj =

m : j∈R(ℓm,im)

so the per-position fan-in is bounded by L regardless of k. Similarly to the gathering pass, the scattering process also has no empty spaces. This final scattered sequence is fully dense, albeit a compressive approximation of full attention.

Algorithm 1: Lighthouse attention (single layer, single head). Input: X ∈ RN×d

model, projections WQ,WK,WV , pyramid params (L,p), budget k Output: O ∈ RN×d

- 1: Q,K,V ← XWQ, XWK, XWV // projections (fused GEMM)
- 2: (Q(ℓ),K(ℓ),V (ℓ)) Lℓ=0−1 ← Poolµ(Q,K,V ) // pyramid (view+mean)
- 3: sQKℓ,i ← Poolmax ∥Qj∥2 , sKQℓ,i ← Poolmax ∥Kj∥2 // scoring (norm+max)
- 4: I ← CHUNKEDBITONICTOPK(sQK,sKQ,k) // custom kernel (§D.2)
- 5: Q, K, V ← Gather Q(·),K(·),V (·); I // torch.gather
- 6: O ← FlashAttention( Q, K, V ) // stock FA-3/FA-4
- 7: O ← SCATTERBACK( O,I) // custom kernel)
- 8: return O

### 4 Design Choices

The Lighthouse pipeline of Sec. 3 makes four design choices that distinguish it from prior selectionbased sparse attention. First, Q is pooled in lockstep with K,V instead of leaving queries dense

- as in NSA [36], HISA [40], and InfLLM-v2 [41]; this is the choice that turns the dense kernel call from O(NSd) to O(S2d) at training time, and keeps pooled queries and pooled keys in the same representation space at every level. Second, the scorer is parameter-free per-head ℓ2 norms of the layer’s own Q(ℓ),K(ℓ) — rather than a learned scoring head as in NSA [36] or DSA [9]; this is the cheaper option and is strictly weaker than any attention- or QK-interaction-based scorer, so any positive result is a lower bound on what richer scorers can extract. The natural QK-interaction alternative we ablate against is a dilated softmax-attention scorer that runs softmax attention over the pyramid with dilation factor δ at O(N2d/δ) per layer sub-quadratic but still super-linear in N, and an order of magnitude more expensive than the projection-norm scorer at long context (Sec. 6.4).

Third, selection is decoupled from attention: top-K produces a contiguous, dense sub-sequence and attention is a stock SDPA or FlashAttention [7, 5] call on it, with no custom sparse-attention kernel coupling the two steps as in NSA [36], DSA [9], or HISA [40]. The same kernel runs at training and inference, and disabling selection cleanly recovers the dense baseline exactly the SDPA-resume test in Sec. 6.2. Fourth, we do not make the top-K differentiable: no straight-through estimator, no Gumbel softmax, no auxiliary scorer loss. Gradients flow only through the gathered Q, K, V into WQ,WK,WV , so the projections learn to be useful when selected rather than to game a learnable scorer. We motivate each choice and discuss alternatives in Appendix C.

### 5 Complexity Analysis and Kernel Design

Algorithm 1 summarizes one Lighthouse attention layer as a sequence of GPU primitives. Most stages are standard operations executed via torch.compile’d PyTorch code; only the top-k selection (stage 2c) and the scatter-back (stage 5) are custom kernels.

#### 5.1 Asymptotic Complexity

Table 3 decomposes per-layer cost by stage. The only super-linear term in N is the dense sub-sequence attention, Θ(S2d), with S = N/pL−1 + (L − 1)pk from Sec. 3.5. Choosing L = logp(N/k) balances the two terms in S, giving S = Θ(k logp(N/k)) and an attention cost of Θ(k2 log2 N · d) — polylogarithmic in N at fixed k. Combined with the linear scoring and Θ(N log k) selection passes, total per-layer compute is linear in N up to a log k factor for bounded k. App. B derives this and compares against dense softmax, log-linear attention, and linear/SSM families.

#### 5.2 Kernel Design and Parallelism

Of the seven stages in Algorithm 1, only top-K and scatter-back are custom kernels in CUDA and triton; the rest reduce to PyTorch primitives that torch.compile fuses into single device passes. Our chunked-bitonic top-K partitions the score stream, maintains an in-register top-m buffer per chunk via bitonic merge, and dispatches chunks as independent CTAs avoiding the shared-memory blow-up of textbook bitonic at k=4096 while producing a stratified selection that resists span collapse. Crucially, gather is decoupled from attention: where NSA [36], DSA [9], HISA [40], and MoBA [20] embed selection inside a custom sparse kernel, Lighthouse hands a contiguous dense sub-sequence to stock FlashAttention [29] — making forward/backward bit-for-bit identical to a dense Transformer’s, letting context parallelism rotate the gather through standard ring attention [19] without any sparsity-aware collective, and enabling 1M-token training across 32 Blackwell GPUs (full details in App. D).

### 6 Experiments

We evaluate Lighthouse along three axes: (1) recoverability: whether lighthouse pretraining damages the model’s ability to use full attention at inference Sec. 6.2); (2) design ablations and throughput over the four knobs (scorer, p, L, k) and the resulting wall-clock cost (Sec. 6.4); and (3) scaling vs. dense attention as a function of context length, including the long-context regime that requires context parallelism (Sec. 6.3). All runs share the architecture and recipe of Sec. 6.1.

#### 6.1 Experimental Setup

Architecture, data, optimizer. A 530M-parameter Llama-3-style decoder (dmodel=1024, 30 layers, H=8, head dim 128, FFN 1536, byte-level tokenizer). Layers {0,1,28,29} retain dense SDPA — PyTorch 2.11.0+cu128’s torch.nn.attention.sdpa kernel routed to cuDNN 9.19.0 on CUDA 12.8; the other 26 use Lighthouse with the same cuDNN-SDPA kernel as the inner attention call on the gathered sub-sequence. Training on C4 at sequence length 98,304, global batch 32, AdamW 2×10−3, β1=0.9,β2=0.95, weight decay 0.1, linear warmup over 2k steps, gradient-norm clip 1, bfloat16, FSDP only.

Two-stage recipe. Stage 1 trains with Lighthouse; stage 2 resumes the stage-1 checkpoint under dense SDPA (same cuDNN backend), with the same optimizer state and dataloader continuation. The total budget is held at 16,000 steps (≈ 50 B tokens); we vary the stage-1 length to test sensitivity to the switch point.

Hardware. A single NVIDIA BGX 8×B200 node is used for 98K-context runs; multi-node configurations are used with intra-node CP for 256K (Table 1). We report training and validation loss, tokens/s per GPU in steady state, and total B200 hours.

#### 6.2 SDPA Recoverability

We test whether a hierarchical-trained Lighthouse model can be restored to dense attention by a brief continuation under stock SDPA. Holding the budget at 16,000 steps (≈ 50B tokens), we vary the stage-1 length (10k / 11k / 12k) and resume the remainder under dense SDPA, against an full SDPA reference at matched architecture, data, and tokens (Table 1, top block). At each resume the training loss transiently spikes (1.12–1.57) as the model is first run through attention it was not trained against, then recovers within ≈ 1–1.5k SDPA steps and crosses below the dense baseline; by step 16,000 all three resume schedules match or beat dense-from-scratch (0.6980–0.7102 vs. 0.7237), with longer dense-resume tails giving lower final loss. Recovery is robust across resume points (the recipe doesn’t pivot on a precise schedule), supporting our load-bearing claim that hierarchical training does not compromise the model’s ability to use full attention at inference, at no additional token cost over dense-from-scratch.

#### 6.3 Scaling Laws vs. Dense Attention

We benchmark single-layer attention latency on a single B200 for contexts from 8K to 512K (bf16, B=1, H=8, d=128, L=3, p=4, sparsity 1:64, medians of 10 steady-state iterations), comparing

###### Configuration Scorer Params LH Total Total B200-Hrs Tok/s (k) Final Loss

Steps Steps Tokens (↓) (↑) (↓) SDPA Baseline (ctx = 98k) — 530M — 16k 50.3B 303.2 45.6 0.7237 SDPA recoverability (L=3, p=2, k=6144, ctx = 98k) LH → SDPA (12k+4k) Dilated 530M 12k 16k 50.3B 214.7 74.7 0.7102

- LH → SDPA (11k+5k) Dilated 530M 11k 16k 50.3B 219.6 75.4 0.7001

- LH → SDPA (10k+6k) Dilated 530M 10k 16k 50.3B 228.0 75.0 0.6980

Hyperparameter ablations (ctx = 98k)

L=3, p=2, k=1536 Dilated 530M 10k 16k 50.3B 203.9 93.9 0.6825 L=3, p=4, k=1536 Dilated 530M 10k 16k 50.3B 197.2 99.5 0.6881

- L=3, p=8, k=1536 Dilated 530M 10k 16k 50.3B 206.2 92.1 0.6828

- L=4, p=2, k=1536 Dilated 530M 10k 16k 50.3B 200.2 96.4 0.6978

- L=5, p=2, k=1536 Dilated 530M 10k 16k 50.3B 201.5 96.3 0.6991

L=3, p=2, k=2048 Dilated 530M 10k 16k 50.3B 208.1 90.9 0.6880 L=3, p=2, k=4096 Dilated 530M 10k 16k 50.3B 215.7 83.5 0.6951

CP training (L=3, p=4)

k=1536, ctx = 98k, CP= 2, DP= 4 Norm 530M 10k 16k 100.7B 208.3 91.8 0.6903 k=2048, ctx = 98k, CP= 2, DP= 4 Norm 530M 10k 16k 100.7B 210.9 89.2 0.6928 k=4096, ctx = 256k, CP= 8, DP= 1 Norm 530M 10k 16k 1.07T 1300.3 48.9 0.6721

- Table 1: Lighthouse ablation summary. 530M Llama-3, 16k optimizer steps; LH Steps run Lighthouse, SDPA Steps run dense SDPA-resume on a single 8×B200 node (the final block adds context parallelism). B200-Hrs: combined wall-clock ×8 GPUs. Tok/s (k): Lighthouse-stage throughput(tps) from torchtitan, aggregated over all ranks (the baseline SDPA shows the throughput when training without LH). The full set of ablations is provided in Table 2, Appendix A.

Lighthouse against cuDNN-backed SDPA. SDPA scales as Θ(N2d) while Lighthouse scales as Θ(S2d) with S defined in Eq. 8, so the gap widens with N (Fig. 3). At N=512K, Lighthouse is 21× faster on the forward pass and 17.3× faster on forward+backward; equivalently, SDPA needs ∼113K (fwd) / ∼122K (fwd+bwd) of context to reach the runtime Lighthouse takes at 512K. Fullmodel training tells a similar story but requires care: with our 530M-parameter architecture a single B200 OOMs beyond ∼100K on activations, gradients, and optimizer state regardless of attention method, so we implement context parallelism (Sec. D.4) where pyramid pooling, scoring, and top-

- K run shard-locally and the gathered sub-sequence rotates through stock ring attention [19] with no sparse-aware collectives. CP introduces a small ring-rotation overhead, costing ∼10% in per-rank throughput vs. the single-device extrapolation, but the Lighthouse-vs-SDPA speedup is preserved under matched CP geometry (Lighthouse-CP retains the same multiplicative gain over SDPA-CP that we see in the non-CP comparison), carrying the advantage cleanly to the 1M-token / 32-GPU regime.

#### 6.4 Design Ablations and Throughput

We sweep four design axes (scorer variant, pooling factor p, number of levels L, top-K budget k), each varied independently while the others stay at the defaults of Sec. 6.1; comparisons use the post-resume training loss at step 16,000. The full grid in Table 1 establishes three things. First, every Lighthouse configuration matches or beats the dense-SDPA-from-scratch baseline of 0.7237, so recoverability is not specific to any one hyperparameter setting.

Second, the projection-norm scorer is within ∼0.01 of dilated softmax in either direction (no uniform winner) but is parameter-free and roughly 9% cheaper in B200-hours (179.6–180.9 vs. 197.2– 199.7 at L=3,p=4).

Third, smaller p, shallower L, and smaller k all help slightly: the lowest-loss cell across the grid is L=3, p=2, k=1536 (dilated, loss 0.6825), Pareto-best on every metric within its Top-K block. The smaller-k direction is the most counter-intuitive: loss decreases monotonically as k shrinks over {1,536,2,048,3,072,4,096} (0.6825 → 0.6880 → 0.6890 → 0.6951) before dipping again at

k=6,144 (0.6831), plausibly because hierarchical selection regularises at our token budget; whether this reverses at substantially larger budgets is left to future work.

[Figure 1]

[Figure 2]

(a) Forward Latency (b) Backward Latency

- Figure 3: Attention latency vs. context length for SDPA (cuDNN) and Lighthouse (w/ cuDNN) on a single B200, L=3, p=4, sparsity ≈ 1:64. SDPA scales as Θ(N2d); Lighthouse scales as Θ(S2d) with S ≪ N. At N=512K, Lighthouse is 21× faster on the forward pass and 17.3× faster on the backward pass, equivalently Lighthouse at 512k takes same runtime as if training SDPA at ∼113K / ∼122K context.

The throughput story is consistent. Lighthouse stage-1 sustains 84–126k tok/s/GPU across the ablation grid against ∼46k for dense SDPA, a roughly 2× per-step advantage that holds across selection budgets; the projection-norm scorer at L=3, p=4, k=1536 tops the range at 126k by skipping the dilated-attention pass entirely.

End-to-end on the 10k+6k recipe, total runtime ranges from 22.5h (179.6 B200-h, norm k=1536,p=4) to 27.0h (215.7 B200-h, k=4096,p=2,L=3) against 37.9h (303.2 B200-h) for dense-SDPA-from-scratch on the same 16k-step / 50.3B-token budget: a 1.40× to 1.69× wall-clock speedup at matched or lower final loss. The saving comes entirely from stage-1; the SDPA-resume tail uses the same kernel as the baseline and matches its throughput. App. E gives the per-axis breakdowns, asymptotic-cost predictions, and per-stage timing decompositions.

### 7 Conclusion

We introduce Lighthouse Attention, a selection-based hierarchical attention for long-context pretraining that pools Q,K,V symmetrically across a multi-resolution pyramid and places selection outside the attention kernel, reducing the attention step to stock FlashAttention on a dense subsequence. The design is parameter-free, trains end-to-end with no auxiliary losses or straightthrough estimators, and inherits upstream FlashAttention improvements unchanged. A brief denseSDPA resumption after Lighthouse pretraining matches or beats dense-from-scratch at matched tokens on training loss and long-context retrieval, with 1.4–1.7× end-to-end speedups against cuDNN SDPA at ≥ 100K context on B200 and clean scaling to 1M tokens on multi-node Blackwell.

Limitations. Symmetric Q/K/V pooling presumes all queries co-occur in one forward pass, which autoregressive decoding violates; we rely on dense-SDPA resumption for an inference-ready model, and every downstream evaluation is run after that resume rather than on the hierarchical forward directly. The inner attention is Θ(S2d) on the gathered sub-sequence: sub-quadratic in N at fixed k but not strictly linear, so regimes where k must scale with N remain uncharacterised.

Future directions. Swapping the dense-SDPA resume for an asymmetric sparse target (DSA, NSA, HISA, MoBA) would yield a natively serveable checkpoint; per-layer or per-head adaptive k may outperform a fixed budget; the multi-scale pyramid extends naturally to vision, audio, and video; and serving integration (continuous batching, speculative decoding, KV-cache management) is needed to translate the training speedups into deployment.

### References

- [1] Anthropic. The Claude 3 model family, 2024. URL https://www.anthropic.com/news/ claude-3-family.
- [2] S. Arora, S. Eyuboglu, A. Timalsina, I. Johnson, M. Poli, J. Zou, A. Rudra, and C. R´e. Zoology: Measuring and improving recall in efficient language models. In ICLR, 2024. arXiv:2312.04927.
- [3] Y. Bengio, N. L´eonard, and A. Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.
- [4] K. Choromanski, V. Likhosherstov, D. Dohan, X. Song, A. Gane, T. Sarlos, P. Hawkins, J. Davis, A. Mohiuddin, L. Kaiser, D. Belanger, L. Colwell, and A. Weller. Rethinking attention with Performers. In ICLR, 2021.
- [5] T. Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In ICLR, 2024. arXiv:2307.08691.
- [6] T. Dao and A. Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In ICML, 2024.
- [7] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. R´e. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In NeurIPS, 2022.
- [8] DeepSeek-AI. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [9] DeepSeek-AI. DeepSeek-V3.2-Exp: Boosting long-context efficiency with DeepSeek sparse attention. arXiv preprint, 2025.
- [10] Q. Fu, M. Cho, T. Merth, S. Mehta, M. Rastegari, and M. Najibi. LazyLLM: Dynamic token pruning for efficient long context LLM inference. arXiv preprint arXiv:2407.14057, 2024.
- [11] Gemini Team, Google DeepMind. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [12] A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.
- [13] H. Guo et al. Log-linear attention. arXiv preprint, 2025.
- [14] E. Jang, S. Gu, and B. Poole. Categorical reparameterization with Gumbel-softmax. In International Conference on Learning Representations (ICLR), 2017.
- [15] H. Jiang, Y. Li, C. Zhang, Q. Wu, X. Luo, S. Ahn, Z. Han, A. H. Abdi, D. Li, C.-Y. Lin, Y. Yang, and L. Qiu. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. arXiv preprint arXiv:2407.02490, 2024.
- [16] X. Lai et al. FlexPrefill: A context-aware sparse attention mechanism for efficient longsequence inference. arXiv preprint, 2025.
- [17] Y. Li, Y. Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai, P. Lewis, and D. Chen. SnapKV: LLM knows what you are looking for before generation. arXiv preprint arXiv:2404.14469, 2024.
- [18] C. Lin et al. Twilight: Adaptive attention sparsity with hierarchical top-p pruning. arXiv preprint, 2025.
- [19] H. Liu, M. Zaharia, and P. Abbeel. Ring attention with blockwise transformers for near-infinite context. arXiv preprint arXiv:2310.01889, 2023.
- [20] E. Lu, Z. Jiang, J. Liu, Y. Du, T. Jiang, C. Hong, S. Liu, W. He, E. Yuan, Y. Wang, et al. MoBA: Mixture of block attention for long-context LLMs. arXiv preprint arXiv:2502.13189, 2025.

- [21] C. J. Maddison, A. Mnih, and Y. W. Teh. The concrete distribution: A continuous relaxation of discrete random variables. In International Conference on Learning Representations (ICLR), 2017.
- [22] Meta AI. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [23] Moonshot AI. Kimi K1.5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [24] Y. Ni et al. DoubleP: Hierarchical cluster-and-refine attention with centroid approximation. arXiv preprint, 2026.
- [25] OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2024.
- [26] M. Oren, M. Hassid, Y. Adi, and R. Schwartz. Transformers are multi-state RNNs. arXiv preprint arXiv:2401.06104, 2024.
- [27] Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2025.
- [28] L. Ribar, I. Chelombiev, L. Hudlass-Galley, C. Blake, C. Luschi, and D. Orr. SparQ attention: Bandwidth-efficient LLM inference. In International Conference on Machine Learning (ICML), 2024.
- [29] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [30] Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, and F. Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.
- [31] J. Tang, Y. Zhao, K. Zhu, G. Xiao, B. Kasikci, and S. Han. Quest: Query-aware sparsity for efficient long-context LLM inference. In International Conference on Machine Learning (ICML), 2024.
- [32] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations (ICLR), 2024.
- [33] R. Xu et al. XAttention: Block sparse attention with antidiagonal scoring. arXiv preprint, 2025.
- [34] S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim. Gated linear attention transformers with hardware-efficient training. arXiv preprint arXiv:2312.06635, 2024.
- [35] Z. Yang, D. Yang, C. Dyer, X. He, A. Smola, and E. Hovy. Hierarchical attention networks for document classification. In NAACL, 2016.
- [36] J. Yuan, H. Gao, D. Dai, J. Luo, L. Zhao, Z. Zhang, Z. Xie, Y. Wei, L. Wang, Z. Xiao, et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. arXiv preprint arXiv:2502.11089, 2025.
- [37] J. Zhang et al. SpargeAttention: Accurate and training-free sparse attention accelerating any model inference. arXiv preprint, 2025.
- [38] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. R´e, C. Barrett,

Z. Wang, and B. Chen. H2O: Heavy-hitter oracle for efficient generative inference of large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

- [39] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. R´e, C. Barrett, Z. Wang, and B. Chen. H2O: Heavy-hitter oracle for efficient generative inference of large language models. In NeurIPS, 2024. arXiv:2306.14048.
- [40] L. Zhao et al. HISA: Efficient hierarchical indexing for fine-grained sparse attention. arXiv preprint arXiv:2603.28458, 2026.
- [41] L. Zhao et al. InfLLM-V2: Dense–sparse switchable attention for seamless short-to-long adaptation. arXiv preprint, 2026.

### A Ablations

###### Configuration Scorer Params LH Total Total B200-Hrs Tok/s (k) Final Loss

Steps Steps Tokens (↓) (↑) (↓) SDPA Baseline (ctx = 98k) — 530M — 16k 50.3B 303.2 45.6 0.7237 SDPA recoverability (L=3, p=2, k=6144, ctx = 98k)

- LH → SDPA (12k+4k) Dilated 530M 12k 16k 50.3B 214.7 74.7 0.7102

- LH → SDPA (11k+5k) Dilated 530M 11k 16k 50.3B 219.6 75.4 0.7001 LH → SDPA (10k+6k) Dilated 530M 10k 16k 50.3B 228.0 75.0 0.6980

Scorer ablation (L=3, p=4, ctx = 98k)

k=1536 Dilated 530M 10k 16k 50.3B 197.2 99.5 0.6881 k=1536 Norm 530M 10k 16k 50.3B 179.6 126.0 0.6946 k=2048 Dilated 530M 10k 16k 50.3B 199.7 97.1 0.6969 k=2048 Norm 530M 10k 16k 50.3B 180.9 122.4 0.6921

Pooling-factor ablation (L=3, ctx = 98k)

p=2, k=1536 Dilated 530M 10k 16k 50.3B 203.9 93.9 0.6825 p=4, k=1536 Dilated 530M 10k 16k 50.3B 197.2 99.5 0.6881 p=8, k=1536 Dilated 530M 10k 16k 50.3B 206.2 92.1 0.6828 p=2, k=2048 Dilated 530M 10k 16k 50.3B 208.1 90.9 0.6880 p=4, k=2048 Dilated 530M 10k 16k 50.3B 199.7 97.1 0.6969

Number-of-levels ablation (p=2, ctx = 98k)

- L=3, k=1536 Dilated 530M 10k 16k 50.3B 203.9 93.9 0.6825
- L=4, k=1536 Dilated 530M 10k 16k 50.3B 200.2 96.4 0.6978
- L=5, k=1536 Dilated 530M 10k 16k 50.3B 201.5 96.3 0.6991

- L=3, k=2048 Dilated 530M 10k 16k 50.3B 208.1 90.9 0.6880
- L=4, k=2048 Dilated 530M 10k 16k 50.3B 202.2 94.5 0.6983
- L=5, k=2048 Dilated 530M 10k 16k 50.3B 206.5 92.3 0.7043

Top-K budget ablation (L=3, p=2, ctx = 98k)

k=1536 Dilated 530M 10k 16k 50.3B 203.9 93.9 0.6825 k=2048 Dilated 530M 10k 16k 50.3B 208.1 90.9 0.6880 k=3072 Dilated 530M 10k 16k 50.3B 214.9 86.1 0.6890 k=4096 Dilated 530M 10k 16k 50.3B 215.7 83.5 0.6951 k=6144 Dilated 530M 10k 16k 50.3B 208.1 88.3 0.6831

CP training (L=3, p=4)

k=1536, ctx = 98k, CP= 2, DP= 4 Norm 530M 10k 16k 100.7B 208.3 91.8 0.6903 k=2048, ctx = 98k, CP= 2, DP= 4 Norm 530M 10k 16k 100.7B 210.9 89.2 0.6928 k=4096, ctx = 256k, CP= 8, DP= 1 Norm 530M 10k 16k 1.07T 1300.3 48.9 0.6721

- Table 2: Full Lighthouse ablation summary. 530M Llama-3, 16k optimizer steps; LH Steps run Lighthouse, SDPA Steps run dense SDPA-resume on a single 8×B200 node (the final block adds context parallelism). B200-Hrs: combined wall-clock ×8 GPUs. Tok/s (k): Lighthouse-stage throughput(tps) from torchtitan, aggregated over 8 ranks (the Dense-SDPA-from-scratch row has no LH stage and reports its dense values). Final Loss: training loss at step 16,000. Bold marks the per-block best on each metric (skipped for throughput in the CP block since context length varies inside it).

### B Complexity Derivation

Stage Primitive Cost Projections Q, K, V GEMM Θ(N dmodel d) Pyramid pool view+mean Θ(N d) Scoring (norms, max-pool) norm+max Θ(N d) Top-k selection chunked bitonic Θ(N log k) Gather to sub-sequence torch.gather Θ(S d) Dense sub-sequence attention FlashAttention Θ(S2 d) Scatter-back custom atomic Θ(N d)

- Table 3: Per-layer complexity of a Lighthouse attention layer, with S = N/pL−1 + (L − 1)pk.

We derive the per-layer compute complexity of Lighthouse and show that, at bounded selection budget k, total compute is linear in the sequence length T.

Setup. Lighthouse with sequence length T, pooling factor p, L pyramid levels, top-k budget k, and head dimension d.

Per-stage cost. Table 3 decomposes one layer into its stages. Sub-sequence size. By construction the gathered sub-sequence has size

T

pL−1 + (L − 1)pk, (12) where the first term is the coarsest level (kept whole) and the second is the contribution of the finer

S =

- L − 1 levels (each of which contributes at most pk entries).

Choice of L. Setting L = logp(T/k) gives pL = T/k and therefore pL−1 = T/(pk). Substituting into Eq. (12),

##### T

pL−1 = pk, and the two terms combine into

S = pk + (L − 1)pk = pk · L = pk · logp(T/k) = Θ(k log T) (13) (treating p as constant). Attention cost in terms of T. The dense FlashAttention call on S tokens costs Θ(S2 · d). Substituting Eq. (13),

S2 · d = Θ k2 log2 T · d , which is polylogarithmic in T for bounded k. Total per-layer compute. Summing the contributions in Table 3 after substituting S = Θ(k log T),

Tlayer = Θ(T · d) + Θ(T log k) + Θ k2 log2 T · d .

For bounded k, the polylog term is sub-linear in T and the Θ(T ·d) term from the projection, pooling, scoring, and scatter stages dominates. Therefore

|Tlayer = Θ(T · d) at bounded k.|
|---|

(14)

Two distinct quantities. We emphasize that two “log” factors appear at different points and should not be conflated:

- • Sub-sequence size: S = Θ(k log T) the size of the data tensor passed to FlashAttention.
- • Per-layer compute: Θ(T · d) the total flops to run one layer.

The logarithmic factor lives in S; the total compute is linear in T because only S tokens (not T) attend to one another, while the linear-cost stages remain linear in T.

Comparison to other attention families. Table 4 places Lighthouse alongside dense softmax, log-linear attention, and linear-attention / SSM families.

Method Per-layer compute Dense softmax Θ(T2 · d) Log-Linear Attention [13] Θ(T log T · d) Lighthouse (bounded k) Θ(T · d) Linear attention / SSMs (fixed budget) Θ(T · d)

- Table 4: Per-layer compute across attention families. Lighthouse and linear/SSM families share the same asymptotic class; log-linear attention is strictly worse asymptotically; dense softmax is quadratic.

### C Design Choices (extended)

The Lighthouse pipeline of Sec. 3 admits several non-trivial design decisions whose rationale is not obvious from the equations alone.

#### C.1 Symmetric Q/K/V Pooling

Prior hierarchical sparse designs [36, 40, 41] compress only the key–value side and leave queries at full resolution. This is natural for inference, where autoregressive decoding presents one query at a time. Training, however, exposes every query in parallel: we can compress the query side too, then recover dense behavior at inference by briefly resuming with stock SDPA [7]. Lighthouse pools Q in lockstep with K,V at every level, reducing the dense FlashAttention call from O(N S d) to O(S2 d) and yielding coherent (Q(ℓ),K(ℓ),V (ℓ)) triples that share a representation space across levels — pooled queries route to pooled keys, producing summary–summary interactions an asymmetric pyramid cannot express. Sec. 6.2 verifies the symmetric design is invariant under the recovery test.

#### C.2 Parameter-Free Scoring

The scoring functional has the widest design space; two natural candidates are (a) a dilated softmax attention over the pyramid (most faithful to “what would softmax do,” as used by the learned selectors in NSA [36] and DSA [9], but O(N2/r)) and (b) the per-head ℓ2 norms of the layer’s own projections, ∥Q(iℓ)∥2 and ∥Ki(ℓ)∥2 (no parameters, no Q–K interaction, Θ(N)). We adopt the projection-norm scorer. It is the cheaper of the two and the more conservative benchmark: a dilatedattention scorer strictly adds Q–K interaction information and can only help. Any positive result with projection-norms is therefore a lower bound on what Lighthouse can deliver, and our ablations in Appendix. A confirm the dilated scorer matches projection-norms within noise — evidence that the selection structure, not the scoring function, drives the long-context behavior.

#### C.3 Selection–Attention Decoupling

Every competing selection method we are aware of fuses its selection machinery into the attention kernel: NSA [36], DSA [9], and HISA [40] each ship a custom sparse-attention kernel that reads indices from a selection step. Lighthouse places that interface outside the kernel: selection produces a dense, contiguous sub-sequence and attention is a stock FlashAttention [7, 5, 29] call on it. Two consequences follow: the same kernel runs at training and inference, so there is no train-vs-serve kernel divergence; and correctness of the attention step can be checked against the dense baseline by running attention with selection disabled, which is exactly what the SDPA-resume evaluation in Sec. 6.2 exercises.

#### C.4 Gradient Flow

The top-K step is discrete and we do not approximate it with a straight-through estimator [3] or Gumbel softmax [14, 21]. Gradients flow back from the loss through the scatter, FlashAtten-

tion [7], and gather into WQ,WK,WV via the differentiable values Q, K, V ; selection indices and the scoring functional carry no gradient. The projections therefore learn to produce values that are useful when selected rather than scores that are good at selecting — avoiding the optimization pathologies (scorer collapse, scorer–attention misalignment, auxiliary-loss tuning) that learnable selectors [36, 9] are prone to.

### D Kernel Design and Parallelism (extended)

#### D.1 Pyramid and Scoring

The pyramid and scoring stages are not custom kernels: pooling is a view + mean, scoring is a pertoken norm followed by view + max per coarser level. All ops are pointwise or reshape-plus-reduce and are fused by torch.compile into a single device pass.

#### D.2 Chunked-Bitonic Top-k Kernel

Selecting k pyramid entries out of Θ(N) candidates is the first stage that warrants a custom kernel. A textbook bitonic top-k sorts the entire score stream in shared memory or registers, which fails at our budgets: k = 4096 over a pyramid of ≈ 2N entries cannot fit in a single thread block, and a global sort serializes across the sequence. Lighthouse instead uses a chunked-bitonic design: the score stream is partitioned into fixed-size chunks of Nchunk = 2048 scores, each chunk maintains a running top-m buffer (m = 128) updated through an in-register bitonic merge, and the N/Nchunk chunks dispatch as independent CTAs. No thread block ever holds more than m scores; the work is fully parallel.

The concatenated indices arrive in score-sorted order within each chunk and chunk-major order across chunks — not the causal order the attention step requires. We apply a single torch.sort pass to re-order them by pyramid position, after which the gathered Q, K, V form a contiguous, causally consistent sub-sequence indistinguishable in shape from a standard dense FlashAttention input. This is what makes the rest of the pipeline kernel-agnostic.

This design does not produce the same index set as a theoretical global top-k: if the k globally highest-scoring entries cluster in one chunk, some are replaced by lower-scoring entries from other chunks. Read positively, this is a stratified top-k that guarantees every region of the sequence contributes some tokens, which empirically yields more balanced attention coverage than strict global top-k and avoids selection collapse onto a narrow span.

#### D.3 Gather and FlashAttention Dispatch

Once I is produced, gather is a stock torch.gather followed by a single FlashAttention call on the gathered tensors. This is where placing selection outside the attention kernel pays off. Prior selection-based methods [36, 9, 40, 20] embed the selection–attention interface inside the kernel: each tile reads a sparse index list and performs an indirect KV load, which forces (i) a custom forward kernel per GPU architecture, (ii) a matching custom backward that inverts the hierarchical pattern, and (iii) ongoing tile/schedule re-tuning as hardware evolves. Lighthouse’s dense subsequence design sidesteps all three: forward and backward are bit-for-bit identical to a standard dense Transformer’s.

#### D.4 Context-Parallel Execution

At sequence lengths beyond 128K we train with context parallelism across W devices, each rank holding a contiguous slice and using standard ring attention. Lighthouse extends cleanly without custom collectives because its pre-attention primitives are local: (i) the coarsest pool window pL−1 (e.g. 64) is orders of magnitude smaller than the shard size (N/W = 128K at N=1M,W=8), so pooling and scoring need no inter-rank communication; (ii) each rank runs the chunked-bitonic top-k on its own pyramid, producing Ir ⊆ Pr from tokens it already owns; (iii) the gathered sub-sequence is dense, so FlashAttention runs under standard ring attention [19] — KV shards rotate through the ring as in a fully dense long-context run, and each rank’s queries see the cross-shard context selected by every other rank’s Lighthouse pipeline. This last property is only possible because Lighthouse’s

selection output is a contiguous tensor; sparse-selection kernels cannot express ring rotation without engineering specific to the sparse layout. The combined design supports 1M-token pretraining across 32 Blackwell GPUs (4 nodes × 8 GPUs, CP degree 8) with no changes to the attention kernel itself.

### E Design Ablations and Throughput (extended)

This appendix gives the per-axis ablation results in detail, the per-stage throughput decomposition, and the asymptotic-cost predictions backing each design choice. All loss numbers are the step-16,000 post-resume training loss of the two-stage Lighthouse-then-SDPA recipe (Sec. 6.2); throughput numbers are aggregated over 8 ranks of one B200 node at 98,304-token context (matching Table 1).

#### E.1 Scorer Variants

We compare the parameter-free projection-norm scorer against the dilated softmax-attention scorer at fixed L=3, p=4. At k=1536, dilated reaches 0.6881 while projection-norm reaches 0.6946; at k=2048 the order reverses, with 0.6969 for dilated and 0.6921 for projection-norm. The two are within ∼0.01 of each other in both directions and neither is uniformly better. Combined with projection-norm’s lower compute cost (179.6–180.9 B200-h vs. 197.2–199.7 for dilated, an ∼9% saving) and absence of additional learnable parameters (Sec. C.2), projection-norm is the throughput-sensitive default; dilated remains a defensible alternative when the lowest absolute loss matters.

#### E.2 Pooling Factor

Holding L=3 and varying p ∈ {2,4,8} at k=1536 gives final losses of 0.6825, 0.6881, 0.6828 respectively; at k=2048, p=2 and p=4 give 0.6880 and 0.6969. Pooling factor has a small effect: p=2 and p=8 are essentially indistinguishable, with p=4 marginally worse by 0.005–0.015. We adopt p=2 as the default; p=4 paired with the projection-norm scorer is the wall-clock-favoured alternative.

#### E.3 Number of Levels

Holding p=2 and varying L ∈ {3,4,5} produces a monotonically increasing final loss: at k=1536, L=3 gives 0.6825, L=4 gives 0.6978, and L=5 gives 0.6991; the ordering carries to k=2048 (0.6880 → 0.6983 → 0.7043). Deeper pyramids spread the same selection budget over more coarse levels, leaving fewer entries at the finest level where the model is most sensitive. L=3 is best across both selection budgets and we adopt it as the default.

#### E.4 Top-K Budget

Holding L=3, p=2 and varying k ∈ {1,536,2,048,3,072,4,096,6,144} gives final losses of 0.6825, 0.6880, 0.6890, 0.6951, 0.6831. The trend is monotonic over k ≤ 4096 and dips back at k=6144. A larger selection budget does not translate to lower post-resume loss within the range we tested. Sparser configurations may regularise against our relatively small training-token budget; investigating whether this trend reverses at much larger budgets is left to future work. The practical implication is that k=1536 is preferred from both a loss and a wall-clock standpoint.

#### E.5 Throughput Decomposition

Stage-1 throughput. The recoverability runs (k=6144, L=3, p=2, dilated) sustain 74.7–75.4k tok/s/GPU. Across the ablation grid (varying k,p,L at the 98K context) the stage-1 throughput range is 83.5–126.0k tok/s/GPU, vs. ∼46k for dense SDPA. The intra-grid trends follow the S = N/pL−1 + (L − 1)pk prediction: raising p from 2 to 4 at k=2048,L=3 lifts throughput from 90.9 to 97.1k; raising L from 3 to 4 at k=2048,p=2 lifts it from 90.9 to 94.5k (deeper pyramid spreads the budget across more coarse levels). Swapping the dilated scorer for projection-norm at L=3, p=4, k=1536 accelerates stage-1 from 99.5 to 126.0k tok/s/GPU because the scorer no longer runs an attention pass over the dilated pyramid.

[Figure 3]

- Figure 4: Needle-in-a-Haystack at 98K training, step 16,000. Four Lighthouse → SDPA configurations (varying k ∈ {1536,2048} and scorer ∈ {dilated, norm} at L=3, p=4) and the dense SDPAfrom-scratch baseline (bottom). Each cell is the mean retrieval rate over 10 single-digit passkeys at the given (context, depth); the per-panel mean is shown in each title. Random chance is 10%.

End-to-end speedup. Aggregating both stages of the 10k+6k recipe across the ablation grid, end-to-end runtime ranges from 22.5h (179.6 B200-h; norm k=1536, p=4, L=3) to 27.0h (215.7 B200-h; k=4096, p=2, L=3) against 37.9h (303.2 B200-h) for dense-SDPA-from-scratch on the same 16,000-step / 50.3B-token budget: a 1.40× to 1.69× wall-clock speedup at matched or lower final loss. The saving comes entirely from stage-1; the SDPA-resume tail runs the same kernel as the baseline and matches its throughput.

### F Long-Context Retrieval (NIAH)

At 530M parameters and only 16,000 training steps (∼50.3B tokens), full prose Needle-in-aHaystack scores near-zero across the board, so we adopt a simplified single-digit variant that isolates the retrieval signal. To complement the loss-based recoverability evaluation in Sec. 6.2, we run this test over five step-16,000 checkpoints: four Lighthouse → SDPA two-stage runs (varying k ∈ {1536,2048} and scorer ∈ {dilated, norm} at L=3, p=4) and the dense-SDPA-from-scratch baseline at matched compute and tokens. Inference uses dense SDPA in every case (Sec. 6.1).

Protocol. A single passkey digit (one of {0,1,...,9}) is hidden in random alphanumeric filler at depths {0,15,30,50,70,85,100}% across context lengths {4,8,16,32,64,96}K. For each cell we

run one forward pass over the full prompt and take an argmax restricted to the 10 digit tokens at the last position. We average the 0/1 score over the full digit sweep, so each cell reports the mean retrieval rate over 10 trials; random chance is 10%.

Findings. Three of the four Lighthouse runs are at or above the SDPA-from-scratch baseline (mean retrieval 0.72): k=2048 dilated wins overall at 0.76, k=1536 dilated reaches 0.73, and k=2048 norm matches the baseline at 0.72; only k=1536 norm dips, to 0.65 (Fig. 4). Two patterns emerge. First, larger k is the dominant axis: k=2048 ≥ k=1536 for both scorers, with a gap of 0.03 (dilated) or 0.07 (norm). Second, the norm scorer hurts retrieval more than it hurts training loss: at fixed k, switching dilated to norm costs 0.04 at k=2048 and 0.08 at k=1536, the largest single-axis gaps in the grid. Combined with the loss-side finding that smaller k regularises better, the right default depends on whether the downstream task is loss- or retrieval-driven.

