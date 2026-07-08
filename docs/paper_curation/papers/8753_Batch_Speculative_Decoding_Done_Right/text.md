## Batch Speculative Decoding Done Right

Ranran Haoran Zhang1 Soumik Dey2 Ashirbad Mishra2 Hansi Wu2 Binbin Li2 Rui Zhang1

# arXiv:2510.22876v3[cs.CL]15Feb2026

### Abstract

Speculative decoding must produce outputs distribution identical to standard autoregressive generation—this output equivalence is not an optimization target but the defining criterion of valid speculative decoding. We demonstrate that all existing batch speculative decoding implementations violate this fundamental requirement, producing corrupted outputs ranging from repetitive tokens to gibberish. These failures stem from the ragged tensor problem: sequences in the same batch accept different numbers of draft tokens, desynchronizing position IDs, attention masks, and KV-cache state. We present the first authentic batch speculative decoding framework. We (1) formalize the synchronization invariants that valid batch speculative decoding must satisfy, (2) present EQSPEC, the first algorithm that guarantees output equivalence, and analyze its cost structure to show that alignment overhead grows superlinearly and consumes up to 40% of computation, and (3) introduce EXSPEC, which reduces this overhead through cross-batch scheduling that dynamically groups same-length sequences. On SpecBench across Vicuna-7B/68M, Qwen38B/0.6B, and GLM-4-9B/0.6B pairs, our methods achieve up to 3× throughput improvement at batch size 8 while maintaining algorithmic correctness. Our methods achieve 95% decoding-equivalence, with residual divergence attributable to floatingpoint non-determinism in GPU inference, not the synchronization failures that cause near-zero equivalence of prior methods. Our code is available at https://github.com/eBay/spec dec

### 1. Introduction

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) accelerates LLM inference by using a small draft

1The Pennsylvania State University, University Park, PA, USA 2eBay Inc, San Jose, CA, USA. Correspondence to: Ranran Haoran Zhang <hzz5361@psu.edu>, Rui Zhang <rmz5227@psu.edu>.

model to propose multiple tokens that the target model verifies in parallel. The fundamental requirement of speculative decoding is output equivalence (Leviathan et al., 2023; Xia

- et al., 2024): the generated token distribution must remain identical to standard autoregressive generation. Any method that violates this requirement, regardless of its throughput, does not constitute valid speculative decoding. Batch speculative decoding aims to combine this per-sequence acceleration with standard batching, but as we demonstrate, all existing implementations fail to preserve output equivalence when batch size exceeds one.

Figure 1 highlights a fundamental but systematically overlooked failure in batch speculative decoding: methods with impressive throughput can produce corrupted outputs. The widely used HuggingFace implementation (Wolf et al., 2020) preserves output equivalence, but only for batch size 1. By contrast, in our tests of public batch implementations—specifically, BSP (Su et al., 2023) and DSD (Yan

- et al., 2025), produce repetitive tokens or <unk> symbols under greedy decoding rather than matching standard generation. These are not minor quality degradations but fundamental violations of decoding equivalence. Equivalence to reference decoding is a yes/no property: prior batch implementations fail to preserve the target model’s output distribution, yielding near-zero exact-match to standard autoregressive decoding due to synchronization errors. The root cause is improper handling of the ragged tensor problem (Qian et al., 2024), which desynchronizes position IDs, attention masks, and KV-cache state across sequences in a batch.

Valid batch speculative decoding requires maintaining synchronization invariants (position IDs, attention masks, KVcache) across ragged tensors, a challenge that prior implementations fail to address. We formalize these invariants, prove they are preserved by our update rule, and present EQSPEC, the first algorithm that guarantees output distribution equivalence by enforcing them after each verification round (Section 3.1). We analyze the cost structure of batch speculative decoding and show that alignment overhead grows superlinearly with batch size (Section 3.2); empirically, realignment consumes up to 40% of computation at batch size 8 (Section 4.3). This is an inherent cost of algorithmic correctness across ragged tensors, not an implementation inefficiency. The superlinear growth of alignment

### YES, High Throughput...

### BUT, Output Gibberish!

###### Throughput(tokens/sec)

Spec-1

EQSPEC EXSPEC

200

Method Generated Output Prompt: ”The weather today is” Ground Truth not as bad as it was yesterday... Spec-1 ✓ Matches ground truth BSP ✗ not not not not not not not... DSD ✗ 〈unk〉〈unk〉 The perfect time...

DSD

175

BSP

150

125

100

75

50

EQSPEC ✓ Matches ground truth EXSPEC ✓ Matches ground truth

25

1 2 4 8 16 32

Batch Size

- Figure 1. Batch speculative decoding on Vicuna-7B/68M: Existing methods achieve high throughput but violate the fundamental requirement of output equivalence by producing corrupted outputs. Our approach preserves reference-decoding equivalence while still achieving competitive performance.

overhead represents a fundamental barrier that explains why prior methods either sacrifice validity or fail to scale.

To reduce this overhead, we introduce EXSPEC (Section 3.2), which expands the scheduling scope beyond a single batch. EXSPEC maintains a pool of active sequences and dynamically groups those with identical lengths, eliminating realignment for homogeneous groups. This strategy preserves the scaling efficiency of standard batching while retaining per-sequence speculation benefits. At batch size 8, EXSPEC achieves a 3× throughput improvement over batch size 1. Beyond batch size 8, throughput degrades as sequence length diversity reduces grouping effectiveness, forcing more frequent fallbacks to realignment. Section 4.3 examines these scaling dynamics in detail.

Our experiments on SpecBench (Xia et al., 2024) yield two main results. First, Algorithmic Correctness: unlike prior approaches such as BSP and DSD, which suffer severe output corruption, our method maintains algorithmic correctness across Vicuna-7B/68M (Zheng et al.,

- 2023), Qwen3-8B/0.6B (Yang et al., 2025), and GLM-49B/0.6B (GLM et al., 2024) model pairs, achieving 95% decoding-equivalence with residual divergence attributable to floating-point non-determinism in GPU inference, not synchronization failures. Second, Scalability: at batch size 8, EXSPEC achieves up to a 3× speedup over batch size 1, successfully combining batch parallelism with per-sequence speculation gains. Our contributions:

through explicit synchronization, with formal proofs that the synchronization invariants are preserved.

• We introduce EXSPEC, which reduces alignment overhead through cross-batch scheduling that dynamically groups same-length sequences. On SpecBench across Vicuna-7B/68M, Qwen3-8B/0.6B, and GLM-49B/0.6B pairs, our methods achieve up to 3× throughput at batch size 8 while maintaining algorithmic correctness.

### 2. Design Space Analysis

When sequences within a batch accept different numbers of draft tokens during verification, tensors become irregular, violating GPUs’ requirement for rectangular layouts—this is the ragged-tensor problem illustrated in Figure 2. Despite batching’s centrality to production deployments, existing implementations lack a principled design that preserves output equivalence with standard decoding while scaling with batch size. We identify three approaches to handle raggedness: Masking, Rollback, and Dynamic Padding. Yet, as our systematic analysis shows, current instantiations of these approaches fail to simultaneously maintain algorithmic correctness and performance at scale. To close this gap, we first analyze the pitfalls of each approach and then introduce the first valid batch speculative decoding with explicit synchronization requirements.

- • We identify that existing batch speculative decoding implementations violate output equivalence due to improper handling of the ragged tensor problem. We formalize the synchronization invariants (position IDs, attention masks, KV-cache alignment) that any valid implementation must satisfy and show precisely how prior methods fail to maintain them, producing corrupted outputs.
- • We present EQSPEC, the first batch speculative decoding algorithm that guarantees algorithmic correctness

✗ Masking Approach (non-contiguous position IDs). This approach operates directly on ragged tensors by masking rejected tokens in attention and reassigning position IDs so new tokens align with their content positions. Across verification rounds with varying rejections, sequences accumulate padding in various positions (middle and right), forming non-contiguous position IDs that standard Transformer implementations handle poorly. BSP (Su et al., 2023) attempts this via masking but fails to maintain position-ID consistency across iterations, yielding corrupted outputs (Figure 1).

[Figure 1]

[Figure 2]

[Figure 3]

- Figure 2. The ragged tensor problem in batch speculative decoding. Differing numbers of accepted draft tokens across sequences in the same batch lead to ragged-shaped input IDs tensors and KV Cache that disrupt subsequent batch operations.

EAGLE’s experimental batching code1 (Li et al., 2025) encounters similar framework limitations. BASS (Qian et al.,

- 2024) addresses this through custom attention kernels that tolerate ragged padding in the middle and right, but this requires coupled modifications to both attention computation and KV-cache indexing, sacrificing portability across model architectures and attention variants.

✗ Rollback Approach (speculation waste). After each verification step, all sequences are truncated to the batch’s minimum accepted length (Wolf et al., 2020; kamilakesbi, 2024). This keeps the batch aligned, but it throws away additional verified tokens that faster sequences had already accepted. As batch size grows and acceptance-rate variance widens, the waste compounds; in the extreme, one persistently rejecting sequence forces single-token progress for the entire batch. In effect, throughput collapses to that of the worst-performing sequence, undermining speculative gains and rendering the approach impractical at larger scales. A variant approach, employed by vLLM’s v0 engine, avoids truncation through batch expansion: each sequence is duplicated K times with progressively longer draft prefixes (e.g., [t1],[t1,t2],...), all verified in a single forward pass. While this preserves accepted tokens, it incurs K× redundant computation and memory overhead that scales with both batch size and draft length. At scale, this memory explosion caused GPU out-of-memory failures, ultimately leading to the deprecation of batch expansion in vLLM v1 (see Appendix C for details).

✓ Dynamic Padding Approach. This approach realigns sequences after each verification by adjusting left padding to maintain right alignment, preserving all accepted tokens. While conceptually simple, algorithmic correctness requires tight synchronization of position IDs, attention masks, and the KV-cache. DSD’s experimental code (Yan et al., 2025) follows this idea but merely repads at each step—adding varying left padding without ever unpadding—thereby inflating sequences. It also contains three critical errors: (i)

1While EAGLE’s main contribution concerns improved draft models rather than batching, its repository includes experimental batch-related code we analyzed for implementation challenges. https://github.com/SafeAILab/EAGLE/issues/250

sampling bonus tokens from the draft model rather than the target model; (ii) redundantly regenerating KV-cache entries, causing memory bloat; and (iii) desynchronizing padding, position IDs, and the KV-cache across iterations. Despite the overhead of repeated realignment, a valid dynamic-padding implementation fits within standard frameworks and preserves all verified tokens.

Among the three approaches, only dynamic padding is viable. (1) Masking with non-contiguous position IDs addresses only half the problem: even if attention is patched to tolerate such layouts (via adjusted position IDs and masking), algorithmic correctness still requires that the KV-cache correspond to the same post-verification token stream across verification rounds. If one fully fixes both sides, the implementation must perform per-round cache/token realignment (i.e., compacting/realigning the batch state after verification), which effectively converges to dynamic padding. (2) Rollback preserves alignment but discards verified tokens by truncating all sequences to the batch’s minimum accepted length, and this waste grows with batch size. (3) Dynamic padding maintains algorithmic correctness by explicitly synchronizing both position IDs and KV-cache after each verification round. We provide a detailed bug taxonomy in Appendix A, distinguishing batch-independent implementation errors (e.g., wrong sampling distribution) from ragged tensor synchronization failures that only manifest at batch size > 1. We identify and address all of these, but the synchronization problem is the core challenge: no prior work formalizes what invariants batch speculative decoding must maintain, and this formalization is our central contribution.

Output Equivalence. We formalize the validity requirement for batch speculative decoding. Under greedy decoding (temperature = 0), speculative decoding must yield outputs identical to standard autoregressive generation, and any token-level divergence indicates an implementation error. For temperature > 0, where outputs are sampled stochastically, speculative decoding must preserve the output distribution of the target model; that is, the probability of generating any token sequence must remain unchanged. This losslessness, whether measured by exact output equivalence or distributional equivalence, is the defining criterion of valid speculative decoding and the hallmark of an authen-

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Figure 3. EQSPEC synchronizes via unpad–append–repad.

tic implementation. Methods that achieve high throughput while violating this criterion, as we demonstrate for BSP and DSD in Figure.1, do not constitute valid speculative decoding regardless of their reported performance metrics.

### 3. Method

We propose a batch speculative decoding method that explicitly maintains synchronization across sequences (position IDs, attention masks, and KV-cache) while also aiming to keep the added synchronization overhead small for higher throughput. The main trade-off is that maintaining equivalence to standard decoding requires re-synchronizing position IDs, attention masks, and the KV-cache after each verification round when the batch becomes ragged, and this synchronization can account for up to 40% of total computation. We introduce two complementary mechanisms: EQSPEC specifies and enforces the minimal synchronization invariants required for valid computation (Section 3.1), while EXSPEC groups same-length sequences to avoid synchronization overhead (Section 3.2).

- 3.1. EQSPEC: Algorithmic Correctness via Synchronization Invariants

Rectangular Alignment. We formalize the algorithmic correctness requirements for batch speculative decoding. Let B denote the batch size, and let sequence i have padding length pi, content length ci, and ai newly accepted tokens (including the bonus token) after verification round t. A valid implementation must maintain rectangular alignment across the batch:

∀i ∈ [1,B], p(it) + c(it) = L(t) (I1) where L(t) is the padded sequence length at round t.

Position-ID Contiguity. Additionally, position IDs must be derived consistently from the attention mask:

- j
- k=0

1[toki[k] ̸= pad] − 1 (I2)

posi[j] = max 0,

which guarantees contiguous position IDs starting from zero at the first content token, as required by RoPE (Su et al., 2024) and other position-aware attention mechanisms. Continuous batching systems (e.g., vLLM, SGLang) process requests using variable-length packing, sidestepping I1, but still require scatter-gather across requests and rollback both position-ID and KV-cache for rejected tokens—a onedimensional analog of alignment. Critically, I2 remains a prerequisite for algorithmic correctness regardless of batching strategy; note that neither vLLM nor SGLang currently supports continuous batching speculative decoding with external draft models (see Appendix.E).

After batch verification, sequence i accepts ai tokens, breaking rectangular alignment since ai varies across sequences. To restore I1, the new padding must satisfy:

∀i ∈ [1,B], p(it+1) + c(it) + ai = L(t+1) (1)

where L(t+1) = maxj(c(jt) + aj). This yields the persequence padding offset:

δi = p(it+1) − p(it) = L(t+1) − L(t) − ai (2)

We formally prove that this update rule preserves rectangular alignment I1 across all verification rounds (Theorem B.1 in Appendix). Invariant I2 is enforced by recomputing position IDs after each repadding operation.

Unpad-Append-Repad. Figure 3 illustrates how EQSPEC restores a valid batch after each verification round. The procedure applies the offset δi to realign all batch state: token sequences receive fresh left padding, attention masks are recomputed, position IDs are recalculated via I2, and KV-cache entries are shifted by δi positions to maintain token-to-cache correspondence.

During batch verification, the target model performs a single forward pass over draft tokens, detects the first mismatch per sequence, and samples a bonus token from the target distribution (Algorithm 3 in Appendix.C). However, since this sampling occurs after the forward pass completes, the bonus token lacks KV-cache entries and must be included in the next forward pass to create them. The complete EQSPEC procedure is detailed in Algorithm 1.

3.2. EXSPEC: Reducing Overhead via Cross-Batch Scheduling

Theoretical Speedup Analysis The speedup of speculative decoding depends on both the token-acceptance rate

- Algorithm 1 EQSPEC: Minimalism Batch Spec

[Figure 8]

Require: Draft model Md, Target model Mt, Prompts P, Max

tokens T, Draft length K Ensure: Generated sequences S

- 1: S ← Tokenize(P) ▷ Batch left padding
- 2: KVCache ← ∅
- 3: while until max new tokens do
- 4: Phase 1: Draft Generation
- 5: D ← Md.Generate(S, K)
- 6: Phase 2: Batch Verification
- 7: ▷ See Alg. 3 in Appendix for BatchVerify
- 8: A, B, KVCache ← BatchVerify(Mt, S, D, KVCache)
- 9: ▷ See Figure 3 for illustration on index offset.
- 10: Phase 3: Unpad-Append-Repad
- 11: for each sequence i in batch do
- 12: S[i] ← Unpad(S[i])
- 13: S[i] ← S[i] ⊕ A[i] ⊕ B[i]
- 14: end for
- 15: S, offset ← BatchRepad(S)
- 16: KVCache ← Realign(KVCache, offset)
- 17: end while
- 18: return S

[Figure 9]

[Figure 10]

[Figure 11]

Figure 4. EXSPEC pools ragged sequences by length, avoiding realignment; only unmatched sequences need syncing, turning fixed overhead into optional cost.

and computational costs. The original formulation by Leviathan et al. (2023) analyzes single-sequence performance, modeling the expected tokens generated per iteration

- as (1−αk+1)/(1−α), where α is the token acceptance rate (TAR) and k is the number of draft tokens per speculation round. This single-sequence view assumes no batch overhead and focuses purely on acceptance dynamics. For batch speculative decoding, we introduce a batch-aware speedup formula:

S =

α · k cdraft + cverify + coverhead(B)

(3)

where S denotes speedup relative to non-speculative decoding (i.e., running the target model alone), cdraft and cverify are the relative costs of draft generation and target verification, and coverhead(B) captures batch-dependent alignment overhead absent from single-sequence analysis.

Critically, coverhead(B) scales superlinearly with batch size B. Inside the alignment overhead, the KV-cache is the major bottleneck, consisting of rank-4 tensors (batch × heads × sequence × dimension), and each padding adjustment triggers allocation and concatenation of high-dimensional zeros. While cdraft and cverify benefit from GPU parallelism, the alignment overhead grows with both batch size and variance in acceptance rates across sequences. Profiling EXSPEC confirms this: alignment consumes 39.4% of computation

- at batch size 8, rising to 46.7% at batch size 16.

However, the speedup analysis reveals a key insight: when all sequences accept the same number of tokens, δi = 0 and realignment overhead vanishes (Corollary B.3). Rather than accelerating these operations, EXSPEC sidesteps them through scheduling.

Cross-Batch Scheduling. EXSPEC differs from EQSPEC in how it manages sequence lifecycles. Rather than maintaining fixed batches that require realignment after every verification, we introduce a SequencePool that holds sequences individually in their ragged states. This enables three optimizations: (i) sequences that complete (reach EOS) are immediately removed, avoiding wasted computation; (ii) lazy realignment defers synchronization until strictly necessary; and (iii) dynamic batch formation over a sliding window of W > B sequences greatly increases the chance of finding same-length groups.

Figure 4 illustrates the flow. After verification, ragged sequences return directly to the pool without realignment. The scheduler scans the active window and attempts to form batches of identical length. Same-length sequences concatenate directly—no padding adjustments, no positionID recomputation, no KV-cache realignment—bypassing coverhead(B) entirely. Only when same-length grouping fails do we fall back to unpad–append–repad. Combined with prompt-length sorting, grouping rates approach unity for similar workloads, turning constant realignment into the rare case. The complete procedure is detailed in Algorithm 2.

### 4. Experiments

We evaluate EQSPEC and EXSPEC along two dimensions often conflated in prior work: output equivalence and throughput. Through systematic evaluation across three model families and comparisons with both research prototypes and production systems, we show that our approach uniquely preserves output equivalence while achieving competitive speedups.

- Algorithm 2 EXSPEC: Cross-Batch Scheduling

ality); (2) Dynamic-padding approaches: DSD (Yan et al., 2025) explores padding but mishandles the KV-cache, while EQSPEC implements correct synchronization and EXSPEC adds cross-batch scheduling; (3) Reference baselines: Spec1 (batch-size-1 speculation from Hugging Face Transformers), which does not support batch speculative decoding2. Production systems (vLLM, SGLang; see also Sections 2 and 3.1) employ continuous batching with paged attention, which sidesteps the ragged tensor problem through different architectural choices. However, neither currently supports batch speculative decoding with external draft models: vLLM deprecated this in v1 due to batch expansion memory overhead, and SGLang only supports EAGLE-family drafters. We include a comparison in Appendix E for completeness, but note these systems solve a related but distinct problem. No existing implementation uses rollback due to its inherent wastefulness.

Require: Draft and Target model Md, Mt, Prompts P, Window

size W, Batch size B

Ensure: Generated sequences S

- 1: Pool ← InitSequencePool(P)
- 2: ▷ Tokenize and optionally sort by length
- 3: Window ← RefillWindow(Pool, W)
- 4: while Pool.hasActive() do
- 5: Phase 1: Lazy Realignment
- 6: B, mask, KV ← GetBatch(Window, B)
- 7: ▷ Try same-length concatenation
- 8: ▷ Fallback to Unpad-Repad Realignment
- 9: Phase 2: Draft Generation
- 10: D ← Md.Generate(B, mask, K)
- 11: Phase 3: Batch Verification
- 12: ▷ See Alg. 3 in Appendix for BatchVerify
- 13: A, B, KV ← BatchVerify(Mt, B, D, KV)
- 14: Phase 4: Write-Back and Window Refill
- 15: for i ∈ B do
- 16: Pool[i] ← Pool[i] ⊕ A[i] ⊕ B[i]
- 17: Pool.KV[i] ← KV[i]
- 18: if isComplete(Pool[i]) then Pool.deactivate(i)
- 19: end for
- 20: Window ← RefillWindow(Pool, W)
- 21: end while
- 22: return Pool.sequences

#### 4.2. Output Equivalence Verification

We verify output equivalence using deterministic greedy decoding to eliminate sampling variance and enable precise bug isolation. This avoids metrics such as ROUGE (Qian et al., 2024), which can mask implementation failures (e.g., repetitive corruption can still score reasonably). Instead, we use exact match (any divergence) and partial match (fraction of tokens before the first mismatch) to diagnose failure modes.

#### 4.1. Experimental Setup

Models. To demonstrate generality, we evaluate three target–draft pairs: Vicuna-7B/68M (Zheng et al., 2023), Qwen3-8B/0.6B (Yang et al., 2025), and GLM-49B/0.6B (GLM et al., 2024). Unless otherwise noted, experiments use NVIDIA A100 80GB GPUs, PyTorch 2.7, HuggingFace Transformers 4.51.3, five draft tokens per speculation round, and greedy decoding for determinism.

Table 1 reveals distinct patterns. Our methods maintain ≈95% exact match across settings, the remaining ≈5% divergence stems from numerical non-determinism in floatingpoint operations (He & Lab, 2025; Gond et al., 2026) and tiebreaking in argmax sampling rather than algorithmic errors. This attribution is corroborated by Liu et al. (2025b), who observe that generation length varies significantly across batch sizes due to kernel-level nondeterminism even in functionally correct implementations. By contrast DSD and BSP fail catastrophically with different signatures. DSD’s near-zero scores indicate immediate position-ID misalignment—the model fails from the first token. BSP shows higher partial match (up to 39.7%) but low exact match, indicating gradual degradation: outputs are initially correct before KV-cache drift misdirects attention, triggering repetition. These complementary patterns—immediate failure vs. gradual decay—reflect distinct root causes: DSD suffers from batch-independent errors (wrong sampling distribution) compounded by synchronization failures, while BSP accumulates KV-cache drift from position-ID desynchronization. A complete taxonomy distinguishing batchindependent bugs from batch-specific synchronization errors appears in Appendix A. We provide additional analysis of

Evaluation and Datasets. We use SpecBench (Xia et al., 2024), a widely adopted benchmark for speculative decoding research. We also use Multi30k (Elliott et al., 2016) for a controlled EXSPEC study that contrasts random sampling with an identical-length subset, isolating sequence-length diversity as the driver of grouping rate. For the main evaluation, we measure: (1) Throughput: tokens/s across batch sizes; and (2) Output Equivalence: exact-match rate (fullsequence equivalence with non-speculative decoding) and partial-match rate (fraction of tokens matching until the first divergence). The partial-match metric helps localize failure modes—early divergence typically indicates position-ID or KV-cache misalignment.

Batch Speculative Decoding Compared. Following our design-space taxonomy (Section 2), we evaluate: (1) Masking approaches: BSP (Su et al., 2023) attempts masking with adaptive speculation but suffers position-ID inconsistencies (BASS (Qian et al., 2024) also follows this approach but requires custom CUDA kernels, limiting gener-

2https://github.com/huggingface/transformers/ issues/32165

Vicuna Qwen3 GLM4 Batch 1 Batch 4 Batch 1 Batch 4 Batch 1 Batch 4

Method

E P E P E P E P E P E P

Non-Spec-Batch – – 53.8 98.2 – – 92.9 96.5 – – 93.3 97.2 Spec-1 97.1 98.4 – – 94.6 97.2 – – 96.0 98.0 – – EQSPEC 97.3 98.6 92.1 98.6 94.6 96.9 92.3 95.7 96.7 98.1 96.5 98.3 EXSPEC 97.3 98.6 90.8 97.6 94.6 96.9 95.0 97.1 96.7 98.1 95.2 97.7 DSD 0.0 8.1 0.0 2.2 0.2 2.2 0.0 0.6 0.0 1.0 0.0 0.8 BSP 1.9 39.7 0.2 31.3 3.5 19.9 2.1 12.6 1.0 15.3 0.6 8.1

- Table 1. Output Equivalence of speculative decoding implementations. Exact match (E) and partial match (P) scores compared against batch=1 non-speculative baseline. Our approach sustains > 95% accuracy, while prior work (DSD, BSP) suffers major drops from KV-cache and position ID errors.

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

40

Random

100

No-Ragged-Scaling

OH/|Spec| (ms)

95.9%

100

All-Mean

RelativeThroughput

89.3%

EQSPEC EXSPEC

OH (%)

84.1%

30

OH(ms)/|Spec|

80

GroupRate(%)

80

70.7%

OH(%)

60

60

20

44.2%

40

40

10

23.6%

17.5%

20

20

4.8%

1.3%

0

0

0

1 2 4 8 16 32

1 2 4 8 16 32

2 4 8 16 32

Batch Size

Batch Size

Batch Size

(a) Batch Throughput Scaling

(b) Batch Realignment Overhead

(c) Cross-Batch Grouping Rate

Figure 5. Decomposing batch speculative decoding performance. (a) Batch scaling efficiency: each method’s throughput normalized to its own BS=1 baseline, isolating scaling behavior from absolute performance. The No-Ragged-Scaling line shows measured autoregressive decoding throughput (not a theoretical bound), serving as a reference for how standard batching scales without ragged tensor overhead. (b) Alignment overhead grows super-linearly with batch size, consuming up to 38% of inference time, validating that coverhead(B) dominates at scale. (c) Cross-batch grouping rates on Multi30k for random vs. uniform-length sequences, showing that length homogeneity transforms grouping effectiveness.

production systems (vLLM, SGLang) in Appendix E.

#### 4.3. Overhead and Scaling Dynamics

We now validate our theoretical predictions through five studies that decompose batch speculative decoding performance. These experiments isolate alignment overhead growth, quantify its impact on batch scaling, identify sequence diversity as the key bottleneck to grouping effectiveness, compare against production systems, and provide mechanistic insights through overhead profiling.

Batch scaling efficiency. Figure 1 shows that EXSPEC outperforms EQSPEC in absolute throughput by combining speculative and batching gains, yet EQSPEC exhibits negative scaling beyond BS=8. To test whether batch speculative decoding can retain GPU-parallelism benefits despite raggedness, Figure 5(a) measures batch scaling efficiency: each method’s throughput at batch size N divided by its own throughput at batch size 1. This normalization isolates how well each method scales with batch size, independent of absolute performance differences. The No-Ragged-Scaling line represents actual measured autoregressive decoding throughput (without speculation), similarly normalized to

its own BS=1 baseline; it is not a theoretical upper bound but rather a reference showing how standard batching scales when no ragged tensor synchronization is required. A notable effect emerges: EXSPEC initially exceeds this reference line. This occurs because each method is normalized independently, and speculation converts memory-bound token generation into compute-bound verification, which benefits more from GPU parallelism. Although EXSPEC may have lower absolute throughput than standard decoding at BS=1 due to draft model overhead, its relative scaling from BS=1 to BS=8 can surpass that of non-speculative decoding. However, at larger batch sizes, alignment overhead dominates and the advantage inverts, confirming that coverhead(B) eventually overwhelms parallelism gains. The key finding is that valid batch speculative decoding need not sacrifice batch scaling efficiency, given that prior methods either produce gibberish outputs or fail to scale.

Alignment overhead growth. Figure 5(b) quantifies realignment costs via two metrics: percentage of total time spent on alignment (OH%) and per-round alignment time (OH/|Spec|). Overhead rises from ∼ 13% at BS=1 to nearly 40% at BS=32, with per-round costs increasing even more. This matches our prediction that cpad(B) + ckv(B) grows

2024) but still verify each request independently. Approximate schemes (Kim et al., 2023; Zhong et al., 2025) trade exactness for speed; our work assumes lossless verification to detect implementation errors rather than approximation artifacts.

Method TPS |Spec| ms/V V% D% OH% EQSPEC 95.6 1469 24.9 44.9 27.4 27.7 EXSPEC 156.4 952 29.1 55.9 29.5 14.6

- Table 2. Overhead anatomy of batch speculation methods. V%=verification, D%=draft, OH%=overhead. Despite slower perverification time, EXSPEC achieves higher throughput by reducing verification calls and alignment overhead.

super-linearly with B. Crucially, this is not merely an implementation inefficiency: the very operations required for algorithmic correctness (unpad–append–repad and KV-cache realignment) become increasingly expensive as sequence lengths diverge.

Grouping rate × sequence-length distribution. Figure 5(c) probes whether cross-batch scheduling limits are algorithmic or circumstantial via Multi30k. Comparing random sampling to an All-Mean subset with identical lengths isolates the bottleneck: sequence diversity, not the method. Random sampling shows grouping rates collapsing with batch size, whereas the All-Mean configuration maintains high grouping effectiveness even at moderate scales, substantially reducing coverhead(B). This contrast suggests that preprocessing strategies (e.g., bucketing, dynamic sorting) can push real workloads toward the ideal, revealing untapped potential when scheduling is paired with workload shaping.

Overhead Anatomy. Table 2 shows that EXSPEC attains higher throughput via a deliberate trade-off. Cross-batch scheduling cuts total verification calls by one-third and halves alignment overhead by grouping same-length sequences. The trade-off is memory locality: dynamic batching scatters KV-cache entries, increasing per-verification latency. Despite slower individual operations, the reduction in operation count yields a 64% overall throughput gain. This balance between operation count and efficiency suggests future work on improving KV-cache locality within the cross-batch framework.

### 5. Related Work

Speculative Decoding. Speculative decoding accelerates LLM inference by verifying draft tokens in parallel. Two verification paradigms dominate: sequence verification—as in SpecDec and follow-ups (Xia et al., 2023; Santilli et al., 2023; Yang et al., 2023; Hooper et al., 2023; Zhang

- et al., 2023; Fu et al., 2024)—which preserves the target model’s output distribution; and tree verification—e.g., Medusa/EAGLE variants (Miao et al., 2024; Spector & Re, 2023; Sun et al., 2023; He et al., 2023; Cai et al., 2024; Li
- et al., 2024a;b; 2025)—which explores multiple branches to raise acceptance rates. Recent works parallelize multiple draft sequences per request (Stewart et al., 2024; Lee et al.,

Batch Speculative Decoding. Extending speculative decoding to batch settings introduces the ragged tensor problem: when sequences accept different numbers of draft tokens, the resulting variable-length tensors break GPUfriendly rectangular operations (Qian et al., 2024). Existing approaches face fundamental limitations detailed in Section 2. Position ID/masking approaches like BSP (Su et al., 2023) suffer from position ID inconsistencies across iterations. Dynamic padding approaches reveal critical implementation errors: both DSD (Yan et al., 2025) and Meta’s recent work (Tang et al., 2025) incorrectly sample bonus tokens from the draft model’s distribution rather than the target model’s, violating the fundamental output equivalence guarantee of speculative decoding. This error compounds with improper KV-cache handling, producing corrupted outputs—BSP generates repetitive tokens while DSD produces <unk> symbols. BASS (Qian et al., 2024) sidesteps these issues through custom CUDA kernels but sacrifices portability. Production systems (vLLM, SGLang) avoid the problem entirely via continuous batching but do not support external draft models at batch scale (Appendix E). These failures reveal that batch speculative decoding is fundamentally challenge—yet no prior work formalizes what invariants must hold. We address this gap by establishing synchronization requirements and providing the first correct implementation.

### 6. Conclusion

We establish that valid batch speculative decoding requires maintaining synchronization invariants, specifically rectangular alignment and position-ID contiguity, which prior implementations fail to preserve. Our theoretical analysis formalizes these invariants and proves that the EQSPEC update rule preserves them across verification rounds, while our cost analysis reveals that alignment overhead grows superlinearly with batch size as an inherent cost of algorithmic correctness. EXSPEC reduces this overhead by exploiting uniform acceptance rates, where realignment cost vanishes, through cross-batch scheduling that dynamically groups same-length sequences. Empirically, our methods achieve 95% exact match across three model families while delivering up to 3× throughput at batch size 8, demonstrating that the synergy between batching and speculation is achievable only through principled synchronization.

### Impact Statement

This work focuses on improving the efficiency of LLM inference without altering model outputs or introducing biases. By providing the first valid batch speculative decoding implementation with output equivalence guarantees, we establish a reliable foundation for future research and production deployment. Prior implementations produced corrupted outputs, limiting their practical utility and potentially misleading subsequent work built upon them. Our approach ensures that acceleration techniques do not compromise model reliability. We will release our implementation as open source upon acceptance, providing the research community with a verified reference implementation for batch speculative decoding.

### References

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple LLM inference acceleration framework with multiple decoding heads. CoRR, abs/2401.10774, 2024. doi: 10.48550/ARXIV.2401.10774. URL https://doi.org/ 10.48550/arXiv.2401.10774.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, JeanBaptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Desmond Elliott, Stella Frank, Khalil Sima’an, and Lucia Specia. Multi30k: Multilingual english-german image descriptions. In Proceedings of the 5th Workshop on Vision and Language, pp. 70–74. Association for Computational Linguistics, 2016. doi: 10.18653/v1/W16-3210. URL http://www.aclweb.org/anthology/W16-3210.

Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of llm inference using lookahead decoding, 2024.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. Chatglm:

- A family of large language models from glm-130b to glm-4 all tools, 2024.

Raja Gond, Aditya K Kamath, Arkaprava Basu, Ramachandran Ramjee, and Ashish Panwar. Llm-42: Enabling determinism in llm inference with verified speculation, 2026. URL https://arxiv.org/abs/2601.17768.

Horace He and Thinking Machines Lab. Defeating nondeterminism in llm inference. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml. 20250910. https://thinkingmachines.ai/blog/defeatingnondeterminism-in-llm-inference/.

Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D. Lee, and Di He. REST: retrieval-based speculative decoding. CoRR, abs/2311.08252, 2023. doi: 10.48550/ARXIV. 2311.08252. URL https://doi.org/10.48550/arXiv. 2311.08252.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Hasan Genc, Kurt Keutzer, Amir Gholami, and Yakun Sophia Shao. SPEED: speculative pipelined execution for efficient decoding. CoRR, abs/2310.12072, 2023. doi: 10.48550/ARXIV.2310.12072. URL https: //doi.org/10.48550/arXiv.2310.12072.

kamilakesbi. Enable speculative decoding with batch size ¿ 1 (github issue #32165). https://github. com/huggingface/transformers/issues/32165,

2024. Hugging Face transformers. Opened 2024-07-23. Accessed 2025-09-24.

Sehoon Kim, Karttikeya Mangalam, Suhong Moon, Jitendra Malik, Michael W. Mahoney, Amir Gholami, and Kurt Keutzer. Speculative decoding with big little decoder. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http: //papers.nips.cc/paper files/paper/2023/hash/ 7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference. html.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Minjae Lee, Wonjun Kang, Minghao Yan, Christian Classen, Hyung Il Koo, and Kangwook Lee. In-batch ensemble drafting: Toward fast and robust speculative decoding for multimodal language models, 2024. URL https:

//openreview.net/forum?id=8o7131Lm83.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pp. 19274–19286. PMLR, 2023.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. In International Conference on Machine Learning, pp. 28935–28948. PMLR, 2024a.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-2: Faster inference of language models with dynamic draft trees. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 7421–7432, 2024b.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE-3: Scaling up inference acceleration of large language models via training-time test, 2025. URL https://arxiv.org/abs/2503.01840.

Xiaoxuan Liu, Jongseok Park, Langxiang Hu, Woosuk Kwon, Zhuohan Li, Chen Zhang, Kuntai Du, Xiangxi Mo, Kaichao You, Alvin Cheung, Zhijie Deng, Ion Stoica, and Hao Zhang. Turbospec: Closed-loop speculation control system for optimizing llm serving goodput, 2025a. URL https://arxiv.org/abs/2406.14066.

Xiaoxuan Liu, Jiaxiang Yu, Jongseok Park, Ion Stoica, and Alvin Cheung. Speculative decoding: Performance or illusion? arXiv preprint arXiv:2601.11580, 2025b.

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, ASPLOS ’24, pp. 932–949, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400703867. doi: 10.1145/3620666.3651335. URL https://doi.org/10.1145/3620666.3651335.

Haifeng Qian, Sujan Kumar Gonugondla, Sungsoo Ha, Mingyue Shang, Sanjay Krishna Gouda, Ramesh Nallapati, Sudipta Sengupta, Xiaofei Ma, and Anoop Deoras. Bass: Batched attention-optimized speculative sampling. In Findings of the Association for Computational Linguistics ACL 2024, pp. 8214–8224, 2024.

Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodol`a. Accelerating transformer inference for translation via parallel decoding. In Anna

Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pp. 12336–12355. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.689. URL https: //doi.org/10.18653/v1/2023.acl-long.689.

SGLang Team. Enabling deterministic inference for sglang. https://lmsys.org/blog/ 2025-09-22-sglang-deterministic/, September 2025. LMSYS Org Blog, published Sep 22, 2025.

Benjamin Spector and Chris Re. Accelerating LLM inference with staged speculative decoding. CoRR, abs/2308.04623, 2023. doi: 10.48550/arXiv.2308. 04623. URL https://doi.org/10.48550/arXiv. 2308.04623.

Lawrence Stewart, Matthew Trager, Sujan Kumar Gonugondla, and Stefano Soatto. The n-grammys: Accelerating autoregressive inference with learning-free batched speculation. arXiv preprint arXiv:2411.03786, 2024.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568: 127063, 2024.

Qidong Su, Christina Giannoula, and Gennady Pekhimenko. The synergy of speculative decoding and batching in serving large language models. arXiv preprint arXiv:2310.18813, 2023.

Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, and Felix X. Yu. Spectr: Fast speculative decoding via optimal transport. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http: //papers.nips.cc/paper files/paper/2023/hash/ 6034a661584af6c28fd97a6f23e56c0a-Abstract-Conference. html.

Bangsheng Tang, Carl Chengyan Fu, Fei Kou, Grigory Sizov, Haoci Zhang, Jason Park, Jiawen Liu, Jie You, Qirui Yang, Sachin Mehta, et al. Efficient speculative decoding for llama at scale: Challenges and solutions. arXiv preprint arXiv:2508.08192, 2025.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, Joe

Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/ 2020.emnlp-demos.6.

Zhaoxuan Wu, Zijian Zhou, Arun Verma, Alok Prakash, Daniela Rus, and Bryan Kian Hsiang Low. Tetris: Optimal draft token selection for batch speculative decoding. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025.

Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, and Zhifang Sui. Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pp. 3909–3925. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023. FINDINGS-EMNLP.257. URL https://doi.org/10. 18653/v1/2023.findings-emnlp.257.

Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics ACL 2024, pp. 7655–7671, Bangkok, Thailand and virtual meeting, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-acl.456. URL https://aclanthology.org/ 2024.findings-acl.456.

Minghao Yan, Saurabh Agarwal, and Shivaram Venkataraman. Decoding speculative decoding. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 6460–6473, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei

Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Seongjun Yang, Gibbeum Lee, Jaewoong Cho, Dimitris S. Papailiopoulos, and Kangwook Lee. Predictive pipelined decoding: A compute-latency trade-off for exact LLM decoding. CoRR, abs/2307.05908, 2023. doi: 10. 48550/ARXIV.2307.05908. URL https://doi.org/10. 48550/arXiv.2307.05908.

Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. Draft & verify: Lossless large language model acceleration via self-speculative decoding. CoRR, abs/2309.08168, 2023. doi: 10.48550/ arXiv.2309.08168. URL https://doi.org/10.48550/ arXiv.2309.08168.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

Meiyu Zhong, Noel Teku, and Ravi Tandon. Speeding up speculative decoding via sequential approximate verification. arXiv preprint arXiv:2502.04557, 2025.

#### Appendix Contents

- A Bug Taxonomy in Prior Implementations
- B Proof of Synchronization Invariants
- C BatchVerify Algorithm Details
- D Latency-Throughput Tradeoff in Online Serving
- E Continuous Batching Systems
- F Discussion
- G Reproducibility Statement
- H Disclose on LLM Usage

### A. Bug Taxonomy in Prior Implementations

We systematically categorize the implementation errors found in prior batch speculative decoding systems to clarify the distinct failure modes and their root causes. Table 3

distinguishes two fundamentally different bug categories based on whether they manifest at batch size 1.

Batch-independent errors. The first category consists of relatively straightforward implementation mistakes—sampling from the wrong distribution or failing to exclude rejected tokens. While these corrupt outputs at any batch size, they could be fixed with careful code review. DSD incorrectly samples the bonus token from the draft model’s distribution rather than the target model’s, violating the fundamental output equivalence guarantee of speculative decoding. BSP’s KV cache retains rejected tokens and relies on framework auto-calculation for position IDs, which produces incorrect values when padding patterns change.

Batch-specific synchronization errors. The second category represents the core contribution of this paper: errors that only manifest when batch size exceeds one. These arise from the ragged tensor problem—when sequences accept different numbers of draft tokens, maintaining synchronized position IDs, attention masks, and KV-cache state across the batch becomes non-trivial. Our synchronization invariants (I1, I2) in Section 3 formalize exactly what must be preserved, and our EQSPEC algorithm provides the first correct implementation.

This taxonomy clarifies that while prior implementations suffer from multiple bug types, our contribution specifically addresses the batch-specific synchronization challenge that has no analog in single-sequence speculative decoding.

- B. Proof of Synchronization Invariants Theorem B.1 (EQSPEC: Preservation of Rectangular Align-

ment). Given a batch of size B, let p(it) and c(it) denote the padding length and content length of sequence i at verifi-

cation step t. If the alignment invariant p(it) + c(it) = L(t) holds for all i ∈ [1,B] at step t, the EQSPEC update rule

guarantees it holds at step t + 1.

Proof. We prove this by induction on the verification step t. Base Case (t = 0): At initialization, all sequences are left-padded to the maximum prompt length L(0). Thus, ∀i,p(0)i + c(0)i = L(0). The invariant holds.

Inductive Step: Assume the invariant holds at step t. During verification, sequence i accepts ai new tokens. The new content length becomes:

c(it+1) = c(it) + ai (4)

The new target batch length is determined by the longest updated sequence:

(c(jt) + aj) (5)

L(t+1) = max

j∈[1,B]

The EQSPEC algorithm calculates the padding offset δi as defined in Eq. (2):

δi = (L(t+1) − L(t)) − ai (6)

The new padding length is updated as p(it+1) = p(it) + δi. We verify the invariant at t + 1 by substituting these terms:

p(it+1) + c(it+1) = (p(it) + δi) + (c(it) + ai)

= p(it) + (L(t+1) − L(t) − ai) + c(it) + ai

= (p(it) + c(it)) − L(t) + L(t+1)

By the inductive hypothesis, p(it) +c(it) = L(t). Substituting this yields:

= L(t) − L(t) + L(t+1)

= L(t+1)

Thus, ∀i,p(it+1) + c(it+1) = L(t+1). The rectangular alignment is preserved.

| |
|---|

- Corollary B.2 (Zero Overhead for Single Sequence). For batch size B = 1, no padding adjustment is required.

Proof. When B = 1, the batch consists of a single sequence with content length c(1t). We can set padding p(1t) = 0 without loss of generality, satisfying the alignment invariant naturally as L(t) = c(1t). Given acceptance a1, the new length is L(t+1) = c(1t) + a1. Substituting into Eq. (2):

δ1 = (L(t+1)−L(t))−a1 = (c(1t)+a1−c(1t))−a1 = 0 (7)

Since δ1 = 0, no unpad-append-repad or KV-cache shifting is required.

| |
|---|

- Corollary B.3 (EXSPEC: Zero Overhead for Uniform Acceptance). If all sequences in a batch accept the same number of tokens k ≥ 1, no padding realignment is required

(δi = 0 for all i), regardless of their initial padding or content lengths.

Proof. Let any sequence i have arbitrary padding p(it) and content c(it). The batch length at step t is defined as L(t) = maxj(p(jt)+c(jt)). Recall that the invariant p(jt)+c(jt) = L(t) holds for all sequences.

If every sequence accepts an identical number of tokens ai = k, then the new content length is c(it+1) = c(it) + k. The new batch length becomes:

(p(jt) + c(jt+1))

L(t+1) = max

j

(p(jt) + c(jt) + k)

= max

j

(p(jt) + c(jt))) + k

= (max

j

= L(t) + k

###### Bug Category Specific Issue System BS=1 BS>1

DSD ✓ ✓

Batch-independent implementation errors

Bonus token sampled from draft model instead of target model

BSP ✓ ✓

KV cache contains rejected tokens; position IDs auto-calculated incorrectly

DSD, BSP ✗ ✓

Ragged tensor synchronization (batch-specific)

Unaligned KV cache across sequences with different accepted lengths

BSP ✗ ✓

Position ID desynchronization across ragged sequences

- Table 3. Taxonomy of implementation errors in batch speculative decoding. Batch-independent errors corrupt outputs at any batch size and could be fixed with careful code review. Ragged tensor synchronization errors are batch-specific and represent the core challenge this paper addresses.

The realignment offset δi is calculated as:

δi = (L(t+1) − L(t)) − ai

= ((L(t) + k) − L(t)) − k

= k − k

= 0

Since δi = 0, the existing padding structure is preserved exactly. No memory reallocation or shifting is required, even if the sequences started with different amounts of padding.

| |
|---|

### C. BatchVerify Algorithm Details

Algorithm 3 implements batch verification through a single forward pass over all draft tokens. On the first iteration, the input concatenates the full sequences with their draft tokens (X ← S ⊕ D) to build the initial KV-cache; subsequent iterations process only the draft tokens since prior context is cached. The target model produces logits for all positions, from which we extract predicted tokens via argmax. Mismatch detection is vectorized: we compare predicted tokens against draft tokens and identify the first disagreement position j per sequence. The accepted tokens A[i] consist of all draft tokens up to (but not including) the first mismatch, while the bonus token B[i] is sampled from the target model’s distribution at the mismatch position—critically, this must come from the target model, not the draft model, to preserve output equivalence. Note that because bonus token sampling occurs after the forward pass completes, the bonus token lacks KV-cache entries and must be included in the next forward pass.

An alternative approach, previously employed by vLLM’s v0 engine, sidesteps variable acceptance lengths through batch expansion: rather than handling ragged outputs, each sequence is duplicated K times with progressively longer draft prefixes (e.g., [t1],[t1,t2],[t1,t2,t3]), all verified in one pass, keeping only the longest correct prefix per sequence. While this maintains rectangular tensors, it incurs K× memory and compute overhead—resources wasted

BS Method Throughput P50 P90 P99

- 1 EQSPEC 15.20 6.70 12.96 16.53

- 1 EXSPEC 15.78 6.08 8.32 9.42

- 2 EQSPEC 26.70 7.96 14.78 19.06

- 2 EXSPEC 30.54 9.75 114.89 134.03

4 EQSPEC 46.11 9.46 16.23 19.13 4 EXSPEC 52.44 8.01 53.21 70.35

8 EQSPEC 76.03 11.26 19.23 22.50 8 EXSPEC 77.54 9.16 19.13 33.93

Table 4. Latency-throughput tradeoff under simulated online serving with heterogeneous request lengths. Throughput is in tokens/s. P50, P90, and P99 denote the 50th, 90th, and 99th percentiles of request completion latency, respectively. Bold values indicate significantly worse tail latencies for EXSPEC.

when early tokens are rejected. This approach was deprecated in vLLM v1 due to GPU memory overflow at scale and incompatibility with CUDA graphs. Our BatchVerify instead accepts the ragged output directly and delegates synchronization to the subsequent unpad-append-repad phase (Section 3.1), avoiding the multiplicative overhead of batch expansion.

### D. Latency-Throughput Tradeoff in Online Serving

To evaluate performance under realistic online serving conditions, we conducted experiments simulating dynamic request arrivals using full multi-turn conversations from SpecBench. We shuffled conversation turns to maximize length diversity and inserted new turns into the next available batch, measuring both throughput (tokens/s) and request completion latency (s) at P50/P90/P99 percentiles. Table 4 compares EQSPEC against EXSPEC across batch sizes 1, 2, 4, and 8.

Both methods achieve positive speedups under heterogeneous multi-turn workloads, with EXSPEC obtaining 2– 14% higher throughput than EQSPEC through cross-batch

grouping of same-length sequences. However, EXSPEC suffers 1.5–7.7× worse P90/P99 latencies when requests are delayed to enable grouping, as early requests must wait for later ones with matching lengths. This latency penalty is particularly severe at smaller batch sizes where grouping success rates are low (23.6% at batch size 2, as shown in Figure 5 c), causing head-of-line blocking that inflates tail latencies to over 100 seconds. In contrast, EQSPEC maintains predictable tail latency with P99 under 23 seconds across all batch sizes, making it suitable for latency-sensitive online serving where service-level objectives must be met. These results demonstrate that the choice between EQSPEC and EXSPEC depends on deployment requirements: EQSPEC provides stable, predictable latency for interactive applications, while EXSPEC maximizes throughput for offline batch processing where individual request latency is less critical. By offering both algorithms with explicit algorithmic correctness guarantees, our work enables practitioners to match their batch speculation strategy to their specific operational constraints.

Algorithm 3 BatchVerify: Single Forward Pass

Require: Target model Mt, Sequences S, Draft tokens D, KV

cache

Ensure: Accepted tokens A, Bonus tokens B

- 1: if first iteration then
- 2: X ← S ⊕ D
- 3: else
- 4: X ← D
- 5: end if
- 6: logits, KVCache ← Mt(X, KVCache)
- 7: pred tokens ← arg max(logits, dim=vocab)

- 8: ▷ Vectorized first mismatch detection
- 9: matches ← (pred tokens = D)

- 10: J ← arg max(¬matches, dim=seq)
- 11: ▷ Ragged shape acceptance, no vectorization
- 12: for each sequence i in batch do
- 13: A[i] ← D[i][: j]
- 14: ▷ Get bonus token from first mismatch
- 15: bonus logit ← logits[i, |S[i]| + j]

- 16: B[i] ← arg max(bonus logit)

- 17: end for
- 18: return A, B, KV Cache

### E. Continuous Batching Systems: A Different Problem Domain Important Caveat

This appendix compares against continuous batching systems (vLLM, SGLang) for completeness, but these systems solve a fundamentally different problem than the static batch speculative decoding addressed in this paper. Continuous batching uses variable-length packing with paged attention, which sidesteps the ragged tensor problem entirely through different architectural choices. The observations below are not claims that these systems are “incorrect” in the same sense as BSP/DSD—rather, they reflect different design tradeoffs, feature scope limitations, and numerical precision characteristics inherent to their architectures.

Architectural differences. Static batching processes a fixed batch of B sequences together, requiring rectangular tensor alignment (Invariant I1) and facing the ragged tensor synchronization challenge when sequences accept different numbers of draft tokens. Continuous batching systems (e.g., vLLM, SGLang) process requests using variablelength packing, sidestepping I1, but still require scattergather across requests and rollback both position-ID and KV-cache for rejected tokens—a one-dimensional analog of alignment. Critically, I2 remains a prerequisite for algorithmic correctness regardless of batching strategy; note that neither vLLM nor SGLang currently supports continuous

batching speculative decoding with external draft models.

Systems evaluated. For informational comparison, we evaluated: vLLM (Kwon et al., 2023) (which subsumes TETRIS (Wu et al., 2025) and TurboSpec (Liu et al., 2025a) as vLLM forks) and SGLang-EAGLE (Zheng et al., 2024; Li et al., 2024b; 2025). We used the vLLM v0 engine because v1 deprecates speculative decoding. SGLang is compatible only with the EAGLE family; we compare Vicuna-7B/EAGLE2 and Qwen3-8B/EAGLE3 (no available weights for GLM-4). We further tested SGLang-EAGLEDeterministic (SGLang Team, 2025), which enables deterministic execution to reduce numerical variance.

Throughput observations. Figure 6 shows throughput characteristics of production frameworks. Note that these systems use continuous batching with dynamically varying effective batch sizes, making direct comparison with static batching infeasible—the x-axis represents “maximum batch size,” not a fixed batch size as in our experiments. The observed patterns (vLLM’s speculative decoding underperforming its baseline at high concurrency, SGLang+EAGLE being slower than non-speculative generation) reflect the inherent challenges of integrating speculation with continuous batching, not the ragged tensor synchronization bugs we identify in BSP/DSD. Independent evaluation by Liu et al. (2025b) confirms these challenges: they report EAGLE speedup degrading from 1.73× to 1.21× as batch size scales from 1 to 128, with verification consuming 42–95% of execution time. vLLM’s v0 engine used batch expansion (a valid but resource-intensive approach) that was deprecated in v1 (vllm-project/vllm#17984) due to K× memory overhead.

Vicuna Qwen3 GLM4 E P E P E P

Method

vLLM + Spec 96.9 98.0 65.6 78.5 72.7 84.5 SGLang + EAGLE 69.8 79.5 47.7 65.4 – – SGLang + EAGLE + Det 85.0 90.0 50.6 69.5 – –

Table 5. Correctness of continuous batching systems. Exact match (E) and partial match (P) scores compared against each system’s own non-speculative mode.

Accuracy observations. Table 5 shows accuracy metrics for vLLM and SGLang. Both systems are accurate on Vicuna but degrade on Qwen3, suggesting modelspecific sensitivities to position encoding or numerical precision. SGLang-EAGLE-Deterministic (SGLang Team, 2025) helps disambiguate the cause: the improvement from 69.8% to 85.0% on Vicuna confirms that most divergences stem from floating-point non-determinism (He & Lab, 2025) rather than algorithmic bugs. Concurrent work by Liu et al. (2025b) reaches similar conclusions using output length correlation, though our exact and partial match metrics provide finer-grained verification by distinguishing complete sequence match from partial divergence. Note that SGLang only supports EAGLE-family drafters, not external draft models.

Comparison considerations. Direct comparison with these systems is complicated by fundamental architectural differences. These frameworks incorporate orthogonal optimizations (CUDA graphs, paged attention, chunked prefilling, continuous batching) independent of speculation. Moreover, continuous batching dynamically varies effective batch size based on request load, exposing only a “maximum batch size” parameter rather than a fixed batch size, making controlled comparisons infeasible. The deprecation of batch expansion also means neither system currently supports speculative decoding with external draft models at scale.

Architectural implications. The contrast between batch expansion and our approach illustrates two strategies for handling ragged tensors: resource duplication versus explicit synchronization. Batch expansion maintains rectangular tensors by duplicating work, trading K× memory for implementation simplicity. Our unpad-append-repad approach instead accepts raggedness and pays synchronization costs only when sequences diverge. Integrating this synchronization-based approach with continuous batching and paged attention remains an open problem for production deployment.

### F. Discussion

Batch parallelism and per-sequence speculative decoding represent two orthogonal acceleration strategies that should,

vLLM (baseline)

1000

Throughput(tokens/s)

vLLM + SpecDec

SGLang (baseline)

800

SGLang + EAGLE

600

400

200

1 2 4 8 16 32

Max Batch Size

Figure 6. Speculative decoding lags non-speculative baselines, with larger batches further degrading throughput.

in principle, multiply together: batching exploits GPU parallelism across sequences while speculation reduces sequential token dependencies within each sequence. However, when combining these techniques in practice, the ragged tensor problem emerges as a fundamental obstacle that breaks this multiplicative relationship. Different sequences accept different numbers of draft tokens, desynchronizing position IDs, attention masks, and KV-cache state across the batch. A correctly implemented batch speculative decoding, as we provide through EQSPEC and EXSPEC, preserves the multiplicative gains but incurs an inherent realignment cost that consumes up to 40% of computation. Critically, batch speculation amplifies per-sequence gains; it cannot create them. If a draft-target pair shows no speedup at batch size 1 due to low acceptance rates or high draft overhead, batching cannot recover what does not exist at the single-sequence level. Our work establishes the synchronization invariants required for algorithmic correctness, quantifies their irreducible costs, and demonstrates through cross-batch scheduling that intelligent system design can mitigate, though not eliminate, the overhead of maintaining output equivalence at scale.

### G. Reproducibility Statement

We provide our complete implementation in the Supplementary Material, including all hyperparameters, experimental configurations, and model specifications detailed in Section 4. Our output equivalence verification framework using exact and partial match metrics enables deterministic validation of both our results and future implementations. All experiments use publicly available models and datasets for

reproducibility.

Model Configurations. We evaluate three target-draft model pairs: Vicuna-7B (lmsys/vicuna-7b-v1.3) with a 68M draft model (double7/vicuna-68m), Qwen3-8B (Qwen/Qwen3-8B) with Qwen3-0.6B (Qwen/Qwen3-0.6B), and GLM-4-9B (zai-org/GLM-4-9B-0414) with a 0.6B draft model (jukofyork/GLM-4.5-DRAFT-0.6B-v3.0). All models were loaded in FP16 precision with greedy decoding (temperature=0, top p=1.0) to ensure deterministic outputs. We use five draft tokens per speculation round across all experiments unless otherwise specified.

Production Systems Configuration. We evaluated two production inference systems for comparison: vLLM version 0.9.1 and SGLang commit c4e314f (the deterministic decoding had not merged to the stable version at the time of submission, so we compiled it from source). For vLLM, we used the V0 engine with speculative decoding enabled, as the V1 engine does not support draft model speculative decoding3. For SGLang, we used EAGLE-based speculation with model-specific draft models: yuhuili/EAGLE-Vicuna7B-v1.3 for Vicuna-7B and Tengyunw/qwen3 8b eagle3 for Qwen3-8B. GLM-4 was not evaluated with SGLang due to the unavailability of compatible EAGLE draft models. We tested both standard SGLang inference and SGLang with deterministic mode enabled (SGLang Team, 2025) to isolate floating-point non-determinism from algorithmic correctness issues.

### H. Disclose on LLM Usage

We used Large Language Models as assistive tools in preparing this manuscript: GPT-5 and Claude Opus 4.1 for polishing writing (grammar correction and sentence restructuring), generating conceptual diagram illustrations, and writing unit tests; NVIDIA/Parakeet-TDT-0.6B-v3 for voice input transcription. LLMs were not used for research ideation, experimental design, data analysis, or scientific conclusions. All core algorithmic implementations and scientific contributions are solely from the authors. We take full responsibility for all content in this paper.

3https://github.com/vllm-project/vllm/issues/ 21797

