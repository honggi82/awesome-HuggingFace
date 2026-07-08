## arXiv:2606.06302v2[cs.LG]15Jun2026

# Tangram: Unlocking Non-Uniform KV Cache Compression for Efficient Multi-turn LLM Serving

Hyungmin Kim∗

Hanyang University Seoul, Republic of Korea kong4274@hanyang.ac.kr

Minsoo Kim∗†

Hanyang University Seoul, Republic of Korea minsoo2333@hanyang.ac.kr

Hongseok Kim

Rebellions Republic of Korea hongseok@rebellions.ai

### Abstract

Multi-turn LLM serving accumulates dialogue history whose Key-Value (KV) cache grows with every turn and every user, quickly exceeding the model weights themselves and making memory—not compute—the binding constraint on throughput. Non-uniform KV compression, which allocates heterogeneous budgets across attention heads, preserves accuracy far better than uniform schemes, yet remains impractical: modern serving stacks assume identical KV lengths across heads, so heterogeneity traps freed memory as page fragmentation, spends up to 25% of prefill time reclaiming scattered pages, and skews GPU workloads that inflate decode latency by up to 1.7× or burn 15–20% of each decode step on re-planning. We observe that this heterogeneity need not be discovered at runtime: head-wise retention follows a two-level structural regularity—an input-invariant head ranking with narrowly bounded per-head ratios—that can be calibrated offline from as few as 50 samples. Building on this insight, we present Tangram, a serving framework that statically resolves what prior systems handle dynamically: Budget Reservation fixes each head’s post-compression footprint at scheduling time, eliminating page reclamation; Ragged Paging clusters similar-budget heads into independent page tables, turning fragmentation into reclaimable memory; and Ahead-of-Time Load Balancing precomputes balanced GPU partitions with zero runtime planning. Implemented on vLLM, Tangram serves as a drop-in substrate for existing non-uniform compression methods, matching their accuracy while improving end-to-end throughput by up to 2.6× over the full-KV baseline. Our implementation is publicly available at https://github.com/aiha-lab/TANGRAM.

### 1 Introduction

Multi-turn interaction has become the dominant way users engage with Large Language Models (LLMs): AI assistants now accumulate dialogue history across sessions to deliver

∗Equal contribution †Currently at Apple ‡Corresponding author

Jungwook Choi‡

Hanyang University Seoul, Republic of Korea choij@hanyang.ac.kr

consistent, personalized responses [1, 3, 24, 28]. To avoid recomputing this ever-growing history H𝑡 at every turn [12, 20, 26, 37], serving systems persist attention states in the Key-Value (KV) cache [32]—but the cache grows linearly with every turn and every concurrent user. For Qwen2.532B, with merely 16 concurrent requests, the accumulated KV cache surpasses the size of the model weights themselves within ten conversation turns, and continues to grow unboundedly thereafter (Figure 1(a)). In multi-turn serving, memory capacity—not compute—is the binding constraint on batch size and therefore on throughput.

KV cache compression is the standard remedy, but how the token budget is distributed across attention heads determines whether accuracy survives. Uniform compression [14, 18, 22, 29, 40, 45] forces every head to retain the same number of tokens, ignoring a fundamental property of attention: critical long-range information is concentrated in a small subset of retrieval heads [10], while other heads attend only locally. Truncating all heads equally starves precisely the heads that matter, and accuracy degrades sharply in multi-turn settings (Figure 1(c)). Non-uniform compression [9, 10, 16, 34, 39] instead allocates heterogeneous per-head budgets that mirror this skew, preserving near-original accuracy even under aggressive compression. Thus, non-uniform compression is the right tool for memory-efficient multi-turn serving.

Systemically, however, non-uniform compression is impractical today. The software stack of state-of-the-art serving systems—PagedAttention [19], continuous batching with chunked prefill [2, 42], and optimized attention kernels [5, 6, 41]—is architected end-to-end around a single implicit assumption: all attention heads hold KV caches of identical length. Non-uniform compression violates this assumption at every layer of the stack, resulting in severe overhead. First, the monolithic page of PagedAttention [19] spans all heads at a single uniform length, so the heterogeneous per-head budgets of non-uniform compression cannot be realized on it in the first place; the memory that compression would free instead stays trapped as unrecoverable page fragmentation (§ 3.1.1). Second, because each head’s post-compression

Comp. KVs Retained KVs

| |
|---|

Llama-3.1-8B

250

100

| |
|---|

| | | | |
|---|---|---|---|
|No Un<br><br>|n-uniform iform| | |
| | | | |
| | | | |

200

Uniform KV Compression

Size(GB)

150

||0.7|
|---|
<br><br>|0.6|
|---|
<br><br>|0.5|
|---|
<br><br>|0.3|
|---|
<br><br>|0.2|
|---|
<br><br>|0.1|
|---|
<br><br>|0.4|
|---|
<br><br>|0.2|
|---|
<br><br>|0.1|
|---|
<br><br>|0.1|
|---|
<br><br>|0.1|
|---|
<br><br>|0.1|
|---|
<br><br>|0.9|
|---|
<br><br>|0.8|
|---|
<br><br>|0.7|
|---|
<br><br>|0.6|
|---|
<br><br>|0.5|
|---|
<br><br>|0.4|
|---|
|
|---|

RelativeAccuracy(%)

50

100

HeadIndex

50

0.9 0.7 0.5 0.3 0.1

Qwen2.5-7B

0

5 10 15 20 25 30

100

| | | | |
|---|---|---|---|

Turn Number

Non-Uniform KV Compression

50

300

||0.7|
|---|
<br><br>|0.6|
|---|
<br><br>|0.5|
|---|
<br><br>|0.3|
|---|
<br><br>|0.2|
|---|
<br><br>|0.1|
|---|
<br><br>|0.4|
|---|
<br><br>|0.2|
|---|
<br><br>|0.1|
|---|
<br><br>|0.1|
|---|
<br><br>|0.1|
|---|
<br><br>|0.1|
|---|
<br><br>|0.9|
|---|
<br><br>|0.8|
|---|
<br><br>|0.7|
|---|
<br><br>|0.6|
|---|
<br><br>|0.5|
|---|
<br><br>|0.4|
|---|
|
|---|

0.9 0.7 0.5 0.3 0.1

HeadIndex

Qwen2.5-14B

Size(GB)

200

100

| | | | |
|---|---|---|---|
| | | | |

100

50

0

2 4 8 16 32 64

Context Length

0.9 0.7 0.5 0.3 0.1

# Requests

Relative KV Size

(a) (b) (c)

Figure 1. (a) KV cache size growth for Qwen2.5-32B with the number of conversation turns (top, # requests = 16) or with the number of concurrent requests (bottom, # turns = 10). The dashed line indicates the model weight size. (b) Comparison of uniform (top) and non-uniform (below) KV compression strategies at a 50% target global retention ratio, where the numbers in each box denote the importance score of each KV entry. (c) Comparative accuracy on long-term conversation QA benchmarks [20] using KVzip [16] with Uniform and non-uniform KV compression.

footprint is unknown until the forward pass computes importance scores, the scheduler must over-allocate and then track, reclaim, and remap scattered pages in flight; this controlplane churn consumes up to 25% of prefill execution time (§3.1.2, Figure11). Third,heterogeneous per-head KV lengths skew the workload across GPU SMs: static kernel partitioning [6] suffers stragglers that inflate decode attention latency by up to 1.7×, while dynamic per-layer re-planning [41] burns 15–20% of every decode step on the CPU (§ 3.1.3). In short, the theoretical memory savings of non-uniform compression evaporate inside the serving system—and can even regress end-to-end throughput.

Rather than treating these runtime overheads as inevitable, our profiling reveals a crucial insight: head-wise KV retention exhibits a two-level structural regularity that is intrinsic to the model rather than driven by the input (Figure 5). Specifically, the ranking of attention heads by retention demand is essentially input-invariant, and each head’s absolute retention ratio varies only within a narrow, estimable band. This regularity fundamentally changes the problem landscape. KV cache heterogeneity need not be discovered dynamically; it can be calibrated once, offline, and treated as a static blueprint. Every runtime burden imposed by non-uniformity—page bookkeeping, reclamation, and kernel planning—becomes a deterministic decision resolvable before execution. Notably, prior heterogeneous memory systems either manage heterogeneity at coarser layer granularity [43] or treat each head independently [44]; neither exploits this cross-head structural regularity.

Building on this insight, we present Tangram, a serving framework that turns non-uniform KV compression

from an algorithmic promise into realized system performance. Tangram resolves each of the three bottlenecks with a dedicated, statically planned mechanism. Budget Reservation (§4.1) fixes each head’s budget to its offline-calibrated value, so the scheduler reserves exactly the post-compression pages at scheduling time—eliminating over-allocation and the entire compress-and-reclaim path. Ragged Paging (§4.2) replaces the monolithic page with finer-grained, per-group page tables, clustering heads of similar budget so that freed capacity is physically reclaimable rather than trapped; a vectorized block table keeps the added control-plane cost negligible. Ahead-of-Time (AOT) Load Balancing (§4.3) precomputes a Workload Split Map from the reserved budget profiles, delivering balanced SM utilization without runtime planning. A natural concern is that freezing budgets offline sacrifices the input-adaptivity that makes non-uniform compression accurate; our evaluation shows the opposite: with a small calibrated safety margin, Tangram matches—and occasionally exceeds—the accuracy of the original dynamic implementations across five models and three state-of-theart compression methods (§5.2).

In summary, this paper makes the following contributions:

- • Characterization. We establish a two-level structural regularity in head-wise KV retention—input-invariant head rankingswithnarrowlybounded ratios—enabling offline calibration of non-uniform compression profiles from as few as 50 samples (§3.2).
- • Co-designed system. Exploiting this regularity, Budget Reservation, Ragged Paging, and AOT Load Balancing statically resolve what prior systems handle at runtime: eliminating page reclamation (up to 25% of prefill time), reclaiming 12–25% more memory, and removing per-layer kernel planning (§4).
- • Generality and results. Built on vLLM, Tangram is a drop-in substrate for existing non-uniform methods [7, 9, 15], preserving their accuracy while improving end-to-end throughput by up to 2.6× over the full-KV baseline (§5.3). Our implementation is publicly available.

### 2 Background

#### 2.1 Non-uniform KV Cache Compression

2.1.1 KV Cache Bottleneck in Multi-Turn LLMs. Multiturn interactions have emerged as the dominant LLM workload, where a model must engage with users over extended periods while maintaining contextual coherence. We formalize each exchange as an interaction unit (𝑢𝑖,𝑎𝑖), consisting of a user utterance𝑢𝑖 and the corresponding model response 𝑎𝑖 in a token sequence. The system then maintains a cumulative dialogue history H𝑡 = {(𝑢𝑖,𝑎𝑖)}𝑖𝑡=−11 for each user request, which serves as the essential context for generating the response at turn 𝑡 [13, 37].

Serving systems maintain this history with the Key-Value (KV) cache, which incrementally stores attention states to avoid redundant re-computation of H𝑡 [32]. For an 𝐿-layer, 𝐻-head Transformer, this requires storing Key and Value tensors for every token across all layers and heads, causing the cache size to scale with the length of the accumulated dialogue [11, 17]. Throughout this paper, 𝐻 denotes the number of KV heads: under grouped-query attention (GQA), multiple query heads share one KV head, and KV cache compression operates at KV-head granularity (e.g., 𝐻=8 for Llama-3.1-8B). As the number of concurrent user requests (i.e., batch size) grows and H𝑡 accumulates across turns, this scaling pressure compounds rapidly. As illustrated in Figure 1(a), the KV cache footprint often surpasses the model size even with a few concurrent requests. Consequently, memory capacity—rather than compute—becomes the primary constraint on system throughput, necessitating efficient compression strategies.

1 Flowchart (§ 2.2.1)

2

Block Table (§2.2.1)

:CPU Control

|Request ID|Sequence Length|Block Table|
|---|---|---|
|A|8192|[1, 2, 3, ..., 8192]|
|B|N|[8193, .., 8192+N]|
|...|...|...|

|Scheduler| |
|---|---|
| | |

Waiting Reqs

Running Reqs

| | |
|---|---|
|Req Manager| |

| | |
|---|---|
|Block Pool| |

|GPU Worker|
|---|

3 4

Scheduling Logic (§ 2.2.2) Write to KV Cache Blocks (§2.2.2)

: Scheduling Step N: Chunk Size Context:Input Answer from ... context:

Attn HeadIdx

4K 4K 1 1

Generated KV:

| |Prefill|Prefill|Decode|Decode|
|---|---|---|---|---|

Write to Page (memcpy)

Alloc Page

Paged KV Addr:

...

Block Pool

0x01 0x02 0x03 0x04 0x05 0xNN

5 Attention Kernel (§2.2.3)

||SM 0|
|---|
|
|---|

||SM 1|
|---|
|
|---|

||SM 2|
|---|
|
|---|

||SM 3|
|---|
|
|---|

||SM N|
|---|
<br><br>|
|---|

Attention Metadata

...

Block Table sequence length

# splits = 2

Figure 2. Main components of vLLM.

- 2.1.2 KV Cache Compression. KV compression reduces the cache footprint by retaining only the most critical tokens per head—those that receive high cumulative attention weights and thus contribute most to the attention output. Formally, for a context of 𝑁 tokens, the importance score

R𝐻𝑁) and applies a single global token budget ⌈𝜌𝐻𝑁⌉:

𝐼ℓ = Top(⌈𝜌𝐻𝑁⌉, 𝑠ℓflat). (3)

This yields a highly irregular implicit per-head retention ratio—some heads retain their full history while others are heavily pruned—naturally mirroring the heterogeneous concentration patterns of attention. However, since this perhead ratio is determined by the global top-⌈𝜌𝐻𝑁⌉ selection over flattened scores, it varies with every input and is unknown before the forward pass. Crucially, Figure 1(c) demonstrates that this head-wise budget heterogeneity preserves high conversational accuracy even under aggressive KV size compression, establishing non-uniform KV compression as essential for memory-efficient multi-turn LLM serving.

𝑠ℓ,ℎ ∈ R𝑁 aggregates the attention weights each token receives at headℎ in layer ℓ. Given a target global retention ratio 𝜌 ∈ (0, 1] (the kept-cache fraction), compression selects a set of retained token indices 𝐼ℓ,ℎ and constructs the compressed KV cache accordingly:

𝐾ℓ,ℎ = 𝐾ℓ,ℎ[𝐼ℓ,ℎ, :], 𝑉ℓ,ℎ =𝑉ℓ,ℎ[𝐼ℓ,ℎ, :]. (1) A fundamental property of the attention mechanism, however, is that heads exhibit diverse concentration patterns: some heads sharply concentrate their attention weights on a small subset of tokens, while others distribute them broadly across the context [38, 39], causing the number of critical tokens to vary substantially across heads. How 𝐼ℓ,ℎ is computed under 𝜌 leads to two fundamentally different compression strategies.

#### 2.2 LLM Serving System

State-of-the-art serving frameworks such as vLLM [19] and SGLang [46] rely on a tightly integrated execution pipeline to manage memory and compute. As illustrated in Figure 2, this pipeline is driven by four core components: a scheduler, a block table, KV cache blocks (pages), and the attention kernel, which together enable efficient batching and memory management [2, 21, 31, 32, 42, 48]. Crucially, this entire system structure is built under the implicit assumption that KV cache lengths remain uniform across all attention heads.

Uniform KV compression [22, 29, 45] ignores head-wise diversity by applying a fixed per-head token budget ⌈𝜌𝑁⌉ identically to every head, as shown in the upper panel of Figure 1(b). Compression then selects the top-⌈𝜌𝑁⌉ tokens independently for each head:

𝐼ℓ,ℎ = Top(⌈𝜌𝑁⌉, 𝑠ℓ,ℎ). (2) However, as shown in the lower panel of Figure 1(b), heads naturally exhibit heterogeneous importance distributions across the context, so applying a uniform token budget indiscriminately truncates each head’s context regardless of its actual distribution, discarding tokens that are critical for broadly-attending heads while over-retaining tokens for narrowly-attending ones.

2.2.1 Continuous Batching. The scheduler manages the lifecycleofincomingandactiveuser requests through iterationlevel continuous batching [42]. At each scheduling step, it inspects the current status of the Block Pool of requests to make decisions on admission and execution (❶). Once a request is considered runnable, the scheduler allocates physical pages to accommodate its KV cache via the Block Table, which maps physical page addresses to specific Request IDs (❷). Modern serving systems pair continuous batching with chunked prefill [2], which splits a long context prefill

Non-uniform KV compression [9, 10, 16] removes the perhead budget constraintby enforcing 𝜌 at the layer level. It flattens importancescoresacrossall𝐻 heads (𝑠ℓflat = concatℎ(𝑠ℓ,ℎ) ∈

request into fixed-size token chunks processed over successive scheduling steps and interleaved with other requests’ decode steps (❸). By preventing any single long-context prefill request from monopolizing an iteration, it bounds the time-to-first-token (TTFT) for concurrent requests, making it indispensable for high-throughput serving. For every iteration, the scheduler allocates pages, computes their physical addresses, and adjusts overall KV usage on the host CPU as part of the control plane—all under the implicit assumption of a static, uniform per-token memory cost.

- 2.2.2 PagedAttention. In the LLM Forward Stage (❸–

❹ in Figure 2), the GPU worker writes the generated KV entries into KV cache blocks. Between consecutive forward passes, a scheduling step requests the pages required for the next pass from the Block Pool and precomputes their physical addresses, which the forward pass then uses to store each KV entry non-contiguously in fixed-size blocks, eliminating external memory fragmentation across requests. A key design constraint of PagedAttention [19] is its unified page structure: a single physical block spans all layers and all attention heads simultaneously, holding 𝐿 ×𝐻 ×2×𝑃 ×𝑑 elements for 𝑃 consecutive tokens (the factor 2 for Key and Value, 𝑑 the per-head dimension), making granular headwise compression impossible. As we show in §3.1.1, this very structure creates a new, internal form of fragmentation once per-head KV lengths diverge.

#### 2.2.3 Attention Kernel Optimization. FlashAttention-

- 2 [5] reduces redundant HBM–SRAM traffic via tiled, fused attention computation along the query dimension. For longcontext decoding, FlashDecoding [6] and FlashInfer [41] further introduce KV-dimension parallelism (❺ in Figure 2), where the number of splits determines how each attention head is partitioned and distributed across SMs during decode attention. While FlashDecoding [6] relies on static heuristics for partitioning, FlashInfer [41] employs a runtime planning phase to identify optimal workload strategies. This planning cost can be amortized through plan reuse: since all layers typically share identical KV structures, the system computes a single plan and reuses it across all𝐿 layers to reduce planning overhead.
- 3 Motivation

While non-uniform KV compression preserves multi-turn accuracy, deployingitonproduction systems such as vLLM [19] reveals severe inefficiencies: every pillar of the serving stack described in §2.2 assumes uniform per-head KV lengths. We analyze the three resulting limitations in turn, each motivating one technique in §4.

- (a)

H1 H2 H3 H4 H1 H2 H3 H4... H1 H2 H3 H4 Layer 1 Layer 2 ... Layer L

: Retained KV

| |
|---|

: Fragmentation

Block Table

(vLLM)

|Req ID|# Pages|
|---|---|
|N|4|

Ragged Block Table

(Tangram)

|Req ID|Group Idx|# Pages|
|---|---|---|
|N|1|4|
| |2|1|

- (b)

|Waiting Reqs| |
|---|---|
| | |

|calc usage()<br><br>| |
|---|---|
| | |

Runnable?

|Running Reqs|
|---|

Yes

No (Stall)

Scheduling Workflow

Dynamic Budget Allocation

KV Buffer

Compressed KV Over-allocation

Memcpy()

Static Budget Allocation

KV Buffer

Compressed KV

| |
|---|

Exact Allocation

Memcpy()

PagePool

Free Pages (Extra Cost)

- (c)

// Each Head is split into num splits TBs int head idx = blockIdx.x / num splits; int split idx = blockIdx.x % num splits; int tid = threadIdx.x; // Load partitioned K, V for this split

| |
|---|
|Blk1|

| |
|---|
|Blk3|

| |
|---|
|Blk4|

| |
|---|
|BlkN|

Launch Kernel

...

shared float smem[...];

Blk2

BlkN-1

// Compute Partial Attention Score compute attn(head idx, split idx, tid);

GPU Thread Blocks

Figure 3. Challenges posed by non-uniform KV compression. (a) Monolithic Page Structure: unified pages span all heads, causing page fragmentation (red dashed: pages allocated per request; teal dashed: per-group allocation under Ragged Paging). (b) Page Management Overhead: reclaiming scattered pages at runtime incurs severe control-plane cost. (c) Workload Imbalance: uniform KV splits across thread blocks cause stragglers under different per-head KV lengths.

#### 3.1 Limitations of Existing Systems

- 3.1.1 Monolithic Page Structure. The first limitation arises directly from PagedAttention’s monolithic page structure (§2.2.2). Figure 3(a) illustrates the problem: under nonuniform compression, each head retains a different number of KV entries (filled blocks), yet the block table records only a single page count per request, so every head in every layer is allocated up to the longest-retaining head (red dashed line). The slots between a head’s actual retention and this allocation ceiling (gray blocks) hold entries that are already evicted but can never be returned to the page pool, because a page is shared by all heads and freed only when every head releases it. We term this page fragmentation. Ragged Paging (§4.2) removes this coupling by managing heads with similar retention in independent page tables, whose per-group allocation (teal dashed line) tracks each group’s own maximum and reclaims the fragmented capacity.
- 3.1.2 Page Management Overhead. The second limitation stems from the interaction between non-uniform compression and the scheduler’s control plane (§2.2.1). As illustrated in Figure 3(b), at each scheduling step the scheduler inspects the block pool, promotes waiting requests to the running set, and allocates the pages each request needs for the current forward step. Under non-uniform compression, however, how much each attention head’s KV cache will

Model # Layers Load Balancing Time (ms) Proportion

Qwen2.5-7B 28 5.64 20.20% Qwen3-32B 64 13.27 15.48% Llama-3.1-8B 32 6.48 17.33% Llama-3.1-70B 80 17.13 16.43%

Table 1. Load balancing overhead. Proportion denotes the fraction of the total decode inference step time spent on the load balancing phase.

be compressed—and thus how many pages it needs—is unknown until the forward pass computes token importance scores at runtime. The scheduler must therefore over-allocate and then run a costly “compress-and-reclaim” process: identifying the scattered pages freed by compression, returning them to the block pool, and updating page tables while the request is in flight. This overhead scales linearly with the number of reclaimed pages, consuming up to 25% of total prefill execution time (Figure 11) and directly limiting throughput.

- 3.1.3 Workload Imbalance. The third limitation occurs at the GPU kernel level during decode attention. GPU architectures achieve peak efficiency under the SIMT paradigm only when parallel threads process uniform workloads. Non-uniform KV compression breaks this uniformity in two distinct ways.

Straggler Effect from Static Partitioning. As shown in Figure 3(c), FlashDecoding [6] parallelizes attention by dispatching fixed-size KV chunks to GPU SMs based on a num_splits parameter applied uniformly across all headsa heuristic valid only when every head has the same context length. Under non-uniform compression, per-head KV lengths can differ severely across heads. As illustrated in Figure 4(a–b), this heterogeneity produces highly skewed per-thread-block workloads: blocks mapped to long-context heads become heavy while short-context blocks finish early and idle. The overall decoding step is gated by the few SMs executing the heaviest blocks, increasing decode attention latency by up to 1.7× compared to the uniform baseline under the same total KV cache size (Figure 4(c)).

Prohibitive Cost of Dynamic Rebalancing. FlashInfer [41] addresses workload imbalance through a runtime planning phase before each decoding step. Under uniform settings, plan reuse amortizes this cost: a single plan is computed once and reused across all 𝐿 layers. Non-uniform compression invalidates this optimization—retained KV lengths differ independently across layers, forcing the planner to recompute a unique partition for every layer at every decoding step. As shown in Table 1, this per-layer planning overhead consumes 15–20% of the total decode iteration time, negating the GPU utilization gains that dynamic balancing is meant to provide.

| | | |
|---|---|---|
| |Max| |
| | | |
| | | |
| | | |

| | |Max|
|---|---|---|
| | | |
| | | |
| | | |

6000

KVcache

4000

Size

2000

0

0 32 64 96

0 32 64 96

ThreadBlock Index

ThreadBlock Index

(a) Uniform

(b) Non-uniform

240

Latency(ms)

1.6×

Uniform

200

Attention

160

Non-uniform

1.7×

120

80

1.7×

1.4× 1.5× 1.4×

40

0

1 2 4 8 16 32

# Requests

(c) Decode Latency Overhead

Figure 4. Workload imbalance on decode attention. (a) Uniform KV compression, (b) Non-uniform KV compression, (c) attention latency across different configurations of requests on Qwen3-4B. The dashed line indicates the maximum workload among all thread blocks.

3.2 Key Observation and System Design Prior workhasestablishedthat attention heads in Transformerbased models exhibit model-intrinsic, specialized roles that persist regardless of the input [27, 36, 38, 39]. Our empirical analysis reveals that this intrinsic specialization is directly reflected in the KV retention profile, exhibiting a two-level structural regularity: (1) the ranking of attention heads by retention demand is essentially input-invariant, and (2) each head’s absolute retention ratio varies only within a narrow, estimable range. This implies that KV cache heterogeneity is not fundamentally unpredictable, but a deterministic structure that can be calibrated offline as a static blueprint for scheduling, memory management, and kernel execution.

As shown in Figure 5, per-head retention rates are highly heterogeneous within a layer, yet each head exhibits a largely stable retention level across inputs—the narrow box widths for each head indicate low variance across the 50 samples. While the absolute retention values may shift moderately across tasks—spanning representative long-context subtasks [23] such as summarization (Summary), code QA (Code), and fact retrieval (Retrieve)—each head’s relative retention level within a layer is consistently preserved, and this pattern holds across diverse models. Note that the heads in Figure 5 are ordered by their calibrated budget: the per-task markers of all three workloads rise largely monotonically along this single ordering, rather than reshuffling it, directly visualizing that diverse workloads induce the same head ranking. The low per-head variance further implies that this profile can be estimated reliably offline from only a handful of calibration samples—50 in all our experiments (§5.1)—and generalizes beyond them. While Figure 5 uses KVzip as the scoring method, this regularity is not an artifact of a particular method: repeating the same analysis for various non-uniform compression methods reproduces the same two-level structure for every method (Appendix A).

Retrieve static budget ( + 2 )

Summary

Code

| |
|---|

| |
|---|

| |
|---|

Layer 6

Layer 19

Layer 27

100

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

Llama-3.1-8B

50

0

3 4 2 1 0 6 5 7

7 4 6 1 5 2 0 3

0 3 6 2 4 7 5 1

Per-headretention(%)

Layer 5

Layer 17

Layer 29

100

Gemma-3-12B

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

50

0

0 2 4 3 1 5 7 6

2 6 5 3 0 1 4 7

2 3 7 0 5 4 1 6

Layer 3

Layer 13

Layer 15

100

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

GPT-OSS-20B

50

0

6 5 7 4 3 1 0 2

2 3 1 6 5 0 7 4

1 5 4 3 6 0 2 7

Attention Head (sorted by retained budget)

- Figure 5.Per-headKV retention rates (%) under non-uniform compression using KVzip [16] at a 50% target global retention ratio, across three model families (Llama-3.1-8B, Gemma-3-12B,GPT-OSS-20B)and three SCBench [23]taskssummarization (Summary), code understanding (Code; RepoQA), and fact retrieval (Retrieve; QA-Eng)—shown for three selected layers per model; for the hybrid models (Gemma-3-12B, GPT-OSS-20B), these are full-attention layers. Heads on the x-axis are sorted by their retained budget. Each box aggregates 50 input samples. Dashed lines mark the per-head budget that Tangram reserves with the safety coefficient 𝛼 = 2 (§4.1).

This stable, model-intrinsic retention structure converts all three sources of overhead into statically resolvable properties: each head’s retention ratio can be fixed offline rather than recomputed per input, and heads with similar stable retention levels can be grouped once per model. Together, these enable three deterministic, pre-scheduled design decisions:

- 1. Budget Reservation (§4.1): Fix each head’s retention ratio offline, letting the scheduler reserve exactly the required pages before execution and eliminating the runtime compress-and-reclaim step.
- 2. Ragged Paging (§4.2): Cluster heads with similar retention levels offline into independent page tables, enabling true physical memory reclamation and breaking monolithic page fragmentation.
- 3. Ahead-of-Time (AOT) Load Balancing (§4.3): PrecomputeoptimalGPUworkload partitions offline based on the fixed head group shapes, achieving balanced SM utilization with zero runtime planning overhead.

### 4 Methodology

We present Tangram, a holistic serving framework that reconciles the theoretical efficiency of non-uniform KV cache compressionwiththepracticalconstraints of high-throughput serving. Throughout, compression remains fully fused with chunked prefill and continuous batching rather than reducing memory in isolation. Figure 6 illustrates how the three stages compose.

#### 4.1 Budget Reservation

Our key observation (§3.2) shows that each attention head’s retention follows a model-intrinsic pattern, confined to a narrow, input-invariant range. Tangram leverages this regularity to fix every head’s budget before execution, replacing dynamic, input-dependent compression with a statically planned memory footprint.

Fusing Compression into Prefill. When a request arrives, Tangram executes compression as an integral part of the prefill phase rather than as a separate post-processing pass. As shown in Figure 6(a), each prefill chunk is compressed as soon as its KV cache is generated, fully fusing compression into chunked prefill and continuous batchingthe backbone of modern serving (§2.2). Within each chunk, the compression method’s scoring function rates the importance of every KV entry (Figure 6(b)). Rather than performing the global top-⌈𝜌𝐻𝑁⌉ selection over flattened scores that yields a single index set 𝐼ℓ (§2.1), Tangram replaces the input-dependent implicit per-head ratio with a static perhead budget ratio 𝐵ℓ,ℎ ∈ (0, 1], decomposing compression into per-head top-⌈𝐵ℓ,ℎ𝑁⌉ selections:

𝐼ℓ,ℎ = Top(⌈𝐵ℓ,ℎ𝑁⌉, 𝑠ℓ,ℎ). (4)

Budget Calibration. The per-head budget ratio 𝐵ℓ,ℎ is calibrated once, offline. Running non-uniform compression on a small set of sample contexts under a target global retention ratio 𝜌, we record the implicit per-head retention ratio each head receives from the global top-⌈𝜌𝐻𝑁⌉ selection, summarized by its mean 𝜇ℓ,ℎ and standard deviation 𝜎ℓ,ℎ. Rather than recomputing this ratio for every input, we fix it with a controlled safety margin:

𝐵ℓ,ℎ = min 1, 𝜇ℓ,ℎ + 𝛼 · 𝜎ℓ,ℎ , (5)

where 𝛼 is a safety-margin coefficient. By construction, the per-head budgets sum to approximately the global token budget: ℎ⌈𝐵ℓ,ℎ𝑁⌉ ≈ ⌈𝜌𝐻𝑁⌉, preserving the same overall retention level as the original non-uniform compression. As shown in Figure 6(c), calibration thereby determines, for every head, both the number of pages it receives during prefill and the head group it is managed with (§ 4.2). Since Tangram only fixes the per-head budget 𝐵ℓ,ℎ and leaves each method’s scoring function untouched, it is directly compatible with the importance-scoring function of any KV compression

Embedding

LayerNorm

LayerNorm

FFNLayer

QKVProj

LMHead

forward

OutProj

Attention (Prefill + Decode)

+

+

Request Scheduler Worker

RoPE

(a) Scheduling Policy : Request B Entered

(c) Static Budget Allocation + Cache

(d) Ragged Block Table

(e) AOT Load Balancing

Total KV Length

|GPU SMs<br><br>||G1|G1|G1|G1|
|---|---|---|---|
|G1|G1|G1|G1|
|G2|G2|G2| |
| | | | |
|
|---|
|
|---|

|Group ID|Input Length|Static Budget|# Page Alloc|
|---|---|---|---|
|1|4|100%|2|
|2| |25%|1|

|Ad|Ad, Bp1+c1|Ad, Bp2+c2|Ad, Bp2+c3|Ad, Bp4+c4|
|---|---|---|---|---|

|G1|G2|
|---|---|
|G1|G2|

d: decode, c: compression, p: prefill

Ragged KV: Compute ∝ Effective Length

Execution Timeline

Static Budget Profile

|Req ID|Group ID|Page Addr|
|---|---|---|
|A|1|...|
| |2|...|
|B|1|[0x00, 0x01]|
| |2|[0x02]|

- : Group 1 (H1, H4)

| |
|---|

- : Group 2 (H2, H3)

(b) Prefill + KV Cache Compression

| |
|---|

- Group 1 (H1, H4):

- Group 2 (H2, H3):

BlockPool

0x00 0x01

Load-Balanced Attention

Request B

Who wrote hamlet ?

Who wrote hamlet ?

###### Group Idx Budget Splits

|0.9|
|---|

|0.8|
|---|

|0.9|
|---|

|0.8|
|---|

|0.9|
|---|

|0.8|
|---|

|0.9|
|---|

|0.8|
|---|

0x02

- H1
- H2
- H3
- H4 Full KV Cache

- H1
- H2
- H3
- H4 Non-uniform KV

Top-K

- 1 100% 8
- 2 25% 3

memcpy()

Allocate Page

|0.2|
|---|

|0.3|
|---|

|0.8|
|---|

|0.2|
|---|

|0.8|
|---|

... 50% 5 ... ... ...

|0.3|
|---|

|0.2|
|---|

|0.9|
|---|

|0.3|
|---|

|0.9|
|---|

Compress (40%)

|0.8|
|---|

|0.9|
|---|

|0.9|
|---|

|0.8|
|---|

|0.8|
|---|

|0.9|
|---|

|0.9|
|---|

|0.8|
|---|

𝑁 100% 8

Cached Page:

Ragged Page Table

Split-KV Table

0x00 0x01 0x02

- Figure 6. System overview of Tangram. (a) chunk-wise prefill with per-chunk compression at scheduling time; (b) a scoring function rates each KV entry to drive compression; (c) the static per-head budget fixes the post-compression memory footprint; (d) a ragged block table manages each head group at its own length; (e) an ahead-of-time workload partition balances decode attention across SMs.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| |U|nif|ied| | |
| | | | | | |
| | | | | | |

H1 H2 H3 H4 H5 H6

KVlength(blocks)

|Group|Shared by|# Pages|Free|
|---|---|---|---|
|All|H1–H6|5|0|

(a) Monolithic Paging

| | | |
|---|---|---|
| | | |
|Gr|oup|1|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
|Gr|oup|2|
| | | |
| | | |

H1 H2 H3 H4 H5 H6

|Group|Shared by|# Pages|Free|
|---|---|---|---|
|G1|H1–H3|5|0|
|G2|H4–H6|3|2|

(b) Ragged Paging

| | | |
|---|---|---|
| | | |
|Gr|oup|1|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
|Gr|oup|2|
| | | |
| | | |

H3 H4 H1 H6 H2 H5

|Group|Shared by|# Pages|Free|
|---|---|---|---|
|G1|H3,H4,H1|5|0|
|G2|H6,H2,H5|1|4|

(c) Ragged Paging (Clustered)

- Figure 7. Freeing KV pages under non-uniform KV compression. A page is freed only when all heads sharing it are evicted. (a) Monolithic shares one page across all heads, so a single long head (H3) blocks every page. (b) Ragged

scattered-page reclamation that underlie the Page Management Overhead (§3.1.2), thereby removing its TTFT cost from the serving path entirely.

Robustness and Overflow Handling. Figure 5 overlays the reserved budget 𝐵ℓ,ℎ on the observed retention distributions: it covers each head’s per-input deviation—across all three workloads with distinct attention behaviors—while spending only a marginal amount of extra budget, indicating that calibration captures a model-intrinsic structure rather than the distribution of its pilot samples. The reserved budget is nonetheless a hard capacity bound: if an input demands more retention than 𝐵ℓ,ℎ for some head, the per-head top⌈𝐵ℓ,ℎ𝑁⌉ selection (Eq. 4) simply retains the highest-scoring entries within capacity—equivalent to running the underlying method at a marginally lower ratio for that head—so overflow degrades gracefully into slightly stronger compression, never into a stall or a correctness failure. The end-toend accuracy results in §5.2 (Figure 9) already include any such residual truncation.

pages each head group (𝐻𝑝=3) independently, freeing pages in groups without a long head. (c) Clustered first sorts heads by retention, packing short heads together to free far more pages.

method [7, 15, 16, 22, 30]—each method retains its own importance scoring, while Tangram calibrates the resulting per-head ratios independently into a static profile entirely offline, adding no cost to the serving path.

#### 4.2 Ragged Paging

Tangram stores the KV cache as a ragged structure: each head group is kept at its own retained length rather than padded to a uniform one—analogous to a ragged tensor such as TensorFlow’s RaggedTensor, whose rows need not share a common length.

Precise Page Allocation. Since every per-head budget ratio 𝐵ℓ,ℎ is fixed, a request’s post-compression footprint is fully determined before execution. The scheduler, therefore, reserves exactly the required number of pages when it admits the request: allocation occurs once, at scheduling time, and every page enters the cache already in its postcompression state. This eliminates the over-provisioning and

4.2.1 Paging at Head-Group Granularity. To realize this structure, Tangram narrows the unit of paging from all attention heads jointly to the head group: a set of𝐻𝑝 attention heads whose KV cache is managed by a single page table,

KV Cache

Page Fragmentation

Reclaimed Page

| |
|---|

| |
|---|

40

6.0

6

| |Bloc|k Tab|le| | | |
|---|---|---|---|---|---|---|
| |Vec Page|torize Fra|d Blo gmen|ck tatio|Table n| |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

NormalizedOverhead(x)

#Pages(K)

5.0

Fragmentation(GB)

4

- (a)

Unified Page(vLLM)

0 50 100 150 200 250

Attn Head Index (Sorted)

0

2

4

6

#Pages(K)

- (b)

30

4.0

2

20

3.0

0

2.0

Head Group Page (Hp=4)

10

1.0

0

0.0

1 2 4 8 16 32 64

Heads per Page (Hp)

(c)

- Figure 8. Comparison of Unified Page and Ragged Page on Llama-3.1-8B under 100K single-request non-uniform compression input at a 30% target global retention ratio: (a) Unified Page (vLLM), where all heads share a single page; (b)

Ragged Page (𝐻𝑝 = 4), where each head group maintains its own independent page; and (c) page fragmentation versus management overhead as a function of heads per page 𝐻𝑝.

where 𝐻𝑝 (i.e., the number of heads sharing one page table) sets the granularity. Under this definition, conventional PagedAttention degenerates to a single group spanning all 𝐿×𝐻 heads, which is precisely why a monolithic page cannot release the memory freed by compression (Figure 7(a)).

The physical page shrinks accordingly: instead of spanning all layers and heads, a page holds 𝐻𝑝 × 2 × 𝑃 × 𝑑 elements for the 𝐻𝑝 heads of one group in one layer, and each group owns an independent page table sized by its local maximum budget maxℎ∈Gℓ,𝑖 𝐵ℓ,ℎ rather than the global maximum (Figure 7(b)). The KV cache thereby becomes ragged: every head group is kept at its own effective length—the entries it actually retains—so both a request’s total footprint and its attention computation scale with the sum of per-group effective lengths (Figure 6(d)). Figure 8 confirms this with a real 100K-token request: ragged paging (b) reclaims the capacity evicted from short-retention groups, which a single global group (a) keeps trapped.

- 4.2.2 Budget-AwareClustering. The remaining question is which heads to manage together. All heads in a head-group share one page table sized to the group’s largest budget, so naively grouping adjacent heads mixes dissimilar budgets and grants low-budget heads far more pages than they need. Co-locating heads of similar budgets in the same group is therefore more memory-efficient: each head-group’s footprint tightly tracks what its members actually retain. Because the ranking of head budgets is model-intrinsic and inputinvariant (§3.2, Figure 5), Tangram performs this BudgetAware Clustering once, offline.

For each layer ℓ, we sort all 𝐻 heads by their calibrated budget (§4.1)—that is, let 𝜋ℓ be a permutation of the head

Algorithm 1 AOT Workload Partitioning with Head Group Require: Calibrated budget-ratio tensor B ∈ (0, 1]𝐿×𝐻

where 𝐿 is the number of layers and 𝐻 is the number of KV heads, available Cooperative Thread Arrays (CTAs) 𝑁CTA, heads per page 𝐻𝑝

Ensure: Static Workload Split Map S ∈ N𝐿×(𝐻/𝐻𝑝)

- 1: S ← 1𝐿×(𝐻/𝐻𝑝) ⊲ initialize split factors per head group
- 2: for ℓ ← 1 to 𝐿 do
- 3: Ωℓ ← ℎ 𝐻=1 𝐵ℓ,ℎ ⊲ total budget of layer ℓ
- 4: if Ωℓ = 0 then continue
- 5: end if
- 6: 𝜏ℓ ← Ωℓ/𝑁CTA ⊲ target budget per split
- 7: for 𝑖 ← 0 to 𝐻/𝐻𝑝 −1 do ⊲ iterate over head groups
- 8: Gℓ,𝑖 ← the 𝑖-th head group of layer ℓ ⊲ grouping from §4.2 (clustered or adjacent)
- 9: Φℓ,𝑖 ← ℎ∈Gℓ,𝑖 𝐵ℓ,ℎ ⊲ aggregated group budget
- 10: 𝑆ℓ,𝑖 ← max 1, round(Φℓ,𝑖/𝜏ℓ) ⊲ CTAs ∝ group budget share
- 11: end for
- 12: end for
- 13: return S

indices such that the budget is monotonically increasing: 𝐵ℓ,𝜋ℓ(0) ≤ 𝐵ℓ,𝜋ℓ(1) ≤ · · · ≤ 𝐵ℓ,𝜋ℓ(𝐻−1)

The 𝑖-th head group is then constructed by taking 𝐻𝑝 consecutive heads from this sorted order:

##### Gℓ,𝑖 = {𝜋ℓ(𝑗) | 𝑗 ∈ [𝑖 · 𝐻𝑝, (𝑖 + 1) · 𝐻𝑝 − 1]}

Figure 7(c) shows the effect: packing short-retention heads together frees the most pages. The slack within a group is not wasted: since the shared page table holds KV up to the group’s budget, every head retains at least its own budget, and smaller-budget heads use the leftover room to keep extra KV—free accuracy headroom at zero reclamation cost.

Balancing Memory Gain and Management Overhead. The number of heads per page 𝐻𝑝 governs a fundamental trade-off between memory efficiency and management overhead:

- • Smaller 𝐻𝑝 tightens budget alignment within each group, enabling fine-grained memory management that maximizesreclamation—but it multiplies the number of page tables (up to𝐻 when𝐻𝑝 = 1), so allocation, compression, and block-table updates grow by 𝐻/𝐻𝑝 per request and can bottleneck the host CPU.
- • Larger𝐻𝑝 amortizesthismanagementoverhead across fewer page tables, but can only group heads coarsely, mixing dissimilar budgets and leaving residual fragmentation.

To balance the two, we select an appropriate𝐻𝑝 as the operating point and introduce a Vectorized Block Table that cuts the management overhead of maintaining multiple page tables.

- 4.2.3 Vectorized Block Table Management. Naively, block-table cost scales with the number of groups (O(𝑁𝑟𝑒𝑞 × 𝐻/𝐻𝑝)), making the CPU-side scheduler a bottleneck at finegrained grouping (small𝐻𝑝). We instead aggregate per-group block mappings into a Vectorized Block Table, parallelizing across head groups with OpenMP and processing each group with SIMD intrinsics (e.g., AVX-512). As shown in Figure 8(c), this shifts the fragmentation–overhead trade-off curve down-

ward, enabling small 𝐻𝑝 to maximize memory savings without degrading end-to-end serving throughput.

4.3 Ahead-of-Time (AOT) Load Balancing: Mitigating Workload Imbalance

Finally, we address the workload skew across GPU SMs in decode attention (§ 3.1.3). Because the per-head footprint is static after calibration, the computational load is fully predictable, allowing the entire load-balancing burden to move off the critical path.

- 4.3.1 Ahead-of-Time (AOT) Workload Split Map. As

shown in Figure 6(e), since the per-head budget 𝐵ℓ,ℎ is determined offline and remains constant across requests, the “shape” of the computation is known before inference. We pre-calculate a static Workload Split Map S ∈ N𝐿×(𝐻/𝐻𝑝) to enforce perfectly balanced parallelism.

Static Partitioning Strategy. To maximize hardware utilization, we first leverage the CUDAMaxOccupancy API to determine the total number of these CTAs, denoted as 𝑁CTA, that the target GPU can execute concurrently for the attention kernel. This value represents the device’s aggregate parallelism capacity. We then employ an Ahead-of-Time (AOT) Workload Partitioning algorithm (Algorithm 1) to distribute these CTAs across head groups proportional to their aggregated computational weight. Specifically, each head group’s budget Φℓ,𝑖 is computed by summing the calibrated budget ratios of all heads within the group. Head groups with large aggregated budgets are assigned higher partition factors, while groups with small aggregated budgets are assigned fewer partitions. The output is stored in the static Workload Split Map S, where each entry 𝑆ℓ,𝑖 dictates exactly how many thread blocks should be allocated for head group 𝑖 in layer ℓ. This ensures that the total work assigned to each CTA is approximately equal, thereby eliminating tail latency in which the entire system stalls while waiting for a single overloaded CTA to complete.

Runtime Execution. During decoding, Tangram simply retrieves the precomputed table S to configure the attention kernel, incurring zero runtime planning cost. The static plan remains valid because S is computed from budget ratios, not absolute lengths: every head group’s retained length scales linearly with the same context length 𝑁 (i.e., ⌈𝐵ℓ,ℎ𝑁⌉), so the relative workload across groups—the only quantity that determines balance—is invariant to 𝑁, and batching

preserves it since each request contributes work in the same fixed proportions. The plan thus stays near-optimal across context lengths and batch sizes—precisely what per-layer dynamic planning [41] spends 15–20% of every decode step rediscovering (Table 1).

### 5 Evaluation

#### 5.1 Evaluation Setup

Models and Workloads. We evaluate Tangram on five models spanning dense and Mixture-of-Experts architectures — Qwen3-4B, Llama-3.1-8B, Gemma-3-12B, GPT-OSS-20B, and Qwen3-30B-A3B—each supporting context windows exceeding 100K tokens, which is necessary for capturing the massive context accumulation that arises in multi-turn LLM serving. We adopt SCBench [23], which evaluates longcontext capability through shared-context, multi-turn interactions spanning retrieval, reasoning, summarization, and code understanding—well-suited for stress-testing both the accuracy and efficiency of KV cache management under realistic serving conditions. For Tangram’s budget reservation, we utilize pre-determined budgets derived offline from 50 pilot samples, setting the safety coefficient 𝛼 to 2 across all evaluations. Throughout, we report the target global retention ratio 𝜌 as a percentage, where 𝜌=100% denotes the uncompressed Full-KV cache.

To systematically quantify the performance gains of our framework across varying context scales, we partition the evaluation tasks into three categories: (1) Short (< 20K tokens), represented by the Many-Shot task; (2) Mid (20K–100K tokens), covering RepoQA, Multi-Choice QA, and MathFind tasks that exercise moderate-to-heavy context accumulation and require selective retrieval over substantial histories; and (3) Long (> 100K tokens), encompassing Retrieve PrefixSuffix, KV, and Summary, where the KV cache footprint of even a few requests dominates GPU memory. Together, these workloads comprehensively evaluate the scalability and efficiency of Tangram across the full spectrum of context lengths encountered in multi-turn LLM serving.

SystemSetup. We implementTangramon top ofvLLM [19],

a state-of-the-art high-throughput serving framework. To support our proposed non-uniform KV cache compression, we integrate specialized CUDA kernels developed based on FlashAttention [5], ensuring our custom operators remain fully compatible with standard attention interfaces 1. All end-to-end experiments are conducted on a dedicated server node equipped with an Intel(R) Xeon(R) Gold 6326 CPU @ 2.90 GHz (16 physical cores) and four NVIDIA A100 GPUs (80GB of memory each). We emphasize that all reported results—including throughput, latency, and fragmentation

1Our customized vLLM implementation remains fully compatible with the open-source frameworks and will be released to the community to accelerate innovation.

w/ Tangram w/o Tangram Full-KV

Qwen3-4B

Llama-3.1-8B

Gemma-3-12B

GPT-OSS-20B

Qwen3-30B-A3B

75

50

50

Ada-Snap

50

50

| |
|---|

| |
|---|

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

KV

| |
|---|

25

25

| |
|---|

25

25

| |
|---|

25

0

0

0

0

0

Accuracy(%)

75

50

| | | | |
|---|---|---|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | |
| | | | |
| | | | |

50

Expected

50

Attention

50

| |
|---|

50

| |
|---|

| |
|---|

25

25

25

25

| |
|---|

25

| |
|---|

0

0

0

0

0

75

50

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
|| |
|---|
<br><br>| | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
|| |
|---|
<br><br>| | | |
| | | | |
| | | | |
| | | | |

50

50

50

FastKV

50

| |
|---|

zip

25

25

25

25

25

0

0

0

0

0

0.7 0.5 0.3

0.7 0.5 0.3

0.7 0.5 0.3

0.7 0.5 0.3

0.7 0.5 0.3

Compression ratio (kept fraction)

- Figure 9. Multi-turn accuracy of three non-uniform KV compression methods (rows) across five models (columns), at target global retention ratios 𝜌 of 0.7/0.5/0.3 and averaged over all SCBench tasks. For each method, w/o Tangram (dashed, open markers) is its original implementation, while w/ Tangram (solid, filled markers) is the same method run under Tangram’s Ragged Paging and budget reservation. The horizontal dashed line denotes the Full-KV reference.

0.000

0.150

0.300

0.450

x1.4

Qwen3-4B

Short

0.000

0.003

0.005

0.007 x1.8

Mid

0.000

0.002

0.003

0.005

x2.3

Long

0.000

0.150

0.300

0.450

x1.5

Llama-3.1-8B

0.000

0.006

0.012

0.018 x1.7

0.000

0.003

0.006

0.009 x2.6

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|x|1.1| | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

0.000

0.080

0.160

0.240

Gemma-3-12B

0.000

0.004

0.008

0.012

x1.4

0.000

0.002

0.004

0.006 x1.6

|x|1.1| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

0.000

0.080

0.160

0.240

GPT-OSS-20B

0.000

0.005

0.010

0.015

x1.3

0.000

0.002

0.004

0.006

x1.5

100% 75% 50% 25%

0.000

0.150

0.300

0.450 x1.4

Qwen3-30B-A3B

100% 75% 50% 25%

0.000

0.005

0.010

0.015 x1.7

100% 75% 50% 25%

0.000

0.003

0.005

0.007 x2.1

Throughput(Req/s)

Compression Rate (kept fraction)

vLLM Tangram (only Static Budget)

| |
|---|

| |
|---|

Tangram (w/ Static Budget, Ragged Page) Tangram (w/ Static Budget, Ragged Page, Load Balancing)

| |
|---|

- Figure 10. Throughput breakdown on SCBench [23] across

Baselines. For the non-uniform compression methods, we use their original PyTorch implementations, which dynamically allocate per-head budgets at runtime to meet a target global retention ratio; these provide the w/o Tangram references in §5.2. For load balancing, we compare against two state-of-the-art attention kernels. FlashDecoding [6] parallelizes attention with a Split-KV strategy; effective for uniform caches, its static, heuristic partitioning suffers severe stragglers under the workload skew of non-uniform caches. FlashInfer [41] instead plans partitions at run time, but recomputing a unique plan for every heterogeneous layer incurs significant CPU latency (15–20% of decoding time), negating its GPU parallelism.

Performance Metrics. Our evaluation relies on a comprehensive set of metrics covering both model quality and system efficiency. For Multi-turn LLM Serving capability, we report the average score across Short, Mid, and Long categories for each model, based on the accuracy metrics provided by the benchmark. For system performance, we focus on four key indicators: (1) Throughput (requests per second), which demonstrates overall system capacity under varying load conditions and isolates the contribution of each proposed technique to the end-to-end throughput improvement; (2) Prefill Latency Breakdown to analyze the efficiency gains from Budget Reservation; (3) Decode Attention Latency to validate the effectiveness of AOT Load Balancing; and (4) TTFT to verify serving performance under realistic multi-turn scenarios.

target global retention ratios (𝐻𝑝 = 4). Against the vLLM baseline (𝜌 = 100%), Tangram’s techniques are applied cumulatively: Budget Reservation, then Ragged Paging, then AOT Load Balancing. The red ×N marks Tangram’s peak throughput (𝜌 = 25%) over vLLM.

rates—are empirical measurements obtained from actual runtime execution on this hardware, rather than analytical estimates or simulations.

#### 5.2 Multi-turn Accuracy Evaluation

We first evaluate how Tangram’s memory management affects accuracy. For each non-uniform compression method,

Page Allocation Compression

Model Execution etc.

Page Reclaim

| |
|---|

| |
|---|

| |
|---|

| |
|---|

1000

+10.4% +15.3% +20.0%

+24.9%

600 +13.3% +19.3%

800

600

400

400

200

200

Latency(ms)

0

0

S D S D S D

S D S D S D

75% 50% 25%

75% 50% 25%

Qwen3-4B

Llama-3.1-8B

1600 +7.8% +11.8% +15.7%

1200 +4.6% +7.0% +10.0%

1200

800

800

400

400

0

0

S D S D S D

S D S D S D

75% 50% 25%

75% 50% 25%

Gemma-3-12B

GPT-OSS-20B

- Figure 11. Latency breakdown across various target global retention ratios, comparing static budget allocation (S) against dynamic allocation (D). While dynamic allocation incurs significant page reclamation overhead (orange) that scales with the eviction rate, Tangram’s budget reservation completely eliminates this extra cost, operating without page reclaim overhead.

1 2 4 8 16 32 64

0.006

0.008

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Qwen3-4B

1 2 4 8 16 32 64

0.015

0.02

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Llama-3.1-8B

1 2 4 8 16 32 64

Heads per Page (Hp)

0.008

0.01

0.012

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Gemma-3-12B

1 2 4 8 16 32 64

Heads per Page (Hp)

0.0125

0.015

0.0175

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

GPT-OSS-20B

Throughput(req/s)

Evict 25% Evict 50% Evict 75%

- Figure 12. Throughput versus heads per page (𝐻𝑝); the three curves correspond to eviction rates of 25/50/75%. Small 𝐻𝑝 minimizes fragmentation but incurs high management overhead, while large 𝐻𝑝 fails to effectively reclaim memory.

we compareitsoriginalimplementation (w/o Tangram) against the same method run under Tangram (w/ Tangram), isolating the accuracy impact of Ragged Paging and budget reservation while holding the compression method fixed. As summarized in Figure 9, across various non-uniform compression methods—Ada-SnapKV (Ada-KV’s [9] non-uniform budget allocation over SnapKV’s [22] importance scores), Expected Attention [7], and FastKVzip [15]—and five models, w/ Tangram closely tracks each method’s original w/o Tangram accuracy across target global retention ratios. Notably, although Tangram pins each head group to a static, offlinecalibrated budget, it matches, and in some cases exceeds,

FlashInfer

FlashDecoding

AOT-LB

| |
|---|

| |
|---|

Qwen3-4B

Llama-3.1-8B

20

20

10

10

AttentionLatency(ms)

0

0

100% 75% 50% 25%

100% 75% 50% 25%

Gemma-3-12B

GPT-OSS-20B

20

10

10

5

0

0

100% 75% 50% 25%

100% 75% 50% 25%

Compression Rate (Kept Fraction) (%)

Figure 13. Attention latency evaluated under the impact of AOT (Ahead-Of-Time) load balancing (fixed batch size of 4).

their original accuracy, all while running free of the memory inefficiencies that previously made these methods impractical to serve. This fidelity is by construction: each head’s calibrated budget tracks the retention that the underlying method would assign at runtime, with the safety margin 𝛼 absorbing per-input deviation (§4.1). Tangram thus acts as a faithful systems substrate for non-uniform compression rather than altering the compression itself.

#### 5.3 End-to-end Performance

We evaluate the serving throughput of Tangram against the vLLM baseline, using FastKVzip [15] as the state-of-the-art non-uniform compression method. As shown in Figure 10, Tangram successfully translates non-uniform KV compression into practical system-level gains, achieving up to a 2.6× throughput improvement, with the gain growing as context length increases from Short to Long. This capacity gain also translates into better latency under heavy load: under 75% eviction, Tangram sustains low TTFT as the request rate grows, whereas vLLM’s TTFT rises sharply (Figure 14). To isolate the contribution of each proposed technique, we incrementally apply Budget Reservation(§ 4.1), Ragged Paging(§ 4.2), and AOT Load Balancing(§ 4.3). The results confirm that each component provides additive throughput gains, collectively bridging the gap between theoretical KV cache reduction and realized system performance.

We note that the 2.6× gain over the vLLM Full-KV baseline combines two effects: the capacity freed by compression itself and the system efficiency contributed by Tangram. The cumulative ablation in Figure 10 isolates the latter—moving from Budget Reservation alone to the full system—while Figure 11 quantifies the cost a dynamic implementation would pay instead. Tangram’s contribution is thus not the compression ratio, which is inherited from the underlying method, but the conversion of that ratio into realized throughput.

vLLM Tangram

Qwen3-4B

Llama-3.1-8B

100

100

75

75

50

50

25

25

| |
|---|

| |
|---|

0

0

| |
|---|

| |
|---|

| |
|---|

MeanTTFT(s)

0.0250.050.10.150.20.350.50.751.01.52.0

0.0250.050.10.150.20.350.50.751.01.52.0

Gemma-3-12B

GPT-OSS-20B

100

100

75

75

50

50

25

25

| |
|---|

| |
|---|

0

0

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.0250.050.10.150.20.350.50.751.01.52.0

0.0250.050.10.150.20.350.50.751.01.52.0

RPS (Request Per Second)

Figure 14. TTFT (Time-To-First-Token) under increasing throughput pressure with 30K average request lengths is maintained through budget reservation and Ragged Paging with 75% of the KV cache evicted.

Eliminating Page Reclamation Overhead. As shown in Figure 11, dynamic compression imposes severe overhead, with page reclamation consuming up to 25% of prefill execution time to track and reclaim scattered pages. In contrast, Tangram incurs little extra cost. Because Budget Reservation defines the exact memory footprint before execution, our system allocates only the required pages from the outset, completely eliminating the need to perform any page reclamation.

Impact of Ragged Paging. Ragged Paging is the key mechanism that translates non-uniform compression into actual memory savings by enabling independent page reclamation at the head group level. However, as discussed in §4.2, the choice of heads per page 𝐻𝑝 introduces a fundamental trade-off: excessively small 𝐻𝑝 proliferates the number of page tables, increasing management overhead that degrades system performance, while excessively large 𝐻𝑝 prevents the system from reclaiming evicted pages, as the group’s allocation remains dictated by its longest-retaining head. As shown in Figure 12, we observe that 𝐻𝑝 = 4–8 strikes the optimal balance, yielding the highest end-to-end throughput across all configurations.

Effectiveness of Budget-Aware Clustering. Beyond the group size, which heads share a page table is equally decisive: grouping dissimilar heads inflates each group’s allocation to its largest member and leaves the surplus unreclaimable. Budget-Aware Clustering removes this slack by co-locating heads of similar budget, so each group’s footprint tightly tracks the retention its members actually need. As shown in Figure 15, clustering reclaims an additional 12–25% of the full KV cache over grouping adjacent heads at the same 𝐻𝑝, consistently across diverse models.

w/o Clustering

w/ Clustering

| |
|---|

MemoryReclaimed(%ofFull-KV)

Qwen3-4B

Llama-3.1-8B

100

100

75

75

+22%

+15%

+25%

50

50

+20%

+15%

+13%

25

25

0

0

70% 50% 30%

70% 50% 30%

Gemma-3-12B

GPT-OSS-20B

100

100

75

75

+21%

+20%

+25%

+21%

50

50

+14%

+12%

25

25

0

0

70% 50% 30%

70% 50% 30%

Compression Rate (Kept Fraction) (%)

Figure 15. Memory reclaimed (fraction of the full KV cache) with and without Budget-Aware Clustering (𝐻𝑝 = 4).

Efficient Load Balancing. As shown in Figure 13, our AOT Load Balancing consistently achieves the lowest decode attention latency. FlashDecoding suffers from straggler effects due to its heuristic static partitioning, while FlashInfer incurs significant overhead from recomputing per-layer partitions at every decoding step. Tangram avoids both issues by pre-calculating optimal workload partitions offline, achieving balanced SM utilization without runtime cost.

### 6 Related Works

Multi-turn LLM Serving. As LLMs evolve into persistent assistants, maintaining user-specific context across sessions has become critical [1, 3, 28, 33]. Recent benchmarks like SCBench [23], RealTalk [20] and LoCoMo [26] highlight the difficulty of recalling long-horizon details, motivating algorithmic solutions such as retrieval augmentation [4, 17, 47]. However, prior work largely overlooks the serving efficiency of these memory-intensive workloads. Our work bridges this gap by addressing the system bottlenecks of managing the rapidly scaling KV cache required for robust long-term memory.

KV Cache compression. Compression techniques are essential for reducing memory pressure. Uniform compression enforces uniform retention across all attention heads [14, 18, 22, 29, 40, 45], simplifying management but often discarding context essential for specific heads. In contrast, non-uniform compression improves accuracy by allowing heterogeneous retention budgets [9, 10, 16, 34, 39]. While algorithmically superior, non-uniform methods have been impractical for deployment due to system-level incompatibilities. We identify and resolve the core barriers—fragmentation, scheduling uncertainty, and workload imbalance—to make non-uniform compression viable in production.

Heterogeneous Memory Management. Recent systems have begun to accommodate KV heterogeneity: Jenga [43] manages it across layer types in hybrid models [8, 25, 35] but does not target the explosive multi-turn KV growth, while

DiffKV [44] compresses caches with sparsity and quantization but manages each head independently. Neither exploits the cross-head structural regularity that Tangram identifies; by clustering heads of similar, model-intrinsic retention into ragged pages, Tangram aligns page boundaries with the actual retention distribution, making head-level heterogeneous memory management practical for high-throughput serving.

### 7 Conclusion

We present Tangram, a serving system that brings nonuniform KV cache compression to real LLM serving stacks. Exploiting the model-intrinsic regularity of head-wise retention, Tangram statically resolves what prior systems handle at runtime—reserving exact post-compression budgets at scheduling time, reclaiming fragmented memory through budget-clustered ragged pages, and balancing GPU workloads with a precomputed Workload Split Map. Together, these make non-uniform compression practical, delivering up to 2.6× higher throughput while matching the accuracy of the underlying non-uniform compression methods.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).
- [2] Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S Gulavani, and Ramachandran Ramjee. 2023. Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills. arXiv preprint arXiv:2308.16369 (2023).
- [3] Anthropic. 2024. Using Claude’s Chat, Search, and Memory to Build on Previous Context. https://support.claude.com/en/articles/11817273.
- [4] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413 (2025).
- [5] Tri Dao. 2023. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691 (2023).
- [6] Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov. 2023. Flash-Decoding for long-context inference. https://crfm.stanford.edu/ 2023/10/12/flashdecoding.html.
- [7] Alessio Devoto, Maximilian Jeblick, and Simon Jégou. 2025. Expected attention: Kv cache compression by estimating attention from future queries distribution. arXiv preprint arXiv:2510.00636 (2025).
- [8] Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Shih-Yang Liu, Matthijs Van Keirsbilck, Min-Hung Chen, Yoshi Suhara, et al. 2024. Hymba: A hybrid-head architecture for small language models. arXiv preprint arXiv:2411.13676 (2024).
- [9] Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S Kevin Zhou. 2024. Ada-kv: Optimizing kv cache eviction by adaptive budget allocation for efficient llm inference. arXiv preprint arXiv:2407.11550 (2024).
- [10] Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong, Yue Dong, and Wen Xiao. 2025. Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=FJFVmeXusW
- [11] Ravi Ghadia, Avinash Kumar, Gaurav Jain, Prashant J. Nair, and Poulami Das. 2025. Dialogue Without Limits: Constant-Sized KV Caches for Extended Response in LLMs. In Forty-second International

- Conference on Machine Learning. https://openreview.net/forum?id= SuYO70ZxZX
- [12] Abhiram Rao Gorle, Amit Kumar Singh Yadav, and Tsachy Weissman. 2025. Quantifying Information Gain and Redundancy in MultiTurn LLM Conversations. In First Workshop on Multi-Turn Interactions in Large Language Models. https://openreview.net/forum?id= 5gpABTkcUJ
- [13] Yuanzhe Hu, Yu Wang, and Julian McAuley. 2026. Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions. In The Fourteenth International Conference on Learning Representations. https: //openreview.net/forum?id=DT7JyQC3MR
- [14] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. arXiv:2310.06825 [cs.CL] https://arxiv.org/abs/2310.06825
- [15] Jang-Hyun Kim, Dongyoon Han, and Sangdoo Yun. 2026. Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction. arXiv preprint arXiv:2601.17668 (2026).
- [16] Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W Lee, Sangdoo Yun, and Hyun Oh Song. 2025. KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction. Advances in Neural Information Processing Systems (2025).
- [17] Minsoo Kim, Arnav Kundu, Han-Byul Kim, Richa Dixit, and Minsik Cho. 2025. EpiCache: Episodic KV Cache Management for Long Conversational Question Answering. arXiv:2509.17396 [cs.CL] https: //arxiv.org/abs/2509.17396
- [18] Minsoo Kim, Kyuhong Shim, Jungwook Choi, and Simyung Chang.

2024. InfiniPot: Infinite Context Processing on Memory-Constrained LLMs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and YunNung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 16046–16060. doi:10.18653/v1/2024.emnlp-main.897

- [19] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica.

2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles. 611–626.

- [20] Dong-Ho Lee, Adyasha Maharana, Jay Pujara, Xiang Ren, and Francesco Barbieri. 2025. Realtalk: A 21-day real-world dataset for long-term conversation. arXiv preprint arXiv:2502.13270 (2025).
- [21] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. 2024. {InfiniGen}: Efficient generative inference of large language models with dynamic {KV} cache management. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). 155–172.
- [22] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen.

2024. SnapKV: LLM Knows What You are Looking for Before Generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. https://openreview.net/forum?id=poE54GOq2l

- [23] Yucheng Li, Huiqiang Jiang, Qianhui Wu, Xufang Luo, Surin Ahn, Chengruidong Zhang, Amir H Abdi, Dongsheng Li, Jianfeng Gao, Yuqing Yang, et al. 2024. Scbench: A kv cache-centric analysis of long-context methods. arXiv preprint arXiv:2412.10319 (2024).
- [24] Yubo Li, Xiaobin Shen, Xinyu Yao, Xueying Ding, Yidi Miao, Ramayya Krishnan, and Rema Padman. 2025. Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models. arXiv:2504.04717 [cs.CL] https://arxiv.org/abs/2504.04717
- [25] Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, et al. 2024. Jamba: A hybrid transformer-mamba language model. arXiv preprint arXiv:2403.19887 (2024).

- [26] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating Very Long-Term Conversational Memory of LLM Agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 13851–13870. doi:10.18653/v1/2024.acl-long.747
- [27] Paul Michel, Omer Levy, and Graham Neubig. 2019. Are Sixteen Heads Really Better than One?. In Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.), Vol. 32. Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2019/ file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf
- [28] OpenAI. 2024. Memory and New Controls for ChatGPT. https:// openai.com/index/memory-and-new-controls-for-chatgpt/.
- [29] Matanel Oren, Michael Hassid, Nir Yarden, Yossi Adi, and Roy Schwartz. 2024. Transformers are Multi-State RNNs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 18724–

18741. doi:10.18653/v1/2024.emnlp-main.1043

- [30] Junyoung Park, Dalton Jones, Matthew Morse, Raghavv Goel, Mingu Lee, and Christopher Lott. 2026. Keydiff: Key similarity-based kv cache eviction for long-context llm inference in resource-constrained environments. Advances in Neural Information Processing Systems 38

(2026), 5983–6019.

- [31] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient generative llm inference using phase splitting. In 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA). IEEE, 118–132.
- [32] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2023. Efficiently scaling transformer inference. Proceedings of machine learning and systems 5 (2023), 606–624.
- [33] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024).
- [34] Hanlin Tang, Yang Lin, Jing Lin, Qingsen Han, Danning Ke, Shikuan Hong, Yiwu Yao, and Gongyi Wang. 2025. RazorAttention: Efficient KV Cache Compression Through Retrieval Heads. In The Thirteenth International Conference on Learning Representations. https: //openreview.net/forum?id=tkiZQlL04w
- [35] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118 (2024).
- [36] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. 2019. Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. InProceedings of the 57th Annual Meeting of the Association for Computational Linguistics, Anna Korhonen, David Traum, and Lluís Màrquez (Eds.). Association for Computational Linguistics, Florence, Italy, 5797–5808. doi:10.18653/ v1/P19-1580
- [37] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum? id=pZiyCaVuti
- [38] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. 2025. Retrieval Head Mechanistically Explains Long-Context

- Factuality. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=EytBpUGB1Z
- [39] Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, junxian guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. 2025. DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=cFu7ze7xUm
- [40] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient Streaming Language Models with Attention Sinks. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=NG7sS51zVF
- [41] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, et al.2025. Flashinfer: Efficient and customizable attention engine for llm inference serving. arXiv preprint arXiv:2501.01005

(2025).

- [42] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A distributed serving system for {Transformer-Based} generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). 521–538.
- [43] Chen Zhang, Kuntai Du, Shu Liu, Woosuk Kwon, Xiangxi Mo, Yufeng Wang, Xiaoxuan Liu, Kaichao You, Zhuohan Li, Mingsheng Long, et al.

2025. JENGA: Effective memory management for serving LLM with heterogeneity. In Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles. 446–461.

- [44] Yanqi Zhang, Yuwei Hu, Runyuan Zhao, John CS Lui, and Haibo Chen. 2025. DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction. In Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles. 431– 445.
- [45] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id=RkRrPp7GKO
- [46] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody_Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. 2023. Efficiently Programming Large Language Models using SGLang. arXiv preprint arXiv:2312.07104 (2023).
- [47] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang.

2024. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 19724–19731.

- [48] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024. {DistServe}: Disaggregating prefill and decoding for goodput-optimized large language model serving. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). 193–210.

Retrieve static budget ( + 2 )

Summary

Code

| |
|---|

| |
|---|

| |
|---|

Ada-SnapKV FastKVzip ExpectedAttn

L7

###### L17

###### L21

###### L7

###### L17

###### L21

###### L7

###### L17

###### L21

100

Llama-3.1-8B

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

50

Per-headretention(%)

0

3 2 4 7 1 6 0 5

7 1 0 3 5 4 6 2

1 4 2 3 6 0 7 5

4 7 3 2 1 5 6 0

3 4 2 0 1 5 6 7

4 1 3 5 7 6 2 0

2 3 4 6 5 7 0 1

3 0 7 5 2 1 4 6

3 6 0 4 7 1 5 2

L17

###### L29

###### L41

###### L17

###### L29

###### L41

###### L17

###### L29

###### L41

100

Gemma-3-12B

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

50

0

7 1 4 0 5 3 6 2

3 2 5 1 4 0 6 7

3 0 7 1 6 5 2 4

2 6 5 3 1 0 4 7

2 3 7 0 4 5 1 6

3 7 5 0 4 2 6 1

7 3 0 4 6 2 5 1

2 7 6 3 5 0 4 1

4 5 3 1 6 0 7 2

L7

###### L11

###### L23

###### L7

###### L11

###### L23

###### L7

###### L11

L23

100

GPT-OSS-20B

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |

50

0

5 2 4 0 6 3 7 1

1 0 7 3 5 4 6 2

1 3 4 0 6 2 7 5

1 2 5 3 4 7 6 0

2 4 3 0 7 6 1 5

3 6 1 5 7 4 2 0

5 1 3 7 2 4 0 6

4 0 2 3 6 5 7 1

1 3 4 5 7 2 0 6

Attention Head (sorted by retained budget)

Figure 16. Per-head KV retention rates (%) under three non-uniform compression methods—Ada-SnapKV [9, 22], FastKVzip [15], and Expected Attention [7]—at a 50% target global budget, across three model families (rows: Llama-3.1-8B, Gemma-3-12B, GPT-OSS-20B) and three SCBench [23] tasks including summarization (Summary), code understanding (Code; RepoQA), and fact retrieval (Retrieve; QA-Eng), shown for three layers per model; for the hybrid models (Gemma-3-12B, GPT-OSS-20B), these are full-attention layers. Within each panel, heads are sorted by retained budget, and each box shows the spread of one head’s retention across the input samples of one task.

### A Retention Regularity across Compression Methods

Figure 5 (§3.2) establishes the two-level structural regularity of head-wise KV retention using KVzip [16] as the scoring method. Figure 16 repeats the same rank-box analysis for the three non-uniform compression methods integrated in our evaluation (§5.1)—Ada-SnapKV, FastKVzip, and Expected Attention—across the same three model families and SCBench tasks.

The regularityisconsistentlyreproduced for every method:

within each panel, each head’s per-task boxes are narrow and remain aligned across tasks, confirming that the head ranking is input-invariant and that each head’s absolute retention ratio varies only within a narrow band. Accordingly, the overlaid static budget (𝜇+2𝜎) covers each head’s observed spread

while spending only a marginal amount of extra budget, for every method. At the same time, the shape of the retention profile is clearly method-specific: Ada-SnapKV spreads the budget relatively evenly across heads, whereas FastKVzip and Expected Attention concentrate retention on a small subset of heads, leaving the remaining heads heavily pruned. This is precisely the setting Tangram is designed for: budget calibration (§4.1) is performed per model and per method, so it faithfully captures whatever retention profile a scoring method induces, while the input-invariance demonstrated here guarantees that the offline-calibrated budgets transfer to unseen inputs. The key observation in §3.2 is therefore a property of non-uniform KV compression itself, not an artifact of a particular scoring method.

