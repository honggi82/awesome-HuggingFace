# arXiv:2605.23081v1[cs.LG]21May2026

## ThriftAttention: Selective Mixed Precision for Long-Context FP4 Attention

Joe Sharratt joesharratt29@gmail.com https://github.com/joesharratt1229/ThriftAttention

### Abstract

Efficient attention algorithms are critical to mitigate the quadratic cost of attention in long-context workloads. Prior work utilises block-scaled quantisation techniques on Blackwell GPUs to move attention computation to 4-bit precision to accelerate inference. However, these techniques result in significant quality degradation in long-context settings. We show that the output impact of quantisation error is highly non-uniform and increases with the importance of each query-key interaction, concentrating functionally relevant error in a small number of attention blocks that contain the most important tokens. We propose ThriftAttention, a low-bit attention variant that delivers near-FP16 long-context quality at FP4 inference efficiency. This approach proceeds in two stages. (1) A heuristic rapidly selects a small number of important query-key block pairs for FP16 precision. (2) The selected blocks are computed in FP16 and the remaining blocks in FP4, with both paths merged via online softmax into a single output. We demonstrate across long-context benchmarks and model families that by computing only 5% of query-key blocks in FP16, ThriftAttention recovers on average 89.1% of the FP4→FP16 performance gap. We show ThriftAttention’s advantage grows with sequence length, mitigating the systematic FP4 quality degradation observed at longer contexts. The code is available at https://github.com/joesharratt1229/ThriftAttention.

### 1 Introduction

Efficient inference is critical for the deployment of large language models [Pope et al., 2023, Wan et al., 2024]. Attention [Vaswani et al., 2017, Zhang et al., 2025b] is a key bottleneck in long-context workloads, where its quadratic cost and KV-cache memory traffic dominate execution time [Kwon

- et al., 2023, Patel et al., 2024]. NVIDIA’s Blackwell architecture [NVIDIA Corporation, 2024] introduces native FP4 Tensor Cores that have 4× the arithmetic throughput of the equivalent FP16 instructions on Blackwell GPUs [NVIDIA Corporation, 2026] whilst reducing KV-cache memory traffic by a similar amount. Recent works, including SageAttention3 [Zhang et al., 2025c], exploit this hardware to accelerate attention. This introduces a fundamental tension between inference efficiency and output quality.

Two lines of prior work address parts of this problem but neither fully resolves it. FP4 attention methods accept the quality degradation as a cost of increased throughput. Sparsity methods [Tang

- et al., 2024, Zhang et al., 2025e, 2023] take the approach of identifying important query-key interactions and computing only those interactions. However, sparsity approaches must drop at least 75% of KV blocks to match FP4 latency during the generation phase at inference. Such aggressive sparsity ratios can become a major source of performance degradation in inference-only sparsity methods because the error from omitting blocks entirely is irrecoverable.

Preprint.

100%

25% 50%

20%

FP16

10%

FP16qualitygaprecovered

80%

5%

- 1%
- 2%

60%

40%

0.5%

20%

FP4

0%

1.0× 1.1× 1.2× 1.3× 1.4× 1.5× 1.6× 1.7×

E2E speedup over FP16

|ThriftAttention budget FP4 (SageAttention3)<br><br>| |
|---|
<br><br>FlashAttention-2 (FP16)|
|---|

Figure 1: ThriftAttention approaches FP4 latency while preserving near-FP16 quality. Pareto frontier of negative log-likelihood (NLL) recovery vs inference efficiency at 131k context length (Qwen3-8B). Performance recovery is measured as the percentage of the FP4-to-FP16 NLL gap recovered.

In this work we develop ThriftAttention, a training-free mixed-precision attention mechanism that delivers near-FP16 long-context quality at FP4 inference efficiency. This addresses the degradation of low-bit attention variants at long contexts. We show that functionally relevant quantisation error is not evenly distributed across query-key interactions. Instead, it is concentrated in a small number of interactions where the attention scores are high in magnitude and most important to the final output distribution.

This finding motivates a simple two-stage approach. First, a lightweight heuristic scores each querykey block pair by Sˆij = q¯i · k¯j, where q¯i and k¯j are the token means of each query block qi and key block kj respectively. The top-k highest-scoring blocks are selected for FP16 precision whilst the remaining query-key block computations are assigned to FP4. Attention is computed for both sets of blocks and then combined via online softmax into a single output.

Results. ThriftAttention recovers most of the quality gap between FP4 and FP16 attention whilst retaining the efficiency benefits of FP4 inference. At a 5% FP16 block budget, ThriftAttention recovers on average 89.1% of the FP4→FP16 performance gap. This increases to 91.8% and 92.4% recovery at 10% and 25% respectively. In end-to-end generation, ThriftAttention reduces inference latency by up to 2× at long contexts. Sequence-length analysis demonstrates ThriftAttention’s advantage grows with context length, where uniform FP4 attention degrades most severely.

Contributions. Our work makes the following contributions:

- • We introduce ThriftAttention, a training-free attention approach that computes the most important block interactions in FP16 and the remainder in FP4. To our knowledge, this is the first work to use sub-byte formats in this mixed-precision manner for attention computation.
- • We evaluate ThriftAttention on LongBench-v1, HELMET, RULER, and PG-19 across Llama, Qwen, and Ministral model families, showing that a small FP16 budget recovers most of the FP16 quality while retaining low-bit inference efficiency.

### 2 Related Work

I/O-Efficient Attention. FlashAttention [Dao et al., 2022] introduced tiling to reduce GPU memory I/O, with subsequent versions [Dao, 2024, Shah et al., 2024, Zadouri et al., 2026] improving parallelism and adding hardware-specific optimisations.

[Figure 1]

Figure 2: Overview of ThriftAttention

Quantised Attention. While post-training quantisation is well-established for linear layers [Dettmers et al., 2022, Frantar et al., 2022, Lin et al., 2024, Dettmers et al., 2023, Ashkboos et al., 2024, Liu et al., 2025], its extension to attention remains limited. SageAttention [Zhang et al., 2025d,a] accelerates attention via INT8/FP8 quantisation with outlier smoothing, and SageAttention3 [Zhang et al., 2025c] extends this to FP4 on Blackwell using two-level microscaling. Other works target KV cache compression [Liu et al., 2024, Hooper et al., 2024, Lin et al., 2025] or combine quantised matmuls with sparsity [Kang et al., 2024].

Sparse Attention. Quest [Tang et al., 2024] performs query-aware KV block selection using coordinate-wise min-max bounds. Token-level eviction and selection strategies [Zhang et al., 2023, Xiao et al., 2024, Li et al., 2024, Jiang et al., 2024] reduce the active KV set, while NSA [Yuan et al., 2025] and SLA [Zhang et al., 2026] learn sparse structure during training. SpargeAttn [Zhang et al., 2025e] combines block-sparsity prediction with quantised attention, skipping near-zero blocks before computing the remainder in INT8/FP8. This is the closest prior work to ours but uses sparsity as the primary acceleration mechanism and does not utilise sub-8-bit numeric formats. Other approaches including linear attention [Katharopoulos et al., 2020, Choromanski et al., 2021, Wang et al., 2020, Qin et al., 2024, Yang et al., 2025b] work less effectively where the attention distribution is peaked.

Positioning. ThriftAttention allocates full precision to the most important blocks rather than imposing uniform quantisation. For a given block the error is upper-bounded by FP4 quantisation noise rather than the magnitude of skipped/approximated attention scores as in sparse methods.

### 3 Method

- 3.1 Motivation for ThriftAttention Consider a single query token attending over N keys. The attention output is

N

√

exp(sj) k exp(sk)

d. (1)

, sj = q · kj/

pj vj, pj =

o =

j=1

FP4 quantisation perturbs each score by some ϵj, giving s˜j = sj + ϵj. The first-order output perturbation is

∂o ∂sj

ϵj. (2)

δo = o˜− o ≈

j

From the softmax Jacobian (∂pj/∂sj = pj(1 − pj), ∂pk/∂sj = −pkpj for k ̸= j):

∂o ∂sj

= pj(vj − o). (3) Substituting and taking norms:

|ϵj| · pj · ∥vj − o∥. (4)

∥δo∥ ≤

j

Each key’s error contribution is the product of three terms: |ϵj|, the score quantisation error; pj, the attention weight; and ∥vj − o∥, the value deviation from the output. The pj factor makes this

Layer0, Head0 Layer0, Head4 Layer0, Head12

10−3

0

|[Figure 2]<br><br>|
|---|

[Figure 3]

[Figure 4]

[Figure 5]

Queryblock

20

40

60

Layer15, Head0 Layer15, Head4 Layer15, Head12

0

[Figure 6]

[Figure 7]

[Figure 8]

|−|PPfp16fp4

Queryblock

20

10−4

40

60

Layer30, Head0

Layer30, Head4

Layer30, Head12

0

[Figure 9]

[Figure 10]

[Figure 11]

Queryblock

20

40

60

10−5

0 25 50

0 25 50

0 25 50

Key block

Key block

Key block

Figure 3: Typical FP16→FP4 attention quantisation error, e = |PFP16 − PFP4|, by query/key blocks across layers and heads in Qwen3-8B (seq=4096)

non-uniform such that tokens with large pre-softmax scores produce large pj through the softmax exponential, amplifying their own quantisation error. Conversely low attention scores dampen the effect of a key token’s quantisation error on o˜.

- Figure 3 illustrates this structure. Visualising e = |PFP16 − PFP4| across query-key block pairs for representative layers and heads in Qwen3-8B, the error concentrates in a small number of blocks per query. These are typically near-diagonal blocks and non-initial attention sinks, precisely where attention scores are largest.

This concentration suggests that most of the quality loss from uniform FP4 attention can be recovered by selectively promoting only these high-error blocks to FP16 and keeping the rest in FP4. This is most critical at long contexts, where per-token error compounds across more positions.

#### 3.2 ThriftAttention Algorithm

FP4 quantisation. Let X ∈ RN×d. We quantise X to an FP4 tensor Xq ∈ RN×d together with an FP8 microscale tensor SX ∈ RN×d/16. We use the NVFP4 microscaling format [NVIDIA Corporation, 2024, Rouhani et al., 2023] supported on Blackwell GPUs, in which each element of Xq is stored in E2M1 format and each per-group scale in SX is stored in E4M3 format. This quantisation is applied independently to Q, K, and V , where Q,K,V ∈ RN×d.

Block-importance scoring. We partition Q into Tq = N/Bq blocks and K, V into Tk = N/Bk blocks:

], where Qi ∈ RB

##### Q = [Q1;...;QT

##### ], K = [K1;...;KT

##### ], V = [V1;...;VT

q

k

k

k×d. Let BiQ and BjK denote the token index sets of query block i and key/value block j, respectively. We compute the token-wise mean of each block:

q×d and Kj,Vj ∈ RB

1 Bq

1 Bk

Q¯i =

Qt, K¯j =

Kt.

t∈BjK

t∈BiQ

The importance score for a block pair (i,j) is then

##### Sˆij = Q¯iK¯j⊤.

Algorithm 1 Forward pass of ThriftAttention. Input: Q,K,V ∈ RN×d; block sizes Bq,Bk; FP16 budget k

- 1: Partition Q into {Qi}Ti=1q and K,V into {Kj}T

k

j=1,{Vj}T

k

j=1

- 2: (Qq,SQ) ← fp4(Q); (Kq,SK) ← fp4(K); (V q,SV ) ← fp4(V )
- 3: Compute block means {Q¯i}Ti=1q and {K¯j}T

k

j=1

- 4: for i = 1 to Tq do
- 5: Ti ← TopK({Q¯iK¯j⊤}T

k

j=1,k)

- 6: m(0)i ← −∞; ℓ(0)i ← 0; Oi ← 0
- 7: for j = 1 to Tk do
- 8: if j ∈ Ti then
- 9: Sij ← QiKj⊤/

√

d

- 10: m(ij) ← max(m(ij−1), rowmax(Sij))
- 11: Pij ← exp(Sij − m(ij))
- 12: ℓ(ij) ← em

(j−1)

i −m(ij) ℓ(ij−1) + rowsum( Pij)

- 13: Oi ← diag(em

(j−1)

i −m(ij))Oi + Pij Vj

- 14: else
- 15: Sij ← MatmulFP4(Qqi,Kjq,SQ,i,SK,j)/

√

d

- 16: m(ij) ← max(m(ij−1), rowmax(Sij))
- 17: Pij ← exp(Sij − m(ij))
- 18: ℓ(ij) ← em

(j−1)

i −m(ij) ℓ(ij−1) + rowsum( Pij)

- 19: s(1)P,ij ← rowmax( Pij)/(448 ×6); ( Pij,SP,ij(2) ) ← ϕ( Pij/s(1)P,ij) // two-scale quantisation from SageAttention3
- 20: Oi ← diag(em

(j−1)

i −m(ij))Oi + s(1)P,ij · MatmulFP4( Pij,Vjq,SP,ij(2) ,SV,j)

- 21: Oi ← diag(ℓi(Tk))−1 Oi
- 22: return O = [O1;...;OT

]

q

Mixed-precision attention computation. For each query block i, we select the top-k key blocks Ti = TopK({Sˆij}T

j=1, k). Query-key block pairs are routed to the FP4 or FP16 path:

k

√

QiKj⊤/

d, j ∈ Ti, MatmulFP4(Qqi,Kjq,SQ,i,SK,j)/

√

Sij =

d, j ∈/ Ti. Pij = OnlineSoftmax(Sij).

The output accumulation follows two paths. For j ∈ Ti, we follow the standard FlashAttention2 online softmax [Dao, 2024] procedure. For j ∈/ Ti, the probability block is quantised via the two-level scheme of SageAttention3 [Zhang et al., 2025c]:

( Pij, SP,ij(2) ) = ϕ( Pij/s(1)P,ij), Oi += s(1)P,ij · MatmulFP4( Pij, Vjq, SP,ij(2) , SV,j),

where s(1)P,ij = rowmax( Pij)/(448 × 6). The FP16 and FP4 output updates are merged online. The full procedure is given in Algorithm 1, which presents the non-causal version. For causal LLM attention, we apply the standard causal mask and restrict block selection to causally visible key blocks.

#### 3.3 Implementation and Optimisation on Hardware

Fused Mixed Precision Kernel. The ThriftAttention mechanism is implemented as a single kernel. To reduce register pressure from supporting two precision paths in a single fused kernel, the FP16 query fragments are scoped only to the selected-block phase. The kernel first processes all non-selected KV blocks through the FP4 path using the FP4 query fragments. It then enters a separate FP16 helper routine for the promoted blocks where the FP16 query tile is loaded into registers.

SageAttn3 / FP4

Thrift 5%

Thrift 25%

| |
|---|

| |
|---|

| |
|---|

a) Prefill kernel

b) Decode kernel

- 0

- 1

- 2

- 3

- 4

- 5

SpeedupvsFA2

SpeedupvsFA2

- 0

- 1

- 2

32k 64k 128k

32k 64k 128k

c) End-to-end prefill

d) End-to-end decode

- 0

- 1

- 2

- 0

- 1

- 2

SpeedupvsFP16

SpeedupvsFP16

32k 64k 128k

32k 64k 128k

- Figure 4: Kernel and end-to-end speedups over FlashAttention-2 for Prefill and Decode. B = 1, nheads = 32, D = 128. Qwen3-8B is used for end-to-end generation.

The same shared memory region is aliased for K, V tiles across both precision paths, with doublebuffered FP4 KV tiles to hide the latency of memory loads behind MMA instructions. Warps/CTAs whose KV range contains no top-k blocks bypass the FP16 path entirely, avoiding both the HBM loads of FP16 Q, K, V tiles and the associated register allocation. We implement our mixed precision kernel in CUDA C++.

### 4 Experiments

We validate the efficiency and performance of ThriftAttention across a diverse set of long-context evaluation tasks and model families. We benchmark kernel speed against FlashAttention-2 and SageAttention3, measure downstream accuracy at varying FP16 budgets on three long-context benchmarks, examine how recovery scales with context length on HELMET, and conduct a negative log-likelihood analysis on PG-19. We use block sizes Bq = Bk = 64 for all experiments.

#### 4.1 Efficiency and Effectiveness

Fig. 4 compares ThriftAttention’s kernel and end-to-end generation latency to FlashAttention-2 and FP4 attention (SageAttention3) on an RTX PRO 6000. Note that FlashAttention-2 is the fastest attention variant supported on RTX 6000 Blackwell. For prefill, ThriftAttention achieves up to a 1.7× kernel speedup over FlashAttention-2. These kernel gains translate to consistent end-to-end prefill improvements, reaching roughly 1.2× speedup at 131k context lengths relative to FP16 attention. ThriftAttention’s decode kernels achieve a 3×–5.5× speed-up compared to FlashAttention-2, with minimal overhead compared to full FP4 attention. This translates to a near 2× end-to-end generation speedup over Qwen3-8B when using full FP16 for attention at 131k context length. By loading most blocks from the KV-cache in FP4, ThriftAttention reduces the dominant bottleneck in decoding.

#### 4.2 ThriftAttention Benchmark Evaluation

Experimental setup. We evaluate ThriftAttention on three benchmarks: LongBench-v1 [Bai et al., 2024], HELMET [Yen et al., 2025], and RULER [Hsieh et al., 2024], across five models [Grattafiori et al., 2024, Yang et al., 2025a, Liu et al., 2026]. We compare performance to full FP4 attention (SageAttention3) and FP16.

Results. Table 1 shows the efficacy of the approach across model families and evaluation benchmarks. Promoting only 5% of blocks to FP16 results in an average recovery of 89.1% of the FP4→FP16 performance gap whilst 10% and 25% budgets push this further to 91.8% and 92.4% respectively. This indicates that the marginal quality returned per additional FP16 block yields diminishing returns once the most important blocks have been promoted to FP16. Results in Table

RULER LongBench v1 HELMET Model Method Score Recov. Score Recov. Score Recov.

FP16 0.728 100 0.267 100 0.289 100 FP4 0.321 0.0 0.132 0.0 0.101 0.0 Top-k=5% 0.543 54.5 0.252 88.9 0.247 77.6 Top-k=10% 0.548 55.8 0.249 86.5 0.251 79.6 Top-k=25% 0.577 62.9 0.240 80.0 0.266 87.6

Llama3.2 (3B)

FP16 0.859 100 0.260 100 0.312 100 FP4 0.385 0.0 0.133 0.0 0.132 0.0 Top-k=5% 0.770 81.2 0.271 108.7 0.287 86.1 Top-k=10% 0.778 82.9 0.275 111.8 0.286 85.6 Top-k=25% 0.800 87.6 0.270 107.9 0.291 88.3

Llama3.1 (8B)

FP16 0.795 100 0.260 100 0.249 100 FP4 0.445 0.0 0.134 0.0 0.130 0.0 Top-k=5% 0.749 86.9 0.254 95.2 0.232 85.7 Top-k=10% 0.756 88.9 0.256 96.8 0.238 90.8 Top-k=25% 0.768 92.3 0.257 97.6 0.246 97.5

Qwen3 (4B)

FP16 0.840 100 0.246 100 0.259 100 FP4 0.484 0.0 0.128 0.0 0.127 0.0 Top-k=5% 0.806 90.4 0.237 92.4 0.246 90.2 Top-k=10% 0.815 93.0 0.241 95.8 0.268 106.8 Top-k=25% 0.819 94.1 0.242 96.6 0.246 90.2

Qwen3 (8B)

FP16 0.848 100 0.267 100 0.330 100 FP4 0.548 0.0 0.130 0.0 0.159 0.0 Top-k=5% 0.866 106.0 0.258 93.4 0.329 99.4 Top-k=10% 0.864 105.3 0.262 96.4 0.331 100.6 Top-k=25% 0.865 105.7 0.264 97.8 0.330 100.0

Ministral 3 (8B)

- Table 1: Downstream accuracy of ThriftAttention at varying top-k values across five models on long-context benchmarks. Recovery measures the fraction of the FP4–FP16 gap closed by each method.

2 and Figure 5 provide further evidence of this. Benchmarks differ in their required FP16 budgets. LongBench v1 saturates at 5% and barely moves with larger budgets. RULER scales steadily from

- 5% to 25%. HELMET sits in between and scales unevenly.

#### 4.3 Sequence Length Experiments

Experimental Setup. We evaluate ThriftAttention on the HELMET benchmark across sequence lengths L ∈ {8192,...,131072} and FP16 budgets k/Tk ∈ {5%,10%,25%}.

Per-length recovery. The sequence-level ablation in Table 2 demonstrates that FP4 quality degrades relative to FP16 performance as sequence length increases. Llama3.1-8B drops from 50% FP4 retention at 8k to 32% at 131k and the two Qwen variants and Llama3.2-3B show comparable falls. Relative to FP4, ThriftAttention’s performance improves at longer contexts, increasing from 2.00× FP4 performance at 8k to 2.2× at 131k with a 5% FP16 block budget. This increase in relative improvement over FP4 demonstrates that the approach becomes increasingly valuable as context length grows. Table 5 in Appendix C shows that FP4 is substantially closer to FP16 on shortcontext evaluation benchmarks, further supporting ThriftAttention’s focus on long-context settings. We analyse context length scaling trends at finer granularity in the following section.

#### 4.4 Negative Log-Likelihood Analysis

We conduct negative log-likelihood (NLL) analysis to thoroughly examine the context-length scaling trend in ThriftAttention. We record per-token NLL on 300 documents sampled from PG-19 [Rae et al., 2020] across context lengths and FP16 budgets.

ThriftAttention more effective at long contexts. Figure 5 reveals a systematic degradation in FP4 quality as context length increases. For sequence lengths below 16k, the FP4 ∆NLL remains

8k 16k 32k 65k 131k Model Method Score Recov. Score Recov. Score Recov. Score Recov. Score Recov.

FP16 0.295 100 0.312 100 0.279 100 0.285 100 0.274 100 FP4 0.118 0.0 0.131 0.0 0.081 0.0 0.094 0.0 0.081 0.0 Top-k=5% 0.276 89.3 0.279 81.8 0.247 83.8 0.228 70.2 0.205 64.2 Top-k=10% 0.278 90.2 0.282 83.4 0.245 82.8 0.233 72.8 0.215 69.4 Top-k=25% 0.282 92.7 0.293 89.5 0.257 88.9 0.255 84.3 0.241 82.8

Llama3.2 (3B)

FP16 0.274 100 0.337 100 0.341 100 0.329 100 0.279 100 FP4 0.138 0.0 0.161 0.0 0.149 0.0 0.127 0.0 0.088 0.0 Top-k=5% 0.265 93.4 0.331 96.6 0.312 84.9 0.310 90.6 0.216 67.0 Top-k=10% 0.261 90.4 0.334 98.3 0.318 88.0 0.303 87.1 0.211 64.4 Top-k=25% 0.262 91.2 0.343 103.4 0.323 90.6 0.312 91.6 0.217 67.5

Llama3.1 (8B)

FP16 0.321 100 0.270 100 0.280 100 0.222 100 0.154 100 FP4 0.175 0.0 0.156 0.0 0.162 0.0 0.084 0.0 0.073 0.0 Top-k=5% 0.303 87.7 0.252 84.2 0.279 99.2 0.195 80.4 0.130 70.4 Top-k=10% 0.312 93.8 0.260 91.2 0.278 98.3 0.202 85.5 0.135 76.5 Top-k=25% 0.318 97.9 0.268 98.2 0.275 95.8 0.222 100.0 0.148 92.6

Qwen3 (4B)

FP16 0.277 100 0.301 100 0.272 100 0.232 100 0.214 100 FP4 0.143 0.0 0.139 0.0 0.154 0.0 0.103 0.0 0.096 0.0 Top-k=5% 0.275 98.5 0.294 95.7 0.271 99.2 0.198 73.6 0.192 81.4 Top-k=10% 0.299 116.4 0.320 111.7 0.296 120.3 0.216 87.6 0.209 95.8 Top-k=25% 0.264 90.3 0.298 98.1 0.241 73.7 0.217 88.4 0.210 96.6

Qwen3 (8B)

FP16 0.325 100 0.318 100 0.343 100 0.338 100 0.326 100 FP4 0.141 0.0 0.155 0.0 0.169 0.0 0.174 0.0 0.157 0.0 Top-k=5% 0.323 98.9 0.318 100.0 0.347 102.3 0.327 93.3 0.331 103.0 Top-k=10% 0.328 101.6 0.318 100.0 0.337 96.6 0.334 97.6 0.338 107.1 Top-k=25% 0.323 98.9 0.318 100.0 0.344 100.6 0.331 95.7 0.334 104.7

Ministral 3 (8B)

- Table 2: Per-length HELMET accuracy of ThriftAttention at varying top-k values across five models.

Method FP16 % FP4 % Skipped % Score ThriftAttention 5% 5.0 95.0 0.0 0.599 Sparse Top-k 28.7% 28.7 0.0 71.3 0.036 Quest 28.7% 28.7 0.0 71.3 0.142

- Table 3: Average performance at sequence length 65,536 on a subset of HELMET tasks under matched FP16-equivalent compute. ThriftAttention computes all blocks using mixed precision, whereas sparse methods compute only 28.7% of blocks and skip the remainder.

approximately constant at 0.04 across token positions. Beyond 32k a positional dependence appears, and at 64k and 128k the FP4 ∆NLL degrades to 0.10 for tokens at the end of the sequence. ThriftAttention mitigates this effect, reducing ∆NLL to ≤ 0.02 across context lengths. Consequently the relative reduction in ∆NLL provided by ThriftAttention grows from approximately 2× at 8k to approximately 5× at the end of a 128k sequence.

The widening advantage at longer contexts can be attributed to the increasingly concentrated structure of attention itself. As sequence length grows, the total number of query-key block interactions increases quadratically, while the subset of interactions carrying most of the attention mass grows more slowly. This means at longer contexts the most sensitive query-key interactions occupy a smaller fraction of all possible block interactions. A fixed fractional top-k budget therefore captures an increasingly large share of the attention mass at longer contexts.

#### 4.5 Comparison to Sparse Attention Baselines at Matched Compute

We evaluate ThriftAttention against inference-time sparse-attention approaches to directly compare our mixed-precision technique to sparsity mechanisms. We compare ThriftAttention at 5% to Quest [Tang et al., 2024] and sparse top-k, both running at a sparsity ratio of 71.3%. This yields an equivalent total FLOP budget between sparse approaches and ThriftAttention at a 5% FP16 budget.

N = 2K

N = 4K

NLLdelta(method-FP16)

0.12

0.12

|FP4<br><br>ThriftAttention (5%)<br><br>ThriftAttention (10%) ThriftAttention (25%)<br><br>|
|---|

0.10

0.10

0.08

0.08

0.06

0.06

0.04

0.04

0.02

0.02

0.00

0.00

0 1K 2K

0 1K 2K 3K 4K

Token position

Token position

N = 8K

N = 16K

NLLdelta(method-FP16)

0.12

0.12

0.10

0.10

0.08

0.08

0.06

0.06

0.04

0.04

0.02

0.02

0.00

0.00

0 2K 4K 6K 8K

0 4K 8K 12K 16K

Token position

Token position

N = 32K

N = 64K

NLLdelta(method-FP16)

0.12

0.12

0.10

0.10

0.08

0.08

0.06

0.06

0.04

0.04

0.02

0.02

0.00

0.00

0 8K 16K 24K 32K

0 16K 32K 48K 64K

Token position

Token position

N = 128K

NLLdelta(method-FP16)

0.12

0.10

0.08

0.06

0.04

0.02

0.00

0 32K 64K 96K 128K

Token position

- Figure 5: Per-token negative log-likelihood increase over the FP16 baseline (∆NLL = NLLmethod − NLLFP16), over PG-19 documents across context lengths for Qwen3-8B.

- Table 3 shows that ThriftAttention outperforms inference-time sparsity approaches at equivalent inference efficiency. This supports a core claim of ThriftAttention. Sparse methods can incur large tail errors because missed blocks are removed entirely, whereas ThriftAttention degrades more smoothly by retaining all interactions in low precision. In ThriftAttention, the same missed block remains available in FP4, so its contribution is degraded by quantisation rather than deleted. Thus, at matched compute, preserving the full attention support in FP4 is more effective than sparsifying aggressively and computing only a small subset in FP16.

### 5 Limitations and Future Work

Our current kernel implementation targets consumer Blackwell GPUs. Extending ThriftAttention to data-center Blackwell would allow the approach to exploit SM100 features such as increased asynchrony which may move FP4 performance closer to the 4× theoretical throughput advantage over FP16. Whilst our kernel improves inference latency, it increases KV-cache memory footprint by 28% by storing FP16 and FP4 caches. ThriftAttention is currently designed for inference acceleration only. Existing 4-bit training methods [NVIDIA et al., 2025, Chmiel et al., 2025, Castro et al., 2025] typically retain attention in higher precision. Selectively promoting sensitive interactions in the forward and backward attention computation to FP16 could help address stability issues in sub-byte attention training.

### 6 Conclusion

We introduced ThriftAttention, a selective mixed-precision attention mechanism for long-context FP4 inference. By selectively promoting only a small number of blocks in the attention computation to FP16, we prevent systematic performance degradation in long-context settings. We demonstrate this behaviour across model families and evaluation benchmarks. Overall, selective precision offers a practical path toward long-context inference that approaches FP4 latency while preserving nearFP16 quality.

### References

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-free 4-bit inference in rotated LLMs. In Advances in Neural Information Processing Systems, 2024.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3119–3137. Association for Computational Linguistics, 2024.

Roberto L. Castro, Andrei Panferov, Soroush Tabesh, Oliver Sieberling, Jiale Chen, Mahdi Nikdan, Saleh Ashkboos, and Dan Alistarh. Quartet: Native FP4 training can be optimal for large language models. arXiv preprint arXiv:2505.14669, 2025.

Brian Chmiel, Maxim Fishman, Ron Banner, and Daniel Soudry. FP4 all the way: Fully quantized training of LLMs. arXiv preprint arXiv:2505.19115, 2025.

Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J. Colwell, and Adrian Weller. Rethinking attention with performers. In International Conference on Learning Representations, 2021.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, 2024.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM.int8(): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems, volume 36, 2023.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. KVQuant: Towards 10 million context length LLM inference with KV cache quantization. In Advances in Neural Information Processing Systems, 2024.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua

- Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. In Advances in Neural Information Processing Systems, 2024.
- Hao Kang, Srikant Bharadwaj, James Hensman, Tushar Krishna, Victor Ruhle, and Saravan Rajmohan. TurboAttention: Efficient attention approximation for high throughputs LLMs. arXiv preprint arXiv:2412.08585, 2024.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, pages 5156–5165, 2020.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023. doi: 10.1145/3600006.3613165.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. In Advances in Neural Information Processing Systems, 2024.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration. In Proceedings of Machine Learning and Systems, 2024.

Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, and Song Han. QServe: W4A8KV4 quantization and system co-design for efficient LLM serving. In Proceedings of Machine Learning and Systems, 2025.

Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, et al. Ministral

3. arXiv preprint arXiv:2601.08584, 2026. doi: 10.48550/arXiv.2601.08584. URL https: //arxiv.org/abs/2601.08584.

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. SpinQuant: LLM quantization with learned rotations. In International Conference on Learning Representations, 2025.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 32332–32344. PMLR, 2024.

NVIDIA, Felix Abecassis, Anjulie Agrusa, Dong Ahn, Jonah Alben, Stefania Alborghetti, et al. Pretraining large language models with NVFP4. arXiv preprint arXiv:2509.25149, 2025.

NVIDIA Corporation. NVIDIA blackwell architecture technical overview, 2024. URL https:

##### //resources.nvidia.com/en-us-blackwell-architecture.

NVIDIA Corporation. Parallel Thread Execution ISA, Version 9.2, 2026. URL https://docs.

##### nvidia.com/cuda/parallel-thread-execution/. CUDA Toolkit 13.2.

Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, ´I˜nigo Goiri, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient generative LLM inference using phase splitting. In 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA), pages 118– 132, 2024. doi: 10.1109/ISCA59077.2024.00019.

Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference. In Proceedings of Machine Learning and Systems, 2023.

Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, and Yiran Zhong. Lightning attention-2: A free lunch for handling unlimited sequence lengths in large language models. arXiv preprint arXiv:2401.04658, 2024.

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In International Conference on Learning Representations, 2020.

Bita Darvish Rouhani, Ritchie Zhao, Ankit More, Mathew Hall, Alireza Khodamoradi, Summer Deng, Dhruv Choudhary, Marius Cornea, Eric Dellinger, Kristof Denolf, et al. Microscaling data formats for deep learning. arXiv preprint arXiv:2310.10537, 2023.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. In Advances in Neural Information Processing Systems, 2024.

Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging BIGBench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051, 2023.

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: Query-aware sparsity for efficient long-context LLM inference. In International Conference on Machine Learning, 2024.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, and Mi Zhang. Efficient large language models: A survey. Transactions on Machine Learning Research, 2024.

Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems, 2024.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In International Conference on Learning Representations, 2025b.

Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen. HELMET: How to evaluate long-context language models effectively and thoroughly. In International Conference on Learning Representations, 2025.

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23078–23097. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.acl-long.1126.

Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, and Tri Dao. FlashAttention-4: Algorithm and kernel pipelining co-design for asymmetric hardware scaling. arXiv preprint arXiv:2603.05451, 2026.

Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. SageAttention2: Efficient attention with thorough outlier smoothing and per-thread INT4 quantization. In International Conference on Machine Learning, 2025a.

Jintao Zhang, Rundong Su, Chunyu Liu, Jia Wei, Ziteng Wang, Pengle Zhang, Haoxu Wang, Huiqiang Jiang, Haofeng Huang, Chendong Xiang, et al. Efficient attention methods: Hardwareefficient, sparse, compact, and linear attention, 2025b. URL https://attention-survey. github.io. Preprint.

Jintao Zhang, Jia Wei, Haoxu Wang, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Kai Jiang, Jianfei Chen, and Jun Zhu. SageAttention3: Microscaling FP4 attention for inference and an exploration of 8-bit training. In Advances in Neural Information Processing Systems, 2025c.

Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, and Jianfei Chen. SageAttention: Accurate 8bit attention for plug-and-play inference acceleration. In International Conference on Learning Representations, 2025d.

Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. SpargeAttention: Accurate and training-free sparse attention accelerating any model inference. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 76397–76413. PMLR, 2025e.

Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, Joseph E. Gonzalez, Jianfei Chen, and Jun Zhu. SLA: Beyond sparsity in diffusion transformers via fine-tunable sparse–linear attention. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= eD8IPvNoZB.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song,

Yuandong Tian, Christopher R´e, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2O: Heavyhitter oracle for efficient generative inference of large language models. In Advances in Neural Information Processing Systems, 2023.

### A Heuristic Ablation

We conducted a heuristic ablation of the ThriftAttention heuristic against randomly promoted blocks and diagonal-only selection.

Method Precision mix Score ThriftAttention 5% FP16 / 95% FP4 0.599 Random block selection 5% FP16 / 95% FP4 0.407 Diagonal block selection 5% FP16 / 95% FP4 0.521

Table 4: Heuristic ablation on a subset of HELMET tasks at sequence length 65,536 with a 5% FP16 budget. ThriftAttention outperforms random and diagonal block selection. The HELMET tasks evaluated are json kv, kilt popqa 3, and long narrative qa.

### B Experiment Design

Code and environment. Experiments use CUDA 12.8, PyTorch 2.8.0, and a single NVIDIA RTX PRO 6000 Blackwell GPU with 96GB GPU memory. The full set of reported downstream benchmark evaluations required approximately 600 GPU-hours. The NLL analysis required approximately 5 GPU-hours. We did not use a larger internal cluster or additional undisclosed compute for the reported results.

#### Models.

Paper label Checkpoint ID

Llama3.2 (3B) meta-llama/Llama-3.2-3B Llama3.1 (8B) meta-llama/Llama-3.1-8B Qwen3 (4B) Qwen/Qwen3-4B Qwen3 (8B) Qwen/Qwen3-8B Ministral 3 (8B) mistralai/Ministral-3-8B-Base-2512

Context, and decoding. The maximum context is 131,072 tokens, clamped by the model limit. Qwen3 uses base RoPE up to 32,768 tokens and YaRN beyond that whilst Llama and Ministral use checkpoint RoPE settings. All generation uses greedy argmax decoding.

ThriftAttention settings. All experiments use block sizes Bq = Bk = 64. Target FP16 budgets are 5%, 10%, and 25%. For a sequence with n = ⌊L/64⌋ key blocks, the integer top-k is chosen by

(kn − k(k − 1)/2)/(n(n + 1)/2) ≈ f, rounded and clamped to [1,n].

Benchmarks. LongBench v1 is evaluated through lm-eval-harness on the English subset: narrativeqa, qasper, multifieldqa en, hotpotqa, 2wikimqa, musique, gov report, qmsum, multi news, trec, triviaqa, samsum, passage count, passage retrieval en, lcc, and repobench-p; the reported score is overall.weighted avg. RULER uses the local NVIDIA/RULER port with 100 samples per task-length cell over 13 tasks: 8 NIAH variants, vt, cwe, fwe, qa 1, and qa 2, at lengths {4096,8192,16384,32768,65536,131072}. HELMET uses the local HELMET config snapshot with 50 samples per task-length cell over recall (json kv), RAG (kilt nq, kilt triviaqa, kilt hotpotqa, kilt popqa 3), reranking (msmarco rerank psg), LongQA (narrativeqa, infbench qa, infbench choice), summarization (infbench sum, multi lexsum), and five ICL tasks, at {8192,16384,32768,65536,131072}.

NLL and sequence-length analysis. NLL uses 300 packed sequences from emozilla/pg19, seed 42, with EOS inserted between documents. We evaluate Qwen3-8B at {2048,4096,8192,16384,32768,65536,131072} using teacher-forced next-token cross-entropy; no generation is used. The sequence-length analysis uses the same HELMET settings and greedy decoding as above.

### C Short-context benchmark results

We also evaluate on short-context reasoning and knowledge benchmarks: BBH [Suzgun et al., 2023], MMLU-Pro [Wang et al., 2024], and GSM8K [Cobbe et al., 2021].

BBH(EM) MMLU-Pro(EM) GSM8K-COT(EM) Model Method Score Recov. FP16% Score Recov. FP16% Score Recov. FP16%

FP16 0.470 – 100 0.248 – 100 0.279 – 100 FP4 0.408 0.0 0 0.207 0.0 0 0.178 0.0 0

- Top-k=1 0.465 91.9 10.8 0.233 63.4 8.2 0.282 103.0 6.3
- Top-k=2 0.462 87.1 21.3 0.242 85.4 16.0 0.273 94.0 12.4 Top-k=4 0.465 91.9 40.1 0.242 85.4 30.5 0.264 85.0 24.6

Llama3.2 (3B)

FP16 0.630 – 100 0.359 – 100 0.547 – 100 FP4 0.516 0.0 0 0.286 0.0 0 0.417 0.0 0

Llama3.1 (8B)

- Top-k=1 0.614 86.0 11.1 0.336 68.5 7.2 0.527 84.8 5.2
- Top-k=2 0.619 90.4 21.6 0.345 80.8 17.0 0.520 79.5 10.1 Top-k=4 0.624 94.7 40.6 0.346 82.2 31.6 0.541 95.9 22.9

FP16 0.752 – 100 0.546 – 100 0.880 – 100 FP4 0.675 0.0 0 0.509 0.0 0 0.835 0.0 0 Top-k=1 0.731 72.7 11.5 0.530 56.8 10.2 0.854 43.3 5.4 Top-k=2 0.732 74.0 21.9 0.536 73.0 18.0 0.863 61.7 11.0 Top-k=4 0.726 66.2 42.0 0.537 75.7 33.5 0.861 56.7 21.7

Qwen3 (4B)

FP16 0.795 – 100 0.597 – 100 0.901 – 100 FP4 0.738 0.0 0 0.564 0.0 0 0.868 0.0 0

Qwen3 (8B)

- Top-k=1 0.775 64.9 10.4 0.581 51.5 7.9 0.873 13.9 5.1
- Top-k=2 0.782 77.2 20.2 0.587 69.7 16.7 0.883 46.5 11.3 Top-k=4 0.772 59.6 40.9 0.588 72.7 30.7 0.896 86.0 22.4

- Table 5: Downstream accuracy of ThriftAttention at varying top-k values across four models evaluated on BBH, MMLU-Pro, and GSM8K. Recov. denotes the percentage recovery from FP4 to FP16 performance.

