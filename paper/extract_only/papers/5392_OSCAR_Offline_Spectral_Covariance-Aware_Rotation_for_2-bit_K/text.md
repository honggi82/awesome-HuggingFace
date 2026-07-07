# arXiv:2605.17757v1[cs.LG]18May2026

[Figure 1]

## OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization

Zhongzhu Zhou∗,1,2 Donglin Zhuang∗,1,2 Jisen Li1,3 Ziyan Chen2 Shuaiwen Leon Song1 Ben Athiwaratkun1 Xiaoxia Wu†,1

1Together AI 2University of Sydney 3University of Illinois Urbana-Champaign ∗Equal contribution. †Corresponding author.

### Abstract

INT2 KV-cache quantization is attractive for long-context LLM serving, but it remains difficult to make both accurate and deployable. Simple rotations such as Hadamard transforms reduce outliers, but still degrade at INT2 because they are not aligned with downstream attention. We propose OSCAR, an Ultra-lowbit KV Cache quantization method that estimates attention-aware covariance structures offline and uses them to derive fixed rotations and clipping thresholds for quantization. In this way, it aligns KV quantization with the covariance structures that attention actually consumes. More importantly, we not only provide theoretical justification but also develop a fully deployable OSCAR system with a custom INT2 attention kernel that remains compatible with paged KV-cache serving and fused kernel pipelines, enabling seamless integration into modern LLM serving frameworks such as SGLang and vLLM.

We evaluate our methods on recent reasoning models with reasoning traces of up to 32k tokens across 5 tasks. On Qwen3-4B-Thinking-2507 and Qwen3-8B, OSCAR reduces the BF16 accuracy gap to 3.78 and 1.42 points, respectively, while naive rotation INT2 collapses to nearly zero. We further scale OSCAR to Qwen3-32B and GLM-4.7 (358B params), where it remains effectively on par with BF16. On long context - RULER-NIAH up to 128K, OSCAR remains robust on both Qwen3 models, while naive rotation INT2 collapses. System-wise, OSCAR reduces KVcache memory by approximately 8×, improves throughput by up to 7× at large batch sizes under the same memory budget, and accelerates batch-size-1 decoding by up to 3× over BF16 due to reduced memory bandwidth overhead.

Code | Website | RotationZoo

[Figure 2]

[Figure 3]

[Figure 4]

### 1 Introduction

Long-context inference has made the key–value (KV) cache one of the main costs of serving large language models. During autoregressive decoding, the cache grows with context length, batch size, and model depth, and every new token must read a large fraction of it from GPU memory. Compressing the KV cache is therefore a direct way to increase batch size and reduce memory

Preprint.

[Figure 5]

[Figure 6]

[Figure 7]

###### OSCAR Visualization

###### Inference

Performance

[Figure 8]

[Figure 9]

Accuracy

###### KV cache layout

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Sink BF16

History rotated + clipped + INT2

Recent BF16

[Figure 15]

Calibration

dataset

old new ★ OSCAR keeps sink/recent in full precision

(offline)

[Figure 16]

###### Hadamard Rotated K Activation

OSCAR Rotated

K activations

K Activation

[Figure 17]

###### Write path

[Figure 18]

[Figure 19]

Rotation

RK

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Rotate Clip

[Figure 25]

Paged cache

τK τV

RK RV

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

KV

INT2

Clip

τK Throughput

[Figure 30]

[Figure 31]

✓ query-aware

✗ biased

- ✗ distorted
- ✗ distorted

Hadamard Rotated V Activation

OSCAR Rotated

###### Read path

V activations

V Activation

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Rotation

Dequantize

Inverse rotate

Paged

RV

Attention

[Figure 39]

[Figure 40]

[Figure 41]

INT2 → FP16

cache

[Figure 42]

Clip

###### τV

✓ attention-score-aware

[Figure 43]

✗ biased

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Output: per-layer RK, RV + τK, τV ★ OSCAR: rotation and clip thresholds calibrated jointly

###### Amazing Performance

###### Fused Triton kernel

★ OSCAR: accuracy preserve and high throughput

★ OSCAR: for read/write path— no extra memory traffic

- Figure 1: OSCAR pipeline overview. Offline, OSCAR estimates attention-aware key/value covariance rotation and shows how the resulting rotation makes KV activations more uniform: Hadamard mixing flattens raw peaks, while the OSCAR rotation separates directions that matter more or less to attention. Online, the serving path keeps sink and recent tokens in BF16 while applying the fixed rotate–clip–INT2 path to history KV tokens inside the paged SGLang cache. The right panels summarize that OSCAR preserves accuracy across models while achieving throughput close to the QuaRot(INT2) serving path and higher throughput than BF16.

traffic [1–8]. Among the available design points, INT2 quantization is especially attractive: it promises a large memory reduction while retaining a hardware-friendly fixed-width representation.

Aggressively compressing KV caches to ultra-low precision (e.g., INT2) remains challenging because KV activations contain severe channel-wise outliers: a small subset of channels often exhibit extremely large magnitudes, while most channels remain relatively well-behaved [4]. Under low-bit quantization, these outliers dominate the quantization scale, compressing most normal values into only a few effective quantization levels and substantially degrading attention quality. Rotation-based quantization addresses this issue by applying a fixed orthogonal transform, such as a Hadamard rotation, that redistributes a few extreme activation values across many channels, producing a more uniform activation distribution that is easier to quantize [9–12]. In addition, rotation preserves tensor dimensionality and applies a fixed linear transform without introducing per-channel routing or irregular sparse metadata. This makes it naturally compatible with paged KV-cache layouts [13], and FlashAttention-style fused decode kernels [14–17]: each KV vector is simply moved into a better-conditioned basis before quantization and moved back when used by page attention [12].

However, a random rotation is still data-oblivious. It can smooth activation ranges, but it does not know which directions are important to attention. At INT2, this distinction matters: only four quantization levels are available, so the error should be pushed into directions that the model reads less strongly. Attention operates on the correlations and score-weighted interactions induced by keys and values, rather than on their raw Euclidean representations. This suggests the optimal rotation target should be derived from attention statistics themselves. Based on this observation, we propose OSCAR, an INT2 KV quantizer that estimates attention-aware covariance structures through a lightweight calibration pass and uses them to derive separate rotations for keys and values, along with per-layer clipping thresholds. Figure 1 (left) illustrates that while data-oblivious rotations can partially smooth outliers, they remain insufficient for INT2 quantization. In contrast, covariance-aware rotations produce substantially smoother activation distributions, enabling effective quantization.

Our contributions are summarized as follows

- • We identify the missing target in INT2 rotation-based KV. Generic rotations mainly scatter activation outliers, but INT2 accuracy depends on the errors in attention scores and layer outputs; the rotation should be induced by downstream attention not by raw cache reconstruction alone.

- • We propose OSCAR, an attention-aware calibration framework for ultra-low-bit KV-cache quantization. OSCAR uses a lightweight calibration set to obtain attention-aware rotations for keys and values, enabling the quantized cache to better preserve downstream attention computation. A theoretical analysis is provided to show that the resulting covariance-target rotations are optimal under a natural frozen-error surrogate. Empirically, we evaluated across a wide range of state-ofthe-art models from 4B to 400B, and retains near-BF16 accuracy at only 2.28 bits per KV element across multiple LLM families, including on a challenging code benchmark (LiveCodeBench).
- • We develop a production-ready INT2 KV-cache serving system. OSCAR preserves compatibility with paged and prefix KV-cache serving by keeping the bulk KV cache in a dense rotated INT2 representation. The system integrates into production SGLang decoding pipelines with customized Triton decoding kernels, such that one can fully utilize the prefix cache techniques [18]. Our system delivers up to 6.2× higher throughput at 100k length and achieves roughly 2× gains in both per-user speed and per-GPU throughput under full-cache workloads. This shows it is both user-friendly (lower latency) and system-efficient (higher GPU utilization).

Due to space constraints, we provide a full discussion in Appendix C, D on how our ideas are inspired by and connected to prior work in the research community.

### 2 Preliminaries and Motivation

Attention and KV cache. We use row-vector notation throughout. For simplicity, we define a singlehead attention [19] as follows. Given a sequence T of hidden states {xt}Tt=1 with xt ∈ R1×d and the weights WQ,WK,WV ,∈ Rd×d, we formulate the query, key, and value as Q = [q1;...;qT] ∈ RT×d,K = [k1;...;kT] ∈ RT×d,V = [v1;...;vT] ∈ RT×d, where d is the head dimension; qt = xtWQ ∈ R1×d, kt = xtWK ∈ R1×d, and vt = xtWV ∈ R1×d.The attention scores is defined as S = softmaxrow(QK⊤/

√

d) ∈ RT×T, and the attention output is O = SV ∈ RT×d, equivalently

oi = Tt=1 Si,tvt. During autoregressive inference, K1:t = [k1;...;kt] and V1:t = [v1;...;vt] are stored in cache – this is the so-called KV cache. Covariance and more details are in Appendices A, B.

Quantization notation. A b-bit quantizer consists of a quantization map Q+ : R → Cb and a dequantization map Q− : Cb → R, where Cb is a discrete set of 2b representable codes. Their composition gives the quantize–dequantize map Q(x) = Q−(Q+(x)). A symmetric uniform map can be written as Q(x) = s · clip(⌊x/s⌉,−τ,τ), where s > 0 is the quantization scale and τ is the clipping limit. For a matrix X consisting of row vectors, we apply quantization element-wise to each row.1. Given rotation matrix R [9], we denote the reconstruction of quantized K and V as K = Q(KR)R⊤ and V = Q(V R)R⊤.

Why raw-cache reconstruction is not enough. Rotation is effective for KV-cache quantization because it spreads large channel values into a basis with a more uniform dynamic range [9]. However, simple data-oblivious rotations such as Hadamard or random orthogonal transforms are often insufficient: they smooth the cached tensors, but do not identify which directions are most important for downstream attention. A standard tensor-reconstruction view minimizes ∥K − Q(K)∥2F and ∥V − Q(V )∥2F, yet attention does not consume K and V through their Euclidean reconstruction errors. Keys are used through logits, and values are used through the attention-weighted aggregation. For keys K, the downstream logit distortion is

∥QK⊤ − Q K⊤∥2F = tr (K − K)Q⊤Q(K − K)⊤ , (1)

which is controlled by the query covariance Q⊤Q rather than by K⊤K alone. For values V with quantized V = Q(V ), the downstream output distortion is

∥SV − S V ∥2F = tr (V − V )⊤S⊤S(V − V ) , (2)

which depends on how attention scores weight the value rows. Thus, if the goal is to reduce empirical attention distortion rather than raw-cache reconstruction error, the rotation should be estimated from target covariance induced by the attention computation itself.

- Figure 2 illustrates this gap. Naive INT2, Hadamard-only rotation, and clipping alone still leave substantial attention-score divergence and output error. In contrast, OSCAR uses attention-aware

1Row-wise is used for our theory. Empirically, quantization is applied to the head dimension with block-size 128, 64 or 32.

KV-cache quantization error decomposition (Qwen3-4B-Thinking, AIME)

###### (c) Attention block output (AV) WO

###### (a) K and V cache quantization noise

###### (b) Attention-score divergence

(d) Layer hidden state (propagated)

| | | |
|---|---|---|
| |Variant| |
| |Baseline INT2| |
| |Hadamard only| |
| |Clip only<br><br>OSCAR (ours)| |
| | |Cache<br><br>K cache (solid, filled) V cache (dashed, open)|

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | |B H<br><br>|aseline adama|INT2 rd only| | |
| | | | | | |C|lip onl|y| | |
| | | | | | | | | | | |
| | | | | | |O|SCAR|(ours)| | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | |Ba Ha<br><br>|seline damard|INT2 only| | | | | |
| | |Cli|p only| | | | | | |
| | |OS|CAR (o|urs)| | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | |B|aseline|INT2| | |
| | | | | | |H|adama|rd only| | |
| | | | | | | | | | | |
| | | | | | |C O<br><br>|lip onl SCAR|y (ours)| | |
| | | | | | | | | | | |

- 100
- 101
- 102

- 100
- 101

- 100
- 101
- 102

- 100
- 101

AbsoluteMSE

AbsoluteMSE

AbsoluteMSE

KL()ppFP16q

10 1

10 1

10 1

10 2

10 2

10 2

10 3

10 3

10 1

10 3

10 4

0 5 10 15 20 25 30 35

0 5 10 15 20 25 30 35

0 5 10 15 20 25 30 35

0 5 10 15 20 25 30 35

Layer

Layer

Layer

Layer

- Figure 2: Quantization error. OSCAR limits it at every stage (Qwen3-4B-Thinking-2507, AIME). (a)

Relative MSE in quantized K and V caches (keys solid, values dashed). (b) KL(pFP16 ∥pq) between FP16 and quantized attention score distributions. (c) Relative MSE at the attention-block output

(post-WO). (d) Relative MSE in the propagated hidden states across layers. Curves compare naive INT2, Hadamard-only preprocessing, clip-only outlier control, and full OSCAR.

calibration target covariance before quantization, reducing the error in four subfigures. We provide a detailed intuition for why each factor is needed and why they appear in this order in App. A.4.

### 3 Algorithm Design: Offline Calibration and Justification

The method has two phases: an offline phase that estimates covariance , constructs layer-wise rotation matrices, and fits per-token clipping thresholds; and an online phase that applies the resulting fixed transforms based on a mixed-precision cache layout during serving. All covariance and rotations are estimated offline from a small calibration dataset. For each layer and attention head, we construct attention-aware target covariance from the calibration activations to determine the base rotations.

Query-aware key target covariance. For a query row q, with the attention-aware key target covariance Q⊤Q introduced in equation (1), its empirical estimator is CQ = N1 Nn=1 qn⊤qn; we apply eigen decomposition as in App. A.2, CQ = UQΛQUQ⊤, and define the rotation as Rk := UQ.

√

Score-aware value target covariance. For the attention matrix S = softmaxrow(QK⊤/

d), we heuristically define the target covariance in equation (2): CS = N1 V ⊤S⊤SV. We then compute the eigendecomposition as in App. A.2, CS = USΛSUS⊤, and define the raw rotation as Rv := US.

OSCAR rotations. Following rotation-based low-bit quantization methods [9, 20], we compose the base rotation with a Hadamard transform HHad (App. A.1) and a bit-reversal permutation Pbr to form the final rotations for keys and values:

RK = UQ HHad Pbr, RV = US HHad Pbr. (3)

Here Rk and Rv are the covariance base rotations, while HHad further improves quantization geometry by redistributing channel energy, and Pbr [21] interleaves large- and small-variance channels so that adjacent channels have similar dynamic range.

Scale determination and per-token clipping. Our quantization backend follows standard posttraining quantization practice. We use affine asymmetric INT2 quantization with scale and zero point for both keys and values, together with percentile-based clipping to control outliers. These mechanisms are common in low-bit activation and KV-cache quantization pipelines [22, 4, 23, 24], with technical details shown in App. A.5.

Optimality of targets under ambient error assumption. Below we propose that under the diagonal residual assumption, our heuristic targets CQ = N1 Nn=1 qn⊤qn and CS = N1 V ⊤S⊤SV achieve the lowest frozen-error surrogate. The proof of the theorem and the justification of surrogate objectives mentioned in the theorem are shown in App. A.6 and App. A.7 respectively.

Theorem 1 (Optimality of the simple spectral variants under ambient-basis diagonal residuals). Consider the frozen-error surrogate objectives for the key and value base rotations: L˜K(Rk) = tr(Rk⊤CQRkEK),Rk⊤Rk = Id and L˜V (Rv) = tr Rv⊤CSRvEV ,Rv⊤Rv = Id. Assume that the frozen (independent of input and rotation) residual covariances are diagonal in the ambient basis:

EK = Nj=1 Q(kjRk) − kjRk ⊤ Q(kjRk) − kjRk = diag(µ1,...,µd), µ1 ≤ ··· ≤ µd, EV = Nj=1 Q(vjRv) − vjRv ⊤ Q(vjRv) − vjRv = diag(η1,...,ηd), η1 ≤ ··· ≤ ηd.

Then, on the calibration dataset: Rk = UQ and Rv = US are minimizers of L˜K(Rk) and L˜V (Rv).

### 4 System Design: Online Serving with 2-bit KV Cache

KV Cache Layout. We integrate OSCAR into the SGLang [18] serving stack as an INT2 KV-cache mode with full compatibility with paged-attention [13]. The implementation preserves two short high-precision windows: the first S0 tokens, which behave as attention sinks, and the most recent W tokens before the current position. The rest of the middle context is stored in INT2 after the fixed OSCAR rotation. Thus, at decoding position t, the logical cache consists of

∥ [S0 + 1,t − W]

∥ [t − W + 1,t]

[1,S0] bf16 sink

int2 history

bf16 recent

KV Cache Update. During prefill, the runtime writes cache rows through a fully fused Triton kernel [25]. For a token with BF16 rows kt,vt and a clip value of τt(K),τt(V ), the stored INT2 are

kt+ = Q+2 (clip(ktRK,τt(K))), vt+ = Q+2 (clip(vtRV ,τt(V ))), with four 2-bit values packed per byte.

New decode tokens will be first written to the window recent window as ktRK and vtRV . As decoding advances, the oldest recent token will be demoted into the INT2 middle region by a fused Triton kernel that applies the same clip–quantize operation. OSCAR then optimize value rotation by absorbing RV into the model’s projection weights, achieving compute saving and latency reduction. Decoding Attention Kernel. During decoding, OSCAR partitions each request’s cache indices into BF16 (sink + recent) and INT2 segments on the GPU. The INT2 kernel unpacks bytes, applies the stored scale/zero parameters, and accumulates them in floating point. Existing decoding attention kernels typically consist of two kernel launches [26], one for parallel processing on KV cache segments along the sequence dimension. The second one is used to merge partial attention results from segments with online softmax [14]. OSCAR introduces an additional kernel for BF16 KV cache attention, then reuses the second merge kernel to piggyback on the merge of high-precision partial results. Since the BF16 (sink + recent) segment has orders of magnitude fewer elements than the INT2 segment, the overhead is negligible.

### 5 Experiments

#### 5.1 Experimental Setup

Models & Benchmarks. We evaluate OSCAR on four model configurations: Qwen3-4B-Thinking2507, Qwen3-8B, Qwen3-32B, and GLM-4.7-FP8 [27, 28]. These models cover a small reasoning model, a mid-sized dense model, and a frontier-scale model, allowing us to test OSCAR across different levels of INT2 robustness. We evaluate accuracy on five reasoning and coding benchmarks: AIME25 [29],GPQA-Diamond [30], HumanEval [31], LiveCodeBench v6 [32], and MATH500 [33].

We also evaluate long-context retrieval with RULER [34] to test the long-sequence robustness. Particularly, RULER-NIAH is the cleanest stress test: the answer is explicitly present in the prompt, and the main question is whether quantized history tokens can be attended to in a long context.

#### Table 1: AIME25 at 32K generation.

Method BPE Qwen3-8B Qwen3-32B Original BF16 16.00 66.00±7.33 72.59±7.41 KIVI-KV2 [22] 2.25 52.33±9.00 57.41±9.26 KIVI-KV2* [22] 2.26 57.67±9.00 59.05±12.38 Kitty [6] 2.39 59.67±10.33 69.26±9.26 OSCAR 2.38 66.67±3.33 74.00±5.48

Hardware and Framework. Qwen3-4BThinking and Qwen3-8B are served on a single NVIDIA H100 (80GB); Qwen3-32B and GLM-4.7-FP8 are served on 2×H100 and 8×H100 with tensor parallelism, respectively. All system-level runs use our SGLang [18] implementation.

Generation Protocol & Calibration. We use temperature T=0.6 (T=1.0), top-p=0.95, and topk=20 for Qwen3-family (GLM-4.7), with thinking mode enabled for reasoning benchmarks. Unless otherwise noted, each Qwen configuration is run with 5 independent seeds and GLM-4.7-FP8 with

- 3 runs; we report mean ± standard deviation. All accuracy evaluations use a maximum generation

Table 2: INT2 KV-cache quantization methods compared on four model configurations and five benchmarks. Entries are µ ± σ over 5 seeds, except GLM-4.7-FP8 (3 runs)*. BPE denotes effective bits per KV element at 128K context length; “no MP” means mixed precision is disabled. “Drop” is the average-column gap to the BF16 upper bound; smaller is better.

Model Method BPE GPQA HumanE LCB v6 AIME25 MATH500 Mean Drop

BF16 16.00 67.27±1.80 94.05±0.54 48.66±2.20 74.67±1.83 93.55±0.33 75.64 – Saw-INT4 [12] 4.25 66.37±2.19 89.78±0.80 46.20±1.94 70.00±4.08 93.19±0.51 73.11 -2.53 TurboQuant (no MP) [36] 3.25 41.41 31.83 0.58 16.67 68.20 31.74 -43.90 QuaRot-INT2 [9] 2.25 0.34±0.48 0.98±0.10 0.00±0.00 0.00±0.00 5.67±0.19 1.40 -74.24 Naive INT2 2.25 0.00 0.00 0.00 0.00 0.00 0.00 -75.64 OSCAR (ours) 2.28 64.95±1.16 92.24±1.02 45.38±1.97 64.00±3.65 92.75±0.39 71.864 -3.78

Thinking-2507

Qwen3-4B

BF16 16.00 56.67±2.30 85.95±1.01 49.01±2.13 70.00±3.33 92.59±0.62 70.84 – Saw-INT4 [12] 4.25 54.85±3.17 86.44±0.42 47.95±2.15 68.00±2.98 92.63±0.27 69.97 -0.87 TurboQuant (no MP) [36] 3.25 55.05 74.63 21.05 46.67 87.00 56.88 -13.96 QuaRot-INT2 [9] 2.25 14.98±0.63 9.80±0.76 0.58±0.58 2.22±1.57 23.13±1.88 10.14 -60.70 Naive INT2 2.25 0.00 0.00 0.00 0.00 0.00 0.00 -70.84 OSCAR (ours) 2.28 55.05±1.47 87.88±0.44 46.32±1.91 66.67±3.33 92.22±0.83 69.416 -1.42

Qwen3

-8B

BF16 16.00 58.49±1.21 91.19±1.11 59.06±1.80 68.67±5.58 93.55±0.41 74.19 – Saw-INT4 [12] 4.25 59.29±0.76 90.85±0.47 56.02±2.41 72.67±4.90 93.31±0.10 74.43 +0.24 TurboQuant (no MP) [36] 3.25 58.69 88.41 55.56 66.67 90.60 71.99 -2.20 QuaRot-INT2 [9] 2.25 19.70±1.09 1.83±0.33 0.39±0.55 0.00±0.00 17.60±1.66 7.90 -66.29 Naive INT2 2.25 0.00 0.00 0.00 0.00 0.00 0.00 -74.19 OSCAR (ours) 2.28 60.40±3.47 90.12±0.71 53.57±0.67 74.00±5.48 92.75±0.46 74.17 -0.02

Qwen3

-32B

BF16 16.00 73.23±1.33 91.46±0.65 49.12±0.59 80.00±3.33 95.66±0.61 77.89 – Saw-INT4 [12] 4.25 73.74±1.01 91.30±0.92 50.49±1.47 78.89±1.92 95.32±0.12 77.95 +0.06 TurboQuant (no MP) [36] 3.25 66.67 90.24 58.48 80.00 95.39 78.15 +0.26 QuaRot-INT2 [9] 2.25 68.01±3.79 89.87±1.78 48.54±4.79 78.89±1.92 90.40±0.35 75.14 -2.75 Naive INT2 2.25 54.55±3.54 86.50±1.43 37.03±6.41 37.78±1.92 86.60±0.80 60.49 -17.40 OSCAR (ours) 2.28 73.57±0.58 91.06±0.25 52.63±1.01 78.89±1.92 94.66±0.76 78.16 +0.27

FP8,358B

GLM-4.7

*TurboQuant entries are single-run results as its vLLM path is too slow for repeated 32K evaluations under our time budget.

length of 32768 tokens and run end-to-end inside SGLang, using the same execution path as our system measurements. All OSCAR parameters are estimated once from a small MMLU-style calibration set. For each model, we run one calibration pass and dump per-layer Q,K,V activations (8878 tokens × number of layers), from which we compute the key/value rotations and per-layer clipping thresholds, then reuse the same parameters for all benchmarks. No task-specific calibration is used.

Baselines. We use two baseline groups. (1) Group A contains channel-wise KV methods such as KIVI and Kitty. These methods require residual buffers, channel-wise scales, promoted channels, or custom page layouts, and we do not have paged/fused kernels for them at 32K generation length.

For the only shared 32K accuracy setting, we show the reported Qwen3-8B and Qwen3-32B AIME25 results in the Table 1; The BPE values include INT payload, BF16 scale/zero metadata with group size 128, and BF16 initial tokens where applicable; the Kitty row is the 12.5% key-channel boost variant. (2) Group B contains rotation-based methods: FP16/BF16, naive INT2/INT4, QuaRot-style Hadamard rotation, block-diagonal Hadamard (Saw-INT4) [12], and TurboQuant [35]. For TurboQuant, we use the official vLLM implementation from PR #38479 [35] and run the same 32K-generation evaluation; for fairness, we disable its mixed precision setup and quantize all layers.

For OSCAR we always pair rotation with sink (S=64) and recent-window (R=256) BF16 protection, calibration-derived per-layer clip thresholds (whose typical values are cK ≈ 0.96 and cV ≈ 0.92, see the calibration paragraph above), and per-channel asymmetric INT2 quantization. In Table 2, BPE counts INT payload plus BF16 scale/zero metadata with block size 128. For OSCAR, we report effective BPE at 128K context length, including the BF16 sink/recent tokens (S=64, R=256). LiveCodeBench v6 is evaluated with the same 32K generation cap as the other tasks; this truncates some long code-generation outputs, so the reported LCB numbers are lower than they would be under a longer 128K generation budget.

#### 5.2 Main Results

- Table 2 reports the main accuracy comparison on Qwen3-4B-Thinking-2507, Qwen3-8B, Qwen3-32B, and GLM-4.7-FP8.

For OSCAR we use the configuration selected by the ablation studies in Section 5.3: sink=64, recent=256, calibration-derived clip thresholds, attention-aware key/value covariance, and asymmetric INT2 quantization. The key observation is that OSCAR is the only near-2-bit method that remains close to the BF16 accuracy frontier under the 32K-generation evaluation. On Qwen3-4B-Thinking2507 and Qwen3-8B, OSCAR reduces the BF16 gap to 3.78 and 1.42 points, respectively, while TurboQuant drops 43.90 and 13.96 points and rotation-only INT2 baselines largely collapse. On Qwen3-32B and GLM-4.7-FP8, OSCAR is essentially tied with BF16 at 2.28 BPE, while Saw-INT4 uses 4.25 BPE.

Long-Context Robustness. Table 3 reports RULER-NIAH accuracy from 4k to 128k tokens on the same serving-compatible baselines used in the main accuracy comparison. The expected pattern is simple: short contexts should remain close to BF16, while longer contexts expose accumulated attention-logit error. OSCAR should degrade more slowly than rotation-only INT2 baselines because its rotation target is chosen from the covariance seen by attention, not from raw-cache reconstruction, and because long contexts amplify accumulated KV quantization error. Figure 3 provides a direct KL-based check before the retrieval results. Across both Qwen3-4B-Thinking-2507 and Qwen3-8B, OSCAR is more stable as sequence length increases: its attention distribution stays closer to FP16, while naive INT2 and Hadamard-only rotation drift more quickly. On Qwen3-4B-Thinking-2507, QuaRot-INT2 is already near zero at short contexts, whereas OSCAR stays close to BF16 through 16k and retains non-trivial retrieval at 64k and 128k. On Qwen3-8B, QuaRot-INT2 remains usable only at 4k–8k and collapses after 16k, while OSCAR keeps substantially higher retrieval accuracy throughout the sweep. The GLM-4.7-FP8 preliminary run shows a complementary large-model case: all three methods remain strong on this retrieval-only task, and OSCAR matches the BF16 curve up to 128k. Together, the RULER results and KL curves indicate that attention-aware covariance calibration mainly helps when long histories make small KV errors accumulate over many steps.

- Table 3: Long-context retrieval accuracy on RULER-NIAH [34] across context lengths from 4k to 128k tokens, on the Group A serving-compatible baselines. Each number is averaged over 3 random seeds, except GLM-4.7-FP8 preliminary results (one run, as GLM-4.7 is very stable).

Model Method 4k 8k 16k 32k 64k 128k

BF16 100.0±0.0 100.0±0.0 99.7±2.0 99.3±1.0 85.3±5.0 81.0±1.4 QuaRot-INT2 [9] 0.0±0.0 0.8±0.2 0.0±0.0 0.0±0.0 15.6±3.2 0.0±0.0

Qwen3-4B

(-Thinking-2507) OSCAR 99.7±0.1 100.0±0.0 97.8±0.2 87.6±0.1 61.9±1.3 39.5±1.0

BF16 100.0±0.0 99.6±0.0 98.9±0.1 97.3±0.0 79.2±0.1 78.2±1.0 QuaRot-INT2 [9] 74.0±1.0 80.6±0.3 19.0±2.0 9.8±1.0 0.0±0.0 0.0±0.0

Qwen3-8B

- OSCAR 99.5±0.2 97.8±0.3 93.9±0.7 86.3±0.6 61.9±1.8 45.0±0.0

GLM-4.7-FP8†

BF16 100.0 100.0 100.0 100.0 98.8 97.2 QuaRot-INT2 [9] 100.0 99.7 100.0 100.0 98.8 96.3

- OSCAR 100.0 100.0 100.0 100.0 98.8 97.2

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

###### KL()(linear)ppFP16q

5.06 4.99

- 4.67

4.92

6.72

7.47

8.08

6.70

7.35

1.59

2.26

3.31

5.03

6.48 6.46

7.46

Qwen3-4B-Thinking-2507 OSCAR vs QuaRot vs Clip-only KL drift (teacher-forced, 32 decode steps)

OSCAR (ours)

QuaRot

Clip only

Figure 3a: Qwen3-4B-Thinking-2507

4K 8K 16K 32K 64K 128K

Prefill context length

0

2

4

6

8

10

12

KL()(linear)ppFP16q

0.01 0.03 0.02 0.02 0.01

0.40

7.22

4.82

12.07 12.12

8.99

11.75

0.62

1.07

2.09

2.94

2.28

3.15

Qwen3-8B OSCAR vs QuaRot vs Clip-only KL drift (teacher-forced, 32 decode steps)

OSCAR (ours)

QuaRot

Clip only beyond native 40K (YaRN×4)

| |
|---|

Figure 3b: Qwen3-8B

Figure 3: OSCAR keeps attention distributions stable as context length grows. KL divergence against the FP16 attention distribution across context lengths for naive INT2, Hadamard, and OSCAR.

- 5.3 Ablation Studies

3.02

0.26

###### 0.040.02 0.11

4K8K 16K 32K 64K 128K 256K

Prefill context length

Rotation Analysis: Decomposing R and Comparing Rotation Targets. OSCAR’s composed rotation is R = U · HHad · Pbr, the product of an attention-aware eigenbasis U, a Hadamard

transform HHad, and a bit-reversal permutation Pbr. Table 4 removes each factor of R in turn (top block) and, on the same footing, changes the PCA target used to compute U (bottom block): OSCAR’s attention-aware targets are replaced by raw-cache reconstruction targets (K⊤K/V ⊤V ), fixed Hadamard rotations, or no learned rotation.

Two observations emerge. First, both the attention-aware eigenbasis U and the Hadamard component HHad contribute substantially; the bit-reversal permutation Pbr does not change accuracy in floatingpoint math but improves quantization geometry by interleaving large and small eigenvalues so that per-group quantization sees a more uniform range. Second, none of the alternative rotation targets (random Hadamard, raw K⊤K/V ⊤V reconstruction targets, random orthogonal) matches the attention-aware CQ/CSQ/CS targets at the same INT2 budget; the score-weighted variant covariance CSQ gives a further improvement over CQ alone. This isolates the central claim of the paper: which covariance matrix one diagonalizes matters more than whether one diagonalizes at all.

Sink and Recent Window Sizes. We sweep (S,R) ∈ {(0,0),(32,128),(64,256),(128,512), (256,1024)} on Qwen3-4B-Thinking-2507 and report accuracy with the additional BF16 KV memory that each protection window introduces (Table 5). A clear knee emerges at (S,R)=(64,256): smaller windows leave noticeable accuracy on the table, while larger windows provide negligible additional accuracy at the cost of substantially more BF16 KV memory. (64,256) is a sweet spot and thus used for the main results.

- Table 4: Decomposition of the composed OSCAR rotation R = U · HHad · Pbr (top block) and comparison against alternative rotation-target choices used by prior work (bottom block), all at the same INT2 bit budget with sink/recent protection on Qwen3-8B. Entries are mean ± std over 3 runs.

Configuration GPQA HumanE LCB v6 AIME25 MATH500 Mean FP16 reference 57.07±0.94 84.71±0.99 48.07±2.20 66.00±2.79 88.28±0.46 68.83 Decomposition of OSCAR’s composed rotation

Full OSCAR: U · HHad · Pbr 55.89±1.27 87.97±0.19 46.00±1.88 67.78±1.92 92.38±0.20 70.01 OSCAR with K⊤K/V ⊤V PCA target 38.72±1.27 35.53±0.67 3.70±0.68 13.33±3.33 64.33±0.53 31.12 w/o Pbr 52.86±1.62 86.71±0.84 45.22±0.34 63.33±8.82 91.85±1.45 68.00 w/o HHad (only U · Pbr) 51.35±1.91 83.78±0.37 19.49±2.05 21.11±1.92 82.97±0.80 51.74 w/o U (QuaRot + Pbr: HHad · Pbr) 40.74±3.44 38.54±1.18 4.48±0.68 15.56±1.92 64.80±1.29 32.82 no rotation (only clip + sink + recent) 18.52±3.29 0.00 0.39±0.68 2.22±1.92 0.00 4.23

- Table 5: Sink and recent window sizes on Qwen3-4B-Thinking-2507 (INT2 with OSCAR rotation and calibration-derived clip). (S,R) denote the number of BF16 sink tokens and recent-window tokens, respectively. “Extra BF16 KV” reports the protected BF16-token fraction, (S + R)/128K.

(S, R) GPQA HumanE LCB v6 AIME25 MATH500 Mean Extra BF16 KV (0, 0) 0.00 0.00 0.00 0.00 0.00 0.00 0% (32, 128) 57.58±2.20 91.91±0.19 40.74±2.36 55.56±5.09 92.65±0.95 67.69 0.12% (64, 256) 64.95±1.16 92.24±1.02 45.38±1.97 64.00±3.65 92.75±0.39 71.86 0.24% (128, 512) 65.32±0.77 93.17±0.32 45.22±0.33 67.78±1.92 93.32±0.12 72.96 0.49% (256, 1024) 64.98±1.27 93.13±0.81 46.20±2.03 67.78±1.92 93.32±0.12 73.08 0.98%

Clip Threshold. The clip thresholds are estimated per layer from rotated calibration activations rather than tuned on downstream tasks. Table 6 sweeps global clip ratios on Qwen3-4B-Thinking and shows that the calibration-derived point (cK,cV ) = (0.96,0.92) is close to the best grid point.

Calibration Data Regime. We also test whether OSCAR is sensitive to calibration volume or domain. Table 7 shows that the default 8k GPQA-Diamond calibration is close to the best setting, supporting a small one-time calibration pass.

#### 5.4 End-to-End Serving Throughput

Kernel-Level Profiling. We profile per-step decode latency on Qwen3-8B and GLM-4.7-FP8 to isolate the runtime cost of OSCAR. Table 8 shows that OSCAR reduces attention time by shrinking KV traffic while adding only a small fused quantization cost.

- Table 6: Validation of the calibration-derived clip thresholds (mean accuracy on Qwen3-4B-Thinking-

2507). Rows: cK; columns: cV . The cell highlighted in bold is the operating point produced by the calibration procedure described in Section 5.1; it is within 0.15 points of the best downstream-task grid point, confirming that OSCAR’s clip thresholds do not need to be hand-tuned.

cK \ cV 0.88 0.92 0.96 0.98 1.00

0.88 67.20 68.27 68.10 67.56 68.25 0.92 69.15 70.00 70.46 70.74 70.33 0.96 68.77 70.59 70.70 68.17 69.14

- 0.98 65.22 65.76 64.12 64.79 67.31

- 1.00 60.35 59.19 58.15 58.63 56.02

- Table 7: Calibration-data regime on Qwen3-4B-Thinking-2507 (INT2 with OSCAR rotation, calibration-derived clip, sink/recent protection). We sweep both the calibration volume (number of tokens) and the calibration domain. The default setting (8k GPQA-Diamond tokens) is highlighted.

Calibration domain Tokens GPQA HumanE LCB v6 AIME25 MATH500 Mean

2k 60.61±2.20 92.52±0.14 39.57±2.36 55.56±5.09 93.12±0.46 68.28 8k 60.44±2.04 92.07±0.76 40.74±0.89 60.00±5.77 92.52±0.58 69.15 16k 64.14±0.51 92.93±0.53 43.86±2.55 61.11±10.72 92.38±0.53 70.88 32k 61.28±1.54 92.20±0.32 46.00±2.36 57.78±5.09 92.25±0.31 69.90

MMLU prompts

WikiText 8k 61.11±0.51 92.72±0.63 42.11±2.11 58.89±3.85 92.65±0.50 69.50 GPQA-Diamond 8k (default) 62.96±1.27 92.56±0.00 44.64±1.47 62.22±1.92 92.65±0.31 71.01

Representative Serving Throughput. Table 9 reports throughput under 32 concurrent requests with 8192-token inputs and 1024-token outputs, giving the per-user and per-GPU throughput used to contextualize the speedup trends below.

Pure Decoding Speed. We first study how effective our kernel is in pure decoding speed. We isolated the prefill time or time to first token (TTFT) by setting a full prefix-cache-hit setting with a batch size 1, where all methods fully reuse the cached KV states throughout decoding. In this regime, throughput differences directly reflect decode efficiency rather than cache residency or scheduling effects. Figure 4 (Left) shows decode throughput normalized to BF16 across three models under varying context lengths. It shows that (i) OSCAR consistently outperforms Saw-INT4 [12] by up to 2x when the sequence length is 100k, reaching the theoretical bound, verifying the efficiency of our kernel design; (ii) The speedup of OSCAR increases with context length, from 1.98× at 30k,

- 2.52× at 60k, to 3.08× at 100k on Qwen3-4B. This trend scales up to 358B. As such, decoding speed becomes increasingly KV-bandwidth-bound at long contexts. By reducing KV-cache memory by a factor of 8 (BF16 to INT2), OSCAR achieves substantially larger gains as the context length grows, while the online rotation overhead remains effectively hidden within the decode kernels.

BF16

Saw-INT4

OSCAR (ours)

BF16

Saw-INT4

OSCAR (ours)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### Qwen3-4B-Thinking-2507

###### Qwen3-8B

GLM-4.7-FP8

###### BS=1

###### BS=8

BS=32

3.5

3.08

7.83

8

3.0

2.88

2.83

SpeedupvsBF16(×)

SpeedupvsBF16(×)

2.52

2.49

2.5

6.17

2.29

5.89

6

5.19

1.98

1.98

2.0

1.84

1.52

1.51

4

1.5

3.63

1.27 1.34 1.37

1.30 1.36

1.33

3.44

3.35

1.25

3.08 2.88 2.83

1.00 1.00 1.00

1.00 1.00 1.00

1.00 1.00 1.00

2.43 2.32 2.33

1.0

2.36

2.09

2

1.51 1.52 1.37

0.5

1.00 1.00 1.00

1.00 1.00 1.00

1.00 1.00 1.00

0.0

0

30k 60k 100k

30k 60k 100k

30k 60k 100k

Qwen3-4BThinking-2507

Qwen3-8B GLM-4.7-FP8

Qwen3-4BThinking-2507

Qwen3-8B GLM-4.7-FP8

Qwen3-4BThinking-2507

Qwen3-8B GLM-4.7-FP8

Input length

Input length

Input length

- Figure 4: Left: Decode throughput speedup relative to BF16 at batch size 1 with a full prefix-cache hit with 1k our output sequence length. OSCAR achieves progressively larger speedups as the context increases. Right: Job-level throughput at 100k sequence, as a function of batch size for Qwen3-4B/8B and GLM-4.7-FP8. Each group of bars compares BF16, Saw-INT4, and ours at BS ∈ {1,8,32}.

Effect of Batch Size. We next study how serving throughput scales with batch size under long-context decoding. Figure 4 (right) reports job-level throughput with prefix warm up at a fixed input length (100k tokens) with batch sizes BS ∈ {1,8,32}. Prefix warm up is used to simulate popular agentic serving workload with long prefix reuse. Three observations stand out. (i) OSCAR consistently outperform BF16 across all batch sizes. On GLM-4.7-FP8, OSCAR achieves a 2.83× speedup

- Table 8: Per-decode kernel profiling on Qwen3-8B, 1×H100, single decoding step latency in milliseconds.

B

BF16 OSCAR

GEMM Attn Quant Other Total GEMM Attn Quant Other Total

1 5.6 (56.0%) 3.8 (37.9%) – 0.6 (6.1%) 9.9 5.6 (69.9%) 1.3 (16.8%) 0.5 (5.9%) 0.6 (7.5%) 8.0 8 5.6 (53.0%) 4.4 (41.0%) – 0.6 (6.0%) 10.7 5.7 (57.4%) 3.0 (30.1%) 0.5 (4.9%) 0.8 (7.7%) 9.9 16 5.7 (37.5%) 8.9 (58.1%) – 0.7 (4.4%) 15.2 5.7 (49.7%) 4.6 (39.4%) 0.5 (4.2%) 0.8 (6.7%) 11.6 32 5.7 (24.3%) 17.0 (72.7%) – 0.7 (3.0%) 23.4 5.7 (37.0%) 8.5 (54.6%) 0.5 (3.2%) 0.8 (5.2%) 15.5 64 capacity limited 5.8 (24.9%) 16.2 (69.1%) 0.5 (2.3%) 0.9 (3.7%) 23.4

128 capacity limited 5.9 (14.9%) 31.9 (81.1%) 0.6 (1.4%) 1.0 (2.6%) 39.3

- Table 9: Representative end-to-end serving throughput under 32 concurrent requests with 8192-token inputs and 1024-token outputs. Qwen3-4B-Thinking and Qwen3-8B are evaluated on 1×H100 (80GB), while GLM-4.7-FP8 is evaluated on 8×H100 (TP=8). U = tokens/s/user, G = tokens/s/GPU. Acc. is averaged over the five benchmarks in Table 2.

Qwen3-4B-Thinking-2507 Qwen3-8B Method U G Acc. U G Acc. BF16 41.1 1187.9 75.64 35.8 999.9 70.84 Naive INT2 66.4 1797.4 0.00 54.2 1415.1 0.00 QuaRot-INT2 [9] 66.3 1795.8 1.40 54.1 1411.9 10.14 Saw-INT4 [12] 50.6 1444.1 73.11 43.0 1183.2 69.97 OSCAR (ours) 63.3 1723.9 71.86 52.5 1374.0 69.42

over BF16 at BS=1, which further increases to 7.83× at BS=32. This behavior highlights that the reduced KV-cache footprint of INT2 methods becomes increasingly beneficial under highly concurrent long-context serving workloads. (ii) OSCAR consistently outperforms Saw-INT4 across all models and batch sizes in Figure 4, showing that the fused online rotation does not introduce an observable throughput penalty. (iii) Throughput improves substantially with larger batch sizes as OSCAR can accommodate more concurrent requests with less memory pressure. OSCAR’s design can also benefit prefix cache [18] thus improve cache hit rate with better job-level cache reuse.

Serving Throughput with Prefix Cache from 0 to 100%. Figure 6 evaluates end-to-end serving throughput under concurrent multi-user workloads by sweeping prefix-cache hit ratio from cache disabled to near-100% cache hits.

###### 100k input / 1024 output, narrativeqa-100k, run1 cold cache

###### Qwen3-4B-Thinking-2507

GLM-4.7-FP8

Aggregateserverthroughput(tok/s)

| |
|---|

| |
|---|

120

150

100

125

80

100

60

75

40

50

20

25

0

0

21 22 23 24 25 26 27 28

21 22 23 24 25 26 27 28

Concurrency (batch size)

Concurrency (batch size)

BF16 baseline QuaRot-INT2 Saw-INT4 OSCAR

- Figure 5: Long-context serving stress test with 100k-token inputs. OSCAR’s uniform INT2 KV-cache scales up to 28 requests with continued throughput gains.

The results show that (i) increasing the prefix-cache hit ratio consistently expands the throughput frontier by reducing prefill recomputation, with the largest gains appearing at higher batch sizes; (ii) OSCAR remains on or near the efficiency frontier across all cache regimes, closely matching aggressive INT2 baselines while avoiding extra decode-time memory traffic or indirection; (iii)

No cache (0% hit)

Partial cache (~80% hit)

Full cache (~100% hit)

GLM-4.7-FP8Thinking-2507

Per-GPUG(tok/s)

300

Qwen3-4B-

200

| | |
|---|---|

100

20 40 60

20 40 60

20 40 60

Per-GPUG(tok/s)

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

20

10

0

5 10 15 20 25

5 10 15 20 25

10 15 20 25

Per-user U median (tok/s)

Per-user U median (tok/s)

Per-user U median (tok/s)

BF16 QuaRot-INT2 Saw-INT4 OSCAR (ours) Qwen3-4B-Thinking-2507 75.64 1.40 73.11 71.86

GLM-4.7-FP8 77.89 75.14 77.95 78.16

BS=1

BS=8 BS=16 BS=32

| |
|---|

- Figure 6: Effect of prefix-cache hit ratio on end-to-end serving throughput (100k ISL, 1K OSL). Each subplot shows median per-user throughput U (x-axis, tok/s) versus mean per-GPU throughput G (y-axis, tok/s), with markers denoting batch sizes BS ∈ {1,8,16,32}. Rows correspond to Qwen34B-Thinking-2507 and GLM-4.7-FP8. Columns correspond to radix cache disabled, radix cache enabled during normal execution, and immediate replay after warmup (near-100% hit ratio).

compared with BF16, low-bit KV quantization enables substantially higher per-GPU throughput because it improves decode efficiency and reduces KV-cache memory footprint, allowing larger concurrent batches under the same memory budget; and (iv) OSCAR achieves the best accuracy among INT2 methods while preserving the standard paged KV-cache abstraction and integrating cleanly into fused SGLang decode kernels, whereas QuaRot-INT2 can approach the throughput ceiling but suffers significant accuracy degradation as shown in Table 2.

Memory and Extreme Batch Scalability. Finally, we study memory usage and batch scalability under long-context serving. We evaluate on a single H100 (80GB) with 100k-token inputs while scaling the concurrent batch size; detailed scaling curves are provided in the appendix. OSCAR reduces the KV-cache footprint by approximately 8× relative to BF16, enabling substantially larger feasible batch sizes under the same memory budget.

As shown in Figure 6, this stress test demonstrates that OSCAR can scale to very large concurrent batch sizes with inputs under 100k-tokens. Under 100k-token inputs, OSCAR’s uniform INT2 KV-cache pages enable substantially larger feasible batch sizes, scaling up to 28 concurrent requests on a single H100 while throughput continues to improve. In contrast, BF16 and INT4 baselines either run out of memory at smaller batch sizes or quickly plateau as concurrency increases. This shows that OSCAR’s design is not only memory-efficient but also practically scalable to extremely long-context serving regimes. More ablation runs and discussion are in Appendix E.3 and D.

### 6 Conclusion

We introduced OSCAR, an INT2 KV-cache quantization method that chooses rotations from attentionaware target covariance rather than raw cache reconstruction. With one offline calibration pass, OSCAR fixes key/value rotations and clip thresholds, then uses a fused SGLang path to quantize history KV tokens while preserving sink and recent tokens. This simple target-and-kernel co-design recovers much of the BF16 accuracy lost by naive INT2 while keeping the memory and serving advantages of 2-bit KV caches.

### References

- [1] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2O: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 2023.
- [2] Akshat Sharma, Hangliang Ding, Jianping Li, Neel Dani, and Minjia Zhang. Minikv: Pushing the limits of llm inference via 2-bit layer-discriminative kv cache. arXiv preprint arXiv:2411.18077, 2024.
- [3] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms. arXiv preprint arXiv:2310.01801,

- 2023.

[4] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. KVQuant: Towards 10 million context length LLM inference with KV cache quantization. In Advances in Neural Information Processing Systems,

- 2024.

- [5] Yuxuan Yue, Zhihang Yuan, Haojie Duanmu, Sifan Zhou, Jianlong Wu, and Liqiang Nie. WKVQuant: Quantizing weight and key/value cache for large language models gains more. arXiv preprint arXiv:2402.12065, 2024.
- [6] Haojun Xia, Xiaoxia Wu, Jisen Li, Robert Wu, Junxiong Wang, Jue Wang, Chenxi Li, Aman Singhal, Alay Dilipbhai Shah, Alpay Ariyak, Donglin Zhuang, Zhongzhu Zhou, Ben Athiwaratkun, Zhen Zheng, and Shuaiwen Leon Song. Kitty: Accurate and efficient 2-bit KV cache quantization with dynamic channel-wise precision boost. arXiv preprint arXiv:2511.18643, 2025.
- [7] Zunhai Su, Zhe Chen, Wang Shen, Hanyu Wei, Linge Li, Huangqi Yu, and Kehong Yuan. RotateKV: Accurate and robust 2-bit KV cache quantization for LLMs via outlier-aware adaptive rotations. arXiv preprint arXiv:2501.16383, 2025.
- [8] Haoyang LI, Yiming Li, Anxin Tian, Tianhao Tang, Zhanchao Xu, Xuejia Chen, Nicole HU, Wei Dong, Li Qing, and Lei Chen. A survey on large language model acceleration based on KV cache management. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=z3JZzu9EA3.
- [9] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-free 4-bit inference in rotated LLMs. In Advances in Neural Information Processing Systems, volume 37, pages 100213–100240, 2024.
- [10] Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Quip#: Even better llm quantization with hadamard incoherence and lattice codebooks. Proceedings of machine learning research, 235:48630, 2024.
- [11] Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. Spinquant: Llm quantization with learned rotations. arXiv preprint arXiv:2405.16406, 2024.
- [12] Jinda Jia, Jisen Li, Zhongzhu Zhou, Jung Hwan Heo, Jue Wang, Tri Dao, Shuaiwen Leon Song, Ben Athiwaratkun, Chenfeng Xu, Tianyi Zhang, et al. SAW-INT4: System-aware 4-bit KV-cache quantization for real-world LLM serving. arXiv preprint arXiv:2604.19157, 2026.
- [13] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, 2023.
- [14] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.

- [15] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023. URL https://arxiv.org/abs/2307.08691.
- [16] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37:68658–68685, 2024.
- [17] Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, and Tri Dao. Flashattention-4: Algorithm and kernel pipelining co-design for asymmetric hardware scaling. arXiv preprint arXiv:2603.05451, 2026.
- [18] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Efficient execution of structured language model programs. In Advances in Neural Information Processing Systems, 2024.
- [19] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.
- [20] Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M. De Sa. QuIP: 2-bit quantization of large language models with guarantees. In Advances in Neural Information Processing Systems, 2023.
- [21] James W. Cooley and John W. Tukey. An algorithm for the machine calculation of complex fourier series. Mathematics of Computation, 19(90):297–301, 1965. doi: 10.1090/ S0025-5718-1965-0178586-1.
- [22] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. arXiv preprint arXiv:2402.02750, 2024.
- [23] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and efficient post-training quantization for large language models. In Proceedings of the 40th International Conference on Machine Learning, 2023.
- [24] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In International Conference on Learning Representations, 2023.
- [25] Philippe Tillet, H. T. Kung, and David Cox. Triton: An intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, 2019.
- [26] Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov. Flash-decoding for long-context inference. https://pytorch.org/blog/flash-decoding/, 2023. PyTorch Blog. Accessed: 2026-05-06.
- [27] Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [28] GLM Team. ChatGLM: A family of large language models from GLM-130B to GLM-4 all tools. arXiv preprint arXiv:2406.12793, 2024.
- [29] Mathematical Association of America. AIME 2025: American invitational mathematics examination. https://maa.org/math-competitions/aime, 2025.
- [30] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof Q&A benchmark. Conference on Language Modeling, 2024.
- [31] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

- [32] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.
- [33] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. NeurIPS Datasets and Benchmarks Track, 2021.
- [34] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, et al. RULER: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.
- [35] vibhavagarwal5. TurboQuant: 2-bit KV cache compression with 4x capacity. https:// github.com/vllm-project/vllm/pull/38479, 2026. vLLM pull request #38479.
- [36] Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni. TurboQuant: Online vector quantization with near-optimal distortion rate. arXiv preprint arXiv:2504.19874, 2025.
- [37] Ky Fan. Maximum properties and inequalities for the eigenvalues of completely continuous operators. Proceedings of the National Academy of Sciences of the United States of America, 37

(11):760–766, 1951. doi: 10.1073/pnas.37.11.760.

- [38] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. arXiv preprint arXiv:2404.14469, 2024.
- [39] Zefan Cai et al. PyramidKV: Dynamic KV cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069, 2024.
- [40] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, 2024.
- [41] Yefei He, Luoming Zhang, Weijia Wu, Jing Liu, Hong Zhou, and Bohan Zhuang. ZipCache: Accurate and efficient KV cache quantization with salient token identification. In Advances in Neural Information Processing Systems, 2024.
- [42] Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, and Tuo Zhao. GEAR: An efficient KV cache compression recipe for near-lossless generative inference of LLM. arXiv preprint arXiv:2403.05527, 2024.
- [43] Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, Chong-Yan Chen, Yu-Fang Hu, Pei-Shuo Wang, Ning-Chi Huang, Luis Ceze, Mohamed S. Abdelfattah, and Kai-Chiang Wu. PALU: Compressing KV-cache with low-rank projection. arXiv preprint arXiv:2407.21118, 2024.
- [44] Chi-Chih Chang, Chien-Yu Lin, Yash Akhauri, Wei-Cheng Lin, Kai-Chiang Wu, Luis Ceze, and Mohamed S. Abdelfattah. xKV: Cross-layer SVD for KV-cache compression. arXiv preprint arXiv:2503.18893, 2025.
- [45] Bokai Lin, Zihao Zeng, Zipeng Xiao, Siqi Kou, Tianqi Hou, Xiaofeng Gao, Hao Zhang, and Zhijie Deng. MatryoshkaKV: Adaptive KV compression via trainable orthogonal projection. arXiv preprint arXiv:2410.14731, 2024.
- [46] Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei Duan, Xingcheng Zhang, and Dahua Lin. SKVQ: Sliding-window key and value cache quantization for large language models. In Conference on Language Modeling, 2024.
- [47] Tengxuan Liu, Shiyao Li, Jiayi Yang, Tianchen Zhao, Feng Zhou, Xiaohui Song, Guohao Dai, Shengen Yan, Huazhong Yang, and Yu Wang. PM-KVQ: Progressive mixed-precision KV cache quantization for long-CoT LLMs. arXiv preprint arXiv:2505.18610, 2025.
- [48] Mohsen Hariri, Alan Luo, Weicong Chen, Shaochen Zhong, Tianyi Zhang, Qifan Wang, Xia Hu, Xiaotian Han, and Vipin Chaudhary. Quantize what counts: More for keys, less for values. arXiv preprint arXiv:2502.15075, 2025.

- [49] Saleh Ashkboos, Mahdi Nikdan, Soroush Tabesh, Roberto L. Castro, Torsten Hoefler, and Dan Alistarh. HALO: Hadamard-assisted lower-precision optimization for LLMs. arXiv preprint arXiv:2501.02625, 2025.
- [50] Seonggon Kim, Juncheol Shin, Seung-taek Woo, and Eunhyeok Park. HOT: Hadamard-based optimized training. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4787–4796, 2025.
- [51] Vage Egiazarian, Roberto L. Castro, Denis Kuznedelev, Andrei Panferov, Eldar Kurtic, Shubhra Pandit, Alexandre Noll Marques, Mark Kurtz, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Bridging the gap between promise and performance for microscaling FP4 quantization. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=zCBGe9AqJZ.
- [52] Utkarsh Saxena and Kaushik Roy. KVLinC: KV cache quantization with hadamard rotation and linear correction. arXiv preprint arXiv:2510.05373, 2025.
- [53] Patrick H. Chen, Hsiang-Fu Yu, Inderjit S. Dhillon, and Cho-Jui Hsieh. DRONE: Data-aware low-rank compression for large NLP models. Advances in Neural Information Processing Systems, 34:29321–29334, 2021.
- [54] Zhihang Yuan, Yuzhang Shang, Yang Song, Qiang Wu, Yan Yan, and Guangyu Sun. ASVD: Activation-aware singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821, 2023.
- [55] Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. SVD-LLM: Truncation-aware singular value decomposition for large language model compression. arXiv preprint arXiv:2403.07378, 2024.
- [56] Xin Wang, Samiul Alam, Zhongwei Wan, Hui Shen, and Mi Zhang. SVD-LLM v2: Optimizing singular value truncation for large language model compression. In Proceedings of the Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, 2025.
- [57] Yibo Yang, Xiaojie Li, Zhongzhu Zhou, Shuaiwen Leon Song, Jianlong Wu, Liqiang Nie, and Bernard Ghanem. CorDA: Context-oriented decomposition adaptation of large language models. arXiv preprint arXiv:2406.05223, 2024.
- [58] Seonggon Kim, Taehyeon Kim, and Eunhyeok Park. HaPPI: Efficient KV cache compression with hadamard PCA-based power iteration. OpenReview, 2025. URL https://openreview. net/forum?id=BRDgQzdtWr. Submitted to ICLR 2026.
- [59] Zhongzhu Zhou, Fengxiang Bie, Ziyan Chen, Zhenyu Zhang, Yibo Yang, Junxiong Wang, Ben Athiwaratkun, Xiaoxia Wu, and Shuaiwen Leon Song. CARE: Covariance-aware and rankenhanced decomposition for enabling multi-head latent attention. In International Conference on Learning Representations, 2026.
- [60] Xianglong Yan, Zhiteng Li, Tianao Zhang, Linghe Kong, Yulun Zhang, and Xiaokang Yang. RecalKV: Low-rank KV cache compression via head reordering and offline calibration. arXiv preprint arXiv:2505.24357, 2025.
- [61] Yixuan Wang, Haoyu Qiao, Lujun Li, Qingfu Zhu, and Wanxiang Che. CommonKV: Compressing KV cache with cross-layer parameter sharing. arXiv preprint arXiv:2508.16134, 2025.
- [62] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration. In Proceedings of Machine Learning and Systems, 2024.
- [63] Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. OmniQuant: Omnidirectionally calibrated quantization for large language models. arXiv preprint arXiv:2308.13137, 2023.

- [64] Akshat Ramachandran, Marina Neseem, Charbel Sakr, Rangharajan Venkatesan, Brucek Khailany, and Tushar Krishna. ThinKV: Thought-adaptive KV cache compression for efficient reasoning models. arXiv preprint arXiv:2510.01290, 2025. URL https://arxiv.org/ abs/2510.01290.

### A Additional Details and Theoretical Analysis

#### A.1 Hadamard Transform

The Hadamard transform is an orthogonal mixing transform with entries of equal magnitude. We use the normalized Walsh-Hadamard matrix HHad, which satisfies HHad⊤ HHad = I. Its dimension is inferred from the key or value vector being transformed. For power-of-two dimensions, it can be defined recursively by H1 = [1] and

1 √2

Hm Hm Hm −Hm

H2m =

.

In low-bit LLM quantization, Hadamard transforms are commonly used to spread outlier energy across channels before rounding [9]. In OSCAR, HHad is composed after the attention-aware covariance rotation to further smooth the rotated coordinates while keeping the transform fixed and efficient.

#### A.2 Principal Component Analysis

Given a symmetric positive semidefinite matrix A ∈ Rd×d, principal component analysis (PCA) diagonalizes it via

A = V ΛV ⊤, Λ = diag(λ1,...,λr,...), λ1 ≥ λ2 ≥ ··· ≥ 0.

Below we recall the important proposition of PCA [37]. Proposition 1 (Indication of top-r eigenvectors). For any 1 ≤ r ≤ d, let

Vr = [v1,...,vr] ∈ Rd×r. Then

Vr = argmaxU⊤U=Irtr(U⊤AU), and the optimal value is

r

tr(U⊤AU) =

max

λi.

U⊤U=Ir

i=1

Note that in our setting, when A is chosen as a target covariance induced by the attention computation, PCA provides an orthogonal basis aligned with the directions that are most important to preserve under KV compression.

Proof. Let U ∈ Rd×r satisfy U⊤U = Ir. Since A = V ΛV ⊤, we have

tr(U⊤AU) = tr(U⊤V ΛV ⊤U). Define

B = V ⊤U ∈ Rd×r. Because V is orthogonal,

B⊤B = U⊤V V ⊤U = U⊤U = Ir. Hence the columns of B are orthonormal. Using cyclicity of trace,

tr(U⊤AU) = tr(B⊤ΛB) = tr(ΛBB⊤). Write

P = BB⊤ ∈ Rd×d. Then P is an orthogonal projection matrix of rank r, so

d

0 ≤ Pii ≤ 1,

Pii = tr(P) = r. Therefore

i=1

d

tr(U⊤AU) = tr(ΛP) =

λiPii.

i=1

Set

wi = Pii, i = 1,...,d. Then

d

0 ≤ wi ≤ 1,

wi = r, and thus

i=1

d

tr(U⊤AU) =

λiwi.

i=1

Since λ1 ≥ λ2 ≥ ··· ≥ λd, the quantity di=1 λiwi is maximized by assigning full weight to the largest r eigenvalues, namely

w1 = ··· = wr = 1, wr+1 = ··· = wd = 0. Hence

r

tr(U⊤AU) ≤

λi.

i=1

Now choose U = Vr. Then

Vr⊤AVr = Vr⊤V ΛV ⊤Vr = diag(λ1,...,λr), so

r

tr(Vr⊤AVr) =

λi.

i=1

Thus Vr attains the maximum, and therefore

Vr ∈ argmaxU⊤U=Irtr(U⊤AU).

| |
|---|

#### A.3 Target Covariance

Given a matrix X = [x1;...;xN] ∈ RN×d whose rows are feature vectors, its empirical covariance, assuming the rows have been centered, is

N

1 N

1 N

x⊤i xi ∈ Rd×d.

X⊤X =

CX =

i=1

In this paper, we use the uncentered second moment in the same algebraic form and refer to it as a covariance target. For any direction u ∈ Rd with ∥u∥2 = 1, the scalar

N

1 N

u⊤CXu =

(xiu)2

i=1

measures the average squared magnitude of the rows of X along direction u. Thus, the eigenvectors of CX identify directions where the row activations have large energy, and PCA uses these eigenvectors

- as a basis aligned with the dominant second-order structure of X.

In OSCAR, the covariance target is not chosen only from the raw cached tensors. The reason is that the rotation should reduce the downstream attention error, not merely the Euclidean reconstruction error of K and V . For keys, the empirical logit distortion satisfies

∥QK⊤ − Q K⊤∥2F = tr (K − K)Q⊤Q(K − K)⊤ .

This identity shows that key errors are weighted by the query covariance Q⊤Q. Heuristically, directions with large eigenvalues of Q⊤Q are directions in which the queries have large energy, so quantization error in those directions causes larger perturbations to the attention logits. Therefore, we use Q⊤Q as the key-side target covariance for determining the raw key rotation.

For values, the downstream output distortion satisfies

##### ∥SV − S V ∥2F = tr (V − V )⊤S⊤S(V − V ) . Equivalently, the matrix

V ⊤S⊤SV = (SV )⊤(SV ) is the target covariance of the attention-weighted value activations SV . It measures which value feature directions remain large after aggregation by the attention scores. Hence, V ⊤S⊤SV is an attention-induced value target covariance, while V ⊤V is only the raw-cache covariance of values.

Therefore, Q⊤Q and V ⊤S⊤SV are covariance targets in our context: they are second-moment matrices induced by the downstream attention computation. This is why OSCAR estimates rotations from these attention-aware covariance targets rather than from raw-cache targets such as K⊤K and V ⊤V .

- A.4 Intuition of the combination RK = UQHHadPK The rotation applied to keys before quantization in OSCAR is the product of three orthogonal factors,

##### RK = UQ HHad PK,

where UQ is the eigenbasis of the key target covariance CQ = Q⊤Q (Sec. A.3), HHad is the normalized Walsh–Hadamard matrix (Sec. A.1), and PK is a permutation matrix that we describe below. The same construction is used for RV with CS = V ⊤S⊤SV in place of CQ. This subsection explains, before proceeding to the formal proof in Sec. A.6, why each factor is needed and why they appear in this order. Each factor addresses a distinct failure mode of per-group low-bit quantization.

Two effects of RK. Recall from Sec. A.3 that the rotated cache K˜ = KRK is what the quantizer sees, and the resulting attention-logit distortion is

##### QK⊤ − Q K⊤ 2F = tr RK⊤CQRK · EK ,

where EK is the residual covariance of K˜ − Q(K˜). The rotation enters in two distinct places:

- (i) it shapes RK⊤CQRK, the query-importance metric in the rotated coordinates;
- (ii) it shapes EK, because the quantizer’s per-token, per-group min–max scale (Sec. A.5) depends on the channelwise distribution of K˜.

The three factors UQ, HHad, PK each address one of these effects in turn. UQ: align channels with query-importance directions. Eigendecompose

##### CQ = UQΛQUQ⊤, ΛQ = diag(λ1,...,λd), λ1 ≥ ··· ≥ λd ≥ 0. Setting RK = UQ diagonalizes the importance metric:

RK⊤CQRK = ΛQ. Geometrically, the j-th channel of K˜ = KUQ is the projection of each key onto the j-th query principal direction uj, and λj = ∥Quj∥22 measures how much total query energy lies along uj. Quantization noise on channel j then propagates to the attention logits with weight λj: large-λ channels are exactly those whose error is most amplified, while small-λ channels can absorb error almost for free. Sec. A.6 formalizes this via a rearrangement-inequality argument on the surrogate L˜K(Rk) = tr(Rk⊤CQRkEK): under the frozen-residual assumption, RK = UQ is the orthogonal rotation that minimizes the importance-weighted error.

This is the right first step but it leaves a per-group quantization problem. The eigenspectrum ΛQ of CQ in real LLM activations is sharply anisotropic: a few eigenvalues carry most of tr(CQ). After RK = UQ, the rotated key channels inherit this imbalance, with a few channels having much larger variance than the rest. A per-group min–max quantizer with a single scale per group then sets its scale based on the largest channel in each group, and the smaller channels in the same group are quantized

- at a granularity far coarser than their dynamic range. Hence EK has a few very large diagonal entries and many tiny ones, and tr(EK) is dominated by the worst groups.

HHad: equalize on the importance metric, suppress outliers on K. The UQ step is chosen for the surrogate, which is a Q-side criterion (CQ = Q⊤Q). The Hadamard composition serves two complementary purposes: on the surrogate side, it equalizes the rotated importance metric across channels; on the actual K distribution, it mixes outlier energy across channels so that the per-group min–max quantizer (Sec. A.5) sees comparable per-channel magnitudes. The first claim is exact; the second is what the standard outlier-suppression literature relies on [9] and only requires that the Hadamard column weights all have equal magnitude.

The exact statement on the importance metric is the following. Lemma 1 (Diagonal equalization under Hadamard conjugation). Let Λ ∈ Rd×d be diagonal and HHad the normalized Walsh–Hadamard matrix. Then

1 d

HHad⊤ ΛHHad ii =

tr(Λ), i = 1,...,d.

√

Proof. Every entry of HHad has magnitude 1/

d. Then (HHad⊤ ΛHHad)ii = j HHad[j,i]2 Λjj =

- 1 d j Λjj = d1tr(Λ), independent of i. Applying the lemma to Λ = ΛQ (which is diagonal because of the UQ step) gives

| |
|---|

1 d

RK⊤CQRK ii = HHad⊤ ΛQHHad ii =

tr(CQ), ∀i.

The peaky eigenspectrum that UQ exposed has been replaced by a uniform diagonal: every channel now carries the same importance weight tr(CQ)/d in the rotated frame.

The K-side effect is heuristic but practically important. After RK = UQHHad, each row of K˜ = KRK is a Walsh-signed sum of the principal-direction projections KUQ, with every entry of weight ±1/

##### √

d. A per-token outlier on any single principal direction is thus redistributed in magnitude 1/

##### √

d across all d channels, shrinking the per-token max by roughly the same factor and bringing

the per-group min–max scale into a much tighter range. The frozen residual EK becomes much smaller in trace and far more uniform across i. The price paid is that the importance metric is no

longer diagonal: RK⊤CQRK = HHad⊤ ΛQHHad has off-diagonals that couple channels. In the regime where per-group quantization noise dominates tr(EK), the reduction in tr(EK) wins over the loss of importance alignment.

Empirical aside: UQ does not accidentally diagonalize ΣK. One might hope that, because WQ and WK are trained on the same activation distribution, CQ = Q⊤Q and ΣK = K⊤K would share an eigenbasis, so that UQ would simultaneously serve both the Q-side (importance) and K-side (variance) objectives. Empirically, this is not the case. On Qwen3-8B with 2000 calibration tokens per layer, the top-8 eigenvector self-alignment 8k=1 |UQQ⊤UQK|/8 ranges from 0.05 to 0.15 across the 36 layers — essentially indistinguishable from the random alignment 1/

##### √

d ≈ 0.09 expected at d = 128. Rotating ΣK by UQ typically reduces the fraction of ΣK’s energy concentrated on the diagonal (e.g., layer 1: 0.90 raw vs. 0.09 after UQ rotation; layer 16: 0.16 vs. 0.07). Figure 7 shows a representative layer.

[Figure 49]

- Figure 7: Layer 16 of Qwen3-8B. From left: heatmap of Q⊤Q; heatmap of K⊤K (same colour scale);

eigenvalue spectra; absolute eigenvector alignment |UQ⊤UQ| with both bases sorted by descending eigenvalue. A diagonal-bright alignment matrix would indicate a shared eigenbasis; the observed

uniform-texture pattern shows the eigenbases are nearly uncorrelated.

The construction is nevertheless robust because the two factors target different things. Lemma 1 guarantees exact diagonal equalization of the importance metric CQ regardless of the eigenbasis of ΣK, since the lemma only relies on ΛQ being diagonal in the post-UQ frame. The K-side outliersuppression effect of HHad described above also does not require alignment between UQ and the eigenbasis of ΣK: the Walsh-signed mixing shrinks per-token outliers by a factor of 1/

##### √

d no matter

which orthogonal basis K was placed in beforehand. The Q-side (UQ) and K-side (HHad) factors thus address genuinely independent objectives and compose without interference, rather than relying

on an approximate WQ ≈ WK similarity that does not hold in practice.

PK: balance importance across groups. Hadamard mixing equalizes marginal channel variance, but the per-group quantizer in Sec. A.5 acts on contiguous blocks of GK channels at a time. What matters at the group level, beyond marginal variance, is whether each group sees a balanced sample of the underlying importance directions: a group whose GK channels are dominated by the same handful of large-λj eigenvectors is statistically harder to quantize than a group whose channels span the full range of λj.

OSCAR uses a permuted bit-reversal (PBR) ordering. Let σ be the descending-eigenvalue permutation, so that λσ(0) ≥ λσ(1) ≥ ···, and let β be the bit-reversal permutation on {0,1,...,d − 1}. The permutation matrix PK places the eigenvector with the k-th largest eigenvalue at position β(k), i.e. (PK):,β(k) = eσ(k). For d = 128, this yields the placement

##### top-1  → 0, top-2  → 64, top-3  → 32, top-4  → 96, top-5..8  → 16,80,48,112, ...

The defining property of bit-reversal is that for any power-of-two group size G | d, the top-d/G eigenvectors land in d/G distinct groups, exactly one per group; the same recursive balance holds at every coarser binary level. Composed with the Hadamard, this means each per-group quantization block sees one representative from each level of the importance hierarchy, regardless of which group size GK is chosen at deployment.

Empirical aside: errors separate after attention. Figure 8 compares layer-wise errors on Qwen34B-Thinking-2507, measured on raw KV tensors and on the downstream quantities consumed by attention. OSCAR is not designed to minimize plain Euclidean reconstruction error of K or V , so its raw K-MSE and V -MSE are not always dramatically smaller than rotation-only baselines. The gap becomes much clearer after the attention computation: the errors in QK⊤ and SV are substantially lower, and the advantage further propagates to attention-output errors. This is exactly what the covariance targets optimize: the key rotation is aligned with query-side covariance, while the value rotation is aligned with the score-weighted value covariance. The trend is consistent with the UK intuition above: aligning channels with query-importance directions need not minimize raw K error, but it directly lowers the error after multiplication by Q; analogously, the value target is chosen so that the advantage appears after aggregation by S.

Worked example: Qwen3-4B-Thinking, layer 10, d = 128. We illustrate the three factors on real activations rather than a stylized spectrum. Calibration sample: layer 10 of Qwen3-4B-Thinking-2507 (32 query heads, 8 key/value heads, GQA ratio = 4), T = 8000 tokens of GPQA-diamond context. We use the production importance metric — Q-side query covariance, averaged within each GQA group of 4 query heads and then across all 8 kv-heads,

Hkv

1 Hkv

1 T · g i∈G

##### Q⊤i Qi,

CQ =

h=1

h

The decode-time per-group min–max quantizer uses INT2 with GK = 64 (two groups per head); we deliberately pick GK < d here so the PBR step has a non-trivial effect (see Step 3 for the GK = d degenerate case). The K activations we display are from kv-head 0; the shared rotation RK = UQHHadPK is the one produced by compute_qqt.

Spectrum at layer 10. Eigendecomposing CQ = UQΛQUQ⊤ yields an anisotropic but not pathologically peaky spectrum (sorted descending),

λ1...8 ≈ (162.5, 85.5, 35.6, 17.4, 11.6, 7.5, 6.1, 5.0), λ121...128 ≈ (0.24, 0.22, 0.22, 0.20, 0.18, 0.05, 0.027, 0.009).

2KreconstructonMSE(absouteper-eement)KKNF

###### Per-layer K reconstruction error (Qwen3-4B-Thinking, 8K GPQA prefill, INT2 per-token)

OSCAR (UKHHadPK + sink/recent/clip)

HHad + sink/recent/clip

HHad only (no sink/recent/clip)

- 100

- 101

- 102

Clip-only (no rotation)

Baseline INT2 (no rotation, no clip)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34

Layer index

Figure 8a: K MSE

###### Per-layer QK raw (pre-softmax) error

2FrobenusMSE(absoute)QKQKQKNF

- 101

- 102

- 103

- 104

OSCAR (UKHHadPK + sink/recent/clip)

HHad + sink/recent/clip

HHad only (no sink/recent/clip)

Clip-only (no rotation)

Baseline INT2 (no rotation, no clip)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34

Layer index

Figure 8c: QK⊤ MSE

###### Per-layer attention softmax KL

OSCAR (UKHHadPK + sink/recent/clip)

HHad + sink/recent/clip

()Attenton-scoreKLKLperayerppFPq

- 100

- 101

HHad only (no sink/recent/clip)

Clip-only (no rotation)

Baseline INT2 (no rotation, no clip)

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

10 1

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34

Layer index

Figure 8e: Attention KL

2VreconstructonMSE(absouteper-eement)VVNF

###### Per-layer V reconstruction error

- 100

- 101

OSCAR (UKHHadPK + sink/recent/clip)

HHad + sink/recent/clip

HHad only (no sink/recent/clip)

| |
|---|

Clip-only (no rotation)

| |
|---|

Baseline INT2 (no rotation, no clip)

| |
|---|

10 1

| |
|---|

| |
|---|

10 2

10 3

10 4

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34

Layer index

Figure 8b: V MSE

###### Per-layer attention output SV (post-softmax, pre-WO) error

- 100

- 101

OSCAR (UKHHadPK + sink/recent/clip)

2FrobenusMSE(absoute)SVSVSVNF

HHad + sink/recent/clip

HHad only (no sink/recent/clip)

| |
|---|

Clip-only (no rotation)

Baseline INT2 (no rotation, no clip)

| |
|---|

10 1

| |
|---|

10 2

10 3

10 4

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34

Layer index

Figure 8d: SV MSE

2()()AttentonbockoutputMSE(absoute)AVWAVWNOOF

###### Per-layer attention block output error (post-WO)

OSCAR (UKHHadPK + sink/recent/clip)

HHad + sink/recent/clip

- 100

- 101

HHad only (no sink/recent/clip)

Clip-only (no rotation)

Baseline INT2 (no rotation, no clip)

| |
|---|

| |
|---|

10 1

| |
|---|

10 2

10 3

10 4

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34

Layer index

Figure 8f: Attention-output MSE

- Figure 8: OSCAR’s advantage appears on attention-consumed quantities on Qwen3-4BThinking-2507. Raw K/V reconstruction MSE alone does not fully explain downstream quality, because OSCAR does not optimize raw-cache reconstruction. Instead, it uses query- and score-aware covariance targets, which more directly reduce QK⊤, SV , and attention-output errors.

The trace is tr(CQ) ≈ 443. The top eigendirection λ1 alone accounts for 36.6% of the trace and the top eight carry 74.7%: λ1/λ¯ ≈ 46.9 where λ¯ = tr(CQ)/d = 3.46. Importance is concentrated in a low-rank subspace, but not as extremely as one might guess from K’s own variance — instead, covariance Q⊤Q reflects what the attention layer actually queries, which is a softer concentration than where K’s variance happens to land.

One token, all 128 channels, split into two GK = 64 groups. The INT2 per-group min–max quantizer fits its scale to maxi∈g K˜t,i − mini∈g K˜t,i /3 per token t per group g. We visualize what the quantizer sees by printing all 128 channels for a single typical non-sink token (t = 5, kv-head 0), reshaped into an 8 × 16 grid (channel j at grid position (⌊j/16⌋, j mod 16)). The horizontal rule splits the grid into the two GK = 64 groups: rows 0–3 are group 0 (channels 0–63), rows 4–7 are group 1 (channels 64–127). Each matrix is annotated with the per-group max–min.

The raw token in the channel basis:





3.42 −0.61 1.59 0.27 −3.14 −0.05 3.55 1.09 2.05 −1.05 1.86 −0.03 −0.42 −0.89 −1.54 2.06

−1.74 −0.15 −0.39 −0.89 −0.02 2.55 0.11 1.79 −0.28 0.25 −1.40 −6.09 −1.04 −0.12 0.64 −0.95 9.06 1.17 0.08 −1.49 0.29 1.13 0.47 0.17 1.22 0.55 14.19 −0.26 −0.09 0.20 2.39 0.40 1.09 −0.06 −30.62 0.00 1.03 0.57 −1.30 2.47 −0.11 −0.08 0.56 −0.47 0.45 −0.88 2.31 −1.05

Kt,· =

- 0.50 0.44 0.10 −0.35 −0.78 1.46 −1.16 2.56 1.64 −0.09 0.90 2.33 1.09 2.72 0.95 0.20
- 1.30 1.82 0.54 1.27 −1.05 0.50 −4.47 1.20 0.32 1.36 −0.81 −0.04 2.27 −0.77 0.64 1.27

 

 

- 0.03 −0.20 0.41 0.58 0.79 1.05 0.46 −0.09 0.49 0.29 −0.02 0.20 1.36 0.77 −1.01 −2.33
- 1.19 −1.77 −0.01 −0.49 0.47 0.91 0.42 0.17 1.38 −0.39 −0.00 0.43 −0.98 −0.50 0.17 −0.50

group 0: max−min = 44.81 | group 1: max−min = 7.19.

The two groups are wildly unbalanced: group 0 holds both outliers (−30.62 at j = 50 and 14.19 at j = 42); group 1 is benign. Group 0’s INT2 scale must fit a range of 44.81, wasting two bits on a few extreme values while the other ∼ 60 channels of that group quantize at granularity an order of magnitude coarser than their actual dynamic range.

- Step 1: K˜ = KUQ (eigenbasis rotation). Projecting K onto the Q⊤Q eigendirections gives:

(KUQ)t,· =



 

0.16 −1.75 1.86 −2.42 1.11 1.05 −0.69 0.26 1.11 −0.55 3.39 −1.80 1.78 1.65 −1.49 −1.32 0.05 −0.18 −2.00 −1.27 3.20 −1.47 0.31 −1.07 0.87 −1.54 −3.78 −0.89 1.60 1.03 0.50 0.55

- 0.49 0.55 −0.90 −0.05 0.17 0.20 0.78 −1.21 −1.06 −1.70 0.63 0.12 −0.28 2.99 2.88 −0.97

- −1.42 2.12 −0.94 −2.88 1.30 −0.42 1.37 0.58 −1.46 −2.04 0.28 −0.28 −2.63 −0.20 −1.77 −0.83

0.50 0.12 −1.19 0.81 −1.90 0.40 −1.31 −0.15 −2.31 1.52 3.20 −0.46 −0.77 −0.09 −1.34 1.72 0.40 −0.72 −0.57 0.40 0.84 −0.72 −1.94 −0.11 1.26 0.07 0.09 −0.06 0.87 −0.92 −0.63 1.19

- −2.31 1.56 1.90 1.26 1.16 −0.72 −0.56 −0.48 1.21 0.20 −0.26 −1.78 1.71 −1.28 1.07 1.27
- −3.32 0.47 −0.43 −1.77 −1.42 1.42 4.99 −1.91 −0.25 −1.02 0.51 −1.65 −1.11 −8.07 15.28 29.77



 

group 0: max−min = 7.16 | group 1: max−min = 37.85.

The raw outlier hasn’t been destroyed; it has been redistributed into the Q⊤Q eigenbasis. Because UQ is built from Q⊤Q, not K⊤K, the K-spike at j = 50 does not project onto a single UQ

direction — it spreads across the lower-eigenvalue tail with its largest projection landing in the bottomrank eigenchannels (the matrix’s bottom-right entries: 29.77,15.28,−8.07,...). The diagonal of UQ⊤CQUQ is exactly ΛQ, so the per-channel importance is now exactly the eigenvalue, but the Kvariance has been pushed into low-importance channels with no improvement in per-group dynamic range — in fact the imbalance between groups has flipped and grown: group 1’s range jumps from 7.19 (raw) to 37.85, while group 0’s drops from 44.81 to 7.16. Per-group INT2 is now harder, not easier.

- Step 2: K˜ = KUQHHad (Walsh-Hadamard mixing). Multiplying by the orthonormal WalshHadamard HHad ∈ R128×128 rewrites each output channel as a ±1-weighted sum of all 128 eigenchannels (each scaled by 1/√128). The single-channel spike of 29.77 is distributed across all 128 outputs with mixed signs:

(KUQHHad)t,· =



 

- 2.24 1.08 −3.73 0.26 −5.76 0.92 4.08 −3.39 −3.72 0.29 5.55 −2.67 3.19 −3.70 −4.16 1.80

−0.87 0.64 4.12 −3.18 5.12 1.16 −6.79 0.97 1.91 −2.08 −4.91 4.21 −1.67 1.97 4.47 −0.63

- −2.86 0.66 6.73 1.83 4.37 1.49 −5.77 0.72 1.52 −0.05 −4.94 3.97 −3.09 2.28 4.34 −0.37 1.90 −1.30 −6.06 −1.90 −2.58 3.27 4.18 −3.10 −2.31 0.88 5.04 −2.61 −0.07 −1.10 −2.17 4.09

- −3.71 2.31 6.89 −1.79 1.12 −0.12 −5.61 3.44 4.10 0.85 −3.94 3.17 −3.51 0.62 7.03 −5.15
- 4.11 1.23 −4.84 0.19 −3.60 2.16 3.36 −0.01 −4.31 0.17 3.97 −2.02 2.49 −0.04 −4.11 2.57

- 3.72 2.20 −3.51 1.67 −5.30 −0.01 4.53 −2.49 −3.30 0.32 2.06 −4.40 3.14 −1.10 −6.54 1.40

−2.85 1.72 5.21 −0.32 4.97 1.00 −5.68 −1.24 3.87 −1.45 −5.40 1.67 −4.30 1.20 3.73 −3.39



 

group 0: max−min = 13.52 | group 1: max−min = 13.58. Every one of the 128 entries now sits in a tight band [−6.79, 7.03], the two groups are within 0.5% of each other in dynamic range, and the absolute range collapses from 37.85 (Step 1) to 13.58 — a 2.8× reduction. The importance metric also equalizes exactly: Lemma 1 applied to ΛQ predicts every diagonal entry of HHad⊤ ΛQHHad collapses to the constant tr(ΛQ)/d = 443/128 ≈ 3.46, regardless of how peaky ΛQ was. Table 10 confirms maxi(RK⊤CQRK)ii/meani(·) = 1.00 exactly: the 46.9× peak in the Q⊤Q importance is compressed to 1.00×. This is the key invariant Step 2 produces; the per-group dynamic range alone is not yet better than pure Hadamard (compare to Step 5 below).

- Step 3: K˜ = KUQHHadPK (permuted bit-reversal). PK is a permutation matrix that reorders the 128 output channels. After UQHHad each Walsh output is a ±1-weighted sum of all eigenchannels, but the variances of those sums are not identical (driven by which Walsh row a high-eigenvalue eigvec activates most strongly in). PBR places the k-th largest eigenvector at Walsh output index β(k) (the

bit-reversal of k). At GK = 64 this guarantees the top-2 eigenvectors split into the two groups (one each), the top-4 distribute one-per-quartile, and so on for any power-of-two grouping, so no single group inherits a disproportionate share of the high-energy eigenchannels.





2.24 −3.71 −2.86 3.72 −0.87 4.11 1.90 −2.85 −3.72 4.10 1.52 −3.30 1.91 −4.31 −2.31 3.87 −5.76 1.12 4.37 −5.30 5.12 −3.60 −2.58 4.97 3.19 −3.51 −3.09 3.14 −1.67 2.49 −0.07 −4.30 −3.73 6.89 6.73 −3.51 4.12 −4.84 −6.06 5.21 5.55 −3.94 −4.94 2.06 −4.91 3.97 5.04 −5.40

4.08 −5.61 −5.77 4.53 −6.79 3.36 4.18 −5.68 −4.16 7.03 4.34 −6.54 4.47 −4.11 −2.17 3.73 1.08 2.31 0.66 2.20 0.64 1.23 −1.30 1.72 0.29 0.85 −0.05 0.32 −2.08 0.17 0.88 −1.45 0.92 −0.12 1.49 −0.01 1.16 2.16 3.27 1.00 −3.70 0.62 2.28 −1.10 1.97 −0.04 −1.10 1.20 0.26 −1.79 1.83 1.67 −3.18 0.19 −1.90 −0.32 −2.67 3.17 3.97 −4.40 4.21 −2.02 −2.61 1.67

(KUQHHadPK)t,· =

 

 

−3.39 3.44 0.72 −2.49 0.97 −0.01 −3.10 −1.24 1.80 −5.15 −0.37 1.40 −0.63 2.57 4.09 −3.39

##### group 0: max−min = 13.82 | group 1: max−min = 9.36.

The set of 128 values is identical to Step 2 — only column-index positions are shuffled — but PK has gathered most of the high-magnitude entries ({7.03,−6.54,−6.79,−6.06,6.89,6.73,...}) into group 0 (rows 0–3) and left group 1 (rows 4–7) with a much tighter [−5.15,4.21] range. On this particular token PBR has increased one group’s dynamic range slightly (13.52 → 13.82) while shrinking the other’s (13.58 → 9.36); the residual on the tighter group drops as (9.36/13.58)2 ≈ 0.48 while the looser group is essentially unchanged. Averaged over all T = 8000 tokens, this gathering effect reduces the mean per-group max–min from 13.40 (UQHHad) to 11.58 (OSCAR), translating to

a 1.23× reduction in tr(EK) (208 → 169). At this layer, PBR carries most of OSCAR’s quantizationresidual improvement over UQHHad alone; the role of UQHHad is to make Lemma 1 fire so that the importance metric is equalized exactly (max/mean = 1.00).

Degenerate case (GK = d). pure Hadamard, QuaRot-style. A natural simpler alternative is to drop UQ entirely and rotate by HHad alone:





2.20 −0.20 3.44 2.25 −1.33 2.33 3.83 2.71 −2.34 −5.96 4.20 5.20 −3.46 −0.77 4.42 4.48 7.15 5.23 −4.17 −3.54 5.85 5.28 −3.61 −1.41 1.05 3.94 −1.22 −2.85 1.48 1.03 2.05 −0.46

1.02 −1.19 −2.61 −2.01 0.93 3.00 −1.01 −3.19 3.09 2.20 −5.85 −5.58 4.90 3.06 −4.39 −3.55 −3.07 −3.65 0.48 4.83 −3.59 −2.92 3.38 5.09 −2.81 −0.89 1.22 0.46 −3.49 −2.21 2.88 0.84 −1.53 1.03 0.94 0.19 −2.30 0.20 5.16 1.89 −1.37 −3.52 4.74 4.87 −3.51 −4.25 4.14 5.80

(KHHad)t,· =

 

 

5.48 5.35 −4.44 −1.88 7.12 5.46 −1.44 −2.71 1.37 3.77 0.45 −2.59 3.21 2.03 −0.67 0.81 −1.55 2.51 −2.80 −3.34 0.29 1.81 −2.03 −0.88 5.68 4.70 −6.89 −5.69 1.38 1.07 −4.79 −2.27 −3.94 −4.43 2.41 5.23 −2.12 −5.19 2.62 2.32 −1.78 −2.21 0.40 0.43 −0.79 −1.48 1.86 2.20

group 0: max−min = 13.12 | group 1: max−min = 14.01. Hadamard mixing on its own balances the two groups (13.12 ≈ 14.01) and on this token gives a per-group dynamic range comparable to UQHHad (Step 2). The crucial structural difference is in the importance metric: CQ is not diagonal in the channel basis, so Lemma 1 does not apply to HHad⊤ CQHHad. We measure max/mean = 1.72 rather than 1.00 — the importance floor is not equalized.

what the quantizer sees on K˜ = KRK importance INT2 residual

maxi(RK⊤CQRK)ii meani(·)

Rotation RK max |K˜| per-group max–min (GK = 64)

tr(EK)

I (no rotation) 36.3 25.7 43.0 233 HHad (pure Hadamard, QuaRot-style) 11.1 13.3 1.72 206 UQ (eigenbasis alone) 29.4 22.4 46.9 1354 UQ HHad 7.46 13.4 1.00 208 UQ HHad PK (OSCAR) 7.46 11.6 1.00 169

Table 10: Layer 10 of Qwen3-4B-Thinking, kv-head 0, d = 128, INT2 per-group (GK = 64, two groups per head), T = 8000 tokens. CQ = Q⊤Q averaged over GQA groups (production setting). max|K˜| is the largest entry across all T × d positions; per-group max–min is Et maxi∈g K˜t,i − mini∈g K˜t,i averaged over the two GK = 64 groups. The importance column measures how peaky the rotated importance metric’s diagonal is (1.00 = all channels equally important; Lemma 1 forces this when CQ is pre-diagonalized by UQ). tr(EK) is the unweighted INT2 residual. At GK = d = 128 (one group per head) the permutation PK would be a no-op and OSCAR collapses to UQHHad; we deliberately use GK = 64 here so the PBR step has a non-trivial effect. Bold entries are extreme values referenced in the text.

The two “what the quantizer sees” columns make the OSCAR advantage on quantization residual concrete. Pure Hadamard cuts the per-group max–min from 25.7 (raw) to 13.3 (1.9×); OSCAR cuts it further to 11.6 (2.2× over raw). Since INT2 residual scales as the square of dynamic range, the tr(EK) ratios should be (25.7/11.6)2 ≈ 4.9 and (13.3/11.6)2 ≈ 1.3, matching empirical 233/169 = 1.4 and 206/169 = 1.2 (the empirical ratios are smaller than the squared dynamic range because the per-token correlation structure of the residual matters; the relation is only proportional, not exact). PBR alone contributes the bulk of OSCAR’s advantage over UQHHad at this layer (tr(EK): 208 → 169, 1.23×). The importance column tells the orthogonal story: only the UQHHad and OSCAR rows achieve max/mean = 1.00, the exact diagonal equalization predicted by Lemma 1; pure Hadamard alone leaves it at 1.72, and raw / UQ alone leave it at 43–47.

Summary. The three factors are not interchangeable, and each provides a guarantee the others cannot:

- • UQ exposes the importance spectrum by diagonalizing CQ, so per-channel importance becomes exactly λj. By itself it makes per-group dynamic range worse on real activations because the K-variance gets pushed into low-importance directions, not flattened.
- • HHad, composed with UQ, equalizes the importance-metric diagonal to a constant tr(CQ)/d via Lemma 1 (max/mean = 1.00 exactly). Without the UQ pre-step (pure Hadamard alone), the importance metric is not equalized (max/mean = 1.72 here), so quantization error in any single

- rotated channel can land disproportionately on a high-importance direction in CQ depending on the layer/token distribution.
- • PK ensures that for any power-of-two group size GK < d, the importance hierarchy is recursively balanced across groups. This is where most of OSCAR’s empirical tr(EK) reduction over UQHHad comes from on real activations. When GK = d it is a column permutation within a single group and has no effect; the PBR step is strictly beneficial only for finer grouping.

The same three-factor construction defines RV = USHHadPV on the value side, with CS = V ⊤S⊤SV (the attention-score-weighted value covariance) in place of CQ.

#### A.5 Scale Determination and Per-token Clipping Details

Quantization is applied after the composed rotations in Eq. (3). At decoding step t, the key and value rows are first mapped into the rotated coordinates

kt = ktRK, vt = vtRV . Before quantization, the implementation applies token-wise percentile clipping separately to keys and values. Given clip ratios ρK,ρV ∈ (0,1], define

τt(K) = quantileρ

{| kt,c|}dc=1 , τt(V ) = quantileρ

{| vt,c|}dc=1 . The clipped rows are

K

V

ktclip = clip kt,−τt(K),τt(K) , vtclip = clip vt,−τt(V ),τt(V ) .

In the current implementation, the default clip ratios are supplied as fixed runtime/calibration hyperparameters, with typical values ρK = 0.96 and ρV = 0.92.

For keys, the default implementation uses asymmetric INT2 quantization, optionally with groups along the channel dimension. Let b = 2 and qmax = 2b − 1 = 3. For a group g of a clipped rotated key row, write

kt,gclip ∈ R1×G

,

K

where GK is the key quantization group size. The implementation computes a dynamic min–max scale and zero point for each token and group:

a(t,gK) = min

kt,g,cclip , b(t,gK) = max

kt,g,cclip ,

c

c

a(t,gK) s(t,gK)

b(t,gK) − a(t,gK) qmax

, zt,g(K) = −

s(t,gK) =

.

The quantized and dequantized key group is then

Q+ kt,gclip = clip

kt,gclip s(t,gK)

+ zt,g(K) ,0,qmax ,

and

Q kt,gclip = Q− Q+ kt,gclip = s(t,gK) Q+ kt,gclip − zt,g(K) .

Thus, unlike a fixed Frobenius-norm scale, the key scale is computed dynamically from the clipped rotated key row, independently for each token and each group. In our default configuration, GK = 64. Values use the same affine asymmetric INT2 quantization rule. For a group g of a clipped rotated value row, write

vt,gclip ∈ R1×G

,

V

where GV is the value quantization group size. The implementation computes

a(t,gV ) = min

vt,g,cclip , b(t,gV ) = max

vt,g,cclip ,

c

c

and

b(t,gV ) − a(t,gV ) qmax

a(t,gV ) s(t,gV )

s(t,gV ) =

, zt,g(V ) = −

.

The quantized and dequantized value group is

Q+ vt,gclip := clip

vt,gclip s(t,gV )

+ zt,g(V ) ,0,qmax ,

and

Q vt,gclip := s(t,gV ) Q+ vt,gclip − zt,g(V ) .

Finally, after quantization–dequantization in the rotated coordinates, the approximations are mapped back to the original coordinate system:

kt = Q ktclip RK⊤, vt = Q vtclip RV⊤.

#### A.6 Proof of Theorem. 1

We prove the key-side statement first. The row-vector rotation convention gives the frozen-error surrogate

LK(Rk) = tr Rk⊤CQRkEK , Rk⊤Rk = Id, where

CQ = UQΛQUQ⊤, ΛQ = diag(λ1,...,λd), λ1 ≥ ··· ≥ λd ≥ 0, and

EK = diag(µ1,...,µd), µ1 ≤ ··· ≤ µd. Here EK is treated as a fixed frozen residual covariance matrix. Let

Rk = UQZ, Z⊤Z = Id. Substituting into the objective gives

LK(Rk) = tr Z⊤UQ⊤CQUQZEK

= tr Z⊤ΛQZEK . Writing Z = (zij), we have

d

Z⊤ΛQZ ii =

λjzji2 . Therefore,

j=1

d

d

µiλjzji2 . (4)

LK(Rk) =

i=1

j=1

Since Z is orthogonal, the matrix W = (wji) with wji = zji2 is doubly stochastic. Hence Eq. (4) is a linear function over the Birkhoff polytope, and its minimum is attained at a permutation matrix.

Thus,

tr Π⊤ΛQΠEK ,

LK(Rk) = min Π∈Pd

min

Rk⊤Rk=Id

where Pd denotes the set of d × d permutation matrices. By the rearrangement inequality, since

λ1 ≥ ··· ≥ λd and µ1 ≤ ··· ≤ µd,

the minimum is achieved by pairing the largest λj with the smallest µi. This is achieved by the identity permutation. Therefore, one minimizer is

##### Z = Id,

and hence

Rk = UQ.

The value-side proof is identical. The value surrogate is

LV (Rv) = tr Rv⊤CSRvEV , Rv⊤Rv = Id, where

CS = USΛSUS⊤, ΛS = diag(ν1,...,νd), ν1 ≥ ··· ≥ νd ≥ 0, and

EV = diag(η1,...,ηd), η1 ≤ ··· ≤ ηd. Again, EV is treated as a fixed frozen residual covariance matrix. Let

Rv = USZV , ZV⊤ZV = Id. Then

LV (Rv) = tr ZV⊤US⊤CSUSZV EV

= tr ZV⊤ΛSZV EV . Writing ZV = (ζij), we obtain

d

d

ηiνjζji2 .

LV (Rv) =

j=1

i=1

As before, (ζji2 ) is doubly stochastic. The minimum is therefore attained at a permutation matrix. Since

ν1 ≥ ··· ≥ νd and η1 ≤ ··· ≤ ηd, the rearrangement inequality shows that the identity permutation is optimal. Thus one minimizer is

ZV = Id, and therefore

Rv = US. This proves both statements.

#### A.7 Justification of Surrogate Objectives in Theorem. 1

To justify the surrogate objectives in Theorem 1, we start from the downstream attention errors induced by quantizing keys and values. For the key branch, the empirical logit distortion on the calibration set is

N

i

qi(kj − kˆj)⊤ 2,

ℓK(Rk) =

i=1

j=1

where kˆj = Q(kjRk)Rk⊤. Writing the rotated quantization residual as

e(jK)(Rk) := Q(kjRk) − kjRk, we have kˆj − kj = e(jK)(Rk)Rk⊤, so that ℓK(Rk) =

j

N

e(jK)(Rk)Rk⊤CjRk e(jK)(Rk)⊤, Cj :=

qi⊤qi.

j=1

i=1

To obtain a tractable objective, we approximate the position-dependent matrices Cj by the global query target covariance

N

1 N

qi⊤qi.

CQ =

i=1

Defining the frozen residual covariance (where Rk is fixed)

N

e(jK)(Rk)⊤e(jK)(Rk),

EK :=

j=1

the key-side frozen-error objective becomes, up to the constant normalization factor N1 ,

L˜K(Rk) = tr(Rk⊤CQRkEK).

Under a similar approximation, the value-side downstream error is heuristically, imitating that of key targets, represented by the frozen-error surrogate

L˜V (Rv) = tr Rv⊤CSRvEV .

Note that the only difference between key and value targets is whether it is purely weighted by target covariance (Q⊤Q for keys) or it is target covariance aware (V ⊤S⊤SV ). Therefore, it is reasonable that we use the same surrogate.

### B Algorithm Flow and Serving Procedure

Algorithm 1 gives the concrete OSCAR procedure used in our experiments and SGLang implementation.

### C Related Work

KV-cache compression. The KV cache has become a major target for post-training compression because its memory footprint grows with both context length and batch size. One line of work reduces or reprioritizes cached content through eviction, budgeted retention, attention-sink preservation, or saliency-aware precision assignment, as in H2O, SnapKV, PyramidKV, StreamingLLM, and ZipCache [1, 38–41]. Other systems compress the cache through low-rank structure or cross-layer sharing, including GEAR, PALU, xKV, and MatryoshkaKV [42–45]. Another line keeps all tokens but lowers the precision of the cached tensors, which is the setting most closely related to OSCAR. KIVI, KVQuant, and WKVQuant study low-bit KV storage and identify channel-wise outliers, especially in keys, as a central obstacle to aggressive quantization [22, 4, 5]. These methods show that KV quantization can be accurate, but they also leave open how far precision can be reduced without changing the serving layout.

2-bit KV quantization. Pushing KV cache quantization to 2 bits usually requires additional structure beyond naive rounding. SKVQ preserves a recent sliding window in higher precision and combines channel reordering with clipped dynamic quantization [46]. Kitty promotes sensitive key-cache channels to higher precision and combines this with a residual cache [6], while PM-KVQ uses progressive mixed precision and block-wise memory allocation for long-CoT models [47]. Related mixed-precision analyses also argue that keys should generally receive more precision than values under a fixed memory budget [48]. RotateKV uses outlier-aware adaptive rotations for 2-bit KV quantization [7]. These methods improve accuracy, but their use of channel-wise precision, residual buffers, progressive bit allocation, or adaptive layouts can complicate integration with paged KVcache serving and fused decode kernels. OSCAR instead targets the same INT2 regime with fixed token-wise transforms and a uniform paged layout, so the quantization path can be fused into production serving.

Rotation-based quantization. Orthogonal transforms are an attractive way to make low-bit quantization more accurate without changing tensor shapes. QuaRot, HALO, and HOT show that rotations can reduce outlier concentration, redistribute activation energy, and improve low-bit rounding behavior [9, 49–51, 12]. In the KV-cache setting, TurboQuant develops a data-oblivious online vector-quantization method based on random rotations, per-coordinate Lloyd–Max quantization, and QJL-style residual coding [36]. KVLinC combines Hadamard rotation with lightweight linear correction to reduce attention errors under aggressive KV quantization [52]. OSCAR follows the same broad rotation principle, but asks which rotation target is appropriate for INT2 KV cache quantization. While [11] effectively mitigates outliers at 4-bit using learned rotations, it collapses at

Algorithm 1 OSCAR calibration, prefill, and decode

- 1: Hyperparameters: sink length S0, recent window W, block size G=128, bit-width b=2
- 2: State: BF16 sink/recent cache, rotated BF16 staging cache, packed INT2 history cache
- 3: procedure Calibrate(D)
- 4: for each layer and KV head do
- 5: Q,K,V ← ForwardDump(D)
- 6: S ← softmaxrow(QK⊤/

√

d + M)

- 7: CK ← N1 n qn⊤qn

- 8: CS ← N1 V ⊤S⊤SV

- 9: UQ ← EigVec(CK), US ← EigVec(CS)
- 10: RK ← UQHHadPbr, RV ← USHHadPbr
- 11: cK,CS ← CalibrateClip(KRK,V RV )
- 12: end for
- 13: return {RK,RV ,cK,CS}
- 14: end procedure
- 15: procedure Prefill(X)
- 16: Q,K,V ← Forward(X)
- 17: K ← KRK, V ← V RV {rotate before prefill write}
- 18: Cachesink ← (K,V )[:S

0]

- 19: Cacherecent ← (K,V )[−W:]
- 20: for i = S0 to |X| − W − 1 do
- 21: QUANTIZEANDWRITE( Ki, Vi,i)
- 22: end for
- 23: end procedure
- 24: procedure DecodeStep(xt)
- 25: qt,kt,vt ← Forward(xt)
- 26: kt ← ktRK, vt ← vtRV
- 27: Append (kt,vt) to Cacherecent; append ( kt, vt) to rotated staging
- 28: if |Cacherecent| > W then
- 29: ( kj, vj) ← PopOldestRotatedStaging()
- 30: QUANTIZEANDWRITE( kj, vj,j)
- 31: end if
- 32: Khist, Vhist ← DequantHistory(CacheINT2)
- 33: Khist ← KhistRK⊤, Vhist ← VhistRV⊤
- 34: Kall,Vall ← Concat(Cachesink, Khist,Cacherecent)
- 35: ot ← softmax(qtKall⊤ /

√

d + M)Vall

- 36: return ot
- 37: end procedure
- 38: function QuantizeAndWrite( k, v,i)
- 39: k¯ ← clip( k,cK), v¯ ← clip( v,CS)
- 40: Ki+ ← AffineINT2(k,G¯ ), Vi+ ← AffineINT2(¯v,G)
- 41: CacheINT2[i] ← (Ki+,Vi+,scale,zero)
- 42: end function

extreme INT2 levels because its rotations fail to align with the actual covariance structures consumed by downstream attention. Furthermore, it focuses primarily on theoretical algorithmic gains, lacking the custom kernels and system-level integration necessary for real-world deployment in modern serving frameworks.

Covariance-aware calibration. A related family uses calibration data or covariance information to choose quantization or compression directions. For weight quantization, GPTQ and QuIP use Hessian information to guide low-bit rounding [24, 20]; for low-rank compression, Drone, ASVD, SVD-LLM, SVD-LLMv2, and CorDA use calibration activations or data-aware statistics to guide factorization [53– 57]. On the KV-cache side, HaPPI, CARE, RecalKV, and CommonKV apply covariance- or Hessianaware factorization to keys and values for KV-side projection or compression [58–61]. Post-training

quantization methods such as SmoothQuant, AWQ, GPTQ, and OmniQuant also use calibration data to estimate scales, equivalent transformations, clipping thresholds, or second-order surrogates before deployment [23, 62, 24, 63]. OSCAR specializes this calibration paradigm to attention: it estimates targets covariance induced by downstream attention, rather than raw-cache statistics, and uses them to derive fixed key/value rotations and clip thresholds for serving-time INT2 KV quantization.

### D Discussion

Limitations. Our theoretical analysis explains why the proposed covariance-target rotations are optimal under a frozen-error surrogate with explicit assumptions (Appendix A), but it is not yet a full proof of optimality for the complete autoregressive decoding process. Tightening this gap—for example, by proving when the calibration estimator is unbiased or optimal for the true attention objective—is an important direction for future work. We also focus on INT2 KV-cache quantization and do not explore other numerical formats or mixed-format designs, such as NVFP4, different weight precisions, or weight/KV precision combinations, which may create different accuracy–systems trade-offs. Similarly, OSCAR uses a fixed bit-width for quantized history tokens and does not study per-token bit allocation or thought-adaptive token precision, which recent work such as ThinKV suggests may be useful for long reasoning traces [64]. On the systems side, our implementation targets SGLang-style paged serving on H100 GPUs; additional tuning is needed to understand the best kernel design on other hardware such as B200. Finally, OSCAR is orthogonal to channel-wise or mixed-precision methods: applying attention-aware/Hadamard rotations inside channel-wise schemes may further improve accuracy, but combining these ideas while preserving an efficient serving layout remains open.

Broader impact. OSCAR reduces KV-cache memory and serving cost, which can make longcontext LLM inference more accessible and energy-efficient. At the same time, cheaper long-context inference may also lower the cost of deploying powerful generative models, including applications with misuse risks; OSCAR does not introduce new model capabilities, but should be deployed with the same safety and usage controls as the underlying LLMs.

### E Additional Experimental Results

- E.1 Additional Motivation Figure The additional motivation figures are shown in Fig 9 and Fig 10.

Qwen3-8B activation distribution layer 10, head 0, history tokens [64, 320) (sink=64, recent=256 excluded)

###### K (raw) max=35.75, max/std=11.4

###### K (+ Hadamard) max=9.61, max/std=5.4

###### K (OSCAR) max=7.37, max/std=4.3

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

30

25

|value|

|value|

|value|

20

15

10

5

0

250

250

250

200

200

200

150

150

150

0

0

0

tokenidx

tokenidx

tokenidx

20

20

20

100

100

100

40

40

40

60

60

60

50

50

50

80

80

80

channel

channel

channel

100

100

100

0

0

0

120

120

120

###### V (raw) max=3.69, max/std=12.5

###### V (+ Hadamard) max=2.06, max/std=7.7

V (OSCAR) max=1.30, max/std=5.5

2.5

1.4

1.4

1.2

1.2

2.0

1.0

1.0

|value|

|value|

|value|

1.5

0.8

0.8

0.6

0.6

1.0

0.4

0.4

0.5

0.2

0.2

0.0

0.0

0.0

250

250

250

200

200

200

150

150

150

0

0

0

tokenidx

tokenidx

tokenidx

20

20

20

100

100

100

40

40

40

60

60

60

50

50

50

80

80

80

channel

channel

channel

100

100

100

0

0

0

120

120

120

- Figure 9: Attention-aware rotations make history-token activations easier to quantize. Hadamard mixing flattens raw activation peaks by spreading energy across channels. OSCAR goes further: the target covariance separates directions that matter more or less to attention, and the Hadamard transform then mixes each part into a more uniform range.

Qwen3-8B activation distribution layer 11, head 0, history tokens [64, 320) (sink=64, recent=256 excluded)

###### K (raw) max=21.25, max/std=10.1

###### K (+ Hadamard) max=8.85, max/std=5.8

###### K (OSCAR) max=6.12, max/std=4.6

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

18

16

14

12

|value|

|value|

|value|

10

8

6

4

2

0

250

250

250

200

200

200

150

150

150

0

0

0

tokenidx

tokenidx

tokenidx

20

20

20

100

100

100

40

40

40

60

60

60

50

50

50

80

80

80

channel

channel

channel

100

100

100

0

0

0

120

120

120

###### V (raw) max=1.46, max/std=9.5

###### V (+ Hadamard) max=1.23, max/std=8.2

V (OSCAR) max=0.69, max/std=5.2

1.4

1.4

0.8

1.2

1.2

1.0

1.0

0.6

|value|

|value|

|value|

0.8

0.8

0.4

0.6

0.6

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

250

250

250

200

200

200

150

150

150

0

0

0

tokenidx

tokenidx

tokenidx

20

20

20

100

100

100

40

40

40

60

60

60

50

50

50

80

80

80

channel

channel

channel

100

100

100

0

0

0

120

120

120

- Figure 10: Attention-aware rotations make history-token activations easier to quantize. Hadamard mixing flattens raw activation peaks by spreading energy across channels. OSCAR goes further: the target covariance separates directions that matter more or less to attention, and the Hadamard transform then mixes each part into a more uniform range.

#### E.2 Full Table For Main Accuracy Run

Table 11 reports the five independent runs for the Qwen3-4B-Thinking main OSCAR result. Table 12– 14 report the five independent runs used to summarize Qwen3-8B OSCAR accuracy under three INT2 grouping configurations. Table 15–17 report the Qwen3-32B OSCAR runs. Table 18–20 report the GLM-4.7-FP8 detailed runs for Naive INT2, QuaRot/Hadamard rotation, and OSCAR.

Table 11: Full Qwen3-4B-Thinking results for the main OSCAR configuration (INT2 group size=128, sink=64, recent=256, cK = 0.96, cV = 0.92).

Task Run 1 Run 2 Run 3 Run 4 Run 5 Mean Std GPQA 66.16 63.64 64.14 64.65 66.16 64.95 1.16 HumanEval 93.17 91.59 92.44 90.85 93.17 92.24 1.02 LiveCodeBench v6 45.03 43.27 45.61 48.54 44.44 45.38 1.97 AIME25 63.33 63.33 63.33 60.00 70.00 64.00 3.65 MATH500 92.59 92.79 92.99 93.19 92.18 92.75 0.39

Table 12: Full Qwen3-8B results for INT2 group=128.

Task Run 1 Run 2 Run 3 Run 4 Run 5 Mean Std GPQA 53.03 54.04 54.55 57.58 55.05 54.85 1.70 HumanEval 86.59 88.54 88.90 87.56 87.68 87.85 0.91 LiveCodeBench v6 45.03 45.03 42.11 47.95 46.20 45.26 2.13 AIME25 66.67 63.33 56.67 63.33 63.33 62.67 3.65 MATH500 92.59 91.38 92.59 92.18 91.98 92.14 0.50

#### E.3 Ablation Detail Runs

- Table 21 reports the individual runs behind the rotation-decomposition study in Table 4.
- Table 22 reports the individual runs behind the sink/recent-window study in Table 5.
- Table 23 reports the per-task results behind the clip-threshold grid in Table 6.
- Table 24 reports the individual runs behind the calibration-data regime study in Table 7.

- Table 13: Full Qwen3-8B results for INT2 group=64.

Task Run 1 Run 2 Run 3 Run 4 Run 5 Mean Std GPQA 53.54 56.06 54.04 57.07 54.55 55.05 1.47 HumanEval 87.56 87.56 88.29 87.56 88.41 87.88 0.44 LiveCodeBench v6 47.37 44.44 49.12 45.03 45.61 46.32 1.91 AIME25 63.33 70.00 66.67 63.33 70.00 66.67 3.33

- MATH500 92.38 92.79 91.58 91.18 93.19 92.22 0.83

Table 14: Full Qwen3-8B results for INT2 group=32.

Task Run 1 Run 2 Run 3 Run 4 Run 5 Mean Std GPQA 52.02 55.56 59.09 52.53 56.57 55.15 2.93 HumanEval 87.93 87.32 88.05 88.78 88.66 88.15 0.59 LiveCodeBench v6 45.61 47.37 43.86 43.86 45.61 45.26 1.47 AIME25 66.67 53.33 63.33 66.67 60.00 62.00 5.58

- MATH500 91.78 92.38 92.59 92.18 92.38 92.26 0.30

- Table 15: Full Qwen3-32B results for OSCAR INT2 group=128. Task Run 1 Run 2 Run 3 Run 4 Run 5 Mean Std

- GPQA 59.60 59.09 57.58 60.10 – 59.09 1.09 HumanEval 90.37 89.76 89.27 89.76 89.27 89.68 0.45 LiveCodeBench v6 54.39 49.71 47.95 52.63 53.80 51.70 2.76 AIME25 63.33 66.67 76.67 76.67 70.00 70.67 5.96

MATH500 92.79 93.19 92.99 91.58 93.19 92.75 0.67

Table 16: Full Qwen3-32B results for OSCAR INT2 group=64.

Task Run 1 Run 2 Run 3 Run 4 Run 5 Mean Std

- GPQA 60.61 63.64 61.11 62.12 54.55 60.40 3.47 HumanEval 89.88 89.76 89.39 91.22 90.37 90.12 0.71 LiveCodeBench v6 53.80 52.63 53.80 54.39 53.22 53.57 0.67 AIME25 76.67 70.00 80.00 66.67 76.67 74.00 5.48

- Table 17: Full Qwen3-32B results for OSCAR INT2 group=32. Task Run 1 Run 2 Run 3 Run 4 Run 5 Mean Std

GPQA 63.64 57.07 56.57 60.10 59.60 59.40 2.83 HumanEval 92.20 90.85 89.76 90.73 90.24 90.76 0.91 LiveCodeBench v6 58.48 54.39 54.39 50.29 55.56 54.62 2.94 AIME25 66.67 70.00 70.00 66.67 63.33 67.33 2.79 MATH500 93.39 92.99 92.79 93.59 93.19 93.19 0.32

- Table 18: Full GLM-4.7-FP8 results for the Naive INT2 baseline. Task Run 1 Run 2 Run 3 Mean Std

- MATH500 93.19 92.18 93.19 92.38 92.79 92.75 0.46

GPQA 58.08 51.01 54.55 54.55 3.54 HumanEval 85.61 88.15 85.73 86.50 1.43 LiveCodeBench v6 33.33 44.44 33.33 37.03 6.41 AIME25 40.00 36.67 36.67 37.78 1.92 MATH500 87.40 85.80 86.60 86.60 0.80

Table 19: Full GLM-4.7-FP8 results for the QuaRot/Hadamard rotation baseline.

Task Run 1 Run 2 Run 3 Mean Std GPQA 70.20 63.64 70.20 68.01 3.79 HumanEval 91.71 89.76 88.15 89.87 1.78 LiveCodeBench v6 47.37 53.80 44.44 48.54 4.79 AIME25 80.00 80.00 76.67 78.89 1.92 MATH500 90.20 90.20 90.80 90.40 0.35

Table 20: Full GLM-4.7-FP8 results for the OSCAR rotation group size=128 configuration.

Task Run 1 Run 2 Run 3 Mean Std GPQA 73.23 74.24 73.23 73.57 0.58 HumanEval 90.85 91.34 90.98 91.06 0.25 LiveCodeBench v6 53.80 52.05 52.05 52.63 1.01 AIME25 76.67 80.00 80.00 78.89 1.92 MATH500 93.79 94.99 95.19 94.66 0.76

Table 21: Full individual runs for the rotation-decomposition study on Qwen3-8B.

Configuration Task Run 1 Run 2 Run 3 Mean Std

GPQA 56.06 54.55 57.07 55.89 1.27 HumanEval 87.93 87.80 88.17 87.97 0.19 LiveCodeBench v6 43.86 47.37 46.78 46.00 1.88 AIME25 66.67 70.00 66.67 67.78 1.92 MATH500 92.18 92.38 92.59 92.38 0.20

Full OSCAR

GPQA 37.37 39.90 38.89 38.72 1.27 HumanEval 35.49 36.22 34.88 35.53 0.67 LiveCodeBench v6 4.09 4.09 2.92 3.70 0.68 AIME25 13.33 16.67 10.00 13.33 3.33 MATH500 64.73 63.73 64.53 64.33 0.53

Tensor-recon. target

GPQA 53.54 54.04 51.01 52.86 1.62 HumanEval 87.20 85.73 87.20 86.71 0.84 LiveCodeBench v6 45.03 45.03 45.61 45.22 0.34 AIME25 73.33 60.00 56.67 63.33 8.82 MATH500 92.79 90.18 92.59 91.85 1.45

w/o Pbr

GPQA 50.00 53.54 50.51 51.35 1.91 HumanEval 84.15 83.41 83.78 83.78 0.37 LiveCodeBench v6 19.30 17.54 21.64 19.49 2.05 AIME25 23.33 20.00 20.00 21.11 1.92 MATH500 83.77 82.97 82.16 82.97 0.80

w/o HHad

GPQA 41.92 43.43 36.87 40.74 3.44 HumanEval 37.68 39.88 38.05 38.54 1.18 LiveCodeBench v6 4.09 4.09 5.26 4.48 0.68 AIME25 13.33 16.67 16.67 15.56 1.92 MATH500 65.33 63.33 65.73 64.80 1.29

w/o U

GPQA 18.69 21.72 15.15 18.52 3.29 HumanEval 0.00 0.00 0.00 0.00 0.00 LiveCodeBench v6 0.00 1.17 0.00 0.39 0.68 AIME25 3.33 0.00 3.33 2.22 1.92 MATH500 0.00 0.00 0.00 0.00 0.00

No rotation

###### Table 22: Full individual runs for the sink/recent-window study on Qwen3-4B-Thinking-2507.

(S, R) Task Run 1 Run 2 Run 3 Run 4 Run 5 Mean Std

GPQA 0.00 0.00 0.00 – – 0.00 0.00 HumanEval 0.00 0.00 0.00 – – 0.00 0.00 LiveCodeBench v6 0.00 0.00 0.00 – – 0.00 0.00 AIME25 0.00 0.00 0.00 – – 0.00 0.00 MATH500 0.00 0.00 0.00 – – 0.00 0.00

(0, 0)

GPQA 60.10 56.57 56.06 – – 57.58 2.20 HumanEval 91.95 92.07 91.71 – – 91.91 0.19 LiveCodeBench v6 40.35 43.27 38.60 – – 40.74 2.36 AIME25 56.67 60.00 50.00 – – 55.56 5.09 MATH500 93.39 92.99 91.58 – – 92.65 0.95

(32, 128)

GPQA 66.16 63.64 64.14 64.65 66.16 64.95 1.16 HumanEval 93.17 91.59 92.44 90.85 93.17 92.24 1.02 LiveCodeBench v6 45.03 43.27 45.61 48.54 44.44 45.38 1.97 AIME25 63.33 63.33 63.33 60.00 70.00 64.00 3.65

(64, 256)

- MATH500 92.59 92.79 92.99 93.19 92.18 92.75 0.39

(128, 512)

GPQA 65.15 66.16 64.65 – – 65.32 0.77 HumanEval 93.29 92.80 93.41 – – 93.17 0.32 LiveCodeBench v6 45.03 45.61 45.03 – – 45.22 0.33 AIME25 70.00 66.67 66.67 – – 67.78 1.92

- MATH500 93.39 93.19 93.39 – – 93.32 0.12

GPQA 66.16 65.15 63.64 – – 64.98 1.27 HumanEval 92.20 93.66 93.54 – – 93.13 0.81 LiveCodeBench v6 45.03 48.54 45.03 – – 46.20 2.03 AIME25 66.67 70.00 66.67 – – 67.78 1.92 MATH500 93.19 93.39 93.39 – – 93.32 0.12

(256, 1024)

###### Table 23: Full per-task results for the clip-threshold sweep on Qwen3-4B-Thinking-2507. Each row is one (cK,cV ) setting with one run per task.

cK cV GPQA HumanE LCB v6 AIME25 MATH500 Mean 0.88 0.88 59.09 91.83 39.18 53.33 92.59 67.20 0.88 0.92 62.12 91.22 39.77 56.67 91.58 68.27 0.88 0.96 57.07 91.10 42.69 56.67 92.99 68.10

- 0.88 0.98 62.12 91.95 37.43 53.33 92.99 67.56
- 0.88 1.00 57.58 91.59 46.78 53.33 91.98 68.25

0.92 0.88 61.62 93.17 41.52 56.67 92.79 69.15 0.92 0.92 62.12 91.22 43.86 60.00 92.79 70.00 0.92 0.96 64.14 92.32 46.20 56.67 92.99 70.46

- 0.92 0.98 65.15 93.90 48.54 53.33 92.79 70.74
- 0.92 1.00 64.65 93.90 46.78 53.33 92.99 70.33 0.96 0.88 61.62 92.80 39.77 56.67 92.99 68.77

###### 0.96 0.92 64.14 93.54 46.20 56.67 92.38 70.59

- 0.96 0.96 63.13 91.59 43.27 63.33 92.18 70.70

- 0.96 0.98 62.12 92.68 40.35 53.33 92.38 68.17
- 0.96 1.00 62.12 91.59 42.11 56.67 93.19 69.14

- 0.98 0.88 55.56 91.71 37.43 50.00 91.38 65.22

- 0.98 0.92 55.05 91.59 36.84 53.33 91.98 65.76

- 0.98 0.96 48.48 90.37 36.84 53.33 91.58 64.12

- 0.98 0.98 55.56 91.71 37.43 46.67 92.59 64.79
- 0.98 1.00 56.57 91.34 39.77 56.67 92.18 67.31

- 1.00 0.88 48.48 89.88 30.99 40.00 92.38 60.35

- 1.00 0.92 44.95 90.98 32.16 36.67 91.18 59.19

- 1.00 0.96 45.45 90.37 30.41 33.33 91.18 58.15

- 1.00 0.98 43.43 90.73 32.16 36.67 90.18 58.63

###### 1.00 1.00 47.47 90.73 30.41 23.33 88.18 56.02

- Table 24: Full individual runs for the calibration-data regime study on Qwen3-4B-Thinking-2507.

Calibration setting Task Run 1 Run 2 Run 3 Mean Std

GPQA 61.62 58.08 62.12 60.61 2.20 HumanEval 92.68 92.44 92.44 92.52 0.14 LiveCodeBench v6 37.43 39.18 42.11 39.57 2.36 AIME25 50.00 56.67 60.00 55.56 5.09 MATH500 93.39 93.39 92.59 93.12 0.46

MMLU 2k

GPQA 62.63 58.59 60.10 60.44 2.04 HumanEval 91.22 92.68 92.32 92.07 0.76 LiveCodeBench v6 40.94 41.52 39.77 40.74 0.89 AIME25 66.67 56.67 56.67 60.00 5.77 MATH500 92.18 93.19 92.18 92.52 0.58

MMLU 8k

GPQA 63.64 64.65 64.14 64.14 0.51 HumanEval 93.29 92.32 93.17 92.93 0.53 LiveCodeBench v6 46.78 42.69 42.11 43.86 2.55 AIME25 73.33 53.33 56.67 61.11 10.72 MATH500 92.18 92.99 91.98 92.38 0.53

MMLU 16k

GPQA 62.63 61.62 59.60 61.28 1.54 HumanEval 92.07 92.56 91.95 92.20 0.32 LiveCodeBench v6 48.54 45.61 43.86 46.00 2.36 AIME25 53.33 63.33 56.67 57.78 5.09

MMLU 32k

- MATH500 92.59 91.98 92.18 92.25 0.31

WikiText 8k

GPQA 61.11 60.61 61.62 61.11 0.51 HumanEval 93.41 92.20 92.56 92.72 0.63 LiveCodeBench v6 39.77 43.86 42.69 42.11 2.11 AIME25 56.67 56.67 63.33 58.89 3.85

- MATH500 93.19 92.59 92.18 92.65 0.50

GPQA 63.13 64.14 61.62 62.96 1.27 HumanEval 92.56 92.56 92.56 92.56 0.00 LiveCodeBench v6 46.20 44.44 43.27 44.64 1.47 AIME25 63.33 63.33 60.00 62.22 1.92 MATH500 92.38 92.59 92.99 92.65 0.31

GPQA-Diamond 8k

