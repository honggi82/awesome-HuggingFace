# arXiv:2602.08426v2[cs.CL]25May2026

OpenMOSS

[Figure 1]

[Figure 2]

## Prism: Spectral-Aware Block-Sparse Attention

Xinghao Wang1,4 Pengyu Wang1,4 Xiaoran Liu1,2,4 Fangxu Liu3 Jason Chu3 Kai Song3 Xipeng Qiu1,2,4,† 1Fudan University 2Shanghai Innovation Institute 3ByteDance Inc. 4OpenMOSS Team

Abstract

Block-sparse attention is promising for accelerating long-context LLM pre-filling, yet identifying relevant blocks efficiently remains a bottleneck. Existing methods typically employ coarse-grained attention as a proxy for block importance estimation, but often resort to expensive token-level searching or scoring, resulting in significant selection overhead. In this work, we trace the inaccuracy of standard coarse-grained attention via mean pooling to a theoretical root cause: the interaction between mean pooling and Rotary Positional Embeddings (RoPE). We prove that mean pooling acts as a low-pass filter that induces destructive interference in high-frequency dimensions, effectively creating a "blind spot" for local positional information (e.g., slash patterns). To address this, we introduce Prism, a training-free spectral-aware approach that decomposes block selection into high-frequency and lowfrequency branches. By applying energy-based temperature calibration, Prism restores the attenuated positional signals directly from pooled representations, enabling block importance estimation using purely block-level operations, thereby improving efficiency. Extensive evaluations confirm that Prism maintains accuracy parity with full attention while delivering up to 5.1× speedup.

Repository: https://github.com/xinghaow99/prism Correspondence: xinghaowang22@m.fudan.edu.cn,xpqiu@fudan.edu.cn

### 1 Introduction

The capacity to process extensive contexts is a defining characteristic of modern Large Language Models (LLMs), unlocking applications ranging from repository-level code understanding to hour-long video understanding [1, 2]. However, handling such long contexts is non-trivial, as the self-attention mechanism scales quadratically with sequence length [3], resulting in massive computational intensity during the tokenparallel pre-filling phase and bottlenecking practical deployment. To mitigate this, block-sparse attention has emerged as a promising solution, approximating full attention by computing only a subset of relevant blocks. The efficacy of this approach hinges on block importance estimation: efficiently identifying relevant blocks without full computation. Standard training-free methods typically employ mean pooling [4, 5] as a coarse-grained proxy. However, this proxy is often inaccurate, forcing state-of-the-art methods to rely on expensive heuristic search and token-level verification to maintain performance. This creates a fundamental trade-off: the heavy estimation overhead often negates the sparsity gains, causing these methods to underperform highly optimized full attention implementations (e.g., FlashAttention [6]) at moderate sequence lengths.

[Figure 3]

[Figure 4]

[Figure 5]

Figure 1 Spectral Disentanglement of Attention Patterns. We visualize the attention score matrices computed using different spectral bands of RoPE. (Left) Low-Frequency Band: Captures global semantic dependencies (e.g., block-sparse patterns / vertical lines), acting as the semantic backbone. (Middle) High-Frequency Band: Strictly encodes fine-grained relative locality (e.g., slash lines), which is critical for local coherence. (Right) Full Spectrum: The superposition of both patterns.

In this work, we trace the inaccuracy of standard coarse-grained attention to a theoretical root cause: the spectral interaction between mean pooling and Rotary Positional Embeddings (RoPE) [7]. As illustrated in

- Figure 1, the spectral heterogeneity of RoPE naturally disentangles attention into distinct structural patterns: high-frequency dimensions strictly encode fine-grained relative positions, while low-frequency dimensions capture global semantic dependencies, manifesting as divergent sparse patterns. However, we mathematically prove that mean pooling acts as a Low-Pass Filter. In high-frequency dimensions, the rapid rotation of RoPE vectors induces destructive interference during aggregation, causing the signal magnitude to collapse. This phenomenon creates a spectral “Blind Spot” that effectively erases fine-grained positional information (e.g., slash patterns) from the pooled representation, explaining why standard methods struggle to maintain local coherence without expensive corrections.

Toaddressthis, weintroducePrism, a spectral-awareframeworkthat disentanglesblock importance estimation into two parallel branches. Instead of treating embeddings as monolithic vectors, Prism explicitly separates the attenuated high-frequency band from the robust low-frequency band. By applying a novel energy-based temperature calibration, Prism restores the attenuated positional signals from pooled representations. This design enables Prism to perform precise importance estimation using exclusively block-level operations, eliminating the selection bottleneck common in prior works.

We evaluate Prism with diverse long-context capabilities, ranging from language modeling (PG19 [8]), long-context understanding (LongBench [1]), long-context retrieval (RULER [9]), and video understanding (VideoMME [10] & LongVideoBench [2]). Experiments demonstrate that Prism closely matches the accuracy of full attention while delivering substantial speedups compared to FlashAttention and state-of-the-art sparse attention methods. Our contributions are summarized as follows:

- • Theoretical Insight: We identify mean pooling as a low-pass filter under RoPE, revealing the “Blind Spot” responsible for the failure of standard block importance estimation.
- • Methodology: We propose Prism, a training-free framework utilizing dual-band scoring and energybased calibration to explicitly preserve high-frequency positional information without token-level overhead.
- • SOTA Efficiency: Prism achieves state-of-the-art accuracy-speedup trade-offs, delivering up to 5.1× speedup at 128K tokens while outperforming baselines in latency across all sequence lengths.

### 2 Related Work

Block-Sparse Attention The quadratic computational complexity of the self-attention mechanism [3] poses a significant bottleneck for processing long contexts in modern LLMs. Fortunately, as a result of the softmax operation, learned attention matrices often exhibit highly sparse patterns; that is, a small subset of tokens

accounts for the majority of the attention mass, providing an opportunity to reduce computational overhead. Early sparse attention approaches relied on static sparse patterns, such as fixed sliding windows [11], dilated windows [12], or global "sink" tokens [13] to maintain local coherence and stability. However, static patterns often fail to capture long-range dependencies scattered arbitrarily across the sequence (the "needle in a haystack" problem). Consequently, recent research has shifted toward dynamic sparse attention, where the attention pattern is determined adaptively based on the input. To implement this efficiently on hardware, block-sparse approaches partition the sequence into fixed-size blocks (e.g., 128×128). This design naturally aligns with the tiling mechanism of FlashAttention [6], which decomposes computation into contiguous blocks for I/O awareness. By restricting the dense computation and online accumulation to a selected subset of block pairs, this granularity allows for optimized GPU kernels (e.g., via Triton or CUDA) while significantly reducing the number of FLOPs during the compute-bound pre-filling stage.

Block Importance Estimation The central challenge in dynamic block-sparse attention is block importance estimation: identifying which Key blocks are relevant to a given Query block without incurring the quadratic cost of the full attention matrix. In the scope of pre-filling, existing training-free approaches typically rely on coarse-grained proxies combined with heuristic pattern matching. Methods such as MInference [4] and FlexPrefill [5] employ offline or online search strategies to classify attention heads into pre-defined categories (e.g., “Vertical Slash” or “Block-Sparse”). Consequently, they adopt divergent estimation techniques, utilizing coarse-level attention for semantic retrieval heads while falling back to selection against certain patterns. Other works aim for a unified estimation metric. SpargeAttention [14] adopts coarse-level attention for all heads while enforcing blocks with low intra-block similarity. XAttention [15] introduces an antidiagonal scoring mechanism to capture both block-sparse and vertical-slash patterns, while PBS-Attn [16] utilizes token permutation to cluster critical tokens for better separability. However, these methods typically involve additional token-level operations, which significantly degrade block selection efficiency, particularly at moderate sequence lengths where the selection overhead outweighs the sparsity gains.

### 3 Method

#### 3.1 Preliminaries

Coarse-grained Attention Block-sparse attention requires a block mask M to determine if a block pair (u,v) should be computed. For efficient estimation of M, a typical approach is to compute a coarse-grained attention matrix S¯. Formally, let Q,K,V ∈ RL×d denote the query, key, and value matrices, where L is the sequence length and d is the head dimension. The sequence is partitioned into N = ⌈BL⌉ blocks, where B is the block size. For the u-th query block and v-th key block, let Iu and Iv denote the sets of token indices belonging to each block, respectively. Coarse-grained attention typically compresses each block into a single representative vector using mean pooling:

1 B i∈I

1 B j∈I

qi, k¯v =

q¯u =

kj (1)

u

v

Let Q¯ ,K¯ ∈ RN×d be the matrices formed by stacking these pooled vectors. Then the coarse-grained attention matrix is computed as:

##### Q ¯ K¯ ⊤

S¯ = softmax

(2)

√

d

Finally, a top-k or top-p selection is applied to S¯ to generate the binary mask M ∈ {0,1}N×N.

Spectral Structure of RoPE Modern large language models (LLMs) [17–20] typically employ rotary positional embeddings (RoPE) [7] to inject positional information. RoPE rotates feature pairs in the complex plane. Let x(nj) denote the j-th feature pair of a vector at position n, represented as a complex number. The embedding

is rotated by an angle dependent on the position n and a frequency θj:

xn(j) = xnope(j) · einθ

j (3)

Crucially, the rotation frequencies are defined as a geometric sequence decaying across the feature dimension index j ∈ {0,...,d/2 − 1}:

θj = b−2j/d (4)

where b is the base (e.g. 1M for Qwen3). This definition creates a Spectral Heterogeneity [21] across the embedding dimensions:

- • High-Frequency Band (j → 0): Dimensions with low indices possess large θj, resulting in rapid rotation. These dimensions encode fine-grained, relative positional information (e.g., local context).
- • Low-Frequency Band (j → d/2): Dimensions with high indices possess θj → 0, resulting in negligible rotation over long distances. These dimensions behave similarly to absolute embeddings, primarily encoding global semantic content.

This spectral distribution implies that linear operations applied across the sequence dimension, such as the mean pooling defined in Eq. 1, will exhibit frequency-dependent behaviors, a phenomenon we analyze in the following section.

Sparse Patterns of Attention Extensive empirical analysis [4, 5, 13] reveals that attention matrices in pretrained LLMs are not uniformly sparse but exhibit distinct structural characteristics, most notably the vertical slash patterns and block-sparse patterns. Prior works typically treat these patterns as mutually exclusive properties of specific attention heads, employing heuristic classifiers to assign distinct estimation strategies [4, 5]. Although Xu et al. [15] attempted to capture both patterns via a unified antidiagonal scoring mechanism, their approach still incurs additional token-level operations, resulting in significant selection overhead at long sequence lengths. We challenge this head-level dichotomy. We posit that these patterns are not spatially separated across heads but are instead spectrally disentangled within individual heads.

As visualized in Figure 1, the high-frequency spectral bands of RoPE (low indices) strictly encode relative locality (slash patterns), while the low-frequency bands (high indices) capture global semantic dependencies (block-sparse patterns). This spectral observation motivates our frequency-decomposed approach.

#### 3.2 Mean Pooling as a Low-Pass Filter

To facilitate efficient block importance estimation, mean pooling(Eq. 1) serves as a common technique to compress a block into a single representative vector. In this section, we theoretically analyze the impact of mean pooling with the consideration of RoPE, which explains why existing methods had to resort to token-level operations for accurate block importance estimation.

Geometric Summation of Mean Pooling Consider the j-th frequency pair of the query vector. Under RoPE, the embedding at position n can be decomposed into a content component c(j) and a positional rotation einθ

j. Assuming the semantic content c(j) remains relatively stable within the local context of a block (a standard assumption for adopting mean pooling), applying the mean pooling over a block of size B starting at position n0 can be formulated as a geometric series summation:

q¯(j) ≈

B−1

c(j) B

c(j)ein

0θj

ei(n

0+k)θj =

B

k=0

B−1

eikθ

j

k=0

Geometric Sum

(5)

Spectral Attenuation The magnitude of this pooled vector dictates the signal strength available for dot-product retrieval. By evaluating the geometric sum, we derive the Spectral Attenuation Factor λj(B), defined as the

1.0

0.8

###### AttenuationFactor

0.6

0.4

Qwen (Base 1M)

0.2

Llama (Base 500k)

Dead Zone (Signal 0)

Transition Zone (Weak Signal)

Semantic Zone (Signal 1)

0.0

0 20 40 60 80 100 120

Feature Dimension Index (0=High Freq, 128=Low Freq)

Figure 2 Spectral attenuation factor λj(B) with block size B = 128 and head dimension d = 128.

ratio of the pooled vector’s magnitude to the original vector’s magnitude:

B−1

λj(B) ≜ |q¯j| |c|

1 B

1 B

sin(Bθj/2) sin(θj/2)

=

j =

einθ

n=0

For small frequencies, this function converges to the normalized sinc function:

(6)

λj(B) ≈ sinc

Bθj 2π

(7)

A detailed derivation is provided in Appendix A. This derivation mathematically reveals that mean pooling functions as a Low-Pass Filter:

- • Destructive Interference (λj → 0): In the high-frequency band where the block size covers full rotation periods (Bθj ≈ 2πk), the vectors sum to near-zero. For a standard block size B = 128, this creates a “Blind Spot” in the first ≈ 30 dimensions (for Base 1M), effectively erasing local positional structures.
- • Constructive Interference (λj → 1): In the low-frequency band where θj → 0, the rotations are negligible, and the signal magnitude is fully preserved.

We quantify this effect using a standard setting with block size B = 128 and head dimension d = 128, considering RoPE bases b = 106 (Qwen3) and b = 5 × 105 (LLaMa 3.1), as visualized in Figure 2. Taking Qwen3 as an example, destructive interference reaches its peak (λj ≈ 0) when the total rotation Bθj = 2π. We solve for the corresponding feature dimension index 2j:

ln(B/2π) ln b

B · b−2j/d = 2π =⇒ 2j = d ·

(8)

Substituting the values yields a cutoff dimension of 2j ≈ 28. Based on this derivation, the spectrum in

- Figure 2 divides into three distinct regimes:

- • The Dead Zone (0 ≤ 2j ≲ 30): The signal magnitude is effectively zero due to full phase cancellation.
- • The Transition Zone (30 ≲ 2j ≲ 60): The signal begins to recover but remains heavily attenuated (λ < 1).
- • The Semantic Zone (2j > 60): The signal magnitude is fully preserved, capturing global semantic information.

2.5

2.0

RMSNorm

1.5

| | |
|---|---|

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

| |
|---|

| | | |
|---|---|---|
| | | |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |

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

1.0

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

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |

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

0.5

0.0

0 10 20 30 Layer Index

0 10 20 30 Layer Index

Semantic Zone Transition Zone Dead Zone Full

- Figure 3 Comparison of Query RMS norms before and after pooling. Left (Token-level): While the Semantic Zone (blue) holds the highest energy, the Dead Zone (green) maintains a robust magnitude (RMS ≈ 1.0), confirming that high-frequency dimensions are actively utilized by the pre-trained model. Right (Block-pooled): After pooling, energy in the Dead Zone collapses to near-zero due to destructive interference, while the Semantic Zone preserves its magnitude.

This analysis theoretically justifies why standard coarse-grained attention is “blind” to fine-grained positional structures encoded in the high-frequency band.

#### 3.3 Energy Analysis

To verify whether the theoretical attenuation derived in Section 3.2 manifests in actual model representations, we analyze the spectral energy distribution using Qwen3-8B. We measure the RMS norms of the query vectors before and after mean pooling across the three spectral zones defined in Figure 2. Ideally, if pooling were lossless, the block-level RMS should mirror the token-level RMS. However, Figure 3 reveals a distinct Spectral Divergence: At the token level (Left), the Dead Zone maintains robust magnitude (RMS ≈ 1.0), confirming that high-frequency positional features are intrinsically significant to the pre-trained model. In contrast, the block-pooled representation (Right) exhibits a dramatic Energy Collapse in the Dead Zone (RMS ≈ 0.1), empirically validating that mean pooling acts as a low-pass filter that suppresses local positional information. Crucially, the RMS of the Semantic Zone consistently surpasses the Full spectrum. This intrinsic divergence is significantly exacerbated post-pooling, as the Full vector is further diluted by the “dead weight” of attenuated high-frequency dimensions. This widened energy gap necessitates the frequency-dependent calibration proposed next.

#### 3.4 Prism: Spectral-Aware Block Selection

To resolve the spectral bias identified above, we propose Prism, a framework that decomposes block selection into two spectral branches based on their characteristics. The overall procedure is summarized in Figure 4 and consists of two core components: (1) Dual-Band Block Importance Estimation, which explicitly isolates the high-frequency and low-frequency bands to avoid signal interference during aggregation; and (2) Energy-Based Temperature Calibration, which derives branch-specific temperatures from spectral energy distributions, restores the logit magnitudes without any hyperparameter tuning. Crucially, this design enables Prism to perform estimation using exclusively block-level operations, minimizing selection overhead.

Dual-Band Block Importance Estimation To best preserve information from both spectral bands, we propose a dual-band block importance estimation strategy that avoids interference between the two bands. Let Q,K ∈ RL×d denote the input query and key matrices. We explicitly isolate the High-Frequency Band by slicing the first dhigh dimensions, yielding Qhigh,Khigh ∈ RL×d

high. Similarly, we slice the last dlow dimensions to form the Low-Frequency Band, Qlow,Klow ∈ RL×d

low. Subsequently, mean pooling with block size B is applied to the high-frequency and low-frequency bands independently, obtaining

Q¯ high,K¯ high ∈ RN×d

high and Q¯ low,K¯ low ∈ RN×d

low, where N = ⌈BL⌉. With the pooled representations, we compute the coarse-grained importance scores for each spectral band z ∈ {high,low}. Furthermore, to account for the distinct spectral energy densities caused by attenuation (as observed in Figure 3), we introduce branch-specific temperature scaling factors τhigh and τlow:

S¯z = softmax

Q ¯ zK¯ ⊤z τz√dz

, for z ∈ {high,low} (9)

Based on the probability distributions S¯high and S¯low, we generate binary block masks Mhigh and Mlow by selecting the top-p cumulative probability mass for each query block. The final block-sparse mask M is obtained by the union of these branch-specific selections:

M = Mhigh ∪ Mlow (10)

Energy-Based Temperature Calibration To align the logit magnitude of the individual spectral bands to the scale of the full spectrum, we derive the branch-specific temperatures τz based on the spectral energy distribution. We employ RMS norm to represent the spectral energy density of a pooled matrix X¯ ∈ RN×d, where

RMS(X¯ ) = N1 Nu=1 ∥x¯u∥2

d . Consider attention logits Lfull = (Q¯ fullK¯ ⊤full)/

√

d. Since the dot product accumulates magnitude across d dimensions, the scale of these logits follows:

√

d · RMS(Q¯ full)RMS(K¯ full) (11)

|Lfull| ∝

Similarly, for a spectral branch z using subspace dimension dz, the uncalibrated logits Lz scale as:

|Lz| ∝ dz · RMS(Q¯ z)RMS(K¯ z) (12)

To restore the signal strength of the partial branch to the baseline level (i.e., |Lz|/τz ≈ |Lfull|), we derive the calibration factor:

RMS(Q¯ z) RMS(Q¯ full) ·

RMS(K¯ z) RMS(K¯ full)

dz d ·

(13)

τz ≈

### 4 Experiments

def prism(Q, K, d_h, d_l, B, p): # Setup dimensions bs, h, L, d = Q.shape N = L // B

- # 1. Pooling & Slicing Qb, Kb = pool(Q, B), pool(K, B) Qh, Ql = Qb[..., :d_h], Qb[..., -d_l:] Kh, Kl = Kb[..., :d_h], Kb[..., -d_l:]
- # 2. RMS Calculation rq, rk = rms(Qb), rms(Kb) rq_h, rk_h = rms(Qh), rms(Kh) rq_l, rk_l = rms(Ql), rms(Kl)
- # 3. Calibration (Eq. 13) th = sqrt(d_h/d) * (rq_h/rq) * (rk_h/rk) tl = sqrt(d_l/d) * (rq_l/rq) * (rk_l/rk)
- # 4. Dual-Band Scoring scale_h = sqrt(d_h) * th scale_l = sqrt(d_l) * tl logits = empty(bs, h, 2N, N) logits[..., :N, :]=(Qh @ Kh.T) / scale_h logits[..., N:, :]=(Ql @ Kl.T) / scale_l
- # 5. Selection P = softmax(logits, dim=-1) Mh, Ml = top_p(P, p).split(N, dim=-2) return Mh | Ml

Figure 4 PyTorch-style implementation of Prism. Prism exclusively uses block-level operations for best efficiency. See Appendix C for top_p implementation.

#### 4.1 Setup

Benchmarks, Models & Baselines To evaluate the versatility and robustness of Prism, we conduct experiments across five categories of long-context tasks: (1) Language Modeling using PG19 [8]; (2) Long-Context Understanding using LongBench [1]; (3) Long-Context Retrieval using RULER [9]; (4) Video Understanding using VideoMME [10] and LongVideoBench [2]; and (5) Video Generation using HunyuanVideo [22] evaluated with VBench prompts [23]. We employ state-of-the-art models including Llama-3.1-8B-Instruct (128K) [17] and the Qwen3-8B [18]. Notably, for Qwen3-8B, we apply YaRN [24] extrapolation to extend the context from 32K to 128K. For multimodal tasks, we utilize Qwen3-VL-8B [25]. This selection specifically enables us to verify Prism’s generalization to RoPE variants, including YaRN, Interleaved M-RoPE, and

3D-RoPE. We compare Prism with FlashAttention-2 [26] (full attention baseline), and state-of-the-art trainingfree dynamic block-sparse methods: MInference [4], FlexPrefill [5], XAttention [15], and PBS-Attn [16]. To ensure fair comparison, we use the official recommended configurations for all baselines. Details in Appendix D.

Implementation Details For Prism, we use a block size B = 128 based on the trade-off analysis in Appendix F. Guided by the spectral analysis in Figure 2, we configure the spectral bands as dhigh = 64 and dlow = 96. This configuration ensures robust signal coverage by overlapping the transition zone, while strictly aligning dimension sizes with multiples of 32 to maximize Tensor Core throughput on GPUs. For Top-P selection, we use a threshold p = 0.95 for Llama-3.1-8B-Instruct and p = 0.93 for Qwen models to balance the trade-off between efficiency and accuracy. For importance estimation and block-sparse attention, we implement custom Triton kernels for best efficiency.

Table 1 Performance comparison on LongBench.

Method Single-Doc QA Multi-Doc QA Summarization Few-shot Learning Code Synthetic Avg. Llama-3.1-8B Full 47.51 43.28 25.9 45.92 18.01 68.18 41.47 MInference 47.42 42.54 25.85 45.58 17.84 67.6 41.14 FlexPrefill 46.13 41.49 25.85 46.63 17.68 25.61 33.90 XAttention 45.89 41.56 26.18 45.86 19.24 59.32 39.68 PBS-Attn 46.53 41.97 25.88 45.92 20.23 65.08 40.94 Prism 47.09 42.13 26 46.4 18.72 66.15 41.08 Qwen-3-8B Full 47.1 40.45 24.07 56.69 1.65 67 39.49 MInference 46.9 40.39 24.07 55.74 1.61 66.33 39.18 FlexPrefill 43.77 39.31 23.99 57.33 1.87 50.5 36.13 XAttention 44.49 40.09 24.12 57.27 1.29 65.67 38.82 PBS-Attn 44.83 40.02 24.04 56.74 2.58 65.83 39.01 Prism 46.47 40.08 24.01 58.36 1.64 64.17 39.12

#### 4.2 Main Results

Minference

1

FlexPrefill XAttention PBS-Attn

- 0

- 1

- 2

- 3

- 4

- 5

[Figure 6]

| |
|---|

| |
|---|

Prism

- 0

- 1

- 2

- 3

- 4

PPLvsFlashAttention

| |
|---|
| |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Speedup(x)

| |
|---|

| |
|---|
| |

| |
|---|

| |
|---|

8K 16K 32K 64K 128K Sequence Length

- Figure 5 Language modeling performance on PG19. We compare the Perplexity Degradation (∆PPL, solid lines, left axis) and Speedup (bars, right axis) across sequence lengths. Prism achieves a double win: it shows no perplexity degradation (sticking to the ∆ ≈ 0 line) while delivering the highest speedup (5.1× at 128K), significantly outperforming baselines that trade off accuracy for speed or suffer from high selection overhead.

Language Modeling We evaluate the modeling capability on long-context sequences using the PG19 benchmark. Figure 5 visualizes the scalability of Prism compared to baselines, plotting Perplexity Degradation (∆PPL) and Speedup. Notably, Prism demonstrates superior robustness, maintaining a perplexity virtually

identical to the Full Attention baseline (∆PPL ≈ 0) across all context lengths. In contrast, baselines like MInference and FlexPrefill suffer from significant perplexity degradation as sequence length increases, especially at 128K. While XAttention achieves high fidelity comparable to Prism, it is bottlenecked by significant estimation overhead. This becomes critical at extreme lengths: at 128K, XAttention is limited to a

- 3.0× speedup, whereas Prism achieves 5.1×. Consequently, Prism achieves a double win, delivering the highest speedup while simultaneously maintaining the perplexity of full attention.

Long-Context Understanding Table 1 presents the evaluation results on LongBench. Prism demonstrates exceptional robustness, achieving average scores of 41.08 on Llama-3.1-8B-Instruct and 39.12 on Qwen-3-8B, showing negligible degradation (< 0.4%) compared to the full attention baseline. While MInference achieves similar accuracy, it relies on a fixed budget strategy that, at the moderate sequence lengths of LongBench (< 16K), often results in selecting nearly all tokens. Consequently, it degenerates to full attention while incurring additional estimation overhead, failing to provide meaningful sparsity. In contrast to other sparse baselines, Prism significantly outperforms FlexPrefill and XAttention on average for both models. Notably, Prism even slightly outperforms full attention on specific tasks (e.g., 58.36 vs. 56.69 on Qwen-3 Few-shot). We attribute this gain to the explicit preservation of high-frequency positional signals. By recovering the fine-grained relative structure essential for Induction Heads [27], Prism enhances the model’s ability to perform in-context pattern copying. Furthermore, unlike full attention, Prism filters out irrelevant semantic blocks, effectively denoising the context for these position-sensitive heads.

Table 2 Performance comparison on RULER.

Method 4K 8K 16K 32K 64K 128K Avg. Llama-3.1-8B Full 95.42 94.38 93.38 87.98 84.72 77.77 88.94 MInference 95.43 94.46 93.42 87.22 83.07 71.04 87.44 FlexPrefill 93.8 92.44 93.28 87.92 84.74 72.41 87.43 XAttention 95.17 94.3 93.28 89.06 82.31 70.52 87.44 PBS-Attn 95.45 94.01 92.54 85.77 83.03 71.69 87.08 Prism 95.28 94.47 92.48 87.67 82.59 72.75 87.54 Qwen-3-8B(YaRN) Full 95.01 92.35 90.04 87.24 79.93 75.09 86.61 MInference 95.08 92.37 89.67 86.01 76.53 70.36 85.00 FlexPrefill 90.89 87.61 87.82 85.58 78.27 73.42 83.93 XAttention 94.55 91.03 87.91 84.37 77.73 72.01 84.60 PBS-Attn 94.83 92.1 88.18 85.97 78.41 72.03 85.25 Prism 94.84 90.95 87.69 86.88 78.58 72.65 85.27

Long-Context Retrieval Table 2 reports the evaluation results on RULER. As shown in the table, all methods show comparable performance with their configured threshold parameters. However, it is crucial to note that Prism achieves this parity using exclusively block-level operations in semantic retrieval. In contrast, baselines like MInference and FlexPrefill rely on token-level estimation using the last query block, a heuristic that is inherently advantageous for RULER’s format, where the query is typically positioned at the end. Despite not being explicitly optimized for such structure, Prism’s Low-Frequency Branch successfully handles these retrieval tasks, validating that our spectral calibration preserves sufficient semantic recall. Notably, the robust results on the YaRN-extrapolated Qwen3-8B demonstrate Prism’s generalizability to RoPE variants without requiring additional adaptations; we provide the corresponding zone-wise analysis in Appendix B.

VideoUnderstandingToassessthegeneralizabilityofPrismtomultimodalscenarios, weevaluateperformance on VideoMME and LongVideoBench using Qwen3-VL-8B. As shown in Table 3, Prism outperforms existing approaches on both benchmarks, achieving performance comparable to the full attention baseline. Crucially, in the Long split of VideoMME, where video durations range from 30 minutes to 1 hour (spanning 54K to 107K tokens), Prism surpasses the full attention baseline (64.00 vs. 63.11). We attribute this to the denoising effect of sparse attention, which effectively filters out irrelevant visual tokens, allowing the model to focus on the most salient visual information. These results also confirm the generalization of Prism to other multimodal

###### Table 3 Performance comparison on long video understanding tasks with Qwen3-VL-8B.

VideoMME LVB

Method

Short Med. Long Overall Overall

Full 79.89 70.67 63.11 71.22 65.00 MInference 79.44 70.00 62.44 70.63 61.48 FlexPrefill 77.67 70.67 62.67 70.34 64.10 XAttention 79.22 69.78 63.44 70.81 64.25 PBS-Attn 79.56 69.56 62.89 70.67 64.17 Prism 79.00 70.67 64.00 71.22 64.25

RoPE variants (i.e., Interleaved M-RoPE [25]), demonstrating its robustness.

Table 4 Performance comparison on video generation.

Method Threshold PSNR↑ SSIM↑ LPIPS↓ Speedup↑ XAttention

0.900.95 21.423.3 0.7250.797 0.2280.171 1.541.33××

0.900.930.950.97 20.721.622.423.5 0.7130.7480.7750.809 0.2590.2240.1980.165 1.761.601.501.37××××

Prism

Video Generation We further evaluate Prism on dense video generation with HunyuanVideo, whose 3D-RoPE introduces rotations along temporal and spatial axes. We select Prism’s spectral dimensions axis-wise according to the same attenuation criterion used for 1D RoPE (Eq. 8); the detailed configuration is provided in Appendix D.2. Table 4 reports fidelity to the full-attention output and end-to-end speedup under different sparsity thresholds. At comparable quality, Prism consistently achieves better efficiency than XAttention. For example, Prism with threshold 0.93 improves PSNR/SSIM/LPIPS over XAttention with threshold 0.90 (21.6/0.748/0.224 vs. 21.4/0.725/0.228) while increasing speedup from 1.54× to 1.60×. In the higher-quality regime, Prism with threshold 0.97 also slightly improves fidelity over XAttention with threshold 0.95 while yielding a higher speedup (1.37× vs. 1.33×). Qualitative results are provided in Appendix E. These results demonstrate that Prism’s spectral-aware block selection extends beyond autoregressive pre-filling and long-video understanding to dense prediction workloads.

#### 4.3 Efficiency Results

Minference FlexPrefill

- 0

- 1

- 2

- 3

- 4

- 5

| |
|---|

XAttention PBS-Attn

400

| |
|---|

Prism

| |
|---|

PrefillLatency(ms)

300

Speedup(x)

200

| |
|---|

100

[Figure 7]

| |
|---|

0

8K 16K 32K 64K 128K Sequence Length

- Figure 6 Efficiency comparison on Llama-3.1-8B-Instruct with an H100 GPU. We report pre-filling latency (bars, left axis) and speedup relative to FlashAttention-2 (lines, right axis). Shaded areas represent the block importance estimation time.

100

Minference FlexPrefill XAttention PBS-Attn

EstimationTime(ms)

| |
|---|

80

| |
|---|

| |
|---|

60

Prism

40

20

0

[Figure 8]

3500

MemoryOverhead(MB)

3000

2500

2000

1500

1000

500

[Figure 9]

0

8K 16K 32K 64K 128K Sequence Length

Figure 7 Estimation overhead comparison. The upper and lower panels illustrate the time and memory overhead of block importance estimation, respectively.

Latency Comparison We evaluate the attention pre-filling latency and speedup of Prism compared to FlashAttention-2 and state-of-the-art sparse attention methods. Figure 6 illustrates the results across sequence lengths from 8K to 128K. Notably, Prism achieves consistent speedups across all sequence lengths. In contrast, baselines such as MInference and FlexPrefill only begin to outperform FlashAttention at 64K and 32K, respectively, as their significant estimation overhead outweighs the sparsity gains at shorter lengths. While XAttention exhibits comparable speedups at moderate lengths, it suffers from diminishing returns at extreme lengths (e.g., 128K) due to increasing selection costs. Prism, however, preserves a robust speedup trajectory throughout, reaching 5.1× at 128K.

Estimation Overhead Comparison We further break down the estimation overhead in Figure 7. The results highlight the structural advantage of Prism’s purely block-level design. Notably, Prism achieves the lowest estimation latency across all sequence lengths. Baselines like MInference and FlexPrefill maintain a relatively high constant overhead due to their token-level estimation components. Furthermore, XAttention suffers from a dramatic latency spike on long sequences (∼ 85 ms at 128K), primarily due to the cost of its token-level access and computation. In contrast, Prism scales gracefully with sequence length, directly benefiting from its efficient matrix-multiplication-based scoring. This advantage extends to memory consumption, where Prism scales efficiently, requiring only ∼ 20% of the memory used by FlexPrefill at 128K and remaining the lowest across all sequence lengths.

#### 4.4 Ablation Studies

D128

| |
|---|

LD96_HD0

35.1

LD96_HD32 LD96_HD64 LD64_HD64

35.0

34.9

| |
|---|

34.8

PPL

34.7

| |
|---|

34.6

| |
|---|

34.5

34.4

0.35 0.40 0.45 0.50 0.55 Density

Figure 8 Perplexity vs. Density with various dimension division strategies at 32K length.

No Calibration

Calibrated

55

| |
|---|

50

| |
|---|

PPL

45

| |
|---|

| |
|---|

40

| |
|---|

| |
|---|

| |
|---|

35

| |
|---|

0.1 0.2 0.3 0.4 0.5 0.6 Density

Figure 9 Effect of Energy-Based Temperature Calibration.

Spectral Division We analyze the impact of different spectral band configurations on the Perplexity-Density trade-off in Figure 8 with the following findings:

- • Mean Pooling is indeed a Low-Pass Filter: Using only the low-frequency band (i.e., dlow = 96, dhigh = 0) exhibits a nearly identical behavior to directly using the full dimension, even lower than the full dimension case, indicating that high-frequency components are acting only as noise in mean pooling block importance estimation.
- • Necessity of Transition Zone in High-Frequency Band: Restricting the high-frequency band to the

theoretical dead zone (dhigh = 32) yields suboptimal performance. This confirms that within the dead zone, positional signals are effectively erased by destructive interference. Consequently, attempting to align and calibrate this subspace only amplifies background noise, causing severe performance degradation. Extending the branch to dhigh = 64 is thus critical to capture the recovering signals in the transition zone for effective restoration.

- • Robustness of Overlapping: While the aggressive semantic slicing (dlow = 64) appears promising at low densities, it exhibits performance instability (a U-shaped curve) at higher densities. We attribute this to the exclusion of the transition zone (d ∈ [32,64]). By extending to dhigh = 96 (red), we create a

spectral overlap where the transition zone is covered by both branches. This design is crucial because the transition band, having moderate energy, acts as a spectral regularizer for the low-frequency branch: it moderates the energy density to prevent over-calibrated temperatures while ensuring signal continuity between positional and semantic regimes.

Effect of Energy-Based Temperature Calibration We validate the necessity of our derived calibration formula by comparing the PPL-Density trade-off against a baseline with fixed temperature (τlow = τhigh = 1.0). As shown in Figure 9, the calibrated configuration consistently dominates the uncalibrated one, pushing the Pareto frontier significantly towards better efficiency. Without calibration, the high-frequency logits remain attenuated, resulting in a flattened softmax distribution (high entropy). Consequently, the adaptive Top-P policy fails to distinguish weak positional signals from background noise, forcing it to select a large number of irrelevant blocks, leading to an inefficient density inflation. In contrast, our calibration restores the logit magnitude, effectively sharpening the distribution to capture salient information within a limited density budget.

### 5 Conclusion

In this work, we identified the spectral attenuation induced by mean pooling under RoPE as the theoretical bottleneck for efficient block importance estimation. To address this, we introduced Prism, a training-free framework that explicitly preserves high-frequency information via dual-band scoring and energy-based calibration. By enabling precise selection using exclusively block-level operations, Prism achieves a 5× speedup at 128K context while maintaining performance parity with full attention, offering a robust and scalable solution for long-context and multimodal LLMs.

### References

- [1] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3119–3137. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.172. URL https://aclanthology.org/2024.acl-long.172/.

- [2] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A benchmark for long-context interleaved video-language understanding. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-0907. URL https://neurips.cc/virtual/2024/poster/97862.

- [3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://papers.nips.cc/paper/7181-attention-is-all-you-need.

- [4] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. MInference 1.0: Accelerating pre-filling for longcontext LLMs via dynamic sparse attention. In Advances in Neural Information Processing Systems, volume 37,

2024. doi: 10.52202/079017-1663. URL https://neurips.cc/virtual/2024/poster/94208.

- [5] XunhaoLai, JianqiaoLu, YaoLuo, YiyuanMa, andXunZhou. FlexPrefill: Acontext-awaresparseattentionmechanism for efficient long-sequence inference. In The Thirteenth International Conference on Learning Representations, 2025. URL https://iclr.cc/virtual/2025/oral/31861.

- [6] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

- [7] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/j.neucom.2023.127063. URL https://www.sciencedirect.com/science/article/pii/S0925231223011864.

- [8] Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In The Eighth International Conference on Learning Representations,

2020. URL https://iclr.cc/virtual_2020/poster_SylKikSYDH.html.

- [9] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? In First Conference on Language Modeling, 2024. URL https://colmweb.org/2024/AcceptedPapers.html.

- [10] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24108–24118, 2025.

- [11] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers,

2019. URL https://arxiv.org/abs/1904.10509.

- [12] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer, 2020. URL https://arxiv.org/abs/2004.05150.
- [13] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2024. URL https: //iclr.cc/virtual/2024/poster/18794.

- [14] Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. SpargeAttention: Accurate and training-free sparse attention accelerating any model inference. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 76397–76413. PMLR, 2025. URL https://proceedings.mlr.press/v267/zhang25ch.html.

- [15] Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. XAttention: Block sparse attention with antidiagonal scoring. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 69819–69831. PMLR, 2025. URL https://proceedings.mlr. press/v267/xu25ag.html.

- [16] Xinghao Wang, Pengyu Wang, Dong Zhang, Chenkun Tan, Shaojun Zhou, Zhaoxiang Liu, Shiguo Lian, Fangxu Liu, Kai Song, and Xipeng Qiu. Sparser block-sparse attention via token permutation, 2025. URL https://arxiv.org/ abs/2510.21270.
- [17] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, et al. The Llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.
- [18] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [19] GLM-4.5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, Yean Cheng, Yifan An, et al. GLM-4.5: Agentic, reasoning, and coding (ARC) foundation models,

2025. URL https://arxiv.org/abs/2508.06471.

- [20] Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. Olmo 3, 2025. URL https://arxiv.org/abs/2512.13961.
- [21] Xiaoran Liu, Hang Yan, Chenxin An, Xipeng Qiu, and Dahua Lin. Scaling laws of RoPE-based extrapolation. In The Twelfth International Conference on Learning Representations, 2024. URL https://iclr.cc/virtual/2024/ poster/18943.

- [22] Weĳie Kong, Qi Tian, Zĳian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. HunyuanVideo: A systematic framework for large video generative models, 2024. URL https://arxiv.org/abs/2412.03603.
- [23] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

- [24] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. YaRN: Efficient context window extension of large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://iclr.cc/virtual/2024/poster/17499.

- [25] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL technical report, 2025. URL https://arxiv.org/abs/2511.21631.
- [26] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, 2024. URL https://iclr.cc/virtual/2024/poster/17889.

- [27] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads, 2022. URL https://arxiv.org/abs/2209.11895.

## Appendix

### A Derivation of Spectral Attenuation Factor

In this section, we provide the detailed derivation of the spectral attenuation factor λj(B) introduced in Eq. 6 and its convergence to the sinc function in Eq. 7.

#### A.1 Setup and Geometric Summation

Consider the j-th frequency component of the query vector under Rotary Positional Embeddings (RoPE). We model the embedding at position n as a complex number:

qn(j) = c(j) · einθ

j (14)

where c(j) represents the semantic content (magnitude and initial phase) and θj is the rotation frequency. To isolate the effect of pooling on positional information, we assume the semantic content c(j) is locally stationary (constant) within the pooling window.

The mean pooling operation over a block of size B (indexed locally from k = 0 to B − 1) yields the pooled vector q¯(j):

B−1

B−1

c(j)ein

1 B

0θj

q¯(j) =

c(j) · ei(n

0+k)θj =

j (15)

eikθ

B

k=0

k=0

where n0 is the start position of the block. The term S = B−1

j)k is a geometric series with ratio r = eiθ

k=0 (eiθ

j. Applying the summation formula for a finite geometric series:

1 − (eiθ

j)B 1 − eiθj

1 − eiBθ

j

S =

=

(16)

1 − eiθj

#### A.2 Magnitude Calculation (The Dirichlet Kernel)

We define the attenuation factor λj(B) as the ratio of the magnitude of the pooled vector to the magnitude of the original content |c(j)|. Note that the phase term |ein

0θj| = 1 and thus does not affect the magnitude.

λj(B) ≜ |q¯(j)| |c(j)|

1 B |S| =

1 B

1 − eiBθ

j

=

(17)

1 − eiθj

To simplify the magnitude of the complex fraction, we utilize the half-angle identity |1−eiϕ| = |eiϕ/2(e−iϕ/2 − eiϕ/2)| = | − 2isin(ϕ/2)| = 2|sin(ϕ/2)|. Applying this to both the numerator (ϕ = Bθj) and the denominator (ϕ = θj):

- 1 B

- 2|sin(Bθj/2)|

1 B

sin(Bθj/2) sin(θj/2)

=

λj(B) =

(18)

2|sin(θj/2)|

This function is known as the normalized Dirichlet kernel, which describes the diffraction pattern of a discrete periodic lattice.

#### A.3 Sinc Approximation

The RoPE frequencies are defined as θj = b−2j/d. For dimensions j away from 0, the frequency θj decays exponentially and becomes very small (θj ≪ 1). We apply the small-angle approximation sin(x) ≈ x to the

denominator term1:

θj 2

sin(θj/2) ≈

(19) Substituting this into the expression for λj(B):

1 B

λj(B) ≈

sin(Bθj/2) θj/2

(20)

We rearrange the terms to match the form of the normalized sinc function, defined as sinc(u) ≜ sin(πuπu):

λj(B) ≈

sin(Bθj

2 ) Bθj 2

Let πu = Bθj

2 , which implies u = Bθj

2π . Substituting u yields the final approximation:

(21)

λj(B) ≈ sinc

Bθj 2π

(22)

This derivation confirms that mean pooling acts as a rectangular window filter in the signal domain, leading to the sinc-shaped spectral response shown in Figure 2.

#### A.4 Relaxing the Locally Stationary Assumption

The locally stationary assumption on c(j) is used above only to isolate the effect of RoPE rotation and make the sinc attenuation factor transparent. The same conclusion still holds when the semantic content varies within a block. Let

B−1

B−1

1 B

c(kj) = c¯(j) + δk(j), c¯(j) =

c(kj),

δk(j) = 0. (23)

k=0

k=0

Then the pooled representation decomposes into a mean-content term and a residual variation term:

ein

0θj

q¯(j) =

B

B−1

B−1

δk(j)eikθ

c ¯(j)

j +

eikθ

j

k=0

k=0

. (24)

The first term is exactly the component analyzed above and is attenuated by λj(B). In the Dead Zone, where λj(B) ≈ 0, this mean-content term is removed by destructive interference. The residual term

B−1

1 B

δk(j)eikθ

Rj =

j (25)

k=0

is always bounded by

B−1

1 B

|δk(j)| ≤ max

|δk(j)|, (26)

|Rj| ≤

k

k=0

so arbitrary intra-block variation cannot restore the vanished mean signal. We can further obtain a tighter typical-case bound. Assume the variations have zero mean and per-token variance at most σj2. Then

1 B2 k,l

E δk(j)δl(j) ei(k−l)θ

E |Rj|2 =

. (27)

j

1The small-angle approximation sin(x) ≈ x holds due to the exponential decay of RoPE frequencies θj = b−2j/d. Taking Qwen3 (b = 106, d = 128) as an instance, the frequency drops to θ10 ≈ 0.11 by the 10th dimension pair. At this point, the relative error is already < 0.2%. Thus, for the vast majority of the spectrum (j > 10), θj is sufficiently small to make the sinc model analytically exact.

For k ̸= l, the cross terms are suppressed either because local semantic deviations are weakly correlated across positions, or because the rapidly rotating phase ei(k−l)θ

j in the Dead Zone causes destructive interference over the sum. Keeping the diagonal terms gives

B−1

σj2 B

1 B2

E |δk(j)|2 ≤

E |Rj|2 ≈

. (28)

k=0

√

B. For the default B = 128, this yields a reduction factor 1/√128 ≈ 0.088, consistent with the empirical energy collapse in Figure 3. The worst-case bound is tight only if the variations align adversarially as δk(j) ∝ e−ikθ

Thus the residual RMS scales as σj/

j, which would require semantic content to oscillate at exactly the RoPE frequency and opposite phase across many Dead Zone dimensions. Such alignment is implausible in trained representations and is not observed empirically.

### B Compatibility with YaRN

For Qwen3-8B, we use YaRN [24] to extend the native context length from 32K to 128K. For spectral-boundary analysis, the relevant component of YaRN is its NTK-by-parts interpolation, which modifies the RoPE frequency of each dimension as

θj s

θj′ = (1 − γj)

+ γjθj, (29)

where s is the extension ratio and γj = γ(rj). YaRN additionally applies an attention scaling factor, but that factor does not change the RoPE frequency boundaries analyzed here. The interpolation coefficient γj is determined by rj = Lθj/(2π):

 

1, rj > β, 0, rj < α, rj−α β−α , α ≤ rj ≤ β,

γj =

(30)



with default α = 1 and β = 32. For Qwen3-8B, we have b = 106, d = 128, native context length L = 32768, and extension ratio s = 4. Computing rj over the spectral zones gives the following behavior:

Zone Dim. range rj range YaRN scaling Effect on Prism

Dead 2j ≲ 30 rj ≳ 2.0×102 γj = 1, unchanged Attenuation analysis preserved Transition (unchanged) 30 ≲ 2j ≲ 47 rj > 32 γj = 1, unchanged Unaffected

Transition (scaled) 47 ≲ 2j ≲ 79 1 < rj < 32 Partial scaling, θj′ < θj Attenuation reduced (favorable) Semantic 2j ≳ 79 rj < 1 Full scaling, θj′ = θj/s Alreadynear-lossless, remainsso

Therefore, YaRN does not modify the Dead Zone frequencies that motivate Prism’s high-frequency branch. Within dhigh = 64, dimensions up to 2j ≈ 47 are unchanged, and only the tail of the transition region receives mild scaling, which reduces rather than increases attenuation. The low-frequency branch covers the fully scaled long-wavelength region and the adjacent transition region; both are near-lossless under mean pooling or become less attenuated after frequency scaling. This explains why the same spectral split remains effective for Qwen3-8B after YaRN extension to 128K.

### C Top-P Block Selection

- Figure 10 provides the PyTorch-style implementation of the Top-P selection process used in Prism. The function takes block-level probabilities as input and sorts the key blocks for each query block based on relevance. Subsequently, it selects the minimal set of blocks required for the cumulative probability to exceed the threshold p. Finally, the original spatial order is restored via a scatter operation.

### D Experimental Setup Details

def top_p(probs, p):

- # 1. Sort probabilities sorted_probs, sorted_indices = sort(probs, descending=True, dim=-1)
- # 2. Compute cumulative probabilities cumulative_probs = cumsum(sorted_probs, dim=-1)
- # 3. Thresholding sorted_mask = (cumulative_probs - sorted_probs) < p
- # 4. Scatter to restore order mask = zeros_like(probs) mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask) return mask

Figure 10 PyTorch-style implementation of the Top-P block selection.

- D.1 Datasets We provide detailed descriptions of the benchmarks used in our evaluation:

- • PG19 [8]: A standard benchmark consisting of full-length books, used to evaluate the model’s ability to model long-range dependencies via perplexity.
- • LongBench [1]: A bilingual, multi-task benchmark consisting of 21 datasets across 6 task categories in both English and Chinese, designed to measure broader understanding capabilities.
- • RULER [9]: A synthetic benchmark designed to measure the retrieval capability of long-context language models.
- • Video Benchmarks: VideoMME [10] and LongVideoBench [2]. We use max pixels of 327680 for each frame and 1 frame per second for video sampling, which translate to approximately 107K tokens per hour.
- • Video Generation: We evaluate HunyuanVideo [22] with prompts sampled from VBench [23].

- D.2 HunyuanVideo Spectral Configuration

HunyuanVideo uses 3D-RoPE with RoPE base b = 256 and partitions each attention head into temporal, height, and width subspaces with dimensions [16,56,56]. Since the base is much smaller than that of LLM RoPE (e.g., 106), high-frequency attenuation is stronger and the LLM setting dhigh = 64,dlow = 96 is not directly transferable. We therefore apply the same attenuation criterion axis-wise. For an axis with subspace dimension da, the first cancellation point satisfies Bθj ≈ 2π with θj = b−2j/d

a, giving the real-dimension cutoff

ln(B/2π) ln b

2ja⋆ ≈ da ·

. (31)

With B = 128 and b = 256, this yields 2jt⋆ ≈ 8.7 for the temporal subspace and 2jh⋆ = 2jw⋆ ≈ 30.4 for the spatial subspaces. Accordingly, we set the high-frequency branches to dthigh = 8 and dhhigh = dwhigh = 32. For the low-frequency branches, we keep the tail dimensions while overlapping the transition region, using dtlow = 16 and dhlow = dwlow = 36. This mirrors the overlap strategy used for LLM experiments: the high-frequency branch covers the strongly attenuated local-position dimensions, while the low-frequency branch preserves the more stable semantic and transition dimensions.

[Figure 10]

(a) Full Attention

[Figure 11]

(b) XAttention (threshold 0.95)

[Figure 12]

(c) Prism (threshold 0.97)

- Figure 11 Qualitative comparison on HunyuanVideo. Each row shows generated frames from the same prompt under the corresponding attention implementation. XAttention uses threshold 0.95, and Prism uses threshold 0.97.

- D.3 Baselines Configuration We compare Prism with the following baselines using their official implementations:

- • MInference: A method employing offline search to classify attention heads into pre-defined heuristic patterns for subsequent block importance estimation. We use the recommended “Vertical-Slash” pattern configurations.
- • FlexPrefill: An approach utilizing online search to dynamically switch between static patterns and mean-pooling based estimation depending on input contexts. We adopt γ = 0.95,τ = 0.1 following the original paper.
- • XAttention: A unified method introducing antidiagonal scoring to capture both geometric and semantic patterns without explicit head classification. We use threshold p = 0.9 and stride S = 8 following the original paper.
- • PBS-Attn: A permutation-based block-sparse attention method that reorders tokens to cluster critical tokens, improving block-level sparsity for block selection. We use p = 0.9 and a segment size of 256 following the original paper.

We do not include SpargeAttention [14] in the main quantitative comparison because it combines coarsegrained block estimation with orthogonal system optimizations such as Q/K quantization and warp-level block skipping. These optimizations are complementary to Prism’s block-importance estimator and make it difficult to isolate the effect of the estimation mechanism itself. We therefore focus the main comparison on training-free dynamic sparse methods whose primary difference lies in how relevant blocks are selected.

### E Qualitative Video Generation Results

We provide qualitative HunyuanVideo results in Figure 11. The visual comparison uses the same prompt and sampling setup for full attention, XAttention, and Prism, showing that Prism preserves visual content and temporal consistency while improving efficiency.

24.7

B=64

B=128 B=256 B=512

24.6

24.5

24.4

PPL

24.3

Full Attention

24.2

24.1

| |
|---|

0.20 0.25 0.30 0.35 0.40 0.45 Density

20

EstimationTime(ms)

15

10

5

0

8192 16384 32768 65536 131072 Sequence Length

- Figure 12 Effect of Block Size B. The upper panel illustrates the perplexity at various densities with a context length of 128K using Llama-3.1-8B-Instruct. The lower panels illustrates the estimation time at various sequence lengths.

### F Effect of Block Size

Theoretically, a smaller block size B enhances the Signal-to-Noise Ratio (SNR) by reducing spectral attenuation, but quadratically increases the estimation overhead due to the larger number of blocks (N = L/B). Figure 12 empirically validates this trade-off. In terms of accuracy (upper panel), finer granularity (B = 64) consistently yields better performance, even outperforming the full attention baseline due to effective noise filtering. B = 128 closely follows this trend, matching full attention at reasonable densities. However, in terms of efficiency (lower panel), the estimation latency for B = 64 rises sharply, reaching ∼ 22 ms at 128K. Although this is still faster than many existing baselines (Figure 7), it is more than double the overhead of B = 128 (∼ 9 ms). Consequently, we select B = 128 for the main experiments, as a good compromise between accuracy and efficiency.

