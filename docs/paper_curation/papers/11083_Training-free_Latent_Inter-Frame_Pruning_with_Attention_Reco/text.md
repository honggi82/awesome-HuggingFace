### Video Compression Meets Video Generation: Latent Inter-Frame Pruning with Attention Recovery

Dennis Menn1, Yuedong Yang1, Bokun Wang1, Xiwen Wei1, Mustafa Munir1, Feng Liang2, Radu Marculescu1, Chenfeng Xu1, and Diana Marculescu1

[Figure 1]

1University of Texas at Austin 2Meta Project page: https://dennismenn.github.io/lipar

## arXiv:2603.05811v2[cs.CV]28Apr2026

Input Video Robot Cartoon Santa King Warped Video TTM

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

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

…

…

…

…

…

…

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Win Win

Draw

16.6 GB 20%

###### 12.2 FPS 45%

LIPAR (Ours) Speed Memory Self-Forcing 20.7 GB

###### LIPAR (Ours)

20.2% 66.2% 13.6%

Smaller

Faster

Self-Forcing 8.4 FPS

Preference

LIPAR (Ours)

Self-Forcing

Fig. 1: Latent Inter-frame Pruning with Attention Recovery (LIPAR). This training-free pruning method extends Inter-Frame Compression from pixel to latent space by reusing edited results from previous frames (gray regions) to save computation. Results shown are from an RTX A6000 (verified consistent on an RTX 4090).

Abstract. Current video generation models suffer from high computational latency, making real-time applications prohibitively costly. In this paper, we address this limitation by exploiting the temporal redundancy inherent in video latent patches. To this end, we propose the Latent Inter-frame Pruning with Attention Recovery (LIPAR) framework, which detects and skips recomputing duplicated latent patches. Additionally, we introduce a novel Attention Recovery mechanism that approximates the attention values of pruned tokens, thereby removing visual artifacts arising from naively applying the pruning method. Empirically, our method increases video editing throughput by 1.53×, achieving an average of 19.3 FPS on an NVIDIA RTX 4090 with the 1.3B Self-Forcing model (4-step denoising, FP16). The proposed method does not compromise generation quality and can be seamlessly integrated with the model without additional training. Our approach effectively bridges the

gap between traditional compression algorithms and modern generative pipelines.

Keywords: Video generation · Efficiency · Token pruning

#### 1 Introduction

Diffusion Transformers (DiTs) have emerged as a dominant force in generative tasks, achieving remarkable success in high-fidelity image and video synthesis [13, 20]. However, their practical deployment is severely constrained by computational inefficiencies [28,29]. Despite recent advances, such as the adaptation of causal attention and few-step distillation [31], video generation remains a compute-demanding task. Furthermore, high computation costs impede realtime human-machine interaction (e.g., 30 fps) on a single GPU [7,9,23,24].

To reduce computational costs, traditional video compression algorithms identify repeated patches in temporal and spatial dimensions to avoid reprocessing them in pixel space [15]. In contrast, the current Latent Diffusion Model (LDM) framework allocates fixed compute for every token, regardless of redundancy in the content [13,22,26]. This is primarily due to the limited understanding of semantics in the latent space and the difficulty in pinpointing redundancy prior to the generation process.

Previous methods have attempted to implicitly exploit this redundancy by merging similar tokens in each attention block to prevent re-computation [2,3,6, 27]. However, these methods suffer from several drawbacks. The computational overhead is large due to the frequent, expensive process of determining similar tokens for each block; additionally, token merging is often restricted to certain layers, thereby failing to save computation across all layers [3,6,27]. Quality-wise, directly merging tokens results in visual artifacts in the causal attention backbone [9] due to the induced training-inference discrepancy arising from pruning.

In this paper, we propose Latent-inter Frame Pruning with Attention Recovery (LIPAR) for conditioned video generation. This training-free method starts by identifying redundant patches in the latent space and performing end-to-end pruning, thereby allowing all layers to benefit from the speedup. Furthermore, we propose an approximation condition that pruning must satisfy, alongside a solution, Attention Recovery, that closes the training-inference gap stemming from pruning, thereby preserving generation quality.

LIPAR, tested on 51 video-text prompts from the Davis dataset [21], achieves a 1.45× speedup in throughput, reaching 12.2 FPS on a single A6000 GPU with a 20% reduction in GPU usage (requiring only 16.6 GB). We further assess generation quality by performing evaluation tests with 14 human participants. The results indicate an 86.4% win-tie rate compared with the original (unpruned) results, demonstrating the high visual quality of the proposed method and a clear improvement compared to existing training-free pruning methods. Additionally,

- our method can be generalized from causal attention [9,31] to bidirectional attention [26]. Our contributions are summarized as follows:

- 1. Observation: In Section 3, we identify strong Pearson correlations between the change of pixel-space and latent-space distances across the temporal axis, which motivates the adaptation of traditional pixel-space video compression algorithms to the modern generative pipeline.
- 2. Theoretical Analysis: In Section 4, we formulate the training-inference discrepancy arising from direct token pruning and establish a general mathematical condition that pruning must satisfy to preserve visual quality.
- 3. Pipeline Design: In Section 5.1, we design a pipeline that integrates Interframe Compression with LDMs in video editing tasks. The proposed method precisely prunes temporally repeated tokens while maintaining the generated token number for decoding.
- 4. Proposed Solution: In Section 5.3, we propose Attention Recovery to approximate the output of the unpruned token sequence. This allows LIPAR to achieve a speedup of O(n) (where n represents the remaining tokens) while maintaining high visual quality in the edited video. The method is training-free and generalized to both causal and bidirectional attention.

#### 2 Related Works: Accelerating Diffusion Models

To mitigate the high computational cost of Transformers, several methods attempt to reduce token counts during inference. Token Merging [2,3] introduced a bipartite matching algorithm to merge redundant tokens in the Transformer architecture. Subsequent works refined the token selection algorithm, utilizing classifier-free guidance or attention weights to select semantically important tokens [6, 16, 27]. Parallel to token reduction, sparse video generation methods [28,29] focus on optimizing attention computation through semantic-aware permutation techniques. Another line of research accelerates generation via feature caching, skipping specific layers across denoising timesteps [11, 18]. Additionally, CausVid [31] applies few-step distillation [30] to accelerate video generative models. Our method is orthogonal to previous acceleration techniques (e.g., feature caching and few-step distillation); instead, LIPAR exploits temporal redundancy within the latent space. Furthermore, compared to previous pruning methods, LIPAR enables end-to-end pruning that utilizes the inherent redundancy in the latent space and formulates approximations that preserve output fidelity. Consequently, we achieve high visual quality with low overhead. Run length tokenization [4] is closest to our work; their approach prunes temporally redundant tokens for sparse prediction tasks (e.g., classification), where pruned tokens do not need to be recovered. Although our method also targets temporally redundant tokens, our core contribution focuses on recovering pruned tokens via Attention Recovery. Furthermore, we explore latent space properties to integrate the approach into the LDM pipeline, which prior work has not addressed.

In Appendix 10, we discuss additional work related to interactive video generation.

#### 3 Motivation: Empirical Evidence

A fundamental concept of video compression in pixel space is that temporally unchanged pixels do not need to be re-transmitted [15]. To adapt the video compression algorithm from pixel to latent space, the latent space may need to inherit this property, i.e., there must exist patches that remain unchanged along the temporal or spatial axis. By identifying these redundant patches, we can copy them from previous frames rather than re-generating them, thereby reducing computational overhead. To validate this property, we measure the correlation between changes in pixel space and changes in latent space across the temporal axis. A strong correlation indicates that pixel-level temporal dynamics are preserved within the latent manifold. Consequently, a patch that remains unchanged in the pixel space is likely to remain unchanged in the latent space. We evaluate this using the following metric:

Corr ∥p(pixelt,x,y) − p(pixelt+1,x,y)∥1, ∥p(latentt,x,y) − p(latentt+1,x,y)∥1 (1)

where p is a patch in the pixel or latent space, and (t,x,y) denotes its spatial location (x,y) and temporal index t. We conducted this analysis on the entire DAVIS 2017 train-val set [21], using a latent patch size of (2,2,2) across the temporal and spatial axes to minimize noise and align with the token dimensions, with the corresponding pixel patch size scaled by the VAE compression rate. To ensure generalizability, we tested both the WAN 2.1 VAE and WAN 2.2 TI2V VAE [26]. We employ the L1 norm to quantify change and Pearson correlation, as in Eqn. 1, to measure the relationship between the two spaces.

Our results show a strong correlation between pixel-space and latent-space changes: 0.69 for WAN 2.1 VAE and 0.77 for WAN 2.2 VAE. It is crucial to highlight that this finding is non-intuitive. Given the heavy spatial compression performed by the encoder, there is no a priori guarantee that the latent manifold would preserve the temporal redundancy and the Pearson correlation coefficient observed in the raw pixel space.

To further test temporal redundancy in the latent space, we select ten videos from the DAVIS dataset and substitute (nearly) unchanged patches with those from the previ-

[Figure 22]

[Figure 23]

CompressedOriginal

- ous frame to create a “compressed” latents. Even after compressing 46% of the latents, the decoded output maintained high visual fidelity, with an averaged Learned Perceptual Image Patch Similarity (LPIPS) ≤ 0.05, compared with the original decoded video [32]. We illustrate one such example in Figure 2. For the detailed experimental settings, please refer to

[Figure 24]

[Figure 25]

Fig. 2: Decoding Compressed Latents. Original: Directly decode the video latents; Compressed: Compressed (nearly) unchanged latent patches.

- Appendix 11. These findings reaffirm that temporal redundancy exists in the latent space and support the adaptation of traditional video compression methods to the latent space.

4 Problem Formulation

- 4.1 Target Objective

Given that temporal redundancy exists in latent space, our next objective is to ensure that the generated token of the temporally pruned sequence approximates that of the full, unpruned sequence. Formally, we require the reconstructed output, obtained by pruning, denoising, and then duplicating, to approximate the original denoised output, as shown on the left side of Eqn. 2 below.

However, since we restrict pruning to temporally redundant tokens, we can show that the left side of Eqn. 2 simplifies to the right side. This is because the values of pruned tokens are close to their predecessors; we only need to ensure that the values of the kept tokens approximate those of the full sequence. Consequently, our goal simplifies to ensuring that the denoising operation commutes with the pruning operation:

R D(P(xt)) ≈ D(xt) Goal

=⇒ D(P(xt)) ≈ P(D(xt)) (2)

where xt is the token sequence at time t, P represents the pruning operator, D is the denoising network, and R denotes the recovery operator (reusing temporal predecessors).

Note that the sufficient condition for Eqn. 2 to hold can be reduced to approximating the Multi-head Self-Attention (MSA) outputs within each block between the pruned and unpruned sequences, as shown in Eqn. 3 below.

MSA(P(xt)) ≈ P(MSA(xt)) (3)

This is because self-attention is the only operation that depends on the entire token sequence. If the self-attention outputs are approximated, the outputs of subsequent layers, e.g., cross-attention and linear layer, which operate per-token, will align correspondingly, preserving the overall approximation. See Appendix 12 for the derivation.

- 4.2 MSA Approximation

To satisfy Eqn. 3, we consider the one-dimensional case where tokens have fixed spatial position with varying temporal positions, as shown in Figure 3. In this example, we assume tokens x2, x3 and x5 are pruned and our goal is to find a function which operates on P(xt) and satisfies Eqn. 3. Note that the derivation for this example extends naturally to the general case.

To ensure compatibility with FlashAttention [5], the proposed function must operate outside the core attention calculation. Specifically, it is restricted to

# ≈

x’ (pruned)

x

x1 x2 x3 x4 x5 x6

x1 x4 x6

- Fig. 3: Illustration of the approximation of pruned tokens to the unpruned token sequence. Dashed circles indicate pruned tokens, where x1 ≈ x2 ≈ x3 and x4 ≈ x5.

modifying either the input vectors (q, k, v) prior to the attention calculation, or the resulting attention output afterward. Mathematically, our objective is to define functions f and g such that the attention output computed from the kept tokens approximates the original output:

Tf(kj,cj))vj j∈R g(eqTf(kj,cj)) ≈

Tkivi

N i=1 eq

j∈R g(eq

(4)

N i=1 eqTki

where N is the total number of (unpruned) tokens, R denotes the set of indices for the tokens that remain after pruning, and cj represents the number of tokens approximated by the unpruned (remaining) token j (such that j∈R cj = N). We require that the approximation error is bounded by O(δ), where δ represents the maximum token approximation error (defined below).

Pruning temporally unchanged tokens ensures that the underlying tokens have similar values, i.e., k1 ≈ k2 ≈ k3 and k4 ≈ k5, as shown in Figure 3. (Note that, in this approximation, we disregard the impact of different noise values added to each token, which will be addressed in the subsequent section). Furthermore, RoPE [25] introduces position-dependent variations in attention, requiring explicit handling of rotational effects.

Replacing the keys from the pruning approximation into the original MSA calculation yields an expanded form of the attention output computed over the full sequence:

cj−1 m=0 eq

T(emθikj) vj

Tkivi

N i=1 eq

≈ j∈R

(5)

cj−1 m=0 eqT(emθikj)

N i=1 eqTki

j∈R

where cj is the number of tokens approximated by the kept token j and mθ is the angle induced by RoPE. Note that the approximation error is bounded by O(δ), where δ = maxi,j,m ∥ki − emθikj∥, due to the Lipschitz continuity of the self-attention calculation with respect to the keys. Consequently, combining Eqn. 4 and Eqn. 5, the objective further simplifies to finding f and g such that, for any query q from the remaining tokens, the following approximation holds:

cj−1

Tf(kj,cj)) ≈

g(eq

m=0

T(emθikj) (6)

eq

##### 4.3 The Impact of I.I.D. Noise

Although the values of two temporally redundant patches may be similar (P1 ≈ P2) in the latent space, each is perturbed by independent Gaussian noise ϵi ∼

N(0,I). Consequently, naively assuming that the resulting tokens are close (x1 ≈ x2) ignores this independence. This introduces artificial correlations, leading to noise amplification during the attention mechanism, as illustrated below.

Suppose that we decompose the tokens into a clean token and its noise components, i.e., xi = x¯i + ϵi. The query, key, and value vectors are respectively:

qi = q¯i + WQϵi, ki = k¯i + WKϵi, vi = v¯i + WV ϵi (7)

where WQ,WK, and WV are the respective projection weight matrices, and the bar notation (¯·) denotes the noise-free (signal) components of q, k, and v. For

√T ki

a fixed query, the attention output over N tokens is Ni=1 σ q

D vj, where σ(·) denotes softmax and D is the token dimension. Expanding the dot product gives:

qTki = q¯Tk¯i + q¯TWKϵi + ϵTWQTk¯i + ϵTWQTWKϵi (8)

where ϵ and ϵi are the noise added to q and ki respectively. Let WQTWK ≈ I for illustrative purposes (for the general case where there is no restriction on

WQTWK, please refer to Appendix 13, where our conclusions still hold). Assume x1 ≈ x2 means ϵ ≈ ϵi. This leads to two critical consequences:

1. Attention Score Calculation: The quadratic noise term ϵTi ϵj changes distribution from the Gaussian Distribution N to Chi squared χ2 distribution.

ϵTi ϵj ∼

N(0,D) if ϵi ̸= ϵj (independent) χ2D if ϵi = ϵj (duplicated)

(9)

Note that by Central Limit Theorem, N(0,D) is an approximation for large token dimension D. The duplicated case introduces a large positive bias (E[χ2D] = D) and higher variance (2D), inflating attention weights on duplicated tokens.

2. Value Aggregation: Duplication changes the summed noise from

WV nj=1 ϵj ( variance O(nID)) to nWV ϵ (variance O(n2ID)), resulting in quadratic variance explosion.

Empirically, forcing x1 = x2 by duplication produces strong, noisy patterns and significantly degrades the quality of the generated videos, as shown in Section 7.1, highlighting the importance of accounting for I.I.D. noise.

#### 5 Methods

##### 5.1 LIPAR Overview

In Figure 4, we present an overview of the proposed pruning framework, which operates in three stages to accelerate the conditioned video generation task. First, we apply Latent Inter-Frame Pruning to remove temporally redundant patches in the latent space by comparing with the previous frame. Note that pruning patches reduces the sequence length N, thereby significantly lowering computational costs due to the transformer’s quadratic O(N2) complexity.

KV-Cache (or Condition)

###### 2. Attention recovery

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

[Figure 26]

t t+1

[Figure 27]

| |
|---|

[Figure 28]

[Figure 29]

[Figure 30]

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Encoder

Decoder

###### 1. LIF pruning 3. Restoration

Source Video Edited Video

- Fig. 4: LIPAR overview: The proposed method consists of three stages: 1. Pruning 2. Attention Recovery and 3. Restoration.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

However, directly removing tokens can disrupt the distribution of input sequences, since training inputs always use complete (unpruned) latent information. This pruning-induced discrepancy alters self-attention computations, leading to visual artifacts. To mitigate this, we propose Attention Recovery, a mathematical approximation that contains an M-Degree Approximation and NoiseAware Duplication, aligning the attention scores from the pruned sequence with those from the original, unpruned calculations. Finally, the Restoration step upsamples the token count for decoding and maps the latents back to pixel space.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

|[Figure 44]<br><br>[Figure 45]| |
|---|---|
| | |

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Fig. 5: Illustration of the Attention Recovery Method. This method preserves visual quality in pruned tokens via two mechanisms: M-Degree Approximation and Noise-Aware Duplication. Pruned keys (k) and values (v) are approximated by copying temporal counterparts from the clean KV-cache (e.g., t−1) to maintain the i.i.d. noise assumption, ensuring the m closest tokens to the query remain populated. For simplicity, we only explicitly draw the Noise-Aware duplication for k.

- 5.2 Token Pruning and Restoration

Latent Inter-Frame Pruning. Diffusion latent space contains temporal redundancy. Inspired by previous works [4,15], we propose Latent Inter-frame (LIF) Pruning to identify and bypass calculating unchanged patches by comparing the difference between temporally consecutive patches at the same spatial location: ∥px,yt − px,yt+1∥1 < τ, where τ is a predefined threshold used to determine if the temporal difference is small enough to consider the patch unchanged.

Due to the high compression rate of the latent space, subtle movements within latent patches can yield difference values that fall below the pruning threshold in the above equation, leading to mispruning. During the restoration stage, erroneously reusing these tokens will repeat the subtle motions, which manifest as glitches upon decoding and degrade overall video quality. To identify subtle movements, we integrate motion detection techniques into LIF pruning by leveraging the spatial and temporal information of neighboring tokens through calculating the difference between consecutive frames, thereby reflecting video

dynamics that typically involve movement at the object-level rather than isolated pixel changes. Additionally, we improve the pruning mask by incorporating both short-term (consecutive) and long-term temporal differences. This dual-term design is important for supporting and preventing the violation of the I.I.D. noise assumption in Attention Recovery, and will be further discussed in Section 5.3. Please refer to Appendix Alg. 1 for the full algorithm.

Latent Patch Restoration. After the denoising process, the Diffusion Transformer outputs a set of pruned and denoised patches. However, since decoding requires patches with fixed dimensions, we must restore them. To achieve this, we reconstruct the pruned patches by duplicating the corresponding patches from the previous frame. Appendix Alg. 2 details this restoration procedure.

##### 5.3 Attention Recovery

Figure 5 illustrates the Attention Recovery method applied to the causal attention backbone [9,31]. This approach preserves visual quality by utilizing the pruned sequence to approximate self-attention outputs for unpruned sequence. The method relies on two core mechanisms: M-Degree Approximation and NoiseAware Duplication. M-Degree Approximation ensures that the m closest keys and values to the query remain unpruned by copying Key (K) and Value (V) vectors from their temporal counterparts. Simultaneously, Noise-Aware Duplication restricts copying to “clean” tokens, i.e., from the KV cache to avoid violating the i.i.d. assumption of noise in diffusion models. While currently applied to causal attention, this method is also extensible to bidirectional attention, as demonstrated in Section 8. Below, we explain the two mechanisms in detail.

M-degree Approximation. To recover the self-attention values from the pruned sequence, our goal is to find functions f and g approximating the exponential sum in Eqn. 10 below, as discussed in Section 4. Note that the approximation error is bounded by O(δ), where δ represents the maximum token approximation error. Here, eθi is the RoPE rotation matrix, q is the query applied with RoPE, and cj represents the number of tokens approximated by the kept token j.

cj−1

Tf(kj,cj)) ≈

g(eq

l=0

T(elθikj) (10)

eq

The right-hand side is an exponential sum derived from the log-sum-exp (LSE) approximation. By exponentiating the standard LSE bound, an m-order approximation refines this by summing over the set of the largest m terms:

cj−1

T(elθikj) ≈

T(elθikj) (11)

eq

eq

l∈M

l=0

where M denotes the set of indices corresponding to the m largest values of the exponent. Mathematically, this approximation strictly bounds the true sum from

below. Note that finding the largest m exponents is equivalent to minimizing the angular deviation between q and k, i.e., |lθ − ϕ|, where ϕ represents the angle rotated with query q. Because queries in a causal attention structure correspond to the most recent tokens, the rotated angle ϕ naturally aligns best with the rotational angles of the latest keys. Therefore, we can effectively find the set M by selecting the m most recent indices, which yields:

cj−1

f(kj,cj) = elθikj cl=0j−1 , g(X) =

Xl (12)

l=cj−m

Crucially, even at full duplication (where m = N), we still achieve a linear speedup by requiring fewer queries in the self-attention layers, thus generating fewer tokens. This reduction accelerates all Transformer layers (Feed-Forward Network, cross-attention) by a factor of N

Nkept , where Ntotal and Nkept are the total number of tokens and the number of kept tokens, respectively. Furthermore, LIPAR is compatible with parallelism tools like FlashAttention [5]; the m-degree approximation enables the pruning of redundant tokens, thus reducing the GPU memory usage and computational complexity in attention layers.

total

Noise-Aware Duplication. Although Equations 12 suggest a straightforward solution to Attention Recovery, this method fails in practice and introduces highfrequency visual artifacts. This is because we duplicate both the clean signal and the noise component, inducing artificial noise correlations across duplicated tokens, as discussed in Section 5.3. To address this, we propose Noise-Aware Duplication, which duplicates only the clean component of tokens to prevent the noise correlations during the self-attention computation.

We achieve this by duplicating the temporally closest clean tokens from the KV cache. All KV-cache tokens are clean because they are generated via an additional denoising step at a zero noise level. However, this introduces a new challenge: while previous approximation allowed Xt−1 ≈ Xt for pruned Xt, we now approximate Xt using Xt−k. Here, k represents the temporal offset, making Xt−k the closest clean token in the KV cache. To ensure a valid approximation, we add a long-term difference constraint to LIPAR. A token is pruned only if both short-term and long-term differences are satisfied, specifically:

1 if t ≡ 0 mod S, t − S⌊t/S⌋ otherwise.

(13)

∥Xt−k − Xt∥1 < τ2, k =

where τ2 is the preset threshold for the long-term difference and S is the denoising block size. This method is not restricted to causal-attention architectures.

#### 6 Experiments

We implement our pruning method on top of the Self-Forcing model [9]. Consistent with CausVid [31] and StreamV2V [17], we employ SDEdit [19] for videoto-video translation. We uses 51 video-prompt pairs from Davis Dataset [21] for the experiments. Please refer to Appendix 15 for detailed experimental settings.

Input video LIPAR (Ours) Self-Forcing StreamV2V StreamDiﬀusion ControlVideo

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Fig. 6: Qualitative comparison with representative low latency V2V models. Our method achieves comparable results to Self-Forcing while having higher throughput, and outperforms the rest of the models. Prompt: “Three corgi puppies sharing a meal together on a kitchen floor.” We encourage readers to refer to the supplementary materials for more video comparisons.

##### 6.1 Comparison with Other Models

In Figure 6, we qualitatively compare our proposed pruning method against several representative V2V models. While Self-Forcing generates high-quality videos by processing all tokens in every frame, re-editing temporally unchanged tokens incurs unnecessary computational cost and introduces temporal instability, resulting in subtle fluctuations in the background (highlighted by the green square). In comparison, StreamDiffusion [12] and StreamV2V [17] often yield lower visual quality characterized by flickering or structural defects, as highlighted by the red squares where the dogs’ heads merge to form unnatural shapes. Similarly, while ControlVideo [33] achieves strong editing effects, the generated video still suffers from structural defects such as the fused dog faces highlighted in the rectangle. In contrast, our method matches or exceeds the visual quality of all baselines, as verified by the human evaluation in Figure 7, while significantly increasing throughput by 1.45×.

Human Evaluation. Following TokenFlow [8] and StreamV2V [17], we assess perceptual quality using a TwoAlternative Forced Choice protocol with 51 video-prompt pairs from the DAVIS dataset [21], where participants select the better of two sideby-side videos. The study involved 14 participants, each performing 100 pairwise comparisons. Refer to Appendix 16 for the evaluation webpage.

- Figure 7 summarizes the hu-

man evaluation results. Participants slightly preferred LIPAR (18.4%) over the unpruned Self-Forcing baseline

Ours Draw Baseline

###### 18.4 68.3 13.3

SF SV2V SD Control

92.4

4.7

2.9

LIPAR

94.0

4.2

1.3

84.9

13.0

2.1

Win Rate (%)

12.2 9.1

LIPAR SD SF

1.34 × 1.45 ×

8.4 5.9

2.07 × 64.2 × Speedup

SV2V

Control

0.19

Throughput (FPS)

Fig. 7: Comparison of user preference and throughput against other models.

(13.3%), with 68.3% tying. We attribute this preference to LIPAR’s reuse of unchanged video patches, which enhances temporal consistency in the background. Furthermore, LIPAR demonstrates a decisive advantage over real-time competitors, achieving win rates exceeding 84% against StreamDiffusion, StreamV2V, and ControlVideo. This confirms that our method significantly outperforms previous state-of-the-art low-latency models. Additionally, compared to SelfForcing, our method increases throughput without compromising quality.

Latency Profiling. We benchmark inference throughput for the entire generation pipeline using videos at 480×832 resolution. In Figure 7, we demonstrate the average throughput (TotalTotalGenerationFramesTime, FPS) calculated over the entire dataset. For a fair comparison, we evaluate all models using their official implementations on a single NVIDIA RTX A6000 GPU. LIPAR achieves the highest throughput among all real-time V2V models and is 1.45× faster than Self-forcing model.

##### 6.2 Comparison with Training-Free Pruning Methods

We compare our method against state-of-the-art training-free pruning methods, including ToMe [3], Importance-based Token Merging [27], and IDM [6]. We integrate these token merging algorithms into the Self-Forcing model with their official codes. Following ToMe [3], we restrict merging operations to the SelfAttention layers and immediately unmerge tokens before the Cross-Attention layers. We evaluate our method using Warp Error [14] and VBench [10], reusing 51 video-text pairs. We fix the pruning rates across all methods and compare them at three rates: 10%, 20%, and 32%. The 32% setting is selected to align with the configuration used in our model comparisons.

- Figure 8 qualitatively compares LIPAR against training-free pruning meth-

ods. The original output from the Self-Forcing model is leftmost. Video from LIPAR closely preserves the fidelity of the original model. In contrast, Importancebased Token Merging introduces noticeable artifacts, specifically small patches with inconsistent coloration. IDM and ToMe exhibit fewer patching artifacts but suffers from severe blurring on the frog’s body. Among all pruning methods, LIPAR is the only one that does not degrade visual quality.

In Table 1, we quantitatively evaluate the generated videos. LIPAR consistently outperforms all pruning methods across nearly all metrics. This performance gap becomes increasingly pronounced as the pruning rate increases and aligns with visual observations. See Appendix 17 for detailed discussions.

Self-Forcing LIPAR (Ours) Important IDM ToME

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

###### Fig. 8: Visual comparison of different pruning methods. LIPAR achieves superior visual quality compared with other token pruning methods. Prompt: “Animation style of a frog dancing and performing acrobatic side somersaults.“

Table 1: Quantitative comparison with other training-free pruning methods grouped by prune rate. Best results are highlighted in bold.

[Figure 73]

V-Bench Quality ↑ Method Prune Rate FPS ↑ Warp Error ↓ Subj. Backg. Motion Img. Qual. Original 0 8.4 75.0 0.921 0.941 0.988 0.678 Important 0.32 9.2 84.4 0.852 0.917 0.988 0.577 IDM 0.32 9.1 85.4 0.843 0.907 0.986 0.585 ToMe 0.32 9.1 85.7 0.856 0.915 0.987 0.622 LIPAR (Ours) 0.32 12.2 64.0 0.923 0.941 0.989 0.676 Important 0.20 8.5 79.4 0.887 0.924 0.988 0.633 IDM 0.20 8.4 81.2 0.876 0.917 0.987 0.629 ToMe 0.20 8.4 82.0 0.883 0.928 0.988 0.653 LIPAR (Ours) 0.20 10.9 67.1 0.921 0.940 0.989 0.676 Important 0.10 8.3 76.8 0.909 0.930 0.988 0.661 IDM 0.10 8.2 77.1 0.903 0.930 0.988 0.653 ToMe 0.10 8.2 78.2 0.903 0.930 0.988 0.668 LIPAR (Ours) 0.10 9.8 71.7 0.920 0.940 0.988 0.677

- (a)
- (b)

- (c)

[Figure 74]

[Figure 75]

Fig. 9: Attention Recovery. a) LIF b) + M-degree Apprx. c) +

#### 7 Ablation Study Noise-aware Dup.

##### 7.1 Generation Quality VS. Proposed Techniques

Figure 9 illustrates the effectiveness of the proposed Attention Recovery, where 33.8% of tokens are pruned in this example. The top image shows that direct pruning leads to a discrepancy between training and inference, resulting in noticeable artifacts highlighted by red rectangles. In the middle image, a partial recovery method utilizes an m-degree approximation, duplicating tokens from previous frames. This introduces noisy patterns due to the violation of the i.i.d. noise assumption from the diffusion model. Finally, the bottom image demonstrates that our complete Attention Recovery, combining the m-degree approximation with Noise-Aware Duplication, successfully preserves visual quality, resulting in clear, high-fidelity video generation.

##### 7.2 Latency vs. Remaining Tokens

We evaluate the relationship between inference latency and the percentage of remaining tokens. The experiment is conducted on an NVIDIA A6000 GPU using a video with a resolution of 480× 832 and 72 frames (4.5 seconds at 16 FPS). In Figure 10, we modulate the pruning rate by adjusting the threshold τ in the LIF pruning and measure the corresponding Latency. Each data point represents the average latency calculated over ten runs. Consistent with the discussion in Section 5.3, we observe a

Fig. 10: Inference latency on a NVIDIA A6000 GPU for generating a 4.5-second video across varying token remains.

strong linear correlation (Pearson r = 0.999) between the percentage of remaining tokens and latency. This empirical evidence verifies that even with Attention Recovery, LIPAR maintains a computational complexity of O(n), where n denotes the number of kept tokens. Furthermore, this linear relationship enables precise latency prediction before video editing, facilitating more efficient GPU resource allocation across concurrent tasks.

#### 8 Motion-Controlled Video Generation

To demonstrate the generalizability of LIPAR across tasks and model architectures, we extend the proposed method to Time-to-Move (TTM) [24]. In TTM, users manipulate a cropped image to generate a warped video sequence; the generative model then transforms warped video into a natural video that adheres to the motion trajectories. TTM is entirely training-free and uses the Wan 2.2 5B model with a bidirectional attention architecture [26]. We adhere to the TTM’s default settings and implement LIPAR on top of it.

We quantitatively evaluated generation quality using VBench and Warp Error with all TTM-provided examples, as summarized in Table 2. The results demonstrate that LIPAR maintains performance comparable to the baseline while achieving a 1.5× increase in inference throughput (FPS). Note that for this throughput calculation, we measure only the latency of the diffusion denoising process; please refer to Appendix 18 for visualizing the generated result.

Table 2: Quantitative comparison of performance and quality on TTM.

V-Bench Quality ↑ Method FPS ↑ Warp Error ↓ Subj. Backg. Motion Img. Qual.

TTM (5b) 0.58 63.6 0.956 0.957 0.991 0.652 LIPAR (Ours) 0.87 40.4 0.961 0.962 0.993 0.665

#### 9 Conclusion

In this paper, we identify and exploit a strong correlation that exists between temporal changes in pixel space and those in latent space. This suggests that unchanged pixels correspond to unchanged latents; hence, just as pixels need not be retransmitted in traditional video compression, the corresponding latents need not be recalculated in modern video generation pipelines, thereby bridging the gap between the two. Additionally, we formalize a general equation that pruning methods must satisfy to preserve generation quality. Finally, we propose a training-free approach, Latent Inter-frame Pruning with Attention Recovery (LIPAR), which achieves an average inference speedup of 1.45× while preserving high visual fidelity, outperforming existing training-free pruning methods. We regard this work as a foundational step toward integrating pixel-level video compression techniques with latent video generation.

#### References

- 1. Boer Bohan, O.: Taehv: Tiny autoencoder for hunyuan video. https://github. com/madebyollin/taehv (2025)
- 2. Bolya, D., Fu, C.Y., Dai, X., Zhang, P., Feichtenhofer, C., Hoffman, J.: Token merging: Your ViT but faster. In: International Conference on Learning Representations (2023)
- 3. Bolya, D., Hoffman, J.: Token merging for fast stable diffusion. CVPR Workshop on Efficient Deep Learning for Computer Vision (2023)
- 4. Choudhury, R., Zhu, G., Liu, S., Niinuma, K., Kitani, K., Jeni, L.: Don’t look twice: Faster video transformers with run-length tokenization. Advances in Neural Information Processing Systems (2024)
- 5. Dao, T., Fu, D.Y., Ermon, S., Rudra, A., Ré, C.: FlashAttention: Fast and memoryefficient exact attention with IO-awareness. In: Advances in Neural Information Processing Systems (NeurIPS) (2022)
- 6. Fang, H., Tang, S., Cao, J., Zhang, E., Tang, F., Lee, T.Y.: Attend to not attended: Structure-then-detail token merging for post-training dit acceleration. In: CVPR

(2025)

- 7. Feng, T., Li, Z., Yang, S., Xi, H., Li, M., Li, X., Zhang, L., Yang, K., Peng, K., Han, S., et al.: Streamdiffusionv2: A streaming system for dynamic and interactive video generation. arXiv preprint arXiv:2511.07399 (2025)
- 8. Geyer, M., Bar-Tal, O., Bagon, S., Dekel, T.: Tokenflow: Consistent diffusion features for consistent video editing. ICLR (2024)
- 9. Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video diffusion. In: Advances in Neural Information Processing Systems (2025)
- 10. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench: Comprehensive benchmark suite for video generative models. In: Computer Vision and Pattern Recognition (2024)
- 11. Kahatapitiya, K., Liu, H., He, S., Liu, D., Jia, M., Zhang, C., Ryoo, M.S., Xie, T.: Adaptive caching for faster video generation with diffusion transformers (2025), https://openreview.net/forum?id=DyyLUUVXJ5
- 12. Kodaira, A., Xu, C., Hazama, T., Yoshimoto, T., Ohno, K., et al.: Streamdiffusion: A pipeline-level solution for real-time interactive generation. arXiv (2023)
- 13. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024)
- 14. Lai, W.S., Huang, J.B., Wang, O., Shechtman, E., Yumer, E., Yang, M.H.: Learning blind video temporal consistency. In: European Conference on Computer Vision

(2018)

- 15. Le Gall, D.: Mpeg: a video compression standard for multimedia applications. Commun. ACM (1991)
- 16. Li, X., Ma, C., Yang, X., Yang, M.H.: Vidtome: Video token merging for zero-shot video editing. arXiv preprint arxiv:2312.10656 (2023)
- 17. Liang, F., Kodaira, A., Xu, C., Tomizuka, M., Keutzer, K., Marculescu, D.: Looking backward: Streaming video-to-video translation with feature banks. ICLR (2024)
- 18. Liu, F., Zhang, S., Wang, X., Wei, Y., Qiu, H., Zhao, Y., Zhang, Y., Ye, Q., Wan, F.: Timestep embedding tells: It’s time to cache for video diffusion model. arXiv preprint arXiv:2411.19108 (2024)

- 19. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: SDEdit: Guided image synthesis and editing with stochastic differential equations. In: International Conference on Learning Representations (2022)
- 20. Peebles, W., Xie, S.: Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748 (2022)
- 21. Pont-Tuset, J., Perazzi, F., Caelles, S., Arbeláez, P., Sorkine-Hornung, A., Van Gool, L.: The 2017 davis challenge on video object segmentation (2017)
- 22. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models (2021)
- 23. Shin, J., Li, Z., Zhang, R., Zhu, J.Y., Park, J., Shechtman, E., Huang, X.: Motionstream: Real-time video generation with interactive motion controls. arXiv preprint:2511.01266 (2025)
- 24. Singer, A., Rotstein, N., Mann, A., Kimmel, R., Litany, O.: Time-to-move: Training-free motion controlled video generation via dual-clock denoising. arXiv

(2025)

- 25. Su, J., Lu, Y., Pan, S., Wen, B., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding (2021)
- 26. Wan, T.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)
- 27. Wu, H., Xu, J., Le, H., Samaras, D.: Importance-based token merging for efficient image and video generation (2025)
- 28. Xi, H., Yang, S., Zhao, Y., Xu, C., Li, M., Li, X., Lin, Y., Cai, H., Zhang, J., Li, D., et al.: Sparse videogen: Accelerating video diffusion transformers with spatialtemporal sparsity. arXiv preprint arXiv:2502.01776 (2025)
- 29. Yang, S., Xi, H., Zhao, Y., Li, M., Zhang, J., Cai, H., Lin, Y., Li, X., Xu, C., Peng, K., et al.: Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. arXiv preprint arXiv:2505.18875 (2025)
- 30. Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T.: Improved distribution matching distillation for fast image synthesis. In: NeurIPS (2024)
- 31. Yin, T., Zhang, Q., Zhang, R., Freeman, W.T., Durand, F., Shechtman, E., Huang, X.: From slow bidirectional to fast autoregressive video diffusion models. In: Computer Vision and Pattern Recognition (2025)
- 32. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR (2018)
- 33. Zhang, Y., Wei, Y., Jiang, D., Zhang, X., Zuo, W., Tian, Q.: Controlvideo: Training-free controllable text-to-video generation. ICLR (2024)

#### 10 Related Work - Real-time Interactive Video Generation

Recent advancements in video generation aim to reduce latency, paving the way for real-time interactive video generation. We focus on two prominent tasks: Real-time Video Editing and Motion Control. Real-time video editing targets live applications, providing instantaneous edits based on user prompts, which replaces the need for sophisticated pre-made filters [7,12,17]. While approaches leveraging few-step image diffusion models achieve low latency on consumergrade GPUs [12,17], maintaining temporal consistency remains challenging. In contrast, [7] adapts video diffusion models for real-time editing, yet computational costs still hinder single-GPU performance.

Motion Control guides synthesis via explicit motion signals. Recently, MotionStream [23] and TTM [24] introduced techniques to generate motion-conditioned videos using warped static images. This enables intuitive interactions, such as dragging a dog’s head to turn [23, 24]. However, achieving real-time (30 FPS) response on a consumer-grade GPU remains challenging.

#### 11 Latents Compression Experiment

[Figure 76]

Fig. 11: LPIPS Score vs. θ. As we increase the threshold θ for compression in Eqn. 14, the compression rate (annotated in black) increases. Notably, high visual similarity (LPIPS ≤ 0.05, dashed line) is maintained even when the compression rate rises to 46%. This quantitatively confirms that substantial temporal redundancy exists in latent space.

There is no guarantee that the temporal redundancy exists in the latent space, despite the redundancy observed in the input video at pixel space. This is because the latent space is heavily compressed via the encoder, making it difficult to determine the real semantics of each latent patch. However, the existence

of redundancy is central to pruning methods. As a result, we must verify this property in advance.

To measure the temporal redundancy in latent space, we conduct an experiment where patches px,yt+1 are replaced (which we refer to as compressed) by their temporal predecessors (if sufficiently similar), and observe whether this influences the decoded results. Mathematically, we formulate this compression as:

pˆx,yt+1 =

px,yt if ∥px,yt+1 − px,yt ∥1 < θ px,yt+1 otherwise

(14)

Here, pˆx,yt+1 is the patch after compression, θ represents the threshold for judging similarity. To validate the fidelity of the compressed video, we require that the similarity between the compressed latents and the original decoded video exceeds a quality threshold τ, i.e., Sim(Dec(ˆp),Dec(p)) > τ, where Dec(·) denotes the decoder mapping latents to pixel space and Sim(·) is the similarity metric (we use LPIPS [32] in this experiment). Figure 2 shows an example where 44.5% of the latent patches are compressed, yet the decoded results are similar to the uncompressed video, showing that temporal redundancy indeed exists.

To further validate these results, we select ten input videos and increase the compression rate by gradually increasing θ, as shown in Figure 11. We observe that a high prevalence of temporal redundancy indeed exists, indicated by the fact that after compressing 46% of tokens, the decoded results still show high visual similarity (LPIPS ≤ 0.05, dashed line) compared with the original decoded video [32]. This confirms that substantial temporal redundancy exists in the latent space, and we can take advantage of this property.

#### 12 Deriving Target Objective

Our goal is to show Eq. 15 is true in the Transformer architecture. MSA(P(xt)) ≈ P(MSA(xt)) =⇒ D(P(xt)) ≈ P(D(xt)) (15)

where xt denotes the token sequence at time t, P represents the pruning operator, D is the denoising network, and R denotes the recovery operator (reusing temporal predecessors).

Since the denoising network D(·) is a Diffusion Transformer composed of stacked attention blocks, ensuring equivalence at all block is a sufficient condition for global approximation (see Eqn. 2):

Block(P(xt)) ≈ P(Block(xt)) (16)

Within a standard Transformer block, the Feed-Forward Network (FFN) and Cross-Attention layers operate point-wise on the video tokens (i.e., x′i = f(xi)). Since the output of a token in these layers depends only on itself, they are unaffected by pruning.

However, the Multi-Head Self-Attention (MSA) layer introduces inter-token dependency, where the calculation for a token xi depends on the entire sequence:

x′i = MSA(x0,x1,...,xN)i (17) Consequently, satisfying Eqn. 16 reduces to ensuring that the output of the self-attention layer remains the same under pruning:

MSA(P(xt)) ≈ P(MSA(xt)) (18)

#### 13 General Case for the Impact of I.I.D Noise

The quadratic noise term ϵTi Wϵj, where W = WQTWK, WQ is the weight matrix for query, WK is the weight matrix for keys, and ϵi and ϵj are the Gaussian Noise added to the tokens, changes distribution from Gaussian distribution N to Chi squared χ2 distribution:

ϵTi Wϵj ∼

N(0,∥W∥2F) if ϵi ̸= ϵj (independent) D m=1 λmχ21 if ϵi = ϵj (duplicated)

(19)

where λm are the eigenvalues of the symmetric part of W, defined as Wsym =

- 1

- 2(W +WT). Note that by the Central Limit Theorem, N(0,∥W∥2F) is an approximation for large token dimension D. The duplicated case introduces a positive

bias (E[ϵTi Wϵi] = Tr(W)) and higher variance (2Tr(Wsym2 )). Since Transformer projection matrices are typically learned such that W has a heavily positive trace (to ensure identical tokens attend to themselves), this bias is large and effectively inflates the attention weights on duplicated tokens.

#### 14 Latent Inter-Frame Pruning and Restoration Full Algorithms

Algorithm 1 Latent Inter-Frame Pruning

Algorithm 2 Latent Patch Restoration

- 1: Input: Latent Patch X, Keep Mask M
- 2: Output: Restored Latent Patch U
- 3: // Initialization
- 4: U ← ∅X.shape {Initialize empty tensor}
- 5: U[M] ← X
- 6: // Temporal Reconstruction Loop
- 7: for t = 0 to T − 1 do
- 8: if t = 0 then
- 9: Continue {First frame is always all true}
- 10: else
- 11: Pruned ← ¬Mt
- 12: Ut[Pruned] ← Ut−1[Pruned]
- 13: end if
- 14: end for
- 15: return U

- 1: Input: Video latent X = {X1, X2, . . . , XT}, Temporal Stride k, Thresholds τ1, τ2
- 2: Output: Keep Mask Mall
- 3:
- 4: Function GetDiffMask(A, B, τ):
- 5: D ← |A − B|
- 6: M ← 3D-GaussianAdaptiveThreshold(D, τ)
- 7: return M
- 8: Initialize Mall ← ∅
- 9: for t = 0 to T − 1 do
- 10: // Compute Short and Long-term Difference
- 11: if t = 0 then
- 12: Mshort ← all-true mask
- 13: else
- 14: Mshort ← GetDiffMask(Xt, Xt−1, τ1)
- 15: end if
- 16: if t ≤ k then
- 17: Mlong ← all-true mask
- 18: else
- 19: Mlong ← GetDiffMask(Xt, Xt−k, τ2)
- 20: end if
- 21: // Combine and Smooth
- 22: Mt ← Mshort ∨ Mlong
- 23: Mt ← 3D-MedianBlur(Mt)
- 24: Mt ← 2D-Morphology(Mt, Smooting)
- 25: Mt ← 3D-Dilation(Mt)
- 26: Append Mt to Mall
- 27: end for
- 28: return Mall

Latent Inter-Frame Pruning. Diffusion latent space contains temporal redundancy, which allows us to consider Inter-Latent Compression [4, 15] to bypass calculating repeated tokens. The core idea of Latent Inter-frame Pruning (LIF)

is to identify similar patches by comparing temporally consecutive patches with the same spatial location, as shown in Alg. 1, Line 5:

∥Xt − Xt+1∥1 < τ, (20)

Due to the high compression rate in the latent space, subtle movements in latent patches can yield small differences in Eqn. 20, causing mispruning. In the restoration stage, we reuse previous tokens. This may result in glitches that propagate when decoded, degrading video quality.

To mitigate this, we integrate motion detection by calculating the difference in frames in Eqn. 20 to avoid mispruning tokens with subtle movement. The idea is to recognize that videos typically involve object-level movements rather than isolated pixel changes. As a result, we leverage temporal and spatial information from neighboring tokens to identify object movement. Specifically, in Alg. 1, L6 and L22-24, we apply a 3D adaptive Gaussian threshold to account for neighboring differences and median blurring when computing frame changes, followed by a closing morphological operator to eliminate isolated pruned tokens. We further dilate the mask to provide a margin around boundary tokens exhibiting minimal changes.

Additionally, we enhance the binary (keep) mask, which is True for kept tokens and False for pruned tokens, by incorporating both short-term (consecutive) and long-term temporal differences, as shown in Line 21. This dual-term design is critical for supporting attention recovery and preventing the violation of the I.I.D. noise assumption.

#### 15 Experimental Settings

We implement our pruning method on top of the Self-Forcing model [9]. Consistent with CausVid [31] and StreamV2V [17], we employ SDEdit [19] for video-tovideo translation. By default, we use a 4-step denoising schedule with an initial noise level of t = 400 (out of 1000). We use a Tiny autoencoder for encoding and decoding [1]. The KV cache is trimmed for denoising and only preserves the most recent 6 frames due to the m-degree approximation.

The pruning thresholds τ1 and τ2 (from Eq. 5.2 and Eq. 13) are set to 0.15 and 0.3, respectively, resulting in an average of 32% tokens pruned. All experiments were conducted on an NVIDIA A6000 GPU with a fixed random seed of 0 for reproducibility.

Following the evaluation methods of TokenFlow [8] and StreamV2V [17], we assess performance on object-centric videos from the DAVIS 2017 train-val dataset [21]. This dataset covers diverse subjects (e.g., humans, animals, cars, etc.). The 51 video-prompt pairs used ranging from stylization to object swaps. We conduct a thorough comparison with state-of-the-art real-time (or low latency) V2V methods, including Self-Forcing [9], StreamV2V [17], StreamDiffusion [12], and ControlVideo [33], using their official implementations under default settings. To evaluate the performance, we rely on fourteen human participants to evaluate video generated. Furthermore, we benchmark our approach

against training-free pruning methods such as ToMe for SD [3], Importancebased Token Merging [27], and IDM [6]. In addition to qualitative observations, we perform quantitative evaluation by reporting Warp Error [14] and VBench scores [10] for video quality, and throughput to measure latency.

#### 16 Webpage for Human Evaluation Test

To validate the perceptual quality of our method, we conducted a user study comparing LIPAR against four baselines. Following TokenFlow [8] and StreamV2V [17], we use the DAVIS dataset [21] with 51 video-prompt pairs. We adopted a TwoAlternative Forced Choice (2AFC) protocol, where participants were presented with two videos side-by-side—one generated by our method and one by a baseline—and asked to select the better result, considering overall video quality (temporal consistency and frame quality), text-prompt alignment, and structural fidelity to the source video. The study involved 14 participants; each participant evaluated 25 randomly selected prompt pairs against all four baselines, resulting in 100 pairwise comparisons per participant. Figure 12 displays the webpage used for conducting the human evaluation test.

[Figure 77]

Fig. 12: Webpage for performing human evaluation test.

#### 17 Further Discussion on Qualitative Comparison with Other Pruning Methods

- 1. Throughput Difference: Despite using identical pruning rates, LIPAR achieves significantly higher throughput (FPS) than the baselines. This is primarily because token merging methods incur substantial overhead by executing merge operations at regular intervals for excessive tokens. In contrast, LIPAR computes the pruning mask only once with small overhead (≈10ms).

- Furthermore, while baseline token merging is restricted to the Self-Attention module (following [3]), our method applies pruning in an end-to-end manner across all layer components, maximizing acceleration.
- 2. Model Susceptibly to Token Merging: The Self-Forcing model’s causal attention mechanism is sensitive to token manipulation if positional encoding and noise correlations are not explicitly handled. Existing pruning methods did not address these factors, resulting in quality degradation. In contrast, we formulate conditions for preserving the pruned token value and handled these factors in LIPAR (see Section 5.3) to preserve quality.

#### 18 Time-To-Move Visualization

For pruning, we identify redundant tokens based on unchanged regions in the warped video and leverage the tokens from image condition in the WAN2.2 model for Attention Recovery. All experiments were conducted on a single NVIDIA RTX A6000 GPU with a fixed random seed of 0, using all motion control examples provided in [24].

In Figure 13, we observe a scenario in which the warped video directs the movement of an owl’s head while the background remains largely unchanged. This high degree of temporal redundancy in the background presents an ideal use case for LIPAR, with 47% of tokens pruned in this example. Visual inspection shows that the video generated with LIPAR results in realistic outputs that are similar to the baseline and faithfully adhere to the motion trajectories defined by the warped video.

#### 19 Limitations and Future Work

While LIPAR demonstrates strong performance on conditioned video generation tasks (video editing and warped-video generation), it still faces several limitations:

Dependence on Priors: LIPAR currently focuses on conditioned video generation because it relies on the source video to derive the pruning mask. However, the gradual refinement property of the diffusion denoising process makes it theoretically possible to adapt this approach for text-to-video (T2V) generation. Future work will explore extending this framework to T2V settings.

Noise Filtering in Bidirectional Models: Attention Recovery requires clean tokens to preserve the i.i.d. Gaussian noise assumption. While this is manageable in causal models utilizing a KV-cache, bidirectional architectures require auxiliary conditioning (e.g., a clean image condition) to function correctly. Future work could investigate noise filtering techniques to lift this constraint.

Optical Flow Integration: The design of LIPAR directly uses the previous frame at the same spatial location when computing temporal redundancy. Future work could incorporate optical flow estimation to compensate for the camera motion and achieve higher efficiency.

[Figure 78]

###### Fig. 13: Qualitative comparison on motion control tasks. We visualize the results of our LIPAR applied to motion control applications compared against baseline (original) methods.

