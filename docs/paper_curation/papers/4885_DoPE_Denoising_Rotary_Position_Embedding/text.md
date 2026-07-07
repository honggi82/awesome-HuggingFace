## DoPE: Denoising Rotary Position Embedding

Jing Xiong1*, Liyang Fan3*, Hui Shen2, Zunhai Su1, Min Yang3†, Lingpeng Kong1, and Ngai Wong1 1The University of Hong Kong 2University of Michigan, Ann Arbor 3Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences

Contact: junexiong@connect.hku.hk Project: https://The-physical-picture-of-LLMs.github.io

# arXiv:2511.09146v2[cs.CL]6Jan2026

### Abstract

Positional encoding is essential for large language models (LLMs) to represent sequence order, yet recent studies show that Rotary Position Embedding (RoPE) can induce massive activation. We investigate the source of these instabilities via a spectral analysis of RoPE, and show that its low-frequency components concentrate structured energy, producing low-rank, over-aligned attention patterns. We theoretically reveal that this low-frequency alignment manifests as activation noise, degrading stability during long-context extrapolation. To mitigate this effect, we introduce Denoising Rotary Position Embedding (DOPE), a training-free method that identifies and suppresses noisy attention heads using truncated matrix entropy, then reparameterizes their attention maps with an isotropic Gaussian distribution. Across a range of settings, DOPE improves length extrapolation performance without fine-tuning, increases robustness to perturbations, and boosts both needle-in-a-haystack and many-shot incontext learning tasks. These results suggest that selective positional encoding is key to robust extrapolation.

### 1 Introduction

Positional encoding is a core component of large language models (LLMs): it is added to query and key vectors to represent token order and shape interactions among tokens. Among many approaches (Press et al., 2021; Chen et al., 2023b; Su et al., 2024; Peng et al., 2023; Wang et al., 2021), Rotary Position Embedding (RoPE) (Su et al., 2024) is widely used because it encodes relative positions within dot-product attention and often extrapolates well to longer contexts. While RoPE provides an explicit mechanism for encoding token order, recent work has shown that causal attention itself (Gu et al., 2024; Köcher et al., 2025)

*Equal contribution †Corresponding author

implicitly captures positional relationships. Interestingly, this implicit encoding can lead to massive activations (Sun et al., 2024; Jin et al., 2025), a behavior closely tied to the attention sink phenomenon (Xiao et al., 2024). Yet how explicit positional encodings, especially RoPE, interact with this implicit positional bias and shape massive activations remains poorly understood (Jin et al., 2025; Wu et al., 2025).

Following these observations, recent studies have questioned the necessity of explicit positional encoding, proposing alternatives such as learnable feature maps applied directly to the attention map (Zheng et al., 2024, 2025) or even removing positional encoding entirely (NoPE) (Haviv et al., 2022; Wang et al., 2024; Ji et al., 2025). These results challenge the necessity of explicit positional encoding and suggest that causal attention may implicitly provide strong length extrapolation capability when paired with an appropriate feature map. However, the attention-sink puzzle remains: how the features induce the attention sink, and their underlying mechanism is still unclear. In this work, we investigate how RoPE injects massive activations across heads and introduces structured noise into the attention map, which manifests as the attention sink phenomenon.

We formalize this view by treating the attention map as a noisy feature map through the lens of truncated matrix entropy (Xiong et al., 2024). This perspective lets us detect heads dominated by massive activations and analyze how RoPE contributes to their emergence. We then suppress positional encoding selectively based on truncated matrix entropy and reparameterize the corresponding feature maps using an isotropic Gaussian distribution, improving stability in length extrapolation. Specifically, our main contributions are as follows:

• We propose DOPE, a training-free denoising scheme for RoPE that selectively suppresses positional encoding and theoretically reveal

how positional encoding shapes massive activation and attention sink

- • We introduce truncated matrix entropy to identify heads dominated by massive activations and reparameterize their attention maps with an isotropic Gaussian distribution.
- • We show that RoPE’s low-frequency alignment induces attention heads with long-range dependency capability, while extrapolative heads are intrinsically low-rank and benefit from preserved positional encoding.

### 2 Related Work

We review length extrapolation methods based on RoPE variants, as well as approaches that extrapolate without explicit positional encodings.

#### 2.1 Length Extrapolation with RoPE

RoPE (Su et al., 2024) is widely adopted because it encodes relative positions directly in dotproduct space and often exhibits strong extrapolation. RoPE and its variants are integrated into opensource LLM families, including LLaMA (Touvron et al., 2023; Dubey et al., 2024), Qwen (Team, 2024; Yang et al., 2025), Mistral (Jiang et al., 2023), and Gemma (Team et al., 2024, 2025).

However, when input sequences exceed the training length (Peng et al., 2023; Chen et al., 2023a; Ding et al., 2024), performance can degrade substantially. This limitation is not unique to RoPE; similar behavior is observed with other relative positional encodings such as ALiBi (Press et al., 2021) and Kerple (Chi et al., 2022).

Notably, several of these extensions modify RoPE at inference time without any training, e.g., by rescaling or interpolating the rotary frequencies. Prior work extends positional encodings in several ways, including interpolation-based (Li et al., 2023; Chen et al., 2023c) and NTK-based methods (Chen et al., 2023a; Peng et al., 2023; bloc97, 2023b,a; emoZilla, 2023), which adjust positional scaling or the frequency spectrum to enlarge the effective context. Another line of research (Chen et al., 2023a; Ding et al., 2024) adopts continuous formulations of positional encodings, modeling them as differential processes to support length extrapolation.

#### 2.2 Length Extrapolation without Explicit Positional Encoding

Positional encodings are often viewed as important for sequence awareness and model expressivity (Shaw et al., 2018; Yun et al., 2019; Luo et al.,

2022). Nevertheless, multiple studies (Haviv et al., 2022; Zuo et al., 2024; Köcher et al., 2025; Wu et al., 2024) suggest that causal attention can implicitly capture token order information. The No Positional Encoding (NoPE) approach (Kazemnejad et al., 2023) argues that the causal mask itself provides sufficient relative position cues, enabling position-aware behavior without explicit positional embeddings. Zuo et al. (2024) further show that such information can emerge through embedding similarity, while Wang et al. (2024) argue that these implicit cues may be insufficient for robust length generalization. This paper therefore examines when positional information should be applied selectively to improve extrapolation.

### 3 Denoising Rotary Position Embedding

In this section, we first review RoPE and NoPE. We then analyze how RoPE induces attention sink under the cone constraint, which motivates our use of a hybrid architecture combining NoPE and RoPE. Finally, we describe how to identify and remove the corresponding frequency components to mitigate anomalies in attention maps. This process is referred to as denoising.

#### 3.1 Preliminary

In this section, we briefly review the RoPE and NoPE method separately.

RoPE. Let the per-head width be dh. Split a head into dh/2 complex components by pairing dimensions (2f,2f+1). The frequency band index f ∈ {0,...,dh/2−1} enumerates the dh/2 twodimensional subspaces, each corresponding to a distinct rotation frequency. For an integer position i and a frequency schedule {θf}dfh=0/2−1 (with base b>1), define the per-band rotation phase θi,f and the corresponding 2 × 2 rotation matrix

R(θi,f) =

cosθi,f −sinθi,f sinθi,f cosθi,f

. (1)

The full rotation operator is then the blockdiagonal matrix

R(θi) = diag(R(θi,0),...,R(θi,dh/2−1)). (2)

A common choice of frequency schedule is θf = b−2f/dh. For any positions i,j, RoPE rotates queries and keys as

##### QRi = R(θi)Qi, KRj = R(θj)Kj. (3)

###### RoPE Denoised Head

Denoised NTK

###### Head Selection KforHead2

###### Denoising

K for Head 2

Mask NTK at

Low Frequency Indices

QforHead 1 QforHead 2 QforHead 3 … QforHead n

𝑐𝑜𝑠 𝜃0 sin 𝜃0 1 0

- Q for Head 2

- Q for Head 3

Mask dimension of frequencies i

- Q for Head 2

K for Head 3

- Q for Head 3

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

…

…

[Figure 1]

2𝜋

K for Head 3

𝜃 =

|sin 𝜃𝑖<br><br>|
|---|

Applied Dynamic NTK

|sin 𝜃𝑖<br><br>|
|---|

|𝑐𝑜𝑠 𝜃𝑖<br><br>|
|---|

training_length

𝑐𝑜𝑠 𝜃𝑖

Position Embedding

1 0 𝜃𝑖 < 𝜃 cos 𝜃𝑖 sin𝜃𝑖 𝜃𝑖 ≥ 𝜃

[Figure 2]

𝑔 𝜃𝑖 = ቊ

𝑔 𝜃𝑖 ~

- Head 2

- Head 3

…

RoPE Extrapolated Head

NTK

Low

- Q for Head 1

- Q for Head 2

- Q for Head 3

- Head 1

- Head 2

- Head 3

|[Figure 3]<br><br>DoPE-by-Gaussian<br><br>[Figure 4]|
|---|

Entropy

###### Unmasked Band

- K for Head 2

- Q for Head 2

K for Head 3

- Q for Head 3

- K for Head 2

- Q for Head 2

K for Head 3

- Q for Head 3

Masked Band

Truncated ERank

Entropy

cos𝜃𝑖 sin 𝜃𝑖

1 0

Keep NTK at

Rank

[Figure 5]

All Frequency Indices

###### high-frequency low-frequency

…

…

…

…

Head 1

…

[Figure 6]

High

freq ≥ θ freq < θ

Entropy

Q for Head n

Head n

DoPE-by-Parts

Head n

Figure 1: Visualization of DoPE. The blue dashed part illustrates how we select positional encodings for masking.

For Qi,f,Kj,f ∈ Rdh denote the two-dimensional components of Qi and Kj obtained by pairing dimensions (2f,2f+1),

dh/2−1

⟨QRi , KRj ⟩ =

⟨R(θi,f)Qi,f, R(θj,f)Kj,f⟩

f=0

dh/2−1

⟨Qi,f, R(θj,f − θi,f)Kj,f⟩,

=

f=0

(4)

so the attention logits depends on the relative positional offset (j−i) while preserving the efficiency of the dot product.

NoPE. In the NoPE method, positional encoding is entirely removed from the attention computation. Queries and keys are learned solely from token content without any explicit positional bias. Although this avoids the attention sinks introduced by RoPE, it undoubtedly requires training the model from scratch, which is computationally prohibitive. Moreover, when and how RoPE should be transformed into NoPE to prevent attention sinks remains theoretically unclear.

#### 3.2 Spectral Amplification of RoPE Bands

In this section, we present our theoretical contributions. We analyze how the massive activations of low-frequency RoPE bands arise through their band-wise Gram matrices, providing a theoretical framework for understanding this underlying “physical picture.”

Cone Constraint. Consider projected keys KRf = βf R(θf)Kf, βf ≥ βmin > 0, (5)

where R(θf) ∈ R2×2 is a rotation matrix by phase θf and βmin denote the minimum scaling factors associated with the query and key frequency bands.

The rotation matrix R(θf) is used to rotate the band-wise matrix Kf by an angle θf.

Following the cone condition of Deshpande et al. (2014), we define that within a low-frequency band, the RoPE rotations stay within a narrow angular cone. There exists a unit vector u and a half-angle γK <

π 2

such that

⟨u, R(θf)KRj,f⟩ ≥ ∥KRj,f∥cosγK,∀f ∈ {1,...,dh/2}, (6)

where operator ∥·∥ denotes the Euclidean norm and symbol ⟨·,·⟩ represents the dot product. Intuitively, this means that the phase rotations do not wrap around the circle within the visible context, so all projected queries and keys roughly align in the same direction.

Lemma 3.1 (Spectral Amplification). Under the above cone condition, the band-wise Gram matrix of a sequence with length N

Σj,f =

N

KRi,f (KRj,f)⊤ (7)

i=1

captures how the frequency band f aligns around key position j. Its top eigenvalue is bounded as

λmax(Σj,f) ≥ N βmin2 ∥KRj,f∥2cos2 γK, σ1(KRj,f) ≥ βmin ∥KRj,f∥

√

N cosγK, (8)

where λmax(·) is the largest eigenvalue, and σ1(·) is the dominant singular value.

Massive Activation. This lemma characterizes that some stable directions are reinforced. As the network depth increases, it leads to the accumulation of large ℓ2 norms and resulting in massive activation. An analogous result holds for QRf , with parameters (αmin,γQ).

Attention Sink. We further extend this result to the scenario where the key and query matrices are multiplied. Specifically, for the attention logits submatrix corresponding to frequency band f, we have the following expression:

QRf (KRj,f)⊤

, (9)

√

Aj,f =

d

where QRf and KRj,f are the query matrix and the key representation of the j-th token, respectively,

for frequency band f, and d is the dimensionality of the query and key vectors. The dominant singular value of the attention matrix for frequency band f satisfies the following inequality:

σ1(Aj,f) ≳ αmin√βdmin N ∥QRf ∥∥KRj,f∥cosγQ cosγK cosψ. (10) The parameters γQ and γK define the half-angles of the cones constraining the directions of the query and key vectors, respectively, while ψ quantifies the angular deviation between the principal directions of QRf and KRf , capturing the misalignment of their low-frequency orientations. Complete proofs and matrix inequalities are provided in Appendix A. This result formalize how low-frequency RoPE bands f contribute to attention sinks.

#### 3.3 Denoising via Truncated Matrix Entropy

Recent studies (Jin et al., 2025; Qiao and Huang, 2025) show that RoPE can induce outlier channels in query and key representations, where certain lowfrequency bands exhibit large ℓ2 norms. However, the ℓ2 norm captures only magnitude, missing the directional anomalies. In this section, we demonstrate how truncated matrix entropy (Xiong et al., 2024) can be used to capture the Spectral Amplification effect described in Lemma 3.1.

Truncated Matrix Entropy. Following Xiong et al. (2024), we define the truncated matrix entropy for attention head h as:

1 r

Hhr =

r

λi log λi, (11)

i=1

where λi are the i-th largest singular values of the Gram matrix Σh, and r denotes the number of singular values considered. This formulate captures the contribution of the top r singular values to the entropy of the parameter matrix, such as the key or query matrix, which allows us to assess the effective rank of the attention head.

Head Selection. We define two types of heads based on their entropy. Heads with low matrix truncated entropy are identified as denoised heads, while the others are treated as extrapolative heads that follow standard dynamic-NTK extrapolation (emoZilla, 2023). We selected the following heads as the denoised heads:

mh = 1[Hhr ≥ τ], (12)

where τ is a quantile threshold. Only heads with mh=0 (low-entropy spectra) undergo denoising.

DOPE-by-parts. Recall that under the cone condition, low-frequency RoPE bands correspond to small phase increments ψ, yielding a narrow angular spread. In practice, we approximate this lowfrequency region by a phase threshold θ = 2π/L, such that bands with θf ≤ ψ are considered to lie within the cone. For selected denoised heads, denoising acts per frequency band:

2π L

, (13) where L is the training length, and yielding

mh,f = 1[θf ≤ ψ], ψ =

dh/2

QRh,D =

mh,fQRh,f, (14)

f=1

dh/2

KRh,D =

mh,fKRh,f. (15)

f=1

This operation removes the corresponding lowfrequency components f from the query and key matrices of head h.

DOPE-by-all. We apply head-level positional encoding masking to the selected denoised heads:

##### KRh,D = mh KRh, QRh,D = mh QRh. (16)

DOPE-by-Gaussian. Alternatively, the positional encodings of denoised heads are fully masked and then replaced with:

KRh,D = (1 − mh)ϵK,hKRh, QRh,D = (1 − mh)ϵQ,hQRh,

(17)

where ϵK,h,ϵQ,h ∼ N(0,σ2I). This can be viewed as reparameterizing the attention map using a isotropic Gaussian distribution.

- Table 1: Summary of denoising configurations and results. Indicator specifies whether matrix entropy is computed

from Query or Key representations for head selection. Entropy Type is vanilla matrix entropy or Trunc-r (Hhr using the top-r singular values). # Heads is the number of selected heads. Criterion indicates when entropy is computed:

pre_ntk (before NTK scaling), ntk (after NTK positional encoding), or post_rope (after RoPE). Sort Order defines masking: ASC removes lower-entropy heads, while DESC removes higher-entropy heads. Results are reported at 24,756 (24k) and 65,536 (64k) tokens.

Method Indicator Entropy Type # Heads Criterion Sort Order Noisy (24k) Original (24k) Noisy (64k) Original (64k)

Dynamic NTK – – – – – 75.417 91.896 40.417 60.938 Dual Chunk Attention – – – – – 77.053 87.896 55.792 66.438 Positional Interpolatio – – – – – 14.583 26.417 9.479 11.771

DOPE-by-Gaussian Query Vanilla 5 post_ntk_query DESC 62.521 94.938 23.208 36.813 DOPE-by-Gaussian Key Trunc-32 3 post_ntk_key ASC 84.354 94.396 40.875 60.896 DOPE-by-Gaussian Key Trunc-16 5 pre_ntk_key ASC 77.417 93.708 40.604 60.313 DOPE-by-Gaussian Query Trunc-16 5 pre_ntk_query ASC 77.104 93.563 25.521 46.813 DOPE-by-Gaussian Key Trunc-16 3 pre_ntk_key ASC 77.438 93.125 41.271 60.021 DOPE-by-Gaussian Key Trunc-8 1 post_ntk_key DESC 75.250 92.229 45.667 64.042 DOPE-by-Gaussian Key Trunc-4 3 post_ntk_key DESC 65.833 89.354 45.375 61.979 DOPE-by-Gaussian Key Vanilla 2 post_ntk_key DESC 73.229 90.188 44.229 64.292 DOPE-by-Gaussian Query Trunc-1 5 post_ntk_query ASC 75.167 92.938 42.208 70.083 DOPE-by-Gaussian Query Trunc-1 3 post_ntk_query ASC 72.583 89.688 41.479 69.438 DOPE-by-Gaussian Query Vanilla 5 post_ntk_query ASC 44.833 76.188 44.042 65.854

DOPE-by-parts Key Trunc-32 30 post_rope_key ASC 76.229 93.063 40.312 60.375 DOPE-by-parts Query Trunc-32 25 post_ntk_query ASC 76.604 93.042 40.458 61.917 DOPE-by-parts Key Trunc-32 30 post_ntk_key ASC 76.458 92.875 40.771 61.333 DOPE-by-parts Key Trunc-32 20 post_ntk_key ASC 76.042 92.854 40.188 60.625 DOPE-by-parts Key Trunc-32 25 post_ntk_key ASC 76.104 92.771 40.021 61.083 DOPE-by-parts Query Trunc-16 2 post_ntk_query DESC 75.438 92.354 42.729 60.729

- DOPE-by-parts Query Trunc-8 2 post_ntk_query DESC 75.229 91.771 42.521 61.104
- DOPE-by-parts Query Trunc-8 3 post_ntk_query DESC 75.271 92.146 42.438 59.583 DOPE-by-parts Query Trunc-32 3 post_rope_query ASC 74.500 92.125 40.313 62.208 DOPE-by-parts Query Vanilla 3 post_rope_query ASC 74.125 92.479 40.125 62.146 DOPE-by-parts Query Trunc-32 5 post_ntk_query DESC 75.438 91.958 40.938 62.125

DOPE-by-all Key Trunc-32 3 post_ntk_key ASC 81.958 93.833 40.917 61.271 DOPE-by-all Key Trunc-16 3 post_rope_key DESC 65.958 93.771 35.354 61.063 DOPE-by-all Key Trunc-16 3 pre_ntk_key ASC 76.583 93.729 41.354 57.833 DOPE-by-all Key Vanilla 3 post_ntk_key DESC 75.625 93.271 39.729 58.021 DOPE-by-all Query Vanilla 3 pre_ntk_query ASC 73.542 93.250 39.333 63.146 DOPE-by-all Key Trunc-8 1 post_ntk_key DESC 74.917 92.000 46.000 63.625 DOPE-by-all Key Trunc-4 3 post_ntk_key DESC 65.958 89.813 45.292 62.646

- DOPE-by-all Query Trunc-1 2 post_ntk_query DESC 75.104 92.354 44.292 64.146 DOPE-by-all Query Trunc-1 5 post_ntk_query ASC 75.000 92.917 42.729 70.083
- DOPE-by-all Query Trunc-1 3 post_ntk_query ASC 73.104 90.063 41.646 69.708 DOPE-by-all Query Trunc-1 3 post_rope_query DESC 46.771 87.521 27.000 69.104

- 4 Experiment

we evaluate standard MICL and its NIH variant at 8K and 16K context lengths. Data are sampled from the MATH dataset (Hendrycks et al., 2021).

- 4.1 Experimental Setup The “needle-in-a-haystack” (NIH) synthesis task benchmarks long-context retrieval by placing a sparse “needle” at different depths and measuring recall. It also allows controlled noise injection (e.g., special tokens). We evaluate two conditions: original setups and noisy setups.

Hyperparameters of Head Selection. We select 1–32 heads based on either ascending or descending scores. We use calibration data matched in length to the test data to precompute matrix entropy. Entropy can be computed at three stages of the forward pass, each isolating a different positionaleffect factor: (1) pre-NTK, on projected query/key representations before positional encoding (no PE); (2) post-NTK, after Dynamic-NTK scaling of the RoPE base frequency (frequency-scaling effect); and (3) post-RoPE, after applying the RoPE rotation (full PE effect).

Original Setups. We insert the needle at various positions under context lengths of 24K and 64K tokens to measure retrieval performance and the lost-in-the-middle effect.

Noisy Setups. Under the same context lengths, we insert attention-sink tokens (e.g., a start-ofsequence symbol) near the needle to test robustness under controlled perturbations and relate performance to attention sinks and matrix entropy.

Entropy can be computed on query or key representations; in practice, the query matrix (Tang et al., 2024) better captures head characteristics. This yields six criteria (3 stages × 2 components),

Many-shot In-context Learning. For many-shot in-context learning (MICL) (Agarwal et al., 2024),

- Table 2: Summary of denoising configurations and results on Qwen2.5-Math-7B for Many-Shot In-Context Learning extrapolation (4K→16K). We evaluate (1) Needle Insertion, where the problem is inserted into the ICL haystack at one of four depths (beginning, 1/3, 2/3, end), and (2) Skip Needle, a no-insertion baseline. Indicator specifies whether denoising is applied to Query or Key representations. Entropy Type is Vanilla (matrix entropy Hh) or

Trunc-r (Hhr with threshold r). # Heads is the number of selected heads. Criterion indicates when entropy is computed: pre_ntk, ntk, or post_rope. Sort Order specifies the selection direction: DESC (highest entropy) or

ASC (lowest entropy). Results report accuracy on 100 sampled MATH problems (400 total configurations across insertion positions).

Method Indicator Entropy Type # Heads Criterion Sort Order Needle Insert (8K) Skip Needle (8K) Needle Insert (16K) Skip Needle (16K)

Zero-shot Baseline – – – – – 0.430 0.430 0.430 0.430 Many-shot Baseline – – – – – 0.373 0.370 0.240 0.230 Dual Chunk Attention – – – – – 0 0.01 0 0 Positional Interpolation – – – – – 0 0.01 0 0

- DOPE-by-Gaussian Query Trunc-1 1 post_ntk_query ASC 0.393 0.410 0.228 0.250 DOPE-by-Gaussian Query Trunc-16 1 post_ntk_query ASC 0.380 0.360 0.225 0.250 DOPE-by-Gaussian Query Trunc-1 3 post_ntk_query ASC 0.375 0.370 0.238 0.220 DOPE-by-Gaussian Query Trunc-4 5 post_ntk_query ASC 0.375 0.440 0.225 0.190 DOPE-by-Gaussian Query Trunc-1 5 post_ntk_query ASC 0.318 0.440 0.238 0.220 DOPE-by-Gaussian Query Trunc-4 3 post_ntk_query ASC 0.358 0.430 0.223 0.210
- DOPE-by-Gaussian Query Trunc-1 2 post_ntk_query ASC 0.345 0.380 0.258 0.240 DOPE-by-Gaussian Query Full 1 post_ntk_query DESC 0.388 0.400 0.258 0.230 DOPE-by-Gaussian Query Full 3 post_ntk_query DESC 0.370 0.340 0.255 0.270 DOPE-by-Gaussian Query Trunc-16 3 post_ntk_query ASC 0.355 0.420 0.248 0.260

DOPE-by-parts Query Trunc-1 1 post_ntk_query ASC 0.388 0.410 0.230 0.250 DOPE-by-parts Query Trunc-16 2 post_ntk_query ASC 0.380 0.330 0.245 0.260 DOPE-by-parts Query Trunc-4 5 post_ntk_query ASC 0.368 0.390 0.220 0.260 DOPE-by-parts Query Trunc-4 3 post_ntk_query ASC 0.360 0.420 0.240 0.230 DOPE-by-parts Query Trunc-8 3 post_ntk_query ASC 0.363 0.390 0.220 0.180 DOPE-by-parts Query Trunc-1 5 post_ntk_query ASC 0.355 0.350 0.245 0.240 DOPE-by-parts Query Trunc-16 5 post_ntk_query ASC 0.365 0.380 0.243 0.260

- DOPE-by-parts Query Full 1 post_ntk_query DESC 0.375 0.350 0.245 0.240
- DOPE-by-parts Query Full 2 post_ntk_query DESC 0.400 0.380 0.258 0.230
- DOPE-by-parts Query Full 3 post_ntk_query DESC 0.388 0.390 0.258 0.250

- DOPE-by-all Query Trunc-1 1 post_ntk_query ASC 0.395 0.430 0.235 0.240 DOPE-by-all Query Trunc-4 2 post_ntk_query ASC 0.383 0.390 0.215 0.240

- DOPE-by-all Query Trunc-8 2 post_ntk_query ASC 0.383 0.390 0.225 0.220 DOPE-by-all Query Trunc-1 5 post_ntk_query ASC 0.338 0.480 0.243 0.220 DOPE-by-all Query Trunc-1 3 post_ntk_query ASC 0.353 0.440 0.258 0.210 DOPE-by-all Query Trunc-4 5 post_ntk_query ASC 0.375 0.440 0.220 0.200
- DOPE-by-all Query Trunc-8 3 post_ntk_query ASC 0.375 0.440 0.205 0.190 DOPE-by-all Query Trunc-16 5 post_ntk_query ASC 0.360 0.360 0.263 0.240 DOPE-by-all Query Full 3 post_ntk_query DESC 0.393 0.350 0.258 0.210

- DOPE-by-all Query Trunc-1 2 post_ntk_query ASC 0.363 0.380 0.243 0.250 DOPE-by-all Query Trunc-16 1 post_ntk_query ASC 0.365 0.370 0.228 0.250 DOPE-by-all Query Trunc-16 3 post_ntk_query ASC 0.353 0.340 0.253 0.240

- Table 3: Ablation study: Performance on 64k extrapolation using attention heads selected at different sequence lengths. Each configuration uses heads identified from sequences of length 24k, 32k, 48k, 56k, and 64k, then evaluates on the 64k task under both Noisy and Original conditions.

24k Heads 32k Heads 48k Heads 56k Heads 64k Heads Method Indicator Entropy Type # Heads Criterion Sort Order Noisy Original Noisy Original Noisy Original Noisy Original Noisy Original

Dynamic NTK – – – – – 40.417 60.938 40.417 60.938 40.417 60.938 40.417 60.938 40.417 60.938 DOPE-by-Gaussian Key Trunc-8 1 post_ntk_key DESC 40.896 62.438 40.417 63.125 28.666 60.104 28.666 60.104 45.667 64.042 DOPE-by-Gaussian Query Trunc-1 5 post_ntk_query ASC 35.667 56.708 30.354 61.979 43.604 69.166 38.020 69.854 42.208 70.083 DOPE-by-parts Query Trunc-16 2 post_ntk_query DESC 41.792 61.271 41.479 65.020 41.479 65.020 41.479 65.020 42.729 60.729 DOPE-by-parts Query Trunc-32 3 post_rope_query ASC 40.313 61.688 39.333 66.979 39.333 66.979 39.333 66.979 40.313 62.208 DOPE-by-all Key Trunc-8 1 post_ntk_key DESC 40.625 62.208 40.541 65.229 29.604 59.979 29.604 59.979 46.000 63.625 DOPE-by-all Query Trunc-1 5 post_ntk_query ASC 37.063 61.292 32.687 65.000 43.000 75.187 40.458 73.812 42.729 70.083

plus an option to compute entropy jointly on query and key. We denote configurations as Criterion; for example, post_ntk_query computes entropy on query representations after NTK scaling.

#### 4.2 Main results

We conducted experiments under two settings: original setups and noisy setups, with the results summarized in Table 1. Our findings are summarized as follows: (i) The model exhibits a sharp performance degradation after introducing attention sink tokens. (ii) Under the shorter context setting (24k tokens), DOPE-by-Gaussian achieves its best performance, improving from the 75.417 baseline to 84.354. The inclusion of a Gaussian distribu-

tion generally promotes isotropy in representations, which usually increases the discriminability of token representations in the denoised head, allowing the model to focus on a few important tokens. (iii) Truncated matrix entropy and (vallina) matrix entropy exhibit distinctly different patterns. For the truncated variant, we sort values in descending order and prune the low-entropy heads; for the matrix entropy, we sort in ascending order and prune the high-entropy heads. Both strategies perform well, but truncated matrix entropy typically achieves better results. (iv) In extremely sparse regimes—for example, with a 64K context length—using the truncated matrix entropy with r = 1 (which can be regarded as equivalent to the spectral norm, i.e.,

High Entropy Head (Average) Key: layer_4_head_12

High Entropy Head (Last Token) Key: layer_4_head_12

High Entropy Head (Average) Key: layer_5_head_11

High Entropy Head (Last Token) Key: layer_5_head_11

Average Attention

Last Token Attention

Average Attention

Last Token Attention

0.012

0.025

0.10

Needle Position

Needle Position

Needle Position

Needle Position

0.025

0.010

0.020

0.08

0.020

0.008

AverageAttention

AverageAttention

AttentionWeight

AttentionWeight

0.015

0.06

0.015

0.006

0.010

0.04

0.010

0.004

0.005

0.02

0.005

0.002

0.000

0.00

0.000

0.000

0 5000 10000 15000 20000 25000 Token Position

0 5000 10000 15000 20000 25000 Token Position

0 5000 10000 15000 20000 25000 Token Position

0 5000 10000 15000 20000 25000 Token Position

Low Entropy Head (Average) Key: layer_5_head_11

Low Entropy Head (Last Token) Key: layer_5_head_11

Low Entropy Head (Average) Key: layer_1_head_2

Low Entropy Head (Last Token) Key: layer_1_head_2

Average Attention

Last Token Attention

Average Attention

Last Token Attention

0.025

Needle Position

Needle Position

Needle Position

Needle Position

0.0020

0.025

0.0020

0.020

0.020

0.0015

0.0015

AverageAttention

AverageAttention

AttentionWeight

AttentionWeight

0.015

0.015

0.0010

0.0010

0.010

0.010

0.0005

0.0005

0.005

0.005

0.000

0.000

0.0000

0.0000

0 5000 10000 15000 20000 25000 Token Position

0 5000 10000 15000 20000 25000 Token Position

0 5000 10000 15000 20000 25000 Token Position

0 5000 10000 15000 20000 25000 Token Position

(a) DoPE by Vanilla Matrix Entropy.

(b) DoPE by Truncated Matrix Entropy.

Figure 2: Comparison of attention distribution across all heads and top-16 heads.

σmax(Σh)) yields the best results. This indicates that the sparser the setting, the sharper the singular value distribution becomes. (v) As shown in Table 3, cross-length calibration generally performs worse than same-length calibration.

#### 4.3 Many-shot In-Context Learning

We reported the model’s performance under manyshot in-context learning (MICL) (Agarwal et al., 2024) in Table 2. We evaluated two settings: inserting the test exemplar into the in-context exemplars (a NIH variant) and omitting the test exemplar (standard in-context learning). Beyond retrieval, this task also probed whether the model extracted reusable reasoning patterns from extended contexts rather than relying on shallow heuristics.

The Curse of Length. We observed that MICL improved reasoning at appropriate lengths, but performance dropped markedly when the context window extended to 16K. Adding more exemplars did not yield further gains, suggesting that learning in ultra-long contexts remained challenging.

The Curse of Shortcut. When we inserted exemplars of the test samples into the in-context examples, we unexpectedly observed a substantial performance drop at 24K and 64K. Instead of copying the correct answers in a “needle-in-a-haystack” manner, the model appeared to fall back on shortcuts that hurt overall accuracy.

Baseline Failure. The two training-free length extrapolation baselines, Dual Chunk Attention and Positional Interpolation, shown in Table 2, achieve accuracies close to zero, demonstrating astonishingly poor performance.

Cross-task Generalization. To evaluate crosstask generalization, we compare head selection using entropy computed on either the MATH dataset or the NIH dataset (Table 4), and finally test under

the MICL task. We find that sparse patterns estimated from synthetic tasks transfer to more complex reasoning tasks, indicating that even without calibration data for the target task, synthetic data still provide useful estimates of the sparse patterns.

Table 4: Ablation on attention-head identification. We select denoising heads using MATH vs. NIH calibration data and evaluate MICL on Qwen2.5-Math-7B at 8K context length. Head selection uses Query representations with the post-NTK criterion. Results report MATH accuracy for Needle Insertion and Skip Needle.

Method Entropy Type # Heads Selection Dataset Needle Insert (8K) Skip Needle (8K)

Zero-shot Baseline – – – 0.430 0.430 Many-shot Baseline – – – 0.240 0.230

Heads selected using MATH dataset

DOPE-by-Gaussian Trunc-4 5 MATH 0.375 0.440 DOPE-by-Gaussian Trunc-1 1 MATH 0.393 0.410 DOPE-by-parts Trunc-4 3 MATH 0.360 0.420 DOPE-by-parts Trunc-1 1 MATH 0.388 0.410 DOPE-by-all Trunc-1 5 MATH 0.338 0.480

- DOPE-by-all Trunc-1 1 MATH 0.395 0.430 Heads selected using NIH dataset

DOPE-by-Gaussian Full 3 NIH 0.365 0.390 DOPE-by-Gaussian Trunc-16 1 NIH 0.375 0.410 DOPE-by-parts Trunc-16 2 NIH 0.372 0.330 DOPE-by-parts Full 2 NIH 0.350 0.390 DOPE-by-all Trunc-16 5 NIH 0.360 0.420

- DOPE-by-all Trunc-1 2 NIH 0.417 0.390

- 4.4 Matrix Entropy Meets Attention Sink To connect our entropy criterion to attention behavior, we visualized attention distributions for heads identified by high truncated matrix entropy. As shown in Fig. 2, truncated matrix entropy aligned closely with the attention sink phenomenon.

In Fig. 2b, we observed that when truncated matrix entropy identified low-entropy heads, these heads often produced attention sinks (recency bias), while the remaining high-entropy heads allocated attention to the needle. In contrast, Fig. 2a showed that high vanilla matrix entropy corresponded to severe attention sinks: although the low-entropy heads displayed more regular attention patterns, they still failed to attend to the needle position.

- 4.5 RoPE Induces Low-rankness We then examined how these entropy-based selections related to representation structure by visualiz-

###### Cosine Similarity of High-Entropy Head L5H11 Near Question Tokens Needle Inserted at 0% Context Depth

0

| |[Figure 7]<br><br>Needle Start/End| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

5

10

TokenPosition(Absolute)

15

20

25

30

0 20 40 60 80 100 120 Cosine Dimension (0-127)

1.00

|[Figure 8]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

0.75

0.50

0.25

CosineSimilarity

0.00

0.25

0.50

0.75

1.00

Figure 3: High matrix entropy head (Layer 5, Head 11)

###### Cosine Similarity of Low-Entropy Head L1H2 Near Question Tokens Needle Inserted at 0% Context Depth

0

| |[Figure 9]<br><br>Needle Start/End| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

5

10

TokenPosition(Absolute)

15

20

25

30

0 20 40 60 80 100 120 Cosine Dimension (0-127)

1.00

|[Figure 10]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

0.75

0.50

0.25

CosineSimilarity

0.00

0.25

0.50

0.75

1.00

Figure 4: Low matrix entropy head (Layer 1, Head 2)

0

| |[Figure 11]<br><br>Needle Start/End| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

- 5

10

15

20

25

30

TokenPosition(Absolute)

Cosine Similarity of High-Entropy Head L4H12 Near Question Tokens Needle Inserted at 0% Context Depth

|[Figure 12]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

1.00

0.75

0.50

0.25

0.00

0.25

0.50

0.75

1.00

CosineSimilarity

Figure 5: High truncated matrix entropy head (Layer 4, Head 12)

| |[Figure 13]<br><br>Needle Start/End| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 20 40 60 80 100 120 Cosine Dimension (0-127)

0

5

10

15

20

25

30

TokenPosition(Absolute)

Cosine Similarity of Low-Entropy Head L5H11 Near Question Tokens Needle Inserted at 0% Context Depth

|[Figure 14]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

1.00

0.75

0.50

0.25

0.00

0.25

0.50

0.75

1.00

CosineSimilarity

Figure 6: Low truncated matrix entropy head (Layer 5, Head 11)

ing the token matrix and its corresponding eigenvectors for the heads identified by Vanilla Matrix Entropy and Truncated Matrix Entropy. Fig. 3 and

- 6 showed the cosine similarity between token query vectors (Y-axis, token position) and eigenvectors spanning the full 128-dimensional space (X-axis). This projection onto a k = 128 basis revealed the effective dimensionality used by each head, and highlighted that the two metrics selected different heads with distinct low-rank structure.

0 20 40 60 80 100 120 Cosine Dimension (0-127)

Low-rankness. We observed clear low-rank structure in heads selected by both entropy metrics. Fig. 4 and Fig. 5 illustrated heads retained due to low Vanilla Matrix Entropy and high Truncated Matrix Entropy, respectively. Visually, the similarity mass concentrated in the first few dimensions, indicating that these “low-rank” heads relied on only a small subspace to support extrapolation.

Periodicity. We also observed that the head selected by truncated matrix entropy exhibited clear periodicity along the sequence dimension (Fig. 5), compared to the head selected by matrix entropy in Fig. 4. This difference helped explain why truncated matrix entropy better identified extrapolative

heads: it captured periodic low-rank structure that vanilla matrix entropy did not reliably distinguish.

### 5 Conclusion

In this paper, we examined how positional encoding shapes long-context behavior in LLMs, with an emphasis on the emergence of massive activation and the attention sink phenomenon. A key takeaway is that RoPE is not merely a benign carrier of relative position: its low-frequency components can also act as a structured amplifier, concentrating energy and promoting over-aligned, lowrank attention patterns that undermine stability as context grows. DOPE follows directly from this interpretation: rather than modifying RoPE globally or removing positional encoding altogether, it treats instability as a head-specific effect and intervenes only where the attention map becomes noise-dominated. The resulting gains under perturbation suggest that long-context robustness is less about choosing a single positional encoding scheme and more about controlling how positional information is injected across heads.

### Limitations

DOPE has practical limitations. Head selection adds computation, and the approach assumes that entropy-based criteria reliably separate noisedominated heads, which may not hold across all models, layers, or domains. Moreover, our evaluation focuses on long-context inference tasks, so generalization to broader settings remains to be validated.

### References

Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, and 1 others. 2024. Many-shot in-context learning. Advances in Neural Information Processing Systems, 37:76930– 76966.

Chenxin An, Fei Huang, Jun Zhang, Shansan Gong, Xipeng Qiu, Chang Zhou, and Lingpeng Kong. 2024. Training-free long-context scaling of large language models. arXiv preprint arXiv:2402.17463.

- bloc97. 2023a. Add NTK-Aware interpolation "by parts" correction. https://github.com/ jquesnelle/scaled-rope/pull/1. Accessed: 2023-07-01.
- bloc97. 2023b. NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation. https://www.reddit.com/ r/LocalLLaMA/comments/14lz7j5/ntkaware_ scaled_rope_allows_llama_models_to_have/. Accessed: 2023-07-01.

Guanzheng Chen, Xin Li, Zaiqiao Meng, Shangsong Liang, and Lidong Bing. 2023a. CLEX: Continuous length extrapolation for large language models. In International Conference on Learning Representations.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023b. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023c. Extending context window of large language models via positional interpolation. https://arxiv.org/abs/2306.15595. Accessed: 2025-12-23.

Ta-Chung Chi, Ting-Han Fan, Peter J Ramadge, and Alexander Rudnicky. 2022. KERPLE: Kernelized relative positional embedding for length extrapolation. Advances in Neural Information Processing Systems, 35:8386–8399.

Yash Deshpande, Andrea Montanari, and Emile Richard. 2014. Cone-constrained principal component analysis. Advances in Neural Information Processing Systems, 27.

Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. 2024. LongRoPE: Extending llm context window beyond 2 million tokens. arXiv preprint arXiv:2402.13753.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and 1 others. 2024. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407.

emoZilla. 2023. Dynamically scaled rope further increases performance of long context llama with zero fine-tuning. https://www.reddit.com/r/ LocalLLaMA/comments/14mrgpr/dynamically_ scaled_rope_further_increases/. Reddit post.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, ..., and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. 2024. When attention sink emerges in language models: An empirical view. arXiv preprint arXiv:2410.10781.

Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, and Omer Levy. 2022. Transformer language models without positional encodings still learn positional information. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 1382–1390.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. In Proceedings of the Thirty-Fifth Conference on Neural Information Processing Systems (NeurIPS) Datasets Benchmarks Track.

Tao Ji, Bin Guo, Yuanbin Wu, Qipeng Guo, Lixing Shen, Zhan Chen, Xipeng Qiu, Qi Zhang, and Tao Gui. 2025. Towards economical inference: Enabling deepseek’s multi-head latent attention in any transformer-based llms. arXiv preprint arXiv:2502.14837.

Albert Q. Jiang, Alexandre Sablayrolles, and 1 others. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Mingyu Jin, Kai Mei, Wujiang Xu, Mingjie Sun, Ruixiang Tang, Mengnan Du, Zirui Liu, and Yongfeng Zhang. 2025. Massive values in self-attention modules are the key to contextual knowledge understanding. arXiv preprint arXiv:2502.01563.

Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. 2023. The impact of positional encoding on length generalization in transformers. Advances in Neural Information Processing Systems, 36:24892–24928.

Chris Köcher, Alexander Kozachinskiy, Anthony Widjaja Lin, Marco Sälzer, and Georg Zetzsche. 2025. Nope: The counting power of transformers with no positional encodings. arXiv preprint arXiv:2505.11199.

Shanda Li, Chong You, Guru Guruganesh, Joshua Ainslie, Santiago Ontanon, Manzil Zaheer, Sumit Sanghai, Yiming Yang, Sanjiv Kumar, and Srinadh Bhojanapalli. 2023. Functional interpolation for relative positions improves long context transformers. In International Conference on Learning Representations.

Shengjie Luo, Shanda Li, Shuxin Zheng, Tie-Yan Liu, Liwei Wang, and Di He. 2022. Your transformer may not be as powerful as you expect. Advances in Neural Information Processing Systems, 35:4301–4315.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2023. YaRN: Efficient context window extension of large language models. In International Conference on Learning Representations.

Ofir Press, Noah Smith, and Mike Lewis. 2021. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations.

Ye Qiao and Sitao Huang. 2025. Q-roar: Outlier-aware rescaling for rope position interpolation in quantized long-context llms. arXiv preprint arXiv:2509.14391.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. 2024. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37:68658–68685.

Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. 2018. Self-attention with relative position representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 464–468.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.

Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. 2024. Massive activations in large language models. arXiv preprint arXiv:2402.17762.

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. 2024. Quest: Queryaware sparsity for efficient long-context llm inference. arXiv preprint arXiv:2406.10774.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, and 1 others. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, and 1 others. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.

Qwen Team. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, and 1 others. 2023. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Jie Wang, Tao Ji, Yuanbin Wu, Hang Yan, Tao Gui, Qi Zhang, Xuanjing Huang, and Xiaoling Wang. 2024. Length generalization of causal transformers without position encoding. arXiv preprint arXiv:2404.12224.

Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li, and Jian Tang. 2021. Kepler: A unified model for knowledge embedding and pre-trained language representation. Transactions of the Association for Computational Linguistics, 9:176–194.

Xinyi Wu, Amir Ajorlou, Yifei Wang, Stefanie Jegelka, and Ali Jadbabaie. 2024. On the role of attention masks and layernorm in transformers. Advances in Neural Information Processing Systems, 37:14774– 14809.

Xinyi Wu, Yifei Wang, Stefanie Jegelka, and Ali Jadbabaie. 2025. On the emergence of position bias in transformers. arXiv preprint arXiv:2502.01951.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient streaming language models with attention sinks. In International Conference on Learning Representations.

Jing Xiong, Jianghan Shen, Fanghua Ye, Chaofan Tao, Zhongwei Wan, Jianqiao Lu, Xun Wu, Chuanyang Zheng, Zhijiang Guo, Lingpeng Kong, and 1 others. 2024. Uncomp: Uncertainty-aware long-context compressor for efficient large language model inference. arXiv preprint arXiv:2410.03090.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. 2024. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J Reddi, and Sanjiv Kumar. 2019. Are transformers universal approximators of sequence-to-sequence functions? arXiv preprint arXiv:1912.10077.

Chuanyang Zheng, Yihang Gao, Han Shi, Minbin Huang, Jingyao Li, Jing Xiong, Xiaozhe Ren, Michael Ng, Xin Jiang, Zhenguo Li, and 1 others. 2024. Dape: Data-adaptive positional encoding for length extrapolation. Advances in Neural Information Processing Systems.

Chuanyang Zheng, Yihang Gao, Han Shi, Jing Xiong, Jiankai Sun, Jingyao Li, Minbin Huang, Xiaozhe Ren, Michael Ng, Xin Jiang, and 1 others. 2025. Dape v2: Process attention score as feature map for length extrapolation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10628– 10666.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody_Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, and 1 others. 2023. Efficiently programming large language models using sglang.

Chunsheng Zuo, Pavel Guerzhoy, and Michael Guerzhoy. 2024. Position information emerges in causal transformers without positional encodings via similarity of nearby embeddings. arXiv preprint arXiv:2501.00073.

### A Theoretical Analysis of Spectral Amplification

#### A.1 Proofs

Lemma A.1 (Entry-level lower bound (rectangular)). Let Σ ∈ Rm×n with largest singular value σ1(Σ). Then

|(Σ)ij| ≥

max

i,j

σ1(Σ) √mn

. (18)

Here i∈{1,...,m} and j ∈{1,...,n} index the rows and columns of Σ, respectively. Proof. By the Frobenius/spectral norm relation,

m

n

∥Σ∥2F =

(Σij)2 ≤ ( max

|(Σij)|)2 mn,

i,j

i=1

j=1

∥Σ∥F √mn ≥

σ1(Σ) √mn

⇒ max

|(Σ)ij| ≥

,

i,j

(19)

since ∥Σ∥2F= r σr(Σ)2 ≥ σ1(Σ)2.

| |
|---|

Remark A.2. In the square case m = n = N, Lemma A.1 reduces to maxi,j|(Σ)ij|≥ σ1(Σ)/N, where N denotes the sequence length when Σ is formed over N tokens.

A. Massive Activation in Band-wise Representations

Lemma A.3 (Cone Condition Implies Coherent Summation). Let { kj}Nj=1 denote the key vectors projected onto a single RoPE frequency band, with

kj = βj R(θj,f)k, βj ≥ βmin > 0,

where R(θj,f)∈R2×2 rotates by phase θj,f =ωfj, and k is a fixed unit vector. Assume the cone condi-

tion: there exists a unit vector uf and half–angle γK <

π 2

such that

⟨uf, R(θj,f)k⟩ ≥ ∥k∥cosγK for all j.

(20) Then the band-wise sum S = Nj=1 kj satisfies

∥S∥ ≥ N βmin ∥k∥cosγK. (21)

Here N is the sequence length (number of token positions in this frequency band).

Proof. Align uf with the mean direction of the projected keys. Then by linearity and the cone condition,

⟨uf, kj⟩ = βj ⟨uf,R(θj,f)k⟩ ≥ βj ∥k∥cosγK.

(22)

Summing over j yields

N

⟨uf,S⟩ ≥ ∥k∥cosγK

βj ≥ N βmin ∥k∥cosγK.

j=1

(23)

Since ∥S∥≥ ⟨uf,S⟩ for any unit uf, the result follows.

| |
|---|

- Theorem A.4 (Spectral Amplification and Massive Activation). Under the cone condition of Lemma A.3, define the band-wise Gram matrix

Σf =

N

j=1

kj k⊤j .

Then

λmax(Σf) ≥ N βmin2 ∥k∥2cos2 γK, (24) and consequently

σ1(KRf ) ≥ βmin ∥k∥

√

N cosγK. (25)

Similarly, for QRf satisfying an analogous cone condition with (αmin,γQ),

σ1(QRf ) ≥ αmin ∥q∥

√

N cosγQ. (26) Proof. By the Rayleigh quotient,

λmax(Σf) ≥ x⊤Σfx =

N

j=1

(⟨x, kj⟩)2

≥

1 N

N

j=1

⟨x, kj⟩

2

= ∥S∥2 N

. (27)

Applying Lemma A.3 gives λmax(Σf) ≥ N βmin2 ∥k∥2cos2 γK, and taking square roots yields the bound on σ1(KRf ). Repeating for QRf gives the symmetric result.

| |
|---|

Discussion. This theorem shows that within low-frequency RoPE bands, coherent phase rotations accumulate along depth and sequence length,

producing√ ℓ2-norm amplification proportional to

N—the hallmark of massive activations observed in RoPE-based transformers.

B. Attention Sink Amplification in Band-wise Attention Logits

- Theorem A.5 (Spectral Amplification of Attention Scores). Given the bounds in Theorem A.4, consider the attention submatrix contributed by band f,

QRf KRf⊤ √dh

. (28)

Af =

Let ψ denote the angle between the dominant singular directions of QRf and KRf . Then its leading singular value satisfies

N ∥q∥∥k∥cosγQ cosγK cosψ. (29) Consequently, by Lemma A.1,

σ1(Af) ≳ αmin√dβmin

h

∥q∥∥k∥cosγQ cosγK cosψ, (30)

maxi,j |(Af)ij| ≥ αmin√dβmin

h

which remains Ω(1) even as sequence length grows.

Proof. Since σ1(Af) ≤ ∥QRf ∥2 ∥KRf ∥2/√dh, using Theorem A.4 gives

σ1(Af) ≳

1 √dh

σ1(QRf )σ1(KRf ) cosψ, (31)

yielding the desired bound. The entry-level lower bound follows by applying Lemma A.1.

| |
|---|

Discussion. This second bound extends the massive-activation effect from hidden representations to attention logits. When both query and key bands satisfy the cone constraint, their coherent multiplication produces an O(N) scaled singular value, concentrating attention mass on a few entries, precisely the phenomenon known as attention sink.

C. Truncated Matrix Entropy and Spectral Amplification

Definition A.6 (Truncated Matrix Entropy). For a head-level Gram matrix Σh with eigenvalues λ1≥ λ2≥···≥λr >0, the truncated matrix entropy of order r is

Hhr =

r

1 r

λi log λi, r≤rank(Σh). (32)

i=1

It measures the information-weighted energy concentration of the top-r singular components within an attention head.

Theorem A.7 (Spectral Amplification Decreases Truncated Entropy). Let Σf be the band-wise Gram matrix defined in Theorem A.4 with eigenvalues λ1 ≥λ2 ≥···. Suppose the cone condition holds and the dominant eigenvalue obeys

##### λ1 ≥ N βmin2 ∥k∥2cos2 γK,

λi ≤ (1 − δ)λ1, δ∈(0,1).

i>1

(33)

Then the truncated matrix entropy of order r≥1 satisfies

Hhr ≤ λ1 log λ1 + 1−rδλ1 log (1−r−δ)1λ1 , (34)

and therefore decreases monotonically with stronger spectral amplification (i.e., larger λ1 or smaller δ).

Proof. Let λ1 be the amplified mode and distribute the remaining trace mass (1 − δ)λ1 equally among the next (r − 1) eigenvalues, an entropymaximizing configuration under the given trace constraint. Then

(1 − δ)λ1 r − 1

(1 − δ)λ1 r − 1

, (35)

Hhr = 1r λ1 log λ1 + (r − 1)

log

which simplifies to the stated bound. As λ1 increases or δ decreases, the first term dominates and the total entropy declines, showing that truncated entropy is inversely related to the degree of spectral concentration.

| |
|---|

Discussion. When RoPE’s low-frequency cone constraint amplifies one dominant spectral direction (large λ1) while suppressing others (small λi>1), the truncated entropy Hhr becomes small. Hence, heads with low truncated entropy correspond to those exhibiting spectral amplification and potential attention sinks. This justifies using Hhr as a quantitative criterion to identify “noisy” heads for denoising in DoPE.

#### A.2 Experimental Setup

Models. Qwen2.5-Math-7B (Yang et al., 2024) and LLaMA-3-8B-Instruct (Grattafiori et al., 2024) are decoder-only transformer models that employ Rotary Positional Embeddings (RoPE) for encoding positional information. Qwen-1.5-7B is trained with a maximum context length of 32K tokens, while LLaMA-3-8B is trained with a 8K-token context window. To support longer contexts beyond their pre-training limits, we apply RoPE-based extrapolation (e.g., Dynamic-NTK), which rescales RoPE frequencies to improve stability and retrieval performance in extended-context settings.

Hyperparameter. All experiments use greedy decoding with temperature set to 0.0 and top-p set to 1.0. For the needle-in-a-haystack (NIH) task on LLaMA-3-8B-Instruct, we set max_new_tokens to 50 with stop conditions including newline characters (<0x0A>) and stop token ID 144. For the many-shot in-context learning (MICL) task

on Qwen2.5-Math-7B, we set max_new_tokens to 2048 with stop sequences </s>, <|im_end|>, <|endoftext|>, and Problem: to prevent generating additional problems. Context buffers of 200 tokens (NIH) and 2,300 tokens (MICL) are reserved for prompt templates and final questions. All experiments are conducted using SGLang (Zheng et al., 2023) (v0.5.3rc0) with the FlashAttention-3 backend (Shah et al., 2024). Tensor parallelism is enabled for multi-GPU inference when necessary. CUDA graphs are disabled to support dynamic context lengths. All experiments are conducted on five A100 GPUs. Head selection is performed globally across all (l × h) attention heads, where l is the number of layers and h is the number of heads per layer (32 layers × 32 heads = 1,024 total heads for LLaMA-3-8B; 28 layers × 28 heads = 784 total heads for Qwen2.5-Math-7B).

For RoPE extrapolation, we apply DynamicNTK scaling (emoZilla, 2023) with the scaling factor computed as α = Ltarget/Loriginal, where Ltarget ∈ {24K,64K,128K} for NIH experiments and Ltarget = 16K for MICL experiments, while Loriginal corresponds to each model’s pre-trained maximum position embeddings (32K for Qwen-1.5-7B, 8K for LLaMA-3-8B-Instruct, and 4K for Qwen2.5-Math-7B). For LLaMA3, we additionally evaluate NTK-by-parts (Peng et al., 2023) with low_freq_factor= 1.0 and high_freq_factor= 32.0. The NIH task uses 10 uniformly spaced depth positions (0%, 10%, ..., 100%) for needle insertion at each context length. The MICL task evaluates 100 sampled problems from the MATH dataset (Hendrycks et al., 2021), with needle insertion at four fixed depth positions (0%, 33%, 67%, 100%, corresponding to beginning, 1/3, 2/3, and end) within the in-context examples, yielding 400 total test configurations.

For DOPE, Gaussian noise is sampled from N(0,1) with standard deviation σ = 1.0, using a fixed random seed (42) to ensure reproducibility. The truncated matrix entropy is computed by retaining the top-k singular values where k ∈ {1,4,8,16,32}, with k = 1 corresponding to using only the spectral norm σmax(Σ). We also evaluate the full (untruncated) matrix entropy for comparison.

Baselines. Several of the baseline models adopt training-free methods to extend effective context length while preserving short-range quality.

Dynamic NTK (emoZilla, 2023) adjusts the ro-

tary position embedding (RoPE) base with a lengthdependent scaling factor at decoding time so that the current effective angular frequency remains closer to the pretraining regime even when the sequence exceeds the original context window. Compared with static “NTK-aware” scaling, the dynamic variant reduces frequency drift as length grows and improves long-range stability without additional training; it is also frequently combined with other RoPE extensions in practice.

Dual Chunk Attention (An et al., 2024) is a training-free attention scheme that partitions long sequences into manageable chunks and combines intra-chunk attention (local fidelity) with interchunk routing/aggregation (global recall). This design preserves token–token interactions within chunks while enabling information flow across distant chunks, scaling models to 100k+ tokens without continual training, and can be composed with RoPE-based extensions such as PI/NTKaware/YaRN.

Positional Interpolation (Chen et al., 2023c) linearly down-scales the input position indices before applying RoPE so that positions beyond the original window are interpolated back into the training range rather than extrapolated. This simple modification allows RoPE-based LLMs to reach 32k-context with minimal fine-tuning while maintaining competitive short-context performance and avoiding unstable attention magnitudes that arise in naive extrapolation.

