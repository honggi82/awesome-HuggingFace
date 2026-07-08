# arXiv:2603.04800v1[cs.CV]5Mar2026

## MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models

Lulu Hu* Wenhu Xiao* Xin Chen* Xinhua Xu Bowen Xu† Kun Li Yongliang Tao Alibaba Cloud Computing, Alibaba Group

{chudu.hll, wenhu.xwh, andy.cx, bowen.xbw}@alibaba-inc.com

### Abstract

Post-training quantization (PTQ) with computational invariance for Large Language Models (LLMs) have demonstrated remarkable advances, however, their application to Multimodal Large Language Models (MLLMs) presents substantial challenges. In this paper, we analyze SmoothQuant as a case study and identify two critical issues: Smoothing Misalignment and CrossModal Computational Invariance. To address these issues, we propose Modality-Aware Smoothing Quantization (MASQuant), a novel framework that introduces (1) Modality-Aware Smoothing (MAS), which learns separate, modality-specific smoothing factors to prevent Smoothing Misalignment, and (2) Cross-Modal Compensation (CMC), which addresses Cross-modal Computational Invariance by using SVD whitening to transform multi-modal activation differences into low-rank forms, enabling unified quantization across modalities. MASQuant demonstrates stable quantization performance across both dual-modal and tri-modal MLLMs. Experimental results show that MASQuant is competitive among the state-of-the-art PTQ algorithms. Source code: https://github.com/ alibaba/EfficientAI.

### 1. Introduction

Post-training quantization (PTQ) has become essential for deploying Large Language Models (LLMs) [1, 9, 32] on resource-constrained devices, and this need is amplified for Multimodal Large Language Models (MLLMs), which demonstrate impressive cross-modal reasoning capabilities [11, 13, 17, 29, 42, 43]. PTQ methods based on computational invariance [19, 24, 27, 31, 36, 37], particularly channel-wise smoothing [19, 27, 37], have proven highly effective for text-only LLMs by redistributing activa-

*Equal contribution. †Corresponding author

tion outliers through channel-level scaling factors. Recent work has also begun recognizing MLLM-specific characteristics—MBQ [16] observes unequal contributions of visual and text tokens to quantization error, while MQuant [41] finds that visual token activations exhibit much higher magnitudes than text tokens. However, the direct application of channel-wise smoothing to MLLMs remains surprisingly underexplored, raising an important question: do these successful channel-wise smoothing PTQ methods transfer seamlessly to the multimodal setting?

Through systematic analysis of vision-language and omni-modal MLLMs [3, 39], we identify a fundamental problem. Different modalities exhibit vastly different activation magnitudes—visual tokens typically show ranges 10–100× larger than text and audio tokens. Channel-wise smoothing computes a single scaling factor per channel, but when modalities with such disparate distributions pass through the same layer, the dominant modality’s larger activations dictate the smoothing factor. Activations from nondominant modalities thus become over-smoothed, crushing their signal and causing severe quantization errors. We term this phenomenon smoothing misalignment. A natural solution is to compute separate smoothing factors for each modality. However, this seemingly simple solution introduces a critical flaw: preserving computational invariance under this scheme requires storing distinct quantized weights for each modality. This defeats the fundamental purpose of quantization, which aims to reduce memory footprint through a single low-precision weight representation. The question becomes: can we collect modalityspecific smoothing factors while maintaining a single quantized weight for inference?

We address this challenge through Modality-Aware Smoothing Quantization (MASQuant). Our key idea is to learn dedicated smoothing factors for each modality, but during inference, we use the text-smoothed weights as a base and apply modality-specific low-rank compensation. This design simultaneously resolves smoothing misalignment and preserves computational invariance. Specifically,

Modality-Aware Smoothing (MAS) optimizes smoothing factors directly for each modality’s activations, eliminating smoothing misalignment and pushing channel-wise smoothing to its optimization limit. To maintain a single weight representation, Cross-Modal Compensation (CMC) leverages a key observation: differences in smoothed activations across modalities are low-rank. We prove this mathematically and use SVD-based whitening to transform these differences into compact low-rank matrices. By compensating for modality-induced variations through lightweight low-rank corrections to the text-smoothed base weights, MASQuant achieves modality-specific adaptation without sacrificing the unified weight structure essential for efficient quantization.

We evaluate MASQuant across diverse MLLM architectures spanning vision-language and omni-modal configurations. Results across all evaluated benchmarks demonstrate consistent superiority over existing channel-wise smoothing PTQ methods.

In summary, our main contributions are:

- • We identify and formalize smoothing misalignment—the fundamental obstacle in applying channel-wise smoothing PTQ to MLLMs—and resolve it through ModalityAware Smoothing.
- • We prove that inter-modal activation differences are lowrank, enabling Cross-Modal Compensation to maintain computational invariance with a single set of quantized weights.
- • We present MASQuant, a PTQ method that is effective on both vision-language and omni-modal MLLMs.

### 2. Related Work

LLMs Quantization. LLM quantization methods are broadly categorized into Quantization-Aware Training (QAT) [4, 6, 21] and Post-Training Quantization (PTQ) [19, 27, 37]. QAT incorporates quantization into training to adapt models to low-precision computation, while PTQ applies quantization directly using calibration data. PTQ approaches include: (1) error compensation via second-order gradients [10] or low-rank correction [40, 44]; (2) channelwise smoothing to mitigate outliers [19, 24, 27, 31, 37]; (3) rotation-based distribution restructuring [2, 5, 22]; and (4) mixed-precision strategies [7, 14, 45].

MLLMs Quantization. Quantizing MLLMs presents unique challenges due to cross-modal activation disparities. MQuant [41] identifies that visual token activations can exceed textual ones by 20×, proposing modality-specific quantization. MBQ [16] observes visual tokens are less quantization-sensitive and introduces gradient-weighted error balancing. QSLAW [38] addresses increased outlier density from multimodal inputs through learnable weightgroup scaling. Despite these advances, activation quan-

tization remains inadequately addressed, motivating our modality-aware smooth quantization approach.

### 3. Preliminaries

#### 3.1. Computational Invariance based PTQ

PTQ addresses significant computational and storage challenges by mapping high-precision floating-point tensor x to low-precision N-bit integer tensor xˆN:

x ∆

+ z,qmin,qmax − z · ∆,

xˆN = Q(x) = clamp

(1) where ∆ is the scale factor, z is zero-point, ⌊·⌉ is the rounding-to-nearset operator, and clamp clips values outside the integer range [qmin,qmax]. We use WxAy notation for x-bit weights and y-bit activations, with two main types: weight-only quantization (e.g., W4A16) and weightactivation quantization (e.g., W8A8, W4A8). For a linear layer Y = XW, where X ∈ RT×D

in, W ∈ RD

in×Dout, the layer can be reformulated based on computational invariance as:

Y = (XS−1) · (SW), (2)

where S can be diagonal matrix [19, 27, 37] or orthogonal matrix [5, 18, 22, 31]. S reduces the outliers in X and lead to better quantization reconstruction loss:

L = L(Q(XS−1) · Q(SW),XW). (3)

In this paper, we demonstrate that applying S for different modalities in MLLMs can achieve robust and effective PTQ performance.

#### 3.2. SVD-based Whitening

Prior work [34, 35] uses SVD-based whitening for low-rank weight compression. Given input activations X and weights W, the whitening transform is derived by decomposing the activation covariance:

###### PΛP⊤ = SVD(X⊤X). (4)

A whitening matrix T = (PΛ1/2)⊤ transforms XW into (XT)(T−1W) which satisfies (XT−1)⊤(XT−1) = I, yielding whitened activations. SVD-LLM v2 [35] shows that performing SVD on TW and truncating to rank r minimizes reconstruction error ∥XW − XW′∥F:

U,S,V = SVD(TW), (5) Ur,Sr,Vr = Truncr(U,S,V), (6)

##### W′ = T−1UrSrVr. (7)

While existing methods apply whitening for weight compression, we are the first to show that whitening can effectively compensate for cross-modal weight differences, enabling unified quantization across modalities.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Text Tokenizer Vision Encoder Audio Encoder

###### SQNR = 5.31 PPL = 18.19

[Figure 7]

Fox

[Figure 8]

Over Smoothed

What brand is in white letters with a red background?

[Figure 9]

[Figure 10]

[Figure 11]

!

!(#!$)

[Figure 12]

[Figure 13]

# !('/#) MBR Loss

RMS Norm

'

$

|q|
|---|

|k|
|---|

|v|
|---|

Channel-wise Smoothing Modality Balance Reconstruction

(b) MBQ + SmoothQuant W4A8

Text serves as the dominant modality

Soft max

[Figure 14]

[Figure 15]

[Figure 16]

###### SQNR = 8.25 PPL = 15.90

[Figure 17]

Coca-Cola

[Figure 18]

[Figure 19]

|o|
|---|

[Figure 20]

[Figure 21]

'" '# '$

"Answer the question using a single word."

[Figure 22]

RMS Norm

!(#"$)

Vision serves as the dominant modality

MBR Loss !('/#)

[Figure 23]

MLP

[Figure 24]

$

×"

Coca-Cola

[Figure 25]

Modality-aware Smoothing Modality Balance Reconstruction

(a) FP16 MLLM Inference

(c) MASQuant W4A8

- Figure 1. (a). Activation distributions during multimodal reasoning in MLLMs. Different dominant modalities emerge across MLLM components, leading to failure of general PTQ methods that diminish vision importance. (b). Impact of SmoothQuant’s uniform smoothing factors S computation on MLLM quantization performance (low SQNR, high PPL). (c). MASQuant addresses smoothing misalignment through the combination of MAS and CMC, thereby significantly enhancing PTQ performance in MLLMs. MBR Loss indicates Modality Balanced Reconstruction Loss.

### 4. MASQuant

whose diagonal entries are smoothing factors si:

In this section, we present our Modality-Aware Smooth Quantization (MASQuant) framework includes ModalityAware Smoothing (MAS) followed by Cross-Modal Compensation (CMC), designed to address the critical issue of smoothing misalignment and cross-modal computational invariance in MLLMs.

S∗ = arg min

L(Q XS−1 Q(SW),XW). (10)

S

This formulation is more flexible—it can discover optimal smoothing patterns beyond what any β-parameterized formula can express.

#### 4.1. Motivation

Smoothing Misalignment. MLLMs process multiple modalities with vastly different activation magnitudes [41]. we measure the activation range per channel: Rmi = maxt |xmt,i| for modality m ∈ M, where xmt,i is the activation of the t-th token in channel i. When calibrating on mixed-modality data, the unified smoothing factor is determined by the maximum range across all tokens:

Revisiting Smoothing Factors. Existing methods compute smoothing factors s through closed-form solutions. SmoothQuant [37] uses:

maxt |xt,i|β maxj |wj,i|1−β

, (8)

si =

while AWQ [19] adopts:

(maxm,t |xmt,i|)β (maxj |wj,i|)1−β

(maxt |xt,i|)β (maxj |wj,i|)1−β

∗

sunii =

n t=1 |xt,i|

∝ Rm

=

i ,

)β,β∗ = argmin

L(s). (9)

si = (

n

β

(11) where m∗ = arg maxm Rmi is the dominant modality. This unified factor suni aligns with the dominant modality’s ideal factor but severely mismatches others—a phenomenon we term smoothing misalignment. Non-dominant modalities suffer degraded quantization quality as their activations are scaled by factors optimized for a different magnitude regime. Figure 1 illustrates this phenomenon in detail.

MBQ [16] extends this to MLLMs by adjusting the contribution of different modality activations when searching for β values. However, all these methods optimize only the hyperparameter β while the smoothing factors themselves are never treated as free parameters. OmniQuant [27] demonstrates that learnable smoothing factors can better minimize quantization error, we adopt this principle for MLLMsinstead of searching over β, we directly optimize matrix S

Unified Smoothing (W4A8)

Optimal Smoothing (W4A8)

20

Unified Smoothing (W4A6)

Optimal Smoothing (W4A6)

Unified Smoothing (W4A4)

15

Optimal Smoothing (W4A4)

SQNR(dB)

10

5

0

0 10 20 30 Transformer Layer Index

- Figure 2. Comparative analysis of SQNR degradation of Qwen2.5Omni-3B under multimodal input condition. We selected 32 samples from OmniBench and computed the average SQNR for each layer.

#### 4.2. Modality-Aware Smoothing

Smoothing misalignment arises from unified scaling across heterogeneous modalities. We address this by maintaining modality-specific smoothing factors, eliminating dominance at its root. Let M denote the set of supported modalities (e.g., text, image, audio). For each modality m ∈ M, we first obtain the initial values of the modalityaware smoothing factors Sm ∈ Rd×d:

maxt |xmt,i| maxj |wj,i|

Sm = diag(sm), smi =

, m ∈ M.

(12) We then optimize the Sm by minimizing MAE loss [16] on modality-specific data, We simplify the notation {Sm}m∈M as {Sm} for brevity:

{S∗m} = arg min

(λm · LMAE(Sm,Xm,W)), (13)

{Sm} m∈M

where λm denotes the loss weight for modality m. For modality m, the quantization reconstruct MAE loss is:

LMAE = ||Q XmS−m1 · Q(SmW) − XmW||. (14)

This ensures S∗m captures modality-specific statistics without cross-modality interference. We quantify benefit of Sm through SQNR [28, 33] analysis. For a token xt, SQNR measures quantization quality as:

SQNR(xt) = 10log10 ∥xt∥2 ∥xt − Q(xt)∥2

, (15)

Under channel-wise smoothing with smoothing factor s, when xt ≫ ∆t in Equation 1, quantization error et can

be approximated as uniformly distributed over −∆

2 , ∆

2 . Thus, the mean squared error across d channels is d · ∆

t

t

2 t

12 . This yields SQNR of SmoothQuant (omitting constants):

2

xt,i si

d i=1

SQNR(s,xt) ∝

∝

2 t

d · ∆

12

2

xt,i si

d i=1

2, (16)

maxi xst,i

i

based on which the degradation can be quantified.

Theorem 1. [SQNR Degradation under Smoothing Misalignment] Consider a layer processing multimodal inputs with dominant modality m and non-dominant modality m′. Let αm,m

′

′

i denote the range ratio at channel i, where Rim and Rm

i = Rim/Rm

′

i are the activation ranges. Using unified smoothing suni ≈ sm yields SQNR degradation (exact as αm,m

′

≫ 1): SQNR(suni,xm

′

′

′

t ) = SQNR(sm

,xm

t )

 .

 d(mini (αm,m

′

i )2)

− 10log10

d i=1

1 (αim,m′)2

(17) Proof. To simplify notation, we let xi = xm

′

t,i ,R = Rm

′

′

i ,and αi = αm,m

i . We compare the upper bounds of the SQNR for the token xm

′

t under two smoothing strategies, which are achieved when the token’s activations are uniform across channels(i.e., x1/R1 ≈ x2/R2 ≈ ··· ≈ xd/Rd): The SQNR for optimal smoothing is proportional to the dimension d:

d i=1(xi/Ri)2

′

SQNR(sm

,x) = 10log10

= 10log10 d,

(maxi |xi/Ri|)2

(18) and SQNR for unified smoothing upper bound is given by:

d i=1 1/αi2

SQNR(suni,x) = log10

. (19)

(maxi 1/αi)2

The difference between the two SQNR upper bounds is:

d

1 αi2

1 d · (maxi α1

∆ = 10log10

)2

i=1

i

≤0, equality iff α1=···=αd

.

(20)

| |
|---|

The results in Figure 2 validate Theorem 1 and Figure 4c and 4d provide distribution of αm,m

′

i is non-uniform. Based on this, the inference process of the MAS becomes as follows:

Y = Q XmS−m1 · Q(SmW), m ∈ M. (21)

Inverting the whitening yields the low-rank correction:

Modality-Aware Smoothing

Cross-Modal Compensation

###### W

Xt

##### ∆W ≈ L1L2, where L1 = T−1Ur, L2 = ΣrVr⊤.

StW

(25)

St

SvW

ΔW

TΔW

L1 L2

SVD

Theorem 2 (Optimal Low-rank Compensation). The rankr matrix L∗ = L1L2, where L1, L2 are given by the rank-r truncated SVD defined in Eq 25, minimizes the reconstruc-

Sv

Inv

###### X

Xv

SVD based whitening

T

S-1v

###### Xv

(a)

tion loss L(L) = XvS−v 1(∆W − L) 2F. i.e,

XvS-1v

###### MLP Block

L1 L2

XvS−v 1(∆W − L) 2F . (26)

L∗ = arg min rank(L)≤r

Q(StWup)

Q(Wdown)

Proof. Considering only two modalities, text and vision, and performing weight-only quantization exclusively, As defined, we have

Q(StWgate) Act

(S-1v,S-1T)

X

L1 L2

XvS-1v

(b)

##### L∗ = T−1 (Truncr(T∆W)), (27)

- Figure 3. The illustrated case demonstrates a text-vision dualmodal setting. (a) Schematic workflow of MAS and CMC with calibration data, (b) Illustration of how low-rank matrices L1 and L2 in CMC are utilized in MASQuant, exemplified with an MLP block.
- 4.3. Cross-Modal Compensation

where Truncr(·) denotes the best rank-r approximation obtained via truncated SVD., and the whitening matrix T = (PΛ1/2)⊤. On the other hand, the SVD decomposition of the covariance matrix is (XvS−v 1)⊤(XvS−v 1) = PΛP⊤. It leads to XvS−v 1 = UΛ1/2P⊤ = UT, where U is orthonormal. Therefore, we have

While MAS eliminates smoothing misalignment, it produces modality-specific quantized weight matrices Q(SmW) that violate computational invariance—PTQ requires a single quantized weight across all modalities.

L(L∗) = ||XvS−v 1(∆W − L∗)||2F

= ||UT(∆W − L∗)||2F

= ||UT(∆W − T−1Truncr(T∆W))||2F

Our strategy is to store only one quantized weight Q(StW) using text smoothing as reference, and compensate for other modalities via low-rank corrections. Consider

= ||T∆W − Truncr(T∆W)||2F

(28)

- = ||SVD(T∆W)||2F
- = ||SVD(U−1XvS−v 1∆W)||2F

vision inputs: ideally we compute XvS−v 1 ·(SvW), but using the shared weight produces a residual:

= ||SVD(XvS−v 1∆W)||2F = L2min.

∆Y = XvS−v 1 · (SvW − Q(StW))

. (22)

So the designed SVD truncation ensures the theoretical minimum truncation loss.

∆W

| |
|---|

A natural approach is low-rank approximation of ∆W. However, directly applying SVD fails: it does not minimize the output residual ∆Y, and ∆W lacks low-rank structure. Our key insight is that whitening the activations induces low-rank structure. We compute the whitening transform via:

The theorem 2 shows that SVD-based whitening enables low-rank compensation to effectively bridge modality gaps. The final inference combines the base quantized output with modality-specific corrections:

 

Q(XmS−m1) · Q(StW), m = text Q(XmS−m1) · Q(StW)

SVD (XvS−v 1)⊤(XvS−v 1) = PΛP⊤, T = (PΛ1/2)⊤.

(29)

Y =

(23) which ensures (XvS−v 1)T−1 is orthonormal. The whitened residual T(∆W) exhibits strong low-rank structure (see Figure 5), enabling accurate approximation via truncated SVD:

m ̸= text



+ XmS−m1 · Lm1 Lm2 .

This maintains a single quantized weight for efficiency while using lightweight low-rank matrices to preserve accuracy across modalities. The complete pipeline is illustrated in Figure 3.

SVD(T(∆W)) = UΣV⊤ ≈ UrΣrVr⊤. (24)

- Table 1. Comparison of MASQuant with existing quantization methods on multimodal benchmarks. OCR: OCRBench. Viz: Vizwiz. S-QA: ScienceQA. T-VQA: TextVQA. SQ: SmoothQuant. The best results are highlighted in bold.

|Method| | |Qwen2.5-VL-3B Qwen2.5-VL-7B| | |
|---|---|---|---|---|---|
| |Bits<br><br>| |MMMU OCR Viz S-QA T-VQA Avg Acc↑ Acc↑ Acc↑ Acc↑ Acc↑ Acc↑| |MMMU OCR Viz S-QA T-VQA Avg Acc↑ Acc↑ Acc↑ Acc↑ Acc↑ Acc↑|

Dense FP16 42.2 79.3 69.1 81.9 77.9 70.1 46.7 83.8 70.8 88.4 82.9 74.5

|RTN AWQ MBQ MASQuant(ours)<br><br>|W4A16| |40.9 77.9 63.0 81.3 75.8 67.8<br>41.9 78.6 68.1 81.7 77.2 69.5 41.9 78.5 68.0 81.5 76.8 69.3 43.3 78.6 67.7 82.4 77.3 69.9<br><br><br>| |43.3 83.7 67.8 81.3 82.1 71.6<br><br>43.3 83.7 70.6 87.8 82.2 73.5<br>44.4 82.8 70.6 87.8 82.9 73.7<br><br><br>44.4 84.6 71.5 87.8 82.5 74.2<br><br><br>|
|---|---|---|---|---|---|
|RTN SQ MBQ MASQuant(ours)<br><br>|W8A8<br><br>| |42.6 78.7 68.2 82.4 77.3 69.8<br>42.6 79.0 68.0 82.2 77.5 69.9 43.3 78.9 68.6 81.8 77.8 70.1 46.6 79.5 68.2 82.4 77.7 70.9<br><br><br>| |45.6 83.8 70.5 88.1 82.5 74.1 43.3 83.8 70.0 88.2 82.6 73.6<br>46.7 83.5 70.6 88.5 82.9 74.4 46.2 84.2 70.6 88.6 82.6 74.4<br><br><br>|
|RTN SQ MBQ MASQuant(ours)<br><br>|W4A8<br><br>| |25.6 0.0 0.0 0.0 0.0 5.1 25.6 66.9 57.5 72.1 63.9 57.4 41.2 66.9 65.0 76.7 73.4 64.6 46.7 67.2 62.7 77.9 69.2 64.7<br><br>| |43.3 68.3 63.2 85.2 76.9 67.4 37.8 70.2 61.5 83.3 71.1 64.8 43.3 74.1 64.3 86.0 74.8 68.5 43.3 72.8 66.4 85.7 77.0 69.0<br><br>|
|SQ MBQ MASQuant(ours)<br><br>|W4A6<br><br>| |22.5 66.7 52.9 69.9 55.9 53.6 38.7 65.6 60.5 64.7 69.1 59.7 40.0 66.7 56.2 71.2 65.3 59.9<br><br>| |28.9 67.7 60.5 78.6 69.3 61.0 30.0 71.7 59.8 80.1 72.9 62.9<br>29.7 70.3 62.6 79.7 72.9 63.0<br><br><br>|

- Table 2. Comparison of MASQuant with existing quantization methods on omni-modal MLLMs (vision, audio and text). SQ: SmoothQuant. Libri: Librispeech. Wen: Wenetspeech. The best results are highlighted in bold.

| | | |Qwen2.5-Omni-3B Qwen2.5-Omni-7B<br><br>| | | | |
|---|---|---|---|---|---|---|---|
|Method<br><br>|Bits| |Audio-Text Vision-Text Libri Wen MMMU<br><br>WER↓ WER↓ Acc↑<br><br>|Vision-Audio-Text Omnibench Acc↑| |Audio-Text Vision-Text Libri Wen MMMU<br><br>WER↓ WER↓ Acc↑<br><br>|Vision-Audio-Text Omnibench Acc↑<br><br>|

Dense FP16 3.9 7.5 43.3 43.8 2.9 7.1 50.0 45.3

|RTN AWQ MBQ MASQuant(ours)<br><br>|W4A16<br><br>| |4.4 9.2 9.4 8.0 8.2 7.0 4.0 8.0<br><br>|32.2 36.7 38.9 39.6<br><br>|39.6 43.8 42.6 46.9<br><br>| |3.4 6.7 8.7 7.1<br>3.5 6.6 2.7 7.4<br><br><br>|44.4 47.8 47.8 48.8<br><br>|43.8 45.3 44.4 45.3<br><br>|
|---|---|---|---|---|---|---|---|---|---|
|RTN SQ MBQ MASQuant(ours)<br><br>|W8A8| |3.8 7.6 3.6 7.5 8.9 7.8 3.8 7.9<br><br>|41.3 41.7 41.8 42.3<br><br>|37.7 44.5 46.1 46.1<br><br>| |8.2 7.0 8.2 7.3 8.1 7.1 2.8 6.9<br><br>|48.9 48.8 50.0 49.9<br><br>|38.4 47.6 46.4 46.9<br><br>|
|RTN SQ MBQ MASQuant(ours)<br><br>|W4A8| |109.7 105.6 77.4 94.2 9.5 8.5 3.6 8.7<br><br>|28.9 30.0 27.8 36.7<br><br>|29.7 27.3 36.7 41.4<br><br>| |9.0 8.7 8.6 8.3 3.8 8.2 2.9 8.0<br><br>|42.2 42.2 47.8 48.8<br><br>|35.2 36.7 40.6 43.8<br><br>|
|RTN SQ MBQ MASQuant(ours)<br><br>|W4A6| |87.8 99.6 87.8 99.5 10.8 10.4 3.7 8.9<br><br>|28.9 28.9 31.4 32.2<br><br>|23.4 18.8 46.9 42.2<br><br>| |11.8 15.7<br><br>5.8 19.8<br><br>6.4 14.8 4.7 8.7<br><br><br>|28.9 37.8 33.3 36.8<br><br>|40.6 35.9 42.1 42.2<br><br>|

A M

S M

S M

S M

### 5. Experiments

#### 5.1. Experimental Setups

We evaluate on Qwen2.5-VL [3] and Qwen2.5-Omni [39], multimodal LLMs supporting text, audio, and vision. The architecture of Omni MLLMs includes 675M Vision Transformer-based [8] vision encoder, Whisperlarge-

v3 [26] based audio encoder, LLM Transformer decoder Thinker, speech output Talker, and Streaming Codec Decoder. For generality, we quantize only the LLM component Thinker. We evaluate several state-of-the-art channelwise smoothing-based quantization methods: AWQ [19], SmoothQuant [37], and MBQ [16]. We evaluate model performance across multiple multimodal benchmarks: Audio-

to-Text Tasks: Evaluated on Librispeech [25] dataset testother split and Wenetspeech [43] dataset test-net split, using Word Error Rate (WER) as metric. Visual Reasoning Tasks: Evaluated on OCRBench [20], TextVQA [30], Vizwiz [12], ScienceQA [23] and MMMU [42].Multimodal Reasoning Tasks: Evaluated on OmniBench [17], covering joint textaudio-visual reasoning.

#### 5.2. Vision-Language MLLMs

- Table 1 evaluates MASQuant on Qwen2.5-VL models. MASQuant matches FP16 performance on both model sizes at W8A8, suggesting that MLLMs can be quantized to 8 bits without accuracy loss when modality-specific characteristics are properly handled. RTN completely fails on W4A8, while SmoothQuant severely degrades. This failure pattern reveals that modality dominance becomes catastrophic at aggressive quantization levels, where smoothing factors dictated by the dominant modality harm the weaker modality.

5.3. Vision-Audio-Language MLLMs

- Table 2 evaluates quantization on omni-modal models handling vision, audio, and text. The results reveal that modality dominance intensifies as modality diversity increases. At W4A8, SmoothQuant’s audio performance collapses catastrophically on 3B: Librispeech WER jumps from 3.9 to 77.4, and Wenetspeech from 7.5 to 94.2. This 20× degradation, far worse than vision-language failures, demonstrates that audio is completely suppressed when competing with vision and text for smoothing resources. MASQuant maintains near-FP16 audio quality, confirming that per-modality smoothing prevents this collapse. Audio’s vulnerability is intuitive: its smaller activation magnitudes make it the first victim of vision-dominated smoothing factors.

#### 5.4. Analysis

Modality Dominance. We investigate modality dominance by applying SmoothQuant to Qwen2.5-VL and Qwen2.5-Omni. Figure 4 shows that visual tokens exhibit significantly larger activations than text in both attention and MLP layers. Consequently, smoothing factors track visual distributions while mismatching text by orders of magnitude. This confirms modality dominance is pervasive in MLLMs. We further analyze αi distribution, which determines MAS’s SQNR gain. Figures 4c and 4d show nonuniform αi across components in both models, validating consistent improvements from MAS.

Effective Rank. CMC relies on weight differences having low-rank structure after whitening. We verify this on Qwen2.5-VL/Omni-3B using effective rank (lower is better). As shown in Figure 5, whitening reduces effective rank substantially, confirming our design.

###### Self-Attention Query (self_attn.q)

###### Self-Attention Query (self_attn.q)

100

Proportion(%)

Proportion(%)

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

Text

Text

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

Vision Audio

Vision

| |
|---|

50

| |
|---|

0

0

MLP Up Projection (mlp.up)

MLP Up Projection (mlp.up)

100

Proportion(%)

Proportion(%)

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |
| | |

| | |
|---|---|
| | |

50

0

0

Self-Attention Output (self_attn.o)

Self-Attention Output (self_attn.o)

100

Proportion(%)

Proportion(%)

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

| |
|---|

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

| |
|---|

| |
|---|

0 5 10 15 20 25 30 35 Transformer Layer Index

0 5 10 15 20 25 30 35 Transformer Layer Index

(a) Qwen2.5-Omni-3B

(b) Qwen2.5-VL-3B

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

Attention Q/K/V MLP Up/Gate Attention O

Attention Q/K/V MLP Up/Gate Attention O

1.2

1.2

1.0

1.0

0.8

0.8

Probability

Probability

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0 1 2 3 4 5 6 7 8 9

0 1 2 3 4 5 6 7 8 9

Range Ratio

Range Ratio

(c) Qwen2.5-Omni-3B

(d) Qwen2.5-VL-3B

- Figure 4. Percentage of unified smoothing factors from different modalities using SmoothQuant on Omni and VL MLLMs.

0 5 10 15 20 25 30 35

0.4

0.6

EffectiveRankRatio

Self-Attention Query (self_attn.q)

white

no_white

0 5 10 15 20 25 30 35

0.4

0.6

EffectiveRankRatio

Self-Attention Output (self_attn.o)

white

no_white

0 5 10 15 20 25 30 35 Layer

0.25

0.50

0.75

EffectiveRankRatio

MLP Up Projection (mlp.up)

white

no_white

(a) Qwen2.5-VL-3B

0 5 10 15 20 25 30 35

0.4

0.6 EffectiveRankRatio

Self-Attention Query (self_attn.q)

white

no_white

0 5 10 15 20 25 30 35

0.4

0.6

EffectiveRankRatio

Self-Attention Output (self_attn.o)

white

no_white

0 5 10 15 20 25 30 35 Layer

0.25

0.50

0.75

EffectiveRankRatio

MLP Up Projection (mlp.up)

white

no_white

(b) Qwen2.5-Omni-3B

- Figure 5. Effective ranks of ∆W is reduced across layers after SVD-based Whitening b SQNR improves as the rank ratio increases on both Qwen2.5-VL-3B and Qwen2.5-Omni-3B.

The Effect of MAS. Table 3 reveals the necessity of modality-aware smoothing in W4A8 quantization. The most striking result is on LibriSpeech: uniform smoothing yields 77.4 WER while MAS achieves 3.8 WER. This dramatic gap exposes a fundamental failure mode where uniform smoothing catastrophically degrades the weaker modality when forced to compromise between modalities with disparate activation ranges. Notably, learnable optimization amplifies rather than closes this gap.

Modality Loss Weight. Table 4 shows ablation results of λt and λv on Qwen2.5-VL-3B. Equal smoothing works best: 17.2 PPL and 56.9% average accuracy. Reducing vision smoothing to λv=0.5 drops MMMU from 37.8 to 30.0 with minimal text benefit. Further reducing to 0.1 collapses PPL to 33.7. Reducing text smoothing (λt=0.5) degrades all metrics. In this work, we set λt=λv=1.0.

Table 3. Effects of Modality-Aware Smoothing (W4A8).

Omni-3B VL-3B Method Opt. Libri OmniB. T-VQA. MMMU

Uniform 77.4 27.3 51.3 25.6 MAS 3.8 36.7 54.7 28.9

Uniform ✓ 6.0 33.3 65.0 33.3 MAS ✓ 3.6 47.7 68.2 46.7

Table 4. Effects of Modality Loss Weight (W4A8).

Qwen2.5-VL-3B

λt λv PPL MMMU T-VQA. S-QA Avg

|1.0 1.0|17.2<br><br>|37.8 55.2 77.8 56.9|
|---|---|---|
|1.0 0.5|17.8<br><br>|30.0 57.8 80.0 55.9|
|1.0 0.1<br><br>|33.7|32.2 65.4 76.7 58.1|
|0.5 1.0<br><br>|17.3|33.3 53.9 68.9 52.0|

Training Epochs. Table 5 shows convergence on Qwen2.5-VL-3B. Performance improves rapidly: PPL drops from 23.9 to 17.0. Average accuracy peaks at epoch 2, then slightly declines. We use 2 epochs—a good trade-off between efficiency and performance.

Table 5. Effects of Modality-Aware Smoothing (W4A8).

Qwen2.5-VL-3B Epoch PPL MMLU T-VQA. S-QA Avg

|1|23.9<br><br>|38.9 52.2 79.5 56.9|
|---|---|---|
|2|18.6<br><br>|37.8 67.5 78.4 61.2|
|5|17.6<br><br>|34.4 65.9 77.4 59.2|
|10|17.0<br><br>|37.8 68.2 76.5 60.8|

The Effect of CMC. Figure 6 plots SQNR vs. rank ratio. CMC dominates the Non-Whitened baseline in the lowrank regime, reducing the rank required to match MBQ by 4x. Specifically, CMC surpasses MAS at ratio of 0.08 and saturates near 3.5 SQNR, whereas the baseline struggles to match MBQ until a ratio of 0.4.

Table 6. Computation cost and memory at decoding phase, where d is hidden size, r is rank, and m is the number of extra modalities.

|Base modality|Computation Memory<br><br>|
|---|---|
|Text<br><br>|d 0|
|Others|d + 2rd 2mrd|

Base Modality Choice. We choose text as the base modality to decouple CMC from decoding. As shown in Table 6, other modalities would require CMC involvement, leading to extra costs of computation and memory.

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

- 0

- 1

- 2

- 3

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

SQNR

| |
|---|

| |
|---|

| |
|---|

CMC

Non-Witened

MAS MBQ

1

0.0 0.2 0.4 0.6 0.8 1.0 Rank Ratio

Figure 6. SQNR as a function of rank ratio for CMC on Qwen2.5Omni-3B under W4A6 quantization. MAS applies independent modality-specific smoothing. MBQ employs a uniform factor optimized by modality balance reconstruction.

- 5.5. Inference Speed Up To validate practical efficiency, we implement a custom CUDA kernel based on Nunchaku [15] that fuses projection and quantization operators to minimize memory access. A multi-modal mask efficiently manages conditional lowrank execution. Table 7 shows that MASQuant achieves a 2.5x speedup over FP16 with marginal latency overhead (5– 10%) compared to MBQ, while maintaining identical decoding latency.

Table 7. End-to-end prefill-stage performance of Qwen2.5-VL-7B on Desktop RTX 4090 with fused GPU kernels (sequence length = 2048) under W4A4 setting. MAS: MASQuant. BS: Batch Size.

BS Method Rank Prefill Speedup Mem Mem

Ratio (ms) (GB) Saving

| |FP16<br><br>|-<br><br>|191.82 /|13.73 /|
|---|---|---|---|---|
|1<br><br>|MBQ MAS MAS MAS<br><br>|0.01 0.02 0.05<br><br>|68.65 2.79×<br><br>71.62 2.67×<br>72.89 2.63× 77.10 2.49×<br>|4.85 2.83×<br><br>4.97 2.76×<br><br>5.04 2.73×<br><br><br>5.37 2.56×<br>|
| |FP16|-|2146.01 /<br><br>|17.27 /|
|8|MBQ MAS MAS MAS<br><br>|0.01 0.02 0.05<br><br>|643.68 3.33× 649.29 3.30× 657.68 3.26× 696.44 3.07×|8.89 1.94×<br><br>9.02 1.92×<br><br><br>9.08 1.90× 9.42 1.83×<br><br>|

- 6. Conclusion

In this work, we identify smoothing misalignment as the primary obstacle preventing channel-wise smoothing from applying to MLLMs. To address this, we propose the MASQuant which applies Modality-aware Smoothing and Cross-Modal Compensation to support modality-specific smoothing factors and single quantized weight during inference. Our approach is simple and highly effective. We demonstrate its efficacy on multi-modal benchmark.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [2] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. Quarot: Outlierfree 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37:100213–100240, 2024. 2
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 6
- [4] Yelysei Bondarenko, Riccardo Del Chiaro, and Markus Nagel. Low-rank quantization-aware training for llms. arXiv preprint arXiv:2406.06385, 2024. 2
- [5] Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M De Sa. Quip: 2-bit quantization of large language models with guarantees. Advances in Neural Information Processing Systems, 36:4396–4429, 2023. 2
- [6] Mengzhao Chen, Wenqi Shao, Peng Xu, Jiahao Wang, Peng Gao, Kaipeng Zhang, and Ping Luo. Efficientqat: Efficient quantization-aware training for large language models. arXiv preprint arXiv:2407.11062, 2024. 2
- [7] Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. Spqr: A sparse-quantized representation for near-lossless llm weight compression. arXiv preprint arXiv:2306.03078, 2023. 2
- [8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 6
- [9] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407,

2024. 1

- [10] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022. 2
- [11] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025. 1
- [12] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham.

- Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617, 2018. 7
- [13] Jacob Kahn, Morgane Riviere, Weiyi Zheng, Evgeny Kharitonov, Qiantong Xu, Pierre-Emmanuel Mazar´e, Julien Karadayi, Vitaliy Liptchinsky, Ronan Collobert, Christian Fuegen, et al. Libri-light: A benchmark for asr with limited or no supervision. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7669–7673. IEEE, 2020. 1
- [14] Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W Mahoney, and Kurt Keutzer. Squeezellm: Dense-and-sparse quantization. arXiv preprint arXiv:2306.07629, 2023. 2
- [15] Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by lowrank components for 4-bit diffusion models. arXiv preprint arXiv:2411.05007, 2024. 8
- [16] Shiyao Li, Yingchun Hu, Xuefei Ning, Xihui Liu, Ke Hong, Xiaotao Jia, Xiuhong Li, Yaqi Yan, Pei Ran, Guohao Dai, et al. Mbq: Modality-balanced quantization for large visionlanguage models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4167–4177, 2025. 1, 2, 3, 4, 6
- [17] Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, et al. Omnibench: Towards the future of universal omni-language models. arXiv preprint arXiv:2409.15272,

2024. 1, 7

- [18] Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei. Duquant: Distributing outliers via dual transformation makes stronger quantized llms. Advances in Neural Information Processing Systems, 37:87766–87800, 2024. 2
- [19] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings of machine learning and systems, 6:87– 100, 2024. 1, 2, 3, 6
- [20] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024. 7
- [21] Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. Llm-qat: Data-free quantization aware training for large language models. arXiv preprint arXiv:2305.17888, 2023. 2
- [22] Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. Spinquant: Llm quantization with learned rotations. arXiv preprint arXiv:2405.16406, 2024. 2

- [23] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521,

2022. 7

- [24] Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, Xuefeng Xiao, Rui Wang, Shilei Wen, Fei Chao, and Rongrong Ji. Affinequant: Affine transformation quantization for large language models. arXiv preprint arXiv:2403.12544, 2024. 1, 2
- [25] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 5206–5210. IEEE, 2015. 7
- [26] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023. 6
- [27] Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. Omniquant: Omnidirectionally calibrated quantization for large language models. arXiv preprint arXiv:2308.13137, 2023. 1, 2, 3
- [28] Tao Sheng, Chen Feng, Shaojie Zhuo, Xiaopeng Zhang, Liang Shen, and Mickey Aleksic. A quantization-friendly separable convolution for mobilenets. In 2018 1st workshop on energy efficient machine learning and cognitive computing for embedded applications (EMC2), pages 14–18. IEEE,

2018. 4

- [29] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 1
- [30] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 7
- [31] Yuxuan Sun, Ruikang Liu, Haoli Bai, Han Bao, Kang Zhao, Yuening Li, Jiaxin Hu, Xianzhi Yu, Lu Hou, Chun Yuan, et al. Flatquant: Flatness matters for llm quantization. arXiv preprint arXiv:2410.09426, 2024. 1, 2
- [32] Qwen Team. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 1
- [33] Mart Van Baalen, Andrey Kuzmin, Ivan Koryakovskiy, Markus Nagel, Peter Couperus, Cedric Bastoul, Eric Mahurin, Tijmen Blankevoort, and Paul Whatmough. Gptvq: The blessing of dimensionality for llm quantization. arXiv preprint arXiv:2402.15319, 2024. 4
- [34] Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. Svd-llm: Truncation-aware singular value decomposition for large language model compression. arXiv preprint arXiv:2403.07378, 2024. 2

- [35] Xin Wang, Samiul Alam, Zhongwei Wan, Hui Shen, and Mi Zhang. Svd-llm v2: Optimizing singular value truncation for large language model compression. arXiv preprint arXiv:2503.12340, 2025. 2
- [36] Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. Outlier suppression: Pushing the limit of low-bit transformer language models. Advances in Neural Information Processing Systems, 35:17402–17414, 2022. 1
- [37] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International conference on machine learning, pages 38087–

38099. PMLR, 2023. 1, 2, 3, 6

- [38] Jingjing Xie, Yuxin Zhang, Mingbao Lin, Liujuan Cao, and Rongrong Ji. Advancing multimodal large language models with quantization-aware scale learning for efficient adaptation. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 10582–10591, 2024. 2
- [39] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025. 1, 6
- [40] Zhewei Yao, Xiaoxia Wu, Cheng Li, Stephen Youn, and Yuxiong He. Zeroquant-v2: Exploring post-training quantization in llms from comprehensive study to low rank compensation. arXiv preprint arXiv:2303.08302, 2023. 2
- [41] JiangYong Yu, Sifan Zhou, Dawei Yang, Shuo Wang, Shuoyu Li, Xing Hu, Chen Xu, Zukang Xu, Changyong Shu, and Zhihang Yuan. Mquant: Unleashing the inference potential of multimodal large language models via full static quantization. arXiv preprint arXiv:2502.00425, 2025. 1, 2, 3
- [42] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 1, 7
- [43] Binbin Zhang, Hang Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xiaoyu Chen, Chenchen Zeng, et al. Wenetspeech: A 10000+ hours multi-domain mandarin corpus for speech recognition. In ICASSP 20222022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6182–6186. IEEE,

2022. 1, 7

- [44] Weibo Zhao, Yubin Shi, Xinyu Lyu, Wanchen Sui, Shen Li, and Yong Li. Aser: activation smoothing and error reconstruction for large language model quantization. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 22822–22830, 2025. 2
- [45] Zhen Zheng, Xiaonan Song, and Chuanjie Liu. Mixllm: Llm quantization with global mixed-precision between outputfeatures and highly-efficient system design. arXiv preprint arXiv:2412.14590, 2024. 2

