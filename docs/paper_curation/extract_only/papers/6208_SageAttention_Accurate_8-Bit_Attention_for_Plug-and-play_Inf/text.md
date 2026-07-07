# arXiv:2410.02367v9[cs.LG]1Oct2025

#### SAGEATTENTION: ACCURATE 8-BIT ATTENTION FOR PLUG-AND-PLAY INFERENCE ACCELERATION

Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen∗ Dept. of Comp. Sci. & Tech., Institute for AI, BNRist Center, Tsinghua-Bosch Joint ML Center, THBI Lab, Tsinghua University {zhang-jt24@mails., jianfeic@, dcszj@}tsinghua.edu.cn

ABSTRACT

The transformer architecture predominates across various models. As the heart of the transformer, attention has a computational complexity of O(N2), compared to O(N) for linear transformations. When handling large sequence lengths, attention becomes the primary time-consuming component. Although quantization has proven to be an effective method for accelerating model inference, existing quantization methods primarily focus on optimizing the linear layer. In response, we first analyze the feasibility of quantization in attention detailedly. Following that, we propose SageAttention, a highly efficient and accurate quantization method for attention. The OPS (operations per second) of our approach outperforms FlashAttention2 and xformers by about 2.1x and 2.7x, respectively. SageAttention also achieves superior accuracy performance over FlashAttention3. Comprehensive experiments confirm that our approach incurs almost no end-to-end metrics loss across diverse models—including those for large language processing, image generation, and video generation. The code is available at https://github.com/thu-ml/SageAttention.

FlashAttention2

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

2x 331

164

SageAttention

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

90s 1.35x

67s

Figure 1: An example of SageAttention on video generation (CogvideoX on RTX4090).

1 INTRODUCTION

Attention is the fundamental component of transformers (Vaswani, 2017), and efficiently computing attention is crucial for transformer-based applications. Moreover, there is a recent trend in processing longer sequences, which further strengthens the need for faster attention. In tasks like video generation (Yang et al., 2024) and language model prefilling (Dubey et al., 2024), the sequence length can easily go up to 8K∼ 128K. Due to its quadratic complexity, the cost of attention dominates all other operations in such scenarios, as illustrated in Figure 2.

Quantization is an effective strategy for enhancing neural networks’ computational and memory efficiency by reducing the numerical precision. There are abundant works on accelerating training (Sun et al., 2019; Xi et al., 2024b; Peng et al., 2023) and inference (Jacob et al., 2018; Xiao et al., 2023a)

∗Corresponding author.

Latency of operators in a transformer model (NumHead=32, HeadDim=64)

800

Attention

Linear

Latency(ms)

600

Others

400

200

0

| |
|---|

1K 2K 4K 8K 16K 32K

Sequence Length

Figure 2: Latency of attention.

[Figure 11]

FP16 Quantize

Q, K, P, V to INT8 directly

[Figure 12]

[Figure 13]

Use FlashAttention3

[Figure 14]

Use SageAttention

Figure 3: A comparison example.

with low-precision numerical formats such as FP8, INT8, or INT4. However, existing works primarily focused on quantizing the linear layer, where attention is left unaccelerated in high-precision, such as FP16. There is not yet a work that systematically investigates the quantization of attention. Moreover, many quantization methods require extra training, and the cost can be prohibitive for large-scale models. While FlashAttention3 (Shah et al., 2024) was released recently and offers an FP8 version, it is tailored to and can only be used with the Nvidia Hopper architecture. This exclusive optimization limits its broader applicability. Furthermore, our analysis demonstrates that directly implementing the FP8 version can lead to performance degradation, as detailed in Table 1.

Quantizing attention is challenging. The computation of attention is more complex than that of linear operations. Attention includes a softmax operation and two matrix multiplication (Matmul) operations: QK⊤ and PV . Direct 8-bit quantization and dequantization of the matrices (Q,K,P,V ) in attention will result in significantly degraded performance across various models. For example, the text-to-image model Unidiffuser (Bao et al., 2023) will generate a completely blurry image with both INT8 and FlashAttention3’s FP8 implementation (See Figure 3), and Llama2 only achieves a random-guessing-level accuracy of 25.5% on the MMLU dataset with INT8 attention. After investigating deeply, we identified two primary challenges: (C1) The matrix K exhibits a significant channel-wise outlier, leading to substantial accuracy loss during quantization. (C2) Simply quantizing (P,V ) into INT8 does not consistently ensure the accuracy of PV across various scenarios.

In this paper, we propose SageAttention, a quantization method to accelerate attention while preserving accuracy. SageAttention is easy-to-use. As a post-training quantization method, it can be used in a plug-and-play manner in inference time by simply replacing the original highprecision implementation. We propose several techniques to achieve this goal. First, we opt to quantize the tensors in attention to INT8 rather than FP8. This decision is based on the fact that INT8 Matmul on some commonly used GPUs, e.g., RTX4090 and 3090, are four times faster than in FP16 and two times faster than FP8. Moreover, INT8 quantization for matrices (Q,K) is more precise than FP8 in attention (See Table 2). To address (C1), we propose a method to smooth the K matrix. This method significantly enhances accuracy with a negligible time overhead (<0.2%). To address (C2), as an alternative to quantizing (P,V ) to 8-bit, we propose a more accurate yet efficient method for the Matmul PV : we maintain (P,V ) in FP16 and use a low-precision FP16 accumulator. This strategy doubles Matmul’s speed without sacrificing any accuracy. Finally, we implement several versions of attention with different speed-accuracy tradeoffs and propose a method to select the fastest attention implementation for each layer while preserving accuracy.

We offer a high-performance implementation of SageAttention on RTX4090 and 3090 GPUs using Triton (Tillet et al., 2019). Our implementation contains a fused kernel combining ROPE with quantization and a fast self-attention kernel inspired by FlashAttention-style tiling. The implementation utilizes the fast INT8 mma(u8.u8.s32) and FP16-with-FP16-accumulator mma(f16.f16.f16) instructions of Nvidia Tensor Core. Our kernel is about 2.1× and 2.7× faster than FlashAttention2 and xformers, respectively. Notably, it achieves 340 TOPS on RTX4090 at headdim=64 and headdim=128, reaching 52% of the theoretical INT8 throughput. In contrast, the peak for the stateof-the-art FlashAttention2 is only 165 TOPS. Moreover, at headdim=64, our throughput on RTX 4090 is even close to the 490 TOPS throughput of FlashAttention3, which is exclusive to the much more powerful and expensive Hopper GPUs. We extensively evaluate the end-to-end metrics of our approach on state-of-the-art image/video generation, image classification, and language models. On all tasks, SageAttention can be directly adopted in a plug-and-play manner with negligible loss in model performance, while offering more than 2× speedup than FlashAttention2 and xformers.

- 2 RELATED WORK

We categorize efficient Attention works into three groups: (1) Sparse Attention. This strategy only selects parts of a sequence from a given context for processing with standard Attention. Implementations like Swin transformer (Liu et al., 2021), Twins (Chu et al., 2021), UniFormer (Li et al.), Attentionsinks (Xiao et al., 2023b), InfLLM (Xiao et al., 2024), LongLora (Chen et al., 2023), Minference (Jiang et al., 2024), and SkipAttention (Venkataramanan et al., 2023) show promise. However, these methods’ limitations are that they only work in a few scenarios because omitted calculations are not always useless. (2) Linear Attention. Techniques that transform Attention computation to reduce time complexity, for example, Linformer (Wang et al., 2020), Performer (Choromanski et al., 2020), MetaFormer (Yu et al., 2022), and LinearAttention (Katharopoulos et al., 2020), which lower the time complexity of Attention from O(N2) into O(N). These methods excel in specific scenarios while standard Attention remains prevalent. (3) Kernel Optimization. Rather than simplifying calculations, these methods exploit hardware capacities to enhance speed. The xformers (Lefaudeux et al., 2022) platform accelerates Attention with customizable blocks and dedicated CUDA kernels. FlashAttention (Dao et al., 2022) proposes tiling to reduce the memory reads/writes between GPU global memory and on-chip SRAM for significant speedups. FlashAttention2 (Dao, 2023) refine the parallelism and warps partition of FlashAttention. Bikshandi & Shah (2023) further optimize FlashAttention2 by kernel fusion. FlashAttention3 (Shah et al., 2024) is proposed for Hopper architecture. However, FlashAttention3 is exclusive to the Hopper GPU architecture, and the accuracy of its quantization version is significantly lower than our method (See Table 1). RingAttention (Liu et al.) scales FlashAttention across multiple GPUs. I-bert (Kim et al., 2021) quantizes all tensors in a transformer block into INT8 but is restricted to RoBERTa. Our method falls under the third category, and is orthotopic with the first and second categories.

- 3 PRELIMINARY

Our method builds on FlashAttention-2 and adopts dynamic quantization. We will begin by reviewing FlashAttention-2, followed by a brief introduction to dynamic quantization techniques.

- 3.1 FLASHATTENTION

The computation of self-attention can be formulated as follows: S = QK⊤/

√

d, P = σ(S), O =

PV , where σ(S)ij = exp(Sij)/ k exp(Sik) is the softmax operation. The matrices Q, K, and V each have dimensions N × d, while the matrices S, P are N × N. While d is typically small,

e.g., 64 or 128, N can be thousands if not millions. Therefore, the N ×N matrices (S,P) are much larger than (Q,K,V ), and a naive implementation suffers from the huge amount of global memory I/O for (S,P) reads/writes. FlashAttention (Dao, 2023) proposes to tile Q, K, and V from the token dimension into blocks {Qi},{Ki},{Vi} with block sizes of bq, bkv, bkv, respectively. Then, to avoid the memory I/O for (S,P), it uses online softmax (Milakov & Gimelshein, 2018) to progressively compute each block of O, i.e., Oi as follows.

First, for each block of {Ki},{Vi}, it computes the following equations iteratively:

Sij = QiKj⊤/

√

d, (mji, Pij) = σ˜(mji−1,Sij), (1) lij = exp(mji−1 − mji)lij−1 + rowsum( Pij), Oij = diag exp(mji−1 − mji) Oij−1 + PijVj (2)

Where mji and lij are bq ×1 vectors, which are initialized to −∞ and 0 respectively. σ˜() is an online softmax operator: mji = max{mji−1,rowmax(Sij)}, Pji = exp(Sij − mji).

Finally, after all iterations, i.e., j = bkv, the output Oi can be computed by Oi = diag(lij)−1Oij.

- 3.2 DYNAMIC QUANTIZATION A matrix multiplication C = AB can be accelerated with quantization as:

(δA,Aˆ) = ψ(A), (δB,Bˆ) = ψ(B), Cˆ = AˆB,ˆ C = ψδ−1

AδB(Cˆ) (3)

Here, ψ is a quantizer which converts a high-precision (e.g., FP32) matrix A to a low-precision format Aˆ (e.g., INT8 or FP8) with a scale δA, and ψ−1 is a dequantizer to convert back to highprecision. We should have ψδ−1

(Aˆ) ≈ A. The actual matrix multiplication AˆBˆ is carried in lowprecision. In modern GPUs, low-precision matrix multiplication is usually multiple times faster than higher-precision ones.

A

Many quantizers depend on the numerical format and granularity, e.g., how many elements share a common scale factor. For example, an INT8 per-tensor dynamic quantizer first computes the scale as the maximum absolute value of the entire tensor, scales the elements to the maximum representable range of INT8 [-127, +127], and then casts to INT8 with rounding: Aˆ = ⌈A/δA⌋,δA = max(|A|)/127. Likewise, per-token quantizer assigns a scale factor for each token of a tensor: Aˆ[i,:] = ⌈A[i,:]/δA⌋,δA[i,:] = max(|A[i,:]|)/127. Also, per-channel quantizer assigns a scale factor for each channel of the tensor, i.e., along the channel dimension: A[:,i] = ⌈A[:,i]/δA⌋,δA = max(|A[:,i]|)/127. Based on the tiling approach of FlashAttention, we can apply per-block quantization correspondingly. per-block quantizer asigns a scale factor for every b = m − n tokens: Aˆ[m : n,:] = ⌈A[m : n,:]/δA⌋,δA = max(|A[m : n,:]|)/127. Dequantization simply involves a element-wise scaling: ψδ−1

(Aˆ) = δAAˆ.

A

- 4 SAGE ATTENTION

In this section, we propose SageAttention, a fast yet accurate method to accelerate attention computation with 8-bit quantization. Considering that most networks are not natively trained with quantized attention, SageAttention is designed to be plug-and-play.

Unlike linear layers, which are easy to quantize, quantizing attention is more complicated. Extra treatment is required to ensure both good accuracy and fast speed. First, we will formulate quantized attention in Section 4.1, followed by introducing our approach.

- 4.1 FORMULATION

Based on the description of FlashAttention and dynamic quantization in Section 3.1 and 3.2, we formulate the quantized attention as follows.

√

Quantization: (δQ, Qˆ) = ψQ(Q/

d), (δK, Kˆ) = ϕK(K), (δP, Pˆ) = ψP( P), (δV , Vˆ) = ψV (V ) (4) Attention: S = ψδ−1

QδK(QˆKˆ⊤), (m′, P) = σ˜(m, S), O = diag exp(m′ − m) O + ψδ−1

PδV (PˆVˆ) (5)

ϕK is a transformation to obtain quantized K, which we shall discuss in subsequent sections. For simplicity, we omit all superscripts and subscripts, but the matrices used in attention are still tiles, and the computation is still organized as FlashAttention described in Section 3.1. Compared to the original full-precision version, as shown in Eq. 4, 5, SageAttention adds quantizers to Q,K,P,V and dequantizers to the product to accelerate both Matmuls of QK⊤ and PV . Online softmax is left in full-precision.

###### Q

K

###### V

###### Q

K

###### V

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

Channel Token

Token

Token

Token

Token

Token

Channel

Channel

Channel

Channel

Channel

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

-11 14 -16 12 -4 5 -210 208 -151 147 -2 3

(a) QKV distribution in Cogvideo

(b) QKV distribution in Unidiffuser

Figure 4: Typical examples of data distribution of (Q, K, V).

Table 1: End-to-end metrics comparison of different quantization methods.

|Quantization (Q, K)<br><br>|Smoothing K|Llama WikiText ↓|CogVideo (Fscore) ↑<br><br>|Unidiffuser (FID) ↓|UltraPixel (FID) ↓<br><br>|TIMM ImageNet ↑|
|---|---|---|---|---|---|---|
|Full-Precision|-|5.823<br><br>|3.768<br><br>|163.33|179.78<br><br>|84.79%|
|Per-token|✗ ✓<br><br>|5.824 5.824<br><br>|1.924 3.734<br><br>|221.18 166.52<br><br>|193.36 179.79<br><br>|84.21% 84.74%<br><br>|
|Per-block|✗ ✓<br><br>|5.825 5.824<br><br>|2.014 3.718<br><br>|229.08 166.93<br><br>|195.67 179.98<br><br>|84.18% 84.76%<br><br>|
|Per-tensor<br><br>|✗ ✓<br><br>|5.826 5.824<br><br>|1.902 3.640<br><br>|267.06 167.65<br><br>|196.26 180.21<br><br>|84.12% 84.69%<br><br>|
|FlashAttn3 (with quant)| |5.850|3.394|394.13<br><br>|383.61<br><br>|84.70%|

- 4.2 SMOOTH MATRIX K

Directly quantizing Q,K often results in a large error. Particularly, quantizing Q,K to INT8 yields completely blurry image/video in text-to-image/video tasks. As shown in Figure 4.1, we visualize two typical groups of Q,K,V from a text-to-image model Unidiffuser (Bao et al., 2023) and a text-to-video model CogvideoX (Yang et al., 2024). Notably, K exhibits distinct channel-wised outliers. However, per-channel quantization cannot be applied for K, because quantization can only be performed at the outer axis (token dim) of the Matmul QK⊤. Moreover, the previous smoothing technique proposed for linear layers (Xiao et al., 2023a) cannot be applied since Q is also heavily affected by outliers. Fortunately, the channel outliers of K have a pattern: Each token’s key is actually a large bias shared by all tokens, plus a small token-wise signal. Therefore, the outlier is not from large variation across tokens, but simply the large bias. Based on this observation, we propose to smooth the matrix K by a transform γ, which subtracts averaged K across all tokens:

γ(K) = K − mean(K) (6)

where mean(K) = N1 Nt=1 K[t,:] is the average key, with a shape 1 × d. Note that such a transformation does not change the attention score P, because for any query q, we have

σ(q(K − mean(K)⊤)) = σ(qK⊤ − q · mean(K)) = σ(qK⊤). Finally, the transformation from full-precision K to quantized Kˆ can be written as ϕK(K) = ψK ◦ γ, where ψK is a quantizer. In other words, a full-precision K is substracted with the mean, before eventually being quantized.

- Table 1 presents end-to-end metrics for different quantization methods with and without smoothing K on various models. The results demonstrate that smoothing K offers significant benefits of accuracy. Moreover, the speed overhead of smoothing K for attention is less than 0.2% (See Table10).
- Table 2: Average accuracy using different data types across all layers of real models.

|Q,K|P,V<br><br>|Cos Sim ↑|Relative L1 ↓<br><br>|RMSE ↓|
|---|---|---|---|---|
|INT8<br><br>|E4M3<br>E5M2 INT8<br>|99.94% 99.81% 99.70%<br><br>|0.0345 0.0572 0.1035<br><br>|3.53e-3 6.11e-3 6.82e-3|
|E4M3|E4M3<br><br>E5M2<br><br><br>INT8<br><br>|99.81% 99.68% 99.58%|0.0607 0.0769 0.1199<br><br>|5.93e-3<br><br>7.72e-3<br><br>8.31e-3<br>|
|E5M2<br><br>|E4M3<br>E5M2 INT8<br>|99.37% 99.22% 99.13%<br><br>|0.1107 0.1213 0.1583|1.09e-2 1.20e-2 1.24e-2<br><br>|

Table 3: Worst accuracy using different data types across all layers of real models.

|Q,K<br><br>|P,V|Cos Sim ↑|Relative L1 ↓<br><br>|RMSE ↓|
|---|---|---|---|---|
|INT8<br><br>|E4M3<br>E5M2 INT8 FP16<br>|76.36% 78.98% 56.40% 99.99%<br><br>|0.5899 0.4233 0.7921 0.0116<br><br>|0.4311 0.4371 0.5405 0.0091|
| | | | | |

- 4.3 QUANTIZATION FOR Q, K, P, V

Quantization granularity for Q,K: ψQ(Q) and ψK(K) can be set with the granularity of pertoken, per-block or per-tensor. This is because per-channel quantization is not feasible, since the scale factors of the inner axis of QK⊤ cannot be used to do dequantization (Xiao et al., 2023a).

Data type of Q,K: We choose INT8 for ψQ(Q) and ψK(K) for two reasons. First, Table 2 shows the average accuracy using different data types (INT8, E4M3, E5M2) for Q,K, P,V across all layers of Llama2 (7B) (Touvron et al., 2023) and Unidiffuser. It shows that quantizing Q,K to INT8 performs higher accuracy than using E4M3 and E5M2. Second, Matmul using INT8 is two times faster than using FP8 in many commonly used GPUs, e.g., RTX4090 and 3090.

Quantization granularity for P,V : We propose to use ψP( P) in per-block and ψV (V ) in perchannel for three reasons. (1) Per-channel quantization for P and per-token quantization for V

are not viable because dequantization requires scale factors of the outer axis. (2) P = exp(Si − rowmax(Si)), where Si is the Matmul result of a block of Q and KT, the max value in each row

of P is 1. Hence, we can assign a single static scale s = 1271 to a block P, whose accuracy equals per-token quantization. (3) Per-channel quantization can address the channel-wised outlier of V .

Data type of P,V : We choose INT8 for ψP( P) and ψV (V ) because Matmul using INT8 is two times faster than using FP8 in some commonly used GPUs, and although the accuracy using ψP( P) and ψV (V ) in INT8 is worse than E4M3 and E5M2, the average accuracy is similar (See Table 2).

Accuracy metrics. We use three metrics to assess the accuracy of quantized attention output O′ compared to attention output in full-precision O: First, we flatten O′ and O into vectors in the shape of 1 × n. Then, Cosine Sim= OO′/ O2 O′2, Relative L1= |O − O′|/ |O|, RMSE= (1/n) (O − O′)2.

Table 5: Worst accuracy using different accumulators across all layers of real models.

Table 4: Average accuracy using different accumulators across all layers of real models.

|Accum.|Cos Sim ↑|Relative L1 ↓<br><br>|RMSE ↓|
|---|---|---|---|
|FP32 FP16<br><br>|99.84% 99.84%<br><br>|0.0511 0.0511<br><br>|4.229e-3 4.229e-3<br><br>|

|Accum.<br><br>|Cos Sim ↑<br><br>|Relative L1 ↓<br><br>|RMSE ↓|
|---|---|---|---|
|FP32 FP16<br><br>|99.98% 99.98%<br><br>|0.0156 0.0156<br><br>|2.94e-3 2.94e-3<br><br>|

- 4.4 FP16 ACCUMULATOR: MUCH MORE ACCURATE AND EFFICIENT SOLUTION

The above solution for ψP( P) and ψV (V ) has one problem, that is, the accuracy using INT8 is very poor in some model layers. Table 3 shows the worst accuracy using different data types for

Q,K, P,V across all layers of Llama2 and Unidiffuser. It shows that INT8 ψP( P) and ψV (V ) bring an unacceptable error. In response, we propose a very accurate and also efficient solution. Specifically, we propose to use FP16 as the data type of Matmul PV with an FP16 accumulator.

The benefit of such a solution is obvious. First, in the context of some commonly used GPUs, e.g., RTX4090 and 3090, the speed of Matmul in FP16 with an FP16 accumulator is 2x faster than that with an FP32 accumulator. Moreover, using FP16 accumulators can save more register resources than using FP32 accumulators, accelerating the computation speed. Second, Table 3 shows that using FP16 for P,V is much more accurate than using all the other 8-bit data types. Moreover, using FP16 accumulators incurs no accuracy loss than using FP32 accumulators. Specifically, Table 4 and 5 show the average and worst accuracy using FP16 or FP32 accumulators on all layers of Llama2 and Unidiffuser, showing that there is no accuracy loss of using the FP16 accumulator.

Table 6: Four kernel implementations of SageAttention.

|Kernel<br><br>|ψQ(Q), ψK(K)|ψP(P)<br><br>|ψV (V )|
|---|---|---|---|
|SAGEAttn-T SAGEAttn-B (Algorithm 1) SAGEAttn-vT (Figure 5(a)) SAGEAttn-vB<br><br>|per-token, INT8 per-block, INT8 per-token, INT8 per-block, INT8|FP16, FP16 Accumulator FP16, FP16 Accumulator per-block, INT8 per-block, INT8<br><br>|FP16, FP16 Accumulator FP16, FP16 Accumulator per-channel, INT8 per-channel, INT8<br><br>|

- 4.5 ADAPTIVE QUANTIZATION

Based on the discussion in Section 4.3 and 4.4, we implement four attention kernels (See Table 6) based on two sets of choices: (1) Using ψQ(Q) and ψK(K) in per-token or per-block.

|FP32<br><br>FP32<br><br>sq<br><br>SAGEAttn-vT Loop for K,V to get a block of O<br><br>QINT8<br><br>|Global Memory|
|---|
|On Chip|
<br><br>KINTT 8<br><br>SM1 SM2<br><br>|SM1<br><br>·<br><br>S =( )⇥ ⇥<br><br>Pe = (Pe ⇤ 127).to(INT8) O = (P )⇥<br><br>Pe = OnlineSoftmax(S)|
|---|
<br><br>O ·<br><br>sk VINT8|FP32<br><br>FP32<br><br>sq<br><br>Loop for K, V to get a block of O<br><br>|On Chip|
|---|
<br><br>|Global Memory|
|---|
<br><br>KINTT 8 VFP16<br><br>QINT8<br><br>sk<br><br>SM1 SM2<br><br>|SM1<br><br>·<br><br>S =( )⇥ ⇥<br><br>,Accum. = FP16)<br><br>Pe = OnlineSoftmax(S)<br><br>Pe = P.toe (FP16) O = (P|
|---|
<br><br>O ·<br><br>SAGEAttn-B|
|---|---|

###### e e

(a) SageAttention (per-token quantize Q,K; INT8 V) (b) SageAttention (per-block quantize Q,K; FP16 V)

Figure 5: Workflow of SageAttention.

Algorithm 1: Implementation of SAGEAttn-B.

Input: Matrices Q(FP16), K(FP16), V (FP16) ∈ RN×d, block size bq, bkv. Preprocessing: K = K − mean(K) ; // Subtracting the mean value across tokens

√

d), (δK, Kˆ) = ψK(K) ; // INT8 per-block quant Divide Qˆ into Tm = N/bq blocks {Qˆi}, and divide Kˆ, V into Tn = N/bkv blocks {Kˆi} and {Vi}; for i in [1, Tm] do ; // Outer loop is paralleled in SMs (stream processors)

Quantization: (δQ, Qˆ) = ψQ(Q/

Load Qˆi and δQ[i] into a SM ; for j in [1, Tn] do

Load Kˆj, Vj, and δK[j] into the SM ; Sij = Matmul(Qˆi, KˆjT) × δQ[i] × δK[j]; mji = max(mji−1, rowmax(Sij)), Pij = exp(Sij − mji), lij = em

j−1

###### i −mji lij−1 + rowsum( Pij) ; Oij = diag(em

j−1

i −mji )Oij−1+ Matmul( Pij.to(FP16), Vj, Accum type = FP16) ;

Oi = diag(liTn)−1OiTn ; Write Oi ;

return O = {Oi};

(2) Using ψP( P) and ψV (V ) in INT8 or retaining P,V in FP16 with an FP16 accumulator. SAGEAttn-B is accurate enough for all models and can achieve a 2x speedup (See Figure 6 and 7). However, SAGEAttn-vB is also accurate for some layers in a model and faster a little (about 4%) than SAGEAttn-B. Therefore, we use various inputs to test the cosine similarity of SAGEAttn-vB for each layer of a model. Then, we will select SAGEAttn-vB for those layers where SAGEAttn-vB’s cosine similarity is bigger than 99.8% (the worst similarity of SAGEAttn-B), and the other layers are left for SAGEAttn-B.

- 4.6 FUSION TRICKS AND PERFORMANCE ANALYSIS

Fusion Tricks. To reduce the overhead of quantization, we fuse the quantization process with the operator preceding the attention layer. For instance, we fuse quantization within the ROPE (Rotary Position Embedding) (Su et al., 2021) layer. Specifically, before the ROPE result (A) is written from shared memory into global memory, we perform δA,Aˆ = ψ(A). Subsequently, the δA,Aˆ are written into global memory. Additionally, we also fuse the coefficient (1/

###### √

d) of QKT into the quantization process rather than leaving it in the attention layer. Specifically, we multiply Q by (1/

###### √

d) on chip before quantizating Q. Performance Analysis. We will take SAGEAttn-B as an example to discuss the acceleration effects on actual hardware: (1) Matmul acceleration. Utilizing INT8 matrix multiplication units on current mainstream hardware can achieve 2-4× throughput. While FP16 accumulators do not offer throughput improvements on most compute cards, on-edge accelerators, such as the RTX4090, can still achieve a 2x improvement over FP32 accumulators.

- (2) Quantization overhead. Quantization and dequantization are considered the main overhead in current quantization methods (Lin et al., 2024). The computational overhead can not be avoided, but through fusing the quantization of Q,K with ROPE, we avoid the IO overhead of quantization.

- (3) Cache and registers. Currently, mainstream accelerators need to store data in a cache (such as SharedMemory) during computation. Using 8-bit data for calculations can reduce the usage of the

- general cache, and using fp16 accumulators can also reduce the usage of accumulation registers.
- (4) Dram access. Using 8-bit data can halve the tensors transfer overhead from DRAM to the compute units. Although quantization introduces additional FP32 scales, these scales can be considered negligible compared to the tensors.

- 5 EXPERIMENTS

Main results. The speed of SageAttention is approximately 2.1× faster than FlashAttention2. Furthermore, SageAttention achieves an average real speedup of 2.83× compared to the original attention in various models, with negligible loss in end-to-end metrics.

- 5.1 EXPERIMENTAL SETUP

Models. We validate the effectiveness of SageAttention across a diverse set of representative models from the fields of language, image, and video generation. Specifically, we conduct experiments on five models: Llama2 (7B) (Touvron et al., 2023) for text2text, CogvideoX (Yang et al., 2024) for text2video, Unidiffuser (Bao et al., 2023) and UltraPixel (Ren et al., 2024) for text2image, TIMM (Wightman, 2019) for image classification, and Llava1.6 (Liu et al., 2024a) for visual question answering.

Datasets. Llama2 is evaluated on three zero-shot tasks: WikiText (Merity et al., 2022) to assess the model’s prediction confidence, LAMBADA (Paperno et al., 2016) evaluate contextual understanding, and MMLU (Hendrycks et al., 2020) for measuring knowledge across various subjects. CogvideoX is evaluated using the open-sora (Zheng et al., 2024c) prompt sets. Both UltraPixel and Unidiffuser are assessed on the COCO annotations (Lin et al., 2014), featuring (prompt, image) pairs. TIMM is evaluated on on three image datasets: ImageNet (Deng et al., 2009), ImageNet-Sketch (Sketch) (Wang et al., 2019), and ImageNet-Rendition (ImageNetr) (Hendrycks et al., 2021). Llava1.6 is evaluated on three datasets: TextVQA (Singh et al., 2019), POPE (Li et al., 2023b), and VQAv2 (Goyal et al., 2017).

Metrics. For Llama2, we use perplexity (ppl.) (Jelinek et al., 1977) for WikiText, and Accuracy (Acc.) for LAMBADA and MMLU. For CogvideoX, folowing (Zhao et al., 2024a), we evaluate the quality of generated videos on five metrics: CLIPSIM and CLIP-Temp (CLIP-T) (Liu et al., 2024b) to measure the text-video alignment; (VQA-a) and (VQA-t) to assess the video aesthetic and technical quality, respectively; and Flow-score (FScore) for temporal consistency (Wu et al., 2023). For UltraPixel and Unidiffuser, generated images are compared with the images in the COCO annotations dataset in three aspects: FID (Heusel et al., 2017) and sFID (Salimans et al., 2016) for fidelity evaluation, Clipscore (CLIP) (Hessel et al., 2021) for text-image alignment, and ImageReward (IR) (Xu et al., 2024) for human preference. For TIMM and Llava1.6, we use Accuracy.

RTX4090, (Head dim = 64, causal = False)

RTX4090, (Head dim = 64, causal = True)

600

600

Torch xformers

FlashAttn2 SageAttn-T

SageAttn-vT SageAttn-B

SageAttn-vB

Torch xformers

FlashAttn2 SageAttn-T

SageAttn-vT SageAttn-B

SageAttn-vB

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

Speed(TOPS)

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

400

400

341

340

339

339

335

329

329

326

325

325

323

319

318

315

312

311

310

307

307

307

307

306

305

304

304

303

298

296

292

292

283

280

280

276

271

270

267

249

245

239

237

226

217

179

177

167

167

167

164

162

161

159

157

153

152

150

143

200

200

128

127

126

125

125

124

124

123

123

117

116

106

94

88

OOM

OOM

OOM

OOM

OOM

OOM

28

26

22

9

8

7

0

0

1K 2K 4K 8K 16K 32K Sequence Length

1K 2K 4K 8K 16K 32K Sequence Length

- Figure 6: Speed comparison between SageAttention and baselines (RTX4090, headdim=64).

1K 2K 4K 8K 16K 32K Sequence Length

0

200

400

600

Speed(TOPS)

46

42

55

OOM

OOM

OOM

93

96

104

105

106

106

145

151

161

164

164

164

237

271

304

303

306

305

239

280

307

310

312

307

276

315

326

325

329

323

270

307

339

340

341

339

RTX4090, (Head dim = 128, Causal = False)

Torch xformers

| |
|---|

| |
|---|

FlashAttn2 SageAttn-T

| |
|---|

| |
|---|

SageAttn-vT SageAttn-B

| |
|---|

| |
|---|

SageAttn-vB

1K 2K 4K 8K 16K 32K Sequence Length

0

200

400

600

16

14

16

OOM

OOM

OOM

78

89

100

104

105

106

118

138

155

160

161

161

152

217

267

283

292

296

158

226

280

298

307

304

177

249

311

318

325

179

245

292

319

329

335

RTX4090, (Head dim = 128, Causal = True)

Torch xformers

| |
|---|

| |
|---|

FlashAttn2 SageAttn-T

| |
|---|

| |
|---|

SageAttn-vT SageAttn-B

| |
|---|

| |
|---|

SageAttn-vB

- Figure 7: Speed comparison between SageAttention and baselines (RTX4090, headdim=128).

RTX3090, (Head dim = 64, causal = False)

RTX3090, (Head dim = 64, causal = True)

Torch xformers

FlashAttn2 SageAttn-T

SageAttn-vT SageAttn-B

SageAttn-vB

Torch xformers

FlashAttn2 SageAttn-T

SageAttn-vT SageAttn-B

SageAttn-vB

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

200

200

Speed(TOPS)

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

137

136

135

135

134

134

134

134

134

133

133

133

133

132

130

128

128

126

126

126

126

125

125

124

124

123

122

122

121

121

118

115

114

112

- 110

124

125

124

126

- 111

102

100

97

94

100

100

83

81

69

69

68

67

66

64

63

62

62

62

58

54

54

54

54

53

53

53

52

50

50

49

49

46

OOM

OOM

OOM

OOM

OOM

OOM

24

19

14

8

6

5

0

0

1K 2K 4K 8K 16K 32K Sequence Length

1K 2K 4K 8K 16K 32K Sequence Length

- Figure 8: Speed comparison between SageAttention and baselines (RTX3090, headdim=64).

1K 2K 4K 8K 16K 32K Sequence Length

0

100

200

Speed(TOPS)

35

24

31

OOM

OOM

OOM

43

47

47

47

46

45

62

70

71

71

70

68

120

125

125

125

119

110

121

128

129

129

125

126

132

135

136

134

134

123

136

139

140

139

141

RTX3090, (Head dim = 128, Causal = False)

Torch xformers

| |
|---|

| |
|---|

FlashAttn2 SageAttn-T

| |
|---|

| |
|---|

SageAttn-vT SageAttn-B

| |
|---|

| |
|---|

SageAttn-vB

1K 2K 4K 8K 16K 32K Sequence Length

0

100

200

13

8

10

OOM

OOM

OOM

38

41

43

44

45

44

56

60

62

63

62

60

110

122

127

130

129

89

109

123

130

133

132

- 94

113

121

127

132

127

- 95

114

126

133

136

136

RTX3090, (Head dim = 128, Causal = True)

Torch xformers

| |
|---|

| |
|---|

FlashAttn2 SageAttn-T

| |
|---|

| |
|---|

SageAttn-vT SageAttn-B

| |
|---|

| |
|---|

SageAttn-vB

- Figure 9: Speed comparison between SageAttention and baselines (RTX3090, headdim=128).

- 5.2 SPEED AND ACCURACY OF ATTENTION KERNELS

Speed. We conduct experiments to compare the Speed of SageAttention against baselines using configurations with headdim=64 or headdim=128, both with and without Causal Mask Vaswani (2017). Specifically, Figure 6 and Figure 7 show the Speed of SageAttention and baselines across varying sequence lengths on RTX4090. These results indicate that SageAttention achieves a peak of 341 TOPS and is 2x faster than FlashAttention2 and 2.9x faster than xformers on average. Figure 8 and Figure 9 illustrate the results on RTX3090, showing a similar speedup performance.

Accuracy. Table 9 shows the numerical error of four implementations of SageAttention compared with attention in full-precision. This experiment is conducted using a set of (Q, K, V) conforming to a normal distribution. It shows the error of the four implementations is rather small. SAGEAttn-T and SAGEAttn-B achieve 100% cosine similarity and RMSE in the e-4 level.

Table 7: Real speedup of SageAttention on RTX4090.

|Model|Shape of Q, K, V<br><br>|Original attention|SageAttention<br><br>|Speedup|
|---|---|---|---|---|
|CogvideoX Llama2 UltraPixel Unidiffuser TIMM<br><br>|(2, 30, 17776, 64) (4, 32, 1536, 128) (2, 32, 7285, 64) (4, 24, 1105, 64) (12, 64, 197, 64)|163.37 (FlashAttn2) 130.99 (FlashAttn2) 152.03 (FlashAttn2) 105.68 (xformers)<br><br>18.910 (Torch)|327.57 231.74 325.18 246.93 111.41<br><br>|2.01x<br><br>1.77x<br><br>2.14x 2.34x 5.89x<br><br><br>|

- 5.3 END-TO-END PERFORMANCE

Speedup. We measure the real speed of SageAttention and the original attention on Unidiffuser, UltraPixel, CogvideoX, Llama2 and TIMM on RTX4090. Table 7 shows that SageAttention outperforms original attention across all models. Specifically, SageAttention yields 2.83x speedup compared to the original attentions on average.

Metrics loss. We assessed the end-to-end metrics of various models using SageAttention compared to using attention in full-precision. Detailed evaluation results are presented in Table 8 for Llama2, CogvideoX, Unidiffuser, UltraPixel, and TIMM, respectively. The results indicate that SageAttention successfully matches the performance of attention in full-precision across all models. Specifically, on Llama2, CogvideoX, UltraPixel, and Unidiffuser, SageAttention resulted in only a minor average degradation of 0.2% compared to attention in full-precision. Moreover, on TIMM, SageAttention even surpasses attention in full-precision.

- Table 8: End-to-end metrics loss across text, image, and video generation models.

|Model<br><br>|Attention<br><br>|WikiText (Ppl.) ↓|Lambda (Acc.) ↑<br><br>|MMLU (Acc.) ↑|
|---|---|---|---|---|
|Llama2|Full-Precision SageAttention<br><br>|5.823 5.824<br><br>|0.886 0.887<br><br>|0.46 0.46<br><br>|

|Model<br><br>|Attention<br><br>|CLIPSIM ↑|CLIP-T ↑<br><br>|VQA-a ↑|VQA-t ↑<br><br>|FScore ↑|
|---|---|---|---|---|---|---|
|CogvideoX|Full-Precision SageAttention<br><br>|0.1837 0.1836<br><br>|0.9976 0.9976<br><br>|68.962 68.839<br><br>|75.925 75.037<br><br>|3.7684 3.8339<br><br>|

|Model|Attention<br><br>|FID ↓|sFID ↓<br><br>|CLIP ↑<br><br>|IR ↑|
|---|---|---|---|---|---|
|Unidiffuser<br><br>|Full-Precision SageAttention<br><br>|163.33 166.49<br><br>|145.08 143.18<br><br>|0.3152 0.3154<br><br>|0.1609 0.1521<br><br>|
|UltraPixel<br><br>|Full-Precision SageAttention<br><br>|179.78 179.79<br><br>|141.35 141.63<br><br>|0.3132 0.3131<br><br>|0.6169 0.6110<br><br>|

|Model<br><br>|Attention<br><br>|ImageNet (Acc.) ↑|Sketch (Acc.) ↑<br><br>|ImageNet-r (Acc.) ↑|
|---|---|---|---|---|
|TIMM|Full-Precision SageAttention<br><br>|84.79% 84.74%<br><br>|45.32% 45.78%<br><br>|59.55%<br><br>60.32%<br>|

|Model<br><br>|Attention|TextVQA (Acc.) ↑|POPE (Acc.) ↑|VQAv2 (Acc.) ↑|
|---|---|---|---|---|
|Llava1.6|Full-Precision SageAttention<br><br>|60.25% 60.09%<br><br>|86.45% 86.44%<br><br>|77.55% 77.47%<br><br>|

- Table 9: Accuracy of SageAttention kernels.

Table 10: Overhead of smoothing K.

|Model<br><br>|Smooth K|TOPS ↑|
|---|---|---|
|CogvideoX<br><br>|✗ ✓<br><br>|327.57 327.52<br><br>|
|UltraPixel|✗ ✓<br><br>|325.18 324.56<br><br>|

|attention<br><br>| |Cos Sim ↑|Relative L1 ↓<br><br>|RMSE ↓|
|---|---|---|---|---|
|SAGEAttn-T SAGEAttn-B SAGEAttn-vT SAGEAttn-vB<br><br>| |1.0 1.0 99.9% 98.9%<br><br>|0.019 0.021 0.064 0.138<br><br>|6.8e-4<br><br>7.3e-4 0.065 0.067<br>|

Table 11: Benefit of adaptive quantization.

|attention| |model<br><br>|CLIPSIM ↑<br><br>|TOPS ↑| |Model<br><br>|MMLU ↑<br><br>|TOPS ↑|
|---|---|---|---|---|---|---|---|---|
|SAGEAttn-T SageAttention<br><br>| |CogvideoX|0.1827 0.1835<br><br>|292.17 327.57<br><br>| |Llama2|0.46 0.46|208.59 231.74<br><br>|

- 5.4 ABLATION STUDY

Overhead of smoothing K. Table 10 presents the overhead associated with smoothing K on the attention speed in real models. The results indicate a minimal reduction, less than 0.2%.

Benefit of adaptive quantization. We analyzed the performance differences between using only SAGEAttn-T and employing an adaptive strategy (SageAttention). Table 11 presents the metrics and average speed of attention on CogvideoX and Llama2. The results indicate that the adaptive strategy increases the speed of attention by 11.7% without any loss in metrics.

- 6 CONCLUSION AND FUTURE WORK

We introduce SageAttention, an efficient and precise INT8 quantization method for attention. First, we propose a method to smooth matrix K, enhancing the accuracy with under 0.2% speed overhead. Second, we use FP16 accumulators in the Matmul of (P, V) to boost both accuracy and speed. Third, we use adaptive quantization to further improve OPS by 12% without sacrificing accuracy. Our method surpasses FlashAttention2 and xformers by approximately 2.1x and 2.7x, respectively. Extensive testing confirms that our approach maintains end-to-end metrics across various models, including language, image, and video generation models.

ACKNKOWLEGEMENT

The authors would like to thank Haofeng Huang for his valuable help with the implementation. This work was supported by the NSFC Project (No. 62376131), Tsinghua Institute for Guo Qiang, and the High Performance Computing Center, Tsinghua University. J.Z is also supported by the XPlorer Prize.

REFERENCES

Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023.

Ganesh Bikshandi and Jay Shah. A case study in cuda kernel fusion: Implementing flashattention-2 on nvidia hopper architecture using the cutlass library. arXiv preprint arXiv:2312.11918, 2023.

Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. arXiv preprint arXiv:2309.12307, 2023.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.

Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haibing Ren, Xiaolin Wei, Huaxia Xia, and Chunhua Shen. Twins: Revisiting the design of spatial attention in vision transformers. Advances in neural information processing systems, 34:9355–9366, 2021.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Tianyu Fu, Haofeng Huang, Xuefei Ning, Genghan Zhang, Boju Chen, Tianqi Wu, Hongyi Wang, Zixiao Huang, Shiyao Li, Shengen Yan, Guohao Dai, Huazhong Yang, and Yu Wang. Moa: Mixture of sparse attention for automatic large language model compression, 2024a.

Tianyu Fu, Tengxuan Liu, Qinghao Han, Guohao Dai, Shengen Yan, Huazhong Yang, Xuefei Ning, and Yu Wang. Framefusion: Combining similarity and importance for video token reduction on large visual language models, 2024b. URL https://arxiv.org/abs/2501.01986.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. 2020.

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. ICCV, 2021.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7514–7528, 2021.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Yuezhou Hu, Weiyu Huang, Zichen Liang, Chang Chen, Jintao Zhang, Jun Zhu, and Jianfei Chen. Identifying sensitive weights via post-quantization integral. arXiv preprint arXiv:2503.01901, 2025.

Weiyu Huang, Yuezhou Hu, Guohao Jian, Jun Zhu, and Jianfei Chen. Pruning large language models with semi-structural adaptive sparse training, 2024. URL https://arxiv.org/abs/ 2407.20584.

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2704–2713, 2018.

Fred Jelinek, Robert L Mercer, Lalit R Bahl, and James K Baker. Perplexity—a measure of the difficulty of speech recognition tasks. The Journal of the Acoustical Society of America, 62(S1): S63–S63, 1977.

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. arXiv preprint arXiv:2407.02490, 2024.

Youhe Jiang, Ran Yan, Xiaozhe Yao, Yang Zhou, Beidi Chen, and Binhang Yuan. Hexgen: Generative inference of large language model over heterogeneous environment. In Forty-first International Conference on Machine Learning.

Youhe Jiang, Fangcheng Fu, Xiaozhe Yao, Guoliang He, Xupeng Miao, Ana Klimovic, Bin Cui, Binhang Yuan, and Eiko Yoneki. Demystifying cost-efficiency in llm serving over heterogeneous gpus. arXiv preprint arXiv:2502.00722, 2025a.

Youhe Jiang, Ran Yan, and Binhang Yuan. Hexgen-2: Disaggregated generative inference of llms in heterogeneous environment. In International Conference on Learning Representations (ICLR), 2025b.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. I-bert: Integeronly bert quantization. In International conference on machine learning, pp. 5506–5518. PMLR,

- 2021.

Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, Daniel Haziza, Luca Wehrstedt, Jeremy Reizenstein, and Grigory Sizov. xformers: A modular and hackable transformer modelling library. https://github.com/facebookresearch/xformers,

- 2022.

Bingrui Li, Jianfei Chen, and Jun Zhu. Memory efficient optimizers with 4-bit states. Advances in Neural Information Processing Systems, 36:15136–15171, 2023a.

Bingrui Li, Wei Huang, Andi Han, Zhanpeng Zhou, Taiji Suzuki, Jun Zhu, and Jianfei Chen. On the optimization and generalization of two-layer transformers with sign gradient descent. In International Conference on Learning Representations (ICLR), 2025.

Kunchang Li, Yali Wang, Gao Peng, Guanglu Song, Yu Liu, Hongsheng Li, and Yu Qiao. Uniformer: Unified transformer for efficient spatial-temporal representation learning. In International Conference on Learning Representations.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023b.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740–755. Springer, 2014.

Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, and Song

- Han. Qserve: W4a8kv4 quantization and system co-design for efficient llm serving, 2024. URL https://arxiv.org/abs/2405.04532.
- Hao Liu, Matei Zaharia, and Pieter Abbeel. Ringattention with blockwise transformers for nearinfinite context. In The Twelfth International Conference on Learning Representations.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024a. URL https:// llava-vl.github.io/blog/2024-01-30-llava-next/.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22139–22149, 2024b.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations, 2022.

Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. arXiv preprint arXiv:1805.02867, 2018.

Denis Paperno, Germ´an Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fern´andez. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534, 2016.

Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, et al. Fp8-lm: Training fp8 large language models. arXiv preprint arXiv:2310.18313, 2023.

PyTorch Contributors. Torch backend documentation. https://pytorch.org/docs/ stable/backends.html#torch.backends.cuda.enable_math_sdp.

Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu. Ultrapixel: Advancing ultra-high-resolution image synthesis to new peaks. arXiv preprint arXiv:2407.02158, 2024.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. arXiv preprint arXiv:2407.08608, 2024.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.

Xiao Sun, Jungwook Choi, Chia-Yu Chen, Naigang Wang, Swagath Venkataramani, Vijayalakshmi Viji Srinivasan, Xiaodong Cui, Wei Zhang, and Kailash Gopalakrishnan. Hybrid 8-bit floating point (hfp8) training and inference for deep neural networks. Advances in neural information processing systems, 32, 2019.

Philippe Tillet, H. T. Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. MAPL 2019, pp. 10–19, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450367196.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. Shashanka Venkataramanan, Amir Ghodrati, Yuki M Asano, Fatih Porikli, and Amirhossein Habib-

ian. Skip-attention: Improving vision transformers by paying less attention. arXiv preprint arXiv:2301.02240, 2023.

Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. Advances in Neural Information Processing Systems, 32, 2019.

Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

Ziteng Wang, Jianfei Chen, and Jun Zhu. Remoe: Fully differentiable mixture-of-experts with relu routing. In International Conference on Learning Representations (ICLR), 2025.

Ross Wightman. Pytorch image models. https://github.com/rwightman/ pytorch-image-models, 2019.

Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20144–20154, 2023.

Haocheng Xi, Han Cai, Ligeng Zhu, Yao Lu, Kurt Keutzer, Jianfei Chen, and Song Han. Coat: Compressing optimizer states and activation for memory-efficient fp8 training. arXiv preprint arXiv:2410.19313, 2024a.

Haocheng Xi, Yuxiang Chen, Kang Zhao, KAI JUN TEH, Jianfei Chen, and Jun Zhu. Jetfire: Efficient and accurate transformer pretraining with int8 data flow and per-block quantization. In Forty-first International Conference on Machine Learning, 2024b.

Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. Infllm: Training-free long-context extrapolation for llms with an efficient context memory. In First Workshop on Long-Context Foundation Models@ ICML 2024, 2024.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, pp. 38087–38099. PMLR, 2023a.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023b.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.

Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. arXiv preprint arXiv:2505.18875, 2025.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10819–10829, 2022.

Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization. In International Conference on Machine Learning (ICML), 2025a.

Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. Sageattention2: Efficient attention with smoothing q and per-thread quantization. 2025b.

Jintao Zhang, Guoliang Li, and Jinyang Su. Sage: A framework of precise retrieval for rag. In 2025 IEEE 41th International Conference on Data Engineering (ICDE). IEEE, 2025c.

Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, and Jianfei Chen. Sageattention3: Microscaling fp4 attention for inference and an exploration of 8-bit training. arXiv preprint arXiv:2505.11594, 2025d.

Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattn: Accurate sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025e.

Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattn: Training-free sparse attention accelerating any model inference. 2025f.

Jintao Zhang, Xiaoming Xu, Jia Wei, Haofeng Huang, Pengle Zhang, Chendong Xiang, Jun Zhu, and Jianfei Chen. Sageattention2++: A more efficient implementation of sageattention2. 2025g.

Pengle Zhang, Jia Wei, Jintao Zhang, Jun Zhu, and Jianfei Chen. Accurate int8 training through dynamic block-level fallback. arXiv preprint arXiv:2503.08040, 2025h.

Tianchen Zhao, Tongcheng Fang, Enshu Liu, Rui Wan, Widyadewi Soedarmadji, Shiyao Li, Zinan Lin, Guohao Dai, Shengen Yan, Huazhong Yang, et al. Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation. arXiv preprint arXiv:2406.02540, 2024a.

Tianchen Zhao, Xuefei Ning, Tongcheng Fang, Enshu Liu, Guyue Huang, Zinan Lin, Shengen Yan, Guohao Dai, and Yu Wang. Mixdq: Memory-efficient few-step text-to-image diffusion models with metric-decoupled mixed precision quantization. In European Conference on Computer Vision, pp. 285–302. Springer, 2024b.

Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908, 2024a.

Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, and Jun Zhu. Diffusion bridge implicit models. arXiv preprint arXiv:2405.15885, 2024b.

Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, and Jun Zhu. Elucidating the preconditioning in consistency distillation. arXiv preprint arXiv:2502.02922, 2025.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024c. URL https://github.com/hpcaitech/Open-Sora.

- A EXPERIMENTAL DETAIL

- A.1 ENVIRONMENT

We implemented our Attention kernels using OpenAI Triton (Tillet et al., 2019) and conducted experiments on Ubuntu 22.04 servers. Tests on the RTX 4090 utilized a server with PCIE 5.0, a 16-core Xeon(R) 6430 CPU, and 120GB DDR4 RAM, while the RTX3090 tests employed a server with a 16-core Xeon(R) 8358P CPU and 80GB DDR4 RAM. To reproduce our results, experiments should be conducted in the environment of torch 2.4.0+cu121, triton-nightly (version of 20240816), python 3.11, and (gcc, g++) in version 9.

- A.2 HYPER-PARAMETERS FOR ATTENTION KERNELS We use 128 for a block size of Q, and 64 for a block size of K and V . The parameters Num Warps and Num Stages, which represent the number of warp schedulers and the number of processing stages in our GPU kernels, respectively, are detailed in Table 12.

Table 12: Hyper-parameters for our Attention Kernels.

|HeadDim|Causal Mask|Num Warps<br><br>|Num Stages<br><br>|
|---|---|---|---|
|64 64 128 128<br><br>|False True False True<br><br>|4 4 8 8|3<br><br>4<br><br>3<br><br>5<br>|

- A.3 DETAILS OF DATASETS AND MODELS

We choose the first 256 annotations from the COCO 2014val dataset as the prompt set for UltraPixel and Unidiffuser image generation. We also used the corresponding 256 images of the 256 prompts as the ground truth images to calculate the FID and sFID. For CogvideoX, the model is trained on long texts, so we applied an open-sora prompt set, each consisting of more than 120 words. The specific model we used for TIMM is vit base patch16 224.augreg2 in21k ft in1k.

- A.4 ADDITIONAL EXPERIMENTS

Table 13: Comparison of SageAttention with AWQ (W4A16) on Llama2.

| |Full-Precision|SageAttention<br><br>|AWQ|AWQ+SageAttention|
|---|---|---|---|---|
|Perplexity↓ Speedup of Linear Computation Speedup of Attention|5.4721 0 0<br><br>|5.4729 0 2x<br><br>|5.5988 0 0<br><br>|5.5998 0 2x|

Table 14: Comparison of SageAttention with Q-diffusion (W8A8) on Unidiffuser.

| |FID↓<br><br>|sFID↓<br><br>|CLIP↑|ImageReward↑<br><br>|
|---|---|---|---|---|
|Full Precision SageAttention Q-diffusion (W8A8)|163.33 166.49 395.99<br><br>|145.08 143.18 178.56|31.52 31.54 18.03<br><br>|0.1609 0.1521 -2.273|

- A.5 COMPARISON WITH OTHER METHODS

There are some task-specific quantization methods, such as AWQ for LLMs, Q-diffusion for textto-image, and ViDiT-Q for text-to-video applications. SageAttention is orthogonal to them because those works are mainly used to quantize the linear layers. Second, AWQ is only used to compress the parameters of LLMs with no acceleration effect in computation. Q-diffusion has not reported its

Table 15: Comparison of SageAttention with VIDIT-Q on CogvideoX.

| |CLIPSIM↑<br><br>|CLIP-T↑|VQA-a↑<br><br>|VQA-t↑|FScore↑<br><br>|End-to-end Speedup↑|
|---|---|---|---|---|---|---|
|Full Precision SageAttention VIDIT-Q (W8A8)<br><br>|0.1837 0.1836 0.1884|0.9976 0.9976 0.9974<br><br>|68.962 68.839 68.185|75.925 75.037 71.011<br><br>|3.7684 3.8339 3.7342|34.3% 22% (theoretical maximum)<br><br>|

acceleration effect in their paper and provided codes with acceleration effect in its official repository. ViDiT-Q has not provided the codes with acceleration effect in its official repository. Nonetheless, we compare SageAttention with those works as follows. (1) We compare the perplexity of Llama2-

- 7B on WikiText and the speedup in the prefilling stage. The results are shown in Table 13. We compare SageAttention with Q-diffusion (W8A8) on Unidiffuser. The results are shown in Table 14. We compare SageAttention with VIDIT-Q on CogvideoX and the results are shown in Table 15. Since the official repository does not provide acceleration code, we estimate a theoretical maximum: the Linear layer accounts for 24% of Cogvideo’s latency, and W8A8 offers at most 4x speedup for

the Linear layer, resulting in a theoretical maximum end-to-end speedup of 100−10024×3

###### % = 22%.

4

- A.6 SOME INSIGHTS

Table 1 shows that the metric of Llama2 remains stable with quantization. The reason is that the distribution of Q,K, and V in the attention of Llama2-7B is relatively uniform. As a result, quantizing Q,K, and V to INT8 or FP8 does not significantly impact the accuracy of attention. This insight inspires the idea that better control over outlier activations in models could lead to more precise quantization results. As works like Fu et al. (2024a); Zhang et al. (2025a;b), we believe SageAttention can also be effectively applied to various applications related to Transformers, such as MOE systems (Wang et al., 2025), linear layer quantization (Hu et al., 2025; Zhang et al., 2025h; Zhao et al., 2024a), RAG systems (Zhang et al., 2025c), training optimization (Li et al., 2023a; 2025; Huang et al., 2024; Xi et al., 2024a), heterogeneous GPU systems (Jiang et al., 2025a; Jiang et al.; 2025b), and diffusion models acceleration (Zheng et al., 2024a;b; 2025; Fu et al., 2024b; Zhao et al., 2024b; Xi et al., 2025; Zhang et al., 2025e;f; Yang et al., 2025; Zhang et al., 2025d;g).

Table 16: SageAttention based on Torch Attention.

|Sequence Length<br><br>|Torch Attention<br><br>|SageAttention based on Torch Attention|
|---|---|---|
|1024 2048 4096 8192<br><br>|46 42 55 OOM<br><br>|48 55 87 OOM|

- B IMPLEMENTATION BASED ON TORCH ATTENTION

FlashAttention is the state-of-the-art and most commonly used standard attention; another commonly used attention is Torch attention (PyTorch Contributors). We report the implementation speeds based on Torch in Table 16.

Table 17: Numerical error of Q · K using different type of quantization.

|Data Type| |Cosine Sim|Relative L1|
|---|---|---|---|
|INT8<br><br>E4M3<br>E5M2<br>| |99.54% 92.83% 77.95%<br><br>|0.084 0.342 0.681|

- B.1 ADDITIONAL PRECISION COMPARISION

- Table 17 shows the precision of Q · K using per-token quantization in different data types compared to Q · K in full precision. This experiment is conducted using Q,K from the 24th layer of

Table 18: Error of quantized attention with or without smoothed K.

|Quantization Type<br><br>|Smoothed K| |Cosine Sim ↑|Relative L1 ↓<br><br>|RMSE ↓|
|---|---|---|---|---|---|
|Per-token (SAGEAttn-T)<br><br>|Without With| |62.24% 99.47%<br><br>|1.187 0.045|0.294 0.031<br><br>|
|Per-block (SAGEAttn-B)<br><br>|Without With| |30.60% 99.31%<br><br>|1.286 0.072|0.464 0.035<br><br>|
|Per-tensor|Without With<br><br>| |41.40% 98.06%|1.554 0.126<br><br>|0.399 0.059|
|FlashAttention-3 (quantized version)| | |26.76%|2.5354<br><br>|0.5378|

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

(a) Full-Precision (b) SAGE Attention

- Figure 10: An image generation example of UltraPixel.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

(b) SAGE Attention

(a) Full-Precision

[Figure 39]

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

[Figure 40]

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

[Figure 46]

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

- Figure 11: A video generation example of Open-Sora.

Table 19: Comparison of real speedup on RTX3090.

|Model<br><br>|Shape of Q, K, V|Original Attention|SAGEAttention<br><br>|Speedup|
|---|---|---|---|---|
|CogvideoX Llama2 UltraPixel Unidiffuser TIMM<br><br>|(2, 30, 17776, 64) (4, 32, 1536, 128) (2, 32, 7285, 64) (4, 24, 1105, 64) (12, 64, 197, 64)<br><br>|71.57 (FlashAttn2) 56.54 (FlashAttn2) 65.86 (FlashAttn2) 47.64 (xformers) 12.33 (Torch)|129.87 108.91 131.74 108.91 66.34<br><br>|1.81x<br><br>1.93x<br><br>2.00x<br><br><br>2.29x 5.38x<br><br><br>|

Unidiffuser. It shows that quantizing Q,K to INT8 performs higher precision than using E4M3 and E5M2.

[Figure 47]

- Figure 12: More image generation examples of UltraPixel, where prompt1=”Two dogs are looking up while they stand near the toilet in the bathroom”, prompt2=”A gray bicycle is locked to some metal doors”, prompt3=”An image of a car driving on the highway”, and prompt4=”A cat on the lid of a toilet looking perturbed”.

[Figure 48]

- Figure 13: More image generation examples of Unidiffuser, where prompt1=”Beautiful view of the Himalayas”, prompt2=”An elephant under the sea”, prompt3=”English Country Garden Design”, and prompt4=”An old red electric rail train in Durango, Colorado”.

[Figure 49]

- Figure 14: More image generation examples of CogvideoX. For more details about the prompts and the full videos, refer to https://anonymous.4open.science/r/image_video_ examples-3E44/README.md.

- Table 18 shows the precision of different quantization methods with and without smoothing K on various models. The results demonstrate that smoothing K offers significant benefits of precision.

- B.2 VISUALIZED RESULTS

- Figure 10 shows the high-resolution images (2560x1536) generated by UltraPixel using Attention of full precision and SageAttention. It can be seen that SageAttention matches the full precision in the high quality and highly detailed images. Figure 11 shows the videos (720x1280) generated by Open-Sora (Zheng et al., 2024c) in different precisions. SageAttention yields identically the same video as the full precision one.

Figure 12, Figure 13, and Figure 14 show more visualized comparison results on UltraPixel, Unidiffuser, and CogvideoX.

- B.3 REAL SPEEDUP ON RTX3090

We further measure the real speed of SageAttention and the original Attention on Unidiffuser, UltraPixel, CogvideoX, Llama2 and TIMM on RTX3090. Table 7 shows that SageAttention outperforms original attention across all models. Specifically, SageAttention yields 2.7× speedup compared to the original Attentions on average.

