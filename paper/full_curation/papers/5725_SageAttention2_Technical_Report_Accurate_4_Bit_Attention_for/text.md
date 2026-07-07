#### SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

Jintao Zhang*1 Haofeng Huang*12 Pengle Zhang1 Jia Wei1 Jun Zhu1 Jianfei Chen1 https://github.com/thu-ml/SageAttention

## arXiv:2411.10958v7[cs.LG]1Oct2025

##### Abstract

Although quantization for linear layers has been widely used, its application to accelerate the attention process remains limited. To further enhance the efficiency of attention computation compared to SageAttention while maintaining precision, we propose SageAttention2, which utilizes significantly faster 4-bit matrix multiplication (Matmul) alongside additional precision-enhancing techniques. First, we propose to quantize matrices (Q,K) to INT4 in a hardware-friendly threadlevel granularity and quantize matrices ( P,V ) to FP8. Second, we propose a method to smooth Q, enhancing the accuracy of INT4 QK⊤. Third, we propose a two-level accumulation strategy for PV to enhance the accuracy of FP8 PV . The operations per second (OPS) of SageAttention2 surpass FlashAttention2 and xformers by about 3x and 4.5x. Moreover, SageAttention2 matches the speed of FlashAttention3(fp8) on the Hopper GPUs, while delivering much higher accuracy. Comprehensive experiments confirm that our approach incurs negligible end-to-end metrics loss across diverse models, including those for language, image, and video generation. The code is available at https://github.com/ thu-ml/SageAttention.

##### 1. Introduction

Due to the quadratic time complexity of attention, its efficient implementation becomes crucial as sequence lengths

*Equal contribution 1Dept. of Comp. Sci. and Tech., Institute for AI, BNRist Center, THBI Lab, Tsinghua-Bosch Joint ML Center, Tsinghua University 2Institute for Interdisciplinary Information Sciences, Tsinghua University. Correspondence to: Jianfei Chen <jianfeic@tsinghua.edu.cn>.

Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

increase in real-world applications (Jiang et al., 2024; Zhang et al., 2025b). Several strategies have been developed to mitigate the computational demands of attention —such as (1) Linear attention methods (Wang et al., 2020; Choromanski et al., 2021; Yu et al., 2022; Katharopoulos et al., 2020) that reduce complexity to O(N) and (2) Sparse attention methods (Liu et al., 2021; Chu et al., 2021; Li et al., 2022; Xiao et al., 2024b;a; Chen et al., 2024; Jiang et al.,

- 2024; Venkataramanan et al., 2024; Gao et al., 2024; Fu et al., 2024; Zhang et al., 2025e; Xi et al., 2025; Yang et al.,
- 2025a; Zhang et al., 2025f;h;i) that selectively process parts of the context — these methods are only suitable for a limited range of models and tasks. The widely used attention methods optimize attention by exploiting hardware capacities, such as FlashAttention V1, V2, V3 (Dao et al., 2022; Dao, 2024; Shah et al., 2024), xformers (Lefaudeux et al., 2022), and SageAttention (Zhang et al., 2025c;d;g;a). These works do not omit computations across the entire sequence and achieve impressive speed and accuracy performance across various tasks.

Motivation. For the two matrix multiplication (Matmul) operations in attention: QK⊤ and PV , SageAttention accelerates them by quantizing the QK⊤ to INT8 and uses FP16 Matmul with FP16 accumulators for PV . Moreover, to maintain attention accuracy, SageAttention proposes smoothing K by eliminating its channel-wise outliers. SageAttention achieves 2 × speedup compared with FlashAttention2 and is the first quantized attention that incurs a negligible end-to-end metrics loss across language, image, and video generation models. However, SageAttention has two weaknesses. (W1) INT8 Matmul achieves only half the speed of INT4. (W2) FP16 Matmul with FP16 accumulators provides a speedup only on RTX 4090 and RTX 3090 GPUs. To leverage the faster INT4 tensor cores for QK⊤ and using a method that can accelerate PV on a broader range of GPUs, we propose to quantize Q,K to INT4 and P,V to FP8.

Challenges. Quantizing Q,K to INT4 and P,V to FP8 presents significant challenges. For example, when only pertensor quantizing Q,K to INT4, the text-to-video model CogvideoX will generate a completely blurry video, and

[Figure 1]

[Figure 2]

| | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

| |
|---|

| |
|---|

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Figure 1. The upper left figure shows the kernel speedup on RTX4090 GPU. The upper right figure shows the end-to-end inference speedup of generating the first token and performance metrics for the Needle-in-a-Haystack task (Kamradt, 2023) with a sequence length of 100K on Llama3.1 on L20 GPU. The figure below shows two videos generated by CogvideoX (1.5-5B) using FlashAttention2 and SageAttention2 on RTX4090, showing that SageAttention2 accelerates generation by 1.8x with no video quality loss.

Llama3 only achieves a random-guessing-level accuracy of 25% on MMLU. After an in-depth investigation, we identified three primary challenges: (C1) The numerical range of INT4 is quite limited compared to FP16 or INT8, leading to significant quantization errors when Q and K have some abnormal values. (C2) We discover that the FP32 accumulator designed for FP8 matrix multiplication in the tensor core (mma.f32.f8.f8.f32) is actually FP22, specifically with 1 sign bit, 8 exponent bits, and 13 mantissa bits. This will lead to an accuracy loss of PV .

Our approach. To address (C1), we first discover that the per-block quantization of Q,K in SageAttention does not achieve sufficient accuracy for INT4 quantization. Simultaneously, to avoid the extra latency caused by per-token dequantization, where each GPU thread corresponds to multiple quantization scales, we propose an adequately precise quantization method that incurs no additional dequantization overhead. Specifically, we introduce a per-thread quantization approach based on the mapping between the GPU threads and the memory layout of matrices as dictated by the PTX mma instructions. This method groups tokens corresponding to the same thread for quantization and dequantization, ensuring that each thread is associated with a single quantization scale. This approach achieves much better accuracy performance than per-block quantization with no additional dequantization overhead. Second, for the significant channel-wise outliers in matrices Q and K, we adopt smoothing K in SageAttention and further propose an effective method to remove these outliers in Q. Specifically, we propose subtracting the average value of the channel dimen-

sion of Q, referred to as −→Qm. Then, we add −→QmK after the QK⊤ Matmul to ensure the correctness of the attention computation. To address (C2), the accuracy loss caused by the 22-bit accumulator for FP8 Matmul of PV , we propose a two-level accumulation strategy that uses an FP32 buffer to accumulate the values from the 22-bit accumulator after each block Matmul of PV . This method confines the errors to the block range. Additionally, we design an optional technique to enhance the accuracy of the 22-bit accumulator. Specifically, we could smooth V by subtracting the average value of its channel dimension and adding the subtracted item to the attention output to maintain the correctness.

Performance. We offer a high-performance implementation of SageAttention2 on RTX4090 and L20 GPUs. This implementation achieves a peak performance of 481 TOPS on the RTX4090, outperforming FlashAttention2 and xformers by approximately 3x and 4.5x, respectively. To support NVIDIA Hopper GPUs, which lack native INT4 tensor core support, we additionally provide SageAttention2-8b, a variant that quantizes Q,K to INT8. FlashAttention3, in contrast, is tailored to and only compatible with the Hopper architecture. Moreover, SageAttention2-8b matches the speed of FlashAttention3(fp8) on Hopper GPUs, while delivering much better accuracy. For example, on popular video generation models, our method does not compromise end-to-end accuracy, whereas FlashAttention3(fp8) brings noticeable degradation, as visualized in Fig. 7 and 9. We extensively evaluate the end-to-end metrics of state-of-the-art text, image, and video generation models. SageAttention2 can accelerate models in a plug-and-

play way with negligible loss in end-to-end metrics.

##### 2. Preliminary

###### 2.1. FlashAttention

The attention computation can be formulated as: S = QK⊤/

√

d, P = σ(S), O = PV , where σ(S)ij =

exp(Sij)/ k exp(Sik). The matrices Q, K, and V each have dimensionality N × d, and S, P are N × N. d is

typically small, e.g., 64 or 128, and N can be thousands or millions. The time complexity of attention is O(N2), primarily due to two matrix multiplications (QK⊤ and PV ), both with complexities of O(N2d). FlashAttention (Dao, 2024) is a GPU-friendly attention implementation, which tiles Q, K, and V from the token dimension into blocks {Qi}ni=1q ,{Ki}n

i=q,{Vi}n

i=1 with block sizes of bq, bk, bv tokens, respectively, where nq,nk,nv are the number of tiles, and bk = bv. FlashAttention computes the output matrix O in parallel in tiles. Each streaming multiprocessor (SM) computes a block Oi (corresponds to a Qi) by iteratively loads Kj,Vj for each j, and update the output with online softmax (Milakov & Gimelshein, 2018):

v

k

√

Sij =QiKj⊤/

d, (mij, Pij) = σ˜(mi,j−1,Sij), (1) lij =exp(mi,j−1 − mij)li,j−1 + rowsum( Pij),

Oij =diag (exp(mi,j−1 − mij))Oi,j−1 + PijVj,

where mij and lij are bq-dimenalional vectors, initialized with −∞ and 0 respectively. σ˜(·) is an online softmax

operator: mij = max{mi,j−1,rowmax(Sij)}, Pij = exp(Sij − mij). Finally, the output is computed as Oi = diag(li,n

.

)−1Oi,n

q

q

###### 2.2. Quantization

A matrix multiplication C = AB can be accelerated with quantization as:

(δA,Aˆ) = ψ(A), (δB,Bˆ) = ψ(B), C = ψδ−1

AδB(AˆBˆ)

ψ is a quantizer which converts a high-precision (e.g., FP32 or FP16) matrix A to a low-precision format Aˆ (e.g., INT4 or FP8) with a scale δA, and ψ−1 is a dequantizer to convert back to high-precision. We should have ψδ−1

(Aˆ) ≈ A. The actual matrix multiplication AˆBˆ

A

is performed with low precision. In modern GPUs, lowprecision matrix multiplication is usually multiple times faster than higher-precision ones. Quantizers differ in numerical format and granularity, e.g., how many elements (“quantization group”) share a common scale factor. For example, an INT4, per-tensor quantizer first computes the scale as the maximum absolute value of the entire tensor, scales the elements to the maximum representable range of INT4 [-7, +7], and then casts to INT4 with rounding:

Aˆ = ⌈A/δA⌋,δA = max(|A|)/7. The dequantization process is an element-wise scaling. For example, for per-tensor

AδB(AˆBˆ) = AˆBˆ × δAδB.

dequantization, ψδ−1

Table 1. Speedup compared to matrix multiplication in FP16 with an FP32 accumulator.

|GPU<br><br>|MM Input|MM Accumulator<br><br>|Speedup|
|---|---|---|---|
|RTX4090<br><br>|FP16 FP8|FP16 FP32<br><br>|2x 2x|
|L40, L20 H100<br><br>|FP16 FP8<br><br>|FP16 FP32|1x<br><br>2x<br>|

###### 2.3. SageAttention

Based on the block tiling in FlashAttention (Dao et al., 2022), SageAttention (Zhang et al., 2025c) quantizes Q,K to INT8 in a per-block granularity, that is, each Qi,Ki has a separate scalar scale: δQ

= max(|Qi|)/127,δK

= max(|Kj|)/127. In this way, the product Sij in Eq. (1) can be approximated as Sij ≈ QˆiKˆj⊤ × (δQ

i

j

√

d). To maintain accuracy, SageAttention proposes a preprocessing technique by subtracting the token-wise mean from K. Additionally, SageAttention keeps Pij and Vj in FP16, but utilizes an FP16 accumulator (rather than FP32) for computing the product PijVj. Reducing accumulator precision can accelerate matrix multiplication (MM) on the RTX4090 GPU. However, other GPUs, such as L20, L40, or H100, do not exhibit this behavior, as shown in Table 1.

δK

/

i

j

##### 3. SageAttention2

In this section, we introduce SageAttention2, an efficient and accurate quantized attention. The workflow of SageAttention2 is shown in Fig. 3. We quantize Q,K to INT4 and P,V˜ to FP8 to maximize the efficiency and propose several techniques, including QK-smoothing, per-thread quantization, and two-level accumulation to preserve the accuracy, which we shall discuss in subsequent subsections.

###### 3.1. Smooth Q

First, we discuss how to accurately compute QK⊤ with INT4. The numerical range of INT4 is notably restrictive. This affects quantization due to the presence of outliers (Lin et al., 2025). Given the INT4 range [-7, +7], any element will be quantized to zero if it is more than 14 times (0.5 vs 7) smaller than the largest element in the group. Since outliers are much larger than other elements, it is likely that many non-outlier elements are quantized to zero, leading to substantial accuracy degradation. Therefore, to keep the quantization accurate, we need to keep the largest element small, making the magnitude of elements as uniform as possible. Such technique is called smoothing.

Here, we propose a smoothing technique inspired by

[Figure 14]

Figure 2. Typical examples of tensors’ data distribution in attention.

QFP16 KFP16 VFP16

| |

###### |

1 1

1

Optional

Qmean Kmean

Vmean

=

=

=

QFP16 KFP16

VFP16

|Mat Mul| | |
|---|---|---|
| | | |

3

- 2 3

Per-thread Quantization

Per-channel Quantization

VFP8

###### sq sk

sv

QINT4 KINT4

SageAttention2 Kernel

4

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

O

5

Optional

+

- Figure 3. Workflow of SageAttention2. 1 Smooth Q, K, V . 2 A GEMV to obtain ∆S. 3 Per-thread quantize Q, K and per-

channel quantize V . 4 Perform the SageAttention2 kernel. 5 Correct the output.

SageAttention (Zhang et al., 2025c). SageAttention observed that Q,K for all tokens are actually highly similar, with only small variations between different tokens (Fig. 2 shows the heatmap distributions of the Q, K, and V randomly sampled from Llama3.1 (Dubey et al., 2024) and CogvideoX (Yang et al., 2025b)). We propose to smooth K as SageAttention does and further smooth Q by subtracting a common mean of each block:

γ(Qi) = Qi − q¯i, γ(Kj) = Kj − k,¯ (2)

where q¯i = mean(Qi),k¯ = mean(K) are 1 × D vectors, the mean is conducted along the token axis, and q¯i,k¯ are broadcasted to tokens in a block and a tensor for subtraction.

With the decomposion, we have Sij = QiKj⊤ = (¯qi + γ(Qi))(k¯ + γ(Kj))⊤ = q¯ik¯⊤ + q¯iγ(Kj)⊤ + γ(Qi)k¯⊤ + γ(Qi)γ(Kj)⊤ = γ(Qi)γ(Kj)⊤+∆Sij +b. Here, ∆Sij = q¯iγ(Kj)⊤ is an 1 × N vector, and b = q¯ik¯⊤ + γ(Qi)k¯⊤ is an N × 1 vector. We do not need to compute b since adding a common bias to an entire row of S does not affect the result after softmax. Therefore, we can accelerate QiKj⊤ with INT4 by the following two stages:

- (1) preprocessing: smooth Q,K according to Eq. (2), ap-

,Kˆj) = ψK(γ(Kj)), and compute ∆Sij = q¯iγ(Kj)⊤. Smoothing,

,Qˆi) = ψQ(γ(Qi)),(δK

ply quantization (δQ

i

j

quantization, and GEMV (general matrix-vector multiplication) for computing ∆S can be fused into a single kernel, which reads the off-chip Q and K only once.

(2) attention: execute the low-precision GEMM, dequantize, and add back the vector ∆S: Sij = ψδ−1

QiδKj (QˆiKˆj⊤)+ ∆Sij. These operations are all done on chip, and the dequantization and vector addition only add a marginal overhead compared to the expensive mma operation for MM.

Importantly, γ(Qi),γ(Kj) are quantized rather than Qi,Kj. Since the smoothed matrices are much smaller in magnitude and contain fewer outliers, the quantization accuracy can be significantly improved. A theoretical analysis of the benefit of smoothing is included in Appendix A.5.

Remark. Classical techniques to improve the activationweight MM, such as per-channel quantization, or SmoothQuant (Xiao et al., 2023) are not applicable here for the query-key MM in attention. Per-channel quantization cannot be applied to Q,K because the quantization must be conducted along the outer axis (token dimension) of QK⊤. On the other hand, both Q and K have significant outliers, so trading the quantization accuracy between them with SmoothQuant cannot work effectively, as shown in Sec. 4. Here, we utilize the unique token similarity pattern in attention to derive a dedicated quantization method for Q and K. The previous work SageAttention only smooths K, so it is less accurate than our method.

Empirical results. Fig. 20 in Appendix A.9 shows an example from CogvideoX of the distribution of Qˆ with and without smoothing Q. We can find that with smoothing Q, the range of INT4 is utilized more uniformly and fully. Table 5 presents end-to-end metrics for different quantization methods with and without smoothing Q+K on Llama3.1 and CogvideoX (2b). The results demonstrate that smoothing Q+K offers significant accuracy benefits. Also, Table 4 and 17 show that the order of effectiveness is smoothing Q+K > smoothing Q > smoothing K > other baselines.

###### 3.2. INT4 Per-thread Quantization

Orthogonal to smoothing, we can mitigate the problem of outliers by refining the quantization granularity so that the number of affected elements by outliers becomes smaller. Although per-token quantization offers a detailed level of granularity, it results in significant overhead

###### Quantization in each Warp

MMA.m16n8k64

Dequantization

|Token0|Token1|Token2|Token3|Token4|Token5|Token6|Token7|
|---|---|---|---|---|---|---|---|

Token1

Token6

Token0

Token4 Token3

Token5

Token2

Token7

B

C = AB

ωK[0]

|Token0|
|---|
|Token1|
|Token2|
|Token3|

(Kˆ→)

→

|Token0|
|---|
|Token1|

|T0|T0|
|---|---|
|T0|T0|

|Token0|
|---|
|Token1|
|Token2|
|Token3|
|Token4|
|Token5|
|Token6|
|Token7|
|Token8|
|Token9|
|Token10|
|Token11|
|Token12|
|Token13|
|Token14|
|Token15|

→

T0

T0

T1

T1

T2

T2

T3

T3

- ωQ[0]

- ωQ[1]

- ωQ[0]

- ωQ[1]

- ωK[0]

- ωK[1]

T4

T4

T5

T5

T6

T6

T7

T7

··· ···

ωK[1]

|Token8|
|---|
|Token9|
|Token10|
|Token11|

→

|Token8|
|---|
|Token9|

|T5|T5|
|---|---|
|T5|T5|

→

# ···

T28

T28

T29

T29

T30

T30 T2 T6

T31

T31

A

C

··· ···

T0

T0

T1 T5

T1

T2

T3

T3

(QˆKˆ→)

(Qˆ)

T4

T4

T5

T6

T7

T7

ωK[3]

→

|Token56|
|---|
|Token57|
|Token58|
|Token59|

|Token24|
|---|
|Token25|

|T31|T31|
|---|---|
|T31|T31|

### ··· ···

→

ωQ[7]

ωQ[7] ωK[3]

QωK(QˆKˆ↑)

T28

T28

T29

T29

T30

T30

T31

T31

ωω→1

Qw

Kj

Each Thread Corresponds to One Scale

- Figure 4. An example of per-thread quantization. The left figure shows the correspondence between the quantization scales and the tokens in each GPU warp. The right figure shows the correspondence between quantization tokens and GPU threads in a MMA.m16n8k64 instruction, showing that each GPU thread only corresponds to one quantization scale in δQ and δK in dequantization.

during dequantization. Specifically, each GPU thread in per-token quantization must handle multiple quantization scales, leading to a high latency of the dot product of the quantization scale vectors δQ and δK. SageAttention uses per-block quantization, where each block Qi (bq tokens) and Ki (bk tokens) have a single quantization scale. Such a quantization strategy could achieve an accuracy performance close to per-token quantization and avoid the high dequantization overhead. However, quantizing Q and K to INT4 demands a finer quantization granularity. To address this, we propose per-thread quantization, a more precise and granular approach than the per-block quantizer, also without the additional overhead of the vector dot product between δQ and δK.

Specifically, each block of Q, i.e., Qi, in SageAttention will be split into cw segments and processed by cw GPU warps in a GPU streaming processor (SM). We call each segment of Qi as Qw, and kw = Kj since Kj is shared among warps. Then, each warp containing 32 threads uses the mma.m16n8k64 PTX instruction (NVIDIA) for the QwKj⊤. According to the layout requirement of this instruction, we find that Qw[8k + i] could share one quantization scale, and Kj[8k + 2i] together with Kj[8k + 2i + 1] could share one quantization scale. Such a quantization method is more fine-grained with no additional overhead. This is because it assigns different GPU threads to distinct quantization groups based on the MMA instruction layout, with each thread performing dequantization only using a single quantization scale value. We show an example of per-thread quantization in Fig. 4. The detailed formulation is shown in Equation 8 and Fig. 18 (please see Appendix A.6 for more detail).

Empirical results. As shown in Table 6 and Table 15, we compare the average and worst accuracy of INT4 quantization at per-token, per-thread, per-block, and per-tensor granularity using real Q,K,V across all layers of CogvideoX. Results indicate that the accuracy of per-thread quantization is very close to per-token and significantly outperforms other granularities. Moreover, Table 19 shows that per-thread quantization incurs almost no speed degradation, while pertoken quantization introduces noticeable overhead due to the reduced hardware efficiency.

3.3. FP8 quantization for PV˜ We now turn to the MM PV˜ , where P˜ij = exp(Sij − mij) is the unnormalized quantity according to Eq. (1). The distribution of P˜ is unique and differs from other activations. First, we note that Sij − mij ≤ 0, so Pij ∈ [0,1] (≤ and ∈ apply element-wise). We find that P˜ often consists of many small elements, but their sum is non-negligible (e.g., 5000 elements around 10−4). In this case, we must represent small elements accurately. INT quantization is unsuitable for this setting, since it distributes the quantization points evenly within the numerical range. SageAttention (Zhang et al., 2025c) choose to retain P˜ and V in FP16, and accelerate the MM by decreasing the accumulator precision. However, this strategy is only effective on very few GPUs. We propose to quantize P,V to FP8 with 4 exponent bits and 3 mantissa bits (E4M3). The numerical range of E4M3 is [−448,+448]. We quantize P with a static scale: δP = 4481 since the original P elements are already in [0,1]. We quantize V per-channel to address the channel-wise outliers shown in Fig. 2. Empirical results in Table 7 and Table 16 show the average and worst accuracy of different

Algorithm 1 Implementation of SageAttention2.

Input: Matrices Q(FP16), K(FP16), V (FP16) ∈ RN×d, block size bq, bkv, warp count cw. Preprocessing: K = K − mean(K), (δV , Vˆ) = ψV (V ). //per-channel. Divide Q to Tm = N/bq blocks {Qi}; divide K, and V to Tn = N/bkv blocks {Ki}, {Vi};

- for i = 1 to Tm do q¯i = mean(Qi), (δQ, Qˆi) = ψQ(Qi − q¯i) //per-thread ;
- for j in [1, Tn] do (δK, Kˆj) = ψK(Kj) //per-thread, w = range(cw), st = w ∗ cw ;

QδK(Matmul(Qˆi[st : st + cw], Kˆj⊤)) + GEMV(¯qi, Kj⊤) ; // Paralleled by cw warps. The ψδ−1

Sij[st : st + cw] = ψδ−1

QδK is

illustrated in Fig. 4. mij = max(mi,j−1, rowmax(Sij)), Pij = exp(Sij − mij), lij = emi,j−1−mijli,j−1 + rowsum( Pij) ; Oij(FP22) = Matmul(( Pij ∗ 448).to(FP8.e4m3), Vj) ; Oij(FP32) = diag(emi,j−1−mij)Oi,j−1(FP32) + Oij(FP22) ;

end for Load δV into an SM ; Oi = diag(li,Tn)−1Oi,Tn(FP32) /448 ∗ δV ; Write Oi ;

end for return O = {Oi}

data types used for P,V across all layers of CogvideoX. The accumulator is always 32-bit. We can see that the accuracy of E4M3 is very close to that of FP16 and superior to E5M2 and INT8. Most modern GPUs have tensor cores that support FP8 Matmul operations, which are twice as fast as those using FP16.

###### 3.4. FP32 MMA Buffer for FP22 Accumulator

While FP8 quantization for PV˜ above is theoretically accurate in simulation, we observe that the actual CUDA implementation suffers a consistent accuracy degradation. After narrowing down the problem, we find that the accumulator for the mma(f32f8f8f32) instruction on the Ada and Hopper architecture is actually FP22, specifically with 1 sign bit, 8 exponent bits, and 13 mantissa bits. Specifically, for mma(f32f8f8f32) instruction C = AB +D, where A,B are FP8 matrices and C,D are FP32 matrices, we initialize the A,B to zero and vary D to test the data type of the accumulator. When D is initialized with 1 sign bit, 8 exponent bits, and 13 mantissa bits, the value of C exactly matches D. However, when D is initialized with more than 13 mantissa bits, the value of C is equal to D with its least significant 10 mantissa bits zeroed out (i.e., truncated). Consequently, matrix multiplication of PV , quantized to FP8, incurs a certain degree of accuracy loss compared to using an FP32 accumulator.

To mitigate this accuracy loss, we adopt a two-level accumulation strategy, which uses an FP32 buffer to accumulate the values of PijVj in FP22. Specifically, we rewrite Eq. (1) as Rij = PijVj,Oij = diag (exp(mi,j−1 − mij))Oi,j−1 + Rij. Here, two sets of accumulators Rij and Oij are maintained in the register. Rij is computed with the mma(f32f8f8f32) instruction, providing 22 effective bits, which is sufficient since we only accumulate over a small number of bk tokens (e.g., bk = 64). Then, Rij is

accumulated to Oij in the high FP32 precision.

Remark. The two-level accumulation strategy is also implemented in CUTLASS (NVIDIA, 2023) and DeepGemm (DeepSeek-AI et al., 2024) for computing weightactivation products in linear layers. To the best of our knowledge, we are the first to discover and investigate the effect of the FP22 accumulator and implement the two-level accumulation for attention.

Optional smooth V technique. We also figure out another way to mitigate the accuracy loss due to the FP22 accumulator when V possesses channel-wise biases: −→Vm = mean(V,axis = 0),V = V −

−→Vm. Furthermore, to maintain the correctness of the attention computation, it is only necessary to add −→Vm to the final calculation of O: O = O + −→Vm. This is because the sum of each row of the P matrix equals 1, so P−→Vm = −→Vm.

Remark. For details on smoothing V, see Appendix A.3. This technique is optional and not employed in our main experiments, as it provides significant benefits only when V exhibits channel-wise bias, which are absent in some models, such as Llama3.1 (see Fig. 2).

##### 4. Experiment

Main result. SageAttention2 is faster than FlashAttention2 and xformers by about 3x and 4.5x. Moreover, SageAttention2 matches the speed of FlashAttn3(fp8) on the Hopper GPUs and is much more accurate than FlashAttn3(fp8). SageAttention2 maintains end-to-end metrics across language, image, and video generation models.

###### 4.1. Setup

Models. We validate the effectiveness of SageAttention2 across a diverse set of repre-

RTX4090, (Head dim = 128, causal = False)

RTX4090, (Head dim = 128, causal = True)

Torch xformers

FlashAttn2 SageAttn1

SageAttn2-8b SageAttn2-4b

Torch xformers

FlashAttn2 SageAttn1

SageAttn2-8b SageAttn2-4b

| |
|---|

| |
|---|

| |
|---|

| |
|---|

800

800

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

600

600

481

479

473

471

465

454

439

432

430

425

424

420

409

405

399

397

391

370

356

344

338

323

322

322

322

317

316

400

400

310

308

307

303

292

259

256

247

188

164

164

164

161

161

161

160

155

151

145

138

118

200

200

106

106

106

105

105

104

104

100

96

93

89

OOM

OOM

OOM

OOM

OOM

OOM

78

55

46

42

16

16

14

0

0

1K 2K 4K 8K 16K 32K Sequence Length

1K 2K 4K 8K 16K 32K Sequence Length

Figure 5. Speed comparison between SageAttention2 and baselines (RTX4090, headdim=128).

- Table 2. End-to-end metrics across text, image, and video generation models. ✗ indicates an inability to generate results for evaluation.

|Model<br><br>|Attention<br><br>|WikiText (Ppl.) ↓|Lambda (Acc.) ↑<br><br>|MMLU (Acc.) ↑|Longbench ↑|
|---|---|---|---|---|---|
|Llama3.1<br><br>|Full-Precision HadmdAttn SmoothAttn SageAttention SageAttn2-4b SageAttn2-8b|6.013 7.872 7.180 6.017 6.256 6.019<br><br>|0.815 0.762 0.783 0.812 0.798 0.811<br><br>|0.635 0.500 0.541 0.634 0.607 0.634|49.40 44.07 44.69 49.55 48.79 49.59<br><br>|
|GLM4<br><br>|Full-Precision HadmdAttn SmoothAttn SageAttention SageAttn2-4b SageAttn2-8b<br><br>|7.241 7.989 8.943 7.243 7.352 7.242|0.432 0.435 0.449 0.433 0.433 0.432<br><br>|0.743 0.669 0.592 0.744 0.725 0.745<br><br>|49.78 45.97 42.20 49.79 49.23 49.60|

|Model|Attention<br><br>|CLIPSIM ↑|CLIP-T ↑<br><br>|VQA-a ↑|VQA-t ↑<br><br>|FScore ↑|
|---|---|---|---|---|---|---|
|CogvideoX (1.5-5B)|Full-Precision<br><br>HadmdAttn<br><br>SmoothAttn<br><br>SageAttention<br><br>FlashAttn3-fp8<br><br>SageAttn2-4b SageAttn2-8b<br><br>|0.1778 0.1576 0.1559 ✗ 0.1562 0.1721 0.1775<br><br>|0.9979 0.9933 0.9950 ✗ 0.9918 0.9978 0.9980|70.231 8.990 8.812 ✗ 6.531 57.729 69.492<br><br>|70.928 2.299 2.277 ✗ 2.181 52.989 74.415|2.507 ✗ ✗ ✗ ✗ 2.884 2.487<br><br>|
|Hunyuan Video<br><br>|Full-Precision HadmdAttn SmoothAttn SageAttention FlashAttn3-fp8 SageAttn2-4b SageAttn2-8b<br><br>|0.1783 0.1727 0.1739 0.1786 0.1742 0.1751 0.1782|0.9995 0.9989 0.9988 0.9995 0.9941 0.9995 0.9996<br><br>|82.516 7.514 6.987 82.496 4.433 81.478 81.786<br><br>|75.934 0.762 0.609 79.843 1.460 65.371 75.354<br><br>|0.604 0.175 0.148 0.597 ✗ 0.610 0.586|
|Mochi<br><br>|Full-Precision HadmdAttn SmoothAttn SageAttention FlashAttn3-fp8 SageAttn2-4b SageAttn2-8b<br><br>|0.1798 0.1733 0.1687 0.1800 0.1762 0.1783 0.1797|0.9986 0.9980 0.9978 0.9987 0.9982 0.9986 0.9986<br><br>|45.549 9.053 3.383 48.707 14.964 35.955 46.760|65.416 25.133 3.480 63.763 13.711 43.735 64.901<br><br>|1.266 0.704 0.241 1.269 0.457 1.137 1.255|

|Model<br><br>|Attention<br><br>|FID ↓|sFID ↓|CLIP ↑<br><br>|IR ↑|
|---|---|---|---|---|---|
|Flux<br><br>|Full-Precision HadmdAttn SmoothAttn<br><br>SageAttention SageAttn2-4b SageAttn2-8b|10.960 11.353 11.149 10.944 10.577 10.927<br><br>|16.648 18.495 19.017 16.641 17.497 16.723<br><br>|26.180 26.123 26.109 26.171 26.141 26.175|1.009 0.965 0.959 1.008 0.998 1.009<br><br>|
|Stable-Dif fusion3.5<br><br>|Full-Precision HadmdAttn SmoothAttn<br><br>SageAttention SageAttn2-4b SageAttn2-8b<br><br>|14.105 14.259 14.161 14.140 14.097 14.106|15.646 15.909 15.649 15.678 15.397 15.647<br><br>|25.505 25.513 25.510 25.503 25.487 25.499<br><br>|0.902 0.886 0.887 0.902 0.895 0.901|

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Figure 6. Comparison examples from HunyuanVideo, prompts are sampled from open-sora prompt sets.

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Figure 7. A comparison example using SageAttn2-8b and FlashAttention3 on CogvideoX-1.5.

sentative models from language, image, and video generation. Specifically, we conduct experiments on ten models: Llama2 (7B) (Touvron et al., 2023), Llama3.1 (8B) (Dubey et al., 2024), and GLM4 (9B) (GLM et al., 2024) for text2text, CogvideoX

- (2B), CogvideoX (1.5-5B) (Yang et al., 2025b), HunyuanVideo (Kong et al., 2024), and Mochi (Team,

2024) for text2video, Flux (schnell) (Black Forest Labs, 2023) and Stable-Diffusion3.5 (turbo) (Stability AI, 2023) for text2image, and TIMM (Wightman, 2019) for image classification.

Datasets and metrics. For Details about the datasets and metrics we used, please refer to Appendix. A.7.

- Table 3. Two kernel implementations of SageAttention2.

|Kernel|ψQ(Q), ψK(K)<br><br>|ψP(P), ψV (V )|
|---|---|---|
|SageAttn2-4b SageAttn2-8b|INT4 per-thread INT8 per-thread<br><br>|FP8 per-block and per-channel FP8 per-block and per-channel|

Implemetation. We implement two attention kernels as shown in Table 3 using CUDA. The 8-bit variant is adapted

for NVIDIA Hopper GPUs, which lack native INT4 tensor core support, and incorporates all techniques described in Sec. 3 except for smoothing Q.

Baselines. (1) SmoothAttn. Following Qserve (Lin et al., 2025), we apply smooth quant for Q,K with smoothing factor α = 0.5. (2) HadmdAttn. Following Quarot (Ashkboos et al., 2024), we apply random Hadamard transformation for Q,K before INT4 quantization. (3) SageAttention (Zhang et al., 2025c), which uses smoothing K, INT8 per-block quantization for Q,K, and FP16 for P,V . (4) FlashAttn3(fp8), the FP8 version of FlashAttention3, which only runs on Hopper GPUs.

###### 4.2. Speed and Accuracy of Kernels

Kernel Speed. We compare the speed of SageAttention2 against baselines using headdim=64 and headdim=128, both with and without Causal Mask (Vaswani, 2017). Detailed setup can be found

- Table 4. Average accuracy across all layers of CogvideoX using different smoothing methods.

Method CosSim ↑ Relative L1 ↓ RMSE ↓

None 80.04% 0.3906 0.2223 HadmdAttn 79.77% 0.3782 0.2180

SmoothAttn 90.21% 0.3383 0.1952 Smooth K 98.07% 0.1493 0.0743 Smooth Q 98.30% 0.1250 0.0712

Smooth Q+K 99.46% 0.0648 0.0334

- Table 5. End-to-end metrics comparison, where Q, K are quantized into INT4, while P, V stay in full precision.

|Q, K|Smooth (Q+K)|Llama3.1 (Lambda) ↑<br><br>|Llama3.1 (WikiText) ↓|CogVideoX (vqa-t) ↑|
|---|---|---|---|---|
|Full-Precision<br><br>|-|81.5%|6.013<br><br>|75.360|
|INT4 Quantization|✗ ✓<br><br>|72.6% 80.8%|11.698 6.219<br><br>|24.670 75.147|

- Table 6. Average accuracy across all layers of CogvideoX using different quantization granularities.

Method Cos Sim ↑ Relative L1 ↓ RMSE ↓ Per-token 99.45% 0.0649 0.0335

Per-thread 99.45% 0.0622 0.0313 Per-block 98.03% 0.1492 0.0744 Per-tensor 97.15% 0.1800 0.0865

- Table 7. Average accuracy using different data types of ( P, V ) across all layers of CogvideoX, where (Q, K) are smoothed.

|Q,K|P,V<br><br>|Cos Sim ↑<br><br>|Relative L1 ↓|RMSE ↓|
|---|---|---|---|---|
|INT4|INT8 E5M2 E4M3 FP16<br><br>|77.05% 99.20%<br><br>99.44%<br><br>99.45%<br><br><br>|0.5618 0.0905 0.0683 0.0649|0.5044 0.0903 0.0347 0.0335<br><br>|

- Table 8. End-to-end generation latency using SageAttention2 (The latency of Llama3.1 is the time to first token generation using different sequence lengths).

SageAttn 2-4b CogvideoX (2B) RTX4090 86 s 54 s 52 s

SageAttn 2-8b

Model GPU Original

CogvideoX (1.5-5B) RTX4090 1040 s 577 s 555 s HunyuanVideo L20 2221 s 1486 s 1435 s

Mochi L20 2336 s 1316 s 1190 s Llama3.1 (48K token) RTX4090 9.2 s 5.7 s 5.6 s Llama3.1 (100K token) L20 39.9 s 25.4 s 23.2 s

in Appendix A.8. Specifically, Fig. 5 shows the speed across varying sequence lengths on RTX4090, indicating that SageAttn2-4b and SageAttn2-8b are approximately 3x and 2.7x faster than FlashAttention2, and about 4.5x and 4x faster than xformers, respectively. Fig. 10, 11, 12, 13, 14, 15, and 16 in Appendix A.2 show more speed results on RTX4090, L20, H20, H100 GPUs.

Accuracy. Table 4 and 17 show the average accuracy of different methods with INT4 Q,K and FP8 P,V across all layers of CogvideoX. The results indicate the accuracy of SageAttn2-4b is superior to other baselines.

###### 4.3. End-to-end Performance

Metrics loss. We assessed the end-to-end metrics of various models using SageAttention2 compared to baselines. Detailed evaluation results are presented in Table 2. The results indicate that SageAttn2-4b outperforms all baselines and maintains most of the end-to-end accuracy across all models. Additionally, SageAttn2-8b incurs almost no metrics loss across various models. More experiment results on other models are shown in Appendix A.9.

Visible image and video examples. Fig. 6, 7, 8, and 9 show some visible comparison examples from HunyuanVideo, Mochi and CogvideoX. We can observe that SageAttn2-8b does not introduce any visible differences compared to full-precision attention, whereas SageAttn2-4b has minor differences but is much better than the baselines.

End-to-end speedup. We compared the original generation latency and the latency using SageAttention2 for models with long sequence lengths in Table 8, observing significant speedup effects. For instance, SageAttention2 achieved a 1.8x speedup in CogvideoX (1.5-5B) without any metrics loss (SageAttn2-8b). SageAttn2-4b further accelerated these models but with a little metrics loss.

###### 4.4. Ablation Study

Overhead of techniques we proposed. As shown in Table 18, the overhead on kernel speed of per-thread quantization, smoothing Q, and two-level accumulation are 0.35%, 3.7%, and 0% compared to the attention kernel.

Benefit of smoothing V. The experiment showing the benefit of smoothing V is shown in Appendix. A.4.

##### 5. Conclusion

We introduce SageAttention2, an efficient and accurate quantized attention. First, we propose to quantize matrices (Q,K) in a thread-level granularity and ( P,V ) to FP8. Second, we propose a method to smooth Q, enhancing the accuracy of QK⊤. Third, we propose a twolevel accumulation strategy to enhance the accuracy of FP8 PV . SageAttention2 is faster than FlashAttention2 and xformers by approximately 3x and 4.5x, respectively. Moreover, SageAttention2 matches the speed of FlashAttention3(fp8) on the Hopper GPUs, but offers significantly higher accuracy. Extensive experiments confirm that our approach maintains end-to-end metrics across language, image, and video generation models.

##### Acknowledgment

This work was supported by the NSFC Projects (Nos. 92270001, 62376131). J.Z is also supported by the XPlorer Prize.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Re, C. Flashattention: Fast and memory-efficient exact attention with IO-awareness. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), Advances in Neural Information Processing Systems, 2022.

##### Impact Statement

This paper presents work that aims to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

##### References

Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Cameron, P., Jaggi, M., Alistarh, D., Hoefler, T., and Hensman, J. Quarot: Outlier-free 4-bit inference in rotated LLMs. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., and Li, J. LongBench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3119–3137, 2024.

Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2023.

Chen, Y., Qian, S., Tang, H., Lai, X., Liu, Z., Han, S., and Jia, J. Longlora: Efficient fine-tuning of long-context large language models. In The International Conference on Learning Representations, 2024.

Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers. In International Conference on Learning Representations, 2021.

- Chu, X., Tian, Z., Wang, Y., Zhang, B., Ren, H., Wei, X., Xia, H., and Shen, C. Twins: Revisiting the design of spatial attention in vision transformers. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems, 2021.
- Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng, Y., Lv, Y., He, J., Lin, J., Zhou, C., and Zhou, J. Qwen2audio technical report, 2024.

Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, 2024.

DeepSeek-AI, Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Guo, D., Yang, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Zhang, H., Ding, H., Xin, H., Gao, H., Li, H., Qu, H., Cai, J. L., Liang, J., Guo, J., Ni, J., Li,

- J., Wang, J., Chen, J., Chen, J., Yuan, J., Qiu, J., Li, J., Song, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang,
- K., Yu, K., Wang, L., Zhang, L., Xu, L., Xia, L., Zhao,
- L., Wang, L., Zhang, L., Li, M., Wang, M., Zhang, M., Zhang, M., Tang, M., Li, M., Tian, N., Huang, P., Wang, P., Zhang, P., Wang, Q., Zhu, Q., Chen, Q., Du, Q., Chen,

- R. J., Jin, R. L., Ge, R., Zhang, R., Pan, R., Wang, R., Xu,

- R., Zhang, R., Chen, R., Li, S. S., Lu, S., Zhou, S., Chen,
- S., Wu, S., Ye, S., Ye, S., Ma, S., Wang, S., Zhou, S., Yu,

- S., Zhou, S., Pan, S., Wang, T., Yun, T., Pei, T., Sun, T., Xiao, W. L., Zeng, W., Zhao, W., An, W., Liu, W., Liang,

- W., Gao, W., Yu, W., Zhang, W., Li, X. Q., Jin, X., Wang,
- X., Bi, X., Liu, X., Wang, X., Shen, X., Chen, X., Zhang,

- X., Chen, X., Nie, X., Sun, X., Wang, X., Cheng, X., Liu,

- X., Xie, X., Liu, X., Yu, X., Song, X., Shan, X., Zhou,

- X., Yang, X., Li, X., Su, X., Lin, X., Li, Y. K., Wang,
- Y. Q., Wei, Y. X., Zhu, Y. X., Zhang, Y., Xu, Y., Xu, Y., Huang, Y., Li, Y., Zhao, Y., Sun, Y., Li, Y., Wang, Y., Yu,

- Y., Zheng, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Tang,

- Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Wu, Y., Ou, Y., Zhu, Y., Wang, Y., Gong, Y., Zou, Y., He,

Y., Zha, Y., Xiong, Y., Ma, Y., Yan, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Wu, Z. F., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Huang, Z., Zhang, Z., Xie, Z., Zhang, Z., Hao, Z., Gou, Z., Ma, Z., Yan, Z., Shao, Z., Xu, Z., Wu, Z., Zhang, Z., Li, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie,

- Z., Song, Z., Gao, Z., and Pan, Z. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. QLoRA: Efficient finetuning of quantized LLMs. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Fu, T., Huang, H., Ning, X., Zhang, G., Chen, B., Wu, T., Wang, H., Huang, Z., Li, S., Yan, S., Dai, G., Yang, H., and Wang, Y. Moa: Mixture of sparse attention for automatic large language model compression. arXiv preprint arXiv:2406.14909, 2024.

Gao, Y., Zeng, Z., Du, D., Cao, S., So, H. K.-H., Cao, T., Yang, F., and Yang, M. Seerattention: Learning intrinsic sparse attention in your llms. arXiv preprint arXiv:2410.13276, 2024.

GLM, T., Zeng, A., Xu, B., Wang, B., Zhang, C., Yin, D., Rojas, D., Feng, G., Zhao, H., Lai, H., Yu, H., Wang, H., Sun, J., Zhang, J., Cheng, J., Gui, J., Tang, J., Zhang, J., Li, J., Zhao, L., Wu, L., Zhong, L., Liu, M., Huang, M., Zhang, P., Zheng, Q., Lu, R., Duan, S., Zhang, S., Cao, S., Yang, S., Tam, W. L., Zhao, W., Liu, X., Xia, X., Zhang,

- X., Gu, X., Lv, X., Liu, X., Liu, X., Yang, X., Song, X., Zhang, X., An, Y., Xu, Y., Niu, Y., Yang, Y., Li, Y., Bai,
- Y., Dong, Y., Qi, Z., Wang, Z., Yang, Z., Du, Z., Hou,
- Z., and Wang, Z. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024.

Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 8340–8349, 2021a.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021b.

Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., and Choi, Y. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7514–7528, 2021.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Hu, Y., Huang, W., Liang, Z., Chen, C., Zhang, J., Zhu, J., and Chen, J. Identifying sensitive weights via postquantization integral. arXiv preprint arXiv:2503.01901, 2025.

Jelinek, F., Mercer, R. L., Bahl, L. R., and Baker, J. K. Perplexity—a measure of the difficulty of speech recognition tasks. The Journal of the Acoustical Society of America, 62(S1):S63–S63, 1977.

Jiang, H., LI, Y., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han, Z., Abdi, A. H., Li, D., Lin, C.-Y., Yang, Y., and Qiu, L. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Kamradt, G. Llmtest needle in a haystack - pressure testing llms. https://github.com/gkamradt/ LLMTest_NeedleInAHaystack, 2023.

Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., Wu, K., Lin, Q., Wang, A., Wang, A., Li, C., Huang, D., Yang, F., Tan, H., Wang, H., Song, J., Bai, J., Wu, J., Xue, J., Wang, J., Yuan, J., Wang, K., Liu, M., Li, P., Li, S., Wang, W., Yu, W., Deng, X., Li, Y., Long, Y., Chen, Y., Cui, Y., Peng, Y., Yu, Z., He, Z., Xu, Z., Zhou, Z., Xu, Z., Tao, Y., Lu, Q., Liu, S., Zhou, D., Wang, H., Yang, Y., Wang, D., Liu, Y., Jiang, J., and Zhong, C. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Lefaudeux, B., Massa, F., Liskovich, D., Xiong, W., Caggiano, V., Naren, S., Xu, M., Hu, J., Tintore, M., Zhang, S., Labatut, P., Haziza, D., Wehrstedt, L., Reizenstein, J., and Sizov, G. xformers: A modular and hackable transformer modelling library. https://github.

com/facebookresearch/xformers, 2022.

Li, D., Kamko, A., Akhgari, E., Sabet, A., Xu, L., and Doshi, S. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.

Li, K., Wang, Y., Peng, G., Song, G., Liu, Y., Li, H., and Qiao, Y. Uniformer: Unified transformer for efficient spatial-temporal representation learning. In International Conference on Learning Representations, 2022.

Lin, Y., Tang, H., Yang, S., Zhang, Z., Xiao, G., Gan, C., and Han, S. QServe:w4a8KV4 quantization and system codesign for efficient LLM serving. In Eighth Conference on Machine Learning and Systems, 2025.

Liu, Y., Cun, X., Liu, X., Wang, X., Zhang, Y., Chen, H., Liu, Y., Zeng, T., Chan, R., and Shan, Y. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22139– 22149, 2024.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. In International Conference on Learning Representations, 2022.

Milakov, M. and Gimelshein, N. Online normalizer calculation for softmax. arXiv preprint arXiv:1805.02867, 2018.

NVIDIA. Parallel thread execution isa version 8.5. https://docs.nvidia.com/cuda/ parallel-thread-execution/.

NVIDIA. CUTLASS: CUDA Templates for Linear Algebra Subroutines and Solvers. GitHub repository, 2023. URL https://github.com/NVIDIA/cutlass.

Panayotov, V., Chen, G., Povey, D., and Khudanpur, S. Librispeech: An asr corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5206–5210, 2015. doi: 10.1109/ICASSP.2015.7178964.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, N.-Q., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fern´andez, R. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525– 1534, 2016.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., and Chen, X. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Stability AI. Introducing stable diffusion 3.5. https://stability.ai/news/ introducing-stable-diffusion-3-5, 2023.

Team, G. Mochi 1. https://github.com/ genmoai/models, 2024.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Vaswani, A. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

Venkataramanan, S., Ghodrati, A., Asano, Y. M., Porikli, F., and Habibian, A. Skip-attention: Improving vision transformers by paying less attention. In The Twelfth International Conference on Learning Representations, 2024.

Wang, H., Ge, S., Lipton, Z., and Xing, E. P. Learning robust global representations by penalizing local predictive power. Advances in Neural Information Processing Systems, 32, 2019.

Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

Wang, Y., Chen, Z., Chen, X., Zhu, J., and Chen, J. Framebridge: Improving image-to-video generation with bridge models. arXiv preprint arXiv:2410.15371, 2024.

Wightman, R. Pytorch image models. https://github. com/rwightman/pytorch-image-models, 2019.

Wu, H., Zhang, E., Liao, L., Chen, C., Hou, J., Wang, A., Sun, W., Yan, Q., and Lin, W. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20144– 20154, 2023.

Xi, H., Yang, S., Zhao, Y., Xu, C., Li, M., Li, X., Lin, Y., Cai, H., Zhang, J., Li, D., et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. In International Conference on Machine Learning (ICML), 2025.

Xiao, C., Zhang, P., Han, X., Xiao, G., Lin, Y., Zhang, Z., Liu, Z., and Sun, M. Infllm: Training-free long-context extrapolation for llms with an efficient context memory. In First Workshop on Long-Context Foundation Models@ ICML 2024, 2024a.

Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, pp. 38087–38099. PMLR, 2023.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2024b.

Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. Imagereward: Learning and evaluating human preferences for text-to-image generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Yang, S., Xi, H., Zhao, Y., Li, M., Zhang, J., Cai, H., Lin, Y., Li, X., Xu, C., Peng, K., et al. Sparse videogen2: Accelerate video generation with sparse attention via semanticaware permutation. arXiv preprint arXiv:2505.18875, 2025a.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In The Thirteenth International Conference on Learning Representations, 2025b.

Yu, W., Luo, M., Zhou, P., Si, C., Zhou, Y., Wang, X., Feng, J., and Yan, S. Metaformer is actually what you need for vision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10819– 10829, 2022.

Zhang, J., Huang, H., Zhang, P., Wei, J., Zhu, J., and Chen, J. Sageattention2: Efficient attention with smoothing q and per-thread quantization. 2025a.

Zhang, J., Li, G., and Su, J. Sage: A framework of precise retrieval for rag. In 2025 IEEE 41st International Conference on Data Engineering (ICDE), pp. 1388–1401. IEEE Computer Society, 2025b.

Zhang, J., Wei, J., Zhang, P., Chen, J., and Zhu, J. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In The International Conference on Learning Representations, 2025c.

Zhang, P., Chen, Y., Su, R., Ding, H., Stoica, I., Liu, Z., and Zhang, H. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025h.

Zhang, P., Huang, H., Chen, Y., Lin, W., Liu, Z., Stoica, I., Xing, E. P., and Zhang, H. Faster video diffusion with trainable sparse attention. arXiv preprint arXiv:2505.13389, 2025i.

Zhang, P., Wei, J., Zhang, J., Zhu, J., and Chen, J. Accurate int8 training through dynamic block-level fallback. arXiv preprint arXiv:2503.08040, 2025j.

Zhang, X., Chen, Y., Hu, S., Xu, Z., Chen, J., Hao, M., Han, X., Thai, Z., Wang, S., Liu, Z., and Sun, M. ∞Bench: Extending long context evaluation beyond 100K tokens. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15262–15277, 2024.

Zhao, M., Zhu, H., Xiang, C., Zheng, K., Li, C., and Zhu, J. Identifying and solving conditional image leakage in image-to-video diffusion model. arXiv preprint arXiv:2406.15735, 2024.

Zhao, M., He, G., Chen, Y., Zhu, H., Li, C., and Zhu, J. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894,

- 2025a.

Zhao, T., Fang, T., Huang, H., Liu, E., Wan, R., Soedarmadji, W., Li, S., Lin, Z., Dai, G., Yan, S., Yang, H., et al. Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation. In International Conference on Learning Representations,

- 2025b.

Zhang, J., Wei, J., Zhang, P., Xu, X., Huang, H., Wang, H., Jiang, K., Zhu, J., and Chen, J. Sageattention3: Microscaling fp4 attention for inference and an exploration of 8-bit training. arXiv preprint arXiv:2505.11594, 2025d.

Zhang, J., Xiang, C., Huang, H., Wei, J., Xi, H., Zhu, J., and Chen, J. Spargeattn: Accurate sparse attention accelerating any model inference. In International Conference on Machine Learning (ICML), 2025e.

Zhang, J., Xiang, C., Huang, H., Wei, J., Xi, H., Zhu, J., and Chen, J. Spargeattn: Training-free sparse attention accelerating any model inference. 2025f.

Zhang, J., Xu, X., Wei, J., Huang, H., Zhang, P., Chendong, X., Zhu, J., and Chen, J. Sageattention2++: A more efficient implementation of sageattention2. arXiv preprint arXiv:2505.21136, 2025g.

Zheng, K., Lu, C., Chen, J., and Zhu, J. Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36:55502–55542, 2023.

Zheng, K., Chen, Y., Mao, H., Liu, M.-Y., Zhu, J., and Zhang, Q. Masked diffusion models are secretly timeagnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908, 2024a.

Zheng, K., He, G., Chen, J., Bao, F., and Zhu, J. Diffusion bridge implicit models. arXiv preprint arXiv:2405.15885, 2024b.

Zheng, K., Chen, Y., Chen, H., He, G., Liu, M.-Y., Zhu, J., and Zhang, Q. Direct discriminative optimization: Your likelihood-based visual generative model is secretly a gan discriminator. arXiv preprint arXiv:2503.01103, 2025.

Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., and You, Y. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024c.

##### A. Appendix

###### A.1. Visible Comparison Exmaples

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Figure 8. Comparison examples from CogvideoX (2B), prompts are sampled from open-sora prompt sets.

###### A.2. Additional Kernel Speed Comparison

Fig. 10, 11, 12, 13, 14, 15, and 16 compare the speed of SageAttention2 against baselines using headdim=64 and headdim=128, both with and without Causal Mask (Vaswani, 2017), on RTX4090, L20, H100, and H20 GPUs.

Table 9 summarizes the performance gain of different attention methods against baselines on various modern GPUs.

Table 9. Speedup of different attention methods on various GPUs.

Method 3090 4090 A100 L40 L20 H100 H20

- FlashAttention2 1.00 1.00 1.00 1.00 1.00 1.00 1.00

- FlashAttention3 ✗ ✗ ✗ ✗ ✗ 1.37 1.57

FlashAttention3 (fp8) ✗ ✗ ✗ ✗ ✗ 2.63 3.06

- SageAttention1 1.97 1.96 1.37 1.45 1.24 1.53 1.52

- SageAttention2 ✗ 2.93 ✗ 2.60 2.46 2.61 3.12

Table 10. An accuracy example on real tensors of CogvideoX model with or without smoothing V .

Smooth V Cos Sim ↑ Relative L1 ↓ RMSE ↓

✗ 98.25% 0.1980 0.2387 ✓ 99.75% 0.0406 0.0773

###### A.3. Smoothing V

As shown in Fig. 17, this strategy could enhance the accuracy of FP22 for values in PV for the following reasons: Each row of P spans a value range from 0 to 1, and each column of V in some models consistently features channel-wise biases that are exclusively positive or negative, for instance, ranging between 8 and 9 in CogvideoX. Consequently, the values of PV could be quite large. However, the floating-point number representation range is not uniform—it is denser near zero. Therefore, by subtracting the mean −→Vm along the channel dimension from V , the values of PV will be closer to zero, resulting in a higher representational precision (see Fig. 17 for a visual demonstration). Additionally, to maintain the correctness of the attention computation, it is only necessary to add −→Vm to the final calculation of O: O = O + −→Vm. This is because the sum of each row of the P matrix equals 1, so P−→Vm = −→Vm. In other words, this method decomposes V

into two parts: −→Vm and V . For V , it centers the values of each column around zero, which leads to the dot product result between a row from the quantized P matrix and a column from the quantized V matrix being closer to zero. This makes the

representation of FP22 more accurate. Meanwhile, −→Vm is retained in FP16 and is added to O at the end without causing a loss of precision for the −→Vm part.

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

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Figure 9. A comparison example using SageAttn2-8b and FlashAttention3 on Mochi and HunyuanVideo.

###### A.4. Experiment of Smoothing V

Table 10 shows the attention accuracy on real tensors sampled from CogvideoX with and without smoothing V . It demonstrates that smoothing V could improve the accuracy of SageAttention2 when quantizing Q,K to INT4 and P,V to FP8. We find that smoothing V is generally effective for diffusion models (Zheng et al., 2023; 2024b;a; 2025; Zhao et al., 2024; 2025a; Wang et al., 2024).

###### A.5. Theoretical Analysis of Smoothing

In this section, we analyze the benefit of smoothing from a theoretical perspective. Let X ∈ RN×d be N activation tokens of dimension d. Following (Dettmers et al., 2023), we suppose that an activation token follows an Gaussian distribution

RTX4090, (Head dim = 64, causal = False)

RTX4090, (Head dim = 64, causal = True)

800

800

Torch xformers

FlashAttn2 SageAttn1

SageAttn2-8b SageAttn2-4b

Torch xformers

FlashAttn2 SageAttn1

SageAttn2-8b SageAttn2-4b

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

600

600

449

446

445

436

434

420

412

406

395

393

390

389

386

384

372

368

360

360

348

334

326

323

322

322

322

320

400

400

315

313

312

308

298

289

273

263

246

174

167

167

167

164

162

161

159

153

150

143

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

200

200

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

Figure 10. Speed comparison between SageAttention2 and baselines (RTX4090, headdim=64).

L20, (Head dim = 64, causal = False)

L20, (Head dim = 64, causal = True)

Torch xformers

FlashAttn2 SageAttn1

SageAttn2-8b SageAttn2-4b

Torch xformers

FlashAttn2 SageAttn1

SageAttn2-8b SageAttn2-4b

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

263

262

259

257

254

250

248

245

235

230

218

205

205

204

204

202

202

199

198

192

192

190

178

157

200

200

144

143

143

142

141

140

139

136

132

125

120

114

113

113

113

113

112

111

111

107

107

101

98

90

90

90

90

89

89

88

88

87

86

84

81

78

OOM

OOM

OOM

OOM

OOM

OOM

25

24

23

9

8

8

0

0

1K 2K 4K 8K 16K 32K Sequence Length

1K 2K 4K 8K 16K 32K Sequence Length

Figure 11. Speed comparison between SageAttention2 and baselines (L20, headdim=64).

L20, (Head dim = 128, causal = False)

L20, (Head dim = 128, causal = True)

Torch xformers

FlashAttn2 SageAttn1

SageAttn2-8b SageAttn2-4b

Torch xformers

FlashAttn2 SageAttn1

SageAttn2-8b SageAttn2-4b

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

273

273

272

270

269

265

264

262

255

249

238

217

217

217

214

214

212

209

207

207

201

201

188

165

200

200

138

137

137

136

135

135

132

130

126

120

116

112

112

111

111

111

110

110

110

108

106

104

98

98

74

74

74

74

73

73

72

72

70

70

67

61

OOM

OOM

OOM

OOM

OOM

OOM

48

45

41

16

16

15

0

0

1K 2K 4K 8K 16K 32K Sequence Length

1K 2K 4K 8K 16K 32K Sequence Length

Figure 12. Speed comparison between SageAttention2 and baselines (L20, headdim=128).

H100, Head Dim = 64, causal = False

H100, Head Dim = 64, causal = True

Xformers FlashAttn2

SageAttention1 FlashAttn3-fp16

FlashAttn3-fp8 SageAttn2-8b

Xformers FlashAttn2

SageAttention1 FlashAttn3-fp16

FlashAttn3-fp8 SageAttn2-8b

| |
|---|

| |
|---|

| |
|---|

| |
|---|

1000

1000

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

750

750

615

606

604

599

599

598

597

588

588

567

564

549

548

524

519

519

498

491

482

474

467

457

456

452

452

450

446

430

407

400

398

393

500

500

383

381

380

374

359

357

355

349

339

331

330

311

303

303

302

300

299

298

297

290

288

274

268

244

228

215

190

181

142

250

250

141

140

139

139

139

135

131

126

124

118

99

0

0

1K 2K 4K 8K 16K 32K Sequence Length

1K 2K 4K 8K 16K 32K Sequence Length

Figure 13. Speed comparison between SageAttention2 and baselines (H100, headdim=64).

N(µ,Σ2), where µ = (µ1,µ2,...,µd) and Σ2 is a diagonal matrix with Σ2 = diag(σ12,σ22,...,σd2). Further, we suppose that different token Xi is i.i.d. sampled from the same distribution. Suppose the absolute maximum value in a quantization group (Hu et al., 2025; Zhang et al., 2025j) is M, and the bit width is b, then there are 2b quantization levels. Under the round-to-nearest strategy, the expected quantization error is 12 · 22Mb

,

which is proportional to the maximum absolute value in the quantization group. So a smaller absolute maximum value leads to smaller quantization error.

H100, Head Dim = 128, causal = False

H100, Head Dim = 128, causal = True

Xformers FlashAttn2

SageAttention1 FlashAttn3-fp16

FlashAttn3-fp8 SageAttn2-8b

Xformers FlashAttn2

SageAttention1 FlashAttn3-fp16

FlashAttn3-fp8 SageAttn2-8b

1500

1500

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

900

892

891

882

878

871

861

852

1000

1000

838

830

817

807

798

776

772

747

707

676

627

614

595

590

576

546

528

518

517

516

506

498

495

495

478

471

470

465

459

453

451

444

441

439

430

410

400

361

343

500

500

339

339

339

339

330

329

328

319

310

307

281

263

217

181

181

180

180

180

177

175

174

169

166

156

134

0

0

1K 2K 4K 8K 16K 32K Sequence Length

1K 2K 4K 8K 16K 32K Sequence Length

Figure 14. Speed comparison between SageAttention2 and baselines (H100, headdim=128).

H20, Head Dim = 64, causal = False

H20, Head Dim = 64, causal = True

Xformers FlashAttn2

SageAttention1 FlashAttn3-fp16

FlashAttn3-fp8 SageAttn2-8b

Xformers FlashAttn2

SageAttention1 FlashAttn3-fp16

FlashAttn3-fp8 SageAttn2-8b

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Speed(TOPS)

400

400

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

277

273

272

271

270

269

266

266

264

263

261

260

255

253

250

247

238

232

227

216

204

185

184

200

200

145

145

142

142

142

141

140

140

140

140

137

137

136

134

132

130

129

123

122

120

118

117

108

97

87

86

86

86

86

85

84

83

80

79

79

74

64

57

57

56

56

56

56

56

55

54

53

51

46

0

0

1K 2K 4K 8K 16K 32K Sequence Length

1K 2K 4K 8K 16K 32K Sequence Length

Figure 15. Speed comparison between SageAttention2 and baselines (H20, headdim=64).

H20, Head Dim = 128, causal = False

H20, Head Dim = 128, causal = True

Xformers FlashAttn2

SageAttention1 FlashAttn3-fp16

FlashAttn3-fp8 SageAttn2-8b

Xformers FlashAttn2

SageAttention1 FlashAttn3-fp16

FlashAttn3-fp8 SageAttn2-8b

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Speed(TOPS)

400

400

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

281

281

279

279

278

276

275

273

272

272

271

266

266

264

260

249

249

247

238

224

218

199

185

200

200

141

141

140

140

140

138

137

136

136

136

135

135

134

134

133

132

132

129

126

124

123

119

115

108

97

90

90

89

89

89

89

87

87

84

82

78

68

63

62

62

62

62

62

61

61

60

59

57

51

0

0

1K 2K 4K 8K 16K 32K Sequence Length

1K 2K 4K 8K 16K 32K Sequence Length

Figure 16. Speed comparison between SageAttention2 and baselines (H20, headdim=128).

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Figure 17. An example of dot product precison a row of P and a column of V presented by FP22 data type.

After smoothing, we have:

1 N

Yij = Xij −

N

Xkj (3)

k=1

and Yij also follows a Gaussian distribution. The mean and variance of Yij can be calculated as follows:

N

N

1 N

1 N

µj = 0 (4)

E[Yij] = E[Xij] −

E[Xkj] = µj −

k=1

k=1

N

(N − 1)2 N2

(N − 1) N

N − 1 N

1 N2

1 N

σj2 + (N − 1)

σj2 =

σj2 (5)

Var[Yij] = Var[

Xij] +

Xkj] =

Var[

k=1,k̸=i

So Yij has a smaller mean and variance compared to Xij. Following the property of Gaussian distribution, we have:

P(|Yij| < ϵ) > P(|Xij| > ϵ), ∀ϵ > 0 (6) So the distribution of Xij is more concentrated toward 0 after smoothing. Then we know that

P(absmax(Yi) < ϵ) =

d

P(|Yij| < ϵ) >

j=1

d

P(|Xij| > ϵ) = P(absmax(Xi) < ϵ) (7)

j=1

So this makes the distribution of absolute max value in a token more concentrated towards 0, leading to smaller quantization error.

###### A.6. Per-Thread Quantization Formulation

[Figure 80]

Figure 18. Memory layout of INT4/INT8 tensor core for accumulator matrix C and D in D = A ∗ B + C among 32 threads (T0 ∼ T31) in a warp. C and D is of shape 16x8. Each thread only holds 4 out of the 128 elements.

To further clarify the per-thread quantization, we first introduce the INT4 MMA instruction of Tensor Core, and then give the formulation of per-thread quantization.

Tensor cores, first introduced in NVIDIA’s Volta architecture, are specialized units designed for efficient matrix-multiplyand-accumulate (MMA) operations. Their usage significantly enhances computational efficiency and performance in AI and high-performance computing (HPC) workloads. Tensor cores compute small tiles of MMA operations, specifically D = A ∗ B + C on a warp (32 contiguous threads) basis. Each thread in the warp holds a fragment of input matrices and will get a fragment of output matrix as a computation result. The INT4 mma.m16n8k64 tensor core operation computes the product of a 16 × 64 INT4 matrix A and a 64 × 8 INT4 matrix B, both stored in registers. It accumulates the result into a 16 × 8 INT32 matrix C, also stored in registers, and returns the final product matrix D, which has the same shape (16 × 8), data type (INT32), and storage location (registers). Each thread holds only 321 of the input and output data. Fig. 18 extracted from the PTX document (NVIDIA) shows the memory layout of matrix C and D among 32 threads in a warp. Each thread only holds 4 out of the 128 result elements.

iδq =⌊(n ∗ 8 ∗ cw/bq)⌋ qi[iδq] ={8 × (n%8) + ⌊(n ∗

cw bq

bq cw }, n ∈ [0,N]

)⌋ ∗

max(|Q[qi[iδq]]|) 7 Qˆ[qi[iδq]] =

δQ[iδq] =

Q[qi[iδq]] δQ[iδq]

iδk =⌊(n ∗ 4/bk)⌋ (8) kn[iδk] ={8 × (n%8) + ⌊n/bk⌋ ∗ bk}∪

{8 × (n%8) + 1 + ⌊n/bk⌋ ∗ bk}, n ∈ [0,N] δK[iδk] =

max(|K[kn[iδk]]|) 7 Kˆ[kn[iδk]] =

K[kn[iδk]] δK[iδk]

By ensuring results held by each thread share a common dequantization scale (belong to the same quantization group), we can avoid the overhead associated with per-token quantization. Leveraging this observation, we design per-thread quantization as formulated in Eq. 8, where cw is the count of GPU Warps, bq and bk are the block size of Q,K, and n is the token index of Q,K. For typical block size of bq = 128, bk = 64 and warp number cw = 4 (as used in FlashAttention2), each warp processes a tile of 32 query tokens and 64 key tokens. Query tokens i,8 + i,16 + i,24 + i (i = 0,1,··· ,7) can be made into one quantization group and key tokens j,1 + j,8 + j,9 + j,··· ,56 + j,57 + j (j = 0,1,2,3) can be made into one quantization group, as visualized in Fig. 4. This design aligns with the memory layout of output matrix D of tensor core shown in Fig. 18, ensuring that each thread only needs one Q scale and one K scale for dequantization.

As a result, this approach creates 32 quantization groups for Q (8 for each of the 4 warps) and 4 quantization groups for K in a 128x64 block, providing 32× and 4× finer granularity compared to per-block quantization for query tokens and key tokens, respectively. Table 6 and Table 15 show the accuracy gains by using per-thread quantization. Per-thread quantization achieves accuracy that closely matches per-token quantization, without introducing any kernel speed degradation (see Table 18 and 19).

###### A.7. Datasets and Metrics in Experiments

Datasets. Text-to-text models are evaluated on four zero-shot tasks: WikiText (Merity et al., 2022) to assess the model’s prediction confidence, LAMBADA (Paperno et al., 2016) evaluate contextual understanding, MMLU (Hendrycks et al., 2021b) for measuring knowledge across various subjects, and Longbench (Bai et al., 2024) for comprehensive assessment of long context understanding capabilities. Text-to-video models are evaluated using the open-sora (Zheng et al., 2024c) prompt sets. Text-to-image models are assessed on MJHQ-30K (Li et al., 2024). TIMM is evaluated on on three image datasets: ImageNet (Deng et al., 2009), ImageNet-Sketch (Sketch) (Wang et al., 2019), and ImageNet-Rendition (ImageNetr) (Hendrycks et al., 2021a).

End-to-end metrics. For text-to-text models, we use perplexity (ppl.) (Jelinek et al., 1977) for WikiText, Accuracy (Acc.) for LAMBADA and MMLU, and Longbench score (Bai et al., 2024). For text-to-video models, following Zhao et al. (2025b), we evaluate the quality of generated videos on five metrics: CLIPSIM and CLIP-Temp (CLIP-T) (Liu et al., 2024) to measure the text-video alignment; (VQA-a) and (VQA-t) to assess the video aesthetic and technical quality, respectively; and Flow-score (FScore) for temporal consistency (Wu et al., 2023). For text-to-image models, generated images are compared with the images in MJHQ-30K dataset in three aspects: FID (Heusel et al., 2017) and sFID (Salimans et al., 2016) for fidelity evaluation, Clipscore (CLIP) (Hessel et al., 2021) for text-image alignment, and ImageReward (IR) (Xu et al., 2023) for human preference. For TIMM, we use classification accuracy.

Accuracy metrics. We use three metrics to assess the accuracy of quantized attention output O′ compared to attention output in full-precision O: First, we flatten O′ and O into vectors in the shape of 1 × n. Then, Cosine similarity: CosSim = OO′/ O2 O′2, Relative L1 distance: L1 = |O − O′|/ |O|, Root mean square error:

RMSE = (1/n) (O − O′)2.

###### A.8. Kernel Benchmark Setup

We benchmark kernel speed with a batch size of 4 and 32 attention heads across a variety of sequence lengths. Benchmarks are conducted using head dimensions of 64 and 128, both with and without Causal Mask (Vaswani, 2017). To generate input tensors for benchmarking, we follow standard practices adopted in prior works such as FlashAttention (Dao et al., 2022). For floating-point data types, inputs are drawn from a Gaussian distribution with mean 0 and standard deviation 1, while for integer data types, inputs are uniformly sampled within the representation range:[-128, 127] for INT8 and [-8, 7] for INT4.

- Table 11. End-to-end metrics on Llama2 (7B).

|Model|Attention<br><br>|WikiText (Ppl.) ↓<br><br>|Lambda (Acc.) ↑<br><br>|MMLU (Acc.) ↑|
|---|---|---|---|---|
|Llama2<br><br>|Full-Precision HadmdAttn SmoothAttn SageAttention SageAttn2-4b SageAttn2-8b<br><br>|5.823 6.771 6.717 5.824 5.912 5.828|0.886 0.860 0.867 0.887 0.881 0.886<br><br>|0.439 0.360 0.392 0.439 0.428 0.438<br><br>|

- Table 12. End-to-end metrics on CogvideoX (2B).

|Model|Attention<br><br>|CLIPSIM ↑|CLIP-T ↑<br><br>|VQA-a ↑|VQA-t ↑<br><br>|FScore ↑|
|---|---|---|---|---|---|---|
|CogvideoX (2B)<br><br>|Full-Precision HadmdAttn SmoothAttn SageAttention SageAttn2-4b SageAttn2-8b<br><br>|0.1836 0.1742 0.1741 0.1833 0.1821 0.1829<br><br>|0.9975 0.9877 0.9870 0.9976 0.9973 0.9977|77.605 29.780 41.703 76.997 77.368 76.532<br><br>|75.360 23.985 47.043 71.360 74.906 74.281<br><br>|3.006 0.499 0.624 2.988 2.603 2.941|

###### C

Table 13. End-to-end metrics on an image classification model.

|Model<br><br>|Attention|ImageNet (Acc.) ↑|Sketch (Acc.) ↑<br><br>|ImageNet-r (Acc.) ↑|
|---|---|---|---|---|
|TIMM|Full-Precision<br><br>HadmdAttn<br><br>SmoothAttn<br><br>SageAttention<br><br>SageAttn2-4b SageAttn2-8b<br><br>|84.79% 84.50% 84.40% 84.74% 86.67% 84.79%<br><br>|45.32%<br><br>44.89%<br><br>44.68%<br>45.38%<br><br><br>45.24% 45.39%<br>|59.55%<br><br>58.80%<br><br>58.73%<br><br>59.95%<br><br><br>59.29%<br><br><br>59.57%|

Table 14. Comparison with FlashAttention3(fp8) on Llama-3-262k (8B) on InfiniBench (Zhang et al., 2024) (H100 GPU).

|Attention<br><br>|Eng.Sum|Eng.QA<br><br>|Eng.MC<br><br>|Code.Debug<br><br>|Math.Find<br><br>|Retr.PassKey|Retr.Num<br><br>|Retr.KV|Avg.|
|---|---|---|---|---|---|---|---|---|---|
|Full-Precision<br><br>FlashAttn3-fp8 SageAttention2<br><br>|18.03 19.03 18.17|12.5 11.73 12.46<br><br>|64.19 55.90 64.19|24.37 22.59 25.63<br><br>|18.29 22.57 17.43<br><br>|100.0 100.0 100.0|100.0 100.0 100.0<br><br>|7.0 0.4 6.6|43.05 41.53 43.06<br><br>|

Table 15. Worst accuracy across all layers of CogvideoX using different quantization granularities.

###### Method Cos Sim ↑ Relative L1 ↓ RMSE ↓

Per-token 96.76% 0.1916 0.0775 Per-thread 96.72% 0.1932 0.0776 Per-block 90.68% 0.3615 0.1490 Per-tensor 85.85% 0.4687 0.2261

NIAH Llama-3-8B-262k

NIAH Llama-3-8B-262k w/ FlashAttn3-fp8

NIAH Llama-3-8B-262k w/ SageAttn2

1.0

1.0

1.0

0.0 11.0 22.0 33.0 44.0 56.0 67.0 78.0 89.0

0.0 11.0 22.0 33.0 44.0 56.0 67.0 78.0 89.0

0.0 11.0 22.0 33.0 44.0 56.0 67.0 78.0 89.0

[Figure 81]

[Figure 82]

[Figure 83]

0.8

0.8

0.8

Depth(%)

Depth(%)

Depth(%)

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

100.0

100.0

100.0

0.0

0.0

0.0

6K38K70K102K134K166K198K230K262K

6K38K70K102K134K166K198K230K262K

6K38K70K102K134K166K198K230K262K

Token Limit

Token Limit

Token Limit

(a) Full Precision

(c) SageAttention2 Figure 19. Needle In A Haystack results on Llama-3-262k (8B).

(b) FlashAttn3-fp8

×105 Quantized value without smoothing Q

Quantized value with smoothing Q

| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

3.0

2.5

Frequency

2.0

1.5

1.0

0.5

0.0

-7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7

-7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7

Figure 20. An example of quantized value distribution of Q before and after smoothing Q.

Table 16. Worst accuracy using different data types of ( P, V ) across all layers of a CogvideoX model, where (Q, K) are smoothed.

|Q,K<br><br>|P,V|Cos Sim ↑<br><br>|Relative L1 ↓|RMSE ↓|
|---|---|---|---|---|
|INT4|INT8 E5M2 E4M3 FP16<br><br>|19.52% 94.94% 96.70% 96.76%<br><br>|0.9579 0.2327 0.1956 0.1916|1.4483 0.2361 0.0779 0.0775<br><br>|

Table 17. Worst accuracy across all layers of CogvideoX using different smooth methods.

###### Method CosSim ↑ Relative L1 ↓ RMSE ↓

None 4.83% 0.9979 0.7784 HadmdAttn 4.85% 0.9978 0.7785

SmoothAttn 64.49% 0.9262 0.7204 Smooth K 90.86% 0.3565 0.1464 Smooth Q 93.10% 0.2989 0.2195

SageAttn2-4b 96.71% 0.1956 0.0779

- A.9. Additional Experiments and Analysis

Additional Results. Table 11, 12 and 13 show results of SageAttention2 and other baselines on Llama2 (7B), CogvideoX (2B) and TIMM.

Results of Super-Long Context. We further conduct experiments on super-long context using Llama-3-262k (8B)1 on

1https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k

Table 18. Overhead of per-thread quantization, smoothing Q, and two-level accumulation techniques measured on L20 GPU.

###### Method TOPS

Attention (INT4 + FP8) 284 + Per-thread quantization 283 + Two-level accumulation 283 + Smoothing Q 273

Table 19. Comparison of different quantization granularities measured on L20 GPU, with QK⊤ in INT4 and PV in FP8.

###### Granularity TOPS

Per-tensor 286 Per-block 284 Per-thread 283 Per-token 268

InfiniBench (Zhang et al., 2024) and Needle-in-a-Haystack (NIAH) (Kamradt, 2023), with sequence lengths reaching up to 262k tokens on an H100 GPU. Since Hopper GPUs lack native INT4 tensor core support, we use SageAttention2-8b for this evaluation. We compare it against FlashAttention3(fp8), ensuring both methods operate under the same bit width. Results are shown in Table 14 and Fig 19. SageAttention2 maintains model performance even under super-long context, while FlashAttention3(fp8) suffers from end-to-end accuracy degradation.

Results of Audio Tasks. We evaluate Qwen2-Audio (7b) (Chu et al., 2024), a speech-to-text model, on the ASR task using the Librispeech (Panayotov et al., 2015) test split and measured its performance with the WER metric (Word Error Rate). As shown in Table 20, SageAttention2 consistently outperforms the baselines, highlighting its effectiveness in audio-related models and benchmarks.

Table 20. End-to-end metrics on Qwen2-Audio (7B).

|Model<br><br>|Attention|Test-Clean ↓<br><br>|Test-Dev ↓|
|---|---|---|---|
|Qwen2-Audio|Full-Precision<br><br>HadmdAttn<br><br>SmoothAttn<br><br>SageAttention<br><br>SageAttn2-4b SageAttn2-8b<br><br>|1.74 1.77 1.76 1.74 1.73 1.72<br><br>|4.01 4.05 4.01 4.02 3.99 4.03|

