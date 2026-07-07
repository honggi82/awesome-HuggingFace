# arXiv:2509.24006v2[cs.LG]19Nov2025

## SLA: BEYOND SPARSITY IN DIFFUSION TRANSFORMERS VIA FINE-TUNABLE SPARSE–LINEAR ATTENTION

Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, Joseph E. Gonzalez, Jun Zhu, Jianfei Chen Tsinghua University, UC Berkeley

{zhang-jt24@mails., jianfeic@, dcszj@}tsinghua.edu.cn

ABSTRACT

In Diffusion Transformer (DiT) models, particularly for video generation, attention latency is a major bottleneck due to the long sequence length and the quadratic complexity. Interestingly, we find that attention weights can be decoupled into two matrices: a small fraction of large weights with high rank and the remaining weights with very low rank. This naturally suggests applying sparse acceleration to the first part and low-rank acceleration to the second. Based on this finding, we propose SLA (Sparse-Linear Attention), a trainable attention method that fuses sparse and linear attention to accelerate diffusion models. SLA classifies attention weights into critical, marginal, and negligible, applying O(N2) attention to critical weights, O(N) attention to marginal weights, and skipping negligible ones. SLA combines these computations into a single GPU kernel and supports both forward and backward passes. With only a few fine-tuning steps using SLA, DiT models achieve a 20× reduction in attention computation, resulting in significant acceleration without loss of generation quality. Experiments show that SLA reduces attention computation by 95% without degrading end-to-end generation quality, outperforming baseline methods. In addition, we implement an efficient GPU kernel for SLA, which yields a 13.7× speedup in attention computation and a 2.2× end-to-end speedup in video generation on Wan2.1-1.3B. The code will be available at https://github.com/thu-ml/SLA.

1 INTRODUCTION

Among the operations in Transformers, attention (Vaswani et al., 2017) is the only one with quadratic computation complexity, while others mostly scale linearly with the sequence length N. In Diffusion Transformer (DiT) models (Peebles & Xie, 2022), especially for video generation, attention becomes the primary computational bottleneck, as the sequence length typically ranges from 10K to 100K. Reducing the cost of attention is therefore critical for improving the efficiency of DiT models. Existing efficient attention methods (Zhang et al.) for DiTs fall into two main categories: (1) numerous sparse attention methods (Li et al., 2025; Zhang et al., 2025a; Xi et al., 2025; Yang et al., 2025a; Zhang et al., 2025c; Wu et al., 2025; Shen et al., 2025; Hassani et al., 2023; Liu et al., 2025), which compute only a subset of attention scores, and (2) a few linear attention methods (Xie et al., 2024; Zhu et al., 2025), which reformulate the operation to achieve O(N) complexity.

Limitation. Despite recent progress, both approaches face challenges in substantially reducing attention computation: (L1) Linear attention methods often fail in practice, especially on video diffusion models. Existing work on linear attention in diffusion is rare and primarily limited to image generation. Our experiments show that when applied to diffusion models, particularly video generation, linear attention severely degrades video quality. (L2) Sparse attention methods rarely achieve very high sparsity and require a considerable fraction of the full complexity of attention. In practice, they typically reach only 40–60% sparsity for sequence length below 50K. Although some recent works (Yang et al., 2025a; Li et al., 2025) report sparsity of 80–85%, such results are obtained on very long sequences (e.g., 100K–300K), where achieving high sparsity is easier.

Key Observation. We find that attention weights in diffusion transformers can be decomposed into two matrices: a small fraction of large weights with high rank and a large fraction of the remaining weights with extremely low rank. This explains why sparse attention or linear attention alone cannot

achieve satisfactory results and naturally suggests applying sparse acceleration to the first part and low-rank acceleration to the second.

Our Method. Based on the observation above, we propose SLA, a trainable hybrid sparse and linear attention for DiT models. Specifically, attention weights are partitioned into blocks and dynamically classified into three categories: critical, marginal, and negligible. Critical blocks are computed exactly using FlashAttention, negligible blocks are skipped, and, unlike existing methods, marginal blocks are processed with linear attention. This design allows sparsity to increase dramatically (e.g., 70%→95%) while maintaining accuracy. Since linear attention is computationally negligible, costing less than 0.5% of full attention in video generation models, SLA is several times faster than sparse attention alone. Furthermore, we implement efficient forward and backward passes for SLA. With a few steps of fine-tuning, SLA significantly reduces the computation complexity and latency of attention while preserving the quality of the generation results.

Result. SLA reduces attention computation by 95% without degrading video generation quality, even at a moderate sequence length of 30K, which is the sequence length in Wan2.1-1.3B. In addition, our implementation achieves a 13.7× speedup in the attention kernel and a 2.2× end-to-end acceleration for video generation, where the attention time becomes almost negligible. SLA consistently surpasses baselines in both generation quality and efficiency.

- 2 PRELIMINARY

- 2.1 BLOCK SPARSE ATTENTION

Given queries, keys, and values Q,K,V ∈ RN×d, the standard attention computes the score matrix S = QK⊤/

√

d and the attention weights P = Softmax(S) to obtain the output O = PV . This is inefficient for large N as it requires O(N2d) operations. The idea of sparse attention is to reduce computation by applying a mask M ∈ {0,1}N×N to the attention weights: P ← P ⊙ M, where ⊙ is the element-wise product. A common strategy is to choose a threshold τ and set Mij = 1 if Pij > τ. For entries with Mij = 0, the multiplications QiKj⊤ and PijVj can be skipped, where Qi = Q[i,:],Kj = K[j,:],Vj = V [j,:].

However, element-wise sparse attention is inefficient on modern GPUs. Practical implementations such as FlashAttention (Dao, 2023) operate at the block level. Specifically, the sparse FlashAttention first partitions Q,K,V,S,P,M into blocks {Qi},{Kj},{Vj},{Sij},{Pij},{Mij}, where Qi ∈ Rb

q×d, Kj,Vj ∈ Rb

kv×d, and Sij,Pij,Mij ∈ Rb

q×bkv. Each block mask Mij is fully filled with either 0 or 1, and we skip the computations of QiK⊤j and PijVj if Mij[:,:] = 0.

- 2.2 LINEAR ATTENTION

Linear attention methods reduce the complexity of standard attention from O(N2d) to O(Nd2). A key idea is to decouple the softmax operation by introducing a feature map ϕ(·) applied to Q and

K. Specifically, it replaces the attention weights in standard attention with ϕ(Q)ϕ(K)

⊤

rowsum(ϕ(Q)ϕ(K)⊤

. This reformulation enables reordering of the matrix multiplications: instead of explicitly computing the attention weights, it first computes ϕ(K)⊤V , and then applies this intermediate result to ϕ(Q):

H = ϕ(K)⊤V, Z = rowsum(ϕ(K)⊤) ∈ Rd×1, O =

ϕ(Q)H ϕ(Q)Z

.

The mapping ϕ(·) is usually an activation function (e.g., ELU + 1 or ReLU (Clevert et al., 2016; Xavier et al., 2011)). This formulation avoids explicitly constructing the N × N matrices S,P and achieves linear computational complexity.

- 3 MOTIVATION AND ANLYSIS

- 3.1 MOTIVATION OF SLA

Due to the softmax operator, the attention weights P lie in [0,1] with each row summing to 1. Furthermore, because of the exponential scaling in softmax, only a small fraction of entries in P

Sparsity = 0% Sparsity = 45% Sparsity = 92%

| |
|---|
| |
| |
| |
|Negligible<br><br>Marginal|
| |
|Critical|

[Figure 1]

[Figure 2]

[Figure 3]

AttentionOutputAttentionWeights

Negligible

Marginal

[Figure 4]

[Figure 5]

[Figure 6]

###### O = PV O = PV O = PV

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Error = 0% Error < 3% Error > 33%

- Figure 1: The left figure shows a typical distribution of attention weights sampled from the Wan2.1 model. The right figure shows the accuracy of sparse attention with different sparsity.

are relatively large, while the vast majority are close to zero. Figure 1 (left) shows the typical distribution of attention weights P sampled from the Wan2.1 model (Wan et al., 2025). We highlight two key observations: (1) Only about 8.1% of the weights are larger than the average value 1/N. (2) A considerable proportion of weights are extremely small. In our case, approximately 45% fall below 1/(100N). As shown in Figure 1 (right), skipping these smallest 45% of weights in sparse attention (i.e., setting the corresponding entries in M to 0) introduces a relative L1 error of less than 3% compared to the full attention output. In contrast, retaining only the largest 8.1% of weights (sparsity = 92%) leads to a sharp increase in error, reaching about 33%. This explains why existing sparse attention methods struggle to achieve a sparsity beyond 90%.

The intermediate values between 1/(100N) and 1/N (the yellow column in Figure 1) present a dilemma: omitting them introduces significant accuracy loss, yet computing them with full attention causes a great decrease in sparsity. Fortunately, these values are far less critical than the largest ones. This finding motivates us to categorize the attention weights into three types: critical, marginal, and negligible. For critical weights, we use sparse FlashAttention to compute the output as they dominate the attention distribution; For negligible weights, we skip the computation; For marginal weights, we employ a linear attention method to reduce the computational complexity to O(Nd2) and enhance the performance of sparse attention.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- Figure 2: Video generation examples on Wan2.1 fine-tuned with full attention, linear attention, sparse attention, and SLA. SLA could achieve a high sparsity of 95% and lossless video quality.

Empirical results. In Figure 2, we present some videos generated by Wan2.1 fine-tuned with different attention methods: using only linear attention, sparse attention with 90% sparsity, and SLA with 95% sparsity. Note that the computational complexity of SLA at 95% sparsity is nearly half that of 90% sparse attention, since the cost of linear attention is almost negligible. For example, in the Wan2.1 model, linear attention accounts for less than 0.5% of the cost of full attention. These empirical results show that SLA significantly outperforms the other two methods in video quality.

- 3.2 SEPARATING ATTENTION WEIGHTS: SPARSE FEW, LOW-RANK MANY

Observation. As shown in Figure 3, full attention weights can be decoupled into two parts: (1) a small subset (< 10%) with rank comparable to full attention, and (2) a large subset (> 90%) with very low rank. Since the methods for accelerating attention focus mainly on sparsity or low-rank structure, this suggests a natural and elegant strategy: apply sparse attention to the first part and low-rank approximation to the second.

Full Attention Weights Top-8% Weights Bottom-92% Weights

[Figure 20]

[Figure 21]

[Figure 22]

= +

Sparse Attention

Low-Rank (Linear) Attention

[Figure 23]

[Figure 24]

[Figure 25]

Rank = 6226 Top-8%, Rank = 6230 Bottom-92%, Rank = 9

- Figure 3: Decomposition of attention weights. We sample attention weights from the Wan2.1 model: the left figure shows the full weights, the middle the top 8%, and the right the bottom 92%.

Previous failures of linear attention are largely due to the high rank of full attention weights (Fan et al., 2025), while linear attention is restricted to a rank at most d. Figure 3 (left) illustrates this with a typical example using the notion of stable rank (Rudelson & Vershynin, 2006). We observe that after removing the top values in the attention weights P, the remaining matrix becomes extremely low-rank. This motivates the decomposition of P using the sparse mask M:

P = P ⊙ M

sparse component

+ P ⊙ (1 − M)

low-rank component

. (1)

Since linear attention is essentially a low-rank version of attention, we are provided with a possibility to replace the low-rank component P ⊙ (1 − M) with linear attention.

- 4 SLA

SLA effectively integrates sparse and linear attention within a unified framework, allowing them to complement each other. In particular, we fuse both attention into a single efficient GPU kernel. In this section, we introduce the sparse and linear attention components of SLA.

SLA first predicts a compressed attention weights matrix Pc ∈ RN/b

q×N/bkv: Pc = Softmax(pool(Q)pool(K)⊤/

√

d). (2)

where pool(·) is a mean pooling operator along the token dimension. For each element of Pc, we classify it into three types and record the results in a compressed mask Mc ∈ RN/b

q×N/bkv.

Specifically, the top kh% positions are marked as critical (labeled 1), the bottom kl% positions as negligible (labeled −1), and the remaining positions as marginal (labeled 0). Formally,

Mc[i,j] = {1 (top kh%), −1 (bottom kl%), 0 (otherwise)}. (3) We apply different methods according to Mc.

- 4.1 SPARSE ATTENTION IN SLA

Guided by the mask Mc, sparse FlashAttention is used to compute the sparse attention output. For each Q block Qi, we iterate over all K,V blocks Kj,Vj with j = 0,...,N/bkv. Whenever Mc[i,j] = 1, we perform:

Sij = QiK⊤j /

√

d, Pij = OnlineSoftmax(Sij), Osi = Osi + PijVj. (4) Here, OnlineSoftmax(·) operator (Milakov & Gimelshein, 2018) computes the softmax of a matrix in a block-wise manner (see lines 10-11 of Algorithm 1 for implementation). The initial value of each Osi is set to zero. Algorithm 1 describes the forward computation of the sparse attention component, and we denote the final output of the sparse attention component Os.

- 4.2 LINEAR ATTENTION IN SLA

Inspired by the idea of low-rank approximation, we replace the low-rank component P ⊙ (1 − M) in Equation 1 with linear attention introduced in Section 2.2 as

ϕ(Q)ϕ(K)⊤ rowsum(ϕ(Q)ϕ(K)⊤) ⊙ (1 − M).

Specifically, the entries of 0 in Mc determine the blocks processed by linear attention. For each query block Qi, we compute the corresponding linear attention output:

ϕ(Qi)Hi ϕ(Qi)Zi

ϕ(Kj)⊤Vj, Zi =

rowsum(ϕ(Kj)⊤), Oli =

. (5)

Hi =

j:Mc[i,j]=0

j:Mc[i,j]=0

Here, as mentioned in Section 2.2, ϕ(·) denotes the activation function, and Hi ∈ Rd×d,Zi ∈ Rd×1 are intermediate results similar to H and Z. Algorithm 1 describes the forward pass of the linear attention component, and the final output of this component is denoted as Ol.

Finally, the overall attention output of SLA is defined as:

##### O = Os + Proj(Ol). (6)

where Proj is a learnable linear transformation Rd → Rd. Applying this projection to Ol helps reduce the distribution mismatch between softmax and linear attention. Its computational cost is O(Nd2), the same as computing Ol and negligible compared with the O(N2d) cost of full attention.

Insight. Linear attention in SLA does not approximate the output corresponding to marginal attention weights, but serves as a learnable compensation that enhances the effectiveness of sparse attention. This is because linear attention alone struggles to approximate the output of full attention (Choromanski et al., 2020; Qin et al., 2022). Therefore, we need to fine-tune the parameters of the target model, enabling it to adapt to the use of linear attention.

###### High-Level Idea Detailed Algorithm

Attention Weight Prediction

###### K→1 K→2 K→3 K→4

bq → d

Pˆ

Q1 K→1 K→2 K→3 K→4 V1 V2 V4

### Q1

###### 0.8 0.1 0 0.1

d → bkv

Sparse FlashAttention

###### Linear Attention

Softmax( pool(Q) pool(K→) )

###### S = (Q1K→1 )

H = ω(K2)→V2 + ω(K4)→V4 Z = sum(ω(K2)→ + ω(K4)→)

Predicted Compressed Attention Weigths

P1 = OnlineSoftmax(S1)

|0.8|0.1|0|0.1|
|---|---|---|---|

Pˆ

Os1 = P1V1 Ol1 = ω(Q1)H/(ω(Q1)Z)

Attention Computation

O1 = Os + Proj(Ol)

|0.8|
|---|

O(N2) Attention O(N) Attention

|0.8|
|---|

(Critical weights) FlashAttention (Marginal weights) Linear Attention

|0.1|
|---|

|0|
|---|

|0.1|
|---|

|0|
|---|

Skip

(Negligible weights) Skip

- Figure 4: Overview of SLA. The left figure illustrates the high-level idea: attention weights are classified into three categories and assigned to computations of different complexity. The right figure shows the detailed forward algorithm of SLA using the predicted compressed attention weights.

- 5 FINE-TUNING USING SLA

To apply SLA to a diffusion model, we can simply replace the original attention with SLA and finetune the model for a few steps on a dataset consistent with the pretraining data. In this section, we describe the forward and backward passes of SLA. Moreover, we detail some additional efficiency optimization for SLA in Appendix A.3.

- 5.1 FORWARD PASS

The formulation of the forward computation was introduced in Section 4. The complete algorithm of the forward pass of SLA is presented in Algorithm 1. It’s worth noting that we precompute hj = ϕ(Kj)⊤Vj and zj = rowsum(ϕ(Kj)⊤) for each pair (Kj,Vj) (Line 4 in Algorithm 1). This design ensures that, when Mc[i,j] = 0, the corresponding operation only involves a single matrix addition (Line 13 in Algorithm 1), thereby improving efficiency. To simplify the notation, we denote Qϕ = ϕ(Q) and Kϕ = ϕ(K) in the following.

- Algorithm 1: Forward pass of SLA.

- 1: Input: Matrices Q, K, V, Qϕ, Kϕ ∈ RN×d, block sizes bq, bkv, hyper-parameters kh, kl.
- 2: Divide Q, Qϕ to Tm = N/bq blocks {Qi} and {Qϕi } ;
- 3: Divide K, V, Kϕ to Tn = N/bkv blocks {Ki}, {Vi} and {Kϕi } ;
- 4: h = {hj} = {(Kϕj )⊤Vj} ; z = {zj} = {rowsum((Kϕj )⊤)} ; // Precompute for linear attention
- 5: Pc = Softmax(pool(Q)pool(K)⊤/

√

d) ; Initialize Mc = 0 ;

- 6: Mc[i, j] = 1 if Pc[i, j] ∈ TopK(Pc[i, :], kh) ; Mc[i, j] = 0 if Pc[i, j] ∈ BottomK(Pc[i, :], kl) ;
- 7: for i = 1 to Tm do
- 8: for j = 1 to Tn do
- 9: if Mc[i, j] = 1 then
- 10: Sij = QiK⊤j /

√

d ; mij = max(mi,j−1, rowmax(Sij)) ; Pij = exp(Sij − mij) ;

- 11: lij = emi,j−1−mijli,j−1 + rowsum(Pij) ; Osij = diag(emi,j−1−mij)Osi,j−1 + PijVj ;
- 12: else if Mc[i, j] = 0 then
- 13: Hi ← Hi + hj; Zi ← Zi + zj ;
- 14: end if
- 15: end for
- 16: Osi = diag(liTn)−1Osi,Tn; Oli = Qϕi Hi/(Qϕi Zi); Li = mi,Tn + log(li,Tn) ;
- 17: end for
- 18: return Os = {Osi}, Ol = {Oli} ;

- Algorithm 2: Backward pass of SLA.

- 1: Input: Q, K, V, Qϕ, Kϕ, Mc, {Li}, {Hi}, {Zi}, Os, Ol from the forward, dOs, dOl ∈ RN×d.
- 2: Ds = rowsum(dOs ⊙ Os), Dl = rowsum(dOl ⊙ Ol), divide Ds, Dl into Tm blocks {Dsi}, {Dli} ;
- 3: for i = 1 to Tm do
- 4: dHi = (Qϕi /(Qϕi Zi))⊤dOli; dZi = −(Qϕi /(Qϕi Zi))⊤Dil ;
- 5: dQϕi = (dOli(Hi)⊤ − DliZ⊤i )/(Qϕi Zi) ;
- 6: end for
- 7: for j = 1 to Tn do
- 8: Initialize dH = 0, dZ = 0 ;
- 9: for i = 1 to Tm do
- 10: if Mc[i, j] = 1 then
- 11: Sij = QiK⊤j /

√

d ; Pij = exp(Sij − Li) ; dVj ← dVj + P⊤ijdOsi ; dPij = dOsijVj⊤ ;

- 12: dSij = Pij ⊙ (dPij − Dsi) ; dQi ← dQi + dSijKj ; dKj ← dKj + dS⊤ijQi ;
- 13: else if Mc[i, j] = 0 then
- 14: dH ← dH + dHi; dZ ← dZ + dZi ;
- 15: end if
- 16: end for
- 17: dKϕj = Vj(dH)⊤ + (dZ)⊤; dVj = Kϕj dH ;
- 18: end for
- 19: return dQ = {dQi}, dK = {dKi}, dV = {dVi}, dQϕ = {dQϕi }, dKϕ = {dKϕi } ;

- 5.2 BACKWARD PASS

The backward pass computes gradients for both the sparse and linear components, which are also fused into a single GPU kernel for efficiency.

Gradient notation. The prefix d or d is used to denote gradients, e.g., dOs,dOl are the gradients of Os,Ol with respect to some loss function ℓ, respectively.

Sparse attention gradients. The output gradient dOs is backpropagated to compute dQ, dK, and dV , following the same derivation as in FlashAttention (Dao, 2023). Given dOs, the backward pass is carried out as follows:

dPij = dOsijVj⊤, Dsi = rowsum(dOsi ⊙ Osi), dSij = Pij ⊙ (dPij − Dsi), dQi = dSijKj, dKj = dS⊤ijQi, dVj = P⊤ijdOsi.

(7)

Here, we consider Dsi ∈ Rb

q×1 as a column vector.

Linear attention gradients. The gradient dOl yields dQϕ,dKϕ,dV through the chain rule:

⊤

⊤

Qϕi Qϕi Zi

Qϕi Qϕi Zi

dOli, Dli = rowsum(dOli ⊙ Oli), dZi = −

#### Dli

dHi =

(8)

(dOli(Hi)⊤ − DliZ⊤i ) Qϕi Zi

dQϕi =

, dKϕj = Vj(dHi)⊤ + (dZi)⊤, dVj = Kϕj dHi

Here, dKϕj and dVj are obtained by aggregating dHi and dZi. Similar to the forward pass, each dHi and dZi is precomputed so that the remaining computation reduces to simple matrix additions. The detailed algorithm is provided in Algorithm 2.

- 6 EXPERIMENT

- 6.1 SETUP

Model and Datasets. We use the Wan2.1-1.3B model (Wan et al., 2025) for video generation experiments in the main text and LightningDiT (Yao et al., 2025) for image generation experiments in the Appendix A.2. For video experiments, we use a private dataset collected from websites such as Pexels (Pexels) and Common Crawl (Common Crawl), consisting of 20,000 5-second videos at 480p resolution for fine-tuning. For image experiments, following LightningDiT (Yao et al., 2025), we use the ImageNet (Deng et al., 2009) dataset at a resolution of 512 × 512.

Baselines. We compare SLA with state-of-the-art sparse attention methods applicable to diffusion models, including (1) VSA (Zhang et al., 2025c), (2) VMoBa (Wu et al., 2025), and (3) the trainingfree SparseAttn (Zhang et al., 2025a) (Sparge-F) and (4) a trainable implementation of SpargeAttn (Sparge-T). For VSA and VMoBa, we use their official implementations, while for Sparse-T, we implement the method ourselves because there is no official implementation. In addition, we design several baselines for ablation studies: (5) Linear Only, which applies only linear attention; (6) Sparse Only, which applies only the sparse attention component of SLA; and (7) L+S, which directly sums the attention outputs of the Linear Only and Sparse Only.

Metrics. For video quality, following Zhang et al. (2024a); Yang et al. (2025b), we use four evaluation dimensions of VBench (Zhang et al., 2024a): Imaging Quality (IQ), Overall Consistency (OC), Aesthetic Quality (AQ), Subject Consistency (SC). We also use the Vision Reward (VR) (Xu et al., 2024) for human preference evaluation, Aesthetic Video Quality (VA), and Techniqual Video Quality (VT) (Liu et al., 2023). For image quality, following Yao et al. (2025), we use FID. For attention computation complexity, we use FLOPs (floating point of operations). For attention efficiency, we use FLOPS (floating-point operations per second) for attention kernel efficiency. Specifically, FLOPS here is O(full attention)/t, where O(·) denotes the operation count and t the attention latency. We use seconds for end-to-end generation latency.

Hyper-parameters. We use a training batch size of 64 and fine-tune the Wan2.1 model for 2000 steps. For the activation function ϕ, we use softmax according to our ablation experiments. kh% is 5% and kl% is 10%. For block size, we use bq = bkv = 64. The hyper-parameters for image generation tasks are detailed in Appendix A.2.

- 6.2 EFFECTIVENESS

Table 1 compares the video generation quality and efficiency of SLA with baseline methods on Wan2.1-1.3B, fine-tuned separately with SLA, Full Attention, and each baseline. SLA delivers about a 19.3× efficiency gain while maintaining video quality comparable to Full Attention. Moreover, compared with the baselines, SLA consistently achieves higher quality even under greater sparsity. For example, 95% (1-5%) sparsity in SLA is actually about 3× more efficient than 85% (1-15%) while still producing better video quality.

- 6.3 EFFICIENCY

- Figure 6 compares the kernel speed and end-to-end latency of SLA on Wan2.1–1.3B with an RTX5090. Note that even VSA in 89% sparsity and VMoBa in 85% sparsity, their generation quality

Table 1: Quality and efficiency comparison of SLA and other baseline methods.

Quality Efficiency

Method

VA ↑ VT ↑ IQ ↑ OC ↑ AQ ↑ SC ↑ VR ↑ FLOPs ↓ Sparsity ↑ Full Attention 76.78 82.88 62.5 23.3 56.1 93.0 0.059 52.75T 0%

Sparge-F 0.002 0.026 26.0 4.6 35.7 85.1 -0.216 7.91T 85% Sparge-T 73.83 77.87 61.9 22.7 55.4 93.1 0.014 7.38T 84%

VMoBa 32.33 35.79 58.0 18.8 46.2 89.9 -0.175 7.91T 85% VSA 55.37 64.61 60.6 22.4 51.9 83.6 -0.069 5.92T 89% SLA 76.96 83.92 62.2 23.6 55.9 93.1 0.048 2.74T 95%

Table 2: Ablation results for SLA.

Quality Efficiency

Method

VA ↑ VT ↑ IQ ↑ OC ↑ AQ ↑ SC ↑ VR ↑ FLOPs ↓ Sparsity ↑ Full Attention 76.78 82.88 62.5 23.3 56.1 93.0 0.059 52.75T 0%

Linear Only 0.042 0.099 39.5 3.6 28.8 90.7 -0.213 0.10T 100% Sparse Only 64.00 70.50 57.2 21.8 51.7 88.7 -0.073 7.91T 85%

L+S 29.65 41.15 58.6 18.8 45.3 87.1 -0.105 5.37T 90% SLA (softmax) 76.96 83.92 62.2 23.6 55.9 93.1 0.048 2.73T 95%

SLA (elu+1) 75.50 81.01 62.8 23.5 55.3 92.9 0.034 2.74T 95% SLA (hedgehog) 74.59 82.62 61.9 22.5 54.3 93.2 0.035 3.11T 95%

SLA (Top 5%) 76.96 83.92 62.2 23.6 55.9 93.1 0.048 2.73T 95% SLA (Top 10%) 75.29 82.20 62.5 22.6 55.8 93.5 0.057 5.38T 90% SLA (Top 20%) 75.81 83.82 62.7 22.4 54.5 92.6 0.059 10.65T 80%

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

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

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

- Figure 5: Video examples using Wan2.1 fine-tuned with SLA and baselines. For Linear Only, Sparse Only, Sparge-F, VSA, and VMoBa, only a single frame per prompt is shown, as their video quality is not sufficient. The full visible comparison is in Figure 7 in Appendix A.1.

is already worse than SLA, so higher sparsity settings (e.g., 95%) are not quality-matched comparisons. In the forward pass, SLA achieves a 13.7× speedup over FlashAttention2 and is 1.93× faster than VSA with 95% sparsity and 3.36× faster than VMoBa with 95% sparsity. In the backward pass, it delivers a 6.8× speedup over FlashAttention2, still outperforming VSA and VMoBa. For end-toend video generation, SLA reduces attention latency from 97s to 11s (8.8× reduction), resulting in a 2.2× end-to-end speedup. For fine-tuning overhead, we train Wan2.1-1.3B for only 2,000 steps with a batch size of 64, which is less than 0.1% of the cost of pretraining (typically 105–106 steps with a batch size of 103–104) (Wan et al., 2025).

Attention Original

Others

| |
|---|

| |
|---|

62s 97s

###### VMoBa

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| |13.7x| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | | | |
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

62s 47s

(Sparsity=85%)

VMoBa

62s 28s

(Sparsity=95%)

###### VSA

62s 26s 2.2x E2E Speedup

(Sparsity=89%)

###### VSA

62s 16s

(Sparsity=95%)

###### SLA

62s 11

(Sparsity=95%)

(a) Attention GPU Kernel speed comparison on RTX5090. (b) End-to-end video generation latency comparison.

- Figure 6: Attention kernel speed and end-to-end generation latency of SLA and baselines on Wan2.1-

- 1.3B with RTX5090. FlashAttn refers to FlashAttn2, the fastest available version on RTX5090.

- 6.4 ABLATION STUDY

Fusing sparse and linear attention. To evaluate the effectiveness of SLA in integrating sparse and linear attention, we compare SLA with Sparse Only, Linear Only, and S+L on Wan2.1 in terms of end-to-end generation quality and efficiency. As shown in Table 2, SLA achieves the best generation quality and is more efficient than Sparse Only and S+L, confirming the effectiveness of our fusion strategy.

Activation function in linear attention. To study the effect of the activation function ϕ in the linear attention component of SLA, we evaluate softmax, elu+1, and hedgehog. Results in Table 2 show that softmax generally provides better generation quality and efficiency.

Impact of parameter kh. We vary kh from 5% to 20% and report the results in Table 2. We find that kh = 5% already yields generation quality close to that of full attention. Since kh = 5% saves about half and a quarter of the computation compared with kh = 10% and kh = 20%, it offers the best trade-off between efficiency and quality.

- 6.5 VISIBLE EXAMPLES

Figure 5 and Figure 7 show video examples from Wan2.1-1.3B fine-tuned using SLA and baselines. SLA produces videos comparable to full attention even at 95% sparsity, while other methods exhibit noticeable distortions even at sparsity levels below 90%.

- 7 RELATED WORK

As sequence lengths in generative models (e.g., language and video) grow, the quadratic cost of attention becomes a key bottleneck (Zhang et al.). Many studies aim to improve efficiency in two main directions: sparse and linear attention. Most sparse attention methods (Xiao et al., 2024b;a; Jiang et al., 2024; Gao et al., 2024; Fu et al., 2024; Xi et al., 2025; Zhang et al., 2025a; Ribar et al., 2023; Yang et al., 2025a) speed up inference without training by masking computation at test time. Some (Zhang et al., 2025c; Wu et al., 2025) add sparsity during training, enabling higher sparsity. Linear attention methods (Wang et al., 2020; Choromanski et al., 2020; Katharopoulos et al., 2020; Qin et al., 2024; Yang et al., 2024; Sun et al., 2023) are mainly studied in language models. For DiT, SANA (Xie et al., 2024) and Dig (Zhu et al., 2025) show linear attention works for image generation pre-training, but in video generation, existing methods cannot rely on it alone for lossless quality. Another direction is hardware-efficient attention (Dao et al., 2022; Dao, 2023; Shah et al., 2024; Zhang et al., 2024c;b; 2025b), which optimizes GPU execution through tiling, kernel fusion, and quantization.

- 8 CONCLUSION

We propose SLA, a trainable attention that unifies sparse and linear attention to accelerate Diffusion Transformers. SLA assigns computation according to importance: it computes O(N2) attention for critical weights, O(N) attention for marginal weights, and skips negligible computations. This design enables substantial reductions in attention cost while preserving effectiveness. Experiments show that just a few fine-tuning steps enable SLA to accelerate models effectively. Specifically, SLA

achieves a 20× reduction in attention computation, along with a 13.7× GPU kernel speedup and a

- 2.2× end-to-end speedup on Wan2.1-1.3B, all without degrading the quality of video generation.

REFERENCES

V. L. Arlazarov, E. A. Dinic, M. A. Kronod, and I. A. Faradzev. On economical construction of the transitive closure of an oriented graph. Soviet Mathematics Doklady, 11:1209–1210, 1970.

Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. Rethinking attention with performers. In International Conference on Learning Representations, 2020.

Djork-Arn´e Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). In Proceedings of the International Conference on Learning Representations (ICLR), 2016.

Common Crawl. Common crawl. https://commoncrawl.org/. Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv

preprint arXiv:2307.08691, 2023.

Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory-efficient exact attention with IO-awareness. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Qihang Fan, Huaibo Huang, and Ran He. Breaking the low-rank dilemma of linear attention. In CVPR, 2025.

Tianyu Fu, Haofeng Huang, Xuefei Ning, Genghan Zhang, Boju Chen, Tianqi Wu, Hongyi Wang, Zixiao Huang, Shiyao Li, Shengen Yan, Guohao Dai, Huazhong Yang, and Yu Wang. Moa: Mixture of sparse attention for automatic large language model compression. arXiv preprint arXiv:2406.14909, 2024.

Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Hayden Kwok-Hay So, Ting Cao, Fan Yang, and Mao Yang. Seerattention: Learning intrinsic sparse attention in your llms. arXiv preprint arXiv:2410.13276, 2024.

Ali Hassani, Steven Walton, Jiachen Li, Shen Li, and Humphrey Shi. Neighborhood attention transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6185–6194, 2023.

Huiqiang Jiang, YUCHENG LI, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: O (n log n) sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025.

Akide Liu, Zeyu Zhang, Zhexin Li, Xuehai Bai, Yizeng Han, Jiasheng Tang, Yuanjie Xing, Jichao Wu, Mingyang Yang, Weihua Chen, et al. Fpsattention: Training-aware fp8 and sparsity co-design for fast video diffusion. arXiv preprint arXiv:2506.04648, 2025.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv preprint arXiv:2310.11440, 2023.

Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. arXiv preprint arXiv:1805.02867, 2018.

William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.

Pexels. Pexels: Free stock photos and videos. https://www.pexels.com/.

Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, and Yiran Zhong. cosformer: Rethinking softmax in attention. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= Bl8CQrx2Up4.

Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun, and Yiran Zhong. Lightning attention-2: A free lunch for handling unlimited sequence lengths in large language models. arXiv preprint arXiv:2401.04658, 2024.

Luka Ribar, Ivan Chelombiev, Luke Hudlass-Galley, Charlie Blake, Carlo Luschi, and Douglas Orr. Sparq attention: Bandwidth-efficient llm inference. arXiv preprint arXiv:2312.04985, 2023.

Mark Rudelson and Roman Vershynin. Sampling from large matrices: an approach through geometric functional analysis, 2006. URL https://arxiv.org/abs/math/0503442.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024.

Xuan Shen, Chenxia Han, Yufa Zhou, Yanyue Xie, Yifan Gong, Quanyi Wang, Yiwei Wang, Yanzhi Wang, Pu Zhao, and Jiuxiang Gu. Draftattention: Fast video diffusion via low-resolution attention guidance. arXiv preprint arXiv:2505.14708, 2025.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

Jianzong Wu, Liang Hou, Haotian Yang, Xin Tao, Ye Tian, Pengfei Wan, Di Zhang, and Yunhai Tong. Vmoba: Mixture-of-block attention for video diffusion models. arXiv preprint arXiv:2506.23858, 2025.

Glorot Xavier, Bordes Antoine, and Bengio Yoshua. Deep sparse rectifier neural networks. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 315–323, 2011.

Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. Infllm: Training-free long-context extrapolation for llms with an efficient context memory. In First Workshop on Long-Context Foundation Models@ ICML 2024, 2024a.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2024b.

Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.

Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. arXiv preprint arXiv:2505.18875, 2025a.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In The Thirteenth International Conference on Learning Representations, 2025b.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Fan Zhang, Shulin Tian, Ziqi Huang, Yu Qiao, and Ziwei Liu. Evaluation agent: Efficient and promptable evaluation framework for visual generative models. arXiv preprint arXiv:2412.09645, 2024a.

Jintao Zhang, Rundong Su, Chunyu Liu, Jia Wei, Ziteng Wang, Pengle Zhang, Haoxu Wang, Huiqiang Jiang, Haofeng Huang, Chendong Xiang, et al. A survey of efficient attention methods: Hardware-efficient, sparse, compact, and linear attention.

Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization. arXiv preprint arXiv:2411.10958, 2024b.

Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. arXiv preprint arXiv:2410.02367, 2024c.

Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattn: Accurate sparse attention accelerating any model inference. In International Conference on Machine Learning (ICML), 2025a.

Jintao Zhang, Xiaoming Xu, Jia Wei, Haofeng Huang, Pengle Zhang, Chendong Xiang, Jun Zhu, and Jianfei Chen. Sageattention2++: A more efficient implementation of sageattention2. arXiv preprint arXiv:2505.21136, 2025b.

Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. Vsa: Faster video diffusion with trainable sparse attention. arXiv preprint arXiv:2505.13389, 2025c.

###### Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, and Xinggang Wang. Dig: Scalable and efficient diffusion models with gated linear attention. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7664–7674, 2025.

A APPENDIX

- A.1 MORE VISIBLE EXAMPLES

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

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

- Figure 7: Full video examples generated by the Wan2.1 fine-tuned with SLA and baseline methods. The first prompt is “A polar bear is playing guitar”. The second prompt is “Pacific coast, carmel by the sea ocean and waves”. The third prompt is “a bird building a nest from twigs and leaves”.

Figure 7 demonstrates some additional video examples generated by the Wan2.1 model, fine-tuned with SLA and other attention methods. We can find that SLA consistently achieves higher quality even under bigger sparsity than baselines.

Table 3: Quality and efficiency comparison of SLA and other baselines on image generation.

Quality Efficiency

Method

FID ↓ FLOPs ↓ Sparsity ↑ Full Attention 31.87 12.88G 0%

SpargeAttn-F 206.11 3.66G 71.57% SpargeAttn-T 46.05 3.16G 75.45%

VSA(2D) 35.75 3.62G 75.00% VMoBA(2D) 39.45 3.22G 75.00%

#### SLA 31.49 1.73G 87.50%

- A.2 EXPERIMENTS FOR IMAGE GENERATION

Experimental setup. As described in Section 6.1, we evaluate SLA and the baselines on a pretraining task of LightningDiT (Yao et al., 2025). Specifically, we use the LightningDiT-1p0B/1 model, consisting of 1.03B parameters, trained on the ImageNet (Deng et al., 2009) dataset at a resolution of 512 × 512.

Hyperparameters. All hyperparameters follow (Yao et al., 2025), except that we train for 100000 steps with a batch size of 128. For SLA, we set ϕ to softmax and use a block size of bq = bkv = 64. Metrics. Following (Yao et al., 2025), we adopt FID to assess image quality and FLOPs to measure computational complexity.

Results. The results are summarized in Table 3. At the highest sparsity level, SLA outperforms all other baselines and even surpasses full attention on the FID metric, confirming the advantage of SLA in preserving image quality. This finding is consistent with the video experiments on Wan2.1 reported in Section 6.2.

- A.3 ADDITIONAL EFFICIENCY OPTIMIZATION

Since the efficiency of SLA depends heavily on the sparsity pattern, we introduce several complementary optimizations tailored to different sparsity levels. These optimizations lead to substantial gains in computational efficiency:

Lookup table. When Mc is highly sparse (e.g., sparsity > 90%), scanning entire rows or columns to read mask values causes significant memory overhead. To mitigate this, we preprocess the nonzero positions of each row and column and store them in a lookup table. During computation, only the lookup table is accessed, substantially reducing memory traffic.

Pre-aggregation for linear attention. Although Line 13 in Algorithms 1 and Line 14 in Algorithm 2 require only a single matrix addition, repeatedly performing such additions incurs high overhead when many entries of Mc are 0 (e.g., > 90%). To address this, we precompute the row/column sums j hj and j zj, and then subtract the contributions corresponding to Mc[i,j] ̸= 0. In this way, 90% of the additions can be replaced by only 10% subtractions.

Method of Four Russians. When the number of blocks with Mc[i,j] = 0 is neither very small nor very large (e.g., around 50%), we provide an efficient implementation for Line 13 in Algorithms 1 and Line 14 in Algorithm 2. Specifically, we adopt the Method of Four Russians (Arlazarov et al., 1970). The key idea is to group hj and zj into segments of g consecutive blocks and precompute all 2g possible subset sums within each segment. During the forward pass, any subset of g blocks can then be obtained by a single look-up, rather than summing them on the fly. This scheme allows a theoretical computation reduction by 1/g.

USE OF LARGE LANGUAGE MODELS

We used a language model only for polishing English writing, while all ideas, experiments, results, and interpretations are our own.

