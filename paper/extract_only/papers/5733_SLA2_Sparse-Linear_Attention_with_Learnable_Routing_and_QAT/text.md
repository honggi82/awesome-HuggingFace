## SLA2: Sparse-Linear Attention with Learnable Routing and QAT

Jintao Zhang1 Haoxu Wang1 Kai Jiang1 Kaiwen Zheng1 Youhe Jiang1 Ion Stoica2 Jianfei Chen1 Jun Zhu1 Joseph E. Gonzalez2

# arXiv:2602.12675v1[cs.LG]13Feb2026

### Abstract

Sparse-Linear Attention (SLA) combines sparse and linear attention to accelerate diffusion models and has shown strong performance in video generation. However, (i) SLA relies on a heuristic split that assigns computations to the sparse or linear branch based on attention-weight magnitude, which can be suboptimal. Additionally, (ii) after formally analyzing the attention error in SLA, we identify a mismatch between SLA and a direct decomposition into sparse and linear attention. We propose SLA2, which introduces (I) a learnable router that dynamically selects whether each attention computation should use sparse or linear attention, (II) a more faithful and direct sparselinear attention formulation that uses a learnable ratio to combine the sparse and linear attention branches, and (III) a sparse + low-bit attention design, where low-bit attention is introduced via quantization-aware fine-tuning to reduce quantization error. Experiments show that on video diffusion models, SLA2 can achieve 97% attention sparsity and deliver an 18.6× attention speedup while preserving generation quality.

### 1. Introduction

Trainable sparse attention methods (Zhang et al., 2025c;i; Wu et al., 2025; Zhan et al., 2025) have shown strong performance in diffusion models. They often achieve higher attention sparsity than training-free sparse attention methods (Zhang et al., 2025f; Xi et al., 2025; Chen et al., 2025a). Among them, Sparse-Linear Attention (SLA) (Zhang et al., 2025c) is a promising approach that introduces a linearattention branch to compensate for the sparse-attention branch, improving overall sparsity. SLA has been validated on both image and video diffusion models, such as TurboDiffusion (Zhang et al., 2025h).

Motivation of SLA. SLA finds that, in diffusion models,

1Tsinghua University 2UC Berkeley. Preprint.

the attention map P could be decomposed into a high-sparse part P1 and a low-rank part P2, and P = P1 +P2. SLA can be formulated to P = Ps + proj(Pl), where Ps and Pl are the attention maps of sparse and linear attention, and proj is a trainable projection.

Limitation of SLA and motivation of SLA2. (L1) Mismatch between SLA output and the original sparse-linear decomposition. After an analysis of the difference of the SLA formulation with the original SLA motivation, we find that the sparse attention map Ps of SLA differs from the decomposed sparse attention map P1 by a constant scaling factor. Specifically, we find P1 = αPs, where α is a ratio vector. To compensate for the mismatch, SLA introduces and trains an additional linear attention projection, which may fail to fully address it. We therefore aim to propose a sparse-linear attention formulation that more directly matches the original motivation. (L2) Heuristic routing for sparse and linear attention branches. SLA does not optimally address the key design choice of how to split computation between the sparse and linear branches. In practice, SLA assigns attention associated with larger attention weights to the sparse branch and routes the remaining computation to the linear branch. This heuristic split is not optimal. For example, moving some weights from P1 to P2 via brute-force selection may not increase the rank of P2, while still improving the sparsity of P1. We therefore aim to design a more principled split, guided by a clear optimization objective. Finally, low-bit attention can be introduced to SLA to obtain an additional speedup. We thus aim to incorporate low-bit attention into SLA in a way that introduces as little quantization error as possible, enabling further attention speedup.

Our method. We propose SLA2, a sparse-linear attention method that reformulates sparse linear attention to (1) better match the original motivation, and (2) optimally route between the sparse and linear attention branches. To address (L1), we directly learn the ratio α to combine the sparse and linear attention branches. This formulation aligns exactly with the sparse and linear components decomposition of attention. To address (L2), we formulate the approximation error of combining sparse attention and linear attention relative to full attention, and build a learnable sparse-attention mask predictor R that supports gradient backpropagation.

We train this predictor by minimizing the formulated error. Furthermore, we build low-bit attention on top of sparse attention to achieve additional attention speedups. To reduce the error introduced by low-bit quantization, we integrate the quantization process into training in a quantization-aware manner, enabling the model to better adapt to low-bit quantization and thus improve the accuracy of low-bit attention at inference time.

Result. SLA2 achieves 97% attention sparsity and an 18.6× attention runtime speedup on both Wan2.1-1.3B and Wan2.1-14B. Please note that 97% sparsity corresponds to about 96.7% computation savings after accounting for the linear-attention branch in SLA2. In terms of video generation quality, even at 97% sparsity, SLA2 outperforms the baselines at 90% sparsity in end-to-end video quality, and it even exceeds full attention, which is 0% sparsity.

Contribution. Our contributions are as follows:

- (1) We carefully analyze the limitations of SLA and propose SLA2, a more reasonable sparse-linear attention method. SLA2 includes a learnable router that splits computation between the sparse and linear attention branches, along with a simple yet effective learnable combination for sparse and linear attention branches. For some insight on the design of SLA2, please see Sections 2.2 and 8.
- (2) We integrate quantization-aware training (QAT) into SLA2 to further accelerate attention without degrading endto-end video generation quality, demonstrating the effectiveness of QAT for low-bit attention.
- (3) Experiments show that SLA2 achieves 97% attention sparsity and an 18.6× attention runtime speedup on video diffusion models while maintaining video quality, surpassing baseline methods.

### 2. Preliminaries

#### 2.1. Sparse-Linear Attention

SLA (Sparse-Linear Attention) (Zhang et al., 2025c) combines sparse softmax attention and linear attention using a heuristic sparse attention mask. Below, we describe the computation of SLA.

tion weights using pooled queries and keys:

Pc = softmax pool(Q)pool(K)⊤/

√

d , (1)

where pool(·) applies mean pooling over the token dimension within each token block. For each row of Pc, SLA assigns the top kh% entries to sparse attention and the bottom kl% entries to skipping, with the remaining entries handled by linear attention. In practice, kl is typically small and can be omitted. This procedure yields a binary mask Mc ∈ {0,1}N/b

q×N/bk, where the top kh% entries in each row are set to 1 and the others to 0. Then, we obtain a M ∈ {0,1}N×N by expanding Mc. Sparse attention output. Given M, SLA computes sparse softmax attention only on entries selected by the mask:

P = softmax(S⊙M) ∈ RN×N, Os = PV ∈ RN×d (2) where ⊙ is element-wise multiplication. Linear attention output. For the remaining entries (1−M), SLA applies linear attention:

ϕ(Q) ϕ(K)⊤((1 − M)V ) ϕ(Q)(ϕ(K)⊤(1 − M)1) ∈ RN×d, (3)

Ol =

where 1 ∈ RN×1 is an all-ones vector, and the division is element-wise to perform row-wise normalization.

Final output. The final SLA output is

O = Os + Proj(Ol), (4) where proj(·) ∈ Rd×d is a learnable linear projection.

2.2. Rethinking Sparse-Linear Attention Original motivation of Sparse-Linear Attention. Let

P = softmax(S) ∈ RN×N be the full-attention probability matrix. Given a binary mask M ∈ {0,1}N×N,

we decompose see full attention into two parts:

Notation. Let Q,K,V ∈ RN×d be the query, key, and value matrices, where N is the sequence length and d is the head dimension. Let

√

S = QK⊤/

d ∈ RN×N

be the attention score matrix. We use softmax(·) to denote row-wise softmax. We use ϕ(·) as the activation function for linear attention.

Mask construction. SLA first computes compressed atten-

##### P = P1+P2, P1 = P⊙M, P2 = P⊙(1−M), (5)

where P1 corresponds to the mask-selected attention positions (computed by sparse softmax attention), and P2 corresponds to the remaining positions (approximated by linear attention). The motivation of SLA is to approximate P1 with a sparse-attention distribution and approximate P2 with a linear-attention distribution. With V ∈ RN×d, the full-attention output is

Of = PV = P1V + P2V ∈ RN×d. (6)

Error of the sparse attention branch. Sparse attention does not directly produce P1, because it renormalizes probabilities over the masked positions in each row. Let α denote the probability sum on the masked positions for each query:

α = P11 ∈ RN×1, (7)

where 1 ∈ RN×1 is an all-one vector. The sparse-attention distribution is

P1

α ∈ RN×N, (8) Therefore, Ps is not aligned with P1; it is obtained by rowwise normalizing P1 so that each row sums to 1. In terms of attention output, with Os = PsV ∈ RN×d, the desired sparse attention output is

Ps =

P1V = (α ⊙ Ps)V = α ⊙ Os. (9) As a result, each row has a scale mismatch controlled by α.

How SLA compensates for the mismatch. SLA output is shown in Equation 4. Comparing Equation 6 and using Equation 9, we can interpret proj(Ol) as jointly accounting for the linear component P2V and the residual induced by the sparse attention branch mismatch:

proj(Ol) ≈ P2V + (α − 1) ⊙ Os. (10)

However, this correction is not directly aligned with the original decomposition motivation: the linear attention branch is also forced to offset the sparse attention branch’s scaling error, making the compensation harder to learn.

A more reasonable formulation. A more faithful way to match the decomposition in Equation 5 is

P ≈ α ⊙ Ps + (1 − α) ⊙ Pl, (11)

where α ∈ RN×1. Here, Ps,Pl ∈ RN×N are the attentionweight matrices corresponding to the sparse attention and the linear attention branchs, and each is row-normalized so that every row sums to 1. The attention output is

O = α ⊙ (PsV ) + (1 − α) ⊙ (PlV ). (12)

Here, α⊙Ps better matches P1, which removes the row-wise scaling mismatch in the sparse attention branch; therefore, an extra proj(·) on Ol for compensation is no longer needed. Moreover, (1 − α) ensures that α ⊙ Ps + (1 − α) ⊙ Pl is row-normalized, avoiding magnitude drift of the output.

### 3. SLA2 Design

According to the analysis in Section 2.2 and Equation 12, we present the overall formulation of SLA2 as follows:

O = α ⊙ Os + (1 − α) ⊙ Ol, (13)

|P| |
|---|---|
| | |

|Os| |
|---|---|
| | |

|Q|
|---|

|softmax(QK→ → M)| |
|---|---|
| | |

|MatMul| |
|---|---|
| | |

Learnable Router

[Figure 1]

Soft-TopK

M ω

|ω → Os| |
|---|---|
| | |

R

|K|
|---|

[Figure 2]

|O|
|---|

###### +

|[Figure 3]| |
|---|---|
|(1 → ω) ↑ Ol| |

1 → ω

|Ol| |
|---|---|
| | |

Linear Attention

V

Figure 1. Attention computation pipeline of SLA2.

where α ∈ RN×1 is a learnable vector with values between 0 and 1, and

√

Os = softmax(QK⊤/

d ⊙ M)V,

Ol = norm(ϕ(Q)ϕ(K)⊤ ⊙ (1 − M))V, M = R(Q,K), (14)

where R is a learnable module, which will be explained in Section 4. ϕ(·) is an activation function for linear attention, and we use the softmax function. norm normalizes the sum of rows in a matrix to 1.

Implementation of getting Os and Ol. From Equation 14, it may appear that computing Os and Ol requires full matmuls QK⊤ and PV . In contrast, our implementation is highly efficient. For Os, built on top of the FlashAttention Algorithm, we only perform the matmuls QK⊤ and PV for the positions where M = 1, and skip the other compu-

tations. For Ol, we also do not compute the matmul QK⊤ directly, but first compute K⊤V according to the positions where M = 0. Then we multiply Q with the result. See Algorithm 2 for more details.

### 4. Learnable Router

The learnable router R aims to dynamically output a mask M to decide which probabilities in P should be computed by the sparse attention branch. Its decisions mainly depend on Q and K, and are independent of V . We therefore take Q and K as inputs to R. However, the sequence length N can be large, making R expensive. To reduce its computational cost, we leverage the fact that adjacent tokens in Q and K often exhibit similar distributions (Zhang et al., 2025f). Following (Jiang et al., 2024; Zhang et al., 2025f; Gao et al., 2024), we apply mean pooling over consecutive bq and bk tokens to compress Q and K:

Q¯ = pool(Q) ∈ RN/b

q×d, K¯ = pool(K) ∈ RN/b

k×d.

(15)

To make R learnable, we further introduce two linear projections projq,projk ∈ Rd×d for Q¯ and K¯, respectively. To get M, we perform

Pc = projq(Q¯)projk(K¯)⊤, Mc = Top-k k%,Pc ∈ RN/b

q×N/bk, (16)

where Top-k is applied row-wise, setting the top k% positions to 1 and the others to 0. The compressed mask Mc can be expanded to an N × N mask to support the computation in Equation 14. In practice, our forward and backward GPU kernels for SLA2 only require Mc, since we implement the method efficiently on top of a block-wise FlashAttentionstyle algorithm. We will elaborate on this in Section 7.

Finally, we note that Top-k avoids gradient propagation during training. We therefore replace Top-k with a learnable version during training. The details and the full training procedure are provided in Section 6.

Note that the equations above describe the mathematical computation rather than the GPU kernel implementation. We build the actual efficient kernel on the FlashAttention algorithm to avoid computing the full score matrix S before applying mask M. Instead, we skip unnecessary computations. The detailed algorithm is provided in Sections 6 and 7.

Backward (FP16-only). Let dOs denote the gradient of Os. In our QAT setting, the backward pass is computed entirely in FP16, using the original FP16 inputs (Q,K,V ) and the forward output Os. The gradient of Q,K,V from the sparse attention branch can be formulated as:

##### dQ, dK, dV = backward(dOs, Os, Q, K, V ).

The detailed backward GPU kernel, along with the complete training pipeline, is provided in Section 6.

### 5. Quantization-aware Training

Post-training quantization (PTQ) (Jacob et al., 2018) applies quantization after a model is fully trained. In contrast, quantization-aware training (QAT) (Nagel et al., 2022) incorporates quantization effects during training, allowing the model to adapt its parameters to the quantization error and thereby improving low-bit accuracy at inference time.

In SLA2, we further accelerate the sparse attention branch Os computation using a low-bit attention in a QAT manner. Concretely, during training, we use low-bit attention only in the forward pass, while the backward pass remains fully in FP16. This design enables the attention speedup brought by low-bit attention while minimizing the end-to-end accuracy drop caused by low-bit quantization.

Forward (low-bit attention). Given Q,K,V ∈ RN×d, we apply a low-bit quantized attention in the forward pass. We first quantize Q (Q,sˆ Q = quant(Q)) and K (K,sˆ K = quant(K)), then compute

√

S = dequant(QˆKˆ⊤/

d, sQ, sK), P = softmax(S ⊙ M),

followed by quantizing P (P,sˆ P = quant(P)) and V (Vˆ,sV = quant(V )) and computing

##### Os = dequant(PˆVˆ, sP, sV ).

Algorithm 1 Fine-tuning a diffusion model using SLA2.

- 1: Stage 1: Initialize R and α:
- 2: Sample Q,K,V tensors as dataset D.
- 3: L = MSE(FullAttn(Q, K, V ), SLA2(Q, K, V, k%, R, α));
- 4: Train R,α under different k% according to the loss L ;
- 5: Stage2: Fine-tune the diffusion model Θ and α:
- 6: Replace the attention in Θ by SLA2;
- 7: Fine-tune Θ, α using an end-to-end diffusion loss.

### 6. Training with SLA2

To fine-tune a diffusion model with SLA2, we adopt a twostage training strategy. 1 In the first stage, we seek a better initialization for R and α to ensure stable and effective subsequent fine-tuning of the diffusion model. 2 In the second stage, we fine-tune the entire diffusion model endto-end. In this stage, we directly optimize the diffusion loss over all model parameters Θ, including α, without R, so that the model adapts to high-sparsity attention and can even achieve better performance under high sparsity.

Specifically, in the first stage, we use the Q, K, and V matrices from every attention layer at each diffusion timestep as training data. For each sparsity setting (i.e., different k%, we use 5%, 4%, and 3%), we train R and α. Note that Top-k is non-differentiable. Therefore, throughout the entire training process, we replace the Top-k operator in Equation 16 with a SoftTop-k operator (Ding et al., 2024):

Here, quant(·) maps an FP16 tensor to a low-bit tensor (e.g., INT8 or FP8) along with its scale, and dequant(·) rescales the result back to FP16. We use Q,ˆ K,ˆ P,ˆ Vˆ to denote the quantized tensors and sQ,sK,sP,sV to denote their scales. Our quantization/dequantization scheme follows SageAttention2++ (Zhang et al., 2025g).

SoftTop-k(k%,Pc)ij = σ

(Pc)ij τ

+ λi , (17)

where σ denotes the sigmoid function, τ is a temperature parameter, and λi is solved via binary search to ensure that each row sums to k% × N/bk. The gradient of

Algorithm 2 Forward pass of SLA2.

core motivation of sparse-linear attention is to decompose the attention weights as P = P1 + P2, where P1 is handled by the sparse branch, and P2 is handled by the linear branch. It aims to route a low-rank portion of P to P2 and make P1 as sparse as possible without harming end-to-end quality. We explain the design choices of R by answering three sub-questions:

- 1: Input: Matrices Q,K,V ∈ RN×d, bq,bk, k%, learn-

able projq, projk ∈ Rd×d, and α ∈ RN/b

q×1.

- 2: K = K−colmean(K) ; // smooth K of SageAttention
- 3: Qϕ,Kϕ = ϕ(Q),ϕ(K), Q,¯ K¯ = pool(Q),pool(K);
- 4: Divide Q,Qϕ to Tm = bN

q

blocks {Qi} and {Qϕi } ;

- 5: Divide K,V,Kϕ to Tn=bN

k

blocks {Ki}, {Vi}, {Kϕi }

- 6: h = {hj} = {(Kϕj )⊤Vj} ;
- 7: z = {zj} = {rowsum((Kϕj )⊤)} ; Mc[:,:] = 0 ;
- 8: Pc = softmax(projq(Q¯)projk(K)⊤/

√

d) ;

- 9: Mc = Top-k(Pc,k%) ; // SoftTop-k in stage1 training
- 10: for i = 1 to Tm do
- 11: for j = 1 to Tn do
- 12: if Mc[i,j] = 1 then
- 13: Sij = dequant(quant(Qi)quant(Kj)⊤)/

√

d;

- 14: mij = max(mi,j−1,rowmax(Sij)) ;
- 15: Pij = exp(Sij − mij) ;
- 16: lij = em

i,j−1−mijli,j−1 + rowsum(Pij) ;

- 17: Otmp = dequant(quant(Pij)quant(Vj) ;
- 18: Osij = diag(em

i,j−1−mij)Osi,j−1 + Otmp ;

- 19: else if Mc[i,j] = 0 then
- 20: Hi ← Hi + hj; Zi ← Zi + zj ;
- 21: end if
- 22: end for
- 23: Osi = diag(lT

n

i )−1Osi,T

n

;

- 24: Oli = Qϕi Hi/(Qϕi Zi); Li = mi,T

n

+ log(li,T

n

) ;

- 25: end for
- 26: Os = {Osi}, Ol = {Oli} ;
- 27: return O = α ⊙ Os + (1 − α) ⊙ Ol ;

- (1.a) Why the input of R are Q and K? For each attention layer, the attention weights are determined by the score ma-

trix S = QK⊤/

√

d followed by a row-wise softmax, i.e., P = Softmax(S). Therefore, deciding which positions of P should be assigned to the sparse branch is fundamentally a decision about which positions of S, i.e., the matrix multiplication between Q and K, are likely to contribute most after softmax. This makes (Q,K) the natural and sufficient signals for routing, while V does not affect the formation of

- P and is thus not needed for the routing decision.

(1.b) Why apply pooling to Q and K in R? A naive router that operates on the full QK⊤ would incur O(N2) complexity, which is too expensive. To reduce this cost, we pool adjacent tokens in Q and K using mean pooling to obtain

- Q¯ and K¯. This is still effective because nearby tokens in diffusion transformers often have similar distribution (Jiang et al., 2024; Zhang et al., 2025f; Gao et al., 2024), so the values in QK⊤ vary smoothly across adjacent positions.

- (1.c) Why using projections (projq and projk) in R? Using Q¯K¯⊤ followed by softmax and Top-k is a simple heuristic and may not yield an optimal split of P into a sparse part and a low-rank part. By introducing learnable

projections projq and projk, the router can learn a task-adaptive representation in which Top-k selection better

SoftTop-k is computed using the reparameterization trick (see Ding et al. (2024)), which enables gradient backpropagation. This operator retains key properties of Top-k, such as enforcing a row-wise sum of k% × N/bk. The overall training algorithm is in Algorithms 1, where we use O = SLA2(Q,K,V,k%,R,α) as SLA2 operator. The forward and backward procedures of SLA2, are provided in Algorithms 2, and 3, respectively.

matches the desired decomposition (making P1 much sparser while leaving a portion that is easier for the linear branch to approximate). In particular, this design gener-

alizes the heuristic routing: setting projq = projk = I recovers the original form, while learning these projections under our training objective can produce a more suitable partition.

(2) Why does SLA2 needs two-stage training? We adopt a two-stage training strategy for both training stability and train–inference consistency. First, before end-to-end finetuning of the entire diffusion model, R should be reasonably initialized. Otherwise, unstable and poor routing can make subsequent fine-tuning difficult. Second, the router used at inference relies on hard Top-k, which is non-differentiable and blocks gradient propagation. To train the projection parameters inside R, we therefore use a differentiable SoftTopk operator during Stage 1. After obtaining a good initialization, Stage 2 fine-tunes the full diffusion model while keeping the routing computation aligned with inference (i.e., using hard Top-k for routing), ensuring that the trained

### 7. Inference with SLA2

During inference, we simply replace the attention modules in the diffusion model with SLA2 and run the SLA2 forward pass described in Algorithm 2. Note that the Top-k operation uses the hard Top-k in Equation 16, rather than SoftTop-k.

### 8. Insights

We summarize key insights on SLA design and training in a question-driven format.

(1) Why is the design of R (Equation 4) reasonable? The

Table 1. Quality and efficiency metrics of SLA2 and the baseline methods.

Quality Efficiency

Model Method

IQ ↑ OC ↑ AQ ↑ MS ↑ SC ↑ VR ↑ FLOPs ↓ Sparsity ↑

Full Attention 63.67 20.27 64.41 98.95 95.40 0.1084 52.75T 0% VMoBA 65.31 20.82 64.14 97.80 86.69 0.0936 5.28T

VSA 59.57 19.27 50.60 97.44 87.98 -0.0881 5.40T SLA 63.10 20.88 64.34 97.90 92.54 0.0872 5.40T

90%

Wan2.1

- -T2V
- -1.3B
- -480P

SLA2 67.70 21.62 64.86 98.69 95.54 0.1093 5.51T VMoBA 63.08 21.07 61.96 97.68 79.83 0.0746 2.64T

VSA 55.50 14.95 42.13 96.19 88.34 -0.1309 2.75T SLA 63.14 21.09 62.91 97.83 94.36 0.0881 2.75T

95%

###### SLA2 67.04 21.55 64.90 98.46 95.27 0.1023 2.87T SLA2 66.64 21.42 64.62 98.04 94.83 0.1039 1.82T 97%

Full Attention 68.01 22.44 64.66 99.14 95.93 0.1238 292.6T 0% VMoBA 67.18 20.85 63.64 98.55 94.50 0.1117 29.26T

VSA 64.03 21.27 63.37 98.90 93.65 0.1074 20.92T SLA 67.58 21.62 63.80 98.78 95.74 0.1166 20.92T

90%

Wan2.1

- -T2V
- -14B
- -720P

SLA2 69.63 20.68 66.41 98.84 95.74 0.1238 21.16T VMoBA 21.27 7.96 33.59 99.99 100 -0.0965 14.63T

VSA 47.69 13.90 34.95 97.09 91.12 -0.1822 14.87T SLA 64.43 20.89 61.89 98.86 94.41 0.1078 14.87T

95%

###### SLA2 69.02 21.11 65.55 98.89 95.53 0.1125 15.11T SLA2 66.93 21.12 65.14 98.71 94.42 0.1149 9.26T 97%

model matches the inference-time computation logic.

### 9. Experiments 9.1. Setup

Model and Baselines. We fine-tune SLA2 and baseline methods on the Wan2.1-1.3B-480P and Wan-2.1-14B-720P models (Wan et al., 2025). For the dataset, we use a private video dataset of 3,000 videos (about 5 seconds each) collected from public sources. To construct text–video pairs, we generate a caption for each video using Qwen3-VLFlash and use these captions as text conditioning for both fine-tuning and evaluation. For baselines, we use Full Attention (without training) implemented with FlashAttn2. We also select several state-of-the-art video generation methods with sparse attention mechanism, including SLA (Zhang et al., 2025c), VSA (Zhang et al., 2025i) and VMoBa (Wu et al., 2025). All results are obtained using the official open-source implementations.

Metrics. Following Zhang et al. (2024); Yang et al. (2025b), we evaluate video quality using multiple dimensions from VBench (Zhang et al., 2024), including Imaging Quality (IQ), Overall Consistency (OC), Aesthetic Quality (AQ), Motion Smoothness (MS) and Subject Consistency (SC). In addition, we assess human preference using the Vision Reward metric (VR) (Xu et al., 2024). To quantify computational cost, we use FLOPs (floating-point operations). For

kernel-level efficiency, we report C/t, where C = 4N2d denotes the theoretical amount of computation and t is the execution latency. We also measure the end-to-end inference latency in seconds.

Hyper-parameters. We fine-tune each method for 500 steps. The batch size is set to 64 for the 1.3B model and 15 for the 14B model. We set the block sizes to bq = 128 and bkv = 64. We use k% of 5%, 4%, and 3% for SLA2. For the temperature parameter τ in SoftTop-k, we use τ = 0.1.

#### 9.2. Effectiveness

Table 1 compares the video generation quality and efficiency of SLA2 against baseline methods on the Wan2.1-T2V-1.3B480P and Wan2.1-T2V-14B-720P models. At sparsity levels of 90% and 95%, SLA2 consistently outperforms all baselines across every video quality metric on both models. Even at a higher sparsity of 97%, SLA2 still surpasses all baseline methods at 90% sparsity, while achieving a 29× speedup over Full Attention. Interestingly, we observe that sparse attention methods can even outperform Full Attention on many metrics after fine-tuning. We attribute this to the higher quality of the fine-tuning dataset compared to the that used during pretraining.

Visible examples. Figure 2 shows an example generated by different methods fine-tuned on Wan2.1-T2V-1.3B-480P. The videos produced by SLA2 exhibit the highest quality and maintain content similar to that generated by Full At-

|[Figure 4]<br><br>Full Attention|
|---|

|[Figure 5]<br><br>SLA2 (Sparsity=97%)|
|---|

|[Figure 6]<br><br>SLA (Sparsity=90%)|
|---|

|[Figure 7]<br><br>SLA2 (Sparsity=95%)|
|---|

|[Figure 8]<br><br>VSA (Sparsity=90%)|
|---|

|[Figure 9]<br><br>VMOBA (Sparsity=90%)|
|---|

- Figure 2. Visible examples of SLA2 and baselines on Wan2.1-T2V-1.3B-480P model. The prompt used for generation is in Appendix B.

|[Figure 10]<br><br>Full Attention|
|---|

|[Figure 11]<br><br>SLA2 (Sparsity=95%)|
|---|

|[Figure 12]<br><br>SLA2 (Sparsity=97%)|
|---|

- Figure 3. Visible examples of SLA2 and baselines on Wan2.1-T2V-14B-720P model. The prompt used for generation is in Appendix B.

tention. In contrast, videos from other methods either differ noticeably from Full Attention or show clear distortions.

- Figure 3 presents an example generated by Full Attention and SLA2 on Wan2.1-T2V-14B-720P model. SLA2 brings almost no degradation in video quality.

- 9.3. Efficiency

- Figure 4 illustrates the forward kernel speed of SLA2 and the baseline methods on an RTX5090, measured in TOPS (trillion operations per second). At 97% sparsity, SLA2 achieves a 18.7× speedup over FlashAttn2, and is 11.7× and 2.6× faster than VMoBA and VSA at 95% sparsity, respectively. Note that SLA2 outperforms all baselines, even when SLA2 uses 97% sparsity and the baselines use 90% or 95% sparsity. Figure 5 presents the end-to-end video generation latencies for SLA2 and the baselines. On the Wan-1.3B-480P model, reducing attention latency from 97s to 7s (13.9× speedup) enables SLA2 to achieve a 2.30× reduction in overall end-to-end latency. On the Wan-14B720P model, SLA2 further reduces end-to-end latency by

FlashAttn VMoBA (90%) VMoBA (95%)

VSA (90%) VSA (95%) SLA (90%)

SLA (95%) SLA2 (95%) SLA2 (97%)

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

Forward Kernel Speed

4079

Speed(C/t)(TOPS)

3610

2996

1866

1553

1002

219 288 348

FlashAttnVMoBA

VMoBA

###### VSA

###### VSA

###### SLA2

###### SLA2

###### SLA

###### SLA

(90%)

(95%)

(90%)

(95%)

(90%)

(95%)

(95%)

(97%)

Figure 4. Kernel speed of SLA2 and baselines with different sparsities.

4.35×. Since the Wan2.1-14B-720P model exceeds the VRAM capacity of a single RTX5090, we enable sequential CPU offloading during evaluation. The reported latency already excludes the offloading overhead.

Attention Original

Others

| |
|---|

| |
|---|

62s 97s

###### VMoBa

(Sparsity=90%) 62s 36s

VMoBa

62s 28s

(Sparsity=95%)

###### VSA

62s 25s

(Sparsity=90%)

###### VSA

62s 16s

(Sparsity=95%)

E2E Speedup 2.3x

###### SLA

62s 18s

(Sparsity=90%)

SLA

62s 11s

(Sparsity=95%)

###### SLA2

|7|
|---|

62s

(Sparsity=97%)

(a) End-to-end video generation latency on Wan2.1-1.3B-480P.

|2550s|
|---|

493s

Original

###### VMoBa

|832s|
|---|

(Sparsity=90%) 493s

VMoBa

493s 680s

(Sparsity=95%)

###### VSA

|651s|
|---|

493s

(Sparsity=90%)

###### VSA

493s 409s

(Sparsity=95%)

E2E speedup 4.35x

###### SLA

493s 457s

(Sparsity=90%)

###### SLA

|285s|
|---|

493s

(Sparsity=95%)

###### SLA2

493s 207

(Sparsity=97%)

(b) End-to-end video generation latency on Wan2.1-14B-720P.

- Figure 5. End-to-end generation latency of SLA2 and baselines with different sparsities.

Table 2. Ablation experiments results.

Quality IQ ↑ OC ↑ AQ ↑ MS ↑ SC ↑ VR ↑

Method

Full Attention 63.67 20.27 64.41 98.95 95.40 0.1084 w/o QAT 65.28 20.66 61.85 97.44 94.65 0.0850 Topk-router 63.66 20.9 62.65 97.86 94.26 0.0876 SLA2 66.64 21.42 64.62 98.04 94.83 0.1039

SLA2 (85%) 67.97 21.98 64.79 98.75 95.79 0.1135 SLA2 (90%) 67.70 21.62 64.86 98.69 95.54 0.1093 SLA2 (95%) 67.04 21.55 64.9 98.46 95.27 0.1023 SLA2 (97%) 66.64 21.42 64.62 98.04 94.83 0.1039

#### 9.4. Ablation Study

Quantization-aware training. To evaluate the impact of quantization-aware training (QAT), we fine-tune the same model without QAT and perform quantized inference. As shown in Table 2, the quality of generated videos drops when inference is performed without QAT, which confirms its effectiveness. For efficiency, we evaluate SLA2 both with and without quantization. Low-bit quantization provides an approximately 1.3x kernel speedup.

Learnable router. To evaluate the benefit of the learnable router, we compare it with the Top-k router used in SLA (Zhang et al., 2025c), which directly selects the largest scores in pool(Q)pool(K)⊤. As shown in Table 2, the

learnable router significantly outperforms the Top-k router. Varying sparsity. We vary the sparsity from 85% to 97% and evaluate SLA2 under different sparsity levels. As summarized in Table 2, lower sparsity consistently leads to better performance. Notably, even with 97% sparsity, SLA2 already outperforms all baselines, as shown in Table 1.

### 10. Related Work

Sparse attention and linear attention are two main ways to speed up attention in Transformer-based models. Sparse attention methods can be grouped by whether they require training. Training-free approaches (Xiao et al., 2024; Jiang

- et al., 2024; Gao et al., 2024; Xi et al., 2025; Zhang et al., 2025f; Ribar et al., 2023; Yang et al., 2025a; Li et al., 2025; Chen et al., 2025a; Lai et al., 2025; Zhang et al., 2023; Tang et al., 2024; Zhu et al., 2025a; Lin et al., 2025; Xu
- et al., 2025; Xia et al., 2025; Chen et al., 2025b; Zhang et al., 2025j; Yang et al., 2024b) reduce inference cost by masking attention patterns at test time, while trainable methods (Zhang et al., 2025i; Wu et al., 2025; Zhang et al., 2025c; Zhan et al., 2025; Zhou et al., 2025; Lu et al., 2025; Yuan et al., 2025; Liu et al., 2025a; Zhang et al., 2026; Cai et al., 2025; Liu et al., 2025b; Sun et al., 2025; Tan et al., 2025; Ding et al., 2023) encourage sparsity during training and can support higher sparsity. Linear attention methods (Wang et al., 2020; Choromanski et al., 2020; Katharopoulos et al., 2020; Qin et al., 2024; Yang et al., 2024a; Sun et al., 2023) are mainly studied in language models. In diffusion transformers, SANA (Xie et al., 2024) and Dig (Zhu et al., 2025b) show that linear attention can work for image-generation pre-training; however, for video generation, linear attention alone often cannot keep quality. In addition, hardwarefocused work (Dao et al., 2022; Dao, 2023; Shah et al., 2024; Zhang et al., 2025d;a;e) speeds up attention by improving GPU execution through tiling, kernel fusion, and quantization.

### 11. Conclusion

We presented SLA2, an trainable sparse-linear attention method for diffusion models. It is motivated by two limitations of SLA: its heuristic routing based on the magnitude of attention weights and a mismatch with the decomposition of sparse and linear attention, revealed by our error analysis. SLA2 addresses these issues by introducing a learnable router and a decomposition-consistent mixing formulation. Moreover, SLA2 adopt a sparse + low-bit attention in a quantization-aware fine-tuning way for further acceleration. Experiments show that SLA2 achieves up to 97% attention sparsity and an 18.6× attention speedup, while preserving video generation quality. We hope SLA2 offers an effective and practical way for efficient attention in diffusion models.

### References

Cai, S., Yang, C., Zhang, L., Guo, Y., Xiao, J., Yang, Z., Xu, Y., Yang, Z., Yuille, A., Guibas, L., et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025.

Chen, P., Zeng, X., Zhao, M., Ye, P., Shen, M., Cheng, W., Yu, G., and Chen, T. Sparse-vdit: Unleashing the power of sparse attention to accelerate video diffusion transformers. arXiv preprint arXiv:2506.03065, 2025a.

Chen, R., Mills, K. G., Jiang, L., Gao, C., and Niu, D. Re-ttention: Ultra sparse visual generation via attention statistical reshape. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b.

Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers. In International Conference on Learning Representations, 2020.

Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Re, C. Flashattention: Fast and memory-efficient exact attention with IO-awareness. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), Advances in Neural Information Processing Systems, 2022.

Ding, G., Ye, Z., Zhong, Z., Li, G., and Shao, D. Separate, dynamic and differentiable (smart) pruner for block/output channel pruning on computer vision tasks, 2024. URL https://arxiv.org/abs/ 2403.19969.

Ding, J., Ma, S., Dong, L., Zhang, X., Huang, S., Wang, W., Zheng, N., and Wei, F. Longnet: Scaling transformers to 1,000,000,000 tokens. arXiv preprint arXiv:2307.02486, 2023.

Gao, Y., Zeng, Z., Du, D., Cao, S., Zhou, P., Qi, J., Lai, J., So, H. K.-H., Cao, T., Yang, F., et al. Seerattention: Learning intrinsic sparse attention in your llms. arXiv preprint arXiv:2410.13276, 2024.

Hu, Y., Huang, W., Liang, Z., Chen, C., Zhang, J., Zhu, J., and Chen, J. Identifying sensitive weights via postquantization integral. arXiv preprint arXiv:2503.01901, 2025.

Hu, Y., Singh, H., Maheswaran, M., Xi, H., Hooper, C., Zhang, J., Tomar, A., Mahoney, M. W., Min, S., Farajtabar, M., et al. Residual context diffusion language models. arXiv preprint arXiv:2601.22954, 2026.

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., and Kalenichenko, D. Quantization and training of neural networks for efficient integerarithmetic-only inference. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2704–2713, 2018.

Jiang, H., Li, Y., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han, Z., Abdi, A. H., Li, D., Lin, C.-Y., et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. Advances in Neural Information Processing Systems, 37:52481–52515, 2024.

Jiang, Y., Fu, F., Zhao, W., Rabanser, S., Lane, N. D., and Yuan, B. Cascadia: A cascade serving system for large language models. arXiv preprint arXiv:2506.04203, 2025.

Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Lai, X., Lu, J., Luo, Y., Ma, Y., and Zhou, X. Flexprefill: A context-aware sparse attention mechanism for efficient long-sequence inference. arXiv preprint arXiv:2502.20766, 2025.

Li, X., Li, M., Cai, T., Xi, H., Yang, S., Lin, Y., Zhang, L., Yang, S., Hu, J., Peng, K., et al. Radial attention: O (nlog n) sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025.

Lin, C., Tang, J., Yang, S., Wang, H., Tang, T., Tian, B., Stoica, I., Han, S., and Gao, M. Twilight: Adaptive attention sparsity with hierarchical top-p pruning. arXiv preprint arXiv:2502.02770, 2025.

Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025a.

Liu, A., Zhang, Z., Li, Z., Bai, X., Han, Y., Tang, J., Xing, Y., Wu, J., Yang, M., Chen, W., et al. Fpsattention: Trainingaware fp8 and sparsity co-design for fast video diffusion. arXiv preprint arXiv:2506.04648, 2025b.

Lu, E., Jiang, Z., Liu, J., Du, Y., Jiang, T., Hong, C., Liu, S., He, W., Yuan, E., Wang, Y., et al. Moba: Mixture of block attention for long-context llms. arXiv preprint arXiv:2502.13189, 2025.

Nagel, M., Fournarakis, M., Bondarenko, Y., and Blankevoort, T. Overcoming oscillations in quantizationaware training. In International Conference on Machine Learning, pp. 16318–16330. PMLR, 2022.

Qin, Z., Sun, W., Li, D., Shen, X., Sun, W., and Zhong, Y. Lightning attention-2: A free lunch for handling unlimited sequence lengths in large language models. arXiv preprint arXiv:2401.04658, 2024.

Ribar, L., Chelombiev, I., Hudlass-Galley, L., Blake, C., Luschi, C., and Orr, D. Sparq attention: Bandwidthefficient llm inference. arXiv preprint arXiv:2312.04985, 2023.

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Sun, W., Tu, R.-C., Ding, Y., Jin, Z., Liao, J., Liu, S., and Tao, D. Vorta: Efficient video diffusion via routing sparse attention. arXiv preprint arXiv:2505.18809, 2025.

Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., and Wei, F. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

Tan, X., Chen, Y., Jiang, Y., Chen, X., Yan, K., Duan, N., Zhu, Y., Jiang, D., and Xu, H. Dsv: Exploiting dynamic sparsity to accelerate large-scale video dit training. arXiv preprint arXiv:2502.07590, 2025.

Tang, J., Zhao, Y., Zhu, K., Xiao, G., Kasikci, B., and Han, S. Quest: Query-aware sparsity for efficient long-context llm inference. arXiv preprint arXiv:2406.10774, 2024.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li, Y., Wu, Y., Liu,

- Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang,
- Z., Han, Z., Wu, Z.-F., and Liu, Z. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

Wu, J., Hou, L., Yang, H., Tao, X., Tian, Y., Wan, P., Zhang, D., and Tong, Y. Vmoba: Mixture-of-block attention for video diffusion models. arXiv preprint arXiv:2506.23858, 2025.

Xi, H., Yang, S., Zhao, Y., Xu, C., Li, M., Li, X., Lin, Y., Cai, H., Zhang, J., Li, D., et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

Xi, H., Yang, S., Zhao, Y., Li, M., Cai, H., Li, X., Lin, Y., Zhang, Z., Zhang, J., Li, X., et al. Quant videogen: Auto-regressive long video generation via 2-bit kv-cache quantization. arXiv preprint arXiv:2602.02958, 2026.

Xia, Y., Ling, S., Fu, F., Wang, Y., Li, H., Xiao, X., and Cui, B. Training-free and adaptive sparse attention for efficient long video generation. arXiv preprint arXiv:2502.21079, 2025.

Xiang, C., Liu, J., Zhang, J., Yang, X., Fang, Z., Wang, S., Wang, Z., Zou, Y., Su, H., and Zhu, J. Geometry-aware rotary position embedding for consistent video world model. 2026.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2024.

Xie, E., Chen, J., Chen, J., Cai, H., Tang, H., Lin, Y., Zhang, Z., Li, M., Zhu, L., Lu, Y., et al. Sana: Efficient highresolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.

Xu, J., Huang, Y., Cheng, J., Yang, Y., Xu, J., Wang, Y., Duan, W., Yang, S., Jin, Q., Li, S., et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

Xu, R., Xiao, G., Huang, H., Guo, J., and Han, S. Xattention: Block sparse attention with antidiagonal scoring. arXiv preprint arXiv:2503.16428, 2025.

Yang, S., Kautz, J., and Hatamizadeh, A. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024a.

Yang, S., Sheng, Y., Gonzalez, J. E., Stoica, I., and Zheng, L. Post-training sparse attention with double sparsity. arXiv preprint arXiv:2408.07092, 2024b.

Yang, S., Xi, H., Zhao, Y., Li, M., Zhang, J., Cai, H., Lin, Y., Li, X., Xu, C., Peng, K., et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. Advances in Neural Information Processing Systems (NeurIPS 2025), 2025a.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In The Thirteenth International Conference on Learning Representations, 2025b.

Yuan, J., Gao, H., Dai, D., Luo, J., Zhao, L., Zhang, Z., Xie, Z., Wei, Y., Wang, L., Xiao, Z., et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 23078–23097, 2025.

Zhan, C., Li, W., Shen, C., Zhang, J., Wu, S., and Zhang, H. Bidirectional sparse attention for faster video diffusion training. arXiv preprint arXiv:2509.01085, 2025.

Zhang, F., Tian, S., Huang, Z., Qiao, Y., and Liu, Z. Evaluation agent: Efficient and promptable evaluation framework for visual generative models. arXiv preprint arXiv:2412.09645, 2024.

Zhang, J., Su, R., Liu, C., Wei, J., Wang, Z., Wang, H., Zhang, P., Jiang, H., Huang, H., Xiang, C., et al. Efficient attention methods: Hardware-efficient, sparse, compact, and linear attention.

Zhang, J., Huang, H., Zhang, P., Wei, J., Zhu, J., and Chen, J. Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization. In International Conference on Machine Learning (ICML 2025), 2025a.

Zhang, J., Li, G., and Su, J. Sage: A framework of precise retrieval for rag. In International Conference on Data Engineering (ICDE 2025), 2025b.

Zhang, J., Wang, H., Jiang, K., Yang, S., Zheng, K., Xi, H., Wang, Z., Zhu, H., Zhao, M., Stoica, I., et al. Sla: Beyond sparsity in diffusion transformers via fine-tunable sparse-linear attention. arXiv preprint arXiv:2509.24006, 2025c.

Zhang, J., Wei, J., Huang, H., Zhang, P., Zhu, J., and Chen, J. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In International Conference on Learning Representations (ICLR 2025), 2025d.

Zhang, J., Wei, J., Zhang, P., Xu, X., Huang, H., Wang, H., Jiang, K., Zhu, J., and Chen, J. Sageattention3: Microscaling fp4 attention for inference and an exploration of 8-bit training. Advances in Neural Information Processing Systems (NeurIPS 2025), 2025e.

Zhang, J., Xiang, C., Huang, H., Xi, H., Zhu, J., Chen, J., et al. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. In Fortysecond International Conference on Machine Learning, 2025f.

Zhang, J., Xu, X., Wei, J., Huang, H., Zhang, P., Xiang, C., Zhu, J., and Chen, J. Sageattention2++: A more efficient implementation of sageattention2. arXiv preprint arXiv:2505.21136, 2025g.

Zhang, J., Zheng, K., Jiang, K., Wang, H., Stoica, I., Gonzalez, J. E., Chen, J., and Zhu, J. Turbodiffusion: Accelerating video diffusion models by 100-200 times. arXiv preprint arXiv:2512.16093, 2025h.

Zhang, J., Jiang, K., Xiang, C., Feng, W., Hu, Y., Xi, H., Chen, J., and Zhu, J. SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top-p Masking and Distillation Fine-Tuning. 2026.

Zhang, P., Chen, Y., Huang, H., Lin, W., Liu, Z., Stoica, I., Xing, E., and Zhang, H. Vsa: Faster video diffusion with trainable sparse attention. arXiv preprint arXiv:2505.13389, 2025i.

Zhang, P., Chen, Y., Su, R., Ding, H., Stoica, I., Liu, Z., and Zhang, H. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025j.

Zhang, P., Wei, J., Zhang, J., Zhu, J., and Chen, J. Accurate int8 training through dynamic block-level fallback. arXiv preprint arXiv:2503.08040, 2025k.

Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., R´e, C., Barrett, C., et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710, 2023.

Zhao, M., Yan, B., Yang, X., Zhu, H., Zhang, J., Liu, S., Li, C., and Zhu, J. Ultraimage: Rethinking resolution extrapolation in image diffusion transformers. arXiv preprint arXiv:2512.04504, 2025a.

Zhao, M., Zhu, H., Wang, Y., Yan, B., Zhang, J., He, G., Yang, L., Li, C., and Zhu, J. Ultravico: Breaking extrapolation limits in video diffusion transformers. arXiv preprint arXiv:2511.20123, 2025b.

Zheng, K., Wang, Y., Ma, Q., Chen, H., Zhang, J., Balaji, Y., Chen, J., Liu, M.-Y., Zhu, J., and Zhang, Q. Large scale diffusion distillation via score-regularized continuoustime consistency. arXiv preprint arXiv:2510.08431, 2025.

Zhou, Y., Xiao, Z., Wei, T., Yang, S., and Pan, X. Trainable log-linear sparse attention for efficient diffusion transformers. arXiv preprint arXiv:2512.16615, 2025.

- Zhu, K., Tang, T., Xu, Q., Gu, Y., Zeng, Z., Kadekodi, R., Zhao, L., Li, A., Krishnamurthy, A., and Kasikci, B. Tactic: Adaptive sparse attention with clustering and distribution fitting for long-context llms. arXiv preprint arXiv:2502.12216, 2025a.
- Zhu, L., Huang, Z., Liao, B., Liew, J. H., Yan, H., Feng, J., and Wang, X. Dig: Scalable and efficient diffusion models with gated linear attention. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7664–7674, 2025b.

### A. Backward Pass of SLA2

The backward pass of SLA2 is presented in Algorithm 3. Following SLA (Zhang et al., 2025c), we manually derive the gradients with respect to Q,K,V,Qϕ and Kϕ, while all remaining gradients are computed via PyTorch’s automatic differentiation. Note that dHi and dZi are precomputed, such that the main procedure involves only a single matrix addition (Line 14), thereby improving computational efficiency.

Algorithm 3 Backward pass of SLA2.

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

### B. Prompts Used

- The prompt we used for Figure 2 is: “A first-person perspective video of a morning makeup routine in a bright, minimalist bathroom. The hands apply moisturizer, followed by foundation, concealer, and setting powder using beauty sponges and brushes. Eyeshadow is blended in neutral tones, eyeliner drawn precisely, and mascara applied to define the lashes. The person dots on lip tint and blush for a natural glow. The camera captures close-up details of each step. Natural light floods the scene.”
- The prompt we used for Figure 3 is: “A fluffy domestic cat running joyfully across a sunlit meadow, its ears perked forward and tail held high with excitement. The cat’s eyes are bright and focused, paws swiftly padding through the tall grass, creating natural motion blur. Golden afternoon light filters through the trees in the background, casting soft shadows. The scene radiates warmth and energy. Shot in smooth 4K slow-motion, low-angle close-up tracking shot following the cat’s playful sprint.”

