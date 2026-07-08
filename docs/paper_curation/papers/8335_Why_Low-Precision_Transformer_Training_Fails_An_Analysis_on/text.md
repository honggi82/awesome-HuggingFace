Published as a conference paper at ICLR 2026
WHY
LOW-PRECISION
TRANSFORMER
TRAINING
FAILS: AN ANALYSIS ON FLASH ATTENTION
Haiquan Qiu1
Quanming Yao1,2,3 ∗
1Department of Electronic Engineering, Tsinghua University
2Beijing National Research Center for Information Science and Technology
3State Key laboratory of Space Network and Communications
qhq22@mails.tsinghua.edu.cn, qyaoaa@tsinghua.edu.cn
ABSTRACT
The pursuit of computational efficiency has driven the adoption of low-precision
formats for training transformer models. However, this progress is often hindered
by notorious training instabilities. This paper provides the first mechanistic expla-
nation for a long-standing and unresolved failure case where training with flash
attention in low-precision settings leads to catastrophic loss explosion. Our in-
depth analysis reveals that the failure is not a random artifact but caused by two
intertwined phenomena: the emergence of similar low-rank representations within
the attention mechanism and the compounding effect of biased rounding errors
inherent in low-precision arithmetic. We demonstrate how these factors create a
vicious cycle of error accumulation that corrupts weight updates, ultimately de-
railing the training dynamics. To validate our findings, we introduce a minimal
modification to the flash attention that mitigates the bias in rounding errors. This
simple change stabilizes the training process, confirming our analysis and offer-
ing a practical solution to this persistent problem. Code is available at https:
//github.com/ucker/why-low-precision-training-fails.
1
INTRODUCTION
The pursuit of training ever-larger and more powerful transformer models is a relentless drive for
computational efficiency (Brown et al., 2020; Hoffmann et al., 2022). A key strategy in this endeavor
is the adoption of low-precision numerical formats (Micikevicius et al., 2017; Wang et al., 2018;
Kalamkar et al., 2019; Liu et al., 2024), which promise substantial reductions in memory footprint
and significant boosts in training speed. In industrial practice, it is common to use BF16 for memory-
bound operations like flash attention while pushing compute-bound operations like FFNs to even
lower precisions such as FP8 (Liu et al., 2024; Qwen-Team, 2025). This highlights the heightened
sensitivity of attention mechanisms to numerical precision. Despite the development of stabilization
techniques like QK normalization (Henry et al., 2020; Qwen-Team, 2025), QK-clip (Kimi-Team,
2025) and Gated Attention (Qiu et al., 2025b; Qwen-Team, 2025), the path to further reducing
precision is often blocked by a lack of understanding of the underlying failure mechanisms.
This paper confronts this challenge by dissecting a notorious and long-standing failure issue involv-
ing flash attention. By reducing the memory complexity of the attention mechanism from quadratic
to linear with respect to sequence length, flash attention has become a cornerstone algorithm for
efficient transformer training, making it indispensable for handling the long contexts required by
modern large-scale models (Dao et al., 2022; Dao, 2024; Shah et al., 2024). This failure, arising
in low-precision settings (flash-attention Issue 337, 2024; nanoGPT Issue 303, 2023; nanoGPT Is-
sue 524, 2024; nanoGPT Issue 554, 2024; Lee et al., 2024; Golden et al., 2024), presents a significant
bottleneck. We focus on a specific, reproducible failure case reported by the community (nanoGPT
Issue 303, 2023; nanoGPT Issue 524, 2024), which has remained unresolved for over two years. Our
in-depth analysis provides the first mechanistic explanation for this failure, revealing that it is not
a random artifact but a direct consequence of two intertwined phenomena: the emergence of simi-
lar low-rank representations across different training steps and tokens, and the compounding effect
∗Corresponding author
1
arXiv:2510.04212v4  [cs.LG]  14 Jun 2026

Published as a conference paper at ICLR 2026
of biased rounding errors inherent in low-precision arithmetic. We demonstrate how these biased
rounding errors act as coefficients for the low-rank representations, causing them to accumulate as
a biased gradient update to the weights. This pushes the spectral norm of weights and activations
to increase abnormally, ultimately overwhelming the training dynamics. To validate our analysis,
we introduce a minimal modification to flash attention that mitigates the bias in rounding errors,
allowing the low-rank weight updates to cancel out during training. This experiment confirms our
analysis and stabilizes the training process, offering a practical solution to this persistent problem.
Notations
We use bold lowercase letters for vectors (e.g., δ, m, rm) and bold uppercase letters for
matrices (e.g., Q, K, V). We use notation dW to denote the gradient of the loss ℓwith respect to
variable W for simplicity, e.g., dQ := dℓ/dQ. diag(v) denotes a diagonal matrix with the elements
of vector v on its diagonal. ◦denotes the element-wise product. We use Python-style indexing, e.g.,
M[i, :] denotes the i-th row of matrix M. Subscripts lp and hp distinguish between low-precision
(BF16) and high-precision (FP32) computations. Binary strings are represented using a typewriter
font (e.g., 101010). The where function mimics torch.where. Unless specified otherwise,
operations follow PyTorch’s broadcasting rules. Key claims are highlighted in light cyan box at
the end of some sections.
2
PRELIMINARY
2.1
LOW-PRECISION TRAINING
Low-precision training is a cornerstone of modern deep learning, enabling the development of in-
creasingly large models by reducing memory usage and accelerating computation (Liu et al., 2024;
Tseng et al., 2025). This is achieved by representing weights, activations, and gradients using nu-
merical formats with fewer bits than the standard 32-bit single-precision (FP32). Mixed-precision
training (Micikevicius et al., 2017) has been widely adopted in practice, which combines 16-bit
formats like FP16 or bfloat16 (BF16) for most computations with an FP32 master copy of weights
to maintain accuracy. While FP16 offers higher precision, its limited dynamic range often leads to
gradient underflow, requiring techniques like loss scaling. In contrast, BF16, originally developed
for Google’s TPUs and now widely supported, provides the same dynamic range as FP32, making it
more robust against underflow and a preferred choice for training large language models (Kalamkar
et al., 2019; Wang & Kanwar, 2019). However, the reduced precision of BF16 can still introduce
numerical errors that lead to training failure, the focus of this paper.
The bfloat16 (BF16) format is a 16-bit floating-point representation with 1 sign, 8 exponent, and 7
significand bits. It matches the dynamic range of 32-bit single-precision (FP32) but has lower pre-
cision, making it a popular choice for balancing computation and numerical range in deep learning.
Adding two BF16 numbers involves aligning their exponents, adding the significands, normalizing
the sum, and rounding the result to fit the 7-bit fraction. This final rounding step, typically “round to
nearest, ties to even”, is one of the primary sources of error. While this rounding method is designed
to be unbiased for random data, a sequence of operations on data with a specific distribution could
lead to a biased rounding error. This accumulation of error in one direction is a critical factor to the
training failure observed in low-precision settings. See Appendix B for details of BF16 addition.
2.2
FLASH ATTENTION
Flash Attention (FA) (Dao et al., 2022; Dao, 2024; Shah et al., 2024) is an I/O-aware exact attention
algorithm designed to overcome the memory bottleneck of standard attention. Standard attention,
defined as O = softmax(αQK⊤)V, requires materializing the N × N attention score matrix
S = αQK⊤, leading to a memory complexity of O(N 2) with respect to the sequence length N.
Flash attention reduces this to O(N) by partitioning the input matrices Q, K, V ∈RN×d into blocks
and processing them iteratively. These blocks are loaded from high-bandwidth memory (HBM) into
fast on-chip SRAM, minimizing costly memory transfers.
In this paper, we focus on analyzing flash attention 2. The forward pass of FA computes the output
O and log-sum-exp statistics L using an online softmax method (Algorithm 1). It iterates through
blocks of Q (outer loop) and blocks of K, V (inner loop). For each query block Qi, it maintains
running statistics: the maximum score mi and the normalization factor ℓi. In each inner loop step,
2

Published as a conference paper at ICLR 2026
Isolate Failure Source: Low precision 𝐎!"
in 𝜹!" = rowsum 𝑑𝐎∘𝐎!"
of FA
causes training failure (Claim 1)
Section 3.2
Mitigate Bias in Rounding Error: use
dynamic maximum in safe softmax
Section 4
Find the Root Cause
Section 3.3
𝜹!" −𝜹#" is biased towards 
positive due to safe softmax
and 𝐏(𝐕(Claim 3)
Section 3.3.2
Loss Exploding!!! 💥💥
Low-rank 𝐏𝐊𝑇$𝐗[𝑇] ≈𝐑
is similar across training
steps and tokens (Claim 2)
High and low precision gradient error (biased weight update)
𝒅𝐖!"
𝑸−𝒅𝐖$"
𝑸≈𝜶&
𝜹$" −𝜹!"
𝑇 𝐑 
𝑵
𝑻'𝟏
Low-precision gradient in training increases spectral norm of 
weights and activations
Section 3.3.1
Figure 1: Analysis in different sections. Our paper traces the causal chain of training failure (blue
box) in reverse to identify the root causes.
it computes unnormalized attention scores ¯P(j)
i
= exp(S(j)
i
−m(j)
i ) and updates an unnormalized
output by accumulating the product ¯P(j)
i Vj. This accumulation is a key focus of our analysis.
After iterating through all key/value blocks, the final output block Oi is correctly normalized. This
tiling strategy avoids materializing the full N × N score matrix. The backward pass (Algorithm 2)
leverages the same tiling strategy. It first computes a key intermediate term, δ = rowsum(dO ◦O),
which is central to our investigation. Then, it recomputes attention scores on-the-fly to calculate the
gradient of the scores, dS(j)
i
= P(j)
i
◦(dP(j)
i
−δi), where dP(j)
i
= dOiV⊤
j . The final gradients
dQ, dK, dV are accumulated block-wise. This approach maintains I/O efficiency in both passes.
3
ROOT CAUSES OF INSTABILITY IN FLASH ATTENTION
We first introduce the failure case in Section 3.1. In Section 3.2, we narrow down the source of the
numerical errors within the flash attention. Further analysis in Section 3.3 reveals that the failure
stems from a combination of two factors: the emergence of low-rank representations and the ac-
cumulation of biased rounding errors inherent to BF16 arithmetic. The full process of such failure
from root cause to loss exploding is shown in Fig. 1.
3.1
THE FAILURE CASE OF LOW-PRECISION FLASH ATTENTION
0
2500
5000
7500
10000
12500
15000
17500
20000
Training Steps
3
4
5
6
7
8
9
10
11
Validation Loss
Training Failure of Low-precision FA
High-precision FA
Low-precision FA
Figure 2: The failure case using BF16
and flash attention results in a sudden
loss explosion, while the stable config-
uration converges.
Our investigation targets a well-documented and per-
sistent failure: the catastrophic loss explosion that oc-
curs when training Generative Pre-trained Transformer
2 (GPT-2) models with flash attention in BF16 preci-
sion (nanoGPT Issue 303, 2023; nanoGPT Issue 524,
2024; nanoGPT Issue 554, 2024). This failure case, re-
ported for over two years, manifests as a sudden loss ex-
plosion after several thousand training steps (see Fig. 10).
While empirical workarounds like reverting to standard
attention or using higher precision (FP32) stabilize train-
ing, they come at the cost of efficiency. This instability is
not an isolated incident; the broader community has ob-
served similar failures when training large language mod-
els (Kimi-Team, 2025; Qwen-Team, 2025). These fail-
ures are often empirically linked to phenomena such as
large spectral norms of weights, large activations (Yang
et al., 2023; Rybakov et al., 2024), and attention sinks (Xiao et al., 2023), leading to a suite of fixes
including QK normalization (Henry et al., 2020; Qwen-Team, 2025), QK-clipping (Kimi-Team,
2025), and Gated Attention (Qiu et al., 2025b; Qwen-Team, 2025). Despite these interventions, a
fundamental understanding of the root causes has remained elusive. The absence of a clear causal
chain from numerical error to loss explosion has left the community reliant on ad-hoc patches rather
3

Published as a conference paper at ICLR 2026
than principled solutions, hindering progress in robust low-precision training. This paper provides
the first mechanistic explanation by reproducing the failure, dissecting its root causes, and proposing
a practical, principled solution.
To reproduce the failure, we employ a GPT-2 architecture with 12 layers, 12 attention heads, an em-
bedding dimension of 768, and a context length of 1024. The model is pre-trained on the OpenWeb-
Text dataset (Gokaslan et al., 2019). For deterministic reproducibility, we deviate from a standard
random data loader by recording and reusing the exact sequence of data batches from an initial run
that led to the failure. This ensures all subsequent experiments process identical data in the same
order, isolating the failure from data-related randomness.
We train the model using the AdamW optimizer with β1 = 0.9, β2 = 0.95, and zero weight
decay. The learning rate follows a cosine schedule with a 2000-iteration linear warmup to a peak of
1 × 10−3, decaying to 1 × 10−5. We apply global gradient clipping with a maximum norm of 1.0.
Training is conducted on 4 NVIDIA A100 (80GB) GPUs using PyTorch’s Distributed Data Parallel
(DDP) module. We use automatic mixed precision with BF16 for the forward pass and FP32 for the
backward pass. Each GPU processes a micro-batch size of 32, and with gradient accumulation over
4 steps, the effective global batch size is 524,288 tokens per optimization step.
3.2
ISOLATING THE SOURCE OF FAILURE WITHIN FLASH ATTENTION
To pinpoint the source of failure within flash attention, we conduct a series of targeted experiments.
We systematically modify the algorithm—disabling tiling, selectively replacing flash attention with
a standard implementation, and performing key computations in high precision—to narrow down
the potential causes of instability. To accelerate this analysis, we monitor spectral alignment (Qiu
et al., 2025a) to quickly identify failing configurations.
Tiling is not the Source of Failure.
To determine if the block-wise processing in flash attention
was responsible for the failure, we conduct an experiment where we disabled tiling by setting the
block size equal to the sequence length. This forces the algorithm to compute with the full matrices
at once. The training process still failed, leading to the same loss explosion. This finding rules out
the tiling strategy as the cause of the problem. For all subsequent experiments, we therefore use this
non-tiled setup to simplify the analysis and focus on the core numerical computations.
Failure Originates in a Single Layer.
We first analyze the spectral norms of weights across all
layers (Yang et al., 2023; Rybakov et al., 2024). This reveals an anomalous spike specifically within
the second layer’s attention (see Fig. 11). We confirm this finding with two targeted experiments: (1)
using flash attention only in layer 2 was sufficient to reproduce the training failure, and (2) replac-
ing flash attention with standard attention in layer 2, while retaining it in all other layers, restored
training stability. These results conclusively identify the flash attention in layer 2 as the origin
of the failure. So, subsequent analysis focuses on this module to dissect the failure mechanism.
Head 1
Head 2
Head 3
Head 4
Head 5
Head 6
Head 7
Head 8
Head 9
Head 10
Head 11
Head 12
Attention Head
2
3
4
5
6
Spectral Norm
4.34
2.68
1.90
2.67
2.04
2.66
4.12
6.27
5.32
1.74
4.64
5.39
Spectral Norm of WQ per Head in Layer 1
Figure 3: WQ of attention head 8 has
the largest spectral norm. Subsequent
analysis focuses on this head.
Failure is Linked to the Computation of δ.
The
backward pass of flash attention uses the term δ
=
rowsum(dO◦O) ∈RN for computational efficiency. An
alternative, mathematically equivalent formulation com-
putes this term as δ = rowsum(dP ◦P), where dP =
dOV⊤.
We find that replacing the efficient computa-
tion with this alternative formulation restored training sta-
bility. This experiment demonstrates that numerical er-
rors introduced when computing O in BF16 is likely the
primary source of failure, as the alternative formulation
which avoids training failure is equivalent to use O com-
puted in FP32.
Numerical Errors in O are the Source of Failure.
Building on the finding that the computation of δ is critical, we further isolate the source of error to
the output matrix Olp in low-precision δlp = rowsum(dO◦Olp). We conduct two key experiments.
First, instead of using the low-precision O from the forward pass to compute δ, we recompute it as
PV in FP32 within the backward pass; this change stabilizes training. Second, we find that com-
puting O in high precision (FP32) during the forward pass, i.e., δhp = rowsum(dO ◦Ohp), while
4

Published as a conference paper at ICLR 2026
keeping all other operations in BF16, also restored stability. This evidence conclusively demon-
strates that numerical errors introduced during the BF16 computation of O are the direct cause of
the failure (refer to Claim 1).
Failure is Localized to Specific Attention Heads.
To further narrow down the source of failure,
we analyze individual attention heads by tracking the spectral norm (Yang et al., 2023; Rybakov
et al., 2024) of their query projection matrices (WQ), as shown in Fig. 3. This reveals that a few
heads exhibit disproportionately large spectral norms. We confirm their role by selectively comput-
ing the output O in high precision for these outlier heads (1, 7, 8, 9, 11, and 12), which is sufficient
to restore training stability. Since head 8 shows the largest spectral norm, we focus our subsequent
analysis on this head to dissect the precise failure mechanism.
Claim 1. Low-precision δlp = rowsum(dO ◦Olp) causes training failure.
3.3
THE ROOT CAUSES OF TRAINING FAILURE
Our investigation in this section uncovers the two interconnected root causes of the training failure.
In Section 3.3.1, we demonstrate how low-precision δlp drives the training failure by a biased weight
update, and find that bias arises from the emergence of similar low-rank representations R, whose
coefficients (δlp −δhp)[T] are biased towards positive values, causing the error to accumulate rather
than cancel out. In Section 3.3.2, we trace the origin of these positive coefficients (δlp −δhp)[T] to
biased rounding errors inherent in the BF16 addition within the ¯PV product.
3.3.1
CAUSE 1. SIMILAR LOW-RANK MATRICES BIAS WEIGHT UPDATES
This section traces the training failure to biased weight update. We first analyze the difference
between the high and low precision gradients calculated with δhp and δlp, respectively. Then we
find that the low-rank representations and biased (δlp −δhp)[T] lead to loss explosion.
Analysis of the Error between High and Low Precision Gradients
To understand how nu-
merical errors propagate into the gradients, we analyze the difference between the high-precision
(hp) and low-precision (lp) gradients for the query matrix, dQ. The gradient dQ is computed
from the gradient of the attention scores, dS, as dQ = dSK. The score gradient is given by
dS = αP ◦(dP −δ), where δ = rowsum(dO ◦O) is the only term that differs between the high-
and low-precision backward passes based on Section 3.2, and α is a scaling factor in attention.
The difference between the high-precision and low-precision query gradients can be derived as:
dQhp −dQlp = (dShp −dSlp)K
= (αP ◦(dP −δhp) −αP ◦(dP −δlp)) K = (αP ◦(δlp −δhp)) K
=α · diag(δlp −δhp)(PK).
(1)
In the final step, we express the row-wise scaling operation P ◦(δlp −δhp) (follow broadcasting
rule) as a matrix multiplication, where diag(δlp −δhp) is a diagonal matrix whose diagonal entries
are the elements of the vector difference δlp −δhp. This formulation reveals that the gradient error
is directly proportional to the error in δ and is modulated by the term PK.
The gradient of the query projection matrix, dWQ, is given by the outer product of the input features
X and the query gradient dQ. The difference between the high-precision (hp) and low-precision (lp)
gradients for WQ can be expressed as:
dWQ
hp −dWQ
lp = (dQhp −dQlp)⊤X = α(PK)⊤diag(δlp −δhp)X,
= α
XN
T =1(δlp −δhp)[T] · (PK)[T]⊤X[T],
(2)
where (PK)[T] and X[T] are the T-th row vectors of their matrices. This equation shows that the
total gradient error is a weighted sum of rank-1 matrices, with weights given by the error in δ.
Similar Low-rank Updates of Weight Cause Training Failure
In Fig. 4, the rows of PK (panels
a, d) and X (panels b, e) exhibit strong structural similarity across different training steps and token
5

Published as a conference paper at ICLR 2026
0 2 4 6 8 101214161820222426283032343638404244464850525456586062
Features
0
27
54
81
108
135
162
189
216
243
270
297
324
351
378
405
432
459
486
513
540
567
594
621
648
675
702
729
756
783
810
837
864
891
918
945
972
999
Tokens
PK (Head 8, Batch Index 190, Training 6610)
20
10
0
10
20
(a)
0
18
36
54
72
90
108
126
144
162
180
198
216
234
252
270
288
306
324
342
360
378
396
414
432
450
468
486
504
522
540
558
576
594
612
630
648
666
684
702
720
738
756
Features
0
27
54
81
108
135
162
189
216
243
270
297
324
351
378
405
432
459
486
513
540
567
594
621
648
675
702
729
756
783
810
837
864
891
918
945
972
999
Tokens
X (Batch Index 190, Training 6610)
10
5
0
5
10
15
20
25
(b)
0
18
36
54
72
90
108
126
144
162
180
198
216
234
252
270
288
306
324
342
360
378
396
414
432
450
468
486
504
522
540
558
576
594
612
630
648
666
684
702
720
738
756
Input Features
0
2
4
6
8
10
12
14
16
18
20
22
24
26
28
30
32
34
36
38
40
42
44
46
48
50
52
54
56
58
60
62
Output Features
(PK[50])
X[50] (Head 8, Batch Index 190, Training 6610)
300
200
100
0
100
200
300
(c)
0 2 4 6 8 101214161820222426283032343638404244464850525456586062
Features
0
27
54
81
108
135
162
189
216
243
270
297
324
351
378
405
432
459
486
513
540
567
594
621
648
675
702
729
756
783
810
837
864
891
918
945
972
999
Tokens
PK (Head 8, Batch Index 209, Training 6619)
20
10
0
10
20
(d)
0
18
36
54
72
90
108
126
144
162
180
198
216
234
252
270
288
306
324
342
360
378
396
414
432
450
468
486
504
522
540
558
576
594
612
630
648
666
684
702
720
738
756
Features
0
27
54
81
108
135
162
189
216
243
270
297
324
351
378
405
432
459
486
513
540
567
594
621
648
675
702
729
756
783
810
837
864
891
918
945
972
999
Tokens
X (Batch Index 209, Training 6619)
10
5
0
5
10
15
20
25
(e)
0
18
36
54
72
90
108
126
144
162
180
198
216
234
252
270
288
306
324
342
360
378
396
414
432
450
468
486
504
522
540
558
576
594
612
630
648
666
684
702
720
738
756
Input Features
0
2
4
6
8
10
12
14
16
18
20
22
24
26
28
30
32
34
36
38
40
42
44
46
48
50
52
54
56
58
60
62
Output Features
(PK[718])
X[718] (Head 8, Batch Index 209, Training 6619)
200
100
0
100
200
(f)
Figure 4: PK, X, and (PK)[T]⊤X[T] at different batch indices and training steps. (c) and (f) show
that (PK)[T]⊤X[T] for different tokens and training steps have some similar columns in input
features 546 and 678.
positions. This implies that the resulting rank-1 matrices, (PK)[T]⊤X[T], are also highly similar
to one another. For instance, Fig. 4 (panels c, f) shows this similarity for tokens 50 and 718 at
training steps 6610 and 6619, respectively. Because these rank-1 error components are structurally
consistent, we can approximate the total gradient difference as
dWQ
hp −dWQ
lp ≈α
XN
T =1(δlp −δhp)[T]R
(3)
where R denotes the common low-rank structure emerging across different tokens and training steps.
Eqn.(3) shows that the accumulation of the low-rank error direction R is governed by the scalar term
PN
T =1(δlp−δhp)[T]. If this sum is biased towards non-zero values, the error across different training
steps will accumulate rather than cancel out. We track the cumulative sum of PN
T =1(δlp −δhp)[T]
over a sequence of training steps (6580 to 6680) leading up to the failure, as shown in Fig. 5(a). The
plot reveals that this sum is consistently positive, indicating a systematic bias. This bias causes the
error in the low-rank direction R to compound with each training step. Because R is also similar
for different steps, this ultimately corrupts the weight updates, increases the spectral norm (Fig. 11)
and activations (Yang et al., 2023; Rybakov et al., 2024), and causes the training failure (refer to
Claim 2). The following section finds the root cause of this positive bias by analyzing the weights
and gradients at training step 6619, a point of significant positive contribution identified in Fig. 5(a).
Claim 2. Weight update in low-precision training is biased by (δlp −δhp)[T]R, which
arises from the structurally similar matrices (denoted as R) across tokens and training
steps, and its positively-biased coefficient (δlp −δhp)[T]. This bias accumulates error,
preventing cancellation, increasing weight spectral norm and activation, and leading to
loss explosion.
3.3.2
CAUSE 2. BIASED ROUNDING ERROR LEADS TO POSITIVE (δlp −δhp)[T]
This section investigates the origin of the positive bias in (δlp −δhp)[T]. We trace this error to the
interaction between dO and the numerical discrepancy in Olp−Ohp, which itself arises from biased
rounding errors in BF16 addition during the ¯PV computation.
Locate the Large Error in (δlp −δhp)[T]
We first investigate the source of the positive bias in
PN
T =1(δlp −δhp)[T]. As analyzed in Section 3.2, the error in δ stems from the product of the
6

Published as a conference paper at ICLR 2026
6580
6600
6620
6640
6660
6680
Iteration
0.00000
0.00005
0.00010
0.00015
0.00020
0.00025
0.00030
0.00035
0.00040
Cumulative Sum of 
lp
hp
Cumulative Sum of 
lp
hp Over Iterations
Cumulative Sum
(a) Positively-biased (δlp−δhp)[T]
0
10
20
30
40
50
60
Features
2.5
2.0
1.5
1.0
0.5
0.0
0.5
1.0
1.5
O[T, : ]
1.5
1.0
0.5
0.0
0.5
1.0
1.5
dO[T, : ]
1e 6
O[T, : ]
dO[T, : ]
Visualization of O[T, : ] and dO[T, : ]
(b) Large values in features 20 &
29
0
10
20
30
40
50
60
Features
1.4
1.2
1.0
0.8
0.6
0.4
0.2
0.0
O[T, : ]lp
O[T, : ]hp
1e 2
1.5
1.0
0.5
0.0
0.5
1.0
1.5
dO[T, : ]
1e 6
O[T, : ]lp
O[T, : ]hp
dO[T, : ]
Visualization of O[T, : ]lp
O[T, : ]hp and dO[T, : ]
(c) Large error in features 20 & 29
Figure 5: Analysis of δ = rowsum(dO ◦O).
0
200
400
600
800
1000
Tokens
0.0
0.2
0.4
0.6
0.8
1.0
P
6
5
4
3
2
1
0
1
2
V
P
V
Visualization of P[T, : ] and V[ : , i]
(a) Most V[:, i] are negative
0
200
400
600
800
1000
Tokens t
0.0
0.2
0.4
0.6
0.8
1.0
P
0.014
0.012
0.010
0.008
0.006
0.004
0.002
0.000
Oerror(t)
P
Oerror(t)
Visualization of P[T, : ] and Oerror(t)
(b) Large error when ¯P[T, i] = 1
630
640
650
660
670
680
Tokens t
0.0
0.2
0.4
0.6
0.8
1.0
P
Detail View of P[T, : ] and Oerror(t)
0.014
0.012
0.010
0.008
0.006
0.004
0.002
0.000
Oerror(t)
P
Oerror(t)
(c) Detailed view of (b)
Figure 6: Analysis of ¯PV
upstream gradient dO and the numerical error in the low-precision output, Olp −Ohp. To dissect
this, we focus on a token position T = 718 where the error component (δlp −δhp)[T] is positive.
In Fig. 5(b) and (c), we observe a strong sign correlation between the gradient dO[T, :] and the
output error Olp[T, :] −Ohp[T, :] for specific feature dimensions such as 20 and 29 (also observed
across other tokens). In these dimensions, both dO and the output error Olp −Ohp are consistently
negative. This alignment ensures their product, which contributes to the error in δ, is positive. The
fact that the output error tends to be negative (Olp[T, i] < Ohp[T, i]) indicates that the low-precision
computation of O is systematically biased towards more negative values. Our subsequent analysis,
therefore, focuses on identifying the origin of this computational bias.
The output O is computed from an intermediate unnormalized output, ¯O. The computation involves
a safe softmax followed by a matrix multiplication and normalization:
¯P = exp(S −rowmax(S)),
¯O = ¯PV,
O = ¯O/rowsum(¯P).
Further experiments pinpoint the source of failure to the computation of the unnormalized output,
¯O = ¯PV. We find that computing only this product in FP32 is sufficient to stabilize training. To
understand the origin of this bias, we examine the difference between the low-precision and high-
precision computation of a single element, ¯O[T, i] (for feature index i = 20 in our analysis):
¯Olp[T, i] −¯Ohp[T, i] = (¯Plp[T, :]V[:, i])lp −(¯Php[T, :]V[:, i])hp
(4)
where the inputs ¯P and V are themselves the results of prior BF16 operations. Specifically, the
subscript (·)lp here computes dot product in FP32 with the final result rounded to BF16, while (·)hp
computes entirely in FP32.
To understand how the error ¯Olp[T, i] −¯Ohp[T, i] becomes systematically negative, we plot the
cumulative error as the sum over token positions progresses in Fig. 6(b) and (c):
¯Oerror(t) =
Xt
t′=1
¯P[T, t′]V[t′, i]

lp −
Xt
t′=1
¯P[T, t′]V[t′, i]

hp .
(5)
The figure shows that the error accumulates in significant negative steps. These steps occur at token
positions t where the corresponding attention probability ¯P[T, t] is exactly 1 (also observed in other
token positions). This happens when the pre-softmax score S[T, t] is the maximum value in its row,
causing exp(S[T, t] −max(S[T, :])) to evaluate to exp(0) = 1.
7

Published as a conference paper at ICLR 2026
Analysis of Biased Rounding Error
Furthermore, the bias arises from the interaction of these
unit values with the distribution of the value matrix V. As observed in Fig. 6(a), for the problematic
feature dimension i = 20, the values of V[:, i] are predominantly negative. When ¯P[T, t] = 1, the
product ¯P[T, t]V[t, i] is simply V[t, i], a negative BF16 number. A systematic error then occurs
when two such negative BF16 numbers are added. In floating-point arithmetic, adding two numbers
with the same sign can cause the resulting significand to overflow (e.g., −1.xxxx + −1.yyyy =
−10.zzzz), requiring a right shift and an exponent increment to re-normalize. The bits shifted out
of the 7-bit BF16 fraction determine the rounding direction. When adding two negative numbers,
the rounding operation (e.g., round-to-nearest) can introduce a consistent bias.
To illustrate how this rounding bias occurs, consider the addition of two significands that cause
an overflow, requiring a right shift for normalization. The bit that is shifted out (the rounding bit)
determines the rounding direction. We show all possible additions of the last two 2-bit numbers,
where the green bit represents the bit that would be shifted out (the rounding bit):
00 + 00 = 00
00 + 01 = 01
00 + 10 = 10
00 + 11 = 11
01 + 01 = 10
01 + 10 = 11
01 + 11 = 100
10 + 10 = 100
10 + 11 = 101
11 + 11 = 110
Because the summation is performed in FP32, the accumulation of lower-order bits from the small
numbers can activate the sticky bit. Consequently, this forces a rounding up when the subsequent
BF16 number is added. Therefore, the rounding bit 1 indicates that rounding up is required. Because
the operands ¯P[T, t]V[t, i] are negative and have a large exponent (because ¯P[T, t] = 1 does not
make ¯P[T, t]V[t, i] smaller), the error of rounding up is magnified, resulting a negative error. When
the rounding bit is 0, the result is rounded down, introducing a positive error. Furthermore, the
positive error is smaller compared to the negative error from rounding up, as the values being added
are usually very small. This asymmetry leads to the rounding error dominated by the rounding up,
resulting in negative rounding error as observed in our analysis. This systematic negative bias in the
computation of ¯O is the ultimate source of the training failure.
Remark 1. When ¯P[T, t] < 1 is less than 1 and has a mix of zero and non-zero bits in its signifi-
cand’s bits, its product with V[t, i] will also have a non-zero value in those bits. When rounding to
BF16, this will not introduce biased rounding error.
Analysis of Rounding Error in Fig. 6
To make this concrete, we now analyze the specific BF16
number addition that causes the large negative error jump shown in Fig. 6(c). Because the sum-
mation is performed in FP32, the first BF16 value is added by some small values from other to-
kens before the second BF16 value is added. This can activate the sticky bit, which can force
a round-up when adding the second BF16 value. For this example, we start with the FP32 rep-
resentations. The first operand is the accumulated sum of previous terms, which includes a BF16
(11000000000110100000000000000000; −2.40625) plus small residual (∼0.00087) that activate
the sticky bit. The second operand is another BF16 value. Their FP32 representations are:
11000000000110100000111000101110 (−2.4071154594421387)
11000000000100110000000000000000 (−2.296875)
Since their exponents are identical, the addition is performed on their significands:
(−1.00110100000111000101110) + (−1.00100110000000000000000)
= −10.01011010000111000101110
The result overflows the significand’s format, requiring normalization. The significand is shifted
right by one bit, and the exponent is incremented:
Significand: 10.0101101 →1.00101101
Exponent: 10000000 →10000001
The exact result in FP32 is 1 10000001 00101101000011100010111, which corresponds to
−4.703990459442139. To store this result in BF16, it must be rounded to 7 fraction bits. The
significand is 1.00101101. The 7-bit fraction is 0010110. Because the rounding bit is 1 and there
are nonzero bits after the rounding bit, the round-to-nearest (ties-to-even) rule rounds up (adding 1
to the last bit of the fraction):
8

Published as a conference paper at ICLR 2026
1k
10k
100k
600k
Training Steps (log scale)
10
1
3 × 10
0
4 × 10
0
6 × 10
0
Validation Loss (log scale)
Compare classical FA and SFA (AdamW Optimizer)
Stabilized Flash Attention
Flash Attention
(a) GPT-2S+AdamW
1k
10k
100k
600k
Training Steps (log scale)
10
1
3 × 10
0
4 × 10
0
6 × 10
0
Validation Loss (log scale)
Compare classical FA and SFA (Muon Optimizer)
Stabilized Flash Attention
Flash Attention
(b) GPT-2S+Muon
1k
5k
10k
Training Steps (log scale)
10
1
3 × 10
0
4 × 10
0
6 × 10
0
Validation Loss (log scale)
Compare classical FA and SFA (AdamW Optimizer)
Stabilized Flash Attention
Flash Attention
(c) GPT-2M+AdamW
Figure 7: Validation losses for classical flash attention and the stablized flash attention with AdamW,
Muon optimizers, and GPT-2M model.
BF16 Fraction: 0010110 + 1 = 0010111
The final BF16 result is 1100000010010111, which represents −4.71875. This rounded value
is more negative than the true sum of −4.703990459442139. The error introduced by this single
addition is −0.014759540557861328. When such rounding events occur systematically across
many additions in the ¯PV product, the errors accumulate, creating the negative bias in O that
ultimately destabilizes training.
Claim 3. With more than one ¯P[T, t] = 1 and negative V[t, i], the addition in ¯PV can
cause overflow of significand. This necessitates a right shift and a round-up of significand
due to sticky bit, introducing negative rounding error in O and leading to positive (δlp −
δhp)[T].
4
EXPERIMENT: MITIGATE BIAS IN ROUNDING ERROR
Our analysis traces the failure to biased rounding errors in the ¯PV computation. This occurs when
multiple identical maxima in a row of the pre-softmax scores S cause corresponding elements in ¯P
to become exactly 1. To validate our findings, we modify the softmax to detect this specific condition
and adjust the normalization, ensuring all elements of ¯P are strictly less than 1. This prevents the
biased rounding and restores training stability.
rm = rowmax(S), rs = rowsum(rm −S ≤ϵ)
m′ = where(rm > 0 ∧rs > 1, βrm, rm), β > 1
m = where(rm < 0 ∧rs > 1, 0, m′)
¯P = exp(S −m)
To prevent biased rounding, we introduce a tar-
geted modification to the safe softmax compu-
tation. The core idea is to dynamically adjust
the normalization factor m only when a row of
the score matrix S contains multiple identical
maximum values. This adjustment ensures that
the argument to the exponential function, S −m, becomes strictly negative at these maximal po-
sitions, which in turn guarantees that all elements of ¯P = exp(S −m) are less than 1. A naive
approach, such as subtracting a small fixed constant, is insufficient as it introduces new systematic
rounding errors (see Appendix C); hence, a dynamic maximum strategy is required. Our modifica-
tion is presented above.
This modification prevents elements of ¯P from becoming exactly 1. We first check whether the row
maximum rm is attained by more than one entry by evaluating rowsum(rm −S ≤ϵ), where ϵ is a
small numerical tolerance (e.g., 10−3) to account for floating-point precision. If repeated maxima
are detected and rm > 0, the normalization factor is adjusted to m = βrm with β > 1. The
new maximum in the exponent then becomes −(β −1)rm, which is strictly negative. If instead
rm < 0 and is repeated, we set m = 0, which likewise ensures that the maximum exponent remains
negative. In both cases, the adjustment guarantees that max(S −m) < 0 and thus max(¯P) < 1,
preventing the conditions that lead to biased rounding.
Crucially, this modification is mathematically equivalent to standard attention in exact arithmetic, as
it leverages the shift-invariance property of the softmax function (softmax(z) = softmax(z −c)).
9

Published as a conference paper at ICLR 2026
Our method simply chooses a different row-wise constant c to ensure numerical stability. In our
experiments, we set β ∈[2, 8], as smaller values risk having the result round back to 1, while
larger values risk underflow. This modification is integrated into the standard flash attention tiling
algorithm (line with magenta in Algorithm 3) without altering the backward pass.
To validate our approach, we pre-train the GPT-2S model for 600K steps in BF16 precision using
our stabilized flash attention (β = 2) with both AdamW and Muon optimizers (Jordan et al., 2024).
The AdamW configuration uses the same hyperparameters as described in Section 3.1. For the
Muon setup, we employ RMS-matched Muon (Liu et al., 2025) with a learning rate of 5 × 10−4 and
weight decay of 0.01, while retaining AdamW for embeddings, the language model head, and other
one-dimensional parameters. We also extend our experiments to the larger GPT-2M model, training
it for 100K steps with the AdamW optimizer under the same hyperparameters as in Section 3.1. As
shown in Fig. 7, the modified flash attention successfully stabilizes training for both optimizers and
larger model, preventing the loss explosion seen in the original implementation.
5
CONCLUSION
This paper presents the first mechanistic explanation for a notorious loss explosion in low-precision
flash attention training. We pinpoint the root cause to an interplay between emergent low-rank rep-
resentations and biased BF16 rounding errors. A minimal, targeted modification to flash attention
validates our analysis by restoring stability. Our analytical workflow provides a blueprint for di-
agnosing similar numerical instabilities in other architectures, scales, and low-precision formats,
paving the way for more robust and efficient large-scale model training.
Discussion
Our findings are consistent across various hardware (NVIDIA A100, RTX 4090,
Huawei Ascend 910B) and mechanistically explain empirical observations of training instability.
The growth of weight spectral norms results from the accumulation of a low-rank error matrix in
the gradients. We also clarify the role of attention sinks: by attracting high attention scores, they
are more likely to produce attention probabilities of 1, which triggers the biased rounding error in
the ¯PV computation. This provides a direct numerical link between the architectural behavior of
sinks and the arithmetic instability that derails training. Finally, our analysis provides a compelling
explanation for the effectiveness of established stabilization techniques. We posit that a root cause of
training failure is the structural similarity within the error matrices, (PK)[T]⊤X[T], which creates
a pathway for rounding errors to accumulate systematically. Techniques like QK normalization and
Gated Attention disrupt this underlying structure. In doing so, they ensure that even when rounding
errors are present, they lack the coherence to compound, thereby averting the instabilities observed
during training.
Limitations
Our analysis focuses on a specific failure case in a GPT-2 model. The generaliz-
ability of our findings to other architectures, larger scales, or different low-precision formats like
FP8 requires further investigation. Additionally, our proposed mitigation is tailored to the specific
rounding error identified and may not address other sources of numerical instability.
Future Work
Our work provides a foundation for exploring numerical stability in low-precision
training. Our analytical framework, i.e., locating the source of error, finding similar update direc-
tions, and tracing back to root causes, can be applied to other architectures and precision formats.
Future work could extend this analysis to FP8 training, larger models, and different architectures.
Additionally, developing automated tools to detect and mitigate such numerical instabilities during
training would be valuable for the community.
ACKNOWLEDGMENT
This work is supported by Beijing Science and Technology Program (No.Z251100008125003).
REFERENCES
Sami Ben Ali, Silviu-Ioan Filip, and Olivier Sentieys. A stochastic rounding-enabled low-precision
floating-point mac for dnn training. In 2024 Design, Automation & Test in Europe Conference &
10

Published as a conference paper at ICLR 2026
Exhibition (DATE), pp. 1–6. IEEE, 2024.
Paul Balanc¸a, Sam Hosegood, Carlo Luschi, and Andrew Fitzgibbon. Scalify: scale propagation for
efficient low-precision llm training. arXiv preprint arXiv:2407.17353, 2024.
Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Y Prince, Bj¨orn Deiseroth,
Andres Felipe Cruz-Salinas, Carlo Luschi, Samuel Weinbach, and Douglas Orr. u-µp: The unit-
scaled maximal update parametrization. arXiv preprint arXiv:2407.17465, 2024.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam
Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm:
Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):
1–113, 2023.
Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In Inter-
national Conference on Learning Representations (ICLR), 2024.
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and
memory-efficient exact attention with IO-awareness. In Advances in Neural Information Process-
ing Systems (NeurIPS), 2022.
Maxim Fishman, Brian Chmiel, Ron Banner, and Daniel Soudry. Scaling fp8 training to trillion-
token llms. arXiv preprint arXiv:2409.12517, 2024.
flash-attention Issue 337.
Discussion on flash-attention issue #337: pretraining loss blows up
with triton flash attention.
https://github.com/Dao-AILab/flash-attention/
issues/337, 2024. Accessed: 2025-09-07.
Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus. http:
//Skylion007.github.io/OpenWebTextCorpus, 2019.
Alicia Golden, Samuel Hsia, Fei Sun, Bilge Acun, Basil Hosmer, Yejin Lee, Zachary DeVito,
Jeff Johnson, Gu-Yeon Wei, David Brooks, et al.
Is flash attention stable?
arXiv preprint
arXiv:2405.02803, 2024.
Xin He, Jianhua Sun, Hao Chen, and Dong Li. Campo:{Cost-Aware} performance optimization
for {Mixed-Precision} neural network training. In 2022 USENIX Annual Technical Conference
(USENIX ATC 22), pp. 505–518, 2022.
Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization
for transformers. arXiv preprint arXiv:2010.04245, 2020.
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza
Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Train-
ing compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.
Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang Wang, and Shiwei Liu. Spam: Spike-
aware adam with momentum reset for stable llm training. arXiv preprint arXiv:2501.06842, 2025.
Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy
Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https:
//kellerjordan.github.io/posts/muon/.
Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee,
Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen,
et al. A study of bfloat16 for deep learning training. arXiv preprint arXiv:1905.12322, 2019.
Kimi-Team. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.
Joonhyung Lee, Jeongin Bae, Byeongwook Kim, Se Jung Kwon, and Dongsoo Lee. To fp8 and
back again: Quantifying the effects of reducing precision on llm training stability. CoRR, 2024.
11

Published as a conference paper at ICLR 2026
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao,
Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint
arXiv:2412.19437, 2024.
Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin,
Weixin Xu, Enzhe Lu, Junjie Yan, et al.
Muon is scalable for llm training.
arXiv preprint
arXiv:2502.16982, 2025.
Naveen Mellempudi, Sudarshan Srinivasan, Dipankar Das, and Bharat Kaul. Mixed precision train-
ing with 8-bit floating point. arXiv preprint arXiv:1905.12334, 2019.
Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia,
Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. Mixed precision
training. arXiv preprint arXiv:1710.03740, 2017.
Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisen-
thwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, et al. Fp8 formats for
deep learning. arXiv preprint arXiv:2209.05433, 2022.
Igor Molybog, Peter Albert, Moya Chen, Zachary DeVito, David Esiobu, Naman Goyal, Punit Singh
Koura, Sharan Narang, Andrew Poulton, Ruan Silva, et al. A theory on adam instability in large-
scale machine learning. arXiv preprint arXiv:2304.09871, 2023.
nanoGPT Issue 303. Discussion on nanogpt issue #303: Gradient explosion when training with
bfloat16. https://github.com/karpathy/nanoGPT/issues/303, 2023. Accessed:
2025-09-07.
nanoGPT Issue 524. Discussion on nanogpt issue #524: Training loss becomes nan when using
bfloat16. https://github.com/karpathy/nanoGPT/issues/524, 2024. Accessed:
2025-09-07.
nanoGPT Issue 554. Discussion on nanogpt issue #554: Loss diverges when using bfloat16 with
flash attention. https://github.com/karpathy/nanoGPT/issues/554, 2024. Ac-
cessed: 2025-09-07.
Badreddine Noune, Philip Jones, Daniel Justus, Dominic Masters, and Carlo Luschi. 8-bit numerical
formats for deep neural networks. arXiv preprint arXiv:2206.02915, 2022.
Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue
Yang, Bolin Ni, Jingcheng Hu, et al. Fp8-lm: Training fp8 large language models. arXiv preprint
arXiv:2310.18313, 2023.
Sergio P Perez, Yan Zhang, James Briggs, Charlie Blake, Josh Levy-Kramer, Paul Balanca, Carlo
Luschi, Stephen Barlow, and Andrew William Fitzgibbon. Training and inference of large lan-
guage models using 8-bit floating point. arXiv preprint arXiv:2309.17224, 2023.
Haiquan Qiu, You Wu, Yingjie Tan, Yaqing Wang, and Quanming Yao. Spectral alignment for early
detection of loss explosion in language models. arXiv preprint arXiv:2510.04202, 2025a.
Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei
Huang, Suozhi Huang, et al. Gated attention for large language models: Non-linearity, sparsity,
and attention-sink-free. arXiv preprint arXiv:2505.06708, 2025b.
Qwen-Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language
models are unsupervised multitask learners. 2019.
Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John
Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models:
Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.
Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations
toward training trillion parameter models. In SC20: International Conference for High Perfor-
mance Computing, Networking, Storage and Analysis, pp. 1–16. IEEE, 2020.
12

Published as a conference paper at ICLR 2026
Oleg Rybakov, Mike Chrzanowski, Peter Dykas, Jinze Xue, and Ben Lanir. Methods of improving
llm training stability. arXiv preprint arXiv:2410.16682, 2024.
Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao.
Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in
Neural Information Processing Systems, 37:68658–68685, 2024.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee
Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and
efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
Albert Tseng, Tao Yu, and Youngsuk Park.
Training llms with mxfp4.
arXiv preprint
arXiv:2502.20586, 2025.
Naigang Wang, Jungwook Choi, Daniel Brand, Chia-Yu Chen, and Kailash Gopalakrishnan. Train-
ing deep neural networks with 8-bit floating point numbers. Advances in neural information
processing systems, 31, 2018.
Ruizhe Wang, Yeyun Gong, Xiao Liu, Guoshuai Zhao, Ziyue Yang, Baining Guo, Zhengjun Zha,
and Peng Cheng. Optimizing large language model training using fp4 quantization. arXiv preprint
arXiv:2501.17116, 2025.
Shibo Wang and Pankaj Kanwar.
Bfloat16:
The secret to high performance on cloud
tpus.
https://cloud.google.com/blog/products/ai-machine-learning/
bfloat16-the-secret-to-high-performance-on-cloud-tpus, August 2019.
Accessed: 2025-09-07.
Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari S. Morcos, Ali Farhadi, and Lud-
wig Schmidt.
Stable and low-precision training for large-scale vision-language models.
In
Thirty-seventh Conference on Neural Information Processing Systems, 2023.
URL https:
//openreview.net/forum?id=sqqASmpA2R.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming
language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.
Greg Yang, Edward J Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ry-
der, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs v: Tuning large neural
networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022.
Greg Yang, James B Simon, and Jeremy Bernstein. A spectral condition for feature learning. arXiv
preprint arXiv:2310.17813, 2023.
Ruizhe Zhao, Brian Vogel, Tanvir Ahmed, and Wayne Luk. Reducing underflow in mixed precision
training by gradient scaling. In Proceedings of the Twenty-Ninth International Conference on
International Joint Conferences on Artificial Intelligence, pp. 2922–2928, 2021.
Jiecheng Zhou, Ding Tang, Rong Fu, Boni Hu, Haoran Xu, Yi Wang, Zhilin Pei, Zhongling Su,
Liang Liu, Xingcheng Zhang, et al. Towards efficient pre-training: Exploring fp4 precision in
large language models. arXiv preprint arXiv:2502.11458, 2025.
13

Published as a conference paper at ICLR 2026
A
RELATED WORK
A.1
MIXED-PRECISION BF16 TRAINING.
Contemporary large language model (LLM) pretraining almost universally employs mixed-precision
arithmetic. Early efforts by Micikevicius et al. (2017) demonstrated that FP16 training—using an
FP32 master copy of weights and fixed loss scaling—could match FP32 accuracy for many models.
However, the narrow exponent range of FP16 often causes many gradients to underflow, necessitat-
ing careful tuning. The bfloat16 (BF16) format, with an 8-bit exponent and 7-bit mantissa, retains
the wide dynamic range of FP32 while halving storage cost. Kalamkar et al. (2019) showed that
BF16 achieves convergence parity with FP32 on large models without specialized tuning. Since
then, BF16 has become the default 16-bit format for many large-scale training frameworks, with
native support in PyTorch and TensorFlow (Wang & Kanwar, 2019).
BF16 mixed precision has enabled training of landmark LLMs at unprecedented scales, including
GPT-3 (175B) (Brown et al., 2020), Google PaLM (540B) (Chowdhery et al., 2023), DeepMind
Gopher (280B) (Rae et al., 2021), Chinchilla (70B) (Hoffmann et al., 2022), and Meta’s LLaMA
family (7B-65B) (Touvron et al., 2023). To handle the massive memory footprint, parallel training
frameworks such as Megatron and DeepSpeed integrate BF16 training with techniques like the Zero
Redundancy Optimizer (ZeRO) (Rajbhandari et al., 2020).
Despite its advantages, BF16 training can still exhibit instabilities. Empirical studies show that FP16
is highly unstable even with loss scaling, whereas BF16 eliminates most precision-related tuning
and failures Wang & Kanwar (2019). However, Lee et al. (2024) report that roughly 10% of GPT-2
pretraining runs diverged under pure BF16, compared to 0% under TF32. This suggests that while
BF16 substantially improves stability, complementary stabilization techniques remain necessary at
scale.
A.2
STABILIZING LOW-PRECISION TRAINING
Gradient Scaling.
Early work by Micikevicius et al. (2017) introduced FP16 mixed-precision
training, where weights, activations, and gradients are stored in half-precision while maintaining
a master FP32 copy. They also proposed loss scaling to prevent FP16 underflows. Even with
loss scaling, some underflow can still occur in deep networks. To address this, Zhao et al. (2021)
introduced gradient scaling, which dynamically computes per-layer scaling factors to avoid both
underflow and overflow.
Ultra-Low-Precision (FP8/INT8) Training.
To further reduce cost, recent works explore FP8
or INT8 precision for training and inference. However, naive FP8 training tends to diverge. Lee
et al. (2024) note that the direct application of FP8 to LLM training is unstable without additional
stabilization techniques. To address this, Perez et al. (2023) propose dynamically adjusted per-tensor
scaling factors for FP8 matrix multiplications. Using this scheme, they successfully train GPT- and
LLaMA-style models up to 70B parameters entirely in FP8. Similarly, Peng et al. (2023) introduce
FP8-LM, a framework that progressively applies FP8 to gradients, optimizer states, and distributed
communication, achieving a 39% memory reduction and 75% speedup compared to BF16. Balanc¸a
et al. (2024) present SCALIFY, which propagates scale factors throughout the computation graph
to ensure stable FP8 operations without manual tuning. These approaches collectively demonstrate
that careful scaling management enables FP8 or INT8 training to match BF16 performance while
reducing memory and compute requirements.
Optimizer and Gradient Stabilization.
Optimizer algorithms play a critical role in training sta-
bility. Molybog et al. (2023) theoretically analyze Adam and show that catastrophic divergence often
arises when the update direction becomes uncorrelated with the true descent direction in large-scale
models. To address gradient instability, Huang et al. (2025) propose SPAM (Spike-Aware Adam
with Momentum Reset), which detects and mitigates rare but severe “gradient spikes” by resetting
momentum and applying spike-aware clipping. In parallel, Wortsman et al. (2023) investigate loss
spikes in vision-language models and show that AdamW often underestimates the second moment
before spikes occur. They propose a hybrid AdamW-AdaFactor optimizer that adaptively corrects
14

Published as a conference paper at ICLR 2026
second-moment underestimation, outperforming gradient clipping alone. These methods highlight
how optimizer modifications directly mitigate divergence in low-precision regimes.
Activation and Architectural Techniques.
The choice of activation functions and initialization
strategies also impacts stability. Fishman et al. (2024) observe that the SwiGLU activation amplifies
outliers during long FP8 training runs. They introduce Smooth-SwiGLU, a modified activation that
prevents outlier amplification, enabling stable trillion-token FP8 training. In the vision-language
domain, Wortsman et al. (2023) show that “layer-scale zero” initialization and carefully designed
low-precision linear layers (e.g., SwitchBack) further improve stability in int8 training.
A.3
FLASH ATTENTION
Flash Attention (Dao et al., 2022) is an I/O-aware exact attention algorithm designed to overcome
the memory bottleneck of standard self-attention, which scales quadratically with sequence length
N. Standard attention requires materializing the full N × N attention score matrix, leading to
O(N 2) memory usage that becomes prohibitive for long sequences. Flash Attention mitigates this
by employing a tiling and kernel fusion strategy. The query, key, and value matrices are partitioned
into smaller blocks, which are iteratively loaded from slow high-bandwidth memory (HBM) into
fast on-chip SRAM.
Within the SRAM, the core computation is performed. A key innovation is the use of an online
softmax algorithm. Instead of computing the full score matrix before normalization, Flash Atten-
tion maintains running statistics (the maximum score and the sum of exponentials) as it processes
blocks of keys and values. This allows it to compute the correctly normalized output block-by-
block without ever storing the entire intermediate attention matrix. By fusing the matrix multi-
plications and softmax operations into a single GPU kernel, it drastically reduces the number of
memory read/write operations to HBM, which is the primary performance bottleneck. This I/O-
aware design reduces the memory complexity from O(N 2) to O(N) and significantly accelerates
computation. Flash Attention 2 (Dao, 2024) further optimized this approach by improving work
partitioning among GPU thread blocks and warps and reducing non-matmul computation overhead,
achieving near-optimal performance and making it an indispensable component for training and in-
ferring long-context LLMs. In the pseudocode below, we absorb the standard attention scaling factor
α = 1/
√
d into Q for notational simplicity. See Algorithm 1 and Algorithm 2 for pseudocode of
Flash Attention 2’s forward and backward passes.
B
BF16 ADDITION
The bfloat16 (Brain Floating-Point) format is a 16-bit floating-point representation widely used in
deep learning for its balance between computational efficiency and numerical range. It consists of
1 sign bit, 8 exponent bits, and 7 fraction (or mantissa) bits. This structure gives bfloat16 the same
dynamic range as the 32-bit single-precision format (FP32) but with significantly less precision.
The addition of two bfloat16 numbers, say a and b, follows the standard procedure for floating-point
arithmetic:
1. Exponent Alignment: The exponents of the two numbers are compared. The number with the
smaller exponent has its significand (the combination of the implicit leading bit and the fraction)
shifted to the right until its exponent matches the larger one. Each right shift increases the ex-
ponent by one. Bits shifted past the available precision are lost, which is an initial source of
error.
2. Significand Addition: The aligned significands are added together. The sign of the result is
determined by the signs and magnitudes of the operands.
3. Normalization: The result is normalized to ensure it conforms to the 1.xxxx... × 2e format.
If the addition resulted in an overflow (e.g., 10.xxxx...), the significand is shifted right and the
exponent is incremented. If it resulted in cancellation (e.g., 0.00xx...), the significand is shifted
left and the exponent is decremented until the leading bit is 1.
4. Rounding: The resulting significand, which may have more than 7 fraction bits after normaliza-
tion, must be rounded. The standard mode is ”round to nearest, ties to even”. This means if the
15

Published as a conference paper at ICLR 2026
Algorithm 1 Flash Attention: Forward Pass
Require: Matrices Q, K, V ∈RN×d, block sizes Bc, Br.
1: Divide Q into Tr =
l
N
Br
m
blocks Q1, . . . , QTr of size Br × d each, and divide K, V in to
Tc =
l
N
Bc
m
blocks K1, . . . , KTc and V1, . . . , VTc, of size Bc × d each.
2: Divide the output O ∈RN×d into Tr blocks O1, . . . , OTr of size Br × d each, and divide the
logsumexp L into Tr blocks L1, . . . , LTr of size Br each.
3: for 1 ≤i ≤Tr do
4:
Initialize O(0)
i
= (0)Br×d ∈RBr×d, ℓ(0)
i
= (0)Br ∈RBr, m(0)
i
= (−∞)Br ∈RBr.
5:
for 1 ≤j ≤Tc do
6:
Compute S(j)
i
= QiKT
j ∈RBr×Bc.
7:
Compute m(j)
i
= max(m(j−1)
i
, rowmax(S(j)
i )) ∈RBr, ¯P(j)
i
= exp(S(j)
i
−m(j)
i ) ∈
RBr×Bc (pointwise), ℓ(j)
i
= em(j−1)
i
−m(j)
i ℓ(j−1)
i
+ rowsum(¯P(j)
i ) ∈RBr.
8:
Compute O(j)
i
= diag(em(j−1)
i
−m(j)
i )O(j−1)
i
+ ¯P(j)
i Vj.
9:
end for
10:
Compute Oi = diag(ℓ(Tc)
i
)−1O(Tc)
i
.
11:
Compute Li = m(Tc)
i
+ log(ℓ(Tc)
i
).
12:
Write Oi as the i-th block of O.
13:
Write Li as the i-th block of L.
14: end for
15: Return the output O and the logsumexp L.
Algorithm 2 Flash Attention: Backward Pass
Require: Matrices Q, K, V, O, dO ∈RN×d, vector L ∈RN, block sizes Bc, Br.
1: Divide Q into Tr =
l
N
Br
m
blocks Q1, . . . , QTr of size Br × d each, and divide K, V in to
Tc =
l
N
Bc
m
blocks K1, . . . , KTc and V1, . . . , VTc, of size Bc × d each.
2: Divide O into Tr blocks O1, . . . , OTr of size Br × d each, divide dO into Tr blocks
dO1, . . . , dOTr of size Br × d each, and divide L into Tr blocks L1, . . . , LTr of size Br each.
3: Initialize dQ = (0)N×d and divide it into Tr blocks dQ1, . . . , dQTr of size Br ×d each. Divide
dK, dV ∈RN×d in to Tc blocks dK1, . . . , dKTc and dV1, . . . , dVTc, of size Bc × d each.
4: Compute δ = rowsum(dO ◦O) ∈RN (pointwise multiply), and divide it into Tr blocks
δ1, . . . , δTr of size Br each.
5: for 1 ≤j ≤Tc do
6:
Initialize dKj = (0)Bc×d, dVj = (0)Bc×d.
7:
for 1 ≤i ≤Tr do
8:
Compute S(j)
i
= QiKT
j ∈RBr×Bc.
9:
Compute P(j)
i
= exp(S(j)
i
−Li) ∈RBr×Bc.
10:
Compute dVj ←dVj + (P(j)
i )⊤dOi ∈RBc×d.
11:
Compute dP(j)
i
= dOiV⊤
j ∈RBr×Bc.
12:
Compute dS(j)
i
= P(j)
i
◦(dP(j)
i
−δi) ∈RBr×Bc.
13:
Update dQi ←dQi + dS(j)
i Kj ∈RBr×d.
14:
Compute dKj ←dKj + dS(j)
i
⊤Qi ∈RBc×d.
15:
end for
16: end for
17: Return dQ, dK, dV.
truncated part is greater than half the value of the last storable bit (LSB), the number is rounded
up. If it is less, it is rounded down. If it is exactly half, it is rounded to the nearest value with an
even LSB.
16

Published as a conference paper at ICLR 2026
The key source of numerical error comes from steps 1 and 4. During exponent alignment, precision
is lost from the smaller-magnitude number. After addition, the result must be rounded back to the
7-bit fraction, which introduces another rounding error. While “round to nearest, ties to even” is
designed to be unbiased for random data, a sequence of additions on data with a specific distribution
(e.g., mostly negative numbers being added together) can lead to a biased rounding error, where the
accumulated error consistently pushes the result in one direction. This accumulation of biased error
is a critical factor in the training failure observed in low-precision settings.
C
DESIGN CONSIDERATIONS FOR MITIGATING BIASED ROUNDING ERROR
IN FLASH ATTENTION
Use Dynamic Maximum Value Rather Than Fixed Offset
A fixed offset would cause the com-
puted values of ¯P to be consistently rounded in one direction during BF16 conversion, introducing
a fixed error. Since the elements of V often share the same sign, this fixed rounding error in ¯P does
not average out to zero when computing ¯PV. This leads to a biased error in the output O, which in
turn creates a biased term δ, reintroducing the very failure we aim to solve.
Dynamic Maximum is Applied Conditionally
Our modification is applied conditionally—only
when a row contains multiple identical maximum values—to avoid introducing new numerical insta-
bilities. An unconditional adjustment is not a better option. For example, if a row has a single, very
large positive maximum value rm, applying our rule would mean calculating exp(S −βrm). The
largest term in the exponent would become −(β −1)rm, and exp(−(β −1)rm) could underflow
to zero. This would cause the normalization factor to become zero, leading to a division-by-zero
error when computing the output O. By applying the modification only in the specific case that
causes biased rounding, we preserve the numerical stability of the standard online softmax in all
other scenarios.
Explanation on Dealing Negative Repeated Row Maximum
We also explore alternative stabi-
lization methods for the negative, repeated row maximum (rm < 0). One approach involves setting
the normalization factor m = γrm for some γ ∈(0, 1). This makes the new maximum value in the
exponent (1 −γ)rm. However, we observe that if γ is close to 1, this new maximum approaches
zero. In low-precision arithmetic, exp((1 −γ)rm) can round to exactly 1, reintroducing the very
failure we aim to prevent. Consequently, we found that setting γ = 0 (i.e., m = 0) is a robust
choice, as it ensures the maximum value in the exponent remains sufficiently negative.
D
SIMILAR PATTERNS DISCOVERED IN LLAMA-3.1-8B
To see if the phenomena that leads to the failure case in GPT-2 also exists in other models, we
present (PK)[T]⊤X[T] and the attention score in Llama-3.1-8B which shows the similar patterns
we discovered in GPT-2. This section is not meant to say Llama 3.1-8B will definitely fail in BF16
training, but rather to show that the phenomena we discovered is not unique to GPT-2.
D.1
STRUCTURALLY SIMILAR (PK)[T]⊤X[T]
Given an input “The quick brown fox jumps over the lazy dog”, we visualize (PK)[T]⊤X[T] for
attention head 13 in layer 1 of Llama-3.1-8B. We visualize the representations and similar columns
in Fig. 8.
D.2
MULTIPLE MAXIMUM IN ¯P
We also observe multiple maximum values in the attention scores of Llama-3.1-8B. We visualize
some examples in Fig. 9. This experiment shows the attention sink (Xiao et al., 2023) behavior as
well, where the first token has high attention scores.
17

Published as a conference paper at ICLR 2026
0
67
134
201
268
335
402
469
536
603
670
737
804
871
938
1005
1072
1139
1206
1273
1340
1407
1474
1541
1608
1675
1742
1809
1876
1943
2010
2077
2144
2211
2278
2345
2412
2479
2546
2613
2680
2747
2814
2881
2948
3015
3082
3149
3216
3283
3350
3417
3484
3551
3618
3685
3752
3819
3886
3953
4020
4087
Hidden dimension (X[T])
0
3
6
9
12
15
18
21
24
27
30
33
36
39
42
45
48
51
54
57
60
63
66
69
72
75
78
81
84
87
90
93
96
99
102
105
108
111
114
117
120
123
126
Head dimension (PK^T[T])
Layer 1 Head 13 - Token 3: ' brown'
6
4
2
0
2
4
6
(a)
0
67
134
201
268
335
402
469
536
603
670
737
804
871
938
1005
1072
1139
1206
1273
1340
1407
1474
1541
1608
1675
1742
1809
1876
1943
2010
2077
2144
2211
2278
2345
2412
2479
2546
2613
2680
2747
2814
2881
2948
3015
3082
3149
3216
3283
3350
3417
3484
3551
3618
3685
3752
3819
3886
3953
4020
4087
Hidden dimension (X[T])
0
3
6
9
12
15
18
21
24
27
30
33
36
39
42
45
48
51
54
57
60
63
66
69
72
75
78
81
84
87
90
93
96
99
102
105
108
111
114
117
120
123
126
Head dimension (PK^T[T])
Layer 1 Head 13 - Token 4: ' fox'
4
2
0
2
4
(b)
0
20
40
60
80
100
120
Head Dimension
0.008
0.006
0.004
0.002
0.000
0.002
0.004
0.006
Value
Token 3 (' brown') - Column 2598
Token 3
0
20
40
60
80
100
120
Head Dimension
0.10
0.05
0.00
0.05
Value
Token 4 (' fox') - Column 2598
Token 4
Column 2598 (Cosine Similarity: 0.999994)
(c)
0
20
40
60
80
100
120
Head Dimension
0.006
0.004
0.002
0.000
0.002
0.004
0.006
0.008
0.010
Value
Token 3 (' brown') - Column 2050
Token 3
0
20
40
60
80
100
120
Head Dimension
0.10
0.05
0.00
0.05
0.10
Value
Token 4 (' fox') - Column 2050
Token 4
Column 2050 (Cosine Similarity: 0.999994)
(d)
0
20
40
60
80
100
120
Head Dimension
0.006
0.004
0.002
0.000
0.002
0.004
0.006
0.008
0.010
Value
Token 3 (' brown') - Column 2050
Token 3
0
20
40
60
80
100
120
Head Dimension
0.10
0.05
0.00
0.05
0.10
Value
Token 4 (' fox') - Column 2050
Token 4
Column 2050 (Cosine Similarity: 0.999994)
(e)
0
20
40
60
80
100
120
Head Dimension
0.100
0.075
0.050
0.025
0.000
0.025
0.050
0.075
Value
Token 3 (' brown') - Column 3899
Token 3
0
20
40
60
80
100
120
Head Dimension
0.15
0.10
0.05
0.00
0.05
0.10
Value
Token 4 (' fox') - Column 3899
Token 4
Column 3899 (Cosine Similarity: 0.999993)
(f)
Figure 8: Similar (PK)[T]⊤X[T] discovered at different tokens in Llama-3.1-8B. (c)-(f) show the
similar columns.
E
OTHER DETAILS
E.1
RELATION BETWEEN MULTIPLE MAXIMA AND LOSS
To investigate the connection between the multiple maxima phenomenon and training instability,
we plot the frequency of multiple maxima occurrences against the training loss in Fig. 13. The
visualization reveals a clear temporal pattern: the number of multiple maxima begins to increase
before the loss explodes (see 7000 step). This strong correlation suggests that the emergence of
multiple maxima is a leading indicator of the impending training failure.
E.2
DERIVING EQUATION 1
The following equations are prepared to derive equation 1 in the main text.
dQ = dSK
(6)
dS = αP ◦(dP −δ)
(7)
Here is the detailed derivation.
We begin with the goal of deriving an expression for the difference dQhp −dQlp.
1. Apply the definition of dQ. We start with the left-hand side of the equation and substitute the
definition from equation 6 for both the high-precision (hp) and low-precision (lp) terms.
dQhp −dQlp = (dShpK) −(dSlpK)
Using the right distributive property of matrix multiplication, we can factor out the matrix K.
dQhp −dQlp = (dShp −dSlp) K
(8)
18

Published as a conference paper at ICLR 2026
0
20
40
60
80
100
120
140
Key Position
0.0
0.2
0.4
0.6
0.8
1.0
Attention Probability
layer=1, head=6, query=44
(a)
0
20
40
60
80
100
120
140
Key Position
0.0
0.2
0.4
0.6
0.8
1.0
Attention Probability
layer=16, head=27, query=11
(b)
0
20
40
60
80
100
120
140
Key Position
0.0
0.2
0.4
0.6
0.8
1.0
Attention Probability
layer=26, head=1, query=45
(c)
0
20
40
60
80
100
120
140
Key Position
0.0
0.2
0.4
0.6
0.8
1.0
Attention Probability
layer=26, head=1, query=82
(d)
0
20
40
60
80
100
120
140
Key Position
0.0
0.2
0.4
0.6
0.8
1.0
Attention Probability
layer=29, head=10, query=96
(e)
0
20
40
60
80
100
120
140
Key Position
0.0
0.2
0.4
0.6
0.8
1.0
Attention Probability
layer=30, head=24, query=49
(f)
Figure 9: Two maxima observed in Llama-3.1-8B. The first token has maxima due to the attention
sink.
2. Substitute the definition of dS. Next, we substitute the expressions for dShp and dSlp using the
definition from equation 7.
dShp = αP ◦(dP −δhp)
dSlp = αP ◦(dP −δlp)
Plugging these into our expression equation 8:
dQhp −dQlp = [(αP ◦(dP −δhp)) −(αP ◦(dP −δlp))] K
(9)
3. Simplify using properties of the Hadamard product. The Hadamard product (◦) is distributive
over matrix addition and subtraction, and we can factor out the common scalar α and matrix P.
dQhp −dQlp = [α (P ◦(dP −δhp) −P ◦(dP −δlp))] K
= [αP ◦((dP −δhp) −(dP −δlp))] K
Now, we simplify the expression inside the parentheses. The dP terms cancel each other out.
dQhp −dQlp = [αP ◦(dP −δhp −dP + δlp)] K
= [αP ◦(δlp −δhp)] K
4. Convert the Hadamard product to matrix multiplication. The final step involves rewriting the
row-wise scaling operation, represented by the Hadamard product with broadcasting, as a standard
matrix multiplication. Let the vector difference be v = δlp −δhp. The expression P ◦(δlp −δhp)
19

Published as a conference paper at ICLR 2026
Algorithm 3 Stabilized Flash Attention by Mitigating Biased Rounding Error: Forward Pass
Require: Matrices Q, K, V ∈RN×d, block sizes Bc, Br, β > 1, ϵ > 0.
1: Divide Q into Tr =
l
N
Br
m
blocks Q1, . . . , QTr of size Br × d each, and divide K, V in to
Tc =
l
N
Bc
m
blocks K1, . . . , KTc and V1, . . . , VTc, of size Bc × d each.
2: Divide the output O ∈RN×d into Tr blocks O1, . . . , OTr of size Br × d each, and divide the
logsumexp L into Tr blocks L1, . . . , LTr of size Br each.
3: for 1 ≤i ≤Tr do
4:
Initialize O(0)
i
= (0)Br×d ∈RBr×d, ℓ(0)
i
= (0)Br ∈RBr, m(0)
i
= (−∞)Br ∈RBr.
5:
for 1 ≤j ≤Tc do
6:
Compute S(j)
i
= QiKT
j ∈RBr×Bc.
7:
rm = rowmax(S(j)
i ), rs = rowsum(rm −S(j)
i
≤ϵ)
8:
m(j)′
i
= where(rm > 0 ∧rs > 1, βrm, rm)
9:
m(j)′
i
= where(rm < 0 ∧rs > 1, 0, m(j)′
i
)
10:
Compute m(j)
i
= max(m(j−1)
i
, m(j)′
i
) ∈RBr, ¯P(j)
i
= exp(S(j)
i
−m(j)
i ) ∈RBr×Bc
(pointwise), ℓ(j)
i
= em(j−1)
i
−m(j)
i ℓ(j−1)
i
+ rowsum(¯P(j)
i ) ∈RBr.
11:
Compute O(j)
i
= diag(em(j−1)
i
−m(j)
i )O(j−1)
i
+ ¯P(j)
i Vj.
12:
end for
13:
Compute Oi = diag(ℓ(Tc)
i
)−1O(Tc)
i
.
14:
Compute Li = m(Tc)
i
+ log(ℓ(Tc)
i
).
15:
Write Oi as the i-th block of O.
16:
Write Li as the i-th block of L.
17: end for
18: Return the output O and the logsumexp L.
means that each row of matrix P is element-wise multiplied by the vector v. Specifically, the i-th
row of P is scaled by the i-th element of v.
Let’s examine this at the element level. The (i, j)-th element of the resulting matrix is:
(P ◦v)ij = Pij · vi
This is precisely the result of pre-multiplying the matrix P by a diagonal matrix whose diagonal
entries are the elements of the vector v. Let D = diag(v) = diag(δlp −δhp). The (i, j)-th element
of the product DP is:
(DP)ij =
X
k
DikPkj
Since D is a diagonal matrix, Dik is non-zero only when k = i, where Dii = vi. Thus, the sum
simplifies to:
(DP)ij = DiiPij = vi · Pij
This confirms that P ◦v = diag(v)P. Applying this identity to our expression equation 10:
dQhp −dQlp = [α diag(δlp −δhp)P] K
Finally, using the associativity of matrix multiplication, we can regroup the terms to arrive at the
final equation.
dQhp −dQlp = α diag(δlp −δhp)(PK)
(10)
This completes the detailed derivation from the initial definitions to the final expression.
F
SUGGESTIONS FOR HANDLING FAILURE IN LOW-PRECISION TRAINING
Our analysis provides a blueprint for diagnosing numerical instabilities that may arise in other low-
precision formats, such as FP8. The analytical workflow presented in this paper is possible to be
generalized to other settings. The key steps are:
20

Published as a conference paper at ICLR 2026
(a) Loss in nanoGPT Issue #303
(b) Loss in nanoGPT Issue #524
Figure 10: Loss curves of two independent runs of GPT-2 training with flash attention in BF16
reported in the Github issue of nanoGPT.
1. Isolate the Source of Error: Systematically narrow down the failure to a specific operation
or module through targeted high-precision substitutions.
2. Identify Error Accumulation Mechanisms: Analyze the gradient error to find structural
patterns, such as the emergence of similar update directions that allow errors to accumulate
rather than cancel out.
3. Trace to Root Arithmetic Cause: Examine the low-level arithmetic of the specific format
to find the origin of any systematic bias.
By following this methodology, researchers and practitioners in this field can move from observing
a high-level training failure to understanding its root cause in the underlying hardware arithmetic,
enabling the development of principled solutions.
21

Published as a conference paper at ICLR 2026
THE USE OF LARGE LANGUAGE MODELS (LLMS)
We use LLMs for various purposes, including: polishing the language of the paper and assistant code
implementation. We carefully review and edit all content generated by LLMs to ensure accuracy and
appropriateness.
22

Published as a conference paper at ICLR 2026
0
2000
4000
6000
8000
10000
12000
Training Steps
0.attn.c_attn.weight.q
0.attn.c_attn.weight.k
0.attn.c_attn.weight.v
0.attn.c_proj.weight
0.mlp.c_fc.weight
0.mlp.c_proj.weight
1.attn.c_attn.weight.q
1.attn.c_attn.weight.k
1.attn.c_attn.weight.v
1.attn.c_proj.weight
1.mlp.c_fc.weight
1.mlp.c_proj.weight
2.attn.c_attn.weight.q
2.attn.c_attn.weight.k
2.attn.c_attn.weight.v
2.attn.c_proj.weight
2.mlp.c_fc.weight
2.mlp.c_proj.weight
3.attn.c_attn.weight.q
3.attn.c_attn.weight.k
3.attn.c_attn.weight.v
3.attn.c_proj.weight
3.mlp.c_fc.weight
3.mlp.c_proj.weight
4.attn.c_attn.weight.q
4.attn.c_attn.weight.k
4.attn.c_attn.weight.v
4.attn.c_proj.weight
4.mlp.c_fc.weight
4.mlp.c_proj.weight
5.attn.c_attn.weight.q
5.attn.c_attn.weight.k
5.attn.c_attn.weight.v
5.attn.c_proj.weight
5.mlp.c_fc.weight
5.mlp.c_proj.weight
6.attn.c_attn.weight.q
6.attn.c_attn.weight.k
6.attn.c_attn.weight.v
6.attn.c_proj.weight
6.mlp.c_fc.weight
6.mlp.c_proj.weight
7.attn.c_attn.weight.q
7.attn.c_attn.weight.k
7.attn.c_attn.weight.v
7.attn.c_proj.weight
7.mlp.c_fc.weight
7.mlp.c_proj.weight
8.attn.c_attn.weight.q
8.attn.c_attn.weight.k
8.attn.c_attn.weight.v
8.attn.c_proj.weight
8.mlp.c_fc.weight
8.mlp.c_proj.weight
9.attn.c_attn.weight.q
9.attn.c_attn.weight.k
9.attn.c_attn.weight.v
9.attn.c_proj.weight
9.mlp.c_fc.weight
9.mlp.c_proj.weight
10.attn.c_attn.weight.q
10.attn.c_attn.weight.k
10.attn.c_attn.weight.v
10.attn.c_proj.weight
10.mlp.c_fc.weight
10.mlp.c_proj.weight
11.attn.c_attn.weight.q
11.attn.c_attn.weight.k
11.attn.c_attn.weight.v
11.attn.c_proj.weight
11.mlp.c_fc.weight
11.mlp.c_proj.weight
Model Layers
Spectral Norm Across Layers and Training Steps
10
20
30
40
50
60
70
80
Spectral Norm Value
Figure 11: Spectral norm across layers and training steps.
23

Published as a conference paper at ICLR 2026
the
Western
classical
canon
have
repeatedly
quoted
or
otherwise
employed
Hay
dn
's
tune
,
as
is
demonstrated
by
the
following chronological
list
.
As
the
tune
was
widely
known
,
the
uses
by
other
compos
ers
were
heard
as
quotations
and
served
as
an
emblem
of
Austria
,
of
Austrian
patriotism
,
or
of
the
Austrian
monarchy
.
Use
in
national
an
the
ms
,
al
ma
mat
ers
,
and
hy
m
ns
[
edit
]
After
the
death
of
Francis
in
18
35
,
the
tune
was
given
new
lyrics
that
praised
his
successor
,
Ferdinand
:
"
Seg
en
Ã
st
're
ich
s
ho
hem
S
oh
ne
/
Un
ser
m
Kaiser
Ferdinand
!"
("
B
less
ings
to
Austria
's
high
son
/
Our
Emperor
Ferdinand
!"
).
After
Ferdinand
's
ab
d
ication
in
18
48
,
the
original
lyrics
were
used
again
because
his
successor
(
Franc
is
Joseph
)
was
also
named
Francis
.
However
,
in
18
54
,
yet
again
new
lyrics
were
selected
:
"
G
ott
er
hal
te
,
Gott
bes
ch
Ã¼
t
ze
/
Un
s
ern
Kaiser
,
un
ser
Land
!"
("
God
preserve
,
God
protect
/
Our
Emperor
,
our
country
!"
).
There
were
versions
of
the
hy
mn
in
several
languages
of
the
Aust
ro
-
Hung
arian
Empire
(
e
.
g
.,
Czech
,
Croatian
,
Sloven
e
,
Hungarian
,
Polish
,
Italian
).
At
the
end
of
the
First
World
War
in
1918
,
the
Aust
ro
-
Hung
arian
Empire
was
abolished
and
divided
into
multiple
states
,
one
of
them
being
the
residual
state
of
Austria
,
which
was
a
republic
and
had
no
emperor
.
The
tune
ceased
to
be
used
for
official
purposes
.
When
the
last
Emperor
,
Charles
I
,
died
in
1922
,
monarch
ists
created
an
original
st
anza
for
his
son
Otto
von
H
abs
burg
.
Since
the
emperor
was
in
fact
never
restored
,
this
version
never
attained
official
standing
.
The
hy
mn
was
revived
in
1929
with
completely
new
lyrics
,
known
as
Se
i
g
es
eg
net
oh
ne
End
e
,
which
remained
the
national
anthem
of
Austria
until
the
An
sch
l
uss
.
The
first
st
anza
of
the
hy
mn
's
18
54
version
was
sung
in
1989
during
the
funeral
of
Empress
Z
ita
of
Austria
[
14
]
and
again
in
2011
during
the
funeral
of
her
son
Otto
von
H
abs
burg
[
15
]
as
a
tribute
to
the
family
.
Germany
[
edit
]
Long
after
Hay
dn
's
death
,
his
melody
was
used
as
the
tune
for
Hoff
mann
von
Fall
ers
le
ben
's
poem
"
D
as
L
ied
der
De
ut
sc
hen
"
(
18
41
).
The
third
st
anza
("
E
in
ig
ke
it
und
Re
cht
und
Fre
i
heit
"),
sung
to
the
melody
,
is
the
national
anthem
of
Germany
,
the
"
De
utsch
land
lied
".
H
ym
ns
[
edit
]
In
the
ordinary
n
omen
cl
ature
of
hy
mn
tunes
,
the
melody
of
"
G
ott
er
hal
te
Franz
den
Kaiser
"
is
classified
as
87
.
87
D
tro
cha
ic
metre
.
When
employed
in
a
hy
mn
it
is
sometimes
known
as
Austria
.
It
has
been
paired
with
various
lyrics
.
University
hy
m
ns
[
edit
]
Various
American
schools
,
colleges
,
and
universities
use
Hay
dn
's
music
as
the
tune
for
their
university
or
school
hy
m
ns
.
Here
is
a
partial
list
.
Middle
school
hy
mn
[
edit
]
In
China
,
the
Ning
bo
No
.
2
Middle
School
[
zh
]
use
Hay
dn
's
music
as
the
tune
for
their
middle
school
hy
mn
.[
17
]
Full
text
[
edit
]
Original
version
(
17
97
)
[
edit
]
German
English
translation
Gott
er
hal
te
Franz
,
den
Kaiser
,
Un
s
ern
g
uten
Kaiser
Franz
!
L
ange
le
be
Franz
,
der
Kaiser
,
In
des
Gl
Ã¼
ck
es
hell
stem
Gl
anz
!
I
hm
er
bl
Ã¼
hen
Lor
beer
re
iser
,
Wo
er
ge
ht
,
z
um
Eh
ren
k
ran
z
!
G
ott
er
hal
te
Franz
,
den
Kaiser
,
Un
s
ern
g
uten
Kaiser
Franz
!
La
Ã
von
se
iner
Fah
ne
Sp
itz
en
St
rah
len
Sieg
und
F
ru
cht
bar
ke
it
!
La
Ã
in
se
inem
Rate
S
itz
en
We
is
heit
,
Kl
ug
heit
,
Red
lich
ke
it
;
Und
mit
Se
iner
Ho
heit
Blitz
en
Sch
al
ten
n
ur
G
ere
cht
ig
ke
it
!
|
:
Gott
er
hal
te
Franz
,
den
Kaiser
,
Un
s
ern
g
uten
Kaiser
Franz
!
:
|
Str
Ã¶
me
de
iner
Gab
en
F
Ã¼
l
le
Ã
ber
i
hn
,
se
in
H
aus
und
Reich
!
B
rich
der
Bos
heit
Mach
t
,
ent
h
Ã¼
l
le
J
eden
Sc
helm
-
und
Bub
en
stre
ich
!
De
in
Ges
etz
se
i
st
ets
se
in
Wil
le
,
D
ies
er
uns
Ges
et
zen
gle
ich
.
|
:
Gott
er
hal
te
Franz
,
den
Kaiser
,
Un
s
ern
g
uten
Kaiser
Franz
!
:
|
F
ro
h
er
le
b
'
er
se
iner
Sum: 1.81e-06
1
0
1
2
3
4
5
6
D_diff Values
1e
7
Figure 12: Token difference visualization
0
2000
4000
6000
8000
10000
12000
14000
Training Steps
4
5
6
7
8
9
10
11
Validation Loss
Loss
Number of Activated SFA Condition
0
2
4
6
8
Number of Activated SFA Condition
1e9
Validation Loss and SFA Condition Activation During Training
Figure 13: Correlation between multiple maxima occurrences and loss curve.
24
