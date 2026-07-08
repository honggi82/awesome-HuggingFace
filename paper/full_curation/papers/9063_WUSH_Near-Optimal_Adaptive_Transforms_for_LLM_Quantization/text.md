## WUSH: Near-Optimal Adaptive Transforms for LLM Quantization

Jiale Chen1 Vage Egiazarian1 Roberto L. Castro2 Torsten Hoefler3 Dan Alistarh12

# arXiv:2512.00956v3[cs.LG]30May2026

### Abstract

Quantizing LLM weights and activations is a standard approach for efficient deployment, but a few extreme outliers can stretch the dynamic range and amplify low-bit quantization errors. Prior transform-based mitigations (e.g., Hadamard rotations) are fixed and data-agnostic, and their optimality for quantization has remained unclear. We derive closed-form optimal linear blockwise transforms for joint weight-activation quantization under standard RTN AbsMax-scaled block quantizers, covering both integer and floatingpoint formats. The resulting construction, WUSH, combines a Hadamard backbone with a datadependent second-moment component to form a non-orthogonal transform that is provably nearoptimal for FP and INT quantizers under mild assumptions while admitting an efficient fused GPU implementation. Empirically, WUSH improves W4A4 accuracy over the strongest Hadamardbased baselines (e.g., on Llama-3.1-8B-Instruct in MXFP4, it gains +2.8 average points with RTN and +0.7 with GPTQ) while delivering up to 5.8× per-layer throughput over BF16 via FP4 MatMul. Source code is available at https:

//github.com/IST-DASLab/WUSH.

### 1. Introduction

Quantization of model weights (Dettmers et al., 2022; Frantar et al., 2023; Lin et al., 2024) or activations (Xiao et al., 2023; Ashkboos et al., 2024b) is now a standard tool for shrinking and accelerating large language models (LLMs) (Kurtic et al., 2025), making low-precision inference feasible on a wide range of hardware. A central difficulty, however, is that a few extreme “outlier” weights and activations expand the dynamic range and thereby degrade the effective resolution of low-bit representations.

1Institute of Science and Technology Austria (ISTA) 2Red Hat AI 3ETH Zürich. Correspondence to: Jiale Chen <jiale.chen@ist.ac.at>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

One way to mitigate such outliers is to apply linear transforms before quantization (Jegou et al., 2008; Jégou et al., 2011); in the case of LLM quantization, this is often in the form of rotations that spread variance more evenly across channels (Chee et al., 2023; Tseng et al., 2024a; Ashkboos et al., 2024b; Liu et al., 2025). For LLMs, Hadamard rotations have been remarkably effective, and blockwise variants aligned with quantization groups have also been shown to be useful in practice. Egiazarian et al. (2026) recently showed that the blockwise Hadamard transform offers the best empirical performance for quantization among a large set of existing transforms. Yet, these transforms are typically fixed and data-agnostic: in particular, the Hadamard transform does not adapt to the statistics of the underlying weights or activations. This raises a natural question: if the Hadamard transform is not data-aware, in what sense can it be considered optimal for quantization?

In this work, we address this question by deriving closedform optimal linear blockwise transforms for joint weightactivation quantization, which are generally non-orthogonal and adaptive, i.e., data-aware. As opposed to prior methods such as SpinQuant (Liu et al., 2025) or FlatQuant (Sun et al., 2025) that learn transforms on calibration data via iterative optimizations (e.g., gradient descent), our approach is a closed-form calibration-driven transform. We call our construction WUSH, which is a mnemonic for its composition in Eq. (7). We show that WUSH is optimal for floating-point (FP) block quantizers and asymptotically optimal for integer (INT) block quantizers. Empirically, WUSH significantly improves end-to-end accuracy over prior transform-based baselines, substantially narrows the gap between NVFP and MXFP formats, and provides consistent benefits when combined with GPTQ (Frantar et al., 2023). It improves W4A4 (4-bit weights, 4-bit activations) accuracy by up to +2.8 average points (MXFP4 RTN) and +0.7 points (MXFP4 GPTQ) over the Hadamard-based baseline on Llama-3.1-8BInstruct. Despite using a distinct (data-aware) transform per block, WUSH admits an efficient fused GPU kernel whose throughput matches that of optimized blockwise Hadamard kernels while delivering up to 5.8× per-layer speedups over BF16 due to the lower-precision (FP4) matrix multiplication.

### 2. Related Work

Post-training quantization (PTQ) and outliers. PTQ schemes, such as RTN (round-to-nearest), OBQ (Frantar & Alistarh, 2022), and GPTQ (Frantar et al., 2023), are sensitive to heavy-tailed outliers in the weights and activations of LLMs because a small number of extreme values can determine the quantization scale, leading to high error.

Non-uniform bitwidths and explicit outlier storage. One common strategy to mitigate outliers is to use non-uniform bitwidths. Methods such as LLM.int8() (Dettmers et al., 2022), SpQR (Dettmers et al., 2024), and QUIK (Ashkboos et al., 2024a) explicitly separate and store outliers in higher precision, while HPTQ (Chen et al., 2026) uses Huffman encoding to compress the outliers and reduce their storage costs. These approaches can achieve high accuracy at a low effective bitwidth; however, the resulting formats are irregular and require specialized kernels.

Transform-based outlier mitigation. Another line of work applies transforms to the weights and activations before quantization to reduce the impact of outliers. SmoothQuant (Xiao et al., 2023) and AWQ (Lin et al., 2024) rescale channels to stabilize the dynamic ranges of weights and activations. QuIP (Chee et al., 2023) introduces an incoherence processing step. QuIP# (Tseng et al., 2024a), QTIP (Tseng et al., 2024b), and QuaRot (Ashkboos et al.,

- 2024b) use Hadamard transforms to spread outlier energy across channel dimensions, while SpinQuant (Liu et al.,
- 2025) and FlatQuant (Sun et al., 2025) learn transforms optimized on calibration data. However, their transforms are heuristic and costly to learn, which limits their accuracy or practicality when applied to fast, per-token activation quantization in large models.

Blockwise transforms and emerging FP formats. Recent FP formats, such as MXFP (MX Alliance, 2023) and NVFP (NVIDIA, 2024), have motivated blockwise transform schemes. Shao et al. (2025) indicates that blockwise Hadamard transforms can be more effective under these FP formats than full-layer transforms, but Egiazarian et al. (2026) highlights the limited gains achievable with Hadamard alone, linking this to the properties of the underlying weight and activation distributions. In contrast, this work (WUSH) derives a closed-form, data-aware optimal blockwise transform and analyzes how such a transform interacts with both FP and INT block quantizers.

### 3. Methodology

Problem setup. We formulate the transformed weightactivation quantization problem as follows. Define W ∈ Rd

in×dout as the weight of a linear layer with din input channels and dout output channels. Define X ∈ Rd

in×dbatch as the calibration input activation with din embedding chan-

nels and dbatch tokens. Define q (·) as a quantizer that maps a matrix of continuous values to quantized values (we will specify q later). Let TW,TX ∈ Rd

in×din be the transforms applied to W and X, respectively. For weightactivation quantization, the output activation W⊤X becomes q (TWW)⊤ q (TXX). Our goal is to choose TW and TX such that the L2 output loss

ℓ = d−out1 d−batch1 q (TWW)⊤ q (TXX) − W⊤X

2 F

(1)

is minimized under the given quantizer q. The constant d−out1 d−batch1 is added to simplify the analysis later.

Block-independent constraint. While the weights can be pre-quantized, the activations must be transformed and quantized dynamically at inference time. To reduce the computational overhead of these operations, it is common (Egiazarian et al., 2026) to constrain TW,TX to be block-diagonal transforms, and q to be an RTN (round-to-nearest) quantizer with AbsMax (the maximum absolute value within a group) scales. We assume that the quantization is applied to each sub-column vector of shape d × 1 and that the transform block size aligns with the quantization group size d, with d being a factor of din and a power of 2. Denote the block partitions of the matrices as

(i)∈Rd×d,

##### TW = diag TW

,...,TW(

din/d) , TW

(1)

(i)∈Rd×d,

##### TX = diag TX

##### ,...,TX(

din/d) , TX

(1)

(2)

- W⊤ = W(1)⊤ ,...,W(⊤d

in/d) , W(i) ∈ Rd×d

out

,

- X⊤ = X(1)⊤ ,...,X(⊤d

in/d) , X(i) ∈ Rd×d

.

batch

Then, we can express the output without and with weightactivation quantization, respectively, as

W⊤X=

q(TWW)⊤q(TXX)=

din/d

W(⊤i)X(i),

i=1

din/d

W(i) ⊤q TX

q TW

(i)

i=1

X(i) .

(i)

(3)

Finally, define the (normalized) blockwise output loss as

W(i) ⊤q TX

ℓ(i)=d−out1 d−batch1 q TW

(i)

(i)

X(i) −W(⊤i)X(i)

2 F

. (4)

We will approximate the layerwise loss as ℓ ≈ di=1in/d ℓ(i) and minimize the blockwise losses ℓ(i) independently.

Optimal block-diagonal transforms. In the following, we will prove, under reasonable assumptions, that there exist closed-form optimal transforms for each block. These transforms are provably optimal for FP and asymptotically

optimal for INT quantization. Let W(′i),X(′i) ∈ Rd×d be matrices that satisfy

- W(′i)W(′⊤i) = d−out1 W(i)W(⊤i),
- X(′i)X(′⊤i) = d−batch1 X(i)X(⊤i),

(5)

respectively. Without loss of generality (Appendix Section A.1), we let W′ and X′ be the lower triangular matrices from the Cholesky decomposition of the second moments d−out1 W(i)W(⊤i) and d−batch1 X(i)X(⊤i). In the case of rank W(i) < d or rank X(i) < d, we can dampen the diagonal of the second moments before the Cholesky decom-

position. Let the orthogonal matrices U(i),V(i) ∈ Rd×d and the diagonal matrix S(i) ∈ Rd×d be the singular value decomposition (SVD) of

W(′⊤i)X(′i) = U(i)S(i)V(⊤i). (6) Let H ∈ ±d−12

d×d be a normalized (orthogonal) Hadamard matrix. Then, we will show that the optimal TW

and TX

can be constructed as

(i)

(i)

1 2

Txvsh(i) = HS−

(i) V(⊤i)X(′⊤i), Twush(i) = HS−

- 1

- 2

(i) U(⊤i)W(′⊤i),

(7)

respectively. Note that1

Txvsh(i) = Twush−⊤ (i), (8) which can be easily verified by calculating Txvsh(i)Twush⊤ (i) = I, where I is the identity matrix. The WUSH construction also jointly balances the weight and activation scales. Appendix Section A.3 discusses and visualizes this effect.

Remark. The Hadamard matrix H is the only data-agnostic ingredient in our optimal formulation in Eq. (7). Incidentally, this explains why it has been empirically observed to be an effective data-agnostic orthogonal transform.

GPTQ integration. In practice, the second-order information d−batch1 X(i)X(⊤i) in Eq. (5) can be obtained either from the calibration activations or from the Hessian matrix; the latter coincides with the Hessian used in GPTQ. Therefore, it is a natural idea to integrate WUSH into GPTQ. However, GPTQ updates the weights iteratively via error propagation, while the WUSH transform is constructed from the secondorder statistics of the updated weights. This coupling between weight updates and transform construction requires an interleaved computational schedule. Algorithm 2 summarizes the procedure used to compute the blockwise WUSH transforms and pre-quantized weights for a linear layer, using either RTN or GPTQ. The GPTQ error propagations are

1The −⊤ superscript means (·)−⊤ = ((·)−1)⊤ = ((·)⊤)−1.

- Algorithm 1 Compute WUSH for One Block

Input: weight second moment MW ∈ Rd×d, activation second moment MX ∈ Rd×d, damping ratio λ ∈ R≥0 Output: WUSH transform Twush ∈ Rd×d

Initialize d × d identity matrix I Initialize normalized d × d Hadamard matrix H

- W′W′⊤ ← Cholesky MW + λd−1 tr(MW)I
- X′X′⊤ ← Cholesky MX + λd−1 tr(MX)I USV ⊤ ← SVD W′⊤X′ ▷ Eq. (6) Twush ← HS−21 U⊤W′⊤ ▷ Eq. (7)

- Algorithm 2 Compute WUSH and Pre-Quantize Weights

#### Input: weights W ∈ Rd

in×dout, activations X ∈ Rd

in×dbatch, damping ratio λ ∈ R≥0 Output: WUSH transform Twush(i) ∈ Rd×d and prequantized weight W(i) ∈ Rd×d

out for each block i ∈ {1,...,din/d}

▷ RTN (round-to-nearest) case for i ← 1 to din/d (in parallel) do

MW(i),MX(i) ← d−out1 W(i)W(⊤i), d−batch1 X(i)X(⊤i) Twush(i) ← WUSH MW(i),MX(i),λ ▷ Algorithm 1 Txvsh(i) ← Twush−⊤(i) ▷ Eq. (8) W(i) ← q Txvsh(i)W(i) ▷ standard RTN

#### end for

▷ GPTQ case Initialize din × din identity matrix I H ← d−batch1 XX⊤ ▷ GPTQ’s Hessian MX(i) i ∈ {1,...,din/d} ← diagonal blocks of H

▷ second moment of activations (same as the RTN case)

##### LL⊤ ← Cholesky H + λd−in1 tr(H)I −1

▷ Cholesky of Hessian inverse (same as GPTQ)

L(i,j) ∈ Rd×d i,j ∈ {1,...,din/d} ← blocks of L

- for i ← 1 to din/d do

MW(i) ← d−out1 W(i)W(⊤i)

▷ second moment of updated weight block Twush(i) ← WUSH MW(i),MX(i),λ ▷ Algorithm 1 Txvsh(i) ← Twush−⊤(i) ▷ Eq. (8) W(i) ← Txvsh(i)W(i) ▷ transformed weights H(i) ← Twush(i) L(i,i)L⊤(i,i) −1Twush⊤(i)

▷ transformed Hessian

W(i) ← GPTQ weight = W(i),Hessian = H(i)

▷ standard GPTQ subroutine (intra-block error propagation) E(i) ← Txvsh−(i)1 W(i) − W(i) ▷ error in original space

- for j ← i to din/d (in parallel) do

W(j) ← W(j) + L(j,i)L−(i,i1)E(i)

▷ inter-block GPTQ error propagation

#### end for end for

decomposed into intra-block and inter-block updates similar to its original paper (Frantar et al., 2023). For the intrablock updates, we apply the standard GPTQ subroutine to the current transformed weight block using the corresponding transformed Hessian. For inter-block propagation, we follow GPTQ’s usual blockwise updates. The model-level calibration pipeline closely follows that of GPTQ: layers are processed sequentially, and after processing a layer, the calibration activations are propagated through the quantized layer to provide the inputs for calibrating the next one. During inference, the forward pass of a linear layer is calculated as

din/d

W(⊤i)q Twush(i) X(i) (9)

i=1

with W(i) ∈ Rd×d

out being the pre-quantized transformed weight blocks and X(i) ∈ Rd×d

batch being the new activation blocks instead of the calibration ones.

Complexity analysis for offline preprocessing. The offline processing cost of WUSH is very similar to that of the standard GPTQ algorithm because WUSH and GPTQ require the same type of activation Hessian information, and the cost is dominated by forwarding the calibration activations through the model. In particular, when combining WUSH and GPTQ, WUSH only adds a negligible overhead on top of GPTQ. For a layer, WUSH processes only din/d diagonal blocks. In a typical setting, d ≪ din ≪ dbatch and d ≪ dout ≪ dbatch. Per block, forming second moments costs O d2dbatch time and O (ddbatch) memory, and applying the transform to the weights (for pre-quantization) costs O d2dout time and O (ddout) memory, while the Cholesky/SVD and the multiplication steps on small d × d matrices cost only O d3 time and O d2 memory. Therefore, the total per-block offline time and memory costs are O d2dbatch and O (ddbatch), and the per-layer costs are O (ddindbatch) and O (dindbatch). As a comparison, computing the Hessian matrix of the standard GPTQ for a layer requires O d2indbatch time and O (dindbatch) memory.

Costs for online inference. The only additional inferencetime overhead is the activation-side block transform consisting of d × d × din/d = ddin elements per layer, while the weight-side transform is absorbed into the pre-quantized weights. With 4-bit weights (plus groupwise 8-bit scales) and 16-bit transforms, this is roughly a 16ddin

relative storage overhead per layer. The memory is dominated by the KV cache and model weights, and the storage cost of the transforms is negligible for typical LLM dimensions. The online transform is fused with activation quantization in a kernel detailed in Section 5, leading to negligible runtime overhead.

4dindout = d4d

out

### 4. Theoretical Derivation

In this section, we focus on the question of finding optimal transforms that minimize the loss of one block ℓ(i) in Eq. (4). To build intuition for the proof, Figure 1 visualizes how different transforms reshape the expected quantization error under FP and INT block quantizers, in a 2D toy setting. Throughout, we omit the block subscript (i) to simplify notation. Due to space limitations, the detailed derivation steps of some equations are deferred to Appendix Section A.4.

#### 4.1. Problem Setup and General Proof Approach

Probabilistic reformulation. We reformulate the problem from a probabilistic perspective. We split the matrices W = [w1,...,wd

] into columns wk,xk ∈ Rd and treat the columns as i.i.d. samples from d-dimensional independent distributions w ∼ Dw and x ∼ Dx, respectively. We motivate this multivariate modeling choice in Appendix Section A.2. We can reinterpret the blockwise loss in Eq. (4) as

] and X = [x1,...,xd

out

batch

ℓ = Ew,x q (TWw)⊤ q (TXx) − w⊤x

2

. (10)

Similarly, we can also reinterpret Eq. (5) so that the matrices W′ and X′ satisfy

- W′W′⊤ = Ewww⊤,
- X′X′⊤ = Exxx⊤,

(11)

respectively.

Unbiased quantization. We assume that q is a stochastic and unbiased quantizer, i.e., for a vector α ∈ Rd, the quantization error ε(α) = q (α) − α is a random vector with Eε(α)ε(α) = 0. We further constrain

##### TW = TX−⊤ (12)

such that q (TWw)⊤ q (TXx) is unbiased with respect to w⊤x. Then, using the first-order approximation, we can split ℓ in Eq. (10) into two non-negative terms:

Ww) X′⊤TW−1ε(TWw) 2

ℓ = Ew,ε(T

Xx) W′⊤TX−1ε(TXx) 2 .

+ Ex,ε(T

(13)

The detailed derivation is given in Appendix Eq. (38).

Problem reduction. We can compute the optimal TW and TX individually for each of the non-negative terms and verify that TW = TX−⊤. The two terms have similar forms and can be optimized in similar ways. We focus on finding the optimal TX for the second term. Define the d-dimensional random variable y = W′⊤x and denote its distribution as

y ∼ Dy. Define the transformation matrix T = TXW′−⊤. Then, the second term in Eq. (13) becomes

FP (Original Space) FP (Transformed Space) INT (Original Space) INT (Transformed Space)

T=IIdentity

Xx) W′⊤TX−1ε(TXx) 2

Ex,ε(T

(14)

- = Ex,ε(T

Xx) TXW′−⊤ −1 ε TX W′−⊤y

2

- = Ey,ε(T y) T−1ε(Ty) 2

−1⊤⊤−1/2⊤−1/2⊤T=SUT=HUWhiteningT=HT=SUT=HSUCalibratedHadamardHadamardWUSWUSH

Therefore, the two-sided quantization problem in Eq. (10) is reduced to finding the optimal T ∈ Rd×d that minimizes the one-sided quantization loss Ey,ε(T y) T−1ε(Ty) 2 and compute

#### TX = TW′⊤. (15)

Transformation parameterization. Let the unknown orthogonal matrices U′,R ∈ Rd×d and the unknown diagonal matrix S′ ∈ Rd×d be the singular value decomposition (SVD) of TUS = U′S′R⊤ with U,S defined in Eq. (6); thus, T can be parameterized as

#### T = U′S′R⊤S−1U⊤. (16)

Denote S = diag (s1,...,sd) and S′ = diag (s′1,...,s′d). Without loss of generality, assume s1 ≥ ··· ≥ sd > 0 and s′1 ≥ ··· ≥ s′d > 0. A useful property of S is

tr S2 ≥ d−1 (tr(S))2 ≥ d−1 tr S2 (17)

with detailed steps in Eq. (39). The equality tr S2 = d−1 (tr(S))2 is attained when all si are the same (all singular values are inliers). The equality d−1 (tr(S))2 = d−1 tr S2 is attained when tr(S) → s1 (a singular value is an extreme outlier).

Coordinate Axis Example Point 1 Random Noise Distributing Zone of Point 1

Basis Grid

Example Point 2

Random Noise Distributing Zone of Point 2

Expected Error (Shown as Radius) of Point 1

Expected Error (Shown as Radius) of Point 2

Change of Expected Error after Transform (Blue ↓ White - Red ↑)

Probabilistic Contour of Data Distribution

Equilibrium Line of Expected Error Change

Theorem 4.1 (Optimal Transform). The optimal configuration of T for floating-point (FP) data types is U′ = H, S′ = S 12, and R = I. For integer (INT) types, the same configuration is optimal up to a do(1) factor for zeromean multivariate Gaussian/Laplacian distributed data and within a d factor for any distribution.

Figure 1. 2D illustration of how transforms shape the one-sided quantization error ∥T−1ε (Ty) ∥2 in Eq. (14) under FP and INT block quantizers. Rows correspond to different transforms T (identity; Hadamard: 45° rotation; calibrated Hadamard: equally spreading energy into each dimension; whitening; WUS: WUSH without Hadamard; WUSH), while columns show FP and INT quantizations in both original and transformed spaces. The dotted grid shows the basis induced by the transform. Two example points are shown in detail with their induced quantization noise support (Sections 4.2.1 and 4.3.1) represented as shaded parallelograms, and the corresponding average (expected) error magnitude (Eq. (14)) represented as the radii of circles (not necessarily proportional to the area of parallelograms). For each point on the plane, we visualize its expected error change (compared to that of the identity transform) calculated in the original space, with blue/red background heatmaps indicating lower/higher and dashed lines indicating the equilibrium set where the expected error is unchanged. No transform can reduce the error for all points on the plane. Thus, the key idea is to reduce the error around typical data distributions. The black ellipse depicts a representative data distribution contour. Overall, WUSH more effectively reduces the error by aligning the ellipse’s major axis to the error reduction (blue) regions.

Proof. We provide detailed proofs for FP and INT in Sections 4.2 and 4.3, respectively. For each data type, we apply a two-step proof strategy. We first (Sections 4.2.1 and 4.3.1) provide a smooth modeling of the quantizer and treat the type casting errors as random noise. Secondly (Sections 4.2.2 and 4.3.2), we calculate the expected quantization error, Eq. (14), with respect to the parameterized transform, Eq. (16), and minimize this expectation by solving the corresponding optimization problem.

| |
|---|

The optimal TX, namely the WUSH construction Twush in Eq. (7), is obtained by applying Theorem 4.1 and Eqs. (15), (16).

#### 4.2. Proof for Floating-Point (FP) Types

- 4.2.1. QUANTIZATION ERROR MODELING

For a number in FP types, the quantization error tends to be proportional to its absolute value. Formally, for a vector α ∈ Rd, we can model the quantization error as

ε(α) = diag (η)α (18)

where η = [η1,...,ηd]⊤ ∈ Rd is a random vector of i.i.d. samples from the distribution η ∼ Dη with Eηη = 0. We provide more rigorous justifications for why this modeling makes sense in Appendix Section A.5.2.

- 4.2.2. LOSS MINIMIZATION

Using the quantization error modeling in Eq. (18), the minimization objective in Eq. (14) becomes

Ey,ε(T y) T−1ε(Ty) 2

= Eηη2 tr T−1 U′S′2U′⊤ ⊙ I T−⊤

(19)

with ⊙ representing elementwise multiplication, and with detailed steps in Eq. (40). The lower bound of the trace term in Eq. (19) is

tr T−1 U′S′2U′⊤ ⊙ I T−⊤ ≥ d−1 (tr(S))2 (20) and the equality can be attained by choosing U′ = H,

- S′ = S 12 , and R = I, with detailed steps in Eqs. (41), (42).

4.2.3. DISCUSSION Under our smooth modeling in Eq. (18), for any orthogonal

- T (including the identity T = I and the Hadamard T = H), the minimization objective becomes trivially

Ey,ε(T y) T−1ε(Ty) 2 = Eηη2 tr S2 (21)

with detailed steps in Eq. (43). Thus, orthogonal transforms will not be helpful for reducing the quantization error. Comparing the trace terms in Eqs. (20), (21) using the inequality Eq. (17), the WUSH transform can reduce the error by at most d times in the extreme outlier scenario. Also note that the non-trivial choices U′ = H and S′ = S 12 are both essential for reducing the error. Either trivial choices of

- U′ = I or S′ = I will lead to the same suboptimal trace term

#### 4.3. Proof for Integer (INT) Data Types

- 4.3.1. QUANTIZATION ERROR MODELING

For a number in INT types, the quantization error tends to be proportional to the maximum absolute value within a quantization group. Formally, for a vector α ∈ Rd, we can model the quantization error as

ε(α) = ∥α∥∞ η (23)

where η = [η1,...,ηd]⊤ ∈ Rd is a random vector of i.i.d. samples from the distribution η ∼ Dη with Eηη = 0. We provide more rigorous justifications for why this modeling makes sense in Appendix Section A.5.1.

- 4.3.2. LOSS MINIMIZATION

Using the quantization error modeling in Eq. (23), the minimization objective in Eq. (14) becomes

Ey,ε(T y) T−1ε(Ty) 2= Eηη2 T−1 2FEy ∥Ty∥2∞ (24) with detailed steps in Eq. (46). We can obtain a lower bound for the term T−1 2F in Eq. (24).

T−1 2F ≥ tr S2S′−2 , (25)

and the equality is attained when R = I, with detailed steps in Eqs. (47), (48).

Next, we obtain near-optimal lower and upper bounds for the term Ey ∥Ty∥2∞ in Eq. (24) by choosing U′ = H.

Ey ∥Ty∥2∞ ≥ d−1 tr S′2 ,

do(1)−1 tr S′2 , tail-bounded Dy, tr S′2 , otherwise.

Ey ∥Ty∥2∞ ≤

(26)

Note that the upper bound is tighter when Dy is a tailbounded (zero-mean multivariate Gaussian or Laplacian) distribution than for a general distribution. For the full derivations of these bounds, please refer to Eqs. (49) to (58).

Taken Eqs. (25), (26) together, by setting R = I and U′ = H, the term T−1 2F Ey ∥Ty∥2∞ is bounded by tr S2S′−2 tr S′2 within a gap of d such that

tr T−1 U′S′2U′⊤ ⊙ I T−⊤ = tr S2 (22)

as that in the case of the orthogonal T in Eq. (21), with detailed steps in Eqs. (44), (45). Figure 1 (left two columns) also visualizes these results.

d−1 tr S2S′−2 tr S′2 ≤ T−1 2F Ey ∥Ty∥2∞ ≤ tr S2S′−2 tr S′2 .

(27)

And for a tail-bounded Dy, the gap is narrowed to do(1) such that

T−1 2F Ey ∥Ty∥2∞=do(1)−1 tr S2S′−2 tr S′2 . (28)

By the Cauchy-Schwarz inequality, the lower bound of the trace product term is

tr S2S′−2 tr S′2 ≥ tr S2S′−2

= (tr(S))2 .

- 1

- 2

S′2

- 1

- 2

2

(29)

and the equality is attained when S′ ∝ S 21 . Without loss of generality, we can choose S′ = S 12 to minimize both the lower and upper bounds in Eqs. (27), (28), and the quantization loss in Eq. (24) becomes near-optimal:

d−1 (tr(S))2 ≤ T−1 2F Ey ∥Ty∥2∞ ≤

do(1)−1 (tr(S))2 , tail-bounded Dy, (tr(S))2 , otherwise.

(30)

- 4.3.3. DISCUSSION

Consider the case of T being orthogonal. For any general distribution Dy, the bounds are

tr S2 ≤ T−1 2F Ey ∥Ty∥2∞ ≤ dtr S2 (31)

with detailed steps in Eq. (59). For a tail-bounded Dy, the Hadamard T = H is empirically the best orthogonal transform, so the bounds may be tightened to

T−1 2F Ey ∥Ty∥2∞ = do(1) tr S2 (32)

Comparing Eq. (30) with Eqs. (31), (32) using the inequality Eq. (17), an orthogonal T is suboptimal, and the error bounds of the Hadamard T = H can be at most d times larger than those of the optimal T, in the extreme outlier scenario. Figure 1 (right two columns) provides visualizations of different transforms.

- 5. GPU Kernel Support

As noted in Eq. (9), part of the WUSH transform has to be applied online during inference, requiring a specialized GPU kernel. Online blockwise Hadamard rotations are efficiently supported in MR-GPTQ’s kernels (Egiazarian et al., 2026). Yet, in their case, the transforms are dataindependent, so it is possible to reuse a single transform across all blocks. By contrast, WUSH assigns a distinct transform to each block. This per-block specialization significantly complicates kernel reuse and fusion in existing low-level libraries.

Fused WUSH + Quant kernel implementation. A first design decision is the storage layout of the WUSH matrices. For block size G and number of blocks C, we store matrices as (G,G,C), i.e., transposed with respect to the channel dimension. Since this transposition is performed offline, it does not introduce any runtime overhead. This layout

choice is motivated by the observation that, for small group sizes (e.g., G=32 for MXFP4), the WUSH transformation is memory-bound. Accordingly, the proposed kernel formulates the operation as a GEMM-equivalent computation, corresponding to C independent dense transforms. Each thread block processes a TileM×G subproblem, where TileM denotes the tile size along the batch dimension. Crucially, each block only needs to load a single (G,G) matrix to produce its output, matching the Hadamard case (H + Quant) in MR-GPTQ, maximizing effective bandwidth utilization.

The kernel is implemented using a CUTLASS GEMM template. The outer dimension M of the activation tensor (with shape (M,K)) and the outer dimension G of the WUSH matrix (with shape (G,G,C)) are mapped to the M and N dimensions of the GEMM, respectively. The internal GEMM dimension K’ is fixed to G, rather than G×C. The index over C is instead handled implicitly as an offset that each thread block applies based on the specific subproblem it is assigned to.

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

Speedupvs.BF16(M=1024)

Qwen3-8B

N=5120 K=5120

N=34816 K=5120

N=5120 K=17408

Qwen3-14B

N=8192 K=8192

N=57344 K=8192

N=8192 K=28672

Llama-3.1-70B

None + None + FP4-MatMul

| |
|---|

H + Quant + FP4-MatMul

| |
|---|

WUSH + Quant + FP4-MatMul

Figure 2. Per-layer MXFP4 (group size G=32) inference speedups relative to BF16 at batch size M=1024 for Qwen3-

- 8B, Qwen3-14B, and Llama-3.1-70B. We compare FP4-MatMul kernel without transform or quantization (None + None), with fused Hadamard and quantization kernel (H + Quant), and with fused WUSH and quantization kernel (WUSH + Quant). For block size G and a layer with K=C×G input and N output channels, the Hadamard matrix size is (G,G), while the WUSH size is (G,G,C). For H + Quant, the QuTLASS implementation is used (Egiazarian et al., 2026), which also fuses the transform with the quantization step.

N=4096 K=4096

N=24576 K=4096

N=4096 K=12288

Performance. Figure 2 reports per-layer inference speedups on RTX 5090 at batch size 1024 for Qwen3-8B, Qwen314B, and Llama-3.1-70B, normalized to BF16. Across all models and layer shapes, the MXFP4 matrix multiplication kernel (FP4-MatMul) achieves speedups of up to 6.6× over BF16, compared to the theoretical peak of 8× on this GPU. When including the fused transform and quantization kernel, WUSH + Quant + FP4-MatMul achieves speedups of up to 5.8× over BF16, and it closely matches the performance of the optimized H + Quant + FP4-MatMul baseline, with an average throughput difference of only 1.3%. Remarkably, in some configurations, WUSH + Quant + FP4-MatMul achieves identical performance to H + Quant + FP4-MatMul.

This negligible difference arises from loading C transform matrices instead of a single one; however, it is almost entirely mitigated by the kernel design and effective reuse via the L1 and L2 caches. Pure-quantization without an online transform (None + Quant + FP4-MatMul) would remove the transform cost but still incur the same activation quantization overhead, and therefore its performance is expected to lie between None + None + FP4-MatMul and H + Quant + FP4-MatMul. These results demonstrate that WUSH introduces negligible additional costs.

### 6. Experiments

Setup. We now present experiments evaluating the layerwise losses and the accuracy of quantized models using the WUSH transform, relative to SmoothQuant (Xiao et al., 2023), QuaRot (Ashkboos et al., 2024b), SpinQuant (Liu et al., 2025), and MR-GPTQ (Egiazarian et al., 2026) baselines. MR-GPTQ is the current state-of-the-art for MXFP/NVFP quantization. Beyond comparisons to prior methods, these experiments also include ablations over both the transform design and the quantization procedure. Specifically, we compare progressively richer transform variants to isolate the effects of orthogonal mixing, the Hadamard backbone, and the adaptive non-orthogonal component, while also evaluating both RTN and GPTQ quantization. We conduct experiments on the Llama-3.2-3B-Instruct, Llama3.1-8B-Instruct, and Qwen3-8B/14B/32B models. We use Platinum Benchmarks (Vendrow et al., 2025) and the LM Evaluation Harness (Gao et al., 2021) for accuracy evaluation.

Offline preprocessing costs. We report empirical offline preprocessing costs (Table 6) and storage overheads (Table 7) in Appendix Section B.2.

#### 6.1. Superior Layerwise Quantization Loss

We first report the layerwise weight-activation RTN quantization loss (L2 loss normalized by the number of elements) for each linear layer (attention projection layers Q, K, V, O and MLP projection layers Gate (G), Up (U), Down (D)) in the 18th transformer block of Qwen3-8B, using 32 calibration samples of sequence length 2048 from the FineWebEdu dataset (Penedo et al., 2024). We found the results to be consistent across blocks and datasets. We compare the WUSH transform with the identity (I), random rotation (R) averaged over 10 runs, Hadamard (H), and WUSH without Hadamard (WUS). Table 1 summarizes the losses for MXFP4, NVFP4, and INT4 formats. Following the format definitions, MXFP4 uses group size 32, and NVFP4 uses group size 16. We use INT4 to denote 4-bit integers with Gaussian MSE clipping and BF16 scales of group size 32. The transform block size is matched to the quantization group size. Across formats, WUSH substantially reduces

- Table 1. Layerwise RTN quantization loss in the unit of 10−3.

Q K V O G U D

MXFP4

I 11.1 12.0 10.7 4.35 7.10 6.56 5.47 R 7.61 7.73 9.14 3.84 5.56 5.68 4.07 H 7.24 7.20 8.60 3.79 5.45 5.61 3.90

WUS 6.27 7.22 4.05 3.57 5.76 4.75 4.46 WUSH 3.34 3.34 3.30 2.76 4.49 4.39 3.39

NVFP4

I 4.23 4.35 4.37 2.34 3.49 3.41 2.41 R 4.98 5.01 5.86 2.54 3.63 3.68 2.66 H 5.60 5.62 6.70 2.58 3.71 3.79 2.78

WUS 2.26 2.36 2.30 2.00 3.04 3.01 2.23 WUSH 2.40 2.44 2.28 1.92 3.09 3.02 2.33

INT4

I 170. 123. 13.2 4.55 55.9 9.83 19.3 R 5.79 5.84 6.96 2.94 4.22 4.29 3.10 H 5.57 5.55 6.80 2.86 4.09 4.25 3.03

WUS 213. 142. 10.7 4.54 50.2 7.42 13.1 WUSH 2.39 2.43 2.54 2.10 3.43 3.43 2.55

the loss: WUSH consistently yields the smallest loss for MXFP4 and INT4, while WUSH and WUS are almost equal for NVFP4. The errors for NVFP4 tend to be lower due to its smaller group size.

Although the theoretical derivation uses a stochastic unbiased quantization model as a tractable surrogate, the resulting transform is applied with deterministic RTN in these layerwise experiments. The trends in Table 1 support this surrogate, suggesting that the assumptions (Section 4.1) and modeling (Sections 4.2.1 and 4.3.1) capture the dominant sources of error for FP and INT block quantizers. At the same time, MXFP4 behaves as a hybrid between ideal FP and INT quantization: the effective mantissa step changes uniformly, imparting INT-like behavior, especially in the subnormal regime and small exponent ranges. This explains why Hadamard transforms provide measurable gains for MXFP4, despite the ideal FP theory (Section 4.2.3) predicting no benefit for purely orthogonal transforms. The poor INT4 performance of WUS further highlights the role of the Hadamard component: without it, the non-orthogonal component can amplify individual coordinates, increasing the AbsMax group scale and therefore the INT quantization error. For NVFP4, the Hadamard alone is harmful due to the top-element preservation effect of this format (Egiazarian et al., 2026). Remarkably, other components in WUSH overcome this effect and produce smaller losses than the identity transform.

6.2. End-to-End LLM Accuracy Benchmarks

- Table 2 compares WUSH relative to prior works for Llama3.1-8B-Instruct on LM Evaluation Harness metrics, in the same setup as Egiazarian et al. (2026). For methods with block transforms (i.e., not applicable to SmoothQuant, QuaRot, and SpinQuant), the transform block size is al-

Table 2. Llama-3.1-8B-Instruct W4A4 accuracy results on the LM Evaluation Harness under different methods.

Format Method MMLU-CoT GSM8K HellaSwag WinoGrande Average Recovery BF16 - 72.76 85.06 80.01 77.90 78.93 100.0

RTN-I 68.26 78.39 78.15 74.11 74.73 94.67 RTN-H 67.41 78.01 77.31 73.48 74.05 93.82

SmoothQuant 68.90 79.50 79.50 74.70 75.70 95.90 QuaRot 66.50 77.40 77.25 75.14 74.10 93.80 SpinQuant 66.50 76.10 76.96 75.32 73.70 93.40 GPTQ-I 68.85 81.25 78.26 74.51 75.72 95.92 GPTQ-H (MR-GPTQ) 69.12 80.80 78.17 75.24 75.84 96.08 RTN-WUSH 68.83 78.57 78.22 75.47 75.28 95.37 GPTQ-WUSH 69.69 80.11 78.52 76.09 76.10 96.40

NVFP4

RTN-I 62.21 67.85 73.99 73.24 69.32 87.83 RTN-H 62.38 72.48 75.29 71.67 70.45 89.26

SmoothQuant 63.93 68.54 75.10 73.56 70.30 89.06 QuaRot 49.86 56.94 73.50 71.43 62.90 79.70 SpinQuant 61.80 68.16 74.87 72.93 69.40 88.00 GPTQ-I 63.49 68.46 76.01 74.51 70.62 89.47 GPTQ-H (MR-GPTQ) 67.19 75.70 76.91 74.80 73.65 93.31 RTN-WUSH 66.85 75.16 77.28 73.56 73.21 92.75 GPTQ-WUSH 67.79 77.41 77.44 74.78 74.35 94.20

MXFP4

Qwen3-8B Platinum Benchmarks

BF16 NVFP4-GPTQ-H

- 99.33 100.0 100.0 97.36 97.39 95.88 97.50 93.68 91.50 97.79 93.79 93.30 82.56 95.39
- 100.0 99.40 99.06 97.21 96.04 93.56 99.00 85.68 93.80 97.02 92.42 88.80 80.00 94.00 99.87 99.80 99.41 96.91 96.12 93.18 99.50 88.53 92.50 96.13 91.93 90.05 76.72 93.90 99.60 99.40 99.06 97.36 97.01 93.86 99.10 88.00 92.60 97.02 92.30 88.90 76.41 93.89

NVFP4-RTN-WUSH NVFP4-GPTQ-WUSH

NVFP4-RTN-I MXFP4-GPTQ-WUSH

- 99.83 99.75 99.26 97.17 96.46 91.85 99.00 89.34 93.25 96.69 92.70 84.57 77.05 93.61
- 100.0 99.60 99.06 96.38 96.57 92.66 98.90 87.05 92.60 96.35 92.92 88.04 75.69 93.52 99.20 99.20 98.94 96.83 95.75 92.13 99.10 87.89 90.20 96.24 90.56 85.84 77.13 93.00 99.60 99.80 99.29 96.45 94.48 92.36 98.80 86.11 91.90 96.13 89.57 86.70 75.18 92.80 99.56 99.67 98.04 96.10 92.91 89.76 98.17 85.44 93.50 95.03 88.41 83.73 72.48 91.75

MXFP4-RTN-WUSH MXFP4-GPTQ-H MXFP4-RTN-H

SingleOpSingleQMultiArithSVAMPGSM8KMMLU-MathBBHDeductionBBHCountingBBHNavigateHotpotQASQuAD DROPWinograd-WSCAverage

Qwen3-8B Platinum Benchmarks Average Accuracy

96.0

BF16

95.0

Accuracy[%]

94.0

93.0

92.0

91.0

90.0

MXFP4-RTN-H MXFP4-GPTQ-HMXFP4-RTN-WUSHMXFP4-GPTQ-WUSH NVFP4-RTN-INVFP4-GPTQ-WUSHNVFP4-RTN-WUSH NVFP4-GPTQ-H

- Figure 3. Comparison of different transforms on Qwen3-8B for both NVFP4 and MXFP4 quantization on Platinum Benchmarks. The top table shows accuracy results across the individual benchmark tasks, while the bottom plot shows the average accuracy scores together with their standard deviations for each transform.

ways the same as the quantization group size. Overall, WUSH generally yields the best results, by a large margin for MXFP4. On Qwen3-8B measured on Platinum Benchmarks (Figure 3), WUSH improves MXFP4 by up to +1.3 (RTN) and +0.7 (GPTQ) average points, while NVFP4 improvements are smaller and often within run-to-run variability. Additional accuracy results are in Appendix Section B.1. Table 3 shows the LM Evaluation Harness results for different models, e.g., for Qwen3-14B WUSH reduces the NVFP4-MXFP4 average gap to 0.36 points, and above 98% recovery. Figures 7 to 10 also reflect similar trends. Table 4 shows WUSH reduces the KL divergence. Table 5 shows WUSH is stable with respect to the choice of calibration dataset. We conclude that WUSH improves the state-of-the-art for weight-activation quantization.

### 7. Conclusion

We have studied weight-activation quantization for LLMs in the presence of heavy-tailed outliers and have shown that this problem admits closed-form linear transforms for both FP and INT block quantizers. Our construction, WUSH, combines a fixed Hadamard backbone with a data-aware component derived from second-order statistics, yielding a non-orthogonal blockwise transform that is provably nearoptimal and remains amenable to efficient GPU kernel implementations. These results clarify the empirical success of Hadamard rotations and indicate that principled, data-aware block transforms can yield significant improvements, e.g., making the MXFP format competitive with NVFP.

### Acknowledgements

We thank Ileana Rugina for useful discussions. We thank Tijmen Blankevoort for useful discussions and for suggesting the name WUSH, which is a clear improvement over our previous name (HSUW). This research was funded in part by the Austrian Science Fund (FWF) 10.55776/COE12, and partially supported by a generous grant from NVIDIA.

### Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Ashkboos, S., Markov, I., Frantar, E., Zhong, T., Wang, X., Ren, J., Hoefler, T., and Alistarh, D. QUIK: Towards end-to-end 4-bit inference on generative large language models. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 3355–3371, Miami, Florida, USA, November 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.197. URL https://ac lanthology.org/2024.emnlp-main.197/.

Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Cameron, P., Jaggi, M., Alistarh, D., Hoefler, T., and Hensman, J. QuaRot: Outlier-free 4-bit inference in rotated LLMs. In Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., and Zhang, C. (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 100213–100240. Curran Associates, Inc., 2024b. doi: 10.52202/079017-3180. URL https:

//proceedings.neurips.cc/paper_files /paper/2024/file/b5b939436789f76f08b 9d0da5e81af7c-Paper-Conference.pdf.

Chee, J., Cai, Y., Kuleshov, V., and De Sa, C. M. QuIP: 2-bit quantization of large language models with guarantees. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 4396–4429. Curran Associates, Inc., 2023. URL https://proceeding s.neurips.cc/paper_files/paper/2023/ file/0df38cd13520747e1e64e5b123a78ef 8-Paper-Conference.pdf.

Chen, J., Shabanzadeh, Y., Crnˇcevi´c, E., Hoefler, T., and Alistarh, D. The geometry of LLM quantization: GPTQ as Babai’s nearest plane algorithm. In The Fourteenth International Conference on Learning Representations,

2026. URL https://openreview.net/forum ?id=NFB4QGGS65.

Défossez, A., Adi, Y., and Synnaeve, G. Differentiable model compression via pseudo quantization noise. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/f orum?id=DijnKziche.

Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. GPT3.int8(): 8-bit matrix multiplication for transformers at scale. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 30318– 30332. Curran Associates, Inc., 2022. URL https:

//proceedings.neurips.cc/paper_files /paper/2022/file/c3ba4962c05c49636d4 c6206a97e9c8a-Paper-Conference.pdf.

Dettmers, T., Svirschevski, R. A., Egiazarian, V., Kuznedelev, D., Frantar, E., Ashkboos, S., Borzunov, A., Hoefler, T., and Alistarh, D. SpQR: A sparsequantized representation for near-lossless LLM weight compression. In The Twelfth International Conference on Learning Representations, 2024. URL https:

//openreview.net/forum?id=Q1u25ahSuy.

Egiazarian, V., Castro, R. L., Kuznedelev, D., Panferov, A., Kurtic, E., Pandit, S., Marques, A. N., Kurtz, M., Ashkboos, S., Hoefler, T., and Alistarh, D. Bridging the gap between promise and performance for microscaling FP4 quantization. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=zCBGe9AqJZ.

Frantar, E. and Alistarh, D. Optimal brain compression: A framework for accurate post-training quantization and pruning. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 4475–4488. Curran Associates, Inc., 2022. URL https:

//proceedings.neurips.cc/paper_files /paper/2022/file/1caf09c9f4e6b0150b0 6a07e77f2710c-Paper-Conference.pdf.

Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. OPTQ: Accurate quantization for generative pre-trained transformers. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=tcbBPnfwxS.

Gao, L., Tow, J., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., McDonell, K., Muennighoff, N., Phang, J., Reynolds, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for few-shot language model evaluation, September 2021. URL ht tps://doi.org/10.5281/zenodo.5371629.

Jegou, H., Douze, M., and Schmid, C. Hamming embedding and weak geometric consistency for large scale image search. In Forsyth, D., Torr, P., and Zisserman, A. (eds.), Computer Vision – ECCV 2008, pp. 304–317, Berlin, Heidelberg, 2008. Springer Berlin Heidelberg. ISBN 9783-540-88682-2. URL https://link.springer.

com/chapter/10.1007/978-3-540-88682-2 _24.

Jégou, H., Douze, M., and Schmid, C. Product quantization for nearest neighbor search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1):117–128, 2011. doi: 10.1109/TPAMI.2010.57. URL https:// ieeexplore.ieee.org/document/5432202.

Kurtic, E., Marques, A. N., Pandit, S., Kurtz, M., and Alistarh, D. “Give me BF16 or give me death”? accuracyperformance trade-offs in LLM quantization. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 26872–26886, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 9798-89176-251-0. doi: 10.18653/v1/2025.acl-long.1304. URL https://aclanthology.org/2025.ac l-long.1304/.

Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., and Han, S. AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration. In Gibbons, P., Pekhimenko, G., and Sa, C. D. (eds.), Proceedings of Machine Learning and Systems, volume 6, pp. 87–100, 2024. URL https://proceedings.mlsys.org/paper_ files/paper/2024/file/42a452cbafa9dd 64e9ba4aa95cc1ef21-Paper-Conference. pdf.

Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D., Krishnamoorthi, R., Chandra, V., Tian, Y., and Blankevoort, T. SpinQuant: LLM quantization with learned rotations. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview .net/forum?id=ogO6DGE6FZ.

MX Alliance, O. C. P. F. OCP microscaling formats (MX) specification version 1.0. Technical specification, Open Compute Project Foundation, September 2023. URL https://www.opencompute.org/document s/ocp-microscaling-formats-mx-v1-0-s pec-final-pdf.

NVIDIA. NVIDIA Blackwell architecture technical brief. Technical report, NVIDIA, 2024. URL https://re sources.nvidia.com/en-us-blackwell-a rchitecture.

Penedo, G., Kydlíˇcek, H., allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C., Von Werra, L., and Wolf, T. The FineWeb datasets: Decanting the web for the finest text data at scale. In Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., and Zhang, C. (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 30811–30849. Curran Associates, Inc., 2024. doi: 10.52202/079017-0970. URL https://proceedings.neurips.cc/paper _files/paper/2024/file/370df50ccfdf8 bde18f8f9c2d9151bda-Paper-Datasets_an d_Benchmarks_Track.pdf.

Shao, Y., Wang, P., Chen, Y., Xu, C., Wei, Z., and Cheng, J. Block rotation is all you need for MXFP4 quantization, 2025. URL https://arxiv.org/abs/2511.0 4214.

Sun, Y., Liu, R., Bai, H., Bao, H., Zhao, K., Li, Y., Hu, J., Yu, X., Hou, L., Yuan, C., Jiang, X., Liu, W., and Yao, J. FlatQuant: Flatness matters for LLM quantization. In Singh, A., Fazel, M., Hsu, D., Lacoste-Julien, S., Berkenkamp, F., Maharaj, T., Wagstaff, K., and Zhu, J. (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 57587–57613. PMLR, 13–19 Jul 2025. URL https://proceedings.ml r.press/v267/sun25l.html.

Tseng, A., Chee, J., Sun, Q., Kuleshov, V., and De Sa, C. QuIP#: Even better LLM quantization with Hadamard incoherence and lattice codebooks. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 48630–48656. PMLR, 21–27 Jul 2024a. URL https: //proceedings.mlr.press/v235/tseng24a. html.

Tseng, A., Sun, Q., Hou, D., and De, C. QTIP: Quantization with trellises and incoherence processing. In Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., and Zhang, C. (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 59597–59620. Curran Associates, Inc., 2024b. doi: 10.52202/079017-1904. URL https://proceedi ngs.neurips.cc/paper_files/paper/202 4/file/6de2e84b8da47bb2eb5e2ac96c63d 2b0-Paper-Conference.pdf.

Vendrow, J., Vendrow, E., Beery, S., and Madry, A. Do large language model benchmarks test reliability?, 2025. URL https://arxiv.org/abs/2502.03461.

Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. SmoothQuant: Accurate and efficient post-training

quantization for large language models. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 38087– 38099. PMLR, 23–29 Jul 2023. URL https://pr oceedings.mlr.press/v202/xiao23c.html.

### A. Detailed Theoretical Results

#### A.1. Second Moment Decompositions

- Lemma A.1 (Invariance of the Decomposition Choices). For any full rank W(′i),X(′i) that satisfy Eq. (5), the matrix products W(′i)U(i) and X(′i)V(i) in Eq. (7) are invariant up to sign and permutation changes. Proof. We omit the subscript (i) and superscript ′ to simplify notation. Let W, X ∈ Rd×d such that W W⊤ = WW⊤ and X X⊤ = XX⊤. Define QW = W−1 W and QX = X−1 X. Then,

QWQ⊤W = W−1 W W⊤W−⊤ = W−1(WW⊤)W−⊤ = I, (33)

and similarly QXQ⊤X = I. Hence QW,QX are orthogonal and W = WQW, X = XQX. By Eq. (6), we have U,S,V ∈ Rd×d as the SVD of W⊤X = USV ⊤. Therefore,

W⊤ X = Q⊤WW⊤XQX = Q⊤W USV ⊤ QX = Q⊤WU S Q⊤XV ⊤ . (34)

Thus, we may take U = Q⊤WU, S = S, V = Q⊤XV (up to the sign and permutation conventions) as the SVD of W⊤ X = U S V ⊤. With this,

- W U = WQWQ⊤WU = WU,
- X V = XQXQ⊤XV = XV .

(35)

| |
|---|

#### A.2. Multivariate vs. Univariate Probabilistic Modeling

A common intuition in prior transform-based quantization methods is largely univariate: apply an orthogonal mixing (Hadamard or learned rotation) so that the distribution of coordinate-wise values becomes more “Gaussian-like,” thereby reducing the impact of heavy-tailed outliers. In contrast, our probabilistic modeling is explicitly multivariate: we model a quantization group as a joint random vector and reason about its second-order structure, enabling analysis in the eigenbasis and construction of transforms that depend on the spectrum (eigenvalues / singular values).

Figure 4 illustrates the key limitation of purely orthogonal “Gaussianization.” In the top row (original distributions), the joint density is anisotropic: the two coordinate-wise marginals differ in spread, and their average can appear heavy-tailed or non-Gaussian depending on the axis alignment. In the bottom row, after the Hadamard (45◦ rotation) transform, the coordinate marginals become nearly identical, and the average marginal appears more Gaussian-like. However, this does not imply that outliers have been removed in the underlying multivariate geometry: orthogonal transforms preserve global energy and, crucially, preserve the eigenvalues of the covariance (they only rotate eigenvectors). Consequently, points that are extreme in the principal directions remain extreme in the joint space, even if they are less visually apparent in any single marginal.

This distinction is central to group quantization. Quantization error is driven by joint properties (e.g., how mass concentrates along dominant directions and how extreme points interact with groupwise scaling), which cannot be certified by marginal statistics alone. Our multivariate formulation therefore targets the spectrum directly: by exposing and using the eigen/singular-value structure of the group statistics, we can design transforms that genuinely reduce the effective anisotropy responsible for outlier-dominated scaling, rather than merely redistributing it across coordinates via rotations.

#### A.3. Equalization Effect

- Figure 5 provides a layerwise visualization of the equalization effect induced by the WUSH construction. Using Eqs. (5), (7), for each block, the second moment of the activation after the WUSH transform satisfies

d−batch1 Twush(i)X(i)X(⊤i)Twush⊤(i) = Twush(i)X(′i)X(′⊤i)Twush⊤(i) = HS(i)H⊤. (36)

Since the normalized Hadamard matrix has entries ±d−21 , the diagonal of the transformed second moment is a constant of d−1 tr S(i) . The analogous property also holds for the transformed weight second moment, so WUSH equalizes the

Joint Distribution (Original)

Marginal Distribution (Original)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

p

p

y

x

Value

Joint Distribution (Rotated)

Marginal Distribution (Rotated)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

p

p

y

x

Value

p(x, y)

- p(x)

- p(y)

p(x), p(y) Mean

- Figure 4. Rotation can equalize marginals without alleviating multivariate outliers. Top: an anisotropic 2D distribution with unequal coordinate marginals. Bottom: after an orthogonal rotation, the two marginals become nearly identical, and their average appears more Gaussian-like. Nevertheless, the joint distribution retains the same eigenvalues (spectrum) under rotation, so extreme points remain extreme in the multivariate space; they are merely “hidden” from any single marginal view. This motivates multivariate, spectrum-aware transforms that act on the joint geometry rather than relying on marginal “Gaussianization.”

0

200

400 InputChannel

0

100

200

Output Channel

0.0

0.1

Absolute Value

[Figure 1]

Weight - I

0

200

400 InputChannel

0

100

200 Token

0

5

Absolute Value

[Figure 2]

Activation - I

0

200

400 InputChannel

0

100

200

Output Channel

0.0

0.1

Absolute Value

[Figure 3]

Weight - R

0

200

400 InputChannel

0

100

200 Token

0

2

Absolute Value

[Figure 4]

Activation - R

0

200

400 InputChannel

0

100

200

Output Channel

0.0

0.1

Absolute Value

[Figure 5]

Weight - H

0

200

400 InputChannel

0

100

200 Token

- 0

- 1

- 2

Absolute Value

[Figure 6]

Activation - H

0

200

400 InputChannel

0

100

200

Output Channel

- 0

- 1

Absolute Value

[Figure 7]

Weight - WUS

0

200

400 InputChannel

0

100

200 Token

0.0

0.5

Absolute Value

[Figure 8]

Activation - WUS

0

200

400 InputChannel

0

100

200

Output Channel

0.0

0.2

0.4

Absolute Value

[Figure 9]

Weight - WUSH

0

200

400 InputChannel

0

100

200 Token

0.0

0.2

0.4

Absolute Value

[Figure 10]

Activation - WUSH

0

200

400 InputChannel

0

100

200

Output Channel

0.0

0.5

Absolute Value

[Figure 11]

Weight - SmoothQuant

0

200

400 InputChannel

0

100

200 Token

0.0

0.5

Absolute Value

[Figure 12]

Activation - SmoothQuant

Magnitude RMS (Root Mean Square)

- Figure 5. Visualization of blockwise RMS equalization induced by WUSH. Each panel uses its own vertical scale for visibility. We visualize absolute values of a representative weight slice (top row; input channel vs. output channel) and calibration activation slice (bottom row; input channel vs. token) after applying identity (I), random rotation (R), Hadamard (H), WUS (WUSH without Hadamard), WUSH, and SmoothQuant transforms. The surfaces show entrywise magnitudes, while the red curves show the per-input-channel RMS across output channels or tokens. Identity leaves the original outliers unchanged. Random rotation and Hadamard spread outliers within each weight or activation block, but as orthogonal transforms, they cannot transfer scale between weights and activations. Hadamard nevertheless equalizes RMS within a block more effectively than a random rotation. SmoothQuant transfers scale between activations and weights through diagonal channelwise rescaling, but can unevenly amplify some channels. WUS applies the adaptive non-orthogonal component but removes the Hadamard factor, leaving the transformed second moment aligned with S and the per-channel RMS unequal. In contrast, WUSH maps each blockwise second moment to HSH⊤, whose diagonal is constant because H is a normalized Hadamard matrix. Therefore, WUSH balances the weight and activation scales and produces the flattest RMS profiles.

per-channel RMS (root mean square) within the corresponding weight and activation blocks. This distinguishes WUSH from orthogonal transforms, which can spread outliers but cannot transfer scale between weights and activations, and from SmoothQuant, which transfers scale only through diagonal channelwise rescaling. Consistent with this, Figure 5 shows that WUSH balances the weight and activation scales and produces the flattest RMS profiles.

- A.4. Equation Derivations Standard basis vector notation. Denote the k-th standard basis vector as ek ∈ {0,1}d, where the k-th element is 1 and 0 otherwise. Basic second-order expectations.

Eyyy⊤ = W′⊤ Exxx⊤ W′ = W′⊤X′X′⊤W′ = USV ⊤V SU⊤ = US2U⊤, (Eqs. (6), (11)) EyTy (Ty)⊤ = T Eyyy⊤ T⊤ = U′S′R⊤S−1U⊤US2U⊤US−1RS′U′⊤ = U′S′2U′⊤, (Eq. (16)) Ey ∥y∥2 = tr Eyyy⊤ = tr US2U⊤ = tr S2 ,

Ey ∥Ty∥2 = tr Ty (Ty)⊤ = tr U′S′2U′⊤ = tr S′2 .

(37)

#### Eq. (13).

2

Ww),ε(TXx) (TWw + ε(TWw))⊤ (TXx + ε(TXx)) − w⊤x

ℓ = Ew,x,ε(T

Ww),ε(TXx) x⊤TX⊤ε(TWw) + w⊤TW⊤ε(TXx) + ε(TWw)⊤ ε(TXx) ≈ 0 (first order approx.)

+w⊤TW⊤TXx − w⊤x

= Ew,x,ε(T

= 0 (TW = TX−⊤)

Ww),ε(TXx) x⊤TX⊤ε(TWw) + w⊤TW⊤ε(TXx) 2

≈ Ew,x,ε(T

Ww),ε(TXx) x⊤TX⊤ε(TWw) 2 + w⊤TW⊤ε(TXx) 2 + x⊤TX⊤ ε(TWw)

w⊤TW⊤ ε(TXx)

= Ew,x,ε(T

E (·) = 0

E (·) = 0

Ww),ε(TXx) x⊤TX⊤ε(TWw) 2 + w⊤TW⊤ε(TXx) 2

= Ew,x,ε(T

Ww)ε(TWw)⊤ TX xx⊤

Xx)ε(TXx)⊤ TW ww⊤

TX⊤ε(TWw) + Ew,x,ε(T

TW⊤ε(TXx)

= Ew,x,ε(T

Exxx⊤ = X′X′⊤

Ewww⊤ = W ′W ′⊤

Ww)ε(TWw)⊤ TXX′X′⊤TX⊤ε(TWw) + Ex,ε(T

Xx)ε(TXx)⊤ TWW′W′⊤TW⊤ε(TXx)

= Ew,ε(T

2

ε(TWw) 2 + Ex,ε(T

ε(TXx) 2

Ww) X′⊤ TX⊤

Xx) W′⊤ TW⊤

= Ew,ε(T

= TW−1

= TX−1

Ww) X′⊤TW−1ε(TWw) 2 + Ex,ε(T

Xx) W′⊤TX−1ε(TXx) 2 .

= Ew,ε(T

(38)

#### Eq. (17).

(tr(S))2 ≥ tr S2 (S ≻ 0) tr S2 ≥ d−1 (tr(S))2 (Jensen’s inequality) ⇒ tr S2 ≥ d−1 (tr(S))2 ≥ d−1 tr S2 . (39)

#### Eq. (19).

Ey,ε(T y) T−1ε(Ty) 2

= Ey,η T−1 diag (η)Ty 2 (Eq. (18))

= Ey,ηy⊤T⊤ diag (η)T−⊤T−1 diag (η)Ty

= Ey,η tr y⊤T⊤ diag (η)T−⊤T−1 diag (η)Ty = Ey,η tr T−1 diag (η)Tyy⊤T⊤ diag (η)T−⊤ = tr T−1 Eη diag (η)T Eyyy⊤ T⊤ diag (η) T−⊤

= tr T−1 Eη diag (η)U′S′2U′⊤ diag (η) T−⊤ (Eq. (37))

= tr T−1 Eηη2 U′S′2U′⊤ ⊙ I T−⊤

= Eηη2 tr T−1 U′S′2U′⊤ ⊙ I T−⊤ .

(40)

#### Eq. (20).

tr T−1 U′S′2U′⊤ ⊙ I T−⊤

= tr T−1 diag S′U′⊤e1 2 ,..., S′U′⊤ed 2 T−⊤

d

d

ejT−1ek 2 S′U′⊤ek 2

=

j=1

k=1

d

T−1ek 2 S′U′⊤ek 2

=

k=1

d

#### USRS′−1U′⊤ek 2 S′U′⊤ek 2 (Eq. (16))

=

k=1

d

SRS′−1U′⊤ek 2 RS′U′⊤ek 2 (rotation-invariance of L2 norm)

=

k=1

d

e⊤k U′S′−1R⊤SRS′U′⊤ek 2 (Cauchy-Schwarz)

≥

k=1

d

#### e⊤k U′S′−1R⊤SRS′U′⊤ek 2

= d d−1

k=1

2

- d

k=1

- e⊤k U′S′−1R⊤SRS′U′⊤ek

≥ d d−1

(Jensen’s inequality)

= d−1 tr U′S′−1R⊤SRS′U′⊤ 2

= d−1 (tr(S))2 .

(41)

The equality can be attained by choosing U′ = H, S′ = S 12 , and R = I such that

tr T−1 U′S′2U′⊤ ⊙ I T−⊤

d

#### SRS′−1U′⊤ek 2 RS′U′⊤ek 2 (Eq. (41))

=

k=1

d

2 2

- 1

- 2 H⊤ek

(plug in)

=

S

k=1

 

 

2

d

d

sj e⊤j H⊤ek 2

=

j=1

k=1

 

 

2

d

d

sjd−1

=

j=1

k=1

 d−1

 

2

d

sj

= d

j=1

= d−1 (tr(S))2 .

(42)

#### Eq. (21).

Ey,ε(T y) T−1ε(Ty) 2

= Ey,η T−1 diag (η)Ty 2 (Eq. (18))

= Ey,ηy⊤T⊤diag (η)T−⊤T−1diag (η)Ty

= Ey,ηy⊤T⊤diag (η)2 Ty (orthogonality of T)

= Eyy⊤T⊤ Eηdiag (η)2 Ty

= Eηη2 Eyy⊤T⊤Ty

= Eηη2 Eyy⊤y (orthogonality of T)

= Eηη2 Ey ∥y∥2

= Eηη2 tr S2 (Eq. (37)).

###### Eq. (22). In the case of U′ = I:

tr T−1 U′S′2U′⊤ ⊙ I T−⊤

= tr USRS′−1U′⊤ U′S′2U′⊤ ⊙ I U′S′−1R⊤SU⊤ (Eq. (16)) = tr USRS′−1 S′2 ⊙ I S′−1R⊤SU⊤ (plug in) = tr USRS′−1S′2S′−1R⊤SU⊤

= tr USRR⊤SU⊤

= tr US2U⊤

= tr S2 .

(43)

(44)

In the case of S′ = I:

tr T−1 U′S′2U′⊤ ⊙ I T−⊤

#### = tr USRS′−1U′⊤ U′S′2U′⊤ ⊙ I U′S′−1R⊤SU⊤ (Eq. (16)) = tr USRU′⊤ U′U′⊤ ⊙ I U′R⊤SU⊤ (plug in) = tr USRU′⊤U′R⊤SU⊤

= tr US2U⊤

= tr S2 .

(45)

#### Eq. (24). Ey,ε(T y) T−1ε(Ty) 2

=Ey,η T−1 ∥Ty∥∞ η 2 (Eq. (23))

=Ey,η ∥Ty∥2∞ η⊤T−⊤T−1η

=Ey,η ∥Ty∥2∞ tr η⊤T−⊤T−1η =Ey,η ∥Ty∥2∞ tr T−1ηη⊤T−⊤ = tr T−1 Eηηη⊤ T−⊤ Ey ∥Ty∥2∞

= Eηη2 tr T−1T−⊤ Ey ∥Ty∥2∞

= Eηη2 T−1 2F Ey ∥Ty∥2∞ .

#### Eq. (25).

T−1 2F = tr T−1T−⊤

= tr USRS′−1U′⊤U′S′−1R⊤SU⊤ (Eq. (16)

= tr SU⊤USRS′−1U′⊤U′S′−1R⊤

= tr S2RS′−2R⊤

(46)

(47)

Because s21 ≥ ··· ≥ s2d and s′−1 2 ≤ ··· ≤ s′−d 2, using von Neumann’s trace inequality,

T−1 2F = tr S2RS′−2R⊤ ≥ tr S2S′−2 . (48) The equality can be attained by choosing R = I.

###### Eq. (26). Express the expectations as

e⊤k Ty 2 ≥ max

Ey e⊤k Ty 2 (Jensen’s inequality) (49) and

Ey ∥Ty∥2∞ = Ey max

k

k

d

d

Ey e⊤k Ty 2 . (50) By the relationships among the mean, maximum, and summation values, we have:

e⊤k Ty 2 =

Ey ∥Ty∥2 = Ey

k=1

k=1

Ey e⊤k Ty 2 , (51) where, by Eq. (37),

Ey e⊤k Ty 2 ≤ Ey ∥Ty∥2∞ ≤ Ey ∥Ty∥2 ≤ dmax

d−1Ey ∥Ty∥2 ≤ max

k

k

Ey ∥Ty∥2 = tr S′2 (52) and

Ey e⊤k Ty 2 = e⊤k T Eyyy⊤ T⊤ek = e⊤k U′S′2U′⊤ek = S′U′⊤ek 2 . (53)

- Lemma A.2 (Maximum Inequalities for Tail-Bounded Distributions). If X1,...,Xd ∈ R are zero-mean Gaussian random variables,

EXk2. (54) If X1,...,Xd ∈ R are zero-mean Laplacian random variables,

Xk2 ≤ min{d,(2ln(2d) + 2)}max

Emax

k

k

Xk2 ≤ 2−1 (lnd)2 + lnd + 1 max

EXk2. (55) Both inequalities hold without the need for independence.

Emax

k

k

Proof. We provide proofs for the Gaussian and Laplacian distributions in Sections A.6.1 and A.6.2, respectively.

| |
|---|

- Using Lemma A.2, for a tail-bounded Dy that is a zero-mean multivariate Gaussian or Laplacian, we can further tighten the upper bound by

Ey e⊤k Ty 2 ,Ey ∥Ty∥2 . (56) When U′ = H, using Eqs. (37), (53), the marginal second moment in Eq. (56) becomes

Ey ∥Ty∥2∞ ≤ min do(1) max

k

Ey e⊤k Ty 2 = S′U′⊤ek 2 = S′H⊤ek 2 =

d

s′j2d−1 = d−1 tr S′2 = d−1Ey ∥Ty∥2 , (57)

j=1

which is equalized for all k, and maxk Ey e⊤k Ty 2 is minimized to d−1Ey ∥Ty∥2. Finally, combining Eqs. (51), (56), (57),

do(1)−1 tr S′2 , tail-bounded Dy, tr S′2 , otherwise.

d−1 tr S′2 ≤ Ey ∥Ty∥2∞ ≤

(58)

Eq. (31). For orthogonal T, we have T−1 2F = d and Ey ∥Ty∥2 = Ey ∥y∥2. Using Eq. (37), Ey ∥y∥2 = tr S2 . By Eq. (51),

tr S2 = d−1 T−1 2F Ey ∥Ty∥2 ≤ T−1 2F Ey ∥Ty∥2∞ ≤ T−1 2F Ey ∥Ty∥2 = dtr S2 . (59)

#### A.5. Quantization Error Modeling Justifications

- A.5.1. INTEGER (INT) TYPES We model the quantization error using pseudo noise similar to DiffQ (Défossez et al., 2022). Define the scalar quantizer of b-bit symmetric integer as

q (x) = xs−1 + 2−1 s, (60) where x ∈ R, scale s ∈ R̸=0, and xs−1 ∈ −2b−1,2b−1 . To simplify the analysis, we model the quantizer as

q (x) = xs−1 + ξ s = x + sξ (61) with random noise ξ ∼ Uniform −2−1,2−1 . Then, the quantization error can be expressed as

q (x) − x = sξ. (62)

For a vector α ∈ Rd, we use the AbsMax scale s = 21−b ∥α∥∞. Define

η = 21−bξ, (63) then we have sξ = η ∥α∥∞. The quantization error of vector α is

ε(α) = ∥α∥∞ η (64) with η = [η1,...,ηd]⊤ ∈ Rd being a random vector of i.i.d. samples η ∼ Dη and

Eηη = 21−bEξξ = 0 (65)

- A.5.2. FLOATING-POINT (FP) TYPES

The FP format EaMb is defined as a tuple (s,e,m) with a sign s ∈ {0,1}, an exponent e ∈ {0,...,2a − 1}, and a mantissa m ∈ 0,...,2b − 1 . Assume we do not need to consider the subnormal and special number cases. The tuple (s,e,m) represents the number

a−1+1 1 + 2−bm . (66)

xEaMb = (−1)s 2e−2

2.5

- y = 1 + x

- y = 2x

2.0

y

1.5

max difference ≈ 0.086

1.0

0.5

-0.5 0.0 0.5 1.0 1.5 x

Figure 6. The two curves y = 1 + x and y = 2x are close for 0 ≤ x ≤ 1.

Using the approximation 1 + (·) ≈ 2(·) for 2−bm ∈ [0,1] (visualized in Figure 6), we define a smoothed version of EaMb, named SEaMb, as

##### (2be+m)−2a−1+1 (67) with

−b

a−1+122

−bm = (−1)s 22

xSEaMb = (−1)s 2e−2

2be + m = 2b log2 |xSEaMb| + 2a−1 − 1 (68) being an (a + b)-bit unsigned integer (concatenating the binary representations of e and m). The minimum and maximum of |xSEaMb| are

a−1+1, max|xSEaMb| = 22

min|xSEaMb| = 2−2

(69)

−b

(2a+b−1)−2a−1+1 = 22

a−1−2−b+1,

respectively.

Denote SEaMb(x) as the quantizer that casts x ∈ [−max|xSEaMb|,max|xSEaMb|] to the nearest number in the SEaMb type. Assume e ∈ (−∞,2a − 1] so that the smaller values can be represented more precisely. SEaMb(x) can be approximated using integer rounding in logarithmic space and first-order expansion. For x ̸= 0,

⌊2b(log2|x|+2a−1−1)⌉−2a−1+1

−b

SEaMb(x) ≈ sgn(x)22

⌊2b log2|x|⌉ ≈ sgn(x) 22

−b

= sgn(x)22

(70)

−b2b log2|x| + 2b log2 |x| − 2b log2 |x| 22

−b2b log2|x|2−b ln2

= x 1 + 2b log2 |x| − 2b log2 |x| 2−b ln2 .

For x = 0,

x 1 + 2b log2 |x| − 2b log2 |x| 2−b ln2 = 0. (71) We model the casting error as

SEaMb(0) ≈ lim

x→0

SEaMb(x) − x = (ln2)2−bξx (72)

where x ∈ [−max|xSEaMb|,max|xSEaMb|] and, similar to the INT types in Eq. (61), the random variable ξ ∼ Uniform −2−1,2−1 .

For a quantization format that uses EaMb for values and E˜aM˜b for scales, we approximate it with a scalar quantizer q (x) = SEaMb xs−1 SE˜aM˜b(s) where x ∈ R, scale s ∈ R̸=0, |s| ≤ max xSE˜aM˜b , and xs−1 ≤ max|xSEaMb|. Then,

q (x) = SEaMb xs−1 SE˜aM˜b(s)

= 1 + (ln2)2−bξ xs−1 1 + (ln2)2−˜bξ˜ s

(73)

= 1 + (ln2)2−bξ 1 + (ln2)2−˜bξ˜ x

with the independent random variables ξ,ξ˜∼ Uniform −2−1,2−1 . Define

η = 1 + (ln2)2−bξ 1 + (ln2)2−˜bξ˜ − 1. (74) Then, the quantization error can be expressed as

q (x) − x = ηx. (75) And the expected quantization error is 0 because

Eηη = Eξ,ξ˜(ln2)2−bξ + (ln2)2−˜bξ˜+ (ln2)2 2−b−˜bξξ˜= 0. (76)

#### A.6. Maximum Inequalities for Tail-Bounded Distributions

- Lemma A.3 (Expectation from the Survival Function). Let X ≥ 0 be a random variable with the density function f and the survival function

S(t) = P(X ≥ t) (77) with t ≥ 0. Let g : [0,∞) → R be integrable and define

t

g (u)du. (78) Assume that

G(t) =

0

G(t)S (t) = 0. (79) Then,

lim

t→∞

∞

g (t)S (t)dt. (80)

EG(X) =

0

Proof. Since X has density f and takes values in [0,∞), EG(X) =

∞

G(t)f (t)dt. (81) The survival function satisfies

0

∞

f (u)du, (82) so S is differentiable almost everywhere with S′ (t) = −f (t). Hence,

S (t) = P(X ≥ t) =

t

∞

∞

G(t)S′ (t)dt. (83) We now integrate by parts, using G′ (t) = g (t),

EG(X) =

G(t)f (t)dt = −

0

0

∞

∞

G(t)S′ (t)dt = −[G(t)S (t)]∞0 +

g (t)S (t)dt. (84)

−

0

0

By definition, G(0) = 0, and by assumption limt→∞ G(t)S (t) = 0, the boundary term [G(t)S (t)]∞0 = 0. Therefore,

∞

g (t)S (t)dt. (85)

EG(X) =

0

| |
|---|

- A.6.1. GAUSSIAN DISTRIBUTION Definition of sub-Gaussian.

- A probability distribution of a random variable X ∈ R is called sub-Gaussian with variance proxy σ2 if Eexp((X − EX)t) ≤ exp 2−1σ2t2 (86)

for all t ∈ R, and the smallest such σ2 is called the optimal variance proxy. Basic properties of sub-Gaussian distributions.

The variance proxy is larger than or equal to the variance: σ2 ≥ E(X − EX)2. For a Gaussian distribution, the optimal variance proxy is the same as the variance.

Chernoff bound on the tail:

##### P(|X − EX| ≥ t) ≤ 2exp −2−1σ−2t2 (87)

for all t ≥ 0. Maximum inequality. Let X1,...,Xd be zero-mean Gaussian random variables with variance σ12,...,σd2. Denote

σ⋆2 = max

σk2 = max

EXk2 (88)

k

k

as the smallest variance proxy for X1,...,Xd. Using Lemma A.3,

Emax

k

Splitting at t⋆ = σ⋆ 2ln(2d) gives

∞

Xk2 = 2

t P max

|Xk| ≥ t dt

k

0

d

∞

t P

{|Xk| ≥ t} dt

= 2

0

k=1

d

∞

t⋆

P(|Xk| ≥ t) dt

≤ 2

t 1 dt + 2

t

0

t⋆

k=1

∞

t⋆

P(|Xk| ≥ t) dt

≤ 2

t dt + 2d

t max

k

0

t⋆

∞

t⋆

texp −2−1σ⋆−2t2 dt

≤ 2

t dt + 4d

0

t⋆

= t2 t0⋆ + 4d −σ⋆2 exp −2−1σ⋆−2t2 ∞t

⋆

= t2⋆ + 4dσ⋆2 exp −2−1σ⋆−2t2⋆ .

(89)

Xk2 ≤ 2σ⋆2 ln(2d) + 2σ⋆2 = (2ln(2d) + 2)max

EXk2 (90)

Emax

k

k

without needing the independence of Xk. Also, considering the trivial bound

then,

d

d

EXk2, (91)

Xk2 ≤ E

Xk2 =

EXk2 ≤ dmax

Emax

k

k

k=1

k=1

Xk2 ≤ min{d,(2ln(2d) + 2)}max

EXk2. (92)

Emax

k

k

- A.6.2. LAPLACIAN DISTRIBUTION For a random variable X ∼ Laplace(µ,b) and t ≥ 0,

∞

2−1b−1 exp −b−1x dx = exp −b−1t . (93)

P(|X − µ| ≥ t) = 2

t

The variance E(X − EX)2 = 2b2. Let X1,...,Xd be zero-mean Laplacian random variables with scale parameters b1,...,bd. Denote

- 1

- 2

1 2

EXk2. (94) For t ≥ 0,

EXk2 =

b⋆ = max

bk = max

max

k

k

k

P(|Xk| ≥ t) = exp −b−k 1t ≤ exp −b−⋆ 1t . (95)

- Using Lemma A.3,

∞

Xk2 = 2

t P max

Emax

|Xk| ≥ t dt

k

k

0

d

∞

t P

{|Xk| ≥ t} dt

= 2

0

k=1

d

∞

t⋆

P(|Xk| ≥ t) dt

≤ 2

t 1 dt + 2

t

0

t⋆

(96)

k=1

∞

t⋆

P(|Xk| ≥ t) dt

≤ 2

t dt + 2d

t max

k

0

t⋆

∞

t⋆

texp −b−⋆ 1t dt

≤ 2

t dt + 2d

0

t⋆

= t2 t0⋆ + 2d −b⋆ (t + b⋆)exp −b−⋆ 1t ∞t

⋆

= t2⋆ + 2db⋆ (t⋆ + b⋆)exp −b−⋆ 1t⋆ .

Splitting at t⋆ = b⋆ lnd gives

Xk2 ≤ b2⋆ (lnd)2 + 2lnd + 2 = 2−1 (lnd)2 + lnd + 1 max

EXk2 (97) without needing the independence of Xk.

Emax

k

k

### B. Further Experimental Results

#### B.1. Additional Accuracy Results

In this section, we report additional evaluations conducted across the Llama 3 and Qwen 3 family models in Tables 3 to 5 and Figures 7 to 10. For the Qwen models, the thinking mode is disabled across all experiments. The evaluation results are obtained using emulated (“fake”) quantization, where quantized values are represented in the BF16/FP16 format. Results are aggregated over multiple random seeds.

- Table 3. W4A4 accuracy results on the LM Evaluation Harness for different models with RTN, GPTQ quantizations, and identity (I), Hadamard (H), WUSH transforms. For the Llama model, the MMLU-CoT metric is reported in the MMLU column.

Model Format Quantization Transform MMLU GSM8K HellaSwag WinoGrande Average Recovery

BF16 - - 64.43 78.01 73.42 70.09 71.49 100.0

I 61.07 72.01 70.90 66.77 67.69 94.68 H 59.91 64.82 69.77 65.59 65.02 90.95

Llama-3.2-3B-Instruct

RTN

NVFP4

- WUSH 59.91 71.24 70.96 67.25 67.34 94.19

GPTQ

H 60.22 70.91 71.06 66.57 67.19 93.99

- WUSH 60.17 72.71 71.26 67.32 67.87 94.93

I 56.81 60.80 67.30 64.56 62.37 87.24 H 55.99 61.11 68.36 64.09 62.39 87.27

RTN

WUSH 59.54 69.49 70.20 66.47 66.43 92.92 GPTQ

MXFP4

H 60.24 68.84 69.58 66.08 66.18 92.58 WUSH 59.39 68.26 69.88 67.07 66.15 92.53

BF16 - - 72.98 90.90 75.52 70.56 77.49 100.0

I 70.47 88.40 74.44 69.53 75.71 97.70 H 70.19 86.35 73.02 68.11 74.42 96.04

RTN

NVFP4

- WUSH 70.86 89.92 74.84 70.02 76.42 98.61

GPTQ

H 70.78 89.11 74.34 69.25 75.87 97.91

- WUSH 71.03 89.56 74.88 70.32 76.45 98.66

Qwen3-8B

I 67.69 84.23 71.24 67.40 72.64 93.74 H 67.53 87.04 71.48 67.64 73.42 94.75

RTN

MXFP4

- WUSH 69.96 86.90 73.87 68.92 74.91 96.68

GPTQ

H 69.58 88.28 73.75 69.25 75.22 97.07

- WUSH 70.40 88.80 74.36 69.19 75.69 97.68

BF16 - - 77.18 91.96 79.84 74.27 80.81 100.0

Qwen3-14B

I 75.20 90.22 77.38 73.32 79.03 97.80 H 74.98 92.04 77.76 72.38 79.29 98.12

NVFP4 RTN

WUSH 75.43 91.78 78.82 73.18 79.80 98.76

I 72.92 90.22 76.68 71.51 77.83 96.31 H 73.13 87.41 76.29 71.67 77.13 95.44

MXFP4 RTN

WUSH 74.56 91.87 78.18 73.15 79.44 98.30

BF16 - - 80.74 92.12 83.96 76.95 83.44 100.0

Qwen3-32B

I 78.92 90.14 82.60 76.64 82.08 98.37 H 79.48 91.81 82.51 74.98 82.20 98.51

NVFP4 RTN

WUSH 79.56 92.72 83.14 75.79 82.80 99.24

I 76.82 75.28 81.32 74.74 77.04 92.33 H 78.63 92.57 81.87 75.53 82.15 98.45

MXFP4 RTN

WUSH 78.70 90.83 82.63 75.87 82.01 98.28

- Table 4. KL-divergence for Qwen3-8B measured on a slice of C4 dataset, with calibration performed on the FineWeb dataset. Results (lower is better) are averaged over multiple random seeds. H denotes the Hadamard transform.

Model Format Quantization Transform KL

GPTQ H 0.069518 RTN WUSH 0.065747 GPTQ WUSH 0.054646

NVFP4

Qwen3-8B

GPTQ H 0.093119 RTN WUSH 0.093739 GPTQ WUSH 0.077909

MXFP4

Table 5. Sensitivity of WUSH to calibration datasets on Qwen3-8B with MXFP4.

Quantization Calibration Dataset LM Eval Harness Average Platinum Benchmarks Average

FineWeb 74.91 93.00 C4 75.57 92.89

RTN

Open-Platypus 74.99 93.12 GPTQ

FineWeb 75.69 93.52 C4 75.78 93.53

Qwen3-14B Platinum Benchmarks

BF16

- 100.0 100.0 98.82 97.74 98.13 97.38 99.50 89.47 98.00 97.24 91.30 91.87 86.67 95.86
- 100.0 100.0 98.82 98.04 97.69 96.40 99.50 88.00 93.90 96.46 87.95 88.33 85.23 94.64
- 99.83 100.0 98.97 98.02 97.20 92.70 99.25 84.21 95.25 97.10 94.72 89.83 82.69 94.60

NVFP4-RTN-WUSH

MXFP4-RTN-H

MXFP4-RTN-WUSH

- 99.87 100.0 98.82 97.81 97.16 95.96 99.00 87.47 94.50 96.57 87.45 87.46 84.82 94.38
- 100.0 100.0 99.18 97.74 97.76 95.51 99.10 85.37 94.80 95.91 84.97 86.32 83.49 93.86

NVFP4-RTN-I

SingleOpSingleQMultiArithSVAMPGSM8KMMLU-MathBBHDeductionBBHCountingBBHNavigateHotpotQASQuAD DROPWinograd-WSCAverage

Qwen3-14B Platinum Benchmarks Average Accuracy

96.0

BF16

95.5

Accuracy[%]

95.0

94.5

94.0

93.5

NVFP4-RTN-I MXFP4-RTN-WUSH MXFP4-RTN-H NVFP4-RTN-WUSH

- Figure 7. Comparison of different transforms on Qwen3-14B for both NVFP4 and MXFP4 quantization on Platinum Benchmarks. The left table shows accuracy results across the individual benchmark tasks, while the right plot shows the average accuracy scores together with their standard deviations for each transform.

SingleOpSingleQMultiArithSVAMPGSM8KMMLU-MathBBHDeductionBBHCountingBBHNavigateHotpotQASQuAD DROPWinograd-WSCAverage

BF16

NVFP4-RTN-H

NVFP4-RTN-WUSH

MXFP4-RTN-WUSH

NVFP4-RTN-I

- MXFP4-RTN-H

- MXFP4-RTN-I

100.0 100.0 99.41 97.36 98.88 94.01 100.0 95.26 98.00 99.45 95.03 94.74 93.85 97.38

99.67 100.0 98.97 99.06 98.13 95.32 99.75 91.18 97.88 97.79 93.32 93.54 92.44 96.70

99.83 100.0 98.82 97.92 98.13 95.60 99.50 91.32 97.25 98.07 93.01 93.42 92.69 96.58

- 99.33 100.0 99.12 98.58 97.76 95.51 99.12 90.66 97.75 98.62 93.79 92.82 91.15 96.48
- 100.0 100.0 99.41 98.58 98.69 94.48 99.75 86.84 97.25 98.34 92.86 94.38 92.31 96.38

- 99.33 100.0 98.24 97.26 97.76 93.63 99.25 81.18 96.50 98.48 93.01 92.58 90.90 95.24
- 100.0 100.0 99.26 98.40 97.67 92.51 99.25 85.26 94.75 97.38 91.77 90.43 90.64 95.18

Qwen3-32B Platinum Benchmarks

MXFP4-RTN-I MXFP4-RTN-H NVFP4-RTN-I MXFP4-RTN-WUSH NVFP4-RTN-WUSH NVFP4-RTN-H

94.5

95.0

95.5

96.0

96.5

97.0

97.5

98.0

Accuracy[%]

BF16

Qwen3-32B Platinum Benchmarks Average Accuracy

- Figure 8. Comparison of different transforms on Qwen3-32B for both NVFP4 and MXFP4 quantization on Platinum Benchmarks. The left table shows accuracy results across the individual benchmark tasks, while the right plot shows the average accuracy scores together with their standard deviations for each transform.

Llama-3.2-3B-Instruct Platinum Benchmarks

BF16 NVFP4-GPTQ-I

98.67 98.00 99.41 91.32 84.70 62.55 73.50 78.95 63.50 81.22 79.50 53.59 63.59 79.11 97.20 98.80 96.82 87.77 79.25 58.73 66.80 68.74 65.30 79.45 71.80 53.68 62.56 75.92

NVFP4-GPTQ-WUSH NVFP4-GPTQ-H NVFP4-RTN-WUSH

- 96.80 98.80 96.59 86.49 79.55 57.15 64.50 69.47 62.90 75.58 73.29 52.92 62.36 75.11
- 97.33 98.80 96.24 87.62 80.60 58.80 61.30 66.42 65.60 73.59 70.68 54.45 62.36 74.91 97.07 98.00 96.12 87.32 80.82 56.25 66.60 69.26 63.90 73.26 71.06 52.54 60.72 74.84

NVFP4-RTN-I MXFP4-GPTQ-WUSH

- 96.50 98.50 95.74 88.49 79.57 58.71 61.25 61.05 65.75 81.22 68.48 54.78 54.87 74.22

- 96.67 98.00 96.59 87.09 79.48 52.66 64.20 64.11 61.30 73.81 71.30 58.09 60.92 74.17
- 97.33 98.80 95.18 87.09 77.91 49.51 63.00 64.00 63.90 68.18 70.43 47.08 60.00 72.49

- 97.33 97.60 96.47 85.28 76.49 52.13 59.20 65.58 61.80 70.17 70.06 50.24 57.44 72.29 94.93 97.20 92.71 83.25 73.66 45.02 59.40 60.00 63.80 69.72 65.22 50.62 57.03 70.20

MXFP4-GPTQ-H MXFP4-RTN-WUSH MXFP4-RTN-H

SingleOpSingleQMultiArithSVAMPGSM8KMMLU-MathBBHDeductionBBHCountingBBHNavigateHotpotQASQuAD DROPWinograd-WSCAverage

Llama-3.2-3B-Instruct Platinum Benchmarks Average Accuracy

80.0

BF16

78.0

Accuracy[%]

76.0

74.0

72.0

70.0

68.0

MXFP4-RTN-HMXFP4-RTN-WUSHMXFP4-GPTQ-HMXFP4-GPTQ-WUSH NVFP4-RTN-INVFP4-RTN-WUSHNVFP4-GPTQ-HNVFP4-GPTQ-WUSHNVFP4-GPTQ-I

- Figure 9. Comparison of different transforms on Llama-3.2-3B-Instruct for both NVFP4 and MXFP4 quantization on Platinum Benchmarks. The left table shows accuracy results across the individual benchmark tasks, while the right plot shows the average accuracy scores together with their standard deviations for each transform.

SingleOpSingleQMultiArithSVAMPGSM8KMMLU-MathBBHDeductionBBHCountingBBHNavigateHotpotQASQuAD DROPWinograd-WSCAverage

BF16

NVFP4-RTN-I

NVFP4-RTN-WUSH

MXFP4-RTN-WUSH

MXFP4-GPTQ-WUSH

MXFP4-RTN-H

97.33 99.00 99.41 92.08 88.81 69.29 79.00 87.89 75.00 86.74 80.75 76.08 77.95 85.33

97.47 99.40 96.35 88.75 86.79 59.10 72.80 80.95 71.20 87.29 78.76 75.89 71.08 81.99

- 97.47 97.60 97.41 88.68 85.30 57.90 74.10 76.42 70.00 84.20 79.38 73.21 68.72 80.80
- 98.00 97.80 96.24 87.40 82.61 51.61 69.50 77.89 66.60 84.86 75.03 71.48 72.41 79.34

97.47 98.40 96.71 88.15 84.25 55.28 71.10 71.79 67.80 83.98 76.02 69.47 69.85 79.25

95.73 97.20 95.18 84.30 79.40 49.74 66.00 71.37 64.90 84.31 71.93 65.74 67.59 76.41

Llama-3.1-8B-Instruct Platinum Benchmarks

MXFP4-RTN-H MXFP4-GPTQ-WUSH MXFP4-RTN-WUSH NVFP4-RTN-WUSH NVFP4-RTN-I

74.0

76.0

78.0

80.0

82.0

84.0

86.0

Accuracy[%]

BF16

Llama-3.1-8B-Instruct Platinum Benchmarks Average Accuracy

- Figure 10. Comparison of different transforms on Llama-3.1-8B-Instruct for both NVFP4 and MXFP4 quantization on Platinum Benchmarks. The left table shows accuracy results across the individual benchmark tasks, while the right plot shows the average accuracy scores together with their standard deviations for each transform.

#### B.2. Computational and Memory Costs

- Table 6 reports the offline preprocessing cost of WUSH with RTN. These measurements include the calibration and transform construction pipeline. Note that the offline programs are generally not optimally implemented, and there are some configuration choices that can significantly vary the measurements. For example, one can choose to offload and recompute the calibration activations to save GPU memory at the cost of increased runtime.
- Table 7 reports the additional whole-checkpoint storage overhead from storing the activation-side WUSH transforms. Table 6. Offline preprocessing cost of WUSH with RTN.

Model GPU Time [min] GPU Memory [GB]

Llama-3.2-3B-Instruct H100 9 10 Llama-3.1-8B-Instruct H100 19 19 Qwen3-8B H100 25 17 Qwen3-14B H100 30 20 Qwen3-32B B200 38 40

Table 7. Whole-checkpoint storage overhead from storing the activation-side WUSH transforms.

Model MXFP4 NVFP4

Llama-3.2-3B-Instruct 2.1% 1.0% Llama-3.1-8B-Instruct 1.4% 0.7% Qwen3-8B 1.4% 0.7% Qwen3-14B 1.2% 0.6% Qwen3-32B 1.2% 0.6%

