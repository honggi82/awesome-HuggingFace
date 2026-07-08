## MSign: An Optimizer Preventing Training Instability in Large Language Models via Stable Rank Restoration

Lianhai Ren12† Yucheng Ding1† Xiao Liu1 Qianxiao Li2 Peng Cheng1 Yeyun Gong1

# arXiv:2602.01734v1[cs.LG]2Feb2026

### Abstract

Training instability remains a critical challenge in large language model (LLM) pretraining, often manifesting as sudden gradient explosions that waste significant computational resources. We study training failures in a 5M-parameter NanoGPT model scaled via µP, identifying two key phenomena preceding collapse: (1) rapid decline in weight matrix stable rank (ratio of squared Frobenius norm to squared spectral norm), and (2) increasing alignment between adjacent layer Jacobians. We prove theoretically that these two conditions jointly cause exponential gradient norm growth with network depth. To break this instability mechanism, we propose MSign, a new optimizer that periodically applies matrix sign operations to restore stable rank. Experiments on models from 5M to 3B parameters demonstrate that MSign effectively prevents training failures with a computational overhead of less than 7.0%.

### 1. Introduction

Modern deep learning optimizers and training techniques – including Adam (Kingma & Ba, 2015), layer normalization (Ba et al., 2016), and learning rate schedules – generally enable reliable training. However, as the model scale increases, pretraining large language models (LLMs) becomes increasingly fragile. Training failures manifest as sudden, unrecoverable gradient explosions and corresponding loss growth, which are difficult to predict and can waste substantial computational resources (Chowdhery et al., 2023; Zhang et al., 2022).

Empirical Investigation: Identifying the Failure Mechanism. We systematically study training failures using

†Work done during their internship at Microsoft Research. 1Microsoft SIGMA Team, Microsoft Research. 2Department of Mathematics, National University of Singapore, 10 Lower Kent Ridge Rd, Singapore 119076. Correspondence to: Peng Cheng <pengc@microsoft.com>, Yeyun Gong <yegong@microsoft.com>.

Preprint. February 3, 2026.

a 5M-parameter NanoGPT model derived from µP scaling (Yang et al., 2022), which provides a controlled environment for reproducible failure analysis. Through extensive monitoring, we identify two critical phenomena that consistently precede training collapse (Figure 1):

- • Observation 1: Stable Rank Collapse. The stable rank of weight matrices – defined as srank(W) =

∥W∥2F/∥W∥22 – declines sharply in the steps preceding failure. This indicates that spectral energy becomes concentrated in the top singular directions, reducing the “effective dimensionality” of the weight matrices.

- • Observation 2: Jacobian Alignment Growth. The alignment between adjacent layer Jacobians increases, meaning the top singular subspaces of consecutive layers become increasingly correlated. It prevents the typical cancellation effects in matrix products.

We prove that these two phenomena jointly cause training instability: low stable rank implies high layer Jacobian spectral norms (since ∥W∥2 = ∥W∥F/ srank(W) for fixed Frobenius norm), and high alignment ensures these norms multiply constructively across layers. The total Jacobian norm grows as (aM)L where a is alignment, M is layer Jacobian norm, and L is depth, leading to exponential gradient explosion when aM > 1.

Our Solution: The MSign Optimizer. To break the stable rank collapse condition, we propose MSign, an optimizer that periodically applies the matrix sign operation sign(W) = UVT (where W = USVT is the SVD). This operation maximizes stable rank by equalizing all non-zero singular values to 1, while preserving the column and row spaces. We restore the original Frobenius norm after the operation to maintain training dynamics. MSign is applied every P steps (typically P = 100) to projection weights, with computational overhead below 7.0%.

Experimental Validation. We validate MSign across four model configurations spanning 5M to 3B parameters: NanoGPT-5M 1, Sigma-40M (hybrid MHA/MLA attention) (Qu et al., 2025; Hu et al., 2025), LLaMA-1B (Kumar et al., 2025), and LLaMA-MoE-3B (mixture of experts). In all cases, baseline training with standard hyperparame-

1https://github.com/karpathy/nanoGPT

ters fails via gradient explosion, while training with MSign converges stably. The intervention maintains stable rank above critical thresholds, controls Jacobian alignment, and keeps gradient norms bounded. Ablation studies reveal that applying MSign to attention layers (particularly output projections) is sufficient, MLP-only application does not prevent failures.

Contributions. We summarize our contributions:

- 1. Mechanism identification and theoretical analysis: We identify stable rank collapse and Jacobian alignment as consistent precursors to LLM training failures, and prove that their combination causes exponential gradient growth with network depth.
- 2. Practical solution: We propose MSign, a new optimizer that prevents training failures by periodically restoring weight stable rank.
- 3. Extensive validation: We demonstrate MSign’s effectiveness across diverse architectures (dense, MoE) and scales (5M–3B parameters) with minimal overhead.

### 2. Literature Review

Training Instability in Large Language Models. Training instability in large language models has been widely observed across major LLM projects. Chowdhery et al. (2023) documented loss spikes during PaLM training that required manual intervention and checkpoint rollbacks, while Zhang et al. (2022) provided detailed logs of OPT-175B training showing dozens of restarts due to hardware failures and gradient explosions. Zeng et al. (2023) reported similar challenges in GLM-130B training. Several factors have been proposed to explain instability: Kaplan et al. (2020) studied the sensitivity of training dynamics to learning rate in the context of scaling laws; Pascanu et al. (2012) analyzed gradient clipping as a remedy for exploding gradients in RNNs; Dong et al. (2021) identified attention entropy collapse as a failure mode in Vision Transformers. More recently, Wortsman et al. (2024) showed that small-scale proxy models can predict instabilities at larger scales, and Moniri et al. (2024) proposed a theoretical framework connecting loss spikes to heavy-tailed gradient noise. However, these explanations typically address symptoms rather than underlying mechanisms.

Low-Rank Structure in Neural Networks. The prevalence of low-rank structure in neural network weights and gradients has been extensively documented. Denil et al. (2013) demonstrated that neural network weights exhibit significant redundancy, with up to 95% of parameters predictable from the remaining 5%. Arora et al. (2019) proved that gradient descent in deep linear networks implicitly biases toward low-rank solutions. In the transformer context, Hu et al. (2022) exploited low-rank structure for parameter-

efficient fine-tuning, while Zhao et al. (2024) showed that gradients during transformer training maintain consistently low stable rank, motivating memory-efficient optimizers. Huh et al. (2024) found that representations across different models and modalities converge to similar low-dimensional structures. Our work extends these observations by connecting low-rank gradient structure directly to training instability through the stable rank mechanism.

Jacobian Analysis in Deep Networks. The role of Jacobians in neural network optimization has received substantial theoretical attention. Classical work by Glorot & Bengio (2010) established the importance of proper weight initialization for maintaining stable gradient flow. Saxe et al. (2013) analyzed dynamics in deep linear networks through the lens of Jacobian singular values. Pennington et al. (2017) and Yang & Schoenholz (2017) developed mean-field theories for Jacobian evolution in residual networks and general deep networks respectively. Fort et al. (2019) empirically studied the relationship between Jacobian eigenvalue spectra and model trainability, finding that successful training correlates with specific spectral properties. Xiao et al. (2018) proposed dynamical isometry as a condition for stable training. Our contribution is identifying inter-layer Jacobian alignment as a key mechanism amplifying gradient explosion, distinct from prior work that focused on individual layer Jacobian norms.

Stable Rank and Matrix Analysis. Stable rank, introduced by Rudelson & Vershynin (2007) in the context of random matrix theory, provides a continuous relaxation of matrix rank that is more robust to small perturbations. Vershynin (2018) provides comprehensive treatment of stable rank properties and its applications in high-dimensional probability. In neural network analysis, stable rank has been used to derive generalization bounds: Neyshabur et al. (2017) used stable rank to obtain PAC-Bayes bounds. Li et al. (2018) analyzed intrinsic rank dynamics during optimization, finding that it correlates with model compressibility. Sanyal et al. (2020) proposed stable rank regularization to improve generalization. Our work reveals a novel connection between stable rank dynamics and training stability, showing that stable rank collapse precedes and causes gradient explosion.

### 3. Empirical Observations: Training Failure Phenomena

We identify reproducible training failures in transformer models under moderate learning rates. Our analysis reveals two consistent phenomena preceding training collapse.

[Figure 1]

[Figure 2]

- Figure 1. Correlation between training failure indicators and gradient norm explosion. Left (Observation 1): Stable rank (geometric mean across early layers) vs. gradient norm over training steps. As stable rank declines sharply around step 20000, gradient norms begin explosive growth. Right (Observation 2): Jacobian alignment (average between adjacent layers) vs. gradient norm. Increasing alignment precedes and accompanies gradient explosion.

#### 3.1. Experimental Setup

We construct a reproducible failure scenario using a modified NanoGPT configuration with standard hyperparameters (refer to Appendix A for details).

From µP perspective (Yang et al., 2022), this configuration corresponds to a 0.02 std and 6 × 10−4 learning rate at 100M scale, which is within typical ranges.

Model and Jacobian Notation. We consider a standard decoder-only transformer with L stacked blocks. At each block ℓ, hidden states H(ℓ−1) ∈ RT×d (sequence length T, hidden dimension d) are first processed by multihead self-attention with query/key/value/output projections

WQ(ℓ),WK(ℓ),WV(ℓ),WO(ℓ), and then by a position-wise MLP with weights W1(ℓ) ∈ Rd×d

ff and W2(ℓ) ∈ Rd

ff×d, with residual connections and LayerNorm around each sublayer:

##### H(ℓ) = F(ℓ)(H(ℓ−1)), ℓ = 1,...,L. (1)

We define the layer Jacobian as J(ℓ) =

∂ vec(H(ℓ))

∂ vec(H(ℓ−1)). Later references to the stable ranks of WQ,WK,WV ,WO,W1,W2 and to ”Jacobian alignment between adjacent layers” all refer to this architecture and these J(ℓ).

#### 3.2. Observations during training failure

- Observation 1: Sharp Stable Rank Decline The first observable phenomenon is the rapid decline in weight stable rank. As shown in Figure 1 (left panel), the geometric averaged stable rank of the first several layers drops sharply around step 20000, preceding the gradient explosion. This phenomenon indicates energy concentration in top singular values.

- Definition 3.1 (Stable Rank). The stable rank of a matrix W ∈ Rm×n is:

srank(W) = ∥W∥2F ∥W∥22

= i

s2i s21

, (2)

where s1 ≥ s2 ≥ ··· ≥ 0 are the singular values.

Observation 2: Increasing Jacobian Alignment Between Adjacent Layers The second critical phenomenon is the increasing alignment between Jacobians of adjacent layers, as illustrated in Figure 1 (right panel).

- Definition 3.2 (Matrix Alignment). For any two matrices A,B for which the product AB is well-defined, their alignment is the cosine similarity between the top right singular vector of A and the top left singular vector of B:

Align(A,B) = |vA,T 1uB,1|, (3)

where vA,1 is the first right singular vector of A and uB,1 is the first left singular vector of B. For simplicity, we denote the alignment of Jacobians for adjacent layers, Align(J(ℓ+1),J(ℓ)), as Align(ℓ + 1,ℓ).

High alignment correlates strongly with both weight scale growth and stable rank decline, as well as gradient growth during the failure phase.

#### 3.3. Conjecture: Low Stable Rank + Jacobian Alignment Drives Training Failure

Based on these two consistent phenomena, we conjecture that the combination of low weight stable rank and high Jacobian alignment creates a destabilizing feedback mechanism that leads to training failure. The remainder of this paper develops this hypothesis theoretically and proposes a solution.

### 4. Theoretical Analysis: Understanding the Failure Mechanism

We now provide theoretical analysis to explain the observed phenomena and their causal relationships. Our analysis is divided into two parts: (1) explaining why Jacobian alignment and low stable rank lead to training failure, and (2) analyzing the positive feedback mechanism that prevents stable rank from increasing and accelerates its collapse.

- 4.1. Part I: From Observations to Training Failure

In this section, we establish the causal chain: Low stable rank + Jacobian alignment ⇒ High total Jacobian norm ⇒ Large weight gradient norm (Training instability).

Remark 4.1 (Simplifying Assumption). We adopt the standard assumption that large gradient norms indicate training instability (Pascanu et al., 2012; Philipp et al., 2017). Our analysis derives lower bounds for gradient norms, with the understanding that such bounds indicate increased risk of divergence.

- 4.1.1. HIGH JACOBIAN ALIGNMENT + HIGH LAYER JACOBIAN NORM ⇒ HIGH TOTAL JACOBIAN NORM

The total Jacobian Jtotal = Lℓ=1 J(ℓ) determines how perturbations at the input propagate to the output. In general,

the norm of a matrix product can be much smaller than the product of norms due to cancellation effects. However, when the singular subspaces of adjacent Jacobians are aligned, these cancellations are suppressed, and the norms multiply constructively.

To formalize this, we use the definition of matrix alignment (Definition 3.2). This quantity measures how well the ”output direction” of J(ℓ) matches the ”input direction” of J(ℓ+1), determining whether the composition J(ℓ+1)J(ℓ) preserves or diminishes the spectral norm.

Theorem 4.2 (Jacobian Product Norm Lower Bound). For a deep network with L layers, let J(ℓ) = ∂h

(ℓ)

denote the Jacobian at layer ℓ. If each layer Jacobian satisfies ∥J(ℓ)∥2 ≥ M and the alignment between adjacent Jacobians satisfies Align(ℓ + 1,ℓ) ≥ a > 0 for all ℓ, then the total Jacobian from input to output has 2-norm:

∂h(ℓ−1)

(aM)L a

∥Jtotal∥2 = ∥J(L)J(L−1) ···J(1)∥2 ≥

. (4)

The proof is provided in Appendix B.1.

Remark 4.3 (Exponential Growth Condition). The key insight is that when aM > 1, the lower bound in Theorem 4.2 grows exponentially with depth L, providing a sufficient condition for large total Jacobian norms. Observation 1 (stable rank collapse) suggests that M can become large (via Theorem 4.4 under approximately fixed Frobenius norms), and Observation 2 shows that a tends to increase during training. In the failure regimes we study, these trends empirically drive aM above 1, which is consistent with the observed gradient explosion, although we do not claim that aM > 1 holds at all times or is necessary for failure.

4.1.2. LOW STABLE RANK ⇒ HIGH LAYER JACOBIAN NORM

We analyze the relationship between stable rank and layer Jacobian norm for the three primary layer types in transformers.

#### Linear Layers.

- Theorem 4.4 (Stable Rank Controls Jacobian Norm: Linear Layer). For a linear layer with weight matrix W ∈ Rm×n,

given fixed Frobenius norm ∥W∥F = F, the operator norm satisfies:

∥W∥2 =

F srank(W)

. (5)

- The proof is provided in Appendix B.2. This establishes the basic principle: as stable rank decreases while the Frobenius norm is kept fixed (or approximately fixed over short training windows, e.g., under Adam-like updates with bounded step sizes), the operator norm increases proportionally. Attention Layers.

Theorem 4.5 (Jacobian Norm Bound: Attention Layer). Consider a single-head attention layer with projections WQ,WK,WV ,WO ∈ Rd×d

k. Let H ∈ Rn×d be the input, A = softmax HWQW

T KHT

√dk be the attention matrix. The Jacobian of the attention output Y = AHWV WO satisfies:

∂Y ∂H 2 ≤∥A∥2∥WV ∥2∥WO∥2

+

∂A ∂H 2 ∥H∥2∥WV ∥2∥WO∥2.

(6)

Defining the logit margin γmin = mini(maxj Si,j − second maxjSi,j) where S = HWQWKT HT, the attention gradient pathway is bounded by:

∂A ∂H 2 ≤

4min((n − 1)e−γ

min

,1) √dk ∥WQ∥2∥WK∥2.

(7)

- The proof is provided in Appendix B.3. Substituting

- Theorem 4.6 (Jacobian Norm Bound: MLP Layer). Con-

∥Wi∥2 = ∥Wi∥F/ srank(Wi) shows that low stable rank in any projection matrix amplifies the Jacobian norm.

#### MLP Layers.

sider a two-layer MLP with weights W1 ∈ Rd×d

ff, W2 ∈ Rd

ff×d, and activation ϕ. The Jacobian satisfies:

Lϕ∥W1∥F∥W2∥F srank(W1) · srank(W2)

, (8)

∥JMLP∥2 ≤

where Lϕ is the Lipschitz constant of ϕ (Lϕ ≈ 1.13 for GELU, Lϕ ≈ 1.1 for SiLU).

[Figure 3]

- Figure 2. Validation of Theorem 4.2: Jacobian product norm lower bound vs. actual gradient norm. The theoretical bound closely tracks observed gradient growth.

- The proof is provided in Appendix B.4. Across all layer types, the Jacobian norm is inversely related to the square root of stable rank, creating conditions for exponential gradient growth when combined with Jacobian alignment (Theorem 4.2).

- 4.1.3. HIGH TOTAL JACOBIAN NORM ⇒ LARGE WEIGHT GRADIENT NORM

We now show that high total Jacobian norms translate into large weight gradients. By the chain rule, the gradient with

respect to weight vector vˆout(i) at layer i decomposes as:

∂L ∂vˆout(i)

=

∂h(i) ∂vˆout(i)

T

∂h(L) ∂h(i)

T ∂L ∂h(L)

. (9)

Assumption 4.7 (Gradient Alignment Conditions (Informal)). We assume: (1) uniform local gradient lower bound γ > 0; (2) local-Jacobian alignment: ∂h

(i) ∂vˆout(i)

is aligned with the top right singular direction of the local layer Jacobian J(i+1), with alignment ≥ a; (3) terminal alignment: the last layer Jacobian J(L)’s top left singular direction is aligned with the loss gradient ∂h∂L

(L)

, with alignment ≥ a. The formal statement is provided in Assumption B.2 in the Appendix.

- Theorem 4.8 (Weight Gradient Norm Lower Bound). Under Assumption 4.7, combined with Theorem 4.2 when ∥J(ℓ)∥2 ≥ M > 1 and alignment ≥ a:

∂L ∂vˆout(i) 2

≥ aγ(aM)L−i ·

∂L ∂h(L) 2

. (10)

The proof is provided in Appendix B.5. Figure 2 validates this bound empirically.

- Theorem 4.9 (Total Gradient Norm Lower Bound). Summing over all layers:

2

where C = a2γ2nw ∂h ∂L

2. When aM > 1, this exhibits exponential growth in depth.

(L)

- The proof is provided in Appendix B.6. From Observations 1 and 2, stable rank collapse and Jacobian alignment together cause aM > 1, triggering exponential gradient explosion.

- 4.1.4. SUMMARY: THE FAILURE PATHWAY

Combining the above results, we establish the complete causal chain:

- 1. Low stable rank (Observation 1) ⇒ High layer Jacobian 2-norm (Theorems 4.4, 4.5, 4.6) under approximately fixed Frobenius norms.
- 2. Jacobian alignment (Observation 2) + High layer Jacobian norm ⇒ High total Jacobian norm (Theorem 4.2).
- 3. High total Jacobian norm + Gradient alignment (Assumption 4.7) ⇒ Large weight gradient (training instability) (Theorems 4.8, 4.9).

Taken together, these ingredients provide a sufficient mechanistic explanation for how low stable rank combined with Jacobian alignment can lead to training failure in the regimes we study; we do not claim that this mechanism is necessary for all possible failure modes.

- 4.2. Part II: The Positive Feedback Mechanism

Having established why low stable rank and Jacobian alignment lead to training failure, we now analyze why these conditions tend to intensify during training.

- 4.2.1. LOW-RANK HIDDEN STATES LEAD TO LOW-RANK GRADIENTS

We first establish that low-rank structure propagates through the network and affects gradient structure.

Theorem 4.10 (Low-Rank Propagation in Attention Layers). Consider an attention layer with query, key, value, and output projection matrices WQ,WK,WV ,WO. If the hidden states H(ℓ−1) and cohidden states H˜ (ℓ) have rank at most r, then the gradients ∇WQ

L,∇WK

L,∇WV

L,∇WO

L all have rank at most r.

- The proof is provided in Appendix B.7.

L

i=1

∂L ∂W(i)

2

(aM)2L − 1 (aM)2 − 1

, (11)

≥ C ·

F

Remark 4.11. This result extends to MLP layers due to their two-layer structure. For MLP up-projection W1: ∇W1

L = HT · (something), so rank is bounded by rank(H). The key insight is that outer-product gradients inherit the rank of their lower-rank factor.

### 5. The MSign Optimizer: Breaking the Feedback Loop

Algorithm 1 MSign Optimizer Input: parameters θ, gradients g, learning rate η, period P, step t, target layers Procedure StepWithMSign: θ ← BaseOptimizerStep(θ,g,η) ▷ Standard optimizer update if t mod P == 0 then

Based on our theoretical analysis and empirical observations, we propose a new optimizer to prevent stable rank collapse: the MSign optimizer. This method directly addresses the root cause by periodically restoring the stable rank of weight matrices via the matrix sign operation.

for each parameter W in target layers do

if W.ndim ≥ 2 then F ← ∥W∥F ▷ Record Frobenius norm U,S,VT ← SVD(W) W ← ∥UVFT∥F UVT ▷ Matrix sign with norm restoration

#### 5.1. The Matrix Sign Operation

Definition 5.1 (Matrix Sign Operator). For any matrix W ∈ Rm×n with reduced (thin) SVD W = USVT, where r = rank(W), U ∈ Rm×r, S ∈ Rr×r, and V ∈ Rn×r, we define:

end if end for

sign(W) = UVT ∈ Rm×n. (13)

end if

- 4.2.2. ALIGNED INPUT-WEIGHT-OUTPUT STRUCTURE ACCELERATES STABLE RANK DECLINE

In general, characterizing how gradient descent affects stable rank dynamics is challenging, as the update direction depends on the complex interplay between input statistics, weight structure, and backpropagated gradients. Here we provide a simplified analysis in a stylized linear setting under strong alignment assumptions that captures the essential feedback mechanism. While these assumptions are restrictive and do not aim to exactly describe full transformer dynamics, they offer theoretical insight into why stable rank can tend to decline during training.

Theorem 4.12 (Stable Rank Feedback Mechanism). Consider a linear layer with weight matrix W = USVT where the input hidden states and output cohidden states satisfy:

- • Input covariance: Σin = E[hinhTin] = VΛinVT (aligned with W’s right singular vectors)
- • Output gradient covariance: Σout = E[h˜outh˜Tout] = UΛoutUT (aligned with W’s left singular vectors)

If the correlation between input and output gradient projections Cov(uTi h˜out,viThin) is negative and satisfy

Cov(uT1 h˜out,v1Thin) Cov(uTi h˜out,viThin)

>

s1 si

,∀1 < i ≤ n (12)

then gradient descent causes the stable rank of W to decrease.

The proof is provided in Appendix B.8. This theorem should be interpreted as an existence result in a highly aligned regime that illustrates one concrete way in which gradient descent can decrease stable rank, rather than as a statement about typical training trajectories.

This operation sets all non-zero singular values to 1, creating a partial isometry that maximizes stable rank for a given matrix rank. Note that using the reduced SVD ensures the product UVT has the correct shape m × n.

#### 5.2. Practical Implementation

Scaling. The matrix sign operation alters the scale of the weight matrix, necessitating rescaling. A straightforward approach is to preserve the original Frobenius norm:

#### Wnew = ∥W∥F

sign(W). (14)

∥sign(W)∥F

In our current implementation, we adopt this Frobeniusnorm preserving rescaling for MSign; however, when the stable rank is sufficiently low, this choice may excessively amplify minor singular values, and designing more principled rescaling schemes is left for future work.

Approximation Strategies. Computing full SVD at every step is prohibitively expensive. We employ several practical strategies to reduce computational cost:

Periodic Application. MSign is applied every P steps (e.g., P = 100) rather than at every update. Our experiments demonstrate that this periodic application maintains effectiveness while substantially reducing computational overhead.

Selective Layer Targeting. Based on our empirical findings, MSign can be selectively applied to the most critical layers:

- • Attention-only: Apply only to self-attention weights (WQ,WK,WV ,WO)
- • 2D-parameters-only: Apply to all 2D parameter tensors (excluding biases and layer norm parameters)

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

- Figure 3. MSign prevents training failures across model scales. Top row: Training loss comparison between baseline (blue) and MSign (orange). Baseline training collapses with sudden loss spikes, while MSign maintains stable convergence. Bottom row: Corresponding gradient norm dynamics. Baseline runs exhibit exponential gradient explosion preceding collapse, while MSign keeps gradient norms bounded throughout training. Training is terminated after failure to conserve computational resources. Results demonstrate that MSign effectively breaks the stable rank collapse feedback loop identified in our theoretical analysis. From left to right: NanoGPT-5M, Sigma40M, LLaMA-1B, LLaMA-MoE-3B.

#### 5.3. Computational Cost Analysis

The computational overhead of MSign depends on the frequency and scope of application. Table 5 in Appendix A.3 provides a detailed comparison of computational costs for a typical transformer layer.

We define the overhead ratio R as the ratio of additional FLOPs introduced by the MSign operation to the original FLOPs of a standard training step:

FLOPsMSign FLOPsoriginal × P

, (15)

R =

where P is the application period (i.e., MSign is applied every P steps).

From the detailed breakdown in Table 5, we obtain the concrete formula for the application:

90d3 (72BTd2 + 12BT2d + O(BT2 + BTd)) × P

.

R =

(16) For a concrete example, consider a typical configuration with batch size B = 16, sequence length T = 1024, hidden dimension d = 2048, and periodic application every P = 100 steps. The MSign overhead per application for attention weights is 52d3 ≈ 4.47×1011 FLOPs. The original per-step FLOPs are approximately 72BTd2 + 12BT2d ≈ 5.36 × 1012 FLOPs. Thus:

4.47 × 1011 5.36 × 1012 × 100 ≈ 0.08%. (17)

R ≈

### 6. Experiment

We validate our approach across multiple experimental settings of increasing scale. Our experiments demonstrate: (1) the effectiveness of MSign in preventing training failures, and (2) the minimal impact on training throughput.

#### 6.1. Experimental Setting

We conduct experiments on four model configurations spanning different scales and architectures: NanoGPT-5M (No RoPE) 2; Sigma-40M (Hybrid Attention) (Qu et al., 2025; Hu et al., 2025); LLaMA-1B (Full RoPE) (Kumar et al., 2025); LLaMA-MOE-3B (Mixture of Experts) (Modified based on LLaMA-1B). Detailed configurations are provided in Table 4 in Appendix A.

#### 6.2. Main Result

MSign Prevents Training Failures. Figure 3 demonstrates that MSign effectively prevents training collapse across all experimental settings. We analyze the results from multiple perspectives.

Training Loss Dynamics (Top Row). The top row of Figure 3 shows training loss trajectories. For all four model scales, baseline training (blue curves) exhibits characteristic instability patterns: initial smooth convergence followed by sudden loss spikes and divergence. The collapse timing varies by model scale, NanoGPT-5M fails around step 30k, Sigma-40M around step 50k, LLaMA-1B around step 2k, and LLaMA-MoE-3B around step 3k, but the pattern is consistent across both dense and sparse architectures. In

2https://github.com/karpathy/nanoGPT

Table 1. Training throughput (tokens/second) across model scales. Measured overhead significantly exceeds theoretical predictions due to implementation factors discussed in Section 6.3.

#### NanoGPT-5M Sigma-40M LLaMA-1B LLaMA-MoE-3B

AdamW 102,708 6,504 1,742 544 MSign 105,199 6,097 1,640 520

Measured Overhead −2.4% 6.7% 6.2% 4.6% Theoretical Overhead (Eq. (16)) < 0.01% < 0.01% 0.03% 0.09%

contrast, MSign-enabled training (orange curves) maintains stable convergence throughout, achieving comparable or better final loss values. Notably, MSign is equally effective for the MoE architecture, where the distributed nature of expert computation does not affect the attention-based instability mechanism.

Gradient Norm Analysis (Bottom Row). The bottom row reveals the underlying mechanism. Baseline runs show exponential gradient explosion (reaching 101–107) immediately preceding each loss spike, confirming that gradient instability is the proximate cause of training failure. With MSign, gradient norms remain bounded within 100 throughout training. The periodic structure visible in the MSign curves corresponds to the application period P = 100, where each MSign application slightly perturbs the optimization trajectory before quickly stabilizing.

#### 6.3. Throughput Analysis

- Table 1 compares training throughput across model scales. For NanoGPT-5M, the measured overhead (−2.4%) falls within system noise, consistent with the theoretical prediction (< 0.1%). For larger models, however, measured overhead (4.6–6.7%) significantly exceeds theoretical values. This discrepancy arises from implementation factors not captured in FLOPs analysis: (1) all gather communication for distributed SVD, (2) disruption of FlashAttention kernel fusion and continuous stream execution, and

(3) pipeline bubbles in distributed training. Despite these overheads, the 4–7% throughput reduction remains modest compared to the computational waste from training failures.

6.4. Ablation Study

We conduct detailed ablation studies on the NanoGPT-5M and Sigma-40M configurations to identify the key factors affecting MSign effectiveness.

Ablation 1: Layer Selection: Attention vs. MLP. A critical finding is that MSign must be applied to attention layers to prevent training failures. Table 2 shows the results:

- Table 2 shows that applying MSign to MLP layers alone fails to prevent training collapse in both NanoGPT-5M and Sigma-40M, whereas attention-only application success-

- Table 2. Layer selection ablation: test perplexity (↓) on NanoGPT5M and Sigma-40M.

Target Layers NanoGPT-5M Sigma-40M

AdamW Failed Failed MSign (All 2D) 102.6 74.00 MSign (Attention only) 118.6 75.68 MSign (MLP only) Failed Failed

fully stabilizes training. This aligns with our theoretical analysis: attention layers create the inter-layer Jacobian structure that propagates through the network. Applying MSign to all 2D parameters achieves the best perplexity on both models (e.g., 102.6 vs. 118.6 on NanoGPT-5M), indicating that including MLP layers significantly improves final model quality.

Ablation 2: Application Period P. The application period P controls the trade-off between computational overhead and stable rank maintenance. Table 3 shows results for different values of P on NanoGPT-5M.

- Table 3. Application period ablation: test perplexity (↓) and throughput (tokens/s) across models.

Period P Test PPL Throughput NanoGPT Sigma Sigma

P = 10 103.9 92.7 18,236 P = 100 102.6 74.0 24,559 P = 1000 99.4 75.7 25,082 P = 10000 104.2 69.5 25,270 P = 100000 Failed Failed —

Table 3 demonstrates the robustness of MSign across a wide range of application periods. All tested values of P from 10 to 10,000 successfully prevent training collapse on both models in terms of final perplexity. However, as shown in Figure 4 in the Appendix, for NanoGPT-5M with P = 10000, the training dynamics exhibit noticeable instability: both loss and gradient norm show increased variance and occasional spikes compared to smaller P values, indicating that the stable rank may temporarily drop below safe thresholds between MSign applications. Since P = 100 already achieves acceptable computational overhead (Section 6.3) while maintaining stable training dynamics, we recommend P = 100 as the conservative default choice

rather than P = 1000, prioritizing training stability over marginal perplexity improvements.

### 7. Conclusion

We identify and analyze the stable rank collapse feedback loop as a fundamental mechanism underlying LLM training instability. Low weight stable rank amplifies layer Jacobian norms, which combined with inter-layer alignment causes exponential gradient growth. The MSign optimizer breaks this feedback loop by periodically restoring stable rank via the matrix sign operation, effectively preventing training failures with minimal overhead (< 7%). Future directions include adaptive scheduling, fused kernels for reduced latency, and extensions to other training pathologies.

While our theoretical results provide a rigorous foundation for understanding the positive feedback mechanism of stable rank collapse (see Theorem 4.12), they rely on strong assumptions—particularly, the uniform negative correlation of input and output gradient projections. These structural conditions may not universally hold in practice. Explicitly characterizing the full range of scenarios where the feedback loop provably dominates remains an open problem. We leave a complete characterization of these conditions and their relaxation to future work, which will further clarify the generality and boundaries of our theory.

### Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Arora, S., Cohen, N., Hu, W., and Luo, Y. Implicit regularization in deep matrix factorization. volume 32, 2019.

Ba, L. J., Kiros, J. R., and Hinton, G. E. Layer normalization. CoRR, abs/1607.06450, 2016. URL http://arxiv. org/abs/1607.06450.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra,

- G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., Schuh, P., Shi, K., Tsvyashchenko, S., Maynez, J., Rao, A., Barnes, P., Tay, Y., Shazeer, N., Prabhakaran, V., Reif, E., Du, N., Hutchinson, B., Pope, R., Bradbury, J., Austin, J., Isard, M., Gur-Ari, G., Yin, P., Duke, T., Levskaya, A., Ghemawat, S., Dev, S., Michalewski, H., Garcia, X., Misra, V., Robinson, K., Fedus, L., Zhou, D., Ippolito, D., Luan, D., Lim, H., Zoph, B., Spiridonov, A., Sepassi, R., Dohan, D., Agrawal, S., Omernick, M., Dai, A. M., Pillai, T. S., Pellat, M.,

Lewkowycz, A., Moreira, E., Child, R., Polozov, O., Lee, K., Zhou, Z., Wang, X., Saeta, B., Diaz, M., Firat, O., Catasta, M., Wei, J., Meier-Hellstern, K., Eck, D., Dean, J., Petrov, S., and Fiedel, N. PaLM: Scaling language modeling with Pathways. Journal of Machine Learning Research, 24(240):1–113, 2023. ISSN 1532-4435.

Denil, M., Shakibi, B., Dinh, L., Ranzato, M., and De Freitas, N. Predicting parameters in deep learning. volume 26, 2013.

Dong, Y., Cordonnier, J.-B., and Loukas, A. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In International conference on machine learning, pp. 2793–2803. PMLR, 2021.

Fort, S., Hu, H., and Lakshminarayanan, B. Deep ensembles: A loss landscape perspective. 2019.

Glorot, X. and Bengio, Y. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249–256. JMLR Workshop and Conference Proceedings, 2010.

Hu, E. J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=nZeVKeeFYf9.

Hu, Q., Lin, Z., Yang, Z., Ding, Y., Liu, X., Jiang, Y., Wang, R., Chen, T., Guo, Z., Xiong, Y., Gao, R., Qu, L., Su, J., Cheng, P., and Gong, Y. Sigma-MoE-Tiny technical report, 2025. URL https://arxiv.org/ abs/2512.16248.

Huh, M., Cheung, B., Wang, T., and Isola, P. Position: The platonic representation hypothesis. In Fortyfirst International Conference on Machine Learning, 2024. URL https://openreview.net/forum? id=BH8TYy0r6u.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. 2020. URL https://arxiv.org/abs/2001.

08361.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In Bengio, Y. and LeCun, Y. (eds.), 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http: //arxiv.org/abs/1412.6980.

Kumar, A., Owen, L., Chowdhury, N. R., and G”ura, F. Zclip: Adaptive spike mitigation for LLM pre-training, 2025. URL https://doi.org/10. 48550/arXiv.2504.02507.

Li, C., Farkhoor, H., Liu, R., and Yosinski, J. Measuring the intrinsic dimension of objective landscapes. In ICLR (Poster), 2018. URL https://openreview.net/ forum?id=ryup8-WCW.

Moniri, B., Lee, D., Hassani, H., and Dobriban, E. A theory of non-linear feature learning with one gradient step in two-layer neural networks. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Neyshabur, B., Bhojanapalli, S., McAllester, D., and Srebro, N. A pac-bayesian approach to spectrallynormalized margin bounds for neural networks. CoRR, abs/1707.09564, 2017. URL http://arxiv.org/ abs/1707.09564.

Pascanu, R., Mikolov, T., and Bengio, Y. On the difficulty of training recurrent neural networks. 30th International Conference on Machine Learning, ICML 2013, pp. 1310– 1318, 11 2012.

Pennington, J., Schoenholz, S. S., and Ganguli, S. Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. CoRR, abs/1711.04735, 2017. URL http://arxiv.org/abs/1711.04735.

Philipp, G., Song, D., and Carbonell, J. G. The exploding gradient problem demystified - definition, prevalence, impact, origin, tradeoffs, and solutions. arXiv preprint arXiv:1712.05577, 2017.

Qu, L., Ren, L., Cheng, P., Gao, R., Wang, R., Chen, T., Liu, X., Zhang, X., Gong, Y., Xiong, Y., Ding, Y., Jiang, Y., Lin, Z., Guo, Z., and Yang, Z. SIGMA: An AIempowered training stack on early-life hardware, 2025. URL https://arxiv.org/abs/2512.13488.

Rudelson, M. and Vershynin, R. Sampling from large matrices: An approach through geometric functional analysis. J. ACM, 54(4):21–es, July 2007. ISSN 00045411. doi: 10.1145/1255443.1255449. URL https: //doi.org/10.1145/1255443.1255449.

Sanyal, A., Torr, P. H., and Dokania, P. K. Stable rank normalization for improved generalization in neural networks and gans. In International Conference on Learning Representations, 2020. URL https://openreview.

net/forum?id=H1enKkrFDB.

Saxe, A. M., McClelland, J. L., and Ganguli, S. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. volume abs/1312.6120,

2013. URL https://api.semanticscholar. org/CorpusID:17272965.

Trefethen, L. N. and Bau III, D. Numerical Linear Algebra. SIAM, Philadelphia, 1997.

Vershynin, R. High-Dimensional Probability: An Introduction with Applications in Data Science, volume 47. Cambridge university press, 2018.

Wortsman, M., Liu, P. J., Xiao, L., Everett, K., Alemi, A., Adlam, B., Co-Reyes, J. D., Gur, I., Kumar, A., Novak, R., Pennington, J., Sohl-dickstein, J., Xu, K., Lee, J., Gilmer, J., and Kornblith, S. Small-scale proxies for largescale transformer training instabilities. In International Conference on Learning Representations, 2024.

Xiao, L., Bahri, Y., Sohl-Dickstein, J., Schoenholz, S., and Pennington, J. Dynamical isometry and a mean field theory of CNNs: How to train 10,000-layer vanilla convolutional neural networks. In International conference on machine learning, pp. 5393–5402. PMLR, 2018.

Yang, G. and Schoenholz, S. S. Mean field residual networks: On the edge of chaos. CoRR, abs/1712.08969, 2017. URL http://arxiv.org/ abs/1712.08969.

Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., and Gao, J. Tensor programs V: Tuning large neural networks via zero-shot hyperparameter transfer. CoRR, abs/2203.03466, 2022. doi: 10.48550/ARXIV.2203.03466. URL https:// doi.org/10.48550/arXiv.2203.03466.

Zeng, A., Liu, X., Du, Z., Wang, Z., Lai, H., Ding, M., Yang, Z., Xu, Y., Zheng, W., Xia, X., Tam, W. L., Ma, Z., Xue, Y., Zhai, J., Chen, W., Zhang, P., Dong, Y., and Tang, J. GLM-130B: An open bilingual pre-trained model. 2023. URL https://arxiv.org/abs/2210.02414.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., et al. OPT: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., and Tian, Y. GaLore: Memory-efficient LLM training by gradient low-rank projection. 2024.

### A. Experimental Details

#### A.1. Model Configurations

- Table 4 summarizes the experimental configurations across four model scales used in our experiments.

Table 4. Experimental configurations across four model scales

Setting NanoGPT-5M Sigma-40M LLaMA-1B LLaMA-MoE-3B

Parameters 5M 40M 1B 1B (active) / 3B (total) Layers 24 28 16 16 Hidden dim 48 128 2048 2048 Attention heads 6 4 16 16 Attention type MHA MHA/MLA alternating MHA MHA Architecture Dense Dense Dense MoE (16 experts, top-4) MLP intermediate 192 448 5440 1360 per expert Position encoding None RoPE (MHA only) RoPE RoPE Layernorm LayerNorm RMSNorm RMSNorm RMSNorm Activation GELU SiLU SiLU SiLU Initial Std1,2 0.02 0.006 0.02 0.02 Dataset OpenWebText Nemotron-cc Nemotron-cc Nemotron-cc Batch size (tokens) 123k 66k 66k 66k Sequence length 256 2048 2048 2048 Learning rate3 6 × 10−4 3.5 × 10−4 1 × 10−3 1 × 10−3 Optimizer AdamW AdamW AdamW AdamW

- 1For NanoGPT-5M and Sigma-40M, the µP scaling rules are applied: the initial 2D weights in transformer layers are scaled by 4×, and query weights (WQ) are initialized to zero.
- 2For NanoGPT-5M and Sigma-40M, the output projection weights (WO, Wdown) are further divided by √2 · #layers as part of the original architecture design.

- 3For NanoGPT-5M and Sigma-40M, the µP scaling rules are applied: the learning rate for 2D weights in transformer layers are scaled by 16×, and for embedding weights are scaled by 4×.

A.2. Common Training Settings The following hyperparameters and architectural choices are shared across all experiments unless otherwise specified:

Optimizer settings. We use the AdamW optimizer with β1 = 0.9, β2 = 0.95, and ϵ = 10−8. Weight decay is set to 0.1 for all experiments. Gradient clipping with max norm 1.0 is applied in all experiments. For the learning rate scheduler, we warm up for 2000 steps and linearly decay to 1/10 learning rate.

Architecture details. Bias terms are disabled in all linear layers (attention projections and MLP layers). Pre-LayerNorm is used in all transformer blocks.

A.3. Computational Cost Details

- Table 5 provides a detailed breakdown of computational costs for applying MSign to a typical transformer layer. The SVD computation dominates the MSign cost, with complexity O(d3) per weight matrix. However, when amortized over P steps (typically P = 100), the overhead becomes negligible compared to the forward and backward pass costs that scale as O(BTd2).

#### A.4. Throughput Model Analysis

To quantify the relationship between application period P and training throughput, we fit a simple analytical model. Let f denote the baseline computation per token (fixed), and F denote the additional computation per MSign application (fixed).

Table 5. Computational cost comparison per layer (hidden dim d, intermediate dim 4d, batch size B, sequence length T, applied every P steps)

Component FLOPs per step MSign FLOPs† Amortized MSign Attention Block WQ/WK/WV /WO 6BTd2 4 × 13d3 52d3/P Attention computation 12BT2d + O(BT2) 0 0 Total 24BTd2 + 12BT2d + O(BT2) 52d3 52d3/P MLP Block Wup/Wdown (d → 4d/4d → d) 24BTd2 2 × 19d3 38d3/P Activation O(BTd) 0 0 Total 48BTd2 + O(BTd) 38d3 38d3/P Entire Layer Original 72BTd2 + 12BT2d + O(BT2 + BTd) 0 0

+ MSign (Attn only) 72BTd2 + 12BT2d + O(BT2 + BTd) 52d3 52d3/P + MSign (All 2D) 72BTd2 + 12BT2d + O(BT2 + BTd) 90d3 90d3/P

†SVD FLOPs computed as 2mn min(m, n) + 11 min(m, n)3 per matrix (Trefethen & Bau III, 1997). For d × d matrix: 13d3; for d × 4d matrix: 19d3.

When the period is P, the amortized overhead per token is F/P. The throughput T(P) (tokens/s) can be modeled as:

T∞ 1 + r/P

, where r = F/f, (18)

T(P) =

and T∞ is the asymptotic throughput as P → ∞ (i.e., baseline without MSign overhead). Using the Sigma-40M throughput measurements from Table 3, we perform least-squares fitting by linearizing: 1/T(P) = (1/T∞)(1 + r/P). This yields:

T∞ ≈ 25,350 tokens/s, r ≈ 3.9. (19) The fitted model predicts:

P 10 100 1000 10000 Measured 18,236 24,559 25,082 25,270 Predicted 18,273 24,399 25,251 25,340

The close agreement validates the model. However, the fitted r ≈ 3.9 significantly exceeds the theoretical prediction. From Section 5.3, the FLOPs-based overhead ratio is R = 52d3/(72BTd2 · P) < 0.1% for typical configurations, implying rtheory ≪ 1.

This gap arises from the implementation factors discussed in Section 6.3: (1) all gather synchronization latency for distributed SVD computation, (2) disruption of FlashAttention kernel fusion and continuous CUDA stream execution, and (3) pipeline bubbles in distributed training. These factors introduce latency-dominated overhead that scales poorly with batch size, explaining why the effective r far exceeds the FLOPs-based prediction. Future work on asynchronous MSign execution and fused SVD kernels could potentially close this gap.

#### A.5. Application Period Analysis

- Figure 4 provides detailed training dynamics for different application periods P on NanoGPT-5M. The left figure shows training loss trajectories, while the right figure shows gradient norm evolution. Several observations emerge from this analysis:

[Figure 12]

[Figure 13]

Figure 4. Training dynamics under different MSign application periods on NanoGPT-5M. Left: Training loss comparison. Right: Gradient norm comparison. While all periods from P = 10 to P = 10000 eventually converge, P = 10000 exhibits noticeably higher gradient norm in step 20000 to 40000, indicating intermittent instability when MSign applications are too infrequent.

- • P = 10 and P = 100: Both show smooth, stable training dynamics with minimal variance in loss and gradient norm. The frequent MSign applications effectively maintain stable rank above critical thresholds throughout training.
- • P = 1000: Training remains stable but shows slightly increased variance compared to smaller P values. The longer intervals between MSign applications allow some transient stable rank decline, but recovery occurs before instability develops.
- • P = 10000: While training eventually converges (PPL 104.2), the dynamics exhibit clear signs of intermittent instability, the gradient norm shows periodic spikes and the loss curve has higher variance. This suggests that 10,000 steps is near the boundary where stable rank can decline sufficiently to trigger partial feedback loop activation before the next MSign intervention.

These findings justify our recommendation of P = 100 as the default: it provides a comfortable safety margin against instability while incurring negligible computational overhead.

### B. Proofs of Main Results

- B.1. Proof of Theorem 4.2 (Jacobian Product Norm Lower Bound) We first establish a key lemma for two-matrix products. Lemma B.1 (Alignment-Preserving Product Bound). For matrices A,B with SVD A = UASAVAT and B = UBSBVBT :

∥AB∥2 ≥ ∥A∥2∥B∥2 · |vA,T 1uB,1|, (20) where vA,1 is the top right singular vector of A and uB,1 is the top left singular vector of B. Proof: Let σA,1 = ∥A∥2 and σB,1 = ∥B∥2. We have:

∥AB∥2 ≥ ∥ABvB,1∥2 = ∥AσB,1uB,1∥2 = σB,1∥AuB,1∥2. (21) Expanding uB,1 in the basis of A’s right singular vectors:

∥AuB,1∥2 = ∥

i

σA,i(vA,iT uB,1)uA,i∥2 ≥ σA,1|vA,T 1uB,1|, (22)

where the inequality uses |vA,T 1uB,1| ≤ 1 and that {uA,i} is orthonormal. Since Align(A,B) = |vA,T 1uB,1| by definition, we obtain:

∥AB∥2 ≥ ∥A∥2∥B∥2 · Align(A,B). (23)

Applying this recursively to the Jacobian product:

□

∥J(L) ···J(1)∥2 ≥ ∥J(L)∥2∥J(L−1) ···J(1)∥2 · Align(L,L − 1) (24) ≥ ∥J(L)∥2∥J(L−1)∥2∥J(L−2) ···J(1)∥2 · a2 (25) ≥ ··· ≥ ML · aL−1 =

(aM)L a

. (26)

#### B.2. Proof of Theorem 4.4 (Stable Rank Controls Jacobian Norm: Linear Layer)

Consider a linear layer that computes hout = Whin, where W ∈ Rm×n is the weight matrix. The Jacobian of this transformation is:

∂hout ∂hin

= W, (27)

J =

since the linear map h  → Wh has constant derivative equal to W itself. The operator norm (also called spectral norm or 2-norm) of a matrix equals its largest singular value:

∥W∥2 = σ1(W) = max

∥x∥2=1

∥Wx∥2. (28)

This represents the maximum factor by which the matrix can stretch any unit vector, i.e., the worst-case signal amplification. From the definition of stable rank (Definition 1):

σi2 σ12

srank(W) = ∥W∥2F ∥W∥22

, (29)

= i

where σ1 ≥ σ2 ≥ ... ≥ 0 are the singular values of W. Let F = ∥W∥F denote the Frobenius norm, which is fixed by assumption. Substituting into the stable rank definition:

F2 ∥W∥22

. (30)

srank(W) =

Solving for the operator norm:

F2 srank(W) ⇒ ∥W∥2 =

F srank(W)

∥W∥22 =

. (31)

This formula reveals a fundamental trade-off: for a matrix with fixed total “energy” (Frobenius norm F), the operator norm, which determines signal amplification, is inversely proportional to the square root of stable rank.

Intuitively, stable rank measures how “spread out” the singular values are. When srank(W) ≈ rank(W) (high stable rank), the singular values are roughly uniform, and ∥W∥2 ≈ F/ rank(W) is relatively small. When srank(W) ≈ 1 (low stable rank), almost all energy is concentrated in the top singular value, and ∥W∥2 ≈ F, the maximum possible for given Frobenius norm. □

#### B.3. Proof of Theorem 4.5 (Jacobian Norm Bound: Attention Layer)

We analyze the Jacobian of a single-head attention layer. The output is Y = AHWV WO, where A = softmax(HWQW

T KHT

√dk ). The derivative of Y with respect to H is a 4th-order tensor. To manage this, we use the Fr´echet derivative, which is a linear operator DH(∆H) that approximates the change in Y for a small perturbation ∆H. Applying the product rule for Fr´echet derivatives:

##### DH(∆H) = [DHA(∆H)](HWV WO) + A[DH(HWV WO)(∆H)]. (32)

The second term is straightforward: DH(HWV WO)(∆H) = (∆H)WV WO. Taking operator norms:

∥DY(H)[∆H]∥2 ≤ ∥DA(H)[∆H]∥2∥HWV WO∥2 + ∥A∥2∥∆HWV WO∥2. (33) Dividing by ∥∆H∥2 and taking the supremum gives:

∥DY(H)∥2 ≤ ∥DA(H)∥2∥H∥2∥WV ∥2∥WO∥2 + ∥A∥2∥WV ∥2∥WO∥2. (34)

Bounding the value gradient pathway. For the second term, define f(H) = HWV WO. Since this is a linear function of

- H, its Fr´echet derivative at any point H is: Df(H)[∆H] = ∆HWV WO. (35)

The operator norm of this linear map is:

∥Df(H)∥2 = sup

∥∆H∥2=1

∥∆HWV WO∥2 = ∥WV WO∥2 ≤ ∥WV ∥2∥WO∥2. (36)

Bounding the attention gradient pathway. The first term requires analyzing how the attention matrix A changes with H. The attention matrix is computed as:

A = softmax

S √dk

#### , where S = QKT = HWQWKT HT. (37)

Note that S depends quadratically on H, making this pathway more complex to analyze. Softmax Fr´echet derivative. Consider the softmax function f(s) = softmax(s) = e

s

j esj applied row-wise to S. For the i-th row, let ai = f(si) where si is the i-th row of S. The Fr´echet derivative Df(si) : Rn → Rn is a linear map with matrix representation:

Df(si)[∆si] = (diag(ai) − aiaTi )∆si. (38) This can be verified by computing: for each component, ∂a∂si,k

= ai,k(δjk − ai,j).

i,j

Bounding the operator norm. Let Ji = diag(ai) − aiaTi . To bound ∥Df(si)∥2 = ∥Ji∥2, we examine its quadratic form. For any x ∈ Rn:

xTJix = xTdiag(ai)x − xTaiaTi x (39)

 

 

2

ai,jx2j −

(40)

=

ai,jxj

j

j

= E[X2] − (E[X])2 = Var(X), (41)

where we interpret j as a random index sampled with probability ai,j and X = xj as the corresponding random variable. Since variance is non-negative, Ji ⪰ 0.

For a positive semi-definite matrix, the spectral norm equals the largest eigenvalue, which is bounded by the trace:

∥Df(si)∥2 = λmax(Ji) ≤ tr(Ji) =

j

ai,j −

j

a2i,j = 1 − ∥ai∥22. (42)

Here we used j ai,j = 1. Factoring gives:

∥Df(si)∥2 ≤ 1 − ∥ai∥22 = (1 − ∥ai∥2)(1 + ∥ai∥2) ≤ 2(1 − maxai), (43) where the last inequality follows from ∥ai∥2 ≥ maxai and 1 + ∥ai∥2 ≤ 2.

Connecting to logit margin. The bound 1 − maxai measures how “spread out” the attention distribution is. When attention is sharply peaked, maxai ≈ 1 and ∥Df(si)∥2 is small. We can express this in terms of the logit margin γi = maxj Si,j − second maxjSi,j. By definition of softmax:

emaxs

emaxs

i

i

. (44)

≥

maxai =

k si,k esi,j

j esi,j

emaxsi + j̸=arg max

Since there are at most n − 1 non-maximal terms, each at most emaxs

i−γi:

emaxs

1

i

1 + (n − 1)e−γi . (45) Rearranging and using 1+xx ≤ min(x,1) for x ≥ 0:

maxai ≥

emaxsi + (n − 1)emaxsi−γi =

(n − 1)e−γ

i

##### 1 + (n − 1)e−γi ≤ min((n − 1)e−γ

##### ,1)△χ. (46)

1 − maxai ≤

i

Since the rows of A are computed independently, the Fr´echet derivative of the full softmax is block-diagonal and its operator norm is the maximum over rows:

∥D(softmax)(S)∥2 = max

∥Df(si)∥2 ≤ 2χ. (47)

i

Jacobian of attention logits. For the attention logits S(H) = HWQWKT HT, we compute the Fr´echet derivative. Using the product rule for bilinear forms:

DS(H)[∆H] = ∆HWQWKT HT + HWQWKT (∆H)T. (48) The operator norm is bounded by:

∥∆HWQWKT HT + HWQWKT (∆H)T∥2 (49)

∥DS(H)∥2 = sup

∥∆H∥2=1

≤ ∥WQWKT HT∥2 + ∥HWQWKT ∥2 (50) ≤ 2∥WQ∥2∥WK∥2∥H∥2. (51)

Combining via chain rule. Define g(S) = softmax(S/√dk) and recall S(H) = HWQWKT HT. By the chain rule for Fr´echet derivatives:

DA(H)[∆H] = Dg(S(H))[DS(H)[∆H]], (52) and the operator norms satisfy:

##### ∥DA(H)∥2 ≤ ∥Dg(S)∥2 · ∥DS(H)∥2. (53) From the softmax analysis, ∥Dg(S)∥2 = √1d

∂softmax

∂S 2 ≤ √2dχ

. Combining:

k

k

2χ √dk · 2∥WQ∥2∥WK∥2∥H∥2 =

4χ √dk ∥WQ∥2∥WK∥2∥H∥2. (54)

∥DA(H)∥2 ≤

Final bound. Substituting the bound on ∥DA(H)∥2 and noting that ∥V∥2 = ∥HWV ∥2 ≤ ∥H∥2∥WV ∥2:

4χ√∥Hdk∥22∥WQ∥2∥WK∥2∥WV ∥2∥WO∥2. (55)

∥DY(H)∥2 ≤ ∥A∥2∥WV ∥2∥WO∥2 +

□

Discussion: Connection to Stable Rank. Substituting ∥Wi∥2 = ∥Wi∥F/ srank(Wi) into the bound reveals that low stable rank in any of the projection matrices amplifies the Jacobian norm. In particular, for attention layers where V and O projections typically have lower stable rank than Q and K, the dominant contribution to gradient magnitude comes through the V-O pathway, which scales as ∥WV ∥2∥WO∥2 ∝ (srank(WV )srank(WO))−1/2.

- B.4. Proof of Theorem 4.6 (Jacobian Norm Bound: MLP Layer) Consider the forward pass through a two-layer MLP. Given input h ∈ Rd, we first compute the hidden representation:

z = hW1 ∈ Rd

, (56)

ff

- where W1 ∈ Rd×d

ff projects from hidden dimension d to feedforward dimension dff (typically dff = 4d). Next, we apply the element-wise activation function ϕ (such as GELU or SiLU):

a = ϕ(z) = [ϕ(z1),ϕ(z2),...,ϕ(zd

ff

)] ∈ Rd

ff

. (57)

Finally, we project back to the hidden dimension:

y = aW2 = ϕ(z)W2 ∈ Rd, (58)

- where W2 ∈ Rd

ff×d. To compute the Jacobian ∂∂hy, we apply the chain rule through this three-stage computation:

#### ∂y ∂h

∂y ∂a ·

∂a ∂z ·

∂z ∂h

. (59)

=

Each factor has a simple form:

- • ∂∂hz = W1: the Jacobian of a linear transformation is the weight matrix itself.

- • ∂∂az = diag(ϕ′(z)): since ϕ acts element-wise, its Jacobian is diagonal with entries ϕ′(zi).

- • ∂∂ya = W2: again a linear transformation.

Combining these:

∂y ∂h

= W1 · diag(ϕ′(z)) · W2. (60)

To bound the operator norm, we use the submultiplicativity property ∥AB∥2 ≤ ∥A∥2∥B∥2: ∂y ∂h 2

= ∥W1 · diag(ϕ′(z)) · W2∥2 ≤ ∥W1∥2 · ∥diag(ϕ′(z))∥2 · ∥W2∥2. (61)

The operator norm of a diagonal matrix is the maximum absolute value of its diagonal entries:

∥diag(ϕ′(z))∥2 = max

|ϕ′(zi)| ≤ sup z∈R

|ϕ′(z)| = Lϕ, (62)

i

where Lϕ is the Lipschitz constant of ϕ. For commonly used activations: GELU has LGELU ≈ 1.13 (achieved near z ≈ −0.75), and SiLU has LSiLU ≈ 1.1 (achieved near z ≈ 1.28).

Thus:

∂y ∂h 2 ≤ ∥W1∥2 · Lϕ · ∥W2∥2. (63)

To express this in terms of stable rank, recall from Definition 1 that srank(W) = ∥W∥2F/∥W∥22, which can be rearranged to:

∥W∥2 = ∥W∥F srank(W)

. (64) Substituting this for both weight matrices:

∥W1∥F srank(W1) ·

∥W2∥F srank(W2)

Lϕ∥W1∥F∥W2∥F srank(W1) · srank(W2)

. (65)

∥JMLP∥2 ≤ Lϕ ·

=

This shows that when stable ranks are low (denominator small) while Frobenius norms remain approximately constant or grow moderately over short training windows (numerator large), the Jacobian norm increases, potentially leading to gradient explosion. □

Discussion: Unified View Across Layer Types. Across all three layer types, linear, attention, and MLP, we observe the same fundamental pattern: the layer Jacobian norm is inversely related to the square root of the stable rank. This means that as training progresses and stable ranks collapse (Observation 1), the per-layer Jacobian norms tend to grow, even when Frobenius norms remain approximately bounded over moderate training windows. Combined with Jacobian alignment (Observation 2), this creates the conditions for exponential gradient growth characterized by Theorem 4.2.

#### B.5. Proof of Theorem 4.8 (Weight Gradient Norm Lower Bound)

We first state the formal version of Assumption 4.7. Assumption B.2 (Gradient Alignment Conditions (Formal)). For a deep network with L layers, we assume the following conditions hold for all layers i ∈ {1,...,L}:

(i) ∂vˆout(i)

≥ γ for all weight vectors vˆout(i) .

#### 1. Uniform local gradient lower bound: There exists γ > 0 such that ∂h

2

(i+1)

#### 2. Local-Jacobian alignment: Let J(i+1) = ∂h

∂h(i) be the local layer Jacobian at layer i + 1, with top right singular vector v1(i+1). The local gradient ∂h

(i) ∂vˆout(i)

satisfies:

∂h(i) ∂vˆout(i)

,v1(i+1) ≥ a. (66)

∂h(i) ∂vˆout(i)

2

(L)

be the last layer Jacobian, with top left singular vector u(1L). The loss gradient satisfies:

#### 3. Terminal alignment: Let J(L) = ∂h

∂h(L−1)

∂L ∂h(L)

,u(1L) ≥ a. (67)

∂L ∂h(L) 2

The proof proceeds by carefully tracking how the loss gradient propagates backward through the network. Starting from the decomposition (9):

∂L ∂vˆout(i) 2

=

∂h(i) ∂vˆout(i)

T

J(i+1:L)

T ∂L ∂h(L)

, (68)

2

where J(i+1:L) = J(L)J(L−1) ···J(i+1) is the cumulative Jacobian. The key insight is that the alignment conditions in Assumption B.2 are stated in terms of local layer Jacobians J(ℓ), which then propagate through the chain rule to yield bounds on the cumulative Jacobian J(i+1:L).

- Step 1: Terminal alignment implies loss gradient projects onto cumulative Jacobian. Let J(L) = U(L)S(L)(V(L))T be the SVD of the last layer Jacobian. By Assumption B.2.3, the loss gradient has alignment ≥ a with u(1L):

∂L ∂h(L)

,u(1L) ≥ a

∂L ∂h(L) 2

. (69)

Applying (J(L))T, the component along u(1L) maps to v1(L) with amplification σ1(L) = ∥J(L)∥2 ≥ M:

(J(L))T

∂L ∂h(L) 2 ≥ a · M ·

∂L ∂h(L) 2

. (70)

- Step 2: Recursive application through layers. By the Jacobian alignment condition (Definition 3.2 and the conditions used in Theorem 4.2), adjacent Jacobians have alignment ≥ a. This means the top right singular direction of J(ℓ+1) aligns with the top left singular direction of J(ℓ). Applying the alignment-preserving product bound recursively from layer L down to layer i + 1:

∂L ∂h(L) 2

∂L ∂h(L) 2 ≥ a · ∥J(i+1:L)∥2 ·

(J(i+1:L))T

. (71)

Moreover, by Theorem 4.2, the cumulative Jacobian norm satisfies:

(aM)L−i a

∥J(i+1:L)∥2 ≥

. (72)

#### Step 3: Local-Jacobian alignment at layer i. The result (J(i+1:L))T ∂h∂L

has its energy concentrated along the direction that propagated through the aligned chain. By Assumption B.2.2, the local gradient ∂h

(L)

(i) ∂vˆout(i)

is aligned with the top right

singular direction of the local Jacobian J(i+1). Since the Jacobian chain is aligned (Theorem 4.2), this direction is consistent with the dominant direction of the backpropagated signal.

(i) ∂vˆout(i)

Combined with the uniform lower bound ∂h

∂h(i) ∂vˆout(i)

T

J(i+1:L)

≥ γ, this yields:

2

T ∂L ∂h(L)

≥ a · γ · a · ∥J(i+1:L)∥2 ·

2

∂L ∂h(L) 2

. (73)

###### Step 4: Final bound. Substituting the lower bound from Theorem 4.2:

(aM)L−i a

∥J(i+1:L)∥2 ≥

, (74) we obtain:

(aM)L−i a ·

∂L ∂h(L) 2

∂L ∂h(L) 2

∂L ∂vˆout(i) 2

= aγ(aM)L−i ·

≥ a2γ ·

. (75)

□ Discussion: Gradient Decomposition Interpretation. The three-part decomposition in (9) has a clear interpretation:

- • ∂h∂L

(L)

: The loss gradient at the final layer output. This is the “signal” that backpropagation aims to transmit.

- • ∂h

(L)

∂h(i) = J(i+1:L): The cumulative Jacobian from layer i to layer L. This acts as the “transmission channel” whose gain is bounded by Theorem 4.2.

- • ∂h

(i) ∂vˆout(i)

: The local gradient at layer i. This represents how changes in the weight affect the layer’s output.

Discussion: Justification of Alignment Assumptions. All three assumptions in Assumption B.2 are stylized but capture a highly aligned regime that we empirically observe near failure in our experiments. Importantly, these assumptions are stated in terms of local layer Jacobians J(ℓ) rather than the cumulative Jacobian J(i+1:L). This is consistent with our theoretical framework where Theorem 4.2 establishes how local Jacobian properties (norms and alignments) propagate to yield bounds on the cumulative Jacobian.

The terminal alignment assumption excludes the degenerate case where ∂h∂L

(L) ≈ 0 or is nearly orthogonal to the last layer Jacobian’s dominant direction. The uniform local gradient lower bound excludes the trivial case where the weight has no effect on the output. The local-Jacobian alignment condition requires ∂h

(i) ∂vˆout(i)

to align with the local Jacobian J(i+1), which is consistent with the structured gradient flow we observe empirically in regimes with strong Jacobian alignment, but is not guaranteed in arbitrary settings.

- B.6. Proof of Theorem 4.9 (Total Gradient Norm Lower Bound) We aggregate the per-weight-vector bounds from Theorem 4.8 across all weights and layers.

Recall that a weight matrix W(i) ∈ Rm×n can be viewed as a collection of nw = n column vectors {vˆout,j(i) }n

j=1, where

w

each vector vˆout,j(i) ∈ Rm. The Frobenius norm of the gradient matrix at layer i equals the sum of squared 2-norms of the gradients with respect to each column:

2

nw

2

∂L ∂vˆout,j(i)

∂L ∂W(i)

. (76)

=

F

j=1

2

This follows from the definition of Frobenius norm: ∥A∥2F = i,j A2ij = j ∥aj∥22 where aj are the columns of A. From Theorem 4.8, we have a lower bound for each weight vector:

∂L ∂vˆout,j(i) 2

≥ aγ(aM)L−i ·

∂L ∂h(L) 2

. (77)

Squaring both sides (which preserves the inequality since both sides are non-negative):

∂L ∂vˆout,j(i)

2

≥ a2γ2(aM)2(L−i)

2

∂L ∂h(L)

2

. (78)

2

Note that the right-hand side is independent of the column index j. Summing over all nw columns:

∂L ∂W(i)

nw

2

=

F

j=1

∂L ∂vˆout,j(i)

2

nw

a2γ2(aM)2(L−i)

≥

j=1

2

∂L ∂h(L)

2

= nwa2γ2(aM)2(L−i)

2

∂L ∂h(L)

2

. (79)

2

Now we sum over all L layers. Since the bound for each layer is independent:

L

i=1

∂L ∂W(i)

2

F

Factoring out terms that do not depend on i:

L

i=1

∂L ∂W(i)

2

F

L

nwa2γ2(aM)2(L−i)

≥

i=1

∂L ∂h(L)

2

. (80)

2

≥ nwa2γ2

∂L ∂h(L)

L

2

(aM)2(L−i). (81)

·

2

i=1

The remaining sum can be evaluated by a change of variables. Let k = L − i. When i = 1, we have k = L − 1; when i = L, we have k = 0. Therefore:

L

(aM)2(L−i) =

i=1

L−1

(aM)2k = 1 + (aM)2 + (aM)4 + ··· + (aM)2(L−1). (82)

k=0

This is a geometric series with first term 1, common ratio r = (aM)2, and L terms. When r ̸= 1, the sum formula gives:

L−1

rL − 1 r − 1

(aM)2L − 1 (aM)2 − 1

rk =

. (83)

=

k=0

When aM > 1, we have (aM)2L ≫ 1 for large L, so the sum is approximately:

L−1

(aM)2L (aM)2 − 1

(aM)2k ≈

= O((aM)2L). (84)

k=0

This shows that the total gradient norm grows exponentially with network depth L. □

Discussion: Connection to Observations. From Observation 1, the stable rank of attention weights drops to near 1, implying ∥W∥2/∥W∥F ≈ 1 and thus high layer Jacobian norms. From Observation 2, Jacobian alignment a increases toward 1. Together, the product aM exceeds 1, triggering the exponential gradient explosion characterized by Theorem 4.9. This explains why training becomes unstable as stable rank declines and Jacobians align.

#### B.7. Proof of Theorem 4.10 (Low-Rank Propagation in Attention Layers)

The key insight is that gradients in neural networks are computed as outer products. For any weight matrix W in a linear layer y = Wx, the gradient with respect to W is:

∇WL =

∂L ∂y

xT = yx˜ T, (85)

where y˜ = ∂L∂y is the cohidden state (gradient of loss w.r.t. the output). This outer product structure implies a fundamental constraint: rank(yx˜ T) ≤ min(rank(y˜),rank(xT)) = 1 for single vectors. When we batch over B samples, the gradient becomes:

∇WL =

B

y˜(b)(x(b))T = Y˜ TX, (86)

b=1

where Y˜ ∈ RB×d

out and X ∈ RB×d

in. By the rank inequality for matrix products:

rank(Y˜ TX) ≤ min(rank(Y˜ T),rank(X)) ≤ min(rank(Y˜ ),rank(X)). (87)

For attention layers specifically, let’s trace through each gradient: Query gradient: The query projection is Q = HWQ. By the chain rule:

L = HT

∇WQ

∂L ∂Q

. (88)

This is a product of HT ∈ Rd×n (where n is sequence length) and ∂∂LQ ∈ Rn×d

k. If H has rank at most r, then HT also has rank at most r, and hence ∇WQ

L has rank at most r. The same argument applies to ∇WK

L and ∇WV

L, since they all have HT as the left factor:

∂L

∂Q(ℓ−1) ⇒ rank ≤ rank(H(ℓ−1)) ≤ r, (89) ∇WK

L = H(ℓ−1)T

∇WQ

∂L

L = H(ℓ−1)T

∂K(ℓ−1) ⇒ rank ≤ r, (90) ∇WV

∂L

L = H(ℓ−1)T

∂V(ℓ−1) ⇒ rank ≤ r. (91) Output projection gradient: For WO, the computation is y = (AV)WO. The gradient is:

∂L ∂y

L = (AV)T

∇WO

= (Attn Output)TH˜ (ℓ). (92)

If the cohidden states H˜ (ℓ) have rank at most r, then ∇WO

L has rank at most r.

This completes the proof: all four attention gradients have rank bounded by max(rank(H(ℓ−1)),rank(H˜ (ℓ))) ≤ r. □

#### B.8. Proof of Theorem 4.12 (Stable Rank Feedback Mechanism)

The gradient of the loss with respect to the weight matrix W in a linear layer hout = Whin is given by the outer product of the output gradient (cohidden state) and the input:

∇WL = E[h˜outhTin], (93) where h˜out = ∂h∂L

is the gradient backpropagated from later layers.

out

Since W = USVT (SVD), and we assume the input/output covariances are aligned with these singular vectors, we can project both h˜out and hin onto the singular bases. Define the projected coordinates:

αi = uTi h˜out, βj = vjThin. (94)

Then the gradient can be written in the singular vector basis as:

∇WL = E[h˜outhTin] =

E[αiβj]uivjT = UMVT, (95)

i,j

where Mij = E[αiβj] = Cov(uTi h˜out,vjThin) (assuming zero-mean projections for simplicity). Under the alignment assumption, the covariance matrix M is approximately diagonal because the input and output gradient are each concentrated along their respective top singular directions. Specifically:

##### Mii = Cov(αi,βi) = ρ E[αi2]E[βi2] = Cov(uTi h˜out,viThin), (96)

where λout,i = E[(uTi h˜out)2] and λin,i = E[(viThin)2] are the variances of the projections. The gradient descent update is W′ = W − η∇WL. Substituting the SVD forms:

##### W′ = USVT − ηUMVT = U(S − ηM)VT. (97)

Since M is approximately diagonal, the updated matrix W′ has the same singular vectors U,V (to first order), but with modified singular values.

Using first-order perturbation theory for SVD (valid when ηM is small relative to the gaps between singular values), the updated singular values are:

##### s′i = si − ηMii = si − ηCov(uTi h˜out,viThin). (98)

Since Cov(uTi h˜out,viThin) (negative correlation, typical in gradient descent where the gradient points toward decreasing loss), we have:

∆si = s′i − si = −ηCov(uTi h˜out,viThin) > 0. (99) Thus all singular values increase, which is consistent with the observed weight norm growth. Now we analyze how stable rank changes. Recall srank(W) =

i s2i s21 . Taking the differential:

2( i s2i)s1 ds1 s41

2 i si dsi s21 −

d(srank) =

2 s31 i

=

si(s1∆si − si∆s1) . (100)

Substituting ∆si = −ηCov(uTi h˜out,viThin):

2 s31 i

∆srank ≈

si(s1∆si − si∆s1)

2

s31 i −ηsi(s1Cov(uTi h˜out,viThin) − siCov(uT1 h˜out,v1Thin) . (101) To determine the sign of ∆srank, we analyze the term in parentheses. Under the assumption, the covariance satisfy:

=

Cov(uT1 h˜out,v1Thin) Cov(uTi h˜out,viThin)

>

s1 si

,∀1 < i ≤ n (102)

This gives:

s1Cov(uTi h˜out,viThin) > siCov(uT1 h˜out,v1Thin),∀1 < i ≤ n. (103) This means the term in parentheses is ≤ 0, and therefore:

2

s31 i −ηsi(s1Cov(uTi h˜out,viThin) − siCov(uT1 h˜out,v1Thin) ≤ 0. (104) Thus stable rank decreases under gradient descent with aligned input-output structure. □

∆srank =

