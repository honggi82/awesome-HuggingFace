# arXiv:2605.26895v1[cs.LG]26May2026

[Figure 1]

## Negligible in Size, Significant in Effect: On Scale Vectors in Large Language Models

### Mingze Wang1,2,†, Shuchen Zhu1,2, Yuxin Fang1, Binghui Li1,2, Kai Shen1, Shu Zhong1,†

1ByteDance Seed, 2Peking University †Corresponding authors

### Abstract

Normalization layers in modern large language models (LLMs) consist of a deterministic normalization operation and a learnable scale vector. While the normalization operation has been extensively studied, the scale vector remains poorly understood despite its ubiquitous use. In this work, we present a systematic study of scale vectors in LLMs from the perspectives of expressivity, optimization, and architectural structure. First, we show empirically that although scale vectors constitute only a negligible fraction of model parameters, removing them substantially degrades LLM pre-training. Our theory further shows that, in Pre-Norm architectures, scale vectors do not increase expressivity; instead, they improve optimization through a self-amplifying preconditioning effect on subsequent linear mappings. Second, we investigate the role of weight decay for scale vectors. By distinguishing Input-Norm and Output-Norm layers, we theoretically show that weight decay is beneficial for the former but harmful for the latter, due to their distinct roles in optimization and expressivity. Third, motivated by this understanding, we propose three lightweight and complementary improvements to scale vectors: branch-specific heterogeneity, improved placement around linear mappings, and magnitude-direction reparameterization. Both theory and experiments show that each improvement yields consistent gains. Finally, we combine these improvements into a unified scale-vector strategy and evaluate it through extensive LLM pre-training experiments on dense and mixture-of-experts models ranging from 0.12B to 2B parameters, across multiple optimizers and learning rate schedules, under industrial-scale token budgets. The unified strategy consistently achieves lower terminal loss than well-tuned baselines and exhibits more favorable scaling behavior, while adding negligible parameter and computational overhead.

Date: May 27, 2026 Correspondence: Mingze Wang at wangmingze.999@bytedance.com, Shu Zhong at zhongshu@bytedance.com

### 1 Introduction

Normalization layers are a fundamental component of modern deep learning and are crucial for the stable and efficient training of large language models (LLMs) [36]. In modern LLM architectures, normalization is typically implemented by RMSNorm [50], whose core structure consists of a deterministic normalization operation followed by a learnable scale vector γ. This simple normalize-then-scale structure has remained largely unchanged across modern LLMs.

The normalization operation itself has been extensively studied across deep learning. Compared with

BatchNorm (BN) in computer vision [14] and LayerNorm (LN) in sequence modeling [2], RMSNorm [50] modifies the normalization operation to improve simplicity and training stability in LLMs. However, its accompanying scale vector has largely retained its original form and received substantially less attention.

While the normalization operation is widely understood to stabilize training, the role of scale vectors remains unclear. Since scale vectors constitute only a negligible fraction of the total parameters, they are often treated as minor architectural details. However, do they truly have a negligible effect on LLM training? Answering this question is challenging, as it requires analyzing scale vectors jointly with other Transformer components from both expressivity and optimization perspectives.

In this work, we systematically investigate scale vectors in LLMs. We provide theoretical understanding of their mechanisms and propose scalable variants that improve LLM pre-training. Our contributions are summarized as follows:

- • Necessity of scale vectors. (Section 2.1) We empirically show that scale vectors substantially affect LLM pre-training despite their negligible parameter count. Our theoretical analysis further reveals that: (i) counterintuitively, scale vectors do not increase expressivity in Pre-Norm architectures; (ii) their benefit arises from a self-amplifying preconditioning effect on the subsequent linear mappings, which accelerates the training dynamics.
- • Weight decay of scale vectors. (Section 2.2) We study an unresolved practical question: whether weight decay (wd) should be applied to scale vectors. We classify normalization layers into Input-Norm and Output-Norm according to whether their scale vectors are immediately followed by linear maps. Our theoretical analysis shows that these two types have distinct mechanisms: (i) for Input-Norm scale vectors, including those in Pre-Norm layers, wd is beneficial as it induces balanced dynamics and controls Hessian sharpness, thereby accelerating and stabilizing training; (ii) for Output-Norm scale vectors, wd is harmful because these scale vectors determine expressivity, which wd can undesirably restrict. LLM experiments validate these theoretical insights.
- • Improving scale vectors. (Section 3) The above understanding motivates three complementary improvements to scale vectors, each with minimal computational and parameter overhead. (i) Heterogeneity. In Transformer blocks such as self-attention, one shared Pre-Norm feeds multiple projections, including query, key, and value; however, these branches exhibit distinct training dynamics. We therefore introduce branch-specific scale vectors to provide tailored self-amplifying preconditioners for different branches. (ii) Placement. In Pre-Norm architectures, scale vectors are consistently applied on the input side of the subsequent linear maps, inducing only row-wise self-amplifying preconditioning. We propose new placements that provably provide both row-wise and column-wise preconditioning, further accelerate training. (iii) Reparameterization. We propose magnitude-direction reparameterizations of scale vectors. Our theoretical analysis shows that these reparameterizations induce more anisotropic preconditioner and further accelerate training. Finally, we provide a unified preconditioning view of these scale-vector designs.
- • Extensive experiments. (Section 4) (i) We conduct systematic pre-training experiments on 0.12B Llama models to evaluate each strategy and its variants, including weight decay, heterogeneity, placement, and reparameterization (Sections 2.2, 3.1, 3.2, 3.3). Each strategy yields clear gains. (ii) We then evaluate a unified scale-vector strategy that combines these improvements. We conduct extensive LLM pre-training experiments on both dense and mixture-of-experts (MoE) models, with sizes ranging from 0.12B to 2B parameters, trained on high-quality pre-training corpus under industrial-scale token budgets. We further evaluate different optimizers, including AdamW and Muon, and different learning rate schedules. Across all settings, our unified strategy consistently achieves lower terminal loss and exhibits more favorable scaling behavior than well-tuned baselines, demonstrating its potential for improved scalability to larger models.

### 2 Understanding Scale Vectors

√

Notations. We denote the Hadamard product by ⊙. For x ∈ Rd, let Norm(x) =

dx/∥x∥. For vectors, ⟨·,·⟩ and ∥ · ∥ denote the Euclidean inner product and norm, respectively. For matrices, λmin(·), λmax(·), Tr(·), and

∥ · ∥F denote the smallest eigenvalue, largest eigenvalue, trace, and Frobenius norm, respectively. For v ∈ Rd, diag(v) ∈ Rd×d denotes the diagonal matrix with diagonal v. We use O(·) to hide problem-independent constants. N(µ,Σ) denotes the Gaussian distribution with mean µ and covariance Σ. For a time-dependent quantity h(t), write h˙ := dh/dt.

We focus on RMSNorm, which is widely used in modern LLMs:

RMSNorm(x;γ) = γ ⊙ Norm(x), x ∈ Rd. (1) Since BN and LN also contain scale vectors, similar mechanisms may apply beyond RMSNorm.

#### 2.1 Necessity of Scale Vectors

Negligible in model size. Most Transformer parameters are matrix weights in the feedforward network (FFN) and self-attention (Attn) blocks, each of size O(D2), where D is the hidden dimension. In contrast, each scale vector has only D parameters. For example, in the Llama-1B model considered in Section 4, the model has 1,028,065,024 parameters in total, whereas all scale vectors together contain only 80,640 parameters, accounting for merely 7.84 × 10−5 of the model size.

No additional expressivity in Pre-Norm architectures. In Pre-Norm architectures such as Llama [35] (Figure 2), each RMSNorm layer is immediately followed by a linear transformation. For example, the FFN block X + FFN(RMSNorm(X;γ)) takes the form

X + Wdown σ(Wgate(γ ⊙ Norm(X))) ⊙ Wup(γ ⊙ Norm(X)) . (2)

The same observation applies to Attn blocks and the final output layer. Since γ appears before linear maps, it can be absorbed into them: for any γ and linear map W2, choosing W1 = W2 diag(γ) yields W1Norm(x) ≡ W2(γ ⊙ Norm(x)). Thus, from the perspective of expressivity, scale vectors in Pre-Norm architectures are redundant. This raises a natural question:

Although negligible in size and expressively redundant, are scale vectors negligible in training?

Indispensable in practice. To answer this question, we conduct a simple ablation study by removing scale vectors. We train 0.12B Llama models; experimental details are provided in Appendix B.2.1. As shown in Figure 1, under the same peak learning rate (lr), the model with scale vectors consistently outperforms the model without them throughout training, reducing the terminal loss by approximately 0.028 and yielding a 1.4× token-efficiency gain. Even after retuning the peak lr for the model without scale vectors, its terminal loss remains approximately 0.015 higher.

###### Llama-0.12B, same lr

###### Llama-0.12B, best lr

3.4

3.4

w scale vectors

w/ scale vectors

w/o scale vectors

w/o scale vectors

validationloss

validationloss

3.3

3.3

w/o scale vectors, 1.4×tokens

w/o scale vectors, 1.2×tokens

3.2

3.2

3.1

3.1

2 4 6 8 10 12 14

2 4 6 8 10 12

tokens (billions)

tokens (billions)

Figure 1 Ablation of scale vectors in 0.12B Llama pre-training. (Left): same peak lr; (Right): separately tuned peak lr.

Theoretical study of training dynamics. These findings suggest that understanding scale vectors requires analyzing their effect on optimization dynamics. We therefore study scale vectors in Pre-Norm architectures through the following simplified but representative setting [30].

Setting 2.1. Let x ∼ N(0,Id) and f⋆(x) = W⋆Norm(x) with some nonzero W⋆ ∈ Rc×d.

We compare two student models under the squared loss. The model with scale vectors is f(x;Wf,γ) = Wf(γ ⊙ Norm(x)) with Lf(Wf,γ) = 12Ex∥f(x;Wf,γ) − f⋆(x)∥2; the model without scale vectors is g(x;Wg) = WgNorm(x) with Lg(Wg) = 12Ex∥g(x;Wg)−f⋆(x)∥2. Both objectives are optimized by gradient flow (GF): (W˙ f(t),γ˙ (t)) = −∇Lf(Wf(t),γ(t)) and W˙ g(t) = −∇Lg(Wg(t)). Following common practice, we initialize the scale vector as γ(0) = 1, while the remaining parameters have small magnitude. For clarity, we

take Wf(0) = Wg(0) = 0. Under this initialization, both models have the same initial loss. For brevity, write Lf(t) := Lf(Wf(t),γ(t)),Lg(t) := Lg(Wg(t)). We obtain the following result.

- Theorem 2.2 (Optimization advantage induced by scale vectors). Consider Setting 2.1. Then, for every t > 0, Lf(t) < Lg(t).

Theoretical insight. The proof of Theorem 2.2 is deferred to Appendix C.1. The key mechanism is that scale vectors transform the original gradient flow into a “self-amplifying preconditioned” flow. Specifically, define the effective parameter of f as Af(t) := Wf(t)diag(γ(t)), whose j-th column is af,j(t) := γj(t)wf,j(t) ∈ Rc. Its dynamics satisfy a˙ f,j(t) = (γj(t)2Ic + wf,j(t)wf,j(t)⊤)(w⋆,j − af,j(t)). By contrast, the model without scale vectors follows w˙g,j(t) = w⋆,j − wg,j(t). Thus, the scale vector induces the state-dependent preconditioner Pf,j(t) := γj(t)2Ic + wf,j(t)wf,j(t)⊤. Under the practical initialization above, gradient flow preserves the invariant γj(t)2 − ∥wf,j(t)∥2 = 1 and hence λmin(Pf,j(t)) ≥ γj(t)2 ≥ 1. Moreover, for every nonzero teacher column w⋆,j ̸= 0, the inequality is strict for all t > 0, yielding strictly faster loss descent.

Takeaway. Although scale vectors are negligible in parameter count and redundant in expressivity within Pre-Norm architectures, they accelerate training through self-amplifying preconditioning.

#### 2.2 Weight Decay for Scale Vectors

An unresolved weight-decay choice. We next study the training dynamics of scale vectors, with a particular focus on weight decay (wd). WD is a standard component of modern LLM training and is typically applied to matrix parameters in Transformers. For scale vectors, however, the appropriate choice remains unclear. Existing LLM implementations, including OLMo, nanoGPT, and Qwen [15, 27, 47, 48], differ in whether they apply wd to scale vectors. This motivates the question:

Should weight decay be applied to scale vectors?

Distinct types of scale vectors. We primarily focus on scale vectors in Pre-Norm architectures. Since Q/KNorm [7] and Output-Norm [32] are increasingly used in recent LLMs, such as Qwen3 and Gemma3 [33], we also include them in our study. Specifically, we consider Gemma3, which contains standard Pre-Norm, Q/K-Norm, and Attn/FFN Output-Norm layers. Importantly, RMSNorm layers at different locations play distinct structural roles. We therefore classify the RMSNorm layers in Gemma3 into two types, as illustrated in Figure 2:

- • Output-Norm layers: the RMSNorm layer not immediately followed by a linear transformation. This type includes Q/K-Norm and Attn/FFN OutputNorm.
- • Input-Norm layers: the RMSNorm layer immediately followed by a linear transformation. This type includes standard Pre-norm layers.

[Figure 2]

Figure 2 Comparison between Llama and Gemma architectures. Llama contains only standard PreNorm layers, whereas Gemma additionally includes Q/K-norm and Attn/FFN Output-Norm layers.

Theoretical insight. The scale vectors in these two types of RMSNorm layers play fundamentally different roles in expressivity and optimization, and should therefore be treated differently under wd:

- • Output-Norm scale vectors. These scale vectors directly parameterize the output of the corresponding submodule and therefore affect its expressivity. Applying wd to them shrinks the output, which can undesirably restrict expressivity and weaken the submodule relative to the residual stream. Thus, weight decay should be avoided for Output-Norm scale vectors.
- • Input-Norm scale vectors. In contrast, these scale vectors add no expressivity (Section 2.1); their role is primarily optimization-related. Applying weight decay to them keeps the parameterization balanced and

controls Hessian sharpness, thereby leading to potentially faster and more stable training. Theorem 2.3 provides a detailed analysis.

We now theoretically study whether wd should be applied to Input-Norm scale vectors. Unlike Section 2.1, which analyzes GF, we consider the more realistic setting of stochastic gradient descent (SGD) through its continuoustime SDE approximation. For clarity, we present the scalar-output model f(x;w,γ) = ⟨w,γ ⊙ Norm(x)⟩ with target f⋆(x) = ⟨w⋆,Norm(x)⟩ under Setting 2.1.

- Theorem 2.3 (Benefits of wd for Input-Norm scale vectors; informal). Consider training the above model by continuous-time SGD. Suppose wd λ > 0 is applied to w, and compare two choices for the Input-Norm scale vector γ: no wd (µ = 0) versus wd (µ > 0).

- • If µ > 0, then E∥γt∥2 < ∞ remains uniformly bounded over time. Consequently, key Hessian sharpness metrics, including λmax(∇2L), Tr(∇2L), and ∥∇2L∥F, also remain bounded.
- • If µ = 0, then E∥γt∥2 becomes unbounded along the trajectory. As a result, the same Hessian sharpness metrics diverge along a sequence of times.

Key mechanism. The formal statement and proof are deferred to Appendix C.2.1. Theorem 2.3 shows that, for Input-Norm scale vectors, wd keeps the parameterization pair (w,γ) balanced, prevents excessive growth of γ, and thereby controls Hessian sharpness during training.

Why sharpness matters. In stochastic optimization, the loss descent of SGD depends directly on Hessian sharpness metrics, such as λmax(∇2L), Tr(∇2L), and ∥∇2L∥F. Therefore, by preventing these quantities from diverging, wd on Input-Norm scale vectors leads to faster training and potentially more stable optimization, including the use of larger lr.

###### Gemma-0.5B, 10B tokens

Gemma-0.5B, 50B tokens

0.010

0.02

| |no w no w|d: none<br><br>d: Input-Norm|
|---|---|---|
| |no w|d: Output-Norm|
| | | |
| | | |
| | | |

relativevalidationloss

relativevalidationloss

0.005

0.01

0.000

0.00

no wd: none

−0.005

−0.01

no wd: Input-Norm

no wd: Output-Norm

−0.010

−0.02

6 8 10

10 20 30 40 50

tokens (billions)

tokens (billions)

Figure 3 WD is beneficial for Input-Norm scale vectors, but harmful for Output-Norm scale vectors.

We outline the main mechanism below and refer to Appendix C.2.2 for details. Let L(·) be a twice-differentiable loss function, and the SGD update can be written as θt+1 = θt − η (∇L(θt) + ξt), where the stochastic noise satisfies E ξt = 0. A second-order Taylor expansion yields the expected loss descent:

η2

2 ∇L(θt)⊤∇2L(θt)∇L(θt) GD term

E[L(θt+1)] = E L(θt) − η∥∇L(θt)∥2 +

η2 2

E ξt⊤∇2L(θt)ξt

+

noise term

+o(η2),

The GD term is contributed by deterministic GD, which is governed by the Hessian spectral norm λmax(∇2L(θt)), since ∇L(θt)⊤∇2L(θt)∇L(θt) ≤ λmax(∇2L(θt))∥∇L(θt)∥2; The noise term is induced by gradient noise ξt, different assumptions on the covariance E[ξtξt⊤] lead to different contribution. However, under a broad class of SGD noise modeling – including bounded variance, isotropic noise, Hessian-aligned noise, and others [3, 8, 10, 11, 25, 41, 43] – this term is consistently controlled by Hessian sharpness metrics such as λmax(∇2L(θt)), Tr(∇2L(θt)), and ∥∇2L(θt)∥2F.

Experimental justification. To validate our theoretical insights, we independently control wd for Input-Norm and Output-Norm scale vectors. We train 0.5B Gemma models for 10B/50B tokens using Muon optimizer [16]; experimental details are provided in Appendix B.2.2. As shown in Figure 3, applying wd to Input-Norm scale vectors consistently improves performance; whereas removing wd from Output-Norm scale vectors performs better than applying it. These results support the following principle.

Individual weight decay(IWD). Apply wd to Input-Norm scale vectors, but not to Output-Norm ones.

### 3 Improving Scale Vectors

In this section, we propose three methods to further exploit scale vectors. For each method, we first present the motivation and theoretical insight. In Section 3.4, we provide a unified view of these designs, and in Section 4, we empirically verify that each design consistently improves LLM pre-training.

#### 3.1 Heterogeneity of Scale Vectors

Shared scale vectors in Attn and FFN. We consider Pre-Norm architecture. In the Attn block, X + Attn(RMSNorm(X;γ)), a single Pre-Norm layer RMSNorm(X;γ) feeds all three projections:

Q = WQ(γ ⊙ Norm(X)), K = WK(γ ⊙ Norm(X)), V = WV (γ ⊙ Norm(X)) (3)

Thus, the scale vector γ is shared across the query, key, and value branches. The FFN block follows the same pattern: a single Pre-Norm output feeds both the gate and up projections.

Heterogeneous scale vectors (HG). Figure 4 shows that the query, key, and value matrices exhibit different growth and decay rates during training. As Section 2.1 shows, in Pre-Norm scale vectors primarily accelerate optimization through self-amplifying preconditioning. From this perspective, different branches should be equipped with separate scale vectors, allowing the induced preconditioners to adapt to branch-specific optimization dynamics. This motivates replacing the shared scale vector in (3) with branch-specific scale vectors γQ, γK, and γV :

Q = WQ(γQ ⊙ Norm(X)), K = WK(γK ⊙ Norm(X)), V = WV (γV ⊙ Norm(X)). (4)

An analogous modification applies to the FFN block. Specifically, we introduce separate scale vectors γgate and γup into the standard FFN in (2), yielding:

X + Wdown σ(Wgate(γgate ⊙ Norm(X))) ⊙ Wup(γup ⊙ Norm(X)) .

This heterogeneity does not increase expressivity; its role is purely optimization-related. Moreover, it introduces only O(d) additional parameters, which is negligible relative to the overall model size.

Llama-0.12B, Layer-3

100

WQ WK WV

90

Frobeniusnorm

80

70

60

50

2 4 6 8 10

tokens (billions)

#### 3.2 Placement of Scale Vectors

Figure 4 Dynamics of query, key, and value projections.

Standard input-side placement. We consider Pre-Norm architectures. As shown in (2) and (3), scale vectors in Attn or FFN blocks are placed before linear maps. The same holds for the RMSNorm layer before the output projection. Thus, Pre-Norm architectures consistently apply scale vectors on the input side of a linear map. This convention, however, is not obviously optimal for optimization.

Alternative placement for faster optimization. Section 2.1 shows that standard input-side scale vectors already accelerate optimization relative to removing them. A natural question is whether alternative placements can yield an even stronger optimization effects. The key observation is that the placement of a scale vector determines which coordinates are directly modulated. Input-side placement rescales coordinates before they are mixed by W, but the resulting output coordinates may still exhibit heterogeneous scales or anisotropic optimization geometry. Hence, the standard placement may not fully exploit the optimization benefits of scale vectors. Motivated by this observation, we consider three increasingly flexible designs.

- • AP: after-placement. The simplest modification moves the scale vector from the input side to the output side of the linear map:

AP: W(γb ⊙ Norm(x))  → γa ⊙ (WNorm(x)). (5)

Compared with the standard design, AP directly modulates output coordinates after linear mixing, and may therefore better match output-side anisotropy.

- • DP: dual-placement. AP moves the scale vector to the output side but removes input-side modulation. A more flexible alternative is to place scale vectors on both sides of the linear map:

DP: W(γb ⊙ Norm(x))  → γa ⊙ (W(γb ⊙ Norm(x))). (6)

Since scale vectors are negligible in size, this two-sided design incurs only a tiny parameter increase while controlling both input-side and output-side coordinates.

- • DNP: dual normalized placement. Although DP enables more flexible modulation around the linear map, the extra multiplicative interactions may reduce training stability. We therefore consider a normalized variant that inserts normalization between W and the output-side scale vector:

DNP: W(γb ⊙ Norm(x))  → γa ⊙ Norm(W(γb ⊙ Norm(x))). (7)

Compared with DP, DNP preserves two-sided modulation while explicitly normalizing the intermediate representation, which may stabilize optimization. For attention projections, normalization is applied separately to each head; for other projections, normalization is applied directly to the hidden states.

When applied to the query and key projections, DNP is equivalent to adding Q/K-Norm. For other matrices, it introduces additional normalization layers. Moreover, when using heterogeneous input-side scale vectors from Section 3.1, it is natural to also use heterogeneous output-side scale vectors.

The following theorem demonstrates that DP can further accelerate the optimization dynamics of scale vectors. Its proof is deferred to Appendix D.1.

- Theorem 3.1 (Instantaneous acceleration of DP). Consider the setting of Theorem 2.2. Let ϕ(x;Wϕ,γa,γb) = γa ⊙ (Wϕ(γb ⊙ Norm(x))) be the DP student model, and let Lϕ be its squared loss. We compare the standard model f and the DP model ϕ under gradient flow. Then, at the same effective state, DP is instantaneously at least as fast as the standard model: for any (Wf,γ) and(Wϕ,γa,γb) satisfying γ = γb and Wf = diag(γa)Wϕ, we have ddtLϕ ≤ ddtLf ≤ 0.

- Key mechanism. Theorem 3.1 establishes the instantaneous acceleration of DP at the same effective state as the baseline. The key mechanism is that DP induces both row-wise and column-wise self-amplifying preconditioning, leading to faster loss descent than the standard single-sided modulation. A finer trajectorylevel analysis, including strict acceleration in both early and global training phases, is deferred to Theorem D.1 in Appendix D.1.

#### 3.3 Reparameterization of Scale Vectors

Why parameterization matters. Another key question is how scale vectors should be parameterized. Even when different parameterizations represent the same functional class, they can induce substantially different optimization dynamics. This issue is relevant for scale vectors, which naturally contain two distinct components: a global magnitude and a directional unit vector. These components play different optimization roles. The magnitude controls the overall strength of modulation, whereas the direction determines how this modulation is distributed across channels. The standard parameterization entangles these two effects in a single vector, potentially leading to suboptimal conditioning. This motivates reparameterizing scale vectors to control magnitude and direction more explicitly.

Magnitude-direction Reparameterization. For nonzero scale vector γ ∈ Rd, we decompose it into its magnitude and direction, and optimize the two components separately. We consider two variants.

- • OR: reparameterization in original space. A natural approach is to separate magnitude and direction in the standard Euclidean geometry:

OR: γ  → β · Norm(α), (8) where β ∈ R controls the global magnitude, and Norm(α) ∈ Rd determines the direction.

- • ER: reparameterization in exponential space. While OR separates magnitude and direction in Euclidean coordinates, scale vectors act multiplicatively in the network. This suggests that a multiplicative parameterization may be more natural. We therefore consider

ER: γ  → eβ · eα−mean(α) (9) where eβ ∈ R controls the overall magnitude, and eα−mean(α) ∈ Rd determines the direction.

Although ER produces only positive entries and therefore appears to remove sign information, it does not reduce expressivity in our setting: since scale vectors always appear together with linear maps, the missing signs can be absorbed into the corresponding linear weights.

The following theorem demonstrates that OR can further accelerate optimization dynamics induced by scale vectors. Its proof is deferred to Appendix D.2.

- Theorem 3.2 (Optimization advantage of OR). Consider the setting of Theorem 2.2. Let ψ(x;Wψ,α,β) = Wψ(βNorm(α) ⊙ Norm(x)) be the reparameterized student model, and let Lψ be its squared loss. We compare the standard model f and the reparameterized model ψ under gradient flow, initialized as α(0) = 1,β(0) = 1,Wψ(0) = 0. Then, at the same effective state, OR is instantaneously at least as fast as the direct

parameterization: for any (Wf,γ) and (Wψ,α,β) satisfying Wf = Wψ and γ = βNorm(α), we have d dtLψ ≤ ddtLf ≤ 0.

- Key mechanism. Theorem 3.2 establishes the instantaneous acceleration of OR at the same effective state as the baseline. The key mechanism is that OR induces an anisotropic preconditioner that treats the magnitude and direction components of the scale vector differently. Under gradient flow, the effective dynamics of

γ := βNorm(α) in ψ satisfy γ˙ = −P∇γLf, where the implicit preconditioner is P = d·(γˆγˆ⊤)+ρ·(I −γˆγˆ⊤), where γˆ = γ/∥γ∥ and ρ = β2d/∥α∥2 ≈ O(1). Unlike the standard parameterization (P = I), this structure amplifies motion along the magnitude direction by a factor of d, aggressively accelerating scale adjustments, while preserving a stable O(1) descent scale for the orthogonal direction. A finer trajectory-level analysis, including strict acceleration in early and global training phases, is deferred to Theorem D.2 in Appendix D.2 and further supported by numerical simulations.

#### 3.4 A Unified Preconditioning View of Scale-Vector Designs

The preceding subsections introduce three different scale-vector designs: heterogeneity, placement, and reparameterization. We now show that they can be understood under a unified principle. For Pre-Norm scale vectors γ, the scale vector is immediately followed by a matrix W and therefore does not enlarge the expressivity of the corresponding linear transformation. Instead, it changes the parameterization of the effective matrix optimized during training. Consequently, these designs act as lightweight preconditioners.

Preconditioning via reparameterization. We first illustrate the general connection between reparameterization and preconditioning. Let p ∈ Rm denote an effective parameter that enters the loss L(p). If p is optimized directly, its gradient flow is:

dp dt

= −∇pL(p). (10) Now consider that p is parameterized by another variable q ∈ Rn with n ≥ m through a differentiable map p = Φ(q), Φ : Rn → Rm.

Under this reparameterization, by the chain rule, ∇qL(Φ(q)) = ∂Φ(∂qq)⊤∇pL(p). Therefore, gradient flow on the parameter q, ddqt = −∇qL(Φ(q)), induces the following dynamics on the effective parameter p = Φ(q):

⊤

dp dt

dq dt

∂Φ(q) ∂q

∂Φ(q) ∂q

∂Φ(q) ∂q

∇pL(p). (11)

= −

=

Comparing (11) with (10), we see that reparameterization induces a state-dependent preconditioned gradient flow in the effective parameter space, with preconditioner

⊤

∂Φ(q) ∂q

∂Φ(q) ∂q

. (12)

PΦ(q) :=

This preconditioner PΦ(q) is positive semi-definite and state-dependent.

Reparameterization forms of scale-vector designs. We now illustrate the above preconditioning view for our scale-vector designs. Consider a group of linear branches {Wc}c following a Pre-Norm layer, such as c ∈ {Q,K,V } in an Attn block, c ∈ {gate,up} in a FFN block, or a single linear branch such as the final output projection. For a branch matrix Wc ∈ Rd

2×d1, a general class of lightweight reparameterizations of the effective matrix can be written as

Wc  → diag(uc)Wc diag(vc) = Wc ⊙ (ucvc⊤), (13) where uc ∈ Rd

1. This form introduces a structured multiplicative scale field over the entries of Wc ∈ Rd

,vc ∈ Rd

2

2×d1. Notably, ucvc⊤ at most rank one, so the design uses only O(d1 + d2) additional parameters while modulating an O(d1d2)-dimensional matrix.

Table 1 summarizes how our scale-vector strategies fit into this unified form, including the standard scale vector, HG, AP, DP, OR, and ER. The only exception is DNP (diag(uc)Norm(Wc diag(vc)·)), which is not a purely reparameterization because it inserts an intermediate normalization to further stabilize the hidden representation. Although each row in Table 1 presents a single strategy, many strategies are complementary and can be combined to obtain richer scale-vector designs.

Table 1 Scale-vector designs as special cases of the unified reparameterization Wc  → Wc ⊙ (ucvc⊤).

Design Choice in the unified form (13) Preconditioning interpretation Standard uc = 1, vc = γ Input-side channel preconditioning HG uc = 1, vc = γc Branch-specific preconditioning AP uc = γa, vc = 1 Output-side channel preconditioning DP uc = γa, vc = γb Two-sided input/output preconditioning OR uc = 1, vc = β · Norm(α) Magnitude-direction anisotropic metric ER uc = 1, vc = eβ · eα−mean(α) Multiplicative/log-space scale metric

Comparison with adaptive optimizer preconditioning. Modern adaptive optimizers, such as RMSProp [34], Adam [17], KFAC [23], Shampoo [9], and Soap [37], introduce an optimizer-induced preconditioner Ft−1 in the raw parameter space. Recent variants further improve optimization by enhancing dynamics along flat directions of the preconditioned geometry [39, 40, 42, 52]. Under adaptive optimization, the gradient flow ddpt = −∇pL(p) is modified to ddpt = −Ft−1(p)∇pL(p), where Ft is estimated from gradient statistics accumulated during training. For example, RMSProp approximately uses Ft = diag( EMAt[(∇pL(pt))⊙2]). In contrast, a reparameterization Φ induces a reparameterization-induced preconditioner PΦ, as in (12). This preconditioner depends on the current model state, but not explicitly on historical gradients. Table 2 summarizes the differences. Thus, the two forms of preconditioning arise from fundamentally different mechanisms: reparameterization reshapes the local optimization geometry through the current model state, whereas adaptive optimizers adjust updates according to accumulated gradient statistics.

Table 2 Comparison between optimizer-induced and reparameterization-induced preconditioning.

Adaptive optimizer Reparameterization Φ

Preconditioner Ft−1(pt) PΦ(qt) = ∂Φ(∂qqt) ∂Φ(∂qqt)⊤ Determined by gradient history {∇pL(ps)}s≤t current model state qt

Combining these two mechanisms further alters the effective preconditioning geometry and may yield additional optimization gains. Intuitively, reparameterization provides a more suitable coordinate system, while adaptive optimizers exploit gradient statistics within that coordinate system. The experiments in Section 4 support this complementary effect.

### 4 Experiments

We evaluate our methods on LLM pre-training across architectures and model scales. The main configurations are summarized below; additional implementation details are provided in Appendix B.1.

Models and Data. We conduct experiments on two widely used LLM architectures: dense models (Llama [35]) and MoE models (LlamaMoE), with sizes from 0.12B to 2B parameters. All models are trained on a high-quality curated pre-training corpus.

Token Budget. Unless otherwise specified, the training budget is approximately 100 tokens per parameter for dense models and 100 tokens per activated parameter for MoE models. This budget substantially exceeds the Chinchilla-optimal regime [12] and is closer to industrial pre-training practice. For small-scale models, we use a total batch size of approximately 0.5M tokens; for large-scale models, we use a large total batch size of approximately 2M tokens.

Optimization. In most experiments, we adopt AdamW as the default optimizer [17, 22]. Following standard Llama pre-training practice [35], we set β1 = 0.9, β2 = 0.95, weight decay λ = 0.1, and gradient clipping threshold 1.0. We use the cos schedule: linear warmup to a peak lr lr_max, followed by cosine decay to the terminal lr lr_min. For each setting, we tune lr_max for the baseline model and use the same value for both the baseline and the corresponding improved model.

To validate our methods beyond this default setting, we additionally report results with the Muon optimizer [16] and the warmup-stable-decay (wsd) scheduler [13].

#### 4.1 Validation of Each Strategy

###### We first evaluate the strategies introduced in Sections 2 and 3: HG for scale-vector heterogeneity (Section 3.1); AP, DP, and DNP for scale-vector placement (Section 3.2); OR and ER for scale-vector reparameterization (Section 3.3); and IWD for individual weight decay of scale vectors (Section 2.2).

###### Weight Decay

###### Heterogeneity

###### Placement

###### Reparameterization

0.02

0.010

relativevalidationloss

relativevalidationloss

relativevalidationloss

relativevalidationloss

HG+DNP+OR HG+DNP+ER

baseline

HG

HG+DP

0.06

0.04

HG

HG+AP HG+DP HG+DNP

HG+DNP

0.005

0.01

HG+DNP+OR+IWD HG+DNP+ER+IWD

HG+DP+OR HG+DP+ER

0.04

0.02

0.000

0.02

HG+DNP+OR HG+DNP+ER

0.00

0.00

0.00

−0.005

−0.01

−0.02

−0.010

−0.02

4 6 8 10

4 6 8 10

4 6 8 10

4 6 8 10

tokens (billions)

tokens (billions)

tokens (billions)

tokens (billions)

Figure 5 Step-by-step validation of scale-vector strategies on Llama-0.12B. Each strategy class yields gains.

Step-by-step verification. All experiments in this subsection are conducted on the 0.12B Llama model, and the strategies are validated incrementally. The results are shown in Figure 5. (i) The Heterogeneity panel shows that heterogeneous scale vectors outperform shared scale vectors. (ii) The Placement panel shows that dual placement (DP) consistently improves performance, whereas dual normalized placement (DNP) does not. After placement (AP) improves early training but is later overtaken. (iii) The Reparameterization panel shows that applying magnitude-direction reparameterization, either OR or ER, improves performance. Among these variants, DNP+OR achieves the largest gain, outperforming DP+OR and DP+ER, and slightly outperforming DNP+ER. (iv) Although vanilla Llama contains only Pre-Norm layers, which are Input-Norm layers, applying DNP introduces both Input-Norm and Output-Norm scale vectors. This allows us to re-examine individual weight decay in this setting. As shown in the Weight Decay panel, OR+IWD achieves the lowest terminal loss. Overall, these results show that all four classes of scale-vector strategies provide gains, further supporting our theoretical understanding.

#### 4.2 Main Results of Unified Strategy

Building on the step-by-step validation in Section 4.1, we combine HG (Section 3.1), DNP (Section 3.2), OR (Section 3.3), and IWD (Section 2.2) into a unified scale-vector strategy.

In this subsection, we systematically compare this strategy with tuned baselines across dense and MoE models ranging from 0.12B to 2B parameters. Across all settings, our method consistently improves pre-training loss.

###### Dense-0.12B

###### Dense-0.25B

###### Dense-0.5B

3.5

baseline

baseline

baseline

3.1

2.9

ours

ours

ours

3.4

validationloss

validationloss

validationloss

2.8

3.0

3.3

2.7

2.9

3.2

2.6

2.8

3.1

2 4 6 8 10

5 10 15 20 25

10 20 30 40 50

tokens (billions)

tokens (billions)

tokens (billions)

###### Dense-1B

###### Dense-0.75B

###### Dense Scaling Laws

2.8

Lbaseline

baseline

baseline

3.0

Lours

ours

ours

validationloss

validationloss

2.7

2.5

###### loss

2.7

2.6

2.4

2.5

1.22×

2.4

25 40 55 70 85 100

15 30 45 60 75

0.1 0.5 1 2 4 8

tokens (billions)

tokens (billions)

PFLOP/s-day

###### Figure 6 Comparison between our unified scale-vector strategy and the Llama baseline on dense models of different sizes. Our strategy consistently achieves lower terminal loss across model sizes and scales better than the baseline.

MoE-0.5B (A-0.18B)

MoE-0.25B (A-0.12B)

MoE-0.8B (A-0.28B)

3.0

2.9

baseline

baseline

baseline

3.2

ours

ours

ours

validationloss

validationloss

validationloss

2.8

2.9

3.1

2.7

3.0

2.8

2.6

2.9

5 7.5 10 12.5 15

4 6 8 10

5 10 15 20 25 30

tokens (billions)

tokens (billions)

tokens (billions)

MoE-1B (A-0.36B)

MoE-2B (A-0.62B)

MoE Scaling Laws

2.8

Lbaseline

baseline

baseline

Lours

2.6

ours

ours

2.8

validationloss

validationloss

2.7

loss

2.5

2.6

2.6

1.25×

2.4

2.4

2.5

8 16 24 32 40

10 20 30 40 50 60

0.5 1 2 4 8

tokens (billions)

tokens (billions)

PFLOP/s-day

###### Figure 7 Comparison between our unified scale-vector strategy and the Llama-MoE baseline on MoE models of different sizes. Our strategy consistently achieves lower terminal loss across model sizes and scales more favorably than the baseline.

Results for Dense Models. Figure 6 compares our method with tuned Llama baselines on dense Llama models with 0.12B, 0.25B, 0.5B, 0.75B, and 1B parameters; 0.75B results are deferred to Appendix B.1.1. Our method consistently achieves lower terminal loss than the tuned Llama baseline and maintains its advantage throughout training, with the gap widening over time. The fitted scaling laws also suggest that these gains may persist at larger scales.

Results for MoE Models. Figure 7 compares our unified scale-vector strategy with tuned Llama-MoE baselines across MoE models with 0.25B (A0.11B), 0.5B (A0.18B), 0.8B (A0.28B), 1B (A0.38B), and 2B (A0.62B) parameters, where “A” denotes the number of activated parameters; 0.8B results are deferred to Appendix B.1.1. Across all settings, our method consistently achieves more than 0.02 lower terminal loss than the well-tuned MoE baseline. Moreover, the advantage is maintained throughout training, with the performance gap widening over time. Furthermore, our method also exhibits a more favorable scaling law than the baseline. In particular, the fitted loss-descent slope of our strategy is slightly steeper than that of the baseline.

#### 4.3 Compatibility with Muon and wsd

###### We further examine whether our strategy is compatible with other training configurations, including the Muon optimizer and the wsd lr schedule. Experimental details are provided in Appendix B.1.2.

###### Dense-0.12B, Muon

###### Dense-0.5B, Muon

###### MoE-0.5B (A-0.18B), Muon

baseline

baseline

baseline

3.3

ours

ours

ours

3.0

2.7

validationloss

validationloss

validationloss

3.2

2.9

2.6

3.1

2.8

2.5

2 4 6 8 10

10 20 30 40 50

5 7.5 10 12.5 15

tokens (billions)

tokens (billions)

tokens (billions)

- Figure 8 Results with the Muon optimizer. Our unified scale-vector strategy consistently improves over the corresponding baselines.

Muon [16, 20] has recently demonstrated strong efficiency and scalability in LLM pre-training. For the Muon baseline, we adopt the implementation of Liu et al. [20]. We conduct experiments on Dense-0.12B and MoE-0.5B models. The results are shown in the two left panels of Figure 8. Compared with the strong Muon baseline, our strategy still reduces the terminal loss by more than 0.015 (0.0156 for Dense-0.12B, 0.0167 for Dense-0.5B, and 0.0187 for MoE-0.5B) and maintains its advantage throughout training. We conjecture that this behavior is consistent with the AdamW results because Muon still optimizes scale vectors using AdamW, while our strategy primarily modifies scale vectors.

2 4 6 8 10

tokens (billions)

3.1

3.2

3.3

3.4

3.5

validationloss

Dense-0.12B, wsd

baseline

ours

5 7.5 10 12.5 15

tokens (billions)

2.8

2.9

3.0

validationloss

MoE-0.5B (A-0.18B), wsd

baseline

ours

Figure 9 Results with the wsd lr schedule. Our unified scale-vector strategy consistently improves over the corresponding baselines.

The wsd schedule has recently become a popular choice of LLM pretraining. We also conduct experiments on Dense-0.12B and MoE-0.5B models. The results are shown in

- Figure 9. Notably, the advantage of our strategy gradually increases throughout the LR stable phase and does not diminish in the decay phase. This suggests that our strategy may be particularly suitable for modern over-training pipelines that adopt wsd schedules.

#### 4.4 Overhead Analysis

We evaluate the parameter and computational overhead of our scale-vector strategy. Experimental details are provided in Appendix B.1.3.

Parameter overhead. Our strategies add only O(d) scale-vector parameters, whereas Transformer matrix parameters scale as O(d2); thus, the overhead is negligible. Nevertheless, we conduct a controlled ablation to isolate the effect of parameter count. Specifically, we compare against a widened Llama-0.12B baseline, denoted Llama-large, whose parameter count matches our largest variant, the unified strategy HG+DNP+OR+IWD (0.11985B vs. 0.11974B for the Llama-0.12B baseline). As shown in Figure 10, our method reduces terminal loss by 0.033, whereas Llama-large yields only a marginal gain. The parameter overhead further decreases at larger scales; for example, on Llama-1B, our method increases the parameter count by only 7.85 × 10−5.

Ablation

0.00

relativevalidationloss

−0.01

Llama-baseline (0.11974B)

Llama-large (0.11985B)

−0.02

ours (0.11985B)

−0.03

4 6 8 10

tokens (billions)

Figure 10 Ablation on parameter count.

The computational overhead is also negligible: our strategies introduce only elementwise operations, which scales as O(d), compared with the O(d2) cost of Transformer linear projections. To empirically measure computational overhead, we train a Llama-1B model with standard torch.compile. As shown in Table 3, our unified strategy incurs only minor increases in wall-clock time and memory usage. Results are averaged over 10 training steps after warmup.

Table 3 Runtime and memory overhead on Dense-1B model. Ratios are relative to the corresponding baseline.

|model<br><br>|wall-clock time (ms)|memory usage (MiB)|
|---|---|---|
|baseline our strategy|151 157 (1.04×)<br><br>|34263 34729 (1.01×)|

### 5 Conclusion

We present a systematic study of scale vectors in LLMs. Although scale vectors account for only a negligible fraction of model parameters, we show that they have a substantial impact on LLM pre-training. Our theory attributes this effect to optimization rather than expressivity, clarifies when weight decay should be applied, and motivates improved designs based on heterogeneity, placement, and reparameterization. Extensive experiments confirm that these designs consistently improve pre-training performance. Future work includes extending our analysis to broader architectures and studying how scale-vector design should evolve with model scale.

### Acknowledgment

We thank Guang Shi and Prof. Weinan E for helpful discussions.

### References

- [1] Sanjeev Arora, Nadav Cohen, and Elad Hazan. On the optimization of deep networks: Implicit acceleration by overparameterization. In International conference on machine learning, pages 244–253. PMLR, 2018.

- [2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

- [3] Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. SIAM review, 60(2):223–311, 2018.

- [4] Wenrui Cai, Defa Zhu, Qingjie Liu, and Qiyang Min. Seednorm: Self-rescaled dynamic normalization. arXiv preprint arXiv:2510.22777, 2025.

- [5] Chen Chen and Lai Wei. Post-layernorm is back: Stable, expressive, and deep. arXiv preprint arXiv:2601.19895, 2026.

- [6] Alex Damian, Tengyu Ma, and Jason D Lee. Label noise SGD provably prefers flat global minimizers. Advances in Neural Information Processing Systems, 34:27449–27461, 2021.

- [7] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In International conference on machine learning, pages 7480–7512. PMLR, 2023.

- [8] Yu Feng and Yuhai Tu. The inverse variance–flatness relation in stochastic gradient descent is critical for finding flat minima. Proceedings of the National Academy of Sciences, 118(9):e2015617118, 2021.

- [9] Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1842–1850. PMLR, 10–15 Jul 2018.

- [10] Jeff Z HaoChen, Colin Wei, Jason Lee, and Tengyu Ma. Shape matters: Understanding the implicit bias of the noise covariance. In Conference on Learning Theory, pages 2315–2357. PMLR, 2021.

- [11] Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in Optimization, 2(3-4): 157–325, 2016.

- [12] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 35:30016–30030, 2022.

- [13] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

- [14] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

- [15] Andrej Karpathy. NanoGPT. https://github.com/karpathy/nanoGPT, 2022.
- [16] Jordan Keller et al. Muon optimizer. URL https://github.com/KellerJordan/Muon?tab=readme-ov-file, 2024.

- [17] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

- [18] Qianxiao Li, Cheng Tai, and E Weinan. Stochastic modified equations and adaptive stochastic gradient algorithms. In International Conference on Machine Learning, pages 2101–2110. PMLR, 2017.

- [19] Zhiyuan Li, Tianhao Wang, and Sanjeev Arora. What happens after SGD reaches zero loss?–a mathematical framework. International Conference on Learning Representations, 2022.

- [20] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025.

- [21] Kangqiao Liu, Liu Ziyin, and Masahito Ueda. Noise and fluctuation of finite learning rate stochastic gradient descent. In International Conference on Machine Learning, pages 7045–7056. PMLR, 2021.

- [22] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

- [23] James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2408–2417. PMLR, 07–09 Jul 2015.

- [24] Takashi Mori, Liu Ziyin, Kangqiao Liu, and Masahito Ueda. Logarithmic landscape and power-law escape rate of SGD. arXiv preprint arXiv:2105.09557, pages 15959–15975, 2021.

- [25] Takashi Mori, Liu Ziyin, Kangqiao Liu, and Masahito Ueda. Power-law escape rate of sgd. In International Conference on Machine Learning, pages 15959–15975. PMLR, 2022.

- [26] Toan Q Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of self-attention. In Proceedings of the 16th international conference on spoken language translation, 2019.

- [27] Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. 2 olmo 2 furious. arXiv preprint arXiv:2501.00656, 2024.

- [28] Zihan Qiu, Zeyu Huang, Kaiyue Wen, Peng Jin, Bo Zheng, Yuxin Zhou, Haofeng Huang, Zekun Wang, Xiao Li, Huaqing Zhang, et al. A unified view of attention and residual sinks: Outlier-driven rescaling is essential for transformer training. arXiv preprint arXiv:2601.22966, 2026.

- [29] Maxim Raginsky, Alexander Rakhlin, and Matus Telgarsky. Non-convex learning via stochastic gradient langevin dynamics: a nonasymptotic analysis. In Conference on Learning Theory, pages 1674–1703. PMLR, 2017.

- [30] Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. International Conference on Learning Representations, 2014.

- [31] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- [32] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

- [33] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, et al. Gemma 3 technical report, 2025.
- [34] T. Tieleman and G. Hinton. Lecture 6.5—RmsProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012.
- [35] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- [37] Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade. Soap: Improving and stabilizing shampoo using adam for language modeling. In International Conference on Learning Representations, volume 2025, pages 93423–93444, 2025.

- [38] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. Deepnet: Scaling transformers to 1,000 layers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(10):6761–6774, 2024.

- [39] Jinbo Wang, Mingze Wang, Zhanpeng Zhou, Junchi Yan, Weinan E, and Lei Wu. The sharpness disparity principle in transformers for accelerating language model pre-training. International Conference on Machine Learning, pages 64859–64879, 2025.

- [40] Jinbo Wang, Mingze Wang, Jiaqi Zhang, Wei Wang, Peng Pei, Xunliang Cai, Weinan E, and Lei Wu. Gradpower: Powering gradients for faster language model pre-training. International Conference on Machine Learning, 2026.

- [41] Mingze Wang and Lei Wu. The noise geometry of stochastic gradient descent: A quantitative and analytical characterization. arXiv preprint arXiv:2310.00692, 2023.

- [42] Mingze Wang, Haotian He, Jinbo Wang, Zilin Wang, Guanhua Huang, Feiyu Xiong, Zhiyu Li, Weinan E, and Lei Wu. Improving generalization and convergence by enhancing implicit regularization. Advances in Neural Information Processing Systems, 2024.

- [43] Max Welling and Yee W Teh. Bayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pages 681–688. Citeseer, 2011.

- [44] Stephan Wojtowytsch. Stochastic gradient descent with noise of machine learning type. part II: Continuous time analysis. arXiv preprint arXiv:2106.02588, 2021.

- [45] Lei Wu, Mingze Wang, and Weijie Su. The alignment property of sgd noise and how it helps select flat minima: A stability analysis. In Advances in Neural Information Processing Systems, volume 35, pages 4680–4693, 2022.

- [46] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In International Conference on Machine Learning, pages 10524–10533. PMLR, 2020.

- [47] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

- [48] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [49] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12104–12113, 2022.

- [50] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

- [51] Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, and Zhuang Liu. Transformers without normalization. In Proceedings of the computer vision and pattern recognition conference, pages 14901–14911, 2025.

- [52] Shuchen Zhu, Rizhen Hu, Mingze Wang, Mou Sun, Xue Wang, Kun Yuan, and Zaiwen Wen. Accelerating llm pre-training through flat-direction dynamics enhancement. arXiv preprint arXiv:2602.22681, 2026.

- [53] Zhijian Zhuo, Yutao Zeng, Ya Wang, Sijun Zhang, Jian Yang, Xiaoqing Li, Xun Zhou, and Jinwen Ma. Hybridnorm: Towards stable and efficient transformer training via hybrid normalization. arXiv preprint arXiv:2503.04598, 2025.

- [54] Liu Ziyin, Mingze Wang, Hongchao Li, and Lei Wu. Parameter symmetry and noise equilibrium of stochastic gradient descent. Advances in Neural Information Processing Systems, 37:93874–93906, 2024.

## Appendix

- A Related Works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B Experimental Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- B.1 Experimental details for Section 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.2 Experimental details for Section 2 and 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- C Proofs in Section 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- C.1 Proofs in Section 2.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.2 Proofs in Section 2.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D Proofs in Section 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- D.1 Proofs in Section 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- D.2 Proofs in Section 3.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

### A Related Works

Normalization operations. The Normalization layer has long been recognized as a key ingredient for stable and efficient deep network training. Early work introduced Batch Normalization (BN) [14], which uses batch statistics and achieved great success in computer vision. Layer Normalization (LN) [2] later removed the dependence on batch statistics and became the standard choice for sequence modeling and Transformers. More recently, Root Mean Square Layer Normalization (RMSNorm) [50] further simplified LN by discarding mean centering, and has become the dominant normalization operation in modern LLMs due to its simplicity and efficiency. Beyond these standard designs, recent work has also explored alternatives to explicit normalization, such as Dynamic Tanh (DyT) [51], which replaces normalization layers with element-wise nonlinear transformations. Overall, prior work has focused primarily on the normalization operation itself, seeking better statistics, lower overhead, or normalization-free replacements. In contrast, our work focuses on the scale vector, a component retained across these designs but receiving much less theoretical and empirical attention.

Location of normalization layers. Another line of work studies where normalization should be placed within Transformer blocks. The original Transformer adopted Post-Norm, applying normalization after each residual addition. Subsequent work showed Pre-Norm improves training stability, which led it to become the default design in many modern LLMs [46]. However, Pre-Norm also introduces structural limitations, including weaker gradient propagation and reduced contribution from deeper layers [26]. Later studies explored combining the advantages of Pre-Norm and Post-Norm, aiming to stable optimization with loss gradient propagation. These works include [5, 38, 53]. In contrast to this line, which focuses on the location of normalization layers, our work studies the design and mechanism of the scale vectors within these layers.

Normalization layers with enhanced nonlinearity. More recent work has enriched normalization layers by introducing stronger nonlinear or adaptive modulation mechanisms, for example through gating modules or shallow feedforward networks [4, 28]. These methods increase the flexibility of normalization layers by making them more adaptive to different inputs. In contrast, rather than modifying normalization layers through additional nonlinear modules, we study the default scale vector itself, provide a theoretical understanding of its necessity and weight decay, and propose several simple but effective improvements. Moreover, several of our designs, including type-dependent weight decay, heterogeneous scale vectors, and dual-sided placements, are largely orthogonal to these methods and can potentially be combined with them.

- B Experimental Details All experiments are conducted on H100 80G GPUs.

#### B.1 Experimental details for Section 4

- B.1.1 Section 4.2 Models. We utilize two popular classes of LLM models for our pre-training experiments:

- • Llama. Llama [35] is a dense decoder-only Transformer architecture that uses Rotary Positional Encoding

(RoPE) [31], SwiGLU, RMSNorm, and a Pre-Norm design. We set dFFN = int(8dmodel/3). We pre-train Llama models ranging from 0.12B to 1B parameters. Detailed configurations are provided in Table 4.

- • LlamaMoE. LlamaMoE is a decoder-only mixture-of-experts architecture based on Llama. Each model uses 32 sparse experts, activates 4 sparse experts per token, and includes one shared expert. Following QwenMoE [47], the hidden dimension of the shared expert is int(8dmodel/3), whereas that of each sparse expert is int(2dmodel/3). We pre-train LlamaMoE models ranging from 0.25B to 2B parameters. Detailed configurations are provided in Table 5.

Token Budget. In this section, the total number of training tokens is set to approximately 100 times the number of model parameters for dense models, or 100 times the number of activated parameters for MoE models. This budget substantially exceeds the Chinchilla-optimal regime [12] and is closer to industrial pre-training practice. As shown in Tables 4 and 5, small-scale models use a sequence length of 1,024 and a batch size of 512, corresponding to a total batch size of approximately 0.5M tokens; large-scale models use a sequence length of 4,096 with a batch size of 512, corresponding to approximately 2M tokens per batch.

Optimization. In this section, we use AdamW as the default baseline optimizer [17, 22]. Following standard Llama pre-training practice [35], we set β1 = 0.9, β2 = 0.95, weight decay λ = 0.1, and the gradient clipping threshold to 1.0. We use the cos schedule: linear warm-up to a peak learning rate lr_max, followed by cosine decay to lr_min=lr_max/20. For each setting, we first tune lr_max for the baseline model and then use the tuned value for both the baseline and the corresponding improved model. The grid search for lr_max is performed over {6e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3}. The selected peak lr’s are reported in Tables 4 and 5.

- Table 4 Model configurations and tuned peak lr’s for dense models

Acronym Size dmodel nhead nlayer total batch size lr_max

Llama-0.12B 0.12B 768 12 6 1024 × 512 4e-3 Llama-0.25B 0.25B 1024 16 12 1024 × 512 2e-3 Llama-0.5B 0.48B 1280 20 18 4096 × 512 2e-3 Llama-0.75B 0.75B 1536 24 21 4096 × 512 2e-3 Llama-1B 1.03B 1792 28 22 4096 × 512 1e-3

- Table 5 Model configurations and tuned peak lr’s for MoE models.

Acronym Size Activated Size dmodel nhead nlayer total batch size lr_max

MoE-0.25B 0.25B 0.11B 640 10 6 1024 × 512 3e-3 MoE-0.5B 0.48B 0.18B 748 12 9 1024 × 512 2e-3 MoE-0.8B 0.80B 0.28B 960 15 10 4096 × 512 2e-3

- MoE-1B 1.06B 0.38B 1024 16 12 4096 × 512 1e-3

- MoE-2B 2.00B 0.62B 1280 20 15 4096 × 512 6e-4

##### B.1.2 Section 4.3

Muon optimizer. We follow the Muon setup of Keller et al. [16]: Muon is applied only to 2D matrix blocks in Transformer layers, while AdamW is used for all other parameters, including scale vectors, embedding layer, and the output layer. Additionally, following Liu et al. [20], we further use: (i) per-parameter update scaling, with learning-rate multiplier c = 0.2 max{m,n} for Muon blocks of shape Rm×n, so that the update RMS norm matches that of AdamW; (ii) Nesterov momentum with coefficient θmuon = 0.95; (iii) weight decay λ = 0.1. For Dense-0.12B and MoE-0.5B, we use the batch sizes reported in Tables 4 and 5. To ensure strong baselines, we tune the peak lr for Muon. The selected lr_max is 5e-3 for Dense-0.12B and 4e-3 for MoE-0.5B, both of which are larger than the corresponding tuned values for AdamW.

Wsd schedule. For AdamW with the wsd schedule [13, 49], we use a linear warmup to the peak learning rate lr_max, followed by a stable phase in which the learning rate remains at lr_max until 80% of the total training steps, and finally a linear decay to zero. For fairness, we tune the peak learning rate under the wsd schedule. The selected lr_max is 3e-3 for Dense-0.12B and 2e-3 for MoE-0.5B.

###### B.1.3 Section 4.1 and 4.4 The baseline model in Figure 5 uses the same training configuration as Llama-0.12B in Appendix B.1.1.

For the model-size ablation (Figure 10), we construct a larger Llama variant based on Llama-0.12B. Directly increasing the hidden dimension would require the dimension to remain compatible with both RoPE and the attention heads, leading to a parameter count much larger than the target. Therefore, we instead increase the FFN hidden dimension from dFFN to dFFN + 8, which closely matches the target parameter count. We retune the learning rate for this larger model, and the selected value remains 4e-3.

The experiments in Table 3 use the same training configuration as Llama-1B in Appendix B.1.1.

#### B.2 Experimental details for Section 2 and 3

##### B.2.1 Section 2.1

The baseline model (w/ scale vectors) in Figure 1 uses the same training configuration as Llama-0.12B in Appendix B.1.1.

The comparison model (w/o scale vectors) in Figure 1 is obtained by removing all scale vectors from Llama0.12B. For the peak lr max_lr, the left subfigure uses the same value as the baseline, 4e-3, whereas the right subfigure retunes this value and selects 2e-3. All other settings follow Appendix B.1.1.

The longer-training experiments only change the number of training steps. Specifically, the model w/o scale vectors is trained for 1.4× more steps in the left subfigure and 1.2× more steps in the right subfigure.

##### B.2.2 Section 2.2

For Figure 3, we train Gemma-0.5B models [33] on the same pre-training corpus as in 4 for approximately 10B and 50B tokens using the Muon optimizer [16].

- • Model. Gemma-0.5B uses the same dmodel, dFFN, nhead, and nlayer as Llama-0.5B in Appendix B.1.1. However, its normalization layers differ from those of Llama. We compare the two architectures in Figure 2.
- • Training. The Muon baseline follows the setup in Appendix B.1.2. We use the same batch size as Llama-0.5B in Table 4. We train for 5,000 steps, corresponding to approximately 10B tokens, denoted by 10BT, and for 25,000 steps, corresponding to approximately 50B tokens, denoted by 50BT. We first tune the peak lr for the baseline setting, which applies weight decay to all scale vectors, under the 50BT training budget. The selected max_lr is 4e-3, which is then used for all other experiments.

- B.2.3 Experimental details for Section 3.1 The model in Figure 4 uses the same training configuration as Llama-0.12B in Appendix B.1.1. C Proofs in Section 2
- C.1 Proofs in Section 2.1 Proof of Theorem 2.2. For clarity, we denote

##### W := Wg, U := Wf.

√

Let z = Norm(x) =

. Since x ∼ N(0,Id) is isotropic, z is uniformly distributed on the sphere of radius √

d ∥xx∥

2

d. Hence E[zz⊤] = Id. Therefore,

- 1

- 2∥W − W⋆∥2F.

Lg(W) = Similarly, writing

f(x;U,γ) = UDγz, Dγ = diag(γ) ∈ Rd×d, we have

- 1

- 2∥UDγ − W⋆∥2F.

Lf(U,γ) =

- Step I: dynamics of the model without scale vectors. Since

Lg(W) = its gradient flow is

- 1

- 2∥W − W⋆∥2F,

dW dt

= W⋆ − W. With W(0) = 0, this gives

W(t) = (1 − e−t)W⋆. Thus

Lg(W(t)) =

1 2

e−2t∥W⋆∥2F.

- Step II: dynamics of the model with scale vectors. Write the columns of U and W⋆ as U = [u1,...,ud], W⋆ = [w1⋆,...,wd⋆],

where uj,wj⋆ ∈ Rc. Then

UDγ = [γ1u1,...,γdud], and the loss decomposes as

d

- 1

- 2

∥γjuj − wj⋆∥22.

Lf(U,γ) =

j=1

For each coordinate j, define

- 1

- 2∥rj∥22.

rj = wj⋆ − γjuj, ℓj = Then

d

Lf(U,γ) =

ℓj.

j=1

The gradient-flow equations for the j-th column are

duj dt

dγj dt

= u⊤j rj. Therefore,

= γjrj,

d dt

dγj dt

duj dt

= uju⊤j rj + γj2rj. Equivalently,

(γjuj) =

uj + γj

drj dt

= −(γj2Ic + uju⊤j )rj. Hence

dℓj dt

drj dt

= −γj2∥rj∥22 − (u⊤j rj)2 = −2γj2ℓj − (u⊤j rj)2. We also have the conservation law

= rj⊤

dγj dt − 2u⊤j

duj dt

d dt

= 2γju⊤j rj − 2u⊤j (γjrj) = 0. Since γj(0) = 1 and uj(0) = 0, it follows that

γj2 − ∥uj∥22 = 2γj

γj(t)2 − ∥uj(t)∥22 = 1. Thus

γj(t)2 = 1 + ∥uj(t)∥22 ≥ 1. (14) Consequently,

dℓj dt

= −2γj2ℓj − (u⊤j rj)2 ≤ −2ℓj. Equivalently,

d dt

e2tℓj(t) = e2t

d dt

ℓj(t) + 2ℓj(t) = −e2t 2(γj(t)2 − 1)ℓj(t) + (uj(t)⊤rj(t))2 ≤ 0. (15)

Therefore,

ℓj(t) ≤ e−2tℓj(0).

- Step III: comparison of the errors. We now prove strict inequality whenever wj⋆ ̸= 0. For such a column, uj(0) = 0, rj(0) = wj⋆,

and hence

d dt

uj(0) = γj(0)rj(0) = wj⋆. Thus, for all sufficiently small s > 0,

uj(s) = swj⋆ + O(s2), rj(s) = wj⋆ + O(s). Therefore, for all sufficiently small s > 0,

uj(s)⊤rj(s) = s∥wj⋆∥22 + O(s2) > 0. Recalling (14), γj(s)2 ≥ 1. Therefore, the integrand

2(γj(s)2 − 1)ℓj(s) + (uj(s)⊤rj(s))2 in (15) is strictly positive on a nontrivial interval contained in (0,t) for every t > 0. Consequently, ℓj(t) < e−2tℓj(0), ∀t > 0.

For the model g, the corresponding column-wise loss is exactly

- 1

- 2

e−2t∥wj⋆∥22 = e−2tℓj(0). Hence

ℓg,j(t) =

ℓj(t) ≤ ℓg,j(t)

for every j, and the inequality is strict for every column satisfying wj⋆ ̸= 0. Since W⋆ ̸= 0, at least one such column exists. Summing over all columns gives

Lf(U(t),γ(t)) < Lg(W(t)), ∀t > 0. This proves the theorem.

| |
|---|

Comparison with Arora et al. [1]. A related line of work studies implicit acceleration by over-parameterization [1]. It shows that deep linear networks (DLNs), WL ···W1z, can accelerate optimization by inducing an implicit preconditioner, without increasing the expressivity of standard linear model. Our setting and results differs in three key aspects.

- • Parameter overhead. Scale vectors introduce substantially smaller overhead. A DLN fDLN(z;θ) =

WL ···W1z replaces a single d0 ·dL linear map Wz with Lℓ=1 dℓ−1 ·dℓ matrix parameters. In contrast, adding a standard input-side scale vector, W(γ ⊙ z), introduces only d0 additional parameters, which is negligible relative to the d0 · dL matrix parameters in linear map. It is therefore notable that such a lightweight parameterization can still improve optimization.

- • Preconditioner structure. The induced preconditioners have different structures. DLNs induce a global spectral preconditioner aligned with the singular directions of the matrix. In contrast, scale vectors induce a channel-wise preconditioner: each input channel has its own preconditioner, which acts directly on the corresponding dynamics. Comparing the regimes in which these two preconditioners are preferable is an interesting direction for future work.
- • Initialization and early-stage dynamics. The two parameterizations also differ in their initialization behavior. Scale vectors are typically initialized as 1, so adding them preserves the initial function. By contrast, DLNs introduce additional matrix factors; under standard small random initialization, the initial end-to-end differs from the baseline unless explicitly matched, and DLNs can be slower than the linear baseline model at early phase, as shown in [1]. In contrast, scale vectors can accelerate training from the beginning, whereas DLN-induced acceleration may emerge only after the end-to-end matrix has grown sufficiently.

#### C.2 Proofs in Section 2.2

##### C.2.1 Proof of Theorem 2.3

Note that

E[Norm(x)Norm(x)⊤] = Id. For notational convenience and without loss of generality, we consider the problem

- 1

- 2∥a − a⋆∥22, at = γt ⊙ wt ∈ Rd, a⋆ = w⋆. (16)

L(w,γ) = L(a) =

Here at is the effective parameter, while wt and γt are the two factors in the scale-vector parameterization. Stochastic differential equation (SDE). The continuous-time modeling of Stochastic gradient descent (SGD) can be written as follows [18]. Let q = ησ2. Then the SDE is

dwt = − γt ⊙ gt + λwt dt + √q dBtw, (17) dγt = − wt ⊙ gt + µγt dt + √q dBtγ, (18)

where

gt = ∇aL(at) = at − a⋆, and dBtw,dBtγ denote independent standard Brownian motions in Rd. By defining the error vector

ϵt = at − a⋆, the loss becomes

- 1

- 2∥ϵt∥22.

Lt =

##### Hessian sharpness metrics of the loss. Let

g = ∇aL(a) = a − a⋆, and define

Dw = diag(w), Dγ = diag(γ), Dg = diag(g). Then the Hessian of L with respect to the parameters (w,γ) is

Dγ2 DγDw + Dg DwDγ + Dg Dw2

∇2(w,γ)L =

. (19)

Since

DγDw + Dg = diag(2γ ⊙ w − a∗), the Hessian is block diagonal across coordinates. The i-th block is

γi2 2γiwi − a∗,i 2γiwi − a∗,i wi2

Hi =

Therefore,

∇2(w,γ)L = diag(H1,...,Hd). The two eigenvalues of Hi are

. (20)

- 1

- 2

γi2 + wi2 ± (γi2 − wi2)2 + 4(2γiwi − a∗,i)2 . (21) Consequently,

λi,± =

λmax ∇2L(w,γ) = max

λi,+. (22)

1≤i≤d

The trace is

d

γi2 + wi2 = ∥γ∥22 + ∥w∥22. (23) The squared Frobenius norm is

Tr ∇2L(w,γ) =

i=1

d

∇2L(w,γ) 2F =

i=1

γi4 + wi4 + 2(2γiwi − a∗,i)2 . (24)

Then the formal version of Theorem 2.3 is as follows: Theorem C.1 (Norm and sharpness Comparison). Consider λ > 0. The trajectory of (17)(18) satisfies supt≥0 E∥wt∥2 < ∞. Moreover,

- • If µ > 0, then supt≥0 E∥γt∥22 < ∞. Consequently, supt≥0 E[λmax(∇2L(tk))] < ∞, supt≥0 E[Tr(∇2L(tk))] < ∞, supt≥0 E∥∇2L(tk)∥F < ∞.

- • In contrast, if µ = 0, then supt≥0 E∥γt∥2 = ∞. Therefore, there exists a sequence tk → ∞ such that

k∥2 → ∞. Consequently, E[λmax(∇2L(tk))] → ∞, E[Tr(∇2L(tk))] → ∞, E∥∇2L(tk)∥F → ∞. Proof. Consider the coordinate-wise scale difference

E∥γt

st,i = γt,i2 − wt,i2 . By Itô’s formula, we have

dst,i = 2 λwt,i2 − µγt,i2 dt + 2√q γt,idBt,iγ − wt,idBt,iw . (25)

Notice that the Itô correction terms cancel because wt,i and γt,i are both one-dimensional coordinates with the same noise variance q.

We first show that E∥wt∥22 is bounded over t ≥ 0. Using Itô’s formula,

d dt

E∥wt∥22 = −2λE∥wt∥22 − 2E (γt ⊙ wt)⊤(at − a⋆) + dq

= −2λE∥wt∥22 − 2E a⊤t (at − a⋆) + dq. (26) For any a, we have

- 1

- 2∥a⋆∥22.

−2a⊤(a − a⋆) = −2∥a∥22 + 2a⊤a⋆ ≤ Therefore,

d dt

- 1

- 2∥a⋆∥22 + dq. (27)

E∥wt∥22 ≤ −2λE∥wt∥22 + By Grönwall’s inequality,

dq + 21∥a⋆∥22 2λ

E∥wt∥22 ≤ e−2λtE∥w0∥22 +

Thus E∥wt∥22 is bounded uniformly over t ≥ 0.

- • Without weight decay on γ (µ = 0): Taking expectation in (25), we get

1 − e−2λt . (28)

d dt

E[st,i] = 2λE[wt,i2 ] ≥ 0. Suppose, for contradiction, that there exists M > 0 such that

E∥γt∥22 ≤ M.

sup

t≥0

Combining this with

E∥wt∥22 < ∞, the occupation measures

sup

t≥0

T

1 T

P (wt,γt) ∈ A dt are uniformly tight. By the Krylov–Bogoliubov theorem, there exists an invariant probability measure π. Let A0 denote the generator of the SDE when µ = 0. From (25),

νT(A) =

0

A0(γi2 − wi2) = 2λwi2. Since π is invariant,

0 = A0(γi2 − wi2)dπ = 2λ wi2dπ. (29)

Since λ > 0, this implies

wi = 0 π-a.s. for every i. Hence

w = 0 π-a.s.

Now apply the generator to ∥w∥22. We have

A0∥w∥22 = dq − 2a⊤(a − a⋆) − 2λ∥w∥22. Since w = 0 π-a.s., we also have

a = γ ⊙ w = 0 π-a.s. Therefore,

A0∥w∥22 = dq π-a.s. Using invariance again,

0 = A0∥w∥22dπ = dq, which contradicts q > 0. Thus

E∥γt∥22 = ∞.

sup

t≥0

- • With weight decay on γ (µ > 0): By Itô’s formula,

d dt

E∥γt∥22 = −2µE∥γt∥22 − 2E (γt ⊙ wt)⊤(at − a⋆) + dq

= −2µE∥γt∥22 − 2E a⊤t (at − a⋆) + dq. (30) Using again

- 1

- 2∥a⋆∥22,

−2a⊤(a − a⋆) ≤ we obtain

d dt

- 1

- 2∥a⋆∥22 + dq.

E∥γt∥22 ≤ −2µE∥γt∥22 + Hence, by Grönwall’s inequality,

dq + 12∥a⋆∥22 2µ

E∥γt∥22 ≤ e−2µtE∥γ0∥22 +

1 − e−2µt .

Therefore,

E∥γt∥22 < ∞.

sup

t≥0

Finally, the estimates of Hessian statistics (λmax(∇2L),Tr(∇2L),∥∇2L∥F) follow (22) (23) (24) naturally.

| |
|---|

This conclusion is related to that of Ziyin et al. [54], who show that weight decay can induce balanced solutions in deep linear networks under stochastic dynamics. The key difference is that their analysis applies the same weight decay to all parameters, whereas our setting distinguishes between the two parameters, allowing us to isolate the effect of applying or omitting weight decay on γ.

##### C.2.2 How Hessian sharpness influences loss descent

We illustrate how Hessian sharpness metrics influence the loss descent of SGD. We begin by examining the loss dynamics. Let L(·) be twice differentiable. Consider the SGD update

θt+1 = θt − η (∇L(θt) + ξt),

where the gradient noise satisfies E[ξt] = 0. A second-order Taylor expansion gives:

η2 2

E[ξt⊤∇2L(θt)ξt] + o(η2), (31) where

E[L(θt+1)] = E[Q(θt)] +

η2

2 ∇L(θt)⊤∇2L(θt)∇L(θt), P(θt) := E[ξt⊤∇2L(θt)ξt],

Q(θt) := L(θt) − η∥∇L(θt)∥2 +

Here Q(θ) denotes the deterministic GD contribution, while P(θ) denotes the contribution from SGD noise.

- • GD term. By the standard descent lemma, the deterministic GD term is controlled by the maximum eigenvalue of the Hessian:

Q(θt) ≤ L(θt) − η 1 −

ηλmax(∇2L(θt)) 2 ∥∇L(θt)∥2.

- • Noise term. The noise contribution is P(θt) = E[ξt⊤∇2L(θt)ξt]. Therefore, different assumptions on

the SGD noise covariance E[ξtξt⊤] lead to different Hessian statistics controlling the noise-induced loss change.

- – Classical optimization analyses [11] often assume the bounded noise variance: E[∥ξt∥2] ≤ σ2, with σ being a fixed constant. It follows that

P(θt) ≤ σ2λmax(∇2L(θt)).

- – Another common assumption is the affine variance condition [3]: E[∥ξt∥2] ≤ σ02 + σ12∥∇L(θt)∥2. Then

P(θt) ≤ (σ02 + σ12∥∇L(θt)∥2)λmax(∇2L(θt)).

- – For regression with squared loss, several works show that the magnitude of SGD noise can be bounded by the loss value [8, 21, 24, 44]: E[∥ξt∥2] ≤ σ2L(θt). It follows that:

P(θt) ≤ σ2L(θt)λmax(∇2L(θt)).

- – A common simplification in SDE analyses of SGD is to approximate the gradient noise by isotropic Gaussian noise [29, 43]: E[ξtξt⊤] ≃ σ2Id. Then

P(θt) ≃ σ2Tr(∇2L(θt)).

- – In near-interpolation regimes, some works model the SGD noise covariance as approximately aligned with the Hessian [6, 10, 19]: E[ξtξt⊤] ≃ σ2∇2L(θ). In this case,

P(θt) ≃ ∥∇2L(θt)∥2F.

- – For squared loss in near-interpolation regimes, a more refined approximation is that the SGD noise

covariance aligns with both the loss value and the Hessian [25, 41, 45]: E[ξtξt⊤] ≃ 2L(θt)∇2L(θt). Then

P(θt) ≃ 2L(θt)∥∇2L(θt)∥2F.

Overall, across these noise assumptions, the SGD noise contribution to the loss is governed by Hessian sharpness metrics, such as

### D Proofs in Section 3

λmax(∇2L(θt)), Tr(∇2L(θt)), ∥∇2L(θt)∥2F.

- D.1 Proofs in Section 3.2 Proof Notations. We compare the input-only scale-vector model

f(x;U,γ) = U(γ ⊙ Norm(x)) with the input-output scale-vector model

ϕ(x;M,γa,γb) = γa ⊙ M(γb ⊙ Norm(x)) .

Here γ ∈ Rd is the input scale vector of the model f, whereas γa ∈ Rc and γb ∈ Rd are the output and input scale vectors of the model ϕ, respectively. Define

Dγ = diag(γ), Da = diag(γa), Db = diag(γb). The effective weights are

Af = UDγ, Aϕ = DaMDb. The corresponding losses are

- 1

- 2∥UDγ − W⋆∥2F,

Lf(U,γ) = and

1 2∥DaMDb − W⋆∥2F.

Lϕ(M,γa,γb) =

For a generic effective state A, define the residual

##### R = W⋆ − A.

- Theorem D.1 (Optimization advantage of input-output scale vectors). Consider the setting above. Under gradient flow, the following statements hold.

- • (i) Instantaneous acceleration at the same effective state. Suppose that f and ϕ are evaluated at parameter values satisfying UDγ = DaMDb,γ = γb. Then their instantaneous loss derivatives satisfy

d dtLϕ ≤

d dtLf ≤ 0.

- • (ii) Local strict acceleration from zero initialization. Let U(0) = 0, M(0) = 0, γ(0) = 1, γa(0) = 1, γb(0) = 1. If W⋆ ̸= 0, then, for all sufficiently small t > 0,

Lϕ(t) < Lf(t). More precisely,

c

- 2

- 3

t3

∥Wi⋆:∥42 + O(t4).

Lϕ(t) − Lf(t) = −

- • (iii) Global strict acceleration under a matching-support teacher. Assume W⋆ ≠ 0 and that its support forms a matching, namely

∥Wi⋆:∥0 ≤ 1 for every i, ∥W:⋆j∥0 ≤ 1 for every j. Under the standard initialization U(0) = 0, M(0) = 0, γ(0) = 1, γa(0) = 1, γb(0) = 1, we have Lϕ(t) < Lf(t), ∀t > 0.

- Proof of Claim (i) in Theorem D.1. For f, since

1 2∥UDγ − W⋆∥2F =

- 1

- 2∥R∥2F,

Lf =

the gradient-flow equations are

U˙ = RDγ, γ˙j = ⟨U:j,R:j⟩. Therefore,

d

dLf dt

= ∥RDγ∥2F +

⟨U:j,R:j⟩2.

−

j=1

Using U:j = A:j/γj, we obtain

dLf dt

−

d

d

γj2∥R:j∥22 +

=

j=1

j=1

⟨A:j,R:j⟩2 γj2

.

For ϕ, the gradient-flow equations are

M˙ = DaRDb, γ˙ia = ⟨Ri:,(MDb)i:⟩, and

γ˙jb = ⟨R:j,(DaM):j⟩. Hence

c

d

dLϕ dt

= ∥DaRDb∥2F +

⟨Ri:,(MDb)i:⟩2 +

⟨R:j,(DaM):j⟩2.

−

i=1

j=1

Because

Ai: γia

A:j γjb

, we get

(MDb)i: =

, (DaM):j =

d

c

c

d

⟨Ai:,Ri:⟩2 (γia)2

⟨A:j,R:j⟩2 (γjb)2

dLϕ dt

(γia)2(γjb)2Rij2 +

−

.

=

+

i=1

j=1

i=1

j=1

Using the matched-input-scale assumption γ = γb, subtracting the expression for Df proves

c

c

d

⟨Ai:,Ri:⟩2 (γia)2

dLf dt −

dLϕ dt

(γia)2 − 1 γj2Rij2 +

=

.

i=1

i=1

j=1

Consequently, due to the conservation law of gradient flow

d dt

(γia)2 − ∥Mi:∥22 = 0, we have

(γia(t))2 = ∥Mi:(t)∥22 + (γia(0))2 − ∥Mi:(0)∥22 ≥ 1, i = 1,...,c.

Therefore,

dLf dt −

dLϕ dt ≥ 0.

| |
|---|

- Proof of Claim (ii) in Theorem D.1. At t = 0, both effective weights are zero: Af(0) = Aϕ(0) = 0.

Moreover, since all scale vectors are initialized at one,

U˙ (0) = W⋆, M˙ (0) = W⋆. Thus

U(t) = tW⋆ + O(t2), M(t) = tW⋆ + O(t2). For the input-only model f, its input scale obeys

γ˙j = ⟨U:j,R:j⟩. Hence

1 2∥W:⋆j∥22t2 + O(t3).

γj(t) = 1 +

For the input-output model ϕ, the output scale obeys

γ˙ia = ⟨Ri:,(MDb)i:⟩. Using M(t) = tW⋆ + O(t2) and R(t) = W⋆ + O(t), we obtain

1 2∥Wi⋆:∥22t2 + O(t3).

γia(t) = 1 +

The input scale in ϕ satisfies the same second-order expansion as the input scale in f:

1 2∥W:⋆j∥22t2 + O(t3).

γjb(t) = 1 +

We now compare the instantaneous loss-decrease rates. Since

Af(t) = tW⋆ + O(t2), Aϕ(t) = tW⋆ + O(t2), and both residuals are

Rf(t) = W⋆ + O(t), Rϕ(t) = W⋆ + O(t), a direct Taylor expansion of the gradient-flow loss-decrease rates gives

Dϕ(t) − Df(t) = 2t2

c

∥Wi⋆:∥42 + O(t3).

i=1

Indeed, the leading difference is exactly the row-wise output-scale contribution in ϕ, while the input-scale contributions agree up to this order. Since

d dt

(Lϕ(t) − Lf(t)) = −(Dϕ(t) − Df(t)),

and Lϕ(0) = Lf(0), integration from 0 to t gives

- 2

- 3

t3

Lϕ(t) − Lf(t) = −

c

∥Wi⋆:∥42 + O(t4).

If W⋆ ̸= 0, then at least one row has positive norm, so

c

∥Wi⋆:∥42 > 0.

i=1

Therefore Lϕ(t) < Lf(t) for all sufficiently small t > 0.

| |
|---|

- Proof of Claim (iii) in Theorem D.1. Because W⋆ has matching support and the zero entries are initialized

at zero, the gradient-flow dynamics preserves the support. Indeed, if Wij⋆ = 0 and Mij(0) = Uij(0) = 0, then initially the effective weight at (i,j) is zero, hence the residual at (i,j) is zero. The gradient equations give

U˙ij = γjRij, M˙ ij = γiaRijγjb,

so Uij and Mij remain zero. Thus the system decomposes into independent scalar problems, one for each nonzero entry of W⋆.

It remains to prove the scalar claim. Let w⋆ ̸= 0. By symmetry, we assume w⋆ > 0. The input-only model has effective weight

af = uγ. Its gradient flow satisfies

u˙ = γ(w⋆ − af), γ˙ = u(w⋆ − af). The conservation law

γ2 − u2 = 1 holds. Therefore

γ2 + u2 = 1 + 4a2f, and hence

a˙f = 1 + 4a2f (w⋆ − af). For the input-output model, the effective weight is

aϕ = γamγb. The scalar gradient flow satisfies

m˙ = γaγb(w⋆ − aϕ), γ˙a = mγb(w⋆ − aϕ), γ˙b = γam(w⋆ − aϕ). The conservation laws are

(γa)2 − m2 = 1, (γb)2 − m2 = 1. Since γa(0) = γb(0) = 1, we have

γa(t) = γb(t) = 1 + m(t)2. Thus

aϕ = m(1 + m2). The effective dynamics is

a˙ϕ = (1 + 4m2 + 3m4)(w⋆ − aϕ). We compare the two scalar speed factors at the same effective weight a > 0. For f,

qf(a) = 1 + 4a2. For ϕ, let m = m(a) > 0 be the unique solution of

a = m(1 + m2).

Then

qϕ(a) = 1 + 4m2 + 3m4. Let y = m2 > 0. Since a2 = y(1 + y)2, we have

qf(a)2 = 1 + 4y(1 + y)2, while

qϕ(a)2 = (1 + 4y + 3y2)2. A direct calculation gives

qϕ(a)2 − qf(a)2 = y(9y3 + 20y2 + 14y + 4) > 0. Hence

qϕ(a) > qf(a), ∀a > 0. Both effective weights solve scalar ODEs of the form

a˙ = q(a)(w⋆ − a), a(0) = 0, where qϕ(a) > qf(a) for all a > 0. The scalar comparison principle implies

aϕ(t) > af(t), ∀t > 0. Both trajectories remain below w⋆, so

|w⋆ − aϕ(t)| < |w⋆ − af(t)|. Therefore

- 1

- 2

1 2

(w⋆ − aϕ(t))2 <

(w⋆ − af(t))2, ∀t > 0.

Applying this scalar comparison independently to every nonzero entry of W⋆, and summing the scalar losses, yields

Lϕ(t) < Lf(t), ∀t > 0.

| |
|---|

- D.2 Proofs in Section 3.3 Proof Notations. We compare the standard scale-vector model

f(x;U,γ) = U(γ ⊙ Norm(x)) with the OR-reparameterized scale-vector model

ψ(x;V ,α,β) = V (βNorm(α) ⊙ Norm(x)).

The corresponding effective weights are

Af = U diag(γ), Aψ = V diag(βNorm(α)). We denote their losses by Lf and Lψ, respectively.

- Theorem D.2 (Optimization advantage of OR reparameterization). Consider the setting in Theorem 2.2. We compare the standard model f with the reparameterized model ψ under gradient flow. The parameters are initialized as α(0) = 1,β(0) = 1,V (0) = 0; γ(0) = 1,U(0) = 0. Then the following statements hold.

- • (i) Instantaneous acceleration at the same effective state. For any (U,γ) in f and (V ,α,β) in ψ satisfying U = V and γ = βNorm(α), the OR reparameterization is at least as fast as the direct

scale-vector parameterization:

d dtLψ ≤

d dtLf ≤ 0.

- • (ii) Local strict acceleration from initialization. If d > 1, then the OR reparameterization is strictly faster in the early phase:

Lψ(t) − Lf(t) = −

- 2

- 3

(1 − 1/d)∥W⋆∥4Ft3 + O(t4).

- • (iii) Global strict acceleration under balanced teacher columns. Suppose d > 1 and ∥w⋆,1∥2 = ··· = ∥w⋆,d∥2 > 0, where w⋆,j denotes the j-th column of W⋆. Then the OR reparameterization is strictly faster throughout training

Lψ(t) < Lf(t),∀t > 0.

In addition, in Figure 11, we provide numerical validation of three claims in Theorem D.2. The simulations support the theoretically predicted acceleration induced by OR.

Verification of Claim (iii): Global Acceleration (Under Balanced Condition ||wj*||2 ≡ const)

Verification of Claim (i): dtd ψ ≤ dtd f ≤ 0 (Evaluated at identical effective states across time)

Verification of Claim (ii): Initial Acceleration Behavior (Perfect match with cubic t3 dynamics)

102

Model f: Standard Scale Vector

0.08

Empirical Difference (f − ψ)

Model ψ (OR Reparameterization) Model f (Standard) at SAME state Extra Dissipation (Advantage of Mψ)

Model ψ: OR Reparameterization Global Optimization Advantage (ψ < f)

102

Theoretical Asymptote: 23(1 − 1d)||W *||4Ft3 Higher-order Error (t4)

| |
|---|

DerivativeMagnitude(LossDissipationRate)

100

0.07

| |
|---|

100

0.06

10−2

PopuatonLoss(t)

0.05

LossDifference

10−2

10−4

0.04

10−4

10−6

0.03

10−6

0.02

10−8

0.01

10−8

10−10

0.00

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4

0.000 0.002 0.004 0.006 0.008 0.010 0.012 0.014

Time t (Gradient Flow Trajectory)

Time t (Gradient Flow Trajectory)

Time t (Early Phase)

Figure 11 Numerical validation of the claims in Theorem D.2.

- Proof of Claim (i) in Theorem D.2. Consider the model ψ(x;V ,β,α) = V βNorm(α) ⊙ Norm(x) ,

where

V ∈ Rc×d, α ∈ Rd, β ∈ R. Let

√

α ∥α∥2

q = Norm(α) =

d

, γ = βq. Then

ψ(x;V ,β,α) = V (γ ⊙ Norm(x)). Thus ψ can be viewed as a magnitude-direction reparameterization of the scale vector γ in f. As before, since

E[Norm(x)Norm(x)⊤] = Id, the loss of ψ is

- 1

- 2∥V Dγ − W⋆∥2F, Dγ = diag(γ).

Lψ(V ,β,α) = Write

V = [v1,...,vd], W⋆ = [w1⋆,...,wd⋆], and define

rj = wj⋆ − γjvj, sj = vj⊤rj, s = (s1,...,sd)⊤. Then

d

- 1

- 2

∥rj∥22.

Lψ =

j=1

The gradient-flow equations are

v˙j = γjrj,

β˙ = q⊤s, and

α˙ = βJq(α)⊤s, where

√

αα⊤ ∥α∥22

∂q ∂α

d ∥α∥2

Id −

.

Jq(α) =

=

√

Since α(0) = 1, we have ∥α(0)∥2 =

d. Moreover, α˙ is always orthogonal to α, so

√

∥α(t)∥2 =

d, ∀t ≥ 0. Therefore,

q(t)q(t)⊤ d

Jq(α(t)) = Pq(t) := Id −

. Consequently,

q˙ = βPqs. Since γ = βq, we obtain

γ˙ = β˙ q + βq˙ = qq⊤s + β2Pqs. Equivalently,

γ˙ = Mψs, Mψ := qq⊤ + β2Pq.

For comparison, in the original scale-vector model f, the corresponding gradient-flow equation is simply

γ˙ = s.

Thus, ψ replaces the Euclidean scale-vector dynamics by a preconditioned dynamics with preconditioner

Mψ = qq⊤ + β2Pq.

Moreover, ψ satisfies the conservation law

d dt

β2 − ∥V ∥2F = 0. Indeed,

d

d

d dt∥V ∥2F = 2

vj⊤v˙j = 2

γjvj⊤rj = 2βq⊤s,

j=1

j=1

while

d dt

β2 = 2ββ˙ = 2βq⊤s. Since β(0) = 1 and V (0) = 0, we have

β(t)2 − ∥V (t)∥2F = 1. In particular,

β(t)2 ≥ 1. Now the loss dissipation of ψ is

d

d dtLψ = −

j=1

γj2∥rj∥22 − (q⊤s)2 − β2∥Pqs∥22.

By contrast, for f at the same effective state, the loss dissipation is

d

d dtLf = −

γj2∥rj∥22 − ∥s∥22.

j=1

Decompose

q⊤s d

q, s⊥ = Pqs. Since ∥q∥22 = d, we have

s = s∥ + s⊥, s∥ =

(q⊤s)2 = d∥s∥∥22. Therefore,

(q⊤s)2 + β2∥Pqs∥22 = d∥s∥∥22 + β2∥s⊥∥22. Since d ≥ 1 and β2 ≥ 1, this implies

(q⊤s)2 + β2∥Pqs∥22 ≥ ∥s∥∥22 + ∥s⊥∥22 = ∥s∥22.

Hence, at the same effective state, the magnitude-direction parameterization ψ dissipates the loss at least as fast as the direct scale-vector parameterization f.

| |
|---|

- Proof of Claim (ii) in Theorem D.2. For both models, at initialization,

vj(0) = uj(0) = 0, rj(0) = wj⋆. Thus

vj(t) = twj⋆ + O(t2), uj(t) = twj⋆ + O(t2), and hence

sj(t) = vj(t)⊤rj(t) = t∥wj⋆∥22 + O(t2) = tρj + O(t2). For f, the scale-vector dynamics are

γ˙j = sj. Therefore,

- 1

- 2

γj(f)(t) = 1 +

ρjt2 + O(t3).

For ψ, at initialization we have

1 d

##### 11⊤.

q(0) = 1, β(0) = 1, Pq(0) = Id −

The induced scale-vector dynamics are

γ˙ = qq⊤s + β2Pqs. Thus, at leading order,

1 d

γ˙j(ψ)(t) = tρj + 1 −

tS + O(t2). Hence

1 d

- 1

- 2

γj(ψ)(t) = 1 +

S t2 + O(t3).

ρj + 1 −

Let

d dtLf(t), Dψ(t) := −

d dtLψ(t).

Df(t) := −

Using the loss-dissipation identities,

d

Df(t) =

j=1

γj(f)(t)

2

∥rj(f)(t)∥22 + ∥s(f)(t)∥22,

while

d

2

γj(ψ)(t)

∥rj(ψ)(t)∥22 + (q(t)⊤s(ψ)(t))2 + β(t)2∥Pq(t)s(ψ)(t)∥22.

Dψ(t) =

j=1

Substituting the expansions above gives

1 d

Dψ(t) − Df(t) = 2 1 −

S2t2 + O(t3).

Therefore,

t

Lψ(t) − Lf(t) = −

(Dψ(s) − Df(s)) ds

0

2 3

1 d

S2t3 + O(t4). This proves the claim.

= −

1 −

| |
|---|

- Proof of Claim (iii) in Theorem D.2. Under the balanced condition and the symmetric initialization q(0) = 1,

the symmetry is preserved along the gradient flow of ψ. Indeed, the vector s remains proportional to 1, and hence

Pqs = 0. Therefore,

q(t) = 1, γ(t) = β(t)1.

For both models, each column remains aligned with its teacher column. Thus we may write

uj(t) = pf(t)wj⋆ for f, and

vj(t) = pψ(t)wj⋆

for ψ. For f, the scalar dynamics are

p˙f = γf(1 − γfpf), γ˙f = ρpf(1 − γfpf), with

pf(0) = 0, γf(0) = 1. For ψ, the scalar dynamics are

p˙ψ = β(1 − βpψ), β˙ = dρpψ(1 − βpψ), with

pψ(0) = 0, β(0) = 1. Both dynamics have the same form

p˙ = η(1 − ηp), η˙ = kρp(1 − ηp),

where k = 1 for f and k = d for ψ. The corresponding invariant is

η2 − kρp2 = 1. Let

θ = ηp. Then

θ˙ = (η2 + kρp2)(1 − θ). Using the invariant, set

z = kρp2. Then

z(1 + z) kρ

η2 = 1 + z, θ2 = η2p2 =

. Therefore,

1 + 4kρθ2 − 1 2

. Hence the effective coefficient θ satisfies the closed scalar ODE

z =

1 + 4kρθ2 − 1 2

θ˙ = 1 + 2 ·

(1 − θ).

Equivalently,

θ˙ = 1 + 4kρθ2 (1 − θ). For ψ, k = d, while for f, k = 1. Since d > 1, we have

1 + 4dρθ2 > 1 + 4ρθ2 for every θ ∈ (0,1). By the scalar comparison principle,

θψ(t) > θf(t), ∀t > 0. The residual coefficient is 1 − θ, so

1 − θψ(t) < 1 − θf(t). Therefore,

dρ 2

dρ 2

(1 − θf(t))2 = Lf(t), ∀t > 0. This proves the claim.

(1 − θψ(t))2 <

Lψ(t) =

| |
|---|

