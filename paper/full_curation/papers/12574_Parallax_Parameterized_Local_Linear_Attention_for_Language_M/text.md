## arXiv:2605.29157v1[cs.LG]27May2026

### Parallax: Parameterized Local Linear Attention for Language Modeling

Yifei Zuo1, Dhruv Pai2, Zhichen Zeng3, Alec Dewulf2, Shuming Hu2, Zhaoran Wang1

1Northwestern University, 2Tilde Research, 3University of Washington

Large Language Models (LLMs) have become the central paradigm in artificial intelligence, yet the core computational primitive of attention has remained structurally unchanged. Local Linear Attention (LLA) is an attention mechanism derived from nonparametric statistics in the test-time regression framework. In contrast to prior research on efficient attention variants, LLA upgrades the local constant estimate in softmax attention to a local linear estimate, yielding provably superior bias-variance tradeoffs for associative memory. However, LLA has not been scaled in LLM pretraining due to computational and numerical stability concerns. We introduce Parallax, a parameterized Local Linear Attention that is scalable for LLMs. Parallax eliminates the numerical solver in LLA and learns an extra query-like projector that probes the KV covariance. We place Parallax within a family of attention mechanisms connected by the bandwidth, the probe construction and the affine structure. We propose a hardware-aware algorithm that increases the arithmetic intensity over FlashAttention, shifting attention into a more compute bound regime. Our prototype decode kernel matches or outperforms FlashAttention 2/3 across diverse batch sizes and context lengths. We pretrain Parallax at 0.6B and 1.7B scales and find consistent perplexity improvements throughout pretraining with gains that transfer to downstream benchmarks. The advantage persists under both parameter-matched and compute-matched controls, demonstrating a Pareto improvement. We perform careful pretraining ablations and identify a novel phenomenon whereby Muon unlocks the capacity of Parallax. To our knowledge, this is the first empirical demonstration of strong architecture-optimizer codesign for attention mechanisms in the architecture research literature.

Date: May 2026 Code: https://github.com/yifei-zuo/Parallax Correspondence: yifeizuo2029@u.northwestern.edu

[Figure 1]

[Figure 2]

dhruv@tilderesearch.com zhaoranwang@gmail.com

[Figure 3]

#### 1 Introduction

Large Language Models (LLMs) have become the central paradigm in artificial intelligence, powering advances in mathematical reasoning, code generation, multimodal processing and scientific discovery. Throughout the rapid progress of LLMs, Softmax Attention (Vaswani et al., 2017) has remained largely unchanged as the backbone of the Transformer architectures. A substantial body of work has sought efficient alternatives to Softmax Attention for long-context generation. For example, Linear Attention such as DeltaNet (Team et al., 2025; Yang et al., 2024b, 2025), and State Space Models (SSMs) such as Mamba (Gu and Dao, 2024) maintain constant-size recurrent states and achieve

subquadratic complexity. Despite the efficiency gains, such models consistently underperform Softmax Attention on in-context information retrieval (Arora et al., 2023; Bick et al., 2025; Jelassi et al., 2024), suggesting the underlying trade-off behind these design choices. The test-time regression framework (Wang et al., 2025a) unifies these attention mechanism designs by interpreting them as in-context regression solvers. Local Linear Attention (LLA) (Zuo et al., 2026) sharpens this perspective beyond Linear Attention by connecting the bias-variance theory with associative memory capacity, and shows that replacing the local constant estimator of Softmax Attention with a local linear estimator yields a strictly richer and more powerful predictor.

Although LLA has theoretical advantages and strong results on synthetic tasks, it has not yet been shown effective for large scale LLM pretraining. Specifically, the per-token conjugate gradient solve introduces both computation and I/O overhead and numerical sensitivity that are difficult to manage at scale. To bridge this gap, we propose Parallax, a parameterized LLA that preserves the local linear principle while being more efficient, scalable, and simpler to implement. It accepts an extra

- R matrix alongside the standard Q,K and V matrices, and learns to probe the KV covariance to improve the prediction. Notably, we demonstrate an optimizer-architecture interaction that was not previously recognized, whereby the correction branch in Parallax depends strongly on the optimizer geometry. Empirically, we find that the Muon optimizer (Jordan et al., 2024) is crucial for Parallax to demonstrate consistent improvements over Softmax Attention.

Contributions. To summarize, our contributions are:

- 1. Architecture. We identify the key challenges in scaling LLA to pretraining and derive Parallax to tackle these issues. We provide a unified interpretation that connects nonparametric attention mechanisms to their parametric counterparts, clarifying their design tradeoffs and complexity.
- 2. Efficiency. We analyze the I/O and compute complexity of Parallax and develop a hardwareaware streaming algorithm. Our custom decode kernel matches or outperforms FlashAttention 2/3 across a wide range of batch sizes and context lengths.
- 3. Experiment. We validate Parallax on synthetic tasks and on LLM pretraining at 0.6B and 1.7B scales, where it consistently improves perplexity and downstream accuracy over Softmax Attention. The improvement persists under both parameter-matched and compute-matched controls. We further characterize a strong optimizer-architecture interaction where Parallax shows substantial advantage under Muon, while the two are comparable under AdamW.

##### 1.1 Related Work.

Efficient Attention Mechanism. The quadratic computation and expensive I/O in Softmax Attention (Vaswani et al., 2017) has motivated a broad search for efficient alternatives. Linear Attention (Katharopoulos et al., 2020) removes the softmax operation, enabling recurrent inference with a constant-size state. Subsequent work has enriched this family through Retention (Sun et al., 2023), Gating (Yang et al., 2024a), Delta-Rules (Yang et al., 2024b, 2025) and Householder products (Siems et al., 2025). Similarly, SSMs such as Mamba (Dao and Gu, 2024; Gu and Dao, 2024; Lahoti et al., 2026) aim to parameterize linear recurrences with structured matrices for long-horizon recall (Gu et al., 2022a,b; Poli et al., 2023). FlashAttention (Dao, 2024; Dao et al., 2022; Shah et al., 2024) explores hardware-aware algorithm innovations, while keeping the underlying mechanism unchanged. Sparse Softmax Attentions (Gao et al., 2024b; Lu et al., 2025; Xiao, 2025; Yuan et al., 2025) and GQA, MLA (Ainslie et al., 2023; DeepSeek-AI et al., 2024) further incorporate the I/O aware design, making the attention more efficient in practice.

Attention as test-time learner. A growing body of work shows that attention mechanisms implicitly implement optimization steps to perform in-context learning (Ahn et al., 2023; Akyürek et al., 2023; Dai et al., 2023; Garg et al., 2022; Kirsch et al., 2024; Mahankali et al., 2024; von Oswald et al., 2023; Zhang et al., 2024). This perspective has motivated a series of attention variants designed around explicit test-time objectives, including Titans (Behrouz et al., 2025), MIRAS (Behrouz et al., 2026), MesaNet (von Oswald et al., 2026), and TTT (Sun et al., 2025b). The test-time regression framework (Wang et al., 2025a) unifies these designs by interpreting them as in-context regression solvers, from which LLA (Zuo et al., 2026) is derived.

Optimizers for LLMs. Adam(W) (Kingma and Ba, 2015; Loshchilov and Hutter, 2019) has long been the de facto choice of optimizer for all stages of the training pipeline. Subsequent work proposes optimizers that use more expressive curvature approximations (Gupta et al., 2018; Martens and Grosse, 2015; Vyas et al., 2024) but these methods have yet to gain traction, partially due to increased memory and compute costs. Recently, Muon (Jordan et al., 2024) has become a popular alternative to Adam(W) for optimizing matrix parameters in the hidden layers. Moonlight (Liu et al., 2025) adds RMS-matched updates and weight decay, making Muon more scalable. Dion (Ahn et al., 2025) explores cheaper ways to orthogonalize the gradient, and methods of reducing Muon’s communication cost in distributed settings. Further work explores more precise Newton-Schulz methods (Amsel et al.,

- 2025; Grishina et al., 2025), which have been shown to improve the downstream performance of Muon. These efforts have culminated in Muon’s application to training frontier-scale models (Team et al.,
- 2026; Zeng et al., 2026).

##### 1.2 Notation

For a matrix X, we denote ∥X∥F the Frobenius norm, ∥X∥2 the spectral norm, and use ⊙ to denote the Hadamard product between matrices. We use srank(X) to denote the stable rank of X, defined

as ∥X∥2F/∥X∥22. For a vector x, we use ∥x∥ to denote its Euclidean norm. To distinguish them from variables in the main text, matrix and vector variables in algorithm descriptions are denoted by X and x, respectively.

#### 2 Preliminary

##### 2.1 Local Linear Attention

Test Time Regression. The test-time regression framework (Wang et al., 2025a) interprets the attention mechanism as a regression solver over the KV pairs Di = {(kj,vj)}j≤i. The key vectors are treated as the training data points and value vectors are the labels. The attention function learns to predict on the test data point qi. Specifically, denote F(qi) the hypothesis space, Ω(f) the regularization and wij the weighting factor. The objective can be generally formulated as

fˆ(qi) = arg min

wij∥f(kj) − vj∥2 + Ω(f) . (1)

f∈F(qi) j≤i

Different attention designs correspond to different specifications of hypothesis spaces, objective functions and optimization methods. For example, Linear Attention family corresponds to the parametric linear estimators with F = {Wx + b} and context-independent weighting. MesaNet (von Oswald et al., 2026) chooses Ω(f) = λ∥W∥2F and solves the optimal ridge regression, while DeltaNet takes one step of stochastic gradient descent on the current KV pair without regularization. In contrast to the parametric approaches, Softmax Attention is nonparametric. It employs the Nadaraya-Watson

(NW) estimator (Bierens, 1988; Nadaraya, 1964; Watson, 1964) with kernel wij = exp(qi⊤kj/h). The hypothesis space simply contains constant functions F(qi) = {c} built for each query.

These design choices, particularly the choice of hypothesis space, fundamentally impact the associative memory capacity of each mechanism. Linear Attention suffers from the irreducible misspecification error, while Softmax Attention suffers from the boundary bias, which can be resolved by upgrading its constant function class to linear function class. Zuo et al. (2026) prove that by doing so the model can achieve strictly smaller integrated MSE. We provide a brief review of the main results in Theorem 2.1.

Theorem 2.1 (Bias-variance separation (Zuo et al., 2026)). Let (Xi,Yi)ni=1 be i.i.d. with Xi ∈ Rd supported on a bounded C2 domain D and Yi = f(Xi) + εi ∈ Rdy, E[εi | Xi] = 0. Let fˆGL, fˆNW, and fˆLL denote the Global Linear, Nadaraya–Watson (Local Constant), and Local Linear estimators with optimal bandwidths, respectively. Under Assumptions A.1–A.4, denote R(fˆ) := E D ∥fˆ(x)−f(x)∥2 dx the integrated mean squared error, then

R(fˆGL) ≫ R(fˆNW) ≫ R(fˆLL). (2)

The lower bound for fˆGL holds whenever f is not globally affine. The lower bound for fˆNW holds whenever f has sufficiently large normal gradient along ∂D (Assumption A.4).

Local Linear Attention. LLA fits a local linear estimator f ∈ F(qi) = {b+W(x−qi)} equipped with kernel weight wij = exp(qi⊤kj/h) and ridge regularization λ∥W∥2F. Let zij = kj − qi, ωi = j≤i wij, µi = j≤i wijzij, and Σi = j≤i wijzijzij⊤ + λI, LLA is the prediction of local linear estimator at qi:

wij(1 − zij⊤ρ⋆i) ωi − µ⊤ i ρ⋆i

oLLAi = fˆLL(qi) =

vj, ρ⋆i = Σ−i 1µi. (3)

j≤i

Intuitively, LLA is a query-centered second-order correction to Softmax Attention leveraging the geometry of key vectors around the query. It provides a better prediction when the keys are not uniformly distributed under the softmax geometry. LLA can also be interpreted as constructing query-dependent states through the kernel, in contrast to the global states in MesaNet. As shown in Figure 1, LLA can degenerate to both mechanisms by tuning λ and h:

oLLAi −−−→λ→∞ oSAi = softmax(qi⊤kj/h)vj, (4) oLLAi −−−−−−−−→h→∞

−1

oMesai =

kjkj⊤ + λI

qi. (5)

vjkj⊤

drop intercept

j≤i

j≤i

Challenges for LLM training with LLA. Despite its appealing theoretical properties and empirical advantages in synthetic tasks, LLA faces several challenges when scaled up to realistic language model training. In particular, the exact LLA forward requires solving a linear system Σix = µi for every query with a parallel conjugate gradient (CG) solver. It introduces several practical issues:

- • Intensive I/O. The CG iteration requires TLd memory access in the forward pass, dominating the 2Ld memory access of Softmax Attention. 1 ≤ T ≤ d is the iteration number.
- • Regularization-expressiveness tradeoff. Large λ ensures Σi ≻ 0 but drives ρ⋆i → 0, making LLA degenerate to Softmax Attention; small λ enables more expressiveness but risks ill-conditioning and instability. We find it nontrivial to balance the tradeoff in practical pretraining settings.
- • Low-precision incompatibility. The stability of CG is sensitive to the precision format, while modern hardware and computation primitives are increasingly shaped around reduced precision.

##### 2.2 Muon Optimizer

Muon is a novel optimizer for matrix parameters in the hidden layers. For a weight matrix Wt ∈ Rm×n with gradient Gt = ∇WL(Wt), Muon maintains a momentum buffer Bt = βBt−1 + Gt with B0 = 0. Letting the singular value decomposition (SVD) of Bt be Bt = UtStVt⊤, Muon forms the polar factor polar(Bt) = UtVt⊤, which is the nearest semi-orthogonal matrix in the Frobenius norm, and updates Wt according to:

Wt+1 = Wt − ηtUtVt⊤. (6)

Note for clarity, weight decay is omitted.

Computing Ut and Vt via SVD is prohibitively expensive. In practice, UtVt⊤ is approximated by Newton–Schulz iterations with precisely tuned matrix polynomials. These methods avoid a full SVD and can converge to a precise estimate of the polar factor in just a small number of steps (Jordan et al., 2024; Liu et al., 2025). This approach has the added benefit of exploiting fast GEMM subroutines on GPUs, making Muon hardware-aligned and feasible to use at scale.

Bernstein and Newhouse (2024) interpret the Muon update as steepest descent under the operator norm ∥ · ∥ℓ2→ℓ2, which for matrices coincides with the spectral norm. The polar factor has all singular values equal to one, and so Muon’s updates are guaranteed to have condition number of exactly one. Previous work has shown this strong conditioning of updates results in the underlying weight matrices themselves becoming better conditioned (Boreiko et al., 2025; Wang et al., 2026). By contrast, matrices trained with AdamW, can exhibit spectral collapse Arefin et al. (2026) whereby their effective rank shrinks rapidly over training. SignSGD and Adam can be interpreted as steepest descent under ∥ · ∥ℓ1→ℓ∞ geometry instead.

#### 3 Parallax Mechanism

##### 3.1 Parameterized Local Linear Attention

We first reformulate LLA as applying an additive correction to Softmax Attention with a projected KV covariance. Write pi = softmax(Kiqi/h), tij = ρ⋆i⊤zij and t¯i = Epi[tij]. The equation (3) can be rewritten as

wij(1 − tij) j≤i wij(1 − tij)

vj = oSAi − (1 + ηi)Σ(KVi) ρ⋆i, ρ⋆i = Σ−i 1µi, (7)

oLLAi =

j≤i

where Σ(KVi) = Epi[(vj − v¯i)(kj − k¯i)⊤], v¯i = Epi[vj], k¯i = Epi[kj] and ηi = t¯i/(1 − t¯i) is the boundary amplification factor. By Proposition 3.1, ηi is non-negative and quantifies the Mahalanobis distance from the query to the key center under pi. Intuitively, if ηi ≈ 0, the query is close to the weighted key center and the correction becomes pure covariance; if ηi ≫ 1, the query is close to the weighted key boundary and the correction is amplified to compensate for the boundary bias.

Proposition 3.1 (Boundary amplification is non-negative). Denote z¯i = Epi[zij] and Ai = ωiVarpi(zij)+ λI ≻ 0. If ρ⋆i = Σ−i 1µi, then t¯i = Epi[ρ⋆i⊤zij] ∈ [0,1), ηi = ωiz¯i⊤A−i 1z¯i ≥ 0 where z¯i = Epi[zij]. The proof is provided in Appendix B.

Parallax formulation. Building on the above reformulation, Parallax eliminates the per-query solve of ρ⋆i by learning a direct mapping from the layer input. Let xi be the input to the layer, we

λ → ∞

Softmax Attn ρ = 0

Parallax ρ = WR x

LLA ρ = Σ−1µ

∥ρ∥ → 0 parametrize

h → ∞ h → ∞ λ → ∞ h → ∞

Affine Linear Attn ρ = WR x o = v¯ − Sρ˜

Affine MesaNet ρ = H˜ −1q˜

Value Avg

ρ = 0 o = v¯

∥ρ∥ → 0 parametrize

o = v¯ − Sρ˜

drop intercept drop intercept

Linear Attn ρ = WR x o = Sρ

MesaNet ρ = H−1q o = Sρ

parametrize

- Figure 1 A family of attention mechanisms and their relationship. Rows are softmax weighted (top), uniform weighted with intercept (middle), and uniform weighted without intercept (bottom); Columns differs in how the probe ρi is obtained: zero (left), parametric (middle) and solved (right).

parameterize ρi = WRxi where WR ∈ Rdqk×d is a learnable projection matrix. Parallax additionally sets ηi = 0 to remove the boundary amplification, yielding the forward equation

wij(1 − tij + t¯i) j′≤i wij′(1 − tij′ + t¯i)

vj = oSAi − Σ(KVi) ρi, ρi = WRxi. (8)

oPLXi =

j≤i

Removing ηi is necessary because the parameterized ρi = WRxi is no longer the constrained solution of the exact LLA. Once that structure is broken, the mean score t¯i no longer admits its geometric interpretation and can take unbounded values. The scaling factor 1/(1 − t¯i) can therefore diverge as t¯i → 1 or flip sign when t¯i > 1, causing training instability. Equivalently, setting ηi = 0 corresponds to replacing tij with the centered statistics tij − t¯i in the scoring form of equation (7). The denominator in the scoring form of equation (8) reduces to ωi, which is bounded away from zero with the safe softmax implementation (Milakov and Gimelshein, 2018).

##### 3.2 Connection to Other Attention Mechanisms

To position Parallax relative to other attention mechanisms, we examine the wide bandwidth limit h → ∞ and strong regularization limit λ → ∞ (equivalently ∥ρi∥ → 0). We use the uniform-weighted running averages and second moments

v¯i = 1i j≤i vj, k¯i = 1i j≤i kj, Si = 1i j≤i vjkj⊤, Hi = 1i j≤i kjkj⊤ + λI, (9) together with the centered correspondences

q˜i = k¯i − qi, S˜i = Si − v¯ik¯i⊤, H˜i = Hi − k¯ik¯i⊤. (10) The connections are summarized in Figure 1.

Wide bandwidth limit. As h → ∞, the kernel weight wij = exp(qi⊤kj/h) → 1 uniformly in j, so the softmax weight degenerates to the uniform distribution pij → 1/i, and ωi → i. Local softmax-weighted statistics become global running averages, and the three nonparametric mechanisms reduce to the affine variants with recurrent states given by the uniform averages,

oSAi −−−→h→∞ v¯i, ρi = 0 (Value Averaging), (11) oPLXi −−−→h→∞ v¯i − S˜iρi, ρi = WRxi (Affine Linear Attention), (12) oLLAi −−−→h→∞ v¯i − S˜iρ⋆i, ρ⋆i = H˜i−1q˜i (Affine MesaNet). (13)

All three share the output template and differ only in the probe. This template is the empirical OLS regression of v on k with intercept v¯i, evaluated at the query. The three mechanisms correspond to evaluating it through a zero, learnable, and fully solved ρi respectively. We refer to the corresponding attention mechanisms as Value Averaging, Affine Linear Attention and Affine MesaNet.

The standard forms of Linear Attention and MesaNet drop the intercept. Algebraically, this is the same as setting v¯i = k¯i = 0 in the affine forms above, which collapses the centered moments to their raw counterparts. The framework also clarifies the dual role of the query across the family. In nonparametric mechanisms, qi shapes the kernel weights, defining where attention concentrates. In the Linear Attention family, what is conventionally considered the query is in fact the probe ρi, a directional readout from the recurrent state that can be completely determined by other statistics as in MesaNet or LLA.

Strong regularization limit. As λ → ∞ or ∥ρi∥ → 0, the probe term is suppressed and the intercept dominates the output. Parallax and LLA degenerate to Softmax Attention, while Affine MesaNet and Affine Linear Attention degenerates to the Value Averaging mechanism. Under this limit, the Linear Attention and MesaNet reduce to nothing for the whole term vanishes. The same parametrization axis explains the relationship between Parallax and LLA, just as MesaNet differs from Linear Attention by probe preconditioning.

Magnitude tension in the affine structure. Parallax and Affine Linear Attention inherit an additive structure in which the output is a sum of an intercept and a linear evaluation through ρi. Since the probe ρi = WRxi is parametric rather than an optimal solve, the strength of the linear evaluation relative to the intercept is not guaranteed. Directionally, only the component of ρi aligned with the exact solve ρ⋆i (Σ−i 1µi for Parallax, H˜i−1q˜i for Affine Linear) remains functional. The orthogonal component is unidentifiable and does not contribute toward the correction. Likewise, the norm of the probe no longer respects ∥ρ⋆i∥. In contrast, the magnitude of the intercept term only depends on the weighted averages of v.

As a result, a poorly aligned or norm-suppressed probe vector renders the covariance correction term functionally inert in prediction, and Parallax collapses in effect toward its Softmax Attention baseline regardless of the affine structure nominally available. Both the alignment and the norm of the probe depend heavily on optimizer choice, which we analyze empirically in Section 4.3.

##### 3.3 Streaming Algorithm

Parallax inherits the streaming structure of FlashAttention (FA) (Dao, 2024; Dao et al., 2022; Shah et al., 2024) with one additional covariance branch. In order to stream the computation of equation (8)

independent

###### P1=QK S2=RK

softmax P2=P1⊙S2

O1=P1V O2=P2V

Tensor Core

Tensor Core

SFU

CUDA Core

(a) Dependency graph and hardware mapping.

Compute-matched PLX

I/O-matched PLX

- 0.5

- 1

- 2

[Figure 4]

[Figure 5]

[Figure 6]

2048

1024

512

256

128

64

###### 1.54 ×

###### 1.14 ×

32

###### speedup

###### speedup

16

8

4

- 1

- 2

12825651210242048409681921638432768

12825651210242048409681921638432768

(b) PLX-CuTeDSL vs best(FA2,FA3) in decoding.

- Figure 2 Figure 2a: Operator dependency and hardware unit assignment for the Parallax forward. Figure 2b: Decoding speedup of Parallax kernels in I/O matched and compute-matched setting. X-axis is the context length and Y-axis is the batch×head dimension. The color indicates the latency ratio of the best(FA2,FA3) over Parallax-CuTeDSL kernel, with warmer colors indicating faster decoding with Parallax. The upper left tiles with backslash indicates OOM in profiling.

in one pass over the KV sequence, we expand the formulation to the following equivalent form,

oPLXi = j≤i pijvj · 1 + j≤i pij · kj⊤ρi − j≤i pij · kj⊤ρi vj, (14)

where pij is the softmax score and pij · kj⊤ρi is the composite score. The computation can be implemented with two parallel scoring and accumulation branches.

Let Qr,Rr denote the tiled matrices for a row block of q and ρ of size Br, and Kc,Vc the tiled matrices for a column block of k and v of size Bc. The softmax branch maintains the running state (mr,d1,O1) as in FA. Parallax additionally maintains the state (d2,O2). In each loop, the covariance branch uses the same Kc,Vc to compute the unnormalized scores

Algorithm 1 Parallax forward core computation.

Require: Input Qr,Rr,K,V,L,s; Output Or;

Running state (mr,d1,2,O1,2).

- 1: for c = 1 to ⌈L/Bc⌉ do
- 2: S1 ← QrK⊤c · s ▷ Apply Masking
- 3: m ← max(mr,rowmax(S1))
- 4: α ← exp2(mr − m)
- 5: P1 ← exp2(S1 − m)
- 6: mr ← m
- 7: S2 ← RrK⊤c ▷ GEMM in TC
- 8: P2 ← P1 ⊙ S2
- 9: d1 ← αd1 + rowsum(P1)
- 10: d2 ← αd2 + rowsum(P2)
- 11: O1 ← αO1 + P1Vc
- 12: O2 ← αO2 + P2Vc ▷ GEMM in TC
- 13: end for
- 14: Or ← O1/d1 · (1 + d2/d1) − O2/d1

- S2 = RrK⊤c , fuses them with the softmax weights as P2 = P1 ⊙S2, and then accumulates O2 = P2Vc alongside O1. The final output combines the two running sums according to equation (14).

Both branches share the online maximum mr, the rescaling factor α and the Kc,Vc tiles. Therefore Parallax does not require extra I/O in each iteration. The detailed algorithm is provided in Algorithm 1, where the additional operations of Parallax are highlighted in red. The operator dependency graph and hardware mapping are shown in Figure 2a.

Arithmetic intensity. The key property of Algorithm 1 is that it increases the arithmetic intensity (AI) over FA, defined as the ratio of floating point operations (FLOPs) to high-bandwidth memory (HBM) traffic in bytes. Write Lq and Lkv as the query and KV sequence lengths respectively and dh

as the head dimension,

4LqLkvdh 2(Lq + 2nrLkv)dh

AIFA ≈

2LqLkv Lq + 2nrLkv

8LqLkvdh 2(2Lq + 2nrLkv)dh

, AIPLX ≈

=

2LqLkv Lq + nrLkv

=

, (15)

where nr = ⌈Lq/Br⌉ is the number of query row blocks. In the regime where nrLkv ≫ Lq, Parallax roughly doubles the arithmetic intensity by adding more compute while reusing the same KV stream. The shift toward a more compute bound operator is what makes decoding a target for kernel level optimization on modern hardware, which we analyze next.

Decode optimization. We prototype Parallax decode kernel in CuTeDSL (Sun et al., 2025a) on NVIDIA Hopper GPUs. The design exploits the structural property that the two branches share the same KV stream, so on I/O-bound decode workloads it consumes essentially no additional HBM traffic. We further exploit that Hopper’s tensor core (TC) matmul instructions (WGMMA) operate on tiles of minimum size 64 rows by construction, whereas a decode step supplies only one query row. The remaining rows of every WGMMA’s accumulator would otherwise be idle. Therefore the QK and RK product can be computed jointly, within the same instructions that standard attention already issues. The same applies to the two PV products in the output accumulation. The prototype kernel also implements few other optimizations such as persistent split over the KV loop and in-kernel reduction. We provide a detailed documentation of the kernel optimization in Appendix C.1.

Profiling Result. We profile the prototype kernel against FA2 and FA3 on H200 GPUs at BF16 precision, sweeping batch sizes from 1 to 2,048 and context lengths from 128 to 32,768, both in powers of two. For each configuration we compare against the best record of FA2 and FA3. Because Parallax doubles the arithmetic intensity, FA cannot be matched on both FLOPs and HBM traffic simultaneously. We therefore report two settings: in the compute-matched setting, Parallax uses dh = 64 such that it matches the FLOPs of FA at dh = 128; in the I/O-matched setting, both kernels use dh = 128 and Parallax doubles the compute of FA with the same HBM traffic. Figure 2b reports the latency-ratio heatmaps for both settings. The prototype Parallax kernel matches or outperforms FA across all configurations. Additional profiling results are in Appendix C.2.

#### 4 Experiment

In this section, we empirically validate Parallax on both synthetic and language modeling benchmarks. We compare Parallax against the Softmax Attention (Attn, Transformer), Mamba, Gated DeltaNet (GDN), MesaNet (Mesa) and Kimi DeltaAttention (KDA) (Team et al., 2025).

##### 4.1 Synthetic Benchmarks

MAD-Benchmark. We evaluate Parallax on the MAD-Benchmark (Poli et al., 2024), which consists of six synthetic tasks designed to evaluate the core ability of sequence mixers1. Particularly, the In-Context-Recall (ICR), Fuzzy-in-Context-Recall (FCR), Noisy-in-Context-Recall (NCR), and SelectiveCopying (SC) tasks assess the model’s ability to recall information from the context, while Compression (CMP) and Memorization (MEM) evaluate the model’s ability to aggregate and memorize information from the training dataset.

1We use the cached data released in the official repository at https://github.com/athms/mad-lab.

###### Scale Model Optim. Sched. RoPE ρ Batch Tokens

- 0.6B

Transformer Muon WSD – 3.93M 78.6B Transformer† Muon WSD – 3.93M 78.6B

Parallax† Muon WSD ✓ 3.93M 78.6B Parallax Muon WSD ✗ 3.93M 78.6B Parallax Muon WSD ✓ 3.93M 78.6B

Transformer AdamW WSD – 3.93M 78.6B

Parallax AdamW WSD ✓ 3.93M 78.6B Transformer AdamW Cosine – 3.93M 78.6B

Parallax AdamW Cosine ✓ 3.93M 78.6B

- 1.7B

Transformer Muon WSD – 7.86M 157.2B Parallax Muon WSD ✗ 7.86M 157.2B Parallax Muon WSD ✓ 7.86M 157.2B

(a) Training configurations.

Optimizer Scheduler Hyperparam. Muon AdamW Hyperparam. WSD Cosine

Learning rate 5×10−3 3×10−4 Warmup 0% 1% Weight decay 0.1 0.1 Decay type Linear Cosine Momentum 0.95 0.9, 0.95 Decay start 80% 1% Embed/Norm lr 0.3×/0.015× 1×/1× Final lr 0 0

(b) Optimizer and learning rate scheduler hyperparameters.

| |
|---|
| |
| |

- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18

- PLX-0.6B-Muon
- PLX-1.7B-Muon

- Attn-0.6B-Muon
- Attn-1.7B-Muon

- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18

PLX-0.6B-AdamW (Cosine) Attn-0.6B-AdamW (Cosine) PLX-0.6B-AdamW (WSD) Attn-0.6B-AdamW (WSD)

0 5K 10K 15K 20K

(c) Training perplexity.

- Table 2 Table 2a and Table 2b detail the training configurations and hyperparameters. We apply the same Muon optimizer settings for 0.6B and 1.7B scale models. Figure 2c shows the training perplexity curves under different optimizers and schedulers. Curves are smoothed for visibility.

Task Attn PLX GDN Mamba Mesa

CMP 0.342 0.332 0.325 0.424 0.310 ICR 0.803 0.951 0.920 0.756 0.998 FCR 0.268 0.356 0.110 0.065 0.219 NCR 0.861 0.937 0.907 0.713 0.999 MEM 0.807 0.733 0.792 0.876 0.861

SC 0.950 0.988 0.939 0.950 0.568 Avg 0.672 0.716 0.665 0.631 0.659

(a) MAD benchmark accuracy.

PLX

Attn

Mamba

Mesa

GDN

| |
|---|

| |
|---|

| |
|---|

| |
|---|

1.0

0.8

0.6

0.4

0.2

0.0

CR(256-512)CR(512-1024)CR(1024-2048)NR(128-512)NR(256-1024)NR(512-2048)SC(128-512)SC(256-1024)SC(512-2048)

(b) MAD-challenge recall accuracy.

All models follow a two-layer architecture with sequence mixer and MLP blocks interleaved. Different from prior work, all the models are trained with Muon optimizer. As reported in Table 1a, Parallax consistently improves on the recall-oriented tasks (ICR, FCR, NCR, SC) while remaining competitive on the compression and memorization tasks (CMP, MEM), and attains the highest overall accuracy.

To further showcase the advantage of Parallax under more challenging recall conditions, we synthesize an additional set of harder tasks by scaling up the KV pairs and the sequence length on ICR, NCR, and SC, with vocabulary size and context length stressed up to 512 and 2048 respectively. We apply the same training specification as in the previous experiment. As shown in Figure 1b, Parallax retains accuracy as the difficulty grows, whereas other baselines degrade dramatically, most visibly on SC at the longest context lengths. Full experimental setup is reported in Appendix F.

##### 4.2 Language Modeling Benchmarks

Having established the recall advantage of Parallax on synthetic tasks, we further evaluate it on LLM pretraining and downstream benchmarks.

Experiment setup. We adopt the Qwen-3 architecture (Qwen Team, 2025) (which applies RMSNorm (Zhang and Sennrich, 2019) to q and k vectors) as implemented in the torchtitan (Liang et al., 2024) repository, and additionally apply RMSNorm to ρ in each Parallax layer. All models are pretrained on the Ultra-FineWeb dataset (Wang et al., 2025b) with a context length of 4096. We compare Parallax against the Transformer baseline at both 0.6B and 1.7B parameter scales. At the 0.6B scale, we also provide the result for KDA, GDN and controlled experiment baselines:

- • Parameter-matched Transformer (Transformer†). Parallax introduces extra parameters from the

WR projection. Transformer† adds the same number of parameters to the Transformer baseline by increasing the query head count in GQA. This choice maintains the KV size and mirrors how Parallax allocates its parameters due to the similarity between q and ρ in computation.

- • Compute-matched Parallax (Parallax†). Parallax doubles the arithmetic intensity compared to FA. Parallax† halves the head dimension to strictly match the attention layer compute. The reduced parameter count is rebalanced by increasing the FFN dimension to match the total parameter count of the standard Parallax.

We additionally ablate the effect of applying RoPE (Su et al., 2024) to ρ in Parallax. The details of training configuration and optimizer hyperparameters are summarized in Table 2a and Table 2b.

Evaluation and benchmarks. For perplexity evaluation, we report both LAMBADA (LMB.) (Paperno et al., 2016) and WikiText (Wiki) (Merity et al., 2017). For zero-shot QA and commonsense reasoning evaluation, we report BoolQ (Clark et al., 2019), HellaSwag (HSwag) (Zellers et al., 2019), PIQA (Bisk et al., 2020), ARC-easy and ARC-challenge (Clark et al., 2018), WinoGrande (Wino.) (Sakaguchi et al., 2020), OpenBookQA (OBQA) (Mihaylov et al., 2018), and SciQ (Welbl et al., 2017). All evaluations are conducted with the LM Evaluation Harness (Gao et al., 2024a).

Results discussion. The results are summarized in Table 3. At the 0.6B scale, Parallax with Muon achieves the best perplexity on both evaluation tasks and the highest average downstream accuracy. This pattern holds at the 1.7B scale, demonstrating that the gains persist at larger scale under Muon. We also find that applying RoPE to the ρ vectors is still beneficial at both scales under Muon, even though the base softmax scores already incorporate positional information.

The two controls answer two distinct questions about the source of the gain:

- • Does the gain come from extra parameters? Transformer† with matched parameters closes only a small fraction of the gap to Parallax, confirming that the improvement is not simply from additional parameters in query-like projections.
- • Does the gain require extra Attention compute? Parallax† with matched attention compute significantly outperforms baseline Transformer and Transformer†, ruling out the added compute as a necessary condition.

These results provide strong evidence that the gain is driven by the mechanism itself. Figure 2c shows the training perplexity curves under different optimizers and schedulers, all with RoPE applied. The curves of Muon models show a substantial gap between Parallax and the Transformer, consistent with the downstream evaluation performance. However, the advantage shrinks markedly or even disappears under AdamW. The performance difference indicates a strong optimizer-architecture interaction, which Section 4.3 further analyze and discuss.

Size Optimizer

Avg ↑ Cosine AdamW

RoPE ρ

LMB. ppl↓

Wiki ppl↓

LMB. acc↑

BoolQ acc↑

HSwag acc↑

PIQA acc↑

ARC-e acc↑

ARC-c acc↑

Wino. acc↑

OBQA acc↑

SciQ acc↑

Model

Transformer — 31.57 26.68 34.93 61.47 46.65 70.35 60.19 30.63 52.33 34.00 80.50 52.34

Parallax ✓ 29.54 26.63 36.48 58.38 46.25 69.75 58.16 30.29 52.96 36.40 74.40 51.45 WSD AdamW

Transformer — 26.63 25.30 36.72 58.41 48.19 70.73 58.08 31.48 54.22 35.20 80.50 52.61 Parallax ✓ 26.59 25.01 37.40 57.16 48.63 71.44 60.90 32.00 52.17 35.60 78.80 52.68

- 0.6B Muon

Kimi DeltaAttn — 25.16 26.81 38.29 55.29 49.30 71.11 62.12 33.19 52.57 34.20 78.50 52.73 Gated DeltaNet — 24.63 26.32 37.69 59.33 50.58 71.71 61.57 32.68 55.41 35.60 78.50 53.67

Transformer — 22.15 23.43 40.07 58.84 52.29 70.73 61.74 33.36 55.41 37.20 81.20 54.54 Transformer† — 22.35 23.36 39.45 56.17 52.35 71.92 63.01 35.49 56.59 36.80 82.30 54.90

Parallax ✗ 19.77 22.69 41.04 60.10 53.14 71.93 64.27 34.73 55.64 35.20 83.80 55.54 Parallax† ✓ 20.29 22.49 40.21 61.44 54.48 72.03 63.80 36.95 55.01 35.80 82.40 55.79

Parallax ✓ 18.56 22.25 41.83 60.24 53.73 72.52 63.51 35.49 56.20 36.80 83.60 55.99

- 1.7B Muon

Transformer — 13.07 18.11 46.77 64.92 62.94 76.39 69.53 42.32 61.01 41.00 88.00 61.43 Parallax ✗ 10.85 17.27 49.54 62.72 64.34 75.79 70.33 43.69 59.91 39.80 89.10 61.69 Parallax ✓ 10.80 17.08 50.26 64.59 64.54 76.39 73.27 42.49 60.77 41.00 88.70 62.45

- Table 3 Downstream perplexity and zero-shot accuracy (↑: higher is better, ↓: lower is better). The average score is computed over the accuracy benchmarks. The AdamW groups are at the 0.6B scale.

0 3 6 9 12 15 18 21 24 27

2

- 4

Muon

Muon

Muon

Muon

0.3

8

AdamW Cosine

AdamW Cosine

AdamW Cosine

AdamW Cosine

40

10

AdamW WSD

AdamW WSD

AdamW WSD

AdamW WSD

6

Random init

0.2

30

5

20

0.1

0 3 6 9 12 15 18 21 24 27

0 3 6 9 12 15 18 21 24 27

0 3 6 9 12 15 18 21 24 27

(a) COR

(b) ∥Corr∥F

(c) CPA

###### (d) ∥ρ∥

- Figure 3 From left to right: (3a) correction-to-output ratio COR; (3b) KV correlation ∥Corr∥F; (3c) covarianceprobe alignment CPA; (3d) probe norm ∥ρ∥. X-axis is the layer index. The dots represent the quantile values across heads and positions, and the line represents the mean.

##### 4.3 Mechanism Analysis

In this section, we quantitatively analyze the optimizer-architecture interaction of Parallax under different optimizers. All the analyses are conducted on the 0.6B scale models (RoPE on ρ applied throughout). Following the magnitude tension discussion in Section 3.2, the headline quantity is the correction-to-output ratio (COR), defined as

CORi = ∥Σ(KVi) ρi∥/∥oSAi ∥, (16)

which measures the relative strength of the covariance correction compared to the intercept term in the affine structure. To decompose any COR gap into directional and magnitude components of the representations, we measure the KV correlation (Corr) and the covariance-probe alignment (CPA) for the directional component, and the probe norm ∥ρi∥ for the magnitude:

- 1

- 2

- 1

- 2

Corri = Σ(i)−

V V Σ(KVi) Σ(i)−

KK , CPAi = ∥Σ(KVi) ρi∥/∥ρi∥∥Σ(KVi) ∥2. (17)

The two metrics are unitless and respectively measure the directional structure of the KV pathway and the alignment of ρi with its leading directions.

The results are shown in Figure 3. COR increases with layer depth under all optimizers, with Muon reaching values above 8 in the deepest layers while AdamW remains below 4. Compared to random initialization, where COR is uniformly high across all layers, training suppresses the correction in early

layers and selectively amplifies it in deeper layers. This amplification barely recovers the initialization level under AdamW, while the deepest layers exceed it under Muon. The probe norm shows the largest optimizer gap, which also grows with layer depth. The directional diagnostics differs as well, with Muon exhibiting higher ∥Corr∥F and higher CPA at most layers. Therefore, the COR gap is not solely a scale effect: Muon also produces richer KV associations and better aligned probes. The training dynamics of COR is shown in Figure 4a.

Gating behavior. The preceding analysis shows that AdamW produces smaller norms and lower COR than Muon. A natural question is whether this simply reflects a scaling convention, or whether AdamW intrinsically fails to utilize the correction branch in training. To test this, we use a learnable sigmoid gate gi = σ(wg⊤xi) that modulates the probe as ρi = gi · WRxi. The gate allows the model to continuously interpolate between suppressing and activating the correction.

- Figure 4b shows that, under Muon, the model learns to open the gate, and the gated run gradually converges to the same final loss as the non-gated baseline. Under AdamW, however, the gate value decreases and stabilizes around 0.26, and it achieves final performance comparable to Transformer, indicating that the model learns to suppress the covariance correction rather than utilize it.

Spectral structure of weights. The activation level differences in both magnitude and direction originate from the weight matrices under different optimizers. We analyze the stable rank of the projection weight in both Parallax and Softmax Attention. Beyond the individual projection matrices, we also analyze the bilinear circuits WQK = WQWK⊤, WOV = WOWV and WRK = WRWK⊤, which provides a more direct view of the effective transformations. The bilinear circuits are constructed for each head separately, and the stable rank is averaged across heads and layers.

The results are summarized in Table 4. Under Muon, all projection matrices develop substantially higher stable rank than under AdamW, consistent with the prior observations. Notably, the stable ranks of WQ, WK and WQK are nearly identical between Parallax and Transformer under every optimizer configuration, confirming the attention scoring structure is shaped by the optimizer alone and unaffected by the architecture. The rank value of this pattern is marked in red.

Softmax Attention Parallax

Muon AW-Cos AW-WSD Muon AW-Cos AW-WSD

- WQ 116.0 17.4↓ 21.7↓ 97.4 20.9↓ 20.7↓ WK 105.8 18.1↓ 22.9↓ 106.4 18.0↓ 18.3↓ WV 145.5 148.1 138.6 199.8↑ 177.2↑ 184.7↑ WO 186.6 109.8 121.6 182.0↑ 137.4↑ 144.7↑
- WR — 134.0 9.3↓ 11.1↓

WQK 22.3 4.6↓ 5.9↓ 25.5 4.9↓ 5.1↓ WOV 24.8 21.2 21.5 34.1↑ 30.4↑ 30.4↑ WRK — 29.1 9.4↓ 9.2↓

For Parallax, WR is disproportionately affected. It exhibits the largest optimizer sensitivity of any projection, with a gap exceeding that of WQ and

Table 4 Stable ranks of projection matrices and bilinear circuits, averaged across heads and layers.

WK. The WRK circuit inherits this bottleneck and mirrors the pattern of the QK circuit, where the stable rank is comparable under Muon but collapses under AdamW. This explains the higher CPA under Muon observed in Figure 3c: a high rank RK circuit allows ρ to align more effectively with the leading covariance directions.

Beyond the optimizer effect, we also observe a consistent architectural effect where WV , WO and WOV circuits have higher stable rank for all optimizers under Parallax. This enrichment reflects the architectural contribution to the value pathway, providing the output projection with a richer set of directions to read from. The rank value of this pattern is marked in green.

While Muon delivers a substantial advantage over AdamW, we do not claim Muon with WSD is the optimal combination for Parallax. We provide additional discussion and visualizations in Appendix G.1

and Appendix G.2.

Muon

AdamW Cosine

AdamW WSD

27

[Figure 7]

15

Mean

Mean

Mean

23

19

10

15

12

5

8

4

0

0

0 5K 10K 15K 20K

5K 10K 15K 20K

5K 10K 15K 20K

(a) Training dynamics of COR under different optimizers.

0.95

AdamW Gate Muon ∆ Loss

0.80

0.2

0.60

∆Loss

Gate

0.40

0.1

0.20

0.0

0.00

0 5K 10K 15K 20K

(b) Gating behavior.

- Figure 4 Training dynamics of COR and the gating behavior. Figure 4a shows the training trajectory of COR at sample layers under different optimizers. The colorbar represents the layer index. Figure 4b shows the gating behavior. The left axis is the gate value and the right axis is the train loss difference.

##### 4.4 Parallax Score Distribution Patterns

Beyond the optimizer driven differences analyzed above, the Parallax mechanism itself produces quantitatively different score distributions from those of Softmax Attention. Parallax output can be written as a weighted average over values according to Equation 8. We denote the per-token Parallax weight by sij ∈ R, analogous to the softmax weight pij,

wij(1 − tij + t¯i) j′ wij′(1 − tij′ + t¯i)

= pij(1 − tij + t¯i). (18)

sij =

Although j≤i sij = 1 holds, the individual weights sij can take negative values and admit values |sij| ≫ 1, which standard softmax weights cannot. This unbounded range provides additional expressive capacity that the correction branch contributes. The following analyses are conducted on a held-out validation batch and averaged across queries and heads.

- 1. Score range. We measure the per-query score range for Parallax with three optimizer configurations. Figure 5a report them as a function of layer depth. Parallax weights routinely take negative values, allowing the model to actively subtract value components from irrelevant tokens rather than merely de-emphasize them. The score range grows with layer depth, with extreme values spanning approximately ±40 in the deepest layers under Muon, consistent with the increasing COR and the stronger correction effect.
- 2. Attention sink. Softmax Attention is known to concentrate excessive probability mass on the first token (Xiao et al., 2024). We quantify the degree of this concentration by measuring the average weight on the first token. For Parallax, we measure the sink ratio for both the base

softmax pij and the combined weights sij. For the combined weights, we use the squared weight share to handle the negative values:

s2i1 j≤i s2ij

pi1 j≤i pij

sinkSAi =

= pi1, sinkPLXi =

. (19)

Figure 5b shows that Parallax substantially reduces the attention sink in both the base softmax and the combined weights, suggesting that the correction branch may absorb the routing role that Softmax Attention typically discharges onto the first token.

PLX (Muon)

PLX (AdamW-Cosine)

20

PLX (AdamW-WSD)

0

−20

−40

0 3 6 9 12 15 18 21 24 27

(a) Score ranges.

|Attn| | |
|---|---|---|
|PLX (<br><br>|Softmax)| |
|PLX (|Square)| |
| | | |
| | | |
| | | |

0.4

0.3

0.2

0.1

0.0

0 10 20

(b) Attention sink.

| |Attn<br><br>PLX (S|oftmax)|
|---|---|---|
| | | |
| | | |
| | | |

- 2
- 3
- 4

0 10 20

(c) Attention entropy.

- Figure 5 Parallax score patterns. Figure 5a, 5b, and 5c respectively show the score range, attention sink and attention entropy patterns of Parallax and Transformer. Dots represent the quantile values across heads and positions, and the line represents the mean.

3. Attention entropy. We measure the dispersion of the softmax distribution in Softmax Attention and the base softmax in Parallax through the Shannon entropy:

Hi = − j≤i pij log pij. (20)

Figure 5c shows that Parallax’s base softmax entropy is consistently higher than that of the Transformer baseline, showing that Parallax produces more diffuse attention weights. Intuitively, Parallax uses the softmax for broader contextual aggregation and offloads fine-grained token discrimination to the correction branch.

We provide additional score distribution visualizations in Appendix H.

- 5 Limitations and Future Directions In this section, we outline several directions opened by this work.

Scaling Parallax. Validating the perplexity gain and optimizer-architecture interaction at larger scale, longer context, in combination with components such as MoE and other architectural modifications is left to future work. The doubled arithmetic intensity opens additional flexibility in tuning the head dimension, head count, and attention to FFN ratio. Identifying the recipe that best balances model performance and throughput on a given hardware target is an important empirical question.

Optimizing the efficiency of Parallax. Parallax inherits the streaming structure of Softmax Attention. Therefore, any contextual sparsity pattern applicable to Softmax Attention, including sliding window, dilated or block sparse, extends directly to Parallax. It is also structurally compatible with other optimization techniques such as MLA. We leave the kernel development and performance evaluation of these variants to future work.

Post-training adaptation from pretrained Transformers. When initialize WR = 0, Parallax layer behaves identically to a Softmax Attention layer at the start of training. A pretrained Transformer checkpoint can therefore be converted into a Parallax model by adding the WR weight and fine-tuning. This contrasts sharply with the Linear Attention family, where no parameter setting recovers Softmax Attention exactly and typically requires retraining to adapt to the new architecture. Whether the post-training adaptation to Parallax is effective under different optimizer settings is an interesting question we leave open.

Theoretical understanding of the optimizer-architecture interaction. In Section 4.3 we empirically characterize the activation and weight spectra of Parallax under different optimizers and diagnose the performance gap between Muon and AdamW. However, the precise characterization behind the observed optimizer dependence of Parallax remains an open question. It also remains to be seen whether this phenomenon happens in other affine mechanisms as discussed in Section 3.2.

Implications for other attention mechanisms. The derivation in Section 3.2 suggests two extensions. First, Linear Attention, DeltaNet and MesaNet all drop the intercept by construction. Reintroducing it yields affine variants, two of which appear in Figure 1 and an affine DeltaNet is defined analogously. Whether these affine variants outperform their intercept-free originals, and whether the optimizerarchitecture interaction observed here recurs in these variants is a valuable future direction. Second, the family in Figure 1 does not yet include DeltaNet, which should be placed between Linear Attention and MesaNet. Deriving the nonparametric counterpart of DeltaNet would be a natural extension of the current work.

#### Author Contributions

Yifei Zuo conceived the project; developed the mathematics, algorithm, and related derivations; implemented the kernel; designed and conducted the experiments; and led the writing of the manuscript. Dhruv Pai contributed substantially to the pretraining experiments and optimizer configuration, and wrote portions of the optimizer-related discussion. Zhichen Zeng contributed to kernel optimization and authored portions of the kernel-related sections. Alec Dewulf and Shuming Hu contributed valuable discussions to this project. Alec also contributed to the writing of the optimizer-introduction sections. Zhaoran Wang provided advisory guidance throughout the project.

#### References

Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. In Advances in Neural Information Processing Systems 37 (NeurIPS 2023), 2023.

Kwangjun Ahn, Byron Xu, Natalie Abreu, Ying Fan, Gagik Magakyan, Pratyusha Sharma, Zheng Zhan, and John Langford. Dion: Distributed orthonormalized updates. arXiv preprint arXiv:2504.05295, 2025.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4895–4901, 2023.

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. In International Conference on Learning Representations (ICLR), 2023.

Noah Amsel, David Persson, Christopher Musco, and Robert M Gower. The polar express: Optimal matrix sign methods and their application to the muon algorithm. arXiv preprint arXiv:2505.16932,

- 2025.

Md Rifat Arefin, Ravid Shwartz-Ziv, Ernie Chang, Chinnadhurai Sankar, Rylan Conway, Aristide Baratin, Adithya Sagar, and Patrick Huber. Learning in transformers under spectral constraints.

In ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling,

- 2026. URL https://openreview.net/forum?id=3ciBX6oWXP.

Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. Zoology: Measuring and improving recall in efficient language models, 2023. URL https://arxiv.org/abs/2312.04927.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview. net/forum?id=8GjSf9Rh7Z.

Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni. It’s all connected: A journey through test-time memorization, attentional bias, retention, and online optimization. In 14th International Conference on Learning Representations (ICLR 2026), 2026. URL https: //openreview.net/forum?id=gZyEJ2kMow.

Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology, 2024. URL https://arxiv.org/abs/2409.20325.

Aviv Bick, Eric Xing, and Albert Gu. Understanding the skill gap in recurrent language models: The role of the gather-and-aggregate mechanism. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 4324–4344. PMLR, 13–19 Jul 2025. URL https://proceedings. mlr.press/v267/bick25a.html.

H. J. Bierens. The nadaraya–watson kernel regression function estimator. Technical Report 1988-58, Faculty of Economics and Business Administration, Vrije Universiteit Amsterdam, Amsterdam, 1988.

Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7432–7439, 2020.

Valentyn Boreiko, Zhiqi Bu, and Sheng Zha. Towards understanding of orthogonalization in muon. In High-dimensional Learning Dynamics 2025, 2025. URL https://openreview.net/forum?id= ppmyFtr9EW.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT), pages 2924–2936, 2019.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge,

2018. URL https://arxiv.org/abs/1803.05457.

Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers, 2023. URL https://arxiv.org/abs/2212.10559.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In 12th International Conference on Learning Representations (ICLR 2024), 2024. URL https: //openreview.net/forum?id=mZn2Xyh9Ec.

Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality, 2024. URL https://arxiv.org/abs/2405.21060.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with i/o-awareness. In Advances in Neural Information Processing Systems 35, pages 16344–16359. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/ 67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf.

DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, T. Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Liu, Xin Xie, Xingkai Yu, Xinnan Song, Xinyi Zhou, Xinyu Yang, Xuan Lu, Xuecheng Su, Y. Wu, Y. K. Li, Y. X. Wei, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Zheng, Yichao Zhang, Yiliang Xiong, Yilong Zhao, Ying He, Ying Tang, Yishi Piao, Yixin Dong, Yixuan Tan, Yiyuan Liu, Yongji Wang, Yongqiang Guo, Yuchen Zhu, Yuduan Wang, Yuheng Zou, Yukun Zha, Yunxian Ma, Yuting Yan, Yuxiang You, Yuxuan Liu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhiniu Wen, Zhipeng Xu, Zhongyu Zhang, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, and Ziwei Xie. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024. URL https://arxiv.org/abs/2405.04434.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 2024a. URL https://github.com/EleutherAI/lm-evaluation-harness.

Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Peiyuan Zhou, Jiaxing Qi, Junjie Lai, Hayden KwokHay So, Ting Cao, Fan Yang, and Mao Yang. Seerattention: Learning intrinsic sparse attention in your llms. arXiv preprint arXiv:2410.13276, 2024b. URL https://arxiv.org/abs/2410.13276.

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pages 30583–30598, 2022.

Ekaterina Grishina, Matvey Smirnov, and Maxim Rakhuba. Accelerating newton-schulz iteration for orthogonalization via chebyshev-type polynomials. arXiv preprint arXiv:2506.10935, 2025.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling (COLM), Jul 2024. URL https://openreview.net/forum?id= tEYskw1VY2. Available on OpenReview.

Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In Proceedings of the International Conference on Learning Representations (ICLR) 2022, 2022a. Outstanding Paper Honorable Mention.

Albert Gu, Isys Johnson, Aman Timalsina, Atri Rudra, and Christopher Ré. How to train your hippo: State space models with generalized orthogonal basis projections, 2022b. URL https: //arxiv.org/abs/2206.12037.

Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In International Conference on Machine Learning, pages 1842–1850. PMLR, 2018.

Samy Jelassi, David Brandfonbrener, Sham M. Kakade, and Eran Malach. Repeat after me: transformers are better than state space models at copying. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cecista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https:// kellerjordan.github.io/posts/muon.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning (ICML 2020), volume 119, pages 5156–5165. PMLR, 2020.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations (ICLR), 2015.

Louis Kirsch, James Harrison, Jascha Sohl-Dickstein, and Luke Metz. General-purpose in-context learning by meta-learning transformers, 2024. URL https://arxiv.org/abs/2212.04458.

Aakash Sunil Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, and Albert Gu. Mamba-3: Improved sequence modeling using state space principles. In 14th International Conference on Learning Representations (ICLR 2026), 2026. URL https: //openreview.net/forum?id=HwCvaJOiCj.

Wanchao Liang, Tianyu Liu, Less Wright, Will Constable, Andrew Gu, Chien-Chin Huang, Iris Zhang, Wei Feng, Howard Huang, Junjie Wang, Sanket Purandare, Gokul Nadathur, and Stratos Idreos. TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training, 2024. URL https://arxiv.org/abs/2410.06511.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training, 2025. URL https://arxiv. org/abs/2502.16982.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations (ICLR), 2019. URL https://openreview.net/forum? id=Bkg6RiCqY7.

Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi Wang, Zhiqi Huang, Huan Yuan, Suting Xu, Xinran Xu, Guokun Lai, Yanru Chen, Huabin Zheng, Junjie Yan, Jianlin Su, Yuxin Wu, Neo Y. Zhang, Zhilin Yang, Xinyu Zhou, Mingxing Zhang, and Jiezhong Qiu. Moba: Mixture of block attention for long-context llms. In Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2025), Spotlight, 2025. URL https://openreview.net/forum?id=RlqYCpTu1P.

Arvind Mahankali, Tatsunori B. Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024.

James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International conference on machine learning, pages 2408–2417. PMLR, 2015.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In 5th International Conference on Learning Representations (ICLR), 2017. URL https: //openreview.net/forum?id=Byj72udxe.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2381–2391, 2018.

Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax, 2018. URL https://arxiv.org/abs/1805.02867.

- E. A. Nadaraya. On estimating regression. Theory of Probability and Its Applications, 9(1):141–142, 1964.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL), pages 1525–1534, 2016.

Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré. Hyena hierarchy: Towards larger convolutional language models. In Proceedings of the 40th International Conference on Machine Learning (ICML 2023), volume 202, pages 28043–28078. PMLR, 2023.

Michael Poli, Armin W Thomas, Eric Nguyen, Pragaash Ponnusamy, Björn Deiseroth, Kristian Kersting, Taiji Suzuki, Brian Hie, Stefano Ermon, Christopher Ré, Ce Zhang, and Stefano Massaroli. Mechanistic design and scaling of hybrid architectures. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An

adversarial Winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740, 2020.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision, 2024. URL https://arxiv.org/abs/2407.08608.

Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, and Riccardo Grazzi. Deltaproduct: Improving state-tracking in linear rnns via householder products, 2025. URL https://arxiv.org/abs/2502.10297.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/j.neucom.2023.127063.

Brandon Sun, Vicki Wang, Feng Xie, Tejash Shah, and Vijay Thakkar. Achieve CUTLASS C++ Performance with Python APIs Using CuTe DSL. https://developer.nvidia. com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/, November 2025a. NVIDIA Technical Blog.

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (Learn at test time): RNNs with expressive hidden states. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 57503–57522. PMLR, 13–19 Jul 2025b. URL https://proceedings.mlr.press/v267/sun25h.html.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models, 2023. URL https://arxiv.org/abs/2307.08621.

Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, Wentao Li, Enzhe Lu, Weizhou Liu, Yanru Chen, Weixin Xu, Longhui Yu, Yejie Wang, Yu Fan, Longguang Zhong, Enming Yuan, Dehao Zhang, Yizhi Zhang, T. Y. Liu, Haiming Wang, Shengjun Fang, Weiran He, Shaowei Liu, Yiwei Li, Jianlin Su, Jiezhong Qiu, Bo Pang, Junjie Yan, Zhejun Jiang, Weixiao Huang, Bohong Yin, Jiacheng You, Chu Wei, Zhengtao Wang, Chao Hong, Yutian Chen, Guanduo Chen, Yucheng Wang, Huabin Zheng, Feng Wang, Yibo Liu, Mengnan Dong, Zheng Zhang, Siyuan Pan, Wenhao Wu, Yuhao Wu, Longyu Guan, Jiawen Tao, Guohong Fu, Xinran Xu, Yuzhi Wang, Guokun Lai, Yuxin Wu, Xinyu Zhou, Zhilin Yang, and Yulun Du. Kimi linear: An expressive, efficient attention architecture, 2025. URL https://arxiv.org/abs/2510.26692.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30 (NIPS 2017), pages 5998–6008. Curran Associates, Inc., 2017.

Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In Proceedings of the 40th International Conference on Machine Learning (ICML), volume 202 of Proceedings of Machine Learning Research, pages 35151–35174. PMLR, 2023.

Johannes von Oswald, Nino Scherrer, Seijin Kobayashi, Luca Versari, Songlin Yang, Maximilian Schlegel, Kaitlin Maile, Yanick Schimpf, Oliver Sieberling, Alexander Meulemans, Guillaume Lajoie, Rif A. Saurous, Charlotte Frenkel, Razvan Pascanu, Blaise Agüera y Arcas, and João Sacramento.

Mesanet: Sequence modeling by locally optimal test-time training. In International Conference on Learning Representations (ICLR), 2026. URL https://openreview.net/forum?id=xa3OnTb6c3.

Nikhil Vyas, Depen Morwani, Rosie Zhao, Mujin Kwun, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade. Soap: Improving and stabilizing shampoo using adam. arXiv preprint arXiv:2409.11321, 2024.

Ke Alexander Wang, Jiaxin Shi, and Emily B. Fox. Test-time regression: a unifying framework for designing sequence models with associative memory, 2025a. URL https://arxiv.org/abs/2501. 12352.

Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Cunxiao Du, Chao Du, Tianyu Pang, Zhuoran Yang, Mingyi Hong, and Vincent Y. F. Tan. Muon outperforms adam in tail-end associative memory learning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=twbMFL0DMp.

Yudong Wang, Zixuan Fu, Jie Cai, Peijun Tang, Hongya Lyu, Yewei Fang, Zhi Zheng, Jie Zhou, Guoyang Zeng, Chaojun Xiao, Xu Han, and Zhiyuan Liu. Ultra-FineWeb: Efficient data filtering and verification for high-quality LLM training data, 2025b. URL https://arxiv.org/abs/2505.05427.

G. S. Watson. Smooth regression analysis. Sankhy¯a: The Indian Journal of Statistics, Series A, 26

(4):359–372, 1964.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings of the 3rd Workshop on Noisy User-generated Text (W-NUT) at EMNLP, pages 94–106, 2017.

Guangxuan Xiao. Statistics behind block sparse attention. https://guangxuanx.com/blog/ block-sparse-attn-stats.html, 2025.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=NG7sS51zVF.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024), 2024a.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), 2024b. doi: 10.52202/079017-3668.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In Proceedings of the 13th International Conference on Learning Representations (ICLR 2025), 2025.

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23078–23097, Vienna, Austria, Jul 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.1126. URL https://aclanthology.

org/2025.acl-long.1126/.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), pages 4791–4800, 2019.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. Curran Associates Inc., Red Hook, NY, USA, 2019.

Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. Journal of Machine Learning Research, 25(49):1–55, 2024.

Yifei Zuo, Yutong Yin, Zhichen Zeng, Ang Li, Banghua Zhu, and Zhaoran Wang. Local linear attention: An optimal interpolation of linear and softmax attention for test-time regression. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=WGpzi489XY.

# Appendix

#### Contents

- A Theorem 24
- B Additional Derivation of Parallax 25

- B.1 Reformulation of LLA . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- B.2 Proof of Proposition 3.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- C Parallax Decode Kernel 26

- C.1 Kernel Optimization Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- C.2 Additional Profiling Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- D Parallax Backward 27
- E Synthetic Experiment Setup 28
- F Pretraining Experiment Setup 28

- F.1 Backbone Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- F.2 Optimizer and Scheduler . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- F.3 Precision and Parallelism . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- G Additional Experiment Results 29

- G.1 Training Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- G.2 Advantage Shrinkage During Decay . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- H Parallax Score Visualizations 30

- A Theorem We briefly state the regularity conditions from (Zuo et al., 2026) (Appendices A–B).

- Assumption A.1 (Domain regularity). The domain D ⊂ Rd has C2 boundary with principal curvatures uniformly bounded by κ2.
- Assumption A.2 (Smoothness). The density p of X satisfies p ∈ C1(D) with p > 0 on D. The

regression function satisfies fj ∈ C2(D) for each output dimension j, and the conditional variance σ2 ∈ C(D).

- Assumption A.3 (Kernel). The kernel K : Rd → [0,∞) is radial, bounded, and compactly supported on the unit ball Bd. The bandwidth matrix H = h2B satisfies h → 0, nhd → ∞, with condition number κ(B) ≤ κ1.
- Assumption A.4 (Boundary gradient). The function f belongs to the class E(D,m,M): there exists a measurable Γ ⊂ ∂D with positive surface measure such that the inward normal derivative satisfies

|∂ef(y)| ≥ m and the tangential gradient satisfies ∥∇Tf(y)∥ < M for all y ∈ Γ, with m,M satisfying a compatibility condition (see (Zuo et al., 2026) Definition B.1 and Lemma B.3).

The Ω(1) lower bound for the global linear estimator requires only that f is not in the affine class G = {x  → β0 + β⊤x} and follows from the projection theorem in L2(D). The NW rates follow from pointwise bias–variance analysis with integrated boundary effects (Appendix A of the original). The LL rates follow from the fact that the local linear estimator achieves O(∥H∥) bias uniformly over D, including near ∂D, eliminating the boundary-induced O(∥H∥1/2) bias of NW. Please refer to the Appendix B of (Zuo et al., 2026) for more details.

#### B Additional Derivation of Parallax

- B.1 Reformulation of LLA We derive equation (7) from the exact LLA forward in equation (3). Starting from

oLLAi =

j≤i

wij(1 − tij) ωi − µ⊤ i ρi

vj, tij = ρ⊤i zij, (21)

divide the numerator and denominator by ωi = j≤i wij, and write pij = wij/ωi for the softmax weight. Using µi = j≤i wijzij, we have µ⊤i ρi/ωi = Epi[tij] = t¯i, so

oLLAi =

Epi (1 − tij)vj

1 − t¯i

. (22)

We expand the numerator. Since zij = kj − qi,

Epi[tijvj] = Epi[vjzij⊤]ρi = Epi[vjkj⊤] − v¯iqi⊤ ρi. (23)

The cross moment factors as Epi[vjkj⊤] = Σ(KVi) + v¯ik¯i⊤, which combined with z¯i = k¯i − qi = Epi[zij] gives

Epi[vjzij⊤] = Σ(KVi) + v¯iz¯i⊤. (24) Therefore

Epi[tijvj] = Σ(KVi) ρi + v¯i (¯zi⊤ρi) = Σ(KVi) ρi + t¯i v¯i, (25) and combining with Epi[vj] = v¯i = oSAi in equation (22),

Epi (1 − tij)vj = (1 − t¯i)oSAi − Σ(KVi) ρi. (26) Dividing by 1 − t¯i and using 1/(1 − t¯i) = 1 + ηi recovers the reformulation

oLLAi = oSAi − (1 + ηi)Σ(KVi) ρi. (27)

- B.2 Proof of Proposition 3.1 Proof. We start by decomposing Σi via the variance identity

wij zijzij⊤ = ωi Epi[zijzij⊤] = ωi Varpi(zij) + ωi z¯iz¯i⊤, (28)

j≤i

which yields the rank one decomposition

Σi = Ai + ωi z¯iz¯i⊤, Ai = ωi Varpi(zij) + λI ≻ 0, (29) where positive definiteness of Ai follows from λ > 0. By the Sherman–Morrison formula,

Define

ωi A−i 1z¯iz¯i⊤A−i 1 1 + ωi z¯⊤

Σ−i 1 = A−i 1 −

. (30)

i A−i 1z¯i

ui := ωi z¯i⊤A−i 1z¯i. (31)

Since A−i 1 ≻ 0, the quadratic form satisfies ui ≥ 0, with equality if and only if z¯i = 0. Using µi = ωi z¯i and applying equation (30),

ui 1 + ui

ρi = Σ−i 1µi = ωi Σ−i 1z¯i = ωi A−i 1z¯i 1 −

Substituting back into t¯i = z¯i⊤ρi gives

ωi A−i 1z¯i 1 + ui

. (32)

=

ωi z¯i⊤A−i 1z¯i 1 + ui

ui

t¯i =

1 + ui ∈ [0,1), (33) which immediately implies

=

t¯i 1 − t¯i

= ui = ωi z¯i⊤A−i 1z¯i ≥ 0. (34)

ηi =

The final expression equals ωi times the squared Mahalanobis distance from qi to the conditional key mean k¯i under the metric A−i 1, justifying the geometric interpretation stated in the main text.

| |
|---|

#### C Parallax Decode Kernel

##### C.1 Kernel Optimization Details

This section extends the discussion in Section 3.3 by describing the three major optimizations that optimize the decode latency of Parallax prototype kernel in CuTeDSL on H200:

- 1. WGMMA sharing. Each compute thread array (CTA) loads Qr and Rr into a single shared memory tile, with Qr as the first row and Rr as the second. The WGMMA then emits S1 and S2 in Algorithm 1 jointly in the same accumulator. After producing P1, we build P2 = P1 ⊙ S2 in registers and stack it with P1 in shared memory. The PV WGMMA then emits O1 and O2 jointly. The covariance branch therefore costs one extra row of register accumulators per CTA, with no additional HBM traffic.
- 2. Persistent split over the KV loop. Decoding presents only BH query rows, often well below the 132 SMs of an H200 in practical configurations. We launch a persistent grid of (B,H,S) CTAs,

where the S number of CTAs share a (B,H) partition the ⌈L/Bc⌉ tile loop of Algorithm 1. The split count S is set so that the launch fits one wave on the device and is rounded to a power of two so that the cross split reduction can be vectorized.

- 3. In-kernel reduction. Each CTA writes its unnormalized partials (m,d1,d2,O1,O2) to a small fp32 HBM workspace and atomically increments a per (B,H) counter. The CTA that observes the final increment is elected the merger: it reads the S partials, runs the log-sum-exp rescaling

in fp32, evaluates (1 + d2/d1)O1/d1 − O2/d1, and writes the output row in the same kernel launch. A compile time branch on S = 1 skips the workspace round trip and writes the output directly from registers, matching the latency of a single CTA kernel on short context shapes.

##### C.2 Additional Profiling Results

We provide the raw profiling results in Figure 2b on H200 in Figure 6. The cuda graph measurement isolates the kernel latency, while the do_bench measurement from Triton captures the end-to-end latency including kernel launch overhead, which FA3 suffers more from. The heatmaps in Figure 2b are derived from the cuda graph measurements, which show a more consistent speedup pattern across different shapes.

#### D Parallax Backward

We derive the closed form gradients dQ,dR,dK,dV of the Parallax forward in equation (14), then briefly describe how the gradients can be streamed in the same row tile and column tile structure as the FA backward.

Following Section 2, let wij = exp(qi⊤kj/h), ωi = j≤i wij and pij = wij/ωi. Recall the composite score tij = ρ⊤i zij and its softmax weighted mean t¯i = Epi[tij]. The Parallax forward in equation (14) admits the equivalent reweighted softmax form

pij (1 + t¯i − tij)vj, (35)

oi =

j≤i

which exposes the two channels through which qi and ρi enter the output. The query shapes the softmax weight pij, and the probe modulates the per token coefficient 1 + t¯i − tij. Let dOi denote the upstream gradient at row i. Define three projections of dOi and the centered gradient δij:

τi = dOi⊤oi, βi = dOi⊤v¯i, aij = dOi⊤vj, δij = aij − βi, (36)

where v¯i = Epi[vj] and τi,βi are row scalars that compress the dependence of the loss on the output and the value mean, while aij,δij resolve the per token contribution. Define two sets of coefficients for the query and probe channels respectively:

gij(1) = pij aij − τi + (t¯i − tij)δij , gij(2) = −pij δij. (37) Differentiating equation (35) and applying the standard softmax derivative gives

gij(1) kj, (38)

dQi = h−1

j≤i

gij(2) kj, (39)

dRi =

j≤i

gij(1) qi +

gij(2) ρi, (40)

dKj = h−1

i≥j

i≥j

pij (1 + t¯i − tij)dOi. (41)

dVj =

i≥j

These gradients admit the same row tile and column tile streaming structure as the FA backward. The forward writes out the cache (oi,v¯i,t¯i,ωi,mi) per row, adding only d + 1 values over the FA cache. Due to the difference in reduction direction, the backward kernel is split into two passes:

- • Row tile pass. Loads (Qr,Rr,dOr) and the cached state, streams over column blocks of (K,V), and accumulates dQr,dRr in parallel.
- • Column tile pass. Loads (Kc,Vc) once and streams over row blocks of (Q,R,dO) in reverse order to accumulate dKc,dVc in parallel.

The Parallax backward therefore can be implemented in an I/O-aware streaming algorithm like the FA backward.

#### E Synthetic Experiment Setup

We follow the pipeline of Poli et al. (2024) without data modification, swapping only the sequence mixer block. Every model is a stack of two mixer and SwiGLU MLP blocks with hidden size d = 128 in bf16 precision. Training proceeds for 60 epochs of Muon with the WSD schedule (0% warmup, last 20% linearly decayed), matching the optimizer settings in Table 2b. We sweep the peak learning rate over {5 × 10−3, 1 × 10−3, 5 × 10−4} and report the best checkpoint per task. All other settings, including data splits, vocabulary, sequence length, number of KV pairs, and evaluation metric, follow the official release at https://github.com/athms/mad-lab. For the harder sweep in Figure 1b, we vary only the data generation parameters: the vocabulary size is raised up to 512 and the context length up to 2048 on ICR, NCR, and SC, while all other training hyperparameters remain unchanged.

#### F Pretraining Experiment Setup

This appendix complements Section 4.2 and Tables 2a and 2b by reporting the implementation details that are not surfaced inline. Each training run is conducted on a node consisting of 8×H100 GPUs.

##### F.1 Backbone Architecture

The Qwen-3 decoder backbone is shared across all language modeling runs. The full set of hyperparameters of 0.6B model is listed in Table 5. The 1.7B model only doubles the hidden dimension and the SwiGLU MLP hidden size. Every run ties the input and output embeddings, applies RMSNorm to q and k vectors, and uses RoPE with base θ = 106. For Parallax runs, the additional projection WR shares the head dimension and head grouping with WQ, and an RMSNorm is applied to ρ. When RoPE is applied to ρ, it uses the same base θ as the queries and keys.

The Transformer† variant raises the query head count while keeping the KV head count fixed, so that its parameter count matches that of Parallax under GQA. We also train a parameter-matched Transformer that extends the FFN hidden dimension to 3712 instead of the head count, and find that it performs similarly to Transformer†. For Parallax†, the head dimension is halved while the head count remains unchanged. The reduced parameter count is compensated by extending the FFN hidden dimension to 4480, so that the total parameter count also matches that of Parallax.

##### F.2 Optimizer and Scheduler

Both optimizers apply gradient norm clipping at 1.0. The Muon implementation uses five Newton– Schulz iterations with the standard quintic coefficient and spectral scaling. Parameters that are not

###### Config Hidden FFN Head dim Q head KV head Layers Vocab RoPE Embed QKNorm

Transformer 1024 3072 128 16 8 28 152k 106 Tied RMSNorm Parallax 1024 3072 128 16 8 28 152k 106 Tied RMSNorm Transformer† 1024 3072 128 24 8 28 152k 106 Tied RMSNorm Parallax† 1024 4480 64 16 8 28 152k 106 Tied RMSNorm

Table 5 Backbone hyperparameters of the 0.6B model. Differences from the Transformer baseline are bolded.

orthogonalizable, including embeddings, norms, and biases, fall back to an Adam style scalar update with (β1,β2) = (0.8,0.95) and ε = 10−7. The AdamW implementation uses ε = 10−8 with decoupled weight decay (Loshchilov and Hutter, 2019). All language modeling runs train for 20,000 optimizer steps. The 1.7B runs double the global batch size relative to the 0.6B runs (Table 2a), so that the total token count scales to approximately 157.2B at the same step count.

##### F.3 Precision and Parallelism

All runs use fully sharded data parallelism without tensor, context, or pipeline parallelism in H100. We apply torchao dynamic fp8 to all linear layers except the LM head, which remains in bf16 for numerical stability.

#### G Additional Experiment Results

##### G.1 Training Dynamics

We track three metrics throughout the 0.6B Parallax pretraining run and report their evolution under different optimizer configurations. Figure 7a reports the layerwise activation norms of q, k, v, and ρ vectors. Figure 7b reports the Frobenius norms of the projection weights WQ, WK, WV , WO, and WR. Figure 4a reports the correction to output ratio COR averaged over the sequence dimension.

Across all three diagnostics, Muon and AdamW exhibit qualitatively similar trajectories during the early phase, after which Muon continues to grow while AdamW saturates. The activation norms ∥v∥ and ∥ρ∥ show the largest separation between optimizers. The COR curves confirm that the correction branch is opened progressively under Muon and reaches its highest values in the deepest layers, whereas it remains largely suppressed under AdamW throughout training. Together, these dynamics provide a temporal view of the optimizer-architecture interaction analyzed in Section 4.3.

##### G.2 Advantage Shrinkage During Decay

Figure 2c shows the training curves of the 0.6B and 1.7B models under Muon with the WSD schedule, where the advantage of Parallax over the Transformer baseline is most pronounced. However, the advantage of Parallax shrinks during the final linear decay phase of the WSD schedule. The training dynamics in Figure 7b shows that the weight norms shrink throughout the decay phase, which may partially explain the shrinkage of the Parallax advantage.

Weight decay annealing. To investigate if the shrinkage of the Parallax advantage is related to the weight norm shrinkage, we run an additional ablation with weight decay annealing (WDA). Concretely, let t ∈ [0,1] denote the fractional progress through the decay stage. WDA replaces the constant weight decay coefficient λ with a schedule

λ(t) = λ · (1 − t)γ, (42)

where γ ≥ 0 controls how aggressively the weight decay is annealed. The choice γ = 0 recovers the standard WSD recipe. For γ > 0, weight decay decreases over the course of the decay stage, with γ = 1 giving linear annealing and γ = 2 giving quadratic annealing that suppresses weight decay more aggressively toward the end of training.

Results. We apply WDA at 0.6B scale with γ ∈ {0.5,1,2} and otherwise identical hyperparameters to the standard Muon with WSD run. Figure 8b confirms that WDA effectively mitigates the weight norm shrinkage, with larger γ giving higher final norms of WR. Figure 8a shows the training loss across the full decay window. All three WDA variants reach lower final training loss than the standard recipe, and the improvement is monotone in γ over the range tested where the largest annealing strength γ = 2 gives the largest gain.

The curves diverge gradually, with the gap widening throughout the first half of the decay stage. However, the gap shrinks during the second half of the decay stage, with the final loss values converging to a narrow range. We annotate the turnaround point where the WDA variants start to lose their advantage in the figure, which occurs at similar step counts across all three WDA variants.

We speculate that this late convergence is a consequence of how WDA affects the effective step size in parameter space. The relative magnitude of a parameter update can be measured by the size of the step relative to the scale of the parameter itself:

∆t = ∥ηt∇WtLt∥F ∥Wt∥F

. (43)

Under standard WSD, both the numerator and the denominator shrink together over the decay stage, partially preserving the relative step size. The shrinkage of denominator ∥Wt∥F is held by WDA, so as ηt → 0 the relative step size collapses faster than under WSD. WDA thus reduces the effective progress late in training.

Implications. WDA produces a clear gain in training performance, confirming that weight norm shrinkage is a real, mechanistic contributor to the decay stage advantage erosion, not an artifact of measurement. This is a preliminary result that we report as evidence that the Muon with WSD recipe is not optimal for Parallax in its current form, and WDA only partially mitigates the issue.

#### H Parallax Score Visualizations

To complement the aggregate score statistics in Section 4.3, we visualize the attention score maps of the Transformer baseline and Parallax. Figure 9a shows the top left corner of the attention map, while Figure 9b shows the bottom right corner. Each block contains 64 × 64 tokens, and the input sequence is sampled from the pretraining data with sequence length 1024. The visualization of Parallax AdamW uses the WSD scheduler.

###### Parallax FA2 FA3

Batch Size 1

Batch Size 2

Batch Size 4

Batch Size 8

Batch Size 16

Batch Size 32

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.100

0.075

0.050

0.025

Batch Size 64

###### Batch Size 128

###### Batch Size 256

###### Batch Size 512

###### Batch Size 1024

Batch Size 2048

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

- 0
- 1
- 2

29 212 215

29 212 215

29 212 215

29 212 215

29 212 215

29 212 215

(a) Latency in cuda graph (dh = 64).

###### Parallax FA2 FA3

Batch Size 1

Batch Size 2

Batch Size 4

Batch Size 8

Batch Size 16

Batch Size 32

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.10

0.05

Batch Size 64

###### Batch Size 128

###### Batch Size 256

###### Batch Size 512

###### Batch Size 1024

Batch Size 2048

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

2.0

1.5

1.0

0.5

0.0

29 212 215

29 212 215

29 212 215

29 212 215

29 212 215

29 212 215

(b) Latency in cuda graph (dh = 128).

###### Parallax FA2 FA3

Batch Size 1 Batch Size 2

Batch Size 4

Batch Size 8

Batch Size 16

Batch Size 32

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.10

0.05

Batch Size 64

###### Batch Size 128

###### Batch Size 256

###### Batch Size 512

###### Batch Size 1024

Batch Size 2048

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

1.5

1.0

0.5

0.0

29 212 215

29 212 215

29 212 215

29 212 215

29 212 215

29 212 215

(c) Latency in do_bench (dh = 64).

###### Parallax FA2 FA3

Batch Size 1 Batch Size 2

Batch Size 4

Batch Size 8

Batch Size 16

Batch Size 32

0.15

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.10

0.05

Batch Size 64

###### Batch Size 128

###### Batch Size 256

###### Batch Size 512

###### Batch Size 1024

Batch Size 2048

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

1.5

1.0

0.5

0.0

29 212 215

29 212 215

29 212 215

29 212 215

29 212 215

29 212 215

(d) Latency in do_bench (dh = 128).

- Figure 6 Kernel latency comparison. X-axis is the context length and Y-axis is the latency in milliseconds. The latency is measured using a cuda graph (top) and using do_bench (bottom). The cuda graph measurement isolates the kernel latency, while the do_bench captures the end-to-end latency including kernel launch overhead.

###### ρ

###### q

k

###### v

- 11

- 12

- 13

14

12

300

Muon

12

27

10

[Figure 8]

200

23

10

8

12

19

14

AdamWCosine

10

6

15

10

Layer

12

5

4

12

8

10

8

2

6

8

4

12

14

10

AdamWWSD

6

0

10

12

5

4

8

10

2

6

0 5K 10K 15K 20K

0 5K 10K 15K 20K

0 5K 10K 15K 20K

0 5K 10K 15K 20K

(a) Evolution of the activation norms of q, k, v, and ρ in the 0.6B Parallax models.

WQ F

WK F

###### WV F

WR F

1250

800

1200

1200

27

[Figure 9]

1000

Muon

1000

1000

600

750

23

800

800

500

400

600

19

22.5

AdamWCosine

35.0

40

24

15

Layer

20.0

32.5

35

22

12

17.5

30.0

30

20

8

45

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

30

50

AdamWWSD

40

25

4

40

25

35

20

0

30

30

20

0 5K 10K 15K 20K

0 5K 10K 15K 20K

0 5K 10K 15K 20K

0 5K 10K 15K 20K

(b) Evolution of the norms of WQ, WK, WV , WO, and WR in the 0.6B Parallax models.

- Figure 7 Training dynamics of activation norms (top) and projection weight norms (bottom).

16K 17K 18K 19K 20K

2.40

2.45

2.50

2.55

2.60

2.65

PLX-0.6B

PLX-0.6B-WDA (γ = 0.5)

- PLX-0.6B-WDA (γ = 1)

- PLX-0.6B-WDA (γ = 2)

(a) Training loss with WDA.

0K 5K 10K 15K 20K

0

100

200

300

400

500

PLX-0.6B

PLX-0.6B-WDA (γ = 0.5)

- PLX-0.6B-WDA (γ = 1)

- PLX-0.6B-WDA (γ = 2)

(b) ∥WR∥F evolution.

- Figure 8 Figure 8a shows the training loss of Parallax with WDA during the decay stage. The turnaround

point where the WDA variants start to lose their advantage is annotated. Figure 8b shows the ∥WR∥F evolution with WDA for the layer 18. WDA effectively mitigates the weight norm shrinkage.

Attn L0 H3

PLX Muon L0 H3

PLX AdamW L0 H3

Attn L1 H0

PLX Muon L1 H0

PLX AdamW L1 H0

1.0

1.0

4

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

0.5

2

0.8

0.8

0.5

2

0.6

0.6

0

0.0

0

0.0

0.4

0.4

−2

−0.5

0.2

0.2

−2

−0.5

−4

0.0

0.0

Attn L9 H8

PLX Muon L9 H8

PLX AdamW L9 H8

Attn L10 H7

PLX Muon L10 H7

PLX AdamW L10 H7

1.0

1.0

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

- 0

- 1

5

5

0.5

0.8

0.8

0.6

0.6

0

0.0

0

0.4

0.4

0.2

0.2

−0.5

−5

−5

−1

0.0

0.0

Attn L18 H7

PLX Muon L18 H7

PLX AdamW L18 H7

Attn L19 H4

PLX Muon L19 H4

PLX AdamW L19 H4

1.0

1.0

5.0

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

- 0

- 1

- 2

4

5

0.8

0.8

2.5

2

0.6

0.6

0

0.0

0

0.4

0.4

−2

−1

−2.5

0.2

0.2

−5

−4

−2

−5.0

0.0

0.0

Attn L26 H3

PLX Muon L26 H3

PLX AdamW L26 H3

Attn L27 H2

PLX Muon L27 H2

PLX AdamW L27 H2

1.0

1.0

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

20

5

10

0.8

0.8

5

10

0.6

0.6

0

0

0

0

0.4

0.4

−10

−5

0.2

0.2

−10

−5

−20

0.0

0.0

(a) Score maps top left.

Attn L0 H3

PLX Muon L0 H3

PLX AdamW L0 H3

Attn L1 H0

PLX Muon L1 H0

PLX AdamW L1 H0

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

4

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

0.050

0.6

0.8

0.5

2

2

0.025

0.6

0.4

0

0.0

0

0.000

0.4

0.2

−0.025

−2

0.2

−2

−0.5

−0.050

−4

0.0

0.0

Attn L9 H8

PLX Muon L9 H8

PLX AdamW L9 H8

Attn L10 H7

PLX Muon L10 H7

PLX AdamW L10 H7

0.4

1.0

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

4

2

- 0

- 1

0.5

0.3

0.2

2

0

0

0.0

0.2

0.1

−2

−0.5

0.1

−1

−2

−4

−1.0

0.0

0.0

Attn L18 H7

PLX Muon L18 H7

PLX AdamW L18 H7

Attn L19 H4

PLX Muon L19 H4

PLX AdamW L19 H4

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

0.6

0.8

- 0

- 1

5

0.5

2

0.6

0.4

0

0

0.0

0.4

0.2

−2

−0.5

−5

0.2

−1

0.0

0.0

Attn L26 H3

PLX Muon L26 H3

PLX AdamW L26 H3

Attn L27 H2

PLX Muon L27 H2

PLX AdamW L27 H2

4

5.0

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

20

0.020

10

0.6

2

2.5

10

0.015

0.4

0

0.0

0

0

0.010

−10

−2.5

−2

0.2

0.005

−10

−20

−5.0

−4

0.0

0.000

(b) Score maps bottom right.

- Figure 9 Score maps of the Transformer and Parallax: top-left corner (top) and bottom-right corner (bottom).

