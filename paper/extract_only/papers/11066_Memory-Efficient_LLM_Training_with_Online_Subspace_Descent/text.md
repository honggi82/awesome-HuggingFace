# arXiv:2408.12857v1[cs.LG]23Aug2024

## Memory-Efficient LLM Training with Online Subspace Descent

Kaizhao Liang†, Bo Liu†, Lizhang Chen†, Qiang Liu† †The University of Texas at Austin {kaizhaol,bliu,lzchen,lqiang}@utexas.edu

### Abstract

Recently, a wide range of memory-efficient LLM training algorithms have gained substantial popularity. These methods leverage the low-rank structure of gradients to project optimizer states into a subspace using projection matrix found by singular value decomposition (SVD). However, convergence of these algorithms is highly dependent on the update rules of their projection matrix. In this work, we provide the first convergence guarantee for arbitrary update rules of projection matrix. This guarantee is generally applicable to optimizers that can be analyzed with Hamiltonian Descent, including most common ones, such as LION, Adam. Inspired by our theoretical understanding, we propose Online Subspace Descent, a new family of subspace descent optimizer without SVD. Instead of updating the projection matrix with eigenvectors, Online Subspace Descent updates the projection matrix with online PCA. Online Subspace Descent is flexible and introduces only minimum overhead to training. We show that for the task of pretraining LLaMA models ranging from 60M to 7B parameters on the C4 dataset, Online Subspace Descent achieves lower perplexity and better downstream tasks performance than state-ofthe-art low-rank training methods across different settings and narrows the gap with full-rank baselines.1

### 1 Introduction

The continual advancement in training large language models (LLMs) presents a compelling challenge in balancing computational efficiency with model performance. As the scope and complexity of these models grow, so does the necessity for innovative strategies that optimize memory usage without compromising the learning capabilities of the model. Recent approaches in low-rank adaptation strategies, including Stochastic Subspace Descent [13], LoRA [11], ReLoRA [15], Gradient LowRank Projection (GaLore) [25] and Sketchy [9] , have paved the way for memory-efficient training by utilizing a periodically updated low-rank projection matrix to manage parameter updates. In particular, GaLore and Sketchy both utilize expensive singular value decomposition to determine the projection matrix, whereas stochastic subspace descent suggests using random matrices as projection matrices and provides convergence analysis on convex functions and objectives. However, to the best of our knowledge, no one has offered any guarantee of convergence for this class of methods on non-convex functions and objectives.

In this work, we provide the first convergence guarantee for arbitrary update rules of the projection matrix. This guarantee is significant because it is broadly applicable to a wide range of optimizers that can be analyzed within the Hamiltonian descent framework [18]. By establishing this convergence guarantee, we demonstrate that our approach is not limited to specific or narrowly defined update rules, but can be extended to include many commonly used optimizers in the field. In particular, this

1Code is available at https://github.com/kyleliang919/Online-Subspace-Descent.

Preprint. Under review.

Algorithm 1 Online Subspace Descent

- 1: Required: Optimizer OptimizerW, learning rate ϵWt , weight decay λW for model weights Wt; and {OptimizerP, ϵPt , λP} for the projection matrix Pt. Proper initialization.
- 2: for iteration t do
- 3: Calculate gradient Gt = ∇L(Wt); Update model weights Wt by (∆ˆ t, Sˆt) = OptimizerW(P⊤t Gt,Sˆt−1), Wt+1 = Wt + ϵWt (Pt∆ˆ t+1 − λWWt)
- 4: Calculate GPt = ∇LG

t

(Pt) for LG

t

(·) in Eq (6); Update the projection Pt by (∆Pt ,SPt ) = OptimizerP(GPt , SPt−1), Pt+1 = Pt + ϵPt (∆Pt − λPPt)

- 5: end for
- 6: Remark: We added weight decay as a common heuristic. We recommend using Adam for both optimizers, and set ϵPt = αϵWt with a constant α (e.g., α = 5), and λW = λP. 2

includes popular algorithms such as LION [4] and Adam [12], which are widely used in various machine learning and optimization tasks. Our results therefore offer a robust theoretical foundation for understanding and analyzing the behavior of these optimizers, ensuring their effectiveness and reliability in diverse applications.

Inspired by our theoretical understanding, we introduce a novel family of memory-efficient optimizers named Online Subspace Descent, which incorporates a dynamically changing projection matrix, replacing the conventional periodic updating approach (SVD) with online PCA. By allowing the projection matrix to evolve in response to the changing gradient landscape, Online Subspace Descent enhances the model’s ability to navigate the parameter space more effectively. This dynamic adaptation aligns more closely with the natural progression of learning in deep neural networks, which is characterized by changes in the importance of different characteristics and interactions as training progresses. Through extensive experiments and comparative analysis, we demonstrate that our approach presents lower perplexity in pretraining LLaMA models (ranging from 60M to 1B parameters) on the C4 dataset compared to state-of-the-art low-rank training methods, closing the perplexity gap with full-rank baselines on language model pretraining.

### 2 Optimization Background

The training of deep learning models reduces to an optimization problem

min

L(W),

W

where W is the set of weight matrices of the model. For simplicity, we assume W ∈ Rn×m is a single matrix of size (n,m) without loss of generality. For notation, we write ⟨A,B⟩ = tr(A⊤B) for inner products of matrices, and ∥A∥2 = tr(A⊤A) the Frobenius norm. We use A ⊙ B to denote the elementwise product, and A⊙2 = A ⊙ A.

- Example 2.1. Update rules of common optimizers:

Gradient Descent : Wt+1 = Wt − ϵt∇L(Wt), Momentum : Wt+1 = Wt − ϵtMt, Mt = (1 − β)∇L(Wt) + βMt−1, Lion-K [4] : Wt+1 = Wt − ϵt∇K(Nt), Nt = (1 − β1)∇L(Wt) + β1Mt

Mt = (1 − β2)∇L(Wt) + β2Mt−1, Adam [12] : Wt+1 = Wt − ϵt

Mt √V t + e

, Mt = (1 − β1t)∇L(Wt) + β1tMt−1,

V t = (1 − β2t)∇L(Wt)⊙2 + β2tV t−1, where ϵt are step sizes, and Mt,V t are the first and second order momentum, and β,β1,β2 are momentum coefficients in (0,1), with βit = βi−β

t+1 i

1−βit+1 , i = 1,2 for β1,β2 ∈ (0,1) in Adam, and K is any convex function with ∇K(0) = 0 for Lion-K [4], and Lion [5] uses K(X) = ∥X∥1,1 and ∇K(X) = sign(X).

These optimizers can be unifiedly viewed as updating Wt together with an optimizer state St:

Wt+1 = Wt + ϕt(St), St = ψt(St−1,∇L(Wt)), (1) with some mapping ϕt,ψt. We have St = Mt for momentum and St = {Mt,V t} for Adam. Note that both Mt,V t are of the same size as the model weights Wt, resulting in high memory consumption for large models. This issue is particularly pronounced for Adam, which typically yields the best performance for large language models (LLMs) but incurs the highest memory cost due to the need to maintain both Mt and V t. One key challenge is to retain the high performance of Adam while enhancing its memory efficiency.

Hamiltonian+Descent One powerful approach to studying the dynamic properties of optimizers is to examine their continuous-time ODE forms in the limit of infinitesimal step size. The continuoustime forms provide clearer insights into the asymptotic convergence of the algorithm, abstracting away the choices of step size, discretization, and stochastic errors. The underlying logic is that a "sound" optimizer should be guaranteed to converge to local optima of the loss, at least when using sufficiently small step sizes.

Inspired by [4, 18], we observe that the continuous-time form of many common optimizers yields a Hamiltian+Descent structure,

d dt

Wt = ∂SH(Wt,St) − Φ(∂W H(Wt,St))

(2)

d dt

St = −∂W H(Wt,St) − Ψ(∂SH(Wt,St)),

where H(W,S) is a Hamiltonian (or Lyapunov) function that satisfies

H(W,S) = L(W), ∀W,

min

S

so that minimizing L(W) reduces to minimizing H(W,S); and Φ(·),Ψ(·) are two monotonic mappings satisfying

∥X∥2Φ := ⟨X,Φ(X)⟩ ≥ 0, ∥X∥2Ψ := ⟨X,Ψ(X)⟩ ≥ 0, ∀X. With Φ(X) = Ψ(X) = 0, the system in (2) reduces to the standard Hamiltonian system that keeps H(Wt,St) = const along the trajectory. When adding the descending components with Φ and Ψ, the system then keeps H(W,S) monotonically non-decreasing:

d dt

d dt

d dt

H(Wt,St) = ∂W Ht,

Wt + ∂SHt,

#### St

(3)

= ⟨∂W Ht,∂SHt − Φ(∂W Ht)⟩ + ⟨∂SHt,−∂W Ht − Ψ(∂SHt)⟩

= −∥∂W Ht∥2Φ − ∥∂SHt∥2Ψ ≤ 0,

where we write ∂W Ht = ∂W H(Wt,St) and similarly for ∂SHt. The main idea is that the cross terms ⟨∂W Ht,∂SHt⟩ are canceled, leaving only the negative terms.

- Example 2.2. The momentum method yields following continuous-time form and Hamiltonian: d

dt

Wt = −Mt,

d dt

Mt = a(∇L(Wt) − Mt), with H(W,M) = L(W) + ∥M∥2

2a

.

- Example 2.3. Adam [12] yields the following continuous-time form and Hamiltonian, d

dt

Wt = −

Mt √V t + e

,

d dt

Mt = a(∇L(Wt) − Mt),

d dt

V t = b(∇L(Wt)⊙2 − V t),

with H(W,M,V ) = L(W) +

- 1

- 2a

M √

V + e

, M ,

for which we can show that ddtH(Wt,Mt,V t) ≤ 0 when a ≥ b/4.

- Example 2.4. The Lion-K optimizer [5, 4] (without weight decay) can be written into d

d dt

Wt = ∇K((1 − b)Mt − b∇L(Wt)),

Mt = −a(∇L(Wt) + Mt),

dt

where a ≥ 0, b ∈ [0,1] and K(X) is any convex function that attains the minimum at X = 0. One of its Hamiltonians that yields the Hamiltonian+descent structure (Eq (13) in Chen et al. [4]) is

1 1 − bK((1 − b)M).

H(W,M) = aL(W) +

### 3 Memory-Efficient Optimizers via Online Subspace Descent

We introduce the idea of constructing memory efficient optimzers by descending in the subspaces that dynamically changes across iterations as motivated by GaLore [25] and Sketchy [9]. We first derive static subspace descent by restricting the whole optimization on a subspace (Section 3.1), and then propose to dynamically change the subspace across iterations as a heuristic to attain the optimization in the full space while only using subspace descent (Section 3.2). In particular, we propose to update the subspaces via continuous online PCA like updates to avoids the need of exact SVD like in GaLore and Sketchy (Section 3.2). Finally, we remark in Section 3.3 the heuristic nature of the derivation of the method and highlight the difficulty in theoretical understanding, which motivates our analysis based on Hamiltonian dynamics in Section 4.

#### 3.1 Static Subspace Descent

One popular approach to improving memory efficiency is to confine the optimization to a lowdimensional space. To do this, we impose a low rank structure of W = PWˆ , where P ∈ Rn×k is a projection matrix to be determined later and Wˆ ∈ Rk×m is a dimension-reduced parameter. When k ≪ n, P and Wˆ are much smaller in size compared to W. Now consider

L(PWˆ ).

min

Wˆ

Applying the optimizer from (1) to update Wˆ along with an optimizer state Sˆ, and mapping the update rule Wˆ t+1 = Wˆ t + ϕt(Sˆt) to that of W = PWˆ t, we get

Wt+1 = Wt + Pϕt(Sˆt), Sˆt = ψt(Sˆt−1,P⊤∇L(Wt)), (4)

where we used the fact that ∇Wˆ L(PWˆ ) = P⊤∇W L(W). This yields a more memory-efficient optimizer, as the size of Sˆt is proportional to that of Wˆt, much smaller than St in (1) when k ≪ n.

#### 3.2 Online Subspace Descent

With a static P, regardless of its values, the parameter W is restricted to have a low rank structure. Although low rank assumption is proved to be useful for fine-tuning with LoRA-like methods [11], it is often too limited for pre-training or when the desirable model weights are not inherently low-rank.

To address this problem, Zhao et al. [25] suggested to keep the projected updated in (4), but use different P across the iterations:

Wt+1 = Wt + Ptϕt(Sˆt), Sˆt = ψt(Sˆt−1,P⊤t ∇L(Wt)), Pt+1 = χt(Pt,Wt,Sˆt), where χt is a update rule of Pt that will be determined in the sequel. The intuition is to open up different projection directions at different iterations, so that optimization can be conducted in different subspaces across different iterations. This is similar to the update of coordinate descent, except in a continuous fashion. Note that the update of Pt can be done in parallel with that of (Wt,Sˆt), and incurs no slowdown once it is fast enough to not cause a speed bottleneck.

- Example 3.1. Examples of common optimizers equipped with online subspace descent: Gradient Descent : Wt+1 = Wt − ϵtPtP⊤t Gt, Gt = ∇L(Wt),

Momentum : Wt+1 = Wt − ϵtPtMˆ t, Mˆ t = (1 − β)P⊤t Gt + βMˆ t−1, Lion-K : Wt+1 = Wt − ϵtPt∇K(Nˆt), Gˆt = P⊤t Gt

Nˆt = (1 − β1)Gˆt + β1Mˆ t, Mˆ t = (1 − β2)Gˆt + β2Mˆ t−1, Adam : Wt+1 = Wt − ϵtPt

Mˆ t Vˆ t + e

, Gˆt = P⊤t Gt,

Mˆ t = (1 − β1t)Gˆt + β1tMˆ t−1, Vˆ t = (1 − β2t)Gˆ⊙t 2 + β2tVˆ t−1.

How Should Pt be Updated? It is useful to draw intuition from the projected gradient descent rule

Wt+1 = Wt − ϵtPtP⊤t Gt, Gt = ∇L(Wt), (5)

in which PtP⊤t can be viewed as a low rank preconditioning of Gt. To make it follow the exact gradient descent, we hope to make PtP⊤t Gt approximate Gt as much as possible. In Galore, this is achieved by performing singular value decomposition (SVD) on Gt periodically every T iterations:

Pt, _, _ = torch.linalg.svd(GT⌊t/T⌋),

where T⌊t/T⌋ is the largest multiple of T less than or equal to t. However, numerical SVD incurs a large computational cost for very large models. Also, since Pt is fully determined by GT⌊t/T⌋ calculated from a single mini-batch at the last periodic point, it does not incorporate the gradient information from all data in a timely fashion.

In this work, we propose to update Pt in a continuous online fashion that incorporates the most recent gradient information in a timely fashion, without calling torch.linalg.decompositions routines.

We view the update of Pt as conducting an online principal component analysis (PCA) based on the streaming of {Gt}. In particular, we propose to update Pt at time t by minimizing the following PCA objective:

2

2

Gt ∥Gt∥

, G˜t =

(P) = PP⊤G˜t − G˜t

+ λ P⊤P − Ik×k

, (6)

#### LG

t

where ∥A∥ = tr(A⊤A)1/2 and Ik×k is the k × k identity matrix; we introduced an auxiliary loss to encourage the columns of P to be orthonormal and normalizes Gt to increase stability.

(P) in (6) is that all its stable local minimum is a global minimum, and P is a global minimum iff PP⊤G˜t forms the optimal rank-k approximation of G˜t [e.g., 3]; moreover, we have P⊤P = Ik×k at optima when λ > 0.

The key property of LG

t

Instead of minimizing LG

(P) exactly, to retain computational efficiency, we propose to update Pt by only performing one step of optimization on LG

t

(P):

t

Pt+1 = OptimizerP.step(Pt, ∇P LG

(Pt)),

t

where OptimizerP.step can be a favorite optimizer, such as gradient descent or Adam. Note that when using Adam, we introduce a copy of optimizer state SPt for Pt. See Algorithm 1. Compared to the exact SVD, each online update of Pt here is fast and can be executed in parallel with the (Wt,Sˆt) updates to avoid slowdown.

#### 3.3 Difficulty in Theoretical Understanding

The idea above of projecting an arbitrary optimizer with a dynamically changing Pt is heuristically motivated and lacks an immediate rigorous theoretical justification. The main challenge lies in the

complex interaction between the update of Ut and the optimization state St, which could potentially degrade the convergence and other theoretical properties of the original optimizer. A key question

is whether we can develop a theoretical framework to understand how Pt impacts the optimizer’s convergence behavior and provide guidance for the design of the update rules of Pt.

To gain understanding, it is useful to first exam the simple case of projected gradient descent in (5) which does not have an optimizer state (St = ∅). In this case, since PtP⊤t is positive semi-finite, the update PtP⊤t Gt is always non-increasing direction of L(W) for any Pt. The algorithm is essentially a variant of coordinate or subspace descent, where Pt defines the subspace on which one step of gradient descent is conduced at iteration t. To ensure that (5) finds a local optimum, we mainly need to ensure that PtGt = 0 only if Gt = 0 to prevent the optimizer from stopping prematurely; this is a mild condition that can be satisfied e.g. when Pt is updated by (online) PCA on Gt.

Unfortunately, this coordinate-descent-like interpretation does not apply to more advanced optimizers that track a momentum state St. This is because St accumulates the information from the projected gradient PτGτ at all earlier iterations τ ≤ t. As Pt changes across time, it is unclear whether the gradient projected to different subspaces Pτ would be coherent with each other, and useful for future updates that are conducted in different subspaces Pt for t > τ. The difficulty is the inertia effect of St that entangles the different subspaces, making the dynamic behavior fundamentally more complicated than naive coordinate descent where the descent in different subspaces is uncoupled. This is what we address in Section 4 via the Hamiltonian descent framework.

### 4 Hamiltonian Descent Meets Subspace Descent: A Lyapunov Analysis

In this section, we show a surprising result that the complication outlined above in Section 3.3 is not a problem for optimizers that yields the Hamiltonian+descent structure in (2). Our result is two-fold:

- • Section 4.1: When applying Online Subspace Descent on systems in (2), the Hamiltonian+descent

structure is preserved once the update rule of Pt has a smooth continuous-time limit. Hence, under very mild conditions, Online Subspace Descent equipped with common optimizers like Adam and Lion automatically yield a Lyapunov function and hence benign continuous-time convergence. Moreover, Pt can, in fact, be generalized to an arbitrary linear operator as shown in Section 4.3.

- • Section 4.2: For any smooth Pt update rules that eliminates the degenerate case of P⊤t Gt = 0 while Gt = 0 at convergence, the online subspace optimizer guarantees to converge in continuous time to a stationary point of the loss L(W). This mild condition is satisfied, for example, when Pt

is updated by a typical optimizer on the online PCA objective LG

##### (P).

t

- 4.1 Online Subspace Descent Preserves the Hamiltonian+Descent Structure Applying dynamic projection to Hamiltonian descent in (2), we obtain the following systems:

d dt

Wt = Pt∂SˆH(Wt,Sˆt) − Φ(∂W H(Wt,Sˆt))

d dt

(7)

Sˆt = −P⊤t ∂W H(Wt,Sˆt) − Ψ(∂SˆH(Wt,Sˆt))

d dt

Pt = Γ(Pt,∇L(Wt)),

where Γ specifies the update rule of Pt. Following essentially the same derivation as (3), one can show that H(W,S) remains a Lyapunov function of (7), regardless of the choice of Γ:

d dt

H(Wt,Sˆt) = −∥∂W Ht∥2Φ − ∥∂SHt∥2Ψ + ⟨∂W Ht,Pt∂SˆHt⟩ − ⟨∂SˆHt,P⊤t ∂W Ht⟩

(8)

= −∥∂W Ht∥2Φ − ∥∂SHt∥2Ψ ≤ 0,

where the key is to use the adjoint property of P and P⊤ that ⟨PtX,Y ⟩ = ⟨X,P⊤t Y ⟩, which cancels the crossing terms, independent of the values of Pt. There is no requirement on Γ here, besides that the derivative in (8) should exist. As shown in Section 4.3, we can generalize (8) by

replacing Pt and P⊤t with a general linear operator Pt and its adjoint Pt∗. The following are examples of continuous-time Momentum, Lion-K and Adam with subspace descent and their Hamiltonian functions.

- Example 4.1. Momentum + Online Subspace Descent is

d dt

Wt = −PtMˆ t, Gˆt = P⊤t ∇L(Wt),

d dt

#### Mˆ t = a(Gˆt − Mˆ t),

with Lyapunov function H(W,Mˆ ) = L(W) + ∥Mˆ ∥2

, for which

2a

d dt

H(Wt,Mˆ t) = −∇L(Wt)⊤PtMˆ t + Mˆ ⊤t (P⊤t ∇L(Wt) − Mˆ t) = − M ˆ t

- Example 4.2. Adam + Online Subspace Descent is

2 2

≤ 0.

Mˆ t Vˆ t + e

d dt

, Gˆt = P⊤t ∇L(Wt),

Wt = Pt

d dt

Mˆ t = a(Gˆt − Mˆ t),

d dt

Vˆ t = b(Gˆ2t − Vˆ t).

- 1

- 2a

with Lyapunov function H(W,M,V ) = L(W) +

M ˆ Vˆ + e

, Mˆ , for which

d dt

#### H(Wt,Mˆ t,Vˆ t)

Mˆ t Vˆ t + e

M ˆ t Vˆ t + e

1 a

b 4a

, a(P⊤t Gt − Mˆ t) −

= − Gt,Pt

+

Mˆ ⊙t 2 Vˆ t + e

M ˆ ⊙t 2 Vˆ t ⊙ ( Vˆ t + e)⊙2

Vˆ t Vˆ t + e

b 4a

b 4a

−

= − 1 −

,

2

2

#### M ˆ tGˆt

M ˆ t Vˆ t + e

b 4a

b 4a

≤ − 1 −

−

≤ 0,

Vˆ t( Vˆ t + e)

4

where we assume a ≥ b/4.

- Example 4.3. The Lion-K + Online Subspace Descent is d

dt

Wt = Pt∇K((1 − b)Mˆ t − bGˆt),

d dt

Mt = −a(Gˆt + Mˆ t), Gˆt = P⊤t ∇L(Wt) Consider the Hamiltonian function in Eq (13) of [4]:

H(W,Mˆ ) = aL(W) +

1 1 − bK((1 − b)Mˆ ).

d dt

H(Wt,Mt) = a⟨Gt,Pt∇K((1 − b)Mˆ t − bGˆt)⟩ − a⟨∇K((1 − b)Mˆ t),Gˆt + Mˆ t⟩

= a⟨Gˆt,∇K((1 − b)Mˆ t − bGˆt) − ∇K((1 − b)Mˆ t)⟩ − a⟨∇K((1 − b)Mˆ t),Mˆ t⟩

= −

- a

- b

[(1 − b)Mˆ t; − bGˆt]∇K −

a (1 − b)

[0; (1 − b)Mˆ t]∇K ≤ 0

where we defined [X;Y ]∇K = ⟨Y , ∇K(X +Y )−∇K(X)⟩ and used the fact that [X;Y ]∇K ≥ 0 by the convexity of K; we used ⟨Gt,PtXt⟩ = ⟨P⊤t Gt,Xt⟩ = ⟨Gˆt,Xt⟩.

In all examples above, although the form of Hamiltonian H(W,Sˆ) is independent of the update rule of Pt, the decreasing rate ddtH(Wt,Sˆt) depends on Pt in a complicate way through Mˆ t,Vˆ t, Gˆt. An interesting direction for future investigation is to find optimal rules of Pt to maximize the decreasing rate as an optimal control problem.

- 4.2 Convergence to Local Optima

M ˆ ⊙t 2 Vˆ t ⊙ ( Vˆ t + e)⊙2

, (Gˆ⊙t 2 − Vˆ t)

, Gˆ⊙t 2

In addition to the Lyapunov structure, we need an additional mild condition on the update rule of Pt to ensure the system converges to the local optimum of the loss L(W). The main idea is to prevent the system from stopping prematurely before reaching zero gradient Gt = 0 by excluding the degenerate case of PtGt = 0 while Gt ̸= 0 in the invariant set of the system.

Assumption 4.4. Assume the functions in system (7) are continuously differentiable and

- i) ddtH(Wt,Sˆt) = 0 implies Gˆt = P⊤t ∇L(Wt) = 0 and ddtWt = 0.

- ii) When Gt ≡ G ̸= 0, the set {P : P⊤G = 0} is not a positive invariant set of ddtPt = Γ(Pt,Gt).

This is a mild condition. Assumption i) says that the optimizer should stop at a point with Gˆt = 0, which is easy to verify for the common optimizers like momentum, Adam, Lion-K. Assumption ii)

ensures Gˆt = 0 would imply Gt = 0 in invariance sets, which is satisfied when for example, Pt is updated by a reasonable optimizer of the online PCA loss that converges to a stable local minimum.

Theorem 4.5. Assume Assumption 4.4 holds. Let (Wt,St,Pt)t be a bounded solution of (7), then all the accumulation points {Wt} as t → +∞ are stationary points of L(W). Proof. By LaSalle’s invariance principle, the positive limit set of (Wt,St,Pt)t must be contained in I, where I = {the union of complete trajectories satisfying ddtH(Wt,Sˆt) = 0, ∀t }.

From the Assumption i), the trajectories contained in I must satisfy ddtWt = 0, which implies d dtGt = ddt∇L(Wt) = 0 and Gˆt = 0 and hence Gt ≡ G is a constant with P⊤t G = 0. Moreover, from Assumption ii), we must have ∇L(Wt) = Gt ≡ 0, since otherwise the trajectory is not

invariant. As a result, all trajectories in the limit set I must have ∇L(Wt) = 0. Because ddtWt = 0, these trajectories are static points of Wt.

| |
|---|

- 4.3 Online Subspace Descent with General Linear Projection Operators We can generalize the online subspace descent with general linear operators:

d dt

Wt = Pt(∂SˆH(Wt,Sˆt)) − Φ(∂W H(Wt,Sˆt))

d dt

Sˆt = −Pt∗(∂W H(Wt,Sˆt)) − Ψ(∂SˆH(Wt,Sˆt))

d dtPt = Γ(Pt,∇L(Wt)),

where we generalize Pt to be any linear operator Pt with an adjoint operator Pt∗, satisfying

⟨X,Pt(Y )⟩ = ⟨Pt∗(X),Y ⟩, ∀X,Y . The derivation of Lyapunov follows a similar way:

d dt

H(Wt,Sˆt) = −∥∂W Ht∥2Φ − ∥∂SHt∥2Ψ + ⟨∂W Ht,Pt(∂SˆHt)⟩ − ⟨∂SˆHt,Pt∗(∂W Ht)⟩

= −∥∂W Ht∥2Φ − ∥∂SHt∥2Ψ ≤ 0,

where the crossing terms are again canceled due to the adjoint property.

As an example of the general framework, consider Pt(X) = PtXQt, where Qt is another projection matrix applied on the different dimension of X (see also [25]). The adjoint operator of Pt is

Pt∗(X) = P⊤t XQ⊤t . This can be verified by

⟨PtXQt,Y ⟩ = tr(PtXQtY ⊤) = tr(XQtY ⊤Pt) = tr(X(P⊤t Y Q⊤t )⊤) = ⟨X,P⊤t Y Q⊤t ⟩. The subspace descent system of this operator is

d dt

Wt = Pt∂SˆH(Wt,Sˆt)Qt − Φ(∂W H(Wt,Sˆt))

d dt

Sˆt = −P⊤t ∂W H(Wt,Sˆt))Q⊤t − Ψ(∂SˆH(Wt,Sˆt))

d dt

Pt = ΓP(Pt,Qt,∇L(Wt))

d dt

Qt = ΓQ(Pt,Qt,∇L(Wt)),

where Pt,Qt can be updated jointly via an online SVD on Gt. Another linear operator that involves two matrices is Pt(X) = PtX+XQt, which yields Pt∗(X) = P⊤t X + XQ⊤t .

5 Experiment

We answer a number of key questions with pretraining experiments of LLaMA [22] on the C4 dataset [20]. All experiments except for large 7B experiments are conducted on a single NVIDIA A100 GPU.

- 5.1 Why do we Need Online Subspace Descent?

Overall, Online Subspace Descent offers two major advantages over previous methods that rely on SVD, better convergence and lower overhead. In this section, we discuss both in detail.

First, Online Subspace Descent closes the gap between the state-of-the-art low-rank method and full rank baseline uniformly across different model sizes, as shown in figure 1. A highlight amongst these results is LLaMA 1B (SS 256). As shown in table 1, Online Subspace Descent attains significant improvement over GaLore in perplexity, while consuming a similar amount of GPU memory (8.64 GB v.s 9.01 GB). One additional observation in 1 shows as model size and sequence length grow, Online Subspace Descent becomes more effective. We hypothesize that this is due to the higher intrinsic rank of the underlying optimization problem in larger models. Hence, the positive impact on the convergence of the online update of Pt becomes more obvious. See more details in Appendix A.

[Figure 1]

Perplexity(↓) 60M 350M 1B

Method

8bit-AdamW (Full Rank) 32.75 30.43 29.40 GaLore (Rank = 512) 57.03 44.34 35.52 Ours (Rank = 512) 56.12 43.67 31.30

- Table 1: Pretraining LLaMA 1B with a sequence length of 256 and for 10K steps, perplexity was reported as the training average of the last 10 steps. AdamW8bit serves as the base optimizer for both. Figure 1: Validation perplexity of LLaMA 1B with

sequence length 256, rank 512 for 10K steps.

Another favorable characteristic of Online Subspace Descent is its minimum overhead. In figure 2, we measure and analyze the execution time of SVD and online PCA on a popular data center GPU (A100) and a consumer GPU (RTX 3090). The typical Pytorch implementation of SVD can be up to 142 times slower than running a single-step online PCA on representative weight tensors from LLaMA architectures. Online PCA is fast because it is implemented as a single optimization step with respect to a simple loss function. Hence, each step of online PCA can be cleverly scheduled and hidden in the weight optimization step when executed in parallel, whereas SVD is too expensive to be hidden.

[Figure 2]

Figure 2: The execution time of torch.svd and that a single-step backward() call for online PCA in PyTorch, on matrices of typical shapes in linear layers in the LLaMA 60M to 7B. Thanks to the high speed of single-step online PCA, Pt updates can be executed in parallel with weight updates, adding no overhead to the training process. In contrast, SVD incurs significant overhead as the model and weight tensor sizes increase.

#### 5.2 What Rank Should we Pick for Online Subspace Descent?

We conduct an ablation study on the rank of Online Subspace Descent. Figure 3 shows that the final perplexity is inversely correlated with rank: higher ranks result in lower convergent perplexity. However, the rate of reduction of perplexity decreases as the rank increases, eventually reaching a saturation point. We propose an intuitive explanation for this phenomenon. In language modeling, high-frequency tokens can be effectively learned with low-rank training. However, learning lowerfrequency tokens requires higher ranks. Once these lower-frequency tokens are adequately learned, further increasing the rank does not significantly decrease perplexity. In conclusion, given sufficient time and resources, higher ranks yield better performance for Online Subspace Descent. It is recommended that the highest rank be selected until the perplexity reduction saturates.

#### 5.3 What are the Best Hyperparameters?

α and λ: The parameter α controls the update speed of Pt, while λ determines the regularization strength on the optimization objective of Pt. Empirically, we find that the result is not sensitive to λ

for small models (60M). and set λ = 0.1 for all subsequent experiments. We find that α must be kept small to avoid instability (Figure 3), and we set α = 5 for all experiments.

Learning rate: For the small model (60M), learning rate choices are more flexible, producing similar results. However, for larger models (350M, 1B), we recommend using a learning rate that is 10 times smaller, specifically 0.001. Larger learning rates cause unrecoverable spikes and instability, a general characteristic observed across all methods. See additional hyperparameter choices in Appendix A.

[Figure 3]

Figure 3: From left to right are loss curves of 10K steps on LLaMA 60M: leftmost is the sweep of rank, middle is the sweep of α and rightmost is the sweep of λ.

#### 5.4 Can Online Subspace Descent be Applied to Different Optimizers?

One straightforward extension of Online Subspace Descent is to apply it to other base optimizers beyond AdamW8bit. We conduct ablation studies on LION [6] and Adafactor [21], finding that Online Subspace Descent behaves similarly to how it does with AdamW8bit. Despite the initial observation that updating Pt with AdamW8bit consistently yields better results, we discover that updating Pt with simple SGD can achieve similar performance.

Method GaLore Ours

Lion Adaf. Lion+Lion Adaf.+Adaf. Lion+AdamW Adaf.+AdamW Perplexity 46.90 34.32 57.97 47.61 44.76 34.15

- Table 2: LLaMA 60M on C4 with sequence length 1024, with optimizers on Pt and Wt, denote as "Ours {Wt optimizer} + {Pt optimizer}". Adaf., and Adam refer to Adafactor and 8bit-AdamW, respectively.

5.5 Can Online Subspace Descent Scale to Larger Model?

We pretrain from scratch a 7B LLaMA model on the C4 dataset for 10K steps, where the Pt matrix is updated by SGD. The perplexity is the lower the better. The final perplexity and training wall-clock time are provided in Table 3. We further provide the downstream evaluation of the pretrained checkpoints using Galore and our method on the GLUE benchmark in Table 4. Our method consistently outperforms Galore when the model size scales up.

Method Perplexity Wall Clock Time (hours)

Galore 51.21 9.7439 Ours 43.72 7.1428

- Table 3: Perplexity and Wall Clock Time for 7B models pretrained on C4 for 10K steps. Lower perplexity is better. Online Subspace Descent can be upto 1.3x faster than GaLore.

- 6 Related Works We discuss related works on memory-efficient optimization and low-rank adaptation techniques.

#### Method MRPC RTE SST2 MNLI QNLI QQP AVG

Galore 0.6838 0.5018 0.5183 0.3506 0.4946 0.3682 0.4862 Ours 0.6982 0.4901 0.5233 0.3654 0.5142 0.3795 0.4951

- Table 4: Standardized GLUE evaluation for 7B model checkpoints using eval-harness. Results are reported for various downstream tasks.

Low-Rank Adaptation Low-Rank Adaptation (LoRA) [11] adds a low-rank adaptor to speficic linear layers in a model, and finetune only the low-rank adaptor. As the adaptors are small, LoRA is widely applied for finetuning large models. Many variants have been proposed since LoRA, including support for multi-task learning Wang et al. [23] and further memory reductions Dettmers et al. [8]. Notably, Lialin et al. [15] proposed ReLoRA for pretraining, requiring a full-rank training warmup to match standard performance levels. It’s important to note that LoRA is fundamentally distinct from subspace descent. While subspace descent optimizes within the original model parameter space, LoRA focuses its optimization efforts within the space of the adaptors.

Memory-Efficient Optimization Several approaches aim to reduce memory costs associated with gradient statistics in adaptive optimization algorithms [21, 2, 7]. In particular, Adafactor [21] factorizes the second-order statistics by a row-column outer product and update the factorized bases on the fly, hence achieving a sub-linear memory cost. K-Fac [19] presents a factorized approximation of the Fisher information matrix which leads to a sublinear natural gradient method. More recently, Feinberg et al. [9] observes that the spectra of the Kronecker-factored gradient covariance matrix in deep learning (DL) training tasks are concentrated on a small leading eigenspace and propose to maintain a matrix preconditioner using the frequent directions sketch. However, their method requires conducting the eigendecomposition at every step, which can be costly for large models. Other than factorization methods, quantization techniques [7, 1, 24, 16] are also widely used, where the gradient (or the momentum and the preconditioner) are directly quantized to tradeoff performance for memory. Fused gradient computation method [17] have also been used to minimize memory costs during training. GaLore [25] is the most relevant work to ours. GaLore focuses on low-rank gradient structures, reducing memory costs for both first and second-order statistics. Our method can be viewed as a general extension to GaLore where we replace the infrequent SVD by a continuous subspace descent [14, 10]. As a result, our method not only provides a more general framework to study memory-efficient subspace descent, but is also more performant than GaLore in practice.

### 7 Conclusion

In conclusion, we provide the first convergence guarantee for arbitrary update rules of projection matrix, applicable to a range of optimizers that can be analyzed using Hamiltonian Descent, including common ones like LION, AdamW, and Adafactor. Inspired by this theoretical foundation, we introduce Dynamic Subspace Descent, a novel family of subspace descent optimizers that eschews SVD in favor of online PCA for updating projection matrix. Dynamic Subspace Descent is both flexible and minimally intrusive, and our experiments show that it achieves lower perplexity in pretraining LLaMA models (ranging from 60M to 1B parameters) on the C4 dataset compared to state-of-the-art low-rank training methods, while also closing the perplexity gap with full-rank baselines.

For future research, we propose several open and intriguing questions: (1) Are there alternative methods for updating projection matrix that could accelerate convergence? (2) What is the impact of weight decay on convergence in Dynamic Subspace Descent? (3) Can low-rank gradients and updates be combined with dynamic low-rank weights (e.g., Mixture of Experts) to further enhance training efficiency? (4) Can this method be applied to problems beyond language modeling? We hope that our work provides a strong foundation for exploring these questions.

### References

- [1] Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. Qsgd: Communication-efficient sgd via gradient quantization and encoding. Advances in neural information processing systems, 30, 2017.
- [2] Rohan Anil, Vineet Gupta, Tomer Koren, and Yoram Singer. Memory efficient adaptive optimization. Advances in Neural Information Processing Systems, 32, 2019.
- [3] Pierre Baldi and Kurt Hornik. Neural networks and principal component analysis: Learning from examples without local minima. Neural networks, 2(1):53–58, 1989.
- [4] Lizhang Chen, Bo Liu, Kaizhao Liang, and Qiang Liu. Lion secretly solves constrained optimization: As lyapunov predicts. arXiv preprint arXiv:2310.05898, 2023.
- [5] Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, et al. Symbolic discovery of optimization algorithms. arXiv preprint arXiv:2302.06675, 2023.
- [6] Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, et al. Symbolic discovery of optimization algorithms. Advances in Neural Information Processing Systems, 36, 2024.
- [7] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise quantization. arXiv preprint arXiv:2110.02861, 2021.
- [8] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. Advances in Neural Information Processing Systems, 36, 2024.
- [9] Vladimir Feinberg, Xinyi Chen, Y Jennifer Sun, Rohan Anil, and Elad Hazan. Sketchy: Memoryefficient adaptive regularization with frequent directions. Advances in Neural Information Processing Systems, 36, 2024.
- [10] Guy Gur-Ari, Daniel A Roberts, and Ethan Dyer. Gradient descent happens in a tiny subspace. arXiv preprint arXiv:1812.04754, 2018.
- [11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.
- [12] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [13] David Kozak, Stephen Becker, Alireza Doostan, and Luis Tenorio. Stochastic subspace descent. arXiv preprint arXiv:1904.01145, 2019.
- [14] Brett W Larsen, Stanislav Fort, Nic Becker, and Surya Ganguli. How many degrees of freedom do we need to train deep networks: a loss landscape perspective. arXiv preprint arXiv:2107.05802, 2021.
- [15] Vladislav Lialin, Sherin Muckatira, Namrata Shivagunde, and Anna Rumshisky. Relora: Highrank training through low-rank updates. In Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ NeurIPS 2023), 2023.
- [16] Bo Liu, Lemeng Wu, Lizhang Chen, Kaizhao Liang, Jiaxu Zhu, Chen Liang, Raghuraman Krishnamoorthi, and Qiang Liu. Communication efficient distributed training with distributed lion. arXiv preprint arXiv:2404.00438, 2024.
- [17] Kai Lv, Yuqing Yang, Tengxiao Liu, Qinghui Gao, Qipeng Guo, and Xipeng Qiu. Full parameter fine-tuning for large language models with limited resources. arXiv preprint arXiv:2306.09782, 2023.
- [18] Chris J Maddison, Daniel Paulin, Yee Whye Teh, Brendan O’Donoghue, and Arnaud Doucet. Hamiltonian descent methods. arXiv preprint arXiv:1809.05042, 2018.

- [19] James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International conference on machine learning, pages 2408–2417. PMLR, 2015.
- [20] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints, 2019.
- [21] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pages 4596–4604. PMLR, 2018.
- [22] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [23] Yiming Wang, Yu Lin, Xiaodong Zeng, and Guannan Zhang. Multilora: Democratizing lora for better multi-task learning. arXiv preprint arXiv:2311.11501, 2023.
- [24] Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. Advances in neural information processing systems, 30, 2017.
- [25] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. Galore: Memory-efficient llm training by gradient low-rank projection. arXiv preprint arXiv:2403.03507, 2024.

### A Experiments

- A.1 Hyperparameters

We sweep learning rate from [0.01, 0.005, 0.001]. For GaLore as well as Adam8bit, we follow the recommended hyperparameters as much as possible. For instance, GaLore update gap is set to recommended default, 200. Warmup is set to 10% of the total training steps. Batch size is set to 512 and gradient clipping is set to 1.0.

- A.2 Rank Sweep

In the following table, is a sweep on different ranks and their final perplexity of LLaMA 60M (SS = 1024) on C4. All other hyperparameters are fixed and using recommended default. Notice that as the rank increases, both Dynamic Subspace Descent and GaLore improve.

Rank Perplexity (Ours) Perplexity (GaLore)

32 85.90 86.16 128 49.01 48.05 512 37.41 36.93 Full 37.18 36.51

Table 5: On LLaMA 60M SS 1024, we sweep across different ranks, the trend is clear and intuitive that higher rank is preferred when it’s feasible.

- A.3 Optimizer Sweep

Method Perplexity

Lion 52.65 Adafactor 33.45 Adam8bit 29.77 SGD 3469.14

Galore LION 46.90 Galore Adafactor 34.32 Galore AdamW8bit 48.05 Ours LION + LION 57.97 Ours Adafactor + Adafactor 47.61 Ours AdamW8bit + AdamW8bit 49.01 Ours LION + Adam8bit 44.76 Ours Adafactor + AdamW8bit 34.15 Ours AdamW8bit + SGD 53.53

- Table 6: In this experiment, we train LLaMA 60M on C4 with sequence length of 1024. We combine different base optimizers to update both Pt and Wt, denote as "Ours weight optimizer + Pt optimizer".

