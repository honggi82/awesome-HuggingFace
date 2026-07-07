# arXiv:2507.12142v2[cs.LG]1Oct2025

## LORA MEETS RIEMANNION: MUON OPTIMIZER FOR PARAMETRIZATION-INDEPENDENT LOW-RANK ADAPTERS

Vladimir Bogachev HSE University, Lomonosov Moscow State University vabogachev@hse.ru

Denis Bobkov HSE University

Vera Soboleva HSE University

Vladimir Aletov HSE University

Aibek Alanov HSE University

Alexander Molozhavenko HSE University

Maxim Rakhuba HSE University

ABSTRACT

This work presents a novel, fully Riemannian framework for Low-Rank Adaptation (LoRA) that geometrically treats low-rank adapters by optimizing them directly on the fixed-rank manifold. This formulation eliminates the parametrization ambiguity present in standard Euclidean optimizers. Our framework integrates three key components to achieve this: (1) we derive Riemannion, a new Riemannian optimizer on the fixed-rank matrix manifold that generalizes the recently proposed Muon optimizer; (2) we develop a Riemannian gradient-informed LoRA initialization, and (3) we provide an efficient implementation without prominent overhead that uses automatic differentiation to compute arising geometric operations while adhering to best practices in numerical linear algebra. Comprehensive experimental results on both LLM and diffusion model architectures demonstrate that our approach yields consistent and noticeable improvements in convergence speed and final task performance over both standard LoRA and its state-of-the-art modifications.

1 INTRODUCTION

Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks Brown et al. (2020); Touvron et al. (2023a;b). However, the computational and storage costs associated with training and deploying such models at scale pose significant challenges. To reduce these costs, parameter-efficient fine-tuning techniques such as low-rank adaptation (LoRA) Hu et al. (2022) have emerged as a practical solution. LoRA enables efficient adaptation of pre-trained models by embedding learnable low-rank matrices into specific weight updates, allowing most of the original parameters to remain frozen. In particular, the main idea of LoRA is to fine-tune a pretrained model using a rank-r correction matrix ∆W:

W + ∆W = W + AB⊤, A ∈ Rm×r, B ∈ Rn×r,

where W remains constant during training and A,B are optimized via gradient-based optimization methods.

Despite its efficiency, the dominant practice of optimizing the LoRA factors (A,B) with Euclidean optimizers such as SGD (Robbins & Monro, 1951), Adam (Kingma & Ba, 2014), Adagrad (Duchi et al., 2011), RMSProp (Tieleman, 2012), etc. that misaligned with the geometry of the low-rank constraint. The same update ∆W can be represented by infinitely many factorizations: for any A ∈ Rm×r, B ∈ Rn×r and any invertible matrix S ∈ Rr×r, we may write:

∆W = AB⊤ = A B⊤, where A = AS, B = BS−⊤. (1)

Ideally, training should be reparameterization (transformation) invariant: the update to ∆W must not depend on which factorization (A,B) is used (Yen et al., 2024). Empirically, this lack of invariance manifests as unbalanced learning where one factor dominates and the other stalls, fragile

hyperparameter sensitivity, and path-dependent solutions. These issues have prompted geometryaware formulations. Riemannian treatments of low-rank models operate on the fixed-rank manifold rather than the ambient factor space, projecting gradients to the tangent space and retracting back to the manifold. Such steps can be implemented efficiently when r ≪ min{m,n} and avoid forming full-size matrices. Within the LoRA literature, existing Riemannian approaches either use standard SGD-type optimizers Mo et al. (2025, LORO) or rely on the Adam (Zhang & Pilanci, 2024) optimizer for auxiliary matrices within the chosen parameterization, which deviate them from the Riemannian framework and introduce dependence on parameterization.

In this work, we introduce a fully Riemannian framework for training LoRA that optimizes the adapter X =∆W directly on the fixed-rank matrix manifold

Mr = {X ∈ Rm×n : rank(X) = r}, eliminating factorization ambiguity by construction. Central to our approach is Riemannion, a new Riemannian optimizer on Mr that generalizes the recently proposed Muon optimizer (Jordan et al., 2024) to the fixed-rank setting. In contrast to prior Riemannian LoRA variants that port Adamlike mechanics to the manifold with ad hoc choices, our design inherits Muon’s geometry-aligned normalization, yielding transformation invariance of the learned update. We further propose a Riemannian gradient–informed initialization that places the initial adapter at a good location on Mr, and we provide a practical, low-overhead implementation that assembles projections, retractions, and vector transports via automatic differentiation, also following best practices from numerical linear algebra. Extensive experiments on LLM and diffusion architectures show consistent gains in convergence speed and final task performance over standard LoRA and recent state-of-the-art modifications.

Our contributions are as follows:

- • Riemannion: Muon on the fixed-rank manifold. We derive Riemannion, the first optimizer that generalizes Muon to the manifold Mr of fixed rank matrices.
- • Riemannian gradient-informed initialization. We propose an initialization strategy which yields best alignment between the initial Riemannian gradient and the Euclidean gradient. We also propose an efficient way for this strategy by using a randomized SVD algorithm with implicit matrix multiplication (Section 5). Finally, we show the connection of this initialization to LoRA-GA.
- • Efficient implementation with automatic differentiation. We pay special attention to numerical implementation to make the method robust without any prominent overhead compared to vanilla LoRA at small ranks.
- • Comprehensive empirical validation. We showcase the performance of our framework for fine-tuning LLMs and in subject-driven generation using diffusion models. Among positive effects that we observe are: boost in target metrics, improved convergence, and reduction of variance.

- 2 RELATED WORK

The problem of an optimal initial guess selection for low-rank LLM adaptation has been addressed in a sequence of works: the authors Meng et al. (2024, PiSSA) have suggested a heuristic that involves using a low-rank truncated SVD of pretrained parameters as an initial point for LoRA and its orthogonal complement as frozen layer’s parameters, so that the tuning process starts without changing the starting value of the loss function. A similar approach was implemented by Wang et al. (2025, MiLoRA) with the main difference of optimizing the smallest singular components of unadapted parameter matrix. A context-aware initialization was considered in (Yang et al., 2024, CorDA) and (Parkina & Rakhuba, 2025, COALA) proposes a numerically robust inversion-free framework for low-rank weighted approximations for this setting. Another idea is to initialize LoRA with a subset of left and right singular vectors of a doubled-rank truncated SVD of the loss function gradient at the starting parameters, proposed by Wang et al. (2024, LoRA-GA). We show direct connection of this method to our Riemannian initialization strategy and propose how to additionally significantly accelerate the computation of SVD using our approach. Attempting to overcome the asymmetry in the initialization of vanilla LoRA fine-tuning process, (Hayou et al., 2024, LoRA+) introduced a scale-free step size selection for LoRA factors.

Riemannian optimization is widely used for algorithms on matrix manifolds and allows for exploiting task geometry or imposing additional constraints. For example, a Riemannian solution for the extreme eigenpairs search problem was described in Absil et al. (2009); Baker (2008), a matrix completion task, which is common in collaborative filtering for recommender systems, via optimization on the fixed-rank manifold (Vandereycken, 2013), a Riemannian approach on the manifold of matrices with orthonormal columns (the Stiefel manifold) was used by Wisdom et al. (2016) for diminishing the problem of vanishing and exploding gradients in recurrent neural networks, etc. The book Trendafilov & Gallo (2021) also presents a comprehensive description of useful manifolds for solutions of the data science problems. For the deeper understanding of applied differential geometry techniques see the books Absil et al. (2009) and Boumal (2023).

The idea of using Riemannian optimization has recently started to emerge for the large language models. For example, the fine-tuning of LLMs with the help of the Stiefel manifold was considered in the work Hu et al. (2024). The authors of (Zhang & Pilanci, 2024) introduced the Riemannian inspired modification of Adam. The authors of Mo et al. (2025, LORO) applied the Riemannian optimization techniques for pretraining LLMs on the fixed-rank manifold. Parametrization that is used in our work can potentially help in this setting as well, by additionally avoiding potential overheads and instabilities, arising due the explicit inversion of Gram matrices.

- 3 PRELIMINARIES

- 3.1 MUON OPTIMIZER

Muon is an optimizer designed specifically for matrix-valued parameters in a network’s hidden layers. Empirically, it accelerates training on language and vision workloads while leaving scalar/vector parameters and the input/output layers to a conventional optimizer such as AdamW. At a high level, Muon takes the step that stochastic gradient descent with momentum (SGDM) would make on a weight matrix and orthogonalizes that update before applying it. Orthogonalization acts as a per-layer, per-step preconditioner that equalizes singular values of the update, which mitigates the collapse of updates into a few dominant directions (Jordan et al., 2024). More specifically, let W ∈ Rn×m be a hidden-layer weight. With gradient Gt = ∇WL(Wt) and momentum Mt = βMt−1 + Gt, Muon computes

#### Mt ≈ Ortho(Mt) and Wt+1 = Wt − η Mt,

where Ortho(·) denotes the nearest semi-orthogonal matrix in Frobenius norm, i.e.,

∥O − G∥F : O⊤O = I or OO⊤ = I . (2)

Ortho(G) = arg max

O

Computing Ortho(G) exactly amounts to taking the SVD G = USV ⊤ and returning UV ⊤, which is too slow to do at every iteration. Muon instead applies a Newton–Schulz (NS) iteration that—after normalizing G — implements a composition of a fixed low-degree polynomial in GG⊤ acting on G and converges to UV ⊤. Leveraging efficient matrix multiplication operations results in a highly

performant iteration. We will write Mt = NS(Mt) for short.

LMO interpretation. Muon’s step admits a clean linear minimization oracle (LMO) interpretation (Bernstein, 2025). Indeed, consider the operator-norm unit ball B2 = {X : ∥X∥2 ≤ 1}. The linear minimization oracle (LMO) over B2 at matrix Mt given by its SVD Mt = USV ⊤ is UV ⊤ ∈ Arg max

⟨Mt,S⟩. (3)

∥S∥2≤1

Applying Muon to LoRA. In LoRA, a frozen weight W0 ∈ Rn×m is adapted via a low-rank update W = W0 + α BA with B ∈ Rn×r, A ∈ Rr×m and small r. Each trainable factor (A and B) is a 2D parameter, so Muon can be applied per factor:

- Mt(A) ← βMt(−A1) + ∇AL, At+1 ← At − ηA NS Mt(A) ,
- Mt(B) ← βMt(−B1) + ∇BL, Bt+1 ← Bt − ηB NS Mt(B) .

Note that acting on the two factors separately makes Muon non–reparameterization-invariant: its per-factor orthogonalization depends on arbitrary scalings or rotations, skewing the weight-space step and often letting one factor dominate.

- 3.2 RIEMANNIAN OPTIMIZATION

Let Mr = {X ∈ Rm×n : rank(X) = r} ⊆ Rm×n be a smooth manifold of fixed-rank matrices (Lee, 2003, Example 8.14). Let every point X of Mr be equipped with a tangent plane TXMr. Thinking geometrically, the tangent plane plays the role of the best local, flat approximation to this curved set: if you “zoom in” at X, the manifold looks like a plane. We will now discuss how to numerically parametrize points on a manifold and its tangent plane. Every rank-r matrix X ∈ Mr can be represented using matrices AL ∈ Rm×r,Br ∈ Rn×r with orthonormal columns and a square matrix G ∈ Rr×r as

X = ALGBR⊤. (4) For example, one may think of a thin SVD, in which case G becomes the diagonal matrix of singular values, but other representations are also possible and will be convenient for our purposes. Using (4), any tangent vector ξ ∈ TXMr can be represented in a matrix factorization format:

ξ = A ˙ AL BR B˙ ⊤ , A˙⊤AL = 0, (5) so that any tangent vector can be identified in terms of the tuple: (A,˙ B˙ ). Since the factor matrices contain 2r columns, we immediately have that rankξ ≤ 2r. Another remarkable fact is that the point X ∈ Mr itself lies in the TXMr with A˙ = 0,B˙ = B. Given a matrix Z ∈ Rm×n, its orthogonal projection PT

XMr Z onto the tangent space TXMr (with the parameterization given in

(5)) can be computed as follows:

XMr(Z) = ALA⊤LZ + I − ALA⊤L ZBRBR⊤. (6)

PT

and, hence, can be represented in the form of (5) with A˙ = (I − ALA⊤L)ZBR, and B˙ = Z⊤AL. Let F : Rm×n → R be a differentiable function with the Euclidean gradient ∇F ∈ Rm×n. Within the Riemannian optimization framework, we solve the following task optimization problem: min X∈Mr

#### F(X).

When constructing the algorithms, we need to work with three key objects: Riemannian gradient, retraction and vector transport. The Euclidean gradient is a direction of the steepest local increase F. Therefore, it is common to use the Riemannian gradient — the direction of the steepest local increase of corresponding smooth function value along the manifold, which lies in the tangent space (Absil et al., 2009, chap. 3.6). Given the Euclidean gradient ∇F, one may endow the tangent space TXMr with a natural scalar product and derive a formula for the direction of the local steepest ascent of F with respect to the manifold. This unique direction is called the Riemannian gradient and can be computed as follows:

XMr (∇F(X)), X ∈ Mr. (7)

gradF(X) = PT

A simple and robust retraction that maps a tangent step ξ (for example, ξ is a negative Riemannian gradient) back to the manifold is the truncated SVD:

#### RX(ξ) ≡ R(X + ξ) = SVDr(X + ξ), (8)

i.e., the best rank-r approximation of X + ξ in Frobenius norm. Note that here we do not need to compute the full SVD and can utilize low-rank structure of X and ξ, leading to O((m + n)r2 + r3) operations (Absil & Oseledets, 2015). Finally, because tangent spaces change from an optimization step to step, momentum (an accumulated tangent vector) must be moved between them via a vector transport. For embedded manifolds like Mr, a standard choice is the projection transport

XMr(ξ), ξ ∈ TY Mr, (9) which simply reprojects the same ambient matrix ξ onto the new tangent space at X.

TY →X(ξ) = PT

- 4 RIEMANNION

In this section, we focus on the setting of parameter-efficient fine-tuning and, hence, to the fixed-rank manifold. For fine-tuning of one layer the optimization problem becomes:

L(W + ∆W) → min

,

∆W∈Mr

where L is a differentiable loss function. Note that optimizing on Mr removes the ambiguity of factorized parameterizations, because all computations are carried out in the intrinsic space of the product X rather than in any particular factorization. So the formulas we write below will naturally be reparameterization-invariant. Let Gt = PT

WMr(∇L(Wt)) be the Riemannian gradient. A Riemannian heavy-ball (Polyak, 1964) momentum step reads

Mt = β Mt−1 + Gt, Mt−1 = TWt−1→Wt(Mt−1), (10) ∆Wt+1 = R (∆Wt − η Mt), (11)

with M0 = 0, momentum parameter β ∈ [0,1), and stepsize η > 0.

Let us now discuss how to introduce a Muon-like variant of this iteration, which we refer to as Riemannion. A direct projection onto the set of orthogonal matrices Ortho(Mt) presents two challenges. First, such a step does not respect the underlying Riemannian geometry. To address this, we propose to find the best approximation of Ortho(Mt) on the tangent plane TWMr. Such a solution is given via the projection onto the tangent plane PT

WMr(Ortho(Mt)). However, a second issue arises: although this projection is low-rank, its computation remains inefficient because the input matrix is of full rank. Let us notice that Mt ∈ T∆W

tMr and, hence, is of rank at most 2r (Section 3.2). At the same time, the LMO interpretation (Section 3.1) provides several admissible low-rank solutions, including one where Orthor(·) replaces only the first 2r singular values with 1, while all others are set to 0. Consequently, we obtain the following update rule:

∆WtMr(Orthor(Mt)). (12) Note that Orthor(·) exactly preserves the column and row spaces of Mt ∈ T∆W

Mt = PT

tMr. Although PT

∆WtMr(Orthor(Mt)) does not yield singular values exactly equal to 1, in practice they remain in a close proximity. In particular, in our experiments they were always in the interval (0.9,1.1). This behavior is reminiscent of the Newton–Schulz iteration, which likewise produces approximate singular values. To eliminate this inexactness and obtain an accurate solution in the intersection of T∆W

tMr with the set of matrices whose first 2r singular values are exactly 1, one could apply various strategies. For example, one may formally apply the alternating projection method (Cheney & Goldstein, 1959; Boyd & Dattorro, 2003, Theorem 4, Section 2):

∆WtMr(Orthor(Mt)))). (13) As an alternative, we could solve the LMO problem on the tangent plane:

Mt = PT

∆WtMr(Orthor(...PT

⟨Mt,S⟩, (14)

Mt ∈ Arg max

S∈T∆WtMr ∥S∥2≤1

where we proposed to apply observations from (Cesista, 2025) to the fixed-rank manifold. In Appendix D, we show that (12) is an approximate solution to the maximization problem (14). Applying (13) or accurately solving (14) on the intersection of two convex sets using convex optimization methods is more computationally expensive, and our experiments indicate that it has little to no impact on the overall convergence of the optimizer.

- Algorithm 1 OrthoLR (efficient computation of Orthor(ξ) for ξ ∈ TXMr).

Require: ξ ∈ TXMr given by (A,˙ B˙ ) from (5); AL,BR such that X = ALGBR⊤ as in (4) Ensure: A ∈ Rm×2r,B ∈ Rm×2r: ABT = Ortho(ξ).

- 1: QL,TL = qr([AL,A˙]), QR,Tr = qr([B,B˙ R]⊤). // O (m + n)r2
- 2: UL,_,Vr⊤ = SVD TL TR⊤ . // O r3
- 3: A˙ = QLUL, B˙ = QRVR. // O (m + n)r2

Let us finally show that Mt from (12) can be computed efficiently using O((m + n)r2 + r3) arithmetic operations. Indeed, first of all we need to apply Orthor(·) to a tangent vector Mt. From (6), we know that a tangent vector can be represented in a form of a rank-2r matrix. Such a representation can always be transformed into the compact SVD form with 2 QR decompositions and a single full SVD of a 2r × 2r matrix, see Algorithm 1 called OrthoLR. As a next step, we need to project the obtained result (decomposed matrix of rank 2r) onto the tangent plane. The operation can also be done efficiently via (6) and is summarized in Algorithm 2 called ProjectLR.

- Algorithm 2 ProjectLR (efficient computation of PT

#### XMr(Z) for a rank-r′ matrix Z). Require: A ∈ Rm×r

′

′

#### such that Z = AB⊤; AL,BR such that X = ALGBR⊤ as in (4). Ensure: ξ = PT

,B ∈ Rn×r

XMr(Z) given by (A,˙ B˙ ) from (5).

#### 1: A˙ := (A − AR(A⊤RA))(B⊤BR), B˙ := B(A⊤AL) // O((m + n)r′2)

- 5 LOCALLY OPTIMAL INITIALIZATION (LOI)

Once the theoretical framework for the Riemannian optimizer is established, it is natural to consider an initialization scheme that accounts for the underlying Riemannian geometry. Given any ∆W ∈ Mr, we may write

L(W) = L(W − ∆W

W′

+∆W) = L(W′ + ∆W). (15)

This raises the question: how should ∆W be chosen to ensure the fastest loss decrease along the manifold? The solution is to consider the following optimization task:

∆W∗(0) ∈ Arg max

∆W∈Mr

∥PT

∆WMr ∇WL(W)∥2F . (16) Since PT

∆WMr is an orthogonal projection matrix to the tangent plane, the task (16) essentially seeks for the point on the fixed-rank manifold, whose tangent space has most alignment with the Euclidean gradient. In other words, this means that the direction of the steepest local function decrease alongside the manifold is aligned with the full model tuning direction. The solution to this task is presented in Theorem 5.1.

Theorem 5.1. Let the SVD of ∇WL(W) be:

∇WL(W) = [U1,r Ur,2r U⊥]

Σ1,r 0 0 0 Σr,2r 0 0 0 Σ⊥

[V1,r Vr,2r V⊥]⊤ ,

and let also σ2r ̸= σ2r+1. Then any optimal solution ∆W∗(0) to the problem (16) has the form:

∆W∗(0) ∈ [U1,r,Ur,2rΣr,2r]

S11 S21

[C21 C22]

Σ1,rV1⊤,r Vr,2r

S =

S11 S12 S21 S22 ∈ GL2r(R), S−1 =

C11 C12 C21 C22

.

(17)

Proof. See Appendix A.

| |
|---|

In our experiments, we use S =

αIr 0 0 Ir

, obtaining: ∆W∗(0) = αU1,rVr,⊤2r ∈ Mr,α ∈ R \ {0}.

Interestingly, Theorem 5.1 relates to the findings of Wang et al. (2024), although their analysis neither adopts a Riemannian framework nor addresses parametrization-free optimization. Our optimizer further differs from Zhang & Pilanci (2024) and Mo et al. (2025, LORO) in that it avoids

inversion of the Gram matrix. As a result, the method remains stable as ∥∆W∗(0)∥ → 0. Empirical results indicate that initializing ∆W∗(0) with a small norm leads to improved performance. The procedure for selecting the scaling parameter α is described in Appendix F.

- 6 SINGLE BACKWARD-PASS GRADIENT TRICK

This section introduces an efficient method for computing gradient-times–matrix in a matrix-free way, at a computational cost equivalent to a single backward pass. We then apply this technique in both the LOI initialization procedure in Algorithm 3 and the Riemannion optimizer in Algorithm 4.

The calculation of the full fine-tuning loss gradient ∇WL(W) with pretrained parameters W ∈ Rm×n is computationally expensive. At the same time, for our framework we only need to compute

the products (∇WL(W)⊤ M) or (∇WL(W)N) for some M ∈ Rm×r,N ∈ Rn×r. Using the trick from (Novikov et al., 2022), we may calculate both quantities simultaneously using a single forward-backward pass with a doubled rank representation.

First, we initialize differentiable parameters Z1 = 0 ∈ Rm×r and Z2 = 0 ∈ Rn×r and perform a simple forward pass L := L W + Z1N⊤ + MZ2⊤ . This step does not violate the pipelines of LoRA framework since it is equivalent to a standard LoRA forward pass with special adapter:

Z1N⊤ + MZ2⊤ = [Z1 M][N Z2]⊤ = A˜B˜⊤, A˜ ∈ Rm×2r,B˜ ∈ Rn×2r. Then we invoke an autodiff algorithm for value L, which ensures us both:

- ∇Z1

- L = ∇Z1L W + Z1N⊤ + MZ2⊤ |Z1=0,Z2=0 = ∇WL(W)N,

∇Z2

- L = ∇Z2L W + Z1N⊤ + MZ2⊤ |Z1=0,Z2=0 = ∇WL(W)⊤M.

(18)

#### Note, that if ∆W = ALB⊤ = ABR⊤ from W + ∆W, then one may take N = BR,M = AL,Z1 = 0,Z2 = B,Y = W + ∆W and the exact same operations work.

Notably, (18) is crucial for computation of the Riemannian gradient in (6). Therefore the proposed approach is a key building block that allows us to avoid forming the full fine-tune gradient loss and effectively compute matrix-matrix multiplications. This idea is the core for both: LOI initialization (Algorithm 3) and Riemannion optimizer (Algorithm 4).

Randomized SVD for efficient initialization The computation of the 2r-truncated SVD of the full loss gradient, as required by Theorem 5.1, has asymptotic complexity O(min{m,n}mn), which may be infeasible for large-scale models. In addition, explicitly forming this gradient is itself computationally expensive, and thus we need to avoid it in practice. To overcome this problem, we propose to use a randomized SVD with power iterations (see (Halko et al., 2011)), which we also enhance with our one-step gradient trick as is described in the Algorithm 3. In a nutshell, we need to compute ∇WL(W)∇WL(W)⊤ q Y , where Y = ∇WL(W)Ω and Ω is sampled from standard normal distribution. This iteration can be done in a robust manner using QR decompositions. The steps 2,4,5,6 correspond to a trick from (18).

Overall, the LOI search procedure has asymptotic complexity O((m+n)r2), plus 2(q+1) additional backward passes. Moreover, since LOI search is executed only once before fine-tuning (which typically involves thousands of backward passes), its runtime overhead is negligible, taking merely 0.25% of the total fine-tuning wall-clock time in our experiments. Section 7.1, about 2 thousand optimization steps were carried out and Algorithm 3 accounted for merely 0.25% of the total finetuning wall-clock time.

- Algorithm 3 BackPropRSVD Require: Weights W ∈ Rm×n, rank r ∈ N, oversampling parameter p, power-step parameter q. Ensure: Randomized r-truncated SVD (Ur,Σr,Vr) of ∇WL(W).

- 1: Choose k = r + p, Sample Ω ∈ Rn×k ∼ N(0,1). // O (nr)
- 2: Y := qr(∇A L(W + AΩ⊤)|A=0).Q. // 1 backward pass +O mr2
- 3: for i := 1,...,q do
- 4: Y := qr([∇B L(W + Y B⊤)|B=0]⊤).Q. // 1 backward pass +O nr2
- 5: Y := qr(∇A L(W + AY ⊤)|A=0).Q. // 1 backward pass+O mr2
- 6: Y := [∇B L(W + Y B⊤)|B=0]⊤. // 1 backward pass
- 7: U,Σ,V ⊤ := truncSVD(Y,r). // O nr2
- 8: Y U,Σ,V ⊤ // O mr2

Efficient Riemannion implementation The Riemannion optimizer is presented in Algorithm 4. Similar to vanilla Muon, it relies on a chosen Ortho procedure. Since the LoRA approach represents the fine-tuning shift ∆W with low-rank matrices, we employ an SVD-based Ortho procedure. For computational efficiency, this procedure is adapted to the fixed-rank manifold, as described in

Section 4 and implemented in the OrthoLR (Algorithm 1). In detail, in step 1 we calculate the Riemannian gradient components via a single backward call (18). Then, in step 2 the algorithm transports the Heavy-Ball tangent direction to the current point via Algorithm 2 that provides a simple but effective implementation of (6). In line 4, we compute the final optimization direction on the tangent space of the given point with a Heavy-Ball momentum coefficient β. In line 5 algorithm performs the retraction step.

The algorithm finalizes with saving the obtained minimization direction for momentum in line 6 and calculating of a new point representation in step 7.

- Algorithm 4 One step of Riemannion

Require: Weight matrix W′ ∈ Rm×n, rank r ∈ N, initial point AL,BR, Heavy-ball momentum AHB,BHB, step size η, momentum coefficient β, weight decay coefficient γ. Ensure: Tuning parameters ∆W∗ ∈ Mr.

- 1: A,˙ B˙ := ∇Z1L W′ + Z1BR⊤ + ALZ2⊤ |Z1=0,Z2=B, ∇Z2L W′ + Z1BR⊤ + ALZ2⊤ |Z1=0,Z2=B. // 1 backward pass
- 2: A˙prev,B˙prev := ProjectLR((AM,BM),AL,BR). // O (m + n)r2
- 3: A,˙ B˙ := βA˙prev + (I − ALA⊤L)A,˙ βB˙prev + B˙ // O (n + m)r2
- 4: A,˙ B˙ := ProjectLR(OrthoLR(AL,BR,A,˙ B,r,˙ ),AL,BR) // O (m + n)r2 + r3
- 5: U,Σ,V ⊤ := RetractionLR([−ηA,A˙ L],[BR,−η(B˙ + γBR)]) // O (m + n)r2 + r3
- 6: AHB,BHB := [A,A˙ L], [BR,B˙ ]
- 7: AL,B := U,ΣV ⊤ // O nr2

Overall, one iteration of the Riemannion loop has asymptotic complexity O((m + n)r2 + r3) and additionally the same number of backward passes as vanilla LoRA. For comparison, the Euclidean Muon optimizer for LoRA (Section 3.1) exhibits the same asymptotic complexity. The complete framework and its time performance are summarized in Algorithm 5 and Appendix I.

- 7 EXPERIMENTS

In this section, we also employ the term LOI (Locally Optimal Initialization), referring to the proposed initialization scheme described in Section 5. This is done to distinguish between using RiemannLoRA as an optimizer with zero initialization (as in basic LoRA) and RiemannLoRA-LOI, which is the proposed combination of initialization and optimization. We conduct a series of LLM fine-tuning experiments for different tasks. All of the experiments were computed on NVIDIA V100-32Gb GPU and A100-80Gb GPU. We ran all the experiments within ∼ 2000 GPU hours.

- 7.1 COMMONSENSE REASONING FINE-TUNING

The results were obtained for the benchmark (Clark et al. (2019, BoolQ), Bisk et al. (2020, PIQA), Sap et al. (2019, SIQA), Zellers et al. (2019, hellaswag), Sakaguchi et al. (2021, winogrande),Clark et al. (2018, ARC), Mihaylov et al. (2018, OBQA)) common reasoning. The structure of the dataset is described in Appendix E. In the following experiments we conduct a fine-tuning procedures for multilayer perceptron (MLP) and attention layers of Llama 3 8b model (Dubey et al. (2024)).

The commonsense reasoning tasks comprise of 8 sub-tasks, each of them contains a predefined training and a testing set. We follow the setting of Hu et al. (2023) and amalgamate the training datasets from all 8 tasks to create the final training dataset and conduct evaluations on the individual testing dataset for each task.

The hyperparameter tuning protocol and the selected hyperparameters are provided in F. Table 1 contains the accuracy of the trained model’s responses on the test dataset. Within the LoRA framework, the proposed Riemannian fine-tuning method delivers clear performance gains. The Riemannion optimizer consistently outperforms LoRA, DoRA, and achieves superior metric results compared to the standard Muon optimizer applied to LoRA factors (see Section 3.1). Furthermore, relative to other Riemannian-geometry–aware approaches such as RPrecAdamW (Zhang & Pilanci, 2024), our

method also demonstrates better results. Finally, the variance of outcomes for the proposed method is the smallest among all compared approaches.

- Table 1: The average accuracy (in %) among 8 tasks of fine-tuned Llama 3-8b using different approaches, tested on Commonsense Reasoning benchmark. LoRA rank is set to 16.

Task BoolQ PIQA SIQA hella- wino- ARC-E ARC-C OBQA All Initialization swag grande

Raw 65.0 76.6 73.0 66.1 61.3 92.5 82.3 79.6 74.5 Adam 74.8±1.9 89.8±0.9 82.6±0.6 96.2±0.3 87.9±1.2 92.4±0.7 84.9±0.7 88.5±0.4 87.1±0.6 DoRA 74.8±0.8 89.4±0.5 82.4±0.7 95.9±0.1 87.8±0.4 90.7±1.2 83.8±0.7 87.8±0.6 86.6±0.3 Muon 72.9±0.0 86.4±0.5 80.8±0.2 94.1±0.2 84.4±0.0 84.2±0.9 77.3±2.5 83.9±1.1 83.0±0.6 DoneRITE 72.2±0.3 88.6±0.1 82.0±0.6 95.1±0.1 85.6±0.2 87.7±1.5 79.3±2.6 85.7±0.5 84.5±0.5 RPrecAdamW 75.8±0.4 89.5±0.4 82.4±0.2 96.1±0.2 87.7±0.9 90.6±1.6 84.1±1.1 87.7±0.5 86.8±0.4 Riemannion 75.7±0.7 91.2±0.2 83.5±0.6 96.7±0.0 88.6±0.4 93.6±0.3 86.4±0.4 89.3±0.8 88.1±0.2

- 7.2 SUBJECT-DRIVEN GENERATION

Subject-driven generation (Ruiz et al., 2023; Gal et al., 2023) is a task in which the user provides several reference photos of an object, called a concept, and uses a diffusion model to generate this concept with certain conditions (e.g., a textual prompt). One way to solve this task is to fine-tune a pre-trained diffusion model using this small set of reference images. However, this technique leads to a degradation in understanding of the conditions and to a fast overfitting of the concept. Furthermore, it requires a high computational cost due to the large number of trainable parameters. This is why previous works, such as (Qiu et al., 2023; Liu et al., 2024; Hu et al., 2022; Tewel et al., 2023; Han et al., 2023; Gorbunov et al., 2024), propose training only a lightweight parameterization for the base model. In this section, we demonstrate the performance of our parameterization in this task.

In our experiments, we used Stable Diffusion 2 (Rombach et al., 2022) as the base model. We choose LoRA Hu et al. (2022) as a baseline and train both models with ranks of 4, 8 and 16.

We predict the parameterization of the q, k, v, and out.0 matrices in all attention layers. The Dreambooth dataset (Ruiz et al., 2023) was used in all our experiments. LoRA was trained using the Adam optimizer. In Figure 2, we present a visual comparison of LoRA with different ranks. As can be seen, even for complex concepts such as ’robot toy’, our method requires only 600 steps to learn the concept while preserving appropriate text similarity. We found that for subject-driven generation tasks, the lower the rank, the faster our method converges. To evaluate this, we also calculated metrics for different learning rates on a subset of the Dreambooth dataset. We use CLIP to measure text similarity and DINO to measure image similarity. Figure 1 shows that, even with different learning rates, our method achieves more accurate results in concept preservation. Further details and a visual comparison can be found in the Appendix G.

0.65

LoRA lr=2e-5 LoRA lr=4e-5 LoRA lr=8e-5 LoRA lr=2e-4 Ours lr=5e-6

0.72

0.60

IS

- Ours lr=1e-5

- Ours lr=2e-5

0.55

Ours lr=4e-5

0.50

0.315 0.320 0.325 0.330 0.335

TS

Figure 1: Comparison of text and image similarities for LoRA and our method with rank 4 at different learning rates on 400 step.

- 8 CONCLUSION

In this work, we propose a novel fully Riemannian framework that integrates a new muon-based optimization method, locally-optimal initialization, and an efficient implementation. This integrated approach yields a reliable reparametrization-invariant method that outperforms competing approaches on fine-tuning large language models (LLMs) and exhibits additional favorable properties for lowrank approximations in diffusion models. Given the promising empirical results, a natural direction for future research is to investigate the theoretical properties of the proposed method.

[Figure 1]

Figure 2: Visual results for Subject-driven generation on 600 training step.

- 9 REPRODUCIBILITY STATEMENT

The hyperparameter selection procedure is described in Appendix F. The datasets used in the experiments and the corresponding preprocessing steps are detailed in Appendix E. The proof of the stated assumptions is provided in Appendix A. The hyperparameters for Subject-driven generation task are provided in the Appendix G.

REFERENCES

P-A Absil and Ivan V Oseledets. Low-rank retractions: a survey and new results. Computational Optimization and Applications, 62(1):5–29, 2015.

P-A Absil, Robert Mahony, and Rodolphe Sepulchre. Optimization algorithms on matrix manifolds. In Optimization Algorithms on Matrix Manifolds. Princeton University Press, 2009.

Christopher G Baker. Riemannian manifold trust-region methods with applications to eigenproblems. The Florida State University, 2008.

Jeremy Bernstein. Deriving muon, 2025. URL https://jeremybernste.in/writing/ deriving-muon.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. PIQA: Reasoning about Physical Commonsense in Natural Language. In Proceedings of the AAAI conference on artificial intelligence, number 05 in 34, pp. 7432–7439, 2020.

Nicolas Boumal. An introduction to optimization on smooth manifolds. Cambridge University Press, 2023.

Stephen Boyd and Jon Dattorro. Alternating projections. 2003. URL https://web. stanford.edu/class/ee392o/alt_proj.pdf.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Franz Louis Cesista. Muon and a selective survey on Steepest Descent in Riemannian and non-Riemannian Manifolds, April 2025. URL http://leloykun.github.io/ponder/ steepest-descent-non-riemannian/.

Ward Cheney and Allen A Goldstein. Proximity maps for convex sets. Proceedings of the American Mathematical Society, 10(3):448–450, 1959.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies,

NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pp. 2924–2936. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1300. URL https://doi.org/10.18653/v1/n19-1300.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge. CoRR, abs/1803.05457, 2018. URL http://arxiv.org/abs/1803.05457.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The Llama 3 herd of models. CoRR, abs/2407.21783, 2024. doi: 10.48550/ARXIV.2407.21783. URL https://doi.org/10.48550/arXiv.2407.21783.

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7), 2011.

Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218, 1936.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview. net/forum?id=NAQvF08TcyG.

Mikhail Gorbunov, Nikolay Yudin, Vera Soboleva, Aibek Alanov, Alexey Naumov, and Maxim Rakhuba. Group and shuffle: Efficient structured orthogonal parametrization. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

N. Halko, P. G. Martinsson, and J. A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288,

2011. doi: 10.1137/090771806. URL https://doi.org/10.1137/090771806.

Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7323–7334, 2023.

Soufiane Hayou, Nikhil Ghosh, and Bin Yu. LoRA+: Efficient Low Rank Adaptation of Large Models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id= NEv8YqBROO.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 1(2):3, 2022.

Jiang Hu, Jiaxi Cui, Lin Lin, Zaiwen Wen, Quanzheng Li, et al. Retraction-free optimization over the Stiefel manifold with application to the loRA fine-tuning, 2024.

Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, Ee-Peng Lim, Lidong Bing, Xing Xu, Soujanya Poria, and Roy Ka-Wei Lee. LLM-adapters: An adapter family for parameter-efficient fine-tuning of large language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 5254–5276. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.319. URL https://doi.org/10.18653/ v1/2023.emnlp-main.319.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https: //kellerjordan.github.io/posts/muon/.

Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. International

Conference on Learning Representations, 12 2014. John M Lee. Smooth manifolds. Springer, 2003.

Weiyang Liu, Zeju Qiu, Yao Feng, Yuliang Xiu, Yuxuan Xue, Longhui Yu, Haiwen Feng, Zhen Liu, Juyeon Heo, Songyou Peng, Yandong Wen, Michael J. Black, Adrian Weller, and Bernhard Schölkopf. Parameter-efficient orthogonal finetuning via butterfly factorization. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=7NzgkEdGyr.

Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

Fanxu Meng, Zhaohui Wang, and Muhan Zhang. PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. Advances in Neural Information Processing Systems, 37:121038–121072, 2024.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pp. 2381–2391. Association for Computational Linguistics, 2018. doi: 10.18653/V1/D18-1260. URL https://doi.org/10.18653/v1/d18-1260.

Zhanfeng Mo, Long-Kai Huang, and Sinno Jialin Pan. Parameter and Memory Efficient Pretraining via Low-rank Riemannian Optimization. In The Thirteenth International Conference on Learning Representations, 2025.

Alexander Novikov, Maxim Rakhuba, and Ivan Oseledets. Automatic differentiation for Riemannian optimization on low-rank matrix and tensor-train manifolds. SIAM Journal on Scientific Computing, 44(2):A843–A869, 2022.

Uliana Parkina and Maxim Rakhuba. Coala: Numerically stable and efficient framework for contextaware low-rank approximation, 2025. URL https://arxiv.org/abs/2507.07580.

Boris T Polyak. Some methods of speeding up the convergence of iteration methods. Ussr computational mathematics and mathematical physics, 4(5):1–17, 1964.

Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard Schölkopf. Controlling text-to-image diffusion by orthogonal finetuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=K30wTdIIYc.

Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical statistics, pp. 400–407, 1951.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, June 2022.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22500– 22510, 2023.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. Communications of the ACM, 64(9):99–106, 2021.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. SocialIQa: Commonsense Reasoning about Social Interactions. CoRR, abs/1904.09728, 2019. URL http: //arxiv.org/abs/1904.09728.

Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-toimage personalization. In ACM SIGGRAPH 2023 Conference Proceedings, pp. 1–11, 2023.

Tijmen Tieleman. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 4(2):26, 2012.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023a. doi: 10.48550/ARXIV.2302.13971. URL https://doi.org/10.48550/arXiv.2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288, 2023b. doi: 10.48550/ARXIV.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288.

Nickolay Trendafilov and Michele Gallo. Multivariate data analysis on matrix manifolds. Springer, 2021.

Bart Vandereycken. Low-rank matrix completion by Riemannian optimization. SIAM Journal on Optimization, 23(2):1214–1236, 2013.

Hanqing Wang, Yixia Li, Shuo Wang, Guanhua Chen, and Yun Chen. MiLoRA: Harnessing minor singular components for parameter-efficient LLM finetuning. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2025 Volume 1: Long Papers, Albuquerque, New Mexico, USA, April 29 - May 4, 2025, pp. 4823–4836. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.NAACL-LONG.248. URL https://doi.org/10.18653/v1/2025.naacl-long.248.

Shaowen Wang, Linxi Yu, and Jian Li. LoRA-GA: Low-rank adaptation with gradient approximation. Advances in Neural Information Processing Systems, 37:54905–54931, 2024.

Scott Wisdom, Thomas Powers, John Hershey, Jonathan Le Roux, and Les Atlas. Full-Capacity Unitary Recurrent Neural Networks. Advances in neural information processing systems, 29, 2016.

Yibo Yang, Xiaojie Li, Zhongzhu Zhou, Shuaiwen Song, Jianlong Wu, Liqiang Nie, and Bernard Ghanem. CorDA: Context-oriented decomposition adaptation of large language models for taskaware parameter-efficient fine-tuning. Advances in Neural Information Processing Systems, 37: 71768–71791, 2024.

Jui-Nan Yen, Si Si, Zhao Meng, Felix Yu, Sai Surya Duvvuri, Inderjit S Dhillon, Cho-Jui Hsieh, and Sanjiv Kumar. Lora done rite: Robust invariant transformation equilibration for lora optimization. arXiv preprint arXiv:2410.20625, 2024.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Anna Korhonen, David R. Traum, and Lluís Màrquez (eds.), Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pp. 4791– 4800. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1472. URL https://doi.org/10.18653/v1/p19-1472.

Fangzhao Zhang and Mert Pilanci. Riemannian preconditioned loRA for fine-tuning foundation models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

- A INITAL POINT SEARCH In this section we introduce the proof of Theorem 5.1

Proof. In order to represent the optimization task (16) in a more simple way, we will use a slightly different formula for the projection of the full loss gradient onto the tangent space of the fixed-rank manifold. Let

∆W = ALB⊤ = ABR⊤ ∈ Mr, and the tangent space is parametrized as follows

#### T∆WMr = {AB˙ R⊤ + ALB˙ ⊤ | B˙ ∈ Rn×r,A˙ ∈ Rm×r,A⊤LA˙ = 0},

then one may derive an orthogonal projection formula for any matrix Z (see, e.g. (Boumal, 2023, eq. 7.53))

∆WMrZ = Z − (I − ALA⊤L)Z(I − BRBR⊤) ∈ T∆WMr.

PT

As the tangent space is a linear space, the operation of orthogonal projection can be written as an optimization task

∥Z − ξ∥2F. (19) Since

PT

∆WMrZ = arg min

ξ∈T∆W Mr

∥∇L∥2F = ∥(∇L − PT

∆WMr∇L∥2F =

∆WMr∇L) + PT

∆WMr∇L∥2F + ∥PT

∆WMr∇L∥2F + 2⟨∇L − PT

= ∥∇L − PT

∆WMr∇L,PT

∆WMr∇L⟩

∆WMr∇L∥2F + ∥PT

∆WMr∇L∥2F + 2⟨PT

= ∥∇L − PT

∆WMr (∇L − PT

∆WMr∇L),∇L⟩

∆WMr∇L∥2F + ∥PT

∆WMr∇L∥2F + 2⟨PT

,∇L⟩

= ∥∇L − PT

∆WMr∇L − PT

∆WMr∇L

=0

∆WMr∇L∥2F + ∥PT

∆WMr∇L∥2F so the optimization task (16) is equivalent to

= ∥∇L − PT

∆WMr∇L∥2F → min ∆W

∥∇L − PT

, which, in turn, using (19) is equivalent to the task

∥∇L − ξ∥2F.

min

min

ξ∈T∆W Mr

∆W

Due to the fact that every vector of the tangent space is an element of a set of all matrices with 2rbounded rank M≤2r, one may use the Eckart-Young-Mirsky theorem (see (Eckart & Young, 1936)) to obtain the following lower bound:

∆WMr∇L∥2F ≥ ∥∇L − truncSVD(∇L,2r)∥F,∀ ∆W ∈ Mr,ξ ∈ T∆WMr. (20) But it is possible to ensure ∆W∗(0) ∈ Mr and ξ∗ ∈ T∆W(0)

∥∇L − PT

Mr which turn the inequality (20) into the equality. One may take

∗

∆W∗(0) = AL(αB)⊤ = αU1,rVr,⊤2r = (αA)BR⊤, α ∈ R, ξ∗ = AB˙ R⊤ + ALB˙ ⊤ = (Ur,2rΣr,2r)BR⊤ + AL (Σ1,rV1,r)⊤ = truncSVD(∇L,2r).

Note, that A˙⊤AL = 0. So the initialization ∆W∗(0) ensures that truncSVD(∇L,2r) lies in the tangent space T∆W(0)

Mr, which means that the first step of the RiemannLoRA Algorithm 6 will get the Riemannian gradient equal to truncSVD(∇L,2r).

∗

To generalize the result, we should change the parametrization of the tangent space. Because of the fact that each of the tangent vectors ξ lies in M≤2r, we may use an unconstrained skeleton decomposition (without constraints for A˙ and AL):

B ˙∗⊤ B∗⊤

B ˙∗⊤ B∗⊤

= A∗,A˙∗ I2r

ξ∗ = A∗,A˙∗

=

B ˙∗⊤ B∗⊤

, S ∈ R2r×2r, detS ̸= 0.

= A∗,A˙∗ S S−1

Representing S and its inverse as block matrices:

S11 S12 S21 S22

C11 C12 C21 C22

, S−1 =

S =

, one arrives to (17):

ξ∗ = A∗,A˙∗

S11 S12 S21 S22

C11 C12 C21 C22

B ˙∗ B∗ ⊤ ,

S11 S21

[C21 C22] B ˙∗′ B′∗ ⊤ .

∆W∗(0) = A′∗,A˙′∗

To derive the version that is utilized in practice, one may take S to be block-diagonal with S11 = αIr,S22 = Ir,α ∈ R.

| |
|---|

- B RIEMANNION WITH LOI

The complete framework is summarized in Algorithm 5, which consists of an LOI initialization step followed by max_iters iterations of the Riemannion optimizer.

Specifically, the first 3 lines are dedicated to the computation of LOI via BackPropRSVD for given layer weights and oversampling parameter p, that increases the accuracy of the singular approximation. The 4-th step initializes Heavy-Ball momentum matrices. Therefore, the asymptotical complexity of this stage is determined by the complexity of Algorithm 3.

- Algorithm 5 Riemannion with LOI

Require: Weights W ∈ Rm×n, rank r ∈ N, step size η, momentum coefficient β, weight decay coefficient γ, oversampling parameter p, power-step parameter q.

Ensure: Tuning parameters ∆W∗ ∈ Mr. Function:

- 1: AL,_,B⊤ := BackPropRSVD(2r,p,q,L,W). // O (m + n)r2
- 2: AL,B⊤ := AL[: , : r], B[: ,r: ]⊤.
- 3: W′ := W − ALB⊤. // O (mn)
- 4: AHB,BHB := 0,0.
- 5: for i := 0,...,max_iters do
- 6: BR := qr(B).Q. // O nr2
- 7: A˙ := ∇Z1L W′ + Z1BR⊤ + ALZ2⊤ |Z1=0,Z2=B. // = ∇WL(W)BR
- 8: B˙ := ∇Z2L W′ + Z1BR⊤ + ALZ2⊤ |Z1=0,Z2=B. // = ∇WL(W)⊤AL
- 9: A˙prev,B˙prev := ProjectLR((AM,BM),AL,BR). // O (m + n)r2
- 10: A,˙ B˙ := βA˙prev + (I − ALA⊤L)A,˙ βB˙prev + B˙ // O (n + m)r2
- 11: A,˙ B˙ := ProjectLR OrthoLR AL,BR,A,˙ B,r,˙ ,AL,BR // O (m + n)r2 + r3
- 12: U,Σ,V ⊤ := RetractionLR −ηA,A˙ L , BR,−η(B˙ + γBR) // O (m + n)r2 + r3
- 13: AHB,BHB := [A,A˙ L], [BR,B˙ ]
- 14: AL,B := U,ΣV ⊤ // O nr2
- 15: return AL,B.

- C RIEMANN-SGD ALGORITHM

In Algorithm 6, we present the RiemannSGD version of the fine-tuning algorithm. The first steps are dedicated to the computation of the optimal initialization using BackPropRSVD(Algorithm 3). After initialization, the algorithm follows an Adam-like procedure adapted to the Riemannian setting: it maintains exponentially smoothed estimates of momentum and gradient norms (4th step). The overall cycle encapsulates the computation of the Riemannian gradient at the current point(steps 7−8), the update of the adaptive momentum terms via ProjectLR(Algorithm 2), and the retraction step that projects the updated tangent direction back onto the manifold(steps 15−17). This approach has asymptotical complexity of O((m+n)r2+r3) and (2(q+1)+max_iters) amount of backward calls.

Algorithm 6 RiemannSGD with LOI Require: Weights W ∈ Rm×n, rank r ∈ N, step size η, momentum coefficient β, oversampling parameter p, power-step parameter q, simulate_Adam, Adam momentum coefficient γ.

Ensure: Tuning parameters ∆W∗ ∈ Mr. Function:

- 1: AL,_,B⊤ := BackPropRSVD(2r,p,q,L,W). // O (m + n)r2
- 2: AL,B⊤ := AL[: , : r], B[: ,r: ]⊤.
- 3: W′ := W − ALB⊤. // O (mn)
- 4: AHB,BHB,SA,SB := 0.
- 5: for i := 0,...,max_iters do
- 6: BR := qr(B).Q. // O nr2
- 7: A˙ := ∇Z1L W + Z1BR⊤ + ALZ2⊤ |Z1=0,Z2=B.
- 8: B˙ := ∇Z2L W + Z1BR⊤ + ALZ2⊤ |Z1=0,Z2=B.
- 9: A˙prev,B˙prev := ProjectLR((AHB,BHB),AL,BR). // O (m + n)r2
- 10: A,˙ B˙ := βA˙prev + (1 − β)(I − ALA⊤L)A,˙ βB˙prev + (1 − β)B˙ // O nr2
- 11: if simulate_Adam then
- 12: SA,SB := γ∥A˙∥F + (1 − γ)SA, γ∥B˙ ∥F + (1 − γ)SB // O ((m + n)r)
- 13: A,˙ B˙ := A/S˙ A,B/S˙ B // O ((m + n)r)
- 14: U,Σ,V ⊤ := RetractionLR ηA,A˙ L , BR,ηB˙ + B // O (m + n)r2 + r3
- 15: AHB,BHB := [A,A˙ L], [BR,B˙ ]
- 16: AL,B := U,ΣV ⊤ // O nr2
- 17: return AL,B.

Table 2: The average accuracy (in %) among 8 tasks of fine-tuned Llama 3.2-1b using different SGD variants, tested on Commonsense Reasoning benchmark. The «R-» prefix stands for Riemannian, the postfix «-LOI» stands for locally optimal initialization. LoRA rank is set to 16. In all experiments excpet for «-LOI» the A factor in AB⊤ has orthonormal columns and B is zero.

Task BoolQ PIQA SIQA hella- wino- ARC-E ARC-C OBQA All Initialization swag grande

Raw 40.1 55.4 50.3 25.8 50.0 61.9 41.8 42.8 46.0 LoRA 64.0±0.2 75.3±0.3 69.6±0.4 82.2±0.1 53.0±1.0 75.4±1.7 56.5±0.6 67.1±2.3 67.9±0.4 LoRA-LOI 64.9±0.4 76.8±0.3 71.2±2.2 84.1±0.1 56.7±0.6 77.0±4.2 61.1±3.6 68.9±3.3 70.1±1.6 RSLoRA 64.5±0.2 77.2±0.3 68.1±0.8 86.3±0.1 58.2±0.5 76.5±1.5 58.6±2.9 70.0±1.9 70.0±0.8 RiemannLoRA 65.1±0.4 77.7±0.3 69.1±1.5 85.8±0.2 58.5±0.4 74.4±1.9 56.9±1.3 69.3±2.5 69.6±0.9 RiemannLoRA-LOI 65.2±0.4 79.4±0.4 75.6±0.4 87.3±0.1 62.4±0.4 79.9±0.8 63.6±1.9 73.8±0.5 73.4±0.3

- D LMO BOUNDS

Let A ∈ TWMr be a vector from the tangent space. Consider the problem

⟨A,X⟩ → max

, s.t. ∥X∥2 ≤ 1,

X

(21)

whose (any) solution will be denoted by X∗ = Ortho(A). Denote f∗ = ⟨A,X∗⟩ the optimal value. Consider the restricted (LMO) problem

⟨A,X⟩ → max

,

X

(22)

s.t. ∥X∥2 ≤ 1, X ∈ TWMr.

Let X∗ denote an optimizer of the task derived in (22) and f∗ = ⟨A,X∗⟩ its optimal value.

- Proposition D.1. Assume A ∈ TWMr and let P be the orthogonal projector onto TWMr. If ∥X∗ − PX∗∥2 ≤ ε

for some ε ≥ 0, then

1 1 + ε

f∗ ≤ f∗ ≤ f∗.

Proof. Since A ∈ TWMr, we have PA = A and P = P⊤. By assumption ∥X∗ − PX∗∥2 ≤ ε and, because ∥X∗∥2 = 1, the reverse triangle inequality yields

∥X∗∥2 − ∥PX∗∥2 ≤ ε, hence

1 − ε ≤ ∥PX∗∥2 ≤ 1 + ε. Define the normalized matrix

Y :=

PX∗ ∥PX∗∥2

∈ TWMr, ∥Y ∥2 = 1.

By definition of f∗ (the maximum of ⟨A,·⟩ over {X ∈ TWMr : ∥X∥2 ≤ 1}) we have

f∗ ≥ ⟨A,Y ⟩ = ⟨A,PX∗⟩ ∥PX∗∥2

= ⟨PA,X∗⟩ ∥PX∗∥2

= ⟨A,X∗⟩ ∥PX∗∥2

=

f∗ ∥PX∗∥2

.

Using the inequality ∥PX∗∥2 ≤ 1 + ε we obtain the claimed lower bound

f∗ ≥

f∗ 1 + ε

.

The upper bound f∗ ≤ f∗ is trivial because the feasible set of the task derived in (22) is a subset of the feasible set of the problem described in (21). Combining the two inequalities gives

1 1 + ε

f∗ ≤ f∗ ≤ f∗, which completes the proof.

| |
|---|

- Proposition D.2. (Upper bound on the projection residual.) With the notation above, the residual ε := ∥X∗ − PX∗∥2

satisfies ε ≤ 1.

Proof. Write the thin SVD of the base point W as W = UrΣrVr⊤, where Ur ∈ St(m,r) and Vr ∈ St(n,r) have r orthonormal columns. Let U⊥ and V⊥ be orthonormal complements so that [Ur U⊥] ∈ Rm×2r and [Vr V⊥] ∈ Rn×2r are orthogonal matrices (here we assume m,n ≥ 2r; if m or n is smaller the argument is adapted in the obvious way).

Because A ∈ TWMr, one can write (using a suitable 2r × 2r block representation) (Vandereycken, 2013)

A ˜ B˜ C˜ 0

Vr⊤ V⊥⊤

A = [Ur U⊥]

,

where A˜,B˜,C˜ ∈ Rr×r (the (2,2) block vanishes for tangent-space elements of this form). The rank of A is therefore at most 2r.

Let X∗ = Ortho(A) be a maximizer of (21); represent X∗ in the same block basis,

Vr⊤ V⊥⊤

X∗ = [Ur U⊥] A B C D

, where the 2r × 2r block matrix is orthogonal,

A B C D

∈ O(2r), (23)

where O(2r) denotes the orthogonal group in R2r×2r. By Lemma D.1 below we have

Vr⊤ V⊥⊤

PX∗ = [Ur U⊥] A B C 0

. Therefore

Vr⊤ V⊥⊤

0 0 0 D

= U⊥DV⊥⊤, and hence

X∗ − PX∗ = [Ur U⊥]

∥X∗ − PX∗∥2 = ∥D∥2. Because the block matrix in (23) is orthogonal, its lower block

C D

∈ R2r×r

has orthonormal columns. Thus D C 2 = 1, and in particular ∥D∥2 ≤ 1. Consequently ε = ∥X∗ − PX∗∥2 ≤ 1, as claimed.

| |
|---|

### Lemma D.1. Let

W = UrΣrVr⊤, Ur ∈ St(m,r), Vr ∈ St(n,r),

and write an arbitrary matrix X ∈ Rm×n in the block basis determined by [Ur U⊥] and [Vr V⊥],

Vr⊤ V⊥⊤

X = [Ur U⊥] A B C D

.

Then the orthogonal projector P onto TWMr acts as

Vr⊤ V⊥⊤

PX = [Ur U⊥] A B C 0

.

Proof. One convenient expression for the orthogonal projector onto the tangent space at W = UrΣrVr⊤ is

P(·) = (·) − (I − UrUr⊤)(·)(I − VrVr⊤). Applying this to X written in the chosen block basis yields

PX = X − (I − UrUr⊤)X(I − VrVr⊤) =

Vr⊤ V⊥⊤ − [0 U⊥] A B

= [Ur U⊥] A B C D

0 V⊥⊤

C D

Vr⊤ V⊥⊤

= [Ur U⊥] A B C D

0 0 0 D

−

Vr⊤ V⊥⊤

= [Ur U⊥] A B C 0

,

which proves the lemma. Remark D.1. In our experiments we observed that the residual ε is typically small (in our reported runs it did not exceed 0.1).

| |
|---|

- E DATASET E.1 COMMONSENSE REASONING

Following the approach of Hu et al. (2023), we combine the training datasets from all 8 tasks to form the final training set and evaluate performance on each task’s individual test dataset. We adopt queries structure from Hu et al. (2023) to Llama 3.2 instruct template. The prompt structure is demonstrated in Table 3.

- F HYPERPARAMETERS

For each optimization method, we carefully preselected the learning rate (step size). For the Riemannion method, we set the momentum to 0.9, and for the Muon method, we set the momentum to 0.95. In both cases, we additionally tuned the weight-decay hyperparameter (see Algorithm 4). The final choices of these hyperparameters are summarized in Table 4.

In the Commonsense reasoning benchmark, the hyperparameters are reported in Table 5 for the SGD-like methods and in Table 4 for the Adam-like methods. For the LOI method, we selected the following hyperparameters: the backprop RSVD oversampling parameter was set to rk = 16, the backprop powerstep parameter was set to q = 1, and the multiplier α was set to −√0.r01.

The non-tuned hyperparameters used for experiments on the Commonsense Reasoning dataset are presented in 6.

- G SUBJECT-DRIVEN GENERATION

Training details We used Stable Diffusion-2-base model with a batch size of 4 for all experiments. For both methods, we set the betas to 0.9 and 0.999 and the weight decay to 0.1. In all variations of our approach, we set q = 15, p = rank and

α = −√rank1 . We used a learning rate of 2e-5 to train both our and the LoRA models, which were used to generate images shown in Figures 2 and 3.

Evaluation details We used the DreamBooth dataset, which contains 25 different prompts and 30 various concepts. Due to computational costs, we only used half of the proposed concepts to evaluate metrics: can, candle, cat, cat2, colorful_sneaker, dog2, dog3, dog5, dog6, dog7, dog8, fancy_boot, grey_sloth_plushie, pink_sunglasses, vase. To measure the similarity between the original concept and the generated images, we used Image Similarity (IS): we synthesized 30 images for the base prompt «a photo of a V*» and 10 images for each of 25 editing prompts (like «a V* on top of a dirt road»). We then measured the average pairwise cosine similarity with reference photos of the concept using the DINO model. To check the correspondence between the generated images and the textual prompts, we calculated Text Similarity (TS): we evaluated each concept with each of 25 prompts, synthesizing 10 images per prompt, and calculating the average pairwise cosine similarity with the prompts using the CLIP ViTB/32 model.

Additional results Figure 3 shows an additional visual comparison of our method and LoRA after 600 training steps. As can be seen, our method learns the concept much faster than the original LoRA, while preserving editing capabilities.

- H ADDITIONAL EXPERIMENTAL RESULTS H.1 ABLATION STUDY ON INITIALIZATION

The fine-tuning optimization was carried out by AdamW (Loshchilov & Hutter (2019)) First of the first, for every approach tested we preselected a suitable optimization step sizes. The list of all selected hyper-parameters for each method is specified in the Appendix F. Table 7 contains the accuracy of the trained model’s responses on the test dataset. The notation «LoRA-A» in the table

Table 3: The structure of the queries for the Commonsense reasoning dataset

Task Role Fine-tuning Data Template BoolQ system Please answer the following question with True or False.

Follow the answer format, full answer not needed. user Question: [QUESTION]

Answer format: True/False assistant The correct answer is [ANSWER]

PIQA system Please choose the correct solution to the question.

Follow the answer format, full answer not needed. user Question: [QUESTION]

- Solution1: [SOLUTION_1]
- Solution2: [SOLUTION_2] Answer format: Solution1/Solution2

assistant The correct answer is [ANSWER]

SIQA system Please choose the correct answer to the question based on the context provided. Follow the answer format, full answer not needed.

user Context: [CONTEXT] Question: [QUESTION]

- A: [ANSWER_A]
- B: [ANSWER_B]
- C: [ANSWER_C] Answer format: A/B/C

assistant The correct answer is [ANSWER] hellaswag system Please choose the correct ending to complete the given sentence.

Follow the answer format, full answer not needed. user [ACTIVITY_lABEL]: [CONTEXT]

- Ending1: [ENDING_1]
- Ending2: [ENDING_2]
- Ending3: [ENDING_3]
- Ending4: [ENDING_4] Answer format: Ending1/Ending2/Ending3/Ending4

assistant The correct answer is [ANSWER]

winogrande system Please choose the correct answer to fill in the blank to complete the given sentence. Follow the answer format, full answer not needed.

user Sentence: [SENTENCE]

- Option1: [OPTION_1]
- Option2: [OPTION_2] Answer format: Option1/Option2

assistant The correct answer is [ANSWER] ARC-e & ARC-c system Please choose the correct answer to the question.

Follow the answer format, full answer not needed. user Question: [QUESTION]

- Answer1: [ANSWER_1]
- Answer2: [ANSWER_2]
- Answer3: [ANSWER_3]
- Answer4: [ANSWER_4] Answer format: Answer1/Answer2/Answer3/Answer4

assistant The correct answer is [ANSWER] OBQA system Please choose the correct answer to the question.

Follow the answer format, full answer not needed. user Question: [QUESTION]

- Answer1: [ANSWER_1]
- Answer2: [ANSWER_2]
- Answer3: [ANSWER_3]
- Answer4: [ANSWER_4] Answer format: Answer1/Answer2/Answer3/Answer4

assistant The correct answer is [ANSWER]

- Table 4: The parameters for different Adam variants for fine-tuning on the Commonsense reasoning dataset

Optimizer learning rate weight decay

Adam 0.0002 0.01 DoRA 0.0003 0.01 Muon 0.0005 0.2 DoneRITE 0.0005 0.0001 RPrecAdamW 0.0005 0.5 Riemannion 0.0001 0.00316

- Table 5: The parameters for different SGD variants (RiemannLoRA with simulate_Adam flag disabled) for fine-tuning on the Commonsense reasoning dataset

Optimizer learning rate LoRA 0.1 LoRA-LOI 0.06 RSLoRA 0.1 RiemannLoRA 0.1 RiemannLoRA-LOI 0.07

- Table 6: Other non-tuned hyperparameter configurations for experiments on Commonsense reasoning dataset

Dataset Commonsense reasoning Hyperparameters

Rank r 16 Dropout 0.05 LR Scheduler Linear Batch size 64 Epochs 2 Warmup ratio 0.1

[Figure 2]

##### Figure 3: Additional visual comparison of our method and LoRA, checkpoint 600

means the utilization of the vanilla LoRA with non-zero A, the notation «LoRA-B» means the same with non-zero B, the notations «Stiefel-A» and «Stiefel-B» indicate vanilla LoRA with orthonormal initialization (for matrix A and B respectively). The naming of «Stiefel-both» involves taking the factors A,B with orthonormal columns and with initialization like in (15).

- Table 7: The average accuracy among 8 tasks of fine-tuned Llama 3.2-1b via Adam using different LoRA initialization variants, tested on Commonsense reasoning benchmark. «LOI» stands for locally optimal initialization. LoRA rank is set to 16.

Task BoolQ PIQA SIQA hella- wino- ARC- ARC- OBQA All Initialization swag grande E C

Raw 40.1 55.4 50.3 25.8 50.0 61.9 41.8 42.8 46.0

- LoRA-A 66.2 80.2 76.0 87.0 65.1 78.4 63.6 76.3 74.1
- LoRA-B 65.9 77.9 74.2 82.9 61.2 73.9 60.6 72.2 71.1 Pissa 66.4 80.1 75.6 87.6 63.1 77.6 64.3 75.0 73.7

- Stiefel-A 65.4 79.5 74.9 87.2 62.4 79.3 62.3 75.5 73.3
- Stiefel-B 66.4 80.6 76.0 87.5 64.3 78.5 64.9 74.6 74.1 Stiefel-both 66.3 79.7 75.8 86.4 64.0 78.3 62.4 73.9 73.4 LOI 65.6 81.3 75.7 87.7 65.7 77.9 65.0 75.2 74.3 LoRA-GA 65.4 77.1 74.9 84.5 58.3 75.4 61.9 72.2 71.2

- I COMPUTATIONAL COST

To measure execution time, we used a virtual server instantiated on a compute node equipped with an Intel Xeon Gold 6240R CPU and Nvidia Tesla V100 GPU. The virtual machine was provisioned with 8 CPU cores, 16 GB of RAM and a single Tesla V100 GPU. Experiments were executed using the Hugging Face transformers library; timing corresponds to the execution time of the trainer.train() call. Measurements were conducted according to the following procedure. For each combination of method, LoRA adapter rank, and batch size, four timing runs were performed and the minimum execution time was selected. The measured quantity was the time spent on 16 optimizer steps. Time spent on initialization was excluded. For each combination of rank and batch size, execution times for each of the methods were measured sequentially. In the present work, rank refers to the rank of the LoRA adapter, and method denotes one of two optimization schemes: Adam (baseline) and Riemannion (proposed method).

Figure 4 shows the relative increase in per-step execution time of the proposed Riemannion method compared with Adam, computed as

Relative overhead on Llama-3-8b

35

r

32.5

4 8 16

30

27.5

25

22.5

Overhead(%)

20

17.5

15

12.5

10

7.5

5

2.5

0

8 16 32 64 128

Batch size

Figure 4: Relative Time Cost calculated as (TRiemannion − TAdam) TAdam of Riemannion vs. Adam during Llama 3-8B fine-tuning, as a function of LoRA rank and batch size.

- J LLM USAGE

We used an LLM only for minor language polishing to improve readability and grammar; it was not involved in research ideation, methodology, analysis, or substantive writing, and all research ideas and arguments were developed entirely by the authors.

