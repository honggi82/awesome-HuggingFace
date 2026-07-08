# arXiv:2605.07825v1[cs.MM]8May2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

## Anisotropic Modality Align

### Xiaomin Yu1,2, Yijiang Li3, Yuhui Zhang4, Hanzhen Zhao2, Yue Yang4, Hao Tang5 Yue Song6, Xiaobin Hu2, Chengwei Qin1, Shuicheng Yan2, Hui Xiong1

1HKUST(GZ), 2NUS, 3UCSD, 4Stanford, 5PKU, 6THU

### Abstract

Training multimodal large language models has long been limited by the scarcity of high-quality paired multimodal data. Recent studies show that the shared representation space of pretrained multimodal contrastive models can serve as a bridge, enabling models to perform multimodal training with unimodal data. However, the key premise of this paradigm remains insufficiently understood: can representations from different modalities be reliably interchanged? The core obstacle lies in the persistent Modality Gap in the shared space. In this work, we revisit the geometric nature of the modality gap. We find that modality representations already share compatible dominant semantic geometry. What truly hinders modality interchangeability is not a simple global shift, but an anisotropic residual structure concentrated along a small number of dominant directions. Based on this finding, we further propose the principle of anisotropic modality gap alignment: effective modality alignment should align with the target-modality distribution while preserving the semantic structure of the source modality. Guided by this principle, we propose an anisotropic geometric correction framework, AnisoAlign, for unpaired modality alignment. This framework leverages the internal geometric prior of the target modality and performs bounded correction on source-modality representations, thereby constructing substitute representations in the target modality. Experiments confirm its benefits in both geometric diagnostics and text-only MLLM training. Overall, this work recasts the modality gap from an empirical observation into a correctable, structured geometric phenomenon and provides a new representation alignment perspective for training multimodal models with unimodal data.

Date: May 11, 2026 Leader: Xiaomin Yu (yuxm02@gmail.com) Correspondence: Yue Song, Xiaobin Hu, Chengwei Qin Github: https://github.com/Yu-xm/Modality_Gap_Theory.git

### 1 Introduction

Multimodal contrastive learning models [7, 12, 20] typically map samples from different modalities into the same normalized representation space, so that semantically corresponding images and texts are close to each other in this space. However, a persistent phenomenon is that, even after large-scale contrastive pretraining, image and text representations often still maintain systematic geometric separation in the shared space. This phenomenon is commonly referred to as the Modality Gap [9, 17, 21]. Some studies exploit this property by geometrically correcting the source-modality representations in the shared representation space and aligning

- them with the target modality, thereby enabling multimodal large language models (MLLMs) [6, 10] to be trained using single-modality data and decoupling the dependence on paired multimodal data [1, 6].

However, existing methods still lack a systematic characterization of the modality gap: do the two modalities share compatible dominant semantic geometry? Does the remaining discrepancy mainly arise from a global centroid shift, or is it concentrated as structured residuals along specific directions? What kind of correction can both preserve source-modality semantics and move representations into the distributional support of the target modality? Answering these questions is particularly critical for unpaired modality alignment, because in the absence of paired supervision, alignment methods must rely on the intrinsic geometric structure of modality distributions to constrain the correction process [17, 21]. This leads to the basic question studied in this work: What kind of geometric discrepancy is the modality gap?

To answer this question, we revisit the modality gap through a sequence of geometric diagnostics. The results show that image and text representations are not arbitrary, unrelated distributions in the shared space. Instead, the two modalities already possess compatible dominant semantic geometry: their covariance spectra exhibit similar long-tail decay, and their principal subspace overlap is significantly higher than the random baseline. This indicates that multimodal contrastive pretraining has already established a shared dominant geometric backbone between the two modalities.

However, the remaining modality gap cannot be simply explained by a global centroid bias. We find that, after globally shifting text representations to the image-modality centroid, most of the cross-modal discrepancy still remains. Further spectral analysis shows that the mean-corrected residual is not isotropic noise, but an anisotropic structure concentrated along a small number of dominant directions. In other words, the modality gap mainly appears as a low-effective-dimensional, direction-dependent residual, rather than an unstructured random offset.

These diagnostics naturally lead to a modality alignment principle: effective modality alignment should not merely minimize global distributional discrepancy, but should satisfy two requirements simultaneously. First, it must preserve the semantic geometry already present in the source modality. Second, it must correct the dominant anisotropic residual directions that prevent the source modality from being compatible with the target-modality distribution. Matching only the target distribution may destroy semantic correspondence; preserving only source semantics may fail to enter the distributional support of the target modality. Therefore, modality alignment is essentially a structured geometric correction problem between semantic preservation and target-distribution compatibility.

Based on this principle, we propose an anisotropic alignment method, AnisoAlign, for unpaired modality alignment. The method first constructs a fixed dominant subspace decomposition, dividing the shared space into a statistically dominant subspace and its orthogonal complement. Then, within the dominant subspace, we introduce a blockwise polar parameterization that decomposes representations into radius and phase structures, thereby explicitly modeling anisotropic geometric variations along dominant directions. To avoid directly learning an unstable cross-modal mapping, we first pretrain a periodic phase prior using only target-modality samples, which captures the internal phase statistics of the target modality. Then, in the second stage, we perform bounded residual correction on source-modality representations, so that they gradually satisfy the target-modality prior while preserving instance-level semantic structure.

Extensive experiments support this view. At the representation level, AnisoAlign better matches the target-modality geometry while preserving source-modality semantics, achieving balanced local support compatibility and reducing dominant anisotropic residual directions. At the MLLM level, the resulting substitute representations lead to stronger performance in both fully text-only training and text-only pretraining

before visual instruction tuning. These results suggest that modality alignment is better understood as structured anisotropic geometric correction, and that large-scale text-only data can be leveraged as a useful substitute for paired image-text supervision.

### 2 Preliminaries

- Definition 2.1 (Modality Gap.). Let X0 and Y0 denote two distinct modalities, let fx : X0 → Sd−1 and fy : Y0 → Sd−1 be pretrained encoders into a shared normalized representation space, and write X = fx(X0) and Y = fy(Y0). Let σ : Sd−1 → S denote the latent semantic map, where S is an abstract semantic space and σ(z) denotes the semantic label associated with z. If, for semantically corresponding cross-modal representations x ∈ X and y ∈ Y , σ(x) = σ(y) while x and y need not coincide geometrically, and this

discrepancy is systematic at the distribution level, µx ̸= µy or Σx ̸= Σy, where µ and Σ denote the mean and covariance, respectively, then such a systematic cross-modal geometric discrepancy is called the Modality Gap phenomenon.

- Definition 2.2 (Modality Align.). In a shared representation space exhibiting modality gap, let Y be the source modality and X the target modality. Modality Align seeks a mapping T : Rd → Rd that rectifies the cross-modal geometric discrepancy such that, given only unpaired samples from X and Y , for any y ∈ Y ,

σ(T(y)) = σ(y) and PT(Y )|σ ≈ PX|σ. The transformed representation T(y) is called a substitute representation of y in the target modality.

- 3 Modality Gap

Two modalities in the shared embedding space often remain separated by a persistent modality gap. This raises a basic geometric question: What kind of discrepancy is the modality gap?

###### 3.1 Geometric Compatibility Across Modalities

We first ask whether the two modalities have compatible global geometry in the shared representation space. This question is essential: if two embeddings were merely two arbitrary and unrelated distributions, then any geometric correction would not preserve semantic consistency. To test this, we compare the dominant covariance structure of the two modalities.

###### (a)

###### (b)

1.0

Amplification

Random q/d

10−3

Observed Oq

Cλ =0.845

10−5

0.5

Oq

λj

j−1 ref.

Image X

10−7

Text Y

0.0

1 4 16 64 256512

100 101 102 103

Subspace Size q

Eigenvalue Rank j

Compatible Spectral Decay. Given 1M paired image-text representations {(xi,yi)}ni=1, where xi ∈ X, yi ∈ Y . Let Σx and Σy denote the centered covariance matrices of the image and text modalities. We compare their covariance spectra by sorting the eigenvalues in descending order and defining the

Figure 1 Image and text modalities share compatible dominant geometry. (a) The normalized covariance spectra of the two modalities exhibit similar long-tail decay, with spectral correlation Cλ = 0.845. (b) Their principal subspace overlap is consistently above the random baseline q/d; at q = 128, O128 = 0.441 versus q/d = 0.100, indicating shared non-random dominant directions.

spectral correlation as Cλ = corr(log λ(Σx),log λ(Σy)). As shown in Fig. 1(a), the normalized spectra of the two modalities exhibit similar long-tail decay. The spectral correlation reaches Cλ = 0.845, indicating that image and text representations distribute their variance energy across dominant directions compatibly.

Shared Principal Structure. Spectral similarity alone does not guarantee that the two modalities use the same directions. We therefore next ask whether their principal subspaces overlap. Let Uxq and Uyq denote the subspaces spanned by the top q eigenvectors of Σx and Σy, respectively. We define the subspace overlap as Oq = 1q (Uxq)⊤Uyq 2F. If the two subspaces were randomly unrelated, the expected overlap would be

approximately q/d. However, Fig. 1(b) shows that the observed Oq is consistently above this random baseline across different subspace sizes. In particular, when q = 128, we obtain O128 = 0.441, whereas the random baseline is only q/d = 0.100. Thus, image and text representations share a set of non-random dominant geometric directions.

Conclusion 1. (Compatible Dominant Geometry). The modality gap does not mean that image and text representations have unrelated global geometry. Instead, the two modalities already share compatible dominant semantic structure in the shared representation space.

#### (a)

#### (b)

#### (c)

1.0

Residual spectrum

Cumulative

|Bias<br><br>| |
|---|
<br><br>Residual|
|---|

CumulativeEnergy

10−2

Isotropic 1/d

Isotropic K/d

1.5

Discrepancy

Amplification

λ/tr(Σ)jr

10−3

Ar =28.6 deff/d=0.284

1.0

0.5

Rres =0.89

‖μx −μy‖2 =0.392 Dcenter =1.264 GΣ =0.066

0.5

10−4

0.0

0.0

100 101 102 103

100 101 102 103

Before After

Top-K Components

Eigenvalue Rank j

Figure 2 The modality gap is dominated by an anisotropic residual. (a) Mean correction removes only a small fraction of the cross-modal discrepancy, leaving a large residual gap. (b) The residual covariance spectrum deviates strongly from the isotropic baseline, with dominant eigen-directions. (c) Residual energy is concentrated in a low-effective-dimensional subspace, with anisotropy ratio Ar = 28.6 and deff/d = 0.284.

###### 3.2 Anisotropic Modality Gap

Having established that the two modalities share compatible dominant geometry, we next ask what form the remaining modality gap takes. A natural hypothesis is that the gap is mainly a global centroid bias. Let (µx,Σx) and (µy,Σy) denote the empirical means and centered covariances of the two modalities, respectively. We measure centroid displacement and covariance-shape discrepancy as Gµ = ∥µx − µy∥2 and GΣ = ∥Σx − Σy∥F/(∥Σx∥F + ϵ).

Centroid Bias Is Insufficient. If the modality gap were dominated by a global mean shift, then translating text representations to the image centroid should remove most of the cross-modal discrepancy. To test this hypothesis, we keep image representations fixed and apply mean correction to text representations as yix = yi − µy + µx. The paired residual after mean correction is ri = xi − yix = (xi − µx) − (yi − µy), with residual covariance Σr = n1 ni=1 riri⊤. Fig. 2(a) confirms that the two modalities have a clear centroid displacement, with Gµ = 0.392. However, the covariance-shape discrepancy is also nonzero, with GΣ = 0.066, suggesting that the misalignment is not purely a difference in mean centers. Although text representations are globally shifted to the image centroid, the corrected paired distance remains high, D = 1.264. The residual ratio is D/D = 0.89. This rules out the simplest explanation that the modality gap is mainly a centroid bias. Anisotropic Residual. We next ask whether the remaining residual is isotropic noise. If this were the case,

- then its covariance would satisfy Σr ≈ σ2I, and its normalized eigenvalue spectrum would be close to the flat isotropic baseline 1/d. However, Fig. 2(b) shows a different pattern. The residual spectrum has dominant eigen-directions whose energy is far above the isotropic average, followed by a long-tail decay. To quantify this deviation, we define the residual anisotropy ratio as Ar = λmax(Σr)/(tr(Σr)/d), where λmax(Σr) is the largest eigenvalue of the residual covariance. Fig. 2(c) shows Ar = 28.6 ≫ 1. Therefore, the residual gap is not random isotropic noise; it is strongly direction-dependent. This anisotropy is further reflected in residual energy concentration. We compute the cumulative energy explained by the top-K residual eigen-directions,

E(K) = Kj=1 λj(Σr)/ dj=1 λj(Σr). As shown in Fig. 2(c), the empirical curve lies far above the isotropic baseline K/d, indicating that residual energy is concentrated in a small number of dominant directions. We

further compute the effective dimension deff(Σr) = tr(Σr)2/tr(Σ2r), obtaining deff/d = 0.284, which confirms that the residual gap lies in a low-effective-dimensional anisotropic subspace.

Conclusion 2 (Anisotropic Residual Gap). The modality gap is dominated by a structured residual: a direction-dependent anisotropic discrepancy concentrated in a low-effective-dimensional subspace.

###### 3.3 Anisotropic Modality Alignment Principle

###### The previous diagnostics reveal two facts. First, image and text representations already share compatible dominant semantic geometry. Second, the remaining modality gap is a low-effective-dimensional anisotropic residual. We therefore ask: What should effective modality alignment preserve, and what should it correct?

###### (a)

###### (b)

###### (c)

10−1

1.00

Δμ ΔΣ

RelativeDiscrepancy

1.00

0.75

Eigenvalue

10−3

0.75

Φ

0.50

Identity

Identity Centroid Moment Permuted

0.50

Centroid Moment

| |
|---|

10−5

0.25

0.25

Oracle α=1 Isotropic 1/d

Oracle path

0.00

0.00

.75 1

Id. Tμ TΣ Per..25

.50

0.00 0.25 0.50 0.75 1.00

100 101 102 103

Target Mixing MkQ

Residual Eigenvalue Rank j

Figure 3 Effective alignment requires both source semantic preservation and target distribution compatibility.

(a) Different transformations exhibit a trade-off between source instance consistency and target local mixing. (b) Centroid and moment corrections reduce global discrepancies, while random target replacement destroys semantic correspondence. (c) Correction along the anisotropic residual subspace reduces dominant residual directions while better preserving source-side semantics.

To answer this question, we compare five diagnostic transformations: ❶ Identity Mapping Tid: the unaligned state; ❷ Centroid Correction Tµ: only removes the global centroid shift; ❸ Moment Correction TΣ: matches global moment statistics; ❹ Random Target Replacement Tperm: serves as a negative control that matches the target distribution but destroys semantic correspondence; and ❺ Tα: provides a controlled interpolation between semantic preservation and target-distribution compatibility by correcting representations along dominant residual directions. The experimental results show that different transformations exhibit clearly different alignment behaviors. As shown in Fig. 3(a), Tµ preserves source-side semantics well, but provides limited improvement in target-side local mixing; Tperm, although drawn from the target distribution, almost completely destroys source semantics, indicating that matching the target distribution alone is insufficient. Fig. 3(b) further shows that TΣ reduces global statistical discrepancy, but introduces noticeable source-side semantic degradation. In contrast, Tα forms a continuous trade-off between source-side semantic preservation and target-side geometric compatibility. Finally, Fig. 3(c) shows that correcting along dominant anisotropic residual directions more directly suppresses the dominant residual components. Therefore, effective alignment should not be viewed as minimizing a single global gap; instead, it should both preserve the semantic geometry of the source modality and correct the dominant anisotropic residuals that prevent compatibility with the target distribution. We provide theoretical support for the geometric diagnostics and the anisotropic alignment principle in Appendix. A. The above diagnostics naturally lead to the following principle:

Principle (Anisotropic Modality Alignment). Effective modality alignment should preserve the source modality’s semantic geometry while correcting the dominant anisotropic residual directions that prevent compatibility with the target-modality distribution.

### 4 AnisoAlign

###### 4.1 Fixed-Frame Subspace Decomposition

- Following Sec. 3.1, we first fix a shared dominant subspace to provide a stable geometric frame for alignment, and identify a shared dominant subspace capturing the major geometric structure of both modalities. Let

µt,µi ∈ Rd denote the empirical means of text embeddings and image embeddings, respectively, and let Σt,Σi ∈ Rd×d denote the corresponding centered covariance matrices. We define the joint structure matrix as Σ = Σt + Σi + λI, where λ > 0 is a regularization parameter and I is the identity matrix. Let QU ∈ Rd×r

consist of the top-r eigenvectors of Σ. Then, Rd can be decomposed into two mutually orthogonal subspaces: Rd = U ⊕ V , with U = span(QU). Under this decomposition, any embedding z ∈ Rd can be uniquely written as:

zU = QUQ⊤Uz, zV = z − zU. (1)

Here, zU denotes the orthogonal projection of z onto the subspace U, capturing its component along the first r dominant statistical directions; zV denotes the remaining component orthogonal to U. All subsequent alignment operations are performed under this fixed decomposition.

Constantρk

bk

###### 4.2 Anisotropic Circular Decoupling

ck = (ak, bk)

- Following Sec. 3.2, we then use blockwise polar coordinates to explicitly model the anisotropic residual structure. We introduce an explicit blockwise polar parameterization protocol within the dominant subspace U. As shown in Fig. 4. We

bk

QU

ρk

QUR

θk

first map the projection Q⊤Uz ∈ Rr into m = r/2 discrete twodimensional subspaces. However, natively constructing these subspaces directly based on the principal component hierarchy introduces an arbitrary dependence on specific eigenvector orderings, making the decomposition sensitive to arbitrary eigenvector orderings. To inoculate the architecture against this basis dependence, we introduce a continuous orthogonal mixing matrix R ∈ Rr×r, subject to the strict constraint

ak

ak

Anisotropic Subspace U

Figure 4 Anisotropic circular decoupling in U subspace.

R⊤R = I. We dynamically redefine the internal coordinate basis as QU ← QUR. This mixing operation preserves the invariant span of the subspace U while autonomously discovering a maximally stable internal coordinate organization for downstream anisotropic decoupling. Based on this optimized coordinate system, let (ak,bk) denote the coordinates of the projected vector c = Q⊤Uz ∈ Rr within the k-th two-dimensional block. We reformulate these Euclidean coordinates into a polar embedding:

ρk = a2k + b2k + ε, θk = atan2(bk,ak) (2)

where ε > 0 ensures numerical stability near the origin. The embedding in U is thus decoupled into blockwise radii ρ = (ρ1,...,ρm) and phases θ = (θ1,...,θm).

###### 4.3 Stage I: Target-Modality Periodic Prior Pretraining

Before learning any modality alignment, we first estimate the phase statistical structure of the target modality in the decoupled phase space using only the image. As shown in Fig. 5. This structure consists of two aspects: first, the marginal distributions of the phase variables of individual two-dimensional blocks; second, the dependency relations among phase differences across different two-dimensional blocks. Stage I does not involve learning a text-to-image mapping. Instead, it constructs a frozen periodic score prior sϕ from the image modality, which is subsequently used in Stage II as a targetmodal constraint.

For an image embedding x, let {(ρ(kx),θk(x))}mk=1 denote its polar embedding. We define the blockwise circular correlation

statistic as:

k −θℓ(x)) ∈ C. (3)

(x)

Mkℓ(x) = E ei(θ

−τ∇ϕΨ

Akℓ, ηkℓ

ϕk

ϕℓ

ϕ, σ˜ t

µϕ

ψ¯k, αk

Figure 5 Target-modality periodic prior in phase space. Image phases define marginal anchors (ψ¯k, αk) and pairwise couplings (Akℓ, ηkℓ), which induce a drift field −τ∇ϕΨ and train the frozen phase score prior sϕ.

Here, |Mkℓ(x)| measures the consistency of the phase difference between the k-th and ℓ-th blocks, while arg(Mkℓ(x)) gives the corresponding empirical phase offset. Instead of selecting globally top-p block pairs over all possible

pairs, we construct the sparse dependency graph in a block-adaptive manner: for each block k, we retain the top-p blocks ℓ ≠ k with the largest |Mkℓ(x)|, and then take the union of all retained undirected pairs. This yields a sparse dependency graph E ⊆ [m] × [m], where [m] := {1,...,m}.

Based on these quantities, we define a drift field in phase space, ∇ϕΨ(ϕ) ∈ Rm, where ϕ = (ϕ1,...,ϕm) ∈ [−π,π)m. Its k-th component is

[∇ϕΨ(ϕ)]k = αk sin(ϕk − ψ¯k) +

Akℓ sin(ϕk − ϕℓ − ηkℓ). (4)

ℓ:(k,ℓ)∈E

Here, Akℓ = |Mkℓ(x)| ∈ R≥0 and ηkℓ = arg(Mkℓ(x)) ∈ [−π,π) denote the coupling strength and empirical phase offset of edge (k,ℓ), respectively; ψ¯k = arg(E[eiθ

(x)

k ]) ∈ [−π,π) denotes the dominant phase location of the k-th two-dimensional block; and αk = E[(ρ(kx))2]/( mu=1 E[(ρ(ux))2] + ε) ∈ R≥0 denotes the relative weight of that block. Given a phase vector ϕ, we first define the drifted phase center µϕ ∈ [−π,π)m as:

µϕ = wrap ϕ − τ∇ϕΨ(ϕ) . (5)

We then construct a perturbed phase sample ϕ˜ ∈ [−π,π)m as ϕ˜ = wrap(µϕ + √2σtϵ), where ϵ ∼ N(0,Im), τ > 0 is the drift step size, and σt > 0 is the noise scale at time step t.

On this basis, we train a phase-aware score network sϕ : Rm × R × Rm → Rm, whose input is (ϕ,t,˜ log ρ) and whose output is the phase score sϕ(ϕ,t,˜ log ρ) ∈ Rm. The Stage-I loss is defined as:

2 2

LI = Et,ϕ˜ λt sϕ(ϕ,t,˜ log ρ) − ∇ϕ˜ log q(ϕ˜ | µϕ,σt)

, (6)

where q(ϕ˜ | µϕ,σt) denotes a wrapped Gaussian distribution centered at µϕ with noise scale σt, ∇ϕ˜ log q(ϕ˜ | µϕ,σt) ∈ Rm is its score with respect to ϕ˜, and λt = 2σt2.

Therefore, Stage I yields a phase score prior determined by the target image distribution. This prior is kept frozen after training and is introduced in Stage II as a target-modal constraint.

###### 4.4 Stage II: Prior-Guided Bounded Alignment

After fixing the periodic prior sϕ of the target modality, Stage II performs a two-stage update on the text embedding y ∈ Rd: a deterministic global initialization followed by an instance-conditioned bounded refinement.

###### 4.4.1 Global Initialization

We first recenter the text embedding by y¯ = y − µt + µi ∈ Rd. On U-side. We project y¯ onto the mixed basis and express it in blockwise polar coordinates (ρ,θ). We set θ(0) = θ ∈ [−π,π)m and define ρ(0)k = Tk(ρk), where Tk(r) = Fk(x) −1 Fk(y)(r) . Here, Fk(x) and Fk(y) denote the empirical radial cumulative distribution functions of images and text, respectively, on the k-th two-dimensional block. This gives ρ(0) = (ρ(0)1 ,...,ρ(0)m ) ∈ Rm>0. On V -side. We define yU = QUQ⊤Uy and yV = y − yU, and set v(0) = µi,V + DV yV − µt,V ∈ Rd, where DV = Diag σV(x)/(σV(y) + ε) , µi,V = PV µi, and µt,V = PV µt. This yields the initialized state (θ(0),ρ(0),v(0)).

###### 4.4.2 Prior-Guided Residual Refinement

Starting from the initialized state, we use an instance-conditioned map gη to predict residual corrections for phase, radius, and the V -subspace component:

(∆θ,∆ρ,∆v) = gη [sinθ(0);cosθ(0);log ρ(0);v(0)] , (7)

where ∆θ,∆ρ ∈ Rm and ∆v ∈ Rd. Since the refinement of the residual component is restricted to the orthogonal complement V , we remove its U-projection and keep only the V -part, i.e., ∆vV = ∆v − QUQ⊤U∆v. Rather than directly denoising toward the target modality, we constrain the refined phase configuration to remain locally compatible with the target prior. The refined phase, radius, and residual component are then given by θˆ = wrap θ(0) + αθ tanh(∆θ) , ρˆk = ρ(0)k exp αρ tanh(∆ρk) , and vˆ = v(0) + αv tanh(∆vV ). so that θˆ ∈ [−π,π)m, ρˆ = (ˆρ1,...,ρˆm) ∈ Rm>0, and vˆ ∈ Rd.

To impose the target-modality prior, instead of using a one-step denoising guidance objective, we construct a prior-matching loss around the refined phase itself. Specifically, we first define µθˆ = wrap θ ˆ− τ∇ϕΨ(θˆ) and then perturb it as θ˜ = wrap µθˆ + √2σtϵ , where ϵ ∼ N(0,Im). We define the prior-matching loss as

LII = Et,ϵ λt sϕ(θ,t,˜ log ρˆ) − ∇θ˜log q(θ˜ | µθˆ,σt)

2 2

. (8)

This objective encourages the refined phase configuration to remain locally compatible with the frozen target-modality periodic prior.

In parallel, reusing the sparse graph E from Stage I, we define ωkℓ(0) = ρ(0)k ρ(0)ℓ (u,v)∈E ρ(0)u ρ(0)v + ε for any (k,ℓ) ∈ E, and impose the relative phase deformation constraint

1 |E|

ωkℓ(0) 1 − cos (θˆk − θˆℓ) − (θk(0) − θℓ(0)) . (9)

LΦ =

(k,ℓ)∈E

Finally, let c(ˆρ,θˆ) ∈ Rr denote the blockwise Cartesian vector generated from (ˆρ,θˆ), whose k-th twodimensional block is (ˆρk cosθˆk,ρˆk sinθˆk). We first reconstruct an intermediate normalized embedding as e′ = Norm QU c(ˆρ,θˆ) + vˆ ∈ Sd−1. After transforming the full text corpus, we further estimate the global mean of the intermediate transformed representations, µˆ = Ey[e′(y)], and perform a final global centroid calibration by defining e = Norm e′ − µˆ + µi ∈ Sd−1. The calibrated representation e is used as the final substitute representation in the target modality.

### 5 Experiments

In this section, we systematically evaluate the effectiveness of our method from two perspectives: representationlevel geometric diagnostics and MLLM training. The experiments are designed to answer six core questions, Q1–Q6. At the representation level, we use images as the target modality and texts as the source modality. We randomly sample 10K paired image-text representation samples for geometric diagnostics. At the MLLM level, we keep the model architecture, decoding settings, training data, and evaluation protocol unchanged. We use LLM2CLIP-Openai-L-14-336 [7] as the encoder and Llama-3-8B-Instruct as the LLM backbone. We compare four methods: Text, C3 [21], ReAlign [17], and AnisoAlign. Detailed experiment settings are provided in Appendix. B.

❶ AnisoAlign Better Match the Target-Modality Geometry? This experiment examines whether the transformed source representations Z = T(Y ) enter the geometric support of the target modality. We first measure the centroid discrepancy ∆µ(T) = ∥µz − µx∥2. As shown in Fig. 6(a), Text shows a global offset with ∆µ = 0.393, and C3 reduces it to 0.276. In contrast, ReAlign and AnisoAlign both reduce it to about 0.012, indicating effective target-centroid calibration. We evaluate local support compatibility. As shown in Fig. 6(b), C3 obtains MkZ = 0.410 but only MkX = 0.075, suggesting sparse target penetration without sufficient target coverage. ReAlign gives more balanced scores, MkZ = 0.357 and MkX = 0.305, while AnisoAlign improves them to MkZ = 0.372 and MkX = 0.337, achieving the best balance between penetration and coverage. The residual spectra in Fig. 6(c) also show that Text and C3 retain clear anisotropic residual structures, whereas ReAlign and AnisoAlign reduce dominant residual directions. AnisoAlign achieves near-zero centroid discrepancy, the most balanced local support matching, and a much weaker structured anisotropic residual.

##### (a)

##### (b)

##### (c)

0.5

10−1

| | | | |
|---|---|---|---|
| | | | |
| | | | |
|Text C3 ReAlign| | | |
|AnisoAli Iso. 1/d|gn| | |
| | | | |

MkZ MkX

Norm.ResidualEig.

0.4

0.4

Norm.Score

10−2

0.3

0.3

Δμ

10−3

0.2

0.2

0.1

10−4

0.1

0.0

0.0

Text C3 ReA. Aniso

Text C3 ReA. Aniso

100 101 102 103

Rank j

###### Figure 6 Target-geometry compatibility of different alignment methods. AnisoAlign achieves near-zero centroid discrepancy, the most balanced local support matching, and a low-anisotropy residual spectrum, outperforming Text and C3 while remaining competitive with ReAlign.

1.00

C3 ReAlign AnisoAlign

❷ Does AnisoAlign Preserve Source-Modality Semantics During Modality Alignment? This experiment evaluates whether modality alignment can preserve the semantic organization of the source modality while performing geometric correction. As shown in Fig. 7, C3 achieves approximately 0.899, 0.925, and 0.840 on Φ, Ψ, and Ωk, respectively, indicating that Gaussian perturbation can preserve certain global pairwise similarity relations, but introduces noticeable disruption to local neighborhood structures. ReAlign performs well in instance-level consistency, with Φ ≈ 0.923, but its relative geometry consistency drops to Ψ ≈ 0.836, suggesting that pointwise closeness alone does not guarantee the stability of semantic relations within the source modality. In contrast, AnisoAlign achieves the best performance on all three metrics, with Φ ≈ 0.941, Ψ ≈ 0.983, and Ωk ≈ 0.945. This shows that AnisoAlign not only preserves instance-level consistency between transformed representations and original text representations, but also more stably maintains the global semantic relations and local neighborhood structure of the source modality.

| |
|---|

0.95

| |
|---|

0.90

Score

0.85

0.80

0.75

Φ Ψ Ωk

Figure 7 Source-modality semantic preservation of different alignment methods. AnisoAlign achieves the best performance across instance consistency, relative geometry consistency, and neighborhood consistency.

❸ Can AnisoAlign Improve Fully Text-Only MLLM Training? We next ask whether AnisoAlign can provide an effective visual representation interface for MLLMs without using any image-text pairs throughout training. In this setting, the model cannot learn from real image features and must rely only on substitute visual representations obtained from aligned text representations. All methods use the same protocol: pretraining on Unicorn-1.2M [16] followed by instruction-tuning on Unicorn-Instruction-417K [16], with identical data, architecture, and training procedure. As shown in Table 1, AnisoAlign achieves the highest average score, 47.49, outperforming ReAlign (45.00), C3 Align (42.44), Unicorn (42.57), and W/o. Align (40.08). This shows that fully text-only training depends not only on the amount of text data, but also on whether text representations can enter the visual representation space in the correct geometric form. W/o. Align leaves substitute representations near the text distribution; C3 Align and ReAlign alleviate this issue through statistical correction or global distribution matching. In contrast, AnisoAlign jointly models target-modality distribution constraints and source-modality semantic preservation, producing substitute representations better suited as a visual interface.

❹ Can AnisoAlign serve as a stronger text-only pretraining interface before visual instruction tuning? We examine whether AnisoAlign can serve as a stronger text-only pretraining interface before visual instruction tuning. This setting asks whether large-scale text-only data can first be used to construct substitute visual representations during pretraining, followed by post-training with real vision-language instruction data. We

###### Table 1 Results on fully text-only MLLM training setting.

General Reasoning Hallucination

Method

Avg. ↑ MME MMStar SQA RWQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

Blind 3.37 8.80 6.17 5.36 19.60 12.44 0.30 1.56 12.90 0.60 15.25 7.85 W/o. Align 46.17 30.67 58.51 37.78 30.69 29.59 25.60 24.38 65.23 55.28 37.01 40.08 Unicorn 60.24 29.27 66.12 37.65 30.46 30.73 25.50 24.16 65.76 55.31 43.01 42.57 C3 Align 62.56 31.40 63.30 36.47 32.67 30.34 26.00 23.27 59.07 54.17 47.63 42.44 ReAlign 67.48 32.80 65.68 40.78 33.61 31.85 26.20 25.95 67.66 56.91 46.06 45.00 AnisoAlign 72.96 34.47 70.81 42.09 37.34 34.05 27.90 27.29 66.36 57.62 51.52 47.49

###### Table 2 Results on text-only pretraining setting.

General Reasoning Hallucination

Method

Avg. ↑ MME MMStar SQA RWQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

Blind 3.37 8.80 6.17 5.36 19.60 12.44 0.30 1.56 12.90 0.60 15.25 7.85 W/o. Align 73.63 35.73 75.23 43.53 28.82 25.38 24.40 21.03 80.82 71.59 42.38 47.50 C3 Align 76.16 34.60 75.52 43.14 30.69 27.20 25.50 19.91 79.99 72.43 43.53 48.06 ReAlign 79.65 36.13 76.71 47.97 31.51 28.39 27.70 22.82 81.78 72.53 46.58 50.16 AnisoAlign 81.22 36.73 76.27 44.58 37.34 32.85 28.10 25.95 82.93 73.65 47.84 51.59

use 1M text samples from Bunny-pretrain [6] for text-only pretraining and InternVL-Chat-V1.2-SFT [3] for visual instruction tuning. As shown in Table 2, AnisoAlign achieves the highest average score of 51.59, outperforming ReAlign (50.16), C3 Align (48.06), and W/o. Align (47.50). These results show that AnisoAlign not only provides substitute visual representations in fully text-only training, but also serves as a better pretraining interface before visual instruction tuning. Compared with ReAlign, AnisoAlign improves by 1.43 points, suggesting that global distribution matching alone is insufficient to fully exploit text-only pretraining signals. Its gains over C3 Align and W/o. Align, 3.53 and 4.09 points respectively, further indicate that coarse perturbation or no explicit alignment cannot construct a stable visual substitute interface.

❺ Can AnisoAlign surpass paired image-text pretraining by scaling up text-only data? We examine a further question: if the scale of text-only data continues to increase, can AnisoAlign approach or even surpass pretraining with real image-text pairs? This experiment is designed to verify the scalability of AnisoAlign. If the quality of substitute visual representations is sufficiently high, then text-only data can not only serve as a supplement to real image data, but may also become a more economical and more scalable pretraining resource in large-scale scenarios. We compare three settings: (1) W/. Image, which uses real images; (2) AnisoAlign-1M, which uses 1M text-only samples; and (3) AnisoAlign-2M, which uses 2M text-only samples. All methods then follow the same downstream training and evaluation pipeline. Table 3 shows that AnisoAlign-1M already reaches an average score of 51.60, close to W/. Image at 52.72. When the text-only data scale is increased to 2M, AnisoAlign-2M further improves to 52.75, slightly surpassing W/. Image at 52.72 and improving over AnisoAlign-1M by 1.15 average points. This indicates that real images are not the only scalable source of visual supervision for MLLM pretraining. AnisoAlign provides a scalable training paradigm: through high-quality anisotropic modality alignment, large-scale text data can be transformed into effective visual-style training signals, and can partially reach or even surpass the performance of real image-text pretraining.

❻ Are All Components Necessary for AnisoAlign? We conduct ablation studies to analyze the contribution of each component in AnisoAlign. As shown in Table. 4. Using only global initialization achieves an average score of 43.59, showing that coarse centroid and distribution calibration already provide a reasonable substitute representation, but remain insufficient for high-quality modality alignment. Adding instanceconditioned refinement improves the average score to 44.93, indicating that bounded sample-specific correction is necessary beyond global statistics. Introducing the target-modality prior LG further raises the score to

###### Table 3 Scaling text-only data with AnisoAlign enables substitute visual representations to approach and slightly surpass real image-based pretraining.

General Reasoning Hallucination

Method

Avg. ↑ MME MMStar SQA RWQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

W/. Image 82.86 37.07 77.67 45.62 38.27 33.04 29.40 27.08 82.73 74.16 48.06 52.72

- AnisoAlign-1M 81.22 36.73 76.27 44.58 37.34 32.85 28.10 25.95 82.93 73.65 47.84 51.60

- AnisoAlign-2M 83.15 37.47 78.60 45.79 38.92 33.86 29.20 27.64 82.17 75.39 49.63 52.75

###### Table 4 Ablation results show that global initialization, bounded refinement, target-prior guidance, and phase-structure preservation each contribute to the final performance of AnisoAlign.

General Reasoning Hallucination

Method

Avg. ↑ MME MMStar SQA RWQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

Global Initialization Only 64.32 32.10 64.85 39.66 33.02 31.41 25.90 25.48 62.19 53.36 47.21 43.59 Global Initialization + Refinement 67.85 32.63 66.94 40.57 34.18 32.16 26.41 25.97 62.65 55.71 49.18 44.93 Global Initialization + Refinement + LII 71.24 33.86 68.47 41.91 35.63 33.41 27.03 26.52 64.94 56.28 52.83 46.56 Global Initialization + Refinement + LΦ 70.38 33.41 69.22 41.67 35.97 32.89 27.46 26.91 65.32 55.97 51.74 46.45

AnisoAlign 72.96 34.47 70.81 42.09 37.34 34.05 27.90 27.29 66.36 57.62 51.52 47.49

46.56, demonstrating that target-side geometric guidance helps the substitute representations better match the visual distribution. Similarly, adding the relative phase constraint LΦ achieves 46.45, confirming the importance of preserving structured phase relations during refinement. The full AnisoAlign model obtains the best average score of 47.49, outperforming all ablated variants and achieving consistent gains across general perception, reasoning, and hallucination-related benchmarks. These results show that global initialization, bounded refinement, target-prior guidance, and phase-structure preservation are complementary and jointly contribute to more effective anisotropic modality alignment.

### 6 Conclusion

This paper revisits the modality gap from a geometric perspective and shows that it is a structured anisotropic residual built upon compatible semantic geometry. Based on this observation, we propose the principle of anisotropic modality alignment. We further propose an unpaired modality alignment method for generating target-modality substitute representations. Experiments show that our method can help eliminate reliance on paired image-text data. Overall, modality alignment should be better understood as a structured geometric correction.

### References

- [1] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024.
- [2] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.
- [3] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks, 2024. URL https://arxiv.org/abs/2312.14238.
- [4] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.
- [5] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14375–14385, 2024.
- [6] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective, 2024. URL https://arxiv.org/abs/2402.11530.
- [7] Weiquan Huang, Aoqi Wu, Yifan Yang, Xufang Luo, Yuqing Yang, Usman Naseem, Chunyu Wang, Chunyu Wang, Qi Dai, Xiyang Dai, Dongdong Chen, Chong Luo, Lili Qiu, and Liang Hu. Llm2clip: Powerful language model unlocks richer cross-modality representation, 2026. URL https://arxiv.org/abs/2411.04997.
- [8] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292–305, 2023.
- [9] Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. Advances in Neural Information Processing Systems, 35:17612–17625, 2022.
- [10] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [11] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35:2507–2521, 2022.
- [12] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [13] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. arXiv preprint arXiv:2308.01907, 2023.
- [14] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.
- [15] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025.
- [16] Xiaomin Yu, Pengxiang Ding, Wenjie Zhang, Siteng Huang, Songyang Gao, Chengwei Qin, Kejian Wu, Zhaoxin Fan, Ziyue Qiao, and Donglin Wang. Unicorn: Text-only data synthesis for vision language model training, 2025. URL https://arxiv.org/abs/2503.22655.
- [17] Xiaomin Yu, Yi Xin, Wenjie Zhang, Chonghan Liu, Hanzhen Zhao, Xiaoxing Hu, Xinlei Yu, Ziyue Qiao, Hao Tang, Xue Yang, Xiaobin Hu, Chengwei Qin, Hui Xiong, Yu Qiao, and Shuicheng Yan. Modality gap-driven

- subspace alignment training paradigm for multimodal large language models, 2026. URL https://arxiv.org/ abs/2602.07026.
- [18] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.
- [19] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134–15186, 2025.
- [20] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [21] Yuhui Zhang, Elaine Sui, and Serena Yeung-Levy. Connect, collapse, corrupt: Learning cross-modal tasks with uni-modal data. arXiv preprint arXiv:2401.08567, 2024.

### A Theoretical Derivation of the Anisotropic Modality Gap

###### A.1 Overview and Notation

This appendix provides theoretical support for the geometric diagnostics in Sec. 3 and the methodological design in Sec. 4. The objective is not to prove the global optimality of the proposed alignment algorithm, but to formalize the following four points: first, why the modality gap should be decomposed into a global centroid displacement and a centered residual component; second, why the centered residual should be compared against an isotropic null hypothesis; third, why the dominant residual directions constitute efficient correction targets for reducing the residual gap; and fourth, why effective correction must be constrained in order to simultaneously preserve the semantic geometry of the source modality and improve compatibility with the target-modality distribution.

Let X denote the target image modality and Y denote the source text modality. Let x ∈ X and y ∈ Y be a paired image-text representation in the shared embedding space Rd. Paired samples are used only for the geometric diagnostics in Sec. 3; the alignment method proposed in Sec. 4 does not rely on paired supervision. We denote the modality means by

µx := E[x], µy := E[y], and define the centered variables as

x¯ := x − µx, y¯ := y − µy. The centered covariance matrices of the two modalities are

Σx := E[¯xx¯⊤], Σy := E[¯yy¯⊤]. The centered cross-modal second-order moments are denoted by

Σxy := E[¯xy¯⊤], Σyx := Σ⊤xy.

For any symmetric matrix M, let λj(M) denote its j-th largest eigenvalue in descending order, let UMq denote the matrix formed by its top q eigenvectors, and let PMq := UMq (UMq )⊤ denote the corresponding orthogonal projector.

###### A.2 Formalizing Dominant Geometric Compatibility

Before studying how to correct the modality gap, we first ask whether the two modalities possess compatible global geometry in the shared representation space. If image and text representations were two arbitrary and unrelated distributions, then any geometric correction would be unlikely to simultaneously achieve distributional alignment and semantic preservation.

###### A.2.1 Spectral Compatibility

Let the eigenvalues of Σx and Σy be sorted in descending order. We measure whether the two modalities allocate variance energy similarly across dominant and tail directions using the log-spectral correlation

Cλ := corr(log λ(Σx),log λ(Σy)).

- A high value of Cλ indicates that the two modalities exhibit similar hierarchical variance-energy profiles. In other words, if one modality allocates substantial variance to certain dominant directions, the other modality tends to allocate substantial variance to directions at similar spectral ranks.

However, spectral similarity alone does not imply that the two modalities use the same directions. Two covariance matrices may have similar eigenvalue spectra while having nearly orthogonal eigenspaces. Therefore, we further compare their principal subspaces.

###### A.2.2 Principal Subspace Overlap and the Random Baseline

Let Uxq and Uyq denote the top-q eigenvector matrices of Σx and Σy, respectively. We define the principal subspace overlap as

1 q

1 q

(Uxq)⊤Uyq 2F =

tr(PxqPyq),

Oq :=

where Pxq = Uxq(Uxq)⊤ and Pyq = Uyq(Uyq)⊤. This quantity lies in [0,1] and measures the degree of overlap between the two q-dimensional principal subspaces.

Lemma A.1 (Random-subspace baseline). Suppose Uyq is sampled uniformly from the Grassmann manifold Gr(q,d) with respect to the Haar measure and is independent of Uxq. Then

E[Pyq] =

q d

q d

Id, E[Oq] =

.

Proof. By the invariance of the Haar measure under the action of the orthogonal group, E[Pyq] must commute with every orthogonal matrix. Hence it must be a scalar multiple of the identity matrix. Since tr(Pyq) = q, we obtain E[Pyq] = (q/d)Id. Therefore,

E[Oq] =

1 q

tr PxqE[Pyq] =

1 q

tr Pxq

q d

q d

Id =

.

| |
|---|

Thus, when the empirical overlap satisfies Oq ≫ q/d, the two modalities do not use randomly unrelated dominant directions. Instead, they share a non-random set of dominant geometric directions.

###### A.2.3 Implication

Spectral compatibility and principal subspace overlap together indicate that the modality gap is not an arbitrary discrepancy between two unrelated distributions. Rather, it is a structured discrepancy built upon a partially shared dominant semantic-geometric backbone. This observation provides a necessary premise for alignment: the transformation should not freely distort the source-modality representation, but should preserve its existing semantic geometry while correcting the residual structure that prevents compatibility with the target modality.

###### A.3 Mean--Residual Decomposition

- A.3.1 Decomposition Identity For a paired representation (x,y), we have

x − y = (µx − µy) + (¯x − y¯). Since E[¯x − y¯] = 0, it follows that

E[⟨µx − µy, x¯ − y¯⟩] = 0. Therefore, the expected squared cross-modal discrepancy admits the orthogonal decomposition

E∥x − y∥22 = ∥µx − µy∥22 + E∥x¯ − y¯∥22.

This decomposition shows that the modality gap contains at least two components: the first-order centroid displacement ∥µx−µy∥22 and the centered residual discrepancy E∥x¯−y¯∥22. Therefore, global mean displacement can only explain first-order centroid mismatch, but not the structured discrepancy that remains after centering.

- A.3.2 Residual after Centroid Correction Consider the global centroid correction applied to the text representation,

yx := y − µy + µx. The paired residual after this correction is

r := x − yx = (x − µx) − (y − µy) = x¯ − y.¯ Thus E[r] = 0, and its covariance matrix is

Σr := E[rr⊤]. Expanding this expression gives

Σr = Σx + Σy − Σxy − Σyx. This identity shows that the centered residual is determined not only by the marginal covariance structures of the two modalities, but also by their cross-modal correspondence structure Σxy. The squared residual energy remaining after centroid correction is

E∥r∥22 = tr(Σr).

Therefore, if the modality gap were mainly dominated by a global centroid displacement, then tr(Σr) should become small after centroid correction. Conversely, if the residual remains large, then the modality gap cannot be explained as a simple global mean shift.

- A.3.3 Residual Ratio To compare across datasets or embedding scales, we may define the energy-based residual ratio as

Renergy :=

E∥r∥22 E∥x − y∥22

=

tr(Σr) ∥µx − µy∥22 + tr(Σr)

.

If the main text reports average distances rather than squared energies, the corresponding distance-based ratio can be defined as

Rdist :=

E∥r∥2 E∥x − y∥2

.

Under either definition, a ratio close to 1 indicates that most of the cross-modal discrepancy remains in the centered residual after removing the global centroid displacement. The large residual ratio observed in the main text therefore rejects the simple explanation that the modality gap is primarily a centroid bias.

- A.4 Isotropic Residual Null Hypothesis

After removing the centroid displacement, a natural null hypothesis is that the remaining residual is merely isotropic noise. Under this hypothesis, the residual has equal variance in all directions and does not contain any dominant geometric structure.

- A.4.1 Null Hypothesis We formalize the isotropic residual null hypothesis as

H0 : Σr = σ2Id for some σ > 0. Under H0, all eigenvalues of Σr are equal:

λ1(Σr) = λ2(Σr) = ··· = λd(Σr) = σ2. This hypothesis implies three direct spectral properties.

- A.4.2 Residual Anisotropy Ratio We define the residual anisotropy ratio as

Ar :=

λmax(Σr) tr(Σr)/d

,

where λmax(Σr) is the largest eigenvalue of the residual covariance and tr(Σr)/d is the average eigenvalue. Since the largest eigenvalue is no smaller than the average eigenvalue, Ar ≥ 1. Under the isotropic null hypothesis, all eigenvalues are equal, hence Ar = 1. Therefore, an empirical observation of Ar ≫ 1 indicates that some directions carry residual energy far above the average level, contradicting the isotropic-noise hypothesis.

- A.4.3 Cumulative Spectral Energy We define the cumulative energy explained by the top K residual eigen-directions as

E(K) :=

K j=1 λj(Σr) d j=1 λj(Σr)

.

Under the isotropic null hypothesis,

E(K) =

K d

.

Thus, if the empirical curve satisfies E(K) ≫ K/d for small K, the residual energy is concentrated in a small number of dominant directions.

- A.4.4 Effective Dimension The effective dimension of the residual covariance is defined as

deff(Σr) :=

tr(Σr)2 tr(Σ2r)

= j

λj(Σr)

2

j λj(Σr)2

.

By the Cauchy–Schwarz inequality, 1 ≤ deff(Σr) ≤ d. Under the isotropic null hypothesis, deff(Σr) = d. Hence, an empirical observation of deff(Σr)/d ≪ 1 indicates that the residual distribution has an effective dimension much smaller than the ambient dimension.

- A.4.5 Implication The isotropic residual null hypothesis is rejected by any of the following empirical patterns:

K d

Ar ≫ 1, E(K) ≫

,

deff(Σr) d ≪ 1.

The residual spectrum reported in the main text satisfies these conditions simultaneously. This indicates that the centered residual is not unstructured isotropic noise, but a structured anisotropic residual concentrated along a small number of dominant directions.

###### A.5 Efficiency of Dominant Residual-Direction Correction

The previous subsection shows that the centered residual energy is concentrated along a small number of dominant directions. We now show that if a correction is restricted to act within a K-dimensional subspace, then choosing the top K eigen-directions of the residual covariance is optimal for minimizing the remaining squared residual energy.

- A.5.1 Optimal Projection Result Let the eigendecomposition of the residual covariance be

Σr = UΛU⊤, Λ = diag(λ1,...,λd),

where λ1 ≥ λ2 ≥ ··· ≥ λd ≥ 0. Consider an oracle correction that removes the residual component in some K-dimensional subspace. Let P be the orthogonal projector onto that subspace. The corrected residual is (I − P)r, and the expected remaining residual energy is

J(P) := E∥(I − P)r∥22. Since

J(P) = tr((I − P)Σr) = tr(Σr) − tr(PΣr), minimizing J(P) is equivalent to maximizing tr(PΣr).

- Proposition A.2 (Optimal rank-constrained residual correction). Among all rank-K orthogonal projectors,

the projector onto the subspace spanned by the top K eigenvectors of Σr minimizes the expected remaining residual energy:

PK⋆ = arg min

rank(P)=K

E∥(I − P)r∥22. The minimum value is

min

rank(P)=K

E∥(I − P)r∥22 =

j>K

λj(Σr).

Proof. By the Ky Fan maximum principle,

max

rank(P)=K

tr(PΣr) =

K

j=1

λj(Σr),

and the maximum is attained by the projector onto the top-K eigenspace of Σr. Substituting this into J(P) = tr(Σr) − tr(PΣr) yields

min

rank(P)=K

J(P) = tr(Σr) −

K

j=1

λj(Σr) =

j>K

λj(Σr).

| |
|---|

- A.5.2 Comparison with Random Correction If the rank-K correction subspace is chosen randomly, with projector Prand, then

K d

E[Prand] =

Id. Therefore, the expected residual energy removed by a random subspace is E[tr(PrandΣr)] =

K d

tr(Σr). By contrast, the top K residual eigen-directions remove

K

λj(Σr) = E(K)tr(Σr).

j=1

The ratio between dominant-direction correction and random correction is

K j=1 λj(Σr)

(K/d)tr(Σr)

E(K) K/d

=

.

When E(K) ≫ K/d, correcting the dominant residual directions is substantially more efficient than correcting random directions or applying isotropic perturbations. In particular, when K = 1, this gain is exactly

λmax(Σr) tr(Σr)/d

Ar =

.

Thus, stronger residual anisotropy implies a larger advantage for dominant-direction correction over random correction.

###### A.5.3 From the Residual Principal Subspace to the Joint Covariance Subspace

The oracle analysis above indicates that, if paired residuals were available, the most direct correction target would be the principal subspace of Σr. However, in the unpaired alignment setting, estimating Σr requires cross-modal correspondence through

Σr = Σx + Σy − Σxy − Σyx.

The cross-modal term Σxy cannot be reliably estimated from unpaired samples alone. Therefore, directly using the residual covariance to define the correction subspace is not suitable for unpaired alignment.

The proposed method instead uses the joint marginal covariance ΣJ := Σx + Σy + λI

and takes its top r eigenvectors to define the dominant subspace U. This choice should not be interpreted as a claim that the principal subspace of ΣJ is strictly identical to that of Σr. Rather, ΣJ provides a computable unpaired surrogate based only on marginal statistics.

This surrogate is motivated by two observations. First, the spectral compatibility and principal subspace overlap in Sec. A.2 indicate that image and text representations already share a non-random dominant geometric backbone. Hence, the principal subspace of Σx + Σy captures high-variance geometric directions jointly occupied by both modalities. Second, the residual spectral diagnostics in Sec. A.4 show that the remaining modality gap is not uniformly distributed over all directions, but concentrated in a low-effectivedimensional structure. Therefore, applying structured correction within the shared dominant geometric backbone is more consistent with the residual geometry than either unconstrained full-space mappings or isotropic perturbations.

To empirically verify whether the joint dominant subspace captures residual energy, one may report the residual-energy coverage ratio

tr(PUΣr) tr(Σr)

ηU :=

,

where PU is the orthogonal projector onto the top-r eigenspace of ΣJ. A high value of ηU indicates that the joint dominant subspace captures a large fraction of the residual energy, further supporting its use as an unpaired surrogate correction subspace.

###### A.5.4 Implication

This subsection yields two conclusions. First, in the oracle setting where the residual covariance is available, correcting the dominant residual directions is optimal for reducing squared residual energy under a rank constraint. Second, in the practical unpaired setting, where the residual covariance cannot be directly constructed, the dominant eigenspace of the joint marginal covariance provides a computable surrogate. Its use is motivated by the observed dominant geometric compatibility and can be further validated by the residual-energy coverage ratio ηU.

###### A.6 Non-identifiability of Distribution Matching Alone

The previous subsection clarifies which directions should be corrected. We now explain why matching the target distribution alone is insufficient.

Let PY and PX denote the source- and target-modality representation distributions, respectively. A distributionmatching alignment seeks a map T such that

T#PY = PX,

where T#PY denotes the pushforward distribution of PY under T. However, this condition alone does not identify a semantics-preserving alignment map.

- Proposition A.3 (Non-identifiability of marginal distribution matching). Suppose T0 satisfies (T0)#PY = PX. For any measurable transformation S that preserves the target distribution, i.e., S#PX = PX, the composite map S ◦ T0 also satisfies (S ◦ T0)#PY = PX. Proof. By the composition property of pushforward measures,

(S ◦ T0)#PY = S#((T0)#PY ) = S#PX = PX.

| |
|---|

Therefore, S ◦ T0 is equally valid as T0 under marginal distribution matching. However, different choices of S may arbitrarily permute or distort instance-level semantic correspondence while preserving the same target marginal distribution. Hence, target distribution matching alone cannot distinguish a semantics-preserving alignment from a semantically destructive transformation with the correct marginal distribution.

This explains the role of the random target replacement Tperm in the main text. It can match the targetmodality distribution, but destroys the semantic correspondence between the original source sample and its transformed representation. Effective modality alignment must therefore impose, either explicitly or implicitly, additional constraints that preserve the semantic geometry of the source modality.

###### A.7 Semantic Preservation under Bounded Correction

The previous subsection shows that target distribution matching alone does not guarantee semantic preservation. We now show that bounded correction controls the distortion of source-modality semantic structure.

###### A.7.1 Similarity Preservation under Additive Perturbation

Let

T(y) = y + δ(y),

where δ(y) is the correction applied to the source representation. Assume the source representation is normalized, i.e., ∥y∥2 = 1, and the correction satisfies ∥δ(y)∥2 ≤ ε. For two source samples yi and yj, let

zi = yi + δi, zj = yj + δj,

where δi = δ(yi) and δj = δ(yj). Lemma A.4 (Similarity stability under bounded correction). If ∥yi∥2 = ∥yj∥2 = 1 and ∥δi∥2,∥δj∥2 ≤ ε, then

|⟨zi,zj⟩ − ⟨yi,yj⟩| ≤ 2ε + ε2. If, in addition, ε < 1 and zˆi = zi/∥zi∥2, then

2ε 1 − ε

4ε 1 − ε

∥zˆi − yi∥2 ≤

, |⟨zˆi,zˆj⟩ − ⟨yi,yj⟩| ≤

.

Proof. For the unnormalized representations, |⟨zi,zj⟩ − ⟨yi,yj⟩| = |⟨yi + δi,yj + δj⟩ − ⟨yi,yj⟩|

= |⟨δi,yj⟩ + ⟨yi,δj⟩ + ⟨δi,δj⟩| ≤ |⟨δi,yj⟩| + |⟨yi,δj⟩| + |⟨δi,δj⟩| ≤ ε + ε + ε2 = 2ε + ε2.

For the normalized representation, ∥zi∥2 = ∥yi + δi∥2 ≥ 1 − ε. Therefore,

yi + δi ∥zi∥2

∥zˆi − yi∥2 =

− yi

2

∥δi∥2 ∥zi∥2

1 ∥zi∥2

≤

− 1 ∥yi∥2

+

ε 1 − ε

ε 1 − ε

2ε 1 − ε

≤

+

=

.

Finally,

|⟨zˆi,zˆj⟩ − ⟨yi,yj⟩| = |⟨zˆi − yi,zˆj⟩ + ⟨yi,zˆj − yj⟩| ≤ ∥zˆi − yi∥2∥zˆj∥2 + ∥yi∥2∥zˆj − yj∥2 ≤

4ε 1 − ε

.

| |
|---|

Thus, as long as the Euclidean norm of the correction is controlled, the pairwise inner-product structure of the source modality cannot be arbitrarily distorted.

###### A.7.2 Connection to Stage-II Bounded Residual Refinement

Stage II in the proposed method does not predict an unconstrained free mapping. Instead, it performs bounded correction on the phase, radius, and V -subspace components:

θˆ = wrap θ(0) + αθ tanh(∆θ) ,

ρˆk = ρ(0)k exp(αρ tanh(∆ρk)), vˆ = v(0) + αv tanh(∆vV ). Since |tanh(·)| ≤ 1, each type of correction is explicitly bounded:

ρˆk ρ(0)k

|θˆk − θk(0)| ≤ αθ,

∈ [e−α

,eα

],

ρ

ρ

and the V -side correction is controlled by αv. For the k-th two-dimensional polar block, let the initialized Cartesian coordinate be

c(0)k = ρ(0)k (cosθk(0),sinθk(0)), and the updated coordinate be

cˆk = ρˆk(cosθˆk,sinθˆk). Let sk := ρˆk/ρ(0)k and let ∆k := θˆk − θk(0) denote the wrapped angular difference. Then sk ∈ [e−α

] and |∆k| ≤ αθ. In complex notation,

,eα

ρ

ρ

cˆk − c(0)k ρ(0)k

(0)

k (skei∆

= eiθ

− 1).

k

Thus,

∥cˆk − c(0)k ∥22 = (ρ(0)k )2|skei∆

− 1|2. Furthermore,

k

− 1|2 = (sk − 1)2 + 4sk sin2(∆k/2). Using the bounds on sk and ∆k, we obtain

|skei∆

k

∥cˆk − c(0)k ∥2 ≤ ρ(0)k κ(αθ,αρ), where

− 1)2 + 4eαρ sin2(αθ/2). For small αθ and αρ, we have the approximation

κ(αθ,αρ) := (eαρ

κ(αθ,αρ) ≈ αθ2 + αρ2. This shows that phase and radial corrections jointly induce a controlled local Euclidean perturbation. For the entire U-subspace, let c(0) be the concatenation of all two-dimensional blocks. Orthogonality across blocks gives

∥cˆ− c(0)∥2 ≤ κ(αθ,αρ)∥c(0)∥2. If the V -side correction is further controlled by norm clipping or regularization such that

∥vˆ − v(0)∥2 ≤ βv, then the overall unnormalized correction satisfies

∥(ˆc,vˆ) − (c(0),v(0))∥2 ≤ κ(αθ,αρ)∥c(0)∥2 + βv. Since ∥c(0)∥2 ≤ 1, we further have

∥(ˆc,vˆ) − (c(0),v(0))∥2 ≤ κ(αθ,αρ) + βv. Define the effective perturbation radius

εeff := κ(αθ,αρ) + βv. When εeff < 1, Lemma A.4 applies directly to the Stage-II bounded residual refinement.

If no explicit V -side norm clipping is used in implementation, the quantity ∥vˆ−v(0)∥2 can instead be monitored empirically and controlled through αv, regularization, or early stopping. In this case, Lemma A.4 provides a conditional guarantee: as long as the realized correction norm remains small, the source-modality similarity structure is stably preserved.

###### A.7.3 Implication

The above analysis shows that the bounded parameterization of Stage II is not merely an engineering choice. It provides a tunable mechanism for balancing semantic preservation and target-distribution compatibility. Larger values of αθ, αρ, and αv allow stronger geometric correction, but may increase the risk of distorting the semantic structure of the source modality. Smaller correction scales better preserve the source geometry, but may be insufficient for entering the target-modality support. Effective alignment therefore requires a structured trade-off between correction strength and semantic stability.

###### A.8 Geometric Motivation for the Periodic Phase Prior

The previous sections explain why dominant residual directions should be corrected and why the correction should be bounded. We now explain why the proposed method uses two-dimensional blockwise polar coordinates in the dominant subspace and learns a target-modality prior in phase space.

###### A.8.1 Two-Dimensional Blockwise Polar Decomposition

Within the dominant subspace U, the method partitions the projected coordinates into multiple twodimensional blocks. For the k-th block, let its Cartesian coordinate be (ak,bk). The corresponding polar coordinates are

ρk = a2k + b2k + ϵ, θk = atan2(bk,ak). Here, ρk represents the radial magnitude or energy of the block, whereas θk represents its direction or phase. This decomposition separates geometric variation in the dominant subspace into two components. Radial variation controls the amplitude or energy of each block, while phase variation controls the directional structure within each block. Since the phase variable lies on the periodic domain [−π,π), it naturally has circular geometry. Consequently, directly modeling phase variables with ordinary Euclidean Gaussian noise is not fully appropriate; a more natural choice is to use wrapped Gaussian noise or other periodic distributions.

###### A.8.2 Phase Marginals and Phase Couplings

The phase distribution of the target image modality contains two types of information. The first is the marginal phase preference of each two-dimensional block, which can be represented by the circular mean

(x)

ψ¯k = arg E[eiθ

k ] .

(x)

If θk(x) is close to uniformly distributed on the circle, then E[eiθ

k ] is close to 0. If it is concentrated around

a certain direction, this magnitude becomes large. The second type of information is the dependency between phase differences across different blocks. We define

k −θℓ(x)) .

(x)

Mkℓ(x) = E ei(θ

The magnitude |Mkℓ(x)| measures the consistency of the phase difference between the k-th and ℓ-th blocks, while arg(Mkℓ(x)) gives the empirical phase offset. Therefore, the marginal anchors ψ¯k, block weights αk, pairwise coupling strengths Akℓ, and phase offsets ηkℓ constructed in Stage I can be interpreted as low-order statistics of the target image modality in phase space.

###### A.8.3 Periodic Potential and Drift Field Based on the marginal and pairwise phase statistics above, we define the periodic potential

αk 1 − cos(ϕk − ψ¯k) +

Akℓ [1 − cos(ϕk − ϕℓ − ηkℓ)].

Ψ(ϕ) =

k

(k,ℓ)∈E

This potential contains two types of terms. The first encourages each phase variable to approach its targetmodality marginal phase anchor. The second encourages phase differences between blocks to follow the dependency structure observed in the target modality.

Taking the gradient with respect to ϕk gives the drift field used in the main text: [∇ϕΨ(ϕ)]k = αk sin(ϕk − ψ¯k) +

Akℓ sin(ϕk − ϕℓ − ηkℓ).

ℓ:(k,ℓ)∈E

Thus, the periodic phase prior in Stage I can be interpreted as a local geometric constraint field in periodic phase space, constructed from the internal phase statistics of the target image modality.

###### A.8.4 Wrapped Gaussian Score Prior

Because phase variables are periodic, perturbed phases must be mapped back to [−π,π) through a wrap operation. The wrapped Gaussian provides a natural local noise model on periodic domains.

Given a phase vector ϕ, we first construct the drifted phase center

µϕ = wrap(ϕ − τ∇ϕΨ(ϕ)). Then we sample a perturbed phase vector

√

ϕ˜ = wrap µϕ +

2σtϵ , ϵ ∼ N(0,I).

The phase score prior sϕ is trained to predict the score of this wrapped Gaussian distribution in phase space. As a result, sϕ does not directly learn a text-to-image mapping; instead, it learns the internal periodic phase structure of the target image modality.

###### A.8.5 Implication

The Stage-I phase prior is not an arbitrary additional module. It follows from three observations: the residual modality gap is direction-dependent; the dominant subspace can be organized into two-dimensional blocks and expressed in polar coordinates; and the target modality exhibits estimable internal statistics over phase variables and phase differences. Therefore, the periodic phase prior provides a target-modality geometric constraint for the Stage-II bounded correction. It guides the source representation toward target-modality compatibility without relying on unconstrained distribution matching.

### B Experiment Details

###### B.1 Setting

❶. Geometric Level. We follow the diagnostic setting in Sec. 3. The target modality is the image representation set X, and the source modality is the text representation set Y . Given paired image-text representations {(xi,yi)}ni=1, where xi ∈ X, yi ∈ Y , and the two representations are semantically matched, all embeddings are evaluated in the shared normalized representation space and are ℓ2-normalized before metric computation. To avoid leaking pairwise correspondence into the alignment process, we separate the data into two parts. The first part is used as the statistic-estimation set, where only the marginal distributions of image and text representations are used to estimate the statistics required by each method, such as means, covariance-related quantities, subspaces, or residual structures. No image-text pairing information is used in this stage. The second part is used as a held-out paired diagnostic set. It is used only for evaluation. For any alignment method T, we transform each source representation yi into a target-modality substitute representation zi = T(yi), and evaluate the relation among the original source representation yi, the transformed representation zi, and the target representation xi on the same held-out pairs. Unless otherwise specified, all metrics are computed on 10K held-out paired samples, and k = 20 is used for nearest-neighbor-based metrics.

❷. MLLM Level. We use Llama-3-8B-Instruct as the language-model backbone and connect modality features to the LLM through a two-layer MLP projector with GELU activation. In our setting, the aligned text representations are treated as substitute visual tokens. These text-induced representations are first projected by the MLP into the LLM embedding space and then used as visual-style inputs for multimodal training. The training procedure follows a two-stage pipeline. ❶ Modality Substitution Pretraining, we train only the projector on the filtered Bunny-1M dataset for 1 epoch, with the LLM frozen. The learning rate is set to 5 × 10−4. ❷ Visual Instruction-Tuning, we initialize the projector from the first stage and conduct full-parameter fine-tuning on InternVL-Chat-V1.2 for 1 epoch. The learning rate is reduced to 1 × 10−5. All experiments are performed on 8 NVIDIA H200 GPUs. With approximately 2.2M training samples in total, the full training pipeline takes about 12 hours.

###### B.2 Metrics

Source Modality. This evaluation does not rely on downstream tasks or human semantic labels. Instead, it directly measures self-consistency in the representation space. We use the original source representations Y as the semantic reference and compare them with the transformed representations Z = T(Y ). ❶ First, we measure instance-level semantic consistency: Φ(T) = n1 ni=1 yi⊤zi. Since all representations are normalized, yi⊤zi is the cosine similarity between the original source representation and its transformed substitute. A larger Φ(T) indicates that the transformed representation remains close to its original source semantic position. ❷ Second, we measure whether the relative geometry within the source modality is preserved. For a randomly sampled pair set P, we define Ψ(T) = corr({yi⊤yj}(i,j)∈P,{zi⊤zj}(i,j)∈P). A larger Ψ(T) means that pairwise semantic relations in the source modality remain stable after transformation. ❸ Third, we measure local neighborhood consistency. Let NkY (yi) denote the k-nearest-neighbor set of yi in the original source space Y , and let NkZ(zi) denote the k-nearest-neighbor set of the same sample after transformation. We define Ωk(T) = n1 ni=1 |NkY (yi)∩NkZ(zi)|/k. A larger Ωk(T) indicates that local semantic neighborhoods are better preserved after alignment.

Target Modality. Second, we evaluate local modality mixing, which captures whether X and Z are locally interleaved rather than only globally close. Let Q = X ∪ Z. For any u ∈ Q, let Nk(u) denote its k-nearestneighbor set in Q \ {u}, and define the local target-modality proportion as pk(u) = k1 v∈N

k(u) 1[v ∈ X]. Here, 1[v ∈ X] is not a semantic label, but a modality-origin indicator. We use the binary entropy H2(p) = −plog2 p − (1 − p)log2(1 − p) to measure local modality mixing, and normalize the score by the expected value under random permutation of modality-origin labels. We report two directional scores: MkZ(T) and MkX(T). MkZ(T) measures whether transformed source representations enter the support region of the target modality, while MkX(T) measures whether the target-modality support is covered by transformed source representations. For each method, we define riT = xi − zi, and compute the residual covariance ΣTr = n1 ni=1 riT(riT)⊤. We examine the normalized residual eigenvalue spectrum λj(ΣTr )/tr(ΣTr ), and compute the residual anisotropy ratio Ar(T) = λmax(ΣTr )/(tr(ΣTr )/d). A smaller Ar(T) and a less concentrated residual spectrum indicate that the method better suppresses the dominant anisotropic residual directions identified in Sec. 3.2.

###### B.3 Baselines

Unicorn. Unicorn is a text-only data synthesis framework for VLM training. It constructs multimodal training data without real images through a three-stage pipeline: diverse caption synthesis, instruction-tuning data generation, and modality representation transfer. In particular, Unicorn first expands sparse caption seeds into diverse captions, then generates instruction-tuning data from these captions, and finally transfers text representations encoded by LLM2CLIP into the visual representation space to obtain synthetic image representations. In our experiments, we use Unicorn as a text-only synthetic visual representation baseline, following its modality representation transfer setting to construct pseudo-visual features from text.

- C3 Align. C3 is a simple training-free modality-gap correction baseline built on the Connect-Collapse-Corrupt principle. The Connect step assumes that multimodal contrastive learning has already placed related concepts from different modalities into a shared representation space. The Collapse step removes the dominant modality gap by subtracting modality-specific embedding means. The Corrupt step injects Gaussian noise as regularization to improve robustness under alignment noise. In our setting, given a source text representation

y, we first shift it toward the target image centroid as yµ = y − µy + µx, then add Gaussian perturbation and normalize the result to obtain the aligned substitute representation.

ReAlign. ReAlign is a training-free statistical alignment baseline that maps source-modality representations into the target-modality distribution using low-order statistics estimated from unpaired data. It consists of three closed-form steps: Anchor Alignment, Trace Alignment, and Centroid Alignment. First, Anchor Alignment removes the first-order mean bias by shifting the centered source representation to the target centroid. Second, Trace Alignment rescales the centered source residual using a global trace-matching factor, thereby matching the target residual energy while preserving the source covariance structure. Finally, after spherical projection, Centroid Alignment corrects the induced centroid drift and re-normalizes the representation on the unit sphere. In our experiments, we apply this operator to text representations to obtain ReAlign substitute visual

representations.

###### B.4 Evaluation Setting

We evaluate the model on a broad set of multimodal benchmarks covering three aspects of visual understanding. ❶ For General Perception, we use MME [4] test, MMStar [2], ScienceQA [11]-image dev&test, and RealWorldQA. For ❷ Complex reasoning, we evaluate on MMMU [18] validation single-image, MMMU-Pro[19] single-image, VisuLogic [15] train, and LogicVista [14]. For ❸ Hallucination assessment, we use CRPE [13], POPE [8], and HallusionBench [5]. Across all benchmarks, we report accuracy (acc) as the unified evaluation metric, enabling a consistent comparison among different methods.

### C Applicability

Our analysis and method are built on the premise that the source and target modalities are embedded into a shared normalized representation space produced by a pretrained multimodal contrastive encoder. In this setting, the modality gap is assumed to arise within an already semantically compatible space: the two modalities share dominant geometric structure, while the remaining discrepancy appears as a structured anisotropic residual. This premise is important because AnisoAlign is designed to correct such residual geometric mismatch, rather than to align two arbitrary or unrelated distributions from scratch. Therefore, when the pretrained encoder fails to establish a meaningful shared semantic space, or when the source and target modalities do not exhibit compatible dominant geometry, the modality-gap structure studied in this work may become weak or absent, and explicit anisotropic correction may be less effective.

