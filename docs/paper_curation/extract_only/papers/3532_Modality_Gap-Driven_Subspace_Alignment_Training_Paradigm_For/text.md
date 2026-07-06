# arXiv:2602.07026v3[cs.CV]5Jun2026

[Figure 1]

[Figure 2]

## Modality Gap–Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models

### Xiaomin Yu1,2, Yi Xin3,4, Yuhui Zhang5, Wenjie Zhang1, Chonghan Liu6 Hanzhen Zhao2, Chen Liu7, Xiaoxing Hu8, Ziyue Qiao9, Hao Tang10 Xiaobin Hu2, Chengwei Qin1, Hui Xiong1, Yu Qiao4, Shuicheng YAN2

1HKUST(GZ), 2NUS, 3sh AILab, 4SII, 5Stanford, 6UCLA, 7Yale, 8SJTU, 9GBU, 10PKU

### Abstract

Despite the success of multimodal contrastive learning in aligning visual and linguistic representations, a persistent geometric anomaly, the Modality Gap, remains: embeddings of distinct modalities expressing identical semantics occupy systematically offset regions. Prior approaches to bridge this gap are largely limited by oversimplified isotropic assumptions, hindering their application in large-scale scenarios. In this paper, we address these limitations by precisely characterizing the geometric shape of the modality gap and leveraging it for efficient model scaling. First, we propose the Fixed-frame Modality Gap Theory, which decomposes the modality gap within a frozen reference frame into stable biases and anisotropic residuals. Guided by this precise modeling, we introduce ReAlign, a training-free modality alignment strategy. Utilizing statistics from massive unpaired data, ReAlign aligns text representation into the image representation distribution via a three-step process comprising Anchor, Trace, and Centroid Alignment, thereby explicitly rectifying geometric misalignment. Building on ReAlign, we propose ReVision, a scalable training paradigm for Multimodal Large Language Models (MLLMs). ReVision integrates ReAlign into the pretraining stage, enabling the model to learn the distribution of visual representations from unpaired text before visual instruction tuning, without the need for large-scale, high-quality image-text pairs. Our framework demonstrates that statistically aligned unpaired data can effectively substitute for expensive image-text pairs, offering a robust path for the efficient scaling of MLLMs.

Date: June 8, 2026 Leader: Xiaomin Yu (yuxm02@gmail.com) Correspondence: Xiaobin Hu, Chengwei Qin Github: https://github.com/Yu-xm/Modality_Gap_Theory.git

### 1 Introduction

Multimodal contrastive learning [11, 20, 22, 32, 38] has established itself as the standard paradigm for aligning visual and text representations. However, despite extensive training on massive image-text pairs, a persistent empirical observation remains:

Definition. Modality Gap Phenomenon

For two modalities that express the same underlying semantics, their representations typically do not coincide. Instead, the two modalities inherently occupy distinct, systematically offset regions of the joint representation space. This phenomenon is called the Modality Gap [15, 39].

While this gap hinders direct cross-modal interchangeability, it also suggests that, if the geometric misalignment can be effectively bridged, abundant text data could serve as a substitute [35] for expensive image-text data in Multimodal Large Language Models (MLLMs) training.

Prior research has largely advanced along two directions, yet both face limitations: 1 Geometric Correction: These approaches attempt to post-hoc correct this gap via explicit geometric projections [33, 39]. However, most of these works are limited to single, small-scale tasks [12, 17, 19, 21, 23] such as Image Captioning, failing to unlock the true potential of modality gap theory for model scaling. Furthermore, they typically rely on isotropic noise assumptions, failing to account for the complex anisotropic structures in high-dimensional spaces.

###### 2 Text-only Large-scale Training: These methods [35] leverage modality gap to synthesize pseudo-visual supervision signals from pure text. While promising, existing methods suffer from substantial performance degradation on fine-grained visual tasks, exposing a pronounced distributional gap between synthetic text representations and real-world image data.

These limitations point to a fundamental mismatch: existing assumptions are overly simplified, resulting in a lack of precise modeling of the modality gap’s geometric shape, which in turn hinders its application in large-scale training. This mismatch naturally motivates two core research questions:

- 1 On Shape: Can we move beyond simple mean assumptions to precisely characterize the intrinsic geometric shape of the modality gap within a stable reference frame?

- 2 On Scale: Can we leverage this precise shape modeling to design a scalable training paradigm that substitutes expensive paired data with massive, easily accessible unpaired data, thereby achieving efficient MLLM scaling?

To address the first question regarding shape, we conduct the first empirical study that trains a contrastive dual-encoder [27] from scratch to precisely track and model the evolution of the modality gap. Based on this microscopic analysis, we propose a unified theoretical framework: we no longer treat the modality gap as random fluctuations but mathematically decompose it within a frozen reference frame (Rd = U ⊕ V ). By explicitly separating the effective task subspace (U), where gradients concentrate and semantic information resides, from its orthogonal complement (V ), we reveal the dual geometric structure of the modality gap: it comprises not only a stable bias component but also a residual component characterized by specific second-moment properties. This discovery allows us to transcend simple mean-based descriptions and fully capture the anisotropic distribution of the modality gap across different subspaces.

Building on this precise geometric modeling, we further address the second question regarding scale. Our approach highlights a perspective different from prior work: while acquiring high-quality paired image-text data [4, 10, 13] is expensive, obtaining massive amounts of unpaired single-modality data is extremely easy. We believe that leveraging such large-scale unpaired data is sufficient to precisely reconstruct the shape of the modality gap via statistical laws, without relying on expensive paired samples. Guided by this insight, we introduce two core contributions:

- 1 ReAlign. Building on the geometric analysis in Sec. 3, we introduce ReAlign, a training-free pre-alignment strategy that maps text representations into the image representation distribution using statistics derived from large-scale unpaired data. ReAlign operates through a three-stage procedure. First, Anchor Alignment matches first-order statistics (means). Second, Trace Alignment matches the global variance scale. Finally,

- Centroid Alignment explicitly rectifies the geometric drift induced by spherical projection. Together, these stages achieve precise cross-modal alignment at the statistical level using only linear transformations and normalization, without requiring any additional training.
- 2 ReVision. We incorporate ReAlign into a two-stage MLLM training pipeline termed ReVision. In the first stage, Modality Substitution Pretraining, the ReAlign operator is used to convert large-scale long-form text into pseudo-visual representations. An adapter is trained on these representations while keeping the LLM backbone frozen, enabling the model to absorb rich world knowledge and visual semantics purely from text data, without relying on costly image-text pairs. In the second stage, Visual Instruction Tuning, real images are introduced for standard supervised learning to supplement fine-grained visual details that may be lost under purely statistical alignment, thereby refining the model’s ability to follow complex instructions.

Overall, we reframe the modality gap as a structured geometric mismatch. By characterizing its first-order, second-order, and normalization-induced components, we derive a training-free alignment strategy that enables scalable use of unpaired data for multimodal learning. This provides both a geometric understanding of modality alignment and a practical path toward more cost-efficient MLLM scaling.

### 2 The Isotropic Fallacy

The C3 [39] framework established the dominant paradigm for addressing the modality gap, and subsequent state-of-the-art strategies have inherited its assumption, characterizing the gap simply as a superposition of a centroid shift and random alignment noise. While their centroid correction effectively rectifies the first-order bias, their treatment of the residual gap relies on a critical simplification: that the residual fluctuations are isotropic. We argue that this assumption is geometrically flawed. Multimodal contrastive representation distributions are inherently anisotropic, where information is encoded in a hierarchical spectral structure rather than a uniform sphere. As we demonstrate in Appendix D, imposing an isotropic prior onto this structured manifold induces a spectral whitening effect, which dilutes the fine-grained semantic hierarchy and distorts the angular topology. This mismatch suggests that merely adjusting the mean is insufficient; the geometric shape of the noise matters. To address this, we must first rigorously characterize the true geometry of these fluctuations. In the following section, we introduce a formal decomposition framework to reveal that the modality gap is driven not by isotropic noise, but by highly structured, direction-dependent biases and residuals.

### 3 Modality Gap

Motivated by the isotropic fallacy discussed in Appendix 2, this section answers one core question: what geometric shape does the modality gap have? To this end, we train a dual-encoder model based on the InfoNCE loss and directly observe the evolution of representation geometry during training. Using a data-driven subspace construction, we explicitly separate bias and residual components. Our objective is to characterize, around a fixed reference time, the slow drift that persists late in training together with its second-moment structure. This perspective has two advantages: 1 bias and residual terms can be estimated separately; 2 all theoretical claims reduce to second-moment conditions estimable from finite-sample statistics. As a result, the shape of the modality gap is characterized by three estimable objects: 1 first-order mean bias,

2 second-order residual, and 3 spherical centroid drift. Appendix A discusses why such a gap structurally persists in dual-encoder contrastive learning. Formal proofs of the decomposition and alignment properties are provided in Appendix B.

#### 3.1 Modality Gap Decomposition Framework

Directly analyzing the full gap vector mixes task-active representation directions with directions that carry little contrastive signal. We therefore construct a data-driven dominant subspace U from the probe covariance, so that the gap can be separated into components aligned with the main representation geometry and components lying in its orthogonal complement.

Prior work has shown that the optimization dynamics of multimodal contrastive learning can be decomposed into two subspaces: (1) an effective subspace, through which most gradients propagate, and (2) an ineffective

subspace, in which gradients have only a limited effect. Following this perspective, we analyze the modality gap by decomposing the representation space into these two subspaces.

Fixed Reference Frame Construction. Let ex(t),ey(t) ∈ Rd denote the unit-normalized embeddings of the image modality x and the text modality y for the same sample at training step t. We fix a reference time t0, and compute the empirical covariance on a held-out probe set as Σ(ˆ t0) := Covp(ex(t0)) + Covp(ey(t0)). We perform eigendecomposition Σ(ˆ t0) = QΛQ⊤, take the top r principal directions to construct the fixed reference subspace U := span{q1,...,qr}, and let V := U⊥. Here, r is determined by a fixed energy threshold. We denote by PU and PV the orthogonal projectors onto U and V , respectively, and keep them fixed throughout the analysis for all t ≥ t0. Here, U is the dominant representation subspace estimated from the probe covariance, rather than a predefined semantic space.

Bias-Residual Decomposition. For a paired sample, define the modality gap as ∆(t) := ex(t) − ey(t), and define the overall mean gap as m(t) := E[∆(t)]. Under the fixed frame U ⊕ V , the first-order mean gap is orthogonally decomposed as m(t) = β(t) + γ(t), where β(t) := PUm(t) and γ(t) := PV m(t). β(t) is the mean component in the dominant representation subspace; γ(t) is the mean component in the orthogonal complement, called POB. Correspondingly, the zero-mean residual is decomposed as δ(t) := PU(∆(t) − m(t)), and ζ(t) := PV (∆(t) − m(t)). Therefore, the fixed-reference decomposition of the modality gap is:

∆(t) = β(t) + γ(t) + δ(t) + ζ(t). (1)

Here, β(t) and γ(t) describe the first-order mean shape, while δ(t) and ζ(t) describe the second-order residual shape.

Along the temporal dimension, we focus on a short late-training window around the reference time t0, denoted by t ∈ T := {t0,...,t0 + τ − 1}. Within this window, the fixed reference subspaces U,V and the projectors PU,PV remain unchanged, so all components β(t),γ(t),δ(t),ζ(t) are compared in the same reference frame. To characterize the rotation of the dominant representation subspace as training progresses, we also re-estimate an instantaneous probe subspace Ut at each logging step using the same covariance construction, and measure its geometric deviation from the fixed reference subspace by the largest principal angle θ(Ut,U).

1.00

| | |
|---|---|
|Leakage le Geo. baseli|akref ne<br><br>|
| |sin|
| | |
| | |

###### LeakageRatio

0.75

0.50

0.25

0.00

0 2000 4000

(a) Gradient Leakage

1.00

###### MetricValue

0.75

0.50

0.25

Stability cos

Drift d(t)

0.00

0 2000 4000

(b) Passive Orthogonal Bias Drift

| | |
|---|---|
| | |
| | |
| |Anisotropy U Spectral corr. eig<br><br>|
| |Diag. corr.<br><br>|
| | |

| | |
|---|---|
| | |
|Anisotropy V Angle ( ,v1V)<br><br>| |

1.00

- 101
- 102
- 103

###### Cond. umber

###### Cond. umber

###### Correlation

###### Angle/vMF

100

0.75

104

0.50

50

0.25

Conc. vMF

0.00

0

0 2000 4000

0 2000 4000

(c) U-side Residual Spectral Locking

(d) V-side Residual Decoupling

- Figure 1 Geometric statistics of the modality gap, showing gradient leakage, passive bias drift, and anisotropic residual structures in the fixed U ⊕ V reference frame.

#### 3.2 Bias Terms: First-order Bias

This section characterizes the shape of the first-order bias in the modality gap. Under the fixed reference frame U ⊕ V , the mean modality gap can be decomposed as m(t) = β(t) + γ(t), where β(t) = PUm(t) and γ(t) = PV m(t). Therefore, the first-order bias consists of two components: the principal mean bias β(t) in the dominant representation subspace U, and the passive orthogonal bias γ(t) in the orthogonal complement V .

PMB: Mean Bias in the Dominant Subspace. β(t) is the mean displacement of the modality gap inside the dominant representation subspace U, which we refer to as the principal mean bias, or PMB. Geometrically, it corresponds to the projection of the anchor mismatch between the two modality representation distributions

- onto U. Therefore, β(t) is not residual noise, but a deterministic centroid offset along task-active representation directions. The main statistical role of PMB is to center the U-side residual. Since δ(t) = PU(∆(t) − m(t)), removing β(t) = PUm(t) makes δ(t) a zero-mean residual. Otherwise, the estimate of ΣU would contain both the first-order mean displacement and the second-order residual fluctuation, and thus could not serve

as a pure statistic of residual shape. Therefore, PMB is both the anchor displacement inside the dominant subspace and the centering term required for the covariance analysis of the U-side residual.

POB: Passive Bias in the Orthogonal Complement. γ(t) is the mean displacement of the modality gap inside the orthogonal complement V , which we refer to as the passive orthogonal bias, or POB. Unlike β(t), which lies in the dominant representation subspace, γ(t) lies outside the subspace where most late-stage optimization signals concentrate. This makes it a passive bias: it is not directly suppressed by a strong gradient component along V , but instead evolves slowly under weak residual updates. To quantify this effect, we measure the reference leakage ratio leakref(t), defined as the fraction of embedding-level gradient RMS energy projected

- onto V . Fig. 1(a) shows that this leakage remains low in late training. This is consistent with the structure of InfoNCE: embedding-level gradients are constrained by the span of the current contrastive set, and late-stage embeddings concentrate near the instantaneous dominant subspace Ut. Since the fixed reference subspace U may rotate relative to Ut, the observed leakage along V is compared with the geometric baseline sinθ(Ut,U). Fig. 1(a) shows that this leakage remains low in late training, indicating that only a small fraction of the

optimization signal directly acts along V . We further compare it with the geometric baseline sinθ(Ut,U), which measures the V -projection induced by the rotation of the instantaneous dominant subspace relative to the fixed reference frame. This comparison suggests that late-stage V -direction optimization pressure is weak, explaining why γ(t) is not rapidly eliminated. Appendix C further analyzes possible weak coupling between U- and V -side residuals. Consequently, γ(t) is not rapidly eliminated. Fig. 1(b) shows that it maintains high adjacent-step cosine stability while undergoing slow cumulative drift. We characterize this behavior by the relative drift magnitude drift(t) := ∥γ(t) − γ(t0)∥/max{∥γ(t0)∥,ε} and the adjacent-step cosine stability cos(γ(t),γ(t − 1)). In late training, this can be approximated as γ(t + 1) = γ(t) + η(t), where ∥η(t)∥ is small. Thus, the typical shape of POB is not rapid convergence to zero, but slow drift along an approximately stable direction.

Conclusion 1. Bias Shape

The first-order bias is a decomposable anchor displacement, m(t) = β(t) + γ(t). Here, β(t) is the anchor displacement inside the dominant representation subspace, serving to center the residual along task-active directions, while γ(t) is the passive mean bias in the orthogonal complement, whose direction remains stable and drifts slowly because late-stage gradients along V are weak. Therefore, the first-order structure of the modality gap should be understood as a translation-correctable anchor mismatch at the mean level.

#### 3.3 Residual Terms: Anisotropic Second-order Residual

After the first-order mean is decomposed and centered, can the remaining residual be regarded as isotropic white noise? We show that the answer is no: the residual itself has a stable second-order shape. For the late-

training window s = 0,...,τ − 1, we compute the residual covariances at each step as Σ(Us) := CovP(δ(t0 + s)) and Σ(Vs) := CovP(ζ(t0 + s)), and then average them over time as Σ¯U := τ1 τs=0−1 Σ(Us) and Σ¯V := τ1 τs=0−1 Σ(Vs). This step-wise covariance estimation avoids mixing the slow drift of POB into the residual covariance. We use the trace to measure the residual energy scale, and the effective condition number κeff to measure the spectral stretching of the dominant residual components.

- U-side Residual. The residual δ(t) lies in the dominant representation subspace U. Fig. 1(c) shows that

κeff(Σ¯U) remains high throughout training, indicating that the U-side residual does not diffuse uniformly. Instead, its energy concentrates along a small number of principal directions. This suggests a stable spectral hierarchy inside the dominant representation subspace, where different directions carry different amounts of residual energy. To compare this residual structure with the optimization signal, let GU(t) denote the embedding-level gradient covariance restricted to U, and define ρeig(t) := corr[λ(ΣU(t)),λ(GU(t))], where λ(·) denotes the eigenvalue vector arranged in descending order. Fig. 1(c) shows that ρeig(t) rises rapidly in early training and remains stable in late training. This indicates a stable correspondence between the spectral energy profile of the U-side residual covariance and that of the gradient covariance. Therefore, the U-side residual is a structured fluctuation with a stable spectral hierarchy, rather than an isotropic perturbation.

- V -side Residual. The residual ζ(t) lies in the orthogonal complement V . Fig. 1(d) shows that κeff(Σ¯V ) remains

higher than the isotropic baseline, indicating that the V -side residual also forms an elongated ellipsoidal shape. Although its total energy tr(Σ¯V ) is lower than that of the U-side residual, it is still directionally structured. Figure 1(d) further shows that γ(t) is persistently close to orthogonal to the first principal direction of Σ¯V . If POB were merely the leading direction of V -side residual spread, these two directions would align. Their near-orthogonality shows that the first-order mean bias and the second-order residual anisotropy are geometrically decoupled in V : the former is a stable mean displacement, while the latter is a residual ellipsoid with its own spectral structure.

Conclusion 2. Residual Shape

After mean alignment, the residual is an anisotropic distributional mismatch. On the U-side, the residual energy concentrates along a few dominant spectral directions and forms a stable spectral hierarchy that is aligned with the spectral profile of the optimization signal. On the V -side, the residual has lower total energy but still forms an elongated ellipsoid, and this ellipsoid is geometrically decoupled from the POB mean direction γ(t). Therefore, the second-order structure of the modality gap is a stable anisotropic residual shape.

#### 3.4 Phantom Drift: Normalization-induced Secondary Centroid Drift

If the first-order mean is already aligned in Euclidean space, can we guarantee that the centroid of the normalized embeddings is also aligned? The answer is no: spherical normalization couples the first-order anchor with the second-order residual shape, producing Phantom Drift.

Let the source-modality representation after linear mean alignment be z = µ + ϵ, where E[ϵ] = 0 and Cov(ϵ) = Σ. Here, µ is the aligned target anchor, and ϵ is the zero-mean residual. In Euclidean space, E[z] = µ, so the first-order mean has already been aligned. However, the actual embedding is projected onto the unit sphere by π(z) := z/∥z∥. The centroid after normalization is µπ := E[π(z)] = E[(µ + ϵ)/∥µ + ϵ∥]. Since π(·) is nonlinear, in general µπ ≠ π(µ). We define the normalization-induced secondary centroid drift as ∆π := µπ − π(µ) = E[(µ + ϵ)/∥µ + ϵ∥] − µ/∥µ∥. This drift depends on both the anchor direction µ and the residual covariance Σ, so an anisotropic residual can shift the spherical centroid even after Euclidean mean alignment. This is Phantom Drift.

Conclusion 3: Phantom Drift

Mean alignment in Euclidean space does not guarantee centroid alignment on the unit hypersphere. After normalization, the anchor direction couples with the residual covariance structure, causing a secondary shift of the spherical centroid of the representation distribution.

### 4 ReAlign: Training-Free Modality Alignment

Sec. 3 shows that the modality gap contains three geometric components: first-order anchor displacement, anisotropic second-order residual mismatch, and spherical-normalization-induced centroid drift. Based on this decomposition, we propose ReAlign, as shown in Fig. 2, a training-free modality alignment operator that estimates low-order statistics from large-scale unpaired data and performs three closed-form calibrations: 1

Anchor Alignment corrects the first-order anchor displacement, 2 Trace Alignment matches the global residual energy scale while preserving the learned anisotropic spectral structure, and 3 Centroid Alignment corrects the Phantom Drift induced by spherical normalization.

#### 4.1 Step 1: Anchor Alignment

- Motivation. Sec. 3.2 shows that the first-order structure of the modality gap appears as an anchor displacement between the two modality representation distributions. This displacement is not a random perturbation, but an estimable and translation-correctable global centroid mismatch at the mean level. Therefore, the first step of ReAlign is to translate the source-modality anchor to the target-modality anchor. Operation. Let ey,ex ∈ Rd denote the unit-normalized embeddings of the source modality y and the target

V

Source (µy)

| | |
|---|---|
| | |

U

Target (µx)

V

|Ancho|rAligned|
|---|---|
| | |

Step 1

U

Step 2

e˙y = (ey − µy) + µx

V

|Ce|ntroid µ′<br><br>|
|---|---|
| |Drift|

Scale s

U

Step 3

###### e˜y = µx + s(ey − µy)

V

V

|Correction|Phantom Drift|
|---|---|
| |Gap|

t

U

≈ 0

Final

|Aligne|Unit<br><br>d|
|---|---|
|Match|(eˆy)<br><br>Center|

Sphere

U

Norm r Match

###### e′′y = e′y − µ′ + µx

###### eˆy = e′′y/∥e′′y∥

(a) Original State

(b) Step1: Anchor Alignment

(c) Step2: Trace Alignment

(d) Step3: Centroid Align

(e) Final State

- Figure 2 The ReAlign pipeline. ReAlign sequentially performs Anchor Alignment, Trace Alignment, and Centroid Alignment to align the source-modality distribution toward the target-modality distribution while preserving its geometric structure on the unit hypersphere.

modality x, respectively. Their population means are µy := E[ey] and µx := E[ex]. To map the source modality y to the reference position of the target modality x, we first center the source-modality representation and then translate it to the target anchor:

e˙y = (ey − µy) + µx.

This operation satisfies E[e˙y] = µx. Therefore, Anchor Alignment removes the first-order mean mismatch between the source and target modalities. Anchor Alignment only corrects the first-order anchor position and does not destroy the second-order spectral structure of the source representation.

#### 4.2 Step 2: Trace Alignment

- Motivation. Sec. 3.3 shows that the residual after mean alignment is not isotropic noise, but an anisotropic residual shape with a stable spectral hierarchy. Therefore, full whitening or isotropic noise injection may remove the learned anisotropic spectral structure that carries task-relevant information.

Observation. Empirically, after contrastive pretraining, the two modalities exhibit compatible trace-normalized spectral profiles. Let the centered covariances be Σy := Cov(ey − µy) and Σx := Cov(ex − µx). Although Σy and Σx may differ in their overall residual energy scale, their trace-normalized spectra are close:

λ(Σy) tr(Σy) ≈

λ(Σx) tr(Σx)

.

This suggests that the remaining second-order calibration should primarily match the global residual energy scale, rather than whiten or reshape the full covariance. Appendix F shows why a stronger blockwise covariance alignment can be less effective.

Operation. We define the residual energy of each modality as the trace of its centered distribution: Ty := E[∥ey− µy∥2] = tr(Σy) and Tx := E[∥ex − µx∥2] = tr(Σx). We compute the trace-matching factor s = Tx/(Ty + ε), where ε is a small constant for numerical stability. Combining Anchor Alignment and Trace Alignment gives the affine-transformed source-modality representation:

e˜y = µx + s(ey − µy).

This transformation satisfies E[˜ey] = µx, and its residual energy is E[∥e˜y − µx∥2] = s2Ty ≈ Tx. This scalar scaling changes the residual energy scale without changing the eigenvectors or the trace-normalized spectrum of the source covariance. After this, we perform the first spherical projection:

e′y =

e˜y ∥e˜y∥

.

4.3 Step 3: Centroid Alignment

- Motivation. Sec. 3.4 shows that even after Anchor Alignment and Trace Alignment calibrate the mean and residual energy in Euclidean space, spherical normalization still induces a new centroid shift. Therefore, the final step of ReAlign explicitly corrects this Phantom Drift.

Operation. Let the centroid after the first spherical projection be µ′y := E[e′y]. Here, µx is used as the empirical spherical target centroid. The corresponding Phantom Drift is ∆πy := µ′y − µx. To correct this spherical centroid shift, we translate the projected source-modality representation back to the target anchor:

e′′y = e′y − µ′y + µx. Finally, we normalize again to return the representation to the unit hypersphere:

e′′y ∥e′′y∥

eˆy =

, eˆx = ex.

In practice, this second projection keeps the embeddings on the unit sphere while leaving only a small residual centroid error. We study the sample efficiency, numerical stability, and domain sensitivity of ReAlign in Appendix E.

### 5 ReVision: A Scalable MLLM Training Paradigm

We introduce ReVision, a two-stage training paradigm for MLLMs. Building on the geometry-preserving ReAlign strategy, ReVision synthesizes pseudo-modality features from unpaired text corpora. This design enables low-cost semantic injection, allowing the model to absorb extensive world knowledge during pretraining without relying on expensive, high-quality paired data.

- Stage 1: Modality Substitution Pretraining Let Eimg and Etext denote the frozen image and text encoders. Freed from the reliance on paired visual-text data, we leverage Etext to encode large-scale unpaired text corpora. Appendix G analyzes why longer captions do not necessarily provide better modality substitution signals. Unlike traditional multimodal datasets limited by data scarcity, this strategy allows training to scale to raw text resources. As a result, the model receives dense semantic supervision and can absorb broader world knowledge during pretraining than standard approaches.

We define a training-free modality substitution operator Sy→x based on Section 4. Given a text sample y, this operator maps its embedding into the image space distribution e˜x = Sy→x(Etext(y)). Here, e˜x serves as a pseudo-visual embedding. Owing to the geometry-preserving nature of ReAlign, it preserves the rich semantics of y while strictly adhering to the anisotropic geometric statistics of real images. We train the adapter ϕ (with LLM θ frozen) to reconstruct the text conditioned on the pseudo-visual embedding e˜x:

Lpre(ϕ) = −

T

t=1

log pθ(yt | y<t,Tϕ(˜ex)). (2)

- Stage 2: Visual Instruction-Tuning While Stage 1 establishes geometric compatibility, Stage 2 focuses on

enhancing capabilities in more challenging scenarios. In this stage, real image embeddings ex = Eimg(x) are introduced. Real images provide fine-grained visual details that may be abstracted away under purely statistical alignment, and are essential for handling complex instructions and intricate reasoning tasks. We fine-tune the model using standard supervised instruction tuning:

L

Lsft(θ,ϕ) = −

t=1

where (I,r) denotes an image-instruction pair.

log pθ(rt | r<t,I,Tϕ(ex)), (3)

Inference During inference, the model directly takes real images as input. This inherent compatibility stems from the asymmetric alignment strategy. By aligning the text representation distribution to the image representation distribution during pretraining, the model supports single-image inference without relying on statistics from multiple images for calibration, incurring no additional computational overhead.

### 6 Experiments

In this section, we systematically investigate the effectiveness of ReVision under the proposed geometrypreserving framework. Our experiments are designed to address three core research questions (RQs). To ensure a fair comparison, all methods employ the same model architecture. We use LLM2CLIP-Openai-L-14336 [11] as the encoder and Llama-3-8B-Instruct [7] as the LLM backbone. For ReVision training, we use bunny-pretrain [10] during the Modality Substitution Pretraining stage and InternVL-Chat-V1-2-SFT [6] for the Visual Instruction Tuning stage. Detailed training, evaluation, and cost settings are provided in Appendix H.

[Figure 3]

[Figure 4]

- RQ1. Does preserving anisotropy reduce the modality gap more effectively than isotropic corruption? We first verify the geometric alignment quality by quantifying the Euclidean distance between the centroids of the aligned modalities. We performed alignment on 100k samples from the Bunnypretrain [10] and DenseFusion [13] datasets. As illustrated in Figure 3, the original representations exhibit a substantial gap (≈ 0.4). While C3 [39] reduces the gap significantly via centroid subtraction, its reliance on isotropic noise limits the precision of the fit. The residual misalignment suggests that spherical noise cannot perfectly cover the anisotropic visual manifold. By explicitly modeling the covariance structure and correcting for manifold drift, ReAlign reduces the gap by orders of magnitude. On Bunny, the gap drops to 2.64 × 10−4, and on DenseFusion to 1.39 × 10−4. Notably, we observe that while the initial gaps vary across datasets (0.3918 for Bunny, 0.4276 for DenseFusion), C3 stagnates at a similar level (≈ 0.0023) for both. This performance plateau indicates a geometric bottleneck: isotropic noise lacks the flexibility to adapt to the representation variations of different distributions. In contrast, ReAlign breaks this bottleneck by adaptively matching the covariance trace of each dataset, demonstrating superior adaptability to the intrinsic geometric structure of diverse data sources. Additional spectral, angular, and visualization diagnostics are reported in Appendix D.
- RQ2. How effective is ReAlign in large-scale MLLM training scenarios? To ensure a fair comparison, all baselines adopt the same two-stage paradigm as ReVision: Stage 1 utilizes 1M text samples from Bunnypretrain, followed by Stage 2 using the InternVL-Chat-V1-2-SFT. We compare ReVision against: 1 Blind: We directly evaluate Qwen3-235B-A22B-Instruct [31] on text-only questions without providing image inputs. This demonstrates that without visual perception, even the most powerful LLMs falter on multimodal tasks. 2 No Align: Utilizing raw text representations. 3 C3 Align: Adopting the C3 strategy, which performs centroid alignment and injects isotropic Gaussian noise into text representations. Results are shown in the Table 1. ReVision achieves the highest average score of 50.16, significantly outperforming C3 (48.06). In reasoningintensive benchmarks, ReVision surpasses C3. This supports our point that C3’s isotropic noise induces a whitening effect that erodes the fine-grained semantic hierarchy essential for complex reasoning. ReAlign preserves the spectral decay, maintaining the structural richness of the features. ReVision demonstrates a clear advantage in hallucination metrics (CRPE:81.78, HallBench:46.58). We attribute this to the correction of Phantom Drift. By ensuring the centroid is aligned on the hypersphere, ReVision prevents the projection layer from overfitting to spurious directional biases, enabling more faithful visual grounding. Additional MLLM ablations and scaling results are provided in Appendix I.

- RQ3. Can scaling up low-cost text-only pretraining surpass the performance of expensive paired image-text training? Finally, we investigate whether ReVision can challenge the traditional paradigm that relies on expensive paired image-text data. To ensure fairness, we sampled a subset of 417k examples from the SFT

(a) Bunny - 100k (b) Densefusion - 100k

Figure 3 We measure the modality gap between aligned centroids on Bunny and DenseFusion. While the baseline C3 stagnates at a geometric bottleneck (≈ 0.0023) due to isotropic assumptions, ReAlign reduces the gap to the 10−4 scale by effectively modeling anisotropic covariance.

Table 1 Performance comparison of different geometric alignment strategies.

General Reasoning Hallucination

Avg. ↑

Method

MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

Blind 3.37 8.80 6.17 5.36 19.60 12.44 0.30 1.56 12.90 0.60 15.25 7.85 No Align 73.63 35.73 75.23 43.53 28.82 25.38 24.40 21.03 80.82 71.59 42.38 47.50 C3 Align [39] 76.16 34.60 75.52 43.14 30.69 27.20 25.50 19.91 79.99 72.43 43.53 48.06

###### ReVision (ours) 79.65 36.13 76.71 47.97 31.51 28.39 27.70 22.82 81.78 72.53 46.58 50.16

Table 2 Cost-performance comparison between paired image-text pretraining and text-only ReVision scaling.

General Reasoning Hallucination

Avg. ↑ Cost ↓

Method Text-only

MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench Paired image-text ✗ 73.59 35.40 76.01 44.18 34.80 27.70 25.90 23.27 80.87 69.13 47.11 48.91 1.00 Unicorn [35] ✓ 60.24 35.13 68.81 42.35 36.87 34.05 26.80 29.53 42.32 64.21 43.01 43.94 3.98 (+298%)

- ReVision-1M (ours) ✓ 72.20 34.33 75.84 43.72 30.22 27.64 25.70 21.03 79.59 71.93 46.37 48.05 0.37 (-63%)

- ReVision-2M (ours) ✓ 74.94 36.40 76.35 45.23 33.49 29.59 26.80 24.38 80.14 72.18 48.26 49.75 0.74 (-26%)

dataset for this comparison, aligning exactly with the SFT data scale used in Unicorn [35]. We define the Cost metric based on the expense of data synthesis using GPT5 APIs, normalizing the cost of 1M image-text pairs to 1.0. We compare four pre-training settings in the Table 2: 1 Paired image-text: 1M ground-truth image-text pairs (upper bound). 2 Unicorn: 1M unpaired text samples, text-only method using mean shift. 3

ReVision-1M: 1M unpaired text samples. 4 ReVision-2M: 2M unpaired text samples. Comparing the text-only methods, ReVision-2M (49.75) dramatically outperforms Unicorn (43.94). Since Unicorn relies on simple mean-shifting, its synthesized features lack the correct geometric shape, leading to poor manifold penetration. ReVision generates pseudo-features that statistically mimic real visual distributions, proving that how you align matters as much as what you align. Additionally, we observe that Unicorn achieves notably high scores on Reasoning benchmarks. This discrepancy stems from the SFT data distribution: the InternVL-Chat-V1-2-SFT used in our controlled experiments focuses primarily on general visual conversation and basic perception, whereas Unicorn’s original SFT dataset incorporates a massive volume of complex reasoning tasks. This aligns with our perspective that unlocking deep reasoning potential relies heavily on complex SFT tasks, suggesting that ReVision’s performance could be further elevated with more challenging instruction tuning data. Importantly, ReVision-2M (49.75) surpasses the w/. Image baseline (48.91) trained on 1M real paired samples. Comparing with paired data, ReVision-2M (49.75) outperforms the 1M paired baseline (48.91) at only 74% of the cost (0.74 vs. 1.00). This highlights a promising scaling trajectory: continuously scaling up low-cost text data can surpass the performance of expensive paired data with significantly lower overhead.

### 7 Conclusion

We address the Modality Gap by establishing the Fixed-frame Theory. Moving beyond isotropic assumptions, we decompose the gap into stable biases and anisotropic residuals, revealing it as a structured geometric phenomenon rather than random noise. Guided by these insights, we introduce ReAlign, a training-free statistical alignment strategy, and ReVision, a scalable paradigm leveraging unpaired text as a substitute for expensive image-text pairs. Experiments show ReVision significantly mitigates the gap and outperforms baselines, proving that high-quality visual structures can be learned from pure text. This work offers a robust, cost-effective pathway for scaling MLLMs.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [3] Junying Chen, Chi Gui, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy Chen, Xidong Wang,

- Zhenyang Cai, Ke Ji, Xiang Wan, et al. Towards injecting medical visual knowledge into multimodal llms at scale. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 7346–7370, 2024.
- [4] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024.

- [5] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.

- [6] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.

- [7] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024.

- [8] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.

- [9] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024.

- [10] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530, 2024.

- [11] Weiquan Huang, Aoqi Wu, Yifan Yang, Xufang Luo, Yuqing Yang, Liang Hu, Qi Dai, Chunyu Wang, Xiyang Dai, Dongdong Chen, et al. Llm2clip: Powerful language model unlocks richer visual representation. arXiv preprint arXiv:2411.04997, 2024.

- [12] Wei Li, Linchao Zhu, Longyin Wen, and Yi Yang. Decap: Decoding clip latents for zero-shot captioning via text-only training. arXiv preprint arXiv:2303.03032, 2023.

- [13] Xiaotong Li, Fan Zhang, Haiwen Diao, Yueze Wang, Xinlong Wang, and Lingyu Duan. Densefusion-1m: Merging vision experts for comprehensive multimodal perception. Advances in Neural Information Processing Systems, 37:18535–18556, 2024.

- [14] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.

- [15] Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. Advances in Neural Information Processing Systems, 35:17612–17625, 2022.

- [16] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

- [17] Yang Liu, Xiaomin Yu, Gongyu Zhang, Zhen Zhu, Christos Bergeles, Prokar Dasgupta, Alejandro Granados, and Sebastien Ourselin. Arcsin: Adaptive ranged cosine similarity injected noise for language-driven visual tasks. arXiv preprint arXiv:2402.17298, 2024.

- [18] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

- [19] David Nukrai, Ron Mokady, and Amir Globerson. Text-only training for image captioning using noise-injected clip. arXiv preprint arXiv:2211.00575, 2022.

- [20] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

- [21] Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier. Language models can see: Plugging visual controls in text generation. arXiv preprint arXiv:2205.02655, 2022.

- [22] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.

- [23] Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. Zerocap: Zero-shot image-to-text generation for visualsemantic arithmetic. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17918–17928, 2022.

- [24] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning, pages 9929–9939. PMLR, 2020.

- [25] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The all-seeing project v2: Towards general relation comprehension of the open world. In European Conference on Computer Vision, pages 471–490. Springer, 2024.

- [26] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

- [27] Kan Wu, Houwen Peng, Zhenghong Zhou, Bin Xiao, Mengchen Liu, Lu Yuan, Hong Xuan, Michael Valenzuela, Xi Stephen Chen, Xinggang Wang, et al. Tinyclip: Clip distillation via affinity mimicking and weight inheritance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21970–21980, 2023.

- [28] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024.

- [29] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

- [30] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025.

- [31] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [32] Can Yaras, Siyi Chen, Peng Wang, and Qing Qu. Explaining and mitigating the modality gap in contrastive multimodal learning. arXiv preprint arXiv:2412.07909, 2024.

- [33] Lingjie Yi, Raphael Douady, and Chao Chen. Decipher the modality gap in multimodal contrastive learning: From convergent representations to pairwise alignment. arXiv preprint arXiv:2510.03268, 2025.

- [34] Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, et al. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. arXiv preprint arXiv:2509.18154, 2025.

- [35] Xiaomin Yu, Pengxiang Ding, Wenjie Zhang, Siteng Huang, Songyang Gao, Chengwei Qin, Kejian Wu, Zhaoxin Fan, Ziyue Qiao, and Donglin Wang. Unicorn: Text-only data synthesis for vision language model training. arXiv preprint arXiv:2503.22655, 2025.

- [36] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

- [37] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134–15186, 2025.

- [38] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

- [39] Yuhui Zhang, Elaine Sui, and Serena Yeung-Levy. Connect, collapse, corrupt: Learning cross-modal tasks with uni-modal data. arXiv preprint arXiv:2401.08567, 2024.

### A Modality Gap Phenomenon: Structural Causes

This appendix explains why a modality gap can persist in the contrastive dual-encoder setting studied in Sec. 3. The key point is that the gap is not merely a consequence of noisy data or imperfect optimization. Rather, it is enabled by the interaction of three structural factors: dual-encoder isolation, spherical normalization, and dot-product contrastive supervision. Together, these factors create a weakly controlled channel in which the orthogonal component of the gap can survive late into training.

Recall. For paired embeddings ex(t),ey(t) ∈ Rd, the modality gap is defined as ∆(t) := ex(t) − ey(t), with mean gap m(t) := E[∆(t)]. Under the fixed reference decomposition Rd = U ⊕ V , Sec. 3.1 decomposes the gap as

∆(t) = β(t) + γ(t) + δ(t) + ζ(t),

where β(t) = PUm(t) is the principal mean bias in the dominant representation subspace, γ(t) = PV m(t) is the passive orthogonal bias in the orthogonal complement, and δ(t),ζ(t) are zero-mean residual components. Sec. 3.2 shows that γ(t) is not rapidly eliminated in late training, while Sec. 3.3 shows that the residual components have anisotropic second-order structure. Sec. 3.4 further shows that spherical normalization couples the mean anchor with the residual covariance, producing Phantom Drift.

Thesis. The persistence of the modality gap is explained by three structural causes. First, dual-encoder isolation creates independent modality anchors before alignment. Second, spherical normalization turns second-order residual anisotropy into centroid drift on the unit hypersphere. Third, the dot-product InfoNCE head restricts embedding-level gradients to the contrastive-set span, leaving directions outside the dominant representation subspace weakly corrected.

#### A.1 Cause 1: Dual-Encoder Isolation

Structural condition. The model uses two separate encoders for the two modalities. Each modality is processed independently until the final shared embedding space, producing unit-normalized embeddings ex(t) and ey(t).

Mechanism. Because the two encoders have different input structures, architectures, and inductive biases, their representation distributions need not have the same centroid. Thus, even when paired samples are semantically aligned, the population mean gap m(t) = E[ex(t) − ey(t)] is generally nonzero. Under the fixed reference frame, this mean mismatch decomposes into β(t) = PUm(t) and γ(t) = PV m(t). The component β(t) is the anchor displacement inside the dominant representation subspace, while γ(t) is the orthogonal mean displacement that can persist when direct optimization pressure along V is weak.

Implication. Dual-encoder isolation is therefore the structural source of the first-order anchor mismatch. If the modalities were fused before the final embedding layer, their intermediate features would be jointly processed and the notion of two independently formed modality anchors would be substantially weakened. In contrast, the dual-encoder design allows each modality distribution to develop its own centroid before contrastive alignment acts on the final embeddings.

#### A.2 Cause 2: Spherical Normalization

Structural condition. The encoder outputs are explicitly normalized by an L2 projection e = z/∥z∥, so all final embeddings lie on the unit hypersphere.

Mechanism. Spherical normalization is nonlinear, so Euclidean mean alignment does not imply spherical centroid alignment. Let a mean-aligned source representation be z = µ + ϵ, where E[ϵ] = 0 and Cov(ϵ) = Σ. Although E[z] = µ, the normalized centroid is

µπ = E

µ + ϵ ∥µ + ϵ∥

,

which generally differs from π(µ) = µ/∥µ∥. Thus the normalization-induced drift is

µ + ϵ ∥µ + ϵ∥

µ ∥µ∥

∆π = E

−

.

This is the Phantom Drift analyzed in Sec. 3.4. Its magnitude and direction depend on both the anchor direction µ and the residual covariance Σ. Therefore, the anisotropic residual structure identified in Sec. 3.3 is not geometrically harmless: after normalization, it can shift the spherical centroid even when the Euclidean mean has already been aligned.

Implication. Spherical normalization turns second-order residual shape into a first-order centroid effect on the hypersphere. Without this nonlinear projection, mean correction would remain a linear Euclidean operation. With normalization, however, residual anisotropy and anchor direction interact, producing an additional centroid mismatch that must be corrected separately.

#### A.3 Cause 3: Dot-Product Contrastive Supervision

Structural condition. The contrastive objective uses dot-product similarity between unit-normalized embeddings, as in InfoNCE. The loss compares each embedding only through its inner products with embeddings from the current contrastive set.

Mechanism. For a mini-batch Bt, let Ut(B) := span({ex,i(t)}i∈Bt ∪ {ey,i(t)}i∈Bt

) be the contrastive-set span. For the mini-batch InfoNCE loss, the embedding-level gradients satisfy gx,i(B)(t) ∈ span{ey,j(t) : j ∈ Bt}, gy,i(B)(t) ∈ span{ex,j(t) : j ∈ Bt}.

Hence, the gradients are constrained by the representation span of the current contrastive set. In late training, embeddings concentrate near the instantaneous dominant subspace Ut, so the main optimization signal is also concentrated near Ut. When we measure the leakage into the fixed orthogonal complement V = U⊥, any vector g ∈ Ut satisfies

∥PV g∥ ∥g∥

≤ ∥PV PU

t∥ = sinθ(Ut,U),

where θ(Ut,U) is the largest principal angle between the instantaneous subspace Ut and the fixed reference subspace U. Thus, a large portion of the observed V -direction leakage can be attributed to subspace rotation rather than to a strong corrective signal along V .

Implication. The dot-product InfoNCE head does not uniformly penalize mismatch in all ambient directions. Instead, it primarily produces gradients along directions spanned by the current contrastive representations. As a result, the orthogonal component γ(t) is only weakly corrected in late training. This explains why the POB behaves as a passive bias: it maintains a stable direction and drifts slowly rather than being rapidly optimized away.

#### A.4 Summary

The modality gap persists because standard contrastive dual-encoder training contains three interacting structural mechanisms. Dual-encoder isolation creates independent modality anchors, giving rise to a nonzero first-order gap m(t) = β(t) + γ(t). Spherical normalization couples this anchor with anisotropic residual covariance, producing Phantom Drift on the unit hypersphere. Dot-product InfoNCE supervision restricts embedding-level gradients to the contrastive-set span, leaving the orthogonal complement V weakly corrected. Therefore, the modality gap should not be viewed as a mere optimization failure or random noise artifact. It is a structural byproduct of the representation geometry induced by dual encoders, spherical embeddings, and contrastive dot-product training, and thus requires explicit geometric rectification.

### B Proof of Theoretical Claims

This appendix formalizes the mathematical claims used in Sec. 3 and Sec. 4. We separate deterministic identities from empirical conditions. In particular, low reference leakage, high effective condition number, and stable spectral profiles are empirical observations measured in the main text; the results below prove what these observations imply under the fixed-reference decomposition.

#### B.1 Preliminaries

- Definition B.1 (Orthogonal fixed reference frame). Let U ⊂ Rd be the dominant representation subspace estimated at the reference time t0, and let V := U⊥. Denote by PU and PV the orthogonal projectors onto U and V , respectively. Then PU + PV = I, PUPV = 0, and every vector a ∈ Rd admits the unique orthogonal decomposition a = PUa + PV a.
- Definition B.2 (Modality gap and residual components). For paired embeddings ex(t),ey(t) ∈ Rd, define the modality gap as ∆(t) := ex(t) − ey(t) and its mean as m(t) := E[∆(t)]. Under the fixed frame U ⊕ V , define

β(t) := PUm(t), γ(t) := PV m(t), δ(t) := PU(∆(t) − m(t)), ζ(t) := PV (∆(t) − m(t)).

#### B.2 Proof of the Bias--Residual Decomposition

- Lemma B.3 (Orthogonal bias–residual decomposition). For every t ≥ t0, the modality gap decomposes as ∆(t) = β(t) + γ(t) + δ(t) + ζ(t).

Moreover, E[δ(t)] = 0, E[ζ(t)] = 0, δ(t) ∈ U, ζ(t) ∈ V , and ⟨δ(t),ζ(t)⟩ = 0 almost surely.

Proof. Since PU + PV = I, we have

∆(t) = m(t) + (∆(t) − m(t)) = PUm(t) + PV m(t) + PU(∆(t) − m(t)) + PV (∆(t) − m(t)). By definition, this is exactly β(t) + γ(t) + δ(t) + ζ(t). Since PU and PV are linear,

E[δ(t)] = PUE[∆(t) − m(t)] = 0, E[ζ(t)] = PV E[∆(t) − m(t)] = 0.

The membership δ(t) ∈ U and ζ(t) ∈ V follows from the definitions of the projectors. Since U ⊥ V , the two residual components are orthogonal almost surely.

| |
|---|

- Proposition B.4 (Mean centering is necessary for pure residual covariance). Let ∆U(t) := PU∆(t) = β(t) + δ(t). Then

E ∆U(t)∆U(t)⊤ = β(t)β(t)⊤ + Cov(δ(t)). Thus, without removing β(t), the second moment in U mixes the first-order mean displacement with the second-order residual covariance. Proof. Since E[δ(t)] = 0, we have

E (β(t) + δ(t))(β(t) + δ(t))⊤ = β(t)β(t)⊤ + E[δ(t)δ(t)⊤]. Because δ(t) is zero-mean, E[δ(t)δ(t)⊤] = Cov(δ(t)). This proves the claim.

| |
|---|

- Proposition B.5 (Translation-correctability of the first-order anchor mismatch). At the Euclidean mean level, the first-order anchor mismatch m(t) = β(t) + γ(t) is removed by the translation ∆(t)  → ∆(t) − m(t). Equivalently, translating one modality by the difference of modality means aligns the two modality anchors.

Proof. Let ey be the source-modality embedding and ex the target-modality embedding, with means µy = E[ey] and µx = E[ex]. Define the translated source embedding e˙y = (ey − µy) + µx. Then

E[˙ey] = E[ey] − µy + µx = µx.

Thus the source anchor is translated to the target anchor. In terms of the gap, subtracting m(t) gives E[∆(t) − m(t)] = 0, so the first-order mean mismatch is removed.

#### B.3 Gradient Concentration and Passive Orthogonal Bias

- Lemma B.6 (InfoNCE gradient span constraint). Consider a mini-batch Bt with normalized embeddings

{ex,i(t)}i∈Bt

and {ey,i(t)}i∈Bt

, and let the image-to-text InfoNCE loss for sample i be

ℓxi →y = −log

exp(⟨ex,i,ey,i⟩/τ) j∈Bt exp(⟨ex,i,ey,j⟩/τ)

.

Then ∇ex,i

ℓxi →y ∈ span{ey,j : j ∈ Bt}. Similarly, for the text-to-image loss, the gradient with respect to ey,i lies in span{ex,j : j ∈ Bt}. Consequently, the embedding-level gradients of the symmetric InfoNCE loss are constrained by the contrastive-set span.

Proof. Let

pij =

exp(⟨ex,i,ey,j⟩/τ) k∈Bt exp(⟨ex,i,ey,k⟩/τ)

.

Differentiating ℓxi →y with respect to ex,i gives

∇ex,i

ℓxi →y =

1 τ

 

j∈Bt

pijey,j − ey,i

 ,

which is a linear combination of {ey,j : j ∈ Bt}. The text-to-image direction is symmetric. For the symmetric InfoNCE objective, gradients are sums of such terms, and hence remain within the corresponding contrastive-set span.

| |
|---|

- Lemma B.7 (Geometric leakage bound). Let Ut be an instantaneous r-dimensional subspace and U the fixed r-dimensional reference subspace, with V = U⊥. Let θ(Ut,U) be their largest principal angle. For any nonzero g ∈ Ut,

∥PV g∥ ∥g∥

≤ ∥PV PU

t∥2 = sinθ(Ut,U).

Proof. Since g ∈ Ut, we have PU

g = g. Therefore, ∥PV g∥ = ∥PV PU

t

g∥ ≤ ∥PV PU

t∥2∥g∥. Dividing by ∥g∥ gives the inequality. The equality ∥PV PU

t

t∥2 = sinθ(Ut,U) is the standard relation between orthogonal projection residuals and the largest principal angle between two subspaces.

| |
|---|

Proposition B.8 (Weak V -direction updates imply slow POB drift). Suppose the orthogonal mean bias evolves as γ(t + 1) = γ(t) + η(t), where η(t) ∈ V is the effective update along V . If ∥η(t)∥ ≤ εt, then over a window T = {t0,...,t0 + τ − 1},

k−1

∥γ(t0 + k) − γ(t0)∥ ≤

εt

0+j.

j=0

Moreover, if ∥η(t)∥ ≤ α∥γ(t)∥ with 0 < α < 1, then the adjacent-step direction changes are small, in the sense that

2α2 (1 − α)2

cos(γ(t + 1),γ(t)) ≥ 1 −

.

Proof. The drift bound follows by telescoping:

k−1

γ(t0 + k) − γ(t0) =

η(t0 + j),

j=0

and applying the triangle inequality. For the cosine bound, let a = γ(t) and b = a+η(t). Since ∥η(t)∥ ≤ α∥a∥, we have ∥b∥ ≥ (1 − α)∥a∥. The normalized direction perturbation satisfies

2∥η(t)∥ ∥b∥

b ∥b∥

a ∥a∥

2α 1 − α

≤

−

≤

.

Using ∥u − v∥2 = 2(1 − ⟨u,v⟩) for unit vectors u,v, we obtain

2

2α2 (1 − α)2

- 1

- 2

2α 1 − α

1 − cos(a,b) ≤

.

=

This proves the claim.

| |
|---|

#### B.4 Residual Covariance and Anisotropic Shape

- Lemma B.9 (Step-wise covariance avoids temporal drift mixing). Let Xs = µs + Rs for s = 0,...,τ − 1, where E[Rs] = 0 and Cov(Rs) = Σs. If S is uniformly sampled from {0,...,τ − 1}, then the covariance of the temporally pooled variable XS satisfies

Cov(XS) =

τ−1

1 τ

Σs + Cov(µS).

s=0

Thus, directly pooling samples across time includes an additional covariance term induced by the drift of the time-varying mean.

Proof. By the law of total covariance,

Cov(XS) = ES[Cov(XS | S)] + CovS(E[XS | S]). Since Cov(XS | S = s) = Σs and E[XS | S = s] = µs, the identity follows.

| |
|---|

Proposition B.10 (Residual anisotropy is a covariance-shape property). Let R be a zero-mean residual supported on a subspace W with covariance ΣW. If ΣW = σ2PW for some σ2 ≥ 0, then R is isotropic in W. Conversely, if the spectrum of ΣW is non-uniform, then the residual has an anisotropic ellipsoidal shape in W. In particular, any spectral stretching measure κeff that equals 1 on trace-matched isotropic covariances and exceeds 1 on non-uniform spectra certifies anisotropy when κeff(ΣW) > 1.

Proof. For a zero-mean random vector, the covariance determines the second-order shape of its distribution. If ΣW = σ2PW, then all directions in W have the same variance:

Var(⟨u,R⟩) = u⊤ΣWu = σ2

for every unit vector u ∈ W. Hence the residual is second-order isotropic in W. If the eigenvalues of ΣW are not all equal, then there exist two unit eigen-directions u,v ∈ W with different variances, so the residual energy is direction-dependent. The covariance ellipsoid is therefore elongated along higher-variance directions, which is exactly second-order anisotropy.

| |
|---|

- Lemma B.11 (Mean direction and residual principal direction are distinct objects). Let γ ∈ V be the mean

bias direction and let v1 be the first principal direction of a residual covariance ΣV . If |⟨γ/∥γ∥,v1⟩| is small, then the first-order mean displacement and the leading second-order residual spread are geometrically decoupled in V .

Proof. The vector γ describes the first moment, i.e., the centroid displacement in V . The vector v1 is an eigenvector of ΣV associated with its largest eigenvalue, and therefore describes the direction of maximum residual variance. If these directions are nearly orthogonal, then the direction of the mean displacement does not coincide with the direction of largest residual energy. Hence the first-order and second-order structures are geometrically distinct.

#### B.5 Phantom Drift Induced by Spherical Normalization

Definition B.12 (Spherical projection and spherical centroid). For z ∈ Rd \ {0}, define the spherical projection π(z) := z/∥z∥. For a random vector Z, define its spherical centroid as E[π(Z)], whenever the expectation exists.

- Theorem B.13 (Euclidean mean alignment does not imply spherical centroid alignment). Let Z = µ + ϵ, where µ ̸= 0, E[ϵ] = 0, and Cov(ϵ) = Σ. Even though E[Z] = µ, in general

E[π(Z)] ̸= π(µ). Therefore, Euclidean mean alignment does not guarantee centroid alignment after spherical normalization. Proof. The map π(z) = z/∥z∥ is nonlinear. Therefore, expectation and projection do not commute in general: E[π(Z)] ̸= π(E[Z]).

Since E[Z] = µ, the right-hand side is π(µ). A concrete example suffices. Let d = 2, µ = (1,0)⊤, and let ϵ take values (0,a)⊤ and (0,−a)⊤ with probability 1/2, where a > 0. Then E[ϵ] = 0, but

E[π(µ + ϵ)] =

1 √1 + a2

,0

⊤

̸= (1,0)⊤ = π(µ).

Thus spherical centroid alignment is not guaranteed by Euclidean mean alignment.

| |
|---|

- Theorem B.14 (Second-order expansion of Phantom Drift). Let Z = µ + ϵ, where µ ≠ 0, E[ϵ] = 0, and Cov(ϵ) = Σ. Let r := ∥µ∥ and a := µ/r. For sufficiently small residual magnitude, the normalization-induced centroid drift

∆π := E[π(µ + ϵ)] − π(µ) admits the second-order approximation

1 r2

- 1

- 2r2

3 2r2

a(a⊤Σa) + O(E∥ϵ∥3/r3). In particular, the tangential component of the drift is

∆π = −

Σa −

a tr(Σ) +

1 r2

(I − aa⊤)Σa + O(E∥ϵ∥3/r3). Thus anisotropic residual covariance can shift the spherical centroid away from the anchor direction. Proof. Let f(z) = z/∥z∥. The first derivative at µ is

(I − aa⊤)∆π = −

1 r

(I − aa⊤)h.

Dfµ[h] =

Since E[ϵ] = 0, the first-order term vanishes after taking expectation. The second derivative satisfies

2 r3

1 r3

3 r5

µ(µ⊤h)2. Applying Taylor expansion around µ and taking expectation gives

h(µ⊤h) −

D2fµ[h,h] = −

µ∥h∥2 +

- 1

- 2

E[D2fµ[ϵ,ϵ]] + O(E∥ϵ∥3/r3). Using E[ϵ(µ⊤ϵ)] = Σµ, E∥ϵ∥2 = tr(Σ), and E[(µ⊤ϵ)2] = µ⊤Σµ, we obtain

E[f(µ + ϵ)] = f(µ) +

1 r3

- 1

- 2r3

3 2r5

µ(µ⊤Σµ) + O(E∥ϵ∥3/r3).

∆π = −

Σµ −

µ tr(Σ) +

Substituting µ = ra yields the claimed expression. Finally, projecting onto the tangent space a⊥ removes the terms parallel to a, leaving

This proves the result.

1 r2

(I − aa⊤)∆π = −

(I − aa⊤)Σa + O(E∥ϵ∥3/r3).

| |
|---|

Corollary B.15 (Role of anisotropy). If Σ = σ2I, then the leading-order tangential drift vanishes. If (I − aa⊤)Σa ≠ 0, then the leading-order tangential drift is nonzero. Therefore, anisotropy that is not aligned with the anchor direction induces angular centroid drift on the unit hypersphere.

Proof. Substituting Σ = σ2I into the tangential term gives (I − aa⊤)Σa = σ2(I − aa⊤)a = 0. Conversely, if (I − aa⊤)Σa ̸= 0, then the leading-order tangential term in the previous theorem is nonzero.

| |
|---|

#### B.6 Properties of the ReAlign Operator

- Proposition B.16 (Anchor Alignment matches the target mean). Let e˙y = (ey − µy) + µx, where µy = E[ey] and µx = E[ex]. Then E[˙ey] = µx. Proof. By linearity of expectation,

E[˙ey] = E[ey] − µy + µx = µx.

| |
|---|

- Proposition B.17 (Trace Alignment matches residual energy and preserves spectral shape). Let e˜y = µx + s(ey − µy), where s = Tx/(Ty + ε), Ty = tr(Σy), and Tx = tr(Σx). Then

Cov(˜ey) = s2Σy, tr(Cov(˜ey)) = s2Ty ≈ Tx. Moreover, scalar scaling preserves the eigenvectors and the trace-normalized spectrum of Σy. Proof. Since e˜y − µx = s(ey − µy), we have

Cov(˜ey) = s2 Cov(ey − µy) = s2Σy.

Taking traces gives tr(Cov(˜ey)) = s2Ty, which equals Tx up to the stabilizing constant ε. If Σy = QΛQ⊤, then s2Σy = Q(s2Λ)Q⊤. Thus eigenvectors are unchanged, and

λ(s2Σy) tr(s2Σy)

=

s2λ(Σy) s2 tr(Σy)

=

λ(Σy) tr(Σy)

.

Therefore Trace Alignment changes only the global residual energy scale while preserving the trace-normalized spectral shape.

| |
|---|

- Proposition B.18 (Centroid Alignment removes the measured spherical centroid shift before the final projection). Let e′y = π(˜ey), µ′y = E[e′y], and e′′y = e′y − µ′y + µx. Then E[e′′y] = µx. Proof. Again by linearity of expectation,

E[e′′y] = E[e′y] − µ′y + µx = µx.

| |
|---|

- Proposition B.19 (Residual centroid error after the final projection is controlled by the second translation). Let e′y be unit-normalized and let h := µx − µ′y. Define e′′y = e′y + h and eˆy = π(e′′y). If ∥h∥ < 1, then

∥E[ˆey] − µx∥ ≤ ∥h∥. Proof. For any unit vector q and any h with ∥h∥ < 1, we have

∥π(q + h) − (q + h)∥ = |1 − ∥q + h∥| ≤ ∥h∥, where the last inequality follows from the reverse triangle inequality. Taking q = e′y and expectation gives

∥E[ˆey] − E[e′′y]∥ ≤ E∥eˆy − e′′y∥ ≤ ∥h∥. Since E[e′′y] = µx, the result follows.

| |
|---|

#### B.7 Summary of Theoretical Implications

The results above justify the three geometric levels used in the main text. First, the modality gap admits an orthogonal decomposition into first-order anchor displacement and zero-mean residual components. Second, after mean centering, the remaining residual shape is governed by its covariance spectrum, so anisotropy is a second-order property rather than a mean effect. Third, spherical normalization does not commute with expectation; it couples the anchor direction with the residual covariance and can create a secondary centroid shift. These facts motivate ReAlign: Anchor Alignment removes the Euclidean mean mismatch, Trace Alignment matches the global residual energy while preserving spectral shape, and Centroid Alignment corrects the measured spherical centroid drift.

### C U--V Weak Coupling Analysis

This appendix provides a diagnostic analysis of weak coupling between the dominant representation subspace U and its orthogonal complement V . In Sec. 3.2, the reference leakage ratio leakref(t) is compared with the geometric baseline sinθ(Ut,U), where Ut is the instantaneous dominant subspace and U is the fixed reference subspace. If the embedding-level gradients were fully constrained to Ut, then their projection onto the fixed complement V = U⊥ would be controlled by this principal-angle baseline. In practice, small deviations from this idealized geometric picture may appear. We model such deviations as weak second-order coupling between U-side and V -side residuals.

#### C.1 Residual-Level Coupling Model

Let δ(t) ∈ U and ζ(t) ∈ V denote the zero-mean residual components defined in Sec. 3.1. To quantify whether fluctuations in U systematically explain fluctuations in V , we introduce the following linear residual-level model:

ζ(t) = Lδ(t) + ξ(t), L : U → V, ∥L∥2 ≤ εUV . (4) Here, L is a weak coupling map from U to V , and ξ(t) ∈ V is a zero-mean background residual that is uncorrelated with δ(t). In reference-basis coordinates, δ(t) ∈ Rr, ζ(t) ∈ Rd−r, and L ∈ R(d−r)×r.

- Lemma C.1 (Second-moment identities under weak coupling). Assume E[δ(t)] = 0, E[ξ(t)] = 0, and E[ξ(t)δ(t)⊤] = 0. Let ΣU := Cov(δ(t)) and Σξ := Cov(ξ(t)). Under Eq. (4),

E[ζ(t)δ(t)⊤] = LΣU, Cov(ζ(t)) = LΣUL⊤ + Σξ. (5)

Equivalently, E[δ(t)ζ(t)⊤] = ΣUL⊤. Proof. Using ζ = Lδ + ξ, we obtain

E[ζδ⊤] = E[(Lδ + ξ)δ⊤] = LE[δδ⊤] + E[ξδ⊤] = LΣU. Similarly,

Cov(ζ) = E[(Lδ + ξ)(Lδ + ξ)⊤] = LΣUL⊤ + Σξ, where the cross terms vanish because ξ and δ are uncorrelated. This proves the claim.

| |
|---|

Interpretation. The map L does not replace the main U ⊕V decomposition. Rather, it measures whether the two residual components are statistically independent at the second-moment level. If L = 0, then the V -side residual contains no linear component predictable from the U-side residual. If ∥L∥2 is small but nonzero, then the two residuals are weakly coupled.

#### C.2 Estimating the Coupling Map

Lemma C.1 shows that L is identifiable from second moments when ΣU is nonsingular on its effective support. The population least-squares map is

L⋆ = E[ζ(t)δ(t)⊤]Σ†U, (6) where Σ†U denotes the Moore–Penrose pseudoinverse. In practice, we estimate L in the reference bases of U and V . Given probe residuals {(δi,ζi)}ni=1, let D := [δ1,...,δn] ∈ Rr×n and Z := [ζ1,...,ζn] ∈ R(d−r)×n. We use the ridge estimator

Lλ = ZD⊤ DD⊤ + λIr −1 . (7) We report the spectral norm ∥ Lλ∥2 as the coupling strength and the explained variance

∥Z − LλD∥2F ∥Z∥2F

Rζ2←δ = 1 −

. (8)

A small Rζ2←δ indicates that the V -side residual is largely independent of the U-side residual, while a nonzero value indicates weak predictable leakage from U to V .

#### C.3 Connection to Gradient Leakage

The coupling model above is defined at the residual level. To connect it to the gradient leakage measured in Sec. 3.2, we state an explicit diagnostic assumption rather than treating it as a consequence of the residual model.

Gradient-level coupling assumption. For an embedding-level gradient g, suppose its fixed-frame V -component admits the approximate decomposition

g + LPUg + rg, ∥rg∥ ≤ ρ∥g∥. (9)

PV g = PV PU

t

The first term is the geometric leakage caused by the rotation between Ut and U; the second term is the weak coupling contribution; and rg is an unexplained remainder. When ρ = 0, the gradient leakage is fully explained by subspace rotation and the coupling map.

- Lemma C.2 (Geometric leakage baseline). For any nonzero g ∈ Ut,

∥PV g∥ ∥g∥

t∥2 = sinθ(Ut,U), (10) where θ(Ut,U) is the largest principal angle between Ut and U. Proof. Since g ∈ Ut, PU

≤ ∥PV PU

g = g. Therefore, ∥PV g∥ = ∥PV PU

t

t∥2∥g∥. Dividing by ∥g∥ gives the inequality. The identity ∥PV PU

g∥ ≤ ∥PV PU

t

t∥2 = sinθ(Ut,U) follows from the standard relation between projection residuals and principal angles.

| |
|---|

Theorem C.3 (Gradient leakage under weak U–V coupling). Under the gradient-level coupling assumption in Eq. (9), the leakage ratio satisfies

∥PV g∥ ∥g∥

≤ sinθ(Ut,U) + ∥L∥2 + ρ. (11)

In the idealized case ρ = 0, the leakage is bounded by the sum of the geometric subspace-rotation baseline and the weak coupling strength.

Proof. By Eq. (9) and the triangle inequality, ∥PV g∥ ≤ ∥PV PU

g∥ + ∥LPUg∥ + ∥rg∥. The first term is bounded by Lemma C.2: ∥PV PU

t

g∥ ≤ sinθ(Ut,U)∥g∥. For the coupling term, ∥LPUg∥ ≤ ∥L∥2∥PUg∥ ≤ ∥L∥2∥g∥. The remainder satisfies ∥rg∥ ≤ ρ∥g∥. Combining these inequalities and dividing by ∥g∥ proves the result.

t

| |
|---|

#### C.4 Implication

The weak coupling analysis should be interpreted as a diagnostic refinement of the main geometric picture. The dominant explanation for reference leakage is still the rotation between the instantaneous subspace Ut and the fixed reference subspace U, captured by sinθ(Ut,U). The coupling map L measures whether there is an additional linear pathway through which U-side residual fluctuations predict V -side residual fluctuations. When ∥ Lλ∥2 and Rζ2←δ are small, the U- and V -side residuals are weakly coupled, supporting the interpretation that the orthogonal bias γ(t) evolves passively rather than being aggressively corrected by direct V -direction optimization.

### D Beyond the Isotropic Assumption: A Geometric Analysis

In this section, we provide a comprehensive empirical verification of the alignment quality. While the Modality Gap metric measures the Euclidean distance between centroids, it fails to capture the structural fidelity of the aligned distribution. To address this, we examine the geometric properties from three perspectives.

#### D.1 Spectral Analysis: Preserving Semantic Hierarchy

The eigenspectrum of the feature covariance matrix characterizes the intrinsic geometry of the data manifold. Deep representations typically exhibit a power-law decay (λk ∝ k−α), indicating a rich semantic hierarchy where variance concentrates in principal directions.

To ensure a fair comparison, we normalize the global trace of all baselines to match the target modality before computing the spectrum. In Fig. 4(a), we plot the log-scaled singular values and quantify the geometric structure using the Power-Law Exponent (α).

Observation. 1 C3 exhibits a significantly elevated tail and a flatter slope, with an α value of approximately 1.06, compared to α ≈ 1.33 for the original text. This confirms that injecting unstructured Gaussian noise reduces spectral anisotropy, producing a whitening effect that effectively dilutes the fine-grained semantic structure. 2 ReAlign. In contrast, ReAlign maintains a decay rate (α ≈ 1.33) that perfectly matches the intrinsic geometry of the source text. While the visual modality naturally possesses a lower rank (α ≈ 1.83), ReAlign achieves alignment without artificially distorting the text’s inherent semantic hierarchy.

#### D.2 Angular Distribution: Matching Cosine Topology

We verify the angular alignment by computing the Kernel Density Estimate (KDE) of pairwise cosine similarities. We further quantify the distributional discrepancy using the Jensen-Shannon (JS) Divergence.

Observation: As shown in Fig. 4(b), the angular topology provides a rigorous test of structural preservation. 1 C3. The noise injection results in a severe distributional shift, destroying the semantic relations on the

[Figure 5]

[Figure 6]

(a) Semantic Hierarchy (b) Angular Topology Matching

- Figure 4 Geometric Fidelity Analysis via Spectral and Angular Properties. (a) Semantic Hierarchy: The eigenspectrum analysis reveals that C3 (red line) exhibits a flattened slope with an elevated tail (α ≈ 1.06), indicating that unstructured noise injection dilutes fine-grained semantic structure. In contrast, ReAlign (blue line) maintains a power-law decay (α ≈ 1.33) that matches the intrinsic geometry of the source text. (b) Angular Topology Matching: KDE plots of cosine similarities demonstrate that C3 causes a severe distributional shift (JS Divergence = 0.1924), destroying angular relationships. ReAlign achieves a near-perfect overlap with the target prior (JS Divergence = 0.0066), validating its ability to restore centroid alignment while preserving the topological structure.

(a) Original Text (b) C3 (c) ReAlign

[Figure 7]

Mixing Rate 0.32% Mixing Rate 1.31% Mixing Rate 4.35%

- Figure 5 Global Alignment Visualization via PCA. We visualize the manifold alignment across three settings: (a) Original Text forms a distinct cluster separated from the image modality with negligible mixing (0.32%). (b) C3 expands the text distribution via noise but fails to effectively penetrate the visual manifold (1.31%). (c) ReAlign successfully shifts the text distribution into the visual support region, achieving a mixing rate of 4.35%. This represents a relative improvement of over 3× compared to the C3 baseline, confirming significant manifold penetration.

hypersphere. Quantitative results show a high JS Divergence of 0.1904. 2 ReAlign. Our method achieves a JS Divergence as low as 0.0067, effectively overlapping with the target prior. This demonstrates that ReAlign restores centroid alignment while preserving the correct angular relationships between samples.

#### D.3 Representation Visualization

We use Principal Component Analysis (PCA) for a qualitative visualization of the global alignment. As shown in Fig. 5. 1 Qualitative Analysis: The Original Text (Left) forms a cluster that is distinct and separated from the image modality. The C3 baseline (Middle) expands the text distribution via noise but fails to penetrate the visual manifold. ReAlign (Right) effectively shifts the text distribution into the visual support region. 2

Quantitative Analysis: We validate this mixing in the high-dimensional space (D = 1280) using the k-NN Mixing Rate with k = 20. The unaligned text shows negligible mixing, at only 0.32%. While C3 slightly improves this to 1.31%, ReAlign achieves a mixing rate of 4.35%. This represents a relative improvement of over 3× compared to the strong C3 baseline, and an improvement of over 10× compared to the unaligned state, confirming significant manifold penetration.

### E Robustness Analysis

In this section, we systematically evaluate the stability and scalability of ReAlign along three critical dimensions:

1 sample complexity for estimating alignment statistics, 2 numerical stability and computational efficiency, and 3 sensitivity to domain shifts.

[Figure 8]

#### E.1 Data Efficiency & Stability

A distinct advantage of ReAlign is its computational efficiency: it relies solely on low-order moments derived from unpaired data, avoiding the computational burden of iterative optimization. Here, we quantify the minimum sample size required for stable alignment.

We randomly sample N unpaired images and texts from the pretraining corpus, with N ∈ {102,5 × 102,103,5×103,104,5×104,105,5×105}. For each N, we estimate the ReAlign population means µ and second-order covariance shapes on these subsets. To ensure numerical stability in high dimensions, we apply standard shrinkage regularization to the covariance estimates. We align the embeddings using these estimates and measure the Modality Gap on a held-out test set. We report the mean ± standard deviation over K = 5 independent trials.

Figure 6 Impact of unpaired sample size on the modality gap for statistical estimation.

As illustrated in Fig. 6, the modality gap decreases rapidly with N and stabilizes once N exceeds approximately 10,000 samples. 1 Rapid Convergence: Unlike neural network training, which requires millions of gradient steps, moment-based estimation converges quickly via the Law of Large Numbers. Empirically, the performance plateaus after a moderate sample size (N ≈ 104). 2 Implication: This confirms that ReAlign can be calibrated to new distributions on the fly with negligible computational cost, eliminating the need for massive paired datasets.

- E.2 Numerical Stability & Complexity We conduct empirical experiments to validate the robustness and efficiency of our implementation.

Accumulating millions of high-dimensional vectors is prone to significant round-off errors. To quantify this, we compared the alignment residual ∥µerr∥2 (the distance between the computed mean and the true ground truth) using single-precision (Float32) versus double-precision (Float64) accumulators. As shown in Fig. 7(a), utilizing Float32 introduces an irreducible alignment error floor. Specifically, at N = 500,000 samples, the error reaches ≈ 9.56 × 10−8, which is non-negligible for precise manifold alignment. In contrast, our Float64 implementation maintains precision near machine epsilon (≈ 10−15), ensuring that the geometric alignment is limited only by statistical properties rather than numerical instability.

We analyze the time and space complexity of the ReAlign algorithm by varying the dataset size N from 100k to 1M samples. 1 Time Complexity: As illustrated in Fig. 7(b), the processing time scales strictly linearly with data size (O(N)), increasing from 0.05s for 100k samples to 0.57s for 1M samples. 2 Space Complexity: Crucially, the peak memory usage remains constant at approximately 48.95 MB, regardless of the dataset size (O(1)). This confirms that ReAlign is highly scalable and capable of processing billions of tokens on standard hardware without memory bottlenecks.

#### E.3 Domain Mismatch Sensitivity

[Figure 9]

We believe that applying general-domain statistics to specialized domains violates the specific distributional reality, leading to geometric misalignment. We quantify this effect using a crossdomain statistics transfer protocol.

25

Figure 8 Comparison of modality gap performance under in-domain and crossdomain statistical alignment for General and Medical domains.

[Figure 10]

[Figure 11]

(a) Numerical Stability (b) Time & RAM

Figure 7 (a) Comparison of alignment residuals using Float32 vs. Float64 accumulators. (b) Trends of processing time (O(N)) and peak memory usage (O(1)) across dataset sizes.

We utilize two distinct distributions: 1 General: BunnyPretrain. 2 Medical: PubMedVision-Pretrain [3], representing a specialized scientific domain.

Protocol. Let Dstat denote the source domain for statistical estimation, and Deval denote the target domain for evaluation. We measure the modality gap on held-out data from Deval under four settings: 1 General → General: In-domain baseline. 2 General → Medical: Cross-domain transfer. 3 Medical → Medical: In-domain alignment. 4 Medical → General: Reverse transfer.

Fig. 8 shows the results. 1 Transfer Degradation: Using General statistics to align Medical data results in a substantially larger modality gap compared to the in-domain baseline. 2 Specificity Requirement: Conversely, applying Medical statistics to General data also degrades performance. These results indicate that the covariance structure of the modality gap is domain-dependent. Therefore, domain-specific calibration is strictly necessary for precise geometric alignment.

### F Failure Strategy Analysis: Blockwise Covariance Alignment

We present a training-free procedure that maps embeddings from one modality into the low-order statistical structure of another, using only first- and second-order moments estimated from large unpaired samples. All operations are performed in the fixed reference frame of Sec. 3.1, where the embedding space decomposes as Rd = U ⊕ V with fixed projectors PU,PV . The procedure consists of three closed-form steps: 1 Anchor align to remove first-order bias effects; 2 Geometry align on U and V via whitening–coloring; and 3 Centroid align to rectify centroid shifts induced by the final spherical projection.

Throughout, we use:

z ∥z∥2

, (12)

normalize(z) =

and treat expectations/covariances as population quantities approximated by empirical statistics from large unpaired datasets.

#### F.1 Step 1: Anchor Align

We first align first-order statistics by translating each modality representation toward a shared anchor µ⋆ in the shared representation space and then projecting back to the unit sphere. Let ea,eb ∈ Rd be unit-normalized

embeddings for two modalities a and b, with population means:

µa := E[ea], µb := E[eb]. (13) In the fixed frame (U,V ), these means decompose into in-subspace (PMB) and orthogonal (COB) components. To map modality a into the embedding distribution of modality b, we choose the anchor µ⋆ = µb and define

e˜a = normalize(ea − µa + µb), e˜b = eb. (14)

In this step, we translate ea so that its mean matches µb, then re-project onto the unit sphere. The subsequent Step 2 then matches both modalities to a shared target geometry (Σ⋆U,Σ⋆V ).

#### F.2 Step 2: Geometry Align

Next, we match second-order statistics on U and V separately. All linear transforms are applied in Euclidean space, and the output is then projected back to the unit sphere. After Step 1, we decompose anchored embeddings using the fixed projectors PU,PV and estimate blockwise covariances:

ΣU,a = Cov(PUe˜a), ΣV,a = Cov(PV e˜a). (15)

Analogously, define ΣU,b and ΣV,b for modality b. We set:

Σ⋆U = ΣU,b, Σ⋆V = ΣV,b, (16) so that modality a is mapped to match the covariance structure of modality b on both U and V . For symmetric substitution, (Σ⋆U,Σ⋆V ) can be chosen as any shared target geometry. Define the blockwise whitening–coloring transforms:

TU,a = (Σ⋆U)1/2 Σ−U,a1/2, TV,a = (Σ⋆V )1/2 Σ−V,a1/2. (17) Using these, construct the substituted embedding by applying the two transforms on their respective blocks:

eˆa = normalize TU,aPUe˜a + TV,aPV e˜a . (18)

#### F.3 Step 3: Centroid Align

Although Step 2 aligns the covariance structure, the final projection onto the unit sphere creates a non-linear distortion that inevitably shifts the population centroid away from the target anchor µ⋆. We term this phenomenon phantom drift. To strictly maintain first-order alignment, we explicitly estimate this drift and perform a final re-centering step.

Let µˆa = E[ˆea] be the drifted mean of the geometrically aligned embeddings from Step 2. We apply a final correction to pull the distribution back to the target anchor:

efinal = normalize(ˆea − µˆa + µ⋆). (19)

This ensures that the final substituted embeddings are both geometrically aligned in terms of covariance and accurately centered around the target semantic anchor.

#### F.4 Failure Analysis: Why Fine-Grained Alignment Fails?

Although the Blockwise Covariance Alignment strategy theoretically offers a stricter geometric match by aligning full second-order moments within subspaces, empirical results indicate that it consistently underperforms the proposed ReAlign method across all benchmarks. We attribute this failure to two primary factors, supported by the quantitative analysis in Fig. 9.

Estimation Variance and Numerical Instability. The ReAlign method relies on matching the global trace T = tr(Σ). Scalars are statistically robust and converge rapidly. In contrast, Blockwise Alignment requires

[Figure 12]

[Figure 13]

(a) Spectrum Decay (b) Semantic Preservation

- Figure 9 Failure mechanism analysis of Blockwise Covariance Alignment. (a) Spectrum Decay: The high condition number (≈ 1.1×103) of text embeddings induces numerical instability during covariance inversion, amplifying tail noise. In contrast, ReAlign maintains stability via isotropic scaling. (b) Semantic Preservation: KNN neighborhood overlap rates reveal that Blockwise’s aggressive whitening causes a collapse of the local semantic topology (retaining only 10% overlap), whereas ReAlign effectively preserves 87% of the semantic structure.

- Table 3 Performance comparison between BC Align and ReAlign. The results demonstrate that ReVision significantly outperforms BC Align across all benchmarks. This confirms that the numerical instability and excessive distortion of the semantic manifold in BC Align lead to degraded model performance.

General Reasoning Hallucination

Avg.

Method

MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench BC Align 76.12 34.33 76.42 45.36 30.69 27.95 24.40 22.60 80.93 71.08 46.27 48.74

##### ReVision 79.65 36.13 76.71 47.97 31.51 28.39 27.70 22.82 81.78 72.53 46.58 50.16

estimating and inverting full covariance matrices. As illustrated in Fig. 9(a), the empirical covariance spectrum of the text embeddings exhibits a rapid decay with a high condition number (κ ≈ 1.10 × 103). Computing the whitening transform Σ−1/2 on such an ill-conditioned matrix inadvertently amplifies noise in the tail dimensions (where eigenvalues λ → 0) by orders of magnitude. This leads to numerical instability and exploding updates in minor feature directions, whereas ReAlign’s isotropic scaling remains stable regardless of the spectral shape.

Semantic Distortion via Aggressive Whitening. Geometric alignment operates on the assumption that the source and target manifolds share a topologically isomorphic structure. ReAlign adopts a conservative approach, isotropic scaling via e˜y = µx + s(ey − µy)., which preserves the relative angular distances between source embeddings. Blockwise whitening, however, applies an anisotropic rotation and scaling to force the covariance shapes to match exactly.

We quantified the impact of this transformation on semantic structure using k-Nearest Neighbor (KNN, k = 10) overlap rates. As shown in Fig. 9(b), this over-alignment causes a catastrophic collapse of the local semantic topology: Blockwise alignment retains only 10.1% of the original semantic neighborhoods. In sharp contrast, ReAlign preserves 87.3% of the local structure, ensuring that fine-grained semantic relationships remain intact for the LLM.

While Blockwise Covariance Alignment is geometrically more rigorous, ReAlign strikes a superior trade-off between geometric compatibility and semantic preservation. Its reliance on robust first-order and scalar second-order statistics provides a stable initialization that is crucial for scalable MLLM pretraining.

### G The Long-Caption Paradox

Intuitively, utilizing longer, denser captions should theoretically provide richer semantic supervision, thereby enhancing cross-modal alignment. This expectation is particularly strong when employing advanced text

- Table 4 Performance Comparison validating the Long-Caption Paradox. Quantitative results demonstrate that ReVision consistently outperforms ReVision-Long across General, Reasoning, and Hallucination benchmarks. This performance gap confirms that for statistical modality alignment, the geometric compactness and high signal-to-noise ratio of short captions are more critical than the raw length of the textual description.

General Reasoning Hallucination

Avg. MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

Method

ReVision-Long 75.74 33.40 74.84 46.41 30.57 27.26 24.70 25.28 80.42 71.47 45.95 48.73 ReVision 79.65 36.13 76.71 47.97 31.51 28.39 27.70 22.82 81.78 72.53 46.58 50.16

[Figure 14]

[Figure 15]

[Figure 16]

(a) Effective Rank (b) Trace (c) Initial Modality Gap

- Figure 10 Geometric Analysis of the Long-Caption Paradox. (a) Effective Rank: Measurements reveal that Long captions exhibit a high effective rank (≈ 52.9) similar to the visual modality (≈ 57.5). This indicates a diffuse, high-entropy covariance structure that is difficult to align. In contrast, Short captions (≈ 41.0) function as a compact, low-rank approximation of the visual content, offering greater statistical stability. (c) Initial Modality Gap: The inclusion of non-visual linguistic noise in long captions acts as a disturbing force, significantly widening the initial modality gap (∥∆µ∥ ≈ 0.51) by approximately 30% compared to concise captions (∥∆µ∥ ≈ 0.39).

encoders like LLM2CLIP, which are explicitly engineered to handle long contexts.

However, our empirical investigation reveals a counter-intuitive phenomenon which we term the Long-Caption Paradox. As demonstrated in Table. 4, the model pretrained on the concise Bunny dataset consistently outperforms the model trained on the linguistically rich DenseFusion dataset. This performance gap persists despite both models utilizing the exact same ReAlign strategy and encoder. We attribute this paradox not to a single factor, but to structural constraints verified by our geometric analysis: 1 Representation Capacity, 2

Diffuse Covariance, and 3 Signal-to-Noise Ratio.

#### G.1 Truncation-Induced Supervision Mismatch

Even with advanced encoders like LLM2CLIP that extend the context window, there remains a hard physical limit on the input sequence length. Unlike the visual encoder, which processes the image holistically, the text encoder operates within a fixed context window. When a dense caption T exceeds this limit, the trailing semantic content is inevitably truncated and discarded before encoding. This truncation creates a fundamental discrepancy between the modalities. The visual embedding v encodes the global visual information, whereas the textual embedding z only captures the truncated prefix of the caption. Consequently, the model attempts to align the full image to an incomplete textual description, as the details described in the truncated tail are absent from the feature space.

#### G.2 Diffuse Covariance Structure

From the perspective of ReAlign, long captions introduce specific geometric challenges. We quantify this using the effective rank to measure the complexity and compactness of the embedding manifolds. As shown in Fig. 10, our measurements reveal that visual embeddings possess a high Effective Rank (≈ 57.5), reflecting their rich detail. Long captions attempt to mimic this complexity, exhibiting a similarly high rank (≈ 52.9). However, this high rank implies a diffuse covariance structure, a high-dimensional cloud with high entropy. In contrast, Short captions exhibit a significantly lower Effective Rank (≈ 41.0). This suggests that short

The image is a colorful poster featuring a stylized illustration related to "nugglife," which is associated with cannabis culture. The poster is dominated by shades of green, yellow, and orange, with a variety of cannabis leaves and buds depicted in a cartoonish style. In the center, there is a caricature of a person with exaggerated facial features, wearing a green and black checkered hat, and holding a pair of scissors in a threatening manner towards the viewer. The scissors are open and positioned as if they are about to cut through the image. The text on the poster is prominently displayed in white and yellow fonts against the green background. It reads "clutter gallery THE WORLD OF NUGGLIFE 02-09-19 - 03-01-19 fueled by LAGUNITAS BREWING COMPANY." The text indicates that the event is associated with Clutter Gallery and takes place from February 9, 2019, to March 1, 2019, and is sponsored by Lagunitas Brewing Company. The font sizes vary, with "THE WORLD OF NUGGLIFE" being the largest and most central, drawing attention to the theme of the event. The overall design of the poster is vibrant and eye-catching, with a playful and edgy aesthetic that is typical for events related to cannabis culture.

[Figure 17]

The image displays a large grid of individual portraits, each person holding a white sign with text. The text on the signs

[Figure 18]

is not legible. The individuals are arranged in a dense, grid-like pattern, covering the entire visible area of the image. They are wearing various colors of clothing, including shades of blue, red, green, and black, among others. The background is uniformly dark, contrasting with the lighter colors of the subjects' clothing. At the bottom of the image, there is a text overlay in a golden font that reads "The Chicago 2015 Golden Gloves 325 registered fighters." Additionally, there is a logo in the bottom right corner with the letters "TB" and the website "TOWNSQUARE.COM" beneath it. The date "2/23/2015" is also present in the bottom right corner, suggesting the image was taken or published on that date. The overall layout of the image is symmetrical and organized, with the subjects' faces and signs being the focal points.

The image displays an interior room with a large mural on the wall. The mural features a botanical theme with a variety of flowers, predominantly in shades of pink and purple, with green foliage. There are multiple birds illustrated within the mural, with one bird perched on a branch in the center and others in flight. In the room, there is a couch with a textured fabric in shades of brown and beige, adorned with several cushions. A coffee table is situated in front of the couch, holding a few items including a blanket with a geometric pattern, a small plant in a pot, and a book. To the right of the couch, there is a lamp with a metallic base and a white shade. The floor is covered with a textured rug that complements the room's earthy color palette. The room is well-lit, suggesting natural light coming from outside the frame. The text overlay on the image reads "SIMSPIRATIONBUILDS" at the top and "BOTANICA COLLECTION MURAL: CARIBBA 4 swatches" at the bottom, indicating the branding and the name of the mural design. The text is in a clean, sans-serif font, with the brand name in a larger size than the mural's name.

[Figure 19]

- Figure 11 Visualization of Linguistic Noise in Dense Captions. We highlight non-visual segments (marked in red) within long captions, such as abstract inferences, contextual associations, and subjective interpretations. These tokens lack direct visual grounding and geometrically act as noise vectors, pulling the semantic centroid away from the true visual anchor.

captions act as a low-rank approximation of the visual content, filtering out background noise and retaining only the most structurally salient principal components. Aligning a compact manifold to the visual manifold is statistically more stable than aligning a diffuse, high-entropy cloud.

#### G.3 Linguistic Noise & Modality Gap

Finally, we distinguish between Linguistic information and Visual Information. A larger token count does not equate to higher visual utility. Dense captions often contain non-visual context (The atmosphere was tense) or abstract inferences, as visualized in Fig. 11. Geometrically, these tokens act as noise vectors that pull the semantic centroid away from the visual anchor. This is empirically verified by the initial modality gap. As shown in our experiments, the long-caption dataset exhibits a significantly larger gap (∥∆µ∥ ≈ 0.51) compared to the Short-Caption dataset (∥∆µ∥ ≈ 0.39).

Conclusion: The extra linguistic information in long captions effectively acts as a disturbing force, increasing the modality gap by ≈ 30%. For statistical modality alignment, the visual grounding and geometric compactness of the text distribution are more critical than raw length.

### H Experiments Setting

#### H.1 CLIP Pretraining Setting

To empirically validate theoretical hypotheses, we established a controlled pretraining environment utilizing a TinyCLIP architecture comprising a 40 million parameter ViT B/32 vision encoder and a 19 million parameter Transformer text encoder. The model was trained on 2 million randomly sampled image-text pairs from LAION 400M with 224 by 224 resolution inputs using the AdamW optimizer with a weight decay of 0.0001 and a cosine learning rate schedule peaking at 0.0001 with a warmup ratio of 0.1. The training process was distributed across 8 NVIDIA A6000 GPUs with a global batch size of 4096. Distinct from standard protocols, we integrated a custom online geometric monitoring system that utilizes a dedicated held-out probe set of 200,000 samples to perform high-frequency spectral analysis every 20 steps. This protocol dynamically updates the reference subspace basis until step 800 before freezing it for stable drift measurement and employs 16 accumulated batches to approximate the Fisher Information Matrix for curvature analysis while tracking the top 128 eigen components to verify spectral decay. All linear algebraic computations are executed in Float32 precision to ensure numerical stability against the BF16 training backdrop.

#### H.2 MLLM Training Setting

We employ Llama-3-8B-Instruction as the LLM backbone, connected to the input modalities via a two-layer MLP projector with GELU activation. Our core design utilizes aligned text representations as pseudo-visual tokens. These text-derived embeddings are mapped directly into the LLM’s feature space through the MLP. Training Pipeline. The training process consists of two stages: 1 Stage 1: Modality Substitution Pretraining. We train the projector for 1 epoch on the filtered Bunny-1M dataset with a learning rate of 5 × 10−4, while keeping the LLM parameters frozen. 2 Stage 2: Visual Instruction Tuning. We perform full-parameter fine-tuning on the InternVL-Chat-V1.2 dataset for 1 epoch. The projector is initialized with weights from Stage 1, and the learning rate is reduced to 1e-5. The experiments were conducted on 8 NVIDIA H200 GPUs. With a total data scale of approximately 2.2M samples, the complete training pipeline was finished in 12 hours.

#### H.3 Eval Setting

We evaluate ReVision across three primary dimensions using a comprehensive suite of benchmarks: General Perception (MME test [8], MMStar [5], ScienceQA-image dev&test [18], and RealWorldQA), Complex Reasoning (MMMU validation single-image [36], MMMU-Pro single-image [37], VisuLogic train [30], and LogicVista [29]), and Hallucination Assessment (CRPE [25], POPE [14], and HallusionBench [9]). For all benchmarks, we report accuracy (acc) as the unified metric to ensure a consistent and rigorous comparison.

#### H.4 Cost Analysis

We quantify the data acquisition cost using the formula Ctotal = Pin · Tin + Pout · Tout, where Ctotal denotes the total cost, and T and P represent the token count and price per million tokens, respectively. To ensure a standardized comparison, we assume unified pricing based on GPT-5 APIs for all methods, with rates set at Pin = $1.25 and Pout = $10.00 per million tokens. Under this setting, the methods exhibit distinct cost profiles: Unicorn incurs the highest expense of $1893.27 (17.50M input, 187.14M output), whereas the standard ReVision (1M) is the most economical at $176.10 (11.64M input, 16.15M output). Scaling to ReVision-2M brings the cost to $352.20 (23.28M input, 32.30M output), which remains notably lower than the paired w/. Image baseline cost of $476.05 (221.64M input, 16.15M output). For clarity, all costs are normalized relative to the w/. Image baseline (1.0).

### I More MLLM Experiments

In this section, we provide additional MLLM experiments to further analyze the effectiveness and scalability of ReVision. We first ablate the three components of ReAlign, then study the role of Modality Substitution Pretraining, and finally evaluate scaling behavior with respect to Stage-2 visual instruction data and LLM backbone size.

- Table 5 Ablation of ReAlign components. Step 1 denotes Anchor Alignment, Step 2 denotes Trace Alignment, and the full ReAlign additionally includes Centroid Alignment.

Method

General Reasoning Hallucination

Avg. ↑

MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

w/. Step 1 74.86 33.18 73.93 42.64 28.48 25.61 22.94 18.86 78.93 70.82 43.24 46.68 w/. Step 1 & 2 78.22 35.71 76.21 45.16 30.05 26.09 26.05 22.16 80.37 71.18 44.14 48.67

ReAlign / ReVision 79.65 36.13 76.71 47.97 31.51 28.39 27.70 22.82 81.78 72.53 46.58 50.16

- Table 6 Ablation of Modality Substitution Pretraining. Removing Stage 1 consistently degrades performance across all benchmarks.

General Reasoning Hallucination

Avg. ↑

Method

MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench w/o Stage 1 66.80 29.70 71.90 39.10 25.60 23.40 20.10 16.90 75.80 68.90 41.20 43.58

ReVision 79.65 36.13 76.71 47.97 31.51 28.39 27.70 22.82 81.78 72.53 46.58 50.16

#### I.1 Ablation of ReAlign Components

To verify the contribution of each step in ReAlign, we conduct a component-wise ablation by progressively adding Anchor Alignment, Trace Alignment, and Centroid Alignment. As shown in Table 5, Anchor Alignment alone already provides a strong baseline by correcting the first-order mean mismatch. Adding Trace Alignment further improves the average score by matching the global residual energy scale while preserving the learned spectral structure. Finally, Centroid Alignment brings additional gains by correcting the centroid drift induced by spherical normalization. These results support the design of ReAlign as a three-stage calibration procedure.

#### I.2 Effect of Modality Substitution Pretraining

We further ablate the first stage of ReVision to examine whether Modality Substitution Pretraining is necessary. In the w/o Stage 1 setting, the model skips pseudo-visual pretraining and directly proceeds to visual instruction tuning. As shown in Table 6, removing Stage 1 leads to a substantial performance drop across all benchmarks, decreasing the average score from 50.16 to 43.58. This confirms that Stage 1 is not merely a generic projector warm-up; instead, it provides a visually compatible semantic interface before real-image instruction tuning.

#### I.3 Scaling Visual Instruction Tuning Data

We study how the amount of Stage-2 visual instruction tuning data affects ReVision. We keep Stage 1 fixed and vary only the amount of Stage-2 SFT data. As shown in Table 7, increasing the SFT data from 30% to 60% and then to 100% consistently improves performance. This trend indicates that Stage 1 establishes a useful cross-modal semantic interface, while Stage 2 progressively injects fine-grained real-image supervision and instruction-following ability.

#### I.4 Scaling the LLM Backbone

We evaluate whether ReVision benefits from scaling the LLM backbone. We replace only the LLM backbone from Llama-3-8B-Instruct to Llama-3-70B-Instruct, while keeping the visual/text encoder, ReAlign operator,

- Stage-1 corpus, Stage-2 SFT dataset, and evaluation suite unchanged. As shown in Table 8, scaling the backbone improves the average score from 50.16 to 52.33. The gains are especially visible on reasoning-intensive benchmarks, suggesting that once the geometric pseudo-visual interface is established, a larger LLM can better exploit the injected semantic supervision.

- Table 7 Scaling Stage-2 visual instruction tuning data. Stage 1 is fixed, and only the amount of Stage-2 SFT data is varied.

- Stage-2 Data

General Reasoning Hallucination

Avg. ↑

MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench

30% 68.53 30.60 70.90 41.88 28.67 25.33 23.60 17.17 74.65 66.28 40.35 44.36 60% 75.99 34.67 75.06 45.36 30.12 27.47 25.10 20.06 78.57 70.53 44.39 47.94

100% 79.65 36.13 76.71 47.97 31.51 28.39 27.70 22.82 81.78 72.53 46.58 50.16

- Table 8 Scaling the LLM backbone under the same ReVision pipeline. Only the LLM backbone is changed, while all other components and data settings remain fixed.

General Reasoning Hallucination

Avg. ↑

Backbone

MME MMStar SQA RealWorldQA MMMU MMMU-P VisuLogic LogicVista CRPE POPE HallBench Llama-3-8B-Instruct 79.65 36.13 76.71 47.97 31.51 28.39 27.70 22.82 81.78 72.53 46.58 50.16

Llama-3-70B-Instruct 80.27 37.84 78.92 49.16 34.99 31.27 31.54 26.11 82.52 73.89 49.15 52.33

### J Qualitative Analysis

To intuitively understand the improvements brought by ReVision, we conduct a comprehensive evaluation across three distinct cognitive levels: General Visual Perception, Domain-Specific Knowledge, and Logical Reasoning. As visualized in Fig. 12, the model demonstrates exceptional capability in handling complex Abstract and Spatial Reasoning tasks. Specifically, in the matrix pattern completion case, the model moves beyond simple texture matching to identify the underlying geometric progression rules. This capacity for mental simulation is further evidenced in the geometry folding problem, where the model successfully performs mental rotation to reconstruct a 3D cube from a 2D net, accurately predicting the spatial adjacency of symbols. Beyond abstract logic, the model exhibits robust Fine-Grained Perception and Knowledge Integration. In the domain of data analysis, it accurately interprets the intersection points of two trend lines in a statistical chart, extracting precise numerical values rather than merely describing general trends. Furthermore, the model effectively grounds visual signals into specific world knowledge. This is demonstrated in the geospatial recognition scenario, where it combines visual map boundaries with geographical facts to identify the easternmost state, and in the physical common sense task, where it correctly deduces magnetic attraction based on the orientation of poles.

### K Related Work

The Modality Gap refers to the systematic distributional distance between representations of paired data from distinct modalities within a shared embedding space, despite the theoretical expectation that semantically identical pairs should align. Early empirical research first identified this geometric anomaly as a cone effect [15], observing that embedding vectors tend to occupy a narrow cone rather than spanning the full hypersphere [24]. The C3 framework [39] provided the first formal description of this structure, characterizing the gap as a superposition of a constant orthogonal displacement and random alignment noise. While these frameworks offer a foundational understanding, they predominantly rely on the assumption that the noise term is isotropic. Our work challenges this simplification, demonstrating that the residual noise exhibits high anisotropy, thereby necessitating more precise second-moment modeling. The absence of this theoretical perspective has directly limited recent explorations into training MLLMs [1, 2, 16, 26, 28, 34] using pure text. Unicorn [35] pioneered the use of the modality gap to convert text representation into the pseudo-visual representation. However, due to their reliance on simple mean shifting, which implies an isotropic assumption, the synthesized representation fails to match the complex geometric shape of the real visual representation. This further highlights the necessity of constructing a novel training paradigm based on precise covariance alignment.

[Figure 20]

Q: Graph compares the 5-year annualized return for 15 mutual funds with the 3-year annualized return for the same funds. The solid line is the line of best fit. The number of funds that have equal 3-year and 5-year returns is?

A: 1

[Figure 21]

Q: Choose the most suitable option from the given four choices to fill in the question mark, so that it presents a certain pattern.

A: B

[Figure 22]

Q: Which of these states is farthest east?

A: The state that is farthest east is Florida.

[Figure 23]

Q: The given figure is a net of the outer surface of a box. Which of the following four options is the result when it is folded?

A: A

Q: Two magnets are placed as shown. Will these magnets attract or repel each other?

[Figure 24]

A: These magnets will attract each other.

- Figure 12 Qualitative analysis examples of ReVision. This figure shows the model’s visual perception ability across various complex tasks.

### L Broader Impact

Our work advances the paradigm of Data-Efficient AI, specifically by decoupling the acquisition of visual semantics from the reliance on massive paired datasets. By demonstrating that the heavy lifting of knowledge injection can be offloaded to massive unpaired text during the Pretraining stage, while reserving limited real images solely for the SFT stage, we fundamentally alter the resource landscape for training MLLMs. This paradigm shift has profound implications in several key areas:

Democratization of Multimodal Research: The prohibitive cost of collecting and cleaning billion-scale imagetext pairs has historically concentrated MLLM development within a few well-resourced institutions. By shifting the data requirement to abundant unpaired text for the bulk of training, our approach significantly lowers the barrier to entry, enabling academic labs and smaller research groups to train high-performance models from scratch.

Expansion to Low-Resource Domains: In specialized fields (e.g., medical imaging, minority languages, or technical diagrams), paired data is scarce, yet textual knowledge is often available. Our ReVision paradigm allows models to learn visual concepts primarily through domain-specific text corpora during pretraining,

requiring only a handful of examples for SFT. This opens new avenues for deploying MLLMs in domains where data scarcity previously made it impossible.

Mitigation of Copyright and Privacy Risks: Large-scale web-scraped image datasets often contain copyrighted artwork and sensitive personal identifiable information (PII). By minimizing the reliance on raw images during the data-hungry pretraining phase and relying instead on text, our method offers a potential pathway to reduce legal and ethical risks associated with dataset curation.

