# arXiv:2605.06169v1[cs.LG]7May2026

## Mean Mode Screaming: Mean–Variance Split Residuals for 1000-Layer Diffusion Transformers

Pengqi Lu Beijing, China luer5old@gmail.com

[Figure 1]

Figure 1: Text-to-image generation samples from our 1000-layer MV-Split DiT. More samples are provided in Appendix M. Code: https://github.com/erwold/mv-split. Model weights: https://huggingface.co/StableKirito/mvsplit-dit-1000l.

### Abstract

Scaling Diffusion Transformers (DiTs) to hundreds of layers introduces a structural vulnerability: networks can enter a silent, mean-dominated collapse state that homogenizes token representations and suppresses centered variation. Through mechanistic auditing, we isolate the trigger event of this collapse as Mean Mode Screaming (MMS). MMS can occur even when training appears stable, with a mean-coherent backward shock on residual writers that opens deep residual branches and drives the network into a mean-dominated state. We show this behavior is driven by an exact decomposition of these gradients into mean-coherent and centered components, compounded by the structural suppression of attention-logit gradients through the null space of the Softmax Jacobian once values homogenize. To address this, we propose Mean–Variance Split (MV-Split) Residuals, which combine a separately gained centered residual update with a leaky trunk-mean replacement. On a 400-layer single-stream DiT, MV-Split prevents the divergent

Preprint.

collapse that crashes the un-stabilized baseline; it tracks close to the baseline’s pre-crash trajectory while remaining substantially better than token-isotropic gating methods such as LayerScale across the full schedule. Finally, we present a 1000-layer DiT as a scale-validation run at boundary scales, establishing that the architecture remains stably trainable at extreme depth.

### 1 Introduction

Scaling laws for generative modeling [1] indicate that depth is an important dimension of capacity and model performance. Training ultra-deep Diffusion Transformers (DiTs) [2–4], however, introduces structural reliability issues that are not well described by standard exploding or vanishing gradient heuristics. In some runs, optimization remains stable for thousands of steps and then diverges within a few updates, with the loss returning near its initialization level and not recovering. These events can occur without NaNs or obvious forward saturation.

In this work, we study a mean-dominated collapse state in ultra-deep DiTs, in which token representations homogenize and centered token variation is suppressed. We reserve the term Mean Mode Screaming (MMS) for the abrupt entry event into this state: a spike in the mean-coherent gradient component, rapid residual branch opening, and subsequent Q/K gradient suppression.

Mechanistically, this failure exploits a geometric asymmetry between the token-mean and centered subspaces. Row-stochastic attention strictly preserves pure-mean states, while the centered component is propagated by a separate mixing operator and can become contractive in deep layers. On the backward pass, gradients admit an exact decomposition into mean-coherent and centered components; as token alignment increases, the mean-coherent component accumulates coherently with sequence length and can dominate the residual branch update. Once values homogenize, attention-logit gradients are suppressed through the null space of the Softmax Jacobian, suppressing Q/K learning and locking the network into the collapsed state.

Existing depth stabilizers suppress the entire residual branch isotropically in token space: ReZero [5] and LayerScale [6] apply scalar and per-channel learnable gates respectively, shrinking the mean and centered components together. This stabilizes training but slows convergence by also damping the centered signal responsible for spatially varying feature learning.

These observations motivate MV-Split Residuals, which combine a separately gained centered residual update with a leaky trunk-mean replacement. By damping the mean path without shrinking the centered path by the same factor, MV-Split stabilizes training without the convergence cost of isotropic residual gating.

Our contributions are:

- 1. Characterization. We characterize a mean-dominated collapse state and distinguish it from MMS, the abrupt entry event into this state. A standard-initialization control reaches the same collapse state more progressively across depth.
- 2. Mechanism. We show that row-stochastic attention preserves pure-mean states, that gradients split exactly into mean-coherent and centered components, with the mean-coherent component entering an O(T) coherent regime when tokens align, and that value homogenization suppresses attention-logit gradients through the null space of the Softmax Jacobian.
- 3. Method and result. We propose MV-Split Residuals, which combine a separately gained centered residual update with a leaky trunk-mean replacement. In matched 400-layer quantitative evaluation, MV-Split removes collapse events and converges faster than LayerScale; in a separate 1000-layer run, the same design remains stably trainable and serves as a scale-validation run at boundary scales.

### 2 Preliminaries

We first describe the backbone, initialization, and training objective used in the main training runs.

DiT-400Layer (failed)

2.2

L0

L224 L255 L288 L320 L352 L368 L384

DiT-400Layer-MVSplit

L16 L32 L64 L96

Text Tokens Image Tokens

80

Image

Caption

DiT-1000Layer-MVSplit

... ... ...

2.0

L128 L160 L192

Qwen3 VAE

[Figure 2]

[Figure 3]

2=(X)/c(X)×10[]TFF

1.8

60

patchify

K-Norm

Q-Norm

TrainLoss

Norm

1.6

Norm

Q

K V

Multi-Head Self-Attention

40

1.4

Transformer Block

[Figure 4]

xN

MLP

0

+

1.2

Zero-Init

unpatchify

RMSNorm

20

1.0

Velocity Prediction

SwiGLU

0

+

Zero-Init W₂

0.8

RMSNorm

0

0 2 4 6 8 10

25 26 27 28 29 30

Training Steps (k)

Training Steps (k)

- Figure 2: Baseline DiT and representative training diagnostics. (Left) Single-stream DiT backbone. (Middle) Training loss over the first 10k steps for the un-stabilized 400-layer baseline and the MV-

Split 400-/1000-layer runs. (Right) Per-layer energy ratio ρT = ∥µ(X)∥F/∥c(X)∥F (Appendix A) across L0–L384 in a baseline run.

#### 2.1 Minimal Single-Stream Multi-Modal Diffusion Transformer

We use a deliberately stripped-down single-stream DiT [4] so that deep residual propagation, rather than external modulation or skip pathways, remains the dominant carrier of both signal and gradients. Concretely, we employ a Post-Norm residual chain [7, 8] (Xl+1 = RMSNorm(Xl + fl(Xl)) [9]) without AdaLN [4] or other per-layer modulation mechanisms, to avoid introducing alternative depthwise control channels that would complicate attribution of the collapse dynamics. Instead of cross-attention, we concatenate VAE-encoded [10, 11] image tokens Ximg and text embedding tokens Xtxt into a unified sequence Xin = [Ximg;Xtxt] [12–15], forcing self-attention [7] to handle all multimodal interaction. For positional encoding, we apply a 2D extension of RoPE [16] to image tokens following recent vision/diffusion Transformer practice [17, 18], and leave text tokens without rotary positional encoding. The left panel of Figure 2 gives the corresponding backbone schematic.

#### 2.2 Residual Writer Zero Initialization

For the main training runs used in the main text, except the LayerScale control, we zero-initialize the residual writers (WO and W2), following the broader practice of identity-initialized residual branches and zero-initialized output pathways in residual and diffusion architectures [19, 20, 4, 21, 22]. Here WO is the attention output projection. For the FFN branch, we write the SwiGLU [23] feed-forward transformation as

[gl,vl] = W13xl, FFN(xl) = W2 SiLU(gl) ⊙ vl , (1)

so W2 is the residual writer of the FFN block. In these zero-writer training runs, the internal branch parameters (e.g., Q,K,V and W13) remain at their standard initialization. Appendix B shows that standard initialization does not avoid the mean-dominated regime; the same collapse appears from the start as a depth-progressive front, rather than through the delayed writer-opening spike that defines MMS in the zero-writer training runs.

#### 2.3 Rectified Flow Matching

We train the model using a Rectified Flow [24, 25] objective. Given a data distribution x0 (VAE latents) and a Gaussian noise distribution x1 ∼ N(0,I), we define a linear interpolation path zt = (1 − t)x0 + tx1 for t ∈ [0,1]. The model vθ is trained to predict the vector field pointing from noise toward data:

0,x1 ∥vθ(zt,Xtxt) − (x0 − x1)∥2 (2)

L = Et,x

### 3 Failure Dynamics: Mean-Dominated Collapse

To understand the failure mode limiting depth scaling, we analyze a representative abrupt-failure run from the main diagnostic regime. We first introduce a token-space decomposition that separates sequence-mean and centered variation. We then use this decomposition to trace the observed

divergence sequence: a mean-coherent gradient shock, residual branch opening, mean-dominated forward collapse, and Q/K gradient suppression. Section 4 explains why this sequence occurs.

#### 3.1 Geometric Preliminaries: Token-Space Asymmetry

The failure dynamics are fundamentally tied to how information is distributed across tokens. Let 1 ∈ RT denote the all-ones vector, and define J ≜ T1 11⊤ and P ≜ I − J. For any token sequence X ∈ RT×D, we write

X = JX + PX ≡ µ(X) + c(X), (3)

where µ(X) ≜ JX is the sequence-mean component and c(X) ≜ PX is the centered variation component. Row-stochastic attention acts asymmetrically on these two subspaces.

- Proposition 1 (Pure-mean component is preserved). For any row-stochastic attention matrix A satisfying A1 = 1, Aµ(X) = µ(X).

Note that Proposition 1 governs only the pure-mean component of the input. For a general input X = JX + PX, the output mean satisfies µ(AX) = JAX = JX + JAPX; centered variation can therefore contribute to the output mean through the leakage term JAPX.

- Proposition 2 (Centered component is governed by PAP). For any row-stochastic attention matrix A satisfying A1 = 1,

c(AX) = PAX = PAPX, and therefore ∥c(AX)∥F ≤ ∥PAP∥2 ∥c(X)∥F. (4)

We denote µeff(A) ≜ ∥PAP∥2. When µeff(A) < 1, the layer is strictly contractive on the centered subspace. This geometric asymmetry imposes a structural vulnerability: row-stochastic attention

leaves pure-mean states invariant, while its action on token-specific variation is governed by µeff and can become contractive. Consequently, the network must rely on residual branches to continuously replenish the centered subspace. If the residual updates become dominated by the mean component, the representation is driven toward a pure-mean state.

#### 3.2 Tracing the Divergence Event: From Trigger to Lock-in

- Figure 3 traces the divergence in a 400-layer baseline through a tight chronological sequence. The backward pass exhibits a mode-selective shock: the gradient spike is concentrated primarily in the mean-coherent component while Q/K gradients collapse in lockstep, leaving residual writers as the dominant active learning channel. This shock then locks in across the forward pass — branches

open into a mean-dominated regime (ρT explodes), and with attention contractive on the centered subspace and no branch-side variance replenishment, tokens homogenize across depth into a trivial mean-prediction baseline.

This empirical sequence isolates two questions for the mechanistic analysis in Section 4: (1) why the gradient amplifies specifically in the mean-coherent direction, and (2) why token homogenization structurally suppresses Q/K gradients.

- 4 Mechanism

#### 4.1 Gradient Decomposition and Backward Alignment Amplification Law

Consider a token-wise linear map W (e.g., residual writers WO,W2) whose gradient takes the form ∇WL = Tt=1 δtyt⊤. Decomposing the forward inputs yt and backward gradients δt into their sequence means (y,¯ δ¯) and centered residuals (y˜t,δ˜t), the cross-terms vanish identically under summation (proof in Appendix C.1), yielding an exact additive decomposition:

+ Tt=1 δ˜ty˜t⊤

∇WL = T δ¯y¯⊤

. (5)

∆Wµ (mean-coherent, O(T) when aligned)

∆Wc (centered, diffusive)

We denote Gmean ≜ ∥∆Wµ∥F and Gctr ≜ ∥∆Wc∥F. This decomposition exposes a scaling transition. The mean component has norm ∥∆Wµ∥F = T ∥δ¯∥∥y¯∥, so it remains small when sequence means

(a) Macroscopic Crash

(b) Mean-Coherent Writer Shock

(c) Q/K Gradient Extinction

2.2

Loss

100

G(WO)

- 100

- 101

10 1

zoom @ spike

g raw

G(Q) G(K)

2.0

10 1

10 1

WriterGradComponent

GradNorm(unclipped)

10 2

1.8

10 2

10 2

GradL2(deep)

10 3

10 3

1.6

10 3

Loss

10 4

10 4

1.4

Attn Gmean: 155× FFN Gmean: 61× Attn Gctr: 1× FFN Gctr: 3×

10 5

10 5

10 4

1.2

26350 26400 26450 26500

10 6

AttnW

1.0

Gmean

O

10 5

AttnW

Gctr

10 7

O

10 1

FFNW

Gmean

2

0.8

FFNW

Gctr

2

10 8

26000 27000 28000 29000 30000

26000 27000 28000 29000 30000

26000 27000 28000 29000 30000

Training Step

Training Step

Training Step

(d) Gate Opening → Mean Dominance

(e) Contractive Lock-in (µeff)

(f) Token Homogenization Across Depth

1.4

384

[Figure 5]

rTR (solid)

L0

2.00

1.0

L0

[Figure 6]

L96

- 100

- 101

- 102

- 103

- 104

L96

L192 L288 L384

1.2

320

L192 L288 L384

1.75

0.8

TokenCosineSimilarity

1.50

1.0

255

Mean/Varenergyρ

Layerdepth

1.25

0.6

192

0.8

rTR

1.00

µe

ρ (dashed)

128

0.6

0.4

L0

0.75

L96

L192 L288 L384

64

0.4

0.50

0.2

0.25

8

0.2

0.0

0.00

0

26000 27000 28000 29000 30000

26000 27000 28000 29000 30000

26000 27000 28000 29000 30000

Training Step

Training Step

Training Step

Mean Mode Screaming

- Figure 3: Empirical trajectory of a representative divergence event (400-layer). The vertical dashed line marks the divergence step. (a–c) Backward trigger: The global gradient norm spikes (a). The spike is concentrated in the mean-coherent gradient component Gmean, while the centered component Gctr shows no comparable amplification (b). After the spike, Q/K gradients drop by roughly four orders of magnitude while WO gradients remain nonzero (c). (d–f) Forward lock-in: The residual branch opens and the mean/centered energy ratio ρT rises sharply (d). Deep attention remains contractive on the centered subspace, with limited variance replenishment (e). Token representations homogenize across depth, with cosine similarity approaching one in deep layers (f).

cancel; under weak centered alignment, ∆Wc sums diffusively. As representations and adjoints homogenize, however, the sequence means stop canceling, ∥y¯∥ and ∥δ¯∥ become order-one, and the rank-1 mean mode enters its coherent O(T) regime. Operationally, Mean Mode Screaming acts as a sharp transition from diffusive cancellation to coherent accumulation.

To quantify this transition, we define the dimensionless alignment amplification A as the ratio of the true gradient energy to the independent-token baseline. As derived in Appendix C.2, expanding this ratio yields an identity linking the cross-token coherent amplification of gradients to microscopic token alignment. Under an equal-magnitude proxy, it takes the compact form:

∥∇WL∥2F t ∥δt∥2∥yt∥2

Amplification A

(δ⊤s δt)(y⊤s yt) t ∥δt∥2∥yt∥2

− 1 = s̸=t

≈ (T − 1) Es̸=t cos(ys,yt)cos(δs,δt)

Pairwise alignment κ

. (6)

Equation 6 identifies when token-wise gradients stop canceling and enter a coherent accumulation regime. When tokens are heterogeneous, signed off-diagonal terms cancel (κ ≈ 0) and A ≈ 1. As both representations and adjoints become aligned in deep layers, the signed off-diagonal terms stop canceling; in the limiting case cos(ys,yt) → 1 and cos(δs,δt) → 1, giving κ → 1 and the gradient enters its O(T) coherent-amplification regime. We empirically audit this transition in Section 6.1 using the absolute-coherence upper-envelope proxy κˆ ≜ Es̸=t[|cos(ys,yt)||cos(δs,δt)|].

#### 4.2 Q/K Gradient Extinction via the Softmax Null Space

A gradient spike alone would not lock in the failure if the attention path could restore token variation. However, once the residual stream becomes mean-dominated, the value vectors homogenize.

Consequently, the Softmax Jacobian zeroes out the constant component of the attention-weight gradient.

Lemma 1 (Softmax null space under value collapse). For one attention row i, if Vj = v¯ for all j, then ∂L/∂Si = 0, where Si is the vector of pre-softmax logits.

By the chain rule, ∂L/∂aij = ⟨∂L/∂Yi,Vj⟩ is independent of j when Vj = v¯, yielding ∂L/∂ai ∝ 1. Because Jsm(ai)1 = 0, the logit gradient strictly vanishes. Under approximate homogeneity, this null space still removes the constant component, strongly suppressing Q/K learning while the residualwriter gradient (Eq. 5) is not zeroed by this Softmax null space (proof in Appendix C.3).

### 5 Method: MV-Split Residuals

Section 4 isolates a single unstable mode: the rank-one mean-coherent gradient update ∆Wµ. We therefore decouple its residual gain from the centered update. Let Xl ∈ RT×D be the trunk and Fl ≜ fl(Xl) the branch output. Using the orthogonal projectors J and P = I − J from Section 3.1, we replace the standard Post-Norm merge Xl+1 = RMSNorm(Xl + Fl) with a subspace-routed merge:

Zl ≜ Xl + β ⊙ (PFl)

, (7)

+ α ⊙ J(Fl − Xl)

centered path

mean path

Xl+1 = RMSNorm(Zl), (8) where α,β ∈ RD are per-block learnable vectors broadcast across tokens. Our multimodal transformer implementation applies the residual projectors segment-wise (Jseg,Pseg) to avoid directly mixing image and text means in the residual control path (Appendix E).

Forward dynamics. Prior to token-dependent RMS normalization, projecting Eq. 7 exactly decouples the pre-normalization merge:

PZl = PXl + β ⊙ (PFl), JZl = (1 − α) ⊙ (JXl) + α ⊙ (JFl). (9) The centered subspace follows a standard residual update with gain β, while the mean subspace becomes a per-feature leaky integrator (when 0 < αd ≤ 1): each layer contracts the trunk mean by 1 − αd before adding a fresh correction.

Backward dynamics. Let Gl ≜ ∂L/∂Zl. Because J,P are self-adjoint and orthogonal, the gradient flowing back into the branch factors along the same split:

∂L ∂Fl

= β ⊙ (PGl) + α ⊙ (JGl). (10)

Centered and mean-coherent gradients receive independent gains. Together with (9), a small α both damps mean-coherent forward accumulation and shrinks the ∆Wµ component of the gradient (Eq. 5) by the same factor, without tying the local centered branch-gradient to the small mean gain α.

Comparison to other residual-gain methods. LayerScale [6] and ReZero [5] apply a single residual gain (per-channel and scalar, respectively) that does not distinguish the mean and centered subspaces, so ∆Wµ and ∆Wc are suppressed jointly. We elaborate on the structural distinctions between MV-Split and these residual-gain methods in Appendix D.

### 6 Experiments

The 400-layer comparison is matched in backbone, optimizer, data, batch size, and non-residual primitives on ImageNet-2012 [26] latents encoded with a frozen FLUX.2 VAE [11, 27] and conditioned on a frozen Qwen3-0.6B text encoder [28]; each stabilizer (un-stabilized Post-Norm baseline, LayerScale controls, MV-Split) uses its standard residual-initialization protocol (Appendix G). A separate 1000-layer run uses the same residual design and is reported as a 1000-layer scale-validation run (Figure 1 and Appendix M), trained from ImageNet pre-training through post-training on a separate ∼50k curated image set. Detailed training configuration is provided in Appendix G. Additional details on how we ruled out alternative explanations for the loss spike and localized the failure to MMS are reported in Appendix F.

#### 6.1 Testing the Alignment-Amplification Law

Figure 4 tests Eq. 6 in a representative unstable 400-layer run whose writergradient norm spikes at step t⋆ = 3400. Before the spike, both writers lie well below the saturation envelope. Absolute cross-token coherence is present, but the signed off-diagonal terms in Eq. 6 still cancel. The small pre-spike slopes therefore measure how loose the envelope is in this run, not new constants.

(a) Attn WO

(b) FFN W2

Pre: slope=0.169, R2=0.306 Crit: slope=0.961, R2=0.808

Pre: slope=0.126, R2=0.499 Crit: slope=0.985, R2=0.829

150

150

100

100

[Figure 7]

1

1

Pre-spike (zoom)

Pre-spike (zoom)

[Figure 8]

[Figure 9]

20

- 0

- 1

- 2

50

50

10

0

[Figure 10]

0

50 100

10 20

0

0 25 50 75 100 125 150 175

0 25 50 75 100 125 150 175

(T 1)

(T 1)

|Pre-spike samples Pre-spike regression Step 3400 layers Spike-step regression y=x<br><br>|
|---|

Figure 4: Writer amplification at the gradient spike (400-layer Base η run, t⋆=3400, measured on the T=256 image-token segment). Each point plots A − 1 against the equal-magnitude absolute-coherence upper-envelope proxy (T − 1)ˆκ for (a) Attn_WO and (b) FFN_W2. Gray points are pre-spike layer-step samples; colored points are active layers at t⋆ (A − 1 > 0.5). The dashed line is the absolute-coherence saturation envelope; fitted slopes and R2 values are shown in each panel.

At step t⋆, the active layers lie close to the saturation envelope for both Attn_WO and FFN_W2. The main observation is that the spike occurs when signed cancellation at the residual writer largely disappears. The same near-saturation appears in the attention and FFN writers, supporting a writer-interface explanation rather than an attention-specific one.

The largest active-layer values reach A − 1 ≈ 167, corresponding to a ∼ 13× writer-gradient norm amplification relative to the independent-token baseline. The shallowest active layer remains below the saturation envelope, consistent with a boundary region where absolute coherence is already high but sign cancellation has not fully disappeared. These measurements support the mechanism in Section 4.1: MMS occurs when residual writers lose signed cancellation across tokens, allowing the mean-coherent update ∆Wµ to approach its coherent O(T) scaling regime.

#### 6.2 MV-Split Shifts the Stability-Constrained Quality Frontier

We next evaluate whether MV-Split changes the usable quality frontier under an explicit stability constraint: a run is treated as usable only if it remains nondivergent over the measured training horizon.

- Figure 5 and Table 1 show the resulting stabilityconstrained quality frontier. The un-stabilized baselines are useful references for early learning speed, but they do not define stable frontier points: both enter the mean-dominated failure state. Reducing the learning rate delays this failure rather than removing it. LayerScale remains stable over the measured horizon, but its token-isotropic per-channel gain also reduces the centered residual updates needed for tokenvarying feature learning.

Baseline-400L

100

Baseline-400L ( /2)

200

LayerScale-400L

MVSplit-400L

50

| |
|---|

MVSplit-1000L

| |
|---|

| |
|---|

training diverged

InceptionScore

150

| |
|---|

###### FID50K

| |
|---|

20

| |
|---|

100

10

Baseline-400L

| |
|---|

| |
|---|

Baseline-400L ( /2)

LayerScale-400L

5

50

MVSplit-400L

1.7×

| |
|---|

| |
|---|

MVSplit-1000L

- 2

- 3

| |
|---|

| |
|---|

training diverged

| |
|---|

| |
|---|

0

0k 10k 20k 30k 40k 50k 60k 70k 80k

0k 10k 20k 30k 40k 50k 60k 70k 80k

Training Steps

Training Steps

- 0.1

- 1

Baseline-400L ( /2)

0.7

LayerScale-400L

GradientNorm

0.5

Baseline-400L MVSplit-400L MVSplit-1000L

0.3

0.2

0.07

0.05

0.03

0 10000 20000 30000 40000 50000 60000 70000 80000

Training step

Figure 5: Quality and optimizer stability over 80k steps (ImageNet 256×256). (Top) FID-50K and Inception Score. (Bottom) Post-clipping global gradient norm. The 400-layer curves define the controlled comparison: among the non-divergent 400-layer runs, MV-Split preserves a higher bounded gradient band than LayerScale while avoiding the spikes of the un-stabilized baselines. The 1000-layer MV-Split trace is included as scale validation.

Under this stability constraint, MV-Split shifts the controlled 400-layer frontier. It does not uniformly dominate the unstable baselines at early checkpoints, but those trajectories leave the stable set; MV-Split preserves much of their early convergence speed while avoiding their collapse. Among the non-divergent 400-layer runs, MV-Split is already substantially ahead of LayerScale by 20k–30k steps, and the added 40k/50k checkpoints show that this advantage persists rather than reflecting a short early transient. The gradient-norm trace also separates MV-Split from simple global shrinkage: it operates in a higher bounded gradient band than LayerScale, while avoiding the spikes seen in the un-stabilized runs.

The 1000-layer run extends this observation to boundary depth. The same residual design remains stable over the measured training horizon and reaches strong fixedcheckpoint FID/IS values at the reported boundary depth. Because this run uses a separate training and post-training pipeline, we do not use it as a matched frontier point against the 400-layer controls. Instead, it serves as scale validation: the residual mechanism that shifts the controlled 400layer frontier remains usable at 1000 layers. Additional GenEval and DPG-Bench measurements for the post-trained checkpoint are reported in Appendix K.1 as calibration rather than as state-of-theart comparison.

Table 1: Stability and convergence across 400-/1000-layer DiT runs. The 400-layer rows define the matched stability-constrained comparison. The 1000-layer row is a separate scale-validation point and is not part of the matched 400-layer frontier comparison. FID-50K and IS are computed with Euler sampling, 25 NFE, and CFG scale w = 2.0 for all rows. “—” denotes divergence before the checkpoint or failure to produce a valid evaluation. Bold highlights the best non-divergent result within the matched 400-layer comparison. The default-LR baseline diverges before the first checkpoint. † The lower-LR baseline diverges later in training and is shown only as a pre-crash speed reference; it is not counted as a stable frontier point. 400L LayerScale reports the best stable member of the λinit sweep.

FID↓/IS↑ Method @10k @20k @30k @40k @50k

400L Base (η) — — — — 400L Base (η/2)† 5.92 / 108.6 3.22 / 152.2 — — 400L LayerScale 14.08 / 59.2 6.50 / 96.6 4.09 / 130.5 3.33 / 149.6 2.90 / 165.5 400L MV-Split 7.23 / 89.8 3.64 / 139.9 3.09 / 166.5 2.79 / 182.0 2.60 / 185.5

1000L MV-Split 5.47 / 117.3 2.92 / 178.2 2.68 / 196.6 2.64 / 209.4 2.77 / 217.3

#### 6.3 Writer-Gradient Mode Decomposition

The convergence curves alone do not distinguish mode-selective control from a smaller effective learning rate. We therefore measure the two writergradient components from Eq. 5: the mean-coherent component Gmean and the centered component Gctr.

Baseline-400L

LayerScale-400L

10 2

LayerScale+Zero-Init-400L

MVSplit-400L

G(writer-grad,centered)ctr

G(writer-grad,mean)mean

10 3

10 3

10 4

10 4

Baseline-400L

LayerScale-400L

LayerScale+Zero-Init-400L

10 5

- Figure 6 shows that LayerScale bounds the meancoherent writer component, but does so by shrinking the centered component as well. This is expected from a token-isotropic residual gain: the same per-channel multiplier is applied before any token-space split, so the method provides no explicit mechanism to preserve centered variation while damping the token-mean component. The resulting low centered-gradient band is consistent with the slower convergence observed in Figure 5.

MVSplit-400L

0 2k 4k 6k 8k 10k 12k 14k

0 2k 4k 6k 8k 10k 12k 14k

Training Steps

Training Steps

Figure 6: Residual-writer gradient mode decomposition. Per-step median across depth of the mean-coherent (Gmean, left) and centered (Gctr, right) writergradient magnitudes; shaded regions denote the interquartile range (IQR; 25– 75% across depth). Token-isotropic per-channel gating compresses both modes; MV-Split bounds the mean-coherent component while preserving a higher, stable centered band.

MV-Split changes this pattern. The mean-coherent component remains bounded, while the centered component stays in a higher stable band. This supports the intended mechanism of Eq. 10: the mean

and centered components receive separate gains at the residual merge. Thus the improved stability in Section 6.2 is not explained by uniformly smaller gradients, but by damping the writer-gradient mode associated with the collapse.

Deferred analyses. Beyond stability, a linear probe confirms the token mean acts as an implicit global timestep carrier (near-perfect R2 predicting t across depth), justifying our design to gain-limit rather than strictly project out the mean subspace (Appendix H). Infrastructure-level optimizations for ultra-deep training are deferred to Appendix I.

- 7 Related Work 7.1 Deep Diffusion Transformers and Residual Stability

Diffusion Transformers replace U-Net backbones [29] with Transformer blocks over latent or image patches. DiT [4] showed that increasing Transformer compute through depth, width, or token count improves generative quality, while U-ViT [12] and MMDiT/Stable Diffusion 3 [13] demonstrate that token-based diffusion backbones can support long skips, multimodal token mixing, and rectifiedflow text-to-image generation. Unlike standard DiT conditioning stacks that inject the noise or timestep level through AdaLN or related modulation paths, recent work suggests that explicit noise/timestep conditioning is not always required for denoising generative models [30, 31]. Our focus is complementary to this objective-level question: we use a noise-agnostic backbone to study a depthwise residual-stream failure mode in ultra-deep DiTs and a residual merge that stabilizes this signal path. Appendix H further shows that our trained network implicitly carries the continuous timestep in the token-mean subspace.

Training instability in deep Transformers is often addressed by changing normalization placement, residual scaling, or residual connectivity. Post-LN Transformers can require warmup because large gradients appear near the output layers at initialization, whereas Pre-LN changes this gradient geometry [8]. Admin [32] attributes instability to residual-branch dependence that amplifies update perturbations. ReZero [5], LayerScale [6], DeepNorm [33], and Keel [34] stabilize depth by gating, rescaling, or altering the residual/carry path, with DeepNorm and Keel reporting 1,000-layer-scale Transformer training. These methods control the residual branch or carry path as a whole. MV-Split targets a different axis: it combines a separately gained centered residual update with a leaky trunkmean replacement, damping the mean-coherent channel without applying the same shrinkage to centered token variation.

MMS is also related to work on training spikes and attention/representation collapse, but its diagnostic object is different. Loss-spike and proxy studies connect large-scale instabilities to sub-layer Jacobian bounds, attention-logit growth, output-logit divergence, and predictive activation/gradient-norm trends [35, 36]. Other work shows that self-attention can drive token representations toward rank-one uniformity with depth [37], that rank collapse can vanish Q/K gradients via signal-propagation arguments [38], and that low attention entropy is associated with unstable or divergent Transformer training [39]. MMS, in contrast, is diagnosed through an exact gradient decomposition into meancoherent and centered components, followed by a mean-dominated forward state in which centered token variation is suppressed and Q/K learning is reduced through the Softmax Jacobian null space. We therefore do not claim that MMS explains all deep-Transformer spikes; it identifies a specific residual-subspace failure pathway in ultra-deep Post-Norm DiTs, and MV-Split acts at the residual interface rather than as a global residual shrinkage or attention-operator correction. Appendix J reports negative controls that test several superficially related alternatives.

- 8 Conclusion

We study a mean-dominated collapse state that limits depth scaling in very deep Diffusion Transformers, and we use Mean Mode Screaming (MMS) for the abrupt writer-gradient event that accompanies entry into this state in zero-writer training runs. The main mechanism is an imbalance between a mean-coherent writer update that can grow as O(T) and a centered path that is not sufficiently replenished once deep layers become contractive. MV-Split addresses this by combining a separately gained centered residual update with a leaky trunk-mean replacement. Under matchedbackbone stabilizer protocols at 400 layers, MV-Split removes collapse events and gives the best

stable frontier among the methods we evaluate; a separate 1000-layer run shows that the same design remains trainable at that depth.

### References

- [1] Zhengyang Liang, Hao He, Ceyuan Yang, and Bo Dai. Scaling laws for diffusion transformers, 2024.
- [2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020.
- [3] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations, 2021.
- [4] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023.
- [5] Thomas Bachlechner, Bodhisattwa Prasad Majumder, Huanru Henry Mao, Garrison W. Cottrell, and Julian McAuley. Rezero is all you need: Fast convergence at large depth, 2020.
- [6] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going deeper with image transformers, 2021.
- [7] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2017.
- [8] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture, 2020.
- [9] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.
- [10] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes, 2013.
- [11] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2022.
- [12] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models, 2022.
- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024.
- [14] Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, Xiangyang Zhu, Manyuan Zhang, Will Beddow, Erwann Millon, Victor Perez, Wenhai Wang, Conghui He, Bo Zhang, Xiaohong Liu, Hongsheng Li, Yu Qiao, Chang Xu, and Peng Gao. Lumina-image 2.0: A unified and efficient image generative framework, 2025.
- [15] Z-Image Team, Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, Zhen Li, Zhong-Yu Li, David Liu, Dongyang Liu, Junhan Shi, Qilong Wu, Feng Yu, Chi Zhang, Shifeng Zhang, and Shilin Zhou. Z-image: An efficient image generation foundation model with single-stream diffusion transformer, 2025.
- [16] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [17] Xiangxiang Chu, Jianlin Su, Bo Zhang, and Chunhua Shen. Visionllama: A unified llama backbone for vision tasks, 2024.
- [18] Zeyu Lu, Zidong Wang, Di Huang, Chengyue Wu, Xihui Liu, Wanli Ouyang, and Lei Bai. Fit: Flexible vision transformer for diffusion model, 2024.
- [19] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour, 2017.
- [20] Hongyi Zhang, Yann N. Dauphin, and Tengyu Ma. Fixup initialization: Residual learning without normalization, 2019.
- [21] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023.

- [22] Jie Zhu, Mingyu Ding, Boqiang Duan, Leye Wang, and Jingdong Wang. Unveiling the secret of adaln-zero in diffusion transformer. https://openreview.net/forum?id=E4roJSM9RM, 2025. ICLR 2025.
- [23] Noam Shazeer. Glu variants improve transformer, 2020.
- [24] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022.
- [25] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2022.
- [26] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge, 2015.
- [27] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.
- [28] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025.
- [29] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis, 2021.
- [30] Qiao Sun, Zhicheng Jiang, Hanhong Zhao, and Kaiming He. Is noise conditioning necessary for denoising generative models? In Proceedings of the 42nd International Conference on Machine Learning, 2025.
- [31] Mojtaba Sahraee-Ardakan, Mauricio Delbracio, and Peyman Milanfar. The geometry of noise: Why diffusion models don’t need noise conditioning, 2026.
- [32] Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. Understanding the difficulty of training transformers, 2020.
- [33] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. Deepnet: Scaling transformers to 1,000 layers, 2022.
- [34] Chen Chen and Lai Wei. Post-layernorm is back: Stable, expressive, and deep, 2026.
- [35] Sho Takase, Shun Kiyono, Sosuke Kobayashi, and Jun Suzuki. Spike no more: Stabilizing the pre-training of large language models, 2023. Published at COLM 2025.
- [36] Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, and Simon Kornblith. Small-scale proxies for large-scale transformer training instabilities, 2023.
- [37] Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth, 2021.
- [38] Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, and Aurelien Lucchi. Signal propagation in transformers: Theoretical perspectives and the role of rank collapse, 2022.
- [39] Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, and Josh Susskind. Stabilizing transformer training by preventing attention entropy collapse, 2023.
- [40] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2019.
- [41] Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer, 2022.
- [42] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

- [43] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost, 2016.
- [44] Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for transformers, 2020.
- [45] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, Rodolphe Jenatton, Lucas Beyer, Michael Tschannen, Anurag Arnab, Xiao Wang, Carlos Riquelme, Matthias Minderer, Joan Puigcerver, Utku Evci, Manoj Kumar, Sjoerd van Steenkiste, Gamaleldin F. Elsayed, Aravindh Mahendran, Fisher Yu, Avital Oliver, Fantine Huot, Jasmijn Bastings, Mark Patrick Collier, Alexey Gritsenko, Vighnesh Birodkar, Cristina Vasconcelos, Yi Tay, Thomas Mensink, Alexander Kolesnikov, Filip Paveti´c, Dustin Tran, Thomas Kipf, Mario Luˇci´c, Xiaohua Zhai, Daniel Keysers, Jeremiah Harmsen, and Neil Houlsby. Scaling vision transformers to 22 billion parameters, 2023.
- [46] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness, 2022.
- [47] Philippe Tillet, H. T. Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, MAPL 2019, page 10–19, New York, NY, USA, 2019. Association for Computing Machinery.
- [48] Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free, 2025.
- [49] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks, 2023.
- [50] Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks. https://kellerjordan.github. io/posts/muon/, 2024.
- [51] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for llm training, 2025.
- [52] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2023.
- [53] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization, 2023.
- [54] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2023.
- [55] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models, 2022.
- [56] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

### A Diagnostic Metrics and Definitions

Table 2 provides the mathematical definitions for all diagnostic metrics referenced in our analysis. Spatially coherent metrics are estimated robustly on sampled token subsets during the live training pass.

Table 2: Diagnostic Metrics Glossary. Definitions for all diagnostic metrics referenced in our analysis. X denotes forward representations; A denotes attention matrices; ∆ denotes backward gradients.

Metric Formal Definition Description

Writer GMD (Gmean, Gctr) ∥∆Wµ∥F , ∥∆Wc∥F Frobenius norms of the matrix components ∆Wµ = Tδ¯y¯⊤ and ∆Wc = t δ˜ty˜t⊤, decoupling the mean-coherent and centered components of the writer-weight update.

Q/K Grad Norm (G(Q), G(K)) RMS(∇WQ,KL) Root-mean-square of the gradients with respect

to the query and key projection weights. Energy Ratio (ρT ) ∥∥c(µX(X)∥)∥F

F +ϵ Ratio between the energy of the mean component µ(X) and the centered component c(X) of the token representation.

TR Ratio (rTR) ∥∥UX(∥X)∥F

F +ϵ Ratio between the residual-branch update U(X)

and the residual-stream state X. Variance Gain (VarGain) ∥∥cc((UX()X∥))∥F

F +ϵ Ratio between the centered energy of the branch update c(U(X)) and the centered energy of the input c(X).

Attn Contraction (µeff) ∥PAP∥2 (centered power iter.) Spectral norm of the attention operator restricted to the centered subspace, where P = I − J projects out the token mean.

Row Diversity (RowDiv) ∥A∥−AJA∥ ∥F

Relative deviation of the attention rows from their column-mean profile.

F

Centered Retention (Ret(c ← c)) ∥c∥(cA(X·c)(∥X))∥F

F +ϵ Fraction of centered input energy that remains in the centered subspace after one attention operation.

Mean Leakage (Leakage(µ ← c)) ∥µ∥(cA(X·c)(∥X))∥F

F +ϵ Fraction of centered input energy mapped into the mean subspace by the attention operator via JAP.

Token Cosine Similarity (TCS) Ei̸=j[cos(Xi, Xj)] Average pairwise cosine similarity between token representations, estimated on sampled token pairs when needed. High values indicate token homogenization.

### B Standard Initialization Enters the Same Mean-Dominated State

The main diagnostic runs use zero-initialized residual writers, a standard identity-start choice for deep residual networks. This choice keeps the early trajectory well behaved and makes the delayed MMS event easy to isolate. We now ask how the same backbone behaves when residual writers are initialized open from the first step. To test this, we train a 128-layer single-stream DiT with no residual gating and Gaussian initialization, W ∼ N(0,0.022) [40], for the residual writers and final projection. This control is not a matched convergence comparison; it tests whether the mean-dominated state is specific to the identity-start writer schedule.

Gmean vs Gctr (writers)

Collapse front (TCS)

Depth profile

0.06

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| |Ret.(c c)<br><br>VarGain| | | | | | | | | | | | |
| |RowDiv| | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| |[Figure 11]| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

112

0.9

10 2

96

1.0

0.05

[Figure 12]

0.8

80

Gradientnorm(logscale)

TokenCosineSimilarity

10 3

0.8

64

Retention/VarGain

0.04

0.7

Layerdepth

48

0.6

RowDiv

10 4

0.03

32

attractor zone

0.6

16

0.4

0.02

10 5

8

0.5

4

0.2

Gmean (WO)

0.01

0.4

Gctr (WO)

- 0
- 1
- 2

10 6

0.0

Gmean (W2)

0.00

0.3

Gctr (W2)

10 7

10 120 230 350 460 570 690

01248163248648096112

01248163248648096112

Training step

Layer depth

Layer depth

Figure 7: Standard-initialization control for a 128-layer DiT. (a) Token cosine similarity (TCS) over training steps and layer depth. The dashed white contour marks TCS = 0.9. (b) Depth profiles of centered retention Ret(c←c), centered branch replenishment VarGain, and attention row diversity RowDiv; curves report the median over diagnostic checkpoints from steps 10–690. (c) Median writer-gradient decomposition for the attention output projection WO and FFN output projection W2, reporting Gmean and Gctr across depth.

- Figure 7 shows that standard initialization enters the same mean-dominated regime. The temporal pattern differs from the zero-writer runs: instead of a delayed writer-gradient spike after an initially stable period, deep layers have high token similarity from the beginning of training, and the highsimilarity region forms a depth-wise collapse front. In this run, the loss quickly reaches a high plateau, consistent with a residual stream that has lost most of its token-varying information in deep layers.

The forward diagnostics connect this behavior to the same subspace mechanism studied in the main text. In deep layers, centered retention and branch-side centered replenishment are both small, so centered variation is not maintained through depth. RowDiv remains nonzero, ruling out the simpler explanation that attention has literally collapsed to identical rows. The failure is instead a subspace imbalance: row-stochastic attention preserves pure-mean states, while the centered component is weakly retained and weakly replenished.

The writer-gradient decomposition shows the same imbalance on the backward path. In the deep collapsed layers, the mean-coherent writer component Gmean dominates the centered component Gctr by several orders of magnitude for both WO and W2. Thus the standard-init run reaches the same endpoint as the zero-writer MMS runs: token variation is suppressed and residual writer updates become mean-dominated.

Writer initialization therefore changes the temporal presentation of the failure, not the subspacelevel failure mode. Identity-start runs expose the failure as a delayed MMS event, while standard initialization exposes it as an early depth-wise collapse front. In both cases, the common failure channel is the imbalance between the invariant pure-mean direction and insufficient centered-subspace maintenance. MV-Split therefore targets the residual writer geometry rather than a peculiarity of the zero-writer schedule.

### C Derivations of Writer Gradient Scaling

#### C.1 Proof of the Gradient Mode Decomposition

In Section 4.1, we state that the gradient admits an exact additive decomposition into a rank-1 meancoherent component ∆Wµ and a centered variation component ∆Wc, with cross-terms identically vanishing. We provide the brief algebraic proof here.

Let the forward input yt ∈ Rn and the backward gradient δt ∈ Rm for token t be decomposed into their sequence means and zero-mean centered components:

T

1 T

yt = y¯ + y˜t, where y¯ =

yt,

t=1

T

1 T

δt = δ¯+ δ˜t, where δ¯ =

δt,

t=1

T

y˜t = 0, (11)

t=1

T

δ˜t = 0. (12)

t=1

The parameter gradient for a token-wise linear map W is the sum of outer products over the sequence length. Substituting the decomposed terms yields:

∇WL =

T

δtyt⊤ =

t=1

T

t=1

δ ¯+ δ˜t y ¯ + y˜t ⊤. (13)

Expanding the outer product gives four summation terms:

∇WL =

T

δ¯y¯⊤ +

t=1

T

δ¯y˜t⊤ +

t=1

T

δ˜ty¯⊤ +

t=1

T

δ˜ty˜t⊤. (14)

t=1

Because the sequence means y¯ and δ¯ are constant across tokens, they can be factored out of the summations for the cross-terms:

T

T

δ¯y˜t⊤ = δ¯

y˜t⊤

= 0, (15)

t=1

t=1

= 0⊤

T

T

δ˜ty¯⊤ =

δ˜t

y¯⊤ = 0. (16)

t=1

t=1

= 0

The cross-terms evaluate identically to zero matrices, and the first term sums to T δ¯y¯⊤. The gradient therefore admits an exact additive decomposition into a mean-coherent rank-1 component (∆Wµ) and a centered component (∆Wc):

T

δ˜ty˜t⊤

∇WL = T δ¯y¯⊤

. (17)

+

t=1

∆Wµ

∆Wc

This recovers Equation 5 as an algebraic identity rather than an approximation.

#### C.2 Derivation of the Alignment-Amplification Law We derive Eq. (6) and its equal-magnitude specialization. Exact expansion and diagonal–off-diagonal split. Let W be a token-wise linear map with gradient

∇WL =

T

δtyt⊤ ∈ Rm×n, (18)

t=1

where yt ∈ Rn is the forward input and δt ∈ Rm is the corresponding backward gradient for token t. Using the Frobenius inner-product identity for rank-1 matrices,

⟨ab⊤, cd⊤⟩F = ⟨a,c⟩⟨b,d⟩, we obtain

T

T

T

T

δtyt⊤,

δsys⊤

∥∇WL∥2F =

⟨yt,ys⟩⟨δt,δs⟩. (19) Separating diagonal and off-diagonal terms gives

=

F

t=1

s=1

t=1

s=1

T

∥∇WL∥2F =

∥yt∥2∥δt∥2

. (20)

⟨yt,ys⟩⟨δt,δs⟩

##### +

t=1

s̸=t

S

C

Here S is the diagonal, no-interaction baseline, while C collects the cross-token interference terms. Rayleigh-quotient form. Define the per-token magnitude

wt ≜ ∥δt∥∥yt∥, w ∈ RT≥0, (21) and the pairwise alignment matrix

##### Mts ≜ cos(yt,ys) cos(δt,δs). (22) Then Eq. (19) becomes

∥∇WL∥2F = w⊤Mw, (23) while the diagonal baseline is

T

wt2 = ∥w∥22, (24) since Mtt = 1. Therefore the alignment amplification (cf. main text Eq. 6) is

S =

t=1

w⊤Mw ∥w∥22

A ≜ ∥∇WL∥2F S

, (25) and subtracting the diagonal baseline yields

=

w⊤(M − I)w ∥w∥22

wtwsMts

= s̸=t

. (26) This is exactly Eq. (6) in the main text.

A − 1 =

T t=1 wt2

Moreover, M is positive semidefinite. Indeed, it is the Hadamard product of the cosine Gram matrix of {yt} and that of {δt}; both are positive semidefinite, and the Schur Product Theorem preserves positive semidefiniteness. Thus A is a Rayleigh quotient of a PSD matrix.

Equal-magnitude specialization. Suppose the per-token magnitude is approximately constant,

wt ≡ w0. (27) Then Eq. (26) reduces to

1 T s̸=t

Mts. (28) Define

A − 1 =

κ ≜ Es̸=t[Mts] = Es̸=t cos(yt,ys)cos(δt,δs) , (29) where Es̸=t denotes the uniform average over ordered pairs (s,t) with s ̸= t. Then

A − 1 = (T − 1)κ. (30) When κ ≈ 0, the cross-terms cancel on average and

√

∥∇WL∥2F = O(T), ∥∇WL∥F = O(

T). When κ → 1, the accumulation becomes coherent and

##### ∥∇WL∥2F = O(T2), ∥∇WL∥F = O(T).

#### Absolute-coherence upper bound. From Eq. (26),

wtws|Mts|

|A − 1| ≤ s̸=t

. (31)

T t=1 wt2

Under the equal-magnitude approximation, this becomes

|A − 1| ≤ (T − 1) ˆκ, κˆ ≜ Es̸=t |cos(yt,ys)||cos(δt,δs)| . (32)

In the main-text experiment, κˆ is used as an absolute-coherence proxy. The gap between (T − 1)ˆκ and the signed quantity A − 1 reflects signed cancellation, together with any looseness introduced by the absolute-value relaxation.

- C.3 Proof of Lemma 1 Let ai ∈ RT denote the i-th attention row written as a column vector, so that

T

aijVj, ai = softmax(Si). (33)

Yi =

j=1

By the chain rule,

∂L ∂Yi

∂L ∂aij

, Vj . (34) If Vj = v¯ for all j, then the right-hand side is independent of j. Hence

=

∂L ∂ai

= γi1, γi ≜

∂L ∂Yi

, v¯ . (35)

The softmax Jacobian at ai is

Jsm(ai) = diag(ai) − aia⊤i , (36) and satisfies

Jsm(ai)1 = ai − ai(1⊤ai) = 0, (37) since 1⊤ai = 1. Exploiting the symmetry of the Softmax Jacobian,

∂L ∂Si

∂L ∂ai

∂L ∂ai

= Jsm(ai)⊤

= γi Jsm(ai)1 = 0. (38) Since this holds for every row i, the logit gradient vanishes identically, which proves Lemma 1.

= Jsm(ai)

Residual-writer gradients bypass the Softmax null space. The null-space argument above concerns only the gradient through the attention logits Si, and therefore the Q/K pathway. It does not zero the attention output projection. If Hi = j aijVj denotes the pre-WO attention output and gi is the upstream adjoint at the output projection, then

##### ∇WOL =

i

giHi⊤, (39)

which bypasses the Softmax Jacobian. Under value homogenization, Hi → H¯ becomes approximately token-constant, so the writer gradient becomes mean-coherent rather than zero. Gradients to the value pathway can also remain nonzero: the strict null-space extinction applies to the logit, and hence to the Q/K pathway, only.

### D Detailed Comparison: MV-Split vs. LayerScale and ReZero

LayerScale [6] parameterizes the merge as ZlLS = Xl+λl⊙Fl with λl ∈ RD a per-channel learnable vector; ReZero [5] is the single-scalar special case λl ∈ R initialized at zero. We show that both differ from MV-Split in three structural respects, each of which corresponds to a specific failure mode in Sec. 4.

#### 1. Open-loop vs. leaky-integrator mean dynamics. Projecting the two merges into the mean subspace via J:

JZlLS = JXl + (λl ⊙ JFl), JZlMV = (1 − α) ⊙ JXl + α ⊙ JFl. (40)

LayerScale leaves the trunk’s mean component untouched at every layer (the coefficient of JXl is identically 1): it does not damp the carried trunk mean and only scales newly injected branch updates. MV-Split contracts the trunk’s mean by 1 − α before each injection, which is a leaky integrator whenever αd ∈ (0,1). The two cannot be made equivalent by any choice of λl: taking Fl ≡ 0 gives JZlLS = JXl while JZlMV = (1 − α) ⊙ JXl, so the dynamics differ whenever α ̸= 0, irrespective of λl.

#### 2. Isotropic vs. anisotropic gain on the residual branch. By Eq. 5 the gradient decomposes

- as ∇WL = ∆Wµ + ∆Wc with ∥∆Wµ∥F ∼ Tκˆ in the coherent regime and ∥∆Wc∥F scaling diffusively under weak centered alignment. In the scalar-gain simplification, both modes are scaled by the same gain:

∥∆WµLS∥F ∥∆WcLS∥F

∝

√

Tκ.ˆ (41)

For scalar gates like ReZero, the ratio is exactly invariant. LayerScale uses a per-channel diagonal operator λl ∈ RD; while feature-wise anisotropy can incidentally alter the mean/centered ratio, it provides no structural token-subspace filter. MV-Split scales the two modes by α and β independently. In the scalar-gain simplification:

∥∆WµMV∥F ∥∆WcMV∥F

∝

α β

√

Tκ,ˆ (42)

allowing the unstable-to-stable ratio α/β to be reduced without coupling to the absolute centered gain β. If the leaky mean replacement term were removed (i.e. replacing α ⊙ J(Fl − Xl) in Eq. 7 by α ⊙ JFl), the special case α = β would reduce to a LayerScale-like token-space isotropic branch gain. With the leaky term in Eq. 7, however, MV-Split remains dynamically distinct from LayerScale even when α = β, because JZlMV = (1−α)⊙JXl +α⊙JFl contracts the trunk’s mean component

- at every layer (cf. §1 above), whereas LayerScale leaves it untouched.

###### 3. Independent gain on the centered path. Whatever absolute gain on the centered branch is needed for stability at a given depth, MV-Split treats it as a free parameter set independently of α (Eq. 9). LayerScale ties the two paths to the same token-independent per-channel gain, so any reduction in the mean-coherent contribution unavoidably reduces centered replenishment by the same factor.

Sec. 4 describes a self-reinforcing failure: a gradient spike injects mean-mode content into the trunk, the trunk’s mean direction aligns with the residual branch’s coherent direction, and this alignment amplifies the next mean-coherent update. The two MV-Split gates intervene at two different points: α contracts the trunk’s mean component at every layer (Eq. 9), bounding mean accumulation; the α/β gap damps the mean-coherent update relative to the centered update (Eq. 10), interrupting the alignment-amplification step. A scalar gate scales both modes equally, while LayerScale applies no explicit token-subspace filter; neither contracts the carried trunk mean.

### E Segment-wise Projectors for Multimodal Sequences

For a sequence partitioned into image tokens I and text tokens T , we define group-mean projectors that average within each segment only:

Jseg = blkdiag(JI,JT ), Pseg = I − Jseg, (43)

where JI = |I|1 1I1⊤I and similarly for JT . The projector does not average across modalities; it prevents the residual merge from directly mixing image and text means through the mean operator, preserving modality-specific mean scales in the residual control path. The diagnostic and mechanistic derivations in the main text use the global sequence-mean projector; the segment-wise projector is used only in the multimodal MV-Split residual merge.

Segment-wise control still acts on the global mean mode. Let Jg = T1 11⊤ denote the global projector. The global mean subspace is contained in the segment-wise mean subspace, so

Jg Jseg = Jg, Jg Pseg = 0. (44)

Because the gates α,β are feature-wise and broadcast across tokens, they commute with the token projectors. Applying Jg to the pre-normalization MV-Split merge with J,P instantiated as Jseg,Pseg (Eq. 7) gives

JgZl = JgXl + β ⊙ Jg PsegFl + α ⊙ Jg Jseg(Fl − Xl) = (1 − α) ⊙ JgXl + α ⊙ JgFl. (45)

Thus the segment-wise implementation applies the same leaky control to the global MMS mode that the main-text theory analyzes, while avoiding direct averaging of image and text means in the residual control path.

### F Step-Level Gradient Trace for Failure Attribution

The main text analyzes MMS through token-space and gradient decompositions. This appendix describes the inline trace that audits the representative gradient spike before those subspace diagnostics are applied: it rules out data/loss-side artifacts, localizes where large gradients appear, and identifies which internal quantities to measure next.

Trace protocol. Figure 8 summarizes the pipeline. When the global gradient norm crosses the threshold, the training loop records (i) per-rank loss, loss weight, and output-gradient statistics, (ii) a NaN/Inf scan over stored parameters, (iii) distributed gradient norms grouped by layer and parameter family, and (iv) at instrumented residual writers, the mean-coherent and centered components from Eq. 5.

For a parameter family τ at layer l, the grouped norm aggregated across R distributed ranks is

 

 

1/2

R

∥∇θLr(t)∥2F

. (46)

Gl,τ(t) =

r=1 θ∈Θl,τ

This top-K grouping localizes which parameter families receive large gradients at the detected step; it does not identify the responsible token-space mode.

- 1. Spike detection

Grad norm > adaptive threshold

- 2. Per-layer autopsy

Top-K params by grad Frobenius norm

- 3. Root cause exclusion Cross-rank context audit

Excluded hypotheses

Per-rank loss normal

Input grad RMS normal

No NaN/Inf in params All clear → internal mechanism

- 4. Type classification

Dominant param → failure mode

Attn_WO / FFN_W2 / Norm

- Figure 8: Step-level gradient trace pipeline. A global-norm threshold (1) triggers a per-family top-K ranking of distributed gradient norms (2) and a cross-rank exclusion audit (3) that checks per-rank loss agreement, final-output-gradient RMS, and NaN/Inf in stored parameters. When all three exclusions pass, the dominant top-K parameter family at the detected step (4) is recorded for the gradient-mode audit and subsequent paragraphs.

Data/loss-side checks. At the detected steps in the representative trace, per-rank losses remain clustered and the final-output-gradient statistics stay small at the printed precision. The maximum per-sample output-gradient norm is also nearly identical across ranks. This rules out two simple explanations: a single-rank data outlier and a global loss-weighting jump. The parameter scan finds no NaN/Inf values in stored parameters, so the event is not explained by persistent parameter corruption.

Parameter-family localization. We examine the representative trace in the lower-learning-rate baseline (Base η/2, no MV-Split, no LayerScale), where divergence at Step 26423 offers a longer pre-spike window than the default-LR run in Section 6.1 (Base η, t⋆=3400). The first warning

snapshots are mixed: their largest entries include embedding/final parameters, Q/K/V projections, FFN input weights, and residual output projections (Figure 9, left). We therefore do not interpret these early warnings as a fixed shallow-layer mechanism. At the escalation step, the largest printed entries shift toward residual output interfaces, with Attn_WO accounting for most of the top-K squared-norm mass and FFN_W2 also appearing. This localization motivates auditing the residual writers directly rather than attributing the event to the output head, the batch, or a specific attention-logit pathology.

###### Top-15 params at onset (Step 26423)

###### Escalation over 14 steps (all ranks uniform)

Step 26423 (onset)

3.0

0.868 0.686

Embed

=0.007

Step 26430 (+7)

Step 26434 (+11) Step 26437 (+14)

L2 Q_Proj L0 FFN_W13 L1 FFN_W13

0.671 0.487

2.5

0.266 0.256

L0 Q_Proj

L0 FFN_W2 L1 V_Proj L2 K_Proj

Per-rankloss

0.232 0.222

2.0

=2.09 loss collapse

0.203 0.202 0.198 0.195 0.191

L0 Attn_WO L1 Attn_WO

1.5

L0 V_Proj L0 K_Proj L1 Q_Proj

Attn WO FFN W2 FFN W13 WQ/WK/WV Embed / Final

| |
|---|

0.170 0.121

L1 FFN_W2 L1 K_Proj

1.0

| |
|---|

| |
|---|

=0.005

| |
|---|

0.0 0.2 0.4 0.6 0.8 1.0

0 1 2 3 4 5 6 7

Gradient Frobenius norm

GPU rank

#### Figure 9: Step-level gradient trace at a representative spike (400 layers, Base η/2). (Left) Top

parameter-family gradient norms Gl,τ at one detected step (Step 26423). The top-K entries span embedding/final parameters, Q/K/V projections, FFN input weights, and residual output projections. (Right) Per-rank loss across four snapshots (Steps 26423, 26430, 26434, 26437). The eight ranks stay tightly clustered (σ ∈ [0.005,0.007]) and the maximum per-sample output-gradient norm

Mout(r) ≈1×10−4 matches across ranks (inset).

Gradient-mode audit. For each instrumented writer, the trace caches the writer input yt during the forward pass and the output adjoint δt during the backward pass. It then computes

∆Wµ = Tδ¯y¯⊤, ∆Wc =

t

δ˜ty˜t⊤, (47)

and reports Gmean = ∥∆Wµ∥F and Gctr = ∥∆Wc∥F. This is the measurement that links the top-K localization to the mechanism in Section 4.1: at the spike, the writer update is amplified in the mean-coherent component, while the centered component does not show a comparable increase.

Attention-branch-only control. To test whether protecting the attention branch alone is sufficient, we apply MV-Split only to the attention output residual branch and leave the FFN residual branch unchanged. We tested this on a 1000-layer configuration. The training still spikes at Step 7415 (Global Norm ≈ 0.665), and the largest printed gradients move to the unprotected FFN branch: Attn_WO disappears from the top-K entirely, while FFN_W2 accounts for 14 of the top 15 contributors and about 93% of the top-K squared-norm mass at this step (Figure 10). Depending on the step, these entries can include both the FFN input transformation and the FFN output projection.

The subsequent escalation is rapid. Within four steps (7415→7419), per-rank loss rises uniformly from ∼0.81 to ∼2.12 across all 16 ranks, with per-step standard deviation remaining low (σ ∈ [0.011,0.023]), consistent with a global update event rather than a single-rank or single-batch fault.

The conclusion is therefore branch-level rather than weight-level: attention-only residual control is insufficient, and both attention and FFN residual branches require the mean/centered split.

###### Top-15 params at spike (Step 7415)

###### Escalation over 4 steps (all ranks uniform)

2.2

Step 7415 (trigger)

0.0721 0.0713

- L45 FFN_W2

- L41 FFN_W2 L49 FFN_W2 L43 FFN_W2 L47 FFN_W2

Final L46 FFN_W2

- L36 FFN_W2 L51 FFN_W2 L40 FFN_W2 L38 FFN_W2

- L37 FFN_W2

- L42 FFN_W2 L32 FFN_W2 L55 FFN_W2

=0.011

- Step 7417 (+2)

- Step 7418 (+3)

- Step 7419 (+4)

2.0

0.0699 0.0696 0.0692

1.8

0.0685 0.0682

Per-rankloss

1.6

=1.31 loss collapse

0.0656

1.4

0.0648 0.0645 0.0642 0.0637 0.0637 0.0634 0.0633

| |
|---|

| |
|---|

| |
|---|

1.2

1.0

FFN W2 (writer)

=0.023

0.8

Embed / Final

0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08

0 3 7 11 15

Gradient Frobenius norm

GPU rank

- Figure 10: Attention-branch-only MV-Split control (1000 layers). (Left) Top parameter-family gradient norms at the detected step (Step 7415). With the attention output branch protected, no Attn_WO entries appear in the top-K; the largest printed entries are FFN_W2. (Right) Per-rank loss over four consecutive steps (7415–7419). Cross-rank losses stay tightly clustered (σ ∈ [0.011,0.023]) while the loss rises uniformly from ∼0.81 to ∼2.12.

### G Training Configuration Details

Table 3 summarizes the architecture and optimization hyperparameters for the four DiT configurations reported in this paper: DiT-400L-Baseline (400-layer Post-Norm without residual gating, used to characterize the Mean-Mode Screaming failure), DiT-400L-LayerScale (the matched 400-layer LayerScale control), DiT-400L-MVSplit (the matched 400-layer model with MV-Split residuals), and DiT-1000L-MVSplit (the 1000-layer text-to-image scale-up demonstration). The base learning rate for each run is obtained from the target value via µP [41] width-scaling, base = target/(0.2√dmodel) with target = 10−3 and dmodel = 1024, yielding base = 1.5625×10−4.

- Table 3: Architecture and training hyperparameters for the four DiT runs reported in this paper. The three 400-layer controls share the same backbone, optimizer, batch size, and non-residual primitives, and vary in residual stabilization and initialization protocol. The 1000-layer run uses the same backbone family and MV-Split residual design, but differs in depth, hardware scale, and post-training pipeline. In MV-Split, α and β are unconstrained learnable vectors; empirically, α remains in [0,1] throughout training in all reported MV-Split runs. For the 1000-layer run, βinit is set

√

to 0.03 ≈ 1/

L following standard depth-variance scaling; the MMS protection itself stems from the anisotropic split (αinit=0 < βinit) rather than from isotropic shrinkage. Field DiT-400L-Baseline DiT-400L-LayerScale DiT-400L-MVSplit DiT-1000L-MVSplit

Pretraining Dataset ImageNet-2012 Image Autoencoder Frozen FLUX.2 VAE Text Encoder Frozen Qwen3-0.6B Trainable Components DiT backbone only Training Hardware 8×H100 8×H100 8×H100 16×H100 Post-training Dataset — — — ∼50k curated images

DiT Params 5.45B 5.45B 5.45B 13.64B Layers 400 400 400 1000 Residual Mode None LayerScale Mean-Variance Split Mean-Variance Split Residual Gates — learnable λ, λinit ∈

learnable α, β; αinit=0, βinit=0.03 Learning Rate 1.5625×10−4,

learnable α, β; αinit=0, βinit=1

{10−2, . . . , 10−5}

1.5625×10−4 1.5625×10−4 1.5625×10−4

7.8125×10−5

Initialization Method zero init WO, W2; standard init others

standard init, N(0, 0.022)

zero init WO, W2; standard init others

zero init WO, W2; standard init others

Dimension (dmodel) 1024 FFN Dimension 3072 FFN Type SwiGLU Attention Heads 8 Attention Head Dim 128 KV Heads 8 Attention Type MHA Position Embedding 2D RoPE RoPE θ 10000 Layer Norm Type RMSNorm, non-affine RMSNorm Affine Gain disabled RMSNorm ϵ 10−6 QK-Norm ✓, non-affine

LR Scheduler warmup → constant Warmup Steps 1000 Global Batch Size 1024 Optimizer AdamW [42] AdamW Betas (0.9, 0.999) AdamW ϵ 10−8 Weight Decay 0.1 (2D Weights Only) Training Steps crashed 100k 100k 100k Gradient Clipping 1.0

### H The Token Mean as an Implicit Timestep Carrier

Our backbone has no AdaLN and no explicit timestep embedding, so the model must infer the continuous rectified-flow time t from the noisy latent itself. In this appendix, we use “timestep” to refer to this continuous interpolation coordinate; equivalently, it is the noise-level coordinate controlling zt = (1 − t)x0 + tx1 from data latent to Gaussian noise.

We run a post-hoc linear probe on ImageNet-2012 validation images. Each image is encoded into the same VAE latent space used during training. We sample x1 ∼ N(0,I) and t ∼ U[0,1], form zt = (1 − t)x0 + tx1, and record hidden states from the trained 400-layer MV-Split checkpoint. For each probed layer, we fit ridge regressors to predict t from the image-token mean mimgl , a centered image-token RMS summary rms(cimgl ), and the text-token mean mtxtl . Train/test splits are grouped by image id; scalar input-statistic, shuffled-label, and untrained-model controls are included. For panel (b), we report the fraction of residual squared error left by scalar input statistics that is removed by adding a hidden-state feature, 1 − SSEinput+h/SSEinput.

- (a)

Linear decodability of noise level

| | | || |
|---|
<br><br>| | |
|---|---|---|---|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |0.998| |
| | | || |
|---|
| | |
| | | | |0.70| |
| | | | || |
|---|
| |
| | | | || |
|---|
| |
| | | | | | |
| | | | | | |

0.00 0.25 0.50 0.75 1.00

Relative depth /L

0.2

0.0

0.2

0.4

0.6

0.8

1.0

Residualvarianceremoved

(b)

Information beyond input statistics

0.00 0.25 0.50 0.75 1.00

Relative depth /L

10 2

10 1

ProbeMAE(logscale)

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

| |
|---|

input stats MAE=0.17

(c)

Probe error

|mimg rms(cimg) mtxt trained untrained<br><br>|
|---|

Figure 11: Real-image timestep linear probe. The backbone has no explicit timestep embedding or AdaLN modulation. (a) The trained image-token mean mimg predicts t with near-perfect linear R2 across depth. The text-token mean mtxt becomes predictive after a few single-stream layers, indicating that the trained model routes image-derived timestep information into the text-token side.

- (b) Adding hidden-state summaries removes nearly all residual error left by scalar input statistics. (c) Probe MAE (mean absolute error) shows the same pattern on a log scale. The result shows that the token mean is not merely a collapse-prone direction: it is also a useful global timestep carrier. The same coordinate is also decodable from centered-energy summaries, so the claim is not uniqueness of the mean subspace but its usefulness and stability as a global-state path.

2RtLinear-probe(predicting)

1.0

| |
|---|

| |
|---|

| |
|---|

0.8

0.6

| |
|---|

0.4

input stats R2 =0.41

0.2

0.0

shuffled labels: |R2|<0.03

0.00 0.25 0.50 0.75 1.00

Relative depth /L

- Figure 11 shows that the trained image-token mean predicts t with near-perfect linear R2 across depth. The trained text-token mean becomes predictive after only a few single-stream layers, indicating that image-derived timestep information is routed into the shared multimodal sequence. A randomly initialized network already exposes substantial decodability in early image-token states, showing that the timestep is structurally available from the input latent; training preserves this signal through depth and routes it to the text-token side.

The token mean is therefore not only a collapse-prone direction but also a useful global-state carrier for the timestep. MV-Split preserves the trunk mean while gain-limiting new mean-path residual writes, controlling the dangerous mean-coherent writer channel without erasing this useful global state.

### I System Implementation: Triton Fusion of RoPE, QK-Norm, SwiGLU, and MV-Split+RMSNorm

At the 8–16 H100 scale used in this work, training the ultra-deep DiT requires activation checkpointing [43] to fit in GPU memory. Checkpointing reduces activation memory, but it also replays checkpointed blocks during the backward pass. As a result, lightweight per-block operators such as RoPE, QK-Norm [44, 45], SwiGLU [23], and MV-Split+RMSNorm are executed far more frequently than in a forward-only view of the model, making their memory-bound overhead non-negligible in ultra-deep training [46].

Fused operators. We implement these operators in Triton [47]. For MV-Split+RMSNorm, an eager implementation materializes the pre-normalized residual state and launches separate kernels for segment-wise correction, residual merging, and RMS normalization. Given precomputed segment means, the fused kernel applies the segment-wise MV-Split update and the subsequent non-affine RMS normalization without materializing the pre-normalized intermediate residual. Its backward uses pointwise recomputation together with compact segment-wise sufficient statistics, rather than caching the full pre-normalized state across the checkpoint boundary.

Two-pass backward recomputation. Let Fl = fl(Xl) denote the residual branch output and let Zl be the pre-normalized MV-Split merge:

Zl = Xl + β ⊙ PsegFl + α ⊙ Jseg(Fl − Xl), Xl+1 = RMSNorm(Zl). (48)

The backward does not require caching Zl across the checkpoint boundary. Pass A recomputes Zl on chip, evaluates the RMSNorm adjoint, and accumulates the segment-wise sufficient statistics needed

for the input and gate gradients; Pass B then applies the closed-form gradients to Xl and Fl using the aggregated statistics. The full derivation is provided in Section I.1 below.

In-situ profiling. We evaluate the fused backend inside the distributed training loop rather than with isolated microbenchmarks. In our 8-GPU, 400-block profiling setup with activation checkpointing applied to three out of every four blocks, 300 blocks are replayed during backward. Consequently, RoPE, QK-Norm, and SwiGLU are each executed 700 times per active optimizer step, while MVSplit+RMSNorm is executed 1400 times.

Relative to a matched eager PyTorch baseline, the fused Triton backend reduces the aggregated self-CUDA time of these operators from 1697.4ms to 614.0ms per active optimizer step (2.76×). The individual reductions are from 359.7ms to 105.6ms for RoPE (3.41×), 255.2ms to 101.2ms for QK-Norm (2.52×), 118.3 ms to 27.9ms for SwiGLU (4.24×), and 964.2ms to 379.3ms for MVSplit+RMSNorm (2.54×). The explicit DiT forward range decreases from 1889.8ms to 1553.9ms, and the in-loop optimizer-step wall-clock decreases by 22.0%, from 5.87s to 4.58s, excluding dataloader wait. QKV projection and SDPA remain within a few percent under the same instrumentation, localizing the speedup to repeated normalization, activation, and residual-merge paths rather than to the main attention kernels.

I.1 Closed-form Backward of MV-Split+RMSNorm For token i in segment s(i), the pre-normalized merge can be written as

Zl,i = Xl,i + β ⊙ (Fl,i − F¯l(s)) + α ⊙ (F¯l(s) − X¯l(s)). (49) Let Gi = ∂L/∂Xl+1,i be the incoming gradient after RMSNorm, and let

ri = D 1 ∥Zl,i∥22 + ϵ −1/2

be the inverse-RMS factor. The pre-normalization adjoint ∆i = ∂L/∂Zl,i is

ri3 D ⟨Gi,Zl,i⟩ . (50)

∆i = riGi − Zl,i

Define the segment-wise mean adjoint

1 |s| i∈s

∆¯(s) =

∆i.

Then the merge gradients are

∂L ∂Xl,i

= ∆i − α ⊙ ∆¯(s(i)), (51) ∂L ∂Fl,i

= β ⊙ ∆i + (α − β) ⊙ ∆¯(s(i)). (52) The gate gradients are

∂L ∂α

∆i ⊙ (F¯l(s) − X¯l(s)), (53)

=

s i∈s

∂L ∂β

∆i ⊙ (Fl,i − F¯l(s)). (54)

=

s i∈s

These expressions require only pointwise recomputation and segment-wise sums, which is what the two-pass kernel above evaluates.

### J Methods we try but failed to prevent MMS

We tested several interventions that target related objects: token means, attention mixing, attentionoutput gating, scalar gradient-norm control, and optimizer-side update geometry. None of these controls removed the mean-dominated failure in this backbone; some additionally degraded optimization. Their common limitation is that they do not combine local mean/centered branch-gradient control with the forward leaky mean replacement used by MV-Split.

Hard centering and attention reparameterizations. Explicit centering, X ← PX, removes the token mean rather than gain-limiting new mean writes. This degraded optimization in our runs and also removes useful global information, including image-level context and the implicit timestep signal discussed in Appendix H. Attention-matrix modifications such as A − I, I − A, or (1 − λ)I + λA change the attention branch but do not protect the FFN branch or the residual merge. Moreover, row-stochastic interpolations still preserve pure-mean states, and in multimodal sequences global centering does not remove segment-wise mean modes (image and text groups may each become internally homogeneous while their global average remains zero). None of these implement the local branch-gradient split

G  → αJG + βPG, α ≪ β, nor the forward leaky mean replacement JXl  → (1 − α)JXl + αJFl that defines MV-Split. Gated attention. We also tested attention-output gates of the form

Yi = gi(Xi) ⊙ SDPA(Q,K,V )i,

where gi is computed token-locally and acts along the head or feature dimension. It is not a sequence-level token-space projector and does not form JY or PY . Such gates [48] can reduce attention-output magnitude and have been reported to mitigate attention-sink behavior [49], but they do not compute JY and PY or apply different gains to them. In the mean-dominated regime, tokens are already aligned, so gi ≈ g¯ tends to be similar across tokens; the gate then scales the mean and centered components together rather than separating them. More generally, attention-only controls leave the FFN W2 uncontrolled. Our attention-only MV-Split trace shows that, once the attention branch is protected, the spike can relocate to the remaining ungated FFN branch (Appendix F).

The gradient-clipping paradox. All runs in the main comparison use global gradient clipping with threshold 1.0 (Appendix G), yet this does not remove MMS. The reason is that MMS is not only a large-norm event; it is a directional collapse of the writer update into the token-mean subspace. For a writer gradient decomposed as G = Gµ + Gc, global norm clipping applies a scalar multiplier clipτ(G) = sG with s = min(1,τ/∥G∥), so

= ∥Gµ∥ ∥Gc∥

clipτ(G) = sGµ + sGc, ∥sGµ∥ ∥sGc∥

.

Clipping can reduce the step length, but it cannot rotate a mean-coherent writer update back into the centered subspace. When Gµ dominates, the same scalar shrinkage also suppresses the already-small centered update, leaving the feature-learning path starved. A global-norm safety check is also blind to subspace structure: it can remain quiet while the residual writer direction has already become structurally unsafe. MV-Split addresses this failure mode at the residual interface by applying different gains to JG and PG, rather than by thresholding the scalar norm of G.

Muon optimizer. Muon [50, 51] orthogonalizes the momentum/update matrix using Newton– Schulz iterations. This can remove singular-value scale from a matrix update, but it acts after token gradients have already been summed into G = t δtyt⊤ = Gµ + Gc. If the momentum is dominated by an isolated mean-coherent term Gµ = Tδ¯y¯⊤ = σuv⊤, the orthogonalized update removes σ but keeps the direction uv⊤. For homogeneous token inputs, this direction still produces the same update for every token and therefore remains mean-coherent. More generally, Muon reshapes singular values in parameter space; it does not implement the token-space split G  → αJG + βPG or the forward leaky mean replacement JXl  → (1 − α)JXl + αJFl used by MV-Split. In our runs, Muon could reduce update magnitude but did not remove the mean-dominated trajectory, consistent with MMS being a residual-subspace failure rather than only an optimizer-magnitude artifact.

### K Additional Results on the MV-Split Runs

#### K.1 Text-Conditioned Evaluation of the 1000-Layer Checkpoint

We report GenEval and DPG-Bench measurements for the post-trained 1000-layer MV-Split checkpoint as a calibration of text-conditioned generation ability. These numbers are not intended as a controlled comparison to large public text-to-image systems: our model is trained on substantially smaller and differently sourced data (ImageNet-2012 pretraining followed by SFT and DPO [52, 53] on ∼50k curated images), uses a shorter training schedule, and uses a simpler post-training pipeline. The purpose is only to confirm that the 1000-layer scale-validation run remains usable as a textconditioned generator, not to claim state-of-the-art text-to-image performance.

Table 4: Text-conditioned evaluation of the 1000-layer MV-Split checkpoint. Reported for calibration only and not used for state-of-the-art comparison.

Metric Score GenEval overall (avg. over tasks) 0.534 GenEval correct images 52.44% GenEval correct prompts 67.63% DPG-Bench overall 74.91

Table 5: GenEval task breakdown (1000-layer MV-Split checkpoint). Task Accuracy

single_object 92.81% two_object 63.64% counting 33.75% colors 72.61% position 25.75% color_attr 31.75%

#### K.2 Full-Horizon Training Loss Curve for the MV-Split Runs

2.0

###### Pretrain

###### SFT DPO

DiT-400Layer-MVSplit

DiT-1000Layer-MVSplit

1.5

DiT-1000Layer-MVSplit SFT

DiT-1000Layer-MVSplit DPO

1.0

Train/DPOLoss

0.8

0.6

0.4

0 20 40 60 80

100 105 110

Training Steps (k)

- Figure 12: Full-horizon training loss for the MV-Split 400-layer and 1000-layer runs. Note that the SFT and DPO stages use a separately curated ∼50k image set rather than the ImageNet-2012 pre-training distribution; since loss values are data-dependent, the curves are shown for reference only.

### L Limitations and Future Work

Our analysis identifies a residual-subspace failure pathway in ultra-deep Diffusion Transformers and shows that MV-Split stabilizes this pathway in the studied setting. The following boundary conditions define natural extensions rather than contradictions of the mechanism.

Predicting the exact onset time of MMS. The alignment-amplification law in Eq. 6 characterizes when token-wise writer gradients stop canceling and enter a coherent accumulation regime. This provides a mechanistic diagnostic for the MMS transition, but it does not by itself predict the exact training step t⋆ at which an un-stabilized run will cross the critical regime before the run is observed. The onset time depends on the coupled evolution of token representations, backward adjoints, optimizer momentum, data ordering, and mini-batch statistics. We therefore view exact onset prediction as a separate problem from architectural stabilization: MV-Split removes the unstable residual interface by controlling the mean and centered writer-gradient components directly, while deriving a closed-form scaling law for t⋆ remains an interesting direction for predictive theories of deep-network training dynamics.

Architectures beyond Softmax attention. Several parts of our analysis use Transformer-specific structure. In particular, row-stochastic attention preserves pure-mean token states (Proposition 1), and value homogenization suppresses Q/K logit gradients through the null space of the Softmax Jacobian (Lemma 1). These arguments do not directly transfer to attention-free sequence mixers such as convolutional diffusers or state-space models such as Mamba [54]. At the same time, the writer-gradient decomposition in Eq. 5 only assumes a token-wise residual writer and is not specific to Softmax attention. This suggests a broader question: which components of the mean-dominated collapse mechanism are consequences of attention, and which are more general consequences of ultra-deep residual streams with token-wise writers? Testing this distinction in convolutional, hybrid, and state-space diffusion backbones is a natural next step.

Extreme-context spatiotemporal generation. Our scale validation focuses on image and text-toimage diffusion. Video, 3D, and other spatiotemporal generators often operate with substantially longer token sequences and additional structure across time, views, or modalities. In the coherentalignment regime, the mean-coherent writer component can scale with sequence length as in Eq. 5, so these settings may place even stronger pressure on the residual interface. MV-Split is designed to decouple this mean-mode accumulation from the centered feature-learning path, but validating and possibly adapting the mechanism for ultra-long-context spatiotemporal DiTs remains an important direction for future large-scale generative modeling.

### M More Visual Results

We present additional uncurated samples from our 1000-layer MV-Split DiT to demonstrate the breadth and fidelity of the model across diverse semantic categories. All images are generated at 256 × 256 resolution using a Euler sampler [55] with 35 NFE steps and classifier-free guidance [56] scale w = 2.0.

Text-Conditioned Generation. Unlike class-conditional DiTs that condition on a one-hot class label, our model is a text-to-image generator. Each sample is conditioned on a natural-language caption drawn from the ImageNet-2012 validation set, where captions were generated by a modern large language model and describe the scene content in 10–25 words (e.g., “A colorful green jacamar on a branch with an insect in its beak, set against a blurred natural background” or “Snow-covered mountains under a dramatic cloudy sky, with sunlit huts and long shadows across the landscape”). The captions vary in viewpoint, lighting, composition, and context, providing a diverse conditioning signal that goes beyond categorical labels. Within each grid below, the 12 images correspond to 12 distinct captions from the same ImageNet class, showcasing the model’s ability to faithfully render varied scene descriptions. In each grid, the top row displays 4 images at 2× magnification for detail inspection, and the bottom row shows 8 additional samples at 1× scale.

[Figure 13]

Figure 13: Class “Alligator lizard” (044). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 14]

- Figure 14: Class “Scorpion” (071). Euler sampler, 35 NFE, CFG w = 2.0.

- Figure 15: Class “Jacamar” (095). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 16]

Figure 16: Class “Rhodesian ridgeback” (159). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 17]

Figure 17: Class “Bloodhound” (163). Euler sampler, 35 NFE, CFG w = 2.0.

Figure 18: Class “Bouvier des Flandres” (233). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 19]

- Figure 19: Class “White wolf” (270). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 20]

- Figure 20: Class “Chimpanzee” (367). Euler sampler, 35 NFE, CFG w = 2.0.

###### Figure 21: Class “Giant panda” (388). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 22]

###### Figure 22: Class “Beaker” (438). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 23]

###### Figure 23: Class “Caldron” (469). Euler sampler, 35 NFE, CFG w = 2.0.

###### Figure 24: Class “Candle” (470). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 25]

###### Figure 25: Class “Car wheel” (479). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 26]

###### Figure 26: Class “Coffeepot” (505). Euler sampler, 35 NFE, CFG w = 2.0.

###### Figure 27: Class “Convertible” (511). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 28]

###### Figure 28: Class “Crock Pot” (521). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 29]

###### Figure 29: Class “Drum” (541). Euler sampler, 35 NFE, CFG w = 2.0.

###### Figure 30: Class “Envelope” (549). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 31]

###### Figure 31: Class “Flute” (558). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 32]

###### Figure 32: Class “Freight car” (565). Euler sampler, 35 NFE, CFG w = 2.0.

###### Figure 33: Class “French horn” (566). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 34]

###### Figure 34: Class “Greenhouse” (580). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 35]

###### Figure 35: Class “Horse cart” (603). Euler sampler, 35 NFE, CFG w = 2.0.

###### Figure 36: Class “Knot” (616). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 37]

###### Figure 37: Class “Loupe” (633). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 38]

###### Figure 38: Class “Mask” (643). Euler sampler, 35 NFE, CFG w = 2.0.

###### Figure 39: Class “Minivan” (656). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 40]

###### Figure 40: Class “Mitten” (658). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 41]

###### Figure 41: Class “Monastery” (663). Euler sampler, 35 NFE, CFG w = 2.0.

- Figure 42: Class “Mountain bike” (671). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 43]

- Figure 43: Class “Pool table” (736). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 44]

Figure 44: Class “Pot” (738). Euler sampler, 35 NFE, CFG w = 2.0.

###### Figure 45: Class “Rugby ball” (768). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 46]

###### Figure 46: Class “Scoreboard” (781). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 47]

###### Figure 47: Class “Sweatshirt” (841). Euler sampler, 35 NFE, CFG w = 2.0.

###### Figure 48: Class “Teapot” (849). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 49]

###### Figure 49: Class “Trombone” (875). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 50]

###### Figure 50: Class “Windsor tie” (906). Euler sampler, 35 NFE, CFG w = 2.0.

###### Figure 51: Class “Alp” (970). Euler sampler, 35 NFE, CFG w = 2.0.

[Figure 52]

###### Figure 52: Class “Groom” (982). Euler sampler, 35 NFE, CFG w = 2.0.

