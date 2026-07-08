# arXiv:2605.14368v1[cs.CL]14May2026

## Where Should Diffusion Enter a Language Model? Geometry-Guided Hidden-State Replacement

Injin Kong1 Hyoungjoon Lee2 Yohan Jo1,† 1Graduate School of Data Science, Seoul National University 2Department of Biosystems & Biomaterials Science and Engineering, Seoul National University mtkong77@snu.ac.kr, hjoon721@snu.ac.kr, yohan.jo@snu.ac.kr

### Abstract

Continuous diffusion language models lag behind autoregressive transformers, partly because diffusion is applied in spaces poorly suited to language denoising and token recovery. We propose DiHAL, a geometry-guided diffusion–transformer hybrid that asks where diffusion should enter a pretrained transformer. DiHAL scores layers with geometry-based proxies, selects a diffusion-friendly hidden-state interface, and replaces the lower transformer prefix with a diffusion bridge while retaining the upper layers and original LM head. By reconstructing the selectedlayer hidden state rather than tokens, DiHAL avoids direct continuous-to-discrete recovery. Experiments on 8B-scale backbones show that the geometry score predicts effective shallow insertion layers under a fixed bridge-training protocol and that hidden-state recovery improves over continuous diffusion baselines in a diagnostic comparison matching the diffusion/recovery training budget. These results suggest that hidden–state geometry helps identify where diffusion–based replacement is feasible inside pretrained language models.

### 1 Introduction

Large language models have achieved remarkable progress across a wide range of language generation tasks, but this progress has come with increasing size and computational cost (Brown et al., 2020; Hoffmann et al., 2022; Yang et al., 2025). Diffusion models offer a different generative paradigm based on iterative denoising and have become a dominant approach in image generation (Song et al., 2021). Their success has motivated growing interest in diffusion-based language generation (Li et al.,

- 2022; Strudel et al., 2023; Nie et al., 2025). However, transferring diffusion from images to text is difficult because text generation must ultimately handle discrete tokens.

A natural response is to adapt diffusion to the discreteness of text. Prior work has explored discrete token corruption, masked diffusion, continuous-to-discrete recovery, and continuous diffusion over token embeddings, self-conditioned embeddings, or learned text latents (Li et al., 2022; Strudel et al.,

- 2023; Lovelace et al., 2023; Gong et al., 2023; Zhang et al., 2025). Despite these efforts, diffusionbased language models still lag behind autoregressive Transformers, particularly in continuous diffusion settings (Jo and Hwang, 2026). A common explanation is that denoised continuous vectors must eventually be mapped back to discrete tokens, so small errors in representation space can change the recovered token (Li et al., 2022).

Why does this gap remain? We start from a complementary hypothesis: discreteness is important but may not fully explain the gap. Transformer language models also use discrete tokens, yet most computation occurs in continuous hidden states later mapped to vocabulary logits (Vaswani et al., 2017). This suggests that the difficulty may arise not from continuity itself, but from applying diffusion in continuous spaces with unsuitable geometry.

†Corresponding author.

Preprint.

[Figure 1]

[Figure 2]

- Figure 1: Locate-and-Replace framework. Layer-wise geometric proxies score transformer layers, select an insertion point, and guide replacement with a diffusion block.

If the choice of continuous space matters, then the central question becomes: what makes a representation space suitable for diffusion? We call such a space diffusion-friendly: a space that is easy to denoise, stable under imperfect score estimates, and simple enough for diffusion to learn. We later motivate these requirements using tools from Langevin dynamics and concentration theory (Villani, 2009; Bakry et al., 2014; Ledoux, 2001).

Where can such a space be found in a language model? A pretrained Transformer already contains many continuous hidden spaces between the token embedding layer and the LM head. These hidden states are not decoded directly into tokens; they are consumed by the remaining Transformer layers before the LM head produces the final token distribution (Vaswani et al., 2017). Diffusion at an internal layer can therefore target hidden-state recovery rather than direct token recovery (Lovelace et al., 2023; Rombach et al., 2022). Since hidden-state geometry varies across depth, we ask: which transformer layer provides the most diffusion-friendly representation space?

To answer this question, we propose DiHAL (Diffusion-Transformer Hybrid Architecture for Language Generation), a hybrid architecture based on a Locate-and-Replace strategy. As illustrated in Figure 1, DiHAL locates diffusion-friendly layers using geometry-based criteria, then replaces the lower transformer layers with a diffusion bridge that reconstructs the selected-layer hidden state while retaining the upper layers and original LM head for token prediction. This reduces continuous-to-discrete recovery error and reframes continuous diffusion for language as a problem of choosing the right internal representation space for denoising. Our contributions are threefold.

- • We formulate diffusion insertion in pretrained transformer language models as a geometry-guided interface-selection problem and propose practical layer-wise proxies—local compactness, global stiffness, and effective rank—for identifying diffusion-friendly hidden spaces.
- • We introduce a fixed geometry score that narrows the search for effective insertion layers without exhaustive layer-wise bridge training and correlates strongly with hidden-state reconstruction quality under a one-epoch bridge-training protocol across 8B-scale backbones.
- • We introduce DiHAL, a Locate-and-Replace hybrid that replaces lower transformer layers with a conditional diffusion bridge and reuses the upper layers and LM head. Under a diagnostic diffusion/recovery budget, DiHAL shows that hidden-state recovery can improve generative perplexity and diversity over embedding-, latent-, and continuous-to-discrete interfaces.

### 2 Background

Transformer language models take discrete tokens as input, but most computation occurs in continuous hidden spaces. Given x1:T, an autoregressive model factorizes p(x1:T) = Tt=1 p(xt | x<t). Each token is mapped to an embedding h(0)t = E(xt)+pt, hidden states are updated as H(ℓ) = Fℓ(H(ℓ−1)), and the final state is projected to vocabulary logits zt = WLMh(tL) + b. Thus, discreteness appears at the input and output interfaces, while intermediate computation is continuous (Vaswani et al., 2017). Diffusion models generate samples by gradually adding noise to data and then learning to reverse this noising process (Song et al., 2021). In continuous space, this process can be written as a

stochastic differential equation dXt = f(Xt,t)dt + g(t)dWt, where Xt is the noisy representation at time t and Wt is Brownian motion. The reverse process depends on the score ∇x log pt(x), which is approximated by a neural network. Applying this idea to language requires choosing what representation the diffusion model should denoise. Discrete diffusion models corrupt tokens directly (Austin et al., 2021; Hoogeboom et al., 2021), while continuous diffusion language models denoise token embeddings or learned latent vectors (Li et al., 2022; Strudel et al., 2023; Lovelace et al., 2023).

Continuous token-level diffusion suffers from recovery errors: small denoising deviations can flip the recovered token (Zhang et al., 2025; Shen et al., 2026). Learned latent diffusion reduces this issue but still requires an interface for converting latents to text. We instead target internal transformer hidden states, where recovery becomes hidden-state reconstruction rather than direct token decoding.

### 3 Method

DiHAL is a diffusion–transformer hybrid architecture that replaces part of a pretrained transformer, rather than a standalone diffusion language model. Figure 1 illustrates our Locate-and-Replace procedure. This section develops DiHAL in three steps: we first motivate diffusion-friendly representations using geometric principles from Langevin dynamics and concentration theory, then instantiate them as layer-wise proxies for locating a suitable hidden-state interface, and finally replace the lower transformer layers with a conditional diffusion bridge while retaining the upper layers and original LM head. Rather than modeling token probabilities or recovering discrete tokens directly, the bridge reconstructs an internal boundary representation that the retained upper layers can already process.

#### 3.1 Geometric Principles for Diffusion-Friendly Layer Selection

We now make the notion of a diffusion-friendly representation space more concrete. Intuitively, a good diffusion space should satisfy three properties: denoising should contract quickly toward the target representation distribution, remain stable under score-estimation error, and have low effective complexity, meaning that variation is concentrated in relatively few active directions.

We formalize the first two properties through overdamped Langevin dynamics, an idealized setting with clean convergence and stability guarantees. The third property is captured by effective rank, which measures active variance directions. The theorem settings in this section are idealized: they motivate geometric surrogates, not assumptions that transformer hidden states exactly satisfy them.

Throughout this section, W2 denotes the 2-Wasserstein distance and P2(Rd) denotes probability measures with finite second moment. For a density p, define U(x) = −log p(x). We interpret strong convexity of U as a curvature-like restoring force toward high-density regions. Theorem 1 introduces the curvature parameter m and shows that larger m yields faster convergence to the target distribution.

Theorem 1 (Wasserstein contraction under strong log-concavity (Villani, 2009; Bakry et al., 2014)). Let µ ∈ P2(Rd) be a probability measure with density p, and define U(x) := −log p(x). Assume that U ∈ C2(Rd), U is m-strongly convex, i.e. ∇2U(x) ⪰ mI for all x ∈ Rd, for some m > 0, and that ∇U is globally Lipschitz. Let (Xt)t≥0 satisfy the overdamped Langevin stochastic differential equation (SDE) dXt = −∇U(Xt)dt + √2dWt, where Wt denotes Brownian motion.

Then µ is an invariant distribution of (Xt), and for every initial law ν0 ∈ P2(Rd),

W2(νt,µ) ≤ e−mtW2(ν0,µ), νt := L(Xt),

where L(Xt) denotes the distribution of Xt. The invariant distribution µ is unique in P2(Rd).

- Theorem 1 gives the first criterion. If the curvature parameter m is large, the distance to the target distribution shrinks as e−mt. Thus, larger m means faster contraction, which is desirable for diffusion because denoising should quickly return noisy samples to the data distribution.

Fast contraction alone is not enough. In practice, the score is unknown and estimated by a neural network. Theorem 2 gives the second criterion. If the score error is at most ε, the induced distributional error is bounded by ε/m. Thus, larger m corresponds to stability under imperfect score estimation.

- Theorem 2 (Stability of an invariant measure under score perturbation). Let µ ∈ P2(Rd) have density p and define U(x) := −log p(x). Assume that U ∈ C2(Rd) is m-strongly convex, and that ∇U is globally Lipschitz. Let s(x) := ∇log p(x) = −∇U(x), and let sˆ : Rd → Rd be globally

Lipschitz and satisfy supx∈Rd ∥sˆ(x)−s(x)∥ ≤ ε. Consider the two SDEs dXt = s(Xt)dt+√2dWt and dXˆt = sˆ(Xˆt)dt + √2dWt. Assume that the second SDE admits an invariant distribution µˆ.

Then µˆ ∈ P2(Rd) and

ε m

W2(ˆµ,µ) ≤

.

Together, Theorems 1 and 2 suggest that curvature is a useful proxy for convergence speed and stability under score-estimation error. However, curvature alone does not capture whether a representation is easy to model: variation may still spread across many directions. We therefore use effective rank as a proxy for dimensionality. If activations concentrate near a low-dimensional manifold, diffusion needs to model a few meaningful directions. Here, tr(Σ) is total variance and ∥Σ∥ is the largest covariance eigenvalue, so reff(Σ) = tr(Σ)/∥Σ∥ measures the effective number of active variance directions.

Lemma 1 (Approximate manifold support implies low effective rank). Let X be an Rd-valued random variable with covariance Σ := Cov(X). Assume there exist a k-dimensional C2 manifold M ⊂ Rd and a measurable map Π : Rd → M such that tr(Cov(Π(X))) ≤ C1k, E∥X−Π(X)∥2 ≤ C2(δ2 + η), and ∥Σ∥ ≥ c > 0. Then

tr(Σ) ∥Σ∥

2C1 c

2C2 c

(δ2 + η).

≤

reff(Σ) :=

k +

In particular, if δ,η are controlled constants and ∥Σ∥ is bounded above and below by constants, then reff(Σ) = O(k).

Lemma 1 justifies using reff(Σ) as an operational intrinsic-dimension proxy: near a k-dimensional manifold with controlled off-manifold error, effective rank is controlled by k rather than ambient dimension d. Theorem 3 combines this dimension control with the curvature conditions from Theorems 1 and 2, connecting low effective dimensionality to representation concentration while curvature controls fluctuations around the mean. The concentration part follows standard Bakry– Émery and Herbst arguments (Bakry et al., 2014; Ledoux, 2001).

- Theorem 3 (Intrinsic dimension and effective representation complexity). Let µ ∈ P2(Rd) have density p(x) = Z−1e−U(x) for some U ∈ C2(Rd), and let Σ := Cov(µ). Assume that ∇2U(x) ⪰ mI for all x ∈ Rd, for some m > 0, and that ∇U is globally Lipschitz.

Assume moreover that there exist a k-dimensional C2 manifold M ⊂ Rd and a measurable map Π : Rd → M. For X ∼ µ, suppose that tr(Cov(Π(X))) ≤ C1k, E∥X − Π(X)∥2 ≤ C2(δ2 + η), and ∥Σ∥ ≥ c0 > 0.

Then

2C1 c0

2C2 c0

(δ2 + η), and hence

reff(Σ) ≤

k +

1/2

2C1 c0

2C2 c0

E∥X − EX∥2 1/2 = tr(Σ) ≤ ∥Σ∥1/2

(δ2 + η)

k +

##### .

Furthermore, there exists an absolute constant c > 0 such that for all t ≥ 0,

##### P(|∥X − EX∥ − E∥X − EX∥| ≥ t) ≤ 2exp(−cmt2).

Theorem 3 combines Lemma 1 with concentration under strong log-concavity. It shows that lowdimensional concentration controls effective representation complexity through effective rank, while the curvature parameter m controls fluctuations around the mean through concentration of measure.

Taken together, these results are not intended as guarantees for transformer activations but as theoretical motivation for what a diffusion-friendly representation should look like. Since reverse diffusion uses time-dependent scores of noisy marginals whereas overdamped Langevin dynamics uses the fixed target-distribution score, we use these results only to motivate qualitative desiderata: contraction-like behavior, robustness to score-estimation error, and low effective complexity. A good layer should therefore exhibit strong curvature-like contraction for stable denoising and low effective dimensionality for easier modeling. Because the true density, Hessian, and manifold structure are unavailable, we approximate these ideas with empirical spectral proxies: local covariance concentration, global precision-based stiffness, and effective rank. All proofs are in Appendix A.

- 3.2 Locate: Finding Diffusion-Friendly Layers

These theoretical results serve as surrogate motivation, not assumptions that hidden states are globally strongly log-concave. Rather than guarantees, they motivate qualitative desiderata for diffusionfriendly representations: contraction-like behavior, robustness to score-estimation error, and low effective complexity. Since the true density, Hessian, and manifold structure are unavailable, we approximate these desiderata using empirical spectral quantities (see Appendix B for details). For each layer, let x ∈ RM×D denote the activation matrix over M tokens and D hidden dimensions.

We compute three statistics on x. First, the local curvature proxy mˆ curv is obtained from the covariance of k-nearest-neighbor neighborhoods: m(curvi) = 1/λmax(Σ(locali) ), with the layer-level value taken as the median; larger values indicate compact neighborhoods. Second, the monotonicity proxy mˆ mono captures global directional stiffness. With P = (Σ+λI)−1 denoting the regularized precision of the empirical covariance Σ, we compute mij = (xi−xj)⊤P(xi−xj)/∥xi−xj∥2 for sampled pairs and take the median. Third, effective intrinsic dimension is estimated as kˆ = reff(Σ) = tr(Σ)/∥Σ∥. Diffusion-friendly layers should have large curvature-related proxies and small effective rank.

We combine these into a selection score: selection_score(ℓ) = z(log mˆ curv(ℓ))+z(log mˆ mono(ℓ))− z(log kˆ(ℓ)), where z(·) denotes layer-wise z-score normalization. The score rewards curvature proxies while penalizing effective rank. We define bridgeability as reconstructability of a layer’s hidden state by the diffusion bridge under a matched training protocol, measured by validation loss. The layer sweep evaluates whether this score predicts bridgeability, not to tune it or select an oracle layer. We select ℓ∗ = arg maxℓ selection_score(ℓ). This is a low-cost layer-selection criterion, not a direct estimator of theoretical constants. Details are in Appendix C.

- 3.3 Replace: Hidden-State Diffusion Module

Given the selected insertion layer ℓ∗, we replace lower transformer layers with a conditional diffusion bridge. Let F1:ℓ∗ denote the original computation up to layer ℓ∗, and Fℓ∗+1:L the retained upper layers. For input x, the original model produces hℓ∗ = F1:ℓ∗(x). The bridge is embedding-conditioned: c(x) is derived from the source model’s embedding output before the first transformer block. It is trained to reconstruct hˆℓ∗ = Dθ(c(x)) in the same hidden space. The bridge does not generate tokens directly. Instead, it reconstructs the selected-layer hidden state hˆℓ∗, which is consumed by the retained upper layers as hL = Fℓ∗+1:L(hˆℓ∗). The original LM head then maps hL to token probabilities.

At inference time, lower layers are skipped and Dθ maps this condition to the selected-layer hidden state. We instantiate Dθ as a UNet-based latent denoiser, using a Stable-Diffusion-style architecture as a conditional denoising backbone for hidden-state activations rather than as an image generator; a small-scale ablation is provided in Appendix D.1. Hidden states are projected into a latent tensor,

denoised, and projected back to yield hˆℓ∗. The bridge is trained on language-model hidden states, with no text-to-image semantics or image supervision. For causal evaluation, DiHAL uses the backbone’s

left-to-right interface: at step t, the condition uses only prefix tokens x≤t, future positions are masked, and the retained causal suffix produces the next-token distribution. Attention and prefix masks are applied consistently to the conditioning pathway and retained suffix.

The main objective is hidden-state denoising rather than standalone text generation. We optimize a diffusion loss Ldiff = Et,ϵ[∥ϵˆθ(zt,t,c) − ϵ∥22] and a reconstruction loss Lrec = ∥hˆℓ∗ − hℓ∗∥22. To preserve compatibility with the retained language-modeling interface, we additionally use nexttoken and logit-distillation losses, LLM and LKD. The overall objective is L = Ldiff + λrecLrec + λLMLLM + λKDLKD. Implementation details are provided in Appendix C.7.

- 4 Experiments

- 4.1 Experimental Setup

We evaluate DiHAL on two representative 8B-scale decoder-only backbones: Llama-3.1-8BInstruct (Grattafiori et al., 2024), which has 32 transformer layers with hidden size 4096, and Qwen3-8B (Yang et al., 2025), which has 36 layers with the same hidden size. For each backbone, we run the source model on 300K sequences from Dolma v1.7 (Soldaini et al., 2024) and save

60

1000

m_curvature (q50, q25-q75) m_monotonicity (q50, q25-q75) k_effective_rank (q50, q25-q75)

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

750

40

mvalues

###### kvalues

500

20

250

0

0

0 5 10 15 20 25 30

Layer

|m_curvature (q50, q25-q75)<br><br>m_monotonicity (q50, q25-q75) k_effective_rank (q50, q25-q75)<br><br>|
|---|

300

20

15

200

mvalues

kvalues

10

100

5

0

0

0 5 10 15 20 25 30 35

Layer

- Figure 2: Layer-wise geometry of hidden representations for Llama-3.1-8B-Instruct (left) and Qwen38B (right). Curvature-related and effective-rank statistics vary substantially across depth. The selected insertion layer is determined by balancing local compactness, global stiffness, and effective representation complexity, rather than by maximizing a single proxy.

layerwise hidden states. We estimate the geometric proxies mˆ curv, mˆ mono, and kˆ = reff(Σ) from 100 repeated 3K-example subsamples, rank candidate insertion layers using the fixed geometry score from Section 3.2, and verify score stability on 30 additional 500-example subsamples.

To test whether the ranking predicts bridgeability, we train one bridge per candidate layer for one epoch on a 150K-example subset with a 9:1 train/validation split and measure validation bridge loss. This sweep evaluates the geometry score but does not fit it. Each bridge is embedding-conditioned and targets the corresponding layer hidden state. We instantiate it with Stable-Diffusion-v1.5-style latent denoising components (Rombach et al., 2022), freezing the VAE while training the UNet and bridge-specific projections. These components are repurposed for hidden-state denoising; training uses no CLIP conditioning, text-to-image objective, or image supervision. Finally, we fully train the highest-scoring layer on the 300K-example corpus for four epochs and report negative log-likelihood (NLL), perplexity (PPL), and output-distribution KL divergence against the original pretrained model.

Evaluation. We evaluate three aspects of DiHAL. For layer selection, we compare the geometry ranking with validation bridge loss and report Spearman correlation, Kendall correlation, the best predicted layer, the best observed layer, and their rank gap. For final model quality, we report NLL and PPL on WikiText-103 (Merity et al., 2016) and held-out Dolma v1.7. For teacher alignment, we compute KL divergence between teacher and DiHAL logits. Additional implementation and hyperparameter details are provided in Appendix D.

#### 4.2 Layer-Wise Geometry

We first examine whether transformer hidden representations exhibit systematic geometric variation across depth. Figure 2 shows clear layer dependence in both backbones: input-adjacent layers tend to have large local curvature values, while global monotonicity and effective rank follow different depth-dependent trends. These patterns suggest that transformer hidden states do not form a uniform sequence of equally suitable diffusion spaces. Large mˆ curv indicates locally compact neighborhoods, but local compactness does not necessarily imply globally coherent stiffness, as measured by mˆ mono. Likewise, low effective rank alone does not guarantee strong curvature-related structure. Thus, layer selection reflects a curvature–dimension trade-off rather than optimization of a single proxy.

The fixed geometry score combines local curvature, global monotonicity, and effective rank. It selects layer 3 for Llama-3.1-8B and layer 2 for Qwen3-8B, rather than defaulting to the largest single proxy. Both selected layers are close to the embedding interface, suggesting that continuous diffusion may be suitable for hidden spaces that retain embedding-like geometric structure while remaining easier to denoise than token embeddings themselves. Since geometry alone does not guarantee bridgeability, we next test whether this ranking predicts bridgeability under a matched budget.

#### 4.3 Fixed-Budget Layer Sweep

We perform a fixed-budget layer sweep to test whether the geometric pattern in Figure 2 translates into bridgeability. For each candidate layer, we train one bridge for one epoch on 150K examples and measure validation bridge loss; the sweep evaluates the geometry score but does not fit it. We

compare against single-proxy baselines using only mˆ curv or kˆ, and depth-based Early/Middle/Late

31

1.2

1.0

Validationloss

0.8

0.6

0.4

3

6 4 2 0 2

Selection score (higher is better)

35

800

700

34

600

Validationloss

500

400

300

200

100

2

0

5 4 3 2 1 0 1 2

Selection score (higher is better)

- Figure 3: Fixed geometry score versus validation bridge loss for Llama-3.1-8B-Instruct (left) and Qwen3-8B (right). Each point is one candidate layer, labeled by index. Higher scores generally correspond to lower validation loss, indicating that the fixed geometry score predicts bridgeability.

- Table 1: Fixed-budget layer sweep results. We compare the geometry-selected layer against representative baselines under the same bridge-training budget. Validation bridge loss measures bridgeability.

Model Layer type Layer Selection score ↑ mˆ curv ↑ mˆ mono ↑ kˆ ↓ Val. bridge loss ↓

Geometry-selected 3 2.360 553.166 227.324 1.332 0.331 Curvature-only 1 1.271 780.707 45.976 1.310 0.324 Dimension-only 2 1.857 701.159 102.836 1.314 0.327 Early baseline 7 2.324 258.717 374.482 1.472 0.362 Middle baseline 17 0.265 51.230 134.336 2.679 0.397 Late baseline 27 -4.094 7.643 23.769 8.305 0.656

Llama-3.1-8B

Geometry-selected 2 2.044 168.963 317.682 4.544 0.060 Curvature-only 1 1.661 292.434 450.606 8.267 0.059 Dimension-only 6 1.492 4.965 7.484 1.008 139.092 Early baseline 7 1.430 4.602 7.237 1.012 139.161 Middle baseline 18 0.616 1.036 3.535 1.097 164.087 Late baseline 30 -3.734 0.034 0.112 4.089 276.584

Qwen3-8B

baselines. Figure 2 suggests that embedding-adjacent layers form a distinct geometric regime, with stronger curvature-related proxies near the input and less favorable effective rank in later layers.

If the score is meaningful, higher-scoring layers should achieve lower validation loss. We therefore correlate the geometry score with negative validation bridge loss, where higher is better. Figure 3 visualizes this trend, and Table 1 shows that the score is not intended to select the exact minimum-loss layer but still selects layers close to the best observed loss and far below middle and late baselines. For Llama-3.1-8B, compared with selected layer 3, layer 17 has much lower curvature (mˆ curv = 51.230 vs. 553.166) and higher effective rank (kˆ = 2.679 vs. 1.332); layer 27 further decreases in curvature to 7.643 and increases in effective rank to 8.305. Qwen3-8B shows a similar pattern. These results suggest that deeper, more task-specialized representations are less favorable for diffusion-based reconstruction under the matched bridge-training protocol.

Across all candidate layers, the fixed score shows strong agreement with bridgeability. Table 2 reports correlations over 30 repeated 500-example score-estimation runs. The score achieves Spearman ρ = 0.9143 ± 0.0069 on Llama-3.1-8B and ρ = 0.9267 ± 0.0157 on Qwen3-8B, with rank gaps of 2 and 1 between the best predicted and best observed layers. These results show that the fixed geometry score is useful for cheaply identifying bridgeable layers under a matched training budget.

#### 4.4 Diagnostic Matched-Budget Comparison of Continuous Diffusion Methods

We compare continuous diffusion methods under the same budget for training diffusion/recovery components in the Qwen3-8B setting. The baselines cover token-embedding denoising (DiffusionLM (Li et al., 2022), SED (Strudel et al., 2023)), learned-latent denoising (LD4LG (Lovelace et al., 2023)), and continuous-to-discrete recovery (CoDAR (Shen et al., 2026)). In contrast, DiHAL denoises an internal hidden state and delegates token prediction to the retained Qwen3-8B suffix and LM head. This is a diagnostic, not pretraining-matched, comparison: all methods use the same

- Table 2: Agreement between fixed geometry score and bridgeability under the fixed-budget sweep. Correlations use negative validation bridge loss; parentheses report standard deviations over 30 repeated 500-example score runs. Geometry proxies use 100 repeated 3K-example subsamples.

Model

Spearman ρ ↑

Kendall τ ↑

Pearson r ↑

Best pred.

Best obs.

Rank gap ↓

Llama-3.1-8B

0.9143

(±0.0069)

0.8011

(±0.0086)

0.8687

(±0.0027) 3 1 2 Qwen3-8B

0.9267

(±0.0157)

0.8208

(±0.0256)

0.8605

(±0.0098) 2 1 1

- Table 3: Diagnostic same-budget comparison of continuous diffusion target spaces and recovery interfaces. This is not a pretraining-matched comparison: DiHAL reuses a pretrained transformer suffix and LM head, while some baselines use smaller standalone recovery modules.

Method Diffusion target Recovery interface Gen.PPL ↓ Div. ↑ DiHAL Hidden state hℓ∗ Retained upper layers + LM head 136.02 0.5913 Diffusion-LM Token embeddings Embedding-to-token recovery 683.43 0.4324 SED Self-conditioned embeddings Embedding-to-token recovery 778.82 0.4942 LD4LG Learned text latent Frozen BART + latent decoder 166.11 0.5797 CoDAR Continuous token/latent states Continuous-to-discrete recovery 144.83 0.4777

300K-example corpus and 40 H100-hour budget for diffusion/recovery training but differ in how they reuse pretrained components.

We evaluate generated text using Gen.PPL, the perplexity assigned by a GPT-2 evaluator, and diversity, defined as the product of Distinct-1 through Distinct-4, where Distinct-n is the ratio of unique n-grams to generated n-grams. Table 3 shows that DiHAL achieves the lowest Gen.PPL and highest diversity in this setting. Compared with CoDAR, DiHAL improves Gen.PPL from 144.83 to 136.02 and diversity from 0.4777 to 0.5913, suggesting that moving diffusion from token recovery to hidden-state recovery can be beneficial in this diagnostic setting. Additional details are in Appendix E.

4.5 Top-Layer Full Training and Evaluation

We fully train the highest-ranked layer to test whether the short-budget signal transfers to a larger optimization setting. We compare it against two controls: a validation-loss oracle, defined as the layer with the lowest validation bridge loss in the fixed-budget sweep, and a worst-layer control, which tests whether poor insertion points degrade the hybrid model. Since identifying the oracle requires explicitly training bridges across candidate layers, this comparison tests whether the geometry score can select a competitive layer without using validation bridge loss as the criterion.

- Table 4 reports the final evaluation. We also include CoDAR, a recent continuous-diffusion baseline, re-evaluated under the same pipeline. The geometry-selected layer improves over the worst-layer control and remains competitive with the validation-loss oracle. On Llama-3.1-8B, it outperforms the oracle in NLL and PPL, showing that the fixed-budget oracle is not always optimal for final language-modeling metrics. On Qwen3-8B, the oracle is slightly better, but the geometry-selected layer remains comparable without layer-wise bridge training and substantially improves over CoDAR. Thus, this evaluation supports geometry-based layer selection and hidden-state recovery, while the remaining gap to the original autoregressive teacher clarifies DiHAL’s scope.
- 5 Related Work

Diffusion language models adapt diffusion to text, where discreteness makes generation less direct than in images (Hoogeboom et al., 2022). Prior work includes discrete token diffusion (Austin et al., 2021; Gong et al., 2023), masked diffusion with iterative refinement (Sahoo et al., 2024; Nie et al., 2025; Ye et al., 2025), and continuous diffusion over embeddings or learned latents (Li et al., 2022; Lovelace et al., 2023). Continuous methods avoid token corruption but require recovering tokens

Table 4: Final evaluation of DiHAL insertion layers against CoDAR on the combined WikiText103 and held-out Dolma evaluation sets. CoDAR is a recent strong continuous-diffusion baseline re-evaluated in the Qwen3-8B setting under the same evaluation pipeline.

Model Method Layer NLL ↓ PPL ↓ KL ↓ Llama-3.1-8B DiHAL, geometry-selected 3 4.91 135.64 0.73

Validation-loss oracle 1 5.11 165.67 0.62 Worst layer 31 5.17 175.91 1.32

Qwen3-8B DiHAL, geometry-selected 2 4.97 144.03 0.53 Validation-loss oracle 1 4.94 139.77 0.54 Worst layer 35 5.23 186.79 1.46

Recent continuous diffusion CoDAR N/A 5.18 177.87 N/A

from denoised vectors, which can introduce projection or decoding errors (Wang et al., 2022; Li et al., 2022). We instead study transformer hidden states as a diffusion-friendly denoising space.

Diffusion behavior depends on representation geometry, including curvature, conditioning, and intrinsic dimension (Pidstrigach, 2022; Rombach et al., 2022). In parallel, efficient generation has been studied through transformer compression, distillation, layer reduction, early exiting, and hybrid modules (Fan et al., 2020; Sanh et al., 2020; Lenz et al., 2025). DiHAL connects these directions by identifying internal transformer representations suitable for diffusion-based replacement.

### 6 Limitations

Continuous diffusion language modeling still faces important limitations at scale, especially because it must learn both continuous denoising and reliable recovery back to discrete language. DiHAL mitigates this difficulty by moving diffusion to an internal hidden-state interface, but it is not a standalone diffusion language model: token prediction still depends on the retained transformer suffix and LM head. Due to compute constraints, we do not fully explore larger bridges, longer training, or deeper replacement. Future work could incorporate the proposed geometric proxies directly into bridge training, making deeper hidden states more bridgeable and potentially enabling replacement of larger transformer prefixes. We therefore view DiHAL as a step toward understanding where continuous diffusion can operate inside pretrained language models.

### 7 Conclusion

We introduced DiHAL, a geometry-guided diffusion–transformer hybrid that moves continuous diffusion from token-level recovery to internal hidden-state reconstruction. Instead of treating the continuous-to-discrete interface as an unavoidable bottleneck, DiHAL asks where inside a pretrained language model diffusion should operate. It locates diffusion-friendly layers using curvature, monotonicity, and effective-rank proxies, then replaces the lower transformer prefix with a conditional diffusion bridge while preserving the upper layers and original LM head. This yields a simple but important reframing: continuous diffusion need not generate language by directly recovering tokens; it can reconstruct a representation that the pretrained transformer already knows how to decode.

Our experiments show that this interface choice is not incidental. Across two 8B-scale backbones, geometry-selected insertion points are embedding-adjacent, predict bridgeability under fixed-budget training, and remain competitive with validation-loss oracles after full training. Middle and late hidden states are much harder to reconstruct, suggesting that diffusion failures in language reflect not only discreteness but also a mismatch between denoising and representation-space geometry.

DiHAL is not yet a standalone diffusion language model and still relies on retained pretrained layers. However, this limitation sharpens the paper’s main lesson: successful continuous diffusion for language may require choosing or learning the right internal interface, rather than applying diffusion uniformly to arbitrary continuous spaces. By identifying where diffusion can effectively enter an existing language model, DiHAL provides a step toward principled diffusion–transformer hybrids and future models that use geometry not only to locate, but also to train, more bridgeable representations.

### References

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=h7-XixPCAL.

Dominique Bakry, Ivan Gentil, and Michel Ledoux. Analysis and Geometry of Markov Diffusion Operators, volume 348 of Grundlehren der mathematischen Wissenschaften. Springer Cham, 2014. ISBN 978-3-319-00227-9. doi: 10.1007/978-3-319-00227-9.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.

Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth on demand with structured dropout. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SylO2yStDr.

Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and Lingpeng Kong. Diffuseq-v2: Bridging discrete and continuous text spaces for accelerated seq2seq diffusion models. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https:// openreview.net/forum?id=OLcDbSRjbx.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan,

Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The

##### llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack William Rae, and Laurent Sifre. An empirical analysis of compute-optimal large language model training. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=iBBcRUlOAPR.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=6nbpPqUCIi7.

Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=Lm8T39vLDTE.

Jaehyeong Jo and Sung Ju Hwang. Continuous diffusion model for language modeling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https: //openreview.net/forum?id=VGv5y60sXC.

M. Ledoux. The Concentration of Measure Phenomenon. Mathematical surveys and monographs. American Mathematical Society, 2001. ISBN 9780821837924. URL https://books.google. com.sg/books?id=mCX_cWL6rqwC.

Barak Lenz, Opher Lieber, Alan Arazi, Amir Bergman, Avshalom Manevich, Barak Peleg, Ben Aviram, Chen Almagor, Clara Fridman, Dan Padnos, Daniel Gissin, Daniel Jannai, Dor Muhlgay, Dor Zimberg, Edden M. Gerber, Elad Dolev, Eran Krakovsky, Erez Safahi, Erez Schwartz, Gal Cohen, Gal Shachaf, Haim Rozenblum, Hofit Bata, Ido Blass, Inbal Magar, Itay Dalmedigos, Jhonathan Osin, Julie Fadlon, Maria Rozman, Matan Danos, Michael Gokhman, Mor Zusman, Naama Gidron, Nir Ratner, Noam Gat, Noam Rozen, Oded Fried, Ohad Leshno, Omer Antverg, Omri Abend, Or Dagan, Orit Cohavi, Raz Alon, Ro’i Belson, Roi Cohen, Rom Gilad, Roman Glozman, Shahar Lev, Shai Shalev-Shwartz, Shaked Haim Meirom, Tal Delbari, Tal Ness, Tomer Asida, Tom Ben Gal, Tom Braude, Uriya Pumerantz, Josh Cohen, Yonatan Belinkov, Yuval Globerson, Yuval Peleg Levy, and Yoav Shoham. Jamba: Hybrid transformer-mamba language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=JFPaD7lpBD.

Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori Hashimoto. DiffusionLM improves controllable text generation. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=3s9IrEsjLyk.

Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Seo Shekhtman, and Kilian Q Weinberger. Latent diffusion for language generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=NKdtztladR.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016. URL https://arxiv.org/abs/1609.07843.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, JUN ZHOU, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/ forum?id=KnqiC0znVF.

Jakiw Pidstrigach. Score-based generative models detect manifolds, 2022. URL https://proceedings.nips.cc/paper_files/paper/2022/file/ e8fb575e3ede31f9b8c05d53514eb7c6-Paper-Conference.pdf.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.

Subham Sekhar Sahoo, Marianne Arriola, Aaron Gokaslan, Edgar Mariano Marroquin, Alexander M Rush, Yair Schiff, Justin T Chiu, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=L4uaAR4ArM.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter, 2020. URL https://www.emc2-ai.org/assets/ docs/neurips-19/emc2-neurips19-paper-33.pdf.

Junzhe Shen, Jieru Zhao, Ziwei He, and Zhouhan Lin. Codar: Continuous diffusion language models are more powerful than you think, 2026. URL https://arxiv.org/abs/2603.02547.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an open corpus of three trillion tokens for language model pretraining research, 2024. URL https://arxiv.org/abs/2402.00159.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= PxTIG12RRHS.

Robin Strudel, Corentin Tallec, Florent Altché, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will Sussman Grathwohl, Nikolay Savinov, Sander Dieleman, Laurent Sifre, and Rémi Leblond. Selfconditioned embedding diffusion for text generation, 2023. URL https://openreview.net/ forum?id=OpzV3lp3IMC.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

Cédric Villani. Optimal Transport: Old and New, volume 338 of Grundlehren der mathematischen Wissenschaften. Springer Berlin, Heidelberg, 2009. ISBN 978-3-540-71050-9. doi: 10.1007/ 978-3-540-71050-9.

Rose E Wang, Esin Durmus, Noah Goodman, and Tatsunori Hashimoto. Language modeling via stochastic processes. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=pMQwKL1yctf.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

Siyue Zhang, Yilun Zhao, Liyuan Geng, Arman Cohan, Anh Tuan Luu, and Chen Zhao. Diffusion vs. autoregressive language models: A text embedding perspective. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference

on Empirical Methods in Natural Language Processing, pages 4273–4303, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/ v1/2025.emnlp-main.213. URL https://aclanthology.org/2025.emnlp-main.213/.

### A Proofs of Theorems Preliminaries

Law of a random variable. Let (Ω,F,P) be a probability space and X : Ω → Rd be a random variable. The law (distribution) of X, denoted by L(X), is the probability measure defined by

L(X)(A) := P(X ∈ A), A ∈ B(Rd).

2-Wasserstein distance. Let P2(Rd) denote the set of probability measures with finite second moment. For ν,µ ∈ P2(Rd),

∥x − y∥2 π(dx,dy),

W22(ν,µ) := inf

π∈Π(ν,µ) Rd×Rd

where Π(ν,µ) denotes the set of couplings of ν and µ. Equivalently,

W22(ν,µ) = inf E∥X − Y ∥2 : L(X) = ν,L(Y ) = µ .

In Rd, an optimal coupling attaining the infimum exists. If preferred, one may instead work with ε-optimal couplings and let ε ↓ 0 at the end.

- Lemma 2 (Strong convexity implies gradient monotonicity). Let U : Rd → R be differentiable and m-strongly convex, i.e.

U(y) ≥ U(x) + ⟨∇U(x),y − x⟩ +

m 2 ∥y − x∥2 for all x,y.

Then

⟨∇U(x) − ∇U(y),x − y⟩ ≥ m∥x − y∥2 for all x,y. Proof. Apply strong convexity twice, swapping x and y:

U(y) ≥ U(x) + ⟨∇U(x),y − x⟩ +

m 2 ∥y − x∥2,

U(x) ≥ U(y) + ⟨∇U(y),x − y⟩ +

m 2 ∥x − y∥2.

Summing the two inequalities yields

⟨∇U(x) − ∇U(y),x − y⟩ ≥ m∥x − y∥2.

| |
|---|

- Lemma 3 (Gibbs invariance and uniqueness). Assume that U ∈ C2(Rd), ∇U is globally Lipschitz, and

∇2U(x) ⪰ mI for all x ∈ Rd for some m > 0. Let

µ(dx) = Z−1e−U(x) dx, Z :=

e−U(x) dx.

Rd

Then Z < ∞, µ ∈ P2(Rd), and µ is invariant for

√

dXt = −∇U(Xt)dt +

2dWt.

Moreover, if for every initial law ν0 ∈ P2(Rd) the law νt := L(Xt) satisfies

W2(νt,µ) ≤ e−mtW2(ν0,µ), then µ is the unique invariant distribution in P2(Rd).

Proof. By strong convexity, for every x ∈ Rd,

m 2 ∥x∥2.

U(x) ≥ U(0) + ⟨∇U(0),x⟩ +

Completing the square, this implies

m 4 ∥x∥2

e−U(x) ≤ C exp −

for some constant C < ∞. Hence Z < ∞ and µ has finite second moment. Let

Lφ(x) := ∆φ(x) − ⟨∇U(x),∇φ(x)⟩ be the generator. For φ ∈ Cc∞(Rd), integration by parts gives

Lφ(x)µ(dx) = Z−1

∆φ(x) − ⟨∇U(x),∇φ(x)⟩ e−U(x) dx = 0.

Rd

Rd

Since this holds for all φ ∈ Cc∞(Rd), the stationary equation L∗µ = 0 holds in the weak sense. Therefore µ is invariant.

Finally, if ν ∈ P2(Rd) is any invariant distribution, then applying the contraction estimate with ν0 = ν yields

##### W2(ν,µ) = W2(νt,µ) ≤ e−mtW2(ν,µ) for all t ≥ 0. Fix any t > 0. Since e−mt < 1, this implies W2(ν,µ) = 0, hence ν = µ.

| |
|---|

- Proof of Theorem 1 By Lemma 3, µ is an invariant distribution for

√

dXt = −∇U(Xt)dt +

2dWt.

Let ν0 := L(X0). Choose an optimal coupling (X0,Y0) between ν0 and µ such that

##### E∥X0 − Y0∥2 = W22(ν0,µ).

(Alternatively, choose an ε-optimal coupling and let ε ↓ 0.) We take this initial coupling to be independent of the Brownian motion below.

Define the synchronous coupling:

√

dXt = −∇U(Xt)dt +

2dWt, dYt = −∇U(Yt)dt +

√

2dWt,

driven by the same Brownian motion Wt. Since Y0 ∼ µ and µ is invariant, we have

L(Yt) = µ for all t ≥ 0. Let Zt := Xt − Yt. Subtracting the SDEs gives

dZt = −(∇U(Xt) − ∇U(Yt))dt. Since the stochastic terms cancel, Zt has absolutely continuous paths and satisfies the pathwise ODE Z˙t = −(∇U(Xt) − ∇U(Yt)) for a.e. t ≥ 0.

##### Therefore the ordinary chain rule applies, and d dt∥Zt∥2 = 2⟨Zt,Z˙t⟩ = −2⟨Zt,∇U(Xt) − ∇U(Yt)⟩.

By Lemma 2,

d dt∥Zt∥2 ≤ −2m∥Zt∥2.

By Grönwall’s inequality,

∥Zt∥2 ≤ e−2mt∥Z0∥2. Taking expectation yields

E∥Xt − Yt∥2 ≤ e−2mtE∥X0 − Y0∥2. Since (Xt,Yt) is a coupling of L(Xt) and µ, by definition of W2,

W22(L(Xt),µ) ≤ E∥Xt − Yt∥2. Therefore,

W22(L(Xt),µ) ≤ e−2mtW22(ν0,µ). Taking square roots gives

W2(L(Xt),µ) ≤ e−mtW2(ν0,µ).

It remains to prove uniqueness of the invariant distribution in P2(Rd). Let π ∈ P2(Rd) be any invariant distribution of the same Langevin SDE. Applying the contraction estimate above with

ν0 = π gives, for every t > 0, W2(π,µ) = W2(L(Xt),µ) ≤ e−mtW2(π,µ),

where we used the invariance of π to obtain L(Xt) = π. Since m > 0 and e−mt < 1 for t > 0, this implies

W2(π,µ) = 0.

Hence π = µ. Therefore µ is the unique invariant distribution in P2(Rd).

| |
|---|

#### Proof of Theorem 2

We prove stability of an invariant measure under a uniformly bounded perturbation of the score function.

Let µ ∈ P2(Rd) have density p and define

U(x) := −log p(x). Assume that U ∈ C2(Rd) is m-strongly convex, i.e.

∇2U(x) ⪰ mI for all x ∈ Rd, for some m > 0, and that ∇U is globally Lipschitz. Define the true score

s(x) := ∇log p(x) = −∇U(x). Let sˆ be a globally Lipschitz function satisfying

∥sˆ(x) − s(x)∥ ≤ ε.

sup

x∈Rd

Consider the SDEs

dXt = s(Xt)dt +

√

2dWt, dXˆt = sˆ(Xˆt)dt +

√

2dWt,

and assume that the second SDE admits an invariant distribution µˆ. By Lemma 3, µ is an invariant distribution of the first SDE.

We first show that µˆ ∈ P2(Rd). Since s = −∇U and U is m-strongly convex, Lemma 2 applied with y = 0 gives

⟨x,s(x) − s(0)⟩ ≤ −m∥x∥2 for all x ∈ Rd. Hence

⟨x,s(x)⟩ = ⟨x,s(x) − s(0)⟩ + ⟨x,s(0)⟩ ≤ −m∥x∥2 + ∥s(0)∥∥x∥. Using ∥sˆ(x) − s(x)∥ ≤ ε, we obtain

⟨x,sˆ(x)⟩ ≤ ⟨x,s(x)⟩ + ε∥x∥ ≤ −m∥x∥2 + (∥s(0)∥ + ε)∥x∥.

By Young’s inequality, there exists a constant C0 < ∞ such that

m 2 ∥x∥2 + C0 for all x ∈ Rd.

⟨x,sˆ(x)⟩ ≤ −

Let

Lfˆ (x) := ⟨sˆ(x),∇f(x)⟩ + ∆f(x) be the generator of the perturbed SDE, and set

##### V (x) := ∥x∥2. Then

LVˆ (x) = 2⟨x,sˆ(x)⟩ + 2d ≤ −m∥x∥2 + C1 = −mV (x) + C1 for some constant C1 < ∞. We now justify the use of this unbounded Lyapunov function by truncation. Let ψR ∈ C2([0,∞)) be nondecreasing and concave, with

0 ≤ ψR′ ≤ 1, ψR′′ ≤ 0, such that

ψR(r) = r for r ≤ R, ψR(r) = constant for r ≥ 2R. Define

VR(x) := ψR(V (x)).

Since VR ∈ Cb2(Rd) and the perturbed SDE has globally Lipschitz drift, the associated Markov semigroup ( Pt)t≥0 has generator Lˆ. Hence

d dt

= LVˆ R. By invariance of µˆ,

PtVR

t=0

VR(x)µˆ(dx) for all t ≥ 0. Differentiating at t = 0 gives

PtVR(x)µˆ(dx) =

Rd

Rd

LVˆ R(x)µˆ(dx) = 0. By the chain rule for Lˆ,

Rd

LVˆ R = ψR′ (V )LVˆ + ψR′′(V )∥∇V ∥2. Since ψR′′ ≤ 0, we have

LVˆ R ≤ ψR′ (V )LVˆ ≤ ψR′ (V )(−mV + C1). Therefore

0 = LV ˆ R dµˆ ≤ −m ψR′ (V )V dµˆ + C1 ψR′ (V )dµ.ˆ Since 0 ≤ ψR′ ≤ 1, this implies

m ψR′ (V )V dµˆ ≤ C1. Moreover, ψR′ (V ) = 1 whenever V ≤ R, and hence

V dµˆ ≤ C1. Letting R → ∞ and using monotone convergence gives

m

{V ≤R}

C1 m

∥x∥2 µˆ(dx) = V dµˆ ≤

< ∞.

Thus µˆ ∈ P2(Rd).

Now choose an optimal coupling (X0,Xˆ0) of µ and µˆ such that

E∥X0 − Xˆ0∥2 = W22(µ,µˆ). We take this initial coupling to be independent of the Brownian motion below. Drive both SDEs by the same Brownian motion (Wt)t≥0 and define

Zt := Xˆt − Xt. Subtracting the two SDEs yields

dZt = s ˆ(Xˆt) − s(Xt) dt. Since the stochastic terms cancel, Zt has absolutely continuous paths and satisfies the pathwise ODE Z˙t = sˆ(Xˆt) − s(Xt) for a.e. t ≥ 0.

Therefore the ordinary chain rule applies, and d dt∥Zt∥2 = 2⟨Zt,sˆ(Xˆt) − s(Xt)⟩.

We decompose

sˆ(Xˆt) − s(Xt) = s ˆ(Xˆt) − s(Xˆt) + s(Xˆt) − s(Xt) . Hence

d dt∥Zt∥2 = 2⟨Zt,sˆ(Xˆt) − s(Xˆt)⟩ + 2⟨Zt,s(Xˆt) − s(Xt)⟩.

For the first term, using the uniform bound on the score error,

2⟨Zt,sˆ(Xˆt) − s(Xˆt)⟩ ≤ 2ε∥Zt∥. For the second term, Lemma 2 implies

⟨Xˆt − Xt,∇U(Xˆt) − ∇U(Xt)⟩ ≥ m∥Xˆt − Xt∥2. Since s = −∇U, we get

⟨Zt,s(Xˆt) − s(Xt)⟩ ≤ −m∥Zt∥2. Combining the two bounds gives

d dt∥Zt∥2 ≤ −2m∥Zt∥2 + 2ε∥Zt∥.

By Young’s inequality,

ε2 m

2ε∥Zt∥ ≤ m∥Zt∥2 +

. Substituting this into the differential inequality, we obtain

ε2 m

d dt∥Zt∥2 ≤ −m∥Zt∥2 +

.

Multiplying both sides by emt and integrating from 0 to t yields

ε2 m2

1 − e−mt a.s. Taking expectations gives

∥Zt∥2 ≤ e−mt∥Z0∥2 +

ε2 m2

E∥Zt∥2 ≤ e−mtE∥Z0∥2 +

1 − e−mt .

Since X0 ∼ µ and Xˆ0 ∼ µˆ are invariant initial laws, we have

##### Xt ∼ µ, Xˆt ∼ µˆ for all t ≥ 0.

Therefore (Xt,Xˆt) is a coupling of µ and µˆ, so

W22(ˆµ,µ) ≤ E∥Xˆt − Xt∥2 = E∥Zt∥2. Combining this with the previous estimate gives

ε2 m2

W22(ˆµ,µ) ≤ e−mtW22(ˆµ,µ) +

1 − e−mt . Letting t → ∞, we obtain

ε2 m2

W22(ˆµ,µ) ≤

. Taking square roots gives

ε m

W2(ˆµ,µ) ≤

.

### Proof of Lemma 1

Let

Σ := Cov(X). Define

R := X − Π(X). Since

EX = EΠ(X) + ER, we may write

X − EX = Π(X) − EΠ(X) + R − ER . Therefore, using the inequality ∥u + v∥2 ≤ 2∥u∥2 + 2∥v∥2, we obtain

tr(Σ) = E∥X − EX∥2 ≤ 2E∥Π(X) − EΠ(X)∥2 + 2E∥R − ER∥2. For the first term, by assumption,

E∥Π(X) − EΠ(X)∥2 = tr(Cov(Π(X))) ≤ C1k. For the second term, since covariance is dominated by the second moment,

E∥R − ER∥2 = tr(Cov(R)) ≤ E∥R∥2. Using the approximation assumption,

E∥R∥2 = E∥X − Π(X)∥2 ≤ C2(δ2 + η). Combining the two bounds yields

tr(Σ) ≤ 2C1k + 2C2(δ2 + η). By definition,

tr(Σ) ∥Σ∥

reff(Σ) =

.

Using the non-degeneracy assumption ∥Σ∥ ≥ c > 0, we obtain

2C1k + 2C2(δ2 + η) c

2C1 c

2C2 c

(δ2 + η).

reff(Σ) ≤

=

k +

In particular, if δ,η = O(1) and ∥Σ∥ is bounded below by a positive constant, then

reff(Σ) = O(k + 1). If additionally k ≥ 1, this simplifies to

reff(Σ) = O(k).

| |
|---|

| |
|---|

#### Proof of Theorem 3

Let X ∼ µ, and denote

x¯ := EX, Σ := Cov(X) = E (X − x¯)(X − x¯)⊤ . By the trace identity,

E∥X − x¯∥2 = Etr (X − x¯)(X − x¯)⊤ = tr E(X − x¯)(X − x¯)⊤ = tr(Σ).

Next, by the assumptions of Lemma 1, we have

tr(Cov(Π(X))) ≤ C1k, E∥X − Π(X)∥2 ≤ C2(δ2 + η), ∥Σ∥ ≥ c0 > 0. Therefore Lemma 1 yields

2C1 c0

2C2 c0

tr(Σ) ∥Σ∥

(δ2 + η). Since

≤

reff(Σ) =

k +

tr(Σ) = ∥Σ∥reff(Σ), it follows that

2C1 c0

2C2 c0

(δ2 + η) . Using E∥X − x¯∥2 = tr(Σ), we obtain

tr(Σ) ≤ ∥Σ∥

k +

2C1 c0

2C2 c0

E∥X − x¯∥2 1/2 = tr(Σ) ≤ ∥Σ∥1/2

(δ2 + η)

k +

1/2

.

It remains to prove the concentration statement. Since

p(x) = Z−1e−U(x) and

∇2U(x) ⪰ mI for all x ∈ Rd, the Bakry–Émery criterion implies that µ satisfies a logarithmic Sobolev inequality with constant of order 1/m, up to the standard normalization convention. Consequently, by the Herbst argument, every 1-Lipschitz function f : Rd → R satisfies a Gaussian concentration bound of the form

P(|f(X) − Ef(X)| ≥ t) ≤ 2exp(−cmt2) for all t ≥ 0, where c > 0 is an absolute constant. Now define

f(x) := ∥x − x¯∥. For any x,y ∈ Rd,

|f(x) − f(y)| = ∥x − x¯∥ − ∥y − x¯∥ ≤ ∥x − y∥,

so f is 1-Lipschitz. Applying the preceding concentration bound to this f gives

P(|∥X − x¯∥ − E∥X − x¯∥| ≥ t) ≤ 2exp(−cmt2) for all t ≥ 0.

This proves the claimed concentration inequality and completes the proof. Corollary 1. By Jensen’s inequality,

| |
|---|

E∥X − EX∥ ≤ E∥X − EX∥2 = tr(Σ). Hence

P ∥X − EX∥ ≥ tr(Σ) + t ≤ 2exp(−cmt2).

In particular, if δ,η are bounded by constants and ∥Σ∥ ≍ 1, then there exists a constant C > 0 such that

√

tr(Σ) ≤ C

k, and therefore

√

k + t ≤ 2exp(−cmt2).

P ∥X − EX∥ ≥ C

### B Interpretation of the Geometric Proxies

In this appendix, we clarify why the empirical quantities used in our layer-selection procedure can be interpreted as proxies for the theoretical geometric terms introduced in Section 3.1. Our goal is not to recover the exact strong-convexity constant or the exact manifold dimension of the layerwise representation distribution. Rather, we seek observable quantities that capture the same functional roles in diffusion-friendly geometry.

Theoretical role of m. In our theory, the quantity m appears as the strong-convexity constant of the potential U(x) = −log p(x), namely

∇2U(x) ⪰ mI.

This means that, in every direction, the potential has at least curvature m. A larger m implies stronger contraction of the corresponding Langevin dynamics and greater robustness to score perturbations. Therefore, from the perspective of diffusion, m measures how strongly the representation distribution exhibits restoring geometry toward high-density regions.

In practice, however, the exact density p(x) of layer activations is unknown, and only a finite sample of activation vectors is available. As a result, neither U(x) nor its Hessian ∇2U(x) can be computed exactly. This motivates the use of empirical curvature-related proxies.

Why mˆ mono is an m-like quantity. Our monotonicity proxy is defined from the empirical covariance Σ and its precision matrix P = Σ−1. For sampled pairs (xi,xj), we compute

(xi − xj)⊤P(xi − xj) ∥xi − xj∥2

mij =

,

and summarize these values by a robust statistic such as the median. This quantity is closely related to the role of m in the Gaussian case. Indeed, if the layerwise representation distribution were exactly Gaussian with mean µ and precision matrix P, then

- 1

- 2

(x − µ)⊤P(x − µ) + const, and hence

U(x) =

∇2U(x) = P. In that setting, the strong-convexity constant is precisely

m = λmin(P).

Moreover, for any displacement vector δ, the Rayleigh quotient

δ⊤Pδ ∥δ∥2

measures directional curvature under the quadratic potential. Therefore, mˆ mono can be interpreted as an empirical summary of the typical directional stiffness of the representation space. Although it is not an exact lower bound on the Hessian, it captures the same global geometric intuition: layers with larger mˆ mono behave as if they are embedded in a more strongly restoring global geometry.

Why mˆ curv is also an m-like quantity. Our local curvature proxy is based on the covariance of local neighborhoods. For each anchor point, we compute the covariance matrix Σlocal of its k-nearest neighbors and define a local score proportional to

1 λmax(Σlocal)

.

The layer-level statistic mˆ curv is then obtained by aggregating these local values across anchors. This proxy is motivated by a local quadratic approximation. If a small neighborhood of the representation distribution is approximately Gaussian, then its local density may be written as

plocal(x) ∝ exp −

- 1

- 2

(x − µlocal)⊤Plocal(x − µlocal) ,

with Plocal ≈ Σ−local1 . Under this approximation, the local Hessian of the negative log-density is given by Plocal, and the corresponding local strong-convexity scale is

1 λmax(Σlocal)

λmin(Plocal) = λmin(Σ−local1 ) =

.

Thus, mˆ curv measures how compact and sharply curved the local representation neighborhoods are. Larger values indicate smaller local spread along the most variable direction, which is consistent with the intuition of stronger local restoring geometry.

Why kˆ reflects intrinsic dimension. The theoretical quantity k is intended to capture the intrinsic complexity of the representation distribution, rather than its ambient dimension d. In our empirical procedure, we measure this using the effective rank of the covariance:

tr(Σ) ∥Σ∥

kˆ = reff(Σ) =

.

This quantity has a direct interpretation in idealized low-dimensional settings. Suppose the data are supported exactly on a k-dimensional isotropic subspace with covariance eigenvalues

λ1 = ··· = λk = σ2, λk+1 = ··· = λd = 0. Then

kσ2 σ2

reff(Σ) =

= k.

Hence, in this ideal case, the effective rank exactly recovers the intrinsic dimension.

More generally, when the representation distribution is concentrated near a low-dimensional manifold or subspace, the covariance spectrum typically contains a small number of dominant eigenvalues and many small residual ones. In such cases, reff(Σ) measures the effective number of active variation directions. For our purposes, this is the relevant notion of intrinsic dimension, since diffusion complexity depends not on the nominal ambient dimension but on how many directions carry meaningful variation.

Interpretation and limitation. Taken together, mˆ mono and mˆ curv serve as global and local proxies for curvature-related restoring geometry, while kˆ serves as an operational measure of intrinsic dimension. These quantities are not exact estimators of the theoretical constants in Section 3.1. Instead, they are empirical surrogates designed to preserve the same geometric design principle: diffusion-friendly layers should be locally compact, globally stable, and effectively low-dimensional.

This distinction is important. Our empirical layer-selection score should therefore be interpreted as an operational criterion motivated by theory, rather than as a direct numerical estimate of the exact constants appearing in the theorems.

### C Geometric Proxy Estimation and Layer Selection

This appendix describes the implementation details of the empirical geometric proxies used in the Locate stage, together with the practical layer-selection procedure. Where Appendix B explains why these quantities can be interpreted as theory-motivated surrogates for curvature and intrinsic dimension, the present appendix focuses on how they are computed in practice and how they are combined into a robust empirical selection rule.

#### C.1 Representation Extraction

For each transformer layer, we begin from hidden activations of shape

acts ∈ RN×S×H,

where N denotes the number of sequences, S the sequence length, and H the hidden dimension. For example, a layer output may have shape [B,S,H] = [32,1024,4096].

To compute geometric statistics, we convert these activations into a set of representation vectors

x ∈ RM×D,

using an extraction function that supports several pooling modes.

Mean pooling. In the mean setting, each sequence is mapped to a single vector by masked averaging over valid tokens:

S t=1 ant hnt

xn =

,

S t=1 ant

where ant ∈ {0,1} is the attention mask and hnt ∈ RH is the hidden state at token position t. This yields one representation vector per sequence.

Last-token pooling. In the last setting, we take the hidden state of the final valid token in each sequence. This again yields one vector per sequence, while preserving a token-level representation associated with the sequence endpoint.

Tokenwise extraction. In the token setting, all valid token representations are flattened across sequences and positions. If necessary, a random subset is retained to control memory and computational cost. This yields a larger collection of vectors and allows the layer geometry to be estimated directly from tokenwise activations.

Mask handling. Padding positions with attention_mask=0 are excluded from all computations. Special tokens are retained whenever they are marked as valid by the attention mask. Thus, the extracted representation set reflects the actual active tokenwise computation of the model.

Recorded representation statistics. For transparency and reproducibility, we store layerwise summary statistics describing the extracted representation set, including the number of sequences, sequence length, hidden dimension, number of vectors before and after projection, and the corresponding feature dimensions. These quantities make it possible to verify that proxy estimation is based on comparable representations across layers. In the main experiments, we use mean pooling and retain at most max_seqs=2,000 sequences and max_tokens=200,000 valid tokens per layerwise geometry run.

#### C.2 Preprocessing and Random Projection

Because hidden representations can be high-dimensional, we optionally apply random projection before estimating the geometric proxies. This serves two purposes: it improves numerical stability in covariance-based calculations, and it reduces the computational cost of repeated layerwise estimation.

Given vectors x ∈ RM×D, we construct a Gaussian random matrix

R ∈ RD×d

,

proj

followed by QR orthonormalization to obtain an approximately orthonormal projection basis. The projected representations are then

xproj = xR ∈ RM×d

.

proj

If the projection dimension satisfies dproj ≤ 0 or dproj ≥ D, projection is skipped and the original vectors are used. This design ensures that the projection acts only as a computational device and does not alter the pipeline when dimensionality reduction is unnecessary.

The main hyperparameters governing this step are the projection dimension, the pooling mode, the maximum number of sequences or tokens retained, and the ridge regularization used in subsequent covariance inversion.

#### C.3 Local Curvature Proxy

To capture local geometric compactness, we estimate a curvature-inspired proxy from neighborhoods in representation space. For a given layer representation set x = {x1,...,xM} ⊂ RD, we sample a set of anchor points and, for each anchor, identify its k nearest neighbors under Euclidean distance. Distances are computed on the same representation matrix used for proxy estimation, i.e., after the optional projection step when projection is enabled.

Let Ni denote the neighborhood of anchor xi. We compute the local covariance matrix

Σ(locali) = Cov(Ni) + λI,

where λ > 0 is a small ridge term added for numerical stability. The local curvature score is then defined as

1 λmax Σ(locali)

m(curvi) =

.

Intuitively, this quantity is large when the local neighborhood is compact even along its most variable direction, which is consistent with the notion of strong local restoring geometry.

To obtain a layer-level statistic, we aggregate the anchorwise values using a robust summary:

mˆ curv = mediani m(curvi) .

In addition, we record quantiles such as the 25th, 50th, and 75th percentiles in order to characterize variation across local neighborhoods and to support uncertainty-aware visualization.

#### C.4 Monotonicity Proxy

To capture a global notion of directional stiffness, we compute a monotonicity-inspired proxy from the empirical covariance of the layer representations. Let

Σ = Cov(x) + λI be the ridge-regularized empirical covariance and let

P = Σ−1

denote its precision matrix. We then sample random pairs (xi,xj) and compute

(xi − xj)⊤P(xi − xj) ∥xi − xj∥2

mij =

.

This is a Rayleigh-quotient-type quantity that measures how strongly a pairwise displacement is penalized by the global precision geometry.

The layer-level monotonicity proxy is defined by a robust summary over sampled pairs:

mˆ mono = median(i,j)(mij).

As with the local curvature proxy, we additionally record quantiles and confidence intervals obtained by bootstrap resampling. In the main experiments, we report 95% bootstrap percentile intervals for the median monotonicity estimate. These summaries allow us to assess not only the central tendency of global directional stiffness, but also its stability under finite-sample variation.

#### C.5 Effective Rank

To estimate the effective intrinsic dimension of the layerwise representation distribution, we compute the effective rank of the empirical covariance:

tr(Σ) ∥Σ∥

kˆ = reff(Σ) =

,

where ∥Σ∥ denotes the spectral norm, i.e., the largest eigenvalue of Σ.

This quantity measures the effective number of active directions of variation in the representation space. A smaller value indicates that the representation is concentrated along fewer dominant directions, which is favorable from the perspective of diffusion complexity.

In the main text, we use effective rank as the primary intrinsic-dimension statistic. In addition to the central estimate kˆ, we optionally record related quantities such as the participation ratio and bootstrap summaries as supplementary diagnostics. To reflect finite-sample uncertainty, we also store quantiles such as

kˆq25, kˆq50, kˆq75.

#### C.6 Selection Score Construction

After computing mˆ curv, mˆ mono, and kˆ for each layer, we combine them into a single empirical layer-selection score.

Baseline score. A simple baseline motivated by the curvature–dimension principle is

selection_scorebase(ℓ) = z(log mˆ curv(ℓ)) − z(log kˆ(ℓ)), where z(·) denotes z-score normalization across layers within the same model. Intuitively, the baseline rewards curvature-related structure while penalizing excessive effective dimension.

Final score used in practice. In the final implementation, we use a fixed score based on logtransformed layerwise statistics:

selection_score(ℓ) = α1z(log mˆ curv(ℓ)) + α2z(log mˆ mono(ℓ)) + α3z(log kˆ(ℓ)). We use the fixed coefficients

(α1,α2,α3) = (1.0, 1.0, −1.0). Thus, the final score becomes

selection_score(ℓ) = z(log mˆ curv(ℓ)) + z(log mˆ mono(ℓ)) − z(log kˆ(ℓ)).

This score reflects the curvature–dimension principle. First, both local and global curvature-related quantities are rewarded through mˆ curv and mˆ mono. Second, effective intrinsic dimension is penalized through kˆ. We use only a linear effective-rank penalty because the sensitivity analysis below shows that an additional quadratic penalty can over-penalize some low-rank layers and lead to unstable layer selection.

The selected target layer is then defined by

ℓ∗ = arg max

selection_score(ℓ). For convenience, we also report

ℓ

predicted_loss(ℓ) = −selection_score(ℓ),

so that lower predicted loss corresponds to a more diffusion-friendly layer.

Sensitivity to coefficient choice. To address concerns that the fixed coefficients might implicitly overfit, we evaluate a small set of coefficient perturbations around the final score and report (i) the selected layer ℓ∗, (ii) the validation loss at ℓ∗, and (iii) rank correlations between the score and the validation loss. Table 5 summarizes the results computed from layerwise_geometry.json and teacher_loss_frozen.csv (excluding layer 0). The final linear score selects the oracle-best layer in this sweep while maintaining a strong monotonic relationship with validation bridge loss. By contrast, adding or strengthening the quadratic effective-rank penalty can shift the selected layer from early layers to layer 6, which yields a large degradation in validation loss in this setting. This suggests that effective rank is useful as a soft complexity penalty, but overly strong rank penalties can hurt top-layer selection.

#### C.7 Implementation Details of the Diffusion Bridge

In the embedding-conditioned setting used in our experiments, the conditioning signal c(x) is the dumped activation tensor embedding_out ∈ RB×S×H, i.e., the hidden state immediately before the first transformer block of the source language model. This tensor is passed through a learned condition projection and mapped to the 768-dimensional encoder_hidden_states interface expected by the pretrained UNet. Thus, the diffusion backbone is used as a conditional latent denoiser over language-model hidden states rather than as a text-to-image model.

To instantiate the bridge, we adopt a Stable-Diffusion-style latent denoising architecture built on a pretrained UNet. The target hidden state hℓ∗ is reshaped and projected into an image-like tensor, encoded by a frozen VAE into a latent zℓ∗, noised to zt at timestep t using a fixed diffusion scheduler, and denoised by the UNet to predict ϵˆθ(zt,t,c). The denoised latent is then decoded and projected back to the original hidden space to obtain the reconstructed hidden state hˆℓ∗.

Preset (α1, α2, α3, α4) ℓ∗ L(ℓ∗) oracle ℓ gap Spearman

final, no k2 (1.0,1.0,-1.0,0.0) 2 0.059 1 0.028 0.940 baseline w/ k2 (1.0,1.0,-1.0,-0.5) 2 0.087 1 0.028 0.859

- curv×0.5 (0.5,1.0,-1.0,-0.5) 6 139.092 1 139.033 0.729
- curv×1.5 (1.5,1.0,-1.0,-0.5) 2 0.087 1 0.028 0.916

- mono×0.5 (1.0,0.5,-1.0,-0.5) 6 139.092 1 139.033 0.733
- mono×1.5 (1.0,1.5,-1.0,-0.5) 2 0.087 1 0.028 0.916

- klin×0.5 (1.0,1.0,-0.5,-0.5) 2 0.087 1 0.028 0.950
- klin×1.5 (1.0,1.0,-1.5,-0.5) 6 139.092 1 139.033 0.725 ksq×0.5 (1.0,1.0,-1.0,-0.25) 2 0.087 1 0.028 0.898 ksq×2.0 (1.0,1.0,-1.0,-1.0) 6 139.092 1 139.033 0.733

- Table 5: Sensitivity of layer selection to coefficient perturbations. Numbers are produced by our analysis script. The final score removes the quadratic effective-rank penalty and uses only a linear curvature–dimension trade-off.

Hidden-to-image layout. Given a selected-layer hidden state hℓ∗ ∈ RB×S×H, we first reshape the sequence dimension into a fixed spatial grid and apply a learned 1 × 1 projection to map the hidden dimension to the channel dimension required by the latent denoising backbone. This produces an image-like tensor of shape (C,Himg,Wimg). The reshape is deterministic and does not use image supervision. After denoising, an inverse learned projection maps the output back to RB×S×H.

In our experiments with S = 1024 and hidden size H = 4096, we use Himg = 32, Wimg = 32, and C = 3, matching the VAE input-channel interface of SD-v1.5. This channel projection should be understood as an architectural interface to the pretrained latent denoising stack, not as an assumption that language hidden states are naturally RGB images.

We do not assume that language hidden states have natural 2D image semantics. The Stable-Diffusionstyle backbone is used only as a high-capacity conditional denoiser. The 2D layout is therefore an architectural interface rather than a semantic image representation. To test this design choice, we compare it against sequence-native deterministic and denoising backbones in Appendix D.1.

For the late-stage alignment losses, the language-modeling term is defined as LLM = CE(pbridge(xi+1 | x≤i),xi+1), and the temperature-scaled distillation term is LKD = T2 KL(softmax(oteacher/T)∥softmax(obridge/T)), where oteacher and obridge denote the teacher and bridge logits, respectively.

During training, we update the UNet denoiser together with the bridge-specific projection modules, including the hidden-to-image mapping, the image-to-hidden mapping, and the condition projection. By contrast, the VAE is kept frozen and the diffusion scheduler is fixed. Under the embeddingconditioned setting described above, no Stable-Diffusion text-conditioning pathway is used. This isolates the Replace stage as a hidden-space learning problem: the bridge is trained to reproduce the selected-layer hidden state while remaining compatible with the retained upper transformer stack.

### D Experimental Setup Details

This appendix provides experimental and implementation details supplementing Section 4. It describes the data sampling pipeline, representation extraction, geometric proxy estimation, and optimization hyperparameters used for the layer-sweep and full-training experiments.

Data sampling and activation dumps. We construct the activation corpus from a local raw-text copy of Dolma v1.7. The dump pipeline samples up to 300,000 text sequences using stratified round-robin sampling over 700 source files, with up to 500 samples per file and random seed 42. Each sequence is tokenized with the corresponding source-model tokenizer and passed through the backbone model to save the embedding output, all decoder-layer outputs, input IDs, and attention masks. Activations are computed in reduced precision and stored as float16 shards for subsequent geometry estimation and bridge training.

Representation extraction. Sequences are tokenized with the source tokenizer under its default special-token handling, so special tokens are retained in the stored input IDs and activation tensors.

Padding positions are tracked by the attention mask and excluded whenever masked averaging is used. When token IDs are decoded back into text for prompt reconstruction, special tokens are removed. For geometry estimation, the implementation supports last-token pooling and token-level sampling, but we use mean pooling by default.

Geometry estimation. The default geometry pipeline uses no random projection (dproj = 0), ridge coefficient 10−3, k = 64 nearest neighbors for local curvature, 512 anchor points, 200,000 sampled pairs for monotonicity, and bootstrap resampling with 95% confidence intervals. In the main experiments, layerwise geometry proxies are estimated from repeated subsamples as described in Section 4.1.

Bridge training. Bridge training uses a deterministic shard-level split with validation ratio 0.1 and split seed 42. In the fixed-budget layer sweep, each candidate layer is trained for one epoch with up to 150,000 training examples, batch size 4, learning rate 3 × 10−5, AdamW optimization, mixed-precision FP16, and a maximum of 37,500 optimization steps. We use bridgeability to mean how easily a layer’s hidden state can be reconstructed by the diffusion bridge under this fixed budget, and measure it by validation bridge loss. Training losses are reported as auxiliary optimization diagnostics. The codebase also supports applying the auxiliary next-token language-modeling loss LLM and teacher-student distillation loss LKD only in the final epoch.

Compute resources. The fixed-budget layer-sweep experiments were conducted on NVIDIA H100 GPUs with a matched 40 GPU-hour budget. Each candidate layer was trained for one epoch on 150K examples using batch size 4 and mixed-precision FP16. The final full-training experiments, which are used for the main model-quality evaluation, were conducted on NVIDIA B200 GPUs for 40 hours per main run, training the selected bridge for four epochs on the 300K-example corpus. We additionally report peak GPU memory, throughput, and latency in the final evaluation table. Activation dumping stores float16 hidden states for the evaluated 8B-scale backbones, and therefore storage requirements scale approximately linearly with the number of layers and sampled sequences.

#### D.1 Diffusion Bridge Architecture and Backbone Choices

Backbone choice. We choose a Stable-Diffusion-style UNet bridge because the replacement module must solve a high-dimensional conditional denoising problem over continuous representations, rather than a standard next-token prediction task. The target is an internal boundary hidden state hℓ∗, and the conditioning signal is the embedding-derived representation of the same input. We therefore require a backbone that can denoise structured, high-dimensional activations while exploiting the conditioning signal effectively.

- Table 6 compares several bridge backbones under the same validation setting on layer 16 (S=512, N=500). The Stable-Diffusion-style UNet bridge achieves the lowest validation loss among the evaluated trainable alternatives. Its mean loss (1163.59) is lower than the MLP-based hidden DDPM bridge (1411.36), corresponding to a 17.6% reduction. It also substantially outperforms the Transformer-based and 1D-convolutional bridges, reducing validation loss by 58.4% relative to ddpm_hidden_transformer and 68.3% relative to ddpm_hidden_conv1d.

These results suggest that reusing the Stable-Diffusion UNet as a conditional denoising backbone is a practical choice in our setting. We do not claim that this architecture is optimal in general; rather, it provides the strongest validation performance among the tested backbones under the same small-scale ablation. Importantly, the model is not used as an image generator and does not rely on image supervision; we only reuse the latent denoising structure as a conditional backbone for reconstructing language-model hidden states.

### E Baseline Details

Parameter accounting. Table 7 reports parameter counts for the same-budget representation-space comparison. These counts are based on the current implementation and notebook configuration used in our experiments. Trainable parameters are those updated by the optimizer. Active bridge/module parameters are parameters used in the forward pass for the corresponding diffusion or recovery module, including frozen modules when applicable. For DiHAL, the retained pretrained upper

Table 6: Validation bridge loss on layer 16 (S=512, N=500).

Bridge Layer Split S N Mean loss ↓

stable_diffusion_UNet 16 valid 512 500 1163.587321 ddpm_hidden 16 valid 512 500 1411.363999 ddpm_hidden_transformer 16 valid 512 500 2799.283554 ddpm_hidden_conv1d 16 valid 512 500 3672.299440

Method Trainable params Active bridge/module params DiHAL 862.7M 946.3M bridge Diffusion-LM 93.7M 93.7M SED 73.7M 73.7M LD4LG 25.6M AE + 188M diffusion 25.6M AE + 188M diffusion CoDAR 130M diffusion + 230.9M AR decoder 360.9M

- Table 7: Trainable and active parameter counts for the same-budget representation-space comparison. Counts are measured or estimated under our experimental configurations. Trainable parameters are updated by the optimizer, while active bridge/module parameters are used in the forward pass, including frozen modules where applicable. For CoDAR, the count is estimated from the reported MDLM-style diffusion backbone and GPT-2-small-style autoregressive decoder with cross-attention under the Qwen tokenizer vocabulary.

transformer layers and original LM head are not counted as trainable bridge parameters, but they are part of the active language-modeling interface.

F Inference Cost Detail G Inference Cost Detail

Because DiHAL replaces a transformer prefix with a diffusion bridge, its end-to-end inference cost depends on two factors: the insertion depth and the number of denoising steps. We therefore measure latency, throughput, peak memory, and the number of function evaluations (NFEs) across several insertion layers. The measurement includes the full hybrid inference path: bridge denoising, retained upper transformer layers, and the original LM head.

- Table 8 shows the resulting cost trade-off. For the geometry-selected shallow insertion layers, DiHAL is slower than the original backbone even with a single denoising step, and latency increases substantially as NFEs grow. This indicates that bridge denoising is the dominant overhead in the current implementation. Deeper insertion layers can reduce latency at NFE=1 by skipping more transformer layers, but these layers are not selected by the geometry criterion and are less reliable as hidden-state reconstruction targets. Peak memory also increases because the diffusion bridge adds extra active modules.

Overall, these results support our limitation claim that the current DiHAL implementation should not be interpreted as an end-to-end acceleration method. Instead, the table clarifies the cost-quality trade-off: deeper replacement can reduce retained-transformer computation, but diffusion denoising overhead and hidden-state reconstruction difficulty limit practical acceleration in the present setting.

### H Existing assets and licenses.

Our experiments use publicly available pretrained backbones, datasets, and model components. We use Llama-3.1-8B-Instruct under the Llama 3.1 Community License and Qwen3-8B under the Apache 2.0 license. We use Dolma v1.7 for activation extraction and bridge training under the ODC-BY license, and WikiText-103 for evaluation under its Creative Commons/GFDL licensing terms. The diffusion bridge reuses Stable Diffusion v1.5 components under the CreativeML OpenRAIL-M license, and generative perplexity is computed using a GPT-2 evaluator under the GPT-2 license

Table 8: End-to-end inference cost. DiHAL cost includes bridge denoising, retained upper transformer layers, and the LM head.

Model Method Insert. layer NFEs Latency/token ↓ Throughput ↑ Peak mem. ↓

Llama-3.1-8B Original – – 0.043 23509 16.2 Llama-3.1-8B DiHAL 3 1 0.074 13481 18.0 Llama-3.1-8B DiHAL 3 4 0.150 6681 18.0 Llama-3.1-8B DiHAL 3 20 0.503 1989 18.0 Llama-3.1-8B DiHAL 10 1 0.065 15314 18.0 Llama-3.1-8B DiHAL 10 4 0.146 6844 18.0 Llama-3.1-8B DiHAL 10 20 0.515 1941 18.0 Llama-3.1-8B DiHAL 20 1 0.051 19467 18.0 Llama-3.1-8B DiHAL 20 4 0.116 8597 18.0 Llama-3.1-8B DiHAL 20 20 0.503 1988 18.0 Llama-3.1-8B DiHAL 30 1 0.035 28676 18.0 Llama-3.1-8B DiHAL 30 4 0.114 8798 18.0 Llama-3.1-8B DiHAL 30 20 0.502 1993 18.0

Qwen3-8B Original – – 0.059 17033 15.9 Qwen3-8B DiHAL 2 1 0.081 12375 17.6 Qwen3-8B DiHAL 2 4 0.130 7668 17.6 Qwen3-8B DiHAL 2 20 0.503 1989 17.6 Qwen3-8B DiHAL 12 1 0.066 15222 17.6 Qwen3-8B DiHAL 12 4 0.132 7554 17.6 Qwen3-8B DiHAL 12 20 0.506 1977 17.6 Qwen3-8B DiHAL 22 1 0.047 21156 17.6 Qwen3-8B DiHAL 22 4 0.119 8428 17.6 Qwen3-8B DiHAL 22 20 0.500 2002 17.6 Qwen3-8B DiHAL 32 1 0.036 27581 17.6 Qwen3-8B DiHAL 32 4 0.093 10769 17.6 Qwen3-8B DiHAL 32 20 0.500 2001 17.6

terms. We cite the original papers and model or dataset sources for all existing assets and use them only for research evaluation consistent with their respective terms.

