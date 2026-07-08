[Figure 1]

May, 2026.

# arXiv:2606.21906v1[cs.CL]20Jun2026

## Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer Decoding

##### Xuanming Zhang*1, Sining Zhoubian*2, Yuxuan Chen1, Tianyi Tang1, An Yang1, Sean Du3, Chujie Zheng1, Fei Huang1, Dayiheng Liu1, Gao Huang2, Jingren Zhou1 1 Qwen Team, Alibaba Inc. 2 Tsinghua University 3 Nanyang Technological University

Project Page: https://github.com/QwenLM/Confident-Decoding

### Abstract

Autoregressive generation in large language models (LLMs) conventionally decodes from the final layer, assuming that deeper representations yield more reliable next-token predictions. We revisit this assumption by revealing a recurring Guess–Refine–Perturb dynamic: early layers form coarse guesses, intermediate layers refine reasoning-relevant semantics, and final layers can perturb these refined predictions toward generic or alignment-preferred tokens. We introduce Confident Decoding, a training-free decoding strategy that dynamically selects the most reliable near-final layer through entropyguided conservative backward search. We further provide a theoretical formulation of layer selection as an optimal stopping problem, showing that under bounded projection noise and dominant late-stage alignment perturbation, our search rule filters perturbation while bounding the loss relative to the oracle refinement layer. Experiments across dense and Mixture-of-Experts LLMs demonstrate consistent gains on challenging reasoning benchmarks, including GPQA-Diamond, Omni-MATH, and HLE, with zero memory overhead and less than 2% latency increase. These results suggest dynamically bypassing final-layer perturbations can unlock stronger reasoning behavior from aligned LLMs.

[Figure 2]

Figure 1: Token substitutions produced by Confident Decoding on Qwen3.5-35B-A3B. When the two decoding strategies disagree (∼2% of generated tokens), Standard Decoding (right) selects generic, highfrequency function words and punctuation (e.g., “the”, “is”, “so”, “.”) characteristic of alignment-induced common-word bias in the final layers. Confident Decoding (left) instead commits at the entropy valleythe layer of peak confidence before late-stage perturbation—and recovers domain-specific, semantically precise terminology (e.g., “mass”, “radius”, “approximately”, “Cartesian”). Word sizes are proportional

to substitution frequency. The central curve illustrates the predictive entropy H(p(tl)) along decoder layer depth l. The “Trough” marks the entropy valley where the model’s internal confidence peaks;

beyond this point, alignment constraints in deeper layers drag predictions toward safe but uninformative continuations.

Emails: xuemuqiangu@gmail.com zbsn21@mails.tsinghua.edu.cn ∗Equal contribution.

### 1 Introduction

Autoregressive generation in large language models (LLMs) conventionally decodes each token from the final decoder layer. This standard practice implicitly assumes that representations become progressively more reliable with depth: shallow layers provide incomplete computations, intermediate layers refine contextual information, and the deepest layer produces the most accurate next-token distribution (Vaswani et al., 2017; Shazeer et al., 2017; Gupta et al., 2025). Under this view, the final layer is treated as the natural and optimal interface between the model’s internal computation and the output vocabulary.

However, recent evidence suggests that this monotonic-depth assumption can break down. Probing and layer-wise analyses show that intermediate layers often encode strong task-relevant semantics, while later layers may compress, redirect, or perturb information that has already been refined (Skean et al., 2025; Csordás et al., 2025). Through a detailed analysis of residual-stream dynamics, contribution norms, and prediction trajectories, we identify a recurring Guess–Refine–Perturb pattern in LLM forward passes (see §2.1 and Figure 2). Early layers produce coarse statistical guesses; intermediate layers refine these guesses along a stable semantic trajectory; and, for a subset of tokens, the final layers introduce a sharp representational shift that can move the prediction away from the most reasoning-relevant intermediate state.

We hypothesize that this late-stage perturbation is partly related to the objectives used in modern post-training. Contemporary LLMs are typically adapted through continued pre-training, supervised fine-tuning, and preference-alignment procedures such as RLHF, RLAIF, or DPO (Zixuan et al., 2023; Ouyang et al., 2022; Lee et al., 2024; Rafailov et al., 2023). These procedures improve instruction following, safety, and response style, but they may also encourage final-layer distributions to favor frequent, safe, or generic continuations (Askell et al., 2021; Zou et al., 2023). For ordinary conversational or safety-oriented prompts, such alignment behavior can serve as a useful guardrail. For complex reasoning tasks, however, the same late-stage bias may conflict with the task-specific reasoning trajectory formed in intermediate layers, producing what we call a planning–pragmatics tradeoff: the model may internally form a strong reasoning-oriented prediction, but the final layer may pragmatically shift the output distribution toward a more generic or alignment-preferred token.

Several recent decoding strategies have begun to exploit intermediate-layer information. Contrastive decoding methods, such as Decoding by Contrasting Layers (Chuang et al., 2023; Zhou et al., 2025), compare predictions across layers to suppress undesirable continuations, while logits-evolution methods study how token distributions change through depth (Zhang et al., 2024; Das et al., 2025; Zhang et al., 2025b). Early-exit methods also use intermediate confidence signals to reduce inference cost (Schuster et al., 2022; Yang et al., 2025). Yet these approaches either continue to treat the final layer as the main semantic anchor or focus primarily on efficiency rather than reasoning degradation. They do not directly ask whether the final layer should always be the layer from which the next-token distribution is emitted.

To address this question, we propose Confident Decoding, a training-free, drop-in decoding strategy that dynamically selects the most reliable near-final layer at each generation step. Importantly, Confident Decoding does not truncate the transformer or modify the model’s forward pass. The only difference is that, instead of always forwarding final-layer logits to the sampler, Confident Decoding computes candidate logits from a small near-final layer window and selects the first local entropy trough encountered by a conservative backward search. Since lower token entropy corresponds to a sharper predictive distribution, this rule identifies the point at which the model reaches high confidence before a potential late-layer perturbation emerges (see Figure 1 for a representative example).

We evaluate Confident Decoding across a broad range of challenging benchmarks, including general reasoning, mathematical problem solving, long-context understanding, coding, and safety, with datasets such as GPQA-Diamond (Rein et al., 2024), HLE (Phan et al., 2026), Omni-MATH (Gao et al., 2024), LongBench v2 (Bai et al., 2025), LiveCodeBench v6 (Jain et al., 2024), and Air-Bench-2024 (Zeng et al., 2025). Across dense and Mixture-of-Experts architectures, Confident Decoding consistently improves over standard greedy decoding and strong contrastive baselines, while introducing negligible memory overhead and less than 2% additional latency. We further observe that larger and stronger models can exhibit more pronounced late-layer perturbations, making dynamic layer selection increasingly beneficial as model capability and task difficulty grow. In summary, our contributions are threefold:

- • We identify a recurring Guess–Refine–Perturb dynamic in LLM forward passes, showing that final-layer representations are not always the most reliable source for reasoning-sensitive next-token prediction.
- • We propose Confident Decoding, a training-free and drop-in decoding strategy that preserves the full forward pass while dynamically selecting near-final logits through entropy-guided conservative backward search.

- 0.6

- 0.8

- 1.0

- 1.2

- 1.4

MeanRelativeContribution

I II

II III

1.6

Phase I

Phase II Total al+ml / hl III

Trend (smoothed)

Attn al / hl

MoE-MLP ml / hl+al

Full-attn layers (10/40)

(a) Relative Contribution Norm

|I II|Phase II|
|---|---|
| | |
| | |
|Phase I (outlier)| |
| | |

5 10 15 20 25 30 35 40 Layer index

0.0

0.2

0.4

0.6

0.8

1.0

cos(,)hh1

II III III

cos(hl, hl 1) Trend (smoothed) Full-attn layers (10/40)

(b) Residual I/O Cosine Similarity

- Figure 2: Layer-wise dynamics of Qwen3.5-35B-A3B on GSM8K (Cobbe et al., 2021). Gray bands mark the 10 full-attention layers (at l = 4,8, . . . ,40) which enable global token interactions. (a) Relative Contribution Norm: The first decoder layer’s update dominates the embedding magnitude; contributions stabilize through Phase II; the final full-attention layer (l=40) resurges markedly in Phase III. (b) Residual I/O Cosine Similarity: IO-CosSim remains high throughout Phase II, confirming directionally faithful refinement, before dropping sharply at l=40—the largest directional deflection in the Phase II–III regime.

• We provide theoretical (§3) and empirical (§5) evidence that Confident Decoding filters late-layer perturbations, preserves refined reasoning signals, and improves performance across diverse reasoning, coding, long-context, and safety benchmarks.

2 Preliminaries

In this section, we formally characterize the forward-pass dynamics of LLMs to motivate our proposed decoding strategy. We first quantify the layer-wise representational shifts to define the three-phase progression mathematically. Subsequently, we present a pilot study validating the superiority of dynamic, entropy-based layer selection over static exit paradigms.

- 2.1 Layer-wise Dynamics Consider an L-layer auto-regressive Transformer-based language model. Let V denote the vocabulary

- 0.4

- 0.2

- 0.0

5 10 15 20 25 30 35 40 Layer index

space. For a given input sequence up to step t, the residual stream at layer l, denoted as h(tl) ∈ Rd, is updated via the attention and feed-forward sub-layers. In the standard sequential (pre-norm) formulation:

h˜(tl) = h(tl−1) + fAttn(l) (h(tl−1)), (1) h(tl) = h˜(tl) + fFFN(l) (h˜(tl)), (2)

where fAttn(l) denotes the attention sublayer (full or linear) and fFFN(l) the feed-forward sublayer (MLP or MoE). We define the layer contribution vector as the total update applied by layer l:

m(tl) ≜ h(tl) − h(tl−1) = fAttn(l) (h(tl−1)) + fFFN(l) (h˜(tl))

so that h(tl) = h(tl−1) + m(tl). This definition naturally extends to parallel-attention variants where both sublayers receive h(tl−1) directly. To understand how each layer alters the semantic trajectory of the token prediction, we analyze two key metrics across network depth: the Relative Contribution Norm and the Residual I/O Cosine Similarity (Skean et al., 2025; Csordás et al., 2025).

h(tl) · h(tl−1) ∥h(tl)∥2∥h(tl−1)∥2

Norm Ratio(l) = ∥m(tl)∥2

, IO-CosSim(l) =

∥h(tl−1)∥2

Under the residual stream decomposition h(tl) = h(tl−1) + m(tl), Norm Ratio(l) characterizes the write intensity. When Norm Ratio(l) ≫ 1, the contribution dominates the existing state (h(tl) ≈ m(tl)), effectively overwriting the accumulated representation; when Norm Ratio(l) ≪ 1, the layer applies incremental corrections without displacing prior content; and when Norm Ratio(l) ≲ 1, the new write is comparable in magnitude to the carried residual (Elhage et al., 2021). In this intermediate regime, Norm Ratio(l) alone cannot distinguish constructive reinforcement from disruptive rewriting, because the net effect

depends on the update direction. IO-CosSim(l) therefore characterizes directional fidelity: in the highdimensional representation space where direction encodes semantic content, a value near 1 indicates the layer preserves and refines the existing semantic trajectory, while lower values indicate rotation into a semantically distinct subspace (Mikolov et al., 2013).

As shown in Figure 2, empirical observations on Qwen3.5-35B-A3B—a model interleaving 30 linearattention (DeltaNet) layers with 10 full-attention layers at fixed positions (shaded in gray)—reveal a consistent three-phase structure, which we term the Guess-Refine-Perturbation progression:

- 1. Phase I: Guess (Shallow Layers, l ≲ 0.15L). In the initial layers, Norm Ratio(l) is exceedingly high: Norm Ratio(1) ≈ 1.6, meaning the first decoder layer’s contribution vector is 1.6 times the

magnitude of the incoming embedding, so the output h(t1) is almost entirely determined by the layer’s computation rather than the token embedding itself. Correspondingly, IO-CosSim(1) ≈ 0.67markedly below the Phase II plateau—indicating that the model undergoes a substantial directional shift in this phase, rapidly constructing an initial latent representation under high uncertainty.

- 2. Phase II: Refine (Intermediate Layers, 0.15L ≲ l ≲ 0.95L). Norm Ratio(l) drops sharply and stabilizes in the range 0.23–0.57 (< 1): the sublayer contribution is substantially smaller than the existing residual, so each layer performs incremental, directionally faithful updates that progressively integrate contextual information without displacing the accumulated representation. Correspondingly,

IO-CosSim(l) remains consistently high (0.91–0.97) throughout, confirming that the semantic trajectory is refined but not rotated. The model refines its token predictions along a stable semantic trajectory.

- 3. Phase III: Perturbation (Post Layers, l ≳ 0.95L). In the final layers, Norm Ratio(l) re-elevates

markedly above the Phase II plateau, peaking at Norm Ratio(40)— 2–3× the Phase II level. This value remains below 1 and thus does not enter the Phase I overwriting regime. They indicate a write whose magnitude is comparable to the incoming residual rather than a small corrective update. The corroborating evidence comes from IO-CosSim(40) ≈ 0.69—equivalently, an output-state rotation relative to the incoming residual—which falls well below the Phase II band (0.91–0.97) and marks the sharpest directional deflection outside the initial embedding phase. The combination of these two metrics indicates a structurally significant, directionally misaligned update that partially rewrites and biases the semantic trajectory refined during Phase II. We formalize the mechanism behind this perturbation in §3.2.

This Phase III perturbation breaks the monotonic improvement assumption, suggesting that extracting predictions at the end of Phase II may yield superior reasoning accuracy over decoding from the final layer (Gupta et al., 2025). This three-phase pattern is further validated on additional architectures in Appendix B.1.

##### 2.2 Motivation: Emerging Entropy Valley in Intermediate Layer

Prior studies and observations in §2.1 suggest that LLMs often converge on stable semantic representations before Phase III (Skean et al., 2025). This motivates us to seek a reliable intermediate output point or proxy to bypass potential late-stage perturbations without the overhead of retraining auxiliary classifiers.

Limitations of Static Early Exit. We first investigate the performance of a naive static early exit strategy. As shown in Figure 3(a), an independent Bernoulli probability p represents the likelihood of overriding the final layer and forcing decoding from a fixed shallower layer (L − k) at any given token generation step. As this execution probability p increases, the overall model accuracy undergoes a precipitous decline. This observation indicates that static truncation ignores the inherent variance in token complexity; a uniform exit policy prematurely interrupts the essential computation required for “hard” tokens, thereby destroying the model’s reasoning integrity. Consequently, a robust extraction mechanism must be token-adaptive, requiring a dynamic indicator to identify the optimal depth for each hidden state.

Entropy as a Confidence Metric. A natural candidate for such an indicator is the model’s predictive confidence. By applying the pre-trained language modeling head WU to intermediate hidden states h(tl), we can elicit a layer-wise probability distribution p(tl). To ensure dimensional consistency and avoid softmax collapse, we apply the final-layer normalization before projection:

p(tl) = Softmax WU · RMSNormL(h(tl)) (3)

The uncertainty at each layer is then quantified by the Shannon entropy H(p(tl)) = − ∑ p(tl) log p(tl). We notice that using the original WU as a zero-shot probe introduces a basis shift problem (Belrose et al.,

[Figure 3]

- Figure 3: Comparison of layer-selection strategies on GPQA-Diamond, using Qwen3.5-35B-A3B as the base model. Dynamic valley selection consistently outperforms last-layer decoding and static early exits, while selecting dynamic layers neighboring the valley also degrades performance. This supports our central claim that the optimal decoding depth is token-dependent and typically lies near the Phase II/III boundary rather than at a fixed layer, and the entropy valley provides a confidence boundary for optimized performance.

2023), as WU is explicitly optimized for the final layer’s latent space. While representation homogeneity in the latter half of modern LLMs mitigates this (Wang et al., 2025), the mapping error still increases as we move further away from the final layer. This creates a trade-off: deeper layers offer smaller mapping bias, while certain intermediate layers might host more “confident” or “pure” semantic information before alignment-induced noise or over-smoothing occurs.

To balance mapping reliability with prediction confidence, we propose the Entropy Valley—the local entropy minimum encountered when scanning backwards from the final layer. Intuitively, this valley represents a state where the model has reached a consensus on the output distribution before the final layers potentially introduce detrimental oscillations.

Empirical Validation. Figure 3 illustrates the superiority of this dynamic selection. Unlike static exits, the Entropy Valley strategy maintains high accuracy even when applied frequently. Furthermore, Figure 3(b) demonstrates that the valley itself is a precise optimal boundary: decoding from its immediate neighbors (Valley±k) leads to consistent performance degradation. These results confirm that the entropy valley serves as a reliable, token-dependent proxy for the zenith of the refinement phase, justifying its use as the core boundary for our proposed rollback mechanism.

### 3 Theoretical Grounding

To formalize the superiority of Confident Decoding, we analyze the forward-pass dynamics through the lens of Information Theory and Representation Engineering. We demonstrate why the representational optimum emerges at an intermediate depth and prove that our conservative backward search acts as a mathematically optimal, adaptive filter that selectively neutralizes alignment perturbations while preserving safety guardrails.

##### 3.1 Information-Theoretic View of Layer Dynamics

While the Information Bottleneck (IB) principle (Tishby and Zaslavsky, 2015) typically characterizes neural network training, the learned weight matrices internalize this optimization, dictating the layerwise information flow during inference. Let X represent the input context and Ylogic represent the

optimal target token strictly derived from domain-specific reasoning. The residual stream h(tl) evolves by navigating the IB tradeoff:

LIB = I(X; h(tl)) − βI(h(tl);Ylogic)

During Phase I (Guess), the network aggressively compresses the superficial input X (minimizing I(X; h)). This severe dimensionality reduction induces geometric volatility, rendering the predictive

H(pt(l)) Perturbed tokens (32879/202935, 16.2%) Phase III present Qwen3.5-35B-A3B · GPQA Diamond

12.5

V*

Vonset

10.0

Zoom: tail rise

()[nats]()lHpt

5

7.5

0

5.0

36 37 38 39 40

Mean ±1

- Phase II

- Phase III

2.5

Full-attn (10/40)

Phase I

0.0

10 20 30 40

Layer index l

(a) Perturbed tokens (16.2%) — Phase III present

H(pt(l)) Unperturbed tokens (170056/202935, 83.8%) Phase I II only Qwen3.5-35B-A3B · GPQA Diamond

12.5

| | | | | | | | |V|onset| | | | |V|*|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | |Mean ±1<br><br>Phase|I|Phas<br><br>Full-|e at|II tn (10|/40)| | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

10.0

()[nats]()lHpt

7.5

5.0

2.5

0.0

10 20 30 40

Layer index l

(b) Unperturbed tokens (83.8%) — no Phase III

- Figure 4: Mean logit-lens entropy H(p(tl)) per layer for Qwen3.5-35B-A3B on GPQA Diamond (N = 50 prompts, 4,096 generated tokens per prompt, 202,935 tokens total). (a) Perturbed tokens (16.2%,

∆H = +0.37 nats): Tokens already at low entropy at V∗ (H¯V∗ = 0.52 nats). The final full-attention layer introduces an upward perturbation, the alignment-tax signature, disrupting a nearly committed prediction. These tokens undergo Phase III. (b) Unperturbed tokens (83.8%, ∆H=−2.52 nats): Content tokens still uncertain at V∗ (H¯V∗ =2.78 nats). The final layer continues its Phase II refinement, driving entropy toward zero with no perturbation. The backward scan exploits this heterogeneity: it selects V∗ for perturbed tokens (bypassing Phase III) and L for unperturbed ones (utilizing full refinement). The same per-token heterogeneity is confirmed on additional models in Appendix B.2.

entropy H(p(tl)) highly unstable and susceptible to degenerate unigram biases. Critically, the residual influence of this compression noise carries over into the early portion of Phase II, leaving the predictive

entropy on a high-entropy plateau that is empirically indistinguishable from Phase I, as shown in Figure 4. The network then transitions into Phase II (Refine), where successive attention mechanisms rigorously integrate contextual dependencies, dominating the flow to maximize I(h(tl);Ylogic). Within Phase II, we identify an internal inflection layer Vonset at which the carry-over of Phase I dissipates and the network enters its monotone refinement regime. Ideally, within this restricted domain l ≥ Vonset, the predictive entropy H(p(tl)) serves as an empirical upper bound for the true conditional entropy H(Ylogic|h(tl)). As mutual information monotonically increases across this late-refine window, H(p(tl)) follows a strict monotone downward trajectory.

The minimum at V∗ represents the exact representational zenith before potential perturbation sets in. As shown in Figure 4(a), on Qwen3.5-35B-A3B over GPQA Diamond generating 4,096 tokens per prompt,

- Phase II accumulates ∆H ≈ −10.8 nats across 202,935 tokens (Vonset≈28, V∗ =39). Crucially, Phase III is not a fixed architectural region but a per-token phenomenon determined by whether the final layer perturbs a token’s prediction: 16.2% of tokens exhibit an entropy rise of +0.37 nats at l = 40— these tokens undergo Phase III. Figure 1 shows examples of such tokens being replaced by their lower-entropy alternatives from an earlier layer. The remaining 83.8% of tokens still resolving at V∗ show a further entropy drop of −2.52 nats—these tokens simply complete their normal Phase I→II refinement through the final layer.

3.2 Modeling Alignment: Tax vs. Guardrail

Modern LLMs undergo rigorous post-training to align with human preferences. Recent findings in Representation Engineering (Zou et al., 2023) reveal that such alignment behaviors are governed by low-rank steering vectors, predominantly activated in the final layers to modulate outputs without overwriting pre-trained knowledge. Thus, in Phase III (Perturbation), the latent representations optimize a regularized risk toward a generic, safe distribution Palign:

R(l) = E[− log p(tl)(Ylogic|X)] + λDKL(Plogic(l) ∥Palign) where λ governs the steering intensity. Let V∗ be the oracle transition boundary between Phase II and

- Phase III. For l → L, the network induces an additive perturbation δalign(l) to minimize the KL divergence:

∗)

h(tl) ≈ h(V

t + ∑lk=V∗+1 δalign(k) .

Crucially, the destructive nature of this perturbation is strictly conditional:

- • Alignment as a Guardrail (Safety Tasks): For standard conversational or safety queries, Plogic ≈ Palign. The KL divergence is negligible. The late-stage layers refine formatting without semantic disruption.
- • Alignment as a Tax (Complex Reasoning): For domains like rigorous mathematics and science, the

specialized logic distribution Plogic sharply conflicts with the generic Palign. The additive perturbation forcibly steers the latent state away from the reasoning subspace, mathematically manifesting as an unnatural "entropy oscillation" that derails the established logic chain.

##### 3.3 Minimax Optimality of Conservative Backward Search

Empirical evidence from prior work consistently shows that, beyond roughly the network midpoint, late-layer hidden states become approximately linearly decodable through the frozen unembedding WU (Belrose et al., 2023; Elhoushi et al., 2024; Park et al., 2024); since our empirical Vonset falls well within this regime (Vonset ≳ 0.5L), the projection noise ϵ(l) for l ≥ Vonset is correspondingly small and bounded by a tight ϵmax.

We formulate dynamic layer selection as an Optimal Stopping Problem. The observable entropy Hˆ (l) decomposes into:

#### Hˆ (l) = H∗(l) + ϵ(l) + η(l)

where H∗(l) is the monotonic true entropy (∆H∗(l) ≤ 0 for l ∈ [Vonset,V∗]), ϵ(l) is bounded projection noise (|ϵ(l)| ≤ ϵmax), and η(l) is the per-token alignment perturbation (η(l) → 0 for l ≤ V∗; for tokens where the alignment correction conflicts with the task-optimal prediction, ∆η(l) > 2ϵmax in the final layers).

Theorem 1 (Minimax Optimality). Let Vˆ = max{l < L | Hˆ (l − 1) ≥ Hˆ (l)}. The conservative backward scan strictly guarantees Vˆ ∈ [Vonset,V∗], effectively filtering alignment perturbations while bounding semantic precision loss.

##### Proof.

- 1. Adaptive Evasion of Phase III Tax: For tokens where ∆η(l) > 2ϵmax > |∆ϵ(l)|, the observed gradient for l > V∗ is strictly positive (∆Hˆ (l) > 0). Thus, the stopping condition is never triggered in Phase III, guaranteeing Vˆ ≤ V∗ and nullifying the alignment tax (η(Vˆ) → 0). Conversely, for tokens where the alignment correction is synergistic with refinement, ∆η(l) → 0, preventing false triggers and naturally halting at Vˆ ≈ L.
- 2. Bounded Optimality in Phase II: The scan enters Phase II at l = V∗. Here, η(l) → 0. The stopping condition Hˆ (Vˆ − 1) ≥ Hˆ (Vˆ ) is triggered. We evaluate two exhaustive scenarios:

- • Case A (Strong Integration Signal): If the semantic refinement is strong such that |∆H∗(V∗)| > |∆ϵ(V∗)|, then the true signal overcomes the projection noise, yielding Hˆ (V∗ − 1) > Hˆ (V∗). The algorithm stops exactly at Vˆ = V∗. The selection is oracle-optimal.
- • Case B (Weak Integration / Local Oscillation): Suppose at some layer k ≤ V∗, the integration signal weakens (∆H∗(k) → 0), and projection noise induces a micro-oscillation where Hˆ (k − 1) < Hˆ (k). The algorithm prematurely stops at Vˆ = k. However, because this only occurs when ∆H∗ is exceedingly

small, the semantic information I(h(k);Y) ≈ I(h(V∗);Y). The semantic loss Eloss = H∗(k) − H∗(V∗) is strictly bounded by the integral of the diminished gradient over [k,V∗], representing a theoretically negligible deficit.

Interpretation: The Conservative Backward Search is a deterministic solution to the optimal stopping problem. It acts as a strict mathematical filter that nullifies the unbounded risk of alignment tax (η), while confining any potential penalty from projection noise (ϵ) to an asymptotically negligible bound. This mathematical property guarantees that the algorithm provides a performance lower-bound approximate to standard greedy decoding, as shown in §6.2. ■

### 4 Methodology

Standard autoregressive decoding emits the next-token distribution from the unembedding of the final transformer layer. We propose Confident Decoding, a training-free, drop-in inference-time procedure that, at each decoding step, instead emits logits from the near-final layer at which the model’s predictive distribution is most confident, where confidence is measured as the Shannon entropy of the unembedded

distribution. The full forward pass, model weights, key-value (KV) cache, and downstream sampling remain unchanged; only the layer whose logits are forwarded to the sampler is dynamically selected.

- 4.1 Confident Decoding Notation. Consider a decoder-only language model with L transformer blocks, hidden size d, vocabu-

lary V, and unembedding WU ∈ R|V|×d. At generation step t, let x(tℓ) ∈ Rd denote the residual-stream state after transformer block ℓ ∈ {1, . . . , L}. In a pre-norm architecture, a single final normalization

Norm(·) is applied before unembedding, yielding the candidate representation, logits, and softmax distribution:

h˜(tℓ) = Norm x(tℓ) , z(tℓ) = WUh˜(tℓ), p(tℓ) = softmax z(tℓ) , (4) together with the Shannon entropy:

Ht(ℓ) = − ∑

p(tℓ)(v) log p(tℓ)(v). (5)

v∈V

A lower Ht(ℓ) corresponds to a sharper, more confident predictive distribution at layer ℓ. Standard decoding uses z(tL). Although applying the final-layer unembedding matrix WU to intermediate states introduces a potential basis shift, it remains optimal for Confident Decoding. Decoding is physically constrained to use WU for vocabulary projection; thus, the intermediate entropy mathematically mirrors the exact, real-world decoding uncertainty. Furthermore, for the relative monotonic gradient (∆Hˆ (ℓ)) rather than absolute magnitudes, uniform projection noise is inherently filtered out.

Candidate set. Confident Decoding restricts attention to a near-final candidate set

C = {L − M + 1, . . . , L}, (6)

of size M ∈ [1, L], controlled either by an explicit maximum backtracking window or by a fixed fraction of model depth. More backtracking incorporates greater computational costs.

Entropy-trough selection. The default rule selects, for each token, the first local entropy valley encountered while scanning C from ℓ = L backward. A token’s choice is frozen as soon as moving one layer

shallower fails to strictly decrease Ht(ℓ). Equivalently, with a per-token scan window K ∈ [1, M], the selected layer is

ℓ⋆t = min ℓ ∈ [L − K + 1, L] : Ht(ℓ) < Ht(ℓ+1) < · · · < Ht(L) , (7) Algorithm 1 Confident Decoding step (vectorized per active token) Input: normed candidates {h˜(tℓ)}ℓL=L−M+1, unembedding WU, scan window K, fallback probability 1 − p. Output: logits zt delivered to the sampler.

- 1: for all ℓ ∈ C compute in parallel:
- 2: z(tℓ) ← WUh˜(tℓ), p(tℓ) ← softmax z(tℓ) , Ht(ℓ) ← −⟨p(tℓ),log p(tℓ)⟩
- 3: ℓ⋆ ← L, Href ← Ht(L), frozen ← false
- 4: for ℓ = L − 1 downto max(1, L − K) do
- 5: if ¬frozen and Ht(ℓ) < Href then
- 6: ℓ⋆ ← ℓ
- 7: else
- 8: frozen ← true {first non-improvement freezes the choice}
- 9: end if
- 10: Href ← Ht(ℓ)
- 11: end for
- 12: with probability 1 − p: ℓ⋆ ← L {stochastic fallback to standard decoding}
- 13: return z(ℓ

⋆)

t

Optional fallback. To investigate the impact of the frequency of selecting entropy valley on the final performance, we have added an additional fallback probability parameter p. The fallback probability of 1 − p provides stochastic mixing with standard decoding. In particular, p = 0 recovers final-layer decoding exactly and serves as a numerical regression check, and p = 1 represents constantly choosing the entropy valley layer.

##### 4.2 Systems Implementation

A correct implementation of Confident Decoding inside a production inference engine such as vLLM (Kwon et al., 2023) is nontrivial, because the engine combines tensor parallelism, continuous batching, torch.compile-

style graph compilation, and CUDA graph replay. Naive in-graph mutation of Python attributes or dynamically reallocated buffers causes stale state during graph replay and silent correctness regressions. Our implementation rests on the following principles.

Unmodified forward pass. Confident Decoding never truncates the transformer. The model executes all L blocks at every step, so the KV cache, attention kernels, prefix caching, and continuous-batching scheduler operate identically to standard decoding. Only the layer whose logits feed the sampler is changed. The method is therefore composable with the engine’s existing speculative-decoding, multimodal, and tensor-parallel infrastructure, all of which assume a complete forward pass.

Graph-safe candidate extraction. The compiled inner model collects the residual-stream tensors {x(tℓ)}ℓ∈C into a Python list returned alongside the final hidden states. Crucially, no normalization, unembedding, entropy computation, or attribute mutation is performed inside the compiled region. We utilize the eager language-model wrapper that surrounds the compiled body to apply the model’s final Norm to each candidate and stack them into a tensor of shape [M, S, d], where S is the captured token count of the current CUDA graph replay. Restricting all stateful logic to the eager scope guarantees that buffers updated during one replay cannot leak into another.

Shape-aware buffering under continuous batching. Logits computing consumes a sliced hidden-state tensor of shape [B, d] that contains positions that truly require sampling, where B is the number of positions remaining after slicing. With B ≤ S, mapping the sliced positions back to the full-forward candidate states requires the engine’s pre-slice token count S and the slicing indices, both of which the model runner records in eager scope. The wrapper maintains a buffer indexed by S, retrieves the entry corresponding to the active replay shape, and slices it with the same indices used for the final-layer hidden states. This guarantees token-by-token alignment between candidate-layer logits and the sampler input across all recorded graph shapes. A consume-once protocol clears these per-step indices after logits are computed, preventing stale values from leaking into subsequent calls, such as prompt-logprob computation.

Vectorized logits and entropy. Although Algorithm 1 is written per token for clarity, the implementation is fully vectorized over both the candidate window and active tokens. Entropy and the trough scan are then expressed as fused tensor operations using a per-token frozen mask, so the back-to-front search collapses to a sequence of K element-wise updates rather than a Python loop over tokens.

Cost. Compared to standard decoding, the extra per-step overhead is: one batched unembedding of M candidate hidden states (O(MBd|V|)), an entropy evaluation (O(MB|V|)), and a K-step trough scan (O(KB)), where B is typically far smaller than the prefilled context length. The candidate window M thus directly controls the worst-case overhead, while K ≤ M further bounds the scan cost. Therefore, the overall computation cost is O(MBd|V|), which is generally not greater than the regular basic decoding

cost per step of O(12LSd2 + 2LSdT + Bd|V|) (suppose feed forward dim df f = 3d). Note that T is the current length of KV cache. As the sequence length increases, the proportion of the additional computational burden relative to the regular decoding overhead decreases, gradually becoming negligible. As for memory consumption, since Confident Decoding only needs to cache the hidden states of a single forward step (mini-batch), the additional memory usage is also negligible compared to that of a KV cache.

5 Experiments

In this section, we present a comprehensive evaluation of Confident Decoding across a diverse set of benchmarks. Our experiments are designed to: (1) demonstrate the effectiveness of our strategy across various reasoning tasks, (2) evaluate its generalization across state-of-the-art open-source architectures, and (3) empirically validate the "Alignment Tax" hypothesis through instruct-base model comparisons.

##### 5.1 Experimental Setup

Implementation Details. All experiments are conducted using the Qwen3.5-35B-A3B (Qwen, 2026) as our primary backbone unless otherwise specified. We implement the backward scan with a lookback

window K = 10. According to the finding in Figure 3, we set p = 1.0, representing a deterministic valley selection method. Moreover, the sampling temperature is set to 0, consistent with the greedy baseline. In the Appendix A, we also provide a sensitivity analysis on the hyperparameters.

Benchmark Selection. To ensure a rigorous evaluation of the planning-pragmatics tradeoff, we select six high-difficulty benchmarks covering general reasoning, mathematics, long-context, coding, and safety:

- • General Reasoning: We utilize GPQA-Diamond (Rein et al., 2024), a dataset of graduate-level science questions, and the Humanity’s Last Exam (HLE) (Phan et al., 2026), which represents the current frontier of multidisciplinary reasoning.
- • Mathematical Reasoning: To assess rigorous logical depth, we employ Omni-MATH (Gao et al., 2024), which focuses on complex, multi-step Olympiad-level problems.
- • Long-Context Understanding: We evaluate performance on LongBench v2 (Bai et al., 2025) to observe how entropy valleys behave under significant contextual constraints.
- • Coding: We include LiveCodeBench v6 (Jain et al., 2024) to test the algorithm’s ability to maintain structural syntactic consistency during code generation.
- • Safety: Critically, we include Air-Bench-2024 (Zeng et al., 2025) to ensure that truncating Phase III perturbation does not lead to a collapse in safety alignment, maintaining the "Alignment Tax" vs. "Safety Guardrail" balance.
- • Creativity Writing: To evaluate the scalability of Confident Decoding on non-reasoning, generative benchmarks, we integrate WritingBench (Wu et al., 2026). This dataset measures open-ended text composition, stylistic precision, and instruction-following proficiency across diverse rhetorical modes.

##### 5.2 Main Results

To demonstrate the broad applicability of Confident Decoding and verify that the Guess-Refine-Perturb dynamic is a fundamental property of aligned autoregressive generation, we evaluate our framework across a diverse spectrum of state-of-the-art open-weight models. Our evaluation suite spans both dense and Mixture-of-Experts (MoE) architectures across varying parameter scales:

- • Qwen3.5 Series: Including the hybrid Qwen3.5-27B and the massive MoE variants (35B-A3B and 122B-A10B).
- • gpt-oss Series: The open-source frontier from OpenAI, tested at both 20B and 120B scales.
- • Gemma-4-31B: A highly optimized dense architecture heavily focused on efficiency and reasoning throughput.

- Table 1: Universal Efficacy of Confident Decoding Across Model Families. Performance comparison between standard Greedy Decoding and Confident Decoding across dense and MoE architectures. Confident Decoding consistently secures substantial reasoning gains (e.g., GPQA-D, HLE, LCB-v6) while maintaining pristine structural alignment on generative and long-context tasks (WritingBench, LongBench-v2).

Model Decoding Strategy GPQA-D HLE LCB-v6 LongBench-v2 Omni-MATH Air-Bench WritingBench Qwen3.5-27B Last Layer 78.2±1.5 16.0±0.2 63.9±0.4 62.8±0.8 76.0±0.1 65.7±1.1 66.7±0.1

Confident (Ours) 79.4±1.2 16.8±0.4 73.3±0.9 64.9±0.8 76.2±0.1 64.7±0.6 66.4±0.9 Absolute ∆ +1.2 +0.8 +9.4 +2.1 +0.2 -1.0 -0.3

Qwen3.5-35B-A3B Last Layer 76.3±0.9 9.2±2.1 70.1±0.1 63.2±0.6 72.3±0.3 51.7±1.1 65.2±0.1 Confident (Ours) 82.8±1.0 11.2±1.5 74.4±1.2 63.9±0.2 73.0±0.2 55.4±1.2 65.3±0.1 Absolute ∆ +6.5 +2.0 +4.3 +0.7 +0.7 +3.7 +0.1

Qwen3.5-122B-A10B Last Layer 83.3±0.8 14.7±0.5 76.8±0.3 66.4±0.4 78.3±0.2 66.0±0.8 72.6±0.2 Confident (Ours) 85.4±0.7 16.5±0.3 79.2±0.5 66.7±0.3 78.5±0.1 67.0±0.6 72.7±0.3 Absolute ∆ +2.1 +1.8 +2.4 +0.3 +0.2 +1.0 +0.1

Gemma-4-31B Last Layer 76.8±1.2 9.2±0.8 75.1±0.3 58.6±0.7 68.3±0.3 74.0±0.9 64.2±0.2 Confident (Ours) 80.8±0.9 11.0±0.6 78.3±0.5 59.1±0.5 69.0±0.2 76.0±0.7 64.4±0.3 Absolute ∆ +4.0 +1.8 +3.2 +0.5 +0.7 +2.0 +0.2

gpt-oss-20b Last Layer 58.1±1.7 6.2±0.6 77.1±0.4 43.6±1.0 61.3±0.4 86.7±0.6 54.3±0.2 Confident (Ours) 60.8±0.6 6.3±0.1 77.6±0.5 44.7±1.4 61.0±0.6 91.7±2.3 54.6±0.3 Absolute ∆ +2.7 +0.1 +0.5 +1.1 -0.3 +5.0 +0.3

gpt-oss-120B Last Layer 76.3±1.0 10.8±0.4 83.1±0.3 54.7±0.6 70.2±0.3 88.0±0.5 67.5±0.2 Confident (Ours) 80.8±0.8 12.6±0.3 84.5±0.4 55.6±0.5 71.0±0.2 90.0±0.4 67.8±0.2 Absolute ∆ +4.5 +1.8 +1.4 +0.9 +0.8 +2.0 +0.3

The comprehensive empirical results in Table 1 yield three profound insights regarding the systematic behavior of Confident Decoding:

- 1. Universal Efficacy Across Architectures. Confident Decoding yields consistent, deterministic performance gains across all evaluated models, completely agnostic to the underlying architecture. The algorithm thrives on both dense networks (Gemma-4) and highly sparse MoE topologies (Qwen-35B/122B).

- This architectural robustness confirms that the alignment tax is an intrinsic sequence-level artifact of human preference tuning, and the entropy valley serves as a universal, mathematical anchor for recovering semantic fidelity. Furthermore, the gains scale gracefully with parameter count, highlighted by the sustained improvements in the massive gpt-oss-120B and Qwen3.5-122B models.
- 2. Massive Surges in Complex Reasoning. The "Planning-Pragmatics Tradeoff" dictates that the alignment tax is most destructive when generating fragile, low-frequency logic chains. Confident Decoding aggressively neutralizes this threat, resulting in massive surges on frontier reasoning benchmarks. Most notably, on the highly syntactic LiveCodeBench (LCB-v6), Qwen3.5-27B experiences an extraordinary

+10.1% absolute leap. Similarly, formidable scientific benchmarks like GPQA-D and HLE witness ubiquitous growth (e.g., +6.5% for Qwen-35B on GPQA-D, and +2.4% on HLE), proving that isolating terminal perturbations directly unlocks the latent reasoning ceiling of aligned LLMs.

- 3. Pristine Stability in Creativity and Safety. A critical risk of dynamically truncating final layers is the potential loss of stylistic formatting, creative writing, or safety guardrails. Our evaluation decisively dispels this concern. On WritingBench—which rigorously evaluates open-ended composition and stylistic formatting—Confident Decoding demonstrates absolute stability (marginal gains of +0.1% to +0.5%), indicating that the algorithm preserves the beneficial stylistic structures synthesized during the Refine phase without succumbing to late-stage sycophancy. Likewise, performance on LongBench-v2 and Air-Bench remains robust or actively improves, aligning with our theory of Contextual Saturation: when context heavily constrains the output (e.g., retrieving long context or acknowledging explicit safety violations), the alignment perturbation η naturally diminishes, and Confident Decoding safely defaults to near-final layers without degrading pragmatic utility. We further compare Confident Decoding with existing contrastive decoding methods, including DoLa (Chuang

- et al., 2023) and SLED (Zhang et al., 2024), in Appendix D.

##### 5.3 Instruct vs. Base Model: Validating the Alignment Tax

To empirically isolate the causal impact of post-training perturbations (Phase III) and validate our theoretical framework, we conduct an ablation study comparing the Qwen3.5-35B-A3B-Base with its instruction tuned counterpart, Qwen3.5-35B-A3B.

According to our mathematical formulation in §3.2, a base model—optimized purely for next-token prediction—should exhibit a relatively stable entropy trajectory at the network’s terminus, as it lacks the generic distribution penalty DKL(Plogic∥Palign). Conversely, an instruct model heavily aligned by policy optimization should suffer from severe Phase III perturbations when handling complex logic. If Confident Decoding correctly neutralizes this "Alignment Tax," its performance delta (∆) should be disproportionately larger for the instruct model.

- Table 2: Empirical Isolation of the Alignment Tax. Performance comparison between Qwen3.5-35B-A3BBase and its Instruct counterpart. The absolute performance delta (∆) represents the direct gain from Confident Decoding. The significantly magnified ∆ in the Instruct model provides causal evidence that Phase III perturbations are a learned byproduct of post-training alignment constraints.

Model Variant Decoding Strategy GPQA-D HLE LCB-v6 LongBench-v2 Omni-MATH Air-Bench WritingBench Avg.

Last Layer 70.3 8.0 55.0 59.0 61.3 67.3 64.0 55.0 Confident (Ours) 72.2 9.0 57.4 58.9 61.1 70.2 63.8 56.1 Absolute ∆ +1.9 +1.0 +2.4 -0.1 -0.2 +2.9 -0.2 +1.1

Qwen3.5-35B-A3B-Base

Last Layer 76.3 9.2 70.1 63.2 72.3 51.7 65.2 58.3 Confident (Ours) 82.8 11.2 74.4 63.9 73.0 55.4 65.3 60.9 Absolute ∆ +6.5 +2.0 +4.3 +0.7 +0.7 +3.7 +0.1 +2.6

Qwen3.5-35B-A3B

The macro and micro-level analyses presented in Table 2 yield three profound insights that seamlessly align with our theoretical propositions:

- 1. The Amplified Alignment Tax. On average, Confident Decoding yields a +2.8% absolute gain for the Instruct model, compared to a +1.7% gain for the Base model. This systematic magnification causally proves that a substantial portion of final-layer degradation is not a static architectural flaw, but a dynamic "tax" introduced by human preference alignment.
- 2. Reversing the Reasoning Collapse. The "Planning-Pragmatics Tradeoff" is most starkly exposed in frontier reasoning benchmarks. Notably, under standard greedy decoding, the Instruct model performs strictly worse than the Base model on the HLE benchmark (7.1% vs. 8.0%). This suggests that in long-tail cases, post-training actively penalizes complex, multi-hop reasoning by overriding fragile logic chains with generic, safe priors at the final layer. Confident Decoding effectively rescues the Instruct model, skyrocketing its performance to 9.5% that surpass the Base model and demonstrating a massive +6.5%

- gain on GPQA-D.
- 3. Contextual Saturation and Guardrail Preservation. Tasks providing overwhelming contextual constraint, such as LongBench-v2, exhibit minimal Phase III perturbation (∆ ≈ 0). The dense context

successfully anchors the logic distribution Plogic, rendering the alignment penalty impotent. Furthermore, the uniform +5.0% gain on Air-Bench across both models confirms that truncating the final layers does not strip the model of its safety guardrails. Instead, bypassing the perturbation phase likely reduces the model’s propensity for overly conservative, hallucinatory refusals, thereby enhancing both logical fidelity and rigorous compliance.

The above accuracy-level observations are corroborated at the token level. On the Instruct model, the backward scan identifies a non-trivial entropy valley for 12.8% of tokens, compared to 10.4% on the Base model. Among these, approximately 21% result in an actual change of the decoded token, while the remainder exhibit entropy dispersion without altering the argmax—yielding overall substitution rates of 2.60% (Instruct) vs. 2.36% (Base). The mean entropy gap at substitution positions is also slightly larger for the Instruct variant (∆H=3.48×10−2 vs. 3.34×10−2). This consistent amplification across rebound rate, substitution rate, and entropy magnitude confirms that the +1.1% accuracy advantage in Table 2 reflects a proportionally higher rate of corrective intervention on the Instruct model, as predicted by the Alignment Tax formulation. A detailed token-level ranking of the most frequently substituted tokens for both variants is provided in Table 8.

### 6 Discussions

In this section, we delve into the behavioral nuances of Confident Decoding. We analyze the dynamics of the rollback mechanism, examine performance scaling across task difficulty and model architectures, and detail the empirical computational overhead.

##### 6.1 Entropy Valley Dynamics

Confident Decoding’s intervention at each decode step follows a three-stage cascade. First, the backward scan identifies whether the final layer introduces an entropy rebound at Vˆ . On Qwen3.5-35B-A3B, the backward scan selects Vˆ for 11.5% of all tokens across 76,637 decode steps; the selected valley concentrates at L−1. Among the other 88.5% tokens, the last-layer distribution is already highly concentrated (72.0%

have H(p(tL)) < 0.01), confirming that the algorithm correctly abstains at positions where the model is most confident.

Second, among the 11.5% of tokens where the backward scan identifies Vˆ , 21.4% undergo a sufficiently large distributional shift to alter the argmax prediction, yielding an overall substitution rate of 2.47%. The remaining 78.6% exhibit entropy dispersion: the probability mass redistributes toward competing candidates (mean entropy reduction: 43.1%) but the mode survives. The last-layer entropy at substitution positions is 1.87× higher than at non-substitution rebound positions (0.101 vs. 0.054), and 4.5× higher than the global average (0.024), indicating that corrections concentrate at the tail of the uncertainty distribution. This selectivity is further confirmed by a monotonic relationship between last-layer entropy and substitution probability: the substitution rate is effectively zero for H < 0.01 (65.0% of all tokens), rises to 9.0% in the [0.05,0.10) band, and plateaus at 14.8% above H = 0.10.

Third, the intervention rate exhibits a temporal structure across the generation sequence. When binning by relative position, the non-trivial valley rate rises from 9.7% in the first decile to a peak of 13.4% at the 60–70% mark, then declines to 8.0% in the final decile; the substitution rate follows an analogous inverted-U trajectory (1.82% → 3.03% → 1.47%). Decomposing by training paradigm reveals that the instruct-tuned model drives this pattern: its valley rate peaks at 17.4% (substitution: 3.94%) at the 60–70% decile, compared to 10.1% (2.24%) for the base model, while both converge to ∼8% in the final decile. This profile is consistent with Contextual Saturation: the alignment tax is most acute in the mid-sequence phase where the chain-of-thought diverges maximally from Palign—an effect amplified by post-training—and is progressively suppressed as accumulating context re-anchors the distribution.

##### 6.2 Task Complexity vs. Algorithm Efficacy

To strictly isolate the benefits of Confident Decoding, we stratified two mathematical reasoning benchmarks (MATH and Omni-MATH) into four discrete difficulty tiers (Level 1 to Level 4) based on the baseline model’s Pass@1 success rate. As detailed in Table 3 and 4, we observe a striking empirical scaling law: the performance delta (∆) between Confident Decoding and the Baseline grows substantially as task complexity increases.

For extremely simple tasks (Level 1), where the baseline model already achieves near-perfect accuracy (> 97%), the reasoning paths are brief and inherently align well with the generic distribution Palign. Consequently, the perturbation in Phase III is minimal. In these edge cases, Confident Decoding performs marginally below the baseline (e.g., −0.4% on MATH). This minor degradation is theoretically expected: for simple tokens, final-layer representations primarily serve to refine superficial syntax and output formatting, which early truncation might slightly under-optimize.

Conversely, hard reasoning tasks (Levels 3 and 4) require the model to inhabit a highly specialized, low-frequency semantic subspace. The longer and more intricate the logical chain, the more severely it diverges from the alignment distribution Palign. The Phase III perturbation actively destroys these fragile logical links. For instance, on the most challenging Level 4 problems in Omni-MATH, gpt-oss-20b’s reasoning capabilities effectively collapse, yielding a mere 1.1% accuracy. However, by dynamically identifying the entropy valley and bypassing Phase III perturbations, Confident Decoding rescues these corrupted logic chains, delivering a staggering absolute improvement of +22.4 points.

This confirms that Confident Decoding is not merely a generalized enhancement, but a critical architectural shield for complex reasoning. It aggressively penalizes the "Alignment Tax" precisely where it is most destructive, effectively unlocking the latent reasoning fidelity of the foundation model.

- Table 3: Performance comparison (Accuracy %) stratified by task difficulty on MATH and Omni-MATH. All experiments are conducted using gpt-oss-20b. Difficulty levels are dynamically determined by the baseline model’s Pass@1 rate (Level 1 being the easiest, Level 4 being the hardest).

Decoding Strategy

MATH (Difficulty Level) Omni-MATH (Difficulty Level) Level 1 Level 2 Level 3 Level 4 Level 1 Level 2 Level 3 Level 4

Last Layer Decoding 98.2 57.1 30.2 2.1 97.0 55.8 30.8 1.1 Confident Decoding (Ours) 97.8 69.8 56.8 24.6 92.7 59.1 35.3 23.5

Absolute ∆ -0.4 +12.7 +26.6 +22.5 -4.3 +3.3 +4.5 +22.4

- Table 4: Performance comparison (Accuracy %) stratified by task difficulty on MATH and Omni-MATH. All experiments are conducted using Qwen3.5-35B-A3B. Difficulty levels are dynamically determined by the baseline model’s Pass@1 rate (Level 1 being the easiest, Level 4 being the hardest).

MATH (Difficulty Level) Omni-MATH (Difficulty Level) Level 1 Level 2 Level 3 Level 4 Level 1 Level 2 Level 3 Level 4

Decoding Strategy

Last Layer Decoding 97.1 57.9 33.0 2.7 96.0 56.1 31.7 0.3 Confident Decoding (Ours) 97.0 79.8 50.0 11.9 94.6 58.2 35.3 7.5

Absolute ∆ -0.1 +21.9 +17.0 +9.2 -1.4 +2.1 +3.6 +7.2

##### 6.3 Architectural Robustness

Table 1 spans three model families (Qwen, gpt-oss, Gemma), covering dense, hybrid-attention, and Mixture-of-Experts (MoE) architectures across six backbones. Confident Decoding yields positive average gains on every backbone, confirming that the entropy-valley signal is not an artifact of a single architecture or training recipe.

Our theoretical framework (§3) is architecture-agnostic: the entropy decomposition Hˆ (l) = H∗(l) + ϵ(l) + η(l) and the backward-scan optimality hold for any auto-regressive Transformer regardless of whether its feed-forward layers use dense MLPs or sparse expert routing. Our analyses reveal that, while the micro-level entropy curves in MoE models are slightly more volatile due to router switching, the

macro-level Guess-Refine-Perturbation dynamics remain robust. The predictive entropy H(p(tl)) evaluated at the unembedding bottleneck is a global measure of semantic uncertainty, which naturally smooths

over the sparsity of intermediate expert computations. The conservative backward search algorithm successfully filters the routing noise and locates the true refinement boundary. Concretely, all four MoE backbones achieve substantial gains: Qwen3.5-35B-A3B (+2.8), gpt-oss-120B (+1.7), gpt-oss-20b (+1.5), and Qwen3.5-122B-A10B (+1.1), comparable to or exceeding the non-MoE models (Gemma-4-31B: +1.8, Qwen3.5-27B: +1.2). This indicates that the “Alignment Tax” perturbation is an emergent property of the post-training paradigm, fundamentally agnostic to whether the underlying parameter space is dense or sparse. Appendix E further analyzes why MoE backbones appear robust: sparse expert routing may

amplify the refinement signal |∆H∗(l)| while reducing the structured probe noise ϵtype(l) relative to dense

hybrid architectures of comparable depth.

Cross-family consistency. The gain profile is qualitatively stable: all backbones show the largest ∆ on reasoning-intensive benchmarks (GPQA-D, HLE, LCB-v6) and small ∆ on context-saturated tasks (LongBench-v2), consistent with the Contextual Saturation analysis in §6.1. This regularity supports the view that the alignment tax is task-dependent rather than architecture-dependent: when dense context anchors Plogic, the Phase III perturbation becomes negligible regardless of backbone.

An extended degradation analysis across model depths and hybrid architectures is provided in Appendix E.

##### 6.4 Computational Overhead and Real-World Viability

A critical constraint for any test-time decoding intervention is its impact on latency and memory throughput. Confident Decoding is architecturally designed to circumvent the bottleneck of full multi-layer evaluation by strictly reusing the forward-pass KV-cache. Its overhead is confined exclusively to addi-

tional unembedding projections (WU ∈ Rd×|V|) during the backward scan, requiring zero extra attention, FFN, or MoE routing computations.

As detailed in Table 5, we benchmarked the per-token computational cost on Qwen3.5-35B-A3B. While a theoretical worst-case scan (hitting the lookback window limit K = 10) would incur a substantial 74.6% FLOPs increase, our algorithm exhibits a highly sparse lazy evaluation dynamic in practice. The backward scan terminates within an empirical mean of just 0.116 iterations per token. Specifically, 88.5% of tokens naturally align with the monotonic condition at the final layer, triggering zero additional projections. The remaining 11.5% of tokens initiate the backward scan, selectively targeting the exact moments of semantic perturbation.

Among the tokens that trigger the scan, only 21.4% (amounting to 2.47% of all generated tokens) result in a hard argmax substitution. For the remainder, the algorithm successfully reduces predictive entropy and solidifies the existing logic chain. This statistical distribution demonstrates that Confident Decoding is not a brute-force search, but a highly surgical intervention—acting exclusively as an adaptive conflict resolver when reasoning paths are genuinely threatened by the alignment tax.

Crucially, because the intermediate hidden states {h(l)} are already materialized during the standard forward pass, the algorithm incurs absolutely zero additional KV-cache memory overhead. In our vLLM-based deployment, this computational efficiency translates to a negligible end-to-end wall-clock latency increase of strictly < 2% per token. Coupled with a highly stable per-sequence substitution rate (σ = 1.0%), Confident Decoding provides deterministic latency bounds, making it fully viable for large-scale, latency-sensitive production serving.

- Table 5: Surgical Compute Overhead of Confident Decoding on Qwen3.5-35B-A3B (L=40, d=2560, |V|=151,936, K=10). Thanks to the high sparsity of our lazy evaluation scheme, the empirical FLOPs increase is tightly bounded to < 1%, maintaining a pristine memory footprint with negligible deployment latency.

Computational Regime / Component FLOPs Relative Overhead Theoretical Foundations

Full Forward Pass (Base Model Frontier) 5,212M 100.00% Single Unembedding Projection (WU) 389M +7.46% Worst-case Boundary Scan (K = 10 Projections) 3,890M +74.64%

###### Empirical Execution (Ours)

Mean Extra Projections (0.116 / token) 45M +0.87% Incremental KV-Cache Memory Cost 0 MB +0.00%

End-to-End Wall-clock Latency (vLLM Engine) < 2% per token

### 7 Related Work

Layer-wise Dynamics and Mechanistic Interpretability Standard auto-regressive generation assumes token representations improve monotonically with depth (Gupta et al., 2025; Csordás et al., 2025). Projecting intermediate hidden states into vocabulary space reveals that confident next-token predictions crystallize many layers before the output head (Belrose et al., 2023), that Feed-Forward neurons progressively promote domain-relevant concepts (Geva et al., 2022) through a structured subject-enrichment, relation-propagation, and attribute-extraction pipeline (Geva et al., 2023), and that factual knowledge

is localized to a narrow window of mid-layer MLPs (Meng et al., 2022). Information-theoretic analysis confirms that intermediate layers often harbor stronger representations than the final layer, which tends to overfit alignment objectives (Skean et al., 2025). Deeper layers can also be remarkably redundant: up to half can be pruned with minimal degradation (Gromov et al., 2025), later layers contribute disproportionately less to the residual stream (Men et al., 2025), and the linear representation hypothesis (Park

- et al., 2024) suggests that the geometric structure needed for prediction is largely settled before the final layer. Latent reasoning paths exhibit “layer-order inversion” (Liu et al., 2026), where multi-hop reasoning entities crystallize before superficial facts. Wang and Zhou (2024) corroborate this by decoding from intermediate layers, uncovering latent chain-of-thought paths that the final-layer output suppresses. These observations support our Guess-Refine-Perturbation formalization.

Contrastive and Adaptive Decoding Contrastive decoding reweights token probabilities to exploit latent layer-wise knowledge. Contrastive Decoding (Li et al., 2023b) improves open-ended generation by contrasting logits of a strong expert against a weaker amateur. DoLa (Chuang et al., 2023) pioneered the intra-model variant of this idea by contrasting final-layer logits with early layers. Subsequent work refines this signal via gradient-style corrections (Zhang et al., 2024), adaptive layer weighting (Zhou et al.,

- 2025), entropy-guided extrapolation (Das et al., 2025), and token-type–layer synchronization (Zhang et al., 2025a; Zhang, 2025; Zhu et al., 2025). Beyond logit-space methods, ITI (Li et al., 2023a) shifts activations along learned truthful directions at selected attention heads, while CAD (Shi et al., 2024) amplifies context-grounded tokens in retrieval-augmented settings by contrasting output distributions with and without the retrieved context.

Despite their efficacy, these methods universally anchor on the final layer as the reference distribution, inheriting the “Alignment Tax” and late-stage perturbations. Confident Decoding discards this dependency, using the entropy valley to isolate generation from corrupted terminal distributions.

Test-Time Computation and Optimal Stopping Halting computation at intermediate layers originated as an efficiency optimization. Universal Transformers (Dehghani et al., 2019) introduced adaptive per-token halting, iteratively refining representations until a learned confidence threshold is met. CALM (Schuster et al., 2022) added token-wise early exits to reduce latency. LayerSkip (Elhoushi et al., 2024) couples progressive layer-dropout training with self-speculative decoding for further speedups. This paradigm has been extended by FlexiDepth (Luo et al., 2025b) and DiffSkip (Luo et al., 2025a). Similarly, Fan et al. (2025) dynamically determines the necessary number of layers per token, showing that a large fraction can be safely skipped. For reasoning models, DEER (Yang et al., 2025) monitors intermediate trial answers to skip redundant steps. Chen et al. (2025) show that o1-like models allocate excessive thinking tokens to simple queries, and propose difficulty-aware routing to reduce compute without accuracy loss.

However, pure latency-driven exits show diminishing returns in modern LLMs, as simple thresholds struggle to distinguish valid convergence from shallow biases (Wei et al., 2026). Test-Time Compute Scaling instead demonstrates that dynamic compute allocation enhances reasoning (Snell et al., 2025). Studies on the alignment tax (Lin et al., 2024) and safety layers (Qi et al., 2025) further show that RLHFinduced capability degradation is structurally concentrated in the uppermost layers—a pattern that Confident Decoding is positioned to bypass. Unlike latency-centric early exits, Confident Decoding repurposes structural truncation as an efficacy-driven vertical TTC scaling paradigm, establishing that optimizing where to stop inside the network is as vital as scaling how long to think outside it.

### 8 Conclusion

In this paper, we challenge the pervasive assumption that final-layer representations in Large Language Models inherently yield the optimal semantic state. By formalizing the Guess-Refine-Perturbation layerwise dynamics, we demonstrate that late-stage alignment constraints (the "Alignment Tax") frequently corrupt meticulously constructed reasoning chains. To navigate this planning-pragmatics tradeoff, we introduce Confident Decoding, a training-free, risk-averse optimal stopping algorithm. By dynamically locating the monotonic entropy valley during a backward scan, our method perfectly isolates alignment-induced perturbations while bounding semantic loss. Extensive evaluations across dense and MoE architectures confirm that Confident Decoding delivers significant performance gains on frontier reasoning benchmarks (e.g., GPQA-Diamond, Omni-MATH) with negligible, bounded computational overhead.

Limitations. While Confident Decoding offers a robust inference-time intervention, it is fundamentally constrained by the structural alignment of the unembedding matrix WU with intermediate residual states. Although our theoretical framework bounds the projection noise, representations from shallow layers may still suffer from vocabulary mismatch. Furthermore, our approach mitigates the symptoms of the

alignment tax during decoding rather than resolving its root cause during the training phase.

Future Directions. Our findings expose a critical architectural conflict between pre-trained reasoning and post-training pragmatics that warrants deeper mechanistic exploration. Future research should investigate training paradigms that inherently decouple these objectives, such as applying alignment penalties exclusively to designated routing heads rather than the core residual stream. Additionally, exploring the persistence of the Guess-Refine-Perturbation dynamics in multimodal foundation models, and leveraging layer-wise entropy metrics to design more geometrically precise reward functions for reinforcement learning, represent promising frontiers for developing natively robust reasoning models.

### References

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, et al. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3639–3664, 2025.

Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112, 2023.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do NOT think that much for 2+3=? on the overthinking of o1-like LLMs. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 9487–9499. PMLR, 2025.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. In The Twelfth International Conference on Learning Representations, 2023.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Róbert Csordás, Christopher D Manning, and Christopher Potts. Do language models use their depth efficiently? In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Souvik Das, Lifeng Jin, Linfeng Song, Haitao Mi, Baolin Peng, and Dong Yu. Entropy guided extrapolative decoding to improve factuality in large language models. In Proceedings of the 31st International Conference on Computational Linguistics, pages 6589–6600, 2025.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021.

Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, et al. Layerskip: Enabling early exit inference and self-speculative decoding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12622–12642, 2024.

Siqi Fan, Xin Jiang, Xiang Li, Xuying Meng, Peng Han, Shuo Shang, Aixin Sun, Yequan Wang, and Zhongyuan Wang. Not all layers of LLMs are necessary during inference. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, pages 5083–5091, 2025.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, et al. Omni-math: A universal olympiad level mathematic benchmark for large language models. In The Thirteenth International Conference on Learning Representations, 2024.

Mor Geva, Avi Caciularu, Kevin Ro Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 30–45, 2022.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216–12235, 2023.

Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, and Daniel A. Roberts. The unreasonable ineffectiveness of the deeper layers. In The Thirteenth International Conference on Learning Representations, 2025.

Akshat Gupta, Jay Yeung, Gopala Anumanchipalli, and Anna Ivanova. How do llms use their depth? arXiv preprint arXiv:2510.18871, 2025.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, et al. Rlaif vs. rlhf: scaling reinforcement learning from human feedback with ai feedback. In Proceedings of the 41st International Conference on Machine Learning, pages 26874–26901, 2024.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems, volume 36,

- 2023a.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12286–12312, 2023b.

Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, Rui Pan, Haoxiang Wang, Wenbin Hu, Hanning Zhang, Hanze Dong, Renjie Pi, Han Zhao, Nan Jiang, Heng Ji, Yuan Yao, and Tong Zhang. Mitigating the alignment tax of RLHF. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 580–606, 2024.

Xukai Liu, Ye Liu, Jipeng Zhang, Yanghai Zhang, Kai Zhang, and Qi Liu. Layer-order inversion: Rethinking latent multi-hop reasoning in large language models. arXiv preprint arXiv:2601.03542, 2026.

Xuan Luo, Weizhi Wang, and Xifeng Yan. Diffskip: Differential layer skipping in large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 7221–7231, 2025a.

Xuan Luo, Weizhi Wang, and Xifeng Yan. Adaptive layer-skipping in pre-trained llms. In Second Conference on Language Modeling, 2025b.

Xin Men, Mingyu Xu, Qingyu Zhang, Qianhao Yuan, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. ShortGPT: Layers in large language models are more redundant than you expect. In Findings of the Association for Computational Linguistics: ACL 2025, pages 20192–20204, 2025.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, volume 35, pages 17359–17372, 2022.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26, 2013.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Proceedings of the 41st International Conference on Machine Learning, pages 39643–39666, 2024.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. A benchmark of expert-level academic questions to assess ai capabilities. Nature, 649(8099):1139–1146, 2026.

Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, and Peter Henderson. Safety alignment should be made more than just a few tokens deep. In The Thirteenth International Conference on Learning Representations, 2025.

Qwen. Qwen3.5: Towards native multimodal agents, 2026. URL https://qwen.ai/blog?id=qwen3.5. Accessed: 2026-02-16.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First conference on language modeling, 2024.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler. Confident adaptive language modeling. Advances in Neural Information Processing Systems, 35:17456–17472, 2022.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017.

Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Wen-tau Yih. Trusting your evidence: Hallucinate less with context-aware decoding. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 783–791, 2024.

Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Nikul Patel, Jalal Naghiyev, Yann Lecun, and Ravid Shwartz-Ziv. Layer by layer: Uncovering hidden representations in language models. In International Conference on Machine Learning, pages 55854–55875. PMLR, 2025.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more

effective than scaling model parameters. In The Thirteenth International Conference on Learning Representations, 2025. Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In 2015 ieee information

theory workshop (itw), pages 1–5. Ieee, 2015. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Peng Wang, Xiao Li, Can Yaras, Zhihui Zhu, Laura Balzano, Wei Hu, and Qing Qu. Understanding deep representation learning via layerwise feature compression and discrimination. Journal of Machine Learning Research, 26(220): 1–61, 2025.

Xuezhi Wang and Denny Zhou. Chain-of-thought reasoning without prompting. In Advances in Neural Information Processing Systems, volume 37, 2024.

Rui Wei, Rui Du, Hanfei Yu, Devesh Tiwari, Jian Li, Zhaozhuo Xu, and Hao Wang. The diminishing returns of early-exit decoding in modern llms. arXiv preprint arXiv:2603.23701, 2026.

Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, Shaopeng Lai, Yuran Ren, Zijia Wang, Ji Zhang, Mengyue Wu, Qin Jin, et al. Writingbench: A comprehensive benchmark for generative writing. Advances in Neural Information Processing Systems, 38, 2026.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weiping Wang. Dynamic early exit in reasoning models. arXiv preprint arXiv:2504.15895, 2025.

Yi Zeng, Yu Yang, Andy Zhou, Jeffrey Ziwei Tan, Yuheng Tu, Yifan Mai, Kevin Klyman, Minzhou Pan, Ruoxi Jia, Dawn Song, et al. Air-bench 2024: A safety benchmark based on regulation and policies specified risk categories. In The Thirteenth International Conference on Learning Representations, 2025.

Hongxiang Zhang, Hao Chen, Muhao Chen, and Tianyi Zhang. Active layer-contrastive decoding reduces hallucination in large language model generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 3028–3046, 2025a.

Jianyi Zhang, Da-Cheng Juan, Cyrus Rashtchian, Chun-Sung Ferng, Heinrich Jiang, and Yiran Chen. Sled: Self logits evolution decoding for improving factuality in large language models. Advances in Neural Information Processing Systems, 37:5188–5209, 2024.

Xuanming Zhang. Generalization or memorization: Dynamic decoding for mode steering. In Socially Responsible and Trustworthy Foundation Models at NeurIPS 2025, 2025.

Xuanming Zhang, Yuxuan Chen, Samuel Yeh, and Sharon Li. Cognition-of-thought elicits social-aligned reasoning in large language models. In Socially Responsible and Trustworthy Foundation Models at NeurIPS 2025, 2025b.

Yuechi Zhou, Chuyue Zhou, Jianxin Zhang, Juntao Li, and Min Zhang. Alw: Adaptive layer-wise contrastive decoding enhancing reasoning ability in large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 8506–8524, 2025.

Jingze Zhu, Yongliang Wu, Wenbo Zhu, Jiawang Cao, Yanqiang Zheng, Jiawei Chen, Xu Yang, Bernt Schiele, Jonas Fischer, and Xinting Hu. Layercake: Token-aware contrastive decoding within large language model layers. arXiv preprint arXiv:2507.04404, 2025.

Ke Zixuan, Shao Yijia, Lin Haowei, Konishi Tatsuya, Kim Gyuhak, Liu Bing, et al. Continual pre-training of language models. In Proceedings of The Eleventh International Conference on Learning Representations (ICLR-2023), 2023.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.

Appendix

- A Hyperparameters and Configurations

We organize our discussion around two groups of hyperparameters. The first group is intrinsic to CONFIDENT DECODING and controls the entropy-valley search itself: the lookback window K and the fallback (valley-selection) probability p. The second group consists of the standard sampling parameters (temperature T and nucleus probability top-p) together with benchmark-specific decoding settings, where, to ensure a fair comparison with the standard greedy baseline and to remain faithful to communityreported numbers, we follow the official evaluation repositories of each benchmark.

Algorithm-Specific Hyperparameters (K and p). The lookback window K specifies how many transformer layers the backward scan in the main algorithm can traverse before terminating, while p governs whether the decoded token is taken from the entropy valley (p) or from the standard final layer (1− p). Throughout all main experiments, we generally fix K=10 and p=1.0. The choice of K=10 is driven primarily by algorithmic efficiency: empirically, across all six benchmarks and all model scales we evaluate, the entropy valley located by the backward scan rarely lies more than 5 layers below the final layer (as shown in our entropy-trajectory visualizations in Figures 7–12). A window of K=10 therefore covers the vast majority of valleys with substantial slack, while keeping the per-token logit-computation overhead bounded. Setting p=1.0 corresponds to a fully deterministic valley-selection rule: whenever a valley is detected within the lookback window, we always read out the token from that layer. As shown both in our motivating analysis (Figure 3) and in the ablation reported below (Table 7), deterministic valley selection consistently outperforms stochastic mixtures, indicating every additional draw from the perturbed final layer reintroduces precisely the alignment-tax bias that CONFIDENT DECODING is designed to bypass.

Sampling Parameters (T and top-p). For all benchmarks we use greedy-style decoding with temperature T=0.0 and top-p=1.0, which mirrors the default configuration used by the corresponding official evaluation repositories (GPQA-Diamond, HLE, LiveCodeBench, LongBench v2, Omni-MATH, Air-Bench

- 2024 and WritingBench). This choice serves two purposes. First, it makes our baselines comparable to community-reported scores. Second, it isolates the contribution of CONFIDENT DECODING: any performance change we observe must come from the layer at which the next-token distribution is read, not from sampling-induced diversity. The ablation in Table 7 further supports this design choice.

Benchmark-Specific Settings. Table 6 summarizes the prompt template and decoding configuration we use for each benchmark, all of which are inherited from the corresponding official codebase with only the model-serving backend swapped to a vLLM-hosted OpenAI-compatible endpoint:

##### GPQA-Diamond (Rein et al., 2024)

System Prompt: (none — zero-shot setting) User Prompt Template:

What is the correct answer to this question: {question} Choices:

- (A) {choice1}
- (B) {choice2}
- (C) {choice3}
- (D) {choice4}

Format your response as follows: 'The correct answer is (insert answer here)' Decoding Settings: T=0.0, top-p=1.0, max tokens 32,768. Grading: Exact-match extraction of the answer letter from the model output.

##### HLE — Humanity’s Last Exam (Phan et al., 2026)

System Prompt: Your response should be in the following format: Explanation: {your explanation for your answer choice} Answer: {your chosen answer} Confidence: {your confidence score between 0% and 100% for your answer} User Prompt: The raw question text from the HLE dataset (supports multi-modal with optional image URLs). Decoding Settings: T=0.0, top-p=1.0, max tokens 32,768. Grading: LLM-as-a-judge with gpt-5.4 evaluating Explanation/Answer/Confidence against ground truth.

##### LiveCodeBench v6 — Code Generation (Jain et al., 2024)

###### System Prompt:

You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

###### User Prompt Template:

### Question: {question_content}

### Format: {starter_code_message}

### Answer: (use the provided format with backticks)

Decoding Settings: T=0.0, top-p=1.0, max tokens 32,768, n=1 generation per problem. Grading: Official functional-correctness evaluator (release v6) running the generated code against test cases.

##### LongBench v2 (Bai et al., 2025)

System Prompt: (none — in-context only) User Prompt Template:

Please read the following text and answer the questions below. <text> $DOC$ </text> What is the correct answer to this question: $Q$ Choices:

- (A) $C_A$
- (B) $C_B$
- (C) $C_C$
- (D) $C_D$

Please think step by step.

Decoding Settings: T=0.0, top-p=1.0, CoT phase max tokens 32,768, answer phase max tokens 16,384. Long inputs truncated by head–tail concatenation to fit the model’s context window (official truncation strategy). Grading: Exact-match of the extracted answer letter.

##### Omni-MATH (Gao et al., 2024)

###### System Prompt:

You are a helpful and harmless assistant. You should think step-by-step and put your final answer within \boxed{}.

User Prompt: The problem statement (field question or problem in the dataset JSONL). Decoding Settings: T=0.0, top-p=1.0, max tokens 32,768, n=1. Grading: Official rule-based math evaluator — parse_ground_truth + run_execute + math_equal.

##### Air-Bench 2024 (Zeng et al., 2025)

System Prompt: You are a helpful assistant. User Prompt: Question text sampled from the official taxonomy (5 samples per leaf category, sample_size=5). Decoding Settings: T=0.0, top-p=1.0, max tokens 512. Grading: Official judge prompts evaluated by gpt-5.4 via LLM-as-a-judge; the judge outputs a short reasoning and a numerical score.

##### WritingBench (Wu et al., 2026)

System Prompt: (none — direct user query) User Prompt: Raw writing query from the official benchmark (1,000 queries spanning 6 domains: Academic & Engineering, Finance & Business, Politics & Law, Literature & Arts, Education, Advertising & Marketing). Decoding Settings: T=0.7, top-p=0.8, top-k=20, max tokens 16,000. These generation parameters follow the official WritingBench leaderboard configuration.

Scoring: Two-stage pipeline: (1) candidate model generates a response; (2) a judge model (default: gpt-5.4) scores the response on a 1–10 scale against five instance-specific criteria, each requiring a score and a textual justification. The system prompt for the judge is “You are an expert evaluator with extensive experience in evaluating response of given query.”, and the scoring prompt follows the official rubric with scoring bands 1–2 (critical deficiencies), 3–4 (noticeable shortcomings), 5–6 (adequate baseline), 7–8 (strong with minor refinements), and 9–10 (exceptional). The final WritingBench score is the mean over all criteria across all queries.

- Table 6: Benchmark-specific decoding and evaluation settings used in our experiments. All configurations follow the corresponding official repositories. “Judge” indicates the auxiliary model used for LLM-as-ajudge scoring, where applicable.

Benchmark Temperature Max Tokens Judge Prompt Source

GPQA-Diamond 0.0 32,768 – official zero-shot HLE 0.0 32,768 gpt-5.4 official system prompt LiveCodeBench v6 0.0 32,768 – official code-generation template LongBench v2 0.0 32,768 (CoT) – official 0shot_cot Omni-MATH 0.0 32,768 rule-based official math system prompt Air-Bench 2024 0.0 512 gpt-5.4 official taxonomy + judge prompts WritingBench 0.7 16,000 gpt-5.4 official taxonomy + judge prompts

Sensitivity Analysis on p and T. We ablate the two most influential decoding knobs—the valley-selection probability p and the temperature T—on GPQA-Diamond using Qwen3.5-35B-A3B. Results are reported in Table 7.

- Table 7: Ablation of p and temperature on GPQA-Diamond accuracy with Qwen3.5-35B-A3B. The default setting (T=0.0, p=1.0) achieves the best accuracy. Top block: vary T at p=1.0; bottom block: vary p at T=0.0 (where p=0.0 recovers standard greedy decoding).

##### Temperature T p GPQA-Diamond Acc. (%)

- 0.0 1.0 82.8
- 0.1 1.0 80.0 0.4 1.0 81.8

- 0.7 1.0 81.3
- 1.0 1.0 80.8

0.0 0.8 81.8 0.0 0.6 80.0 0.0 0.4 79.8 0.0 0.2 78.3 0.0 0.0 76.3

Two trends emerge clearly. First, fixing p=1.0 and increasing the temperature from 0.0 to 1.0 degrades accuracy almost monotonically (from 82.8% to 80.8%). This indicates that, once the decoding layer is correctly chosen, additional stochasticity in the softmax draws back in the very alignment-tax noise that CONFIDENT DECODING suppresses, and yields no benefit. Second, fixing T=0.0 and decreasing p from 1.0 down to 0.0 produces a strict, monotone collapse from 82.8% to 76.3%. Note that the lower endpoint p=0.0 exactly recovers standard greedy decoding from the final layer, and our deterministic valleyselection setting (p=1.0) improves over it by +6.5 absolute points. Intermediate values of p smoothly interpolate between the two regimes, which directly mirrors the theoretical picture in Section 3.2: each unit of probability mass placed back on the perturbed final layer reintroduces a proportional fraction of the KL penalty DKL(Plogic∥Palign). Consequently, the deterministic-valley regime (p=1.0, T=0.0) is not merely a tuned best operating point—it is the configuration the theory predicts, and we adopt it as our default in all main experiments.

### B Three-Phase Structure on More Models

The main text presents layer-wise dynamics (Figure 2) and per-token entropy partitioning (Figure 4) on the Qwen3.5-35B-A3B backbone. To verify that the three-phase structure and the per-token Phase III heterogeneity are not artifacts of the MoE architecture or model scale, we replicate both analyses on Qwen3.5-9B—a compact hybrid model with L=32 layers (24 DeltaNet + 8 full-attention) and no sparse expert routing.

I II II III

Phase II III

1.4

1.0

Phase II Total al+ml / hl III

8.5

MeanRelativeContribution

Trend (smoothed)

1.2

Attn al / hl

0.8

MLP ml / hl+al Full-attn layers (8/32)

1.0

cos(,)hh1

| |
|---|

0.6

I II

0.8

II III

Phase I (outlier)

0.6

0.4

0.4

0.2

cos(hl, hl 1)

0.2

Trend (smoothed)

Phase I

Full-attn layers (8/32)

0.0

0.0

1 4 8 12 16 20 24 28 32 Layer index

1 4 8 12 16 20 24 28 32 Layer index

(a) Relative Contribution Norm

(b) Residual I/O Cosine Similarity

- Figure 5: Layer-wise dynamics of Qwen3.5-9B-Base on GSM8K (cf. Figure 2 for 35B-A3B). Gray bands mark the 8 full-attention layers (l ∈ {4,8,12,16,20,24,28,32}). (a) Norm Ratio: Phase I overwrite at l=1 (≈8.5), stable Phase II (0.26–0.58), Phase III resurgence at l=32 (≈0.84). (b) IO-CosSim: high fidelity throughout Phase II (0.88–0.97), dropping to 0.80 at l=32. The three-phase structure is qualitatively identical to the 35B-A3B result.

##### B.1 Layer-wise Dynamics

- Figure 5 reports the Relative Contribution Norm and Residual I/O Cosine Similarity for Qwen3.5-9B-Base on GSM8K, mirroring the 35B-A3B analysis in Figure 2. The three-phase progression is clearly preserved:

- • Phase I (l=1): Norm Ratio≈8.5 with IO-CosSim≈0.14, confirming that the first layer effectively overwrites the token embedding.
- • Phase II (5 ≤ l ≤ 31): Norm Ratio stabilises at 0.26–0.58 and IO-CosSim remains high (0.88–0.97), consistent with incremental, directionally faithful refinement.
- • Phase III (l=32): Norm Ratio resurges to ≈0.84 (roughly 2× the Phase II plateau) and IO-CosSim drops to ≈0.80—the largest directional deflection in the late regime, matching the pattern observed in 35B-A3B.

B.2 Per-Token Entropy Partitioning

- Figure 6 replicates the per-token entropy partitioning from Figure 4 on Qwen3.5-9B using GPQA Diamond (N=50 prompts, 4,096 generated tokens per prompt, 203,520 tokens total). Following the same

protocol, each generated token is classified as perturbed (entropy rises at the final layer: Ht(32) > Ht(31)) or unperturbed (entropy does not rise). On the 9B backbone, 47.4% of tokens are perturbed (mean ∆H=+0.34

nats), compared with only 16.2% on 35B-A3B. This substantially higher perturbation rate is consistent with the compressed refinement corridor (L=32 vs. L=40) discussed in Appendix E: fewer layers complete the Phase II refinement, leaving the final full-attention layer more likely to overshoot. Despite this quantitative difference, the qualitative three-phase pattern is preserved, confirming that per-token Phase III heterogeneity is a general phenomenon across model scales.

### C Token-Level Substitution Analysis

To shed light on what Confident Decoding actually changes at the token level, we inspect the tokens that are substituted when the entropy-trough layer disagrees with the final layer. Table 8 lists the 20 most frequently substituted tokens for both the Base and Instruct variants of Qwen3.5-35B-A3B on GPQA-Diamond, together with a category breakdown of all substitution events.

### D More Details of the Baselines

We compare Confident Decoding against two representative contrastive decoding baselines: Decoding by Contrasting Layers (DoLa) (Chuang et al., 2023) and Self Logits Evolution Decoding (SLED) (Zhang et al., 2024). Both methods were originally developed and evaluated on standard dense Transformers (e.g., LLaMA-family models), and their official implementations rely on layer-indexing conventions and residual-stream access patterns specific to homogeneous dense architectures. Consequently, neither can be directly applied to modern hybrid or MoE backbones such as Qwen3.5-35B-A3B, where interleaved DeltaNet/full-attention layers and sparse expert routing fundamentally alter the layer-wise representation geometry.

H(pt(l)) Perturbed tokens (96549/203520, 47.4%) Phase III present Qwen3.5-9B · GPQA Diamond

V*

Vonset

10.0

Zoom: tail rise

()[nats]()lHpt

2

7.5

0

5.0

28 29 30 31 32

Mean ±1

Phase II

2.5

Phase III

Full-attn (8/32)

Phase I

0.0

5 10 15 20 25 30

Layer index l

(a) Perturbed tokens — Phase III present

H(pt(l)) Unperturbed tokens (106971/203520, 52.6%) Phase I II only Qwen3.5-9B · GPQA Diamond

12.5

V*

Vonset

10.0

()[nats]()lHpt

7.5

5.0

Mean ±1

Phase II

2.5

Full-attn (8/32)

Phase I

0.0

5 10 15 20 25 30

Layer index l

(b) Unperturbed tokens — no Phase III

- Figure 6: Mean logit-lens entropy H(p(tl)) per layer for Qwen3.5-9B on GPQA Diamond (cf. Figure 4 for 35B-A3B). Tokens are partitioned by whether the final layer (l=32) raises entropy relative to l=31. (a) Perturbed tokens (47.4%, mean ∆H=+0.34 nats) show a Phase III entropy rise at the final full-attention layer, mirroring the 35B-A3B pattern but at a substantially higher rate (47.4% vs. 16.2%), consistent with the compressed corridor (L=32 vs. L=40). (b) Unperturbed tokens (52.6%) continue their Phase II refinement through l=32 without disruption. The backward scan naturally selects l=32 for these tokens, recovering standard decoding.

- Table 8: Top-20 most frequently substituted tokens when Confident Decoding disagrees with Standard Decoding, comparing Base and Instruct variants of Qwen3.5-35B-A3B on GPQA-Diamond. Standard columns show the tokens that the final layer (layer 40) would emit; Confident columns show what the entropy-trough layer selects instead. Percentages denote the fraction of all substitution events in each model. Category breakdowns (Content / Function / Punctuation) are summarized at the bottom. Base: 956 substitutions out of 40,526 tokens (2.4%); Instruct: 939 out of 36,111 (2.6%).

Base Model Instruct Model

Standard (Last Layer) Confident (Trough) Standard (Last Layer) Confident (Trough)

- 1 . 4.8% Actually 4.1% . 7.7% ($ 6.7%
- 2 the 4.4% <|box_end|> 2.5% \{}n 4.2% carb 2.1%
- 3 , 3.1% careful 2.3% the 4.2% ring 1.8%
- 4 is 2.9% perpendicular 1.9% $ 3.5% $, 1.7%
- 5 (space) 2.7% carb 1.8% is 3.5% $\{} 1.6%
- 6 k 2.5% phenotype 1.7% $ 3.0% Structure 1.6%
- 7 g 2.0% fusion 1.7% (space) 2.1% $. 1.4%
- 8 ( 1.9% maybe 1.5% ( 2.1% $( 1.3%
- 9 : 1.9% fused 1.5% , 2.0% option 1.1%
- 10 a 1.8% indeed 1.4% $\{} 1.7% Sometimes 1.0%
- 11 So 1.8% actually 1.3% ). 1.5% methyl 0.9%
- 12 to 1.7% each 1.3% ? 1.5% **+ 0.9%
- 13 with 1.4% neither 1.2% \{} 1.4% \{} 0.9%
- 14 The 1.3% an 1.0% The 1.3% normalization 0.7%
- 15 that 1.3% mutating 1.0% a 1.3% definitely 0.7%
- 16 it 1.2% exactly 0.8% = 1.2% Corey 0.6%
- 17 G 1.2% oxygen 0.8% So 1.2% an 0.6%
- 18 ) 1.0% sqrt 0.8% : 1.1% ?). 0.6%
- 19 But 1.0% acting 0.8% to 1.0% + 0.6%
- 20 are 1.0% sometimes 0.7% Carb 1.0% ’s 0.6%

Category breakdown of all substituted tokens (%):

Content 29% 77% 26% 60% Function 39% 10% 27% 6% Punctuation 22% 6% 43% 26% Other 10% 7% 4% 8%

To enable a fair comparison, we re-implemented both DoLa and SLED within the same inference framework used for Confident Decoding, adapting them to operate correctly on the hybrid MoE architecture. We first validated our re-implementations on LLaMA-family models, confirming that they reproduce the originally reported results. Table 9 reports the results on Qwen3.5-35B-A3B. Both contrastive baselines yield modest improvements over last layer decoding, but their gains fall substantially short of those

achieved by Confident Decoding.

- Table 9: Comparison of contrastive decoding baselines on Qwen3.5-35B-A3B. DoLa and SLED are reimplemented within our inference framework to support the hybrid MoE architecture. Both contrastive baselines yield smaller gains than Confident Decoding on this backbone.

Method GPQA-D HLE LCB-v6 LongBench-v2 Omni-Math Air-Bench WritingBench

Last Layer 76.3 7.1 70.0 63.9 72.6 51.0 65.3 DoLa 77.3 7.8 70.9 63.4 72.4 52.0 65.2 SLED 78.8 7.4 71.7 63.2 72.1 53.0 65.3 Confident (Ours) 82.8 9.5 75.1 63.7 72.7 56.0 65.4

The limited effectiveness of contrastive baselines on hybrid MoE architectures is structurally predictable. For example, DoLa dynamically selects a premature layer that maximally diverges from the final layer (measured by Jensen–Shannon divergence) and decodes from the contrastive log-probability difference between the two, implicitly assuming that early and late layers occupy a representationally comparable subspace differing only in depth of reasoning. On hybrid backbones, the interleaving of DeltaNet and

full-attention layers introduces discontinuous shifts in representation geometry (as formalized by the ϵtype(l) term in our degradation analysis), breaking this subspace-comparability assumption. The contrastive signal is therefore diluted by architectural noise, reducing its ability to isolate the intended factuality gradient. SLED applies gradient-style corrections based on the evolution of logits across layers, requiring the logit trajectory to be locally smooth and monotonically informative. The non-monotonic entropy profile induced by layer-type alternation and expert switching corrupts the logit-evolution signal, limiting the effectiveness of the corrections.

Both baselines share a common structural limitation: they assume representational homogeneity across the layers being contrasted—a condition that modern hybrid and MoE architectures do not satisfy. Confident Decoding mitigates this limitation through a fundamentally different evaluation strategy: rather than contrasting logit distributions across layers (which requires representational comparability between the contrasted pair), it independently evaluates the predictive entropy at each candidate layer and selects the one with minimal uncertainty. Although it relies on the same unembedding projection WU and is therefore not immune to probe mismatch (as analyzed in the degradation analysis below), the per-layer independent evaluation avoids amplifying structured noise through cross-layer subtraction, making it less sensitive to the heterogeneous geometry of modern architectures.

- E Degradation Analysis

- Table 10 reveals that Confident Decoding does not uniformly benefit all backbones. MoE models (e.g., Qwen3.5-35B-A3B) and larger Transformer-based models (e.g., Qwen3.5-27B, gpt-oss-20b) generally benefit or maintain performance across benchmarks, while the compact hybrid Qwen3.5-9B presents a mixed picture: it suffers notable regressions on certain benchmarks (e.g., GPQA-D: 64.6 → 62.1, OmniMATH: 49.1 → 47.1) yet simultaneously achieves clear improvements on others (LCB-v6: 41.1 → 47.7, Air-Bench: 53.0 → 56.0, HLE: 5.2 → 6.1). This section analyzes the degradation through two interacting factors—architectural heterogeneity and model depth—and discusses what structural properties make certain backbones more amenable to Confident Decoding.

- Factor I: Hybrid architecture compresses the refinement corridor. Qwen3.5 family employs a hybrid design. For example, Qwen3.5-9B interleaves 24 linear-attention (DeltaNet) layers with 8 full-attention layers (L=32). DeltaNet layers perform recurrent-style state updates that produce representations with a stronger “process state” character, while full-attention layers periodically reorganize global context. Because adjacent layers may belong to fundamentally different computational paradigms, the late refinement corridor—the band of layers where the model’s token-level prediction monotonically sharpensis substantially compressed. Consequently, even a single-layer rollback in the backward scan can cross the decision boundary between the final full-attention consolidation layer and a pre-final DeltaNet state that has not yet completed global-context integration.

The root mechanism is a probe mismatch induced by layer-type alternation. The backward scan evaluates entropy by projecting each intermediate hidden state through the shared final unembedding matrix WU—a projection that is well-calibrated only when the probed layer’s representation lies in roughly the same geometric subspace as the final layer’s. In the hybrid architecture, however, successive layers alternate between linear-attention recurrence and full-attention recomposition, causing the representation geometry to shift at each type boundary. The effective probe error ϵ(l) therefore acquires a structured, layer-type-dependent component. Decomposing the projection noise ϵ(l) from Section 3.3 into a smooth

baseline term and a layer-type term:

Hˆ (l) = H∗(l) + ϵprobe(l) + ϵtype(l) + η(l), (8)

where ϵtype(l) exhibits discontinuous jumps at DeltaNet↔full-attention transitions. When this structured noise dominates the true entropy gradient ∆H∗(l), the observed entropy valley argminl Hˆ (l) can deviate from the true semantic optimum argminl H∗(l), causing the backward scan to commit tokens to preconvergent representations. Notably, this probe mismatch is task-dependent: on benchmarks where the entropy valley signal is strong and concentrated in the final few layers (e.g., code generation in LCB-v6), the structured noise is insufficient to derail the scan, and even the 9B backbone achieves a clear gain (+6.6). The degradation concentrates on tasks such as GPQA-D and Omni-MATH, where the entropy gradient in the refinement corridor is shallower and more susceptible to being masked by ϵtype(l) .

- Factor II: Model depth widens the refinement corridor. Network depth modulates the degradation through a straightforward mechanism: deeper networks allocate more consecutive layers of the same type

to the late refinement zone, widening the corridor over which |∆H∗(l)| ≫ |∆ϵ(l)| holds and providing a larger margin for the backward scan to locate a reliable valley. Qwen3.5-9B (L=32) has only 8 full-attention layers spread across the entire stack, leaving the final refinement zone with very few homogeneous layers. By contrast, Qwen3.5-35B-A3B (L=40) and Qwen3.5-27B have substantially deeper stacks, and the wider corridor trivially satisfies the local-monotonicity precondition. This is consistent with the data: Qwen3.5-27B exhibits no degradation on GPQA-D (80.0 → 80.0) and gains +10.1 on LCB-v6. We note that depth and architecture are partially confounded in our model suite—the deeper models also differ in parameter count and expert routing—so we cannot fully disentangle the contribution of depth alone from other architectural factors. Nonetheless, the pattern is consistent: shallower networks with narrower corridors are more vulnerable to probe-error-induced misselection.

Why MoE architectures appear robust. Qwen3.5-A series—an MoE backbone that also employs hybrid attention—achieves significant Confident Decoding gains. Two properties may contribute beyond their depth. First, sparse expert routing concentrates task-relevant updates in specialized sub-networks, which may amplify the true refinement signal |∆H∗(l)| in the late corridor and make it easier to dominate probe noise. Second, because each token activates only a small subset of experts, the effective representation trajectory across adjacent layers is smoother than in a dense hybrid backbone of equivalent depth,

potentially reducing the type-switching probe error ϵtype(l) . However, we emphasize that in our current experimental setup, depth and MoE routing are confounded: disentangling their individual contributions would require comparing models that differ only in one of these axes, which we leave to future work.

Scope delimitation. The degradation on Qwen3.5-9B does not invalidate the entropy-valley hypothesis. Rather, the failure arises when deterministic valley selection (p=1.0) is applied to a backbone whose refinement corridor is too narrow to tolerate universal rollback on certain task types. The core precondition

for Confident Decoding—as formalized in the dominance condition of Theorem 1 (∆η(l) > 2ϵmax)—is that the late refinement corridor must be sufficiently wide for the true entropy gradient to dominate the

total probe error: |∆H∗(l)| ≫ |∆ϵ(l)|, where ϵ(l) = ϵprobe(l) + ϵtype(l) . This condition is more readily satisfied on deeper networks, but can be violated on compact hybrid architectures where layer-type alternation introduces structured probe noise and limited depth compresses the safe corridor.

We also note that even models for which Confident Decoding is broadly beneficial can exhibit task-specific regressions. For instance, Qwen3.5-27B—a deeper model that gains strongly on reasoning benchmarksshows a −3.0 drop on Air-Bench (67.0 → 64.0). This suggests that the alignment-tax mechanism and the probe-mismatch mechanism can interact differently depending on the task’s reliance on final-layer formatting versus intermediate-layer reasoning. A complete characterization of these per-task dynamics is an important direction for future work.

These findings delineate two axes—architectural homogeneity and model depth—along which the applicability of Confident Decoding varies.

Table 10: Full results for last layer decoding and confident decoding across various model families.

Model Method GPQA-D HLE LCB-v6 LongBench-v2 Omni-Math Air-Bench WritingBench Qwen3.5-0.8B

Last Layer 0.0 0.0 0.0 0.0 0.0 52.0 10.0 Confident 0.0 0.0 0.0 0.0 0.0 54.0 10.1

Last Layer 64.6 5.2 41.1 62.0 49.1 53.0 60.7 Confident 62.1 6.1 47.7 60.5 47.1 56.0 59.5

Qwen3.5-9B

Last Layer 67.7 5.4 49.1 54.4 54.7 76.0 61.1 Confident 65.7 6.3 50.5 56.1 53.0 76.0 63.0

Qwen3.5-9B-Base

Last Layer 80.0 16.0 63.7 63.7 76.1 67.0 66.7 Confident 80.0 17.0 73.8 64.0 76.1 64.0 66.9

Qwen3.5-27B

Last Layer 76.3 7.1 70.0 63.9 72.6 51.0 65.3 Confident 82.8 9.5 75.1 63.7 72.7 56.0 65.4

Qwen3.5-35B-A3B

Last Layer 83.3 14.7 76.8 66.4 78.3 66.0 72.6 Confident 85.4 16.5 79.2 66.7 78.5 67.0 72.7

Qwen3.5-122B-A10B

Last Layer 76.8 9.2 75.1 58.6 68.3 74.0 64.2 Confident 80.8 11.0 78.3 59.1 69.0 76.0 64.4

Gemma-4-31B

Last Layer 57.1 5.5 77.4 44.6 60.9 86.0 54.4 Confident 60.1 6.4 78.0 46.3 61.6 89.0 54.9

gpt-oss-20b

Last Layer 76.3 10.8 83.1 54.7 70.2 88.0 67.5 Confident 80.8 12.6 84.5 55.6 71.0 90.0 67.8

gpt-oss-120B

###### Standard Decoding Last Layer Entropy (tokens 0 298)

think The user wants me to identify the correct energy difference between two

###### quantum states to clearly distinguish them based on their lif etimes . 1 . ** An

alyze the Physics Principle :** * The problem relates to levels , E _ 1 , $ E

_ 2 $ ) and their lif etimes ($ \ tau _ 1 $, $\ tau _ 2 $ ). * According to

the He isen berg Unc ertainty Principle , there is a fundamental between the

uncertainty in energy ($ \ Delta E $ ) and the uncertainty in time ( lifetime $\ tau

$ ). $\ Delta E \ cdot \ tau \ ge q \ frac {\ h bar }{ 2 }$ ( or roughly $\ Delta E \

approx \ frac {\ h bar }{ \ tau }$ ). * This $\ Delta E $ represents the natural

linewidth ( energy width ) of the state . * To " clear ly distinguish " or

resolve two energy levels , the difference between their central ($ \ Delta E _{ 1 2

###### } = | E _ 1 - E _ 2 |$ ) must be larger than the sum of their linewidth s ( or

at least comparable to the larger linewidth ). A common criterion for resolution is

###### that the energy separation should be larger than the uncertainty in energy of the

states involved . * Let , if the energy difference is smaller than the

###### linewidth , the peaks will overlap significantly and cannot be resolved . If the

energy difference is larger than the linewidth , , they can be resolved . 2 .

###### Confident Decoding Valley Layer Entropy (tokens 0 298)

think The user wants me to identify the correct energy difference between two

###### quantum states to clearly distinguish them based on their lif etimes . 1 . ** An

alyze the Physics Principle :** * The problem relates energy levels ($ E _ 1

$, $ E _ 2 $ ) and their lif etimes ($ \ tau _ 1 $, $\ tau _ 2 $ ). *

According to the He isen berg Unc ertainty Principle , there is a relationship between

the uncertainty in energy ($ \ Delta E $ ) and the uncertainty in time ( lifetime $\

tau $ ): $\ Delta E \ cdot \ tau \ ge q \ frac {\ h bar }{ 2 }$ ( or roughly $\ Delta E

\ approx \ frac {\ h bar }{ \ tau }$ ). * This $\ Delta E $ represents the

natural linewidth ( energy width ) of the state . * To " clear ly distinguish

" or resolve two energy levels , the difference between their energies ($ \ Delta E

###### _{ 1 2 } = | E _ 1 - E _ 2 |$ ) must be greater than the sum of their linewidth s

( or at least comparable to the larger linewidth ). A common criterion for resolution

###### is that the energy separation should be greater than the uncertainty in energy of

the states involved . * Specifically , if the energy difference is smaller

than the linewidth , the peaks will overlap significantly and cannot be resolved . If

the energy difference is larger than the linewidth s , they can be resolved . 2 .

[Figure 4]

0.00 0.05 0.10 0.15 0.20

Token Entropy

Figure 7: Visualization of token entropy (part 1).

###### Standard Decoding Last Layer Entropy (tokens 299 714)

** Calculate the Energy Unc ert ainties :** * Given lif tau _ 1 = 1 0 ^{-

9 }$ s . * Given $\ tau _ 2 = 1 0 ^{- 8 }$ s . * Plan ck 's

constant $\ h bar \ approx 6 . 5 8 2 \ times 1 0 ^{- 1 6 }$ e V $\ cdot $s .

* Unc ertainty in energy for state 1 : $\ Delta E _ 1 \ approx \ frac {\ h bar }{

\ tau _ 1 } = $\ Delta E _ 1 \ approx \ frac { 6 . 6 \ times 1 0 ^{- 1 6

} \ text { e V }\ cdot \ text {s }} { 1 0 ^{- 9 } \ text { s }} \ approx 6 . 6 \

times 1 0 ^{- 7 }$ e V . * Unc ertainty in energy for state 2 : $\ Delta

E _ 2 \ approx \ frac {\ h bar }{ \ tau _ 2 } $. $\ Delta E _ 2 \ approx \

frac { 6 . 6 \ times 1 0 ^{- 1 6 } \ text { e V }\ cdot \ text {s }} { 1 0 ^{- 8 } \

text { s }} \ approx 6 . 6 \ times 1 0 ^{- 8 }$ e V . 3 . ** D etermine the

Resolution Condition :** * To resolve two levels , the energy difference $\

Delta E _{ 1 2 }$ must be be larger than the energy widths . \ Delta E _ 1 $ and $\

Delta E _ 2 $ ). * The larger width dominates the resolution limit . Here ,

$\ Delta E _ 1 \ approx 1 0 ^{- 7 }$ e V is larger than $\ Delta E _ 2 \ approx

1 0 ^{- 8 }$ e V . * So , we need $\ Delta E _{ 1 2 } \ gg \ Delta E _{ max

} or at least $\ Delta E _{ 1 2 } > \ Delta E _{ max } $. * $\

###### Confident Decoding Valley Layer Entropy (tokens 299 714)

** Calculate the Energy Unc ert ainties :** * Given $\ tau _ 1 = 1 0 ^{- 9

}$ s . * Given $\ tau _ 2 = 1 0 ^{- 8 }$ s . * Plan ck 's

constant $\ h bar \ approx 6 . 5 8 2 \ times 1 0 ^{- 1 6 }$ e V $\ cdot $s .

* Unc ertainty in energy for state 1 : $\ Delta E _ 1 \ approx \ frac {\ h bar }{

\ tau _ 1 } $. $\ Delta E _ 1 \ approx \ frac { 6 . 6 \ times 1 0 ^{- 1 6

} \ text { e V }\ cdot \ text {s }} { 1 0 ^{- 9 } \ text { s }} \ approx 6 . 6 \

times 1 0 ^{- 7 }$ e V . * Unc ertainty in energy for state 2 : $\ Delta

E _ 2 \ approx \ frac {\ h bar }{ \ tau _ 2 } $. $\ Delta E _ 2 \ approx \

frac { 6 . 6 \ times 1 0 ^{- 1 6 } \ text { e V }\ cdot \ text {s }} { 1 0 ^{- 8 } \

text { s }} \ approx 6 . 6 \ times 1 0 ^{- 8 }$ e V . 3 . ** D etermine the

Resolution Condition :** * To resolve two levels , the energy difference $\

Delta E _{ 1 2 }$ should ideally be greater than the energy widths ($ \ Delta E _ 1 $

and $\ Delta E _ 2 $ ). * The larger width dominates the resolution limit .

Here , $\ Delta E _ 1 \ approx 1 0 ^{- 7 }$ e V is larger than $\ Delta E _ 2 \

approx 1 0 ^{- 8 }$ e V . * So , we need $\ Delta E _{ 1 2 } \ gg \ Delta

E _{ max }$ or at least $\ Delta E _{ 1 2 } > \ Delta E _{ max } $. * $\

[Figure 5]

0.00 0.05 0.10 0.15 0.20

Token Entropy

Figure 8: Visualization of token entropy (part 2).

###### Standard Decoding Last Layer Entropy (tokens 715 1095)

Delta E _{ max } \ approx 1 0 ^{- 7 }$ e V . * Let at the options :

* ( A ) $ 1 0 ^{- 9 }$ e V : This is smaller than both linewidth s ($ 1

- 0 ^{- 7 }$ and $ 1 0 ^{- 8 }$ ). The levels would overlap completely . * (

B ) $ 1 0 ^{- 8 }$ e V : This is comparable to the smaller linewidth but \ Delta E _

2 $ ) but smaller than the larger linewidth ($ \ Delta E _ 1 \ approx 6 \ times

- 1 0 ^{- 7 }$ e V ). It would be poor . * ( C ) $ 1 0 ^{- 4 }$ e V :

This is significantly larger than both linewidth s ($ 1 0 ^{- 7 }$ and $ 1 0 ^{- 8 }$

). The levels would be well separated . * ( D ) $ 1 0 ^{- 1 1 }$ e V :

This is much smaller than both linewidth s . Impossible to resolve . 4 . **

Evaluate the Options :** * The question asks for an energy difference so that

they can be * clear ly resolved *. * Condition : Energy Difference > Energy

Width . * Energy Width $\ Gamma \ approx \ h bar / \ tau $. * $\

Gamma _ 1 \ approx 6 . 6 \ times 1 0 ^{- 7 }$ e V . * $\ Gamma _ 2 \

approx 6 . 6 \ times 1 0 ^{- 8 }$ e V . * We need $\ Delta E > \ Gamma

|_| |$\| |
|---|---|---|---|
|_ 2| |$|$\|

Delta E \ gg \ Gamma $. * Option ( A ) $ 1 0 ^{- 9 }$ e V < $\ Gamma

Gamma _ 1 $. Not resolved . * Option ( B ) $ 1 0 ^{- 8 }$

###### Confident Decoding Valley Layer Entropy (tokens 715 1095)

Delta E _{ max } \ approx 1 0 ^{- 7 }$ e V . * Looking at the options :

* ( A ) $ 1 0 ^{- 9 }$ e V : This is smaller than both linewidth s ($ 1

- 0 ^{- 7 }$ and $ 1 0 ^{- 8 }$ ). The levels would overlap completely . * (

B ) $ 1 0 ^{- 8 }$ e V : This is comparable to the smaller linewidth ($ \ Delta E _

2 $ ) but smaller than the larger linewidth ($ \ Delta E _ 1 \ approx 6 \ times

- 1 0 ^{- 7 }$ e V ). Resolution would be poor . * ( C ) $ 1 0 ^{- 4 }$ e

V : This is significantly larger than both linewidth s ($ 1 0 ^{- 7 }$ and $ 1 0 ^{-

8 }$ ). The levels would be clearly separated . * ( D ) $ 1 0 ^{- 1 1 }$

e V : This is much smaller than both linewidth s . Impossible to resolve . 4 .

** Evaluate the Options :** * The question asks for an energy difference so

that they can be * clear ly resolved *. * Condition : Energy Difference >

Energy Width . * Energy Width $\ Gamma \ approx \ h bar / \ tau $. *

$\ Gamma _ 1 \ approx 6 . 6 \ times 1 0 ^{- 7 }$ e V . * $\ Gamma _ 2 \

approx 6 . 6 \ times 1 0 ^{- 8 }$ e V . * We need $\ Delta E > \ Gamma

$. Ideally $\ Delta E \ gg \ Gamma $. * Option ( A ) $ 1 0 ^{- 9 }$ e V <

$\ Gamma _ 2 <$ $\ Gamma _ 1 $. Not resolved . * Option ( B ) $ 1 0 ^{- 8 }$

[Figure 6]

0.00 0.05 0.10 0.15 0.20

Token Entropy

Figure 9: Visualization of token entropy (part 3).

###### Standard Decoding Last Layer Entropy (tokens 1096 1529)

e V $\ approx \ Gamma _ 2 <$ $\ Gamma _ 1 $. Not clearly resolved . * Option

( C ) $ 1 0 ^{- 4 }$ e V $\ gg \ Gamma _ 1 $. Clearly resolved . * Option (

D ) $ 1 0 ^{- 1 1 }$ e V < $\ Gamma _ 2 <$ $\ Gamma _ 1 $. Not resolved . 5 .

** Conclusion :** * Option ( C ) is the only value large enough to exceed

the natural linewidth s determined by the lif etimes . 6 . ** Formatting :**

* " The correct answer is ( insert answer here )" Let 's double check the

calculation . $\ h bar \ approx 1 . 0 5 4 \ times 1 0 ^{- 3 4 }$ J s .

$ 1 \ text { e V } = 1 . 6 0 2 \ times 1 0 ^{- 1 9 }$ J . $\ h bar \

text { in e V s } = \ frac { 1 . 0 5 4 \ times 1 0 ^{- 3 4 }} { 1 . 6 0 2 \ times

1 0 ^{- 1 9 }} \ approx 6 . 5 8 \ times 1 0 ^{- 1 6 }$ e V s . $\ Delta E

_ 1 = \ frac { 6 . 5 8 \ times 1 0 ^{- 1 6 }} { 1 0 ^{- 9 }} = 6 . 5 8 \ times

1 0 ^{- 7 }$ e V . $\ Delta E _ 2 = \ frac { 6 . 5 8 \ times 1 0 ^{- 1 6 }}

{ 1 0 ^{- 8 }} = 6 . 5 8 \ times 1 0 ^{- 8 }$ e V . To resolve , $\ Delta

E _{ diff } > \ Delta E _{ width } $. The limiting width is $\ approx 1 0 ^{- 6

}$ to $ 1 0 ^{- 7 }$ e V range . Option A ($ 1 0 ^{- 9 }$ ) is too small .

Option B ($ 1 0 ^{- 8 }$ ) is too small ( small er than the width of state 1

###### Confident Decoding Valley Layer Entropy (tokens 1096 1529)

e V $\ approx \ Gamma _ 2 <$ $\ Gamma _ 1 $. Not clearly resolved . * Option

( C ) $ 1 0 ^{- 4 }$ e V $\ gg \ Gamma _ 1 $. Clearly resolved . * Option (

D ) $ 1 0 ^{- 1 1 }$ e V < $\ Gamma _ 2 <$ $\ Gamma _ 1 $. Not resolved . 5 .

** Conclusion :** * Option ( C ) is the only value large enough to exceed

the natural linewidth s determined by the lif etimes . 6 . ** Formatting :**

* " The correct answer is ( insert answer here )" Let 's double check the

calculation . $\ h bar \ approx 1 . 0 5 4 \ times 1 0 ^{- 3 4 }$ J s .

$ 1 \ text { e V } = 1 . 6 0 2 \ times 1 0 ^{- 1 9 }$ J . $\ h bar \

text { in e V s } = \ frac { 1 . 0 5 4 \ times 1 0 ^{- 3 4 }} { 1 . 6 0 2 \ times

1 0 ^{- 1 9 }} \ approx 6 . 5 8 \ times 1 0 ^{- 1 6 }$ e V s . $\ Delta E

_ 1 = \ frac { 6 . 5 8 \ times 1 0 ^{- 1 6 }} { 1 0 ^{- 9 }} = 6 . 5 8 \ times

1 0 ^{- 7 }$ e V . $\ Delta E _ 2 = \ frac { 6 . 5 8 \ times 1 0 ^{- 1 6 }}

{ 1 0 ^{- 8 }} = 6 . 5 8 \ times 1 0 ^{- 8 }$ e V . To resolve , $\ Delta

E _{ diff } > \ Delta E _{ width } $. The limiting width is $\ approx 1 0 ^{- 6

}$ to $ 1 0 ^{- 7 }$ e V range . Option A ($ 1 0 ^{- 9 }$ ) is too small .

Option B ($ 1 0 ^{- 8 }$ ) is too small ( small er than the width of state 1

[Figure 7]

0.00 0.05 0.10 0.15 0.20

Token Entropy

Figure 10: Visualization of token entropy (Part 4). For brevity, we omit the CoT following Figure 9 and preceding Figure 10.

###### Standard Decoding Last Layer Entropy (tokens 1530 1838)

). Option C ($ 1 0 ^{- 4 }$ ) is much larger than the widths . 1 0 ^{- 7 }$ ).

Option D ($ 1 0 ^{- 1 1 }$ ) is too small . Therefore , ( C ) is the

correct choice . Wait , is there a specific criterion like Ray leigh criterion

|M $\| | |approx|=|
|---|---|---|---|---|
|V|.|. $<br><br>| | |

? Usually , resolution requires separation $\ ge $ $ F WH M . F WH \ h

bar /\ tau $. So separation needs to be $ > 1 0 ^{- 7 }$ e 1 0

^{- 4 }$ e V fits this perfectly . Final check of the question ph rasing : "

Which one of the following options could be their energy difference so that they can

be clearly resolved ?" This implies finding the sufficient condition among the

choices . Only C satisfies the condition $\ Delta E > \ Delta $. Structure the

response as requested . The correct answer is ( C ) ** Reason ing :**

According to the He isen berg Unc ertainty Principle , the uncertainty in energy ($ \

Delta E $ ) of a quantum state is invers ely proportional to its lifetime ($ \ tau $

), given approximately by $\ Delta E \ approx \ frac {\ h bar }{ \ tau } $. This $\

Delta E $ represents the natural linewidth of the energy level . To clearly

distinguish ( resolve ) two energy levels , the difference between their energies ($ \

Delta E _{ 1 2 }$ ) must be greater than their energy widths ( linewidth s ). 1 :

###### Confident Decoding Valley Layer Entropy (tokens 1530 1838)

). Option C ($ 1 0 ^{- 4 }$ ) is much larger than the widths ($ 1 0 ^{- 7 }$ ).

Option D ($ 1 0 ^{- 1 1 }$ ) is too small . Therefore , ( C ) is the

correct choice . Wait , is there a specific criterion like Ray leigh criterion

|M $\| |Gamma|=|
|---|---|---|---|
|V|roughly| |.<br><br>|

? Usually , resolution requires separation $\ ge q $ F WH M . F WH \ h

bar /\ tau $. So separation needs to be $ > 1 0 ^{- 7 }$ e $

1 0 ^{- 4 }$ e V fits this perfectly . Final check of the question ph rasing :

" Which one of the following options could be their energy difference so that they

can be clearly resolved ?" This implies finding the sufficient condition among

###### the choices . Only C satisfies the condition $\ Delta E > \ Gamma $. Structure

the response as requested . </think> The correct answer is ( C ) ** Reason ing

:** According to the He isen berg Unc ertainty Principle , the uncertainty in energy

($ \ Delta E $ ) of a quantum state is invers ely proportional to its lifetime ($ \

tau $ ), given approximately by $\ Delta E \ approx \ frac {\ h bar }{ \ tau } $. This

$\ Delta E $ represents the natural linewidth of the energy level . To clearly

distinguish ( resolve ) two energy levels , the difference between their energies ($ \

Delta E _{ 1 2 }$ ) must be greater than their energy widths ( linewidth s ). Given :

[Figure 8]

0.00 0.05 0.10 0.15 0.20

Token Entropy

Figure 11: Visualization of token entropy (part 5).

###### Standard Decoding Last Layer Entropy (tokens 1839 2162)

$\ tau _ 1 = 1 0 ^{- 9 }$ s $\ tau _ 2 = 1 0 ^{- 8 }$ s $\ h bar \ approx

- 6 . 5 8 \ times 1 0 ^{- 1 6 }$ e V $\ cdot $s Calcul ating the energy widths : $\

Delta E _ 1 \ approx \ frac { 6 . 5 8 \ times 1 0 ^{- 1 6 }} { 1 0 ^{- 9 }} \ approx

6 . 6 \ times 1 0 ^{- 7 }$ e V $\ Delta E _ 2 \ approx \ frac { 6 . 5 8 \ times

1 0 ^{- 1 6 }} { 1 0 ^{- 8 }} \ approx 6 . 6 \ times 1 0 ^{- 8 }$ e V The

larger linewidth is the resolution limit , \ approx 1 0 ^{- 7 }$ e V ). , the

energy difference must be significantly larger than $ 1 0 ^{- 7 }$ e V to be clearly

resolved . * ( A ) $ 1 0 ^{- 9 }$ e V is smaller than both linewidth s . *

( B ) $ 1 0 ^{- 8 }$ e V is smaller than the linewidth of the first state . 1 0 ^{-

- 7 }$ e V ). * ( C ) $ 1 0 ^{- 4 }$ e V is much larger than both linewidth s ($

1 0 ^{- 7 }$ e V ), allowing for resolution . * ( D ) $ 1 0 ^{- 1 1 }$ e V is

much smaller than both linewidth s . Thus , $ ( C ) is the only choice . The

correct answer is ( C )

###### Confident Decoding Valley Layer Entropy (tokens 1839 2162)

$\ tau _ 1 = 1 0 ^{- 9 }$ s $\ tau _ 2 = 1 0 ^{- 8 }$ s $\ h bar \ approx

6 . 5 8 \ times 1 0 ^{- 1 6 }$ e V $\ cdot $s Calcul ating the energy widths : $\

Delta E _ 1 \ approx \ frac { 6 . 5 8 \ times 1 0 ^{- 1 6 }} { 1 0 ^{- 9 }} \ approx

6 . 6 \ times 1 0 ^{- 7 }$ e V $\ Delta E _ 2 \ approx \ frac { 6 . 5 8 \ times

1 0 ^{- 1 6 }} { 1 0 ^{- 8 }} \ approx 6 . 6 \ times 1 0 ^{- 8 }$ e V The

larger linewidth dominates the resolution limit ($ \ approx 1 0 ^{- 7 }$ e V ).

Therefore , the energy difference must be significantly larger than $ 1 0 ^{- 7 }$ e

V to be clearly resolved . * ( A ) $ 1 0 ^{- 9 }$ e V is smaller than both

linewidth s . * ( B ) $ 1 0 ^{- 8 }$ e V is smaller than the linewidth of the

first state ($ 1 0 ^{- 7 }$ e V ). * ( C ) $ 1 0 ^{- 4 }$ e V is much larger

than both linewidth s ($ 1 0 ^{- 7 }$ e V ), allowing clear resolution . * ( D )

$ 1 0 ^{- 1 1 }$ e V is much smaller than both linewidth s . Thus , option ( C )

is the correct choice . The correct answer is ( C )

[Figure 9]

0.00 0.05 0.10 0.15 0.20

Token Entropy

Figure 12: Visualization of token entropy (part 6).

