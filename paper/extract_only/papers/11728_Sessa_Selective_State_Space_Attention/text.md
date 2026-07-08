# arXiv:2604.18580v2[cs.LG]21Apr2026

## Sessa: Selective State Space Attention

Liubomyr Horbatko liubomir.horbatko@gmail.com

Abstract

Modern sequence modeling is dominated by two families: Transformers, whose self-attention can access arbitrary elements of the visible sequence, and structured state-space models, which propagate information through an explicit recurrent state. These mechanisms face different limitations on long contexts: when attention is diffuse, the influence of individual tokens is diluted across the effective support, while recurrent state propagation can lose long-range sensitivity unless information is actively preserved. As a result, both mechanisms face challenges in preserving and selectively retrieving information over long contexts. We propose Sessa, a decoder that places attention inside a recurrent feedback path. This creates many attention-based paths through which past tokens can influence future states, rather than relying on a single attention read or a single recurrent chain. We prove that, under explicit assumptions and matched regimes, Sessa admits power-law memory tails 𝑂(ℓ−𝛽) for 0 < 𝛽 < 1, with slower decay than in the corresponding Transformer and Mamba-style baselines. We further give an explicit construction that achieves this power-law rate. Under the same assumptions, Sessa is the only model class among those considered that realizes flexible selective retrieval, including profiles whose influence does not decay with distance. Consistent with this theoretical advantage, across matched experiments, Sessa achieves the strongest performance on long-context benchmarks while remaining competitive with Transformer and Mamba-style baselines on short-context language modeling.

### 1 Introduction

Long-context sequence modeling is central to modern foundation models across language, vision, speech, time series, and genomics (Bommasani et al., 2021; Brown et al., 2020; Dosovitskiy et al., 2021; Baevski et al., 2020; Ansari et al., 2024; Dalla-Torre et al., 2025). Despite the architectural flexibility of the foundation-model paradigm, state-of-the-art systems are still overwhelmingly based on the Transformer and its self-attention mechanism (Vaswani et al., 2017).

A useful lens is to describe modern sequence mixers by how they route information from the past and how they maintain memory over time. In many modern architectures, routing decisions are input-dependent: the model uses the current token and its context to decide which parts of the visible history to consult. Under this view, self-attention implements an input-dependent direct-read mechanism: at each position, it computes a querydependent pattern of relevance over the visible context and uses it to read out information from selected past positions. This framing highlights attention’s key strength, a selection mechanism over variable support length, but also a structural limitation: the retrieval is performed in a single pass, without an internal feedback loop that would repeatedly incorporate past readouts into an evolving state. Separately, standard implementations are also computationally expensive at long contexts due to quadratic time/memory scaling (Vaswani et al., 2017; Rabe and Staats, 2021).

In parallel, structured recurrent sequence models, especially state space models (SSMs), which realize long-range dynamics through a latent state and an explicit feedback path, have re-emerged as a compelling alternative for long-context modeling (Gu et al., 2022a,b). SSMs can be interpreted as modern descendants of classical dynamical systems (Kalman, 1960) and admit linear (or near-linear) scaling in sequence length. However, for informationdense discrete data, a persistent challenge is that stable feedback dynamics often exhibit rapid attenuation of distant information (commonly exponential forgetting (Huang et al., 2025)), which can hinder integrating multiple far-apart evidence snippets under heavy distractors. Selective SSMs (e.g., Mamba) can conditionally slow this

attenuation by modulating the effective transition (Gu and Dao, 2024; Dao and Gu, 2024) (e.g., 𝐴ssm,𝑡 ≈ 𝐼 on selected steps, “freeze time” (Huang et al., 2025)), but this mechanism is input-dependent and can fail when relevant and irrelevant positions induce similar local representations, leading to preserving or overwriting the wrong content.

These perspectives suggest complementary long-context failure modes. Stable feedback dynamics can suffer from exponential forgetting. Attention, while input-dependent, can suffer from dilution: when attention mass is spread across a large effective support of competing tokens (e.g., many near-tied logits), individual weights, and thus per-token contributions and sensitivities, decrease roughly inversely with that support (often behaving like 𝑂(1/𝑆eff(𝑡)), and in the worst case like 𝑂(1/𝑇) when the effective support grows proportionally with context length 𝑇)(Mudarisov et al., 2025). In practice, both effects can limit reliable long-range evidence integration.

We introduce Sessa, a decoder architecture that injects input-dependent attention into a feedback (recurrent) path, combining direct-read input-dependent routing with stateful aggregation through the feedback channel. Viewed through a temporal routing lens, for a fixed source token 𝜏 and target position 𝑡 (lag ℓ = 𝑡 − 𝜏), a single self-attention layer routes influence via a single routing step (a direct edge 𝜏 → 𝑡), while chain-structured state-space recurrences propagate along the unique length-ℓ temporal chain. Sessa introduces route diversity within a single layer: its attention-induced feedback operator aggregates contributions over multiple internal routing depths (and, in dense patterns, many temporal paths), which can help sustain long-range sensitivity when routing is diffuse (formalized in Section 4.2). Concretely, while self-attention corresponds to an inputdependent direct-read system (in the values), Sessa realizes an input-dependent feedback system: it maintains a latent state over unbounded horizons, while the feedback dynamics remain input-dependent via attention-based routing inside the loop (potentially over variable-support patterns). Intuitively, Sessa retains the representational benefits of recurrence for long-range accumulation while leveraging attention as an input-dependent mechanism within the feedback pathway.

Related architectural ideas have introduced recurrence or feedback into sequence modeling (Dai et al., 2019; Fan et al., 2020; Bulatov et al., 2022; Hutchins et al., 2022; Hwang et al., 2024). These approaches span a variety of feedback constructions and are typically presented in architecture-specific terms. Our contribution is complementary but mathematically different: we propose a routing-induced systems perspective that separates how context produces routing/mixing coeﬀicients from how those coeﬀicients are composed over time, and we use this lens to relate input-dependent routing directly to long-context sensitivity and memory-decay behavior.

Our contributions are:

- • Architecture. We propose the Sessa sequence mixer, integrating attention into the recurrent feedback pathway under an otherwise standard decoder macro-architecture.
- • Memory. We characterize long-range sensitivity of Sessa and identify a heavy-tail memory regime in which

the feedback solve induces a power-law influence tail in the lag ℓ of order 𝑂(ℓ−𝛽

tail

) with 0 < 𝛽tail < 1. In this diffuse, low-separation routing regime, attenuation is asymptotically slower than the exponential forgetting exhibited by many stable or contractive SSM regimes, and it mitigates inverse-support dilution effects under the stated assumptions (Section 4.2; Theorem 8).

- • Selective retrieval. In the matched theoretical regime, we show that deep Sessa realizes flexible selective retrieval profiles, including non-decaying ones, whereas diffuse fixed-depth Transformers and failed-freezetime fixed-depth Mamba do not (Section 4.2.8; Theorem 12; Proposition 13).
- • Empirics. Under matched architectures and training budgets, Sessa achieves the strongest performance on our long-context benchmarks while remaining competitive on short-context language modeling.

We additionally prove a universal approximation result for a broad class of causal sequence mappings in Appendix I (Theorem 14).

### 2 Background

We separate two largely independent aspects of causal mixers:

- (i) how routing/mixing coeﬀicients are produced from context, and
- (ii) whether information is accessed via a single read or accumulated through feedback.

Terminology We use system to refer to the memory mechanism (direct-read or feedback). We use routing to refer to the coeﬀicients that specify how information flows over time for example attention weights 𝛼fwd, the induced feedback matrix 𝐵fb, or the transition operators in a recurrence. Routing is the collection of coeﬀicients, meaning weights or operators, that determine information flow over time. The system determines whether routing is applied once (direct-read) or repeatedly composed via feedback.

We model a broad class of sequence mixers by expressing each output as a mixture of a chosen stream 𝑢𝑡 with coeﬀicients that may depend on the available context 𝑥0∶𝑡.

- 2.1 Direct-read and feedback systems

- Definition 1 (Direct-read variable-support system). We say that ℱ is a direct-read system with respect to a chosen stream 𝑢𝑡 if, for every 𝑡,

𝑦𝑡 = ∑ 𝜏∈𝑆𝑡

𝐾𝑡,𝜏(𝑥0∶𝑡)𝑢𝜏, 𝑆𝑡 ⊆ {0,…,𝑡}, (1)

so each 𝑦𝑡 is produced by a single input-addressed read, i.e., a mixture over the visible index set 𝑆𝑡. If |𝑆𝑡| varies with 𝑡, we call the system variable-support. If there exists 𝑊 ≥ 1 such that 𝐾𝑡,𝜏 ≡ 0 whenever 𝑡 − 𝜏 ≥ 𝑊, equivalently, 𝑆𝑡 ⊆ {max(0,𝑡 − 𝑊 + 1),…,𝑡}, we call it bounded-support direct-read.

- Remark 2.1 (Kernel representations alone do not distinguish direct-read or feedback). On any finite horizon 𝑇, any causal linear map admits a lower-triangular kernel representation (Kalman, 1960; Antsaklis and Michel, 2006).

𝑦𝑡 = ∑𝜏≤𝑡 𝐾𝑡,𝜏𝑢𝜏, so kernel form alone does not identify whether influence is produced by a single read or by an internal recurrence. Here, direct-read refers to the computation graph: 𝑦𝑡 is formed by one read/mix over a visible set, without repeated composition of the same mixing primitive inside the layer.

Dimensions. 𝑢𝜏 ∈ ℝ𝐷, 𝑦𝑡 ∈ ℝ𝐷, and 𝐾𝑡,𝜏(𝑥0∶𝑡) is a linear map of the appropriate shape. In contrast, models with an explicit state and feedback naturally take a feedback form.

Definition 2 (Feedback system: state-space or operator form). We say that 𝒢 is a feedback system with respect to a chosen stream 𝑢𝑡 if there exist states ℎ𝑡 in a possibly time-varying state space ℋ𝑡 such that, for each 𝑡 ≥ 0,

with, e.g., ℎ−1 = 0, ℎ𝑡 = 𝐴ssm,𝑡(𝑥0∶𝑡)ℎ𝑡−1 + 𝐵ssm,𝑡(𝑥0∶𝑡)𝑢𝑡, 𝑦𝑡 = 𝐶ssm,𝑡(𝑥0∶𝑡)ℎ𝑡 + 𝐷ssm,𝑡(𝑥0∶𝑡)𝑢𝑡. (2)

The recurrence composes the routing over time, so 𝑦𝑡 can depend on arbitrarily old inputs even when each update is local in ℎ𝑡−1.

- Remark 2.2 (One-hop and multi-hop routing). We view routing as propagation on a directed acyclic graph (DAG) over time indices induced by the mixing coeﬀicients. Fix a horizon 𝑇 and nodes {0,…,𝑇 − 1}.

Input sequence

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Mamba

Sessa

Transformer

(one path, multi-hop)

(one path, one hop)

(many paths, multi-hop)

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

(+ ... many)

[Figure 15]

[Figure 16]

[Figure 17]

(+ ... many)

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Figure 1: One-hop and multi-hop temporal routing within a single mixer layer. Transformer: influence from 𝜏 to 𝑡 follows a single direct edge (one-hop). Mamba: influence from 𝜏 to 𝑡 follows the chain 𝜏 → ⋯ → 𝑡 (multi-hop along a single path). Sessa: influence from 𝜏 to 𝑡 aggregates over many paths with varying hop counts (multi-hop over many paths).

Direct-read (one-hop). A direct-read system forms 𝑦𝑡 by a single read from a visible set 𝑆𝑡 using coeﬀicients 𝐾𝑡,𝜏: in the routing graph, this corresponds to using only direct edges 𝜏 → 𝑡. Influence from 𝜏 reaches 𝑡 in one routing step.

Feedback (multi-hop). A feedback mechanism can apply routing repeatedly through an internal state or solve, allowing influence from 𝜏 to reach 𝑡 through paths with intermediate nodes. This repeated composition is what we call multi-hop routing.

The classical finite-dimensional state-space case corresponds to ℋ𝑡 = ℝN with fixed N for all 𝑡. Structured SSM layers (e.g., S4/S4D and Mamba) are instances of this special case.

Hop counts in the solve Sessa’s mixer output 𝑠 is defined by a causal lower-triangular solve

(𝐼 − 𝐵fb)𝑠 = 𝑓, [𝐵fb]𝑡,𝑗 = 0 for 𝑗 ≥ 𝑡, (3)

On any finite horizon 𝑇, 𝐵fb is strictly lower-triangular and hence nilpotent (𝐵fb𝑇 = 0) (Horn and Johnson, 2012). Hence,

𝑇−1

𝑇−1

(𝐼 − 𝐵fb)−1 =

∑

𝐵fb𝑘 , and 𝑠 =

∑

𝐵fb𝑘 𝑓. (4)

𝑘=0

𝑘=0

Each term 𝐵fb𝑘 𝑓 corresponds to routing through 𝑘 feedback steps, a 𝑘-hop contribution. Equivalently, for indices 𝜏 ≤ 𝑡,

𝑘

∏

[𝐵fb]𝑖

𝑟,𝑖𝑟−1, 𝑘 ≥ 1, (5)

(𝐵fb𝑘 )𝑡,𝜏 = ∑

𝜏=𝑖0<𝑖1<⋯<𝑖𝑘=𝑡

𝑟=1

which is a sum over all length-𝑘 directed paths from 𝜏 to 𝑡 in the feedback-induced routing graph. This explicit path expansion is the mechanism behind heavy-tail regimes analyzed later: even if individual edges are small under diffuse routing, the number of admissible paths grows with lag, and the solve aggregates contributions across all hop counts.

#### 2.2 Self-attention as direct-read

Standard causal self-attention fits Definition 1 when the mixed stream is the sequence of value vectors. At position 𝑡, over a visible index set 𝒲𝑡 ⊆ {0,…,𝑡}:

exp(𝜎𝑘 𝑞𝑡⊤𝑘𝑗) ∑𝑖∈𝒲

𝑦𝑡 = ∑ 𝑗∈𝒲𝑡

𝛼fwd𝑡,𝑗 𝑣𝑗, 𝛼fwd𝑡,𝑗 =

, (6)

exp(𝜎𝑘 𝑞𝑡⊤𝑘𝑖)

𝑡

with 𝑞𝑡 = 𝑊𝑄𝑥𝑡, 𝑘𝑗 = 𝑊𝐾𝑥𝑗, and 𝑣𝑗 = 𝑊𝑉 𝑥𝑗. Lemma 2.3 (Self-attention is a direct-read system in 𝑉 ). At each position 𝑡, self-attention computes 𝑦𝑡 by a single input-addressed read from the visible set 𝒲𝑡, mixing the value vectors (𝑣𝑗)𝑗∈𝒲

weights 𝛼fwd𝑡,𝑗 . Full-prefix, windowed, and sparse attention all fit the same direct-read template through the choice of visible set 𝒲𝑡 (Child et al., 2019; Beltagy et al., 2020; Zaheer et al., 2020; Ding et al., 2023).

𝑡

with context-dependent

#### 2.3 State-space models as feedback

Structured state-space models (SSMs) implement sequence mixing through a latent state and a (possibly selective) recurrence. A standard form is

ℎ𝑡 = 𝐴ssm ℎ𝑡−1 + 𝐵ssm 𝑥𝑡, 𝑦𝑡 = 𝐶ssm ℎ𝑡, (7)

where 𝐴ssm ∈ ℝN×N encodes temporal dynamics and is typically constrained (diagonal/structured/low-rank) for eﬀiciency.

ℎ𝑡 = 𝐴ssm,𝑡(𝑥0∶𝑡)ℎ𝑡−1 + 𝐵ssm,𝑡(𝑥0∶𝑡)𝑥𝑡, 𝑦𝑡 = 𝐶ssm,𝑡(𝑥0∶𝑡)ℎ𝑡. (8)

Modern language-oriented SSMs such as Mamba often employ input-dependent recurrences that fit Definition 2:

𝐴ssm,𝑡 = diag(exp(−𝜆𝑛Δ𝑡)), so a lag-ℓ memory factor contains terms of the form

In Mamba, the discrete transition commonly takes the form

𝑡

exp( − 𝜆𝑛

∑

Δ𝑟).

𝑟=𝑡−ℓ+1

Accordingly, long-range memory is preserved only when the model can create a long preserve corridor of steps with Δ𝑟 ≈ 0.

This suggests the matched comparison principle used later in the paper. For attention, broken sharp selection means that softmax mass cannot concentrate on a small set of indices. For Mamba, the analogous failure mode is failed freeze time: the model cannot sustain a long preserve corridor on the relevant interval. For the three-way comparison in this paper, we say that a Mamba layer is in a failed freeze-time regime on an input set of interest

if there exists 𝑐Δ > 0 such that for every relevant pair 𝜏 < 𝑡,

𝑡

∑

Δ𝑟 ≥ 𝑐Δ(𝑡 − 𝜏).

𝑟=𝜏+1

Equivalently, the average discretization step along every relevant interval is bounded below by a positive constant. In Mamba this implies

𝑡

exp( − 𝜆𝑛

∑

Δ𝑟) ≤ 𝑒−𝜆𝑛𝑐Δ(𝑡−𝜏),

𝑟=𝜏+1

so long-range influence is exponentially small in the lag. This is the Mamba counterpart of diffuse attention used in the matched comparisons below: in attention, the selector cannot concentrate mass on a few indices; in Mamba, the model cannot maintain Δ𝑟 ≈ 0 on a long relevant corridor.

### 3 Model Architecture

We instantiate the one-hop and multi-hop routing viewpoint of Section 2.1 with a concrete layer, Sessa. Sessa uses a single gated-MLP-style block that wraps a recurrent mixer, rather than alternating separate attention and MLP blocks. The mixer itself combines (i) a standard causal forward-attention signal and (ii) a feedback term that mixes past mixer outputs.

The oﬀicial implementation is available at https://github.com/LibratioAI/sessa.

Notation. Inputs and outputs have shape 𝑥,𝑦 ∈ ℝ𝐵

batch×𝑇×𝐷 with 𝑡 ∈ {0,…,𝑇 − 1}. We use an internal key

and query width 𝑑𝑘 and scale 𝜎𝑘 = 𝑑𝑘−1/2. All definitions apply per batch element; we omit the batch index when clear.

Given 𝑥 ∈ ℝ𝐵

- 3.1 Sessa block

batch×𝑇×𝐷, the block applies pre-norm, a gated projection, the mixer, and a residual connection:

𝑥̃ = LN(𝑥), (9) (𝑎,𝑔) = split(𝑥̃𝑊in + 𝑏in), 𝑎,𝑔 ∈ ℝ𝐵

batch×𝑇×𝐷, (10) 𝑎̄ = GELU(𝑎), (11) 𝑠 = Mixer(𝑎̄) ∈ ℝ𝐵

batch×𝑇×𝐷, (12) 𝑦 = 𝑥 + ((𝑠 ⊙ 𝑔)𝑊out + 𝑏out). (13)

We use Layer Normalization (Ba et al., 2016) and the GELU nonlinearity (Hendrycks and Gimpel, 2016). Here 𝑊in ∈ ℝ𝐷×2𝐷 and 𝑊out ∈ ℝ𝐷×𝐷. The elementwise gate 𝑔 plays the usual role of gated MLP variants (Hua et al., 2022; Shazeer, 2020): it modulates the mixer output before the residual add.

Linear

Mixer

σ

Linear

Linear

| | | |
|---|---|---|
| | | |

LayerNorm

Figure 2: Sessa Layer.

The mixer maps 𝑎̄ ∈ ℝ𝐵

batch×𝑇×𝐷 to 𝑠 ∈ ℝ𝐵

- 3.2 Sessa mixer

batch×𝑇×𝐷. It uses two causal attention mechanisms: (i) a forward causal

attention that produces a forward signal 𝑓𝑡 ∈ ℝ𝐷, and (ii) a feedback attention that produces weights over the strict past, used inside a causal feedback solve.

Projections. At each time 𝑡, we form forward queries, keys, and values, as well as feedback queries and keys, using standard linear projections:

𝑞𝑡𝑓 = 𝑎̄𝑡𝑊𝑄𝑓, 𝑘𝑡𝑓 = 𝑎̄𝑡𝑊𝐾𝑓, 𝑣𝑡 = 𝑎̄𝑡𝑊𝑉 , 𝑞𝑡𝑏 = 𝑎̄𝑡𝑊𝑄𝑏, 𝑘𝑡𝑏 = 𝑎̄𝑡𝑊𝐾𝑏, (14)

where 𝑞𝑓,𝑘𝑓,𝑞𝑏,𝑘𝑏 ∈ ℝ𝑑𝑘 and 𝑣𝑡 ∈ ℝ𝐷. We apply RoPE to the forward pair (𝑞𝑓,𝑘𝑓). We use rotary position embeddings in the forward branch (Su et al., 2021).

Forward attention. Define causal weights over 𝑗 ≤ 𝑡:

𝛼fwd𝑡,𝑗 = softmax0≤𝑗≤𝑡(𝜎𝑘⟨RoPE(𝑞𝑡𝑓),RoPE(𝑘𝑗𝑓)⟩), (15) and the forward signal

𝑡

𝛼fwd𝑡,𝑗 𝑣𝑗 ∈ ℝ𝐷. (16) This is a one-hop mixture of values (𝑣𝑗)𝑗≤𝑡 over a finite visible set. Feedback attention. Define feedback weights over the strict past 𝑗 < 𝑡:

𝑓𝑡 =

∑

𝑗=0

softmax0≤𝑗≤𝑡−1(𝜎𝑘⟨𝑞𝑡𝑏,𝑘𝑗𝑏⟩), 𝑡 ≥ 1, 𝑗 < 𝑡, 0, 𝑗 ≥ 𝑡,

𝛼fb𝑡,𝑗 = {

𝛼fb0,𝑗 = 0 ∀𝑗. (17)

Feedback gain. We modulate the feedback with a scalar gain 𝛾𝑡 ∈ (−1,1):

𝛾𝑡 = tanh(⟨𝑎̄𝑡,𝑤𝛾⟩ + 𝑏𝛾). (18)

The bound controls feedback magnitude: since 𝛼fb𝑡,⋅ is a convex distribution over 𝑗 < 𝑡, the feedback term is a convex combination of past states scaled by |𝛾𝑡| < 1.

[𝐵fb]𝑡,𝑗 = 𝛾𝑡 𝛼fb𝑡,𝑗, [𝐵fb]𝑡,𝑗 = 0 for 𝑗 ≥ 𝑡. (19)

Feedback routing matrix.

Scalar routing and feature-wise solve. Here 𝐵fb is a scalar strictly lower-triangular routing matrix (each [𝐵fb]𝑡,𝑗 ∈ ℝ). The solve (𝐼 − 𝐵fb)𝑠 = 𝑓 is applied independently to each feature dimension of 𝑠,𝑓 ∈ ℝ𝑇×𝐷: for every 𝑑 ∈ {1,…,𝐷},

(𝐼 − 𝐵fb)𝑠∶,𝑑 = 𝑓∶,𝑑, In vectorized form,

(𝐼𝐷 ⊗ (𝐼 − 𝐵fb))vec(𝑠) = vec(𝑓).

The resulting recurrence (22) therefore uses scalar–vector multiplication ([𝐵fb]𝑡,𝑗 𝑠𝑗 with [𝐵fb]𝑡,𝑗 ∈ ℝ and 𝑠𝑗 ∈ ℝ𝐷).

Lower-triangular solve. The mixer output 𝑠 ∈ ℝ𝑇×𝐷 is the unique solution of

(𝐼 − 𝐵fb)𝑠 = 𝑓 (20)

which is a unit-lower-triangular solve with 𝐷 right-hand sides. This can be implemented with optimized triangularsolve routines (e.g., batched solve_triangular/TRSM kernels), avoiding explicit formation of (𝐼 −𝐵fb)−1. Thus, in the dense full-prefix formulation, the mixer remains quadratic in 𝑇. Equivalently, forward substitution gives the explicit recurrence

𝑠0 0, (21)

𝑡−1

𝑡−1

𝑠𝑡 = 𝑓𝑡 +

∑

[𝐵fb]𝑡,𝑗 𝑠𝑗 = 𝑓𝑡 + 𝛾𝑡

∑

𝛼fb𝑡,𝑗 𝑠𝑗, 𝑡 ≥ 1. (22)

𝑗=0

𝑗=0

- Remark 3.1 (Multi-hop routing view: exact on finite horizons). Since 𝐵fb is strictly lower-triangular on a finite horizon 𝑇, it is nilpotent (𝐵fb𝑇 = 0) and therefore

𝑇−1

𝑇−1

(𝐼 − 𝐵fb)−1 =

∑

𝐵fb𝑘 and hence 𝑠 =

∑

𝐵fb𝑘 𝑓.

𝑘=0

𝑘=0

The term 𝐵fb𝑘 𝑓 aggregates contributions that traverse 𝑘 internal routing steps through the feedback operator. Thus, unlike self-attention’s one-hop read, the solve realizes multi-hop routing, which can produce the heavy-tail

influence regimes analyzed in Section 4.2.

RoPE in the forward path. In the forward attention (16) we apply RoPE to (𝑞𝑓,𝑘𝑓), following common practice in decoder-only Transformers (Touvron et al., 2023; Black et al., 2022). This injects relative positional information into the attention logits while preserving causal masking.

- 3.3 Positional encoding

No positional encoding in feedback. We do not apply RoPE, or any other positional encoding, to the feedback attention (17). The feedback path already induces an absolute time direction: the strictly lower-triangular

feedback operator (19) and the causal solve (20) correspond to a forward substitution recurrence (22), whose output at time 𝑡 depends on an iterated aggregation of the strict past. This temporal asymmetry can generate position-dependent signals even when the mixer input is time-constant.

Corollary I.8, proved in Appendix I.5, shows that a single Sessa block can produce a deterministic, positiondependent additive offset: there exist parameters and vectors (𝑝𝑡)𝑇−1𝑡=0 ⊂ ℝ𝐷 such that for all inputs 𝑥 in any fixed compact set 𝒟 ⊂ ℝ𝐵

batch×𝑇×𝐷,

𝑦𝑡 = 𝑥𝑡 + 𝑝𝑡, 𝑡 = 0,…,𝑇 − 1. Moreover, these offsets can be chosen separated on 𝒟 in the following sense: there exist a unit direction 𝑢 ∈ ℝ𝐷 and a scale 𝜆 > 0 such that 𝑝𝑡 = 𝑐𝑡(𝜆𝑢) with 𝑐𝑡 pairwise distinct and the scalar ranges {⟨𝑥𝑡 + 𝑝𝑡,𝑢⟩ ∶ 𝑥 ∈ 𝒟} are pairwise disjoint over 𝑡. By Corollary 4.13, the position index 𝑡 is recoverable by a continuous token-wise map on the set of shifted tokens, so the feedback mechanism can supply an absolute positional signal internally.

### 4 Theory

This section establishes four properties of Sessa:

- (i) stability of the feedback solve,
- (ii) long-range memory, including flexible selective retrieval,
- (iii) internal positional encoding,
- (iv) universal approximation.

- Remark 4.1 (LayerNorm). All stability and Jacobian statements in this section are stated for the formulation with Norm = Id. For the pre-norm LayerNorm extension relevant to universal approximation, we assume an explicit 𝜀 > 0 and use the corresponding Lipschitz bounds for the normalization map; see Appendix J.

- 4.1 Stability of the feedback solve We isolate the operation in Sessa that induces multi-hop behavior: the causal lower-triangular solve

(𝐼 − 𝐵fb(𝑥))𝑠 = 𝑓(𝑥), [𝐵fb]𝑡,𝑗(𝑥) = 𝛾𝑡(𝑥)𝛼fb𝑡,𝑗(𝑥), [𝐵fb]𝑡,𝑗(𝑥) = 0 for 𝑗 ≥ 𝑡, (23)

where 𝛼fb𝑡,⋅(𝑥) is a convex distribution over the strict past, 𝑗 < 𝑡, produced by the feedback attention, and 𝛾𝑡(𝑥) ∈ (−1,1) is a bounded scalar gain. The quantity 𝑓(𝑥) is the forward aggregation defined in Section 3.

Scalar feedback matrix Throughout the stability analysis, 𝐵fb(𝑥) ∈ ℝ𝑇×𝑇 is scalar-valued: each entry [𝐵fb]𝑡,𝑗(𝑥) ∈ ℝ. The solve acts feature-wise on 𝑠,𝑓 ∈ ℝ𝑇×𝑟. In vectorized form, (𝐼𝑟 ⊗ (𝐼 − 𝐵fb))vec(𝑠) = vec(𝑓).

Norms For a finite or infinite token sequence 𝑢 = (𝑢𝑡) with 𝑢𝑡 ∈ ℝ𝑟, define

‖𝑢‖∞,2 ∶= sup

‖𝑢𝑡‖2,

𝑡

and for a finite tensor 𝑈 ∈ ℝ𝑇×𝑟, define ‖𝑈‖∞,2 ∶= max0≤𝑡≤𝑇−1 ‖𝑈𝑡‖2. Assumption 1 (Uniform row contraction on the feedback margin). For every radius 𝑅 ≥ 0 there exists 𝜌(𝑅) ∈ [0,1) such that for all inputs 𝑥 with ‖𝑥‖∞,2 ≤ 𝑅,

|𝛾𝑡(𝑥)| ≤ 𝜌(𝑅) < 1. (24)

𝑡

sup

Since each 𝛼fb𝑡,⋅(𝑥) is a convex distribution over 𝑗 < 𝑡, Assumption 1 implies the row-sum bound sup

∑

|[𝐵fb]𝑡,𝑗(𝑥)| ≤ 𝜌(𝑅) < 1. (25)

𝑡≥1

𝑗<𝑡

- Lemma 4.2 (Causal lower-triangular solve is bounded on ℓ∞). Let 𝐵fb be strictly lower-triangular, possibly on an

infinite horizon, and define (𝐵fb𝑠)𝑡 ∶= ∑𝑗<𝑡 [𝐵fb]𝑡,𝑗𝑠𝑗. If sup𝑡 ∑𝑗<𝑡 |[𝐵fb]𝑡,𝑗| ≤ 𝜌 < 1, then for every 𝑓 ∈ ℓ∞(ℕ,ℝ𝑟) there exists a unique 𝑠 ∈ ℓ∞(ℕ,ℝ𝑟) solving (𝐼 − 𝐵fb)𝑠 = 𝑓, and

1 1 − 𝜌

‖𝑠‖∞,2 ≤

‖𝑓‖∞,2.

Proof sketch. Forward substitution gives existence and uniqueness. The bound follows by a standard induction on the partial maxima max𝑘≤𝑡 ‖𝑠𝑘‖2 using the row-sum estimate. See Appendix D.4.

- Proposition 2 (One-block stability bound). Fix a Sessa block 𝐺 acting on finite or infinite sequences with the feedback solve (23). Assume moreover that all tokenwise aﬀine maps appearing in the block (in particular, the output projection and the residual aﬀine terms) are fixed and have finite operator norms and finite bias magnitudes. Assume that for every 𝑅 ≥ 0 there exist finite constants 𝐹𝑅,𝐺𝑅 < ∞ such that on the ball ‖𝑥‖∞,2 ≤ 𝑅,

| |
|---|

|𝛾𝑡(𝑥)| ≤ 𝜌(𝑅) < 1,

‖𝑓(𝑥)‖∞,2 ≤ 𝐹𝑅, ‖𝑔(𝑥)‖∞,2 ≤ 𝐺𝑅, sup

𝑡

Here 𝑔(𝑥) denotes the tokenwise gate, the Hadamard multiplier applied to 𝑠 before the output projection. Then there exists 𝐶𝑅 < ∞ such that ‖𝐺(𝑥)‖∞,2 ≤ 𝐶𝑅 for all ‖𝑥‖∞,2 ≤ 𝑅. In particular, 𝐺 is BIBO-stable on ℓ∞(ℕ,ℝ𝐷).

Proof sketch. By Lemma 4.2 and (25), ‖𝑠‖∞,2 ≤ (1 − 𝜌(𝑅))−1‖𝑓‖∞,2. Then ‖𝑠 ⊙ 𝑔‖∞,2 ≤ ‖𝑠‖∞,2‖𝑔‖∞,2. Since bounded tokenwise aﬀine maps send bounded sets to bounded sets, the output projection together with the residual aﬀine terms yields a ball-to-ball bound for 𝐺. Appendix Proposition 25 strengthens this by giving an explicit ball-to-ball constant in terms of matrix/operator norms and bias magnitudes; see Appendix D.

| |
|---|

#### 4.2 Long-range memory

We compare long-range memory through Jacobian-based diagnostics that separate the memory mechanism from routing adaptation. Let 𝑦 = 𝐺(𝑥) denote the output of a causal mixer or block applied to an input token sequence 𝑥 = (𝑥0,…,𝑥𝑇−1), and fix a source position 𝜏 ≤ 𝑡 with lag

ℓ ∶= 𝑡 − 𝜏. Our analysis uses three related diagnostics.

##### Diagnostics.

- (i) Fixed-routing influence Jacobians. We first freeze a realized routing pattern and differentiate only the induced linear map from an injected stream to the output. This yields, for example,

𝐽attn =

𝜕𝑦 𝜕𝑣

∣

𝛼fwd

, 𝐽sessa =

𝜕𝑠 𝜕𝑓

∣

𝐵fb

,

and the corresponding SSM impulse Jacobian 𝐽ssm induced by a realized sequence (𝐴ssm,𝑡,𝐵ssm,𝑡,𝐶ssm,𝑡). These quantities isolate the memory mechanism under a common realized routing regime.

- (ii) End-to-end block Jacobians. We then return to the full input-dependent block and measure the actual

sensitivity of output token 𝑦𝑡 to a past input token 𝑥𝜏:

𝜕𝑦𝑡(𝑥) 𝜕𝑥𝜏

𝐽𝑡,𝜏e2e(𝑥) ∶=

.

Unlike the fixed-routing Jacobians, these derivatives include both transport through the memory mechanism and the dependence of the routing coeﬀicients on the input. They are the relevant one-block quantities for comparing diffuse attention, failed-freeze-time Mamba, and Sessa under smooth-routing assumptions.

- (iii) Scalar transport scores for deep retrieval. For selective retrieval we extract scalar scores from deep end-toend Jacobians. For a depth-𝑁layer stack with hidden states

layer), we write

ℎ(0) = 𝑥, ℎ(1),…,ℎ(𝑁

𝜕ℎ(𝑁𝑡 layer)(𝑥) 𝜕ℎ(0)𝜏 (𝑥)

.

𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥) ∶=

Later we evaluate these blocks against source and target probes to obtain scalar transport scores, written generically as S, which are the quantities used in the selective-retrieval theorem.

These diagnostics play complementary roles. Fixed-routing Jacobians expose the structural difference between one-hop direct read, chain-structured feedback, and Sessa’s many-path feedback solve. End-to-end block Jacobians capture the actual behavior of the nonlinear input-dependent block. Scalar transport scores are needed for the positive retrieval statements, since they let us compare source and distractor influence after composing end-to-end Jacobians across layers.

All decay statements in this subsection are expressed in the lag ℓ = 𝑡 − 𝜏, not in the context length 𝑇. The key structural difference is that, for Sessa, the fixed-routing solve

(𝐼 − 𝐵fb)−1

aggregates contributions over multiple hop counts and, in dense regimes, over many temporal paths. This accumulation across hop counts and paths is the mechanism behind the polynomial tail analyzed below.

##### 4.2.1 Fixed-routing Jacobians

We begin with realized routing patterns and isolate the induced memory operators. Worst-case comparisons over all inputs and parameters are uninformative, since any model can suppress a token. Instead, we compare the architectures within common diffuse-weight regimes by studying the corresponding fixed-routing influence operators.

Attention value Jacobian For causal self-attention, for a given set of attention weights 𝛼fwd𝑡,𝜏 , the map from values to output is linear:

𝑦𝑡 = ∑ 𝜏≤𝑡

𝛼fwd𝑡,𝜏 𝑣𝜏

𝜕𝑦𝑡 𝜕𝑣𝜏

We define the value influence Jacobian

𝐽𝑡,𝜏attn ∶=

∣

= 𝛼fwd𝑡,𝜏 𝐼𝐷. (26)

𝛼fwd

Solve Jacobian In Sessa, for a given feedback matrix 𝐵fb, i.e., a given routing pattern inside the loop, the lower-triangular solve

(𝐼 − 𝐵fb)𝑠 = 𝑓

is linear in 𝑓. We define the solve influence Jacobian

𝜕𝑠 𝜕𝑓

∣

= (𝐼 − 𝐵fb)−1, 𝐽𝑡,𝜏sessa = [(𝐼 − 𝐵fb)−1]𝑡,𝜏. (27)

𝐽sessa ∶=

𝐵fb

Because 𝐵fb is scalar-valued, the solve acts identically on each feature dimension; equivalently, if 𝑓𝑡,𝑠𝑡 ∈ ℝ𝑑𝑓, the full feature-block Jacobian is

𝐽𝑡,𝜏sessa𝐼𝑑

.

𝑓

SSM impulse Jacobian For a feedback recurrence ℎ𝑡 = 𝐴ssm,𝑡ℎ𝑡−1 + 𝐵ssm,𝑡𝑢𝑡, 𝑦𝑡 = 𝐶ssm,𝑡ℎ𝑡, given a realized sequence of transitions (𝐴ssm,𝑡,𝐵ssm,𝑡,𝐶ssm,𝑡), the impulse influence from 𝑢𝜏 to 𝑦𝑡 is

𝑡

𝐽𝑡,𝜏ssm ∶= 𝐶ssm,𝑡(

∏

𝐴ssm,𝑟)𝐵ssm,𝜏, 0 ≤ 𝜏 ≤ 𝑡. (28)

𝑟=𝜏+1

Convention: time-ordered product. We interpret the matrix product in (28) as the left-to-right time-unrolling consistent with the recurrence ℎ𝑡 = 𝐴ssm,𝑡ℎ𝑡−1 + ⋯:

𝑡

𝐴ssm,𝑟 ∶= 𝐴ssm,𝑡𝐴ssm,𝑡−1 ⋯𝐴ssm,𝜏+1.

∏

𝑟=𝜏+1

Equivalently, the product is time-ordered with later-time factors on the left. For the empty product we use

𝑡

∏

(⋅) ∶= 𝐼,

𝑟=𝑡+1

so that the definition also covers the case 𝑡 = 𝜏. These Jacobians isolate the memory mechanism under a common routing regime.

- Definition 3 (End-to-end block Jacobian). Let 𝑦 = 𝐺(𝑥) denote the output of a single mixer/block 𝐺 applied to an input token sequence 𝑥 ∈ (ℝ𝐷)𝑇. We define the end-to-end Jacobian blocks by

𝐽𝑡,𝜏e2e(𝑥) ∶=

𝜕𝑦𝑡(𝑥) 𝜕𝑥𝜏

∈ ℝ𝐷×𝐷.

For 𝜏 < 𝑡, 𝐽𝑡,𝜏e2e(𝑥) measures long-range influence without freezing routing.

- Definition 4 (Diffuse attention regime). We say that an attention mechanism is in a diffuse, low-separation regime on a horizon 𝑇 if, for each 𝑡, its pre-softmax logits ℶ𝑡,𝑗 over the visible set satisfy a bounded spread

- 4.2.2 End-to-end Jacobians

ℶ𝑡,𝑗 ≤ Δ for some finite Δ,

ℶ𝑡,𝑗 − min 𝑗∈𝒲𝑡

𝑗∈𝒲𝑡

max

uniformly over the inputs under consideration. In this regime, softmax weights are near-uniform: Appendix Lemma C.1 implies that for full-prefix attention with |𝒲𝑡| = 𝑡 + 1,

𝛼fwd𝑡,𝑗 = Θ(1/|𝒲𝑡|). In particular, for full-prefix causal attention one has

𝒲𝑡 = {0,…,𝑡}, |𝒲𝑡| = 𝑡 + 1,

𝒲𝑡 = {0,…,𝑡 − 1}, |𝒲𝑡| = 𝑡 for 𝑡 ≥ 1.

whereas for strictly-lower attention one has

We state diffuse bounds in terms of the visible-set size |𝒲𝑡| to cover full-prefix and strict-past attention uniformly.

We assume diffuse attention rows 𝛼fwd𝑡,𝑗 ≤ 𝑐2/|𝒲𝑡| (Definition 4), together with the following smooth-routing bound on the input set of interest:

𝜕𝛼fwd𝑡,𝑗 (𝑥) 𝜕𝑥𝜏

𝐿𝛼 |𝒲𝑡|

∑

∥

∥

≤

, 𝜏 < 𝑡. (29)

2

𝑗∈𝒲𝑡

Appendix B derives this from standard softmax calculus under mild logit-sensitivity control.

𝛼fwd𝑡,⋅ (𝑥) = softmax(ℶ𝑡,0(𝑥),…,ℶ𝑡,𝑡(𝑥)) with logits ℶ𝑡,𝑗(𝑥) = ⟨𝑞(𝑥𝑡), 𝑘(𝑥𝑗)⟩ where 𝑞,𝑘 are tokenwise maps. Then for every 𝜏 < 𝑡,

- Lemma 4.3 (Smooth-routing for standard causal attention). Assume a single-head causal attention row is

𝜕𝛼fwd𝑡,𝑗 (𝑥) 𝜕𝑥𝜏

𝜕ℶ𝑡,𝜏(𝑥) 𝜕𝑥𝜏

∑

∥

∥

∥

≤ 2𝛼fwd𝑡,𝜏 (𝑥)∥

.

2

2

𝑗≤𝑡

In particular, if ‖𝜕ℶ𝑡,𝜏/𝜕𝑥𝜏‖2 ≤ 𝐿ℶ on 𝒳𝑅, then

𝜕𝛼fwd𝑡,𝑗 (𝑥) 𝜕𝑥𝜏

1 |𝒲𝑡|

≤ 2𝐿ℶ 𝛼fwd𝑡,𝜏 (𝑥) ≲

∑

∥

∥

2

𝑗≤𝑡

in the diffuse regime of Definition 4. For full-prefix attention one has |𝒲𝑡| = 𝑡 + 1. Full proof in Appendix C.1.

- 4.2.3 Exponential forgetting in LTI systems Consider a finite-dimensional linear time-invariant feedback system in state-space form:

ℎ𝑡 = 𝐴ssmℎ𝑡−1 + 𝐵ssm𝑢𝑡, 𝑦𝑡 = 𝐶ssmℎ𝑡, (30)

with constant matrices (𝐴ssm,𝐵ssm,𝐶ssm). Under an impulse input at time 𝜏, i.e. 𝑢𝜏 ≠ 0 and 𝑢𝑡 = 0 for 𝑡 ≠ 𝜏, the contribution to 𝑦𝑡 is mediated by 𝐴𝑡−𝜏ssm = 𝐴ℓssm. Proposition 3 (Exponential decay in BIBO-stable LTI feedback systems). Assume (30) is BIBO-stable. Then there exist constants 𝑐 > 0 and 𝜅 ∈ (0,1) such that for all lags ℓ ≥ 0,

‖𝐶ssm𝐴ℓssm𝐵ssm‖ ≤ 𝑐 𝜅ℓ.

Equivalently, the impulse response and long-range influence mediated by the state transition decay exponentially in the lag ℓ.

Proof sketch. BIBO stability implies internal stability of any minimal controllable and observable realization, hence 𝜌spec(𝐴ssm,co) < 1 (Dahleh et al., 2011c). Therefore ‖𝐴ℓssm,co‖ ≤ 𝑐𝜅ℓ and ‖𝐶ssm𝐴ℓssm𝐵ssm‖ = ‖𝐶ssm,co𝐴ℓssm,co𝐵ssm,co‖ ≤ 𝑐′𝜅ℓ. Proof in Appendix C.3.

| |
|---|

- 4.2.4 Exponential forgetting in Mamba

Mamba-style layers fit Definition 2 as feedback systems. Their update maps 𝐴ssm,𝑡(𝑥0∶𝑡),𝐵ssm,𝑡(𝑥0∶𝑡),𝐶ssm,𝑡(𝑥0∶𝑡) depend on the input.

Convention: discrete scan coeﬀicients In what follows, 𝐴ssm,𝑡,𝐵ssm,𝑡,𝐶ssm,𝑡 denote the discrete-time scan coeﬀicients actually used in the recurrence ℎ𝑡 = 𝐴ssm,𝑡ℎ𝑡−1 + 𝐵ssm,𝑡𝑢𝑡 after discretization, such as ZOH, unless

stated otherwise.

Exponential forgetting is not automatic for general input-dependent feedback systems. Section 4.2.6 gives a counterexample in a diffuse feedback-routing regime. For Mamba, the relevant condition is failed freeze time: the model cannot sustain a long interval with Δ𝑡 ≈ 0.

Accumulated discretization time In Mamba’s standard ZOH-diagonal parameterization, long-range influence is controlled by the accumulated discretization time

𝑡

∑

Δ𝑟,

𝑟=𝜏+1

since the transition product contains factors of the form

𝑡

exp( − 𝑎𝑛

∑

Δ𝑟).

𝑟=𝜏+1

Accordingly, failed freeze time converts control in accumulated discretization time into exponential decay in the lag.

- Proposition 4 (Mamba end-to-end Jacobian bound). Consider a Mamba block with state ℎ𝑡 ∈ ℝ𝑑

state and output 𝑦𝑡 ∈ ℝ𝐷:

ℎ−1 = 0, ℎ𝑡 = 𝐴ssm,𝑡(𝑥𝑡)ℎ𝑡−1 + 𝐺ssm,𝑡(𝑥𝑡)𝐵̃ssm,𝑡(𝑥𝑡)𝑢𝑡(𝑥𝑡), 𝑦𝑡 = 𝐶ssm,𝑡(𝑥𝑡)ℎ𝑡, where the parametrization is local and ZOH-diagonal: for each mode 𝑛,

1 − exp(−𝑎𝑛Δ𝑡(𝑥𝑡)) 𝑎𝑛

,

[𝐴ssm,𝑡(𝑥𝑡)]𝑛 = exp(−𝑎𝑛Δ𝑡(𝑥𝑡)), [𝐺ssm,𝑡(𝑥𝑡)]𝑛 =

𝑎𝑛 ≥ 𝜆 > 0 for all modes 𝑛.

with input-independent rates satisfying

Assume there exist constants 𝑈𝑅,𝐺max,𝐶𝑅,𝐿𝐴,𝐿𝐵,𝐿𝑢 < ∞ such that for all 𝑥 ∈ 𝒳𝑅 and all 𝑡, ‖𝑢𝑡(𝑥𝑡)‖ ≤ 𝑈𝑅, ‖𝐵̃ssm,𝑡(𝑥𝑡)‖ ≤ 𝐺max, ‖𝐶ssm,𝑡(𝑥𝑡)‖ ≤ 𝐶𝑅,

𝜕𝐵̃ssm,𝑡(𝑥𝑡) 𝜕𝑥𝑡

𝜕𝐴ssm,𝑡(𝑥𝑡) 𝜕𝑥𝑡

𝜕𝑢𝑡(𝑥𝑡) 𝜕𝑥𝑡

∥ ≤ 𝐿𝑢. For 𝜏 < 𝑡 with lag ℓ = 𝑡 − 𝜏, define

∥

∥ ≤ 𝐿𝐴, ∥

∥ ≤ 𝐿𝐵, ∥

𝑡

Π𝑡,ℓ(𝑥) ∶= exp( − 𝜆

∑

Δ𝑟(𝑥)).

𝑟=𝜏+1

Then for every 𝑥 ∈ 𝒳𝑅 and every 𝜏 < 𝑡,

𝜕𝑦𝑡(𝑥) 𝜕𝑥𝜏

∥

∥ ≤ 𝐶(𝑅)Π𝑡,ℓ(𝑥),

𝐶(𝑅) ∶= 𝐶𝑅 𝐽𝑅, with

where one may take

𝐿𝐴 𝜆

1 𝜆

𝐺max 𝑈𝑅 𝜆

(𝐿𝐵 𝑈𝑅 + 𝐺max 𝐿𝑢), 𝐻𝑅 ∶=√𝑑state

𝐽𝑅 ∶= 𝐿𝐴 𝐻𝑅 +

𝐺max𝑈𝑅 +

.

Proof sketch. Differentiate the ZOH recurrence. By locality, for 𝑡 > 𝜏 one has

𝜕ℎ𝑡 𝜕𝑥𝜏

𝜕ℎ𝑡−1 𝜕𝑥𝜏

= 𝐴ssm,𝑡(𝑥𝑡)

.

Thus the long-range dependence is controlled by the transition product. Lemma 4.4 yields the uniform state bound 𝐻𝑅, which controls the source-time injection derivative 𝜕ℎ𝜏/𝜕𝑥𝜏. Since each diagonal transition satisfies

𝑡

𝑡

∥

∏

𝐴ssm,𝑟(𝑥𝑟)∥ ≤ exp( − 𝜆

∑

Δ𝑟(𝑥)) = Π𝑡,ℓ(𝑥),

𝑟=𝜏+1

𝑟=𝜏+1

the displayed bound follows. Proof in Appendix C.4.

| |
|---|

ZOH discretization under freezing In Mamba, the discrete-time coeﬀicients arise from a stable continuoustime diagonal kernel via ZOH (Gu and Dao, 2024). For each mode with continuous parameter 𝐴 = −𝑎 with 𝑎 > 0 and step size Δ𝑡 ≥ 0,

1 − 𝑒−𝑎Δ𝑡 𝑎

𝐴𝑡̄ = 𝑒−𝑎Δ𝑡 ∈ [0,1], 𝐵̄𝑡 =

𝐵̃ssm,𝑡.

Here 𝐴ssm,𝑡 = 𝐴𝑡̄ and 𝐵̄𝑡𝑢𝑡 = 𝐺ssm,𝑡 𝐵̃ssm,𝑡𝑢𝑡. In particular, when “freezing time” with Δ𝑡 = 0 one has 𝐴𝑡̄ = 1 and 𝐵̄𝑡 = 0, so the update injects no new input while holding the state.

1 − 𝑒−𝑎Δ𝑡 𝑎

- Lemma 4.4 (Bounded state for ZOH-diagonal Mamba channels). Consider the scalar ZOH recurrence

ℎ−1 = 0, ℎ𝑡 = 𝑒−𝑎Δ𝑡 ℎ𝑡−1 +

𝑏𝑡, 𝑎 ≥ 𝑎min > 0, Δ𝑡 ≥ 0.

If |𝑏𝑡| ≤ 𝑀 for all 𝑡, then sup𝑡 |ℎ𝑡| ≤ 𝑀/𝑎min. More generally, sup𝑡 |ℎ𝑡| ≤ max{|ℎ−1|, sup𝑠 |𝑏𝑠|/𝑎min}.

𝑎 with 𝜃𝑡 ∶= 𝑒−𝑎Δ𝑡 ∈ [0,1]. Thus ℎ𝑡 is a convex combination of ℎ𝑡−1 and 𝑏𝑡

Proof sketch. Write ℎ𝑡 = 𝜃𝑡ℎ𝑡−1 + (1 − 𝜃𝑡) 𝑏

𝑎 , yielding |ℎ𝑡| ≤ max{|ℎ𝑡−1|, |𝑏𝑡|/𝑎}. Since 𝑎 ≥ 𝑎min, we have |𝑏𝑡|/𝑎 ≤ |𝑏𝑡|/𝑎min, and the claim follows by induction. Proof in Appendix C.5.

𝑡

Failure of freeze time Mamba may slow decay by keeping 𝜆𝑛Δ𝑡 ≈ 0 over selected steps. We rule out this behavior by assuming that accumulated discretization time grows linearly on every relevant interval.

| |
|---|

- Proposition 5 (Failed freeze time yields exponential forgetting). Consider a single-mode diagonal selective SSM channel with memory factor

𝑡

𝑡

Π𝑡,ℓ ∶=

∏

exp(−𝜆Δ𝑟) = exp( − 𝜆

∑

Δ𝑟), 𝜆 > 0.

𝑟=𝑡−ℓ+1

𝑟=𝑡−ℓ+1

Assume there exists 𝑐Δ > 0 such that for every relevant pair 𝜏 < 𝑡,

𝑡

∑

Δ𝑟 ≥ 𝑐Δ(𝑡 − 𝜏).

𝑟=𝜏+1

Π𝑡,ℓ ≤ exp( − 𝜆𝑐Δℓ).

Then

Equivalently, once freeze time cannot be maintained over a long interval, the memory factor is exponentially small in the lag.

𝑡

Π𝑡,ℓ = exp( − 𝜆

∑

Δ𝑟)

Proof sketch. This is immediate from

𝑟=𝜏+1

and the assumed linear lower bound on the accumulated discretization time. Proof in Appendix C.7.

| |
|---|

For causal self-attention, the direct contribution of token 𝜏 to 𝑦𝑡 is the one-hop weight 𝛼fwd𝑡,𝜏 . In diffuse regimes this is 𝑂(1/|𝒲𝑡|), hence 𝑂(1/(𝑡 + 1)) for full-prefix attention. For very old tokens with 𝜏 = 𝑂(1) and 𝑡 ≍ ℓ, this becomes 𝑂(1/ℓ). This is a dilution phenomenon controlled primarily by the query time 𝑡, rather than a multi-hop forgetting mechanism.

##### 4.2.5 Attention dilution

###### 4.2.6 Polynomial decay in Sessa We formalize a regime in which the Sessa feedback solve yields polynomial decay in the lag ℓ.

Scalar recursion Let (𝛾𝑡)𝑡≥0 be scalars and let (𝛼fb𝑡,𝑗)𝑡≥1,0≤𝑗<𝑡 satisfy 𝛼fb𝑡,𝑗 ≥ 0 and ∑𝑗<𝑡 𝛼fb𝑡,𝑗 ≤ 1. Given a forward sequence (𝑓𝑡), define

𝑡−1

𝑦0 = 𝑓0, 𝑦𝑡 = 𝑓𝑡 + 𝛾𝑡

∑

𝛼fb𝑡,𝑗𝑦𝑗, 𝑡 ≥ 1. (31)

𝑗=0

For an impulse input at time 𝜏, set 𝑓𝜏 = 1 and 𝑓𝑡 = 0 for 𝑡 ≠ 𝜏. This yields an influence profile 𝑦𝑡 supported on 𝑡 ≥ 𝜏; the relevant memory variable is the lag ℓ = 𝑡 − 𝜏.

- Assumption 6 (Diffuse feedback routing envelope). There exists 𝑐2 ∈ (0,∞) such that for all 𝑡 ≥ 1 and all 0 ≤ 𝑗 < 𝑡,

𝛼fb𝑡,𝑗 ≤

𝑐2 𝑡

. (32)

- Assumption 7 (Bounded feedback gain). There exists 𝛾max ∈ [0,1) such that |𝛾𝑡| ≤ 𝛾max for all 𝑡 ≥ 0. Define 𝛽tail ∶= 1 − 𝛾max𝑐2 and assume 𝛾max𝑐2 < 1, so 𝛽tail ∈ (0,1].

Theorem 8 (Polynomial decay of impulse influence). Under Assumptions 6–7 and 𝛽tail ∶= 1 − 𝛾max𝑐2 ∈ (0,1], the impulse influence induced by (31) satisfies, for all lags ℓ ≥ 1,

. uniformly over the impulse time 𝜏 (when the same constants apply).

|𝑦𝜏+ℓ| ≤ 𝐶 ℓ−𝛽

, for instance 𝐶 = (1 − 𝛽tail)𝑒1−𝛽

tail

tail

Proof sketch. Shift the recursion to start at 𝜏 and apply a comparison argument controlling partial sums by a harmonic-growth recursion, yielding ℓ−𝛽

tail. For 0 < 𝛽tail < 1, the full proof appears in Appendix E, Corollary E.4 with 𝑗 = 𝜏. The endpoint case 𝛽tail = 1 corresponds to 𝜂 = 𝛾max𝑐2 = 0, hence 𝛾𝑡 = 0 for all 𝑡 and therefore 𝑦𝜏+ℓ = 0 for all ℓ ≥ 1; see also Remark E.2.

| |
|---|

Remark 4.5 (Subcriticality). Whenever we refer in prose to a polynomial tail induced by diffuse feedback routing, this always means the subcritical regime

𝑐2 𝑡

, |𝛾𝑡| ≤ 𝛾max, 𝛾max𝑐2 < 1. Equivalently,

𝛼fb𝑡,𝑗 ≤

𝛽tail ∶= 1 − 𝛾max𝑐2 ∈ (0,1].

The nontrivial heavy-tail case is 0 < 𝛽tail < 1. The endpoint 𝛽tail = 1 corresponds to 𝛾max𝑐2 = 0, in which case the post-source impulse is identically zero; see Remark E.2. Thus bounded gains alone do not suﬀice: the strict

subcriticality condition 𝛾max𝑐2 < 1 is essential in every use of Theorem 8. Comparison and sharpness. Under the subcritical diffuse-routing assumptions above, Sessa yields a polynomial tail ℓ−𝛽

time Mamba (Section 4.2.4). The exponent is sharp: in the explicit uniform-routing regime 𝛼fb𝑡,𝑗 = 1𝑡1[𝑗 < 𝑡] with constant 𝛾 ∈ (0,1), Appendix Corollary F.2 gives the closed form

tail, unlike the exponential forgetting of stable LTI feedback systems (Proposition 3) and failed-freeze-

Γ(𝜏 + 1) Γ(𝜏 + 1 + 𝛾)

Γ(𝜏 + ℓ + 𝛾) Γ(𝜏 + ℓ + 1)

𝑦𝜏+ℓ = 𝛾

⋅

,

and hence 𝑦𝜏+ℓ = Θ𝜏(ℓ−𝛽

) with 𝛽tail = 1 − 𝛾 for every fixed 𝜏. Appendix Corollary F.3 further gives a uniform two-sided envelope on every bounded source family for a single layer. These one-layer statements are distinct from the deep selective-retrieval theorem below, which uses a different multi-layer construction.

tail

Connection to attention dilution Diffuse attention in a one-hop mixer yields per-token weights of order 𝑂(1/𝑡) and, for very old tokens, 𝑂(1/ℓ). In contrast, under the diffuse-routing assumptions of Theorem 8, Sessa yields a tail 𝑂(ℓ−𝛽

) with 𝛽tail < 1, which is asymptotically slower than 1/ℓ and therefore can mitigate dilution by sustaining longer-range influence through the stateful feedback channel while remaining BIBO-stable under Section 4.1.

tail

- Proposition 9 (Decay envelopes in the diffuse regime). Fix a horizon 𝑇 and consider the fixed-routing influence Jacobians of Section 4.2.1. The three items below are stated under the mechanism-specific assumptions introduced above.

- (i) Transformer. In the diffuse regime with full-prefix visibility, the value Jacobian satisfies

‖𝐽𝑡,𝜏attn‖ = 𝛼fwd𝑡,𝜏 = Θ(

1 𝑡 + 1

) (𝜏 ≤ 𝑡),

and in particular for a fixed old source 𝜏 = 𝑂(1) and lag ℓ = 𝑡 − 𝜏,

‖𝐽𝜏+ℓ,𝜏attn ‖ = Θ(1/ℓ).

- (ii) Mamba. Assume the realized recurrence has diagonal transitions

𝐴ssm,𝑟 = diag(exp(−𝑎𝑛Δ𝑟)), 𝑎𝑛 ≥ 𝜆 > 0, and bounded input/output factors sup𝑟 ‖𝐵ssm,𝑟‖,sup𝑟 ‖𝐶ssm,𝑟‖ < ∞. If, on the region of interest,

𝑡

∑

𝑟=𝜏+1

Δ𝑟 ≥ 𝑐Δ(𝑡 − 𝜏),

then the impulse Jacobian obeys

‖𝐽𝑡,𝜏ssm‖ ≤ 𝑐 exp( − 𝜆𝑐Δ(𝑡 − 𝜏)) = 𝑐 𝑒−𝜆𝑐Δℓ.

This expresses exponential forgetting under failed freeze time: the model cannot maintain a long preserve corridor, so accumulated discretization time grows linearly in the lag.

- (iii) Sessa. Under the hypotheses of Theorem 8, the solve Jacobian column corresponding to an impulse in 𝑓 obeys the polynomial envelope

|𝐽𝜏+ℓ,𝜏sessa | ≤ 𝐶 ℓ−𝛽

, 𝛽tail ∈ (0,1],

tail

⎧ ⎨{⎩

- 0, 𝑡 = 0, 𝛾 𝑡

- 1[𝑗 < 𝑡], 𝑡 ≥ 1,

as in Theorem 8. Moreover, in the explicit uniform-routing regime [𝐵fb]𝑡,𝑗 =

𝛾 ∈ (0,1) and 𝛽tail = 1 − 𝛾, this envelope is tight in the following qualified sense: for every fixed source position 𝜏,

with

), by Corollary F.2. Moreover, for every bounded source family 0 ≤ 𝜏 ≤ 𝜏max there exist constants 𝑐𝜏−

|𝐽𝜏+ℓ,𝜏sessa | = Θ𝜏(ℓ−𝛽

> 0 such that

,𝑐𝜏+

tail

𝑐𝜏−

ℓ−𝛽

≤ |𝐽𝜏+ℓ,𝜏sessa | ≤ 𝑐𝜏+

ℓ−𝛽

max

max

for all 0 ≤ 𝜏 ≤ 𝜏max and all ℓ ≥ 1, by Corollary F.3. In particular, the same two-sided bound holds uniformly on every fixed finite horizon.

tail

tail

max

max

- Proposition 10 (End-to-end decay envelopes). Fix a horizon 𝑇 and consider one-block end-to-end Jacobians. In item (i) we assume the diffuse smooth-routing regime of Section 4.2.2. Assume additionally that tokenwise maps are bounded and Lipschitz on the input set: ‖𝑣(𝑥𝑡)‖ ≤ 𝑉𝑅 and ‖𝜕𝑣(𝑥𝑡)/𝜕𝑥𝑡‖ ≤ 𝐿𝑣.

Proof in Appendix C.2.

- (i) Transformer. For 𝑦𝑡 = ∑𝑗≤𝑡 𝛼fwd𝑡,𝑗 (𝑥)𝑣(𝑥𝑗) and any 𝜏 < 𝑡,

∥

𝜕𝑦𝑡 𝜕𝑥𝜏

∥ ≤ 𝛼fwd𝑡,𝜏 𝐿𝑣 + 𝑉𝑅 ∑ 𝑗≤𝑡

∥

𝜕𝛼fwd𝑡,𝑗 𝜕𝑥𝜏

∥ ≲

1 𝑡 + 1

.

In particular, for a fixed old source 𝜏 = 𝑂(1) and lag ℓ = 𝑡 − 𝜏, one gets ‖𝐽𝜏+ℓ,𝜏e2e ‖ = 𝑂(1/ℓ).

- (ii) Mamba. Assume the block admits a local ZOH-diagonal parametrization as in Proposition 4. If, on the input set of interest, there exists 𝑐Δ > 0 such that for every 𝜏 < 𝑡,

𝑡

∑

𝑟=𝜏+1

Δ𝑟(𝑥) ≥ 𝑐Δ(𝑡 − 𝜏),

then Corollary 4.6 yields

∥

𝜕𝑦𝑡 𝜕𝑥𝜏

∥ ≤ 𝐶(𝑅)exp( − 𝜆𝑐Δ(𝑡 − 𝜏)) = 𝐶(𝑅)𝑒−𝜆𝑐Δℓ.

- (iii) Sessa. Assume additionally the hypotheses of Corollary B.7. Under the diffuse feedback routing assumptions of Appendix B,

𝜕𝑦𝑡 𝜕𝑥𝜏

∥

(1 + log(1 + ℓ)), 𝛽tail ∈ (0,1), via Corollary B.7.

∥ ≤ 𝐶 ℓ−𝛽

tail

Proof sketch. (i) Differentiate 𝑦𝑡 = ∑𝑗≤𝑡 𝛼fwd𝑡,𝑗 (𝑥)𝑣(𝑥𝑗): one term is controlled by 𝛼fwd𝑡,𝜏 𝐿𝑣 and the other by 𝑉𝑅 ∑𝑗∈𝒲

‖𝜕𝛼fwd𝑡,𝑗 /𝜕𝑥𝜏‖. Under the diffuse smooth-routing regime both are 𝑂(1/|𝒲𝑡|), hence 𝑂(1/(𝑡 + 1)) for full-prefix attention.

𝑡

- (ii) Combine Proposition 4 with the deterministic failed-freeze-time condition

𝑡

∑

𝑟=𝜏+1

Δ𝑟(𝑥) ≥ 𝑐Δ(𝑡 − 𝜏),

or equivalently use Corollary 4.6.

- (iii) This follows from Corollary B.7 under the additional Sessa assumptions stated in item (iii).

| |
|---|

- Corollary 4.6 (Failed freeze time implies exponential decay of Mamba end-to-end Jacobians). Under the hypotheses of Proposition 4, assume additionally the failed freeze-time condition of Proposition 5, namely that there

exists 𝑐Δ > 0 such that

𝑡

∑

Δ𝑟(𝑥) ≥ 𝑐Δ(𝑡 − 𝜏)

𝑟=𝜏+1

for every relevant pair 𝜏 < 𝑡 and every 𝑥 ∈ 𝒳𝑅. Then

𝜕𝑦𝑡(𝑥) 𝜕𝑥𝜏

∥

∥ ≤ 𝐶(𝑅)exp( − 𝜆𝑐Δ(𝑡 − 𝜏)).

Proof sketch. Combine Proposition 4 with Proposition 5.

| |
|---|

##### 4.2.7 Deep end-to-end bounds

The fixed-routing Jacobians remain useful as mechanism diagnostics, but deep architectural statements must be made for the end-to-end Jacobians

𝜕ℎ(𝑁𝑡 layer)(𝑥) 𝜕ℎ(0)𝜏 (𝑥)

𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥) ∶=

∈ ℝ𝐷×𝐷,

since these are the quantities that compose across layers by the chain rule. The next theorem gives the corresponding deep path-sum expansion.

- Theorem 11 (Deep end-to-end aggregation). Fix a depth 𝑁layer ≥ 1, a finite horizon 𝑇, and a compact input set 𝒳0. Let

layer−1)), 𝑛layer = 1,…,𝑁layer, where each block 𝐹𝑛

ℎ(0) = 𝑥 ∈ 𝒳0, ℎ(𝑛

layer) = 𝐹𝑛

(ℎ(𝑛

layer

layer−1 ∘ ⋯ ∘ 𝐹1(𝒳0). Assume that for each layer 𝑛layer there exist constants

𝒳𝑛

layer−1 ∶= 𝐹𝑛

is causal and continuously differentiable on the relevant compact set

layer

≥ 0, and a scalar lower-triangular kernel

≥ 0, 𝜆𝑛

𝑑𝑛

layer

layer

𝐾𝑛

∶ {(𝑡,𝜏) ∶ 0 ≤ 𝜏 < 𝑡 ≤ 𝑇 − 1} → [0,∞)

such that for every 𝑢 ∈ 𝒳𝑛

layer−1 and every 0 ≤ 𝜏 ≤ 𝑡 ≤ 𝑇 − 1,

layer

𝜕𝐹𝑛

layer,𝑡(𝑢) 𝜕𝑢𝜏

∥ ≤ 𝑑𝑛

∥

1[𝑡 = 𝜏] + 𝜆𝑛

𝐾𝑛

(𝑡,𝜏)1[𝜏 < 𝑡]. (33)

Then for every 𝑥 ∈ 𝒳0 and every 0 ≤ 𝜏 < 𝑡 ≤ 𝑇 − 1,

layer

layer

layer

⎛⎜ ⎝

𝑑𝑚⎞⎟

𝑁layer

∏

∑

∑

∥𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥)∥ ≤

⎠ ⋅ ∑

1≤𝑛layer,1<⋯<𝑛layer,𝑘≤𝑁layer

𝑘=1

𝑚∉{𝑛layer,1,…,𝑛layer,𝑘}

𝑘

∏

𝜆𝑛

𝐾𝑛

(𝑖𝑟,𝑖𝑟−1). (34)

𝜏=𝑖0<𝑖1<⋯<𝑖𝑘=𝑡

𝑟=1

layer,𝑟

layer,𝑟

The same expansion also gives the diagonal bound

𝑁layer

∥𝐽𝑡,𝑡e2e,(𝑁layer)(𝑥)∥ ≤

∏

𝑑𝑛

.

𝑛layer=1

layer

Proof sketch. This is a direct chain-rule expansion for the full block Jacobian. Proof in Appendix H.

| |
|---|

Thus deep long-range memory is controlled by the path sum induced by the one-block end-to-end Jacobian envelopes.

For the family-over-horizon comparison used below, one needs a horizon-uniform version of this calculus, i.e., bounds whose constants are independent of the context length 𝑇. The fixed-horizon model-class estimates and the abstract horizon-uniform lifting are recorded in Appendix H–H.5. Here we state only the resulting horizonuniform decay envelopes needed for the comparison-class impossibility argument.

- (i) Transformer. Assume that for each layer 𝑛layer there exists 𝑎𝑛

layer

> 0 such that

𝐾𝑛

layer

(𝑡,𝜏) ≤

𝑎𝑛

layer

𝑡 + 1

, 𝜏 < 𝑡.

Fix a bounded source family 0 ≤ 𝜏 ≤ 𝜏max. Then for every ℓ ≥ 1,

sup

𝑇≥𝜏max+ℓ+1

sup

0≤𝜏≤𝜏max

sup

𝑥∈𝒳(𝑇)0

∥𝐽𝜏+ℓ,𝜏e2e,(𝑁layer)(𝑥;𝑇)∥ ≲𝜏

max,𝑁layer

(log(1 + ℓ))𝑁

layer−1

1 + ℓ

.

In particular, the right-hand side tends to 0 as ℓ → ∞, so this is a genuine horizon-uniform asymptotic dilution law on bounded-source families.

- (ii) Mamba. Assume that for each layer 𝑛layer there exist 𝑎𝑛

layer

> 0 and 𝑐𝑛

layer

> 0 such that

𝐾𝑛

layer

(𝑡,𝜏) ≤ 𝑎𝑛

layer

𝑒−𝑐

𝑛layer(𝑡−𝜏), 𝜏 < 𝑡.

Set 𝑐∗ ∶= min𝑛

layer

𝑐𝑛

layer

. Then for every ℓ ≥ 1,

sup

𝑇≥ℓ+1

sup

0≤𝜏≤𝑇−ℓ−1

sup

𝑥∈𝒳(𝑇)0

∥𝐽𝜏+ℓ,𝜏e2e,(𝑁layer)(𝑥;𝑇)∥ ≲𝑁

layer

(1 + ℓ)𝑁

layer−1𝑒−𝑐∗ℓ.

In particular, this yields a genuine horizon-uniform exponential forgetting law in the lag ℓ.

- (iii) Sessa. Assume that for each layer 𝑛layer there exist 𝑎𝑛

- Corollary 4.7 (Horizon-uniform deep decay envelopes). Assume the hypotheses of Appendix Theorem 35.

> 0 and a common exponent 𝛽tail ∈ (0,1) such that

(1 + log(1 + 𝑡 − 𝜏)), 𝜏 < 𝑡. Then for every ℓ ≥ 1,

𝐾𝑛

(𝑡,𝜏) ≤ 𝑎𝑛

(𝑡 − 𝜏)−𝛽

layer

tail

layer

layer

𝑁layer

∑

ℓ𝑘(1−𝛽

tail)−1(1 + log(1 + ℓ))𝑘.

∥𝐽𝜏+ℓ,𝜏e2e,(𝑁layer)(𝑥;𝑇)∥ ≲𝑁

layer,𝛽tail

𝑇≥ℓ+1

0≤𝜏≤𝑇−ℓ−1

𝑥∈𝒳(𝑇)0

𝑘=1

sup

sup

sup

𝑁layer(1 − 𝛽tail) < 1,

In particular, if

then the right-hand side tends to 0 as ℓ → ∞, yielding a genuine horizon-uniform asymptotic decay law in the lag. Outside this subcritical regime, one still retains a controlled horizon-uniform upper envelope.

Proof sketch. Apply the horizon-uniform residual calculus in Appendix Theorem 35. The Transformer, Mamba, and Sessa kernel-class estimates are proved in Appendix Propositions 32, 33, and 34, respectively. Combining those bounds yields the stated horizon-uniform envelopes.

| |
|---|

layer−1/ℓ on bounded-source families, failed-freeze-time Mamba attenuates exponentially, and Sessa retains the stated heavy-tail upper envelope. These are upperenvelope results. They are the right tool for the impossibility statements for the comparison classes, but they do not yet yield a positive retrieval theorem for Sessa. The next subsection does.

Consequence The fixed-horizon deep bounds are recorded in Appendix H, whereas Corollary 4.7 gives lag laws uniform in 𝑇. Thus diffuse Transformers dilute like (logℓ)𝑁

##### 4.2.8 Flexible finite-horizon selective retrieval

We now state the main positive memory theorem of the section. The point is not merely that Sessa admits a heavy-tail upper envelope, but that on each finite-horizon family it can realize prescribed retrieval exponents 𝜈𝑘(𝛽) = 𝑘(1 − 𝛽) − 1, with constants uniform in both the horizon 𝐻 and the source index 𝜏∗. For each 𝐻 and 𝜏∗, the realizing network may depend on (𝐻,𝜏∗), while the retrieval-profile constants remain uniform in both parameters.

- Definition 5 (Flexible finite-horizon profile realization). Fix an integer 𝜏max ≥ 0, an exponent 𝜈 ∈ ℝ, and for each 𝐻 ≥ 1 a horizon

𝑇𝐻 ∶= 𝜏max + 𝐻 + 1. Let 𝒳(𝐻)0 ⊂ (ℝ𝐷)𝑇𝐻 be compact input sets satisfying the uniform bound sup

‖𝑥‖∞,2 ≤ 𝑅 < ∞.

𝐻≥1

𝑥∈𝒳(𝐻)0

sup

Let ℭ be an architecture class. We say that ℭ realizes the profile 𝜈 on the bounded source family

0 ≤ 𝜏∗ ≤ 𝜏max if there exist constants

𝑚− > 0, 𝑚+ < ∞, 𝑐− > 0, independent of 𝐻 and 𝜏∗, such that for every 𝐻 ≥ 1 and every source index 𝜏∗ ∈ {0,…,𝜏max}, there exist

- (i) a network 𝐺𝐻,𝜏

∗

∈ ℭ acting on (ℝ𝐷)𝑇𝐻,

- (ii) a source probe

𝑐(𝐻,𝜏∗) ∈ ℝ𝐷 and target probes 𝜌(𝐻,𝜏

∗)

𝑡 ∈ ℝ𝐷, 0 ≤ 𝑡 ≤ 𝑇𝐻 − 1, satisfying the normalization bounds

‖𝑐(𝐻,𝜏∗)‖2 ≤ 1, ‖𝜌(𝐻,𝜏

∗)

𝑡 ‖2 ≤ 1 (0 ≤ 𝑡 ≤ 𝑇𝐻 − 1),

- (iii) the full end-to-end Jacobian blocks

𝐽𝐺𝐻,𝜏

∗

𝑡,𝜏 (𝑥) ∶=

𝜕𝐺𝐻,𝜏

∗,𝑡(𝑥) 𝜕𝑥𝜏

∈ ℝ𝐷×𝐷,

- (iv) the scalar transport score

𝑡,𝜏 (𝑥) ∶= (𝜌(𝐻,𝜏

𝑡 )⊤𝐽𝐺𝐻,𝜏

𝑡,𝜏 (𝑥)𝑐(𝐻,𝜏∗),

S(𝐻,𝜏∗)

∗)

∗

- (v) and the corresponding selective margin

𝑡,𝜏∗ (𝑥) ∶= S(𝐻,𝜏∗)

𝑡,𝜏∗ (𝑥) − ∑

∣S(𝐻,𝜏∗)

𝑡,𝜏 (𝑥)∣.

M(𝐻,𝜏∗)

0≤𝜏<𝑡 𝜏≠𝜏∗

These data are required to satisfy, for every 𝑥 ∈ 𝒳(𝐻)0 ,

𝜏∗+1,𝜏∗(𝑥) ≤ 𝑚+, and

𝑚− ≤ M(𝐻,𝜏∗)

M(𝐻,𝜏∗) 𝜏∗+ℓ,𝜏∗(𝑥) ≥ 𝑐−(1 + ℓ)𝜈, 1 ≤ ℓ ≤ 𝐻.

- Theorem 12 (Flexible finite-horizon selective retrieval for deep Sessa). Work in the identity-normalized formulation with the exact GELU activation

GELU(𝑧) = 𝑧 Φ(𝑧), and assume

𝐷 ≥ 7. Fix

𝛽 ∈ (0,1), 𝑘 ≥ 1, 𝜏max ≥ 0, and define

𝜈𝑘(𝛽) ∶= 𝑘(1 − 𝛽) − 1.

Let {𝒳(𝐻)0 }𝐻≥1 be a uniformly bounded family of compact sets as in Definition 5. Then the class of LN-free Sessa networks realizes the profile 𝜈𝑘(𝛽) on the bounded source family 0 ≤ 𝜏∗ ≤ 𝜏max in the sense of Definition 5.

𝑚− > 0, 𝑚+ < ∞, 𝑐− > 0,

More precisely, there exist constants

depending only on (𝑘,𝛽,𝜏max,𝑅), but independent of 𝐻 and 𝜏∗, such that for every 𝐻 ≥ 1 and every 𝜏∗ ∈ {0,…,𝜏max}, there exist a finite-depth LN-free Sessa network

𝐺𝐻,𝜏

∶ (ℝ𝐷)𝑇𝐻 → (ℝ𝐷)𝑇𝐻

∗

and a scalar channel score S(𝐻,𝜏∗) with selective margin M(𝐻,𝜏∗) such that for every 𝑥 ∈ 𝒳(𝐻)0 ,

𝑚− ≤ M(𝐻,𝜏∗)

𝜏∗+1,𝜏∗(𝑥) ≤ 𝑚+, and

M(𝐻,𝜏∗) 𝜏∗+ℓ,𝜏∗(𝑥) ≥ 𝑐−(1 + ℓ)𝜈𝑘(𝛽), 1 ≤ ℓ ≤ 𝐻.

Consequently: if 𝜈𝑘(𝛽) < 0, deep Sessa realizes a decaying profile; if 𝜈𝑘(𝛽) = 0, it realizes a frozen profile; and if 𝜈𝑘(𝛽) > 0, it realizes an increasing profile.

Proof sketch. Composite architecture. Fix 𝐻 ≥ 1 and 0 ≤ 𝜏∗ ≤ 𝜏max. Set

𝐿𝐻 ∶= 𝜏max + 𝐻, 𝑇𝐻 ∶= 𝐿𝐻 + 1. We construct

∗,𝜀𝐻 ∘ 𝑄𝐻 ∘ 𝑃𝐻. Here 𝑃𝐻 writes a strictly ordered positional code, 𝑄𝐻 is a signal-transparent preparatory network producing the

𝐺𝐻,𝜏

= 𝑀𝐻,𝑘 ∘ ⋯ ∘ 𝑀𝐻,1 ∘ 𝑆𝐻,𝜏

∗

power profile, 𝑆𝐻,𝜏

∗,𝜀𝐻 selects the source 𝜏∗, and 𝑀𝐻,1,…,𝑀𝐻,𝑘 are diffuse profile-compensated macro-layers.

By Corollaries 4.11 and 4.12, 𝑃𝐻 writes a strictly ordered positional code on 𝑒pos while remaining transparent to perturbations along 𝑒sig. Corollary K.21 yields a constant-depth network 𝑄𝐻 that preserves the signal and positional channels and writes a profile

𝑟𝑡 ≍ (𝑡 + 1)1−𝛽.

Lemma K.12 yields a selector with gain ≍ 1 at 𝜏∗ and off-target suppression 𝜀𝐻 ≍ (𝐻 +1)−1. Lemma K.22 yields macro-layers whose selected-channel transport has kernel size ≍ (𝑖 + 1)−𝛽.

Appendix Lemma K.9 identifies the selected-channel transport of the post-preparatory stack with the actual Jacobian score. The desired lower bound follows by restricting to balanced 𝑘-jump paths and applying Lemma K.25, while the competitor contribution is controlled by Lemma K.26. Choosing the construction constants appropriately makes the competitor mass absorbable for all 1 ≤ ℓ ≤ 𝐻, yielding the stated anchored bounds.

| |
|---|

- (i) for 𝑘 = 1, one has 𝜈1(𝛽) = −𝛽 < 0,

so only decaying profiles occur;

- (ii) for 𝑘 ≥ 2 and

𝛽 = 1 −

1 𝑘

, one gets the frozen profile 𝜈𝑘(𝛽) = 0;

- (iii) for 𝑘 ≥ 2 and

- Corollary 4.8 (Flexible frozen and increasing profiles require depth). Under Theorem 12:

1 𝑘

, one gets the increasing profile 𝜈𝑘(𝛽) > 0.

0 < 𝛽 < 1 −

This is the matching negative statement in the same family-over-𝐻 regime. By the horizon-uniform end-to-end envelopes from Section 4.2.7, diffuse fixed-depth Transformers and failed-freeze-time fixed-depth Mamba admit only decaying upper bounds, so they cannot realize frozen or increasing retrieval profiles.

- 4.2.9 Impossibility for the comparison classes in the same flexible finite-horizon regime

Proposition 13 (Comparison-class impossibility for flexible selective retrieval). Fix 𝜏max ≥ 0, and let

𝑇𝐻 = 𝜏max + 𝐻 + 1.

Assume we are given, for every 𝐻 ≥ 1 and every 𝜏∗ ∈ {0,…,𝜏max}, a network 𝐺comp𝐻,𝜏

∗

from one of the following two comparison classes: a depth-𝐿 causal Transformer in the diffuse smooth-routing regime, or a depth-𝐿 causal Mamba stack in the failed-freeze-time regime.

Assume moreover that, in the Transformer case, the family satisfies the hypotheses of Corollary 4.7, item (i), with constants independent of 𝐻 and 𝜏∗, and that, in the Mamba case, the family satisfies the hypotheses of Corollary 4.7, item (ii), with constants independent of 𝐻 and 𝜏∗.

Then no such comparison-class family can realize a frozen or increasing profile in the sense of Definition 5. More precisely:

- (i) Transformer. There do not exist constants 𝑚− > 0, 𝑚+ < ∞, 𝑐− > 0, and 𝜈 ≥ 0, independent of 𝐻 and

𝜏∗, such that

𝜏∗+1,𝜏∗(𝑥) ≤ 𝑚+, and

𝑚− ≤ M(𝐻,𝜏∗)

𝜏∗+ℓ,𝜏∗(𝑥) ≥ 𝑐−(1 + ℓ)𝜈, 1 ≤ ℓ ≤ 𝐻, hold uniformly for all 𝐻,𝜏∗,𝑥.

M(𝐻,𝜏∗)

- (ii) Mamba. The same impossibility holds for failed-freeze-time Mamba families.

𝑡 ‖2 ≤ 1. Hence for every admissible 𝐻,𝜏∗,𝑥,𝑡,𝜏,

‖𝑐(𝐻,𝜏∗)‖2 ≤ 1, ‖𝜌(𝐻,𝜏

Proof. Assume toward a contradiction that such a realization exists. By Definition 5, the probes satisfy

∗)

𝑡,𝜏 (𝑥)∥. Therefore

∣S(𝐻,𝜏∗)

𝑡,𝜏 (𝑥)∣ = ∣(𝜌(𝐻,𝜏

𝑡 )⊤𝐽𝐺

𝑡,𝜏 (𝑥)𝑐(𝐻,𝜏∗)∣ ≤ ∥𝐽𝐺

∗)

comp 𝐻,𝜏∗

comp 𝐻,𝜏∗

𝑡,𝜏∗ (𝑥)∥. For Transformers, Corollary 4.7, item (i), applied to the family 𝐺comp𝐻,𝜏

𝑡,𝜏∗ (𝑥) ≤ ∣S(𝐻,𝜏∗)

𝑡,𝜏∗ (𝑥)∣ ≤ ∥𝐽𝐺

M(𝐻,𝜏∗)

comp 𝐻,𝜏∗

∗

(log(1 + ℓ))𝐿−1 1 + ℓ

, gives the horizon-uniform bounded-sourcefamily envelope

∥𝐽𝐺

𝜏+ℓ,𝜏 (𝑥)∥ ≲

,

comp 𝐻,𝜏∗

uniformly over all admissible 𝐻,𝜏∗,𝑥 and all 0 ≤ 𝜏 ≤ 𝜏max. This tends to 0 as ℓ → ∞. For Mamba, item (ii) gives

∥𝐽𝐺

𝜏+ℓ,𝜏 (𝑥)∥ ≲ (1 + ℓ)𝐿−1𝑒−𝑐ℓ, uniformly over all admissible 𝐻,𝜏∗,𝑥,𝜏. This also tends to 0. Since a frozen or increasing profile would require

comp 𝐻,𝜏∗

M(𝐻,𝜏∗) 𝜏∗+ℓ,𝜏∗(𝑥) ≥ 𝑐−(1 + ℓ)𝜈 (𝜈 ≥ 0),

uniformly in all admissible 𝐻,𝜏∗,𝑥,ℓ, this is impossible in either comparison class.

| |
|---|

- Corollary 4.9 (Flexible selective retrieval separates Sessa from the comparison classes). In the regime of Definition 5:

- (i) deep identity-normalized Sessa realizes the full exponent family

𝜈𝑘(𝛽) = 𝑘(1 − 𝛽) − 1;

- (ii) diffuse fixed-depth Transformers and failed-freeze-time fixed-depth Mamba do not realize frozen or increasing profiles.

Thus, in this uniform finite-horizon family-over-𝐻 regime, deep Sessa supports flexible selective retrieval, whereas the two comparison classes do not.

#### 4.3 Internal positional encoding

Sessa does not require an explicit absolute positional embedding in the feedback branch. The key point is that the feedback solve can itself write a separated absolute positional signal. The main lemma gives this positional writer, and the corollaries record the two refinements used later: one-directional writing with signal transparency, and continuous recovery of the position index.

Lemma 4.10 (Feedback generates ordered separated positional codes). Fix 𝑇 ≥ 2 and model width 𝑚 ≥ 1. There exists a single width-𝑚 Sessa block 𝐺(1) and vectors 𝑝0,…,𝑝𝑇−1 ∈ ℝ𝑚 such that for all token sequences ℎ ∈ ℝ𝑇×𝑚,

𝐺(1)(ℎ)𝑡 = ℎ𝑡 + 𝑝𝑡, 𝑡 = 0,…,𝑇 − 1.

Moreover, for any compact 𝒦_set ⊂ ℝ𝑇×𝑚 the offsets can be chosen so that there exist a unit direction 𝑢 ∈ ℝ𝑚 and pairwise disjoint compact intervals

𝐽0 < 𝐽1 < ⋯ < 𝐽𝑇−1 ⊂ (0,∞) with

⟨ℎ𝑡 + 𝑝𝑡,𝑢⟩ ∈ 𝐽𝑡 for all ℎ ∈ 𝒦_set, 𝑡 = 0,…,𝑇 − 1.

Proof sketch. Choose parameters so that the mixer input is constant, the forward branch produces a constant forward signal, and the feedback routing is chosen so that the induced scalar solve generates a deterministic strictly increasing sequence on the finite prefix. Project that scalar sequence onto a chosen direction, then shift and rescale it so that the resulting compact scalar ranges are pairwise disjoint, strictly ordered, and contained in (0,∞). See Appendix I.5.

| |
|---|

- Corollary 4.11 (One-directional internal positional writer). Under the hypotheses of Lemma 4.10, the block can be chosen so that there exists a unit direction 𝑒pos ∈ ℝ𝑚 and scalars 𝜆0,…,𝜆𝑇−1 with

𝐺(1)(ℎ)𝑡 = ℎ𝑡 + 𝜆𝑡𝑒pos, 𝑡 = 0,…,𝑇 − 1,

for all token sequences ℎ ∈ ℝ𝑇×𝑚. Moreover, for any compact 𝒦_set ⊂ ℝ𝑇×𝑚, the same block can be chosen so that there exist pairwise disjoint compact intervals

𝐽0 < 𝐽1 < ⋯ < 𝐽𝑇−1 ⊂ (0,∞) with

⟨𝐺(1)(ℎ)𝑡,𝑒pos⟩ ∈ 𝐽𝑡 for all ℎ ∈ 𝒦_set, 𝑡 = 0,…,𝑇 − 1.

Proof. In the construction underlying Lemma 4.10, the deterministic scalar sequence generated by the feedback solve is written onto a chosen output direction. Choosing that output direction to be 𝑒pos and writing no offset on the orthogonal complement yields the form

𝐺(1)(ℎ)𝑡 = ℎ𝑡 + 𝜆𝑡𝑒pos. The interval-separation conclusion is exactly the same as in Lemma 4.10.

| |
|---|

- Corollary 4.12 (Signal transparency of the one-directional positional writer). Under the hypotheses of Corollary 4.11, let 𝑒sig ∈ ℝ𝑚 be any unit vector with

𝑒sig ⟂ 𝑒pos. Then for every token sequence ℎ ∈ ℝ𝑇×𝑚, every source index 𝜏 ∈ {0,…,𝑇 − 1}, and every scalar 𝑎 ∈ ℝ,

𝐺(1)(ℎ + 𝑎𝑒sig1[⋅ = 𝜏])𝑡 = 𝐺(1)(ℎ)𝑡 + 𝑎𝑒sig1[𝑡 = 𝜏], 𝑡 = 0,…,𝑇 − 1. In particular,

⟨𝐺(1)(ℎ + 𝑎𝑒sig1[⋅ = 𝜏])𝑡,𝑒pos⟩ = ⟨𝐺(1)(ℎ)𝑡,𝑒pos⟩ ∀𝑡, so perturbations along 𝑒sig leave the internally written positional coordinate unchanged.

𝐺(1)(ℎ)𝑡 = ℎ𝑡 + 𝜆𝑡𝑒pos. Therefore

Proof. By Corollary 4.11,

𝐺(1)(ℎ + 𝑎𝑒sig1[⋅ = 𝜏])𝑡 = ℎ𝑡 + 𝑎𝑒sig1[𝑡 = 𝜏] + 𝜆𝑡𝑒pos = 𝐺(1)(ℎ)𝑡 + 𝑎𝑒sig1[𝑡 = 𝜏]. Since 𝑒sig ⟂ 𝑒pos, taking the 𝑒pos-coordinate gives the second claim.

| |
|---|

- Corollary 4.13 (Continuous recovery of the position index). Under the hypotheses of Corollary 4.11, fix a compact set

𝒦_set ⊂ ℝ𝑇×𝑚, and choose the block so that there exist pairwise disjoint compact intervals

𝐽0 < 𝐽1 < ⋯ < 𝐽𝑇−1 ⊂ (0,∞) with

⟨𝐺(1)(ℎ)𝑡,𝑒pos⟩ ∈ 𝐽𝑡 ∀ℎ ∈ 𝒦_set, ∀𝑡 = 0,…,𝑇 − 1. Then there exists a continuous map

𝜓 ∶ ℝ𝑚 → ℝ such that

𝜓(𝐺(1)(ℎ)𝑡) = 𝑡 ∀ℎ ∈ 𝒦_set, ∀𝑡 = 0,…,𝑇 − 1. In particular, the position index 𝑡 is recoverable by a continuous tokenwise map on the shifted-token set

𝑇−1

⋃

{𝐺(1)(ℎ)𝑡 ∶ ℎ ∈ 𝒦_set}.

𝑡=0

𝐽𝑡 = [𝑎𝑡,𝑏𝑡]. Since the intervals are pairwise disjoint and ordered, one has

Proof. Write each compact interval as

𝑏𝑡 < 𝑎𝑡+1 (𝑡 = 0,…,𝑇 − 2). Define a continuous function 𝑔 ∶ ℝ → ℝ by requiring

𝑔(𝑠) = 𝑡 for all 𝑠 ∈ 𝐽𝑡,

interpolating linearly on each gap [𝑏𝑡,𝑎𝑡+1], and extending constantly on (−∞,𝑎0] and [𝑏𝑇−1,∞). Then 𝑔 is continuous on ℝ and satisfies 𝑔|𝐽

≡ 𝑡 for every 𝑡. Now define

𝑡

𝜓(𝑧) ∶= 𝑔(⟨𝑧,𝑒pos⟩), 𝑧 ∈ ℝ𝑚. Since 𝑧 ↦ ⟨𝑧,𝑒pos⟩ is continuous, 𝜓 is continuous. Moreover, for every ℎ ∈ 𝒦_set and every 𝑡,

⟨𝐺(1)(ℎ)𝑡,𝑒pos⟩ ∈ 𝐽𝑡, hence

𝜓(𝐺(1)(ℎ)𝑡) = 𝑔(⟨𝐺(1)(ℎ)𝑡,𝑒pos⟩) = 𝑡.

| |
|---|

Consequence Sessa can internally generate an absolute positional code through feedback, even when the forward branch uses only relative-position-aware routing such as RoPE.

#### 4.4 Universal approximation of causal maps

We state a universal approximation result for Sessa networks on compact domains, in the standard causal decoder setting. Since intermediate constructions may require an internal width 𝑚 ≥ 𝐷, we state the result for Sessa with tokenwise linear adapters 𝐷 → 𝑚 → 𝐷.

Definition 6 (Causality). A map 𝐹 ∶ 𝒟 → ℝ𝑇×𝐷 is causal if for every 𝑡 and all 𝑥,𝑥′ ∈ 𝒟, 𝑥0∶𝑡 = 𝑥′0∶𝑡 implies 𝐹(𝑥)𝑡 = 𝐹(𝑥′)𝑡.

Theorem 14 (Universal approximation by concrete Sessa with adapters). Let 𝒟 ⊂ ℝ𝑇×𝐷 be compact and let 𝐹 ∶ 𝒟 → ℝ𝑇×𝐷 be continuous and causal. Then for any 𝜀 > 0 there exist an even query/key width 𝑑𝑘 ≥ 2, a model width 𝑚 ≥ 𝐷, tokenwise adapters

Embed ∶ ℝ𝐷 → ℝ𝑚, Unembed ∶ ℝ𝑚 → ℝ𝐷, and a finite-depth width-𝑚 concrete Sessa network 𝐺 such that

∥𝐹(𝑥) − Unembed(𝐺(Embed(𝑥)))∥

< 𝜀.

𝐹

𝑥∈𝒟

sup

Proof sketch. (i) Use a single Sessa block to write an internal positional code.

- (ii) Use a finite stack of concrete Sessa blocks to encode each relevant causal prefix into dedicated internal coordinates.
- (iii) Apply a finite tokenwise readout stack, again implemented by concrete Sessa blocks, to approximate the desired causal output on the resulting compact encoded-state set.

Details appear in Appendix I, in the proof of Theorem 14.

| |
|---|

### 5 Experiments

We compare three model variants that share the same decoder macro-architecture and training setup and differ only in the sequence mixer. The mixers are Sessa mixer, multi-head self-attention, and Mamba2 mixer. We match parameter count, use the same optimizer and training schedule, and train all models for the same number of optimization steps.

We do not report aggregate results on the full Long Range Arena (LRA) suite (Tay et al., 2021). Although LRA was originally proposed as a testbed for long-range dependencies, subsequent analyses have highlighted several issues suggesting that strong performance on LRA can be confounded by factors unrelated to robust long-context reasoning. (Tay et al., 2021; Miralles-González et al., 2025) We evaluate long-context behavior on SymbolSoup and Diffuse MQAR, and short-context language modeling on SimpleStories. (Finke et al., 2025; SimpleStories Project, 2025)

#### 5.1 Synthetic long-range tasks

##### 5.1.1 Datasets and tasks

SymbolSoup. SymbolSoup is a long-range classification dataset with two informative stylized blocks separated by label-independent noise. Each example contains three noise blocks and two stylized blocks, one from each style family. The order of the two stylized blocks is randomized.

noise <sep1> first/second stylized part <sep2> noise <sep1> second/first stylized part <sep2> noise <sep> <label>.

The label is the pair of styles used in the two stylized blocks. Stylized blocks are generated by a Markov-like process with unigram and bigram preferences and occasional motif insertion plus small symbol noise.

Diffuse MQAR. We additionally evaluate on a modified multi-query associative recall benchmark based on MQAR (Arora et al., 2024). Relative to the original formulation, our variant uses multi-token keys, structured distractors with shared prefixes and mismatched suﬀixes, and explicit control of the source–query lag. Each example contains a prefix memory block of key–value pairs, a noise block populated with distractor key–valuelike patterns, and a terminal query block. The test split includes retrieval lags up to 4× larger than those seen during training.

Table 1: Long-context test results (mean ± std over 2 seeds). For SymbolSoup we report classification accuracy; for Diffuse MQAR we report token accuracy.

Model SymbolSoup Acc ↑ Diffuse MQAR Token Acc ↑ Sessa 0.8601 ± 0.0016 0.1541 ± 0.0071 Transformer 0.7921 ± 0.0070 0.1222 ± 0.0003 Mamba2 0.0500 ± 0.0000 0.0021 ± 0.0000

Mamba-2 did not converge on SymbolSoup or Diffuse MQAR. We view this as qualitatively consistent with our selective-SSM theory: when noise makes the selection signal weakly separable, the resulting non-vanishing freeze-time errors restore exponential attenuation of long-range influence, as formalized in Proposition 5 and Corollary 4.6. This interpretation is relevant to Mamba-2 because it is itself a selective SSM, specifically a scalar-identity restricted variant in the SSD framework (Dao and Gu, 2024).

#### 5.2 SimpleStories language modeling

##### 5.2.1 Dataset and task

For the short-context regime we use a SimpleStories corpus of short, synthetically generated stories. Each story is written in simplified English with a small vocabulary and constrained syntax.

We treat this corpus as a causal language modeling benchmark. The text is tokenized with a subword tokenizer shared across all architectures, and training sequences are formed by concatenating stories and splitting them into fixed-length segments. The model predicts the next token at each position using a left-to-right mask. We report validation perplexity.

Table 2: SimpleStories test results (mean ± std over 2 seeds).

Model Perplexity ↓ Top-1 acc ↑ Top-5 acc ↑ Transformer 7.6701 ± 0.0313 50.441 ± 0.059% 78.497 ± 0.062% Mamba2 7.7229 ± 0.0207 50.299 ± 0.046% 78.302 ± 0.043% Sessa 8.3700 ± 0.0482 49.144 ± 0.081% 77.119 ± 0.090%

We hypothesize that the small performance drop of Sessa in the short-context regime is due to the feedback mechanism being less necessary for this task. Under matched parameter count, a portion of Sessa’s capacity is allocated to the feedback branch, which may be weakly utilized on short-context. To test this interpretation, we ran a control experiment with the feedback branch removed while keeping the remainder of the architecture unchanged. The ablated model improves over full Sessa on SimpleStories, reducing test perplexity from 8.3700 ± 0.0482 to 8.0902 ± 0.0192 and increasing top-1 accuracy from 49.144 ± 0.081% to 49.648 ± 0.026%. This supports the view that feedback is less beneficial in the short-context regime, while remaining consistent with Sessa’s stronger results on long-context tasks, where feedback appears to be more useful.

### 6 Discussion

The main comparison in this paper is not between favorable operating regimes of Transformers, Mamba, and Sessa, but between matched regimes in which sharp retrieval is unavailable. For attention, this appears as diffuse, low-separation routing, so the selector cannot concentrate mass on a small set of relevant indices. For Mamba,

the analogous failure is failed freeze time, so the model cannot maintain a long preserve corridor on the relevant interval. These are natural failure regimes for the respective architectures, and they provide a common basis for comparison.

In this matched setting, the difference comes from the memory mechanism rather than from access to sharp routing. Diffuse attention remains one-hop and therefore suffers dilution. Failed-freeze-time Mamba remains chain-structured and therefore exhibits exponential attenuation. Sessa is also studied in a diffuse regime, but its feedback solve aggregates influence over multiple hop counts and, in dense settings, over many temporal paths. This is the structural source of its slower long-range decay.

The main separation is not only in the polynomial tail, but in the selective-retrieval result. In the same familyover-𝐻 regime, deep Sessa realizes flexible selective retrieval profiles, whereas diffuse fixed-depth Transformers and failed-freeze-time fixed-depth Mamba do not realize frozen or increasing profiles. Thus the separation is not merely quantitative at the level of decay rates; it is qualitative at the level of what retrieval behavior the architectures can realize under the same matched breakdown of sharp retrieval.

The broader point is that long-context behavior depends not only on how routing coeﬀicients are produced, but also on how they are composed over time. When sharp retrieval fails, as can become increasingly likely as context length grows, this distinction becomes decisive. In that regime, Sessa can still support flexible selective retrieval through its multi-hop feedback structure.

### References

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, et al. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. doi: 10.48550/arXiv.2403.07815. URL https: //openreview.net/forum?id=gerNCVqqtR. Accepted by TMLR (OpenReview); arXiv:2403.07815.

Panos J. Antsaklis and Anthony N. Michel. Linear Systems. Birkhäuser, Boston, 1 edition, 2006. Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and

Christopher Ré. Zoology: Measuring and improving recall in eﬀicient language models. In International Conference on Learning Representations (ICLR), 2024. doi: 10.48550/arXiv.2312.04927. URL https: //openreview.net/forum?id=LY3ukUANko. ICLR 2024 poster; arXiv:2312.04927.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. 2016. doi: 10.48550/arXiv.1607.

06450. URL https://arxiv.org/abs/1607.06450.

Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for selfsupervised learning of speech representations. In Advances in Neural Information Processing Systems (NeurIPS),

2020. URL https://arxiv.org/abs/2006.11477. arXiv:2006.11477. Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020. doi: 10.48550/arXiv.2004.05150. URL https://arxiv.org/abs/2004.05150.

Sidney Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, Usvsn Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. GPT-NeoX-20B: An open-source autoregressive language model. In Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models, pages 95–136, virtual+Dublin, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.bigscience-1.9. URL https://aclanthology.org/2022.bigscience-1.9/.

Rishi Bommasani et al. On the opportunities and risks of foundation models. CoRR, abs/2108.07258, 2021. doi: 10.48550/arXiv.2108.07258. URL https://arxiv.org/abs/2108.07258. Stanford CRFM report.

Tom B. Brown et al. Language models are few-shot learners. Advances in Neural Information Processing Systems (NeurIPS), 2020. URL https://arxiv.org/abs/2005.14165. arXiv:2005.14165.

Aydar Bulatov, Yuri Kuratov, and Mikhail Burtsev. Recurrent memory transformer. In Advances in Neural Information Processing Systems (NeurIPS), 2022. doi: 10.48550/arXiv.2207.06881. URL https://arxiv.org/ abs/2207.06881. NeurIPS 2022; arXiv:2207.06881.

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019. doi: 10.48550/arXiv.1904.10509. URL https://arxiv.org/abs/1904. 10509.

Mohammed Dahleh, Munther A. Dahleh, and George Verghese. Lectures on dynamic systems and control, chapter 15: External input-output stability. MIT OpenCourseWare (6.241J/16.338J), course notes, 2011a. URL https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/ 5b744a33f5db9b0cc70dbc04a9de5706_MIT6_241JS11_chap15.pdf.

Mohammed Dahleh, Munther A. Dahleh, and George Verghese. Lectures on dynamic systems and control, chapter 27: Poles and zeros of mimo systems. MIT OpenCourseWare (6.241J/16.338J), course notes, 2011b. URL https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/ 8a8013268491f54fd65614f299a05290_MIT6_241JS11_chap27.pdf.

Mohammed Dahleh, Munther A. Dahleh, and George Verghese. Lectures on dynamic systems and control, chapter 30: Minimality and stability of interconnected systems. MIT OpenCourseWare (6.241J/16.338J), course notes, 2011c. URL https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/ 5e41c2e287bde74f5326d258e89c951c_MIT6_241JS11_chap30.pdf.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformerxl: Attentive language models beyond a fixed-length context. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019. doi: 10.18653/v1/P19-1285. URL https: //aclanthology.org/P19-1285/. arXiv:1901.02860; doi:10.48550/arXiv.1901.02860.

Hugo Dalla-Torre et al. The nucleotide transformer: Building and evaluating robust foundation models for human genomics. Nature Methods, 22(2):287–297, 2025. doi: 10.1038/s41592-024-02523-z. URL https://www.nature. com/articles/s41592-024-02523-z. Version of record published online 28 Nov 2024; issue date Feb 2025.

Tri Dao and Albert Gu. Transformers are ssms: Generalized models and eﬀicient algorithms through structured state space duality. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 10041–10071. PMLR, 2024. doi: 10.48550/arXiv.2405.21060. URL https://proceedings.mlr.press/v235/dao24a.html. ICML 2024; introduces Mamba-2 via the SSD framework; arXiv:2405.21060.

Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Nanning Zheng, and Furu Wei. Longnet: Scaling transformers to 1,000,000,000 tokens. arXiv preprint arXiv:2307.02486, 2023. doi: 10.48550/arXiv.2307.02486. URL https://arxiv.org/abs/2307.02486.

Alexey Dosovitskiy et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021. URL https://arxiv.org/abs/2010.

11929. arXiv:2010.11929.

Angela Fan, Thibaut Lavril, Edouard Grave, Armand Joulin, and Sainbayar Sukhbaatar. Addressing some limitations of transformers with feedback memory. arXiv preprint arXiv:2002.09402, 2020. doi: 10.48550/ arXiv.2002.09402. URL https://arxiv.org/abs/2002.09402. OpenReview submission notes it was under review for ICLR 2021.

Lennart Finke, Chandan Sreedhara, Thomas Dooms, Mat Allen, Emerald Zhang, Juan Diego Rodriguez, Noa Nabeshima, Thomas Marshall, and Dan Braun. Parameterized synthetic text generation with simplestories. In NeurIPS 2025 Datasets and Benchmarks Track, 2025. doi: 10.48550/arXiv.2504.09184. URL https:// openreview.net/forum?id=sVh3eQ642W. NeurIPS 2025 Datasets and Benchmarks Track poster (OpenReview); arXiv:2504.09184.

Walter Gautschi. Some elementary inequalities relating to the gamma and incomplete gamma function. Journal of Mathematics and Physics, 38:77–81, 1959. doi: 10.1002/sapm195938177.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In Conference on Language Modeling (COLM), 2024. doi: 10.48550/arXiv.2312.00752. URL https://openreview.net/forum? id=tEYskw1VY2. COLM 2024 (OpenReview); arXiv:2312.00752.

Albert Gu, Karan Goel, and Christopher Ré. Eﬀiciently modeling long sequences with structured state spaces. In International Conference on Learning Representations (ICLR), 2022a. doi: 10.48550/arXiv.2111.00396. URL https://arxiv.org/abs/2111.00396. arXiv:2111.00396.

Albert Gu, Ankit Gupta, Karan Goel, and Christopher Ré. On the parameterization and initialization of diagonal state space models. In Advances in Neural Information Processing Systems (NeurIPS), 2022b. doi: 10.48550/ arXiv.2206.11893. URL https://arxiv.org/abs/2206.11893. arXiv:2206.11893; introduces S4D.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). 2016. doi: 10.48550/arXiv.1606.08415. URL https://arxiv.org/abs/1606.08415.

Roger A. Horn and Charles R. Johnson. Matrix Analysis. Cambridge University Press, 2 edition, 2012. ISBN

9780521839402. doi: 10.1017/CBO9781139020411. Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359–366, 1989. doi: 10.1016/0893-6080(89)90020-8. Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc V. Le. Transformer quality in linear time. In Proceedings of the

39th International Conference on Machine Learning (ICML), volume 162 of Proceedings of Machine Learning Research, pages 9099–9117. PMLR, 2022. URL https://proceedings.mlr.press/v162/hua22a.html.

Ningyuan Huang, Miguel Sarabia, Abhinav Moudgil, Pau Rodriguez, Luca Zappella, and Federico Danieli. Understanding input selectivity in mamba: Impact on approximation power, memorization, and associative recall capacity. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 25693–25727. PMLR, 2025. doi: 10.48550/arXiv.2506.11891. URL https://proceedings.mlr.press/v267/huang25ab.html. ICML 2025; arXiv:2506.11891.

DeLesley Scott Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, and Behnam Neyshabur. Block-recurrent transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2022. doi: 10.48550/arXiv. 2203.07852. URL https://arxiv.org/abs/2203.07852. NeurIPS 2022; arXiv:2203.07852.

Dongseong Hwang, Weiran Wang, Zhuoyuan Huo, Khe Chai Sim, and Pedro Moreno Mengibar. Transformerfam: Feedback attention is working memory. arXiv preprint arXiv:2404.09173, 2024. doi: 10.48550/arXiv.2404.09173. URL https://arxiv.org/abs/2404.09173.

Rudolf E. Kalman. A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82

(1):35–45, 1960. doi: 10.1115/1.3662552.

Moshe Leshno, Vladimir Ya. Lin, Allan Pinkus, and Shimon Schocken. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural Networks, 6(6):861–867, 1993. doi: 10.1016/S0893-6080(05)80131-5.

Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, and David Camacho. On the locality bias and results in the long range arena. arXiv preprint arXiv:2501.14850, 2025. doi: 10.48550/arXiv.2501.14850. URL https://arxiv.org/abs/2501.14850.

Timur Mudarisov, Mikhail Burtsev, Tatiana Petrova, and Radu State. Limitations of normalization in attention mechanism. In Advances in Neural Information Processing Systems (NeurIPS 2025), 2025. doi: 10.48550/arXiv.2508.17821. URL https://arxiv.org/abs/2508.17821. NeurIPS 2025 poster (OpenReview id: 16kX08MCav); arXiv:2508.17821v2 (revised 20 Oct 2025).

Markus N. Rabe and Charles Staats. Self-attention does not need O(𝑛2) memory. arXiv preprint arXiv:2112.05682,

2021. doi: 10.48550/arXiv.2112.05682. URL https://arxiv.org/abs/2112.05682. Noam Shazeer. Glu variants improve transformer. 2020. doi: 10.48550/arXiv.2002.05202. URL https://arxiv. org/abs/2002.05202. SimpleStories Project. SimpleStories/SimpleStories. Hugging Face Datasets, 2025. URL https://huggingface. co/datasets/SimpleStories/SimpleStories. Accessed: 2026-01-29.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. 2021. doi: 10.48550/arXiv.2104.09864. URL https://arxiv.org/abs/2104. 09864.

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for eﬀicient transformers. In International Conference on Learning Representations (ICLR), 2021. URL https://openreview.net/forum?id= qVyeW-grC2k. arXiv:2011.04006.

Heinrich Tietze. über funktionen, die auf einer abgeschlossenen menge stetig sind. Journal für die reine und angewandte Mathematik, 145:9–14, 1915. doi: 10.1515/crll.1915.145.9.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and eﬀicient foundation language models. 2023. doi: 10.48550/arXiv.2302.13971. URL https://arxiv.org/abs/2302.13971.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30 (NIPS 2017), pages 5998–6008, 2017. doi: 10.48550/arXiv.1706.03762. URL https://arxiv.org/abs/1706.03762.

Ruibin Xiong, Yunchang Yang, Di He, et al. On layer normalization in the transformer architecture. In Proceedings of the 37th International Conference on Machine Learning (ICML), 2020. URL https://proceedings.mlr. press/v119/xiong20b.html.

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big bird: Transformers for longer sequences. In Advances in Neural Information Processing Systems (NeurIPS), 2020. doi: 10.48550/arXiv.2007.14062. URL https://arxiv.org/abs/2007.14062. arXiv:2007.14062.

### Appendix

### A Definitions and notation

- A.1 Sequence norms and bounded-input sets

Definition 7 (Sup–ℓ2 norm and bounded-input balls). Fix a horizon 𝑇 ∈ ℕ∗ and token width 𝐷 ∈ ℕ∗. For a finite sequence 𝑥 = (𝑥0,…,𝑥𝑇−1) ∈ (ℝ𝐷)𝑇 define

‖𝑥‖∞,2 ∶= max

0≤𝑡≤𝑇−1

‖𝑥𝑡‖2.

For 𝑅 ≥ 0 define the ball

𝒳𝑅 ∶= {𝑥 ∈ (ℝ𝐷)𝑇 ∶ ‖𝑥‖∞,2 ≤ 𝑅}. For infinite sequences (𝑥𝑡)𝑡≥0 we use the analogous norm ‖𝑥‖∞,2 ∶= sup𝑡≥0 ‖𝑥𝑡‖2 ∈ [0,∞].

‖𝑋‖∞,2 ≤ ‖𝑋‖𝐹 ≤

√

𝑇 ‖𝑋‖∞,2 for 𝑋 ∈ ℝ𝑇×𝐷. (35)

- A.2 BIBO stability on ℓ∞

Definition 8 (BIBO stability on ℓ∞). A map 𝒩 ∶ ℓ∞(ℕ,ℝ𝐷) → ℓ∞(ℕ,ℝ𝐷) is BIBO-stable with respect to ‖⋅‖∞,2 if for every 𝐵 ≥ 0 there exists 𝐶𝐵 < ∞ such that

‖𝑥‖∞,2 ≤ 𝐵 ⟹ ‖𝒩(𝑥)‖∞,2 ≤ 𝐶𝐵.

### B Jacobian tails under diffuse feedback routing

Fix a horizon 𝑇 ∈ ℕ∗ and token width 𝐷 ∈ ℕ∗. Let 𝑥 = (𝑥0,…,𝑥𝑇−1) ∈ (ℝ𝐷)𝑇 be the input token sequence. Let 𝑓(𝑥) = (𝑓0(𝑥),…,𝑓𝑇−1(𝑥)) ∈ (ℝ𝑟)𝑇 be the forward sequence, where 𝑟 is the value space dimension, and let 𝛼fb(𝑥) = (𝛼fb𝑡,𝑗(𝑥))0≤𝑗<𝑡≤𝑇−1 be the strictly-lower attention weights. Let 𝛾(𝑥) = (𝛾0(𝑥),…,𝛾𝑇−1(𝑥)) be the feedback gains.

- B.1 Sessa feedback solve as a parametric linear system

Define the strictly lower-triangular matrix 𝐵fb(𝑥) ∈ ℝ𝑇×𝑇 by

𝛾𝑡(𝑥)𝛼fb𝑡,𝑗(𝑥), 𝑗 < 𝑡, 0, 𝑗 ≥ 𝑡.

[𝐵fb]𝑡,𝑗(𝑥) = {

The mixer output 𝑠(𝑥) = (𝑠0(𝑥),…,𝑠𝑇−1(𝑥)) ∈ (ℝ𝑟)𝑇 is defined as the unique solution to the causal solve

(𝐼 − 𝐵fb(𝑥))𝑠(𝑥) = 𝑓(𝑥). (36) Equivalently, by forward substitution,

𝑡−1

∑

𝑠0 = 𝑓0, 𝑠𝑡 = 𝑓𝑡 + 𝛾𝑡

𝛼fb𝑡,𝑗𝑠𝑗, 𝑡 ≥ 1. (37)

𝑗=0

𝜕𝑠𝑡(𝑥) 𝜕𝑥𝜏

We measure long-range sensitivity by the Jacobian blocks

𝐽𝑡,𝜏(𝑥) ∶=

∈ ℝ𝑟×𝐷, 0 ≤ 𝜏 ≤ 𝑡 ≤ 𝑇 − 1.

Throughout this appendix we focus on the long-range case 𝜏 < 𝑡 and lag ℓ ∶= 𝑡 − 𝜏 ≥ 1.

#### B.2 Assumptions for diffuse routing and smoothness Fix a radius 𝑅 ≥ 0 and work on the ball 𝒳𝑅 from Definition 7.

- Remark B.1 (On the use of 𝑡+1 and 𝑡 in dilution bounds). In this appendix the feedback attention is strictly-lower,

meaning that 𝑗 < 𝑡, so |𝒲𝑡| = 𝑡 for 𝑡 ≥ 1. We write 𝑂(1/(𝑡 + 1)) to avoid a special case at 𝑡 = 0 and to match harmonic-series bounds; for 𝑡 ≥ 1 this is equivalent to 𝑂(1/𝑡) up to absolute constants.

- Assumption 15 (Row-stochasticity and diffuse envelope of feedback attention). For every 𝑥 ∈ 𝒳𝑅 and every 𝑡 ≥ 1,

𝛼fb𝑡,𝑗(𝑥) ≥ 0,

𝑡−1

∑

𝑗=0

𝛼fb𝑡,𝑗(𝑥) = 1, 𝛼fb𝑡,𝑗(𝑥) ≤

𝑐2 𝑡

∀𝑗 < 𝑡,

for some constant 𝑐2 = 𝑐2(𝑅) ∈ (0,∞). We set 𝛼fb0,⋅ ≡ 0.

- Assumption 16 (Bounded feedback gain and nontrivial diffuse regime). For every 𝑥 ∈ 𝒳𝑅 and every 𝑡, |𝛾𝑡(𝑥)| ≤ 𝛾max < 1,

and the diffuse feedback mass satisfies

𝜂 ∶= 𝛾max𝑐2 < 1, 𝛽tail ∶= 1 − 𝜂 ∈ (0,1).

- Assumption 17 (Token-wise local feedback gain). On 𝒳𝑅, the feedback gain is token-wise: for each 𝑡 one has 𝛾𝑡(𝑥) = 𝛾(𝑥𝑡). In particular, for 𝜏 < 𝑡,

𝜕𝛾𝑡(𝑥) 𝜕𝑥𝜏

= 0. Assume additionally the token-wise Jacobian is bounded:

∥

𝜕𝛾(𝑥𝑡) 𝜕𝑥𝑡

∥

2

≤ 𝐿𝛾 for all ‖𝑥𝑡‖2 ≤ 𝑅.

- Assumption 18 (Causality of forward branch and routing). For each time 𝑘, the quantities 𝑓𝑘(𝑥), 𝛼fb𝑘,⋅(𝑥), and 𝛾𝑘(𝑥) depend only on the prefix 𝑥0∶𝑘. Equivalently, for any 𝜏 > 𝑘,

𝜕𝑓𝑘(𝑥) 𝜕𝑥𝜏

= 0,

𝜕𝛼fb𝑘,𝑗(𝑥) 𝜕𝑥𝜏

= 0 (∀𝑗 < 𝑘),

𝜕𝛾𝑘(𝑥) 𝜕𝑥𝜏

= 0.

- Assumption 19 (Local, same-token smoothness bounds). There exist finite constants 𝐿𝑓,0 = 𝐿𝑓,0(𝑅) and 𝐿𝛼,0 = 𝐿𝛼,0(𝑅) such that for all 𝑥 ∈ 𝒳𝑅 and all 𝑡,

∥

𝜕𝑓𝑡(𝑥) 𝜕𝑥𝑡

∥

2

≤ 𝐿𝑓,0,

𝑡−1

∑

𝑗=0

∥

𝜕𝛼fb𝑡,𝑗(𝑥) 𝜕𝑥𝑡

∥

2

≤ 𝐿𝛼,0.

- Assumption 20 (Bounded forward sequence). There exists 𝐹𝑅 < ∞ such that ‖𝑓(𝑥)‖∞,2 ≤ 𝐹𝑅 ∀𝑥 ∈ 𝒳𝑅.
- Assumption 21 (Forward-branch dilution of cross-token Jacobians). There exists 𝐿𝑓 = 𝐿𝑓(𝑅) < ∞ such that

for all 𝑥 ∈ 𝒳𝑅, all 𝑡 ≥ 𝜏, and all 𝜏 < 𝑡,

𝐿𝑓 𝑡 + 1

𝜕𝑓𝑡(𝑥) 𝜕𝑥𝜏

∥

∥

≤

.

2

Here ‖ ⋅ ‖2 is the operator norm of the matrix ℝ𝐷 → ℝ𝑟.

- Assumption 22 (Smooth routing: 𝛼-weighted logit sensitivity). Let 𝛼fb𝑡,⋅(𝑥) = softmax(ℶ𝑡,0(𝑥),…,ℶ𝑡,𝑡−1(𝑥)) denote the feedback-attention row at time 𝑡, over 𝑗 < 𝑡, with pre-softmax logits ℶ𝑡,𝑖(𝑥) that may depend on the full prefix 𝑥0∶𝑡. There exists 𝐿route = 𝐿route(𝑅) < ∞ such that for all 𝑥 ∈ 𝒳𝑅 and all 𝑡 > 𝜏 ≥ 0,

𝜕ℶ𝑡,𝑖(𝑥) 𝜕𝑥𝜏

𝐿route 𝑡 + 1

𝑡−1

.

∑

𝛼fb𝑡,𝑖(𝑥)∥

∥

≤

2

𝑖=0

𝜕𝛼fb𝑡,𝑗(𝑥) 𝜕𝑥𝜏

2𝐿route 𝑡 + 1

𝑡−1

∑

∥

∥

≤

.

Consequently, by Lemma B.4,

2

𝑗=0

- Remark B.2 (When Assumption 22 holds). If the feedback query is token-wise, 𝑞𝑡 = 𝑞(𝑥𝑡), then for 𝜏 < 𝑡 the dependence of 𝛼fb𝑡,⋅ on 𝑥𝜏 typically enters only through key-side logits involving 𝑘𝜏, so only a small subset of logits have nonzero 𝜕ℶ𝑡,𝑖/𝜕𝑥𝜏. In that case, Assumption 22 reduces to the corresponding localized logit-sensitivity bound. More generally, if 𝑞𝑡, or other components upstream of logits, has cross-token sensitivity, Assumption 22 requires that the resulting 𝛼fb-weighted logit sensitivities still dilute as 𝑂(1/(𝑡 + 1)) on 𝒳𝑅.

- Lemma B.3 (Bound on the mixer state). Under Assumption 16–20, for all 𝑥 ∈ 𝒳𝑅,

‖𝑠(𝑥)‖∞,2 ≤ 𝑆𝑅 ∶=

𝐹𝑅 1 − 𝛾max

.

Proof. Since each 𝛼fb𝑡,⋅ is a convex distribution and |𝛾𝑡| ≤ 𝛾max, ‖𝑠𝑡‖2 ≤ ‖𝑓𝑡‖2 + 𝛾max max

𝑗<𝑡

‖𝑠𝑗‖2.

A standard induction on max𝑘≤𝑡 ‖𝑠𝑘‖2 yields ‖𝑠‖∞,2 ≤ (1 − 𝛾max)−1‖𝑓‖∞,2 ≤ (1 − 𝛾max)−1𝐹𝑅.

| |
|---|

- Lemma B.4 (Softmax row derivative: total variation bound). Let 𝛼 = softmax(ℶ) ∈ ℝ𝑛 with logits ℶ ∈ ℝ𝑛 depending on a parameter 𝑧. Then

∑

𝑗

∥

𝜕𝛼𝑗 𝜕𝑧

∥ ≤ 2∑

𝑖

𝛼𝑖 ∥

𝜕ℶ𝑖 𝜕𝑧

∥

2

.

Proof. The softmax Jacobian satisfies 𝜕𝛼𝑗/𝜕ℶ𝑖 = 𝛼𝑗(1[𝑗 = 𝑖] − 𝛼𝑖). Thus

𝑛

∑

𝑗=1

∣

𝜕𝛼𝑗 𝜕ℶ𝑖

∣ = 2𝛼𝑖(1 − 𝛼𝑖) ≤ 2𝛼𝑖.

By the chain rule, ∑𝑗 ‖𝜕𝛼𝑗/𝜕𝑧‖ ≤ ∑𝑖(∑𝑗 |𝜕𝛼𝑗/𝜕ℶ𝑖|)‖𝜕ℶ𝑖/𝜕𝑧‖ ≤ 2∑𝑖 𝛼𝑖‖𝜕ℶ𝑖/𝜕𝑧‖.

| |
|---|

- Lemma B.5 (Polynomial tail of the inverse kernel entries). Fix 𝑥 ∈ 𝒳𝑅 and let 𝐾(𝑥) ∶= (𝐼 − 𝐵fb(𝑥))−1. Under Assumptions 15–16, there exists a constant

- B.3 Auxiliary lemmas

𝐶𝐾 ∶= 𝜂𝑒𝜂 = (1 − 𝛽tail)𝑒1−𝛽

tail

such that for all 0 ≤ 𝑘 < 𝑡 ≤ 𝑇 − 1,

|𝐾𝑡,𝑘(𝑥)| ≤ 𝐶𝐾 (𝑡 − 𝑘)−𝛽

, and 𝐾𝑡,𝑡(𝑥) = 1.

Proof. Fix 𝑥 ∈ 𝒳𝑅, and abbreviate 𝐵fb ∶= 𝐵fb(𝑥), 𝛼𝑡,𝑗 ∶= 𝛼fb𝑡,𝑗(𝑥), 𝐾 ∶= 𝐾(𝑥) = (𝐼 − 𝐵fb)−1. Since 𝐵fb is strictly lower-triangular on the finite horizon {0,…,𝑇 − 1}, one has 𝐵fb𝑇 = 0, hence

tail

𝑇−1

𝐾 = (𝐼 − 𝐵fb)−1 =

∑

𝐵fb𝑚.

𝑚=0

Therefore 𝐾 is lower-triangular with unit diagonal:

𝐾𝑡,𝑡 = 1, 𝐾𝑡,𝑘 = 0 for 𝑡 < 𝑘. It remains to prove the off-diagonal estimate. Fix a source index 𝑘 ∈ {0,…,𝑇 − 1}, and define

𝑢𝑡 ∶= |𝐾𝑡,𝑘| (𝑡 ≥ 𝑘). Then 𝑢𝑘 = |𝐾𝑘,𝑘| = 1. Also, since (𝐼 − 𝐵fb)𝐾 = 𝐼, equivalently 𝐾 = 𝐼 + 𝐵fb𝐾, for every 𝑡 > 𝑘 we have

𝐾𝑡,𝑘 = ∑ 𝑗<𝑡

[𝐵fb]𝑡,𝑗 𝐾𝑗,𝑘.

Because 𝐾𝑗,𝑘 = 0 for 𝑗 < 𝑘, this reduces to

𝑡−1

𝑡−1

𝐾𝑡,𝑘 =

∑

[𝐵fb]𝑡,𝑗 𝐾𝑗,𝑘 = 𝛾𝑡(𝑥)

∑

𝛼𝑡,𝑗 𝐾𝑗,𝑘.

𝑗=𝑘

𝑗=𝑘

Taking absolute values and using Assumption 16,

𝑡−1

𝑡−1

𝑢𝑡 ≤ |𝛾𝑡(𝑥)|

∑

𝛼𝑡,𝑗 𝑢𝑗 ≤ 𝛾max

∑

𝛼𝑡,𝑗 𝑢𝑗, 𝑡 > 𝑘. (1)

𝑗=𝑘

𝑗=𝑘

We now compare 𝑢 to an explicit impulse-response sequence. Define (𝑣𝑡(𝑘))𝑡≥0 by

⎧

- 0, 𝑡 < 𝑘,
- 1, 𝑡 = 𝑘,

𝑣𝑡(𝑘) ∶=

⎨ { ⎩

𝑡−1

𝛾max

∑

𝛼̃𝑡,𝑗 𝑣𝑗(𝑘), 𝑡 > 𝑘,

𝑗=0

where the coeﬀicients 𝛼̃𝑡,𝑗 are the following extension of the finite-horizon row weights:

⎧

𝛼𝑡,𝑗, 0 ≤ 𝑗 < 𝑡 ≤ 𝑇 − 1, 0, 𝑡 ≥ 𝑇, 0 ≤ 𝑗 < 𝑡.

𝛼̃𝑡,𝑗 ∶=

⎨{⎩

Then 𝛼̃𝑡,𝑗 ≥ 0, ∑𝑗<𝑡 𝛼̃𝑡,𝑗 ≤ 1 for every 𝑡 ≥ 1, and by Assumption 15,

𝑐2 𝑡

𝛼̃𝑡,𝑗 ≤

(𝑡 ≥ 1, 0 ≤ 𝑗 < 𝑡).

Thus the scalar recursion defining 𝑣(𝑘) satisfies the hypotheses of Corollary E.4 with impulse position 𝑗 = 𝑘, attention envelope constant 𝑐2, and feedback bound 𝛾max. In particular, with

𝜂 ∶= 𝛾max𝑐2, 𝛽tail ∶= 1 − 𝜂 ∈ (0,1), that corollary yields

𝑣𝑡(𝑘) ≤ 𝜂𝑒𝜂 (𝑡 − 𝑘)−𝛽

tail for all 𝑡 > 𝑘. (2)

It remains to show that 𝑢𝑡 ≤ 𝑣𝑡(𝑘) for all 𝑡 ∈ {𝑘,…,𝑇 − 1}. We prove this by induction on 𝑡. For 𝑡 = 𝑘, one has 𝑢𝑘 = 1 = 𝑣𝑘(𝑘). Now let 𝑡 > 𝑘, and assume 𝑢𝑗 ≤ 𝑣𝑗(𝑘) for every 𝑗 ∈ {𝑘,…,𝑡 − 1}. Using (1), the nonnegativity of the coeﬀicients 𝛼𝑡,𝑗, and the induction hypothesis, we obtain

𝑡−1

𝑡−1

𝑢𝑡 ≤ 𝛾max

∑

𝛼𝑡,𝑗 𝑢𝑗 ≤ 𝛾max

∑

𝛼𝑡,𝑗 𝑣𝑗(𝑘).

𝑗=𝑘

𝑗=𝑘

Since 𝑣𝑗(𝑘) = 0 for 𝑗 < 𝑘 and 𝛼̃𝑡,𝑗 = 𝛼𝑡,𝑗 for 𝑡 ≤ 𝑇 − 1, this is exactly

𝑡−1

𝑢𝑡 ≤ 𝛾max

∑

𝛼̃𝑡,𝑗 𝑣𝑗(𝑘) = 𝑣𝑡(𝑘).

𝑗=0

This closes the induction. Combining the comparison 𝑢𝑡 ≤ 𝑣𝑡(𝑘) with (2), we conclude that for every 0 ≤ 𝑘 < 𝑡 ≤ 𝑇 − 1,

|𝐾𝑡,𝑘(𝑥)| = 𝑢𝑡 ≤ 𝑣𝑡(𝑘) ≤ 𝜂𝑒𝜂 (𝑡 − 𝑘)−𝛽

. Thus the claim holds with

tail

. Together with 𝐾𝑡,𝑡 = 1, this proves the lemma.

𝐶𝐾 ∶= 𝜂𝑒𝜂 = (1 − 𝛽tail)𝑒1−𝛽

tail

- Lemma B.6 (A convolution bound). Let 𝛽tail ∈ (0,1). There exists 𝐶𝛽

< ∞ such that for all integers ℓ ≥ 1 and all 𝜏 ≥ 0,

| |
|---|

1 (𝜏 + ℓ − 𝑘)𝛽tail

1 𝑘 + 1

𝜏+ℓ−1

∑

⋅

≤ 𝐶𝛽

ℓ−𝛽

(1 + log(1 + ℓ)).

tail

𝑘=𝜏

tail

tail

2𝛽

𝐶𝛽

∶= 2𝛽

+

.

One may take, for instance,

1 − 𝛽tail

tail

tail

Proof. Write 𝑘 = 𝜏 + 𝑚 where 𝑚 = 0,…,ℓ − 1:

tail

1 (ℓ − 𝑚)𝛽tail

1 𝜏 + 𝑚 + 1

ℓ−1

∑

⋅

.

𝑚=0

Split into 𝑚 ≤ ⌊ℓ/2⌋ and 𝑚 > ⌊ℓ/2⌋.

If 𝑚 ≤ ℓ/2, then (ℓ − 𝑚)−𝛽

≤ (ℓ/2)−𝛽

= 2𝛽

ℓ−𝛽

tail and

1 𝜏 + 𝑚 + 1

𝑑𝑚 𝜏 + 𝑚 + 1

tail

tail

tail

⌊ℓ/2⌋

ℓ/2

∑

≤ 1 + ∫

0

𝑚=0

Thus this part is ≤ 2𝛽

(1 + log(1 + ℓ)). If 𝑚 > ℓ/2, then 𝜏 + 𝑚 + 1 ≥ ℓ/2, so (𝜏 + 𝑚 + 1)−1 ≤ 2/ℓ, hence

ℓ−𝛽

tail

tail

2 ℓ

1 𝑟𝛽tail

1 (ℓ − 𝑚)𝛽tail

1 𝜏 + 𝑚 + 1

2 ℓ

⌊ℓ/2⌋

ℓ/2

∑

∑

(1 + ∫

≤

𝑟−𝛽

⋅

≤

1

𝑟=1

𝑚>ℓ/2

tail

Combine the two bounds.

≤ 1 + log(1 + ℓ).

2 ℓ

𝑑𝑟) ≤

1 1 − 𝛽tail

(

⋅

ℓ 2

1−𝛽tail

)

=

2𝛽

ℓ−𝛽

.

1 − 𝛽tail

tail

tail

| |
|---|

#### B.4 Polynomial Jacobian tail

Theorem 23 (Polynomial Jacobian tail under diffuse routing). Assume Assumptions 15–22, 17, 18, and 19 hold on 𝒳𝑅, and let 𝛽tail ∶= 1 − 𝛾max𝑐2 ∈ (0,1) as in Assumption 16. Then there exists a constant 𝐶(𝑅) < ∞ such that for every 𝑥 ∈ 𝒳𝑅 and every pair 𝜏 < 𝑡 with lag ℓ = 𝑡 − 𝜏 ≥ 1,

𝜕𝑠𝑡(𝑥) 𝜕𝑥𝜏

∥

∥

≤ 𝐶(𝑅)ℓ−𝛽

(1 + log(1 + ℓ)).

2

tail

In particular, long-range sensitivity decays at least polynomially in the lag, up to a logarithmic factor. One may take explicitly

𝐶(𝑅) ∶= 𝐶̃𝐾(𝐴0(𝑅) + (1 + 𝐶𝛽

)𝐴1(𝑅)), 𝐶̃𝐾 ∶= max{1,𝐶𝐾}, 𝐶𝐾 = 𝜂𝑒𝜂, 𝜂 = 𝛾max𝑐2,

where 𝐶𝛽

tail

𝐹𝑅 1 − 𝛾max

is as in Lemma B.6 and

𝐴1(𝑅) ∶= 𝐿𝑓 + 2𝛾max 𝑆𝑅 𝐿route, 𝐴0(𝑅) ∶= 𝐿𝑓,0 + 𝐿𝛾 𝑆𝑅 + 𝛾max 𝑆𝑅 𝐿𝛼,0, 𝑆𝑅 =

.

tail

Proof. Fix 𝑥 ∈ 𝒳𝑅 and a source index 𝜏. Differentiate the solve (36) with respect to 𝑥𝜏:

𝜕𝑠 𝜕𝑥𝜏

𝜕𝐵fb 𝜕𝑥𝜏

𝜕𝑓 𝜕𝑥𝜏

(𝐼 − 𝐵fb)

−

𝑠 =

.

Multiplying by 𝐾 = (𝐼 − 𝐵fb)−1 gives

𝜕𝑠 𝜕𝑥𝜏

𝜕𝑓 𝜕𝑥𝜏

𝜕𝐵fb 𝜕𝑥𝜏

𝑠). (38) Taking the 𝑡-th row and operator norms yields

= 𝐾(

+

𝜕𝑠𝑡 𝜕𝑥𝜏

𝜕𝑓𝑘 𝜕𝑥𝜏

𝜕𝐵fb 𝜕𝑥𝜏

𝑡

∥

∥

≤

∑

|𝐾𝑡,𝑘| ⋅ ∥

+ (

𝑠)𝑘∥

. (39)

2

2

𝑘=0

By Assumption 18, if 𝑘 < 𝜏 then 𝜕𝑓𝑘/𝜕𝑥𝜏 = 0 and 𝜕𝐵fb,𝑘,⋅/𝜕𝑥𝜏 = 0, hence the sum starts at 𝑘 = 𝜏. Bounding the forcing term. We treat the single index 𝑘 = 𝜏 separately from the range 𝑘 > 𝜏.

- Case 1: 𝑘 > 𝜏. For 𝑘 > 𝜏, Assumption 21 gives

∥

𝜕𝑓𝑘 𝜕𝑥𝜏

∥

2

≤

𝐿𝑓 𝑘 + 1

.

It remains to bound ‖(𝜕𝐵fb/𝜕𝑥𝜏)𝑠‖. For 𝑘 > 𝜏 we use the full decomposition

𝜕[𝐵fb]𝑘,𝑗 𝜕𝑥𝜏

=

𝜕𝛾𝑘 𝜕𝑥𝜏

𝛼fb𝑘,𝑗 + 𝛾𝑘

𝜕𝛼fb𝑘,𝑗 𝜕𝑥𝜏

.

By Assumption 17, 𝜕𝛾𝑘/𝜕𝑥𝜏 = 0 for 𝑘 > 𝜏, so only the second term remains. Therefore, using Lemma B.3 and Assumption 22,

∥(

𝜕𝐵fb 𝜕𝑥𝜏

𝑠)𝑘∥

2

≤ |𝛾𝑘|∑ 𝑗<𝑘

∥

𝜕𝛼fb𝑘,𝑗 𝜕𝑥𝜏

∥

2

⋅ ‖𝑠𝑗‖2 ≤ 𝛾max 𝑆𝑅 ∑ 𝑗<𝑘

∥

𝜕𝛼fb𝑘,𝑗 𝜕𝑥𝜏

∥

2

≤ 𝛾max 𝑆𝑅 ⋅

2𝐿route 𝑘 + 1

.

Thus for all 𝑘 > 𝜏,

∥

𝜕𝑓𝑘 𝜕𝑥𝜏

+ (

𝜕𝐵fb 𝜕𝑥𝜏

𝑠)𝑘∥

2

≤

𝐴1(𝑅) 𝑘 + 1

, 𝐴1(𝑅) ∶= 𝐿𝑓 + 2𝛾max 𝑆𝑅 𝐿route.

- Case 2: 𝑘 = 𝜏. Using Assumption 19 and Lemma B.3, we bound

𝜕𝑓 𝜕𝑥𝜏

∥

∥

≤ 𝐿𝑓,0.

2

Moreover, since [𝐵fb]𝜏,𝑗 = 𝛾𝜏 𝛼fb𝜏,𝑗 for 𝑗 < 𝜏,

𝛼fb𝜏,𝑗‖𝑠𝑗‖2 + |𝛾𝜏|∑ 𝑗<𝜏

∥𝜕𝛼

𝜕𝑥𝜏 𝑠)𝜏∥

≤ ∥𝜕𝑥𝜕𝛾

∥

⋅ ∑

𝜕𝑥𝜏 ∥

⋅ ‖𝑠𝑗‖2 ≤ 𝐿𝛾 𝑆𝑅 + 𝛾max 𝐿𝛼,0 𝑆𝑅.

∥(𝜕𝐵

fb 𝜏,𝑗

2

2

2

𝑗<𝜏

𝜏

fb

𝜕𝑓 𝜕𝑥𝜏

𝜕𝐵fb 𝜕𝑥𝜏

∥

+ (

𝑠)𝜏∥

≤ 𝐴0(𝑅), 𝐴0(𝑅) ∶= 𝐿𝑓,0 + 𝐿𝛾 𝑆𝑅 + 𝛾max 𝑆𝑅 𝐿𝛼,0.

Hence

2

Kernel tail and convolution. Plugging the forcing bound into (39) and using Lemma B.5 yields

𝜕𝑠𝑡 𝜕𝑥𝜏

𝐴1(𝑅) 𝑘 + 1

1 𝑡 + 1

1 𝑘 + 1

𝑡

𝑡−1

∥

∥

≤ |𝐾𝑡,𝜏|𝐴0(𝑅) +

∑

|𝐾𝑡,𝑘| ⋅

≤ |𝐾𝑡,𝜏|𝐴0(𝑅) + 𝐴1(𝑅)(

+

∑

𝐶𝐾 (𝑡 − 𝑘)−𝛽

⋅

).

2

𝑘=𝜏+1

𝑘=𝜏+1

tail

Let ℓ = 𝑡 − 𝜏 ≥ 1. We keep the 𝑘 = 𝑡 term explicit and show it can be absorbed into the final tail factor:

1 𝜏 + ℓ + 1

1 ℓ + 1

1 𝑡 + 1

≤

≤

≤ ℓ−1.

Since 𝛽tail ∈ (0,1) and ℓ ≥ 1, we have ℓ1−𝛽

≥ 1, hence ℓ−𝛽

ℓ−1 ≥ ℓ−1. Therefore,

= ℓ1−𝛽

tail

tail

tail

1 𝑡 + 1

(1 + log(1 + ℓ)), (40)

≤ ℓ−1 ≤ ℓ−𝛽

≤ ℓ−𝛽

so the 𝑘 = 𝑡 contribution 𝐴1(𝑅)

𝑡+1 is dominated by the same ℓ−𝛽

(1 + log(1 + ℓ)) envelope, with constant 1.

tail

tail

tail

For the isolated term, Lemma B.5 gives |𝐾𝑡,𝜏| ≤ 𝐶𝐾 ℓ−𝛽

tail. For the remaining sum, apply Lemma B.6 Note that ∑𝑡−1𝑘=𝜏+1 ≤ ∑𝑡−1𝑘=𝜏:

1 𝑘 + 1

𝑡−1

∑

(𝑡 − 𝑘)−𝛽

⋅

≤ 𝐶𝛽

ℓ−𝛽

(1 + log(1 + ℓ)).

𝑘=𝜏

tail

tail

tail

𝜕𝑠𝑡 𝜕𝑥𝜏

Therefore

∥

∥

≤ 𝐶𝐾 𝐴0(𝑅)ℓ−𝛽

+ 𝐴1(𝑅)ℓ−𝛽

(1 + log(1 + ℓ)) + 𝐶𝐾 𝐶𝛽

𝐴1(𝑅)ℓ−𝛽

(1 + log(1 + ℓ)).

2

tail

tail

tail

(1 + log(1 + ℓ)) for ℓ ≥ 1 and 𝐶̃𝐾 = max{1,𝐶𝐾} ≥ 1 and 𝐶̃𝐾 ≥ 𝐶𝐾, we obtain

Since ℓ−𝛽

≤ ℓ−𝛽

tail

𝜕𝑠𝑡 𝜕𝑥𝜏

≤ 𝐶̃𝐾(𝐴0(𝑅) + (1 + 𝐶𝛽

∥

∥

)𝐴1(𝑅))ℓ−𝛽

(1 + log(1 + ℓ)),

tail

tail

2

tail

which is the claim with the stated 𝐶(𝑅).

tail

| |
|---|

- B.5 Jacobian tail for block outputs Consider the simplified block output of the form

𝑦𝑡 = 𝑥𝑡 + 𝑊out (𝑠𝑡 ⊙ 𝑔𝑡) + 𝑏out,

where 𝑔𝑡 = 𝑔𝑡(𝑥𝑡) is token-wise and serves as a gate, and 𝑊out is a fixed matrix. Corollary B.7 (Jacobian tail for block outputs). Under the assumptions of Theorem 23, suppose additionally that ‖𝑔(𝑥)‖∞,2 ≤ 𝐺𝑅 for all 𝑥 ∈ 𝒳𝑅. Then for every 𝜏 < 𝑡 with lag ℓ = 𝑡 − 𝜏 ≥ 1,

𝜕𝑦𝑡(𝑥) 𝜕𝑥𝜏

∥

∥

(1 + log(1 + ℓ)), ∀𝑥 ∈ 𝒳𝑅.

≤ ‖𝑊out‖2 𝐺𝑅 ⋅ 𝐶(𝑅)ℓ−𝛽

2

tail

Proof. For 𝜏 < 𝑡, 𝜕𝑥𝑡/𝜕𝑥𝜏 = 0, and since 𝑔𝑡 is token-wise, 𝜕𝑔𝑡/𝜕𝑥𝜏 = 0. Thus

𝜕𝑦𝑡 𝜕𝑥𝜏

𝜕𝑠𝑡 𝜕𝑥𝜏

= 𝑊out Diag(𝑔𝑡)

.

Taking operator norms and using ‖Diag(𝑔𝑡)‖2 ≤ ‖𝑔𝑡‖2 ≤ 𝐺𝑅 plus Theorem 23 gives the result.

| |
|---|

### C Proofs for Section 4.2

Lemma C.1 (Bounded logit spread implies near-uniform softmax weights). Let ℐ be a finite index set with 𝑛 ∶= |ℐ|, and let (ℶ𝑗)𝑗∈ℐ ⊂ ℝ be logits. Define the softmax weights

𝑒ℶ𝑗 ∑𝑖∈ℐ 𝑒ℶ𝑖

, 𝑗 ∈ ℐ.

𝛼𝑗 =

ℶ𝑖 ≤ Δ0 < ∞, then for every 𝑗 ∈ ℐ,

Δ ∶= max

ℶ𝑖 − min

If the logit spread is bounded by

𝑖∈ℐ

𝑖∈ℐ

𝑒Δ0 𝑛

𝑒−Δ0 𝑛

≤ 𝛼𝑗 ≤

. (41)

Equivalently, for all 𝑖,𝑗 ∈ ℐ one has 𝑒−Δ0 ≤ 𝛼𝑖/𝛼𝑗 ≤ 𝑒Δ0. In particular, if Δ0 is uniformly bounded while 𝑛 grows, then 𝛼𝑗 = Θ(1/𝑛) uniformly over 𝑗 ∈ ℐ.

Proof. Let ℶmin ∶= min𝑖∈ℐ ℶ𝑖. Then ℶmin ≤ ℶ𝑗 ≤ ℶmin + Δ0 for all 𝑗 ∈ ℐ, hence 𝑒ℶ

≤ 𝑒ℶ𝑗 ≤ 𝑒ℶ

min+Δ0 and

𝑛𝑒ℶ

≤ ∑

𝑒ℶ𝑖 ≤ 𝑛𝑒ℶ

min+Δ0.

min

𝑖∈ℐ

min

Dividing 𝑒ℶ𝑗 by these bounds yields (41).

| |
|---|

- Proof of Lemma 4.3. Fix a time 𝑡 and an index 𝜏 < 𝑡. Write

- C.1 Proof of Lemma 4.3

𝛼fwd𝑡,⋅ (𝑥) = softmax(ℶ𝑡,0(𝑥),…,ℶ𝑡,𝑡(𝑥)), 𝛼𝑗 ∶= 𝛼fwd𝑡,𝑗 (𝑥), 𝛽𝑗 ∶= ℶ𝑡,𝑗(𝑥), 0 ≤ 𝑗 ≤ 𝑡.

Thus 𝛼 = softmax(𝛽) ∈ ℝ𝑡+1 and ∑𝑗≤𝑡 𝛼𝑗 = 1. Recall the standard softmax Jacobian identity: for all 𝑗,𝑖 ∈ {0,…,𝑡}, the softmax partial derivatives satisfy

𝜕𝛼𝑗 𝜕𝛽𝑖

= 𝛼𝑗(1[𝑗 = 𝑖] − 𝛼𝑖). (42)

By assumption, for each 𝑗 ≤ 𝑡,

𝛽𝑗 = ℶ𝑡,𝑗(𝑥) = ⟨𝑞(𝑥𝑡), 𝑘(𝑥𝑗)⟩,

where 𝑞,𝑘 are token-wise maps. Since 𝜏 < 𝑡, the quantity 𝑞(𝑥𝑡) depends only on 𝑥𝑡, hence 𝜕𝑞(𝑥𝑡)/𝜕𝑥𝜏 = 0. Similarly, 𝑘(𝑥𝑗) depends only on 𝑥𝑗, hence 𝜕𝑘(𝑥𝑗)/𝜕𝑥𝜏 = 0 unless 𝑗 = 𝜏. Therefore,

𝜕𝛽𝑖 𝜕𝑥𝜏

𝜕𝛽 𝜕𝑥𝜏

= 0 for all 𝑖 ≠ 𝜏, and potentially

≠ 0. (43)

𝜕𝛼𝑗 𝜕𝑥𝜏

𝜕𝛼𝑗 𝜕𝛽𝑖

𝜕𝛼𝑗 𝜕𝛽𝜏

𝜕𝛽𝑖 𝜕𝑥𝜏

𝜕𝛽 𝜕𝑥𝜏

𝜕𝛽 𝜕𝑥𝜏

Consequently, by the chain rule and (43),

= ∑

=

= 𝛼𝑗(1[𝑗 = 𝜏] − 𝛼𝜏)

,

𝑖≤𝑡

𝜕𝛼𝑗 𝜕𝑥𝜏

𝜕𝛽 𝜕𝑥𝜏

where we used (42) in the last step. Taking operator norms gives

∥

∥

= ∣𝛼𝑗(1[𝑗 = 𝜏] − 𝛼𝜏)∣ ⋅ ∥

∥

. (44)

2

2

Summing (44) over 𝑗 ≤ 𝑡 yields

𝜕𝛼𝑗 𝜕𝑥𝜏

𝜕𝛽 𝜕𝑥𝜏

∥

= (∑

∣𝛼𝑗(1[𝑗 = 𝜏] − 𝛼𝜏)∣)∥

∥

.

∑

∥

2

2

𝑗≤𝑡

𝑗≤𝑡

𝛼𝑗𝛼𝜏 ⏟ 𝑗≠𝜏

= 𝛼𝜏(1 − 𝛼𝜏) + 𝛼𝜏 ∑ 𝑗≠𝜏

𝛼𝑗 = 2𝛼𝜏(1 − 𝛼𝜏) ≤ 2𝛼𝜏,

+ ∑

To evaluate the scalar sum, note that ∑

∣𝛼𝑗(1[𝑗 = 𝜏] − 𝛼𝜏)∣ = 𝛼⏟⏟⏟⏟⏟𝜏(1 − 𝛼𝜏)

𝑗≤𝑡

𝑗≠𝜏

𝑗=𝜏

since ∑𝑗≠𝜏 𝛼𝑗 = 1 − 𝛼𝜏 and 1 − 𝛼𝜏 ≤ 1. Therefore,

𝜕𝛼fwd𝑡,𝑗 (𝑥) 𝜕𝑥𝜏

𝜕ℶ𝑡,𝜏(𝑥) 𝜕𝑥𝜏

∑

∥

∥

∥

≤ 2𝛼fwd𝑡,𝜏 (𝑥)∥

,

2

2

𝑗≤𝑡

which is the first claim.

In particular. If ‖𝜕ℶ𝑡,𝜏(𝑥)/𝜕𝑥𝜏‖2 ≤ 𝐿ℶ on 𝒳𝑅, then

𝜕𝛼fwd𝑡,𝑗 (𝑥) 𝜕𝑥𝜏

∑

∥

∥

≤ 2𝐿ℶ 𝛼fwd𝑡,𝜏 (𝑥).

2

𝑗≤𝑡

In the diffuse regime of Definition 4, Lemma C.1 implies 𝛼fwd𝑡,𝜏 (𝑥) = Θ(1/|𝒲𝑡|) uniformly over 𝜏 ∈ 𝒲𝑡, hence the right-hand side is ≲ 1/|𝒲𝑡|. For full-prefix attention |𝒲𝑡| = 𝑡 + 1.

| |
|---|

- C.2 Proof of Proposition 9 Proof of Proposition 9. Fix a horizon 𝑇 and work with the fixed-routing Jacobians from Section 4.2.1.

- (1) Transformer: attention one-hop dilution. By definition of the value influence Jacobian under realized attention weights, by Eq. (26),

𝐽𝑡,𝜏attn =

𝜕𝑦𝑡 𝜕𝑣𝜏

∣

𝛼fwd

= 𝛼fwd𝑡,𝜏 𝐼. Taking operator norms and using ‖𝐼‖ = 1 gives

‖𝐽𝑡,𝜏attn‖ = ‖𝛼fwd𝑡,𝜏 𝐼‖ = 𝛼fwd𝑡,𝜏 .

Assume the shared diffuse (low-separation) regime of Definition 4 with full-prefix visibility 𝒲𝑡 = {0,…,𝑡}, so |𝒲𝑡| = 𝑡 + 1. The bounded logit spread over 𝒲𝑡 implies, by Lemma C.1, that for every 𝜏 ≤ 𝑡,

𝑒−Δ 𝑡 + 1

≤ 𝛼fwd𝑡,𝜏 ≤

𝑒Δ 𝑡 + 1

,

hence 𝛼fwd𝑡,𝜏 = Θ(1/(𝑡 + 1)) and therefore

‖𝐽𝑡,𝜏attn‖ = Θ(

1 𝑡 + 1

) (𝜏 ≤ 𝑡).

For a fixed old source 𝜏 = 𝑂(1) and lag ℓ = 𝑡 − 𝜏, we have

‖𝐽𝜏+ℓ,𝜏attn ‖ = 𝛼fwd𝜏+ℓ,𝜏 = Θ(

1 𝜏 + ℓ + 1

) = Θ(1/ℓ),

since 𝜏 is fixed and ℓ → ∞.

- (2) Mamba under failed freeze time. By definition of the fixed-routing impulse Jacobian for an SSM, by Eq. (28),

𝑡

𝐽𝑡,𝜏ssm = 𝐶ssm,𝑡(

∏

𝐴ssm,𝑟)𝐵ssm,𝜏, 0 ≤ 𝜏 ≤ 𝑡.

𝑟=𝜏+1

𝐴ssm,𝑟 = diag(exp(−𝑎𝑛Δ𝑟)), 𝑎𝑛 ≥ 𝜆 > 0, and bounded input/output factors

Assume the realized recurrence has diagonal transitions

‖𝐵ssm,𝑟‖ ≤ 𝐵max, sup

‖𝐶ssm,𝑟‖ ≤ 𝐶max.

𝑟

𝑟

sup

𝑡

𝑡

𝑡

Δ𝑟). Under the failed-freeze-time condition

∥

∏

𝐴ssm,𝑟∥ = max

exp( − 𝑎𝑛

∑

Δ𝑟) ≤ exp( − 𝜆

∑

Then

𝑛

𝑟=𝜏+1

𝑟=𝜏+1

𝑟=𝜏+1

𝑡

Δ𝑟 ≥ 𝑐Δ(𝑡 − 𝜏), it follows that

∑

𝑟=𝜏+1

‖𝐽𝑡,𝜏ssm‖ ≤ 𝐶max𝐵max exp( − 𝜆𝑐Δ(𝑡 − 𝜏)). Setting 𝑐 ∶= 𝐶max𝐵max and ℓ ∶= 𝑡 − 𝜏 gives

‖𝐽𝑡,𝜏ssm‖ ≤ 𝑐 𝑒−𝜆𝑐Δℓ.

- (3) Sessa: diffuse feedback routing. For a realized feedback matrix 𝐵fb, the solve Jacobian is the resolvent given by Eq. (27)

𝐽sessa = (𝐼 − 𝐵fb)−1, 𝐽𝑡,𝜏sessa = [(𝐼 − 𝐵fb)−1]𝑡,𝜏. Since 𝐵fb is scalar-valued, 𝐽𝑡,𝜏sessa ∈ ℝ is a scalar coeﬀicient shared across features.

Fix 𝜏 and consider the impulse in the forward stream 𝑓 at time 𝜏: 𝑓𝜏 = 1 and 𝑓𝑡 = 0 for 𝑡 ≠ 𝜏. Let 𝑠 be the solution to (𝐼 − 𝐵fb)𝑠 = 𝑓. By linearity, 𝑠𝑡 = 𝐽𝑡,𝜏sessa for all 𝑡. Moreover, by forward substitution (equivalently (31)), 𝑠𝜏 = 1 and for 𝑡 > 𝜏,

𝑡−1

𝑡−1

𝑠𝑡 = 𝑓𝑡 + 𝛾𝑡

∑

𝛼fb𝑡,𝑗𝑠𝑗 = 𝛾𝑡

∑

𝛼fb𝑡,𝑗𝑠𝑗,

𝑗=𝜏

𝑗=0

since 𝑓𝑡 = 0 for 𝑡 ≠ 𝜏 and 𝑠𝑗 = 0 for 𝑗 < 𝜏 in a strictly causal solve.

Under Assumptions 6–7 we have 𝛼fb𝑡,𝑗 ≤ 𝑐2/𝑡 for all 𝑗 < 𝑡 and |𝛾𝑡| ≤ 𝛾max < 1, and defining 𝛽tail ∶= 1−𝛾max𝑐2 ∈ (0,1], with 𝛾max𝑐2 < 1, Theorem 8 applies to this impulse recursion, shifted to start at 𝜏, and yields that for all lags ℓ ≥ 1,

, for an explicit constant 𝐶, e.g. 𝐶 = (1 − 𝛽tail)𝑒1−𝛽

|𝐽𝜏+ℓ,𝜏sessa | = |𝑠𝜏+ℓ| ≤ 𝐶 ℓ−𝛽

tail

tail.

⎧ ⎨{⎩

- 0, 𝑡 = 0, 𝛾 𝑡

- 1[𝑗 < 𝑡], 𝑡 ≥ 1,

Tightness. In the explicit uniform-routing regime

[𝐵fb]𝑡,𝑗 =

𝛾 ∈ (0,1),

one has 𝛼fb𝑡,𝑗 = 𝑡−11[𝑗 < 𝑡] and constant gain 𝛾𝑡 ≡ 𝛾, hence 𝛽tail = 1 − 𝛾. Appendix Corollary F.2 gives, for every fixed source position 𝜏,

|𝐽𝜏+ℓ,𝜏sessa | = Θ𝜏(ℓ−𝛽

).

Moreover, Appendix Corollary F.3 yields the stronger uniform statement that for every 𝜏max < ∞ there exist constants 𝑐𝜏−

tail

,𝑐𝜏+

> 0 such that

𝑐𝜏−

ℓ−𝛽

≤ |𝐽𝜏+ℓ,𝜏sessa | ≤ 𝑐𝜏+

ℓ−𝛽

for all 0 ≤ 𝜏 ≤ 𝜏max and all ℓ ≥ 1. Thus the one-layer envelope is tight for each fixed source and uniformly on every bounded source family, in particular on every fixed finite horizon.

max

max

tail

tail

max

max

| |
|---|

#### C.3 Proof of Proposition 3

Proof. The claim is about the input–output map and is independent of the chosen realization. By the controllable and observable decomposition, also known as the Kalman decomposition (Antsaklis and Michel, 2006), there exists

a similarity transform that isolates the controllable and observable subsystem (𝐴ssm,co,𝐵ssm,co,𝐶ssm,co) such that for all ℓ ≥ 0,

𝐶ssm𝐴ℓssm𝐵ssm = 𝐶ssm,co 𝐴ℓssm,co 𝐵ssm,co. Moreover, (𝐴ssm,co,𝐵ssm,co,𝐶ssm,co) is a minimal realization of the same transfer function, so it admits no pole–zero cancellations and its poles coincide with the reachable and observable eigenvalues of 𝐴ssm,co (Dahleh et al., 2011b). Since the transfer function is BIBO stable, all its poles lie strictly inside the unit disk (DT case) (Dahleh et al., 2011a); hence 𝜌spec(𝐴ssm,co) < 1. It follows from standard finite-dimensional matrix power bounds that there exist 𝑐 > 0 and 𝜅 ∈ (0,1) such that ‖𝐴ℓssm,co‖ ≤ 𝑐 𝜅ℓ for all ℓ, and therefore ‖𝐶ssm𝐴ℓssm𝐵ssm‖ = ‖𝐶ssm,co𝐴ℓssm,co𝐵ssm,co‖ ≤ 𝑐′𝜅ℓ.

| |
|---|

#### C.4 Proof of Proposition 4

The key point is that, under ZOH discretization, the state-transition product is controlled by the accumulated discretization time

𝑡

∑

Δ𝑟(𝑥),

𝑟=𝜏+1

since each channel contributes a factor exp(−𝑎𝑛Δ𝑟(𝑥)). Accordingly, the proof first obtains an end-to-end Jacobian bound in terms of

𝑡

Δ𝑟(𝑥)), and only then converts this into exponential-in-lag decay under failed freeze time.

Π𝑡,ℓ(𝑥) = exp( − 𝜆

∑

𝑟=𝜏+1

Proof. Fix 𝑥 ∈ 𝒳𝑅 and indices 𝜏 < 𝑡, and set ℓ ∶= 𝑡 − 𝜏 ≥ 1. Write 𝐽𝑡,𝜏ℎ ∶= 𝜕ℎ𝑡(𝑥)/𝜕𝑥𝜏 and 𝐽𝑡,𝜏e2e ∶= 𝜕𝑦𝑡(𝑥)/𝜕𝑥𝜏. We use the product convention

𝑡

𝑡

∏

∏

𝐴ssm,𝑟 ∶= 𝐴ssm,𝑡𝐴ssm,𝑡−1 ⋯𝐴ssm,𝜏+1,

(⋅) ∶= 𝐼.

𝑟=𝜏+1

𝑟=𝑡+1

State bound via ZOH convexity. In a ZOH-diagonal channel, each mode 𝑛 evolves as the scalar recursion

1 − 𝑒−𝑎𝑛Δ𝑡 𝑎𝑛

(ℎ𝑡)𝑛 = 𝑒−𝑎𝑛Δ𝑡 (ℎ𝑡−1)𝑛 +

(𝑏𝑡)𝑛, 𝑎𝑛 ≥ 𝜆, Δ𝑡 ≥ 0,

𝑏𝑡 ∶= 𝐵̃ssm,𝑡(𝑥𝑡)𝑢𝑡(𝑥𝑡). By the bounds on 𝐵̃ssm,𝑡 and 𝑢𝑡 on 𝒳𝑅, we have

where we take

‖𝑏𝑡‖ ≤ 𝐺max𝑈𝑅,

and hence |(𝑏𝑡)𝑛| ≤ 𝐺max𝑈𝑅 for each mode. Since ℎ−1 = 0, Lemma 4.4 applied componentwise with 𝑎min = 𝜆 gives

𝐺max𝑈𝑅 𝜆

for every mode 𝑛. Therefore

|(ℎ𝑡)𝑛| ≤

𝑡

sup

𝐺max𝑈𝑅 𝜆

‖ℎ𝑡‖2 ≤√𝑑state ‖ℎ𝑡‖∞ ≤√𝑑state

=∶ 𝐻𝑅.

Jacobian recursion for 𝑡 > 𝜏. For 𝑡 > 𝜏, locality implies

𝜕𝐵̃ssm,𝑡(𝑥𝑡) 𝜕𝑥𝜏

𝜕𝐴ssm,𝑡(𝑥𝑡) 𝜕𝑥𝜏

𝜕𝐺ssm,𝑡(𝑥𝑡) 𝜕𝑥𝜏

𝜕𝑢𝑡(𝑥𝑡) 𝜕𝑥𝜏

=

=

=

= 0.

ℎ𝑡 = 𝐴ssm,𝑡(𝑥𝑡)ℎ𝑡−1 + 𝐺ssm,𝑡(𝑥𝑡)𝐵̃ssm,𝑡(𝑥𝑡)𝑢𝑡(𝑥𝑡) with respect to 𝑥𝜏 yields

Differentiating

𝐽𝑡,𝜏ℎ = 𝐴ssm,𝑡(𝑥𝑡)𝐽𝑡−1,𝜏ℎ , 𝑡 > 𝜏. Iterating gives

𝑡

∏

𝐽𝑡,𝜏ℎ = (

𝐴ssm,𝑟(𝑥𝑟))𝐽𝜏,𝜏ℎ .

𝑟=𝜏+1

Source-time derivative bound. At 𝑡 = 𝜏, write 𝑏𝜏 ∶= 𝐵̃ssm,𝜏(𝑥𝜏)𝑢𝜏(𝑥𝜏) and differentiate the ZOH update:

𝜕𝐴ssm,𝜏(𝑥𝜏) 𝜕𝑥𝜏

𝜕𝐺ssm,𝜏(𝑥𝜏) 𝜕𝑥𝜏

𝜕𝑏 𝜕𝑥𝜏

)ℎ𝜏−1 + (

)𝑏𝜏 + 𝐺ssm,𝜏(𝑥𝜏)

𝐽𝜏,𝜏ℎ = (

.

𝜕𝐵̃ssm,𝜏(𝑥𝜏) 𝜕𝑥𝜏

𝜕𝑏 𝜕𝑥𝜏

𝜕𝑢𝜏(𝑥𝜏) 𝜕𝑥𝜏

)𝑢𝜏 + 𝐵̃ssm,𝜏(𝑥𝜏)(

). Since

= (

Moreover,

1 − [𝐴ssm,𝜏(𝑥𝜏)]𝑛 𝑎𝑛

, we have the operator bounds

)

𝐺ssm,𝜏(𝑥𝜏) = diag(

𝑛

𝜕𝐴ssm,𝜏(𝑥𝜏) 𝜕𝑥𝜏

𝜕𝐺ssm,𝜏(𝑥𝜏) 𝜕𝑥𝜏

1 𝜆

1 𝜆

∥ ≤

∥

∥.

‖𝐺ssm,𝜏(𝑥𝜏)‖ ≤

, ∥

‖ℎ𝜏−1‖ ≤ 𝐻𝑅, ‖𝑏𝜏‖ ≤ 𝐺max𝑈𝑅, together with the derivative bounds gives

Using

𝐿𝐴 𝜆

1 𝜆

(𝐿𝐵 𝑈𝑅 + 𝐺max 𝐿𝑢) =∶ 𝐽𝑅.

‖𝐽𝜏,𝜏ℎ ‖ ≤ 𝐿𝐴 𝐻𝑅 +

𝐺max𝑈𝑅 +

Transition product bound by accumulated discretization time. Since each 𝐴ssm,𝑟 is diagonal with entries exp(−𝑎𝑛Δ𝑟) and 𝑎𝑛 ≥ 𝜆,

𝑡

𝑡

𝑡

∥

∏

𝐴ssm,𝑟(𝑥𝑟)∥ = max

exp( − 𝑎𝑛

∑

Δ𝑟(𝑥)) ≤ exp( − 𝜆

∑

Δ𝑟(𝑥)) =∶ Π𝑡,ℓ(𝑥).

𝑛

𝑟=𝜏+1

𝑟=𝜏+1

𝑟=𝜏+1

‖𝐽𝑡,𝜏ℎ ‖ ≤ Π𝑡,ℓ(𝑥)‖𝐽𝜏,𝜏ℎ ‖ ≤ 𝐽𝑅 Π𝑡,ℓ(𝑥).

Therefore

Output Jacobian. For 𝜏 < 𝑡, locality implies 𝜕𝐶ssm,𝑡(𝑥𝑡)/𝜕𝑥𝜏 = 0, so

𝜕𝑦𝑡 𝜕𝑥𝜏

= 𝐶ssm,𝑡(𝑥𝑡)𝐽𝑡,𝜏ℎ .

𝜕𝑦𝑡(𝑥) 𝜕𝑥𝜏

∥ ≤ ‖𝐶ssm,𝑡(𝑥𝑡)‖‖𝐽𝑡,𝜏ℎ ‖ ≤ 𝐶𝑅 𝐽𝑅 Π𝑡,ℓ(𝑥). Thus the claim holds with

∥

Hence

𝐶(𝑅) ∶= 𝐶𝑅𝐽𝑅.

| |
|---|

- Proof of Lemma 4.4. Fix 𝑡 ≥ 0 and define 𝜃𝑡 ∶= 𝑒−𝑎Δ𝑡 ∈ [0,1], since 𝑎 > 0 and Δ𝑡 ≥ 0. Then 1−𝜃𝑡 = 1−𝑒−𝑎Δ𝑡 ∈ [0,1], and the update can be rewritten as

#### C.5 Proof of Lemma 4.4

𝑏𝑡 𝑎

ℎ𝑡 = 𝜃𝑡 ℎ𝑡−1 + (1 − 𝜃𝑡)

.

|𝑏𝑡| 𝑎

Taking absolute values and using the triangle inequality yields

|ℎ𝑡| ≤ 𝜃𝑡|ℎ𝑡−1| + (1 − 𝜃𝑡)

.

Since 𝜃𝑡 ∈ [0,1], for any 𝑢,𝑣 ≥ 0 one has 𝜃𝑡𝑢 + (1 − 𝜃𝑡)𝑣 ≤ max{𝑢,𝑣}, hence

|𝑏𝑡| 𝑎

|𝑏𝑡| 𝑎min

|ℎ𝑡| ≤ max{|ℎ𝑡−1|,

} ≤ max{|ℎ𝑡−1|,

},

using 𝑎 ≥ 𝑎min. Define

|𝑏𝑠| 𝑎min

𝐵𝑡 ∶= max{|ℎ−1|, max 0≤𝑠≤𝑡

}.

We claim by induction that |ℎ𝑡| ≤ 𝐵𝑡 for all 𝑡 ≥ 0. For 𝑡 = 0 this follows from the previous inequality. If |ℎ𝑡−1| ≤ 𝐵𝑡−1, then

|𝑏𝑡| 𝑎min

|𝑏𝑡| 𝑎min

|ℎ𝑡| ≤ max{|ℎ𝑡−1|,

} = 𝐵𝑡, proving the induction. Taking sup𝑡≥0 gives

} ≤ max{𝐵𝑡−1,

|𝑏𝑠| 𝑎min

|ℎ𝑡| ≤ max{|ℎ−1|, sup 𝑠≥0

},

𝑡≥0

sup

which is the general bound. If additionally |𝑏𝑡| ≤ 𝑀 for all 𝑡 and ℎ−1 = 0, then the right-hand side is at most 𝑀/𝑎min, proving sup𝑡 |ℎ𝑡| ≤ 𝑀/𝑎min.

- Remark C.2 (Vector and diagonal case). For diagonal 𝐴 = −diag(𝑎𝑛) with min𝑛 𝑎𝑛 ≥ 𝑎min, the bound holds componentwise for each mode and channel, and hence yields the uniform bound ‖ℎ𝑡‖∞ ≤ sup𝑠 ‖𝑏𝑠‖∞/𝑎min. More generally, for any monotone norm ‖ ⋅ ‖ one has ‖ℎ𝑡‖ ≤ ‖1‖ sup𝑠 ‖𝑏𝑠‖∞/𝑎min.

| |
|---|

#### C.6 Proof of Corollary 4.6 Proof. Proposition 4 gives

𝜕𝑦𝑡(𝑥) 𝜕𝑥𝜏

𝑡

∥

∥ ≤ 𝐶(𝑅)Π𝑡,ℓ(𝑥), Π𝑡,ℓ(𝑥) = exp( − 𝜆

∑

###### Δ𝑟(𝑥)).

𝑟=𝜏+1

𝑡

∑

Δ𝑟(𝑥) ≥ 𝑐Δ(𝑡 − 𝜏).

Under failed freeze time,

𝑟=𝜏+1

Π𝑡,ℓ(𝑥) ≤ exp( − 𝜆𝑐Δ(𝑡 − 𝜏)), and therefore

Applying Proposition 5 yields

𝜕𝑦𝑡(𝑥) 𝜕𝑥𝜏

∥

∥ ≤ 𝐶(𝑅)exp( − 𝜆𝑐Δ(𝑡 − 𝜏)).

- Remark C.3 (Local windows). If 𝐴ssm,𝑡,𝐵̃ssm,𝑡,𝐶ssm,𝑡,𝑢𝑡 depend on a fixed window 𝑥𝑡−𝐾∶𝑡, the same argument yields

| |
|---|

𝜕𝑦𝑡 𝜕𝑥𝜏

𝑡

∥

∥ ≤ 𝐶(𝑅) exp( − 𝜆

∑

Δ𝑟(𝑥)) (𝑡 > 𝜏 + 𝐾),

𝑟=𝜏+𝐾+1

so the same failed-freeze-time conclusion holds up to a finite-window slack.

#### C.7 Proof of Proposition 5 Proof. By definition,

𝑡

Δ𝑟). Under the failed-freeze-time condition

Π𝑡,ℓ = exp( − 𝜆

∑

𝑟=𝜏+1

𝑡

∑

Δ𝑟 ≥ 𝑐Δ(𝑡 − 𝜏) = 𝑐Δℓ,

𝑟=𝜏+1

Π𝑡,ℓ ≤ exp( − 𝜆𝑐Δℓ). This is exactly the claim.

we obtain

| |
|---|

𝛼𝑡,𝑗(𝑥)𝑣(𝑥𝑗). For 𝜏 < 𝑡, differentiate:

##### Proof. (1) Transformer attention in the no-freeze setting. Let 𝑦𝑡(𝑥) = ∑𝑗∈𝒲

#### C.8 Details for Proposition 10

𝜕𝛼𝑡,𝑗(𝑥) 𝜕𝑥𝜏

𝜕𝑦𝑡 𝜕𝑥𝜏

𝜕𝑣(𝑥𝜏) 𝜕𝑥𝜏

𝑡

= 𝛼𝑡,𝜏

+ ∑

𝑣(𝑥𝑗).

𝑗∈𝒲𝑡

Taking operator norms and using ‖𝜕𝑣(𝑥𝜏)/𝜕𝑥𝜏‖ ≤ 𝐿𝑣 and ‖𝑣(𝑥𝑗)‖ ≤ 𝑉𝑅 yields

𝜕𝛼𝑡,𝑗 𝜕𝑥𝜏

𝜕𝑦𝑡 𝜕𝑥𝜏

∥

∥ ≤ 𝛼𝑡,𝜏𝐿𝑣 + 𝑉𝑅 ∑ 𝑗∈𝒲𝑡

∥

∥.

Under the shared regime in Section 4.2.2, 𝛼𝑡,𝜏 ≤ 𝑐2/|𝒲𝑡| and ∑𝑗∈𝒲

‖𝜕𝛼𝑡,𝑗/𝜕𝑥𝜏‖ ≤ 𝐿𝛼/|𝒲𝑡|, hence ‖𝜕𝑦𝑡/𝜕𝑥𝜏‖ ≲ 1/|𝒲𝑡|. For full-prefix attention |𝒲𝑡| = 𝑡 + 1, recovering ‖𝜕𝑦𝑡/𝜕𝑥𝜏‖ ≲ 1/(𝑡 + 1).

𝑡

(2) Mamba under failed freeze time. Item (2) follows by combining Proposition 4 with failed freeze time, namely

𝑡

∑

###### Δ𝑟(𝑥) ≥ 𝑐Δ(𝑡 − 𝜏),

𝑟=𝜏+1

### D BIBO stability on infinite horizons and uniform-in-𝑇 bounds

that is, by Corollary 4.6.

| |
|---|

We extend the finite-horizon BIBO statement to infinite sequences under an explicit row-contraction condition, and to uniform-in-𝑇 bounds for truncated length-𝑇 networks without appealing to compactness.

- D.1 Sequence norms and stability definition We use the norm ‖ ⋅ ‖∞,2 and balls from Definition 7. For finite tensors we also use the comparison (35).
- D.2 Feedback matrix and row-contraction condition

Fix a causal width-𝑚 Sessa block 𝐺 as in Section 3.1, but now acting on infinite sequences in ℓ∞(ℕ,ℝ𝑚). We emphasize that the block input and output live in ℝ𝑚, while the triangular solve (𝐼 − 𝐵fb)𝑠 = 𝑓 is performed in a value space ℝ𝑟: in our definition, 𝑠𝑡 ∈ ℝ𝑟, 𝑓𝑡 ∈ ℝ𝑟, 𝑔𝑡 ∈ ℝ𝑟, and 𝑧𝑡 = 𝑠𝑡 ⊙ 𝑔𝑡 ∈ ℝ𝑟, and the output projection is token-wise aﬀine 𝑜 ∶ ℝ𝑟 → ℝ𝑚.

Causal feedback-attention weights. For each input 𝑥, the masked softmax in the feedback branch defines strictly lower-triangular weights (𝛼fb𝑡𝜏(𝑥))𝑡,𝜏≥0 with

𝛼fb𝑡𝜏(𝑥) ≥ 0, 𝛼fb𝑡𝜏(𝑥) = 0 for 𝜏 ≥ 𝑡, ∑ 𝜏<𝑡

𝛼fb𝑡𝜏(𝑥) = 1 for 𝑡 ≥ 1, (45)

with the empty sum = 0 for 𝑡 = 0. These properties hold as follows: for 𝑡 ≥ 1 each row 𝑡 is a softmax over the finite set {0,…,𝑡 − 1}, hence 𝛼fb𝑡𝜏 ≥ 0 and ∑𝜏<𝑡 𝛼fb𝑡𝜏 = 1; for 𝑡 = 0 we set 𝛼fb0𝜏 = 0 for all 𝜏, i.e. the context is empty, so the empty sum equals 0.

##### Feedback attention matrix. Define Αfb(𝑥) ∶= (𝛼fb𝑡𝜏(𝑥))𝑡,𝜏≥0.

Feedback coeﬀicient and the Sessa matrix 𝐵fb. By definition of the Sessa block, the feedback coeﬀicient is

𝛾𝑡(𝑥) = tanh(𝑢𝑡(𝑥)) ∈ (−1,1), computed token-wise from the block input, via aﬀine maps and element-wise nonlinearities. Define the diagonal operator Γfb(𝑥) ∶= diag(𝛾𝑡(𝑥))𝑡≥0 and the strictly lower-triangular matrix

𝐵fb(𝑥) ∶= Γfb(𝑥) Αfb(𝑥) ⟺ [𝐵fb]𝑡,𝜏(𝑥) = 𝛾𝑡(𝑥)𝛼fb𝑡𝜏(𝑥). (46)

Assumption 24 (Uniform feedback margin and row contraction). For every radius 𝑅 ≥ 0 there exists 𝜌(𝑅) ∈ [0,1) such that for all inputs 𝑥 ∈ ℓ∞(ℕ,ℝ𝑚) with ‖𝑥‖∞,2 ≤ 𝑅,

|𝛾𝑡(𝑥)| ≤ 𝜌(𝑅). (47)

𝑡≥0

sup

In particular, using (45)–(46), for every 𝑥, sup

∑

|[𝐵fb]𝑡,𝜏(𝑥)| = sup 𝑡≥1

∑

|[𝐵fb]𝑡,𝜏(𝑥)| = sup 𝑡≥1

|𝛾𝑡(𝑥)| ≤ sup 𝑡≥0

|𝛾𝑡(𝑥)| ≤ 𝜌(𝑅) < 1. (⋆)

𝑡≥0

𝜏<𝑡

𝜏<𝑡

- Remark D.1 (An explicit choice of ￿(R)). If 𝑢𝑡(𝑥) is produced by a token-wise feedforward stack of aﬀine maps and element-wise nonlinearities 𝜎 satisfying |𝜎(𝑧)| ≤ |𝑧| coordinate-wise; this holds for GELU. Aﬀine and linear maps

are handled separately via spectral norms as in Lemma D.2. Then for some explicit constants 𝑐𝛾 ≥ 0, 𝐿𝛾,pre ≥ 0 depending only on the block parameters,

|𝑢𝑡(𝑥)| ≤ 𝑐𝛾 + 𝐿𝛾,pre‖𝑥‖∞,2. (48)

𝑡≥0

sup

Hence on the ball ‖𝑥‖∞,2 ≤ 𝑅 one can take

𝜌(𝑅) ∶= tanh(𝑐𝛾 + 𝐿𝛾,pre𝑅) < 1. (49) The strict inequality holds since 𝑐𝛾 + 𝐿𝛾,pre𝑅 < ∞ and tanh(⋅) < 1 for finite arguments.

#### D.3 Causal triangular solve on ℓ∞

The only operation that truly changes nature at 𝑇 = ∞ is the lower-triangular solve. We treat it as a causal linear system.

Proof. Let 𝐵fb = ([𝐵fb]𝑡,𝜏)𝑡,𝜏≥0 be strictly lower-triangular and define the causal operator (𝐵fb𝑠)𝑡 ∶= ∑𝜏<𝑡 [𝐵fb]𝑡,𝜏𝑠𝜏,

#### D.4 Proof of Lemma 4.2

a finite sum for each fixed 𝑡, acting on ℝ𝑟-valued sequences. Here [𝐵fb]𝑡,𝜏 ∈ ℝ is scalar and multiplies 𝑠𝜏 ∈ ℝ𝑟, i.e. scalar–vector multiplication. Assume

∑

|[𝐵fb]𝑡,𝜏| ≤ 𝜌 < 1.

𝑡≥0

𝜏<𝑡

sup

Then for every bounded input 𝑓 ∈ ℓ∞(ℕ,ℝ𝑟) there exists a unique bounded solution 𝑠 ∈ ℓ∞(ℕ,ℝ𝑟) to

𝑠 = 𝑓 + 𝐵fb𝑠 equivalently, (𝐼 − 𝐵fb)𝑠 = 𝑓, and it satisfies the explicit bound

1 1 − 𝜌

‖𝑠‖∞,2 ≤

‖𝑓‖∞,2. (50)

Existence and uniqueness follow by forward substitution: for 𝑡 = 0, 𝑠0 = 𝑓0; for 𝑡 ≥ 1,

𝑠𝑡 = 𝑓𝑡 + ∑ 𝜏<𝑡

[𝐵fb]𝑡,𝜏𝑠𝜏

depends only on previously defined (𝑠𝜏)𝜏<𝑡. Thus a unique sequence 𝑠 exists. For the bound, define the partial maxima

𝑀𝑡 ∶= max 0≤𝑘≤𝑡

‖𝑠𝑘‖2 (𝑡 ≥ 0).

For 𝑡 = 0 we have 𝑠0 = 𝑓0, hence 𝑀0 = ‖𝑠0‖2 ≤ ‖𝑓‖∞,2. For 𝑡 ≥ 1, using the row-sum estimate and 𝑀𝑡−1 ≥ ‖𝑠𝜏‖2 for all 𝜏 < 𝑡,

|[𝐵fb]𝑡,𝜏|‖𝑠𝜏‖2 ≤ ‖𝑓‖∞,2 + 𝜌 𝑀𝑡−1. We now prove by induction that for all 𝑡 ≥ 0,

‖𝑠𝑡‖2 ≤ ‖𝑓𝑡‖2 + ∑ 𝜏<𝑡

1 1 − 𝜌

‖𝑓‖∞,2.

𝑀𝑡 ≤

The base case 𝑡 = 0 holds since 𝑀0 ≤ ‖𝑓‖∞,2 ≤ 1−𝜌1 ‖𝑓‖∞,2. Assume the claim holds for 𝑡 − 1 with some 𝑡 ≥ 1.

1 1 − 𝜌

1 1 − 𝜌

Then the previous estimate gives

‖𝑠𝑡‖2 ≤ ‖𝑓‖∞,2 + 𝜌 𝑀𝑡−1 ≤ ‖𝑓‖∞,2 + 𝜌

‖𝑓‖∞,2 =

‖𝑓‖∞,2.

Hence 𝑀𝑡 = max{𝑀𝑡−1,‖𝑠𝑡‖2} ≤ 1−𝜌1 ‖𝑓‖∞,2, completing the induction. Taking sup𝑡≥0 gives ‖𝑠‖∞,2 = sup𝑡 ‖𝑠𝑡‖2 = sup𝑡 𝑀𝑡 ≤ 1−𝜌1 ‖𝑓‖∞,2, which is (50).

| |
|---|

- D.5 Explicit one-block bound without compactness We now bound one Sessa block on ℓ∞ balls by tracking constants explicitly.

- Lemma D.2 (Token-wise aﬀine bound). Let 𝑦𝑡 = 𝑥𝑡𝑊 + 𝑏 with 𝑊 ∈ ℝ𝑑×𝑑′ and 𝑏 ∈ ℝ𝑑′, where the same 𝑊 and 𝑏 are used for all tokens. Then for any sequence 𝑥, finite or infinite,

‖𝑦‖∞,2 ≤ ‖𝑊‖2 ‖𝑥‖∞,2 + ‖𝑏‖2, where ‖ ⋅ ‖2 is the spectral norm for matrices and Euclidean norm for vectors.

- Lemma D.3 (Causal attention is ℓ∞-nonexpansive). Let Αfb = (𝛼fb𝑡𝜏) satisfy (45). Then for any value sequence 𝑣, the sequence 𝑦 defined by 𝑦𝑡 ∶= ∑𝜏<𝑡 𝛼fb𝑡𝜏𝑣𝜏 satisfies

‖𝑦‖∞,2 ≤ ‖𝑣‖∞,2.

Proof. For 𝑡 ≥ 1, 𝑦𝑡 is a convex combination of {𝑣𝜏}𝜏<𝑡, hence

‖𝑦𝑡‖2 ≤ sup 𝜏<𝑡

‖𝑣𝜏‖2 ≤ ‖𝑣‖∞,2.

For 𝑡 = 0 the sum is empty, hence 𝑦0 = 0 and ‖𝑦0‖2 ≤ ‖𝑣‖∞,2 as well. Taking the supremum over 𝑡 ≥ 0 gives ‖𝑦‖∞,2 ≤ ‖𝑣‖∞,2.

Proposition 25 (One Sessa block: explicit ball-to-ball bound). Consider one width-𝑚 Sessa block 𝐺 ∶ ℓ∞(ℕ,ℝ𝑚) → ℓ∞(ℕ,ℝ𝑚). Assume:

| |
|---|

- • the feedback matrix is 𝐵fb(𝑥) = Γfb(𝑥)Αfb(𝑥) with Αfb(𝑥) satisfying (45) and 𝛾𝑡(𝑥) = tanh(𝑢𝑡(𝑥)) as above;
- • the block produces sequences 𝑓(𝑥),𝑔(𝑥) ∈ ℓ∞(ℕ,ℝ𝑟) and an output projection 𝑜 ∶ ℝ𝑟 → ℝ𝑚 given token-wise by

𝑜(𝑧)𝑡 = 𝑧𝑡𝑊out + 𝑏out, 𝑊out ∈ ℝ𝑟×𝑚, 𝑏out ∈ ℝ𝑚;

- • the block output is 𝐺(𝑥) = 𝑥 + 𝑜(𝑧) with 𝑧𝑡 = 𝑠𝑡 ⊙ 𝑔𝑡 ∈ ℝ𝑟 and the solve is in value space: 𝑧𝑡 = 𝑠𝑡 ⊙ 𝑔𝑡 ∈ ℝ𝑟, (𝐼 − 𝐵fb(𝑥))𝑠 = 𝑓(𝑥), 𝑠 ∈ ℓ∞(ℕ,ℝ𝑟).

Suppose there exist explicit constants 𝑐𝑓,𝑐𝑔,𝑐𝛾 ≥ 0 and 𝐿𝑓,𝐿𝑔,𝐿𝛾,pre ≥ 0, depending only on the block parameters, such that for all inputs 𝑥,

‖𝑓(𝑥)‖∞,2 ≤ 𝑐𝑓 + 𝐿𝑓‖𝑥‖∞,2, ‖𝑔(𝑥)‖∞,2 ≤ 𝑐𝑔 + 𝐿𝑔‖𝑥‖∞,2, sup

|𝑢𝑡(𝑥)| ≤ 𝑐𝛾 + 𝐿𝛾,pre‖𝑥‖∞,2. (51)

𝑡

Define, for 𝑅 ≥ 0,

𝜌𝑅 ∶= tanh(𝑐𝛾 + 𝐿𝛾,pre𝑅) ∈ [0,1), 𝐹𝑅 ∶= 𝑐𝑓 + 𝐿𝑓𝑅, 𝐺𝑅 ∶= 𝑐𝑔 + 𝐿𝑔𝑅.

Then for all 𝑥 with ‖𝑥‖∞,2 ≤ 𝑅, the block output satisfies the explicit bound

𝐹𝑅 𝐺 1 − 𝜌𝑅

‖𝐺(𝑥)‖∞,2 ≤ 𝑅 + ‖𝑊out‖2

+ ‖𝑏out‖2. (52)

Proof. On ‖𝑥‖∞,2 ≤ 𝑅, (51) gives ‖𝑓‖∞,2 ≤ 𝐹𝑅 and ‖𝑔‖∞,2 ≤ 𝐺𝑅. Also sup𝑡 |𝑢𝑡(𝑥)| ≤ 𝑐𝛾 + 𝐿𝛾,pre𝑅, hence sup𝑡 |𝛾𝑡(𝑥)| ≤ 𝜌𝑅. Using (⋆), we get sup𝑡 ∑𝜏<𝑡 |[𝐵fb]𝑡,𝜏(𝑥)| ≤ 𝜌𝑅 < 1. Lemma 4.2 then yields

1 1 − 𝜌𝑅

𝐹𝑅 1 − 𝜌𝑅

‖𝑠‖∞,2 ≤

‖𝑓‖∞,2 ≤

.

For the element-wise product in ℝ𝑟, for each 𝑡,

‖𝑧𝑡‖2 = ‖𝑠𝑡 ⊙ 𝑔𝑡‖2 ≤ ‖𝑠𝑡‖2 ‖𝑔𝑡‖2, since

𝑠2𝑡𝑖(∑

𝑔𝑡𝑗2 ) = ‖𝑠𝑡‖2‖𝑔𝑡‖2.

‖𝑠𝑡 ⊙ 𝑔𝑡‖2 = ∑

𝑠2𝑡𝑖𝑔𝑡𝑖2 ≤ ∑

𝑖

𝑖

𝑗

𝐹𝑅 1 − 𝜌𝑅

‖𝑧‖∞,2 ≤ ‖𝑠‖∞,2 ‖𝑔‖∞,2 ≤

𝐺𝑅.

Hence

Finally, by Lemma D.2 for 𝑜(𝑧) = 𝑧𝑊𝑜 + 𝑏𝑜 and the residual 𝐺(𝑥) = 𝑥 + 𝑜(𝑧),

‖𝐺(𝑥)‖∞,2 ≤ ‖𝑥‖∞,2 + ‖𝑜(𝑧)‖∞,2 ≤ 𝑅 + ‖𝑊out‖2‖𝑧‖∞,2 + ‖𝑏out‖2, which gives (52).

| |
|---|

- Remark D.4 (Explicit dependence of the constants in (51)). Each branch, including the query, key, and value maps and the MLPs producing 𝑓, 𝑔, and 𝑢 and related components, is a finite composition of token-wise aﬀine maps, RoPE𝑡 rotations that are orthogonal and norm-preserving, masked softmax attention as in Lemma D.3,

and element-wise nonlinearities whose growth is at most linear on bounded sets. The solve (𝐼 − 𝐵fb)𝑠 = 𝑓 and the Hadamard product 𝑧 = 𝑠 ⊙ 𝑔 take place in the value space ℝ𝑟, while the output projection 𝑜 ∶ ℝ𝑟 → ℝ𝑚 is

token-wise aﬀine. Thus one can always choose 𝑐• and 𝐿• explicitly from the operator norms of the weight matrices involved and the norms of the biases, by repeated use of Lemma D.2 and the inequality ‖GELU(𝑣)‖2 ≤ ‖𝑣‖2.

### E Polynomial decay of token influence in the feedback recursion

We work on discrete time 𝑡 ∈ ℕ = {0,1,2,…}. Let (𝛾𝑡)𝑡≥0 be a sequence in ℝ, and let {𝛼fb𝑡,𝑗}𝑡≥1, 0≤𝑗<𝑡 be nonnegative weights such that, for every 𝑡 ≥ 1,

- E.1 Scalar recursion and impulse response

𝑡−1

∑

𝛼fb𝑡,𝑗 ≤ 1. (53)

𝛼fb𝑡,𝑗 ≥ 0,

𝑗=0

Given an input sequence (𝑓𝑡)𝑡≥0, consider the recursion

𝑡−1

∑

𝛼fb𝑡,𝑗𝑦𝑗, 𝑡 ≥ 1. (54)

𝑦0 = 𝑓0, 𝑦𝑡 = 𝑓𝑡 + 𝛾𝑡

𝑗=0

To isolate the influence of a single token, we consider the impulse input at time 0:

𝑓0 = 1, 𝑓𝑡 = 0 for 𝑡 ≥ 1, so that (54) reduces to the impulse response recursion

⎧ ⎨ { ⎩

0 = 1,

𝑡−1

𝑦𝑡 = 𝛾𝑡

∑

𝛼fb𝑡,𝑗𝑦𝑗, 𝑡 ≥ 1.

(55)

𝑗=0

In the full vector model, 𝑦𝑡 can be interpreted as a scalar influence coeﬀicient, e.g. an entry of (𝐼 − 𝐵fb)−1.

- Assumption 26 (Upper envelope on attention). There exists a constant 𝑐2 ∈ (0,∞) such that for all 𝑡 ≥ 1 and all 0 ≤ 𝑗 < 𝑡,

𝛼fb𝑡,𝑗 ≤

𝑐2 𝑡

, and (53) holds. (56)

Remark E.1 (On the size of 𝑐2). Under (53) with ∑𝑡−1𝑗=0 𝛼fb𝑡,𝑗 ≤ 1, the conclusion 𝑐2 ≥ 1 no longer follows. If one additionally has ∑𝑡−1𝑗=0 𝛼fb𝑡,𝑗 = 1 for all 𝑡, then 𝑐2 ≥ 1 is necessary.

- Assumption 27 (Bounded feedback). There exists 𝛾max ∈ [0,1) such that for all 𝑡 ≥ 0, |𝛾𝑡| ≤ 𝛾max. (57)

#### E.2 Assumptions

𝜂 ∶= 𝛾max𝑐2, (58) and assume the nontrivial feedback regime

Define the feedback mass parameter

0 < 𝜂 < 1. (59) Equivalently, define the tail exponent

𝛽tail ∶= 1 − 𝜂 = 1 − 𝛾max𝑐2 ∈ (0,1], (60) so that 𝜂 = 1 − 𝛽tail.

Remark E.2 (Degenerate case 𝜂 = 0). If 𝜂 = 0 then 𝛾max = 0 and hence 𝛾𝑡 = 0 for all 𝑡. The recursion (55) has no feedback and the impulse response is trivial: 𝑦0 = 1 and 𝑦𝑡 = 0 for all 𝑡 ≥ 1. We therefore focus on 0 < 𝜂 < 1 when stating a genuine power-law tail.

#### E.3 Bounded logits imply near-uniform softmax weights

Bounded logits imply near-uniform softmax weights. This is an immediate specialization of Lemma C.1. Indeed, fix 𝑡 ≥ 1 and take the index set ℐ = {0,…,𝑡 − 1} with 𝑛 = |ℐ| = 𝑡. If the logits satisfy ℶmin ≤ ℶ𝑡,𝑗 ≤ ℶmax for all 𝑗 ∈ ℐ, then the spread is Δ0 = ℶmax − ℶmin, and Lemma C.1 gives, for all 𝑗 < 𝑡,

𝑒ℶ

𝑒ℶ

≤ 𝛼fb𝑡,𝑗 ≤

. (61)

min−ℶmax

max−ℶmin

𝑡

𝑡

In particular, Assumption 26 holds with 𝑐2 = 𝑒ℶ

max−ℶmin.

#### E.4 Polynomial decay theorem

Theorem 28 (Polynomial decay of the impulse response). Consider the impulse recursion (55). Suppose Assumptions 26 and 27 hold and 0 < 𝜂 = 𝛾max𝑐2 < 1; equivalently 𝛽tail = 1 − 𝜂 ∈ (0,1). Then for all 𝑡 ≥ 1,

|𝑦𝑡| ≤ 𝐶 𝑡−𝛽

, where one may take 𝐶 ∶= (1 − 𝛽tail)𝑒1−𝛽

= 𝜂𝑒𝜂. (62)

In particular, since 𝛽tail > 0, we have lim𝑡→∞ 𝑦𝑡 = 0.

tail

tail

Proof. Assume 0 < 𝜂 < 1. The degenerate case 𝜂 = 0 is covered by Remark E.2. Let 𝑧𝑡 ∶= |𝑦𝑡|. From (55) and Assumptions 26–27, for 𝑡 ≥ 1,

𝑡−1

𝑡−1

𝑡−1

𝑧𝑡 = ∣𝛾𝑡

∑

𝛼fb𝑡,𝑗𝑦𝑗∣ ≤ |𝛾𝑡|

∑

𝛼fb𝑡,𝑗|𝑦𝑗| ≤ 𝛾max

∑

𝛼fb𝑡,𝑗𝑧𝑗.

𝑗=0

𝑗=0

𝑗=0

Define the comparison sequence (𝑦̃𝑡)𝑡≥0 by

𝑡−1

∑

𝑦̃0 = 1, 𝑦̃𝑡 = 𝛾max

𝛼fb𝑡,𝑗𝑦̃𝑗, 𝑡 ≥ 1. (63)

𝑗=0

By induction on 𝑡, using 𝛼fb𝑡,𝑗 ≥ 0, we have 𝑧𝑡 ≤ 𝑦̃𝑡 for all 𝑡, hence |𝑦𝑡| = 𝑧𝑡 ≤ 𝑦̃𝑡 ∀𝑡. (64)

Let 𝑠𝑡 ∶= ∑𝑡𝑘=0 𝑦̃𝑘. Since 𝑦̃𝑘 ≥ 0, the sequence 𝑠𝑡 is increasing and 𝑠𝑡 ≥ 1. Using (63) and 𝛼fb𝑡,𝑗 ≤ 𝑐2/𝑡 we obtain, for 𝑡 ≥ 1,

𝑐2 𝑡

𝜂 𝑡

𝑡−1

𝑡−1

∑

∑

𝑦̃𝑡 = 𝛾max

𝛼fb𝑡,𝑗𝑦̃𝑗 ≤ 𝛾max

𝑦̃𝑗 =

𝑠𝑡−1.

𝑗=0

𝑗=0

𝜂 𝑡

𝜂 𝑡

), 𝑡 ≥ 1. (65) Taking logarithms and using log(1 + 𝑥) ≤ 𝑥 for 𝑥 > −1,

𝑠𝑡−1 = 𝑠𝑡−1(1 +

𝑠𝑡 = 𝑠𝑡−1 + 𝑦̃𝑡 ≤ 𝑠𝑡−1 +

Therefore,

𝜂 𝑡

𝜂 𝑡

𝑛

𝑛

∑

) ≤

∑

log(1 +

log𝑠𝑛 ≤ log𝑠0 +

= 𝜂𝐻𝑛,

𝑡=1

𝑡=1

where 𝐻𝑛 = ∑𝑛𝑡=1 1𝑡 is the 𝑛-th harmonic number. Using 𝐻𝑛 ≤ 1 + log𝑛 for 𝑛 ≥ 1 gives

𝑠𝑛 ≤ 𝑒𝜂𝑛𝜂 ∀𝑛 ≥ 1. (66)

Finally, for 𝑡 ≥ 1 we use 𝑠𝑡−1 ≤ 𝑠𝑡 and (66):

𝜂 𝑡

𝜂 𝑡

𝜂 𝑡

𝑠𝑡−1 ≤

𝑠𝑡 ≤

𝑒𝜂𝑡𝜂 = 𝜂𝑒𝜂 𝑡𝜂−1.

𝑦̃𝑡 ≤

Since 𝜂 − 1 = −(1 − 𝜂) = −𝛽tail, we obtain 𝑦̃𝑡 ≤ 𝜂𝑒𝜂 𝑡−𝛽

tail. Combining with (64) yields (62) with 𝐶 = 𝜂𝑒𝜂.

| |
|---|

- Corollary E.3 (Finite-horizon bound). Fix 𝑇 ∈ ℕ∗ and consider (55) only for 𝑡 ∈ {0,1,…,𝑇 −1}. Assume that Assumptions 26 and 27 hold for all 1 ≤ 𝑡 ≤ 𝑇 − 1 with the same constants 𝑐2 and 𝛾max, and 0 < 𝜂 = 𝛾max𝑐2 < 1; equivalently 𝛽tail = 1 − 𝜂 ∈ (0,1). Then (62) holds for all 𝑡 ∈ {1,…,𝑇 − 1} with the same constant 𝐶 = 𝜂𝑒𝜂 =

#### E.5 Finite-horizon formulation

(1 − 𝛽tail)𝑒1−𝛽

tail.

Proof. This is an immediate restriction of Theorem 28 to 1 ≤ 𝑡 ≤ 𝑇 − 1.

#### E.6 Impulse at an arbitrary position 𝑗

| |
|---|

- Corollary E.4 (Decay from an impulse at position 𝑗). Fix an index 𝑗 ≥ 0. Consider (54) with the impulse input at 𝑗:

𝑓𝑗 = 1, 𝑓𝑡 = 0 for 𝑡 ≠ 𝑗,

and with 𝑦𝑡 = 0 for 𝑡 < 𝑗. Equivalently, 𝑦0 = 0 if 𝑗 > 0 and the recursion is started from 𝑡 = 𝑗. Assume Assumptions 26–27 and 0 < 𝜂 = 𝛾max𝑐2 < 1; equivalently 𝛽tail = 1 − 𝜂 ∈ (0,1). Then for all 𝑡 > 𝑗,

|𝑦𝑡| ≤ 𝐶 (𝑡 − 𝑗)−𝛽

, where one may take 𝐶 ∶= 𝜂𝑒𝜂 = (1 − 𝛽tail)𝑒1−𝛽

.

Proof. Define 𝑢𝑛 ∶= |𝑦𝑗+𝑛| for 𝑛 ≥ 0. Then 𝑢0 = |𝑦𝑗| = 1. For 𝑛 ≥ 1, since 𝑦𝑘 = 0 for 𝑘 < 𝑗 and 𝛼fb𝑗+𝑛,𝑘 ≥ 0,

tail

tail

𝑗+𝑛−1

𝑗+𝑛−1

𝑛−1

𝑢𝑛 = |𝑦𝑗+𝑛| = ∣𝛾𝑗+𝑛

∑

𝛼fb𝑗+𝑛,𝑘𝑦𝑘∣ ≤ |𝛾𝑗+𝑛|

∑

𝛼fb𝑗+𝑛,𝑘|𝑦𝑘| ≤ 𝛾max

∑

𝛼fb𝑗+𝑛,𝑗+𝑟 𝑢𝑟.

𝑟=0

𝑘=0

𝑘=𝑗

𝑐2 𝑗 + 𝑛

𝑐2 𝑛

𝛼fb𝑗+𝑛,𝑗+𝑟 ≤

≤

(𝑛 ≥ 1),

Moreover, by Assumption 26,

since 𝑗 + 𝑛 ≥ 𝑛. Thus the sequence 𝑢𝑛 satisfies the same comparison inequality as in the proof of Theorem 28, with the same 𝛾max and the envelope 𝑐2/𝑛, so repeating that argument yields

. Substituting 𝑛 = 𝑡 − 𝑗 yields the claim.

𝑢𝑛 ≤ 𝜂𝑒𝜂 𝑛𝜂−1 = (𝜂𝑒𝜂)𝑛−𝛽

tail

| |
|---|

### F Tightness of the polynomial tail in a realizable regime

This section complements Theorem 28 with the upper bound 𝑂(ℓ−𝛽

) by exhibiting a concrete diffuse routing regime in which the impulse influence is exactly polynomial, that is, Θ(ℓ−𝛽

). This eliminates the semantic ambiguity that an upper bound alone does not preclude faster decay, for instance exponential decay.

tail

tail

- F.1 Gamma-ratio inequality of Gautschi Lemma F.1 (Gautschi inequality for 0 < 𝛾 < 1). Let 𝛾 ∈ (0,1) and 𝑡 ≥ 1 be an integer. Then

Γ(𝑡 + 𝛾) Γ(𝑡 + 1)

(𝑡 + 1)𝛾−1 ≤

≤ 𝑡𝛾−1. (67)

Equivalently, with 𝛽tail ∶= 1 − 𝛾 ∈ (0,1),

Γ(𝑡 + 𝛾) Γ(𝑡 + 1)

(𝑡 + 1)−𝛽

≤

≤ 𝑡−𝛽

.

tail

tail

Proof. By Gautschi’s inequality (Gautschi, 1959), for 𝑥 > 0 and 0 < 𝛾 < 1,

Γ(𝑥 + 1) Γ(𝑥 + 𝛾)

𝑥1−𝛾 <

< (𝑥 + 1)1−𝛾.

Setting 𝑥 = 𝑡 and taking reciprocals yields

Γ(𝑡 + 𝛾) Γ(𝑡 + 1)

(𝑡 + 1)𝛾−1 ≤

≤ 𝑡𝛾−1,

#### F.2 Uniform routing yields a Θ(ℓ−𝛽

#### ) tail

which is (67).

| |
|---|

tail

We consider the scalar impulse recursion from Section E:

𝑡−1

𝑦0 = 𝑓0, 𝑦𝑡 = 𝑓𝑡 + 𝛾𝑡

∑

𝛼fb𝑡,𝑗𝑦𝑗, 𝑡 ≥ 1. (68)

𝑗=0

1 𝑡

Proposition 29 (Tightness under uniform routing). Assume uniform routing, which is maximally diffuse, and constant positive feedback:

1[𝑗 < 𝑡], 𝛾𝑡 ≡ 𝛾 ∈ (0,1).

𝛼fb𝑡,𝑗 =

Consider an impulse at time 0: 𝑓0 = 1 and 𝑓𝑡 = 0 for all 𝑡 ≥ 1. Then for every 𝑡 ≥ 1 the impulse influence admits the closed form

𝛾 Γ(1 + 𝛾)

Γ(𝑡 + 𝛾) Γ(𝑡 + 1)

𝑦𝑡 =

⋅

. (69)

Consequently, letting 𝛽tail ∶= 1 − 𝛾 ∈ (0,1), one has the two-sided bound

𝛾 Γ(1 + 𝛾)

𝛾 Γ(1 + 𝛾)

(𝑡 + 1)−𝛽

≤ 𝑦𝑡 ≤

𝑡−𝛽

, 𝑡 ≥ 1, (70)

tail

tail

𝑦𝑡 = Θ(𝑡−𝛽

) and hence 𝑦𝑡 = Ω(𝑡−𝛽

).

and in particular

Proof. Define partial sums 𝑆𝑡 ∶= ∑𝑡𝑘=0 𝑦𝑘. Under the stated assumptions and for 𝑡 ≥ 1,

tail

tail

𝛾 𝑡

𝛾 𝑡

𝛾 𝑡

𝑡−1

𝑦𝑡 =

∑

𝑦𝑗 =

𝑆𝑡−1, 𝑆𝑡 = 𝑆𝑡−1 + 𝑦𝑡 = 𝑆𝑡−1(1 +

),

𝑗=0

with 𝑆0 = 𝑦0 = 𝑓0 = 1. Thus

𝑖 + 𝛾 𝑖

𝛾 𝑖

Γ(𝑡 + 1 + 𝛾) Γ(1 + 𝛾)Γ(𝑡 + 1)

𝑡

𝑡

) =

∏

∏

(1 +

𝑆𝑡 =

=

.

𝑖=1

𝑖=1

Using 𝑦𝑡 = 𝛾𝑡 𝑆𝑡−1 and Γ(𝑡 + 1) = 𝑡Γ(𝑡) gives

𝛾 𝑡

Γ(𝑡 + 𝛾) Γ(1 + 𝛾)Γ(𝑡)

𝛾 Γ(1 + 𝛾)

Γ(𝑡 + 𝛾) Γ(𝑡 + 1)

𝑦𝑡 =

⋅

=

⋅

,

which is (69). The two-sided bound (70) follows directly from Lemma F.1 with 𝛽tail = 1 − 𝛾.

| |
|---|

- Corollary F.2 (Uniform routing with an impulse at an arbitrary source position). Assume the same explicit uniform-routing regime as in Proposition 29:

1 𝑡

1[𝑗 < 𝑡], 𝛾𝑡 ≡ 𝛾 ∈ (0,1).

𝛼fb𝑡,𝑗 =

Consider an impulse at time 𝜏 ≥ 0, i.e.

𝑓𝜏 = 1, 𝑓𝑡 = 0 for 𝑡 ≠ 𝜏, with 𝑦𝑡 = 0 for 𝑡 < 𝜏. Then for every ℓ ≥ 1,

Γ(𝜏 + 1) Γ(𝜏 + 1 + 𝛾)

Γ(𝜏 + ℓ + 𝛾) Γ(𝜏 + ℓ + 1)

𝑦𝜏+ℓ = 𝛾

⋅

. (71)

Consequently, with 𝛽tail ∶= 1 − 𝛾 ∈ (0,1), for every fixed source position 𝜏, 𝑦𝜏+ℓ = Θ𝜏(ℓ−𝛽

) (ℓ → ∞). Moreover, the prefactor depends on 𝜏 and satisfies

tail

Γ(𝜏 + 1) Γ(𝜏 + 1 + 𝛾)

≍ 𝜏−𝛾 (𝜏 → ∞).

𝛾

In particular, there is no positive lower constant 𝑐− > 0 such that 𝑦𝜏+ℓ ≥ 𝑐− ℓ−𝛽

for all source positions 𝜏 and all ℓ ≥ 1 on an unbounded horizon.

tail

𝑡

∑

𝑆𝑡 ∶=

𝑦𝑘, 𝑡 ≥ 𝜏.

Proof. Define partial sums

𝑘=𝜏

Then 𝑆𝜏 = 𝑦𝜏 = 1. For 𝑡 ≥ 𝜏 + 1, the recursion gives

𝛾 𝑡

𝛾 𝑡

𝛾 𝑡

𝑡−1

∑

𝑦𝑗 =

𝑆𝑡−1, 𝑆𝑡 = 𝑆𝑡−1 + 𝑦𝑡 = 𝑆𝑡−1(1 +

).

𝑦𝑡 =

𝑗=𝜏

Hence, for every 𝑡 ≥ 𝜏 + 1,

𝑖 + 𝛾 𝑖

Γ(𝑡 + 1 + 𝛾)Γ(𝜏 + 1) Γ(𝜏 + 1 + 𝛾)Γ(𝑡 + 1)

𝛾 𝑖

𝑡

𝑡

) =

∏

=

.

𝑆𝑡 =

∏

(1 +

𝑖=𝜏+1

𝑖=𝜏+1

Using 𝑦𝑡 = 𝛾𝑡 𝑆𝑡−1 and Γ(𝑡 + 1) = 𝑡Γ(𝑡) yields

𝛾 𝑡

Γ(𝑡 + 𝛾)Γ(𝜏 + 1) Γ(𝜏 + 1 + 𝛾)Γ(𝑡)

Γ(𝜏 + 1) Γ(𝜏 + 1 + 𝛾)

Γ(𝑡 + 𝛾) Γ(𝑡 + 1)

𝑦𝑡 =

⋅

= 𝛾

⋅

.

Setting 𝑡 = 𝜏 + ℓ gives (71). For fixed 𝜏, the factor

Γ(𝜏 + 1) Γ(𝜏 + 1 + 𝛾)

𝛾

is a positive constant depending only on 𝜏, while Lemma F.1 gives

Γ(𝜏 + ℓ + 𝛾) Γ(𝜏 + ℓ + 1)

(𝜏 + ℓ + 1)−𝛽

≤

≤ (𝜏 + ℓ)−𝛽

.

tail

tail

Since 𝜏 is fixed, this implies

Γ(𝜏 + ℓ + 𝛾) Γ(𝜏 + ℓ + 1)

= Θ𝜏(ℓ−𝛽

),

). The Gamma-ratio asymptotic gives

hence 𝑦𝜏+ℓ = Θ𝜏(ℓ−𝛽

tail

tail

Γ(𝜏 + 1) Γ(𝜏 + 1 + 𝛾)

≍ (𝜏 + 1)−𝛾 (𝜏 → ∞),

so the source-dependent prefactor decays polynomially with 𝜏. Moreover, taking ℓ = 1 in (71) gives

Γ(𝜏 + 1) Γ(𝜏 + 1 + 𝛾)

Γ(𝜏 + 1 + 𝛾) Γ(𝜏 + 2)

𝛾 𝜏 + 1

⋅

=

.

𝑦𝜏+1 = 𝛾

Hence 𝑦𝜏+1 → 0 as 𝜏 → ∞. Therefore no positive lower constant independent of 𝜏 can satisfy 𝑦𝜏+ℓ ≥ 𝑐− ℓ−𝛽

for all source positions 𝜏 and all ℓ ≥ 1 on an unbounded horizon.

tail

- Corollary F.3 (Uniform two-sided heavy-tail envelope on a bounded source family). Fix 𝜏max ∈ ℕ. Under the

| |
|---|

max,𝛾 > 0 such that for every source position 0 ≤ 𝜏 ≤ 𝜏max and every ℓ ≥ 1,

max,𝛾,𝑐𝜏+

regime of Corollary F.2, there exist constants 𝑐𝜏−

max,𝛾 ℓ−𝛽

, 𝛽tail ∶= 1 − 𝛾.

max,𝛾 ℓ−𝛽

≤ 𝑦𝜏+ℓ ≤ 𝑐𝜏+

𝑐𝜏−

tail

tail

In particular, the explicit uniform-routing regime realizes a uniform two-sided heavy-tail envelope on every bounded source family, and hence on every fixed finite horizon.

Γ(𝜏 + 1) Γ(𝜏 + 1 + 𝛾)

𝑎𝜏 ∶= 𝛾

.

Proof. Write

Since the set {0,…,𝜏max} is finite and each 𝑎𝜏 is positive, 𝑚𝜏

max,𝛾 ∶= min

𝑎𝜏 > 0, 𝑀𝜏

max,𝛾 ∶= max

𝑎𝜏 < ∞.

0≤𝜏≤𝜏max

0≤𝜏≤𝜏max

Γ(𝜏 + ℓ + 𝛾) Γ(𝜏 + ℓ + 1)

𝑦𝜏+ℓ = 𝑎𝜏

.

By Corollary F.2,

Γ(𝜏 + ℓ + 𝛾) Γ(𝜏 + ℓ + 1)

. Therefore

(𝜏 + ℓ + 1)−𝛽

≤

≤ (𝜏 + ℓ)−𝛽

- Lemma F.1 yields

tail

tail

. Also, since 0 ≤ 𝜏 ≤ 𝜏max and ℓ ≥ 1,

𝑦𝜏+ℓ ≤ 𝑀𝜏

max,𝛾 (𝜏 + ℓ)−𝛽

≤ 𝑀𝜏

max,𝛾 ℓ−𝛽

tail

tail

𝜏 + ℓ + 1 ≤ 𝜏max + ℓ + 1 ≤ (𝜏max + 2)ℓ, hence

(𝜏 + ℓ + 1)−𝛽

≥ (𝜏max + 2)−𝛽

ℓ−𝛽

.

tail

tail

tail

. So one may take

𝑦𝜏+ℓ ≥ 𝑚𝜏

max,𝛾 (𝜏 + ℓ + 1)−𝛽

≥ 𝑚𝜏

max,𝛾 (𝜏max + 2)−𝛽

ℓ−𝛽

Thus

tail

tail

tail

𝑐𝜏−

max,𝛾 ∶= 𝑚𝜏

max,𝛾 (𝜏max + 2)−𝛽

, 𝑐𝜏+

max,𝛾 ∶= 𝑀𝜏

max,𝛾.

tail

Consequence for the influence kernel In the lower-triangular solve 𝑠 = 𝐾𝑓 with 𝐾 = (𝐼 − 𝐵fb)−1, choosing

| |
|---|

⎧ ⎨{⎩

- 0, 𝑡 = 0, 𝛾 𝑡

- 1[𝑗 < 𝑡], 𝑡 ≥ 1,

[𝐵fb]𝑡,𝑗 = 𝛾 𝛼fb𝑡,𝑗 =

yields that the column 𝐾⋅,0 is precisely the impulse response (𝑦𝑡)𝑡≥0 above. Hence, |𝐾𝑡,0| = Θ(𝑡−𝛽

),

tail

so the polynomial envelope in Theorem 8 is sharp, and the rate is attained by a concrete heavy-tailed memory mode.

Remark F.4 (Impulse at time 𝜏). Assume 𝛾 ∈ (0,1). The same computation applies to an impulse at time 𝜏. If 𝑓𝜏 = 1, 𝑓𝑡 = 0 for 𝑡 ≠ 𝜏, and 𝑦𝑡 = 0 for 𝑡 < 𝜏, then for 𝑡 ≥ 𝜏 + 1

Γ(𝑡 + 𝛾)Γ(𝜏 + 1) Γ(𝑡 + 1)Γ(𝜏 + 1 + 𝛾)

Γ(𝑡 + 𝛾) Γ(𝑡 + 1)

𝑦𝑡 = 𝛾

= 𝐶(𝜏,𝛾) ⋅

,

with 𝐶(𝜏,𝛾) ∶= 𝛾 Γ(𝜏 + 1)/Γ(𝜏 + 1 + 𝛾) > 0. Hence, for ℓ = 𝑡 − 𝜏, the lag-ℓ tail is again Θ(ℓ−𝛽

) by Lemma F.1, in agreement with Corollary E.4.

tail

### G Heavy-tail convolution estimates

Definition 9 (Discrete convolution on positive lags). For nonnegative sequences 𝑎,𝑏 ∶ ℕ∗ → [0,∞), define

𝑛−1

(𝑎 ∗ 𝑏)(𝑛) ∶=

∑

𝑎(𝑛 − 𝑚)𝑏(𝑚), 𝑛 ≥ 2,

𝑚=1

and (𝑎 ∗ 𝑏)(1) ∶= 0. Inductively define 𝑎(∗1) ∶= 𝑎 and 𝑎(∗𝑘) ∶= 𝑎(∗(𝑘−1)) ∗ 𝑎 for 𝑘 ≥ 2.

- Lemma G.1 (Discrete power convolution). Let 𝜎,𝜌 > 0, and define

𝑢𝜎(𝑛) ∶= 𝑛𝜎−1, 𝑢𝜌(𝑛) ∶= 𝑛𝜌−1, 𝑛 ∈ ℕ∗. Then there exist constants 𝑐𝜎,𝜌,𝐶𝜎,𝜌 ∈ (0,∞) such that

𝑐𝜎,𝜌 𝑛𝜎+𝜌−1 ≤ (𝑢𝜎 ∗ 𝑢𝜌)(𝑛) ≤ 𝐶𝜎,𝜌 𝑛𝜎+𝜌−1, 𝑛 ≥ 2.

Proof. Fix 𝑛 ≥ 2. For the upper bound, split the sum into the two regions

𝑛 2

𝑛 2

1 ≤ 𝑚 ≤ ⌊

⌋ and ⌊

⌋ + 1 ≤ 𝑚 ≤ 𝑛 − 1.

If 1 ≤ 𝑚 ≤ 𝑛/2, then 𝑛 − 𝑚 ∈ [𝑛/2,𝑛 − 1], hence

(𝑛 − 𝑚)𝜎−1 ≤ 𝐶𝜎 𝑛𝜎−1, 𝐶𝜎 ∶= max{1,21−𝜎}. Therefore

⌊𝑛/2⌋

⌊𝑛/2⌋

∑

(𝑛 − 𝑚)𝜎−1𝑚𝜌−1 ≤ 𝐶𝜎𝑛𝜎−1

∑

𝑚𝜌−1.

𝑚=1

𝑚=1

Since 𝜌 > 0, the standard integral comparison gives

⌊𝑛/2⌋

𝑛/2

𝑥𝜌−1 𝑑𝑥 ≤ 𝐶𝜌′ 𝑛𝜌

∑

𝑚𝜌−1 ≤ 1 + ∫

1

𝑚=1

for some constant 𝐶𝜌′ depending only on 𝜌. Hence

⌊𝑛/2⌋

∑

(𝑛 − 𝑚)𝜎−1𝑚𝜌−1 ≤ 𝐶𝜎𝐶𝜌′ 𝑛𝜎+𝜌−1.

𝑚=1

If ⌊𝑛/2⌋ + 1 ≤ 𝑚 ≤ 𝑛 − 1, then 𝑚 ∈ [𝑛/2,𝑛 − 1], hence

𝑚𝜌−1 ≤ 𝐶𝜌 𝑛𝜌−1, 𝐶𝜌 ∶= max{1,21−𝜌}. Therefore

𝑛−1

𝑛−1

∑

(𝑛 − 𝑚)𝜎−1.

∑

(𝑛 − 𝑚)𝜎−1𝑚𝜌−1 ≤ 𝐶𝜌𝑛𝜌−1

𝑚=⌊𝑛/2⌋+1

𝑚=⌊𝑛/2⌋+1

After the change of variable 𝑟 = 𝑛 − 𝑚, the inner sum becomes

⌈𝑛/2⌉−1

∑

𝑟𝜎−1 ≤ 𝐶𝜎′𝑛𝜎

𝑟=1

for some constant 𝐶𝜎′ depending only on 𝜎. Hence

𝑛−1

(𝑛 − 𝑚)𝜎−1𝑚𝜌−1 ≤ 𝐶𝜌𝐶𝜎′ 𝑛𝜎+𝜌−1.

∑

𝑚=⌊𝑛/2⌋+1

Adding the two estimates proves the upper bound. For the lower bound, restrict the sum to the central block

𝑛 4

- 3𝑛

- 4

⌊

⌋ ≤ 𝑚 ≤ ⌊

⌋.

For every such 𝑚 and every 𝑛 ≥ 4 one has

𝑛 4

- 3𝑛

- 4

𝑛 4

- 3𝑛

- 4

≤ 𝑚 ≤

,

≤ 𝑛 − 𝑚 ≤

.

𝑚𝜌−1 ≥ 𝑐𝜌 𝑛𝜌−1, (𝑛 − 𝑚)𝜎−1 ≥ 𝑐𝜎 𝑛𝜎−1, where one may take

Hence

𝑐𝜌 ∶= min{1,41−𝜌}, 𝑐𝜎 ∶= min{1,41−𝜎}. Indeed, if 𝜌 ≤ 1, then 𝑚 ≤ 𝑛 implies 𝑚𝜌−1 ≥ 𝑛𝜌−1; if 𝜌 ≥ 1, then 𝑚 ≥ 𝑛/4 implies 𝑚𝜌−1 ≥ 41−𝜌𝑛𝜌−1. The same

argument applies to (𝑛 − 𝑚)𝜎−1. Therefore every summand in the central block is bounded below by

𝑐𝜎𝑐𝜌 𝑛𝜎+𝜌−2. The number of integers in the central block is at least 𝑛/2 − 2. Consequently, for all 𝑛 ≥ 8,

𝑐𝜎𝑐𝜌 4

𝑛 2

(𝑢𝜎 ∗ 𝑢𝜌)(𝑛) ≥ (

− 2)𝑐𝜎𝑐𝜌 𝑛𝜎+𝜌−2 ≥

𝑛𝜎+𝜌−1.

Since only finitely many values 2 ≤ 𝑛 < 8 remain, their minimum ratio to 𝑛𝜎+𝜌−1 is positive. Adjusting the constant completes the proof.

- Theorem 30 (Heavy-tail convolution class). Fix 𝛽tail ∈ (0,1) and define

| |
|---|

𝑓𝛽

(𝑛) ∶= 𝑛−𝛽

, 𝑛 ∈ ℕ∗.

Then, for every fixed 𝑘 ≥ 1, there exist constants 𝑐𝑘,𝛽

,𝐶𝑘,𝛽

∈ (0,∞) such that

tail

tail

𝑐𝑘,𝛽

𝑛𝑘(1−𝛽

tail)−1 ≤ 𝑓𝛽(∗𝑘)

(𝑛) ≤ 𝐶𝑘,𝛽

𝑛𝑘(1−𝛽

tail)−1, 𝑛 ≥ 𝑘. (72)

tail

tail

tail

tail

tail

𝜎 ∶= 1 − 𝛽tail ∈ (0,1). Then

Proof. Set

= 𝑛𝜎−1 = 𝑢𝜎(𝑛). We prove by induction on 𝑘 that there exist constants 𝑎𝑘,𝑏𝑘 > 0 such that 𝑎𝑘 𝑛𝑘𝜎−1 ≤ 𝑢(∗𝑘)𝜎 (𝑛) ≤ 𝑏𝑘 𝑛𝑘𝜎−1, 𝑛 ≥ 𝑘. (73)

𝑓𝛽

(𝑛) = 𝑛−𝛽

tail

tail

For 𝑘 = 1, this is exactly

𝑢𝜎(𝑛) = 𝑛𝜎−1.

Assume now that (73) holds for some 𝑘 ≥ 1. Fix 𝑛 ≥ 𝑘 + 1. By definition,

𝑛−1

𝑢(∗(𝑘+1))𝜎 (𝑛) =

∑

𝑢(∗𝑘)𝜎 (𝑛 − 𝑚)𝑢𝜎(𝑚).

𝑚=1

For the upper bound, note that 𝑢(∗𝑘)𝜎 (𝑟) = 0 for 𝑟 < 𝑘, since it is a 𝑘-fold convolution of positive-lag sequences. Hence, after enlarging 𝑏𝑘 if necessary, we may write

𝑢(∗𝑘)𝜎 (𝑟) ≤ 𝑏𝑘 𝑟𝑘𝜎−1 for every 𝑟 ≥ 1. Therefore

𝑛−1

𝑢(∗(𝑘+1))𝜎 (𝑛) ≤ 𝑏𝑘

∑

(𝑛 − 𝑚)𝑘𝜎−1𝑚𝜎−1.

𝑚=1

Applying Lemma G.1 with exponents 𝑘𝜎 and 𝜎 yields

𝑢(∗(𝑘+1))𝜎 (𝑛) ≤ 𝑏𝑘+1 𝑛(𝑘+1)𝜎−1 for some constant 𝑏𝑘+1 > 0.

For the lower bound, rewrite the sum using 𝑟 ∶= 𝑛 − 𝑚:

𝑛−1

∑

𝑢(∗(𝑘+1))𝜎 (𝑛) =

𝑢(∗𝑘)𝜎 (𝑟)𝑢𝜎(𝑛 − 𝑟).

𝑟=1

Since 𝑢(∗𝑘)𝜎 (𝑟) = 0 for 𝑟 < 𝑘, this becomes

𝑛−1

∑

𝑢(∗(𝑘+1))𝜎 (𝑛) =

𝑢(∗𝑘)𝜎 (𝑟)(𝑛 − 𝑟)𝜎−1.

𝑟=𝑘

Applying the lower induction hypothesis on the range 𝑟 ≥ 𝑘 gives

𝑛−1

∑

𝑢(∗(𝑘+1))𝜎 (𝑛) ≥ 𝑎𝑘

𝑟𝑘𝜎−1(𝑛 − 𝑟)𝜎−1.

𝑟=𝑘

𝑛−1

𝑛−1

𝑘−1

∑

∑

∑

𝑟𝑘𝜎−1(𝑛 − 𝑟)𝜎−1 =

𝑟𝑘𝜎−1(𝑛 − 𝑟)𝜎−1 −

𝑟𝑘𝜎−1(𝑛 − 𝑟)𝜎−1.

Now write

𝑟=1

𝑟=1

𝑟=𝑘

- By Lemma G.1, the full sum is bounded below by 𝑐 𝑛(𝑘+1)𝜎−1

for some constant 𝑐 > 0 depending only on 𝑘 and 𝜎. On the other hand, since 𝑘 − 1 is fixed,

𝑘−1

∑

𝑟𝑘𝜎−1(𝑛 − 𝑟)𝜎−1 ≤ 𝐶 𝑛𝜎−1

𝑟=1

for some constant 𝐶 > 0 depending only on 𝑘 and 𝜎. Because 𝑘𝜎 > 0, one has 𝑛𝜎−1 = 𝑜(𝑛(𝑘+1)𝜎−1) as 𝑛 → ∞. Hence there exist constants 𝑐′ > 0 and 𝑁𝑘 such that, for all 𝑛 ≥ 𝑁𝑘,

𝑛−1

∑

𝑟𝑘𝜎−1(𝑛 − 𝑟)𝜎−1 ≥ 𝑐′ 𝑛(𝑘+1)𝜎−1.

𝑟=𝑘

Therefore, for all 𝑛 ≥ 𝑁𝑘,

𝑢(∗(𝑘+1))𝜎 (𝑛) ≥ 𝑎𝑘𝑐′ 𝑛(𝑘+1)𝜎−1.

It remains to treat the finitely many values 𝑘 + 1 ≤ 𝑛 < 𝑁𝑘. For each such 𝑛, one has 𝑢(∗(𝑘+1))𝜎 (𝑛) > 0 because 𝑛 can be written as a sum of 𝑘 + 1 positive integers. Hence the ratio

𝑢(∗(𝑘+1))𝜎 (𝑛) 𝑛(𝑘+1)𝜎−1

is positive for each of those finitely many 𝑛. Taking the minimum of these finitely many positive ratios and 𝑎𝑘𝑐′ gives a constant 𝑎𝑘+1 > 0 such that

𝑢(∗(𝑘+1))𝜎 (𝑛) ≥ 𝑎𝑘+1 𝑛(𝑘+1)𝜎−1 for all 𝑛 ≥ 𝑘 + 1. This closes the induction.

Since 𝑓𝛽

= 𝑢𝜎 with 𝜎 = 1 − 𝛽tail, we obtain

tail)−1, 𝑛 ≥ 𝑘. This is (72)

𝑓𝛽(∗𝑘)

(𝑛) ≍ 𝑛𝑘(1−𝛽

tail

tail

| |
|---|

### H Deep Jacobian estimates

- H.1 Setup Fix a depth 𝑁layer ≥ 1, a finite horizon 𝑇, and a compact input set 𝒳0. Let

layer−1)), 𝑛layer = 1,…,𝑁layer, where each 𝐹𝑛

ℎ(0) = 𝑥 ∈ 𝒳0, ℎ(𝑛

layer) = 𝐹𝑛

(ℎ(𝑛

layer

layer−1 ∘ ⋯ ∘ 𝐹1(𝒳0). For each layer 𝑛layer and each 0 ≤ 𝜏 ≤ 𝑡 ≤ 𝑇 − 1, define the one-block Jacobian block

𝒳𝑛

layer−1 ∶= 𝐹𝑛

is causal and continuously differentiable on the relevant compact set

layer

𝜕𝐹𝑛

layer,𝑡(𝑢) 𝜕𝑢𝜏

layer−1. Define also the full end-to-end Jacobian blocks

∈ ℝ𝐷×𝐷, 𝑢 ∈ 𝒳𝑛

𝐽𝑡,𝜏(𝑛layer)(𝑢) ∶=

𝜕ℎ(𝑁𝑡 layer)(𝑥) 𝜕ℎ(0)𝜏 (𝑥)

∈ ℝ𝐷×𝐷.

𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥) ∶=

For scalar lower-triangular kernels 𝒜,ℬ on

{(𝑡,𝜏) ∶ 0 ≤ 𝜏 ≤ 𝑡 ≤ 𝑇 − 1}, we use the standard kernel product

𝑡

∑

(𝒜ℬ)(𝑡,𝜏) ∶=

𝒜(𝑡,𝑗)ℬ(𝑗,𝜏).

𝑗=𝜏

- Theorem 31 (Residual calculus). Assume that for each layer 𝑛layer there exist constants

- H.2 Residual calculus

≥ 0, and a scalar lower-triangular kernel

𝑑𝑛

≥ 0, 𝜆𝑛

layer

layer

𝐾𝑛

∶ {(𝑡,𝜏) ∶ 0 ≤ 𝜏 < 𝑡 ≤ 𝑇 − 1} → [0,∞)

such that for every 𝑢 ∈ 𝒳𝑛

layer−1 and every 0 ≤ 𝜏 ≤ 𝑡 ≤ 𝑇 − 1,

layer

‖𝐽𝑡,𝜏(𝑛layer)(𝑢)‖ ≤ 𝑑𝑛

1[𝑡 = 𝜏] + 𝜆𝑛

𝐾𝑛

(𝑡,𝜏)1[𝜏 < 𝑡]. (74)

layer

layer

layer

Then, for every 𝑥 ∈ 𝒳0, every 0 ≤ 𝜏 < 𝑡 ≤ 𝑇 − 1, and every depth 𝑁layer ≥ 1,

⎛⎜ ⎝

𝑑𝑚⎞⎟

𝑁layer

∑

∑

∏

‖𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥)‖ ≤

⎠ ⋅ ∑

1≤𝑛layer,1<⋯<𝑛layer,𝑘≤𝑁layer

𝑘=1

𝑚∉{𝑛layer,1,…,𝑛layer,𝑘}

𝑘

∏

𝜆𝑛

𝐾𝑛

(𝑖𝑟,𝑖𝑟−1). (75)

𝜏=𝑖0<𝑖1<⋯<𝑖𝑘=𝑡

𝑟=1

layer,𝑟

layer,𝑟

Moreover, for the diagonal blocks one has

𝑁layer

‖𝐽𝑡,𝑡e2e,(𝑁layer)(𝑥)‖ ≤

∏

𝑑𝑛

.

𝑛layer=1

layer

Proof. For each layer 𝑛layer, define the scalar diagonal kernel 𝒟𝑛

1[𝑡 = 𝜏], and the scalar strictly lower-triangular kernel

(𝑡,𝜏) ∶= 𝑑𝑛

layer

layer

(𝑡,𝜏)1[𝜏 < 𝑡]. Then (74) says precisely that

𝒢𝑛

(𝑡,𝜏) ∶= 𝜆𝑛

𝐾𝑛

layer

layer

layer

‖𝐽𝑡,𝜏(𝑛layer)(𝑢)‖ ≤ 𝒟𝑛

(𝑡,𝜏) + 𝒢𝑛

(𝑡,𝜏) ∀𝑢 ∈ 𝒳𝑛

layer−1.

We prove by induction on the depth 𝑝 ∈ {1,…,𝑁layer} that

layer

layer

𝜕ℎ(𝑝)𝑡 𝜕ℎ(0)𝜏 (𝑥)

∥

∥ ≤ [(𝒟𝑝 + 𝒢𝑝)⋯(𝒟1 + 𝒢1)](𝑡,𝜏) (0 ≤ 𝜏 ≤ 𝑡 ≤ 𝑇 − 1). (76)

For 𝑝 = 1, (76) is exactly (74) evaluated at 𝑢 = 𝑥 ∈ 𝒳0. Assume now that (76) holds for some 𝑝 − 1 ≥ 1. By the chain rule,

𝜕ℎ(𝑝−1)𝑗 (𝑥) 𝜕ℎ(0)𝜏 (𝑥)

𝜕𝐹𝑝,𝑡(ℎ(𝑝−1)(𝑥)) 𝜕ℎ(𝑝−1)𝑗 (𝑥)

𝜕ℎ(𝑝)𝑡 𝜕ℎ(0)𝜏 (𝑥)

𝑡

=

∑

⋅

.

𝑗=𝜏

𝜕ℎ(𝑝−1)𝑗 (𝑥) 𝜕ℎ(0)𝜏 (𝑥)

Taking operator norms and using submultiplicativity gives

𝜕𝐹𝑝,𝑡(ℎ(𝑝−1)(𝑥)) 𝜕ℎ(𝑝−1)𝑗 (𝑥)

(𝑝) 𝑡

𝑡

∥ ≤

∑

∥

∥ ⋅ ∥

∥.

∥

𝜕ℎ(0)𝜏 (𝑥)

𝑗=𝜏

Since ℎ(𝑝−1)(𝑥) ∈ 𝒳𝑝−1, the one-block bound (74) applies:

𝜕𝐹𝑝,𝑡(ℎ(𝑝−1)(𝑥)) 𝜕ℎ(𝑝−1)𝑗 (𝑥)

∥ ≤ 𝒟𝑝(𝑡,𝑗) + 𝒢𝑝(𝑡,𝑗).

∥

𝜕ℎ(𝑝)𝑡 𝜕ℎ(0)𝜏 (𝑥)

Using the induction hypothesis for the second factor, we get

𝑡

∥

∥ ≤

∑

(𝒟𝑝 + 𝒢𝑝)(𝑡,𝑗)[(𝒟𝑝−1 + 𝒢𝑝−1)⋯(𝒟1 + 𝒢1)](𝑗,𝜏).

𝑗=𝜏

[(𝒟𝑝 + 𝒢𝑝)⋯(𝒟1 + 𝒢1)](𝑡,𝜏),

This is exactly

which proves (76) for depth 𝑝. Taking 𝑝 = 𝑁layer yields

‖𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥)‖ ≤ [(𝒟𝑁

+ 𝒢𝑁

)⋯(𝒟1 + 𝒢1)](𝑡,𝜏).

We now expand the right-hand side. Since each 𝒟𝑛

𝐼 as a kernel, one has the exact product expansion

is diagonal and equals 𝑑𝑛

layer

layer

layer

layer

→

(𝒟𝑁

+ 𝒢𝑁

)⋯(𝒟1 + 𝒢1) = ∑

( ∏

𝑑𝑚)

∏

𝒢𝑛

,

𝑛layer∈𝑆

𝑚∉𝑆

𝑆⊆{1,…,𝑁layer}

where the ordered product is taken in increasing layer order. For 𝜏 < 𝑡, the empty-set term vanishes because it is purely diagonal. Thus

layer

layer

layer

⎛⎜ ⎝

𝑑𝑚⎞⎟ ⎠

𝑁layer

‖𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥)‖ ≤

∑

∑

∏

(𝒢𝑛

⋯𝒢𝑛

)(𝑡,𝜏).

1≤𝑛layer,1<⋯<𝑛layer,𝑘≤𝑁layer

𝑘=1

layer,𝑘

layer,1

𝑚∉{𝑛layer,1,…,𝑛layer,𝑘}

Finally, by repeated expansion of the kernel product,

𝑘

∏

(𝒢𝑛

⋯𝒢𝑛

)(𝑡,𝜏) = ∑

𝜆𝑛

𝐾𝑛

(𝑖𝑟,𝑖𝑟−1),

𝜏=𝑖0<𝑖1<⋯<𝑖𝑘=𝑡

𝑟=1

layer,𝑘

layer,1

layer,𝑟

layer,𝑟

which gives (75). For the diagonal blocks 𝜏 = 𝑡, only the empty-set term survives, hence

𝑁layer

‖𝐽𝑡,𝑡e2e,(𝑁layer)(𝑥)‖ ≤

∏

𝑑𝑛

.

𝑛layer=1

layer

| |
|---|

For diffuse Transformer blocks the one-block kernel depends on the query time 𝑡. The next lemma gives the corresponding convolution bound for

- H.3 A harmonic-kernel bound

1 𝑡 + 1

ℋ(𝑡,𝜏) ∶=

1[𝜏 < 𝑡].

Lemma H.1 (Nested harmonic bound). Fix 𝑘 ≥ 1 and define

1 𝑡 + 1

ℋ(𝑡,𝜏) ∶=

1[𝜏 < 𝑡].

Then for every 0 ≤ 𝜏 < 𝑡 ≤ 𝑇 − 1,

1 𝑡 + 1

𝐻𝑡𝑘−1 (𝑘 − 1)!

, (77) where

(ℋ𝑘)(𝑡,𝜏) ≤

⋅

1 𝑚

𝑡

𝐻𝑡 ∶=

∑

𝑚=1

is the 𝑡-th harmonic number, with the convention 𝐻0 ∶= 0. Consequently, for every fixed 𝑘,

(log(1 + 𝑡))𝑘−1 𝑡 + 1

(ℋ𝑘)(𝑡,𝜏) ≲𝑘

.

Proof. For 𝑘 = 1 the claim is immediate:

1 𝑡 + 1

1 𝑡 + 1

ℋ(𝑡,𝜏) =

1[𝜏 < 𝑡] ≤

.

Assume now 𝑘 ≥ 2. By the kernel-product expansion,

1 𝑖𝑟 + 1

𝑘

(ℋ𝑘)(𝑡,𝜏) = ∑

∏

.

𝜏=𝑖0<𝑖1<⋯<𝑖𝑘=𝑡

𝑟=1

Since 𝑖𝑘 = 𝑡, the last factor is exactly 𝑡+11 , hence

1 𝑡 + 1

1 𝑖𝑟 + 1

𝑘−1

(ℋ𝑘)(𝑡,𝜏) =

∏

∑

.

𝜏<𝑖1<⋯<𝑖𝑘−1<𝑡

𝑟=1

Dropping the lower bound 𝜏 only enlarges the sum, so

1 𝑡 + 1

1 𝑖𝑟 + 1

𝑘−1

∑

∏

(ℋ𝑘)(𝑡,𝜏) ≤

.

0<𝑖1<⋯<𝑖𝑘−1<𝑡

𝑟=1

1 𝑚 + 1

𝑘−1

𝑡−1

(

∑

)

.

Now expand

𝑚=1

Every strictly increasing (𝑘 − 1)-tuple

0 < 𝑖1 < ⋯ < 𝑖𝑘−1 < 𝑡 appears exactly (𝑘 − 1)! times among the ordered monomials in this expansion. Therefore

1 𝑖𝑟 + 1

1 (𝑘 − 1)!

1 𝑚 + 1

𝐻𝑡𝑘−1 (𝑘 − 1)!

𝑘−1

𝑡−1

𝑘−1

∑

∏

(

∑

)

≤

≤

.

0<𝑖1<⋯<𝑖𝑘−1<𝑡

𝑟=1

𝑚=1

1 𝑡 + 1

𝐻𝑡𝑘−1 (𝑘 − 1)!

Substituting this into the previous display gives

(ℋ𝑘)(𝑡,𝜏) ≤

⋅

,

which is (77). Since 𝐻𝑡 ≲ log(1 + 𝑡), the logarithmic form follows.

| |
|---|

#### H.4 Model-specific bounds

- Proposition 32 (Deep Transformer bound). Assume the hypotheses of Theorem 31. Assume in addition that

for each layer 𝑛layer there exists 𝑎𝑛

layer

> 0 such that

𝐾𝑛

layer

(𝑡,𝜏) ≤

𝑎𝑛

layer

𝑡 + 1

, 𝜏 < 𝑡.

Fix a bounded source family 0 ≤ 𝜏 ≤ 𝜏max. Then for every 𝑥 ∈ 𝒳0 and every ℓ ≥ 1 with 𝜏 + ℓ ≤ 𝑇 − 1,

∥𝐽𝜏+ℓ,𝜏e2e,(𝑁layer)(𝑥)∥ ≲𝜏

max,𝑁layer

(log(1 + ℓ))𝑁

layer−1

1 + ℓ

.

Proof. Fix an ordered layer subset

1 ≤ 𝑛layer,1 < ⋯ < 𝑛layer,𝑘 ≤ 𝑁layer. Define

ℋ(𝑡,𝜏) ∶=

1 𝑡 + 1

1[𝜏 < 𝑡]. By the assumption on 𝐾𝑛

layer

,

𝐾𝑛

layer,𝑟

(𝑖𝑟,𝑖𝑟−1) ≤ 𝑎𝑛

layer,𝑟

ℋ(𝑖𝑟,𝑖𝑟−1) ∀𝑟. Therefore

∑

𝜏=𝑖0<⋯<𝑖𝑘=𝑡

𝑘

∏

𝑟=1

𝜆𝑛

layer,𝑟

𝐾𝑛

layer,𝑟

(𝑖𝑟,𝑖𝑟−1) ≤ (

𝑘

∏

𝑟=1

𝜆𝑛

layer,𝑟

𝑎𝑛

layer,𝑟

)(ℋ𝑘)(𝑡,𝜏).

By Lemma H.1,

(ℋ𝑘)(𝑡,𝜏) ≲𝑘

(log(1 + 𝑡))𝑘−1 𝑡 + 1

. Insert this estimate into Theorem 31:

‖𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥)‖ ≲𝑁

layer

𝑁layer

∑

𝑘=1

∑

1≤𝑛layer,1<⋯<𝑛layer,𝑘≤𝑁layer

⎛⎜ ⎝

∏

𝑚∉{𝑛layer,1,…,𝑛layer,𝑘}

𝑑𝑚⎞⎟ ⎠

(

𝑘

∏

𝑟=1

𝜆𝑛

layer,𝑟

𝑎𝑛

layer,𝑟

)

(log(1 + 𝑡))𝑘−1 𝑡 + 1

.

Since 𝑁layer is fixed, the finite sum is bounded by

𝐶𝑁

layer

(log(1 + 𝑡))𝑁

layer−1

𝑡 + 1

.

Now restrict to the bounded source family 0 ≤ 𝜏 ≤ 𝜏max and set 𝑡 = 𝜏 + ℓ. Then 𝑡 + 1 = 𝜏 + ℓ + 1 ≍𝜏

max

1 + ℓ, log(1 + 𝑡) ≍𝜏

max

log(1 + ℓ),

uniformly for 0 ≤ 𝜏 ≤ 𝜏max. Hence

∥𝐽𝜏+ℓ,𝜏e2e,(𝑁layer)(𝑥)∥ ≲𝜏

max,𝑁layer

(log(1 + ℓ))𝑁

layer−1

1 + ℓ

.

| |
|---|

- Proposition 33 (Deep Mamba bound under failed freeze time). Assume the hypotheses of Theorem 31. Assume

in addition that for each layer 𝑛layer there exist 𝑎𝑛

> 0 and 𝑐𝑛

> 0 such that

𝐾𝑛

(𝑡,𝜏) ≤ 𝑎𝑛

𝑒−𝑐

𝑛layer(𝑡−𝜏), 𝜏 < 𝑡.

layer

layer

layer

layer

𝑐∗ ∶= min

𝑐𝑛

.

Set

1≤𝑛layer≤𝑁layer

Then for every 𝑥 ∈ 𝒳0 and every 𝜏 < 𝑡,

layer

∥𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥)∥ ≲𝑁

(1 + 𝑡 − 𝜏)𝑁

layer−1𝑒−𝑐∗(𝑡−𝜏).

layer

1 ≤ 𝑛layer,1 < ⋯ < 𝑛layer,𝑘 ≤ 𝑁layer and write ℓ ∶= 𝑡 − 𝜏. For every temporal path 𝜏 = 𝑖0 < ⋯ < 𝑖𝑘 = 𝑡, one has

Proof. Fix an ordered layer subset

𝑘

𝑘

𝑘

𝑘

∏

∏

)exp(−

∑

∏

)𝑒−𝑐∗ℓ.

𝐾𝑛

(𝑖𝑟,𝑖𝑟−1) ≤ (

𝑎𝑛

𝑐𝑛

(𝑖𝑟 − 𝑖𝑟−1)) ≤ (

𝑎𝑛

𝑟=1

𝑟=1

𝑟=1

𝑟=1

layer,𝑟

layer,𝑟

layer,𝑟

layer,𝑟

𝜏 = 𝑖0 < 𝑖1 < ⋯ < 𝑖𝑘 = 𝑡 is the number of compositions of ℓ into 𝑘 positive integers, namely

The number of strictly increasing temporal paths

ℓ − 1 𝑘 − 1

(

),

with the convention that this is 0 if ℓ < 𝑘. Therefore

ℓ − 1 𝑘 − 1

𝑘

𝑘

∏

𝜆𝑛

𝐾𝑛

(𝑖𝑟,𝑖𝑟−1) ≤ (

∏

𝜆𝑛

𝑎𝑛

)(

∑

)𝑒−𝑐∗ℓ.

𝜏=𝑖0<⋯<𝑖𝑘=𝑡

𝑟=1

𝑟=1

layer,𝑟

layer,𝑟

layer,𝑟

layer,𝑟

Insert this estimate into Theorem 31:

ℓ − 1 𝑘 − 1

⎛⎜ ⎝

𝑑𝑚⎞⎟ ⎠

𝑁layer

𝑘

∑

∑

∏

(

∏

)(

)𝑒−𝑐∗ℓ.

‖𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥)‖ ≤

𝜆𝑛

𝑎𝑛

𝑟=1

1≤𝑛layer,1<⋯<𝑛layer,𝑘≤𝑁layer

𝑘=1

layer,𝑟

layer,𝑟

𝑚∉{𝑛layer,1,…,𝑛layer,𝑘}

Since 𝑁layer is fixed and

ℓ − 1 𝑘 − 1

) ≲𝑘 (1 + ℓ)𝑘−1, the finite sum is bounded by a constant multiple of

(

(1 + ℓ)𝑁

layer−1𝑒−𝑐∗ℓ.

| |
|---|

layer 𝑛layer there exist 𝑎𝑛

> 0 and a common exponent 𝛽tail ∈ (0,1) such that

- Proposition 34 (Deep Sessa bound). Assume the hypotheses of Theorem 31. Assume in addition that for each

(1 + log(1 + 𝑡 − 𝜏)), 𝜏 < 𝑡.

𝐾𝑛

(𝑡,𝜏) ≤ 𝑎𝑛

(𝑡 − 𝜏)−𝛽

layer

Then for every 𝑥 ∈ 𝒳0 and every 𝜏 < 𝑡,

tail

layer

layer

𝑁layer

∑

(𝑡 − 𝜏)𝑘(1−𝛽

tail)−1(1 + log(1 + 𝑡 − 𝜏))𝑘.

∥𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥)∥ ≲𝑁

layer,𝛽tail

𝑘=1

In particular, since 𝑁layer is fixed,

∥𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥)∥ ≲𝑁

layer,𝛽tail (𝑡 − 𝜏)𝑁

layer(1−𝛽tail)−1(1 + log(1 + 𝑡 − 𝜏))𝑁layer.

Proof. Fix 𝜏 < 𝑡 and write ℓ ∶= 𝑡 − 𝜏. Fix an ordered layer subset

1 ≤ 𝑛layer,1 < ⋯ < 𝑛layer,𝑘 ≤ 𝑁layer. For every temporal path 𝜏 = 𝑖0 < ⋯ < 𝑖𝑘 = 𝑡, set

𝑚𝑟 ∶= 𝑖𝑟 − 𝑖𝑟−1 ∈ ℕ∗. Then

𝑚1 + ⋯ + 𝑚𝑘 = ℓ. Using the bound on 𝐾𝑛

,

𝑘

𝑘

𝑘

layer

∏

∏

)

∏

𝑟 tail(1 + log(1 + 𝑚𝑟)).

𝐾𝑛

(𝑖𝑟,𝑖𝑟−1) ≤ (

𝑎𝑛

𝑚−𝛽

𝑟=1

𝑟=1

𝑟=1

layer,𝑟

layer,𝑟

Since every 𝑚𝑟 ≤ ℓ, one has

1 + log(1 + 𝑚𝑟) ≤ 1 + log(1 + ℓ). Therefore

𝑘

𝑘

𝑘

∏

𝑚−𝛽

𝑟 tail.

∏

𝐾𝑛

(𝑖𝑟,𝑖𝑟−1) ≤ (

∏

𝑎𝑛

)(1 + log(1 + ℓ))𝑘

𝑟=1

𝑟=1

𝑟=1

layer,𝑟

layer,𝑟

Summing over all temporal paths from 𝜏 to 𝑡 gives

𝑘

∏

𝜆𝑛

𝐾𝑛

(𝑖𝑟,𝑖𝑟−1)

∑

𝜏=𝑖0<⋯<𝑖𝑘=𝑡

𝑟=1

layer,𝑟

layer,𝑟

𝑘

≤ (

∏

𝜆𝑛

𝑎𝑛

)(1 + log(1 + ℓ))𝑘 ∑

𝑚−𝛽

1 ⋯𝑚−𝛽

𝑘 .

𝑟=1

1,…,𝑚𝑘≥1 𝑚1+⋯+𝑚𝑘=ℓ

layer,𝑟

layer,𝑟

tail

tail

The remaining sum is exactly the 𝑘-fold positive-lag convolution

. By Theorem 30,

(𝑛) ∶= 𝑛−𝛽

𝑓𝛽(∗𝑘)

(ℓ), 𝑓𝛽

tail

tail

tail

tail)−1. Hence

𝑓𝛽(∗𝑘)

(ℓ) ≲𝑘,𝛽

ℓ𝑘(1−𝛽

tail

tail

𝑘

𝑘

∏

𝜆𝑛

𝐾𝑛

(𝑖𝑟,𝑖𝑟−1) ≲𝑘,𝛽

(

∏

𝜆𝑛

𝑎𝑛

)ℓ𝑘(1−𝛽

tail)−1(1 + log(1 + ℓ))𝑘.

∑

𝜏=𝑖0<⋯<𝑖𝑘=𝑡

𝑟=1

𝑟=1

layer,𝑟

layer,𝑟

layer,𝑟

layer,𝑟

tail

𝑘 = 1,…,𝑁layer. Since 𝑁layer is fixed, the finite sum yields the stated bound.

Insert this estimate into Theorem 31 and sum over

The final simplified estimate follows because, for 𝛽tail ∈ (0,1), the exponent 𝑘(1 − 𝛽tail) − 1 is increasing in 𝑘, so the 𝑘 = 𝑁layer term dominates the smaller-𝑘 terms up to a constant.

| |
|---|

- H.5 Horizon-uniform bounds We now state the horizon-uniform version used in Section 4.2.7. Theorem 35 (Horizon-uniform residual calculus). Fix a depth 𝑁layer ≥ 1. For each horizon 𝑇 ≥ 1, let

ℎ(0,𝑇) = 𝑥 ∈ 𝒳(𝑇)0 , ℎ(𝑛

layer,𝑇) = 𝐹𝑛(𝑇)

(ℎ(𝑛

layer−1,𝑇)), 𝑛layer = 1,…,𝑁layer,

where 𝒳(𝑇)0 ⊂ (ℝ𝐷)𝑇 is compact and each 𝐹𝑛(𝑇)

layer

is causal and continuously differentiable on the relevant compact set

layer−1 ∘ ⋯ ∘ 𝐹1(𝑇)(𝒳(𝑇)0 ). Define the full end-to-end Jacobian blocks by

𝒳(𝑇)𝑛

layer−1 ∶= 𝐹𝑛(𝑇)

layer

𝜕ℎ(𝑁𝑡 layer,𝑇)(𝑥) 𝜕ℎ(0,𝑇)𝜏 (𝑥)

∈ ℝ𝐷×𝐷, 0 ≤ 𝜏 ≤ 𝑡 ≤ 𝑇 − 1.

𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥;𝑇) ∶=

Assume that for each layer 𝑛layer there exist constants 𝑑𝑛

≥ 0, independent of 𝑇, and a scalar lower-triangular kernel

≥ 0, 𝜆𝑛

layer

layer

𝐾𝑛

∶ {(𝑡,𝜏) ∶ 0 ≤ 𝜏 < 𝑡 < ∞} → [0,∞)

independent of 𝑇, such that for every horizon 𝑇 ≥ 1, every 𝑢 ∈ 𝒳(𝑇)𝑛

layer−1, and every 0 ≤ 𝜏 ≤ 𝑡 ≤ 𝑇 − 1,

layer

𝜕𝐹𝑛(𝑇) layer,𝑡(𝑢) 𝜕𝑢𝜏

∥

∥ ≤ 𝑑𝑛

1[𝑡 = 𝜏] + 𝜆𝑛

𝐾𝑛

(𝑡,𝜏)1[𝜏 < 𝑡].

layer

layer

layer

Then for every horizon 𝑇 ≥ 1, every 𝑥 ∈ 𝒳(𝑇)0 , and every 0 ≤ 𝜏 < 𝑡 ≤ 𝑇 − 1,

⎛⎜ ⎝

𝑑𝑚⎞⎟

𝑁layer

∥𝐽𝑡,𝜏e2e,(𝑁layer)(𝑥;𝑇)∥ ≤

∑

∑

∏

⎠ ⋅ ∑

1≤𝑛layer,1<⋯<𝑛layer,𝑘≤𝑁layer

𝑘=1

𝑚∉{𝑛layer,1,…,𝑛layer,𝑘}

𝑘

∏

𝜆𝑛

𝐾𝑛

(𝑖𝑟,𝑖𝑟−1). (78)

𝜏=𝑖0<𝑖1<⋯<𝑖𝑘=𝑡

𝑟=1

layer,𝑟

layer,𝑟

𝑁layer

∥𝐽𝑡,𝑡e2e,(𝑁layer)(𝑥;𝑇)∥ ≤

∏

𝑑𝑛

.

Moreover,

𝑛layer=1

In particular, the right-hand side of (78) is independent of 𝑇.

layer

Proof. Fix a horizon 𝑇 ≥ 1. Apply Theorem 31 to the horizon-𝑇 stack

𝐹1(𝑇),…,𝐹𝑁(𝑇)

on the compact input set 𝒳(𝑇)0 . The hypotheses of Theorem 31 are satisfied with the same layerwise constants 𝑑𝑛

layer

, because these are assumed to be independent of 𝑇. Therefore, for this fixed horizon 𝑇, Theorem 31 gives exactly the path-sum bound (78) and the same diagonal estimate. Since the displayed right-hand side contains no dependence on 𝑇, the same bound holds verbatim for every horizon 𝑇 ≥ 1.

,𝜆𝑛

and the same kernels 𝐾𝑛

layer

layer

layer

| |
|---|

- (i) Transformer. Assume that for each layer 𝑛layer there exists 𝑎𝑛

layer

> 0 such that

𝐾𝑛

layer

(𝑡,𝜏) ≤

𝑎𝑛

layer

𝑡 + 1

, 𝜏 < 𝑡.

Fix a bounded source family 0 ≤ 𝜏 ≤ 𝜏max. Then

sup

𝑇≥𝜏max+ℓ+1

sup

0≤𝜏≤𝜏max

sup

𝑥∈𝒳(𝑇)0

∥𝐽𝜏+ℓ,𝜏e2e,(𝑁layer)(𝑥;𝑇)∥ ≲𝜏

max,𝑁layer

(log(1 + ℓ))𝑁

layer−1

1 + ℓ

.

- (ii) Mamba. Assume that for each layer 𝑛layer there exist 𝑎𝑛

layer

> 0 and 𝑐𝑛

layer

> 0 such that

𝐾𝑛

layer

(𝑡,𝜏) ≤ 𝑎𝑛

layer

𝑒−𝑐

𝑛layer(𝑡−𝜏), 𝜏 < 𝑡.

Set 𝑐∗ ∶= min𝑛

layer

𝑐𝑛

layer

. Then

sup

𝑇≥ℓ+1

sup

0≤𝜏≤𝑇−ℓ−1

sup

𝑥∈𝒳(𝑇)0

∥𝐽𝜏+ℓ,𝜏e2e,(𝑁layer)(𝑥;𝑇)∥ ≲𝑁

layer

(1 + ℓ)𝑁

layer−1𝑒−𝑐∗ℓ.

- (iii) Sessa. Assume that for each layer 𝑛layer there exist 𝑎𝑛

- Corollary H.2 (Horizon-uniform decay bounds). Assume the hypotheses of Theorem 35.

> 0 and a common exponent 𝛽tail ∈ (0,1) such that

(1 + log(1 + 𝑡 − 𝜏)), 𝜏 < 𝑡. Then

𝐾𝑛

(𝑡,𝜏) ≤ 𝑎𝑛

(𝑡 − 𝜏)−𝛽

layer

tail

layer

layer

𝑁layer

∑

tail)−1(1 + log(1 + ℓ))𝑘.

∥𝐽𝜏+ℓ,𝜏e2e,(𝑁layer)(𝑥;𝑇)∥ ≲𝑁

ℓ𝑘(1−𝛽

layer,𝛽tail

𝑇≥ℓ+1

0≤𝜏≤𝑇−ℓ−1

𝑥∈𝒳(𝑇)0

𝑘=1

sup

sup

sup

In particular, if 𝑁layer(1 − 𝛽tail) < 1, then the right-hand side tends to 0 as ℓ → ∞.

Proof. Apply Theorem 35 and then repeat exactly the kernel-class estimates used in the proofs of Propositions 32, 33, and 34. Because the layerwise envelope parameters are horizon-uniform, the resulting constants are independent of 𝑇. Taking the indicated suprema over all admissible horizons therefore leaves the bounds unchanged. For the Transformer case, the passage from 𝑡 = 𝜏 + ℓ to 1 + ℓ is uniform on bounded-source families 0 ≤ 𝜏 ≤ 𝜏max. For the Sessa case, the final asymptotic decay to 0 occurs exactly when the largest power

ℓ𝑁

layer(1−𝛽tail)−1

has negative exponent, i.e. when 𝑁layer(1 − 𝛽tail) < 1.

| |
|---|

### I Universal approximation for Sessa with adapters

#### I.1 Preliminaries and notation Fix 𝑇 ≥ 3 and 𝑑ext ∈ ℕ∗. Inputs are

, and outputs are in ℝ𝑇×𝑑

𝑥 = (𝑥0,…,𝑥𝑇−1) ∈ (ℝ𝑑

)𝑇 ≅ ℝ𝑇×𝑑

ext. For 𝑋 ∈ ℝ𝑇×𝑑

ext

ext

ext define

𝑇−1

‖𝑋‖2𝐹 =

∑

‖𝑋𝑡‖2.

𝑡=0

Let 𝒟 ⊂ ℝ𝑇×𝑑

𝑀𝒟 ∶= sup 𝑥∈𝒟

‖𝑥‖𝐹 < ∞.

ext be compact and

Hence ‖𝑥𝑡‖2 ≤ 𝑀𝒟 for all 𝑥 ∈ 𝒟 and all 𝑡. Definition 10 (Causality). 𝐹 ∶ 𝒟 → ℝ𝑇×𝑑

ext is causal if for every 𝑡 and all 𝑥,𝑥′ ∈ 𝒟, 𝑥0∶𝑡 = 𝑥′0∶𝑡 implies 𝐹(𝑥)𝑡 = 𝐹(𝑥′)𝑡.

𝒟 ⊂ ℝ𝑇×𝑑

- Lemma I.1 (Prefix factorization of continuous causal maps). Let

ext

𝐹 ∶ 𝒟 → ℝ𝑇×𝑑

be compact and let

be continuous and causal. For each 𝑡 ∈ {0,…,𝑇 − 1}, define

ext

)𝑡+1, 𝑝𝑡(𝑥) ∶= 𝑥0∶𝑡, and

𝑝𝑡 ∶ 𝒟 → (ℝ𝑑

ext

𝒫pref𝑡 ∶= 𝑝𝑡(𝒟). Then there exists a unique continuous map

𝐹̂𝑡 ∶ 𝒫pref𝑡 → ℝ𝑑

ext

𝐹̂𝑡(𝑥0∶𝑡) = 𝐹(𝑥)𝑡 ∀𝑥 ∈ 𝒟.

such that

Proof. Uniqueness is immediate because 𝑝𝑡 is surjective onto 𝒫pref𝑡 . Causality ensures that 𝐹̂𝑡 is well defined: if 𝑝𝑡(𝑥) = 𝑝𝑡(𝑥′), then 𝑥0∶𝑡 = 𝑥′0∶𝑡, hence

𝐹(𝑥)𝑡 = 𝐹(𝑥′)𝑡.

, pr𝑡(𝑦) ∶= 𝑦𝑡, and define

pr𝑡 ∶ ℝ𝑇×𝑑

###### → ℝ𝑑

Let

ext

ext

. Then

𝑔𝑡 ∶= pr𝑡 ∘𝐹 ∶ 𝒟 → ℝ𝑑

ext

𝑔𝑡 = 𝐹̂𝑡 ∘ 𝑝𝑡.

Let 𝐶 ⊂ ℝ𝑑

ext be closed. Since 𝑔𝑡 is continuous, 𝑔𝑡−1(𝐶) is closed in the compact set 𝒟, hence compact. Applying 𝑝𝑡, the image

𝑝𝑡(𝑔𝑡−1(𝐶)) is compact in 𝒫pref𝑡 , hence closed because 𝒫pref𝑡 is Hausdorff. Moreover, 𝐹̂𝑡−1(𝐶) = 𝑝𝑡(𝑔𝑡−1(𝐶)).

Therefore 𝐹̂𝑡 is continuous.

| |
|---|

Sessa blocks of width 𝑚 Fix an even query–key width 𝑑𝑘 ∈ 2ℕ, a model width 𝑚 ∈ ℕ∗, and a tokenwise pre-normalization map

- I.2 Architecture and function classes

Norm ∶ ℝ𝑚 → ℝ𝑚 applied independently to each token. We consider two choices:

Norm = Id and Norm = LN𝜀

(𝜀ln > 0).

A width-𝑚 Sessa block is the block of Section 3 specialized to model width 𝑚, and we use the following RoPE convention throughout this section.

ln

Write every 𝑧 ∈ ℝ𝑑𝑘 as

𝑧 = (𝑧(0),𝑧(1),…,𝑧(𝑑𝑘/2−1)), 𝑧(𝑟) ∈ ℝ2. Fix a RoPE base 𝜗 > 1 and define the standard pairwise frequencies

𝜔𝑟 ∶= 𝜗−2𝑟/𝑑𝑘, 𝑟 = 0,…,𝑑𝑘/2 − 1. In particular,

𝜔0 = 1. For every 𝜏 ∈ ℝ define

RoPE𝜏(𝑧) ∶= (𝑅𝜔

0𝜏𝑧(0),𝑅𝜔

1𝜏𝑧(1),…,𝑅𝜔

𝑑𝑘/2−1𝜏𝑧(𝑑𝑘/2−1)),

where 𝑅𝜃 denotes the planar rotation by angle 𝜃. In the architecture, 𝜏 = 𝑡 ∈ {0,…,𝑇 − 1}; in the constructions below we also allow shifts such as 𝜏 = −ℓ. All diagonalization arguments use only the first rotary pair. Hence, whenever 𝑞,𝑘 ∈ ℝ𝑑𝑘 are supported on that first pair,

⟨RoPE𝑡(𝑞),RoPE𝑗(𝑘)⟩ = ⟨𝑅𝑡𝑞1∶2, 𝑅𝑗𝑘1∶2⟩. The comparison RoPE-Transformer class uses the same convention.

𝑊in ∈ ℝ𝑚×2𝑚, 𝑏in ∈ ℝ2𝑚, 𝑊out ∈ ℝ𝑚×𝑚, 𝑏out ∈ ℝ𝑚,

Parameters and dimensions

𝑊𝑄𝑓,𝑊𝐾𝑓,𝑊𝑄𝑏,𝑊𝐾𝑏 ∈ ℝ𝑚×𝑑𝑘, 𝑊𝑉 ∈ ℝ𝑚×𝑚, 𝑤𝛾 ∈ ℝ𝑚, 𝑏𝛾 ∈ ℝ.

##### Tokenwise preprocessing Given 𝑥 ∈ ℝ𝑇×𝑚:

𝑥̃𝑡 = Norm(𝑥𝑡) ∈ ℝ𝑚,

𝑡 = 𝑥̃𝑡𝑊in + 𝑏in ∈ ℝ2𝑚, 𝑢𝑡 = (𝑎𝑡,𝑔𝑡), 𝑎𝑡,𝑔𝑡 ∈ ℝ𝑚, 𝑎̄𝑡 = GELU(𝑎𝑡) ∈ ℝ𝑚.

𝜎𝑘 ∶= 𝑑𝑘−1/2. Define

Attention-feedback operator We fix the attention scale to

𝑞𝑡𝑓 = 𝑎̄𝑡𝑊𝑄𝑓, 𝑘𝑡𝑓 = 𝑎̄𝑡𝑊𝐾𝑓, 𝑣𝑡 = 𝑎̄𝑡𝑊𝑉 , 𝑞𝑡𝑏 = 𝑎̄𝑡𝑊𝑄𝑏, 𝑘𝑡𝑏 = 𝑎̄𝑡𝑊𝐾𝑏, with

𝑞𝑡𝑓,𝑘𝑡𝑓,𝑞𝑡𝑏,𝑘𝑡𝑏 ∈ ℝ𝑑𝑘, 𝑣𝑡 ∈ ℝ𝑚. For the causal forward branch (𝑗 ≤ 𝑡), define

𝑞̃𝑡𝑓 = RoPE𝑡(𝑞𝑡𝑓), 𝑘̃𝑗𝑓 = RoPE𝑗(𝑘𝑗𝑓), and define

exp(𝜎𝑘⟨𝑞̃𝑡𝑓,𝑘̃𝑗𝑓⟩)1[𝑗 ≤ 𝑡] ∑𝜏≤𝑡 exp(𝜎𝑘⟨𝑞̃𝑡𝑓,𝑘̃𝜏𝑓⟩)

𝛼fwd𝑡,𝑗 =

𝛼fwd𝑡,𝑗 𝑣𝑗.

, 𝑓𝑡 = ∑ 𝑗≤𝑡

For the strictly lower feedback branch (𝑗 < 𝑡), define

exp(𝜎𝑘⟨𝑞𝑡𝑏,𝑘𝑗𝑏⟩)1[𝑗 < 𝑡] ∑𝜏<𝑡 exp(𝜎𝑘⟨𝑞𝑡𝑏,𝑘𝜏𝑏⟩)

𝛼fb𝑡,𝑗 =

, 𝛼fb0,⋅ = 0.

𝛾𝑡 = tanh(⟨𝑎̄𝑡,𝑤𝛾⟩ + 𝑏𝛾) ∈ (−1,1). [𝐵fb]𝑡,𝑗 = 𝛾𝑡𝛼fb𝑡,𝑗, [𝐵fb]𝑡,𝑗 = 0 for 𝑗 ≥ 𝑡. The mixer output is defined by the exact solve

(𝐼 − 𝐵fb)𝑠 = 𝑓. Since 𝐵fb is strictly lower triangular, the system has a unique solution. Residual update

𝑦𝑡 = 𝑥𝑡 + ((𝑠𝑡 ⊙ 𝑔𝑡)𝑊out + 𝑏out). Function classes Let

ConcreteSessaBlocksNorm(𝑑𝑘,𝑚)

denote the set of all width-𝑚 concrete Sessa blocks above with the chosen pre-normalization map Norm. Define

Ω𝑑

Sessa,Norm(𝑚) ∶= {𝐺𝑁

∘ ⋯ ∘ 𝐺1 ∶ 𝐺𝑛

∈ ConcreteSessaBlocksNorm(𝑑𝑘,𝑚) for all 𝑛layer, 𝑁layer ∈ ℕ∗}.

𝑘

layer

layer

Tokenwise input and output adapters Fix the external data dimension 𝑑ext and a model width 𝑚 ≥ 𝑑ext. Define tokenwise aﬀine adapters

Embed(𝑥)𝑡 ∶= 𝑥𝑡𝑊emb + 𝑏emb ∈ ℝ𝑚, Unembed(ℎ)𝑡 ∶= ℎ𝑡𝑊un + 𝑏un ∈ ℝ𝑑

.

ext

𝑊emb ∈ ℝ𝑑

ext×𝑚, 𝑏emb ∈ ℝ𝑚, 𝑊un ∈ ℝ𝑚×𝑑

, 𝑏un ∈ ℝ𝑑

.

Parameters and dimensions

ext

ext

Unembed ∘ Embed = Id on ℝ𝑇×𝑑

.

ext

𝑥 ↦ Unembed(𝐺(Embed(𝑥))), with

We consider Sessa networks of the form

Sessa,Id(𝑚) in the main LN-free theorem, and

𝐺 ∈ Ω𝑑

𝑘

Sessa,LN𝜀ln(𝑚) in the LayerNorm extension.

𝐺 ∈ Ω𝑑

𝑘

Causal RoPE-Transformer class We also define a causal decoder-only RoPE-Transformer class of functions from ℝ𝑇×𝑑

→ ℝ𝑇×𝑑

ext, with internal model width 𝑚 and adapters.

A width-𝑚 RoPE-Transformer block is a standard decoder block operating on ℝ𝑇×𝑚: it consists of causal selfattention with 𝑗 ≤ 𝑡, RoPE applied to queries and keys in the logits, and fixed scale 𝜎𝑘 = 𝑑𝑘−1/2, together with a tokenwise FFN of hidden width 𝑟 and residual connections in ℝ𝑚. An absolute positional embedding 𝐸 ∈ ℝ𝑇×𝑚 is added once at the network input. Let Ω𝐻,𝑑

ext

RoPETr,cau(𝑚) be the set of finite compositions of such blocks

𝑘,𝑟

on ℝ𝑇×𝑚. Finally define the adapted function class

Ω𝐻,𝑑

RoPETr,cau(𝑑ext → 𝑚 → 𝑑ext) ∶= {𝑥 ↦ Unembed(𝑔̃(Embed(𝑥) + 𝐸)) ∶ 𝑔̃ ∈ Ω𝐻,𝑑

RoPETr,cau(𝑚), 𝐸 ∈ ℝ𝑇×𝑚}.

𝑘,𝑟

𝑘,𝑟

- Lemma I.2 (Softmax concentration). Let 𝑣 ∈ ℝ𝑛 and let 𝑖∗ = argmax𝑖 𝑣𝑖 be unique. Let Δ = 𝑣𝑖∗ −max𝑖≠𝑖∗ 𝑣𝑖 > 0 and fix 𝛿 ∈ (0,1). For 𝜎𝑘 > 0, define 𝑝 = softmax(𝜎𝑘𝑣).

- I.3 Softmax lemmas

𝑛 − 1 𝛿

𝑝𝑖∗ ≥ 1 − 𝛿 whenever 𝜎𝑘Δ ≥ log

.

∑𝑖≠𝑖

𝑒𝜎𝑘𝑣𝑖 ∑𝑖 𝑒𝜎𝑘𝑣𝑖 ≤

(𝑛 − 1)𝑒𝜎𝑘(𝑣𝑖∗−Δ) 𝑒𝜎𝑘𝑣𝑖∗

= (𝑛 − 1)𝑒−𝜎𝑘Δ.

1 − 𝑝𝑖∗ =

Proof.

∗

Thus 1 − 𝑝𝑖∗ ≤ 𝛿 if 𝜎𝑘Δ ≥ log 𝑛−1𝛿 .

- Corollary I.3 (Sharpening at fixed attention scale). Let 𝑣 ∈ ℝ𝑛 and let 𝑖∗ = argmax𝑖 𝑣𝑖 be unique. Let

| |
|---|

Δ = 𝑣𝑖∗ − max𝑖≠𝑖∗ 𝑣𝑖 > 0, fix 𝛿 ∈ (0,1), and fix the attention scale 𝜎𝑘 > 0. For 𝑐 > 0, define

𝑝(𝑐) ∶= softmax(𝜎𝑘 𝑐2𝑣). Then

𝑛 − 1 𝛿

𝑝𝑖(𝑐)∗ ≥ 1 − 𝛿 whenever 𝜎𝑘 𝑐2Δ ≥ log

.

Thus, in the concrete architecture where 𝜎𝑘 = 𝑑𝑘−1/2 is fixed, arbitrarily sharp softmax rows are obtained by scaling the query and key vectors by a common factor 𝑐.

Proof. Apply Lemma I.2 to the logits 𝑐2𝑣.

- Lemma I.4 (Error of an almost one-hot mixture). Let (𝑤𝑗)𝑗∈𝐽 ⊂ ℝ𝑚 and let 𝑝𝑗 ≥ 0, ∑𝑗∈𝐽 𝑝𝑗 = 1. If 𝑝𝑗∗ ≥ 1 − 𝛿 then

∥∑

𝑗∈𝐽

𝑝𝑗𝑤𝑗 − 𝑤𝑗∗∥

2

≤ 2𝛿 ⋅ 𝑉max,

where 𝑉max ∶= max𝑗∈𝐽 ‖𝑤𝑗‖2. Proof.

∑

𝑗

𝑝𝑗𝑤𝑗 − 𝑤𝑗∗ = (𝑝𝑗∗ − 1)𝑤𝑗∗ + ∑ 𝑗≠𝑗∗

𝑝𝑗𝑤𝑗.

Since ∑𝑗≠𝑗

∗

𝑝𝑗 = 1 − 𝑝𝑗∗ ≤ 𝛿,

∥∑

𝑗

𝑝𝑗𝑤𝑗 − 𝑤𝑗∗∥

2

≤ |1 − 𝑝𝑗∗|‖𝑤𝑗∗‖2 + ∑ 𝑗≠𝑗∗

𝑝𝑗‖𝑤𝑗‖2 ≤ 2𝛿 𝑉max,

where 𝑉max ∶= max𝑗 ‖𝑤𝑗‖2.

| |
|---|

I.4 RoPE diagonalization and triangular solve

- Lemma I.5 (RoPE-diagonalization). Fix 𝑇 ≥ 2 and an even query–key width 𝑑𝑘 ∈ 2ℕ. For any 𝛿 ∈ (0,1) there exists a parameter choice with one head and this 𝑑𝑘 such that for all 𝑡,

| |
|---|

𝛼fwd𝑡,𝑡 ≥ 1 − 𝛿, ∑ 𝑗≤𝑡 𝑗≠𝑡

𝛼fwd𝑡,𝑗 ≤ 𝛿.

At the architectural scale 𝜎𝑘 = 𝑑𝑘−1/2, it suﬀices to scale the active query/key pair by a common factor 𝑐diag > 0 such that

𝑇 − 1 𝛿

𝜎𝑘 𝑐diag2 Δ𝑇 ≥ log

, Δ𝑇 ∶= 1 − max

cos(𝑠) > 0.

𝑠∈{1,…,𝑇−1}

Proof. Under the RoPE convention above, RoPE𝑡 acts pairwise on consecutive 2-dimensional coordinates with frequencies (𝜔𝑟)𝑑

𝑘/2−1

𝜔0 = 1. Activate only the first 2-dimensional pair by choosing

𝑟=0 and

𝑞0 = (1,0,0,…,0) ∈ ℝ𝑑𝑘, 𝑘0 = (1,0,0,…,0) ∈ ℝ𝑑𝑘, and then setting

𝑞 = 𝑐diag𝑞0, 𝑘 = 𝑐diag𝑘0.

With RoPE, 𝑞̃𝑡 = RoPE𝑡(𝑞) and 𝑘̃𝑗 = RoPE𝑗(𝑘) satisfy

⟨𝑞̃𝑡,𝑘̃𝑗⟩ = 𝑐diag2 cos(𝑡 − 𝑗),

since all coordinate pairs except the first are identically zero, and the first pair rotates with frequency 𝜔0 = 1. For fixed 𝑡 and 𝑗 ≤ 𝑡, the unique maximum equals 𝑐diag2 at 𝑗 = 𝑡. For 𝑗 ≠ 𝑡, 𝑠 = 𝑡−𝑗 ∈ {1,…,𝑇 −1} so cos(𝑠) ≤ 1−Δ𝑇. Hence the logit gap is at least 𝑐diag2 Δ𝑇. Apply Corollary I.3.

- Lemma I.6 (Mixing error under diagonalization). Assume ‖𝑣𝑗‖2 ≤ 𝑉max. If 𝛼fwd𝑡,𝑡 ≥ 1 − 𝛿, then

∥∑

𝑗≤𝑡

𝛼fwd𝑡,𝑗 𝑣𝑗 − 𝑣𝑡∥

2

≤ 2𝛿𝑉max, ‖𝑓 − 𝑣‖𝐹 ≤ 2𝛿𝑉max

√

𝑇.

Proof. Lemma I.4 with 𝑗∗ = 𝑡, then sum over 𝑡.

| |
|---|

- Lemma I.7 (Lower-triangular inversion). For every input 𝑥, 𝐵fb(𝑥) ∈ ℝ𝑇×𝑇 is strictly lower-triangular. Hence 𝐵fb(𝑥) is nilpotent, with 𝐵fb(𝑥)𝑇 = 0.

| |
|---|

𝑇−1

∑

(𝐼 − 𝐵fb(𝑥))−1 =

𝐵fb(𝑥)𝑘.

𝑘=0

Proof. A strictly lower-triangular 𝑇 ×𝑇 matrix is nilpotent of index at most 𝑇. Hence 𝐵fb𝑇 = 0, and the Neumann series terminates after 𝑇 − 1 terms.

| |
|---|

#### I.5 Generating positional codes via feedback

Corollary I.8 (A Sessa block can generate separated positional codes). Fix any tokenwise pre-normalization map

Norm ∶ ℝ𝑚 → ℝ𝑚

(applied independently to each token), any even query/key width 𝑑𝑘 ≥ 2, and any model width 𝑚 ≥ 1. Then there exists a single width-𝑚 concrete Sessa block

𝐺pos ∈ ConcreteSessaBlocksNorm(𝑑𝑘,𝑚) and vectors 𝑝0,…,𝑝𝑇−1 ∈ ℝ𝑚 such that:

- (i) for all ℎ ∈ ℝ𝑇×𝑚 and all 𝑡, 𝐺pos(ℎ)𝑡 = ℎ𝑡 + 𝑝𝑡;
- (ii) for any prescribed unit vector 𝑢 ∈ ℝ𝑚, one may choose

𝑝𝑡 = (𝜆𝑐𝑡)𝑢

with pairwise distinct scalars (𝑐𝑡)𝑇−1𝑡=0 and some 𝜆 > 0, so that on any compact 𝒦_set ⊂ ℝ𝑇×𝑚 the scalar sets

ℐ𝑡 ∶= {⟨ℎ𝑡 + 𝑝𝑡,𝑢⟩ ∶ ℎ ∈ 𝒦_set} are pairwise disjoint after choosing 𝜆 large enough.

Proof. Fix a prescribed unit vector 𝑢 ∈ ℝ𝑚. The construction does not depend on Norm: setting 𝑊in = 0 gives

𝑢𝑡 = 𝑥̃𝑡𝑊in + 𝑏in = 𝑏in,

for all 𝑡. Choose 𝑊in = 0 and choose 𝑏in so that for every token

𝑎𝑡 ≡ 𝑎∗𝑒1, 𝑔𝑡 ≡ 𝑒1, for some 𝑎∗ > 0. Set

𝐴 ∶= GELU(𝑎∗) > 0. Then

𝑎̄𝑡 = 𝐴𝑒1 ∀𝑡. Choose

𝑊𝑄𝑓 = 0, 𝑊𝐾𝑓 = 0. Then all forward logits vanish, so each forward row is a causal probability vector. Choose 𝑊𝑉 so that

𝑣𝑡 = 𝑒1 ∀𝑡. Therefore

𝑓𝑡 = ∑ 𝑗≤𝑡

𝛼fwd𝑡,𝑗 𝑣𝑗 = 𝑒1 ∀𝑡.

𝑊𝑄𝑏 = 0, 𝑊𝐾𝑏 = 0. Then for 𝑡 ≥ 1,

Choose

1 𝑡

1[𝑗 < 𝑡], 𝛼fb0,⋅ = 0.

𝛼fb𝑡,𝑗 =

Fix any constant 𝛾 ∈ (0,1), and choose

𝑤𝛾 = 0, 𝑏𝛾 = arctanh(𝛾). Then

⎧ ⎨{⎩

- 0, 𝑡 = 0, 𝛾 𝑡

- 1[𝑗 < 𝑡], 𝑡 ≥ 1.

𝛾𝑡 ≡ 𝛾, [𝐵fb]𝑡,𝑗 =

Since 𝑓𝑡 = 𝑒1, we have

𝑠𝑡 = 𝑐𝑡𝑒1, where

𝛾 𝑡

𝑡−1

∑

𝑐𝑗 (𝑡 ≥ 1).

𝑐0 = 1, 𝑐𝑡 = 1 +

𝑗=0

𝑆𝑡 𝑡 + 1

𝑡

𝑆𝑡 ∶=

∑

𝑐𝑗, 𝜇𝑡 ∶=

.

Let

𝑗=0

𝛾 𝑡

)𝑆𝑡−1 + 1, hence

𝑆𝑡 = (1 +

Then

𝑡 + 𝛾 𝑡 + 1

1 𝑡 + 1

1 − (1 − 𝛾)𝜇𝑡−1 𝑡 + 1

𝜇𝑡 =

𝜇𝑡−1 +

, 𝜇𝑡 − 𝜇𝑡−1 =

.

Since 𝜇0 = 1 < 1−𝛾1 , an induction gives

1 1 − 𝛾

∀𝑡, so

𝜇𝑡 <

𝜇𝑡 − 𝜇𝑡−1 > 0 ∀𝑡 ≥ 1. Now

𝑐1 = 1 + 𝛾 > 1 = 𝑐0, and for 𝑡 ≥ 1,

𝑆𝑡 𝑡 + 1

𝑆𝑡−1 𝑡

𝑐𝑡+1 − 𝑐𝑡 = 𝛾(

−

) = 𝛾(𝜇𝑡 − 𝜇𝑡−1) > 0.

Therefore (𝑐𝑡) is strictly increasing. Choose 𝑊out so that its first row is 𝜆𝑢⊤ and all other rows are zero, and set 𝑏out = 0. Since

𝑠𝑡 ⊙ 𝑔𝑡 = (𝑐𝑡𝑒1) ⊙ 𝑒1 = 𝑐𝑡𝑒1, the residual update equals

(𝑠𝑡 ⊙ 𝑔𝑡)𝑊out = 𝑐𝑡(𝜆𝑢) =∶ 𝑝𝑡. Hence

𝐺pos(ℎ)𝑡 = ℎ𝑡 + 𝑝𝑡. Let 𝒦_set ⊂ ℝ𝑇×𝑚 be compact and set

𝑅 ∶= sup

‖ℎ𝑡‖2 < ∞.

𝑡

ℎ∈𝒦_set

max

|⟨ℎ𝑡,𝑢⟩| ≤ 𝑅 ∀ℎ ∈ 𝒦_set, ∀𝑡. Since the 𝑐𝑡 are pairwise distinct, let

Then

|𝑐𝑠 − 𝑐𝑡| > 0. Choose

Δ𝑐 ∶= min

𝑠≠𝑡

2𝑅 Δ𝑐

. Then the shifted scalar sets

𝜆 >

ℐ𝑡 = {⟨ℎ𝑡 + 𝑝𝑡,𝑢⟩ ∶ ℎ ∈ 𝒦_set} = {⟨ℎ𝑡,𝑢⟩ + 𝜆𝑐𝑡 ∶ ℎ ∈ 𝒦_set} are pairwise disjoint.

| |
|---|

- Lemma I.9 (Composition error on thickened compacts). Let (𝑋,𝑑) be a metric space such that closed neighborhoods of compact sets are compact, for example, 𝑋 = ℝ𝑛 with the Euclidean metric. Fix a compact 𝒦_set1 ⊂ 𝑋 and continuous maps 𝑓𝑖 ∶ 𝑋 → 𝑋 for 𝑖 = 1,…,𝐿. Fix 𝜌nbhd > 0 and define recursively

- I.6 Composition error control

𝒦̃_set1 ∶= 𝒦_set1, 𝒦_set𝑖+1 ∶= 𝑓𝑖(𝒦̃_set𝑖), 𝒦̃_set𝑖+1 ∶= 𝒩𝜌

(𝒦_set𝑖+1) = {𝑥 ∈ 𝑋 ∶ 𝑑(𝑥,𝒦_set𝑖+1) ≤ 𝜌nbhd}.

Then each 𝒦̃_set𝑖 is compact.

nbhd

For every 𝜀 > 0 there exist tolerances 𝛿1,…,𝛿𝐿 > 0 such that: for any continuous maps 𝑔𝑖 ∶ 𝒦̃_set𝑖 → 𝑋 satisfying, for each 𝑖,

𝑑(𝑓𝑖(𝑥),𝑔𝑖(𝑥)) ≤ 𝛿𝑖 and 𝛿𝑖 ≤ 𝜌nbhd,

𝑥∈𝒦̃_set𝑖

sup

the compositions 𝐹 ∶= 𝑓𝐿 ∘ ⋯ ∘ 𝑓1 and 𝐺 ∶= 𝑔𝐿 ∘ ⋯ ∘ 𝑔1 are well-defined on 𝒦_set1 (and in fact 𝑔𝑖(𝒦̃_set𝑖) ⊂ 𝒦̃_set𝑖+1), and

𝑑(𝐹(𝑥),𝐺(𝑥)) ≤ 𝜀.

𝑥∈𝒦_set1

sup

Proof. Well-definedness is immediate. Fix 𝑖 and 𝑥 ∈ 𝒦̃_set𝑖. By definition, 𝑓𝑖(𝑥) ∈ 𝒦_set𝑖+1 = 𝑓𝑖(𝒦̃_set𝑖), hence 𝑑(𝑓𝑖(𝑥),𝒦_set𝑖+1) = 0. Therefore

𝑑(𝑔𝑖(𝑥),𝒦_set𝑖+1) ≤ 𝑑(𝑔𝑖(𝑥),𝑓𝑖(𝑥)) + 𝑑(𝑓𝑖(𝑥),𝒦_set𝑖+1) ≤ 𝛿𝑖 ≤ 𝜌nbhd,

so 𝑔𝑖(𝑥) ∈ 𝒦̃_set𝑖+1. Thus 𝑔𝑖(𝒦̃_set𝑖) ⊂ 𝒦̃_set𝑖+1 and all compositions are defined. The remainder of the proof is by induction on 𝐿. For 𝐿 = 1 it is immediate. Assume the claim holds for 𝐿 − 1. Let

𝐹<𝐿 ∶= 𝑓𝐿−1 ∘ ⋯ ∘ 𝑓1, 𝐺<𝐿 ∶= 𝑔𝐿−1 ∘ ⋯ ∘ 𝑔1.

Since 𝒦̃_set𝐿 is compact and 𝑓𝐿 is continuous, 𝑓𝐿 is uniformly continuous on 𝒦̃_set𝐿. Pick 𝜂 > 0 such that 𝑑(𝑢,𝑣) ≤ 𝜂 ⇒ 𝑑(𝑓𝐿(𝑢),𝑓𝐿(𝑣)) ≤ 𝜀/2 ∀𝑢,𝑣 ∈ 𝒦̃_set𝐿.

Set 𝛿𝐿 ∶= min(𝜌nbhd,𝜀/2). By the inductive hypothesis applied with target accuracy 𝜂, choose 𝛿1,…,𝛿𝐿−1 > 0 so that

𝑑(𝐹<𝐿(𝑥),𝐺<𝐿(𝑥)) ≤ 𝜂.

𝑥∈𝒦_set1

sup

Then for 𝑥 ∈ 𝒦_set1, noting that 𝐺<𝐿(𝑥) ∈ 𝒦̃_set𝐿 by well-definedness, 𝑑(𝐹(𝑥),𝐺(𝑥)) ≤ 𝑑(𝑓𝐿(𝐹<𝐿(𝑥)),𝑓𝐿(𝐺<𝐿(𝑥))) + 𝑑(𝑓𝐿(𝐺<𝐿(𝑥)),𝑔𝐿(𝐺<𝐿(𝑥))) ≤ 𝜀/2 + 𝛿𝐿 ≤ 𝜀.

- Lemma I.10 (Tokenwise GELU approximation). Let 𝑆 ⊂ ℝ𝑚 be compact and let Θ ∶ 𝑆 → ℝ𝑝 be continuous. Then for every 𝜂 > 0 there exist 𝑟 ∈ ℕ∗ and aﬀine maps

| |
|---|

𝐴 ∶ ℝ𝑚 → ℝ𝑟, 𝐵 ∶ ℝ𝑟 → ℝ𝑝 such that

‖𝐵(GELU(𝐴(𝑧))) − Θ(𝑧)‖2 ≤ 𝜂.

𝑧∈𝑆

sup

Moreover, if a larger width 𝑟′ ≥ 𝑟 is prescribed in advance, the same conclusion still holds with 𝑟′ in place of 𝑟, by padding the hidden layer with unused coordinates.

Proof. Apply the standard one-hidden-layer universal approximation theorem for non-polynomial activations coordinatewise to the components of Θ, and concatenate the resulting hidden units into a single hidden layer. Since GELU is continuous and non-polynomial, the theorem applies; see, e.g., Hornik et al. (1989); Leshno et al. (1993). The padding claim is immediate by adding hidden coordinates with zero incoming and outgoing weights.

| |
|---|

- Lemma I.11 (Tokenwise GELU approximation with zero-padding). Let 𝑆 ⊂ ℝ𝑚 be compact, let Θ ∶ 𝑆 → ℝ𝑝0 be continuous, let 𝜂 > 0, and let 𝑟0 ∈ ℕ∗. For each 𝑟 ≥ 𝑟0, let

𝐸𝑟 ∶ ℝ𝑝0 ↪ ℝ𝑝(𝑟) be a coordinate zero-padding embedding, where 𝑝(𝑟) may depend on 𝑟. Then there exist 𝑟 ≥ 𝑟0 and aﬀine maps

𝐴 ∶ ℝ𝑚 → ℝ𝑟, 𝐵 ∶ ℝ𝑟 → ℝ𝑝(𝑟) such that

sup

𝑧∈𝑆

∥𝐵(GELU(𝐴(𝑧))) − 𝐸𝑟(Θ(𝑧))∥2 ≤ 𝜂.

Proof. By Lemma I.10, there exist 𝑠 ∈ ℕ∗ and aﬀine maps

𝐴̄ ∶ ℝ𝑚 → ℝ𝑠, 𝐵̄ ∶ ℝ𝑠 → ℝ𝑝0 such that

sup

𝑧∈𝑆

∥𝐵(̄ GELU(𝐴(𝑧)))̄ − Θ(𝑧)∥2 ≤ 𝜂. Set

𝑟 ∶= max{𝑟0,𝑠}. Let

𝐼𝑠→𝑟 ∶ ℝ𝑠 ↪ ℝ𝑟 be the coordinate zero-padding inclusion into the first 𝑠 coordinates, and let

Π𝑟→𝑠 ∶ ℝ𝑟 → ℝ𝑠 be the projection onto those first 𝑠 coordinates. Define

𝐴 ∶= 𝐼𝑠→𝑟 ∘ 𝐴,̄ 𝐵 ∶= 𝐸𝑟 ∘ 𝐵̄ ∘ Π𝑟→𝑠. Then 𝐴 is aﬀine and 𝐵 is aﬀine. Since GELU(0) = 0 and GELU acts coordinatewise,

Π𝑟→𝑠(GELU(𝐴(𝑧))) = Π𝑟→𝑠(GELU(𝐼𝑠→𝑟𝐴(𝑧)))̄ = GELU(𝐴(𝑧)).̄ Hence

𝐵(GELU(𝐴(𝑧))) = 𝐸𝑟(𝐵(̄ GELU(𝐴(𝑧)))).̄ Because 𝐸𝑟 is coordinate zero-padding, it is an isometric embedding for the Euclidean norm, so

∥𝐵(GELU(𝐴(𝑧))) − 𝐸𝑟(Θ(𝑧))∥2 = ∥𝐵(̄ GELU(𝐴(𝑧)))̄ − Θ(𝑧)∥2. Taking the supremum over 𝑧 ∈ 𝑆 gives the claim.

| |
|---|

I.7 Stability of finite-horizon RoPE attention

For fixed 𝑇, causal RoPE attention depends continuously on the query, key, and value arrays. The next two lemmas collect the continuity and near-diagonal transport estimates used below.

- Lemma I.12 (Stability of finite-horizon RoPE attention). Fix a horizon 𝑇 ≥ 1, number of heads 𝐻 ≥ 1, even key/query width 𝑑𝑘 ≥ 2, value width 𝑑𝑣 ≥ 1, attention scale 𝜎𝑘 > 0, and an output matrix

𝑊𝑂 ∈ ℝ𝐻𝑑𝑣×𝑚.

Let 𝒦_set ⊂ ℝ𝑇×𝑚 be compact, and define the compact token set

𝑆𝒦_set ∶= {𝑢𝑡 ∶ 𝑢 ∈ 𝒦_set, 0 ≤ 𝑡 ≤ 𝑇 − 1} ⊂ ℝ𝑚.

For each head 𝑎 = 1,…,𝐻, let

𝑞𝑎,𝑘𝑎,𝑞̂𝑎,𝑘̂𝑎 ∶ 𝑆𝒦_set → ℝ𝑑𝑘, 𝑣𝑎,𝑣̂𝑎 ∶ 𝑆𝒦_set → ℝ𝑑𝑣 be continuous. Let 𝐴,𝐴̂∶ 𝒦_set → ℝ𝑇×𝑚 be the corresponding causal RoPE-attention maps: for 𝑢 ∈ 𝒦_set,

𝐴(𝑢)𝑡 = (concat𝐻𝑎=1 𝑧𝑡𝑎(𝑢))𝑊𝑂, 𝑧𝑡𝑎(𝑢) ∶= ∑ 𝑗≤𝑡

𝛼𝑎𝑡,𝑗(𝑢)𝑣𝑎(𝑢𝑗),

exp(𝜎𝑘⟨RoPE𝑡(𝑞𝑎(𝑢𝑡)), RoPE𝑗(𝑘𝑎(𝑢𝑗))⟩)1[𝑗 ≤ 𝑡] ∑𝜏≤𝑡 exp(𝜎𝑘⟨RoPE𝑡(𝑞𝑎(𝑢𝑡)), RoPE𝜏(𝑘𝑎(𝑢𝜏))⟩)

𝛼𝑎𝑡,𝑗(𝑢) =

,

where

and similarly 𝐴̂ is defined from (𝑞̂𝑎,𝑘̂𝑎,𝑣̂𝑎). Then for every 𝜀 > 0 there exists 𝜂 > 0 such that

(‖𝑞𝑎(𝑧) − 𝑞̂𝑎(𝑧)‖2 + ‖𝑘𝑎(𝑧) − 𝑘̂𝑎(𝑧)‖2 + ‖𝑣𝑎(𝑧) − 𝑣̂𝑎(𝑧)‖2) ≤ 𝜂

1≤𝑎≤𝐻

𝑧∈𝑆𝒦_set

sup

max

‖𝐴(𝑢) − 𝐴(𝑢)‖̂ 𝐹 ≤ 𝜀.

implies

𝑢∈𝒦_set

sup

Proof. Define the finite-dimensional array space

𝒳 ∶= ((ℝ𝑑𝑘)𝐻)

###### × ((ℝ𝑑𝑘)𝐻)

###### × ((ℝ𝑑𝑣)𝐻)

,

𝑇

𝑇

𝑇

and equip it with the max norm

‖𝑣𝑡𝑎‖2}.

‖𝑘𝑡𝑎‖2, max

‖𝑞𝑡𝑎‖2, max

‖(𝑄,𝐾,𝑉 )‖max ∶= max{max

𝑡,𝑎

𝑡,𝑎

𝑡,𝑎

𝒜 ∶ 𝒳 → ℝ𝑇×𝑚

Let

denote the finite-horizon causal RoPE-attention operator defined by the displayed formulas above. RoPE attention is continuous as a composition of continuous finite-dimensional operations.

Ξ,Ξ̂ ∶ 𝒦_set → 𝒳 by collecting the tokenwise arrays:

Now define continuous maps

Ξ(𝑢) ∶= ((𝑞𝑎(𝑢𝑡))𝑡,𝑎, (𝑘𝑎(𝑢𝑡))𝑡,𝑎, (𝑣𝑎(𝑢𝑡))𝑡,𝑎),

Ξ(𝑢)̂ ∶= ((𝑞̂𝑎(𝑢𝑡))𝑡,𝑎, (𝑘̂𝑎(𝑢𝑡))𝑡,𝑎, (𝑣̂𝑎(𝑢𝑡))𝑡,𝑎). Then

𝐴 = 𝒜 ∘ Ξ, 𝐴̂= 𝒜 ∘ Ξ.̂

The image Ξ(𝒦_set) ⊂ 𝒳 is compact. Fix 𝜂0 > 0; then its closed 𝜂0-neighborhood 𝒩𝜂

(Ξ(𝒦_set))

0

is compact as well. Hence 𝒜 is uniformly continuous on this neighborhood. Therefore, for the given 𝜀 > 0, there exists 𝛿 > 0 such that

𝑥,𝑥′ ∈ 𝒩𝜂

(Ξ(𝒦_set)), ‖𝑥 − 𝑥′‖max ≤ 𝛿 ⟹ ‖𝒜(𝑥) − 𝒜(𝑥′)‖𝐹 ≤ 𝜀.

0

Set 𝜂 ∶= min{𝜂0,𝛿}. If the stated tokenwise bound holds, then for every 𝑢 ∈ 𝒦_set, ‖Ξ(𝑢) − Ξ(𝑢)‖̂ max ≤ 𝜂, because each of the three summands is individually bounded by 𝜂. In particular, Ξ(𝑢)̂ ∈ 𝒩𝜂

(Ξ(𝒦_set)).

0

Applying the uniform continuity estimate to Ξ(𝑢) and Ξ(𝑢)̂ gives

‖𝐴(𝑢) − 𝐴(𝑢)‖̂ 𝐹 = ‖𝒜(Ξ(𝑢)) − 𝒜(Ξ(𝑢))‖̂ 𝐹 ≤ 𝜀 ∀𝑢 ∈ 𝒦_set. Taking the supremum over 𝑢 ∈ 𝒦_set proves the claim.

- Lemma I.13 (Near-diagonal attention transports values). Fix a horizon 𝑇 ≥ 1, an output width 𝑠 ≥ 1, and a compact set

| |
|---|

𝒦_set′ ⊂ ℝ𝑇×𝑚. Let

𝑆𝒦_set′ ∶= {𝑢𝑡 ∶ 𝑢 ∈ 𝒦_set′, 0 ≤ 𝑡 ≤ 𝑇 − 1} ⊂ ℝ𝑚. Let 𝜙,𝑣 ∶ 𝑆𝒦_set′ → ℝ𝑠 be continuous, and define

𝑀𝜙 ∶= sup

‖𝜙(𝑧)‖2 < ∞.

𝑧∈𝑆𝒦_set′

Suppose a one-head causal attention mechanism on 𝒦_set′ produces weights 𝛼𝑡,𝑗(𝑢) and outputs

𝑓𝑡(𝑢) ∶= ∑ 𝑗≤𝑡

𝛼𝑡,𝑗(𝑢)𝑣(𝑢𝑗), 𝑢 ∈ 𝒦_set′.

Assume that for some 𝛿 ∈ (0,1) and 𝜂 ≥ 0,

𝛼𝑡,𝑡(𝑢) ≥ 1 − 𝛿 ∀𝑢 ∈ 𝒦_set′, ∀𝑡 ∈ {0,…,𝑇 − 1}, and

‖𝑣(𝑧) − 𝜙(𝑧)‖2 ≤ 𝜂.

𝑧∈𝑆𝒦_set′

sup

‖𝑓𝑡(𝑢) − 𝜙(𝑢𝑡)‖2 ≤ 2𝛿(𝑀𝜙 + 𝜂) + 𝜂.

Then

0≤𝑡≤𝑇−1

𝑢∈𝒦_set′

sup

max

Proof. Fix 𝑢 ∈ 𝒦_set′ and 𝑡 ∈ {0,…,𝑇 − 1}. Set

𝑤𝑗 ∶= 𝑣(𝑢𝑗) ∈ ℝ𝑠, 0 ≤ 𝑗 ≤ 𝑡.

Then (𝛼𝑡,𝑗(𝑢))𝑗≤𝑡 is a convex distribution and

𝑓𝑡(𝑢) = ∑ 𝑗≤𝑡

𝛼𝑡,𝑗(𝑢)𝑤𝑗.

‖𝑤𝑗‖2 ≤ ‖𝜙(𝑢𝑗)‖2 + ‖𝑣(𝑢𝑗) − 𝜙(𝑢𝑗)‖2 ≤ 𝑀𝜙 + 𝜂 ∀𝑗 ≤ 𝑡.

Moreover,

Since 𝛼𝑡,𝑡(𝑢) ≥ 1 − 𝛿, Lemma I.4 yields

‖𝑓𝑡(𝑢) − 𝑤𝑡‖2 ≤ 2𝛿(𝑀𝜙 + 𝜂). Also,

‖𝑤𝑡 − 𝜙(𝑢𝑡)‖2 ≤ 𝜂. Hence

‖𝑓𝑡(𝑢) − 𝜙(𝑢𝑡)‖2 ≤ ‖𝑓𝑡(𝑢) − 𝑤𝑡‖2 + ‖𝑤𝑡 − 𝜙(𝑢𝑡)‖2 ≤ 2𝛿(𝑀𝜙 + 𝜂) + 𝜂. Since this bound is uniform in 𝑢 and 𝑡, the claim follows.

| |
|---|

#### I.8 Universal approximation for causal RoPE-Transformers with adapters

𝒟 ⊂ ℝ𝑇×𝑑

- Lemma I.14 (Universality of causal RoPE-Transformers with adapters). Let

ext

𝐹 ∶ 𝒟 → ℝ𝑇×𝑑

be compact and let

be continuous and causal. Then for any 𝜀 > 0 there exist finite (𝐻,𝑑𝑘,𝑟,𝑚) and 𝑔 ∈ Ω𝐻,𝑑

ext

RoPETr,cau(𝑑ext → 𝑚 → 𝑑ext) such that

𝑘,𝑟

‖𝐹(𝑥) − 𝑔(𝑥)‖𝐹 < 𝜀.

𝑥∈𝒟

sup

Moreover, the construction in the proof allows an arbitrary choice of distinct scalars (𝑐𝑡)𝑇−1𝑡=0 in Paragraph 3, hence an arbitrary absolute embedding 𝐸 supported on the pos-scalar coordinate of slice ℎ = 1 with distinct entries.

Proof. Fix 𝜀 > 0.

- 0. Causal factorization For each 𝑡 ∈ {0,…,𝑇 − 1}, define the compact set of attainable prefixes

)𝑡+1. By Lemma I.1, there exists a unique continuous map

𝒫pref𝑡 ∶= {(𝑥0,…,𝑥𝑡) ∶ 𝑥 ∈ 𝒟} ⊂ (ℝ𝑑

ext

𝐹̂𝑡 ∶ 𝒫pref𝑡 → ℝ𝑑

, 𝐹̂𝑡(𝑥0,…,𝑥𝑡) ∶= 𝐹(𝑥)𝑡 (𝑥 ∈ 𝒟). Since 𝒫pref𝑡 is compact in Euclidean space, it is closed in (ℝ𝑑

)𝑡+1. By Tietze extension applied coordinatewise (Tietze, 1915), extend 𝐹̂𝑡 to a continuous map

ext

ext

𝐹𝑡 ∶ (ℝ𝑑

)𝑡+1 → ℝ𝑑

such that 𝐹(𝑥)𝑡 = 𝐹𝑡(𝑥0,…,𝑥𝑡) for all 𝑥 ∈ 𝒟. Let 𝑀𝒟 ∶= sup𝑥∈𝒟 ‖𝑥‖𝐹.

ext

ext

- 1. Model width Set the number of heads to be

𝐻 ∶= 𝑇 + 1, 𝑑𝑘 ∶= 2, and choose the per-head value width

𝑑𝑣 ∶= 𝑑ext + 2. Define

𝑚 ∶= 𝐻 𝑑𝑣 = (𝑇 + 1)(𝑑ext + 2). We index coordinates of ℝ𝑚 by head-slices:

ℝ𝑚 ≅

𝐻

⨁

ℎ=1

ℝ𝑑𝑣,

and within each slice ℝ𝑑𝑣 we separate content coordinates, the first 𝑑ext coordinates, a constant coordinate with index 𝑑ext + 1, and a pos-scalar coordinate with index 𝑑ext + 2.

- 2. Adapters We now fix concrete adapters Embed,Unembed of the form introduced in Paragraph I.2. This

choice satisfies Unembed ∘ Embed = Id on ℝ𝑇×𝑑

ext. Define the sequence-level aﬀine adapter Embed ∶ ℝ𝑇×𝑑

ext

→ ℝ𝑇×𝑚

tokenwise by placing 𝑥𝑡 into the content coordinates of slice ℎ = 1, setting the constant coordinate to 1, and all other coordinates to 0:

Embed(𝑥)𝑡 = ((𝑥𝑡, 1, 0) ; 0 ; 0 ; ⋯ ; 0) ∈

𝐻

⨁

ℎ=1

ℝ𝑑

ext+2.

This is an aﬀine map 𝑥𝑡 ↦ 𝑥𝑡𝑊emb + 𝑏emb for suitable 𝑊emb and 𝑏emb. Define Unembed ∶ ℝ𝑇×𝑚 → ℝ𝑇×𝑑

ext tokenwise by reading out the content coordinates of slice ℎ = 1:

Unembed(ℎ)𝑡 ∶= (ℎ(ℎ=1)𝑡 )1∶𝑑

ext

∈ ℝ𝑑

ext

,

which is exactly a coordinate projection (equivalently, an aﬀine map with 𝑏un = 0) and satisfies Unembed∘Embed = Id on ℝ𝑇×𝑑

ext. Thus Unembed is linear and non-expansive in Frobenius norm:

‖Unembed(𝑈) − Unembed(𝑈′)‖𝐹 ≤ ‖𝑈 − 𝑈′‖𝐹 ∀𝑈,𝑈′ ∈ ℝ𝑇×𝑚.

Let 𝑥̄ ∶= Embed(𝑥) ∈ ℝ𝑇×𝑚. The set 𝒟̄ ∶= Embed(𝒟) is compact.

- 3. Absolute positional code Choose distinct scalars 𝑐0,…,𝑐𝑇−1 ∈ ℝ and define 𝐸 ∈ ℝ𝑇×𝑚 by: 𝐸𝑡 is zero in all coordinates except the pos-scalar coordinate of slice ℎ = 1, where it equals 𝑐𝑡.

Thus for all 𝑥 ∈ 𝒟 and all 𝑡,

(𝑥̄𝑡 + 𝐸𝑡)(ℎ=1)𝑑

ext+2 = 𝑐𝑡, i.e. the pos-scalar is exactly 𝑐𝑡, independent of 𝑥.

- 4. Prefix encoding Fix a diagonalization tolerance 𝛿 ∈ (0,1), to be chosen suﬀiciently small later. Under the standing RoPE convention fixed above, when 𝑑𝑘 = 2 there is only one rotary pair and 𝜔0 = 1, so

RoPE𝑡(𝑧) = 𝑅𝑡𝑧

with 𝑅𝑡 the planar rotation by angle 𝑡 radians (Su et al., 2021). Construct a single causal RoPE-attention sublayer whose output at time 𝑡 stores

𝑥𝑡, 𝑥𝑡−1, …, 𝑥0

in the content coordinates of slices ℎ = 2,3,…,𝑡 + 2, respectively. Equivalently, lag ℓ = 0,…,𝑡 is stored in slice ℎ = ℓ + 2, and all active slices ℎ = 2,…,𝐻 are controlled uniformly via the one-hot estimates below.

Because slice ℎ = 1 has a constant coordinate equal to 1, we may choose the linear maps 𝑊ℎ𝑄,𝑊ℎ𝐾 so that for every token representation 𝑢:

ext+1 𝑘̄ = 𝑘̄ ∈ ℝ2,

𝑞𝑡(ℎ) = (𝑢(ℎ=1)𝑡 )𝑑

ext+1 𝑞̄(ℎ) = 𝑞̄(ℎ) ∈ ℝ2, 𝑘𝑗(ℎ) = (𝑢(ℎ=1)𝑗 )𝑑

for fixed vectors 𝑞̄(ℎ),𝑘̄ ∈ ℝ2. Fix a scaling factor 𝑐pack > 0. We set 𝑘̄ = 𝑐pack(1,0) and for head ℎ ∈ {2,…,𝐻} set 𝑞̄(ℎ) ∶= 𝑐packRoPE−(ℎ−2)(1,0) ∈ ℝ2. Under RoPE inside logits, for 𝑗 ≤ 𝑡,

⟨RoPE𝑡(𝑞̄(ℎ)), RoPE𝑗(𝑘)⟩̄ = 𝑐pack2 cos((𝑡 − (ℎ − 2)) − 𝑗). Define for each (𝑡,ℎ) the maximizer

𝑗∗(𝑡,ℎ) ∈ arg max 0≤𝑗≤𝑡

cos((𝑡 − (ℎ − 2)) − 𝑗).

For ℎ ≤ 𝑡 + 2, the unique maximizer is 𝑗∗(𝑡,ℎ) = 𝑡 − (ℎ − 2), since the maximum value 1 is attained only at argument 0. For ℎ > 𝑡 + 2, all arguments (𝑡 − (ℎ − 2)) − 𝑗 are distinct negative integers, and the corresponding cosine values are pairwise distinct (since cos(𝑎) = cos(𝑏) implies 𝑎 = ±𝑏 + 2𝜋𝑘 for some 𝑘 ∈ ℤ, and for integers 𝑎,𝑏 this forces 𝑘 = 0 because 2𝜋 is irrational, hence 𝑎 = ±𝑏). Thus the maximizer is unique for every (𝑡,ℎ).

𝑣𝑡,ℎ(𝑗) ∶= cos((𝑡 − (ℎ − 2)) − 𝑗), 𝑗 ∈ {0,…,𝑡}, and for 𝑡 ≥ 1 define

Let

𝑣𝑡,ℎ(𝑗) > 0. Since the set of pairs (𝑡,ℎ) is finite, the uniform gap

Δ𝑡,ℎ ∶= 𝑣𝑡,ℎ(𝑗∗(𝑡,ℎ)) − max

𝑗∈{0,…,𝑡}∖{𝑗∗(𝑡,ℎ)}

Δ∗ ∶= min

Δ𝑡,ℎ

𝑡∈{1,…,𝑇−1} ℎ∈{2,…,𝐻}

is strictly positive. For 𝑡 = 0, the row is exactly one-hot. Choose 𝑐pack such that

𝑇 − 1 𝛿

𝜎𝑘𝑐pack2 Δ∗ ≥ log

. Then by Corollary I.3, for every 𝑥 ∈ 𝒟, every 𝑡 ≥ 1, and every head ℎ ∈ {2,…,𝐻},

𝛼fwd𝑡,𝑗∗,(ℎ)(𝑡,ℎ) ≥ 1 − 𝛿. For 𝑡 = 0 the distribution is exactly one-hot on 𝑗 = 0. For heads ℎ = 2,…,𝐻, choose 𝑊ℎ𝑉 so that the value vector copies the content coordinates of slice ℎ = 1 (and has zeros in the last two coordinates of the head output):

𝑣𝑗(ℎ) = (𝑥𝑗, 0, 0) ∈ ℝ𝑑

ext+2.

For head ℎ = 1, set 𝑊1𝑉 ≡ 0, so head 1 contributes 0. Let 𝑓𝑡 ∈ ℝ𝑚 denote the concatenation of head outputs. Choose 𝑊𝑂 = 𝐼𝑚. Since slices ℎ ≥ 2 are initially zero, the residual update

ℎ𝑡 ← ℎ𝑡 + 𝑓𝑡

injects the head outputs directly into these slices. Let 𝑉max ∶= sup𝑥∈𝒟 max𝑗 ‖𝑥𝑗‖2 ≤ 𝑀𝒟. For each 𝑡 and each head ℎ ∈ {2,…,𝐻}, by Lemma I.4,

∥(𝑓𝑡(ℎ))1∶𝑑

− 𝑥𝑗∗(𝑡,ℎ)∥

≤ 2𝛿 𝑉max ≤ 2𝛿 𝑀𝒟.

2

In particular, for ℎ ≤ 𝑡+2 we have 𝑗∗(𝑡,ℎ) = 𝑡−(ℎ−2), hence slices ℎ = 2,…,𝑡+2 recover (𝑥𝑡,𝑥𝑡−1,…,𝑥0) with per-slice content error at most 2𝛿 𝑉max.

ext

- 5. Ideal encoded state and target map Fix 𝐻 ∶= 𝑇 + 1 heads indexed by ℎ = 1,…,𝐻, with head ℎ = 1 unused as before. For each (𝑡,ℎ) with 𝑡 ∈ {0,…,𝑇 − 1} and ℎ ∈ {2,…,𝐻} define the deterministic index

𝑗∗(𝑡,ℎ) ∈ arg max 0≤𝑗≤𝑡

cos((𝑡 − (ℎ − 2)) − 𝑗).

With the same 𝑐pack chosen in Paragraph 4 so that

𝑇 − 1 𝛿

𝜎𝑘 𝑐pack2 Δ∗ ≥ log

,

Corollary I.3 gives, for every 𝑥 ∈ 𝒟, every 𝑡 ≥ 1, and every head ℎ ∈ {2,…,𝐻}, the causal attention distribution over 𝑗 ≤ 𝑡 satisfies

𝛼fwd𝑡,𝑗∗,(ℎ)(𝑡,ℎ) ≥ 1 − 𝛿.

For 𝑡 = 0 the attention is exactly one-hot. Define ℎ̂𝑡(𝑥) ∈ ℝ𝑚, where 𝑚 = (𝑇+1)(𝑑ext+2), by letting slice ℎ = 1 equal (𝑥𝑡,1,𝑐𝑡) in coordinates (1∶𝑑ext, 𝑑ext+1, 𝑑ext+2) and zero elsewhere, and for each slice ℎ = 2,…,𝐻 placing 𝑥𝑗∗(𝑡,ℎ) in the first 𝑑ext coordinates and zeros in the last two; and set

𝑆̂∶= {ℎ̂𝑡(𝑥) ∶ 𝑥 ∈ 𝒟, 𝑡 ∈ {0,…,𝑇 − 1}} ⊂ ℝ𝑚. Then 𝑆̂ is compact as a continuous image of a compact set. For each fixed 𝑡 ∈ {0,…,𝑇 − 1}, define the aﬀine map, in fact linear,

)𝑡+1 by reading the content coordinates of slices ℎ = 2,…,𝑡 + 2 in reverse order: Read𝑡(𝑢) ∶= ((𝑢(𝑡+2))1∶𝑑

Read𝑡 ∶ ℝ𝑚 → (ℝ𝑑

ext

).

, …, (𝑢(2))1∶𝑑

, (𝑢(𝑡+1))1∶𝑑

Equivalently, for ℓ = 0,…,𝑡,

ext

ext

ext

(Read𝑡(𝑢))ℓ = (𝑢(𝑡−ℓ+2))1∶𝑑

. By construction of the ideal encoded state and because 𝑗∗(𝑡,ℎ) = 𝑡 − (ℎ − 2) for ℎ ≤ 𝑡 + 2, Read𝑡(ℎ̂𝑡(𝑥)) = (𝑥0,…,𝑥𝑡) ∀𝑥 ∈ 𝒟.

ext

Thus the pos-scalar coordinate identifies 𝑡, while the encoded slices determine the prefix (𝑥0,…,𝑥𝑡). Decompose 𝑆̂ as the finite disjoint union 𝑆̂ = ⨆𝑇−1𝑡=0 𝑆̂𝑡 where 𝑆̂𝑡 ∶= {ℎ̂𝑡(𝑥) ∶ 𝑥 ∈ 𝒟}. Each 𝑆̂𝑡 is compact and

ext+2 = 𝑐𝑡}. Since the scalars 𝑐𝑡 are distinct, the sets 𝑆̂𝑡 are pairwise separated. Therefore Φ̂ is continuous on 𝑆̂once each restriction Φ|̂ 𝑆̂

contained in the aﬀine hyperplane {𝑢 ∈ ℝ𝑚 ∶ (𝑢(ℎ=1))𝑑

is continuous. Now fix 𝑡. For every 𝑢 = ℎ̂𝑡(𝑥) ∈ 𝑆̂𝑡, by the defining property of 𝐹𝑡 from Paragraph 0 and by the readout identity above,

𝑡

Φ(𝑢)̂ = 𝐹(𝑥)𝑡 = 𝐹𝑡(𝑥0,…,𝑥𝑡) = 𝐹𝑡(Read𝑡(𝑢)). Therefore

Φ|̂ 𝑆̂

= 𝐹𝑡 ∘ Read𝑡|𝑆̂

.

is continuous. Thus Φ̂ is continuous on 𝑆̂. By Tietze extension applied coordinatewise, extend Φ̂ to a continuous Φ̃ ∶ ℝ𝑚 → ℝ𝑑

ext is continuous, so Φ|̂ 𝑆̂

Read𝑡 is a linear map, and 𝐹𝑡 ∶ (ℝ𝑑

)𝑡+1 → ℝ𝑑

𝑡

𝑡

𝑡

ext

ext.

- 6. FFN approximation Let ℎenc𝑡 (𝑥) ∈ ℝ𝑚 denote the token state after the first RoPE-attention block, constructed in Paragraph 4, with 𝑊𝑂 = 𝐼𝑚, head ℎ = 1 set to zero, and the FFN set to zero. Slice ℎ = 1 is

unchanged by the residual, since the concatenated head output has zero slice ℎ = 1, so (ℎenc𝑡 (𝑥))(ℎ=1) = (𝑥𝑡,1,𝑐𝑡) exactly.

For each head slice ℎ ∈ {2,…,𝐻}, by the encoding construction in Paragraph 4 we have ‖𝑣𝑗(ℎ)‖2 ≤ 𝑉max and 𝛼fwd𝑡,𝑗∗,(ℎ)(𝑡,ℎ) ≥ 1 − 𝛿. Therefore Lemma I.4 gives, for each 𝑥 ∈ 𝒟, each 𝑡, each ℎ ∈ {2,…,𝐻},

∥(ℎenc𝑡 (𝑥))(ℎ)1∶𝑑

− 𝑥𝑗∗(𝑡,ℎ)∥

≤ 2𝛿𝑉max,

2

and the last two coordinates of each slice are exactly zero on both sides. Therefore, for each (𝑥,𝑡),

ext

√

∥ℎenc𝑡 (𝑥) − ℎ̂𝑡(𝑥)∥2 ≤ √ ⎷

𝐻

∑

(2𝛿𝑉max)2 = 2𝛿𝑉max

𝑇.

ℎ=2

√

∥ℎenc𝑡 (𝑥) − ℎ̂𝑡(𝑥)∥2 ≤ 2𝛿𝑉max

𝑇.

In particular,

𝑡

𝑥∈𝒟

sup

max

𝑆enc ∶= {ℎenc𝑡 (𝑥) ∶ 𝑥 ∈ 𝒟, 𝑡 = 0,…,𝑇 − 1} ⊂ ℝ𝑚 (compact). Since 𝑆̂ is compact, for every radius 𝑟nbhd > 0 the closed neighborhood 𝒩𝑟

Let

(𝑆)̂ ∶= {𝑢 ∈ ℝ𝑚 ∶ dist(𝑢,𝑆)̂ ≤ 𝑟nbhd}

is compact. Fix such an 𝑟nbhd > 0. By uniform continuity of Φ̃ on the compact set 𝒩𝑟

nbhd

(𝑆)̂ , there exists a continuity tolerance

𝛿UC > 0 such that

nbhd

√

(𝑆),̂ ‖𝑢 − 𝑣‖2 ≤ 𝛿UC ⟹ ‖Φ(𝑢)̃ − Φ(𝑣)‖̃ 2 ≤ 𝜀/(3

𝑢,𝑣 ∈ 𝒩𝑟

𝑇).

Now choose the diagonalization parameter 𝛿 ∈ (0,1) above small enough so that

nbhd

√

2𝛿𝑉max

𝑇 ≤ min{𝑟nbhd,𝛿UC}.

(𝑆)̂ , and for all 𝑥 ∈ 𝒟 and all 𝑡,

Then 𝑆enc ⊂ 𝒩𝑟

‖ℎenc𝑡 (𝑥) − ℎ̂𝑡(𝑥)‖2 ≤ 𝛿UC. Hence

nbhd

√

‖Φ(ℎ̃ enc𝑡 (𝑥)) − Φ(̃ ℎ̂𝑡(𝑥))‖2 ≤ 𝜀/(3

𝑇). Since Φ(̃ ℎ̂𝑡(𝑥)) = Φ(̂ ℎ̂𝑡(𝑥)) = 𝐹(𝑥)𝑡 by construction, it follows that

√

‖Φ(ℎ̃ enc𝑡 (𝑥)) − 𝐹(𝑥)𝑡‖2 ≤ 𝜀/(3

𝑇).

Define the continuous map Ψ ∶ 𝑆enc → ℝ𝑑

Ψ(𝑢) ∶= Φ(𝑢)̃ − (𝑢(ℎ=1))1∶𝑑

,

ext by

i.e. the increment needed (in slice ℎ = 1 content) to turn the current content into Φ(𝑢)̃ . By the universal approximation theorem for tokenwise GELU FFNs (Leshno et al., 1993; Hornik et al., 1989), there exists a tokenwise FFN (hidden width 𝑟 large enough) whose output FFN(ℎ)𝑡 ∈ ℝ𝑚 is supported only on slice ℎ = 1 content coordinates and satisfies

ext

√

∥(FFN(𝑢))(ℎ=1)1∶𝑑

𝑇),

− Ψ(𝑢)∥

≤ 𝜀/(3

2

𝑢∈𝑆enc

sup

and FFN(𝑢) equals 0 on all other coordinates. Applying this tokenwise, define the sequence-level FFN by FFN(ℎ)𝑡 ∶= FFN(ℎ𝑡). Using the residual connection in the second block (with its attention set to zero), the slice ℎ = 1 content becomes

ext

≈ Φ(ℎ̃ enc𝑡 (𝑥)) ≈ 𝐹(𝑥)𝑡. Combining the encoding and FFN errors yields for each 𝑡

+ (FFN(ℎenc𝑡 (𝑥)))(ℎ=1)1∶𝑑

(ℎenc𝑡 (𝑥))(ℎ=1)1∶𝑑

ext

ext

√

∥(ℎout𝑡 (𝑥))(ℎ=1)1∶𝑑

− 𝐹(𝑥)𝑡∥

≤ 𝜀/

𝑇,

2

hence ‖𝐹(𝑥) − 𝑔(𝑥)‖𝐹 ≤ 𝜀 uniformly on 𝒟 after applying Unembed.

ext

| |
|---|

#### I.9 Direct Sessa building blocks Storage decomposition Fix a model width

𝑚 = (𝑇 + 1)𝑑ext + 2. Write ℝ𝑚 as the orthogonal direct sum of coordinate subspaces

ℝ𝑚 = 𝑈0 ⊕ 𝑈1 ⊕ ⋯ ⊕ 𝑈𝑇−1 ⊕ 𝑈out ⊕ span{𝑒const,𝑒pos}, where each 𝑈ℓ is a coordinate copy of ℝ𝑑

ext and 𝑈out is a coordinate copy of ℝ𝑑

ext. Fix linear isometries

→ 𝑈out, and let

𝐽ℓ ∶ ℝ𝑑

→ 𝑈ℓ (ℓ = 0,…,𝑇 − 1), 𝐽out ∶ ℝ𝑑

ext

ext

𝑅ℓ ∶= 𝐽ℓ−1 ∶ 𝑈ℓ → ℝ𝑑

, 𝑅out ∶= 𝐽out−1 ∶ 𝑈out → ℝ𝑑

.

Let 𝜋ℓ ∶ ℝ𝑚 → 𝑈ℓ denote the projection onto 𝑈ℓ, let 𝜋out ∶ ℝ𝑚 → 𝑈out denote the projection onto 𝑈out, and

ext

ext

𝜋st ∶ ℝ𝑚 → 𝑈0 ⊕ ⋯ ⊕ 𝑈𝑇−1 ⊕ span{𝑒const,𝑒pos} denote the projection onto the storage slice. For each ℓ ∈ {1,…,𝑇 − 1}, let

let

𝑇0→ℓ ∶= 𝐽ℓ ∘ 𝑅0 ∶ 𝑈0 → 𝑈ℓ denote the fixed coordinate-copy isomorphism, and let

𝑇0→out ∶= 𝐽out ∘ 𝑅0 ∶ 𝑈0 → 𝑈out denote the corresponding copy map into the output slice. Let

𝜄st ∶ 𝜋st(ℝ𝑚) → ℝ𝑚 denote the linear lift obtained by restoring the output slice as the copy of 𝑈0, i.e. 𝜋st(𝜄st(𝑧)) = 𝑧, 𝜋out(𝜄st(𝑧)) = 𝑇0→out(𝜋0(𝑧)).

- Lemma I.15 (Uniform small-signal linearization of GELU). Let 𝐾 ⊂ ℝ𝑞 be compact. Then

sup

𝑢∈𝐾

∥

2 𝜀

GELU(𝜀𝑢) − 𝑢∥

2

⟶ 0 as 𝜀 ↓ 0.

Consequently, for every compact 𝐾 ⊂ ℝ𝑝, every linear map 𝐿 ∶ ℝ𝑝 → ℝ𝑞, and every 𝜂 > 0, there exists 𝜀 > 0 such that

sup

𝑧∈𝐾

∥

2 𝜀

GELU(𝜀𝐿𝑧) − 𝐿𝑧∥

2

≤ 𝜂.

Proof. GELU is 𝐶1 and GELU′(0) = 1/2. Hence

GELU(𝑢) =

1 2

𝑢 + 𝑟(𝑢),

‖𝑟(𝑢)‖2 ‖𝑢‖2

→ 0 as 𝑢 → 0.

Apply this uniformly on the compact set 𝜀𝐾. The second statement follows by substituting 𝑢 = 𝐿𝑧.

| |
|---|

- Lemma I.16 (A single Sessa block copies one lag into a dedicated slice). Fix ℓ ∈ {1,…,𝑇 − 1} and a compact set 𝒦_set ⊂ ℝ𝑇×𝑚. Define the compact source-token set

𝑆0 ∶= {𝜋0(ℎ𝑡) ∶ ℎ ∈ 𝒦_set, 0 ≤ 𝑡 ≤ 𝑇 − 1} ⊂ 𝑈0. Then for every 𝜂 > 0 there exists a width-𝑚 concrete Sessa block

𝐺lagℓ ∈ ConcreteSessaBlocksId(2,𝑚) such that:

- (i) feedback is turned off identically, i.e. 𝛾𝑡 ≡ 0;
- (ii) for every ℎ ∈ 𝒦_set and every 𝑡, the block can be chosen so that its input projection depends only on the source slice 𝑈0 (and fixed biases), i.e. it ignores all coordinates in 𝑈𝑟 for 𝑟 ≠ 0, as well as 𝑈out, 𝑒const, and 𝑒pos;

𝜋𝑟(𝐺lagℓ (ℎ)𝑡) = 𝜋𝑟(ℎ𝑡) for all 𝑟 ∈ {0,…,𝑇 − 1} ∖ {ℓ}, and the coordinates in 𝑈out, 𝑒const, and 𝑒pos are unchanged;

cos((𝑡 − ℓ) − 𝑗), then

𝑗∗(𝑡,ℓ) ∈ arg max 0≤𝑗≤𝑡

- (iii) if

∥𝜋ℓ(𝐺lagℓ (ℎ)𝑡) − 𝜋ℓ(ℎ𝑡) − 𝑇0→ℓ(𝜋0(ℎ𝑗∗(𝑡,ℓ)))∥

≤ 𝜂.

2

0≤𝑡≤𝑇−1

ℎ∈𝒦_set

sup

max

In particular, for 𝑡 ≥ ℓ one has 𝑗∗(𝑡,ℓ) = 𝑡 − ℓ.

Proof. Reserve one coordinate of 𝑎𝑡 for a constant bias so that the corresponding coordinate of 𝑎̄𝑡 is strictly positive. Fix a diagonalization tolerance 𝛿 ∈ (0,1), to be chosen suﬀiciently small later. Choose 𝑊𝑄𝑓,𝑊𝐾𝑓 so that only the designated constant coordinate of 𝑎̄𝑡 contributes to the forward queries and keys, and set

𝑞𝑡𝑓 ≡ 𝑞diag(ℓ) ∶= 𝑐ℓ RoPE−ℓ(1,0), 𝑘𝑡𝑓 ≡ 𝑘diag ∶= 𝑐ℓ(1,0) ∈ ℝ2, for some scale 𝑐ℓ > 0. Then for 𝑗 ≤ 𝑡,

⟨RoPE𝑡(𝑞𝑡𝑓), RoPE𝑗(𝑘𝑗𝑓)⟩ = 𝑐ℓ2 cos((𝑡 − ℓ) − 𝑗).

For each 𝑡, the maximizer of 𝑗 ↦ cos((𝑡−ℓ)−𝑗) on {0,…,𝑡} is unique; denote it by 𝑗∗(𝑡,ℓ). Uniqueness is proved as in Lemma I.5: for 𝑡 ≥ ℓ, the maximizer is 𝑗 = 𝑡−ℓ, while for 𝑡 < ℓ the arguments are distinct negative integers and therefore yield distinct cosine values. Hence, by the proof of Lemma I.5 together with Corollary I.3, after choosing 𝑐ℓ large enough we obtain

𝛼𝑡,𝑗∗(𝑡,ℓ) ≥ 1 − 𝛿 ∀𝑡 = 0,…,𝑇 − 1.

Use 𝑑ext further coordinates of 𝑎𝑡 to encode the source slice via 𝑎src𝑡 = 𝜀𝜋0(ℎ𝑡) ∈ 𝑈0. By Lemma I.15, after choosing 𝜀 > 0 small enough, these coordinates of 𝑎̄𝑡 = GELU(𝑎𝑡)

can be linearly mapped by 𝑊𝑉 to approximate 𝑇0→ℓ(𝜋0(ℎ𝑡)) uniformly on the compact source-token set 𝑆0. Choose 𝑊𝑉 so that the resulting value vector lives only in the destination slice 𝑈ℓ. Set 𝑔 ≡ 1, set 𝑊out to be the identity on 𝑈ℓ and zero on all other coordinates, and set 𝑏out = 0. Choose the feedback branch identically zero.

𝑆𝒦_set ∶= {ℎ𝑡 ∶ ℎ ∈ 𝒦_set, 0 ≤ 𝑡 ≤ 𝑇 − 1} ⊂ ℝ𝑚, and let

Define the compact token set

𝜙(𝑧) ∶= 𝑇0→ℓ(𝜋0(𝑧)), 𝑧 ∈ 𝑆𝒦_set. Set

𝑀ℓ ∶= sup

‖𝜙(𝑧)‖2 < ∞.

𝑧∈𝑆𝒦_set

Choose the small-signal approximation so that the induced value map 𝑣 ∶ 𝑆𝒦_set → 𝑈ℓ satisfies sup

‖𝑣(𝑧) − 𝜙(𝑧)‖2 ≤ 𝜂val.

𝑧∈𝑆𝒦_set

Then for every ℎ ∈ 𝒦_set and every 𝑡, Lemma I.4 applied to

𝑓𝑡(ℎ) = ∑ 𝑗≤𝑡

𝛼𝑡,𝑗 𝑣(ℎ𝑗)

with distinguished index 𝑗∗(𝑡,ℓ) gives

∥𝑓𝑡(ℎ) − 𝑣(ℎ𝑗∗(𝑡,ℓ))∥2 ≤ 2𝛿 (𝑀ℓ + 𝜂val). Therefore

∥𝑓𝑡(ℎ) − 𝜙(ℎ𝑗∗(𝑡,ℓ))∥2 ≤ 2𝛿 (𝑀ℓ + 𝜂val) + 𝜂val. Since

𝜙(ℎ𝑗∗(𝑡,ℓ)) = 𝑇0→ℓ(𝜋0(ℎ𝑗∗(𝑡,ℓ))),

choosing 𝛿 and 𝜂val suﬀiciently small makes the total error at most 𝜂. All remaining coordinates are unchanged by construction.

| |
|---|

𝐴 ∶ 𝜋st(ℝ𝑚) → ℝ𝑞 be aﬀine, with

- Lemma I.17 (A diagonal Sessa block realizes a block of tokenwise GELU units). Let

𝑞 ∈ {1,…,𝑚 − 1}, and let

𝐵 ∶ ℝ𝑞 → 𝑈out be linear. Fix a compact set

𝑆 ⊂ 𝜋st(ℝ𝑚). Then for every 𝜂 > 0 there exists a width-𝑚 concrete Sessa block

𝐺batch ∈ ConcreteSessaBlocksId(2,𝑚) such that:

- (i) feedback is turned off identically;
- (ii) the storage coordinates are preserved exactly:

𝜋st(𝐺batch(ℎ)𝑡) = 𝜋st(ℎ𝑡) ∀ℎ,∀𝑡;

- (iii) the input projection ignores the current output slice, i.e. it depends only on 𝜋st(ℎ𝑡);
- (iv) for every sequence ℎ whose tokenwise storage states lie in 𝑆,

∥𝜋out(𝐺batch(ℎ)𝑡) − 𝜋out(ℎ𝑡) − 𝐵(GELU(𝐴(𝜋st(ℎ𝑡))))∥2 ≤ 𝜂.

𝑡

sup

Proof. Let the first 𝑞 coordinates of 𝑎𝑡 encode the aﬀine preactivations 𝐴(𝜋st(ℎ𝑡)).

Reserve one additional coordinate of 𝑎𝑡 for a constant bias so that the corresponding coordinate of 𝑎̄𝑡 is strictly positive. Choose 𝑊𝑄𝑓,𝑊𝐾𝑓 so that only that coordinate contributes to the forward queries and keys, yielding constant queries and keys that make the forward attention arbitrarily close to diagonal uniformly in 𝑡 by

- Lemma I.5.

Choose 𝑊𝑉 so that the resulting value vector equals 𝐵(𝑎̄1∶𝑞) ∈ 𝑈out

in the output slice and is zero on the storage slice. Choose 𝑔 ≡ 1, choose 𝑊out to be the identity on 𝑈out and zero on the storage slice, set 𝑏out = 0, and set the columns of the input projection corresponding to the current

output slice 𝑈out to zero. Choose the feedback branch identically zero. Let

𝜙(𝑧) ∶= 𝐵(GELU(𝐴(𝑧))), 𝑧 ∈ 𝑆, and set

𝑀𝜙 ∶= sup 𝑧∈𝑆

‖𝜙(𝑧)‖2 < ∞.

Because the input projection ignores the current output slice, the preactivations 𝑎𝑡 depend only on 𝜋st(ℎ𝑡), hence for every sequence ℎ whose tokenwise storage states lie in 𝑆, the resulting value vector is exactly

𝑣𝑡 = 𝜙(𝜋st(ℎ𝑡)) ∈ 𝑈out.

By the diagonal forward-attention construction, after choosing the diagonalization tolerance 𝛿 ∈ (0,1) suﬀiciently small we have

𝛼𝑡,𝑡 ≥ 1 − 𝛿 ∀𝑡 = 0,…,𝑇 − 1. Therefore, for every such sequence ℎ and every 𝑡, Lemma I.4 applied to

𝑓𝑡 = ∑ 𝑗≤𝑡

𝛼𝑡,𝑗𝑣𝑗

with distinguished index 𝑗∗ = 𝑡 gives

‖𝑓𝑡 − 𝑣𝑡‖2 ≤ 2𝛿𝑀𝜙. Choosing 𝛿 so that 2𝛿𝑀𝜙 ≤ 𝜂 (trivial if 𝑀𝜙 = 0) yields

‖𝑓𝑡 − 𝜙(𝜋st(ℎ𝑡))‖2 ≤ 𝜂.

𝑡

sup

Since the residual update is added only in 𝑈out, this gives the desired conclusion. Corollary I.18 (Tokenwise GELU approximation by stacked Sessa blocks). Let

| |
|---|

𝑆 ⊂ 𝜋st(ℝ𝑚) be compact and let

Θ ∶ 𝑆 → 𝑈out be continuous. Then for every 𝜂 > 0 there exists a finite composition

𝐺tok = 𝐺batch𝑀 ∘ ⋯ ∘ 𝐺batch1 , 𝐺𝑏batch ∈ ConcreteSessaBlocksId(2,𝑚), such that:

- (i) every 𝐺batch𝑏 preserves the storage slice exactly and ignores the current output slice in its input projection;
- (ii) for every sequence ℎ whose tokenwise storage states lie in 𝑆,

𝜋st(𝐺tok(ℎ)𝑡) = 𝜋st(ℎ𝑡) ∀𝑡,

∥𝜋out(𝐺tok(ℎ)𝑡) − 𝜋out(ℎ𝑡) − Θ(𝜋st(ℎ𝑡))∥2 ≤ 𝜂.

and

𝑡

sup

Proof. By Lemma I.10, for every 𝜂′ > 0 there exist a width 𝑅 ∈ ℕ∗, an aﬀine map

𝐴tot ∶ 𝜋st(ℝ𝑚) → ℝ𝑅, and an aﬀine map

𝐵tot ∶ ℝ𝑅 → 𝑈out such that

‖𝐵tot(GELU(𝐴tot(𝑧))) − Θ(𝑧)‖2 ≤ 𝜂′.

𝑧∈𝑆

sup

𝐵tot(𝑢) = 𝐿tot𝑢 + 𝑏tot, where

Write

𝐿tot ∶ ℝ𝑅 → 𝑈out is linear and

𝑏tot ∈ 𝑈out. Partition the 𝑅 hidden units into batches of size at most 𝑚 − 1:

𝑅 = 𝑞1 + ⋯ + 𝑞𝑀, 1 ≤ 𝑞𝑏 ≤ 𝑚 − 1. Write accordingly

𝐴tot = (𝐴1,…,𝐴𝑀), with each

𝐴𝑏 ∶ 𝜋st(ℝ𝑚) → ℝ𝑞𝑏 aﬀine, and decompose the linear map 𝐿tot as

𝑀

𝐿tot(𝑢(1),…,𝑢(𝑀)) =

∑

𝐿𝑏𝑢(𝑏),

𝑏=1

𝐿𝑏 ∶ ℝ𝑞𝑏 → 𝑈out

where each

is linear. Choose 𝜂′ > 0 so that

𝜂′ ≤ 𝜂/2 and

‖𝐵tot(GELU(𝐴tot(𝑧))) − Θ(𝑧)‖2 ≤ 𝜂′.

𝑧∈𝑆

sup

Apply Lemma I.17 to each pair (𝐴𝑏,𝐿𝑏) with accuracy

𝜂 2(𝑀 + 1)

.

𝐺batch𝑏 ∈ ConcreteSessaBlocksId(2,𝑚), 𝑏 = 1,…,𝑀,

This yields concrete Sessa batch blocks

such that each block preserves the storage slice exactly, ignores the current output slice in its input projection, and contributes

𝐿𝑏(GELU(𝐴𝑏(⋅))) to the output slice up to error at most 𝜂/(2(𝑀 + 1)). It remains to represent the constant term 𝑏tot. Choose the scalar constant hidden map

𝐴const ∶ 𝜋st(ℝ𝑚) → ℝ, 𝐴const(𝑧) ≡ 1, and the linear map

𝜉 GELU(1)

𝑏tot. Then

𝐿const ∶ ℝ → 𝑈out, 𝐿const(𝜉) ∶=

𝐿const(GELU(𝐴const(𝑧))) = 𝑏tot ∀𝑧 ∈ 𝑆. Apply Lemma I.17 once more to (𝐴const,𝐿const), again with accuracy

𝜂 2(𝑀 + 1)

.

Since each batch block preserves storage exactly and ignores the current output slice in its input projection, all blocks act on the same storage input and their contributions add in 𝑈out. Hence the cumulative implementation error of the 𝑀 linear batches together with the one constant batch is at most

𝜂 2(𝑀 + 1)

𝜂 2

(𝑀 + 1) ⋅

=

.

Combining this with the approximation error 𝜂′ ≤ 𝜂/2 gives the total error bound 𝜂.

| |
|---|

Theorem (Universal approximation for Sessa with adapters). Let 𝒟 ⊂ ℝ𝑇×𝑑

- I.10 Sessa universality for causal maps

ext be compact and let 𝐹 ∶ 𝒟 → ℝ𝑇×𝑑

be continuous and causal. Then for any 𝜀 > 0 there exist a model width 𝑚 ∈ ℕ∗, an even key/query width 𝑑𝑘 (in fact 𝑑𝑘 = 2 suﬀices), tokenwise adapters

ext

, and a finite-depth network

Embed ∶ ℝ𝑑

→ ℝ𝑚, Unembed ∶ ℝ𝑚 → ℝ𝑑

ext

ext

Sessa,Id(𝑚) consisting only of the concrete Sessa blocks from Section 3, such that

𝐺 ∈ Ω𝑑

𝑘

∥𝐹(𝑥) − Unembed(𝐺(Embed(𝑥)))∥

< 𝜀.

𝐹

𝑥∈𝒟

sup

Proof of Theorem 14. Fix 𝜀 > 0.

- Step 0: causal factorization. For each 𝑡 ∈ {0,…,𝑇 − 1}, define the compact set of attainable prefixes

𝒫pref𝑡 ∶= {(𝑥0,…,𝑥𝑡) ∶ 𝑥 ∈ 𝒟} ⊂ (ℝ𝑑

)𝑡+1.

ext

, 𝐹̂𝑡(𝑥0,…,𝑥𝑡) ∶= 𝐹(𝑥)𝑡 (𝑥 ∈ 𝒟). Since 𝒫pref𝑡 is compact in Euclidean space, it is closed in (ℝ𝑑

𝐹̂𝑡 ∶ 𝒫pref𝑡 → ℝ𝑑

By Lemma I.1, there exists a unique continuous map

)𝑡+1. By Tietze extension applied coordinatewise, extend 𝐹̂𝑡 to a continuous map

ext

𝐹𝑡 ∶ (ℝ𝑑

)𝑡+1 → ℝ𝑑

ext

ext

ext

𝐹(𝑥)𝑡 = 𝐹𝑡(𝑥0,…,𝑥𝑡) ∀𝑥 ∈ 𝒟.

such that

- Step 1: width and adapters Set 𝑚 ∶= (𝑇 + 1)𝑑ext + 2.

Use the storage decomposition introduced above. Define the tokenwise embedding by

Embed(𝑥)𝑡 = 𝐽0(𝑥𝑡) + 𝐽out(𝑥𝑡) + 𝑒const,

that is, place 𝑥𝑡 in both 𝑈0 and 𝑈out, set the constant coordinate to 1, and set all other coordinates to 0. Define Unembed tokenwise by

Unembed(ℎ)𝑡 ∶= 𝑅out(𝜋out(ℎ𝑡)) ∈ ℝ𝑑

ext

. Then

Unembed(Embed(𝑥)) = 𝑥 ∀𝑥 ∈ ℝ𝑇×𝑑

ext

, Embed(𝒟) is compact, and Unembed is linear and non-expansive in Frobenius norm.

- Step 2: positional code Apply Corollary I.8 with 𝑢 = 𝑒pos to obtain a block 𝐺pos ∈ ConcreteSessaBlocksId(2,𝑚)

and pairwise distinct scalars 𝑐0,…,𝑐𝑇−1 such that

𝐺pos(ℎ)𝑡 = ℎ𝑡 + 𝑐𝑡𝑒pos ∀ℎ,∀𝑡. By construction, 𝐺pos leaves 𝑈0,…,𝑈𝑇−1 and 𝑈out unchanged.

- Step 3: prefix encoding Fix a packing tolerance

𝛿pack > 0,

to be specified later in Step 4. For each lag ℓ = 1,…,𝑇 − 1, apply Lemma I.16 successively on the compact set obtained after the previous blocks to construct a concrete Sessa block

𝐺lagℓ ∈ ConcreteSessaBlocksId(2,𝑚)

that preserves all coordinates except 𝑈ℓ and writes an approximation of the lag-ℓ token from 𝑈0 into 𝑈ℓ. For 𝑡 ∈ {0,…,𝑇 − 1} and ℓ ∈ {1,…,𝑇 − 1}, define

𝑗∗(𝑡,ℓ) ∈ arg max 0≤𝑗≤𝑡

cos((𝑡 − ℓ) − 𝑗).

For 𝑡 ≥ ℓ one has 𝑗∗(𝑡,ℓ) = 𝑡 − ℓ.

Define the ideal encoded state ℎ̂𝑡(𝑥) ∈ ℝ𝑚 by: 𝜋0(ℎ̂𝑡(𝑥)) = 𝐽0(𝑥𝑡), 𝜋ℓ(ℎ̂𝑡(𝑥)) = 𝐽ℓ(𝑥𝑗∗(𝑡,ℓ)) (1 ≤ ℓ ≤ 𝑇 − 1), 𝜋out(ℎ̂𝑡(𝑥)) = 𝐽out(𝑥𝑡), ⟨ℎ̂𝑡(𝑥),𝑒const⟩ = 1, ⟨ℎ̂𝑡(𝑥),𝑒pos⟩ = 𝑐𝑡.

Since each lag block depends only on the exact source slice 𝑈0 and fixed biases, while writing only to its own destination slice and preserving all previously written slices, the packing errors do not propagate to later lag

blocks. Hence, choosing per-lag accuracies 𝜂ℓ > 0 with

𝑇−1

∑

𝜂ℓ2 ≤ 𝛿pack2 ,

ℓ=1

𝐺pack ∶= 𝐺lag𝑇−1 ∘ ⋯ ∘ 𝐺lag1 ∘ 𝐺pos that

we obtain for

∥𝐺pack(Embed(𝑥))𝑡 − ℎ̂𝑡(𝑥)∥

≤ 𝛿pack.

2

0≤𝑡≤𝑇−1

𝑥∈𝒟

sup

max

##### Step 4: target map For each 𝑡, let

𝑆̂𝑡.

𝑆̂𝑡 ∶= {ℎ̂𝑡(𝑥) ∶ 𝑥 ∈ 𝒟} ⊂ ℝ𝑚, 𝑆̂∶=

𝑇−1

⋃

𝑡=0

Each 𝑆̂𝑡 is compact. Since the 𝑒pos-coordinate equals 𝑐𝑡 on 𝑆̂𝑡 and the scalars 𝑐𝑡 are distinct, the sets 𝑆̂𝑡 are pairwise disjoint and positively separated.

)𝑡+1 by

Read𝑡 ∶ ℝ𝑚 → (ℝ𝑑

Define the linear readout

ext

Read𝑡(𝑢) ∶= (𝑅𝑡𝜋𝑡(𝑢), 𝑅𝑡−1𝜋𝑡−1(𝑢), …, 𝑅0𝜋0(𝑢)). For 𝑢 = ℎ̂𝑡(𝑥), one has

𝑅0𝜋0(ℎ̂𝑡(𝑥)) = 𝑥𝑡, and for 1 ≤ ℓ ≤ 𝑡,

𝑅ℓ𝜋ℓ(ℎ̂𝑡(𝑥)) = 𝑥𝑗∗(𝑡,ℓ). Since 𝑗∗(𝑡,ℓ) = 𝑡 − ℓ for 1 ≤ ℓ ≤ 𝑡, it follows that

Read𝑡(ℎ̂𝑡(𝑥)) = (𝑥0,…,𝑥𝑡).

Φ̂ ∶ 𝑆̂→ 𝑈out by

Define

Φ(𝑢)̂ ∶= 𝐽out(𝐹𝑡(Read𝑡(𝑢))) for 𝑢 ∈ 𝑆̂𝑡. This is well defined because the index 𝑡 is uniquely determined by the 𝑒pos-coordinate of 𝑢, and if

𝑢 = ℎ̂𝑡(𝑥) = ℎ̂𝑡(𝑥′), then

Read𝑡(𝑢) = (𝑥0,…,𝑥𝑡) = (𝑥′0,…,𝑥′𝑡),

so the value of 𝐽out(𝐹𝑡(Read𝑡(𝑢))) does not depend on the choice of 𝑥. Moreover, on each 𝑆̂𝑡 one has

Φ|̂ 𝑆̂

= 𝐽out ∘ 𝐹𝑡 ∘ Read𝑡|𝑆̂

,

hence Φ̂ is continuous on each 𝑆̂𝑡, and therefore continuous on 𝑆̂. Apply Tietze extension coordinatewise to the ℝ𝑑

𝑡

𝑡

𝑅out ∘ Φ̂ ∶ 𝑆̂→ ℝ𝑑

. This yields a continuous extension

ext-valued map

ext

Φ̄ ∶ ℝ𝑚 → ℝ𝑑

of 𝑅out ∘ Φ̂. Set

ext

Φ̃ ∶= 𝐽out ∘ Φ̄ ∶ ℝ𝑚 → 𝑈out.

Then Φ̃ extends Φ̂. Fix 𝜌 > 0 and let

𝑁 ∶= 𝒩𝜌(𝑆)̂ ⊂ ℝ𝑚. Then 𝑁 is compact, so Φ̃ is uniformly continuous on 𝑁. Choose 𝛿UC > 0 such that 𝑢,𝑣 ∈ 𝑁, ‖𝑢 − 𝑣‖2 ≤ 𝛿UC ⟹ ‖Φ(𝑢)̃ − Φ(𝑣)‖̃ 2 ≤

𝜀 2

√

.

𝑇

Choose 𝛿pack > 0 small enough that

𝛿pack ≤ min{𝜌,𝛿UC} and that the encoding construction of Step 3 yields

∥𝐺pack(Embed(𝑥))𝑡 − ℎ̂𝑡(𝑥)∥

≤ 𝛿pack.

2

0≤𝑡≤𝑇−1

𝑥∈𝒟

sup

max

Then for every 𝑥 ∈ 𝒟 and every 𝑡 one has

𝐺pack(Embed(𝑥))𝑡 ∈ 𝑁, and

𝜀 2

∥Φ(𝐺̃ pack(Embed(𝑥))𝑡) − 𝐽out(𝐹(𝑥)𝑡)∥

√

.

≤

𝑇

2

𝑆st ∶= {𝜋st(𝐺pack(Embed(𝑥))𝑡) ∶ 𝑥 ∈ 𝒟, 0 ≤ 𝑡 ≤ 𝑇 − 1}. Define

- Step 5: tokenwise readout Define the compact storage-token set

Θ ∶ 𝑆st → 𝑈out, Θ(𝑧) ∶= Φ(𝜄̃ st(𝑧)) − 𝑇0→out(𝜋0(𝑧)). Since 𝜄st is linear and Φ̃ is continuous, Θ is continuous. Moreover, for every 𝑥 ∈ 𝒟 and every 𝑡, 𝜄st(𝜋st(𝐺pack(Embed(𝑥))𝑡)) = 𝐺pack(Embed(𝑥))𝑡, since Embed initializes the output slice as a copy of 𝑈0 and 𝐺pack preserves 𝑈out. Hence Θ(𝜋st(𝐺pack(Embed(𝑥))𝑡)) = Φ(𝐺̃ pack(Embed(𝑥))𝑡) − 𝜋out(𝐺pack(Embed(𝑥))𝑡),

so Θ is exactly the tokenwise increment that must be added in 𝑈out. Apply Corollary I.18 to 𝑆st and Θ. This yields a finite composition

𝐺tok = 𝐺batch𝑀 ∘ ⋯ ∘ 𝐺batch1 of concrete Sessa blocks such that every batch block preserves the storage coordinates exactly, every batch block ignores the current output slice in its input projection, and for all 𝑥 ∈ 𝒟 and all 𝑡,

𝜀 2

∥𝜋out(𝐺tok(𝐺pack(Embed(𝑥)))𝑡) − Φ(𝐺̃ pack(Embed(𝑥))𝑡)∥

√

≤

.

𝑇

2

- Step 6: conclusion Set 𝐺 ∶= 𝐺tok ∘ 𝐺pack ∈ Ω2Sessa,Id(𝑚).

Unembed(ℎ)𝑡 = 𝑅out(𝜋out(ℎ𝑡)), combining the two error bounds and using that 𝑅out is an isometry gives

Since

𝜀 √

‖Unembed(𝐺(Embed(𝑥)))𝑡 − 𝐹(𝑥)𝑡‖2 = ‖𝑅out(𝜋out(𝐺(Embed(𝑥))𝑡)) − 𝐹(𝑥)𝑡‖2 ≤

∀𝑥 ∈ 𝒟, ∀𝑡.

𝑇

∥Unembed(𝐺(Embed(𝑥))) − 𝐹(𝑥)∥

< 𝜀.

Hence

𝐹

𝑥∈𝒟

sup

| |
|---|

### J Universal approximation in the pre-norm LayerNorm setting

We now extend Theorem 14 from Norm = Id to the pre-norm LayerNorm case Norm = LN𝜀

with 𝜀ln > 0 (Xiong et al., 2020), after a width expansion via a fixed scaffold.

ln

- J.1 Tokenwise LayerNorm Fix a width 𝑚 ≥ 2 and 𝜀ln > 0. For 𝑧 ∈ ℝ𝑚, define

𝜇ln(𝑧) ∶=

1 𝑚

⟨𝑧,1⟩, 𝜎ln(𝑧) ∶= √

1 𝑚

‖𝑧 − 𝜇ln(𝑧)1‖2 + 𝜀ln, LN𝜀

ln

(𝑧) ∶=

𝑧 − 𝜇ln(𝑧)1 𝜎ln(𝑧)

.

With 𝜀ln > 0, LN𝜀

ln

is well-defined and continuous on all of ℝ𝑚, in particular, there is no singularity at nearlyconstant tokens.

- J.2 Zero-mean scaffold embedding

Fix a “dynamic” width 𝑚0 ≥ 1 and let 𝑚sc ≥ 2 be an even scaffold width. Let 𝑚 ∶= 𝑚0 + 𝑚sc and define, for 𝑐 > 0, the fixed zero-mean scaffold vector

𝑠𝑐,𝑚

∶= (𝑐,…,𝑐⏟

,⏟⏟⏟⏟⏟− 𝑐,…,−𝑐

) ∈ ℝ𝑚

, ⟨𝑠𝑐,𝑚

,1𝑚

⟩ = 0.

𝑚sc/2

𝑚sc/2

sc

sc

sc

sc

Φ𝑐,𝑚

∶ ℝ𝑚0 → ℝ𝑚, Φ𝑐,𝑚

(𝑢) ∶= (𝑢, 𝑠𝑐,𝑚

).

Define the scaffold embedding

sc

sc

sc

Let 𝜋dyn ∶ ℝ𝑚 → ℝ𝑚0 be the projection onto the first 𝑚0 coordinates, and let 𝜋sc ∶ ℝ𝑚 → ℝ𝑚

sc be the projection onto the last 𝑚sc coordinates:

𝜋dyn(𝑧1,…,𝑧𝑚

0+𝑚sc) = (𝑧1,…,𝑧𝑚

), 𝜋sc(𝑧1,…,𝑧𝑚

0+𝑚sc) = (𝑧𝑚

0+1,…,𝑧𝑚

0+𝑚sc).

0

- Lemma J.1 (Approximate linearity of LayerNorm on scaffold sets). Fix 𝑚0 ≥ 1, 𝜀ln > 0, a compact set 𝒦_set ⊂ ℝ𝑚0, and 𝛿 > 0. Then there exist an even 𝑚sc ≥ 2, a scalar 𝑐 > 0, and a constant 𝑎 > 0 such that

∥𝜋dyn(LN𝜀

(Φ𝑐,𝑚

(𝑢))) − 𝑎𝑢∥

≤ 𝛿.

2

𝑢∈𝒦_set

sup

Moreover, 𝜋sc(Φ𝑐,𝑚

(𝑢)) ≡ 𝑠𝑐,𝑚

is constant on 𝒦_set.

ln

sc

Proof. Let 𝑅 ∶= sup𝑢∈𝒦_set ‖𝑢‖2 < ∞ and fix an even 𝑚sc ≥ 2. Set 𝑚 ∶= 𝑚0 + 𝑚sc. For 𝑢 ∈ 𝒦_set write 𝑧 ∶= Φ𝑐,𝑚

sc

sc

(𝑢) = (𝑢,𝑠𝑐,𝑚

) ∈ ℝ𝑚.

Since ⟨𝑠𝑐,𝑚

,1𝑚

⟩ = 0, we have

sc

sc

√

√

1 𝑚

1 𝑚

𝑚0

𝑚0

∣

‖𝑢‖2 ≤

𝑅.

∑

𝑢𝑖 =∶ 𝜇𝑢, |𝜇𝑢| ≤

∑

𝑢𝑖∣ ≤

𝜇ln(𝑧) =

sc

sc

0

0

𝑚

𝑚

𝑖=1

𝑖=1

Define the mean-centered dynamic vector 𝑢̄ ∶= 𝑢 − 𝜇𝑢1𝑚

0

𝑢̄ 𝜎ln(𝑧)

. Then the dynamic slice of LayerNorm equals

𝜋dyn(LN𝜀

(𝑧)) =

.

ln

1 𝑚

1 𝑚

1 𝜎0

Define the reference scale

(0)) = √

‖2 + 𝜀ln = √

𝜎0 ∶= 𝜎ln(Φ𝑐,𝑚

‖𝑠𝑐,𝑚

(𝑚sc𝑐2) + 𝜀ln, 𝑎 ∶=

.

sc

sc

𝑢̄ − 𝑢 𝜎ln(𝑧)

1 𝜎ln(𝑧)

1 𝜎0

𝑢̄ 𝜎ln(𝑧)

− 𝑎𝑢∥

≤ ∥

∥

+ ∥𝑢(

−

)∥

=∶ 𝑇1 + 𝑇2.

∥

We estimate

2

2

2

For the term 𝑇1 (mean leakage), Since 𝑢̄ − 𝑢 = −𝜇𝑢1𝑚

√

√

0

‖𝜇𝑢1𝑚

‖2 𝜎ln(𝑧)

𝑚0|𝜇𝑢| √

𝑚0 √

,

𝑚0𝑅 𝑚

𝑇1 =

≤

≤

⋅

𝑅 =

.

√

0

𝜀ln

𝜀ln

𝑚

𝜀ln

0

for the term 𝑇2 (variance perturbation), Note that 𝜎ln(𝑧)2 = 𝑚1 ‖𝑧 − 𝜇𝑢1‖2 + 𝜀ln and, because ⟨𝑠𝑐,𝑚

⟩ = 0, we have the exact decomposition

,1𝑚

sc

sc

‖𝑧 − 𝜇𝑢1‖2 = ‖𝑢 − 𝜇𝑢1𝑚

‖2 + ‖𝑠𝑐,𝑚

− 𝜇𝑢1𝑚

‖2 = ‖𝑢̄‖2 + ‖𝑠𝑐,𝑚

‖2 + 𝑚sc𝜇2𝑢, and the cross term vanishes since ⟨𝑠𝑐,𝑚

0

,1𝑚

⟩ = 0. Therefore

sc

sc

sc

𝑚0𝑅2 𝑚2

2𝑅2 𝑚

1 𝑚

1 𝑚

1 𝑚

(‖𝑢̄‖2 + 𝑚sc𝜇2𝑢) ≤

(‖𝑢‖2 + 𝑚sc𝜇2𝑢) ≤

(𝑅2 + 𝑚sc ⋅

) ≤

,

𝜎ln(𝑧)2 − 𝜎02 =

sc

sc

since 𝑚sc ≤ 𝑚 implies 𝑚sc𝑚0/𝑚2 ≤ 𝑚0/𝑚 ≤ 1 for 𝑚 ≥ 𝑚0.

√

√

√

√

√

Using |

𝐴 −

𝐵| ≤ |𝐴 − 𝐵|/(

𝐴 +

𝐵) and 𝜎ln(𝑧),𝜎0 ≥

𝜀ln,

|𝜎ln(𝑧)2 − 𝜎02| 𝜎ln(𝑧) + 𝜎0

(2𝑅2/𝑚) 2

𝑅2 𝑚

|𝜎ln(𝑧) − 𝜎0| ≤

≤

=

.

√

√

𝜀ln

𝜀ln

1 𝜎ln(𝑧)

1 𝜎0

|𝜎ln(𝑧) − 𝜎0| 𝜎ln(𝑧)𝜎0

𝑅2 𝑚

1 𝜀ln

𝑅2 𝑚𝜀3/2ln

∣

∣ =

.

−

≤

⋅

=

√

𝜀ln

Hence

1 𝜎ln(𝑧)

1 𝜎0

𝑅3 𝑚𝜀3/2ln

𝑅2 𝑚𝜀3/2ln

𝑇2 ≤ ‖𝑢‖2 ∣

−

∣ ≤ 𝑅 ⋅

=

.

Therefore

𝑅3 𝑚𝜀3/2ln

𝑚0𝑅 𝑚

∥𝜋dyn(LN𝜀

(Φ𝑐,𝑚

(𝑢))) − 𝑎𝑢∥

≤

+

.

√

Combining,

𝜀ln

2

𝑢∈𝒦_set

sup

Choose 𝑚sc (hence 𝑚 = 𝑚0 + 𝑚sc) large enough so that the right-hand side is ≤ 𝛿. This proves the claim; note that 𝑐 > 0 can be arbitrary and only changes the scaling 𝑎.

ln

sc

| |
|---|

We call a pre-norm LN-Sessa block a Sessa block with Norm = LN𝜀

- J.3 Simulating identity-normalized Sessa blocks with pre-norm LN-Sessa blocks

in the tokenwise preprocessing stage, i.e. 𝑥̃𝑡 = LN𝜀

(𝑥𝑡), and residual 𝑦𝑡 = 𝑥𝑡 + 𝑜𝑡.

- Lemma J.2 (Simulation of an identity-normalized block by a pre-norm LN block on a scaffold). Let 𝐺 ∶ ℝ𝑇×𝑚0 → ℝ𝑇×𝑚0 be a width-𝑚0 concrete Sessa block from Section 3, with Norm = Id. Fix a compact set 𝒦_set ⊂ ℝ𝑇×𝑚0 and 𝜀sim > 0. Then there exist an even 𝑚sc ≥ 2, a scalar 𝑐 > 0, and a width-𝑚 pre-norm LN-Sessa block

ln

ln

𝐺̃∶ ℝ𝑇×(𝑚0+𝑚sc) → ℝ𝑇×(𝑚0+𝑚sc) with Norm = LN𝜀

such that, with 𝑚 ∶= 𝑚0 + 𝑚sc,

∥𝜋dyn(𝐺(Φ̃ 𝑐,𝑚

≤ 𝜀sim, and 𝜋sc(𝐺(Φ̃ 𝑐,𝑚

(𝑥))) − 𝐺(𝑥)∥

(𝑥))) ≡ 𝑠𝑐,𝑚

.

ln

𝐹

𝑥∈𝒦_set

sup

Here Φ𝑐,𝑚

(𝑥) denotes the tokenwise application of Φ𝑐,𝑚

sc

sc

sc

.

sc

sc

𝑆𝒦_set ∶= {𝑥𝑡 ∶ 𝑥 ∈ 𝒦_set, 𝑡 = 0,…,𝑇 − 1} ⊂ ℝ𝑚0. Choose once and for all

Proof. Define the compact set of attainable tokens

𝑎 ∈ (0,𝜀−1/2ln ). Define the continuous map

Δ ∶ ℝ𝑇×𝑚0 → ℝ𝑇×𝑚0, i.e. given 𝑣 ∈ ℝ𝑇×𝑚0, run the Sessa block from the stage after normalization, with the dynamic weights scaled by 1/𝑎, i.e. with first input projection on the dynamic slice 𝑊̃dynin ∶= 𝑎−1𝑊in, 𝑏̃in ∶= 𝑏in, and all other dynamic parameters copied from 𝐺. Then, by construction,

𝐺(𝑥) = 𝑥 + Δ(𝑎𝑥) ∀𝑥 ∈ ℝ𝑇×𝑚0.

Since 𝒦_set is compact, so is 𝑎𝒦_set, and Δ is uniformly continuous on a compact neighborhood of 𝑎𝒦_set. Choose 𝜂UC > 0 such that

‖𝑣 − 𝑣′‖𝐹 ≤ 𝜂UC ⇒ ‖Δ(𝑣) − Δ(𝑣′)‖𝐹 ≤ 𝜀sim for all 𝑣,𝑣′ in that neighborhood.

√

𝜂LN ∶= 𝜂UC/

𝑇.

Fix an even 𝑚sc ≥ 2 (to be chosen large enough), set 𝑚 ∶= 𝑚0 + 𝑚sc, and define

𝑐 ∶= √

(𝑎−2 − 𝜀ln) > 0.

𝑚sc

𝜎0 = √𝑚sc𝑐2

1 𝜎0

Then the reference scale in Lemma J.1 equals exactly

+ 𝜀ln = 𝑎−1, hence

= 𝑎.

𝑚

Inspecting the proof of Lemma J.1, the approximation bound depends on 𝑚 = 𝑚0 + 𝑚sc (and on 𝑆𝒦_set,𝜀ln) and tends to 0 as 𝑚 → ∞; therefore, after increasing the even 𝑚sc if needed, we obtain

∥𝜋dyn(LN𝜀

(Φ𝑐,𝑚

(𝑢))) − 𝑎𝑢∥

≤ 𝜂LN.

2

𝑢∈𝑆𝒦_set

sup

ln

sc

Write the width-𝑚0 input projection of 𝐺 as

𝑊in = [𝑊𝑎 𝑊𝑔], 𝑏in = (𝑏𝑎,𝑏𝑔), with

𝑊𝑎,𝑊𝑔 ∈ ℝ𝑚0×𝑚0, 𝑏𝑎,𝑏𝑔 ∈ ℝ𝑚0. Decompose the widened coordinates as

, where the first summand is the dynamic slice and the second is the scaffold slice. Define

ℝ𝑚 = ℝ𝑚0 ⊕ ℝ𝑚

sc

𝑎−1𝑊𝑔 0 0 0

𝑎−1𝑊𝑎 0 0 0

], 𝑊̃𝑔 = [

𝑊̃𝑎 = [

] ∈ ℝ𝑚×𝑚, and

𝑊̃in = [𝑊̃𝑎 𝑊̃𝑔] ∈ ℝ𝑚×2𝑚, 𝑏̃in = (𝑏𝑎,0𝑚

, 𝑏𝑔,0𝑚

) ∈ ℝ2𝑚.

sc

sc

𝑊𝑄𝑓 0

𝑊𝐾𝑓 0

𝑊𝑄𝑏 0

𝑊𝐾𝑏 0

For the mixer parameters define

𝑊̃𝑄𝑓 = [

], 𝑊̃𝐾𝑓 = [

], 𝑊̃𝑄𝑏 = [

], 𝑊̃𝐾𝑏 = [

] ∈ ℝ𝑚×𝑑𝑘,

𝑊𝑉 0 0 0

) ∈ ℝ𝑚, 𝑏̃𝛾 ∶= 𝑏𝛾.

𝑊̃𝑉 = [

] ∈ ℝ𝑚×𝑚, 𝑤̃𝛾 = (𝑤𝛾,0𝑚

sc

𝑊out 0 0 0

For the output map define

] ∈ ℝ𝑚×𝑚, 𝑏̃out = (𝑏out,0𝑚

𝑊̃out = [

) ∈ ℝ𝑚.

sc

All remaining scaffold rows and columns are set to zero. Thus, once the pre-norm token

𝑧𝑡 ∶= LN𝜀

(𝑋𝑡)

is formed, every learned linear map in 𝐺̃ reads only 𝜋dyn(𝑧𝑡), while the residual increment has zero scaffold coordinates.

ln

For 𝑋 = Φ𝑐,𝑚

(𝑥), define

(𝑋𝑡)) ∈ ℝ𝑚0. Then the widened block has

𝑣𝑡 ∶= 𝜋dyn(LN𝜀

sc

ln

), hence

𝑎̃𝑡 = (𝑎−1𝑣𝑡𝑊𝑎 + 𝑏𝑎, 0𝑚

), 𝑔̃𝑡 = (𝑎−1𝑣𝑡𝑊𝑔 + 𝑏𝑔, 0𝑚

sc

sc

).

GELU(𝑎̃𝑡) = (GELU(𝑎−1𝑣𝑡𝑊𝑎 + 𝑏𝑎), 0𝑚

Therefore the forward logits, feedback logits, gains, dynamic mixer output, and dynamic residual increment of 𝐺̃coincide exactly with those of the width-𝑚0 block defining Δ(𝑣), whereas the scaffold part of 𝑓, 𝑠, and of the residual increment is identically zero. Consequently

sc

𝜋dyn(𝐺(Φ̃ 𝑐,𝑚

(𝑥))) = 𝑥 + Δ(𝑣), 𝜋sc(𝐺(Φ̃ 𝑐,𝑚

(𝑥))) = 𝑠𝑐,𝑚

.

For 𝑥 ∈ 𝒦_set, the tokenwise bound above implies

sc

sc

sc

√

𝑇 = 𝜂UC, hence

∥𝜋dyn(LN𝜀

(Φ𝑐,𝑚

(𝑥))) − 𝑎𝑥∥

≤ 𝜂LN

𝐹

ln

sc

∥𝜋dyn(𝐺(Φ̃ 𝑐,𝑚

≤ 𝜀sim. Finally, since the increment has zero scaffold coordinates, the scaffold stays constant: 𝜋sc(𝐺(Φ̃ 𝑐,𝑚

(𝑥))) − 𝐺(𝑥)∥

= ∥Δ(𝜋dyn(LN𝜀

(Φ𝑐,𝑚

(𝑥)))) − Δ(𝑎𝑥)∥

𝐹

𝐹

(𝑥))) ≡ 𝑠𝑐,𝑚

sc

ln

sc

.

sc

sc

| |
|---|

- Corollary J.3 (Universal approximation for pre-norm LN-Sessa). Let 𝒟 ⊂ ℝ𝑇×𝑑

- J.4 Universal approximation for pre-norm LN-Sessa ext be compact and let

𝐹 ∶ 𝒟 → ℝ𝑇×𝑑

be continuous and causal. Fix 𝜀ln > 0 for tokenwise LayerNorm. Then for any 𝜀 > 0 there exist a model width 𝑚 ∈ ℕ∗, an even key/query width 𝑑𝑘, tokenwise adapters

ext

, and a finite-depth pre-norm LN-Sessa network

→ ℝ𝑚, Unembed ∶ ℝ𝑚 → ℝ𝑑

Embed ∶ ℝ𝑑

ext

ext

Sessa,LN𝜀ln(𝑚), such that

𝐺ln ∈ Ω𝑑

𝑘

∥𝐹(𝑥) − Unembed(𝐺ln(Embed(𝑥)))∥

< 𝜀.

𝐹

𝑥∈𝒟

sup

Proof. By Theorem 14 for Norm = Id, choose adapters

, and a concrete Sessa network with Norm = Id

Embed0 ∶ ℝ𝑑

→ ℝ𝑚0, Unembed0 ∶ ℝ𝑚0 → ℝ𝑑

ext

ext

𝐺⋆ ∈ Ω𝑑Sessa𝑘,0 ,Id(𝑚0)

of depth 𝑁layer such that

∥𝐹(𝑥) − Unembed0(𝐺⋆(Embed0(𝑥)))∥

< 𝜀/2. Write

𝐹

𝑥∈𝒟

sup

∘ ⋯ ∘ 𝐺1 as a composition of concrete Sessa blocks with Norm = Id on ℝ𝑇×𝑚0. Let 𝒦_set1 ∶= Embed0(𝒟) (compact). Fix 𝜌nbhd > 0 and define the thickened compacts recursively as in Lemma I.9: 𝒦̃_set1 ∶= 𝒦_set1, 𝒦_set𝑛

𝐺⋆ = 𝐺𝑁

layer

(𝒦̃_set𝑛

), 𝒦̃_set𝑛

layer+1 ∶= 𝐺𝑛

layer+1 ∶= 𝒩𝜌

(𝒦_set𝑛

layer+1) for 𝑛layer = 1,…,𝑁layer.

Since 𝑁layer is finite, the union of attainable token sets

layer

nbhd

layer

𝑁layer

{𝑢𝑡 ∶ 𝑢 ∈ 𝒦̃_set𝑛

⋃

𝑆 ∶=

, 𝑡 = 0,…,𝑇 − 1} ⊂ ℝ𝑚0.

𝑛layer=1

layer

is approximated on 𝒦̃_set𝑛

is a finite union of compact sets and hence compact. By Lemma I.9, choose tolerances 𝜀sim𝑛

> 0 such that if each block 𝐺𝑛

within 𝜀sim𝑛

, then the composed approximation error on 𝒦_set1 is at most 𝜀/2. Moreover, by the same lemma we may (and do) choose them so that

layer

layer

layer

layer

≤ 𝜌nbhd, 𝑛layer = 1,…,𝑁layer.

𝜀sim𝑛

layer

𝑎 ∈ (0,𝜀−1/2ln ). For each layer 𝑛layer, apply the construction from the proof of Lemma J.2 with target accuracy 𝜀sim𝑛

Fix once and for all a scale

scale 𝑎. This yields a required tokenwise LN-approximation tolerance 𝜂LN(𝑛layer) > 0 such that the layer simulation error is ≤ 𝜀sim𝑛

and prescribed

layer

whenever

∥𝜋dyn(LN𝜀

(Φ𝑐,𝑚

(𝑢))) − 𝑎𝑢∥

≤ 𝜂LN(𝑛layer).

layer

2

𝑢∈{𝑣𝑡∶𝑣∈𝒦̃_set𝑛layer, 𝑡=0,…,𝑇−1}

sup

ln

sc

𝜂LN(𝑛layer). Applying the proof of Lemma J.1 to the compact token set 𝑆, choose one even 𝑚sc ≥ 2 and one 𝑐 > 0 such that:

𝜂LN ∶= min

Set

1≤𝑛layer≤𝑁layer

• the induced reference scale equals the prescribed 𝑎, and

∥𝜋dyn(LN𝜀

(Φ𝑐,𝑚

(𝑢))) − 𝑎𝑢∥

≤ 𝜂LN.

•

2

𝑢∈𝑆

sup

Let 𝑚 ∶= 𝑚0 + 𝑚sc and write Φ ∶= Φ𝑐,𝑚

ln

sc

For each 𝑛layer, apply the construction of Lemma J.2 with this common scaffold (𝑚sc,𝑐) to obtain a pre-norm LN concrete Sessa block

.

sc

𝐺̃𝑛

∈ ConcreteSessaBlocksLN

(𝑑𝑘,0,𝑚)

𝜀ln

layer

𝐺̃𝑛

∶ ℝ𝑇×𝑚 → ℝ𝑇×𝑚 such that

viewed as a map

∥𝜋dyn(𝐺̃𝑛

(Φ(ℎ))) − 𝐺𝑛

(ℎ)∥𝐹 ≤ 𝜀sim𝑛

.

layer

ℎ∈𝒦̃_set𝑛layer

sup

layer

layer

layer

𝜋sc(𝐺̃𝑛

∀ℎ ∈ 𝒦̃_set𝑛

. Define the induced dynamic maps

(Φ(ℎ))) ≡ 𝑠𝑐,𝑚

and

layer

sc

layer

(ℎ) ∶= 𝜋dyn(𝐺̃𝑛

∶ 𝒦̃_set𝑛

(Φ(ℎ))). Then

𝐺dyn𝑛

→ ℝ𝑇×𝑚0, 𝐺dyn𝑛

layer

layer

layer

layer

‖𝐺dyn𝑛

(ℎ) − 𝐺𝑛

(ℎ)‖𝐹 ≤ 𝜀sim𝑛

.

ℎ∈𝒦̃_set𝑛layer

sup

layer

layer

layer

𝐺̃𝑛

(ℎ)) ∀ℎ ∈ 𝒦̃_set𝑛

(Φ(ℎ)) = Φ(𝐺dyn𝑛

.

Moreover, by scaffold invariance,

Applying Lemma I.9 to the maps 𝐺𝑛

and 𝐺dyn𝑛

on the dynamic space ℝ𝑇×𝑚0 yields

layer

layer

layer

∘ ⋯ ∘ 𝐺dyn1 )(Embed0(𝑥))∥

≤ 𝜀/2.

∥𝐺⋆(Embed0(𝑥)) − (𝐺dyn𝑁

layer

layer

𝐹

𝑥∈𝒟

sup

layer

𝐺ln ∶= 𝐺̃𝑁

∘ ⋯ ∘ 𝐺̃1 ∈ Ω𝑑Sessa𝑘,0 ,LN

(𝑚).

Define

𝜀ln

layer

Embed(𝑥) ∶= Φ(Embed0(𝑥)) ∈ ℝ𝑇×𝑚, Unembed(𝑢) ∶= Unembed0(𝜋dyn(𝑢)). Since

Finally, define new adapters

Unembed0(ℎ)𝑡 = 𝑅out(𝜋out(ℎ𝑡)), with 𝜋out an orthogonal projection and 𝑅out an isometry, Unembed0 is non-expansive in Frobenius norm. Unembed(𝐺ln(Embed(𝑥))) = Unembed0((𝐺dyn𝑁

∘ ⋯ ∘ 𝐺dyn1 )(Embed0(𝑥))) ∀𝑥 ∈ 𝒟. Therefore,

layer

∥Unembed0(𝐺⋆(Embed0(𝑥))) − Unembed(𝐺ln(Embed(𝑥)))∥

≤ 𝜀/2.

𝐹

𝑥∈𝒟

sup

Combining this with the approximation error 𝜀/2 from the Norm = Id case gives the claim.

| |
|---|

### K Proofs for flexible finite-horizon selective retrieval

- Lemma K.1 (Predecessor focusing from ordered codes). Fix 𝑇 ≥ 1 and 𝜇 ∈ (0,1). Let 𝐼0 < 𝐼1 < ⋯ < 𝐼𝑇 be pairwise disjoint compact intervals in ℝ, and assume all of them lie in (0,∞). Then there exist scalar linear feedback-query/key maps on a single coordinate such that for every token sequence 𝑢 satisfying

⟨𝑢𝑡,𝑒pos⟩ ∈ 𝐼𝑡, 0 ≤ 𝑡 ≤ 𝑇,

the resulting strict-past feedback attention row satisfies

𝑡−2

∑

𝛼𝑏𝑡,𝑡−1 ≥ 1 − 𝜇,

𝛼𝑏𝑡,𝑗 ≤ 𝜇, 1 ≤ 𝑡 ≤ 𝑇.

𝑗=0

Proof. If 𝑇 = 1, the claim is trivial, since the strict past of 𝑡 = 1 contains only the index 0. Assume henceforth that 𝑇 ≥ 2. Let

𝑧𝑡 ∶= ⟨𝑢𝑡,𝑒pos⟩, 0 ≤ 𝑡 ≤ 𝑇. By assumption,

𝑧𝑡 ∈ 𝐼𝑡, 𝐼0 < 𝐼1 < ⋯ < 𝐼𝑇 ⊂ (0,∞).

To implement the focusing inside an actual LN-free Sessa block, we first realize a single dedicated post-GELU scalar coordinate carrying a strictly ordered positive code. Choose one 𝑎-branch coordinate to be

𝑎pos𝑡 = 𝑐 𝑧𝑡 with some fixed 𝑐 > 0. Since 𝑧𝑡 > 0 on all intervals and the exact GELU satisfies GELU′(𝑥) = Φ(𝑥) + 𝑥𝜙(𝑥) > 0 (𝑥 > 0), the scalar map 𝑥 ↦ GELU(𝑐𝑥) is strictly increasing on (0,∞). Hence the post-GELU coordinate

𝜉𝑡 ∶= GELU(𝑐𝑧𝑡) ranges in compact intervals

𝐽𝑡 ∶= GELU(𝑐𝐼𝑡) satisfying

𝐽0 < 𝐽1 < ⋯ < 𝐽𝑇 ⊂ (0,∞). Now define scalar feedback queries and keys from that post-GELU coordinate:

𝑞𝑡𝑏 = Λ𝜉𝑡, 𝑘𝑗𝑏 = Λ𝜉𝑗, with Λ > 0 to be chosen. All unused heads and coordinates are set to zero. Let

𝑚𝑡 ∶= inf𝐽𝑡, 𝑀𝑡 ∶= sup𝐽𝑡. For 2 ≤ 𝑡 ≤ 𝑇, compactness and strict ordering give

Δ𝑡 ∶= 𝑚𝑡−1 − 𝑀𝑡−2 > 0. Set

Δ ∶= min

Δ𝑡 > 0, 𝑚∗ ∶= min

𝑚𝑡 > 0.

2≤𝑡≤𝑇

0≤𝑡≤𝑇

For every 2 ≤ 𝑡 ≤ 𝑇, every 𝑗 ≤ 𝑡 − 2, and every admissible input 𝑢,

𝑞𝑡𝑏𝑘𝑡−1𝑏 − 𝑞𝑡𝑏𝑘𝑗𝑏 = Λ2𝜉𝑡(𝜉𝑡−1 − 𝜉𝑗) ≥ Λ2𝑚∗Δ. Hence each non-predecessor strict-past logit is smaller than the predecessor logit by at least Λ2𝑚∗Δ.

𝑡−2

∑

exp(⟨𝑞𝑡𝑏,𝑘𝑗𝑏⟩ − ⟨𝑞𝑡𝑏,𝑘𝑡−1𝑏 ⟩) ≤ 𝑇 𝑒−Λ2𝑚∗Δ.

Therefore

𝑗=0

Choose Λ so large that

𝜇 1 − 𝜇

. Then the softmax formula yields

𝑇 𝑒−Λ2𝑚∗Δ ≤

1 1 + ∑𝑡−2𝑗=0 𝑒⟨𝑞

𝛼𝑏𝑡,𝑡−1 =

𝑡,𝑘𝑏𝑗⟩−⟨𝑞𝑡𝑏,𝑘𝑏𝑡−1⟩ ≥ 1 − 𝜇,

𝑏

𝑡−2

∑

𝛼𝑏𝑡,𝑗 ≤ 𝜇.

and consequently

𝑗=0

For 𝑡 = 1 the strict past contains only the predecessor 0, so the claim is trivial.

- Lemma K.2 (RoPE self-focusing). Fix 𝑇 ≥ 0 and 𝜇 ∈ (0,1). Let 𝐼0 < 𝐼1 < ⋯ < 𝐼𝑇 be pairwise disjoint compact intervals in (0,∞). Then there exist forward query/key maps realized inside a single actual RoPE forward branch of an LN-free Sessa block such that for every token sequence 𝑢 satisfying

| |
|---|

⟨𝑢𝑡,𝑒pos⟩ ∈ 𝐼𝑡, 0 ≤ 𝑡 ≤ 𝑇, the resulting full-prefix forward attention row satisfies

𝑡−1

∑

𝛼𝑓𝑡,𝑡 ≥ 1 − 𝜇,

𝛼𝑓𝑡,𝑗 ≤ 𝜇, 0 ≤ 𝑡 ≤ 𝑇.

𝑗=0

Proof. If 𝑇 = 0, the statement is trivial. Assume henceforth that 𝑇 ≥ 1. Let

𝑧𝑡 ∶= ⟨𝑢𝑡,𝑒pos⟩, 𝑧𝑡 ∈ 𝐼𝑡.

- As in the proof of Lemma K.1, choose one dedicated 𝑎-branch coordinate

𝑎pos𝑡 = 𝑐 𝑧𝑡 with 𝑐 > 0, and let

𝜉𝑡 ∶= GELU(𝑐𝑧𝑡). Because 𝑧𝑡 > 0 and GELU is strictly increasing on (0,∞), the ranges

𝐽𝑡 ∶= GELU(𝑐𝐼𝑡) are compact, strictly ordered, and positive:

𝐽0 < 𝐽1 < ⋯ < 𝐽𝑇 ⊂ (0,∞).

𝑚𝑡 ∶= inf𝐽𝑡, 𝑀𝑡 ∶= sup𝐽𝑡. Since the intervals are strictly ordered and compact,

Let

𝛿𝑡 ∶= 𝑚𝑡 − 𝑀𝑡−1 > 0, 1 ≤ 𝑡 ≤ 𝑇.

𝛿 ∶= min

𝛿𝑡 > 0, 𝑚∗ ∶= min

𝑚𝑡 > 0.

Set

1≤𝑡≤𝑇

0≤𝑡≤𝑇

𝑞𝑡𝑓 = Λ𝜉𝑡 𝑒1, 𝑘𝑗𝑓 = Λ𝜉𝑗 𝑒1 inside the first 2-dimensional RoPE plane, with all other coordinates and heads set to zero. Let

Now realize the forward query/key pair on a single RoPE plane by setting, before RoPE,

ℓ𝑡,𝑗 ∶= 𝜎𝑘⟨RoPE(𝑞𝑡𝑓),RoPE(𝑘𝑗𝑓)⟩. Then for every 𝑗 ≤ 𝑡,

ℓ𝑡,𝑗 = 𝜎𝑘Λ2𝜉𝑡𝜉𝑗 cos(𝜗𝑡 − 𝜗𝑗) for the corresponding RoPE phases 𝜗𝑡,𝜗𝑗 on that plane. Hence for every 𝑗 < 𝑡,

ℓ𝑡,𝑡 − ℓ𝑡,𝑗 = 𝜎 𝑡(𝜉𝑡 𝑗 cos(𝜗𝑡 − 𝜗𝑗))

𝜉𝑡(𝜉𝑡 − 𝜉𝑗) since cos(⋅) ≤ 1 ≥ 𝜎𝑘Λ2𝑚∗𝛿.

Therefore, for every 1 ≤ 𝑡 ≤ 𝑇,

𝑡−1

∑

exp(ℓ𝑡,𝑗 − ℓ𝑡,𝑡) ≤ 𝑇 𝑒−𝜎𝑘Λ2𝑚∗𝛿.

𝑗=0

Choose Λ so large that

𝜇 1 − 𝜇

. Then the softmax formula gives

𝑇 𝑒−𝜎𝑘Λ2𝑚∗𝛿 ≤

1

𝛼𝑓𝑡,𝑡 =

1 + ∑𝑡−1𝑗=0 𝑒ℓ𝑡,𝑗−ℓ𝑡,𝑡 ≥ 1 − 𝜇, and consequently

𝑡−1

∑

𝛼𝑓𝑡,𝑗 ≤ 𝜇.

𝑗=0

For 𝑡 = 0 the statement is trivial.

- Lemma K.3 (Scaled GELU uniformly approximates ReLU). Assume the exact GELU activation GELU(𝑥) = 𝑥Φ(𝑥).

For 𝐿 > 0, define

1 𝐿

GELU(𝐿𝑢). Then

𝑅𝐿(𝑢) ∶=

1 𝐿

√

∣𝑅𝐿(𝑢) − 𝑢+∣ ≤

, 𝑢+ ∶= max{𝑢,0}.

2𝜋

𝑢∈ℝ

sup

Proof. Since GELU(𝑥) = 𝑥Φ(𝑥),

𝑅𝐿(𝑢) = 𝑢Φ(𝐿𝑢).

| |
|---|

If 𝑢 ≥ 0, then

𝑅𝐿(𝑢) − 𝑢+ = 𝑢Φ(𝐿𝑢) − 𝑢 = −𝑢(1 − Φ(𝐿𝑢)). By the Mills bound

𝜙(𝑣) 𝑣

(𝑣 > 0), we obtain for 𝑢 > 0,

1 − Φ(𝑣) ≤

𝜙(𝐿𝑢) 𝐿

1 𝐿

√

|𝑅𝐿(𝑢) − 𝑢+| = 𝑢(1 − Φ(𝐿𝑢)) ≤

≤

.

2𝜋

The same bound is trivial at 𝑢 = 0. If 𝑢 < 0, then 𝑢+ = 0 and

|𝑅𝐿(𝑢)| = |𝑢|Φ(𝐿𝑢) = |𝑢|(1 − Φ(−𝐿𝑢)). Applying the same Mills bound with 𝑣 = −𝐿𝑢 > 0 yields

𝜙(−𝐿𝑢) 𝐿

𝜙(𝐿𝑢) 𝐿

1 𝐿

√

=

≤

.

|𝑅𝐿(𝑢)| ≤

2𝜋

Combining the two cases proves the claim.

| |
|---|

- Lemma K.4 (Symmetrized scaled GELU equals the identity). Assume the exact GELU activation GELU(𝑥) = 𝑥Φ(𝑥).

For 𝐿 > 0, define

1 𝐿

𝑅𝐿(𝑥) ∶=

GELU(𝐿𝑥), Id𝐿(𝑥) ∶= 𝑅𝐿(𝑥) − 𝑅𝐿(−𝑥). Then

Id𝐿(𝑥) = 𝑥 ∀𝑥 ∈ ℝ. In particular,

2 𝐿

√

.

|Id𝐿(𝑥) − 𝑥| = 0 ≤

2𝜋

𝑥∈ℝ

sup

Proof. Since GELU(𝑥) = 𝑥Φ(𝑥),

𝑅𝐿(𝑥) = 𝑥Φ(𝐿𝑥). Hence

Id𝐿(𝑥) = 𝑥Φ(𝐿𝑥) − (−𝑥)Φ(−𝐿𝑥) = 𝑥(Φ(𝐿𝑥) + Φ(−𝐿𝑥)) = 𝑥, because Φ(−𝑧) = 1 − Φ(𝑧).

- Corollary K.5 (Exact channel read on the 𝑎-branch). Fix a unit vector 𝑒 ∈ ℝ𝑚 and 𝐿 > 0. In an LN-free concrete Sessa block, if two 𝑎-coordinates are chosen as

| |
|---|

𝑎(+)𝑡 = 𝐿⟨𝑢𝑡,𝑒⟩, 𝑎(−)𝑡 = −𝐿⟨𝑢𝑡,𝑒⟩, then the corresponding post-GELU coordinates satisfy

1 𝐿

(𝑎̄(+)𝑡 − 𝑎̄(−)𝑡 ) = ⟨𝑢𝑡,𝑒⟩ ∀𝑡. Hence any scalar input channel can be read exactly by a linear value projection from two 𝑎-slots. Proof. Apply Lemma K.4 pointwise with 𝑥 = ⟨𝑢𝑡,𝑒⟩.

| |
|---|

- Lemma K.6 (Plateau window from four scaled GELUs). Fix 𝑇 ≥ 0 and pairwise disjoint compact intervals

𝐼0 < 𝐼1 < ⋯ < 𝐼𝑇 ⊂ (0,∞). Fix a target index 𝜏∗ ∈ {0,…,𝑇} and an accuracy parameter 𝜂 ∈ (0,1). Then there exist real numbers

𝑎− < 𝑎+ < 𝑏− < 𝑏+ and a scalar function 𝑊𝜂 ∶ ℝ → ℝ of the form

𝑅𝐿(𝑥 − 𝑏−) − 𝑅𝐿(𝑥 − 𝑏+) 𝑏+ − 𝑏− for some 𝐿 > 0, such that

𝑅𝐿(𝑥 − 𝑎−) − 𝑅𝐿(𝑥 − 𝑎+) 𝑎+ − 𝑎−

𝑊𝜂(𝑥) =

−

, |𝑊𝜂(𝑥)| ≤ 𝜂 for 𝑥 ∈⋃

𝜂(𝑥) − 1| ≤ 𝜂 for 𝑥 ∈ 𝐼𝜏

𝐼𝑡,

∗

𝑡≠𝜏∗

|𝑊𝜂(𝑥)| ≤ 1 + 𝜂.

and

𝑥∈ℝ

Moreover, 𝑊𝜂 is realizable exactly as a linear combination of four 𝑎-branch GELU coordinates inside a single LN-free Sessa block.

sup

< 𝑏− < 𝑏+ such that

𝑎− < 𝑎+ < inf𝐼𝜏

≤ sup𝐼𝜏

Proof. Because the intervals are pairwise disjoint, compact, and strictly ordered, one can choose

∗

∗

⊂ [𝑎+,𝑏−], ⋃ 𝑡≠𝜏∗

𝐼𝜏

𝐼𝑡 ⊂ (−∞,𝑎−] ∪ [𝑏+,∞).

∗

(𝑥 − 𝑎−)+ − (𝑥 − 𝑎+)+ 𝑎+ − 𝑎−

(𝑥 − 𝑏−)+ − (𝑥 − 𝑏+)+ 𝑏+ − 𝑏−

Define the exact piecewise-linear plateau window

𝑤(𝑥) ∶=

−

.

𝑤(𝑥) = 1 on [𝑎+,𝑏−] ⊃ 𝐼𝜏

,

By construction,

𝑤(𝑥) = 0 on (−∞,𝑎−] ∪ [𝑏+,∞) ⊃⋃ 𝑡≠𝜏∗

𝐼𝑡,

∗

0 ≤ 𝑤(𝑥) ≤ 1 ∀𝑥 ∈ ℝ.

and

1 𝐿

Now replace each ReLU ramp by the scaled-GELU ramp from Lemma K.3:

𝑅𝐿(𝑢) =

GELU(𝐿𝑢).

𝑅𝐿(𝑥 − 𝑎−) − 𝑅𝐿(𝑥 − 𝑎+) 𝑎+ − 𝑎−

𝑅𝐿(𝑥 − 𝑏−) − 𝑅𝐿(𝑥 − 𝑏+) 𝑏+ − 𝑏−

𝑊𝐿(𝑥) ∶=

−

.

Set

2 𝐿

1 𝑎+ − 𝑎−

1 𝑏+ − 𝑏−

Using Lemma K.3 on each of the four ramp terms,

√

‖𝑊𝐿 − 𝑤‖∞ ≤

(

+

).

2𝜋

Choose 𝐿 so large that the right-hand side is at most 𝜂. Then on 𝐼𝜏

, where 𝑤 ≡ 1,

∗

|𝑊𝐿 − 1| ≤ 𝜂,

and on ⋃𝑡≠𝜏

𝐼𝑡, where 𝑤 ≡ 0,

|𝑊𝐿| ≤ 𝜂. Also, since 0 ≤ 𝑤 ≤ 1,

∗

|𝑊𝐿(𝑥)| ≤ |𝑤(𝑥)| + 𝜂 ≤ 1 + 𝜂 ∀𝑥.

Set 𝑊𝜂 ∶= 𝑊𝐿. Finally, 𝑊𝜂 is realizable exactly inside one LN-free Sessa block because each term

1 𝐿

GELU(𝐿(𝑥 − 𝑐))

𝑅𝐿(𝑥 − 𝑐) =

is one 𝑎-branch GELU coordinate applied to an aﬀine function of the tokenwise scalar 𝑥, and the displayed linear combination is absorbed into the value projection.

- Lemma K.7 (Writing a window into an auxiliary channel). Fix 𝑇 ≥ 0, 𝜏∗ ∈ {0,…,𝑇}, and 𝜀 ∈ (0,1). Let 𝒦_set ⊂ (ℝ𝑚)𝑇+1 be compact. Assume that for some unit vector 𝑒pos ∈ ℝ𝑚,

| |
|---|

𝐼𝑡 ∶= {⟨𝑢𝑡,𝑒pos⟩ ∶ 𝑢 ∈ 𝒦_set}, 0 ≤ 𝑡 ≤ 𝑇, are compact and strictly ordered:

𝐼0 < 𝐼1 < ⋯ < 𝐼𝑇 ⊂ (0,∞). Fix orthonormal directions

𝑒pos, 𝑒sig, 𝑒aux

and let 𝐸carry ⊂ ℝ𝑚 be any fixed subspace orthogonal to all three. Assume moreover that 𝑚 ≥ 6. Then there exists a single LN-free Sessa block

∗,𝜀 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1 such that the feedback branch is switched off, the 𝑒pos-, 𝑒sig-, and 𝐸carry-channels are preserved exactly, and, writing 𝑎𝑡(𝑢) ∶= ⟨𝑊𝑇,𝜏write

𝑊𝑇,𝜏write

∗,𝜀(𝑢)𝑡, 𝑒aux⟩, one has uniformly on 𝒦_set,

(𝑢) − 1| ≤ 𝜀, |𝑎𝑡(𝑢)| ≤ 𝜀 (𝑡 ≠ 𝜏∗), and

|𝑎𝜏

∗

|𝑎𝑡(𝑢)| ≤ 2.

0≤𝑡≤𝑇

𝑢∈𝒦_set

sup

sup

Proof. Choose 𝜂 ∈ (0,𝜀) so small that

𝜂 + 𝜂(1 + 𝜂) ≤ 𝜀.

- Apply Lemma K.6 to obtain a scalar function 𝑊𝜂 satisfying

), |𝑊𝜂(𝑥)| ≤ 𝜂 (𝑥 ∈⋃ 𝑡≠𝜏∗

|𝑊𝜂(𝑥) − 1| ≤ 𝜂 (𝑥 ∈ 𝐼𝜏

𝐼𝑡), sup

|𝑊𝜂(𝑥)| ≤ 1 + 𝜂.

𝑥

∗

Next apply Lemma K.2 with parameter 𝜇 ∶= 𝜂. This gives a forward branch whose full-prefix row satisfies

𝛼𝑓𝑡,𝑡 ≥ 1 − 𝜂, ∑ 𝑗<𝑡

𝛼𝑓𝑡,𝑗 ≤ 𝜂 (0 ≤ 𝑡 ≤ 𝑇).

We now build the block. Values. Choose a positive constant 𝑐1 such that

GELU(𝑐1) = 1. Realize the first value coordinate by a constant 𝑎-branch coordinate equal to 𝑐1, so that

𝑣𝑡(0) ≡ 1. Realize the second value coordinate as

𝑣𝑡(1) = 𝑊𝜂(⟨𝑢𝑡,𝑒pos⟩),

using Lemma K.6. Gate and output on the auxiliary channel. Choose two gate coordinates

𝑔𝑡(0) = ⟨𝑢𝑡,𝑒aux⟩, 𝑔𝑡(1) ≡ 1.

Choose the output projection on the 𝑒aux-channel with coeﬀicients (−1,+1) on the two gated coordinates and zero on all other channels. Because the row sum of attention is exactly 1,

𝑠(0)𝑡 = ∑ 𝑗≤𝑡

𝛼𝑓𝑡,𝑗 ⋅ 1 = 1.

𝑎𝑡(𝑢) = ⟨𝑢𝑡,𝑒aux⟩ − 𝑠(0)𝑡 ⟨𝑢𝑡,𝑒aux⟩ + 𝑠(1)𝑡 = 𝑠(1)𝑡 , where

Hence the auxiliary output becomes

𝛼𝑓𝑡,𝑗 𝑊𝜂(⟨𝑢𝑗,𝑒pos⟩). Thus the block overwrites the auxiliary channel by the forward average of 𝑊𝜂. All other output columns are zero, so the 𝑒pos-, 𝑒sig-, and 𝐸carry-channels are preserved exactly. It remains to bound 𝑎𝑡 = 𝑠(1)𝑡 . Target time 𝑡 = 𝜏∗. All indices 𝑗 < 𝜏∗ are off-target, hence

𝑠(1)𝑡 = ∑ 𝑗≤𝑡

|𝑊𝜂(⟨𝑢𝑗,𝑒pos⟩)| ≤ 𝜂.

,𝑒pos⟩) ∈ [1 − 𝜂,1 + 𝜂]. Therefore

𝑊𝜂(⟨𝑢𝜏

- At the target index,

∗

(𝑢) ≥ (1 − 𝜂)(1 − 𝜂) − 𝜂 ⋅ 𝜂 ≥ 1 − 2𝜂, and

𝑎𝜏

∗

(𝑢) ≤ (1 − 𝜂)(1 + 𝜂) + 𝜂 ⋅ 𝜂 ≤ 1 + 𝜂. Hence

𝑎𝜏

∗

|𝑎𝜏

(𝑢) − 1| ≤ 2𝜂 ≤ 𝜀.

∗

Off-target times 𝑡 < 𝜏∗. Then all visible indices 𝑗 ≤ 𝑡 are off-target, so |𝑎𝑡(𝑢)| ≤ 𝜂 ≤ 𝜀.

Off-target times 𝑡 > 𝜏∗. Then self-mass is on an off-target index, so the self contribution is at most 𝜂 in magnitude, while all nonself mass is at most 𝜂 and every visible value has magnitude at most 1 + 𝜂. Thus

|𝑎𝑡(𝑢)| ≤ 𝜂 + 𝜂(1 + 𝜂) ≤ 𝜀.

|𝑎𝑡(𝑢)| = |𝑠(1)𝑡 | ≤ ∑ 𝑗≤𝑡

𝛼𝑓𝑡,𝑗 sup

|𝑊𝜂(𝑥)| ≤ 1 + 𝜂 ≤ 2,

Finally, from

𝑥

we obtain the uniform bound.

- Definition 11 (Signal-fiber saturation). Fix 𝑇 ≥ 0, a unit signal direction 𝑒sig ∈ ℝ𝑚, and a set 𝒦_set ⊂ (ℝ𝑚)𝑇+1. For 𝛿 ≥ 0, define

Satsig𝛿 (𝒦_set) ∶= {𝑢 + 𝑧 ∶ 𝑢 ∈ 𝒦_set, 𝑧𝑡 = 𝑎𝑡𝑒sig, max 0≤𝑡≤𝑇

|𝑎𝑡| ≤ 𝛿}. Equivalently,

Satsig𝛿 (𝒦_set) = {𝑢 +

𝑇

∑

𝑡=0

𝑎𝑡𝑒sig1[⋅ = 𝑡] ∶ 𝑢 ∈ 𝒦_set, max

𝑡

|𝑎𝑡| ≤ 𝛿}.

If 𝒦_set is compact, then Satsig𝛿 (𝒦_set) is compact.

- Definition 12 (Exact signal transport). Fix 𝑇 ≥ 0, a unit signal direction 𝑒sig ∈ ℝ𝑚, and a control subspace 𝐸ctrl ⊂ ℝ𝑚 with 𝑒sig ⟂ 𝐸ctrl. Let Πctrl denote the orthogonal projection onto 𝐸ctrl, and let

| |
|---|

𝜋sig(𝑣) ∶= ⟨𝑣,𝑒sig⟩.

For 𝑢 = (𝑢𝑡)𝑇𝑡=0 ∈ (ℝ𝑚)𝑇+1, write

𝑐𝑡𝑢 ∶= Πctrl𝑢𝑡, 𝑥𝑢𝑡 ∶= 𝜋sig(𝑢𝑡). A causal map

𝐵 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1 is said to have exact signal transport along 𝑒sig over 𝐸ctrl on a set 𝒦_set ⊂ (ℝ𝑚)𝑇+1 if: (i) 𝐵 preserves the control channels exactly:

Πctrl𝐵(𝑢)𝑡 = 𝑐𝑡𝑢 ∀𝑢 ∈ 𝒦_set, ∀0 ≤ 𝑡 ≤ 𝑇; (ii) there exists a scalar lower-triangular kernel

𝒯𝑢𝐵(𝑖,𝑗), 0 ≤ 𝑗 ≤ 𝑖 ≤ 𝑇, depending only on the control stream 𝑐𝑢 = (𝑐𝑡𝑢)𝑇𝑡=0, such that

𝑖

∑

𝒯𝑢𝐵(𝑖,𝑗)𝑥𝑢𝑗 ∀𝑢 ∈ 𝒦_set, ∀0 ≤ 𝑖 ≤ 𝑇.

𝜋sig(𝐵(𝑢)𝑖) =

𝑗=0

- Lemma K.8 (Transport calculus on signal fibers). Fix 𝑇 ≥ 0, 𝑒sig, 𝐸ctrl, and a compact set 𝒦_set ⊂ (ℝ𝑚)𝑇+1. Fix 𝛿 > 0.

- (i) Jacobian extraction. Assume 𝐵 is continuously differentiable on a neighborhood of Satsig𝛿 (𝒦_set), and

that 𝐵 has signal-blind exact scalar transport along 𝑒sig over 𝐸ctrl on Satsig𝛿 (𝒦_set), with kernel 𝒯𝑢𝐵. Then for every 𝑢 ∈ 𝒦_set and every 0 ≤ 𝑗 ≤ 𝑖 ≤ 𝑇,

𝑒⊤sig

𝜕𝐵(𝑢)𝑖 𝜕𝑢𝑗

𝑒sig = 𝒯𝑢𝐵(𝑖,𝑗).

- (ii) Composition. Assume 𝐵1 has signal-blind exact scalar transport along 𝑒sig over 𝐸ctrl on Satsig𝛿 (𝒦_set),

with kernel 𝒯𝑢𝐵

, and preserves the control channels exactly there. Assume 𝐵2 has signal-blind exact scalar transport along 𝑒sig over 𝐸ctrl on 𝐵1(Satsig𝛿 (𝒦_set)), with kernel 𝒯𝑣𝐵

1

exactly there. Then 𝐵2∘𝐵1 also has signal-blind exact scalar transport along 𝑒sig over 𝐸ctrl on Satsig𝛿 (𝒦_set), and its kernel is the lower-triangular kernel product

2

, and preserves the control channels

𝑖

∑

𝒯𝑢𝐵

2∘𝐵1(𝑖,𝑗) =

𝒯

𝐵2 (𝑖,𝑟)𝒯𝑢𝐵

(𝑟,𝑗).

1(𝑢)

𝑟=𝑗

1

Proof. For (i), fix 𝑢 ∈ 𝒦_set, 𝑗 ≤ 𝑖, and define

𝑢(ℎ) ∶= 𝑢 + ℎ𝑒sig1[⋅ = 𝑗].

For |ℎ| < 𝛿, one has 𝑢(ℎ) ∈ Satsig𝛿 (𝒦_set). Because 𝑒sig ⟂ 𝐸ctrl,

Πctrl𝑢(ℎ)𝑡 = Πctrl𝑢𝑡 ∀𝑡,

so the control stream is unchanged. Since the transport kernel depends only on the control stream, the same kernel 𝒯𝑢𝐵 applies to both 𝑢 and 𝑢(ℎ). Therefore

𝑖

𝒯𝐵𝑢 (𝑖,𝑟)(𝑥𝑢𝑟(ℎ) − 𝑥𝑢𝑟)

𝜋sig(𝐵(𝑢(ℎ))𝑖) − 𝜋sig(𝐵(𝑢)𝑖) =

∑

𝑟=0

= ℎ𝒯𝑢𝐵(𝑖,𝑗). Divide by ℎ and let ℎ → 0. Since 𝐵 is 𝐶1,

𝜕𝐵(𝑢)𝑖 𝜕𝑢𝑗

𝑒⊤sig

𝑒sig = 𝒯𝑢𝐵(𝑖,𝑗).

For (ii), let 𝑢 ∈ Satsig𝛿 (𝒦_set). Because 𝐵1 preserves the control channels exactly,

Πctrl𝐵1(𝑢)𝑡 = Πctrl𝑢𝑡, so the control stream of 𝐵1(𝑢) equals that of 𝑢. Hence

𝑟

𝜋sig(𝐵1(𝑢)𝑟) =

∑

𝒯𝑢𝐵

(𝑟,𝑗)𝑥𝑢𝑗 .

𝑗=0

1

Applying 𝐵2 and using exact control preservation again,

𝑖

𝜋sig(𝐵2(𝐵1(𝑢))𝑖) =

∑

𝒯

𝐵2 (𝑖,𝑟)𝜋sig(𝐵1(𝑢)𝑟)

1(𝑢)

𝑟=0

𝑖

𝑟

=

∑

𝒯

𝐵2 (𝑖,𝑟)

∑

𝒯𝑢𝐵

(𝑟,𝑗)𝑥𝑢𝑗

1(𝑢)

𝑟=0

𝑗=0

1

𝑖

𝑖

∑

(

∑

=

𝒯

𝐵2 (𝑖,𝑟)𝒯𝑢𝐵

(𝑟,𝑗))𝑥𝑢𝑗 .

1(𝑢)

𝑟=𝑗

𝑗=0

1

This is exactly the stated kernel-product formula.

- Definition 13 (Transparent preprocessing). Fix 𝑇 ≥ 0, a unit signal direction 𝑒sig ∈ ℝ𝑚, and a control subspace 𝐸ctrl ⊂ ℝ𝑚 with 𝑒sig ⟂ 𝐸ctrl. Let

| |
|---|

Πctrl ∶ ℝ𝑚 → 𝐸ctrl be the orthogonal projection and

𝜋sig(𝑣) ∶= ⟨𝑣,𝑒sig⟩. A causal map

𝑅 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1

is said to be signal-transparent along 𝑒sig over 𝐸ctrl on a set 𝒦_set ⊂ (ℝ𝑚)𝑇+1 if for every 𝑢 ∈ 𝒦_set, every 𝜏 ∈ {0,…,𝑇}, and every suﬀiciently small scalar 𝑎 such that

𝑢(𝑎,𝜏) ∶= 𝑢 + 𝑎𝑒sig1[⋅ = 𝜏] remains in the domain under consideration, one has

Πctrl𝑅(𝑢(𝑎,𝜏))𝑡 = Πctrl𝑅(𝑢)𝑡 ∀𝑡, and

𝜋sig(𝑅(𝑢(𝑎,𝜏))𝑡) = 𝜋sig(𝑅(𝑢)𝑡) + 𝑎1[𝑡 = 𝜏] ∀𝑡.

- Lemma K.9 (Transparent preprocessing and Jacobians). Fix 𝑇 ≥ 0, 𝑒sig, and 𝐸ctrl. Let 𝑅 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1, 𝐵 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1

be continuously differentiable on neighborhoods of 𝒦_set and Satsig𝛿 (𝑅(𝒦_set)), respectively, for some 𝛿 > 0. Assume:

- (i) 𝑅 is signal-transparent along 𝑒sig over 𝐸ctrl on 𝒦_set;
- (ii) 𝐵 has signal-blind exact scalar transport along 𝑒sig over 𝐸ctrl on Satsig𝛿 (𝑅(𝒦_set)), with kernel

𝒯𝑣𝐵(𝑖,𝑗), 𝑣 ∈ Satsig𝛿 (𝑅(𝒦_set)), 0 ≤ 𝑗 ≤ 𝑖 ≤ 𝑇. Then for every 𝑢 ∈ 𝒦_set and every 0 ≤ 𝑗 ≤ 𝑖 ≤ 𝑇,

𝜕(𝐵 ∘ 𝑅)(𝑢)𝑖 𝜕𝑢𝑗

𝑒sig = 𝒯𝑅(𝑢)𝐵 (𝑖,𝑗).

𝑒⊤sig

Proof. Fix 𝑢 ∈ 𝒦_set and 0 ≤ 𝑗 ≤ 𝑖 ≤ 𝑇. For suﬀiciently small 𝑎, define

𝑢(𝑎,𝑗) ∶= 𝑢 + 𝑎𝑒sig1[⋅ = 𝑗].

𝑣 ∶= 𝑅(𝑢), 𝑣(𝑎) ∶= 𝑅(𝑢(𝑎,𝑗)). By signal-transparency of 𝑅,

Set

Πctrl𝑣𝑡(𝑎) = Πctrl𝑣𝑡 ∀𝑡, and

𝜋sig(𝑣𝑡(𝑎)) = 𝜋sig(𝑣𝑡) + 𝑎1[𝑡 = 𝑗] ∀𝑡.

Hence 𝑣(𝑎) ∈ Satsig𝛿 (𝑅(𝒦_set)) for all suﬀiciently small |𝑎|, and 𝑣(𝑎) and 𝑣 have the same control stream. Therefore the same kernel 𝒯𝑣𝐵 applies to both 𝑣 and 𝑣(𝑎), so

𝑖

𝜋sig(𝐵(𝑣(𝑎))𝑖) − 𝜋sig(𝐵(𝑣)𝑖) =

∑

𝒯𝑣𝐵(𝑖,𝑟)(𝜋sig(𝑣𝑟(𝑎)) − 𝜋sig(𝑣𝑟))

𝑟=0

= 𝑎𝒯𝑣𝐵(𝑖,𝑗). Divide by 𝑎 and let 𝑎 → 0. Since 𝐵 ∘ 𝑅 is continuously differentiable,

𝜕(𝐵 ∘ 𝑅)(𝑢)𝑖 𝜕𝑢𝑗

𝑒⊤sig

𝑒sig = 𝒯𝑅(𝑢)𝐵 (𝑖,𝑗).

Corollary K.10 (Signal-fiber stability of the control-driven blocks). Fix 𝛿 ≥ 0. In each of Lemmas K.11, K.12, K.15, K.17, and K.20, replace the base compact set 𝒦_set (or 𝒦_set𝐻) by its bounded signal-fiber saturation Satsig𝛿 (𝒦_set) (or Satsig𝛿 (𝒦_set𝐻)). Then the same concrete block or network satisfies the same conclusion, with the same constants.

| |
|---|

In particular, whenever one of these lemmas yields signal-blind exact scalar transport along 𝑒sig, that exact transport statement also holds on every bounded signal-fiber saturation of the same control-side compact set.

Proof. In each listed lemma, the hypotheses and parameter choices depend only on channels orthogonal to 𝑒sig: ordered positional ranges, two-sided tail/profile bounds, exact vanishing of designated scratch/profile channels,

and carried control channels. These quantities are unchanged when 𝒦_set is replaced by Satsig𝛿 (𝒦_set). Moreover, the concrete constructions preserve the relevant control channels exactly and treat the 𝑒sig-channel linearly. Therefore the original proofs apply verbatim on the saturated set, with the same constants.

- Lemma K.11 (Local multiplier). Fix 𝑇 ≥ 0 and 𝛿 > 0. Let 𝒦_set ⊂ (ℝ𝑚)𝑇+1 be compact. Assume that for some unit vector 𝑒pos ∈ ℝ𝑚,

| |
|---|

𝐼𝑡 ∶= {⟨𝑢𝑡,𝑒pos⟩ ∶ 𝑢 ∈ 𝒦_set}, 0 ≤ 𝑡 ≤ 𝑇, are compact and strictly ordered in (0,∞). Fix orthonormal directions

𝑒pos, 𝑒sig, 𝑒aux

and let 𝐸carry ⊂ ℝ𝑚 be any fixed subspace orthogonal to all three. Assume moreover that 𝑚 ≥ 4. Assume moreover that the auxiliary channel is uniformly bounded:

∣⟨𝑢𝑡,𝑒aux⟩∣ ≤ 𝑀

0≤𝑡≤𝑇

𝑢∈𝒦_set

sup

sup

for some finite 𝑀.

𝑀𝑇,𝛿loc ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1

Then there exists a single LN-free Sessa block

such that the feedback branch is switched off, the 𝑒pos-, 𝑒aux-, and 𝐸carry-channels are preserved exactly, and 𝑀𝑇,𝛿loc has signal-blind exact scalar transport along 𝑒sig over

𝐸ctrl ∶= span{𝑒pos,𝑒aux} ⊕ 𝐸carry, with diagonal kernel

𝒯𝑢𝑀

(𝑖,𝑗) = 𝐷loc𝑢 (𝑖)1[𝑖 = 𝑗];

∣𝐷loc𝑢 (𝑡) − ⟨𝑢𝑡,𝑒aux⟩∣ ≤ 𝛿 ∀𝑢 ∈ 𝒦_set, ∀0 ≤ 𝑡 ≤ 𝑇. In particular,

loc

𝜕𝑀𝑇,𝛿loc (𝑢)𝑖 𝜕𝑢𝑗

𝑒⊤sig

𝑒sig = 𝐷loc𝑢 (𝑖)1[𝑖 = 𝑗].

𝜇 ∈ (0,1)

Proof. Choose a parameter

to be fixed later, and apply Lemma K.2 with this 𝜇. Choose a positive constant 𝑐1 such that

GELU(𝑐1) = 1. Realize one forward value coordinate by the constant 1:

𝑣𝑡(0) ≡ 1.

Next read the auxiliary channel exactly using Corollary K.5. Choose two 𝑎-slots

𝑎(+)𝑡 = 𝐿⟨𝑢𝑡,𝑒aux⟩, 𝑎(−)𝑡 = −𝐿⟨𝑢𝑡,𝑒aux⟩, for any fixed 𝐿 > 0, and choose the value projection so that

1 𝐿

(𝑎̄(+)𝑡 − 𝑎̄(−)𝑡 ) = ⟨𝑢𝑡,𝑒aux⟩.

𝑣𝑡(1) =

𝑔𝑡(0) = ⟨𝑢𝑡,𝑒sig⟩, 𝑔𝑡(1) = ⟨𝑢𝑡,𝑒sig⟩.

Choose two gate coordinates, both equal to the signal:

Choose the output projection on the 𝑒sig-channel with coeﬀicients (−1,+1) on these two gated coordinates and zero on all other output channels.

Since the forward row sums to 1,

𝛼𝑓𝑡,𝑗 ⋅ 1 = 1.

𝑠(0)𝑡 = ∑ 𝑗≤𝑡

⟨𝑀𝑇,𝛿loc (𝑢)𝑡,𝑒sig⟩ = ⟨𝑢𝑡,𝑒sig⟩ − 𝑠(0)𝑡 ⟨𝑢𝑡,𝑒sig⟩ + 𝑠(1)𝑡 ⟨𝑢𝑡,𝑒sig⟩ = 𝑠(1)𝑡 ⟨𝑢𝑡,𝑒sig⟩, where

Hence the signal output equals

𝑠(1)𝑡 = ∑ 𝑗≤𝑡

𝛼𝑓𝑡,𝑗 𝑣𝑗(1) = ∑ 𝑗≤𝑡

𝛼𝑓𝑡,𝑗 ⟨𝑢𝑗,𝑒aux⟩.

𝐷loc𝑢 (𝑡) ∶= 𝑠(1)𝑡 . Then

Define

⟨𝑀𝑇,𝛿loc (𝑢)𝑡,𝑒sig⟩ = 𝐷loc𝑢 (𝑡)⟨𝑢𝑡,𝑒sig⟩, which is exactly signal-blind exact scalar transport with diagonal kernel

𝒯𝑢𝑀

(𝑖,𝑗) = 𝐷loc𝑢 (𝑖)1[𝑖 = 𝑗].

The coeﬀicient 𝐷loc𝑢 (𝑡) depends only on the forward weights and on the auxiliary values ⟨𝑢𝑗,𝑒aux⟩. By construction, both depend only on the 𝑒pos-, 𝑒aux-, and 𝐸carry-channels, not on the signal channel. Thus the transport is signalblind over

loc

𝐸ctrl ∶= span{𝑒pos,𝑒aux} ⊕ 𝐸carry.

All output columns except the signal column are zero, so the 𝑒pos-, 𝑒aux-, and 𝐸carry-channels are preserved exactly. It remains to estimate 𝐷loc𝑢 (𝑡). Since the auxiliary read is exact,

𝐷loc𝑢 (𝑡) − ⟨𝑢𝑡,𝑒aux⟩ = ∑ 𝑗≤𝑡

𝛼𝑓𝑡,𝑗(⟨𝑢𝑗,𝑒aux⟩ − ⟨𝑢𝑡,𝑒aux⟩) = ∑ 𝑗<𝑡

𝛼𝑓𝑡,𝑗(⟨𝑢𝑗,𝑒aux⟩ − ⟨𝑢𝑡,𝑒aux⟩).

𝛼𝑓𝑡,𝑗.

|𝐷loc𝑢 (𝑡) − ⟨𝑢𝑡,𝑒aux⟩| ≤ 2𝑀 ∑ 𝑗<𝑡

Therefore,

∑

𝛼𝑓𝑡,𝑗 ≤ 𝜇.

By self-focusing,

𝑗<𝑡

|𝐷loc𝑢 (𝑡) − ⟨𝑢𝑡,𝑒aux⟩| ≤ 2𝑀𝜇. Choose

Hence

- 1

- 2

𝛿 2max{𝑀,1}

𝜇 ≤ min{

}. Then

,

|𝐷loc𝑢 (𝑡) − ⟨𝑢𝑡,𝑒aux⟩| ≤ 𝛿 ∀𝑢 ∈ 𝒦_set, ∀0 ≤ 𝑡 ≤ 𝑇.

For any 𝜂 > 0, replacing 𝒦_set by Satsig𝜂 (𝒦_set) leaves the ordered positional ranges (𝐼𝑡)𝑇𝑡=0 and the auxiliary bound 𝑀 unchanged, since only the 𝑒sig-channel is perturbed. The same concrete construction therefore yields the same exact diagonal signal-transport formula on Satsig𝜂 (𝒦_set), with the same coeﬀicients 𝐷loc𝑢 (𝑖), because the forward weights depend only on the positional-control stream and the exact auxiliary read depends only on the 𝑒aux-channel. Applying Lemma K.8(i) gives

𝜕𝑀𝑇,𝛿loc (𝑢)𝑖 𝜕𝑢𝑗

𝑒⊤sig

𝑒sig = 𝐷loc𝑢 (𝑖)1[𝑖 = 𝑗].

- Lemma K.12 (Two-block selector). Fix 𝑇 ≥ 0, 𝜀 ∈ (0,1), and a compact set 𝒦_set ⊂ (ℝ𝑚)𝑇+1. Assume that for some unit vector 𝑒pos ∈ ℝ𝑚 the scalar position ranges

| |
|---|

𝐼𝑡 ∶= {⟨𝑢𝑡,𝑒pos⟩ ∶ 𝑢 ∈ 𝒦_set}, 0 ≤ 𝑡 ≤ 𝑇,

𝐼0 < 𝐼1 < ⋯ < 𝐼𝑇 ⊂ (0,∞). Fix a source index 𝜏∗ ∈ {0,…,𝑇} and orthonormal directions

are compact and strictly ordered:

𝑒pos, 𝑒sig, 𝑒aux.

Let 𝐸carry ⊂ ℝ𝑚 be any fixed subspace orthogonal to these three directions. Assume moreover that 𝑚 ≥ 6. Then there exists a depth-2 LN-free Sessa network

𝑆𝑇,𝜏

∗,𝜀 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1

such that both constituent blocks have the feedback branch switched off, the 𝑒pos-channel and every channel in 𝐸carry are preserved exactly, and 𝑆𝑇,𝜏

∗,𝜀 has signal-blind exact scalar transport along 𝑒sig over

𝐸ctrl ∶= span{𝑒pos} ⊕ 𝐸carry, with diagonal kernel

𝒯𝑢𝑆(𝑖,𝑗) = 𝐷sel𝑢 (𝑖)1[𝑖 = 𝑗]; Uniformly for all 𝑢 ∈ 𝒦_set,

- 1

- 2

≤ 𝐷sel𝑢 (𝜏∗) ≤ 2, |𝐷sel𝑢 (𝑡)| ≤ 𝜀 (𝑡 ≠ 𝜏∗). In particular,

𝜕𝑆𝑇,𝜏

∗,𝜀(𝑢)𝑖 𝜕𝑢𝑗

𝑒sig = 𝐷sel𝑢 (𝑖)1[𝑖 = 𝑗].

𝑒⊤sig

𝜀 4

𝜀 4

𝜀wr ∶=

, 𝛿mul ∶=

.

Proof. Set

- Apply Lemma K.7 with accuracy 𝜀wr. This yields a forward-only block

𝑊𝑇,𝜏write

∗,𝜀wr

which preserves the 𝑒pos-, 𝑒sig-, and 𝐸carry-channels exactly and writes an auxiliary channel 𝑎𝑡(𝑢) ∶= ⟨𝑊𝑇,𝜏write

∗,𝜀wr(𝑢)𝑡, 𝑒aux⟩ satisfying

𝜀 4

𝜀 4

(𝑡 ≠ 𝜏∗), and

|𝑎𝜏

(𝑢) − 1| ≤

, |𝑎𝑡(𝑢)| ≤

∗

|𝑎𝑡(𝑢)| ≤ 2 ∀𝑡. Now apply Lemma K.11 to the image

∗,𝜀wr(𝒦_set), with the same 𝑒pos,𝑒sig,𝑒aux,𝐸carry, the bound 𝑀 = 2, and accuracy 𝛿mul = 𝜀/4. This yields a forward-only block 𝑀𝑇,𝛿loc

𝒦_set′ ∶= 𝑊𝑇,𝜏write

mul

(𝑤)𝑡,𝑒sig⟩ = 𝐷loc𝑤 (𝑡)⟨𝑤𝑡,𝑒sig⟩ (𝑤 ∈ 𝒦_set′), with

⟨𝑀𝑇,𝛿loc

whose signal transport is exact and diagonal:

𝜀 4

mul

|𝐷loc𝑤 (𝑡) − ⟨𝑤𝑡,𝑒aux⟩| ≤

.

∗,𝜀wr. Since the writer preserves the signal channel exactly,

𝑆𝑇,𝜏

∗,𝜀 ∶= 𝑀𝑇,𝛿loc

∘ 𝑊𝑇,𝜏write

Define

mul

∗,𝜀wr(𝑢)𝑡,𝑒sig⟩ = ⟨𝑢𝑡,𝑒sig⟩. Therefore

⟨𝑊𝑇,𝜏write

loc (𝑡)⟨𝑢𝑡,𝑒sig⟩. Set

⟨𝑆𝑇,𝜏

∗,𝜀(𝑢)𝑡,𝑒sig⟩ = 𝐷𝑊

write(𝑢)

loc (𝑡). Then

𝐷sel𝑢 (𝑡) ∶= 𝐷𝑊

write(𝑢)

∗,𝜀(𝑢)𝑡,𝑒sig⟩ = 𝐷sel𝑢 (𝑡)⟨𝑢𝑡,𝑒sig⟩, so the signal transport is exact and diagonal.

⟨𝑆𝑇,𝜏

The coeﬀicient 𝐷sel𝑢 (𝑡) depends only on the 𝑒pos-, 𝑒aux-, and 𝐸carry-channels of the intermediate state 𝑊write(𝑢). The writer preserves 𝑒pos and 𝐸carry exactly, and its written auxiliary channel 𝑎𝑡(𝑢) is itself a deterministic function of the positional-control coordinate only. Hence 𝐷sel𝑢 (𝑡) depends only on the original 𝑒pos- and 𝐸carry-channels, not on the signal channel. Thus the transport is signal-blind over 𝐸ctrl.

The 𝑒pos-channel and all of 𝐸carry are preserved exactly by both blocks, hence by the composition. Finally, at the selected source,

𝜀 4

𝜀 4

𝜀 2

|𝐷sel𝑢 (𝜏∗) − 1| ≤ |𝐷sel𝑢 (𝜏∗) − 𝑎𝜏

(𝑢)| + |𝑎𝜏

(𝑢) − 1| ≤

+

=

,

∗

∗

so since 𝜀 < 1,

1 2

3 2

< 2. For 𝑡 ≠ 𝜏∗,

≤ 𝐷sel𝑢 (𝜏∗) ≤

𝜀 4

𝜀 2

𝜀 4

+

=

≤ 𝜀.

|𝐷sel𝑢 (𝑡)| ≤ |𝐷sel𝑢 (𝑡) − 𝑎𝑡(𝑢)| + |𝑎𝑡(𝑢)| ≤

For any 𝜂 > 0, replacing 𝒦_set by Satsig𝜂 (𝒦_set) leaves the ordered positional ranges (𝐼𝑡)𝑇𝑡=0 unchanged. Moreover, in the concrete two-block construction, the writer depends only on the positional coordinate and preserves the signal channel exactly, while the local multiplier depends only on the positional and auxiliary channels and acts diagonally on the signal channel. Hence the same concrete construction yields the same exact diagonal signal-transport formula on Satsig𝜂 (𝒦_set), with the same coeﬀicients 𝐷sel𝑢 (𝑖). Applying Lemma K.8(i) gives

𝜕𝑆𝑇,𝜏

∗,𝜀(𝑢)𝑖 𝜕𝑢𝑗

𝑒sig = 𝐷sel𝑢 (𝑖)1[𝑖 = 𝑗].

𝑒⊤sig

| |
|---|

Remark K.13 (The selector depends only on position). In the concrete construction used in the proof of Lemma K.12,

the diagonal transport coeﬀicient 𝐷sel𝑢 (𝑡) depends only on the positional stream

(⟨𝑢𝑠,𝑒pos⟩)𝑇𝑠=0, and is independent of the signal channel 𝑒sig and of the carried channels 𝐸carry.

- Lemma K.14 (Selector preserves signal fibers). Under the hypotheses of Lemma K.12, let

𝑆𝑇,𝜏

∗,𝜀 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1 be the selector block constructed there. Then for every 𝛿 ≥ 0 there exists 𝛿′ = 𝛿′(𝛿,𝒦_set) < ∞ such that

𝑆𝑇,𝜏

∗,𝜀(Satsig𝛿 (𝒦_set)) ⊂ Satsig𝛿′ (𝑆𝑇,𝜏

∗,𝜀(𝒦_set)). More precisely, if

𝑢′ = 𝑢 +

𝑇

∑

𝑡=0

𝑎𝑡𝑒sig1[⋅ = 𝑡], 𝑢 ∈ 𝒦_set, max

𝑡

|𝑎𝑡| ≤ 𝛿, then

𝑆𝑇,𝜏

∗,𝜀(𝑢′)𝑖 = 𝑆𝑇,𝜏

∗,𝜀(𝑢)𝑖 + 𝐷sel𝑢 (𝑖)𝑎𝑖 𝑒sig, 0 ≤ 𝑖 ≤ 𝑇, where 𝐷sel𝑢 (𝑖) is the selector transport coeﬀicient from Lemma K.12. In particular, one may take 𝛿′ ∶= 𝛿 sup

𝑢∈𝒦_set

sup

0≤𝑖≤𝑇

|𝐷sel𝑢 (𝑖)| ≤ 2𝛿.

Proof. Fix 𝑢 ∈ 𝒦_set and

𝑢′ = 𝑢 +

𝑇

∑

𝑡=0

𝑎𝑡𝑒sig1[⋅ = 𝑡]

with max𝑡 |𝑎𝑡| ≤ 𝛿. By Remark K.13, the coeﬀicient 𝐷sel𝑢 (𝑖) depends only on the positional stream

(⟨𝑢𝑠,𝑒pos⟩)𝑇𝑠=0, which is unchanged under perturbations along 𝑒sig. Moreover, in the concrete construction of 𝑆𝑇,𝜏

∗,𝜀, all non-signal output channels are independent of the input signal channel: the writer 𝑊𝑇,𝜏write

∗,𝜀wr preserves 𝑒sig exactly and writes only the auxiliary channel as a function of the positional coordinate, while 𝑀𝑇,𝛿loc

mul

preserves the positional and

auxiliary channels exactly and modifies the output only on the signal channel. Therefore

𝑆𝑇,𝜏

∗,𝜀(𝑢′)𝑖 = 𝑆𝑇,𝜏

∗,𝜀(𝑢)𝑖 + 𝐷sel𝑢 (𝑖)𝑎𝑖 𝑒sig, and the claim follows.

| |
|---|

- Lemma K.15 (Active diffusive transport). Fix 𝛽 ∈ (0,1) and set 𝛾 ∶= 1−𝛽. Let 𝑇 ≥ 0 and let 𝒦_set ⊂ (ℝ𝑚)𝑇+1 be compact. Assume that for some orthonormal directions

𝑒pos, 𝑒sig, 𝑒src, 𝑒tgt ∈ ℝ𝑚 the scalar position ranges

𝐼𝑡 ∶= {⟨𝑢𝑡,𝑒pos⟩ ∶ 𝑢 ∈ 𝒦_set}, 0 ≤ 𝑡 ≤ 𝑇, are compact and strictly ordered:

𝐼0 < 𝐼1 < ⋯ < 𝐼𝑇 ⊂ (0,∞). Let 𝐸carry ⊂ ℝ𝑚 be any fixed subspace orthogonal to 𝑒pos,𝑒sig,𝑒src,𝑒tgt.

Then there exists a depth-2 LN-free Sessa network

𝐴act𝑇,𝛽 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1

such that the first constituent block has the feedback branch switched off, while the second constituent block uses a strict-past uniform feedback solve with constant gain 𝛾, the 𝑒pos-channel and every channel in 𝐸carry are preserved exactly, and 𝐴act𝑇,𝛽 has signal-blind exact scalar transport along 𝑒sig over

𝐸ctrl ∶= span{𝑒pos} ⊕ 𝐸carry, with kernel

𝒯𝑢𝐴act(𝑖,𝑗) = 𝐷act𝑢 (𝑖)1[𝑖 = 𝑗] + 𝐾act𝑢 (𝑖,𝑗)1[𝑗 < 𝑖]; There exist constants

0 < 𝑑act ≤ 𝑑act < ∞, 0 < 𝑎−act ≤ 𝑎+act < ∞, depending only on 𝛽, but independent of 𝑇, such that

𝑑act ≤ 𝐷act𝑢 (𝑖) ≤ 𝑑act, 0 ≤ 𝑖 ≤ 𝑇, and

𝑎−act(𝑗 + 1)−𝛾(𝑖 + 1)−𝛽 ≤ 𝐾act𝑢 (𝑖,𝑗) ≤ 𝑎+act(𝑗 + 1)−𝛾(𝑖 + 1)−𝛽, 0 ≤ 𝑗 < 𝑖 ≤ 𝑇. In particular,

𝜕𝐴act𝑇,𝛽(𝑢)𝑖 𝜕𝑢𝑗

𝑒sig = 𝐷act𝑢 (𝑖)1[𝑖 = 𝑗] + 𝐾act𝑢 (𝑖,𝑗)1[𝑗 < 𝑖].

𝑒⊤sig

𝐴act𝑇,𝛽 = 𝑅𝑇,𝛽 ∘ 𝐶𝑇, where 𝐶𝑇 is a forward-only copy block and 𝑅𝑇,𝛽 is a single feedback-transport block.

Proof. We construct

𝐶𝑇 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1 such that

- Step 1: copy of the signal into a scratch source channel. Build a forward-only LN-free Sessa block

⟨𝐶𝑇(𝑢)𝑡,𝑒src⟩ = ⟨𝑢𝑡,𝑒sig⟩ (0 ≤ 𝑡 ≤ 𝑇), while the 𝑒pos-, 𝑒sig-, 𝑒tgt-, and 𝐸carry-channels are preserved exactly. Switch off the feedback branch and choose two forward value coordinates equal to 1:

𝑣𝑡(0) ≡ 1, 𝑣𝑡(1) ≡ 1. Hence

𝑠(0)𝑡 = 1, 𝑠(1)𝑡 = 1. Choose the associated gate coordinates

𝑔𝑡(0) = ⟨𝑢𝑡,𝑒src⟩, 𝑔𝑡(1) = ⟨𝑢𝑡,𝑒sig⟩, and choose the output projection on the 𝑒src-channel with coeﬀicients (−1,+1). Then ⟨𝐶𝑇(𝑢)𝑡,𝑒src⟩ = ⟨𝑢𝑡,𝑒src⟩ − ⟨𝑢𝑡,𝑒src⟩ + ⟨𝑢𝑡,𝑒sig⟩ = ⟨𝑢𝑡,𝑒sig⟩.

𝑤 ∶= 𝐶𝑇(𝑢), 𝑥𝑗 ∶= ⟨𝑢𝑗,𝑒sig⟩. Then

Let

⟨𝑤𝑗,𝑒src⟩ = 𝑥𝑗, ⟨𝑤𝑗,𝑒sig⟩ = 𝑥𝑗. (79)

𝑅𝑇,𝛽 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1.

- Step 2: the feedback-transport block. Now build a single LN-free Sessa block

On one dedicated feedback channel, choose all feedback queries and keys identically zero. Then the strict-past feedback softmax is exactly uniform:

1 𝑖

𝛼𝑏𝑖,𝑗 =

, 0 ≤ 𝑗 < 𝑖, 1 ≤ 𝑖 ≤ 𝑇.

𝛾𝑖 ≡ 𝛾 = 1 − 𝛽. Hence the scalar feedback matrix on that channel is

Choose the feedback gain to be the constant

𝛾 𝑖

𝐵𝑖,𝑗 =

1[𝑗 < 𝑖].

For the forward branch, fix 𝜇𝑇 ∈ (0, 12], to be chosen below, and apply Lemma K.2 to the image 𝐶𝑇(𝒦_set) on the ordered positional-control coordinate. Because 𝐶𝑇 preserves the 𝑒pos-channel exactly, the hypotheses still hold. This yields weights 𝛼𝑓𝑖,𝑗(𝑤) satisfying

𝑖−1

𝛼𝑓𝑖,𝑖(𝑤) ≥ 1 − 𝜇𝑇,

∑

𝛼𝑓𝑖,𝑗(𝑤) ≤ 𝜇𝑇, 0 ≤ 𝑖 ≤ 𝑇. (80)

𝑗=0

In particular, for every 𝑗 < 𝑖,

𝛼𝑓𝑖,𝑗(𝑤) ≤ 𝜇𝑇. (81)

To read the source scratch channel exactly, use Corollary K.5 on the input 𝑤 and the direction 𝑒src: choose two 𝑎-slots

𝑎(+)𝑗 = 𝐿⟨𝑤𝑗,𝑒src⟩, 𝑎(−)𝑗 = −𝐿⟨𝑤𝑗,𝑒src⟩. Choose 𝑊𝑉 so that one forward value coordinate is

1 𝐿

(𝑎̄(+)𝑗 − 𝑎̄(−)𝑗 ) = ⟨𝑤𝑗,𝑒src⟩ = 𝑥𝑗. Let

𝑣𝑗src =

𝛼𝑓𝑖,𝑗(𝑤)𝑥𝑗

𝛼𝑓𝑖,𝑗(𝑤)𝑣𝑗src = ∑ 𝑗≤𝑖

𝑓𝑖 ∶= ∑ 𝑗≤𝑖

be the forward signal entering the scalar feedback solve, and let 𝑠𝑖 denote the corresponding solve output:

𝛾 𝑖

𝑠0 = 𝑓0, 𝑠𝑖 = 𝑓𝑖 +

∑

𝑠𝑗, 1 ≤ 𝑖 ≤ 𝑇.

𝑗<𝑖

Choose the gate on that transport coordinate to be the constant 1, and choose the output projection so that the

signal channel receives exactly +𝑠𝑖, while the 𝑒pos- and 𝐸carry-channels are untouched. Therefore ⟨𝑅𝑇,𝛽(𝑤)𝑖,𝑒sig⟩ = ⟨𝑤𝑖,𝑒sig⟩ + 𝑠𝑖 = 𝑥𝑖 + 𝑠𝑖.

Θ𝑖,𝑗 ∶= [(𝐼 − 𝐵)−1]𝑖,𝑗, 0 ≤ 𝑗 ≤ 𝑖 ≤ 𝑇. Then Θ𝑖,𝑖 = 1, and for 𝑗 < 𝑖,

##### Step 3: resolvent kernel. Let

𝛾 𝑖

𝑖−1

Θ𝑖,𝑗 =

∑

Θ𝑟,𝑗.

𝑟=𝑗

𝑖

𝑆𝑖(𝑗) ∶=

∑

Θ𝑟,𝑗.

As in the original proof, define

𝑟=𝑗

Then 𝑆𝑗(𝑗) = 1 and

𝛾 𝑖

)𝑆𝑖−1(𝑗) , hence

𝑆𝑖(𝑗) = (1 +

Γ(𝑖 + 1 + 𝛾)Γ(𝑗 + 1) Γ(𝑗 + 1 + 𝛾)Γ(𝑖 + 1)

𝑆𝑖(𝑗) =

. Therefore, for 𝑗 < 𝑖,

𝛾 𝑖

Γ(𝑖 + 𝛾) Γ(𝑖 + 1)

Γ(𝑗 + 1) Γ(𝑗 + 1 + 𝛾)

Θ𝑖,𝑗 =

. Since 𝛾 ∈ (0,1), standard Gamma-ratio bounds yield constants

𝑆𝑖−1(𝑗) = 𝛾

0 < 𝑐Θ− ≤ 𝑐Θ+ < ∞ depending only on 𝛽, such that

𝑐Θ−(𝑗 + 1)−𝛾(𝑖 + 1)−𝛽 ≤ Θ𝑖,𝑗 ≤ 𝑐Θ+(𝑗 + 1)−𝛾(𝑖 + 1)−𝛽, 0 ≤ 𝑗 < 𝑖 ≤ 𝑇. (82)

Also, since 𝛾 = 1 − 𝛽 ∈ (0,1),

𝑛

∑

𝑟−𝛾 ≲𝛽 𝑛𝛽.

𝑟=1

Combining this with (82), there exists a constant 𝐶Σ < ∞, depending only on 𝛽, such that

𝑖

Θ𝑖,𝑘 ≤ 𝐶Σ (0 ≤ 𝑗 < 𝑖 ≤ 𝑇). (83)

∑

𝑘=𝑗+1

Finally, since 𝑗 + 1 ≤ 𝑖 + 1 ≤ 𝑇 + 1,

𝑐Θ− 𝑇 + 1

Θ𝑖,𝑗 ≥ 𝑐Θ−(𝑖 + 1)−1 ≥

. (84)

##### Step 4: transport formula. Since 𝑠 = Θ𝑓,

𝑖

𝑖

𝑘

𝑖

𝑖

𝑠𝑖 =

∑

Θ𝑖,𝑘𝑓𝑘 =

∑

Θ𝑖,𝑘

∑

𝛼𝑓𝑘,𝑗(𝑤)𝑥𝑗 =

∑

(

∑

###### Θ𝑖,𝑘𝛼𝑓𝑘,𝑗(𝑤))𝑥𝑗.

𝑗=0

𝑗=0

𝑘=0

𝑘=0

𝑘=𝑗

𝑖

⟨𝐴act𝑇,𝛽(𝑢)𝑖,𝑒sig⟩ = 𝑥𝑖 + 𝑠𝑖 = (1 + 𝛼𝑓𝑖,𝑖(𝑤))𝑥𝑖 + ∑ 𝑗<𝑖

(

∑

Θ𝑖,𝑘𝛼𝑓𝑘,𝑗(𝑤))𝑥𝑗.

Therefore

𝑘=𝑗

𝑖

∑

𝐷act𝑢 (𝑖) ∶= 1 + 𝛼𝑓𝑖,𝑖(𝑤), 𝐾act𝑢 (𝑖,𝑗) ∶=

Θ𝑖,𝑘𝛼𝑓𝑘,𝑗(𝑤) (𝑗 < 𝑖).

Define

𝑘=𝑗

⟨𝐴act𝑇,𝛽(𝑢)𝑖,𝑒sig⟩ = 𝐷act𝑢 (𝑖)𝑥𝑖 + ∑ 𝑗<𝑖

𝐾act𝑢 (𝑖,𝑗)𝑥𝑗.

Then

This is exact scalar transport. The coeﬀicients depend only on the positional stream of 𝑤, because the forward weights 𝛼𝑓 were built from the positional-control coordinate only; and 𝐶𝑇 preserves the positional coordinate exactly, so this is the same as the positional stream of 𝑢. The 𝑒pos- and 𝐸carry-channels are preserved exactly by construction. Thus the transport is signal-blind over 𝐸ctrl.

- 1 − 𝜇𝑇 ≤ 𝛼𝑓𝑖,𝑖(𝑤) ≤ 1,

so

- 2 − 𝜇𝑇 ≤ 𝐷act𝑢 (𝑖) ≤ 2.

- Step 5: kernel bounds. From (80),

Since 𝜇𝑇 ≤ 12,

3 2

≤ 𝐷act𝑢 (𝑖) ≤ 2. Thus we may take

3 2

𝑑act ∶=

, 𝑑act ∶= 2.

For the off-diagonal coeﬀicient, all summands are nonnegative. Hence for 𝑗 < 𝑖,

- 1

- 2

𝐾act𝑢 (𝑖,𝑗) ≥ Θ𝑖,𝑗𝛼𝑓𝑗,𝑗(𝑤) ≥ (1 − 𝜇𝑇)Θ𝑖,𝑗 ≥

Θ𝑖,𝑗. Combining with (82) gives

- 1

- 2

𝐾act𝑢 (𝑖,𝑗) ≥

𝑐Θ−(𝑗 + 1)−𝛾(𝑖 + 1)−𝛽. For the upper bound,

𝑖

𝑖

𝐾act𝑢 (𝑖,𝑗) = Θ𝑖,𝑗𝛼𝑓𝑗,𝑗(𝑤) +

∑

Θ𝑖,𝑘𝛼𝑓𝑘,𝑗(𝑤) ≤ Θ𝑖,𝑗 + 𝜇𝑇

∑

Θ𝑖,𝑘,

𝑘=𝑗+1

𝑘=𝑗+1

𝑐Θ− 4𝐶Σ(𝑇 + 1)

- 1

- 2

}. Then by (83),

𝜇𝑇 ∶= min{

,

by (81). Now choose

𝑐Θ− 4(𝑇 + 1)

𝑖

𝜇𝑇

∑

Θ𝑖,𝑘 ≤

.

𝑘=𝑗+1

𝑐Θ− 4(𝑇 + 1)

1 4

≤

Θ𝑖,𝑗.

By (84),

5 4

Θ𝑖,𝑗. Using (82),

𝐾act𝑢 (𝑖,𝑗) ≤

Hence

5 4

𝑐Θ+(𝑗 + 1)−𝛾(𝑖 + 1)−𝛽. Thus the stated two-sided bounds hold with

𝐾act𝑢 (𝑖,𝑗) ≤

- 1

- 2

5 4

𝑎−act ∶=

𝑐Θ−, 𝑎+act ∶=

𝑐Θ+.

For any 𝜂 > 0, replacing 𝒦_set by Satsig𝜂 (𝒦_set) leaves the ordered positional ranges (𝐼𝑡)𝑇𝑡=0 unchanged. In the concrete construction, the copy block writes the source scratch channel from the signal channel exactly and is

independent of the incoming 𝑒src-channel, while the transport block uses forward and feedback weights depending only on the positional stream and reads the copied source scratch channel exactly. Hence the same concrete

construction yields the same exact scalar transport formula on Satsig𝜂 (𝒦_set), with the same coeﬀicients 𝐷act𝑢 (𝑖) and 𝐾act𝑢 (𝑖,𝑗). Applying Lemma K.8(i) gives

𝜕𝐴act𝑇,𝛽(𝑢)𝑖 𝜕𝑢𝑗

𝑒⊤sig

𝑒sig = 𝐷act𝑢 (𝑖)1[𝑖 = 𝑗] + 𝐾act𝑢 (𝑖,𝑗)1[𝑗 < 𝑖].

| |
|---|

Remark K.16 (Active diffusive transport depends only on position). In the concrete construction used in the proof of Lemma K.15, the coeﬀicients

𝐷act𝑢 (𝑖), 𝐾act𝑢 (𝑖,𝑗), 0 ≤ 𝑗 < 𝑖 ≤ 𝑇, depend only on the positional stream

(⟨𝑢𝑠,𝑒pos⟩)𝑇𝑠=0, and are independent of the signal channel 𝑒sig and of the carried channels 𝐸carry.

- Lemma K.17 (Transparent source-0 tail channel). Fix 𝛽 ∈ (0,1), set 𝛾 ∶= 1 − 𝛽, fix 𝜏max ≥ 0, and let 𝐿𝐻 ∶= 𝜏max + 𝐻.

Let 𝒦_set𝐻 ⊂ (ℝ𝑚)𝐿𝐻+1 be compact. Assume orthonormal directions

𝑒sig, 𝑒pos, 𝑒tail, 𝑒aux, 𝑒src, 𝑒tgt ∈ ℝ𝑚 and a subspace 𝐸carry ⊂ ℝ𝑚 orthogonal to all six, such that

𝐼𝑡 ∶= {⟨𝑢𝑡,𝑒pos⟩ ∶ 𝑢 ∈ 𝒦_set𝐻}, 0 ≤ 𝑡 ≤ 𝐿𝐻, are compact and strictly ordered:

𝐼0 < 𝐼1 < ⋯ < 𝐼𝐿

⊂ (0,∞).

𝐻

𝑇𝐻tail ∶ (ℝ𝑚)𝐿𝐻+1 → (ℝ𝑚)𝐿𝐻+1

Then there exists a constant-depth LN-free Sessa network

such that the 𝑒sig-channel, the positional-control coordinate 𝑒pos, and every channel in 𝐸carry are preserved exactly

𝑔𝑡(𝑢) ∶= ⟨𝑇𝐻tail(𝑢)𝑡, 𝑒tail⟩, 0 ≤ 𝑡 ≤ 𝐿𝐻, there exist constants 𝑐𝑔−,𝑐𝑔+ > 0, independent of 𝐻, such that

and, writing

𝑐𝑔−(𝑡 + 1)−𝛽 ≤ 𝑔𝑡(𝑢) ≤ 𝑐𝑔+(𝑡 + 1)−𝛽, 0 ≤ 𝑡 ≤ 𝐿𝐻, 𝑢 ∈ 𝒦_set𝐻; 𝑇𝐻tail is signal-transparent along 𝑒sig with respect to the control pair

(𝑒pos,𝑒tail) ∶ for every 𝑢 ∈ 𝒦_set𝐻, every 𝜏 ∈ {0,…,𝐿𝐻}, and every scalar 𝑎 ∈ ℝ,

tail 𝐻 sig1 𝑡, 𝑒pos 𝐻tail 𝑡, 𝑒pos

⟨𝑇𝐻tail(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡, 𝑒tail⟩ = ⟨𝑇𝐻tail(𝑢)𝑡, 𝑒tail⟩, ⟨𝑇𝐻tail(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡, 𝑒sig⟩ = ⟨𝑇𝐻tail(𝑢)𝑡, 𝑒sig⟩ + 𝑎1[𝑡 = 𝜏], 0 ≤ 𝑡 ≤ 𝐿𝐻.

Proof. All auxiliary directions used below are part of the hypotheses; no fresh direction is chosen inside the construction. We construct

𝑇𝐻tail = 𝐴tail𝐻 ∘ 𝑆𝐻tail ∘ 𝐶𝐻,

where 𝐶𝐻 writes a constant seed on the prescribed tail direction 𝑒tail, 𝑆𝐻tail selects source 0 on that tail channel, and 𝐴tail𝐻 transports the selected seed by the active diffusive block.

𝐶𝐻 ∶ (ℝ𝑚)𝐿𝐻+1 → (ℝ𝑚)𝐿𝐻+1

- Step 1: constant seed writer on the prescribed tail direction. Build a forward-only LN-free Sessa block

as follows. Choose two forward value coordinates equal to 1:

𝑣𝑡(0) ≡ 1, 𝑣𝑡(1) ≡ 1. Hence the corresponding forward aggregates satisfy

𝑠(0)𝑡 = 1, 𝑠(1)𝑡 = 1. Choose two gate coordinates

𝑔𝑡(0) = ⟨𝑢𝑡,𝑒tail⟩, 𝑔𝑡(1) ≡ 1,

and choose the output projection on the 𝑒tail-channel with coeﬀicients (−1,+1) on these two gated coordinates and zero on all other output channels. Then

⟨𝐶𝐻(𝑢)𝑡,𝑒tail⟩ = ⟨𝑢𝑡,𝑒tail⟩ − 𝑠(0)𝑡 ⟨𝑢𝑡,𝑒tail⟩ + 𝑠(1)𝑡 = 1. Thus 𝐶𝐻 overwrites the 𝑒tail-channel by the constant seed 1. Because the output projection vanishes on the 𝑒sig-, 𝑒pos-, and 𝐸carry-channels, these channels are preserved exactly: ⟨𝐶𝐻(𝑢)𝑡,𝑒sig⟩ = ⟨𝑢𝑡,𝑒sig⟩, ⟨𝐶𝐻(𝑢)𝑡,𝑒pos⟩ = ⟨𝑢𝑡,𝑒pos⟩, and likewise on 𝐸carry.

Moreover, since the written tail seed is constant and independent of the input, for every 𝑎 ∈ ℝ,

⟨𝐶𝐻(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡,𝑒tail⟩ = ⟨𝐶𝐻(𝑢)𝑡,𝑒tail⟩ = 1,

while the 𝑒sig-channel passes through exactly. So 𝐶𝐻 is already signal-transparent along 𝑒sig with respect to (𝑒pos,𝑒tail).

𝒦_set(1)𝐻 ∶= 𝐶𝐻(𝒦_set𝐻).

##### Step 2: positional selector on the tail channel. Let

Apply Lemma K.12 to 𝒦_set(1)𝐻 with signal direction 𝑒selsig ∶= 𝑒tail, positional-control direction 𝑒pos, auxiliary direction 𝑒aux, source index 𝜏∗ = 0, and carried-through subspace

𝐸carrysel ∶= span{𝑒sig} ⊕ 𝐸carry.

Choose an exponent 𝑀 > 𝛽 and set

𝜀𝐻 ∶= 𝑐0(𝐻 + 1)−𝑀, where 𝑐0 > 0 will be chosen later. The lemma yields a depth-2 network 𝑆𝐻tail ∶= 𝑆𝐿

𝐻,0,𝜀𝐻

which preserves 𝑒pos, the original 𝑒sig, and every channel in 𝐸carry exactly, and whose exact diagonal transport on the tail channel is

⟨𝑆𝐻tail(𝑣)𝑡,𝑒tail⟩ = 𝐷sel𝑣 (𝑡)⟨𝑣𝑡,𝑒tail⟩. Since ⟨𝐶𝐻(𝑢)𝑡,𝑒tail⟩ ≡ 1, the selected seed stream is

sel (𝑡). By Lemma K.12,

𝑧𝑡(𝑢) ∶= ⟨𝑆𝐻tail(𝐶𝐻(𝑢))𝑡,𝑒tail⟩ = 𝐷𝐶

𝐻(𝑢)

- 1

- 2

≤ 𝑧0(𝑢) ≤ 2, |𝑧𝑡(𝑢)| ≤ 𝜀𝐻 (𝑡 ≥ 1).

By Remark K.13, in the concrete construction of 𝑆𝐻tail = 𝑆𝐿

𝐻,0,𝜀𝐻 the coeﬀicient 𝐷𝐶

sel (𝑡) depends only on the positional stream

𝐻(𝑢)

(⟨𝐶𝐻(𝑢)𝑠,𝑒pos⟩)𝐿𝑠=0𝐻 . Since 𝐶𝐻 preserves the positional coordinate exactly,

⟨𝐶𝐻(𝑢)𝑠,𝑒pos⟩ = ⟨𝑢𝑠,𝑒pos⟩, 0 ≤ 𝑠 ≤ 𝐿𝐻, it follows that

sel (𝑡) depends only on the original positional stream and not on the original signal channel.

𝑧𝑡(𝑢) = 𝐷𝐶

𝐻(𝑢)

𝒦_set(2)𝐻 ∶= 𝑆𝐻tail(𝒦_set(1)𝐻 ).

##### Step 3: active diffusive transport on the same prescribed tail direction. Let

Apply Lemma K.15 to 𝒦_set(2)𝐻 with positional direction 𝑒pos, signal direction 𝑒actsig ∶= 𝑒tail, scratch directions 𝑒src,𝑒tgt, and carried-through subspace

𝐸carryact ∶= span{𝑒sig} ⊕ 𝐸carry. Denote the resulting network by

𝐴tail𝐻 .

By the lemma, 𝐴tail𝐻 preserves 𝑒pos, the original 𝑒sig, and 𝐸carry exactly, and has exact scalar transport on the tail channel:

⟨𝐴tail𝐻 (𝑤)𝑡,𝑒tail⟩ = 𝐷act𝑤 (𝑡)⟨𝑤𝑡,𝑒tail⟩ + ∑ 𝑗<𝑡

𝐾act𝑤 (𝑡,𝑗)⟨𝑤𝑗,𝑒tail⟩.

𝑔𝑡(𝑢) ∶= ⟨𝑇𝐻tail(𝑢)𝑡,𝑒tail⟩, we have

Therefore, for

𝑔𝑡(𝑢) = 𝐷act𝑤 (𝑡)𝑧𝑡(𝑢) + ∑ 𝑗<𝑡

𝐾act𝑤 (𝑡,𝑗)𝑧𝑗(𝑢), 𝑤 ∶= 𝑆𝐻tail(𝐶𝐻(𝑢)).

By Remark K.16, in the concrete construction of 𝐴tail𝐻 the coeﬀicients

𝐷act𝑤 (𝑡), 𝐾act𝑤 (𝑡,𝑗) depend only on the positional stream

(⟨𝑤𝑠,𝑒pos⟩)𝐿𝑠=0𝐻 .

Since both 𝐶𝐻 and 𝑆𝐻tail preserve the positional coordinate exactly, this is the same as the original positional stream of 𝑢. Hence these coeﬀicients are independent of the original signal channel.

##### Step 4: two-sided tail bounds. At 𝑡 = 0, the sum is empty, so

𝑔0(𝑢) = 𝐷act𝑤 (0)𝑧0(𝑢). By Lemma K.15,

𝑑act ≤ 𝐷act𝑤 (0) ≤ 𝑑act, hence

- 1

- 2

𝑑act ≤ 𝑔0(𝑢) ≤ 2𝑑act.

Now fix 𝑡 ≥ 1. Using the exact transport formula, the bounds on 𝑧𝑗(𝑢), and the coeﬀicient bounds from Lemma K.15, we obtain

𝑡−1

𝑔𝑡(𝑢) ≥ 𝐾act𝑤 (𝑡,0)𝑧0(𝑢) − |𝐷act𝑤 (𝑡)𝑧𝑡(𝑢)| −

∑

𝐾act𝑤 (𝑡,𝑗)|𝑧𝑗(𝑢)|

𝑗=1

- 1

- 2

𝑡−1

≥

𝑎−act(𝑡 + 1)−𝛽 − 𝑑act 𝜀𝐻 − 𝑎+act𝜀𝐻

∑

(𝑗 + 1)−𝛾(𝑡 + 1)−𝛽.

𝑗=1

Since 𝛾 = 1 − 𝛽 ∈ (0,1),

𝑡−1

∑

(𝑗 + 1)−𝛾 ≲𝛽 (𝑡 + 1)𝛽,

𝑗=1

𝑔𝑡(𝑢) ≥ 𝑐1(𝑡 + 1)−𝛽 − 𝑐2𝜀𝐻

hence

for constants 𝑐1,𝑐2 > 0 independent of 𝐻. Now 𝑀 > 𝛽, so

𝜀𝐻 = 𝑐0(𝐻 + 1)−𝑀 ≤ 𝑐0(𝐻 + 1)−𝛽. Also 0 ≤ 𝑡 ≤ 𝐿𝐻 = 𝜏max + 𝐻, hence

(𝐻 + 1)−𝛽 ≤ (𝜏max + 1)𝛽(𝑡 + 1)−𝛽. Therefore

𝑐0(𝑡 + 1)−𝛽. Choosing 𝑐0 > 0 suﬀiciently small makes the error absorbable, so

𝜀𝐻 ≲𝜏

max

𝑔𝑡(𝑢) ≥ 𝑐𝑔−(𝑡 + 1)−𝛽

for some 𝑐𝑔− > 0 independent of 𝐻. Similarly,

𝑡−1

∑

𝑔𝑡(𝑢) ≤ |𝐷act𝑤 (𝑡)𝑧𝑡(𝑢)| + 𝐾act𝑤 (𝑡,0)|𝑧0(𝑢)| +

𝐾act𝑤 (𝑡,𝑗)|𝑧𝑗(𝑢)|

𝑗=1

𝑡−1

∑

≤ 𝑑act𝜀𝐻 + 2𝑎act+ (𝑡 + 1)−𝛽 + 𝑎+act𝜀𝐻

(𝑗 + 1)−𝛾(𝑡 + 1)−𝛽,

𝑗=1

𝑔𝑡(𝑢) ≤ 𝑐𝑔+(𝑡 + 1)−𝛽

hence

for some 𝑐𝑔+ < ∞ independent of 𝐻. Thus

𝑐𝑔−(𝑡 + 1)−𝛽 ≤ 𝑔𝑡(𝑢) ≤ 𝑐𝑔+(𝑡 + 1)−𝛽, 0 ≤ 𝑡 ≤ 𝐿𝐻.

- Step 5: signal-transparency along 𝑒sig. Let 𝑢(𝑎,𝜏) ∶= 𝑢 + 𝑎𝑒sig1[⋅ = 𝜏].

Since 𝑒sig ⟂ 𝑒pos, we have

⟨𝑢(𝑎,𝜏)𝑡 ,𝑒pos⟩ = ⟨𝑢𝑡,𝑒pos⟩ ∀𝑡. By Step 1,

⟨𝐶𝐻(𝑢(𝑎,𝜏))𝑡,𝑒tail⟩ = ⟨𝐶𝐻(𝑢)𝑡,𝑒tail⟩ = 1, and

⟨𝐶𝐻(𝑢(𝑎,𝜏))𝑡,𝑒sig⟩ = ⟨𝐶𝐻(𝑢)𝑡,𝑒sig⟩ + 𝑎1[𝑡 = 𝜏].

- By the dependence analysis in Step 2, 𝑧𝑡(𝑢) depends only on the positional stream, so 𝑧𝑡(𝑢(𝑎,𝜏)) = 𝑧𝑡(𝑢).
- By the dependence analysis in Step 3, the coeﬀicients 𝐷act𝑤 ,𝐾act𝑤 also depend only on the positional stream, hence they are unchanged under the perturbation. Therefore the tail output is unchanged:

𝑔𝑡(𝑢(𝑎,𝜏)) = 𝑔𝑡(𝑢).

Since each constituent block preserves the original 𝑒sig-channel exactly, the full composition satisfies

⟨𝑇𝐻tail(𝑢(𝑎,𝜏))𝑡,𝑒sig⟩ = ⟨𝑇𝐻tail(𝑢)𝑡,𝑒sig⟩ + 𝑎1[𝑡 = 𝜏]. The 𝑒pos-coordinate is preserved exactly at each stage as well. This proves signal-transparency.

- Lemma K.18 (Residual zero-writer). Fix 𝑇 ≥ 0, a compact set 𝒦_set ⊂ (ℝ𝑚)𝑇+1, orthonormal directions

| |
|---|

𝑒sig, 𝑒pos, 𝑒zero ∈ ℝ𝑚, and a subspace 𝐸carry ⊂ ℝ𝑚 orthogonal to all three. Then there exists a single LN-free Sessa block 𝑍𝑇,𝑒

∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1

such that the feedback branch is switched off, the 𝑒sig-channel, the 𝑒pos-channel, and every channel in 𝐸carry are preserved exactly, the prescribed channel is written to zero exactly:

zero

⟨𝑍𝑇,𝑒

(𝑢)𝑡, 𝑒zero⟩ = 0 ∀𝑢 ∈ 𝒦_set, ∀0 ≤ 𝑡 ≤ 𝑇; 𝑍𝑇,𝑒

is signal-transparent along 𝑒sig with respect to the control pair (𝑒pos,𝑒zero): for every 𝑢 ∈ 𝒦_set, every 𝜏 ∈ {0,…,𝑇}, every scalar 𝑎 ∈ ℝ, and every 0 ≤ 𝑡 ≤ 𝑇,

zero

zero

(𝑢)𝑡, 𝑒pos⟩, ⟨𝑍𝑇,𝑒

⟨𝑍𝑇,𝑒

(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡, 𝑒pos⟩ = ⟨𝑍𝑇,𝑒

(𝑢)𝑡, 𝑒zero⟩ = 0, and

(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡, 𝑒zero⟩ = ⟨𝑍𝑇,𝑒

zero

zero

⟨𝑍𝑇,𝑒

(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡, 𝑒sig⟩ = ⟨𝑍𝑇,𝑒

(𝑢)𝑡, 𝑒sig⟩ + 𝑎1[𝑡 = 𝜏].

zero

zero

zero

zero

Proof. Switch off the feedback branch. Choose a positive constant 𝑐1 such that

GELU(𝑐1) = 1. Realize one forward value coordinate by the constant 1:

𝑣𝑡(0) ≡ 1. Since every forward attention row sums to 1, the corresponding forward aggregate is

𝑠(0)𝑡 = ∑ 𝑗≤𝑡

𝛼𝑓𝑡,𝑗 ⋅ 1 = 1 (0 ≤ 𝑡 ≤ 𝑇).

𝑔𝑡(0) = ⟨𝑢𝑡,𝑒zero⟩. Choose the output projection so that this gated coordinate contributes −𝑒zero and all other output columns are zero. Then the residual update adds −𝑠(0)𝑡 𝑔𝑡(0) 𝑒zero = −⟨𝑢𝑡,𝑒zero⟩𝑒zero.

Choose one gate coordinate equal to the prescribed channel:

(𝑢)𝑡 = 𝑢𝑡 − ⟨𝑢𝑡,𝑒zero⟩𝑒zero. Taking the 𝑒zero-coordinate gives

𝑍𝑇,𝑒

Therefore

zero

(𝑢)𝑡,𝑒zero⟩ = ⟨𝑢𝑡,𝑒zero⟩ − ⟨𝑢𝑡,𝑒zero⟩ = 0, which proves the exact zero-writing claim. Because the update is supported only on the 𝑒zero-direction, and

⟨𝑍𝑇,𝑒

zero

𝑒sig, 𝑒pos, 𝐸carry ⟂ 𝑒zero,

the 𝑒sig-channel, the 𝑒pos-channel, and all channels in 𝐸carry are preserved exactly. This proves the exact preservation claim.

𝑢(𝑎,𝜏) ∶= 𝑢 + 𝑎𝑒sig1[⋅ = 𝜏]. Since 𝑒sig ⟂ 𝑒zero,𝑒pos, one has

For signal-transparency, let

⟨𝑢(𝑎,𝜏)𝑡 ,𝑒zero⟩ = ⟨𝑢𝑡,𝑒zero⟩, ⟨𝑢(𝑎,𝜏)𝑡 ,𝑒pos⟩ = ⟨𝑢𝑡,𝑒pos⟩. Applying the explicit formula for 𝑍𝑇,𝑒

yields

(𝑢)𝑡 + 𝑎𝑒sig1[𝑡 = 𝜏]. Taking the 𝑒pos-, 𝑒zero-, and 𝑒sig-coordinates gives the stated signal-transparency property.

𝑍𝑇,𝑒

(𝑢(𝑎,𝜏))𝑡 = 𝑢𝑡 + 𝑎𝑒sig1[𝑡 = 𝜏] − ⟨𝑢𝑡,𝑒zero⟩𝑒zero = 𝑍𝑇,𝑒

zero

zero

zero

- Lemma K.19 (Exact reset of finitely many scratch channels). Fix 𝑇 ≥ 0, orthonormal directions

| |
|---|

𝑒sig, 𝑒𝑧,1,…,𝑒𝑧,𝑝 ∈ ℝ𝑚,

and a subspace 𝐸keep ⊂ ℝ𝑚 orthogonal to all of them. Then there exists a single forward-only concrete LN-free Sessa block

𝑧,𝑟} ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1 such that 𝑍𝑇,{𝑒scr

𝑍𝑇,{𝑒scr

𝑧,𝑟} preserves 𝑒sig and every channel in 𝐸keep exactly, and for every 𝑢 and every 𝑡, ⟨𝑍𝑇,{𝑒scr

𝑧,𝑟}(𝑢)𝑡,𝑒𝑧,𝑟⟩ = 0 (𝑟 = 1,…,𝑝); 𝑍𝑇,{𝑒scr

𝑧,𝑟} is signal-transparent along 𝑒sig over 𝐸keep. Proof. Switch off the feedback branch and choose the forward queries and keys identically zero, so that every forward row has sum 1. Choose a positive constant 𝑐∗ with

GELU(𝑐∗) = 1. Activate one constant 𝑎-slot:

𝑎(1)𝑡 ≡ 𝑐∗.

Then one post-GELU coordinate is identically 1. Choose 𝑊𝑉 so that the first 𝑝 forward value coordinates are all equal to 1. Since each forward row sums to 1, the corresponding forward aggregates satisfy

𝑠(𝑟)𝑡 = 1 (𝑟 = 1,…,𝑝).

Choose the first 𝑝 gate coordinates as

𝑔𝑡(𝑟) = ⟨𝑢𝑡,𝑒𝑧,𝑟⟩, 𝑟 = 1,…,𝑝,

and set all remaining gate coordinates to 0. Finally choose 𝑊out so that the 𝑟-th active gated coordinate contributes −𝑒𝑧,𝑟, with all other output columns equal to 0. Then the residual update equals

𝑝

−

∑

⟨𝑢𝑡,𝑒𝑧,𝑟⟩𝑒𝑧,𝑟,

𝑟=1

𝑝

𝑍𝑇,{𝑒scr

𝑧,𝑟}(𝑢)𝑡 = 𝑢𝑡 −

∑

⟨𝑢𝑡,𝑒𝑧,𝑟⟩𝑒𝑧,𝑟.

so

𝑟=1

Hence each scratch channel is reset exactly to zero, while 𝑒sig and every channel in 𝐸keep are preserved exactly. Now let

𝑢(𝑎,𝜏) ∶= 𝑢 + 𝑎𝑒sig1[⋅ = 𝜏]. Because 𝑒sig ⟂ 𝑒𝑧,𝑟 for every 𝑟, the reset term is identical for 𝑢(𝑎,𝜏) and for 𝑢. Therefore 𝑍𝑇,{𝑒scr

𝑧,𝑟}(𝑢)𝑡 + 𝑎𝑒sig1[𝑡 = 𝜏]. This is exactly signal-transparency along 𝑒sig over 𝐸keep.

𝑧,𝑟}(𝑢(𝑎,𝜏))𝑡 = 𝑍𝑇,{𝑒scr

- Lemma K.20 (Transparent damped predecessor integrator). Fix 𝛽 ∈ (0,1), set 𝛾 ∶= 1 − 𝛽, and let

| |
|---|

𝐿𝐻 ∶= 𝜏max + 𝐻. Let 𝒦_set𝐻 ⊂ (ℝ𝑚)𝐿𝐻+1 be compact. Assume orthonormal directions

𝑒sig, 𝑒pos, 𝑒tail, 𝑒prof ∈ ℝ𝑚 and a subspace 𝐸carry ⊂ ℝ𝑚 orthogonal to all four, such that:

- (i) the positional-control ranges

𝐼𝑡 ∶= {⟨𝑢𝑡,𝑒pos⟩ ∶ 𝑢 ∈ 𝒦_set𝐻}, 0 ≤ 𝑡 ≤ 𝐿𝐻, are compact and strictly ordered:

𝐼0 < 𝐼1 < ⋯ < 𝐼𝐿

𝐻

⊂ (0,∞);

- (ii) the auxiliary tail input channel 𝑔𝑡(𝑢) ∶= ⟨𝑢𝑡,𝑒tail⟩

satisfies

𝑐𝑔−(𝑡 + 1)−𝛽 ≤ 𝑔𝑡(𝑢) ≤ 𝑐𝑔+(𝑡 + 1)−𝛽, 0 ≤ 𝑡 ≤ 𝐿𝐻, 𝑢 ∈ 𝒦_set𝐻;

- (iii) the profile input channel is identically zero on 𝒦_set𝐻: ⟨𝑢𝑡,𝑒prof⟩ = 0 ∀𝑢 ∈ 𝒦_set𝐻, ∀0 ≤ 𝑡 ≤ 𝐿𝐻.

𝐼𝐻 ∶ (ℝ𝑚)𝐿𝐻+1 → (ℝ𝑚)𝐿𝐻+1

Then there exists a single LN-free Sessa block

such that the 𝑒sig-channel, the 𝑒pos-coordinate, the 𝑒tail-channel, and every channel in 𝐸carry are preserved exactly

𝑟𝑡(𝑢) ∶= ⟨𝐼𝐻(𝑢)𝑡, 𝑒prof⟩, there exist constants 𝑐𝑟−,𝑐𝑟+ > 0, independent of 𝐻, such that

and, writing

𝑐𝑟−(𝑡 + 1)𝛾 ≤ 𝑟𝑡(𝑢) ≤ 𝑐𝑟+(𝑡 + 1)𝛾, 0 ≤ 𝑡 ≤ 𝐿𝐻, 𝑢 ∈ 𝒦_set𝐻;

𝐼𝐻 is signal-transparent along 𝑒sig with respect to the control pair (𝑒pos,𝑒prof) ∶ for every 𝑢 ∈ 𝒦_set𝐻, every 𝜏 ∈ {0,…,𝐿𝐻}, every scalar 𝑎 ∈ ℝ, and every 0 ≤ 𝑡 ≤ 𝐿𝐻,

⟨𝐼𝐻 sig1 𝑡, 𝑒pos 𝐻 𝑡, 𝑒pos⟩, ⟨𝐼𝐻(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡, 𝑒prof⟩ = ⟨𝐼𝐻(𝑢)𝑡, 𝑒prof⟩, ⟨𝐼𝐻(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡, 𝑒sig⟩ = ⟨𝐼𝐻(𝑢)𝑡, 𝑒sig⟩ + 𝑎1[𝑡 = 𝜏]. Proof. Fix a small constant

0 < 𝜅𝜇 ≤ 1 to be chosen later, and set

1 4(𝐿𝐻 + 1)

𝜆𝐻 ∶= 1 −

∈ (0,1), 𝜇𝐻 ∶= 𝜅𝜇(𝐿𝐻 + 1)−3.

- Step 1: choose the attention patterns. Use Lemma K.1 on the positional-control coordinate 𝑒pos with parameter 𝜇𝐻. This yields strict-past feedback attention satisfying

𝛼𝑏𝑡,𝑡−1 ≥ 1 − 𝜇𝐻,

𝑡−2

∑

𝑗=0

𝛼𝑏𝑡,𝑗 ≤ 𝜇𝐻.

Use Lemma K.2 on the same positional-control coordinate, again with parameter 𝜇𝐻, so that the forward row satisfies

𝛼𝑓𝑡,𝑡 ≥ 1 − 𝜇𝐻, ∑ 𝑗<𝑡

𝛼𝑓𝑡,𝑗 ≤ 𝜇𝐻.

Both 𝛼𝑏 and 𝛼𝑓 depend only on the positional stream.

- Step 2: feed the tail channel into the solve. Read the tail input channel exactly using Corollary K.5. Choose two 𝑎-slots

𝑎(+)𝑡 = 𝐿⟨𝑢𝑡,𝑒tail⟩, 𝑎(−)𝑡 = −𝐿⟨𝑢𝑡,𝑒tail⟩, for any fixed 𝐿 > 0, and choose one dedicated transport value coordinate

1 𝐿

𝑣𝑡tail =

(𝑎̄(+)𝑡 − 𝑎̄(−)𝑡 ) = ⟨𝑢𝑡,𝑒tail⟩ = 𝑔𝑡(𝑢).

𝛾𝑡 ≡ 𝜆𝐻. Let 𝑓𝑡(𝑢) denote the forward signal entering the scalar solve on that dedicated coordinate:

Choose the feedback gain constant

𝛼𝑓𝑡,𝑗(𝑢)𝑔𝑗(𝑢).

𝑓𝑡(𝑢) = ∑ 𝑗≤𝑡

Let 𝑠𝑡(𝑢) be the corresponding solve output:

𝑠0(𝑢) = 𝑓0(𝑢), 𝑠𝑡(𝑢) = 𝑓𝑡(𝑢) + 𝜆𝐻 ∑ 𝑗<𝑡

𝛼𝑏𝑡,𝑗(𝑢)𝑠𝑗(𝑢), 𝑡 ≥ 1.

Choose the gate on that dedicated coordinate to be the constant 1, and choose the output projection so that this solve output is written onto the prescribed profile direction 𝑒prof, with all output columns on

𝑒sig, 𝑒pos, 𝑒tail, 𝐸carry

set to zero. Because the input profile channel is identically zero on 𝒦_set𝐻, the residual formula gives

⟨𝐼𝐻(𝑢)𝑡,𝑒prof⟩ = ⟨𝑢𝑡,𝑒prof⟩ + 𝑠𝑡(𝑢) = 𝑠𝑡(𝑢). Hence

𝑟𝑡(𝑢) ∶= ⟨𝐼𝐻(𝑢)𝑡,𝑒prof⟩ = 𝑠𝑡(𝑢).

The 𝑒sig-, 𝑒pos-, 𝑒tail-, and 𝐸carry-channels are preserved exactly, because the output projection vanishes on those directions.

𝑟̃0(𝑢) ∶= 𝑔0(𝑢), 𝑟̃𝑡(𝑢) ∶= 𝑔𝑡(𝑢) + 𝜆𝐻 𝑟̃𝑡−1(𝑢), 𝑡 ≥ 1, so that

##### Step 3: compare with the ideal predecessor recursion. Define the ideal predecessor recursion

𝑡

∑

𝑟̃𝑡(𝑢) =

𝜆𝐻𝑡−𝑚𝑔𝑚(𝑢).

𝑚=0

Since 0 ≤ 𝑚 ≤ 𝑡 ≤ 𝐿𝐻 and 𝜆𝐻 = 1 − 4(𝐿1

𝐻+1),

𝑒−1/4 ≤ 𝜆𝐻𝑡−𝑚 ≤ 1. Therefore

𝑡

𝑡

𝑔𝑚(𝑢). Using

𝑒−1/4

∑

𝑔𝑚(𝑢) ≤ 𝑟̃𝑡(𝑢) ≤

∑

𝑚=0

𝑚=0

𝑐𝑔−(𝑚 + 1)−𝛽 ≤ 𝑔𝑚(𝑢) ≤ 𝑐𝑔+(𝑚 + 1)−𝛽 and

𝑡

∑

(𝑚 + 1)−𝛽 ≍ (𝑡 + 1)1−𝛽 = (𝑡 + 1)𝛾,

𝑚=0

we obtain constants 𝑐̃𝑟−,𝑐̃𝑟+ > 0, independent of 𝐻, such that 𝑐̃𝑟−(𝑡 + 1)𝛾 ≤ 𝑟̃𝑡(𝑢) ≤ 𝑐̃𝑟+(𝑡 + 1)𝛾.

###### Step 4: control the perturbation error. Let 𝐵𝐻(𝑢) be the actual feedback matrix on the dedicated profile coordinate and 𝐵𝐻∗ the ideal predecessor matrix

(𝐵𝐻∗ )𝑡,𝑡−1 = 𝜆𝐻, (𝐵𝐻∗ )𝑡,𝑗 = 0 (𝑗 < 𝑡 − 1).

∑

|(𝐵𝐻(𝑢) − 𝐵𝐻∗ )𝑡,𝑗| ≤ 𝐶 𝜇𝐻

By the predecessor-focusing estimate,

𝑡

𝑗<𝑡

sup

for an absolute constant 𝐶. Also,

𝑓𝑡(𝑢) − 𝑔𝑡(𝑢) = ∑ 𝑗≤𝑡

𝛼𝑓𝑡,𝑗(𝑢)(𝑔𝑗(𝑢) − 𝑔𝑡(𝑢)) = ∑ 𝑗<𝑡

𝛼𝑓𝑡,𝑗(𝑢)(𝑔𝑗(𝑢) − 𝑔𝑡(𝑢)),

|𝑓𝑡(𝑢) − 𝑔𝑡(𝑢)| ≤ 2𝑐𝑔+ ∑ 𝑗<𝑡

𝛼𝑓𝑡,𝑗(𝑢) ≤ 2𝑐𝑔+𝜇𝐻.

hence

‖𝑓(𝑢) − 𝑔(𝑢)‖∞ ≤ 2𝑐𝑔+𝜇𝐻. Now

Therefore

𝑟(𝑢) = (𝐼 − 𝐵𝐻(𝑢))−1𝑓(𝑢), 𝑟̃(𝑢) = (𝐼 − 𝐵𝐻∗ )−1𝑔(𝑢), so

𝑟(𝑢) − 𝑟̃(𝑢) = (𝐼 − 𝐵𝐻(𝑢))−1((𝑓(𝑢) − 𝑔(𝑢)) + (𝐵𝐻(𝑢) − 𝐵𝐻∗ )𝑟̃(𝑢)). Since the row sum of 𝐵𝐻(𝑢) is at most 𝜆𝐻 < 1,

1 1 − 𝜆𝐻

‖(𝐼 − 𝐵𝐻(𝑢))−1‖∞→∞ ≤

= 4(𝐿𝐻 + 1).

‖𝑟̃(𝑢)‖∞ ≲ (𝐿𝐻 + 1)𝛾. Therefore there exists a constant 𝐶∗ > 0, independent of 𝐻, such that ‖𝑟(𝑢) − 𝑟̃(𝑢)‖∞ ≤ 𝐶∗(𝐿𝐻 + 1)𝛾+1𝜇𝐻 = 𝐶∗𝜅𝜇(𝐿𝐻 + 1)𝛾−2. Since 𝐿𝐻 = 𝜏max + 𝐻 ≥ 𝜏max + 1, we have

Also

(𝐿𝐻 + 1)𝛾−2 ≤ (𝜏max + 2)𝛾−2. Choose 𝜅𝜇 > 0 so small that

1 2

𝑐̃𝑟−. Then uniformly in 𝐻,

𝐶∗𝜅𝜇(𝜏max + 2)𝛾−2 ≤

1 2

‖𝑟(𝑢) − 𝑟̃(𝑢)‖∞ ≤

𝑐̃𝑟−.

Hence for every 0 ≤ 𝑡 ≤ 𝐿𝐻,

- 1

- 2

- 1

- 2

𝑐̃𝑟−. Since (𝑡 + 1)𝛾 ≥ 1,

𝑟𝑡(𝑢) ≥ 𝑟̃𝑡(𝑢) −

𝑐̃𝑟− ≥ 𝑐̃𝑟−(𝑡 + 1)𝛾 −

- 1

- 2

- 1

- 2

𝑐̃𝑟−(𝑡 + 1)𝛾 −

𝑐̃𝑟−(𝑡 + 1)𝛾. So

𝑐̃𝑟− ≥

1 2

𝑟𝑡(𝑢) ≥

𝑐̃𝑟−(𝑡 + 1)𝛾.

- 1

- 2

- 1

- 2

𝑐̃𝑟−. Again using (𝑡 + 1)𝛾 ≥ 1,

𝑟𝑡(𝑢) ≤ 𝑟̃𝑡(𝑢) +

𝑐̃𝑟− ≤ 𝑐̃𝑟+(𝑡 + 1)𝛾 +

Similarly,

- 1

- 2

𝑟𝑡(𝑢) ≤ (𝑐̃𝑟+ +

𝑐̃𝑟−)(𝑡 + 1)𝛾.

- 1

- 2

1 2

Thus the stated two-sided profile bound holds with

𝑐𝑟− ∶=

𝑐̃𝑟−, 𝑐𝑟+ ∶= 𝑐̃𝑟+ +

𝑐̃𝑟−.

𝑢(𝑎,𝜏) ∶= 𝑢 + 𝑎𝑒sig1[⋅ = 𝜏]. Since 𝑒sig ⟂ 𝑒pos,𝑒tail,𝑒prof, one has

- Step 5: verify signal-transparency. Let

⟨𝑢(𝑎,𝜏)𝑡 ,𝑒pos⟩ = ⟨𝑢𝑡,𝑒pos⟩, ⟨𝑢(𝑎,𝜏)𝑡 ,𝑒tail⟩ = ⟨𝑢𝑡,𝑒tail⟩, ⟨𝑢(𝑎,𝜏)𝑡 ,𝑒prof⟩ = ⟨𝑢𝑡,𝑒prof⟩ = 0.

Therefore the feedback weights 𝛼𝑏 are unchanged, since they depend only on the positional stream. The forward weights 𝛼𝑓 are also unchanged for the same reason. Finally, the forward values 𝑔𝑡 are unchanged, since they are exact reads of the tail channel. Hence the actual forward signal 𝑓𝑡, the actual feedback matrix 𝐵𝐻, and therefore the solve output 𝑟𝑡 are all unchanged under perturbations along 𝑒sig:

𝑟𝑡(𝑢(𝑎,𝜏)) = 𝑟𝑡(𝑢). By construction, the output projection vanishes on the 𝑒sig-channel, so that channel passes through exactly: ⟨𝐼𝐻(𝑢(𝑎,𝜏))𝑡,𝑒sig⟩ = ⟨𝐼𝐻(𝑢)𝑡,𝑒sig⟩ + 𝑎1[𝑡 = 𝜏]. The 𝑒pos-coordinate is preserved exactly as well. This proves the stated signal-transparency property.

Corollary K.21 (Transparent power-profile block). Fix 𝛽 ∈ (0,1), set 𝛾 ∶= 1 − 𝛽, fix 𝐻 ≥ 1, and let 𝐿𝐻 ∶= 𝜏max + 𝐻. Let 𝒦_set𝐻 ⊂ (ℝ𝑚)𝐿𝐻+1 be the compact input set under consideration.

| |
|---|

Assume 𝒦_set𝐻 carries orthonormal directions

𝑒sig, 𝑒pos ∈ ℝ𝑚 such that:

- (i) the original signal channel is 𝑡,𝑒sig⟩;
- (ii) the positional-control coordinate is 𝑢 ↦ ⟨𝑢𝑡,𝑒pos⟩,

⊂ (0,∞). Fix additional orthonormal directions

𝐼0 < 𝐼1 < ⋯ < 𝐼𝐿

with ordered positive ranges

𝐻

𝑒prof, 𝑒tail, 𝑒aux, 𝑒src, 𝑒tgt ∈ ℝ𝑚 orthogonal to both 𝑒sig and 𝑒pos.

𝑄𝐻 ∶ (ℝ𝑚)𝐿𝐻+1 → (ℝ𝑚)𝐿𝐻+1

Then there exists a constant-depth LN-free Sessa network

⟨𝑄𝐻(𝑢)𝑡,𝑒sig⟩ = ⟨𝑢𝑡,𝑒sig⟩ (0 ≤ 𝑡 ≤ 𝐿𝐻, 𝑢 ∈ 𝒦_set𝐻);

such that the original signal channel is preserved exactly:

⟨𝑄𝐻(𝑢)𝑡,𝑒pos⟩ = ⟨𝑢𝑡,𝑒pos⟩ (0 ≤ 𝑡 ≤ 𝐿𝐻, 𝑢 ∈ 𝒦_set𝐻);

the positional-control coordinate is preserved exactly:

the profile channel on the prescribed direction 𝑒prof satisfies the uniform two-sided bound 𝑐𝑟−(𝑡 + 1)𝛾 ≤ ⟨𝑄𝐻(𝑢)𝑡,𝑒prof⟩ ≤ 𝑐𝑟+(𝑡 + 1)𝛾, 0 ≤ 𝑡 ≤ 𝐿𝐻, 𝑢 ∈ 𝒦_set𝐻,

with constants independent of 𝐻; and 𝑄𝐻 is signal-transparent along 𝑒sig with respect to the control pair (𝑒pos,𝑒prof): for every 𝑢 ∈ 𝒦_set𝐻, every 𝜏 ∈ {0,…,𝐿𝐻}, and every scalar 𝑎 ∈ ℝ,

𝐻 sig1 𝑡, 𝑒pos 𝐻 𝑡, 𝑒pos 𝐻,

⟨𝑄𝐻(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡, 𝑒prof⟩ = ⟨𝑄𝐻(𝑢)𝑡, 𝑒prof⟩, 0 ≤ 𝑡 ≤ 𝐿𝐻, and

⟨𝑄𝐻(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡, 𝑒sig⟩ = ⟨𝑄𝐻(𝑢)𝑡, 𝑒sig⟩ + 𝑎1[𝑡 = 𝜏], 0 ≤ 𝑡 ≤ 𝐿𝐻. Proof. The auxiliary orthonormal directions

𝑒prof, 𝑒tail, 𝑒aux, 𝑒src, 𝑒tgt are fixed by hypothesis and are orthogonal to both 𝑒sig and 𝑒pos.

𝑒zero ∶= 𝑒prof, 𝐸carry ∶= {0}. This yields a forward-only block

##### Step 1: clear the profile channel. Apply Lemma K.18 with

𝑍𝐻prof ∶ (ℝ𝑚)𝐿𝐻+1 → (ℝ𝑚)𝐿𝐻+1 such that

⟨𝑍𝐻prof(𝑢)𝑡,𝑒sig⟩ = ⟨𝑢𝑡,𝑒sig⟩, ⟨𝑍𝐻prof(𝑢)𝑡,𝑒pos⟩ = ⟨𝑢𝑡,𝑒pos⟩, ⟨𝑍𝐻prof(𝑢)𝑡,𝑒prof⟩ = 0.

Moreover, 𝑍𝐻prof is signal-transparent along 𝑒sig with respect to (𝑒pos,𝑒prof). Let

𝒦_set(0)𝐻 ∶= 𝑍𝐻prof(𝒦_set𝐻).

###### Step 2: build the tail channel. Apply Lemma K.17 to 𝒦_set(0)𝐻 , with 𝐸carry ∶= span{𝑒prof}.

𝑇𝐻tail ∶ (ℝ𝑚)𝐿𝐻+1 → (ℝ𝑚)𝐿𝐻+1

This yields a constant-depth network

⟨𝑇𝐻tail(𝑣)𝑡,𝑒sig⟩ = ⟨𝑣𝑡,𝑒sig⟩, ⟨𝑇𝐻tail(𝑣)𝑡,𝑒pos⟩ = ⟨𝑣𝑡,𝑒pos⟩, ⟨𝑇𝐻tail(𝑣)𝑡,𝑒prof⟩ = ⟨𝑣𝑡,𝑒prof⟩, and the tail channel

such that

𝑔𝑡(𝑣) ∶= ⟨𝑇𝐻tail(𝑣)𝑡,𝑒tail⟩ satisfies

𝑐𝑔−(𝑡 + 1)−𝛽 ≤ 𝑔𝑡(𝑣) ≤ 𝑐𝑔+(𝑡 + 1)−𝛽. Because the carried profile channel is identically zero on 𝒦_set(0)𝐻 and is preserved exactly by 𝑇𝐻tail, one still has ⟨𝑇𝐻tail(𝑣)𝑡,𝑒prof⟩ = 0 ∀𝑣 ∈ 𝒦_set(0)𝐻 .

𝒦_set(1)𝐻 ∶= 𝑇𝐻tail(𝒦_set(0)𝐻 ).

Let

- Step 3: clear the scratch channels. Apply Lemma K.19 to the scratch directions

𝑒aux, 𝑒src, 𝑒tgt, with

𝐸keep ∶= span{𝑒pos,𝑒tail,𝑒prof}. This yields a forward-only concrete block

𝑍𝐻scr ∶ (ℝ𝑚)𝐿𝐻+1 → (ℝ𝑚)𝐿𝐻+1 such that it preserves

𝑒sig, 𝑒pos, 𝑒tail, 𝑒prof exactly and writes

⟨𝑍𝐻scr(𝑤)𝑡,𝑒aux⟩ = ⟨𝑍𝐻scr(𝑤)𝑡,𝑒src⟩ = ⟨𝑍𝐻scr(𝑤)𝑡,𝑒tgt⟩ = 0. Since 𝑍𝐻scr preserves the tail channel exactly, the same bounds

𝑐𝑔−(𝑡 + 1)−𝛽 ≤ ⟨𝑍𝐻scr(𝑤)𝑡,𝑒tail⟩ ≤ 𝑐𝑔+(𝑡 + 1)−𝛽

hold on the image. Let

𝒦̃_set𝐻 ∶= 𝑍𝐻scr(𝒦_set(1)𝐻 ).

On 𝒦̃_set𝐻 we therefore retain the same ordered positional ranges as on 𝒦_set𝐻, the same tail bounds 𝑐𝑔±(𝑡+1)−𝛽, an identically zero profile channel, and identically zero scratch channels 𝑒aux,𝑒src,𝑒tgt.

- Step 4: integrate the tail channel. Apply Lemma K.20 to 𝒦̃_set𝐻, with 𝐸carry ∶= span{𝑒aux,𝑒src,𝑒tgt}.

Because these carried channels are already identically zero on 𝒦̃_set𝐻, this application is fully legitimate and keeps them zero. We obtain a single LN-free Sessa block

𝐼𝐻 ∶ (ℝ𝑚)𝐿𝐻+1 → (ℝ𝑚)𝐿𝐻+1

⟨𝐼𝐻(𝑤)𝑡,𝑒sig⟩ = ⟨𝑤𝑡,𝑒sig⟩, ⟨𝐼𝐻(𝑤)𝑡,𝑒pos⟩ = ⟨𝑤𝑡,𝑒pos⟩, ⟨𝐼𝐻(𝑤)𝑡,𝑒tail⟩ = ⟨𝑤𝑡,𝑒tail⟩, and

such that

𝑐𝑟−(𝑡 + 1)𝛾 ≤ ⟨𝐼𝐻(𝑤)𝑡,𝑒prof⟩ ≤ 𝑐𝑟+(𝑡 + 1)𝛾.

𝑄𝐻 ∶= 𝐼𝐻 ∘ 𝑍𝐻scr ∘ 𝑇𝐻tail ∘ 𝑍𝐻prof. The exact preservation and two-sided profile bounds follow immediately from the four stages above.

##### Step 5: define the preparatory network. Set

###### Step 6: verify signal-transparency. Fix 𝑢 ∈ 𝒦_set𝐻, 𝜏 ∈ {0,…,𝐿𝐻}, and 𝑎 ∈ ℝ. Define 𝑢(𝑎,𝜏) ∶= 𝑢 + 𝑎𝑒sig1[⋅ = 𝜏].

By signal-transparency of 𝑍𝐻prof,

𝑍𝐻prof(𝑢(𝑎,𝜏)) = 𝑍𝐻prof(𝑢) + 𝑎𝑒sig1[⋅ = 𝜏] on the signal channel, while the 𝑒pos- and 𝑒prof-channels are unchanged. Applying signal-transparency of 𝑇𝐻tail then gives

𝑇𝐻tail(𝑍𝐻prof(𝑢(𝑎,𝜏))) = 𝑇𝐻tail(𝑍𝐻prof(𝑢)) + 𝑎𝑒sig1[⋅ = 𝜏] on the signal channel, while the 𝑒pos- and 𝑒tail-channels are unchanged and the 𝑒prof-channel remains zero. Now 𝑍𝐻scr preserves 𝑒sig,𝑒pos,𝑒tail,𝑒prof exactly, so

𝑍𝐻scr(𝑇𝐻tail(𝑍𝐻prof(𝑢(𝑎,𝜏)))) = 𝑍𝐻scr(𝑇𝐻tail(𝑍𝐻prof(𝑢))) + 𝑎𝑒sig1[⋅ = 𝜏] on the signal channel, and the 𝑒pos-, 𝑒tail-, and 𝑒prof-channels are unchanged.

Thus the two inputs fed into 𝐼𝐻 differ only on the 𝑒sig-channel and have the same 𝑒pos-, 𝑒tail-, and 𝑒prof-streams. In the concrete construction of Lemma K.20, the feedback weights 𝛼𝑏 and forward weights 𝛼𝑓 depend only on

the positional stream, while the forward values 𝑔𝑡 are exact reads of the 𝑒tail-channel. Hence the forward signals 𝑓𝑡, the feedback matrices 𝐵𝐻, and the solve outputs 𝑟𝑡 are identical for the two inputs. Moreover, the output projection of 𝐼𝐻 vanishes on the 𝑒sig-, 𝑒pos-, and 𝑒tail-channels, so the 𝑒sig-channel passes through exactly and the 𝑒pos-coordinate is unchanged. Therefore

𝐻(𝑢(𝑎,𝜏))𝑡,𝑒pos 𝐻 𝑡,𝑒pos⟩,

⟨𝑄𝐻(𝑢(𝑎,𝜏))𝑡,𝑒prof⟩ = ⟨𝑄𝐻(𝑢)𝑡,𝑒prof⟩, ⟨𝑄𝐻(𝑢(𝑎,𝜏))𝑡,𝑒sig⟩ = ⟨𝑄𝐻(𝑢)𝑡,𝑒sig⟩ + 𝑎1[𝑡 = 𝜏]. This proves the stated signal-transparency property.

Lemma K.22 (Profile-compensated macro-layer). Fix 𝛽 ∈ (0,1), set 𝛾 ∶= 1 − 𝛽, and fix 𝑇 ≥ 0. Let 𝒦_set ⊂ (ℝ𝑚)𝑇+1 be compact. Assume orthonormal directions

| |
|---|

𝑒sig, 𝑒pos, 𝑒prof, 𝑒src ∈ ℝ𝑚 and a subspace 𝐸carry ⊂ ℝ𝑚 orthogonal to all four, such that:

- (i) the positional-control ranges

𝐼𝑡 ∶= {⟨𝑢𝑡,𝑒pos⟩ ∶ 𝑢 ∈ 𝒦_set}, 0 ≤ 𝑡 ≤ 𝑇, are compact and strictly ordered:

𝐼0 < 𝐼1 < ⋯ < 𝐼𝑇 ⊂ (0,∞);

- (ii) the profile channel 𝑟𝑡(𝑢) ∶= ⟨𝑢𝑡,𝑒prof⟩

𝑐𝑟−(𝑡 + 1)𝛾 ≤ 𝑟𝑡(𝑢) ≤ 𝑐𝑟+(𝑡 + 1)𝛾, 0 ≤ 𝑡 ≤ 𝑇, 𝑢 ∈ 𝒦_set. Then there exists a constant-depth LN-free Sessa macro-layer

satisfies

𝑀𝑇 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1

such that the 𝑒pos-channel, the 𝑒prof-channel, and every channel in 𝐸carry are preserved exactly, and 𝑀𝑇 has signal-blind exact scalar transport along 𝑒sig over

𝐸ctrl ∶= span{𝑒pos,𝑒prof} ⊕ 𝐸carry, with kernel

(𝑖,𝑗) = 𝐷mac𝑢 (𝑖)1[𝑖 = 𝑗] + 𝐾mac𝑢 (𝑖,𝑗)1[𝑗 < 𝑖]; There exist constants

𝒯𝑢𝑀

𝑇

1 ≤ 𝑑mac− ≤ 𝑑mac+ < ∞, 0 < 𝑎−mac ≤ 𝑎+mac < ∞, depending only on (𝛽,𝑐𝑟−,𝑐𝑟+), but independent of 𝑇, such that

𝑑mac− ≤ 𝐷mac𝑢 (𝑖) ≤ 𝑑mac+ , 0 ≤ 𝑖 ≤ 𝑇, and

𝑎−mac(𝑖 + 1)−𝛽 ≤ 𝐾mac𝑢 +mac(𝑖 + 1)−𝛽, 0 ≤ 𝑗 < 𝑖 ≤ 𝑇. In particular,

𝐾mac𝑢 (𝑖,𝑗) ≤ 𝑎+mac(𝑖 − 𝑗 + 1)−𝛽. Consequently,

𝜕𝑀𝑇(𝑢)𝑖 𝜕𝑢𝑗

𝑒⊤sig

𝑒sig = 𝐷mac𝑢 (𝑖)1[𝑖 = 𝑗] + 𝐾mac𝑢 (𝑖,𝑗)1[𝑗 < 𝑖].

𝑥𝑡 ∶= ⟨𝑢𝑡,𝑒sig⟩, 𝑟𝑡(𝑢) ∶= ⟨𝑢𝑡,𝑒prof⟩, 0 ≤ 𝑡 ≤ 𝑇. We construct

Proof. Write

𝑀𝑇 = 𝐴diff𝑇 ∘ 𝑊𝑇src, where 𝑊𝑇src is a local source writer and 𝐴diff𝑇 is the diffuse transport-bearing block.

- Step 1: local source writer. Choose a parameter 𝜇 ∈ (0, 12] and apply Lemma K.2 to the ordered positionalcontrol coordinate 𝑒pos. This yields a forward attention row satisfying

𝛼𝑓𝑡,𝑗 ≤ 𝜇, 0 ≤ 𝑡 ≤ 𝑇.

𝛼𝑓𝑡,𝑡 ≥ 1 − 𝜇, ∑ 𝑗<𝑡

𝑊𝑇src ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1.

We now build a forward-only LN-free Sessa block

Choose one forward value coordinate equal to 1:

𝑣𝑡(0) ≡ 1. Hence

𝑠(0)𝑡 = ∑ 𝑗≤𝑡

𝛼𝑓𝑡,𝑗 ⋅ 1 = 1.

Next read the profile channel exactly using Corollary K.5. Choose two 𝑎-slots

𝑎(+)𝑡 = 𝐿⟨𝑢𝑡,𝑒prof⟩, 𝑎(−)𝑡 = −𝐿⟨𝑢𝑡,𝑒prof⟩ for any fixed 𝐿 > 0, and choose the value projection so that

1 𝐿

(𝑎̄(+)𝑡 − 𝑎̄(−)𝑡 ) = ⟨𝑢𝑡,𝑒prof⟩ = 𝑟𝑡(𝑢). Let

𝑣𝑡(1) =

𝑚𝑢𝑡 ∶= 𝑠(1)𝑡 ∶= ∑ 𝑗≤𝑡

𝛼𝑓𝑡,𝑗 𝑟𝑗(𝑢).

𝑔𝑡(0) = ⟨𝑢𝑡,𝑒src⟩, 𝑔𝑡(1) = ⟨𝑢𝑡,𝑒sig⟩ = 𝑥𝑡, and choose the output projection on the 𝑒src-channel with coeﬀicients (−1,+1). Then

Choose two gate coordinates

⟨𝑊𝑇src(𝑢)𝑡,𝑒src⟩ = ⟨𝑢𝑡,𝑒src⟩ − 𝑠(0)𝑡 ⟨𝑢𝑡,𝑒src⟩ + 𝑠(1)𝑡 𝑥𝑡 = 𝑚𝑢𝑡 𝑥𝑡.

All other output columns are zero, so the 𝑒sig-, 𝑒pos-, 𝑒prof-, and 𝐸carry-channels are preserved exactly. It remains to bound 𝑚𝑢𝑡 . Since every 𝑟𝑗(𝑢) ≥ 0,

𝑚𝑢𝑡 ≥ 𝛼𝑓𝑡,𝑡 𝑟𝑡(𝑢) ≥ (1 − 𝜇)𝑐𝑟−(𝑡 + 1)𝛾. Also, for every 𝑗 ≤ 𝑡,

𝑟𝑗(𝑢) ≤ 𝑐𝑟+(𝑗 + 1)𝛾 ≤ 𝑐𝑟+ 𝛾, so

𝛼𝑓𝑡,𝑗𝑟𝑗(𝑢) ≤ 𝑐𝑟+(𝑡 + 1)𝛾.

𝑚𝑢𝑡 = ∑ 𝑗≤𝑡

𝑚−(𝑡 + 1)𝛾 ≤ 𝑚𝑢𝑡 ≤ 𝑚+(𝑡 + 1)𝛾, 𝑚− ∶= (1 − 𝜇)𝑐𝑟−, 𝑚+ ∶= 𝑐𝑟+.

Therefore

- Step 2: diffuse transport block. Let 𝑤 ∶= 𝑊𝑇src(𝑢).

𝐴diff𝑇 ∶ (ℝ𝑚)𝑇+1 → (ℝ𝑚)𝑇+1 as follows.

We now build a single LN-free Sessa block

𝑞𝑘𝑓 ≡ 0, 𝑘𝑗𝑓 ≡ 0. Hence the forward row is exactly uniform on the visible prefix:

Forward branch. Choose all forward queries and keys equal to zero:

1 𝑘 + 1

𝛼𝑓𝑘,𝑗 =

1[𝑗 ≤ 𝑘].

Read the source scratch channel exactly using Corollary K.5. Choose two 𝑎-slots

𝑎(+)𝑗 = 𝐿⟨𝑤𝑗,𝑒src⟩, 𝑎(−)𝑗 = −𝐿⟨𝑤𝑗,𝑒src⟩, and choose the value projection so that

1 𝐿

(𝑎̄(+)𝑗 − 𝑎̄(−)𝑗 ) = ⟨𝑤𝑗,𝑒src⟩ = 𝑚𝑢𝑗 𝑥𝑗. Thus the forward signal is

𝑣𝑗src =

1 𝑘 + 1

𝑘

𝑓𝑘 = ∑ 𝑗≤𝑘

𝛼𝑓𝑘,𝑗𝑣𝑗src =

∑

𝑚𝑢𝑗 𝑥𝑗.

𝑗=0

𝑞𝑖𝑏 ≡ 0, 𝑘𝑗𝑏 ≡ 0, 𝛾𝑖 ≡ 𝛾 = 1 − 𝛽. Therefore the strict-past feedback row is exactly uniform:

Feedback branch. Choose all feedback queries and keys equal to zero and the feedback gain constant:

1 𝑖

𝛼𝑏𝑖,𝑘 =

1[𝑘 < 𝑖], 1 ≤ 𝑖 ≤ 𝑇,

𝛾 𝑖

𝐵𝑖,𝑘 =

1[𝑘 < 𝑖].

and the scalar feedback matrix is

Θ𝑖,𝑘 ∶= [(𝐼 − 𝐵)−1]𝑖,𝑘, 0 ≤ 𝑘 ≤ 𝑖 ≤ 𝑇. Exactly as in the proof of Lemma K.15, one has

Let

Θ𝑖,𝑖 = 1, and for 𝑘 < 𝑖,

Γ(𝑘 + 1) Γ(𝑘 + 1 + 𝛾)

Γ(𝑖 + 𝛾) Γ(𝑖 + 1)

. Hence there exist constants

Θ𝑖,𝑘 = 𝛾

0 < 𝑐Θ− ≤ 𝑐Θ+ < ∞ depending only on 𝛽, such that

𝑐Θ−(𝑘 + 1)−𝛾(𝑖 + 1)−𝛽 ≤ Θ𝑖,𝑘 ≤ 𝑐Θ+(𝑘 + 1)−𝛾(𝑖 + 1)−𝛽, 0 ≤ 𝑘 < 𝑖 ≤ 𝑇.

Write transport into the signal channel. Choose one gate coordinate identically 1, and choose the output projection so that the solve output adds +𝑠𝑖 to the 𝑒sig-channel and all output columns on

𝑒pos, 𝑒prof, 𝐸carry

vanish. Therefore

⟨𝐴diff𝑇 (𝑤)𝑖,𝑒sig⟩ = ⟨𝑤𝑖,𝑒sig⟩ + 𝑠𝑖 = 𝑥𝑖 + 𝑠𝑖, where

𝑖

𝑠𝑖 =

∑

Θ𝑖,𝑘𝑓𝑘.

𝑘=0

Since 𝑊𝑇src preserves 𝑒sig,𝑒pos,𝑒prof,𝐸carry exactly, the full macro-layer 𝑀𝑇 = 𝐴diff𝑇 ∘𝑊𝑇src also preserves 𝑒pos,𝑒prof,𝐸carry exactly.

- Step 3: exact transport formula. Substituting the expression for 𝑓𝑘, we get

𝑠𝑖 =

𝑖

∑

𝑘=0

Θ𝑖,𝑘

1 𝑘 + 1

𝑘

∑

𝑗=0

𝑚𝑢𝑗 𝑥𝑗 =

𝑖

∑

𝑗=0

(𝑚𝑢𝑗

𝑖

∑

𝑘=𝑗

Θ𝑖,𝑘 𝑘 + 1

)𝑥𝑗.

Define

𝐿(𝑖,𝑗) ∶=

𝑖

∑

𝑘=𝑗

Θ𝑖,𝑘 𝑘 + 1

, 0 ≤ 𝑗 ≤ 𝑖 ≤ 𝑇.

Then

⟨𝑀𝑇(𝑢)𝑖,𝑒sig⟩ = 𝑥𝑖 +

𝑖

∑

𝑗=0

𝑚𝑢𝑗 𝐿(𝑖,𝑗)𝑥𝑗.

Since Θ𝑖,𝑖 = 1, we have

𝐿(𝑖,𝑖) =

1 𝑖 + 1

. Therefore

⟨𝑀𝑇(𝑢)𝑖,𝑒sig⟩ = (1 +

𝑚𝑢𝑖 𝑖 + 1

)𝑥𝑖 + ∑ 𝑗<𝑖

𝑚𝑢𝑗 𝐿(𝑖,𝑗)𝑥𝑗.

Define

𝐷mac𝑢 (𝑖) ∶= 1 +

𝑚𝑢𝑖 𝑖 + 1

, 𝐾mac𝑢 (𝑖,𝑗) ∶= 𝑚𝑢𝑗 𝐿(𝑖,𝑗) (𝑗 < 𝑖). This yields exact scalar transport on the signal channel:

⟨𝑀𝑇(𝑢)𝑖,𝑒sig⟩ = 𝐷mac𝑢 (𝑖)𝑥𝑖 + ∑ 𝑗<𝑖

𝐾mac𝑢 (𝑖,𝑗)𝑥𝑗.

The coeﬀicient 𝑚𝑢𝑗 depends only on the 𝑒pos- and 𝑒prof-control streams, because the source writer uses positional self-focusing and an exact read of the profile channel only. The kernel 𝐿(𝑖,𝑗) depends only on the fixed diffuse

transport block. Hence 𝐷mac𝑢 (𝑖) and 𝐾mac𝑢 (𝑖,𝑗) depend only on the control stream

(Πctrl𝑢𝑡)𝑇𝑡=0, 𝐸ctrl ∶= span{𝑒pos,𝑒prof} ⊕ 𝐸carry. Thus 𝑀𝑇 has signal-blind exact scalar transport over 𝐸ctrl.

- Step 4: diagonal bounds. Since

𝑚−(𝑖 + 1)𝛾 ≤ 𝑚𝑢𝑖 ≤ 𝑚+(𝑖 + 1)𝛾,

𝑚𝑢𝑖 𝑖 + 1

≤ 1 + 𝑚+(𝑖 + 1)𝛾−1 = 1 + 𝑚+(𝑖 + 1)−𝛽 ≤ 1 + 𝑚+. Hence we may take

1 ≤ 𝐷mac𝑢 (𝑖) = 1 +

we obtain

𝑑mac− ∶= 1, 𝑑mac+ ∶= 1 + 𝑚+.

- Step 5: off-diagonal upper bound. Fix 0 ≤ 𝑗 < 𝑖 ≤ 𝑇. Using Θ𝑖,𝑖 = 1 and the upper bound on Θ𝑖,𝑘 for 𝑘 < 𝑖,

𝐿(𝑖,𝑗) ≤

1 𝑖 + 1

+ 𝑐Θ+(𝑖 + 1)−𝛽

𝑖−1

∑

𝑘=𝑗

(𝑘 + 1)−1−𝛾.

Since

1 𝑖 + 1

≤ (𝑗 + 1)−𝛾(𝑖 + 1)−𝛽, and

𝑖−1

∑

𝑘=𝑗

(𝑘 + 1)−1−𝛾 ≤

∞

∑

𝑘=𝑗

(𝑘 + 1)−1−𝛾 ≲𝛾 (𝑗 + 1)−𝛾,

there exists 𝐶𝐿+ < ∞, depending only on 𝛽, such that

𝐿(𝑖,𝑗) ≤ 𝐶𝐿+(𝑗 + 1)−𝛾(𝑖 + 1)−𝛽. Therefore

𝐾mac𝑢 (𝑖,𝑗) = 𝑚𝑢𝑗 𝐿(𝑖,𝑗) ≤ 𝑚+(𝑗 + 1)𝛾 ⋅ 𝐶𝐿+(𝑗 + 1)−𝛾(𝑖 + 1)−𝛽. Hence

𝐾mac𝑢 (𝑖,𝑗) ≤ 𝑎+mac(𝑖 + 1)−𝛽, 𝑎+mac ∶= 𝑚+𝐶𝐿+.

- Step 6: off-diagonal lower bound. Fix 0 ≤ 𝑗 < 𝑖 ≤ 𝑇.

- Case 0: 𝑗 = 0. Since Θ𝑖,0 appears in the sum defining 𝐿(𝑖,0), we have 𝐿(𝑖,0) ≥ Θ𝑖,0.

By the resolvent bound,

Θ𝑖,0 ≥ 𝑐Θ−(0 + 1)−𝛾(𝑖 + 1)−𝛽 = 𝑐Θ−(𝑖 + 1)−𝛽. Also 𝑚𝑢0 ≥ 𝑚−. Therefore

𝐾mac𝑢 (𝑖,0) = 𝑚𝑢0 𝐿(𝑖,0) ≥ 𝑚−𝑐Θ−(𝑖 + 1)−𝛽.

- Case 1: 1 ≤ 𝑗 ≤ 𝑖/2. Then 2𝑗 ≤ 𝑖, so

𝐿(𝑖,𝑗) ≥

2𝑗−1

∑

𝑘=𝑗

Θ𝑖,𝑘 𝑘 + 1

≥ 𝑐Θ−(𝑖 + 1)−𝛽

2𝑗−1

∑

𝑘=𝑗

(𝑘 + 1)−1−𝛾.

Since the sum over one dyadic block is comparable to (𝑗 + 1)−𝛾, there exists 𝑐𝐿(1) > 0, depending only on 𝛽, such that

𝐿(𝑖,𝑗) ≥ 𝑐𝐿(1)(𝑗 + 1)−𝛾(𝑖 + 1)−𝛽. Hence

𝐾mac𝑢 (𝑖,𝑗) = 𝑚𝑢𝑗 𝐿(𝑖,𝑗) ≥ 𝑚−(𝑗 + 1)𝛾 ⋅ 𝑐𝐿(1)(𝑗 + 1)−𝛾(𝑖 + 1)−𝛽 = 𝑚−𝑐𝐿(1)(𝑖 + 1)−𝛽.

- Case 2: 𝑗 > 𝑖/2. Then

1 𝑖 + 1

,

𝐿(𝑖,𝑗) ≥

𝑚𝑢𝑗 𝑖 + 1

𝑚−(𝑗 + 1)𝛾 𝑖 + 1

𝐾mac𝑢 (𝑖,𝑗) = 𝑚𝑢𝑗 𝐿(𝑖,𝑗) ≥

≥

.

so

Since 𝑗 + 1 > 𝑖+12 ,

(𝑗 + 1)𝛾 ≥ 2−𝛾(𝑖 + 1)𝛾. Therefore

𝐾mac𝑢 (𝑖,𝑗) ≥ 𝑚−2−𝛾(𝑖 + 1)𝛾−1 = 𝑚−2−𝛾(𝑖 + 1)−𝛽. Combining the three cases gives

𝐾mac𝑢 (𝑖,𝑗) ≥ 𝑎−mac(𝑖 + 1)−𝛽, 𝑎−mac ∶= min{𝑚−𝑐Θ−, 𝑚−𝑐𝐿(1), 𝑚−2−𝛾}.

For any 𝜂 > 0, replacing 𝒦_set by Satsig𝜂 (𝒦_set) leaves the ordered positional ranges and the two-sided profile bounds unchanged, since only the 𝑒sig-channel is perturbed. The same source-writer plus diffuse-transport construction therefore yields the same exact scalar transport formula on Sat𝜂sig(𝒦_set), with the same coeﬀicients 𝐷mac𝑢 (𝑖) and 𝐾mac𝑢 (𝑖,𝑗), because these coeﬀicients depend only on the control stream (𝑒pos,𝑒prof,𝐸carry). Applying Lemma K.8(i) gives

𝜕𝑀𝑇(𝑢)𝑖 𝜕𝑢𝑗

𝑒⊤sig

𝑒sig = 𝐷mac𝑢 (𝑖)1[𝑖 = 𝑗] + 𝐾mac𝑢 (𝑖,𝑗)1[𝑗 < 𝑖].

| |
|---|

𝐸ctrl ∶= span{𝑒pos,𝑒prof} ⊕ 𝐸carry, Πctrl ∶ ℝ𝑚 → 𝐸ctrl, 𝜋sig(𝑣) ∶= ⟨𝑣,𝑒sig⟩,

Corollary K.23 (Macro-layer transport). Under the hypotheses of Lemma K.22, let

and let 𝑀𝑇 be the concrete macro-layer constructed there. Then for every 𝛿 ≥ 0, 𝑀𝑇 has signal-blind exact scalar transport along 𝑒sig over 𝐸ctrl on Satsig𝛿 (𝒦_set), with the same scalar transport kernel 𝒯𝑢𝑀

(𝑖,𝑗) as on 𝒦_set. More precisely, if

𝑇

𝑇

𝑎𝑡𝑒sig1[⋅ = 𝑡], 𝑢 ∈ 𝒦_set, then

𝑣 = 𝑢 +

∑

𝑡=0

Πctrl𝑀𝑇(𝑣)𝑖 = Πctrl𝑣𝑖, 0 ≤ 𝑖 ≤ 𝑇, and

𝑖

𝜋sig(𝑀𝑇(𝑣)𝑖) =

∑

𝒯𝑢𝑀

(𝑖,𝑗)𝜋sig(𝑣𝑗), 0 ≤ 𝑖 ≤ 𝑇.

𝑗=0

𝑇

The right-hand side depends only on the control stream of 𝑣, hence is independent of the choice of 𝑢 ∈ 𝒦_set with the same control stream.

𝑀𝑇 = 𝐴diff𝑇 ∘ 𝑊𝑇src

Proof. Write

exactly as in the proof of Lemma K.22. Fix

𝑇

∑

𝑣 = 𝑢 +

𝑎𝑡𝑒sig1[⋅ = 𝑡], 𝑢 ∈ 𝒦_set.

𝑡=0

Since 𝑣 differs from 𝑢 only on the 𝑒sig-channel, the 𝑒pos-, 𝑒prof-, and 𝐸carry-streams are unchanged. Hence the

𝑚𝑣𝑡 = 𝑚𝑢𝑡 , 0 ≤ 𝑡 ≤ 𝑇.

self-focused profile averages from the source-writer stage are unchanged:

⟨𝑊𝑇src(𝑣)𝑡,𝑒src⟩ = 𝑚𝑢𝑡 𝜋sig(𝑣𝑡), 0 ≤ 𝑡 ≤ 𝑇.

Therefore the explicit source-writer formula gives

Moreover, 𝑊𝑇src preserves the channels in 𝐸ctrl exactly, because it modifies only the 𝑒src-channel. In the diffuse stage, the forward row is the exact uniform prefix average, so the forward signal entering the fixed feedback solve is

1 𝑘 + 1

𝑘

∑

𝑚𝑢𝑗 𝜋sig(𝑣𝑗), 0 ≤ 𝑘 ≤ 𝑇.

𝑓𝑘(𝑣) =

𝑗=0

The feedback matrix 𝐵, its resolvent Θ, and the kernel

Θ𝑖,𝑘 𝑘 + 1

𝑖

𝐿(𝑖,𝑗) ∶=

∑

𝑘=𝑗

depend only on 𝛽, hence are independent of 𝑣. Thus the solve output satisfies

𝑖

𝑖

𝑠𝑖(𝑣) =

∑

Θ𝑖,𝑘𝑓𝑘(𝑣) =

∑

𝑚𝑢𝑗 𝐿(𝑖,𝑗)𝜋sig(𝑣𝑗).

𝑗=0

𝑘=0

𝑚𝑢𝑖 𝑖 + 1

Using the definitions from Lemma K.22,

𝐷mac𝑢 (𝑖) ∶= 1 +

, 𝐾mac𝑢 (𝑖,𝑗) ∶= 𝑚𝑢𝑗 𝐿(𝑖,𝑗) (𝑗 < 𝑖), we obtain

𝑖

𝜋sig(𝑀𝑇(𝑣)𝑖) = 𝜋sig(𝑣𝑖) + 𝑠𝑖(𝑣) =

∑

𝒯𝑢𝑀

(𝑖,𝑗)𝜋sig(𝑣𝑗).

𝑗=0

𝑇

Finally, 𝐴diff𝑇 modifies only the 𝑒sig-channel and preserves 𝑒pos,𝑒prof,𝐸carry exactly. Hence 𝑀𝑇 preserves 𝐸ctrl exactly on Satsig𝛿 (𝒦_set). Since the coeﬀicients 𝑚𝑗𝑢, and therefore 𝒯𝑢𝑀

(𝑖,𝑗), depend only on the control stream, the displayed kernel is independent of the choice of 𝑢 ∈ 𝒦_set with the same control stream. This proves the claim.

𝑇

| |
|---|

Πsrc(𝑣)𝑡 ∶= 𝑣𝑡 − ⟨𝑣𝑡,𝑒src⟩𝑒src, 0 ≤ 𝑡 ≤ 𝑇, be the tokenwise orthogonal projection that kills the 𝑒src-channel, and define

- Lemma K.24 (Projected macro-layer). Under the hypotheses of Lemma K.22, let

𝑀̄𝑇 ∶= Πsrc ∘ 𝑀𝑇. Then:

- (i) 𝑀𝑇 is blind to the incoming 𝑒src-channel: 𝑀𝑇 = 𝑀𝑇 ∘ Πsrc.
- (ii) 𝑀̄𝑇 preserves the 𝑒pos-channel, the 𝑒prof-channel, and every channel in 𝐸carry exactly.

- (iii) 𝑀̄𝑇 has signal-blind exact scalar transport along 𝑒sig over 𝐸ctrl ∶= span{𝑒pos,𝑒prof} ⊕ 𝐸carry,

with exactly the same scalar transport kernel as 𝑀𝑇:

𝒯𝑢𝑀̄

𝑇

(𝑖,𝑗) = 𝒯𝑢𝑀

𝑇

(𝑖,𝑗), 0 ≤ 𝑗 ≤ 𝑖 ≤ 𝑇.

- (iv) For every 𝛿 ≥ 0 there exists 𝛿′ = 𝛿′(𝛿,𝒦_set) < ∞ such that

𝑀̄𝑇(Satsig𝛿 (𝒦_set)) ⊂ Satsig𝛿′ (𝑀̄𝑇(𝒦_set)). More precisely, if

𝑢′ = 𝑢 +

𝑇

∑

𝑡=0

𝑎𝑡𝑒sig1[⋅ = 𝑡], 𝑢 ∈ 𝒦_set, max

𝑡

|𝑎𝑡| ≤ 𝛿, then

𝑀̄𝑇(𝑢′)𝑖 = 𝑀̄𝑇(𝑢)𝑖 + (

𝑖

∑

𝑗=0

𝒯𝑢𝑀

𝑇

(𝑖,𝑗)𝑎𝑗)𝑒sig, 0 ≤ 𝑖 ≤ 𝑇.

- (v) For every 𝛿 ≥ 0, 𝑀̄𝑇 has signal-blind exact scalar transport along 𝑒sig over 𝐸ctrl ∶= span{𝑒pos,𝑒prof} ⊕ 𝐸carry

on Satsig𝛿 (𝒦_set), with the same scalar transport kernel as 𝑀𝑇. More precisely, if

𝑇

𝑣 = 𝑢 +

∑

𝑎𝑡𝑒sig1[⋅ = 𝑡], 𝑢 ∈ 𝒦_set,

𝑡=0

Πctrl𝑀̄𝑇(𝑣)𝑖 = Πctrl𝑣𝑖, 0 ≤ 𝑖 ≤ 𝑇, and

then

𝜋sig(𝑀̄𝑇(𝑣)𝑖) =

𝑖

∑

𝒯𝑢𝑀

(𝑖,𝑗)𝜋sig(𝑣𝑗), 0 ≤ 𝑖 ≤ 𝑇.

𝑗=0

𝑇

The right-hand side depends only on the control stream of 𝑣, hence is independent of the choice of 𝑢 ∈ 𝒦_set with the same control stream.

𝑀𝑇 = 𝐴diff𝑇 ∘ 𝑊𝑇src

Proof. Write

- as in the proof of Lemma K.22. For item (i), the explicit source-writer formula there gives

⟨𝑊𝑇src(𝑢)𝑡,𝑒src⟩ = 𝑚𝑢𝑡 ⟨𝑢𝑡,𝑒sig⟩,

where 𝑚𝑢𝑡 depends only on the control stream (𝑒pos,𝑒prof,𝐸carry), and not on the incoming 𝑒src-coordinate. All other channels used by 𝑊𝑇src are likewise independent of the incoming 𝑒src-channel. Hence

𝑊𝑇src(𝑢) = 𝑊𝑇src(Πsrc𝑢). Applying 𝐴diff𝑇 yields

𝑀𝑇(𝑢) = 𝑀𝑇(Πsrc𝑢),

which is item (i). Item (ii) follows because 𝑀𝑇 already preserves 𝑒pos,𝑒prof,𝐸carry exactly by Lemma K.22, and Πsrc acts as the identity on those channels.

- For item (iii), Πsrc acts as the identity on the 𝑒sig-coordinate, so ⟨𝑀̄𝑇(𝑢)𝑖,𝑒sig⟩ = ⟨𝑀𝑇(𝑢)𝑖,𝑒sig⟩.

Since 𝑀𝑇 has signal-blind exact scalar transport with kernel 𝒯𝑢𝑀

𝑇

, the same is true for 𝑀̄𝑇, with the same kernel.

- For item (iv), fix 𝑢 ∈ 𝒦_set and

𝑢′ = 𝑢 +

𝑇

∑

𝑡=0

𝑎𝑡𝑒sig1[⋅ = 𝑡], max

𝑡

|𝑎𝑡| ≤ 𝛿.

The control stream is unchanged, so the same transport kernel 𝒯𝑢𝑀

𝑇

applies to both 𝑢 and 𝑢′. By item (iii),

⟨𝑀̄𝑇(𝑢′)𝑖 − 𝑀̄𝑇(𝑢)𝑖,𝑒sig⟩ =

𝑖

∑

𝑗=0

𝒯𝑢𝑀

𝑇

(𝑖,𝑗)𝑎𝑗.

In the concrete construction of Lemma K.22, the source writer modifies only the 𝑒src-channel and the diffuse block modifies only the 𝑒sig-channel; every channel orthogonal to

span{𝑒sig,𝑒pos,𝑒prof,𝑒src} ⊕ 𝐸carry

is preserved exactly. Thus the only possible signal-dependent non-signal output channel is 𝑒src, and Πsrc removes it. Hence

𝑀̄𝑇(𝑢′)𝑖 − 𝑀̄𝑇(𝑢)𝑖 = (

𝑖

∑

𝑗=0

𝒯𝑢𝑀

𝑇

(𝑖,𝑗)𝑎𝑗)𝑒sig,

which is exactly a bounded signal-fiber perturbation over 𝑀̄𝑇(𝑢). Since 𝑇 is finite and 𝒦_set is compact, the quantity

sup

𝑢∈𝒦_set

sup

0≤𝑖≤𝑇

𝑖

∑

𝑗=0

|𝒯𝑢𝑀

𝑇

(𝑖,𝑗)|

is finite, so one may take

𝛿′ ∶= 𝛿 sup

𝑢∈𝒦_set

sup

0≤𝑖≤𝑇

𝑖

∑

𝑗=0

|𝒯𝑢𝑀

𝑇

(𝑖,𝑗)|.

- For item (v), fix 𝛿 ≥ 0 and 𝑣 ∈ Satsig𝛿 (𝒦_set). Write

𝑇

𝑣 = 𝑢 +

∑

𝑎𝑡𝑒sig1[⋅ = 𝑡] with 𝑢 ∈ 𝒦_set.

𝑡=0

𝑀̄𝑇(𝑣)𝑖 = 𝑀̄𝑇(𝑢)𝑖 + (

𝑖

∑

𝒯𝑢𝑀

(𝑖,𝑗)𝑎𝑗)𝑒sig.

By item (iv),

𝑗=0

𝑇

Taking the 𝑒sig-coordinate and using item (iii) on 𝑢 ∈ 𝒦_set, we obtain

𝜋sig(𝑀̄𝑇(𝑣)𝑖) = 𝜋sig(𝑀̄𝑇(𝑢)𝑖) +

𝑖

∑

𝒯𝑢𝑀

(𝑖,𝑗)𝑎𝑗

𝑗=0

𝑇

𝑖

𝑖

=

∑

𝒯𝑢𝑀

(𝑖,𝑗)𝜋sig(𝑢𝑗) +

∑

𝒯𝑢𝑀

(𝑖,𝑗)𝑎𝑗

𝑗=0

𝑗=0

𝑇

𝑇

𝑖

=

∑

𝒯𝑢𝑀

(𝑖,𝑗)𝜋sig(𝑣𝑗).

𝑗=0

𝑇

Moreover, from the explicit construction, 𝑊𝑇src modifies only the 𝑒src-channel, 𝐴diff𝑇 modifies only the 𝑒sig-channel, and Πsrc kills only the 𝑒src-channel. Hence 𝑀̄𝑇 acts as the identity on

𝐸ctrl = span{𝑒pos,𝑒prof} ⊕ 𝐸carry for every input, and therefore

Πctrl𝑀̄𝑇(𝑣)𝑖 = Πctrl𝑣𝑖. Finally, since 𝒯𝑢𝑀

𝑢 ∈ 𝒦_set with the same control stream as 𝑣. Thus 𝑀̄𝑇 has signal-blind exact scalar transport on Satsig𝛿 (𝒦_set) with the same kernel as 𝑀𝑇. This proves the claim.

𝑇

depends only on the control stream, the displayed kernel is independent of the choice of

- Lemma K.25 (Balanced path lower bound). Fix 𝛽 ∈ (0,1), set 𝛾 ∶= 1 − 𝛽, fix 𝑘 ≥ 1, and fix 𝜏max ≥ 0. Then

there exists a constant 𝑐𝑘,𝛽,𝜏bal

max

> 0 such that for every 0 ≤ 𝜏∗ ≤ 𝜏max and every ℓ ≥ 𝑘, with 𝑡 = 𝜏∗ + ℓ,

∑

𝜏∗=𝑖0<𝑖1<⋯<𝑖𝑘=𝑡 ℓ

2𝑘 ≤𝑖𝑟−𝑖𝑟−1≤ 2ℓ𝑘 ∀𝑟

𝑘

∏

𝑟=1

(𝑖𝑟 + 1)−𝛽 ≥ 𝑐𝑘,𝛽,𝜏bal

max

(1 + ℓ)𝑘(1−𝛽)−1.

Proof. The number of balanced paths is ≳𝑘 ℓ𝑘−1 for all ℓ ≥ 𝑘. For every balanced path and every 𝑟 = 1,…,𝑘,

𝑖𝑟 + 1 ≍𝑘,𝜏

max

1 + ℓ. Hence every balanced path contributes at least

𝐶𝑘,𝛽,𝜏−1

max

(1 + ℓ)−𝑘𝛽. Multiplying by the number of balanced paths gives

≳ ℓ𝑘−1(1 + ℓ)−𝑘𝛽 ≍ (1 + ℓ)𝑘−1−𝑘𝛽 = (1 + ℓ)𝑘(1−𝛽)−1.

| |
|---|

- Lemma K.26 (Competitor suppression). Fix 𝛽 ∈ (0,1), set 𝛾 ∶= 1 − 𝛽, fix 𝑘 ≥ 1, and fix 𝜏max ≥ 0. Consider a depth-(𝑘 + 1) exact scalar transport stack on a distinguished signal channel, consisting of one selector block

| |
|---|

𝑆𝐻,𝜏

∗,𝜀𝐻 followed by 𝑘 diffuse profile-compensated macro-layers. Let 𝒯𝑢stack(𝑡,𝜏)

- 1

- 2

denote the resulting exact scalar transport kernel on that signal channel. Assume the selector satisfies

≤ 𝐷sel𝑢 (𝜏∗) ≤ 2, |𝐷sel𝑢 (𝜏)| ≤ 𝜀𝐻 (𝜏 ≠ 𝜏∗), uniformly in 𝑢, and each macro-layer satisfies

1 ≤ 𝐷mac𝑢 (𝑖) ≤ 𝑑mac+ , 𝐾mac𝑢 (𝑖,𝑗) ≤ 𝑎+mac(𝑖 + 1)−𝛽. Then there exists 𝐶comp < ∞, independent of 𝐻, such that for every

𝑡 = 𝜏∗ + ℓ, 1 ≤ ℓ ≤ 𝐻, one has

∑

∣𝒯𝑢stack(𝑡,𝜏)∣ ≤ 𝐶comp 𝜀𝐻 (1 + ℓ)𝑘(1−𝛽).

0≤𝜏<𝑡 𝜏≠𝜏∗

𝜀𝐻 ≤ 𝑐0(𝐻 + 1)−1 with 𝑐0 > 0 small enough, then

In particular, if

- 1

- 2

∑

∣𝒯𝑢stack(𝑡,𝜏)∣ ≤

𝑐sig(1 + ℓ)𝑘(1−𝛽)−1

0≤𝜏<𝑡 𝜏≠𝜏∗

for any prescribed 𝑐sig > 0 after reducing 𝑐0.

Proof. Fix a competitor source 𝜏 ≠ 𝜏∗ with 𝜏 < 𝑡. Any path from 𝜏 to 𝑡 through the selector-plus-𝑘-macro-layer stack must contain at least one genuine jump, because diagonal propagation alone cannot change the time index.

Fix a path with exactly 𝑗 jump layers, where 1 ≤ 𝑗 ≤ 𝑘, and let

𝜏 = 𝑖0 < 𝑖1 < ⋯ < 𝑖𝑗 = 𝑡 be the corresponding jump times. The selector contributes at most 𝜀𝐻 at the source 𝜏 ≠ 𝜏∗. Each jump contributes

- at most 𝑎+mac(𝑖𝑟 + 1)−𝛽, 𝑟 = 1,…,𝑗.

Each non-jump macro-layer contributes at most the diagonal bound 𝑑mac+ . Hence every such path has weight bounded by

𝑗

𝐶0 𝜀𝐻

∏

(𝑖𝑟 + 1)−𝛽,

𝑟=1

where 𝐶0 depends only on 𝑘 and 𝑑mac+ . Now sum over all jump times for fixed 𝑗:

𝑗

𝑗−1

∑

∏

(𝑖𝑟 + 1)−𝛽 = (𝑡 + 1)−𝛽 ∑

∏

(𝑖𝑟 + 1)−𝛽.

𝜏=𝑖0<𝑖1<⋯<𝑖𝑗=𝑡

𝜏<𝑖1<⋯<𝑖𝑗−1<𝑡

𝑟=1

𝑟=1

Using the elementary symmetric-sum bound,

1 (𝑗 − 1)!

𝑗−1

𝑗−1

𝑡−1

∑

∏

(

∑

(𝑚 + 1)−𝛽)

(𝑖𝑟 + 1)−𝛽 ≤

,

𝜏<𝑖1<⋯<𝑖𝑗−1<𝑡

𝑟=1

𝑚=1

𝑡−1

∑

(𝑚 + 1)−𝛽 ≲ (1 + 𝑡)1−𝛽,

and

𝑚=1

𝑗

∑

∏

(𝑖𝑟 + 1)−𝛽 ≤ 𝐶𝑗(1 + 𝑡)𝑗(1−𝛽)−1.

we obtain

𝜏=𝑖0<𝑖1<⋯<𝑖𝑗=𝑡

𝑟=1

𝑘

∑

|𝒯𝑢stack(𝑡,𝜏)| ≤ 𝐶1 𝜀𝐻

(1 + 𝑡)𝑗(1−𝛽)−1 ≤ 𝐶2 𝜀𝐻(1 + 𝑡)𝑘(1−𝛽)−1,

Therefore

𝑗=1

since 𝑘 is fixed. Now 𝑡 = 𝜏∗ + ℓ with 0 ≤ 𝜏∗ ≤ 𝜏max, so

1 + 𝑡 ≍𝜏

1 + ℓ. Hence

|𝒯𝑢stack(𝑡,𝜏)| ≲ 𝜀𝐻(1 + ℓ)𝑘(1−𝛽)−1. Finally sum over all competitors 𝜏 < 𝑡. There are at most 𝑡 ≲𝜏

max

1 + ℓ of them, so

∑

|𝒯𝑢stack(𝑡,𝜏)| ≲ 𝜀𝐻(1 + ℓ)𝑘(1−𝛽).

max

0≤𝜏<𝑡 𝜏≠𝜏∗

This proves the first claim. For the in-particular clause, use 1 + ℓ ≤ 𝐻 + 1:

𝜀𝐻(1 + ℓ)𝑘(1−𝛽) ≤ 𝑐0(𝐻 + 1)−1(1 + ℓ)𝑘(1−𝛽) ≤ 𝑐0(1 + ℓ)𝑘(1−𝛽)−1. Reducing 𝑐0 if necessary yields the desired factor 12𝑐sig. Remark K.27 (Width bookkeeping). After the positional writer has fixed the direction 𝑒pos, choose once and for all six orthonormal directions

| |
|---|

𝑒sig, 𝑒prof, 𝑒tail, 𝑒aux, 𝑒src, 𝑒tgt, all orthogonal to 𝑒pos.

The preparatory network 𝑄𝐻 uses 𝑒prof,𝑒tail,𝑒aux,𝑒src,𝑒tgt; the selector block reuses 𝑒aux and preserves 𝑒prof; each diffuse profile-compensated macro-layer reuses 𝑒src and preserves 𝑒prof; the direction 𝑒tgt remains available as an auxiliary spare scratch direction. No block requires any additional fresh ambient direction beyond these seven coordinates.

In the concrete architecture, each width-𝐷 block also provides 𝐷 𝑎-slots and 𝐷 𝑔-slots in the split (𝑎,𝑔) = split(𝑥𝑊in + 𝑏in).

The constructions below use at most six active 𝑎-slots and at most three active 𝑔-slots in any single block: the plateau window uses four 𝑎-slots, the window writer uses six 𝑎-slots and two 𝑔-slots, the local multiplier uses four 𝑎-slots and two 𝑔-slots, the repaired source writer uses four 𝑎-slots and two 𝑔-slots, the repaired diffuse transport block uses two 𝑎-slots and one 𝑔-slot, the damped predecessor integrator uses three 𝑎-slots and one 𝑔-slot, and the simultaneous scratch reset uses one 𝑎-slot and three 𝑔-slots.

𝐷 ≥ 7

Hence the same condition

simultaneously provides the seven persistent ambient directions and enough concrete 𝑎-/𝑔-slots for every primitive block.

Proof of Theorem 12. Fix 𝐻 ≥ 1 and 0 ≤ 𝜏∗ ≤ 𝜏max. Set 𝐿𝐻 ∶= 𝜏max + 𝐻, 𝑇𝐻 ∶= 𝐿𝐻 + 1.

Composite architecture. For each horizon parameter 𝐻 ≥ 1 and source index 0 ≤ 𝜏∗ ≤ 𝜏max, we construct

𝐺𝐻,𝜏

= 𝑀𝐻,𝑘 ∘ ⋯ ∘ 𝑀𝐻,1 ∘ 𝑆𝐻,𝜏

∗,𝜀𝐻 ∘ 𝑄𝐻 ∘ 𝑃𝐻.

∗

Here 𝑃𝐻 writes a one-directional positional code, 𝑄𝐻 builds a signal-transparent preparatory power-profile channel, 𝑆𝐻,𝜏

∗,𝜀𝐻 is a selector that isolates the chosen source 𝜏∗, and 𝑀𝐻,1,…,𝑀𝐻,𝑘 are the diffuse profile-compensated macro-layers that generate the target polynomial transport envelope.

Inside the proof we also introduce projected variants of the macro-layers in order to expose the exact signalchannel transport kernel while removing an auxiliary scratch channel. This internal projection does not change the realized map on the relevant signal fibers, so it is used only as a bookkeeping device in the kernel calculation.

- Step 1: write the positional code. Apply Corollary 4.11 on the finite prefix {0,…,𝐿𝐻}. This yields a block 𝑃𝐻 ∶ (ℝ𝐷)𝑇𝐻 → (ℝ𝐷)𝑇𝐻

and a unit direction 𝑒pos such that

𝑃𝐻(ℎ)𝑡 = ℎ𝑡 + 𝜆𝑡𝑒pos, 0 ≤ 𝑡 ≤ 𝐿𝐻, for some scalars 𝜆𝑡, and such that on

𝒦_set𝐻 ∶= 𝑃𝐻(𝒳(𝐻)0 ) the scalar ranges

𝐼𝑡 ∶= {⟨𝑢𝑡,𝑒pos⟩ ∶ 𝑢 ∈ 𝒦_set𝐻} are compact and strictly ordered:

𝐼0 < ⋯ < 𝐼𝐿

𝐻

⊂ (0,∞).

Since 𝐷 ≥ 7, after fixing 𝑒pos we may choose orthonormal directions 𝑒sig, 𝑒prof, 𝑒tail, 𝑒aux, 𝑒src, 𝑒tgt

all orthogonal to 𝑒pos; see Remark K.27. By Corollary 4.12, for every 𝑥 ∈ 𝒳(𝐻)0 , every 𝜏, and every scalar 𝑎,

𝑃𝐻(𝑥 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡 = 𝑃𝐻(𝑥)𝑡 + 𝑎𝑒sig1[𝑡 = 𝜏]. In particular,

⟨𝑃𝐻(𝑥 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡,𝑒pos⟩ = ⟨𝑃𝐻(𝑥)𝑡,𝑒pos⟩.

- Step 2: build the preparatory power-profile network. Apply Corollary K.21 to the compact set 𝒦_set𝐻, with the fixed orthonormal directions

𝑒sig, 𝑒pos, 𝑒prof, 𝑒tail, 𝑒aux, 𝑒src, 𝑒tgt,

𝑄𝐻 ∶ (ℝ𝐷)𝑇𝐻 → (ℝ𝐷)𝑇𝐻 with the following properties.

which satisfy the hypotheses of that corollary. This yields a constant-depth network

⟨𝑄𝐻(𝑢)𝑡,𝑒sig⟩ = ⟨𝑢𝑡,𝑒sig⟩.

Signal preservation. The signal channel is preserved exactly:

⟨𝑄𝐻(𝑢)𝑡,𝑒pos⟩ = ⟨𝑢𝑡,𝑒pos⟩.

Positional preservation. The positional-control coordinate is preserved exactly:

Profile growth. The profile channel on the prescribed direction 𝑒prof satisfies

𝑐𝑟−(𝑡 + 1)𝛾 ≤ ⟨𝑄𝐻(𝑢)𝑡,𝑒prof⟩ ≤ 𝑐𝑟+(𝑡 + 1)𝛾, 𝛾 = 1 − 𝛽.

Signal transparency. The map 𝑄𝐻 is signal-transparent relative to (𝑒pos,𝑒prof): for every 𝑢, every 𝜏, and every scalar 𝑎,

𝐻 sig1 𝑡,𝑒pos 𝐻 𝑡,𝑒pos⟩,

⟨𝑄𝐻(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡,𝑒prof⟩ = ⟨𝑄𝐻(𝑢)𝑡,𝑒prof⟩, ⟨𝑄𝐻(𝑢 + 𝑎𝑒sig1[⋅ = 𝜏])𝑡,𝑒sig⟩ = ⟨𝑄𝐻(𝑢)𝑡,𝑒sig⟩ + 𝑎1[𝑡 = 𝜏]. Write

𝑅𝐻 ∶= 𝑄𝐻 ∘ 𝑃𝐻.

- Step 3: select the source index. Apply Lemma K.12 on the image of 𝑅𝐻, using the already fixed directions 𝑒pos,𝑒sig,𝑒aux, with

𝐸carry ∶= span{𝑒prof}, 𝜀𝐻 ∶= 𝑐0(𝐻 + 1)−1, where 𝑐0 > 0 will be fixed later. This yields a selector module

𝑆𝐻,𝜏

∗,𝜀𝐻

which preserves the positional and profile channels and has exact diagonal signal transport

𝒯𝑢𝑆(𝑖,𝑗) = 𝐷sel𝑢 (𝑖)1[𝑖 = 𝑗] with

- 1

- 2

≤ 𝐷sel𝑢 (𝜏∗) ≤ 2, |𝐷sel𝑢 (𝜏)| ≤ 𝜀𝐻 (𝜏 ≠ 𝜏∗).

- Step 4: add the 𝑘 macro-layers. Define

∗,𝜀𝐻(𝑅𝐻(𝒳(𝐻)0 )). This is compact. By Step 2 and Step 3, on 𝒦_setmac𝐻,0 the positional-control ranges are still 𝐼0 < ⋯ < 𝐼𝐿

𝒦_setmac𝐻,0 ∶= 𝑆𝐻,𝜏

⊂ (0,∞),

𝐻

𝑐𝑟−(𝑡 + 1)𝛾 ≤ ⟨𝑢𝑡,𝑒prof⟩ ≤ 𝑐𝑟+(𝑡 + 1)𝛾, 0 ≤ 𝑡 ≤ 𝐿𝐻.

and the profile channel still satisfies

Apply Lemma K.22 with 𝑇 = 𝐿𝐻 to 𝒦_setmac𝐻,0, using the fixed directions

𝑒sig, 𝑒pos, 𝑒prof, 𝑒src, 𝐸carry ∶= {0}, to obtain 𝑀𝐻,1. Define

𝑀̄𝐻,1 ∶= Πsrc ∘ 𝑀𝐻,1. If 𝑘 ≥ 2, set

𝒦_setmac𝐻,1 ∶= 𝑀̄𝐻,1(𝒦_setmac𝐻,0). Inductively, suppose that for some 1 ≤ 𝑟 ≤ 𝑘 − 1 we have already constructed

𝑀𝐻,1,…,𝑀𝐻,𝑟, 𝑀̄𝐻,1,…,𝑀̄𝐻,𝑟, and compact sets

𝒦_setmac𝐻,0,…,𝒦_setmac𝐻,𝑟 such that for each 1 ≤ 𝑠 ≤ 𝑟,

𝒦_setmac𝐻,𝑠 = 𝑀̄𝐻,𝑠(𝒦_setmac𝐻,𝑠−1), and on every 𝒦_setmac𝐻,𝑠 the same ordered positional ranges

⊂ (0,∞) and the same two-sided profile bounds

𝐼0 < ⋯ < 𝐼𝐿

𝐻

𝑐𝑟−(𝑡 + 1)𝛾 ≤ ⟨𝑢𝑡,𝑒prof⟩ ≤ 𝑐𝑟+(𝑡 + 1)𝛾

hold. Apply Lemma K.22 to 𝒦_setmac𝐻,𝑟, with the same fixed directions, to obtain 𝑀𝐻,𝑟+1. Define

𝑀̄𝐻,𝑟+1 ∶= Πsrc ∘ 𝑀𝐻,𝑟+1. If 𝑟 + 1 ≤ 𝑘 − 1, set

𝒦_setmac𝐻,𝑟+1 ∶= 𝑀̄𝐻,𝑟+1(𝒦_setmac𝐻,𝑟).

- By Lemma K.24(ii)–(iii), each 𝑀̄𝐻,𝑟 preserves the 𝑒pos- and 𝑒prof-channels exactly and has the same exact signalchannel transport kernel as 𝑀𝐻,𝑟. Therefore the induction is well-posed, and after 𝑘 steps we obtain macro-layers

𝑀𝐻,1,…,𝑀𝐻,𝑘, 𝑀̄𝐻,1,…,𝑀̄𝐻,𝑘−1, all preserving the positional and profile channels and having exact signal transport kernels 𝒯𝑢𝑀

(𝑖,𝑗) = 𝐷mac𝑢 ,𝑟(𝑖)1[𝑖 = 𝑗] + 𝐾mac𝑢 ,𝑟(𝑖,𝑗)1[𝑗 < 𝑖], with uniform bounds

𝐻,𝑟

1 ≤ 𝐷mac𝑢 ,𝑟(𝑖) ≤ 𝑑mac+ , 𝑎−mac(𝑖 + 1)−𝛽 ≤ 𝐾mac𝑢 ,𝑟(𝑖,𝑗) ≤ 𝑎+mac(𝑖 + 1)−𝛽 (𝑗 < 𝑖).

𝑀𝐻,𝑟+1 = 𝑀𝐻,𝑟+1 ∘ Πsrc (𝑟 = 1,…,𝑘 − 1), hence the actual network from the theorem statement satisfies

Moreover, by Lemma K.24(i),

∗,𝜀𝐻 ∘ 𝑄𝐻 ∘ 𝑃𝐻 = 𝐺̂𝐻,𝜏

∘ 𝑅𝐻, where

𝐺𝐻,𝜏

= 𝑀𝐻,𝑘 ∘ ⋯ ∘ 𝑀𝐻,1 ∘ 𝑆𝐻,𝜏

∗

∗

𝐺̂𝐻,𝜏

∶= 𝑀𝐻,𝑘 ∘ 𝑀̄𝐻,𝑘−1 ∘ ⋯ ∘ 𝑀̄𝐻,1 ∘ 𝑆𝐻,𝜏

∗,𝜀𝐻, 𝑅𝐻 ∶= 𝑄𝐻 ∘ 𝑃𝐻.

- By Lemma K.24(iii), each 𝑀̄𝐻,𝑟 has the same signal-channel transport kernel as the corresponding 𝑀𝐻,𝑟, so all of the above kernel bounds remain unchanged.

Step 5: identify the score with the transport kernel. Take the normalized probes in Definition 5 to be

𝑐(𝐻,𝜏∗) ∶= 𝑒sig, 𝜌(𝐻,𝜏

∗)

𝑡 ∶= 𝑒sig (0 ≤ 𝑡 ≤ 𝐿𝐻). These are independent of 𝑥, common to all source indices 𝜏, and satisfy

‖𝑐(𝐻,𝜏∗)‖2 = 1, ‖𝜌(𝐻,𝜏

∗)

𝑡 ‖2 = 1.

Set

𝑅𝐻 ∶= 𝑄𝐻 ∘ 𝑃𝐻. By Step 1 and Step 2, 𝑅𝐻 is signal-transparent along 𝑒sig over

𝐸ctrl ∶= span{𝑒pos,𝑒prof}

on 𝒳(𝐻)0 . Fix some 𝛿∗ > 0, for example 𝛿∗ = 1, and define

𝒴𝐻 ∶= Satsig𝛿

∗

(𝑅𝐻(𝒳(𝐻)0 )).

This set is compact. Define

𝒴𝐻,0 ∶= 𝑆𝐻,𝜏

∗,𝜀𝐻(𝒴𝐻). By Lemma K.14, there exists a finite 𝛿𝐻,0 such that

𝒴𝐻,0 ⊂ Satsig𝛿

𝐻,0

(𝒦_setmac𝐻,0).

For 𝑟 = 1,…,𝑘 − 1, define inductively

𝒴𝐻,𝑟 ∶= 𝑀̄𝐻,𝑟(𝒴𝐻,𝑟−1).

- By Lemma K.24(iv), there exists a finite 𝛿𝐻,𝑟 such that

∗

𝒴𝐻,𝑟 ⊂ Satsig𝛿

(𝒦_setmac𝐻,𝑟), 𝑟 = 1,…,𝑘 − 1.

𝐻,𝑟

By Corollary K.10, the selector 𝑆𝐻,𝜏

∗,𝜀𝐻 has signal-blind exact scalar transport along 𝑒sig over 𝐸ctrl = span{𝑒pos,𝑒prof}

on 𝒴𝐻. For each 𝑟 = 1,…,𝑘 − 1, Lemma K.24(v) shows that 𝑀̄𝐻,𝑟 has signal-blind exact scalar transport along 𝑒sig over the same control subspace on 𝒴𝐻,𝑟−1. Finally, since

𝒴𝐻,𝑘−1 ⊂ Satsig𝛿

(𝒦_setmac𝐻,𝑘−1),

𝐻,𝑘−1

Corollary K.23 implies that the final macro-layer 𝑀𝐻,𝑘 has signal-blind exact scalar transport along 𝑒sig over the same control subspace on 𝒴𝐻,𝑘−1, with the same kernel 𝒯𝑢𝑀

as on 𝒦_setmac𝐻,𝑘−1. Repeated application of Lemma K.8(ii) therefore yields that the full post-preparatory stack

𝐻,𝑘

𝐺̂𝐻,𝜏

= 𝑀𝐻,𝑘 ∘ 𝑀̄𝐻,𝑘−1 ∘ ⋯ ∘ 𝑀̄𝐻,1 ∘ 𝑆𝐻,𝜏

∗,𝜀𝐻 has signal-blind exact scalar transport along 𝑒sig over

∗

𝐸ctrl = span{𝑒pos,𝑒prof} on 𝒴𝐻, with transport kernel

𝒯𝑢𝐺̂

(𝑡,𝜏).

𝐻,𝜏∗

𝑅 = 𝑅𝐻, 𝐵 = 𝐺̂𝐻,𝜏

, 𝒦_set = 𝒳(𝐻)0 .

Hence Lemma K.9 applies with

∗

Therefore, for every 𝑥 ∈ 𝒳(𝐻)0 and every 0 ≤ 𝜏 ≤ 𝑡 ≤ 𝐿𝐻,

𝜕𝐺𝐻,𝜏

∗,𝑡(𝑥) 𝜕𝑥𝜏

𝑒⊤sig

𝑒sig = 𝒯𝑅

𝐺̂𝐻,𝜏∗ (𝑡,𝜏). By our choice of score channels,

𝐻(𝑥)

𝐺̂𝐻,𝜏∗ (𝑡,𝜏). Set

𝑡,𝜏 (𝑥)𝑒sig = 𝒯𝑅

𝑡,𝜏 (𝑥) = (𝜌(𝐻,𝜏

𝑡,𝜏 (𝑥)𝑐(𝐻,𝜏∗) = 𝑒⊤sig𝐽𝐺𝐻,𝜏

𝑡 )⊤𝐽𝐺𝐻,𝜏

S(𝐻,𝜏∗)

𝐻(𝑥)

∗)

∗

∗

𝑢 ∶= 𝑅𝐻(𝑥).

𝑡 = 𝜏∗ + ℓ, ℓ ≥ 𝑘. Expand the kernel product along the intermediate states. Writing

- Step 6: lower-bound the balanced paths. Fix

𝑢(0) ∶= 𝑢, 𝑢(𝑟) ∶= 𝑀̄𝐻,𝑟 ∘ ⋯ ∘ 𝑀̄𝐻,1 ∘ 𝑆𝐻,𝜏

∗,𝜀𝐻(𝑢) (1 ≤ 𝑟 ≤ 𝑘 − 1), one has

𝒯𝑢𝑀(𝑘−2)̄

⋯𝒯𝑢𝑀(0)̄

𝒯𝑢𝑆

.

𝒯𝑢𝐺̂

= 𝒯𝑢𝑀(𝑘−1)

𝐻,𝑘

𝐻,𝜏∗,𝜀𝐻

𝐻,𝑘−1

𝐻,1

𝐻,𝜏∗

Since every factor preserves the control channels exactly and its kernel depends only on the control stream, all intermediate control streams equal that of 𝑢. Hence the same pathwise kernel bounds apply throughout. Moreover, by Lemma K.24,

𝒯𝑢𝑀(𝑟−1)̄

(𝑖,𝑗) (𝑟 = 1,…,𝑘 − 1). Consider the family of paths that use all 𝑘 macro-layers as jumps and whose jump times are balanced:

(𝑖,𝑗) = 𝒯𝑢𝑀(𝑟−1)

𝐻,𝑟

𝐻,𝑟

2ℓ 𝑘

ℓ 2𝑘

≤ 𝑖𝑟 − 𝑖𝑟−1 ≤

.

𝜏∗ = 𝑖0 < 𝑖1 < ⋯ < 𝑖𝑘 = 𝑡,

For each such path, the selector contributes at least 12, and each jump contributes at least

𝑎−mac(𝑖𝑟 + 1)−𝛽. Hence

- 1

- 2

𝑘

𝒯𝑢𝐺̂

(𝑡,𝜏∗) ≥

(𝑎−mac)𝑘 ∑

∏

(𝑖𝑟 + 1)−𝛽.

𝜏∗=𝑖0<⋯<𝑖𝑘=𝑡 balanced

𝑟=1

𝐻,𝜏∗

𝒯𝑢𝐺̂

(𝑡,𝜏∗) ≥ 𝑐good(1 + ℓ)𝑘(1−𝛽)−1.

By Lemma K.25,

𝐻,𝜏∗

- Step 7: handle small lags. There are only finitely many pairs (𝜏∗,ℓ) with 0 ≤ 𝜏∗ ≤ 𝜏max, 1 ≤ ℓ < 𝑘.

For each such pair, choose the path that jumps in the first ℓ macro-layers and then propagates diagonally. Since all indices lie in the finite set {0,…,𝜏max + 𝑘 − 1}, the corresponding exact path weight is bounded below by a positive constant depending only on (𝑘,𝛽,𝜏max). Therefore there exists

𝑐small > 0 such that

𝒯𝑢𝐺̂

𝐻,𝜏∗

(𝜏∗ + ℓ,𝜏∗) ≥ 𝑐small (1 ≤ ℓ < 𝑘).

Combining the large- and small-lag cases, there exists 𝑐sig > 0 such that for all 1 ≤ ℓ ≤ 𝐻,

𝒯𝑢𝐺̂

𝐻,𝜏∗

(𝜏∗ + ℓ,𝜏∗) ≥ 𝑐sig(1 + ℓ)𝜈𝑘(𝛽), 𝜈𝑘(𝛽) = 𝑘(1 − 𝛽) − 1.

- Step 8: suppress the competitors. Apply Lemma K.26 to the selector-plus-macro transport kernel. By Lemma K.24(iii), each projected macro-layer 𝑀̄𝐻,𝑟 has exactly the same signal-channel transport kernel as the corresponding macro-layer 𝑀𝐻,𝑟, so the lemma applies verbatim to the post-preparatory stack

𝐺̂𝐻,𝜏

= 𝑀𝐻,𝑘 ∘ 𝑀̄𝐻,𝑘−1 ∘ ⋯ ∘ 𝑀̄𝐻,1 ∘ 𝑆𝐻,𝜏

∗,𝜀𝐻. Since the exact transport coeﬀicient equals the Jacobian score coeﬀicient on the signal channel, ∑

∗

∣S(𝐻,𝜏∗)

∣𝒯𝑢𝐺̂

𝑡,𝜏 (𝑥)∣ = ∑

(𝑡,𝜏)∣ ≤ 𝐶comp𝜀𝐻(1 + ℓ)𝑘(1−𝛽).

0≤𝜏<𝑡 𝜏≠𝜏∗

0≤𝜏<𝑡 𝜏≠𝜏∗

𝐻,𝜏∗

Choose 𝑐0 > 0 small enough that

- 1

- 2

𝑐sig(1 + ℓ)𝜈𝑘(𝛽) (1 ≤ ℓ ≤ 𝐻). Then

𝐶comp𝜀𝐻(1 + ℓ)𝑘(1−𝛽) ≤

- 1

- 2

M(𝐻,𝜏∗) 𝜏∗+ℓ,𝜏∗(𝑥) ≥

𝑐sig(1 + ℓ)𝜈𝑘(𝛽). So we may take

- 1

- 2

𝑐− ∶=

𝑐sig.

##### Step 9: anchor bounds. At ℓ = 1,

𝜏∗+1,𝜏∗(𝑥) ≥ 𝑐−(1 + 1)𝜈𝑘(𝛽) = 2𝜈𝑘(𝛽)𝑐−. Hence we may take

M(𝐻,𝜏∗)

𝑚− ∶= 2𝜈𝑘(𝛽)𝑐− > 0. For the anchor upper bound, note first that

𝜏∗+1,𝜏∗(𝑥)∣. By Step 5,

𝜏∗+1,𝜏∗(𝑥) ≤ ∣S(𝐻,𝜏∗)

M(𝐻,𝜏∗)

𝐺̂𝐻,𝜏∗ (𝜏∗ + 1,𝜏∗). Since the selector is diagonal, any path from 𝜏∗ to 𝜏∗ + 1 through

𝜏∗+1,𝜏∗(𝑥) = 𝒯𝑅

S(𝐻,𝜏∗)

𝐻(𝑥)

𝐺̂𝐻,𝜏

= 𝑀𝐻,𝑘 ∘ 𝑀̄𝐻,𝑘−1 ∘ ⋯ ∘ 𝑀̄𝐻,1 ∘ 𝑆𝐻,𝜏

∗,𝜀𝐻

∗

must contain exactly one off-diagonal jump, and that jump must occur in one of the 𝑘 macro-layers. Therefore 𝒯𝑅

𝐺̂𝐻,𝜏∗ (𝜏∗ + 1,𝜏∗)

𝐻(𝑥)

𝑘

= 𝐷sel𝑢 (𝜏∗)

∑

(∏

𝐷mac𝑢 ,𝑞(𝜏∗))𝐾mac𝑢 ,𝑟(𝜏∗ + 1,𝜏∗)(∏ 𝑞>𝑟

𝐷mac𝑢 ,𝑞(𝜏∗ + 1)),

𝑞<𝑟

𝑟=1

where 𝑢 = 𝑅𝐻(𝑥). Using

𝐷sel𝑢 (𝜏∗) ≤ 2, 𝐷mac𝑢 ,𝑞(𝑖) ≤ 𝑑mac+ , 𝐾mac𝑢 ,𝑟(𝜏∗ + 1,𝜏∗) ≤ 𝑎+mac(𝜏∗ + 2)−𝛽 ≤ 𝑎+mac, we obtain

∣𝒯𝑅

𝐺̂𝐻,𝜏∗ (𝜏∗ + 1,𝜏∗)∣ ≤ 2𝑘 (𝑑mac+ )𝑘−1𝑎+mac. Hence one may take

𝐻(𝑥)

𝑚+ ∶= 2𝑘 (𝑑mac+ )𝑘−1𝑎+mac, which is independent of 𝐻, 𝜏∗, and 𝑥. Consequently,

𝜏∗+1,𝜏∗(𝑥) ≤ 𝑚+.

M(𝐻,𝜏∗)

𝜈𝑘(𝛽) = 𝑘(1 − 𝛽) − 1.

This verifies Definition 5. The sign classification follows immediately from the sign of

| |
|---|

