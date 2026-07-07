## NITP: Next Implicit Token Prediction for LLM Pre-training

Xiangdong Zhang12 Debing Zhang2 Shaofeng Zhang3 Xiaohan Qin12 Yu Cheng4 Junchi Yan1

# arXiv:2605.24956v1[cs.CL]24May2026

### Abstract

[Figure 1]

[Figure 2]

Standard Next-Token Prediction (NTP) supervises language models solely through discrete labels in the output logit space. We argue that this sparse, one-hot supervision leaves the latent representation space under-constrained, allowing hidden states to drift into degenerate and anisotropic configurations that limit generalization. To address this issue, we propose Next Implicit Token Prediction (NITP), which augments discrete prediction with dense, continuous supervision directly in the representation space. NITP requires the model to predict the implicit semantic content of the next token, using shallow-layer representations from the same model as stable self-supervised targets. Theoretically, we show that NITP regularizes the optimization landscape by mitigating under-constrained degrees of freedom and enforcing a compact, structured representation geometry. Empirically, across dense and MoE models ranging from 0.5B to 9B parameters, NITP consistently improves downstream performance with negligible computational overhead. Notably, on the 9B MoE model, NITP achieves a 5.7% absolute improvement on MMLU-Pro, along with gains of 6.4% on C3 and 4.3% on CommonsenseQA, with ∼2% additional training FLOPs and no additional inference cost. Our implementation is available at https://github.com/aHapBean/NITP.

(a) Effective rank (b) Cosine similarity

[Figure 3]

[Figure 4]

###### +3.32

###### +3.38

(c) Avg performance (MoE)

(d) Avg performance (Dense)

Figure 1. Top: Representation geometry of the last hidden states under NTP and NITP. Bottom: Average downstream performance of 9B MoE and 2B dense models (details in Appendix B).

Team et al., 2025; He & Su, 2025; Chen et al., 2024). By maximizing the likelihood of the next token over massive corpora, this paradigm learns general-purpose representations that support a wide range of downstream tasks (Guo et al., 2025). Although subsequent fine-tuning stages further adapt these models, their effectiveness critically depends on the representations learned during base pre-training (Yue et al., 2025). As a result, the pre-training objective plays a central role in determining the capability of LLMs.

Despite its empirical success, standard NTP provides supervision only through discrete one-hot targets in the output token space. While gradients propagate to hidden states via the output projection, the objective primarily constrains representations along the target logit direction, leaving many degrees of freedom in the latent space that remain weakly constrained. This raises a fundamental question: Does nexttoken prediction alone sufficiently supervise the geometry of hidden representations?

### 1. Introduction

Large language models (LLMs) have achieved remarkable success through large-scale pre-training with the NextToken Prediction (NTP) (Liu et al., 2024b; Yang et al., 2025;

1School of AI, Shanghai Jiao Tong University 2Xiaohongshu Inc. 3University of Science and Technology of China 4The Chinese University of Hong Kong. Correspondence to: Junchi Yan <yanjunchi@sjtu.edu.cn>.

Prior work has identified a phenomenon known as representation degeneration, where likelihood-based training drives learned embeddings to collapse into a narrow, anisotropic cone (Ethayarajh, 2019; Wang et al., 2020; Barbero et al., 2024; Gao et al., 2019). Such geometric collapse limits the expressive capacity of representations and has been linked to

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

degraded generalization on downstream tasks (Ethayarajh, 2019; Zhao et al., 2024). To examine this issue, we track the geometric evolution of last hidden states throughout training. We utilize two established metrics: (1) Effective Rank (Roy & Vetterli, 2007), which quantifies the effective dimensionality of the utilized subspace, and (2) Average Cosine Similarity between paired tokens within training batches, serving as a proxy for the global anisotropy of the representation space. As illustrated in Figure 1, for NTP, the effective rank diminishes rapidly while cosine similarity increases, indicating that hidden representations drift toward degenerate, anisotropic configurations (detailed analysis in Appendix D). This confirms that standard NTP is geometrically under-constrained: it permits representations to sacrifice expressiveness and distinctiveness without penalty, as long as the next token is correctly predicted.

We attribute this behavior to the under-constrained nature of hidden representations under NTP, where the lack of geometric supervision allows representations to drift into low-expressivity configurations that may harm performance (Ethayarajh, 2019). To bridge this gap, we propose Next Implicit Token Prediction (NITP), an auxiliary pretraining objective that augments discrete next-token prediction with continuous supervision in the representation space. Instead of predicting only the token identity, NITP tasks the model with predicting an implicit token, defined as a latent semantic representation of the next token. We construct this supervision signal from the model’s own shallow layers, which empirically preserve richer lexical and local semantic content than deeper, task-specialized layers (Liu et al., 2024c; Skean et al., 2025). This design enables scalable, self-supervised targets without introducing external encoders or additional annotations. As evidenced by Figure 1, NITP effectively mitigates representation degeneration, maintaining higher effective rank and preventing excessive anisotropy compared to standard NTP. Our main contributions are summarized as follows:

- 1) We analyze the training objective of NTP and show it provides weak supervision for hidden representations, allowing them to drift toward degenerate space.
- 2) We propose Next Implicit Token Prediction (NITP), a novel objective that augments next-token prediction with dense, continuous supervision in the latent space.
- 3) We introduce a self-supervised design for NITP by using shallow-layer representations as implicit tokens to serve as prediction targets, enabling semantically rich supervision.
- 4) We provide theoretical analysis and empirical evidence showing that NITP improves representation geometry and

consistently enhances downstream performance.

### 2. Related Work

Pre-training objectives for large language models. The next-token prediction (NTP) objective is the dominant pretraining paradigm for modern large language models (Liu et al., 2024b; Yang et al., 2025), and has been shown to scale effectively with data and model size (Mann et al., 2020; Hoffmann et al., 2022; Kaplan et al., 2020). By optimizing the likelihood of the next discrete token given its context, NTP enables models to learn general-purpose representations that support a wide range of downstream tasks. To extend the predictive horizon beyond a single token, several works have explored multi-token prediction (MTP) (Gloeckle et al., 2024; Samragh et al., 2025; Liu et al., 2024b) and related objectives (Mahajan et al., 2025; Tack et al., 2025; Zuhri et al., 2025; Liu et al., 2025). These methods introduce auxiliary output heads to predict multiple future tokens or compressed summaries of future context, with the goal of improving planning or long-horizon modeling (Mahajan et al., 2025). Despite their differences, these approaches operate primarily in the discrete token space and rely on token-level supervision derived from one-hot targets. In contrast, our proposed Next Implicit Token Prediction departs from prior works by augmenting standard NTP with dense, continuous supervision at the representation level.

Representation-level supervision. Beyond discrete tokenlevel objectives, prior work has explored supervising models directly in continuous representation spaces (Bengio et al., 2013; Oord et al., 2018; Jiao et al., 2020). A prominent direction is layer-wise distillation, which aligns intermediate representations between models to transfer structural or task-relevant knowledge (Romero et al., 2014; Sun et al.,

- 2019; Jiao et al., 2020; Liang et al., 2023). This paradigm has recently been extended to generative LLMs (Sun et al.,
- 2020b; Gu et al., 2024a; Xia et al., 2023). Notably, Sheared LLaMA (Xia et al., 2023) aligns hidden states to accelerate knowledge recovery after structured pruning. Another line of research focuses on self-supervised consistency, encouraging agreement between latent representations across time or views without discrete labels, e.g., Contrastive Predictive Coding and BYOL (Oord et al., 2018; Grill et al., 2020), as well as self-distillation methods (Zhang et al., 2019; Lee et al., 2022). NITP differs from these approaches in both motivation and formulation: motivated by the underconstrained hidden states during LLM pre-training, NITP adopts an autoregressive objective in the latent space that requires the model to predict the semantic representation of the next token.

### 3. Methodology

In this section, we present Next Implicit Token Prediction (NITP), a representation-level objective that explicitly supervises hidden states to complement standard Next-Token

contenders. Thus, the objective effectively constrains ht primarily along the direction of the target embedding wx

- 1 2 3 4 5

allo layers

Deep layers

- 2 3 4 5 6 X

Discrete one-hot tokens Implicit tokens

.

X X

t+1

Next token prediction loss (cross-entropy loss)

Last hidden states

#### 3.2. The Geometric Blind Spot of Next-Token Prediction

Padding tokens

PD

unembedding layer

Projector

The gradient dominance described above creates a structural “blind spot” in the optimization landscape: the loss becomes near-invariant to perturbations within the vast subspace orthogonal to the target direction. Theoretically, this implies that a wide range of geometrically distinct latent states can yield identical token-level likelihoods.

1 2 3 4 5

Next implicit token prediction loss (cosine loss)

Stop gradient

PD

2 3 4 5

In practice, this structural freedom does not yield diverse features but instead leads to representation degeneration (Ethayarajh, 2019; Wang et al., 2020; Barbero et al., 2024), where hidden states drift into a low-dimensional, anisotropic cone without penalty. We empirically confirm this collapse by tracking the geometric evolution of hidden states in Figure 1. Under standard NTP, we observe a rapid deterioration in Effective Rank (Roy & Vetterli, 2007) coinciding with a sharp rise in global Cosine Similarity. These trends indicate that the model sacrifices semantic richness for discriminative efficiency, compressing representations even as token-level accuracy improves.

Temporal shift (shift left)

1

Shallow layers

Drop

Figure 2. Overview of NITP. Next Implicit Token Prediction supervises hidden states by predicting temporally shifted implicit tokens, i.e. shallow-layer representations, and is jointly optimized with the standard next-token prediction objective. These shallow representations act as semantics anchors (see Section 3.3). The last-layer hidden states are projected to match the implicit targets using a cosine similarity loss.

Prediction (NTP), as illustrated in Figure 2. By encouraging the model to anticipate the semantic content of the next token in latent space, NITP regularizes representation geometry beyond discrete token-level supervision.

We first revisit the standard NTP paradigm and its limitations in supervising latent representations. We then introduce NITP, which trains the last hidden state ht to predict an implicit token—the shallow-layer representation of the (t+1)-th token with stop-gradient—using a cosine similarity loss jointly optimized with the NTP objective. As the implicit targets are self-generated during the forward pass, NITP incurs negligible computational overhead and scales naturally with model size (see Section 4 for results).

#### 3.1. Preliminary: Next-Token Prediction

From the perspective of the NTP objective, these degenerate configurations are valid as long as they preserve the correct logit ranking, yet they severely limit the expressive capacity of the latent space. Crucially, such a drop in representation expressiveness has been linked to degraded generalization on downstream tasks (Ethayarajh, 2019; Zhao et al., 2024). This observation underscores a critical limitation: NTP defines what to predict, but fails to supervise how predictions are represented, necessitating explicit guidance to enforce a semantically structured geometry. To this end, we propose Next Implicit Token Prediction (NITP), complementing NTP by explicitly supervising the latent representation space. Instead of constraining only discrete token identities, NITP requires the model to predict the implicit semantic representation of the next token, thereby directly supervising the geometry and evolution of hidden states.

Given a token sequence x = {x1,...,xT}, let ht ∈ Rd denote the final hidden state at step t, serving as the contextual representation. Given an unembedding matrix W ∈ R|V|×d, the standard next-token prediction (NTP) objective minimizes the negative log-likelihood of the target token xt+1:

exp(h⊤t wx

) j∈V exp(h⊤

, (1)

LNTP = −Ex log

t+1

t wj)

where wj denotes the embedding vector for token j. We omit layer normalization terms for brevity. While the NTP loss theoretically involves all vocabulary items via the normalization term, the gradient signal is empirically dominated by the target token xt+1 and a few high-probability

#### 3.3. Shallow Layers as Intrinsic Semantic Anchors

To construct an effective supervision target for NITP, we first clarify the semantic role of the last hidden state ht. Prior works (Pal et al., 2023; Liu et al., 2024c) suggest that ht contains recoverable semantic information about the next token. Building on this view, we consider an idealized regime that the model predicts the correct next token xt+1 with high confidence. In this setting, ht serves as the intermediate representation through which the prediction is realized, and thus is expected to encode semantic information relevant to xt+1 in a form that supports the prediction task.

This observation naturally raises the question: what con-

stitutes an appropriate supervision target for ht? A naive choice would be the static embedding of xt+1. However, static embeddings suffer from polysemy—the same token (e.g., “bank”) can correspond to vastly different meanings depending on the context. To account for this ambiguity, a suitable supervision target should be a contextualized representation that captures the precise semantics of the token conditioned on the context x<t+1.

While external encoders (Raffel et al., 2020; Zhang et al., 2025) could provide such contextualized targets, they introduce prohibitive computational overhead and domain shifts. Instead, we propose an efficient, self-supervised solution: utilizing the model’s own shallow layers as intrinsic semantic anchors. Our rationale is threefold:

- • Semantic richness: Recent analysis (Skean et al., 2025) reveals that shallow layers act as powerful contextual encoders, resolving lexical ambiguity while preserving fine-grained semantic details. Studies (Liu et al., 2024c) explicitly demonstrate that semantic richness peaks in these early layers and degrades in deeper layers, as the model progressively discards details to focus solely on the sparse discriminative features required for NTP.
- • Stability: Transformer training typically exhibits a bottom-up convergence pattern (Skean et al., 2025), where layers closer to the input stabilize much faster than deeper layers. This ensures that the generated implicit tokens are stable relative to the deep predictive state ht.
- • Efficiency: Utilizing shallow layers incurs negligible computational overhead. Unlike external models, the targets are extracted directly from intermediate activations computed during the standard forward pass.

By treating the semantic-rich, contextualized shallow representation of xt+1 as the next implicit token, we impose a fidelity constraint that forces the deep state ht to maintain a structured geometry capable of containing full semantic details, thereby preventing representation degeneration.

#### 3.4. Next Implicit Token Prediction

We introduce Next Implicit Token Prediction (NITP), a continuous representation-level supervision objective designed to complement the discrete next-token prediction. While NTP predicts the discrete identity of the next token, NITP predicts its semantic essence in the latent space.

We formally distinguish between the predictive state ht and the prediction target zt+1. We define the implicit token zt+1 as the dense, contextualized semantic representation of the next token, derived from the model’s own shallow layers. The NITP objective forces the final hidden state to align with this implicit token, thereby enforcing a structured and semantic representation geometry.

Implicit token construction. For a given input sequence x, let Eshallow(·) denote the forward pass through the first k layers of the model (see Section 4.3 for details on the choice of k). We construct the implicit token zt+1, which serves as the optimization target, by extracting the hidden representation at the future time step t + 1:

zt+1 = sg Eshallow(x≤t+1)(t+1) ∈ Rd, (2)

where (·)(t+1) denotes selecting the vector at position t + 1, and sg[·] indicates the stop-gradient operator. Crucially, by sourcing zt+1 from shallow layers, we capture the contextualized semantics of the next token while avoiding the representation collapse often observed in deeper layers, making it a stable and semantically rich anchor.

The prediction objective. NITP tasks the last hidden state ht with predicting the implicit token zt+1. To bridge the potential distributional gap between the deep predictive state and the shallow target, we employ a projection head P(·) (e.g., an MLP). The objective minimizes the geometric misalignment using cosine similarity:

P(ht)⊤zt+1 ∥P(ht)∥2 · ∥zt+1∥2

. (3)

LNITP(θ) = 1 −

We favor cosine similarity over other loss functions to enforce scale invariance as demonstrated in Section 4.3.

Combined training. The overall pre-training framework acts as a dual-supervision mechanism, combining the discrete NTP loss with the continuous NITP supervision:

Ltotal(θ) = LNTP(θ) + λLNITP(θ), (4)

where λ > 0 is a hyperparameter controlling the weight of the implicit prediction. By requiring the model to predict both the discrete identity and the continuous implicit token, NITP explicitly regularizes the hidden representations toward a more semantically structured feature space.

3.5. Theoretical Analysis: Regularizing the Semantic Manifold

Standard Next-Token Prediction (NTP) supervises the hidden state ht primarily by maximizing its dot product with the target token embedding wx

. However, this objective is invariant to perturbations in the subspace orthogonal to the informative directions of the unembedding matrix. Mathematically, this manifests as a rank-deficient Hessian matrix with respect to the semantic geometry, allowing hidden states to drift into degenerate configurations without penalty—a phenomenon closely linked to representation anisotropy (Ethayarajh, 2019).

t+1

We formally analyze how Next Implicit Token Prediction (NITP) mitigates this by regularizing the angular geometry

of the latent space. Unlike NTP, which allows unconstrained drift in null directions, NITP acts as a geometric anchor, introducing positive curvature to the entire semantic subspace.

Setup. Let h ∈ Rd be the last hidden state used for implicit prediction. In practice, the NITP objective utilizes a non-linear projection P(·). To characterize the regularization effect, we adopt the Generalized Gauss-Newton (GGN) approximation (Martens, 2020), which models the Hessian with respect to h as JP⊤HNITPJP, where HNITP denotes the Hessian of the NITP loss. Crucially, this quadratic form preserves the spectral properties (specifically positive definiteness) of HNITP, provided that the projector is locally well-conditioned (i.e., JP is full-rank). This implies that the regularization is fundamentally governed by the structure of the loss function. Therefore, we simplify the analysis by examining the NITP objective directly on h to isolate its intrinsic geometric properties from the architectural details. We decompose variations in h into a radial component (change in norm) and an angular component (change in direction). Let r = ∥h∥2 and u = h/r. Let z ∈ Rd be the fixed implicit target derived from shallow layers, with normalized direction v = z/∥z∥2. The NITP objective is defined as LNITP(h) = 1 − cos(h,z) = 1 − u⊤v.

Lemma 3.1 (Hessian of the NITP Objective). Let s := u⊤v be the cosine alignment. The Hessian of LNITP with respect to h is given by:

1 r2

s(I − uu⊤) + (uA⊤ + Au⊤) , (5)

HNITP(h) =

where A = v − su is the tangential difference vector. Near convergence, where the representation aligns with the target (s → 1 and implies A → 0), the Hessian simplifies to:

1 r2

(I − uu⊤) P⊥u

. (6)

HNITP(h) ≈

Here, P⊥u denotes the projection matrix onto the subspace orthogonal to h (the tangent space of the hypersphere).

Proof. See Appendix A.

Remark: necessity of semantic targets. The curvature guarantee relies on high alignment (s → 1). This justifies the choice of predictable semantic targets derived from shallow layers over arbitrary vectors, which would yield weak alignment (s ≈ 0). Empirically, we observe that the NITP loss, i.e. 1 − s consistently converges below 0.1 (e.g., 0.04 for the 9B MoE), confirming that the optimization stably operates in the region where the Hessian is positive definite and geometric regularization is effective.

Analysis of spectral properties. Lemma 3.1 reveals two critical properties of NITP:

- • Radial null space: u⊤HNITPu = 0. NITP imposes no curvature on the vector norm, allowing the NTP objective to freely adjust the scale (r)—for instance, to optimize softmax temperature—without interference.
- • Angular spectral lifting: For any semantic deviation vector w orthogonal to the current state (w ⊥ u), we have strictly positive curvature:

1

w⊤HNITPw ≈

r2∥w∥22 > 0. (7) Theorem 3.2 (Null Space Mitigation in Semantic Subspace). Consider the total loss Hessian Htotal = HNTP + λHNITP. Let Nsem be the semantic null space of NTP, defined as the set of directions w ⊥ u such that w⊤HNTPw ≈ 0. With the addition of NITP, for any w ∈ Nsem, the curvature becomes strict:

λ r2∥w∥22 > 0. (8)

w⊤Htotalw ≈ 0 +

See detailed analysis at Appendix A.

Interpretation. Theorem 3.2 demonstrates that NITP performs spectral lifting on the angular manifold. While standard NTP leaves vast degrees of freedom orthogonal to the target logit unpenalized, NITP effectively regularizes these under-constrained directions by introducing positive curvature. This mitigates the “flat valleys” in the optimization landscape responsible for semantic drift, forcing the model to maintain robust representations aligned with the semantic anchors derived from shallow layers.

### 4. Experiments

In this section, we conduct a comprehensive empirical evaluation to assess the effectiveness of Next Implicit Token Prediction (NITP) in terms of downstream performance. We train both dense and mixture-of-experts models from scratch, covering model scales from hundreds of millions to billions of parameters, and assess performance on a broad suite of knowledge and reasoning benchmarks. Additionally, we conduct ablation studies to examine the effect of key design choices in NITP and to verify its effectiveness.

#### 4.1. Experiment Setups

Training settings. We evaluate NITP on both Mixture-ofExperts (MoE) (Shazeer et al., 2017) and dense language models to assess its generality across model architectures and scales. For MoE models, we adopt the DeepSeekV2 (Liu et al., 2024a) architecture employing 144 experts with top-8 routing, where each expert is implemented as a SwiGLU-based (Shazeer, 2020) FFN. We scale this architecture up to a total parameter count of 9B, corresponding to

- Table 1. Main results. Performance comparison between NTP and NITP. We categorize benchmarks into Knowledge & Language (5 tasks) and Reasoning & Problem Solving (8 tasks). Best results are bolded. ‘A’ denotes the number of activated parameters per token, and the number before ‘A’ represents the total parameters in MoE. Abbreviations: M-Pro denotes MMLU-Pro, LAMB. denotes LAMBADA, CSQA denotes CommonsenseQA, COPA denotes Balanced COPA.

Knowledge & Language Understanding Reasoning & Problem Solving

Model Method

Avg. MMLU M-Pro C-Eval Xiezhi LAMB. BBH ARC-C CSQA COPA C3 AGIEval GSM8k LCBench

###### NTP 31.05 7.14 32.31 32.13 50.36 17.70 24.74 25.38 61.20 32.21 24.57 6.06 1.56 26.65 NITP 31.68 7.47 33.16 31.22 50.52 19.11 29.20 26.61 62.90 29.69 26.69 6.44 2.26 27.46

1.9bA0.3b

###### NTP 34.60 11.00 33.53 36.53 55.45 21.92 32.65 34.15 65.60 39.06 26.67 12.81 3.82 31.37 NITP 37.37 12.29 33.12 39.61 55.45 26.14 37.11 37.92 65.60 44.38 27.76 14.40 4.17 33.49

3bA0.5b

###### NTP 43.71 15.29 38.97 45.59 62.32 28.07 51.20 45.70 68.10 56.65 30.91 30.09 6.95 40.27 NITP 46.14 21.00 40.72 49.10 64.49 28.67 53.95 49.96 70.70 63.01 29.97 32.52 8.00 42.94

9bA1b

approximately 1B activated parameters per token. For dense models, we conduct experiments on model sizes ranging from 0.5B to 3B parameters. All models are trained from scratch with token budgets scaled according to their parameters. Specifically, the 9B MoE model is trained on 330B high-quality tokens comprising a diverse mix of English, Chinese, code, mathematics, and reasoning data. The context length is fixed at 8192 tokens unless stated otherwise. Comprehensive training details are provided in Appendix B.

Evaluation. We conduct comprehensive few-shot evaluations on a diverse suite of benchmarks to assess the effectiveness of NITP relative to the standard NTP objective. We categorize these benchmarks into two primary domains:

- 1) Knowledge & Language Understanding: To evaluate broad world knowledge and multi-domain comprehension, we utilize English benchmarks MMLU (Hendrycks et al.,

2020) and MMLU-Pro (Wang et al., 2024), alongside Chinese datasets C-Eval (Huang et al., 2023) and Xiezhi (Gu et al., 2024b). We additionally include LAMBADA (Paperno et al., 2016) to test long-range dependency modeling.

- 2) Reasoning & Problem Solving: To examine logical deduction and complex problem-solving, we employ ARCChallenge (Clark et al., 2018) for scientific reasoning, together with CommonsenseQA (Talmor et al., 2019) and Balanced COPA (Kavumba et al., 2019) for commonsense reasoning. We also include BBH (Suzgun et al., 2023) and C3 (Sun et al., 2020a) to evaluate challenging multi-step inference and reading comprehension. Furthermore, we assess rigorous application skills using AGIEval (Zhong et al.,

2024) on standardized exams, GSM8k (Cobbe et al., 2021) on mathematics, and OpenCompass LCBench (Contributors, 2023) (a collection of programming problems from Leetcode weekly competitions) on code generation.

#### 4.2. Main Results

Evaluation on MoE models. We mainly evaluate the performance of the proposed NITP against the NTP baseline across three Mixture-of-Experts (MoE) scales: 1.9B (0.3B activated), 3B (0.5B activated), and 9B (1B activated). The

models were pre-trained from scratch with increasing token budgets proportional to their scale. The quantitative results are reported in Table 1. The results demonstrate that NITP consistently outperforms standard NTP across all model sizes, underscoring the method’s scalability. Specifically, NITP yields average score improvements of 0.8, 2.1, and 2.7 points for the 1.9B, 3B, and 9B models, respectively. For instance, on the 9B model, NITP achieves a 5.71% absolute improvement on MMLU-Pro (rising from 15.29% to 21.00%), a benchmark specifically designed to test robust reasoning beyond simple knowledge retrieval. Similarly, we observe remarkable boosts in reading comprehension and commonsense reasoning, with C3 scores increasing by 6.36% (56.65% to 63.01%) and CommonsenseQA by 4.26% (45.70% to 49.96%). Also, on symbolic and math tasks like ARC-C and GSM8k, NITP maintains a consistent lead.

Evaluation on dense models. To assess the generality of NITP beyond MoE architectures, we evaluate dense models ranging from 0.5B to 3B parameters on seven representative benchmarks, including MMLU, C-Eval, BBH, ARC-C, C3, AGIEval, and LCBench. The quantitative results are reported in Table 2. The results demonstrate that NITP is architecture-agnostic, consistently outperforming the NTP baseline across all model scales. On the 0.5B model, NITP induces notable improvements; specifically, C3 scores surge by 3.89 points, and the average score increases from 24.42% to 25.44%. For the 2B model, NITP improves the average score by 1.79 points (31.91% to 33.70%), with notable gains on C3 (+4.16) and MMLU (+2.18). Similarly, on the 3B model, NITP achieves an average improvement of 1.35 points, accompanied by substantial gains on AGIEval and C3. Overall, these results confirm that NITP consistently enhances representation quality across diverse tasks, independent of model architecture such as MoE or dense.

Collectively, these empirical results validate the core design philosophy of NITP. By augmenting discrete prediction with semantic-rich continuous supervision, NITP effectively constrains the latent geometry, preventing hidden states from drifting into degenerate configurations. This translates

- Table 2. Results for dense models. Performance comparison between NTP and NITP on seven representative benchmarks. Best results are bolded. Avg. denotes the arithmetic mean across all benchmarks.

Model Method MMLU C-Eval BBH ARC-C C3 AGIEval LCBench Avg. 0.5B

NTP 30.59 33.72 18.96 31.95 28.21 27.01 0.52 24.42 NITP 31.01 32.78 19.92 34.71 32.10 26.15 1.39 25.44

- 2B

NTP 37.96 35.67 26.59 39.52 49.26 30.51 3.83 31.91 NITP 40.14 37.81 27.33 41.92 53.42 31.66 3.65 33.70

- 3B

NTP 43.54 39.17 31.25 51.20 59.01 31.77 4.35 37.18 NITP 44.95 40.14 29.40 51.55 63.67 35.11 4.87 38.53

into consistent performance improvements across diverse architectures (MoE and Dense). Crucially, the substantial gains observed at larger scales further support the scalability of NITP, with additional larger-scale MoE results provided in Table 5. Beyond downstream benchmark accuracy, we directly evaluate the utility of the learned hidden states as frozen sentence representations in Appendix E, where NITP improves 23 out of 25 MTEB tasks while preserving validation cross-entropy loss.

[Figure 5]

Figure 3. Loss comparison between whether temporal shift or not.

Training efficiency. We analyze the computational efficiency of NITP by comparing its complexity with the standard NTP baseline. For a MoE model, the training cost is dominated by L layers of backbone blocks and the unembedding projection, scaling as O(L(d2 + kdde) + V d), where k is the number of activated experts, de is the expert dimension, and V is the vocabulary size. In contrast, the NITP objective introduces a single projection head and a cosine loss, with a total complexity of O(d2). Crucially, since the implicit targets are derived from existing intermediate activations and decoupled via a stop-gradient operator, NITP incurs no additional backbone forward passes or backward propagation through the early layers. Given that L ≫ 1, the relative complexity ensures that the extra overhead is inherently marginal. Numerical instantiation on our 9B MoE model (d = 1280,L = 24) confirms this advantage. The additional training FLOPs from both forward and backward passes account for approximately 2% of the total computation. This negligible cost ensures that NITP serves as a highly scalable supervision objective without sacrificing training throughput. See details in Appendix G.

#### 4.3. Ablation Study

We conduct ablation studies to understand how each component contributes to its effectiveness. In particular, we focus on the following aspects: 1) how to construct the implicit prediction target, including the choice of target layer and temporal shift. 2) how to design a stable and effective auxiliary objective, including the loss function and its role beyond generic regularization. Unless otherwise specified, all ablation experiments are conducted on a 3B-parameter

[Figure 6]

Figure 4. Average performance under different NITP loss.

MoE model in Table 4, trained on 200B tokens.

Importance of temporal shift. To distinguish NITP from static layer-wise alignment, we ablate the temporal structure of the implicit prediction objective. Specifically, we compare predicting the semantic representation of the next token, i.e. the next implicit token, with aligning semantic representations at the current time step. As shown in Table 3 and Figure 3, predicting the next implicit token significantly outperforms same-position alignment, while the latter leads to terrible loss and performance degradation. Notably, sameposition alignment can achieve a very low alignment loss (e.g., 0.03), yet results in substantially worse downstream performance. This indicates that simply regularizing representations or minimizing alignment loss is insufficient. Without the autoregressive temporal shift, the objective fails to provide a meaningful predictive signal of the next token and can collapse into trivial solutions. These results highlight that the effectiveness of NITP crucially depends on predicting the future semantic representation, rather than arbitrarily aligning hidden states across layers.

- Table 3. Ablation studies. We evaluate the impact of key design choices in NITP using the 3B MoE model trained on 200B tokens, including the target layer selection, temporal shift strategy, loss function, and regularization. The default NITP configuration (shallow L4, next-token target, cosine loss) achieves the best performance. L4 denotes selecting the 4th layer out of 17 layers as the target layer.

Ablation Factor Setting MMLU MMLU-Pro CSQA BBH LCBench Avg. Baseline: NTP 34.60 11.00 34.15 21.92 3.82 21.10

Deep (L14) 35.79 10.43 38.90 23.25 2.43 22.16 Middle (L8) 35.33 11.57 34.72 22.07 2.43 21.22 Shallow (L4) 37.37 12.29 37.92 26.14 4.17 23.58

Target Layer

Current-step (t → t) 33.09 8.14 29.15 20.96 2.43 18.75 Next-token (t → t + 1) 37.37 12.29 37.92 26.14 4.17 23.58

Temporal Shift

MSE Loss 32.77 10.29 30.38 21.55 1.91 19.38 Smooth L1 39.72 11.43 37.51 24.74 2.78 23.24 Distillation (KL) 34.67 11.14 38.08 26.88 3.65 22.88 Cosine Similarity 37.37 12.29 37.92 26.14 4.17 23.58

Loss Function

Generic Cosine Reg. 34.45 10.14 33.25 22.29 3.82 20.79 NITP 37.37 12.29 37.92 26.14 4.17 23.58

Regularization

Loss function. We compare the Cosine Similarity against regression-based losses (MSE, Smooth-ℓ1) and KL Divergence in Table 3. The results highlight significant stability differences. MSE leads to catastrophic optimization divergence that loss is observed to diverge temporarily during training, as its quadratic penalty amplifies the inherent scale mismatch between layers, triggering destructive gradient spikes. Though Smooth-ℓ1 maintains stability via implicit gradient clipping, it still remains unknown for larger-scale training due to its hard alignment form. KL Divergence underperforms as it treats continuous feature vectors as distributions, causing geometric distortion. Ultimately, Cosine Similarity proves to be the superior choice. Empirically, it is the most stable objective without any divergence issues and consistently delivers superior performance across tasks.

Target layer selection. A key design choice in NITP is the use of shallow-layer representations as prediction targets (implicit tokens) under the temporal-shifted prediction objective. To validate this choice, we vary the source layer of the implicit tokens across shallow, middle, and deep layers of the model. The results in Table 3 show that using shallowlayer targets consistently outperforms the alternatives. We attribute this behavior to the fact that shallow layers preserve richer lexical and local semantic structure, whereas deeper layers are more specialized for token discrimination and exhibit worse semantics (Skean et al., 2025). Among shallow layers, we further find that selecting an early layer around 20% of the total depth yields the best performance (see Tables 3 and 7), which is consistent with prior findings that semantic richness peaks in specific early layers (Skean et al., 2025; Liu et al., 2024c) and aligns with our analysis in Section 3.3.

Is NITP merely a hidden state regularization? To examine whether NITP reduces to generic hidden-state regularization, we compare it with a cosine-based baseline inspired by (Gao et al., 2019). Specifically, this baseline applies a

cosine similarity penalty between hidden states of different tokens within sequences, encouraging them to be less aligned. While this regularization constrains the geometry of hidden representations, it does not provide a prediction target or token-level supervision. In contrast, NITP introduces a prediction-aligned auxiliary objective by training the final hidden state to predict the next implicit token, which encodes context-dependent information about the upcoming token. As shown in Table 3, we find that the cosine regularization baseline yields no improvements, whereas NITP consistently improves performance, indicating that the gains stem from prediction-aligned semantic supervision rather than merely representation regularization.

NITP loss weight. We study the effect of the NITP loss weight by sweeping λ in Equation (4). As shown in Figure 4, performance consistently peaks around λ = 1.0, which emerges as the most reliable choice across settings. While the exact optimum may vary slightly across model configurations, strong performance is consistently observed for values around 1.0, indicating low sensitivity within this range. The specific loss weights used for different models are summarized in Table 4.

### 5. Conclusion

We propose Next Implicit Token Prediction (NITP), a novel pre-training objective that augments standard next-token prediction with continuous supervision in the representation space. Motivated by the observation that NTP leaves hidden representations under-constrained and prone to geometric degeneration, NITP requires the model to predict the implicit semantic representation of the next token using shallow-layer states as self-supervised targets, which we term as next implicit token. This objective explicitly constrains the free degrees of freedom ignored by token-level supervision and prevents representation degeneration. We

provide a theoretical analysis showing that NITP mitigates the optimization null space, thereby regularizing representation geometry. Experiments across both dense and MoE models demonstrate that NITP consistently improves downstream performance with minimal computational overhead.

Limitations. NITP introduces additional hyperparameters, including the target layer and NITP loss weight. While our experiments indicate that these choices are highly stable across models, further validation on more models is required to fully confirm this robustness.

### Acknowledgements

This work was in part supported by Scientific Research Innovation Capability Support Project for Young Faculty (U40) of the Ministry of Education of China (SRICSPYFZY2025019) and Xiaohongshu Inc.

### Impact Statement

This work introduces a representation-level auxiliary objective for language model pre-training, with the goal of improving optimization stability and the geometric structure of hidden representations. The proposed method modifies only the training objective and does not involve new data sources, model architectures, or deployment settings. As a result, the societal and ethical impacts of this work are expected to be aligned with those of existing large language models trained using standard next-token prediction. Although improved representation quality and training efficiency may lead to better downstream performance, this does not fundamentally change the usage scenarios or risk profile of current language modeling systems.

### References

Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y., Lebr´on, F., and Sanghai, S. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

Barbero, F., Banino, A., Kapturowski, S., Kumaran, D., Madeira Ara´ujo, J., Vitvitskyi, O., Pascanu, R., and Veliˇckovi´c, P. Transformers need glasses! information over-squashing in language tasks. Advances in Neural Information Processing Systems, 37:98111–98142, 2024.

Bengio, Y., Courville, A., and Vincent, P. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8): 1798–1828, 2013.

Chen, L., Wang, Z., Ren, S., Li, L., Zhao, H., Li, Y., Cai, Z., Guo, H., Zhang, L., Xiong, Y., et al. Next token predic-

tion towards multimodal intelligence: A comprehensive survey. arXiv preprint arXiv:2412.18619, 2024.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Contributors, O. Opencompass: A universal evaluation platform for foundation models. https://github. com/open-compass/opencompass, 2023.

Ethayarajh, K. How contextual are contextualized word representations? comparing the geometry of bert, elmo, and gpt-2 embeddings. arXiv preprint arXiv:1909.00512, 2019.

Gao, J., He, D., Tan, X., Qin, T., Wang, L., and Liu, T.-Y. Representation degeneration problem in training natural language generation models. arXiv preprint arXiv:1907.12009, 2019.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Gloeckle, F., Idrissi, B. Y., Rozi`ere, B., Lopez-Paz, D., and Synnaeve, G. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737, 2024.

Grill, J.-B., Strub, F., Altch´e, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.

- Gu, Y., Dong, L., Wei, F., and Huang, M. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, volume 2024, pp. 32694–32717, 2024a.
- Gu, Z., Zhu, X., Ye, H., Zhang, L., Wang, J., Zhu, Y., Jiang, S., Xiong, Z., Li, Z., Wu, W., et al. Xiezhi: An ever-updating benchmark for holistic domain knowledge evaluation. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 18099–18107, 2024b.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. Deepseekv3 technical report. arXiv preprint arXiv:2412.19437, 2024b.

He, H. and Su, W. J. A law of next-token prediction in large language models. Physical Review E, 112(3):035317, 2025.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556,

- 2022.

Huang, Y., Bai, Y., Zhu, Z., Zhang, J., Zhang, J., Su, T., Liu, J., Lv, C., Zhang, Y., Fu, Y., et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. Advances in Neural Information Processing Systems, 36:62991–63010, 2023.

Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., and Liu, Q. Tinybert: Distilling bert for natural language understanding. In Findings of the association for computational linguistics: EMNLP 2020, pp. 4163– 4174, 2020.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Kavumba, P., Inoue, N., Heinzerling, B., Singh, K., Reisert, P., and Inui, K. When choosing plausible alternatives, clever hans can be clever. arXiv preprint arXiv:1911.00225, 2019.

Lee, S., Kang, M., Lee, J., Hwang, S. J., and Kawaguchi, K. Self-distillation for further pre-training of transformers. arXiv preprint arXiv:2210.02871, 2022.

Liang, C., Zuo, S., Zhang, Q., He, P., Chen, W., and Zhao, T. Less is more: Task-aware layer-wise distillation for language model compression. In International Conference on Machine Learning, pp. 20852–20867. PMLR, 2023.

Liu, X., Xia, X., Zhao, W., Zhang, M., Yu, X., Su, X., Yang, S., Ng, S.-K., and Chua, T.-S. L-mtp: Leap multi-token prediction beyond adjacent context for large language models. arXiv preprint arXiv:2505.17505, 2025.

Liu, Z., Kong, C., Liu, Y., and Sun, M. Fantastic semantics and where to find them: Investigating which layers of generative llms reflect lexical semantics. arXiv preprint arXiv:2403.01509, 2024c.

Mahajan, D., Goyal, S., Idrissi, B. Y., Pezeshki, M., Mitliagkas, I., Lopez-Paz, D., and Ahuja, K. Beyond multi-token prediction: Pretraining llms with future summaries. arXiv preprint arXiv:2510.14751, 2025.

Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 1(3):3, 2020.

Martens, J. New insights and perspectives on the natural gradient method. Journal of Machine Learning Research, 21(146):1–76, 2020.

Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. Mteb: Massive text embedding benchmark. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pp. 2014– 2037, 2023.

Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Pal, K., Sun, J., Yuan, A., Wallace, B. C., and Bau, D. Future lens: Anticipating subsequent tokens from a single hidden state. arXiv preprint arXiv:2311.04897, 2023.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, N.-Q., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fern´andez, R. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 1525– 1534, 2016.

Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D., et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024a.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21 (140):1–67, 2020.

Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., and Bengio, Y. Fitnets: Hints for thin deep nets. arxiv

2014. arXiv preprint arXiv:1412.6550, 2014.

Roy, O. and Vetterli, M. The effective rank: A measure of effective dimensionality. In 2007 15th European signal processing conference, pp. 606–610. IEEE, 2007.

Samragh, M., Kundu, A., Harrison, D., Nishu, K., Naik, D., Cho, M., and Farajtabar, M. Your llm knows the future: Uncovering its multi-token prediction potential. arXiv preprint arXiv:2507.11851, 2025.

Shazeer, N. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

Skean, O., Arefin, M. R., Zhao, D., Patel, N., Naghiyev, J., LeCun, Y., and Shwartz-Ziv, R. Layer by layer: Uncovering hidden representations in language models. arXiv preprint arXiv:2502.02013, 2025.

Sun, K., Yu, D., Yu, D., and Cardie, C. Investigating prior knowledge for challenging chinese machine reading comprehension. Transactions of the Association for Computational Linguistics, 8:141–155, 2020a.

Sun, S., Cheng, Y., Gan, Z., and Liu, J. Patient knowledge distillation for bert model compression. arXiv preprint arXiv:1908.09355, 2019.

Sun, S., Gan, Z., Fang, Y., Cheng, Y., Wang, S., and Liu, J. Contrastive distillation on intermediate representations for language model compression. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 498–508, 2020b.

Suzgun, M., Scales, N., Sch¨arli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q., Chi, E., Zhou, D., et al. Challenging big-bench tasks and whether chain-ofthought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 13003–13051, 2023.

Tack, J., Lanchantin, J., Yu, J., Cohen, A., Kulikov, I., Lan, J., Hao, S., Tian, Y., Weston, J., and Li, X. Llm pretraining with continuous concepts. arXiv preprint arXiv:2502.08524, 2025.

Talmor, A., Herzig, J., Lourie, N., and Berant, J. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language

Technologies, Volume 1 (Long and Short Papers), pp. 4149–4158, 2019.

Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Wang, L., Huang, J., Huang, K., Hu, Z., Wang, G., and Gu, Q. Improving neural language generation with spectrum control. In International Conference on Learning Representations, 2020.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

Wen, K., Li, Z., Wang, J., Hall, D., Liang, P., and Ma, T. Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective. arXiv preprint arXiv:2410.05192, 2024.

Xia, M., Gao, T., Zeng, Z., and Chen, D. Sheared llama: Accelerating language model pre-training via structured pruning. arXiv preprint arXiv:2310.06694, 2023.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., and Huang, G. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Zhang, L., Song, J., Gao, A., Chen, J., Bao, C., and Ma, K. Be your own teacher: Improve the performance of convolutional neural networks via self distillation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3713–3722, 2019.

Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

Zhao, Q., He, R., Zhang, J., Liu, C., and Wang, B. Representation degeneration problem in prompt-based models for natural language understanding. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pp. 13946–13957, 2024.

Zhong, W., Cui, R., Guo, Y., Liang, Y., Lu, S., Wang, Y., Saied, A., Chen, W., and Duan, N. Agieval: A humancentric benchmark for evaluating foundation models. In Findings of the Association for Computational Linguistics: NAACL 2024, pp. 2299–2314, 2024.

Zuhri, Z. M., Fuadi, E. H., and Aji, A. F. Predicting the order of upcoming tokens improves language modeling. arXiv preprint arXiv:2508.19228, 2025.

### A. Detailed Proofs for NITP Geometry

In this section, we provide the detailed derivation of the gradient and Hessian matrix for the Next Implicit Token Prediction (NITP) objective. We further analyze the spectral properties of the Hessian to validate the geometric regularization claims made in Section 3.5, providing the formal proof for Lemma 3.1 and Theorem 3.2.

#### A.1. Notation and Preliminaries

Consistent with the setup in Section 3.5, let h ∈ Rd denote the representation used for the implicit prediction loss. We treat h as the projected state to isolate the geometric incentives of the objective function. Let z ∈ Rd denote the fixed implicit target derived from shallow layers. We define the following variables:

h r

r := ∥h∥2, u :=

, v :=

z ∥z∥2

, s := u⊤v = cos(h,z).

We assume r > 0. The tangential difference vector is defined as A := v − su. Note that A is orthogonal to u:

u⊤A = u⊤(v − su) = u⊤v − s(u⊤u) = s − s = 0. (9)

The NITP objective function is:

h⊤z ∥h∥2∥z∥2

L(h) = 1 − cos(h,z) = 1 −

##### A.2. Gradient Derivation First, recall the Jacobian of the normalization map u(h) = h/r:

= 1 − u⊤v. (10)

∂u ∂h

=

Ju(h) =

Let f(h) = u⊤v be the cosine similarity. By the chain rule:

1 r

(I − uu⊤). (11)

∇hf(h) = Ju(h)⊤v =

1 r

(I − uu⊤)v

1 r

(v − u(u⊤v))

=

1 r

A r

. (12) Thus, the gradient of the loss L(h) = 1 − f(h) is:

(v − su) =

=

A r

. (13)

∇hL(h) = −

##### A.3. Hessian Derivation (Proof of Lemma 3.1) The Hessian is the Jacobian of the gradient vector. We differentiate ∇hL(h) = −r−1A with respect to h:

1 r∇hA + A(∇hr−1)⊤ . (14)

A r

= −

HNITP(h) = ∇h −

We compute the two terms separately.

#### Term 1: Derivative of r−1. Using ∇hr = u:

1 r2∇hr = −

u r2

∇h(r−1) = −

. (15)

Thus, the second part of the product rule is:

u⊤ r2

A(∇hr−1)⊤ = A −

1 r2

Au⊤. (16)

= −

- Term 2: Derivative of A. Recall A = v − su. Since v is constant: ∇hA = −∇h(su) = − u(∇hs)⊤ + s∇hu . (17)

Substituting ∇hs = ∇hf = A/r and ∇hu = 1r(I − uu⊤):

∇hA = − u

A r

⊤

+ s

1 r

(I − uu⊤)

1 r

= −

uA⊤ + s(I − uu⊤) . (18)

Combining the Terms. Substituting back into the expression for HNITP:

1 r

1 r2

1 r −

(uA⊤ + s(I − uu⊤)) + −

Au⊤

HNITP(h) = −

1 r2

1 r2

(uA⊤ + s(I − uu⊤)) +

Au⊤

=

1 r2

s(I − uu⊤) + (uA⊤ + Au⊤) . (19)

=

This provides the exact Hessian form.

Approximation near convergence. As the training progresses and the prediction aligns with the target, we have s → 1. Consequently, the difference vector A = v − su approaches zero (∥A∥ → 0). Under this condition, the term (uA⊤ + Au⊤) vanishes. The Hessian simplifies to:

1 r2

1 r2

(I − uu⊤) =

P⊥u. (20) This confirms the simplified form presented in Lemma 3.1. □

HNITP(h) ≈

- A.4. Analysis of Spectral Properties We now verify the properties stated in the main text regarding the null space and spectral lifting.

- Property 1: Radial Null Space. We examine the curvature along the radial direction u. We compute the quadratic form u⊤HNITPu:

u⊤HNITPu =

1 r2

  su⊤(I − uu⊤)u

0

+u⊤(uA⊤)u

u⊤u(A⊤u)

+u⊤(Au⊤)u

u⊤A(u⊤u)

  . (21)

Since I − uu⊤ projects onto the orthogonal complement of u, the first term is 0. For the remaining terms, recall that A ⊥ u, so A⊤u = 0 and u⊤A = 0.

u⊤HNITPu =

1 r2

[0 + 1 · 0 + 0 · 1] = 0. (22)

This confirms that the loss function imposes zero curvature (is locally linear or flat) with respect to changes in the vector norm.

- Property 2: Angular Spectral Lifting. Consider a perturbation vector w ∈ Rd such that w ⊥ u (representing a purely semantic change). We compute w⊤HNITPw:

1 r2

w⊤HNITPw =

sw⊤(I − uu⊤)w + w⊤(uA⊤ + Au⊤)w . (23)

Since w ⊥ u: 1. w⊤(I − uu⊤)w = w⊤w − (w⊤u)2 = ∥w∥22. 2. w⊤uA⊤w = (w⊤u)(A⊤w) = 0. 3. w⊤Au⊤w = (w⊤A)(u⊤w) = 0.

Thus, the curvature in the tangential direction is:

s r2∥w∥22. (24)

w⊤HNITPw =

Assuming reasonable alignment during training (s = cos(h,z) > 0), the Hessian is strictly positive definite on the subspace orthogonal to h.

#### A.5. Proof of Theorem 3.2 (Null Space Mitigation)

We consider the total loss Hessian Htotal = HNTP + λHNITP.

Existence of null space. As analyzed in Section 3.2, although the NTP objective theoretically involves all vocabulary items via the Softmax normalization, the gradient signal is empirically dominated by the target token and a sparse set of high-probability contenders. Consequently, the curvature of HNTP is concentrated within this low-dimensional subspace, leaving the vast orthogonal complement effectively unconstrained (rank-deficient).

Based on this, let Nsem be the semantic null space of the standard NTP objective, defined as the set of tangential directions w ⊥ u that lie in this unconstrained subspace (w⊤HNTPw ≈ 0).

For any such degenerate direction w ∈ Nsem, we compute the quadratic form of the total Hessian:

w⊤Htotalw = w⊤(HNTP + λHNITP)w

+λw⊤HNITPw. (25)

= w⊤HNTPw

≈0

Substituting the result from Property 2 derived above (w⊤HNITPw = rs2 ∥w∥22):

s r2∥w∥22 . (26)

w⊤Htotalw ≈ 0 + λ

Under the empirical observation that NITP converges to high alignment (s → 1), we have s > 0. Given λ > 0, it follows that:

w⊤Htotalw > 0 for all w ∈ Nsem,w ̸= 0. (27) This proves that the addition of NITP mitigates the semantic null space by ensuring strictly positive curvature in these previously unconstrained directions. □

### B. Experiment Details

Hyperparameter settings. Table 4 summarizes the hyperparameter configurations used for pretraining both MoE and Dense models. Across all model scales, we adopt a largely consistent training recipe to ensure fair comparisons between standard NTP and NITP. All models are optimized using AdamW with momentum coefficients (β1,β2) = (0.9,0.95), weight decay set to 0.1, and global gradient clipping at 1.0. We employ a warmup–stable–decay (WSD) learning rate schedule (Wen et al., 2024) with 2,000 warmup steps and a fixed decay ratio of 0.2. The learning rate and global batch size are scaled according to model size, while the context length is fixed to 8192 tokens for all experiments.

Model architectures. For MoE models, the backbone consists of a shallow dense stage followed by multiple MoE layers, with the number of MoE layers increasing with model capacity. Each MoE layer contains 144 routed experts and one shared expert, using top-8 routing throughout. To control the overall parameter budget, the expert feed-forward hidden size is adjusted across scales, while the dense FFN size is kept fixed. All MoE models use multi-head attention with 8 query heads and 4 key–value heads, with head dimensions scaled proportionally to the hidden size.

Dense models follow a standard Transformer architecture with 24–28 layers depending on model size. They use a larger number of attention heads and higher-dimensional FFNs to match the overall parameter scale of the MoE counterparts. For both MoE and Dense models, the NITP target layer is chosen from intermediate depths, and the NITP loss weight is set close to 1.0, with minor adjustments for larger models. All models are pretrained on large-scale tokenized corpora using the Qwen2 tokenizer. The total number of training tokens ranges from 105B to 330B, depending on model size and architecture.

Table 4. Hyper-parameters for MoE and Dense model pretraining.

###### Parameters MoE Models Dense Models 1.9bA0.3b 3bA0.5b 9bA1b 0.5B 2B 3B

lr-schedule WSD WSD WSD WSD WSD WSD lr 6.0e-4 8.7e-4 4.2e-4 4.2e-4 4.2e-4 4.2e-4 warmup steps 2,000 2,000 2,000 2,000 2,000 2,000 decay-ratio 0.2 0.2 0.2 0.2 0.2 0.2 AdamW-β (0.9, 0.95) (0.9, 0.95) (0.9, 0.95) (0.9, 0.95) (0.9, 0.95) (0.9, 0.95) weight-decay 0.1 0.1 0.1 0.1 0.1 0.1 grad clip 1.0 1.0 1.0 1.0 1.0 1.0 global batch size 256 512 1024 512 1024 1024

Training

hidden dim. 512 768 1280 896 1792 2560 #dense layers 1 1 1 24 28 28 #moe layer 15 16 23 - - head dim. 64 96 160 64 128 128 #q heads 8 8 8 14 14 20 #kv heads 4 4 4 2 2 4 context-length 8192 8192 8192 8192 8192 8192 #routed experts 144 144 144 - - #shared experts 1 1 1 - - topK route 8 8 8 - - dense FFN size 10944 10944 10944 4864 10752 10240 MoE FFN size 496 544 640 - - shared exp. size 496 544 640 - - NITP target layer 3/16 4/17 5/24 5/24 6/28 6/28 NITP loss weight 1.0 1.0 0.8 1.0 1.0 0.8

Model

tokenizer qwen2 qwen2 qwen2 qwen2 qwen2 qwen2 #tokens(B) 105B 200B 330B 200B 330B 330B

Data

Table 5. Scaling result on a 45B MoE model. We report results on a larger MoE model trained for 240B tokens. Best results are bolded.

Model Method MMLU M-Pro C-Eval Xiezhi LAMB. BBH ARC-C CSQA COPA C3 AGIEval GSM8k LCBench Avg. 45B MoE

###### NTP 55.09 31.71 47.67 57.65 69.03 42.89 67.70 59.46 75.20 75.51 35.08 50.27 13.39 52.36 NITP 55.51 32.29 50.87 59.97 69.38 43.48 70.10 59.95 75.20 78.36 38.69 52.62 15.83 54.02

For the average downstream performance shown in the bottom row of Figure 1, we average over four representative benchmarks: MMLU, C3, C-Eval, and ARC-Challenge.

### C. Scaling to 45B MoE

To examine whether NITP remains effective beyond the 1.9B–9B MoE scales used in our main experiments, we further evaluate a 45B-A5.5B MoE model. We use the hyperparameter recipe suggested by our ablations: the implicit target is taken from a shallow contextualized layer around 20% of the total depth, and the NITP loss weight is selected from a moderate range. Across the 3B and 9B models, λ ∈ [0.8,1.0] performs consistently well. For the larger 45B model, we find that a slightly smaller weight, λ = 0.6, works best, continuing the trend from 3B to 9B MoE where the selected weight decreases from 1.0 to 0.8. This suggests that the auxiliary representation-level signal can be mildly down-weighted as the parameter scale increases. Specifically, the 45B-A5.5B model follows a DeepSeek-V2-style MoE architecture and is trained for 240B tokens. Following the fields in Table 4, it has 24 layers, hidden size 3072, 144 routed experts, one shared expert, top-12 routing, GQA with 24 query heads and 8 KV groups, head dimension 128, dense FFN size 12288, and MoE/shared expert FFN size 1408. It uses a context length of 8192, the Qwen2 tokenizer, global batch size 768, AdamW with (β1,β2) = (0.9,0.95), weight decay 0.1, gradient clipping at 1.0, and a WSD-exp learning-rate schedule with peak learning rate 2.8 × 10−4. For NITP, the target layer is 5/24.

- Table 5 shows that NITP continues to improve performance at this larger scale, increasing the average score from 52.36 to

54.02. This result supports a simple empirical scaling strategy for NITP: choose the target layer near 20% depth and use a moderate auxiliary weight, with slightly smaller weights for larger models.

### D. Analysis of Representation Geometry Dynamics

We provide a detailed analysis of the geometric phenomena observed in Figure 1, focusing on the joint behavior of effective rank and average cosine similarity during training.

Effective rank dynamics. We periodically measure the effective rank of the last-layer hidden states during training by collecting all valid token representations within a mini-batch, applying mean-centering, and computing the entropy-based effective rank from the eigenvalue spectrum of the empirical covariance matrix. Measurements are reported every fixed number of training steps to track long-term geometric trends.

- As shown in Figure 1(a), where the background indicates the variance of effective rank, standard NTP exhibits a rapid and monotonic collapse of effective rank, converging to a low-variance regime early in training. This behavior indicates that the learned hidden states concentrate on a small, fixed subspace, regardless of contextual variation. Such convergence reflects a degenerate equilibrium: once the model discovers a narrow anisotropic cone that suffices for next-token prediction, no gradient signal encourages re-expansion along semantically meaningful but task-irrelevant directions.

In contrast, NITP maintains a substantially higher effective rank with noticeably larger variance over training. Importantly, this variance should not be interpreted as optimization instability. Rather, it is a direct consequence of mitigating the semantic null space inherent to NTP. By explicitly supervising the prediction of the next token in latent space, NITP introduces positive curvature along directions that are otherwise unconstrained under likelihood-based training. As a result, the model dynamically activates different subsets of the representation space depending on contextual demands, leading to controlled fluctuations in effective rank.

Cosine similarity and anisotropy. To quantify global anisotropy, we report the average cosine similarity between randomly sampled pairs of token representations from the last layer. At each reporting step, a fixed number of token pairs is sampled across sentences, and cosine similarity is computed after ℓ2 normalization.

- As shown in Figure 1(b), under NTP the average cosine similarity increases steadily throughout training, signaling growing anisotropy and alignment toward a common dominant direction. This behavior is consistent with prior observations of representation degeneration, where embeddings become increasingly indistinguishable despite maintaining predictive accuracy. NITP alleviates this effect. Although cosine similarity still increases as training progresses, the overall magnitude is consistently lower, indicating that representations preserve greater angular diversity. This reduced anisotropy aligns with the higher effective rank observed under NITP, jointly suggesting that the representation space remains more expressive and better conditioned.

Implications. Taken together, these results clarify that NITP does not merely slow down representation collapse, but fundamentally alters the geometric equilibrium of training. While NTP permits hidden states to sacrifice expressiveness as long as token prediction remains correct, NITP enforces semantic consistency in latent space, preventing representations from collapsing into a static anisotropic cone. This geometric regularization provides a mechanistic explanation for the improved downstream generalization observed with NITP.

### E. Hidden-State Representation Quality

NITP is designed to improve the representations learned during pre-training, rather than merely to change final benchmark scores through the output head. While Appendix D characterizes this effect through representation geometry, we further examine whether the learned hidden states encode richer semantic information than the baseline.

Hidden-state quality on MTEB. We evaluate frozen sentence representations on MTEB v2.12.0 (Muennighoff et al.,

- 2023). Following standard practice, for each example, we extract the last-layer hidden states from the 3B MoE models,

mean-pool over valid tokens, and apply ℓ2 normalization. No fine-tuning, contrastive training, or task-specific adaptation is applied, so the evaluation serves as a direct probe of the representation quality induced by the pre-training objective. We evaluate 25 English MTEB tasks and exclude seven large-scale clustering and reranking tasks due to computational cost.

[Figure 7]

Figure 5. Training dynamics of the NITP loss. Evolution of LNITP during pre-training, exhibiting a characteristic three-phase behavior: an initial collapse induced by random initialization, a transient hump caused by the emergence of structured shallow-layer targets, and a long-term stable convergence.

According to the task types used in our evaluation pipeline, the remaining tasks are grouped into Classification (10 tasks), STS/Similarity (9 tasks), and Retrieval/Duplicate-detection (6 tasks).

The evaluated tasks are: Classification: AmazonCounterfactualClassification, AmazonReviewsClassification, Banking77Classification, EmotionClassification, MTOPDomainClassification, MTOPIntentClassification, MassiveIntentClassification, MassiveScenarioClassification, ToxicConversationsClassification, and TweetSentimentExtractionClassification; STS/Similarity: BIOSSES, SICK-R, STS12, STS13, STS14, STS15, STS16, STS17, and STSBenchmark; and Retrieval/Duplicate-detection: AskUbuntuDupQuestions, SciDocsRR, SprintDuplicateQuestions, StackOverflowDupQuestions, TwitterSemEval2015, and TwitterURLCorpus.

- Table 6. MTEB evaluation of hidden-state quality. Scores are reported as percentages and macro-averaged over task-level main scores within each group. NITP improves representation utility on 23 out of 25 tasks using frozen last hidden states.

#### Task Group NTP NITP ∆

Classification (10) 40.09 42.29 +2.20 STS / Similarity (9) 35.66 38.58 +2.93 Retrieval / Dup. (6) 43.19 44.83 +1.64

#### Overall (25 tasks) 39.24 41.56 +2.33

As shown in Table 6, NITP improves the overall MTEB score from 39.24 to 41.56. The gains are consistent across task families, with improvements of +2.20 points on classification, +2.93 points on semantic textual similarity, and +1.64 points on retrieval/duplicate-detection tasks. At the individual task level, NITP improves 23 out of 25 tasks; the only regressions are small drops on AskUbuntuDupQuestions (−0.15 points) and ToxicConversationsClassification (−0.88 points). These results indicate that NITP improves the transferability of the hidden states themselves, providing direct evidence for its role as a representation-level pre-training signal.

Language modeling preservation. Finally, we verify that the stronger representations do not come from sacrificing the original language modeling objective. On the Pile (Gao et al., 2020) validation set, NITP achieves nearly identical validation cross-entropy loss to NTP across model configurations: 3B MoE obtains 2.006 vs. 2.006 (PPL 7.43 vs. 7.43), 9B MoE obtains 1.841 vs. 1.840 (PPL 6.30 vs. 6.30), and 3B Dense obtains 1.884 vs. 1.882 (PPL 6.58 vs. 6.57), where each pair reports NTP vs. NITP. This suggests that token-level cross-entropy alone does not fully characterize the quality of the learned hidden states: models with matched next-token likelihood can still differ substantially in representation geometry and transferability, consistent with our core claim. NITP therefore preserves validation loss under the original next-token prediction objective while learning hidden-state representations with better semantic transferability.

### F. Loss Curve of NITP

To further understand the optimization dynamics of Next Implicit Token Prediction, we monitor the trajectory of LNITP throughout pre-training (Figure 5). The loss curve exhibits a characteristic three-phase evolution:

- • Phase I: random-initialization collapse. In the very early stage of training, the NITP loss drops sharply from its initial value within the first few dozen steps. This behavior is primarily induced by random initialization: both the

predictive state ht and the implicit target zt+1 are nearly isotropic and weakly differentiated across tokens, causing their cosine similarity to rapidly concentrate around a common value. As a result, the NITP objective collapses quickly before any meaningful representation learning takes place.

- • Phase II: emergence of structured targets. Following the initial drop, the loss exhibits a transient increase, forming a local peak (around 40–2k steps, i.e. warm-up phase). This behavior is consistent with the bottom-up convergence pattern observed in language models (Skean et al., 2025). As shallow layers converge faster, they begin to extract more

structured and contextualized representations, transforming the implicit target zt+1 from a near-constant signal into a more informative prediction target. This transition temporarily increases the difficulty of the NITP objective for the deeper predictive state ht.

- • Phase III: steady convergence. After the peak, the loss enters a prolonged and stable decline, indicating that deeper layers progressively adapt to the representational manifold defined by shallow layers. The final convergence to a low loss value (e.g., below 0.05 for our 9B model) is consistent with NITP providing a stable and learnable auxiliary supervision signal throughout pre-training.

Overall, the observed hump in the loss curve suggests that the NITP objective functions not merely as a static regularizer, but as an adaptive predictive task whose difficulty evolves alongside the model’s internal representations.

### G. Training FLOPs Analysis

This section provides a detailed analysis of the training FLOPs introduced by NITP, in comparison with the baseline NextToken Prediction (NTP) objective. Following common practice (Kaplan et al., 2020; Hoffmann et al., 2022), we approximate total training FLOPs as approximately 6× the number of parameters involved in forward computation, accounting for forward, backward, and parameter gradient updates.

#### G.1. Baseline NTP FLOPs

We consider a transformer-based MoE model with model hidden size d and vocabulary size V . Each layer consists of a self-attention module and a MoE block. Additionally, the unembedding layer projects the final hidden state to the vocabulary space. For each token:

- • Self-attention (GQA) (Ainslie et al., 2023): Adopting Grouped Query Attention (GQA) with 8 query heads and 4 KV heads, the parameter count is reduced compared to standard Multi-Head Attention. The projections

(WQ,WK,WV ,WO) contribute ≈ 3d2 parameters (where WK and WV are halved), resulting in ≈ 18d2 training FLOPs.

- • MoE MLP (SwiGLU (Shazeer, 2020)): Each expert uses three projections (d × de). For k activated experts, the training FLOPs are ≈ 6 × (3kdde) = 18kdde.
- • Unembedding Layer: The final projection to the large vocabulary space (d × V ) contributes significant computation. The training FLOPs are ≈ 6V d. Note that the input embedding layer is excluded as it involves only lookup operations (0 FLOPs).

The total baseline training FLOPs per token (summing over L layers and the output head) are:

FLOPsNTP ≈ L × (18d2 + 18kdde) + 6V d. (28)

#### G.2. Additional FLOPs Introduced by NITP

NITP introduces a projection head and a cosine loss. Crucially, extracting the target zt+1 incurs no cost due to the stop-gradient.

Projection head. We employ a SwiGLU projection head with an intermediate dimension of 4d. It consists of three linear transformations (d → 4d, d → 4d, and 4d → d), resulting in 12d2 parameters. The training FLOPs are:

FLOPsproj ≈ 6 × 12d2 = 72d2. (29)

Cosine similarity loss. The computation involves dot products and norms (≈ 6d operations). Including the backward pass:

FLOPscos ≈ 3 × 6d = 18d. (30)

#### G.3. Numerical Instantiation: 9B MoE Model

Using the 9B MoE model parameters specified in Table 4 (d = 1280, de = 640, k = 9, L = 24, V = 152064), we estimate the FLOPs as follows:

- • NITP Overhead (Total): 72d2 + 18d ≈ 72 × 12802 ≈ 1.18 × 108 FLOPs.
- • Baseline (Backbone Layers): 24 × (18 × 12802 + 18 × 9 × 1280 × 640) ≈ 24 × 1.62 × 108 ≈ 3.89 × 109 FLOPs.
- • Baseline (Unembedding): 6 × 152064 × 1280 ≈ 1.17 × 109 FLOPs.
- • Baseline (Total): 3.89 × 109 + 1.17 × 109 ≈ 5.06 × 109 FLOPs.

The resulting relative training overhead is:

#### G.4. Empirical Wall-Clock Overhead

1.18 × 108 5.06 × 109 ≈ 2.3%. (31)

FLOPsNITP FLOPsNTP ≈

We also measure the practical training-time overhead in the same 9B MoE setting. Using the same number of GPUs and a global batch size of 1024, we compare NTP and NITP over a 5k-step run. The NTP baseline takes 16h 1m, while NITP takes 16h 18m, corresponding to an additional 17 minutes, or approximately 1.8% wall-clock overhead.

This empirical overhead is consistent with the FLOPs estimate above. The measured wall-clock increase is slightly smaller than the theoretical FLOPs ratio because large-scale MoE training also involves communication and system overheads that are largely unaffected by the NITP projection head and cosine loss.

Conclusion. The analysis confirms that the overhead introduced by NITP remains marginal (∼ 2%), ensuring scalability.

### H. Inference FLOPs Analysis

Crucially, NITP introduces zero additional computational overhead during inference. Since the NITP objective serves as an auxiliary supervision signal, the projection head is discarded after pre-training. The model architecture used for deployment remains identical to the standard transformer backbone. Therefore, the inference FLOPs per token are exactly the same as the baseline NTP model, ensuring that NITP improves model performance without compromising generation speed or increasing serving costs.

### I. Additional Ablation Studies

Target-layer sweep. We further study the choice of the layer used to construct the implicit target. As shown in Table 7, the best target consistently appears in a shallow but contextualized region: layer 4 out of 17 for the 3B MoE model and layer 5 out of 24 for the 9B MoE model, both close to 20% of the total depth. Using the embedding layer as the target is weaker, suggesting that purely lexical embeddings lack sufficient contextual information. Moving the target deeper also reduces the gain, likely because deeper representations become increasingly specialized for token discrimination and provide less complementary semantic supervision.

Stop-gradient on implicit targets. We ablate whether gradients are allowed to propagate into the implicit target representations in Equation (2). Allowing gradients to flow into the targets leads to co-adaptation between the predicted and

- Table 7. Target-layer sweep across scales. We report the average score over MMLU, MMLU-Pro, CSQA, BBH, and LCBench. The best target layer consistently lies in a shallow contextualized region around 20% of the total depth.

Model / Setting Target layer Avg. 3B MoE (17 layers)

NTP baseline – 21.10 Embedding target 0 20.72 Candidate target 2 22.01 Selected target 4 23.58 Candidate target 6 22.12 Candidate target 8 21.22

9B MoE (24 layers)

NTP baseline – 27.95 Candidate target 4 30.05 Selected target 5 30.75 Candidate target 6 29.80

- Table 8. Additional ablations for NITP design choices. We report results on key reasoning benchmarks. Avg. denotes the arithmetic mean of MMLU, MMLU-Pro, CSQA, BBH, and LCBench.

Ablation Factor Setting MMLU MMLU-Pro CSQA BBH LCBench Avg. Baseline: NTP 34.60 11.00 34.15 21.92 3.82 21.10

Disabled 31.89 8.43 27.51 21.25 2.09 18.23 Enabled 37.37 12.29 37.92 26.14 4.17 23.58

Stop Gradient

3000 35.54 10.14 31.85 21.25 2.09 20.17 2000 36.17 10.28 34.72 23.03 2.78 21.40 1000 35.93 11.57 31.22 23.40 4.00 21.22 0 37.37 12.29 37.92 26.14 4.17 23.58

NITP Start Step

Projector × 35.40 11.71 29.48 24.00 2.61 20.64 ✓ 37.37 12.29 37.92 26.14 4.17 23.58

target representations, which frequently results in unstable optimization and degraded performance. In contrast, applying a stop-gradient operation consistently stabilizes training and yields substantial improvements across downstream benchmarks. The results in Table 8 highlight that treating the implicit target as a fixed semantic anchor is critical for effective latent-space prediction, whereas disabling stop-gradient performs significantly worse, undermining the supervisory signal by allowing the target to drift during training. Accordingly, we adopt stop-gradient on implicit targets in all experiments.

Start step of NITP. Motivated by the three-phase dynamics observed in the NITP loss curve, we study when NITP supervision should be activated during training. We compare enabling NITP from the beginning with introducing it at later stages. As shown in Table 8, delaying NITP activation leads to consistent performance degradation, particularly on reasoning-oriented benchmarks, while applying NITP from step 0 yields the best overall results. These results indicate that NITP is most effective when applied early, where it can shape representation geometry before higher-level semantics stabilize, thereby promoting more coherent alignment between shallow and deep representations throughout training.

Effect of the projection head. We study the role of the projection head in NITP by comparing our default design, which applies a two-layer MLP projector using SwiGLU on the last hidden states, with a projector-free variant that directly aligns the last hidden states with the implicit targets. Removing the projector consistently results in notable performance degradation across all benchmarks as shown in Table 8. We attribute this performance drop to a distributional mismatch between deep representations and shallow implicit targets. Directly enforcing alignment imposes an overly restrictive constraint, while a lightweight MLP projection head provides sufficient flexibility to bridge this mismatch without limiting the backbone capacity.

