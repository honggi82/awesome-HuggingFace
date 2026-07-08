# arXiv:2601.21192v1[cs.AI]29Jan2026

## Do Reasoning Models Enhance Embedding Models?

Wun Yu Chan1, Shaojin Chen1, Huihao Jing1, Kwun Hang Lau1, Elton Chun-Chai Li1, Zihao Wang1, Haoran Li1, Yangqiu Song1

1CSE, HKUST

Correspondance: wychanbu@connect.ust.hk Reasoning-Embedding Reasoning-Embedding

[Figure 1]

January 30, 2026

Abstract

State-of-the-art embedding models are increasingly derived from decoder-only Large Language Model (LLM) backbones adapted via contrastive learning. Given the emergence of reasoning models trained via Reinforcement Learning with Verifiable Rewards (RLVR), a natural question arises: do enhanced reasoning translate to superior semantic representations when these models serve as embedding initializations? Contrary to expectation, our evaluation on MTEB and BRIGHT reveals a null effect: embedding models initialized from RLVR-tuned backbones yield no consistent performance advantage over their base counterparts when subjected to identical training recipes. To unpack this paradox, we introduce Hierarchical Representation Similarity Analysis (HRSA), a framework that decomposes similarity across representation, geometry, and function levels. HRSA reveals that while RLVR induces irreversible latent manifold’s local geometry reorganization and reversible coordinate basis drift, it preserves the global manifold geometry and linear readout. Consequently, subsequent contrastive learning drives strong alignment between base- and reasoning-initialized models, a phenomenon we term Manifold Realignment. Empirically, our findings suggest that unlike Supervised Fine-Tuning (SFT), RLVR optimizes trajectories within an existing semantic landscape rather than fundamentally restructuring the landscape itself.

### 1 Introduction

Vector representations of text, known as text embeddings, are a core abstraction in modern natural language processing (NLP) (Mikolov et al., 2013). As Large Language Models (LLMs) continue to evolve, embedding models have now been built by adapting decoder-only LLMs (Lee et al., 2025a; Zhang et al., 2025; Lee et al., 2025b) as backbones to leverage the rich semantics and world knowledge stored in their parameters.

Most recently, reasoning models optimized via Reinforcement Learning with Verifiable Rewards (RLVR) on base models have demonstrated a qualitative leap in complex problem-solving and reasoning (DeepSeek-AI, 2025; Lambert et al., 2024; Xu et al., 2025; Zheng et al., 2025). This development raises a natural hypothesis for representation learning: Does the enhanced reasoning translate to a superior text embedding space? Intuitively, a model that "thinks" more deeply should structure semantic relationships more effectively.

Counter-intuitively, our results reveal a null effect. Across comprehensive benchmarks including MTEB(Multilingual,

v2) (Enevoldsen et al., 2025), MTEB(Code, v1) (Muennighoff et al., 2023), and BRIGHT (Su et al., 2025), embedding models initialized from RLVR-tuned reasoning models perform statistically identically to base-initialized models after contrastive learning (van den Oord et al., 2018; Gao et al., 2021). This observation presents a scientific puzzle: Why do reasoning and non-reasoning backbones yield indistinguishable results following contrastive learning?

In this paper, we argue that existing performance metrics are insufficient for diagnosing the internal dynamics of representations. We introduce Hierarchical Representation Similarity Analysis (HRSA), a hierarchical analysis framework inspired by Representational Similarity Analysis (RSA) (Kornblith et al., 2019). HRSA allows us to dissect model similarity at increasing levels of abstraction:

- • Representation Level: Focus on the coordinate basis and features.
- • Geometry Level: Focus on the shape (geometry) of the latent manifold.

[Figure 2]

Figure 1: Latent manifold and model relationships. CL, SFT, and RLVR denote Contrastive Learning, Supervised Fine-Tuning, and Reinforcement Learning with Verifiable Rewards, respectively. z indicates the representations of the corresponding models. Suffix “-Emb” is added to the model name to indicate the embedding model. We demonstrate the ideas of similar and dissimilar representations of RLVR-tuned pairs and SFT-tuned pairs, respectively.

• Function Level: Focus on the input-output mappings.

Applying HRSA uncovers a phenomenon we term Manifold Realignment. We find that RLVR largely preserves the global geometry of the latent manifold, including the linear readout associated with downstream tasks, while irreversibly reshaping the manifold’s local geometry. The resulting drift in the coordinate basis is modest under typical training regimes but becomes pronounced under prolonged RLVR. Strikingly, when these backbones are later adapted into embedding models via contrastive learning, both base- and reasoning-initialized models exhibit strong realignment even in the presence of coordinate basis changes. We interpret the realignment as evidence that representational drift is largely reversible at the global level, yet accompanied by irreversible local distortions. Overall, our results suggest that, unlike SFT, RLVR primarily optimizes trajectories through an existing semantic landscape rather than fundamentally redrawing the landscape itself.

Our contributions are as follows:

- 1. Systematic Benchmarking: We conduct the first controlled comparison of RLVR-optimized vs. its base model as backbones for text embeddings by fine-tuning a diverse suite of state-of-the-art reasoning models into embedding models and evaluate them against their base counterparts, establishing that current RLVR methods do not inherently improve embedding quality.
- 2. The HRSA Framework: We propose a hierarchical RSA framework (Representation, Geometry, Function) to diagnose why models behave similarly, offering a toolkit for future interpretability studies, and unifying the disorganized RSA framework.
- 3. Discover Manifold Realignment: We demonstrate that RLVR do not fundamentally alter the latent manifold, but it can reorganize the local neighborhood structure, and only the coordinate basis will be drifted when training is prolonged. However, the contrastive learning can overwrite the reversible drift and exhibit strong alignment between base- and reasoning-initialized embedding models.

### 2 Embedding Model Performances

We first unify our terminology by explicitly separating a starting checkpoint from the fine-tuning stage.

Backbone LLMs. Given a base model Mbase as a trained LLM, we consider a reasoning model Mreason as an LLM that undergoes either Supervised Fine-Tuning (SFT) or RLVR directly on top of the base model. We focus on

- Table 1: Mean embedding benchmark performance (3 seeds). We compare the base backbone Mbase versus its RLVR-tuned reasoning model backbone Mreason; we also include an SFT-tuned backbone for reference. The ∆ (Std) column (gray) shows the mean performance gap ± standard deviation. The near-zero deltas for RLVR indicate that RLVR largely preserves the base model’s semantic effectiveness, contrasting with larger shifts under SFT.

MTEB(Multilingual, v2) MTEB(Code, v1) BRIGHT Model Pair Backbone MEmbbase MEmbreason ∆ (Std) MEmbbase MEmbreason ∆ (Std) MEmbbase MEmbreason ∆ (Std) SFT Qwen3-0.6B-Base vs Qwen3-0.6B 53.50 41.47 -12.03 ±0.14 55.29 56.28 +0.99 ±0.05 13.06 13.71 +0.65 ±0.06 RLVR

Qwen2.5-1.5B vs Qwen2.5-1.5B-SRL-Zoo 54.73 54.54 -0.19 ±0.07 58.98 58.72 -0.26 ±0.08 17.71 17.89 +0.18 ±0.04 Qwen2.5-0.5B vs Qwen2.5-0.5B-SRL-Zoo 51.25 51.27 +0.02 ±0.09 57.41 57.53 +0.12 ±0.07 14.05 13.99 -0.06 ±0.05 DS-Distill-1.5B vs NV-ProRL 46.19 46.25 +0.06 ±0.06 45.47 45.87 +0.40 ±0.07 9.02 9.47 +0.45 ±0.02 Qwen3-4B vs Qwen3-4B-PSR 59.85 59.79 -0.06 ±0.05 63.90 64.57 +0.67 ±0.03 18.10 18.17 +0.07 ±0.03

zero-RL (DeepSeek-AI, 2025) where RLVR starts directly from the base model without performing a warm-start SFT stage. Concretely, we evaluate and compare a matched pair of Mbase and Mreason, where Mreason must be fine-tuned on Mbase. The SFT-tuned pairs are used as an explicit control to highlight the very close similarity observed in the RLVR-tuned comparisons.

Embedding models. We term the base embedding model MEmbbase and reasoning embedding model MEmbreason with the backbone Mbase and Mreason respectively. The embedding models are formed by removing the language modeling head and applying a pooling operator to the final-layer hidden states to produce a fixed-dimensional vector. We train embedding models with an InfoNCE objective (van den Oord et al., 2018) to align semantically similar texts. Within a pair the two embedding models share identical architectures and training recipes, differing only in their backbone initialization (i.e., Mbase vs. Mreason). Appendix A provides training details.

To rigorously assess the impact of RLVR optimization on embedding benchmark, we trained and evaluated multiple

matched pairs consisting MEmbbase and MEmbreason. We evaluated these models across a diverse suite of benchmarks, including MTEB(Multilingual, v2) (Enevoldsen et al., 2025), MTEB(Code, v1) (Muennighoff et al., 2023), and

BRIGHT (Su et al., 2025), to ensure coverage of retrieval, clustering, and semantic similarity tasks, as well as the data in the same domain as trained in RLVR.

The results, presented in Table 1, reveal that MEmbreason with RLVR-tuned backbone Mreason consistently achieve

performance parity with MEmbbase across all benchmarks. Rather than interpreting this as a limitation, we view it as a significant indicator of representational robustness. The RLVR process refines the model’s trajectory-generation policy

without destructively overwriting the rich world knowledge and semantic relationships established during pre-training.

### 3 The HRSA Framework: Dissecting Model Similarity

In Section 2, we established that MEmbreason with RLVR-tuned backbone maintain performance parity with MEmbbase across all benchmarks, exhibiting no degradation in general semantic tasks. This macroscopic observation suggests a

hypothesis that the RLVR optimization trajectory preserves the intrinsic geometry of the pre-trained Latent Manifold Z, altering only the policy for traversing it rather than the landscape itself.

To rigorously test this hypothesis, we must look beyond aggregate benchmark scores, which can mask internal representational shifts. We introduce HRSA to dissect the relationship between Mbase and Mreason at three nested levels of abstraction.

Crucially, HRSA is not defined by the specific metrics used in this study (e.g., CKA), but by the invariance properties required at each level of abstraction. Researchers can substitute metrics or theoretical constraints, provided they respect the hierarchy’s invariance rules. See Table 8.

[Figure 3]

Figure 2: The overview of HRSA.

#### 3.1 Common Setup and Notation

To analyze the structural differences between models, we compare their representations on a shared sequence of N token positions. Let X,Y ∈ RN×D denote the D-dimensional token-level representation matrices produced by Mbase (or MEmbbase ) and Mreason (or MEmbreason) respectively. The i-th row of each matrix, denoted as xi or yi, represents a single token embedding such that xi,yi ∈ Z, where Z ⊂ RD is the latent manifold induced by the distribution of the mapped inputs within the high-dimensional ambient space.

#### 3.2 Representation-Level Analysis

Representation-level analysis focuses on the specific coordinate basis of the latent manifold and tests feature-wise correspondences between models. At this level, we treat the coordinate basis themselves as meaningful objects, where rotating or permuting features can change the outcome of the analysis. In other words, representation-level metrics are not invariant to orthogonal transformations or neuron permutations (see Appendix B.1 for a formal discussion).

Intuitively, this level investigates whether two models implement similar features along similar coordinate basis or realize similar solutions in very different coordinate systems. We operationalize this question with two complementary tools:

- 1. Dimension-Wise Correlation: probe direct, axis-aligned feature correspondence (Beyer et al., 2020).
- 2. Orthogonal Procrustes Analysis: probe global linear equivalence (Schönemann, 1966).

##### 3.2.1 Dimension-Wise Correlation

Dimension-wise correlation tests whether each coordinate in one model can be matched to the same coordinate in another model. Let X:j and Y:j denote the j-th column vectors (features) of the matrices X and Y, respectively, corresponding to feature j across all token positions. After centering each column over tokens, we define the correlation for dimension j as

(X:j)⊤Y:j ∥X:j∥∥Y:j∥

, j = 1,...,D. (1)

ρj(X,Y) =

We summarize {ρj}Dj=1 via mean. High per-dimension correlations indicate that many features are already aligned one-to-one without any transformation, while low per-dimension correlations suggest that, even if the models encode similar information overall, information carried by a single feature in one model may be distributed across multiple features in the other.

##### 3.2.2 Orthogonal Procrustes Analysis

Dimension-wise correlation is strict: it does not allow any mixing between feature dimensions. Orthogonal Procrustes analysis relaxes this by asking whether one representation space can be mapped to the other via a single orthogonal transformation. Formally, we solve

O∗ = arg min

∥XO − Y∥2F, (2)

O⊤O=I

where O∗ ∈ RD×D is orthogonal and ∥ · ∥F denotes the Frobenius norm. This objective allows a global orthogonal mixing of features: each feature of Y can be an orthogonal combination of features in X. If O⋆ is a near-diagonal

- Table 2: HRSA result summary. It shows how different training algorithms impact the model’s manifold. SFT causes fundamental restructuring, whereas RLVR acts as a trajectory optimization. Contrastive Learning (CL) successfully realigns the latent manifold.

Level Metric Focus SFT (Backbone) RLVR (Backbone) Post-CL (Embedding)

Mbase vs Mreason Mbase vs Mreason MEmbbase vs MEmbreason

- 1. Representation Coordinate Basis Destructive Mixing Preserved Re-Aligned

- 2. Geometry

Global Geometry Anisotropic Distortion Isometric Stable Local Geometry Reorganized Reorganized Irreversible

- 3. Function Linear Readout Degraded Transferred Aligned Manifold Status Fundamental Restructuring Trajectory Optimization Manifold Realignment

or near-permutation, then features can be matched almost one-to-one after a simple rotation or permutation. This corresponds to relatively localized, interpretable feature correspondences. In contrast, if O⋆ is dense, then each feature in one model is a distributed combination of many features in the other. The same information may be present, but in an entangled, non-localized form. We consider the inverse row entropy of O∗ as the quantified metric. See Appendix B.1.2 for the details.

#### 3.3 Geometry-Level Analysis

Geometry-level analysis moves one step up in abstraction. Instead of caring about the specific coordinate system, we focus on the shape of the latent manifold. At this level, orthogonal rotations and neuron permutations are treated as irrelevant; the relative arrangement of points is paramount. Geometry-level metrics are therefore invariant to changes of coordinate basis, but sensitive to deformations that alter distances or local neighborhoods. See Appendix B.2.

Conceptually, this level investigates whether two models organize embeddings into similar manifold shapes even using different axes. We study geometry-level similarity using two complementary metrics:

- 1. Linear Centered Kernel Alignment (Linear CKA): measures global geometry of manifold, via their Gram matrices, up to orthogonal transforms and isotropic scaling (Kornblith et al., 2019).
- 2. k-Nearest Neighbors (k-NN) Overlap: measures local geometry of manifold by quantifying the preservation of nearest-neighbor relationships (Lin & Smith, 2019).

##### 3.3.1 Linear CKA

CKA compares representations via their kernel matrices rather than their raw coordinates. In Linear CKA, we consider the linear kernel KX = XX⊤ and KY = YY⊤, where KX,KY ∈ RN×N. The Hilbert–Schmidt Independence Criterion (HSIC) (Gretton et al., 2005) between the kernels KX and KY is

1 (N − 1)2

tr(KXHKY H). (3)

HSIC(KX,KY ) =

where H = I − N1 11⊤ ∈ RN×N is the centering matrix, I is the identity matrix and 1 is a vector of 1. Linear CKA is then

HSIC(KX,KY ) HSIC(KX,KX)HSIC(KY ,KY )

. (4)

CKA(X,Y) =

Linear CKA quantifies how similarly two models organize the global geometry of manifolds, capturing features like cluster structure and anisotropy, by assessing whether their manifolds can be aligned through orthogonal transformations and uniform scaling, without the need for nonlinear deformations.

Harvey et al. (2025) has shown that Linear CKA quantifies the average alignment between optimal linear readouts across a distribution of decoding tasks. However, it can be manipulated without large changes in functional behavior under high dimensions (Davari et al., 2023; Hayne et al., 2024), so it should not be interpreted as a direct proxy for linear separability or task equivalence. Here, we use Linear CKA specifically as a global geometry descriptor.

LLM

BaseModelLayerIndex

###### EmbedModel

Dimension-Wise Correlation Linear CKA SFT RLVR SFT RLVR

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Qwen2.5 vs DS DS vs ProRL Qwen2.5 vs DS DS vs ProRL

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Qwen2.5-Emb vs DS-Emb DS-Emb vs ProRL-Emb Qwen2.5-Emb vs DS-Emb DS-Emb vs ProRL-Emb

Reasoning Model Layer Index

[Figure 13]

- Figure 3: Heatmap of Dimension-Wise Correlation (left) and Linear CKA (right). Columns: SFT vs. RLVR. Rows: Mbase vs. Mreason and MEmbbase vs. MEmbreason.

- 3.3.2 k-NN Overlap

While CKA captures global manifold geometry, k-NN overlap focuses on local manifold geometry. Intuitively, it investigates whether each embedding’s local neighborhood is preserved between two models.

⊤ i zj

Let the cosine similarity be sZ(i,j) = z

∥zi∥ ∥zj∥, where zi ∈ RD is the i-th embedding (row) of the representation matrix Z. We define the k-nearest neighbor sets under models with representations X and Y as NkX(i) = TopKj sX(i,j) and NkY (i) = TopKj sY(i,j), respectively. The k-NN overlap score s˜k is

N

NkX(i) ∩ NkY (i) NkX(i) ∪ NkY (i)

1 N

(5)

s˜k =

i=1

where we use the Jaccard index to quantify agreement of neighbor sets. Because neighborhood relations are preserved under orthogonal transforms and permutations, but disrupted by non-isometric distortions (e.g., anisotropic scaling), k-NN overlap directly reflects how similarly two models instantiate the local geometry of the manifold.

#### 3.4 Function-Level Analysis

Function-level analysis abstracts away from the internal representation manifold to focus strictly on the input–output transformations exhibited during downstream tasks. Two models may have very different representation- or geometrylevel metrics, yet still be functionally similar if the same tasks are solvable with comparable readouts. Conversely, even modest changes in embeddings can yield different behaviors under specific decoders. See Appendix B.3.

We instantiate function-level similarity with the Cross-Model Linear Probes (Nikooroo & Engel, 2025) to test whether the same linear readout generalizes across models.

##### 3.4.1 Cross-Model Linear Probes

Cross-model linear probes provide a task-conditioned measure of function-level similarity between embedding spaces. Let y ∈ RN be the labels. We first fit a linear probe on X:

yˆ = XWX + bX (6)

where (WX,bX) are learned via logistic or ridge regression, depending on the task. We then freeze (WX,bX) and apply the same linear map to representations from the other model. High cross-model performance (probe trained on

X, evaluated on Y) relative to self-performance (trained and evaluated on X) indicates that the same linear decision boundary is useful in both spaces. This implies strong function-level similarity: two models support essentially the same set of linearly decodable functions for that task.

#### 3.5 From Representations to Functions: How the Levels Fit Together

HRSA forms a hierarchy of abstraction over the same underlying representations. Representation level asks whether the latent manifolds of two models share the same coordinate basis. Geometry level discards the choice of coordinate basis and asks whether the latent manifold has a similar shape globally and locally. Function level discards most geometric detail and asks which input–output mappings are supported and realized.

Table 3: The inverse row entropy of the orthogonal matrix O∗. For each training method (SFT or RLVR), we report Mbase vs. Mreason and MEmbbase vs. MEmbreason (suffix -Emb) comparisons. A higher inverse row entropy indicates O∗ corresponds to one-to-one feature mapping.

By separating these levels, we can distinguish:

Model Pair Inverse Row Entropy ↑ SFT: Qwen2.5 vs DS 0.108

- • Cases where Mbase and Mreason differ mainly by a reparameterization (e.g., rotation) but preserve geometry and function.
- • Cases where global or local geometry changes but downstream behavior remains similar, suggesting redundant internal solutions.
- • Cases where modest representational changes induce large functional differences in readout directions, revealing sensitive or brittle aspects of the model’s decision rules.

→ Qwen2.5-Emb vs DS-Emb 0.142 RLVR: DS vs ProRL 0.161 → DS-Emb vs ProRL-Emb 0.863

This decomposition allows us to turn the initial puzzle—why MEmbbase and MEmbreason look so similar on benchmarks—into a structured investigation of where any differences live. Is the reason of the negligible deviation lived in their underlying backbone models?

### 4 Evaluation Setups

To empirically validate the hypothesis, we apply the HRSA framework across two dimensions: LLMs (Mbase vs. Mreason) and downstream adaptation (MEmbbase vs. MEmbreason). Furthermore, we extend this analysis by also comparing the SFT-tuned reasoning models, showing the clear differences in SFT-tuned and RLVR-tuned reasoning models. See Appendix D for more experiment results.

Datasets. To verify if the latent manifold is preserved even within reasoning trajectories, we construct a Chainof-Thought (CoT) dataset and use the hard-level subset for evaluation. Further generation details are provided in Appendix C. In function-level analysis, we use the AG’s News Topic Classification Dataset (Zhang et al., 2015) to evaluate the linear readout directions.

Models. For SFT comparison, we use Qwen2.5-Math-1.5B (base) (Yang et al., 2024) vs. DeepSeek-R1-Distill-Qwen1.5B (reasoning) (DeepSeek-AI, 2025). For RLVR comparison, we use DeepSeek-R1-Distill-Qwen-1.5B (base) vs. Nemotron-Research-Reasoning-Qwen-1.5B (Liu et al., 2025b) (reasoning, trained with prolonged RLVR). We abbreviate these as Qwen2.5, DS, and ProRL respectively. Downstream embedding models add “-Emb” suffix. Additional results for RLVR models with different training algorithms (GRPO (Shao et al., 2024), DAPO (Yu et al., 2025)) and training datasets are in Appendix D, showing consistent latent manifold across variations. We also demonstrate the training dynamic of manifold realignment.

Our HRSA analysis examines model activations at every layer, before any pooling. Specifically, for each model

in a matched pair with L layers, we collect the entire set of hidden states: {Xl}Ll=1 and {Yl}Ll=1, where each Xl,Yl ∈ RN×D. This per-layer, per-token perspective preserves the full representational structure for a more comprehensive analysis, avoiding any information loss due to pooling.

SFT RLVR

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Accuracy

(a) Qwen2.5 vs DS (b) Qwen2.5-Emb vs DS-Emb (c) DS vs ProRL (d) DS-Emb vs ProRL-Emb

- Figure 4: Cross-Model Linear Probe Results. For each dataset split (train, dev, test), the left bar corresponds to

Mbase (or MEmbbase) and the right bar corresponds to Mreason (or MEmbreason). The linear probe is trained on Mbase (or MEmbbase in embedding model analysis) representations and evaluated on both models. The smaller the ∆, the stronger the cross-model linear probe transfer.

- 5 Results

#### 5.1 Representation-Level Results

Dimension-Wise Correlation Figure 3 (left) shows that SFT yields weak axis-aligned feature correspondence, while RLVR retains substantially higher per-dimension correlations. Notably, the clearest deviation from diagonal structure appears only under prolonged RLVR (our main RLVR example), whereas contrastive learning largely restores axis alignment between the resulting embedding models, consistent with Manifold Realignment.

Table 4: k-NN mean overlap across layers between Mbase and Mreason (and their Emb variants). Higher mean overlap indicates more preservation in local geometry of latent manifold.

Mean Overlap ↑

Model Pairs

k=5 k=10 k=50 SFT: Qwen2.5 vs DS 0.052 0.068 0.132

Orthogonal Procrustes Analysis Table 3 supports this global alignment perspective. While SFT results in a dense orthogonal map O⋆ (implying high feature mixing), prolonged RLVR yields an O⋆ that is nearly a permutation matrix, becoming strongly permutative after contrastive learning. Table 12 shows that O⋆ is already

→ Qwen2.5-Emb vs DS-Emb 0.069 0.068 0.091 RLVR: DS vs ProRL 0.455 0.484 0.577 → DS-Emb vs ProRL-Emb 0.451 0.474 0.531

near-permutation for most Mbase vs. RLVR-tuned Mreason comparisons. This suggests RLVR does not induce feature mixing; instead, coordinate basis drift is limited to prolonged training scenarios (as in ProRL). RLVR encourages the model to construct correct paths using existing capabilities, learning only the sequence of feature activations required for rewards. Thus, the coordinate basis remains largely unchanged.

#### 5.2 Geometry-Level Results

Linear CKA Figure 3 (right) shows a sharp contrast in global manifold geometry. Linear CKA drops under SFT but remains high under RLVR, consistent with an approximately isometric relationship. After contrastive learning, the MEmbbase and MEmbreason move even closer in CKA, highlighting Manifold Realignment at the geometry level. RLVR functions as a near-isometric transformation, rigidly preserving the shape of the latent manifold. Consequently, the semantic distances established during pre-training remain invariant, which explains why downstream embedding performance does not improve.

k-NN Overlap Table 4 shows that RLVR preserves substantially more local structure (higher mean overlap) than SFT, yet overlap remains substantially below 1, indicating local geometry reorganization. This gap persists even when the embedding model manifolds are pulled closer by contrastive learning, showing the idea that RLVR introduces irreversible local geometry reorganization, which is different to the rigid global geometry. We hypothesize that this irreversible local reorganization reflects RLVR optimization in grouping related reasoning steps effectively, clusters the decision trajectory without altering the global semantic map.

#### 5.3 Function-Level Results

Cross-Model Linear Probes Figure 4 shows stronger cross-model probe transfer under RLVR than SFT, implying that task-relevant linear readout directions of the latent manifold are more stable. For the embedding model pairs, transfer remains consistently high, reflecting Manifold Realignment: contrastive learning maintains strong functional alignment even when local geometry do not fully coincide.

#### 5.4 Manifold Realignment in Training Dynamics

Figure 5 illustrates the dynamics of adapting LLMs to embedding models over training steps. By applying HRSA to intermediate checkpoints, we observe that manifold realignment occurs rapidly in the early training stages (Steps 0–200), after which representational similarity stabilizes. This trajectory demonstrates the manifold realignment, which contrastive learning effectively drives strong alignment between base- and reasoning-initialized embedding models. In contrast, the k-NN mean overlap across layers decreases during this process, confirming that the RLVR-induced reorganization of local geometry is irreversible.

[Figure 18]

Figure 5: The training dynamics of the embedding model pairs DS-Emb vs ProRL-Emb. Step 0 indicates LLM backbones, and step 781 indicates the final checkpoint of the embedding models.

### 6 Related Works

#### 6.1 Reinforcement Learning with Verifiable Rewards (RLVR)

RLVR optimizes models using deterministic, verifiable rewards rather than heuristic preference signals (DeepSeek-AI, 2025). Recent analyses suggest RLVR stays close to the pretrained solution (e.g., KL-anchored/on-policy behavior) (Shenfeld et al., 2025) and improves via weight updates that avoid large principal-subspace changes (Zhu et al., 2025a), without introducing fundamentally novel reasoning beyond the base model (Yue et al., 2025). However, these works do not directly characterize the representational changes induced by RLVR; we show (via HRSA) that RLVR largely preserves global manifold structure while reorganizing local geometry and, with prolonged training, exhibiting some coordinate basis drift.

#### 6.2 Embedding Models

Many state-of-the-art text embedding models now leverage decoder-only LLM backbones with bidirectional attention and contrastive training to produce strong encoders (Zhang et al., 2025; Lee et al., 2025b). While reward-driven or RLbased embedding learning has been explored (Tennenholtz et al., 2024; Gui & Cheng, 2025), it remains unclear whether RLVR-tuned reasoning models improve embedding geometry or retrieval. Our study directly tests this connection and finds that RLVR-tuned reasoning models do not reliably enhance embedding quality.

#### 6.3 Representational Similarity Analysis

Representational similarity analysis (RSA) and related metrics (e.g., CKA) are widely used to compare layer representations across models and tasks (Kriegeskorte et al., 2008; Kornblith et al., 2019; Klabunde et al., 2023; Yousefi et al., 2023; Liu et al., 2025c). Prior work typically reports single-level alignment and does not organize how changes manifest across abstraction levels, nor does it connect RLVR update properties to representation geometry (Shenfeld

et al., 2025; Balashov, 2025). We address this with HRSA, which disentangles coordinate basis, manifold geometry, and readout-direction changes, showing substantial global preservation in RLVR-tuned reasoning models.

### 7 Discussion and Conclusion

In this paper, we introduced HRSA, a hierarchical representation similarity analysis framework for diagnosing how training reshapes the latent manifold, and conducted the first systematic benchmarking of RLVR-optimized vs. its base model as backbones for text embedding models. Applying HRSA to base backbones and their RLVR-tuned backbones, we identified which components of the latent manifold change and characterized a consistent pattern we term manifold realignment. Across settings, RLVR largely preserves global geometry and linear readout, while producing irreversible reorganization of local geometry. Coordinate basis drift emerges primarily under prolonged RLVR, but appears reversible: subsequent contrastive learning corrects this drift and reinstates strong realignment.

These results support the view that RLVR primarily optimizes trajectories through an existing semantic landscape rather than rewriting that landscape itself. As latent-space-centric paradigms such as World Models (Ha & Schmidhuber, 2018) and JEPA (Huang et al., 2025) gain prominence, our findings point to a practical trade-off: RLVR tends to preserve the base model’s representational backbone (which may help retain broad generalization), yet on its own is unlikely to fundamentally improve the underlying global organization of the latent manifold. Put differently, RLVR seems to “move behavior” mainly by reshaping local geometry (how nearby states relate) while leaving the large-scale coordinate system and linear readout mostly intact.

Our analysis also suggests an actionable hypothesis for training design: if RLVR’s distinctive footprint is local geometry reorganization under global geometry stability, then similar behavior might be achievable via SFT augmented with geometry- and basis-aware regularization. For example, one could explicitly constrain global manifold distances or penalize excessive coordinate basis drift while encouraging controlled local geometry reorganization. Testing whether such constrained SFT can match RLVR’s representational effects offers a concrete direction for follow-up work.

Several open questions remain about the mechanism. In particular, we do not yet fully explain why RLVR produces persistent local geometry reorganization while leaving global geometry and linear readout directions relatively stable, nor what training signals govern the onset and reversibility of coordinate basis drift. Progress here may require controlled interventions (e.g., reward shaping, curriculum, or KL/entropy constraints) paired with HRSA to isolate which components of the RLVR objective drive each geometric effect.

Finally, while our experiments focus on text embedding models, the hierarchy of effects uncovered by HRSA reflects a training-agnostic geometric signature rather than a modality-specific artifact. We therefore expect manifold realignment to be a general phenomenon that extends to representation learning in vision and audio, and we position HRSA as a practical diagnostic to verify this claim across modalities and objectives.

### References

An, C., Xie, Z., Li, X., Li, L., Zhang, J., Gong, S., Zhong, M., Xu, J., Qiu, X., Wang, M., and Kong, L. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025. URL https: //hkunlp.github.io/blog/2025/Polaris.

Balashov, A. Reinforcement learning fine-tunes a sparse subnetwork in large language models. arXiv preprint arXiv:2507.17107, 2025.

Beyer, A., Kauermann, G., and Schütze, H. Embedding space correlation as a measure of domain similarity. In LREC,

pp. 2431–2439. European Language Resources Association, 2020. Bhaskar, A., Ye, X., and Chen, D. Language models that think, chat better. CoRR, abs/2509.20357, 2025. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R.,

Hesse, C., and Schulman, J. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. Davari, M., Horoi, S., Natik, A., Lajoie, G., Wolf, G., and Belilovsky, E. Reliability of CKA as a similarity measure in deep learning. In ICLR. OpenReview.net, 2023.

de Souza Pereira Moreira, G., Osmulski, R., Xu, M., Ak, R., Schifferer, B., and Oldridge, E. Nv-retriever: Improving text embedding models with effective hard-negative mining. CoRR, abs/2407.15831, 2024.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948, 2025.

DeepSeek-AI. Deepseek-v3.2-exp: Boosting long-context efficiency with deepseek sparse attention, 2025.

Enevoldsen, K. C., Chung, I., Kerboua, I., Kardos, M., Mathur, A., Stap, D., Gala, J., Siblini, W., Krzeminski, D., Winata, G. I., Sturua, S., Utpala, S., Ciancone, M., Schaeffer, M., Misra, D., Dhakal, S., Rystrøm, J., Solomatin, R., Çagatan, Ö. V., Kundu, A., and et al. MMTEB: massive multilingual text embedding benchmark. In ICLR. OpenReview.net, 2025.

Gao, T., Yao, X., and Chen, D. Simcse: Simple contrastive learning of sentence embeddings. CoRR, abs/2104.08821, 2021.

Gretton, A., Bousquet, O., Smola, A., and Schölkopf, B. Measuring statistical dependence with hilbert-schmidt norms. In Algorithmic Learning Theory (ALT 2005), 16th International Conference, Proceedings, volume 3734 of Lecture Notes in Artificial Intelligence, pp. 63–77. Springer, 2005.

Gui, Y. and Cheng, J. Search-r3: Unifying reasoning and embedding generation in large language models. CoRR,

abs/2510.07048, 2025. Ha, D. and Schmidhuber, J. World models. CoRR, abs/1803.10122, 2018. Harvey, S. E., Lipshutz, D., and Williams, A. H. What representational similarity measures imply about decodable

information. In UniReps, volume 285 of Proceedings of Machine Learning Research, pp. 140–151. PMLR, 2025. Hayne, L., Jung, H., and Carter, R. M. Does representation similarity capture function similarity? Trans. Mach. Learn.

Res., 2024, 2024. Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the MATH dataset. In NeurIPS Datasets and Benchmarks, 2021. Huang, H., LeCun, Y., and Balestriero, R. LLM-JEPA: large language models meet joint embedding predictive architectures. CoRR, abs/2509.14252, 2025. Klabunde, M., Amor, M. B., Granitzer, M., and Lemmerich, F. Towards measuring representational similarity of large language models. CoRR, abs/2312.02730, 2023. Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. E. Similarity of neural network representations revisited. In ICML, volume 97 of Proceedings of Machine Learning Research, pp. 3519–3529. PMLR, 2019.

Kriegeskorte, N., Mur, M., and Bandettini, P. A. Representational similarity analysis - connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience, Volume 2 - 2008, 2008. ISSN 1662-5137. doi: 10. 3389/neuro.06.004.2008. URL https://www.frontiersin.org/journals/systems-neuroscience/articles/ 10.3389/neuro.06.004.2008.

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N., Lyu, S., Gu, Y., Malik, S., Graf, V., Hwang, J. D., Yang, J., Bras, R. L., Tafjord, O., Wilhelm, C., Soldaini, L., Smith, N. A., Wang, Y., Dasigi, P., and Hajishirzi, H. Tülu 3: Pushing frontiers in open language model post-training. CoRR, abs/2411.15124, 2024.

Lee, C., Roy, R., Xu, M., Raiman, J., Shoeybi, M., Catanzaro, B., and Ping, W. Nv-embed: Improved techniques for training llms as generalist embedding models. In ICLR. OpenReview.net, 2025a.

Lee, J., Chen, F., Dua, S., Cer, D., Shanbhogue, M., Naim, I., Ábrego, G. H., Li, Z., Chen, K., Vera, H. S., Ren, X., Zhang, S., Salz, D., Boratko, M., Han, J., Chen, B., Huang, S., Rao, V., Suganthan, P., Han, F., Doumanoglou, A., Gupta, N., Moiseev, F., Yip, C., Jain, A., Baumgartner, S., Shahi, S., Gomez, F. P., Mariserla, S., Choi, M., Shah, P., Goenka, S., Chen, K., Xia, Y., Chen, K., Duddu, S. M. K., Chen, Y., Walker, T., Zhou, W., Ghiya, R., Gleicher, Z., Gill, K., Dong, Z., Seyedhosseini, M., Sung, Y., Hoffmann, R., and Duerig, T. Gemini embedding: Generalizable embeddings from gemini. CoRR, abs/2503.07891, 2025b.

Lin, L. H. and Smith, N. A. Situating sentence embedders with nearest neighbor overlap. CoRR, abs/1909.10724, 2019. Liu, J., Liu, H., Xiao, L., Wang, Z., Liu, K., Gao, S., Zhang, W., Zhang, S., and Chen, K. Are your llms capable of

stable reasoning? In ACL (Findings), pp. 17594–17632. Association for Computational Linguistics, 2025a. Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., and Dong, Y. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. CoRR, abs/2505.24864, 2025b. Liu, X., Hsiung, L., Yang, Y., and Yan, Y. Spectral insights into data-oblivious critical layers in large language models. CoRR, abs/2506.00382, 2025c. Mikolov, T., Chen, K., Corrado, G., and Dean, J. Efficient estimation of word representations in vector space. In ICLR (Workshop Poster), 2013. Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. MTEB: massive text embedding benchmark. In EACL, pp. 2006–2029. Association for Computational Linguistics, 2023. Muennighoff, N., Su, H., Wang, L., Yang, N., Wei, F., Yu, T., Singh, A., and Kiela, D. Generative representational

instruction tuning. CoRR, abs/2402.09906, 2024. Nikooroo, S. and Engel, T. Cross-model semantics in representation learning. CoRR, abs/2508.03649, 2025. Schönemann, P. H. A generalized solution of the orthogonal procrustes problem. Psychometrika, 31(1):1–10, 1966. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. CoRR,

abs/1707.06347, 2017. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. Shenfeld, I., Pari, J., and Agrawal, P. Rl’s razor: Why online reinforcement learning forgets less. CoRR, abs/2509.04259, 2025.

Su, H., Yen, H., Xia, M., Shi, W., Muennighoff, N., Wang, H., Liu, H., Shi, Q., Siegel, Z. S., Tang, M., Sun, R., Yoon, J., Arik, S. Ö., Chen, D., and Yu, T. BRIGHT: A realistic and challenging benchmark for reasoning-intensive retrieval. In ICLR. OpenReview.net, 2025.

Tennenholtz, G., Chow, Y., Hsu, C., Shani, L., Liang, E., and Boutilier, C. Embedding-aligned language models. CoRR, abs/2406.00024, 2024.

van den Oord, A., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. CoRR, abs/1807.03748, 2018.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., Li, T., Ku, M., Wang, K., Zhuang, A., Fan, R., Yue, X., and Chen, W. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In NeurIPS, 2024.

Xu, S., Zhou, Y., Wang, W., Min, J., Yin, Z., Dai, Y., Liu, S., Pang, L., Chen, Y., and Zhang, J. Tiny model, big logic: Diversity-driven optimization elicits large-model reasoning ability in vibethinker-1.5 b. arXiv preprint arXiv:2511.06221, 2025.

Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3 technical report. CoRR, abs/2505.09388, 2025.

Yousefi, S., Betthauser, L., Hasanbeig, H., Millière, R., and Momennejad, I. Decoding in-context learning: Neuroscienceinspired analysis of representations in large language models. arXiv preprint arXiv:2310.00313, 2023.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Fan, T., Liu, G., Liu, L., Liu, X., Lin, H., Lin, Z., Ma, B., Sheng, G., Tong, Y., Zhang, C., Zhang, M., Zhang, W., Zhu, H., Zhu, J., Chen, J., Chen, J., Wang, C., Yu, H., Dai, W., Song, Y., Wei, X., Zhou, H., Liu, J., Ma, W., Zhang, Y., Yan, L., Qiao, M., Wu, Y., and Wang, M. DAPO: an open-source LLM reinforcement learning system at scale. CoRR, abs/2503.14476, 2025.

Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., and Huang, G. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

- Zhang, X., Zhao, J., and LeCun, Y. Character-level convolutional networks for text classification, 2015.
- Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., and Zhou, J. Qwen3 embedding: Advancing text embedding and reranking through foundation models. CoRR, abs/2506.05176, 2025.

Zheng, C., Liu, S., Li, M., Chen, X., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., Zhou, J., and Lin, J. Group sequence policy optimization. CoRR, abs/2507.18071, 2025.

Zhu, H., Zhang, Z., Huang, H., Su, D., Liu, Z., Zhao, J., Fedorov, I., Pirsiavash, H., Sha, Z., Lee, J., et al. The path not taken: Rlvr provably learns off the principals. arXiv preprint arXiv:2511.08567, 2025a.

Zhu, X., Xia, M., Wei, Z., Chen, W., Chen, D., and Meng, Y. The surprising effectiveness of negative reinforcement in LLM reasoning. CoRR, abs/2506.01347, 2025b.

### A Embedding Model Training

In this section, we reveal all the training details of the embedding models.

#### A.1 Training Details

We optimize the InfoNCE loss (van den Oord et al., 2018) defined in Equation 7. This objective aims to maximize the similarity between the query q and the positive passage p, while simultaneously minimizing the similarity between q and the negative passages. Let B denote the set of in-batch passages (which includes p and negatives from other instances), N be the set of hard negatives, and sim be the cosine similarity. The loss is calculated as:

exp(sim(q,p)/τ) d∈B∪N exp(sim(q,d)/τ)

(7)

L(q,p,B,N) = −log

We select decoder-only LLMs as the embedding model backbone, take the last layer’s activation as the final output, and perform mean pooling to obtain a fixed-dimension embedding vector. We also enable bi-directional attention in the backbone by discarding the causal attention mask to capture more semantic details and relationships between tokens. We use mixed precision with bfloat16 and gradient checkpointing to reduce the memory pressure on the hardware. We use Flash Attention 2 as the attention backend algorithm. For more details on the settings, the reader can refer to Table 5. We employ the instruction-tuning technique. In particular, we use the instruction template Instruction:

{instruction}\nQuery: query, where {instruction} and {query} are the placeholders for the instruction and query, respectively. All of our training is conducted on 4x Nvidia L20 GPUs, with VRAM 44GB per GPU.

Table 5: Training Hyperparameters

Variables Values

Batch Size 2048 Learning Rate (LR) 2 × 10−5 LR Warm-up Ratio 0.03 LR Scheduler Cosine Weight Decay 0.05 Optimizer AdamW Padding Side Right Number of data 1,603,172 Number of training steps 782 Number of hard negatives 3 Temperature 0.02 Pooling Mean

Although many prior works (Zhang et al., 2025; Lee et al., 2025b,a) use LoRA to train the embedding models, in our work, we discard it, since we find that training without LoRA yields better performance, and full parameters can better record the training dynamics. See Table 6.

Table 6: LoRA comparison on performance in MTEB (Multilingual, v2).

Model Performance With LoRA

DS-Distill-Qwen-1.5B-Emb 42.450 NV-ProRL-Emb 42.064

###### Without LoRA

DS-Distill-Qwen-1.5B-Emb 46.185 NV-ProRL-Emb 46.247

#### A.2 Training Data Statistics

We consider a wide range of datasets, forming the training dataset by composing 11 separate datasets. We used Qwen3-Embedding-0.6B (Zhang et al., 2025) to mine 3 hard negatives per query, and employ the positive-aware hard negative mining technique introduced in de Souza Pereira Moreira et al. (2024), with 95% margin to the positive score.

Table 7: Training Datasets Details

Datasets Number of Samples

FEVER 105,893 NaturalQuestions 97,912 NLI 277,217 MSMARCO 499,184 Quora 94,443 Mr.Tydi 102,796 DUReader 17,493 TriviaQA 65,465 HotpotQA 167,808 SQuAD 84,494 T2Ranking 90,467

Total 1,603,172

### B HRSA Proof

In this section, we provide more details on HRSA, including all the invariance properties of each level analysis and the proof of their invariance properties.

We emphasize again that HRSA is not dependent on the specific metrics selected for this study, such as DimensionWise Correlation or Linear CKA. Rather, it is grounded in the hierarchy of invariance properties established earlier. Consequently, any metric that satisfies the invariance requirements of a specific level can be employed to analyze that level’s focus. Refer to Table 8 for a summary of these properties and a catalog of alternative valid metrics.

#### B.1 Representation-Level Proof

- Definition 1 (Representation-Level Analysis). The representation-level analysis examines the explicit coordinate basis of the latent manifold. A metric at this level must demonstrate sensitivity to coordinate basis rotations. Specifically:

• Non-invariant to: Orthogonal transformations (Rotation/Permutation) and General Linear transformations.

- B.1.1 Dimension-Wise Correlation Recall the definition of Dimension-Wise Correlation from equation 1.

- Proposition 1. Dimension-Wise Correlation is non-invariant to orthogonal transformations.

Proof. Let Q ∈ RD×D be an orthogonal matrix (Q⊤Q = I) such that X′ = XQ. The j-th column becomes x′:j = Dk=1 X:kQkj. The correlation of the j-th column becomes:

( k X:kQkj)⊤y:j ∥ k X:kQkj∥2∥y:j∥2

. (8)

ρj(XQ,Y ) =

Since Q mixes information from multiple columns X:k into the new column x′:j, the correlation with the fixed target y:j changes arbitrarily depending on Q. Thus, ρj(XQ,Y ) ̸= ρj(X,Y ), satisfying the requirement for coordinate basis sensitivity.

| |
|---|

Table 8: HRSA Framework Extensibility. The HRSA framework is defined by invariance properties, not specific metrics. Researchers can select alternative metrics (right column) for different modalities or theoretical needs, provided they respect the invariance constraints of the target analysis level.

Level Invariance Constraints Default Metric Alternative Valid Metrics

Non-Invariant to: Dimension-Wise

Optimal Transport (Wasserstein) Orthogonal Transformation Measures cost to move mass from basis

Correlation

Representation

X to Y without rotation.

Goal: Assess alignment Orthogonal

###### Manifold Alignment Loss

Procrustes (O∗)

of specific axes. Direct penalization of feature mismatch.

Invariant to:

RBF Kernel CKA Orthogonal Transformation Captures non-linear similarity. Non-Invariant to:

Linear CKA

Geometry

###### Riemannian Metrics

k-NN Overlap (Jaccard)

Invertible Linear Transforms

Geodesic distance comparison.

(Scaling/Shear)

Invariant to: Linear Probing

Mutual Information I(X; Y ) Any transform preserving Information theoretic upper bound. the decision boundary. Zero-Shot

Transfer

Function

###### Behavioral Consistency

Accuracy

Exact match on downstream tasks.

##### B.1.2 Orthogonal Procrustes Analysis

Recall the Orthogonal Procrustes solution O∗ defined in Equation 2. To quantify the extent of coordinate alignment, we introduce the inverse row entropy, denoted as Hinv. We interpret the squared elements of each row in O∗ as a probability distribution. This is mathematically valid because O∗ is orthogonal, meaning its rows have unit Euclidean norm (i.e.,

j(Oij∗ )2 = 1).

We compute Hinv by calculating the mean row entropy, normalizing it by the maximum possible entropy (log D), and taking the complement:

D

1 D log D

H = −

i=1

D

(Oij∗ )2 log(Oij∗ )2

j=1

Hinv = 1 − H

where Oij∗ denotes the element of O∗ at row i and column j, and D represents the dimensionality. The intermediate term H is normalized to the range [0,1]. Consequently, a higher Hinv indicates that the coordinate basis is preserved (i.e., O∗ is sparse and approximates a permutation matrix), whereas a lower Hinv indicates that features are "smeared" or rotated across multiple dimensions.

- Proposition 2. The structure of the optimal mapping in Orthogonal Procrustes Analysis is non-invariant to orthogonal transformations.

Proof. For any orthogonal matrices Q,R ∈ RD×D, if we transform X,Y to X′ = XQ, Y ′ = Y R, then an optimal map for the new problem is

O∗(X′,Y ′) = Q⊤O∗(X,Y )R. (9) This is a conjugation of O∗ by orthogonal matrices, which in general destroys diagonality or one-hot structure.

| |
|---|

Remark. One may claim that Orthogonal Procrustes Analysis should be classified as a geometry-level measurement because the residual is invariant to orthogonal transformation. In our work, we only focus on the structure of O∗, specifically by considering its inverse row entropy. As shown in Proposition 2, O∗ remains dependent on the chosen coordinate system.

#### B.2 Geometry-Level Proof

- Definition 2 (Geometry-Level Analysis). The geometry-level analysis examines the intrinsic shape and topology of the latent manifold Z. Metrics at this level must quantify the arrangement of points relative to one another, independent of the specific coordinate system used to describe them.

- • Invariant to: Similarity transformations, defined as the composition of orthogonal rotation/reflection (Q ∈ RD×D,Q⊤Q = I) and isotropic scaling (c ∈ R,c > 0).
- • Non-invariant to: Anisotropic linear transformations (e.g., non-uniform scaling, shearing) where the transformation matrix A satisfies A⊤A ̸= cI.

In the following, we provide proofs for the invariance properties of Linear CKA and Cosine k-NN Overlap.

##### B.2.1 Linear CKA

Recall that Linear CKA is defined via the Hilbert–Schmidt Independence Criterion (HSIC) of centered Gram matrices. Let KX = XX⊤ and H = I − N1 11⊤.

- Proposition 3. Linear CKA is invariant to similarity transformations X  → cXQ where c > 0 and Q is orthogonal. Proof. Let X′ = cXQ. We first derive the Gram matrix for the transformed representation:

KX′ = (cXQ)(cXQ)⊤ = c2XQQ⊤X⊤. (10) Since Q is orthogonal (QQ⊤ = I), this simplifies to:

KX′ = c2XX⊤ = c2KX. (11) Now we examine the HSIC term in the numerator. Using the property tr(cA) = ctr(A):

HSIC(KX′,KY ) =

1 (N − 1)2

tr(KX′HKY H)

=

1 (N − 1)2

tr(c2KXHKY H)

= c2 · HSIC(KX,KY ).

(12)

Similarly, for the normalization term in the denominator:

HSIC(KX′,KX′) =

1 (N − 1)2

tr(c2KXHc2KXH)

= c4 · HSIC(KX,KX).

(13)

Substituting these into the full Linear CKA equation:

CKA(X′,Y ) =

c2HSIC(KX,KY ) c4HSIC(KX,KX) · HSIC(KY ,KY )

=

c2HSIC(KX,KY ) c2 HSIC(KX,KX) · HSIC(KY ,KY )

= CKA(X,Y ).

(14)

The scalar factors cancel perfectly, proving invariance.

| |
|---|

- Proposition 4. Linear CKA is generally non-invariant to anisotropic linear transformations. Proof. Let X′ = XA, where A ∈ RD×D is invertible and anisotropic (AA⊤ ̸= cI). The Gram matrix becomes:

###### KX′ = XAA⊤X⊤. (15)

Let M = AA⊤. The numerator HSIC term becomes proportional to tr(XMX⊤HKY H). Unlike the isotropic case, the matrix M is "trapped" between X and X⊤ inside the trace. Unless M is a scalar multiple of the identity, it reweights the singular values of X, effectively altering the principal components of the representation space. Since Linear CKA measures the alignment of these principal components, CKA(XA,Y ) ̸= CKA(X,Y ).

| |
|---|

##### B.2.2 k-NN Overlap

⊤v ∥u∥∥v∥.

As defined in the Section 3.3.2, k-NN overlap relies on the ranking of cosine similarities s(u,v) = u

Proposition 5. Cosine-based k-NN Overlap is invariant to similarity transformations. Proof. Let x and y be any two embedding vectors (rows of X). We apply the transformation x′ = cQx and y′ = cQy, with c > 0 and Q⊤Q = I. The cosine similarity between the transformed vectors is:

(cQx)⊤(cQy) ∥cQx∥∥cQy∥

s(x′,y′) =

c2x⊤Q⊤Qy (cQx)⊤(cQx) (cQy)⊤(cQy)

.

=

Using the orthogonality property Q⊤Q = I:

(16)

c2x⊤y √

s(x′,y′) =

c2x⊤x c2y⊤y

c2(x⊤y) c∥x∥ · c∥y∥

(17)

=

x⊤y ∥x∥∥y∥

=

= s(x,y).

Since the pairwise similarity scores remain exactly the same, the ranking of neighbors is preserved. Thus, the set of top-k nearest neighbors is identical: NX

′

k (i) = NkX(i), and the overlap score is invariant. Proposition 6. Cosine-based k-NN Overlap is generally non-invariant to anisotropic linear transformations. Proof. Let x′ = Ax and y′ = Ay with anisotropic A. The transformed similarity is:

| |
|---|

x⊤A⊤Ay √

s(x′,y′) =

. (18)

x⊤A⊤Ax y⊤A⊤Ay

Let M = A⊤A. This expression represents the cosine of the angle between x and y in a space equipped with the inner product ⟨u,v⟩M = u⊤Mv.

Because A is anisotropic, M has distinct eigenvalues. This transformation distorts angles: vectors aligned with the large eigenvectors of M are "pulled" closer together in angular space, while vectors aligned with small eigenvectors are pushed apart.

Consequently, if we have s(x,y) > s(x,z) (meaning y is a closer neighbor to x than z), an anisotropic A can reverse this relationship such that s(x′,z′) > s(x′,y′). This alters the composition of the k-nearest neighbor sets, changing the overlap score.

| |
|---|

#### B.3 Function-Level Proof

- Definition 3 (Function-Level Analysis). The function-level analysis examines the usable information accessible via linear readouts (probes) or the final behavioral output. This level specifically tests whether two models share the same

“readout directions” for solving a task.

- • Invariant to: Isomorphic transformations if and only if the readout mechanism is transformed correspondingly.
- • Non-invariant to: Linear Reparameterization under a fixed readout hypothesis.

##### B.3.1 Cross-Model Linear Probes

Let wX∗ be the optimal probe weights for task Z on representations X, i.e., wX∗ = argminw∥Xw − Z∥2. We evaluate these weights on Y : Error = ∥Y wX∗ − Z∥2.

###### Table 9: LLM pairs used in additional HRSA analyses, separated by the training algorithms (SFT, RLVR).

Base Model Mbase Reasoning Model Mreason Algorithm Data SFT Qwen2.5-Math-1.5B DeepSeek-R1-Distill-Qwen-1.5B SFT Mixed RLVR Qwen3-4B Polaris-4B-Preview DAPO Math DeepSeek-R1-Distill-Qwen-7B Polaris-7B-Preview DAPO Math Qwen2.5-7B zero__ppo__think__Qwen2.5-7B PPO Chat Qwen2.5-1.5B Qwen-2.5-1.5B-SimpleRL-Zoo GRPO Math Qwen2.5-0.5B Qwen-2.5-0.5B-SimpleRL-Zoo GRPO Math DeepSeek-R1-Distill-Qwen-1.5B Nemotron-Research-Reasoning-Qwen-1.5B GRPO Math Qwen3-4B Qwen3-4B-PSR PSR Math

- Table 10: Embedding model pairs used in additional HRSA analyses. All of the embedding models are trained on the same dataset with InfoNCE loss. They are separated by the training algorithms used to train their reasoning model backbone.

Base Embedding Model MEmbbase Reasoning Embedding Model MEmbreason SFT Qwen2.5-Math-1.5B-Emb DeepSeek-R1-Distill-Qwen-1.5B-Emb Qwen3-0.6B-Base-Emb Qwen3-0.6B-Emb RLVR Qwen2.5-1.5B-Emb Qwen-2.5-1.5B-SimpleRL-Zoo-Emb Qwen2.5-0.5B-Emb Qwen-2.5-0.5B-SimpleRL-Zoo-Emb DeepSeek-R1-Distill-Qwen-1.5B-Emb Nemotron-Research-Reasoning-Qwen-1.5B-Emb Qwen3-4B-Emb Qwen3-4B-PSR-Emb

Proposition 7. Cross-Model Linear Probes are non-invariant to linear reparameterization under a fixed readout. Proof. Assume Y contains the exact same information as X but is linearly transformed: Y = XA (where A is invertible). The prediction using transferred weights is:

###### ZˆY = Y wX∗ = (XA)wX∗ . (19)

The original prediction was ZˆX = XwX∗ . For the predictions to be identical (ZˆY = ZˆX) for all X, we require XAwX∗ = XwX∗ , implying AwX∗ = wX∗ . This equality only holds if wX∗ is an eigenvector of A with eigenvalue 1. For a general transformation A, AwX∗ ̸= wX∗ . Therefore, even if Y is geometrically isomorphic to X, the cross-model probe will fail if the direction of the solution has shifted. This proves the metric satisfies the requirement set in Definition 3.

| |
|---|

### C CoT Datasets

To rigorously verify if the latent manifold is preserved within reasoning trajectories (as discussed in Section 4), we constructed a specialized CoT-Activations dataset. Unlike standard semantic datasets, this corpus focuses on long-range, multi-step reasoning traces generated by state-of-the-art reasoning models.

#### C.1 Dataset Composition and Hierarchy

We curated a diverse suite of mathematical reasoning benchmarks to ensure our analysis covers varying degrees of reasoning complexity, ranging from elementary arithmetic to competition-level problem solving. The dataset is stratified into three difficulty levels:

- • Easy: Sourced from GSM8K (Cobbe et al., 2021), focusing on grade-school math word problems that require multi-step arithmetic but limited abstract reasoning.
- • Moderate: Sourced from MATH-500 (Hendrycks et al., 2021) (a curated subset of the MATH benchmark including AMC/AIME problems) and NuminaMath (CN K-12 curriculum). These datasets introduce higherdimensional algebraic and geometric reasoning.
- • Hard: Sourced from LiveMathBench (Liu et al., 2025a) (2025 Hard Subset). These are recent competition-level problems requiring extremely long context windows and complex logical deductions.

Table 11 summarizes the statistics of the generated CoT dataset.

#### C.2 Generation Protocol

To extract high-quality reasoning traces, we utilized Qwen3-32B (Yang et al., 2025) as the generator backbone. The generation process was designed to maximize the explicitness of the internal reasoning process (the “chain of thought”).

Inference Configuration. We enabled the internal “thinking” mode (enable_thinking: true) to expose the raw reasoning tokens before the final answer. The generation parameters were set to temperature T = 0.6 and nucleus sampling probability p = 0.95 to balance creativity with logical coherence.

Token Limits. To accommodate deep reasoning, we set a high context limit. For standard datasets, we allowed up to 8,000 reasoning tokens. For the LiveMathBench subset, we removed the CoT token limit entirely to allow for exhaustive search trajectories in hard problems.

Prompting. We employed a standardized two-message chat format to enforce rigorous step-by-step reasoning. The System Prompt was defined as:

- Prompt 1: System Prompt of the CoT dataset generation

|You are a helpful and rigorous math reasoning assistant.<br><br>|
|---|

The User Prompt wrapped the specific dataset problem with instructions to act as a competition solver:

- Prompt 2: User Prompt of the CoT dataset generation

|You are an expert competition math solver. Read the problem carefully and solve it step by step.<br><br>Problem: {Problem}<br><br>|
|---|

#### C.3 Quality Control and Evaluation

To ensure that our latent manifold analysis is based on valid reasoning trajectories rather than hallucinations, we implemented a strict verification pipeline using an LLM-as-a-Judge approach.

We employed DeepSeek-V3.2-exp (DeepSeek-AI, 2025) as the external evaluator. To ensure deterministic and strictly formatted outputs, we used greedy decoding (T = 0) and explicitly disabled the model’s internal chain-ofthought feature (enable_thinking: False).

The interaction was structured as follows:

- Prompt 3: System Prompt of the external evaluator

|You are a precise math answer evaluator. Respond only with 0 or 1.<br><br>|
|---|

- Prompt 4: User Prompt of the external evaluator

You are an expert math problem evaluator. Your task is to determine if the provided answer correctly solves the given problem.

Problem: {Problem} Answer: {Answer}

Evaluate whether the answer is correct. Respond with ONLY "1" if the answer is correct , or "0" if it is incorrect. Do not provide any explanation.

The evaluator’s output was parsed using a simple inclusion check: if the token “1” appeared in the response, the reasoning trace was marked as valid (correctness_label = 1); otherwise, it was discarded.

- Table 11: Statistics of the Generated CoT Reasoning Dataset. We report the yield of our generation pipeline across difficulty tiers. Total: Number of initial prompts; Valid: Traces that passed the correctness verification (Correctness

= 1). Acc.: The effective yield rate (Valid / Total).

Generation Statistics Dataset (Difficulty) Total Valid Acc. (%)

GSM8K (Easy) 500 471 92.80 NuminaMath (Moderate) 161 154 91.30 MATH-500 (Moderate) 479 364 75.78 LiveMathBench (Hard) 57 32 56.14

Total / Average 1,197 1,021 85.30

### D Additional Results

In this section, we demonstrate additional results of HRSA applying on more model pairs, including Mbase vs. Mreason and MEmbbase vs. MEmbreason (suffix −Emb). See Table 9 and Table 10 for the detailed model pairs.

Instead of only considering the CoT dataset, we also apply HRSA with the MMLU-Pro (Wang et al., 2024) dataset to study the difference (if any) between the models in a general field rather than only the maths domain.

#### Base Models Mbase vs. Reasoning Models Mreason Dataset: CoT Datset

[Figure 19]

Qwen2.5-Math-1.5B

DeepSeek-R1-Distill-Qwen1.5B

[Figure 20]

Qwen2.5-1.5B

Qwen-2.5-1.5B-SimpleRL-Zoo

[Figure 21]

DeepSeek-R1-Distill-Qwen-7B

Qwen3-4B

Polaris-4B-Preview

Polaris-7B-Preview

[Figure 22]

Qwen2.5-0.5B

Qwen-2.5-0.5B-SimpleRL-Zoo

[Figure 23]

DeepSeek-R1-Distill-Qwen-1.5B

Nemotron-ResearchReasoning-Qwen-1.5B

[Figure 24]

[Figure 25]

Qwen2.5-7B

zero__ppo__think__Qwen2.57B

[Figure 26]

Qwen3-4B

Qwen3-4B-PSR

#### Dataset: MMLU-Pro

[Figure 27]

Qwen2.5-Math-1.5B

DeepSeek-R1-Distill-Qwen1.5B

[Figure 28]

Qwen2.5-1.5B

Qwen-2.5-1.5B-SimpleRL-Zoo

[Figure 29]

DeepSeek-R1-Distill-Qwen-7B

Qwen3-4B

Polaris-4B-Preview

Polaris-7B-Preview

[Figure 30]

Qwen2.5-0.5B

Qwen-2.5-0.5B-SimpleRL-Zoo

[Figure 31]

DeepSeek-R1-Distill-Qwen-1.5B

Nemotron-ResearchReasoning-Qwen-1.5B

[Figure 32]

[Figure 33]

Qwen2.5-7B

zero__ppo__think__Qwen2.57B

[Figure 34]

Qwen3-4B

Qwen3-4B-PSR

- Figure 6: Additional Results on Dimension-Wise Correlation separated by dataset. The vertical axis and horizontal axis are Base Model Layer Index and Reasoning Model Layer Index, respectively. The red background indicates SFT-tuned pairs.

#### Base Embedding Models MEmbbase vs. Reasoning Embedding Models MEmbreason Dataset: CoT Datset

[Figure 35]

[Figure 36]

[Figure 37]

EmbQwen2.5-Math-1.5B-

EmbQwen3-0.6B-Base-

EmbQwen2.5-1.5B-

Qwen-2.5-1.5B-SimpleRLZoo-Emb

DeepSeek-R1-Distill-Qwen1.5B-Emb

Qwen3-0.6B-Emb

[Figure 38]

[Figure 39]

[Figure 40]

DeepSeek-R1-Distill-Qwen-1.5B-

EmbQwen2.5-0.5B-

EmbQwen3-4B-

Emb

Nemotron-ResearchReasoning-Qwen-1.5B-Emb

Qwen-2.5-0.5B-SimpleRLZoo-Emb

Qwen3-4B-PSR-Emb

[Figure 41]

#### Dataset: MMLU-Pro

[Figure 42]

[Figure 43]

[Figure 44]

EmbQwen2.5-Math-1.5B-

EmbQwen3-0.6B-Base-

EmbQwen2.5-1.5B-

DeepSeek-R1-Distill-Qwen1.5B-Emb

Qwen-2.5-1.5B-SimpleRLZoo-Emb

Qwen3-0.6B-Emb

[Figure 45]

[Figure 46]

[Figure 47]

DeepSeek-R1-Distill-Qwen-1.5B-

EmbQwen2.5-0.5B-

EmbQwen3-4B-

Emb

Nemotron-ResearchReasoning-Qwen-1.5B-Emb

Qwen-2.5-0.5B-SimpleRLZoo-Emb

Qwen3-4B-PSR-Emb

[Figure 48]

- Figure 7: Additional Results on Dimension-Wise Correlation separated by dataset. The vertical axis and horizontal axis are Base Model Layer Index and Reasoning Model Layer Index, respectively. The red background indicates their backbone LLMs are SFT-tuned pairs.

- Table 12: Inverse row entropy Hinv of the orthogonal matrix O∗ for Mbase vs. Mreason across different datasets. Higher inverse row entropy indicates more axis-aligned correspondence, while lower inverse row entropy indicates more globally mixed features. The model pairs are separated by the algorithm used to train their reasoning model Mreason.

#### Orthogonal Procrustes Analysis Base Models Mbase vs. Reasoning Models Mreason Dataset: CoT Dataset

Model Pair Hinv ↑ SFT Qwen2.5-Math-1.5B vs DeepSeek-R1-Distill-Qwen-1.5B 0.1076 RLVR

Qwen3-4B vs Polaris-4B-Preview 0.4365 DeepSeek-R1-Distill-Qwen-7B vs Polaris-7B-Preview 0.6576 Qwen2.5-7B vs zero__ppo__think__Qwen2.5-7B 0.6338 Qwen2.5-1.5B vs Qwen-2.5-1.5B-SimpleRL-Zoo 0.9481

- Qwen2.5-0.5B vs Qwen-2.5-0.5B-SimpleRL-Zoo 0.9923 DeepSeek-R1-Distill-Qwen-1.5B vs Nemotron-Research-Reasoning-Qwen-1.5B 0.1613 Qwen3-4B vs Qwen3-4B-PSR 0.8122

Dataset: MMLU-Pro

Model Pair Hinv ↑ SFT Qwen2.5-Math-1.5B vs DeepSeek-R1-Distill-Qwen-1.5B 0.2229 RLVR

Qwen3-4B vs Polaris-4B-Preview 0.9711 DeepSeek-R1-Distill-Qwen-7B vs Polaris-7B-Preview 0.9623 Qwen2.5-7B vs zero__ppo__think__Qwen2.5-7B 0.9922

- Qwen2.5-1.5B vs Qwen-2.5-1.5B-SimpleRL-Zoo 0.9963 Qwen2.5-0.5B vs Qwen-2.5-0.5B-SimpleRL-Zoo 0.9981 DeepSeek-R1-Distill-Qwen-1.5B vs Nemotron-Research-Reasoning-Qwen-1.5B 0.8336 Qwen3-4B vs Qwen3-4B-PSR 0.9843

- Table 13: Inverse row entropy Hinv of the orthogonal matrix O∗ for MEmbbase vs. MEmbreason across different datasets. Higher inverse row entropy indicates more axis-aligned correspondence, while lower inverse row entropy indicates more globally mixed features. The model pairs are separated by the algorithm used to train their reasoning model backbone Mreason.

#### Orthogonal Procrustes Analysis Base Embedding Models MEmbbase vs. Reasoning Embedding Models MEmbreason Dataset: CoT Dataset

Model Pair Hinv ↑ SFT

Qwen2.5-Math-1.5B-Emb vs DeepSeek-R1-Distill-Qwen-1.5B-Emb 0.1429 Qwen3-0.6B-Base-Emb vs Qwen3-0.6B-Emb 0.4915

RLVR Qwen2.5-1.5B-Emb vs Qwen-2.5-1.5B-SimpleRL-Zoo-Emb 0.8826

- Qwen2.5-0.5B-Emb vs Qwen-2.5-0.5B-SimpleRL-Zoo-Emb 0.9835 DeepSeek-R1-Distill-Qwen-Emb vs Nemotron-Research-Reasoning-Qwen-Emb 0.8637 Qwen3-4B-Emb vs Qwen3-4B-PSR-Emb 0.5863

Dataset: MMLU-Pro

Model Pair Hinv ↑ SFT

Qwen2.5-Math-1.5B-Emb vs DeepSeek-R1-Distill-Qwen-1.5B-Emb 0.7105 Qwen3-0.6B-Base-Emb vs Qwen3-0.6B-Emb 0.9164

RLVR

- Qwen2.5-1.5B-Emb vs Qwen-2.5-1.5B-SimpleRL-Zoo-Emb 0.9794 Qwen2.5-0.5B-Emb vs Qwen-2.5-0.5B-SimpleRL-Zoo-Emb 0.9978 DeepSeek-R1-Distill-Qwen-Emb vs Nemotron-Research-Reasoning-Qwen-Emb 0.9814 Qwen3-4B-Emb vs Qwen3-4B-PSR-Emb 0.9933

#### Base Models Mbase vs. Reasoning Models Mreason Dataset: CoT Datset

[Figure 49]

Qwen2.5-Math-1.5B

DeepSeek-R1-Distill-Qwen1.5B

[Figure 50]

Qwen2.5-1.5B

Qwen-2.5-1.5B-SimpleRL-Zoo

[Figure 51]

DeepSeek-R1-Distill-Qwen-7B

Qwen3-4B

Polaris-4B-Preview

Polaris-7B-Preview

[Figure 52]

Qwen2.5-0.5B

Qwen-2.5-0.5B-SimpleRL-Zoo

[Figure 53]

DeepSeek-R1-Distill-Qwen-1.5B

Nemotron-ResearchReasoning-Qwen-1.5B

[Figure 54]

[Figure 55]

Qwen2.5-7B

zero__ppo__think__Qwen2.57B

[Figure 56]

Qwen3-4B

Qwen3-4B-PSR

#### Dataset: MMLU-Pro

[Figure 57]

Qwen2.5-Math-1.5B

DeepSeek-R1-Distill-Qwen1.5B

[Figure 58]

Qwen2.5-1.5B

Qwen-2.5-1.5B-SimpleRL-Zoo

[Figure 59]

DeepSeek-R1-Distill-Qwen-7B

Qwen3-4B

Polaris-4B-Preview

Polaris-7B-Preview

[Figure 60]

Qwen2.5-0.5B

Qwen-2.5-0.5B-SimpleRL-Zoo

[Figure 61]

DeepSeek-R1-Distill-Qwen-1.5B

Nemotron-ResearchReasoning-Qwen-1.5B

[Figure 62]

[Figure 63]

Qwen2.5-7B

zero__ppo__think__Qwen2.57B

[Figure 64]

Qwen3-4B

Qwen3-4B-PSR

- Figure 8: Additional Results on Linear CKA separated by dataset. The vertical axis and horizontal axis are Base Model Layer Index and Reasoning Model Layer Index, respectively. The red background indicates SFT-tuned pairs.

#### Base Embedding Models MEmbbase vs. Reasoning Embedding Models MEmbreason Dataset: CoT Datset

[Figure 65]

[Figure 66]

[Figure 67]

EmbQwen2.5-Math-1.5B-

EmbQwen3-0.6B-Base-

EmbQwen2.5-1.5B-

Qwen-2.5-1.5B-SimpleRLZoo-Emb

DeepSeek-R1-Distill-Qwen1.5B-Emb

Qwen3-0.6B-Emb

[Figure 68]

[Figure 69]

[Figure 70]

DeepSeek-R1-Distill-Qwen-1.5B-

EmbQwen2.5-0.5B-

EmbQwen3-4B-

Emb

Nemotron-ResearchReasoning-Qwen-1.5B-Emb

Qwen-2.5-0.5B-SimpleRLZoo-Emb

Qwen3-4B-PSR-Emb

[Figure 71]

#### Dataset: MMLU-Pro

[Figure 72]

[Figure 73]

[Figure 74]

EmbQwen2.5-Math-1.5B-

EmbQwen3-0.6B-Base-

EmbQwen2.5-1.5B-

Qwen-2.5-1.5B-SimpleRLZoo-Emb

DeepSeek-R1-Distill-Qwen1.5B-Emb

Qwen3-0.6B-Emb

[Figure 75]

[Figure 76]

[Figure 77]

DeepSeek-R1-Distill-Qwen-1.5B-

EmbQwen2.5-0.5B-

EmbQwen3-4B-

Emb

Nemotron-ResearchReasoning-Qwen-1.5B-Emb

Qwen-2.5-0.5B-SimpleRLZoo-Emb

Qwen3-4B-PSR-Emb

[Figure 78]

- Figure 9: Additional Results on Linear CKA separated by dataset. The vertical axis and horizontal axis are Base Model Layer Index and Reasoning Model Layer Index, respectively. The red background indicates their backbone LLMs are SFT-tuned pairs.

#### Base Models Mbase vs. Reasoning Models Mreason Dataset: CoT Datset

Qwen2.5-Math-1.5B

[Figure 79]

DeepSeek-R1-Distill-Qwen-1.5B

[Figure 80]

Qwen3-4B

Polaris-4B-Preview

DeepSeek-R1-Distill-Qwen-7B

[Figure 81]

Polaris-7B-Preview

[Figure 82]

Qwen2.5-7B

zero__ppo__think__Qwen2.5-7B

[Figure 83]

Qwen2.5-1.5B

Qwen-2.5-1.5B-SimpleRL-Zoo

[Figure 84]

Qwen2.5-0.5B

Qwen-2.5-0.5B-SimpleRL-Zoo

DeepSeek-R1-Distill-Qwen-1.5B

[Figure 85]

Nemotron-Research-ReasoningQwen-1.5B

Model Layer Index

[Figure 86]

Qwen3-4B

Qwen3-4B-PSR

#### Dataset: MMLU-Pro

Qwen2.5-Math-1.5B

[Figure 87]

DeepSeek-R1-Distill-Qwen-1.5B

[Figure 88]

Qwen3-4B

Polaris-4B-Preview

DeepSeek-R1-Distill-Qwen-7B

[Figure 89]

Polaris-7B-Preview

[Figure 90]

Qwen2.5-7B

zero__ppo__think__Qwen2.5-7B

[Figure 91]

Qwen2.5-1.5B

Qwen-2.5-1.5B-SimpleRL-Zoo

[Figure 92]

Qwen2.5-0.5B

Qwen-2.5-0.5B-SimpleRL-Zoo

DeepSeek-R1-Distill-Qwen-1.5B

[Figure 93]

Nemotron-Research-ReasoningQwen-1.5B

##### Model Layer Index

[Figure 94]

Qwen3-4B

Qwen3-4B-PSR

- Figure 10: Additional Results on k-NN Overlap separated by dataset. The vertical axis and horizontal axis are Mean Overlap and Model Layer Index, respectively. The red background indicates SFT-tuned pairs.

#### Base Embedding Models MEmbbase vs. Reasoning Embedding Models MEmbreason Dataset: CoT Datset

EmbQwen2.5-Math-1.5B-

[Figure 95]

DeepSeek-R1-Distill-Qwen-1.5B-Emb

EmbQwen3-0.6B-Base-

[Figure 96]

Qwen3-0.6B-Emb

[Figure 97]

EmbQwen2.5-1.5B-

Qwen-2.5-1.5B-SimpleRL-Zoo-Emb

[Figure 98]

EmbQwen2.5-0.5B-

Qwen-2.5-0.5B-SimpleRL-Zoo-Emb

DeepSeek-R1-Distill-Qwen-1.5B-

[Figure 99]

Emb

Nemotron-Research-Reasoning-Qwen-1.5BEmb

Model Layer Index

[Figure 100]

EmbQwen3-4B-

Qwen3-4B-PSR-Emb

#### Dataset: MMLU-Pro

EmbQwen2.5-Math-1.5B-

[Figure 101]

DeepSeek-R1-Distill-Qwen-1.5B-Emb

EmbQwen3-0.6B-Base-

[Figure 102]

Qwen3-0.6B-Emb

[Figure 103]

EmbQwen2.5-1.5B-

Qwen-2.5-1.5B-SimpleRL-Zoo-Emb

[Figure 104]

EmbQwen2.5-0.5B-

Qwen-2.5-0.5B-SimpleRL-Zoo-Emb

DeepSeek-R1-Distill-Qwen-1.5B-

[Figure 105]

Emb

Nemotron-Research-Reasoning-Qwen-1.5BEmb

##### Model Layer Index

[Figure 106]

EmbQwen3-4B-

Qwen3-4B-PSR-Emb

- Figure 11: Additional Results on k-NN Overlap separated by dataset. The vertical axis and horizontal axis are Mean overlap and Model Layer Index, respectively. The red background indicates their backbone LLMs are SFT-tuned pairs.

#### Base Models Mbase vs. Reasoning Models Mreason Dataset: AG’s News Topic Classification

[Figure 107]

Qwen2.5-Math-1.5B

DeepSeek-R1-Distill-Qwen-1.5B

[Figure 108]

Qwen3-4B

Polaris-4B-Preview

DeepSeek-R1-Distill-Qwen-7B

[Figure 109]

Polaris-7B-Preview

[Figure 110]

Qwen2.5-7B

zero__ppo__think__Qwen2.5-7B

[Figure 111]

Qwen2.5-1.5B

Qwen-2.5-1.5B-SimpleRL-Zoo

[Figure 112]

Qwen2.5-0.5B

Qwen-2.5-0.5B-SimpleRL-Zoo

DeepSeek-R1-Distill-Qwen-1.5B

[Figure 113]

Nemotron-Research-ReasoningQwen-1.5B

[Figure 114]

Qwen3-4B

Qwen3-4B-PSR

##### Dataset Types

[Figure 115]

- Figure 12: Additional Results on Cross-Model Linear Probe. The vertical axis and horizontal axis are the Accuracy of the linear probe and Dataset types (train, dev, test), respectively. The red background indicates SFT-tuned pairs.

#### Base Embedding Models MEmbbase vs. Reasoning Embedding Models MEmbreason Dataset: AG’s News Topic Classification

[Figure 116]

EmbQwen2.5-Math-1.5B-

DeepSeek-R1-Distill-Qwen-1.5B-Emb

[Figure 117]

EmbQwen3-0.6B-Base-

Qwen3-0.6B-Emb

[Figure 118]

EmbQwen2.5-1.5B-

Qwen-2.5-1.5B-SimpleRL-Zoo-Emb

[Figure 119]

EmbQwen2.5-0.5B-

Qwen-2.5-0.5B-SimpleRL-Zoo-Emb

[Figure 120]

DeepSeek-R1-Distill-Qwen-1.5B-

Emb

Nemotron-Research-Reasoning-Qwen-1.5BEmb

[Figure 121]

EmbQwen3-4B-

Qwen3-4B-PSR-Emb

##### Dataset Types

[Figure 122]

Figure 13: Additional Results on Cross-Model Linear Probe. The vertical axis and horizontal axis are the Accuracy of the linear probe and Dataset types (train, dev, test), respectively. The red background indicates their backbone LLMs are SFT-tuned pairs.

