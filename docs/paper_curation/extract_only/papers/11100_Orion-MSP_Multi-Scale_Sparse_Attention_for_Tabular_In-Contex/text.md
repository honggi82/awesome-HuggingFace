# arXiv:2511.02818v3[cs.AI]7Nov2025

[Figure 1]

### Mohamed Bouadi, Pratinav Seth Aditya Tanna, Vinay Kumar Sankarapu Lexsi Labs, India & France Abstract

Tabular data remain the predominant format for real-world applications. Yet, developing effective neural models for tabular data remains challenging due to heterogeneous feature types and complex interactions occurring at multiple scales. Recent advances in tabular in-context learning (ICL), such as TabPFN and TabICL, have achieved state-of-the-art performance comparable to gradient-boosted trees (GBTs) without task-specific fine-tuning. However, current architectures exhibit key limitations: (1) single-scale feature processing that overlooks hierarchical dependencies, (2) dense attention with quadratic scaling in table width, and (3) strictly sequential component processing that prevents iterative representation refinement and cross-component communication. To address these challenges, we introduce Orion-MSP, a tabular ICL architecture featuring three key innovations: (1) multi-scale processing to capture hierarchical feature interactions; (2) block-sparse attention combining windowed, global, and random patterns for scalable efficiency and long-range connectivity; and (3) a Perceiver-style memory enabling safe bidirectional information flow across components. Across diverse benchmarks, Orion-MSP matches or surpasses state-of-the-art performance while scaling effectively to high-dimensional tables, establishing a new standard for efficient tabular in-context learning. The model is publicly available at https://github.com/Lexsi-Labs/Orion-MSP.

Keywords: Tabular Foundation Models, In-Context Learning, AutoML, Benchmarking.

[Figure 2]

## 1 Introduction

Tabular data remain the most prevalent form of data in real-world applications, spanning critical systems across healthcare, finance, and scientific research. Despite the remarkable progress of deep learning in natural language processing [27, 42] and computer vision [11], gradient boosted trees (GBTs) remain the predominant state-of-the-art (SOTA) for tabular prediction tasks. In other data modalities, foundation models—particularly Large Language Models (LLMs) [46, 26]—have significantly advanced the ability to tackle new tasks and few-shot learning. This is largely due to their remarkable in-context learning (ICL) capabilities [45, 4], which enable them to capture patterns directly from prompts without updating their parameters. This success combined with the pervasiveness of tables have spurred interest in tabular foundation models [38].

Although LLMs are primarily designed to process natural language, recent efforts have explored fine-tuning them for tabular data tasks [14, 8]. These approaches typically rely on table serialization, which is the process of converting table rows into text or sentences suitable for tokenization. For instance, [9] fine-tuned a Llama 3-8B model on a large corpus of serialized tables and demonstrated that this strategy can outperform traditional tree-based models in few-shot scenarios. However, such language model–based approaches face inherent challenges. Their limited context windows

restrict the number of serialized examples that can be processed simultaneously (e.g., up to 32 or 64 shots in [9]), and it remains uncertain whether LLMs can reliably interpret and reason over numerical values [37].

Adopting a fundamentally different strategy, the authors of [16] introduced TabPFN, a transformer-based tabular foundation model designed for classification tasks and pretrained exclusively on synthetic tabular data. A key feature of TabPFN is its ability to perform in-context learning directly on tables, removing the need for tokenization and allowing efficient processing of relatively small datasets—up to 1K samples and 100 features. Building on this foundation, TabICL [32] introduced a simplified three-component architecture comprising: (1) column-wise embeddings via Set Transformers to capture distribution-aware feature semantics, (2) row-wise interactions with rotary positional encodings to model inter-feature dependencies, and (3) dataset-level ICL prediction through split attention, ensuring a clear separation between training and test samples. These developments position tabular foundation models as a compelling alternative to traditional approaches, particularly for zero-shot prediction tasks where dataset-specific training is infeasible.

However, current table-native ICL architectures face several fundamental limitations that hinder their practical deployment and scalability. First, existing tabular ICL architectures, including TabICL, process features uniformly at a single scale, missing hierarchical interaction patterns that naturally occur in real-world tabular data. Just as computer vision benefits from multi-scale processing—capturing edges at fine scales and objects at coarse scales—tabular data exhibits structure at multiple granularities: individual features interact locally (e.g., age and income), feature clusters form semantic groups (e.g., demographic attributes), and high-level blocks represent major data divisions (e.g., personal attributes versus behavioral patterns). Processing all features uniformly fails to capture these hierarchical relationships, limiting the model’s ability to learn robust and interpretable representations.

Second, the dense attention mechanisms scale quadratically with feature count (O(m2)), where m denotes the number of features. While TabICL addresses sample scalability through its column-then-row architecture, the quadratic feature complexity becomes computationally prohibitive for high-dimensional tables with more than 100 features common in genomics, finance, and sensor applications. For tables with m = 100 features, dense attention requires 10,000 attention operations per layer, with memory requirements growing quadratically. This fundamental scalability barrier limits the practical deployment of tabular foundation models on wide real-world datasets.

Third, the strictly sequential processing pipeline in TabICL (column embedding → row interaction → ICL prediction) prevents iterative refinement and bidirectional information flow between architectural components. While each component produces rich representations, the unidirectional nature of the pipeline means that downstream insights (e.g., dataset-level patterns discovered during ICL) cannot inform upstream representations (e.g., refining feature embeddings based on dataset context). This limitation constrains the model’s ability to leverage holistic dataset understanding for improved predictions, and prevents the kind of iterative refinement that has proven beneficial in multimodal architectures.

To address these limitations, we introduce Orion-MSP, a novel tabular foundation model that extends TabICL with three synergistic architectural innovations. First, we propose multi-scale hierarchical feature processing that simultaneously captures interactions at multiple granularities (individual features, groups of 4, and groups of 16), enabling the model to learn representations at different levels of abstraction analogous to hierarchical processing in computer vision. Second, we design structured block-sparse attention patterns combining windowed local attention, global tokens for long-range dependencies, and random connectivity for universal approximation, reducing computational complexity from O(H2) to O(H.logH) while maintaining expressiveness. Third, we introduce Perceiver-style cross-component memory that enables bidirectional information flow between architectural stages while provably maintaining in-context learning safety constraints—ensuring test data never influences training representations through formal ICL safety analysis.

The column-wise embedding component of Orion-MSP follows TabICL’s approach [32], using Set Transformers with Induced Set Attention Blocks (ISAB) [25] to create distribution-aware feature embeddings in a permutation-invariant manner. The multi-scale row interaction component processes these embeddings at multiple resolutions, with each scale using sparse attention patterns tailored to its granularity. The resulting multi-scale representations are aggregated into unified row embeddings, which then interact with the Perceiver memory before proceeding to the final ICL prediction stage. This ICL component employs split attention with label injection, ensuring proper train-test separation.

Through extensive experiments across diverse tabular benchmarks, we demonstrate that Orion-MSP achieves competitive accuracy with state-of-the-art tabular ICL methods while enabling scalability to tables with more than 100 features where existing methods fail due to memory constraints. Our work establishes that hierarchical multi-scale processing, structured sparsity, and cross-component memory can simultaneously improve both effectiveness and efficiency in tabular foundation models, opening new application domains previously inaccessible to tabular in-context learning methods.

## 2 Related Work

Tabular In-Context Learning: The application of in-context learning (ICL) to tabular data has recently attracted significant attention. TabPFN [16] pioneered this direction by meta-training a transformer on synthetically generated datasets using structural causal models. Its encoder–decoder design allows test samples to attend to training examples, enabling zero-shot predictions without gradient-based fine-tuning. While TabPFN demonstrated strong performance on small datasets, its alternating column- and row-wise attention mechanisms make scaling to larger tables computationally prohibitive.

TabDPT [28] showed that comparable performance can be achieved on real-world datasets by using similarity-based retrieval to construct contextual examples—an idea first explored in TabR [12]. The authors extended this paradigm by integrating diffusion-based representation learning, improving robustness to missing values and distributional shifts. However, the diffusion process introduces substantial computational overhead and retains dense attention, limiting scalability. Similarly, TabPFN-v2 [17] introduced cell-based in-context learning, extending row-wise encoding to datasets exceeding 10,000 samples, but it still inherits quadratic attention costs in high-dimensional tables.

Building on these foundations, TabICL [32] proposed a table-native transformer architecture with three components: column embedding via Set Transformers, row-wise interaction with rotary positional encodings, and an in-context learning prediction module. This design achieved state-of-the-art results across diverse benchmarks while maintaining architectural simplicity and training efficiency. Nonetheless, dense attention in row interactions and the strictly sequential pipeline limit iterative refinement, cross-component communication, and scalability to tables with more than 100 features.

ContextTab [35] further enhanced tabular in-context learning by incorporating contextualized feature embeddings and attention mechanisms tailored for heterogeneous tabular data. While improving performance in complex datasets, it still processes features at a single scale and relies on dense attention, limiting computational efficiency on high-dimensional tables.

Collectively, existing tabular in-context learning models demonstrate strong performance yet share core limitations: dense quadratic attention, uniform single-scale processing, and lack of cross-component feedback.

Sparse Attention Mechanisms: Sparse attention techniques from natural language processing offer a promising route to improve computational efficiency in tabular in-context learning. BigBird [43] and Longformer [1] demonstrated that block-sparse attention patterns can approximate dense attention with linear complexity while maintaining strong theoretical guarantees. Similarly, Sparse Transformers [5, 20] employ structured sparsity for generative modeling, reducing computation without substantial performance degradation. Despite their success in sequential data, these methods have yet to be systematically adapted for tabular in-context learning, where the primary challenge lies in feature dimension rather than sequence length.

Hierarchical and Multi-Scale Architectures: Hierarchical architectures have proven effective in other domains. Funnel Transformers [6] and Swin Transformers [30] use multi-scale processing and pooling to capture information at different resolutions, while Set Transformers [34, 10] leverage pooling by multihead attention for permutation-invariant set processing. Although TabICL [32] employs Set Transformers for column embeddings, it does not incorporate hierarchical multi-scale processing or iterative pooling across feature groups, limiting its ability to model complex interactions in high-dimensional tables.

Cross-Component Communication: Cross-component memory and iterative refinement have shown success in multimodal learning. Perceiver [19] and Perceiver IO [18] introduce latent bottlenecks to compress and share information across modalities, and vision-language models [39] leverage iterative cross-attention for refinement. However, these approaches do not address the causal constraints of in-context learning, where test examples must never influence the representation of training data, leaving a gap for tabular in-context learning.

## 3 Proposed Approach: Orion-MSP

### 3.1 Problem Formulation

Consider a tabular dataset D = {(xi,yi)}ni=1 with n samples and m features. Let X ∈ Rn×m denote the feature matrix, where each column cj ∈ Rn (j ∈ {1,...,m}) represents the values of the j-th feature across all samples.

In the in-context learning setting, we are given a context set of ntrain labeled examples:

C = {(xi,yi)}n

train

i=1 and a set of ntest query samples:

Q = {xi}n

test

i=1

Our goal is to predict the conditional distribution of the target for each query sample given the context set:

p(y|x,C),∀x ∈ Q

### 3.2 High-level Structure: From Data to ICL

Orion-MSP consists of four core components that collectively enable efficient and generalizable tabular in-context learning: (1) Column Embedding: transforms raw tabular features into dense, semantically meaningful representations; (2) Multi-Scale Sparse Row Interaction: captures dependencies at multiple granularities via an hierarchy of attention scales, combining CLS and GLOBAL tokens for local and long-range connectivity; (3) Cross-Component Perceiver Memory: introduces a latent memory bottleneck that enables safe bidirectional communication between modules, promoting iterative refinement without information leakage; (4) Dataset-wise In-Context Learning Predictor: leverages the enriched representations to perform zero-shot prediction across new tasks without gradient updates. An overview of the complete architecture is shown in Figure 1.

Orion-MSP extends the original TabICL [32] architecture with three complementary innovations designed to address the fundamental challenges of tabular data processing: computational inefficiency, limited feature interaction modeling, and the need for hierarchical pattern recognition. Our approach maintains the core in-context learning paradigm while introducing architectural enhancements that significantly improve both efficiency and performance.

### 3.3 Column-wise Embedding

Tabular data exhibits unique characteristics compared to other modalities: each column represents a distinct feature with its own distribution, scale, and statistical properties (e.g., mean, variance, skewness, kurtosis). To capture these distributional characteristics, we adopt the original TabICL [32] column-wise embedder to map each scalar cell in a column cj ∈ Rn to a d-dimensional representation using a shareable Set Transformer, TFcol, that treats the column as a permutation-invariant set of values. Our goal is to transform each cell value Xij into a d-dimensional embedding Eij ∈ Rd that encodes both:

- 1. The value of the cell (Xij)
- 2. The distributional context of the column (cj)

This differs fundamentally from standard embedding approaches (e.g., word embeddings) where each discrete token has a fixed embedding regardless of context. In tabular data, the meaning of a value depends heavily on the column’s distribution: a value of 50 may be typical in one feature but an outlier in another.

Concretely, TFcol predicts a per-cell affine map, assigning each cell its own weight and bias. The process consists of three main steps:

- 3.3.1 Initial Projection: Project the column values into a d-dimensional embedding space:

Uj = Linearproj(cj) ∈ Rn×d (1)

where Linearproj : R → Rd is a learned linear transformation. This creates initial token embeddings for each cell in the column.

- 3.3.2 Induced Set Attention Blocks (ISAB):

To efficiently capture global distributional information while maintaining computational tractability, we employ ISAB [25] with k learnable inducing points. It consists of two sequential Multi-Head Attention Blocks (MAB1,MAB2):

##### Mj = MAB1(I,Utrainj ,Utrainj ) ∈ Rk×d (2) Vj = MAB2(Uj,Mj,Mj) ∈ Rn×d (3)

[Figure 3]

- Figure 1 An overview of Orion-MSP architecture. First, column-wise embedding transforms input table into embedding vectors E. Next, multi-scale sparse row interaction prepends learnable [CLS] and [GLOBAL] tokens to E, processes features at multiple granularities (scales 1, 4, and 16) with sparse attention transformers, and aggregates [CLS] outputs across scales to yield row

embeddings H. Cross-component Perceiver memory enables bidirectional communication: training rows write to latent memory, which all rows read for enhanced representations R. Finally, ICL predicts test labels from R in a single forward pass.

where I ∈ Rk×d denote k trainable inducing point embeddings (k ≪ n), which serve as a compressed representation of the column distribution.

We define a Multi-Head Attention Block as:

MAB(Q,K,V) = LayerNorm(H + MultiHead(Q,K,V)) (4) where H is a residual connection (set to Q if dimensions match, otherwise passed through a projection), and:

MultiHead(Q,K,V) = Concat(head1,...,headh)WO (5) with each head defined as:

headi = Attention(QWQi ,KWKi ,VWVi ) (6)

Following TabICL [32], we use d=128, k=128, 4 heads, and 3 ISAB blocks. Crucially, in Equation 2, we use only training samples Utrainj ∈ Rn

train×d as keys and values. This ensures that the inducing points Mj capture the distribution of the training data only, preventing information leakage from test samples during embedding. This is crucial for maintaining the in-context learning paradigm.

In Equation 3, all samples (training and test) query the inducing points to obtain their contextualized embeddings. The inducing points act as a distributional summary: they encode statistical properties (e.g., mean, variance, skewness) of the training column values, and each cell embedding is adjusted based on where it lies within this learned distribution.

- 3.3.3 Weight and Bias Generation: The ISAB output Vj is passed through a feedforward network to generate cell-specific weights and biases:

##### Wj,Bj = FFN(Vj),Wj,Bj ∈ Rn×d (7) where:

FFN(Vj) = Linearout(GELU(Linearhidden(Vj))) (8) The final embeddings are then computed as:

##### E:,j,: = Wj ⊙ cj + Bj ∈ Rn×d (9)

where ⊙ denotes element-wise (Hadamard) product, and cj is broadcasted to shape (n,d). This formulation allows each cell’s embedding to be a function of both its raw value (cj) and the column’s learned distributional properties (Wj,Bj). Note that, in our architecture, row-wise interaction requires prepending special tokens (e.g., [CLS], [GLOBAL]) to each row. To accommodate these, the column embedding reserves C positions at the beginning of each column:

E ∈ Rn×(m+C)×d (10) For the reserved positions (indices 1 to C), we use a skippable linear layer that outputs zeros or small random values:

SkipLinear(cj) if j ≤ C (reserved) Wj ⊙ cj + Bj if j > C (features)

(11)

E:,j,: =

where SkipLinear is a linear layer with very small initialization, allowing the model to learn appropriate embeddings for reserved positions during training.

The Set Transformer architecture ensures that TFcol is permutation-invariant with respect to the order of samples within a column. Formally, let π : [T] → [T] be any permutation, and let c′j = Pπcj where Pπ is the corresponding permutation matrix. Then:

TFcol(c′j) = Pπ TFcol(cj) (12)

This property is inherited from the attention mechanism in ISAB, where the softmax normalization and weighted aggregation are invariant to input order.

The inducing points Mj ∈ Rk×d learned by the first MAB serve as a distributional summary of column j. Empirically, we observe that:

- • Columns with similar statistical moments (mean, variance, skewness, kurtosis) have similar inducing point representations (measured by cosine similarity).
- • The inducing points capture multi-modal distributions: for categorical features encoded numerically, different modes correspond to different cluster centers in the inducing point space.
- • Outliers in cj receive distinct embeddings, as their attention weights to Mj differ significantly from typical values.

### 3.4 Multi-Scale Sparse Row-Wise Interaction

While column-wise embedding captures distributional properties of individual features, row-wise interaction must model complex dependencies across features to extract meaningful sample representations. However, directly applying dense self-attention to all feature tokens incurs quadratic complexity O(m2) and may overfit when the number of features varies significantly across datasets. To address these challenges, we introduce a hierarchical multi-scale sparse attention mechanism that processes features at multiple granularities with efficient block-sparse patterns.

- 3.4.1 Motivation and Design Principles Tabular datasets exhibit several unique characteristics that complicate feature interaction modeling:

- 1. Variable feature counts: The number of features m varies dramatically across datasets, making fixed-scale architectures suboptimal.
- 2. Heterogeneous feature relationships: Some features interact locally (e.g., age and age-related health metrics), while others have global dependencies (e.g., categorical indicators).
- 3. Computational constraints: Dense attention over m features has complexity O(m2), becoming prohibitive for wide tables or long context windows.
- 4. Overfitting risks: Full attention can memorize training-specific feature correlations that do not generalize to new datasets.

Inspired by hierarchical representations in vision [41] and multi-resolution modeling in speech [44], Orion-MSP decomposes feature interactions into multiple resolution levels:

- • Fine scale (s = 1): Captures detailed pairwise dependencies between individual features.
- • Coarse scales (s > 1): Aggregates semantically related features into groups, reducing sequence length and enabling broader contextual reasoning.
- • Scale aggregation: Combines representations across scales to balance local precision and global context.

[Figure 4]

- Figure 2 Building blocks of the attention mechanism used in Orion-MSP. White color indicates absence of attention. (a) special attention includes CLS = 4 and global attention with GB = 4, (b) sliding window attention with w = 8, (c) random attention with

r = 2, (d) the combined row representation of Orion-MSP model.

To further improve efficiency and generalization, we adopt a block-sparse attention pattern inspired by Longformer [1] and BigBird [43], as depicted in Figure 2:

- • Sliding window attention: Local connectivity within a fixed radius w, preserving fine-grained structure.
- • Global tokens: Specialized tokens with full connectivity, ensuring stable long-range information flow.
- • Random links: Optional sparse stochastic connections that enhance expressivity and global reachability..

This design reduces attention complexity from O(m2) to O(m · (w + g + r)) where w, g and r are the window size, the number of global tokens, and the number of random links respectively.

Formally, the multi-scale sparse row-wise transformer, TFMSrow, processes column-embedded features E ∈ RB×n×(m+C)×d to generate row-wise embeddings H ∈ RB×n×(Ncls·d):

H = TFMSrow(E,dvalid) ∈ RB×n×(N

cls·d) (13)

where B is the number of datasets, n the number of samples per dataset, m the number of features, and d the embedding dimension. The constant C = Ncls + Nglobal accounts for special token slots, anddvalid ∈ RB optionally indicates the number of valid features per dataset for handling variable-length inputs.

The transformation proceeds through the following steps:

#### 3.4.2 Multi-Scale Feature Grouping:

First, for each scale s ∈ S = {s1,s2,...,sM} (e.g., S = {1,4,16}), we group the m feature tokens into Ks = ⌈m/s⌉ groups of size s.

The default grouping strategy uses a learnable soft grouping via Pooling by Multihead Attention (PMA) [24] to adaptively attend to features:

s×d

Qs = Seeds + PE(Ks) ∈ RK

Ks = LinearK(E:,:,C:,:) ∈ RB·n×m×d Vs = LinearV (E:,:,C:,:) ∈ RB·n×m×d

QsK⊤s

s×n

d ∈ RK

As = softmax

√

Fs = AsVs ∈ RB×n×K

s×d

where Seeds ∈ RK

s×d is a learnable seed embedding, and PE(Ks) adds sinusoidal positional encodings. PMA allows the model to learn which features to group together, adapting to dataset-specific correlation structures.

###### 3.4.3 Special Tokens Injection: For each row at each scale, we prepend special tokens:

- 1. CLS ∈ RN

cls×d (learnable, per-row summary)

- 2. GLOBAL ∈ RN

global×d (learnable, long-range connectivity) The full sequence at scale s becomes:

special+Ks)×d (14) where Nspecial = Ncls + Nglobal.

Xs = [CLS,GLOBAL,G(s)] ∈ RB×n×(N

#### 3.4.4 Block-Sparse Attention Mask:

As depicted in Figure 2, for each scale, we construct a sparse attention mask Ms ∈ RL

s×Ls where Ls = Nspecial + Ks. The mask follows these rules:

- 1. Fully Connected Special Tokens: The first Nspecial tokens (CLS and GLOBAL) are fully connected to all other tokens and to each other (Figure 2.a):

Ms[i,j] = 0 ∀i ∈ [1,Nspecial] or j ∈ [1,Nspecial] (15)

- 2. Sliding Window Attention: Feature tokens (indices > Nspecial) attend to neighbors within a window of radius w = 8 (Figure 2.b):

Ms[i,j] =

0 if |i − j| ≤ w and i,j > Nspecial −∞ otherwise

(16)

- 3. Random Links (Optional): For each feature token i > Nspecial, we randomly select r additional tokens to attend to (Figure 2.c):

Ms[i,jk] = 0 for k ∈ [1,r], jk ∼ Uniform({Nspecial + 1,...,Ls} \ {i}) (17) The final mask is (Figure 2.d):

Ms[i,i] = 0 ∀i ∈ [1,Ls] (self-attention always allowed) (18)

- 3.4.5 Transformer Encoder per Scale: For each scale s ∈ S, we apply a dedicated Transformer encoder:

Zs = Encoders(Xs,Ms) ∈ RB×n×L

s×d (19) where Encoders consists of Nblocksrow /|S| stacked Transformer blocks with:

- • Multi-head self-attention: MHA(Q,K,V,Ms) with sparse mask Ms
- • Rotary positional encoding (RoPE): Applied to queries and keys before attention [36]
- • Feed-forward network: Two-layer MLP with GELU activation
- • Pre-norm architecture: Layer normalization before each sub-layer

The multi-head attention with sparse masking is computed as:

headi = Attention(Qi,Ki,Vi,Ms) (20)

= softmax

QiK⊤i √dk

+ Ms Vi (21)

where Ms contains 0 for allowed positions and −∞ for disallowed positions (additive masking). After processing through Encoders, we extract the CLS token representations:

Hs = Zs[:,:,1 : Ncls,:] ∈ RB×n×N

cls×d (22) These represent the row-wise features at scale s. We then aggregate these representations across all scales by averaging:

Hagg =

1 |S| s∈S

Hs ∈ RB×n×N

cls×d (23)

This simple averaging strategy ensures that each scale contributes equally, balancing fine-grained and coarse-grained information. Next, the CLS tokens are flattened and normalized to produce the final row embeddings:

H = LayerNorm(Flatten(Hagg)) ∈ RB×n×(N

cls·d) (24)

where Flatten concatenates the Ncls token embeddings. Algorithm 1 summarizes the complete multi-scale sparse row-wise interaction process.

- 3.4.6 Computational Complexity

For a given scale s with Ks = ⌈m/s⌉ grouped feature tokens, the per-layer computational complexity of the sparse attention mechanism is:

##### O(B · n · Ls · (w + Nglobal + r) · d) (25)

- Algorithm 1 Multi-Scale Sparse Row-Wise Interaction (TFMSrow)

Require: Embeddings E ∈ RB×n×(m+C)×d, valid features dvalid ∈ RB Require: Scales S = {s1,s2,...,sM}, window w, random links r Ensure: Row embeddings H ∈ Rn×m×(N

cls·d)

- 1: Initialize learnable tokens CLS ∈ RN

cls×d, GLOBAL ∈ RN

global×d

- 2: Hall ← [] {Store CLS outputs from all scales}
- 3: for each scale s ∈ S do
- 4: Ks ← ⌈m/s⌉ {Number of groups at scale s}
- 5: // Feature Grouping
- 6: G(s) ← PMA(E[:,:,C :,:],Ks)
- 7: // Construct Sequence
- 8: Xs ← [CLS,GLOBAL,G(s)] {Shape: (B,n,Ls,d) where Ls = Nspecial + Ks}
- 9: // Build Sparse Mask
- 10: Ms ← BuildBlockSparseMask(Ls,Nspecial,w,r)
- 11: // Process Through Transformer
- 12: Zs ← Encoders(Xs,Ms) {Transformer with RoPE and sparse attention}
- 13: // Extract CLS Tokens
- 14: Hs ← Zs[:,:,1 : Ncls,:]
- 15: Hall.append(Hs)
- 16: end for
- 17: // Aggregate Across Scales
- 18: Hagg ← |S|1 |S|s=1 Hall[s]

- 19: // Flatten and Normalize
- 20: H ← LayerNorm(Flatten(Hagg))
- 21: return H

where B is the batch size, n the number of samples per dataset, m the number of features, d the embedding dimension, and w, Nglobal, and r denote the sliding-window size, number of global tokens, and number of random links, respectively.

For M scales and a total of Nblocksrow Transformer layers distributed evenly across scales, the overall complexity becomes:

Ototal =

O(B · n · Ks · (w + Nglobal + r) · d ·

s∈S

Nblocksrow M

) (26)

Since s∈S Ks ≈ m · 1 + s1

2

+ ... + s1

M

and typically w,Nglobal,r ≪ m, this simplifies to:

Ototal ≈ O(B · n · m · (w + Nglobal + r) · d · Nblocksrow ) (27) compared to the dense attention cost of O(B · n · m2 · d · Nblocksrow ).

For typical hyperparameters (m ∈ [10,100], w = 8, Nglobal = 4, r = 2), this results in a reduction from quadratic O(m2) to near-linear O(m · 14) complexity—achieving linear scaling while preserving both local and global feature dependencies.

### 3.5 Cross-Component Memory with Perceiver Architecture

While the column-wise embedding and row-wise interaction components of tabular transformers independently model feature- and sample-level dependencies, richer contextual understanding can emerge if information is shared across these components. However, direct cross-component communication poses a major risk to the in-context learning (ICL) paradigm: naive attention between components can leak test-set information, violating the principle that predictions for test samples must depend solely on training examples and the test input itself.

To overcome this limitation, we introduce a Perceiver-style latent memory module [19] that enables safe, leak-free communication between architectural components. This latent memory acts as a shared representation space that can be written to by training samples and read from by both training and test samples, ensuring compliance with ICL constraints while promoting global knowledge sharing.

In standard transformer-based tabular architectures such as TabICL [32], model components operate in a strictly sequential and isolated fashion:

- 1. Column Embedding (TFcol): Encodes feature-wise statistics across samples to capture column-level distributions.
- 2. Row Interaction (TFrow): Models dependencies across features within each sample.
- 3. ICL Prediction (TFicl): Performs in-context learning to infer test labels from training examples.

This separation simplifies optimization and ensures ICL safety, but also introduces significant limitations:

- • No backward adaptation: Column embeddings cannot adjust based on row-level feature interactions.
- • Limited contextual refinement: Row-level interactions lack access to global, dataset-level statistics beyond static column embeddings.
- • Dataset isolation: Each dataset is processed independently, preventing cross-dataset generalization within a batch.

A fundamental ICL constraint is that test samples must not influence the model’s internal state in a way that affects training representations. Formally, letting

##### Dtrain = {(xi,yi)}n

i=1 ,Dtest = {xj}nj=ntrain+1 (28) the prediction for a test sample xj must satisfy:

train

##### P(ˆyj | Dtrain,Dtest) = P(ˆyj | Dtrain,xj) (29)

That is, the prediction depends only on the training set and the test features, never on other test representations or their labels.

Inspired by the Perceiver architecture [19], we introduce a learnable latent memory L ∈ RP×d with P memory slots. The key idea is:

- 1. Write Phase (train-only): Memory attends to training representations to extract relevant global patterns.
- 2. Read Phase (all samples): Both training and test samples attend to the memory to retrieve learned context, but cannot modify it.

This asymmetry guarantees ICL safety, since only training data influence the memory’s contents. The memory serves as a compressed, permutation-invariant summary of the training context that enables consistent feature refinement across samples.

The memory module is incorporated inside the ICL transformer (TFicl), refining the row embeddings before label injection and prediction. Given row embeddings H ∈ RB×n×d

h (where B is the batch size and n the number of samples per dataset), the Perceiver memory transformation produces refined representations:

R = PerceiverMemory(H,ntrain) ∈ RB×n×d

H (30) with:

- • dH = Ncls · d the hidden dimension after multi-head projection,
- • P the number of latent memory slots (a hyperparameter),
- • and ntrain the number of labeled training examples.

The Perceiver memory consists of three key stages, each composed of multiple cross-attention layers with residual connections and feed-forward transformations.

- 1. Latent Memory Initialization: We initialize a set of P learnable latent vectors:

L0 ∈ HP×d

H (31)

drawn from a truncated normal distribution N(0,0.022). These latents act as a universal memory bank, shared across all datasets in the batch and reused across forward passes, providing a stable foundation for information aggregation.

- 2. Cross-Attention Block: At the core of the memory is a cross-attention mechanism allowing one representation to attend to another. Given query set Q and key–value set KV, we define:

CrossAttn(Q,KV) = softmax

QWQ(KVWK)⊤ √dk

(KVWV ) (32)

where WQ,WK,WV ∈ Rd

R×dk are projection matrices and dk = dR/h is the per-head dimension.

Each cross-attention block is followed by layer normalization, residual connections, and a feed-forward layer:

Q′ = LayerNorm(Q) (33) KV′ = LayerNorm(KV) (34)

Z = Q + MultiHeadCrossAttn(Q′,KV′) (35) Z′ = Z + FFN(LayerNorm(Z)). (36)

This block structure ensures stable training and supports multi-head feature integration.

- 3. Write Phase: Memory Encoding: In the write phase, the memory attends to training samples only to extract and store relevant patterns. For each dataset b in the batch:

H(trainb) = H(b)[: Ttrain,:] ∈ Rn

train×dH (37)

and initialize the dataset-specific memory as L(0b) = L0. We then apply Nwrite cross-attention blocks where memory latents query the training representations:

L(i+1b) = CrossAttnBlock(Q = L(ib),KV = H(trainb) ) for i = 0,...,Nwrite − 1 (38) The final encoded memory is:

L(b) = L(Nb)

write

∈ RP×d

H (39) Importantly, L(b) depends only on training representations, ensuring no test leakage.

- 4. Read Phase: Sample Refinement: In the read phase, all samples (training and test) attend to the memory to retrieve stored context. For dataset b:

H(0b) = H(b) ∈ Rn×d

H (40) We apply Nread cross-attention blocks where sample queries attend to the memory:

R(i+1b) = CrossAttnBlock(Q = R(ib),KV = L(b)) for i = 0,...,Nread − 1 (41) The final refined embeddings are:

R(b) = R(Nb)

∈ Rn×d

R (42) This asymmetric read–write design preserves the integrity of in-context learning:

read

- • Only training samples write to the memory.
- • Both training and test samples read from it.
- • The memory functions as a shared, compressed abstraction of the training data that can be safely leveraged for inference.

The complete ICL forward pass with Perceiver memory is described in Algorithm 2:

### 3.6 Dataset-wise In-Context Learning

After column-wise embedding, multi-scale sparse row-wise interaction, and optional cross-component memory refinement, each sample is represented by a fixed-dimensional row embedding:

R ∈ RB×n×d

R (43)

- Algorithm 2 ICL with Perceiver Memory

Require: Row embeddings H ∈ RB×n×d

H, training labels ytrain ∈ RB×n

train

Ensure: Predictions yˆ ∈ RB×(n−n

train)×C for C classes

- 1: // Perceiver Memory (optional)
- 2: if P > 0 then
- 3: for each dataset b = 1 to B do
- 4: H(trainb) ← H(b)[: Ttrain,:] {Extract training samples}
- 5: L(b) ← L0 {Initialize memory}
- 6: // Write: Memory attends to training samples
- 7: for i = 1 to Nwrite do
- 8: L(b) ← CrossAttnBlock(L(b),H(trainb) )
- 9: end for
- 10: // Read: All samples attend to memory
- 11: R(b) ← H(b)
- 12: for i = 1 to Nread do
- 13: R(b) ← CrossAttnBlock(R(b),L(b))
- 14: end for
- 15: end for
- 16: R ← R {Use refined embeddings}
- 17: end if
- 18: // Label Injection (training samples only)
- 19: R[:,: ntrain,:] ← H[:,: ntrain,:] + OneHot(ytrain)Wlabel
- 20: // ICL Transformer with Split Mask
- 21: H ← TFicl(H,attn_mask = ntrain) {Prevent test-to-train leakage}
- 22: // Prediction Head
- 23: R ← LayerNorm(R)
- 24: logits ← FFNdecoder(R[:,ntrain :,:]) {Predict test labels only}
- 25: return logits

where B is the number of datasets in the batch, n the total number of samples per dataset, and dR the embedding dimension.

The final component, dataset-wise in-context learning (TFicl), leverages these embeddings to predict test labels by conditioning on labeled training examples—all within a single forward pass and without any gradient-based parameter updates..

Formally, for each dataset b in the batch:

Dtrain(b) = {(R(ib),yi(b))}n

i=1 (44) Dtest(b) = {R(jb)}nj=ntrain+1 (45)

train

The objective is to predict test labels yˆj(b) for j > ntrain using in-context reasoning from training examples only.

yˆtest = TFicl(R,ytrain) (46) The ICL module consists of three main stages:

- 1. Label Encoding and Injection: To ensure consistency across datasets with potentially different label

spaces, training labels ytrain ∈ RB×n

train are first normalized to contiguous indices:

y˜i = argsort(unique(ytrain))[ytrain[i]] (47) mapping any label set {2,5,9} → {0,1,2}. Normalized labels are embedded using one-hot encoding followed by a linear projection:

ey = OneHot(˜y,Cmax) · Wy ∈ Rd

R (48)

where Cmax is the maximum number of classes (e.g., Cmax = 10), and Wy ∈ RC

max×dR is a learned projection

matrix. Label embeddings are injected only into training samples via additive combination:

R[:,: ntrain,:] ← R[:,: ntrain,:] + ey(ytrain) (49) ensuring test samples remain unaffected and ICL constraints are preserved.

- 2. Split-Masked Transformer: The augmented embeddings R are processed by a split-masked Transformer, enforcing ICL-safe attention between training and test samples. The attention mask Msplit is defined as:

Msplit[i,j] =

 



0 if i ≤ ntrain and j ≤ ntrain (train-to-train) 0 if i > ntrain and j ≤ ntrain (test-to-train) −∞ if i ≤ ntrain and j > ntrain (train-to-test: blocked) 0 if i > ntrain and j > ntrain (test-to-test)

(50)

No leakage from test to train samples.

- • Training samples attend only to other training samples (learn from labeled context).
- • Test samples attend to training samples and other test samples (contextual reasoning).
- • No leakage from test to train samples.

The Transformer applies Nicl blocks of multi-head self-attention and feed-forward layers:

H(0) = R (51) H(ℓ+1) = TransformerBlock(H(ℓ),Msplit) for ℓ = 0,...,Nicl − 1 (52)

with the final output normalized via:

H = LayerNorm(H(N

icl)) (53)

- 3. Prediction head: Test sample representations H[:,ntrain :,:] are passed through a two-layer MLP decoder:

test×2dR (54) logits = zW2 + b2 ∈ RB×n

z = GELU(H[:,ntrain :,:]W1 + b1) ∈ RB×n

test×Cmax (55) Predictions are obtained via softmax with temperature τ:

yˆtest = softmax(logits[:,:,: K]/τ) (56) where K is the number of classes in the current dataset (inferred from training labels), and τ = 0.9 by default. When the number of classes K > Cmax (e.g., K > 10), we employ a hierarchical classification strategy: Grouping: Partition

- (a) Grouping: Partition K classes into G = ⌈K/Cmax⌉ balanced groups.
- (b) First-level prediction: Predict which group a test sample belongs to.
- (c) Second-level prediction: For each group, train a classifier on the subset of classes within that group.
- (d) Combination: Multiply group probability with intra-group probability to obtain final prediction.

This hierarchical mechanism preserves the ICL paradigm while scaling to hundreds of classes. During pretraining, the model is trained with cross-entropy loss on test samples:

B

n

1 B · ntest

log p(yj(b) | R(b),ytrain(b) ) (57)

L = −

j=ntrain+1

b=1

Critically, gradients flow through the entire architecture (column embedding, row interaction, memory, ICL transformer, decoder) in an end-to-end manner, enabling the model to learn representations optimized for in-context learning.

- Algorithm 3 Dataset-wise In-Context Learning

Require: Row embeddings R ∈ RB×n×d

R, training labels ytrain ∈ RB×n

train

Ensure: Predictions yˆtest ∈ RB×(n−n

train)×K

- 1: // Optional: Perceiver Memory
- 2: if memory enabled then
- 3: R ← PerceiverMemory(R,ntrain)
- 4: end if
- 5: // Label Encoding and Injection
- 6: y˜train ← NormalizeLabels(ytrain) {Map to {0,1,...,K − 1}}
- 7: ey ← OneHotLinear(y˜train) {Shape: (B,ntrain,dR)}
- 8: R[:,: ntrain,:] ← R[:,: ntrain,:] + ey
- 9: // Split-Masked Transformer
- 10: Msplit ← BuildSplitMask(n,ntrain)
- 11: H ← TFicl(R,Msplit)
- 12: H ← LayerNorm(H)
- 13: // Prediction Head
- 14: Htest ← H[:,ntrain :,:] {Extract test representations}
- 15: logits ← MLPdecoder(Htest) {Shape: (B,Ttest,Cmax)}
- 16: logits ← logits[:,:,: K] {Select active classes}
- 17: yˆtest ← softmax(logits/τ)
- 18: return yˆtest

- 4 Experimental Evaluation

We conduct a comprehensive evaluation of Orion-MSP. Below, we describe our experimental setup and present detailed results.

### 4.1 Experimental Setting

#### 4.1.1 Benchmark Suites and Datasets.

Our experimental evaluation spans three widely recognized benchmark suites: TALENT [40] (181 automatically discovered classification datasets), OPENML-CC18 [2] (72 curated datasets), and TABZILLA [29] (36 heterogeneous tasks). Together, these benchmarks enable a comprehensive assessment across diverse tabular learning scenarios. In addition, we perform domain-specific evaluations in high-impact application areas such as healthcare and finance to examine the real-world relevance of our method. All experiments strictly follow the official dataset splits provided by each benchmark to ensure reproducibility and fairness.

For consistency across model families, results are reported only on the intersection of datasets available to all evaluated models within each benchmark suite. This unified evaluation protocol ensures that observed performance differences arise from methodological advances rather than variations in dataset coverage. After filtering, our evaluation encompasses 154 of 181 datasets from TALENT, 63 of 72 from OpenML-CC18, and 27 of 36 from TabZilla. A small number of datasets were excluded due to out-of-memory (OOM) errors or CUDA-related issues, primarily affecting TabPFN-based architectures even on H200 GPUs.

Finally, we emphasize that models with higher mean ranks may not always achieve the highest absolute accuracy or F1-scores on every dataset. Rankings based on accuracy are computed per dataset and then averaged across all datasets, providing a normalized indicator of overall consistency rather than peak task-specific performance. In contrast, absolute metrics highlight maximum achievable performance on individual tasks. Comprehensive dataset statistics are presented in Appendix C.

#### 4.1.2 Models and Baselines.

We compare our model with six state-of-the-art tabular foundation models: TABPFN [16], TABICL [32], ORIONBIX, MITRA, CONTEXTTAB [35], and TABDPT [28]. In addition, we include established traditional baselines using autogloun [7] such as XGBOOST, LIGHTGBM, CATBOOST, and RANDOM FOREST as strong reference models for comparison.

#### 4.1.3 Hardware Configuration.

Experiments are executed on NVIDIA L40S GPUs, with H200 GPUs used for memory-intensive cases. This infrastructure ensures consistent execution across all experiments while handling the computational demands of large transformer-based models.

###### 4.1.4 Evaluation Metrics. Our evaluation considers two complementary aspects:

Performance. We measure predictive capability using standard classification metrics—Accuracy (ACC), AUC-ROC, and weighted F1-score (F1)—computed across the benchmark suites TALENT, OpenML-CC18, and TabZilla. These benchmarks encompass datasets with diverse characteristics, including varying sample sizes, feature dimensionalities, and class balance, allowing a comprehensive assessment of model generalization. It is important to clarify how MEAN RANK values are derived. Within each benchmark suite, models are ranked by accuracy on every dataset (lower rank = better performance), and these per-dataset ranks are averaged to obtain the overall mean rank. Thus, a lower mean rank indicates stronger and more consistent performance across datasets, rather than the highest score on any single task. While absolute metrics (ACCURACY, F1) reflect peak task-level performance, mean rank provides a normalized measure of cross-dataset generalization consistency.

Scalability. We further analyze model robustness as dataset complexity increases by examining performance trends with respect to sample size, feature dimensionality, and class imbalance. This analysis uses the same benchmark datasets, aggregated along these axes to reveal systematic scalability behaviors and guide practical model selection.

### 4.2 Results

Table 1 Performance comparison across three benchmark suites—TALENT, OpenML-CC18, and TabZilla. Ranks denote the mean rank based on accuracy per benchmark suite (lower is better). Metrics: ACC = Accuracy, F1 = Weighted F1. The “All” column reports the aggregated rank across all suites. Formatting: 1st place; 2nd place.

All TALENT OpenML-CC18 TabZilla Rank Rank ACC F1 Rank ACC F1 Rank ACC F1

Models

XGBoost 6.70 6.02 0.8403 0.8360 5.89 0.8558 0.8537 6.07 0.8612 0.8326 CatBoost 6.43 5.57 0.8336 0.8259 6.25 0.8588 0.8520 7.13 0.8579 0.8384 Random Forest 7.38 6.15 0.8285 0.8209 6.36 0.8547 0.8497 8.42 0.8358 0.8399 LightGBM 6.78 6.11 0.8331 0.8245 6.18 0.8581 0.8493 5.25 0.8618 0.8211 TabICL 4.96 4.09 0.8471 0.8379 4.69 0.8667 0.8623 5.89 0.8734 0.8698 OrionBiX 5.37 4.59 0.8346 0.8260 4.98 0.8653 0.8596 4.89 0.8728 0.8628 OrionMSP 3.58 3.26 0.8461 0.8360 4.12 0.8722 0.8676 3.84 0.8821 0.8786 TabPFN 4.61 3.72 0.8514 0.8412 4.76 0.8714 0.8663 4.86 0.8752 0.8716 Mitra 11.77 10.38 0.3921 0.2868 10.52 0.3614 0.2522 11.21 0.3152 0.1830 ContextTab 9.70 9.84 0.5474 0.4596 6.28 0.8639 0.8581 7.13 0.8389 0.8334 TabDPT 5.42 5.19 0.8408 0.8318 4.64 0.8672 0.8625 3.94 0.8814 0.8775

Table 1 summarizes results across the TALENT, OpenML-CC18, and TabZilla benchmark suites, reporting mean rank, classification accuracy (ACC), and weighted F1-score (F1) for all evaluated models.

Our experiments confirm that classical machine learning methods remain strong baselines, achieving mean accuracies between 0.833 and 0.861 with aggregated ranks around 6.0. In contrast, pretrained tabular foundation models (TFMs) demonstrate superior generalization, even without task-specific fine-tuning. Notably, our model, ORION-MSP, achieves the best overall zero-shot rank of 3.58, with ACC/F1 scores of 0.8461/0.8360 on TALENT, 0.8722/0.8676 on OpenMLCC18, and 0.8821/0.8786 on TabZilla.

TABPFN follows closely, attaining an overall rank of 4.61 and scores of 0.8514/0.8412 on TALENT and up to 0.8752/0.8716 on TabZilla. TABDPT ranks 5.42, achieving 0.8408/0.8318 on TALENT and 0.8814/0.8775 on TabZilla. By contrast, Mitra (rank 11.77, ACC < 0.40) and ContextTab (rank 9.70) perform substantially worse, highlighting the advantages of hierarchical multi-scale processing and efficient attention in Orion-MSP.

Overall, TABPFN and ORION-MSP emerge as the strongest models, with ACC ranging from 0.85 to 0.88 and ranks between 3.26 and 4.61. ORION-MSP peaks on OpenML-CC18 (rank 4.12, ACC 0.8722) and TabZilla (rank 3.84, ACC

Table 2 Performance variation by dataset size across all benchmark suites. Rank denotes the accuracy-based ranking per size category (lower is better). ACC = Accuracy; F1 = Weighted F1-score, averaged across datasets within each size category: Small (<1K samples), Medium (1K–10K), and Large (>10K).

Small (<1K) Medium (1K-10K) Large (>10K) Rank ACC F1 Rank ACC F1 Rank ACC F1

Models

XGBoost 7.70 0.8168 0.7964 6.88 0.8363 0.8314 5.41 0.8969 0.8920 CatBoost 7.88 0.8124 0.7935 6.47 0.8340 0.8264 5.48 0.8797 0.8733 Random Forest 8.55 0.7988 0.8187 7.16 0.8285 0.8221 7.30 0.8694 0.8628 LightGBM 7.80 0.8143 0.7789 6.94 0.8314 0.8226 5.63 0.8827 0.8764 TabICL 6.04 0.8301 0.8338 4.77 0.8486 0.8398 4.61 0.8802 0.8743 OrionBiX 6.32 0.8330 0.8150 5.48 0.8348 0.8260 4.42 0.8729 0.8670 OrionMSP 5.93 0.8232 0.8194 3.70 0.8494 0.8402 3.04 0.8843 0.8768 TabPFN 6.50 0.8325 0.8131 3.81 0.8557 0.8462 5.73 0.8783 0.8713 Mitra 13.88 0.4334 0.3236 11.59 0.3600 0.2553 11.11 0.3837 0.2754 ContextTab 9.60 0.7578 0.7363 9.52 0.6210 0.5566 10.22 0.6388 0.5638 TabDPT 5.48 0.8333 0.8271 5.40 0.8424 0.8339 5.26 0.8831 0.8765

Table 3 Performance variation by feature dimensionality (dataset width) across all benchmark suites. Rank denotes the accuracy-based ranking averaged within each width category (lower is better). ACC = Accuracy; F1 = Weighted F1-score. Values are on a 0–1 scale (higher is better). Formatting: 1st place ; 2nd place within each group.

Narrow (<10) Medium (10-100) Wide (>100) Rank ACC F1 Rank ACC F1 Rank ACC F1

Models

XGBoost 6.77 0.8222 0.8159 6.90 0.8482 0.8410 4.79 0.9140 0.9039 CatBoost 5.63 0.8145 0.8067 6.88 0.8441 0.8344 5.50 0.9157 0.9084 Random Forest 7.15 0.8005 0.7044 7.44 0.8410 0.8235 7.52 0.9034 0.8936 LightGBM 6.15 0.8128 0.7907 6.92 0.8458 0.8326 7.47 0.8999 0.8908 TabICL 5.14 0.8208 0.8119 4.61 0.8627 0.8549 6.46 0.9101 0.8936 OrionBiX 4.64 0.8112 0.8043 5.46 0.8510 0.8417 6.73 0.8859 0.8849 OrionMSP 3.76 0.8394 0.8314 4.09 0.8572 0.8478 5.69 0.8860 0.8837 TabPFN 5.30 0.8187 0.8092 4.07 0.8676 0.8589 6.141 0.9129 0.9111 Mitra 11.25 0.3737 0.2683 11.84 0.3886 0.2781 13.03 0.2521 0.1497 ContextTab 9.52 0.6391 0.5719 9.59 0.6480 0.5843 10.97 0.6017 0.5651 TabDPT 4.66 0.8262 0.8189 5.45 0.8566 0.8483 7.23 0.8845 0.8820

0.8821), while TABPFN leads on TALENT (ACC 0.8514) and maintains stable performance across all benchmark suites.

To further investigate the sources of Orion-MSP’s performance gains, we analyze results across key dataset characteristics. All analyses partition datasets based on inherent properties rather than performance outcomes.

Dataset Size. Table 2 reports model performance aggregated by dataset size: Small (< 1K samples), Medium (1K-10K), and Large (> 10). Performance trends reveal that Orion-MSP consistently performs well across small, medium, and large datasets. Classical ML models such as XGBoost excel on large datasets due to abundant training examples, achieving the highest ACC/F1 in the > 10K sample category. Orion-MSP, however, maintains competitive performance across all size categories, outperforming most baselines on small and medium datasets. This demonstrates the ability of multi-scale sparse attention to generalize effectively in low-data regimes while scaling gracefully to larger datasets. TabPFN also performs strongly, particularly on medium-sized datasets, but Orion-MSP’s consistent performance across size scales highlights the robustness of its hierarchical and sparse design.

Feature Dimensionality. Table 3 presents performance trends across narrow (< 10 features), medium (10 - 100) and wide ( > 100) datasets. When evaluating dataset width, Orion-MSP shows the highest accuracy on narrow datasets (<10 features) and strong performance on medium and wide datasets (10–100 and >100 features). This suggests that sparse multi-scale attention enables effective learning even in high-dimensional feature spaces, where dense models such as TabICL exhibit diminished scalability to high-dimensional feature spaces.

Based on Class Imbalance. Partitioning datasets based on class balance reveals that Orion-MSP achieves its strongest gains on imbalanced datasets. The model ranks second in this category, achieving ACC = 0.8840 and F1 = 0.8731. This highlights that multi-scale sparse attention amplifies signals from underrepresented classes while avoiding overfitting to dominant classes. On balanced datasets, performance gains are smaller, suggesting that the architectural complexity of Orion-MSP is most advantageous when datasets exhibit skewed distributions. In comparison, TabPFN maintains strong performance on both balanced and imbalanced datasets, but Orion-MSP’s design more effectively addresses minority-class patterns due to hierarchical attention and cross-scale reasoning.

Table 4 Performance variation by class imbalance across all benchmark suites. ACC = Accuracy; F1 = Weighted F1-score, averaged within each imbalance category. Rank denotes the mean rank within each category (lower is better). Formatting: 1st place ; 2nd place within each group.

Balanced (≥ 0.6) Imbalanced (<0.6) Rank ACC F1 Rank ACC F1

Models

XGBoost 7.00 0.8175 0.8110 6.23 0.8859 0.8785 CatBoost 7.15 0.8076 0.8020 5.65 0.8785 0.8665 Random Forest 7.92 0.7983 0.7955 6.77 0.8741 0.8646 LightGBM 7.32 0.8071 0.7977 6.19 0.8775 0.8633 TabICL 4.72 0.8279 0.8233 5.08 0.8806 0.8698 OrionBiX 5.65 0.8096 0.8040 5.04 0.8787 0.8683 OrionMSP 4.22 0.8265 0.8202 3.38 0.8840 0.8731 TabPFN 3.85 0.8367 0.8309 5.37 0.8808 0.8697 Mitra 12.26 0.2763 0.1540 11.24 0.4794 0.3858 ContextTab 9.66 0.5079 0.4487 9.72 0.7850 0.7192 TabDPT 5.16 0.8233 0.8189 5.65 0.8798 0.8690

Domain-specific Analysis. Domain-wise evaluation provides deeper insight into Orion-MSP’s strengths (Table 5):

- • Medical datasets: Orion-MSP achieves the highest ACC = 0.8045 and F1 = 0.7916, ranking second overall behind Orion-BiX. These datasets often involve hierarchical biological structures and complex interdependencies among features, which align naturally with Orion-MSP’s multi-scale representation. Fine-grained scales capture local dependencies, while coarser scales aggregate contextual information, leading to improved predictive accuracy.
- • Finance datasets: Orion-MSP ranks first in mean rank (4.60), achieving ACC = 0.8158 and F1 = 0.8047. Financial datasets frequently involve layered dependencies between assets, instruments, and market indicators. Orion-MSP’s cross-component memory allows information to propagate across scales, capturing global dependencies that standard dense transformers or classical ML models fail to exploit.

Overall, domain-specific results highlight that Orion-MSP excels in high-dimensional, context-rich datasets, where hierarchical patterns and feature correlations are prevalent.

- Table 5 Domain-specific performance for Medical and Finance datasets from the benchmark suites. Rank denotes the mean rank within each domain (lower is better). ACC = Accuracy; F1 = Weighted F1-score (0–1 scale, higher is better). Formatting: 1st place;

2nd place within each group.

Medical Finance Rank ACC F1 Rank ACC F1

Models

XGBoost 6.32 0.7834 0.7669 6.62 0.7958 0.7885 RandomForest 6.38 0.7779 0.7752 7.32 0.8052 0.8001 CatBoost 6.36 0.7784 0.7594 5.82 0.8117 0.8015 LightGBM 5.32 0.7949 0.7614 6.17 0.8095 0.7974 TabICL 5.54 0.7819 0.7696 6.60 0.8125 0.7942 OrionBiX 4.10 0.7893 0.7759 5.39 0.8206 0.8125 OrionMSP 4.50 0.8045 0.7916 4.60 0.8158 0.8047 TabPFN 5.04 0.7984 0.7857 7.17 0.8094 0.7919 Mitra 10.77 0.3935 0.2863 13.67 0.5340 0.4250 ContextTab 8.66 0.6681 0.6129 11.25 0.7430 0.6834 TabDPT 6.86 0.7764 0.7641 8.00 0.8080 0.7960

#### Deep Analysis and Interpretation

A detailed examination by dataset characteristics demonstrates why Orion-MSP’s design is most effective under certain conditions:

- • Class imbalance: Multi-scale sparse attention amplifies underrepresented patterns without overfitting to majority classes. Minority-class recognition improves substantially on datasets where the minority class constitutes less than 30% of the data. Balanced datasets show smaller gains, indicating that the hierarchical complexity is most beneficial in skewed settings.
- • Hierarchical structure and cross-component memory: In domains such as healthcare and finance, datasets involve natural hierarchies and complex inter-feature relationships. Orion-MSP’s multi-scale design allows

- it to reason at both fine-grained and coarse-grained levels. Sparse attention reduces computational cost and provides implicit regularization, mitigating overfitting in high-dimensional or correlated-feature settings. Cross-component memory further enables information exchange across scales without violating ICL safety, enhancing performance on context-dependent tasks.
- • Computational efficiency: Linear attention complexity with respect to feature number and attention window size allows Orion-MSP to scale to high-dimensional tables. Memory usage grows proportionally with input dimensions, making the model practical for large real-world datasets, unlike dense attention alternatives with quadratic scaling.

In short, fine-grained scales capture subtle minority-class patterns, while coarser scales aggregate global context, yielding a balanced representations of local and global dependencies. Sparse attention improves efficiency and regularization, reducing overfitting in high-dimensional or correlated-feature settings. The Perceiver memory enhances the model’s capacity to store and retrieve non-local patterns, enabling cross-scale reasoning—particularly valuable in contextdependent domains. However, the added architectural complexity offers limited benefit for simpler, low-dimensional datasets, suggesting future directions in adaptive designs with data-driven scale selection and dynamic sparsity control.

## 5 Conclusion

In this work, we introduced Orion-MSP, a novel tabular in-context learning model that leverages multi-scale sparse attention and cross-component memory to capture both fine-grained and coarse-grained dependencies in tabular data. Through extensive experiments across diverse benchmark suites—including TALENT, OpenML-CC18, and TabZilla—as well as domain-specific datasets in healthcare and finance, we demonstrated that Orion-MSP consistently achieves state-of-the-art zero-shot performance, particularly on imbalanced, high-dimensional, and context-rich datasets.

Our detailed analyses highlight that the hierarchical design, sparse attention, and cross-component memory collectively contribute to robust generalization, efficient computation, and improved representation of complex interdependencies. These architectural choices enable Orion-MSP to outperform existing tabular foundation models in challenging realworld scenarios while maintaining practical scalability.

Nonetheless, we observe that the benefits of multi-scale sparse attention are less pronounced on simple, low-dimensional datasets, where the additional architectural complexity may not be fully leveraged. This limitation motivates future work on adaptive scale selection and data-aware sparsity scheduling, allowing model complexity to adjust dynamically to dataset characteristics. Such extensions could further enhance both efficiency and generality, enabling Orion-MSP to provide strong performance across the full spectrum of tabular learning tasks.

In summary, Orion-MSP represents a promising step toward scalable, adaptive, and context-aware tabular in-context learning, with significant potential for real-world applications and future improvements in dynamic model adaptation.

## References

- [1] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. CoRR, abs/2004.05150, 2020.
- [2] Bernd Bischl, Giuseppe Casalicchio, Matthias Feurer, Pieter Gijsbers, Frank Hutter, Michel Lang, Rafael G Mantovani, Jan N van Rijn, and Joaquin Vanschoren. Openml benchmarking suites. arXiv preprint arXiv:1708.03731, 2017.
- [3] Stephan Bongers, Patrick Forré, Jonas Peters, and Joris M Mooij. Foundations of structural causal models with cycles and latent variables. The Annals of Statistics, 49(5):2885–2915, 2021.
- [4] Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom B. Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Raffel. Extracting training data from large language models. In Michael D. Bailey and Rachel Greenstadt, editors, 30th USENIX Security Symposium, USENIX Security 2021, August 11-13, 2021, pages 2633–2650. USENIX Association, 2021.
- [5] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. CoRR, abs/1904.10509, 2019.
- [6] DongHyun Choi, Lucas Spangher, Chris Hidey, Peter Grabowski, and Ramy Eskander. Revisiting funnel transformers for modern LLM architectures with comprehensive ablations in training and inference configurations. CoRR, abs/2504.02877, 2025.
- [7] Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, and Alexander Smola. Autogluon-tabular: Robust and accurate automl for structured data. arXiv preprint arXiv:2003.06505, 2020.

- [8] Xi Fang, Weijie Xu, Fiona Anting Tan, Ziqing Hu, Jiani Zhang, Yanjun Qi, Srinivasan H. Sengamedu, and Christos Faloutsos. Large language models (llms) on tabular data: Prediction, generation, and understanding - A survey. Trans. Mach. Learn. Res., 2024, 2024.
- [9] Josh Gardner, Juan C. Perdomo, and Ludwig Schmidt. Large scale transfer learning for tabular data via language modeling. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.
- [10] Roger Girgis, Florian Golemo, Felipe Codevilla, Martin Weiss, Jim Aldon D’Souza, Samira Ebrahimi Kahou, Felix Heide, and Christopher Pal. Latent variable sequential set transformers for joint multi-agent motion prediction. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.
- [11] Micah Goldblum, Hossein Souri, Renkun Ni, Manli Shu, Viraj Prabhu, Gowthami Somepalli, Prithvijit Chattopadhyay, Mark Ibrahim, Adrien Bardes, Judy Hoffman, et al. Battle of the backbones: A large-scale comparison of pretrained models across computer vision tasks. Advances in Neural Information Processing Systems, 36:29343– 29371, 2023.
- [12] Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko. Tabr: Tabular deep learning meets nearest neighbors. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.
- [13] Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems (NeurIPS), 35:507–520, 2022.
- [14] Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David A. Sontag. Tabllm: Few-shot classification of tabular data with large language models. In Francisco J. R. Ruiz, Jennifer G. Dy, and Jan-Willem van de Meent, editors, International Conference on Artificial Intelligence and Statistics, 25-27 April 2023, Palau de Congressos, Valencia, Spain, volume 206 of Proceedings of Machine Learning Research, pages 5549–5581. PMLR, 2023.
- [15] D Hendrycks. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
- [16] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. Tabpfn: A transformer that solves small tabular classification problems in a second. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.
- [17] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nat., 637(8044):319–326, 2025.
- [18] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier J. Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, and João Carreira. Perceiver IO: A general architecture for structured inputs & outputs. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.
- [19] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021.
- [20] Sebastian Jaszczur, Aakanksha Chowdhery, Afroz Mohiuddin, Lukasz Kaiser, Wojciech Gajewski, Henryk Michalewski, and Jonni Kanerva. Sparse is enough in scaling transformers. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 9895–9907, 2021.
- [21] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [22] Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. Self-normalizing neural networks. Advances in neural information processing systems, 30, 2017.
- [23] Alex Krizhevsky, Geoff Hinton, et al. Convolutional deep belief networks on cifar-10. Unpublished manuscript, 40(7):1–9, 2010.
- [24] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In International conference on machine learning, pages 3744–3753. PMLR, 2019.
- [25] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, ICML 2019,

- 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pages 3744–3753. PMLR, 2019.
- [26] Yinghui Li, Qingyu Zhou, Yuanzhen Luo, Shirong Ma, Yangning Li, Hai-Tao Zheng, Xuming Hu, and Philip S. Yu. When llms meet cunning texts: A fallacy understanding benchmark for large language models. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.
- [27] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM computing surveys, 55(9):1–35, 2023.
- [28] Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Hamidreza Kamkari, Alex Labach, Jesse C. Cresswell, Keyvan Golestan, Guangwei Yu, Maksims Volkovs, and Anthony L. Caterini. Tabdpt: Scaling tabular foundation models. CoRR, abs/2410.18164, 2024.
- [29] Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Ganesh Ramakrishnan, Vishak Prasad, Micah Goldblum, and Colin White. When do neural nets outperform boosted trees on tabular data? In Advances in Neural Information Processing Systems, 2023.
- [30] Li Meng, Morten Goodwin, Anis Yazidi, and Paal Engelstad. Deep reinforcement learning with swin transformers. In Proceedings of the 8th International Conference on Digital Signal Processing, ICDSP 2024, Hangzhou, China, February 23-25, 2024, pages 205–211. ACM, 2024.
- [31] Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do bayesian inference. In International Conference on Learning Representations (ICLR), 2022.
- [32] Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. Tabicl: A tabular foundation model for in-context learning on large data. CoRR, abs/2502.05564, 2025.
- [33] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. Advances in neural information processing systems, 20, 2007.
- [34] Kira A. Selby, Ahmad Rashid, Ivan Kobyzev, Mehdi Rezagholizadeh, and Pascal Poupart. Learning functions on multiple sets using multi-set transformers. In James Cussens and Kun Zhang, editors, Uncertainty in Artificial Intelligence, Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, UAI 2022, 1-5 August 2022, Eindhoven, The Netherlands, volume 180 of Proceedings of Machine Learning Research, pages 1760–1770. PMLR, 2022.
- [35] Marco Spinaci, Marek Polewczyk, Maximilian Schambach, and Sam Thelin. Contexttab: A semantics-aware tabular in-context learner. CoRR, abs/2506.10707, 2025.
- [36] Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [37] Avijit Thawani, Jay Pujara, Filip Ilievski, and Pedro A. Szekely. Representing numbers in NLP: a survey and a vision. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tür, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pages 644–656. Association for Computational Linguistics, 2021.
- [38] Boris van Breugel and Mihaela van der Schaar. Position: Why tabular foundation models should be a research priority. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27,

2024. OpenReview.net, 2024.

- [39] Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun Zhang, James Zou, and Huaxiu Yao. Mmed-rag: Versatile multimodal RAG system for medical vision language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.
- [40] Han-Jia Ye, Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, and De-Chuan Zhan. A closer look at deep learning methods on tabular datasets. arXiv preprint arXiv:2407.00956, 2024.
- [41] Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure Leskovec. Hierarchical graph representation learning with differentiable pooling. Advances in neural information processing systems, 31, 2018.
- [42] Fei Yu, Hongbo Zhang, Prayag Tiwari, and Benyou Wang. Natural language reasoning, a survey. ACM Computing Surveys, 56(12):1–39, 2024.
- [43] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big bird: Transformers for longer sequences. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors,

- Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.
- [44] Neil Zeghidour and David Grangier. Wavesplit: End-to-end speech separation by speaker clustering. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:2840–2849, 2021.
- [45] Jingyang Zhang, Jingwei Sun, Eric C. Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao (Frank) Yang, and Hai Li. Min-k%++: Improved baseline for pre-training data detection from large language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.
- [46] Yujia Zhou, Zheng Liu, and Zhicheng Dou. Boosting the potential of large language models with an intelligent information assistant. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

## A Pretraining and Implementation Details

### A.1 Pretraining Data Generation

Following the pretraining paradigm of TabICL [32], we train our model on synthetically generated datasets to learn generalizable representations for in-context learning on tabular data. Unlike natural language or vision domains where large-scale real data is available, tabular datasets exhibit extreme heterogeneity in schemas, distributions, and task objectives. Synthetic data generation via structural causal models (SCMs) enables us to control dataset diversity while ensuring coverage of diverse statistical patterns.

#### A.1.1 Structural Causal Model (SCM) Prior

We generate synthetic datasets using SCM-based priors [3, 31], where features are related through nonlinear causal relationships. For a dataset with m features, we define a directed acyclic graph (DAG) G = (V,E) where each node v ∈ V represents a feature, and edges E encode causal dependencies.

Each feature Xj is computed as:

##### Xj = fj(Pa(Xj),ϵj) (58)

where Pa(Xj) are the parent features of Xj in G, fj is a nonlinear activation function, and ϵj ∼ N(0,σj2) is Gaussian noise.

Activation Function Diversity To ensure broad coverage of feature transformations observed in real-world tabular data, we used the following activation functions:

- • Identity,
- • tanh,
- • LeakyReLU,
- • ELU
- • Standard nonlinearities: ReLU, ReLU6 [23], SELU [22], SiLU (Swish) [15], Softplus
- • Bounded functions: Hardtanh(x) = max(−1,min(1,x)), Signum function sgn(x)
- • Periodic functions: sin(x) (captures cyclic patterns, e.g., time-of-day)
- • Radial basis function: RBF(x) = exp(−x2) (models local interactions)
- • Exponential growth/decay: exp(x) (models compounding effects, e.g., financial data)
- • Power functions: f(x) = |x|, f(x) = x2, f(x) = |x| (models scaling relationships)

- • Indicator function: f(x) = ⊮|x|≤1 (models threshold effects)
- • Random Fourier features: f(x) = ϕ(x)⊤z where z ∼ N(0,I) and the feature map ϕ : R → RN is defined as:

wi ∥w∥2

· sin(aix + bi), i ∈ {1,...,N} (59)

ϕi(x) :=

with N = 256, bi ∼ U[0,2π], ai ∼ U[0,N], and wi := a−i exp(u) where u ∼ U[0.7,3.0]. This random function approximates complex, non-parametric relationships [33].

For each feature Xj, an activation function fj is sampled uniformly from this extended set, ensuring diverse nonlinear transformations across the dataset.

#### A.1.2 Tree-Based SCM Prior

To complement the continuous SCM prior, we also generate datasets using a tree-based SCM prior [13], where the causal mechanism fj is a decision tree or random forest. This prior is particularly important for modeling categorical interactions and hierarchical decision boundaries commonly observed in real-world tabular data (e.g., credit scoring, medical diagnosis).

For each feature Xj, we construct a random decision tree Tj with:

- • Splitting criteria: Random thresholds on parent features Pa(Xj)
- • Leaf values: Sampled from a Gaussian or uniform distribution
- • Tree depth: Sampled uniformly from {1,2,3,4} to vary complexity

The tree-based prior generates datasets with piecewise-constant relationships, contrasting with the smooth transformations of the MLP-based SCM prior.

### A.2 Pretraining Details

We employ a three-stage curriculum learning strategy that progressively increases dataset size (number of samples per dataset) and refines different architectural components.

- 1. Stage 1 (25K steps, 2,048 datasets) trains all components end-to-end with NB = 8 micro-batches for gradient accumulation, where each dataset contains a fixed size of 1,024 samples. This stage establishes foundational representations across column embedding, multi-scale sparse row interaction, Perceiver memory, and ICL prediction.
- 2. Stage 2 (2K steps, 512 datasets) reduces micro-batch size to NB = 1 and samples dataset sizes from a log-uniform distribution Ulog[1024,40000], exposing the model to variable context lengths while maintaining architectural diversity. Within each micro-batch, all datasets share the same sample count, but this count varies across micro-batches.
- 3. Stage 3 (50 steps, 512 datasets) focuses exclusively on long-context ICL by freezing all components except

TFicl and sampling dataset sizes uniformly from U[40000,60000], ensuring robust in-context learning on large datasets.

Across all stages, we use the Adam optimizer [21] with gradient norm clipping at 1.0 and a learning rate schedule shown in Figure 3. This curriculum, progressing from small, uniform datasets to large, variable datasets with selective fine-tuning, enables the model to generalize effectively across diverse dataset scales while preventing overfitting to specific sample counts.

[Figure 5]

[Figure 6]

[Figure 7]

(a) Cosine decay for stage 1 (b) Polynomial decay for stage 2 (c) Constant lr for stage 3 Figure 3 Learning rate schedules for pretraining stages.

### A.3 Implementation Details

This section provides comprehensive hyperparameters and training configurations for all architectural components of ORION-MSP. Our implementation is built on PyTorch and trained on NVIDIA H200 GPUs.

#### A.3.1 Column-wise Embedding (TFcol) Hyperparameters

- • Embedding dimension: d = 128
- • Number of inducing points: k = 128
- • Attention heads: h = 4 (head dimension dk = 32)
- • ISAB blocks: L = 3
- • Feedforward dimension: dff = 256
- • Dropout rate: p = 0.0
- • Activation: GELU
- • Layer norm: Pre-norm

Initialization and Training Considerations

- • Inducing points initialized from N(0,0.022)
- • Linear and attention weights: Xavier/Glorot uniform
- • Column dropout: pcol = 0.1
- • Gradient clipping: max-norm = 1.0

#### A.3.2 Multi-Scale Sparse Row Interaction Hyperparameters

- • Embedding dimension: d = 128
- • Transformer blocks: Nblocksrow = 6 (2 per scale)
- • Scales: S = {1,4,16}
- • Attention heads: h = 8
- • Window size: w = 8, Global tokens: Nglobal = 4, Random links: r = 2
- • Dropout: p = 0.0
- • Positional encoding: RoPE (θ = 100000)

Training Details

- • Sparse attention via PyTorch SDPA
- • Cosine learning rate schedule with 5% warmup
- • Mixed precision: FP16 (forward), FP32 (softmax)

#### A.3.3 Cross-Component Perceiver Memory Hyperparameters

- • Memory slots: P = 32
- • Write layers: Nwrite = 2, Read layers: Nread = 2
- • Attention heads: h = 4
- • Feedforward dimension: dff = 2dR
- • Dropout: p = 0.0
- • Activation: GELU, Layer norm: Pre-norm

#### Training Considerations

- • Random initialization of memory L0 ∼ N(0,0.022)
- • Gradient clipping (max-norm = 1.0)
- • Memory disabled (P = 0) for ablation

- A.3.4 Dataset-wise In-Context Learning Hyperparameters

- • Transformer blocks: Nicl = 12
- • Embedding dimension: dR = 512
- • Feedforward dimension: dff = 1024
- • Max classes: Cmax = 10
- • Temperature: τ = 0.9
- • Dropout: p = 0.0
- • Activation: GELU
- • Layer norm: Pre-norm Initialization
- • Label encoder, Transformer, and decoder weights: Xavier/Glorot uniform
- • Layer norm: γ = 1, β = 0

## B Further Experiments

Figure 4 presents the relative accuracy improvement (%) of each method over XGBoost, evaluated per dataset across the three benchmarks: TALENT (Figure 4a), TabZilla (Figure 4b), and OpenML-cc18 (Figure 4c). Each point corresponds to a per-dataset delta, while boxplots summarize the distribution (median, interquartile range, and whiskers). The dashed vertical line denotes parity with XGBoost (0%).

Across all three benchmarks, Orion-MSP consistently improves upon the strong XGBoost baseline. On TabZilla and OpenML-cc18, Orion-MSP achieves positive median relative accuracy with a compact interquartile range and few negative outliers, indicating both higher accuracy and greater reliability. On TALENT, Orion-MSP reaches parity in median performance but exhibits lower variance than most neural baselines.

Among classical boosted trees, LightGBM, CatBoost, and Random Forest cluster tightly around parity, showing comparable behavior. In contrast, tabular foundation models (TFMs), notably Mitra and ContextTab, exhibit pronounced negative shifts and high variance. TabPFN and TabICL perform competitively and occasionally outperform Orion-MSP on specific datasets, yet their broader variance and heavier left tails reveal less consistent behavior. Overall, Orion-MSP matches or surpasses their central performance and achieves the best mean–variance trade-off, confirming the benefits of our model design for tabular generalization.

To further quantify cross-dataset performance, we computed per-dataset ranks across all 11 methods (lower is better) for each benchmark, averaged them over datasets, and conducted a one-way Friedman test to assess overall differences. When significant, Nemenyi post-hoc tests were applied, and the resulting critical difference (CD) diagrams were plotted at α = 0.05; methods connected by the CD bar are not significantly different.

As shown in Figure 5, Orion-MSP attains the best average rank on all three benchmarks—3.09 on TALENT, 3.32 on TabZilla, and 3.35 on OpenML-cc18—appearing as the leftmost method in each CD diagram. TabICL, TabPFN, and TabDPT follow closely, typically lagging by ≈ 0.5–1.5 rank points. Tree ensembles (XGBoost, LightGBM, CatBoost, Random Forest) occupy a middle tier with average ranks of 4.6–6.2, showing parity among themselves but a consistent gap to Orion-MSP and the stronger TFMs. Finally, ContextTab and Mitra form the rightmost group (ranks ≈ 9–10), confirming the underperformance seen in the improvement plots.

In summary, the CD diagrams corroborate our main finding: Orion-MSP is the top performer across diverse tabular benchmarks, outperforming tree ensembles on average and matching or exceeding pretrained tabular foundation models while maintaining a favorable significance profile.

## C Datasets

Full details of all datasets and benchmarks are summarized in below, with their row and column distributions visualized in Figure 6.

#### OpenML-CC18 Benchmark Datasets

- Table 6 lists all datasets from the OpenML-CC18 benchmark suite used in our evaluation.

[Figure 8]

- (a) Relative accuracy improvement over XGBoost on TALENT Benchmark

[Figure 9]

- (b) Relative accuracy improvement over XGBoost on TabZilla Benchmark

[Figure 10]

(c) Relative accuracy improvement over XGBoost on OPENML-CC18 Benchmark Figure 4 Relative accuracy improvement over XGBoost on three benchmarks.

[Figure 11]

- (a) Relative accuracy improvement over XGBoost on TALENT Benchmark

[Figure 12]

- (b) Relative accuracy improvement over XGBoost on TabZilla Benchmark

[Figure 13]

(c) Relative accuracy improvement over XGBoost on OPENML-CC18 Benchmark Figure 5

[Figure 14]

Figure 6 Column and row distribution of the evaluated datasets.

#### TALENT Benchmark Datasets

- Table 7 lists all datasets from the TALENT benchmark suite used in our evaluation. TabZilla Benchmark Datasets
- Table 8 lists all datasets from the TabZilla benchmark suite. TabZilla uses OpenML dataset IDs, and these datasets are specifically selected for evaluating neural network performance on tabular data.

Table 6 OpenML-CC18 benchmark datasets (72 datasets).

Benchmark Dataset Name Domain Samples Features Classes Task Type Used In Experimentation OpenML OpenML-ID-3 Other 3196 37 2 binclass Yes OpenML OpenML-ID-6 Handwriting 20000 17 26 multiclass No OpenML OpenML-ID-11 Other 625 5 3 multiclass Yes OpenML OpenML-ID-12 Other 2000 217 10 multiclass Yes

- OpenML OpenML-ID-14 Other 2000 77 10 multiclass Yes
- OpenML OpenML-ID-15 Healthcare 699 10 2 binclass Yes
- OpenML OpenML-ID-16 Other 2000 65 10 multiclass Yes OpenML OpenML-ID-18 Other 2000 7 10 multiclass Yes

- OpenML OpenML-ID-22 Other 2000 48 10 multiclass Yes
- OpenML OpenML-ID-23 Healthcare 1473 10 3 multiclass Yes

- OpenML OpenML-ID-28 Handwriting 5620 65 10 multiclass Yes
- OpenML OpenML-ID-29 Finance 690 16 2 binclass Yes

- OpenML OpenML-ID-31 Finance 1000 21 2 binclass Yes
- OpenML OpenML-ID-32 Handwriting 10992 17 10 multiclass Yes OpenML OpenML-ID-37 Healthcare 768 9 2 binclass Yes OpenML OpenML-ID-38 Healthcare 3772 30 2 binclass Yes OpenML OpenML-ID-44 Other 4601 58 2 binclass Yes OpenML OpenML-ID-46 Other 3190 61 3 multiclass Yes OpenML OpenML-ID-50 Other 958 10 2 binclass Yes OpenML OpenML-ID-54 Other 846 19 4 multiclass Yes OpenML OpenML-ID-151 Other 45312 9 2 binclass Yes OpenML OpenML-ID-182 Other 6430 37 6 multiclass Yes OpenML OpenML-ID-188 Other 736 20 5 multiclass Yes OpenML OpenML-ID-300 Other 7797 618 26 multiclass No OpenML OpenML-ID-307 Other 990 13 11 multiclass No OpenML OpenML-ID-458 Other 841 71 4 multiclass Yes OpenML OpenML-ID-469 Healthcare 797 5 6 multiclass Yes OpenML OpenML-ID-554 Other 70000 785 10 multiclass Yes

- OpenML OpenML-ID-1049 Other 1458 38 2 binclass Yes
- OpenML OpenML-ID-1050 Other 1563 38 2 binclass Yes OpenML OpenML-ID-1053 Other 10885 22 2 binclass Yes OpenML OpenML-ID-1063 Other 522 22 2 binclass Yes

- OpenML OpenML-ID-1067 Other 2109 22 2 binclass Yes
- OpenML OpenML-ID-1068 Other 1109 22 2 binclass Yes

- OpenML OpenML-ID-1461 Finance 45211 17 2 binclass Yes
- OpenML OpenML-ID-1462 Finance 1372 5 2 binclass Yes OpenML OpenML-ID-1464 Healthcare 748 5 2 binclass Yes OpenML OpenML-ID-1468 Other 1080 857 9 multiclass Yes OpenML OpenML-ID-1475 Other 6118 52 6 multiclass Yes OpenML OpenML-ID-1478 Other 10299 562 6 multiclass Yes OpenML OpenML-ID-1480 Healthcare 583 11 2 binclass Yes

- OpenML OpenML-ID-1485 Other 2600 501 2 binclass Yes
- OpenML OpenML-ID-1486 Other 34465 119 2 binclass No
- OpenML OpenML-ID-1487 Other 2534 73 2 binclass Yes OpenML OpenML-ID-1489 Other 5404 6 2 binclass Yes OpenML OpenML-ID-1494 Other 1055 42 2 binclass Yes OpenML OpenML-ID-1497 Other 5456 25 4 multiclass Yes OpenML OpenML-ID-1501 Other 1593 257 10 multiclass Yes OpenML OpenML-ID-1510 Other 569 31 2 binclass Yes OpenML OpenML-ID-1590 Other 48842 15 2 binclass Yes

Table 6 Details of OpenML-CC18 benchmark datasets.

Benchmark Dataset Name Domain Samples Features Classes Task Type Used In Experimentation OpenML OpenML-ID-4134 Other 3751 1777 2 binclass No OpenML OpenML-ID-4534 Other 11055 31 2 binclass Yes OpenML OpenML-ID-4538 Other 9873 33 5 multiclass Yes OpenML OpenML-ID-6332 Other 540 40 2 binclass Yes OpenML OpenML-ID-23381 Retail 500 13 2 binclass Yes OpenML OpenML-ID-23517 Other 96320 22 2 binclass No OpenML OpenML-ID-40499 Other 5500 41 11 multiclass No OpenML OpenML-ID-40668 Games 67557 43 3 multiclass No OpenML OpenML-ID-40670 Other 3186 181 3 multiclass Yes OpenML OpenML-ID-40701 Other 5000 21 2 binclass Yes OpenML OpenML-ID-40923 Other 92000 1025 46 multiclass No OpenML OpenML-ID-40927 Handwriting 60000 3073 10 multiclass No OpenML OpenML-ID-40966 Other 1080 82 8 multiclass No OpenML OpenML-ID-40975 Other 1728 7 4 multiclass Yes

- OpenML OpenML-ID-40978 Other 3279 1559 2 binclass Yes
- OpenML OpenML-ID-40979 Other 2000 241 10 multiclass Yes

- OpenML OpenML-ID-40982 Other 1941 28 7 multiclass Yes
- OpenML OpenML-ID-40983 Other 4839 6 2 binclass Yes
- OpenML OpenML-ID-40984 Other 2310 20 7 multiclass Yes OpenML OpenML-ID-40994 Other 540 21 2 binclass Yes OpenML OpenML-ID-40996 Other 70000 785 10 multiclass No OpenML OpenML-ID-41027 Games 44819 7 3 multiclass Yes

Table 7 TALENT benchmark datasets (auto-discovered, multiple domains). Benchmark Dataset Name Domain Samples Features Classes Task Type Used In Experimentation

TALENT ASP-POTASSCO-class Other 1294 141 11 multiclass Yes TALENT Amazon_employee_access Other 32769 7 2 binclass Yes TALENT BLE_RSSI__Indoor_localization Other 9984 3 3 multiclass Yes TALENT BNG(breast-w) Healthcare 39366 9 2 binclass Yes TALENT BNG(cmc) Other 55296 9 3 multiclass Yes TALENT BNG(tic-tac-toe) Other 39366 9 2 binclass Yes TALENT Bank_Customer_Churn Finance 10000 10 2 binclass Yes TALENT Basketball_c Retail 1340 11 2 binclass Yes TALENT CDC_Diabetes_Health Healthcare 253680 21 2 binclass Yes TALENT California-Housing-Class Other 20640 8 2 binclass No TALENT Cardiovascular-Disease Healthcare 70000 11 2 binclass Yes TALENT Click_prediction_small Other 39948 3 2 binclass Yes TALENT Credit_c Finance 100000 22 3 multiclass Yes TALENT Customer_Personality_Analysis Retail 2240 24 2 binclass Yes TALENT DataScience_Kiva_Crowdfunding Other 671205 11 4 multiclass No TALENT Diabetic_Retinopathy_Debrecen Healthcare 1151 19 2 binclass Yes TALENT E-CommereShippingData Other 10999 10 2 binclass Yes TALENT Employee Other 4653 8 2 binclass Yes TALENT FICO-HELOC-cleaned Other 9871 23 2 binclass Yes TALENT FOREX_audcad-day-High Finance 1834 10 2 binclass No TALENT FOREX_audcad-hour-High Finance 43825 10 2 binclass No TALENT FOREX_audchf-day-High Finance 1833 10 2 binclass No TALENT FOREX_audjpy-day-High Finance 1832 10 2 binclass No TALENT FOREX_audjpy-hour-High Finance 43825 10 2 binclass No TALENT FOREX_audsgd-hour-High Finance 43825 10 2 binclass No TALENT FOREX_audusd-hour-High Finance 43825 10 2 binclass No TALENT FOREX_cadjpy-day-High Finance 1834 10 2 binclass No TALENT FOREX_cadjpy-hour-High Finance 43825 10 2 binclass No TALENT Firm-Teacher_Clave-Direction Other 10800 16 4 multiclass Yes TALENT Fitness_Club_c Other 1500 6 2 binclass Yes TALENT GAMETES_Epistasis_2-Way Games 1600 20 2 binclass Yes

TALENT GAMETES_Heterogeneity Games 1600 20 2 binclass Yes TALENT Gender_Gap_in_Spanish Other 4746 13 3 multiclass Yes TALENT GesturePhaseSegmentation Other 9873 32 5 multiclass Yes TALENT HR_Analytics_Job_Change Other 19158 13 2 binclass Yes TALENT IBM_HR_Analytics Other 1470 31 2 binclass Yes TALENT INNHotelsGroup Other 36275 17 2 binclass Yes TALENT Indian_pines Other 9144 220 8 multiclass Yes TALENT JapaneseVowels Other 9961 14 9 multiclass Yes TALENT KDDCup09_upselling Other 5128 49 2 binclass Yes TALENT MIC Other 1649 104 2 binclass Yes TALENT MagicTelescope Other 19020 9 2 binclass Yes TALENT Marketing_Campaign Finance 2240 27 2 binclass Yes TALENT Mobile_Price_Classification Telcom 2000 20 4 multiclass Yes TALENT National_Health_and_Nutrition Healthcare 2278 7 2 binclass Yes TALENT PhishingWebsites Other 11055 30 2 binclass Yes TALENT PieChart3 Other 1077 37 2 binclass Yes TALENT Pima_Indians_Diabetes Healthcare 768 8 2 binclass Yes TALENT PizzaCutter3 Other 1043 37 2 binclass Yes TALENT Pumpkin_Seeds Other 2500 12 2 binclass Yes TALENT QSAR_biodegradation Healthcare 1054 41 2 binclass Yes TALENT Rain_in_Australia Other 145460 18 3 multiclass No TALENT SDSS17 Other 100000 12 3 multiclass Yes TALENT Satellite Other 5100 36 2 binclass Yes TALENT Smoking_and_Drinking Other 991346 23 2 binclass No TALENT Telecom_Churn_Dataset Telcom 3333 17 2 binclass Yes TALENT UJI_Pen_Characters Other 1364 80 35 multiclass Yes TALENT Water_Quality_and_Potability Manufacturing 3276 8 2 binclass Yes TALENT Wilt Other 4821 5 2 binclass Yes TALENT abalone Other 4177 8 3 multiclass Yes TALENT accelerometer Other 153004 4 4 multiclass No TALENT ada Other 4147 48 2 binclass Yes TALENT ada_agnostic Other 4562 48 2 binclass Yes TALENT ada_prior Other 4562 14 2 binclass Yes TALENT airlines_seed_0_nrows Telcom 2000 7 2 binclass Yes TALENT allbp Other 3772 29 3 multiclass Yes TALENT allrep Other 3772 29 4 multiclass Yes TALENT analcatdata_authorship Other 841 69 4 multiclass Yes TALENT artificial-characters Other 10218 7 10 multiclass Yes TALENT autoUniv-au4-2500 Other 2500 100 3 multiclass Yes TALENT autoUniv-au7-1100 Other 1100 12 5 multiclass Yes TALENT bank Finance 45211 16 2 binclass Yes TALENT banknote_authentication Finance 1372 4 2 binclass Yes TALENT baseball Other 1340 16 3 multiclass Yes TALENT car-evaluation Other 1728 21 4 multiclass Yes TALENT churn Finance 5000 20 2 binclass Yes TALENT cmc Other 1473 9 3 multiclass Yes TALENT company_bankruptcy_prediction Finance 6819 95 2 binclass Yes TALENT compass Other 16644 17 2 binclass Yes TALENT contraceptive_method_choice Other 1473 9 3 multiclass Yes TALENT credit Finance 16714 10 2 binclass Yes TALENT customer_satisfaction_in_airline Retail 129880 21 2 binclass No TALENT dabetes_130-us_hospitals Healthcare 101766 20 2 binclass Yes TALENT default_of_credit_card_clients Finance 30000 23 2 binclass Yes TALENT delta_ailerons Other 7129 5 2 binclass Yes TALENT dis Other 3772 29 2 binclass Yes TALENT dna Healthcare 3186 180 3 multiclass Yes TALENT drug_consumption Healthcare 1884 12 7 multiclass Yes TALENT dry_bean_dataset Other 13611 16 7 multiclass Yes TALENT eeg-eye-state Other 14980 14 2 binclass Yes

Continued on next page

TALENT electricity Manufacturing 45312 8 2 binclass Yes TALENT estimation_of_obesity_levels Other 2111 16 7 multiclass Yes TALENT eye_movements Healthcare 10936 27 3 multiclass Yes TALENT eye_movements_bin Healthcare 7608 20 2 binclass Yes TALENT first-order-theorem-proving Other 6118 51 6 multiclass Yes TALENT gas-drift Other 13910 128 6 multiclass Yes TALENT gina_agnostic Other 3468 970 2 binclass Yes TALENT golf_play_dataset_extended Other 1095 9 2 binclass Yes TALENT heloc Other 10000 22 2 binclass Yes TALENT hill-valley Other 1212 100 2 binclass No TALENT house_16H Other 13488 16 2 binclass Yes TALENT htru Other 17898 8 2 binclass Yes TALENT ibm-employee-performance Other 1470 30 2 binclass Yes TALENT in_vehicle_coupon_recos Retail 12684 21 2 binclass Yes TALENT internet_firewall Other 65532 7 4 multiclass Yes TALENT internet_usage Other 10108 70 46 multiclass Yes TALENT jm1 Other 10885 21 2 binclass No TALENT jungle_chess_2pcs_raw Other 44819 6 3 multiclass Yes TALENT kc1 Other 2109 21 2 binclass No TALENT kdd_ipums_la_97-small Other 5188 20 2 binclass Yes TALENT kr-vs-k Other 28056 6 18 multiclass Yes TALENT kropt Other 28056 6 18 multiclass Yes TALENT led24 Other 3200 24 10 multiclass Yes TALENT led7 Other 3200 7 10 multiclass Yes TALENT letter Other 20000 15 26 multiclass Yes TALENT madeline Other 3140 259 2 binclass Yes TALENT mammography Other 11183 6 2 binclass Yes TALENT maternal_health_risk Healthcare 1014 6 3 multiclass Yes TALENT mfeat-factors Other 2000 216 10 multiclass Yes TALENT mfeat-fourier Other 2000 76 10 multiclass Yes TALENT mfeat-karhunen Other 2000 64 10 multiclass Yes TALENT mfeat-morphological Other 2000 6 10 multiclass Yes TALENT mfeat-pixel Other 2000 240 10 multiclass Yes TALENT mfeat-zernike Other 2000 47 10 multiclass Yes TALENT mice_protein_expression Other 1080 75 8 multiclass Yes TALENT microaggregation2 Other 20000 20 5 multiclass Yes TALENT mobile_c36_oversampling Telcom 51760 6 2 binclass Yes TALENT mozilla4 Other 15545 4 2 binclass Yes TALENT naticusdroid+android Healthcare 29332 86 2 binclass Yes TALENT national-longitudinal-survey-

Other 4908 16 2 binclass Yes

binary

TALENT okcupid_stem Other 26677 13 3 multiclass Yes TALENT one-hundred-plants-margin Other 1600 64 100 multiclass Yes TALENT one-hundred-plants-shape Other 1600 64 100 multiclass Yes TALENT one-hundred-plants-texture Other 1599 64 100 multiclass Yes TALENT online_shoppers Retail 12330 14 2 binclass No TALENT optdigits Other 5620 64 10 multiclass Yes TALENT ozone-level-8hr Other 2534 72 2 binclass Yes TALENT page-blocks Other 5473 10 5 multiclass Yes TALENT pc1 Other 1109 21 2 binclass No TALENT pc3 Other 1563 37 2 binclass No TALENT pc4 Other 1458 37 2 binclass No TALENT pendigits Other 10992 16 10 multiclass Yes TALENT phoneme Other 5404 5 2 binclass Yes TALENT pol Other 10082 26 2 binclass Yes TALENT predict_students_dropout Other 4424 34 3 multiclass Yes TALENT qsar Other 1055 40 2 binclass Yes TALENT rice_cammeo_and_osmancik Other 3810 7 2 binclass Yes TALENT ringnorm Other 7400 20 2 binclass Yes

Continued on next page

TALENT rl Other 4970 12 2 binclass No TALENT satimage Other 6430 36 6 multiclass Yes TALENT segment Other 2310 17 7 multiclass Yes TALENT seismic+bumps Other 2584 18 2 binclass Yes TALENT semeion Other 1593 256 10 multiclass No TALENT shuttle Other 58000 9 7 multiclass Yes TALENT spambase Other 4601 57 2 binclass Yes TALENT splice Other 3190 60 3 multiclass Yes TALENT sports_articles_for_objectivity Other 1000 59 2 binclass Yes TALENT statlog Other 1000 20 2 binclass Yes TALENT steel_plates_faults Other 1941 27 7 multiclass Yes TALENT sylvine Other 5124 20 2 binclass Yes TALENT taiwanese_bankruptcy Finance 6819 95 2 binclass Yes TALENT telco-customer-churn Telcom 7043 18 2 binclass Yes TALENT texture Other 5500 40 11 multiclass Yes TALENT thyroid Healthcare 7200 21 3 multiclass Yes TALENT thyroid-ann Healthcare 3772 21 3 multiclass Yes TALENT thyroid-dis Healthcare 2800 26 5 multiclass Yes TALENT turiye_student_evaluation Other 5820 32 5 multiclass Yes TALENT twonorm Other 7400 20 2 binclass Yes TALENT vehicle Other 846 18 4 multiclass Yes TALENT volkert Other 58310 180 10 multiclass Yes TALENT walking-activity Other 149332 4 22 multiclass No TALENT wall-robot-navigation Other 5456 24 4 multiclass Yes TALENT water_quality Manufacturing 7996 20 2 binclass Yes TALENT waveform-5000 Other 5000 40 3 multiclass Yes TALENT waveform_database_generator Other 4999 21 3 multiclass Yes TALENT waveform_database_generator-v2 Other 5000 21 3 multiclass Yes TALENT website_phishing Other 1353 9 3 multiclass Yes TALENT wine Manufacturing 2554 4 2 binclass No TALENT wine-quality-red Manufacturing 1599 4 6 multiclass Yes TALENT wine-quality-white Manufacturing 4898 11 7 multiclass Yes TALENT yeast Other 1484 8 10 multiclass Yes

Table 8 TabZilla benchmark datasets (36 datasets via OpenML). Benchmark Dataset Name Domain Samples Features Classes Task Type Used In Experimentation

TabZilla OpenML-ID-999 Health 226 70 2 binclass Yes TabZilla OpenML-ID-10 Health 148 19 4 multiclass Yes TabZilla OpenML-ID-11 Other 625 5 3 multiclass Yes TabZilla OpenML-ID-14 Other 2000 77 10 multiclass Yes TabZilla OpenML-ID-22 Other 2000 48 10 multiclass Yes TabZilla OpenML-ID-29 Finance 690 16 2 binclass Yes TabZilla OpenML-ID-27 Health 368 23 2 binclass Yes TabZilla OpenML-ID-31 Finance 1000 21 2 binclass Yes TabZilla OpenML-ID-46 Other 3190 61 3 multiclass Yes TabZilla OpenML-ID-54 Other 846 19 4 multiclass Yes TabZilla OpenML-ID-333 Other 556 7 2 binclass Yes TabZilla OpenML-ID-1067 Other 2109 22 2 binclass Yes TabZilla OpenML-ID-1468 Other 1080 857 9 multiclass Yes TabZilla OpenML-ID-1494 Other 1055 42 2 binclass Yes TabZilla OpenML-ID-43973 Other 3172 6 2 binclass Yes TabZilla OpenML-ID-1043 Other 4562 49 2 binclass Yes TabZilla OpenML-ID-43945 Other 38474 9 2 binclass Yes TabZilla OpenML-ID-1486 Other 34465 119 2 binclass No TabZilla OpenML-ID-42825 Other 8378 123 – – No TabZilla OpenML-ID-4538 Other 9873 33 5 multiclass Yes

Continued on next page

Table 8 Details of TabZilla benchmark datasets.

Benchmark Dataset Name Domain Samples Features Classes Task Type Used In Experimentation TabZilla OpenML-ID-23512 Other 98050 29 2 binclass No TabZilla OpenML-ID-4134 Other 3751 1777 2 binclass Yes TabZilla OpenML-ID-470 Other 672 10 2 binclass No TabZilla OpenML-ID-1493 Other 1599 65 100 multiclass Yes TabZilla OpenML-ID-1459 Other 10218 8 10 multiclass Yes TabZilla OpenML-ID-41027 Games 44819 7 3 multiclass Yes TabZilla OpenML-ID-40981 Other 690 15 2 binclass Yes TabZilla OpenML-ID-934 Other 1156 6 2 binclass Yes TabZilla OpenML-ID-1565 Health 294 14 5 multiclass Yes TabZilla OpenML-ID-41150 Other 130064 51 2 binclass No TabZilla OpenML-ID-41159 Other 20000 4297 2 binclass No TabZilla OpenML-ID-846 Other 16599 19 2 binclass Yes TabZilla OpenML-ID-1169 Other 539383 8 2 binclass No TabZilla OpenML-ID-41147 Other 425240 79 2 binclass Yes TabZilla OpenML-ID-41143 Other 2984 145 2 binclass Yes TabZilla OpenML-ID-1567 Other 1025009 11 10 multiclass No

