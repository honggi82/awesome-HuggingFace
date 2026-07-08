## Xetrieval: Mechanistically Explaining Dense Retrieval

### Zhixin Cai1⋆, Jun Bai2⋆, Yang Liu2⋆, Jiaqi Li2, Yichi Zhang1, Taichuan Li1, Zhuofan Chen1, Zixia Jia2, Zilong Zheng2†, Wenge Rong1

- 1School of Computer Science and Engineering, Beihang University
- 2State Key Laboratory of General Artificial Intelligence, BIGAI

# arXiv:2605.29507v1[cs.AI]28May2026

### Abstract

Explaining why dense retrievers assign high relevance scores remains challenging because retrieval decisions are made through opaque high-dimensional embeddings. Existing explanations often focus on surface signals, such as lexical matches, token alignments, or post-hoc textual rationales, and thus provide limited insight into the latent factors that shape dense retrieval behavior at the embedding level. We propose Xetrieval, an embedding-level mechanistic framework for explaining dense retrieval. Xetrieval first introduces a lightweight reasoning internalizer that approximates Chain-ofThought reasoning directly in the embedding space with a single forward pass, enriching sentence embeddings with reasoning-oriented information while avoiding expensive autoregressive generation. It then decomposes these reasoning-enhanced embeddings into sparse, human-interpretable features, each associated with a coherent natural language description. By aggregating sparse feature overlaps across multiple document-side views, Xetrieval provides feature-level explanations of individual retrieval decisions. Experiments on diverse retrievers and benchmarks show that Xetrieval uncovers coherent interpretable features, yields stronger pair-level intervention effects, and supports task-level feature steering1.

### 1 Introduction

Dense retrieval (DR) has become central to information retrieval, achieving state-of-the-art performance across diverse tasks (Xiao et al., 2024; Zhang et al., 2025a; Günther et al., 2025). However, this success comes at the cost of transparency: relevance is computed through high-dimensional query and document embeddings, making it difficult to understand why a particular document is

⋆Equal contribution. †Corresponding author.

1The project page and source code are available at https: //hihiczx.github.io/Xetrieval

###### Dense Retrieval is Opaque

Relevance = 0.87

[Figure 1]

###### Retrieved Document

Query

Dense Retriever

Document (black box)

Why this doc?

Figure 1: Dense retrieval offers limited insight into the rationales underlying individual retrieval results.

retrieved for a given query (cf. Fig. 1) (Opitz et al., 2025). As dense retrieval systems are increasingly deployed in real-world applications, this opacity limits their use in settings that require accountability, diagnosis, and systematic error analysis (Hou et al., 2025; Bai et al., 2025).

Existing work has explained dense retrieval through lexical or token-level evidence (Formal et al., 2021; Khattab and Zaharia, 2020), inherently interpretable embedding spaces based on semantic aspects or QA dimensions (Opitz and Frank, 2022; Benara et al., 2024), and post-hoc analyses of fixed encoders via attribution, subspace probing, or embedding decoding (Moeller et al., 2023; Nikolaev and Padó, 2023; Kang et al., 2025; Park et al., 2025; Saxena et al., 2026). Despite this progress (Opitz et al., 2025), these methods often rely on surface-level evidence, predefined semantic dimensions, or architectural and training modifications, offering limited insight into the latent factors encoded in standard dense embeddings where retrieval scores are computed. This motivates a framework that directly explains off-theshelf dense retrievers by decomposing embedding similarity into sparse, human-interpretable factors.

We propose Xetrieval, a sparse feature-based framework for explaining dense retrieval. Xetrieval decomposes query and document embeddings into sparse, interpretable features, each associated with

Reasoning Feature Enrichment

Sentence Encoding

Document Retrieval

- 1. Document A
- 2. Document B

Relevance Scores

Reasoning Internalizer

| | |
|---|---|
| | |
| | |

Query Embedding

Retrieved Documents

Query

Encoder

Summary

Reason. Embed.

Document

Encoder

Doc Embedding

Purpose

Reason. Embed.

###### Mechanistic Explanation

Explaining Relevance Score

QA

Reason. Embed.

Black Box Lack of Explainability

sparse activations

Steering Key Features

Efficient Internal Reasoning

Mechanistic Explainer

Interpretable Reasoning Features

- Figure 2: Overview of the Xetrieval framework. The reasoning internalizer injects reasoning-oriented signals into sentence embeddings, while the mechanistic explainer decomposes these enriched embeddings into sparse, human-readable features for feature-level analysis and intervention on retrieval behavior.

a coherent natural-language description. For each retrieval decision, it identifies the features jointly activated by the query and the retrieved document, and attributes the dense relevance score to these shared feature-level matches. In this way, Xetrieval reveals which latent semantic factors drive querydocument similarity, providing a model-internal and embedding-level mechanistic explanation of dense retrieval decisions.

However, standard sentence embeddings often encode relevance in an entangled form, providing limited reasoning-oriented clues for explaining retrieval decisions (Park et al., 2025). To address this limitation, we enrich Xetrieval with LLM-generated Chain-of-Thought (CoT) reasoning, which injects reasoning-centric information, such as query intent, latent constraints, and evidence requirements, into the embedding space (Qin et al., 2025; Zhang et al., 2025b; Chen et al., 2025). Since explicit CoT generation incurs substantial auto-regressive decoding cost (Jin et al., 2026; Li et al., 2026), we further introduce a lightweight reasoning internalizer that learns to approximate this reasoning-enhanced representation directly within the embedding space. This enables Xetrieval to obtain reasoning-aware sparse features in a single forward pass, bypassing costly generation while preserving the explanatory benefits of CoT-enriched embeddings. As a result, mechanistically explainable dense retrieval becomes practical for largescale retrieval scenarios.

Experiments across multiple retrievers and

benchmarks demonstrate that Xetrieval efficiently internalizes LLM reasoning and produces higherquality sparse representations. Feature-quality analyses show that the learned sparse features are coherent and human-interpretable, while feature-level intervention experiments verify that intervening on these features changes retrieval outcomes, providing evidence that Xetrieval captures feature-level mechanisms underlying dense retrieval decisions.

### 2 The Xetrieval Framework

As illustrated in Fig. 2, Xetrieval combines a reasoning internalizer with a mechanistic explainer to provide embedding-level explanations for dense retrieval. The reasoning internalizer approximates LLM-generated CoT reasoning directly in the embedding space, enriching embeddings with reasoning-oriented information such as query intent, latent constraints, and evidence requirements. This yields more structured representations that facilitate the decomposition of dense embeddings into sparse, interpretable factors.

Given a query and its retrieved documents, the mechanistic explainer decomposes their enriched embeddings into sparse, human-interpretable features. For each query-document pair, it identifies the features jointly activated by both sides and attributes the relevance score to these shared featurelevel matches. These sparse features provide a model-internal account of individual retrieval decisions and also support controllable interventions on retrieval behavior. The following sections first

introduce the necessary preliminaries, and then describe the reasoning internalizer and the mechanistic explainer in detail.

#### 2.1 Preliminaries

Notation. We denote queries and documents by q and d, and vectors by bold symbols (e.g., q,z). For dimension m, ⟨·,·⟩ denotes the inner product and ∥ · ∥2 the Euclidean norm.

Dense Retrieval. A dense retriever maps queries and documents into a shared embedding space and ranks documents by relevance. With query encoder EQ(·) and document encoder ED(·), for query q and document d:

q = EQ(q) ∈ Rm, z = ED(d) ∈ Rm. (1)

A standard relevance score is the dot product or cosine similarity:

s(q,d) = ⟨q,z⟩ or s(q,d) = ⟨q,z⟩ ∥q∥2 ∥z∥2

. (2)

At inference time, document embeddings are precomputed and indexed offline in practice, and retrieval reduces to nearest-neighbor search in Rm.

Explaining Relevance Score. Explainable dense retrieval identifies latent factors underlying querydocument relevance. In Xetrieval, these explanations are sparse mechanistic factors co-activated in query and document representations.

Let q˜ and z˜ denote the query and document representations analyzed by the mechanistic explainer, respectively, and let

##### cq = g(q˜), cd = g(z˜). (3)

be their sparse codes generated by the encoder g(·), which are binarized into activation supports:

##### aq,j = I[cq,j > τ], ad,j = I[cd,j > τ]. (4)

where τ is an activation threshold. The shared support between the query and document is

O(q,d) = {j | aq,jad,j = 1}. (5) We return the explanation for a pair (q,d) as

E(q,d) = {(j,hj)}j∈O(q,d). (6)

where hj is the natural-language hypothesis associated with sparse feature j, and O(q,d) denotes the shared active features selected for presentation.

Thus, E(q,d) consists of shared sparse factors that connect the query and the retrieved document in the mechanistic feature space.

We seek explanations that are (i) embeddinglevel, derived from the representations used by the retrieval scorer; (ii) interpretable, expressed through human-readable feature hypotheses; and (iii) efficient, scaling to large corpora.

2.2 Reasoning Internalizer The reasoning internalizer injects reasoning features into sentence embeddings in a single step.

- 2.2.1 Architecture Design We instantiate three aspect-specific reasoning internalizers to capture complementary reasoning aspects: SUMMARY, PURPOSE, and QA. Here, SUMMARY captures the input’s core semantics, PURPOSE reflects its retrieval-oriented intent and utility, and QA encodes questionanswering-style evidence needs. Formally, let T := {SUMMARY, PURPOSE, QA} denote the set of reasoning aspects. For each t ∈ T , the inter-

nalizer Rt is implemented as a one-hidden-layer MLP with a tanh activation, mapping a raw sentence embedding zi ∈ Rm to a reasoning-enhanced embedding of the same dimension:

zˆ(it) = Rt(zi), zˆ(it) ∈ Rm. (7)

- 2.2.2 Training the Reasoning Internalizer To construct supervision for reasoning internalization, we collect documents from StackExchange (Lambert et al., 2023), covering a wide

range of tasks. For each document di, we prompt an LLM to generate 3 task-oriented reasoning texts, corresponding to the aspects in T . The original document and each generated reasoning text are then encoded by the same dense encoder, yielding the raw embedding zi and the aspect-specific reasoning target z(it).

The internalizer Rt is trained to approximate this reasoning-enhanced target directly from the raw embedding. For each aspect t, we minimize the mean squared error:

Lt = Ei Rt(zi) − z(it)

2 2

. (8)

After training, Rt can produce reasoning-enhanced embeddings through a single forward pass, avoiding autoregressive LLM generation during retrieval and explanation.

#### 2.3 Mechanistic Explainer

The mechanistic explainer decomposes reasoningenhanced embeddings into sparse, interpretable features for explaining query-document relevance.

- 2.3.1 Architecture Design We instantiate the mechanistic explainer with a SAE (Cunningham et al., 2023), which decomposes dense embeddings into sparse feature activations. Conceptually, an SAE extends dictionary learning by representing an input vector x ∈ Rm

- as sparse activations over learned feature directions (Rajamanoharan et al., 2024a). This suits dense retrieval explanation by identifying a small set of latent features activated in both queries and retrieved documents.

Given an embedding x, the SAE encoder g(·) produces a sparse code c, from which the decoder reconstructs x using the learned feature dictionary:

c = g(x), x˜ = Wc + b. (9)

Here, the columns of W correspond to learned feature directions, while nonzero entries in c indicate the sparse features activated by x. After retrieval, the mechanistic explainer applies the SAE encoder to the reasoning-enhanced embeddings of the query and retrieved documents, obtaining sparse feature representations that can be compared and attributed

- at the feature level.

- 2.3.2 Training the Mechanistic Explainer To capture reasoning-related sparse features, we construct the SAE training set from StackExchange (Lambert et al., 2023), including both raw document embeddings and reasoning-enhanced embeddings produced by the reasoning internalizer. We evaluate several SAE variants implemented in the dictionary_learning library (Marks et al., 2024), including ReLU (Cunningham et al., 2023), TopK (Gao et al., 2024), BatchTopK (Bussmann et al., 2024), Gated (Rajamanoharan et al., 2024a), JumpReLU (Rajamanoharan et al., 2024b), PAnnealing (Karvonen et al., 2024), and GatedAnnealing (Rajamanoharan et al., 2024a).

The explainer parameters (g,W,b) are optimized with reconstruction and sparsity losses:

L = Ex x − (Wg(x) + b) 22

+ λΩ g(x) .

(10)

where Ω(·) enforces sparsity and λ controls the strength of the sparsity penalty.

### 3 Experiments

#### 3.1 Experimental Setup

Benchmarks. We evaluate Xetrieval on 7 retrieval benchmarks: BRIGHT (Su et al., 2024), NQ (Kwiatkowski et al., 2019), MuTual (Cui et al., 2020), TREC-NEWS (Soboroff et al., 2019), Signal-1M (Suarez et al., 2018), ArguAna (Wachsmuth et al., 2018), and Robust04 (Voorhees, 2005). They span reasoningintensive retrieval, open-domain QA, multi-turn dialogue, news, argument, and robust ad-hoc retrieval. We use NDCG@10 as the main metric.

LLMs. We use DeepSeek-V2-Lite (Liu et al.,

- 2024a), DeepSeek-V3 (Liu et al., 2024b), DeepSeek-R1 (Guo et al., 2025), Qwen332B (Yang et al., 2025), GPT-OSS-20B, and GPTOSS-120B (Agarwal et al., 2025) to generate aspect-specific reasoning texts. These texts are used as supervision for reasoning internalization.

Dense Retrievers. We adopt eight dense retrievers across multiple model families and parameter scales: e5-small (Wang et al., 2024), e5-base (Wang et al., 2022), and gte-base (Li et al., 2023) at around 0.1B parameters; e5large (Wang et al., 2022), gte-large (Li et al., 2023), and Snowflake-Arctic-Embed (Yu et al., 2024) at around 0.3B parameters; and Qwen3-Embedding0.6B and Qwen3-Embedding-4B (Zhang et al.,

- 2025a) as recent LLM-based embedding models.

#### 3.2 Best Practice of Mechanistic Explainer

We adopt a multi-faceted evaluation framework (Park et al., 2025) to examine how SAE structures affect the mechanistic explainer.

- • Reconstruction Error: It computes the mean squared error between the original embeddings and the reconstructed embeddings, indicating how well the sparse features preserve the geometric structure of the embedding space.
- • Mono-Semanticity: For each sparse feature, we select its 9 most activating documents and add one non-activating intruder. LLM intruder-detection accuracy is used as the monosemanticity score, with higher values indicating stronger semantic coherence.
- • Retrieval Retention: It performs dense retrieval using embeddings reconstructed by the mechanistic explainer and reports NDCG@10, measuring how well the sparse reconstruction retains taskrelevant retrieval behavior.

###### RetrievalRetention(%)

12

###### MonoSemanticity(%)

| | |
|---|---|
| | |

###### ReconstructionError

Baseline (11.0)

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| |
|---|

10

80

10 4

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

8

| |
|---|

| |
|---|

| | |
|---|---|
| | |

60

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| |
|---|

6

| | |
|---|---|
| | |

| |
|---|

| | |
|---|---|
| | |

| |
|---|

40

4

| |
|---|

| |
|---|

2

20

50 150 250 350 450

50 150 250 350 450

50 150 250 350 450

Sparsity (L0)

Sparsity (L0)

Sparsity (L0)

BatchTopK GatedAnneal GatedSAE JumpReLU P-Anneal ReLU TopK

- Figure 3: SAEs comparison across sparsity levels (L0), measured by reconstruction error, mono-semanticity, and retrieval retention. The dashed line shows the original dense-embedding performance without SAE reconstruction.

Retriever Enhancement BRIGHT NQ Mutual Trec. Signal1m Robust04 ArguAna Avg.

None 37.0 81.0 28.8 92.2 73.8 77.1 41.7 61.7 Reasoning Internalizer 39.0 80.8 29.6 92.3 74.2 80.2 40.9 62.4

gte-base

###### CoT Reasoner 43.8 83.3 30.3 93.4 74.6 84.0 41.7 64.4

None 31.5 83.3 47.1 90.4 66.8 77.3 34.2 61.5 Reasoning Internalizer 37.9 84.2 46.5 90.3 70.3 81.1 39.2 64.2

e5-large

###### CoT Reasoner 43.8 86.3 47.0 92.8 72.0 82.1 41.3 66.5

None 51.2 84.0 45.2 92.3 74.1 87.0 50.7 69.2 Reasoning Internalizer 51.7 83.5 44.9 91.9 72.8 87.1 49.3 68.7

qwen3-4b

CoT Reasoner 54.8 84.6 45.8 92.9 73.2 86.7 43.8 68.8

None 34.8 48.1 36.2 22.5 64.8 24.1 37.2 38.3 Reasoning Internalizer 38.8 68.9 36.3 64.9 67.9 42.7 38.6 51.2

snowflake

CoT Reasoner 44.0 74.2 33.0 77.6 67.4 46.0 40.5 54.7

Table 1: NDCG@10 (%) of dense retrievers under different enhancements. The reasoning internalizer and CoT reasoner are powered by DeepSeek-V3; NONE denotes the unenhanced baseline.

As shown in Fig. 3, a clear trade-off emerges among the three evaluation axes. As L0 increases, more sparse features are allowed to be active, which improves reconstruction quality and retrieval retention but generally weakens mono-semanticity. Conversely, enforcing stronger sparsity with a smaller L0 produces more selective and interpretable features, but increases reconstruction error and weakens retrieval retention.

Overall, TopK exhibits the most favorable tradeoff across all three axes: it consistently attains low reconstruction error while maintaining the strongest mono-semanticity over a wide range of sparsity levels. At L0 = 256, TopK preserves strong mono-semanticity while achieving near-baseline retrieval retention, with competitive reconstruction error. We therefore adopt TopK-SAE with k = 256 as the backbone of the mechanistic explainer.

#### 3.3 Reasoning Benefits Explainability

Retrieval-based Validation. We first verify whether the reasoning internalizer preserves retrieval-relevant reasoning signals in the embedding space. Here, the CoT reasoner denotes an

explicit LLM-based module that generates aspectspecific reasoning texts for each document and encodes them as reasoning embeddings. The reasoning internalizer is trained to approximate these CoT-derived embeddings directly from the raw document embedding, avoiding autoregressive generation at inference time. For this diagnostic evaluation, each document di is represented by its raw embedding zi and a set of internalized reasoning embeddings {zˆ(it)}t∈T . Given a query embedding q, we compute the query-document score as

⟨q,zˆ(it)⟩. (11)

s(q,di) = ⟨q,zi⟩ +

t∈T

Table 1 reports the retrieval performance of dense retrievers augmented with either the reasoning internalizer or the explicit CoT reasoner. The reasoning internalizer consistently improves over the base retriever in most settings and recovers part of the retrieval gain achieved by the CoT reasoner. For stronger embedding backbones such as Qwen3-Embedding, additional reasoning views still improve BRIGHT, although the average gain is smaller because the base retriever already performs

| |
|---|

###### NumberofActiveFeatures

5000

10 4

| |
|---|

###### MSELoss(logscale)

| |
|---|

| |
|---|

4000

| |
|---|

3000

| |
|---|

| |
|---|

2000

| |
|---|

| |
|---|

| |
|---|

10 5

| |
|---|

1000

| |
|---|

0 100 200 300 400 500

0 100 200 300 400 500

Sparsity (L0)

Sparsity (L0)

Raw Embedding

Reasoned Embedding

| | |
|---|---|
| | |

- Figure 4: Comparison of reconstruction error (Left side) and the number of active features (Right side) between raw and reasoned embeddings.

- 0

- 1

- 2

- 3

- 4

Raw SAE

Random SAE

Mechanistic Explainer

Density

0.0 0.2 0.4 0.6 0.8 1.0 Detection Score

Figure 5: Detection score distribution of Raw SAE, Random SAE, and Mechanistic Explainer estimated using kernel density estimation.

strongly on several benchmarks. Although it does not fully match the CoT-enhanced retriever, it preserves useful retrieval-relevant reasoning signals within the embedding space.

Effect on Mechanistic Explainability. We further examine how internalized reasoning affects the mechanistic explainer. Specifically, we compare the explainer on raw embeddings from e5-large and reasoned embeddings produced by the reasoning internalizer. We evaluate reconstruction and decomposition quality using MSE and Active Feature Count, where the latter denotes the average number of sparse features whose activations exceed the threshold for each embedding. As shown in Fig. 4, reasoned embeddings achieve lower reconstruction error and activate more sparse features under the same sparsity-control settings. This suggests that reasoning internalization makes the embedding space more amenable to sparse decomposition, enabling the mechanistic explainer to recover richer feature-level factors without sacrificing reconstruction quality. Unless otherwise specified, we report results with e5-large as the retriever and DeepSeek-V3 as the CoT reasoner2.

#### 3.4 Interpretability of Sparse Features

After decomposing sentence embeddings into sparse features, we adopt an automated explanation pipeline (Paulo et al., 2024; Park et al., 2025) to equip these sparse features with natural language descriptions. Specifically, for each active sparse feature, we retrieve the top-activating samples from the training dataset. An LLM is then invoked to summarize these sentences into a concise semantic hypothesis that characterizes the feature.

To assess the semantic coherence of the generated feature descriptions, we compute the

2Results under other configurations are provided in Appendix A.2.

Detection Score (Paulo et al., 2024). For each feature-hypothesis pair, we present an LLM with a balanced set of activating and non-activating sentences and ask it to determine whether each sentence conforms to the hypothesis. The resulting classification accuracy (Detection Score) serves as a proxy for feature mono-semanticity and semantic coherence of the generated feature descriptions. We compare the mechanistic explainer with two baselines: a Random SAE, which serves as an untrained control, and a Raw SAE, which is trained on raw embeddings. As shown in Fig. 5, the mechanistic explainer augmented with the reasoning internalizer substantially outperforms both baselines, producing features that are markedly more distinguishable. This improvement can be attributed to the reasoned embeddings generated by the reasoning internalizer, which encode richer reasoningrelated features and provide a more structured and semantically coherent representation space for the mechanistic explainer to disentangle.

3.5 Explaining Retrieval with Xetrieval 3.5.1 Feature-based Explanation

Given a query-document pair (q,d), Xetrieval explains the retrieval decision by identifying sparse features jointly activated by the query and document-side views.

For a document embedding zd, the reasoning

internalizer produces aspect-specific views zˆ(dt) = Rt(zd), where t ∈ T . Together with the original document embedding, these views form

V(d) = {zd} ∪ {zˆ(dt) : t ∈ T }. (12)

Let g(·) denote the SAE encoder used by the mechanistic explainer. For the query, we compute its sparse code and binary activation indicators as

##### cq = g(q), aq,j = I[cq,j > τ]. (13)

50

- 100

- 101

- 102

- 103

- 104

40

###### Latency(seconds)

###### NDCG@10(%)

| |
|---|

30

| |
|---|

| |
|---|

| |
|---|

20

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

10

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

10 1

10,000 20,000 30,000 40,000 50,000 60,000

10,000 20,000 30,000 40,000 50,000 60,000

Number of Documents

Number of Documents

Baseline Xetrieval Baseline w/ CoT Reasoner

Figure 6: Left side: Comparison of explanation time trends between the CoT reasoner and the Xetrieval on the Biology subset of BRIGHT. Right side: Comparison of retrieval performance trends between the base retriever, the retriever with CoT reasoner, and Xetrieval.

For each document view v ∈ V(d), we compute

cv = g(v), av,j = I[cv,j > τ]. (14)

Xetrieval aggregates the feature overlaps between the query and all document views:

av,j = 1 . (15)

O(q,d) = j | aq,j · max

v∈V(d)

The final explanation is

E(q,d) = {(j,hj)}j∈O(q,d). (16)

where hj is the natural-language description associated with feature j.

Unlike direct decomposition, Xetrieval aggregates feature overlaps across multiple document views, revealing relevance features that are weak or entangled in the original representation but become salient after reasoning internalization. Steering experiments further confirm their stronger connection to query-document relevance.

#### 3.5.2 Explanation Efficiency

To evaluate explanation efficiency, we compare Xetrieval with a CoT reasoner on the Biology subset of BRIGHT, scaling the corpus size and measuring explanation time.

As shown in Fig. 6 left side, the CoT reasoner incurs substantial computational overhead that grows approximately linearly with the number of documents. In contrast, Xetrieval operates with only a lightweight feed-forward pass over sentence embeddings, introducing negligible additional computation even as the corpus size scales. Importantly, as the candidate set expands (see Fig. 6 right side), Xetrieval consistently outperforms the basic dense retriever and achieves performance that is competitive with the CoT-reasoner-enhanced retriever.

###### Erase Intervention

0.075

ScoreDrop

0.050

0.025

0.000

0.025

###### Keep-only Intervention

0.075

0.050

ScoreDelta

0.025

0.000

0.025

0.050

ArguAna BRIGHT Mutual NQ Robust04 Signal1M TREC-News

Direct Decomposition Xetrieval Overlap

Non-overlap Active

| |
|---|

Figure 7: Pair-level document-side intervention results. We report cosine-similarity changes after erasing or retaining selected feature spans for Xetrieval, direct decomposition, and non-overlap active features.

#### 3.6 Feature-level Intervention Analyses

We next examine whether the selected sparse features are interventionally linked to retrieval behavior. We consider two complementary settings: document-side intervention for local attribution, and task-level steering for global utility.

- 3.6.1 Local Attribution Given the feature set O(q,d) returned for a querydocument pair, we treat the corresponding explainer directions as the explanation span. We intervene on the original document embedding by either erasing the component aligned with this span or retaining only this component.

We evaluate three feature sets: Xetrieval features, direct decomposition features, and non-overlap active features. As shown in Fig. 7, erasing Xetrieval features yields the largest decrease in the original similarity score. In contrast, erasing non-overlap active features often increases the score, suggesting that these features capture query-irrelevant or distracting document information. The retention intervention shows a complementary pattern: retaining only Xetrieval features preserves or increases the similarity more effectively than direct decomposition, whereas retaining only non-overlap active features decreases it. These results indicate that Xetrieval selects feature spans that are more closely tied to the local query-document relevance signal.

- 3.6.2 Task-level Feature Steering We further examine whether sparse features can capture task-level mechanisms that consistently af-

###### BRIGHT

###### ArguAna

###### NQ

| |
|---|

35

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |

10

| |
|---|

| |
|---|

80

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

30

| |
|---|

| |
|---|

| |
|---|

70

| |
|---|

###### NDCG@10(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

8

25

| |
|---|

60

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

20

50

| |
|---|

6

40

15

| |
|---|

| |
|---|

30

4

10

0.0 0.5 1.0 1.5 2.0

0.0 0.5 1.0 1.5 2.0

0.0 0.5 1.0 1.5 2.0

Scaling Factor ( )

Scaling Factor ( )

Scaling Factor ( )

SAE (Key) SAE (Non-Key) Xetrieval (Key) Xetrieval (Non-Key)

Figure 8: Retrieval results when steering key features and non-key features identified by basic SAE and Xetrieval.

fect ranking performance. For each feature fj, we define its co-activation indicator as

##### Ij(q,d) = aq,jad,j. (17)

where aq,j and ad,j indicate whether feature j is active in the query and document representations. Each feature is then scored by the Retrieval Utility Score (RUS), a contrastive co-activation frequency:

##### RUS(fj) =

Ij(q,d), (18)

Ij(q,d) −

(q,d)∈Dpos

(q,d)∈Dneg

where Dpos and Dneg denote matched and unmatched query-document pairs, respectively. We select the top-ranked features as the key set S and compare them with a same-sized non-key set Sc. Before decoding sparse codes, we scale selected activations by α, where α > 1 amplifies features and α < 1 suppresses them. Retrieval is then evaluated with the intervened embeddings on BRIGHT, ArguAna, and NQ.

As shown in Fig. 8, amplifying key features improves retrieval performance, while suppressing them leads to clear degradation. Steering non-key features causes smaller and less consistent changes. Compared with direct decomposition using raw SAE, Xetrieval identifies key features with stronger steering effects, suggesting that its sparse features better capture high-impact retrieval mechanisms.

### 4 Related Work

Recently, dense retrieval has advanced substantially in model scale, training strategies, and data construction. This progress has produced specialized embedding models such as E5 (Wang et al., 2022), GTE (Li et al., 2023), and BGE (Xiao et al., 2024), which improve representation quality and retrieval performance. More recently, LLM-driven retrievers, including Qwen3 Embedding (Zhang et al.,

2025a) and Jina Embedding (Günther et al., 2025), have leveraged LLMs’ semantic understanding to generate richer embeddings. Meanwhile, increasing attention has been paid to reasoning-intensive retrieval (Su et al., 2024), where CoT-enhanced dense retrievers support complex inference and multi-step reasoning (Shao et al., 2025).

Parallel to these advances, growing efforts have sought to explain dense retrieval, mainly through inherently interpretable architectures and post-hoc explanations (Opitz et al., 2025). The former reshapes embedding spaces around human-understandable features, such as predefined question answers (Benara et al., 2024), semantic aspects (Opitz and Frank, 2022), sparse lexical weights as in SPLADE (Formal et al., 2021), or token-level alignments as in ColBERT (Khattab and Zaharia, 2020). The latter explains black-box retrievers via interaction attributions (Moeller et al., 2023), surrogate models (Nikolaev and Padó, 2023), or SAE-based decomposition into sparse latent features (Park et al., 2025; Kang et al., 2025; Lupart et al., 2026). However, existing methods either depend on specialized architectures, expose mainly lexical evidence, or analyze raw embedding spaces without targeting reasoning-oriented relevance factors that connect semantically distant query-document pairs.

### 5 Conclusion and Future Work

We propose Xetrieval, an embedding-level framework for explaining dense retrieval beyond opaque similarity scores. By internalizing reasoning and decomposing embeddings into interpretable features, Xetrieval traces decisions to latent querydocument factors. Interventions show that these features are locally grounded in similarity computation and globally useful for retrieval behavior. Future work may extend Xetrieval to multi-modal and cross-lingual retrieval, adaptive reasoning pathways, and fairness-aware explanation evaluation.

### Limitations

While Xetrieval reveals the latent factors driving retrieval scores, our analysis is confined to the sentence embedding level, the output layer of the embedding model, without probing the internal circuits of the model itself. A deeper understanding of retrieval behavior would require investigating the internal representations and interactions throughout the full embedding network. Additionally, we rely on SAE to decompose sentence embeddings and attribute retrieval decisions, which, although effective, offers limited fidelity and granularity compared to more advanced mechanisms such as Transcoder. Future work should explore these stronger interpretability frameworks to provide more precise and mechanistic explanations of dense retrieval outcomes.

### Ethical Considerations

This work aims to improve the transparency of dense retrieval systems by exposing sparse, humanreadable factors behind retrieval decisions. It may benefit auditing, debugging, and failure analysis for retrieval applications. Potential risks include over-interpreting imperfect explanations or using them as definitive justifications in high-stakes settings. We therefore recommend using Xetrieval as an analysis tool rather than as a standalone decisionmaking mechanism. All datasets used in this study are publicly available research resources, and no private user data is used.

### References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, and 1 others. 2025. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925.

Jun Bai, Zhenzi Li, Bo Zhao, Chen Li, Chenghua Lin, and Wenge Rong. 2025. Rectifying and discriminating hard negatives for biomedical retrieval question answering. IEEE Transactions on Computational Biology and Bioinformatics.

Vinamra Benara, Chandan Singh, John X Morris, Richard J Antonello, Ion Stoica, Alexander G Huth, and Jianfeng Gao. 2024. Crafting interpretable embeddings for language neuroscience by asking llms questions. Advances in neural information processing systems, 37:124137.

Bart Bussmann, Patrick Leask, and Neel Nanda. 2024. Batchtopk: A simple improvement for topksaes. In AI Alignment Forum, page 17.

Peter Baile Chen, Tomer Wolfson, Michael Cafarella, and Dan Roth. 2025. Enrichindex: Using llms to enrich retrieval indices offline. arXiv preprint arXiv:2504.03598.

Leyang Cui, Yu Wu, Shujie Liu, Yue Zhang, and Ming Zhou. 2020. Mutual: A dataset for multi-turn dialogue reasoning. arXiv preprint arXiv:2004.04494.

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. 2023. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600.

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. Splade: Sparse lexical and expansion model for first stage ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2288–2292.

Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. 2024. Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093.

Michael Günther, Saba Sturua, Mohammad Kalim Akram, Isabelle Mohr, Andrei Ungureanu, Bo Wang, Sedigheh Eslami, Scott Martens, Maximilian Werk, Nan Wang, and 1 others. 2025. jina-embeddings-v4: Universal embeddings for multimodal multilingual retrieval. In Proceedings of the 5th Workshop on Multilingual Representation Learning, pages 531–550.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Abe Bohan Hou, Orion Weller, Guanghui Qin, Eugene Yang, Dawn Lawrie, Nils Holzenberger, Andrew Blair-Stanek, and Benjamin Van Durme. 2025. Clerc: A dataset for us legal case retrieval and retrievalaugmented analysis generation. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 7898–7913.

Jiajie Jin, Yanzhao Zhang, Mingxin Li, Dingkun Long, Pengjun Xie, Yutao Zhu, and Zhicheng Dou. 2026. Laser: Internalizing explicit reasoning into latent space for dense retrieval. arXiv preprint arXiv:2603.01425.

Hao Kang, Tevin Wang, and Chenyan Xiong. 2025. Interpret and control dense retrieval with sparse latent features. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 700–709.

Adam Karvonen, Benjamin Wright, Can Rager, Rico Angell, Jannik Brinkmann, Logan Smith, Claudio

Mayrink Verdun, David Bau, and Samuel Marks. 2024. Measuring progress in dictionary learning for language model interpretability with board game models. Advances in Neural Information Processing Systems, 37:83091–83118.

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39– 48.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, and 1 others. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466.

Nathan Lambert, Lewis Tunstall, Nazneen Rajani, and Tristan Thrush. 2023. Huggingface h4 stack exchange preference dataset. URL: https://huggingface. co/datasets/HuggingFaceH4/stack-exchangepreferences.

Juncai Li, Ru Li, Yuxiang Zhou, Boxiang Ma, and Jeff Z Pan. 2026. Chain of thought compression: A theoritical analysis. arXiv preprint arXiv:2601.21576.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, and 1 others. 2024a. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024b. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Simon Lupart, Maxime Louis, Thibault Formal, Hervé Déjean, and Stéphane Clinchant. 2026. On the challenges and opportunities of learned sparse retrieval for code. arXiv preprint arXiv:2603.22008.

Samuel Marks, Adam Karvonen, and Aaron Mueller.

2024. dictionary_learning. Lucas Moeller, Dmitry Nikolaev, and Sebastian Padó.

2023. An attribution method for siamese encoders. arXiv preprint arXiv:2310.05703.

Dmitry Nikolaev and Sebastian Padó. 2023. Investigating semantic subspaces of transformer sentence embeddings through linear structural probing. arXiv preprint arXiv:2310.11923.

Juri Opitz and Anette Frank. 2022. Sbert studies meaning representations: Decomposing sentence embeddings into explainable semantic features. arXiv preprint arXiv:2206.07023.

Juri Opitz, Lucas Moeller, Andrianos Michail, Sebastian Padó, and Simon Clematide. 2025. Interpretable text embeddings and text similarity explanation: A survey. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 22314–22330.

Seongwan Park, Taeklim Kim, and Youngjoong Ko. 2025. Decoding dense embeddings: Sparse autoencoders for interpreting and discretizing dense retrieval. arXiv preprint arXiv:2506.00041.

Gonçalo Paulo, Alex Mallen, Caden Juang, and Nora Belrose. 2024. Automatically interpreting millions of features in large language models. arXiv preprint arXiv:2410.13928.

Xubo Qin, Jun Bai, Jiaqi Li, Zixia Jia, and Zilong Zheng.

2025. Tongsearch-qr: Reinforced query reasoning for retrieval. arXiv preprint arXiv:2506.11603.

Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Tom Lieberum, Vikrant Varma, János Kramár, Rohin Shah, and Neel Nanda. 2024a. Improving dictionary learning with gated sparse autoencoders. arXiv preprint arXiv:2404.16014.

Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda. 2024b. Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders. arXiv preprint arXiv:2407.14435.

Yash Saxena, Ankur Padia, Kalpa Gunaratna, and Manas Gaur. 2026. IMRNNs: An efficient method for interpretable dense retrieval via embedding modulation. In Findings of the Association for Computational Linguistics: EACL 2026, pages 6324–6337, Rabat, Morocco. Association for Computational Linguistics.

Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muennighoff, Xi Victoria Lin, Daniela Rus, Bryan Kian Hsiang Low, Sewon Min, Wen-tau Yih, Pang Wei Koh, and 1 others. 2025. Reasonir: Training retrievers for reasoning tasks. arXiv preprint arXiv:2504.20595.

Ian Soboroff, Shudong Huang, and Donna Harman.

2019. Trec 2019 news track overview. In TREC, volume 409, page 410.

Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan Shi, Zachary S Siegel, Michael Tang, and 1 others. 2024. Bright: A realistic and challenging benchmark for reasoning-intensive retrieval. arXiv preprint arXiv:2407.12883.

Axel Suarez, Dyaa Albakour, David Corney, Miguel Martinez, and José Esquivel. 2018. A data collection for evaluating the retrieval of related tweets to news articles. In European Conference on Information Retrieval, pages 780–786.

Ellen Voorhees. 2005. Overview of the trec 2004 robust retrieval track.

Henning Wachsmuth, Shahbaz Syed, and Benno Stein. 2018. Retrieval of the best counterargument without prior topic knowledge. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, pages 241–251.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weaklysupervised contrastive pre-training. arXiv preprint arXiv:2212.03533.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Multilingual e5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack: Packed resources for general chinese embeddings. In Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, pages 641–649.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Puxuan Yu, Luke Merrick, Gaurav Nuti, and Daniel Campos. 2024. Arctic-embed 2.0: Multilingual retrieval without compromise. arXiv preprint arXiv:2412.04506.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, and 1 others. 2025a. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

Yichi Zhang, Jun Bai, Zhixin Cai, Shuhan Qin, Zhuofan Chen, Jinghua Guan, and Wenge Rong. 2025b. Your dense retriever is secretly an expeditious reasoner. arXiv preprint arXiv:2510.21727.

### A Details of Reasoning Internalizer

#### A.1 Training Details

Data Construction. We construct the training pairs from a StackExchange-derived corpus (Lambert et al., 2023). Each training instance consists of an original document d and a reasoned text r(t)(d)

Community # Docs Community # Docs politics 1,000 mathematica 1,000 codereview 600 economics 600 cs 600 chemistry 600 StackOverflow 600 ai 600 bioinformatics 600 codegolf 600 math 600 robotics 600 earthscience 600 mathoverflow 600 biology 600 philosophy 600 softwareengineering 600 sustainability 432 computergraphics 364 Total 11,796

Table 2: Domain distribution of the StackExchange corpus used to construct reasoning internalizer training pairs.

produced by an LLM teacher for a reasoning aspect t ∈ {SUMMARY, PURPOSE, QA}. We then embed the original document and the reasoned text using the same retriever, yielding paired embeddings (z,z(t)). We train one reasoning internalizer Rt per aspect t to reason z  → z(t).

Domain Distribution. To improve coverage and reduce domain bias, we sample documents from multiple StackExchange communities. Table 2 summarizes the domain distribution of the sampled corpus (total 11,796 documents).

LLM Teacher Prompts. For each document text, we prompt LLM teacher to generate reasoning contents of three aspects. We use the following prompts (Table 3). For the QA aspect, we treat the returned list of question–answer pairs as a single text block and embed it as z(QA).

Model Architecture. Each reasoning internalizer Rt is a one-hidden-layer MLP with tanh activation:

Rt(z) = Norm W2 tanh(W1z) . (19)

where W1 ∈ Rm×h, W2 ∈ Rh×m, m is the embedding dimension of the underlying encoder, h is the hidden size, and Norm(·) denotes ℓ2-normalization along the feature dimension. We train three separate reasoning internalizer (for SUMMARY, PURPOSE, and QA).

Optimization. We train reasoning internalizer using mean squared error (MSE) between predicted and target embeddings:

##### Lt = Ei ∥Rt(zi) − z(it)∥22 . (20)

Type Prompt PURPOSE Given the following text, describe the purpose of this text in layman’s terms

in one paragraph. {doc}

SUMMARY Given the following text, summarize this text in layman’s terms in one paragraph. {doc}

QA Given the following text, generate at most 20 distinct question-answer pairs on this text. The questions should be general, and phrased in layman’s terms, using vocabulary that can be distinct from the text, but still requires explicit or implicit knowledge from the text. Only output the question-answer pairs, no other explanation.

{doc}

Table 3: LLM teacher prompts used to generate task-oriented reasoning content from StackExchange documents.

We use Adam with learning rate 5 × 10−4, batch size 128, and train for up to 100 epochs. We split the embedding pairs into 85% training and 15% validation and apply early stopping with patience 5 based on validation loss. We set the hidden dimension to h = 512.

Time Cost. In practice, each reasoning internalizer converges quickly and typically finishes training within 1–2 minutes due to the lightweight architecture. At inference time, reasoning internalizer performs a single feed-forward pass on cached embeddings and typically completes reasoning in seconds.

#### A.2 Evaluation Details

Dataset Sampling Strategy. Considering the high cost of generating LLM-based CoT reasoning content for large-scale corpora, we sample a subset from each benchmark.

For BRIGHT, we process each domain subset independently: we first collect all ground-truth documents from each subset, then randomly sample additional documents from the full corpus to reach 1,000 documents per subset. We then aggregate all queries and documents across subsets to form a unified BRIGHT evaluation set.

For other benchmarks (NQ, MuTual, TRECNEWS, Signal-1M, Robust04, ArguAna), we follow a similar approach: we collect all ground-truth documents, and if the corpus size is below 10,000, we randomly sample additional documents to reach this target. Table 4 summarizes the final corpus statistics across all evaluated benchmarks.

Dataset # Queries # Documents BRIGHT 1,384 12,000 NQ 8,383 8,383 MuTual 846 3,542 TREC-NEWS 57 9,968 Signal-1M 97 10,000 Robust04 249 15,790 ArguAna 1,406 8,674

Table 4: Statistics of the sampled benchmarks.

Additional Results. Table 6-11 reports NDCG@10 when training reasoning internalizer with supervision generated by different LLM teachers. Across all dense retrievers, we observe the same qualitative trend: regardless of the training data source, reasoning internalizer consistently outperforms the base retriever and closely approaches the performance of the CoT reasoner, indicating reasoning internalizer well preserves LLM reasoning content within the embedding space.

### B Mechanistic Explainer Details

#### B.1 SAE Details

Training Data Construction. We build the SAE training corpus on top of the reasoning internalizer training dataset and further include additional StackExchange domains that are relevant to retrieval and reasoning. In total, the SAE training corpus contains 84,860 documents.

We use Deepseek-V3 as the CoT reasoner with the prompt shown in Table 3 to generate CoT reasoning contents for SAE training.

###### MS Prompt

You are an expert linguist analyzing pieces of documents. Below, you will see a set of documents that has some common features, but one of them is an intruder (it does not have that common feature in it). Your task is to identify the intruder document and explain why it does not fit.

The last line of your response must be the formatted response, using “[intruder]:Document#”

{documents} Which document is the intruder, and why?

- Table 5: Prompt used for Mono-Semanticity evaluation via intruder detection.

Evaluation. As described in Section 3.1, We evaluate the learned sparse features using three complementary metrics: Reconstruction Error, MonoSemanticity, and Retrieval Retention. Reconstruction Error is computed as the mean squared error between the original embeddings and their SAE reconstructions over 100 documents sampled from BRIGHT Biology subset. Retrieval Retention is measured by conducting retrieval with reconstructed embeddings on the BRIGHT benchmark.

For Mono-Semanticity evaluation, we apply the intruder detection paradigm to the entire SAE training corpus. For each feature, we first identify documents where the feature activation exceeds a minimum threshold of 50. From this pool, we sample 500 features uniformly randomly. For each sampled feature, we select the top-9 documents with the highest activation values and insert one randomly sampled non-activating document as an intruder. These 10 documents is then presented to Qwen3-32B with the prompt shown in Table 5.

#### B.2 Explaining Details

We use Qwen3-32B to generate feature explanation and make evaluation. The prompts are shown in Table 12.

#### B.3 Case Studies

To further illustrate how Xetrieval mechanistically explains retrieval decisions across diverse query aspects, we present four additional case studies in Table 13-16. Each case demonstrates the semantic gap between raw document embeddings and query embeddings, and how the reasoning internalizer bridges this gap by uncovering deeper reasoning

aspects.

### C Feature-level Intervention Details

#### C.1 Local Attribution

The pair-level intervention experiment asks whether the features returned as an explanation for a particular query-document pair are locally tied to the similarity decision. We therefore keep the query representation fixed and intervene only on the original document embedding.

Pair Sampling. For each query, we first rank the corpus with the original retriever using cosine similarity between the original query and document embeddings. We use the top-K retrieved documents as the candidate pool, with K=32 in our experiments. From this pool, we construct two types of query-document pairs: true positives, whose document id appears in the relevance annotations, and false positives, which are retrieved in the top-K but are not annotated as relevant. Dataset-provided excluded documents are removed before ranking. To avoid a few queries dominating the average, we sample at most four pairs per query from the union of true-positive and false-positive candidates, using a fixed random seed. Duplicate documents for the same query are removed before sampling.

Feature Set Construction. Let zq and zd denote the original query and document embeddings. For direct decomposition, we encode both embeddings with the SAE trained on original embeddings and select the overlap features

Sdirect(q,d) = supp(graw(zq))∩supp(graw(zd)).

(21) For Xetrieval, the query side is still the original query embedding, while the document side is expanded by the reasoning internalizer. Specifically, we map zd into three reasoning-oriented views, corresponding to QA, summary, and purpose. Together with the original document embedding, these form

V(d) = {zd,Rqa(zd),Rsummary(zd),Rpurpose(zd)}.

(22) We then encode the query and all document views with the Xetrieval SAE and take the union of all query-document overlaps:

(supp(gx(zq)) ∩ supp(gx(v))).

Sx(q,d) =

v∈V(d)

(23)

Reasoning Internalizer 36.8 82.4 41.6 90.5 71.4 79.5 39.8 63.2 CoT Reasoner 43.6 84.4 41.5 92.1 72.7 81.1 41.8 65.3

e5-base

None 31.5 83.3 47.1 90.4 66.8 77.3 34.2 61.5 Reasoning Internalizer 38.7 84.0 46.8 89.7 69.3 81.8 40.6 64.4

e5-large

###### CoT Reasoner 44.8 85.6 45.3 93.0 72.5 82.2 42.1 66.5

None 23.4 77.0 38.5 86.3 60.8 70.4 29.1 55.1 Reasoning Internalizer 30.6 77.0 38.8 88.3 64.9 76.0 36.8 58.9

e5-small

###### CoT Reasoner 37.6 80.7 37.5 91.6 68.8 74.9 38.8 61.4

None 37.0 81.0 28.8 92.2 73.8 77.1 41.7 61.7 Reasoning Internalizer 39.4 80.6 29.9 91.8 73.9 80.0 40.6 62.3

gte-base

###### CoT Reasoner 44.4 83.2 28.1 92.9 73.2 83.0 41.8 63.8

None 41.2 83.0 31.2 92.0 73.6 79.3 41.8 63.2 Reasoning Internalizer 42.8 82.8 31.4 92.7 73.0 81.3 41.5 63.6

gte-large

###### CoT Reasoner 46.7 84.5 31.7 93.0 73.5 83.9 41.7 65.0

None 34.8 48.1 36.2 22.5 64.8 24.1 37.2 38.3 Reasoning Internalizer 39.7 69.8 36.9 66.4 68.3 42.9 38.8 51.8

snowflake

###### CoT Reasoner 45.9 74.5 34.2 82.0 68.6 49.3 40.4 56.4

None 44.5 78.0 40.0 89.7 71.4 83.6 48.2 65.1 Reasoning Internalizer 45.9 77.8 39.3 89.8 70.7 83.6 47.2 64.9

qwen3-0.6b

CoT Reasoner 49.2 80.0 39.4 90.9 70.6 84.0 44.8 65.6

None 51.2 84.0 45.2 92.3 74.1 87.0 50.7 69.2 Reasoning Internalizer 51.7 83.6 44.5 92.3 73.6 87.1 49.7 68.9

qwen3-4b

CoT Reasoner 54.6 84.4 45.0 93.0 72.2 86.4 45.1 68.7

- Table 6: Retrieval NDCG@10 (%) scores when dense retriever enhanced by reasoning internalizer and CoT reasoner (both empowered by DeepSeek-R1) under varying configurations (None denotes no enhancement is employed, i.e., the baseline dense retriever).

Thus, the reasoning internalizer is used only to expose additional candidate features for the explanation; the intervention target remains the original document embedding zd. As a control, we also evaluate non-overlap active features, defined as active features of the original document embedding under the corresponding SAE dictionary, excluding the selected overlap set.

Decoder-direction Intervention. Because a TopK SAE may not activate every relevant feature on the original document embedding, we do not edit the sparse code directly. Instead, we use the decoder directions associated with the selected features as a linear span in the embedding space. For a feature set S, let WS be the matrix of decoder directions and let b be the decoder bias. We compute the ridge projection

PS(zd−b) = WS(WS⊤WS +λI)−1WS⊤(zd−b), (24)

with λ=10−6. We evaluate two complementary interventions:

z\dS = zd − PS(zd − b), (25)

zSd = b + PS(zd − b). (26) The first erases the component aligned with the selected feature span, while the second retains only that component. After each edit, we normalize the document embedding and measure the change in cosine similarity with the unchanged query embedding. We report the average score change for direct decomposition, Xetrieval, and the non-overlap active-feature control.

#### C.2 Task-Level Steering Details

We perform task-level feature steering on the top-k features identified by RUS (Eq. 18). For BRIGHT, we set k=256; for ArguAna and NQ, we set k=1024. These values were chosen according to the domain breadth of each dataset.

### D LLM Usage

We used ChatGPT and Gemini as a tool for drafting and refining text. All content produced with the

Reasoning Internalizer 36.0 82.5 42.1 91.1 71.4 79.0 39.2 63.0 CoT Reasoner 42.1 84.7 43.9 91.9 73.1 81.4 41.4 65.5

e5-base

None 31.5 83.3 47.1 90.4 66.8 77.3 34.2 61.5 Reasoning Internalizer 37.9 84.2 46.5 90.3 70.3 81.1 39.2 64.2

e5-large

###### CoT Reasoner 43.8 86.3 47.0 92.8 72.0 82.1 41.3 66.5

None 23.4 77.0 38.5 86.3 60.8 70.4 29.1 55.1 Reasoning Internalizer 29.6 77.3 38.8 87.5 65.0 75.4 35.7 58.5

e5-small

###### CoT Reasoner 36.5 80.8 39.7 91.1 69.5 73.6 38.0 61.3

None 37.0 81.0 28.8 92.2 73.8 77.1 41.7 61.7 Reasoning Internalizer 39.0 80.8 29.6 92.3 74.2 80.2 40.9 62.4

gte-base

###### CoT Reasoner 43.8 83.3 30.3 93.4 74.6 84.0 41.7 64.4

None 41.2 83.0 31.2 92.0 73.6 79.3 41.8 63.2 Reasoning Internalizer 42.3 82.6 31.3 92.0 72.8 81.6 41.3 63.4

gte-large

###### CoT Reasoner 46.1 84.7 32.6 92.8 74.4 84.2 41.4 65.2

None 34.8 48.1 36.2 22.5 64.8 24.1 37.2 38.3 Reasoning Internalizer 38.8 68.9 36.3 64.9 67.9 42.7 38.6 51.2

snowflake

###### CoT Reasoner 44.0 74.2 33.0 77.6 67.4 46.0 40.5 54.7

None 44.5 78.0 40.0 89.7 71.4 83.6 48.2 65.1 Reasoning Internalizer 45.5 77.6 39.0 89.5 70.8 83.6 46.4 64.6

qwen3-0.6b

###### CoT Reasoner 48.8 80.2 40.6 90.7 71.8 83.9 43.7 65.7

None 51.2 84.0 45.2 92.3 74.1 87.0 50.7 69.2 Reasoning Internalizer 51.7 83.5 44.9 91.9 72.8 87.1 49.3 68.7

qwen3-4b

CoT Reasoner 54.8 84.6 45.8 92.9 73.2 86.7 43.8 68.8

- Table 7: Retrieval NDCG@10 (%) scores when dense retriever enhanced by reasoning internalizer and CoT reasoner (both empowered by DeepSeek-V3) under varying configurations (None denotes no enhancement is employed, i.e., the baseline dense retriever).

assistance of LLM was reviewed, revised, and verified by the authors. LLM contributed to wording suggestions and phrasing improvements but did not contribute independently to research ideation, experimental design, or result analysis. The authors take full responsibility for all content in this paper.

Reasoning Internalizer 35.3 82.1 41.7 91.0 71.4 79.3 39.8 62.9 CoT Reasoner 40.5 83.5 40.5 91.8 73.5 80.8 40.0 64.4

e5-base

None 31.5 83.3 47.1 90.4 66.8 77.3 34.2 61.5 Reasoning Internalizer 37.5 83.9 46.7 91.1 71.7 81.7 40.7 64.8

e5-large

###### CoT Reasoner 42.2 85.0 45.0 92.3 72.9 82.4 40.4 65.8

None 23.4 77.0 38.5 86.3 60.8 70.4 29.1 55.1 Reasoning Internalizer 27.4 77.0 38.5 87.8 65.8 75.0 37.3 58.4

e5-small

###### CoT Reasoner 32.1 79.5 37.4 90.0 69.1 77.1 37.2 60.3

None 37.0 81.0 28.8 92.2 73.8 77.1 41.7 61.7 Reasoning Internalizer 39.1 80.9 29.8 91.7 74.1 80.0 41.3 62.4

gte-base

###### CoT Reasoner 41.8 82.0 27.6 93.9 74.0 82.7 41.7 63.4

None 41.2 83.0 31.2 92.0 73.6 79.3 41.8 63.2 Reasoning Internalizer 42.4 82.7 31.3 92.3 73.7 81.7 41.7 63.7

gte-large

###### CoT Reasoner 44.6 83.5 30.6 93.3 74.4 83.9 41.2 64.5

None 34.8 48.1 36.2 22.5 64.8 24.1 37.2 38.3 Reasoning Internalizer 38.3 68.5 37.0 62.2 68.0 40.8 38.7 50.5

snowflake

###### CoT Reasoner 41.7 73.3 29.3 77.7 65.7 43.6 40.5 53.1

None 44.5 77.9 40.0 89.7 71.4 83.6 48.2 65.1 Reasoning Internalizer 45.2 77.6 38.9 89.5 71.3 83.7 45.9 64.6

qwen3-0.6b

CoT Reasoner 47.3 78.6 39.1 91.2 71.3 83.2 43.0 64.8

None 51.2 83.9 45.2 92.3 74.0 87.0 50.7 69.2 Reasoning Internalizer 51.1 83.4 45.2 92.0 73.4 86.9 48.8 68.7

qwen3-4b

CoT Reasoner 53.5 83.3 43.8 92.6 72.9 86.5 44.2 68.1

- Table 8: Retrieval NDCG@10 (%) scores when dense retriever enhanced by reasoning internalizer and CoT reasoner (both empowered by DeepSeek-V2-Lite) under varying configurations (None denotes no enhancement is employed, i.e., the baseline dense retriever).

Reasoning Internalizer 36.3 82.4 41.5 90.6 71.1 79.6 39.4 63.0 CoT Reasoner 44.0 84.3 40.4 93.0 72.2 81.8 41.4 65.3

e5-base

None 31.5 83.3 47.1 90.4 66.8 77.3 34.2 61.5 Reasoning Internalizer 38.2 84.0 46.8 89.9 69.5 81.9 40.9 64.5

e5-large

###### CoT Reasoner 44.9 85.4 44.8 93.0 71.8 83.1 41.3 66.3

None 23.4 77.0 38.5 86.3 60.8 70.4 29.1 55.1 Reasoning Internalizer 30.4 77.0 39.4 88.3 64.6 75.4 35.8 58.7

e5-small

###### CoT Reasoner 38.0 79.9 37.8 91.5 69.0 78.5 38.1 61.8

None 37.0 81.0 28.8 92.2 73.8 77.1 41.7 61.7 Reasoning Internalizer 39.2 80.8 29.8 92.2 73.8 80.3 40.4 62.4

gte-base

CoT Reasoner 44.4 82.5 28.7 93.5 73.1 82.7 41.0 63.7

None 41.2 83.0 31.2 92.0 73.6 79.3 41.8 63.2 Reasoning Internalizer 42.4 82.7 31.7 92.9 73.0 81.1 40.8 63.5

gte-large

###### CoT Reasoner 47.0 83.8 30.7 93.4 73.7 83.7 40.8 64.7

None 34.8 48.1 36.2 22.5 64.8 24.1 37.2 38.3 Reasoning Internalizer 39.2 69.4 36.9 67.0 67.8 43.2 38.8 51.8

snowflake

###### CoT Reasoner 45.4 73.5 32.0 80.6 66.0 51.7 39.9 55.6

None 44.5 77.9 40.0 89.7 71.4 83.6 48.2 65.1 Reasoning Internalizer 46.1 77.7 38.8 90.3 71.2 83.9 47.3 65.0

qwen3-0.6b

###### CoT Reasoner 50.2 79.4 38.5 91.6 71.9 84.4 44.6 65.8

None 51.2 84.0 45.2 92.3 74.1 87.0 50.7 69.2 Reasoning Internalizer 52.8 83.7 44.0 92.3 73.1 87.0 49.6 68.9

qwen3-4b

CoT Reasoner 55.8 84.3 43.5 93.7 72.3 87.2 44.5 68.8

- Table 9: Retrieval NDCG@10 (%) scores when dense retriever enhanced by reasoning internalizer and CoT reasoner (both empowered by GPT-OSS-120B) under varying configurations (None denotes no enhancement is employed, i.e., the baseline dense retriever).

Reasoning Internalizer 36.2 82.4 42.0 90.4 71.5 79.6 39.1 63.0 CoT Reasoner 43.3 83.3 41.1 92.0 72.5 81.3 40.5 64.9

e5-base

None 31.5 83.3 47.1 90.4 66.8 77.3 34.2 61.5 Reasoning Internalizer 38.1 84.2 47.5 91.0 70.3 81.3 40.6 64.7

e5-large

###### CoT Reasoner 44.8 84.7 45.1 93.2 71.7 82.6 41.0 66.1

None 23.4 77.0 38.5 86.3 60.8 70.4 29.1 55.1 Reasoning Internalizer 30.4 77.2 39.0 88.4 64.8 75.3 34.8 58.6

e5-small

###### CoT Reasoner 38.1 79.3 37.1 91.6 66.1 78.0 38.2 61.2

None 37.0 81.0 28.8 92.2 73.8 77.1 41.7 61.7 Reasoning Internalizer 39.3 80.7 29.8 92.0 73.5 79.9 40.4 62.2

gte-base

CoT Reasoner 44.4 81.9 25.2 93.6 72.8 82.9 40.2 63.0

None 41.2 83.0 31.2 92.0 73.6 79.3 41.8 63.2 Reasoning Internalizer 42.6 82.7 31.7 92.5 73.3 81.2 40.6 63.5

gte-large

CoT Reasoner 47.4 83.1 28.0 93.5 73.4 83.8 39.5 64.1

None 34.8 48.1 36.2 22.5 64.8 24.1 37.2 38.3 Reasoning Internalizer 39.6 68.2 36.4 66.2 68.0 43.2 38.5 51.4

snowflake

###### CoT Reasoner 45.7 71.2 31.7 80.6 63.1 48.4 39.0 54.2

None 44.5 77.9 40.0 89.7 71.4 83.6 48.2 65.1 Reasoning Internalizer 45.9 77.8 39.5 89.9 70.8 83.5 47.3 65.0

qwen3-0.6b

CoT Reasoner 50.0 78.7 36.9 90.9 71.0 84.3 44.0 65.1

None 51.2 84.0 45.2 92.3 74.1 87.0 50.7 69.2 Reasoning Internalizer 52.0 83.7 44.3 92.4 73.3 87.3 49.7 69.0

qwen3-4b

CoT Reasoner 56.1 82.9 41.3 92.9 72.6 86.9 42.7 67.9

- Table 10: Retrieval NDCG@10 (%) scores when dense retriever enhanced by reasoning internalizer and CoT reasoner (both empowered by GPT-OSS-20B) under varying configurations (None denotes no enhancement is employed, i.e., the baseline dense retriever).

###### Retriever Enhancement BRIGHT NQ Mutual Trec. Signal1m Robust04 ArguAna Avg.

None 30.8 81.9 40.3 90.3 68.1 76.5 32.9 60.1 Reasoning Internalizer 36.1 82.3 41.4 90.5 71.4 78.9 39.6 62.9

e5-base

###### CoT Reasoner 41.6 82.8 41.9 92.9 73.3 81.7 40.4 65.0

None 31.5 83.3 47.1 90.4 66.8 77.3 34.2 61.5 Reasoning Internalizer 38.4 84.0 46.0 91.5 71.0 81.4 40.2 64.6

e5-large

###### CoT Reasoner 41.2 84.5 45.4 93.5 72.1 83.2 39.9 65.7

None 23.4 77.0 38.5 86.3 60.8 70.4 29.1 55.1 Reasoning Internalizer 29.4 77.1 38.8 87.0 64.7 75.1 35.4 58.2

e5-small

###### CoT Reasoner 30.8 78.8 37.5 90.5 68.4 78.1 36.2 60.0

None 37.0 81.0 28.8 92.2 73.8 77.1 41.7 61.7 Reasoning Internalizer 39.0 80.8 29.7 91.9 73.5 80.2 40.8 62.3

gte-base

CoT Reasoner 43.6 81.4 28.4 93.5 73.4 83.4 40.9 63.5

None 41.2 83.0 31.2 92.0 73.6 79.3 41.8 63.2 Reasoning Internalizer 42.5 82.8 31.5 92.7 73.2 81.7 41.3 63.7

gte-large

CoT Reasoner 45.4 82.8 30.8 93.0 73.8 84.1 39.3 64.2

None 34.8 48.1 36.2 22.5 64.8 24.1 37.2 38.3 Reasoning Internalizer 39.1 69.5 36.4 65.9 68.2 44.1 38.7 51.7

snowflake

CoT Reasoner 43.9 71.9 31.5 79.6 66.5 48.4 38.5 54.3

None 44.5 77.9 40.0 89.7 71.4 83.6 48.2 65.1 Reasoning Internalizer 45.1 77.7 38.9 90.2 71.0 83.5 46.3 64.7

qwen3-0.6b

CoT Reasoner 48.2 77.9 39.7 90.6 71.4 83.6 43.0 64.9

None 51.2 84.0 45.2 92.3 74.1 87.0 50.7 69.2 Reasoning Internalizer 51.6 83.5 44.8 91.7 73.4 86.8 48.9 68.7

qwen3-4b

CoT Reasoner 54.8 82.7 44.2 93.6 72.9 86.7 43.1 68.3

- Table 11: Retrieval NDCG@10 (%) scores when dense retriever enhanced by reasoning internalizer and CoT reasoner (both empowered by Qwen3-32B) under varying configurations (None denotes no enhancement is employed, i.e., the baseline dense retriever).

Prompt Content Explain You are a meticulous AI researcher conducting an important investigation into

patterns found in language. Your task is to analyze text and provide an interpretation that thoroughly encapsulates possible patterns found in it. Guidelines: You will be given a list of text examples on which a certain common pattern might be present. How important each text is for the pattern is listed after each text.

- - Try to produce a concise final description. Simply describe the text latents that are common in the examples, and what patterns you found.
- - If the examples are uninformative, you don’t need to mention them. Don’t focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- - Based on the found patterns, summarize your interpretation in 1–8 words.
- - Do not make lists of possible interpretations. Keep your interpretations short and concise.
- - The last line of your response must be the formatted interpretation, using [EXPLANATION]:

Evaluate You are an intelligent and meticulous linguistics researcher. You will be given a latent explanation (a hypothesis) that describes a sentence-level concept. You will then be given several full text examples (each is a whole sentence/document). Your task is to determine which examples possess the latent implied by the explanation.

Table 12: Prompts used for features explanation and evaluation.

###### Textual Snippets Activated Features

[Query] Let ABC be a triangle inscribed in circle ω. Let the tangents to ω at B and C intersect at point D, and let AD intersect ω at P. If AB = 5, BC = 9,

SAE [F4783] Technical explanations with code examples and mathematical reasoning [F5773] Step-by-step guides for technical tasks [F2344] Technical explanations of computational concepts [F2905] Explanations of fundamental concepts with clarifications and equations [F3341] Code explanation with problem-solving logic and algorithmic steps

and AC = 10, AP can be written as the form mn , where m and n are relatively prime integers. Find

m + n.

[Document] Circles ω1 and ω2 intersect at points X and Y . Line ℓ is tangent to ω1 and ω2 at A and B, respectively, with line AB closer to point X than to

SAE [F2530] Simple, dictionary-style definitions of slang/idioms [F6936] Analytical explanations of technical topics [F3549] Technical guides [F6930] Software and programming tutorials [F5773] Step-by-step guides for technical tasks ←

- Y . Circle ω passes through A and B intersecting ω1 again at D ̸= A and intersecting ω2 again at C ̸= B. The three points C, Y , D are collinear, XC = 67, XY = 47, and XD = 37. Find AB2. Let
- Z = XY ∩ AB. By the radical axis theorem AD, XY, BC are concurrent, say at P. Moreover,

Xetrieval [F4783] Technical explanations with code examples and mathematical reasoning ← [F2344] Technical explanations of computational concepts ← [F2905] Explanations of fundamental concepts with clarifications and equations ← [F574] Technical concept explanations with clarifications and examples [F3341] Code explanation with problem-solving logic and algorithmic steps ←

△DXP ∼ △PXC by simple angle chasing. Let y = PX, x = XZ. Then ... Now, ... Solving, we get 1 4AB2 = 12(y − 47) · 12(y + 47) =⇒ AB2 = 37 · 67 − 472 =

|270|
|---|

Table 13: Case study: Geometric problem-solving.

Textual Snippets Activated Features

[Query] Let u and v be integers satisfying 0 < v < u. Let A = (u, v), let B be the reflection of A across the line y = x, let C be the reflection of B across the y-axis... The area of pentagon ABCDE is 451. Find u + v.

SAE [F3089] Computational geometry and discrete math explanations [F8048] Algorithmic approaches to computational problems [F4783] Technical explanations with mathematical reasoning [F4347] Systematic problem-solving with step-by-step logic [F5344] Math education: geometric series formulas, derivations, and applications

[Document] In △PQR, PR = 15, QR = 20, and PQ = 25. Points A and B lie on PQ, points C and D lie on QR, and points E and F lie on PR, with PA = QB = QC = RD = RE = PF = 5. Find

SAE [F4564] Technical guides [F4230] Technical explanations of probability and statistics [F7064] Analytical explanations of technical topics [F24] Programming concepts and data structures [F4347] Systematic problem-solving with step-by-step logic ←

the area of hexagon ABCDEF. Let R be the origin. Noticing that the triangle is a 3-4-5 right triangle, we can see that A = (4, 12), B = (16, 3), C =

Xetrieval [F4564] Technical guides [F8048] Algorithmic approaches to computational problems ← [F4783] Technical explanations with mathematical reasoning ← [F4347] Systematic problem-solving with step-by-step logic ← [F5344] Math education: geometric series formulas, derivations, and applications ←

(15, 0), D = (5, 0), E = (0, 5), and F = (0, 10). Using the shoelace theorem, the area is

|120|
|---|

.

Shoelace theorem:Suppose the polygon P has vertices (a1, b1), (a2, b2), ... , (an, bn), listed in clockwise order. Then ...

Table 14: Case study: Coordinate geometry and algorithmic reasoning.

###### Textual Snippets Activated Features

[Query] A question on Marx’ “Value, price and profit”: In his lecture, Karl Marx argues that profit is made by capitalists by selling commodities for their real price, paying workers the real value of commodities they produce but letting them work more time than needed...

SAE [F5370] Economic theory explanations with conceptual frameworks [F2690] Philosophical debates on abstract concepts [F4807] Explanations of complex ideas with clear reasoning [F6660] Government policies and their impact on economic systems and individual rights [F1552] Detailed explanations of concepts

[Document] It is the employing capitalist who immediately extracts from the labourer this surplus value, whatever part of it he may ultimately be able to keep for himself. Upon this relation, therefore between the employing capitalist and the wages labourer the whole wages system and the whole present system of production hinge. Some of the citizens who took part in our debate were, there, wrong in trying to mince matters, and to treat this fundamental relation between the employing capitalist and the working man as a secondary question, although they were right in stating that, under given circumstances, a rise of prices might affect in very unequal degrees the employing capitalist, the landlord ...

SAE [F4959] Simple definitions of words with dictionary references [F6605] Database system explanations [F743] Minimalist text with direct statements [F5370] Economic theory explanations with conceptual frameworks ← [F1552] Detailed explanations of concepts ←

Xetrieval [F5370] Economic theory explanations with conceptual frameworks ← [F2690] Philosophical debates on abstract concepts ← [F4807] Explanations of complex ideas with clear reasoning ← [F6660] Government policies and their impact on economic systems and individual rights ← [F4959] Simple definitions of words with dictionary references

Table 15: Case study: Economic and philosophical reasoning.

Textual Snippets Activated Features

[Query] Custom hardware interface type: I would like to write a controller that needs all joint states to update a single joint. My idea was to create a class MyStateInterface which inherits from hardware_interface::StateInterface... I want to know if there is a way to pass a class full of control data...

SAE [F6031] Step-by-step tech how-to guides with specific instructions and links [F6575] Technical explanations of computer science concepts with practical examples [F4097] Technical process descriptions with step-by-step explanations [F2676] Code reviews with technical feedback and suggestions [F5532] SEO/robotics troubleshooting advice with tool recommendations

[Document] There was a problem hiding this comment... Choose a reason for hiding this comment. The reason will be displayed to describe this comment to others. [Learn more] Suggested change: virtual std::vector<InterfaceDescription> export_state_interface_description()... Sorry, something went wrong. All reactions 7 hidden conversations. Load more...

SAE [F2519] Address data handling in software development [F6031] Step-by-step tech how-to guides with specific instructions and links ← [F7229] Textbook-style explanations with question-answer format [F7495] Explanations of complex concepts with clear examples and logical flow [F2676] Code reviews with technical feedback and suggestions ←

Xetrieval [F6031] Step-by-step tech how-to guides with specific instructions and links ← [F6575] Technical explanations of computer science concepts with practical examples ← [F4097] Technical process descriptions with step-by-step explanations ← [F2676] Code reviews with technical feedback and suggestions ← [F5532] SEO/robotics troubleshooting advice with tool recommendations ←

Table 16: Case study: Hardware interface programming and system design.

