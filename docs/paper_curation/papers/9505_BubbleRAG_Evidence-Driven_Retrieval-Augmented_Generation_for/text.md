# BubbleRAG: Evidence-Driven Retrieval-Augmented Generation for Black-Box Knowledge Graphs

Duyi Pan

HKUST (GZ) China dpan457@connect.hkust-gz.edu.cn

Tianao Lou

HKUST (GZ) China tianaolou@gmail.com

Xin Li

HKUST (GZ) China xli420@connect.hkust-gz.edu.cn

Haoze Song

HKUST (GZ) China hsong492@connect.hkust-gz.edu.cn

Yiwen Wu

HKUST (GZ) China ywu240@connect.hkust-gz.edu.cn

Mengyi Deng

HKUST (GZ) China mdeng974@connect.hkust-gz.edu.cn

Mingyu Yang

HKUST (GZ) & HKUST China myang250@connect.hkust-gz.edu.cn

Wei Wang

HKUST (GZ) & HKUST China weiwcs@ust.hk

## Abstract

Large Language Models(LLMs) exhibit hallucinations in knowledgeintensive tasks. Graph-based retrieval augmented generation (RAG) has emerged as a promising solution, yet existing approaches suffer from fundamental recall and precision limitations when operating over black-box knowledge graphs—graphs whose schema and structure are unknown in advance. We identify three core challenges that cause recall loss (semantic instantiation uncertainty and structural path uncertainty) and precision loss (evidential comparison uncertainty). To address these challenges, we formalize the retrieval task as the Optimal Informative Subgraph Retrieval (OISR) problem—a variant of Group Steiner Tree—and prove it to be NP-hard and APX-hard. We propose BubbleRAG, a training-free pipeline that systematically optimizes for both recall and precision through semantic anchor grouping, heuristic bubble expansion to discover candidate evidence graphs (CEGs), composite ranking, and reasoning-aware expansion. Experiments on multi-hop QA benchmarks demonstrate that BubbleRAG achieves state-of-the-art results, outperforming strong baselines in both F1 and accuracy while remaining plug-and-play.

## Keywords

Large Language Models, Retrieval-Augmented Generation, Knowledge Graphs

## 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing and reasoning across diverse domains[24, 31, 43]. Despite these advancements, LLM inference remains prone to hallucinations, and the static nature of their training data often leads to outdated knowledge[12, 16, 23, 30, 36]. To mitigate these limitations, Retrieval-Augmented Generation (RAG) has emerged as a critical paradigm, augmenting generation by retrieving external information at inference time[4, 5, 17].

For complex queries that span multiple documents or require multi-step reasoning chains, graph-structured retrieval is especially

beneficial. Knowledge Graphs (KGs) provide a fixcomplementary form of external knowledge that alleviates key limitations of textcentric approaches[10, 13]. By organizing information into entities, relations, and constraints, KGs make cross-document dependencies explicit and support structured evidence composition and symbolic reasoning[2, 20, 40, 45]. The retrieval component of such a KG-RAG system must achieve two goals simultaneously: high recall, which retrievesall evidence relevant tothe query, andhigh precision, which avoids the inclusion of noisy or irrelevant context that degrades generation quality.

However, achieving both high recall and high precision in graphbased RAG is fundamentally difficult when the retriever operates over a black-box KG, one whose schema, entity types, and relational structure are unknown in advance. We identify three core challenges: (1) Semantic instantiation uncertainty. A query concept may be materialized in multiple heterogeneous forms, including explicit labels, aliases, attribute values, or even implicit relational patterns, so a query token does not uniquely determine where and how the concept appears in the graph. Consequently, retrieval can fail before any reasoning begins due to misalignment between query semantics and KG realizations. This directly causes recall loss: relevant entities are missed because the retriever cannot locate them under their actual surface forms. (2) Structural path uncertainty. Even when relevant entities are found, the retriever still lacks the schema knowledge needed to identify informative relational connections for the query. The same high-level relation may correspond to direct edges, multi-hop chains, or composite structures, making fixed-hop traversal or predefined patterns brittle across graphs. This again causes recall loss: relevant relational chains are missed because the retriever applies the wrong traversal strategy. (3) Evidential comparison uncertainty. When multiple candidates satisfy the constraints, the KG rarely encodes high-level notions such as expertise or importance explicitly. The retriever must instead aggregate implicit signals (e.g., publications, affiliations, citations) to support query-specific ranking, turning retrieval from graph matching into multi-factor evidence aggregation and

Useful Paths Not useful Paths

- DOMAIN1 - MEDICAL Myeloid Leukemia

Heart Disease

Machine Learning Logistic Regression

- DOMAIN2 - CS/AI

Q:Find a ML expert

publish about

is a researcher

...

focus on lab of

works at related to

ML

related to

related to

ML

Expert

unrelated

cowoker of

like

The algorithm used to develop GPT

UNCERTAIN

Many more possible paths...

...

......

OTHER DOMAIN ...

(b) Structural Path Uncertainty

(a) Semantic Instantiation Uncertainty

STRONG EVIDENCE

Li is a leader of XXAl Affiliation: XXAl Citations: 5000

John is a researcher in OpenAl Affiliation: OpenAl l Citations: 8000

Wang published ML paper on ICLR Venue: lCLR l Citations: 500

Zhang read two papers about ML published on ICML

Prof. Hu's daughter interested in SVM Interest only

WEAK EVIDENCE

... ...

(c) Evidential Comparison uncertainty

Figure 1: Three challenges in black-box Knowledge Graph retrieval that limit recall and precision: (a) Semantic instantiation uncertainty when grounding query concepts into KG entities, risking recall loss; (b) Structural path uncertainty when determining relevant relational chains, risking recall loss; and (c) Evidential comparison uncertainty when ranking candidates based on implicit evidence, risking precision loss

comparative inference. This causes precision loss: without discriminative signals, irrelevant or lower-quality candidates are ranked alongside or above the true evidence.

Together, the first two challenges cause the retriever to miss relevant evidence, while the third prevents it from distinguishing the most informative evidence among candidates, jointly constraining both the recall and precision of KG retrieval.

Example 1. As illustrated in Fig. 1, consider the query find an ML expert. For the keyword ML in a black-box KG, it could be mapped to the Medical domain or the CS/AI domain, since we do not know the KG’s detailed schema. Even if we identify the right domain, it could still be mapped to different concepts, such as Logistic Regression or The algorithm used to develop GPT. The latter is harder to locate and may require reasoning. However, they are all concepts that represent ML. Even if we find the correct concept, we still need to determine how ML is connected to Expert. There could be many ways to do this within our KG, and it is hard to know the optimal linking path in advance in a black-box setting. Consequently, after finding numerous possible concepts and the linking ways between them, some will be useful while others are not. To truly find an expert, the system must assess the level of their expertise and compare them against other candidates. In this example, the first two uncertainties risk missing the correct expert entirely (recall failure), while the third risks ranking a less-qualified candidate above the true expert (precision failure).

PriorSolutions.Priorworktacklestheblack-boxdifficulty through four main paradigms, yet each suffers from fundamental recall or precision limitations. (a) Schema-translation methods[1, 22, 25] generate ideal subgraph patterns or decompose queries into preconceived triple structures for alignment. However, the generated patterns can only cover linking structures that the LLM already knows about, leading to limited recall when the actual graph topology diverges from the LLM’s prior assumptions, particularly for distant or implicit connections. (b) Random-walk methods[7, 8, 33, 46] utilize algorithms like Personalized PageRank (PPR)[26] from semantic anchors to estimate node relevance. Yet they inherently suffer from hub bias, where high-degree nodes absorb disproportionate probability mass regardless of query relevance, and they can be inefficient when the evidence subgraph is large or spans multipleloosely connectedcomponents. (c)Iterative multi-hopmethods[19, 21, 29, 35, 38, 42] expand from single anchor nodes via beam search or constrained breadth-first search. They are highly sensitive to initial anchor selection, and a single misalignment can cause cascading retrieval failures across the reasoning chain. In addition, they typically operate only over vertex-centric expansion, missing evidence encoded in edge labels or relational predicates. (d) Preindexed structure methods[3, 18, 28] build hierarchical indices or community structures during offline pre-processing. These static, query-agnostic structures are expensive to construct and maintain, domain-dependent, and ill-suited for handling the diverse requirements of dynamic queries[44].

None of these paradigms jointly optimizes for recall and precision within a unified retrieval objective: methods in (a)–(c) focus on finding some relevant subgraph without systematic comparison, while (d) provides global context at the cost of query specificity.

Our Solution. In this paper, we propose BubbleRAG, a searchbased and training-free retrieval framework for black-box knowledge graphs that explicitly optimizes for both recall and precision. To achieve high recall, BubbleRAG models the retrieval task as finding candidate evidence graphs (CEGs), which are connected subgraphs that cover groups of semantic anchors derived from the query. Each query concept is mapped to a group of candidate anchor nodes or edges in the KG, tolerating the heterogeneous realizations typical of black-box graphs. We formalize the task of selecting the most informative CEG as the Optimal Informative Subgraph Retrieval (OISR) problem, a variant of Group Steiner Tree, and design a heuristic Bubble Expansion algorithm to efficiently enumerate high-quality CEGs. To achieve high precision, BubbleRAG ranks the discovered CEGs by a composite score that balances semantic relevance and structural completeness, then applies controlled LLM-guided expansion to the top-ranked candidates to refine their evidence coverage. Unlike existing methods that treat semantic grounding, structural discovery, and evidence ranking as separable components, BubbleRAG unifies them under a single optimizationdriven pipeline, directly addressing the coupled challenges of blackbox KG retrieval.

Contribution. We summarize our main contributions as follows:

Problem formulation. We formalize KG retrieval as the OISR problem, a variant of Group Steiner Tree that captures the dual goals of semantic coverage (recall) and information density (precision).

We prove that OISR is NP-hard and APX-hard, establishing the theoretical foundations for our heuristic approach.

Framework. We introduce BubbleRAG, a novel search-based and training-free graph RAG pipeline for black-box knowledge graphs. It systematically addresses recall through semantic anchor grouping and heuristic bubble expansion, and precision through candidate evidence graph ranking and reasoning-aware expansion.

Plug-and-play Efficiency. BubbleRAG is computationally practical and plug-and-play. It requires no retriever fine-tuning or modifications to the underlying KG structure. Its localized subgraph construction and anisotropic search ensure that the retrieval complexity remains largely independent of the global graph size, making it easily scalable to massive knowledge graphs.

Experiment. Comprehensive experiments on complex multi-hop QA benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA) demonstrate that BubbleRAG achieves state-of-the-art results. It consistently outperforms strong structure-aware baselines in both F1 score and accuracy, with the largest gains on the most challenging benchmark (MuSiQue).

- 2 Preliminaries

- 2.1 Graph-Based RAG

A graph-basedRAGsystemtypically consists of three stages: (i)graph

indexing, (ii) graph retrieval, and (iii) answer generation. In the indexing stage, input documents are segmented into text chunks, and an LLM extracts triples to construct a knowledge graph 𝐺 = (𝑉,𝐸); the original chunks are typically retained as an additional retrieval source. In the retrieval stage, given a user query 𝑞, the system searches the graph to collect a subgraph 𝐺∗ ⊆ 𝐺 that contains the evidence needed to answer 𝑞. In the generation stage, the LLM produces an answer conditioned on the retrieved subgraph and, optionally, the associated text chunks for grounding.

Crucially, entities and relations extracted from different chunks, potentially from different source documents, are linked into a single graph, enabling cross-document reasoning that text-chunk retrieval alone cannot support. This distinguishes graph-based RAG from standard vector-similarity retrieval (NaiveRAG): the retriever must respect and exploit the graph’s relational structure, not merely rank isolated text passages by embedding proximity.

- 2.2 Black-Box Knowledge Graphs

We call a knowledge graph black-box if its schema, the set of entity types, relation types, and their constraints, is not provided to the retrieval system. The retriever can only access the graph’s topology and the textual content associated with nodes and edges, without knowing, for instance, which relation types connect which entity types.

This is the common case in practice. LLM-extracted KGs from heterogeneous corpora have no standardized schema; the same real-world relation may appear under different labels across documents. Real-world KGs such as Wikidata have complex, evolving schemas that are difficult to expose fully to a downstream retriever. Enterprise KGs may be proprietary or partially documented. Without schema knowledge, the retriever cannot rely on predefined

Table 1: Summary of Notations Notation Description 𝐺 = (𝑉,𝐸) Knowledge graph with nodes 𝑉, edges 𝐸 val(·) Value function for nodes and edges 𝑞 Input user query 𝑡𝑖 Keyword extracted from query 𝑞 S = {𝑆1, . . .,𝑆𝑚}

Semantic anchor groups

𝑤𝑖 Importance weight of group 𝑆𝑖 𝐺∗ = (𝑉∗,𝐸∗) Optimal informative subgraph (target) 𝐺′ = (𝑉′,𝐸′) Localized search space subgraph ℎ Hop threshold for subgraph extraction 𝐵 Search budget for bubble expansion Costsem(𝑇 ) Semantic dissonance cost of CEG 𝑇

𝑟miss Missing mass (uncovered group weights) 𝛼 Penalty factor for structural completeness

Penaltymiss(𝑇 ) Structural incompleteness penalty 𝑛 Number of top CEGs selected 𝑑 Max depth for multi-hop expansion 𝐺final Unified evidence graph merging CEGs 𝐶text Text chunks linked with evidence graph

traversal patterns, meta-paths, or type constraints; it must reason about the graph structure at query time.

## 2.3 Motivations

Our approach is grounded in two key characteristics of answersupporting subgraphs.

- • Motivation 1 (Semantic Anchoring). An evidence subgraph capable of answering 𝑞 should contain nodes that align explicitly or implicitly with the key concepts in the query. Even when the terminology varies, the relevant subgraph is expected to include semantically corresponding entities or relations that can serve as anchors for retrieval.
- • Motivation 2 (Topological Cohesion). Relevant evidence is not arbitrarily scattered across the graph, but tends to form a connected and compact structure. In particular, the concepts required by 𝑞 are likely linked through relatively tight relational paths, suggesting that the answer-supporting evidence can be captured by discovering a cohesive connecting backbone and then expanding locally as needed.

Example 2. Consider the query “When did Lothair II’s mother die?” Motivation 1 holds because the query concepts, Lothair II, mother, and death date, have corresponding anchors in the KG, even if they appear as “Lothar II,” a child_of relation, or an intermediate family node. Motivation 2 holds because these anchors and the answer entity

are connected through a compact subgraph (e.g., Lothair II −−−−−→mother Gisela −−−→died 860 AD), rather than being scattered across disconnected components.

Together, these motivations suggest a retrieval strategy that (i) identifies candidate anchor nodes for each query concept (Motivation 1), (ii) discovers compact connected subgraphs spanning these anchors (Motivation 2), and (iii) ranks the resulting candidate evidence graphs (CEGs) by their relevance to the query. We formalize the first two requirements in the following definition.

## 2.4 Problem Formulation

Motivations 1–2 motivate a retrieval formulation where the goal is to find a connected subgraph of 𝐺 that (a) covers a set of anchor groups derived from the query concepts, and (b) maximizes information density. Let S = {𝑆1,𝑆2, . . .,𝑆𝑘} be a collection of semantic anchor groups, where each 𝑆𝑖 contains candidate nodes or edges corresponding to the 𝑖-th query concept. Let val(·) be a value function measuring the semantic relevance of a node or edge to the query. We assume 𝐺 is informationally sufficient for 𝑞, i.e., at least one connected subgraph satisfying the coverage constraints below exists.

definition 1 (Optimal Informative Subgraph Retrieval (OISR)).

- • Input: A knowledge graph 𝐺 = (𝑉,𝐸), where each node 𝑣 ∈ 𝑉 and each edge 𝑒 ∈ 𝐸 is associated with a value function val(·), and a collection of anchor sets S = {𝑆1,𝑆2, . . .,𝑆𝑘}, where each 𝑆𝑖 ⊆ (𝑉 ∪ 𝐸).
- • Output: A subgraph 𝐺′ = (𝑉′,𝐸′) such that 𝐺′ ⊆ 𝐺.
- • Constraints:

- – Connectivity: 𝐺′ is a connected graph.
- – Multi-Set Coverage: For each anchor set 𝑆𝑖 ∈ S, the retrieved subgraph must contain at least one element from that

set, i.e., (𝑉′ ∪ 𝐸′) ∩ 𝑆𝑖 ≠ ∅,∀𝑖 ∈ {1, . . .,𝑘}.

- • Objective: Maximize the average value of the nodes and edges in the selected subgraph:

Φ(𝐺′) = ∑︁𝑣∈𝑉′ val(𝑣) + ∑︁𝑒∈𝐸′ val(𝑒)

max

(1)

|𝑉′| + |𝐸′|

𝐺′⊆𝐺

The average-value objective encodes an information density principle: it rewards compact subgraphs whose nodes and edges are all query-relevant, while penalizing subgraphs that achieve coverage through long, irrelevant paths. Total value would favor sprawling subgraphs; minimum value would be overly conservative and brittle to noise. Note that anchor sets may include edges (𝑆𝑖 ⊆ 𝑉 ∪ 𝐸), reflecting our system’s ability to match query concepts against both entities and relations (see §3).

We prove in §3.3 that OISR is NP-hard and APX-hard (Theorems 1–2), motivating the heuristic approach developed in Section 3.

## 3 The BubbleRAG Framework

BubbleRAG consists of an offline data preparation phase followed by four online retrieval stages, as illustrated in Figure 4. The complete pipeline is summarized as follows:

- • Step 1 (Data Preparation, §3.1): Build a knowledge graph from the text corpus with semantically enriched edges, ensuring that both entities and relations carry matchable content for downstream retrieval.
- • Step 2 (Semantic Anchor Grouping, §3.2): Extract keywords from the query, find matching KG nodes/edges, and organize them into weighted anchor groups—serving recall by tolerating aliases and schema variations.
- • Step 3 (CEG Discovery, §3.3): Connect anchor groups via the Bubble Expansion heuristic to enumerate candidate evidence

- graphs (CEGs)—serving recall by discovering connected evidence structures.
- • Step 4 (CEG Ranking, §3.4): Score candidates by semantic relevance and structural completeness—serving precision by filtering out noisy or incomplete candidates.
- • Step5(Reasoning-AwareExpansion, §3.5):ApplyLLM-guided multi-hop expansion to top-ranked CEGs—serving precision by refining evidence coverage for the most promising candidates. We detail each step below.

## 3.1 Data Preparation

Following prior work on LLM-based knowledge graph construction [6, 7], BubbleRAG builds its knowledge graph using a standard pipeline: (1) chunking the corpus into passages, (2) using an LLM to extract triplets from each passage, and (3) indexing the triplets into a graph structure. Crucially, entities appearing in different chunks, potentially from different source documents, are linked into a single graph, enabling cross-document retrieval that textchunk approaches cannot support.

A key distinction of BubbleRAG lies in its edge representation. Rather than treating relations as simple labels between entities, we embed rich text directly into the edges. Specifically, for a standard triplet (𝐴,𝑅,𝐵), the edge connecting node 𝐴 and node 𝐵 stores the combined textual content 𝐴𝑅𝐵. This enrichment enables edge-level semantic matching during anchor grouping (§3.2), allowing the system to match query concepts against both entities and relations, which is essential for handling queries where the key constraint is a relational predicate (e.g., authored by, died in).

## 3.2 Semantic Anchor Grouping

This step operationalizes Motivation 1 (Semantic Anchoring) by ensuring that each query concept is mapped to a group of candidate anchors, tolerating the heterogeneous realizations typical of blackbox KGs. The group-based design directly serves recall: rather than committing to a single best-matching node (which may be wrong), we maintain multiple candidates per concept.

The primary objective of this stage is to extract necessary keywords, which covering both core concepts and relations—from a user query and map them to anchors within the knowledge graph. A semantic anchor group 𝑆𝑖 is a set of KG nodes or edges that are candidate realizations of the 𝑖-th query concept. The collection S = {𝑆1, . . .,𝑆𝑚}, together with importance weights {𝑤1, . . .,𝑤𝑚}, forms the input to the next phase.

Given a query 𝑞, the system first retrieves the top-𝑘 relevant text chunks from the corpus. If these chunks contain sufficient information to directly answer 𝑞, BubbleRAG terminates the graph traversal and outputs the response. If the information is insufficient, these chunks act as contextual evidence for the subsequent graph anchor extraction.

Keyword Extraction and Latent Inference. A naive approach to obtaining keywords relies on Named Entity Recognition (NER), treating extracted surface mentions as direct query signals. However, real-world user queries frequently contain latent information critical for retrieval that is not explicitly stated, such as implied concepts, implicit relations, or underspecified constraints. BubbleRAG

Data Preparation

CEG Ranking

CEG Discovery

Chunking

...

chunk1 chunk2 chunk...

Corpus

Vector Store

CEG1: 0.90, CEG2: 0.83, CEG3: ...

Anchor groups

Triplets Extraction

Top 2 CEGs

(Li, Li is leader of XXAI, XXAI) (A, A relation B, B)

KG Database

... ... ...

Reasoning-Aware Expansion

Q: Find a ML expert

[Figure 1]

Semantic Anchor Grouping

Li, works for an AI company for many years...

ICLR, a famous conference about ML...

Machine Learning is interesting...

...

before after

Top 3 chunks

[Figure 2]

Final Answer

[Figure 3]

ICLR, ML, Machine...

group for "ML"

Candidata CEGs

After reasoning, I think XXX is a ML expert ... Following its work is essential for understanding the future of AI .

Li, expert, ML expert ....

KG Database

group for "expert"

Figure 2: Pipeline of BubbleRAG.

overcomes this by leveraging the reasoning capabilities of LLMs to go beyond literal matching: given a query 𝑞, the LLM extracts not only explicit keywords but also infers implicit yet necessary concepts. This facilitates more accurate mapping to anchors in a black-box knowledge graph, thereby reducing the reliance on long, sparse multi-hop traversals and enabling the system to target the relevant answer subgraph more directly. By inferring latent keywords, the system recovers query concepts that would be invisible to surface-level extraction, preventing recall loss from implicit references.

Example 3. For the query “Find scientific papers authored by the winner of the 1921 Nobel Prize in Physics,” a simple extractor might only identify Nobel Prize, 1921, and scientific papers. BubbleRAG, instead, infers that Albert Einstein is the key anchor concept for the query.

Anchor Specialization. After extracting keywords {𝑡𝑖} from 𝑞, BubbleRAG retrieves a top-𝑘 pool of candidate nodes for each 𝑡𝑖. However, many keywords become underspecified when used as standalone query signals. A generic keyword like mother matches thousands of nodes globally, but only a few are relevant to the specific query context. We prompt the LLM to rewrite each underspecified keyword into a query-conditioned constraint, explicitly binding it to other anchor intents already present in 𝑞. For instance,

instead of retrieving candidates for the generic term mother, BubbleRAG refines the signal into Lothair II’s mother. This forces the similarity search to prioritize locally compatible candidates, improving precision (filtering global false positives) without sacrificing recall (the refined query is strictly more specific, so any true match still qualifies).

Example 4. For the query “When did Lothair II’s mother die?”, the keyword Lothair II typically maps to a tight candidate pool near the intended subject region. Conversely, the standalone keyword mother retrieves a flood of generic “mother-of” relations and maternal entities scattered across the graph. Crucially, many of these top-𝑘 matches fall outside the vicinity of the Lothair II anchor candidates, thereby consuming the limited retrieval budget without improving the likelihood of forming a high-quality anchor group.

Schema Relaxation. While anchor specialization tightens the search, the opposite problem can also arise: a query concept may not match any KG label exactly due to schema differences. Before finalizing the graph query, BubbleRAG uses the text chunks retrieved in the pre-retrieval step as previews of local communities within the knowledge graph. A retrieved chunk often reveals the co-occurrence of multiple keyword intents, providing preliminary evidence that certain anchors are locally connected in a promising region. Guided by this preview, BubbleRAG can strategically relax schema-sensitive conditions to preserve recall: if a chunk confirms

the presence of highly relevant neighbors, a strict structural condition may be relaxed to a necessary (but not sufficient) one. Schema relaxation prevents recall loss caused by rigid label matching in KGs with non-standardized schemas.

Example 5. Consider the query “Find the second-married woman whose son started a war.” A strict search for a relation labeled “sec-

- ond marriage” may fail if the graph merely uses a generic “spouse” label. If a retrieved chunk states “...she married X, and later her son Y launched an invasion...”, BubbleRAG detects the high relevance of this region and allows “second marriage” to be relaxed to “marriage”. Although “marriage” is globally noisy, within this specific community, it is highly likely to be the correct anchor because the required contextual neighbors (son, war) are confirmed to be nearby.

Anchor Grouping and Importance Weighting. Following the retrieval of top-𝑘 candidates, we accumulate an initial pool of potential anchors. Since distinct keywords may map to overlapping concepts, BubbleRAG invokes an LLM to perform semantic anchor grouping, merging candidates that refer to the same underlying intent into cohesive groups, denoted as S = {𝑆1,𝑆2, . . .,𝑆𝑚}. Concretely, the LLM receives the full set of retrieved candidate nodes along with the query, and clusters them into groups based on which query concept they correspond to. Candidates matching the same concept (possibly via different keywords) are merged into

- one group. In real-world knowledge graphs, perfect coverage of all query el-

ements is rarely guaranteed; some concepts may be entirely missing or structurally disconnected. BubbleRAG therefore permits partial alignment while establishing a mechanism to evaluate match quality. The LLM assigns a normalized importance weight 𝑤𝑖 to each group 𝑆𝑖 (∑︁𝑚𝑖=1 𝑤𝑖 = 1). The LLM assigns weights based on centrality to the query’s answering logic: the core subject entity typically receives the highest weight, while peripheral modifiers (e.g., temporal constraints, relational predicates) receive lower weights. For instance, in When did Lothair II’s mother die?, Lothair II is indispensable (high weight ≈ 0.5), mother is a necessary relation (medium weight ≈ 0.3), while death date is the target attribute (lower weight ≈ 0.2). Missing the core subject is catastrophic; missing the temporal modifier may still allow partial retrieval. The weights enable graceful degradation: the ranking step (§3.4) penalizes missing highweight groups severely but tolerates missing low-weight groups, allowing the system to retrieve partially informative subgraphs rather than returning nothing.

- 3.3 Candidate Evidence Graph Discovery Following semantic anchor grouping, the query is represented by

a collection of weighted anchor groups S = {𝑆1,𝑆2, . . .,𝑆𝑚}. To operationalize Motivation 2 (Topological Cohesion), we now seek candidate evidence graphs (CEGs)—connected subgraphs of 𝐺 that cover these anchor groups while maximizing information density. This corresponds to the OISR problem defined in §2.

Problem Hardness. To determine the appropriate algorithmic strategy, we analyze the computational complexity of OISR. We prove that OISR can be reduced from the NP-hard Group Steiner Tree (GST) problem[9, 14].

Theorem 1. The Optimal Informative Subgraph Retrieval problem is NP-hard.

Proof. We define the decision version of OISR as follows: Given a graph 𝐺 = (𝑉,𝐸), a value function val(·), a collection of anchor sets S = {𝑆1, . . .,𝑆𝑘}, and a threshold 𝛼 ∈ R+, does there exist a connected subgraph 𝐺′ = (𝑉′,𝐸′) ⊆ 𝐺 such that:

- (i) (𝑉′ ∪ 𝐸′) ∩𝑆𝑖 ≠ ∅ for all 𝑖 ∈ {1, . . .,𝑘}, meaning all anchor sets are covered;
- (ii) The average value satisfies ∑︁𝑣∈𝑉′ val(𝑣)+∑︁𝑒∈𝐸′ val(𝑒)

|𝑉′|+|𝐸′| ≥ 𝛼.

Now, consider an arbitrary unweighted Group Steiner Tree (GST) instance with graph 𝐺(𝑉,𝐸), node groups S𝐺𝑆𝑇 = {𝑆1, . . .,𝑆𝑘} where𝑆𝑖 ⊆ 𝑉, and a total size limit𝛾. We construct an OISR instance as follows:

- • Use the same graph 𝐺 as the knowledge graph.
- • Construct the anchor sets as S = {𝑆1, . . .,𝑆𝑘}∪{𝑆0}, where 𝑆0 = {𝑣∗} for an arbitrary node 𝑣∗ ∈ 𝑉. (The choice of 𝑣∗ does not affect the reduction’s correctness.)
- • Set val(𝑣∗) = 1, and set val(𝑥) = 0 for all other nodes 𝑥 ∈ 𝑉 \ {𝑣∗} and all edges 𝑒 ∈ 𝐸.
- • Set the threshold 𝛼 = 𝛾1.

Because any feasible subgraph 𝐺′ must cover 𝑆0, it must contain 𝑣∗. Since𝑣∗ is the only element with a non-zero value, the numerator of the objective function is exactly 1. Condition (ii) thus becomes:

1 |𝑉′| + |𝐸′|

1 𝛾

###### =⇒ |𝑉′| + |𝐸′| ≤ 𝛾

≥

Therefore, a Group Steiner tree of size ≤ 𝛾 exists in the GST instance if and only if a feasible subgraph exists in the OISR instance. This establishes a polynomial-time reduction, proving that OISR is NP-hard. □

Theorem 2. The OISR problem is hard to approximate.

Proof. We show that a constant-factor approximation for OISR would imply a constant-factor approximation for GST, contradicting established inapproximability bounds. Using the approximation-

preserving nature of the reduction established above, let 𝑂𝑃𝑇𝐺𝑆𝑇 be the size of the optimal Group Steiner Tree for the given GST instance. Based on our construction, the optimal objective value for the OISR instance is exactly 𝑂𝑃𝑇𝑂𝐼𝑆𝑅 = 𝑂𝑃𝑇1

.

𝐺𝑆𝑇

Suppose there exists a polynomial-time algorithm that achieves a constant approximation ratio 𝜌 > 0 for OISR. Let 𝐴𝐿𝐺𝑂𝐼𝑆𝑅 be the objective value obtained by this algorithm, which satisfies:

𝐴𝐿𝐺𝑂𝐼𝑆𝑅 ≥ 𝜌 · 𝑂𝑃𝑇𝑂𝐼𝑆𝑅 (2)

By the construction of our OISR instance, any retrieved subgraph yielding an objective value of 𝐴𝐿𝐺𝑂𝐼𝑆𝑅 corresponds to a feasible Group Steiner Tree of size 𝐴𝐿𝐺𝐺𝑆𝑇 = 𝐴𝐿𝐺1

. Substituting this relation into the inequality gives:

𝑂𝐼𝑆𝑅

1 𝐴𝐿𝐺𝐺𝑆𝑇 ≥ 𝜌 ·

1 𝑂𝑃𝑇𝐺𝑆𝑇

1 𝜌 · 𝑂𝑃𝑇𝐺𝑆𝑇 (3)

=⇒ 𝐴𝐿𝐺𝐺𝑆𝑇 ≤

Since 𝜌 is a constant, this implies the existence of a polynomialtime algorithm that approximates the Group Steiner Tree problem within a constant factor of 𝜌1 . However, it is a well-established result that GST on general graphs cannot be approximated within a factor of 𝑂(log2−𝜖 𝑘) unless NP ⊆ ZTIME(𝑛polylog(𝑛))[9]. This

contradiction proves that no such constant-factor approximation algorithm can exist for OISR, rendering the problem APX-hard. Since this hardness holds even for the restricted class of binaryvalued instances (val ∈ {0, 1}), it holds a fortiori for general OISR. □

Since no efficient exact or constant-factor approximation algorithm exists, we design a targeted heuristic to generate high-quality CEGs in practice.

Heuristic Solution: Bubble Expansion. We term this heuristic Bubble Expansion because the cost-guided wavefronts expanding from each anchor group resemble inflating bubbles: they grow anisotropically through low-resistance (query-aligned) regions and merge upon collision with bubbles from other groups.

To bridge the OISR objective with a practical graph traversal algorithm, we instantiate the value function as val(𝑣) = cos(𝑧𝑞,𝑧𝑣) and define a node cost:

cost(𝑣) = 1 − val(𝑣) = 1 − cos(𝑧𝑞,𝑧𝑣) (4)

By driving the traversal via the accumulation of this semantic cost, the algorithm operationalizes the information density objective to rapidly discover query-aligned structural skeletons. The search consists of three phases:

- (1) Localized Graph Construction. To constrain the search space, we first extract a localized subgraph𝐺′ = (𝑉′,𝐸′). Starting from

the anchor nodes in each semantic group 𝑆𝑖, we collect their ℎ-hop neighborhoods (where parameter ℎ controls the localization range) and take the union of these subgraphs to form 𝐺′. This step substantially reduces computation by focusing exclusively on regions topologically reachable from the anchors within a limited budget.

- (2) AnisotropicExpansion(Multi-source, Cost-guided Search). Within 𝐺′, we initiate a multi-source, Dijkstra-like expansion from all anchor nodes. Unlike standard Breadth-First Search (BFS) that expands uniformly by hop count, our expansion is anisotropic: it propagates preferentially through nodes with a low accumulated semantic cost. Concretely, for each node, we maintain: (i) the minimum accumulated cost to reach it from a given source group, (ii) a predecessor pointer for path reconstruction, and (iii) a bitmask recording which semantic anchor groups have reached it. The search naturally advances rapidly through low-resistance (query-aligned) regions while stalling in high-resistance (irrelevant) areas.
- (3) Collision Detection and Subgraph Fusion. A critical event occurs when an expansion frontier reaches a node already covered by different semantic anchor groups (i.e., its bitmask indicates multi-group coverage). We treat such a node as a candidate Steiner node (meeting point) and trigger a backtracing procedure: we reconstruct the low-cost paths from this node back to the involved anchor nodes using the predecessor pointers. These paths are subsequently fused into a connected CEG and added to the candidate set for final ranking.

Handling Intra-Group and Failed Connections. To ensure robustness in real-world scenarios, our heuristic accounts for two edge cases:

Algorithm 1: Bubble Expansion Algorithm

Input :Knowledge graph 𝐺 = (𝑉,𝐸), semantic anchor groups S = {𝑆1, . . .,𝑆𝑚}, localization hop ℎ, expansion budget 𝐵 Output:Set of candidate subgraphs C

- 1 // Phase 1: Localized Graph Construction 𝐺′ ← ∅
- 2 for each 𝑆𝑖 ∈ S do

- 3 𝑁𝑖 ← ℎ-hop neighborhood of nodes in 𝑆𝑖
- 4 𝐺′ ← 𝐺′ ∪ 𝑁𝑖

- 5 𝐺′ ← (𝑉′,𝐸′) where𝑉′,𝐸′ are nodes and edges in the union
- 6 // Phase 2: Anisotropic Expansion Initialize priority queue 𝑃𝑄 with all terminals 𝑡 ∈ ⋃︁𝑖 𝑆𝑖
- 7 Initialize 𝑐𝑜𝑠𝑡[𝑡] ← 0, 𝑚𝑎𝑠𝑘[𝑡] ← group bitmask of 𝑡, 𝑝𝑟𝑒𝑑[𝑡] ← ∅
- 8 while 𝑃𝑄 is not empty and |𝐶| < 𝐵 do

- 9 (𝑣,𝑐) ← 𝑃𝑄.𝑝𝑜𝑝() with minimum accumulated cost
- 10 if 𝑐 > 𝑐𝑜𝑠𝑡[𝑣] then

- 11 continue

- 12 for each neighbor 𝑢 of 𝑣 in 𝐺′ do

- 13 𝑛𝑒𝑤_𝑐𝑜𝑠𝑡 ← 𝑐 + 𝑐𝑜𝑠𝑡(𝑢)
- 14 𝑛𝑒𝑤_𝑚𝑎𝑠𝑘 ← 𝑚𝑎𝑠𝑘[𝑣] OR group bitmask of 𝑢
- 15 if 𝑛𝑒𝑤_𝑐𝑜𝑠𝑡 < 𝑐𝑜𝑠𝑡[𝑢] then

- 16 𝑐𝑜𝑠𝑡[𝑢] ← 𝑛𝑒𝑤_𝑐𝑜𝑠𝑡
- 17 𝑚𝑎𝑠𝑘[𝑢] ← 𝑛𝑒𝑤_𝑚𝑎𝑠𝑘
- 18 𝑝𝑟𝑒𝑑[𝑢] ← 𝑣
- 19 𝑃𝑄.𝑝𝑢𝑠ℎ((𝑢,𝑛𝑒𝑤_𝑐𝑜𝑠𝑡))
- 20 // Collision Detection if 𝑛𝑒𝑤_𝑚𝑎𝑠𝑘 covers multiple groups then

- 21 𝑇 ← backtrack paths from 𝑢 to terminals using 𝑝𝑟𝑒𝑑
- 22 𝐶 ← 𝐶 ∪ {𝑇}

- 23 // Phase 3: Fallback Handling if 𝐶 is empty then

- 24 𝐶 ← all terminals

- 25 return C

- • Intra-Group Connections. BubbleRAG permits connections among anchor nodes within the same semantic group. Strong intra-group connectivity often indicates that relevant evidence is highly concentrated locally. We allow such fusions into the candidate set but apply a mild penalty during the ranking phase to encourage inter-group coverage and prevent a single group from dominating the retrieved subgraph.
- • Fallback Mechanism. In exceptionally sparse graphs, anchor nodes may fail to collide within the expansion budget. Instead of returning an empty set, BubbleRAG falls back to the currently identified anchor nodes as a baseline. Although disconnected, this preserves non-empty evidence for subsequent reasoning.

Complexity Analysis. We analyze the computational complexity of the CEG discovery phase across its two main steps:

#### Q: Who played the main villain in the movie where Keanu Reeves and Laurence Fishburne starred together?

### Mapping to

play a role in c = 0.47

Laurence Fishburne c = 0.26

Keanu Reeves c = 0.21

main villain

main villain is antagonist of Agent Smith

* play a role in * c = 0.47

movie John Wick 2 The Matrix

The Matrix c = 0.31

K. Reeves c = 0.35

play a role in c = 0.47

Keanu Reeves Keanu Reeves K. Reeves Laurence Fishburne Laurence Fishburne

John Wick 2 c = 0.36

one of the main role is c = 0.53

|main_villain c = 0.31| |
|---|---|
| | |

|Hugo Weaving c = 0.63|
|---|

is antagonist of c = 0.33

Agent Smith c = 0.22

played_by is a role of c = 0.62

c = 0.52

| | |
|---|---|
|Darth Vader c = 0.53| |

|James Earl Jones c= 0.71|
|---|

|Riccardo Scamarcio c = 0.66|
|---|

Santino D'Antonio c = 0.62

played by c = 0.53

played by c = 0.53

Figure 3: An example of the Candidate Evidence Graph (CEG) generated by Bubble Expansion. Different highlight colors in the query represent different extracted keywords, which are mapped to their corresponding semantic anchor groups in the graph. Elements with red borders indicate the nodes and edges that have been successfully incorporated into a CEG.

- • Localized Graph Construction. Given 𝑚 semantic anchor

groups comprising 𝑛 = ∑︁𝑚𝑖=1 |𝑆𝑖| anchor nodes, we extract the ℎ-hop neighborhood for each anchor. The maximum number of nodes reachable within ℎ hops from a single anchor is bounded by 𝑂(𝑑avgℎ ), where 𝑑avg is the average node degree. Thus, the localized subgraph 𝐺′ = (𝑉′,𝐸′) satisfies |𝑉′| ≤ 𝑛 · 𝑂(𝑑avgℎ ). For sparse LLM-constructed knowledge graphs and a small ℎ, this step runs efficiently in 𝑂(𝑛 · 𝑑avgℎ ) time.

- • Anisotropic Bubble Expansion. The expansion maintains a state for each node 𝑣 ∈ 𝑉′ containing: (i) minimum accumulated costs from each source group, (ii) predecessor pointers, and (iii) a bitmask of size 𝑚. The worst-case time complexity, implemented via a priority queue, is 𝑂(|𝐸′| ·𝑚 ·log |𝑉′|). In practice, the anisotropic nature of the search prunes high-cost paths early, keeping the empirical running time significantly below the worst-case bound. Collision detection and subgraph fusion run in 𝑂(|𝑉′| ·𝑚) time via bitmask overlap checks.
- • Overall Complexity. Combining both steps, the total time complexity is:

𝑂(︁𝑛 · 𝑑avgℎ + |𝐸′| ·𝑚 · log |𝑉′| + |𝑉′| ·𝑚)︁

Crucially, because |𝑉′|l|𝑉 | due to the localized construction, the retrieval complexity per query is largely independent of the global graph size |𝑉 |. This ensures that BubbleRAG scales efficiently to massive knowledge graphs. In our experiments, the localized subgraph 𝐺′ typically contains fewer than 103 nodes even when the full KG has 105+ nodes, confirming this scalability. Moreover, the memory footprint is dominated by the localized

subgraph 𝐺′ and the expansion state, requiring 𝑂(|𝑉′| + |𝐸′| +𝑚 ·

|𝑉′|) space. This remains highly tractable for graphs with millions of nodes, provided the localization parameter ℎ is properly configured.

## 3.4 Candidate Evidence Graph Ranking

A key design choice in BubbleRAG is the decoupling of CEG discovery (§3.3) from CEG ranking. The discovery phase generates candidates using a fast graph-based heuristic with a simple cost function. The ranking phase then re-evaluates these candidates using a richer criterion that considers both semantic relevance and structural completeness. This decoupling offers two benefits: (a) the heuristic can explore broadly without being slowed by an expensive scoring function, and (b) the scoring function is plug-and-play, meaning it can be replaced or augmented without changing the discovery algorithm.

Bubble Expansion in the previous phase generates a set of CEGs. While these candidates are structurally connected, connectivity alone does not guarantee evidentiary quality: a candidate may bridge anchors via a generic hub, yielding a connected but semantically vacuous structure. Conversely, a high-quality candidate may miss a minor keyword yet still capture the core intent.

We therefore introduce a discriminative ranking mechanism that integrates semantic dissonance and structural incompleteness. To align with intuitive ranking (higher is better), we define the final score as the inverse of a composite cost:

1 Costsem(𝑇) · Penaltymiss(𝑇) + 𝜖

Score(𝑇) =

, (5)

where𝜖 is a small constant to prevent division by zero. The semantic dissonance approximates 1−Φ(𝐺′) (the inverse of the average value in the OISR objective), while the missing-group penalty enforces the coverage constraint with graceful degradation.

Semantic dissonance. This component measures how well the nodes in a CEG align with the query. A high-quality CEG should contain nodes that are all semantically close to the query, not just a few relevant nodes diluted by many irrelevant ones. For a node 𝑣, let cost(𝑣) = 1 − cos(𝑧𝑞,𝑧𝑣). The semantic cost of 𝑇 is:

∑︂

1 |𝑉𝑇 |

Costsem(𝑇) =

cost(𝑣).

𝑣∈𝑉𝑇

We use the mean rather than the sum to ensure size invariance: the discovery phase is search-oriented and favors low-resistance connectivity, whereas ranking compares candidates of potentially different sizes. Averaging prevents information-rich, longer evidence chains from being penalized solely for containing more nodes, as long as those nodes remain query-relevant (low cost).

Structural incompleteness. This component enforces the coverage requirement from the OISR formulation. A CEG that covers all anchor groups is strongly preferred over one that misses a group, especially a high-weight group. Let 𝑤𝑖 be the importance weight

- of group 𝑆𝑖 (e.g., 𝑤Lothair = 0.8, 𝑤Mother = 0.2). The missing mass is 𝑟miss = ∑︁𝑖:𝑆𝑖∩𝑉𝑇=∅ 𝑤𝑖, and the penalty is:

Penaltymiss(𝑇) = 𝑒𝛼·𝑟miss. (6) This implements a weighted tolerance strategy: with a sufficiently large 𝛼, missing a critical group (high 𝑤𝑖) sharply increases the penalty and strongly suppresses the score, whereas missing a peripheral group incurs only a mild penalty. The hyperparameter 𝛼 controls the strictness of the completeness requirement.

Supporting Diverse Query Semantics. The penalty factor 𝛼 provides a principled mechanism to support different query semantics beyond the default balanced setting:

- • AND queries (all concepts required): A large 𝛼 (e.g., 𝛼 ≫ 1) makes 𝑒𝛼·𝑟miss grow sharply whenever any anchor group is uncovered, forcing the ranking to strongly prefer CEGs that cover all query concepts simultaneously, effectively implementing conjunction semantics.
- • OR queries (any concept sufficient): Setting 𝛼 ≈ 0 collapses the penalty to 1 regardless of missing groups, so ranking is

driven purely by Costsem(𝑇). CEGs matching any subset of anchor groups are treated equally in terms of coverage, effectively implementing disjunction semantics.

- • Comparison queries (rank multiple candidates against each other):BubbleRAGnaturallysupports comparison queries through its top-𝑛 CEG selection. When a query asks to compare entities (e.g., Who is more influential: Author A or Author B?), each topranked CEG captures evidence centered on a different candidate. The 𝑛 selected CEGs are passed together to the reasoning-aware expansion and answer generation stages, where the LLM can directly compare evidence subgraphs side by side, with no architectural change required.

This flexibility allows BubbleRAG to handle a broad spectrum of query types by adjusting a single hyperparameter, without modifying the retrieval pipeline.

Based on Score(𝑇), we select the top-𝑛 CEGs (parameter 𝑛 controls the number of selected candidates) as evidentiary foundations for the subsequent reasoning phase.

weight = 0.1 0.1 0.4 0.4

Q: Find the villain actor in a movie involving Keanu Reeves or Laurence Fishburne

###### CEG 1 CEG 2 CEG 3

Keanu Reeves and Laurence Fishburne both starred in The Matrix, whose villain character Agent Smith is portrayed by Hugo Weaving.

Laurence Fishburne appeared in Man of Steel, which features the villain General Zod portrayed by Michael Shannon.

Keanu Reeves starred in John Wick, which features the villain Viggo Tarasov portrayed by Michael Nyqvist.

No miss Miss Laurence Fishburne Miss Keanu Reeves

Score = 1/(0.63* e𝛼*0 ) Score = 1/(0.66* e𝛼*0.4 ) Score = 1/(0.64* e𝛼*0.4 )

Figure 4: An example of Candidate Evidence Graph (CEG) Ranking. Different highlight colors in the query represent different semantic anchor groups, each assigned an importance weight. The final score of a CEG is determined by two components: its average semantic cost and a structural incompleteness penalty derived from missing concept groups. By simply adjusting the hyperparameter 𝛼, the system can dynamically alter the ranking results to support AND operation and OR operation. The algorithm naturally supports compare queries by selecting the top-𝑛 CEGs.

## 3.5 Reasoning-Aware Expansion

The CEGs discovered in §3.3 represent minimal connected backbones spanning the anchor groups. However, the final answer entity may not be within this backbone but in its immediate neighborhood, one or two hops beyond the backbone’s boundary. More generally, the backbone captures the reasoning chain leading to the answer, while the answer itself requires one final inference step. This pattern arises whenever the query asks about a property of an entity that must itself be identified through intermediate reasoning.

For example, consider: Who are the lead actors in the movies co-directed by Director A and Director B? A ranked subgraph may already bridge the two director anchors and identify their collaborative work (e.g., Movie X). Yet the user intent targets the actors of this movie, which are neighbors of Movie X but were not part of the initial anchor set. This pattern is general: the backbone provides the reasoning chain, while expansion uncovers the target property.

This step serves precision by focusing LLM reasoning on the most promising CEGs. Rather than expanding all candidates (expensive), we apply expansion only to the top-𝑛 ranked CEGs, using the LLM’s judgment to select the most relevant neighbors at each hop.

To address this, BubbleRAG performs a controlled, LLM-guided multi-hop expansion. Starting from each selected candidate𝑇∗, we iteratively expand the current frontier up to a maximum depth 𝑑 (parameter 𝑑 controls the reasoning depth). At each hop ℓ ∈ [1,𝑑], we retrieve the immediate neighbors N𝑑(𝑇∗) adjacent to the current frontier, and prompt the LLM with the query and current evidence to select the most promising neighbors. In the example above, once Movie X is identified, the LLM preferentially selects neighbors connected by cast-related relations (e.g., Actor Y), while discarding irrelevant attributes such as release date or production company. The selected nodes/edges are then added to the evidence structure:

𝑇𝑑∗ =𝑇𝑑∗−1 ∪ Selected(N𝑑) .

The expansion continues until the depth limit 𝑑 is reached, or terminates early when the LLM selects no new neighbors (or indicates that the current evidence is sufficient). The expansion is naturally an anytime algorithm: it can be applied sequentially to CEGs in rank order, and stopped at any point when the time budget is exhausted, making BubbleRAG’s precision-refinement phase controllable and cost-aware.

## 3.6 Answer Generation

Uponcompletionofreasoning-aware expansion, BubbleRAG merges the expanded subgraphs from the top-𝑛 candidates into a single Unified Evidence Graph 𝐺final, consolidating redundant nodes and edges that overlap across candidates. We then map nodes in 𝐺final back to their provenance in the source corpus (via the chunk pointers retained during indexing) and retrieve the corresponding original text chunks 𝐶text. The assembled hybrid context, combining (i) structured triples serialized from 𝐺final as a reasoning skeleton and (ii) the associated raw text chunks as descriptive grounding, is fed into the LLM along with the original user query to produce the final answer.

4 Experiments

## 4.1 Experimental Setup

Baselines. To strictly evaluate the performance of BubbleRAG, we selected a diverse set of baselines categorized into four groups: general LLMs and standard retrieval methods, iterative multi-hop methods, random-walk methods, and pre-indexed structure methods.

General LLMs and Standard Retrieval methods. We first evaluate foundational models and standard pipelines. Vanilla LLM relies solely on parametric knowledge to answer queries without external retrieval. LLM + CoT[34] enhances this by utilizing a standardized Chain-of-Thought template to guide step-by-step reasoning before generating a final answer. NaiveRAG[5] represents the standard pipeline, retrieving text chunks based on vector embedding similarity and concatenating them as context for the LLM.

Iterative multi-hop methods. This paradigm constructs explicit reasoning chains across graph topologies. We evaluate ToG (Thinkon-Graph) [29], which performs a constrained beam search over the knowledge graph to discover relational paths connecting query entities to potential answers.

Random-walk methods This category captures structural centrality using network algorithms. We evaluate HippoRAG2[8], which integrates passage retrieval into the Personalized PageRank (PPR) algorithm [26] to compute associative relationships between semantic query terms and document chunks.

Pre-indexed structure methods. We compare against frameworks that rely on pre-built hierarchical or multi-partite structures. RAPTOR [27] constructs a hierarchical tree to cluster and summarize documents, enabling retrieval across different levels of abstraction. ClueRAG builds a static multi-partite graph index to model interactions between questions, documents, and concepts, utilizing the LLM to navigate this structure. LightRAG [6] indexes documents into a comprehensive graph of entities and relations. We assess its

three distinct retrieval modes: LightRAG-L (Local) for low-level entity details, LightRAG-G (Global) for broader conceptual queries, and LightRAG-H (Hybrid), which combines both strategies.

Datasets. We evaluated BubbleRAG on three benchmark datasets designed for multi-hop Question Answering (QA): MuSiQue[32], HotpotQA[39], and 2WikiMultiHopQA [11]. These datasets require aggregating information from multiple documents to derive the correct answer. Following established protocols [7, 32], we used the validation split of each dataset (consisting of 1,000 questions per dataset) for evaluation. The corresponding corpora were preprocessed into chunks to serve as the retrieval source.

Metrics. Consistent with previous research[7, 28], we employ two primary metrics to assess performance. First, we use the F1 Score to measure the lexical overlap between the generated answer and the ground truth, assessing the precision and recall of the exact tokens appearing in the prediction. Second, since multi-hop questions often have valid answers that differ in phrasing (e.g., synonyms or sentence structures), exact matching can be misleading. To address this, we employ LLM-as-a-Judge Accuracy (𝐴𝐶𝐶𝐿). We utilize Qwen3-7B[37] as a judge to evaluate the semantic equivalence between the model’s output and the gold standard, assigning a score of 1 for correct answers and 0 otherwise.

Implementation Details. To ensure a fair comparison across all methods, we use Qwen3-Embedding-8B[41] embedding model for every framework in our experiments. In addition, following common fairness settings in prior work [28], we impose a unified retrieval budget: each method is allowed to observe at most 15 text chunks per query. This constraint prevents performance gains that stem purely from larger contexts and ensures that improvements reflect retrieval quality rather than input size. Unless otherwise specified, all other hyperparameters follow the default configurations recommended in the original implementations.

## 4.2 Main Results

Table 2 presents the comparative performance of BubbleRAG and ten baseline methods across three multi-hop QA benchmarks using both 30B and 8B parameter models. The results demonstrate the consistent superiority of BubbleRAG, which achieves the highest average F1 and LLM-as-a-Judge Accuracy scores across all settings. Specifically, on the 30B model setting, BubbleRAG surpasses the strongest baseline, HippoRAG2, by an average of 2.52% in F1 and 2.23% in accuracy. BubbleRAG’s superior F1 score reflects its balanced optimization of recall (via group-covering expansion) and precision (via discriminative ranking).

The advantage of BubbleRAG is particularly pronounced on the MuSiQue dataset, which requires complex multi-hop reasoning. While other structure-aware baselines like Clue-RAG and LightRAG struggle to surpass Naive RAG on this dataset, BubbleRAG achieves an F1 score of 53.03 with the 30B model, outperforming HippoRAG2 by approximately 8 percentage points. MuSiQue requires 3–4 hop reasoning, where existing methods’ single-anchor or fixed-hop strategies break down. BubbleRAG’s group-aware expansion naturally handles variable-length reasoning chains without

Table 2: Comparison of Different Methods on Multi-hop QA Benchmarks. The best results are highlighted in bold. The rightmost two columns show the average F1 and ACC𝐿 across all datasets.

30B Model 8B Model Average

Method

HotpotQA MuSiQue 2Wiki HotpotQA MuSiQue 2Wiki F1 ACC𝐿 F1 ACC𝐿 F1 ACC𝐿 F1 ACC𝐿 F1 ACC𝐿 F1 ACC𝐿 F1 ACC𝐿

Vanilla LLM 17.84 29.70 5.80 10.40 20.14 30.30 11.41 30.10 5.18 12.90 17.46 29.90 12.97 23.88 CoT + LLM 18.83 33.50 8.42 13.20 18.25 25.50 16.35 30.40 7.11 12.00 18.11 25.50 14.51 23.35 Naive RAG 73.07 80.50 43.68 42.20 52.15 54.80 67.41 75.00 34.53 35.00 50.23 52.80 53.51 56.71

ToG 40.07 48.50 16.76 21.60 35.88 38.60 23.56 35.30 8.39 13.90 20.22 23.10 24.15 30.16 RAPTOR 66.75 68.00 33.10 32.20 40.11 43.90 62.66 68.60 29.35 28.90 36.25 40.20 44.70 46.97

LightRAG (local) 57.05 61.80 41.72 43.80 60.17 64.90 54.25 54.30 36.57 37.70 54.61 55.11 50.73 52.94 LightRAG (global) 35.28 39.10 20.56 22.50 23.20 26.30 30.79 30.80 20.11 20.30 23.41 24.00 25.56 27.17 LightRAG (hybrid) 54.87 59.10 39.57 42.20 55.67 60.10 54.32 54.50 34.21 35.10 47.79 48.53 47.74 49.92

HippoRAG2 71.72 80.20 45.04 47.20 67.65 71.70 73.33 79.00 42.92 41.60 62.40 66.70 60.50 64.40 Clue-RAG 62.87 67.30 33.01 33.90 45.63 47.40 54.95 69.20 27.39 27.80 35.87 46.30 43.29 48.65

BubbleRAG 74.35 82.60 53.03 53.10 72.54 76.60 71.82 79.10 45.15 44.00 64.97 67.60 63.02 66.63

predefined hop limits. Unlike approaches prone to error propagation, the heuristic Steiner Tree search in BubbleRAG ensures semantically grounded and complete reasoning chains.

Furthermore, BubbleRAG maintains its leading position across model scales. Notably, the method achieves an average F1 of 63.02 using the smaller 8B model, which is comparable to, and often better than, the performance of many baselines using larger 30B models. This demonstrates that retrieval quality, not model size, is the primary bottleneck for multi-hop QA: the high-quality context provided by BubbleRAG effectively compensates for the reduced parametric knowledge of smaller models.

## 4.3 Ablation Studies

To validate the contribution of each component in BubbleRAG and analyze the impact of key hyper parameters, we conducted comprehensive ablation studies and sensitivity analyses. All ablation experiments use Qwen3-8B[37] as the backbone LLM to ensure fair comparison.

w/o Anchor Specialization: To evaluate the importance of our query-conditioned constraint refinement, we replaced the LLMbased rewriting module with a direct keyword extraction approach. In this variant, we extract explicit entities from the query using a standard extractor and map them to graph nodes without refining generic terms (e.g., mother) into specific constraints (e.g., Lothair II’s mother). This setting tests the system’s ability to handle ambiguity without query-specific context injection.

w/o Schema Relaxation: To assess the contribution of the chunkguided schema relaxation mechanism, we retained the constraint refinement but removed the retrieval of text chunks prior to anchor selection. In this setting, the system relies strictly on the refined keywords to locate anchors in the knowledge graph, without the

Table 3: Ablation study of BubbleRAG components on the 2Wiki and HotpotQA datasets. We report the F1 score and LLM-as-a-Judge Accuracy (ACC). The Full Method row represents the standard BubbleRAG performance.

2Wiki HotpotQA

Method Variant

F1 ACC F1 ACC BubbleRAG (Full Method) 64.97 67.60 71.82 79.10 w/o Anchor Specialization 60.45 65.00 69.28 76.10 w/o Schema Relaxation 53.62 59.20 66.65 75.30 w/o CEG Ranking 58.76 64.40 70.20 78.90

flexibility to relax schema-sensitive conditions based on local text evidence.

w/o CEG Ranking: To verify the necessity of the discriminative ranking mechanism, we removed the scoring function defined in Eq. (5). Instead of re-evaluating the candidate subgraphs based on semantic dissonance and structural completeness, we simply selected the first valid connected subgraph returned by the heuristic expansion. This variant highlights the role of the ranking module in filtering out structurally connected but semantically irrelevant noise.

Ablation Analysis: The ablation experiments are conducted on the full validation sets of 2Wiki and HotpotQA (1,000 questions each). As shown in Table 3, removing each component leads to measurable performance degradation, validating the importance of our design choices. The most critical component is:

Schema Relaxation, whose removal causes the largest performance drop: F1 decreases by 11.35 points on 2Wiki and 5.17 points on

HotpotQA. This confirms that chunk-guided anchor selection effectively bridges the gap between schema-level graph entities and textual evidence, preventing the system from anchoring on semantically mismatched nodes.

Anchor Specialization also demonstrates significant value, with F1 drops of 4.52 and 2.54 points on 2Wiki and HotpotQA respectively. This indicates that query-conditioned constraint rewriting is essential for resolving entity ambiguity. Without refining generic terms, the system fails to locate the correct anchor nodes in the knowledge graph.

CEG Ranking contributes to performance, though its impact varies across datasets. On 2Wiki, removing it causes a 6.21-point F1 drop, while on HotpotQA the degradation is relatively minor (1.62 points). This suggests that the ranking module is particularly crucial for complex reasoning tasks requiring precise evidence selection, while simpler datasets can tolerate some noise in the retrieved context.

In summary, removing Anchor Specialization and Schema Relaxation primarily reduces recall (the system fails to find correct anchors), while removing CEG Ranking reduces precision (noisy candidates are selected). This validates BubbleRAG’s design of addressing recall and precision through dedicated pipeline stages.

## 4.4 Parameter Sensitivity

We further analyze how the system’s performance and efficiency fluctuate with key hyperparameters: the expansion budget 𝐵, the reasoning depth 𝑑, and the penalty factor 𝛼. The localization hop ℎ is held constant at ℎ = 6 throughout, as preliminary experiments showed this value consistently provides sufficient local coverage for the datasets studied.

Table 4: Sensitivity analysis on the 2Wiki dataset. We report F1 scores and average Inference Latency (Lat.) in seconds. The default settings are 𝐵 = 10,ℎ = 6,𝑑 = 6,𝛼 = 1.0.

##### Parameter Value F1 Score Lat. (s)

5 59.48 18.72 10 60.52 20.99 20 58.32 33.67 50 60.42 44.98

Budget (𝐵)

2 58.25 16.17 4 59.07 19.39 6 60.52 20.99 8 62.28 32.67

Depth (𝑑)

- 0.1 59.11 –
- 1.0 60.52 –
- 2.0 60.37 – 5.0 57.35 –

Penalty (𝛼)

Sensitivity Analysis: The parameter sensitivity analysis is conducted on a randomly sampled subset of 100 queries from 2Wiki. As shown in Table 4, this analysis reveals the impact of key hyperparameters on both performance and efficiency.

Expansion Budget (𝐵): The budget parameter controls the number of candidates explored during bubble expansion. We observe that 𝐵 = 10 achieves an optimal balance, yielding an F1 of 60.52 with a moderate latency of 20.99 seconds. Increasing 𝐵 to 20 or 50 does not yield meaningful performance gains (F1 drops to 58.32 and 60.42 respectively) but significantly increases latency to 33.67s and 44.98s. This indicates that the bubble expansion algorithm efficiently identifies high-quality candidates early, and excessive exploration introduces noise without improving answer quality.

Reasoning Depth (𝑑): As the reasoning depth increases from 2 to 8, we observe consistent F1 improvements: 58.25 → 59.07 → 60.52 → 62.28. However, the latency grows substantially, particularly when 𝑑 = 8 (32.67s). This suggests that deeper reasoning chains are beneficial for complex multi-hop questions but come with a significant computational cost. For practical deployment, 𝑑 = 6 offers the best trade-off between accuracy and speed.

Penalty Factor (𝛼): The penalty factor in Eq (6) controls the strictness of the structural completeness requirement: higher 𝛼 penalizes CEGs that miss anchor groups more severely. We observe that the performance peaks at 𝛼 = 1.0 (yielding an F1 of 60.52) but experiences a noticeable drop when 𝛼 is increased to 5.0 (F1 drops to 57.35). This indicates that while a moderate completeness enforcement is necessary to capture diverse query concepts, an overly strict requirement (𝛼 = 5.0) hurts performance. Enforcing a rigid ANDstyle conjunction can overly penalize highly relevant candidate graphs that simply miss an unimportant anchor.

## 4.5 Efficiency & Cost Analysis

Given that BubbleRAG involves a multi-stage pipeline with LLM interactions (Steps 2 & 5) and heuristic graph algorithms (Steps 3–4), we evaluate its computational overhead against representative baselines. We conducted this analysis on a randomly sampled subset of 100 queries to measure the average build time, inference latency, and token consumption.

Table 5: Efficiency comparison on 100 sampled queries (single A100 GPU). Latency: average seconds per query. Query: total tokens consumed across all 100 queries at inference time. Index: total tokens for offline corpus indexing

##### Method Latency (s) Query Index

Naive RAG 0.67 249,476 HippoRAG2 4.26 418,812 4,575,580 ToG 45.93 765,915 BubbleRAG 20.99 1,064,052 3,840,320

Table 5 presents a comprehensive efficiency comparison. While Naive RAG has the lowest latency (0.67s) due to simple vector retrieval, its performance is significantly inferior. BubbleRAG (20.99s) is substantially faster than ToG (45.93s) while delivering superior accuracy, demonstrating the efficiency of our heuristic bubble expansion algorithm compared to ToG’s iterative LLM-based graph traversal.

Compared to HippoRAG2 (4.26s), BubbleRAG incurs higher latency. The ≈5× latency difference is largely attributable to the

LLM interactions in anchor grouping (Step 2) and reasoning-aware expansion (Step 5). This overhead is justified by its consistent performance gains across all benchmarks. The token consumption analysis reveals that BubbleRAG (1,064,052 tokens) uses more tokens than baselines, which is expected as it encodes graph structure into prompts. However, this cost is offset by the higher answer quality, and the CEG Ranking phase ensures only relevant evidence reaches the generation stage, avoiding wasteful processing of noisy contexts.

Furthermore, BubbleRAG’s index construction cost (3,840,320 tokens) is comparable to HippoRAG2 (4,575,580 tokens) and is amortized across all queries, making it practical for real-world deployments where the knowledge base is queried repeatedly.

- 5 Related Work We review existing approaches for knowledge graph retrieval.

Query Rewriting and Schema-Aligned Matching. Methods in this paradigm, such as SimGRAG [1] and KG-GPT [15], align queries with pre-conceived structural templates. SimGRAG utilizes a two-stage process where LLM transforms a query into an abstract pattern graph for subgraph alignment. Conversely, KG-GPT employs a divide-and-conquer strategy, segmenting complex queries into discrete (head, relation, tail) triple structures to retrieve candidate subgraphs. These methods rely on the LLM’s internal prior knowledge. When generated patterns do not exist in the underlying topology, it results in hallucinations and retrieval gaps.

Iterative Multi-Hop Exploration. This paradigm initiates retrieval at semantic seed nodes and expands outward to construct reasoning chains. ToG [29] implements a constrained beam search directly on the topology, while ToG2 [21] incorporates document embeddings to filter contextually irrelevant nodes during traversal. RoG [19] generates a discrete sequence of relation types as a search plan, executing a constrained breadth-first search to extract grounded paths. These mechanisms are highly sensitive to initial anchor extraction; a single misalignment at the seed node stage causes cascading retrieval failures across the multi-hop chain.

Stochastic Traversal via Random Walks. Methods in this category, such as HippoRAG, HippoRAG2 [7, 8], LinearRAG [46], and AGRAG [33], utilize algorithms like Personalized PageRank (PPR) [26] from semantic anchors to capture structural centrality and associative memory. HippoRAG extracts named entities from the query as seed nodes, executing PPR over an open knowledge graph to distribute probability mass and rank passages. HippoRAG2 extends this by integrating passage nodes directly into the graph topology alongside phrase nodes, assigning PPR reset probabilities based on embedding similarity. LinearRAG constructs a relation-free hierarchical graph of entity, sentence, and passage nodes, applying PPR for global importance aggregation from locally activated seed entities. AGRAG computes PPR to establish node influence scores and combines them with semantic edge costs to extract a Minimum Cost Maximum Influence (MCMI) reasoning subgraph. Because probabilistic walks necessitate a starting point, their retrieval performance is upper-bounded by the accuracy of the initial anchor extraction; erroneous anchors will misguide the subsequent topological exploration.

Structure-Augmented Retrieval with Auxiliary Graphs. Methods like GraphRAG [3], KAG [18], and ClueRAG [28] rely on preprocessing to construct hierarchical indices or community structures. GraphRAG extracts an entity knowledge graph from source documents, applies the Leiden algorithm to partition the graph into a nested hierarchy of modular communities, and uses a language model to generate bottom-up summaries for these partitions. KAG constructs a mutual-index between the graph structure and raw text chunks using a hierarchical framework, applying semantic reasoning during offline indexing to complete concept relations and align fragmented instances. ClueRAG builds a multi-partite graph index comprising three distinct node layers: text chunks, knowledge units, and entities. Because these static, pre-built structures are domain-dependent and constructed independently of user queries, they are poorly suited for the diverse and specific requirements of dynamic real-world queries.

BubbleRAG differs fundamentally from all four paradigms. Unlike query rewriting methods, it does not assume prior schema knowledge or rely on the LLM’s ability to generate structurally valid graph patterns. Unlike iterative multi-hop and random-walk methods, it initializes from groups of anchors rather than a single seed node, avoiding the cascading failure mode inherent in single-anchor strategies. Unlike pre-indexed structure methods, its evidence structures are query-specific and dynamically constructed at retrieval time. Most distinctively, BubbleRAG jointly optimizes for recall (through group-covering bubble expansion) and precision (through discriminative CEG ranking), treating retrieval as a formal optimization problem (OISR) rather than a traversal heuristic.

## 6 Conclusion

We addressed the problem of retrieval from black-box knowledge graphs, where the retriever has no access to the graph’s schema. We formalized this as the Optimal Informative Subgraph Retrieval (OISR) problem, a variant of Group Steiner Tree that captures the dual goals of semantic coverage (recall) and information density (precision), and proved it to be NP-hard and APX-hard. Based on this formalization, we proposed BubbleRAG, a training-free pipeline that systematically addresses recall through semantic anchor grouping and heuristic bubble expansion, and precision through candidate evidence graph ranking and reasoning-aware expansion.

Comprehensiveexperimentsonmulti-hop QA benchmarks demon-

strate that BubbleRAG consistently outperforms state-of-the-art baselines, including HippoRAG2, achieving superior accuracy and F1 scores. Notably, our framework exhibits exceptional robustness, delivering competitive performance even with smaller language models (8B) compared to baselines using larger models (30B). BubbleRAG’s plug-and-play design requires no retriever fine-tuning or modifications to the underlying KG structure, and its localized subgraph construction ensures scalability to massive knowledge graphs.

## References

[1] Yuzheng Cai, Zhenyue Guo, YiWen Pei, WanRui Bian, and Weiguo Zheng. 2025. SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation. In Findings of the Association for Computational Linguistics: ACL 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.).

- [2] Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei Cao, Jie Ma, Daoyu Wang, and Enhong Chen. 2025. A Survey on Knowledge-Oriented Retrieval-Augmented Generation. arXiv:2503.10677 [cs.CL] https://arxiv.org/abs/2503.10677
- [3] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130 (2024).
- [4] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (Barcelona, Spain) (KDD ’24). Association for Computing Machinery, New York, NY, USA, 6491–6501. https://doi.org/10.1145/3637528.3671470
- [5] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL] https://arxiv.org/abs/2312.10997
- [6] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG: Simple and Fast Retrieval-Augmented Generation. In Findings of the Association for Computational Linguistics: EMNLP 2025, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational Linguistics, Suzhou, China, 10746–10761. https://doi.org/10. 18653/v1/2025.findings-emnlp.568
- [7] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. HippoRAG: neurobiologically inspired long-term memory for large language models. In Proceedings of the 38th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’24). Curran Associates Inc., Red Hook, NY, USA, Article 1902, 38 pages.
- [8] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025. From RAG to Memory: Non-Parametric Continual Learning for Large Language Models. In Forty-second International Conference on Machine Learning. https: //openreview.net/forum?id=LWH8yn4HS2
- [9] Eran Halperin and Robert Krauthgamer. 2003. Polylogarithmic inapproximability. In Proceedings of the Thirty-Fifth Annual ACM Symposium on Theory of Computing (San Diego, CA, USA) (STOC ’03). Association for Computing Machinery, New York, NY, USA, 585–594. https://doi.org/10.1145/780542.780628
- [10] Haoyu Han, Li Ma, Yu Wang, Harry Shomer, Yongjia Lei, Zhisheng Qi, Kai Guo, Zhigang Hua, Bo Long, Hui Liu, Charu C. Aggarwal, and Jiliang Tang. 2026. RAG vs. GraphRAG: A Systematic Evaluation and Key Insights. arXiv:2502.11371 [cs.IR] https://arxiv.org/abs/2502.11371
- [11] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In Proceedings of the 28th International Conference on Computational Linguistics, Donia Scott, Nuria Bel, and Chengqing Zong (Eds.). International Committee on Computational Linguistics, Barcelona, Spain (Online), 6609–6625. https://doi.org/10.18653/v1/2020.coling-main.580
- [12] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Transactions on Information Systems 43, 2 (Jan. 2025), 1–55. https://doi.org/10.1145/3703155
- [13] Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S. Yu. 2022. A Survey on Knowledge Graphs: Representation, Acquisition, and Applications. IEEE Transactions on Neural Networks and Learning Systems 33, 2 (Feb. 2022), 494–514. https://doi.org/10.1109/tnnls.2021.3070843
- [14] Richard M Karp. 2009. Reducibility among combinatorial problems. In 50 Years of Integer Programming 1958-2008: from the Early Years to the State-of-the-Art. Springer, 219–241.
- [15] Jiho Kim, Yeonsu Kwon, Yohan Jo, and Edward Choi. 2023. KG-GPT: A General Framework for Reasoning on Knowledge Graphs Using Large Language Models. arXiv:2310.11220 [cs.CL] https://arxiv.org/abs/2310.11220
- [16] Yubin Kim, Hyewon Jeong, Shan Chen, Shuyue Stella Li, Chanwoo Park, Mingyu Lu, Kumail Alhamoud, Jimin Mun, Cristina Grau, Minseok Jung, Rodrigo Gameiro, Lizhou Fan, Eugene Park, Tristan Lin, Joonsik Yoon, Wonjin Yoon, Maarten Sap, Yulia Tsvetkov, Paul Liang, Xuhai Xu, Xin Liu, Chunjong Park, Hyeonhoon Lee, Hae Won Park, Daniel McDuff, Samir Tulebaev, and Cynthia Breazeal. 2025. Medical Hallucinations in Foundation Models and Their Impact on Healthcare. arXiv:2503.05777 [cs.CL] https://arxiv.org/abs/2503.05777
- [17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401 [cs.CL] https://arxiv.org/abs/2005.11401
- [18] Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong Zhao, Mengshu Sun, Zhiqiang Zhang, Jun Zhou, Wenguang Chen, Wen Zhang, and Huajun Chen. 2025. KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation. In Companion Proceedings of the ACM on Web Conference

- 2025 (Sydney NSW, Australia)(WWW ’25).Association for Computing Machinery, New York, NY, USA, 334–343. https://doi.org/10.1145/3701716.3715240
- [19] LINHAO LUO, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=ZGNWW7xZ6Q
- [20] Chuangtao Ma, Yongrui Chen, Tianxing Wu, Arijit Khan, and Haofen Wang.

2025. Large Language Models Meet Knowledge Graphs for Question Answering: Synthesis and Opportunities. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing.

- [21] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian Guo. 2025. Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation. In The Thirteenth International Conference on Learning Representations. https: //openreview.net/forum?id=oFBu7qaZpS
- [22] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query Rewriting in Retrieval-Augmented Large Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 5303–5315. https://doi.org/10.18653/v1/2023.emnlpmain.322
- [23] Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, and Daniel E. Ho. 2024. Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools. arXiv:2405.20362 [cs.CL] https: //arxiv.org/abs/2405.20362
- [24] Andrea Matarazzo and Riccardo Torlone. 2025. A Survey on Large Language Models with some Insights on their Capabilities and Limitations. arXiv:2501.04040 [cs.CL] https://arxiv.org/abs/2501.04040
- [25] Costas Mavromatis, Soji Adeshina, Vassilis N. Ioannidis, Zhen Han, Qi Zhu, Ian Robinson, Bryan Thompson, Huzefa Rangwala, and George Karypis. 2025. BYOKG-RAG: Multi-Strategy Graph Retrieval for Knowledge Graph Question Answering. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational Linguistics, Suzhou, China, 27881–27898. https://doi.org/10.18653/v1/2025.emnlp-main.1417
- [26] Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. 1999. The PageRank citation ranking: Bringing order to the web. Technical Report. Stanford infolab.
- [27] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. arXiv:2401.18059 [cs.CL] https://arxiv.org/abs/2401. 18059
- [28] Yaodong Su, Yixiang Fang, Yingli Zhou, Quanqing Xu, and Chuanhui Yang. 2025. Clue-RAG: Towards Accurate and Cost-Efficient Graph-based RAG via MultiPartite Graph and Query-Driven Iterative Retrieval. arXiv:2507.08445 [cs.IR] https://arxiv.org/abs/2507.08445
- [29] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph. In The Twelfth International Conference on Learning Representations. https: //openreview.net/forum?id=nnVO1PvbTv
- [30] S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha, and Amitava Das. 2024. A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models. arXiv:2401.01313 [cs.CL] https://arxiv.org/abs/2401.01313
- [31] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971 [cs.CL] https://arxiv.org/abs/2302.13971
- [32] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.

2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics 10 (2022), 539–554. https://doi.org/10.1162/tacl_a_00475

- [33] Yubo Wang, Haoyang Li, Fei Teng, and Lei Chen. 2025. AGRAG: Advanced Graphbased Retrieval-Augmented Generation for LLMs. arXiv:2511.05549 [cs.LG] https://arxiv.org/abs/2511.05549
- [34] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv:2201.11903 [cs.CL] https: //arxiv.org/abs/2201.11903
- [35] Xiaojun Wu, Cehao Yang, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Yuanliang Sun, Hui Xiong, Jia Li, and Jian Guo. 2026. Think-on-Graph 3.0: Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dual-Evolving Context Retrieval. https://openreview.net/forum?id=KV9hrBIqA9
- [36] Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. 2025. Hallucination is Inevitable: An Innate Limitation of Large Language Models. arXiv:2401.11817 [cs.CL] https://arxiv.org/abs/2401.11817

- [37] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https://arxiv.org/abs/2505.09388
- [38] Cehao Yang, Xiaojun Wu, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Yuanliang Sun, Jia Li, Hui Xiong, and Jian Guo. 2025. GraphSearch: An Agentic Deep Searching Workflow for Graph Retrieval-Augmented Generation. arXiv:2509.22009 [cs.CL] https://arxiv.org/abs/2509.22009
- [39] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for Computational Linguistics, Brussels, Belgium, 2369–2380. https://doi.org/10. 18653/v1/D18-1259
- [40] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Hao Chen, Yilin Xiao, Chuang Zhou, Junnan Dong, Yi Chang, and Xiao Huang. 2025. A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models. arXiv:2501.13958 [cs.CL] https://arxiv.org/

- abs/2501.13958
- [41] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou.

2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv:2506.05176 [cs.CL] https://arxiv.org/abs/2506.05176

- [42] Zhuocheng Zhang, Yang Feng, and Min Zhang. 2025. LevelRAG: Enhancing Retrieval-Augmented Generation with Multi-hop Logic Planning over Rewriting Augmented Searchers. arXiv:2502.18139 [cs.CL] https://arxiv.org/abs/2502.18139
- [43] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2025. A Survey of Large Language Models. arXiv:2303.18223 [cs.CL] https://arxiv.org/abs/2303.18223
- [44] Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He, Yongwei Zhang, Sicong Liang, Xilin Liu, Yuchi Ma, and Yixiang Fang. 2026. InDepth Analysis of Graph-Based RAG in a Unified Framework. Proc. VLDB Endow. 18, 13 (Jan. 2026), 5623–5637. https://doi.org/10.14778/3773731.3773738
- [45] Zulun Zhu, Tiancheng Huang, Kai Wang, Junda Ye, Xinghe Chen, and Siqiang Luo. 2026. Graph-Based Approaches and Functionalities in Retrieval-Augmented Generation: A Comprehensive Survey. ACM Comput. Surv. (Feb. 2026). https: //doi.org/10.1145/3795880
- [46] Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang, and Xiao Huang. 2025. LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora. arXiv:2510.10114 [cs.CL] https://arxiv.org/abs/2510.10114

