# arXiv:2604.28158v2[cs.AI]1May2026

[Figure 1]

## Intern-Atlas: A Methodological Evolution Graph as Research Infrastructure for AI Scientists

Yujun Wu1,2∗, Dongxu Zhang1,3∗, Xinchen Li1,4∗, Jinhang Xu1,5∗, Yiling Duan6, Yumou Liu1,7, Jiabao Pan1,8, Qiyuan Zhu1, Xuanhe Zhou7, Jingxuan Wei9, Siyuan Li4, Jintao Chen4, Conghui He1, Cheng Tan1,†

1Shanghai Artificial Intelligence Laboratory, 2Peking University, 3Xi’an Jiaotong University, 4Zhejiang University, 5East China Normal University, 6Hunan University, 7Shanghai Jiao Tong University, 8Shanghai University, 9University of Chinese Academy of Sciences

Existing research infrastructure is fundamentally document-centric, providing citation links between papers but lacking explicit representations of methodological evolution. In particular, it does not capture the structured relationships that explain how and why research methods emerge, adapt, and build upon one another. With the rise of AI-driven research agents as a new class of consumers of scientific knowledge, this limitation becomes increasingly consequential, as such agents cannot reliably reconstruct method evolution topologies from unstructured text. We introduce Intern-Atlas, a methodological evolution graph that automatically identifies method-level entities, infers lineage relationships among methodologies, and captures the bottlenecks that drive transitions between successive innovations. Built from 1,030,314 papers spanning AI conferences, journals, and arXiv preprints, the resulting graph comprises 9,410,201 semantically typed edges, each grounded in verbatim source evidence, forming a queryable causal network of methodological development. To operationalize this structure, we further propose a self-guided temporal tree search algorithm for constructing evolution chains that trace the progression of methods over time. We evaluate the quality of the resulting graph against expert-curated ground-truth evolution chains and observe strong alignment. In addition, we demonstrate that Intern-Atlas enables downstream applications in idea evaluation and automated idea generation. We position methodological evolution graphs as a foundational data layer for the emerging automated scientific discovery.

Date: May 4, 2026 Correspondence: Cheng Tan, tancheng@pjlab.org.cn

Website Dataset

[Figure 2]

[Figure 3]

## 1 Introduction

Scientific progress in artificial intelligence has traditionally been documented through papers, and the infrastructure developed to organize this body of knowledge reflects a document-centric paradigm. Platforms such as Google Scholar [1], Semantic Scholar [2], and OpenAlex [3] treat the paper as the atomic unit: they index titles, abstracts, and citation counts, and they connect papers to one another through citation links. This paradigm has served human researchers for decades. To understand the emergence of vision transformers, for example, a researcher can retrieve a set of relevant papers from these systems, read them, and reconstruct the lineage from convolutional neural networks through self-attention mechanisms to modern architectures. The critical step in this workflow, extracting the structural relationships among methods from narrative text and assembling them into a coherent evolutionary picture, is performed entirely inside the researcher’s head (Figure 1, left).

*Equal contribution.

###### make evolution explicit

###### Intern-Atlas

[Figure 4]

query

[Figure 5]

###### implicit method evolution

method lineage

citation

[Figure 6]

[Figure 7]

BERT

Transformer → BERT/GPT/ViT

adapts

[Figure 8]

Paper A

Paper B

citation

bottleneck evidence

[Figure 9]

[Figure 10]

[Figure 11]

Transformer GPT

adapts

extends

adapts

Google Scholar

open research gap

ViT

[Figure 12]

Attention

Semantic Scholar

Paper A

Paper D

citation

？ open gap

[Figure 13]

[Figure 14]

OpenAlex

verbatim evidence bottleneck → mechanism

[Figure 15]

idea generation

idea evaluation

Papers are linked by citations

Method evolution becomes explicit

Agent uses structured research infrastructure

Human searches papers

Intern-Atlas turns literature into an agent-readable map of methods, bottlenecks, and research gaps

- Figure 1: From paper lists to an agent-readable methodology atlas. Citation-based discovery tools (left) leave method evolution (Transformer → BERT/GPT/ViT) implicit in human expertise. INTERNATLAS (middle) makes it explicit as a typed graph with causal edge labels and verbatim bottleneckto-mechanism evidence, providing an LLM agent (right) with direct queries for method lineage, bottleneck evidence, idea evaluation, and idea generation.

- A new class of knowledge consumer is now emerging. AI-driven research agents that automate the scientific pipeline from hypothesis generation to experimentation [4, 5, 6] represent a fundamentally different kind of knowledge consumer. Unlike humans, these agents cannot reliably reconstruct method evolution topologies from unstructured text. Their parametric memory is a lossy compression that underrepresents low-frequency or long-tail methodological knowledge.Their autoregressive inference operates as fixed-depth forward computation rather than explicit graph traversal, limiting their ability to enumerate branching method spaces. Most critically, they lack the capacity to distinguish between genuine gaps in the research landscape and gaps in their own internal representations, since both manifest as an absence of relevant activations. As a result, their most persistent limitation lies in idea generation, where output quality depends on structural understanding of the methodological landscape: not only which methods exist, but how they evolved, what constraints they addressed, and which directions remain unexplored.

This gap reflects a recurring historical pattern. Structured knowledge infrastructure becomes essential not when humans require it, but when automated systems emerge that cannot operate effectively over unstructured data. The Protein Data Bank (PDB) [7] standardized protein structures decades before AlphaFold [8] revealed their full value as machine-consumable training data. ImageNet [9] organized visual data into hierarchical labels prior to the widespread adoption of convolutional neural networks that depended on large-scale structured annotations. In each case, the emergence of a new computational consumer transformed latent structure into an explicit requirement. A similar inflection point is now occurring in scientific methodology: AI research agents have emerged, yet the structured data layer required to support them remains absent.

We introduce Intern-Atlas, a methodological evolution graph that fills this infrastructure gap. Atlas processes papers from top-tier AI venues, extracts method entities with alias resolution, classifies every citation edge into semantic types, and grounds each non-background edge in a verbatim quote with structured bottleneck and mechanism annotations. The resulting graph forms a queryable causal topology of methodological evolution, providing a persistent data layer that can be directly consumed by downstream systems (Figure 1, right). Beyond graph construction, identifying meaningful evolution chains introduces an additional challenge. Methodological progress forms a directed acyclic graph:

a single method such as the Transformer [10] gives rise to multiple branches, including BERT [11], GPT [12, 13, 14], Vision Transformer [15, 16, 17], and DETR [18]. Greedy traversal strategies that commit at each branching point discard alternative trajectories. To address this, we propose Self-Guided Temporal Monte Carlo Tree Search (SGT-MCTS), which balances exploitation of high-confidence paths with exploration of under-visited branches while enforcing temporal coherence, enabling more faithful reconstruction of methodological evolution.

We evaluate Intern-Atlas along three complementary dimensions. We first assess intrinsic graph quality by quantitatively comparing constructed evolution chains against expert-curated chains. We then examine its utility for idea evaluation, testing whether graph-grounded scoring signals exhibit monotonic alignment with human tiers. Finally, we investigate its value for idea generation by comparing outputs produced with and without access to graph-derived evolutionary context. Taken together, our results suggest that methodological evolution graphs can serve as a foundational data layer for the emerging ecosystem of automated scientific discovery.

## 2 Related Work

### 2.1 Scientific Knowledge Graphs

Tracing knowledge flow through citation networks has a long history. Main Path Analysis [19] identifies heavily traversed citation routes via flow-based traversal counts, representing the earliest formalization of evolution chain reconstruction. CiteSpace [20] detects research fronts through keyword burst analysis and co-citation clustering. Both operate on papers or keywords as atomic units, with edges defined by statistical co-occurrence rather than semantic causality. Modern large-scale platforms have significantly expanded the metadata available for scientific literature. OpenAlex [3] indexes hundreds of millions of works; Semantic Scholar [2] and S2ORC [21] augment standard citation graphs with contextual sentences and influential-citation markers; and Papers With Code [22] introduces structured Task-Dataset-Metric triples. While these platforms serve as invaluable foundations, their network edges remain fundamentally untyped at the methodological level: a citation from paper A to paper B indicates general relevance but carries no machine-consumable structural information regarding whether A extends B’s architecture, resolves B’s specific bottleneck, or merely cites B as background context. Intern-Atlas bridges this critical infrastructure gap by shifting the fundamental unit of analysis from monolithic documents to granular method entities. By coupling a robust methodology registry (including alias resolution) with explicitly typed causal edges (e.g., extends, improves, replaces) grounded in verbatim textual evidence, Intern-Atlas transforms flat citation networks into a directed, queryable topology of methodological evolution. Ultimately, this structural shift provides the explicit, machineconsumable data layer required for automated AI research agents to systematically traverse and reason over the scientific landscape.

### 2.2 Evaluating Research Ideas

Evaluating research ideas is inherently noisy. Human evaluation exhibits high variance, demonstrated by low inter-annotator agreement [23] and inconsistent peer reviews [24]. Automating this with Large Language Models (LLMs) introduces systematic bias: LLM-judged novelty correlates negatively with scientific impact [25], as models inherently favor safe, highly connected concepts [26] due to their reliance on parametric familiarity rather than structural gap reasoning. Recent literature formalizes multi-dimensional evaluation criteria: AI Idea Bench [27] models novelty as historical difference and contemporary influence penalized by conformity; IdeaBench [28] exposes the rarity of simultaneously high novelty and feasibility; and the Ideation-Execution Gap [29] demonstrates quality collapse when highly novel ideas lack practical grounding. Despite their theoretical value, these frameworks remain purely descriptive, lacking the infrastructure for deterministic computation. Intern-Atlas operationalizes these descriptive frameworks into executable scoring functions. We translate

theoretical metrics into explicit topological signals: HindSight’s dimensions are used to determine graph weightings, SciMuse’s connectivity findings are mapped to calculable disconnection ratios, and IdeaBench’s formulaic components are explicitly grounded in our computable topology.

### 2.3 LLM-based Scientific Ideation

End-to-end automated research systems have rapidly evolved from template-driven pipelines to sophisticated multi-agent loops. Early iterations like AI Scientist v1 [4] demonstrated fully automated workflows but relied heavily on hand-crafted templates. Subsequent architectures have introduced dynamic problem-solving: AI Scientist v2 [30] utilizes agentic tree search to generate workshop-accepted papers; CycleResearcher [31] closes the research-review-revision loop to reach preprint-level quality; Dolphin [32] integrates exception-traceback-guided debugging; and AIGS [33] explicitly embeds falsification principles into the research process. Parallel to execution, automated idea generation has seen significant focus. Chain of Ideas (CoI) [34] structures literature into linear chains for LLM-guided extrapolation, while systems like Nova [35] and SciMON [36] optimize iterative search for novelty. However, both ideation and execution agents share a critical structural bottleneck: they construct their knowledge representations from scratch at task launch. CoI assembles chains transiently within prompts; AI Scientist hardcodes knowledge into templates; and SPARK [5] re-retrieves unstructured text per query. This repeated, transient reinvention across independent systems constitutes explicit behavioral evidence of a missing infrastructure layer. Intern-Atlas fills this void by serving as their foundational prerequisite, providing the persistent, queryable, and method-granular knowledge base that current agents fundamentally lack.

## 3 Method

### 3.1 Overview

Existing scholarly platforms [2, 3] treat the paper as the atomic unit and connect papers through citations that carry no semantic information. For an AI research agent, this representation is impoverished: it cannot tell whether paper A extends paper B’s architecture, addresses B’s specific bottleneck with a different mechanism, or merely mentions B as background. We introduce Intern-Atlas, a method-centric heterogeneous graph G that closes this gap. Each edge in G carries a type drawn from a seven-category vocabulary, and every causal edge additionally carries a direct quote from the citing paper that names the bottleneck the edge resolves.

The system (Figure 2) operates in two stages. The first stage, Methodological Graph Construction (§3.2), turns raw AI papers into G by extracting method entities from text, typing every citation edge with one of the seven relations, and attaching to each causal edge a four-field evidence record: three text spans quoted directly from the citing paper (the bottleneck, the mechanism, and the trade-off), plus an LLM-reported confidence. The second stage, Operators over the Graph (§3.3), instantiates three reasoning tasks on top of G: lineage reconstruction traces the causal ancestors and descendants of a target method through G; idea evaluation scores a research idea by where its constituent methods sit in G, treating the graph as a structural map of the field’s methodological landscape; and idea generation searches G for under-explored regions and proposes new ideas to occupy them.

### 3.2 Methodological Graph Construction

We build G from a large-scale corpus of 1,030,314 AI papers over the 1965–2025 period in three steps: (i) entity resolution turns raw text into typed nodes, (ii) edge typing classifies every citation into one of seven causal relations, and (iii) evidence extraction attaches a structured record to every causal edge.

###### Paper Corpus Reference Resolution Two-Phase LLM Extraction Verbatim Validator

[Figure 16]

Canonical + alias registry

Substring match

Large-Scale Data Sourcing 1,030,314 papers

Phase 1 Type classify

Phase 2 Structured record

[Figure 17]

(8,155 methods/9,545 aliases)

Symmetry check

[Figure 18]

BERT

MLLM

RoBERTa DeBERTa

MethodDAG

GPT-4

paper method stub strong-causal (4 types) non-strong

extends

𝑉𝑃

Masked

GPT-2 GPT-3

GPT

Decoder

LM

LM

variant_of combines

𝑉𝑀

Transformer

Patch Embedding

SwinTransformer

MAE

ViT

replaces

𝑉𝑆

Decoder LM

MoE Routing

Set Prediction

DETR

DINOv2

DINO

Stable

specializes

Contrastive Pre-training

LoRA

CLIP LLaMA

Diffusion

LoRA

Parameter

(3 types)

9,410,201 edges | 8,155 methods

AlphaFold 2

Efficient FT

###### Lineage SGT - MCTS

Graph-Grounded Evaluation Strategy-Driven Generation

[Figure 19]

[Figure 20]

[Figure 21]

Novelty

Paradigm Challenge 𝑟𝑒𝑝𝑙𝑎𝑐𝑒 ∩ 𝑆𝑠ℎ𝑎𝑟𝑒𝑑 Evidence

Graph-derived features Red-flag layer

[Figure 22]

temporal coherence exploit

[Figure 23]

Edge e Verbatim bottleneck

Trend Extrapolation 𝑇𝑐ℎ𝑎𝑖𝑛⋂𝐷𝑔𝑎𝑝

[Figure 24]

[Figure 25]

explore

𝑇𝐶(∆𝜏)

[Figure 26]

Bottleneck Resolution 𝐵𝑜𝑝𝑒𝑛

Quote

[Figure 27]

[Figure 28]

Openness justification

temporal+ evidence prior

𝛺𝑐𝑟𝑜𝑠𝑠regularizer

Cross-Pollination 𝑈𝑐𝑜𝑚𝑏𝑜

Validity Significance

[Figure 29]

[Figure 30]

∆𝜏(𝑦𝑒𝑎𝑟𝑠)

- Figure 2: Overview of INTERN-ATLAS. Top: From 1,030,314 AI publications spanning conferences, journals, and arXiv preprints, we construct a typed methodology graph G = (V, E, τ, ρ) via reference

[Figure 31]

[Figure 32]

resolution into papers, canonical methods, and stubs (VP/VM/VS; 8,155 methods, 9,545 aliases), twophase LLM extraction (type classification → structured record), and a code-only verbatim validator. Middle: the resulting graph contains 9,410,201 typed edges over 8,155 canonical methods; strongcausal edges (4 types, solid) induce the lineage subgraph Gstrong, non-strong edges (3 types, dashed) supply retrieval context, and the projected method-level DAG GM. Bottom: Lineage SGT-MCTS reconstruction with an evidence-physics prior over edge confidence and temporal coherence TC(∆τ); graph-grounded idea evaluation on five axes with a parameter-free core function, red-flag detectors, and the cross-dimensional regularizer Ωcross; and strategy-driven idea generation along four topological strategies, each proposal certified by a verbatim evidence record.

No LLM scalar scoring in the core function

- Step 1: entity resolution. The graph has three node sets: papers VP (the parsed corpus), methods VM (one node per method), and stubs VS (placeholders for cited works that fall outside the corpus, totaling 3,173,187 nodes). VM is seeded from a hand-curated list of well-known methods and expanded by an LLM proposer scanning the corpus for additional candidates. Because the same method appears under different names in raw text (e.g., Transformer vs. vanilla Transformer), we collect these names into a lookup table A : string → VM that routes every observed name to the matching method node. The output of Step 1 is a citation graph in which every reference is mapped to a node in VM ∪ VS.
- Step 2: edge typing. Each resolved reference becomes a typed edge e = (u, v) from a method

node u ∈ VM to its target v ∈ VM ∪ VS. An LLM classifier reads the citing paper’s text near the reference and assigns e one of seven labels, ordered by decreasing causal strength: extends, improves, replaces, adapts, uses_component, compares, and background. We write type(e) for this label. The first four categories denote a direct methodological lineage and form the strong-causal subset Esc ⊆ E, on which lineage reconstruction operates; the remaining three are retained for retrieval context. Edges with type(e) ̸= background are called causal edges and proceed to Step 3.

- Step 3: evidence extraction. For every causal edge, an LLM extractor populates a four-field evidence record:

ρ(e) = be, me, te, ce , (1)

where be is the bottleneck addressed, me is the mechanism, te is the trade-off (each a text span quoted from the citing paper), and ce ∈ [0,1] is an LLM-reported confidence. Each bottleneck is also tagged with one of 14 categories from a taxonomy D (e.g., computational cost, long-range dependency). This grouping lets the idea generator reason about which problem categories still lack a strong solution, rather than treating every bottleneck quote as unique. A deterministic post-checker discards any edge whose quoted spans fail substring match against the citing paper, whose endpoints violate publication-year ordering, or whose citing paper already carries an opposite-direction edge to the same target. Further details are in Appendix A.

### 3.3 Operators over the Methodological Graph

All three operators begin the same way. Given a textual input x (a target method, an idea description, or a research question), each operator first uses the lookup table A to find method nodes matching x, then expands outward through BM25 [37] keyword retrieval over VP to collect related papers and the typed edges among them. We denote the resulting localized context by:

Cx = Retrieve(x, G, A). (2)

The three operators differ only in what they do with Cx: lineage reconstruction (§3.3.1) traverses Esc to recover evolution chains, idea evaluation (§3.3.2) scores the input by where its constituent methods sit in Cx, and idea generation (§3.3.3) searches Cx for structural gaps and proposes ideas to fill them.

#### 3.3.1 Lineage Reconstruction with SGT-MCTS

Lineage reconstruction unfolds a method’s evolutionary history. Starting from each seed in S(q) ⊆ Cq that matches the query q exactly, it returns a small set of evolution chains Πq, where each π ∈ Πq is a directed path through the strong causal subgraph (VM, Esc) respecting publication-year order. Because hub methods have multiple successors, we adopt MCTS. We modify its UCT [38] selection rule to use the typed metadata in G. At node u, child v is scored by:

SGT-UCT(v) = UCT(u, v) + λ · αG(u, v), (3)

where UCT(u, v) is the standard exploit-explore criterion and αG(u, v) = conf(eu→v) · TC(∆τuv) is a graph-aware prior built from two factors. The first, conf(eu→v) ∈ [0,1], is the LLM-reported edge confidence stored in ρ(eu→v). The second, TC(∆τuv), is a function of the publication-year gap ∆τuv peaked at ∆τ ∈ [1, 3] years. Their product favors children that are both well-evidenced and temporally plausible. Backward and forward chains from each seed are joined and ranked by:

rank(π) = wℓ L|π|

max

+ wc conf(π) + wm N(π), (4)

combining three complementary terms. |π|/Lmax is the chain’s normalized length and rewards longer chains. conf(π) is the mean edge confidence in π and rewards chains whose every step is wellevidenced. N(π) is the mean MCTS visit count in π, following HopRAG [39] in rewarding chains that many independent rollouts converge on. To recover parallel evolutions at hubs, the lineage operator re-runs from every branch point with covered edges masked, yielding the final Πq.

#### 3.3.2 Graph-Grounded Idea Evaluation

Idea evaluation places a research idea. Taking a textual description d as input, it produces a final score s∗(d, G) that reflects where d’s constituent methods sit on this landscape. Free-text LLM judges reward shallow stacking of popular method names; their novelty assessments correlate negatively with eventual scientific impact while feasibility ratings correlate positively [25, 26]. Our evaluator therefore computes every per-dimension score through a deterministic graph query.

Per-dimension scoring. Each evaluation dimension probes a different structural property of d’s methods inside G, so its score can be computed directly from graph statistics rather than text-based LLM reasoning. The evaluator first resolves the methods named in d into nodes Md ⊆ VM via the lookup table A, then scores each of five dimensions (Novelty N, Feasibility F, Significance S, Validity V, Clarity C) independently from the position of Md in d’s retrieved context Cd:

w(jk) · ϕ(jk)(Md, Cd) , k ∈ {N, F, S,V,C}. (5)

sk(d,G) = clip[1,10] bk +∑

j

For each dimension k, the score combines a baseline bk with a sum of weighted graph statistics, where each ϕ(jk) is a deterministic statistic computed from Md and Cd, and w(jk) is its fixed scalar weight. One of the summed terms ϕ(jF) is an anti-stacking statistic that down-weights ideas dominated by overstudied methods, directly countering the LLM’s positive feasibility bias toward popular method stacks. The complete catalogue of ϕ(jk) across the five dimensions is detailed in Appendix C.3.

Cross-dimensional aggregation. Reducing the five dimensions to a single verdict requires more than a weighted average: certain dimension combinations carry qualitative meaning that a linear aggregation cannot express. The evaluator then aggregates the five raw scores into a final score by combining a linear weighted sum with a non-linear correction:

s∗(d, G) = clip[1,10] w⊤s + Ωcross(s) , s = (sN, sF, sS, sV, sC). (6)

Here w is a fixed weight vector and Ωcross implements four hand-designed conjunctive penalties. A representative penalty fires when an idea scores high on Novelty but low on Feasibility, reflecting that novel but infeasible typically signals a flawed core proposal regardless of its Significance and Validity scores. Finally, an optional LLM reviewer may only lower s∗, providing a one-sided veto that catches occasional errors in the graph-based scoring without reintroducing the LLM’s upward bias.

#### 3.3.3 Strategy-Driven Idea Generation

While evaluation asks “how good is this idea?”, generation asks the inverse question: “which new idea should be proposed?” Operating on the retrieval context Cq (q is a research method of interest), the generator scans for under-explored structural regions and returns a candidate research proposal d′ tied to a concrete bottleneck the literature has yet to solve. Grounding the search in graph structure, rather than free-text reasoning, prevents the LLM from fabricating proposals out of nothing.

Structural gap patterns. From Cq and lineage chains Πq, the generator extracts four structural patterns that together pinpoint where the literature has unfilled research opportunities. These patterns (open axes, recent improvement directions, sacrifice axes, and disconnected pairs) are bundled into a structural summary Φ(q, G) via deterministic graph queries (no LLM call).

Strategy-conditioned proposal. To translate Φ into a concrete proposal, we pair each pattern type with a dedicated generation strategy: Bottleneck Resolution for open axes, Trend Extrapolation for recent improvement directions, Cross-Pollination for disconnected pairs, and Paradigm Challenge for sacrifice axes. This one-to-one mapping converts the LLM’s task from open-ended ideation to constrained completion. The chosen strategy σ fixes both the structural target and the proposal’s high-level form, leaving the LLM only to fill in technical specifics:

d′ = Generate(Φ(q, G), σ). (7)

The strategy alone, however, does not stop the LLM from inventing a bottleneck and dressing it up in the right form. We close this loophole by requiring every proposal to carry an evidence certificate:

[Figure 33]

LLM / Language

Attention Is All You Need (NeurIPS 2017)

Agentic AI 2024 Transformer

ChatGPT 2022

Large Language Model In-Context Learning

Graph / Structure

RL / Agent

Transformer

BERT (NAACL 2019)

RLHF (2024-2025)

Graph Neural Network

Reinforcement Learning

Vision-Language Model

Instruction Tuning (2023)

Chain-of-Thought (2022)

Semantic Segmentation

Vision / Generative

Large Language Model

Object Detection

CNN

Diffusion

Vision / Discriminative

Contrastive Learning

GPT-3 / Few-Shot Learners (NeurIPS 2020)

Self-Supervised Learning

In-Context Learning

Self-supervised

- Figure 3: Intern-Atlas method landscape. Left: AI papers cluster into six paradigm continents connected by typed evolution edges (extends/improves/adapts/replaces). For visual clarity, only a representative subset of nodes is displayed. Right: Detailed view of the LLM continent with landmarks such as Attention, BERT, GPT-3, CoT, Instruction Tuning, and RLHF.

a triple (e, be, j) where e is a specific causal edge from Cq, be is the bottleneck text already stored in ρ(e) (reproduced exactly), and j explains why this bottleneck is still unresolved. Before a proposal is returned, we exact-match be against the stored quote in ρ(e); if it fails, the LLM output is discarded and a deterministic fallback emits a minimal valid proposal from graph content. The result is that every returned proposal is backed by a real edge in G with verifiable bottleneck evidence: no fabricated motivations leak through, and the system always returns a valid proposal.

- 4 Experiment

- Figure 3 provides a high-level view of the Intern-Atlas method landscape, showing major paradigm clusters and their evolution links. We evaluate Intern-Atlas from two perspectives: whether the graph reliably represents methodological evolution, and whether this structure is useful for downstream research tasks. For graph quality, Sec. 4.1 compares the constructed graph with survey-derived expert references and tests whether SGT-MCTS can recover coherent method lineages from the graph. For downstream utility, Sec. 4.2 studies whether graph-derived evidence supports idea evaluation, while Sec. 4.3 studies whether retrieved method lineages help generate better research ideas. Together, these experiments evaluate Intern-Atlas at both representation and application level.

### 4.1 Evaluating Graph Construction and Lineage Reconstruction

This section evaluates whether Intern-Atlas captures meaningful method-evolution structure. The evaluation covers two aspects: whether the graph contains the key methods and evolution relations described in survey-derived benchmark, and whether SGT-MCTS can recover method lineages.

- 91.0%

89.7%

- 92.0%

NMR

ERR

PSC

0 25 50 75 100

Score (%)

(a) Graph quality scores

- 5

- 6

- 7

- 8

- 9

- 10

8.48

Overallideascore

7.83

6.85

5.84

Top-tier Core Workshop Rejected

(b) Overall tier scatter

- Figure 4: Graph construction quality and graph-grounded idea evaluation. (a) Static graph quality on the method-evolution benchmark: node match ratio (NMR), edge reachable ratio (ERR), and path semantic correctness (PSC). (b) Distribution of Overall scores across strata (300 papers each); black markers indicate the sample mean ± 1 s.d.

Method-evolution Benchmark and Metrics. We build a benchmark from 30 high-impact survey papers covering major AI subfields, containing 30 survey-derived method-evolution graphs with 2,268 nodes, 1,462 edges, and 133 evolution chains. Each graph consists of method nodes and evolution relations from the corresponding survey, and the extracted chains serve as references for lineage reconstruction evaluation. For graph construction evaluation, we report Node Match Ratio (NMR), measuring how many survey methods can be matched to Intern-Atlas, and Edge Reachable Ratio (ERR), measuring how many method relations can be recovered as directed paths in Intern-Atlas. We further report Path Semantic Correctness (PSC), which measures whether the recovered path is semantically correct compared to the reference method-evolution relation. For lineage reconstruction evaluation, the chains retrieved by each method are compared with the reference chains using Node Recall (NR), Edge Recall (ER), and Chain Alignment Score (CAS). NR and ER measure the coverage of reference methods and reference transitions, while CAS measures whether the retrieved methods preserve the ordering of the reference chain. More details are provided in Appendix D.1.

Table 1: Lineage-search algorithm performance. ‘@k‘ denotes beam width or RW rollouts.

Evaluation of SGT-MCTS algorithm. Table 1 compares SGT-MCTS with two graph-search baselines. Beam search expands all valid next-hop methods at each step and keeps the top-k highest-scored partial chains. Random walk starts from the same seed method and repeatedly samples a valid next-hop method uniformly at random until the depth limit or a dead end. Compared with the strongest baseline, Beam@10, it improves NR by 39.9 points, ER by 55.8 points, and CAS by 39.9 points, indicating better recovery of the reference methods, transitions, and ordering in target evolution chains. These results support the effectiveness of SGT-MCTS for method-lineage reconstruction. Given the same graph and seed method, SGT-MCTS finds chains that are much closer to the expected evolution lineage of the method than the baselines.

Method NR ER CAS Beam@1 41.0 18.6 41.0 Beam@5 43.4 21.6 43.4 Beam@10 44.9 23.2 44.9 RW@5 28.1 0.7 28.1 SGT-MCTS 84.8 79.0 84.8

Analysis of graph construction quality. Figure 4a shows Intern-Atlas performance on methodevolution benchmark across method coverage, relation reachability, and semantic correctness. The NMR of 91.0% indicates that most methods in the reference graphs can be matched to nodes in InternAtlas. The ERR of 89.7% further indicates that most reference evolution relations can be recovered in Intern-Atlas as directed paths from the source method to the target method. These results suggest

Table 2: Mean idea evaluation scores across publication strata. Scores are computed from extracted ideas. The overall score and all five dimensions decrease from top-tier conference papers to rejected submissions, with the largest gaps appearing in Significance and Validity.

#### Novelty Feasibility Significance Validity Clarity Overall

Top-tier conferences 7.27 7.09 9.49 8.62 6.57 8.48 Core conferences 6.54 6.80 7.16 6.47 6.52 7.83 Workshop papers 5.01 6.38 5.77 6.16 6.17 6.85 Rejected submissions 4.56 5.50 4.91 5.27 5.54 5.84

that Intern-Atlas captures both method entities and their evolution relations, rather than only isolated paper or method records. The PSC of 92.0% further confirms that these recovered paths capture correct evolution semantics. The remaining mismatches are expected because the survey-derived graphs and Intern-Atlas are constructed with different procedures and levels of granularity. Overall, these results show that Intern-Atlas can correctly construct method-evolution structures.

### 4.2 Evaluating Graph-Grounded Idea Evaluator

Experimental details. We evaluate Intern-Atlas as an infrastructure for automated idea evaluation using the Strata Dataset, which contains 1,200 papers (300 per stratum) across four outcome categories: (1) Top-tier AI conferences (ICLR 2026, ICML 2025, NeurIPS 2025), (2) Core AI conferences (AAAI 2026, IJCAI 2025), (3) Workshop papers (from ICLR 2026), and (4) Rejected submissions (from ICLR 2026). For each paper, we extract a concise summary of its core idea as the evaluator input. Our evaluator (described in Sec. 3.3.2) then assesses these extracted ideas, outputting scores across five dimensions (novelty, feasibility, significance, validity, and clarity) and an overall score. More details are in Appendix D.3.

Validation across publication strata. We use the Strata Dataset to examine whether Intern-Atlas scores follow the broad quality ordering of different publication strata. As shown in Table 2 and Figure 4b, the overall score follows the expected order: top-tier conference papers receive the highest mean score (8.48), followed by core conference papers (7.83), workshop papers (6.85), and rejected submissions (5.84). This trend also holds for all five evaluation dimensions. The largest gaps appear in Significance and Validity, suggesting that the method-evolution graph is especially useful for judging whether an idea is tied to important and well-supported methodological directions. Novelty also shows a clear decrease across strata, while Clarity varies more mildly, which is reasonable because an extracted idea can be clearly stated even when its contribution is limited. Overall, these results suggest that Intern-Atlas can support idea-quality evaluation by using method-evolution structure as evidence, producing judgments that broadly align with academic consensus on publication quality.

Human evaluation. For alignment with human expert review, we sample 100 idea profiles from the Strata Dataset and ask 10 AI PhD researchers to score them under the same five-dimensional rubric used by Intern-Atlas. We compare Intern-Atlas with a pure LLM-as-Judge baseline that uses only the idea profile and the scoring rubric as input. As shown in Figure 5a, Intern-Atlas is more closely correlated with expert ratings than the pure LLM baseline across all five dimensions. The overall correlation is 0.81 for Intern-Atlas and 0.58 for the pure LLM baseline. The difference is most visible on Novelty (0.84 vs. 0.52) and Significance (0.82 vs. 0.55), where evaluating an idea requires comparing it with prior methods and understanding its position in methodological development. This suggests that large-scale method-evolution structure provides a more reliable basis for idea evaluation than prompting an LLM to judge from the idea text alone.

###### vs. Expert

###### Overall

1.0

|0.84|0.52|
|---|---|
|0.76|0.61|
|0.82 0.55| |
|0.74|0.58|
|0.79|0.63|

|1.00|0.81|0.58|
|---|---|---|
|0.81|1.00|0.64|
|0.58|0.64|1.00|

[Figure 34]

Novelty

Expert

0.9

WinRate(%)

Feasibility

0.8

Intern-Atlas

Significance

0.7

Validity

0.6

PureLLM

Clarity

0.5

Intern-AtlasPureLLM

ExpertIntern-AtlasPureLLM

(a) Human-alignment correlations

vs. No-KB vs. OpenAlex vs. BM25 RAG

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

80

60

40

20

0

NoveltyFeasibilitySignificance Validity Clarity Overall

(b) Pairwise generation win rate of Intern-Atlas

- Figure 5: Downstream utility of Intern-Atlas for idea evaluation and idea generation. (a) Spearman correlations for idea evaluation: left, dimension-wise alignment of Intern-Atlas and a pure LLM judge with expert scores; right, overall-score correlations among expert judgments, peer reviews, Intern-Atlas, and the pure LLM baseline. (b) Pairwise win rates of Intern-Atlas against idea-generation baselines across five dimensions and overall quality.

Table 3: Idea-quality scores for generated proposals under different knowledge sources.

#### Method Novelty Feasibility Significance Validity Clarity Overall

No-KB 4.85 6.52 5.70 4.00 7.21 5.78 OpenAlex 5.27 7.08 5.34 4.56 7.52 6.03 Semantic Scholar 5.40 7.32 5.39 4.70 7.75 6.18 BM25 RAG 5.39 7.30 5.39 4.66 7.70 6.15 Intern-Atlas 6.37 7.21 6.30 6.26 7.67 7.20

### 4.3 Evaluating Graph-Grounded Idea Generator

Experimental details. To evaluate Intern-Atlas for idea generation, we employ the unified pipeline described in Sec. 3.3.3, varying only the source of background knowledge. We compare Intern-Atlas against three baseline categories: (1) No-KB, which generates ideas directly from queries without external grounding; (2) External Search, using scholarly engines (OpenAlex and Semantic Scholar) as evidence sources; and (3) Local RAG, utilizing the same paper library as Intern-Atlas via standard BM25 retrieval. For input, we use 100 research questions curated by 10 AI PhD researchers. After generation, we score each generated idea using the idea-evaluation pipeline described in Sec. 3.3.2. We further ask the same expert panel to compare Intern-Atlas against the baselines through blind pairwise win-rate evaluation. Additional details are provided in Appendix D.4.

Evaluation Results. Table 3 reports the idea-evaluation scores for ideas generated with different knowledge sources. Intern-Atlas achieves the best Overall score (7.20) and ranks first in Novelty (6.37), Significance (6.30), and Validity (6.26). The gain is most visible in Significance, where method-evolution chains help the generator identify recurring limitations in prior work and propose ideas around more central research problems. The scores on Feasibility and Clarity are close across knowledge-grounded methods, since all methods use the same LLM generator and produce similarly readable proposals. Compared with the best baseline, Intern-Atlas improves the overall score by 1.02 points. Human-expert blind pairwise evaluation in Figure 5b shows the same trend, with Intern-Atlas winning on overall quality by 88.0%, 82.0%, and 81.0% against No-KB, OpenAlex, and BM25 RAG, respectively. Together, these results suggest that method-evolution context is more useful than document-level retrieval for generating high-quality research ideas.

## 5 Conclusion

We have introduced INTERN-ATLAS, a typed methodology graph that promotes flat paper citations into a queryable causal network of methodological evolution. Built from 1,030,314 papers spanning AI conferences, journals, and arXiv preprints, the graph carries 9,410,201 semantically typed edges with verbatim bottleneck-to-mechanism evidence on every causal relation, and exposes three operators over the same substrate: SGT-MCTS lineage reconstruction, graph-grounded idea evaluation, and strategydriven idea generation. Empirically, Intern-Atlas recovers expert-curated evolution chains more faithfully than beam-search and random-walk baselines, produces quality signals that monotonically stratify across publication tiers and align with independent expert review, and generates ideas preferred over external scholarly search and standard RAG baselines under label-blind human judgment. Just as the Protein Data Bank preceded AlphaFold and ImageNet preceded modern visual recognition, we see methodological evolution graphs as a foundational data layer for the emerging ecosystem of automated scientific discovery. We release the graph and pipeline as open infrastructure, and hope future work will build on this direction toward AI research agents that reason over the full causal lineage of human knowledge rather than isolated papers.

## References

- [1] Google scholar. https://scholar.google.com.
- [2] Waleed Ammar, Dirk Groeneveld, Chandra Bhagavatula, Iz Beltagy, Miles Crawford, Doug Downey, Jason Dunkelberger, Ahmed Elgohary, Sergey Feldman, Vu Ha, et al. Construction of the literature graph in semantic scholar. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 3 (Industry Papers), pages 84–91, 2018.
- [3] Jason Priem, Heather Piwowar, and Richard Orr. Openalex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv preprint arXiv:2205.01833, 2022.
- [4] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.
- [5] Aishik Sanyal, Samuel Schapiro, Sumuk Shashidhar, Royce Moon, Lav R Varshney, and Dilek Hakkani-Tur. Spark: A system for scientifically creative idea generation. arXiv preprint arXiv:2504.20090, 2025.
- [6] Tianyu Fan, Fengji Zhang, Yuxiang Zheng, Bei Chen, Xinyao Niu, Chengen Huang, Junyang Lin, and Chao Huang. Deepinnovator: Triggering the innovative capabilities of llms. arXiv preprint arXiv:2602.18920, 2026.
- [7] Helen M Berman, John Westbrook, Zukang Feng, Gary Gilliland, Talapady N Bhat, Helge Weissig, Ilya N Shindyalov, and Philip E Bourne. The protein data bank. Nucleic acids research, 28(1):235–242, 2000.
- [8] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate protein structure prediction with alphafold. nature, 596(7873):583–589, 2021.
- [9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [10] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.
- [12] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- [13] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [14] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- [16] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. ACM computing surveys (CSUR), 54(10s):1–41, 2022.
- [17] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, et al. A survey on vision transformer. IEEE transactions on pattern analysis and machine intelligence, 45(1):87–110, 2022.
- [18] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.

- [19] Norman P Hummon and Patrick Dereian. Connectivity in a citation network: The development of dna theory. Social networks, 11(1):39–63, 1989.
- [20] Chaomei Chen. Citespace ii: Detecting and visualizing emerging trends and transient patterns in scientific literature. Journal of the American Society for information Science and Technology, 57(3):359–377, 2006.
- [21] Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel S Weld. S2orc: The semantic scholar open research corpus. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 4969–4983, 2020.
- [22] OpenRaiser. Nanoresearch, 2026.
- [23] Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers. arXiv preprint arXiv:2409.04109, 2024.
- [24] Alina Beygelzimer, Yann N Dauphin, Percy Liang, and Jennifer Wortman Vaughan. Has the machine learning review process become more arbitrary as the field has grown? the neurips 2021 consistency experiment. arXiv preprint arXiv:2306.03262, 2023.
- [25] Chris Latimer, Nicoló Boschi, Andrew Neeser, Chris Bartholomew, Gaurav Srivastava, Xuan Wang, and Naren Ramakrishnan. Hindsight is 20/20: Building agent memory that retains, recalls, and reflects. arXiv preprint arXiv:2512.12818, 2025.
- [26] Xuemei Gu and Mario Krenn. Interesting scientific idea generation using knowledge graphs and llms: Evaluations with 100 research group leaders. arXiv:2405.17044, 2024.
- [27] Yansheng Qiu, Haoquan Zhang, Zhaopan Xu, Ming Li, Diping Song, Zheng Wang, and Kaipeng Zhang. Ai idea bench 2025: Ai research idea generation benchmark. arXiv preprint arXiv:2504.14191, 2025.
- [28] Sikun Guo, Amir Hassan Shariatmadari, Guangzhi Xiong, Albert Huang, Myles Kim, Corey M Williams, Stefan Bekiranov, and Aidong Zhang. Ideabench: Benchmarking large language models for research idea generation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 5888–5899, 2025.
- [29] Chenglei Si, Tatsunori Hashimoto, and Diyi Yang. The ideation-execution gap: Execution outcomes of llm-generated versus human research ideas. arXiv preprint arXiv:2506.20803, 2025.
- [30] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.
- [31] Yixuan Weng, Minjun Zhu, Guangsheng Bao, Hongbo Zhang, Jindong Wang, Yue Zhang, and Linyi Yang. Cycleresearcher: Improving automated research via automated review. arXiv preprint arXiv:2411.00816, 2024.
- [32] Jiakang Yuan, Xiangchao Yan, Botian Shi, Tao Chen, Wanli Ouyang, Bo Zhang, Lei Bai, Yu Qiao, and Bowen Zhou. Dolphin: Closed-loop open-ended auto-research through thinking, practice, and feedback. arXiv e-prints, pages arXiv–2501, 2025.
- [33] Zijun Liu, Kaiming Liu, Yiqi Zhu, Xuanyu Lei, Zonghan Yang, Zhenhe Zhang, Peng Li, and Yang Liu. Aigs: Generating science from ai-powered automated falsification. arXiv preprint arXiv:2411.11910, 2024.
- [34] Long Li, Weiwen Xu, Jiayan Guo, Ruochen Zhao, Xingxuan Li, Yuqian Yuan, Boqiang Zhang, Yuming Jiang, Yifei Xin, Ronghao Dang, et al. Chain of ideas: Revolutionizing research via novel idea development with llm agents. arXiv preprint arXiv:2410.13185, 2024.
- [35] Xiang Hu, Hongyu Fu, Jinge Wang, Yifeng Wang, Zhikun Li, Renjun Xu, Yu Lu, Yaochu Jin, Lili Pan, and Zhenzhong Lan. Nova: An iterative planning and search approach to enhance novelty and diversity of llm generated ideas. arXiv preprint arXiv:2410.14255, 2024.
- [36] Qingyun Wang, Doug Downey, Heng Ji, and Tom Hope. Scimon: Scientific inspiration machines optimized for novelty. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 279–299, 2024.
- [37] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and beyond. Found. Trends Inf. Retr., 3(4):333–389, April 2009.

- [38] Levente Kocsis and Csaba Szepesvári. Bandit based monte-carlo planning. In European conference on machine learning, pages 282–293. Springer, 2006.
- [39] Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu Xiong, Qinhan Yu, and Wentao Zhang. Hoprag: Multihop reasoning for logic-aware retrieval-augmented generation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 1897–1913, 2025.

## Appendix Table of Contents

- A Graph Construction 17

- A.1 Schema . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.2 Corpus, PDF Pipeline, and Reference Resolution . . . . . . . . . . . . . . . . . . . . . . . 18
- A.3 Two-Phase Extraction Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- B Lineage Reconstruction (SGT-MCTS) 19
- C Idea Evaluation 20

- C.1 Common Retrieval Routine . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.2 Duplicate-Risk Detection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.3 Per-Dimension Signal Specifications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.4 Cross-Dimensional Regularizer Ωcross . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.5 Graceful Fallback and Optional Adjudication . . . . . . . . . . . . . . . . . . . . . . . . . 21

- D Evaluation Setup and Benchmarks 22

- D.1 Method-evaluation Benchmark and Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- D.2 Lineage-Search Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- D.3 Dataset and Baselines Details for Evaluating Idea Evaluator . . . . . . . . . . . . . . . . 23
- D.4 Experimental Details for Evaluating Idea Generation . . . . . . . . . . . . . . . . . . . . 24

- E Case Studies 24 E.1 Case Study on Lineage Search . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- F Limitations and Broader Impact 25

## A Graph Construction

- A.1 Schema Node types. V partitions into three disjoint classes:

- • Paper nodes VP: parsed full texts within the collection window, with resolved references.
- • Method entity nodes VM: canonical methods, each tagged with a canonical name; surface-form mapping is handled by the alias registry A below.
- • Stub nodes VS: cited papers outside the collection window, retained as metadata-only placeholders so that historical lineages remain reachable during SGT-MCTS.

The publication-year map τ : VP ∪ VS → Z is populated from DOI / OpenAlex metadata; missing entries are treated as a neutral default by the temporal-coherence function in Appendix B.

Bottleneck dimension taxonomy D. The 14 axes in Table 4 were bootstrapped by clustering 500 randomly sampled bottleneck quotes from a pilot run on NeurIPS-2024 full texts and consolidated by the authors into a minimal spanning set. Their operational definitions are reproduced verbatim in the Phase-2 extraction prompt and are used wherever a record field is typed by D.

- Table 4: The 14-axis bottleneck dimension taxonomy D. Each axis names a fundamental cost or quality of methodological design; axes type the bottleneck and impact fields of ρ(e) and feed the feasibility/significance signals of the idea evaluator. Definitions are reproduced verbatim in the Phase-2 extraction prompt.

Dimension Operational definition computational complexity asymptotic or wall-clock compute at fixed scale memory efficiency peak activation / parameter memory footprint parallelization degree of across-device or across-token parallelism accuracy task-level correctness or quality metric generalization out-of-distribution / cross-domain transfer scalability behavior as model / data / context size grows data efficiency sample complexity at fixed quality target training stability variance / divergence risk during optimization inference speed serving-time latency or throughput expressiveness function class or representational capacity simplicity implementation, conceptual, or interface simplicity robustness behavior under perturbation or adversarial input hyperparameter sensitivity outcome variance w.r.t. hyperparameter choice training complexity engineering difficulty of the training recipe

Edge vocabulary T . Edges E ⊆ (VP ∪ VS)2 carry a type from a nine-class causal vocabulary (Table 5), whose glossary is reproduced verbatim in the Phase-1 extraction prompt.

Strong-causal subgraph Gstrong. Tstrong = {extends,improves,replaces,adapts} induces Gstrong = (VP ∪ VS, Estrong, τ, ρ). Excluding uses_component separates methodological evolution (inheriting the parent mechanism) from compositional reuse (borrowing only a module). Component reuse is preserved inside the method-level DAG GM for evaluation and generation but does not drive lineage traversal.

- Table 5: The seven causal edge types of T . Definitions are reproduced verbatim in the Phase-1

extraction prompt; the first four form the strong-causal subset Tstrong on which SGT-MCTS performs lineage traversal.

Edge type Operational definition (Phase-1 prompt glossary)

extends adds a new capability or component on top of the cited method improves optimizes an existing component along ≥ 1 dimension in D without altering the

core formulation replaces substitutes a load-bearing component with a qualitatively different mechanism adapts ports the cited method to a new domain, modality, or task setting uses_component re-uses the cited method as an auxiliary module, not as the central contribution compares cites purely as a baseline or head-to-head comparison background cites as context, motivation, or non-methodological background

Method-level DAG GM. GM = (VM, EM) is derived from E by the deterministic projection in Table 6. Two further relations, optimizes and inspired_by, are not produced by citation-anchored extraction: optimizes typically surfaces in release-note or blog text lacking a formal citation, while inspired_by appears in narrative framing rather than bibliographic anchors. We source both from hand-curated seed annotations, yielding TM = {variant_of,specializes,component_of,optimizes,inspired_by}.

- Table 6: Deterministic projection from paper-level causal edges T to method-level relations TM. compares and background are excluded as non-methodological, and uses_component is reversed in direction so that the cited method is encoded as a component of the citing one.

Causal type in T Method-level type in TM extends, improves variant_of adapts, replaces specializes uses_component component_of (direction reversed) compares, background excluded (non-methodological)

Alias registry A. A : VM → 2Σ∗ maps each canonical method to a set of surface forms. Lookup is substring-based with case/punctuation normalization, word-boundary enforcement (preventing “GPT” matching inside “lgpto”), longest-match priority (“GPT-4 Turbo” over “GPT-4” over “GPT”), version-suffix consolidation (“-v2”, “-Large”, etc. attach to the parent unless a distinct canonical node exists), and a manually curated negative-surface list for ambiguous names (e.g., “Mamba” the state-space model versus the Python linter). The released registry contains 8,155 canonical methods and 9,545 aliases; construction details are in §A.2.

### A.2 Corpus, PDF Pipeline, and Reference Resolution

Collection window and venues. Intern-Atlas indexes a comprehensive corpus of major AI venues, journals, and preprint servers over the window 1965–2025, spanning core ML (NeurIPS, ICML, ICLR), computer vision (CVPR, ICCV, ECCV), NLP (ACL, EMNLP, NAACL), general AI (AAAI, IJCAI), and data/information (KDD, SIGIR). ICCV and ECCV run in alternating years, explaining their narrower temporal footprint; all other venues contribute three editions each. Main-conference, Findings (for ACL/EMNLP/NAACL), and accepted-workshop tracks are included uniformly.

PDF parsing pipeline. Full texts are obtained from the official proceedings (where openly licensed) and from OpenReview / arXiv preprints as fallback. Each PDF is parsed by NOUGAT-V1.0 for born-

digital PDFs, with GROBID-V0.8.0 as a fallback for scanned or degraded files that Nougat fails on (3.7% of the corpus). The parser emits structured JSON with section headings, paragraph-level text, and a reference list. Downstream extraction consumes only the Introduction, Method/Approach, and Related Work sections of the citing paper; for cited references only the title and abstract are consumed in Phase 1, and the title plus parsed bibliographic metadata in Phase 2.

Method entity curation. VM is populated in two stages. First, a seed list of 247 widely-used methods is hand-curated by combining the Papers With Code method ontology with mentions enumerated in 18 pilot review articles spanning 2020–2024. Second, an LLM expansion pass scans full-text Method sections for proper-noun method names following capitalization and acronym heuristics; candidates are retained iff they co-occur with ≥ 3 distinct papers’ Method sections and pass adjudication by one of the authors.

### A.3 Two-Phase Extraction Protocol

Extraction decomposes into a high-recall Phase-1 classification over T followed by a Phase-2 recordcompletion pass over non-background candidates. We use QWEN3.6-35B-A3B as the production extraction model for both phases and CLAUDE-SONNECT-4.6 as the audit model used only in the reliability benchmarks. All calls use temperature 0 and deterministic decoding; full prompts are released with the code repository.

- Phase 1 (classification). The model receives the citing paper’s Introduction and Method sections, a list of references with title and abstract, and the verbatim glossary of Table 5. For each reference, it returns a JSON record {type, evidence_snippet}, with background reserved for citations that describe no methodological relation. Phase 1 is tuned for high recall: the prompt explicitly instructs the model to prefer a causal type over background in borderline cases, and the downstream validator together with Phase 2 suppresses false positives. Each prompt batches all references from one citing paper (mean 21.2, max 96); mean input/output lengths are 5.8k / 2.1k tokens.
- Phase 2 (record completion). Phase 2 consumes only the non-background candidates from Phase 1, batched 10 at a time together with the citing paper’s Method section and the verbatim taxonomy of Table 4. For each (citing, cited) pair already typed, the model fills in the structured record ρ(e) ={bottleneck:{description, severity, verbatim_quote, dimension}, mechanism:{description, type, verbatim_quote}, impact:{improvement_dim, sacrifice_dim, tradeoff_sentence}, confidence}, with all verbatim_quote fields required to be copied verbatim from the citing paper’s text. Malformed or schema-violating batches are retried once with a repair prompt; a second failure drops the batch and logs it for manual inspection. Mean input/output lengths are 7.9k / 2.8k tokens.

- B Lineage Reconstruction (SGT-MCTS) Hyperparameters. c = √2 (UCT exploration), λ = 0.5 (prior weight), per-direction-per-seed budget

- B = 200, rollout depth cap dmax = 5, top-K primary lineages K = 5, deduplication node-Jaccard threshold θdup = 0.8, dead-end backpropagation penalty η = −0.05.

#### Temporal coherence.



0.40 −1 ≤ ∆τ < 0

- 0.85 ∆τ = 0
- 1.00 1 ≤ ∆τ ≤ 3 0.80 4 ≤ ∆τ ≤ 6 max(0.30, 1.00 − 0.08(∆τ − 6)) ∆τ > 6 0.70 τ missing



TC(∆τ) =



Edges with ∆τ < −1 in the search direction are hard-filtered prior to evaluation.

(8)

Algorithm. For each seed s ∈ S(q) and each direction in {back,fwd}, MCTS runs for B iterations on Gstrong. Each iteration follows four standard phases: (i) selection by SGT-UCT (Eq. 3); (ii) expansion of the highest-confidence untried child after discarding cycles and edges violating the hard temporal filter; (iii) greedy rollout that at each step picks the child maximizing conf · TC and terminates at dmax, a leaf, or a cycle, scored by R(π); (iv) backpropagation along the selection path, with an additional penalty η to ancestors of dead-end leaves. The two trees per seed are spliced through s by concatenating each top-5 backward path (ranked by accumulated Q) with each top-5 forward path; the resulting candidates are deduplicated under θdup.

Branch discovery. After the top-K primary lineages are finalized, SGT-MCTS is re-run from each branch point, defined as a node with ≥ 2 strong-causal children in the search direction of which only one was traversed by the primary set. Each re-run masks already-covered edges, uses budget B/2, and contributes its surviving lineages to the output pool under the same deduplication rule, surfacing parallel evolutionary trajectories that pure greedy traversal would collapse.

## C Idea Evaluation

- C.1 Common Retrieval Routine Context retrieval. Given x (an idea d or a query q), the routine returns the tuple (Mx, Px, Ex, Bx):

- 1. Mx ⊆ VM: canonical methods resolved from x via the alias registry A.
- 2. Px ⊆ VP: the top-K papers (K = 500) under a hybrid score combining alias-count over Mx with BM25 keyword relevance on the remaining content words.
- 3. Ex ⊆ E: non-background causal edges incident on Px.
- 4. Bx: bottleneck records aggregated from ρ(Ex).

- C.2 Duplicate-Risk Detection

A separate retrieval stack, invoked only by the Novelty signal, ranks VP against d in three stages. Stage 1 (candidate pooling) fuses a dense ranking from BGE-SMALL-EN-V1.5 (384-dim cosine similarity) with

a sparse BM25-Okapi ranking (k1 = 1.5, b = 0.75) via Reciprocal Rank Fusion (kRRF = 60), yielding the top-20 candidates. Stage 2 (rerank) re-scores those candidates with the MS-MARCO-MINILML-12-V2 cross-encoder on (d, p) pairs. Stage 3 (final score) returns fused(d, p) = 0.5dense(d, p) +

- 0.5 σ rerank_logit(d, p) ; the sparse signal contributes only through Stage 1 pooling and does not enter

the final fusion. Writing s = maxp fused(d, p), the Novelty signal applies a step penalty of −0.5 at s≥0.55, −1.5 at s≥0.65, −2.5 at s≥0.75, and −4.0 at s≥0.85 (zero otherwise).

### C.3 Per-Dimension Signal Specifications

Bases and weights. All five dimensions share base bk = 5.0. Aggregation weights are w = (wN, wF, wS, wV, wC) = (0.20,0.20,0.25,0.20,0.15). Signal contributions are additive on bk; the sum is clipped to [1,10].

Per-dimension signals. Load-bearing components are listed below.

Novelty. (i) Topological disconnection of Md in the method-co-utilization graph, up to +2.0; (ii) mechanism-level Jaccard distance against Ed, +[0, 1.5]; (iii) frontier-leaf bonus, +0.8; (iv) duplicate-risk penalty −pen(d) ∈ [−4.0,0] from §C.1.

Feasibility. A sweet-spot maturity curve over each method’s paper-count pc(m),

 

1.5 + 1.5 · pc500(m) pc(m) ≤ 500 3.0 − 1.0 · pc(m1500)−500 500 < pc(m) ≤ 2000 1.5 pc(m) > 2000

ϕF(m) =



averaged over Md and capped at +3.5. The 500/2000 thresholds are the 60th and 90th percentiles of pc(·) on the released corpus; the non-monotonic shape blocks mashups from inflating F via pervasive

terms. Auxiliary signals: full-text resource availability and a complexity penalty for |Md| ≥ 4.

Significance. Time-decayed in-degree on Pd (5-year half-life, non-background); frontier presence (≥ 3 non-background out-edges since 2021); and the method-frontier regularizer, the dominant anti-mashup

term, with mean popularity over Md[: 5] contributing +2.5 when < 300 (niche) and decaying linearly to −2.0 at > 1000 (saturated).

Validity. Bottleneck grounding against Bd (matched dimension tag plus a description bigram, up to

+3.5); ancestry consistency in GM within depth 4; type-weighted edge density on Ed. Clarity. Method recognition rate; specificity peak at |Md| ∈ {2,3} decaying for |Md| ≥ 6; structural completeness across {problem, approach, target}; length adequacy on 20–200 words.

### C.4 Cross-Dimensional Regularizer Ωcross

Table 7: Four empirical cross-dimensional priors comprising Ωcross. Each row triggers an additive adjustment to the overall score s∗ when its conjunctive condition over the post-red-flag scores s′ holds, encoding patterns where one dimension’s score becomes more or less informative in conjunction with another.

Prior Trigger Adjustment

Ideation–execution gap s′N ≥ 7 ∧ s′F < 4 −0.6 Validity–feasibility coherence sV′ ≥ 7 ∧ s′F ≥ 7 +0.2 Significance amplification s′S ≥ 6 +0.4

5 ≤ s′S < 6 +0.2 Balanced coherence max(s′) − min(s′) ≤ 2 ∧ min(s′) ≥ 5 +0.3

### C.5 Graceful Fallback and Optional Adjudication

Graceful fallback. If Md = ∅, the pipeline returns a population prior s∗ = 6.5, abstaining from the graph-grounded functional rather than systematically penalizing purely theoretical contributions whose novelty is orthogonal to the method registry. On a 147-idea pilot from NeurIPS-2024 submissions,

text-only signals under Md = ∅ gave Spearman ρ = 0.08 (p = 0.31) against peer-review scores, justifying abstention.

Adjudicator (optional). When enabled, two LLM calls modify s∗ via bounds rather than recompute it. Part A (duplicate verification): the adjudicator classifies the relationship between d and the top-3 retrieved candidates as {duplicate, related, unrelated}; the strongest verdict determines the restoration rate of the automatic duplicate-risk penalty: 0% for duplicate (full penalty retained), 60% for related, 90% for unrelated. Part B (per-dimension upper bound): the adjudicator rates the idea on coherence, novelty validity, and claim plausibility on 1–10; each rating acts as an upper bound on its mapped Atlas dimension via sk ≤ llm_scorek + 1.0, with coherence → {V,C}, novelty → N, plausibility → {F, S}. Any sub-score below 3 triggers a hard cap s∗ ≤ 6.0. The adjudicator therefore never computes s∗ de novo, preserving the zero-trainable-parameter property of the core scorer.

## D Evaluation Setup and Benchmarks

### D.1 Method-evaluation Benchmark and Metrics

Purpose and selection. We construct the method-evolution benchmark from 30 high-impact survey papers, using surveys as proxies for expert consensus on methodological evolution. Candidate surveys are selected according to five criteria: authority, impact, classic status, subfield diversity, and methodcentricity. In practice, we prioritize surveys published in recognized venues, widely cited by the community, commonly used as field-level taxonomies, covering diverse AI subfields, and containing explicit discussions of method evolution rather than only application summaries. These surveys provide external references for both static graph coverage and lineage reconstruction evaluation.

Construction. Each survey is processed in two stages. First, an LLM extracts discussed methods, directed evolution relations, and narration-based method chains from the survey text and taxonomy figures. Second, domain researchers manually audit the extracted results by removing hallucinated methods, correcting relation directions, merging duplicate method names, and checking each retained relation against the original survey text or figure. The final benchmark contains 30 survey-derived method-evolution graphs with 2,268 nodes, 1,462 directed evolution edges, and 133 reference evolution chains.

Evaluation protocols. We evaluate the benchmark in two parts, following the main text.

- 1. Static Graph Coverage. Given a survey-derived graph G∗ and Intern-Atlas graph G, we report Node Match Ratio (NMR), Edge Reachable Ratio (ERR), and Path Semantic Correctness (PSC). NMR measures the fraction of survey methods matched to Intern-Atlas nodes. ERR measures the fraction of reference edges that can be recovered as directed paths in G. PSC measures whether reachable paths preserve the intended method-evolution semantics, as judged against the survey reference. Shortest-hop statistics are used only as a diagnostic for path compactness.
- 2. Lineage Reconstruction. For each reference chain, every search method starts from the same seed and returns a candidate chain. We report Node Recall (NR), Edge Recall (ER), and Chain Alignment Score (CAS). NR measures how many reference methods are recovered. ER measures how many adjacent reference transitions are recovered. CAS measures how well the retrieved chain preserves the ordering of the reference chain.

### D.2 Lineage-Search Baselines

To evaluate the lineage reconstruction capability of SGT-MCTS, we compare it against two standard graph-search baselines. All algorithms operate on the identical knowledge graph, originate from the

same query seeds, and are constrained by the same maximum depth and computational budget.

- • Beam Search: Expands successors and selectively retains the top-k partial paths at each step. It utilizes the identical local heuristic signals as SGT-MCTS but lacks global rollout-based exploration. This serves as a baseline for greedy semantic search, demonstrating the limitations of purely exploitative pathfinding without long-term planning. We evaluate beam widths of k ∈ {1,5,10}.
- • Random Walk (RW): Uniformly samples valid outgoing edges from the current node until the depth cap is reached. This process is repeated iteratively to match the total node-evaluation budget of SGT-MCTS. As a naïve exploration baseline, it isolates and validates the necessity of heuristic guidance and value estimation in our method.

All retrieved lineages from these baselines are evaluated against the same reference chains and metric suite described previously.

### D.3 Dataset and Baselines Details for Evaluating Idea Evaluator

To validate the graph-grounded idea evaluator, we construct the Strata Dataset and apply a unified extraction protocol that standardizes the input format across all compared methods. We use the full dataset for publication-strata analysis and a 100-sample expert-rated subset for human-alignment evaluation.

The Strata Dataset. This dataset tests whether the evaluator’s outputs follow the broad quality ordering reflected by different publication strata. We collected 1,200 papers evenly split across four tiers (300 each): (1) Top-tier AI conferences: accepted papers from ICLR 2026, ICML 2025, and NeurIPS 2025; (2) Core AI conferences: papers from AAAI 2026 and IJCAI 2025; (3) Workshop papers: submissions accepted to ICLR 2026 workshops; and (4) Rejected submissions: papers declined at ICLR 2026. All texts and metadata were obtained from OpenReview and publicly available proceedings. Venue and outcome labels are used only for aggregate analysis and are not provided to the evaluator.

Human-Rated Subset. For human-alignment evaluation, we sample 100 idea profiles from the Strata Dataset and ask 10 AI PhD researchers to rate each profile on a 1-10 scale across the same five dimensions used by Intern-Atlas: novelty, feasibility, significance, validity, and clarity. The experts are active AI researchers and are shown only the extracted idea profiles, not the paper source, venue, acceptance outcome, Intern-Atlas scores, or LLM baseline scores. We average expert ratings for each profile and compute Spearman correlations between the expert scores and the scores produced by Intern-Atlas and the pure LLM-as-Judge baseline.

Idea Extraction Protocol. Before scoring, we extract a standardized summary from each paper to ensure the evaluator focuses on the research idea itself rather than writing style. Specifically, given a paper’s abstract, introduction, and core method sections, an LLM-based extractor produces a structured Idea Profile with four fields: (1) Problem Formulation: the specific limitation or bottleneck the paper addresses; (2) Core Innovation: the key idea or mechanism proposed; (3) Technical Implementation: the main algorithmic components needed to realize the idea; and (4) Target Setting: the task or application domain. The extractor is prompted to use a neutral tone and omit background exposition, rhetorical framing, and unsubstantiated claims, retaining only the factual methodological content. The evaluator then operates on this extracted profile rather than the raw paper text, which keeps the input format consistent across all compared methods and avoids confounding the evaluation with writing quality. The exact prompt is provided in the code release.

Pure LLM-as-Judge Baseline. This baseline employs Qwen3.5-397B-A17B to evaluate the standardized Idea Profile in a zero-shot setting. It is prompted with the exact same five-dimensional 1–10 scoring rubric used by Intern-Atlas and the human experts, but relies strictly on its internal parametric knowledge. It computes the evaluation without access to external document retrieval engines or any structured method-evolution graph context.

### D.4 Experimental Details for Evaluating Idea Generation

Query Construction and Distribution. To standardize the evaluation of automated idea generation, we curated a diverse set of 100 high-quality research queries. To ensure clear constraints and evaluation objectives, each query follows a strict tripartite structure:

- • Domain Setting: Establishes the specific AI subfield (e.g., “In retrieval-augmented generation...”).
- • Core Challenge: Identifies the fundamental bottleneck or limitation of existing methods (e.g., “...multi-step retrieval is fundamentally restricted by high resource cost...”).
- • Expectation: Specifies the desired capabilities of the target solution (e.g., “Therefore, there is a critical need to build a resource-efficient mechanism...”).

To comprehensively evaluate the generative generalizability across different topics, these 100 queries are distributed across six major AI fields: NLP & LLMs (34%), General ML & Optimization (19%), Computer Vision (14%), Reinforcement Learning & Agents (14%), Multimodal & Speech (13%), and Graph Learning (6%). The full list of queries is provided in the supplementary material.

Human Expert Win-Rate Evaluation. To provide a rigorous, unbiased assessment of idea quality and compute the win rates, we assembled a panel of human experts, comprising active researchers with publication records in top-tier AI venues. The evaluation followed a strict double-blind pairwise comparison protocol. For each query, the experts were presented with the reference query alongside anonymized idea proposals generated by Intern-Atlas and the baselines. They were tasked with selecting the superior idea (or declaring a tie) based on overall methodological quality, specifically factoring in Novelty, Feasibility, and Significance. Crucially, experts were permitted to consult external academic search tools during the evaluation to verify factual correctness and the true novelty of the proposed mechanisms.

## E Case Studies

### E.1 Case Study on Lineage Search

We use a familiar ConvNet lineage to illustrate the behavior of different search algorithms. All searches are seeded at ConvNeXt V2. The reference chain is a survey-level summary of the residual ConvNet backbone: VGG scales ConvNets through deeper stacks of small convolutions; ResNet addresses the optimization difficulty of very deep networks with residual mappings; ResNeXt extends residual blocks with aggregated transformations; ConvNeXt modernizes ResNet-style ConvNets using Transformerera design choices; and ConvNeXt V2 further extends ConvNeXt with a fully convolutional masked autoencoder and global response normalization.

Reference: VGG → ResNet → ResNeXt → ConvNeXt → ConvNeXt V2.

SGT-MCTS recovers the complete reference subsequence in the correct order. The extra Caffe node provides implementation context before VGG, and the final node is a downstream user of ConvNeXt V2; neither disrupts the main residual ConvNet backbone. Beam search recovers the later segment from ResNet to ConvNeXt V2, but misses the VGG depth-scaling step and enters the chain through an

- Table 8: Qualitative comparison of lineage-search results for a ConvNeXt-V2-centered ConvNet lineage.

Method Retrieved lineage

Reference VGG → ResNet → ResNeXt → ConvNeXt → ConvNeXt V2 SGT-MCTS Caffe → VGG → ResNet → ResNeXt → ConvNeXt → ConvNeXt V2 → Robust Context-Aware Object

Recognition

Beam@10 Speech Recognition with Deep RNNs → LSTM ASR → Highway Networks → ResNet → ResNeXt → ConvNeXt → ConvNeXt V2 RW@100 Learning to Forget → Highway Networks → ResNet → Shake-Shake → Decoupled Weight Decay → ConvNeXt → ConvNeXt V2

RNN/Highway-Network branch. Random walk recovers only a shorter local subsequence around ResNet, ConvNeXt, and ConvNeXt V2, while skipping both VGG and ResNeXt. This case shows the value of MCTS-style exploration in a high-branching architecture graph: by keeping multiple plausible ancestors alive, SGT-MCTS preserves the ordered VGG–ResNet–ResNeXt–ConvNeXt backbone more faithfully than the greedy and random baselines.

## F Limitations and Broader Impact

Extraction. Phase-1 edge-type classification accuracy ranges from 70.4% (production model) to 93.0% (audit model), reflecting the inherent difficulty of distinguishing fine-grained causal relations such as extends versus improves. Downstream operators treat edge types as routing rather than ground truth, so localized misclassifications shift ranking quality without breaking the operator interface. The 14-axis bottleneck taxonomy D is fixed at release, and emerging dimensions are mapped to the closest existing axis until a future taxonomy revision. Substring-based alias resolution favors precision over coverage and uses a manually maintained negative list to handle truly ambiguous names.

Algorithmic. The temporal coherence function TC is calibrated on post-2015 AI literature, so fields with markedly different research cadences would require recalibration. The evaluator’s zero-trainableparameter design is a deliberate commitment to deterministic reproducibility, trading potential accuracy gains from a learned scorer for full auditability. The four generation strategies cover the most frequent topological moves we observed; less common modes such as theoretical unification of two previously distinct frameworks are natural extensions.

Broader impact. Intern-Atlas is a research-infrastructure artifact, and its risks are largely indirect. A downstream agent consuming the graph could amplify existing citation biases when allocating credit across methodological traditions. We mitigate this through (i) verbatim-grounded evidence on every non-background edge so that consumers can audit provenance, (ii) public release of the reliability audit alongside the graph, and (iii) a stated commitment that the artifact is not to be used for authoror institution-level ranking. We see no plausible pathway to direct harm from the artifact itself.

