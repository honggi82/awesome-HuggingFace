## arXiv:2605.04615v2[cs.SE]8May2026

### Beyond Retrieval: A Multitask Benchmark and Model for Code Search

Siqiao Xue‡, Zihan Liao, Jin Qin, Ziyin Zhang, Yixiang Mu‡, Fan Zhou‡, Hang Yu†

Ant Group Hangzhou, China

Website Code Data

#### Abstract

Code search now underpins not only developer-facing tools but also modern AI coding agents (e.g., SWE-agent, OpenHands, Cursor). Yet existing benchmarks evaluate only the embedding stage, ignoring the reranker and developerstyle queries that production pipelines actually use, and additionally suffer from data contamination, label noise, and degenerate binary relevance. In this paper, we introduce COREB, a contamination-limited, multitask code retrieval and reranking benchmark, together with a fine-tuned code reranker, that goes beyond retrieval to cover the full code search pipeline. COREB is built from counterfactually rewritten LiveCodeBench problems in five programming languages and delivered as timed releases with graded relevance judgments. We benchmark eleven embedding models and five rerankers across three tasks: text-tocode, code-to-text, and code-to-code. Our experiments reveal that: ① codespecialized embeddings dominate code-to-code retrieval (∼2× over general encoders), yet no single model wins all three tasks; ② short keyword queries, the format closest to real developer search, collapse every model to near-zero nDCG@10; ③ off-the-shelf rerankers are task-asymmetric, with a 12-point swing on code-to-code and no baseline net-positive across all tasks; ④ our fine-tuned COREB-RERANKER is the only reranker we evaluate that achieves consistent gains across all three tasks. The data and model are released via the project site:

https://hq-bench.github.io/coreb-page/.

#### 1 Introduction

Large language models (LLMs) for code (Zhang et al., 2024b) have rapidly become foundational components of modern software engineering. Among their most impactful applications is code search, which enables developers to efficiently locate relevant code snippets, explanations, and semantically similar implementations, as exemplified by systems like GitHub Code Search (GitHub, 2023). Modern code search is rarely a single-stage process: it pairs an embedding model that maps programs into continuous vector spaces for fast first-stage retrieval with a downstream reranker that refines the final ranking. The same retrieve-and-rerank pipeline now also powers a new generation of AI coding agents—such as SWE-agent (Yang et al., 2024), OpenHands (Wang et al., 2025), and IDE assistants like Cursor (Anysphere, 2024)—which rely on code search to ground their LLMs in the right files and snippets before editing or repairing code. Improvements at either stage therefore propagate beyond search itself and lift the entire AI coding pipeline.

Despite rapid progress, evaluating the full code search pipeline remains challenging. We identify four issues that limit current practice. (D1) Missing reranking support. No existing code retrieval benchmark evaluates or provides a reranking stage. Practitioners must assemble their own pipelines using generic encoders not trained on code, which may hurt retrieval quality on code-specific tasks.

*‡Work done at Alipay. †Corresponding author. Preprint.

The remaining issues concern the retrieval stage specifically. Through a dataset-level analysis of CoIR (Li et al., 2024a), the most prominent benchmark for code retrieval, we identify: (D2) Contamination and benchmark overfitting. CodeSearchNet (Husain et al., 2020) and its derivatives together account for over 85% of CoIR’s corpus volume and have served as pre-training data for a line of code representation models (CodeBERT (Feng et al., 2020), CodeT5 (Wang et al., 2021), and CodeRetriever (Li et al., 2022)), many of which also serve as backbones for downstream code embedding models. This training–evaluation overlap can inflate metrics by up to 100% (Allamanis, 2019); inter-dataset leakage further distorts results (Hernandez Lopez et al., 2024); and Karaman & Akarsu (2025) report 15–25% test/train near-duplication in CodeSearchNet-style data, while Siddiq et al. (2024) find evidence of data contamination in APPS. (D3) Label noise and trivial matching. Gong et al. (2026) estimate that around 51% of pairs in CoSQA (Huang et al., 2021b), CoIR’s only human-annotated dataset, are mismatched under functional verification; our own manual inspection of 80 test pairs finds approximately 60% to be problematic. CodeSearchNet pairs code with its own verbatim docstring, a task the original authors flag as “overly simplistic” due to shared authorship vocabulary (Husain et al., 2020), and Li et al. (2024b) confirm produces “documentation strings or comments rather than natural language questions,” limiting real-world applicability. Beyond label quality, three of ten CoIR datasets do not test code retrieval at all: CodeSearchNet-CCR randomly splits functions mid-token as a string-completion proxy, CodeFeedback-MT is dialogue continuation where prior assistant turns already contain complete solutions, and StackOverflow-QA contains virtually no code. (D4) Degenerate relevance structure. All ten datasets assign exactly one relevant document per query with binary scores and no explicit hard negatives, collapsing nDCG@k and MRR into redundant hit-or-miss signals and making Recall@k binary, a limitation the CoIR authors themselves acknowledge (Li et al., 2024a). Moreover, meaningful tasks cover only Python and SQL (text-to-code) or Python and C++ (code-to-code), leaving languages such as Go and Ruby entirely untested (Diera et al., 2023). See sections B.12.1 to B.12.6 for a dataset-by-dataset analysis.

To fill this gap, we introduce COREB, a multitask, contamination-limited code retrieval and reranking benchmark, together with a fine-tuned code reranker, that addresses the above flaws by design and covers both stages of the code search pipeline.

How COREB addresses D1–D4. To resolve D1, we train and release a code reranker on hard negatives mined from COREB’s own retrieval runs, providing a ready-to-use two-stage pipeline that improves over generic reranking on every retrieval direction we evaluate.

For D2, rather than mining from public repositories, we build COREB around counterfactually rewritten LiveCodeBench (LCB) (Jain et al., 2024) problems and regenerated artifacts: each problem statement is rewritten to remove memorized surface forms while preserving its functional semantics, and frontier code models are re-evaluated on the refreshed instances.

Our annotation study shows that Pass@1 consistently decreases after rewriting whenever the underlying problems fall within a model’s training cutoff, indicating that some original LCB problems are partially memorized by contemporary models such as Gemini 3 Flash and Claude Sonnet 4.5. Using the refreshed problems and their resulting test cases, we automatically construct text-to-code, code-to-text, and code-to-code tasks across five programming languages, and re-run this pipeline whenever LCB releases new problems; we elaborate on the construction pipeline in section 3.

We resolve D3 by deriving relevance labels programmatically from execution outcomes: every code candidate is run against test oracles and pass/fail status determines relevance, eliminating the human annotation noise that causes ∼51% mislabeling in CosQA (Gong et al., 2026).

Finally, we address D4 with graded relevance judgments that mix multiple positives and explicit hard negatives in a single relevance scheme: each judgment assigns relevance=2 to true positives, relevance=1 to same-problem hard negatives (e.g., failed code solutions for text-to-code and code-to-code; LLM-generated noise descriptions for code-to-text), and treats unjudged items as easy negatives (Table 5). Across our two releases, 68% of text-to-code queries have two or more correct solutions and code-to-code queries average 2.2 valid cross-language translations, so ranking metrics such as nDCG and Recall reflect meaningful ordering rather than a one-hit test; evaluation uses relevance level=2, so a model that retrieves a hard negative above a true positive is penalized.

Empirical findings. We evaluate eleven embedding models and five rerankers across all three tasks. Key findings include: ① No single model wins every task; leaderboard rankings shift across retrieval directions. ② Code-specialised training matters more than scale: a 0.5B code-trained model

###### 45%

###### 49%

7%

Text-to-Code (T2C) Code-to-Code (C2C) Code-to-Text (C2T)

| |
|---|

| |
|---|

| |
|---|

(a) By main task

15%

16%

14%

26%

16%

13%

Any Python Java

C++ Ruby Go

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

(b) By language constraint

7% 16%

34%

10%

34%

Canonical Retro Full Retro Search Cross-Lingual Binary Match

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

(c) By subtask variant

Figure 1: Aggregated query distribution across both COREB releases (5,087 queries total). Percentages <8% are shown outside with leader lines. Per-release breakdowns are in Figure 11.

outperforms general-purpose 8B encoders, with the premium concentrated on code-to-code (∼2×). ③ Two systemic failures span every model: short keyword queries collapse to near-zero nDCG@10, and low-resource languages lag by up to 0.33 points. ④ Reranking is high-stakes: a 12-point swing separates the best and worst baselines on code-to-code, and no off-the-shelf reranker is net-positive across all tasks. Our fine-tuned COREB-RERANKER is the first to achieve consistent gains on all three tasks, establishing a complete two-stage pipeline released alongside the benchmark.

#### 2 Related Work

Code LLMs have progressed from specialized systems like Codex (Chen et al., 2021) to frontier models such as GPT-4.1 (OpenAI, 2025), Claude 3.5 (Anthropic, 2024), and Gemini 3 Pro (Google DeepMind, 2025), as well as open families including StarCoder2 (Lozhkov et al., 2024) and Qwen2.5-Coder (Hui et al., 2024), where strong coding capability is integrated alongside general reasoning. For evaluation, BEIR (Thakur et al., 2021) and MTEB (Muennighoff et al., 2022) cover text retrieval broadly but include few code tasks; code-specific suites like CoIR (Li et al., 2024a) and CPRet (Deng et al., 2025) unify multiple datasets yet inherit structural problems from their constituents: degenerate 1-to-1 qrels, label noise, contamination risk, and tasks that reduce to string matching rather than genuine retrieval (sections B.12.1 to B.12.6). On the retrieval-model side, dense encoders from CodeBERT (Feng et al., 2020) to CodeSage (Zhang et al., 2024a) and QodoEmbed (Qodo Ltd., 2025) achieve strong results on these benchmarks, but their behavior on broader, contamination-limited scenarios remains underexplored. See Appendix B.11 for a fuller discussion.

#### 3 Benchmark Construction

Our benchmark is constructed through a multi-stage pipeline (Figure 2) that converts recent competitive-programming problems into a suite of code representation and retrieval tasks. The design emphasizes contamination resistance, semantic diversity, and evaluation realism.

- Step 1: Seed problem sourcing from LCB. We source problems from LCB, a continuously updated benchmark that mitigates contamination via temporal filtering. Mirroring LCB’s incremental model, COREB is published as timed releases (Table 2):1 v202602 (covering contest problems from Sep 2024 to Jan 2025) and v202603 (Jan to Apr 2025). Dataset statistics are aggregated across both releases unless a per-release breakdown is shown.

1Source test splits: https://huggingface.co/datasets/livecodebench/code_generation_lite/ blob/main/test5.jsonl, https://huggingface.co/datasets/livecodebench/code_generation_ lite/blob/main/test6.jsonl

###### MAIN TASK SUBTASK QUERY DESCRIPTION AVG. TOKENS

CANONICAL RETRO† ABBREVIATED PROBLEM DESCRIPTIONS 120 FULL RETRO† FULL PROBLEM STATEMENTS 431 SEARCH†‡ LLM-GENERATED SEARCH QUERIES 19

TEXT-TO-CODE

CODE-TO-CODE CROSS-LINGUAL CODE SNIPPET (ANCHOR) 207

CANONICAL RETRO† CODE SOLUTION 246 FULL RETRO† CODE SOLUTION 246 PAIR MATCH CODE SNIPPET; 1 RELEVANT DOC 252

CODE-TO-TEXT

- Table 1: Query subtasks in COREB with per-subtask average query length. Token counts use cl100k base; exact per-language counts in Table 11; corpus statistics in Table 16. †Each subtask has 6 language variants (Any + Python, Java, C++, Ruby, Go); bar colours in Figure 1 show this breakdown. ‡The Search subtask refers to a specific query type (short developer-style keyword queries); it is distinct from the broader “code search” used in the paper title, which encompasses the full retrieve-then-rerank pipeline.

LIVECODEBENCH SYNTHETIC RELEASE SOURCE CONTEST PERIOD # SEED

PROBLEMS

# CODE CORPUS

# TEXT CORPUS

- V202602 RELEASE V5 SEP’24–JAN’25 167 1,670 835

- V202603 RELEASE V6 JAN’25–APR’25 175 1,744 875 AGGREGATED 342 3,414 1,710

- Table 2: COREB releases. Contest period gives the date range of LiveCodeBench problems included in that snapshot. # Code Corpus counts generated solutions across five languages (2 models × 5 languages × # seed problems); # Text Corpus counts original descriptions plus LLM-generated hard negatives (4 per problem). The two releases draw from disjoint LiveCodeBench snapshots, so the Aggregated row is a straight sum of distinct records. Per-task query counts are in Table 11.

0

Pass@1(pp)

- Step 2: Counterfactual rewriting and code generation. We apply light counterfactual rewriting (Wu et al., 2024) to each problem’s statement and test cases by modifying named entities, variable names, narrative framing, and I/O examples while preserving the formal specification and algorithmic structure. This rewriting acts as a controlled intervention: it alters the surface form through which a model might recognize a problem while holding fixed the underlying algorithmic task and the space of valid solutions. Rewrites are drafted by GPT-o1 and human-verified for semantic equivalence; we additionally re-run the full test suite to confirm that solutions passing the originals still pass the rewritten versions (Appendix A.1). We then generate solution candidates in five programming languages (Python, C++, Java, Go, Ruby) using two frontier LLMs (Gemini 3 Flash and Claude Sonnet 4.5) and execute each against the full rewritten test suite, recording pass/fail as metadata (Appendix A.8). All candidates are retained regardless of correctness: 3,414 across both releases, of which 1,065 pass every test case (verified-correct).

5

10

15

Python Java C++ GoRubyOverall

- v202602 (Sep 24 Jan 25)

| |
|---|

- v202603 (Jan Apr 25)

Figure 3: Pass@1 change (pp) after rewriting for Gemini 3 Flash across two releases covering different contest windows.

To test whether rewriting suppresses memorization, we run a controlled 2×2 study: the same two models are each evaluated on both original and rewritten problems across two releases (Figure 3). Because the original and rewritten versions encode the same formal specification, a drop in Pass@1 isolates sensitivity to the problem’s surface realization rather than a change in intrinsic difficulty. For Gemini, Pass@1 decreases consistently after rewriting on both releases and across all five languages, indicating that surface-level recognition inflates apparent solving ability even when the algorithmic task is unchanged. Claude exhibits a release-dependent pattern: it drops on the older release, whose problems are more likely to overlap with its training data, but shows little change on the newer one (per-language breakdowns for both models in Table 13). These results suggest that data leakage varies across models and releases in ways that are difficult to predict without a controlled test. By

###### Step 1

###### Step 2

###### Step 3

###### Step 4

###### Step 5

###### Query Generation

###### Seed Problem Sourcing

###### Rewriting & Code Generation

###### Retrieval Instance Formulation

###### Dataset Format

LCB incremental snapshot + official tests

rewrite statements & test cases; 2 models × 5 langs

task-level variants (T2C, C2C, C2T); LLM + human

query-corpus pairing; graded relevance labels

JSONL queries, corpora, qrels; HuggingFace

LLM rewrite statements & test cases

LLM draft task-level queries

Human verification

Human review

Figure 2: Benchmark construction pipeline. Each step is detailed in the corresponding paragraph below.

rewriting problem statements while preserving their algorithmic content, we aim to reduce the influence of memorization so that benchmark scores better reflect genuine coding and retrieval ability.

- Step 3: Query generation. We create multiple task-level query variants per problem with LLM assistance, then human-review each for faithfulness and absence of leakage. Representative instances, full schemas, and per-subtask examples are in Appendix A.5.
- Step 4: Retrieval instance formulation. Each retrieval instance pairs a query with its taskspecific corpus and is assigned a three-level graded relevance label. Relevance 2 (positive): for text-to-code and code-to-code, verified-correct solutions to the queried problem; for code-to-text, the original problem description corresponding to the queried code. Relevance 1 (hard negative): same-problem items that are superficially plausible but incorrect: failed solution candidates for textto-code and code-to-code, and LLM-generated noisy descriptions for code-to-text. Unjudged (easy negative): all remaining corpus items, which carry no explicit label. For all metrics we set relevance level=2: binary metrics (Recall, MRR) count only relevance=2 items as relevant, and for nDCG hard negatives (relevance=1) are zeroed so they contribute zero gain yet still penalize by occupying top ranks that true positives could occupy. This formulation is deliberately stricter than CoIR: retrieving a same-problem but incorrect item is not treated as success. The benchmark can therefore distinguish models that merely recognize the associated problem from those that correctly rank solutions above plausible distractors.

- Step 5: Dataset format. COREB is distributed as JSONL files for queries, corpora (code and text), and relevance judgments. Full field schemas and examples are in Appendices A.4 and A.5.

#### 4 Experiments

We evaluate COREB as a two-stage code search benchmark: a first-stage retrieval followed by a reranking stage. This mirrors the retrieve-then-rerank pipeline used in production code search systems and lets us assess both stages independently.

- 4.1 Experimental Setup Retrieval models. We evaluate eleven embedding models spanning 0.5B–8B parameters:

- • C2LLM (0.5B, 7B) (Qin et al., 2025): code-specialized, adaptive multi-head attention pooling.
- • F2LLM (0.6B, 1.7B, 4B) (Zhang et al., 2025c): code-specialized, fine-tuned on large datasets.
- • Jina-code-embeddings (0.5B, 1.5B) (Kryvosheieva et al., 2025): code-specialized embedding models.
- • Qwen3-Embedding (0.6B, 4B, 8B) (Zhang et al., 2025b): general-purpose encoders.
- • GemEmb-2 (Google, 2026): closed API (Gemini Embedding 2 preview); nominal 3B.

Reranker baselines. For the reranking stage, we compare against four publicly released rerankers, against which we benchmark our fine-tuned COREB-RERANKER (section 4.3):

- • Jina Reranker v2 (base multilingual) and v3 (Sturua et al., 2024): general-purpose rerankers.

MODEL TEXT-TO-CODE CODE-TO-TEXT CODE-TO-CODE† OVERALL NDCG RECALL NDCG RECALL NDCG RECALL NDCG RECALL

GEMEMB-2 0.434 0.749 0.813 0.842 0.698 1.000 0.637 0.819 C2LLM-7B 0.443 0.753 0.795 0.842 0.659 0.997 0.629 0.820 C2LLM-0.5B 0.430 0.716 0.753 0.840 0.656 0.970 0.603 0.800 JINA-CODE-EMB-1.5B 0.414 0.705 0.763 0.835 0.671 0.973 0.603 0.794 JINA-CODE-EMB-0.5B 0.386 0.650 0.755 0.822 0.677 0.963 0.588 0.763 F2LLM-4B 0.407 0.695 0.763 0.837 0.500 0.766 0.581 0.768 QWEN3-EMB-4B 0.390 0.626 0.728 0.828 0.392 0.603 0.546 0.717 F2LLM-1.7B 0.383 0.603 0.715 0.805 0.383 0.562 0.536 0.692 F2LLM-0.6B 0.344 0.545 0.665 0.793 0.334 0.491 0.491 0.654 QWEN3-EMB-8B 0.328 0.521 0.660 0.780 0.320 0.450 0.481 0.633 QWEN3-EMB-0.6B 0.349 0.541 0.617 0.755 0.384 0.551 0.477 0.641

- Table 3: First-stage retrieval on COREB v202603 (graded qrels, relevance level=2). Columns “nDCG” and “Recall” denote nDCG@10 and Recall@10. Overall is query-count-weighted. Hard negatives (rel=1) penalize nDCG but do not count toward Recall. Bold marks the per-column best.

• Qwen3-Reranker (0.6B, 4B) (Zhang et al., 2025b): instruction-tuned rerankers, also serve as the backbones for our fine-tuned variants.

Metrics. All subtasks are evaluated with a uniform ranked-retrieval protocol: each model encodes the query, ranks all corpus items by cosine similarity, and is scored on the resulting ranked list. We report Recall@k and normalized discounted cumulative gain (nDCG@k) in the main text, and include mean reciprocal rank (MRR) in the appendix. Formal definitions are in Appendix B.4.

- 4.2 Retrieval Results and Analysis All analyses in this section are based on v202603; v202602 results are in Appendix B.5.

Main results. Table 3 reports per-task and overall nDCG@10 and Recall@10 for all eleven models. GemEmb-2 achieves the highest overall nDCG@10 and leads on code-to-text and code-tocode, while C2LLM-7B is the strongest open-weight model and leads on text-to-code. Across the board, code-specialized open models (C2LLM, Jina-code) outperform general-purpose encoders of comparable or larger size. Crucially, no single model dominates all three tasks, confirming that text-to-code, code-to-text, and code-to-code probe complementary capabilities. Rankings are highly consistent across releases: four of the top-5 models are shared across v202602 and v202603, with per-task nDCG@10 differences within 0.03 for every model (Table 14).

- Analysis I: How much do the three tasks differ in difficulty? The three tasks span a wide difficulty range (Figure 4). Code-to-text is the easiest (model-averaged nDCG@10 of 0.73), textto-code is the hardest (0.39), and code-to-code sits in between (0.52) yet is the most discriminative, with a cross-model spread twice that of code-to-text. The Qwen3 family illustrates this divergence starkly: it scores competitively on code-to-text yet collapses on code-to-code, showing that crossmodal alignment does not transfer to cross-lingual code retrieval. A single-task benchmark would miss these complementary failure modes entirely.
- Analysis II: Does scaling up consistently improve retrieval? Larger models do not reliably outperform smaller ones within the same family. Qwen3-Emb-8B slightly trails Qwen3-Emb-0.6B overall, and drops noticeably on code-to-code (0.320 vs 0.384), the opposite of what a simple scaling law would predict. Similarly, F2LLM-1.7B underperforms F2LLM-4B despite having triple the parameters. These non-monotonic curves suggest that training-data composition mediates the size– quality relationship: scaling up a general-purpose encoder does not guarantee better code retrieval.
- Analysis III: Can small specialized models compete with large general ones? Among open checkpoints, Jina-code-emb-0.5b attains the second-highest code-to-code nDCG@10, trailing only the closed-API GemEmb-2 and edging C2LLM-7B despite being 14× smaller. More broadly, C2LLM-0.5B outperforms the 16× larger Qwen3-Emb-8B by over 12 points overall. The pattern is consistent: whenever a code-specialized variant exists alongside a general-purpose model

of comparable or larger size, the specialized model wins on code-heavy tasks. For code retrieval, domain-aligned training data matters more than raw model capacity, and a 0.5B specialized checkpoint can be a better practical choice than an 8B general-purpose one.

nDCG@1 nDCG@5 nDCG@10

0.8

0.77

0.76

| |
|---|

0.69

0.6

0.56

nDCG

0.45 0.42

0.4

0.30

0.21

0.19

0.2

0.0

text-to-code code-to-text code-to-code

- Figure 4: nDCG@k at k∈{1, 5, 10} per task, averaged over all eleven models on v202603. Codeto-text saturates early; text-to-code and code-tocode grow more steeply with k.

MODEL CANONICAL FULL SEARCH

GEMEMB-2 0.573 0.565 0.000 C2LLM-7B 0.582 0.578 0.004 C2LLM-0.5B 0.566 0.560 0.003 F2LLM-4B 0.529 0.535 0.007 JINA-CODE-1.5B 0.539 0.543 0.006 F2LLM-1.7B 0.481 0.520 0.008 QWEN3-4B 0.510 0.504 0.015 QWEN3-0.6B 0.452 0.460 0.008 F2LLM-0.6B 0.450 0.452 0.000 QWEN3-8B 0.477 0.381 0.005 JINA-CODE-0.5B 0.497 0.514 0.003

Table 4: Text-to-code nDCG@10 by subtask (v202603). Canonical and Full use long queries; Search uses short keyword queries (∼19 tokens). Every model collapses on Search.

Analysis IV: Training data, language, and query length shape retrieval more than model size. Beyond the scaling and specialization patterns above, three finer-grained factors emerge from persubtask and per-language breakdowns (Figures 5 and 6 and Table 4):

- ① Code-to-code exposes training regime. GemEmb-2 and the Jina-code models lead code-to-code by a wide margin, while Qwen3 models collapse to roughly half their nDCG@10 despite competitive text-to-code and code-to-text scores. Cross-language code-pair training drives this gap; even scaling Qwen3 from 0.6B to 8B does not close it with the far smaller Jina-code-emb-0.5b. Per-language breakdowns (Table 15) show C++ and Go anchors are easier to match than Java and Python, consistent with the greater idiomatic diversity of the latter two languages (Zhang et al., 2025a).
- ② Short queries collapse all models. On the text-to-code Search subtask (19-token keyword queries), every model drops to near-zero nDCG@10, two orders of magnitude below the long-query Canonical subtask (Table 4). Current retrievers have saturated in the long-query regime while the shortquery regime, closest to real search, remains unsolved (see section 4.4 for potential mitigations).
- ③ Target language introduces systematic bias. Language-agnostic text-to-code queries score substantially higher than language-constrained ones (Figure 6), with Ruby and Go lagging consistently across all models. The gradient tracks training-corpus coverage: Python and Java dominate public code, so models embed them more faithfully.

[Figure 1]

Python Java C++ Go

C2LLM-7B C2LLM-0.5B

F2LLM-4B Jina-code-1.5B Jina-code-0.5B

- F2LLM-0.6B

EmbGemma

- F2LLM-1.7B Jina-v4

Qwen3-0.6B Qwen3-8B Qwen3-4B

0.68 0.49 0.76 0.75 0.56 0.58 0.73 0.77 0.45 0.54 0.74 0.69 0.78 0.47 0.79 0.73 0.75 0.50 0.79 0.73 0.49 0.52 0.70 0.68 0.33 0.58 0.69 0.64 0.54 0.54 0.76 0.70 0.35 0.42 0.69 0.70

- 0.41 0.36 0.44 0.21
- 0.42 0.38 0.59 0.49 0.14 0.28 0.39 0.20

|[Figure 2]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

- Figure 5: Code-to-code nDCG@10 by anchor language on v202603 (after anchor exclusion).

text-to-code nDCG@10 by language

[Figure 3]

C2LLM-7B C2LLM-0.5B

0.76 0.67 0.42 0.47 0.45 0.50

0.8

- 0.72 0.73 0.54 0.39 0.37 0.40
- 0.73 0.67 0.53 0.50 0.37 0.41 0.72 0.58 0.48 0.44 0.43 0.48

[Figure 4]

F2LLM-4B Jina-code-1.5B Jina-code-0.5B

0.7

0.6

- 0.71 0.65 0.39 0.44 0.47 0.38

- 0.70 0.66 0.51 0.51 0.34 0.35 0.68 0.54 0.70 0.44 0.26 0.45

0.72 0.66 0.49 0.53 0.36 0.38 0.68 0.66 0.45 0.47 0.27 0.44

- 0.72 0.70 0.53 0.41 0.43 0.33

0.71 0.53 0.57 0.49 0.53 0.25

- 0.73 0.74 0.57 0.50 0.32 0.27

- F2LLM-0.6B

EmbGemma

- F2LLM-1.7B Jina-v4

0.5

0.4

0.3

Qwen3-0.6B Qwen3-8B Qwen3-4B

0.2

AnyPython Java Cpp Ruby Go

Figure 6: Text-to-code nDCG@10 by target language on v202603 (excluding Search subtask).

Full per-subtask and per-language tables are in Appendix B.6.

- Analysis V: Do hard negatives provide additional evaluation signal? A key design choice in COREB is the explicit inclusion of same-problem hard negatives (relevance=1) in the qrels. Unlike benchmarks with binary or absent-is-irrelevant qrels, our graded scheme exposes fine-grained discriminability: a model that ranks a failed code solution or a noise description above the true positive is penalized, even if it places that true positive within the top k.

To quantify this, we measure the hard-negative intrusion rate: among queries where both a positive and a hard negative appear in the top 10, what fraction have at least one hard negative ranked above the highest positive? Figure 7 shows the result. On text-to-code, the intrusion rate exceeds 55% for every model—more than half of all queries have a failed code solution outranking the correct one. Notably, stronger models (e.g., GemEmb-2, 64%) can have higher intrusion than weaker ones because they retrieve more same-problem content overall (2.1 hard negatives per query vs. 1.3 for Qwen3-8B), creating more opportunities for mis-ranking; the metric is conditioned on queries where both a positive and a hard negative reach the top 10. On code-to-code, intrusion ranges from 43% (GemEmb-2) to 59% (F2LLM-4B), with code-specialized models consistently lower than generalpurpose ones. Code-to-text intrusion is much lower (6–30%), confirming that text retrieval is easier but still exposing a 5× gap between the best and worst models. This failure mode, retrieving plausible but incorrect items above true positives, is invisible to benchmarks without explicit hard negatives, yet directly affects deployment quality.

Hard-negativeintrusionrate(%)

Text-to-Code

Code-to-Code

Code-to-Text

70

| |
|---|

| |
|---|

60

50

40

30

20

10

0

GemEmb-2 C2LLM 7B

jina 1.5b

C2LLM 0.5B

jina 0.5b

F2LLM 4B

Qwen3 4B

F2LLM 1.7B

Qwen3 0.6B

F2LLM 0.6B

Qwen3 8B

Figure 7: Hard-negative intrusion rate: fraction of queries where at least one hard negative ranks above the best true positive in the top 10. Higher means worse discrimination. Models sorted by overall nDCG@10.

##### 4.3 Reranker Evaluation

10

Main results on reranking. We rerank the top-128 candidates retrieved by C2LLM-7B (the strongest open-weight retriever) with four baseline rerankers: Jina Reranker v2, Jina Reranker v3, Qwen3-Reranker-0.6B, and Qwen3-Reranker-4B. Figure 8 reports the nDCG@10 delta per task. No baseline is netpositive across all three tasks. ① Code-to-text: all four baselines degrade performance (from −3.2% for Qwen3-4B to −22.4% for Jina v2), because the retriever already saturates near 0.8 nDCG@10 on the compact text corpus and reranking amplifies noise. ② Text-to-code: deltas range from −8.3% (Jina v2) to −0.1% (Qwen3-4B); no baseline improves this task. ③

Text-to-Code

Code-to-Text

Code-to-Code

| |
|---|

| |
|---|

5

nDCG@10(%)

0

5

10

15

20

25

Jina v2 Jina v3 Qwen3 0.6B

Qwen3 4B

CoREB Reranker

Figure 8: ∆nDCG@10 (%) after reranking (k=128) on top of C2LLM-7B. No baseline is net-positive across all three tasks; only our fine-tuned COREBRERANKER achieves this.

Code-to-code: the only task where reranking helps, with Qwen3-4B gaining +3.3%, as crosslanguage disambiguation benefits from fine-grained pairwise scoring; however, the other three baselines still hurt. Overall, a 12-point swing separates the best and worst baseline on the same task, showing that reranker selection matters as much as the decision to rerank itself.

Fine-tuned COREB-RERANKER. To mitigate task asymmetry limitations observed in baseline models, we fine-tune Qwen3-Reranker-4B (Zhang et al., 2025b) via LoRA (Hu et al., 2022) on a

- 3.1M-sample corpus, merging COREB v202602 with datasets including CodeSearchNet (Husain et al., 2019; Li et al., 2025; Lu et al., 2021), APPS (Hendrycks et al., 2021), CosQA (Huang et al., 2021a), and CodeFeedback (Zheng et al., 2024). Training instances are formatted as triplets (q,d,y). We ensure data balance by doubling positive samples and pairing each with one easy and one hard negative. Problem-level disjointness between v202602 and the v202603 test set is maintained. The released checkpoint is the uniform model soup (Wortsman et al., 2022) of two LoRA-fine-tuned variants trained from the same initialization with different seeds and data shuffles. (See Appendix B.3.)

As shown in Figure 8, COREB-RERANKER is the only reranker in our evaluation that is net-positive across all three tasks, establishing a complete two-stage pipeline, retrieval (C2LLM-7B) followed by in-domain reranking, that improves over the retriever alone on every task.

0.3B 0.5B 1B 2B 4B 8B

Parameter count (log scale)

0.54

0.56

0.58

0.60

0.62

0.64

0.66

OverallnDCG@10

EmbGemma-300M

C2LLM-0.5B

C2LLM-7B

Code-specialized General-purpose Pareto frontier

| |
|---|

Figure 9: Overall nDCG@10 vs. parameter count (log scale) for the ten open-weight models. Circles = code-specialized; squares = general-purpose. The dashed line marks the Pareto frontier. GemEmb-2 is excluded due to its unknown parameter size.

0.0 0.5 1.0 1.5 2.0

Overall nDCG@10 / billion parameters

C2LLM-0.5B Jina-code-0.5B Qwen3-0.6B F2LLM-0.6B

F2LLM-4B Qwen3-4B C2LLM-7B Qwen3-8B

1.21 1.19

0.74 0.73

0.14 0.12

0.09 0.05

··· Code-specializedGeneral-purpose

Figure 10: Parameter efficiency (nDCG@10 per billion parameters) for the four most and four least efficient models; middle-ranked models are omitted (full ranking in Figure 13).

- 4.4 Practical Guidance for Code Search Our analyses yield concrete recommendations for practitioners building code search systems.

Choose code-specialized embeddings. Figure 9 plots overall nDCG@10 against parameter count for all eleven models. Among open-weight checkpoints, two lie on the Pareto frontier: C2LLM-

- 0.5B and C2LLM-7B; every other open model is simultaneously outperformed and outscaled by a smaller open checkpoint. C2LLM-0.5B reaches 95.9% of the best open-weight score at only 7% of the parameter count, and 0.5B code-specialized models deliver roughly 13× more nDCG per billion parameters than 8B general-purpose ones (Figure 10). For latency- or memory-constrained deployments, a 0.5B code-specialized checkpoint is the clear choice at any scale evaluated here.

Reranker selection is high-stakes. Off-the-shelf rerankers are task-asymmetric: a 12-point swing separates the best and worst baseline on the same task (Figure 8). A poorly chosen reranker can degrade code-to-text by over 20 points. In-domain fine-tuning on graded qrels (as in our COREBRERANKER) is needed to achieve consistent gains across all three retrieval directions.

Short-query retrieval remains an open problem. The near-zero performance on keyword-style Search queries (Table 4) is the single largest unsolved gap. Neither scaling the embedding model nor adding a reranker closes it; query expansion techniques such as HyDE (Gao et al., 2023) or Query2doc (Wang et al., 2023) are the most promising path forward.

#### 5 Conclusion

We presented COREB, a contamination-limited benchmark and reranker that covers both stages of the code search pipeline across three tasks and five languages. Our experiments show that codespecialised training matters more than scale, that short keyword queries and low-resource languages remain unsolved failure modes, and that reranker selection is high-stakes.

#### References

Allamanis, M. The adverse effects of code duplication in machine learning models of code. In Proceedings of the ACM SIGPLAN International Symposium on New Ideas, New Paradigms, and Reflections on Programming and Software (Onward!), 2019. arXiv:1812.06469.

Anthropic. Claude 3.5 sonnet. https://www.anthropic.com/news/claude-3-5-sonnet, 2024.

Anthropic. Introducing claude sonnet 4.5. https://www.anthropic.com/news/

claude-sonnet-4-5, 2025.

Anysphere. Cursor: The ai code editor. https://cursor.com, 2024.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., HerbertVoss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Deng, H., Meng, Y., Tang, S., Ouyang, W., and Ma, X. Cpret: A dataset, benchmark, and model for retrieval in competitive programming. arXiv preprint arXiv:2505.12925, 2025.

Diera, A., Dahou, A., Galke, L., Karl, F., Sihler, F., and Scherp, A. GenCodeSearchNet: A benchmark test suite for evaluating generalization in programming language understanding. In Proceedings of the 1st GenBench Workshop on Generalisation (Benchmarking) in NLP (EMNLP), 2023.

Du, K., Peng, Y., Gao, C., Zhou, F., and Xue, S. Doraemon: A unified library for visual object modeling and representation learning at scale. arXiv preprint arXiv:2511.04394, 2025.

Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L., Qin, B., Liu, T., Jiang, D., and Zhou, M. CodeBERT: A pre-trained model for programming and natural languages. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.

Gao, C., Xue, S., Fu, J., Gu, T., Li, S., and Zhou, F. Lookbench: A live and holistic open benchmark for fashion image retrieval. arXiv preprint arXiv:2601.14706, 2026.

Gao, L., Ma, X., Lin, J., and Callan, J. Precise zero-shot dense retrieval without relevance labels. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023.

GitHub. Github code search. https://github.com/features/code-search, 2023. Gong, J., Wu, Y., Liang, L., Wang, Y., Chen, J., Liu, M., and Zheng, Z. Cosqa+: Pioneering the

multi-choice code search benchmark with test-driven agents. arXiv preprint arXiv:2406.11589, 2026.

Google. Introducing Gemini 3 Flash. https://blog.google/products-and-platforms/

products/gemini/gemini-3-flash, December 2025.

Google. Gemini Embedding 2: A natively multimodal embedding model, 2026. Vertex AI model documentation (public preview API). Developer overview: https://blog.google/technology/developers-tools/gemini-embedding-2/ (March 2026).

Google DeepMind. Gemini 3 pro. https://deepmind.google/models/gemini/pro/, 2025. Grattafiori, A., Dubey, A., and et al., A. J. The llama 3 herd of models. arXiv preprint

arXiv:2407.21783, 2024.

Guo, D., Ren, S., Lu, S., Feng, Z., Tang, D., Liu, S., Zhou, L., Duan, N., Svyatkovskiy, A., Fu, S., Tufano, M., Deng, S. K., Clement, C., Drain, D., Sundaresan, N., Yin, J., Jiang, D., and Zhou, M. Graphcodebert: Pre-training code representations with data flow. In Proceedings of the International Conference on Learning Representations (ICLR), 2021.

Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., and Steinhardt, J. Measuring coding challenge competence with APPS. In Vanschoren, J. and Yeung, S. (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021.

Hernandez Lopez, J. A., Chen, B., Saad, M., Sharma, T., and Varro, D. On inter-dataset code duplication and data leakage in large language models. arXiv preprint arXiv:2401.07930, 2024.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Lowrank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.

Huang, J., Tang, D., Shou, L., Gong, M., Xu, K., Jiang, D., Zhou, M., and Duan, N. Cosqa: 20, 000+ web queries for code search and question answering. In Zong, C., Xia, F., Li, W., and Navigli, R. (eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pp. 5690–5700. Association for Computational Linguistics, 2021a.

Huang, J., Tang, D., Shou, L., Gong, M., Xu, K., Jiang, D., Zhou, M., and Duan, N. CoSQA: 20,000+ web queries for code search and question answering. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), Online, August 2021b. Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Lu, K., Dang, K., Fan, Y., Zhang, Y., Yang, A., Men, R., Huang, F., Zheng, B., Miao, Y., Quan, S., Feng, Y., Ren, X., Ren, X., Zhou, J., and Lin, J. Qwen2.5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Husain, H., Wu, H., Gazit, T., Allamanis, M., and Brockschmidt, M. Codesearchnet challenge: Evaluating the state of semantic code search. CoRR, abs/1909.09436, 2019.

Husain, H., Wu, H.-H., Gazit, T., Allamanis, M., and Brockschmidt, M. Codesearchnet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436, 2020.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Karaman, R. K. and Akarsu, M. Code2Doc: A quality-first curated dataset for code documentation. arXiv preprint arXiv:2512.18748, 2025.

Kryvosheieva, D., Sturua, S., G¨unther, M., Martens, S., and Xiao, H. Efficient code embeddings from code generation models. arXiv preprint arXiv:2508.21290, 2025.

Li, R., Fu, J., Zhang, B.-W., Huang, T., Sun, Z., Lyu, C., Liu, G., Jin, Z., and Li, G. TACO: Topics in algorithmic COde generation dataset. arXiv preprint arXiv:2312.14852, 2023.

Li, X., Gong, Y., Shen, Y., Qiu, X., Zhang, H., Yao, B., Qi, W., Jiang, D., Chen, W., and Duan, N. Coderetriever: Unimodal and bimodal contrastive learning for code search. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022.

Li, X., Dong, K., Lee, Y. Q., Xia, W., Yin, Y., Zhang, H., Liu, Y., Wang, Y., and Tang, R. Coir: A comprehensive benchmark for code information retrieval models. arXiv preprint arXiv:2407.02883, 2024a.

Li, X., Dong, K., Lee, Y. Q., Xia, W., Zhang, H., Dai, X., Wang, Y., and Tang, R. Coir: A comprehensive benchmark for code information retrieval models. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pp. 22074–22091. Association for Computational Linguistics, 2025.

Li, Z., Zhang, J., Yin, C., Ouyang, Y., and Rong, W. ProCQA: A large-scale community-based programming question answering dataset for code search. In Proceedings of LREC-COLING, 2024b. arXiv:2403.16702.

Lozhkov, A., Li, R., Allal, L. B., Cassano, F., Lamy-Poirier, J., Tazi, N., Tang, A., Pykhtar, D., Liu, J., Wei, Y., Liu, T., Tian, M., Kocetkov, D., Zucker, A., Belkada, Y., Wang, Z., Liu, Q., Abulkhanov, D., Paul, I., Li, Z., Li, W.-D., Risdal, M., Li, J., Zhu, J., Zhuo, T. Y., Zheltonozhskii, E., Dade, N. O. O., Yu, W., Krauß, L., Jain, N., Su, Y., He, X., Dey, M., Abati, E., Chai, Y., Muennighoff, N., Tang, X., Oblokulov, M., Akiki, C., Marone, M., Mou, C., Mishra, M., Gu, A., Hui, B., Dao, T., Zebaze, A., Dehaene, O., Patry, N., Xu, C., McAuley, J., Hu, H., Scholak, T., Paquet, S., Robinson, J., Anderson, C. J., Chapados, N., Patwary, M., Tajbakhsh, N., Jernite, Y., Ferrandis, C. M., Zhang, L., Hughes, S., Wolf, T., Guha, A., von Werra, L., and de Vries, H. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173, 2024.

Lu, S., Guo, D., Ren, S., Huang, J., Svyatkovskiy, A., Blanco, A., Clement, C. B., Drain, D., Jiang, D., Tang, D., Li, G., Zhou, L., Shou, L., Zhou, L., Tufano, M., Gong, M., Zhou, M., Duan, N., Sundaresan, N., Deng, S. K., Fu, S., and Liu, S. Codexglue: A machine learning benchmark dataset for code understanding and generation. In Vanschoren, J. and Yeung, S. (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021.

Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. Mteb: Massive text embedding benchmark.

arXiv preprint arXiv:2210.07316, 2022. OpenAI. Introducing GPT-4.1 in the API. https://openai.com/index/gpt-4-1/, 2025. Qin, J., Liao, Z., Zhang, Z., Yu, H., Di, P., and Wang, R. C2llm technical report: A new frontier in

code retrieval via adaptive cross-attention pooling. arXiv preprint arXiv:2512.21332, 2025. Qodo Ltd. Qodo-embed-1-7b: A 7b parameter code embedding model for software retrieval.

###### https://huggingface.co/Qodo/Qodo-Embed-1-7B, 2025. Accessed: 2025-05-10.

Siddiq, M. L., Dristi, S., Saha, J., and Santos, J. C. The fault in our stars: Quality assessment of code generation benchmarks. In Proceedings of the IEEE International Working Conference on Source Code Analysis and Manipulation (SCAM), 2024. arXiv:2404.10155.

Sturua, S., Mohr, I., Akram, M. K., G¨unther, M., Wang, B., Krimmel, M., Wang, F., Mastrapas, G., Koukounas, A., Wang, N., and Xiao, H. jina-embeddings-v3: Multilingual embeddings with task lora. arXiv preprint arXiv:2409.10173, 2024.

Thakur, N., Reimers, N., R¨uckl´e, A., Srivastava, A., and Gurevych, I. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Wainakh, Y., Rauf, M., and Pradel, M. Idbench: Evaluating semantic representations of identifier names in source code. arXiv preprint arXiv:1910.05177, 2021.

Wang, L., Yang, N., and Wei, F. Query2doc: Query expansion with large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

- Wang, X., Li, B., Song, Y., Xu, F. F., Tang, X., Zhuge, M., Pan, J., Song, Y., Li, B., Singh, J., et al. OpenHands: An open platform for AI software developers as generalist agents. In Proceedings of the International Conference on Learning Representations (ICLR), 2025.
- Wang, Y., Wang, W., Joty, S., and Hoi, S. C. H. Codet5: Identifier-aware unified pre-trained encoderdecoder models for code understanding and generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021.

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., and Schmidt, L. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In Proceedings of the International Conference on Machine Learning (ICML), 2022.

Wu, J., Yang, L., Wang, Z., Okumura, M., and Zhang, Y. Cofca: A step-wise counterfactual multihop qa benchmark. In Proceedings of the International Conference on Learning Representations (ICLR), 2024.

Xue, S., Jiang, C., Shi, W., Cheng, F., Chen, K., Yang, H., Zhang, Z., He, J., Zhang, H., Wei, G., Zhao, W., Zhou, F., Qi, D., Yi, H., Liu, S., and Chen, F. Db-gpt: Empowering database interactions with private large language models. arXiv preprint arXiv:2312.17449, 2023.

Xue, S., Qi, D., Jiang, C., Shi, W., Cheng, F., Chen, K., Yang, H., Zhang, Z., He, J., Zhang, H., Wei, G., Zhao, W., Zhou, F., Yi, H., Liu, S., Yang, H., and Chen, F. Demonstration of DB-GPT: Next generation data interaction system empowered by large language models. In Proceedings of the VLDB Endowment, 2024.

Xue, S., Zhu, Z., Zhang, W., Cai, R., Wang, R., Mu, Y., Zhou, F., Li, J., Di, P., and Yu, H. Quitobench: A high-quality open time series forecasting benchmark. arXiv preprint arXiv:2603.26017, 2026.

Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., and Press, O. SWEagent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

- Zhang, D., Ahmad, W. U., Tan, M., Ding, H., Nallapati, R., Roth, D., Ma, X., and Xiang, B. CODE REPRESENTATION LEARNING AT SCALE. In Proceedings of the International Conference on Learning Representations (ICLR), 2024a.
- Zhang, E. et al. Across programming language silos: A study on cross-lingual retrieval-augmented code generation. arXiv preprint arXiv:2506.03535, 2025a.

- Zhang, Y., Li, M., and Team, Q. Qwen3-embedding: Advancing text embedding and ranking with foundation models. arXiv preprint arXiv:2506.05176, 2025b.
- Zhang, Z., Chen, C., Liu, B., Liao, C., Gong, Z., Yu, H., Li, J., and Wang, R. Unifying the perspectives of NLP and software engineering: A survey on language models for code. Transactions on Machine Learning Research, 2024b. ISSN 2835-8856.

Zhang, Z., Liao, Z., Yu, H., Di, P., and Wang, R. F2llm technical report: Matching sota embedding performance with 6 million open-source data, 2025c.

Zheng, T., Zhang, G., Shen, T., Liu, X., Lin, B. Y., Fu, J., Chen, W., and Yue, X. Opencodeinterpreter: Integrating code generation with execution and refinement. In Ku, L., Martins, A., and Srikumar, V. (eds.), Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pp. 12834–12859. Association for Computational Linguistics, 2024.

Zhou, F., Xue, S., Qi, D., Shi, W., Zhao, W., Wei, G., Zhang, H., Jiang, C., Jiang, G., Chu, Z., and Chen, F. Db-gpt-hub: Towards open benchmarking text-to-sql empowered by large language models. arXiv preprint arXiv:2406.11434, 2024.

# Appendices

#### A Dataset Details

##### A.1 Annotation Details

Code Generation. Each model receives a fixed prompt that specifies the target programming language, problem title, and full problem statement. Template variables are written as {{ }}.

You are an expert competitive−programming assistant.

Task −−−− Write an ∗entire∗, ∗runnable∗ program in {{language}} that solves the problem below.

Title : {{question title}} Statement: {{question content}}

Requirements −−−−−−−−−−−−

- 1. Provide ONLY executable source code −− no comments, no markdown, no explanations.

- 2. Implement a ‘main()‘ function that: − reads all input from stdin, − computes the answer, − writes the result to stdout.

- 3. Add any necessary helper functions (without comments).

- 4. Call ‘main()‘ at the bottom of the file.

- 5. Do not modify the supplied starter skeleton: {{starter code}}

Output Format −−−−−−−−−−−−− Wrap the final program in XML−style tags:

<code>

...your code... </code>

Generate the COMPLETE solution. Do not stop mid−function. Listing 1: Prompt template for code generation.

Counterfactual Rewriting. To reduce surface-level contamination, each problem is passed through an LLM that applies five transformations while preserving the underlying algorithmic challenge. Purely numerical test cases are never modified; non-numerical test cases receive only the minimal changes required to match the new surface context.

# Code Problem Annotation Task ## Task Description You are tasked with transforming a coding problem into a counterfactual version to avoid data contamination. Your goal is to preserve the core algorithmic challenge without changing the output while changing superficial details. ## Transformation Instructions Please create a counterfactual version of this problem by applying these transformations:

- 1. ∗∗Named Entity Replacement∗∗:

− Replace all proper nouns, character names, company names, etc. − Example: "Alice wants to sort her books"

−> "Marcus needs to organize his collection"

- 2. ∗∗Domain/Context Shifting∗∗: − Change the problem domain while keeping the algorithmic challenge

identical − Example: "Calculate profit from stock trades" −> "Determine score changes in a game tournament"

- 3. ∗∗Noun Phrase Substitution∗∗: − Replace key objects/items with different but functionally equivalent

ones − Example: "array of integers" can stay the same, but "list of books" −> "array of products"

- 4. ∗∗Synonym Replacement∗∗: − Replace verbs and adjectives with synonyms − Example: "maximize profit" −> "optimize earnings"

- 5. ∗∗Variable/Function Name Changes∗∗: − If example code is provided, rename variables and functions − Example: ‘calculateSum()‘ −> ‘computeTotal()‘

## Important Guidelines − Preserve the exact same algorithmic challenge and difficulty − Maintain the same input/output structure and constraints − Keep the same time/space complexity requirements − Ensure the transformed problem requires the same solution approach − The starter code should remain syntactically correct and functionally

equivalent

## Test Case Guidelines When handling test cases, follow these strict rules:

- 1. For purely numerical test cases (containing only numbers, basic operators, and data structures): − DO NOT MODIFY them at all − keep them exactly as they are − Example: Leave "[1, 2, 3] −> 6" or "5 + 10 = 15" unchanged

- 2. For non−numerical test cases (containing domain−specific terms): − Make MINIMAL changes necessary to match your transformed problem

context − PRESERVE the exact same algorithmic structure and complexity − Maintain the same input/output patterns and edge cases − Example: If you changed "count books on shelf" to "count tools in

box", then "books=[’novel’,’textbook’] −> 2" becomes "tools=[’hammer’,’wrench’] −> 2"

- 3. ALL test cases must: − Remain syntactically correct in the target language − Test exactly the same edge cases and functionality − Have the same expected outputs for equivalent inputs

## Your Counterfactual Version Given the original problem: Title: {{question title}} Content: {{question content}} Starter Code: {{starter code}}

Public Test Cases: {{public test cases}}

Private Test Cases: {{private test cases}}

Please provide your transformed version in JSON format with this structure: {

"annotate question title": "<transformed title>", "annotate question content": "<transformed content>", "annotate starter code": "<transformed starter code>", "annotate public test cases": "<transformed public test cases>", "annotate private test cases": "<transformed private test cases>"

###### }

Note: Ensure that your transformed version preserves all the algorithmic details while changing the superficial context. The code should remain valid and compilable.

Listing 2: Simplified prompt template for counterfactual problem rewriting. The actual implementation includes Jinja2 template conditionals for handling optional test case fields.

Question Abbreviation. To support the Canonical Retro retrieval sub-tasks, each (rewritten) problem description is further condensed into a retrieval-optimized summary of 50–150 words that retains only the core goal, key constraints, and distinctive terminology.

# Question Content Abbreviation Task ## Task Description You are an expert at creating concise, retrieval−optimized summaries of coding problem descriptions. Your goal is to distill the essential information from a problem statement into a compact format that preserves key details while removing redundancy. ## Input Question title: {{question title}} Content: {{question content}} ## Core Principles ∗∗Essential Information Preservation∗∗ − Retain the core problem goal, key constraints, and unique requirements − Preserve specific numerical limits, data types, and optimization

objectives − Keep distinctive algorithmic concepts or problem−specific terminology ∗∗Conciseness Optimization∗∗ − Remove verbose explanations, examples, and repetitive content − Eliminate unnecessary background information or motivational text − Compress similar concepts into unified statements ∗∗Retrieval−Friendly Format∗∗ − Use clear, structured language that matches how developers search − Include specific technical terms and constraints that distinguish this

problem − Maintain logical flow from problem statement to requirements ## Abbreviation Guidelines

- 1. ∗∗Problem Goal∗∗: Start with a clear, concise statement of what needs to be accomplished

- 2. ∗∗Key Constraints∗∗: Include specific limits, ranges, and requirements (e.g., "array size <= 10ˆ5", "time complexity O(n log n)")

- 3. ∗∗Input/Output Format∗∗: Specify data types and formats when critical to the problem

- 4. ∗∗Unique Requirements∗∗: Highlight distinctive aspects that differentiate this problem from similar ones

- 5. ∗∗Algorithmic Hints∗∗: Include key algorithmic concepts if they’re central to the solution approach

## Output Requirements − ∗∗Length∗∗: 50−150 words maximum − ∗∗Structure∗∗: Use clear, declarative sentences − ∗∗Precision∗∗: Include specific numerical constraints and technical

details − ∗∗Clarity∗∗: Avoid ambiguous pronouns or references − ∗∗Completeness∗∗: Ensure all essential problem−solving information is

preserved Please provide your abbreviated version in JSON format: {

"abbreviated content": "<your concise abbreviation>" }

Listing 3: Prompt template for question abbreviation.

##### A.2 Hard Negative Generation

To strengthen text-retrieval evaluation, we augment each release’s original problem descriptions with four LLM-generated hard negatives per problem, yielding 668 hard negatives in v202602 and 700 in v202603 (1,368 aggregated across releases). A hard negative is a problem description that shares surface-level vocabulary or structural similarity with the original but differs in its core algorithmic challenge, making it difficult for a retrieval model to distinguish from the true positive without deep semantic understanding.

Hard negatives are generated by prompting Qwen-32B with one of four perturbation strategies, applied independently and with varied temperature (∈ [0.85,1.0]) to encourage diversity:

- 1. Operation-type change. Alter the fundamental operation the algorithm must perform (e.g., replace a “remove / delete” goal with a “select / construct / merge” goal) so that the core algorithmic action is different, not merely renamed.
- 2. Optimization-objective change. Invert or replace the optimization criterion (e.g., change “maximize” to “minimize” or “count distinct”) so that what the algorithm optimizes for changes structurally.
- 3. Algorithmic-approach change. Replace the algorithmic paradigm required to solve the problem (e.g., subsequence reasoning → contiguous-array or graph problems; greedy → dynamic programming; two-pointer → binary search) so that the required solution strategy is qualitatively different.
- 4. Problem-domain change. Alter input data types and the problem context (e.g., strings → graphs, arrays → trees), producing a structurally distinct problem that shares no obvious surface mapping to the original.

Generated texts are post-processed with a regular-expression pass to strip LLM-produced markdown headers and formatting artifacts (e.g., “Modified Problem Description:”) before being added to the corpus. Each release’s text corpus (835 entries in v202602 and 875 in v202603; 1,710 aggregated) serves as the retrieval corpus for its code-to-text and text-to-code subtasks; see Table 12 for the per-release and aggregated composition.

Code hard negatives in qrels (v2 scheme). In addition to LLM-generated text hard negatives, the v2 qrel scheme also encodes code hard negatives directly in the relevance judgments. For textto-code and code-to-code, same-problem code solutions that failed execution tests are assigned relevance=1; when the failed pool is insufficient, correct solutions that do not qualify as positives under the current subtask constraints (e.g., wrong language for a language-specific query) are used instead. For code-to-text, the four LLM-generated noise descriptions per problem are assigned relevance=1, including texts from the same problem (highest confusability) and texts from unrelated problems as a fallback. These explicitly judged hard negatives allow a retrieval model’s score

###### TASK REL. V202602 V202603 SOURCE T2C

2 (POS) 2,742 2,814 PASSED SOLUTIONS

- 1 (HARD NEG) 3,320 3,136 FAILED SOLUTIONSA

C2T

- 2 (POS) 1,064 1,010 ORIGINAL DESCRIPTIONS

aSame-problem code

- 1 (HARD NEG) 3,810 3,600 LLM NOISE TEXTSB

C2C

- 2 (POS) 367 623 CROSS-LANG TRANSLATIONS 1 (HARD NEG) 507 834 FAILED / EXCLUDED SOLUTIONSC

TOTAL 11,810 12,017

that failed execution tests; when insufficient, correct solutions excluded by subtask constraints (e.g., wrong language). bFour LLM-generated noise descriptions per problem, prioritizing same-problem texts. cFailed solutions for the same problem, supplemented by correct solutions that do not qualify as positives (e.g., same-language pairs in a cross-language subtask).

Table 5: Graded relevance judgment counts per release under the v2 scheme. relevance=2 items are true positives; relevance=1 items are hard negatives that penalize nDCG when ranked above true positives. Unjudged corpus items act as easy negatives.

to be penalized when it ranks a plausible-but-incorrect document above a true positive, a failure mode that absent-is-irrelevant schemes cannot detect.

##### A.3 Main Tasks and Subtasks

CoREB structures evaluation as a hierarchy of main tasks and subtasks. A main task specifies the retrieval direction (i.e., what modality is used as the query and what modality constitutes the retrieval corpus). A subtask instantiates a main task under a particular configuration (e.g., query style, language constraint, or invariance setting), enabling controlled comparisons and fine-grained diagnosis.

Main task definition. Each query instance belongs to one of three main tasks, recorded in the split field:

- • text-to-code: the query is a natural language problem statement and the retrieval target is a code implementation.
- • code-to-code: the query is a code snippet and the retrieval target is another code implementation with equivalent semantics.
- • code-to-text: the query is a code implementation and the retrieval target is a natural language problem statement.

The main task determines the query modality, corpus modality, and evaluation protocol.

Subtask definition. Each query carries a subtask identifier that names the complete subtask, e.g. t2c canonical retro python or c2c cross lang. The subtask variant component captures the evaluation setting (query style, matching regime), while the language suffix, when present, specifies the language slice for that instantiation. Query-level metadata fields provide additional context (e.g., anchor language, anchor model, sub query type).

text-to-code subtasks. Three subtask variants are defined:

- • canonical retro: the query is an abbreviated problem description retaining only the core goal and key constraints, omitting narrative detail (sub query type=abbreviated).

- • full retro: the query is the full problem statement including contextual details, examples, and constraints (sub query type=full description).

- • search: the query is an LLM-generated developer-style search string. The sub query type field further distinguishes intent-focused natural language (description search), technique-focused queries naming specific algorithms or data structures (algorithm search), and broad languageagnostic queries (language agnostic).

Language slicing. Each variant is instantiated for each of the five target languages (language constraint ∈ {python, java, cpp, ruby, go}) and for an unconstrained variant (suffix * any, language constraint=none), yielding complete subtasks such as t2c canonical retro python and t2c search any. When a concrete language is specified, only corpus entries in that language are treated as valid retrieval targets; otherwise any language is accepted.

code-to-code subtasks. The current release contains one code-to-code subtask variant, using code as both query and corpus. The anchor (query) language and model are recorded in anchor language and anchor model; intended target languages are listed in meta.target languages.

• cross lang: cross-language semantic alignment. The anchor is a solution in one language (Python, Java, C++, or Go) generated by Claude Sonnet 4.5, and the task is to retrieve the semantically equivalent solution in a different target language by the same model (e.g., Java anchor → Python target, or Python anchor → Ruby target), testing cross-language transfer robustness.

Language is encoded per-query via anchor language and meta.target languages; the subtask therefore carries no language suffix.

code-to-text subtasks. Three subtask variants are defined:

- • canonical retro: retrieve abbreviated problem descriptions from code (meta.text type=canonical description).

- • full retro: retrieve full problem descriptions from code (meta.text type=full description).

- • match: code–text pair retrieval. Each query is a code snippet and the target is its corresponding problem description among the release’s full text corpus (835 entries in v202602, 875 in v202603). Queries carry a binary label (1 = positive pair, 0 = negative pair) and a meta.pair type field for analysis; however, evaluation uses the same ranked-retrieval protocol as the other subtasks (nDCG@10, Recall@10, etc.). Positive queries have exactly one relevant document; negative queries have none and are excluded from metric averages following standard IR convention.

Language slicing. For the retrieval variants (canonical retro, full retro), the anchor language is recorded in anchor language. The language-agnostic variant (* any) mixes queries from all anchor languages and is evaluated as standard description retrieval; language-specific variants fix the anchor language and probe cross-language invariance, yielding subtasks such as c2t full retro java. The match subtask is language-agnostic and carries no language suffix.

Rationale. This hierarchy isolates different sources of difficulty (linguistic variability, verbosity, keyword-style queries, and cross-language generalization) while preserving a consistent evaluation interface across tasks. As a result, CoREB supports both overall benchmarking and diagnostic analysis at the subtask level.

##### A.4 Dataset Format and Naming Convention

Query data format. The triple (main task, subtask type, language constraint) is encoded in the query identifier (not the subtask definition itself). Specifically, each query ID follows:

q {main.task} {subtask.type} {language.constraint} {index},

where main.task ∈ {t2c,c2c,c2t}, subtask.type is the subtask-type label, language.constraint ∈ {any,python,java,cpp,ruby,go}, and index is a zero-padded unique number. For example, q t2c full retro python 0001 denotes a text-to-code query of type full retro with a Python target-language constraint. Listing 4 lists the full field schema; concrete per-task examples are shown in Appendix A.5.

{

"query id": "<string>", // e.g. q t2c canonical retro any 0001

COMPLETE TASK NAME TEXT TYPE QUERY TYPE LANGUAGE COUNT CONSTRAINT

t2c canonical retro any ABBREVIATED LANGUAGE AGNOSTIC ANY 103 t2c canonical retro python ABBREVIATED LANGUAGE SPECIFIC PYTHON 79 t2c canonical retro java ABBREVIATED LANGUAGE SPECIFIC JAVA 63 t2c canonical retro cpp ABBREVIATED LANGUAGE SPECIFIC CPP 70 t2c canonical retro ruby ABBREVIATED LANGUAGE SPECIFIC RUBY 73 t2c canonical retro go ABBREVIATED LANGUAGE SPECIFIC GO 63 t2c full retro any FULL DESC LANGUAGE AGNOSTIC ANY 103 t2c full retro python FULL DESC LANGUAGE SPECIFIC PYTHON 79 t2c full retro java FULL DESC LANGUAGE SPECIFIC JAVA 63 t2c full retro cpp FULL DESC LANGUAGE SPECIFIC CPP 70 t2c full retro ruby FULL DESC LANGUAGE SPECIFIC RUBY 73 t2c full retro go FULL DESC LANGUAGE SPECIFIC GO 63 t2c search any SEARCH MIXED LANGUAGE AGNOSTIC ANY 13 t2c search python SEARCH MIXED LANGUAGE SPECIFIC PYTHON 50 t2c search java SEARCH MIXED LANGUAGE SPECIFIC JAVA 50 t2c search cpp SEARCH MIXED LANGUAGE SPECIFIC CPP 50 t2c search ruby SEARCH MIXED LANGUAGE SPECIFIC RUBY 50 t2c search go SEARCH MIXED LANGUAGE SPECIFIC GO 50

Table 6: Summary of text-to-code subtasks. COMPLETE TASK NAME TEXT TYPE QUERY TYPE ANCHOR COUNT

LANGUAGE

c2c cross lang N/A CROSS LANG PYTHON 35 c2c cross lang N/A CROSS LANG JAVA 49 c2c cross lang N/A CROSS LANG CPP 47 c2c cross lang N/A CROSS LANG GO 38

Table 7: Overview of code-to-code subtasks.

"query": "<string>", // natural−language problem description "split": "<string>", // text2code | code2code | code2text "subtask": "<string>", // e.g. t2c canonical retro any "query type": "<string>", // language agnostic | language specific "sub query type": "<string>", // abbreviated | full description | ... "language constraint": "<string>",// none | python | java | cpp | ruby | go "meta": {

"source problem id": "<string>" // originating LiveCodeBench problem ID }

###### }

Listing 4: JSON schema of a text-to-code query record. See Appendix A.5 for full per-task examples.

Text corpus data format. The text corpus record, illustrated in Listing 5, stores two aligned natural-language views of the same seed problem. The field text contains the full problem statement (used in text-to-code queries when sub query type=full description), while abbreviated text provides a condensed version for the canonical retro setting (used when sub query type=abbreviated). The attribute text style specifies the formatting template of the full statement (e.g., title plus description), and the lengths text length and abbreviated text length are tracked for analysis of length sensitivity. Finally, meta.source problem id links the entry back to the originating seed problem, enabling grouping and consistency checks across variants.

- • text provides the full problem statement (used by text-to-code queries with sub query type=full description).

- • abbreviated text is a condensed version of the same problem (used by text-to-code queries with sub query type=abbreviated).

COMPLETE TASK NAME TEXT TYPE QUERY TYPE ANCHOR COUNT LANGUAGE

c2t canonical retro any CANONICAL DESC DESCRIPTION RETRIEVAL PYTHON 79 c2t canonical retro any CANONICAL DESC DESCRIPTION RETRIEVAL CPP 9 c2t canonical retro any CANONICAL DESC DESCRIPTION RETRIEVAL RUBY 6 c2t canonical retro any CANONICAL DESC DESCRIPTION RETRIEVAL GO 6 c2t canonical retro any CANONICAL DESC DESCRIPTION RETRIEVAL JAVA 3 c2t canonical retro python CANONICAL DESC CROSS LANGUAGE INVARIANCE PYTHON 67 c2t canonical retro java CANONICAL DESC CROSS LANGUAGE INVARIANCE JAVA 63 c2t canonical retro cpp CANONICAL DESC CROSS LANGUAGE INVARIANCE CPP 67 c2t canonical retro ruby CANONICAL DESC CROSS LANGUAGE INVARIANCE RUBY 68 c2t canonical retro go CANONICAL DESC CROSS LANGUAGE INVARIANCE GO 61 c2t full retro any FULL DESC DESCRIPTION RETRIEVAL PYTHON 79 c2t full retro any FULL DESC DESCRIPTION RETRIEVAL CPP 9 c2t full retro any FULL DESC DESCRIPTION RETRIEVAL RUBY 6 c2t full retro any FULL DESC DESCRIPTION RETRIEVAL GO 6 c2t full retro any FULL DESC DESCRIPTION RETRIEVAL JAVA 3 c2t full retro python FULL DESC CROSS LANGUAGE INVARIANCE PYTHON 67 c2t full retro java FULL DESC CROSS LANGUAGE INVARIANCE JAVA 63 c2t full retro cpp FULL DESC CROSS LANGUAGE INVARIANCE CPP 67 c2t full retro ruby FULL DESC CROSS LANGUAGE INVARIANCE RUBY 68 c2t full retro go FULL DESC CROSS LANGUAGE INVARIANCE GO 61 c2t match N/A PAIR MATCH RETRIEVAL PYTHON 316 c2t match N/A PAIR MATCH RETRIEVAL CPP 36 c2t match N/A PAIR MATCH RETRIEVAL GO 24 c2t match N/A PAIR MATCH RETRIEVAL RUBY 24 c2t match N/A PAIR MATCH RETRIEVAL JAVA 12

Table 8: Summary of code-to-text subtasks. MAIN TASK SUBTASK V202602 V202603

CANONICAL RETRO 123 117 FULL RETRO 448 414 SEARCH 25 13

T2C

CROSS-LINGUAL 215 200 CROSS-MODEL — 182 MONO-LANG — 200

C2C

CANONICAL RETRO 255 238 FULL RETRO 255 238 PAIR MATCH 259 244

C2T

- Table 9: Per-release average query length (tokens, cl100k base) by subtask. “—” indicates a subtask absent from that release. Cross-Model and Mono-Lang are v202603-only C2C subtasks. Aggregated values are reported in the main text (section 3).

- • text style indicates the formatting template of the full text (e.g., title + description).

- • text length and abbreviated text length track query length for analysis and stratification.

- • meta.source problem id links the text entry back to the original seed problem, enabling grouping and deduplication across variants.

{

"text id": "text v202601 00001", "text style": "title plus description", "text": "Determine Maximum Removals from Playlist While Preserving Favorite

Sequence\n"

###### TASK SUB-TASK ANY PY JAVA C++ RUBY GO TOTAL

CANONICAL RETRO 103 79 63 70 73 63 451 FULL RETRO 103 79 63 70 73 63 451 SEARCH 13 50 50 50 50 50 263

TEXT-TO-CODE

CANONICAL RETRO 103 67 63 67 68 61 429 FULL RETRO 103 67 63 67 68 61 429

CODE-TO-TEXT

TOTAL 425 342 302 324 332 298 2,023

- Table 10: Per-language query counts for language-variant sub-tasks in COREB. Any = language-agnostic variant. Variation in counts across languages reflects the available evaluated solutions per problem.

DIMENSION CATEGORY V202602 V202603 AGGREGATED

MAIN TASK

TEXT-TO-CODE 1,165 1,117 2,282 CODE-TO-CODE 169 166 335 CODE-TO-TEXT 1,270 1,200 2,470

LANGUAGE

ANY 425 391 816 PYTHON 693 635 1,328 JAVA 363 317 680 C++ 407 403 810 RUBY 356 344 700 GO 360 393 753

SUBTASK VARIANT

CANONICAL RETRO 880 838 1,718 FULL RETRO 880 838 1,718 SEARCH 263 261 524 CROSS-LINGUAL 169 166 335 PAIR MATCH 412 380 792

TOTAL 2,604 2,483 5,087

- Table 11: Detailed query counts for each COREB release and the aggregated total, broken down by main task, language constraint, and subtask variant. Supplements the high-level corpus overview in Table 2; visualised in Figures 1 and 11.

"You are given a playlist ‘playlist‘ of size n, a list ‘ favoriteSequence‘ that is a subsequence of ‘playlist‘, "

"and a sorted integer array ‘removableIndices‘ with distinct values in [0, n−1].\n"

"An operation removes a song at index idx such that idx is in ‘

removableIndices‘ and ‘favoriteSequence‘ remains " "a subsequence after removal.\n" "Performing an operation does not change the indices of the other songs

in ‘playlist‘...\n", "text length": 2076, "abbreviated text": "Given a playlist ‘playlist‘ of size n, a subsequence ‘

favoriteSequence‘, and a sorted array "

"‘removableIndices‘ of distinct indices in [0, n−1], find the maximum number of songs that can be "

"removed while ensuring ‘favoriteSequence‘ remains a

subsequence...", "abbreviated text length": 562, "meta": {

"source problem id": "lcb 3487" }

}

- Listing 5: Example text corpus record for text-to-code: full description and its abbreviated form with metadata.

###### 45%

###### 49%

6%

Text-to-Code (T2C) Code-to-Code (C2C) Code-to-Text (C2T)

| |
|---|

| |
|---|

| |
|---|

(a) v202602 — by main task

14%

16%

14%

27%

16%

14%

Any Python Java

C++ Ruby Go

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

(b) v202602 — by language

6% 16%

34%

10%

34%

Canonical Retro Full Retro Search Cross-Lingual Binary Match

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

(c) v202602 — by subtask

###### 45%

###### 48%

7%

Text-to-Code (T2C) Code-to-Code (C2C) Code-to-Text (C2T)

| |
|---|

| |
|---|

| |
|---|

(d) v202603 — by main task

16%

16%

14%

26%

16%

13%

Any Python Java

C++ Ruby Go

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

(e) v202603 — by language

7% 15%

34%

11%

34%

Canonical Retro Full Retro Search Cross-Lingual Binary Match

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

(f) v202603 — by subtask

Figure 11: Query distribution per COREB release: v202602 (2,604 queries, top row) and v202603 (2,483 queries, bottom row). The two releases exhibit nearly identical proportions; see Table 11 for exact counts.

Code corpus data format. The code-corpus record shown in Listing 6 represents a single candidate implementation tied to a seed problem via meta.source problem id. The raw program text is stored in code, while language and model identify the programming language and the generator (or source) of the solution, which enables code-to-code evaluation along the cross-language axis. The field code length is recorded for length-based analyses. Finally, meta captures execution results from the unit-test harness (passed, pass rate, test passed/test total). The corpus retains all 1,744 generated candidates, both correct and incorrect, so that the passed field can be used for task-specific relevance labeling: code-to-code tasks restrict positives to verified-correct solutions, while text-to-code tasks treat all solutions for the queried problem as relevant (see ??).

{

"code id": "code v202601 00001", "code": "using namespace std;\n\nclass Solution {\npublic:\n int maxRemovals

(string playlist, string favoriteSequence, vector<int>& removableIndices) {\n

int n = playlist.length();\n int m = favoriteSequence.length();\n \n auto canRemove = [&](int k) −> bool {\n set<int>

###### CORPUS TYPE V202602 V202603 AGGREGATED

PYTHON 334 349 683 JAVA 334 348 682 C++ 334 348 682 GO 334 349 683 RUBY 334 350 684

CODE CORPUS

###### TOTAL 1,670 1,744 3,414

ORIGINAL DESCRIPTIONS 167 175 342 ABBREVIATED DESCRIPTIONS 167 175 342 HARD NEGATIVES 668 700 1,368

TEXT CORPUS

###### TOTAL 835 875 1,710

- Table 12: Composition of code and text corpora in COREB per release and aggregated across both releases. Each release regenerates candidates with its own code models and draws from a freshly sourced LiveCodeBench snapshot, so aggregated artifacts are sums of distinct records (no cross-release ID overlap). In v202603, 537 of 1,744 generated code candidates pass every test case (verified-correct); the remainder carry passed = false metadata (see Appendix A.8 for the per-model breakdown). Hard negatives are generated with four strategies (operation-type change, objective change, algorithmic-approach change, and problem-domain change); see Appendix A.2.

removed;\n for (int i = 0; i < k; i++) {\n removed. insert(removableIndices[i]);\n }\n \n int j = 0;\n for (int i = 0; i < n && j < m; i++) {\n if ( removed.count(i)) continue;\n if (playlist[i] == favoriteSequence[ j]) {\n j++;\n }\n }\n return j == m;\n };\n \n int left = 0, right = removableIndices.size();\n int result = 0;\n \n while (left <= right) {\n int mid = (left + right) / 2;\n if (canRemove (mid)) {\n result = mid;\n left = mid + 1;\n

} else {\n right = mid − 1;\n }\n }\n \n return result;\n }\n};\n\nint main() {\n string playlist,

favoriteSequence;\n cin >> playlist >> favoriteSequence;\n \n int k;\n

cin >> k;\n vector<int> removableIndices(k);\n for (int i = 0; i < k; i++) {\n cin >> removableIndices[i];\n }\n \n Solution sol;\n

cout << sol.maxRemovals(playlist, favoriteSequence, removableIndices) << endl

;\n \n return 0;\n}\n\nmain();", "language": "cpp", "model": "claude−sonnet−4−5", "code length": 1460, "meta": {

"source problem id": "lcb 3487", "solution key": "claude−sonnet−4−5 cpp", "passed": false, "pass rate": 0.0, "test passed": 0, "test total": 1

} }

- Listing 6: Example code-corpus record: a candidate solution with language/model metadata and execution outcomes. This entry has passed = false; the corpus retains such entries alongside verified-correct ones (see text).

##### A.5 Task Instance Examples

The following listings show representative query instances and their relevance judgments (qrels) for each of the three main tasks. All examples are drawn directly from the CoREB dataset. Code snippets are truncated for readability; the full texts are available in the released data files.

text-to-code. text-to-code queries are natural-language problem descriptions targeting a code corpus. Listing 7 shows an abbreviated (canonical retro) query with no language constraint. Listing 8 shows a short LLM-generated developer-style search query restricted to Python.

// −−− Query −−− {

"query id": "q t2c canonical retro any 0001", "query": "Given m arenas and t days, find maximum points a player can

accumulate. Each day the player can stay (earn restBonus[i][current]) or switch arena (earn switchBonus[current][dest]). Player may start at any arena. Constraints: 1<=m,t<=200.",

"split": "text2code", "subtask": "t2c canonical retro any", "query type": "language agnostic", "sub query type": "abbreviated", "language constraint": "none", "meta": { "source problem id": "lcb 3587" }

} // −−− Qrel −−− { "query id": "q t2c canonical retro any 0001",

"doc id": "code v202601 00028", "relevance": 1 }

- Listing 7: text-to-code example: abbreviated (canonical retro) query with language-agnostic constraint and its qrel.

// −−− Query −−− {

"query id": "q t2c search python 0787", "query": "Find the Kth character after repeatedly inverting case

and concatenating a string 10ˆ100 times", "split": "text2code", "subtask": "t2c search python", "query type": "language specific", "sub query type": "language agnostic", "language constraint": "python", "meta": { "source problem id": "lcb abc380 d" }

} // −−− Qrel −−− { "query id": "q t2c search python 0787",

"doc id": "code v202601 00...", "relevance": 1 }

- Listing 8: text-to-code example: LLM-generated developer-style search query with Python language constraint and its qrel.

code-to-code. code-to-code queries are code snippets; the goal is to retrieve semantically equivalent implementations. The current release contains only the cross-language subtask. Listing 9 shows a cross-language query (Java anchor by Claude Sonnet 4.5, Python target).

// −−− Query −−− {

"query id": "q c2c cross lang 0002", "query": "import java.util.∗;\n\npublic class Solution {\n public

static void main(String[] args) {\n Scanner sc = new Scanner(System.in);\n ... (truncated)",

"split": "code2code", "subtask": "c2c cross lang", "anchor language": "java", "anchor model": "claude−sonnet−4−5", "meta": {

"source problem id": "lcb 3616",

"anchor code id": "code v202601 00173", "target languages": ["python"], "num target solutions": 1

###### }

} // −−− Qrel −−− { "query id": "q c2c cross lang 0002", "doc id": "code v202601 00174", "relevance": 1 }

- Listing 9: code-to-code example: cross-language query (Java anchor by Claude Sonnet 4.5, Python target) and its qrel.

code-to-text. code-to-text queries are code snippets; the retrieval target is a natural-language problem description. Listing 10 shows a full-retro retrieval query. Listing 11 shows a pair-match instance, which provides a pre-paired (code, text) with a binary label; evaluation still ranks the full text corpus.

// −−− Query −−− {

"query id": "q c2t full retro any 0001", "query": "from typing import List\n\nclass Solution:\n def

maxPoints(self, m: int, t: int, restBonus: List[List[int]], switchBonus: List[List[int]]) −> int:\n dp = [[0]∗m for

in range(t+1)]\n ... (truncated)", "split": "code2text", "subtask": "c2t full retro any", "anchor language": "python", "anchor model": "claude−sonnet−4−5", "meta": {

"source problem id": "lcb 3587", "anchor code id": "code v202601 00028", "task type": "description retrieval", "text type": "full description"

###### }

} // −−− Qrel −−− { "query id": "q c2t full retro any 0001",

"doc id": "text v202601 00001", "relevance": 1 }

Listing 10: code-to-text example: full-retro retrieval query (Python, language-agnostic) and its qrel.

###### {

"query id": "q c2t match 0641", "code": "from typing import List\n\nclass Solution:\n def

maxPoints(self, m: int, t: int, ...) −> int:\n ... (truncated)",

"text id": "text v202601 00001", "split": "code2text", "subtask": "c2t match", "label": 1, "text type": "full", "anchor language": "python", "anchor model": "claude−sonnet−4−5", "meta": {

"source problem id": "lcb 3587", "anchor code id": "code v202601 00028", "pair type": "positive"

} }

Listing 11: code-to-text example: pair-match instance. The code and a specific text entry are pre-paired; label=1 means they correspond (positive pair). Evaluation still follows the standard retrieval protocol: the code is used

- as a query against the full text corpus.

- A.6 Evaluation Protocol and Metrics All subtasks, including c2t match, are evaluated with a single, uniform ranked-retrieval protocol. Given a query, every model encodes it and ranks all items in the corresponding corpus by cosine similarity; the resulting ranked list is scored with nDCG@k (k=1,3,5,10), MAP@k (k=10,100), Recall@k (k = 1,3,5,10,100), MRR@k (k = 10), and Precision@k (k = 1,3,5,10). Negativepair instances in c2t match have only hard-negative qrels (rel=1) and no true positives (rel=2); under relevance level=2 they contribute a score of zero, acting as a fixed penalty that rewards models only for the positive-pair and retro subtasks.

Graded relevance and evaluation strictness. Starting from COREB v2, qrels use a three-level graded relevance scheme: relevance=2 (true positive), relevance=1 (hard negative), and absent (easy negative / unjudged). All metrics are computed with relevance level=2: standard binary metrics (Recall, MAP, Precision, MRR) treat only rel=2 items as relevant. For nDCG, hard negatives are zeroed to rel=0 before scoring so that they contribute zero gain regardless of rank, yet they still penalize by occupying positions that true positives could occupy. This design is strictly harder than benchmarks using binary qrels (e.g., BEIR, CoIR, MTEB) where all explicitly judged items count as relevant: under those schemes, a model retrieving a failed code solution or a noise description would receive credit; under our graded scheme, it would not. Evaluation is performed via pytrec eval with the filtered qrels passed as-is to the relevance evaluator.

code-to-code anchor exclusion. For the code-to-code subtask, the anchor code item (i.e., the exact code snippet used as the query) is always present in the shared retrieval corpus. Because every model assigns near-perfect similarity to the anchor, it is trivially retrieved at rank 1 without providing any signal about cross-language retrieval ability. We therefore remove the anchor from each query’s ranked list before computing any metric; the reported nDCG@k, Recall@k, and MRR@k values for code-to-code thus reflect retrieval quality over the remaining corpus positions (see Appendix B.9 for full details).

- A.7 Experimental Validation: Effect of Annotation

To assess how our annotation pipeline affects evaluation reliability, we run a controlled 2×2 comparison: two frontier models (Gemini 3 Flash and Claude Sonnet 4.5) evaluated on two LiveCodeBench releases (release v5 / v202602 and release v6 / v202603). The two releases cover non-overlapping contest windows: v202602 spans Sep 2024–Jan 2025, while v202603 spans Jan 2025–Apr 2025. Both models share a nominal knowledge cutoff of January 2025: Gemini 3 Flash (released Dec 17, 2025; knowledge cutoff Jan 2025 (Google, 2025)) and Claude Sonnet 4.5 (released Sep 29, 2025; reliable knowledge cutoff Jan 2025, training data cutoff Jul 2025 (Anthropic, 2025)). For each problem, we prompt each model using listing 1 to generate a single solution in each target language and score it against the corresponding test suite. We report Pass@1, i.e., the fraction of problems for which the generated solution passes all test cases.

Results. Table 13 reports Pass@1 on original and annotated problems across all four model– release combinations.

Gemini 3 Flash shows a consistent drop in both releases: −8.6 points on v202602 (43% → 34%) and −6.5 points on v202603 (38% → 31%), across all five languages without exception. Because v202602 problems fall squarely within Gemini’s training window (before Jan 2025), leakage is strongest there; the smaller but still consistent drop on v202603 (starting Jan 4, 2025) suggests Gemini’s training data includes problems right up to its nominal cutoff.

Claude Sonnet 4.5 reveals a striking asymmetry: Pass@1 drops by 12.3 points on v202602 (35% → 22%) — a larger drop than Gemini on the same release — yet shows virtually no change on v202603 (29% → 31%, +2.0 points). This dissociation is directly explained by Claude’s reliable knowledge cutoff of January 2025: the v202602 contest problems (ending Jan 4, 2025) lie entirely within Claude’s reliably memorized window, whereas the v202603 problems (starting Jan 4, 2025) fall beyond it, yielding near-zero leakage.

Taken together, the two models yield complementary evidence. Gemini leaks on both releases; Claude leaks strongly on one but not the other. The pattern is model-dependent and releasedependent, and unpredictable without a controlled test: a benchmark builder cannot know a priori which problems any given model has memorized. This is precisely why counterfactual annotation is necessary. Using original problem statements in an embedding benchmark would introduce an uncontrolled performance bias; our rewriting pipeline eliminates it across both models and both releases.

Two secondary factors may also contribute modestly: (i) reformulation can reduce underspecification present in original statements, and (ii) test-suite validation during annotation eliminates solutions that passed only because of weak original tests.

LANGUAGE ORIG ANN ∆

- Gemini 3 Flash v202602 (Sep 2024–Jan 2025) PYTHON .491 .377 −.114 JAVA .293 .251 −.042 C++ .425 .383 −.042 GO .407 .305 −.102 RUBY .515 .383 −.132 OVERALL .426 .340 −.086

- Gemini 3 Flash v202603 (Jan 2025–Apr 2025) PYTHON .411 .337 −.074 JAVA .274 .217 −.057 C++ .360 .349 −.011 GO .354 .314 −.040 RUBY .480 .337 −.143 OVERALL .376 .311 −.065

- Claude Sonnet 4.5 v202602 (Sep 2024–Jan 2025) PYTHON .395 .246 −.150 JAVA .281 .180 −.102 C++ .347 .246 −.102 GO .371 .240 −.132 RUBY .341 .210 −.132 OVERALL .347 .224 −.123

- Claude Sonnet 4.5 v202603 (Jan 2025–Apr 2025) PYTHON .345 .326 −.019 JAVA .241 .257 +.016 C++ .259 .286 +.027 GO .276 .326 +.050 RUBY .306 .320 +.014 OVERALL .285 .303 +.017

- Table 13: Pass@1 on original vs. annotated problems across a 2×2 design (two models × two releases). ∆ = Ann − Orig. Gemini 3 Flash drops consistently in both releases; Claude Sonnet 4.5 drops sharply on v202602 (older problems, likely in training data) but shows no drop on v202603 (newer problems, likely past training cutoff). The model-dependent, release-dependent pattern demonstrates that data leakage is unpredictable without controlled annotation.

##### A.8 Code Solution Generation

The code corpus underlying COREB is regenerated in each release by two frontier LLMs: Gemini 3 Flash and Claude Sonnet 4.5 For each annotated problem, both models are prompted to produce a single solution in each of the five target languages (Python, Java, C++, Go, Ruby). In v202603, this yields 1,744 candidate solutions (the nominal 175×5×2 = 1,750 minus 6 missing model–language combinations); in v202602, the corresponding count is 1,670 (the nominal 167×5×2 = 1,670, all combinations present). Every candidate is executed against the full test suite; the outcome (passed, pass rate) is recorded as metadata. All candidates are retained so that each problem has a (nearly)

complete set of solutions across languages and models. The verified-correct counts (passing every test) are 528 of 1,670 in v202602 (31.6%) and 537 of 1,744 in v202603 (30.8%), for a total of 1,065 verified-correct solutions aggregated across releases. Relevance labels are task-specific: code-to-code tasks restrict positives to verified-correct solutions, while text-to-code tasks treat all solutions for the queried problem as relevant.

The two models exhibit complementary strengths across languages (Table 13), ensuring that few problems are left without any correct solution and that every language is represented in the verifiedcorrect subset used for code-to-code relevance labeling.

#### B Experiment Details

##### B.1 Implementation Details

For the analysis and data processing framework, we use the code from the public Github repository at https://github.com/wuji3/Doraemon (Du et al., 2025) with GNU General Public License.

For the evaluation framework, we use the code from the public Github repository at https://github.com/beir-cellar/beir (Thakur et al., 2021) and https://github.com/ SerendipityOneInc/look-bench (Gao et al., 2026; Xue et al., 2026) with Apache License and https://github.com/eosphoros-ai/DB-GPT-Hub (Xue et al., 2023, 2024; Zhou et al., 2024) with MIT License.

##### B.2 Inference and Evaluation Details

All embedding models are evaluated in inference-only mode; no fine-tuning or adaptation is performed on CoREB data. For each model, we encode all corpus documents and queries with the model’s default pooling strategy (e.g., CLS token, mean pooling, or instruction-prefixed mean pooling as specified by the model authors). Similarity scores are computed via dot product or cosine similarity, depending on each model’s recommended configuration. Retrieval rankings are produced by scoring every (query, corpus entry) pair; dense nearest-neighbor search is used for efficiency where corpus size permits. All experiments are run on a single NVIDIA RTX 4090 GPU with 24GB VRAM. We report nDCG@10 and Recall@10 as primary metrics; full metric tables including MRR@10, Recall@k (k∈{1,3,5,10,100}), and nDCG@k (k∈{1,3,5,10}) are available via the project repository.

##### B.3 Fine-Tuned COREB-RERANKER Training Protocol

This appendix documents the full fine-tuning protocol for the COREB-RERANKER (4B) model introduced in section 4.3. The checkpoint will be released alongside the benchmark.

Architecture and base models. We initialise from the publicly released Qwen3-Reranker-4B model (Zhang et al., 2025b). Unlike conventional cross-encoders with a scalar classification head, Qwen3-Reranker is used as a causal-LM reranker: for each (query, document) pair, the model is prompted to answer whether the document satisfies the query, and relevance is represented by the probability assigned to the verbal answer yes. We retain the released tokenizer, chat template, instruction format, and architectural configuration. Fine-tuning is parameter-efficient: we attach LoRA adapters to all linear modules with rank and alpha to 16 and dropout 0.05, while keeping the base model weights frozen.

Training data construction. Training uses a mixed reranker corpus consisting of CoREB v202602, CodeSearchNet (including code-to-code, code-to-text, and text-to-code retrieval (Husain et al., 2019; Li et al., 2025; Lu et al., 2021), APPS (Hendrycks et al., 2021), CosQA (Huang et al., 2021a), and single-turn and multi-turn CodeFeedback (Zheng et al., 2024). Each JSONL record is normalized into independent binary reranking examples of the form

(instruction,q,d,y), y ∈ {yes,no}. For each source record, positive documents are duplicated twice, and we sample one easy negative and one hard negative when available. Positive examples are labeled yes, while both easy and hard negatives are labeled no. The resulting examples are shuffled across tasks before training. We reserve a small held-out split for validation.

Prompt format and loss function. We fine-tune Qwen3-Reranker in its generative yes/no reranking format. Each training example is serialized with the following system prompt:

Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no"

The user message is formatted as

<Instruct>: {instruction} <Query>: {query} <Document>: {document}

where the instruction is task-specific. For text-to-code (T2C), we use:

Given a natural language programming task, retrieve code that correctly solves or implements the task.

For code-to-code (C2C), we use:

Given a code snippet, retrieve code that is semantically equivalent or solves the same task.

For code-to-text (C2T), we use:

Given a code snippet, retrieve the natural language description or problem statement that best matches the code.

The assistant response is the verbal label y ∈ {yes,no}, with positives labeled yes and both easy and hard negatives labeled no. Training optimizes the causal language-modeling cross-entropy only over the assistant answer region:

1 |M| t∈M

log pθ(xt | x<t),

L = −

where M masks out all system and user prompt tokens and keeps only the answer tokens. Thus, hard negatives and easy negatives are not placed in a shared listwise denominator; they contribute

- as independent no-labeled pairwise training examples.

Optimization. We fine-tune for 2 epochs with AdamW (β1=0.9, β2=0.999, ϵ=10−8, weight decay 0.01), a peak learning rate of 1×10−5, linear warm-up for 100 configured steps, and a linear learning-rate decay schedule. Training uses bf16, FlashAttention-2, gradient checkpointing, DeepSpeed ZeRO-2 through accelerate, and a per-device batch size of 12 with gradient accumulation 1. The maximum sequence length is 4,096 tokens, with right-side truncation and left padding. We train the reranker on 8 NVIDIA A100 GPUs.

Checkpoint merging. The released COREB-RERANKER checkpoint is the uniform model soup (Wortsman et al., 2022) of two independently fine-tuned LoRA variants. Both variants share the same base model, LoRA configuration, optimizer, and training corpus, and differ only in random seed and data shuffle order. The soup is obtained by averaging their LoRA adapter weights with equal coefficients before merging into the base model. We selected the two variants by validation nDCG@10 on a held-out split prior to averaging; the soup outperforms either individual checkpoint on the mean of the three test tasks while matching their per-task best on T2C and C2T.

Evaluation. At test time, we (i) run C2LLM-7B retrieval on the v202603 corpus to obtain the top-128 candidates per query, (ii) score each (query, candidate) pair with the trained reranker, and (iii) re-sort by the reranker score before computing nDCG@10 and Recall@10 against the graded qrels (Appendix B.4). The reranker never sees v202603 problems, queries, or qrels during training, so the evaluation is fully out-of-sample at the problem level.

##### B.4 Evaluation Metrics

Recall@k. Recall is a commonly used metric in retrieval tasks, measuring the proportion of relevant items successfully retrieved within the top-k results. Formally, given a query with R relevant items in the dataset, Recall@k is defined as:

k

1 R

rel(i), (1)

Recall@k =

i=1

where rel(i) ∈ {0,1} indicates whether the i-th retrieved item is relevant. Since Recall@k evaluates coverage rather than ordering, it ranges between 0 and 1 and increases monotonically with k.

nDCG@k. Normalized Discounted Cumulative Gain (nDCG) is a weighted ranking metric that evaluates the quality of the ordering of retrieved items. The discounted cumulative gain at rank k (DCG@k) is computed as:

k

rel(i) log2(i + 1)

, (2)

DCG@k =

i=1

where rel(i) ∈ {0,1} denotes the relevance of the i-th retrieved item. The ideal DCG (IDCG@k) is defined as the maximum obtainable DCG@k:

min(k,R)

1 log2(i + 1)

, (3)

IDCG@k =

i=1

where R is the total number of relevant items for the query. nDCG@k is then obtained by normalizing DCG@k:

DCG@k IDCG@k

, (4)

nDCG@k =

which ensures that nDCG@k lies in the range [0,1]. MRR. Mean Reciprocal Rank (MRR) evaluates the position of the first relevant retrieved item. For a set of Q queries, MRR is computed as:

Q

1 rank(i)

1 Q

, (5)

MRR =

i=1

where rank(i) is the position of the first relevant result for the i-th query. If no relevant item appears in the retrieved list, we set rank(1 i) = 0. MRR ranges between 0 and 1 and is particularly sensitive to early retrieval performance.

##### B.5 Per-Release Retrieval Results

The main text reports results on v202603; here we provide the corresponding v202602 results and a cross-release comparison. Table 14 reports per-task nDCG@10 and Recall@10 for the eleven models evaluated on v202602.

TEXT-TO-CODE CODE-TO-TEXT CODE-TO-CODE† OVERALL NDCG RECALL NDCG RECALL NDCG RECALL NDCG RECALL

MODEL

C2LLM-7B 0.435 0.765 0.822 0.838 0.661 0.998 0.639 0.815 C2LLM-0.5B 0.429 0.713 0.799 0.833 0.664 0.978 0.625 0.789 GEMEMB-2 0.420 0.755 0.764 0.838 0.709 1.000 0.607 0.811 JINA-CODE-EMB-1.5B 0.405 0.713 0.767 0.827 0.686 0.976 0.600 0.786 F2LLM-4B 0.400 0.694 0.788 0.825 0.515 0.839 0.597 0.767 JINA-CODE-EMB-0.5B 0.397 0.679 0.742 0.808 0.699 0.980 0.585 0.761 QWEN3-EMB-4B 0.399 0.651 0.763 0.813 0.386 0.713 0.576 0.734 F2LLM-1.7B 0.377 0.625 0.761 0.819 0.408 0.668 0.567 0.722 F2LLM-0.6B 0.348 0.583 0.742 0.795 0.336 0.576 0.540 0.686 QWEN3-EMB-8B 0.341 0.551 0.726 0.786 0.299 0.535 0.526 0.664 QWEN3-EMB-0.6B 0.350 0.579 0.665 0.757 0.419 0.719 0.508 0.675

†Anchor excluded; Overall is query-count-weighted.

Table 14: Per-task retrieval results on COREB v202602. Format matches Table 3.

Cross-release comparison. For the nine models evaluated on both releases, rankings are largely stable: C2LLM-7B leads on both, and the top-5 share the same members. Per-task nDCG@10 differences are small (median |∆| = 0.02), with no model changing by more than 0.08 on any single task. Text-to-code scores are slightly higher on v202603 for most models, while code-to-text scores are slightly lower; these shifts likely reflect differences in the underlying problem distributions rather than systematic difficulty changes. The consistency across temporally disjoint releases supports the stability of COREB as an evaluation instrument.

##### B.6 Subtask and Language Analysis

[Figure 5]

C2LLM-7B C2LLM-0.5B

0.99 0.99 0.50 0.96 0.96 0.48 0.95 0.95 0.48 0.94 0.94 0.47 0.91 0.91 0.46

1.0

|[Figure 6]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

F2LLM-4B Jina-code-1.5B Jina-code-0.5B

0.9

0.8

- F2LLM-0.6B

EmbGemma

- F2LLM-1.7B Jina-v4

- 0.91 0.91 0.46
- 0.92 0.92 0.46 0.85 0.85 0.44 0.89 0.89 0.46 0.88 0.88 0.45

0.7

0.6

0.5

Qwen3-0.6B Qwen3-8B Qwen3-4B

- 0.87 0.87 0.43
- 0.88 0.88 0.43

0.4

Canonical Full Match

Figure 12: Code-to-text nDCG@10 by subtask type on v202603.

How do models handle short keyword queries? The text-to-code Search subtask uses short developer-style queries (19 tokens on average). On this subtask every model collapses to nearzero nDCG@10, two orders of magnitude below the Canonical subtask (Table 4). The gap stems from short queries carrying too little context for dense retrieval, not from corpus difficulty (all subtasks share the same 1,744-item corpus). Going from Canonical to Full (448 tokens) yields at most a marginal delta. Current models have saturated in the long-query regime while the short-query regime, which most closely mirrors real developer search, remains unsolved.

Does target language affect retrieval quality? Text-to-code performance drops sharply when queries specify a target language rather than accepting any language (Figure 6). Averaged across models and non-search subtasks, language-agnostic queries reach 0.714 nDCG@10, while language-specific constraints reduce this to 0.649 for Python, 0.514 for Java, 0.465 for C++, and 0.383–0.387 for Ruby and Go. Part of this gap is a confound: “any” queries accept solutions in all five languages and therefore have more valid positives per query. However, a residual bias remains: every model scores lowest on Ruby and Go, regardless of family or scale. The gradient tracks training-corpus coverage: Python and Java dominate publicly available code, so models embed those languages more faithfully.

##### B.7 Per-Subtask and Per-Language Detailed Results

This section provides fine-grained breakdowns that complement the aggregate numbers in the main text. Table 15 reports code-to-code nDCG@10 stratified by the anchor language, and Table 16 summarizes the corpus composition and token-length statistics.

##### B.8 Full Parameter Efficiency Figure

Figure 13 presents the complete three-panel efficiency analysis. The main text (Figures 9 and 10) shows only the Pareto frontier scatter plot and the top/bottom efficiency bars; the figure below adds per-task scatter and the full bar ranking.

##### B.9 Anchor Exclusion in Code-to-Code Evaluation

The code-to-code cross-language subtask has a structural property that requires explicit handling in the evaluation. Each query is a code snippet in a source language; by construction, the anchor code item the query was generated from is also present in the shared retrieval corpus. Every embedding model assigns near-perfect cosine similarity (≥ 0.99) to the anchor, which is byte-for-byte identical to the query, placing it at rank 1 without exception. However, the anchor is not a positive in the qrels (only cross-language translations are labeled relevant), so counting it in the ranked list would assign every model a structurally wasted rank 1, collapsing nDCG@1 and MRR@1 to zero universally and masking genuine differences in cross-language retrieval ability. To prevent this artifact, COREB’s

MODEL PYTHON JAVA C++ GO GEMEMB-2 0.681 0.674 0.789 0.673

- JINA-CODE-0.5B 0.690 0.493 0.788 0.684
- JINA-CODE-1.5B 0.688 0.499 0.790 0.650 C2LLM-7B 0.639 0.510 0.799 0.690 C2LLM-0.5B 0.652 0.576 0.729 0.662 F2LLM-4B 0.534 0.276 0.648 0.448 QWEN3-4B 0.498 0.174 0.504 0.186 QWEN3-0.6B 0.488 0.147 0.445 0.244 F2LLM-1.7B 0.485 0.137 0.512 0.192 F2LLM-0.6B 0.482 0.105 0.357 0.113 QWEN3-8B 0.446 0.064 0.390 0.137

- Table 15: code-to-code nDCG@10 by anchor language (after anchor exclusion). Each value is the mean over all queries whose anchor is in the given language.

V202602 V202603 AGGREGATED

CORPUS #ITEMS AVG. TOK #ITEMS AVG. TOK #ITEMS AVG. TOK CODE 1,670 371 1,744 401 3,414 386 TEXT‡ 835 687 875 667 1,710 677

ORIGINAL 167 461 175 464 342 463 NOISE 668 744 700 718 1,368 731

‡Text corpora: originals per release (167 / 175) + LLM-generated hard negatives (668 / 700); Aggregated columns sum across both releases.

- Table 16: Corpus statistics for COREB per release and aggregated; token counts from cl100k base (tiktoken). “Avg. tok” is the average token length per item, computed over that release’s items (and weighted across releases in the aggregated column).

code-to-code evaluation removes the anchor from each query’s ranked list before computing metrics, so rank 1 in every reported result corresponds to the model’s best genuine retrieval candidate.

With anchor exclusion, nDCG@1 becomes a meaningful first-rank precision signal. Jina-code-0.5b achieves the highest nDCG@1 (0.378) among evaluated models, followed by Jina-code-1.5b (0.356) and GemEmb-2 (0.331); C2LLM-7B reaches 0.327 at rank 1. By contrast, Qwen3-Emb-8B reaches nDCG@1 = 0.155, rarely returning the correct translation first. The sharp Recall@k curve for code-to-code (Figure 14) therefore reflects genuine cross-language retrieval behavior: GemEmb-2 reaches Recall@10 = 1.000 on this subtask (with nDCG@10 = 0.698), while C2LLM-7B achieves Recall@10 = 0.997 (nDCG@10 = 0.659) and Qwen3-Emb-8B lags behind (Recall@10 = 0.450, nDCG@10 = 0.320). These diverging nDCG and Recall values reveal that the primary bottleneck on this subtask is not retrieval coverage but cross-language ranking precision: pushing the correct translation to the very top rather than merely within the top-10 window.

The practical implication for benchmark users is straightforward: any evaluation runner for codeto-code tasks that does not exclude the anchor will systematically report inflated Recall (since the anchor trivially satisfies coverage) and deflated precision metrics (since nDCG@1 and MRR@1 are wasted on an irrelevant anchor hit). COREB provides the c2c anchor map (queryid → anchorcodeid) alongside the task data so that custom evaluation pipelines can reproduce the corrected metrics without re-running the full benchmark.

0.65

0.7

C2LLM-7B C2LLM-0.5B

Jina-code-1.5B

0.60

Jina-code-0.5B

0.6

nDCG@10

OverallnDCG@10

F2LLM-4B

0.55

0.5

0.50

F2LLM-1.7B

Qwen3-4B

0.4

Text-to-Code Code-to-Text Code-to-Code

Qwen3-0.6B

0.45

| |
|---|

| |
|---|

F2LLM-0.6B

Qwen3-8B

| |
|---|

Pareto frontier

0.3

0.40

0.3B 0.5B 1B 2B 4B 8B

0.3B 0.5B 1B 2B 4B 8B

Parameter count (log scale)

Parameter count (log scale)

1.21 1.19

1.2

Code-specialized General-purpose General-purpose (bar)

OverallnDCG@10/Bparams

1.0

0.8

0.74 0.73

0.6

0.40

0.4

0.29

0.2

0.14 0.12

0.09

0.05

0.0

C2LLM-0.5BJina-code-0.5BQwen3-0.6BF2LLM-0.6BJina-code-1.5BF2LLM-1.7BF2LLM-4BQwen3-4BC2LLM-7BQwen3-8B

- Figure 13: Full model parameter efficiency (expanded version of Figures 9 and 10). Top-left: Overall nDCG@10 vs. parameter count with all model labels. Top-right: Per-task nDCG@10 vs. parameter count; vertical lines connect the same model across tasks. Bottom: Parameter efficiency (nDCG@10 per billion parameters) for all ten open-weight models, sorted descending. Hatched bars = general-purpose models.

B.10 nDCG–Recall Divergence Across Tasks

text-to-code code-to-text code-to-code

0.0

0.2

0.4

0.6

0.8

1.0

Recall

0.08

0.69

0.10

0.38

0.82

0.62

0.72

0.83

0.90

Recall@1 Recall@5 Recall@10

| |
|---|

| |
|---|

- Figure 14: Recall@k at k ∈ {1, 5, 10} per task, averaged over all models on v202603. Code-to-code recall rises most sharply from k=1 to k=10, confirming relevant items are retrieved but not top-ranked.

Comparing nDCG@10 and Recall@10 across tasks (Figures 4 and 14) reveals a task-dependent gap between the two metrics. On code-to-text the metrics track closely, meaning that when a relevant

item appears in the top-10 window it also tends to sit near rank one. On text-to-code and code-tocode the gap is much larger: models do retrieve relevant items within the top ten, but place them

- at lower ranks, which suppresses nDCG while leaving Recall largely intact. This pattern points to ranking precision (not coverage) as the primary bottleneck on the harder tasks, and suggests that reranking or listwise training objectives could yield targeted improvements without needing better recall.

##### B.11 Extended Related Work

LLMs for code. Recent progress in both general-purpose and code-specialized LLMs has dramatically improved program synthesis and software engineering assistance, building on early systems such as Codex (Chen et al., 2021). Frontier assistants such as GPT-4.1 (OpenAI, 2025) and Claude 3.5 (Anthropic, 2024) treat coding as a first-class capability, combining strong reasoning, long-context understanding, and agentic workflows. On the open-source side, StarCoder2 (Lozhkov et al., 2024) and the Qwen2.5-Coder family (Hui et al., 2024) offer 0.5B–32B parameter models trained on multi-language code corpora with strong results across generation, completion, and repair benchmarks. More broadly, foundation models such as Llama 3 (Grattafiori et al., 2024), Gemini 3 Pro (Google DeepMind, 2025), and Claude 4.5 Sonnet (Anthropic, 2025) natively support coding alongside multilingual and reasoning capabilities. These trends highlight a shift from narrow codeonly systems toward general LLMs where strong coding competence is integrated into a broader suite of capabilities.

Code embedding benchmarks. General-purpose embedding benchmarks such as BEIR (Thakur et al., 2021) and MTEB (Muennighoff et al., 2022) have substantially advanced the evaluation of text and image embeddings but include only a small number of code-related retrieval tasks. For code specifically, IdBench (Wainakh et al., 2021) evaluates embeddings of individual identifier names via human similarity judgments, offering fine-grained insights but focusing on token-level semantics rather than whole-program behavior. More recent suites, including CoIR (Li et al., 2024a) and CPRet (Deng et al., 2025), extend embedding evaluation to code retrieval across multiple datasets and task types yet remain primarily retrieval-oriented and often reuse heavily studied datasets such as CodeSearchNet (Husain et al., 2020) and related variants, raising concerns about overfitting and limited task diversity. Beyond these issues, these benchmarks exhibit deeper structural problems: universal 1-to-1 qrels that collapse ranking metrics, widespread label noise (∼51% in CosQA (Gong et al., 2026)), tasks that reduce to string matching or dialogue completion rather than genuine code retrieval, and high contamination risk from datasets that have been public training data for years (Allamanis, 2019; Hernandez Lopez et al., 2024); see sections B.12.1 to B.12.6 for a full dataset-bydataset analysis. In contrast, CoREB targets a broader notion of code representation quality by evaluating contamination-limited embeddings across diverse tasks with multi-relevant qrels, explicit hard negatives, and balanced multilingual coverage.

Retrieval models. Code retrieval models typically pair a pretrained code encoder with a dense retrieval architecture, mapping natural-language queries and code snippets into a shared embedding space for similarity search. Early systems build on encoder-only models such as CodeBERT (Feng et al., 2020) and GraphCodeBERT (Guo et al., 2021), trained with contrastive objectives on NL–code pairs for generic code search. More recent work introduces large-scale, code-specialized retrievers such as CodeSage (Zhang et al., 2024a) and Qodo-Embed (Qodo Ltd., 2025), which leverage webscale multilingual code corpora and retrieval-aware training pipelines to set state-of-the-art results on benchmarks like CoIR. However, these models are mostly evaluated on standard code-search suites, and their behavior on the broader set of code- and problem-centric retrieval scenarios we target remains underexplored.

##### B.12 Structural Flaws in CoIR

CoIR (Li et al., 2024a) aggregates ten code retrieval datasets under a unified evaluation protocol. While this standardization is valuable, a systematic audit reveals structural issues that affect every constituent dataset and cast doubt on the reliability of CoIR-based model comparisons. We summarize the universal and dataset-specific flaws below.

##### B.12.1 Universal: Trivial 1-to-1 Qrels

All ten CoIR datasets assign exactly one relevant document per query, with all relevance scores equal to 1 (no graded relevance, no hard negatives). The query–document ID suffixes are numerically

aligned (e.g., qN→dN), turning retrieval into bipartite matching. Because the ideal DCG for a query with a single binary-relevant document is always 1/log2 2 = 1.0, nDCG@k reduces to a function of that document’s rank position alone; nDCG and MRR become perfectly correlated, and Recall@k is binary (0 or 1). The CoIR paper itself acknowledges this limitation: “each query corresponding to exactly one ground-truth corpus” fails to capture multi-answer scenarios (Li et al., 2024a).

COREB addresses this with multi-relevant qrels: 68% of text-to-code queries have 2–10 relevant documents (cross-language solutions); code-to-code queries average 2.2 relevant translations; and all three tasks include explicit hard negatives (relevance=1) alongside true positives (relevance=2), totaling 11,810 and 12,017 graded judgments in v202602 and v202603 respectively (Table 5). These give nDCG and Recall distinct, meaningful roles.

##### B.12.2 CosQA: ∼51% Mislabeled Pairs

CosQA pairs Bing web search logs with CodeSearchNet GitHub functions. This domain mismatch causes systematic mislabeling, independently confirmed by CoSQA+ (Gong et al., 2026), which reports that “around 51% of queries are paired with mismatching code.” Our manual review of 80 pairs found ∼37.5% clear mislabels (code does the opposite or an unrelated thing), ∼22.5% weak/tangential matches, and only ∼40% reasonable matches. Concrete examples:

- • “python check file is readonly” is paired with a function testing os.access(f, os.R OK) (read permission, the opposite of read-only).

- • “how to make seconds to time in python” is paired with time2seconds, the exact inverse conversion.
- • “python making string lower case” is paired with a CamelCase converter (to camel).

The test split contains only 500 pairs, of which roughly 200 are reasonable, giving an extremely high-variance evaluation surface.

##### B.12.3 CodeSearchNet: Docstring Retrieval, Not Code Search

CodeSearchNet pairs code with its own docstring. The original paper acknowledges that documentation “is often written at the same time and by the same author as the documented code, and hence tends to use the same vocabulary, unlike search queries” (Husain et al., 2020). ProCQA (Li et al.,

- 2024b) confirms that the queries are “either documentation strings or comments rather than natural language questions, limiting its practicality in real scenarios.” Code2Doc (Karaman & Akarsu,
- 2025) found that aggressive quality filtering retains only 25.6% of CodeSearchNet-style repositoryscraped data, and reports 15–25% test/train near-duplication. Because CodeSearchNet has served as public training data since 2019, models evaluated on it face severe contamination risk (Allamanis, 2019; Hernandez Lopez et al., 2024).

##### B.12.4 CodeSearchNet-CCR: String Completion

CCR randomly splits each CodeSearchNet function at a character position between 40–70%. The first half is the query; the second half is the corpus item. This produces splits mid-docstring sentence, mid-control-flow statement, or mid-variable name. Both halves share the function name as a metadata title field, providing trivial lexical leakage. The task measures string continuation memory, not code retrieval ability.

##### B.12.5 CodeFeedback-MT: Dialogue Completion

Multi-turn conversations where the query is the full dialogue history (including prior assistant turns with code) and the target is the final assistant response. The query already contains complete implementations of the same algorithm repeated across turns, enabling near-deduplication matching. Average query length is 4,558 characters, well beyond the context window of most embedding models.

##### B.12.6 Other Datasets

CodeFeedback-ST: Both queries and corpus are LLM-generated, creating artificial stylistic coherence. Some corpus items contain bare output values ([3,4,5]) or pure prose with zero code, labeled as “Python.” Synthetic Text2SQL: Each query includes CREATE TABLE DDL with unique table/column names that appear verbatim in the target SQL; a BM25 baseline scores highly without semantic understanding. StackOverflow QA: Only ∼0.15% of queries contain code; this is effec-

tively text QA, not code retrieval. APPS: Python-only, with no quality control; corpus items include personal signatures, opaque variable names, and dead docstrings placed after return statements (Li et al., 2023; Siddiq et al., 2024). CodeTransOcean: Too small for reliable evaluation (180–446 test queries).

##### B.12.7 Summary: CoIR vs. CoREB

Table 17: Structural comparison between CoIR and COREB.

Dimension CoIR CoREB Qrel structure 1-to-1, score = 1 only Multi-relevant + graded hard negatives

(relevance = 1) Label quality ∼51% mislabeled (CosQA) Programmatically verified Languages Python + SQL only (meaningful

5 languages, balanced across all tasks

tasks)

Contamination risk High (public since 2019) Low (fresh LLM-generated, Feb 2025) Hard negatives None across all 10 datasets 668/700 noise corpus items per release

(1,368 aggregated) + relevance-1 qrels

##### B.13 Limitations

COREB inherits several scope constraints. First, the corpus is drawn exclusively from competitive programming via LiveCodeBench, so the benchmark may not fully represent retrieval over enterprise or library code. Second, only five programming languages are covered; extending to lower-resource languages (e.g., Rust, Kotlin) could change the difficulty landscape. Third, all queries are LLMgenerated from problem statements rather than collected from real developer search logs, which limits ecological validity for the short-query setting. Finally, the current evaluation is offline and single-turn; interactive or multi-turn code search scenarios are not addressed.

