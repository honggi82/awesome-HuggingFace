# arXiv:2606.09323v1[cs.AI]8Jun2026

## TRL-BENCH: Standardizing Cross-Paradigm Representation-Level Evaluation of Tabular Encoders

###### Wei Pang1,∗, Xiangru Jian2,∗, Hehan Li1,∗, Zhixuan Yu1,∗, Alex Xue2,∗ Jinyang Li3, Zhengyuan Dong2, Xinjian Zhao1, Hao Xu4 Chao Zhang5, Reynold Cheng3, M. Tamer Özsu2, Tianshu Yu1,†

1The Chinese University of Hong Kong, Shenzhen 2University of Waterloo 3The University of Hong Kong 4The University of Sydney 5Université Lyon 1

∗Core contributors †Corresponding author: yutianshu@cuhk.edu.cn

##### Abstract

Tabular encoders are usually evaluated inside task-specific end-to-end pipelines, so models from different training paradigms are difficult to compare directly even when they operate on similar tabular signals. We introduce TRL-BENCH, a multigranular tabular representation learning (TRL) benchmark that standardizes crossparadigm representation-level evaluation: each encoder exports row-, column-, or table embeddings through its supported wrapper, and shared lightweight heads probe them across three suites: TRL-CTBENCH (column/table), TRL-RBENCH (row), and TRL-DLTE (compositional Data-Lake Table Enrichment spanning all three granularities). To support this standardized setting, we release curated benchmark assets and task reformulations, including 50 OpenML tables with 123 verified targets, 16 row-pair linkage rewrites, and a 47,772-table DLTE lake derived from 1,379 parent tables. Across 20 models and 16 tasks, TRL-BENCH shows that once downstream conditions are standardized, encoder quality is capability-specific rather than captured by a single leaderboard. In TRL-CTBENCH, generic text encoders often lead on tasks with strong surface-text signal, while tabular specialists win where their pretraining objective aligns with the task. In TRL-RBENCH, within-table prediction and cross-table linkage favor different training regimes, with atomic linkage performance correlating strongly with the row-matching stage of DLTE pipelines. In TRL-DLTE, the strongest pipelines combine capabilitymatched specialists rather than reuse a single encoder, and top end-to-end quality depends on non-additive compositional fit rather than per-stage marginal rank alone. TRL-BENCH provides a common protocol for measuring reusable signal in exported tabular representations under shared downstream conditions. Code: https://github.com/LOGO-CUHKSZ/TRL-Bench; data is released on Hugging Face.1

##### 1 Introduction

Tables have long been recognized as the fundamental data structures for storing structured data, and there has been considerable work on using them across a wide range of analytical workloads. Recent work has produced strong row-, column-, and table-level encoders for reasoning over tabular data. Many of these are useful as reusable components: tables can be encoded once and their

1TRL-CTBENCH: https://huggingface.co/datasets/logo-lab/trl-ctbench TRL-RBENCH: https://huggingface.co/datasets/logo-lab/trl-rbench TRL-DLTE: https://huggingface.co/datasets/logo-lab/trl-dlte

Preprint.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

DATA 2 ENCODERS 3 EMBEDDINGS

TASKS

5 FINDINGS

4

Source tables (encoded once)

TRL-CTBENCH

###### TRL-DLTE

###### Column/Table families

###### TRL-CTBENCH

Table embeddings

Row signal is not single-faceted

Transfer is capability-specific

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Retrieve candidate tables using table embeddings

…

Col Centric

Generic Text

Table Text

Table Structure

20 datasets 13 tasks

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

...

###### Row regimes

TRL-DLTE

Column embeddings

Schema Joinability

[Figure 19]

TransferBased

PriorBased

Target-Table Learners

[Figure 20]

…

Lake of 47,772 tables

[Figure 21]

[Figure 22]

Align columns + decide union / join / none using column embeddings

[Figure 23]

2

[Figure 24]

...

Row embeddings

[Figure 25]

Unionability Grounding

TRL-RBENCH

###### Tabular Encoder

Capability-matched hybrids beat singleencoder reuse

Compositional fit matters

…

50 OpenML tables 123 targets

[Figure 26]

[Figure 27]

Match rows + merge using row embeddings

###### TRL-RBENCH

[Figure 28]

[Figure 29]

16 record-linkage datasets

Reusable across tasks and datasets

[Figure 30]

[Figure 31]

Legend

[Figure 32]

Embedding granularities (by color / icon) Table embedding Column embedding

Benchmark suites (by color) Blue = TRL-CTBENCH Purple = TRL-DLTE Yellow = TRL-RBENCH

...

[Figure 33]

[Figure 34]

Hand-verified targets

Record linkage

Row prediction

[Figure 35]

End-to-end metric: UJ-H

###### Leakage-audited pairs

Row embedding

- Figure 1: TRL-Bench at a glance. Each model is processed once through its supported wrapper to export row-, column-, or table embeddings, and shared lightweight modules then evaluate those embeddings across TRL-CTBENCH (schema, joinability, unionability, grounding), TRL-RBENCH (row prediction, record linkage), and TRL-DLTE (multi-stage data-lake enrichment).

embeddings indexed and reused across tasks and large multi-table corpora such as data lakes, where per-task fine-tuning is often impractical [5, 24, 23]. In such encode-once, reuse-many settings, the representation itself, not the task-specific wrapper, is the object of evaluation. Yet these encoders are still mostly evaluated inside task-specific end-to-end pipelines, so models from different training paradigms are difficult to compare directly: a strong result may come from the wrapped predictor, training budget, and task-specific adaptation as much as from the encoder itself. This motivates a comparability question: under one shared evaluation protocol over the exported representations, how do heterogeneous tabular encoders actually differ?

TRL-BENCH is designed around that question and complements end-to-end task benchmarks by isolating reusable representation quality under shared downstream conditions (Figure 1). Each model is run once through its supported wrapper to export the row-, column-, or table embeddings it exposes, and shared lightweight downstream modules evaluate those embeddings across tasks rather than re-optimizing the encoder end to end. Throughout, we use “encoder” operationally to denote any tabular model that exposes reusable row-, column-, or table-level embeddings.

To make this evaluation comprehensive rather than task-specific, TRL-BENCH treats retrieval, schema alignment, linkage, prediction, and grounding as atomic capabilities that serve as reusable building blocks for downstream tabular systems in the encode-once, reuse-many setting. The three suites measure these capabilities at the granularities where embeddings are reused: TRL-CTBENCH for column/table transfer, TRL-RBENCH for row transfer, and TRL-DLTE for compositional data-lake table enrichment.

When the benchmark is used to test 20 models and 16 tasks, three empirical findings emerge. First, once downstream conditions are standardized, transfer is capability-specific: in TRL-CTBENCH, generic text encoders often lead on tasks with strong surface-text signal, while the remaining wins are better explained by pretraining–task alignment than by any single dominant encoder class. Second, row signal is not single-faceted: within-table prediction and noisy cross-table linkage separate model families by training scope. Third, compositional fit shapes pipeline quality: in TRL-DLTE, the best pipelines are capability-matched hybrids that consistently outperform single-encoder reuse. Per-stage marginals are informative but do not determine the top pipelines. End-to-end quality depends on how well retrieval, column alignment, and row matching compose, not on per-stage rank in isolation.

###### Contributions.

- 1. Standardized cross-paradigm protocol: heterogeneous encoders export row-, column-, or table-level embeddings, and shared lightweight readouts evaluate them under common task definitions, enabling direct comparison without end-to-end fine-tuning.
- 2. Comprehensive benchmark for reusable tabular signal: TRL-CTBENCH, TRL-RBENCH, and TRL-DLTE cover column/table transfer, row transfer, and compositional enrichment over 16 tasks and 87 datasets from SATO [82], SOTAB [43], WikiCT [17], Spider [81], Valentine [45], OpenML [73], and DeepMatcher/WDC [53, 63] (Appendix F).
- 3. Curated assets and task reformulations: we contribute (a) 20 column/table datasets standardized for representation-level evaluation, (b) 50 OpenML-derived row-prediction tables with 123 hand-verified targets, (c) 16 record-linkage datasets rewritten as explicit row-pair matching tasks, and (d) a 47,772-table enrichment lake built from 1,379 parent tables, together with representationcentric rewrites of heterogeneous source tasks such as WTQ [60] and DeepMatcher.

- Table 1: Comparison with prior tabular evaluation resources (✓=supported). Cross-paradigm: multiple training paradigms under one protocol. Repr.-level eval.: representation-level evaluation as primary intended use. Downstream transfer: downstream task performance vs. intrinsic properties. Task family: broad problem class (e.g., row prediction, semantic typing). Observatory reports intrinsic properties only, hence “–”.

Crossparadigm

Repr.-level eval.

Downstream transfer

# Task fam.

Benchmark Col. Table Row Comp.

OpenML suites [73, 8, 25] ✗ ✗ ✓ ✗ ✗ ✗ ✓ 1 TabArena [21] ✗ ✗ ✓ ✗ ✗ ✗ ✓ 1 DeepMatcher [53] ✗ ✗ ✓ ✗ ✗ ✗ ✓ 1 LakeBench [68, 18] ✓ ✓ ✗ ✗ ✗ ✗ ✓ 2 SANTOS / TUS [54, 39] ✓ ✓ ✗ ✗ ✗ ✗ ✓ 1 Valentine [45] ✓ ✗ ✗ ✗ ✗ ✗ ✓ 1 SemTab / SOTAB [36, 43] ✓ ✗ ✗ ✗ ✗ ✗ ✓ 1 Observatory [15] ✓ ✓ ✓ ✗ ✓ ✓ ✗ –

TRL-BENCH (ours) ✓ ✓ ✓ ✓ ✓ ✓ ✓ 7

4. A cross-paradigm empirical study: across 20 models and 16 tasks, we show that no single pretraining recipe behaves as a universal tabular representation, and we identify structural gaps in model choice, transfer scope, and pipeline composition that single-paradigm or single-granularity evaluations cannot isolate.

##### 2 Related Work and Positioning

We situate TRL-BENCH relative to both prior tabular model families and existing benchmark resources. Appendix B gives additional citations and comparisons.

Related work: model families and evaluation traditions. Prior tabular work is fragmented by granularity. Row-level models focus on supervised prediction and transfer [33, 66, 72, 48, 80, 6, 74, 76, 31, 32, 64, 27, 12, 28, 41, 87], while column/table models target schema semantics, grounding, retrieval, and discovery [19, 50, 29, 78, 17, 75, 34, 24, 40]. These families are usually evaluated in task-specific settings rather than under a shared multi-granular representation-level protocol. Observatory [15] is the closest prior resource that compares frozen tabular embeddings across model families, but it measures perturbation- and invariance-style intrinsic properties such as sample fidelity and order insignificance rather than downstream task performance.

Positioning of TRL-BENCH. Prior benchmarks each target narrow task scopes (Table 1) [73, 8, 25, 21, 53, 63, 36, 43, 45, 54, 39, 68, 18] and usually compare models within a single task family or end-to-end pipeline rather than under a shared representation-level protocol. TRLBENCH complements these resources by standardizing heterogeneous tabular encoders into a shared representation-level evaluation protocol. Its main distinctions are: (i) multi-granular evaluation across columns, rows, and tables, (ii) direct cross-paradigm comparison under common task definitions and lightweight downstream heads, and (iii) a compositional DLTE benchmark testing whether strong atomic capabilities compose into an end-to-end pipeline. Because this standardized comparison operates on exported representations, it applies to models that expose reusable row-, column-, or table-level embeddings, either natively or via a natural extraction point in the architecture. Generative table LLMs [85, 69] generally do not provide such an interface, while heavily task-specific fine-tuned systems [49, 70, 35] are formulated as end-to-end predictors rather than reusable representations.

##### 3 Benchmark Design

TRL-BENCH asks a comparability question: once heterogeneous tabular encoders are evaluated under one shared representation-level protocol, how do they differ across rows, columns, and tables, and how do they compose end-to-end? The benchmark has three suites, TRL-CTBENCH, TRL-RBENCH, and TRL-DLTE. Figure 1 gives the high-level view.

###### 3.1 Problem Setting and Standardized Representation-Level Protocol

Throughout, we adopt two established properties of good representations from the representationlearning literature: recoverability under simple, capacity-limited readouts (the probing tradition [2]),

###### (a) ROW PREDICTION CURATION TRL-Rbench

###### (b) DLTE LAKE CONSTRUCTION TRL-DLTE

fragment

screen

release

combine

INPUT

OUTPUT

INPUT

###### OUTPUT

review

compose

audit

+ distractors

###### 3 HUMAN REVIEW & REPAIR

###### 1

###### 2

###### 3

1

###### 2 DEGENERACY AUDIT

#### 158

#### 50

###### 1,379

HARD DISTRACTORS

COMPOSITIONAL TARGETS

###### FRAGMENT GENERATION

###### RULE SCREEN

seed · union · join

candidate tables

drop ID, nearunique, leaked, constant targets

parent tables

curated tables

verify reg/cls fix labels

≥ 2 valid targets/table

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

###### 36,740

###### 11,032

targets · distractors

[Figure 36]

union + join targets

###### 47,772

CKAN tables

TabArena

###### 77 classification

###### 989 TabFact

OpenML-CC18

shared retrieval lake

46 regression >=2 targets/table

390 WTQ

OpenML-CTR23

827/207/345 train/dev/test

- Figure 2: Curation of TRL-RBENCH row-prediction tables and assembly of the TRL-DLTE lake. (a) Row prediction curation: 158 candidate tables filtered through rule screening, degeneracy audit, and human review with label repair into 50 tables with 123 targets. (b) DLTE lake assembly: 1,379 TabFact/WTQ parents fragmented into seed queries and union/join targets at four noise tiers. 11,032 targets are embedded alongside 36,740 CKAN distractors in a 47,772-table lake.

and transferability across many downstream tasks [7]. We take a reusable tabular representation to be good to the extent that a single exported embedding satisfies both. TRL-BENCH standardizes heterogeneous tabular models at the level of exported representations, not by forcing a single input serialization. For a table T with columns C(T) = (c1,...,cM) and rows R(T) = (r1,...,rN), a tabular encoder fθ may expose column, row, or table representations,

Ecol(T) = (ecol1 ,...,ecolM ), Erow(T) = (erow1 ,...,erowN ), etbl(T).

For each task, write e for the relevant exported encoder output(s) (a column, row, table, or pair thereof). A downstream module r then maps these embeddings to the task output, and TRL-BENCH uses three downstream-module types. Training-free modules r(e) have no task-specific learned parameters and operate directly on embedding geometry, e.g., cosine ranking for schema matching or union search, or k-means for column clustering. Learned modules rψ(e) are lightweight supervised probes trained on exported encoder embeddings, e.g., column type prediction, join/union classification, join search, row prediction, and record linkage. Query-conditioned modules rψ(q,e) additionally consume a frozen text-query embedding q = ftext(query), e.g., a dual-projection head for table retrieval or a decoder for table QA. Operationalizing recoverability, all downstream heads are intentionally lightweight and held fixed across encoder families, so comparisons reflect the choice of exported embedding under common readouts rather than the choice of downstream predictor. Transferability is then tested by reusing each exported embedding across the multiple tasks within its suite, and, for models that expose multiple granularities, across suites as well (Sec. 3.2–3.4).

The learned-module category follows a unified supervised-probe protocol. For every supervised probe task, we train both a linear head and a one-hidden-layer MLP (hidden size 256, robust to head size and depth, Appendix Table 24) on exported encoder embeddings with Adam [42] under standardized settings (Appendix R). We use the arithmetic average of the two as the canonical score: the linear head tests linearly accessible signal, and the MLP tests whether a small nonlinear readout can recover it. Per-head diagnostics, including a cosine reference for record linkage, are reported in Appendices K.2 and L.3. For table-level tasks, if a model supports multiple table aggregations (e.g., cls, col-mean, tok-mean), we apply this protocol to each aggregation and report the strongest (per-model, full ablations in Appendix K.3). Training-free tasks (Column Clustering, Union Search, Schema Matching), query-conditioned tasks (Table QA, Table Retrieval), and the DLTE pipeline (Sec. 3.4) follow their corresponding module types defined above.

Pairwise tasks use fixed combination operators, typically concatenation. Appendix Tables 7–8 give per-task details. A model is evaluated only when the required granularity is natively exposed or obtainable by supported pooling. Appendix C summarizes supported granularities, and Appendix D states the benchmark’s wrapper policy.

###### 3.2 TRL-CTBENCH: Column- and Table-Level Transfer

TRL-CTBENCH contains 13 tasks: 8 column-level and 5 table-level, grouped into schema understanding, joinability, unionability, and grounding.

Schema understanding. These tasks consume a single column embedding or an ordered intra-table column pair and test whether exported column representations expose semantic type and intra-table structure. They include column type prediction, column clustering, and column relation prediction.

Joinability. Joinability asks whether two tables are complementary, i.e., one can add attributes to the other through overlapping columns. Tasks are join search, column overlap, and table-level join classification. Since raw cosine between column embeddings does not directly model directional value containment (i.e., whether one column’s values are contained in another’s), the main join-search setting uses a minimal learned projection head on top of the exported embeddings.

Unionability. Unionability asks whether two tables are stackable after schema alignment. Tasks are union search, schema matching, union classification, union regression, and table subset. Union search uses SANTOS [39], UGEN [58], and TUS [54], with an added low-overlap TUS-hard variant (Appendix K.5) that removes positive pairs whose directed column containment is ≥ 0.70, creating an explicit contrast between the original TUS setting and one without high value-containment positives.

Grounding. Grounding tests whether a representation can ground a natural-language query in structured table content. These tasks are query-conditioned: a frozen text encoder embeds the question, and a lightweight head combines that query embedding with table-side representations exported by the evaluated model. Table QA uses question and column embeddings with a lightweight decoder, while table retrieval trains a dual projection head over query and table embeddings.

Task adaptation and curation. Most source tasks were not originally formulated for standardized representation-level evaluation. We therefore standardize table identifiers, align label schemas, rewrite end-to-end datasets into representation-centric variants, and clean dataset layouts where needed. Across the benchmark, this rewriting is concrete: for WikiTableQuestions [60], we replace joint encoder fine-tuning with exported column/question embeddings plus a lightweight decoder. For the four pairwise tasks whose original splits exhibit table-level overlap (join classification, column overlap, union classification, and union regression), we enforce table-disjoint train/dev/test splits that prevent any test table from appearing during training. The remaining supervised table-pair task (table subset) already has table-disjoint splits in the source data. Appendix Tables 7–8 list per-task evaluation modes, split types, and metrics. Appendix E.1 summarizes the benchmark’s shortcut and leakage mitigations.

###### 3.3 TRL-RBENCH: Row-Level Transfer

TRL-RBENCH asks whether exported row embeddings transfer both within a table and across tables. It contains row prediction and record linkage.

Row prediction. For row prediction, the encoder sees only the observed columns X, produces one target-agnostic embedding per row, and that same embedding is reused to predict each curated target column yk ∈ Y = {y1,...,yK} (K ≥ 2) with a lightweight probe under the protocol of Sec. 3.1. This asks whether a single row embedding can be reused across multiple targets from the same table, including tables with mixed classification and regression targets. The suite contains 50 OpenML-derived tables with 123 curated targets (77 classification, 46 regression), filtered from 158 candidates from TabArena [21], OpenML-CC18 [8], and OpenML-CTR23 [25]. Every released table has 2–3 targets (mean 2.46). Human curators selected target columns, repaired label issues where needed, verified classification-versus-regression typing, and removed degenerate or leaked targets such as constant columns, near-duplicate targets, and label columns recoverable from the input (Figure 2). All target columns are excluded from encoder input.

Record linkage. Record linkage complements intra-table prediction with inter-table matching: given a pair of rows from two tables, predict whether they refer to the same entity. We adopt 16 datasets from two entity-matching benchmark families [53, 63, 62]: 8 clean DeepMatcher [53] benchmarks, 4 dirty DeepMatcher variants with synthetic schema noise, and 4 size variants of the WDC Products Large-Scale Product Matching (LSPM) benchmark. For analysis, we split these into Clean Linkage (the 8 clean DeepMatcher datasets) and Robust Linkage (the 4 dirty DeepMatcher + 4 WDC datasets). We rewrite these sources as explicit row tables with labeled row pairs, retain the original source pair-disjoint splits, which are the canonical evaluation protocol in the entity-matching literature, and feed paired exported row embeddings via concatenation to a lightweight supervised probe under the protocol of Sec. 3.1. Appendix L.4 reports per-source train/test pair and row-overlap statistics, and documents the removal of label-equivalent columns before any encoder consumes a row.

###### 3.4 TRL-DLTE: Multi-Stage Data Lake Table Enrichment

Atomic tasks test local transfer, but not whether row-, column-, and table-level representations compose into a full multi-stage pipeline. TRL-DLTE addresses this gap. We start from a complete parent table, the ground-truth table to be reconstructed. From it, we remove a block of rows and a block of columns. The remaining subtable is the seed query. The removed rows form the union target (same schema as the seed, additional rows), and the removed columns form the join target (same rows as the seed, additional attributes). Given only the seed and a data lake, the system must recover both targets by retrieving relevant tables, deciding whether each candidate contributes by union, join, or neither, aligning columns, matching rows, and merging the result.

We build TRL-DLTE from filtered parent tables drawn from TabFact [13] and WTQ [60]. Figure 2(b) summarizes the construction and counts: each parent is fragmented at four cumulative noise tiers (clean, schema, cell, hard) into a seed query, a union target, and a join target, and the targets are inserted into a shared retrieval lake together with CKAN distractors. Seeds serve only as queries and are not lake members. Parent tables are split before fragmentation so that train/dev/test remain parent-disjoint.

Evaluation has three stages. Stage 1 retrieves candidate tables using table embeddings. Stage 2 aligns columns and predicts union/join/none using column embeddings. Stage 3 matches rows and merges content using row embeddings. Pipelines can use a single multi-granular model or combine different specialists across stages, making DLTE a composition test for the benchmark as a whole. Full stage-wise operator specifications, including the Stage-2 threshold calibration procedure and the dev-based selection of the headline pipeline reported in Sec. 4.4, are in Appendix M.

We introduce UJ-H as the primary end-to-end score, which summarizes recovery of both the union and join targets. Let Runion be the fraction of removed-row-block cells recovered in the seed columns, and Rjoin the fraction of removed-column-block cells recovered for the seed rows. UJ-H is the perquery harmonic mean of these two recalls, averaged over queries (zero when both recalls vanish):

2Runion Rjoin Runion + Rjoin

UJ-H =

penalizing pipelines that succeed on only one enrichment path. We additionally report Cell F1 in Appendix N.1. It is a multiset F1 over recovered cells pooled across the removed-row and removed-column blocks, used as a complementary diagnostic of pooled cell-recovery yield.

##### 4 Experiments

We now use the standardized representation-level protocol of Sec. 3.1 to compare heterogeneous tabular encoders across the three benchmark suites introduced in Sec. 3: TRL-CTBENCH, TRLRBENCH, and TRL-DLTE.

###### 4.1 Experimental Setup

We follow the standardized representation-level protocol in Sec. 3.1. Here we summarize only the choices needed to read the result tables.

Compared models. We evaluate 20 models spanning generic text encoders, table-aware and structureaware encoders, column-specialized models, target-table self-supervised learners, and meta-pretrained priors. Appendix Table 5 lists the full inventory.

Baselines and controls. Each task reports the applicable simple non-neural or analytical baselines alongside learned encoders (e.g., TF-IDF for column-level tasks, value overlap for search). “Best∗” in Table 2 denotes the strongest applicable non-neural baseline, with markers a–d identifying which one. Full specifications and 5-seed per-dataset results are in Appendix G.

Reporting and Metrics. CTBench reports raw metrics plus per-family normalized-rank (NR, lower is better) aggregates. For table-level tasks, the main comparison uses the strongest supported aggregation (Appendix K.3). Row prediction averages over 123 targets (77 classification and 46 regression, with TABTRANSFORMER covering 63 due to its categorical-feature requirement), with linkage split into Clean Linkage (DM-C) and Robust Linkage (DM-D + WDC NR aggregate). DLTE uses UJ-H as the primary end-to-end score. To avoid selection bias from reporting the maximum over 1,120 test evaluations, headline DLTE pipelines are selected on the development split by UJ-H

- Table 2: Column- and table-level results on 13 TRL-CTBENCH tasks spanning four capability families. Join, Union, and Grounding are evaluated at both column and table granularity (superscripts c/t). NR reports the mean normalized rank within each family. Dashes indicate unsupported granularities. Formula of metrics in Appendix H.

Schema Join Union Grounding

Col Overlapc† nRMSE↓

Join Class.t† F1 ↑

Union Class.t† F1 ↑

Union Reg.t† nRMSE↓

Join Searchc MAP↑

Union Searchc MAP↑

Schema Matchc

Tbl QAc Acc↑

Tbl Ret.t MRR↑

Col Typec

Col Clustc NMI↑

Col Relc

Tbl Subsett

NR

NR

NR

NR

Family Model

↓

↓

↓

↓

R@GT↑

F1 ↑

F1 ↑

F1 ↑

Baseline Best∗ 0.813a 0.400a 0.015 — 0.155b 1.014 0.516 — 0.574c 0.473d 0.500 1.138 0.458 — 0.204 0.131 —

±.007 ±.001 ±.001 ±.000 ±.003 ±.014 ±.000 ±.000 ±.004 ±.024 ±.027 ±.004 ±.120

###### BERT 0.926 0.516 0.826 0.000 0.434 0.786 0.553 0.048 0.564 0.423 0.857 0.592 0.545 0.260 0.255 0.367 0.397

- ±.001 ±.000 ±.002 ±.000 ±.001 ±.010 ±.000 ±.000 ±.002 ±.009 ±.005 ±.004 ±.008

GTE 0.922 0.466 0.811 0.190 0.469 0.817 0.535 0.243 0.608 0.417 0.843 0.600 0.544 0.343 0.245 0.476 0.429

- ±.002 ±.001 ±.002 ±.002 ±.002 ±.019 ±.000 ±.000 ±.002 ±.009 ±.002 ±.002 ±.003

Generic Text

Tabular-Pretrained

###### TaBERT 0.874 0.514 0.760 0.381 0.406 0.855 0.498 0.630 0.619 0.430 0.760 0.615 0.540 0.457 0.267 0.372 0.198

±.003 ±.003 ±.002 ±.002 ±.002 ±.034 ±.000 ±.000 ±.004 ±.009 ±.004 ±.005 ±.013

###### TAPAS 0.868 0.448 0.769 0.476 0.320 0.823 0.544 0.323 0.529 0.415 0.837 0.607 0.567 0.419 0.254 0.295 0.579

Table-Text

±.001 ±.001 ±.003 ±.001 ±.002 ±.011 ±.000 ±.000 ±.004 ±.005 ±.004 ±.003 ±.006

###### TAPEX — — — — — — 0.538 0.333 — — 0.854 0.609 0.558 0.185 — 0.332 0.333

±.021 ±.002 ±.006 ±.003 ±.005

###### TABBIE 0.892 0.262 0.785 0.476 0.208 0.862 0.542 0.693 0.410 0.205 0.833 0.663 0.546 0.727 0.276 0.170 0.516

±.004 ±.007 ±.002 ±.001 ±.002 ±.022 ±.000 ±.000 ±.002 ±.001 ±.004 ±.004 ±.004

###### TURL 0.814 0.406 0.758 0.667 0.299 0.809 0.532 0.471 0.575 0.340 0.814 0.657 0.507 0.673 0.277 0.199 0.389

Table-Struct.

±.001 ±.004 ±.002 ±.000 ±.001 ±.014 ±.000 ±.001 ±.001 ±.009 ±.004 ±.005 ±.010

###### TUTA — — — — — — 0.468 1.000 — — 0.810 0.652 0.447 0.778 — 0.260 0.556

±.010 ±.003 ±.007 ±.008 ±.013

Starmie 0.789 0.404 0.698 0.810 0.316 0.847 0.510 0.640 0.662 0.764 0.853 0.662 0.539 0.356 0.266 0.018 0.714

±.004 ±.000 ±.001 ±.001 ±.001 ±.019 ±.000 ±.000 ±.002 ±.003 ±.005 ±.005 ±.002

Col.-Centric

###### TabSketchFM 0.566 0.252 0.373 1.000 0.265 0.946 0.516 0.841 0.531 0.155 0.737 0.668 0.553 0.787 0.235 0.218 0.833

±.001 ±.007 ±.003 ±.001 ±.001 ±.015 ±.000 ±.000 ±.002 ±.010 ±.005 ±.005 ±.011

∗Best baseline per task: unmarked =Random; aTF-IDF, b,cvalue overlap, dJaccard. Table-level results report, for each model, the best-performing supported table aggregation among CLS, COL-MEAN, and TOK-MEAN under the unified supervised-probe protocol of Sec. 3.1; full aggregation ablations in Tables 17 and 18.

within the relevant candidate set and evaluated once on test. Top-50 frequencies, per-stage marginals, and Oracle-RA (Stages 1–2 replaced by ground truth) are test-set descriptive analyses over the full pipeline space. Stage-2 thresholds are calibrated on development with macro-F1, independently of headline pipeline selection (Appendix M). NR averages a model’s normalized rank over units of an aggregate (tasks within a CTBench family, target columns, or linkage datasets), excluding missing units. Formulas and standard metric definitions are in Appendix H. In all tables, ↓ marks lower-is-better. The colors orange / blue / light purple highlight ranks 1/2/3 per column, and † marks table-disjoint splits.

###### 4.2 Column- and Table-Level Results

- Table 2 reports the main column/table results across 13 TRL-CTBENCH tasks, evaluated on the 10 of 20 models that natively expose column or table embeddings, spanning schema understanding, joinability, unionability, and grounding. Two empirical patterns stand out.

Generic-text rankings track surface-text signal. Family-level NR (lower is better) for BERT and GTE worsens from Schema through Grounding (BERT 0.000 → 0.048 → 0.260 → 0.397, and GTE 0.190 → 0.243 → 0.343 → 0.429). A task-level surface-text audit (Appendix K.1) is consistent with this interpretation: generic text encoders are strongest on tasks where headers and short cell strings carry most of the signal, and the CTBench tasks where a tabular specialist beats them (Union Search, Schema Matching, Table Subset, and Table QA) all sit in the Union and Grounding families, which instead reward cross-table alignment geometry and grounded table understanding.

Pretraining alignment matters beyond surface cues. The four CTBench tasks won by tabular specialists are each consistent with their winner’s pretraining design. Column-centric STARMIE’s contrastive objective matches the cosine-scoring setup for Union Search (0.662 MAP) and Schema Matching (0.764 R@GT). On Table Subset, the top three are all tabular specialists (TAPAS 0.567 F1, TAPEX 0.558, TABSKETCHFM 0.553), placing structural pretraining above text serialization on this task. In Grounding, the Table-Text family leads at the family level: TABERT’s joint text-table pretraining delivers the best Grounding NR (0.198), while task-level wins split between structureaware TURL (Table QA, 0.277 Acc) and generic-text GTE (Table Retrieval, 0.476 MRR). Although GTE is classified as a generic text model, it is pretrained with a retrieval-contrastive objective [50]. Its Table Retrieval win is itself an instance of pretraining-task alignment rather than an exception.

4.3 Row-Level Results

- Table 3 reports row-level transfer under the shared probe protocol of Sec. 3.1. Three empirical patterns stand out.

- Table 3: Row-level results across four evaluation categories. Row prediction averages over 77 classification and 46 regression targets (TABTRANSFORMER: 63 targets due to categorical-feature

requirement). SGM is SGM0.01(nRMSE). Linkage columns report binary F1 (match class) on DM-C, DM-D, and WDC. See Appendix H for the convention. Rank columns aggregate ranks over individual targets (classification, regression) or datasets (linkage). Formula of metrics in Appendix H.

Classification Regression Clean Linkage Robust Linkage

Family Model AUROC↑ Macro

F1 ↑ NR ↓ SGM↓ NR ↓ DM-C

F1 ↑ NR ↓ DM-D

WDC

F1 ↑ NR ↓ Baseline

F1 ↑

Dummy 0.500±.000 0.304±.000 — 1.004±.000 — 0.000±.000 — 0.000±.000 0.000±.000 Random 0.506±.001 0.348±.001 — 1.103±.000 — 0.179±.007 — 0.223±.003 0.128±.003 —

BERT 0.791±.000 0.635±.000 0.378 0.704±.002 0.559 0.418±.004 0.096 0.464±.005 0.236±.003 0.163 GTE 0.770±.000 0.610±.001 0.544 0.765±.001 0.714 0.392±.006 0.173 0.516±.005 0.311±.001 0.048 TABBIE 0.770±.001 0.599±.001 0.541 0.766±.003 0.643 0.365±.005 0.250 0.330±.011 0.140±.009 0.404 TUTA 0.720±.000 0.553±.002 0.551 0.725±.003 0.632 0.377±.011 0.231 0.451±.006 0.227±.005 0.154

TransferBased

TabICL 0.816±.001 0.671±.002 0.164 0.505±.001 0.139 0.316±.011 0.423 0.318±.006 0.147±.006 0.394 TabPFN 0.793±.001 0.621±.002 0.492 0.607±.002 0.499 0.254±.008 0.596 0.251±.023 0.087±.010 0.663

PriorBased

VIME 0.794±.000 0.640±.001 0.385 0.556±.003 0.367 0.257±.026 0.529 0.259±.027 0.099±.006 0.596 SCARF 0.794±.001 0.642±.001 0.371 0.571±.002 0.399 0.266±.010 0.510 0.258±.021 0.069±.008 0.683 DAE 0.793±.001 0.643±.001 0.379 0.575±.002 0.392 0.241±.013 0.644 0.252±.022 0.098±.004 0.615 TabBinning 0.792±.001 0.640±.001 0.396 0.573±.002 0.397 0.256±.011 0.615 0.279±.025 0.068±.004 0.673 SAINT 0.768±.002 0.595±.004 0.543 0.617±.009 0.561 0.167±.047 0.712 0.176±.048 0.133±.004 0.606 SubTab 0.731±.001 0.550±.002 0.798 0.782±.003 0.779 0.094±.005 0.933 0.121±.005 0.009±.005 0.962 TabTransformer 0.768±.003 0.594±.004 0.497 0.666±.003 0.447 0.083±.017 0.942 0.089±.041 0.020±.004 0.942 TransTab 0.778±.001 0.608±.001 0.477 0.611±.020 0.441 0.339±.009 0.346 0.423±.032 0.400±.025 0.096

Target-Table Learners

Prediction and linkage decouple by model family. Prior-based TABICL leads prediction (AUROC 0.816, Macro-F1 0.671, SGM 0.505), while linkage leaders are dominated by transfer-oriented encoders: BERT on Clean Linkage (F1 0.418, NR 0.096) and GTE on Robust Linkage (NR 0.048), with TRANSTAB second on Robust Linkage (NR 0.096) via its cross-table contrastive objective. Target-table SSL methods are generally competitive on prediction but weak on linkage.

Intra- and inter-table transfer follows training scope. Row prediction operates within a table (intra-table transfer), while record linkage operates across tables (inter-table transfer). Target-table SSL methods, trained from scratch on each target, fit locally: they are competitive on prediction (mean NR 0.48/0.47 on classification/regression) but trail on linkage (0.65/0.65 on Clean/Robust). Transfer-based encoders, applying one shared model to every table, produce comparable row spaces: they lead linkage (mean NR 0.19/0.19) but sit mid-pack on prediction (0.50/0.64). Two designs combine both axes. TRANSTAB layers a cross-table contrastive objective onto per-table SSL, taking second on Robust Linkage (NR 0.096) while staying competitive on prediction. TABICL layers target-table adaptation onto a shared meta-pretrained prior, leading prediction and ranking 5th of 14 on Robust Linkage (NR 0.394). Combining intra-table and inter-table transfer is thus hard but achievable through different design paths, an open direction for further study.

Geometric diagnostics cross-validate the task-based rankings. Alongside the task-based protocol, we also evaluate exported row embeddings through task-free geometric diagnostics (Appendix L.5). The two axes agree: row-linkage utility correlates with embedding anisotropy at |ρ¯| ≈ 0.80 (Linear head, αreq), and regression utility correlates with effective rank at |ρ¯| ≈ 0.36 (MLP head, RankMe⋆). See Figure 9. Intrinsic embedding geometry thus offers a task-free lens on row-level transfer, with broadly consistent agreement across the full diagnostic family.

###### 4.4 Compositional Results on TRL-DLTE

We analyze the full 10 × 8 × 14 = 1120 pipeline space (Stage-1 table encoders × Stage-2 column models × Stage-3 row models, with pool composition given in Appendix N). Table 4 and Figure 3 summarize the resulting top-50 memberships, per-stage marginals, and full pipeline landscape. Three findings matter most.

Capability-matched hybrids beat single-encoder reuse. Under this dev-selection protocol, the best hybrid TUTA/GTE/GTE reaches 0.229 UJ-H, 0.090 above the best dev-selected monolithic BERT/BERT/BERT (0.139). Development and test rankings are similar over the 1,120 pipelines (Spearman ρ = 0.96, top-50 overlap 42/50, see Appendix N), so the dev-selected hybrid result is consistent with the broader test landscape. Frontier presence tracks atomic-task leadership at Stages 2–3 (Table 4): the Stage-2 frontier picks all lead at least one CTBench task (e.g., BERT on Schema NR and Join NR, GTE on Join Search and Table Retrieval, TURL on Table QA), and the Stage-3 top-three frontier picks (GTE, TRANSTAB, TABICL) are exactly the top three row models

on Oracle-RA (Appendix N.4). GTE and TRANSTAB additionally take the top two slots on Robust Linkage NR, with TUTA third.

Table 4: Per-stage view of the DLTE pipeline space (5-round avg, test set). Top 5 models per stage by top-50 frequency. UJH is the per-stage marginal over all 1,120 pipelines. Bottom row lists marginal leaders.

Compositional fit shapes pipeline quality. Atomic strength is thus necessary for frontier presence but not sufficient for the best assembly. Per-stage marginals summarize average main effects, not optimal compositions. The test Stage-1/2/3 marginal leaders assemble to STARMIE/TABBIE/TRANSTAB at 0.134 UJ-H (Fig. 3), well below both the test rank-1 pipeline STARMIE/GTE/GTE (0.253) and the devselected hybrid TUTA/GTE/GTE (0.229). Development marginals assemble to a different pipeline, STARMIE/BERT/TRANSTAB, scoring 0.231 on test (Fig. 3). Marginals therefore carry signal, but the 0.097 swing between the two marginal assemblies, despite this split stability, shows that top marginal ranks are a lossy selection rule. Two decouplings in Table 4 explain why. At Stage 1, atomic retrieval strength decouples from compositional utility: GTE leads CTBench Table Retrieval (0.476 MRR, Table 2) and DLTE target recall@100 (0.801), yet ranks only third on the Stage-1 UJ-H marginal, behind STARMIE and TUTA. At Stage 2, average marginal strength decouples from top-pipeline membership: TABBIE leads the Stage-2 marginal (unrounded) but never enters the top-50, where GTE and TURL dominate. DLTE therefore rewards compositional fit, defined here as non-additive compatibility among retrieval, column alignment, and row matching, not independent per-stage rank alone. These patterns hold separately on TabFact-only and WTQ-only parents (Spearman ρ = 0.871 cross-source rank agreement across all 1,120 canonical pipelines, see Appendix N.8).

Stage Model # % UJ-H

- Stage 1 (Tbl)

Starmie 19 38% 0.144 TUTA 18 36% 0.138 TAPEX 4 8% 0.127 GTE 3 6% 0.129 BERT 3 6% 0.128

- Stage 2 (Col)

TURL 16 32% 0.143 GTE 16 32% 0.141 BERT 10 20% 0.135 TAPAS 7 14% 0.132 TaBERT 1 2% 0.128

- Stage 3 (Row)

TransTab 12 24% 0.132 GTE 12 24% 0.131 TabICL 12 24% 0.130 TUTA 7 14% 0.130 BERT 6 12% 0.128

Per-stage marginal leaders (UJ-H, unrounded means): Starmie 0.144 / TABBIE 0.143 / TransTab 0.132.

A shared identity-resolution capability across RBench and DLTE. DLTE Stage 3 (row matching) is the compositional counterpart of RBench’s Record Linkage task: both test whether exported row embeddings can resolve cross-table row identity under noise. Oracle-RA (Appendix N.4) shows that the two settings share a strong rowmodel rank signal: Stage-3 row-model rankings agree with the RBench Robust Linkage NR ranking at Spearman |ρ| = 0.80 (p = 6.3×10−4). The agreement is strongest at the top two models, GTE and TRANSTAB, with only a small middle-pack shift at TABICL (third on OracleRA versus fifth on Robust Linkage NR). This identifies a shared identity-resolution capability of frozen row embeddings, surfaced consistently by both atomic record linkage (RBench) and compositional row matching (DLTE Stage 3). This Stage-3 row-model ranking persists across all four DLTE noise tiers (Appendix N).

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Stage 1 Stage 2

Stage 3

[Figure 45]

[Figure 46]

###### GTE TURL TABBIE

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Star./BERT/TrTab=0.231

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

BERT / BERT / BERT = 0.139

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

TUTA / GTE / GTE = 0.229

[Figure 63]

[Figure 64]

Star./TABBIE/TrTab=0.134

- Figure 3: DLTE pipeline landscape (test split). Axes sort by per-stage marginal UJ-H. Color encodes end-to-end UJ-H. Halo boxes mark marginal and dev-selected compositions.

###### 4.5 Three Open Gaps Across the Suites

Read jointly, the three suites expose three structural gaps that single-paradigm or single-granularity evaluations cannot isolate. At the level of model choice, TRL-CTBENCH (Sec. 4.2) reveals a specialization gap, where no pretraining recipe behaves like a universal representation. At the level of transfer, TRL-RBENCH (Sec. 4.3) reveals a transfer-scope gap, where intra-table adaptation and cross-table comparability pull learning in different directions. At the level of deployment, TRLDLTE (Sec. 4.4) reveals a composition gap, where granularity choices interact rather than stack independently. Together, these mark where tabular representation research has yet to converge on a unified account.

##### 5 Conclusion

TRL-BENCH reframes tabular encoder evaluation around the artifact many downstream systems actually reuse: exported embeddings. Under a single representation-level protocol, heterogeneous encoders become comparable without conflating their embeddings with task-specific wrappers, retraining budgets, or adaptation. By making this setting measurable across diverse encoders, datasets, and downstream uses, TRL-BENCH provides a common reference point for building tabular models as portable representation learners rather than one-off task solvers.

##### References

- [1] Kumar Krishna Agrawal, Arnab Kumar Mondal, Arna Ghosh, and Blake Richards. α-ReQ: Assessing representation quality in self-supervised learning by measuring eigenspectrum decay. In NeurIPS, 2022.
- [2] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.
- [3] Alcoholrithm. TabularS3L: A PyTorch Lightning-based library for self- and semi-supervised learning on tabular data. https://github.com/Alcoholrithm/TabularS3L, 2024.
- [4] Alessio Ansuini, Alessandro Laio, Jakob H. Macke, and Davide Zoccolan. Intrinsic dimension of data representations in deep neural networks. In NeurIPS, 2019.
- [5] Gilbert Badaro, Mohammed Saeed, and Paolo Papotti. Transformers for tabular data representation: A survey of models and applications. Transactions of the Association for Computational Linguistics, 11:227–249, 2023.
- [6] Dara Bahri, Heinrich Jiang, Yi Tay, and Donald Metzler. SCARF: Self-supervised contrastive learning using random feature corruption. In ICLR, 2022.
- [7] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8): 1798–1828, 2013.
- [8] Bernd Bischl, Giuseppe Casalicchio, Matthias Feurer, Pieter Gijsbers, Frank Hutter, Michel Lang, Rafael Gomes Mantovani, Jan N. van Rijn, and Joaquin Vanschoren. OpenML benchmarking suites. In NeurIPS Datasets and Benchmarks Track, 2021.
- [9] Alex Bogatu, Alvaro A. A. Fernandes, Norman W. Paton, and Nikolaos Konstantinou. Dataset discovery in data lakes. In ICDE, pages 709–720, 2020.
- [10] Vadim Borisov, Kathrin Seßler, Tobias Leemann, Martin Pawelczyk, and Gjergji Kasneci. Language models are realistic tabular data generators. In ICLR, 2023.
- [11] Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and Gjergji Kasneci. Deep neural networks and tabular data: A survey. IEEE Transactions on Neural Networks and Learning Systems, 35(6):7499–7519, 2024.
- [12] Jintai Chen, Jiahuan Yan, Qiyuan Chen, Danny Ziyi Chen, Jian Wu, and Jimeng Sun. ExcelFormer: A neural network surpassing GBDTs on tabular data. arXiv preprint arXiv:2301.02819, 2023.
- [13] Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. TabFact: A large-scale dataset for table-based fact verification. In ICLR, 2020.
- [14] Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 2018.
- [15] Tianji Cong, Madelon Hulsebos, Zhenjie Sun, Paul Groth, and H. V. Jagadish. Observatory: Characterizing embeddings of relational tables. Proceedings of the VLDB Endowment, 17(4): 849–862, 2023.

- [16] Janez Demšar. Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7:1–30, 2006.
- [17] Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and Cong Yu. TURL: Table understanding through representation learning. Proceedings of the VLDB Endowment, 14(3):307–319, 2020.
- [18] Yuhao Deng, Chengliang Chai, Lei Cao, Qin Yuan, Siyuan Chen, Yanrui Yu, Zhaoze Sun, Junyi Wang, Jiajun Li, Ziqi Cao, Kaisen Jin, Chi Zhang, Yuqing Jiang, Yuanfang Zhang, Yuping Wang, Ye Yuan, Guoren Wang, and Nan Tang. LakeBench: A benchmark for discovering joinable and unionable tables in data lakes. Proceedings of the VLDB Endowment, 17(8):1925–1938, 2024. doi: 10.14778/3659437.3659448.
- [19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019.
- [20] Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, and Alexander Smola. AutoGluon-Tabular: Robust and accurate AutoML for structured data. arXiv preprint arXiv:2003.06505, 2020.
- [21] Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, and Frank Hutter. TabArena: A living benchmark for machine learning on tabular data. In NeurIPS Datasets and Benchmarks Track, 2025.
- [22] Elena Facco, Maria d’Errico, Alex Rodriguez, and Alessandro Laio. Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports, 7(1):12140, 2017.
- [23] Grace Fan, Jin Wang, Yuliang Li, and Renée J. Miller. Table discovery in data lakes: State-ofthe-art and future directions. In SIGMOD Companion, 2023.
- [24] Grace Fan, Jin Wang, Yuliang Li, Dan Zhang, and Renée J. Miller. Semantics-aware dataset discovery from data lakes with contextualized column-based representation learning. Proceedings of the VLDB Endowment, 16(7):1726–1739, 2023.
- [25] Sebastian Fischer, Liana Harutyunyan, Matthias Feurer, and Bernd Bischl. OpenML-CTR23: A curated tabular regression benchmarking suite. In AutoML Conference, 2023.
- [26] Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann LeCun. RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank. In ICML, 2023.
- [27] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning models for tabular data. In NeurIPS, 2021.
- [28] Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko. TabR: Tabular deep learning meets nearest neighbors. In ICLR, 2024.
- [29] Jonathan Herzig, Pawel Krzysztof Nowak, Thomas Müller, Francesco Piccinno, and Julian Martin Eisenschlos. TaPas: Weakly supervised table parsing via pre-training. In ACL, 2020.
- [30] Jonathan Herzig, Thomas Müller, Syrine Krichene, and Julian Eisenschlos. Open domain question answering over tables via dense retrieval. In NAACL, 2021.
- [31] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer That solves small tabular classification problems in a second. In ICLR, 2023.
- [32] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 2025.
- [33] Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin. TabTransformer: Tabular data modeling using contextual embeddings. arXiv preprint arXiv:2012.06678, 2020.
- [34] Hiroshi Iida, Dung Thai, Varun Manjunatha, and Mohit Iyyer. TABBIE: Pretrained representations of tabular data. In NAACL, 2021.

- [35] Zhengbao Jiang, Yi Mao, Pengcheng He, Graham Neubig, and Weizhu Chen. OmniTab: Pretraining with natural and synthetic data for few-shot table-based question answering. In NAACL, 2022.
- [36] Ernesto Jiménez-Ruiz, Oktie Hassanzadeh, Vasilis Efthymiou, Jiaoyan Chen, and Kavitha Srinivas. SemTab 2019: Resources to benchmark tabular data to knowledge graph matching systems. In ESWC, pages 514–530, 2020. doi: 10.1007/978-3-030-49461-2_30.
- [37] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547, 2021.
- [38] James Jordon, Jinsung Yoon, and Mihaela van der Schaar. PATE-GAN: Generating synthetic data with differential privacy guarantees. In ICLR, 2019.
- [39] Aamod Khatiwada, Grace Fan, Roee Shraga, Zixuan Chen, Wolfgang Gatterbauer, Renée J. Miller, and Mirek Riedewald. SANTOS: Relationship-based semantic table union search. In SIGMOD, 2023.
- [40] Aamod Khatiwada, Harsha Kokel, Ibrahim Abdelaziz, Subhajit Chaudhury, Julian Dolby, Oktie Hassanzadeh, Zhenhan Huang, Tejaswini Pedapati, Horst Samulowitz, and Kavitha Srinivas. TabSketchFM: Sketch-based tabular representation learning for data discovery over data lakes. In ICDE, 2025.
- [41] Myung Jun Kim, Léo Grinsztajn, and Gaël Varoquaux. CARTE: Pretraining and transfer for tabular learning. In ICML, 2024.
- [42] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
- [43] Keti Korini, Ralph Peeters, and Christian Bizer. SOTAB: The WDC Schema.org table annotation benchmark. In SemTab @ ISWC, 2022.
- [44] Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, and Artem Babenko. TabDDPM: Modelling tabular data with diffusion models. In ICML, 2023.
- [45] Christos Koutras, George Siachamis, Andra Ionescu, Kyriakos Psarakis, Jerry Brons, Marios Fragkoulis, Christoph Lofi, Angela Bonifati, and Asterios Katsifodimos. Valentine: Evaluating matching techniques for dataset discovery. In ICDE, 2021.
- [46] Harold W. Kuhn. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1–2):83–97, 1955.
- [47] Guillaume Lample, Alexis Conneau, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. Word translation without parallel data. In ICLR, 2018.
- [48] Kyungeun Lee, Ye Seul Sim, Hye-Seung Cho, Moonjung Eo, Suhee Yoon, Sanghyu Yoon, and Woohyung Lim. Binning as a pretext task: Improving self-supervised learning in tabular domains. In ICML, 2024.
- [49] Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, and Wang-Chiew Tan. Deep entity matching with pre-trained language models. Proceedings of the VLDB Endowment, 14(1): 50–60, 2020.
- [50] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023.
- [51] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Isolation forest. In ICDM, 2008.
- [52] Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, and Jian-Guang Lou. TAPEX: Table pre-training via learning a neural SQL executor. In ICLR, 2022.
- [53] Sidharth Mudgal, Han Li, Theodoros Rekatsinas, AnHai Doan, Youngchoon Park, Ganesh Krishnan, Rohit Deep, Esteban Arcaute, and Vijay Raghavendra. Deep learning for entity matching: A design space exploration. In SIGMOD, 2018.

- [54] Fatemeh Nargesian, Erkang Zhu, Ken Q. Pu, and Renée J. Miller. Table union search on open data. Proceedings of the VLDB Endowment, 11(7):813–825, 2018.
- [55] Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David Schnurr, Felipe Petroski Such, Kenny Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter Welinder, and Lilian Weng. Text and code embeddings by contrastive pre-training. arXiv preprint arXiv:2201.10005, 2022.
- [56] Jianmo Ni, Gustavo Hernández Ábrego, Noah Constant, Ji Ma, Keith B. Hall, Daniel Cer, and Yinfei Yang. Sentence-T5: Scalable sentence encoders from pre-trained text-to-text models. In Findings of ACL, 2022.
- [57] OpenAI. New embedding models and API updates. https://openai.com/index/ new-embedding-models-and-api-updates/, 2024. Released January 25, 2024.
- [58] Koyena Pal, Aamod Khatiwada, Roee Shraga, and Renée J. Miller. Generative benchmark creation for table union search. arXiv preprint arXiv:2308.03883, 2023.
- [59] Wei Pang, Masoumeh Shafieinejad, Lucy Liu, Stephanie Hazlewood, and Xi He. ClavaDDPM: Multi-relational data synthesis with cluster-guided diffusion models. In NeurIPS, 2024.
- [60] Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. In ACL, 2015.
- [61] Neha Patki, Roy Wedge, and Kalyan Veeramachaneni. The synthetic data vault. In DSAA, 2016.
- [62] Ralph Peeters, Anna Primpeli, Benedikt Wichtlhuber, and Christian Bizer. Using schema.org annotations for training and maintaining product matchers. In WIMS, 2020.
- [63] Anna Primpeli, Ralph Peeters, and Christian Bizer. The WDC training dataset and gold standard for large-scale product matching. In Companion of The 2019 World Wide Web Conference (WWW ’19 Companion), ECNLP Workshop, 2019.
- [64] Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A tabular foundation model for in-context learning on large data. In ICML, 2025.
- [65] Ravid Shwartz-Ziv and Amitai Armon. Tabular data: Deep learning is not all you need. Information Fusion, 81:84–90, 2022.
- [66] Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, and Tom Goldstein. SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training. arXiv preprint arXiv:2106.01342, 2021.
- [67] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. MPNet: Masked and permuted pre-training for language understanding. In NeurIPS, 2020.
- [68] Kavitha Srinivas, Julian Dolby, Ibrahim Abdelaziz, Oktie Hassanzadeh, Harsha Kokel, Aamod Khatiwada, Tejaswini Pedapati, Subhajit Chaudhury, and Horst Samulowitz. LakeBench: Benchmarks for data discovery over data lakes. arXiv preprint arXiv:2307.04217, 2023.
- [69] Aofeng Su, Aowen Wang, Chao Ye, et al. TableGPT2: A large multimodal model with tabular data integration. arXiv preprint arXiv:2411.02059, 2024.
- [70] Yoshihiko Suhara, Jinfeng Li, Yuliang Li, Dan Zhang, Ça˘gatay Demiralp, Chen Chen, and Wang-Chiew Tan. Annotating columns with pre-trained language models. In SIGMOD, 2022.
- [71] Anton Tsitsulin, Marina Munkhoeva, and Bryan Perozzi. Unsupervised embedding quality evaluation. arXiv preprint arXiv:2305.16562, 2023.
- [72] Talip Ucar, Ehsan Hajiramezanali, and Lindsay Edwards. SubTab: Subsetting features of tabular data for self-supervised representation learning. In NeurIPS, 2021.

- [73] Joaquin Vanschoren, Jan N. van Rijn, Bernd Bischl, and Luis Torgo. OpenML: Networked science in machine learning. ACM SIGKDD Explorations Newsletter, 15(2):49–60, 2014.
- [74] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML, 2008.
- [75] Zhiruo Wang, Haoyu Dong, Ran Jia, Jia Li, Zhiyi Fu, Shi Han, and Dongmei Zhang. TUTA: Tree-based transformers for generally structured table pre-training. In KDD, 2021.
- [76] Zifeng Wang and Jimeng Sun. TransTab: Learning transferable tabular transformers across tables. In NeurIPS, 2022.
- [77] Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. Modeling tabular data using conditional GAN. In NeurIPS, 2019.
- [78] Pengcheng Yin, Graham Neubig, Wen tau Yih, and Sebastian Riedel. TaBERT: Pretraining for joint understanding of textual and tabular data. In ACL, 2020.
- [79] Jinsung Yoon, James Jordon, and Mihaela van der Schaar. GAIN: Missing data imputation using generative adversarial nets. In ICML, 2018.
- [80] Jinsung Yoon, Yao Zhang, James Jordon, and Mihaela van der Schaar. VIME: Extending the success of self- and semi-supervised learning to tabular domain. In NeurIPS, 2020.
- [81] Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In EMNLP, 2018.
- [82] Dan Zhang, Yoshihiko Suhara, Jinfeng Li, Madelon Hulsebos, Ça˘gatay Demiralp, and WangChiew Tan. Sato: Contextual semantic type detection in tables. Proceedings of the VLDB Endowment, 13(11):1835–1848, 2020.
- [83] Hengrui Zhang, Jiani Zhang, Balasubramaniam Srinivasan, Zhengyuan Shen, Xiao Qin, Christos Faloutsos, Huzefa Rangwala, and George Karypis. Mixed-type tabular data synthesis with score-based diffusion in latent space. In ICLR, 2024.
- [84] Jun Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao. PrivBayes: Private data release via bayesian networks. ACM Transactions on Database Systems, 2017.
- [85] Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. TableLlama: Towards open large generalist models for tables. In NAACL, 2024.
- [86] Yi Zhang and Zachary G. Ives. Finding related tables in data lakes for interactive data science. In SIGMOD, 2020.
- [87] Bingzhao Zhu, Xingjian Shi, Nick Erickson, Mu Li, George Karypis, and Mahsa Shoaran. XTab: Cross-table pretraining for tabular transformers. In ICML, 2023.
- [88] Erkang Zhu, Dong Deng, Fatemeh Nargesian, and Renée J. Miller. JOSIE: Overlap set similarity search for finding joinable tables in data lakes. In SIGMOD, 2019.

### Appendix

- A Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B Extended Related Work and Scope . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C Model Inventory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D Model Input Policy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- E Appendix Task Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23 E.1 Benchmark Protocol Adaptations under Frozen Multi-Granular Transfer . . . . . . 23
- F Full Dataset Inventory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- G Task-Local Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- H Metric Definitions and Normalized Rank . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- I Table-Footprint Coverage Across Suites . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- J Family-Level Performance Summary Figure . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- K CTBench Diagnostics and Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

- K.1 Observational Lexical-Accessibility Proxies for CTBench . . . . . . . . . . . . . . . . 36
- K.2 Probe Head Complexity for Column/Table-Level Tasks . . . . . . . . . . . . . . . . . 36
- K.3 Aggregation Ablation for Table-Level Embeddings . . . . . . . . . . . . . . . . . . . . 37
- K.4 Join Search: Direct Cosine vs. Learned Projection . . . . . . . . . . . . . . . . . . . . . 37
- K.5 Union Search: TUS vs. TUS-hard . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
- K.6 Table Retrieval: Model-Only vs. Hybrid Mode . . . . . . . . . . . . . . . . . . . . . . . 38
- K.7 Query Encoder Sensitivity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- K.8 Pair-Level Random vs. Table-Disjoint Split Ablation . . . . . . . . . . . . . . . . . . . 41

- L RBench Diagnostics and Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43

- L.1 Row-Prediction Probe Diagnostics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- L.2 Embedding Dimension for Record Linkage . . . . . . . . . . . . . . . . . . . . . . . . . 45
- L.3 Probe Head for Record Linkage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
- L.4 Record Linkage Split and Leakage Audit . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- L.5 Intrinsic Embedding-Geometry Diagnostics for Row Encoders . . . . . . . . . . . . . 49
- L.6 Intrinsic-Geometry Diagnostics: Per-Head Breakdowns . . . . . . . . . . . . . . . . . 53

- M DLTE Operator Specification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55

- M.1 Stage-1 Retrieval: Scoring and Pool . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
- M.2 Stage-2 Alignment and Classification: Thresholds and Grid . . . . . . . . . . . . . . 55
- M.3 Stage-3 Row Matching: CSLS and Profiles . . . . . . . . . . . . . . . . . . . . . . . . . 56
- M.4 Stage-3 Second-Pass Join on Union-Appended Rows . . . . . . . . . . . . . . . . . . 57
- M.5 What is and is not tuned . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57

- N DLTE Detailed Rankings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58

- N.1 Cell F1 as a Complementary Diagnostic . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
- N.2 Pipeline Component Sensitivity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
- N.3 Per-Stage Marginal Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
- N.4 Oracle-RA Row-Model Diagnostic . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62
- N.5 Full Per-Stage Model Rankings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63
- N.6 Stage 3 Row Matching: Full Pipeline Rankings . . . . . . . . . . . . . . . . . . . . . . . 65
- N.7 Top-20 Pipeline Combinations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
- N.8 Source Split: TabFact vs. WTQ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68

- O Proprietary Embedding Ablation: Retrieval vs. Structural Grounding . . . . . . . . 70
- P Robustness . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71

- P.1 Unified Aggregation Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71
- P.2 Task and Dataset Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72
- P.3 Sample Fidelity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72
- P.4 Perturbation Robustness . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
- P.5 Row/Column Order Insignificance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
- P.6 Cross-Task Comparison of Shared Models . . . . . . . . . . . . . . . . . . . . . . . . . . 74
- P.7 Implementation Notes and Caveats . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75

- Q Computational Efficiency . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 76
- R Reproducibility Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 78
- S Statistical Reporting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79
- T Dataset Counting Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 82
- U Broader Impact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 83
- V Licenses and Asset Documentation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 84

##### A Limitations

TRL-BENCH is designed to standardize cross-paradigm comparison at the representation level, not to report the best end-to-end system for each task. Its scores therefore answer a narrower question, namely what common lightweight readouts can extract from exported embeddings, rather than replacing fully adapted-system benchmarks. The protocol standardizes task definitions and downstream evaluation, not raw model preprocessing: each encoder is run in its documented operating regime through its standard supported wrapper rather than a single forced serialization, which improves fidelity to the original models but leaves some wrapper-induced variation. Appendix D summarizes this policy, and the exact wrapper settings are documented in the released code. Finally, dataset counts and normalized-rank summaries are convenience aggregates and should be read alongside the per-task results.

##### B Extended Related Work and Scope

This section provides the fuller narrative context omitted from the compressed main-text related-work discussion. Table 1 in the main paper remains the compact structural summary.

Row-level tabular learning and evaluation. A large body of work in tabular deep learning focuses on row-level predictive modeling, using feature-token architectures, denoising losses, contrastive selfsupervision, cross-table transfer, or prior-based/meta-pretrained predictors [33, 66, 72, 48, 80, 6, 74, 76, 31, 32, 64]. See Borisov et al. [11] and Shwartz-Ziv and Armon [65] for surveys. Recent strong backbones such as FT-Transformer, ExcelFormer, TabR, CARTE, and XTab further strengthen this row-level tradition [27, 12, 28, 41, 87]. Because these methods are usually presented as end-to-end row predictors rather than as encoders exposing reusable row embeddings under a shared multigranular representation-level protocol, we treat them as important row-level references rather than as direct baselines for TRL-BENCH’s encode-once comparison. These methods have substantially improved supervised tabular prediction, but their evaluation is still centered on row-level predictive benchmarks, which do not test whether one exported row representation can be reused across multiple targets from the same table or transferred to inter-table tasks such as record linkage.

Column- and table-centric representation learning. A separate line of work studies schema semantics, table-language grounding, retrieval, and discovery. Generic text encoders can be applied to serialized tables [19, 50], while table-aware and column-aware models [29, 78, 52, 17, 75, 34, 24, 40] inject structural or contrastive inductive bias through pretraining. Their evaluations are typically task-specific (semantic typing, relation prediction, question answering, table retrieval, or data discovery) rather than unified transfer across columns, rows, and tables. Across both rowlevel and column/table-level work, evaluation remains largely single-granularity and task-specific, which makes cross-granularity transfer hard to assess. A notable exception is Observatory [15], which characterizes learned representations along perturbation- and invariance-style properties such as sample fidelity, perturbation robustness, and order insignificance, but does not relate those measurements to downstream transfer on the same encoder outputs. Our robustness appendix directly applies Observatory’s three diagnostics to the models in our benchmark pool, and Appendix L.5 additionally pairs downstream row-level evaluation with intrinsic embedding-geometry diagnostics (spectral spread, spectral shape, spatial structure) from the broader representation-learning literature on the same exported row embeddings used by the benchmark protocol, cross-validating the two views. The main benchmark suites themselves are organized around downstream task transfer under a shared representation-level protocol.

Task-specific benchmarks and broader evaluation suites. The tabular ecosystem already contains strong benchmarks for semantic typing and schema understanding [36, 82, 43], schema matching [45], join and union discovery [39, 68, 18, 24], entity matching [53, 63], table question answering [60], question-to-table retrieval [30], and row prediction [73, 8, 25, 21]. LakeBench [68] is the closest prior resource, covering both join and union discovery with systematic model comparisons, but it does not evaluate row-level transfer, record linkage, or multi-stage enrichment composition, and it does not separate training regimes. TabArena [21] provides carefully curated row-level benchmarking with living maintenance, but evaluates supervised end-to-end prediction rather than frozen-representation transfer. These resources are indispensable ingredients for TRL-BENCH, but they differ substantially in split design, task formulation, supervision, and permitted adaptation, and they generally do not combine representation-level downstream evaluation with explicit separation of training regimes under a common protocol. Prior work on data lakes and table augmentation, including discovery systems such as D3L [9] and Juneau [86], typically evaluates retrieval, join recommendation, or schema alignment stages in isolation [39, 68, 18, 24]. TRL-DLTE instead benchmarks their composition under a common representation-centric protocol.

Scope and model selection. TRL-BENCH complements these model papers and task-specific benchmarks rather than replacing them. Rather than reproducing each task family’s strongest end-toend stack, it standardizes comparison around transferable signal already present in a representation. Accordingly, the main leaderboard focuses on models that can participate in the shared representationlevel protocol and keeps scale roughly controlled to the ∼1M–1B parameter band studied in this paper. End-to-end generative table LLMs and heavily task-specific fine-tuned systems (e.g., TableLlama, TableGPT2, Ditto, DODUO, OmniTab) are complementary but out of scope for this comparison [85,

69, 49, 70, 35]: the former typically operate at 7B+ scale and do not expose compatible stable multigranular embeddings, while the latter belong to a separate task-specific end-to-end paradigm. The benchmark differs from prior work in three ways. First, it is explicitly multi-granular across columns, rows, and tables. Second, it is explicitly cross-paradigm, comparing externally pretrained transfer, target-table self-supervision, and prior-based predictors under one interface. Third, it includes a compositional data-lake enrichment benchmark that tests whether strong atomic representations actually compose into a useful end-to-end system.

Other tabular ML threads. Tabular data is studied across many research threads in addition to representation learning. Tabular synthesis spans GANs, diffusion models, language models, and relational generators [77, 44, 83, 10, 61, 59], with privacy-preserving variants targeting differentialprivacy guarantees [38, 84]. Other threads include AutoML and hyperparameter-optimization stacks for tabular prediction [20], causal inference and treatment-effect estimation on tabular covariates [14], anomaly and outlier detection [51], and missing-value imputation [79]. Although diverse in goal, several of these threads increasingly intersect with representation learning, whether through learned latent spaces in generative models or embedding-based scoring in anomaly and outlier detection, making the quality of reusable encoders a cross-cutting concern that TRL-BENCH is designed to characterize.

##### C Model Inventory

- Table 5: Model inventory in TRL-BENCH. Parameter counts come from the loaded checkpoint (frozen encoders), the official Python package (TabICL, TabPFN), or the trained model under the default training configuration (target-table SSL, per-model architectures in Appendix D). ‡Retrained per dataset, reported at the median input-feature count (31) across the 50 TRL-RBENCH OpenML tables.

Model Granularity Family Adaptation Pretraining source Dim Params

BERT Column, Row, Table Generic text encoder Frozen General text 768 110M GTE Column, Row, Table Generic text encoder Frozen General text 768 110M TabSketchFM Column, Table Column-specialized Frozen Tables only 768 135M TAPAS Column, Table Table-Text Frozen Tables + NL 768 111M TaBERT Column, Table Table-Text Frozen Tables + NL 768 181M TABBIE Column, Row, Table Structure-aware Frozen Tables only 768 170M Starmie Column, Table Column-specialized Frozen Tables only (contrastive) 768 125M TURL Column, Table Structure-aware Frozen Entity-linked tables 312 314M TUTA Table, Row Structure-aware Frozen Tables only 768 134M TAPEX Table Table-Text Frozen Tables + SQL execution 768 139M TabICL Row Prior-based Meta-pretrained Tabular tasks/examples 512 27M TabPFN Row Prior-based Meta-pretrained Synthetic priors 192 11M SCARF Row Target-table SSL Target-table SSL Target-table unlabeled data 512 0.7M‡ SAINT Row Target-table SSL Target-table SSL Target-table unlabeled data 512 38M SubTab Row Target-table SSL Target-table SSL Target-table unlabeled data 512 0.8M‡ TabTransformer Row Target-table SSL Target-table SSL Target-table unlabeled data 512 0.7M‡ TabBinning Row Target-table SSL Target-table SSL Target-table unlabeled data 512 0.7M‡ TransTab Row Target-table SSL Target-table SSL Target-table unlabeled data 512 19M VIME Row Target-table SSL Target-table SSL Target-table unlabeled data 512 0.7M‡ DAE Row Target-table SSL Target-table SSL Target-table unlabeled data 512 0.6M‡

##### D Model Input Policy

Tabular encoders expose different native interfaces, for example serialized text, structured cells, or direct feature tensors, and therefore come with model-specific preprocessing and context regimes. In TRL-BENCH, each encoder is evaluated under its standard public operating regime: the preprocessing and context configuration described in its source paper or released with its official code, using the paper-default or released-default settings whenever available. When several supported settings are possible, we choose the most permissive deterministic setting that stays within the model’s documented operating regime and preserves benchmark coverage. We deliberately do not force a single common input serialization or context length, because doing so would push most models outside the regime in which they were designed and validated, and would distort rather than equalize the comparison. The benchmark should therefore be read as a comparison of encoder packages under their supported interfaces, with the per-model launch scripts (exact tokenization, truncation, and configuration) provided in the codebase.

Of the 20 models, 7 target-table SSL methods (VIME, SCARF, SUBTAB, TABBINNING, TABTRANSFORMER, SAINT, DAE) are implemented through the TabularS3L framework [3], and 2 generic text encoders (BERT, GTE) are applied to serialized tabular input. Wrappers also introduce model-specific choices that we note here. TABBIE and TUTA row embeddings are synthesized via per-row mini-tables (no published row-extraction head exists in either model). TABPFN is run with ignore_pretraining_limits=True and TURL with max_entities=12000, both librarysupported settings that admit inputs beyond the models’ pretrained sizes. DAE’s wrapper uses TabularS3L’s tabular Swap corruption [80]; see also [74] for the original image-domain DAE. Permodel citations are in Table 6. Per-model launch scripts (exact tokenization, truncation, output extraction) live in the released codebase.

For target-table SSL methods, the Table 5 parameter counts are measured from the trained model under the default training configuration: hidden size 512 with 3 hidden layers for the MLP-based learners; SAINT uses a 6-layer d=512 feature-token transformer; TRANSTAB uses a 2-layer d=512 encoder whose size is dominated by a ∼16M shared token-embedding table. Because these models are retrained per dataset, the backbone grows approximately linearly with input dimension (feature counts range from 7 to 1,777 across the 50 TRL-RBENCH OpenML tables), yielding an empirical upper bound of roughly 1.6–2.0M parameters for the largest-feature dataset.

- Table 6: Per-model source provenance. The Source column lists the model’s source paper and (where relevant) the implementation library. Wrapper scripts (exact tokenization, truncation, output extraction) live in models/<model>/ in the released codebase. Wrapper-introduced choices are documented in the policy paragraph above.

Model Family Source

BERT Generic text [19] GTE Generic text [50]

TAPAS Table-Text [29] TaBERT Table-Text [78] TAPEX Table-Text [52]

Starmie Column-spec. [24] TabSketchFM Column-spec. [40]

TABBIE Struct.-aware [34] TURL Struct.-aware [17] TUTA Struct.-aware [75]

TabICL Prior-based [64] TabPFN Prior-based [31, 32]

TransTab Target-table SSL [76] VIME Target-table SSL [80]; [3] SCARF Target-table SSL [6]; [3] SubTab Target-table SSL [72]; [3] TabBinning Target-table SSL [48]; [3] TabTransformer Target-table SSL [33]; [3] SAINT Target-table SSL [66]; [3] DAE Target-table SSL [74]; [80]; [3]

- Table 7: Summary of column- and table-level tasks in TRL-BENCH. Module form is the fine-grained downstream-module instantiation. These map to the three downstream-module types of Sec. 3.1: Geometry → training-free, Probe and Learned proj. → learned, and Dual proj. and Decoder → query-conditioned. DLTE’s Pipeline module (Table 8) is a stage-wise composition specified in Sec. 3.4. Split: † = table-disjoint (re-split by this project), orig. = table-disjoint in source data, – = no split (training-free task).

Task Level Module form Split Protocol summary Metric Sources Column Type Pred. Column Probe Original Frozen column embeddings

with linear / MLP probe

F1 SATO, SOTAB Column Clustering Column Geometry – Training-free clustering on

frozen column embeddings

NMI SATO, SOTAB

Column Relation Pred. Column Probe Original Ordered column-pair probe over concatenated frozen embeddings

F1 WikiCT (relation)

Join Search Column Learned proj. Query-disjoint Retrieval over frozen column embeddings with a small learned projection

MAP OpenData variants

Column Overlap Column Probe Table-disjoint† Frozen column-pair regression probe

nRMSE Wiki Containment

Union Search Column Geometry – Retrieval under one-to-one column alignment over frozen embeddings

MAP SANTOS, UGEN, TUS

Schema Matching Column Geometry – Training-free ranking of crosstable column pairs by cosine similarity

R@GT Valentine

Table QA Column Decoder Original Table QA with frozen column representations on the table side

Accuracy WTQ

Join Classification Table Probe Table-disjoint† Frozen table-pair probe with lightweight head

F1 Spider Join Union Classification Table Probe Table-disjoint† Frozen table-pair probe with

lightweight head

F1 Wiki Union Union Regression Table Probe Table-disjoint† Frozen table-pair regression

probe

nRMSE ECB Union Table Subset Table Probe Tbl-disjoint (orig.) Frozen table-pair probe with

lightweight head

F1 CKAN Subset

Table Retrieval Table Dual proj. Original Dual-projection question-totable retrieval over frozen table embeddings

MRR NQ-Tables

- Table 8: Summary of row-level and compositional tasks in TRL-BENCH. See Table 7 caption for column definitions.

Task Level Module form Split Protocol summary Metric Sources

Macro-F1, AUROC, SGM

Row Prediction Row Probe Original (OpenML) One frozen row embedding per record, reused across multiple targets from the same table

50 OpenML tables

Record Linkage Row Probe Original (source) Pair classification over concatenated frozen row embeddings; headline averages linear and MLP probe heads

F1 DeepMatcher, WDC

Tbl + Col

DLTE

Pipeline Parent-disjoint Three-stage retrieval → alignment → merge over frozen encoder outputs; operators in Appendix M

UJ-H, Cell F1 TabFact, WTQ

+ Row

- E Appendix Task Summary E.1 Benchmark Protocol Adaptations under Frozen Multi-Granular Transfer

The protocol of Section 3.1 fixes the common evaluation infrastructure (frozen embeddings, shared lightweight readouts), but several reused source tasks need targeted protocol adaptations to remain meaningful tests of multi-granular representation transfer. Each paragraph below states one consideration, the choice we adopt in TRL-BENCH, and the empirical evidence motivating that choice within this setting.

Cross-table generalization in reused pair tasks. Affected tasks: Join Classification, Column Overlap, Union Classification, Union Regression. Choice: table-disjoint train/dev/test splits. Why: These source tasks were created for different objectives; for our transfer-oriented use, table-disjoint

- Table 9: Grouping of the 16 benchmark tasks by downstream-module type (Sec. 3.1). The three primary module types are training-free, learned, and query-conditioned. DLTE is handled separately as a multi-stage pipeline (Sec. 3.4). The fine-grained Module form entries in Tables 7–8 refine this taxonomy.

Downstream-module type Tasks

Training-free Column Clustering; Union Search; Schema Matching Learned Column Type Pred.; Column Relation Pred.; Join Search; Column Overlap; Join Classification; Union

Classification; Union Regression; Table Subset; Row Prediction; Record Linkage Query-conditioned Table QA; Table Retrieval Pipeline DLTE

splits ensure test tables are unseen during training and lower Join Classification F1 by 0.212 on average relative to pair-random splits (Table 23).

High-overlap positives in union search. Affected task: Union Search (TUS). Choice: TUS-hard variant (containment ≥ 0.70 removed). Why: In our frozen retrieval setting, removing the highestoverlap positives helps distinguish lexical overlap from broader union signal; the value-overlap baseline drops from 1.000 to 0.008 and rankings shift substantially (Table 20).

Degenerate or mislabeled targets. Affected task: Row Prediction. Choice: human review, label repair, and degeneracy audits on 158 candidate tables, retaining 50 for release. Why: Removes constant-column targets, near-duplicate targets, and labeling issues from the OpenML candidate pool before reusable row-transfer evaluation.

Label-equivalent columns leaking match identity. Affected datasets: Record Linkage on WDC and Fodors–Zagats. Choice: remove cluster_id and identifiers (WDC) and class (Fodors– Zagats) before any encoder consumes a row. Why: These columns are deterministic functions of the match label; without removal, frozen text encoders trivially reach near-perfect F1 on these sources (Appendix L.4).

Train/test row overlap in reused linkage sources. Affected task: Record Linkage. Choice: retain source pair-disjoint splits, audit per-source row overlap, and report a strict row-disjoint ablation on the

- 10 viable sources. Why: Pair-disjoint splits are the entity-matching canon; the strict ablation confirms rankings are stable across all 14 row models (Spearman ρ = 0.94, p = 5.6 × 10−7; Appendix L.4).

End-to-end scoring across removed blocks. Affected task: DLTE. Choice: UJ-H (harmonic mean of union and join recall) as the primary metric. Why: UJ-H directly tracks recovery of both removed blocks; Cell F1 (pooled cell-recovery yield) is reported as a complementary diagnostic in Appendix N.1.

Retrieval difficulty in DLTE. Affected task: DLTE (Stage 1). Choice: include 36,740 CKAN distractor tables in the lake. Why: A large distractor pool makes retrieval a meaningful tablerepresentation test; the lake contains 11,032 targets among 47,772 tables total.

Shared query-side signal in hybrid retrieval. Affected task: Table Retrieval. Choice: model-only mode (no query-encoder table embedding concatenated on the table side). Why: Hybrid mode adds a strong common signal from the query encoder and compresses model differences into a narrow MRR band (Table 21), whereas model-only better isolates table-side transfer.

##### F Full Dataset Inventory

CTBench datasets (20). Schema Understanding: SATO, SOTAB, WikiCT (relation). Joinability: OpenData (main), OpenData CAN, OpenData USA, OpenData UK/SG, Wiki Containment (wiki_containment), Spider Join (spider_join). Unionability: SANTOS, UGEN-v1, UGEN-v2, TUS, TUS-hard, Valentine, Wiki Union (wiki_union), ECB Union (ecb_union), CKAN Subset (ckan_subset). Grounding: WTQ (WikiTableQuestions), NQ-Tables.

Row-Prediction dataset inventory real statistics from the 50 OpenML tables

###### (a) Subject domain of the 50 source datasets

(b) Table-level task profile (50 tables)

Games & Other

Education

Classification-only

Finance & Economics

4 (8.0%)

2 (4.0%)

###### 11 (22.0%)

###### 12 (24.0%)

Software & Security

6 (12.0%)

3 (6.0%)

Regression-only

6 (12.0%)

8 (16.0%)

Engineering & Industrial

36 (72.0%)

Business & Marketing

6 (12.0%)

###### 6 (12.0%)

Mixed (CLF + REG)

Natural Sciences

Healthcare & Medicine

- Figure 4: Row-Prediction dataset inventory. Real statistics computed from the 50 source datasets and their per-target metadata. (a) Subject-domain distribution of the 50 OpenML tables, hand-curated from each dataset’s public OpenML description: 12 Finance & Economics, 8 Business & Marketing, 6 Healthcare & Medicine, 6 Natural Sciences, 6 Engineering & Industrial, 6 Software & Security, 2 Education, and 4 Games & Other. (b) Table-level task profile of the 50 tables: 11 host only classification targets, 3 host only regression targets, and 36 host both classification and regression targets on the same table, enabling the “one frozen row embedding reused across multiple targets” protocol of TRL-BENCH.

Row prediction (50 OpenML tables, 123 targets). OpenML dataset IDs: 3, 38, 458, 1063, 1486, 4534, 6332, 40668, 40966, 40978, 44958, 44967, 44975, 44984, 44992, 46906, 46907, 46908, 46910, 46911, 46912, 46915, 46916, 46918, 46919, 46920, 46922, 46923, 46927, 46929, 46930, 46932, 46933, 46934, 46935, 46937, 46939, 46940, 46950, 46952, 46955, 46956, 46958, 46960, 46961, 46963, 46964, 46969, 46979, 46980. Per-table target counts range from 2 to 3 (77 classification + 46 regression). Sourced from TabArena, OpenML-CC18, and OpenML-CTR23; filtered from 158 candidates.

Record linkage (16 datasets). DeepMatcher clean (8): amazon-google, beer, dblp-acm, dblpscholar, fodors-zagats, itunes-amazon, walmart-amazon, abt-buy. DeepMatcher dirty (4): dblp-acm, dblp-scholar, itunes-amazon, walmart-amazon. WDC Products (4 sizes): small (∼2.5K pairs), medium (∼8K pairs), large (∼18K pairs), xlarge (∼30K pairs).

DLTE. Parent tables: 1,379 (989 from TabFact, 390 from WTQ). Split: 827/207/345 train/dev/test. Fragments: 5,516 seeds + 5,516 union targets + 5,516 join targets = 16,548 total. Distractors: 36,740 CKAN tables. Total lake: 47,772 tables. Noise tiers: clean, schema, cell, hard.

- Table 10: Per-task applicability of task-local baselines in TRL-BENCH. ✓ = baseline is evaluated on the task. Blank = not applicable. Coverage rationale is in Appendix G.

Inv. Idx.

Hung. Set

Cos. Thr.

Rnd. TF-IDF Chance Dum.

Jacc. Dist.

Task

Column Clustering ✓ ✓ ✓ Column Type Prediction ✓ ✓ ✓ ✓ Column Relation Prediction ✓ ✓ ✓ Join Search (cosine) ✓ ✓ ✓ Join Search (learned) ✓ ✓ Column Overlap ✓ ✓ ✓ Union Search ✓ ✓ ✓ Schema Matching ✓ ✓ ✓ ✓ Table QA ✓ ✓ Join Classification ✓ ✓ ✓ Table Subset ✓ ✓ ✓ Union Classification ✓ ✓ ✓ Union Regression ✓ ✓ ✓ Table Retrieval ✓ ✓ Record Linkage ✓ ✓ ✓ ✓ Row Prediction ✓ ✓ ✓

##### G Task-Local Baselines

Each downstream task in TRL-BENCH includes simple task-local baselines in addition to learned encoders. These baselines serve three distinct roles. Embedding baselines (Random, TF-IDF) produce frozen vectors that flow through the same downstream pipeline as neural encoders. They test whether the pipeline itself drives performance rather than the embedding. Embedding-free baselines (InvertedIndex Containment, Hungarian Set Match, Jaccard and Distribution matchers from Valentine) bypass the pipeline and operate on raw table data. They represent classical, task-specific methods and provide a reference for what is achievable without learned representations. Analytical baselines (Chance, Dummy) compute expected performance from dataset statistics alone. Table 10 lists per-task applicability. The baselines themselves are documented below.

Random embeddings. Random vectors of the same dimension as the neural encoder replace learned embeddings throughout the downstream pipeline, at every supported granularity (column, row, table). Any benchmark entry that significantly under-performs Random indicates a failure mode. Entries that match Random indicate that the probe head, not the encoder, is doing the work.

Chance and Dummy. Chance is an analytical floor: for classification tasks, the expected score from random class assignment proportional to class frequencies; for retrieval, the expected recall from uniform random ranking. Dummy is the strictly stronger majority-class (classification) or mean (regression) predictor trained on frozen embeddings. It exposes cases where the label distribution alone is enough. Every supervised probe task reports a dummy head alongside the linear and MLP heads.

TF-IDF embeddings. A character-n-gram TF-IDF vectorizer (charwb analyzer, range [3,5], 256 dimensions) produces a per-column embedding from the column header concatenated with up to 50 sampled cell values. Fit per dataset. Restricted to the two column-level tasks with compatible serialization (column type prediction and column clustering) on the sato and SOTAB datasets.

TF-IDF row embeddings. A row-level analogue of the column TF-IDF baseline, used as the string-similarity reference for record linkage. Each row is serialized as col: val | col: val | ... (the same template used by the GTE / BERT row encoders), then character-n-gram TF-IDF (charwb, range [3,5], 512 dimensions) is fit per dataset on the union of tableA and tableB rows so that paired rows live in a shared vocabulary. The resulting row vectors flow through the same downstream record-linkage probe as the neural row encoders, including all four heads (cosine threshold, linear, MLP, dummy). Because record linkage is dominated by surface character overlap on many sources

(Sec. 3.3), this baseline is the appropriate floor for a learned row encoder: an encoder that does not beat TF-IDF row at matching head type is not capturing match signal beyond raw character overlap.

Jaccard token-overlap row embeddings. A second row-level non-neural baseline that complements char-TF-IDF row at the token level. Each row is serialized identically and tokenized into word unigrams. Rows are then represented as L2-normalized binary token-presence vectors (TfidfVectorizer(analyzer=’word’, binary=True, use_idf=False, norm=’l2’), 512 dimensions). Cosine of two such vectors equals the Ochiai coefficient |A ∩ B|/ |A| · |B|, a token-overlap similarity that is a near-monotone transform of the token Jaccard |A ∩ B|/|A ∪ B|, so under the cosine-threshold head this baseline reads as a token-overlap threshold close to a Jaccard threshold. Under the learned MLP / linear heads the per-token presence indicators remain directly accessible to the probe. Together, char-TF-IDF row (sub-word) and Jaccard row (token-level) form the two-way string-similarity floor against which a neural row encoder must compete on linkage. On the canonical avg(MLP, linear) probe protocol, TF-IDF row reaches F1 = 0.380/0.495/0.227 on DM-C / DM-D / WDC, and Jaccard row reaches F1 = 0.353/0.481/0.255 on the same subsets, both well above the Random / Dummy floors (Table 12) and below the strongest learned encoders (Table 3).

Inverted-Index Containment. An embedding-free join-search baseline. Given a query column Q and a lake column C, the score is the containment |Q ∩ C|/|Q| computed via an inverted index over normalized cell values (posting-list prune at 10,000 to remove ubiquitous values). This is a strong baseline in the JOSIE [88] lineage of work, since LakeBench-style ground truth is defined by value overlap.

Hungarian Set Match. The union-search counterpart to Inverted-Index Containment. Columnto-column containment scores are assembled into a bipartite matrix and solved with the Hungarian algorithm [46]. The table-level score is the mean of matched column scores. Identical to the embedding-based union-search pipeline except cosine similarity is replaced by value containment.

Valentine matchers (Jaccard, Distribution). Two embedding-free schema-matching baselines from the Valentine library. Jaccard scores column pairs by a weighted combination of header character 3-gram Jaccard and value-set Jaccard. Distribution scores by comparing value distributions (KS/EMD for numerical columns, frequency for categorical). Both are evaluated on the Valentine benchmark under the same Recall@GT metric as embedding-based schema matching.

Cosine-threshold (record linkage). An unsupervised head that replaces the learned linear/MLP probe with a single cosine-similarity threshold between paired row embeddings, tuned on the validation split. It is reported as a fourth head in Table 27 and consolidated in Table 13 here. It is the only baseline that specifically exercises geometry rather than supervised readout.

Per-dataset results. Tables 11–13 report 5-seed means (± std) for every baseline on every dataset where it applies. Table 11 gives the embedding-free matchers. Table 12 summarizes Random, Dummy, and TF-IDF by task-level means (probe tasks) or by classification/regression/linkage block (row-level tasks). Table 13 gives Cosine-threshold across the 14 row models used in Table 3. Overlap with the main results tables is intentional and serves as a cross-validation check: e.g., Jaccard’s Valentine R@GT matches the Best∗ marker (d) of Table 2, and Cosine-threshold values match the Cos column of Table 27.

- Table 11: Non-neural matching baselines, per-dataset 5-seed means. Values are deterministic given fixed inputs. Reported ± indicates seed-to-seed variation in the upstream data pipeline. Metrics: col-MAP for Inv.-Index Containment, MAP@10 for Hungarian Set Match, Recall@GT for Valentine matchers.

Baseline Dataset Value nseeds Inv.-Index Cont. opendata 0.148±0.000 5

opendata_CAN 0.182±0.000 5 opendata_USA 0.138±0.000 5 opendata_UK_SG 0.152±0.000 5

Hungarian Set Match santos 0.975±0.000 5 tus 1.000±0.000 5 tus_hard 0.008±0.000 5 ugen_v1 0.647±0.000 5 ugen_v2 0.239±0.000 5

Jaccard (Valentine) valentine 0.473±0.000 5 Distribution (Valentine) valentine 0.394±0.002 5

- Table 12: Random, Dummy, and TF-IDF baselines per probe task (5-seed mean ± std). Values are computed under the paper’s main convention (Tables 2 and 3): avg(MLP, linear) probe where both heads exist, best table-aggregation per task, strict (table-disjoint) splits for the †-marked pairwise tasks (COLOVERLAP, JOINCLS, UNIONCLS, UNIONREG), TBLRET via the model_only retrieval

pipeline, and binary match-class F1 for record linkage. NA = baseline not applicable to the task per Table 10.

Task (metric) Random Dummy TF-IDF

ColType (F1) 0.132±0.003 0.178±0.000 0.813±0.007 ColClust (NMI) 0.032±0.001 NA 0.400±0.001

ColRel (F1) 0.015±0.001 0.000±0.000 NA ColOverlap (nRMSE) 1.014±0.003 1.000±0.000 NA JoinCls (F1) 0.516±0.014 0.414±0.000 NA UnionCls (F1) 0.500±0.004 0.331±0.000 NA UnionReg (nRMSE) 1.138±0.024 1.001±0.000 NA TblSubset (F1) 0.458±0.027 0.369±0.000 NA TblQA (Acc) 0.204±0.004 NA NA TblRet (MRR) 0.131±0.120 NA NA

RowPred (AUROC) 0.506±0.001 0.500±0.000 NA RowPred (Macro-F1) 0.348±0.001 0.304±0.000 NA RowPred Reg. (SGM↓) 1.103±0.000 1.004±0.000 NA

RecLink DM-C (F1) 0.179±0.007 0.000±0.000 NA RecLink DM-D (F1) 0.223±0.003 0.000±0.000 NA RecLink WDC (F1) 0.128±0.003 0.000±0.000 NA

- Table 13: Cosine-threshold baseline on record linkage (binary match-class F1, 5-seed mean ± std). An unsupervised baseline that thresholds cosine similarity between frozen row embeddings. The same numbers appear in the “Cos” column of Table 27.

Model DM-C DM-D WDC All

BERT 0.390±0.000 0.315±0.000 0.390±0.000 0.371±0.000 GTE 0.698±0.000 0.728±0.000 0.511±0.000 0.659±0.000 TABBIE 0.309±0.000 0.296±0.000 0.389±0.000 0.326±0.000 TUTA 0.363±0.000 0.397±0.000 0.354±0.000 0.369±0.000 TabICL 0.377±0.000 0.328±0.000 0.341±0.000 0.356±0.000 TabPFN 0.260±0.007 0.285±0.000 0.387±0.000 0.298±0.004 VIME 0.242±0.000 0.295±0.000 0.423±0.000 0.301±0.000 SCARF 0.350±0.001 0.352±0.002 0.357±0.000 0.352±0.001 DAE 0.265±0.000 0.297±0.000 0.429±0.000 0.314±0.000 TabBinning 0.387±0.002 0.421±0.002 0.383±0.000 0.395±0.002 SAINT 0.251±0.002 0.286±0.008 0.429±0.000 0.304±0.003 SubTab 0.275±0.000 0.331±0.000 0.428±0.000 0.327±0.000 TabTransformer 0.257±0.000 0.301±0.000 0.426±0.000 0.310±0.000 TransTab 0.410±0.001 0.311±0.000 0.567±0.004 0.425±0.001

##### H Metric Definitions and Normalized Rank

Normalized rank (NR). NR denotes the mean normalized rank of a model across the finest evaluation unit u available for a given aggregate:

ranku(m) − 1 Nu − 1

1 |U(m)|

NR(m) =

,

u∈U(m)

where Nu is the number of models with a score on unit u, U(m) is the subset of units on which m is scored, and ties are broken by min. Missing units are excluded from m’s own average and do not penalize other models’ per-unit ranks. Lower is better. The unit u differs by aggregate: CTBENCH family NRs average ranks over the individual tasks in a family (e.g., Schema NR over ColType, ColClust, ColRel); row-prediction NRs average ranks over individual target columns (77 classification, 46 regression; classification ranks are averaged separately over AUROC and Macro F1); and Clean/Robust Linkage NRs average ranks over individual datasets (8 clean DM; 4 dirty DM +

- 4 WDC). Because dynamic range widens when fewer units are averaged, absolute NR magnitudes should be read within a column and not compared across suites.

Task metrics. F1 scores are macro-averaged over classes by default. This covers all multi-class tasks in the benchmark, including column type prediction, row-prediction classification, and the table-pair classification tasks (join classification, union classification, table subset). Record linkage is the sole exception: because it is a binary classification task, we follow the entity-matching convention established by DeepMatcher [53] and WDC Products [63] and report binary F1 on the match (positive) class, equivalent to sklearn.metrics.f1_score(..., average=’binary’,

pos_label=1). Group-level linkage scores (DM-C, DM-D, WDC, and the “All (16 pairs)” columns of Tables 26 and 27) are the unweighted mean of per-dataset binary F1. For regression tasks, we use nRMSE = √1 − R2, a monotone transform of the coefficient of determination (lower is better; values above 1 correspond to negative R2). To aggregate nRMSE across the 46 regression targets, we report the shifted geometric mean SGMε with ε = 0.01:

K

1/K

− ε,

SGMε(x1,...,xK) =

(xi + ε)

i=1

which reduces sensitivity to outlier targets while penalizing consistently poor performance. The shift prevents the product from collapsing when any xi is exactly zero. AUROC for row-prediction classification is the area under the ROC curve, computed per target column and then averaged across the 77 classification targets; binary AUROC is used for binary targets and weighted one-vs-rest AUROC is used for multi-class targets, following the standard sklearn.metrics.roc_auc_score convention. MAP denotes mean average precision computed over the full ranked list (not truncated to a fixed K). R@GT (Recall at Ground Truth) follows the Valentine convention [45]: all m × n candidate column pairs are ranked by cosine similarity, the top k pairs are retained with k = |ground truth|, and recall is the fraction of true correspondences among them. R@k (Recall at k) is the fraction of ground-truth targets present in the top-k retrieved items; we report k=100 as the Stage-1 retrieval diagnostic in DLTE. NMI is standard normalized mutual information with arithmetic averaging. MRR is mean reciprocal rank. Acc on Table QA is exact-match denotation accuracy on WikiTableQuestions, following the source convention [60]. UJ-H is defined in Sec. 3.4, and Cell F1, a complementary DLTE diagnostic, is defined in Appendix N.1.

##### I Table-Footprint Coverage Across Suites

For each table T, let nrow(T) and ncol(T) denote its row and column counts (equivalently, |R(T)| and |C(T)| in the notation of Sec. 3.1), and define its cell footprint as Fcell(T) = nrow(T)ncol(T). A loadable table input is a concrete table object returned by a benchmark evaluation loader for a table-valued role (feature table, left/right entity table, query table, or lake/corpus table), before model-specific serialization or truncation. We count each distinct table once within each role; if the same physical table appears under multiple roles, it is counted once per role. Thus row-prediction datasets contribute one feature table each, record-linkage datasets contribute their left and right entity tables, column- and table-CTBench datasets contribute their table-valued query/lake/corpus inputs as applicable, and TRL-DLTE is summarized by its 47,772 retrieval-lake candidates. Labeled table pairs, their labels, and train/valid/test split indices are metadata rather than additional table inputs: a table referenced by many pairs or splits is still counted once in its role. Figure 5 shows the joint (nrow,ncol) density per suite, Figure 6 marginalizes to Fcell on a logarithmic axis with 50 common bins, and Table 14 lists per-dataset summary statistics for the 87 dataset-source entries, grouped into seven benchmark categories: Schema, Joinability, Unionability, and Grounding from CTBench, Row Prediction and Record Linkage from RBench, and DLTE on its own. These statistics are descriptive only: benchmark scores are computed per task and are not weighted by table count or cell footprint.

Joint rows-cols footprint of TRL-Bench table inputs (dashed diagonals: constant Fcell = nrow ncol)

Column-CTBench (15 datasets; 284,377 table inputs)

Table-CTBench (5 datasets; 267,705 table inputs)

- 100
- 101
- 102
- 103
- 104

|[Figure 65]<br><br>Fcell {102 106}<br><br>|
|---|

|[Figure 66]<br><br>Fcell {102 106}<br><br>|
|---|

[Figure 67]

[Figure 68]

Colspertable()(logscale)nTco

100

100

%ofsuite

%ofsuite

10 1

10 1

10 2

10 2

10 3

10 3

Row-Rbench (66 datasets; 82 table inputs)

DLTE (1 dataset; 47,772 table inputs)

6 × 100

- 100
- 101
- 102
- 103
- 104

[Figure 69]

[Figure 70]

|[Figure 71]<br><br>Fcell {102 106}<br><br>|
|---|

|[Figure 72]<br><br>Fcell {102 106}<br><br>|
|---|

Colspertable()(logscale)nTco

100

- 2 × 100

- 3 × 100

- 4 × 100

%ofsuite

%ofsuite

10 1

10 2

100 101 102 103 104 105 106 107

100 101 102 103 104 105 106 107

Rows per table nrow(T) (log scale)

Rows per table nrow(T) (log scale)

###### Figure 5: Joint rows-columns footprint of TRL-Bench table inputs. Each panel plots a 2D

density of (nrow(T),ncol(T)) over the counted loadable table inputs of one suite, on log–log axes. Bin intensities are normalized within each suite to “% of suite”, so a smaller suite is not visually dominated by a larger one. Gray dashed diagonals mark constant-footprint contours Fcell ∈ {102, 103, 104, 105, 106}. The black × in each panel sits at the suite-wise median of nrow and the suite-wise median of ncol (computed independently along each axis).

Per-suite cell-footprint distributions of TRL-Bench table inputs

Column-CTBench (15 datasets; 284,377 tables)

Table-CTBench (5 datasets; 267,705 tables)

25000

p50=72

p50=40

p95=290,304

p95=28,410

30000

20000

#tables

15000

20000

10000

10000

5000

0

0

Row-Rbench (66 datasets; 82 tables)

DLTE (1 dataset; 47,772 tables)

p50=95,702

p50=2,532

12.5

2500

p95=5,051,753

p95=112,500

10.0

2000

#tables

7.5

1500

5.0

1000

2.5

500

0.0

0

100 101 102 103 104 105 106 107 108 109

100 101 102 103 104 105 106 107 108 109

Cells per table C = Rows Cols (log scale)

Cells per table C = Rows Cols (log scale)

###### Figure 6: Per-suite cell-footprint distributions of TRL-Bench table inputs. Each panel histograms

the footprint values Fcell(T) = nrow(T)ncol(T) for the counted loadable table inputs in one suite, using 50 common logarithmically spaced bins over 100–109 cells. The black dashed line marks the median footprint within that suite, and the gray dotted line marks the corresponding 95th percentile. Each panel’s y-axis reports the number of counted table inputs per bin and is scaled independently because the four suites differ by orders of magnitude in the number of counted inputs.

- Table 14: Per-dataset cell-footprint statistics across TRL-Bench suites. Each row is one of the 87 dataset-source entries in the benchmark, grouped by benchmark category (Schema, Joinability, Unionability, Grounding, Row Prediction, Record Linkage, DLTE). Cell footprint

Fcell(T) = nrow(T)ncol(T), i.e. row count times column count, is computed once per counted loadable table input under its dataset role. Labeled pairs and train/valid/test split indices are not expanded into additional table inputs. Mean, median, and Std. are computed over the counted inputs for that dataset. Std. uses the population convention (ddof=0) when at least two inputs are present. Row Prediction entries contain one counted feature table each, so Mean equals Median. Their Std. cells are shown with a dash because there is no within-dataset dispersion to summarize.

Dataset Category # table inputs Mean (cells) Median (cells) Std. (cells) Schema (col)

SOTAB Schema 72,629 1,887 260 9,922 WikiCT (rel.) Schema 53,567 42 22 105 sato Schema 78,733 29 10 120

Joinability (col/tbl)

OpenData (main) Joinability 16,823 1,025,435 258,792 3,585,105 OpenData CAN Joinability 4,960 1,245,433 359,628 3,858,684 OpenData UK/SG Joinability 3,090 272,228 54,610 2,082,714 OpenData USA Joinability 5,165 1,087,988 263,340 4,209,460 Spider Join Joinability 15,996 104,286 1,000 852,833 Wiki Containment Joinability 39,084 134 110 89

Unionability (col/tbl)

CKAN Subset Unionability 36,846 33,039 6,696 87,250 ECB Union Unionability 4,226 11,457 1,134 49,893 SANTOS Unionability 600 96,975 17,116 244,244 TUS Unionability 1,651 41,815 37,746 30,518 TUS-hard Unionability 2,769 44,109 41,850 29,579 UGEN v1 Unionability 1,050 77 70 44 UGEN v2 Unionability 1,050 307 140 437 Valentine Unionability 1,098 315,128 210,000 248,934 Wiki Union Unionability 40,752 133 110 89

Grounding (col/tbl)

NQ-Tables Grounding 169,885 47 18 155 WTQ Grounding 2,108 173 90 292

Row Prediction (row)

kc2 Row Prediction 1 10,440 10,440 – nomao Row Prediction 1 4,032,405 4,032,405 – kr-vs-kp Row Prediction 1 108,664 108,664 – sick Row Prediction 1 94,300 94,300 – connect-4 Row Prediction 1 2,769,837 2,769,837 – MiceProtein Row Prediction 1 81,000 81,000 – Internet-Ads Row Prediction 1 5,105,403 5,105,403 – auction-verification Row Prediction 1 10,215 10,215 – student-perf-por Row Prediction 1 18,172 18,172 – wave-energy Row Prediction 1 3,384,000 3,384,000 – cps88wages Row Prediction 1 140,775 140,775 – fps-benchmark Row Prediction 1 935,712 935,712 – PhishingWebsites Row Prediction 1 309,540 309,540 – analcatdata-authorship Row Prediction 1 58,029 58,029 – anneal Row Prediction 1 26,042 26,042 – Fiat-500 Row Prediction 1 9,228 9,228 – APSFailure Row Prediction 1 12,692,000 12,692,000 – bank-marketing Row Prediction 1 497,321 497,321 – Bank-Churn Row Prediction 1 90,000 90,000 – Bioresponse Row Prediction 1 6,658,025 6,658,025 – churn Row Prediction 1 85,000 85,000 – coil2000 Row Prediction 1 815,226 815,226 – credit-g Row Prediction 1 18,000 18,000 – credit-card-default Row Prediction 1 630,000 630,000 – airline-satisfaction Row Prediction 1 2,467,720 2,467,720 – Diabetes130US Row Prediction 1 3,075,274 3,075,274 – diamonds Row Prediction 1 431,520 431,520 – Fitness-Club Row Prediction 1 7,500 7,500 – GiveMeSomeCredit Row Prediction 1 1,350,000 1,350,000 – hazelnut-spread Row Prediction 1 69,600 69,600 – heloc Row Prediction 1 230,098 230,098 – hiva-agnostic Row Prediction 1 6,213,520 6,213,520 – houses Row Prediction 1 123,840 123,840 – HR-Analytics Row Prediction 1 191,580 191,580 – in-vehicle-coupon Row Prediction 1 291,732 291,732 – kddcup09-appetency Row Prediction 1 10,400,000 10,400,000 – Marketing-Campaign Row Prediction 1 51,520 51,520 – polish-bankruptcy Row Prediction 1 366,420 366,420 – qsar-biodeg Row Prediction 1 41,106 41,106 – SDSS17 Row Prediction 1 780,530 780,530 – seismic-bumps Row Prediction 1 33,592 33,592 – splice Row Prediction 1 188,210 188,210 – students-dropout Row Prediction 1 150,416 150,416 – superconductivity Row Prediction 1 1,701,040 1,701,040 – website-phishing Row Prediction 1 9,471 9,471 – wine-quality Row Prediction 1 64,970 64,970 – NATICUSdroid Row Prediction 1 629,244 629,244 – jm1 Row Prediction 1 217,700 217,700 – MIC Row Prediction 1 185,191 185,191 – cylinder-bands Row Prediction 1 17,820 17,820 –

Record Linkage (row)

DM-abt_buy Record Linkage 2 3,260 3,260 16 DM-amazon_google Record Linkage 2 6,884 6,884 2,794 DM-beer Record Linkage 2 14,690 14,690 2,690 DM-dblp_acm Record Linkage 2 9,820 9,820 644 DM-dblp_acm_dirty Record Linkage 2 9,820 9,820 644 DM-dblp_scholar Record Linkage 2 133,758 133,758 123,294 DM-dblp_scholar_dirty Record Linkage 2 133,758 133,758 123,294

DM-fodors_zagats Record Linkage 2 2,160 2,160 505 DM-itunes_amazon Record Linkage 2 251,320 251,320 196,064 DM-itunes_amazon_dirty Record Linkage 2 251,320 251,320 196,064 DM-walmart_amazon Record Linkage 2 61,570 61,570 48,800 DM-walmart_amazon_dirty Record Linkage 2 61,570 61,570 48,800 WDC-large Record Linkage 2 97,136 97,136 32 WDC-medium Record Linkage 2 79,688 79,688 245 WDC-small Record Linkage 2 56,948 56,948 3,440 WDC-xlarge Record Linkage 2 100,380 100,380 798

DLTE (dlte) DLTE-Lake DLTE 47,772 23,967 2,532 63,592

##### J Family-Level Performance Summary Figure

###### (a) Column / Table

###### (b) Row

Schema

Classification

| | |
|---|---|
| | |

Robust Linkage

Regression

Grounding

Join

Clean Linkage

Union

|BERT GTE TaBERT TAPAS TABBIE TURL Starmie TabSFM TabICL TransTab DAE<br><br>|
|---|

- Figure 7: Granularity-dependent transfer profiles. Radar plots summarize family-level performance for representative models on the two atomic suites. Panel (a) compares column/table encoders across the four TRL-CTBENCH capability families: Schema, Join, Union, and Grounding. Panel (b) compares row encoders across Classification, Regression, Clean Linkage, and Robust Linkage in TRL-RBENCH. BERT, GTE, and TABBIE appear in both panels because they expose both columnand row-level embeddings. The remaining models in each panel are granularity specialists and are evaluated only within their supported suite. Values are rank-normalized within each axis (farther from center is better). No single model dominates all axes, and strengths shift substantially with representation granularity and task family. This figure is a qualitative summary of Tables 2 and 3.

##### K CTBench Diagnostics and Ablations

###### K.1 Observational Lexical-Accessibility Proxies for CTBench

To support the Sec. 4.2 reading that generic text encoders remain strong on many column/table tasks through lexical accessibility, we report two observational proxies in Table 15 computed directly from the main CTBench results. The first is the gap between the strongest applicable non-neural baseline and the best neural encoder on the task: small gaps (after direction correction) indicate that surface lexical statistics, such as Jaccard, TF-IDF, value overlap, and Valentine matchers, already recover most of the task signal. The second is the direction-corrected generic-text advantage, the mean score of BERT/GTE minus the mean of tabular specialists (TABERT/TAPAS/STARMIE/TURL) on the same task. Generic text encoders lead on 10 of 13 CTBench tasks, with the strongest advantages on tasks whose input is dominated by short natural-language text (table retrieval +0.200, join search +0.116, column type prediction +0.088, column relation prediction +0.072, and column clustering +0.048), consistent with these tasks being accessible from headers and short cell strings. The three exceptions where the four-specialist mean beats the two-generic-text mean are exactly the tasks whose pretraining objective is tightly aligned: STARMIE’s column-level contrastive objective wins schema matching (−0.067) and union search (−0.010), and TURL’s table-language modeling wins Table QA (−0.016). Sec. 4.2 additionally counts Table Subset as a specialist-won task because the individual best model is TAPAS (0.567 F1, beating both BERT and GTE); on this mean-vs.-mean proxy Table Subset is borderline (+0.006 to generic text) since the four-specialist mean dilutes a single-specialist win. This is an observational hint, not a causal test. A direct header-masking ablation would provide stronger evidence and is left as future work. Within the current data, however, the ordering aligns with the main-text reading: CTBench task families differ systematically in whether text- or structure-pretrained encoders provide the dominant signal.

Table 15: Observational proxies for surface-text signal in TRL-CTBENCH tasks, ordered by descending generic-text advantage. Best baseline is the strongest task-local non-neural baseline from Table 2 (TF-IDF, Jaccard, Hungarian/Valentine, value-overlap, etc.). Best neural is the strongest CTBench encoder on that task. Gap is the metric-direction-corrected difference (best-neural − best-baseline, with positive meaning neural is better). Generic-text advantage is the mean of BERT, GTE minus the mean of tabular specialists TABERT, TAPAS, STARMIE, TURL on the same task (direction-corrected). Tasks near the top of the list are those most consistent with the claim that generic text encoders remain competitive on CTBench through surface-text signal. Tasks near the bottom are those where tabular pretraining provides a substantial objective-specific gain.

Task Dir. Best baseline Best neural (model) Gap Generic-text adv.

Table Retrieval ↑ 0.131 0.476 (GTE) +0.345 +0.200 Join Search ↑ 0.155 0.469 (GTE) +0.314 +0.116 Col Type ↑ 0.813 0.926 (BERT) +0.113 +0.088 Col Rel ↑ 0.015 0.826 (BERT) +0.811 +0.072 Col Clust ↑ 0.400 0.516 (BERT) +0.116 +0.048 Union Reg. ↓ 1.138 0.592 (BERT) +0.546 +0.039 Union Class ↑ 0.500 0.857 (BERT) +0.357 +0.034 Col Overlap ↓ 1.014 0.786 (BERT) +0.228 +0.032 Join Class ↑ 0.516 0.553 (BERT) +0.037 +0.023 Table Subset ↑ 0.458 0.567 (TAPAS) +0.109 +0.006 Union Search ↑ 0.574 0.662 (Starmie) +0.088 −0.010 Table QA ↑ 0.204 0.277 (TURL) +0.073 −0.016 Schema Match ↑ 0.473 0.764 (Starmie) +0.291 −0.067

###### K.2 Probe Head Complexity for Column/Table-Level Tasks

Main column- and table-level results follow the unified supervised-probe protocol of Sec. 3.1 (averaging linear and MLP heads). This appendix reports the linear and MLP components separately to show per-task head sensitivity. Table 16 compares MLP, linear, and dummy probes across five column/table-level tasks.

Frozen embeddings carry most of the signal. The ∆L-D column shows that the linear probe already far outperforms the dummy baseline on every task: averages of +0.65 (ColType), +0.72

- Table 16: Ablation: Probe head complexity for column/table-level tasks. MLP = two-layer MLP head. Linear = logistic regression / ridge. Dummy = majority-class or mean prediction. Best

embedding per model, 5-seed average. ∆M-L = MLP − Linear (positive = MLP is better). ∆L-D = Linear − Dummy (capacity of the frozen embedding itself).

Join Class.t† F1 ↑

Union Class.t† F1 ↑

Col Typec

###### Col Relc

Tbl Subsett

F1 ↑ Model MLP Lin Dum ∆M-L ∆L-D MLP Lin Dum ∆M-L ∆L-D MLP Lin Dum ∆M-L ∆L-D MLP Lin Dum ∆M-L ∆L-D MLP Lin Dum ∆M-L ∆L-D BERT 0.928 0.924 0.178 +0.00 +0.75 0.834 0.819 0.000 +0.01 +0.82 0.529 0.578 0.414 −0.05 +0.16 0.962 0.752 0.331 +0.21 +0.42 0.665 0.424 0.369 +0.24 +0.06 GTE 0.922 0.923 0.178 −0.00 +0.74 0.826 0.797 0.000 +0.03 +0.80 0.519 0.551 0.414 −0.03 +0.14 0.950 0.736 0.331 +0.21 +0.41 0.662 0.425 0.369 +0.24 +0.06 TaBERT 0.874 0.874 0.178 +0.00 +0.70 0.771 0.749 0.000 +0.02 +0.75 0.468 0.527 0.414 −0.06 +0.11 0.818 0.702 0.331 +0.12 +0.37 0.677 0.403 0.369 +0.27 +0.03 TAPAS 0.869 0.867 0.178 +0.00 +0.69 0.782 0.755 0.000 +0.03 +0.76 0.555 0.532 0.414 +0.02 +0.12 0.931 0.741 0.331 +0.19 +0.41 0.695 0.439 0.369 +0.26 +0.07 TAPEX — — — — — — — — — — 0.530 0.497 0.414 +0.03 +0.08 0.951 0.757 0.331 +0.19 +0.43 0.718 0.396 0.369 +0.32 +0.03 TABBIE 0.881 0.904 0.178 −0.02 +0.73 0.777 0.793 0.000 −0.02 +0.79 0.523 0.560 0.414 −0.04 +0.15 0.931 0.736 0.331 +0.20 +0.40 0.697 0.396 0.369 +0.30 +0.03 TURL 0.836 0.792 0.178 +0.04 +0.61 0.777 0.740 0.000 +0.04 +0.74 0.510 0.554 0.414 −0.04 +0.14 0.942 0.686 0.331 +0.26 +0.36 0.635 0.379 0.369 +0.26 +0.01 TUTA — — — — — — — — — — 0.465 0.470 0.414 −0.01 +0.06 0.906 0.713 0.331 +0.19 +0.38 0.472 0.421 0.369 +0.05 +0.05 Starmie 0.767 0.811 — −0.04 — 0.695 0.701 — −0.01 — 0.512 0.509 — +0.00 — 0.949 0.757 — +0.19 — 0.672 0.407 — +0.26 TabSketchFM 0.583 0.550 0.178 +0.03 +0.37 0.390 0.356 0.000 +0.03 +0.36 0.482 0.550 0.414 −0.07 +0.14 0.785 0.689 0.331 +0.10 +0.36 0.663 0.443 0.369 +0.22 +0.07 Avg. 0.832 0.830 0.178 +0.00 +0.65 0.731 0.714 0.000 +0.02 +0.72 0.509 0.533 0.414 −0.02 +0.12 0.913 0.727 0.331 +0.19 +0.39 0.656 0.413 0.369 +0.24 +0.05

F1 ↑

F1 ↑

(ColRel), +0.12 (JoinCls), +0.39 (UnionCls), and +0.05 (TblSubset). This confirms that the frozen representations encode task-relevant structure without any task-specific training.

MLP adds little over linear on column tasks. For ColType and ColRel, the average ∆M-L is +0.00 and +0.02 respectively, indicating that a linear probe is sufficient. For UnionCls and TblSubset, the MLP gains are more consistent (+0.19 and +0.24 on average), suggesting that nonlinear separation helps when table-pair geometry is more complex.

JoinCls favors linear probes. The average ∆M-L for JoinCls is −0.02, and several models show negative gaps (e.g., BERT −0.05, TaBERT −0.06). This indicates that the MLP head overfits on the relatively small join classification training sets, and a linear probe is the more reliable choice for this task.

###### K.3 Aggregation Ablation for Table-Level Embeddings

For models that expose multiple candidate table embeddings, we compare three aggregation strategies: CLS (the [CLS] token of the linearized table), COL-MEAN (the mean of per-column embeddings), and TOK-MEAN (the mean of all non-padding token hidden states). Tables 17 and 18 report the full MLP- and linear-probe breakdowns, respectively.

Task-level averages. With an MLP probe, TOK-MEAN gives the best average on UnionCls, UnionReg (lower is better), and TblRet, while COL-MEAN is narrowly best on JoinCls and TblSubset. With a linear probe, COL-MEAN is best on JoinCls and UnionReg, while TOK-MEAN is best on UnionCls and TblSubset. However, several gaps are tiny (e.g., 0.494/0.500/0.497 on MLP JoinCls for CLS/COL-MEAN/TOK-MEAN and 0.724/0.725/0.734 on linear UnionCls), so these averages should be read as rough tendencies rather than definitive rankings.

Per-model variation. The per-model tables show some encoder-specific preferences: TABBIE often peaks with CLS under MLP, whereas TAPAS and several text-pretrained models more often favor COL-MEAN or TOK-MEAN. Nevertheless, the model-dependent variation is modest relative to the task-dependent variation, so we do not treat aggregation choice as a major axis of analysis. The main comparison in Table 2 reports each encoder’s strongest supported aggregation averaged over MLP and linear probes.

###### K.4 Join Search: Direct Cosine vs. Learned Projection

Interpretation. Direct cosine is an informative no-training control, but in our frozen-transfer reuse of join search the relation of interest is directional containment rather than pure semantic similarity. The learned projection is therefore the canonical main-table setting: it remains a minimal probe over frozen embeddings while matching the transfer objective more closely.

###### K.5 Union Search: TUS vs. TUS-hard

- Table 20 compares model performance on TUS (original) and TUS-hard (low-overlap variant). TUShard filters out positive pairs whose directed column containment is at least 0.70, removing the

- Table 17: Effect of table-level embedding aggregation (MLP probe). For each model we evaluate every supported aggregation strategy: CLS ([CLS] token from the linearized table), COL-MEAN (mean of per-column embeddings), and TOK-MEAN (mean of all non-padding token hidden states).

Metrics match Table 2: F1 = macro F1, nRMSE = √1 − R2. Table 2 reports avg(MLP, linear) for the best supported aggregation. Tables 17–18 break down the per-probe results. Values are mean ± std over 5 random seeds. Bold / underlined values highlight best/second-best aggregation for each model and metric (shown only when ≥2 variants exist). †Table-disjoint split. Dashes indicate the model does not produce that embedding variant.

Family Model Agg. JoinCls† UnionCls† UnionReg† TblSubset TblRet

F1 ↑ F1 ↑ nRMSE ↓ F1 ↑ MRR ↑

CLS 0.503±0.033 0.956±0.003 0.509±0.009 0.615±0.010 0.321±0.013 COL-MEAN 0.493±0.040 0.961±0.002 0.454±0.009 0.665±0.009 0.357±0.009 TOK-MEAN 0.529±0.019 0.962±0.004 0.449±0.018 0.653±0.008 0.367±0.008

BERT

Generic Text

CLS 0.491±0.029 0.946±0.005 0.640±0.011 0.567±0.014 0.476±0.003 COL-MEAN 0.519±0.037 0.944±0.002 0.475±0.019 0.662±0.004 0.450±0.013 TOK-MEAN 0.453±0.035 0.950±0.004 0.578±0.018 0.543±0.044 0.473±0.008

GTE

TaBERT COL-MEAN 0.468±0.068 0.818±0.007 0.536±0.017 0.677±0.007 0.372±0.013 TAPAS

CLS 0.460±0.035 0.856±0.006 0.576±0.018 0.689±0.007 0.265±0.009 COL-MEAN 0.555±0.022 0.929±0.007 0.494±0.008 0.695±0.009 0.285±0.013 TOK-MEAN 0.503±0.031 0.931±0.006 0.488±0.010 0.693±0.011 0.295±0.006

Table-Text

0.530±0.046 0.942±0.002 0.468±0.012 0.718±0.009 TOK-MEAN 0.526±0.043 0.951±0.004 0.471±0.010 0.704±0.007 —

TAPEX CLS

0.523±0.043 0.931±0.003 0.542±0.002 0.697±0.007 0.170±0.004

TABBIE CLS

COL-MEAN 0.474±0.034 0.919±0.002 0.665±0.011 0.681±0.009 0.102±0.002 TURL COL-MEAN 0.510±0.028 0.942±0.002 0.499±0.018 0.635±0.007 0.199±0.010 TUTA CLS 0.465±0.019 0.906±0.006 0.511±0.013 0.472±0.016 0.260±0.013

Table-Struct.

Starmie COL-MEAN 0.512±0.017 0.949±0.004 0.536±0.006 0.672±0.010 0.018±0.002 TabSketchFM

CLS 0.482±0.032 0.779±0.004 0.549±0.024 0.659±0.015 0.218±0.011 COL-MEAN 0.471±0.026 0.785±0.002 0.512±0.021 0.647±0.004 0.197±0.014 TOK-MEAN 0.476±0.021 0.783±0.004 0.513±0.013 0.663±0.009 0.193±0.035

Col.-Centric

Avg. across models CLS 0.494 0.902 0.542 0.631 0.285 COL-MEAN 0.500 0.906 0.521 0.667 0.248 TOK-MEAN 0.497 0.915 0.500 0.651 0.332

highest-overlap 36% of positives so that our frozen-transfer evaluation can better separate lexical overlap from broader union signal.

Interpretation. On original TUS, the value-overlap baseline achieves MAP 1.000 and BERT, GTE, and TURL all score above 0.95. On TUS-hard, the baseline drops to 0.008 and the ranking shifts substantially (Spearman ρ = −0.67). The models most robust under this low-overlap variant are STARMIE and TABERT, whose pretraining objectives emphasize cross-table structure. We therefore report TUS-hard alongside TUS in TRL-BENCH not as a replacement for the original resource, but as a complementary variant for the frozen-transfer setting, where it is useful to distinguish overlap-driven retrieval from transfer that remains helpful when overlap is limited.

###### K.6 Table Retrieval: Model-Only vs. Hybrid Mode

- Table 21 compares model-only and hybrid retrieval modes. In model-only mode, the projection head operates solely on the model’s own table embedding. In hybrid mode, the model’s table embedding is concatenated with the query encoder’s (MPNet [67] or sentence-T5 [56]) table embedding before projection, bridging the modality gap between the model’s table space and the query space.

Interpretation. Hybrid mode uniformly improves MRR for every model, with gains ranging from +0.057 (GTE) to +0.509 (STARMIE). However, all hybrid MRRs converge to a narrow band of 0.509–0.555, regardless of the model’s own retrieval quality (0.018–0.476 in model-only mode). This indicates that the added query-encoder table embedding contributes most of the shared signal in the augmented setting. For TRL-BENCH’s main comparison we therefore use model-only mode, because it isolates the transfer quality of the model’s own table representation rather than performance in a pipeline with added query-side table evidence.

- Table 18: Effect of table-level embedding aggregation (LINEAR probe). For each model we evaluate every supported aggregation strategy: CLS ([CLS] token from the linearized table), COL-MEAN (mean of per-column embeddings), and TOK-MEAN (mean of all non-padding token hidden states). Metrics match Table 2: F1 = macro F1, nRMSE = √1 − R2. Table 2 reports the strongest supported aggregation in the corresponding main comparison. Values are mean ± std over 4–5 random seeds. Bold / underlined values highlight best/second-best aggregation for each model and metric (shown only when ≥2 variants exist). †Table-disjoint split. Dashes indicate the model does not produce that embedding variant.

Family Model Agg. JoinCls† UnionCls† UnionReg† TblSubset

F1 ↑ F1 ↑ nRMSE ↓ F1 ↑

Generic Text

BERT

CLS 0.526±0.000 0.745±0.000 0.770±0.000 0.436±0.000 COL-MEAN 0.500±0.000 0.745±0.000 0.736±0.000 0.424±0.000 TOK-MEAN 0.578±0.000 0.752±0.000 0.735±0.000 0.408±0.000

GTE

CLS 0.503±0.000 0.726±0.000 0.830±0.000 0.421±0.000 COL-MEAN 0.551±0.000 0.735±0.000 0.726±0.000 0.425±0.000 TOK-MEAN 0.478±0.000 0.736±0.000 0.754±0.000 0.416±0.000

Table-Text

TaBERT COL-MEAN 0.527±0.000 0.702±0.000 0.694±0.000 0.403±0.001 TAPAS

CLS 0.531±0.000 0.707±0.000 0.760±0.000 0.432±0.000 COL-MEAN 0.532±0.000 0.746±0.000 0.731±0.000 0.439±0.001 TOK-MEAN 0.516±0.000 0.741±0.000 0.726±0.000 0.436±0.001

TAPEX CLS 0.497±0.000 0.747±0.000 0.750±0.000 0.396±0.000 TOK-MEAN 0.550±0.000 0.757±0.000 0.772±0.000 0.411±0.000

Table-Struct.

TABBIE CLS 0.560±0.000 0.736±0.000 0.784±0.000 0.396±0.000

COL-MEAN 0.581±0.000 0.737±0.000 0.757±0.000 0.387±0.000 TURL COL-MEAN 0.554±0.000 0.686±0.000 0.814±0.000 0.379±0.000 TUTA CLS 0.470±0.000 0.713±0.000 0.792±0.000 0.421±0.001

Col.-Centric

Starmie COL-MEAN 0.509±0.031 0.757±0.000 0.789±0.000 0.407±0.000 TabSketchFM

CLS 0.550±0.007 0.695±0.000 0.864±0.001 0.432±0.002 COL-MEAN 0.513±0.013 0.689±0.000 0.825±0.001 0.445±0.003 TOK-MEAN 0.533±0.015 0.684±0.000 0.830±0.001 0.443±0.001

Avg. across models CLS 0.520 0.724 0.793 0.419 COL-MEAN 0.533 0.725 0.759 0.414 TOK-MEAN 0.531 0.734 0.763 0.423

- Table 19: Join search MAP: direct cosine similarity vs. learned linear projection, averaged across

- 5 embedding rounds. The learned projection trains a shared Linear(d,d) head with multi-positive InfoNCE loss on a fixed 20%/80% query-role-disjoint split and evaluates on the held-out 80%. Results are column-level MAP, macro-averaged over queries. Bold orange / Underlined blue /

Light purple highlights indicate best/second-best/third-best per column (non-baseline).

All CAN USA UK+SG Family Model Cos. Proj. Cos. Proj. Cos. Proj. Cos. Proj. Generic Text

BERT 0.387 0.431 0.300 0.328 0.447 0.508 0.431 0.470 GTE 0.411 0.461 0.331 0.370 0.460 0.535 0.478 0.508

Tabular-Pretrained Table-Text

TaBERT 0.326 0.458 0.295 0.346 0.377 0.525 0.262 0.296 TAPAS 0.246 0.332 0.230 0.295 0.285 0.383 0.208 0.272

TURL 0.271 0.292 0.248 0.267 0.299 0.341 0.269 0.295 TABBIE 0.189 0.218 0.174 0.176 0.207 0.231 0.206 0.206

Table-Struct.

Starmie 0.248 0.298 0.230 0.295 0.290 0.379 0.206 0.293 TabSketchFM 0.231 0.277 0.195 0.227 0.276 0.339 0.188 0.218

Col.-Centric

- Table 20: Union search MAP on TUS vs. TUS-hard (5-seed average). TUS-hard filters out positive pairs with high directed containment (≥ 0.70), creating a low-overlap variant that better separates lexical overlap from broader union signal in our frozen-transfer setting. Ranks are among neural models only. Bold orange / Underlined blue / Light purple highlights indicate best/secondbest/third-best on TUS-hard. Spearman ρ = −0.67 between the two rankings.

Type Model

MAP TUS ↑

Rank TUS

MAP TUS-hard ↑

Rank Hard

Drop (%) Baseline

Random 0.209 — 0.079 — 62.3 Val. Ovlp 1.000 — 0.008 — 99.2

Generic Text

BERT 0.959 1 0.307 6 68.0 GTE 0.954 2 0.293 8 69.2

Table-Text

TaBERT 0.926 5 0.436 2 52.9 TAPAS 0.891 6 0.376 3 57.8

Table-Struct.

TABBIE 0.700 8 0.317 5 54.8 TURL 0.953 3 0.304 7 68.1

Col.-Centric

Starmie 0.844 7 0.523 1 38.1 TabSketchFM 0.941 4 0.376 4 60.1

- Table 21: Table retrieval MRR: model-only vs. hybrid mode. In model-only mode, the projection head operates on the model’s own table embedding. In hybrid mode, the model’s table embedding is concatenated with the query encoder’s (mpnet or sentence-t5) table embedding before projection. For each model, we report the best aggregation × sentence-encoder combination. Values are mean ± std over 5 seeds.

Model-Only MRR ↑

Hybrid MRR ↑

Family Model

∆

###### BERT 0.367±0.008 0.553±0.007 +0.186 GTE 0.476±0.003 0.533±0.008 +0.057

Generic Text

TaBERT 0.372±0.013 0.555±0.006 +0.183 TAPAS 0.295±0.006 0.526±0.008 +0.231 TAPEX 0.332±0.005 0.536±0.028 +0.204

Table-Text

###### TABBIE 0.170±0.004 0.516±0.007 +0.347 TURL 0.199±0.010 0.521±0.031 +0.322 TUTA 0.260±0.013 0.509±0.012 +0.249

Table-Struct.

Starmie 0.018±0.002 0.527±0.012 +0.509 TabSketchFM 0.218±0.011 0.522±0.008 +0.304

Col.-Centric

###### K.7 Query Encoder Sensitivity

Table 22: Ablation: Query encoder for grounding tasks. Table retrieval uses MRR (↑, model_only). Semantic parsing uses accuracy (↑). Each table model is paired with two query encoders. ∆ = MPNet − ST5. 5-seed average. Table retrieval uses best embedding per model.

Tbl Ret.t

Tbl QAc

Acc↑ Model MPNet ST5 ∆ MPNet ST5 ∆ BERT 0.368 0.352 +0.02 0.238 0.271 −0.03

MRR↑

± 0.007 ± 0.007 ± 0.005 ± 0.004

GTE 0.478 0.440 +0.04 0.233 0.256 −0.02

± 0.004 ± 0.005 ± 0.004 ± 0.005

TaBERT 0.372 0.324 +0.05 0.252 0.281 −0.03

± 0.013 ± 0.007 ± 0.007 ± 0.005

TAPAS 0.296 0.276 +0.02 0.240 0.269 −0.03

± 0.006 ± 0.007 ± 0.005 ± 0.008

TAPEX 0.376 0.341 +0.03 — — —

± 0.097 ± 0.073

TABBIE 0.170 0.145 +0.02 0.261 0.290 −0.03

± 0.004 ± 0.002 ± 0.004 ± 0.008

TURL 0.197 0.199 −0.00 0.274 0.281 −0.01

± 0.014 ± 0.010 ± 0.006 ± 0.004

TUTA 0.260 0.235 +0.03 — — —

± 0.013 ± 0.013

Starmie 0.018 0.012 +0.01 0.248 0.283 −0.03

± 0.002 ± 0.002 ± 0.008 ± 0.008

TabSketchFM 0.221 0.191 +0.03 0.216 0.253 −0.04

± 0.013 ± 0.006 ± 0.006 ± 0.006

Avg. 0.276 0.252 +0.02 0.245 0.273 −0.03

- Table 22 compares MPNet [67] and sentence-T5 [56] (ST5) as query encoders across table retrieval (MRR) and table QA (accuracy). MPNet consistently outperforms ST5 on table retrieval for almost every model (average ∆=+0.02), with the largest gain for TaBERT (+0.05). The pattern reverses for table QA: ST5 is better for all models with available results (average ∆=−0.03), suggesting ST5’s longer context pretraining better supports semantic parsing. In both tasks the absolute differences are small (≤0.05), indicating that query encoder choice has limited sensitivity on these tasks. MPNet is used as the default query encoder in the main evaluation.

###### K.8 Pair-Level Random vs. Table-Disjoint Split Ablation

Four table-pair tasks (join classification, column overlap, union classification, and union regression) support both the original pair-level random splits and the table-disjoint splits used in our frozentransfer evaluation, where training and test tables are separated so the task measures cross-table generalization. Table 23 compares the two protocols.

Interpretation. Under the table-disjoint protocol, performance is uniformly lower. The effect is largest on join classification, where the mean per-model F1 drop is 0.212 (aggregate means: 0.736 → 0.523), and union regression shows the next-largest change (average nRMSE increases by 0.110). This is consistent with our transfer-oriented setting being harder: models must generalize to unseen tables rather than to new pairs drawn from already observed tables. Relative rankings are largely preserved, suggesting that table-disjoint evaluation mainly changes the difficulty level and the degree of cross-table separation required by our protocol. We therefore use table-disjoint splits as the default when repurposing these tasks for frozen cross-table transfer.

- Table 23: Pair-level random vs. table-disjoint split comparison on the four tasks that support both protocols. Each cell reports the 5-seed average (avg. of MLP and linear probes, best aggregation for

table-level tasks). ∆ = table-disjoint − pair-random. Negative ∆ for F1 and positive ∆ for nRMSE both indicate that table-disjoint evaluation is harder. Values are mean ± std over 5 seeds. Dashes indicate the model does not produce that embedding variant. Avg. is computed over all models with available data per task.

JoinCls

ColOverlap

UnionCls

UnionReg

nRMSE↓

nRMSE↓

F1 ↑

F1 ↑

Type Model P-R T-D ∆ P-R T-D ∆ P-R T-D ∆ P-R T-D ∆ Baseline Random 0.704 0.516 −0.189 0.972 1.012 +0.040 0.666 0.500 −0.166 0.868 1.138 +0.270

###### BERT 0.768 0.553 −0.215 0.758 0.786 +0.028 0.877 0.857 −0.020 0.485 0.592 +0.107

± 0.013 ± 0.010 ± 0.001 ± 0.001 ± 0.001 ± 0.002 ± 0.004 ± 0.009

Generic Text

###### GTE 0.760 0.535 −0.225 0.775 0.817 +0.042 0.868 0.843 −0.025 0.504 0.600 +0.096

± 0.027 ± 0.019 ± 0.000 ± 0.002 ± 0.001 ± 0.002 ± 0.005 ± 0.009

###### TaBERT 0.759 0.498 −0.261 0.812 0.855 +0.044 0.819 0.760 −0.059 0.520 0.615 +0.095

± 0.021 ± 0.034 ± 0.000 ± 0.002 ± 0.001 ± 0.004 ± 0.003 ± 0.009

###### TAPAS 0.740 0.544 −0.196 0.782 0.823 +0.041 0.867 0.837 −0.030 0.502 0.607 +0.105

Table-Text

± 0.010 ± 0.011 ± 0.000 ± 0.002 ± 0.001 ± 0.004 ± 0.002 ± 0.005

###### TAPEX 0.784 0.538 −0.246 — — — 0.877 0.854 −0.023 0.494 0.609 +0.115

± 0.010 ± 0.021 ± 0.000 ± 0.002 ± 0.002 ± 0.006

###### TABBIE 0.763 0.542 −0.221 0.832 0.862 +0.030 0.861 0.833 −0.028 0.540 0.663 +0.123

± 0.009 ± 0.022 ± 0.001 ± 0.002 ± 0.001 ± 0.002 ± 0.003 ± 0.001

###### TURL 0.626 0.532 −0.094 0.778 0.809 +0.031 0.842 0.814 −0.028 0.549 0.657 +0.108

Table-Struct.

± 0.014 ± 0.014 ± 0.001 ± 0.001 ± 0.001 ± 0.001 ± 0.003 ± 0.009

###### TUTA 0.692 0.468 −0.224 — — — 0.850 0.810 −0.041 0.531 0.652 +0.121

± 0.028 ± 0.010 ± 0.001 ± 0.003 ± 0.005 ± 0.007

###### Starmie 0.715 0.510 −0.205 0.808 0.847 +0.039 0.876 0.853 −0.023 0.560 0.662 +0.102

± 0.005 ± 0.019 ± 0.000 ± 0.001 ± 0.000 ± 0.002 ± 0.003 ± 0.003

Col.-Centric

###### TabSketchFM 0.750 0.516 −0.234 0.887 0.946 +0.059 0.794 0.737 −0.057 0.545 0.668 +0.123

± 0.015 ± 0.015 ± 0.001 ± 0.001 ± 0.001 ± 0.002 ± 0.002 ± 0.010

###### Avg. 0.736 0.523 −0.212 0.804 0.843 +0.039 0.853 0.820 −0.033 0.523 0.633 +0.110

- Table 24: Row-prediction probe-head sweep. Here d counts linear layers. The default one-hiddenlayer MLP (h=256,d=2) is near-optimal for classification. Deeper/wider heads do not provide a consistent regression benefit.

Head config Avg Macro-F1 Avg SGM↓

- h=256, d=2 (default) 0.6173 0.757 h=128, d=3 0.6173 0.750
- h=256, d=3 0.6144 0.772

- h=512, d=4 0.6064 0.755
- h=512, d=5 0.5983 0.750

##### L RBench Diagnostics and Ablations

###### L.1 Row-Prediction Probe Diagnostics

Representative linear-vs.-MLP cases. For classification, several models are already strongest with a linear probe: the text-transfer encoders BERT (0.6322 MLP → 0.6360 linear) and GTE (0.6004 → 0.6200), as well as the target-table contrastive learner TRANSTAB (0.6016 → 0.6151). By contrast, the feature-corruption target-table SSL models benefit primarily on regression, e.g., DAE improves from 0.5984 to 0.5357 SGM, SCARF from 0.5917 to 0.5462, and TABBINNING from 0.5929 to 0.5448 when moving from linear to MLP.

Dimensionality check. Dimensionality alone does not explain the row ranking. Even after upgrading SSL row encoders to 768 dimensions, TABICL (512-d) remains strongest on both classification and regression in this dim-controlled comparison (0.6744 Macro-F1 and 0.4873 SGM, computed on the dim-ablation subset; the main-table values are 0.671 and 0.505 under the standard probe protocol). Embedding dimensions range from 192 (TABPFN) to 768 (generic text and table-aware models), with most SSL row models at 512. Because the MLP probe uses a fixed hidden size of 256, the first-layer parameter count scales linearly with input dimension (e.g., 768 × 256 vs. 192 × 256). The linear-probe comparison above serves as a dimension-proportional control: linear probes have exactly d × C parameters (where d is the embedding dimension and C the number of classes), so they do not introduce a fixed-width bottleneck. The main row ranking is consistent across linear and MLP probes, indicating that the ranking reflects embedding quality rather than probe-capacity artifacts. For pairwise tasks (record linkage), concatenation doubles the input dimension, producing 1536-d inputs for 768-d models and 1024-d for 512-d models with the same hidden-size-256 MLP. The same linear-probe consistency check applies.

Per-target comparison. Figure 8 plots per-target scores for TABICL against the strongest comparator on each regime: BERT for classification (AUROC) and DAE for regression (nRMSE). TABICL wins on 57/77 classification targets and 38/46 regression targets, confirming that its advantage is broad rather than driven by a few outlier tasks.

Adaptation regime explains a substantial fraction of row-level variance. Table 25 quantifies the Sec. 4.3 claim that training regime (externally pretrained transfer vs. target-table self-supervision vs. prior-based meta-pretraining) materially shapes row-level rankings, using the per-sub-task normalizedrank aggregates from Table 3. The decomposition uses the classical 1-way ANOVA sums-of-squares with regime identity as the factor, so between-regime variance and within-regime variance sum to the total sub-task variance (an η2-style fraction). The regime effect is largest on the two linkage sub-tasks: between-regime variance accounts for 64% of cross-model NR variance on clean linkage and 48% on robust linkage. The Kruskal–Wallis test rejects the null of identical regime distributions at α = 0.05 on clean linkage (p = 0.014) and is borderline on robust linkage (p = 0.064). Regression shows a smaller regime effect (41% between/total) with a marginally significant Kruskal–Wallis statistic (p = 0.090), while classification has the weakest effect (17% between/total, p = 0.373), consistent with the observation that frozen text encoders and target-table learners both reach competitive classification AUROC for different reasons. The direction of the regime effect also depends on the sub-task: Transfer and Prior-based encoders dominate linkage, whereas Target-Table learners are more competitive on prediction regression. This supports keeping the three regimes separately rather than flattening them into one leaderboard. (We note that the Prior-based regime contains only

###### (a) Classification

(b) Regression

1.75

1.0

TabICL wins 57/77 targets

1.50

0.9

1.25

TabICLAUROC

TabICLnRMSE

0.8

1.00

0.7

0.75

0.50

0.6

0.25

0.5

TabICL wins 38/46 targets

0.00

0.5 0.6 0.7 0.8 0.9 1.0

0.0 0.5 1.0 1.5

BERT AUROC

DAE nRMSE

- Figure 8: Per-target pairwise comparison of TABICL against the strongest comparator. (a) Classification AUROC: TABICL vs. BERT across 77 targets. (b) Regression nRMSE: TABICL vs. DAE across 46 targets. Points above the diagonal in (a) and below it in (b) indicate TABICL wins. Each color represents a different source dataset.

two models, so the Kruskal–Wallis χ2 approximation is borderline at the α = 0.05 level for robust linkage. The between/total η2 summary does not share this small-sample limitation.)

- Table 25: Regime-wise normalized-rank summary for TRL-RBENCH. For each of the four sub-tasks we report mean ± std normalized rank within each adaptation regime (lower is better), together with a Kruskal–Wallis H-test for the null that all regimes have equal NR distributions and the fraction of total sub-task variance explained by between-regime variance. Values are computed from the per-model NR aggregates in Table 3. The “between / total” column quantifies how much of the cross-model variation is captured by regime identity alone.

Kruskal–Wallis H (p)

Between/Total

Sub-task Transfer Target-Table Prior-Based

Classification 0.504±0.073 0.481±0.134 0.328±0.164 2.0 (0.373) 0.17 Regression 0.637±0.055 0.473±0.129 0.319±0.180 4.8 (0.090) 0.41 Clean Linkage 0.188±0.060 0.654±0.193 0.509±0.086 8.6 (0.014) 0.64 Robust Linkage 0.192±0.130 0.647±0.249 0.528±0.135 5.5 (0.064) 0.48

###### L.2 Embedding Dimension for Record Linkage

- Table 26 reports binary F1 (match class) for eight target-table learners across five embedding dimensions (64, 128, 256, 512, 768) on the record linkage task. Performance increases monotonically

with dimension for most models: the average All-pairs F1 rises from 0.058 at d=64 to 0.139 at d=768. The WDC group is consistently the hardest across all dimensions, reflecting the greater heterogeneity of product-matching pairs. TransTab improves substantially with larger embeddings (0.139 → 0.254 overall), while SubTab remains near zero regardless of dimension, suggesting its representations lack pairwise match signal at any scale. These results justify using the native embedding size (512-d for target-table SSL models) as the default throughout the main evaluation.

- Table 26: Ablation: Embedding dimension for record linkage. Binary F1 (match class, see Appendix H), 5-seed average, linear probe head. Results are broken down by dataset group: DM-C = 8 clean DeepMatcher pairs, DM-D = 4 dirty DeepMatcher pairs, WDC = 4 WDC-Products pairs, All

= unweighted mean over all 16 pairs. Bold = best dimension per model, underline = second best.

DM-C F1↑ DM-D F1↑ WDC F1↑ All (16 pairs) F1↑ Model 64 128 256 512 768 64 128 256 512 768 64 128 256 512 768 64 128 256 512 768 VIME 0.095 0.101 0.112 0.150 0.191 0.043 0.110 0.130 0.162 0.225 0.007 0.009 0.018 0.047 0.075 0.060 0.081 0.093 0.127 0.171

± 0.001 ± 0.026 ± 0.000 ± 0.002 ± 0.000 ± 0.000 ± 0.008 ± 0.000 ± 0.001 ± 0.000 ± 0.000 ± 0.001 ± 0.000 ± 0.002 ± 0.000 ± 0.000 ± 0.011 ± 0.000 ± 0.002 ± 0.000

###### SCARF 0.097 0.150 0.158 0.159 0.175 0.041 0.073 0.130 0.120 0.143 0.009 0.008 0.021 0.040 0.056 0.061 0.095 0.117 0.120 0.138

± 0.001 ± 0.006 ± 0.000 ± 0.006 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.010 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.001 ± 0.001 ± 0.000 ± 0.003 ± 0.000 ± 0.000 ± 0.000

###### DAE 0.078 0.126 0.119 0.160 0.183 0.072 0.092 0.127 0.157 0.200 0.007 0.008 0.019 0.049 0.075 0.059 0.088 0.096 0.131 0.160

± 0.000 ± 0.018 ± 0.000 ± 0.002 ± 0.000 ± 0.000 ± 0.009 ± 0.000 ± 0.004 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.001 ± 0.000 ± 0.000 ± 0.011 ± 0.000 ± 0.000 ± 0.000

TabBinning 0.087 0.124 0.117 0.161 0.149 0.099 0.087 0.125 0.152 0.161 0.008 0.012 0.025 0.046 0.064 0.070 0.087 0.096 0.130 0.131

± 0.000 ± 0.000 ± 0.000 ± 0.001 ± 0.000 ± 0.000 ± 0.001 ± 0.000 ± 0.010 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.003 ± 0.000

###### SAINT 0.058 0.078 0.144 0.187 0.204 0.058 0.061 0.093 0.194 0.168 0.000 0.000 0.011 0.034 0.064 0.044 0.054 0.098 0.150 0.160

± 0.000 ± 0.000 ± 0.000 ± 0.003 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.013 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.003 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.001 ± 0.000

SubTab 0.000 0.033 0.014 0.035 0.043 0.000 0.010 0.017 0.017 0.043 0.000 0.000 0.000 0.000 0.001 0.000 0.019 0.011 0.022 0.032

± 0.000 ± 0.004 ± 0.000 ± 0.003 ± 0.000 ± 0.000 ± 0.022 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.003 ± 0.000 ± 0.001 ± 0.000

TabTransf. 0.064 0.075 0.071 0.103 0.100 0.000 0.010 0.042 0.062 0.056 0.002 0.002 0.008 0.000 0.010 0.032 0.041 0.048 0.067 0.067

± 0.000 ± 0.000 ± 0.000 ± 0.012 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.006 ± 0.000 ± 0.000 ± 0.001 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.000 ± 0.004 ± 0.000

TransTab 0.179 0.205 0.271 0.266 0.286 0.198 0.239 0.310 0.393 0.369 0.000 0.000 0.016 0.052 0.075 0.139 0.162 0.217 0.245 0.254

± 0.003 ± 0.011 ± 0.001 ± 0.002 ± 0.013 ± 0.003 ± 0.003 ± 0.008 ± 0.007 ± 0.001 ± 0.000 ± 0.000 ± 0.001 ± 0.002 ± 0.001 ± 0.002 ± 0.006 ± 0.002 ± 0.003 ± 0.006

Avg. 0.082 0.112 0.126 0.153 0.166 0.064 0.085 0.122 0.157 0.171 0.004 0.005 0.015 0.034 0.052 0.058 0.078 0.097 0.124 0.139

###### L.3 Probe Head for Record Linkage

- Table 27: Ablation: Probe head for record linkage. Four evaluation protocols on frozen row embeddings (base 768-dim): Cosine = cosine-similarity thresholding (unsupervised). Linear = logistic regression probe. MLP = one-hidden-layer MLP probe (hidden size 256). Dummy = majorityclass baseline. Binary F1 (match class, see Appendix H), 5-seed average, grouped by dataset family.

∆L-C = Linear − Cosine (gain from supervised probing).

DM-C (8 pairs) F1↑ DM-D (4 pairs) F1↑ WDC (4 pairs) F1↑ All (16 pairs) F1↑ Model Cos Lin MLP Dum ∆L-C Cos Lin MLP Dum ∆L-C Cos Lin MLP Dum ∆L-C Cos Lin MLP Dum ∆L-C

BERT 0.390 0.349 0.487 0.000 −0.04 0.315 0.388 0.541 0.000 +0.07 0.390 0.093 0.379 0.000 −0.30 0.371 0.295 0.473 0.000 −0.08 GTE 0.698 0.334 0.451 0.000 −0.36 0.728 0.442 0.589 0.000 −0.29 0.511 0.072 0.550 0.000 −0.44 0.659 0.295 0.510 0.000 −0.36 TUTA 0.363 0.317 0.437 0.000 −0.05 0.397 0.370 0.532 0.000 −0.03 0.354 0.090 0.365 0.000 −0.26 0.369 0.273 0.443 0.000 −0.10 TABBIE 0.309 0.309 0.421 0.000 +0.00 0.296 0.302 0.359 0.000 +0.01 0.389 0.092 0.188 0.000 −0.30 0.326 0.253 0.347 0.000 −0.07 TabICL 0.377 0.187 0.444 0.000 −0.19 0.328 0.193 0.442 0.000 −0.14 0.341 0.048 0.245 0.000 −0.29 0.356 0.154 0.394 0.000 −0.20 TabPFN 0.260 0.165 0.343 0.000 −0.10 0.285 0.153 0.348 0.000 −0.13 0.387 0.023 0.152 0.000 −0.36 0.298 0.126 0.296 0.000 −0.17

TransTab 0.410 0.258 0.419 0.000 −0.15 0.311 0.363 0.482 0.000 +0.05 0.567 0.041 0.760 0.000 −0.53 0.425 0.230 0.520 0.000 −0.19 VIME 0.242 0.150 0.364 0.000 −0.09 0.295 0.148 0.370 0.000 −0.15 0.423 0.050 0.149 0.000 −0.37 0.301 0.125 0.311 0.000 −0.18 SCARF 0.350 0.148 0.384 0.000 −0.20 0.352 0.107 0.408 0.000 −0.24 0.357 0.037 0.102 0.000 −0.32 0.352 0.110 0.320 0.000 −0.24 DAE 0.265 0.152 0.330 0.000 −0.11 0.297 0.145 0.360 0.000 −0.15 0.429 0.051 0.145 0.000 −0.38 0.314 0.125 0.291 0.000 −0.19 TabBinning 0.387 0.141 0.371 0.000 −0.25 0.421 0.141 0.417 0.000 −0.28 0.383 0.041 0.095 0.000 −0.34 0.395 0.116 0.313 0.000 −0.28 SAINT 0.251 0.169 0.165 0.000 −0.08 0.286 0.163 0.189 0.000 −0.12 0.429 0.032 0.235 0.000 −0.40 0.304 0.133 0.188 0.000 −0.17 SubTab 0.275 0.034 0.154 0.000 −0.24 0.331 0.014 0.229 0.000 −0.32 0.428 0.000 0.017 0.000 −0.43 0.327 0.020 0.138 0.000 −0.31 TabTransf. 0.257 0.097 0.068 0.000 −0.16 0.301 0.066 0.111 0.000 −0.24 0.426 0.022 0.018 0.000 −0.40 0.310 0.071 0.066 0.000 −0.24

Avg. 0.345 0.201 0.346 0.000 −0.14 0.353 0.214 0.384 0.000 −0.14 0.415 0.049 0.243 0.000 −0.37 0.365 0.166 0.330 0.000 −0.20

- Table 27 compares cosine similarity, linear probing, an MLP probe, and a dummy baseline across 14 models on the record linkage task.

Cosine outperforms linear for most models. The average ∆L-C is −0.20 overall, meaning supervised linear probing hurts relative to unsupervised cosine matching. This holds across all three

dataset families and is most severe on WDC (−0.37), where linear probes average only F1=0.049 versus cosine’s 0.415.

A nonlinear head recovers most of what the linear probe leaves on the table. Averaged across all 16 pairs, the MLP head lifts F1 by +0.164 over the linear probe (0.330 vs. 0.166). The gap is largest on WDC (+0.194) where linear probes nearly collapse, and still substantial on DM-D (+0.170) and DM-C (+0.145). At the model level, WDC produces the most dramatic reversals for TRANSTAB (0.041 Lin → 0.760 MLP), GTE (0.072 → 0.550), and BERT (0.093 → 0.379). These models encode usable entity-matching structure that a linear probe cannot access in 5-seed training. Overall, the MLP score 0.330 (All) sits below cosine’s 0.365, with MLP beating cosine on DM-D (+0.031) and matching it on DM-C (+0.001). Cosine retains its lead on WDC (−0.172).

WDC is the hardest for linear probes, and still hard for some MLP probes. The linear probe nearly collapses on WDC for target-table SSL models (e.g., SubTab: 0.000, TabTransf.: 0.022), whereas cosine similarity remains non-trivial (∼0.38–0.43). The MLP head partially rescues this for TRANSTAB (0.760, target-table SSL with cross-table contrastive objective), GTE (0.550), BERT (0.379), and TABICL (0.245), but it stays near-zero for the target-table SSL encoders on WDC (SubTab: 0.017, TabTransf.: 0.018, TabBinning: 0.095, SCARF: 0.102). These embeddings appear to lack WDC-relevant entity-matching signal rather than merely hiding it in non-linear form.

Implication for the main evaluation. Main record linkage results follow the unified supervisedprobe protocol of Sec. 3.1, which averages the linear and MLP probe heads. We headline avg(linear, MLP) rather than cosine because it is the only head that applies uniformly to every row sub-task (cosine is undefined for prediction) and because the linear and MLP heads span the linear-vs-nonlinear capacity axis at a fixed supervision level. Averaging does not privilege either regime, which matters because some encoder/source combinations carry linkage signal that is linearly accessible (text encoders on DeepMatcher) while others need a nonlinear readout (TRANSTAB, GTE, BERT on WDC). This appendix disentangles the two: the MLP head recovers a meaningful slice of the transferable matching signal that cosine can extract, while the linear probe under-reads it, especially on WDC. Averaging the two therefore pulls headline numbers downward for models with strong nonlinear structure (TRANSTAB, GTE, BERT on WDC), and the cosine column here is best read

- as a training-free reference rather than an upper bound. The relative strength of learned versus training-free matching varies substantially across dataset families, with cosine stronger on WDC and the MLP head competitive on the DeepMatcher benchmarks.

- Table 28: Source-split row overlap audit for the 16 record-linkage datasets in TRL-RBENCH. Pair overlap is the fraction of test pairs that also appear in the train pair list. Row overlap reports the fraction of distinct test-side tableA / tableB rows that already appear in the train+valid pair lists. The original DeepMatcher [53] and WDC LSPM [62] splits are pair-disjoint by construction (last column), but most sources keep individual rows across splits because each row participates in many candidate pairs. Beer, iTunes-Amazon, iTunes-Amazon-D, WDC-medium, and WDC-small are the only sources where fewer than half of the test-side rows are seen during training on at least one side.

Test rows in train+valid Dataset train valid test tableA (%) tableB (%) Pair overlap (%) DeepMatcher Clean (DM-C)

Abt-Buy 5,743 1,916 1,916 94.2 95.0 0.00 Amazon-Google 6,874 2,293 2,293 90.9 85.5 0.00 Beer 268 91 91 40.8 51.8 0.00 DBLP-ACM 7,417 2,473 2,473 87.0 87.7 0.00 DBLP-Scholar 17,223 5,742 5,742 95.8 69.5 0.00 Fodors-Zagats 567 190 189 81.0 86.8 0.00 iTunes-Amazon 321 109 109 25.0 15.1 0.00 Walmart-Amazon 6,144 2,049 2,049 87.9 54.8 0.00

DeepMatcher Dirty (DM-D)

DBLP-ACM-D 7,417 2,473 2,473 87.0 87.7 0.00 DBLP-Scholar-D 17,223 5,742 5,742 95.8 69.5 0.00 iTunes-Amazon-D 321 109 109 25.0 15.1 0.00 Walmart-Amazon-D 6,144 2,049 2,049 87.9 54.8 0.00

WDC Products LSPM v2

WDC-small 7,230 1,808 4,398 22.0 21.5 0.00 WDC-medium 20,453 5,114 4,398 53.4 49.1 0.00 WDC-large 82,714 20,683 4,398 74.2 68.9 0.00 WDC-xlarge 171,714 42,947 4,398 75.0 70.8 0.02

###### L.4 Record Linkage Split and Leakage Audit

The 16 record-linkage sources keep their original DeepMatcher [53] and WDC LSPM [62] pairdisjoint splits, so essentially no pair appears in both train and test (pair overlap ≤ 0.02% across all 16 datasets in Table 28). Because each tableA / tableB row participates in many candidate pairs, however, the same row can appear on both sides of the split: in 11 of 16 sources, more than half of the test-side rows on both tableA and tableB already appear in the train+valid pair lists (e.g., Abt-Buy: 94.2% / 95.0%; DBLP-Scholar: 95.8% / 69.5%). Beer, iTunes-Amazon, iTunes-Amazon-D, WDC-medium, and WDC-small are the five sources with cross-split row overlap below 50% on at least one side. We keep all sources to remain comparable with the entity-matching literature.

Removal of label-equivalent columns. WDC LSPM v2 raw records ship with three columns that do not belong in feature input: cluster_id is the gold product cluster identifier (column equality reproduces the test label at 99.5% precision and 99.6% recall on every WDC size), and identifiers contains GTIN/MPN unique product IDs (99.8% precision and 37.5% recall) that are excluded from features in the standard WDC LSPM evaluation protocol. Fodors-Zagats analogously exposes a class column whose value is the entity cluster ID (100% precision, 100% recall on the test split). TRL-BENCH removes cluster_id and identifiers from the four WDC tables and class from Fodors-Zagats before any encoder serializes a row, so the gold label cannot enter the row representation as a feature. The retained columns are brand, category, description, keyValuePairs, price, specTableContent, title for WDC, and name, addr, city, phone, type for Fodors-Zagats.

Row-disjoint strict-test ablation. To check that the row-overlap reported above does not distort cross-model rankings, we build a row-disjoint variant of each source: train+valid stay as-is, and the test pair list is filtered to pairs whose tableA and tableB rows do not appear in train+valid. The filter strips most pairs from the high-overlap sources, so we report the strict ablation only on the 10 sources whose strict-test stays ≥ 30 pairs and keeps a ≥ 10% minority class: 6 DeepMatcher viable sources (Amazon-Google, Beer, iTunes-Amazon / iTunes-Amazon-Dirty, WalmartAmazon / Walmart-Amazon-Dirty) and 4 WDC sizes; the 6 skipped sources (Abt-Buy, Fodors-Zagats, DBLP-ACM / -Dirty, DBLP-Scholar / -Dirty) lose either viable pair count or label balance. Rerunning the unified probe protocol of Sec. 3.1 on the strict-test subset at seed 42 and comparing per-model strict-test NR rankings with canonical 5-seed-mean original-protocol rankings on the same

10 viable strict-test sources, the strict-vs-original linkage rankings are highly correlated across all 14 row models: Spearman ρ = 0.94 (p = 5.6 × 10−7) over the full 10-source set, ρ = 0.97 on DM-viable (6 sources), and ρ = 0.95 on WDC (4 sources). The model-family conclusions of Sec. 4.3 (Transfer-Based encoders dominate Robust Linkage; Target-Table SSL trail) hold under the strict-test ablation. Row overlap does not change the qualitative reading. Table 29 reports per-source strict-test pair counts, positive rates, and absolute F1 for the two top-ranked Robust Linkage models (GTE and TRANSTAB).

- Table 29: Strict-test row-disjoint ablation at seed 42: per-source pair counts and the per-model linkage

F1 (avg of MLP and linear probes) of the two top-ranked Robust Linkage row models (GTE and TRANSTAB), compared with the legacy 5-seed avg. on the same source. Strict-test rows are absent

from train+valid by construction (Sec. L.4). “Strict F1” is single-seed (seed 42).

GTE TRANSTAB

###### Source Strict pairs Retained % pos Strict F1 Legacy F1 ∆ Strict F1 Legacy F1 ∆

Amazon-Google 50 2.2% 74.0% 0.700 0.310 +0.389 0.636 0.281 +0.354 Beer 30 33.0% 33.3% 0.502 0.411 +0.092 0.297 0.285 +0.011 iTunes-Amazon 70 64.2% 30.0% 0.565 0.665 -0.100 0.493 0.574 -0.080 iTunes-Amazon-D 70 64.2% 30.0% 0.640 0.677 -0.037 0.648 0.649 -0.001 Walmart-Amazon 105 5.1% 45.7% 0.389 0.199 +0.190 0.401 0.169 +0.232 Walmart-Amazon-D 105 5.1% 45.7% 0.415 0.204 +0.210 0.258 0.171 +0.087 WDC-small 2718 61.8% 26.9% 0.289 0.313 -0.025 0.264 0.402 -0.139 WDC-medium 1195 27.2% 23.8% 0.306 0.307 -0.001 0.228 0.391 -0.163 WDC-large 587 13.3% 15.8% 0.303 0.314 -0.011 0.262 0.406 -0.144 WDC-xlarge 577 13.1% 15.3% 0.309 0.311 -0.002 0.190 0.401 -0.212

###### L.5 Intrinsic Embedding-Geometry Diagnostics for Row Encoders

We complement TRL-RBENCH’s downstream scores with an intrinsic embedding-geometry analysis of the same exported row embeddings used by the standardized protocol. Using eight established diagnostics from the broader representation-learning literature, covering spectral spread, spectral shape, and spatial structure (“task-free” metrics in the prior literature), we ask which geometric properties of row-embedding spaces co-rank with downstream utility on row prediction and record linkage. To our knowledge, prior tabular benchmark resources do not pair representation-level downstream evaluation with intrinsic embedding-geometry diagnostics on the same encoder outputs under a common protocol.

Diagnostic families and formulas. We group the eight task-free diagnostics used in this analysis into three complementary families that we introduce here to organise the discussion: Spectral Spread, Spectral Shape, and Spatial Structure. Each underlying metric is drawn from prior work (cited

- at its definition below). The three-family taxonomy itself is our framing. For each (model, dataset) we form a single frozen embedding matrix X ∈ Rn×d, the row-embedding matrix for that table (for target-table SSL encoders this is the same matrix the encoder was trained on; for frozen transfer and prior-based encoders it is the inference-time row matrix), and evaluate all eight diagnostics from its singular value decomposition

X = U ΣV ⊤, (1) and from the eigenspectrum λ1 ≥ ··· ≥ λd ≥ 0 of the centred covariance

C = n1 Xc⊤ Xc, Xc = X − 1x¯⊤, x¯ = n1 X⊤1. (2)

Let σ1 ≥ σ2 ≥ ··· denote the singular values of X and r its numerical rank. All eight diagnostics are deterministic functions of X.

Spectral Spread. If variance in an embedding matrix concentrates in only a handful of singular directions, most of the ambient dimensions are redundant: the representation effectively lives on a low-dimensional subspace, and the remaining capacity is unavailable to the downstream head. Spectral Spread diagnostics measure how evenly variance is allocated across the spectrum. Higher values mean more independent directions actively carry information, a necessary condition for rich, transferable embeddings.

###### RankMe [26].

min(n,d)

σi j σj

. (3)

RankMe(X) = exp −

pi log pi , pi =

i=1

The exponentiated entropy of the singular-value distribution. It equals 1 for a rank-one spectrum and min(n,d) when all singular values are equal.

###### RankMe⋆ [26, 71].

RankMe(X)

RankMe⋆(X) =

min(n,d) ∈ [0,1]. (4) A dimension-normalised variant: the fraction of the available embedding dimensions the representation actually uses.

###### NESum.

d

1 λ1

λi. (5)

NESum(X) =

i=1

Total variance divided by leading variance. It equals 1 under rank-one collapse and approaches min(n,d) as the covariance spectrum flattens.

Spectral Shape. Two embeddings can have the same effective rank and still look very different along the spectrum: one may drop off sharply after the top few singular directions, while another decays as a slow power-law that keeps weak but non-trivial signal in many more directions. Spectral Shape diagnostics capture this tail profile, which determines how much low-variance structure survives to support tasks whose discriminative signal is not confined to the dominant directions.

###### Pseudo κ.

σmax σmin nonzero

. (6)

κ(X) =

A coarse condition-number proxy. Large κ signals a near-degenerate spectrum in which a few directions dominate.

###### αreq [1].

αreq(X) = −βˆ1, log λi = β0 + β1 log i + ϵi, i = 1,...,r, (7)

where (βˆ0,βˆ1) is the ordinary least-squares estimator. That is, αreq is the decay exponent of a power-law fit λi ∝ i−α to the centred-covariance spectrum. Larger αreq means a faster-decaying, more heavy-tailed spectrum.

Convention note on αreq. We follow the original Agrawal et al. [1] definition, which fits the slope on covariance eigenvalues λi. Tsitsulin et al. [71] restate the same metric on singular values σi. Because λi = σi2/n, the two fitted slopes differ by exactly a factor of two, so our reported αreq values live on the Agrawal scale and should be divided by 2 to compare against Tsitsulin-scale results. To prevent power-law fits through float-precision noise on (nearly) collapsed spectra we drop eigenvalues below ε · d · λ1 (numpy’s default rank tolerance) before the fit. When fewer than two eigenvalues survive we record αreq = NaN rather than a spurious finite slope.

Spatial Structure. Two embeddings can share an identical singular-value spectrum yet lay their points out very differently: one spreading them uniformly, another concentrating them on a lowdimensional manifold or in tight clusters. Spatial Structure diagnostics measure these point-cloudlevel properties, which are invisible to the spectrum alone and reveal whether a representation has acquired a meaningful geometric organisation (beneficial for retrieval- and clustering-style downstream tasks) or has instead collapsed structure that a purely spectral view would not detect.

###### dˆTwoNN [22, 4].

dˆTwoNN(X) = βˆ1, −log 1 − Nk = β0 + β1 log µ(k) + ϵk, k = 1,...,⌊0.9N⌋, (8)

where (βˆ0,βˆ1) is the ordinary least-squares estimator, µi = d2,i/d1,i with d1,i,d2,i the distances from row i to its first and second nearest neighbours, µ(k) the sorted statistic, and N the sample size. This is the manifold-hypothesis intrinsic dimension: the number of independent directions the data actually spans. We average over 20 random 90% subsets of at most 2000 rows (seed 42).

###### Coherence µ0.

n r

d r

∥Ui,:∥22,

∥Vj,:∥22 , (9)

max

max

µ0(X) = max

i

j

computed on the top-r singular vectors. Coherence is high when a small number of rows or dimensions disproportionately drive the representation, and low when energy is spread evenly across all of them.

###### Self-Cluster [71].

d∥X˜X˜⊤∥2F − n(d + n − 1) (d − 1)(n − 1)n

Xi,: ∥Xi,:∥2

, X˜i,: =

. (10)

SC(X) =

Zero in expectation when the row-normalised vectors are i.i.d. isotropic, positive under clustering, so it measures how much the embedding departs from a uniform distribution on the sphere. Rows with zero norm (from upstream non-finite sanitisation) are dropped before normalisation. Table 30 reports the Spearman rank correlation (ρ) between eight diagnostics, grouped into the three families above, and downstream performance on the two row-level task categories. Each cell is a per-task average: we compute Spearman ρ within every task (across 13–14 row-capable models per task after task-specific filtering, with RANDOM excluded throughout), then average the per-task ρ’s. The p-values come from Wilcoxon signed-rank tests of the per-task ρ distribution against zero. This matches the aggregation used in the pertask breakdown tables (Tabs. 32–34), so the headline numbers and the per-head detail tell a single consistent story. The Row Prediction column combines classification and regression tasks, using macro-F1 as the performance-oriented score for classification rows and −nRMSE for regression rows. The per-task-type breakdowns (pure-classification vs. pure-regression, per-head) are in Tables 33 and 34. Table 31 provides the per-task breakdown for Row Prediction (Regression) with MLP and linear heads side by side. Tables 32–34 report per-head detail for all three task types, combining MLP and Linear heads side by side for direct comparison.

Spectral Spread is the most predictive family. RankMe (ρ¯ = +0.485, p < 0.001), RankMe* (+0.471), and NESum (+0.460) are the strongest observed positive correlates: embeddings with

Record Linkage (linear head) Best metric: req ( Spearman = 0.80, 16 tasks, 276 pairs)

Regression (mlp head) Best metric: RankMe ( Spearman = +0.36, 46 tasks, 711 pairs)

Classification (mlp head) Best metric: NESum ( Spearman = +0.32, 76 tasks, 1182 pairs)

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

[Figure 73]

[Figure 74]

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

- 16

- 17

1 2 7 5 1

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

- 16

- 17

- 6 4 2 5 1 6 8 4 2 1 1 3 2 1
- 6 5 3 4 4 2 3 4 2 2 1 2 2 4 1 1 5 6 3 4 8 3 2 1 3 1 2 4 1 1 2 5 2 6 7 6 3 1 2 3 1 2 3 2 2 1

[Figure 75]

[Figure 76]

[Figure 77]

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

- 16

- 17

10 15 6 4 14 5 5 5 3 4 2 1 1 1

[Figure 78]

1 3 2 5 4 1 5 4 1 3 1

- 7 8 7 6 6 6 3 1 8 5 1 3 5 4 4 2

- 4 7 8 9 8 8 9 4 3 1 3 5 2 2 3 6 9 8 6 4 7 6 2 7 5 5 3 3 3 2
- 5 4 7 8 7 8 3 8 2 6 6 4 3 2 3
- 6 6 10 7 2 4 8 6 5 2 5 4 5 1 3 2 4 5 4 5 9 4 5 7 5 4 4 3 6 3 5 3

- 8 3 8 6 2 1 8 6 7 3 9 3 6 4 1 1 6 4 3 7 3 4 5 1 6 6 5 7 7 2 5 5

20.0

1 4 3 1 1 1 1 2 1 2 3 4 2 1 1 1

Rankofperformanceacrossmodels

(1=bestafterdirectioncorrection)

Rankofperformanceacrossmodels

17.5

(1=bestafterdirectioncorrection)

- 3 5 9 3 5 5 4 2 2 1 1 1 4 1
- 4 4 7 4 6 5 5 3 1 2 1 2 2 2 5 4 4 2 1 3 3 3 5 4 2 1 3 3 1

- 1 4 1 7 1 2 4 5 2 4 3 3 5 4

- 1 1 3 1 2 3 1 4 4 5 3 4 3 5 6

5 1 3 2 3 4 2 1 3 4 5 3 2 3 3 2

- 2 4 2 1 3 5 2 1 4 3 6 6 4 2 1 4 1 1 1 1 1 3 4 2 5 4 5 8 2 4

1 1 1 3 5 1 4 7 4 2 4 2 8 3 3 1 2 1 2 2 5 2 7 5 2 3 5 3 3

- 2 1 6 3 5 4 3 8 7 3 4 1 1 1 1 3 2 2 2 5 3

Rankofperformanceacrossmodels

(1=bestafterdirectioncorrection)

1 4 2 1 4 2 1 2 3 3 2 2 1

Countof(task,model)pairs

Countof(task,model)pairs

15.0

Countof(task,model)pairs

- 1 1 2 2 1 2 2 2 1 2 2 1 3 2 3 1 1
- 1 1 2 2 1 3 2 1 1 2 1 1 5 1 2 3 2 1

1 1 1 3 2 2 2 2 1 1

- 1 1 2 2 2 3 3 2
- 2 1 1 4 1 2 1 3 1 1 5 3 3 2 1 1

- 4 1 3 1 2 1 1 1 4 3 2 1

12.5

10.0

- 3 3 5 5 2 4 2 7 8 9 4 6 7 7 4
- 4 2 5 2 3 8 4 5 4 6 7 8 4 7 5 2

7.5

- 2 1 3 3 1 9 3 6 5 5 7 7 8 9 5 2
- 3 2 1 3 1 1 6 8 4 7 4 10 10 8 5 3
- 4 3 3 3 4 2 5 10 3 4 7 7 3 9 8 1 4 4 2 2 5 3 1 6 5 5 3 4 1 8 22 1

5.0

2.5

1 2 4 1 3 1 3 3 4 3 6 1 10

0.0

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17

Rank of req across models (1 = highest metric value)

Rank of RankMe across models (1 = highest metric value)

Rank of NESum across models (1 = highest metric value)

(a) Classification (MLP). Bestcorrelated prior metric: NESum (ρ¯ = +0.32, 76 tasks, with one of 77 classification targets dropped for constant performance across models on MLP). Density on the main diagonal.

(b) Regression (MLP). Bestcorrelated prior metric: RankMe* (ρ¯ = +0.36, 46 tasks). Density on the main diagonal.

(c) Record Linkage (Linear). Bestcorrelated prior metric: αreq (ρ¯ = −0.80, 16 tasks). Density on the anti-diagonal.

- Figure 9: Rank–rank density heatmaps for row tasks. For every task, models are ranked by the best-correlated intrinsic-geometry diagnostic (x-axis, 1 = highest value) and by direction-corrected downstream performance (y-axis, 1 = best). Cell numbers count (task,model) pairs per rank bin. Diagonal concentration indicates positive rank agreement. Anti-diagonal concentration indicates negative agreement.

- Table 30: Spearman rank correlation (ρ) between embedding quality metrics and downstream task performance, reported as per-task averages so the numbers align with the pertask breakdown tables (Tabs. 32–34). Record Linkage: Spearman ρ computed per (dataset, head) across 16 models, then averaged over n = 32 (dataset, head) tasks (16 datasets × {MLP, linear}). Row Prediction: Spearman ρ per (dataset, label, head) across models, averaged over n = 241 task-head cells: classification (76 MLP + 73 linear) and regression (46 MLP + 46 linear) from 77 classification and 46 regression targets crossed with {MLP, linear} probe heads, less 5 classification rows dropped for near-constant performance (undefined ρ). Overall: unweighted mean of Record Linkage and Row Prediction ρ. Reported p is a two-sided Wilcoxon signed-rank test of the per-task ρ distribution against zero. Overall p is the smaller of the two (anti-conservative). Bold ρ indicates p < 0.05. The RANDOM baseline encoder is excluded throughout.

Record Linkage Row Prediction Overall Family Metric ρ p ρ p ρ¯ p

RankMe +0.714 <.001 +0.256 <.001 +0.485 <.001 RankMe* +0.684 <.001 +0.258 <.001 +0.471 <.001 NESum +0.657 <.001 +0.262 <.001 +0.460 <.001

Spec. Spread

Pseudo κ +0.116 0.036 +0.032 0.039 +0.074 0.036 αreq -0.746 <.001 +0.003 0.939 -0.372 <.001

Spec. Shape

dˆTwoNN +0.398 <.001 -0.042 0.095 +0.178 <.001 Coherence µ0 -0.549 <.001 -0.030 0.235 -0.289 <.001 Self-Cluster -0.182 <.001 -0.237 <.001 -0.210 <.001

Spatial Struct.

more uniform singular-value distributions (i.e., higher effective rank) consistently rank higher on both tasks. The αreq metric in Spectral Shape is the strongest observed negative correlate overall (ρ¯ = −0.372), but its signal is concentrated on Record Linkage (ρ = −0.746, p < 0.001). On Row Prediction the per-task mean is essentially zero (ρ = +0.003, p = 0.94) because the classification and regression signs partially cancel (see the per-head breakdown in Tab. 31 and the companion classification table). Heavy-tailed spectral decay therefore predicts entity-matching performance but is an ambivalent signal for feature-based prediction. Within Spatial Structure, dˆTwoNN is informative

for Record Linkage (ρ = +0.398, p < 0.001) but shows no significant signal for Row Prediction (ρ = −0.042, p = 0.095), indicating task-specific utility. Correlations are uniformly stronger for Record Linkage than Row Prediction, implying that intrinsic quality metrics are better proxies for entity-matching tasks than for feature-based prediction.

- Table 31: Row Prediction, Regression: per-task correlation between embedding metrics and −nRMSE, computed across models within each (dataset, label), then aggregated.

MLP Head (46 tasks) Linear Head (46 tasks) Pearson Spearman Dir. Rate Pearson Spearman Dir. Rate

Family Metric Mean Med. Mean Med. P S Mean Med. Mean Med. P S

RankMe +0.317 +0.410 +0.328 +0.480 80.4% 78.3% +0.235 +0.395 +0.257 +0.418 78.3% 76.1% RankMe* +0.341 +0.462 +0.356 +0.502 80.4% 78.3% +0.261 +0.415 +0.287 +0.472 78.3% 73.9% NESum +0.248 +0.346 +0.250 +0.296 78.3% 76.1% +0.225 +0.303 +0.231 +0.315 76.1% 80.4%

Spec. Spread

Pseudo κ +0.118 -0.021 +0.036 +0.118 50.0% 56.5% +0.093 -0.038 -0.006 +0.054 56.5% 54.3% αreq +0.012 +0.014 +0.080 +0.144 52.2% 63.0% +0.119 +0.141 +0.186 +0.229 63.0% 76.1%

Spec. Shape

dˆTwoNN -0.120 -0.288 -0.105 -0.230 65.2% 63.0% -0.165 -0.212 -0.174 -0.266 65.2% 69.6% Coherence µ0 -0.042 -0.044 +0.034 +0.079 52.2% 58.7% +0.011 +0.031 +0.097 +0.137 56.5% 67.4% Self-Cluster -0.316 -0.374 -0.327 -0.414 80.4% 82.6% -0.281 -0.408 -0.296 -0.368 73.9% 80.4%

Spatial Struct.

Row Prediction (Regression) is the most variable. Spectral Spread is the strongest positive signal: RankMe and RankMe* reach MLP Pearson +0.317/+0.341 (Spearman +0.328/+0.356, ∼80% directional), with the linear head slightly weaker (Pearson +0.235/+0.261). NESum follows closely (MLP Pearson +0.248, Spearman +0.250). Spectral Shape is much weaker: Pseudo κ registers MLP Pearson +0.118 (Spearman +0.036), and αreq is essentially flat on the MLP head (Pearson +0.012, Spearman +0.080), emerging only on the linear head (Pearson +0.119, Spearman +0.186). The dominant negative signal is in Spatial Structure: Self-Cluster reaches MLP Pearson −0.316 and Spearman −0.327 (∼80%/83% directional), meaning that tighter intra-model clustering of row embeddings co-varies with worse regression. dˆTwoNN is consistently weakly negative (MLP Pearson −0.120, Spearman −0.105), and Coherence µ0 is near-zero. Relative to classification, regression amplifies both the Spectral Spread positive signal and the Self-Cluster / dˆTwoNN negative signals. αreq flips from weakly negative (classification, MLP Pearson −0.105) to flat or weakly positive (regression). Note: the per-task correlations here (and in Tables 32–34) exclude the RANDOM baseline, matching Table 30. RANDOM is a high-leverage NESum outlier (NESum ≈ 409 vs. ≈4.1 for real encoders, roughly 102× on this aggregation basis) and can exert undue leverage on scale-sensitive Pearson fits.

###### L.6 Intrinsic-Geometry Diagnostics: Per-Head Breakdowns

The following three tables combine MLP and Linear heads side by side for each task type, enabling direct comparison of how probe complexity modulates the predictive power of each quality metric.

- Table 32: Per-task correlation between embedding metrics and record linkage performance (F1, higher is better). We report MLP, Linear, and Cosine-Threshold heads.

MLP head Linear head Cosine Thr. head Metric Spearman Pearson SC Spearman Pearson SC Spearman Pearson SC NESum

0.67

[0.53, 0.76]

0.58 [0.44, 0.69] 0.94

0.64

[0.58, 0.71]

0.59 [0.52, 0.66] 1.00

0.54

[0.45, 0.63]

0.73

[0.62, 0.83] 1.00 RankMe

0.69

[0.54, 0.80]

0.59 [0.42, 0.71] 0.94

0.74

[0.67, 0.79]

0.70 [0.64, 0.74] 1.00

0.56

[0.48, 0.64]

0.83

[0.78, 0.88] 1.00 RankMe⋆

0.68

[0.53, 0.79]

0.60 [0.44, 0.71] 0.94

0.69

[0.63, 0.74]

0.63 [0.57, 0.70] 1.00

0.52

[0.43, 0.61]

0.77

[0.69, 0.84] 1.00 αreq −0.69

[−0.80, −0.54]

−0.69 [−0.77, −0.59] 0.94 −0.80

[−0.85, −0.73]

−0.59 [−0.71, −0.47] 1.00 −0.52

[−0.60, −0.45]

−0.44

[−0.51, −0.37] 1.00 Pseudo κ

0.11

[−0.01, 0.25]

0.16 [−0.04, 0.36] 0.62

0.12

[−0.02, 0.24]

0.05 [−0.11, 0.20] 0.75 −0.02

[−0.13, 0.10]

−0.11

[−0.22, 0.00] 0.56 µ0-coherence −0.57

[−0.67, −0.45]

−0.44 [−0.53, −0.35] 0.94 −0.53

[−0.61, −0.46]

−0.40 [−0.48, −0.32] 1.00 −0.54

[−0.61, −0.47]

−0.38

[−0.44, −0.32] 1.00 Self-cluster −0.25

[−0.33, −0.17]

−0.22 [−0.29, −0.15] 0.94 −0.11

[−0.19, −0.03]

−0.05 [−0.12, 0.04] 0.81 −0.18

[−0.24, −0.12]

−0.18

[−0.24, −0.12] 0.94 TwoNN ID

0.33

[0.20, 0.46]

0.13 [−0.06, 0.31] 0.81

0.46

[0.35, 0.57]

0.32 [0.18, 0.47] 1.00

0.07

[−0.08, 0.20]

0.10 [−0.04, 0.25] 0.53

- Table 33: Per-task correlation between embedding metrics and classification performance (macro-F1, higher is better). Spearman / Pearson cells show mean with bootstrap 95% CI in brackets over per-task correlations. SC is the sign-consistency fraction (fraction of tasks whose Spearman is in the dominant-sign direction).

MLP head Linear head Metric Spearman Pearson SC Spearman Pearson SC NESum

0.32

0.29 [0.23, 0.35] 0.86

0.23

0.27

[0.22, 0.32] 0.79 RankMe

[0.25, 0.38]

[0.17, 0.29]

0.28

0.26 [0.20, 0.32] 0.79

0.19

0.23

[0.17, 0.29] 0.74 RankMe⋆

[0.20, 0.35]

[0.11, 0.26]

0.27

0.26 [0.20, 0.32] 0.76

0.17

0.22

[0.15, 0.28] 0.66 αreq −0.08

[0.19, 0.34]

[0.09, 0.25]

###### −0.11 [−0.19, −0.02] 0.61 −0.07

−0.09

[−0.17, −0.01] 0.60 Pseudo κ

[−0.15, −0.02]

[−0.14, 0.00]

0.06

0.14 [0.07, 0.22] 0.60

0.02

0.16

[0.08, 0.24] 0.55 µ0-coherence −0.11

[−0.01, 0.13]

[−0.04, 0.09]

###### −0.21 [−0.29, −0.13] 0.64 −0.07

−0.19

[−0.28, −0.10] 0.61 Self-cluster −0.23

[−0.18, −0.04]

[−0.14, 0.00]

###### −0.22 [−0.28, −0.15] 0.76 −0.15

−0.16

[−0.23, −0.10] 0.67 TwoNN ID

[−0.31, −0.16]

[−0.23, −0.07]

###### −0.01 [−0.10, 0.09] 0.53 −0.02

−0.05 [−0.14, 0.04] 0.58

0.05

[−0.02, 0.12]

[−0.10, 0.07]

Table√1 −34:R2,Per-tasklower iscorrelationbetter, withbetweenper-taskembeddingSGM(nRMSEmetrics) = nRMSE).and regressionWe correlateperformanceagainst(nRMSE−nRMSE= so positive values mean “higher metric → better performance.”

MLP head Linear head Metric Spearman Pearson SC Spearman Pearson SC NESum

###### 0.25

0.25 [0.14, 0.35] 0.76

0.23

###### 0.23

[0.12, 0.33] 0.80 RankMe

[0.14, 0.36]

[0.12, 0.33]

###### 0.33

0.32 [0.22, 0.41] 0.78

0.26

###### 0.23

[0.11, 0.35] 0.76 RankMe⋆

[0.22, 0.43]

[0.13, 0.38]

###### 0.36

0.34 [0.24, 0.43] 0.78

0.29

0.26

[0.14, 0.37] 0.74 αreq

[0.24, 0.46]

[0.16, 0.41]

0.08

0.01 [−0.09, 0.11] 0.63

0.19

0.12

[0.02, 0.22] 0.76 Pseudo κ

[−0.01, 0.17]

[0.09, 0.28]

###### 0.12 [0.02, 0.22] 0.57 −0.01

0.04

0.09

[−0.00, 0.19] 0.54 µ0-coherence

[−0.06, 0.13]

[−0.09, 0.08]

−0.04 [−0.16, 0.07] 0.59

0.03

0.10

0.01

[−0.11, 0.12] 0.67 Self-cluster −0.33

[−0.09, 0.15]

[−0.02, 0.22]

###### −0.32 [−0.41, −0.22] 0.83 −0.30

−0.28

[−0.40, −0.16] 0.80 TwoNN ID −0.11

[−0.42, −0.23]

[−0.41, −0.18]

###### −0.12 [−0.25, 0.02] 0.63 −0.17

−0.17 [−0.31, −0.02] 0.70

[−0.23, 0.02]

[−0.31, −0.04]

##### M DLTE Operator Specification

This section specifies the complete DLTE pipeline operators summarized in Sec. 3.4. Algorithm 1 is the end-to-end procedure. The following subsections give the CSLS formula, Stage-2 threshold grid, Stage-3 match-profile scalars, and the union-appended-row second-pass in Stage 3. Nothing below is trained end-to-end: Stage-2 calibrates five scalars per (Stage-1 backbone, column model) pair on the dev split by grid search over macro-F1. Stage-1 and Stage-3 have no tunable parameters beyond the fixed profile scalars reported here.

Algorithm 1 TRL-DLTE pipeline (per query q, lake L).

Require: Frozen encoder outputs etbl, ecol, erow; retrieval depth K = 100;

- Stage-2 thresholds τfloor, τu, τus, τjm, τks calibrated per (Stage-1 backbone, column model) pair (Ta-

ble 35);

- Stage-3 match profiles Πu (union) and Πj (join) from Table 36.

- 1: Stage 1 (table retrieval). L2-normalize etbl(q) and query the pre-built FAISS [37] inner-product index over the lake’s normalized table embeddings; return top-K candidates CK(q).
- 2: Stage 2 (column alignment + relation classification).
- 3: for all c ∈ CK(q) do
- 4: Form cost Dij = 1−cos(ecolq,i, ecolc,j); solve Hungarian assignment on D to get matched cosine similarities S = (s1, . . . , sL).
- 5: Let n⋆ = |{sk ≥ τfloor}|, r = n⋆/|C(q)|, µ = mean{sk : sk ≥ τfloor}, m = max S.
- 6: if r ≥ τu and µ ≥ τus then
- 7: yˆ(q, c) ← UNION.
- 8: else if r ≤ τjm and m ≥ τks and n⋆ ∈ {1, 2, 3} then
- 9: yˆ(q, c) ← JOIN; key pair ← arg maxk sk.
- 10: else
- 11: yˆ(q, c) ← NONE.
- 12: end if
- 13: end for
- 14: Stage 3 (row matching + merge).
- 15: Pick cu ← arg maxyˆ(q,c)=UNION κu(q, c) and cj ← arg maxyˆ(q,c)=JOIN κj(q, c) over CK(q), if any; ties broken by Stage-1 rank (see Sec. M.3 for κ).
- 16: Initialize q′ ← q.
- 17: if cu exists then
- 18: Dedup union. Reciprocal-match R(cu) against R(q) via CSLS + profile Πu; append unmatched rows of cu to q′ using the Stage-2 column alignment.
- 19: end if
- 20: if cj exists then
- 21: Join. Iteratively reciprocal-match seed rows R(q) against R(cj) via CSLS + Πj for up to Ij = 10 rounds; copy non-key join columns into q′.
- 22: Second pass. Reciprocal-match union-appended rows against as-yet-unmatched R(cj) (Sec. M.4).
- 23: end if
- 24: return enriched query q′.

###### M.1 Stage-1 Retrieval: Scoring and Pool

Given table embeddings etbl(t) ∈ Rd, we L2-normalize each vector and build a FAISS IndexFlatIP over the 47,772-table lake. Inner-product search on unit vectors returns cosine-ranked candidates without approximation error. For the Stage-1 retrieval pool specifically, column-capable encoders are pooled to a table embedding via column mean (variant column_mean). The two native table encoders use their native variants (TAPEX: table_embedding, TUTA: cls_embedding). We retrieve the top K = 100 candidates and pass all of them to Stage 2.

###### M.2 Stage-2 Alignment and Classification: Thresholds and Grid

Hungarian assignment is solved with scipy.optimize.linear_sum_assignment on cost Dij = 1 − cos ecolq,i,ecolc,j with L = min(|C(q)|,|C(c)|) matched pairs. Column embeddings are L2normalized before cosine. NONE is the default label and the per-pair statistics used for classification are the matched similarities S and derived scalars n⋆,r,µ,m defined in Algorithm 1. When n⋆ = 0 (no matched pair clears τfloor) we set µ ← 0 by convention.

Five thresholds are grid-searched per (Stage-1 backbone, column model) pair on the dev split to maximize three-way macro-F1 over {UNION, JOIN, NONE}. The objective is macro-F1 because NONE dominates (∼98% of pairs) and accuracy optimization collapses to the majority class. Calibration is per-pair because different Stage-1 backbones yield different candidate distributions entering Stage-2, so a per-pair operating point isolates Stage-2 classification quality conditional on the retrieval geometry rather than conflating the two. The grid (Table 35) visits 5 × 6 × 6 × 5 × 5 = 4500 combinations per pair across 10 × 8 = 80 (Stage-1, Stage-2) pairs; total calibration cost is small because per-pair alignments are computed once and only the threshold-dependent per-pair statistics are revisited across the grid. The resulting threshold vector is held fixed across all 14 Stage-3 row models paired with that (Stage-1, Stage-2) pair, and is stored alongside the predictions in the released code. Both calibrated thresholds and downstream metrics are deterministic given the embeddings.

- Table 35: Stage-2 threshold search space. Reported results always use the dev-selected threshold vector of the corresponding (Stage-1, Stage-2) pair. No shared default vector is used in evaluation.

Symbol Interpretation Grid range (step) τfloor Min. per-pair similarity to count as matched [0.70, 0.90] (0.05) τu Min. match ratio r for UNION [0.50, 1.00] (0.10) τus Min. mean matched similarity µ for UNION [0.70, 0.95] (0.05) τjm Max. match ratio r for JOIN [0.20, 0.60] (0.10) τks Min. key-column similarity m for JOIN [0.75, 0.95] (0.05)

###### M.3 Stage-3 Row Matching: CSLS and Profiles

Candidate-selection scores. Stage 3 consumes, for each query q, the highest-confidence UNION and JOIN candidate from the K Stage-2 outputs. We use a class-conditional score κ matched to the decision rule: for a candidate classified as UNION we set κu(q,c) = min(r,µ) (the binding scalar of the union rule), and for JOIN we set κj(q,c) = m (the key-column similarity). Ties are broken by Stage-1 retrieval rank. All Stage-3 operations reuse the precomputed frozen row embeddings of the source tables; after union append or join merge, the enriched table q′ is never re-encoded.

CSLS similarity. Let M ∈ R|A|×|B| be the raw cosine matrix between two row sets A,B. For kcsls = 5, let ri be the mean of the top-kcsls entries of row i of M and cj the mean of the top-kcsls entries of column j. The CSLS-normalized similarity is

sCSLS(i,j) = 2Mij − ri − cj. (11)

CSLS [47] discounts hub-like rows/columns whose neighborhoods are dense and makes mutual top-1 pairs more robustly reciprocal.

Reciprocal matching with local confidence filters. Given sCSLS, a pair (i,j) is mutual top-1 if j = arg maxj′ sCSLS(i,j′) and i = arg maxi′ sCSLS(i′,j). For each row i, let si(1) ≥ si(2) denote its top-two CSLS scores and let µi,σi be the mean and standard deviation of the i-th row of sCSLS. We define the standardized signals

σi if σi ≥ ϵ 1 otherwise

si(1) − si(2) σi+

si(1) − µi σi+

, zimargin =

zibest =

, σi+ ≡

(12)

with ϵ = 10−12 (the second case only fires on degenerate rows where every CSLS score is numerically identical), and the analogous candidate-side quantities zjbest,zjmargin computed over the column axis of sCSLS. A mutual top-1 pair (i,j) is accepted iff min(zibest,zjbest) ≥ zminbest and min(zimargin,zjmargin) ≥ zminmargin. Iterative matching removes accepted pairs and re-computes sCSLS on the remaining rows/columns for up to a profile-specific number of rounds.

Profiles Πu and Πj. The two enrichment paths have asymmetric costs: false union appends (duplicate rows) are costlier than missed joins (unmatched new columns on some rows). We therefore use two fixed profile vectors (Table 36) rather than a single shared scalar. Profiles are not tuned per model. They are held fixed across all Stage-3 row encoders, so the reported Stage-3 variation reflects row-embedding geometry rather than operator calibration.

- Table 36: Stage-3 match profiles. Both profiles use CSLS with kcsls = 5. Profiles are fixed across all row models.

Scalar Πu (union, precision-first) Πj (join, recall-first)

Imax (iterations) 3 10 zminbest 1.00 0.75 zminmargin 0.25 0.10 smin (absolute floor) disabled disabled

###### M.4 Stage-3 Second-Pass Join on Union-Appended Rows

The join phase first matches the seed rows of q against R(cj). This leaves union-appended rows (new rows introduced by the union path) without join-side coverage. A second reciprocal pass then matches those appended rows against the remaining rows of cj (those not yet consumed by the first pass) under the same profile Πj. Cells are filled for newly matched pairs using the Stage-2 key-pair alignment, and the second-pass match count is logged separately. This mechanism is what produces the hard-region recall (Table 39) of the enriched-table quadrant where new rows meet new columns; without it, hard-region recall collapses to zero for all row models.

###### M.5 What is and is not tuned

Stages 1 and 3 are fully fixed across models and splits: all scalars in Table 36 and the FAISS retrieval depth K are constants, independent of the encoder under evaluation. Stage 2 calibrates a five-scalar operating point per (Stage-1 backbone, column model) pair (80 calibration runs in total) on the dev split, using Stage-2 three-way macro-F1 over {UNION, JOIN, NONE} as the sole objective. Neither test labels nor end-to-end Cell-F1 / UJ-H, and no Stage-3 row-model choice, enter this calibration. The resulting dev-selected threshold vector of each pair is held fixed across all 14 Stage-3 row models paired with that (Stage-1, Stage-2) pair throughout the full pipeline evaluation. Headline pipeline selection (Sec. 4.4) is a separate model-selection step that uses dev UJ-H as the sole criterion and does not enter this Stage-2 calibration.

##### N DLTE Detailed Rankings

[Figure 79]

[Figure 80]

0.20

0.20

TabTransf.

TabTransf.

0.18

0.18

SAINT

SAINT

SubTab

SubTab

0.16

0.16

TABBIE

TABBIE

TabBinning

TabBinning

0.14

0.14

Stage 3 (row model)

Stage 3 (row model)

VIME

VIME

esseUH

esseUH

TabPFN

TabPFN

0.12

0.12

SCARF

SCARF

DAE

DAE

BERT

BERT

0.10

0.10

TUTA

TUTA

TabICL

TabICL

GTE

GTE

TransTab

TransTab

TABBIE

TABBIE

TabSketchFMTAPAS

TabSketchFMTAPAS

TABBIE

TABBIE

TURL

TURL

TURL

TURL

0.08

0.08

TaBERT

TaBERT

GTE

GTE

TAPEX

TAPEX

Stage1(tablemodel)

Stage1(tablemodel)

BERT

BERT

BERT

BERT

TAPAS

TAPAS

Stage2(columnmodel)

Stage2(columnmodel)

GTE

GTE

TaBERT StarmieTabSketchFM

TaBERT StarmieTabSketchFM

TUTA

TUTA

Starmie

Starmie

(a) Full cube: all 10 × 8 × 14 = 1120 pipelines. LowUJ-H voxels fade into light blue, high-UJ-H voxels pop in deep purple.

(b) Top 40% pipelines only (UJ-H ≥ 60th percentile), highlighting the strong-performing regions of the cube.

- Figure 10: Voxel visualisation of the DLTE Stage-3 pipeline space over UJ-H. Axes: Stage 1 (table model, 10) × Stage 2 (column model, 8) × Stage 3 (row model, 14). Colour encodes UJ-H (light blue → deep purple). Axes are reordered by marginal-mean UJ-H so the best-performing corner is contiguous. See the full per-pipeline breakdown in Table 44 (Sec. N.6).

Dev/test rank stability. The dev-selection protocol of Sec. 4.4 relies on dev-test pipeline rank similarity. Across all 1,120 canonical pipelines (5-round mean UJ-H per pipeline), Spearman ρ(dev,test) = 0.96 (p ≪ 10−100, n = 1,120; Kendall τ = 0.84), and the top-50 by dev UJ-H and the top-50 by test UJ-H share 42 of 50 pipelines. This justifies treating dev-selected pipelines as descriptive of the broader test landscape.

###### N.1 Cell F1 as a Complementary Diagnostic

Cell F1 is the multiset F1 score over recovered cells, pooling the removed-row and removed-column blocks for each query. With Cp(q) and Cg(q) the multisets of cells in the pipeline’s predicted enrichment and the ground-truth blocks for query q,

2|Cp(q) ∩ Cg(q)| |Cp(q)| + |Cg(q)|

CellF1(q) =

,

averaged over queries. It measures how well a pipeline reconstructs parent-table cells in raw cell terms, regardless of how that recovery is distributed between the union and join paths. This makes Cell F1 a complement to UJ-H, the primary end-to-end score for joint recovery of the union and join targets, and we use it here to diagnose stage behavior and high-volume recovery modes in the same 1,120-pipeline space.

Per-stage observations. At Stage 1 (table retrieval), the Cell F1 marginal identifies STARMIE as the strongest retriever (0.601), followed by TUTA (0.593) and GTE (0.591). This agrees with the top of the UJ-H marginal ranking and shows that retrieval quality is a shared driver under both pooled-cell and joint-recovery views (Table 38). At Stage 2 (column alignment plus union/join/none decisions), TABERT leads on Cell F1 (0.628), ahead of GTE (0.601) and TAPAS (0.600). The

- Stage 2 Cell F1 span is 0.084, the largest of the three stages, indicating that pooled cell recovery is especially sensitive to the column-side model. At Stage 3 (row matching and merge), the Cell F1 marginals are compressed (span 0.026), with TABTRANSFORMER (0.591), SUBTAB (0.591), and SAINT (0.590) leading. Oracle-RA (Table 39) clarifies the mechanism: these row models obtain

their Cell F1 primarily from near-complete union-side recovery when retrieval and alignment are supplied, separating a union-preservation behavior from the identity-resolution behavior surfaced by Robust Linkage and Oracle-RA UJ-H.

Pipeline-level signature. The highest-Cell F1 pipelines share a consistent composition: all top-20 use TABERT at Stage 2, and Stage 3 concentrates on TABTRANSFORMER, SUBTAB, SAINT, and

TABBIE (Table 45). The best Cell F1 pipeline is STARMIE/TABERT/TABTRANSFORMER at 0.679. The signature is therefore strong table retrieval, TABERT column-side decisions, and row models that preserve union-side cells. This is an auxiliary lens for workloads that prioritize total recovered cell yield, or for diagnosing which stage limits pooled cell recovery. Overall, Cell F1 adds a practical diagnostic layer: it confirms the Stage 1 retrieval signal, identifies pooled-cell yield as most sensitive to Stage 2, and exposes a Stage 3 union-preservation mode. The full marginal, pipeline, Oracle-RA, and source-split tables report both metrics so the joint-recovery and pooled-cell views can be read directly (Tables 38, 41–43, 45, 44, 39; Appendix N.8).

###### N.2 Pipeline Component Sensitivity

Table 37: Ablation: DLTE pipeline component sensitivity. End-to-end cell F1 (↑) on the test set as a function of Stage 1 table retrieval model (rows) and Stage 2 column alignment model (columns).

- Stage 3 row matching is held fixed to the best-performing row model ( TabTransf. , selected as the

Stage 3 model with the highest mean cell F1 across all (Stage 1, Stage 2) pairs). Each cell shows the full pipeline performance for that (retrieval, alignment) pair. Bold = best retrieval model per column alignment. underline = second.

Stage 2: Column Alignment Model

Stage 1: Retrieval BERT GTE TaBERT TAPAS TURL Starmie TabSketchFM TABBIE Avg.

BERT 0.607 0.608 0.673 0.612 0.603 0.544 0.554 0.570 0.597 GTE 0.608 0.609 0.677 0.612 0.604 0.544 0.555 0.579 0.598 TaBERT 0.608 0.611 0.676 0.626 0.614 0.544 0.554 0.567 0.600 TAPAS 0.608 0.609 0.676 0.609 0.605 0.544 0.553 0.565 0.596 TURL 0.606 0.605 0.667 0.610 0.599 0.544 0.557 0.575 0.595 Starmie 0.609 0.639 0.679 0.645 0.615 0.544 0.566 0.573 0.609 TabSketchFM 0.627 0.601 0.650 0.615 0.602 0.544 0.552 0.556 0.593 TABBIE 0.586 0.571 0.575 0.573 0.579 0.544 0.550 0.541 0.565 TAPEX 0.575 0.573 0.575 0.565 0.584 0.544 0.547 0.512 0.559 TUTA 0.602 0.628 0.650 0.629 0.615 0.544 0.554 0.567 0.599

Avg. 0.604 0.605 0.650 0.610 0.602 0.544 0.554 0.560 0.591

- Table 37 reports end-to-end Cell F1 as a function of Stage 1 (retrieval) and Stage 2 (column alignment) model choices, with Stage 3 fixed to the best Cell F1 row model in this ablation, TABTRANSFORMER. Among Stage 1 models, Starmie leads (row average 0.609), followed by TaBERT (0.600) and the native table encoder TUTA (0.599). TAPEX trails at 0.559 and TABBIE at 0.565, indicating that weak retrieval creates a hard ceiling for downstream performance. The 10-model Stage 1 pool covers all table-capable encoders: the 8 column-capable models (whose column embeddings are pooled to a table embedding) plus the two native table encoders TAPEX and TUTA. Among Stage 2 models, TaBERT dominates column alignment (column average 0.650), far ahead of the next-best TAPAS (0.610). Starmie at Stage 2 collapses to a flat 0.544 regardless of retrieval model. The wide Stage 2 spread (0.106) confirms that column alignment choice has the largest downstream effect. These results isolate column alignment under a fixed Stage 3 model (TABTRANSF.). They are conditional, not the unconditional Stage-2 marginal used in Sec. 4.4 (which leads with TABBIE on test and BERT on dev). The discrepancy is expected under non-additive composition: a column model’s apparent strength depends on the upstream retriever and downstream row matcher with which it is paired.

N.3 Per-Stage Marginal Analysis

- Table 38 decomposes end-to-end performance into per-stage marginal contributions. For a fixed table model, scores are averaged over all 112 compatible pipelines, for a fixed column model over all 140, and for a fixed row model over all 80.

- Stage 2 (column model) exhibits the widest UJ-H span (0.060) and the widest Cell F1 span (0.084), confirming it has the largest average downstream effect under the current pipeline. Stage 1 (table model) shows a notable disconnect between retrieval recall and end-to-end contribution: STARMIE ranks 1st on marginal Cell F1 despite only 3rd-best recall@100, while GTE achieves the highest recall (0.801) but ranks 3rd on Cell F1, behind TUTA (2nd at 0.593). TAPEX further illustrates the disconnect in the opposite direction: it has the lowest marginal Cell F1 (0.558) and nearlowest recall (0.247 R@100, above only TABBIE’s 0.108), consistent with poor retrieval limiting

downstream quality. Stage 3 (row model) has the narrowest span (0.026 Cell F1) and an even smaller UJ-H span (0.013), confirming that upstream errors largely mask row-model differences in the full pipeline. Marginal rankings are main-effect summaries rather than globally optimal compositions. For a pipeline p = (t,c,r) with end-to-end score y(p), the per-stage marginals are mT(t) = Ec,r[y(t,c,r)], mC(c) = Et,r[y(t,c,r)], and mR(r) = Et,c[y(t,c,r)], and the additive

- Table 38: Per-stage marginal contributions in TRL-DLTE (5-round average, test set). For a fixed table model, scores are averaged over all 112 compatible pipelines. For a fixed column model, over all

140. For a fixed row model, over all 80. Bold orange / Underlined blue / Light purple highlights indicate best/second-best/third-best per column within each stage panel. Stage 1 additionally reports target recall@100 (mean fraction of the two relevant targets recovered among top-100 candidates) for reference. Performance span = best − worst marginal score within the stage.

Stage 1: Table Model Stage 2: Column Model Stage 3: Row Model Model Cell

UJ-H ↑

Tgt. R@100↑ Model Cell

UJ-H ↑ Model Cell

UJ-H ↑

F1 ↑

F1 ↑

F1 ↑

Starmie .601 .144 .740 TaBERT .628 .128 TabTransf. .591 .119 TUTA .593 .138 .585 GTE .601 .141 SubTab .591 .121 GTE .591 .129 .801 TAPAS .600 .132 SAINT .590 .119 TaBERT .590 .124 .720 TURL .597 .143 TABBIE .589 .122

BERT .589 .128 .763 BERT .595 .135 BERT .581 .128 TAPAS .587 .118 .615 TABBIE .554 .143 TransTab .574 .132 TURL .587 .124 .597 TabSketchFM .553 .100 GTE .574 .131 TabSketchFM .584 .116 .413 Starmie .544 .084 TabICL .566 .130 TABBIE .560 .109 .108 TAPEX .558 .127 .247

Span .043 .036 .693 Span .084 .060 Span .026 .013

Note. Stage 3 shows the top 4 and bottom 4 (by Cell F1) of 14 row models, separated by a rule; the full ranking is in Appendix N. Stage 2 has the widest average downstream UJ-H span (0.060) and the widest Cell F1 span (0.084) under the current pipeline, indicating that column-model choice has the largest mean effect among the three stages. Stage 1’s target recall@100 ranking does not match its downstream Cell F1 ranking (e.g., GTE has the best retrieval score but STARMIE yields the best downstream Cell F1 marginal), showing that retrieval quality alone does not determine end-to-end enrichment.

###### Stage 3: Transfer-Based (overall avg = 0.1277)

- Stage 2 (Col Model)

0.138 0.135 0.157 0.094

0.140 0.125 0.143 0.090

0.148 0.129 0.136 0.091

0.156 0.133 0.149 0.096

- Stage 3: Prior-Based

###### Stage 3: Target-Table Learners (overall avg = 0.1241)

###### (overall avg = 0.1279)

[Figure 81]

[Figure 82]

[Figure 83]

0.138 0.137 0.157 0.093

0.130 0.131 0.152 0.093

Generic Text

Stage1(TableModel)

0.137 0.125 0.142 0.090

0.128 0.128 0.143 0.090

Table-Text

0.149 0.129 0.136 0.091

0.133 0.129 0.135 0.091

Table-Struct.

0.157 0.134 0.149 0.095

0.140 0.132 0.145 0.095

Col.-Centric

Generic Text Table-Text Table-Struct. Col.-Centric

Generic Text Table-Text Table-Struct. Col.-Centric

Generic Text Table-Text Table-Struct. Col.-Centric

[Figure 84]

0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16

UJ-H

- Figure 11: DLTE category-level UJ-H heatmap (5-round average, test set). Panels correspond to

- Stage 3 (row model) families. Rows = Stage 1 (table model), columns = Stage 2 (column model). The column-driven gradient confirms Stage 2’s dominant effect. Near-identical panels show Stage 3 differences are largely masked end to end.

main-effect score mT(t)+mC(c)+mR(r)−2¯y is maximized by the per-stage rank-1 assembly. On test, this assembly is STARMIE/TABBIE/TRANSTAB and scores 0.134 UJ-H, while the test rank-1 pipeline STARMIE/GTE/GTE scores 0.253 and the dev-selected headline TUTA/GTE/GTE scores 0.229. On development, the marginal-leader assembly changes to STARMIE/BERT/TRANSTAB and is competitive on test (0.231), confirming that marginal main effects carry signal. The change in the leader assembly across splits, TABBIE’s absence from the top-50, and the gap between the marginal-leader assembly and the end-to-end optima together show that the top of the DLTE space is shaped by residual non-additive stage interactions. We refer to this residual structure as compositional fit.

###### N.4 Oracle-RA Row-Model Diagnostic

- Table 39 reports Oracle-RA results on the test split. Oracle-RA bypasses Stage 1 (retrieval) and Stage 2 (alignment) with ground-truth data, isolating Stage 3 row matching quality. The UJ-H spread across row models is 0.546, compared with a marginal end-to-end span of just 0.013 (Table 38), confirming that upstream errors mask most row-model differences in the full pipeline.

- Table 39: Oracle-RA row-model diagnostic (test set). Stages 1–2 use ground truth. Only Stage 3 row matching varies.

# Row Model UJ-H↑ Cell F1↑ Union↑ Join↑ Hard↑

- 1 GTE 0.683 0.802 0.696 0.763 0.550
- 2 TransTab 0.658 0.802 0.698 0.721 0.607
- 3 TabICL 0.606 0.789 0.609 0.664 0.737
- 4 TUTA 0.487 0.743 0.829 0.403 0.418
- 5 BERT 0.454 0.735 0.817 0.386 0.376
- 6 SCARF 0.340 0.709 0.864 0.246 0.318
- 7 DAE 0.333 0.713 0.893 0.234 0.305
- 8 VIME 0.318 0.710 0.900 0.223 0.289
- 9 TabPFN 0.293 0.706 0.925 0.196 0.256
- 10 TabBinning 0.259 0.697 0.902 0.181 0.232
- 11 TABBIE 0.231 0.695 0.935 0.154 0.197
- 12 SAINT 0.168 0.684 0.940 0.116 0.144
- 13 SubTab 0.164 0.685 0.957 0.109 0.133
- 14 TabTransformer 0.137 0.680 0.961 0.092 0.111

Key observations. Row models divide into two groups: identity-resolving models (GTE, TransTab, TabICL) that achieve balanced union/join recovery, and union-dedup specialists (TabTransformer, SAINT, SubTab) with near-perfect union recall but near-zero join recall. This distinction is reflected in the Cell F1 / UJ-H pipeline-level pattern: Cell F1 captures pooled cell-recovery yield (well-served by high union recall), while UJ-H captures balanced recovery of both removed blocks and therefore requires both recalls to be high (Appendix N.1). TabICL attains the best hard-region recall (0.737), showing that the second-pass join mechanism can recover cells in the new-rows×new-columns quadrant.

Per-noise-tier breakdown. Table 40 decomposes Oracle-RA UJ-H across the four TRL-DLTE noise tiers (cumulative clean → schema → cell → hard; see Sec. 3.4). The key observation is that the cross-row-model UJ-H span is large and largely tier-invariant (0.562/0.563/0.506/0.553 for clean/schema/cell/hard), and the top/bottom row-model families are stable at every corruption level: the identity-resolving row models (GTE, TRANSTAB, TABICL) occupy the top three positions in every tier, and the union-dedup specialists (TABTRANSFORMER, SAINT, SUBTAB) occupy the bottom three in every tier. The Stage 3 separability exposed by Oracle-RA is therefore not a noise-sensitivity artifact: it persists across all levels of upstream corruption. The fact that uniondedup specialists rank strongly on end-to-end Cell F1 (Table 43) but rank last under Oracle-RA on both UJ-H and join recall indicates that their end-to-end Cell F1 advantage reflects union-side recovery behavior under upstream error, not strong identity-resolution Stage 3 behavior on balanced enrichment.

Cross-validation against RBench robust linkage. The identity-resolving/union-dedup split identified by Oracle-RA is not an artifact of the DLTE pipeline: the same taxonomy is visible in atomic cross-table record linkage. Over all 14 row models, the Oracle-RA UJ-H ranking and the RBench Robust Linkage NR ranking (aggregated over DM-D and WDC; Table 3) are strongly correlated: |ρSpearman| = 0.80 (p = 6.3×10−4) and |τKendall| = 0.63 (p = 1.2×10−3). The top two row models agree across both views (GTE and TRANSTAB: Robust Linkage NR = 0.048 and 0.096, and Oracle-RA UJ-H also leads with these two), with TABICL third on Oracle-RA but fifth on Robust Linkage NR (= 0.394), trailing GTE and TRANSTAB as well as TUTA (= 0.154) and BERT (= 0.163). The union-dedup specialists highlighted by Oracle-RA anchor the other end: SUBTAB and TABTRANSFORMER occupy the bottom two positions of the Robust Linkage NR column (NR = 0.962, 0.942), with their near-zero join recall in Oracle-RA mirrored by near-zero

- Table 40: Oracle-RA per-noise-tier UJ-H for each row model (test split, 5-round mean). Tiers are the cumulative noise levels used in TRL-DLTE construction (clean → schema → cell → hard). The final row reports the cross-row-model span within each tier, summarizing how Stage 3 separability varies with upstream noise. Row models are sorted by mean UJ-H across tiers.

Row model Clean Schema Cell Hard GTE 0.703 0.704 0.633 0.692 TRANSTAB 0.652 0.675 0.612 0.691 TABICL 0.613 0.612 0.546 0.653 TUTA 0.502 0.499 0.449 0.499 BERT 0.480 0.480 0.431 0.425 SCARF 0.351 0.338 0.308 0.363 DAE 0.323 0.323 0.296 0.392 VIME 0.307 0.318 0.275 0.373 TABPFN 0.306 0.300 0.267 0.297 TABBINNING 0.256 0.256 0.240 0.283 TABBIE 0.236 0.229 0.207 0.255 SAINT 0.171 0.173 0.150 0.177 SUBTAB 0.170 0.172 0.151 0.164 TABTRANSFORMER 0.141 0.141 0.127 0.139 Span (max−min) 0.562 0.563 0.506 0.553

WDC F1 (e.g., TABTRANSFORMER WDC F1 = 0.020). SAINT sits at NR = 0.606, mid-pack on Robust Linkage despite its union-heavy profile. Rank orderings within each family differ for the same reason: DM-C (clean linkage) does not fully separate identity-resolution behavior, which is why BERT leads there but drops to 5th in Oracle-RA. Even so, the two tests agree at both ends of the ranking. This is consistent with a shared identity-resolution capability of frozen row embeddings, surfaced consistently by both entity matching (RBench) and compositional enrichment (DLTE).

N.5 Full Per-Stage Model Rankings

Tables 41–43 report the complete marginal rankings for each DLTE stage (test split, 5-round mean ± std) over the full 10×8×14 = 1120 table × column × row search space. The 10-model Stage 1 pool covers the 8 column-capable encoders (whose column embeddings are pooled to a table embedding) plus the two native table encoders TAPEX and TUTA. Stage 2 remains the 8 column-capable encoders since TAPEX/TUTA do not expose compatible column embeddings. For a fixed table model, scores are averaged over all 8 × 14 = 112 compatible pipelines, for a fixed column model over all 10 × 14 = 140 pipelines, and for a fixed row model over all 10 × 8 = 80 pipelines.

- Table 41: Stage 1 (table model) marginal rankings, sorted by Cell F1 (mean ± std over 5 rounds). R@100 is Stage 1 retrieval recall against any gold candidate.

Rank Model (Family) Cell F1 ↑ UJ-H ↑ R@100 ↑

- 1 Starmie (Col.-Centric) 0.601±0.001 0.144±0.003 0.740±0.000
- 2 TUTA (Table-Struct.) 0.593±0.001 0.138±0.001 0.585±0.000
- 3 GTE (Generic Text) 0.591±0.002 0.129±0.001 0.801±0.000
- 4 TaBERT (Table-Text) 0.590±0.001 0.124±0.001 0.720±0.000
- 5 BERT (Generic Text) 0.589±0.002 0.128±0.002 0.763±0.000
- 6 TAPAS (Table-Text) 0.587±0.001 0.118±0.001 0.615±0.000
- 7 TURL (Table-Struct.) 0.587±0.000 0.124±0.001 0.597±0.007
- 8 TabSketchFM (Col.-Centric) 0.584±0.001 0.116±0.002 0.413±0.001
- 9 TABBIE (Table-Struct.) 0.560±0.000 0.109±0.001 0.108±0.000
- 10 TAPEX (Table-Text) 0.558±0.001 0.127±0.001 0.247±0.000

Retrieval recall does not dictate downstream contribution. Table 41 exposes a non-monotone relationship between raw Stage 1 retrieval quality and downstream end-to-end contribution. GTE attains the highest recall-at-100 (0.801) among the ten table models but does not lead downstream. STARMIE has lower recall-at-100 (0.740) yet attains the best downstream Cell F1 (0.601) and UJ-H (0.144). The ordering is neither retrieval-dominated nor inverse: TABBIE is worst on recall-at-100 (0.108) while TAPEX is lowest on Cell F1 (0.558), and TABERT and BERT sit in the middle of

- Table 42: Stage 2 (column model) marginal rankings, sorted by Cell F1 (mean ± std over 5 rounds). Rank Model (Family) Cell F1 ↑ UJ-H ↑

- 1 TaBERT (Table-Text) 0.628±0.001 0.128±0.000
- 2 GTE (Generic Text) 0.601±0.001 0.141±0.002
- 3 TAPAS (Table-Text) 0.600±0.001 0.132±0.001
- 4 TURL (Table-Struct.) 0.597±0.004 0.143±0.004
- 5 BERT (Generic Text) 0.595±0.000 0.135±0.001
- 6 TABBIE (Table-Struct.) 0.554±0.001 0.143±0.000
- 7 TabSketchFM (Col.-Centric) 0.553±0.001 0.100±0.001
- 8 Starmie (Col.-Centric) 0.544±0.000 0.084±0.000

- Table 43: Stage 3 (row model) marginal rankings, sorted by Cell F1 (mean ± std over 5 rounds). Rank Model (Family) Cell F1 ↑ UJ-H ↑

- 1 TabTransformer (Target-Table) 0.591±0.001 0.119±0.001
- 2 SubTab (Target-Table) 0.591±0.001 0.121±0.001
- 3 SAINT (Target-Table) 0.590±0.002 0.119±0.001
- 4 TABBIE (Table-Struct.) 0.589±0.000 0.122±0.000
- 5 TabPFN (Prior-Based) 0.589±0.000 0.126±0.000
- 6 DAE (Target-Table) 0.587±0.000 0.127±0.001
- 7 TabBinning (Target-Table) 0.587±0.003 0.123±0.003
- 8 VIME (Target-Table) 0.587±0.001 0.125±0.001
- 9 SCARF (Target-Table) 0.585±0.001 0.127±0.001
- 10 TUTA (Table-Struct.) 0.584±0.000 0.130±0.001
- 11 BERT (Generic Text) 0.581±0.000 0.128±0.001
- 12 TransTab (Target-Table) 0.574±0.001 0.132±0.001
- 13 GTE (Generic Text) 0.574±0.000 0.131±0.001
- 14 TabICL (Prior-Based) 0.566±0.000 0.130±0.001

both orderings. This non-monotone pattern shows that Stage 1’s downstream contribution is not determined by recall volume alone. It also depends on the structure of the retrieved candidate set and on whether those candidates are usable by the downstream column-alignment and row-matching models. This is a Stage-1 instance of compositional fit beyond per-stage marginal rank.

###### N.6 Stage 3 Row Matching: Full Pipeline Rankings

- Table 44: Ablation: DLTE Stage 3 (row matching). Rows are (Stage 1, Stage 2) configurations. Columns are Stage 3 row models. All ten Stage 1 models are included (8 column-capable + TAPEX

and TUTA native table encoders). Cell F1 (↑) on test set. Column groupings by adaptation regime: Transfer = externally pretrained, used frozen. Prior = meta-pretrained prior-fitted. Learner = target-table feature-corruption SSL.

Transfer Prior Learner

TabBinning SAINT SubTab TabTransf.

BERT GTE TABBIE TUTA TabICL TabPFN TransTab VIME SCARF DAE

Avg.

Stage 1 Stage 2

Starmie TaBERT 0.638 0.607 0.671 0.643 0.589 0.670 0.609 0.662 0.654 0.661 0.663 0.674 0.677 0.679 0.650 GTE 0.637 0.632 0.639 0.639 0.614 0.638 0.634 0.636 0.633 0.636 0.635 0.638 0.639 0.639 0.635 TAPAS 0.624 0.608 0.641 0.631 0.592 0.641 0.608 0.637 0.631 0.636 0.636 0.643 0.645 0.645 0.630 TURL 0.616 0.612 0.614 0.616 0.601 0.616 0.613 0.613 0.612 0.615 0.612 0.612 0.615 0.615 0.613 BERT 0.609 0.606 0.610 0.613 0.595 0.610 0.610 0.606 0.607 0.608 0.606 0.609 0.609 0.609 0.608 TabSketchFM 0.565 0.562 0.567 0.566 0.561 0.567 0.561 0.564 0.565 0.564 0.564 0.565 0.566 0.566 0.564 TABBIE 0.559 0.550 0.570 0.563 0.543 0.569 0.550 0.567 0.565 0.567 0.568 0.572 0.573 0.573 0.564 Starmie 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544

TAPAS TaBERT 0.637 0.610 0.669 0.644 0.593 0.668 0.613 0.662 0.654 0.661 0.662 0.672 0.675 0.676 0.650 GTE 0.597 0.589 0.609 0.600 0.576 0.607 0.587 0.605 0.602 0.606 0.604 0.608 0.609 0.609 0.601 BERT 0.595 0.584 0.607 0.598 0.573 0.606 0.585 0.605 0.601 0.604 0.604 0.608 0.608 0.608 0.599 TAPAS 0.593 0.582 0.606 0.596 0.572 0.607 0.583 0.603 0.599 0.604 0.603 0.608 0.609 0.609 0.598 TURL 0.592 0.582 0.602 0.595 0.570 0.602 0.583 0.599 0.596 0.600 0.599 0.603 0.605 0.605 0.595 TABBIE 0.552 0.546 0.563 0.557 0.542 0.562 0.548 0.560 0.558 0.560 0.560 0.563 0.564 0.565 0.557 TabSketchFM 0.550 0.551 0.554 0.553 0.549 0.553 0.550 0.552 0.553 0.553 0.553 0.553 0.553 0.553 0.552 Starmie 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544

GTE TaBERT 0.636 0.607 0.669 0.643 0.589 0.668 0.611 0.662 0.653 0.660 0.662 0.672 0.676 0.677 0.649 TURL 0.604 0.607 0.603 0.607 0.593 0.605 0.608 0.604 0.603 0.604 0.603 0.603 0.604 0.604 0.604 TAPAS 0.599 0.594 0.608 0.603 0.580 0.611 0.594 0.607 0.603 0.607 0.606 0.610 0.611 0.612 0.603 GTE 0.601 0.592 0.608 0.601 0.581 0.607 0.591 0.605 0.604 0.606 0.605 0.608 0.610 0.609 0.602 BERT 0.600 0.589 0.606 0.602 0.575 0.606 0.591 0.601 0.600 0.602 0.602 0.607 0.607 0.608 0.600 TABBIE 0.565 0.557 0.577 0.569 0.552 0.576 0.558 0.574 0.572 0.573 0.575 0.577 0.579 0.579 0.570 TabSketchFM 0.553 0.552 0.556 0.555 0.552 0.555 0.552 0.554 0.555 0.555 0.554 0.554 0.555 0.555 0.554 Starmie 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544

TaBERT TaBERT 0.636 0.610 0.669 0.644 0.592 0.667 0.612 0.661 0.653 0.660 0.662 0.671 0.674 0.676 0.649 TAPAS 0.605 0.593 0.622 0.609 0.580 0.623 0.594 0.619 0.613 0.619 0.618 0.624 0.626 0.626 0.612 TURL 0.600 0.592 0.611 0.605 0.579 0.611 0.593 0.608 0.605 0.609 0.608 0.611 0.613 0.614 0.604 GTE 0.600 0.594 0.611 0.602 0.582 0.608 0.592 0.607 0.605 0.609 0.607 0.610 0.611 0.611 0.604 BERT 0.593 0.577 0.604 0.595 0.565 0.604 0.579 0.600 0.598 0.600 0.600 0.606 0.606 0.608 0.595 TABBIE 0.554 0.546 0.565 0.557 0.541 0.564 0.547 0.562 0.560 0.562 0.563 0.566 0.567 0.567 0.559 TabSketchFM 0.550 0.550 0.554 0.553 0.549 0.553 0.550 0.552 0.553 0.553 0.553 0.553 0.554 0.554 0.552 Starmie 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544

BERT TaBERT 0.634 0.606 0.666 0.641 0.587 0.664 0.608 0.658 0.650 0.657 0.659 0.669 0.672 0.673 0.646 TAPAS 0.599 0.594 0.610 0.606 0.581 0.612 0.594 0.608 0.604 0.609 0.607 0.611 0.612 0.612 0.604 TURL 0.603 0.603 0.602 0.606 0.591 0.603 0.605 0.602 0.601 0.602 0.601 0.602 0.604 0.603 0.602 GTE 0.599 0.589 0.607 0.599 0.578 0.605 0.588 0.604 0.602 0.605 0.604 0.606 0.608 0.608 0.600 BERT 0.599 0.586 0.605 0.599 0.574 0.605 0.588 0.602 0.600 0.602 0.602 0.607 0.607 0.607 0.599 TABBIE 0.557 0.548 0.568 0.561 0.544 0.567 0.551 0.565 0.563 0.565 0.565 0.569 0.570 0.570 0.562 TabSketchFM 0.554 0.555 0.556 0.556 0.553 0.555 0.554 0.554 0.555 0.555 0.554 0.555 0.555 0.554 0.555 Starmie 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544

TURL TaBERT 0.632 0.608 0.661 0.638 0.590 0.660 0.610 0.654 0.646 0.654 0.654 0.663 0.666 0.667 0.643 GTE 0.607 0.607 0.605 0.607 0.591 0.605 0.603 0.604 0.603 0.603 0.603 0.604 0.605 0.605 0.604 TAPAS 0.591 0.578 0.606 0.594 0.566 0.606 0.579 0.603 0.599 0.604 0.603 0.609 0.610 0.610 0.597 BERT 0.592 0.575 0.602 0.592 0.564 0.601 0.576 0.599 0.596 0.599 0.599 0.605 0.606 0.606 0.594 TURL 0.588 0.578 0.597 0.589 0.569 0.597 0.578 0.595 0.593 0.596 0.595 0.597 0.599 0.599 0.591 TABBIE 0.561 0.553 0.573 0.565 0.548 0.572 0.555 0.570 0.568 0.570 0.570 0.573 0.575 0.575 0.566 TabSketchFM 0.555 0.553 0.558 0.556 0.550 0.557 0.552 0.556 0.556 0.556 0.556 0.557 0.558 0.557 0.556 Starmie 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544

TabSketchFM TaBERT 0.618 0.598 0.645 0.626 0.583 0.644 0.600 0.638 0.634 0.639 0.640 0.647 0.649 0.650 0.629 BERT 0.606 0.590 0.623 0.610 0.575 0.623 0.590 0.619 0.615 0.619 0.620 0.624 0.626 0.627 0.612 TAPAS 0.595 0.582 0.611 0.598 0.571 0.611 0.582 0.606 0.602 0.608 0.607 0.613 0.614 0.615 0.601 TURL 0.589 0.580 0.599 0.591 0.570 0.599 0.580 0.597 0.594 0.598 0.597 0.600 0.602 0.602 0.593 GTE 0.587 0.578 0.598 0.591 0.568 0.597 0.578 0.595 0.593 0.596 0.596 0.598 0.601 0.601 0.591 TabSketchFM 0.550 0.549 0.553 0.552 0.548 0.552 0.549 0.551 0.551 0.551 0.552 0.552 0.552 0.552 0.551 TABBIE 0.543 0.540 0.554 0.549 0.537 0.554 0.542 0.552 0.550 0.552 0.553 0.555 0.556 0.556 0.549 Starmie 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544

TUTA TaBERT 0.619 0.598 0.645 0.624 0.584 0.643 0.600 0.639 0.633 0.638 0.639 0.646 0.649 0.650 0.629 GTE 0.625 0.621 0.627 0.628 0.603 0.627 0.621 0.622 0.624 0.625 0.624 0.627 0.628 0.628 0.624 TAPAS 0.616 0.604 0.626 0.619 0.591 0.626 0.605 0.622 0.618 0.621 0.621 0.626 0.629 0.629 0.618 TURL 0.614 0.611 0.614 0.618 0.598 0.616 0.609 0.613 0.613 0.615 0.613 0.613 0.616 0.615 0.613 BERT 0.604 0.603 0.600 0.610 0.589 0.602 0.600 0.599 0.600 0.601 0.600 0.602 0.602 0.602 0.601 TABBIE 0.555 0.548 0.566 0.559 0.543 0.565 0.550 0.562 0.561 0.563 0.564 0.566 0.568 0.567 0.560 TabSketchFM 0.553 0.552 0.555 0.554 0.550 0.554 0.551 0.553 0.554 0.554 0.554 0.554 0.554 0.554 0.553 Starmie 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544

TAPEX GTE 0.589 0.599 0.575 0.589 0.587 0.577 0.595 0.574 0.578 0.578 0.574 0.574 0.574 0.573 0.581 TURL 0.581 0.574 0.583 0.582 0.568 0.584 0.577 0.583 0.582 0.583 0.583 0.583 0.584 0.584 0.581 BERT 0.571 0.563 0.573 0.573 0.555 0.573 0.565 0.571 0.570 0.571 0.571 0.574 0.574 0.575 0.570 TaBERT 0.567 0.558 0.573 0.570 0.556 0.573 0.561 0.571 0.571 0.572 0.572 0.574 0.574 0.575 0.569 TAPAS 0.565 0.561 0.565 0.565 0.557 0.565 0.562 0.563 0.563 0.564 0.563 0.564 0.565 0.565 0.563 TabSketchFM 0.546 0.546 0.547 0.547 0.546 0.547 0.545 0.546 0.547 0.546 0.546 0.546 0.547 0.547 0.546 Starmie 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 TABBIE 0.508 0.511 0.511 0.512 0.516 0.512 0.514 0.511 0.511 0.511 0.512 0.512 0.512 0.512 0.512

TABBIE BERT 0.574 0.559 0.584 0.574 0.549 0.580 0.559 0.580 0.576 0.579 0.579 0.584 0.585 0.586 0.575 TURL 0.571 0.562 0.577 0.571 0.557 0.577 0.563 0.576 0.574 0.576 0.576 0.577 0.579 0.579 0.572 TaBERT 0.564 0.557 0.574 0.567 0.552 0.573 0.559 0.571 0.570 0.572 0.572 0.574 0.575 0.575 0.568 TAPAS 0.565 0.561 0.573 0.566 0.556 0.572 0.562 0.570 0.569 0.571 0.571 0.573 0.573 0.573 0.568 GTE 0.566 0.560 0.570 0.567 0.558 0.569 0.562 0.568 0.567 0.568 0.568 0.570 0.571 0.571 0.567 TabSketchFM 0.551 0.546 0.550 0.551 0.543 0.549 0.543 0.548 0.548 0.549 0.549 0.550 0.550 0.550 0.548 Starmie 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 0.544 TABBIE 0.533 0.532 0.540 0.537 0.533 0.540 0.534 0.539 0.538 0.538 0.539 0.540 0.541 0.541 0.537

Global Avg. 0.580 0.573 0.588 0.583 0.566 0.588 0.575 0.587 0.585 0.587 0.587 0.590 0.591 0.591 0.584

- Table 44 shows Cell F1 for all (Stage 1, Stage 2) configurations across all ten Stage 1 models, crossed with all 14 Stage 3 row models. The best pipelines use TaBERT at Stage 2 (top rows of each group), confirming Stage 2 dominance regardless of Stage 1 choice. Starmie leads as Stage 1 (best avg 0.650),

while TAPEX has the lowest average Cell F1 as Stage 1 (0.558), narrowly below TABBIE (0.560), with both lagging the rest of the pool by a clear margin. Within any given (Stage 1, Stage 2) pair,

the Stage 3 spread is narrow: for the best configuration (Starmie → TaBERT), Cell F1 ranges from 0.589–0.679, a span of only ∼0.090. The global average Cell F1 by Stage 3 column is nearly flat (0.566–0.591), with SAINT, SubTab, and TabTransf. consistently at the top and TabICL at the bottom, confirming that Stage 3 row model choice has only marginal end-to-end impact and that, within this Cell F1 ablation, the dominant lever is the Stage 2 column alignment model.

###### N.7 Top-20 Pipeline Combinations

Tables 45 and 46 list the top-20 pipeline combinations by Cell F1 and UJ-H, respectively. The two lists rank pipelines along different axes, consistent with their definitions (Appendix N.1): Cell F1 captures pooled cell-recovery yield, while UJ-H captures balanced recovery of both removed blocks.

- Table 45: Top-20 DLTE combinations by Cell F1 (5-round average, test set). All entries use TaBERT

- as Stage 2 column model. Their UJ-H ranks (#513–#671) reflect the different axis Cell F1 captures relative to UJ-H (Appendix N.1).

# Stage 1 (Tbl) Fam. Stage 2 (Col) Fam. Stage 3 (Row) Fam. Cell F1 UJ-H UJ-H #

- 1 Starmie Col-Cen TaBERT Tbl-Txt TabTransf. Tgt-Tbl 0.679 0.128 #532
- 2 Starmie Col-Cen TaBERT Tbl-Txt SubTab Tgt-Tbl 0.677 0.128 #523
- 3 GTE Gen. Txt TaBERT Tbl-Txt TabTransf. Tgt-Tbl 0.677 0.125 #589
- 4 TAPAS Tbl-Txt TaBERT Tbl-Txt TabTransf. Tgt-Tbl 0.676 0.127 #544
- 5 GTE Gen. Txt TaBERT Tbl-Txt SubTab Tgt-Tbl 0.676 0.125 #597
- 6 TaBERT Tbl-Txt TaBERT Tbl-Txt TabTransf. Tgt-Tbl 0.676 0.125 #608
- 7 TAPAS Tbl-Txt TaBERT Tbl-Txt SubTab Tgt-Tbl 0.675 0.127 #549
- 8 Starmie Col-Cen TaBERT Tbl-Txt SAINT Tgt-Tbl 0.674 0.126 #573
- 9 TaBERT Tbl-Txt TaBERT Tbl-Txt SubTab Tgt-Tbl 0.674 0.125 #616
- 10 BERT Gen. Txt TaBERT Tbl-Txt TabTransf. Tgt-Tbl 0.673 0.126 #575
- 11 GTE Gen. Txt TaBERT Tbl-Txt SAINT Tgt-Tbl 0.672 0.123 #656
- 12 BERT Gen. Txt TaBERT Tbl-Txt SubTab Tgt-Tbl 0.672 0.126 #577
- 13 TAPAS Tbl-Txt TaBERT Tbl-Txt SAINT Tgt-Tbl 0.672 0.126 #581
- 14 Starmie Col-Cen TaBERT Tbl-Txt TABBIE Transfer 0.671 0.127 #541
- 15 TaBERT Tbl-Txt TaBERT Tbl-Txt SAINT Tgt-Tbl 0.671 0.123 #671
- 16 Starmie Col-Cen TaBERT Tbl-Txt TabPFN Prior 0.670 0.129 #513
- 17 GTE Gen. Txt TaBERT Tbl-Txt TABBIE Transfer 0.669 0.124 #651
- 18 TAPAS Tbl-Txt TaBERT Tbl-Txt TABBIE Transfer 0.669 0.125 #596
- 19 TaBERT Tbl-Txt TaBERT Tbl-Txt TABBIE Transfer 0.669 0.123 #659
- 20 BERT Gen. Txt TaBERT Tbl-Txt SAINT Tgt-Tbl 0.669 0.124 #620

- Table 46: Top-20 DLTE combinations by UJ-H (5-round average, test set). Stage 2 is dominated by GTE, BERT, and TURL. Stage 1 is led by Starmie (11 entries), with TUTA appearing 7 times (from the native table-encoder pool), TURL once (#17), and TAPEX once (#19). Zero overlap with the Cell F1 top-20 list (Table 45).

# Stage 1 (Tbl) Fam. Stage 2 (Col) Fam. Stage 3 (Row) Fam. Cell F1 UJ-H CF1 #

- 1 Starmie Col-Cen GTE Gen. Txt GTE Transfer 0.632 0.253 #114
- 2 Starmie Col-Cen GTE Gen. Txt TransTab Tgt-Tbl 0.634 0.251 #108
- 3 Starmie Col-Cen GTE Gen. Txt TabICL Prior 0.614 0.236 #184
- 4 Starmie Col-Cen BERT Gen. Txt TransTab Tgt-Tbl 0.610 0.231 #226
- 5 TUTA Tbl-Struct GTE Gen. Txt GTE Transfer 0.621 0.229 #149
- 6 Starmie Col-Cen BERT Gen. Txt GTE Transfer 0.606 0.228 #313
- 7 Starmie Col-Cen BERT Gen. Txt TabICL Prior 0.595 0.228 #504
- 8 Starmie Col-Cen TURL Tbl-Struct GTE Transfer 0.612 0.225 #201
- 9 Starmie Col-Cen TURL Tbl-Struct TransTab Tgt-Tbl 0.613 0.225 #186
- 10 Starmie Col-Cen TURL Tbl-Struct TabICL Prior 0.601 0.220 #429
- 11 TUTA Tbl-Struct GTE Gen. Txt TransTab Tgt-Tbl 0.621 0.218 #153
- 12 TUTA Tbl-Struct BERT Gen. Txt GTE Transfer 0.603 0.216 #380
- 13 Starmie Col-Cen GTE Gen. Txt TUTA Transfer 0.639 0.215 #85
- 14 Starmie Col-Cen GTE Gen. Txt BERT Transfer 0.637 0.211 #97
- 15 TUTA Tbl-Struct BERT Gen. Txt TransTab Tgt-Tbl 0.600 0.210 #437
- 16 TUTA Tbl-Struct GTE Gen. Txt TabICL Prior 0.603 0.210 #386
- 17 TURL Tbl-Struct GTE Gen. Txt GTE Transfer 0.607 0.205 #284
- 18 TUTA Tbl-Struct GTE Gen. Txt TUTA Transfer 0.628 0.204 #120
- 19 TAPEX Tbl-Txt GTE Gen. Txt GTE Transfer 0.599 0.204 #461
- 20 TUTA Tbl-Struct BERT Gen. Txt TabICL Prior 0.589 0.204 #545

###### N.8 Source Split: TabFact vs. WTQ

To test whether the TRL-DLTE findings are tied to a particular parent source, we partition the 345 test parents into TabFact (246) and WTQ (99) and recompute every pipeline-level metric separately on each partition. All 1,120 canonical pipelines (10 Stage-1 × 8 Stage-2 × 14 Stage-3 models) are evaluated under the same 5 rounds, so the source-split numbers are directly comparable to the main-text aggregates.

- Table 47: Source-split UJ-H for headline DLTE pipelines (5-round mean over the test split). The hybrid-vs-monolith gap holds separately on TabFact (246 test parents) and WTQ (99 test parents), and is wider on WTQ. Dev-selected pipelines are the rank-1 hybrid and rank-1 monolithic by dev UJ-H. The test-set unconstrained block is the rank-1 across all 1,120 pipelines on test, included as a sensitivity reference. Pipeline notation: Stage 1 (Tbl) / Stage 2 (Col) / Stage 3 (Row).

Pipeline TabFact WTQ All Dev-selected (top-1 by dev UJ-H, evaluated on test):

Best hybrid: TUTA/GTE/GTE 0.225 0.238 0.229 Best monolith: BERT/BERT/BERT 0.139 0.140 0.139 Hybrid − monolith +0.086 +0.099 +0.090

Test-set unconstrained reference (max over 1,120 pipelines):

Best hybrid: STARMIE/GTE/GTE 0.242 0.281 0.253 Best monolith: GTE/GTE/GTE 0.140 0.157 0.145 Hybrid − monolith +0.102 +0.125 +0.108

- Table 48: Per-stage marginal UJ-H top-5 under each source, averaged over all 1,120 canonical pipelines. Stage 1 is STARMIE-led in both sources. Stage 2 has a TABBIE ↔ TURL swap at rank 1 but the same top models recur. Stage 3 is TRANSTAB-led in both.

Stage Source Rank 1 Rank 2 Rank 3 Rank 4 Rank 5

- Stage 1 (Tbl)

TabFact STARMIE (0.140) TUTA (0.137) GTE (0.125) BERT (0.124) TAPEX (0.122) WTQ STARMIE (0.155) TUTA (0.140) TAPEX (0.138) GTE (0.138) TURL (0.137) All STARMIE (0.144) TUTA (0.138) GTE (0.129) BERT (0.128) TAPEX (0.127)

- Stage 2 (Col)

TabFact TABBIE (0.140) GTE (0.137) TURL (0.137) BERT (0.132) TAPAS (0.132) WTQ TURL (0.159) TABBIE (0.153) GTE (0.150) TABERT (0.149) BERT (0.140) All TABBIE (0.143) TURL (0.143) GTE (0.141) BERT (0.135) TAPAS (0.132)

- Stage 3 (Row)

TabFact TRANSTAB (0.128) GTE (0.127) TABICL (0.127) TUTA (0.126) BERT (0.123) WTQ TRANSTAB (0.142) TUTA (0.140) GTE (0.140) TABICL (0.139) BERT (0.138) All TRANSTAB (0.132) GTE (0.131) TABICL (0.130) TUTA (0.130) BERT (0.128)

Headline pipelines hold in both sources. Table 47 shows that the hybrid-vs-monolith gap persists separately on TabFact and WTQ: the dev-selected best hybrid (TUTA/GTE/GTE) beats the devselected best monolith (BERT/BERT/BERT) by +0.086 UJ-H on TabFact and by +0.099 on WTQ. The gap is if anything wider on WTQ, so the hybrid advantage is not a TabFact-specific artifact. The unconstrained test maximum (STARMIE/GTE/GTE, dev rank 8) reaches 0.242 / 0.281 on TabFact / WTQ with the same wider-on-WTQ pattern. The unconstrained test-set max monolith shifts to GTE/GTE/GTE (0.140 / 0.157, pooled 0.145), still well below every hybrid pipeline reported here.

Per-stage marginal top models are largely source-invariant. Table 48 shows the top-5 models at each stage under each source. Stage 1 is STARMIE-led in both sources, with TUTA second. Stage 2 has a TABBIE↔TURL swap at rank 1, and four of the top-5 column models (TABBIE, TURL, GTE, BERT) are common to both sources. Stage 3 is TRANSTAB-led in both, with GTE, TABICL, and TUTA occupying ranks 2–4 in both sources (in varying order). No stage changes leader family under the source split.

Strong rank agreement across sources. Across all 1,120 canonical pipelines, Spearman ρ(TabFact,WTQ) = 0.871 (p ≪ 10−100), confirming that pipeline rankings are largely sourceagnostic. For Oracle-RA, which isolates Stage 3 identity resolution by replacing Stages 1–2 with ground truth (Table 39), the row-model ranking agrees at Spearman ρ = 0.987 (p = 7.4 × 10−11,

n = 14): GTE, TRANSTAB, TABICL occupy the top three positions in both sources, and TABTRANSFORMER, SUBTAB, SAINT the bottom three in both. Table 49 reports the full per-source Oracle-RA UJ-H for every row model. Combined with the per-noise-tier breakdown in Table 40, the Stage-3 row-model ranking is stable on both the noise-tier axis (cross-tier spans 0.506–0.563) and the parent-source axis (cross-source spans 0.533–0.551), so the identity-resolving/union-dedup split exposed at Stage 3 does not depend on either the particular noise regime or the particular parent source.

- Table 49: Oracle-RA per-source UJ-H for each row model (test split, 5-round mean). Sources are the two TRL-DLTE parent pools (TabFact: 246 test parents, WTQ: 99 test parents). The “All” column is the pooled mean over the 345 test parents × 4 noise tiers (1,380 query-tier evaluations). The final row reports the cross-row-model span within each source, comparable in magnitude to the per-tier spans (0.506–0.563) in Table 40. Row models are sorted by pooled UJ-H.

Row model TabFact WTQ All

GTE 0.684 0.680 0.683 TRANSTAB 0.677 0.609 0.658 TABICL 0.615 0.585 0.606 TUTA 0.499 0.457 0.487 BERT 0.467 0.421 0.454 SCARF 0.361 0.288 0.340 DAE 0.346 0.302 0.333 VIME 0.336 0.275 0.318 TABPFN 0.304 0.265 0.293 TABBINNING 0.252 0.274 0.259 TABBIE 0.237 0.219 0.231 SAINT 0.161 0.185 0.168 SUBTAB 0.166 0.161 0.164 TABTRANSFORMER 0.133 0.147 0.137

Span (max−min) 0.551 0.533 0.546

- Table 50: Proprietary embedding ablation across column/table and row tasks. All three OpenAI variants dominate matching tasks (TblRet ranks 1–3, RecLink ranks 1–3) but are mid-pack on structural grounding (TblQA ranks 5–7) and behind the task-adaptive model on row prediction

(RowPred ranks 2, 3, 5). Row metrics average MLP and linear probes. RecLink reports binary F1 (match class), averaged unweighted over all 16 linkage datasets (the main table reports the same metric per group). Values are mean ± std over 5 seeds. Bold = best, underline = second best. Dashes indicate the model does not support the required embedding granularity.

Column / Table Row Type Model TblRet

TblQA

RowPred

RecLink

MRR ↑

Acc ↑

AUROC ↑

F1 ↑

###### BERT 0.367±0.008 0.255±0.004 0.791±0.000 0.384±0.002 GTE 0.476±0.003 0.245±0.002 0.770±0.000 0.403±0.003

Generic Text

TaBERT 0.372±0.013 0.267±0.005 — TAPAS 0.295±0.006 0.254±0.003 — TAPEX 0.376±0.097 — — —

Table-Text

###### TABBIE 0.170±0.004 0.276±0.004 0.770±0.001 0.300±0.005 TURL 0.199±0.010 0.277±0.005 — TUTA 0.260±0.013 — 0.720±0.000 0.358±0.005

Table-Struct.

Starmie 0.018±0.002 0.266±0.005 — TabSketchFM 0.218±0.011 0.235±0.005 — —

Col.-Centric

Meta-Pretr. TabICL — — 0.816±0.001 0.274±0.005 Tgt-Tbl TransTab — — 0.778±0.001 0.375±0.017

TE3-Small 0.490±0.007 0.260±0.005 0.801±0.000 0.412±0.004 TE3-Large 0.511±0.006 0.265±0.003 0.797±0.001 0.426±0.004 Ada-002 0.540±0.007 0.266±0.004 0.789±0.001 0.423±0.002

Proprietary

##### O Proprietary Embedding Ablation: Retrieval vs. Structural Grounding

We evaluate three OpenAI embedding variants [55, 57]: TE3-SMALL (text-embedding-3-small, 768d), TE3-LARGE (text-embedding-3-large, 768-d), and ADA-002 (text-embedding-ada-002, 1536-d). These cover four representative tasks spanning column/table and row levels: Table Retrieval, Table QA, Row Prediction, and Record Linkage. TE3-Small and TE3-Large are requested at 768-d via OpenAI’s dimensions API parameter for dimensional parity with the 768-d open-source encoders in our pool. Ada-002 does not expose this option and retains its native 1536-d. Table 50 reports the results alongside the open-source models from the main tables.

Column/table level. All three proprietary variants lead Table Retrieval, occupying ranks 1–3 (ADA-002: 0.540 MRR, vs. GTE: 0.476). This is consistent with their training objective: large-scale semantic retrieval over diverse text, which transfers directly to table-to-query matching. On Table QA, however, the same models rank only 5–7 out of 11, behind TURL (0.277), TABBIE (0.276), and STARMIE (0.266), models whose pretraining encodes table structure. The contrast within a single task family shows that retrieval quality and structural understanding are distinct capabilities.

Row level. The pattern extends to rows. On Record Linkage, an entity-matching task driven by text similarity, the proprietary models again occupy the top three ranks (TE3-LARGE: 0.426 F1, vs. GTE: 0.403). On Row Prediction, however, TE3-SMALL reaches 0.801 AUROC (rank 2) but trails TABICL (0.816), a meta-pretrained model whose in-context conditioning adapts to each task. Across both granularities, proprietary embeddings lead matching tasks (retrieval, linkage) but do not displace specialized models on understanding tasks (QA, prediction), reinforcing that no single model family, open or proprietary, is universally dominant.

Scope and model selection. The general benchmark scope rationale is discussed in Appendix B. The proprietary OpenAI ablation above should therefore be read as a controlled scaling study within the embedding-only regime, not as a proxy for evaluating 7B–70B generative systems.

##### P Robustness

This section reports three auxiliary stability diagnostics introduced by Observatory [15], namely Sample Fidelity, Perturbation Robustness, and Row/Column Order Insignificance, applied directly to the models in our pool. The diagnostic definitions, evaluation datasets, and headline metrics all follow the Observatory originals. These diagnostics are not part of our contribution and are not counted toward the 16 benchmark tasks. All numerical results below are recomputed from our finalized embeddings.

All headline statistics use a unified table-first aggregation protocol. For Sample Fidelity and Row/Column Order Insignificance, we compute two canonical metrics, table_cosine_similarity and table_mcv, and then report the dataset-level mean and standard deviation over valid tables. For Sample Fidelity, each table cell below follows the format cosine mean ± std / MCV mean ± std. For Row/Column Order Insignificance, cosine similarity and MCV are reported in separate tables. For Perturbation Robustness, following the original Observatory protocol, we report changed-only cosine similarity only. Cosine similarity is the primary metric for cross-task comparison.

- P.1 Unified Aggregation Protocol Let T(0) be the original table and {T(1),...,T(K)} be its transformed variants. For a table item i (a

column, a row, or the whole table depending on the task) with embedding zi(k) under variant k, we compute

1 |Vi| k∈V

cos zi(0),zi(k) , (13)

si =

i

ci = MCV {zi(0)} ∪ {zi(k) | k ∈ Vi} , (14) where Vi is the set of valid transformed variants for item i. The table-level metrics are then

m

1 m

table_cosine =

si, (15)

i=1

m

1 m

table_mcv =

ci, (16)

i=1

with m being the number of valid items in the table. Finally, the appendix tables report the mean and standard deviation of these table-level metrics over the evaluation set. MCV is computed with the Observatory multivariate coefficient of variation implementation µ⊤Σµ/(µ⊤µ)2.

For Perturbation Robustness, we use a strict changed-only protocol: only columns that were actually modified by a perturbation are included in the headline cosine metric; tables without changed columns are excluded from the dataset-level aggregate.

###### P.2 Task and Dataset Summary

Table 51: Overview of the three appendix diagnostics.

Diagnostic Core question Data and scale Embeddings and model coverage

wiki_tables; 4,964 base tables; sampling ratios 0.25, 0.50, and 0.75.

Sample Fidelity Whether a table representation remains stable after observing only a subset of rows.

Column embeddings; the 8 column-capable models from our pool (shared across all three robustness diagnostics): bert, gte, starmie, tabbie, tabert, tabsketchfm, tapas, turl.

Perturbation Robustness Whether column representations remain stable under schema and content perturbations.

Database tables; 80 base tables; three perturbation families: DB_schema_synonym, DB_schema_abbreviation, and DB_DBcontent_equivalence.

Column embeddings; the same 8 column-capable models as above.

Row/Column Order Insignificance Whether row, column, and table representations are invariant to row-order and column-order permutations.

Column, row, and table embeddings; the 8 column-capable models above, 14 row-capable models, and 10 table-capable models from our pool.

wiki_tables; 4,964 base tables; each table has either 11 or 6 shuffle variants depending on how many unique permutations are available.

###### P.3 Sample Fidelity

The Sample Fidelity diagnostic measures whether the semantic representation of a table remains stable when only a subset of rows is observed. Following the Observatory Sample Fidelity protocol [15], we construct subsampled variants at ratios 0.25, 0.50, and 0.75, and compare each sampled table against the original table through column embeddings.

For each column, we compute the average cosine similarity between the original column embedding and all sampled variants, and the MCV over the set consisting of the original column embedding plus all sampled versions. The final table score is the mean over all columns in the same table.

- Table 52: Dataset parameters for Sample Fidelity. The 0.25 setting has fewer unique subsamples for small tables. The 0.50 and 0.75 settings use 11 versions per base table throughout.

Sampling ratio Base tables 11 variants 8 variants 7/6 variants 0.25 4,964 3,816 335 475 / 338 0.50 4,964 4,964 0 0 / 0 0.75 4,964 4,964 0 0 / 0

Analysis. Three patterns are especially clear. First, all eight models become more stable as the retained row fraction grows from 0.25 to 0.75, which means that sample fidelity is strongly tied to how much of the original row distribution remains visible. Second, tabert is the most stable model

- at every sampling ratio, indicating that its column representations are highly insensitive to the removal of rows. Third, tabbie is the most balanced non-tabert model, combining high cosine similarity

- Table 53: Full Sample Fidelity results. Each cell reports cosine mean ± std on the first line and MCV mean ± std on the second line.

Model 0.25 0.50 0.75

bert 0.9138±0.0284 / 0.0211±0.0083 0.9619±0.0175 / 0.0144±0.0073 0.9828±0.0108 / 0.0110±0.0076 gte 0.8290±0.0564 / 0.0316±0.0134 0.9180±0.0342 / 0.0212±0.0116 0.9620±0.0199 / 0.0175±0.0124 starmie 0.9198±0.1161 / 0.0306±0.0407 0.9846±0.0291 / 0.0123±0.0175 0.9923±0.0194 / 0.0084±0.0146 tabbie 0.9567±0.0363 / 0.0099±0.0087 0.9873±0.0136 / 0.0032±0.0032 0.9959±0.0052 / 0.0014±0.0017 tabert 0.9933±0.0029 / 0.0025±0.0012 0.9965±0.0015 / 0.0017±0.0007 0.9982±0.0009 / 0.0013±0.0006 tabsketchfm 0.4692±0.0984 / 0.9489±0.1555 0.6596±0.0916 / 0.7524±0.1485 0.8084±0.0720 / 0.5623±0.1356 tapas 0.7961±0.0668 / 0.3975±0.1023 0.9146±0.0407 / 0.2619±0.0666 0.9602±0.0260 / 0.1827±0.0533 turl 0.8679±0.0736 / 0.4658±0.1899 0.9506±0.0313 / 0.2669±0.0988 0.9782±0.0197 / 0.1735±0.0726

- Table 54: Dataset parameters for Perturbation Robustness. “Valid tables” are tables with at least one changed column under the given perturbation type.

Perturbation Original Valid Skipped Changed Unchanged Changed type tables tables tables pairs pairs columns

Content equivalence 80 29 51 238 1,320 101 Schema abbreviation 80 76 4 691 1,456 232 Schema synonym 80 69 11 453 1,708 202

with low variance across all three settings, whereas tabsketchfm is much more sample-sensitive and degrades substantially when only 25% of rows are retained.

###### P.4 Perturbation Robustness

The Perturbation Robustness diagnostic evaluates whether column representations remain stable after semantically valid modifications to the table schema or content. Following the Observatory Perturbation Robustness protocol [15], we use the same three perturbation families: DB_schema_synonym (schema names replaced by synonyms), DB_schema_abbreviation (schema names replaced by abbreviations), and DB_DBcontent_equivalence (schema or content rewritten in a semantically equivalent form).

The analysis uses a strict changed-only headline protocol. For each original table and each column that is modified by a perturbation, we compute the average cosine similarity between the original column embedding and all changed variants. The table-level score is the mean over all changed columns in that table.

Analysis. The hardest perturbation is content equivalence, followed by the two schema-only perturbations. This shows that semantic re-expression of column content is much more difficult than renaming a schema attribute with a synonym or abbreviation. tabbie is the strongest overall model and remains the most robust on the hardest content-equivalence setting. starmie is nearly invariant to schema-level changes, but drops more than tabbie under content-level perturbations, suggesting that its robustness is particularly strong at the schema level. By contrast, gte and especially tabsketchfm are clearly more sensitive to semantic perturbations. Note that the content equivalence evaluation uses only 29 valid tables (Table 54), so these comparisons should be treated as indicative rather than definitive.

###### P.5 Row/Column Order Insignificance

The Row/Column Order Insignificance diagnostic evaluates whether learned representations are invariant to row-order and column-order permutations. Following the Observatory Row/Column Order Insignificance protocol [15], each base table is paired with multiple shuffled variants from wiki_tables. In our data, 4,451 base tables have 11 variants (the original table plus 10 shuffles), while 513 tables have 6 variants.

We study six evaluation conditions: column/column, column/row, row/column, row/row, table/column, and table/row. The first term denotes the embedding granularity and the second term denotes the applied shuffle type. Thus, column/column measures the invariance of column embeddings to column permutation, column/row tests whether column semantics change under row

- Table 55: Full Perturbation Robustness results. Each cell reports changed-only cosine mean with small-font standard deviation.

Model Content equivalence Schema abbreviation Schema synonym

bert 0.8420±0.0358 0.9453±0.0294 0.9524±0.0277 gte 0.5777±0.0961 0.7577±0.0914 0.8283±0.0644 starmie 0.9261±0.0715 0.9998±0.0008 0.9998±0.0009 tabbie 0.9783±0.0224 0.9982±0.0028 0.9987±0.0017 tabert 0.9451±0.0181 0.9507±0.0160 0.9635±0.0134 tabsketchfm 0.2939±0.1748 0.6471±0.2820 0.5459±0.2646 tapas 0.6605±0.0866 0.9545±0.0342 0.9584±0.0309 turl 0.7551±0.0965 0.9857±0.0425 0.9905±0.0344

- Table 56: Cosine similarity results for the Row/Column Order Insignificance diagnostic. The six metric columns are grouped by embedding granularity and shuffle type. NA means that the corresponding embedding granularity is not available for that model.

Column embeddings Row embeddings Table embeddings Model column shuffle row shuffle column shuffle row shuffle column shuffle row shuffle

bert 1.0000± 0.0000 0.9809± 0.0147 0.9754± 0.0102 1.0000± 0.0000 0.9731± 0.0144 0.9824± 0.0168 dae NA NA 0.5015± 0.0750 0.9770± 0.0082 NA NA gte 1.0000± 0.0000 0.9704± 0.0190 0.9556± 0.0204 1.0000± 0.0000 0.9590± 0.0174 0.9648± 0.0199 saint NA NA 0.7533± 0.1345 0.8857± 0.0823 NA NA scarf NA NA 0.5371± 0.0700 0.9741± 0.0092 NA NA starmie 0.9988± 0.0027 0.9708± 0.0346 NA NA 0.9988± 0.0027 0.9580± 0.0518 subtab NA NA 0.8233± 0.0762 0.9245± 0.0282 NA NA tabbie 0.9966± 0.0039 0.9986± 0.0021 0.9941± 0.0029 1.0000± 0.0000 0.9975± 0.0028 0.9988± 0.0021 tabert 0.9706± 0.0056 0.9947± 0.0021 NA NA 0.9958± 0.0012 0.9984± 0.0009 tabicl NA NA 0.6155± 0.0854 1.0000± 0.0000 NA NA tabpfn NA NA 0.9998± 0.0001 1.0000± 0.0000 NA NA tabsketchfm 0.9598± 0.0255 1.0000± 0.0000 NA NA 0.9902± 0.0063 1.0000± 0.0000 tabtransformer NA NA 0.4226± 0.0808 0.4370± 0.0761 NA NA tabular_binning NA NA 0.5587± 0.0592 0.9725± 0.0084 NA NA tapas 0.9277± 0.0242 0.9821± 0.0108 NA NA 0.9461± 0.0680 0.9705± 0.0624 tapex NA NA NA NA 0.9944± 0.0045 0.9988± 0.0029 transtab NA NA 0.8712± 0.0621 0.9627± 0.0349 NA NA turl 0.9974± 0.0125 0.9970± 0.0128 NA NA 0.9974± 0.0128 0.9972± 0.0130 tuta NA NA 0.9279± 0.0262 1.0000± 0.0000 0.9278± 0.0547 0.9387± 0.0549 vime NA NA 0.5365± 0.0757 0.9783± 0.0078 NA NA

shuffling, row/column and row/row evaluate the stability of row embeddings under column-order and row-order changes, respectively, and table/column and table/row test whole-table invariance to column and row permutations. When needed, shuffled items are realigned to their original indices before comparison; models that already canonicalize the relevant order do not require this extra step.

Analysis. The strongest qualitative pattern is that many models are almost perfectly stable on row/row, whereas the more difficult condition is often row/column, which requires row semantics to survive a change in column order. Among shared column/table models, tabbie is the most balanced model across all available conditions, while turl and tabert also show very strong order robustness. tabsketchfm is especially notable because it is extremely stable under order perturbations even though it is much weaker on the other two tasks. For row representations, tabpfn is almost ideal, tabbie, bert, and gte form a strong second tier, and tabtransformer is the least robust model in this setting, consistent with its use of learned column positional embeddings that make row representations inherently sensitive to column order.

###### P.6 Cross-Task Comparison of Shared Models

To avoid confounding caused by different model coverage across tasks, the cross-task comparison uses only the eight models that appear in all three evaluations: bert, gte, starmie, tabbie, tabert, tabsketchfm, tapas, and turl. We rank models within each task by their average cosine performance and then compute the average rank across tasks (an ordinal summary that reflects relative positioning rather than the magnitude of score differences).

Analysis. tabbie is the most balanced model across tasks. It is not always the single best model on every individual condition, but it remains near the top under sampling, semantic perturbation, and

- Table 57: MCV results for the Row/Column Order Insignificance task. The six metric columns are grouped by embedding granularity and shuffle type. NA means that the corresponding embedding granularity is not available for that model.

Column embeddings Row embeddings Table embeddings Model column shuffle row shuffle column shuffle row shuffle column shuffle row shuffle

bert 0.0000± 0.0000 0.0097± 0.0054 0.0127± 0.0054 0.0000± 0.0000 0.0129± 0.0063 0.0102± 0.0057 dae NA NA 0.1699± 0.0371 0.0412± 0.0135 NA NA gte 0.0000± 0.0000 0.0052± 0.0031 0.0074± 0.0035 0.0000± 0.0000 0.0064± 0.0033 0.0052± 0.0031 saint NA NA 0.0545± 0.0357 0.0289± 0.0247 NA NA scarf NA NA 0.1551± 0.0317 0.0324± 0.0086 NA NA starmie 0.0028± 0.0039 0.0222± 0.0158 NA NA 0.0028± 0.0039 0.0250± 0.0236 subtab NA NA 0.3267± 0.1011 0.2524± 0.0550 NA NA tabbie 0.0010± 0.0012 0.0004± 0.0005 0.0037± 0.0013 0.0000± 0.0000 0.0014± 0.0013 0.0009± 0.0009 tabert 0.0038± 0.0009 0.0019± 0.0008 NA NA 0.0010± 0.0004 0.0006± 0.0003 tabicl NA NA 0.0439± 0.0167 0.0000± 0.0000 NA NA tabpfn NA NA 0.0001± 0.0000 0.0000± 0.0000 NA NA tabsketchfm 0.0066± 0.0034 0.0000± 0.0000 NA NA 0.0013± 0.0008 0.0000± 0.0000 tabtransformer NA NA 0.2524± 0.0582 0.2470± 0.0540 NA NA tabular_binning NA NA 0.1514± 0.0326 0.0394± 0.0111 NA NA tapas 0.0276± 0.0088 0.0142± 0.0048 NA NA 0.0288± 0.0245 0.0197± 0.0230 tapex NA NA NA NA 0.0191± 0.0096 0.0046± 0.0036 transtab NA NA 0.0279± 0.0204 0.0136± 0.0100 NA NA turl 0.0027± 0.0091 0.0029± 0.0089 NA NA 0.0044± 0.0148 0.0046± 0.0143 tuta NA NA 0.0587± 0.0190 0.0000± 0.0000 0.0514± 0.0348 0.0430± 0.0307 vime NA NA 0.1638± 0.0378 0.0430± 0.0144 NA NA

- Table 58: Cross-task ranking over the eight column-capable models from our pool that appear in all three robustness evaluations. Lower average rank means better overall robustness.

Model Sample rank Perturbation rank Order rank Average rank tabbie 2 1 1 1.33 tabert 1 3 3 2.33 starmie 3 2 6 3.67 turl 5 5 2 4.00 bert 4 4 5 4.33 tabsketchfm 8 8 4 6.67 gte 6 7 7 6.67 tapas 7 6 8 7.00

structural reordering, which makes it the strongest all-round choice in our study. tabert is the best model for sample fidelity, while starmie is especially strong for schema-level perturbations. turl is consistently robust without being the single best model in any one dimension. Finally, tabsketchfm is the clearest example of a task-specialized behavior: it is highly order-invariant, but much weaker under row subsampling and semantic perturbation.

###### P.7 Implementation Notes and Caveats

All numbers in this section are recomputed from finalized embeddings using a unified table-first headline protocol rather than copied from older logs. This recomputation includes three practical decisions that are important for interpretation:

- • Table-first aggregation. All three tasks first aggregate to per-table scores and only then compute dataset-level mean and standard deviation.
- • Changed-only perturbation scoring. For Perturbation Robustness, only changed columns contribute to the headline cosine metric. We do not report MCV for this task, matching the original Observatory perturbation protocol.
- • MCV comparability. MCV is computed from different numbers of embedding variants in the sample-fidelity and order-insignificance tasks (up to 11 for order shuffles), and the covariance estimate is rank-deficient in all cases (K+1 samples in d-dimensional space, K+1 ≪ d). MCV magnitudes should therefore be compared only within the same task.

These caveats do not change the main qualitative conclusions, but they matter for correct interpretation of the absolute numbers and for reproducible comparison across properties.

##### Q Computational Efficiency

- Table 59: Column embedding generation cost (median wall-clock seconds ± IQR across the efficiency test suite). All models are frozen inference.

Model Time (s) Datasets BERT 5.3 ± 4.9 58 GTE 5.7 ± 1.3 58 TABBIE 7.4 ± 3.2 59 TAPAS 7.8 ± 3.9 58 TURL 11.2 ± 1.8 55 TABERT 13.2 ± 9.7 58 TABSKETCHFM 14.1 ± 5.4 58

- Table 60: Table embedding generation cost (median wall-clock seconds ± IQR). Both models are frozen inference.

Model Time (s) Datasets TAPEX 6.1 ± 3.3 58 TUTA 119.6 ± 138.8 58

We report per-model embedding generation cost across three workloads (column, table, row) on a controlled efficiency test suite. All measurements use a single NVIDIA L40S GPU with 32 GB RAM per job.

Efficiency test suite. The suite comprises: (i) Eff-Real: 8 anchor tables selected from the 50 TRL-RBENCH OpenML tables via metadata-space clustering (covering 1000–71518 rows and 18–1775 columns with diverse type mixes and missingness rates); (ii) Eff-Scale: 47 semi-synthetic tables generated by varying one factor at a time from a baseline (row track: N ∈ {500,...,100k}, D ∈ {8,...,256}, categorical share, cardinality, missingness; column track: C ∈ {4,...,128}, context rows, cell token length, type mix); and (iii) Bridge: 3 tables valid for both row and column workloads. Each model is timed using its unmodified production embedding script via a thin wall-clock wrapper, ensuring that the measured cost exactly matches the actual benchmark pipeline.

Column and table embedding cost. Table 59 reports column-level results. Generic text encoders are fastest (BERT 5.3s, GTE 5.7s median), while TABERT (13.2s) and TABSKETCHFM (14.1s) are slowest, a 2.7× spread. For table embeddings (Table 60), TAPEX (6.1s) is 20× faster than TUTA (119.6s), reflecting TUTA’s cell-level tokenization overhead.

Row embedding cost. Table 61 reports row-level results. The 101× spread between the fastest (TABICL, 8.7 s) and slowest (TRANSTAB, 875 s) models is driven primarily by the frozen-vs.-trained regime distinction: all target-table self-supervised models include per-table training, which dominates their wall-clock cost. Within the frozen regime, TABICL and TABPFN (20.7s) are fastest despite requiring a fit step, because their meta-learned priors avoid gradient-based training.

Scaling behavior. Figure 12 shows how row embedding cost scales with table size. Training-based models (TRANSTAB, SAINT) scale super-linearly with row count, while frozen inference models scale approximately linearly. Feature count scaling is more uniform across the models in the sweep. Column-embedding cost (Table 59) is dominated by per-column tokenization overhead: TABERT and TABSKETCHFM are roughly 2.5–2.7× slower per job than BERT/GTE. The column track of EFF-SCALE (C ∈ {4,...,128}) confirms that this gap widens approximately linearly with column count (full curves in the supplementary material).

Support envelope. Not all models can handle all scales under the 1-hour budget. TURL runs out of memory on tables with >208 columns or >256 features. TABPFN cost rises sharply with feature count on EFF-REAL anchors (18s at 28 features, 263 s at 208 features, and timeout at 1 775 features), roughly a 15× slowdown over a 7× increase in feature count. SAINT and TABTRANSFORMER fail

- Table 61: Row embedding generation cost (median wall-clock seconds ± IQR). “Train” models include per-table self-supervised training. “Infer” models are frozen or meta-pretrained.

Model Type Time (s) Datasets TABICL Infer 8.7 ± 8.8 8 TABBINNING Train 10.2 ± 1.4 38 TABPFN Infer 20.7 ± 125.0 7 SCARF Train 30.9 ± 6.1 38 GTE Infer 31.1 ± 14.5 38 DAE Train 44.8 ± 27.1 38 BERT Infer 50.1 ± 17.4 38 TABBIE Infer 57.1 ± 24.6 38 SUBTAB Train 62.9 ± 20.0 38 VIME Train 71.8 ± 32.2 38 TABTRANSFORMER Train 86.5 ± 38.0 36 SAINT Train 167.9 ± 64.3 37 TRANSTAB Train 875.1 ± 512.1 38

- 101

- 102

- 103

- 104

BERT

SubTab

BERT

SubTab

TABBIE

VIME

TABBIE

VIME

TabularBinning

TabTransformer

TabularBinning

TabTransformer

GTE

GTE

SCARF

SCARF

SAINT

SAINT

SwitchTab

SwitchTab

Wall-clocktime(s)

Wall-clocktime(s)

- 101

- 102

- 103

TransTab

TransTab

DAE

DAE

103 104 105 Number of rows (N)

101 102 Number of features (D)

(a) Row scaling (varying N)

(b) Feature scaling (varying D)

- Figure 12: Embedding generation cost vs. table size (log-log scale). Training-based models scale super-linearly with rows.

on the widest anchor table (1775 columns). These limits are important for practitioners selecting models for large-scale deployment.

Methodology. TABICL and TABPFN are timed only on EFF-REAL anchor tables. Their context-fit step is most representative of production usage when run on real labeled splits, while the synthetic EFFSCALE suite is unlabeled. This labeled-split timing reflects deployment cost. The row-embedding matrix consumed by every TRL-RBENCH task is extracted from a target-agnostic forward pass conditioned only on the unlabeled X rows, so the same matrix is reused across all curated targets within a table. Each measurement times the unmodified production script end-to-end (including model loading, preprocessing, and any per-table training) via a subprocess wrapper that records wallclock time and polls nvidia-smi for peak GPU VRAM at 0.5s intervals. Results are recorded as individual JSON files with full provenance (hostname, GPU type, SLURM job ID, return code, output verification). The complete 994-run result set and analysis code are included in the supplementary material.

##### R Reproducibility Details

Data access. All source datasets are publicly available. OpenML tables are accessed via the OpenML API using the dataset IDs listed in Appendix F. DeepMatcher datasets are from the original DeepMatcher release. WDC Products data is from the WDC Product Data Corpus (the LSPM v2 release). CTBench datasets (SATO, SOTAB, SANTOS, Valentine, etc.) are from their respective original releases. DLTE parent tables are derived from TabFact and WTQ.

Splits. Row prediction uses the canonical OpenML train/test splits. Record linkage retains the original DeepMatcher (3:1:1 train/valid/test) and WDC splits; both benchmarks define exactly two tables per dataset, so table-disjoint evaluation does not apply.

For the four CTBench pairwise tasks marked † (join classification, column overlap, union classification, union regression), we generate table-disjoint splits as follows. All pairs from the source pair-level random splits are pooled, and the set of unique tables is partitioned randomly (seed 42) into disjoint train/valid/test sets at a 70/15/15 ratio. Each pair is then assigned to the split containing both of its tables, and cross-partition pairs are discarded. For spider_join, tables are grouped by database prefix before partitioning, so all tables from the same database land in the same split. Retention rates range from 53% to 100% of original pairs depending on the dataset. The remaining supervised table-pair task (table subset) already has table-disjoint splits in the source data.

DLTE uses a parent-table-level 827/207/345 train/dev/test split fixed at benchmark construction (before any pipeline evaluation) by two successive calls to train_test_split: first 75/25 train+dev vs. test, then 80/20 train vs. dev within the train+dev portion, yielding an effective 60/15/25 split. The split is random, not stratified by source (TabFact vs. WTQ). Given the 989/390 source composition, each split is expected to contain roughly 72% TabFact and 28% WTQ parents. Split manifests are included in the released code. All 1,120 canonical pipelines completed across all 5 rounds.

Hyperparameters. Supervised probe training follows the unified protocol of Sec. 3.1: for each supervised probe task, we train both a linear head and a one-hidden-layer MLP with hidden size 256 using Adam (learning rate 10−3) for up to 100 epochs with early stopping on the validation set, and the headline score is the arithmetic average of the two heads.

Seeds. All supervised probe results, including record linkage, are averaged over 5 random seeds: 42, 52, 62, 72, 82.

Model wrappers and launch configurations. Exact model-specific wrapper parameters, truncation rules, and embedding-generation launch configurations are provided in the released codebase. Appendix D states the benchmark-level policy. The codebase contains the executable per-model settings used to produce the released embeddings.

Third-party model packages. TabPFN and TabICL are used as provided by their respective authors. Their internal weights are not modified during evaluation. The benchmark consumes only the exported intermediate row representations used by the shared probe protocol of Sec. 3.1, never their task predictions as final benchmark outputs.

##### S Statistical Reporting

For column/table tasks, we report means over 5 random seeds for all supervised probe results. Standard deviations are reported per cell in the main results tables (Tables 2, 3) and in the aggregation ablation tables (Tables 17, 18). Training-free tasks (column clustering, schema matching) are deterministic given fixed embeddings. For row prediction, the reported metrics are macro-averaged across all 123 targets. The sole exception is TABTRANSFORMER, which covers 63 targets due to its categorical-feature requirement. For DLTE, end-to-end results are 5-round averages. We do not report confidence intervals for the normalized-rank aggregates, as these are summary statistics over heterogeneous per-task metrics. Per-task raw scores with standard deviations are the appropriate unit of statistical comparison.

Significance tests. To assess whether model differences are statistically meaningful, we apply Friedman omnibus tests and Holm-corrected pairwise Wilcoxon signed-rank tests, using per-task mean scores (averaged over 5 seeds) as the unit of analysis. Table 62 summarizes the results.

- Table 62: Friedman omnibus tests and Holm-corrected pairwise Wilcoxon signed-rank results. Each row uses per-task means (averaged over 5 seeds) as the unit of analysis. “Sig. pairs” reports the

number of pairwise comparisons with padj < 0.05 after Holm correction, out of k2 total pairs for k models. TABTRANSFORMER is excluded from the RBench rows because its partial target coverage

(63 of 123) breaks the matched-sample requirement of Friedman/Wilcoxon tests, leaving 13 of the 14 non-baseline row models.

Friedman χ2 (p)

Sig. pairs (Holm) All tasks CTBench 13 8 32.1 (<0.0001) 6/28

Scope Suite Tasks Models

Schema CTBench 3 8 18.0 (0.012) Join CTBench 3 8 13.0 (0.072) Union CTBench 5 8 11.0 (0.139) —

Classification RBench 77 13 209.4 (3.7×10−38) 26/78 Regression RBench 46 13 154.6 (6.5×10−27) 38/78

On TRL-CTBENCH (13 tasks, 8 fully supported models), the Friedman test rejects the null hypothesis that all models perform equally (χ2 = 32.1, p < 0.0001). However, only 6 of 28 pairwise Wilcoxon signed-rank comparisons are significant after Holm correction (all padj < 0.05), and all six involve TABSKETCHFM being significantly weaker than the top models. The remaining models (BERT, GTE, TABERT, TAPAS, TURL, TABBIE, and STARMIE) are not significantly different from each other across the full task set. Per-family Friedman tests are significant for Schema (p = 0.012) but not for Join (p = 0.072) or Union (p = 0.139), consistent with the observation that Union-family differences are especially narrow. Figure 13 renders the corresponding Demšar critical-difference (CD) diagram [16] using post-hoc Nemenyi at α = 0.05: BERT holds the best mean rank (2.23), and under the CD threshold only pairs involving TABSKETCHFM (vs. BERT or GTE) and BERT vs. TABBIE clear significance. Most CTBench models form a single overlapping clique, visually confirming the near-tie pattern.

On TRL-RBENCH, the larger number of targets provides substantially more statistical power. Classification (77 targets, 13 models; TABTRANSFORMER excluded due to partial target coverage) yields χ2 = 209.4, p = 3.7 × 10−38, with 26/78 pairwise comparisons significant. Regression (46 targets, 13 models) yields χ2 = 154.6, p = 6.5 × 10−27, with 38/78 pairs significant. TABICL’s advantage over all other models is confirmed on both classification and regression (all padj < 0.001). Among the remaining transfer-based and target-table learners, most pairwise differences are non-significant (padj = 1.0), supporting the paper’s conclusion that training regime and task family matter more than individual model choice within a regime.

TRL-DLTE per-stage significance. For DLTE we test whether each stage’s model identity has a significant end-to-end effect, blocking by the remaining two stages and using the aggregate of 5-round mean scores per pipeline as the unit of analysis. All three stages have balanced complete-block designs: Stage 1 has 112 blocks (8 column × 14 row other-stage combinations) with 10 target models, Stage 2 has 140 blocks (10 tables × 14 rows) with 8 target models, and Stage 3 has 80

CTBench critical-difference diagram (k=8 models, N=13 tasks, Friedman 2=32.1, p=3.97e-05; Nemenyi CD =0.05=2.91)

CD = 2.91

2 3 4 5 6 7 8

BERT (2.23) GTE (3.08)

TabSFM (6.92) TABBIE (5.38) Starmie (5.08) TURL (5.08)

TAPAS (4.08) TaBERT (4.15)

- Figure 13: CTBench critical-difference diagram (Demšar style). The horizontal axis shows mean rank across 13 CTBench tasks for the 8 fully-supported models. Horizontal bars group models whose mean-rank differences are below the Nemenyi critical difference (CD= 2.91), so models sharing a bar are not significantly different at α = 0.05. Only pairs involving TABSKETCHFM and the two best-ranked generic-text encoders (BERT, GTE) or BERT vs. TABBIE clear the CD threshold. Most of the top-eight CTBench models live in a single overlapping clique.

blocks (10 tables × 8 columns) with 14 target models. Table 63 reports Friedman omnibus statistics with Kendall’s concordance W as an effect-size summary, together with Holm-corrected pairwise Wilcoxon signed-rank tests for all three stages on the primary metric UJ-H and the complementary Cell F1 diagnostic. All six stage-metric combinations reject the null at p ≤ 1.1×10−13 (Stage 3 UJ-H), with the other five well below 10−16, but the interesting picture is the effect-size pattern rather than the p-values. Stage 2 (column alignment) carries the single largest effect on both metrics (Kendall’s W = 0.79 on Cell F1, 0.66 on UJ-H, with 26/28 and 24/28 Holm-significant pairs), consistent with the finding that column alignment is the most sensitive lever in the pipeline. Stage 1 (retrieval) has a moderate effect on both metrics (W = 0.57 on Cell F1 and W = 0.33 on UJ-H, with 38/45 and 38/45 Holm-significant pairs). Stage 3 exhibits a striking metric-dependent asymmetry. On Cell F1 the effect is strong (W = 0.61, χ2 = 638.7) with 87 of 91 pairwise comparisons Holmsignificant, reflecting the sharp separation between union-dedup specialists (TABTRANSFORMER, SUBTAB, SAINT) and the other row models on union-side raw cell recovery. On UJ-H, by contrast, the effect collapses to W = 0.09 (weak) and only 31 of 91 pairwise comparisons survive Holm correction, even though the omnibus remains significant. This is not evidence of absent Stage 3 signal: the Oracle-RA diagnostic (Table 39) shows a latent cross-row-model UJ-H spread of 0.546, far above the end-to-end marginal span of ∼ 0.013. The metric-dependent asymmetry instead reflects the DLTE composition bottleneck quantified in Sec. 4.4: Cell F1 is sensitive to Stage 3 union-side behavior, which passes through the pipeline largely intact, while UJ-H additionally requires join-side recovery, which is masked by upstream retrieval/alignment error and therefore compresses the visible Stage 3 effect within the end-to-end pipeline. These tests quantify average stage effects, not the additivity of best compositions. Large average effects do not by themselves determine which model combination is best, and top DLTE quality depends on non-additive compositional fit.

- Table 63: DLTE per-stage statistical significance on the primary metric UJ-H and the complemen-

tary Cell F1 diagnostic. Each row tests whether stage-specific model identity affects end-to-end performance, blocking by the remaining two stages (one block per (other-stage-a, other-stage-b)

combination, with the aggregate 5-round mean score per pipeline as the unit of analysis). Kendall’s W is the effect-size summary of Friedman, W = χ2/(n(k − 1)), bounded in [0,1]. W > 0.5 is a strong effect, W < 0.1 is weak. “Sig. pairs (Holm)” counts Holm-corrected pairwise Wilcoxon signed-rank

tests with padj < 0.05, out of k2 total pairs for k models. All three stages have complete block designs and yield highly significant Friedman omnibus tests on both metrics. The most informative

contrast is the large metric-dependent Stage 3 effect-size gap (W = 0.61 on Cell F1 vs. W = 0.09 on UJ-H), which reflects how upstream retrieval/alignment errors mask the join-side Stage 3 signal in the balanced UJ-H metric.

Friedman χ2 (p)

Kendall’s W

Sig. pairs (Holm, < 0.05)

Target stage Metric Blocks k

- Stage 1 (table model)

Cell F1 112 10 569.6 (7.1×10−117) 0.57 38/45 UJ-H 112 10 332.6 (3.2×10−66) 0.33 38/45

- Stage 2 (column model)

Cell F1 140 8 776.7 (2.0×10−163) 0.79 26/28 UJ-H 140 8 649.9 (4.3×10−136) 0.66 24/28

- Stage 3 (row model)

###### Cell F1 80 14 638.7 (4.3×10−128) 0.61 87/91 UJ-H 80 14 90.6 (1.1×10−13) 0.09 31/91

##### T Dataset Counting Protocol

The total dataset count depends on the grouping granularity. At the finest level, treating each distinct dataset source as one entry, the benchmark contains 87 datasets (20 CTBench + 50 OpenML rowprediction + 16 record-linkage + 1 DLTE enrichment lake). Alternative aggregations include: 84 (grouping the 4 OpenData regional splits as one source), 77 (additionally grouping the 4 WDC Products size variants as one entry and pairing each DeepMatcher dirty variant with its clean counterpart), and 28 (additionally grouping all 50 OpenML row-prediction tables as one curated collection). Throughout the main paper, we use the dataset-source level as the default unless otherwise noted.

##### U Broader Impact

A benchmark for tabular representations can improve scientific comparability and reduce evaluation fragmentation. It can also help practitioners identify when specialized table-aware models are necessary and when simpler frozen encoders suffice.

At the same time, stronger table representations can be used in settings involving sensitive records, schema inference, or large-scale data linkage. Benchmark releases should therefore carefully respect data licenses, privacy constraints, and documentation requirements. In particular, record linkage and data lake retrieval settings can raise concerns around surveillance, re-identification, and inappropriate dataset fusion. A responsible release should document dataset provenance and usage restrictions.

##### V Licenses and Asset Documentation

All source datasets are used under their original licenses. TRL-BENCH creates derived benchmark assets from public source data, including curated target selections, rewritten row-pair tasks, new table-disjoint split manifests, and TRL-DLTE table fragments. For each source, we document the transformation applied (label repairs, split regeneration, row-pair rewrites, fragmentation) and respect the redistribution terms of the original license: where source licenses permit redistribution we release the derived files directly, and where they do not we release reconstruction scripts and manifests rather than mirrored raw files. For platform-hosted collections (e.g., OpenML, CKAN / open government portals), licenses can vary by dataset. The release therefore includes a per-asset manifest with the original source, URL, license, and redistribution status. The benchmark code will be released under an open-source license.

- Table 64: Licenses and provenance for assets used in TRL-BENCH. License entries reflect the upstream code/data licenses observed at the official source as of preparation. Where the source repository or release page does not state an explicit dataset license, we mark the cell accordingly. “Code” and “data” licenses are listed separately when they differ. For benchmarks redistributed through LakeBench, both the LakeBench redistribution license and the upstream license are noted.

Asset Citation License Notes OpenML tables [73] Per-dataset (commonly CC0 / CC-BY) Via OpenML API; licenses are set per

dataset by the uploader SATO [82] Apache 2.0 Public release; tables from the WebTables

corpus within VizNet SOTAB [43] Not explicitly stated on dataset page

Public release on Web Data Commons; benchmark data must not appear in training corpora (per WDC notice)

(CEUR paper itself: CC-BY 4.0)

WikiCT (relation) [17] Apache 2.0 (TURL code); CC-BY-SA inherited from upstream Wikipedia content

Relation-extraction split from TURL; tables from the WikiTables / TabEL corpus

Wiki Containment / Wiki Union

[68] CC-BY 4.0 (LakeBench Zenodo record); CC-BY-SA 4.0 (label files and Wikipedia content)

Derived from Wikipedia tables; splits from LakeBench (Zenodo archives wiki-containment and wiki-union)

SANTOS [39] BSD-3-Clause (code); CC-BY 4.0 (data on Zenodo)

Public release

UGEN [58] MIT GitHub: northeastern-datalab/gen TUS / TUS-hard [54] Unspecified (source repo has no LICENSE

TUS from Nargesian et al.; TUS-hard is a repo-derived low-overlap subset (Hungarian-matched containment < 0.70)

file)

Valentine [45] Apache 2.0 Public release; benchmark assets in a separate (also Apache 2.0) repo

Spider Join [81, 68] CC-BY-SA 4.0 (Spider upstream; ShareAlike propagates to Spider-derived artifacts); CC-BY 4.0 (LakeBench-original metadata)

Databases from Spider; join-classification benchmark from LakeBench

OpenData (main / CAN / USA / UK / SG)

[18] Unspecified (BIT-DataLab/LakeBench source repo has no LICENSE; data hosted on Google Drive)

Join- and union-search benchmarks from the Deng et al. LakeBench; per-portal open-data licenses for source tables (Canadian / UK / US / Singapore open-data portals)

ECB Union / CKAN Subset

[68] CC-BY 4.0 (LakeBench Zenodo); per-portal open licenses for source tables

Benchmarks from LakeBench; source data from open govt. portals

DeepMatcher [53] BSD-3-Clause (code); benchmarks released without explicit dataset license

Public release WDC Products [63, 62] Not explicitly stated on Web Data

Training/gold-standard release at WWW ’19 Companion (ECNLP); LSPM v2 extension from WIMS 2020

Commons; publicly released

WTQ [60] CC-BY-SA 4.0 Table QA + DLTE parent tables NQ-Tables [30] Apache 2.0 (TAPAS release artifact, NQ

Public release

source repo); CC-BY-SA 3.0 (Wikipedia text upstream, pre-2023 NQ era)

TabFact [13] CC-BY 4.0 (data); MIT (code) DLTE parent tables CKAN distractors [68] CC-BY 4.0 (LakeBench Zenodo);

TRL-BENCH-derived selection from the LakeBench CKAN Subset pool; not a separately published LakeBench split

per-portal open licenses for source tables

LakeBench (Srinivas et al.)

[68] CC BY-NC-ND 4.0 (code at IBM/tabsketchfm); CC-BY 4.0 / CC-BY-SA 4.0 (data on Zenodo)

Public release; TRL-BENCH derives only from the CC-BY 4.0 Zenodo data, not the NC-ND code

TabArena [21] Apache 2.0 Public release; per-table licenses inherited from upstream sources (UCI, Kaggle, OpenML, etc.)

