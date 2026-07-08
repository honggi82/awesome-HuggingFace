# arXiv:2606.07454v1[cs.IR]5Jun2026

[Figure 1]

###### PaperFlow: Profiling, Recommending, and Adapting Across Daily Paper Streams

Fuqiang Wang1, Song Tan1, Zheng Guo1, Jiaohao Fu1, Xinglong Xu2, Bihui Yu2, Jie Dong2, Zheng Sun2, Siyuan Li1, Jingxuan Wei1, Cheng Tan3

1Key Laboratory of Computing Power Network and Information Security, Ministry of Education, Shandong Computer Science Center (National Supercomputer Center in Jinan), Qilu University of Technology (Shandong Academy of Sciences), 2University of Chinese Academy of Science, 3Shanghai Artificial Intelligence Laboratory

Scientific paper recommendation is typically evaluated as static ranking over a fixed candidate set, yet real scientific reading unfolds as a daily, longitudinal process in which interests shift and feedback accumulates. We introduce PaperFlow, a framework that organizes it into three coupled stages: Profiling, which constructs and maintains a structured, inspectable scholarly profile from heterogeneous cold-start evidence; Recommending, which ranks each date-specific paper stream through multi-signal aggregation under a fixed display budget; and Adapting, which updates user state from semantically distinct feedback signals and models interest drift across days. We further define a longitudinal user-day benchmark that fixes users, dates, candidate pools, visible inputs, and hidden simulated relevance labels under a shared temporal information boundary. The benchmark contains 24 simulated research users, 50 daily paper streams, 1,200 user-day episodes, 20,727 unique papers, and 497,448 episode–paper records. We additionally specify a blind human-evaluation protocol to validate alignment between automatic metrics and expert judgments. Experiments against five scientific recommendation baselines show that PaperFlow achieves the strongest oracle-based ranking, the highest behavioral alignment with simulated reading selections, and the best blind humanevaluation score.

Code Website Dataset

[Figure 2]

[Figure 3]

###### 1 Introduction

Scientific paper recommendation is typically framed as a one-shot ranking problem: given a user representation and a fixed candidate set, produce a relevance-sorted list. This formulation overlooks the temporal structure of real scientific reading. In practice, researchers confront a fresh paper stream every day, allocate scarce reading time to a small subset, skip many plausible papers, and continuously revise their interests as projects mature and new methods emerge [32, 35, 19, 31]. A system that serves this workflow must solve three tightly coupled problems: it must profile the researcher, maintaining a structured, inspectable representation of evolving interests; it must recommend, selecting from today’s candidate stream under a strict display budget; and it must adapt, updating user state from semantically heterogeneous feedback so that each day’s reading informs the next day’s ranking. Figure 1 illustrates this shift from static recommendation to a daily scientific reading loop.

Existing work addresses parts of this loop, but usually in separate settings. Scholar Inbox is the closest predecessor, studying daily scientific recommendations with explicit feedback and cold-start initialization [4]; natural-language profile and paper-assistant systems improve profile transparency, search, question answering, survey generation, and reading support [21, 29, 5]; and dynamic recommendation work models feedback and interest drift in evolving user representations [15]. However, these efforts do not yet define a controlled temporal user-day task: one in which methods rank the same daily candidate pools using only visible pre-ranking inputs, consume cross-day feedback and drift state in

###### ParadigmA: Static Embedding & Explicit Feedback

###### ParadigmB (Ours): PaperFlow

###### steps

[Figure 4]

###### Fine-grained Neuro-Symbolic Loop

Traditional Recommendation Baselines

steps

Sparse Implicit Feedback

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

- -New Paper Tracking
- -Manual Cold Start
- -Explicit Feedback
- -Flat Behavior Logging
- -Opaque History
- -Drift Insensitivity

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Black-box Embedding

Scholar Inbox

[Figure 28]

"Transforming implicit reading behaviors into explicit structured logic to enable precise, white-box tracking of long-term cognitive drift with zero interaction burden."

[Figure 29]

Weekly Offline Reflection

Online Scoring

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Traditional Recommendation Pipeline

Multi-grained Concretized Profile

Recommendation

User History

[Figure 35]

[Figure 36]

[Figure 37]

- -Opaque Vector
- -Static Embedding
- -Latent State
- -Final Output
- -Static Ranking

## ?

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Zero-Burden Mechanism?

###### Logic Transparency？

[Figure 42]

[Figure 43]

- 0 1 0 1 1 0 1 0
- 1 0 1 1 0 0 1 1

Explicit Verifiability Zero-Operation Profiling Natural-Read Capture

Purely Implicit Sensing

Explicit Rating

Verifiable Evidence Tag-Based Provenance

[Figure 44]

- 0 1 1 0 1 0 1 0
- 1 0 0 1 1 0 1 0

❌Limitations

Ours：advantages

✔ ️

- 1. High feedback overhead 2. Drift Blindness

1. Implicit-Only Profiling 2. Continual Adaptation

- 3. Black-box Bias 4. Dimensionality Loss

3. White-box Traceability 4. Materialization

Figure 1: Motivation and paradigm comparison between traditional scientific paper recommendation and PaperFlow.

strict temporal order, and are evaluated against hidden relevance labels without future information.

We propose PaperFlow, a framework that couples the three stages within a single daily loop. For profiling, PaperFlow maintains a structured, inspectable scholarly profile that separates research-interest state, controllable preferences, and drift state into distinct, editable fields. For recommending, it ranks each date-specific candidate pool through multi-signal aggregation, combining semantic matching, author and institution priors, behavioral signals, and explicit rules under a fixed display budget. For adapting, it updates user state with signal-specific semantics: selections, explicit edits, and sustained reading provide strong interest evidence, whereas skips supply only weak, context-dependent signals; a drift module uses cross-day behavior to distinguish transient exploration from sustained migration. In parallel, a reading-report channel provides post-selection reading assistance whose feedback adjusts report organization and evidence density rather than directly changing research-interest weights.

We further construct a longitudinal benchmark for user-day evaluation. Each episode fixes a simulated research user, a date, a candidate pool, visible pre-ranking inputs, and historical feedback under a shared temporal boundary. Methods rank the same frozen candidate pools; same-day selections, postrecommendation states, drift outcomes, and hidden simulated relevance labels remain unavailable until evaluation, preventing future-information leakage. The labels enable reproducible comparison but are not human-annotated ground truth. The current snapshot contains 24 simulated users, 50 days, 1,200 user-day episodes, 20,727 unique papers, and 497,448 episode–paper records. A blind human-evaluation protocol is defined separately to validate alignment between automatic metrics and expert judgments. Together, this work contributes a sequential user-day formulation of scientific paper recommendation, the PaperFlow framework that couples structured profiling, multi-signal daily ranking, signal-aware state updating, and behavior-driven drift modeling within a single loop. Experiments against five external baselines that characterize main-setting gains and the trade-off between static relevance ranking and dynamic adaptation.

###### 2 Related Work

- 2.1 Scientific Paper Recommendation

Scientific paper recommendation must jointly address relevance estimation, cold start, and profile interpretability. Scholar Inbox is the workflow-level predecessor, combining daily paper recommendations with explicit feedback and cold-start initialization [4]; but it relies on opaque embeddings and does not model how interests evolve after initialization. Content-based methods improve paper matching through user-behavior modeling [33], while OMRC-MR strengthens scientific-paper representations through QA-style discourse summarization, multi-level contrastive learning, and structure-aware re-ranking [28]. Citation-enhanced and entity-enhanced methods incorporate external impact signals or fine-grained scientific knowledge [17, 16]. Contextualized paper alerts and natural-language profiles increase transparency and support user-specific explanation [11, 21]. Recommender-agent and conversational systems further explore tool use, multi-turn decision making, and preference elicitation [9, 37, 30, 36, 24, 6, 3, 12]. PaperFlow differs from these approaches by maintaining a structured, updateable scholarly profile across daily recommendation and feedback loops rather than treating profile evidence primarily as one-time initialization input.

- 2.2 Paper Reading Assistants

Paper reading assistants support paper search, survey generation, synthesis, and reading comprehension. SurveyAgent and PaSa organize retrieval, recommendation, query generation, and paper screening for academic search [29, 7]. OpenScholar, Arxiv Copilot, and scientific language agents support retrieval-augmented synthesis and personalized academic assistance [1, 14, 25]. Readingsupport systems cover interactive scholarly reading, intelligent skimming, localized citation contexts, and mixed-initiative synthesis [18, 5, 20, 10]. Socially grounded systems bring paper recommendation into research group contexts [26, 27]. PaperFlow instead treats cross-day reading as the primary unit, organizing selections, skips, explicit corrections, intensive-reading requests, report feedback, and profile updates into user-day episodes.

- 2.3 Dynamic Feedback and Interest Drift

Dynamic feedback and interest-drift studies emphasize that personalization must distinguish shortterm exploration from stable preference migration. IDURL models interest drift in sequential recommendation [15], while PISA studies the stability-plasticity trade-off in continual recommendation [34]. Planning and feedback-loop frameworks formulate recommendation as multi-round interaction optimization [23, 2]. Agentic recommender benchmarks emphasize cold start, evolving interests, dynamic information acquisition, and preference updating [22, 8, 13]. PaperFlow grounds these issues in scientific reading, where feedback signals have different semantics and adaptation must be evaluated under date-frozen paper streams.

###### 3 PaperFlow Method

- 3.1 System Overview PaperFlow models paper recommendation as a dynamic personalized reading loop. As shown in

- Figure 2, it constructs a structured scholarly profile from cold-start evidence, ranks each daily candidate pool, and folds selections, skips, reading requests, report feedback, and profile edits back into the next-day profile and drift state, with a reading-report channel for post-selection assistance.

[Figure 45]

[Figure 46]

###### System Key Components

###### Comprehensive Evaluation & Outputs

###### System Workflow Loop

###### 1. Cold Start Profile Construction

- 1. Metadata & Content Fusion

Stable divergence based on stable Signal-shifting,recovered states

- 2. Profile Update & Feedback Learning

- 3. Close-loop Reading & Generation

[Figure 47]

[Figure 48]

[Figure 49]

###### 1. Multi-dimensional Assessment

[Figure 50]

[Figure 51]

OpenReview RSS

arXiv

l Titles l Authors l Abstracts l Venues

Doi Data Open Alex Crossref

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Core Direction Match Must-Read Hit Rate Interest Drift State Detection

[Figure 57]

[Figure 58]

[Figure 59]

Initial Core Directions Must-Read Rules Interest Vectors

Natural language description

Profile

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Generated Report Quality

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Representative paper

Personal web page

[Figure 74]

Reading Engagement Analysis

[Figure 75]

[Figure 76]

###### 2. User Profile Representation Multi-grained structure

###### 2. Multi-module Evaluation

[Figure 77]

[Figure 78]

[Figure 79]

Interest Drift Sensing

[Figure 80]

[Figure 81]

like dislike

[Figure 82]

Key Words

[Figure 83]

Must-Read Rules(fixed)

实时发布成果产出、团队荣誉与学术交流信息，展现实验室科研活力与品牌影 响力的动态窗口。

Core Directions Weights

口。

[Figure 84]

PaperFlow System

[Figure 85]

correcting a group

Topic Weights Author/Affiliation Heat

Interest Vector(drift-aware) Reading Peport Preference

[Figure 86]

Content Match

stable,shifting,recovered states

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Text input...

###### 3. Subjective Assessment(1-5)

###### 3. Personalized Recommendation & Ranking

Structured Features

[Figure 92]

Must-Read Hit (TOP,Stable)

Semantic Embeddings

[Figure 93]

Relevancy Novelty Report Usefulness Explanation Credibility Profile Consistency

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

High Revelancy

[Figure 98]

Topic Tags

Ranking module

[Figure 99]

May Interested

Author Fields

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

selected paper source page OCR Fallback report chart

Marginal

Quality

Figure 2: Overview of the PaperFlow dynamic personalized scientific reading loop.

###### 3.2 Structured Scholarly Profile

Let t index a daily recommendation round, i.e., day t. PaperFlow maintains an editable and continuously updated structured scholarly profile pt at the beginning of day t, decomposed as follows:

pt = {Dt, Tt, At, It, Mt, vt, τt, Qt, Bt, dt}, (1)

where Dt, Tt, vt capture research directions, topic weights, and semantic interest vectors; At, It, Mt encode author, institution, and must-read priors; τt stores method and paper-type preferences; and Qt, Bt, dt represent report preferences, reading-behavior signals, and the drift-adaptation state. We separate Qt from Bt so that report-style feedback does not contaminate research-interest weights.

When no historical feedback exists, PaperFlow builds p0 from multi-source cold-start evidence, including research descriptions, profile pages, representative papers or PDFs, and manual preferences. The benchmark uses reproducible simulated profiles, while these sources describe broader cold-start inputs supported by the method.

Cold-start construction extracts and canonicalizes directions, methods, application contexts, authors, institutions, and preference clues from heterogeneous sources. The LLM supports extraction and canonicalization; repeated support raises initial weight or confidence, explicit inputs remain inspectable rules or fields, and all outputs must fit fixed profile fields and pass structural validation before entering shared state.

###### 3.3 Daily Updating

On day t, PaperFlow ranks the date-specific candidate pool Ct with the current profile pt, aggregating long-term interest matching, prior signals, dynamic behavioral signals, and explicit rules.

For a candidate paper c, the ranking signal is decomposed as:

score(c, pt) = Smatch(c, pt) + Sprior(c, pt)

+ Sdyn(c, pt) + Smust(c, pt).

(2)

The terms group matching, priors, dynamics, and must-read rules: Smatch covers semantic-interest similarity and topic matching; Sprior covers author, institution, and candidate-paper quality; Sdyn covers drift state, anchor directions, old-topic suppression, and recent reading behavior; and Smust covers must-read rules. The feature weights and normalization are in Appendix C.1. Candidate-paper quality in Sprior is distinct from profile-local Qt.

Recommendations use four system-side display and diagnostic tiers: must_read, high_relevant, maybe_interested, and edge_relevant. Non-must-read tiers use score and rank-aware thresholds; must-read matches receive a small bonus while preserving personalized relevance.

Feedback updating keeps signal semantics separate: selections, corrections, and profile edits provide strong interest evidence; skips provide weak negative evidence; reading requests, PDF uploads, and repeated reading update Bt; and report feedback updates only Qt. Research-interest updates mainly use selections, sustained reading, explicit edits, and cross-day drift evidence.

We abstract feedback-driven updating as:

pt+1 = U(pt, Ft, Bt, dt), (3)

where pt+1 is the next-round profile, U is the update process, Ft is daily feedback, Bt is the readingbehavior state, and dt is the adaptation strength between long- and short-term evidence.

###### 3.4 Behavior-Driven Interest Drift

PaperFlow uses behavioral evidence to distinguish transient exploration from sustained interest migration. Repeated selections, explicit requests, intensive reading, PDF uploads, and manual profile edits can support a new direction, while unsupported old directions are gradually decayed in the profile and ranking process.

Let Lt be the long-term topic distribution, St the recent short-term topic distribution, Ft the feedback set for the day, and Bt the reading-behavior state. PaperFlow computes an internal drift-evidence signal:

gt = Drift(Lt, St, Ft, Bt), (4) which measures the deviation between long-term interest and recent behavior and controls drift-state transitions and profile-update strength. It increases under sustained evidence for a low-weight new direction and decreases when evidence weakens or realigns with the long-term profile. To prevent a single interaction from causing profile jumps, PaperFlow bounds the per-round update of each topic direction z:

|Tt+1(z) − Tt(z)| ≤ ϵ, (5) where ϵ is the per-round upper bound on topic-weight change.

PaperFlow uses four drift states. Stable means recent behavior matches the long-term profile; Observing marks an emerging direction with insufficient evidence; Shifting increases sustained new-topic weight under the update constraint; and Recovered rebalances new and old interests after confirmed migration. Transitions are not triggered by a single click; they depend on drift-evidence thresholds, sustained evidence windows, and directional consistency. Detailed threshold and window settings are given in Appendix C.3.

###### 4 PaperFlow Benchmark

###### 4.1 Benchmark Construction

As shown in Figure 3, PaperFlow is constructed as a longitudinal reading pipeline rather than a static set of user–paper pairs. The pipeline freezes daily paper pools, initializes dynamic researcher profiles,

[Figure 106]

[Figure 107]

- 1. Paper Stream & Candidate Pool Daily paper pool Current frozen benchmark

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

interest

[Figure 112]

50 days

[Figure 113]

time(days)

[Figure 114]

- 2. Dynamic User Profiles

[Figure 115]

PaperFlow Dataset Construction Pipeline

[Figure 116]

[Figure 117]

- 3. Daily Recommendation Episodes

[Figure 118]

Top-20 shown selected papers

[Figure 119]

[Figure 120]

[Figure 121]

1,200 user-day episodes

Top-20 shown selected papers

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

- 4. Reading Reports & Profile Update 3,104 reports

Clean Benchmark Export users.json episodes.jsonl episode.papers.jsonl profiles.jsonl

[Figure 126]

Why PaperFlow?

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Dynamic profiles

[Figure 132]

Evaluation Signals

[Figure 133]

[Figure 134]

[Figure 135]

aracle_label shown selected

[Figure 136]

Interest drift

[Figure 137]

[Figure 138]

system_rank report_source

[Figure 139]

[Figure 140]

###### Quality Control

[Figure 141]

frozen pools clean inputs

[Figure 142]

[Figure 143]

[Figure 144]

Recommendation

profile/drift update

[Figure 145]

[Figure 146]

top-20 budget

+Deep reading

[Figure 147]

reproducible scripts

[Figure 148]

Users

Days

Episodes

Paper pool

Episode-paper rows Shown

Selected

PDF success

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

3,104

24

50

1200

20,727 497,448

24,000

99.0%

- Figure 3: Construction pipeline of the PaperFlow benchmark. Daily paper streams and simulated researcher profiles are converted into date-frozen user-day episodes, providing clean method inputs alongside hidden pseudo-oracle labels, reading-report records, and drift diagnostics.

executes daily recommendation episodes, records reading and profile-update traces, and exports clean benchmark files with evaluation signals kept separate from method inputs.

Paper stream and candidate pool. PaperFlow converts the arXiv daily paper stream into date-indexed candidate pools. Each date-specific pool contains only papers visible that day and is frozen before evaluation, strictly isolating ranking from temporal data leakage and eliminating variation from dynamic crawling or later metadata updates.

Dynamic user profiles. To establish reproducible cold-start conditions, the pipeline initializes 24 simulated researcher profiles spanning scientific domains. Each profile contains structured interests, preferences, and drift plans, so episodes can test both stable personalization and controlled interest migration without relying on private user logs.

Daily recommendation episodes. The benchmark then rolls profiles through the frozen paper stream to produce user-day episodes. For each episode, a method observes the allowed pre-ranking state and returns a Top-20 list from Ct; the simulator records shown papers, selected papers, hidden relevance labels, system ranks, and reading signals for later analysis and state updates.

Reading reports, profile updates, and export. Selected papers trigger reading-report generation and profile/drift updates, producing the downstream traces needed to study continuous use. The final export separates clean inputs from oracle labels, shown/selected flags, report sources, and qualitycontrol fields, enforcing a strict temporal boundary for comparable evaluation. Appendix A gives the complete field specification.

###### 4.2 Benchmark Analysis

- Figure 3 summarizes the scale of the frozen snapshot. PaperFlow contains 24 simulated researchers, 50 daily paper pools, and 1,200 user-day episodes. The 20,727-paper pool expands to 497,448 episode–

Table 1: Main results on the PaperFlow benchmark. Automatic metrics are computed on the full 1,200 user-day episodes with a Top-20 display budget. All scores are reported on a 0–100 scale; ratio metrics are multiplied by 100. HumanEval is a blind listwise human score on sampled Top-20 lists.

gNDCG @20

Useful @5

Useful @20

Selected NDCG@20

Rec. Score

Method

###### HumanEval

Scholar Inbox 39.00 25.92 14.63 33.47 46.30 55.56 Citation-Enhanced 32.67 24.45 12.68 34.24 42.34 44.44 OMRC-MR 19.95 15.52 8.68 25.59 27.37 30.00 UPR 37.54 26.43 13.85 29.74 43.70 53.33 KUCNet 33.14 23.93 12.69 33.47 42.27 35.56

PaperFlow 50.65 34.90 17.56 70.88 55.31 65.56

paper records, creating a daily filtering problem in which hundreds of candidates must be compressed into a Top-20 reading list. A Top-20 list shows only about 4.8% of an average daily pool, and the simulator selects 3,104 of the 24,000 shown papers for deeper reading. This creates sparse but temporally ordered feedback: most papers are merely skipped, while selected papers generate reading reports and profile-update evidence.

###### 4.3 Evaluation Protocol and Labels

PaperFlow evaluates personalized scientific reading as strictly sequential user-day ranking. For each simulated user and date, a method produces:

Rt = Alg(u, t, xu,Ct, F<t, qu,<t), s.t. |Rt| = 20, Rt ⊆ Ct,

(6)

where xu is visible user metadata, Ct is the date-frozen candidate pool, F<t contains only earlier feedback, and qu,<t is the pre-ranking dynamic state. Same-day selections, oracle labels, and drift outcomes remain hidden until evaluation.

All methods rank the same users, dates, and candidate pools under a fixed-round offline protocol. Automatic evaluation uses controlled pseudo-oracle relevance labels (strong_relevant, relevant, weak_relevant, irrelevant); these labels provide reproducible evaluation targets but are not human ground truth.

We report three complementary metric families: (1) oracle-based ranking metrics (e.g., gNDCG@20, OracleRecall@20, MRR@20), summarized by a composite RecommendationScore ranging from 0 to 100; (2) SelectedNDCG@20, measuring agreement with simulated reading selections; and (3) HumanEval, a blind listwise score over sampled anonymized Top-20 lists to validate automatic–human alignment. All metrics are computed strictly after recommendation and never exposed as same-day inputs. Detailed definitions and annotator agreement are in Appendices D and F.

###### 5 Experiments

###### 5.1 Baselines

We compare PaperFlow with five representative scientific-paper recommendation baselines: Scholar Inbox [4], Citation-Enhanced Literature [17], OMRC-MR Content [28], Natural-Language User Profile [21], and KUCNet Enhanced Recommendation [16]. All methods rank the same candidate pools under the same Top-20 budget. The compared methods differ in whether they support cold-start profiling, profile updating, interest drift, feedback use, and reading-report assistance, summarized in Appendix A.5.

###### 5.2 Main Recommendation Results

- Table 1 shows that PaperFlow achieves the strongest automatic performance and the highest blind listwise human score. Compared with Scholar Inbox, the strongest external baseline by RecommendationScore, PaperFlow improves gNDCG@20 from 39.00 to 50.65, RecommendationScore from 46.30 to 55.31, and HumanEval from 55.56 to 65.56.

The largest qualitative difference appears in behavior alignment: SelectedNDCG@20 increases from 33.47 to 70.88. This suggests that PaperFlow not only retrieves relevant papers under hidden labels, but also ranks papers closer to those the simulated user later chooses for reading. Other baselines recover useful signals in specific settings, but remain weaker on overall ranking quality and downstream behavioral alignment.

5.3 LLM Comparison Results

We compare LLM backbones within PaperFlow, using Gemini 3 Flash as the default. The frozen benchmark, user profiles, candidate pools, Top-20 budget, embedding model, and metric computation are fixed; only the LLM used for structured judgment, recommendation explanation, feedback parsing, and reading-report generation is changed. We report quality, human alignment, and cost separately: ModelAutoScore combines recommendation and report quality, ModelHumanScore provides the corresponding blind human score. Full definitions are given in Appendix D.10, D.11, and F.

Table 2: LLM backbone comparison under PaperFlow.

Group Model

Rec. Score

Report Auto

Model Auto

Model Human

Closed

GPT-5.4 54.77 99.31 63.68 69.41

- Qwen3.5-Plus 55.23 99.18 64.02 74.07 Gemini 3.1 Pro 55.09 99.10 63.89 70.30 Claude Sonnet 4.6 55.36 99.28 64.15 82.07
- Qwen3.6-Plus 55.22 99.80 64.14 77.11 Qwen3.6-Max 55.11 99.21 63.93 74.15 Grok 4.3 56.31 99.56 64.96 94.07 Gemini 3 Flash 55.31 99.70 64.19 76.74

Open

MiMo-V2.5-Pro 54.89 99.94 63.90 73.93 DeepSeek-V4-Pro 55.16 99.52 64.04 76.44 DeepSeek-V4-Flash 55.43 99.95 64.34 82.15 Kimi K2.6 55.91 99.82 64.69 86.67 GLM-5.1 55.03 99.77 63.98 75.41 MiniMax-M2.7 55.48 99.19 64.22 81.78

- Table 2 shows that model choice materially affects both automatic and human-aligned quality under identical retrieval, embedding, and evaluation conditions. Among closed-source models, Grok 4.3 achieves the highest RecommendationScore (56.31) and ModelHumanScore (94.07) while also maintaining the lowest token cost (Figure 5), making it the most cost-effective closed-source option. Gemini 3 Flash offers the best overall balance as default backbone with strong ModelAutoScore (64.19) and competitive human alignment.

- Figure 4 plots ModelAutoScore against ModelHumanScore for all backbones, yielding Pearson’s r=0.9632. This strong correlation validates automatic metrics as a reliable proxy for human judgment and suggests that improvements observed in automatic evaluation translate consistently to perceived recommendation quality. Figure 5 reports token cost in millions across all LLM backbones. The results suggest that token efficiency is largely model-dependent rather than correlated with quality.

[Figure 156]

- Figure 4: Automatic–human metric alignment (ModelAutoScore vs. ModelHumanScore).

[Figure 157]

Figure 5: Token-cost across LLM backbones. Bar values are reported in million tokens; lower is better.

- Table 3: Ablation and diagnostic results. Simplified variants can improve oracle-based relevance concentration, while SelectedNDCG@20 measures alignment with simulated downstream reading selections. Full PaperFlow performs best on this behavior-alignment metric.

gNDCG @20 ↑

Useful @5 ↑

Useful @20 ↑

Oracle Rec@20 ↑

Lift @20 ↑

StrictR @20+ ↑

MRR @20 ↑

Selected NDCG@20 ↑

Rec. Score ↑

Method

PaperFlow 0.5065 0.3490 0.1756 0.2232 12.73 0.8613 0.6044 0.7088 55.31 Fixed Profile 0.5346 0.3710 0.1877 0.2302 13.36 0.8881 0.6265 0.6955 57.81 w/o Explicit Pref. 0.5148 0.3643 0.1731 0.1971 12.86 0.7607 0.6435 0.6659 54.37 w/o Drift 0.5147 0.3543 0.1804 0.2298 12.95 0.8869 0.6066 0.7039 56.36 w/o Reading Signal 0.5126 0.3537 0.1786 0.2278 12.90 0.8789 0.6053 0.7074 56.06

###### 5.4 Ablation Analysis

- Table 3 separates oracle-based relevance from behavioral alignment. Several simplified variants obtain higher oracle-based scores than Full PaperFlow, because removing profile updates, drift modeling, or reading signals makes the system more conservative and closer to stable pseudo-oracle labels.

In contrast, Full PaperFlow achieves the best SelectedNDCG@20, showing stronger agreement with simulated downstream reading selections. The ablation results therefore reveal a static–dynamic trade-off: simplified variants can improve static relevance concentration, while PaperFlow’s adaptive components better track what users choose to read over time.

###### 5.5 Interest Drift Analysis

We evaluate PaperFlow on controlled interest-drift episodes, comparing PaperFlow with w/o Drift and Fixed Profile. The goal is to test whether the system exposes new-interest papers, suppresses stale old-topic exposure, adapts quickly, and remains aligned with later reading choices.

Figure 6 shows that w/o Drift and Fixed Profile can slightly improve some PostDrift oracle metrics, but PaperFlow is strongest on adaptation-oriented signals: it achieves the highest PostDrift SelectedNDCG@20, the highest NewTopicRecall@20, the lowest OldTopicRate@20, and the shortest adaptation delay. PaperFlow also has the best DriftAutoScore (72.76) and AdaptationHumanScore (68.75), indicating that the drift module improves adaptation rather than merely maximizing static oracle relevance. The drift-score and human-evaluation protocol are in Appendix D.12 and Appendix F.

[Figure 158]

Figure 6: Interest-drift analysis. Cell color is normalized within each metric, with darker green indicating better performance. PostDrift metrics are computed on the post-drift window; NewTopicR, OldTopicR, AdaptDelay, DriftAuto, and AdaptHuman summarize adaptation-oriented behavior.

###### 5.6 Real-User Pilot Study

We conduct a real-user pilot study with five graduate students in computer science. Each participant uses PaperFlow for 5–7 interaction rounds over daily paper pools. The study records actual reading decisions and post-round Likert ratings; participant information, metric definitions, questionnaire items, and per-user results are provided in Appendix G. As shown in Table 4, the real-user results mirror the simulated benchmark: PaperFlow consistently outperforms baselines in both actual readingselection rates and user satisfaction.

- Table 4: Real-user results averaged over five participants. Precision@k and ReadRate are computed from actual reading decisions; Satisfaction is a 1–5 Likert score.

Method Prec@5 Prec@20 ReadRate Sat. Daily arXiv Email 0.40 0.26 0.18 2.8 Static Profile 0.58 0.38 0.27 3.5 PaperFlow 0.71 0.47 0.34 4.0

###### 5.7 Case Study

We present a representative interaction to illustrate how PaperFlow operates as a closed-loop reading assistant rather than a one-shot ranker. Figure 7 shows four stages in one daily use case: cold-start profile construction, daily paper push, user feedback, and reading-report generation. The example demonstrates how user feedback is converted into reading signals and then carried forward as context for the next recommendation round.

PaperFlow converts research-interest descriptions into an editable scholarly profile. It then filters the daily paper stream into a compact candidate list, accepts feedback through paper numbers, and generates reading reports for the selected papers.

###### 6 Conclusion

We introduced PaperFlow for personalized paper reading, combining structured scholarly profiles, multi-signal daily ranking, feedback-driven state updates, and interest-drift modeling. The benchmark fixes longitudinal user-day episodes, candidate pools, hidden relevance labels, and diagnostic trajectories under a shared temporal boundary.

###### Limitations

PaperFlow primarily uses simulated research users and simulated relevance labels. Although we include a real-user pilot study, it is intended as a user-experience sanity check. This design provides

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

###### 1.proflie 2. Daily Push

[Figure 164]

My research interests are NLP, RAG, and information extraction.

[Figure 165]

Push today's papers.

[Figure 166]

[Figure 167]

Scholarly profile (cold start): core = NLP 0.80;RAG 0.70; information extraction 0.60. Preferences: data-driven, systematic,open-source, language & knowledge applications. Please confirm or edit ...

Apr. 21: crawled 256 papers

->filtered 20. Must-read hits:01 AIQuraishi, 02 Jian Tang,03 Shanghai AI Lab... Highly relevant: MIT, Stanford,Google, CMU... Reply with paper numbers.

[Figure 168]

[Figure 169]

###### 3. Feedback 4. Reading Report

[Figure 170]

1-5 6 9 11 15

[Figure 171]

Reading reports generated:

- 1. Geometric Pretraining forProtein Complexes [link];
- 2. Scaling Molecular Generation with Flow Matching [link];

Received. 9 papers enteredthe deep-reading queue.Reports will be generatedand sent as document links.

[Figure 172]

... context updated for thenext daily push.

[Figure 173]

proflie push feedback report next context

[Figure 174]

Figure 7: Representative PaperFlow case study.

[Figure 175]

reproducibility and controlled temporal comparison, but the simulated labels should not be interpreted as human-annotated truth or as deployment logs from real users. The current benchmark is mainly derived from arXiv daily paper streams, so coverage may differ across fields and publication venues. Future work should connect the protocol with larger-scale human evaluation, additional scholarly sources, and deployment studies.

###### References

- [1] Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, Amanpreet Singh, Joseph Chee Chang, Kyle Lo, Luca Soldaini, Sergey Feldman, Mike D’Arcy, David Wadden, Matt Latzke, Jenna Sparks, Jena D. Hwang, Varsha Kishore, Minyang Tian, Pan Ji, Shengyan Liu, Hao Tong, Bohao Wu, Yanyu Xiong, Luke Zettlemoyer, Graham Neubig, Daniel S. Weld, Doug Downey, Wen-Tau Yih, Pang Wei Koh, and Hannaneh Hajishirzi. Synthesizing Scientific Literature with Retrieval-Augmented Language Models. Nature, 650(8103):857–863, 2026. doi: 10.1038/s41586-025-10072-4. URL https://www.nature.com/articles/s41586-025-10072-4.
- [2] Shihao Cai, Jizhi Zhang, Keqin Bao, Chongming Gao, Qifan Wang, Fuli Feng, and Xiangnan He. Agentic Feedback Loop Modeling Improves Recommendation and User Simulation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2235–2244,

2025. doi: 10.1145/3726302.3729893. URL https://doi.org/10.1145/3726302.3729893.

- [3] Jiabao Fang, Shen Gao, Pengjie Ren, Xiuying Chen, Suzan Verberne, and Zhaochun Ren. A Multi-Agent Conversational Recommender System. arXiv preprint arXiv:2402.01135, 2024. URL https://arxiv.org/abs/ 2402.01135.
- [4] Markus Flicke, Glenn Angrabeit, Madhav Iyengar, Vitalii Protsenko, Illia Shakun, Jovan Cicvaric, Bora Kargi,

- Haoyu He, Lukas Schuler, Lewin Scholz, Kavyanjali Agnihotri, Yong Cao, and Andreas Geiger. Scholar Inbox: Personalized Paper Recommendations for Scientists. In ACL 2025 System Demonstrations, 2025. URL https://aclanthology.org/2025.acl-demo.30/.
- [5] Raymond Fok, Hita Kambhamettu, Luca Soldaini, Jonathan Bragg, Kyle Lo, Andrew Head, Marti A. Hearst, and Daniel S. Weld. Scim: Intelligent Skimming Support for Scientific Papers. In IUI 2023, 2023. doi: 10.1145/3581641.3584034. URL https://doi.org/10.1145/3581641.3584034.
- [6] Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System. arXiv preprint arXiv:2303.14524, 2023. URL https://arxiv.org/abs/2303.14524.
- [7] Yichen He, Guanhua Huang, Peiyuan Feng, Yuan Lin, Yuchen Zhang, Hang Li, and Weinan E. PaSa: An LLM Agent for Comprehensive Academic Paper Search. In ACL 2025 Long Papers, 2025. URL https: //aclanthology.org/2025.acl-long.572/.
- [8] Jiani Huang, Shijie Wang, Liang bo Ning, Wenqi Fan, Shuaiqiang Wang, Dawei Yin, and Qing Li. Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs (RecBench+). In WSDM 2026, 2026. doi: 10.1145/3773966.3777954. URL https://doi.org/10.1145/ 3773966.3777954.
- [9] Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, and Xing Xie. Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations. ACM Transactions on Recommender Systems, 2025. doi: 10.1145/3731446. URL https://doi.org/10.1145/3731446.
- [10] Hyeonsu B. Kang, Sherry Tongshuang Wu, Joseph Chee Chang, and Aniket Kittur. Synergi: A MixedInitiative System for Scholarly Synthesis and Sensemaking. In UIST 2023, 2023. doi: 10.1145/3586183.3606759. URL https://doi.org/10.1145/3586183.3606759.
- [11] Yoonjoo Lee, Hyeonsu B. Kang, Matt Latzke, Juho Kim, Jonathan Bragg, Joseph Chee Chang, and Pao Siangliulue. PaperWeaver: Enriching Topical Paper Alerts by Contextualizing Recommended Papers with User-collected Papers. In CHI 2024, 2024. doi: 10.1145/3613904.3642196. URL https://doi.org/10.1145/ 3613904.3642196.
- [12] Chuang Li, Yang Deng, Hengchang Hu, Min-Yen Kan, and Haizhou Li. Incorporating External Knowledge and Goal Guidance for LLM-based Conversational Recommender Systems. In Findings of NAACL 2025, 2025. URL https://aclanthology.org/2025.findings-naacl.17/.
- [13] Yuan Liang, Jiaxian Li, Yuqing Wang, Piaohong Wang, Motong Tian, Pai Liu, Shuofei Qiao, Runnan Fang, He Zhu, Ge Zhang, Minghao Liu, Yuchen Eleanor Jiang, Ningyu Zhang, and Wangchunshu Zhou. Towards Personalized Deep Research: Benchmarks and Evaluations (PDR-Bench). In ICLR 2026 Poster, 2026. URL https://openreview.net/forum?id=51LIRzF53v.
- [14] Guanyu Lin, Tao Feng, Pengrui Han, Ge Liu, and Jiaxuan You. Arxiv Copilot: A Self-Evolving and Efficient LLM System for Personalized Academic Assistance. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 122–130, 2024. doi: 10.18653/v1/2024. emnlp-demo.13. URL https://aclanthology.org/2024.emnlp-demo.13/.
- [15] Xiaolin Lin, Weike Pan, and Zhong Ming. Towards Interest Drift-driven User Representation Learning in Sequential Recommendation (IDURL). In SIGIR 2025, 2025. doi: 10.1145/3726302.3730099. URL https: //doi.org/10.1145/3726302.3730099.
- [16] Guangyi Liu, Quanming Yao, Yongqi Zhang, and Lei Chen. Knowledge-Enhanced Recommendation with User-Centric Subgraph Network. arXiv preprint arXiv:2403.14377, 2024. URL https://arxiv.org/abs/2403. 14377.
- [17] Kun Liu, Yan Zhang, Rui Pan, Tianchen Gao, and Hansheng Wang. Academic Literature Recommendation in Large-scale Citation Networks Enhanced by Large Language Models. arXiv preprint arXiv:2503.01189,

2025. URL https://arxiv.org/abs/2503.01189.

- [18] Kyle Lo, Joseph Chee Chang, Andrew Head, Jonathan Bragg, Amy X. Zhang, Cassidy Trier, Chloe Anastasiades, Tal August, Russell Authur, Danielle Bragg, Erin Bransom, Isabel Cachola, Stefan Candra, Yoganand Chandrasekhar, Yen-Sung Chen, Evie Yu-Yen Cheng, Yvonne Chou, Doug Downey, Rob Evans, Raymond

- Fok, Fangzhou Hu, Regan Huff, Dongyeop Kang, Tae Soo Kim, Rodney Kinney, Aniket Kittur, Hyeonsu Kang, Egor Klevak, Bailey Kuehl, Michael Langan, Matt Latzke, Jaron Lochner, Kelsey MacMillan, Eric Marsh, Tyler Murray, Aakanksha Naik, Ngoc-Uyen Nguyen, Srishti Palani, Soya Park, Caroline Paulic, Napol Rachatasumrit, Smita Rao, Paul Sayre, Zejiang Shen, Pao Siangliulue, Luca Soldaini, Huy Tran, Madeleine van Zuylen, Lucy Lu Wang, Christopher Wilhelm, Caroline Wu, Jiangjiang Yang, Angele Zamarron, Marti A. Hearst, and Daniel S. Weld. The Semantic Reader Project: Augmenting Scholarly Documents through AI-Powered Interactive Reading Interfaces. Communications of the ACM, 2024. doi: 10.1145/3654777. URL https://dl.acm.org/doi/10.1145/3654777.
- [19] Chenkai Pan, Xinglong Xu, Yuhang Xu, Yujun Wu, Siyuan Li, Jintao Chen, Conghui He, Jingxuan Wei, and Cheng Tan. Programming with data: Test-driven data engineering for self-improving llms from raw corpora. arXiv preprint arXiv:2604.24819, 2026.
- [20] Napol Rachatasumrit, Jonathan Bragg, Amy X. Zhang, and Daniel S Weld. CiteRead: Integrating Localized Citation Contexts into Scientific Paper Reading. In IUI 2022, 2022. doi: 10.1145/3490099.3511162. URL https://doi.org/10.1145/3490099.3511162.
- [21] Jerome Ramos, Hossen A. Rahmani, Xi Wang, Xiao Fu, and Aldo Lipani. Transparent and Scrutable Recommendations Using Natural Language User Profiles. In ACL 2024 Long Papers, 2024. URL https: //aclanthology.org/2024.acl-long.753/.
- [22] Yu Shang, Peijie Liu, Yuwei Yan, Zijing Wu, Leheng Sheng, Yuanqing Yu, Chumeng Jiang, An Zhang, Fengli Xu, Yu Wang, Min Zhang, and Yong Li. AgentRecBench: Benchmarking LLM Agent-based Personalized Recommender Systems. In NeurIPS 2025 Datasets and Benchmarks Track Spotlight, 2025. URL https:// openreview.net/forum?id=fm77rDf9JS.
- [23] Wentao Shi, Xiangnan He, Yang Zhang, Chongming Gao, Xinyue Li, Jizhi Zhang, Qifan Wang, and Fuli Feng. Large Language Models are Learnable Planners for Long-Term Recommendation. In SIGIR 2024, 2024. doi: 10.1145/3626772.3657683. URL https://doi.org/10.1145/3626772.3657683.
- [24] Yubo Shu, Haonan Zhang, Hansu Gu, Peng Zhang, Tun Lu, Dongsheng Li, and Ning Gu. RAH! RecSys– Assistant–Human: A Human-Centered Recommendation Framework With LLM Agents. IEEE Transactions on Computational Social Systems, 11(5):6759–6770, 2024. doi: 10.1109/TCSS.2024.3404039. URL https://doi. org/10.1109/TCSS.2024.3404039.
- [25] Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J. Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, and Andrew D. White. Language agents achieve superhuman synthesis of scientific knowledge. arXiv preprint arXiv:2409.13740, 2024. URL https://arxiv.org/abs/2409. 13740.
- [26] Ruotong Wang, Xinyi Zhou, Lin Qiu, Joseph Chee Chang, Jonathan Bragg, and Amy X. Zhang. PaperPing: A Socially-aware AI Agent that Recommends Academic Papers to Research Group Chats with Contextualized Explanations. In CSCW Companion 2025, 2025. doi: 10.1145/3715070.3757230. URL https://doi.org/10. 1145/3715070.3757230.
- [27] Ruotong Wang, Xinyi Zhou, Lin Qiu, Joseph Chee Chang, Jonathan Bragg, and Amy X. Zhang. SocialRAG: Retrieving from Group Interactions to Socially Ground AI Generation. In CHI 2025, 2025. doi: 10.1145/3706598.3713749. URL https://doi.org/10.1145/3706598.3713749.
- [28] Shenghua Wang and Zhen Yin. Discourse-Aware Scientific Paper Recommendation via QA-Style Summarization and Multi-Level Contrastive Learning. arXiv preprint arXiv:2511.03330, 2025. URL https: //arxiv.org/abs/2511.03330.
- [29] Xintao Wang, Jiangjie Chen, Nianqi Li, Lida Chen, Xinfeng Yuan, Wei Shi, Xuyang Ge, Rui Xu, and Yanghua Xiao. SurveyAgent: A Conversational System for Personalized and Efficient Research Survey. arXiv preprint arXiv:2404.06364, 2024. URL https://arxiv.org/abs/2404.06364.
- [30] Yancheng Wang, Ziyan Jiang, Zheng Chen, Fan Yang, Yingxue Zhou, Eunah Cho, Xing Fan, Xiaojiang Huang, Yanbin Lu, and Yingzhen Yang. RecMind: Large Language Model Powered Agent For Recommendation. In Findings of NAACL 2024, 2024. URL https://aclanthology.org/2024.findings-naacl.271/.

- [31] Jingxuan Wei, Xi Bai, Shan Liu, Caijun Jia, Zheng Sun, Xinglong Xu, Siyuan Li, Linzhuang Sun, Bihui Yu, Conghui He, et al. Pager: Bridging the semantic-execution gap in point-precise geometric gui control. arXiv preprint arXiv:2605.15963, 2026.
- [32] Jingxuan Wei, Siyuan Li, Yuhang Xu, Zheng Sun, Junjie Jiang, Hexuan Jin, Caijun Jia, Honghao He, Xinglong Xu, Chang Yu, et al. The trinity of consistency as a defining principle for general world models. arXiv preprint arXiv:2602.23152, 2026.
- [33] Zhelin Xu, Shuhei Yamamoto, and Hideo Joho. Research Paper Recommender System by Considering Users’ Information Seeking Behaviors. In JSAI 2025, 2025. URL https://www.jstage.jst.go.jp/article/pjsai/ JSAI2025/0/JSAI2025_1D4OS24b02/_article/-char/en.
- [34] Hyunsik Yoo, SeongKu Kang, Ruizhong Qiu, Charlie Xu, Fei Wang, and Hanghang Tong. Embracing Plasticity: Balancing Stability and Plasticity in Continual Recommender Systems. In SIGIR 2025, 2025. doi: 10.1145/3726302.3729964. URL https://doi.org/10.1145/3726302.3729964.
- [35] Bihui Yu, Xinglong Xu, Junjie Jiang, Jiabei Cheng, Caijun Jia, Siyuan Li, Conghui He, Jingxuan Wei, and Cheng Tan. Paperfit: Vision-in-the-loop typesetting optimization for scientific documents. arXiv preprint arXiv:2605.10341, 2026.
- [36] Junjie Zhang, Yupeng Hou, Ruobing Xie, Wenqi Sun, Julian McAuley, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems. In The Web Conference 2024 / WWW 2024, 2024. doi: 10.1145/3589334.3645537. URL https: //doi.org/10.1145/3589334.3645537.
- [37] Yuyue Zhao, Jiancan Wu, Xiang Wang, Wei Tang, Dingxian Wang, and Maarten de Rijke. Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning. In SIGIR 2024, 2024. doi: 10.1145/3626772.3657828. URL https://doi.org/10.1145/3626772.3657828.

### Appendix

###### Appendix Table of Contents

###### A Benchmark and Data Setting 18

- A.1 Benchmark Scope . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.2 Episode and Subsession Definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.3 Recommendation Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- A.4 Fixed Data and Fair Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- A.5 Capability Coverage of Compared Methods . . . . . . . . . . . . . . . . . . . . . . . . . . 20

###### B User Profiles 20

- B.1 Profile Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- B.2 The 24 Simulated Researchers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

###### C Hyperparameters 21

- C.1 Ranking Score Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.2 Explicit Preference Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- C.3 Interest-Drift Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- C.4 Reading-Signal Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- C.5 Reading-Report Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- C.6 LLM and Embedding API Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

###### D Evaluation Metrics 25

- D.1 Main Metric Suite . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- D.2 gNDCG@20 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- D.3 Useful@5 and Useful@20 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- D.4 OracleRecall@20 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- D.5 Lift@20 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- D.6 StrictR@20+ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- D.7 MRR@20 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- D.8 SelectedNDCG@20 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- D.9 RecommendationScore . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- D.10 ReportAutoScore . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- D.11 ModelAutoScore . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- D.12 DriftAutoScore . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- D.13 Relationship Among Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

###### E Prompt Templates 29

- E.1 How Prompts Are Used . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- E.2 User-Profile Parsing Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- E.3 Research-Direction Extraction Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- E.4 Recommendation-Explanation Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- E.5 Reading-Report Generation Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- E.6 Reading-Report Markdown Template . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- E.7 Human-Evaluation Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34

###### F Human Evaluation Protocol 35

- F.1 Main-Experiment Human Evaluation: HumanEval . . . . . . . . . . . . . . . . . . . . . . . 35
- F.2 Model-Comparison Human Evaluation: ModelHumanScore . . . . . . . . . . . . . . . . . 36
- F.3 Drift-Specific Human Evaluation: AdaptationHumanScore . . . . . . . . . . . . . . . . . . 36
- F.4 Inter-Annotator Agreement and Correlation Analysis . . . . . . . . . . . . . . . . . . . . 37

###### G Real-User Pilot Study Details 37

- G.1 Study Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
- G.2 Participants . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- G.3 Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
- G.4 Real-User Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- G.5 Questionnaire . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- G.6 Per-User Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- G.7 Ethics and Anonymization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

###### H Case Studies 40

- H.1 Case-Study Selection Criteria . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
- H.2 Successful Recommendation: A Dense Top-20 for an NLP User . . . . . . . . . . . . . . . 40
- H.3 Interest Drift: From GUI/Web Agents to Multimodal Reasoning . . . . . . . . . . . . . . 40
- H.4 Behavior Consistency: A High-SelectedNDCG List . . . . . . . . . . . . . . . . . . . . . . . 40
- H.5 Failure or Boundary Case: Disagreement Between System Labels and Oracle Labels . . 41
- H.6 Reading Report: PDF-Evidence-Driven Reading Assistance . . . . . . . . . . . . . . . . . 41

###### I Model Comparison Details 41

- I.1 Purpose of Model Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- I.2 Open and Closed Model Scope . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- I.3 Model-Comparison Controls . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- I.4 Token Usage Records . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43

###### J Reproducibility Notes 43

- J.1 Runtime Environment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- J.2 Output Files . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- J.3 Error Handling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43

###### A Benchmark and Data Setting

###### A.1 Benchmark Scope

We use the same PaperFlow benchmark snapshot for the main experiment, baseline comparison, and LLM comparison. During evaluation, the paper pool is not expanded. Instead, all methods operate on a pre-built historical paper database. This design isolates the effect of the recommendation method itself and avoids incomparability caused by paper-pool expansion, retrieval-time differences, or data updates.

Table 5: Benchmark setting used by PaperFlow.

Item Setting Data directory data/benchmark_full_24users_20260301_20260419

_show20_with_reading Date range March 1, 2026 to April 19, 2026 Number of users 24 simulated researchers Number of episodes 1,200 user-day episodes Average candidate-pool size 414.54 papers per episode Display budget Top-20 Average number of selected papers

2.59 papers per episode

Paper-pool policy Fixed paper pool; no new papers are added during evaluation Main-experiment LLM Gemini 3 Flash Preview Embedding model Qwen3-Embedding-8B Random seed 42 Reading reports Enabled for selected papers

Each episode represents the recommendation process for one user on one day. The system first ranks the candidate papers using the user profile, paper content, must-read rules, interest-drift state, and recent reading signals. It then shows the Top-20 papers and simulates the user’s selection of a small number of papers for subsequent reading. Each episode records the user-profile snapshot, candidatepool statistics, Top-20 list, system labels, oracle labels, user selections, drift state, reading-signal state, and token usage.

This fixed-pool design has two advantages. First, it ensures that all methods are compared on exactly the same candidate sets, avoiding data bias introduced by dynamic crawling or real-time retrieval. Second, it preserves temporal order. The system must not only rank a single candidate pool, but also maintain user profiles, react to interest changes, and accumulate reading signals across consecutive days. The benchmark therefore evaluates both static ranking quality and longitudinal user modeling.

###### A.2 Episode and Subsession Definition

We treat each user-day episode as one subsession. A subsession is not a separate model prompt; it is the logging and contextual unit used by the experiment. It contains the full recommendation context for a user on a specific day, including the user profile, candidate pool, Top-20 list, user selections, interest-drift state, and reading reports.

The oracle_label and selected fields are evaluation and logging fields. They are not used directly as model inputs during recommendation. At ranking time, the system can only use the user profile, paper metadata, paper content, explicit preferences, drift state, and historical behavioral signals. This avoids evaluation leakage and ensures that the metrics reflect recommendation ability.

Table 6: Fields recorded in a user-day subsession.

Field Meaning subsession_id Unique identifier composed of user ID and date, e.g., user_-

role6::2026-03-16 date Current episode date user_id Simulated user ID profile_snapshot User-profile snapshot before recommendation candidate_pool Candidate papers for the day shown_list Top-20 recommendation list shown by the system system_label Recommendation label generated by rules and scores oracle_label Offline evaluation label, used only for evaluation and log analysis selected Whether the simulated user selected the paper; used only for behavior

evaluation and later profile updates drift_state Current interest-drift state reading_signal_state Recent reading-signal state reading_report Reading report for a selected paper

- A.3 Recommendation Pipeline Each user-day episode is executed as follows:

- 1. Retrieve the candidate papers for the corresponding date from the fixed paper pool.
- 2. Compute the base relevance score from the user profile and paper content.
- 3. Check must-read author, institution, and keyword matches.
- 4. Apply interest-drift weighting to new interest topics and downweight suppressed old topics.
- 5. Apply recent reading-signal weights to short-term topics that repeatedly appear and are selected by the user.
- 6. Generate the Top-20 recommendation list.
- 7. Simulate user selections based on oracle labels, system rank, system label, and drift-topic matches.
- 8. Generate reading reports for selected papers and record token usage, episode metadata, and the drift timeline.

This pipeline connects recommendation, selection, profile update, and reading assistance into a closed loop. The recommendation stage determines what the user sees. The selection stage simulates what the user actually clicks or reads. The profile-update stage adjusts short-term state based on selections. The reading-report stage produces deeper reading assistance for selected papers. PaperFlow therefore evaluates not only whether the ranking is correct, but also whether the system can maintain user interests, respond to interest changes, and provide explainable reading support during continuous use.

- A.4 Fixed Data and Fair Comparison

All main and related comparison experiments use the same benchmark snapshot. The dataset is fixed for three reasons.

First, paper recommendation is highly sensitive to the candidate pool. If different models face different candidate papers, ranking differences may come from the data distribution rather than from the method. A fixed candidate pool ensures that all methods compete over the same papers.

Table 7: Capability coverage of compared methods. A check mark denotes full coverage, a triangle denotes partial coverage, and a cross denotes absent coverage.

Method CS Prof. Upd. Drift MR Rank Rpt. Fb. Wkly. Impact Behav. Corr.

Scholar Inbox △ ✓ △ × × ✓ × × × × × × Citation-Enhanced × △ × × × ✓ × × × ✓ × × OMRC-MR × × × × × ✓ × × × × × × UPR △ ✓ × × × ✓ × × × × × × KUCNet × △ × × × ✓ × × × △ × × PaperFlow ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

Second, scientific papers are updated daily. If the experiment re-crawls data during evaluation, the runtime of different models would introduce external variation. A fixed historical paper pool avoids this source of bias.

Third, both interest drift and reading signals depend on temporal order. If the paper pool changes across runs, the candidate distribution on later dates may affect drift triggering and recovery. A fixed date range makes longitudinal behavior comparison reproducible.

Thus, all comparable results in the paper use the same fixed-pool setting. In the LLM comparison, only the language model used in recommendation and reading-report stages is replaced. The embedding model, user profiles, candidate pool, Top-20 budget, and metric computation remain fixed.

###### A.5 Capability Coverage of Compared Methods

Table 7 summarizes which system capabilities are covered by each compared method. The table is descriptive rather than an evaluation metric; the main result tables evaluate whether these capabilities improve recommendation quality under the same benchmark setting.

The column abbreviations are: CS = cold start, Prof. = structured profile, Upd. = profile update, Drift = interest-drift handling, MR = must-read rules, Rank = ranked recommendation, Rpt. = reading report, Fb. = feedback use, Wkly. = weekly or longitudinal use, Impact = citation or impact signal, Behav. = behavioral signal, and Corr. = explicit correction support.

###### B User Profiles

###### B.1 Profile Design

The 24 simulated users in PaperFlow cover diverse research directions, including GUI agents, AI for science, literature mining, embodied AI, multimodal reasoning, NLP, bioinformatics, protein structure, single-cell analysis, neuroscience, climate science, materials informatics, chemistry, highenergy physics, medical imaging, public health, agriculture, ocean science, psychology, economics, education research, astronomy, renewable energy, and science of science.

Each user profile contains four types of information: long-term interests, short-term interests, explicit preferences, and behavioral state. Long-term interests represent stable research directions. Short-term interests represent temporary topics formed by recent reading behavior. Explicit preferences represent must-read authors, institutions, or keywords. Behavioral state records interest drift and reading history.

A user profile is not a simple keyword set; it is a multi-layer structure. For example, an NLP user may have nlp, large-language-model, and information-extraction as core directions, and may also include secondary topics such as retrieval-augmented generation, long-context reasoning, and bench-

[Figure 176]

[Figure 177]

###### The 24 Simulated Researcher Profiles

[Figure 178]

[Figure 179]

Why these profiles?

[Figure 180]

AI& reasoning.

[Figure 181]

[Figure 182]

Cold-start diversity

user_role2

user_role4 user_role5

user_role6

user_role3

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

user_role1

[Figure 188]

embodied AI, reinforcement learning,robotics

literature mining, knowledge graph, hypothesis generation

vision-language model, multimodal reasoning, multimodal evaluation

AI for science, scientific reasoning, theorem proving

NLP,large language model,information extraction

GUI agent,computer vision, web agent

[Figure 189]

Start from a wide range of backgrounds to reduce bias and enable diverse interaactions

[Figure 190]

[Figure 191]

Bio& health.

user_role7 user_role8

user_role9

user_role10

user_role15

user_role16

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

protein structure, protein language model, drug discovery

[Figure 198]

systems biology, network biology, bioinformatics

pathology AI, medical imaging, radiology

epidemiology, public health, disease modeling

neuroscience, brain imaging, connectomics

genomics,single-cell analysis,spatial transcriptomics

###### Cross-domain coverage

[Figure 199]

Span major scientific domains to support broad generalization and discovery.

[Figure 200]

Earth& physical.

[Figure 201]

user_role11 user_role12

user_role17

user_role18

user_role13 user_role14

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

climate science, earth-system modeling, remote sensing

materials informatics, crystal structure, property prediction

chemistry,reaction prediction,molecular generation

high-energy physics, particle physics, detector analysis

agriculture,crop science,precision agriculture

oceanography,marine biology,aguatic ecosystems

[Figure 208]

###### Controlled adaptation

Social& science.

[Figure 209]

[Figure 210]

Enable fair comparison and systematic study under controlled adaptation settings.

user_role23

user_role19

user_role20

user_role21

user_role24

user_role22

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

psychology,cognitive science,behavioral science

economics, econometric causal inference

education research, learning science, educational technology

science of science, bibliometrics, research evaluation

renewable energy, battery technology, electrochemistry

astronomy,cosmology, planetary science

Figure 8: The 24 simulated researcher profiles.

mark construction. The system must combine long-term directions, short-term behavior, and explicit rules to decide whether a paper is worth showing to the user.

###### B.2 The 24 Simulated Researchers

- Figure 8 summarizes the 24 simulated users in PaperFlow. Each user has three main directions. The first direction has the largest weight, while the second and third directions create more specific cross-topic interests.

The goal of this user set is not to reproduce the full user distribution of a real platform. Instead, it creates a broad benchmark for testing cross-disciplinary recommendation ability. Compared with evaluation in a single field, the 24-user setup requires the system to handle AI, life science, physics, earth science, medicine, social science, and science-of-science topics, which is closer to the complexity of real scientific-paper recommendation.

###### C Hyperparameters

###### C.1 Ranking Score Parameters

The base PaperFlow ranking score combines semantic matching, profile matching, preference signals, paper quality, drift adaptation, and recent reading feedback. Let I, T, H, and Q denote interest-vector similarity, topic-weight match, author/institution heat, and paper-quality signal, respectively. Let Bm, Bd, and Br denote the must-read, drift, and reading-signal bonuses, and let Ps denote the suppression penalty. The ranking score is defined as

Srank = 0.35I + 0.25T + 0.20H + 0.20Q + Bm + Bd + Br − Ps.

(7)

Table 8: Ranking-score parameters.

Parameter Value Description

- w1_interest_vector 0.35 Semantic similarity between paper content and the user’s interest vector
- w2_topic_weight 0.25 Match between paper topics and user-profile topic weights
- w3_author_institution 0.20 Author, institution, and explicit-source preference weight
- w4_quality_signal 0.20 Paper-quality proxy weight bonus_must_read 0.15 Bonus for a must-read rule match threshold_high_relevant 0.40 Direct threshold for high_relevant threshold_maybe_interested 0.25 Direct threshold for maybe_interested threshold_edge_relevant 0.15 Threshold for edge_relevant min_relevance_signal 0.08 Minimum personal relevance signal before a regular paper

can enter the pushed candidate set rank_high_fraction 0.10 Top 10% of relevant candidates can be promoted to high_relevant if they reach the maybe threshold rank_maybe_fraction 0.40 Top 40% of relevant candidates can be promoted to

maybe_interested if they reach the edge threshold drift_bonus_shifting 0.08 Weight for a new direction during drift shifting drift_bonus_recovered 0.04 Retained lightweight weight after drift recovery or

stabilization

drift_short_topic_bonus 0.03 Additional weight for short-term drift-topic matches reading_signal_short_term_bonus 0.05 Additional weight for recent reading-signal matches

Table 9: Explicit preference types.

Type Example Effect Author preference A specified author or team Raises ranking position after a match Institution preference A specified university, lab, or company Raises candidate priority after a match Keyword preference long-context LLM, protein language

Increases topic-match score after a match

model

Task preference information extraction benchmark, GUI grounding

Supports more fine-grained task matching

Negative preference not interested, downweight, remove, do not recommend

Downweights the corresponding topic

We use four system labels: must_read, high_relevant, maybe_interested, and edge_relevant. The must_read label is determined by explicit preference rules and has higher priority than score thresholds, because it represents authors, institutions, datasets, tasks, or keywords that the user explicitly wants to track. Other labels are determined jointly by the total score, the personal relevance floor, and rank-aware promotion. A regular paper must first satisfy min_relevance_signal. It is then categorized by thresholds 0.40, 0.25, and 0.15. If a paper ranks near the top among relevant candidates, it can be promoted to a higher display category even when its total score only reaches a lower threshold.

###### C.2 Explicit Preference Parameters

Explicit preferences simulate rules that a user states directly, such as “always show papers by this author,” “prioritize this institution,” or “track papers related to this benchmark or dataset.” This module is intended to capture stable, high-confidence user preferences.

Explicit preferences complement embedding similarity. Embeddings capture semantic relatedness, but may not express a user’s strong rule about a specific author, institution, or task. Explicit preferences fill

Table 10: Interest-drift parameters.

Parameter Value Meaning drift_probability 0.5 Probability of a drift opportunity in the simulation

environment ANCHOR_SIGNAL_WINDOW 3 Sliding window for observing new-topic signals ANCHOR_REQUIRED_CONSECUTIVE_DAYS 2 Number of consecutive days required for a new-topic

signal ANCHOR_SIGNAL_MIN_HITS 2 Minimum hits needed to lock a new topic ANCHOR_SIGNAL_MIN_RATIO 0.30 Minimum share of the new topic among selected

papers ANCHOR_SIGNAL_MIN_MARGIN 1 Minimum hit margin over the second-highest topic ANCHOR_INTENT_INCREMENT 0.15 Intent-score increment when anchor evidence is

observed ANCHOR_INTENT_DECAY 0.05 Intent-score decay when anchor evidence is insufficient ANCHOR_LOCK_THRESHOLD 0.30 Intent-score threshold for locking an anchor topic ANCHOR_PROGRESS_STEP 0.40 Drift-progress increment per update ANCHOR_SCORE_STEP 0.24 Drift-score increment per update ANCHOR_PRIMARY_BOOST 0.12 Primary-direction boost when the anchor is written into

core directions ANCHOR_SECONDARY_BOOST 0.06 Secondary boost for topics related to the anchor ANCHOR_DOWNWEIGHT_STEP 0.08 Per-step downweighting amount for old directions

during drift ANCHOR_MIN_WEIGHT 0.05 Minimum retained direction weight after downweighting ANCHOR_COMMITMENT_DAYS 3 Number of days to keep pushing the new direction

after anchor lock SIMULATION_MAX_DRIFT_OPPORTUNITIES 5 Maximum drift opportunities per user SIMULATION_CHECKFILE_COOLDOWN_ EPISODES

8 Cooldown episodes after completing one drift event

this gap. Conversely, relying only on explicit rules would miss semantically relevant papers without keyword matches. PaperFlow therefore combines both types of signals.

###### C.3 Interest-Drift Parameters

The interest-drift module simulates how a user’s long-term interests change with reading behavior. The system first enters an observing state to monitor new-topic signals. When a new topic appears consecutively and passes the thresholds, the system locks an anchor topic and enters the shifting state. When the user returns to old core directions or the new and old directions become balanced, the system enters the recovered or stable state.

A drift state does not directly imply higher recommendation quality. Its main purpose is to improve responsiveness to long-term interest changes. It may therefore introduce a small exploration cost on static oracle-based ranking metrics, while improving behavior consistency and long-term adaptation.

###### C.4 Reading-Signal Parameters

Reading signals represent short-term feedback formed by recent user selections. They are not the same as reading reports. The system updates short-term interest signals from the topics of recently selected papers and gives lightweight ranking bonuses to corresponding topics.

Reading signals are intentionally lightweight. They should not overwrite the long-term profile or

Table 11: Reading-signal parameters.

Parameter Value Meaning

PaperFlow_READING_SIGNAL_WINDOW_DAYS 21 Recent reading-signal window PaperFlow_READING_SIGNAL_ACTIVATION_COUNT 2 Minimum number of selections required to

activate a topic PaperFlow_READING_SIGNAL_TOPIC_SEED_WEAK 0.18 Initial strength for a weak topic signal PaperFlow_READING_SIGNAL_ TOPIC_SEED_STRONG

0.38 Initial strength for a strong topic signal

PaperFlow_READING_SIGNAL_TOPIC_DELTA_WEAK 0.03 Weak topic increment PaperFlow_READING_SIGNAL_ TOPIC_DELTA_STRONG

0.08 Strong topic increment

PaperFlow_READING_SIGNAL_CORE_SEED_STRONG 0.45 Initial strength for a strong core-direction signal PaperFlow_READING_SIGNAL_ CORE_DELTA_STRONG

0.08 Strong core-direction increment

PaperFlow_READING_SIGNAL_SHORT_TERM_BASE 0.35 Base strength for short-term signals PaperFlow_READING_SIGNAL_SHORT_TERM_STEP 0.18 Progress step for short-term signals PaperFlow_READING_SIGNAL_ SHORT_TERM_STRONG_BONUS

0.22 Additional bonus for strong short-term signals

Table 12: Reading-report parameters.

Parameter Value

READING_REPORT_PDF_TIMEOUT 60 seconds READING_REPORT_ARXIV_TIMEOUT 12 seconds READING_REPORT_ABSTRACT_CHARS 1200 READING_REPORT_SECTION_CHARS 1800 READING_REPORT_CHUNK_CHARS 1200 READING_REPORT_CHUNK_OVERLAP 180 READING_REPORT_EVIDENCE_TOP_K 3 READING_REPORT_PROFILE_RETRIEVAL_WEIGHT 0.25 READING_REPORT_LLM_MAX_TOKENS 4096

amplify one accidental selection into a long-term interest change. The system therefore uses a window, an activation count, and weak/strong signal distinctions. A topic becomes a short-term preference only after it has been selected multiple times in the recent window.

###### C.5 Reading-Report Parameters

The reading-report module performs PDF- or metadata-driven structured analysis for selected papers. Report generation prefers parsed PDF content and semantic-retrieval evidence. If PDF retrieval fails, the system falls back to a simplified analysis based on title, abstract, and metadata.

A reading report is not part of the ranking metric. It is a post-recommendation reading-assistance capability. It converts selected papers into structured reading material, including a one-sentence summary, research background, core method, key results, contributions, limitations, relevance to the user’s research, and suggested reading focus. For researchers, the recommendation list identifies what to read first; the reading report helps decide how and why to read it.

###### C.6 LLM and Embedding API Setting

In the model-comparison experiment, the LLM API and embedding API are configured separately. This ensures that the LLM comparison changes only the language model used in generation, parsing,

Table 13: API and output settings for model comparison.

Module Setting LLM API Configured separately for each experimental model, e.g., Gemini,

GPT, Qwen, DeepSeek, and Xiaomi MiMo Embedding API Separately configured and fixed to Qwen3-Embedding-8B Token records Daily records of embedding tokens, LLM tokens, total tokens, and

call count Output directory Each model writes to an independent results directory Database Before each full run, benchmark state is reset while the papers

table is retained

Table 14: Main recommendation metrics.

Metric Meaning

gNDCG@20 Top-20 ranking quality computed from oracle gains Useful@5 Fraction of useful papers in the Top-5 Useful@20 Fraction of useful papers in the Top-20 OracleRecall@20 Recall of strictly relevant papers in the Top-20 Lift@20 Top-20 useful rate divided by candidate-pool useful rate StrictR@20+ Whether a strictly relevant paper is recalled in the Top-20

when such papers exist MRR@20 Reciprocal rank of the first positive item in the Top-20 SelectedNDCG@20 Top-20 NDCG using simulated user selection as gain RecommendationScore Aggregate recommendation score

or judging stages, while leaving candidate-paper embeddings unchanged. Both the main experiment and model comparison use Qwen3-Embedding-8B as the embedding model.

This design avoids confounds in model comparison. If different LLMs also used different embedding models, performance differences could come from embeddings rather than LLM reasoning or parsing. Fixing the embedding model makes the comparison closer to replacing the LLM within the same recommendation framework.

###### D Evaluation Metrics

###### D.1 Main Metric Suite

The main experiment and related comparisons use the same recommendation metrics so that methods can be compared directly. The metrics cover ranking quality, recall, behavioral consistency, and aggregate recommendation quality.

Figure 9 summarizes the metric families and their aggregation relationships before we define each metric individually.

###### D.2 gNDCG@20

gNDCG@20 measures the overall ranking quality of the Top-20 list. Similar to standard NDCG, it checks whether highly relevant papers are ranked earlier. We convert oracle labels into gains: strong_relevant, relevant, weak_relevant, and irrelevant correspond to 2.0, 1.0, 0.5, and 0.0, respectively.

This metric evaluates whether the system concentrates valuable papers near the front of a limited display budget. In paper recommendation, users usually do not browse the entire candidate pool, but focus on the first few items. Thus, even if the Top-20 contains useful papers, usefulness decreases if

[Figure 217]

[Figure 218]

[Figure 219]

gNDCG@20: ranks high-gain oracle papers earlier Useful@5 / Useful@20: useful-paper density in Top-5 or Top-20 StrictR@20+: hits at least one strictly relevant paper when available SMRR@20: how early the first useful paper appears Lift@20: useful-paper enrichment over the candidate pool OracleRecall@20: recalls strictly relevant papers in Top-20 RecommendationScore: aggregate of gNDCG@20, Useful@5,Useful@20, StrictR@20+, MRR@20, and clipped Lift@20

[Figure 220]

[Figure 221]

PaperFlow Evaluation Metrics

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Oracle-based ranking quality

RecommendationScore (w/o OracleRecall@20)

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

Behavior alignment

[Figure 233]

[Figure 234]

SelectedNDCG@20: agreement with simulated reading selections

[Figure 235]

PostDrift metrics: selected post-drift ranking and behavior metrics NewTopicRecall@20: recall of new-interest papers OldTopicRate@20: old-topic exposure; lower is better AdaptationDelayDays: adaptation speed; lower is better DriftAutoScore: aggregate of post-drift quality, new-topic recall,old-topic suppression, and adaptation speed

[Figure 236]

[Figure 237]

[Figure 238]

Interestdriftadaptation

[Figure 239]

[Figure 240]

aggregated into DriftAutoScore

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

ReportAutoScore:100(0.70Section Completeness + 0.30Evidence) ModelAutoScore:Coverage) 0.80Recommendation Score + 0.20ReportAutoScore TokenCost: reported separately; notincluded in ModelAutoScore or ModelHumanScore

[Figure 246]

[Figure 247]

Report and model quality

[Figure 248]

[Figure 249]

quality and cost arereported separately

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

HumanEval: main recommendation-listhuman score

[Figure 256]

HumanEval: main recommendation-list human score

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Human validation

###### Human validation

[Figure 262]

ModelHumanScore, including HumanEval

ModelHumanScore: 0.80recommendation-listhuman score +0.20 report-human component

[Figure 263]

[Figure 264]

AdaptationHumanScore:human score fordriftadaptation

[Figure 265]

[Figure 266]

[Figure 267]

Prec@5 / Prec@20:user-marked worth-readingdensity ReadRate: actualopen /read /report-requestaction rate Sat. / ReportUse.: 1--5Likertuserratings

[Figure 268]

Real-user pilot

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

Figure 9: Overview of PaperFlow evaluation metrics. The figure groups metrics by oracle-based ranking quality, behavior alignment, interest-drift adaptation, report and model quality, human validation, and real-user pilot evaluation. Composite scores aggregate the indicated automatic metrics, while separately reported diagnostics, human scores, token cost, and real-user pilot metrics are not used as inputs to the main automatic recommendation score.

those papers are placed too late.

###### D.3 Useful@5 and Useful@20

Useful@5 is the fraction of papers in the Top-5 whose oracle label is strong_relevant, relevant, or weak_relevant. It focuses on recommendation density at the very top of the list.

Useful@20 is the fraction of useful papers in the Top-20. It measures the information density of the full displayed list. Compared with Useful@5, Useful@20 better captures how many potentially useful papers are covered within the display budget.

Both metrics are needed because researchers may either skim only the first few papers or inspect the full Top-20. Useful@5 is stricter and emphasizes head-of-list quality, while Useful@20 is broader and emphasizes overall display quality.

###### D.4 OracleRecall@20

OracleRecall@20 measures the fraction of strictly relevant papers recalled in the Top-20. Strict relevance typically includes strong_relevant and relevant, but not weak_relevant. This metric evaluates whether the system misses papers that are truly important to the user.

In scientific recommendation, missing one highly relevant paper can be more harmful than recommending several weakly relevant ones. This is especially true for must-read authors, key benchmarks, or highly matched research directions. OracleRecall@20 is therefore an important coverage metric.

###### D.5 Lift@20

Lift@20 is the Top-20 useful rate divided by the useful rate of the entire candidate pool. If useful papers are rare in the candidate pool but the system concentrates more of them in the Top-20, Lift@20 becomes high.

This metric measures the gain of recommendation over randomly browsing the candidate pool. Since the average daily candidate pool contains more than 400 papers and the user sees only 20, the system’s core value is to extract a higher-density set of useful papers. Lift@20 directly reflects this filtering ability.

###### D.6 StrictR@20+

StrictR@20+ measures whether the system recalls at least one strictly relevant paper in the Top-20 for episodes where strictly relevant papers exist. It is a recall- and usability-oriented metric that asks whether the system gives the user at least one truly important paper in key scenarios.

This metric is especially important for long-tail episodes. On some days, only a few strictly relevant papers may appear in the candidate pool. If the system misses all of them, the user experience for that day drops sharply. StrictR@20+ captures this failure mode.

###### D.7 MRR@20

MRR@20 computes the reciprocal rank of the first positive item within the Top-20. If the first recommendation is useful, the episode MRR is 1. If the first useful paper appears at rank 5, the MRR is 1/5.

MRR asks how quickly the user encounters the first useful recommendation. In a real system, earlier useful items help users trust the system and continue using it. This metric complements NDCG and useful-rate metrics.

###### D.8 SelectedNDCG@20

SelectedNDCG@20 uses simulated user selection behavior as gain and measures agreement between ranking and the papers that the user actually selects. If selected papers are ranked early, the score is high; if selected papers appear late, the score is low.

This metric differs from oracle-label metrics. Oracle labels measure static relevance to the user profile, while selection behavior reflects what the user might actually read in the current context. They need not perfectly agree. For example, a paper may be labeled weakly relevant by the oracle, but may still be selected because of recent reading signals or interest drift. SelectedNDCG@20 therefore captures dynamic behavioral fit.

###### D.9 RecommendationScore

RecommendationScore is the aggregate quality score of recommendations reported in the main result tables. Let G, U5, U20, Q, M, and L denote gNDCG@20, Useful@5, Useful@20, StrictR@20+, MRR@20,

and Lift@20, respectively. We define RecommendationScore Rs as

Rs = 100(0.25G + 0.15U5 + 0.15U20

+ 0.20Q + 0.15M

+ 0.10min(L/15,1)).

(8)

The Lift@20 term is clipped by min(L/15,1) to prevent extremely large lift values from dominating the aggregate score.

The score is used only as a compact display metric. It does not replace the individual metrics. We therefore interpret RecommendationScore together with the component ranking metrics, SelectedNDCG@20, and human evaluation. In particular, a high RecommendationScore does not imply that a method is better in every dimension; it mainly summarizes oracle-based ranking quality under a fixed Top-20 recommendation budget.

###### D.10 ReportAutoScore

ReportAutoScore is an automatic proxy for reading-report quality. It measures whether the generated report is structurally complete and supported by evidence anchors. Let Cs and Ec denote SectionCompleteness and EvidenceCoverage, respectively. We define ReportAutoScore Ra as

Ra = 100(0.70Cs + 0.30Ec). (9)

SectionCompleteness measures whether the required report fields are filled, and EvidenceCoverage measures whether the report contains retrieved evidence, evidence anchors, or full-text parsing support.

###### D.11 ModelAutoScore

The automatic model-quality score Ma combines recommendation quality and report quality, while token cost is reported separately:

Ma = 0.80Rs + 0.20Ra. (10) Because ReportAutoScore already includes both structure completeness and evidence coverage, these sub-metrics are not added again to ModelAutoScore.

###### D.12 DriftAutoScore

DriftAutoScore (Ds) summarizes adaptation-oriented signals into a 0–100 aggregate. Let O denote OldTopicRate@20 and D denote AdaptationDelayDays. We first define old-topic suppression So and normalized delay score Sd:

D − Dmin Dmax − Dmin

So = 1 − O, Sd = 1 −

. (11)

Let Gd, Ud, Rd, Bd, and Nd denote PostDrift gNDCG@20, PostDrift Useful@20, PostDrift OracleRecall@20, PostDrift SelectedNDCG@20, and NewTopicRecall@20, respectively. We define DriftAutoScore

Ds as

Ds = 100(0.10Gd + 0.10Ud + 0.10Rd

+ 0.20Bd + 0.25Nd

(12)

+ 0.15So + 0.10Sd).

The score weights post-drift ranking quality, behavioral alignment with post-drift selections, recall of new-topic papers, suppression of residual old-topic exposure, and adaptation speed. It is intended to compare adaptation behavior on drift episodes rather than to replace static oracle-based ranking metrics.

###### D.13 Relationship Among Metrics

We use both oracle-based and behavior-based metrics because scientific recommendation has two different goals. The first is to rank statically relevant papers near the top according to predefined relevance criteria. The second is to predict what the user will actually choose at the current stage of continuous reading.

Oracle-based metrics include gNDCG@20, Useful@5, Useful@20, OracleRecall@20, Lift@20, StrictR@20+, and MRR@20. They are suited for comparing ranking ability under fixed labels. The main behavior-based metric is SelectedNDCG@20, which evaluates agreement with simulated user behavior.

The two metric families can conflict slightly. For example, removing the drift module may make the system more conservative and prioritize papers aligned with stable long-term profiles, improving oracle-based ranking metrics. Full PaperFlow, however, retains drift exploration and may better match subsequent user selections, producing higher SelectedNDCG@20. This is not a metric bug; it reflects the trade-off between static relevance and dynamic behavioral adaptation.

###### E Prompt Templates

###### E.1 How Prompts Are Used

Prompts in PaperFlow are mainly used for structured parsing and reading-report generation, rather than directly asking an LLM to produce recommendations from evaluation labels. Recommendation ranking is determined by user profiles, paper content, rule-based signals, and structured fields produced by the model. Publishing the prompt templates improves reproducibility and reduces ambiguity caused by unstable LLM output formats.

We do not design a separate prompt for a subsession. A subsession is a logging and contextual unit, while prompts are called for profile parsing, direction extraction, recommendation-explanation generation, and reading-report generation.

###### E.2 User-Profile Parsing Prompt

This prompt parses a user’s natural-language description of research interests and converts it into structured profile-update signals.

User-Profile Parsing Prompt

System: You are an academic profile parsing assistant. Your task is to parse a user's description of research interests and extract structured information.

Pay special attention to negation, correction, contrast, and modal phrases. If the user expresses "not interested", "downweight", "remove", "do not recommend", or similar meanings, treat it as a negative adjustment rather than a positive interest.

Phrases such as "GUI Agent", "Cold Start", and "protein language model" must be kept as complete topics. Do not split them into separate words.

User: {user_input}

Return JSON only: {

"action": "adjust_interest | adjust_weight | add_must_read |

remove_must_read | unknown", "direction": "increase | decrease | null", "topics": ["recognized research directions or topics"], "confidence": 0.0, "reasoning": "brief explanation"

}

The key requirement is to recognize negative expressions and preserve complete phrases. Many scientific interests are fixed multi-word concepts, such as protein language model, world model, and GUI agent. Incorrectly splitting them would harm subsequent topic matching.

###### E.3 Research-Direction Extraction Prompt

This prompt extracts up to three research directions from a user’s self-description and prioritizes specific topics.

Research-Direction Extraction Prompt

System: Extract up to 3 research directions from a user's self-description. Keep multi-word phrases intact and prefer specific topics over broad umbrellas. Do not split phrases like "protein language model" or "world model for epidemiology". Known directions include: {known_direction_list}. Return JSON only.

User: {user_description}

Return: {

"directions": [ {

"name": "english-kebab-case-or-normalized-name", "display_name": "human-readable display name", "confidence": 0.0, "source_text": "matched phrase from the user text", "is_known": true

} ]

}

The purpose is to normalize natural-language interest descriptions into computable directions. The normalized directions are used for profile construction, topic-weight computation, and embeddingquery construction.

###### E.4 Recommendation-Explanation Prompt

This prompt generates a short explanation for papers in the Top-20 list. The explanation is not the final ranking criterion; it explains why the system considers a paper relevant.

Recommendation-Explanation Prompt

System: You are an assistant for explaining scientific-paper recommendations. Based on the user profile, paper metadata, system label, and matching signals, generate a concise, specific explanation that is faithful to the input information.

Requirements:

- 1. Do not invent methods, experiments, or conclusions that do not appear in the paper information.

- 2. If the paper is only topically related, explicitly say it is topically related. Do not exaggerate it as directly solving the user's problem.

- 3. If a must-read author, institution, or keyword is matched, state the reason.

- 4. If the paper is related to an interest-drift direction, state the corresponding new interest topic.

- 5. Output only one or two sentences.

User: {

"user_profile": { "core_directions": {...}, "must_read": {...}, "drift_state": "...", "reading_signal_topics": [...]

}, "paper": {

"title": "...", "abstract": "...", "authors": [...], "institutions": [...], "keywords": [...]

}, "signals": {

"system_label": "...", "score": 0.0, "matched_topics": [...], "matched_must_read": [...], "drift_matches": [...]

}

} Return JSON only: {

"reason": "recommendation explanation", "faithfulness_note": "which input fields support this explanation"

}

The prompt emphasizes faithfulness and restraint. Over-generated explanations can create false expectations. This is especially harmful in scientific-paper recommendation, where a system must avoid exaggerating “topic similarity” into “direct methodological relevance” or “immediately usable experimental findings.”

###### E.5 Reading-Report Generation Prompt

This prompt generates a structured reading report for a selected paper. The input includes paper metadata, abstract, PDF section snippets, user profile, a heuristic draft, and semantic-retrieval evidence.

Reading-Report Generation Prompt

System: You are a scientific-paper reading assistant. Based on the provided paper metadata, abstract, available section excerpts, and user profile, generate a structured reading report suitable for a research document.

If retrieved_evidence is provided, prioritize the evidence snippets retrieved from the PDF. Then use heuristic_draft for polishing and

completion. When retrieved_evidence conflicts with heuristic_draft, prioritize retrieved_evidence. Do not invent specific experimental numbers. If information is insufficient, be conservative and state that the original paper should be checked. Output only a JSON object. Do not output Markdown, explanations, or code blocks. User: {

"paper": { "title": "...", "authors": ["..."], "abstract": "...", "venue": "...", "publish_date": "...", "arxiv_id": "...", "doi": "..."

}, "sections": {

"introduction": "...", "method": "...", "results": "...", "discussion": "...", "conclusion": "..."

}, "user_profile": {

"top_directions": ["..."], "methodology_preferences": {...}, "report_preferences": {...}

}, "heuristic_draft": {...}, "retrieved_evidence": {...}, "field_evidence_map": {...}

} Return: {

"one_sentence_summary": "...", "research_background": "...", "core_method": "...", "key_results": "...", "main_contributions": ["..."], "limitations": ["..."], "relevance_points": ["..."], "reading_focus": ["..."], "recommendation_label": "Strongly recommended | Recommended |

Worth skimming | Read if needed", "analysis_note": "..."

}

The key design goals are evidence priority and hallucination prevention. The system first uses evidence retrieved from the PDF. If PDF retrieval fails, it falls back to a simplified report based on title, abstract, and metadata. Experimental numbers, dataset results, and conclusions must come from the input evidence rather than from model completion.

###### E.6 Reading-Report Markdown Template

Reading-Report Markdown Template {paper_title} Basic Information Authors: {authors} Institutions: {institutions} Source: {source} Date: {publish_date} Recommendation level: {recommendation_label} Estimated reading time: {estimated_reading_minutes} minutes Analysis source: {analysis_source} One-Sentence Summary {one_sentence_summary} Research Background {research_background} Core Method {core_method} Key Results {key_results} Contributions

- {contribution_1}

- {contribution_2}

- {contribution_3} Limitations

- {limitation_1}

- {limitation_2} Relation to My Research

- {relevance_point_1}

- {relevance_point_2} How to Read It

- {reading_focus_1}

- {reading_focus_2} PDF Evidence Anchors

Background evidence: {background_evidence} Method evidence: {method_evidence} Result evidence: {result_evidence}

Code and Resources

Code: {code_url_or_na} Data: {data_url_or_na} Project page: {project_url_or_na} Paper: {paper_url}

My Notes Point I most want to reproduce or borrow: Connection to my current research: Whether to follow the author, code, or follow-up work:

This template can be pasted directly into a research document. It separates paper metadata, content understanding, user relevance, and next actions, allowing the user not only to understand the paper but also to decide whether to read it deeply, reproduce it, or track follow-up work.

###### E.7 Human-Evaluation Prompts

The human-evaluation prompts correspond to the three human-score definitions in Appendix F. The main experiment evaluates only recommendation quality. The model comparison evaluates both recommendation and reading-report quality. The drift-specific evaluation evaluates only adaptation after interest change. All prompts use a 1–5 Likert scale. Under blind evaluation, method name, oracle label, selected field, and system score are hidden.

###### Main listwise recommendation-quality prompt.

Main Listwise Recommendation-Quality Prompt

You will see a researcher profile and an anonymized Top-20 recommendation list for one user-day episode. Please evaluate the whole list, not only a single paper.

Use a 1-5 Likert scale:

- 1 = very poor

- 2 = poor

- 3 = fair

- 4 = good

- 5 = very good Evaluate the following dimensions:

- 1. ProfileMatch: whether the list matches the user's profile and current research interests.

- 2. RankingQuality: whether stronger or more useful papers appear earlier in the Top-20 list.

- 3. DecisionUsefulness: whether the list helps the user decide what to read, skim, or skip.

- 4. DiversityFocusBalance: whether the list balances focused relevance with useful breadth.

Return JSON: {

"ProfileMatch": 1, "RankingQuality": 1, "DecisionUsefulness": 1, "DiversityFocusBalance": 1, "comments": "brief rationale"

}

###### Model-comparison report prompt.

Model-Comparison Report Prompt

You will see a researcher profile, a recommended paper, the paper abstract, a system recommendation explanation, and a reading report. Please evaluate recommendation quality and report quality separately.

Use a 1-5 Likert scale and return JSON: {

"HumanRelevance": 1, "HumanUsefulness": 1, "RecommendationDecisionHelpfulness": 1, "ReportFaithfulness": 1, "ReportSpecificity": 1, "ReportDecisionHelpfulness": 1, "comments": "brief rationale"

}

###### Interest-drift prompt.

Interest-Drift Prompt

You will see a user profile with interest drift, the interest directions before and after the drift, and recommended papers after the drift. Please evaluate whether the recommendation reasonably adapts to the user's new interests.

Use a 1-5 Likert scale and return JSON: {

"NewTopicFit": 1, "AdaptationAppropriateness": 1, "OldNewBalance": 1, "DriftDecisionHelpfulness": 1, "comments": "brief rationale"

}

###### F Human Evaluation Protocol

This section defines the human evaluation protocol. To avoid mixing human scores from different tasks, we define separate scores for main-experiment recommendation quality, LLM comparison, and interest-drift adaptation. The main paper reports only the corresponding human-score columns; all automatic-human correlation plots are placed in the appendix.

In this paper, blind human evaluation means that annotators are shown only the information needed for the judgment task, such as the user profile, anonymized recommendation list, and, for model comparison, sampled reading-report excerpts. They are not shown method or model identities, oracle relevance labels, simulated selections, system-side scores, or automatic metric values. The protocol is metric-aligned in the sense that the human dimensions evaluate the same recommendation, report, or drift-adaptation constructs measured by the automatic scores, but annotators do not see those scores and their judgments are not used to define the simulator labels.

###### F.1 Main-Experiment Human Evaluation: HumanEval

In the main experiment, HumanEval denotes a blind listwise human evaluation score over anonymized Top-20 recommendation lists. It measures ordinary recommendation quality only and is not used as human-labeled relevance ground truth. Each annotation unit is one method–episode Top-20 list.

Let Hp, Hr, and Hd denote ProfileMatch, RankingQuality, and DecisionUsefulness, respectively. We define the main HumanEval score as

He = 20 · mean(Hp, Hr, Hd). (13)

Table 15: Human dimensions for main-experiment listwise recommendation quality.

Dimension Meaning ProfileMatch Whether the list matches the user’s profile and current

research interests RankingQuality Whether stronger or more useful papers appear earlier in the Top-20 list DecisionUsefulness Whether the list helps the user decide what to read, skim, or skip DiversityFocusBalance Whether the list balances focused relevance with useful breadth

We additionally collect DiversityFocusBalance as a diagnostic dimension, but do not include it in the primary HumanEval score.

For the main human evaluation, we sample six user-day episodes and score six methods per episode, yielding 36 anonymized method–episode Top-20 lists. Three annotators independently score each list. Annotators see the user profile and the anonymized recommendation list, but not the method name, oracle labels, simulated selections, system scores, or automatic metric values.

###### F.2 Model-Comparison Human Evaluation: ModelHumanScore

The LLM comparison evaluates both recommendation quality and reading-report quality, so we define ModelHumanScore. Let Hrrec, Hurec, and Hdrec denote HumanRelevance, HumanUsefulness, and RecommendationDecisionHelpfulness, respectively. The recommendation component Mhrec is defined as

Mhrec = 20 · mean(Hrrec, Hurec, Hdrec). (14)

Let Hrepf , Hsrep, and Hdrep denote ReportFaithfulness, ReportSpecificity, and ReportDecisionHelpfulness, respectively. We define the report component Mhrep as

Mhrep = 20 · mean(Hrepf , Hsrep, Hdrep). (15)

The final ModelHumanScore Mh combines recommendation and report quality:

Mh = 0.80Mhrec + 0.20Mhrep. (16) It is also normalized to a 0–100 scale.

The recommendation-list component uses three dimensions: relevance to the user profile, usefulness of the ranked list, and helpfulness for deciding what to read. The reading-report component uses three dimensions: faithfulness to the paper information, specificity rather than genericness, and helpfulness for the reading decision. We sample six common user-day episodes across the 14 completed LLM backbones, yielding 84 anonymized model–episode list-level samples. Three annotators score each sample, producing 18 annotations per model. Reviewers see the user profile, the anonymized Top-10 recommendation list, and up to two reading-report excerpts, but not the model name, oracle labels, selected fields, system scores, or automatic metric values.

###### F.3 Drift-Specific Human Evaluation: AdaptationHumanScore

To test whether the interest-drift mechanism genuinely adapts to changing user interests, we define AdaptationHumanScore. This score is used only for episodes involving interest drift and is not mixed

Table 16: Human dimensions for drift-specific adaptation.

Dimension Meaning NewTopicFit Whether the recommendation fits the new in-

terest direction AdaptationAppropriateness Whether the system’s response to the interest change is appropriate OldNewBalance Whether the balance between old and new interests is reasonable

DriftDecisionHelpfulness Whether the recommendation helps the user decide whether to continue along the new direction

with ordinary HumanEval. Let An, Ap, Ab, and Ad denote NewTopicFit, AdaptationAppropriateness, OldNewBalance, and DriftDecisionHelpfulness, respectively. We define AdaptationHumanScore Ah

- as Ah = 20 · mean(An, Ap, Ab, Ad). (17)

It is normalized to a 0–100 scale and used only for drift episodes.

Drift human evaluation samples only episodes in observing, shifting, or recovered states, focusing on how the system responds to the new interest direction before and after drift. For finer analysis, one can score recommendations before, during, and after drift separately and compare how human scores change over states.

F.4 Inter-Annotator Agreement and Correlation Analysis

We report pairwise Pearson and Spearman correlations among the three annotators. For the main listwise evaluation, agreement is high across dimensions: mean pairwise Spearman correlation is 0.923 for ProfileMatch, 0.909 for RankingQuality, 0.872 for DecisionUsefulness, 0.844 for DiversityFocusBalance, and 0.901 for OverallRank.

To test alignment between automatic metrics and human judgment, we compute automatic–human correlations on the sampled blind-evaluation set. For the main experiment, each point is one sampled method–episode Top-20 list. On the 36 annotated lists, RecommendationScore correlates with HumanEval

- at Pearson r = 0.8626 and Spearman ρ = 0.8631. The component metric gNDCG@20 also correlates strongly with RankingQuality, with Pearson r = 0.8723 and Spearman ρ = 0.8774.

For the drift-specific evaluation, each point is one method–drift-episode recommendation list. On the 72 annotated lists, DriftAutoScore correlates with AdaptationHumanScore at Pearson r = 0.9149 and Spearman ρ = 0.8904.

For the model-comparison evaluation, each point is one LLM backbone. Across the 14 completed backbones, ModelAutoScore correlates with ModelHumanScore at Pearson r = 0.9632 and Spearman ρ = 0.9648. The model-comparison plot uses ModelAutoScore and ModelHumanScore, while the driftspecific plot uses DriftAutoScore and AdaptationHumanScore. These analyses check whether automatic metrics show the same trend as human judgment; they do not replace the individual metrics in the main result tables.

###### G Real-User Pilot Study Details

###### G.1 Study Setup

The real-user pilot study is designed to complement the simulated benchmark with a small amount of direct user-experience evidence. Five graduate students in computer science and AI participate

in the study. Each participant uses PaperFlow for 5–7 interaction rounds. In each round, the system presents a Top-20 daily paper list, and the participant marks papers they would read, papers they open or inspect in detail, and optional reading-report requests.

The study is not intended as a statistically powered deployment evaluation. Instead, it provides a sanity check on whether the ranking and reading-assist workflow produces useful recommendations for real researchers. All participant identifiers are anonymized.

- G.2 Participants Table 17: Participant information for the real-user pilot study.

User Research Area Experience Rounds

- U1 NLP / Summarization MS-2nd year 6
- U2 CV / Video Generation MS-1st year 5
- U3 RecSys / LLM-based Systems MS-2nd year 7
- U4 Multimodal Learning MS-1st year 5
- U5 RL / Code Generation MS-2nd year 6

- G.3 Metrics

The real-user pilot uses two behavioral metrics and two questionnaire metrics. The behavioral metrics are computed from participants’ actions on each Top-20 list, while the questionnaire metrics are collected after each interaction round.

For participant u and interaction round t, let R(uk,t) denote the top-k recommended papers. Let Su,t denote the set of papers that the participant marks as worth reading. This is a broad positive-feedback

set: it captures papers that the participant considers relevant or potentially useful after seeing the recommendation list. We compute:

Prec@k(u, t) = |R(uk,t) ∩ Su,t|

. (18)

k

Thus, Prec@5 measures the concentration of worth-reading papers near the top of the list, while Prec@20 measures the overall usefulness of the full displayed list.

ReadRate measures a stricter form of engagement. Let Du,t denote the set of papers that receive an explicit reading action, including opening the paper, inspecting it in detail, or requesting a reading report. These actions indicate stronger engagement than simply marking a paper as worth reading. We compute:

ReadRate(u, t) = |R(u20,t ) ∩ Du,t|

. (19)

20

Compared with Prec@20, ReadRate is therefore a more action-oriented metric: Prec@20 asks whether the Top-20 list contains papers the participant would consider reading, whereas ReadRate asks how many papers actually trigger a concrete reading action.

After each round, participants answer a 1–5 Likert questionnaire. Sat. is the overall satisfaction score: Sat.(u, t) = Qsat(u, t). (20)

ReportUse. is the reading-report usefulness score and is computed only for PaperFlow, because the two pilot baselines do not provide reading reports.

For each participant, metric values are first averaged over that participant’s interaction rounds. The final pilot-study results are then averaged over the five participants. In this sense, Prec@5, Prec@20, and ReadRate summarize observed behavior, while Sat. and ReportUse. summarize subjective user experience.

###### G.4 Real-User Baselines

We compare PaperFlow with two lightweight baselines in the pilot study. Daily arXiv Email corresponds to a date-based paper feed without personalization beyond broad topic subscription. Static Profile ranks papers using the initial user profile but does not apply profile updates, reading signals, or drift adaptation. These baselines are used only for the real-user pilot and are not intended to replace the controlled benchmark baselines in the main experiment.

###### G.5 Questionnaire After each interaction round, participants rated the following items on a 1–5 Likert scale.

Table 18: Questionnaire items for the real-user pilot study.

ID Question

- Q1 How relevant are today’s recommended papers to your current research?
- Q2 How accurately does the system represent your research interests?
- Q3 How useful are the reading reports for understanding selected papers?
- Q4 Compared with previous rounds, did today’s recommendations improve?
- Q5 Would you be willing to continue using this system?
- Q6 Overall, how satisfied are you with today’s recommendation list?

###### G.6 Per-User Results Table 19: Per-user averages in the real-user pilot study.

User Prec@5 Prec@20 ReadRate Sat. ReportUse.

- U1 0.72 0.48 0.35 4.1 4.3
- U2 0.64 0.42 0.30 3.8 4.1
- U3 0.80 0.53 0.40 4.4 4.6
- U4 0.68 0.45 0.33 3.9 4.0
- U5 0.70 0.46 0.32 4.0 4.4 Mean 0.71 0.47 0.34 4.0 4.3

###### G.7 Ethics and Anonymization

The pilot study uses anonymized participant identifiers and reports only aggregate or per-user averaged statistics. Participants are informed that the study is used to evaluate research-paper recommendation experience, and no private paper-reading logs are released. The pilot is limited in scale, so we avoid statistical significance claims and use the results only as complementary evidence.

###### H Case Studies

###### H.1 Case-Study Selection Criteria

Case studies complement aggregate metrics with qualitative evidence about why a recommendation list succeeds, adapts, or fails. As summarized in Figure 10, we organize the analysis around representative cases covering successful recommendation, interest drift, behavior consistency, boundary conditions, and downstream reading support.

This design avoids selecting only strong outputs. Instead, the case-study set checks whether PaperFlow is useful under different recommendation states, including aligned profile–paper matches, changing user interests, behavior-aware ranking effects, and cases where automatic oracle labels do not fully capture user behavior.

Operationally, these cases are selected from episodes with high ranking quality, drift-state transitions, high behavioral agreement, boundary disagreement between system and oracle labels, or complete PDF-based reading reports.

###### H.2 Successful Recommendation: A Dense Top-20 for an NLP User

This case examines user_role6::2026-03-16, an NLP user whose directions include NLP, large language models, and information extraction. The episode reaches gNDCG@20 = 1.0000, with 20 useful papers in the Top-20, 10 strictly relevant papers, and five selected papers. Figure 11 visualizes the Top-20 ranking and the evidence behind the dense high-rank recommendations.

The purpose of the case is to show the mechanism behind a strong list rather than repeat every ranked item in text. Representative top-ranked papers include work on long-context question answering, retrieval-augmented generation, and LLM-supported scientific discovery. When profile match, topic similarity, and must-read rules agree, PaperFlow concentrates useful papers near the top and remains consistent with the user’s later selections.

###### H.3 Interest Drift: From GUI/Web Agents to Multimodal Reasoning

This case focuses on adaptation rather than static relevance. The example user is user_role1, whose original directions include GUI agents, computer-vision grounding, and web automation. The recorded drift event is an anchor_lock on 2026-03-02, moving from observing to shifting toward multimodal reasoning and computer-using agents. Figure 12 summarizes how accumulated new-topic evidence changes the ranking emphasis.

The key observation is that PaperFlow does not immediately replace the user’s profile after a single new-topic hit. It waits for repeated evidence, locks the emerging topic as an anchor after the threshold is reached, and then increases exposure to papers aligned with the new direction.

###### H.4 Behavior Consistency: A High-SelectedNDCG List

Behavior-consistency cases focus on whether system ranking agrees with simulated user choices. Figure 13 shows a high-SelectedNDCG list in which papers selected by the user appear near the front of the Top-20 and have clear connections to the long-term profile, short-term reading signals, or drift direction.

This case follows the behavior-consistency criterion: selected papers should appear near the top or match emerging interests. It clarifies the difference between static relevance and behavioral adaptation. SelectedNDCG@20 rewards agreement with later user choices, while oracle labels measure content-level relevance. Qualitative inspection is therefore useful for interpreting cases where a paper is only weakly relevant under the oracle but remains consistent with the user’s recent reading trajectory.

Table 20: Boundary case with disagreement between system and oracle labels.

Field Content Episode user_role3::2026-03-24 User directions literature mining, scientific knowledge graph, hypothesis generation Daily result gNDCG@20 = 0.0000; 0 useful papers in Top-20 User selections Number of selected papers = 6 Top-ranked examples Permutation-Symmetrized Diffusion for Unconditional Molecular Genera-

tion; Describe-Then-Act; Off-Policy Value-Based Reinforcement Learning for Large Language Models

###### H.5 Failure or Boundary Case: Disagreement Between System Labels and Oracle Labels

Example episode:

This is a typical boundary case. The system may rank some papers highly because of short-term behavior signals, explicit rules, or topic-word matches, while the oracle still labels them as irrelevant. At the same time, the simulated user selects several papers, showing that selection behavior and oracle relevance are not identical. The paper should not describe such a case as a complete system failure. Instead, it reveals possible discrepancies among automatic oracle labels, behavior-selection signals, and system labels, which is one reason human evaluation is needed.

The case also suggests that relevance in scientific recommendation is layered. Some papers may not directly belong to a user’s core direction, but may still be inspiring because of their method, data structure, or task formulation. Automatic oracle labels may not fully capture such cross-domain inspiration. Human evaluation and case studies can therefore complement automatic metrics.

###### H.6 Reading Report: PDF-Evidence-Driven Reading Assistance Example report:

This case shows that PaperFlow does not only output a recommendation list. It also converts selected papers into structured reading-assistance material. Report generation combines the abstract, PDF sections, semantic-retrieval evidence, and user profile, so it can explain why the paper is relevant to the user, which parts should be read first, and how the paper might support the user’s current research. The main paper can show only a short excerpt and place the full report in the appendix.

From the system-value perspective, reading reports distinguish PaperFlow from ordinary paper recommenders. Traditional systems usually provide only titles, abstracts, and relevance scores. PaperFlow additionally generates profile-aware reading guidance. For researchers, this can reduce the reading cost after paper screening.

###### I Model Comparison Details

###### I.1 Purpose of Model Comparison

The LLM comparison evaluates how different large language models behave inside the PaperFlow framework. Since the recommendation pipeline includes LLM-supported profile parsing, recommendation-explanation generation, and reading-report generation, model differences in structured judgment, long-context understanding, reading-report generation, and cost can affect overall system performance.

The model-comparison experiment does not construct a new benchmark. It replaces the language

Table 21: PDF-evidence-driven reading-report case.

Field Content User user_role1 Date 2026-03-01 Paper The Informational Cost of Agency: A Bounded Measure of Interaction

Efficiency for Deployed Reinforcement Learning arXiv ID 2603.01283v2 Report source PDF Report fields one-sentence summary, research background, core method, key re-

sults, contributions, limitations, relation to user research, reading suggestions, and PDF evidence anchors

model within the same benchmark snapshot. This ensures that different models face the same users, the same candidate pools, and the same evaluation metrics. Results report RecommendationScore, Report AutoScore, ModelAutoScore, and ModelHumanScore; TokenCost is reported separately as an efficiency metric and is not included in the quality score.

ReportAutoScore follows the definition in the model-comparison section and combines SectionCompleteness and EvidenceCoverage. Operationally, SectionCompleteness corresponds to ReportStructureScore, and EvidenceCoverage corresponds to ReportEvidenceRate.

ParsingSuccess measures structured-output stability, but all completed model runs achieve 100% non-empty report generation success. It is therefore omitted from the main model-comparison table.

The cost-free ModelAutoScore follows the definition in the model-comparison section and is used as the quality score in the model-comparison table. TokenCost is derived from token-usage logs and reported separately as an efficiency metric.

###### I.2 Open and Closed Model Scope

The model-comparison study uses only the completed model runs reported in Table 2; candidate models that were planned but not run are excluded. We group the backbones by access mode. The closed API group contains GPT-5.4, Qwen3.5-Plus, Gemini 3.1 Pro Preview, Claude Sonnet 4.6, Qwen3.6-Plus, Qwen3.6-Max-Preview, Grok 4.3, and the default PaperFlow setting based on Gemini 3 Flash Preview. The open/open-access group contains MiMo-V2.5-Pro, DeepSeek-V4-Pro, DeepSeekV4-Flash, Kimi K2.6, GLM-5.1, and MiniMax-M2.7.

All models are evaluated with the same benchmark snapshot, embedding model, Top-20 display budget, and evaluation metrics.TokenCost is reported only as an efficiency metric and is not included in ModelAutoScore or ModelHumanScore.

###### I.3 Model-Comparison Controls The model comparison controls the following variables:

- 1. The candidate paper pool remains fixed.
- 2. User profiles remain fixed.
- 3. The Top-20 display budget remains fixed.
- 4. The embedding model remains fixed.
- 5. Evaluation metrics remain fixed.
- 6. Each model writes to a separate output directory.

- 7. Token usage is recorded by date.
- 8. TokenCost is reported only as an efficiency metric and is not included in ModelAutoScore or ModelHumanScore.

If a model encounters JSON parsing errors, API connection errors, or token-usage accounting anomalies, these should be recorded separately and should not be directly compared with models that completed full runs. Missing results caused by model-call failures should be marked as incomplete or unstable in the paper.

###### I.4 Token Usage Records

Each model experiment records daily token usage, including embedding tokens, LLM tokens, total tokens, and call count. Variation in embedding tokens is usually related to candidate-pool embedding cache status. If embedding tokens are zero on some dates, the corresponding embeddings may already be cached. LLM tokens mainly come from structured parsing, recommendation explanations, and reading-report generation.

Token usage can support cost analysis and explain runtime differences across models. A full benchmark contains 1200 episodes and reading reports for selected papers, so total runtime and token usage can be substantial.

###### J Reproducibility Notes

###### J.1 Runtime Environment

The experiment runs in a Python environment and uses a SQLite database to store user profiles, the paper table, behavior logs, and task state. Each model package contains independent run scripts, configuration files, result directories, and token-usage logs.

Before a full run, the system resets benchmark state while retaining the papers table. This removes user behavior, task state, and result records generated by a previous model run without rebuilding the paper pool.

###### J.2 Output Files Each full experiment produces multiple types of output files.

These outputs are used for later statistics, table generation, case-study selection, and human-evaluation sample construction.

###### J.3 Error Handling

External API or network errors can occur during the experiment, including arXiv API timeout, 429 rate limits, PDF download failure, LLM connection error, or JSON parsing error. The system provides fallback mechanisms for some errors. For example, if arXiv metadata retrieval fails, it attempts to download the PDF; if PDF retrieval fails, it uses the title and abstract to generate a simplified report.

Two types of errors should be distinguished. The first type is recoverable, such as an arXiv timeout followed by successful PDF parsing; this usually does not affect the main flow. The second type is unrecoverable, such as path-length write failure, complete LLM API unavailability, or unparseable output; these require code or configuration fixes and rerunning the experiment.

In the paper, if a model experiment is incomplete because of API instability, it should be explicitly marked as incomplete rather than directly comparing partial results with complete runs.

Table 22: Runtime records and output files.

Category Field Meaning Token usage date Date Token usage embedding_tokens Embedding tokens used on that day Token usage llm_tokens LLM tokens used on that day Token usage total_tokens Sum of embedding and LLM tokens Token usage call_count Number of API calls that day Runtime environment

Database SQLite

Runtime environment

Main script scripts/simulate_historical_episodes.py

Database cleanup script scripts/clear_database.py

Runtime environment

Runtime environment

Result directory results/{model_key}

Runtime environment

Token log results/{model_key}/token_usage.jsonl

Runtime environment

Reading-report directory results/{model_key}/reading_reports_md

Output files Episode metadata Recommendation context and results for each user-

day episode Output files Ranking results Top-20 recommendation lists and scores Output files Behavior logs Simulated user-selection behavior Output files Reading reports Reading reports for selected papers Output files Token usage Daily token usage Output files Drift timeline Interest-drift state-change records Output files Aggregate metrics Aggregated main metrics and model-comparison re-

sults

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

#### H.1 Case-Study Selection Criteria

[Figure 277]

Successful Recommendation

[Figure 278]

###### Rationale:

[Figure 279]

Case studies explain recommendation behavior behind aggregate metrics. We recommend including at least five representative types so that the analysis covers strong performance, adaptation, consistency, failure boundaries, and downstream reading outcomes.

[Figure 280]

Behavior Consistency

[Figure 281]

Drift Adaptation Case Study Set

[Figure 282]

[Figure 283]

[Figure 284]

Failure Boundary Case

Reading-Report Case

[Figure 285]

###### Recommended Case Types:

[Figure 286]

[Figure 287]

[Figure 288]

###### Successful Recommendation

###### Drift Adaptation

###### Behavior Consistency

1 2 3

[Figure 289]

Strong alignment between profile and paper pool

Interest shifts and topic reweighting

Ranking agrees with later user actions

[Figure 290]

[Figure 291]

###### Failure / Boundary Case

Reading-Report Case

4 5

Weak alignment or sparse candidate support

Evidence from subsequent reading outcomes

[Figure 292]

[Figure 293]

- Figure 10: Case-study selection criteria. The figure summarizes five representative case types for inspecting PaperFlow beyond aggregate metrics: successful recommendation, drift adaptation, behavior consistency, boundary disagreement, and reading-report support.

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

###### H.2 Successful Recommendation: A Dense Top-20 for an NLP User

[Figure 298]

[Figure 299]

[Figure 300]

###### Example episode:

This case shows that when the user's long-term profile sufficiently overlaps with the daily paper pool, PaperFlow can rank relevant papers near the top. Several Top-5 papers are labeled must_read by the system and relevant by the oracle, indicating agreement among explicit preferences, topic matching, and content-relevance signals. The user selects five papers, showing that the ranking works not only under static oracle labels but also under subsequent reading behavior.

[Figure 301]

[Figure 302]

[Figure 303]

###### Evidence:

###### Mechanism:

[Figure 304]

[Figure 305]

The success comes from aligned signals. The user's long-termdirections are NLP, LLMs, and information extraction. The candidatepool that day contains multiple papers on long-context questionanswering, RAG, information extraction, and LLM evaluation.Embedding similarity, topic weights, and must-read rules all pointto the same directions, yielding stable and dense rankings.

- 1. Several Top-5 papers are both must_read and relevant.

[Figure 306]

- 2. The user selects five papers from the list.

[Figure 307]

- 3. The ranking remains strong beyond static oracle labels.

[Figure 308]

Top-20 Overview: Density of Relevant/Must-read Papers

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

Rank Relevant/Must_read Rank Relevant/Must_read

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

User Long-term Interests

Aligned Signals Paper Topics in Daily Pool

Top-5 (1-5)

[Figure 318]

[Figure 319]

Long-context QA

NLP

[Figure 320]

Embedding Similarity

Top-10 (6-10)

[Figure 321]

RAG

[Figure 322]

Topic Weights

LLMs

Information Extraction

[Figure 323]

Top-20 (11-10)

[Figure 324]

Must-read Rules

Information Extraction (IE)

[Figure 325]

LLM Evaluation

[Figure 326]

[Figure 327]

[Figure 328]

relevant/must_read relevant(not must_read) not relevant

- Figure 11: Successful recommendation case for an NLP user. The episode shows that the user’s NLP/LLM/information-extraction profile aligns with the daily candidate pool, producing a dense Top-20 list with useful papers near the top.

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

###### H.3 Interest Drift: From GUI/Web Agents to Multimodal Reasoning

[Figure 333]

###### Example user:

[Figure 334]

[Figure 335]

[Figure 336]

Observing State

Drifted Anchor (After Threshould)

Older Interests (Initial State)

(Accumulating New-Topic Hits)

[Figure 337]

[Figure 338]

This case illustrates the role of the interest-drift module. User_role1 initially focuses on GUI agents and web automation, but the drift plan includes a shift from rule-like web automation to stronger multimodal interaction agents. In the observing state, the system records new-topic hits. Once the signal passes the threshold, it locks multimodal reasoning as the anchor topic, increases the ranking weight of related papers in subsequent recommendations, and reduces the influence of the old topic, web automation.

[Figure 339]

New-topic hit (e.g., multimodal reasoning)

[Figure 340]

[Figure 341]

[Figure 342]

GUI Agents Web Automation

Mutimodal Reasoning

[Figure 343]

Cumulative New-Topic Hits

[Figure 344]

Drift Threshold

[Figure 345]

[Figure 346]

Rule-like web automation

Stronger multimodal interaction agents

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

Focus on Web Automation Shift to Multimodal Reasoning

[Figure 351]

[Figure 352]

###### Observed Signal:

###### Update Mechanism:

[Figure 353]

[Figure 354]

- 1. The user initially concentrates on GUI agents and web automation.
- 2. The observing state accumulates repeated new-topic hits.
- 3. The signal eventually crosses the drift threshold.

After the threshold is crossed, the system locks multimodal reasoning as the anchor topic. It then increases the ranking weight of papers related to multimodal interaction and reasoning in future recommendations, while reducing the influence of the earlier web-automation topic.

[Figure 355]

###### New-Topic Hits over Time

[Figure 356]

[Figure 357]

New-topic hits (per observation) Drift threshold

[Figure 358]

Before Drift (Web Automation Focus)

###### After Drift (Multimodal Anchor)

[Figure 359]

[Figure 360]

[Figure 361]

Rank Paper(Topic) Rank Paper(Topic)

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

threshold(30)

###### Reweighting After Threshold

[Figure 366]

Time (observations)

- Figure 12: Interest-drift case from GUI/Web agents to multimodal reasoning. Repeated evidence for a new direction leads PaperFlow to lock a multimodal-reasoning anchor and reweight later recommendations away from stale web-automation interests.

[Figure 367]

[Figure 368]

##### H.4 Behavior Consistency: A High-SelectedNDCG List

[Figure 369]

[Figure 370]

[Figure 371]

###### Behavior-consistency case:

[Figure 372]

Behavior-consistency cases focus on whether system ranking agrees with simulated user choices. In an ideal case, papers eventually selected by the user appear near the front of the Top-20 and have clear connections to the long-term profile, short-term reading signals, or drift direction.

[Figure 373]

[Figure 374]

###### Interpretation:

###### Why SelectedNDCG@20 Can Increase:

[Figure 375]

The case illustrates that behavioral consistency is not identical to static relevance. Selected papers are not always those with the highest oracle gain, but they may better match the user's recent reading trajectory. Longitudinal scientific recommendation should therefore evaluate both static ranking quality and behavioral adaptation.

[Figure 376]

[Figure 377]

[Figure 378]

- 1. Some papers may be only weak_relevant under the oracle.
- 2. The user may still click or read them because similar topicswere selected recently.
- 3. If the system ranks these papers early, SelectedNDCG@20 increases.

[Figure 379]

Top-20 List with Early User Selections

[Figure 380]

Two Complementary Perspectives

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

Static Relevance

Behavioral Adaptation

[Figure 387]

[Figure 388]

[Figure 389]

Focus on content-level relevance and intrinsicquality

[Figure 390]

Focus on user behavior,recent signals,and evolution

- Figure 13: Behavior-consistency case with a high-SelectedNDCG list. Selected papers appear near the front of the Top-20, showing that behavior-based agreement can complement static oracle relevance in longitudinal recommendation evaluation.

