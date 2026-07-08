# arXiv:2606.15345v2[cs.CL]17Jun2026

## Beyond Monolingual Deep Research: Evaluating Agents and Retrievers with Cross-Lingual BrowseComp-Plus

Yuheng Lu1*, Qingcheng Zeng2*, Heli Qi1,3, Puxuan Yu4, Fuheng Zhao5, Rui Yang6, Hitomi Yanaka7,3, Naoto Yokoya7,3, Weihao Xuan7,3† 1Waseda University, 2Northwestern University, 3RIKEN AIP, 4Snowflake Inc., 5University of Utah, 6Duke-NUS Medical School, 7The University of Tokyo

### Abstract

Deep research agents are increasingly evaluated on their ability to search for evidence, reason over retrieved sources, and produce grounded answers. Existing browsing benchmarks, however, largely assume that the user’s query and the supporting evidence are written in the same language, leaving open whether agentic search systems can operate when relevant evidence appears in another language. We introduce XBCP (Cross-lingual BrowseCompPlus), a controlled benchmark that preserves the English question-and-answer space of BrowseComp-Plus but varies the languages of the supporting documents. XBCP instantiates two complementary settings. In the crosslingual setting, each query is paired with evidence in a single assigned language. In the multilingual setting, the full evidence corpus is distributed equally and randomly across 12 languages spanning high-resource and lowresource regimes. We evaluate four deep research agents using sparse and dense multilingual retrievers, measuring answer accuracy, evidence recall, search behavior, calibration, citation fidelity, and oracle retrieval. Results reveal substantial degradation when evidence is translated. Even strong, dense retrievers lose evidence recall, and agents become less calibrated and cite evidence less reliably. Notably, accuracy remains lower even when all gold evidence is supplied directly. These findings suggest that cross-lingual deep research exposes both retrieval failures and an independent, agent-side difficulty in integrating language-mismatched evidence.

### 1 Introduction

Large language model (LLM) agents represent a shift from models that answer from parametric knowledge alone to systems that actively acquire, filter, and synthesize external evidence.

*Equal contribution. †Corresponding author.

Deep research systems are a representative instance of this shift: given a complex information need, an agent must plan searches, inspect retrieved sources, judge whether the evidence is sufficient, and compose a grounded answer (OpenAI, 2025a). This broader movement has made browsing-based evaluation a central test of agentic capability. BrowseComp (Wei et al., 2025) crystallizes the challenge by posing difficult but verifiable questions whose answers require nontrivial web exploration, thereby stressing both search behavior and evidence-grounded reasoning. However, evaluations over live web search measure an entire time-varying system at once, entangling the language model, retrieval method, ranking API, and underlying corpus. BrowseComp-Plus (Chen et al., 2025) addresses this limitation by grounding BrowseComp-style questions in a fixed, humanverified corpus with supporting documents and hard negatives, turning browsing evaluation into a controlled setting where retrievers and LLM agents can be studied both separately and in interaction. This controlled view of deep research, however, remains largely confined to monolingual settings. The limitation matters because multilingual and cross-lingual retrieval have long been central concerns in information retrieval, and recent multilingual embedding models have greatly expanded the ability to retrieve across languages (Yu et al., 2024; Zhang et al., 2024, 2025). Most evaluations of these models still treat retrieval as a standalone ranking problem: a query is matched against a fixed collection, and success is measured by documentlevel relevance. This abstraction is useful for isolating retrieval quality, but it does not capture what happens when retrieval is part of an agentic search process. In that setting, the system must issue and refine searches, compare partial evidence, and decide how retrieved information should support an answer. Recent browsing-agent benchmarks beyond English, such as BrowseComp-ZH (Zhou et al., 2025), broaden the linguistic scope of agent evaluation but remain primarily monolingual: ques-

tions, evidence, and answers all stay within the same language. They therefore leave open the genuinely cross-lingual case, where an information need expressed in one language must be answered using evidence written in another. A crosslingual extension of BrowseComp-Plus is needed to make this setting measurable. Such a benchmark would test whether multilingual retrievers can surface the right evidence during agentic search and whether LLM agents can integrate languagemismatched evidence into faithful answers. To make this setting measurable, we introduce Crosslingual BrowseComp-Plus (XBCP). To the best of our knowledge, XBCP is the first benchmark to formalize cross-lingual deep research, extending the controlled evaluation paradigm of BrowseCompPlus from monolingual to multilingual retrieval. XBCP preserves the task structure of BrowseCompPlus: questions are posed in English, answers are expected in English, and the evidence is grounded in a fixed corpus. The key difference is that the supporting evidence is no longer assumed to be written in the same language as the question. We instantiate this design with two complementary configurations. In the cross-lingual setting, all supporting documents for a given query appear in the same language, while the assigned language varies across queries. This tests whether systems remain robust as otherwise comparable tasks move across languages. In the multilingual setting, the evidence corpus is randomly but equally assigned to 12 languages spanning high-resource and low-resource regimes, enabling controlled evaluation of English queries against language-specific evidence documents. Together, these configurations allow XBCP to evaluate both whether multilingual retrievers can surface language-mismatched evidence during agentic search and whether LLM agents can integrate such evidence into faithful English answers. Our experiments reveal large drops in accuracy and evidence recall across retrievers, reduced citation reliability, and persistent degradation even under oracle retrieval. These findings indicate that cross-lingual deep research stresses both retrieval and agent-side evidence integration. Figure 1 summarizes the construction and evaluation pipeline. Code and datasets are publicly available in our GitHub repository and Hugging Face collections.

### 2 Related Works

Deep Research Systems. Deep research systems extend tool-augmented LLMs from single-step re-

trieval to long-horizon information seeking, where agents must plan searches, interact with external sources, verify intermediate evidence, and synthesize grounded answers. OpenAI Deep Research (OpenAI, 2025a) exemplifies this paradigm and has motivated a growing line of open research agents that scale the underlying capabilities in different ways: Tongyi DeepResearch (Team et al., 2026) combines agentic mid-training and post-training with large-scale synthetic trajectories, MiroThinker (MiroMind Team et al., 2026) studies model, context, and interaction scaling, and Marco DeepResearch (Zhu et al., 2026) emphasizes verificationcentric training and inference to reduce error propagation in long-horizon search. Benchmarking has also moved toward more demanding settings, including Chinese web browsing in BrowseCompZH (Zhou et al., 2025), expert-level financial search in FinSearchComp (Hu et al., 2025), and noisy or conflicting search results in SealQA (Pham et al., 2026). These efforts have substantially advanced both systems and evaluations, but remain largely monolingual or domain-specific, leaving cross-lingual deep research underexplored.

Multilingual and Cross-lingual Retrieval. Multilingual and cross-lingual retrieval has moved from translation-mediated CLIR toward shared embedding spaces. ME5 (Wang et al., 2024) extends the E5 recipe with billion-scale multilingual contrastive pre-training and supervised finetuning, while later systems expand the design space through long-context encoders in MGTE (Zhang

- et al., 2024), efficiency- and compression-aware multilingual embeddings in ARCTIC-EMBED 2.0 (Yu et al., 2024), and foundation-model-based multilingual training in QWEN3 EMBEDDING (Zhang
- et al., 2025). This progress is accompanied by a broader recognition that CLIR is not simply monolingual retrieval plus translation: retrieval quality depends on cross-lingual representation alignment, resource imbalance, domain transfer, and evaluation design (Goworek et al., 2025). Evaluation has therefore expanded to representative benchmarks such as MMTEB (Enevoldsen et al., 2025), MIRACL (Zhang et al., 2023), and MLDR (Chen et al., 2024), but it remains a fixed-collection ranking problem. Large-scale CLIR experiments show that multilingual bi-encoders and translation-based lexical retrieval dominate across different datasets and language regimes (Zuo et al., 2025); task-specific fact-checking studies further show that multilingual

###### 1 Data Preparation

###### Original BrowseComp-Plus

- • ~100K English docs
- • 5,040 evidence docs
- • 830 queries

Language Assignment Different assignment to evidence docs of:

- • Crosslingual corpus
- • Multilingual corpus

###### Translate Translate evidence docs via GPT-5.4

Language Split(12) High-Resource(8): zh, en, fr, de, ja, ko, pt, es Low-resource(4): sw, wo, yo, zu

A ⽂ あ 한

###### Quality Check

Translation quality check by commercial companies

###### XBCP Pipeline

Corpora and Index Rebuild

3 Agent Experiments

###### 4 Evaluation

2

Retrievers

Agents

Original Corpus

###### LLM-as-Judge

via GPT-5.4

BM25

~100K original docs

GPT-OSS-20B GPT-OSS-120B

Qwen3Embedding4B/8B

remain unchanged

###### Metrics

- • Accuracy
- • Retriever Recall
- • Calibration Error
- • Avg Search Turns
- • ……

Arcticembed-l-v2.0

Qwen3.6-35B-A3B

###### Multilingual Corpus

- • 4620 translated+420 English evidence docs
- • ~95K original negative docs(en)

Multilinguale5-large

DeepSeek-V4-Pro

###### Result matrix

Randomly selected 420 evidence docs per language

Experiment Settings

Agents tested in 3 corpora with different retrievers.

Standard runs

###### Cross-lingual Corpus

- • 4619 translated+421 English evidence docs
- • ~95K original negative docs(en)

Reasoning effort change

A ⽂ あ 한 é ß

Per-language evaluation

Reasoning-aware retrieval

Performance broke down by language

ã ñ ŋ ẹ sw zu

No overlap between different query groups

Oracle retrieval

Figure 1: Overview of the XBCP pipeline. We translate and reorganize the evidence side of BrowseComp-Plus into cross-lingual and multilingual corpora, rebuild retrieval indexes for controlled agent experiments, and evaluate agents and retrievers with end-to-end accuracy, evidence recall, calibration, oracle retrieval, and per-language analysis.

and cross-lingual retrieval yield different model rankings and gains from supervised adaptation (Ramponi et al., 2025). These works provide strong retrievers and ranking-oriented evaluations, but not a view of cross-lingual retrieval inside the iterative search, evidence selection, and answer synthesis loop of deep research agents.

### 3 Building XBCP

#### 3.1 Translation-Based Construction

We build XBCP by translating the evidence side of BrowseComp-Plus (Chen et al., 2025): questions remain in English, final answers are evaluated in English, and only the evidence documents vary in languages. We use GPT-5.4 (OpenAI, 2026) as the translation model with a single languageconditioned prompt that requests complete translation into the target language, including titles, terminology, proper nouns, and metadata field names, while preserving URLs, email addresses, formulas, and code blocks; the full prompt is shown in Appendix B. This prompt is applied to each source document for the non-English target languages, while English documents are retained unchanged. The resulting evidence languages are designed to span different resource conditions. We include relatively high-resource languages with substantial web and retrieval coverage, namely Chinese, English, French, German, Japanese, Korean, Portuguese, and Spanish, as well as low-resource

African languages, namely Swahili, Wolof, Yoruba, and Zulu. This language set allows XBCP to test whether cross-lingual deep research systems degrade smoothly across resource regimes or fail disproportionately when evidence appears in languages with weaker retrieval and modeling support.

The translated corpus supports two evaluation configurations. In the cross-lingual setting, each query is assigned to one evidence language, so all supporting documents for that query appear in the same language(English serves as an untranslated reference).Appendix Table 8 reports the resulting 830 query assignments and 5,040 evidencedocument assignments. In the multilingual setting, 5,040 evidence document instances are randomly but equally assigned to 12 languages, making 420 evidence docs per language; Appendix Table 9 gives the per-language document counts. This construction lets us vary the linguistic form of the evidence while preserving the original task semantics, making retrieval failures and agent-side synthesis failures comparable across languages.

#### 3.2 Verification and Quality Control

To assess the quality of the translated evidence, we conduct an independent expert verification study following the translation-evaluation rubric of MMLU-ProX (Xuan et al., 2025). The rubrics is in Appendix C. We sample 200 translated documents from each of 11 non-English languages,

yielding 2200 translation instances in total. Expert annotators compare each translation against the original English document and rate it along the same three dimensions in MMLU-ProX, accuracy, fluency, and completeness on 1-5 scale, so that the verification focuses on whether the translated documents preserve the evidence needed for retrieval and answer synthesis. Verification results are in Appendix D. All language-level mean scores exceed 4.0, suggesting that the translated evidence is generally usable for controlled evaluation, while residual artifacts may remain.

### 4 Experiments and Results

#### 4.1 Experimental Setup

Following the evaluation protocol of BrowseCompPlus (Chen et al., 2025), we evaluate XBCP by pairing search agents with controlled retriever tools over fixed corpora. We consider four agents: GPT-OSS-20B (OpenAI et al., 2025), GPTOSS-120B (OpenAI et al., 2025), QWEN3.635B-A3B (Qwen Team, 2026), and DEEPSEEKV4-PRO (DeepSeek-AI, 2026). For retrieval, we compare a sparse lexical baseline, BM25 (Robertson and Zaragoza, 2009), with four dense multilingual retrievers: QWEN3-EMBEDDING4B, QWEN3-EMBEDDING-8B (Zhang et al.,

- 2025), MULTILINGUAL-E5-LARGE (Wang et al., 2024), and ARCTIC-EMBED-L-2.0 (Yu et al., 2024). GPT-OSS-20B, GPT-OSS-120B, and QWEN3.6-35B-A3B are evaluated with all five retrievers, while DEEPSEEK-V4-PRO is evaluated with BM25 and QWEN3-EMBEDDING-8B. Each available agent-retriever pair is evaluated on three corpus conditions.

Evaluations are at two complementary levels. First, end-to-end agent performance captures whether an agent can answer correctly while using a retriever as its search tool. Accuracy scores final answer correctness; evidence recall, computed over the union of documents returned across the agent’s search trajectory, measures retriever-side coverage of human-verified evidence independent of downstream agent behavior; average search calls captures exploration cost; and calibration error measures the mismatch between the agent’s stated confidence and its observed correctness.

Second, we analyze retriever behavior as it appears inside the agent loop. In this setting, retrieval quality is not only a top-k ranking property: a useful retriever should surface supporting docu-

ments consistently enough for the agent to find them through iterative search, reduce unnecessary follow-up searches, and provide evidence that can be cited in the final response. We therefore report citation coverage, average citation count, citation precision, and citation recall to measure whether retrieved evidence is carried through into faithful source attribution.

Beyond these two levels, we additionally evaluate an oracle retrieval setting that bypasses search and ranking by supplying all supporting evidence directly to the agent, isolating reasoning errors from retrieval errors. We also report three supplementary analyses: a per-language decomposition, a reasoning-based query expansion experiment, and a reasoning-effort control study.

Since our benchmarks are set in multilingual and crosslingual settings, the original selected models QWEN3-32B (Yang et al., 2025) and GPT-4.1 (OpenAI, 2025b) in LLM-as-Judge in BrowseComp-Plus are not suitable in our experiments. We therefore adopt GPT-5.4 (OpenAI, 2026) and change the judge prompt for evaluation. The new judge prompt is in Appendix E.

4.2 Main Results 4.2.1 End-to-End Agent Evaluation

Table 1 reports end-to-end accuracy and evidence recall. The strongest overall performance is obtained by DEEPSEEK-V4-PRO with QWEN3EMBEDDING-8B, reaching 64.70% accuracy on the original corpus, 48.80% in the multilingual setting, and 42.29% in the cross-lingual setting. Among the agents evaluated with the full retriever suite, QWEN3-EMBEDDING-8B also gives the strongest original-corpus performance, consistent with the BrowseComp-Plus finding that stronger retrievers improve deep-research agents by surfacing more useful evidence during iterative search (Chen et al., 2025).

The same table shows that translated evidence introduces a large additional difficulty. With QWEN3EMBEDDING-8B, accuracy drops by roughly 16– 23 pp across agents when moving from the original corpus to the translated settings. The degradation appears not only with BM25 but also with dense multilingual retrievers. Meanwhile, multilingual and cross-lingual results are close across most agent–retriever pairs, suggesting that the primary bottleneck is language mismatch rather than the specific language-assignment regime.

Accuracy (%) Evidence Recall (%) Orig. Multi. ∆M Cross. ∆C Orig. Multi. ∆M Cross. ∆C

Agent Retriever

BM25 15.18 3.13 -12.05 3.49 -11.69 22.58 5.10 -17.48 5.59 -16.99 QWEN3-EMBEDDING-4B 29.04 11.57 -17.47 11.81 -17.23 38.13 23.74 -14.39 23.12 -15.01 QWEN3-EMBEDDING-8B 32.89 12.05 -20.84 11.93 -20.96 42.91 24.60 -18.31 23.95 -18.96 MULTILINGUAL-E5-LARGE 20.84 4.10 -16.74 3.37 -17.47 24.28 4.76 -19.52 4.40 -19.88 ARCTIC-EMBED-L-2.0 28.80 12.17 -16.63 10.96 -17.84 37.27 20.92 -16.35 20.42 -16.85

GPT-OSS-20B

BM25 22.65 5.90 -16.75 5.42 -17.23 31.21 9.14 -22.07 8.24 -22.97 QWEN3-EMBEDDING-4B 35.42 13.25 -22.17 15.30 -20.12 45.55 28.53 -17.02 28.50 -17.05 QWEN3-EMBEDDING-8B 38.07 14.58 -23.49 15.18 -22.89 48.19 29.85 -18.34 28.85 -19.34 MULTILINGUAL-E5-LARGE 20.84 6.51 -14.33 5.54 -15.30 25.26 6.32 -18.94 6.60 -18.66 ARCTIC-EMBED-L-2.0 33.61 14.82 -18.79 14.46 -19.15 43.73 26.83 -16.90 24.41 -19.32

GPT-OSS-120B

BM25 21.69 5.42 -16.27 5.18 -16.51 24.12 6.91 -17.21 6.10 -18.02 QWEN3-EMBEDDING-4B 32.29 18.31 -13.98 16.51 -15.78 37.33 25.66 -11.67 24.44 -12.89 QWEN3-EMBEDDING-8B 38.55 18.19 -20.36 17.95 -20.60 43.14 28.08 -15.06 26.74 -16.40 MULTILINGUAL-E5-LARGE 24.82 5.66 -19.16 5.42 -19.40 24.31 4.79 -19.52 4.57 -19.74 ARCTIC-EMBED-L-2.0 35.18 17.95 -17.23 15.90 -19.28 37.46 21.74 -15.72 20.51 -16.95

QWEN3.6-35B-A3B

DEEPSEEK-V4-PRO BM25 45.06 18.55 -26.51 17.47 -27.59 51.08 18.15 -32.93 18.51 -32.57 QWEN3-EMBEDDING-8B 64.70 48.80 -15.90 42.29 -22.41 72.77 59.78 -12.99 53.82 -18.95

- Table 1: End-to-end agent performance across corpus conditions. Multi. denotes the multilingual corpus, Cross.

denotes the cross-lingual corpus, and ∆M and ∆C denote changes from the original corpus to the multilingual and cross-lingual corpora, respectively. DEEPSEEK-V4-PRO is evaluated with BM25 and QWEN3-EMBEDDING-8B.

Agent Corpus Search Cal.Err.

GPT-OSS-20B

Original 13.24 34.64 Multilingual 14.20 42.25 Cross-lingual 14.16 42.56

GPT-OSS-120B

Original 25.35 42.50 Multilingual 30.01 57.78 Cross-lingual 30.45 57.45

QWEN3.6-35B-A3B

Original 13.93 36.18 Multilingual 16.07 44.06 Cross-lingual 16.31 45.90

DEEPSEEK-V4-PRO

Original 28.52 11.94 Multilingual 33.10 16.95 Cross-lingual 34.93 19.42

- Table 2: Search efficiency and calibration error with QWEN3-EMBEDDING-8B. Search denotes average search calls per query; calibration error is reported in percentages.

while BM25 drops sharply under translated evidence, confirming that lexical matching is poorly suited to English queries over non-English documents. Other dense multilingual retrievers recover part of the loss, but still trail the strongest retriever and remain substantially weaker after translation. Thus, standard multilingual retrieval ability does not directly translate into robust retrieval for complex agentic search.

We further examine whether retrieved evidence is used correctly in final answers. Table 3 shows that citation coverage, precision, and recall all decline once evidence is translated. This indicates that language mismatch affects not only retrieval, but also whether retrieved sources are carried through into faithful attribution. We provide a citation-error case study in Appendix G.

The efficiency and calibration trends reinforce this conclusion. Table 2 shows that agents generally issue more searches after evidence is translated, but these additional searches do not recover the lost accuracy. Calibration error also increases in both translated settings, indicating that cross-lingual evidence makes agents not only less accurate, but also less reliable in estimating their own correctness.

#### 4.2.2 Retriever Evaluation

Evidence recall in Table 1 makes the retrieval bottleneck visible. QWEN3-EMBEDDING-8B consistently retrieves the most supporting evidence,

#### 4.2.3 Oracle Retrieval

The oracle setting provides a diagnostic decomposition of the end-to-end results. Table 4 compares the strongest tool-based condition, using QWEN3-EMBEDDING-8B, with an oracle condition in which all supporting evidence is supplied directly. The retrieval/search gap remains large in every corpus condition: oracle retrieval improves accuracy by over 55 pp on the original corpus and by roughly 65–75 pp after translation. Thus, the largest absolute headroom still lies in getting the right evidence into the agent’s context during iterative search.

Agent Corpus Cov. Avg. Cit. Prec. Rec.

Original 50.36 2.38 66.33 29.50 Multilingual 31.93 1.91 45.99 16.65 Cross-lingual 30.36 2.01 42.58 18.78

GPT-OSS-20B

Original 60.60 4.11 50.89 28.99 Multilingual 43.86 3.88 24.30 12.18 Cross-lingual 42.65 3.76 26.26 14.73

GPT-OSS-120B

Original 41.45 3.06 72.39 40.16 Multilingual 24.82 2.78 59.55 31.80 Cross-lingual 25.66 2.74 61.11 31.34

QWEN3.6-35B-A3B

Original 88.07 4.03 85.80 61.30 Multilingual 79.64 3.57 79.64 50.09 Cross-lingual 74.46 3.63 70.76 48.34

DEEPSEEK-V4-PRO

- Table 3: Citation behavior with QWEN3-EMBEDDING8B. Cov., Prec., and Rec. denote citation coverage, citation precision, and citation recall, all in percentages.

Agent Corpus Tool Oracle Ret. Gap Lang. Gap

GPT-OSS-20B

Orig. 32.89 90.36 57.47 – Multi. 12.05 81.20 69.15 9.16 Cross. 11.93 77.59 65.66 12.77

GPT-OSS-120B

Orig. 38.07 94.70 56.63 – Multi. 14.58 89.52 74.94 5.18 Cross. 15.18 85.28 70.10 9.42

QWEN3.6-35B-A3B

Orig. 38.55 93.86 55.31 – Multi. 18.19 90.00 71.81 3.86 Cross. 17.95 89.16 71.21 4.70

- Table 4: Oracle retrieval and error decomposition. Tool accuracy uses QWEN3-EMBEDDING-8B. Ret. Gap is oracle accuracy minus tool-based accuracy under the same corpus condition; Lang. Gap is the drop from original-corpus oracle accuracy to translated-evidence oracle accuracy.

At the same time, oracle retrieval does not eliminate the cross-lingual penalty. Even with all required evidence provided, translated-evidence oracle accuracy remains below original-corpus oracle accuracy for all agents. These gaps reveal an agentside bottleneck beyond retrieval: the model must identify relevant facts, align them with the English question, and synthesize an English answer without losing the evidential constraint. We further decompose this bottleneck using a fully target-language oracle variant in Appendix F.

4.3 Supplementary Analyses

- 4.3.1 Per-Language Decomposition

- Table 5 reports a per-language decomposition for QWEN3.6-35B-A3B with QWEN3-EMBEDDING8B, with English as an untranslated reference and the remaining languages grouped by resource level. Full results for other agent–retriever pairs appear in Appendix H.

Two patterns stand out. First, resource level is most visible before oracle retrieval. Highresource languages average 18.39% tool accuracy

Language N Tool Acc. Ev. Rec. Oracle Acc. O–T Gap Untranslated reference English 70 42.86 49.20 92.86 50.00 High-resource translated languages

Chinese 70 15.71 27.92 91.43 75.72 French 69 26.09 35.15 97.10 71.01 German 69 27.54 37.00 94.20 66.66 Japanese 69 4.35 12.29 73.91 69.56 Korean 69 10.14 17.71 85.51 75.37 Portuguese 69 23.19 29.43 95.65 72.46 Spanish 69 21.74 39.86 89.86 68.12 Low-resource translated languages

Swahili 69 17.39 23.70 89.86 72.47 Wolof 69 14.49 20.50 86.96 72.47 Yoruba 69 7.25 15.39 94.20 86.95 Zulu 69 4.35 12.41 78.26 73.91

High-resource avg. 484 18.39 28.48 89.67 71.28 Low-resource avg. 276 10.87 18.00 87.32 76.45

Table 5: Per-language results in the cross-lingual setting for QWEN3.6-35B-A3B with QWEN3-EMBEDDING8B, plus oracle accuracy for the same agent. All scores are percentages except N. O–T Gap denotes oracle accuracy minus tool-based accuracy. Group averages are weighted by the number of queries and exclude the untranslated English reference.

and 28.48% evidence recall, whereas low-resource languages average 10.87% and 18.00%, respectively. Yet their oracle accuracies remain relatively close, at 89.67% and 87.32%. This suggests that the low-resource penalty in this batch is driven primarily by retrieval failure rather than by an intrinsic inability to answer once evidence is provided. Swahili and Wolof illustrate this most sharply: oracle accuracy stays near 86–90% while tool-based accuracy collapses to roughly 15%.

Second, resource level alone does not explain all variations. Within the high-resource group, French, German, Portuguese, and Spanish substantially outperform Japanese and Korean, with Japanese also showing one of the lowest oracle accuracies; Zulu exhibits an analogous pattern among low-resource languages. Cross-lingual deep research is therefore shaped by two separable but interacting factors: the retriever’s ability to surface evidence across languages, and the agent’s ability to align languagespecific evidence with an English query.

#### 4.3.2 The Impact of Query Expansion

Chen et al. (2026) argue that deep research agents expose a retrieval signal that conventional retrievers ignore: before issuing a search query, the agent often writes a natural-language reasoning trace that clarifies the task intent, summarizes prior findings, and identifies unresolved evidence needs. Their full AGENTIR system trains a retriever to jointly embed the reasoning trace and the issued query. We

study a lighter-weight variant in XBCP: without any retriever training or index changes, we use the agent’s current reasoning trace as query expansion by concatenating it with the issued search query before passing the input to QWEN3-EMBEDDING8B. This isolates whether agent-side reasoning is already useful as a retrieval signal, and whether the benefit survives when the relevant evidence is written in another language.

Corpus Method Acc. Ev. Rec. Cal.Err. Search

Standard 32.89 42.91 34.64 13.24 +Reason. 36.14 47.77 34.02 12.96 ∆ +3.25 +4.86 -0.62 -0.28

Orig.

Standard 12.05 24.60 42.25 14.20 +Reason. 14.10 27.55 41.86 14.18 ∆ +2.05 +2.95 -0.39 -0.02

Multi.

Standard 11.93 23.95 42.56 14.16 +Reason. 14.60 27.00 41.33 13.92 ∆ +2.67 +3.05 -1.23 -0.24

Cross.

- Table 6: AGENTIR-style zero-training query expansion for GPT-OSS-20B with QWEN3-EMBEDDING8B. +Reason. denotes concatenating the agent’s current reasoning trace with the issued query. Acc., Ev. Rec., and Cal.Err. are percentages; Search denotes average search calls per query.

Table 6 shows that reasoning-based expansion consistently improves performance across all three corpus conditions. On the original corpus, accuracy increases by 3.25 pp and evidence recall by 4.86 pp, while calibration error and search turns both decrease. The same pattern holds after translation, although with smaller gains. The improvements therefore do not come from additional exploration, since the expanded runs use slightly fewer search calls on average; rather, the reasoning trace appears to make each search query more informative.

From the perspective of XBCP, this result has two implications. First, cross-lingual deep research should treat query formulation as part of the retrieval problem: the agent’s reasoning can help disambiguate underspecified sub-queries and expose more supporting evidence even without retriever fine-tuning. Second, the smaller gains under translated evidence show that reasoning-aware query expansion is not sufficient by itself. The system still depends on the retriever’s cross-lingual alignment to bridge the language gap.

#### 4.3.3 The Impact of Reasoning Effort

Following BrowseComp-Plus (Chen et al., 2025), we further examine how reasoning effort affects both answer quality and search behavior. This is a particularly important diagnostic for agentic search:

Effort Corpus Acc. Ev. Rec. Search Cal.Err.

Orig. 15.18 18.16 2.05 35.67 Multi. 4.58 9.56 2.01 44.58 Cross. 4.94 9.33 2.01 43.94

Low

Orig. 32.89 42.91 13.24 34.64 Multi. 12.05 24.60 14.20 42.25 Cross. 11.93 23.95 14.16 42.56

Medium

Orig. 36.02 52.53 26.31 23.36 Multi. 15.30 34.09 29.31 36.02 Cross. 15.18 33.44 28.66 37.42

High

Table 7: Impact of reasoning effort for GPT-OSS-20B with QWEN3-EMBEDDING-8B. Acc., Ev. Rec., and Cal.Err. are percentages; Search denotes average search calls per query.

increasing the inference budget may improve final reasoning, but it can also change search iterations and exposed evidence before answering. We therefore vary the reasoning-effort mode of GPT-OSS20B while holding the retriever fixed to QWEN3EMBEDDING-8B. This setup asks whether crosslingual failures can be mitigated by deeper deliberation, or whether language mismatch persists regardless of search effort.

Table 7 shows that higher reasoning effort consistently improves both accuracy and evidence recall. From low to high effort, an increase from 15.18% to 36.02% is observed for the original corpus, and over 10 pp increases are observed for both translated settings. Evidence recall follows the same pattern, increasing in all 3 settings. These gains come with a clear efficiency cost: high effort requires over 26 search calls per query, compared with roughly 2 calls under low effort. Calibration also improves at high effort, suggesting that more extensive search and deliberation make the agent less overconfident.

The comparison with the original corpus is more revealing. High-effort cross-lingual and multilingual runs reach only about the accuracy of the low-effort original run, despite using more than 14 times as many search calls; they remain far below the medium-effort original run. Thus, additional reasoning effort improves the agent in every corpus condition, but it does not turn cross-lingual evidence into a monolingual problem. In conclusion, the dominant difficulty is the language mismatch between the English information need and translated evidence, rather than the specific corpus assignment regime.

### 5 Discussion

Our experiments identify cross-linguality as a structural source of difficulty for deep research agents, not merely as a perturbation to first-stage retrieval. By varying only the evidence language, XBCP isolates how language mismatch propagates through the evidence-seeking pipeline. This design brings together two evaluation traditions that have largely remained separate. Multilingual and cross-lingual retrieval benchmarks (Zhang et al., 2023; Enevoldsen et al., 2025; Zuo et al., 2025) isolate whether a system can rank relevant documents across languages in a fixed collection, while deep research benchmarks (Wei et al., 2025; Chen et al., 2025) evaluate iterative evidence seeking and grounded answer synthesis but typically assume that questions and evidence are linguistically aligned. XBCP connects these views by asking whether cross-lingual retrieval remains effective once it becomes part of an agentic search process.

This perspective first reveals a retrieval/search bottleneck. Our results show that dense multilingual retrievers outperform BM25 after translation. Yet conventional retrieval success does not guarantee that an agent will find the right evidence during iterative search. This gap is consistent with prior work showing that multilingual and cross-lingual retrieval can exhibit different behavior across language regimes and retrieval configurations (Ramponi et al., 2025; Zuo et al., 2025; Zeng et al.,

- 2026). In XBCP, the same issue appears inside the agent loop: translated corpora reduce evidence recall, increase search effort, and lower citation reliability even when the retriever is dense and multilingual. The implication is that cross-lingual retrievers should not be evaluated only by whether they rank relevant documents highly in isolation, but also by whether they expose the evidence at the right point in an agent’s search trajectory.

XBCP also separates this retrieval/search bottleneck from an evidence-integration bottleneck. Recent work on multilingual and cross-lingual RAG has shown that language-mismatched evidence can complicate retrieval, consistency, and reasoning over multilingual contexts (Liu et al., 2025; Ranaldi et al., 2026; Qi et al., 2026). However, these studies remain focused on relatively short-chain single-hop or multi-hop QA settings, leaving the long-horizon deep research setting underexamined. Our oracle results instantiate this distinction: providing all gold evidence substantially raises accuracy, con-

firming that finding evidence is a major bottleneck, but translated oracle accuracy remains below original one. Thus, cross-lingual deep research is decomposable into two linked questions: whether system can find language-mismatched evidence, and whether it can use evidence faithfully once it is found. The latter requires the agent to identify relevant facts in non-English sources, align them with an English question and answer space, and preserve the evidential constraint during synthesis.

The per-language results further suggest that low-resource effects enter the system primarily before evidence reaches the model. Multilingual retrieval evaluation has long emphasized that language resource level, typology, and annotation coverage shape retrieval behavior (Zhang et al., 2023); multilingual LLM research similarly identifies language imbalance and multilingual alignment as central challenges (Xu et al., 2025). In XBCP, low-resource languages show substantially lower tool-based accuracy and evidence recall than high-resource languages, but their oracle accuracy is comparable. It indicates that the largest lowresource penalty appears during retrieval: once strong agents receive the gold documents, they can still extract and integrate the relevant information. Resource-level effects enter the system primarily before evidence reaches the model.

Taken together, these findings point toward language-aware agentic search rather than simply stronger multilingual retrieval. Active retrieval work argues that systems should decide dynamically when and what to retrieve during generation (Jiang et al., 2023), while CLIR research has increasingly moved from translation-based methods toward LLM-based alignment, with cross-lingual representation alignment remaining a central challenge (Goworek et al., 2025). XBCP extends this view to deep research: agents need to recognize the language of available evidence, formulate queries across languages and entity variants, decide when translation or language-specific search is needed, and preserve source attribution across the final answer. Cross-lingual deep research therefore requires coordination between the retriever, query planner, reader, and citation mechanism, so that language-mismatched evidence can be found, interpreted, and cited as part of a single grounded reasoning process.

### Limitations

Our main experiments report a single evaluation run per agent–retriever–corpus configuration. Running agents over full search trajectories with multiple retrievers across three corpus conditions is computationally expensive, and we did not repeat each configuration over multiple random seeds. While the gaps between corpus conditions and between retrievers are large and consistent across agents, formal variance estimates and significance tests over multiple runs are left to future work.

We use a single set of inference hyperparameters per agent, following each model’s recommended generation configuration, without tuning sampling temperature or top-p. This keeps comparisons across conditions controlled, but conditionspecific tuning, particularly for low-resource languages, may partially reduce the observed gaps. A systematic study of inference configuration is beyond the scope of this work.

### References

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335, Bangkok, Thailand. Association for Computational Linguistics.

Zijian Chen, Xueguang Ma, Shengyao Zhuang, Jimmy Lin, Akari Asai, and Victor Zhong. 2026. Agentir: Reasoning-aware retrieval for deep research agents. Preprint, arXiv:2603.04384.

Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, Sahel Sharifymoghaddam, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Nandan Thakur, Crystina Zhang, Luyu Gao, Wenhu Chen, and Jimmy Lin. 2025. Browsecomp-plus: A more fair and transparent evaluation benchmark of deep-research agent. Preprint, arXiv:2508.06600.

DeepSeek-AI. 2026. Deepseek-v4: Towards highly efficient million-token context intelligence.

Kenneth Enevoldsen, Isaac Chung, Imene Kerboua, Márton Kardos, Ashwin Mathur, David Stap, Jay Gala, Wissam Siblini, Dominik Krzemi´nski, Genta Indra Winata, Saba Sturua, Saiteja Utpala, Mathieu Ciancone, Marion Schaeffer, Gabriel Sequeira, Diganta Misra, Shreeya Dhakal, Jonathan Rystrøm, Roman Solomatin, and 67 others. 2025. Mmteb: Massive multilingual text embedding benchmark. arXiv preprint arXiv:2502.13595.

Roksana Goworek, Olivia Macmillan-Scott, and Eda B. Özyi˘git. 2025. Bridging language gaps: Advances in cross-lingual information retrieval with multilingual llms. Preprint, arXiv:2510.00908.

Liang Hu, Jianpeng Jiao, Jiashuo Liu, Yanle Ren, Zhoufutu Wen, Kaiyuan Zhang, Xuanliang Zhang, Xiang Gao, Tianci He, Fei Hu, Yali Liao, Zaiyuan Wang, Chenghao Yang, Qianyu Yang, Mingren Yin, Zhiyuan Zeng, Ge Zhang, Xinyi Zhang, Xiying Zhao, and 4 others. 2025. Finsearchcomp: Towards a realistic, expert-level evaluation of financial search and reasoning. Preprint, arXiv:2509.13160.

Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7969–7992, Singapore. Association for Computational Linguistics.

Wei Liu, Sony Trenous, Leonardo F. R. Ribeiro, Bill Byrne, and Felix Hieber. 2025. XRAG: Cross-lingual retrieval-augmented generation. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 15669–15690, Suzhou, China. Association for Computational Linguistics.

MiroMind Team, Song Bai, Lidong Bing, Carson Chen, Guanzheng Chen, Yuntao Chen, Zhe Chen, Ziyi Chen, Jifeng Dai, Xuan Dong, Wenhan Dou, Yue Deng, Yunjie Fu, Junqi Ge, Chenxia Han, Tammy Huang, Zhenhang Huang, Jerry Jiao, Shilei Jiang, and 36 others. 2026. Mirothinker: Pushing the performance boundaries of open-source research agents via model, context, and interactive scaling. Preprint, arXiv:2511.11793.

OpenAI, :, Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita Brett, Eugene Brevdo, Greg Brockman, Sebastien Bubeck, and 108 others. 2025. gpt-oss-120b & gptoss-20b model card. Preprint, arXiv:2508.10925.

- OpenAI. 2025a. Deep Research System Card.
- OpenAI. 2025b. Introducing gpt-4.1 in the api. https: //openai.com/index/gpt-4-1/.

OpenAI. 2026. GPT-5.4 Thinking System Card.

Thinh Pham, Nguyen Nguyen, Pratibha Zunjare, Weiyuan Chen, Yu-Min Tseng, and Tu Vu. 2026. Sealqa: Raising the bar for reasoning in search-augmented language models. Preprint, arXiv:2506.01062.

Rui Qi, Fengran Mo, Sijin Lu, Yufeng Chen, Jian-Yun Nie, and Kaiyu Huang. 2026. Crosearch-r1: Better leveraging cross-lingual knowledge for retrievalaugmented generation. Preprint, arXiv:2604.25182.

Qwen Team. 2026. Qwen3.6-35B-A3B: Agentic coding power, now open to all.

Alan Ramponi, Marco Rovera, Robert Moro, and Sara Tonelli. 2025. Multilingual vs crosslingual retrieval of fact-checked claims: A tale of two approaches. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 29057–29076, Suzhou, China. Association for Computational Linguistics.

Leonardo Ranaldi, Barry Haddow, and Alexandra Birch. 2026. Multilingual retrieval-augmented generation for knowledge-intensive question answering task. In Findings of the Association for Computational Linguistics: EACL 2026, pages 697–716, Rabat, Morocco. Association for Computational Linguistics.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: Bm25 and beyond. Found. Trends Inf. Retr., 3(4):333–389.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, Kuan Li, Liangcai Su, Litu Ou, Liwen Zhang, Pengjun Xie, Rui Ye, Wenbiao Yin, Xinmiao Yu, Xinyu Wang, and 38 others. 2026. Tongyi deepresearch technical report. Preprint, arXiv:2510.24701.

Tongyi DeepResearch Team. 2025. Tongyi deepresearch: A new era of open-source ai researchers. https://github.com/Alibaba-NLP/ DeepResearch.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Multilingual e5 text embeddings: A technical report. Preprint, arXiv:2402.05672.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. 2025. Browsecomp: A simple yet challenging benchmark for browsing agents. Preprint, arXiv:2504.12516.

Yuemei Xu, Ling Hu, Jiayi Zhao, Zihan Qiu, Kexin Xu, Yuqi Ye, and Hanwen Gu. 2025. A survey on multilingual large language models: corpora, alignment, and bias. Frontiers of Computer Science, 19(11).

Weihao Xuan, Rui Yang, Heli Qi, Qingcheng Zeng, Yunze Xiao, Aosong Feng, Dairui Liu, Yun Xing, Junjue Wang, Fan Gao, Jinghui Lu, Yuang Jiang, Huitao Li, Xin Li, Kunyu Yu, Ruihai Dong, Shangding Gu, Yuekang Li, Xiaofei Xie, and 13 others. 2025. Mmlu-prox: A multilingual benchmark for advanced large language model evaluation. Preprint, arXiv:2503.10497.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Puxuan Yu, Luke Merrick, Gaurav Nuti, and Daniel Campos. 2024. Arctic-embed 2.0: Multilingual retrieval without compromise. Preprint, arXiv:2412.04506.

Qingcheng Zeng, Yuheng Lu, Zeqi Zhou, Heli Qi, Puxuan Yu, Fuheng Zhao, Hitomi Yanaka, Weihao Xuan, and Naoto Yokoya. 2026. Code-switching information retrieval: Benchmarks, analysis, and the limits of current retrievers. Preprint, arXiv:2604.17632.

Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. mGTE: Generalized longcontext text representation and reranking models for multilingual text retrieval. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1393–1412, Miami, Florida, US. Association for Computational Linguistics.

Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. 2023. MIRACL: A multilingual retrieval dataset covering 18 diverse languages. Transactions of the Association for Computational Linguistics, 11:1114–1131.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 embedding: Advancing text embedding and reranking through foundation models. Preprint, arXiv:2506.05176.

Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang, Yifan Shao, Qichen Ye, Dading Chong, Zhiling Jin, Chenxuan Xie, Meng Cao, Yuxin Gu, Sixin Hong, Jing Ren, Jian Chen, Chao Liu, and Yining Hua. 2025. Browsecomp-zh: Benchmarking web browsing ability of large language models in chinese. Preprint, arXiv:2504.19314.

Bin Zhu, Qianghuai Jia, Tian Lan, Junyang Ren, Feng Gu, Feihu Jiang, Longyue Wang, Zhao Xu, and Weihua Luo. 2026. Marco deepresearch: Unlocking efficient deep research agents via verification-centric design. Preprint, arXiv:2603.28376.

Longfei Zuo, Pingjun Hong, Oliver Kraus, Barbara Plank, and Robert Litschko. 2025. Evaluating large language models for cross-lingual retrieval. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 11415–11429, Suzhou, China. Association for Computational Linguistics.

### A XBCP Construction Details

Language Queries Evidence Docs Chinese 70 416 English 70 421 French 69 401 German 69 431 Japanese 69 411 Korean 69 419 Portuguese 69 443 Spanish 69 397 Swahili 69 428 Wolof 69 412 Yoruba 69 443 Zulu 69 418 Total 830 5,040

- Table 8: Language assignment statistics for the crosslingual setting.

Language Documents Chinese 420 English 420 French 420 German 420 Japanese 420 Korean 420 Portuguese 420 Spanish 420 Swahili 420 Wolof 420 Yoruba 420 Zulu 420 Total 5,040

- Table 9: Per-language corpus coverage in the multilingual setting. The 420 English source documents are retained unchanged, and the remaining 4,620 document instances are produced by translation.

- B Translation Prompt Prompt used for document translation

Instruction. Translate the following document completely into {target_language}. Translate everything including proper nouns, titles, terminology, and metadata field names according to

{target_language} conventions. For example, name: should become the equivalent in {target_language}, birth_date: should become the equivalent in {target_language}, etc.

Rules.

- 1. Ensure cultural appropriateness for {target_language} speakers.
- 2. If works such as books, movies, TV shows, songs, or other literary/entertainment titles have wellknown translations in {target_language}, use those established translations.
- 3. Preserve all URLs, email addresses, math formulas, and code blocks unchanged.
- 4. Output only the translated document. Do not add explanations.

- C Translation Verification Rubrics

This translation verification rubrics follows the rubrics conducted by MMLU-ProX (Xuan et al., 2025).

Prompt used for expert translation verification

Instruction. You are an expert bilingual evaluator. Compare the source document with its machine-translated version in the target language. Rate the translation on accuracy, fluency, and completeness using the criteria below. Provide a score from 1 to 5 for each dimension and a brief justification for any score below 5.

Evaluation Criteria for Expert Rating of Machine Translation Results

1. Accuracy (1-5):

- • 5 (Highly Accurate):

- – All key terms and concepts are translated correctly with no errors.
- – Every technical term corresponds precisely to the original text, with no mistranslations or incorrect word choices.
- – The most appropriate and professional terminology in the target language is used.
- – Expressions align with commonly used terminology in professional or technical contexts.

- • 4 (Accurate):

- – Most terms and concepts are translated correctly, with only a few minor errors that do not affect overall comprehension.
- – Some terms may be slightly imprecise, but the translation remains generally accurate.
- – Uses appropriate terminology in the target language in most cases.
- – A few terms may be simplified but remain understandable within the intended domain.

##### • 3 (Moderately Accurate):

- – Key terms and concepts are mostly correct but contain some errors that may cause partial misunderstandings.
- – Some critical terms are inaccurately translated, requiring the reader to infer the intended meaning.
- – Slight deviations in the use of target-language terminology.
- – Occasionally uses uncommon or outdated terms.

##### • 2 (Somewhat Inaccurate):

- – Many key terms and concepts are mistranslated, significantly affecting comprehension.
- – Important concepts are incorrectly translated, leading to potential misunderstandings of the original text.
- – Uses incorrect or inappropriate terminology in the target language.
- – Terminology is inconsistent, reducing the text’s professionalism.

##### • 1 (Inaccurate):

- – Frequent and severe mistranslations of key terms and concepts, failing to convey the original meaning.
- – Most of the content does not match the original text.
- – Lacks proper use of target-language terminology.
- – Terminology is chaotic, possibly using irrelevant or incorrect vocabulary entirely.

2. Fluency (1–5):

- • 5 (Highly Fluent):

- – The target-language expression is natural and smooth, making it effortless to read.
- – The language style is refined and appropriate for professional or formal contexts.
- – The sentence structure fully adheres to natural conventions in the target language, with no grammatical or lexical errors.

- • 4 (Fluent):

- – The target-language expression is generally natural, with only minor linguistic imperfections that do not affect comprehension.
- – Some sentences may sound slightly stiff.
- – Sentence structures mostly conform to targetlanguage norms, with very few grammatical errors.

- • 3 (Moderately Fluent):

- – The target-language expression is somewhat unnatural, requiring the reader to adjust their understanding slightly.
- – Some inappropriate word choices or rigid sentence structures are present.
- – Sentence structures are mostly correct, but some grammatical errors exist.

##### • 2 (Somewhat Unnatural):

- – The target-language expression lacks fluency, making it difficult to read smoothly.
- – Sentence transitions are awkward, and logical connections are unclear.
- – Many structural issues exist, with frequent grammatical errors.

##### • 1 (Not Fluent):

- – The target-language expression is highly unnatural or difficult to understand.
- – Literal translation is evident, lacking natural phrasing in the target language.
- – The sentence structure is disorganized, with severe grammatical mistakes, making the text unreadable.

3. Completeness (1–5):

- • 5 (Fully Complete):

- – The full meaning of the original text is retained with no omissions or additions.
- – All details, data, and annotations are accurately conveyed.
- – The translation maintains the same length and depth as the original text.

- • 4 (Complete):

- – The primary meaning of the original text is retained, with only a few minor details omitted or slightly unclear.
- – Some less critical information may be left out.
- – The translation generally corresponds to the original content.

- • 3 (Moderately Complete):

- – Most of the original meaning is conveyed, but some information is missing or added.
- – Important details may be overlooked.
- – The translation differs from the original in certain aspects, requiring readers to infer some content.

- • 2 (Somewhat Incomplete):

- – The core information from the original text is not fully conveyed, with noticeable omissions or unnecessary additions.
- – Potential inclusion of unrelated information.
- – The translation does not fully correspond to the original, affecting comprehension.

- • 1 (Incomplete):

- – Significant omissions or added incorrect information prevent an accurate reflection of the original text.
- – Important sections or sentences are missing.
- – The translation deviates heavily from the original, making it difficult to understand the intended meaning.

### D Translation Verification Results

- Table 10 in this appendix report per-language translation results in our corpora. For each language in translation, we adopt three dimensions in evaluation: accuracy, fluency and completeness. Each language evaluation has 200 samples and the results are reported in average value.

Language Accuracy Fluency Completeness

Chinese 4.82 4.88 4.74 French 4.91 4.90 4.86 German 4.74 4.78 4.68 Japanese 4.44 4.66 4.48 Korean 4.46 4.48 4.52 Portuguese 4.89 4.88 4.88 Spanish 4.87 4.86 4.84 Swahili 4.52 4.68 4.70 Wolof 4.18 4.34 4.38 Yoruba 4.12 4.42 4.36 Zulu 4.10 4.22 4.28

- Table 10: Per-language translation verification results. All values are on 1–5 scale.

- E Judge Prompt Prompt used for LLM-as-Judge

Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question} [response]: {response} [correct_answer]: {correct_answer}

The evidence documents used to answer this question are in another language. As a result, the extracted answer may be written in another language rather than English. The [correct_answer] is in English. You must judge whether the extracted answer and the correct answer refer to the same entity, concept, or value, regardless of language differences. For example, “ハーバード大学” and Harvard University, “迈克尔·乔丹“ and “Michael Jordan”, or “ᄉ울” and ”Seoul” should be considered equivalent.

Your judgement must be in the format and criteria specified below:

- 1. extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as “None” if there is no exact, final answer to extract from the response.
- 2. reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on whether they refer to the same entity or value. If they are in different languages, determine whether they are translations or transliterations of each other. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.
- 3. correct: Answer “yes” if extracted_final_answer matches the [correct_answer] given above, or is a translation/transliteration of it, or is within a small margin of error for numerical problems. Answer “no” otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
- 4. confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.

- F Decomposing the Agent Cross-lingual Bottleneck

Our oracle experiments show that providing gold evidence directly to the agent does not fully recover monolingual performance, revealing an agent-side cross-lingual bottleneck. A natural follow-up question is whether this bottleneck arises because the agent must reason over non-English evidence, or because it must also switch between an English prompt and non-English content. To disentangle these factors, we introduce a fully target-language

oracle variant (ORACLE-TQ+TP), in which the system prompt, the query, and the evidence documents are all presented in the target language. This removes any language switching and tests whether a monolingual non-English environment helps the agent reason more effectively.

Oracle Variant GPT-OSS-20B GPT-OSS-120B

EN Oracle (en→en) 90.36 94.70 Oracle (en→xx) 77.59 85.28 Oracle-tq+tp (xx→xx) 71.67 79.90

- Table 11: Oracle accuracy (%) on the cross-lingual corpus under three prompt–evidence language configurations. EN Oracle: English prompt + English evidence (upper bound). Oracle: English prompt + target-language evidence. Oracle-tq+tp: target-language prompt + target-language evidence.

Table 11 shows that, contrary to our expectation, Oracle-tq+tp performs worse than the standard Oracle with English prompt: GPT-OSS-20B drops by 5.92pp and GPT-OSS-120B by 5.38pp. The agent reasons less effectively when the prompt is also in the target language, even though language switching is eliminated. This reveals that the agent’s crosslingual weakness has two distinct components:

- 1. Evidence understanding bottleneck (EN Oracle → Oracle): the agent loses 12.77pp (20B) / 9.42pp (120B) from reading non-English evidence, even under English instructions.
- 2. Prompt language penalty (Oracle → Oracletq+tp): switching the prompt to the target language costs an additional 5.92pp (20B) / 5.38pp (120B), indicating that these models follow instructions more reliably in English.

These results have two implications. First, the agent bottleneck is intrinsic to the model’s multilingual reasoning capability, not a surface-level language-switching artifact. Providing a fully monolingual target-language environment does not help; it makes things worse. Second, English serves as the agent’s “native language” for instruction following: even when all content is non-English, the agent benefits from receiving its task description in English. This suggests that improving cross-lingual agent performance requires stronger multilingual pretraining, not prompt translation.

### G Citation Precision Error Analysis

GPT-OSS-120B exhibits the steepest citation precision drop among all agents: from 50.89% on the original corpus to 24.30% (multilingual) and

26.26% (cross-lingual), a reduction of roughly 50% (Table 3). To diagnose this degradation, we classify every query where GPT-OSS-120B made citations but failed to cite any gold evidence document into two mutually exclusive error types: (1) the agent retrieved at least one gold document but cited other documents instead (mapping failure); (2) no gold document was retrieved and the agent cited English negative documents instead (no gold retrieved). We include GPT-OSS-20B and QWEN3.6-35B-A3B as reference points in Table 12.

Agent Corpus Prec. Errors Map.Fail No Gold

Orig. 50.89 226 97 (42.92%) 129 (57.08%) Multi. 24.30 272 104 (38.24%) 168 (61.76%) Cross. 26.26 248 84 (33.87%) 164 (66.13%)

GPT-OSS-120B

Orig. 66.33 172 85 (49.42%) 87 (50.58%) Multi. 45.99 165 57 (34.55%) 108 (65.45%) Cross. 42.58 166 72 (43.37%) 94 (56.63%)

GPT-OSS-20B

Orig. 72.39 108 50 (46.30%) 58 (53.70%) Multi. 59.55 98 29 (29.59%) 69 (70.41%) Cross. 61.11 90 22 (24.44%) 68 (75.56%)

QWEN3.6-35B

Table 12: Citation error classification with QWEN3EMBEDDING-8B. Prec. is citation precision (%) among queries with citations. Errors is the number of queries that cited zero gold documents. Map.Fail: gold was retrieved but agent cited other documents. No Gold: no gold document was retrieved. Percentages in parentheses sum to 100% within each row.

For GPT-OSS-120B, the dominant error type is no gold retrieved, accounting for 57.08% of errors on the original corpus and rising to 66.13% on the multilingual corpus. In these cases, the retriever never surfaced the gold document during the agent’s search trajectory, so the agent cited English negative documents that appeared topically related but did not contain the correct evidence. Mapping failures account for the remaining 33.87–42.92% of errors and decline as a share after translation, not because the agent improves at citation mapping, but because fewer gold documents are retrieved in the first place.

Compared with GPT-OSS-20B and QWEN3.635B-A3B, GPT-OSS-120B has substantially more total errors (226–272 vs. 108–172). This is driven by its higher citation coverage (60.6% vs. 50.4% and 41.5%): the 120B model cites documents more frequently, creating more opportunities for incorrect citations.

### H Additional Per-Language Results

All tables in this appendix report per-language results in the cross-lingual setting. Q3-4B and Q38B denote QWEN3-EMBEDDING-4B and QWEN3EMBEDDING-8B, respectively.

Language N BM25 Q3-4B Q3-8B E5 Arctic

- Chinese 70 0.00 8.57 8.57 1.43 7.14 English 70 17.14 27.14 42.86 20.00 30.00 French 69 4.35 13.04 15.94 0.00 17.39

- German 69 4.35 33.33 15.94 1.45 18.84 Japanese 69 0.00 1.45 1.45 2.90 1.45 Korean 69 4.35 4.35 5.80 4.35 1.45 Portuguese 69 4.35 18.84 18.84 2.90 17.39

- Spanish 69 1.45 15.94 13.04 1.45 26.09 Swahili 69 2.90 10.14 10.14 2.90 1.45 Wolof 69 0.00 5.80 5.80 1.45 2.90 Yoruba 69 1.45 2.90 2.90 0.00 7.25 Zulu 69 1.45 0.00 1.45 1.45 0.00 Total 830 3.49 11.81 11.93 3.37 10.96

- Table 13: Per-language tool-based accuracy for GPTOSS-20B. All values are percentages.

Language N BM25 Q3-4B Q3-8B E5 Arctic Chinese 70 1.43 10.00 5.71 1.43 10.00 English 70 30.00 38.57 40.00 34.29 37.14 French 69 7.25 24.64 18.84 2.90 17.39 German 69 5.80 33.33 26.09 1.45 24.64 Japanese 69 2.90 1.45 2.90 1.45 5.80 Korean 69 4.35 2.90 8.70 2.90 7.25 Portuguese 69 2.90 26.09 20.29 5.80 26.09 Spanish 69 2.90 14.49 21.74 5.80 23.19 Swahili 69 2.90 13.04 17.39 4.35 11.59 Wolof 69 2.90 7.25 5.80 4.35 1.45 Yoruba 69 1.45 5.80 7.25 0.00 7.25 Zulu 69 0.00 5.80 7.25 1.45 1.45 Total 830 5.42 15.30 15.18 5.54 14.46

- Table 14: Per-language tool-based accuracy for GPTOSS-120B. All values are percentages.

Language N BM25 Q3-4B Q3-8B E5 Arctic Chinese 70 1.43 15.71 15.71 2.86 17.14 English 70 18.57 42.86 42.86 25.71 35.71

- French 69 5.80 18.84 26.09 1.45 18.84 German 69 10.14 31.88 27.54 4.35 33.33

- Japanese 69 0.00 5.80 4.35 2.90 2.90 Korean 69 5.80 8.70 10.14 4.35 8.70 Portuguese 69 7.25 20.29 23.19 8.70 20.29 Spanish 69 1.45 15.94 21.74 1.45 20.29 Swahili 69 1.45 13.04 17.39 5.80 18.84 Wolof 69 5.80 11.59 14.49 4.35 5.80 Yoruba 69 2.90 11.59 7.25 0.00 8.70 Zulu 69 1.45 1.45 4.35 2.90 0.00 Total 830 5.18 16.51 17.95 5.42 15.90

- Table 15: Per-language tool-based accuracy for QWEN3.6-35B-A3B. All values are percentages.

Language N BM25 Q3-4B Q3-8B E5 Arctic Chinese 70 0.71 20.99 22.75 0.82 18.16 English 70 28.38 37.31 48.93 25.75 41.18 French 69 6.80 26.41 29.59 5.39 24.35

- German 69 7.69 49.43 36.92 2.21 34.51

Japanese 69 1.57 9.28 10.03 1.10 12.78 Korean 69 0.95 19.23 17.68 1.14 15.95

- Portuguese 69 2.12 29.06 24.54 3.74 22.32 Spanish 69 3.35 28.85 34.73 1.92 34.04 Swahili 69 3.50 18.37 20.85 2.18 14.12 Wolof 69 2.77 18.87 16.09 2.77 7.29 Yoruba 69 4.46 12.01 13.99 3.76 15.98 Zulu 69 4.48 7.49 10.99 1.80 4.13 Total 830 5.59 23.12 23.95 4.40 20.42

Table 16: Per-language evidence recall for GPT-OSS20B. All values are percentages.

Language N BM25 Q3-4B Q3-8B E5 Arctic Chinese 70 0.93 28.74 27.32 1.54 21.75 English 70 39.85 50.82 49.60 34.31 45.33 French 69 11.41 34.28 31.12 5.58 28.06 German 69 7.70 48.48 45.79 3.27 36.86 Japanese 69 1.83 13.14 10.78 2.43 17.60 Korean 69 0.57 20.15 22.32 1.49 17.34

- Portuguese 69 3.63 38.70 31.40 6.00 31.96 Spanish 69 8.10 36.03 45.56 7.09 39.05 Swahili 69 6.14 18.39 31.20 3.94 21.40 Wolof 69 7.61 24.30 17.86 6.71 10.72 Yoruba 69 6.97 14.27 16.96 2.44 15.52 Zulu 69 3.76 14.39 16.05 4.04 7.11 Total 830 8.24 28.50 28.85 6.60 24.41

- Table 17: Per-language evidence recall for GPT-OSS120B. All values are percentages.

### I Tongyi-DeepResearch Results

We additionally evaluate TONGYIDEEPRESEARCH-30B-A3B (Tongyi DeepResearch Team, 2025), a deep research agent built on a Qwen3-based MoE architecture. Unlike the other agents in our study, Tongyi uses an in-band ReAct-style tool calling protocol with <tool_call> XML tags rather than the OpenAI function-calling API. Table 21 reports its performance with BM25 and QWEN3-EMBEDDING-8B across all three corpus conditions. Tongyi’s ReActstyle output format does not reliably produce

Language N BM25 Q3-4B Q3-8B E5 Arctic Chinese 70 0.57 24.21 27.92 0.29 17.23 English 70 30.74 49.38 49.20 26.56 44.41 French 69 5.18 27.47 35.15 2.68 27.47 German 69 5.86 36.73 37.00 1.71 28.58 Japanese 69 1.90 13.54 12.29 1.59 8.49 Korean 69 0.24 21.95 17.71 0.74 16.57 Portuguese 69 3.90 28.09 29.43 5.41 23.62 Spanish 69 3.61 33.65 39.86 2.12 33.91 Swahili 69 3.60 18.40 23.70 2.64 19.52 Wolof 69 6.23 16.61 20.50 5.16 8.66 Yoruba 69 5.79 12.33 15.39 2.25 13.22 Zulu 69 5.28 10.52 12.41 3.42 4.18 Total 830 6.10 24.44 26.74 4.57 20.51

- Table 18: Per-language evidence recall for QWEN3.635B-A3B. All values are percentages.

Language N BM25 Acc. BM25 Rec. Q3-8B Acc. Q3-8B Rec. Chinese 70 10.00 3.40 51.43 56.94 English 70 65.71 69.28 71.43 75.40 French 69 13.04 19.09 50.72 58.59 German 69 21.74 21.66 57.97 70.51 Japanese 69 10.14 6.15 20.29 40.13 Korean 69 4.35 0.65 42.03 47.04 Portuguese 69 14.49 12.62 56.52 60.49 Spanish 69 10.14 11.81 42.03 63.77 Swahili 69 11.59 18.57 34.78 51.30 Wolof 69 15.94 16.39 31.88 42.72 Yoruba 69 17.39 24.78 24.64 44.22 Zulu 69 14.49 17.21 23.19 34.35 Total 830 17.47 18.51 42.29 53.82

- Table 19: Per-language tool-based performance for DEEPSEEK-V4-PRO. Acc. and Rec. denote accuracy and evidence recall; all values are percentages.

Language N OSS-20B OSS-120B Qwen3.6 Chinese 70 80.00 77.14 91.43 English 70 91.43 92.86 92.86 French 69 89.86 91.30 97.10 German 69 88.41 92.75 94.20 Japanese 69 57.97 75.36 73.91 Korean 69 66.67 76.81 85.51 Portuguese 69 89.86 97.10 95.65 Spanish 69 86.96 92.75 89.86 Swahili 69 76.81 83.82 89.86 Wolof 69 60.87 79.71 86.96 Yoruba 69 75.36 86.96 94.20 Zulu 69 66.67 76.81 78.26 Total 830 77.59 85.28 89.16

- Table 20: Per-language oracle accuracy in the crosslingual setting. All values are percentages.

per-query confidence scores despite prompt-level instructions, making the metric calibration error unreliable. Therefore, we exclude it from our main results but put it in the appendix for reference.

Retriever Corpus Acc. Ev.Rec. Search

Orig. 22.77 34.06 22.3 Multi. 9.88 17.64 35.9 Cross. 8.80 17.01 35.6

BM25

Orig. 39.64 58.12 27.7 Multi. 26.14 48.18 34.4 Cross. 25.06 44.97 34.5

Q3-8B

Table 21: TONGYI-DEEPRESEARCH-30B-A3B results. Acc. and Ev.Rec. are percentages; Search is average search calls per query. Q3-8B denotes QWEN3EMBEDDING-8B. Calibration error is omitted because Tongyi’s ReAct-style output format does not reliably produce per-query confidence scores despite promptlevel instructions, making the metric unreliable.

Tongyi achieves 39.64% accuracy on the original corpus with QWEN3-EMBEDDING-8B, the highest among all agents at comparable parameter counts. Its evidence recall (58.12%) also exceeds GPT-OSS-20B (42.91%) and QWEN3.6-35BA3B (43.14%). After translation, accuracy drops by 13.50–14.58pp with QWEN3-EMBEDDING-8B, a smaller relative degradation than GPT-OSS-20B (20.84–20.96pp).

### J Inference Hyperparameters

For each agent we follow the generation configuration recommended by the model release, applied uniformly across all corpus conditions and evidence languages. GPT-OSS-20B and GPTOSS-120B are served locally with vLLM in temperature 1.0, top-p 1.0. QWEN3.6-35B-A3B is served locally with vLLM in temperature 0.7, topp 0.8. DEEPSEEK-V4-PRO is accessed through its official API in default settings(temperature 1.0, top-p 1.0). All other generation parameters are left at each model’s default value.

### K License Statement

BrowseComp-Plus. Our benchmark, XBCP, is derived from BrowseComp-Plus (Chen et al., 2025), which is released under the MIT License. We use BrowseComp-Plus in accordance with the MIT License terms, retaining the original copyright notice and license text in all derived artifacts.

Models. We use the following models under their respective licenses: GPT-OSS-20B (OpenAI et al.,

- 2025), GPT-OSS-120B (OpenAI et al., 2025), QWEN3-EMBEDDING-4B, QWEN3-EMBEDDING8B (Zhang et al., 2025), and ARCTIC-EMBED-L-

2.0 (Yu et al., 2024) are released under the Apache License 2.0; MULTILINGUAL-E5-LARGE (Wang et al., 2024) is released under the MIT License. QWEN3.6-35B-A3B (Qwen Team, 2026) is released under the Apache License 2.0 and is used locally via vLLM. DEEPSEEK-V4-PRO (DeepSeekAI, 2026), whose model weights are released under the MIT License, is accessed in our experiments through its official API under the DeepSeek Open Platform Terms of Service. GPT-5.4 (OpenAI,

- 2026), a proprietary model accessible only through OpenAI’s API, is used solely to generate translations for the XBCP evidence corpora; its outputs are used in accordance with OpenAI’s Terms of Use, which grant users ownership of model outputs subject to OpenAI’s usage policies. All use of these models is for non-commercial academic research.

Release. We will release XBCP under the MIT License, consistent with the license of the underlying BrowseComp-Plus benchmark. The release will include the translated evidence corpora, query–language assignments, and evaluation scripts, with attribution to BrowseComp-Plus and to each model whose outputs contributed to the construction of the benchmark.

### L GenAI Statement

We disclose the use of generative AI tools in this work in accordance with the ACL Policy on the Use of AI Writing Assistance.

AI use in research artifacts. Generative AI played a central role in constructing the XBCP benchmark. Specifically, we used GPT-5.4 (OpenAI, 2026) as the translation engine to render the English evidence documents of BrowseComp-Plus (Chen et al., 2025) into the eleven non-English target languages used in our cross-lingual and multilingual corpora. The exact prompt is provided in Appendix B. Translation quality was assessed through expert human verification on samples of all eleven non-English using the rubric in Appendix C; we discuss the implications and limitations of automatic translation in the Limitations section.

AI use in experiments. The agents and retrievers evaluated in this work are themselves LLMbased or neural systems (GPT-OSS-20B, GPTOSS-120B, QWEN3.6-35B-A3B, DEEPSEEK-

V4-PRO, and four multilingual embedding models). Their use is the subject of study rather than an auxiliary tool, and is fully described in Section 4.

AI use in writing. We used AI assistants (Claude and ChatGPT) for surface-level writing support, including grammar correction, sentencelevel rephrasing for clarity and concision, and LaTeX formatting suggestions. All scientific claims, experimental design choices, analyses, and conclusions are authored and verified by the human authors. AI assistants were not used to generate citations, statistical results, or any factual content reported in this paper.

Responsibility. The authors take full responsibility for the content of this paper, including any text that may have been initially drafted or edited with AI assistance.

### M Ethics

XBCP is a translation-based benchmark for evaluating deep research agents. Translations are produced by GPT-5.4, and despite expert verification on a sample, residual translation artifacts may propagate into low-resource-language evaluation, potentially under- or over-estimating system performance for those languages. XBCP is derived from BrowseComp-Plus, which is built from publicly available web documents. We do not collect new personal data from individuals. The benchmark therefore inherits the question scope of BrowseComp-Plus and is intended for research evaluation, not for deployment-grade safety claims.

Expert bilingual annotators were recruited through commercial language-service companies. They were compensated according to standard professional translation-evaluation rates.

