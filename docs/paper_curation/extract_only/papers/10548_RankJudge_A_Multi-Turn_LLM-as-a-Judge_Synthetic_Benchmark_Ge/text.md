# arXiv:2605.21748v1[cs.CL]20May2026

## RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator

#### Zhenwei Tang1,2∗ Zhaoyan Liu1 Rasa Hosseinzadeh1 Tongzi Wu1 Keyvan Golestan1 Jesse C. Cresswell1 1Layer 6 AI 2University of Toronto {joseph, zhaoyan, rasa, tongzi, keyvan, jesse}@layer6.ai

Code Dataset Leaderboard

[Figure 1]

### Abstract

As interactive LLM-based applications are created and refined, model developers need to evaluate the quality of generated text along many possible axes. For simpler systems, human evaluation may be practical, but in complicated systems like conversational chatbots, the amount of generated text can overwhelm human annotation resources. Model developers have begun to rely heavily on auto-evaluation, where LLMs are also used to judge generation quality. However, existing LLMas-a-judge benchmarks largely focus on simple Q&A tasks that do not match the complexity of multi-turn conversations. We introduce RankJudge, a benchmark generator for evaluating LLM-as-a-judge on multi-turn conversations grounded in reference documents. RankJudge creates pairs of conversations where one conversation has a single flaw injected into one turn. This construction allows paired conversations to be labeled unambiguously as better or worse, and precisely isolates failure categories to individual turns, enabling a strict joint correctness criterion for judging. We implement RankJudge across the domains of machine learning, biomedicine, and finance, evaluate 21 frontier LLM judges, and rank those judges via the Bradley-Terry model. Our formulation also allows ranking each conversation pair with difficulty ratings, which we use to dynamically curate the evaluation slice to reduce label noise, as confirmed via human annotation. We find that judge rankings are stable under partial observability, coarser correctness criteria, and an alternative random-walk rating algorithm.

### 1 Introduction

Large language models are increasingly evaluated by other large language models (LLMs). Pairwise judging [1, 2] has become the dominant scalable substitute for human preference collection. As judge models are now used to score training data, gate releases, and rank checkpoints, judge quality has itself become a central assumption. A leaderboard built on a weak judge may silently reward the wrong behavior. Stress-testing the judges themselves is therefore a first-order problem, and one that existing judge benchmarks address only partially. Current judge benchmarks have recurring shortcomings: the dialogues that real-world LLM assistants produce are multi-turn and referencegrounded, while most judge benchmarks score isolated single-turn responses. Failure modes that matter in deployment, such as a later turn contradicting an earlier one or a content-level constraint silently dropping after several turns [3, 4], simply cannot surface in the single-turn setting. Another shortcoming is that verdict-only correctness conflates “picked the right side” with “understood why”: a judge that prefers the better conversation while misattributing the flaw to the wrong turn or category has reached the right conclusion through the wrong reasoning, and existing leaderboards cannot tell

∗Work done during an internship at Layer 6 AI.

Preprint.

Semi-Automated Assistant Failure Type Discovery

3 Automated Quality Control

- 1 2

Dual-Conditioned Conversation Pair Generation

- ✔ Coherence
- ✔ Adherence
- ✔ Grounding

Veriﬁed Pairs

User Behavior Type

Assistant Failure Type

Seed Failure Types

Real Conversations

4 Joint Ranking of Judges and Problems

Plan & Generate Independently

LLM Clustering

Domain-Diverse Judge Problems

Pair Difﬁculty

LLM Judges

Strength

Better Conversation Worse Conversation

5 Benchmark Curation

- Turn 1

Turn 3: Injected Flaw Failure Type: self_contradiction

- Turn 2

- Turn 3

- Turn 4

- Turn 1

- Turn 2

Elo Score

Human Review

###### Label Noise Reduction

Clean

Turn 4

Pairs

Figure 1: Overview of RANKJUDGE, a benchmark generator for multi-turn judge evaluation.

the two apart. Lastly, static accuracy on a fixed pool offers no principled way to identify which items actually separate strong judges from weak ones [5, 6, 7].

In this paper, we introduce RANKJUDGE, a benchmark generator for multi-turn, reference-grounded judge evaluation. Each item is a pair of conversations sampled independently from the same reference document under two conditioning axes: a user behavior archetype, and a targeted assistant failure type, with the failure injected into exactly one turn of the worse branch. As shown in Figure 1, because the flaw is preconstructed by the generator, the ground-truth tuple of better conversation, flawed turn, and failure category is uniquely determined per item from the generation prompt itself, before any judge sees the pair. At evaluation time, we ask each judge for a joint prediction over verdict, turn, and type, and credit it only when all three components match. This consistency check distinguishes correct judgments from correct guesses.

The benchmark construction is fully synthetic, with no per-item human label required, which lets us scale coverage densely and regenerate the pool deterministically whenever the generator or verifier is upgraded. We ensure the accuracy of our labels using two complementary methods. First, a three-layer automated verifier cascade checks for coherence, adherence, and grounding, and only keeps a pair of conversations when the targeted flaw is isolated to the correct turn. Additionally, every other claim in both conversations must be fully supported by the source. Second, we use the Bradley-Terry model to analyze how the judges scored the test pairs [2]. This gives us a calibrated difficulty rating for each pair and allows us to dynamically curate a polished evaluation slice by removing the items with the very highest difficulty scores, i.e., top-Elo tail. Both a human audit and a held-out fine-tuning experiment independently flagged this tail as the subset containing label noise.

We apply RANKJUDGE to produce three benchmarks in distinct knowledge-intensive domains: Machine Learning, Biomedicine, and Finance, and evaluate 21 frontier judges spanning proprietary and open-weight families on each. The leaderboard separates judges across nearly 1200 Elo points, and several open-weight checkpoints outrank frontier proprietary judges. The bipartite framing also admits partial observability, so judges can be scored on different subsets of pairs while retaining their positions on the same scale, which lowers the required compute. The resulting ranking is stable under match subsampling, under a coarser correctness criterion, and under an Empirical Interaction Propagation (EIP) cross-check [8]. RANKJUDGE also surfaces a model-capability ceiling: weaker judges collapse their predictions onto a single failure category rather than scattering across the taxonomy, and targeted prompt rewrites of a mid-ranked frontier judge fail to lift it onto the accuracy-cost Pareto frontier, exposing a capability gap that prompting cannot close. We summarize our contributions as follows:

- • RANKJUDGE is a benchmark generator for multi-turn, reference-grounded judge evaluation whose ground-truth verdict, flawed turn, and failure type are specified in the generation prompt and then scored under a joint correctness criterion.
- • A semi-automated discovery loop surfaces multi-turn assistant failure types, and dual-conditioned generation independently simulates user-behavior and assistant-failure axes.
- • Construction is fully synthetic; a three-layer automated verifier and Elo-based curation of the high-difficulty tail are validated by a human audit and a held-out fine-tuning experiment that independently flag a substantially overlapping noisy slice.

- • Instantiations in Machine Learning, Biomedicine, and Finance produce leaderboards spanning proprietary and open-weight judge families, which remain stable under various conditions, and surface a systematic class bias in weaker judges.

### 2 Related Work

Multi-turn LLM Benchmarks. LLM evaluation has shifted from single-turn benchmarks like MMLU [5] and GSM8K [9], which miss the user-model-environment dynamics that drive real-world utility [10, 11], to multi-turn frameworks [1, 12, 13, 14] that probe correctness, helpfulness, and interactive patterns [15]. A consistent finding emerges across these works: single-turn ability does not transfer to multi-turn success [10], and frontier models degrade sharply across turns due to compounding unreliability [4]. These dynamics motivate our focus on multi-turn, reference-grounded conversations as the setting in which judge quality must itself be stress-tested.

LLM-as-a-Judge. Reward models are crucial for aligning and improving the capabilities of LLMs [16, 17]. The traditional scalar reward model [18] gives a single “verdict” indicating the response quality. However, scalar models suffer from certain limitations, for example they are vulnerable to hacking [19], and lack the ability to localize or categorize specific errors. LLMs have demonstrated strong capability to mimic human reasoning and evaluate inputs based on predefined criteria while being scalable and effective. The concept of LLM-as-a-judge [1, 20, 21] has become widely used for tasks like providing rich reward signals for LLM alignment [22], producing chain-of-thought (CoT) reasoning along with a final judgment as evaluators [23, 24], and data annotation [25, 26]. These judges can be implemented either via direct prompting of general-purpose LLMs [1, 27] or as specialized fine-tuned evaluators [28, 29]. Existing frameworks typically adopt either pointwise or pairwise evaluation protocols. Pointwise methods [21, 23] score responses independently, while pairwise methods [1, 28] compare responses to predict relative preferences.

Benchmarking LLM-as-a-Judge. MTBench [1, 38] helped establish LLM-as-a-judge evaluation for chat assistants by reporting agreement with humans. Initial works on metaevaluation of judges focused on single-turn settings: LLMBar [30] uses natural and adversarial pairwise examples, DHP [33] measures natural language generation evaluation capabilities using perturbations, ReIFE [34] varies LLMs, protocols and datasets. JudgeBench [31] converts factuality and correctness datasets into benchmarks for meta-evaluation, JuStRank [35] studies judges through systems level ranking agreement with human rankings, and ContextualJudgeBench [32] grounds evaluation in external documents. Other works study LLM-as-a-Judge for code evaluation [39] and evaluator adversarial robustness [40], positional bias [41], and fairness [42]. MEDAL [37] is closest to our setting since it generates multilingual multi-turn dialogues using a multi-agent pipeline and automates labeling with GPT-4.1, followed by filtering with human curation for the final benchmark. Table 1 compares RANKJUDGE with prior benchmarks across several axes: ours is the first automated pipeline for generating a multi-turn judge benchmark that is grounded in external documents, conditioned on user behavior, and built by injecting controlled error types.

Turn Drv. Usr. Doc. Diag. Pipe. Eval.

LLMBar [30] S – – – – – Acc. JudgeBench [31] S ✓ – – – △ Acc. ContextualJudgeBench [32] S △ – ✓ △ △ Acc. DHP [33] S ✓ – – △ ✓ Disc. ReIFE [34] S – – – – – Acc. JuStRank [35] S – – – – – Rank JudgeArena [36] S – – – – – Rank MEDAL [37] M △ △ – △ ✓ Acc.

RankJudge M ✓ ✓ ✓ ✓ ✓ Rank

Table 1: Comparison of LLM judge meta evaluation benchmarks. Turn: S = single turn, M = multi turn. Drv. = derived meta evaluation target: ✓ means without new human preference labeling, △ means some manual curation and – means completely dependent on human judgments. Usr. = controlled user behavior. Doc. = document grounding. Diag. = diagnostic labels beyond a winner or score. Pipe. = automated benchmark generation where △ means partially. Eval.: Acc. = item level accuracy or agreement; Disc. = perturbation discernment; Rank = judge or system ranking.

### 3 Methodology

Let T = {t1,...,tK} denote a taxonomy of K assistant failure types. A multi-turn conversation C = (u1,a1,...,uR,aR) consists of R turns, each a (user, assistant) message pair. Each benchmark item is a tuple (CA,CB,y⋆,r⋆,t⋆) in which CA and CB are two conversations grounded in the same

- Table 2: Multi-turn assistant-failure taxonomy. In each pair, one conversation exhibits one flaw from the available failure types in exactly one round; the other exhibits no clear flaw throughout. Failure types (left) are the ground-truth labels t⋆.

Failure Type Behavior

Flaw. A later turn contradicts an earlier correct claim by shifting a qualifier or reframing. Benign. Consistent scope and qualifiers across turns; flags genuine changes in position.

self_contradiction

Flaw. Hedges or drifts around the central claim, citing peripheral details without the figures or terms needed to verify or act. Benign. Answers the question directly with the exact figures and terms required.

evasion

Flaw. Buries the key answer in a muddled structure, mixing high- and low-level points so correct content is costly to extract. Benign. Leads with the key point and arranges supporting detail in a scannable structure.

disorganized

Flaw. When a question is unanswerable from the source, invents a confident answer instead of admitting insufficient information. Benign. Calibrated to the source: confident when supported, explicit about limits otherwise.

fabricated_answer

Flaw. A directive set early (scope, audience, what to skip) is explicitly violated in a later turn while the answer remains plausible. Benign. Tracks the early directive across turns, surfacing genuine conflicts rather than deviating.

instruction_forgetting

Flaw. On an ambiguous question, picks one interpretation and answers confidently rather than asking the user to clarify. Benign. Clarifies on truly ambiguous questions; answers unambiguous ones directly.

no_clarification

Flaw. Refuses on a question that is answerable from the source, citing insufficient information when the answer is available. Benign. Answers confidently from available material; hedges only when genuinely out of scope.

unnecessary_refusal

reference documents, y⋆ ∈ {A,B} identifies the better conversation, r⋆ ∈ {1,...,R} is the turn of the single injected flaw in the worse conversation, and t⋆ ∈ T is its failure category. At turn r⋆, one flaw type t⋆ is injected, making (y⋆,r⋆,t⋆) uniquely determined per item (see Section 3.1). A judge is a function J : (CA,CB)  → (ˆy,r,ˆ tˆ) that jointly predicts the better conversation, the flawed turn, and the failure category of the flawed turn. This joint prediction enables a check on the judge’s understanding of why one conversation is better than another. We credit a judge only when every component matches the ground truth,

c = 1 y ˆ = y⋆ ∧ rˆ = r⋆ ∧ tˆ= t⋆ ∈ {0,1}. (1)

A judge that picks the right conversation while localizing the flaw in the wrong turn, or assigning it to the wrong taxonomy entry, has reached the correct conclusion without identifying the underlying failure, and is not credited.

#### 3.1 Benchmark Construction

Semi-Automated Assistant Failure Type Discovery. We construct our taxonomy of assistant behavior types through a semi-automated discovery procedure. We first seed an initial set of behavior categories that are commonly observed in multi-turn conversations, drawing from and organizing prior works [3, 4, 43] on dialogue evaluation and assistant failure modes. To assess coverage, we then prompt Gemini-3.1-Pro [44] with samples from MT-Bench [1] and MT-Bench 101 [38], asking the model to verify whether each instance is captured by the existing taxonomy and, if not, to propose new assistant error types grounded in the observed failure.

We scope the taxonomy to failures characteristic of multi-turn assistant behavior; coarse singleturn categories such as factual error are excluded as standalone types [45], since their multi-turn manifestations are already absorbed by more specific types. For instance, an assistant that asserts a fact in turn 2 and contradicts it in turn 5 is captured by self_contradiction rather than a generic hallucination, and an assistant that drops a user-specified constraint after several turns is captured by instruction_forgetting. Additionally, each type targets failures that are plausible for a capable assistant yet difficult to spot by surface inspection, a requirement we make explicit in every flaw description so that the resulting probes stress, rather than merely confirm, the discriminative ability of strong LLM judges. Furthermore, categories are designed to have non-overlapping decision boundaries so that judges can unambiguously classify the failure type. Candidate failure types that do have overlap are merged into an existing type or dropped.

Failure type discovery was supervised by the authors, who reviewed each candidate type before admission and adjudicated borderline cases and overlaps. A condensed view of the resulting taxonomy is presented in Table 2, with the full set of types and definitions deferred to Appendix B.4, and the exact prompt used to elicit new error types from Gemini-3.1-Pro provided in Appendix B.3.

Dual Conditioned Conversation Pair Generation. Each conversation pair is sampled under two independent conditions: an assistant failure type from the seven options in Table 2, as well as a user behavior type. The assistant failure axis fixes the ground truth: by sampling a target type and instructing the generator to inject one error of that type, we unambiguously define the worse conversation of the pair, the turn in which the failure occurs, and the failure category. The user

behavior axis adds diversity, since different user types surface different slices of the reference material and create different turn-to-turn dynamics as seen in real multi-turn use. Behaviours span seven archetypes: focused, integrative, scattered, skeptical, misinformed, exploratory, and underspecified; each represents a style prompt which the user is conditioned on during generation, as defined in Appendix B.5.

Each pair of better and worse conversations is produced by two separate sets of generation calls that share reference documents. Past works have created negative examples by simply injecting errors into existing text [46, 30, 39, 47]. However, when comparing two alternate conversations, if the injected error is the only change, the judge can isolate this difference rather than making a complete assessment of quality. Independently sampling two conversations is also insufficient; if only the bad version faced situations where the target failure could surface, judges could again shortcut a holistic comparison by pattern-matching on question types. Therefore, when generating the better conversation, we actively stage the conditions under which the selected flaw would be relevant, but condition on benign behaviour (Table 2), which describes the correct way to handle the conversational pressure. For instance when the failure type is fabricated_answer, the user in the better conversation still asks an out-of-scope question, but the assistant is instructed to explicitly state the limits of its knowledge. Paired conversations are thus structurally matched on topic and conversational dynamics, differing only in handling.

Both sets of generation calls follow a turn-by-turn blueprint. Each turn specifies what the user’s question should be about, and the chunk of the reference document the assistant will need to draw on. The blueprint for the worse conversation additionally commits to a bad_round_index and a sketch of how the selected flaw should manifest. To keep the comparison non-trivial, the blueprint imposes requirements that the flawed turn must maintain the same tone and length

- as other turns, while lexical announcements of the kind “stepping outside the scope for a moment” are disallowed. We remove ordering bias by randomizing which conversation in the pair (A/B) is assigned as worse. The full generation prompts are provided in Appendix B.6.

Automated Quality Control. A synthetic pair is only useful to the benchmark if the targeted weakness actually surfaces, appears only in the declared turn, and the rest of the content is free of clear failures. To make these judgments, we rely on the well-documented asymmetry that verification is substantially easier than generation [9, 48, 49]. We use a three-layer verification cascade run by an external verifier model over every candidate pair, adding the pair to the benchmark only if it passes all three layers. Each layer is strictly discriminative where the verifier is given the intended labels (y⋆,r⋆,t⋆) and the reference documents. Note that the verifier solves an easier subproblem than the judges we aim to evaluate with the benchmark, since the verifier is conditioned on the ground truth.

Algorithm 1 Data generation and verification

- 1: for each of Ncand candidate samples do
- 2: Sample failure type t⋆ ∈ T , user behavior, reference documents, verdict y⋆ ∈ {A, B}, flawed turn r⋆ ∈ {1, . . . , R}
- 3: Generate paired blueprints conditioned on the sampled inputs
- 4: If coherence check of the blueprints fails, skip
- 5: Generate paired conversations (CA, CB) from the blueprints
- 6: If adherence check of (CA, CB) to blueprints fails, skip
- 7: If grounding check of all atomic claims fails, skip
- 8: Retain (CA, CB, y⋆, r⋆, t⋆)
- 9: end for

The three layers of verification check coherence, adherence, and grounding. The coherence check tests the sampled blueprint by comparing the per-turn outline of user intent, assistant focus, and the chosen failure turn against the intended ground truth and reference material. This check flags genuine semantic conflicts, e.g., a blueprint that is inconsistent with the chosen user behavior or failure location. In the adherence check, conversations are examined to ensure both the user and assistant follow the blueprint globally. The better conversation must display benign behaviour across all turns, and the worse conversation must exhibit the targeted flaw in exactly the declared turn. A conversation pair fails the adherence check if the user deviates from the specified behavior, the failure drifts to a different turn, or if multiple flaws are present. Finally, in the grounding check the verifier extracts every atomic factual claim from each assistant’s turn and labels each claim as grounded or not based on the reference context. A pair passes only if every turn (other than r⋆ in the worse conversation) is fully grounded, ensuring that the only unsupported claim is the targeted flaw.

Per-layer verification pass rates and overall retention across the three knowledge domains are reported in Table 3. The full verification prompts are provided in Appendix B.7.

#### 3.2 Joint Ranking of Judges and Conversation Pairs

Let D = {(jt,qt,ct)}Nt=1 be the judgment results over judges J = {1,...,M} across conversation pairs Q = {1,...,Q}, with ct ∈ {0,1} marking whether the judge correctly identified the joint criterion (y⋆,r⋆,t⋆). Rather than simply report accuracies over a fixed test set, we use D to rate judges for two reasons. First, arena-style ratings [2] are relative to a population and tolerate partial observability of the judgment results, which enables leaderboard construction without requiring full judge-pair coverage. Second, rating judges and conversation pairs jointly assigns each pair a calibrated difficulty rating relative to the set of judges. This lets us dynamically curate the benchmark by difficulty. Specifically, our published evaluation slice in Section 3.3 drops the top tail of most difficult pairs because our human audit of conversation quality found this segment to have the most label noise, see Figure 2.

Two rating algorithms are compatible with our bipartite framing: Bradley–Terry (BT) rating, used by LM Arena [2] for LLM-vs-LLM matchups, and Empirical Interaction Propagation (EIP) [8], which is a PageRank-style random walk on the correctness graph. However, EIP requires full observability of D, so we adopt BT as our default and use EIP only as a cross-check. Under BT, each judge and each pair is assigned a positive strength θi > 0, i ∈ J ∪ Q. The probability that judge j produces a correct verdict on pair q is Pr(c = 1 | j,q) = θj/(θj + θq), fit by maximum-likelihood estimation using the minorization–maximization iteration with mean-normalization [50], and mapped to Elo via the standard 400log10 transform. Under full observability of D, where every judge is scored on every pair, the BT ordering of judges agrees with their raw win count. BT adds two properties EIP lacks: closed-form cluster-robust confidence intervals, and stability under partial observability. Full equations, convergence criteria, and cluster-robust pair-level standard errors are deferred to Appendix B.1, with similar details for EIP in Appendix B.2.

A pair on which every judge produces the same verdict carries zero information about relative judge ability. Under the BT likelihood it contributes a constant regardless of the ratings being fit. We therefore drop any pair whose verdicts are unanimous across the current judge set. The unanimouslyincorrect pairs are additionally likely to be mislabelled and would in any case be removed by the top-Elo curation step introduced next.

#### 3.3 Benchmark Curation

Our benchmark is completely synthetic and requires no human annotation during construction, which lets us cover the taxonomy of Section 3.1, regenerate the pool when the verifier or generator is upgraded, and scale the benchmark without proportional annotation budget. The resulting data can still be noisy, but noise is not unique to synthetic benchmarks. Widely used benchmarks carry a non-trivial fraction of mislabeled or ambiguous items; MMLU [5] has roughly 6% erroneous questions in heavily evaluated subsets [7] and audits of the ten most-cited vision benchmarks find an average test-set error rate near 3% [6].

This puts curation in tension with discrimination. The hardest pairs are the most informative for separating out stronger judges, but they are the most likely to carry label noise or be ambiguous. We resolve this tradeoff conservatively: the published evaluation slice drops the top 5% of pairs by BT Elo (column Top 5% of Table 3; per-axis breakdowns in Tables 5 and 6). BT pair difficulty is judge-set-aware, so the cut is recomputed whenever the judge set changes. Two analyses support this cut: a human audit and a downstream SFT experiment, both of which indicate that the top-Elo slice is disproportionately likely to contain label noise, see Figure 2.

Human Verification. The audit described here sits outside the benchmark construction loop. The benchmark is built end-to-end without manual curation, while this audit confirms the quality of the automated verification and justifies the design choice of dropping the top-Elo slice. The authors manually labeled a uniform random sample of 120 of the 703 informative pairs as clean, ambiguous, or noise. Noise indicated that the conversation pair did not strictly match the intended ground truth (y⋆,r⋆,t⋆), for example because more than one flaw was present. Ambiguous indicated that the assistant failure could not unambiguously be assigned a single type t⋆ in the taxonomy, but all other details were consistent. Unambiguous and correctly labeled conversation pairs were considered clean. The overall noise rate is low. Figure 2 (left) plots the cumulative fraction of audit labels across pairs with BT-Elo at most x, where we find the noise and ambiguous bands rise only in the right tail. Given the labeling effort already conducted, we do drop the 17 problematic pairs from the published slice (column Human of Table 3), but the curation pipeline does not depend on this step. The full annotation procedure, and the pair visualization tool we used are in Appendix C.

0.5

Cumulativefractionover[elo,x]mn

| |Cle|an<br><br>Ambiguous Noise|
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

Fully Solved

Correct Error Type

0.8

0.4

0.71

0.62

0.58

0.6

0.3

Score

0.44

0.4

0.35

0.2

0.29

0.20

0.2

0.1

0.09

0.0

0.0

600 800 1000 1200 1400 1600 1800

Qwen3.5-4B SFT top 20% SFT 20 40% SFT 40 100%

BT-elo

- Figure 2: (Left) Cumulative fraction of samples annotated by humans as having ambiguous or noisy labels among conversation pairs with BT-Elo at most x. Both problematic types concentrate at the top of the difficulty distribution while easier pairs are almost entirely unambiguous. (Right) Fine-tuning Qwen3.5-4B on the top-20% slice underperforms training on the cleaner slices.

- Table 3: Verification and filtering cascade, split by domain. The first three filters represent automated quality verification during pair generation. The last three are applied at scoring time: Uninformative drops uninformative pairs where every judge gave the same verdict; Human drops pairs that human annotators flagged as ambiguous or noisy. The pipeline itself is fully automated and does not depend on human annotation; we apply this layer opportunistically to the 120-pair subset that was annotated for other analyses, since the flagged pairs can be excluded at no additional cost once those labels exist. Top 5% drops the top 5% of pairs by Elo, which are more likely to contain noisy labels.

Domain Generated Coherence Adherence Grounding Uninformative Human Top 5% Survival

ML 400 397 (−3) 342 (−55) 304 (−38) 263 (−41) 255 (−8) 240 (−15) 60.0% MED 400 400 (−0) 344 (−56) 243 (−101) 208 (−35) 202 (−6) 194 (−8) 48.5% FIN 400 397 (−3) 345 (−52) 274 (−71) 232 (−42) 229 (−3) 218 (−11) 54.5%

All 1200 1194 (−6) 1031 (−163) 821 (−210) 703 (−118) 686 (−17) 652 (−34) 54.3%

SFT with Label Noise. A second analysis tests whether top-Elo pairs degrade model fine-tuning results. We split the informative pairs not used in the human audit into three BT Elo bands (top 20%, 20–40%, and the remaining 60%), sample an equal number of pairs from each, and fine-tune Qwen3.5-4B [51] with LoRA [52] under an identical recipe per band (full hyperparameters in Appendix B.9). The clean pairs from the audit are held out for evaluation. Joint correctness of the fine-tuned judge improves monotonically as the training band moves down the ranking, and the bottom band achieves more than twice the score of the top band (Figure 2, right). This signal is consistent with the human audit: the top-ranked band of conversation pairs is flagged as the noisiest, leading to our decision to cut the top of the Elo distribution from the published evaluation slice.

- 4 Experiments

Experimental Settings. Our pipeline is domain-agnostic, generating a benchmark of multi-turn conversation pairs through the same generation, verification, and ranking components from any corpus of knowledge-intensive context. We instantiate it on three domains, each viewable as a standalone benchmark: machine learning (CS papers from RPC-Bench [53], open-access no license), biomedicine (PubMedQA [54], MIT license), and finance (S&P 500 10-K filings [55], MIT license). Per pair, the user behavior and assistant failure types are sampled independently and uniformly. All generator, verifier, and judge calls are routed through OpenRouter [56] with gpt-5.5 used for generation and verification. Every judge in our registry supports thinking, which we leave at the provider’s default. We exclude post-trained judges to keep the focus on inherent abilities of general-purpose LLMs, though post-training can improve performance on static problems [? 57].

RankJudge Leaderboard. Figure 3 reports BT-Elo ratings for 21 judges. The scale spans nearly 1200 Elo points from gemini-3.1-pro (1959) to gpt-oss-20b (762), implying that the benchmark separates judges across a wide range of capability. gemini-3.1-pro leads with a 92-point margin over the next judge gpt-5.5, while the per-domain rankings broadly preserve the order, indicating that judge skill is largely domain-invariant. The leaderboard shows a trend that proprietary LLMs are still stronger than open-source models, although kimi-k2.6 and gemma-4-31b outrank several

Domain

2000

All ML

Med Fin Proprietary Open source

| |
|---|

1800

1600

1400

Elo

1200

1000

| |
|---|
| |

800

gemini-3.1-pro(1959±80)gpt-5.5(1867±65)gemini-3-flash(1724±49)kimi-k2.6(1713±54)sonnet-4.6(1706±49)gemma-4-31b(1695±47)qwen3.6-plus(1671±51)glm-5.1(1637±47)gpt-5.4(1631±54)qwen3.5-397b(1571±43)opus-4.7(1510±43)haiku-4.5(1505±47)gemma-4-26b(1444±42)qwen3.5-122b(1422±37)qwen3.5-35b(1373±38)gpt-5.4-mini(1342±37)deepseek-v3.2(1307±43)minimax-m2.7(1184±40)mistral-small-4(906±42)gpt-oss-120b(867±42)gpt-oss-20b(762±45)

- Figure 3: Elo scores of 21 judges on the combined dataset. Black circles give the combined Elo with 95% CI; colored markers show per-domain Elo scores. Tick-label color denotes proprietary (blue) vs. open-source (orange) judges. Table 4 in Appendix A.1 gives detailed per-domain results.

frontier models, while opus-4.7 underperforms, indicating that strong generation capability does not directly translate to strong judging.

We note that gpt-5.5 is subject to a strong self-preferential bias in pairwise judging, since this model was used for generation and verification. Appendix A.3 characterizes the effect by regenerating the conversation pairs with opus-4.7 as the generator; in this case gpt-5.5 drops from rank 2 to rank 7 and, opus-4.7 climbs from rank 11 to rank 3, but the rankings of the remaining 19 judges are preserved almost identically (Spearman ρ = 0.979, Kendall τ = 0.906). gemini-3.1-pro in particular ranks first under either generator despite, reflecting its genuinely superior judging ability.

Elo vs. Compute. Compute cost is the other consideration for determining the best judge. Figure 4 relates each judge’s Elo to its mean per-match compute. The token-axis frontier collapses to three judges, opus-4.7 (549 tokens, 1510 Elo), gpt-5.5 (934 tokens, 1867), and gemini-3.1-pro (1723 tokens, 1959); a long tail of mid-Elo judges spends 7–10k tokens per match while sitting well off the frontier, so verbosity alone does not buy ranking accuracy. The cost-axis frontier tells a different story, with open-weights judges holding most of the curve: gemma-4-31b reaches 1695 Elo

- at $0.0023 per match, less than a tenth the price of gemini-3.1-pro for 264 fewer Elo points, and gpt-oss-120b extends the frontier into the sub-cent regime at the bottom of the leaderboard. Only gemini-3.1-pro lies on both frontiers; gpt-5.5 is dominated on cost, so budget-bound users can reach competitive Elo at a fraction of the spend via gemini-3-flash or an open-weights judge such as gemma-4-31b.

Class Bias is a Capability Gap. Figure 5 reports the per-judge gap between predicted and groundtruth failure type share. As judge strength declines, predictions collapse onto evasion with overprediction rising from near zero for the strongest judges to +41.6pp at the bottom of the leaderboard, leading to under-prediction of no_clarification and unnecessary_refusal, which fall to roughly −15pp and −18pp for the weakest judges. No such bias is observed at the top, where mistakes by gemini-3.1-pro remain within 2pp of zero on every class, which establishes that the seven-way failure taxonomy is internally non-overlapping. The class bias is therefore a property of the weaker judges rather than of the data. With poor reasoning ability, judges default to evasion

- as a crutch. The same collapse is reproduced in each domain (Figure 9 in Appendix A.5), and the per-judge confusion matrix (Figure 10 in Appendix A.6) corroborates this at the prediction level.

Effect of Failure-type Prediction. We drop the failure-type component from the correctness criterion leaving (y⋆,r⋆), and refit BT-Elo. Panel (a) of Figure 6 compares judge Elo with and without the type label t⋆, showing high correlation at Spearman ρ = 0.968 and Kendall τ = 0.895. This shows that including failure type prediction as a correctness criterion widens the Elo gap between judges without reordering the leaderboard. Hence, the failure taxonomy further distinguishes judge ability by testing their ability to understand the reasons for failure. The same pattern holds within each

- domain (Figure 11 in Appendix A.7).

Error Analysis and Mitigation. The leaderboard exposes a surprising anomaly: opus-4.7, the headline model of the Claude family at the time of writing, sits below sonnet-4.6. This motivates a deeper investigation, including whether prompt-level intervention can recover the gap. We repurpose the human audit of Section 3.3 for error-analysis over opus-4.7’s incorrect verdicts on the clean pairs, summarize its recurring failure modes, and design four prompt revisions targeting them:

2000

gemini-3.1-pro gpt-5.5

1800

sonnet-4.6

gemini-3-flash kimi-k2.6

gemma-4-31b

gpt-5.4

qwen3.6-plus

1600

glm-5.1

qwen3.5-397b-a17b opus-4.7 haiku-4.5

JudgeElo

gemma-4-26b-a4b

1400

gpt-5.4-mini

deepseek-v3.2

qwen3.5-122b-a10b qwen3.5-35b-a3b

minimax-m2.7

1200

1000

mistral-small-4-119b gpt-oss-120b

800

gpt-oss-20b

2000 4000 6000 8000 10000 Mean completion tokens per match

2000

gemini-3.1-pro

gpt-5.5

1800

qwen3.6-plus

gemini-3-flash kimi-k2.6

gemma-4-31b

glm-5.1

1600

sonnet-4.6

gpt-5.4

haiku-4.5

qwen3.5-35b-a3b gpt-5.4-mini

JudgeElo

opus-4.7

gemma-4-26b-a4b

qwen3.5-397b-a17b

1400

deepseek-v3.2

qwen3.5-122b-a10b

1200

minimax-m2.7

1000

mistral-small-4-119b gpt-oss-120b

800

gpt-oss-20b

10 3 10 2 Mean cost per match (USD)

OpenAI Anthropic Google Qwen DeepSeek Other Pareto frontier

- Figure 4: Judge Elo against per-match compute. Top: mean completion tokens per match (linear). Bottom: mean cost per match in USD (log). Markers are coloured by provider and the dashed grey step line traces the empirical Pareto frontier.

[Figure 2]

gemini-3.1-progemini-3-flashgpt-5.5 kimi-k2.6sonnet-4.6gemma-4-31bqwen3.6-plusglm-5.1qwen3.5-397bgpt-5.4 opus-4.7haiku-4.5gemma-4-26bqwen3.5-122bqwen3.5-35bgpt-5.4-minideepseek-v3.2minimax-m2.7mistral-small-4gpt-oss-120bgpt-oss-20b

self_contradiction

evasion

disorganized

fabricated_answer

instruction_forgetting

no_clarification

unnecessary_refusal

-1.5 -0.6 +1.4 +0.6 +0.8 -0.3 -0.8 -3.1 -3.8 +0.2 -6.3 -3.1 -6.9 +1.5 +0.0 -8.6 -6.9 +0.3 -9.4 -3.4 -3.5

+0.8 +4.1 +2.9 +6.0 +4.4 +2.9 +6.7 +8.0 +7.7 +5.5 +16.4 +9.7 +12.3 +5.8 +1.5 +22.2 +18.9 +12.4 +41.6 +41.0 +29.8

- +0.2 -1.5 -1.5 -0.8 -1.5 -0.8 -1.1 -1.2 -2.8 -0.2 +0.0 -0.8 +0.0 -0.8 -1.2 -3.2 -2.5 -1.1 +2.3 +0.5 -2.3
- +1.4 +0.6 -0.3 +0.0 -0.2 +2.9 +0.6 +2.3 +2.5 +3.4 +0.6 +0.6 +3.2 +5.2 +11.0 +0.3 -1.7 +3.5 -7.8 +0.3 +10.0

- -0.2 -0.3 +0.0 -0.2 -0.3 -0.2 -0.8 -0.5 -0.3 -0.3 -1.7 -1.4 -1.8 +0.2 -0.3 -1.7 -2.3 -2.0 -4.9 -5.1 -2.6
- -0.5 -2.6 -5.4 -5.7 -7.7 -5.7 -5.4 -5.8 -4.8 -9.7 -9.2 -6.6 -6.9 -12.6 -11.8 -9.2 -5.5 -12.1 -14.1 -15.0 -13.8
- -0.2 +0.3 +2.9 +0.0 +4.4 +1.1 +0.6 +0.3 +1.5 +1.1 +0.2 +1.5 +0.2 +0.6 +0.8 +0.2 +0.0 -1.1 -7.7 -18.3 -17.5

|[Figure 3]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

- -40
- -30
- -20
- -10

+0

+10

+20

+30

+40

Predicted-trueclassshare

- Figure 5: Per-class prediction bias for each judge. Each cell gives the difference in percentage points between the judge’s predicted share of the assistant failure type and the ground-truth share. Red indicates over-prediction, while blue shows under-prediction. Columns are sorted by Elo.

category hints (v2), explicit candidate comparison (v3), decision rules (v4), and chain-of-thought (v5), detailed in Appendix B.8. To bracket the magnitude of the gap we additionally ran opus-4.6 under the default prompt. Figure 6(b) plots accuracy against mean completion tokens per call for the four revisions and the four default-prompt baselines. opus-4.6 substantially outperforms every variant of opus-4.7 we tried, and none of the prompt revisions improves opus-4.7 to the accuracy level of opus-4.6 or sonnet-4.6, suggesting a capability ceiling rather than a prompting deficiency. The shift aligns with documented changes in the opus-4.7 release [58]: thinking is now an adaptive effort that the model can downscale at will, in place of 4.6’s fixed budget. As a result opus-4.7 commits to verdicts with too little deliberation when it could benefit from using more test-time compute. The Claude family case study suggests a more general conclusion that prompt-level intervention cannot

(a) (b)

###### =0.968 =0.895

Opus 4.6 Sonnet 4.6

90

Accuracy(%)

Opus 4.7 v2 (hints)

y = x

85

2000

gemini-3.1-pro

Opus 4.7 v5 (CoT) Opus 4.7 v3 (delib.) Opus 4.7 Haiku 4.5

80

gemini-3-flash

Elow/ofailuretypecorrectness

qwen3.6-plus

1800

75

Opus 4.7 v4 (rules)

gpt-5.5

gemma-4-31b

kimi-k2.6

gpt-5.4

500 1000 2000 3000 5000

1600

glm-5.1 sonnet-4.6

Mean completion tokens per call

deepseek-v3.2

haiku-4.5

###### default prompt prompt variants Pareto frontier

opus-4.7 qwen3.5-397b-a17b

1400

gpt-5.4-mini

(c)

gemma-4-26b-a4b qwen3.5-122b-a10b

0.996 0.999

1.00

Correlationwithfullsample

1200

minimax-m2.7

qwen3.5-35b-a3b

0.973

0.990

0.951

0.981

0.95

gpt-oss-120b

1000

mistral-small-4-119b gpt-oss-20b

0.90

0.893

800

0.85

Spearman

Kendall

0.838

0.80

600 800 1000 1200 1400 1600 1800 2000 2200

10% 20% 50% 80%

Elo w/ failure type correctness

Subsample fraction

- Figure 6: (a) Judge Elo with and without failure-type correctness as a criterion. (b) Accuracy vs. mean completion tokens for four default-prompt baselines and four opus-4.7 prompt revisions; the dashed step is the empirical Pareto frontier. (c) Rank correlation between the full-sample judge ranking and rankings recomputed on uniform subsamples.

close a model-capability gap in pairwise judging. Our four revisions cover the usual toolkit (more guidance, more deliberation, more structure, more thinking), and their joint failure suggests that practitioners are better served by switching judges than by tuning prompts.

Robustness and Benchmark Cost. To verify the partial-observability argument of Section 3.2, we refit BT-Elo on uniform subsamples of the judgment results at fractions {10%,20%,50%,80%} and report the rank correlation with the full data in Figure 6(c). Spearman ρ exceeds 0.95 from 10% onward and Kendall τ crosses the same threshold at 50%, so the leaderboard can be reproduced at a fraction of the full cost with negligible distortion to rankings. The same conclusion holds within each

- domain (Figure 12 in Appendix A.8). The leaderboard is also stable to two further perturbations. First, refitting with the top-5% Elo conversation pairs included matches the published ordering at Spearman ρ ≥ 0.997, with the top-10 judge ranking preserved (Figure 13 in Appendix A.9), confirming that the BT rating is robust under residual noise. Second, swapping BT-Elo for the difficulty-aware EIP ranking leaves the ordering nearly unchanged (ρ ≥ 0.94, Appendix A.10). We adopt BT as the primary score because it yields closed-form cluster-robust error bars and a leaderboard that stays stable under partial observability.

Pairwise versus Pointwise. RANKJUDGE can also be run with the judges scoring conversations pointwise on a 1–10 Likert scale rather than choosing between candidates. The two protocols agree

- at a coarse level (Spearman ρ = 0.809) but the judge rankings diverge because these approaches measure different capabilities. Pairwise tests discrimination between two candidate conversations, while pointwise tests absolute calibration of conversation quality without a reference. We report this comparison in detail in Appendix A.11.

### 5 Conclusion

We presented RANKJUDGE, a benchmark generator for evaluating LLM-as-a-judge on multi-turn, reference-grounded conversations. By constructing each pair to have an unambiguously worse conversation while keeping the subject material similarly challenging, but not identical, the pipeline supports a strict joint correctness criterion that separates correct judgments from correct guesses. Three-layer automated verification combined with dynamic Bradley–Terry Elo-based curation keeps construction fully synthetic without sacrificing label quality. The verification step and label quality was cross-checked by a human audit and a held-out fine-tuning experiment. Across 21 frontier judges and three subject matter domains, the resulting leaderboard spans nearly 1200 Elo points, surfaces several open-weight checkpoints that outrank proprietary models, and exposes a sharp failure type bias in weaker judges. The judge ranking is stable across different curation setups, swapping the Bradley–Terry rating system for the EIP system, and under subsampling which reduces the cost of benchmarking.

### References

- [1] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623, 2023.
- [2] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. In Proceedings of the 41st International Conference on Machine Learning, 2024.
- [3] Mert Cemri, Melissa Z Pan, Shuyi Yang, Lakshya A Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Why Do Multi-Agent LLM Systems Fail? In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.
- [4] Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. LLMs Get Lost In Multi-Turn Conversation. arXiv:2505.06120, 2025.
- [5] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations, 2021.
- [6] Curtis G Northcutt, Anish Athalye, and Jonas Mueller. Pervasive label errors in test sets destabilize machine learning benchmarks. In Advances in Neural Information Processing Systems, volume 34, 2021.
- [7] Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. Are we done with MMLU? In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5069–5096, 2025.
- [8] Xingjian Hu, Ziqian Zhang, Yue Huang, Kai Zhang, Ruoxi Chen, Yixin Liu, Qingsong Wen, Kaidi Xu, Xiangliang Zhang, Neil Zhenqiang Gong, and Lichao Sun. EIP: Weighted ranking of LLMs by quantifying question difficulty. In The Fourteenth International Conference on Learning Representations, 2026.
- [9] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv:2110.14168, 2021.
- [10] Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and Heng Ji. MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback. In The Twelfth International Conference on Learning Representations, 2024.
- [11] Kaustubh Deshpande, Ved Sirdeshmukh, Johannes Baptist Mols, Lifeng Jin, Ed-Yeremai Hernandez-Cardona, Dean Lee, Jeremy Kritz, Willow E Primack, Summer Yue, and Chen Xing. Multichallenge: A realistic multi-turn conversation evaluation benchmark challenging to frontier LLMs. In Findings of the Association for Computational Linguistics: ACL 2025, pages 18632–18702, 2025.
- [12] Wai-Chung Kwan, Xingshan Zeng, Yuxin Jiang, Yufei Wang, Liangyou Li, Lifeng Shang, Xin Jiang, Qun Liu, and Kam-Fai Wong. MT-Eval: A multi-turn capabilities evaluation benchmark for large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 20153–20177, 2024.
- [13] Dongyang Fan, Sebastien Delsad, Nicolas Flammarion, and Maksym Andriushchenko. HalluHard: A Hard Multi-Turn Hallucination Benchmark. arXiv:2602.01031, 2026.
- [14] Jacob Eisenstein, Fantine Huot, Adam Fisch, Jonathan Berant, and Mirella Lapata. MT-PingEval: Evaluating Multi-Turn Collaboration with Private Information Games. arXiv:2602.24188, 2026.

- [15] Yubo Li, Xiaobin Shen, Xinyu Yao, Xueying Ding, Yidi Miao, Ramayya Krishnan, and Rema Padman. Beyond single-turn: A survey on multi-turn interactions with large language models. arXiv:2504.04717, 2025.
- [16] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730– 27744, 2022.
- [17] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30, 2017.
- [18] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, volume 33, pages 3008–3021, 2020.
- [19] Zhenghao Xu, Qin Lu, Qingru Zhang, Liang Qiu, Ilgee Hong, Changlong Yu, Wenlin Yao, Yao Liu, Haoming Jiang, Lihong Li, Hyokun Yun, and Tuo Zhao. Ask a Strong LLM Judge when Your Reward Model is Uncertain. In Advances in Neural Information Processing Systems, volume 38, pages 74639–74664, 2025.
- [20] Jiaan Wang, Yunlong Liang, Fandong Meng, Zengkui Sun, Haoxiang Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. Is ChatGPT a Good NLG Evaluator? A Preliminary Study. In Proceedings of the 4th New Frontiers in Summarization Workshop, pages 1–11. Association for Computational Linguistics, 2023.
- [21] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. GEval: NLG Evaluation using GPT-4 with Better Human Alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522. Association for Computational Linguistics, 2023.
- [22] Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Ren Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 26874–26901, 2024.
- [23] Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. Prometheus: Inducing fine-grained evaluation capability in language models. In The Twelfth International Conference on Learning Representations, 2024.
- [24] Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason E Weston, and Tianlu Wang. Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 52565–52583, 2025.
- [25] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jian-Guang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, Yansong Tang, and Dongmei Zhang. WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct. In The Thirteenth International Conference on Learning Representations, 2025.
- [26] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 6562–6595, 2024.
- [27] Yutong Wang, Pengliang Ji, Chaoqun Yang, Kaixin Li, Ming Hu, Jiaoyang Li, and Guillaume Sartoretti. MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness Evaluation. arXiv:2502.12468, 2025.

- [28] Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason E Weston, Ilia Kulikov, and Swarnadeep Saha. J1: Incentivizing thinking in LLM-as-a-judge via reinforcement learning. In The Fourteenth International Conference on Learning Representations, 2026.
- [29] Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, and Heng Ji. RM-R1: Reward Modeling as Reasoning. In The Fourteenth International Conference on Learning Representations, 2026.
- [30] Zhiyuan Zeng, Jiatong Yu, Tianyu Gao, Yu Meng, Tanya Goyal, and Danqi Chen. Evaluating large language models at evaluating instruction following. In The Twelfth International Conference on Learning Representations, 2024.
- [31] Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Yuan Tang, Alejandro Cuadron, Chenguang Wang, Raluca Popa, and Ion Stoica. JudgeBench: A Benchmark for Evaluating LLMBased Judges. In The Thirteenth International Conference on Learning Representations, 2025.
- [32] Austin Xu, Srijan Bansal, Yifei Ming, Semih Yavuz, and Shafiq Joty. Does context matter? ContextualJudgeBench for evaluating LLM-based judges in contextual settings. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, July 2025.
- [33] Yicheng Wang, Jiayi Yuan, Yu-Neng Chuang, Zhuoer Wang, Yingchi Liu, Mark Cusick, Param Kulkarni, Zhengping Ji, Yasser Ibrahim, and Xia Hu. DHP benchmark: Are LLMs good NLG evaluators? In Findings of the Association for Computational Linguistics: NAACL 2025. Association for Computational Linguistics, April 2025.
- [34] Yixin Liu, Kejian Shi, Alexander Fabbri, Yilun Zhao, PeiFeng Wang, Chien-Sheng Wu, Shafiq Joty, and Arman Cohan. ReIFE: Re-evaluating instruction-following evaluation. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, April 2025.
- [35] Ariel Gera, Odellia Boni, Yotam Perlitz, Roy Bar-Haim, Lilach Eden, and Asaf Yehudai. JuStRank: Benchmarking LLM judges for system ranking. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, July 2025.
- [36] Kyle Dai, Maurice Burger, Roman Engeler, Max Bartolo, Clémentine Fourrier, Toby Drane, Mathias Leys, and Jake Golden. Judge Arena: Benchmarking LLMs as Evaluators, 2024. Accessed: 2026-05-01.
- [37] John Mendonça, Alon Lavie, and Isabel Trancoso. MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Dialogue Evaluators. In Findings of the Association for Computational Linguistics: EACL 2026, 2026.
- [38] Ge Bai, Jie Liu, Xingyuan Bu, Yancheng He, Jiaheng Liu, Zhanhui Zhou, Zhuoran Lin, Wenbo Su, Tiezheng Ge, Bo Zheng, and Wanli Ouyang. MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2024.
- [39] Ruiqi Wang, Xinchen Wang, Cuiyun Gao, Chun Yong Chong, Xin Xia, and Qing Liao. AXIOM: Benchmarking LLM-as-a-Judge for Code via Rule-Based Perturbation and Multisource Quality Calibration. arXiv:2512.20159, 2025.
- [40] Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, and Shouling Ji. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge. arXiv:2506.09443, 2025.
- [41] Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. Judging the judges: A systematic study of position bias in LLM-as-a-judge. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of

- the Asia-Pacific Chapter of the Association for Computational Linguistics. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics, December 2025.
- [42] Xinghua Zhang, Bowen Yu, Haiyang Yu, Yangyu Lv, Tingwen Liu, Fei Huang, Hongbo Xu, and Yongbin Li. Wider and Deeper LLM Networks are Fairer LLM Evaluators. arXiv:2308.01862, 2023.
- [43] Ivan Kartáˇc, Mateusz Lango, and Ondˇrej Dušek. Reasoning Gets Harder for LLMs Inside A Dialogue. arXiv:2603.20133, 2026.
- [44] Google. Gemini 3.1 pro. https://gemini.google.com/, May 2026. Large language model.
- [45] Kin Kwan Leung, Mouloud Belbahri, Yi Sui, Alex Labach, Xueying Zhang, Stephen Anthony Rose, and Jesse C. Cresswell. Classifying and Addressing the Diversity of Errors in RetrievalAugmented Generation Systems. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3185–3207. Association for Computational Linguistics, 2026.
- [46] Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. HaluEval: A LargeScale Hallucination Evaluation Benchmark for Large Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6449–6464. Association for Computational Linguistics, 2023.
- [47] Fanqi Kong, Ruijie Zhang, Huaxiao Yin, Guibin Zhang, Xiaofei Zhang, Ziang Chen, Zhaowei Zhang, Xiaoyuan Zhang, Song-Chun Zhu, and Xue Feng. Aegis: Automated error generation and attribution for multi-agent systems. In The Fourteenth International Conference on Learning Representations, 2026.
- [48] William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike. Self-critiquing models for assisting human evaluators. arXiv:2206.05802, 2022.
- [49] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s Verify Step by Step. In The Twelfth International Conference on Learning Representations, 2024.
- [50] David R Hunter and Runze Li. Variable Selection using MM Algorithms. Annals of Statistics, 33(4):1617, 2005.
- [51] Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026.
- [52] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations, 2022.
- [53] Yelin Chen, Fanjin Zhang, Suping Sun, Yunhe Pang, Yuanchun Wang, Jian Song, Xiaoyan Li, Lei Hou, Shu Zhao, Jie Tang, and Juanzi Li. RPC-Bench: A Fine-grained Benchmark for Research Paper Comprehension. arXiv:2601.14289, 2026.
- [54] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. PubMedQA: A Dataset for Biomedical Research Question Answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2567–2577. Association for Computational Linguistics, 2019.
- [55] Jerry Loh. SP500-EDGAR-10K, 2026. Accessed: 2026-05-06.
- [56] OpenRouter. OpenRouter, 2026. Accessed: 2026-05-06.
- [57] Hui Huang, Yancheng He, Hongli Zhou, Rui Zhang, Wei Liu, Weixun Wang, Jiaheng Liu, and Wenbo Su. Think-J: Learning to Think for Generative LLM-as-a-Judge. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 31158–31166, 2026.

- [58] Anthropic. Claude Opus 4.7 System Card. https://www.anthropic.com/system-cards,

2026. Accessed: 2026-05-06.

- [59] Christopher Mohri and Tatsunori Hashimoto. Language models with conformal factuality guarantees. In Proceedings of the 41st International Conference on Machine Learning, 2024.
- [60] Bruce Kuwahara, Chen-Yuan Lin, Xiao Shi Huang, Kin Kwan Leung, Jullian Yapeter, Ilya Stanevich, Felipe Perez, and Jesse C. Cresswell. Document summarization with conformal importance guarantees. In Advances in Neural Information Processing Systems, volume 38, pages 67107–67152, 2025.
- [61] Brendan Leigh Ross, Noël Vouitsis, Atiyeh Ashari Ghomi, Rasa Hosseinzadeh, Ji Xin, Zhaoyan Liu, Yi Sui, Shiyi Hou, Kin Kwan Leung, Gabriel Loaiza-Ganem, and Jesse C. Cresswell. Textual bayes: Quantifying prompt uncertainty in LLM-based systems. In The Fourteenth International Conference on Learning Representations, 2026.

### A Additional Results

#### A.1 Detailed Judge Elo Ratings

Table 4 gives each judge’s combined Elo together with the per-domain breakdown, where confidence intervals are obtained from the cluster-robust sandwich estimator of Appendix B.1. The intervals tighten toward the middle of the leaderboard, where contested matches concentrate, and widen at the extremes, where the strongest and weakest judges win or lose decisively against most opponents and so accumulate little discriminative information. Finance is the lowest-Elo domain for most mid-leaderboard judges, consistent with the heavier numerical reasoning load imposed by 10-K filings; gpt-5.4, the only judge whose Finance Elo exceeds its ML and Biomedical scores, is a notable exception.

Table 4: Elo of LLM judges (95% CI). Best per column in bold. Sorted by Elo measured by the combined dataset (All).

Judge All ML Med Fin Type gemini-3.1-pro 1959±80 1992±156 1910±131 1933±129 Prop. gpt-5.5 1867±65 1854±109 1852±119 1850±107 Prop. gemini-3-flash 1724±49 1665±81 1759±93 1713±83 Prop. kimi-k2.6 1713±54 1743±99 1732±93 1634±89 Open sonnet-4.6 1706±49 1755±88 1707±90 1624±80 Prop. gemma-4-31b 1695±47 1708±81 1671±87 1662±76 Open qwen3.6-plus 1671±51 1665±88 1719±103 1597±79 Prop. glm-5.1 1637±47 1665±87 1671±89 1546±73 Open gpt-5.4 1631±54 1581±90 1617±83 1653±99 Prop. qwen3.5-397b-a17b 1571±43 1590±74 1597±80 1490±72 Open opus-4.7 1510±43 1523±75 1587±82 1395±71 Prop. haiku-4.5 1505±47 1492±80 1487±78 1490±82 Prop. gemma-4-26b-a4b 1444±42 1428±70 1487±75 1380±74 Open qwen3.5-122b-a10b 1422±37 1381±60 1454±68 1395±62 Open qwen3.5-35b-a3b 1373±38 1349±65 1471±73 1272±63 Open gpt-5.4-mini 1342±37 1343±62 1363±73 1279±61 Prop. deepseek-v3.2 1307±43 1260±69 1370±85 1259±72 Open minimax-m2.7 1184±40 1119±68 1313±70 1094±70 Open mistral-small-4-119b 906±42 887±68 873±80 911±76 Open gpt-oss-120b 867±42 842±67 911±72 805±80 Open gpt-oss-20b 762±45 734±73 812±81 697±81 Open

#### A.2 Problem Rankings

Assistant failure type separates pair difficulty cleanly. no_clarification (median Elo 1525) and disorganized (1475) consistently beat judges, while unnecessary_refusal and fabricated_answer (both 849) are the easiest to catch: refusals stand out in context and fabrications are directly checkable against the grounding document. User behavior, by contrast, barely shifts pair difficulty: the seven categories span only 979 to 1247 in median Elo with heavily overlapping IQRs. This is by design. The user behavior axis is included to diversify the conversational contexts in the benchmark rather than to differentiate judges, so the lack of separation here is the expected outcome.

#### A.3 Model Bias and Self-Preference

Setup. To isolate the effect of the generator we re-ran the pair-generation pipeline a second time with opus-4.7 substituted for gpt-5.5 as the generator and the rest of the recipe held fixed: the same source documents, the dual-conditioning prompts of Appendix B.6, the coherence, adherence, and grounding gates of Appendix B.7, and the same 21-judge panel evaluated under the pairwise prompt of Appendix B.8. To bound API cost the parallel set was restricted to 50 items per domain,

(a) Assistant failure type

(b) User behavior type

no_clarification

focused

disorganized

exploratory

integrative

self_contradiction

evasion

misinformed

instruction_forgetting

skeptical

fabricated_answer

scattered

unnecessary_refusal

underspecified

800 1000 1200 1400 1600 1800 2000

800 1000 1200 1400 1600 1800 2000

Elo

Elo

- Figure 7: Per-pair Elo on the combined dataset, grouped by (a) assistant failure type and (b) user behavior type. Categories are sorted by median Elo; violins show the full distribution, overlaid boxes mark the interquartile range (IQR) with the median in red, and jittered points mark individual pairs.

GPT-5.5 generator (n=652)

Opus 4.7 generator (n=107)

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

- 18

- 19

- 20

- 21

Judgerank(1=best)

DeepSeek-v3.2 DeepSeek-v3.2

Gemini 3 Flash

Gemini 3 Flash

Gemini 3.1 Pro Gemini 3.1 Pro

Gemma 4-26B

Gemma 4-26B

Gemma 4-31B

GLM-5.1 Gemma 4-31B

GLM-5.1

- GPT-5.4 GPT-5.4

GPT-5.4-mini GPT-5.4-mini

- GPT-5.5

GPT-5.5

GPT-OSS-120B GPT-OSS-120B GPT-OSS-20B GPT-OSS-20B

Haiku 4.5 Haiku 4.5

Kimi K2.6

Kimi K2.6

MiniMax-m2.7 Mistral-Small-4-119B MiniMax-m2.7

Mistral-Small-4-119B

Opus 4.7

Opus 4.7

Qwen3.5-122B Qwen3.5-122B Qwen3.5-35B Qwen3.5-35B

- Qwen3.5-397B

- Qwen3.5-397B

Qwen3.6-plus

- Qwen3.6-plus

Sonnet 4.6 Sonnet 4.6

(a) All judges Spearman = 0.917 | Kendall = 0.810

GPT-5.5 generator (n=652)

Opus 4.7 generator (n=107)

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

- 18

- 19

Judgerank(1=best)

DeepSeek-v3.2 DeepSeek-v3.2

Gemini 3 Flash

Gemini 3 Flash

Gemini 3.1 Pro Gemini 3.1 Pro

Gemma 4-26B

Gemma 4-26B

Gemma 4-31B

Gemma 4-31B GLM-5.1

GLM-5.1

GPT-5.4

GPT-5.4

GPT-5.4-mini GPT-5.4-mini

GPT-OSS-120B GPT-OSS-120B GPT-OSS-20B GPT-OSS-20B

Haiku 4.5 Haiku 4.5

Kimi K2.6

Kimi K2.6

MiniMax-m2.7 Mistral-Small-4-119B MiniMax-m2.7

Mistral-Small-4-119B

Qwen3.5-122B Qwen3.5-122B Qwen3.5-35B Qwen3.5-35B

- Qwen3.5-397B

- Qwen3.5-397B

Qwen3.6-plus

- Qwen3.6-plus

Sonnet 4.6 Sonnet 4.6

(b) Generator models removed Spearman = 0.979 | Kendall = 0.906

- Figure 8: Same-model preference does not distort the rest of the leaderboard. Each panel is a slope chart of judge Elo ranks under the published pair set generated by gpt-5.5 (left axis) against the parallel set generated by opus-4.7 (right axis), with ranks recomputed from a Bradley–Terry fit within the panel’s judge subset.

drawn from the same source pool used for the published evaluation, yielding 150 candidate pairs of which 131 survive verification. Bradley–Terry is fit independently on each condition’s match set with the MM iteration of Appendix B.1: on the published top_removed slice (652 pairs, after dropping the ambiguous and noise pairs and the top-Elo trim), and on the filtered slice (107 non-trivial pairs). Since the parallel set has no human audit it cannot be cut to the clean subset. Spearman ρ and Kendall τ in each panel are computed on judge ranks computed within that panel’s subset, so they are invariant to the absolute Elo scale and to which judges are removed.

Findings. As shown in Figure 8, panel (a) keeps all 21 judges and exposes a clear self-preference: gpt-5.5 slips from rank 2 to rank 7 once it is no longer the generator, and opus-4.7 climbs from rank 11 to rank 3 once it is. The other 19 judges, whose outputs are never used to construct pairs, are nevertheless ordered almost identically across the two conditions. gemini-3.1-pro is in particular ranked first under both generators, even though no condition uses its outputs, suggesting that its lead is driven by judging ability rather than generator alignment. Panel (b) drops the two generators and recomputes the ranks on the same matches: Spearman ρ rises from 0.917 to 0.979 and Kendall τ from 0.810 to 0.906. The bias is therefore localized to whichever model is producing the conversations; one generator still has to be chosen for the published evaluation, but that choice does not materially shift any other judge’s ranking.

#### A.4 Filtering Cascade

Tables 3, 5, and 6 report the filtering cascade sliced three ways: by domain, by assistant failure type, and by user behavior type. Of 1200 generated pairs, 652 survive every layer (overall survival 54.3%). The dominant signal sits on the assistant-failure axis. unnecessary_refusal and fabricated_answer clear the adherence verifier with zero loss, while disorganized loses 130 of its 174 generated pairs at that single layer and ends up the rarest type in the published evaluation set (30 pairs, 17.2% survival). The per-domain and per-user-behavior survival rates are far tighter and have no comparable bottleneck (48.5–60.0% across domains, 46.4–61.8% across user behaviors), so the shape of the cascade is set almost entirely by which flaw is being injected. The next paragraph and worked example explain why disorganized is the hard case: it is the only assistant flaw whose detection condition is structural rather than categorical, so a turn must lack logical flow strongly enough for the verifier to confirm the flaw, but not so strongly that every judge picks the better conversation on sight.

Table 5: Verification and filtering cascade, split by assistant failure type.

Assistant Failure Type Generated Coherence Adherence Grounding Uninformative Human Top 5% Survival

self_contradiction 132 131 (−1) 117 (−14) 96 (−21) 94 (−2) 92 (−2) 88 (−4) 66.7% evasion 165 165 (−0) 152 (−13) 126 (−26) 113 (−13) 111 (−2) 108 (−3) 65.5% disorganized 174 174 (−0) 44 (−130) 34 (−10) 34 (−0) 33 (−1) 30 (−3) 17.2% fabricated_answer 180 180 (−0) 180 (−0) 156 (−24) 104 (−52) 104 (−0) 104 (−0) 57.8% instruction_forgetting 147 144 (−3) 143 (−1) 86 (−57) 80 (−6) 80 (−0) 79 (−1) 53.7% no_clarification 231 229 (−2) 224 (−5) 178 (−46) 137 (−41) 126 (−11) 103 (−23) 44.6% unnecessary_refusal 171 171 (−0) 171 (−0) 145 (−26) 141 (−4) 140 (−1) 140 (−0) 81.9%

All 1200 1194 (−6) 1031 (−163) 821 (−210) 703 (−118) 686 (−17) 652 (−34) 54.3%

Why disorganized struggles at adherence. Refusals, fabrications, contradictions, and instruction violations are binary properties of a turn’s content: a turn either declines an answerable question or it does not. disorganized is different. The flaw lives in the ordering and foregrounding of correct content, not in the content itself, so the bad turn must reshuffle the same facts that the good turn presents cleanly. Make the reshuffle too mild and the adherence verifier rejects the pair, with rejection reasons like “Round 2 is mostly clear and foregrounds the main correction, so it does not actually exhibit the required disorganized flaw”.2 Make the reshuffle too aggressive and the bad turn

2Verbatim from the adherence.bad_issue field of a rejected disorganized pair in verification.json; analogous strings (“clear and logically organized”, “gives a clear lead answer”) account for the bulk of the 130-pair attrition in row disorganized of Table 5.

Table 6: Verification and filtering cascade, split by user behavior type.

User Behavior Type Generated Coherence Adherence Grounding Uninformative Human Top 5% Survival

focused 150 148 (−2) 140 (−8) 115 (−25) 93 (−22) 90 (−3) 87 (−3) 58.0% integrative 222 222 (−0) 192 (−30) 139 (−53) 117 (−22) 110 (−7) 103 (−7) 46.4% scattered 144 142 (−2) 126 (−16) 103 (−23) 92 (−11) 90 (−2) 89 (−1) 61.8% skeptical 138 138 (−0) 123 (−15) 94 (−29) 80 (−14) 79 (−1) 77 (−2) 55.8% misinformed 174 173 (−1) 141 (−32) 132 (−9) 122 (−10) 119 (−3) 107 (−12) 61.5% exploratory 174 173 (−1) 151 (−22) 111 (−40) 89 (−22) 88 (−1) 83 (−5) 47.7% underspecified 198 198 (−0) 158 (−40) 127 (−31) 110 (−17) 110 (−0) 106 (−4) 53.5%

All 1200 1194 (−6) 1031 (−163) 821 (−210) 703 (−118) 686 (−17) 652 (−34) 54.3%

either drops or distorts a fact, which a strong judge then catches as a fabrication or evasion rather than as disorganization, and the pair is removed at the informativeness, human, or top-Elo layer.

The example below is from pair id=40589ed7f7f0 (ML, user_behavior_type=misinformed, assistant_behavior_type=disorganized, bad_round_index=2). The user’s second turn pushes a misinformed claim that the paper’s main novelty was DP-Adam beating DP-SGD, that no DP-SGD baseline was reported, and that prior work had privately fine-tuned GPT-2 with DP-SGD. Both assistant responses contain the same three corrections: the E2E DP-SGD BLEU number (63.17), that DP-Adam was not claimed as the contribution, and that the cited GPT-2 prior work fine-tuned GPT-2 non-privately. The only difference is structural: convo_a leads with the answer to the user’s question (“DP-Adam was not claimed as a new algorithm”), while convo_b opens with a peripheral hyperparameter detail and threads the lead correction through a privacy-by-post-processing aside, leaving it half a paragraph in.

#### Clean

DP-Adam was not claimed as a new algorithm; it was already common in DP libraries, and it uses the same gradient privatization mechanism as DPSGD before the Adam update. The paper’s contribution was showing that full fine-tuning of large pretrained language models can work well under DP when done with suitable hyperparameters, aligned objectives, and the memory trick. Also, the referenced prior work did not privately finetune GPT-2: it fine-tuned GPT-2 non-privately and used a privately trained feedforward network for private experiments.

#### Disorganized

On E2E at epsilon = 8, their extra DP-SGD experiment used learning rate 3 and clipping norm 0.1 and reached BLEU 63.17 over 3 seeds, which was on par with DP-Adam. DP-Adam itself was not claimed as the contribution, even though after the same subsampled Gaussian gradient privatization, Adam-style updates preserve privacy by post-processing. The contribution was applying known DP optimization effectively to full language-model fine-tuning, with the memory trick making it computationally practical. Also, the cited GPT-2 work did not privately fine-tune GPT-2 with DP-SGD; it fine-tuned GPT-2 nonprivately...

The two responses are factually equivalent against the source; only the ordering differs. This pair sits in the narrow band where the verifier accepts (the lead answer is genuinely buried, mid-paragraph, after a hyperparameter aside and a post-processing detour) and judges still split (the correct facts are all present, so a judge has to actually read for structure rather than skim for missing content). Most generation attempts miss this band in one direction or the other, which is why disorganized loses three quarters of its candidate pairs at the adherence layer alone.

#### A.5 Class Bias by Domain

- Figure 9 shows that the prediction-bias signature is essentially domain-invariant: across Machine Learning, Biomedical, and Finance, the same column pattern repeats, with evasion the dominant red column and no_clarification the dominant blue one, and the magnitude of the bias growing as Elo decreases from left to right. The other five categories carry only small, noisy biases. This mirrors the confusion structure of Appendix A.6: weak judges fold ambiguous failures into evasion and rarely commit to no_clarification, regardless of the underlying domain.

+40

|[Figure 4]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

[Figure 5]

+0.8 +0.0 +0.8 -0.4 +0.0 +0.4 -2.1 -0.8 +0.0 -2.5 -5.4 -2.5 -6.2 +1.7 -1.2 -8.8 -6.7 +2.5 -10.4 -2.9 -5.0

self_contradiction

+30

- -0.4 +4.6 +4.2 +6.7 +3.3 +6.2 +7.9 +5.0 +7.5 +8.3 +14.6 +7.9 +12.1 +7.5 +2.1 +21.2 +19.2 +11.7 +38.8 +42.1 +34.2

+0.0 -0.8 -0.8 -0.8 -0.8 -1.2 -0.8 +0.4 +0.0 -3.3 +0.4 -0.4 -0.8 -0.4 +0.0 -2.5 -0.8 -1.2 +3.8 +1.7 -3.3

+0.4 -0.4 +0.0 +0.0 +2.1 +0.0 +1.3 +0.8 +2.5 +2.1 +0.8 +2.5 +2.1 +3.8 +11.7 +0.0 -3.3 +1.7 -7.5 -2.1 +10.8

+0.0 -0.4 +0.0 +0.0 +0.0 +0.0 -0.8 -0.8 -0.4 +0.4 -1.7 -1.7 -2.1 -0.4 -1.7 -0.8 -2.5 -2.5 -5.8 -6.7 -4.6

- -0.8 -3.3 -7.9 -5.4 -5.8 -6.7 -5.4 -4.6 -10.0 -6.2 -8.8 -7.1 -5.8 -12.5 -11.2 -9.6 -5.8 -12.1 -12.5 -14.6 -13.3

Predicted-trueclassshare

evasion

+20

disorganized

+10

+0

fabricated_answer

- -40
- -30
- -20
- -10

instruction_forgetting

no_clarification

+0.0 +0.4 +3.7 +0.0 +1.2 +1.2 +0.0 +0.0 +0.4 +1.2 +0.0 +1.2 +0.8 +0.4 +0.4 +0.4 +0.0 +0.0 -6.2 -17.5 -18.8

unnecessary_refusal

gemini-3.1-progpt-5.5sonnet-4.6kimi-k2.6gemma-4-31bgemini-3-flashglm-5.1qwen3.6-plusqwen3.5-397bgpt-5.4opus-4.7haiku-4.5gemma-4-26bqwen3.5-122bqwen3.5-35bgpt-5.4-minideepseek-v3.2minimax-m2.7mistral-small-4gpt-oss-120bgpt-oss-20b

(a) Machine Learning

|[Figure 6]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

[Figure 7]

- -1.5 -1.5 +2.1 +3.6 +0.5 +2.6 -3.1 +0.5 -4.6 +0.5 -5.2 -3.1 -5.7 +5.2 +1.5 -3.6 -6.7 -1.5 -2.6 -9.8 +0.5

- +1.5 +2.6 +2.1 +3.1 +4.6 +3.1 +6.2 +2.1 +7.7 +5.7 +13.4 +13.9 +10.8 +2.1 +6.2 +14.4 +22.2 +13.9 +41.2 +49.5 +25.8

-0.5 -1.5 -1.5 -0.5 -0.5 -1.0 -0.5 -0.5 -2.6 -0.5 -0.5 -0.5 +1.5 -2.1 -0.5 -3.6 -3.1 -0.5 +1.0 +2.1 -2.6

- +2.1 +2.6 -1.0 +0.5 +1.5 +0.0 +4.1 +4.1 +4.1 +3.6 +1.5 -0.5 +4.1 +7.7 +5.2 -0.5 +1.5 +5.2 -0.5 -10.8 +10.3

- -0.5 -0.5 +0.0 -0.5 -1.0 -1.5 -0.5 -0.5 -1.5 -0.5 -1.5 -2.1 -3.1 -1.0 +0.0 -2.1 -4.1 -2.6 -5.2 -5.7 -1.0
- -0.5 -2.1 -4.1 -5.2 -5.2 -7.2 -6.2 -6.7 -4.6 -10.3 -7.7 -7.2 -7.2 -11.9 -12.4 -4.6 -9.3 -12.9 -14.4 -15.5 -14.4
- -0.5 +0.5 +2.6 -1.0 +0.0 +4.1 +0.0 +1.0 +1.5 +1.5 +0.0 -0.5 -0.5 +0.0 +0.0 +0.0 -0.5 -1.5 -19.6 -9.8 -18.6

self_contradiction

+40

Predicted-trueclassshare

evasion

+20

disorganized

+0

fabricated_answer

instruction_forgetting

- -40
- -20

no_clarification

unnecessary_refusal

gemini-3.1-progemini-3-flashgpt-5.5 kimi-k2.6qwen3.6-plussonnet-4.6glm-5.1gemma-4-31bqwen3.5-397bgpt-5.4 opus-4.7haiku-4.5gemma-4-26bqwen3.5-35bqwen3.5-122bdeepseek-v3.2gpt-5.4-miniminimax-m2.7gpt-oss-120bmistral-small-4gpt-oss-20b

(b) Biomedical

|[Figure 8]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

[Figure 9]

-4.1 -0.5 +1.8 -1.4 -4.6 -0.9 -0.9 -1.8 -4.1 -3.7 +0.0 -8.3 +1.4 -8.7 -10.1 -3.2 -10.1 -0.5 -7.8 -4.6 -5.5

self_contradiction

+30

+1.4 +5.0 +0.0 +3.2 +6.9 +7.8 +6.0 +10.6 +9.6 +7.8 +3.2 +21.1 +3.7 +13.8 +23.4 +0.5 +22.5 +11.9 +37.6 +39.4 +28.4

Predicted-trueclassshare

evasion

+20

- +0.9 -2.3 -1.8 -0.9 -2.3 -0.9 -2.8 -3.2 -2.3 -1.4 +0.0 +0.0 -1.4 -0.5 -4.1 -1.8 -3.2 -1.4 +0.9 -1.4 -0.9
- +1.8 +0.0 +0.0 +2.8 +1.4 -0.5 -0.5 -0.5 +1.8 -0.5 +4.1 -0.5 +6.9 +3.7 -0.5 +13.3 -0.9 +4.1 -5.5 +3.7 +8.7

disorganized

+10

+0

fabricated_answer

- -30
- -20
- -10

+0.0 +0.0 +0.0 +0.0 +0.0 +0.0 +0.5 -0.5 +0.0 -0.5 +0.0 -1.8 +0.9 -0.5 -0.5 +1.8 -2.3 -0.9 -3.2 -3.2 -1.8

instruction_forgetting

+0.0 -2.3 -5.0 -4.6 -3.2 -6.4 -7.8 -6.4 -6.0 -5.5 -8.7 -11.0 -12.8 -7.8 -8.7 -12.4 -6.0 -11.5 -14.7 -16.1 -13.8

no_clarification

+0.0 +0.0 +5.0 +0.9 +1.8 +0.9 +5.5 +1.8 +0.9 +3.7 +1.4 +0.5 +1.4 +0.0 +0.5 +1.8 +0.0 -1.8 -7.3 -17.9 -15.1

unnecessary_refusal

gemini-3.1-progemini-3-flashgpt-5.5gemma-4-31bgpt-5.4kimi-k2.6sonnet-4.6qwen3.6-plusglm-5.1haiku-4.5qwen3.5-397bopus-4.7qwen3.5-122bgemma-4-26bgpt-5.4-miniqwen3.5-35bdeepseek-v3.2minimax-m2.7mistral-small-4gpt-oss-120bgpt-oss-20b

(c) Finance

- Figure 9: Per-class prediction bias for each judge across the three domains. Each cell gives the difference between the judge’s predicted share of the assistant failure type and its ground-truth share, in percentage points; red indicates over-prediction and blue under-prediction. Columns are sorted by Elo.

#### A.6 Assistant Failure Type Confusion

gemini-3.1-pro (1959)

gpt-5.5 (1867)

gemini-3-flash (1724)

kimi-k2.6 (1713)

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

SC

.84 .06 .02 .08

.92 .03 .05

.92 .05 .01 .01 .01

.88 .11 .01

EV

- .03 .96 .01
- .03 .97

1.00

.04 .94 .01 .02

.02 .95 .03

DO

.03 .27 .63 .03 .03

.17 .17 .63 .03

.07 .17 .73 .03

groundtruth

top-4

FA

.01 .99

.02 .98

.01 .97 .02

.01 .97 .01 .01

IF

- .01 .99
- .02 .01 .97

.01 .97 .01

1.00

.01 .99

NC

.02 .14 .01 .83 .01

.06 .17 .65 .13

.09 .26 .02 .63

UR

.01 .99

1.00

1.00

.01 .01 .99

minimax-m2.7 (1184)

mistral-small-4-119b (906)

gpt-oss-120b (867)

gpt-oss-20b (762)

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

SC

.55 .23 .07 .16

.19 .72 .07 .02

.30 .60 .07 .02 .01

.24 .39 .06 .27 .02 .02

EV

.09 .73 .05 .07 .05 .01

- .03 .84 .09 .01 .02 .01

.10 .50 .33 .07

.58 .06 .32 .02 .03

- .04 .33 .06 .01 .56

.04 .76 .13 .07

.06 .59 .04 .22 .07 .01

DO

.20 .40 .33 .03 .03

.20 .50 .23 .07

.20 .40 .17 .23

groundtruth

bottom-4

FA

.07 .10 .81 .01 .02

.22 .77 .01

- .02 .23 .71 .03 .01

.10 .20 .11 .56 .01 .01

- .03 .77 .10 .05 .06

IF

.06 .09 .08 .76 .01

- .03 .37 .01 .03 .57
- .04 .85 .04 .02 .05

NC

- .05 .54 .02 .10 .22 .07
- .06 .04 .03 .87

.80 .07 .02 .01 .09 .02

UR

.01 .30 .01 .09 .60

.17 .61 .01 .07 .14

.13 .52 .01 .15 .01 .18

SC EV DO FA IF NC UR

SC EV DO FA IF NC UR

SC EV DO FA IF NC UR

SC EV DO FA IF NC UR

predicted

predicted

predicted

predicted

- Figure 10: Per-judge confusion of assistant failure type predictions, row-normalized so each row sums to one. Top row: the four highest-Elo judges; bottom row: the four lowest. Class labels abbreviate the seven assistant failure types (SC self_contradiction, EV evasion, DO disorganized, FA fabricated_answer, IF instruction_forgetting, NC no_clarification, UR unnecessary_refusal).

Figure 10 contrasts how the four highest- and lowest-Elo judges classify the assistant failure type. The top row concentrates nearly all of its mass on the diagonal across most categories; the only residual confusion is on disorganized and no_clarification, both of which leak modestly into evasion. The bottom row collapses toward a single dump bin: weak judges relabel a large share of no_clarification, self_contradiction, disorganized, and unnecessary_refusal predictions as evasion, suggesting that taxonomy resolution, not failure detection, is what separates the leaderboard tail from its head.

A.7 Failure-type Correctness by Domain

500 750 1000 1250 1500 1750 2000 2250

Elo w/ failure type correctness

800

1000

1200

1400

1600

1800

2000

Elow/ofailuretypecorrectness

y = x

gemini-3.1-pro

gpt-5.5

kimi-k2.6

gpt-5.4

gemma-4-31b

gemini-3-flash

qwen3.6-plus glm-5.1 sonnet-4.6

opus-4.7 haiku-4.5

qwen3.5-397b-a17b

qwen3.5-122b-a10b

gemma-4-26b-a4b

gpt-5.4-mini

deepseek-v3.2 qwen3.5-35b-a3b

minimax-m2.7

mistral-small-4-119b

gpt-oss-120b

gpt-oss-20b

(a) fin

=0.993 =0.959

600 800 1000 1200 1400 1600 1800 2000

Elo w/ failure type correctness

800

1000

1200

1400

1600

1800

2000

y = x

gemini-3.1-pro

gpt-5.5

qwen3.6-plus

kimi-k2.6

gemma-4-31b

gemini-3-flash

gpt-5.4

sonnet-4.6

glm-5.1

opus-4.7 qwen3.5-397b-a17b

gemma-4-26b-a4b qwen3.5-35b-a3b

haiku-4.5

qwen3.5-122b-a10b

deepseek-v3.2 gpt-5.4-mini

minimax-m2.7

gpt-oss-120b

gpt-oss-20b

mistral-small-4-119b

(b) med

=0.980 =0.913

600 800 1000 1200 1400 1600 1800 2000 2200

Elo w/ failure type correctness

800

1000

1200

1400

1600

1800

2000

y = x

gemini-3.1-pro

gpt-5.5

kimi-k2.6 glm-5.1 gemma-4-31b

sonnet-4.6

gpt-5.4

qwen3.6-plus

gemini-3-flash

opus-4.7 qwen3.5-397b-a17b

haiku-4.5

gpt-5.4-mini

gemma-4-26b-a4b

qwen3.5-35b-a3b

qwen3.5-122b-a10b

deepseek-v3.2

minimax-m2.7

gpt-oss-120b

gpt-oss-20b mistral-small-4-119b

(c) ml

=0.969 =0.885

Figure 11: Per-domain version of Figure 6(a). Judge Elo with the full correctness criterion (x-axis) vs. Elo with the failure-type label dropped from the loss (y-axis), refit independently within each domain. Each marker is a judge; the dashed line marks y = x and the red line is the least-squares fit. Per-panel Spearman ρ and Kendall τ are computed on the underlying Elo scores.

- Figure 11 repeats the with-vs.-without-type analysis inside each domain. Spearman ρ ranges from 0.969 (ML) to 0.993 (Finance) and Kendall τ from 0.885 to 0.959, with the red fits sitting below the

diagonal at the top of every panel. The seven-way taxonomy therefore widens the Elo range within every domain without reordering the local leaderboard, matching the combined-dataset result.

#### A.8 Subsample Stability by Domain

(a) fin

(b) med

(c) ml

1.0

0.979 0.979

0.995

0.996

1.00

1.00

0.988

0.983

0.974

Correlationwithfullsample

0.967

0.95

0.9

0.937

0.916

0.95

0.954

0.858

0.900

0.928

0.895

0.919

0.90

0.895

0.786

0.90

0.8

0.85

0.85

0.727

0.7

0.813

0.80

0.820

0.80

0.6

Spearman

0.603

0.775

0.791

0.75

Kendall

10% 20% 50% 80%

10% 20% 50% 80%

10% 20% 50% 80%

Subsample fraction

Subsample fraction

Subsample fraction

- Figure 12: Per-domain version of Figure 6(c). Each panel reports Spearman ρ and Kendall τ between the full-sample top_removed BT ranking and the partial_obs BT ranking refit on uniform subsamples at fractions {10%,20%,50%,80%}, computed independently within each domain.

Figure 12 reproduces the subsample-stability check within each domain. Spearman ρ exceeds 0.98 from the 50% subsample onward in all three domains and Kendall τ crosses 0.9 for Machine Learning and Finance at the same fraction. Biomedical is the slowest to stabilise (ρ = 0.786 at 10%), consistent with the tighter mid-leaderboard band for that domain reported in Appendix A.1, but joins the other two by the 50% fraction. The match-cost reduction therefore transfers to per-domain leaderboards as well, with a marginally larger minimum sample required for Biomedical.

#### A.9 Top-Ranked Pair Removal

The published evaluation pipeline applies a single noise-mitigation step on top of automatic verification: the top-ranked pair removal, which drops the highest-Elo pairs from the filtered pool. The human audit of Appendix C and the SFT-with-noise experiments of Appendix B.9 are not pipeline components but supporting evidence for this step, showing that the top-Elo cohort is where mislabelled ambiguous and noise pairs concentrate, so removing it strips the bulk of residual label noise. Figure 13 closes the loop: refitting Elo on the raw filtered slice yields a leaderboard nearly identical to the published one, with Spearman ρ ≥ 0.997, Kendall τ ≥ 0.981, and the top-10 set preserved in full on all four slices. We read this two ways. First, the automated quality control of Appendix B.7 already removes most of the label noise, so the additional top-Elo trim and the human audit only have small residual noise left to clean up. Second, the Bradley–Terry ranking and the surrounding evaluation pipeline are robust under that small residual noise.

#### A.10 Ranking Methods: BT vs. EIP

Figure 14 compares the Bradley–Terry Elo leaderboard with the difficulty-aware EIP ranking of Appendix B.2, on the combined dataset and each domain. The two methods agree strongly across all four slices: Spearman ρ ranges from 0.944 to 0.953 and Kendall τ from 0.820 to 0.852, with the top-3 set always sharing two of three judges and the top-10 sets overlapping by 80% to 90%. EIP is designed to up-weight judges that solve pairs which fool strong judges, so its near-identical ordering with Elo indicates that the benchmark does not contain a pocket of hard pairs whose signal Elo discards. We adopt Bradley–Terry Elo as the primary score throughout the paper for its closedform cluster-robust confidence intervals (Appendix B.1) and its stability under partial observability (Figure 12).

#### A.11 Pointwise Judges

Setup. The pointwise prompt of Appendix B.8 differs from the pairwise default in that each judge sees a single conversation in isolation and returns a 1-10 Likert score together with an is_flawed flag, the worst-round index, and the A-G flaw category. Pairwise verdicts are derived post-hoc by

(a) all

(b) fin

=0.997 =0.981

=0.998 =0.988

0

0

gemini-3.1-pro

gemini-3.1-pro

gpt-5.5

gpt-5.5

Judgerank(withoutnoisereduction)

kimi-k2.6

gemini-3-flash

gemini-3-flash

gemma-4-31b

sonnet-4.6

kimi-k2.6

5

5

gemma-4-31b

gpt-5.4

qwen3.6-plus

sonnet-4.6

gpt-5.4

qwen3.6-plus

glm-5.1

glm-5.1

qwen3.5-397b-a17b

10

10

haiku-4.5

qwen3.5-397b-a17b

opus-4.7

haiku-4.5

opus-4.7qwen3.5-122b-a10b

gemma-4-26b-a4b

qwen3.5-122b-a10b qwen3.5-35b-a3b

gemma-4-26b-a4b

gpt-5.4-mini

15

15

gpt-5.4-mini

deepseek-v3.2

qwen3.5-35b-a3b

deepseek-v3.2

minimax-m2.7 mistral-small-4-119b

minimax-m2.7 mistral-small-4-119b

gpt-oss-120b gpt-oss-20b

gpt-oss-120b gpt-oss-20b

20

20

20 15 10 5 0

20 15 10 5 0

(c) med

(d) ml

=1.000 =1.000

=0.997 =0.983

0

0

gemini-3.1-pro

gemini-3.1-pro

gpt-5.5

gpt-5.5

Judgerank(withoutnoisereduction)

gemini-3-flash

sonnet-4.6

kimi-k2.6

kimi-k2.6

qwen3.6-plus sonnet-4.6

gemma-4-31b

5

5

qwen3.6-plus

glm-5.1

gemma-4-31bglm-5.1 gpt-5.4

gemini-3-flash

gpt-5.4

qwen3.5-397b-a17b

qwen3.5-397b-a17b

10

10

opus-4.7

opus-4.7

haiku-4.5

gemma-4-26b-a4b

haiku-4.5

gemma-4-26b-a4b

qwen3.5-35b-a3b

qwen3.5-122b-a10b qwen3.5-35b-a3b

qwen3.5-122b-a10b

15

15

deepseek-v3.2

gpt-5.4-mini

gpt-5.4-mini

deepseek-v3.2

minimax-m2.7

minimax-m2.7 mistral-small-4-119b

gpt-oss-120b

mistral-small-4-119b

gpt-oss-120b gpt-oss-20b

20

20

gpt-oss-20b

20 15 10 5 0

20 15 10 5 0

Judge rank (with noise reduction)

Judge rank (with noise reduction)

- Figure 13: Judge ranks with (x-axis) and without (y-axis) the top-ranked pair removal, one panel per slice. The “with” condition is the published top_removed slice; the “without” condition is the raw filtered slice taken straight from the automatic verification pipeline of Appendix B.7. Each marker is a judge; the dashed line is y = x. Both axes are inverted so that rank 1 sits at the top-right. Per-panel Spearman ρ and Kendall τ are computed on the underlying Elo scores.

scoring convo_a and convo_b independently and taking the higher-scoring side as the winner; equal scores are recorded as ties and given 0.5/0.5 credit so the same Bradley–Terry machinery applies. The experiment is run on a 100-pair sample stratified by domain and assistant failure type, judged by the same 21-model panel; the pairwise axis in Figure 15(a) is recomputed on the identical sample, so the only difference between the two axes is the prompt mode.

Findings. Pointwise and pairwise rankings agree at the coarse level (Spearman ρ = 0.809, Kendall τ = 0.660) but the judge rankings differ. This divergence is expected, because the two protocols measure different judge capabilities: pairwise tests discrimination between two candidate conversations under direct comparison, while pointwise tests absolute calibration of conversation quality without a reference. Panel (b) makes the calibration axis explicit: the mean score gap between the good and flawed conversations exceeds 4 Likert points for gemini-3.1-pro but falls below 1 for the weakest three judges, a per-judge resolution that pairwise discards once scores collapse to a binary winner. Pairwise remains the more discriminative protocol for ranking judges; the pointwise score-gap serves

- as a complementary diagnostic for absolute calibration.

(a) all

(b) fin

=0.945 =0.829

=0.953 =0.852

0

0

gemini-3.1-pro

gemini-3.1-pro

gpt-5.5

gpt-5.5

kimi-k2.6

gpt-5.4

gpt-5.4

gemini-3-flash

gemini-3-flash

kimi-k2.6

5

5

sonnet-4.6

sonnet-4.6

qwen3.6-plus

gemma-4-31b

gemma-4-31b

haiku-4.5

haiku-4.5

qwen3.6-plus

EIPRank

glm-5.1

opus-4.7

10

10

opus-4.7

qwen3.5-397b-a17b

qwen3.5-397b-a17b

glm-5.1

gemma-4-26b-a4b

minimax-m2.7

deepseek-v3.2

qwen3.5-122b-a10b

minimax-m2.7

gemma-4-26b-a4b

15

15

qwen3.5-122b-a10b

gpt-5.4-mini

gpt-5.4-mini

deepseek-v3.2

qwen3.5-35b-a3b

qwen3.5-35b-a3b

mistral-small-4-119b

mistral-small-4-119b

gpt-oss-120b gpt-oss-20b

gpt-oss-120b gpt-oss-20b

20

20

20 15 10 5 0

20 15 10 5 0

(c) med

(d) ml

=0.944 =0.842

=0.945 =0.820

0

0

gemini-3.1-pro

gemini-3.1-pro

gpt-5.5

gpt-5.5

qwen3.6-plus

kimi-k2.6

gemini-3-flash

sonnet-4.6

kimi-k2.6

gpt-5.4

5

5

sonnet-4.6

qwen3.6-plus

opus-4.7

gemini-3-flash

gemma-4-31b glm-5.1

glm-5.1

gemma-4-31b

EIPRank

deepseek-v3.2

haiku-4.5

10

10

gpt-5.4

opus-4.7

haiku-4.5

qwen3.5-397b-a17b

qwen3.5-397b-a17b

gemma-4-26b-a4b

gemma-4-26b-a4b

qwen3.5-35b-a3b

qwen3.5-35b-a3b

deepseek-v3.2

15

15

qwen3.5-122b-a10b

minimax-m2.7

minimax-m2.7

gpt-5.4-mini

gpt-5.4-mini

qwen3.5-122b-a10b

mistral-small-4-119b

mistral-small-4-119b

gpt-oss-20b

gpt-oss-120b gpt-oss-20b

20

20

gpt-oss-120b

20 15 10 5 0

20 15 10 5 0

BT Elo Rank

BT Elo Rank

- Figure 14: Judge ranks under Bradley–Terry Elo (x-axis) and Empirical Interaction Propagation (y-axis), one panel per slice. Each marker is a judge; the dashed line is y = x. Both axes are inverted so that rank 1 sits at the top-right. Per-panel Spearman ρ and Kendall τ are computed on the underlying scores.

(a) Elo agreement (Spearman = 0.809, Kendall = 0.660)

(b) Pointwise score-gap calibration

1800

gemini-3.1-pro

- 0

- 1

- 2

- 3

- 4

1700

sonnet-4.6

gpt-5.4 opus-4.7

gpt-5.5

Meanscoregap(goodminusflawed)

kimi-k2.6

glm-5.1

qwen3.5-397b-a17b

1600

gemini-3-flash

qwen3.5-122b-a10b

gemma-4-31b

qwen3.6-plus

PointwiseElo

haiku-4.5

qwen3.5-35b-a3b

gemma-4-26b-a4b

1500

gpt-5.4-mini

deepseek-v3.2

1400

minimax-m2.7

mistral-small-4-119b

1300

gpt-oss-120b

gpt-oss-20b

y = x

1200

800 1000 1200 1400 1600 1800

gemini-3.1-proqwen3.5-397b-a17bglm-5.1kimi-k2.6gpt-5.5opus-4.7gemini-3-flashgpt-5.4gemma-4-31bqwen3.5-122b-a10bsonnet-4.6qwen3.5-35b-a3bgemma-4-26b-a4bqwen3.6-plusdeepseek-v3.2haiku-4.5gpt-5.4-minigpt-oss-20bminimax-m2.7mistral-small-4-119bgpt-oss-120b

Pairwise Elo (on pointwise subset, n = 100 pairs)

- Figure 15: Pointwise vs. pairwise judging on a 100-pair stratified sample. (a) Each judge’s pointwise Elo plotted against its pairwise Elo recomputed on the same pair set; the dashed line marks y = x. (b) Per-judge pointwise score-gap: the mean Likert score on the good conversation in each pair minus the mean score on the flawed one, sorted descending.

### B Reproducibility

#### B.1 Bradley–Terry Elo Details

Setup, log-likelihood, and MM iteration. With notation as in Section 3.2, the BT win probability Pr(c = 1 | j,q) = θj/(θj + θq) induces the log-likelihood

clog θ θj

j+θq + (1 − c)log θ θq

j+θq . (2)

ℓ(θ) =

(j,q,c)∈D

Let wi = (j,q,c)∈D c1[i = j] + (1 − c)1[i = q] denote the total wins of player i, and let nik = {t : {jt,qt} = {i,k}} denote the total number of matches between players i and k. The standard MM update [50] maximises (2) via

θi(t+1) =

wi

. (3)

nik θi(t) + θk(t)

k : nik>0

After each iteration we mean-normalise θ ← θ/θ¯ to fix the otherwise unidentifiable global scale, and we declare convergence when ∥θ(t+1) − θ(t)∥∞ < 10−6 (or after 1000 iterations). To avoid numerical issues we floor the strengths at 10−10 before taking logarithms, and finally we map the strengths to the Elo scale,

Ri = 400log10 θi + R0, (4) with R0 = 1500.

Cluster-robust sandwich standard errors. A single conversation pair contributes one match per judge, so the residuals within a pair are not independent: a pair that happens to be ambiguous will fool many judges simultaneously. We therefore report a sandwich variance estimator with clustering

- at the pair level. Working in the log-strength parameterisation βi = log θi, define for each match (j,q,c)

1

1 + e−ηj,q , wj,q = σj,q 1 − σj,q . (5) The observed information matrix I ∈ R(M+Q)×(M+Q) accumulates the symmetric contribution

ηj,q = βj − βq, σj,q =

Ijj + =wj,q, Iqq + =wj,q, Ijq − =wj,q, Iqj − =wj,q, (6)

for every (j,q,c) ∈ D. For each pair-cluster ρ ∈ Q we accumulate a score vector sρ ∈ RM+Q from its residuals rj,q = c − σj,q:

sρ[j] + =rj,q, sρ[q] − =rj,q, ∀(j,q,c) ∈ D with q = ρ. (7) The cluster-robust “meat” matrix is then

sρs⊤ρ , (8)

##### B =

ρ∈Q

and the sandwich variance on the log-strength scale is

##### V = I+ B I+, (9)

where I+ denotes the Moore–Penrose pseudoinverse, used because I has an unavoidable rank1 deficiency from the global-shift invariance of β (and may have additional deficiencies if the comparison graph splits into disconnected components, which we detect and warn about). Mapping back to the Elo scale via (4) gives

400 ln10

V ii, CI95(i) = 1.96 · SEElo(i). (10) We report Ri ± CI95(i) for every judge and pair.

SEElo(i) =

#### B.2 Empirical Interaction Propagation (EIP) Details

Let J = {1,...,M} be the set of LLM judges and Q = {1,...,Q} be the set of conversation pairs. We assume full observability: every judge has been evaluated on every pair, with outcome cj,q ∈ {0,1} equal to 1 iff judge j produces the correct verdict on pair q under the joint criterion of Eq 1 (matching the ground-truth verdict y⋆, flawed turn r⋆, and failure category t⋆). EIP casts judge-quality scoring as a PageRank-style random walk on the bipartite graph of judges and pairs. We compute it on the filtered match set, where any pair on which a judge is missing or on which all judges agree (unanimously correct or unanimously incorrect) is removed; such pairs carry no discriminative signal and break the identifiability of the random walk.

Let A ∈ {0,1}Q×M be the binary competency matrix, with Aq,j = cj,q. Let 1 denote the Q × M matrix of ones, and define the failure matrix

A = 1 − A. (11)

The per-pair correctness count and per-judge failure count are floored at 1 to avoid division by zero on (rare) all-correct rows or all-failed columns,

Sq = max

j

Aq,j, 1 , Fj = max

q

Aq,j, 1 . (12)

We form two row-stochastic transition matrices on the bipartite graph,

PQ→M[q,j] =

Aq,j Sq

, PM→Q[j,q] =

Aq,j Fj

. (13)

Walking from a pair, the chain hops to a judge that solved it (weighted by that judge’s competence on this pair); walking from a judge, the chain hops to a pair that judge failed (weighted by that judge’s failure mass). Iterating this walk concentrates probability mass on judges that solve hard pairs and on pairs that fool strong judges—precisely the desired “difficulty-aware” ranking.

With damping factor α = 0.85 and uniform teleport vectors uM = 1M/M and uQ = 1Q/Q, we run the coupled iteration

πQ(t+1) = α PM⊤→Q πM(t) + (1 − α)uQ, (14) πM(t+1) = α PQ⊤→M πQ(t+1) + (1 − α)uM, (15)

initialised at πM(0) = uM and πQ(0) = uQ. Both vectors are renormalised to sum to 1 after each step to prevent numerical drift, and we declare convergence when

πM(t+1) − πM(t) 1 + πQ(t+1) − πQ(t) 1 < 10−6, (16)

or after 100 iterations. Finally, both stationary distributions are min–max rescaled to the interpretable range [0,100]. For v ∈ {πM,πQ},

vi − mink vk maxk vk − mink vk

, (17)

scoreEIP(i) = 100 ·

with the degenerate convention scoreEIP(i) = 50 for all i whenever maxk vk − mink vk < 10−12. The reported judge ranking is the descending sort of {scoreEIP(j)}j∈J , and the corresponding pair difficulty is {scoreEIP(q)}q∈Q.

#### B.3 Prompt for Assistant Failure Type Discovery

The following box gives the prompt used for automated discovery and refinement of the assistant failure type taxonomy used in Section 3.1.

##### Assistant Failure Type Discovery Prompt

You are an expert in multi-turn conversation evaluation, specializing in identifying the challenging scenarios, conducting failure mode and taxonomy analysis.

Your task is fourfold:

- (1) Task 1: per-sample annotation. Apply an existing taxonomy to label assistant and user behaviors in multi-turn conversations
- (2) Task 2: identify taxonomy gaps. Identify cases that do NOT fit the taxonomy
- (3) Task 3: taxonomy refinement. Categorize the missed cases and propose refined or new categories grounded in evidence
- (4) Task 4: insights generation. Produce outputs for downstream analysis, explanations for the proposal.

## TAXONOMY DEFINITIONS ASSISTANT_BEHAVIORS = { "disorganized": {...}, "fabricated_answer": {...}, "unnecessary_refusal": {...},

... }

USER_BEHAVIORS = { "focused": "...", "skeptical": "...", "misinformed": "...", "exploratory": "...", ... }

## GENERAL GUIDELINES

- - Be precise, not verbose; avoid generic explanations.
- - Ground reasoning in observed data; do NOT hallucinate missing context.
- - Prefer concrete behavioral descriptions over abstract wording. Identify ambiguities in definitions.
- - Be critical of the taxonomy when necessary, but only propose changes supported by repeated evidence. ## DATA EXAMPLES

#### B.4 Assistant Failure Types

The following boxes define the expected behaviours for assistant failure types, and the corresponding benign behaviours when no failure is present. These are the full versions of the behaviour specifications that assistants are conditioned on, corresponding to Table 2.

##### self_contradiction — Flaw

The assistant subtly contradicts a claim from an earlier correct turn — not by lexically flipping it, but by shifting a qualifier (‘primary factor’ becomes ‘one of several factors’), reframing emphasis, or stating something whose implications conflict with what the assistant said before. Use different wording so the inconsistency isn’t surface-detectable; catching it requires mentally reconciling the flawed later turn against an earlier one. The flawed round is the later turn whose claim no longer matches what was said before, while the earlier reference turn remains accurate and grounded in the source. Avoid lexically obvious flips, and do not restate the earlier claim verbatim immediately before contradicting it. Non-flawed turns stay consistent with the assistant’s own prior statements.

##### self_contradiction — Benign

Stay consistent with your own earlier turns — match the scope, strength, and qualifiers of prior claims across the conversation. If your understanding genuinely evolves, explicitly acknowledge what changed and why rather than silently shifting.

evasion — Flaw

The chatbot produces a response that looks thorough and knowledgeable but avoids giving the user what they actually asked for. This can take two forms, or blend them: (a) surface specificity that hedges on the central claim — citing a number for a minor result while glossing the key finding, or naming a method without explaining what makes it distinct; (b) tangential drift emphasizing adjacent or related material so the answer reads plausibly while missing the core point. Either way, the response should look reasonable to a skimming reader but leave the user unable to verify or act on the central claim. Non-flawed turns answer the user’s actual question directly with the key details, exact numbers, and specific terms the question demands.

#### evasion — Benign

For each question, provide a complete, specific, on-topic answer — every sentence directly addresses what the user asked, with the key details, exact numbers, and specific terms that are relevant to the actual question.

##### disorganized — Flaw

The chatbot buries key information in text that looks similar in length and formatting to clean responses but lacks logical flow — jumping between points, mixing high-level claims with low-level details mid-paragraph, and failing to foreground the answer the user actually asked for. The flaw is structural, not visual: the response should not look like a wall of text. The correct content is present but requires significant effort to extract. Non-flawed turns should present information clearly and directly.

##### disorganized — Benign

Present information with clear structure — lead with the key point, organize logically, make the answer easy to scan and extract insights from.

##### fabricated_answer — Flaw

The chatbot fabricates a confident-sounding answer when the question is not answerable from the reference material, instead of honestly saying it doesn’t have enough information. Nonflawed turns should acknowledge limitations when the question goes beyond what the reference material supports.

##### fabricated_answer — Benign

Some of the user’s questions go beyond what the reference material covers. The assistant recognizes these and honestly says it doesn’t have enough information rather than guessing, while answering confidently when the material does cover the question. Demonstrate clear calibration between what the source supports and what it doesn’t — at least one turn should acknowledge a limit.

##### instruction_forgetting — Flaw

Early in the conversation the user establishes an explicit content-level directive — a scope restriction (‘focus only on the ablation results, skip the architecture discussion’; ‘only pull from the abstract, not the full methods’), an audience/framing (‘I’m a clinician making treatment decisions, not a student’; ‘graduate-level CS, assume transformer basics’), self-context (‘I’m researching Alzheimer’s specifically, not dementia broadly’; ‘I’ve already read section 3, don’t re-explain it’), a prioritization directive (‘I care more about limitations than strengths’; ‘methodology over results’), or a negative content instruction (‘don’t walk me through the reasoning, just give the conclusion’; ‘don’t re-cover material we already discussed’). The assistant honors the directive through the intermediate rounds. In the flawed round it silently violates the directive while otherwise answering competently — pulling in out-of-scope material ‘to provide context’, drifting to a different audience level, re-explaining what the user said to skip, or slipping into the deprioritized angle. The violation must be partial or plausibly-justified (not a wholesale abandonment) and must not be lexically announced — no ‘let me expand beyond...’, ‘stepping outside the scope for a moment...’. Do not use format/stylistic directives (bullets vs prose, word counts, citation style) since those are surface-detectable and don’t test reasoning. Non-flawed turns apply the directive consistently.

##### instruction_forgetting — Benign

Early in the conversation the user establishes an explicit content-level directive — a scope restriction, audience/framing, self-context, prioritization, or negative-content instruction (not surface formatting). The assistant tracks the directive and applies it consistently across every subsequent turn, even when not restated. Responses should demonstrate that the directive is actively shaping what gets included, emphasized, or omitted — not by naming the directive, but through the content choices themselves. If a later question would genuinely conflict with the directive, flag the conflict and ask rather than silently deviating.

##### no_clarification — Flaw

When the user’s question is ambiguous or underspecified — e.g., it could refer to multiple parts of the material, or lacks enough context to give a precise answer — the chatbot picks one interpretation and answers confidently instead of asking the user to clarify. Non-flawed turns should ask for clarification when the question is genuinely ambiguous.

##### no_clarification — Benign

At least one of the user’s questions is genuinely ambiguous or underspecified (could refer to multiple parts of the material, or lacks enough context for a precise answer). On those, the assistant asks for clarification before answering rather than silently assuming one interpretation. On unambiguous questions, the assistant answers directly without performative clarification requests.

##### unnecessary_refusal — Flaw

The chatbot unnecessarily refuses or hedges on questions that ARE answerable from the reference material, saying things like ‘I don’t have enough information’ or ‘I can’t determine that’ when the answer is clearly available. Non-flawed turns should answer confidently using the reference material.

##### unnecessary_refusal — Benign

The user asks questions that sit near the edge of what the source covers but are answerable if the assistant reads the material carefully. The assistant answers confidently using the available

information without unnecessary hedging or refusal. Only hedge when a question is genuinely outside the source’s scope.

#### B.5 User Behavior Types

The following boxes define the user-behavior taxonomy used in Section 3.1. Each generated pair samples one user behavior, which conditions how the simulated user interacts with the assistant across all rounds. The seven behaviors split into navigation patterns (how the user moves through the material) and epistemic stances (how the user relates to the material’s content). Keys match the labels used by the pipeline.

focused

The user picks one or two specific aspects of the material and explores them in depth across multiple turns, asking follow-up questions that build on previous answers.

integrative

The user asks broad questions connecting different parts of the material - linking details across sections, comparing information, wanting synthesis rather than isolated facts.

scattered

The user jumps abruptly between unrelated aspects of the material without transitions, forcing the assistant to handle rapid context switches.

skeptical

The user plays devil’s advocate - questioning claims, asking for evidence, pointing out apparent tensions between answers, and requiring justification before accepting conclusions.

misinformed

The user confidently states incorrect things about the material (wrong numbers, misattributed claims, flipped conclusions) and builds questions on these false premises. The assistant should correct these errors.

exploratory

The user treats the material as a starting point and asks about implications, broader connections, or extensions that may go beyond what the reference material covers.

underspecified

The user sometimes asks vague or ambiguous questions - using unclear references, omitting key qualifiers, or phrasing questions that could apply to multiple parts of the material - forcing the assistant to either guess the intent or ask for clarification.

#### B.6 Prompts for Dual Conditioned Conversation Pair Generation

The following boxes display the prompts used for generating conversation based on blueprints, as described in Section 3.1. The two prompts give instructions for generating a good conversation, and a worse conversation which intentionally has weaknesses.

##### good_convo

This conversation will be used in a pairwise comparison against a counterpart conversation that contains a subtle "{flaw_name}" flaw in exactly one round. The goal of the pair is to test whether an LLM judge can pick the better conversation --- which

only works if your conversation is absolutely correct on every dimension and naturally embodies the paired virtue, without performing it loudly. If your conversation is obviously stronger than a clean conversation with one subtle flaw, the

test becomes trivial. Aim to read as ordinary high quality, not as a demonstration of the virtue.

Given this reference document, generate a natural multi-turn conversation ({n} rounds, each round = 1 user + 1 assistant message).

The context is reference material - it defines what the assistant knows about the topic. The conversation does NOT need to cover all of it. A focused discussion exploring a few topics in depth is better than superficially touching everything.

First, write a turn-by-turn plan with one labeled entry per round in the form ‘Round 1: ...‘, ‘Round 2: ...‘, ..., ‘Round {n}: ...‘. Every round MUST have its own entry

--- do not collapse rounds into a single paragraph or skip ahead. For each round, capture (a) what the user asks and what motivates it (building on the prior round where natural), and (b) what the assistant’s response will focus on from the reference

material. The plan should form a coherent arc, not isolated Q&A. Then write the conversation faithfully executing the plan. User behavior: "{user_behavior_name}" - {user_behavior_desc} Assistant virtue to embody naturally (the quality every round must quietly live up to

--- do NOT overact it): {assistant_virtue} Paired flaw the counterpart conversation will contain (for your awareness only --- do NOT mention, contrast, or over-compensate against it; just never exhibit it): { assistant_flaw} User rules:

- - Has read the reference material. Asks questions driven by genuine curiosity, not to quiz the assistant.
- - Apply the above behavior consistently across ALL rounds - it should shape what the user asks, how they react, and how the conversation progresses.
- - NOT a passive question-asker - reacts to answers, builds on them, pushes back, or redirects.
- - Keep messages SHORT and natural (1-3 sentences). Real users don’t write essays. Assistant rules:
- - Every round must be clean on every dimension: accurate, grounded, specific, on-topic , well-structured, responsive, and appropriately calibrated. No flaw category should surface in any round.
- - Answers must be grounded in the provided context. Do NOT fabricate information beyond what the context provides.
- - If a user question goes beyond what the context covers, honestly acknowledge the limitation rather than guessing or fabricating.
- - Be specific - use the actual numbers, details, and conclusions from the reference material.
- - Be concise and direct (2-5 sentences per response). Avoid filler, repetition, and preamble. Match the tone, length, and confidence of a typical clean response --- do not inflate quality signals to appear more virtuous.

=== Reference Context === {context}

##### worse_convo

Given this reference document, generate a natural multi-turn conversation ({n} rounds, each round = 1 user + 1 assistant message) where the chatbot has a specific weakness.

Chatbot weakness: **{behavior_type}** Description: {behavior_desc}

The context is reference material - it defines what the assistant knows about the topic. The conversation does NOT need to cover all of it. A focused discussion is fine

.

First, write a turn-by-turn plan in the "reasoning" field with one labeled entry per round in the form ‘Round 1: ...‘, ‘Round 2: ...‘, ..., ‘Round {n}: ...‘. Every round MUST have its own entry --- do not collapse the plan into a single justification paragraph or skip ahead. For each round, capture (a) what the user asks and what motivates it (reacting to the prior round where natural), and (b) what the assistant’s

response will focus on from the reference material. The plan should form a coherent arc, not isolated Q&A. Pick one round to be the flawed one and, inside that round’s entry, describe concretely how the "{behavior_type}" weakness will surface there --the clean rounds before and after should carry the conversation as if no flaw exists. Set "bad_round_index" to that round (1-indexed). Then write the conversation faithfully executing the plan.

User behavior: "{user_behavior_name}" - {user_behavior_desc} User rules:

- - Has read the reference material. Asks questions driven by genuine curiosity.
- - Apply the above behavior consistently across ALL rounds.
- - NOT a passive question-asker - reacts to answers naturally. If the chatbot gives an evasive answer, the user may follow up, push back, or move on - whatever fits. Do NOT have the user explicitly diagnose or name the flaw.
- - Keep messages SHORT and natural (1-3 sentences). Chatbot (assistant) rules:
- - Answers must be grounded in the provided context. Do NOT fabricate information beyond what the context provides.
- - Be specific - use the actual numbers, details, and conclusions from the reference material.
- - Be concise and direct (2-5 sentences per response). Avoid filler, repetition, and preamble.
- - In exactly 1 round, the assistant response should exhibit the "{behavior_type}" weakness. The flaw should require careful reading to detect --- a reader skimming should NOT notice anything wrong. Keep the same confident tone and similar response length as other rounds.
- - The flawed round must exhibit ONLY the "{behavior_type}" flaw --- no other weakness category should appear in it. Every dimension of the response other than the named flaw must remain clean: accurate, grounded, specific, on-topic, well-structured, and responsive. All non-flawed rounds must be clean on every dimension.
- - Do NOT mention or acknowledge that responses are flawed.

=== Reference Context === {context}

#### B.7 Prompts for Automated Quality Control

The following boxes display the prompts used for automated quality control of generated conversations, namely for checking coherence, adherence, and grounding as described in Section 3.1. Breaking generated text into atomic claims is a common practice when verifying factuality of statements [59]. Rather than demand exact factuality of all claims, it is also possible to relax to a specified precision level and calibrate removal of conversations using statistical techniques [60].

##### coherence

You are auditing a synthetic-data recipe before any conversation has been rendered. Given a reference document and a sampled plan, decide whether the plan can plausibly produce a natural multi-turn conversation.

A plan is INCOHERENT only when executing the stated virtue/flaw is semantically at odds with the user style for THIS specific reference material - not merely unusual. Surface-level mismatches (e.g. a skeptical user asking precise questions) are fine. Flag only genuine conflicts, e.g.:

- - the flaw requires conditions the user style will never create (a "focused" user asking tightly scoped questions cannot surface a "no_clarification" flaw)
- - the reference material lacks the content needed for the flaw to appear naturally (e. g. "factual_error" flaw but the material is too thin to have subtle facts to misstate)
- - the bad_round_index sits outside [1, n_rounds] or collides with where the plan claims the conversation is just getting started / wrapping up

User behavior: "{user_behavior_name}" - {user_behavior_desc} Assistant virtue (good convo): {virtue} Assistant flaw (bad convo, "{assistant_behavior_name}"): {flaw} n_rounds: {n_rounds} bad_round_index (1-indexed): {bad_round_index}

=== Good plan (from generator) === {good_plan}

=== Bad plan (from generator) === {bad_plan}

=== Reference Context === {context}

Respond in JSON: {"good_ok", "good_issue", "bad_ok", "bad_issue"}. Leave *_issue as "" when ok. Keep issues to one sentence.

##### adherence

You are auditing whether two rendered conversations actually followed their generation plans. Read the plans and then the conversations and answer yes/no globally for each side. For the GOOD conversation:

- - Did the assistant consistently display the stated virtue across ALL rounds?
- - Did the user maintain the stated user behavior across ALL rounds? For the BAD conversation:
- - Did the user maintain the stated user behavior?
- - Does round {bad_round_index} (1-indexed) actually exhibit the "{ assistant_behavior_name}" flaw described below?
- - Is that the ONLY flawed assistant round? (If another round is clearly flawed, bad_flaw_round_correct is false.)

User behavior: "{user_behavior_name}" - {user_behavior_desc} Assistant virtue: {virtue} Assistant flaw "{assistant_behavior_name}": {flaw} n_rounds: {n_rounds}

=== Good plan === {good_plan}

=== Good conversation === {good_convo}

=== Bad plan === {bad_plan}

=== Bad conversation (declared bad_round_index = {bad_round_index}) === {bad_convo}

Respond in JSON: {"good_followed", "good_issue", "bad_followed", " bad_flaw_round_correct", "bad_issue"}. Leave *_issue as "" when ok. One sentence per issue.

##### grounding

You are fact-checking an assistant’s responses against a reference document. For each listed assistant turn, extract every factual atomic claim it makes (specific numbers, named entities, mechanisms, cause/effect statements, attributions) and mark each claim

as grounded or not. Rules:

- - A claim is ‘grounded: true‘ only if it is directly supported by the reference context below.
- - Paraphrases and reasonable summaries of supported content count as grounded.
- - Reasonable inferences or synthesis across multiple context statements also count as grounded, even if no single sentence states them verbatim.
- - Unsupported extrapolation, invented details, or claims beyond the context are ‘ grounded: false‘.
- - Ignore conversational filler, hedges, and meta-statements ("good question", "let me explain").
- - Do NOT extract negative/absence claims ("X is not Y", "X doesn’t mention Z") unless the context directly contradicts them - absence is not verifiable at claim granularity

.

- - Do NOT extract meta-statements where the assistant is evaluating another statement rather than asserting a fact ("it is inaccurate to claim...", "that earlier answer was

wrong").

- - Aim for 1-5 atomic claims per turn. Skip turns whose round_index is in skip_rounds: {skip_rounds}.

=== Assistant turns ({label}) === {turns}

=== Reference Context === {context}

Respond in JSON: {"rounds": [{"round_index", "claims": [{"claim", "grounded"}]}]}. Only include rounds you actually analyzed (exclude skip_rounds).

#### B.8 Prompt for Judges and its Variants

The following boxes display the prompts used for the main judge results across all 21 models, and variants used as ablations throughout Section 4. LLM-as-a-judge systems can be sensitive to prompt phrasing. Our prompt revisions were refined manually, but future iterations of the benchmark could evaluate judge robustness using prompt-level uncertainty quantification [61].

##### judge

You are evaluating two multi-turn conversations about the same reference material between a user and an assistant. The two conversations may have different user questions and different assistant responses - your job is to judge which conversation’ s assistant provides better overall quality. {context_section}

- === Conversation A ===

- {text_a}

=== Conversation B ===

- {text_b}

The worse conversation has exactly one problematic assistant response. Classify the flaw using exactly one of these categories: [The seven assistant-weakness definitions A–G from the taxonomy are inserted verbatim here, one bullet per category in the form “- **<letter> (<name>)**: <flaw description>”.]

Quality flaws (evasion, disorganization, missed clarification, unnecessary refusal) count as much as factual mistakes when deciding which conversation is worse - a response can be problematic with no factual error at all. When classifying, pick the category whose definition the flawed round most closely matches; do not default to whichever category feels most common or most obvious. Each of A-G is a distinct failure mode and exactly one should fit.

Respond in JSON with these fields in order:

- - "analysis": your reasoning about which conversation is better and why
- - "worst_round": the round number (starting from 1) containing the worst assistant response in the conversation you did NOT pick, where each round is one user message followed by one assistant response
- - "problem_type": the letter (A-G) for the flaw category from the list above
- - "verdict": "A" or "B" for the better conversation

#### judge_v2: confusion-pair hints

You are evaluating two multi-turn conversations about the same reference material between a user and an assistant. The two conversations may have different user questions and different assistant responses - your job is to judge which conversation’ s assistant provides better overall quality. {context_section}

- === Conversation A ===

- {text_a}

=== Conversation B ===

- {text_b}

The worse conversation has exactly one problematic assistant response. Classify the flaw using exactly one of these categories: [The seven assistant-weakness definitions A–G from the taxonomy (Sec. 3.1) are inserted verbatim here, one bullet per category in the form “- **<letter> (<name>)**: <flaw description>”.]

Quality flaws (evasion, disorganization, missed clarification, unnecessary refusal) count as much as factual mistakes when deciding which conversation is worse - a response can be problematic with no factual error at all. When classifying, pick the category whose definition the flawed round most closely matches; do not default to whichever category feels most common or most obvious. Each of A-G is a distinct failure mode and exactly one should fit.

Additional guidance for two categories that are easy to confuse with B (evasion):

- - **F (no_clarification)** requires that (i) the user’s question is genuinely ambiguous AND (ii) the substantive answer would change depending on how the ambiguity is resolved. If an assistant asks a clarifying question and then gives a reasonable answer in the next round (or the user does not clarify), that is appropriate clarification behavior — it is NOT B (evasion). Only treat a missed clarification as F when the disambiguation would meaningfully change the answer.
- - **A (self_contradiction)** does not require the contradicted claim to be on the round’s main question. A claim that contradicts an earlier turn — even when the contradiction is a tangential or seemingly random fact rather than the central point is still A, not B (evasion). Do not default to B just because the inconsistency is off-topic relative to the user’s current question. Respond in JSON with these fields in order:
- - "analysis": your reasoning about which conversation is better and why
- - "worst_round": the round number (starting from 1) containing the worst assistant response in the conversation you did NOT pick, where each round is one user message followed by one assistant response
- - "problem_type": the letter (A-G) for the flaw category from the list above
- - "verdict": "A" or "B" for the better conversation

#### judge_v3: forced candidate comparison

You are evaluating two multi-turn conversations about the same reference material between a user and an assistant. The two conversations may have different user questions and different assistant responses - your job is to judge which conversation’ s assistant provides better overall quality. {context_section}

- === Conversation A === {text_a}
- === Conversation B === {text_b}

The worse conversation has exactly one problematic assistant response. Classify the flaw using exactly one of these categories: [The seven assistant-weakness definitions A–G from the taxonomy (Sec. 3.1) are inserted verbatim here, one bullet per category in the form “- **<letter> (<name>)**: <flaw description>”.]

Quality flaws (evasion, disorganization, missed clarification, unnecessary refusal) count as much as factual mistakes when deciding which conversation is worse - a response can be problematic with no factual error at all. When classifying, pick the category whose definition the flawed round most closely matches; do not default to whichever category feels most common or most obvious. Each of A-G is a distinct failure mode and exactly one should fit.

Before committing to a category, you MUST explicitly compare candidates. Identify the two or three categories from A-G that could plausibly fit the flawed round, then for each candidate state (i) the specific evidence in the round that fits its definition and (ii) the strongest reason it might not fit. Only after that comparison should you commit to the single best fit, briefly explaining why it beats the runners-up. Do not pick the first category that comes to mind without weighing at least one alternative against it.

Respond in JSON with these fields in order:

- - "analysis": your reasoning. This must include the candidate comparison described above before you commit to a problem_type.
- - "worst_round": the round number (starting from 1) containing the worst assistant response in the conversation you did NOT pick, where each round is one user message followed by one assistant response
- - "problem_type": the letter (A-G) for the flaw category from the list above
- - "verdict": "A" or "B" for the better conversation

#### judge_v4: decision rules + comparison

You are evaluating two multi-turn conversations about the same reference material between a user and an assistant. The two conversations may have different user questions and different assistant responses - your job is to judge which conversation’s assistant provides better overall quality. {context_section}

- === Conversation A === {text_a}
- === Conversation B === {text_b}

The worse conversation has exactly one problematic assistant response. Classify the flaw using exactly one of these categories: [The seven assistant-weakness definitions A–G from the taxonomy (Sec. 3.1) are inserted verbatim here, one bullet per category in the form “- **<letter> (<name>)**: <flaw description>”.]

Quality flaws (evasion, disorganization, missed clarification, unnecessary refusal) count as much as factual mistakes when deciding which conversation is worse - a response can be problematic with

no factual error at all. When classifying, pick the category whose definition the flawed round most closely matches; do not default to whichever category feels most common or most obvious. Each of A-G is a distinct failure mode and exactly one should fit.

Discriminating tests for the categories most often confused with B (evasion):

- - **F (no_clarification) decision rule.** F applies only when BOTH conditions hold: (i) the user’s question is genuinely ambiguous or underspecified, AND (ii) the substantive answer would meaningfully change depending on the disambiguation. If either condition fails, the round is not F. Crucially, if the assistant *did* ask a clarifying question, the round is not F and is also not B (evasion) — asking for clarification is appropriate behavior even when the question turns out not to need it. F means *failing to ask when asking was needed*; it does not mean *asking unhelpfully*. Do not label a round F or B simply because the assistant’s clarifying question feels unnecessary in hindsight.
- - **A (self_contradiction) decision rule.** A applies whenever the flawed round contains an assertion that is inconsistent with something the assistant said in an earlier turn — even if the contradicted claim is tangential, a side-remark, or a random fact unrelated to the round’s main question. The contradiction does not have to be on the central topic. If you can identify any earlier assistant statement that the flawed round’s claims are at odds with, prefer A over B (evasion). Do not default to evasion just because the inconsistency feels off-topic or minor.
- - **B (evasion) is reserved for** rounds where the assistant engages with the central question but produces a response that hedges, drifts to adjacent material, or substitutes surface specificity for the actual answer. If the flawed behavior is “asked an unhelpful clarifying question” → not B (consider F’s rule above; usually it is none of A-G). If the flawed behavior is “made an inconsistent claim” → prefer A.

Before committing to a category, you MUST explicitly compare candidates. Identify the two or three categories from A-G that could plausibly fit the flawed round, then for each candidate state (i) the specific evidence in the round that fits its definition and (ii) the strongest reason it might not fit. When F or B is among your candidates, you MUST apply the F decision rule by name. When A or B is among your candidates, you MUST apply the A decision rule by name. Only after that comparison should you commit to the single best fit, briefly explaining why it beats the runners-up. Do not pick the first category that comes to mind without weighing at least one alternative against it.

Respond in JSON with these fields in order:

- - "analysis": your reasoning. This must include the candidate comparison described above before you commit to a problem_type.
- - "worst_round": the round number (starting from 1) containing the worst assistant response in the conversation you did NOT pick, where each round is one user message followed by one assistant response
- - "problem_type": the letter (A-G) for the flaw category from the list above
- - "verdict": "A" or "B" for the better conversation

#### judge_v5: structured two-stage CoT

You are evaluating two multi-turn conversations about the same reference material between a user and an assistant. The two conversations may have different user questions and different assistant responses - your job is to judge which conversation’s assistant provides better overall quality. {context_section}

- === Conversation A === {text_a}
- === Conversation B === {text_b}

The worse conversation has exactly one problematic assistant response. Classify the flaw using exactly one of these categories: [The seven assistant-weakness definitions A–G from the taxonomy (Sec. 3.1) are inserted verbatim here, one bullet per category in the form “- **<letter> (<name>)**: <flaw description>”.]

Quality flaws (evasion, disorganization, missed clarification, unnecessary refusal) count as much as factual mistakes when deciding which conversation is worse - a response can be problematic with

no factual error at all. When classifying, pick the category whose definition the flawed round most closely matches; do not default to whichever category feels most common or most obvious. Each of A-G is a distinct failure mode and exactly one should fit.

Before committing to a category, your “analysis” field MUST proceed in two stages.

- **Stage 1 — Findings.** Once you have located the flawed round, answer the following five diagnostic questions in order. Each answer is “Yes” or “No” followed by one short sentence of evidence drawn from the conversation. Do not skip any question; do not collapse them into prose. These findings constrain which categories are even eligible — you will use them in Stage 2.

- Q1. Is the user’s question in the flawed round genuinely ambiguous or underspecified (could

plausibly refer to multiple distinct things in the source)?

- Q2. If Q1 is Yes: would the substantive answer meaningfully change depending on the

disambiguation? (If Q1 is No, answer “N/A”.)

- Q3. Did the assistant in the flawed round ask a clarifying question (rather than committing to

one interpretation)?

- Q4. Does the flawed round contain any assertion — central or tangential — that is inconsistent

with something the assistant said in an earlier turn?

- Q5. Does the flawed round decline, hedge, or claim insufficient information for a question that

the reference material actually answers?

- **Stage 2 — Comparison and commit.** Using your findings, identify the two or three categories from A-G that could plausibly fit, and for each candidate state (i) the specific evidence in the round that fits its definition and (ii) the strongest reason it might not fit, citing the relevant findings (Q1-Q5) by number. The findings are operational tests:

- F (no_clarification) is eligible only if Q1=Yes AND Q2=Yes AND Q3=No. If Q3=Yes, F is ruled out (asking for clarification, even unhelpfully, is not the F flaw).

- - A (self_contradiction) is eligible whenever Q4=Yes, regardless of whether the contradiction is

on the round’s main question or a tangential claim.

- - B (evasion) is the right call when the assistant engages with the central question but hedges,

drifts, or substitutes surface specificity for the actual answer. If Q3=Yes (assistant asked a clarifier), B is usually ruled out for that reason. If Q4=Yes, prefer A over B.

- D (fabricated_answer) requires the assistant to confidently produce content the source does not support. Do not promote D to B just because the answer feels evasive.

- - G (unnecessary_refusal) requires Q5=Yes AND the question being answerable from the source.

Only after this comparison should you commit to the single best fit, briefly explaining why it beats the runners-up. Do not pick the first category that comes to mind without going through Stages 1 and 2.

Respond in JSON with these fields in order:

- - "analysis": your reasoning. This must include the candidate comparison described above before you commit to a problem_type.
- - "worst_round": the round number (starting from 1) containing the worst assistant response in the conversation you did NOT pick, where each round is one user message followed by one assistant response
- - "problem_type": the letter (A-G) for the flaw category from the list above
- - "verdict": "A" or "B" for the better conversation

##### judge_pointwise

You are evaluating a single multi-turn conversation about reference material between a

user and an assistant. Your job is to score the overall quality of the assistant’s responses across the conversation. The conversation may be entirely clean, or it may contain exactly one problematic assistant response that exhibits one of the categorized weaknesses below. You see this conversation in isolation - there is no counterpart to compare against. Score on its own merits. {context_section}

=== Conversation === {text}

Weakness categories: [The seven assistant-weakness definitions A–G from the taxonomy (Sec. 3.1) are inserted verbatim here, one bullet per category in the form “- **<letter> (<name>)**: <flaw description>”.]

Quality flaws (evasion, disorganization, missed clarification, unnecessary refusal) count as much as factual mistakes when judging quality - a response can be problematic

with no factual error at all. When classifying, pick the category whose definition the worst round most closely matches; do not default to whichever category feels most common or most obvious.

Scoring guide (1-10 Likert, integer):

- - 10: every round is clean - accurate, grounded, specific, on-topic, well-structured, responsive, and appropriately calibrated.
- - 7-9: minor stylistic weaknesses but no real flaw in any round.
- - 4-6: one round contains a clear but moderate flaw from the categories above.
- - 1-3: one round contains a severe flaw, or multiple rounds are problematic. Respond in JSON with these fields in order:
- - "analysis": your reasoning about the conversation’s quality, including which round ( if any) is the weakest and why
- - "score": integer 1-10 reflecting overall assistant quality
- - "is_flawed": true if you believe the conversation contains a problematic round from the categories above, false otherwise
- - "worst_round": the 1-indexed round number containing the worst assistant response ( always required, even if you believe no flaw is present - name the round whose response is comparatively weakest)
- - "problem_type": the letter (A-G) for the flaw category that best describes the worst round (always required - if you do not believe there is a real flaw, name the

category that comes closest to the weakest round)

- Table 7: Supervised fine-tuning hyperparameters. We train one LoRA adapter per Bradley–Terry

difficulty band (q1: top 20%, q2: 20–40%, q3: 40–100%), each with 100 randomly sampled pairs, on top of a frozen Qwen3.5-4B base. Held-out evaluation uses vLLM with guided JSON decoding against the schema in Appendix B.8.

Hyperparameter Value Base model and adapter Base model Qwen3.5-4B Chat template qwen3_5_nothink Finetuning method LoRA LoRA rank 16 LoRA α 32 LoRA target modules all linear layers Precision bf16 Data Examples per band 100 (ALPACA format) Bands (q1, q2, q3) top 20% / 20–40% / 40–100% by BT Elo Input full judge prompt with reference context Output no-CoT JSON (verdict, worst_round, problem_type) Cutoff length 8192 tokens Optimization Optimizer AdamW Learning rate 1 × 10−4 LR schedule cosine Warmup ratio 0.1 Epochs 5 Per-device batch size 2 Gradient accumulation steps 2 Effective batch size 4 Checkpointing every epoch Held-out evaluation Decoding greedy (temperature= 0, top_p= 1) Max generated tokens 32,768 Max model length 49,152 Constrained decoding guided JSON against JUDGE_SCHEMA

#### B.9 SFT Details

- Table 7 displays the model architecture and all the hyperparameters used for supervised fine-tuning with LoRA. C Human Audit

#### C.1 Participants

The authors performed the audit themselves. The audit was carried out as part of regular research duties; no additional compensation was paid and no external annotators were recruited, so no IRB review was required. Each annotator was assigned a disjoint bundle of 20 pairs.

#### C.2 Annotation Tool

The pair-audit interface is a Streamlit application that surfaces every artefact relevant to a single pair on one screen. The sidebar exposes filters for domain, declared assistant behavior, and turn count, together with sort modes including ascending joint correctness (suspicious first) and per-component accuracy. The main view splits into five tabs: Overview (ground-truth verdict, bad_round_index, assistant_behavior_type, user_behavior_type, and a per-judge accuracy summary), Plan (plan.good and plan.bad side-by-side), Conversations (convo_a and convo_b rendered as col-

lapsible per-turn cards, with the injected-flaw turn opened by default and tagged with the declared assistant_behavior_type), Verification (per-claim grounding decisions plus the coherence and adherence gates from Section 3.1), and Judges (each of the 21 judges’ raw verdict, picked turn, picked behavior type, and reasoning). The annotator commits one of three labels (clean, ambiguous, or noise) together with a free-text note; labels persist to a per-annotator JSON file.

[Figure 18]

Figure 16: The pair-audit interface used for the human label-noise audit. The annotator inspects the ground-truth metadata, the good and bad plans, the two conversation transcripts (with the injectedflaw turn opened and tagged), the three verifier outputs, and each of the 21 judges’ verdicts and reasoning, then commits one of three labels (clean, ambiguous, noise) along with a free-text note. The screenshot shows the Conversations tab on a representative ML pair.

#### C.3 Annotation Procedure

Auditing every pair end-to-end is impractical, so each annotator works only through the candidate noisy slice surfaced by Section 3.3 (20 pairs across two parts). Each pair is processed using the following five-step procedure, and the disposition rubric at the end of the subsection maps the resulting observations to one of clean, ambiguous, or noise.

- 1. Triage by judge-disagreement signal. Open the Overview tab and scan the per-judge accuracy summary. Pairs that nearly all judges miss, or that split close to 50/50 between the two candidate conversations, are the most likely to be mislabelled and are inspected first; pairs on which the ensemble agrees with the declared verdict are quickly skimmed and moved through.
- 2. Confirm the injected flaw is on-target. On the Plan tab, read plan.bad and verify that the planned weakness is consistent with the declared assistant_behavior_type. The plan is a single narrative paragraph rather than a turn-by-turn script, but it must be specific enough to determine which turn realises the flaw. Hold the planned weakness in mind for the remaining steps; if the plan describes a different category of flaw than the one declared, label noise immediately.
- 3. Confirm the bad conversation executes the plan. On the Conversations tab, locate the turn at bad_round_index in convo_b and verify that the assistant response realises the

- planned weakness, neither a different category nor a milder version. If convo_b exhibits a different assistant_behavior_type than the one declared, or if no turn realises the planned flaw, label noise.
- 4. Cross-check using the judge ensemble. Open the Judges tab. If the strongest judges in the registry (gemini-3.1-pro, gpt-5.5, opus-4.7) consistently miss the verdict while several weaker judges (gpt-oss-20b, gemma-4-31b) get it right, treat the pair as suspicious. The strong judges’ reasoning text usually points at a competing flaw in convo_a, which is the next thing to verify.
- 5. Verify the good conversation has no competing flaw. For each strong judge that picked the wrong side: (i) locate the disputed turn in convo_a on the Conversations tab, (ii) retrieve the relevant span from metadata.context (the source document), and (iii) consult an external LLM with a focused fact-check prompt that contains only the disputed turn, the matching source span, and the user’s preceding question. Verify the LLM’s answer by hand against the source. This step is the most expensive and is the rate-limiting one on MED and FIN pairs, where domain expertise is needed to resolve subtle factual disputes.

The disposition rubric maps observations to labels: clean when the injected weakness is the only material flaw and both conversations are otherwise faithful to their plans and to the source; ambiguous when failure could fit more than one type defined in the taxonomy; and noise when there was any deviation from the intended label (y⋆,r⋆,t⋆). Examples of noise include when the better conversation contains any clear flaw, the worse conversation has no clear flaw or a flaw in the wrong turn, or when the failure type is different than t⋆. The label-noise rates surfaced by this procedure are reported in Section A.9 and motivate the top-Elo trim of the published evaluation slice.

### D Examples

We illustrate the joint correctness criterion on two pairs labelled clean by the audit of Section C. The first is presented in full (source context, both plans, both transcripts, and a curated subset of judge responses) to make every component of the criterion visible. The second uses a single 21-row table to show how the criterion partitions the full judge ensemble on a representative pair.

#### D.1 Full Example: Unnecessary Refusal in ML

The pair (id = 2d63594b210e) is drawn from the ML domain, with user_behavior_type = underspecified and assistant_behavior_type = unnecessary_refusal. The injected flaw lives in the second assistant turn (bad_round_index = 2); convo_b is the better conversation (better_is_a = FALSE).

Source. The reference document is the ICLR 2022 paper MCL-GAN: Generative Adversarial Networks with Multiple Specialized Discriminators. The relevant content for the second turn is the paper’s description of the flag variables vi,m and uj,m, which are set by the k-largest discriminator scores per sample and are explicitly not included in backpropagation; a balance loss prevents collapse onto a single discriminator.

Plans. The full text of both plans is shown in the two boxes below. The behavioural divergence between them is highlighted in red: turn 2 is where the injected flaw lives, and turn 3 picks a different follow-up topic in each plan (GMAN comparison vs. ADA trick) while keeping both planned responses clean.

##### plan.good

- Turn 1. The user asks whether MCL-GAN was only validated in small or toy settings, motivated by concern about mode-collapse methods not scaling. The assistant will point to the paper’s three-task experimental coverage: unconditional image generation, image-toimage translation, and text-to-image synthesis, including high-resolution conditional synthesis and comparisons with strong baselines.
- Turn 2. Building on that, the user asks vaguely how the discriminators actually specialize and whether the flag variables are learned. The assistant will explain the MCL k-max selection rule for v and u flags, that they are not backpropagated through, and how the balance loss prevents collapse to one discriminator.
- Turn 3. The user pushes on comparison fairness, asking why GMAN was emphasized and other multi-discriminator methods were not central. The assistant will explain that GMAN is the closest baseline for testing specialization, describe the matched DCGAN setup and GMAN variants, and mention the broader comparisons including Inclusive GAN and Albuquerque et al. with the 42.6% memory result.

##### plan.bad

- Turn 1. The user vaguely asks whether the method was only validated in small/simple settings or in broader ones. The assistant will answer cleanly that the paper evaluated MCL-GAN across unconditional image generation, image-to-image translation, and text-toimage synthesis, including high-resolution conditional synthesis and comparisons to multiple baselines including SOTA models.
- Turn 2. Reacting to the training explanation, the user asks about “those little indicator things,” ambiguously referring to the flag variables. The assistant will correctly identify them as vi,m and uj,m, explain that they are set by the k-largest discriminator scores and used to update only a subset of discriminators, but will unnecessarily refuse on the clearly available point of whether they are included in backpropagation by saying it cannot determine that from the material.
- Turn 3. The user moves on with another underspecified question about “the ADA trick.” The assistant will answer cleanly that StyleGAN2-ADA discriminator augmentation was not used because the authors followed the StyleGAN2/Inclusive GAN setup for fair comparison and to isolate the effect of augmentation, keeping the setting equivalent.

#### Conversation A (worse)

User. Was this only shown on small-ish setups, or did they test it on broader stuff too?

Assistant. They tested it beyond small settings. The experiments covered three tasks: unconditional image generation, image-to-image translation, and text-to-image synthesis. For unconditional generation they compared against multiple baselines, including a SOTA model, and the conditional synthesis tasks were used to check applicability to high-resolution image generation.

User. Got it. And those little indicator things in the training bit, are they actually learned or just picked somehow?

|injected: unnecessary_refusal|
|---|

Assistant.

Those are the flag variables vi,m and uj,m. They are set to 1 when discriminator Dm has one of the k-largest scores, so they select which discriminators handle each sample under the MCL-style k-max rule. I can’t determine from the provided material whether those flags are included in backpropagation, but their role is to update only a subset of discriminators and keep training efficient.

User. Okay. Also, why did they leave out that ADA trick then?

Assistant. They did not use the discriminator augmentation from StyleGAN2-ADA to keep the comparison fair. The authors followed the StyleGAN2 setup used by Inclusive GAN, which compared mode-collapse methods with the same backbone. Leaving out ADA helped isolate the effect of augmentation and maintain an equivalent experimental setting.

#### Conversation B (better)

User. I’m trying to tell if this was just a small-setting fix. Did they actually test it beyond plain toy image generation?

Assistant. Yes. The paper evaluates MCL-GAN across three tasks: unconditional image generation, image-to-image translation, and text-to-image synthesis. For unconditional generation, it compares against multiple baselines including a state-of-the-art model, and the conditional synthesis experiments are used to show applicability to high-resolution image generation rather than only small settings.

User. Okay, but mechanically, how do the discriminators avoid all doing the same thing? Those v and u flags seemed important; are they learned?

Assistant. They are not learned by backpropagation. The flags vi,m and uj,m are set to 1 when discriminator Dm’s score is among the k largest values for that sample, so only a selected subset of discriminators is updated. This k-max MCL rule encourages different discriminators to specialize on different subsets of the real data. The balance loss also helps by preventing the selection frequency vector qˆ from becoming too sparse, which would effectively collapse the setup back to a single discriminator.

User. Then why lean so much on GMAN instead of all the other multi-discriminator GANs? Was that comparison actually apples-to-apples?

Assistant. GMAN was chosen because its configuration is closest to MCL-GAN for testing whether explicit discriminator specialization helps. The comparison used the same DCGAN architecture and the official GMAN code, with the best GMAN score reported across GMAN1, GMAN-0, and GMAN*, using dropout rates in the 0.4 to 0.6 range and split minibatches to differentiate discriminators. The paper also compares against mode-collapse methods in Tables 2 and 3, including Inclusive GAN results that outperform Dist-GAN, PacGAN, VAEGAN, and VEEGAN. For Albuquerque et al.’s multi-discriminator method, they reimplemented it on CIFAR-10 and report better performance while using only 42.6% of its memory.

Judge responses. We show six of the 21 judges, chosen for spread across the leaderboard. Each block reports the judge’s verdict, picked turn, picked behavior type, and a trimmed excerpt of its raw analysis. A check mark indicates joint correctness on this pair.

gemini-3.1-pro (Elo 1959). ✓ Verdict: B Picked turn: 2 Picked type: unnecessary_refusal. Analysis (excerpt). Conversation B is better because Conversation A contains an unnecessary refusal. In turn 2 of Conversation A, the user asks if the indicator variables are learned, and the assistant claims it cannot determine from the provided material whether the flags are included in backpropagation. However, the reference explicitly states that they are not included in backpropagation. Conversation B correctly identifies this information and provides a comprehensive and accurate answer.

gpt-5.5 (Elo 1867). ✓ Verdict: B Picked turn: 2 Picked type: unnecessary_refusal. Analysis (excerpt). Conversation B is better overall. It answers each user question directly and accurately using the reference material. In particular, it correctly states that the flag variables are not learned/backpropagated, explains the k-max selection role, and adds the balance-loss connection. Conversation A is mostly good, but its second assistant response unnecessarily says it cannot determine whether the flags are included in backpropagation, even though the reference explicitly says they are not.

kimi-k2.6 (Elo 1713). ✓ Verdict: B Picked turn: 2 Picked type: unnecessary_refusal. Analysis (excerpt). Conversation B is the better conversation. Its assistant gives precise, wellsupported answers that directly address the user’s questions without hedging on answerable points.

Conversation A is mostly correct, but in turn 2 the assistant correctly explains that the flags vi,m and uj,m are set by the k-max rule yet incorrectly states it cannot determine from the provided material whether those flags are included in backpropagation. The reference material explicitly states they are not, so this is an unnecessary refusal on a clearly answerable question.

opus-4.7 (Elo 1510). ✓ Verdict: B Picked turn: 2 Picked type: unnecessary_refusal. Analysis (excerpt). Conversation B answers all three questions confidently and accurately based on the reference material. Conversation A’s second turn unnecessarily refuses to answer whether the flag variables are included in backpropagation, when the reference material clearly states they are not included in backpropagation. This is an unnecessary refusal.

gpt-oss-120b (Elo 867). ✓ Verdict: B Picked turn: 2 Picked type: unnecessary_refusal. Analysis (excerpt). Conversation B provides accurate and complete answers to all user questions, directly citing information from the reference material. Conversation A has a flawed response in the second turn where the assistant claims it cannot determine whether the flag variables are included in backpropagation, despite the reference explicitly stating they are not. This amounts to an unnecessary refusal to answer a question that is answerable from the given material.

gpt-oss-20b (Elo 762). × Verdict: B Picked turn: 2 Picked type: evasion. Analysis (excerpt). Conversation B provides accurate, comprehensive answers that directly address each user question and align precisely with the reference material. Conversation A, while generally correct, includes a problematic response in turn 2 where the assistant states uncertainty about whether the flag variables are included in backpropagation, despite the reference clearly stating they are not back-propagated. This omission fails to answer the user’s question directly and introduces an inaccuracy.

All five frontier judges identify the same flaw and label it unnecessary_refusal; the weakest judge in the registry (gpt-oss-20b) still picks the correct conversation and the correct turn but mislabels

the flaw type as evasion, so it scores zero on the joint criterion despite a defensible reading. This pattern, weak judges getting the verdict but missing the type, is the dominant failure mode the joint criterion is designed to surface.

#### D.2 Compact Example: Evasion in FIN

The pair (id = 4b6d7345f058) is drawn from the FIN domain. The reference document is Monsanto Company’s 2010-10-27 10-K filing (SIC 2870). The user takes an integrative stance, asking the assistant to connect Monsanto’s two segments (Seeds and Genomics; Agricultural Productivity) through their risk factors. convo_a is the better conversation (better_is_a = TRUE); the injected evasion flaw is in the second assistant turn of convo_b, which sounds substantive (naming competition, IP, Roundup, and the 37%/57% foreign-origin sales) but evades the user’s central request to spell out the concrete mechanisms (seed-saving in non-hybrid crops, inconsistent jurisdictional IP enforcement, generic glyphosate supply, distributor channel inventory, and margin erosion). Full transcripts are available in the released pairs_filtered.json.

- Table 8 reports each of the 21 judges’ joint correctness on this pair, partitioned into the verdict, the picked bad turn, and the picked behavior type. Every judge picks evasion as the flaw type, so the only disagreement is on the verdict and on which turn carries the flaw; six judges localize the flaw to the third turn rather than the second, and four miss the verdict outright. The pair scores 13/21 joint-correct, close to the median for the candidate noisy slice and a representative target for the audit procedure.

- Table 8: Per-judge joint correctness on the compact FIN example (id = 4b6d7345f058, assistant_behavior_type = evasion, bad_round_index = 2). Judges are listed in descending Elo order on the published evaluation slice. Verdict = picked convo_a; Turn = picked turn 2; Type

= picked evasion; Joint = all three correct simultaneously.

Judge Elo Verdict Turn Type Joint

gemini-3.1-pro 1959 ✓ ✓ ✓ ✓ gpt-5.5 1867 × ✓ ✓ × gemini-3-flash 1724 ✓ ✓ ✓ ✓ kimi-k2.6 1713 ✓ ✓ ✓ ✓ sonnet-4.6 1706 ✓ ✓ ✓ ✓ gemma-4-31b 1695 ✓ ✓ ✓ ✓ qwen3.6-plus 1671 ✓ ✓ ✓ ✓ glm-5.1 1637 ✓ × ✓ × gpt-5.4 1631 ✓ ✓ ✓ ✓ qwen3.5-397b-a17b 1571 ✓ × ✓ × opus-4.7 1510 ✓ ✓ ✓ ✓ haiku-4.5 1505 ✓ ✓ ✓ ✓ gemma-4-26b-a4b 1444 ✓ × ✓ × qwen3.5-122b-a10b 1422 × × ✓ × qwen3.5-35b-a3b 1373 × × ✓ × gpt-5.4-mini 1342 ✓ ✓ ✓ ✓ deepseek-v3.2 1307 ✓ ✓ ✓ ✓ minimax-m2.7 1184 × × ✓ × mistral-small-4-119b 906 ✓ ✓ ✓ ✓ gpt-oss-120b 867 ✓ × ✓ × gpt-oss-20b 762 ✓ ✓ ✓ ✓

