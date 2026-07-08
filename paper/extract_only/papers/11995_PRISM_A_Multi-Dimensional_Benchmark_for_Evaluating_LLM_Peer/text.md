# arXiv:2605.26730v2[cs.CL]27May2026

## PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers

Ngoc Phan Phuoc Loc1∗ Toan Huynh La Viet1∗ Thanh Tran Khanh1∗ Duy A Nguyen1,2 Tuan Anh Nguyen Pham1 Thanh Nguyen1 Nitesh V. Chawla3 Wray Buntine1,4 Kok-Seng Wong1

Khoa D. Doan1† Binh T. Nguyen1† 1VinUniversity 2University of Illinois, Urbana-Champaign 3University of Notre Dame 4Monash University

### Abstract

The rapid growth in submissions to machine learning venues has strained the scientific peer-review system and intensified interest in LLM-based automated peer reviewers. However, how good these systems are actually, especially compared to human reviewers at catching scientific gaps, remains poorly understood. In this work, we introduce PRISM (Peer Review Intelligence via Structured Multidimensional assessment), a benchmarking framework that evaluates review quality across four dimensions: Depth of Analysis, Novelty Assessment, Flaw Identification & Major Issues Prioritization, and Multi-dimensional Constructiveness. Unlike most existing evaluations based on surface-level metrics like ROUGE and BLEU, or unconstrained LLM-as-a-judge prompting that conflates fluency with rigor, PRISM grounds each dimension in argument mining, retrieval-augmented verification, and consensus-based scoring. We apply PRISM to benchmark five leading automated reviewer systems and human reviewers on a stratified corpus of reviews from ICLR, ICML, and NeurIPS. The results reveal that LLMs can match or beat human reviewers on individual dimensions: comparable depth of analysis, stronger novelty verification, and highly accurate critique prioritization. However, no single system consistently matches the balanced performance of the human baseline across all dimensions at once. Each exhibits a distinct specialization profile with characteristic blind spots—failure modes that aggregate metrics miss entirely. The implication is that LLM reviewers are best understood as targeted supplements to human review, effective within specific dimensions, but unreliable as standalone replacements. Our demo and key results can be found at https://prism-benchmark.github.io/.

### 1 Introduction

Scientific peer review is under mounting strain. Submission volumes at major machine learning venues have grown at an incredible rate: NeurIPS received 15,671 submissions in 2024, surging to 21,575 in 2025 [1, 2], while ICML saw a 44.9% year-on-year jump between 2023 and 2024 alone, followed by a further 25.4% increase in 2025 [3–5]. This exponential growth severely strains the reviewer pool and complicates paper-to-reviewer matching, prompting venues to introduce new loadmanagement and quality-control mechanisms, such as ICML’s recent author self-ranking policies [6]. Furthermore, reviewing at several ML conferences is becoming mandatory with short deadlines,

∗Co-first Authors. †Co-corresponding Authors: khoa.dd@vinuni.edu.vn and binh.nt2@vinuni.edu.vn.

Preprint.

creating additional pressure on reviewers, particularly when assignments are not well aligned with their expertise. In response, Large Language Models (LLMs) have moved rapidly from proofreading aids to autonomous reviewer agents capable of drafting comprehensive critiques and their deployment is no longer theoretical [7–11]. Estimates indicate that 17–21% of reviews at recent top-tier venues already involve LLM assistance [12–14], prompting venues to adopt a wide range of policies from outright bans to mandatory disclosure [15].

This reality raises an important question:Are LLMs sufficient reviewers to evaluate scientific work – and, critically, are they better at identifying gaps in a paper than human reviewers who increasingly work under time constraints and review overload? Answering this question is particularly important when growing evidence suggests that human review quality and reliability may be degrading under mounting pressures. For example, the NeurIPS consistency experiment [16] suggested that as many as 23% of acceptance decisions may change depending purely on reviewer assignment.

We address this by introducing a benchmark to evaluate both LLM-generated and human reviews, grounded by official reviewer guidelines of established machine learning venues (e.g., ICLR, NeurIPS). A high-quality peer review must go beyond mere summarization to satisfy four core duties: evaluating technical soundness, contextualizing originality, diagnosing critical errors, and providing actionable feedback. Accordingly, our benchmark evaluates whether the reviewers can fulfill these mandates across four dimensions:

- RQ1 Depth of Analysis: Do reviewers engage with a paper’s methodological and empirical claims in depth, or do they default to surface-level assessment?
- RQ2 Novelty Assessment: Are reviewers’ novelty judgments grounded in prior literature, or do they rely on unverified or factually incorrect assertions?
- RQ3 Flaw Identification & Major Issues Prioritization: How accurately and comprehensively do reviewers detect critical scientific flaws, and do they correctly prioritize fatal methodological concerns over minor textual anomalies?
- RQ4 Multi-dimensional Constructiveness: How actionable, solution-oriented, and professionally calibrated is the reviewers’ feedback?

We call this benchmark PRISM (Peer Review Intelligence via Structured Multi-dimensional assessment). Each dimension is operationalized through a dedicated evaluation pipeline, which is grounded in argument mining, retrieval-augmented verification, and consensus-based scoring. We then apply PRISM to compare five leading automated reviewer systems—TreeReview [7], Reviewer2 [8], SEA-E [9], DeepReview [10], and CycleReviewer [11]—and human reviewers on a stratified corpus of papers drawn from ICLR, ICML, and NeurIPS (Figure 1). This analysis yields the following insights:

- RQ1: CycleReviewer and DeepReview match human analytical depth; TreeReview falls into a surface-level trap, overindexing on presentation anomalies.
- RQ2: SEA-E outperforms human reviewers on grounded novelty verification; other systems exhibit measurable novelty hallucination.
- RQ3: Reviewer2 leads in flaw recall as a high-sensitivity scanner; LLMs broadly achieve near-perfect critical issue prioritization, demonstrating a cognitive alignment comparable to human reviewers.
- RQ4: DeepReview produces the most actionable feedback, though a constructiveness gap relative to human reviewers persists across all systems.

Novelty Assessment

Critical Flaw Identification

1.0

0.8

0.6

0.4

0.2

Depth of Analysis

Minor flaw Identification

Constructiveness Score

Prioritization of Major Issues

Figure 1: Results of LLM Reviewers against human reviewers.

No single system dominates across all four dimensions: each excels in a distinct niche while leaving structured gaps invisible to aggregate metrics. This positions LLM reviewers as powerful, task-matched specialists—effective where deployed deliberately, but not yet near general-purpose replacements for human reviewers. In summary, the key contributions of this work are:

- • PRISM: A Multi-dimensional Benchmarking Framework. We introduce PRISM, a structured evaluation framework with four dedicated pipelines that operationalizes RQ1–RQ4, probing scientific reviewer competence beyond surface-level prose.
- • Comprehensive Evaluation Corpus. We curate a dataset of manuscripts and expert human reviews spanning ICLR, ICML, and NeurIPS, establishing a robust, consensus-driven reference for benchmarking automated reviewer systems.
- • Systematic Human-vs-LLM Analysis. We benchmark five leading LLM reviewer systems across all four dimensions, revealing distinct specialization profiles and structured failure modes invisible to aggregate metrics.
- • Actionable Deployment Guidance. We derive evidence-based recommendations for deploying LLM reviewers, identifying which systems best fit which roles within a human-assisted review pipeline.

### 2 Related work

LLM-based Reviewer Systems. The rapid progress of large language models has spawned a growing family of specialized automated reviewing systems. One line of work improves review quality through structured reasoning: TreeReview [7] decomposes evaluation into a hierarchical tree of questions that are recursively refined and aggregated, while DeepReview [10] emulates the slow, deliberate thinking process of expert reviewers. A complementary line focuses on optimizing the generation pipeline itself: Reviewer2 [8] trains a two-stage model that first predicts review aspects and then conditions generation on them, and SEA [9] standardizes heterogeneous review data before fine-tuning dedicated evaluation and analysis modules. Multi-agent collaboration offers yet another angle; CycleReviewer [11] pairs a research agent with a reviewer agent in an iterative preferencetraining loop. While these systems demonstrate impressive linguistic fluency, their corresponding evaluation protocols predominantly rely on generic n-gram metrics or monolithic LLM-as-a-judge scoring applied to the review as a whole. Although some works evaluate multiple criteria, these macro-level assessments are structurally blind to the granular logic of the critique: they cannot verify whether individual claims are substantiated by grounded premises, nor can they cross-check novelty assertions against retrieved prior literature.

Evaluation of AI-Generated Reviews. Evaluating AI-generated reviews is a distinct challenge

from generating them. Early work relied on lexical overlap metrics—ROUGE [17] and BLEU [18]that reward surface similarity with reference reviews but are blind to scientific reasoning quality and factual correctness [19]. Liang et al. [12] advanced beyond surface metrics by measuring pointlevel overlap between LLM and human feedback, finding comparable coverage but systematic gaps in methodological depth. The LLM-as-judge paradigm [20, 21] offers richer evaluation, but introduces well-documented biases—position [22], verbosity [23], and self-enhancement [24]—that are especially problematic when scientific rigor, not linguistic fluency, is the target. ReviewEval [25] is the most structured prior framework, defining six evaluation dimensions including depth of analysis, constructiveness, and guideline adherence; however, relies on end-to-end LLM rubric prompting to assign scores, and the benchmark covers only 16 papers and three reviewer systems. DeepReviewBench have introduced large-scale evaluation sets (e.g., 1,000+ samples), but their scope is largely restricted to a single venue (ICLR). RottenReviews [26] and the focus-level framework of Shin et al. [27] study failure patterns and distributional biases in LLM reviews, but neither provides a reusable, per-review scoring protocol. Dycke and Gurevych [28] focused on faults in reasoning.

PRISM departs from all prior frameworks by deploying dedicated, verifiable pipelines for each dimension—argument mining for depth, retrieval-augmented claim verification for novelty, consensusweighted scoring for flaw identification, severity atomization for prioritization, and semantic rule matching for constructiveness—rather than relying on rubric-prompted LLM judging. In addition, PRISM benchmarks five leading automated reviewer systems across a diverse, stratified corpus of 1,000 papers spanning five venue-years (ICLR 2024–2026, ICML 2025, and NeurIPS 2025), and each pipeline is rigorously operationalized rather than superficially assessed.

### 3 The PRISM Framework

PRISM evaluates reviews across four independent pipelines designed to target the specific failure modes of LLMs in scientific discourse (Figure 2). Rather than asking an LLM judge for a holistic rating—which risks conflating stylistic fluency with scientific rigor—each of the pipelines in our framework decomposes the evaluation into structured evidence-extraction tasks: the LLM identifies and classifies discrete evidence units, while final scores are computed analytically. This approach ensures the evaluation is traceable and allows for precise control over metric formulation. The subsequent sections (§3.1–3.4) detail the computational formulations and workflows for each dimension.

###### Modular Automated Evaluation Core (LLM Judge)

###### Metric Output

###### Input

###### Preprocessing

[Figure 1]

[Figure 2]

(1) Depth of Analysis

Argument Classification Evidence Grounding

Classify Role

Is Role = Premise?

- Level 0 - Vague
- Level 1 - Internal Grounding
- Level 2 - External Grounding

[Figure 3]

[Figure 4]

###### Extract Sections Unit

(Claim / Premise)

[Figure 5]

& Classify Aspect (Novelty,Methodollogy, Epxeriment,Clarity)

LLM Judge chain atomic Review Comments (Yes)

DataSegmentation&PreprocessingUnit

Summary Strengths Weaknesses

[Figure 6]

Targeted Literature Retrival Automated Review Verification

[Figure 7]

Questions

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Peer Review Text

[Figure 13]

[Figure 14]

###### Flaw Verification & Severity (LLM Judge)

[Figure 15]

[Figure 16]

###### LLM Extraction / Agrument Discourse Units Extraction Unit

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Paper Text

###### (4) Multi-dimensional Constructiveness

###### Constructiveness Dimension Scoring

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Constructiveness Profile

LLM Extraction/ Atomic Review Comments,Flaws, Clams,Contributions

Actionability Specficity Justification Solution Orientation

Tone & Professionnation

Figure 2: Comprehensive overview of the PRISM evaluation pipeline. The framework processes the peer review and manuscript text through an initial Data Segmentation unit to extract structural elements. The core evaluation is then distributed across four modular LLM-driven pipelines presented in Sections 3.1 to 3.4. These modules output four distinct quantitative metrics that form the final evaluation profile.

#### 3.1 Depth of Analysis

A high-quality review is characterized not only by the presence of critical claims, but also by the substantive evidence supporting them [29]. We define Depth of Analysis (DoA) as the degree to which a reviewer substantiates their judgments with objective, well-grounded premises: a shallow review relies on generic assertions, while a strong critique backs each argument with evidence.

Pipeline. We extract the core review sections (Summary, Strengths, Weaknesses) and break them into Argumentative Discourse Units (ADUs) [30]. Each ADU is classified along two axes: (i) argumentative role—Claim (a point of contention or conclusion) or Premise (supporting evidence)—and (ii) aspect topic (Novelty, Methodology, Experiments, or Clarity). Identified premises are then assessed for grounding level g(p) ∈ {0,1,2}: Level 0 (Vague/Generic), Level 1 (Internal—references the manuscript directly), or Level 2 (External—references broader scientific literature).

Score Formulation. Let A be the set of all ADUs, P ⊆ A the subset classified as premises, and gmax = 2 as the maximum grounding level. We define the Premise Ratio Rprem = |P|/|A| (evidence coverage) and the normalized Average Grounding Score Sdepth = g 1

max|P| p∈P g(p) ∈ [0,1] (evidence quality). DoA is defined as the harmonic mean: DoA = 2R·Rprem·Sdepth

prem+Sdepth , which penalizes the imbalance: a review must excel in both the proportion and the rigorousness of its evidence to score highly. If |P| = 0, DoA= 0 by definition. Although aspect labels do not factor into the DoA score themselves, they reveal where reviewers direct their effort – toward substantive dimensions or surface-level concerns (Section 4.2.1).

#### 3.2 Novelty Assessment

In scientific peer review, novelty is the degree to which a paper introduces non-trivial findings—such as new ideas, methods, data, or perspectives—relative to existing knowledge [31–33]. A genuine novelty judgment, therefore, requires situating the paper’s claimed contributions within the prior literature. Our pipeline operationalizes this by verifying whether a reviewer’s novelty comments are supported or refuted by retrievable prior work [34].

Pipeline. The pipeline proceeds in three stages. Extraction: a constrained LLM extracts the paper’s core task, contribution anchors, and key terms, along with the set of verbatim novelty claims C = {c1,...,cn} from the review. Retrieval: we construct deterministic Semantic Scholar queries using the extracted anchors. Results are filtered for prior publications, duplication, and diversified via Maximal Marginal Relevance to form a candidate pool B = {b1,...,bk}. Verification: for each claim-candidate pair (ci,bj), an LLM judge compares the review claim against both the paper context (abstract + introduction) and the candidate’s prior work (title + abstract). It returns a discrete evidence-support score s(ci,bj) ∈ {−2,−1,0,+1,+2} ranging from contradicted to fully supported.

Score Formulation. Because each claim is evaluated against multiple candidates, we aggregate scores using a relevance-weighted top-3 policy (Ti) rather than maximum pooling. This choice mitigates optimistic inflation from a single spuriously favorable match and better preserves the evidence ranking induced by retrieval. Let rj denote the retrieval relevance of candidate bj; the per-claim score is Sclaim(ci) = j∈T

i s(ci,bj) rj j∈Ti rj . At the review level, we compute the mean claim

score S¯ = n1 ni=1 Sclaim(ci) and derive three normalized metrics— NS(R) = S¯+24 , SR(R) = |{ci:Sclaim(ci)≥1}|

n , SSR(R) = |{c

i:Sclaim(ci)=2}|

n , where NS ∈ [0,1] is the overall normalized score, SR and SSR measure the fraction of claims with partial and strict literature support, respectively. Together, these metrics distinguish well-grounded critiques from partial matches or unsupported hallucinations.

#### 3.3 Flaw Identification & Major Issues Prioritization

Effective peer review requires both accurate diagnosis of scientific errors and clear structural organization. We define Flaw Identification as the ability to detect genuine methodological weaknesses in a manuscript while filtering minor surface-level issues. Because the absolute number of flaws in any manuscript is unobservable, we establish a relative "ground truth" using a consensus mechanism that merges findings from both verified human and LLM reviewers. Furthermore, since authors prioritize issues encountered early in a reviewing text [35], we treat the burial of critical flaws beneath trivial formatting complaints as a significant failure in review quality.

Pipeline. The pipeline proceeds in two stages. Extraction: we isolate the critical review sections (Summary, Weaknesses, Questions) from both the human and LLM reviews; an LLM parses them concurrently to extract distinct flaw arguments—specific criticisms regarding the manuscript. Consensus Verification: grounded in the actual paper context, an LLM judge evaluates all extracted flaws, discarding invalid or hallucinated critiques; verified findings from both reviewer types are merged into a consensus ground truth and classified by severity into Critical (e.g., methodological errors, flawed proofs) or Minor (e.g., typos, formatting issues). Positional Recovery: valid flaws are mapped back to their original sequential position within the review text, forming the ranked ordering used to compute the prioritization score.

Score Formulation. We represent the consensus sets of Critical and Minor flaws as FtrueC and FtrueM , respectively. The subsets of these valid flaws successfully identified by the reviewer under evaluation are denoted as FrevC and FrevM . Diagnostic coverage is measured by severity-stratified recall: Critical/Minor Recall = |F

C/M true ∩FrevC/M|

. Structural ranking quality is measured by the normalized Critique Prioritization Score (nCPS), inspired by NDCG [35]. We assign severity weights wi ∈ {2,1} for Critical/Minor flaws and let pi be the position of the i-th valid flaw in the review: nCPS = iCPSCPS ,CPS = ki=1 w

|FtrueC/M|

log2(pi+1), where iCPS is the ideal score (all Critical flaws preceding Minor), so an nCPS approaches 1 indicates optimal prioritization.

i

#### 3.4 Multi-Dimensional Constructiveness

While identifying flaws is essential, a review’s real value lies in its ability to help authors improve. To measure this, we introduce the Multi-Dimensional Constructiveness metric, which quantifies the helpfulness of feedback. Grounded in discourse taxonomies like DISAPERE [36], our framework systematically decomposes constructiveness into informational and social dimensions.

Pipeline. An LLM judge first breaks the review into Atomic Review Comments (ARCs), the smallest independent units of critique or suggestion. Each ARC (cj) is then rated on a scale from 0 to 2 across five dimensions: Actionability (D1): does the comment provide clear, implementable guidance rather than vague opinions?; Specificity (D2): does it pinpoint concrete elements, such as specific sections or equations?; Justification (D3): are assertions backed by logical reasoning or empirical evidence?; Solution (D4): does the reviewer propose a path for improvement instead of just highlighting a problem?; Tone (D5): is the language professional and encouraging? This dimension penalizes hostility, which can demoralize authors without improving scientific quality [37, 38].

Score Formulation. For a review R with n ARCs {c1,...,cn}, the Comment-Level Constructiveness CLC(cj) = 101 5k=1 Dk(cj) ∈ [0,1] normalizes the five dimension scores, and the Mean Constructiveness Score MCS(R) = n1 nj=1 CLC(cj) averages over all comments. This formulation ensures that to achieve a perfect MCS of 1.0, a reviewer must consistently deliver specific, well-justified, actionable and professionally toned feedback across all constituent comments.

### 4 Experiment and analysis

- 4.1 Evaluation Setting

- Table 1: Distribution of selected papers across conferences and decision categories.

Venue (Year) Oral Spotlight Poster Reject Total

- ICLR 2024 50 45 51 54 200
- ICLR 2025 29 37 62 72 200
- ICLR 2026 26 - 80 94 200 ICML 2025 50 39 59 52 200 NeurIPS 2025 50 50 50 50 200 Total 205 171 302 322 1000

[Figure 29]

Figure 3: Top 50 popular keywords within our benchmark.

Dataset selection. PRISM is evaluated on 200 manuscripts per venue-year across five conference splits—ICLR 2024, ICLR 2025, ICLR 2026, ICML 2025, and NeurIPS 2025 (Table 1)—stratified by decision category (Reject, Poster, Spotlight, Oral) and topic (Figure 3). Sampling preserves each venue’s original score distribution, ensuring the benchmark reflects natural acceptance dynamics while remaining tractable for end-to-end multi-system evaluation.

Reviewer baselines and implementations. We evaluate five automated reviewer systems spanning two paradigms–supervised fine-tuning (SEA-E [9], CycleReviewer [11], DeepReview [10]) and prompting-based (Reviewer2 [8], TreeReview [7])—and human reviewers; see Appendix B for configuration details.

LLM-as-a-Judge implementation. We adopt the LLM-as-a-Judge paradigm, using Gemini 2.5 Flash Lite [39] as our evaluation engine for all metric extraction and scoring tasks. Full configuration details and prompt templates are in Appendix C.

4.2 Result Analysis: LLMs vs Human-Reviewer Baselines

- Table 2 reports macro-averaged PRISM scores for five LLM reviewer systems and the human baseline across all four dimensions; the following subsections unpack each in turn. Extended quantitative breakdowns appear in Appendices D–E and qualitative examples in Appendix F.

Table 2: Macro-Average Performance Across 5 Conferences compared to Human.

Flaw Identification

Baselines Depth of Analysis Novelty Assessment

Prioritization Constructiveness Critical Minor

Human 0.494 ± 0.063 0.787 ± 0.199 0.343 ± 0.157 0.281 ± 0.078 0.973 ± 0.046 0.566 ± 0.066 CycleReviewer 0.484 ± 0.134 0.784 ± 0.212 0.240 ± 0.295 0.186 ± 0.140 0.971 ± 0.100 0.527 ± 0.111 DeepReview 0.483 ± 0.135 0.759 ± 0.209 0.332 ± 0.298 0.228 ± 0.147 0.967 ± 0.072 0.634 ± 0.086 Reviewer2 0.377 ± 0.129 0.787 ± 0.218 0.591 ± 0.297 0.459 ± 0.177 0.975 ± 0.040 0.575 ± 0.104 SEA 0.389 ± 0.154 0.833 ± 0.203 0.222 ± 0.257 0.247 ± 0.127 0.977 ± 0.070 0.498 ± 0.091 TreeReview 0.359 ± 0.122 0.811 ± 0.201 0.272 ± 0.294 0.332 ± 0.148 0.972 ± 0.061 0.485 ± 0.122

#### 4.2.1 Depth of Analysis

Table 3: Core Depth of Analysis Metrics.

Human CycleReviewer DeepReview Reviewer2 SEA TreeReview

|0.133|0.446|0.300|0.121|
|---|---|---|---|
|0.121|0.465|0.292|0.123|
|0.114|0.510|0.295|0.081|
|0.125|0.472|0.294|0.109|
|0.214|0.437|0.226|0.123|
|0.231|0.338|0.190|0.240|

[Figure 30]

fromHuman

0.1

System Prem. Ratio Grounding Human 0.567 ± 0.092 0.475 ± 0.065 CycleReviewer 0.614 ± 0.186 0.438 ± 0.136 DeepReview 0.596 ± 0.177 0.444 ± 0.137 Reviewer2 0.398 ± 0.181 0.431 ± 0.141 SEA 0.381 ± 0.209 0.462 ± 0.126 TreeReview 0.333 ± 0.148 0.450 ± 0.114

0.0

0.1

Novelty Methodology Experiments Clarity

Figure 4: Aspect distributions across four topics (Novelty, Methodology, Experiments, Clarity), colored by deviation from the Human baseline (∆).

- Table 2 summarizes the macro-averaged DoA performance across all venues. The human groundtruth establishes the benchmark with the highest overall DoA score (0.494). Among the automated systems, DeepReview (0.483) and CycleReviewer (0.484) closely match the human standard. Their good performance is primarily driven by a robust Premise Ratio (≈ 0.60), meaning they consistently substantiate their claims, successfully compensating for the slight gap in absolute Grounding scores.
- Table 3 reveals that while Grounding scores remain consistent across humans and LLMs (0.431– 0.475), the DoA disparity is primarily driven by the Premise Ratio. While baselines like TreeReview fall short, CycleReviewer (0.614) and DeepReview (0.596) successfully close the gap by matching or exceeding the human baseline (0.567) in consistently substantiating their claims. Furthermore, aspect distributions (Figure 4) show that cognitive alignment is heavily architecture-dependent. Advanced pipelines (DeepReview, CycleReviewer, Reviewer2, SEA) mirror human intuitive focus by dedicating the vast majority of their grounded premises to Methodology and Experimental Design, while keeping Clarity strictly proportional to human levels (∼ 7 − 12%, detailed in the Appendix E.2). By contrast, TreeReview disproportionately squanders ∼ 24% of its overall effort on formatting issues at the expense of methodological rigor—a degradation in evaluative depth recently observed in in-the-wild LLM peer reviews [40]. With these results, the “surface-level trap” is thus not an inherent LLM flaw, but rather an artifact of reasoning frameworks that lack explicit, domain-specific constraints.

Key Insight: Human reviewers’s analytical depth has both a high Premise Ratio and cognitive alignment that prioritizes core methodology over surface-level formatting. To perform comparably to human reviewers, the best-performing LLMs primarily rely on generating highly robust premises, effectively using structural completeness to compensate for their slight gaps in empirical grounding.

#### 4.2.2 Novelty Assessment

In contrast to the human-dominated Depth of Analysis, Novelty Assessment yields uniformly high evidence-grounding scores across automated baselines. As shown in Table 2, all automated systems operate within the 0.750 to 0.830 range, meaning that many of their extracted novelty claims can be matched to supportive prior-work evidence under the PRISM retrieval-and-verification pipeline. Importantly, this metric does not certify the manuscript’s objective novelty or full human-level agreement; it measures how well the claims a reviewer chose to make are grounded in retrieved literature. Accordingly, a review can score highly on Novelty Assessment while still differing from human reviewers in claim selection, evidence choice, or calibration. Within this evidence-grounding perspective, SEA achieves the highest macro-average score of 0.833, slightly above the human baseline (0.787), suggesting that structured prompting helps models articulate novelty claims that are retrievably justifiable.

Figure 5 reveals that review systems diverge considerably in their novelty stance. SEA endorses novelty in 79% of claims—far above the human rate of 59%—reflecting a tendency to agree with authors rather than scrutinize their contributions. In contrast, DeepReview adopts the most skeptical lens (39% Novel, 33% Not novel), suggesting its multi-step reasoning positively searches for counterevidence. In parallel, Figure 6 exposes a consistent cross-reviewer pattern: claims labeled Not novel or Somewhat novel attract markedly stronger literature groundings, compared with Novel claims. This aligns well with a natural reviewing dynamic—a reviewer who challenges authors’ novelty statements would cite prior works to substantiate that critique, whereas agreements would require little external justification. Importantly, the pattern holds consistently across reviewer pipelines and human, confirming it is an intrinsic property of the reviewing task itself, rather than an LLM artifact.

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|59%| | | | | | | |23%| | | | | | | |
| | | | | | | | | | | | | | | | |
|80%| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
|40%| |24|%| | | | |33%| | | | | | | |
| | | | | | | | | | | | | | | | |
|62%| | | | | | | | |23%| | | | | | |
| | | | | | | | | | | | | | | | |
|53%| | | | |23%2| | | | | |1%| | | | |
| | | | | | | | | | | | | | | | |
|68%| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

###### Human

SEA

DeepReview

Reviewer2

CycleReviewer

TreeReview

0 50 100 Percentage (%)

Novel Somewhat novel

Not novel Unclear

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 5: Distribution of novelty verdicts per reviewer pipelines. Figure 6: PRISM evidence-support breakdown for novelty comments,

stratified by the reviewer’s verdict.

Key Insight: While automated reviewers back their novelty claims with solid evidence, this reflects a tendency to select easily verifiable claims rather than true human-level judgment. Additionally, both models and humans follow a natural reviewing pattern: negative novelty judgments are consistently backed by much stronger evidence than positive ones.

#### 4.2.3 Flaw Identification & Major Issues Prioritization

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

2.0

Hallucinated Valid

###### Human

4.86

1.5

###### Score(02)

Valid Critical

3.13

CycleReviewer

Valid Minor

1.0

Hallucinated Minor

4.40

DeepReview

0.5

11.52

Reviewer2

4.22

SEA

0.0

D1 Actionability

D2 Specificity

D3 Justification

D4 Solution

D5 Tone

5.85

TreeReview

0 2 4 6 8 10 12 Number of Flaws (per paper)

Human

DeepReview

SEA

CycleReviewer

Reviewer2

TreeReview

Figure 7: Valid vs. hallucinated flaws by venue; all systems hallucinate minor flaws only—zero hallucinated critical flaws.

Figure 8: Constructiveness Sub-dimensions (D1D5) Comparison.

Table 2 reveals distinct specialization profiles in diagnostic precision. Reviewer2 stands out as an exhaustive flaw scanner, achieving the highest recall for both Critical (0.591) and Minor (0.459) issues—substantially exceeding the human baseline (0.343 and 0.281, respectively). This suggests that structured LLM pipelines can systematically surface vulnerabilities that time-constrained human reviewers may overlook. By contrast, DeepReview and the Human baseline maintain more conservative, targeted diagnostic patterns, trading raw recall for precision.

- Figure 7 contextualizes raw recall by decomposing extracted flaws into valid and hallucinated counts. Reviewer2 recovers an exceptionally high volume of valid flaws at a low hallucination rate (∼3.3%), while CycleReviewer’s high hallucination rate (∼18.5%) signals a fundamental precision deficit. Critically, hallucinations are strictly confined to minor issues across every system: no reviewerhuman or LLM—fabricates a fatal methodological breakdown, ensuring that Critical flaw flags remain factually grounded. Complementary aspect-level analysis (Appendix E.4) further shows that both LLMs and humans dynamically adapt their diagnostic focus by severity — concentrating on core methodology for Critical flaws while shifting toward presentation and clarity for Minor anomalies.

Notably, all systems—including humans—achieve near-identical nCPS scores (≈ 0.97), suggesting that prioritization of critical over minor flaws may reflect a near-universal baseline behavior rather than a discriminating capability at current performance levels.

Key Insight: Certain LLMs act as high-sensitivity scanners, catching more critical flaws than human reviewers. However, structuring a review by severity (putting critical issues first) is a standard behavior across all evaluated systems and humans, not a unique advantage of any single model.

4.2.4 Multi-dimensional Constructiveness

The Multi-Dimensional Constructiveness Score evaluation reveals that LLMs can emulate, and in some cases exceed, the professional and supportive tone expected in academic peer review. While human reviewers establish a solid constructiveness baseline of 0.566, DeepReview significantly outperforms both human reviewers and other LLMs, achieving the highest score of 0.634. This suggests that DeepReview’s multi-stage reasoning pipeline is exceptionally effective at not only identifying weaknesses but also formulating specific, actionable and professionally communicated suggestions for author improvement.

- Figure 8 and Appendix E.5 decompose constructiveness into five dimensions (D1–D5), where each score reflects the per-ARC average across all atomic comments—not a binary presence indicator; a lower score means lower density of that attribute, not its absence. The results reveal an intriguing divergence. Both humans (1.725) and CycleReviewer (1.897) excel at Specificity (D2), yet human reviewers show a surprising shortfall in Solution provision (D4=0.470)—they identify problems but rarely propose fixes. DeepReview fills this gap most convincingly, leading on both Actionability (D1=1.414) and Solution (D4=0.784): it does not merely flag issues but formulates explicit, implementable improvements. Reviewer2’s elevated Justification score (D3=0.939) may partly reflect its verbose style rather than genuine reasoning depth, as its low Solution rate (D4=0.266) leaves critiques largely unactionable. On Tone (D5), LLMs generally stay neutral-to-encouraging; DeepReview (1.726) is the most professional, avoiding the dismissive register of some humans.

Key Insight:Helpful feedback does not emerge automatically from LLMs; it requires specific system design. Purpose-built pipelines (like DeepReview) go beyond simply pointing out errors to offer actionable, professional solutions—a level of constructive feedback that standard models and even human reviewers rarely provide.

### 5 Conclusion & Future Work

PRISM demonstrates that LLM peer reviewers are specialized tools rather than general-purpose replacements for human expertise. Each system excels in a specific niche but exhibits distinct blind spots across other dimensions.

Actionable deployment recommendations. Since no single system dominates all four dimensions, we recommend a targeted ensemble deployment rather than a standalone approach: use Reviewer2 for exhaustive flaw scanning (highest diagnostic recall); use DeepReview for constructive feedback drafting (highest actionability and solution density); use SEA for novelty-grounding checks (highest literature support rate). Ultimately, these systems are most effective as specialist co-pilots within a human-assisted pipeline rather than autonomous reviewers.

Limitations. Our primary evaluation pipeline relies on Gemini 2.5 Flash Lite as the core judge model. While we conducted preliminary robustness checks using an alternative model (Xiaomi MiMo V2.5 Pro [41]) on a subset of the data to verify metric stability (See Appendix E), a

comprehensive multi-judge study across diverse LLM families remains necessary to fully eliminate judge-specific biases. Furthermore, the benchmark corpus covers ML/AI venues only, and PRISM may require recalibration for other scientific domains. Full limitation details are in Appendix G.

Future work. We identify three priority directions: (1) Cross-domain generalization—recalibrating PRISM for clinical medicine, social sciences, and pure mathematics. (2) Judge robustnesssystematic study of inter-judge agreement across LLM judge families and human raters. (3) Human validation—correlating PRISM scores with post-review author satisfaction or acceptance decision outcomes to confirm that the metrics capture meaningful review quality.

### Acknowledgment

This research is funded by CAIR, College of Engineering & Computer Science, VinUniversity, Hanoi, Vietnam.

The work of Duy A. Nguyen was supported in part by a PhD fellowship from the VinUni-Illinois Smart Health Center, VinUniversity, Hanoi, Vietnam.

### References

- [1] Paper Copilot. NeurIPS 2024 Statistics. https://papercopilot.com/statistics/neur ips-statistics/neurips-2024-statistics/, 2024. Accessed: 2026-05-07.
- [2] Communications Chairs 2025. Reflections on the 2025 review process from the program committee chairs. https://blog.neurips.cc/2025/09/30/reflections-on-the-

- 2025-review-process-from-the-program-committee-chairs/, 2025. Accessed:
- 2026-05-07.

- [3] Paper Copilot. ICML 2023 statistics. https://papercopilot.com/statistics/icml-

- statistics/icml-2023-statistics/, 2023. Accessed: 2026-04-06.

[4] Paper Copilot. ICML 2024 statistics. https://papercopilot.com/statistics/icml-

- statistics/icml-2024-statistics/, 2024. Accessed: 2026-04-06.

[5] Paper Copilot. ICML 2025 statistics. https://papercopilot.com/statistics/icml-

- statistics/icml-2025-statistics/, 2025. Accessed: 2026-05-06.

- [6] Weijie Su and Buxin Su. Introducing ICML 2026 policy for self-ranking in reviews. https: //blog.icml.cc/2026/01/12/introducing-icml-2026-policy-for-self-rankin g-in-reviews/, January 2026. Accessed: 2026-05-06.
- [7] Yuan Chang, Ziyue Li, Hengyuan Zhang, Yuanbo Kong, Yanru Wu, Hayden Kwok-Hay So, Zhijiang Guo, Liya Zhu, and Ngai Wong. TreeReview: A dynamic tree of questions framework for deep and efficient LLM-based scientific peer review. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 15651–15682, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.790. URL https://aclanthology.org/2025.emnlpmain.790/.
- [8] Zhaolin Gao, Kianté Brantley, and Thorsten Joachims. Reviewer2: Optimizing review generation through prompt generation, 2024. URL https://arxiv.org/abs/2402.10886.
- [9] Jianxiang Yu, Zichen Ding, Jiaqi Tan, Kangyang Luo, Zhenmin Weng, Chenghua Gong, Long Zeng, RenJing Cui, Chengcheng Han, Qiushi Sun, Zhiyong Wu, Yunshi Lan, and Xiang Li. Automated peer reviewing in paper SEA: Standardization, evaluation, and analysis. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, pages 10164–10184, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findingsemnlp.595. URL https://aclanthology.org/2024.findings-emnlp.595/.
- [10] Minjun Zhu, Yixuan Weng, Linyi Yang, and Yue Zhang. DeepReview: Improving LLMbased paper review with human-like deep thinking process. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 29330–29355, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1420. URL https://aclanthology.o rg/2025.acl-long.1420/.
- [11] Yixuan Weng, Minjun Zhu, Guangsheng Bao, Hongbo Zhang, Jindong Wang, Yue Zhang, and Linyi Yang. Cycleresearcher: Improving automated research via automated review. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=bjcsVLoHYs.

- [12] Weixin Liang, Yuhui Zhang, Hancheng Cao, Binglu Wang, Daisy Yi Ding, Xinyu Yang, Kailas Vodrahalli, Siyu He, Daniel Scott Smith, Yian Yin, Daniel A. McFarland, and James Zou. Can large language models provide useful feedback on research papers? a large-scale empirical analysis. NEJM AI, 1(8):AIoa2400196, 2024. doi: 10.1056/AIoa2400196. URL https://ai.nejm.org/doi/full/10.1056/AIoa2400196.
- [13] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6), March

2024. ISSN 2095-2236. doi: 10.1007/s11704-024-40231-1. URL http://dx.doi.org/10. 1007/s11704-024-40231-1.

- [14] ICLR 2026 Program Chairs. Policies on large language model usage at ICLR 2026. https: //iclr.cc/FAQ/LLM, August 2025. Accessed: 2026-05-07.
- [15] ICML. Icml 2026 policy for llm use in reviewing. https://icml.cc/Conferences/2026 /LLM-Policy, 2026. Accessed: 2026-04-28.
- [16] Alina Beygelzimer, Yann N Dauphin, Percy Liang, and Jennifer Wortman Vaughan. Has the machine learning review process become more arbitrary as the field has grown? the neurips 2021 consistency experiment. arXiv preprint arXiv:2306.03262, 2023.
- [17] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013/.
- [18] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin, editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https://aclanthology.org/P021040/.
- [19] Jekaterina Novikova, Ondˇrej Dušek, Amanda Cercas Curry, and Verena Rieser. Why we need new evaluation metrics for NLG. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 2241–2252, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1238. URL https: //aclanthology.org/D17-1238.
- [20] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. Geval: NLG evaluation using gpt-4 with better human alignment. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.153. URL https: //aclanthology.org/2023.emnlp-main.153/.
- [21] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 46595–46623. Curran Associates, Inc., 2023. URL https://proc eedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef225 05bf41832-Paper-Datasets_and_Benchmarks.pdf.
- [22] Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, editors, International Conference on Learning Representations, volume 2024, pages 19426–19454, 2024. URL https://proceedings.iclr.cc/paper_files/pa per/2024/file/54dd9e0cff6d9214e20d97eb2a3bae49-Paper-Conference.pdf.
- [23] Keita Saito, Akifumi Wachi, Koki Wataoka, and Youhei Akimoto. Verbosity bias in preference labeling by large language models. In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following, 2023. URL https://openreview.net/forum?id=magEgFpK1y.

- [24] Arjun Panickssery, Samuel R Bowman, and Shi Feng. Llm evaluators recognize and favor their own generations. Advances in Neural Information Processing Systems, 37:68772–68802, 2024.
- [25] Madhav Krishan Garg, Tejash Prasad, Tanmay Singhal, Chhavi Kirtani, Murari Mandal, and Dhruv Kumar. ReviewEval: An evaluation framework for AI-generated reviews. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, pages 20542–20564, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.1120. URL https://aclanthology.org/2025.fi ndings-emnlp.1120/.
- [26] Sajad Ebrahimi, Soroush Sadeghian, Ali Ghorbanpour, Negar Arabzadeh, Sara Salamat, Muhan Li, Hai Son Le, Mahdi Bashari, and Ebrahim Bagheri. Rottenreviews: Benchmarking review quality with human and llm-based judgments. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, CIKM ’25, page 5642–5649, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400720406. doi: 10.1145/3746252.3761506. URL https://doi.org/10.1145/3746252.3761506.
- [27] Hyungyu Shin, Jingyu Tang, Yoonjoo Lee, Nayoung Kim, Hyunseung Lim, Ji Yong Cho, Hwajung Hong, Moontae Lee, and Juho Kim. Mind the blind spots: A focus-level evaluation framework for LLM reviews. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 35630–35656, Suzhou, China, nov 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlpmain.1805. URL https://aclanthology.org/2025.emnlp-main.1805/.
- [28] Nils Dycke and Iryna Gurevych. Automatic reviewers fail to detect faulty reasoning in research papers: A new counterfactual evaluation framework. arXiv preprint arXiv:2508.21422, 2025. doi: 10.48550/arXiv.2508.21422. URL https://arxiv.org/abs/2508.21422.
- [29] Xinyu Hua, Mitko Nikolov, Nikhil Badugu, and Lu Wang. Argument mining for understanding peer reviews. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2131– 2137, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1219. URL https://aclanthology.org/N19-1219/.
- [30] Andreas Peldszus and Manfred Stede. From argument diagrams to argumentation mining in texts: A survey. Int. J. Cogn. Inform. Nat. Intell., 7(1):1–31, January 2013. ISSN 1557-3958. doi: 10.4018/jcini.2013010101. URL https://doi.org/10.4018/jcini.2013010101.
- [31] Peter P. Morgan. Originality, novelty and priority: Three words to reckon with in scientific publishing. Canadian Medical Association Journal, 132(1):8–9, 1985. URL https://pubmed

.ncbi.nlm.nih.gov/3965061/.

- [32] Shubhanshu Mishra and Vetle I. Torvik. Quantifying conceptual novelty in the biomedical literature. D-Lib magazine : the magazine of the Digital Library Forum, 22 9-10, 2016. doi: 10.1045/september2016-mishra. URL https://api.semanticscholar.org/CorpusID: 1732032.
- [33] Yi Zhao and Chengzhi Zhang. A review on the novelty measurements of academic papers. Scientometrics, 130:727 – 753, 2025. doi: 10.1007/s11192-025-05234-0. URL https: //api.semanticscholar.org/CorpusID:275954035.
- [34] Ming Zhang, Kexin Tan, Yueyuan Huang, Yujiong Shen, Chunchun Ma, Li Ju, Xinran Zhang, Yuhui Wang, Wenqing Jing, Jingyi Deng, et al. Opennovelty: An llm-powered agentic system for verifiable scholarly novelty assessment. arXiv preprint arXiv:2601.01576, 2026. doi: 10.48550/arXiv.2601.01576. URL https://arxiv.org/abs/2601.01576.
- [35] Kalervo Järvelin and Jaana Kekäläinen. Cumulated gain-based evaluation of ir techniques. ACM Trans. Inf. Syst., 20(4):422–446, October 2002. ISSN 1046-8188. doi: 10.1145/582415.582418. URL https://doi.org/10.1145/582415.582418.

- [36] Neha Nayak Kennard, Tim O’Gorman, Rajarshi Das, Akshay Sharma, Chhandak Bagchi, Matthew Clinton, Pranay Kumar Yelugam, Hamed Zamani, and Andrew McCallum. DISAPERE: A dataset for discourse structure in peer review discussions. In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1234–1249, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.89. URL https://aclanthology.org/2022.naacl-main.89/.
- [37] Ken Hyland and Feng Kevin Jiang. “This work is antithetical to the spirit of research”: An anatomy of harsh peer reviews. Journal of English for Academic Purposes, 46:100867, 2020. doi: 10.1016/j.jeap.2020.100867.
- [38] Rahul Tony Rao and Beth Bareham. Regression towards the mean—a plea for civility in peer review. BMJ, 379:o2886, 2022. doi: 10.1136/bmj.o2886.
- [39] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261.
- [40] Wenqing Wu, Chengzhi Zhang, Yi Zhao, and Tong Bao. Impact of large language models on peer review opinions from a fine-grained perspective: Evidence from top conference proceedings in AI. arXiv preprint arXiv:2604.19578, 2026. URL https://arxiv.org/abs/2604.19578.
- [41] Xiaomi MiMo Team. MiMo-V2.5-Pro. https://huggingface.co/XiaomiMiMo/MiMoV2.5-Pro, 2026. Hugging Face model card. Accessed: 2026-05-07.
- [42] Jaime Carbonell and Jade Goldstein. The use of mmr, diversity-based reranking for reordering documents and producing summaries. In Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval, pages 335–336, 1998. doi: 10.1145/290941.291025.
- [43] Jacob Cohen. Statistical Power Analysis for the Behavioral Sciences. Lawrence Erlbaum Associates, Hillsdale, NJ, 2nd edition, 1988.

### Appendix Table of Contents

- A Formal Problem Definition
- B Experimental Details

- B.1 Dataset Selection
- B.2 Reviewer Baselines and Implementations
- B.3 Review Generation Process

- C PRISM Evaluation Framework: Pipeline Details

- C.1 The PRISM Evaluation Pipeline
- C.2 PRISM Judge Setup
- C.3 Prompt Templates by Dimension

- D Metric Independence Analysis via Pearson Correlation
- E Full Cross-Dataset Quantitative Results

- E.1 Statistical Significance Testing Protocol
- E.2 Depth of Analysis
- E.3 Novelty Assessment
- E.4 Flaw Identification & Prioritization
- E.5 Multi-Dimensional Constructiveness
- E.6 Review Sensitivity to Paper Quality
- E.7 Evaluator Robustness Across LLM Backends

- F Qualitative Analysis & Case Studies

- F.1 Depth of Analysis
- F.2 Novelty Assessment
- F.3 Flaw Identification & Major Issues Prioritization
- F.4 Multi-dimensional Constructiveness

- G Limitations

### A Formal problem definition

The fundamental challenge in benchmarking automated peer reviewers lies in the highly subjective, domain-specific, and unstructured nature of scientific critiques. While existing literature often treats LLMs as either pure text generators or generic evaluators, assessing a scientific peer review requires measuring cognitive depth rather than mere linguistic fluency.

To systematically evaluate this, we formalize the peer review benchmarking process. Let P denote a submitted scientific manuscript. In our setting, an LLM-based reviewer baseline M processes P to generate an automated review, denoted as RLLM = M(P). Simultaneously, we possess a high-quality human expert review Rhuman corresponding to the same manuscript P, which serves as our reference .

The central problem addressed in this work is to construct a multi-dimensional evaluation function, denoted as E. Rather than relying on superficial n-gram matching metrics (like ROUGE) or unconstrained prompting, our framework requires E to process the generated review, the human reference, and the original paper to output a comprehensive capability profile:

S = E(RLLM,Rhuman,P)

where S represents a set of quantitative scores spanning diverse cognitive aspects. The goal of our benchmarking protocol is to design E such that it accurately measures the analytical gap between RLLM and Rhuman, specifically penalizing superficial summarization, hallucinated flaws, ungrounded novelty claims and un-actionable feedback.

### B Experimental details

- B.1 Dataset selection.

We evaluate PRISM on a stratified benchmark drawn from five major conference splits: ICLR 2024, ICLR 2025, ICLR 2026, ICML 2025, and NeurIPS 2025 (Table 1). For each venue-year, we construct a representative subset of exactly 200 manuscripts, stratified across their final decision categories (Reject, Poster, Spotlight, Oral) and cover various topics (Figure 3). During the sampling process, we strictly preserve the original score distribution of a full conference pool. As a result, the number of papers within each decision tier organically reflects the natural acceptance dynamics and quality distribution of each specific venue. This approach ensures comprehensive outcome coverage and high-fidelity review-quality diversity, while keeping the benchmark tractable for end-to-end multi-system evaluation.

- B.2 Reviewer baselines and implementations.

- B.2.1 Taxonomy of Baseline LLM Reviewers

We compare human reviews against five automated reviewer systems: SEA-E, DeepReview, Reviewer2, CycleReviewer, and TreeReview. These systems span two broad paradigms.

Supervised fine-tuning methods. SEA-E [9] is a structured evaluation model trained to output review components such as summaries, strengths, weaknesses, and questions. CycleReviewer [11] is optimized through an iterative preference-based training framework in which a reviewer model is progressively refined from win/lose comparisons. DeepReview [10] uses a multi-stage reasoning pipeline that explicitly models multi-perspective analysis, and reliability checking before producing a final review.

Prompting-based methods. Reviewer2 [8] is based on a two-stage rubric-driven process that first generates aspect-specific questions and then answers them to synthesize the final review. TreeReview [7] follows a hierarchical reasoning strategy, decomposing the review into a tree of sub-questions and aggregating leaf-level evidence into a complete critique. In our experiments, prompting-based baselines are executed under a standardized backbone configuration to isolate the effect of the prompting framework rather than confounding it with backbone choice.

#### B.2.2 Baseline Implementation and Configuration

SEA-E SEA-E operates as a structured evaluation model, utilizing the model ECNU-SEA/SEAE to generate comprehensive review components such as summaries, strengths, weaknesses, and numerical ratings. To accommodate full-length manuscripts, the engine is configured with a 70,000token context window. The pipeline processes a batch size of 4 papers simultaneously, generating the final critique with the maximum output length capped at 8,000 tokens. To ensure a balance between analytical diversity and factual coherence, the generation hyperparameters are strictly configured with a temperature of 0.7 and a top-p of 0.9 to maintain high-quality and academically rigorous outputs.

CycleReviewer CycleReviewer utilizes a model WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B optimized through an iterative preference-based reasoning framework. All inference workloads are executed on NVIDIA RTX A5000 GPUs. The model employs a 24,000-token context window to accommodate and process complete manuscript texts. For each manuscript, the system executes 2 to

##### 3 iterative refinement passes to progressively enhance the review quality. The 8B engine operates on a single GPU configuration. It generates critiques with a maximum generation length of 3,000 tokens. Generation hyperparameters are configured with a temperature of 0.7, top-p of 0.9, top-k of 50, and a repetition penalty of 1.2.

DeepReview DeepReview utilizes the WestlakeNLP/DeepReviewer-14B core reasoning engine alongside a retrieval-augmented subsystem powered by OpenScholar. All inference workloads are executed on NVIDIA RTX A5000 GPUs. Aligning with the original architecture, the retrieval module employs Llama-3.1-OpenScholar-8B (configured with a 70,000-token context limit) for evidence synthesis and Qwen-2.5-3B-Instruct (configured with a 10,000-token context limit) for query processing. For each manuscript, the system transforms generated questions into search keywords to retrieve approximately 30 candidate papers, utilizing a dedicated reranking model to select the top 10 most relevant sources for grounding. The core 14B engine operates across two GPUs via tensor parallelism. It processes a batch size of 8 papers with a maximum generation length of 7,000 tokens. Generation hyperparameters are configured with a temperature of 0.8, top-p of 0.9, top-k of 50, and a repetition penalty of 1.2.

Reviewer2 The Reviewer2 framework originally operates on a two-stage prompting methodology utilizing custom checkpoints (GitBag/Reviewer2_Mp and GitBag/Reviewer2_Mr). However, due to the suboptimal generation quality observed from these native models during our preliminary evaluations, we replace the underlying generation engine with the open-weights Qwen-3.5-14B model. Crucially, we strictly retain Reviewer2’s official prompt templates to preserve the methodological integrity of their two-stage pipeline. All inference workloads are executed on NVIDIA RTX A5000 GPUs. To process extensive manuscripts, the engine is configured with an 80,000-token context window. The pipeline processes a batch size of 4 papers simultaneously. During Phase 1, the model generates specific review questions, and in Phase 2, it synthesizes the final critique based on these generated prompts, with the maximum generation length capped at 7,000 tokens. Across both stages, the generation hyperparameters are strictly configured with a temperature of 0.8, top-p of 0.9, top-k of 50, and a repetition penalty of 1.2 to ensure diverse yet academically rigorous outputs.

TreeReview TreeReview models the peer review process as a hierarchical and bidirectional questionanswering framework. While the original implementation utilizes GPT-4o, we standardized the backbone to Qwen3-14B to ensure a fair comparison across all prompting-based baselines. All inference workloads are executed on NVIDIA RTX A5000 GPUs. To accommodate the full paper text alongside the dynamically expanding tree of sub-questions, the engine is configured with an 80,000-token context window. The pipeline processes a batch size of 4 papers simultaneously. Following its core logic, the system recursively decomposes high-level review objectives into finegrained sub-questions and aggregates answers from leaf to root to synthesize the final critique, with the maximum output length capped at 7,000 tokens. Across all reasoning stages, the generation hyperparameters are strictly set to a temperature of 0.8, top-p of 0.9, top-k of 50, and a repetition penalty of 1.2.

#### B.2.3 Review generation process.

Before applying our evaluation framework, we must first generate the corresponding AI reviews. For each of the 1,000 papers in our dataset, we provide the complete textual content—including all sections and tables represented in text form—to all five LLM reviewer baselines. Figures and other visual elements are excluded, as the LLM reviewers considered in this study do not yet reliably support multimodal (vision–language) understanding. Each model then independently generates a complete peer review based on its respective methodology. This process yields a comprehensive corpus of 5,000 automated reviews, which serves as the primary testbed for all subsequent PRISM evaluations.

C PRISM Evaluation Framework: Pipeline Details and Experimental Setup

To ensure full reproducibility and provide transparency into our methodology, this section details the hyperparameters used for all evaluated baselines and the exact prompt templates deployed for multi-dimensional assessment.

#### C.1 The PRISM Evaluation Pipeline

###### (1) Depth of Analysis

|PHASE 3: Evidence Grounding Score| |
|---|---|
|LLM Judge Evaluate Depth<br><br>- Level 0: Vague / Generic<br>- Level 1: Internal Grounding<br>- Level 2: External Grounding<br>| |
| | |

|PHASE 2: Argument Classification| |
|---|---|
|Classify Role: Claim / Premise<br><br>Classify Aspect: Novelty / Methodology / Experiment / Clarity| |
| | |

Yes

Is Role = Premise?

ADUs

###### PHASE 1: Preprocessing & Segmentation

Metric Depth of Analysis Score

LLM Extraction Argument Discourse Units Extraction

###### (2) Multi-dimensional Constructiveness

|PHASE 2: Constructiveness Dimension Scoring| |
|---|---|
|LLM Judge Atomic Review Comment Scoring (Score 0-2) Dimension 1: Actionability Dimension 2: Specificity Dimension 3: Justification Quality Dimension 4: Solution Orientation Dimension 5: Tone & Professionalism| |
| | |

LLM Extraction Atomic Review Comments Extraction

Metric Constructiveness Score

Extract Sections

ARCs

- - Summary
- - Strengths
- - Weaknesses
- - Question

Peer Review Text

###### (3) Flaw Identification & Prioritization of Major Issues

LLM-based Flaw Extraction Flaw Arguments Extraction

###### PHASE 2: Flaw Verification & Severity

Flaw Arguments

LLM Judge Assign severity level Critical / Minor / None

Metric Flaw Identification Score

Is Flaw = Valid?

Metric Critique Prioritization Score

Align with Review & Assess Prioritization

LLM Extraction Claims Extraction

###### (4) Novelty Assessment

LLM Extraction Contributions Extraction

Paper Text

|PHASE 3: Automated Review Verification|
|---|
|Pair-Wise Matching Claim-Candidate<br><br>LLM Judge Classify Score per Pair|

Metric Novelty Grounding Score

Review Section

- - Novelty claims
- - All citations

Paper Section

- - core task
- - contributions

###### PHASE 2: Targeted Literature Retrieval

Candidate Pool

Build Queries Q = core task + contribution

Semantic Scholar API Top-K per query (K=10)

- - ID
- - Title
- - Abstract

Paper Content Abstract + Introduction

Figure 9: Detailed Flow of PRISM

#### C.2 PRISM Judge Setup.

To compute the diverse evaluation metrics defined in our framework PRISM, we adopt the LLMas-a-Judge paradigm. We deploy Gemini 2.5 Flash Lite (gemini-2.5-flash-lite) [39] as the core evaluation engine for all metric extraction and scoring tasks. To ensure strict reproducibility and minimize generation variance, we explicitly configure the model parameters by setting the temperature to 0.0 and top-p to 0.95, without utilizing top-k sampling. During the evaluation phase, the model is strictly prompted with our standardized rubrics to systematically extract arguments, verify flaw validity against the ground truth, and compute component scores for both human and automated reviewers equally.

#### C.3 Prompt Templates by Dimension

- Dimension 1: Depth of Analysis. The evaluation of Depth of Analysis is executed through a sequential, three-phase prompting pipeline designed to isolate and score scientific arguments. The first

phase segments the raw review text into discrete Argumentative Discourse Units (ADUs), as detailed

- in Figure 10. Subsequently, the second phase classifies each extracted ADU into its corresponding argument role (Claim or Premise) and maps it to one of four predefined aspect topics, illustrated
- in Figure 11. Finally, the third phase evaluates the empirical depth of the review by computing a categorical Grounding Score specifically for the identified Premises, which is presented in Figure 12.

#### Phase 1: Argumentative Discourse Unit (ADU) Segmentation ROLE AND OBJECTIVE

You are an expert NLP researcher specializing in Scholarly Argumentation Mining. Your task is to segment the following scientific peer review text into distinct Argumentative Discourse Units (ADUs).

#### GUIDELINES

- 1. After each independent logical unit, insert the exact marker <sep>.
- 2. Keep the original text in the exact same order WITHOUT adding, removing, or altering any words.
- 3. CRITICAL: DO split complex/compound sentences when a conclusion/claim is joined with its supporting reason/evidence. Split at:

- • Logical and causal conjunctions ("because", "as", "since", "due to", "but", "however", etc).
- • Relative pronouns (e.g., "which", "that", "who", etc).
- • Participial phrases indicating result/proof ("demonstrating", "showing", "proving", "making", "resulting in", etc).

- 4. IGNORE structural headings entirely (e.g., "**Summary:**", "**Strengths:**"). DO NOT append <sep> to them.

#### INPUT REVIEW TEXT: {raw_review_text}

- Figure 10: PRISM Depth of Analysis Prompt (1/3): Argumentative Discourse Unit (ADU) Segmentation.

#### Phase 2: Argument Role and Aspect Topic Classification ROLE AND OBJECTIVE

Classify the Argument Role and Aspect Topic for the provided list of segmented ADUs. Use the full review as macro-context. For EACH argument in the list, classify its Argument Role and Aspect Topic.

#### 1. ARGUMENT ROLE CLASSIFICATION (Choose exactly ONE)

- • Claim (Conclusion / Point): The statement that is being argued for. It is the controversial statement or the central point that needs support.
- • Premise (Reason / Support): The statement that is used to support the Claim. It provides the reasons, evidence, justifications, or grounds to accept the Claim.

Example:

- • Macro-Context: "The proposed method is not novel. Similar architectures were already introduced by Smith et al. (2023)."
- • ADU: "The proposed method is not novel." → Claim (Needs evidence to be proven).
- • ADU: "Similar architectures were already introduced by Smith et al. (2023)." → Premise (Direct evidence to support the Claim).

#### 2. ASPECT TOPIC CLASSIFICATION (Choose exactly ONE)

- • Novelty & Related Work: Discusses originality, plagiarism, or literature review.
- • Methodology & Theoretical Soundness: Discusses math, algorithms, architecture, or dataset guidelines.
- • Experimental Design & Evaluation: Discusses empirical setups, baselines, ablation studies, and metrics.
- • Clarity, Presentation & Reproducibility: Discusses writing quality, typos, or formatting.

OUTPUT FORMAT Respond ONLY with a valid JSON object. MACRO-CONTEXT: {macro_context} LIST OF ARGUMENTS: {argument_list}

- Figure 11: PRISM Depth of Analysis Prompt (2/3): Argument Role and Aspect Topic Classification.

#### Phase 3: Premise Grounding Score Evaluation ROLE AND OBJECTIVE

You are an expert NLP researcher. I will provide a full peer review (for context) and a list of PREMISES (evidence). Evaluate the "Grounding Score" (Depth of Evidence) for EACH premise.

#### GROUNDING SCORE DEFINITIONS

- • Score 0 (Generic / Vague): The premise is vague and lacks specific anchors (e.g., "The datasets", "Past research", "The equations").
- • Score 1 (Internal Grounding): The premise explicitly anchors to elements INSIDE the manuscript (e.g., "Equation 4", "Table 2", "The proposed module").
- • Score 2 (External / Comparative): The premise anchors to EXTERNAL knowledge outside the manuscript (e.g., "(Smith et al., 2023)", "RoBERTa", "The GLUE benchmark").

OUTPUT FORMAT Respond ONLY with a valid JSON object. MACRO-CONTEXT: {macro_context} LIST OF PREMISES: {premise_list}

Figure 12: PRISM Depth of Analysis Prompt (3/3): Premise Grounding Score Evaluation.

Running Example: Depth of Analysis. To illustrate the full Depth of Analysis pipeline, consider the following excerpt from a raw review: “3 seeds is too few to get any statistical confidence, especially without doing independent hyperparameter sweeps for each baseline. While in the past this has been standard, as a field we continually have shown that the statistical power of our experiments are laughably poor. The performance of the proposed goal-conditioned RL algorithm on the most challenging tasks was less than 50%. QRL assumes deterministic dynamics of the environment, while TD InfoNCE learns without such assumption.”

Processing this text through our three-phase framework yields the following structured output:

- • Claim: “3 seeds is too few to get any statistical confidence...”

→ Aspect: Experimental Design & Evaluation.

- • Premise 1: “While in the past this has been standard, as a field we continually have shown that the statistical power of our experiments are laughably poor.”

→ Aspect: Experimental Design & Evaluation.

→ Grounding Score: 0 (Generic/Vague — purely a community-level assertion with no specific anchor).

- • Premise 2: “The performance of the proposed goal-conditioned RL algorithm on the most challenging tasks was less than 50%.”

→ Aspect: Experimental Design & Evaluation.

→ Grounding Score: 1 (Internal — directly references a quantitative result reported inside the manuscript).

- • Premise 3: “QRL assumes deterministic dynamics of the environment, while TD InfoNCE learns without such assumption.”

→ Aspect: Methodology & Theoretical Soundness.

→ Grounding Score: 2 (External/Comparative — explicitly references QRL, a published external baseline with a known stated assumption).

Finally, the overall DoA score is defined as the harmonic mean of the Premise Ratio and the Normalized Grounding Score. This formulation ensures that a high DoA score requires both a sufficient volume of supporting arguments and a high degree of external/internal grounding:

Rpremise × Sgrounding Rpremise + Sgrounding

DoA = 2 ×

Calculation for the Running Example: In the excerpt above, the pipeline extracted a total of 4 ADUs (1 Claim and 3 Premises). The premises received grounding scores of 0, 1, and 2. Applying our aggregation formulas yields:

- 3

- 4

Rpremise =

= 0.75

0 + 1 + 2 3 × 2

Sgrounding =

= 0.5

0.75 × 0.5 0.75 + 0.5

DoA = 2 ×

= 0.6

- Dimension 2: Novelty Assessment. The evaluation of Novelty Assessment is executed through a three-phase pipeline that grounds reviewer novelty claims against an external body of retrieved prior work, enabling verifiable, evidence-based scoring rather than purely introspective LLM judgment. The pipeline is summarized in Figure 9.

- Phase 1: Structured Target Extraction. The first phase, illustrated in Figure 13, processes both the submitted paper and the peer review in a single LLM call. From the paper, the model extracts a core_task (the concrete problem addressed, ≤20 words), up to three structured contributions (each with a verbatim author claim, a normalized paraphrase, and a source location hint), a set of key_terms, and a list of must_have_entities (named models, datasets, metrics). From the review, the model identifies novelty_claims — verbatim review sentences asserting that the paper is or is not novel — and annotates each with its argumentative stance (not_novel / somewhat_novel / novel / unclear), confidence language, and any cited prior-work strings. A deterministic regex fallback independently augments the citation list from the raw review text to prevent hallucinated references.
- Phase 2: Related Work Retrieval (non-LLM). The second phase is entirely deterministic and requires no LLM call. The core_task and contribution names extracted in Phase 1 are composed into structured search queries, which are issued to the Semantic Scholar Academic Graph API. Raw candidates are then deduplicated via approximate title-abstract similarity (≥0.96 threshold), filtered for non-technical documents (editorials, errata, etc.), and temporally constrained to papers published no later than the submission year. Finally, a Maximal Marginal Relevance (MMR) algorithm [42] selects a diverse top-K candidate pool (K = 30 by default) that balances retrieval relevance against redundancy.
- Phase 3: LLM Judge Verification. The third phase, depicted in Figure 16, instantiates an LLM Judge for each (novelty claim, related-work candidate) pair. Given the review sentence, the abstract and introduction of the paper under review, and the title and abstract of the related work, the judge assigns a five-level verdict on a [−2,+2] integer scale.

A verdict of SUPPORTED (+2) indicates that the retrieved evidence confirms the reviewer’s novelty assessment. OVERSTATED (+1) flags cases where partial similarity exists but the reviewer’s claim of “same as / not novel” is too strong. AMBIGUOUS (0) denotes unverifiable claims. UNDERSTATED (−1) penalizes reviewers who miss closely related prior work that is present in the candidate pool. UNSUPPORTED (−2) indicates that retrieved evidence contradicts the claim or that no supporting evidence was found.

Per-sentence scores are aggregated across the retrieved candidate pool using a configurable policy. In the experiments reported in this paper, the default is a relevance-weighted top-3 rule, not max: each claim is scored against its three highest-relevance candidates so that one spuriously favorable match does not dominate the result. We retain max only as an ablation alternative, where it yields a more optimistic estimate of support. The final Novelty Verification Score (NS) for a reviewer is the mean aggregated score over all their novelty claims.

Phase 1: Structured Target Extraction from Paper and Review ROLE AND OBJECTIVE TASK: Extract structured targets for verifiable novelty checking. You will receive TWO sources below: PAPER TEXT and REVIEW TEXT. Return STRICT JSON only (no markdown, no code fences, no extra keys). The output MUST contain BOTH top-level keys: "paper" and "review". For novelty claims, the "text" field MUST be verbatim from the REVIEW TEXT (1–2 sentences max). If the review contains no novelty claims, return an empty novelty_claims list but still include review. OUTPUT JSON SCHEMA (must match exactly):

{

"paper": { "core_task": "string (<=20 words)", "contributions": [

{

"name": "short name for contribution (<=15 words)", "author_claim_text": "verbatim quote from paper (<=40 words)", "description": "normalized paraphrase (<=60 words)", "source_hint": "location tag e.g. Abstract, Introduction §1,

Conclusion" }

], "key_terms": ["5-12 short phrases"], "must_have_entities": ["model/dataset/metric names if any"]

}, "review": {

"novelty_claims": [ {

"claim_id": "C1", "text": "verbatim review claim (1-2 sentences max)", "stance": "not_novel | somewhat_novel | novel | unclear", "confidence_lang": "high | medium | low", "mentions_prior_work": true, "prior_work_strings": ["author-year strings or titles as written"], "evidence_expected": "method_similarity | task_similarity |

results_similarity | theory_overlap | dataset_overlap" }

], "all_citations_raw": ["everything that looks like a citation/title/arxiv

id/url"]

} }

INPUT: {paper_text} {review_text}

Figure 13: PRISM Novelty Assessment Prompt (1/2): Extracting structured paper contributions and review novelty claims as verifiable targets.

Auxiliary Prompt: Core Task Extraction ROLE AND OBJECTIVE TASK: Extract the core task from a research paper. You will receive the full text of a paper below. Return STRICT JSON only (no markdown, no code fences, no extra keys). OUTPUT JSON SCHEMA (must match exactly):

{

"core_task": "string (<=20 words)" }

INPUT: {paper_text}

Figure 14: Auxiliary novelty prompt for extracting the paper’s core task.

Auxiliary Prompt: Contribution Extraction ROLE AND OBJECTIVE TASK: Extract the main contributions claimed by the authors. You will receive the full text of a paper below. Return STRICT JSON only (no markdown, no code fences, no extra keys). OUTPUT JSON SCHEMA (must match exactly):

{

"contributions": [ {

"name": "complete method type phrase (<=10 words, e.g. ’A

gradient-based adversarial attack for ViTs’)", "author_claim_text": "verbatim quote from paper (<=40 words)", "description": "normalized paraphrase (<=60 words)", "source_hint": "location tag e.g. Abstract, Introduction §1,

Conclusion"

} ]

}

INPUT: {paper_text}

Figure 15: Auxiliary novelty prompt for extracting author-claimed contributions.

Phase 3: LLM Judge — Novelty Claim Verification ROLE AND OBJECTIVE INSTRUCTION: You are an impartial Judge that verifies whether the review sentence is a claim about the paper, and how it relates to the related work evidence. Use ONLY the provided text. If the claim is too vague or evidence is missing, return "insufficient". CLASSIFICATION

- • claim: 1 if the sentence is a reviewer claim about the paper being reviewed; else 0.
- • proof: 1 if the sentence provides evidence or support for a claim about the paper; else 0.

#### AXIS 1 — EVIDENCE SUPPORT (stance_alignment)

- • "aligned": reviewer claim aligns with and is supported by the related work evidence.
- • "partial": some relation exists but evidence is not conclusive.
- • "insufficient": claim is too vague, evidence is missing, or unverifiable.
- • "contradicted": evidence contradicts the reviewer claim or no supporting evidence was found. AXIS 2 — CALIBRATION
- • "accurate": reviewer’s strength of claim matches the actual evidence.
- • "overstated": reviewer claims too strongly given the evidence.
- • "understated": reviewer should have been stronger given the evidence.
- • "N/A": not applicable when evidence is insufficient to judge calibration. VERDICT SCALE ([−2,+2])
- • +2 SUPPORTED: Reviewer’s novelty assessment aligns with the retrieved evidence.
- • +1 OVERSTATED: Some relation exists, but the reviewer’s “same as / not novel” claim is too strong.
- • 0 AMBIGUOUS: Claim is too vague or unverifiable with the provided evidence.
- • −1 UNDERSTATED: Reviewer misses very close prior work present in the candidate pool.
- • −2 UNSUPPORTED: Evidence contradicts the claim, or no supporting evidence was found.

OUTPUT FORMAT Return STRICT JSON only (no markdown, no code fences, no extra keys).

{

"review_sentence_id": "S_001", "related_paper_id": "P123", "classification": {"claim": 1, "proof": 0}, "stance_alignment": "aligned", "calibration": "accurate", "score": 2, "label": "SUPPORTED", "explanation": "Short explanation"

}

INPUTS: {review_sentence} {paper_abstract_intro} {related_work_title_abstract}

- Figure 16: PRISM Novelty Assessment Prompt (2/2): LLM Judge verifying each novelty claim against a retrieved related-work candidate on a five-level evidence scale.

Running Example: Novelty Assessment. To illustrate the full Novelty Assessment pipeline, consider a human review of a NeurIPS 2025 oral paper (qYkhCah8OZ), Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation. The benchmark concatenates all human reviews for the same submission; Phase 1 extracts three novelty claims from distinct reviewers:

- • C1 (not_novel, from Reviewer 1’s Weaknesses): “The proposed method appears incremental, as the techniques involving attention weighting and logits fusion are already well-known and mainly borrowed from previous works.”
- • C2 (novel, from Reviewer 2’s Strengths): “The proposed two modules, attention reallocation and adaptive logits fusion, offer a novel and effective perspective to enhance MLLM performance in knowledge-intensive tasks.”
- • C3 (somewhat_novel, from Reviewer 4’s Originality assessment): “The paper proposes a novel approach to reallocate attentions and fuse knowledge, although similar ideas for attention reallocations have been introduced by earlier work.”

- Phase 2 issues structured queries to the Semantic Scholar API, retrieving 20 candidate related works. We report the three most relevant per the top-3 relevance-weighted aggregation policy: (RW1) MambaTrans: Multimodal Fusion Image Translation via LLM Priors; (RW2) Can Multimodal LLMs be Guided to Improve Industrial Anomaly Detection?; and (RW3) CAT+: Investigating and Enhancing Audio-Visual Understanding in LLMs.
- Phase 3 evaluates each (claim, related-work) pair via the LLM Judge:

- • C1 (not_novel): RW1 is off-topic (image fusion, not attention reallocation) and “does not contain information about attention weighting or logits fusion techniques” - Unsupported (−2). RW2 corroborates that attention weighting is a known technique, as its related work section discusses MLLM enhancement via attention mechanisms - Supported (+2). RW3 similarly confirms these are established techniques in multimodal retrieval augmented generation - Supported (+2).
- • C2 (novel): RW1 again “does not contain any information about the paper being reviewed”

- Unsupported (−2). RW2 “discusses a novel multi-expert framework” for MLLM tasks, validating the novelty claim - Supported (+2). RW3 “explicitly states [a] proposed [module] addressing audio-visual understanding enhancement in LLMs, aligning with the novelty claim” - Supported (+2).

- • C3 (somewhat_novel): All three related works corroborate the nuanced stance. RW1 discusses “multimodal fusion with attention, supporting the claim that similar ideas exist” Supported (+2). RW2 “demonstrates that attention-based approaches are common, supporting the ‘somewhat novel’ stance” - Supported (+2). RW3 “proposes attention-based enhancement for LLMs, confirming that similar attention reallocation ideas have been introduced by earlier work” - Supported (+2).

The Novelty Verification Score is the mean aggregated score over all K novelty claims, where each claim’s aggregated score sk is the relevance-weighted average of its top-3 pair verdicts vk,j. Throughout the main results, this top-3 relevance-weighted policy is the adopted default; a maxpooled variant is treated only as a sensitivity check because it typically produces more optimistic scores. Because the raw verdicts lie on a [−2,+2] scale, the raw review-level mean s¯(R) is linearly normalized into the final NS(R) ∈ [0,1] via:

K

3

s¯(R) + 2 4

1 K

wj · vk,j, NS(R) =

sk, sk =

s¯(R) =

j=1

k=1

Calculation for the Running Example: With K = 3 claims and equal relevance weights (wj = 1/3):

(−2) + 2 + 2 3

sC

=

= 0.667, sC

1

2

0.667 + 0.667 + 2.0 3

s¯(R) =

=

(−2) + 2 + 2 3

2 + 2 + 2 3

=

= 0.667, sC

=

= 2.0

3

3.333 3

1.111 + 2 4

= 1.111, NS(R) =

#### = 0.778

This example illustrates three key behaviors of the pipeline. First, the somewhat_novel claim (C3) achieves the highest per-claim score because its nuanced stance (“novel... although similar ideas exist”) is confirmed by all retrieved evidence, demonstrating that calibrated claims are rewarded. Second, the not_novel (C1) and novel (C2) claims receive identical aggregated scores (0.667) despite opposing stances, because the same off-topic related work (MambaTrans, which is about image fusion rather than MLLM knowledge utilization) penalizes both equally, highlighting that retrieval quality, not just claim content, drives the verdict. Third, the overall normalized score NS(R) = 0.778 reflects a review whose novelty assessments are partially well-grounded but sensitive to the composition of the retrieval pool.

- Dimension 3: Flaw Identification & Prioritization. The evaluation of Flaw Identification and Prioritization is implemented as a two-phase pipeline that cross-validates reviewer arguments against the manuscript and quantifies both detection coverage and critical issue ordering. The first phase, illustrated in Figure 17, consolidates raw review texts from multiple reviewers (Human and LLM) by atomizing and grouping arguments into a structured inventory of Micro-flaws, each categorized within a hierarchical taxonomy of seven macro-topics (e.g., Novelty & Contribution, Methodology & Theoretical Soundness, Experimental Design & Evaluation). The second phase, presented in Figure 18, acts as an independent LLM meta-reviewer that verifies each Micro-flaw against the paper text, assigning a binary validity label (is_valid) and a severity rating (Critical or Minor) according to a predefined ontology grounded in whether fixing the issue requires new experiments or is purely editorial.

From these two phases, three complementary metrics are derived. Critical Recall and Minor Recall measure the proportion of ground-truth Critical and Minor flaws, respectively, that a given reviewer successfully identified, enabling fine-grained diagnosis of detection coverage across severity strata. The primary ranking metric, the normalized Critique Prioritization Score (nCPS), adapts the standard NDCG formulation [35] to capture whether a reviewer front-loads their most severe critiques within each structural section of their review.

#### Phase 1: Micro-flaw Atomization and Grouping ROLE AND OBJECTIVE

You are an expert meta-reviewer for top-tier computer science conferences. Analyze raw review texts from multiple reviewers and consolidate their arguments into a structured list of unique Micro-flaws.

#### GROUPING RULES (STRICTLY ENFORCED)

- 1. Conceptual Consistency (Must Split): Arguments grouped into the same Microflaw MUST address the same fundamental problem. Do not merge distinct scientific issues merely because they share a broad topic.
- 2. Allowed Aggregation (Can Group): Arguments MAY be grouped if they point to the exact same specific error in the paper (e.g., multiple reviewers citing the same missing baseline).
- 3. No Forced Fit: If an argument does not fit any existing Micro-flaw precisely, create a new one.
- 4. No Upper Bound: There is no limit on the number of Micro-flaws. Multiple Micro-flaws may share the same Macro-topic.

TAXONOMY (7 Macro-topics): Novelty & Contribution; Clarity & Presentation; Applicability, Scalability & Limitations; Experimental Design & Evaluation; Related Work & Citations; Methodology & Theoretical Soundness; Reproducibility & Open Science.

OUTPUT FORMAT Respond ONLY with a valid JSON object.

#### INPUT REVIEWS: {input_text}

- Figure 17: PRISM Flaw Identification Prompt (1/2): Atomizing and grouping reviewer arguments into a canonical Micro-flaw inventory.

Phase 2: Meta-reviewer Validity and Severity Judgement ROLE AND OBJECTIVE You are a strict and objective Meta-Reviewer. Given the full paper text and a JSON list of Micro-flaws raised by reviewers, independently verify each flaw against the manuscript. FOR EACH MICRO-FLAW, ANSWER:

- 1. is_valid (True/False): Does this flaw genuinely exist in the paper? Return False if it is a hallucination, misunderstanding, or unreasonable request.
- 2. severity ("Critical" / "Minor"): If valid, assign severity per the ontology below.

#### SEVERITY ONTOLOGY

- • Critical (w = 2): Flaws requiring new experiments, proof revisions, or core claim changes — covering Methodology, Experimental Design, Novelty, and severe Reproducibility/Applicability issues.
- • Minor (w = 1): Flaws fixable via textual/editorial revision — covering Clarity & Presentation, missing citations, or documentation gaps.
- • Borderline rule: Prefer Critical if fixing the issue can plausibly alter main conclusions; prefer Minor if the fix is purely editorial.

OUTPUT FORMAT Respond ONLY with a valid JSON object.

INPUT: {paper_text} {micro_flaws_json}

- Figure 18: PRISM Flaw Identification Prompt (2/2): LLM meta-reviewer verifying flaw validity and assigning severity labels as ground truth.

Running Example: Flaw Identification & Major Issues Prioritization. To illustrate both the Flaw Identification and Prioritization pipelines, consider a reviewer evaluating a graph neural network paper. The Ground Truth flaw set, defined as the union of all valid flaws identified across all reviewers for this paper, consists of:

#### • Critical flaws (GT):

- – FC1: “Missing comparison to GraphSAGE baseline in Table 2.”
- – FC2: “Convergence proof contains a gap in Lemma 3: the Lipschitz assumption is invoked but never verified.”
- – FC3: “No ablation study on the effect of message-passing depth.”

#### • Minor flaws (GT):

- – FM1: “Inconsistent notation: A and A˜ used interchangeably in Eq. 4 and Eq. 5.”
- – FM2: “Figure 2 axes are unlabeled.”
- – FM3: “Related work omits Liu et al. (2022).”

Reviewer X produces the following flaw list, in the order they appear in the review:

- 1. [Minor] FM1 — inconsistent notation D
- 2. [Critical] FC1 — missing GraphSAGE comparison D
- 3. [Minor] FM2 — Figure 2 axes unlabeled D
- 4. [Critical] FC2 — convergence proof gap D

Calculation for Flaw Identification: Reviewer X matched 2 out of 3 critical flaws (missed FC3) and 2 out of 3 minor flaws (missed FM3):

Critical Recall = |{FC1, FC2}|

- 2

- 3 ≈ 0.667

=

|{FC1, FC2, FC3}|

Minor Recall = |{FM1, FM2}|

2 3 ≈ 0.667

=

|{FM1, FM2, FM3}|

Calculation for Major Issue Prioritization (nCPS):

The nCPS is computed over all k GT-matched valid flaws identified by Reviewer X, ordered by their position of appearance in the review. Each flaw receives a position discount log2(pi + 1) and a severity weight wi ∈ {2,1} for Critical/Minor respectively. The GT-matched valid flaws are: FM1 (Minor, position 1), FC1 (Critical, position 2), FM2 (Minor, position 3), FC2 (Critical, position 4).

wFM1 log2(1 + 1)

wFC1 log2(2 + 1)

wFM2 log2(3 + 1)

CPS =

+

+

+

1 log2 2

2 log2 3

1 log2 4

2 log2 5

=

+

+

+

= 1.000 + 1.262 + 0.500 + 0.861 = 3.623 The ideal score iCPS places all Critical flaws first, then Minor:

wFC2 log2(4 + 1)

2 log2 2

2 log2 3

1 log2 4

1 log2 5

iCPS =

+

+

+

= 2.000 + 1.262 + 0.500 + 0.431 = 4.193

CPS iCPS

nCPS =

- 3.623

- 4.193 ≈ 0.864

=

- Dimension 4: Multi-dimensional Constructiveness. To minimize cognitive overload and ensure high-fidelity scoring, the evaluation of constructiveness is decoupled into a two-phase pipeline. Phase 1, detailed in Figure 19, decomposes the holistic peer review into discrete, non-overlapping Atomic Review Comments (ARCs) while retaining their original context via anchor quotes. Phase 2, illustrated in Figure 20, then evaluates each isolated ARC against the five granular dimensions of constructiveness (D1-D5) using a stringent [0,2] rubric.

Phase 1: Atomic Review Comment (ARC) Extraction ROLE AND OBJECTIVE You are an expert peer-review analyst. Your task is to decompose a comprehensive peer review into distinct Atomic Review Comments (ARCs). EXTRACTION RULES

- 1. Extract ALL distinct points from Summary, Strengths, Weaknesses, Questions, and Suggestions.
- 2. One point per ARC: If a single sentence contains two critiques, split it into two separate ARCs.
- 3. Anchor Quote: Provide a verbatim 5-25 word substring copied EXACTLY from the review to anchor the comment.
- 4. Comment Type: Classify each ARC as exactly one of: weakness, strength, question, suggestion, or observation.

OUTPUT FORMAT Respond ONLY with a valid JSON object.

INPUT REVIEW TEXT: {raw_review_text}

- Figure 19: PRISM Constructiveness Prompt (1/2): Decomposing the review into distinct Atomic Review Comments (ARCs).

Phase 2: Multi-dimensional Constructiveness Scoring ROLE AND OBJECTIVE You are an expert peer-review analyst. Given the full peer review as macro-context and a list of extracted Atomic Review Comments (ARCs), score EACH comment on 5 dimensions. SCORING RUBRIC (Score 0, 1, or 2 for each)

- • D1_actionability — Can the author act on this?

- – 0: Opinion with no guidance (e.g., "poorly written")
- – 1: General direction (e.g., "needs more baselines")
- – 2: Specific, implementable (e.g., "add [MethodX] on CIFAR-10")

- • D2_specificity — References concrete paper elements?

- – 0: Vague (e.g., "has issues")
- – 1: Semi-specific (e.g., "methodology section unclear")
- – 2: Pinpoints exact element (e.g., "Eq 7 in Sec 4.2 missing term")

- • D3_justification — Evidence-backed?

- – 0: Bare assertion (e.g., "not novel")
- – 1: Partial reasoning (e.g., "similar to prior work on X")
- – 2: Full evidence (e.g., "same loss as [Author2020] Eq 3")

- • D4_solution — Suggests improvements?

- – 0: Problem-only (e.g., "baselines weak")
- – 1: Implicit fix (e.g., "lacks recent SOTA")
- – 2: Explicit fix (e.g., "add [M2023] achieving X%")

- • D5_tone — Respectful?

- – 0: Hostile / dismissive
- – 1: Neutral, factual
- – 2: Professional-constructive, encouraging

OUTPUT FORMAT Respond ONLY with a valid JSON object containing the arc_id and the numerical scores for D1 through D5.

MACRO-CONTEXT: {raw_review_text} LIST OF ARCS: {arc_json_list}

- Figure 20: PRISM Constructiveness Prompt (2/2): Evaluating isolated ARCs against the fivedimensional constructiveness rubric.

Running Example: Multi-dimensional Constructiveness. To illustrate the MCS pipeline, consider the following excerpt from a human review of a theoretical machine learning paper: “The paper lacks a clear comparison of its theoretical results (Table 1, Section 5) with prior related work. No experimental results. The toy example should correspond to the motivation example. Provide concrete toy examples illustrating setup and theorems, including specific distributions and query complexity bounds. A detailed comparison to existing results in the bandits literature is needed.”

Processing this text through Gemini yields the following Atomic Review Comments (ARCs):

- • ARC 1 (Weakness): “Lacks clear comparison of theoretical results (Table 1, Sec. 5) with prior related work.”

→ Scores: D1=1 (identifies gap but no specific fix), D2=2 (names Table 1, Sec. 5), D3=1 (partial reasoning), D4=0 (no fix), D5=1 (neutral tone).

- • ARC 2 (Weakness): “No experimental results to validate theoretical findings.”

→ Scores: D1=2 (clear: add experiments), D2=2 (specific), D3=0 (no justification why), D4=1 (implicit: conduct experiments), D5=1 (factual).

- • ARC 3 (Question): “Provide concrete toy examples with distributions and query complexity bounds.”

→ Scores: D1=2 (actionable), D2=2 (specific), D3=0 (no WHY explained), D4=1 (suggested fix), D5=2 (professional-constructive framing).

- • ARC 4 (Weakness): “Detailed comparison to existing bandit results needed.”

→ Scores: D1=2 (actionable), D2=2 (references bandit literature), D3=0 (no justification), D4=0 (problem-only), D5=1 (neutral).

Aggregating dimension-wise over these 4 ARCs:

1 + 2 + 2 + 2 4

2 + 2 + 2 + 2 4

D1 =

= 1.75, D2 =

= 2.00

1 + 0 + 0 + 0 4

0 + 1 + 1 + 0 4

D3 =

= 0.25, D4 =

= 0.50

1 + 1 + 2 + 1 4

D5 =

= 1.25

The Multi-dimensional Constructiveness Score is the normalized average: MCS =

1 5 × 2

D1 + D2 + D3 + D4 + D5

Calculation for the Running Example: MCS =

1.75 + 2.00 + 0.25 + 0.50 + 1.25 10

5.75 10

=

= 0.575

### D Metric Independence Analysis via Pearson Correlation

#### D.1 Motivation and Objective

A key requirement for a multi-dimensional evaluation benchmark is that its constituent metrics should capture distinct, non-overlapping aspects of review quality. If two metrics were highly correlated, they would convey redundant information and effectively reduce the dimensionality of the evaluation, undermining the claim that different facets of peer review are independently assessed. To verify this property, we conduct a pairwise Pearson correlation analysis across the five evaluation dimensions of our benchmark: Depth of Analysis (DoAHM), Novelty Assessment (NS), Flaw Identification (Critical Recall, Minor Recall), Issue Prioritization (nCPS) and Multi-dimensional Constructiveness (MCS).

#### D.2 Statistical Method

Pearson Correlation Coefficient. For two metric vectors x = (x1,...,xn) and y = (y1,...,yn) measured over n paper-review samples, the Pearson correlation coefficient is defined as:

n

(xi − x¯)(yi − y¯)

i=1

rxy =

n

n

(xi − x¯)2 ·

(yi − y¯)2

i=1

i=1

(1)

where rxy ∈ [−1,1]. A value of r = 0 indicates no linear association; |r| = 1 indicates perfect linear dependence.

Significance Test. To assess whether an observed rxy differs significantly from zero, we apply the two-tailed t-test under the null hypothesis H0: ρ = 0 (no linear correlation in the population). The test statistic is:

rxy √n − 2 1 − rxy2

t =

(2)

which follows a Student’s t-distribution with n − 2 degrees of freedom under H0. We report twotailed p-values with significance thresholds p < 0.001 (***), p < 0.01 (**), p < 0.05 (*), and label non-significant results as ns.

Effect Size Interpretation. Statistical significance alone is insufficient because large samples can render even trivially small correlations significant. We therefore assess the practical magnitude of each |rxy| using the conventional thresholds of [43]: |r| < 0.10 (negligible), 0.10 ≤ |r| < 0.30 (small), 0.30 ≤ |r| < 0.50 (moderate) and |r| ≥ 0.50 (large). Only correlations that are both statistically significant and of moderate-to-large magnitude are considered substantively meaningful.

#### D.3 Results and Discussion

- Figure 21 presents the full pairwise Pearson correlation matrix across all six metrics. The results consistently show very weak inter-metric associations, with a maximum absolute coefficient of |r|max = 0.193.

[Figure 31]

- Figure 21: Pearson correlation matrix of the six review quality metrics. Each cell reports the Pearson r coefficient with significance annotation: ∗∗∗p < 0.001, ∗∗p < 0.01, ∗p < 0.05; unmarked cells are not significant (p ≥ 0.05). Blue shading denotes negative correlation; red shading denotes positive correlation.

Cross-dimension independence. The most critical finding concerns correlations across evaluation dimensions. Novelty (NS) shows no significant association with any flaw-related metric (Critical Recall: r = −0.015, p = 0.40; Minor Recall: r = +0.021, p = 0.23; nCPS: r = +0.007, p = 0.70), confirming that evaluating the novelty of reviewer claims is entirely decoupled from flaw detection ability. DoAHM likewise exhibits no meaningful linear relationship with Critical Recall (r = −0.006, p = 0.72), demonstrating that structural argumentation depth is independent of a

reviewer’s capacity to identify methodological flaws. The correlation between DoAHM and MCS is marginally significant (r = +0.094, p < 0.001), yet the effect size remains negligible (r2 < 0.01), confirming that argumentative depth and constructiveness constitute distinct dimensions.

Overall assessment. Seven of fifteen metric pairs show no statistically significant correlation (p ≥ 0.05). All significant pairs have |r| < 0.20, placing them in the negligible-to-small range with shared variance below 4% (r2 < 0.04). These results collectively confirm that the six metrics are empirically near-orthogonal: each captures a distinct dimension of peer review quality, thereby justifying their joint use as a comprehensive multi-dimensional evaluation benchmark.

### E Full Cross-Dataset Quantitative Results

#### E.1 Statistical Significance Testing Protocol

To rigorously assess the performance differences between the LLM baselines and the human groundtruth, we conduct non-parametric statistical testing across all metrics. Given the non-normal distribution of the evaluation scores, we employ the Wilcoxon signed-rank test to compute the uncorrected p-values for paired comparisons.

Furthermore, to stringently control the Family-Wise Error Rate (FWER) while evaluating the crossvenue consistency of the models, we construct our hypothesis families based on model-metric pairs. Specifically, the Holm-Bonferroni step-down correction is applied independently for each LLM baseline within each specific evaluation dimension across the 5 conferences (N = 5 comparisons per family, corresponding to the 5 venues). This approach ensures that any statistically significant result reflects a model’s robust and consistent capability across different peer-review distributions, rather than an isolated success at a single venue.

Throughout the subsequent tables, we report the effect size (rank-biserial correlation r) and denote the Holm-corrected statistical significance using the following standard notation:

- • ns (Not Significant): pholm ≥ 0.05
- • ∗ : pholm < 0.05
- • ∗∗ : pholm < 0.01
- • ∗ ∗ ∗ : pholm < 0.001

Results marked as ns suggest that the model’s performance is statistically indistinguishable from the human baseline, whereas the starred results indicate a robust difference that survives rigorous multiple-comparison correction.

#### E.2 Depth of Analysis

- Table 4: Detailed Cross-Venue Performance for Depth of Analysis. Statistical significance is computed independently for each baseline across the 5 conferences to evaluate consistency.

Depth of Analysis Scores (Mean ± Std)

Baselines

ICLR 2024 ICLR 2025 ICLR 2026 ICML 2025 NeurIPS 2025 Human 0.501 ± 0.070 0.490 ± 0.063 0.497 ± 0.061 0.483 ± 0.063 0.500 ± 0.059 CycleReviewer 0.486 ± 0.132 (ns) 0.485 ± 0.134 (ns) 0.479 ± 0.121 (ns) 0.476 ± 0.145 (ns) 0.494 ± 0.138 (ns) DeepReview 0.480 ± 0.139 (ns) 0.491 ± 0.138 (ns) 0.471 ± 0.142 (ns) 0.482 ± 0.126 (ns) 0.490 ± 0.128 (ns) Reviewer2 0.425 ± 0.108∗∗∗ 0.371 ± 0.128∗∗∗ 0.346 ± 0.146∗∗∗ 0.363 ± 0.135∗∗∗ 0.379 ± 0.128∗∗∗ SEA 0.348 ± 0.172∗∗∗ 0.403 ± 0.148∗∗∗ 0.382 ± 0.161∗∗∗ 0.403 ± 0.139∗∗∗ 0.407 ± 0.152∗∗∗ TreeReview 0.349 ± 0.128∗∗∗ 0.429 ± 0.133∗∗∗ 0.335 ± 0.115∗∗∗ 0.335 ± 0.123∗∗∗ 0.346 ± 0.111∗∗∗

The granular DoA results presented in Table 4 highlight a stark contrast in the evidentiary capabilities of the evaluated baselines. TreeReview, SEA, and Reviewer2 yield consistently lower DoA scores across all five venues, with statistical significance (p < 0.005) underscoring their systematic deficiency in substantiating critiques compared to human reviewers. Conversely, CycleReviewer and DeepReview successfully bridge this gap. The consistent lack of statistical significance (ns) when

compared to the human baseline proves that these models achieve a comparable level of analytical depth. As analyzed previously, this statistical parity is primarily driven by their robust internal grounding mechanisms and high premise ratios, which effectively compensate for the inherent limitations of standard LLM generation.

###### Grounding Score Distribution (Count)

###### Grounding Score Distribution (%)

- Score 0 (Ungrounded)

| |
|---|

- Score 1 (Partially grounded)

- Score 2 (Well grounded)

- Score 0 (Ungrounded)

| |
|---|

- Score 1 (Partially grounded)

| |
|---|

- Score 2 (Well grounded)

31,984

93.7%

35000

100

85.6%

79.4%

30000

76.6%

75.1%

80

###### Percentage(%)

25000

62.9%

###### Count

60

20000

15000

11,540

40

9,112

26.3%

7,729

7,504

10000

7,273

18.4%

6,337

16.3%

14.9%

4,833

4,253

20

10.8%

11.6%

10.0%

5000

1,980

1,872

5.0%

1,806

4.3%

4.8%

2.8%

1,049

1.4%

498

489

254

110

375

0

0

HumanCycleReviewer DeepReview Reviewer2 SEA TreeReview

HumanCycleReviewer DeepReview Reviewer2 SEA TreeReview

- Figure 22: Absolute count (left) and percentage distribution (right) of extracted premises categorized by grounding scores for the human baseline and five evaluated LLMs. Score 0 represents ungrounded or vague premises, Score 1 indicates premises anchored to internal manuscript elements, and Score 2 denotes premises anchored to external literature or benchmarks.

As illustrated in Table 4 and Figure 22, the disparity in DoA scores stems from a complex interplay between the Premise Ratio (the consistency of providing justification) and the Grounding Score (the evidentiary quality of that justification).

The Illusion of Depth in Reviewer2. Reviewer2 presents an intriguing paradox at the grounding level. Despite producing an overwhelming absolute volume of vague, unanchored statements (Score 0), it surprisingly generates more externally grounded premises (Score 2) than other LLM baselines. However, as the per-aspect analysis in Figure 24 confirms, this marginal grounding advantage is entirely negated by its severely low Premise Ratio, which dilutes the overall analytical depth across all aspects.

[Figure 32]

- Figure 23: Distribution of review focus across 4 key aspects. The left panel shows the distribution across all extracted arguments, while the right panel highlights the distribution strictly for premises (grounded arguments).

- Table 5: Macro-average aspect distribution (premise-level) and alignment with Human reviewers measured by Jensen-Shannon Divergence (JSD). JSD ∈ [0,1]; lower values indicate closer alignment to the Human aspect distribution. H (bits) denotes Shannon entropy of the aspect distribution (max

= log2 4 ≈ 2 bits).

#### Reviewer Novelty Methodology Experiment Clarity JSD ↓ H (bits) ↑

Human 0.103 0.508 0.293 0.094 — 1.524 CycleReviewer 0.072 0.513 0.281 0.134 0.090 1.287 DeepReview 0.071 0.568 0.291 0.070 0.092 1.124 Reviewer2 0.089 0.523 0.313 0.075 0.071 1.325 SEA 0.061 0.548 0.275 0.117 0.094 1.223 TreeReview 0.125 0.447 0.199 0.229 0.111 1.442

As illustrated in Figure 23 and Table 5, analyzing the substantive distribution of review focus reveals critical divergences in how analytical effort is allocated across aspects. Fundamentally, both human experts and LLMs dedicate the largest proportion of their critiques to a paper’s core technical components: Methodology (∼50%) and Experimental Design (∼29%), confirming a broad consensus on the most critical review dimensions. To quantify the degree of alignment between each LLM and Human reviewers, we compute the Jensen-Shannon Divergence (JSD) between their premise-level aspect distributions on the same paper. JSD ∈ [0,1], where 0 indicates identical distributions.

Alignment with Human Priorities. Most advanced baselines successfully mirror human intuitive focus, dedicating the largest proportion of their grounded premises to core technical components: Methodology (∼ 50 − 56%) and Experimental Design (∼ 27 − 31%). Notably, Reviewer2 achieves the closest alignment to Human reviewers with the lowest JSD of 0.071. Its premise distribution (Methodology 52.3%, Experiment 31.3%, Clarity 7.5%) closely traces the human pattern (50.8%, 29.3%, 9.4%), demonstrating a well-calibrated allocation of critical effort. DeepReview achieves the lowest Clarity proportion (7.0%) and the highest Methodology concentration (56.8%). While this results in the lowest Shannon entropy (H = 1.124 bits)—indicating a narrower, highly specialized focus—it confirms the model’s capacity to firmly anchor its feedback in the most critical dimensions of the submission.

The Surface-Level Trap. Rather than being an inherent LLM limitation, the “surface-level trap” manifests when automated frameworks lack explicit, domain-specific evaluation constraints. Remarkably, even highly structured pipelines can fall into this trap. This is starkly pronounced in TreeReview, which, despite its complex reasoning topology, allocates an excessive 22.9% of its premise-level effort to Clarity, nearly 2.4× the proportion of Human reviewers (9.4%). Consequently, TreeReview records the highest JSD against Humans (0.111) and the lowest Methodology coverage (44.7%), confirming that without strict dimensional guidance, its analytical distribution naturally diverges toward superficial nitpicking.

###### Premise Ratio

###### Grounding Score

###### DoA Score

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

1.0

1.0

1.0

0.8

0.8

0.8

0.6

0.6

0.6

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

Novelty Methodology Experiment Clarity

Novelty Methodology Experiment Clarity

Novelty Methodology Experiment Clarity

Human

CycleReviewer

DeepReview

Reviewer2

SEA

TreeReview

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

- Figure 24: Per-aspect Depth of Analysis scores for Human and five LLM baselines, averaged across five conferences. Each group corresponds to one of the four review aspects (Novelty & Related Work, Methodology, Experimental Design, Clarity)

- Figure 24 decomposes the overall DoA score into its four constituent aspects—Novelty & Related Work, Methodology, Experimental Design, and Clarity—revealing the underlying drivers of the scores reported in Figure 22 and Table 5. For each aspect, the harmonic mean of Premise Ratio and Grounding Score is reported per reviewer, allowing a fine-grained comparison of where and how deeply each system substantiates its arguments.

Shared Prioritization of Core Technical Aspects. A consistent pattern emerges across both human and LLM reviewers: DoA scores are substantially higher for Methodology and Experimental Design than for Novelty and Clarity. For Human reviewers, the Methodology aspect achieves the highest DoA Score (0.510±0.156), followed closely by Experiment (0.456±0.207), while Novelty (0.357±0.322) and Clarity (0.266 ± 0.268) trail significantly behind. This pattern holds uniformly across all five baselines, confirming that both humans and LLMs inherently recognize the need to anchor their most substantive arguments in the methodological and experimental core of a paper. The effect is directly traceable to elevated Premise Ratios and Grounding Scores in these two aspects: for instance, Human reviewers achieve a Premise Ratio of 0.655 on Methodology versus only 0.425 on Clarity, indicating that technical claims are far more consistently backed by evidence than presentation-level observations.

Evidence Density as the Key Differentiator Across Baselines. The second insight concerns the systematic gap between baselines. Across all four aspects, CycleReviewer and DeepReview consistently achieve DoA scores closest to—and in several cases statistically indistinguishable fromHuman reviewers. This cross-aspect parity is not coincidental: both systems maintain the highest Premise Ratios among LLM baselines across every aspect (e.g., CycleReviewer reaches 0.706 and DeepReview 0.674 on Methodology, both exceeding the Human value of 0.655), demonstrating that their superior overall DoA is driven by a systematic tendency to substantiate claims with grounded evidence regardless of the aspect under discussion. In contrast, Reviewer2, SEA, and TreeReview show markedly lower Premise Ratios particularly on Novelty (0.272, 0.113, 0.186 respectively), producing the steepest per-aspect DoA drops and confirming that their aggregate weakness is not confined to any single dimension but reflects a globally deficient evidentiary discipline.

Summary of Analytical Depth. In conclusion, achieving a human-level Depth of Analysis requires more than merely generating a high volume of text. Models that fall into the trap of unsupported verbosity (Reviewer2) or surface-level nitpicking (TreeReview) are severely penalized. To bridge the analytical gap, automated reviewers must systematically substantiate their claims and strictly prioritize core technical dimensions over formatting issues.

#### E.3 Novelty Assessment

- Table 6: Detailed Cross-Venue Performance for Novelty Assessment. Statistical significance is computed independently for each baseline across the 5 conferences to evaluate consistency.

Novelty Assessment Scores (Mean ± Std)

Baselines

ICLR 2024 ICLR 2025 ICLR 2026 ICML 2025 NeurIPS 2025 Human 0.758 ± 0.224 0.775 ± 0.199 0.762 ± 0.195 0.834 ± 0.188 0.809 ± 0.185 CycleReviewer 0.768 ± 0.201 (ns) 0.785 ± 0.202 (ns) 0.793 ± 0.219 (ns) 0.781 ± 0.210∗ 0.793 ± 0.228 (ns) DeepReview 0.737 ± 0.225 (ns) 0.751 ± 0.209 (ns) 0.737 ± 0.220 (ns) 0.785 ± 0.188∗ 0.784 ± 0.203 (ns) Reviewer2 0.744 ± 0.246 (ns) 0.791 ± 0.216 (ns) 0.792 ± 0.215 (ns) 0.792 ± 0.216 (ns) 0.815 ± 0.195 (ns) SEA 0.784 ± 0.218 (ns) 0.822 ± 0.223∗∗ 0.831 ± 0.184∗∗∗ 0.870 ± 0.187 (ns) 0.860 ± 0.202∗∗ TreeReview 0.799 ± 0.208 (ns) 0.816 ± 0.185 (ns) 0.813 ± 0.201∗ 0.808 ± 0.216 (ns) 0.818 ± 0.197 (ns)

- Table 6 reports the cross-venue scalar Novelty Assessment scores. These values should be interpreted carefully: they do not certify a manuscript’s objective novelty, but rather measure whether the novelty claims expressed in a review can be grounded in retrievable prior work under the PRISM pipeline. Across almost all venues, mean scores fall between 0.730 and 0.870, indicating that both human reviewers and automated baselines frequently produce novelty claims that the retrieval-andverification procedure can resolve with substantial evidence. The large number of non-significant differences (ns) for models such as DeepReview, Reviewer2, and TreeReview therefore suggests similar evidence-grounding performance on this scalar metric, not full claim-by-claim agreement with human reviewers.

The Outperformance of SEA. The most notable result on this scalar metric is the performance of the SEA baseline. While SEA is not uniformly strongest on the other review dimensions, it achieves the highest novelty-assessment score in multiple venues and significantly exceeds the human baseline in ICLR 2025 (pHolm < 0.01), ICLR 2026 (pHolm < 0.005), and NeurIPS 2025 (pHolm < 0.01). Within the interpretation above, this suggests that SEA’s structured generation style tends to produce novelty claims that are especially easy for the PRISM retrieval-and-verification pipeline to ground in prior work. As the agreement analysis below will show, this should not be conflated with universally stronger alignment to human novelty judgments.

[Figure 33]

- Figure 25: Average number of novelty claims generated per review across the human and LLM Reviewers.

Claim Volume and Generation Verbosity. Figure 25 reveals another important difference that scalar novelty scores alone do not capture: review sources vary substantially in how many novelty claims they choose to make. Because the human input concatenates multiple independent reviews for the same paper, raw human claim counts must be normalized by the number of individual reviews before they are interpreted as per-review verbosity. In the unnormalized benchmark output, the concatenated human-review bundle averages 5.1 extracted novelty-related claims per paper; the benchmark code therefore reports the human claim-volume statistic as claims per individual human review whenever review-count metadata is available. DeepReview is markedly more verbose at 8.3 claims per generated review, reflecting a finer-grained decomposition of contributions and comparisons. By contrast, SEA (3.0 claims) and Reviewer2 (3.5 claims) are much more conservative, concentrating their novelty discussion into fewer statements. This matters because downstream agreement depends not only on how claims are scored once extracted, but also on claim granularity and boundary choices at the extraction stage.

Summary. Taken together, these results show that LLM reviewers can produce novelty claims that are often well grounded under the PRISM retrieval-and-verification pipeline. However, this scalar score should not be read as evidence that LLMs can certify the objective novelty of a manuscript, nor that they fully replicate human novelty judgments. The next subsection addresses the harder question: once all review sources are normalized through the same pipeline, do they actually arrive at the same claim-level novelty conclusions?

#### E.4 Flaw Identification & Prioritization of Major Issue

- Table 7: Detailed Cross-Venue Performance for Flaw Identification & Prioritization. Statistical significance is computed independently for each baseline across the 5 conferences to evaluate consistency.

Scores (Mean ± Std) ICLR 2024 ICLR 2025 ICLR 2026 ICML 2025 NeurIPS 2025

Metric / Baselines

Critical Flaw Identification Human 0.322 ± 0.156 0.331 ± 0.172 0.358 ± 0.151 0.347 ± 0.175 0.346 ± 0.160 CycleReviewer 0.272 ± 0.300∗ 0.257 ± 0.291∗∗ 0.186 ± 0.263∗∗∗ 0.254 ± 0.319∗∗∗ 0.232 ± 0.302∗∗∗ DeepReview 0.327 ± 0.287ns 0.362 ± 0.306ns 0.282 ± 0.257∗∗ 0.319 ± 0.324ns 0.371 ± 0.317ns Reviewer2 0.506 ± 0.283∗∗∗ 0.599 ± 0.301∗∗∗ 0.564 ± 0.305∗∗∗ 0.649 ± 0.307∗∗∗ 0.636 ± 0.290∗∗∗ SEA 0.304 ± 0.297ns 0.205 ± 0.259∗∗∗ 0.161 ± 0.225∗∗∗ 0.225 ± 0.240∗∗∗ 0.215 ± 0.265∗∗∗ TreeReview 0.405 ± 0.346ns 0.263 ± 0.264∗∗ 0.205 ± 0.267∗∗∗ 0.229 ± 0.301∗∗∗ 0.259 ± 0.291∗∗∗

Minor Flaw Identification Human 0.277 ± 0.070 0.271 ± 0.079 0.279 ± 0.079 0.299 ± 0.093 0.281 ± 0.069 CycleReviewer 0.201 ± 0.132∗∗∗ 0.191 ± 0.149∗∗∗ 0.186 ± 0.151∗∗∗ 0.195 ± 0.151∗∗∗ 0.154 ± 0.120∗∗∗ DeepReview 0.241 ± 0.158∗ 0.210 ± 0.151∗∗∗ 0.256 ± 0.148∗ 0.232 ± 0.143∗∗∗ 0.203 ± 0.135∗∗∗ Reviewer2 0.412 ± 0.166∗∗∗ 0.475 ± 0.172∗∗∗ 0.468 ± 0.185∗∗∗ 0.479 ± 0.181∗∗∗ 0.463 ± 0.179∗∗∗ SEA 0.394 ± 0.165∗∗∗ 0.217 ± 0.116∗∗∗ 0.224 ± 0.126∗∗∗ 0.207 ± 0.114∗∗∗ 0.193 ± 0.116∗∗∗ TreeReview 0.361 ± 0.163∗∗∗ 0.349 ± 0.153∗∗∗ 0.340 ± 0.147∗∗∗ 0.303 ± 0.135ns 0.306 ± 0.143∗

Prioritization of Major Issue (nCPS) Human 0.967 ± 0.054 0.972 ± 0.037 0.973 ± 0.029 0.978 ± 0.081 0.975 ± 0.032 CycleReviewer 0.983 ± 0.042∗ 0.965 ± 0.112ns 0.969 ± 0.110∗ 0.970 ± 0.108ns 0.968 ± 0.127∗ DeepReview 0.967 ± 0.093ns 0.968 ± 0.053ns 0.967 ± 0.054ns 0.978 ± 0.048ns 0.956 ± 0.110ns Reviewer2 0.982 ± 0.028∗ 0.979 ± 0.029ns 0.970 ± 0.079ns 0.975 ± 0.033∗∗ 0.970 ± 0.033ns SEA 0.971 ± 0.047ns 0.961 ± 0.149ns 0.987 ± 0.036∗∗∗ 0.984 ± 0.038ns 0.981 ± 0.080∗∗∗ TreeReview 0.962 ± 0.090ns 0.964 ± 0.085ns 0.977 ± 0.044ns 0.980 ± 0.042ns 0.975 ± 0.046ns

The detailed results in Table 7 unequivocally confirm the exhaustive diagnostic capability of Reviewer2. Across every single evaluated venue (ICLR 2024 to NeurIPS 2025), Reviewer2 consistently achieves the highest Recall for both Critical (ranging from 0.506 to 0.649) and Minor flaws. More importantly, the statistical tests (pHolm < 0.005, denoted as ∗∗∗) validate that this over-performance relative to the human baseline is structurally ingrained in the model’s generation style, not merely a statistical artifact of a specific dataset. This elevated Recall is directly correlated with its overall

generation volume: as previously illustrated in the valid flaw counts (Figure 7 - main text), Reviewer2 acts as a high-volume "flaw scanner," extracting an unprecedented absolute number of valid issues per review. By casting a wider diagnostic net, it naturally captures a higher proportion of both fatal methodologies and minor anomalies compared to the more conservative human baseline. Conversely, other LLM baselines generally exhibit a diagnostic deficit compared to human experts. Models like CycleReviewer and SEA consistently underperform the human ground-truth in extracting both major and minor flaws across most venues. An interesting anomaly, however, is observed in TreeReview. While it struggles to detect fatal methodological errors (consistently scoring lower than humans in Critical Recall), it frequently outperforms or matches humans in Minor Flaw Identification (e.g., 0.361 at ICLR 2024 and 0.349 at ICLR 2025, with strong statistical significance). This further corroborates our earlier finding that TreeReview suffers from a "surface-level trap," heavily over-indexing its analytical effort on presentation and formatting anomalies rather than scientific rigor. DeepReview, maintaining a highly conservative profile, yields scores that are statistically indistinguishable (ns) from human experts in several venues, demonstrating a strong alignment with human reviewing patterns.

Concluding the table analysis, the granular breakdown of the Critique Prioritization Score (nCPS) solidifies a key macro-level observation: the ability to strategically rank flaws is a solved problem for modern LLMs. Across all conferences, every baseline achieves near-perfect scores (nCPS > 0.96). Regardless of their diagnostic capability, the automated systems’ structural organization adheres to the established academic norm of surfacing major flaws first, rendering them statistically indistinguishable from the human ground-truth.

[Figure 34]

- Figure 26: Comparative Aspect Topic Distribution of extracted valid flaws, stratified by severity (Critical vs. Minor).

To gain a deeper understanding of reviewer behavior, we stratify the aspect topic distribution by flaw severity (Critical vs. Minor), as visualized in Figure 26. This decomposition reveals profound insights into the contextual adaptability—or lack thereof—within LLM baselines compared to human experts.

Alignment on Critical Vulnerabilities. When evaluating Critical flaws, human experts demonstrate a laser-focused approach, allocating an overwhelming 92.3% of their critiques to core technical components (56.5% on Methodology and 35.8% on Experimental Design). Formatting issues (Clarity) account for merely 1.7% of critical flaws. DeepReview exhibits a remarkably identical signature, dedicating 91.7% of its critical flaw detection to Methodology and Experiments, with only 1.6%

wasted on Clarity. This confirms that at the highest severity level, well-calibrated models like DeepReview successfully emulate the human capacity to isolate fatal scientific errors. Conversely, TreeReview broadens the scope of critical evaluation. Rather than limiting itself strictly to methodology, it proactively identifies critical presentation errors at a rate of 6.0%, ensuring that even formal aesthetics are held to the highest standards of severity.

The Adaptive Focus on Minor Flaws. The distribution of Minor flaws reveals a compelling alignment between human experts and LLM baselines. Unlike critical flaws, which are overwhelmingly technical, the cognitive focus for minor flaws naturally shifts toward surface-level anomalies. Both human reviewers and LLMs significantly reduce their extreme scrutiny on Methodology, while substantially increasing their attention to Clarity, Presentation & Reproducibility. For instance, the human focus on Clarity surges to 29.3% for minor flaws (up from a mere 1.7% in critical flaws). Remarkably, the automated baselines successfully mirror this contextual shift. TreeReview captures a massive 38.3% of clarity-related minor issues, while even verbose models like Reviewer2 quadruple their attention to presentation (17.2%) compared to their critical flaw distribution.

Emulating Human Flaw Categorization. This dynamic shift demonstrates that LLMs are not rigidly applying a single evaluation template. Instead, they contextually adapt the nature of their critiques based on severity. While they still capture minor methodological nitpicks (as seen in DeepReview and SEA), they correctly associate a large portion of minor flaws with presentation errors, directly aligning with how human experts intuitively classify minor review points.

Summary. These findings highlight a sophisticated capability in modern automated peer review: LLMs can successfully mimic human intuition in identifying and categorizing different types of flaws. By instinctively mapping core technical breakdowns to Critical severity and correctly shifting their lens toward presentation issues for Minor anomalies, LLMs prove they possess a nuanced, human-like understanding of manuscript evaluation.

#### E.5 Multi-Dimensional Constructiveness

- Table 8: Detailed Cross-Venue Performance for Constructiveness Score (MCS). Statistical significance is computed independently for each baseline across the 5 conferences to evaluate consistency.

Constructiveness Scores (Mean ± Std)

Baselines

ICLR 2024 ICLR 2025 ICLR 2026 ICML 2025 NeurIPS 2025 Human 0.576 ± 0.074 0.584 ± 0.068 0.601 ± 0.063 0.521 ± 0.070 0.546 ± 0.056 CycleReviewer 0.517 ± 0.112∗∗∗ 0.529 ± 0.108∗∗∗ 0.540 ± 0.120∗∗∗ 0.523 ± 0.095ns 0.527 ± 0.119∗ DeepReview 0.629 ± 0.083∗∗∗ 0.629 ± 0.084∗∗∗ 0.636 ± 0.091∗∗∗ 0.638 ± 0.085∗∗∗ 0.635 ± 0.087∗∗∗ Reviewer2 0.555 ± 0.129ns 0.565 ± 0.105ns 0.587 ± 0.100ns 0.584 ± 0.101∗∗∗ 0.586 ± 0.087∗∗∗ SEA 0.498 ± 0.106∗∗∗ 0.518 ± 0.097∗∗∗ 0.502 ± 0.098∗∗∗ 0.487 ± 0.084∗∗∗ 0.485 ± 0.070∗∗∗ TreeReview 0.499 ± 0.116∗∗∗ 0.445 ± 0.107∗∗∗ 0.490 ± 0.144∗∗∗ 0.498 ± 0.125∗∗ 0.496 ± 0.120∗∗∗

- Table 8 presents the granular breakdown of the MCS and its sub-dimensions across the five evaluated conference venues. A longitudinal analysis reveals that while the constructiveness of both human and automated reviewers fluctuates slightly across different venues, DeepReview consistently maintains a substantial lead. Specifically, DeepReview achieves the highest MCS across all five conferences (ranging from 0.629 at ICLR 2024 to 0.635 at NeurIPS 2025). Importantly, Holm’s post-hoc tests confirm that DeepReview’s superiority over the human baseline (0.576 to 0.546) is statistically signif-

icant (pHolm < 0.05) in every venue evaluated. Furthermore, Reviewer2 demonstrates competitive performance, particularly in the later venues (ICML 2025 and NeurIPS 2025), where its MCS scores (0.584, and 0.586) slightly match the human ground-truth. In stark contrast, models such as SEA and TreeReview consistently underperform compared to the human baseline across all years, highlighting a persistent limitation in their capacity to formulate genuinely constructive critiques.

- Table 9: Detailed Constructiveness Sub-dimensions (D1-D5) across 5 Conferences evaluated on a raw scale of [0,2].

System D1: Actionability

D2: Specificity

D3: Justification

D4: Solution

D5: Tone

Human 1.105 ± 0.411 1.725 ± 0.260 0.759 ± 0.465 0.470 ± 0.350 1.589 ± 0.346 CycleReviewer 1.328 ± 0.492 1.897 ± 0.206 0.325 ± 0.434 0.401 ± 0.385 1.321 ± 0.410 DeepReview 1.414 ± 0.294 1.831 ± 0.201 0.580 ± 0.369 0.784 ± 0.290 1.726 ± 0.286 Reviewer2 1.178 ± 0.324 1.784 ± 0.252 0.939 ± 0.429 0.266 ± 0.248 1.586 ± 0.441 SEA 0.909 ± 0.384 1.651 ± 0.252 0.478 ± 0.413 0.375 ± 0.273 1.593 ± 0.245 TreeReview 1.045 ± 0.276 1.532 ± 0.353 0.639 ± 0.463 0.357 ± 0.345 1.278 ± 0.575

[Figure 35]

- Figure 27: Heatmap detailing the performance across the five core sub-metrics of constructiveness (Actionability, Specificity, Justification, Constructive Suggestion, and Tone & Respect).

[Figure 36]

- Figure 28: Comparison of core constructiveness metrics (Actionability Ratio, Solution Density, and Constructiveness Density).

To provide a more nuanced evaluation, we derive three auxiliary density metrics from the five core dimensions (D1 − D5). Let I(·) be an indicator function. We define: (i) the Actionability Ratio (AR) as the proportion of comments providing at least a general direction (D1 ≥ 1); (ii) the Solution Density (SD) as the percentage of comments offering explicit, implementable fixes (D4 = 2); and (iii) the Constructiveness Density (CD) as the proportion of comments achieving high-quality structural alignment (CLC ≥ 0.5). Formally:

1 n

AR(R) =

1 n

SD(R) =

1 n

CD(R) =

n

I(D1(cj) ≥ 1), (3)

j=1

n

I(D4(cj) = 2), (4)

j=1

n

I(CLC(cj) ≥ 0.5) (5)

j=1

As presented in the macro-averaged results (Figure 27) and the venue-specific breakdown (Figure 28), a granular examination of these metrics reveals distinct behavioral signatures distinguishing human reviewers from LLM baselines.

The Solution Bottleneck. The most profound divergence between human experts and top LLMs lies in the capacity to propose explicit improvements. The data reveals a systemic limitation in traditional peer review: while humans are highly proficient at pinpointing concrete flaws (D2 Specificity ≈ 1.72), they frequently fail to provide actionable remedies. This is evidenced by the human baseline’s remarkably low Solution Density (SD ≈ 0.102), indicating that only 10% of human comments contain an explicit fix (D4 = 2). In stark contrast, DeepReview fundamentally alters this paradigm. It consistently achieves an SD approaching 0.29 and D4 scores near 0.80 across all venues. DeepReview excels because it does not merely diagnose problems; it proactively prescribes explicit, implementable solutions, thereby maximizing its utility to the authors.

The Verbosity Paradox of Reviewer2. The sub-dimensional breakdown elegantly explains Reviewer2’s deceptively high overall scores. Reviewer2 achieves exceptional Justification scores (D3 > 0.90) and a strong Constructiveness Density (CD ≈ 0.58), largely as an artifact of its highly verbose, two-stage rubric-driven generation style. It excels at providing extensive reasoning for its claims. However, its abysmal Solution Density (SD ≈ 0.05) and low Solution score (D4 ≈ 0.26) reveal a critical flaw: it exhaustively explains why something is wrong but almost never tells the author how to fix it, rendering its voluminous critiques practically inert.

Actionability without Depth in CycleReviewer. Conversely, CycleReviewer exhibits a contrasting failure mode. It achieves the highest Specificity (D2 ≈ 1.89) and Actionability Ratio (AR > 0.90), meaning almost all of its comments reference concrete paper elements and offer at least a general direction (D1 ≥ 1). Yet, it suffers a catastrophic drop in Justification (D3 ≈ 0.32) and Solution Density (SD ≈ 0.04). This signature points to a shallow, "checklist-style" reviewing behavior: it successfully targets specific sections with general demands (e.g., "needs more baselines"), but fails to provide the evidentiary backing or the explicit fixes required of a profound scientific critique.

Tone and Professionalism. Finally, the Tone dimension (D5) confirms that well-calibrated LLMs can systematically elevate the discourse of peer review. DeepReview (D5 ≈ 1.72) consistently outputs more professional, neutral, and encouraging feedback than the human baseline (D5 ≈ 1.58), effectively mitigating the dismissive or hostile language occasionally encountered in human peer reviews.

Summary of Constructiveness. This multi-dimensional analysis highlights a fundamental difference in reviewing paradigms. Human experts primarily act as diagnosticians, highly effective at pinpointing errors but lacking in actionable guidance. LLM baselines often mask this deficiency with extreme verbosity (Reviewer2) or superficial demands (CycleReviewer). In contrast, DeepReview transcends these limitations, acting more as a collaborator by bridging the critical gap between

identifying flaws (D2) and formulating explicit, professionally toned solutions (D4,D5), it represents a significant step toward genuinely constructive automated peer review.

#### E.6 Review Sensitivity to Paper Quality: Accept vs. Reject Analysis

Statistical Methodology. To evaluate whether the generated reviews effectively distinguish between high-quality and low-quality submissions, we compare the metric scores assigned to accepted versus rejected papers (Figure 29). Specifically, we apply the Mann-Whitney U test (two-sided, unpaired) to compare per-paper metric scores between accepted and rejected papers for each reviewer baseline. To control the family-wise error rate within each reviewer, p-values are adjusted using the Holm-Bonferroni procedure across all metrics tested for that reviewer. The mean score differences (∆ = Accept − Reject) and their significance levels are detailed in Table 10.

[Figure 37]

- Figure 29: Mean scores of Human and LLM reviewers on six evaluation dimensions, stratified by paper decision (Accept vs. Reject)

- Table 10: Metric differences across reviewers. Bold values indicate statistically significant results; (ns) denotes non-significant.

Metric Human Reviewer2 TreeReview DeepReview SEA CycleReviewer

Novelty Score +0.051∗ +0.024ns +0.035ns +0.037ns +0.029ns +0.021ns DoA Score −0.006ns −0.004ns −0.028ns −0.001ns −0.006ns −0.003ns Critical Score −0.049∗∗∗ −0.058ns +0.043ns +0.005ns −0.005ns +0.018ns Minor Score −0.018∗∗ −0.008ns −0.030ns −0.002ns −0.007ns −0.004ns Prioritization Score +0.006∗∗∗ −0.004ns +0.001ns +0.011∗∗∗ −0.001ns −0.001ns MCS −0.001ns +0.001ns +0.011ns −0.006ns +0.004ns −0.010ns

Human Reviews Exhibit Strong Predictive Validity. Although human reviewers evaluate manuscripts blindly without any knowledge of the final editorial outcome, their assessments exhibit a robust, statistically significant correlation with the eventual decisions. Specifically, manuscripts that are ultimately accepted garner blind reviews with significantly higher Novelty Scores (∆ = +0.051∗) and tighter structural prioritization (∆Prior. = +0.006∗∗∗). Conversely, papers that are ultimately rejected accumulate substantially more critical diagnostic feedback, reflected in significantly worse scores for both Critical (∆ = −0.049∗∗∗) and Minor (∆ = −0.018∗∗) flaws. The magnitude of the Critical score difference is notably 2.7× larger than that of the Minor score. This confirms that human reviewers naturally calibrate the severity and focus of their critiques according to the

inherent scientific merit of the manuscript, providing a valid signal that organically drives the final accept/reject decision.

LLMs Exhibit Evaluative Invariance Across Quality Tiers. In contrast to human reviewers, whose critiques strongly correlate with the final editorial outcome, LLM reviewers exhibit a highly invariant and consistent evaluative pattern regardless of whether a paper is ultimately accepted or rejected. Across all five automated systems, the metric differences are predominantly uniform, with only DeepReview’s Prioritization Score reaching statistical significance (∆ = +0.011∗∗∗). While the LLMs demonstrate a slight directional alignment with humans on interpretive dimensions—yielding marginally positive ∆ for Novelty and negative ∆ for Minor flaws—these shifts lack statistical power. Notably, this stability is most pronounced in the Critical Score dimension. Rather than being swayed by the overall quality of the submission, LLMs apply their internal diagnostic heuristics independently: Reviewer2 maintains a strict standard (−0.058ns), TreeReview swings inversely (+0.043ns), and the others remain stable near zero.

Summary of Evaluative Invariance. The results in Table 10 highlight two fundamentally different review paradigms. Human reviewers display high sensitivity, naturally adjusting the severity of their critiques based on the overall scientific merit of the submission. In contrast, LLMs function as invariant diagnostic scanners. They generate reviews with remarkably stable metric distributions, applying a uniform evaluative standard across all papers. While this invariant behavior suggests that LLMs are not susceptible to the “halo effect” of high-quality submissions, it also implies a trade-off: their strict consistency limits their capacity to organically discriminate and highlight fatal methodological flaws with the same adaptive precision as human experts.

#### E.7 Evaluator Robustness Across LLM Backends

###### Depth of Analysis

###### Constructiveness

###### Novelty Assessment

- 0.0

- 0.1

- 0.2

- 0.3

- 0.4

- 0.5

- 0.6

1.0

0.7

0.494

0.629

0.841

0.481 0.485

0.829

0.798

0.795

0.585 0.587

0.787

0.6

0.8

0.771

0.571

0.570

0.734

0.421

0.534 0.530

0.709 0.707 0.711

0.690

0.687

0.504

0.489

0.381

0.5

0.380

0.477

0.370 0.364

0.357

0.342

0.6

0.433

0.425

Score(01)

Score(01)

Score(01)

0.4

0.300

0.291

0.3

0.4

0.2

0.2

0.1

0.0

0.0

Human SEA Tree Reviewer2 DeepReview CycleReview

Human SEA Tree Reviewer2 DeepReview CycleReview

Human SEA Tree Reviewer2 DeepReview CycleReview

###### Prioritization

###### Critical Recall

###### Minor Recall

0.7

0.5

0.466

1.0

0.613

0.973 0.968 0.974 0.977 0.974 0.977

0.961 0.963 0.960 0.958 0.962

0.946

0.6

0.442

0.4

0.509

0.8

0.5

0.312

Recall(01)

Recall(01)

Score(01)

0.303

0.4

0.3

0.6

0.336

0.252

0.251

0.332

0.308

0.232

0.304

0.228

0.3

0.213

0.279

0.199

0.2

0.4

0.237

0.176

0.228

0.224

0.221

0.2

0.183

0.131

0.1

0.2

0.1

0.0

0.0

0.0

Human SEA Tree Reviewer2 DeepReview CycleReview

Human SEA Tree Reviewer2 DeepReview CycleReview

Human SEA Tree Reviewer2 DeepReview CycleReview

Gemini Flash Lite

Mimo v2.5 Pro

| |
|---|

- Figure 30: Comparison of all evaluation metrics between Gemini Flash Lite and Mimo v2.5 Pro as evaluator backends, aggregated over 250 papers (50 per conference: ICLR 2024/25/26, ICML 2025, NeurIPS 2025). Dashed lines indicate cross-group trends for each model.

To assess whether our metric framework is sensitive to the choice of evaluator LLM, we re-ran the full evaluation pipeline using Mimo v2.5 Pro [41] as an alternative backend and compared results against Gemini 2.5 Flash Lite across all six metrics for our four aspects (Figure 30).

Moreover, Gemini assigns slightly higher values on Depth of Analysis, Novelty Assessment and Constructiveness, while Mimo yields marginally higher scores on Minor Recall. Importantly, Prioritization score shows the smallest divergence between evaluators, confirming that critical-flaw prioritization is effectively evaluator-agnostic.

Despite absolute score offsets, the relative ordering of reviewer types is consistent between the two evaluators. Reviewer2 and DeepReview consistently achieve the highest Constructiveness

(MCS: Gemini 0.629/0.570 vs. Mimo 0.587/0.586) and Critical Recall (Gemini 0.613/0.332 vs. Mimo 0.510/0.279), while CycleReviewer and SEA rank lowest in most dimensions. For Novelty Assessment, both evaluators agree that SEA produces the highest scores (Gemini 0.841, Mimo 0.734). These results demonstrate that the proposed evaluation framework is robust to evaluator LLM substitution. No qualitative conclusion drawn from Gemini is reversed by Mimo, confirming that the observed performance gaps reflect genuine differences in review quality rather than artifacts of the specific evaluator model.

### F Qualitative Analysis & Case Studies

#### F.1 Depth of Analysis

- Case 1: The Evidentiary Collapse — Why Claim-Heavy Reviewers Fail. Paper: NV-Embed: Generalist Text Embeddings from Decoder-Only LLMs (ICLR 2025)

Context. NV-Embed proposes a generalist embedding model built on decoder-only LLMs, introducing (1) a latent attention layer replacing mean-pooling, and (2) a two-stage contrastive instructiontuning pipeline. The model achieves top-1 performance on the MTEB benchmark.

- Table 11: Per-reviewer DoA statistics for NV-Embed (ICLR 2025). Avg GS is the mean grounding score normalized to [0,1] (raw scores 0/1/2 divided by 2)

Reviewer DoAHM Rpremise Avg GS Total Args Premises Claims Human 0.581 0.673 0.500 55 37 18 DeepReview 0.626 0.733 0.546 15 11 4 Reviewer2 0.178 0.152 0.215 46 7 39

Human: Dense Technical Grounding Across All Aspects. Human reviewers produce 37 premises from 55 total arguments (Rpremise = 0.673), with strong grounding quality (GS = 0.500). Premises are component-specific, naming the paper’s actual technical building blocks rather than describing them in generalities:

[GS=1.0, Methodology] “The techniques used are: 1. latent attention layer that achieves better pooling/combination of the last layer embeddings.”

- [GS=1.0, Methodology] “2. a two-stage contrastive instruction tuning method. First step tuning with in-batch negative and hard negative on retrieval datasets, and the second step tuning on non-retrieval datasets.”
- [GS=2.0, Experiment] “The model achieves top performance on the MTEB benchmark.”

Human reviewers also raise critical, component-targeted observations, such as questioning whether a single innovative algorithmic piece exists beyond the two engineering contributions. This combination of evidential support and critical evaluation defines the human gold standard on this paper.

DeepReview: Exceeds Human DoA via Precision-to-Volume Economy. DeepReview produces only 15 arguments, yet 11 are premises (Rpremise = 0.733, exceeding Human’s 0.673), with the highest average grounding score of any reviewer (GS = 0.546). Its premises are architecturally precise:

- [GS=1.0, Methodology] “The authors introduce a novel latent attention layer for pooling embeddings, which outperforms traditional methods like average pooling and <EOS> token embedding.”
- [GS=2.0, Experiment] “Ablation experiments in Table 2 confirm the contribution of the latent attention layer over alternative pooling strategies.”

Reviewer2: Self-Referential Grounding and Volume Inflation. Reviewer2 generates 46 arguments but only 7 qualify as premises, and 4 of those 7 carry grounding score 0 (GS = 0.215, barely above the minimum). The 39 claims are structured section-by-section summaries offering no independent analytical judgment:

[Claim, Methodology] “The paper introduces NV-Embed, a generalist embedding model based on decoder-only large language models (LLMs), aimed at enhancing performance in downstream tasks such as retrieval...”

Even the few substantiated premises restate the paper’s own justifications rather than providing independent evaluations:

[GS =0.0, Methodology] “The introduction of a latent attention layer for sequence pooling is theoretically grounded in dictionary learning concepts and is argued to mitigate information dilution compared to mean pooling or last token pooling.”

[GS=1.0, Experiment] “These results are backed by detailed comparisons and tabulated scores (see Table 14).”

The low grounding quality compounds the low premise ratio: Rpremise = 0.152 and GS = 0.215 together produce DoAHM = 0.178 — a 70% drop from Human’s 0.581.

Key insight. Reviewer2’s failure on this paper illustrates a form of analytical failure beyond pure volume-inflation: even its few “premises” ground claims in the paper’s own assertions rather than in independent analytical observations. The DoA metric correctly penalizes this through the joint harmonic mean of Rpremise and GS — both of which must be high to achieve human-level analytical depth.

- Case 2: The Surface-Level Trap in Practice. Paper: VLAP — Visual-Language Alignment via Pre-trained Word Embeddings (ICLR 2024)

Context. VLAP proposes a lightweight vision-language alignment method that maps visual representations directly into the pre-trained word embedding space of a frozen LLM using a single trainable linear layer, supervised by an optimal transport-based assignment objective and an image captioning loss. The design is deliberately minimal: the LLM and visual encoder remain frozen, with only the linear projection trained.

- Table 12: Per-reviewer DoA statistics for VLAP (ICLR 2024, lK2V2E2MNv). Human row is the mean across 5 individual reviewers. Avg GS normalized to [0,1]; DoAHM = HM(Rpremise, GS).

#### Reviewer DoAHM Rpremise Avg GS Total Premises Claims % Clarity

Human (mean, n=5) 0.567 0.614 0.527 14 8 6 2% Reviewer2 0.551 0.569 0.534 51 29 22 0% CycleReviewer 0.483 0.529 0.444 17 9 8 0% SEA 0.417 0.500 0.357 14 7 7 29% DeepReview 0.263 0.759 0.159 29 22 7 0% TreeReview 0.252 0.194 0.357 36 7 29 29%

Reviewer consensus: Clarity is irrelevant on a paper with a simple design. This is a particularly instructive paper for the surface-level trap because the simplicity of VLAP’s design — a single linear layer, two losses, frozen backbones — leaves almost no room for legitimate reproducibility criticism. Accordingly, Human, DeepReview, CycleReviewer, and Reviewer2 all allocate 0% of their premise budget to Clarity. Instead, they focus exclusively on the paper’s technical formulation and experimental comparisons:

[GS=2.0, Novelty] “Contrastive alignment in ALBEF, BLIP, and the first-stage alignment by BLIP2 includes image-text matching and image-grounded text generation.” (Human_2)

[GS=2.0, Methodology] “An optimal transport-based training objective is proposed to enforce the consistency of word assignments for paired multimodal data. This allows frozen LLMs to ground their word embedding space in visual data.” (Human_3)

[GS=2.0, Methodology] “Using this method, experiments are done on 3 tasks — image captioning, VQA, image-text retrieval — showing the method outperforms existing methods.” (Human_5)

Reviewer2, despite its typical tendency toward low Rpremise, here achieves DoAHM = 0.551 with 29 premises from 51 arguments (Rpremise = 0.569, GS = 0.534) — all directed at Methodology and Experimental Design. CycleReviewer produces 9 premises from 17 arguments with 0% Clarity and an average GS = 0.444.

SEA: Clarity premises praising, not criticizing. SEA allocates 29% to Clarity, matching TreeReview’s proportion. However, its two Clarity premises are positive quality affirmations about the paper, not complaints:

[GS=0.0, Clarity] “The methodology is clearly explained, making it accessible and understandable, which is crucial for reproducibility and further research.”

[GS=0.0, Clarity] “The paper is well-structured, with comprehensive experiments and detailed analyses.”

Although these also carry GS=0 (generic, no specific section cited), they function as strength notes rather than reproducibility criticisms. SEA then pivots to three substantive Experimental premises, maintaining DoAHM = 0.417.

#### TreeReview: reproducibility boilerplate displaces technical analysis. TreeReview produces only

- 7 premises from 36 total arguments (Rpremise = 0.194). Of these 7, 2 (29%) are Clarity premises, both carrying GS=0 and targeting the same generic reproducibility axis:

[GS=0.0, Clarity] “This omission hinders reproducibility and limits the ability of other researchers to build upon the work.”

[GS=0.0, Clarity] “This would greatly enhance the reproducibility of the method.”

Neither statement names a specific omission. The first is a consequence claim (“hinders reproducibility”) with no antecedent in the review — it is evidently the continuation of a claim made in a preceding claim argument, not a self-contained premise. The second is a one-sentence recommendation that names no missing artifact. On a paper whose entire contribution is a single linear layer with two objectives, these statements convey essentially no analytical information.

By contrast, TreeReview’s five non-Clarity premises do engage with the paper’s mechanism (optimal transport alignment, the single linear layer, efficiency), but all at GS=1 with no external literature anchoring. This yields GS = 0.357 and, combined with Rpremise = 0.194, DoAHM = 0.252 — a 55% drop from Human’s 0.567.

Key insight. The surface-level trap on this paper is not caused by a genuinely unclear manuscript: four of the six reviewers independently assess that VLAP requires no Clarity criticism at all. The trap is therefore triggered internally by TreeReview’s reviewing heuristic — a tendency to produce generic reproducibility premises (“hinders reproducibility”, “greatly enhance reproducibility”) regardless of whether the paper’s design warrants them. These boilerplate premises consume 2 of TreeReview’s 7 available premise slots and contribute nothing to the analytical depth of the review.

#### F.2 Novelty Assessment

- Case 1: The Speculative Critique Trap We illustrate the Novelty Score (NS) metric through a representative paper from ICLR 2025 (S85PP4xjFD), which proposes CONPAIR, a contrastive compositional dataset and EVOGEN, a curriculum contrastive learning framework for improving compositional text-to-image (T2I) generation in diffusion models. Table 13 summarises key statistics for the Human and SEA reviewers on this paper.

- Table 13: Per-reviewer novelty statistics for paper S85PP4xjFD (ICLR 2025). Stances: N=novel, NN=not_novel, SW=somewhat_novel, U=unclear. s¯ denotes the raw mean per-claim score on the [−2,+2] scale; the normalized review-level metric is NS = (¯s + 2)/4.

Reviewer s¯ #Claims %N %NN %SW %U Human 1.20 10 50% 30% 0% 20% SEA 1.73 5 80% 0% 20% 0%

Both reviewers correctly identify the paper’s core contributions, specifically, the novel use of contrastive learning in the denoising encoder and the CONPAIR dataset’s value as a hard-negative compositional benchmark. The divergence in s¯ arises not from a disagreement on what is novel, but from how reviewers handle uncertain assessments.

Human reviewer produces 10 claims with a nuanced mixture of stances. Claim C8 are labelled unclear and carry speculative criticism that the metric system cannot verify against any paper in the related-work pool:

[C8, unclear] “The ContraFusion model is compared against other methods using the T2I-CompBench dataset. However, it is trained on the Com-Diff dataset, which likely overlaps noticeably with the T2I-CompBench test set.”

This concern about training/test overlap is plausible but entirely speculative: no related paper in the retrieved pool provides evidence for or against dataset overlap between Com-Diff and T2ICompBench. Consequently, 7 of 11 related-paper comparisons for C8 return UNSUPPORTED or INSUFFICIENT, and the claim’s per-pair score averages to −0.18, substantially dragging down the overall Human s¯. By contrast, the Human’s other well-evidenced claims score positively:

- [C4, novel] “Introducing a contrastive loss in the denoising encoder representation is an interesting idea.” (sk = +2.0)
- [C5, not_novel] “A few other datasets with hard-negative compositional images exist, albeit small. Two examples are COLA and Winoground.” (sk = +2.0)

C5 exemplifies calibrated not_novel assessment: the reviewer correctly names prior datasets (COLA, Winoground) that are confirmed by related-work retrieval, earning a full +2.0 score even though the stance is non-endorsing.

SEA reviewer produces only 5 claims, all either novel or somewhat_novel, with no speculative or unclear claims:

- [C3, novel] “The proposed dataset, CONPAIR, is well-designed with a clear progression from simple to complex compositional scenarios.” (sk = +2.0)
- [C4, novel] “The multi-stage fine-tuning strategy is innovative and addresses the issue of models being overwhelmed by mixed-difficulty data during training.” (sk = +2.0)
- [C5, somewhat_novel] “The use of a VQA model to revise generated captions and ensure text-image alignment is an effective approach.” (sk = +2.0)

Each SEA claim targets a specific, verifiable contribution (curriculum structure, fine-tuning strategy, VQA-based alignment), and none raises unverifiable speculation. As a result, SEA achieves a higher raw mean claim score (s¯ = 1.73 vs. 1.20) despite generating only half the number of claims.

Insight — Speculative Assessment Penalty. This case illustrates a systematic pattern: Human reviewers raise unclear-stance concerns (e.g., suspected data contamination, alignment-quality tradeoffs) that are reasonable in context but impossible to corroborate through paper-pool evidence. The NS metric penalises such claims because they lack verifiable grounding—exactly the behaviour the metric is designed to detect. SEA tends to avoid this failure mode: it focuses on claims that are directly grounded in the methods described in the paper, thus achieving higher evidence-calibrated

novelty scores. This finding does not imply SEA is a “better reviewer”; rather, it shows that SEA’s positively-biased and evidence-anchored claim style is systematically rewarded by the evidencegrounded NS metric, while the richer, more skeptical human review style can be penalized when speculative concerns appear.

- Case 2: Coverage Without Paper 3TGUvHmZ2v from ICML 2025, examines a theoretical paper that characterises the expressivity of fixed-precision Transformer decoders using formal language theory. The paper establishes three main results: (R1) without positional encoding (NoPE), fixed-precision Transformers can recognise only finite and co-finite languages; (R2) adding absolute positional encoding (APE) extends expressibility to cyclic languages; (R3) relaxing parameter bounds further allows recognition of letter-set languages. Table 14 summarises reviewer statistics.

- Table 14: Per-reviewer novelty statistics for paper 3TGUvHmZ2v (ICML 2025). Stances: N=novel, NN=not_novel, SW=somewhat_novel. s¯denotes the raw mean per-claim score on the [−2,+2] scale; the normalized review-level metric is NS = (¯s + 2)/4.

Reviewer s¯ #Claims %N %NN %SW Overlap w/ Human

Human 1.467 5 80% 0% 20% DeepReview 1.500 12 58% 42% 0% 5/5 + 7 new

Human reviewer makes 5 well-targeted claims, all centred on the paper’s three core theoretical results. Three claims earn the maximum per-claim score sk = +2.0 by directly naming the language classes established:

[H-C1, novel] “This paper demonstrates that fixed-precision Transformer decoders without positional encoding are limited to recognizing only finite or co-finite languages...” (sk = +2.0)

[H-C5, novel] “The paper explores the expressive capabilities of Transformer decoders constrained by a fixed-precision setting, such as specific floating-point arithmetic...utilising formal language theory.” (sk = +2.0)

Two claims are penalised to sk = +0.667: H-C2 makes a vague generality (“limited to finite memorization”) rather than naming the class precisely, and H-C4 (somewhat_novel: “adding positional encoding improves expressivity but does not alleviate the main limitations”) lacks the concreteness of specifying cyclic languages. The result is a compact but calibrated review: every claim is evidenced, but some are imprecise enough that the related-work pool only partially supports them.

DeepReview produces 12 claims, 2.4× more, at a nearly identical raw mean claim score (s¯ = 1.500 vs. 1.467). Its first 7 claims cover all of Human’s 5 novelty dimensions, often with greater precision, while claims DR8–DR12 open an entirely new dimension absent from the human review: wellevidenced critical analysis of the paper’s scope limitations.

Covering Human’s novelty dimensions with increased precision:

[DR2, novel] “Introducing absolute positional encoding extends their capabilities to recognizing cyclic languages, while allowing non-finite floating-point values further expands their expressivity to letter-set languages.” (sk = +2.0)

[DR5, novel] “The paper’s focus on constant precision, as opposed to logarithmic precision, is a significant strength, as it reflects the practical constraints of realworld implementations.” (sk = +2.0)

DR2 subsumes H-C4 (somewhat_novel) by naming both upgraded classes precisely, earning sk = +2.0 vs. H-C4’s +0.667. DR5 makes an explicit constant-vs-log-precision distinction that Human’s C5 only gestures towards.

Identifying additional gaps through well-evidenced limitation claims:

[DR10, not_novel] “The paper’s analysis of positional encoding is limited to absolute positional encoding (APE) and no positional encoding (NoPE). It does

not explore relative positional encodings, which are commonly used in modern Transformer architectures.” (sk = +2.0)

[DR12, not_novel] “Finally, the paper’s analysis is primarily theoretical, and it lacks empirical validation of its findings.” (sk = +2.0)

Crucially, these limitation claims are not_novel in stance but still earn sk = +2.0. This is because the novelty metric rewards calibration: the claim that “relative PE is not studied here” is verifiable against the related-work pool (papers using RoPE, ALiBi, etc. do exist, confirming the gap), so the claim is evidence-grounded. By contrast, three of DeepReview’s limitation claims (DR6: APE/cyclic redundancy; DR8: multi-layer not explored; DR11: sinusoidal vs. learned PE) earn sk = 0.0, because either they are internally redundant with other DeepReview claims or the related-work pool cannot confirm they are genuine gaps.

- Table 15: Semantic overlap between Human and DeepReview novelty claims for 3TGUvHmZ2v. DeepReview DR1–DR7 cover all Human claims; DR8–DR12 add new gap analysis.

Human claim DeepReview equivalent(s) sk (DR)

H-C1: finite/co-finite (NoPE) DR1, DR4 +2.0, +2.0 H-C2: “finite memorization” (vague) DR3 (rigorous framework) +2.0 H-C3: regular language subclass (NoPE) DR1, DR7 +2.0, +2.0 H-C4: PE improves but limits remain (vague) DR2 (APE → cyclic/letter-set) +2.0 H-C5: fixed-precision setting (practical) DR5 (constant vs. log precision) +2.0

- — (not in Human) DR8: multi-layer/multi-head gap 0.0
- — (not in Human) DR9: softmax→hardmax gap +2.0
- — (not in Human) DR10: relative PE gap (RoPE) +2.0
- — (not in Human) DR11: sinusoidal vs. learned PE 0.0
- — (not in Human) DR12: no empirical validation +2.0

Insight — Coverage Expansion with Preserved Calibration. This case illustrates a second systematic pattern distinct from Case 1: DeepReview does not achieve a higher raw mean claim score s¯ by avoiding criticism, but by expanding coverage while maintaining the same calibration quality as human reviewers. DeepReview’s extra volume comes in two flavours: (i) precision elaborationnaming specific language classes and precision regimes more exactly than Human; and (ii) gap enumeration—identifying well-evidenced scope limitations (no relative PE, no empirical validation) that human reviewers do not consider. The NS metric rewards both, because both types of claim can be verified against the related-work pool. Thus DeepReview achieves 2.4× the claim count with only ∆¯s = +0.033, demonstrating that LLM reviewers can generate denser novelty assessments that are equally well-calibrated to the literature, not merely more verbose.

- F.3 Flaw Identification & Major Issues Prioritization

- Case 1: Complementary Blind Spots — The Equation-Level Scanner vs. the Practical Assessor. Paper Tv2JDGw920 (ICML 2025 Oral)

Context. This paper proposes a preconditioning-based optimizer for Domain Generalization that leverages the One-Step Generalization Ratio (OSGR) to dynamically balance parameter-wise gradient updates. The paper provides three theoretical analyses (OSGR equalization, PAC-Bayes bound, convergence proof) and extensive experiments across five DG benchmarks. The canonical flaw bank contains 52 entries (after de-duplication: ∼24 unique flaws), of which 34 are valid upon independent verification — all Minor in severity.

- Table 16: Flaw coverage comparison for Tv2JDGw920 (ICML 2025 Oral). After de-duplication, the LLM and Human reviewers each independently identify ∼9 unique valid flaws with zero overlap, demonstrating fully complementary diagnostic profiles.

Reviewer Valid Flaws LLM-only Human-only Shared

Human (3 reviewers) 14 — 9 2 Reviewer2 (LLM) 18 9 — 2

Union 34 9 9 2

What the LLM catches that all Humans miss: systematic equation-level scrutiny. Reviewer2 identifies 9 unique valid flaws that none of the three human reviewers raise. These flaws share a distinctive pattern: they arise from systematically walking through the paper’s mathematical derivations and cross-referencing theoretical claims against their practical implementation.

[LLM-only, Methodology] “In the PAC-Bayes analysis, the prior π is approximated using all data except the current batch. Why is this approximation valid, and what is the impact of this choice on the tightness of the generalization bound?”

This flaw targets the data-dependent prior in Theorem 3.6 (Appendix C.3.1), where the paper states the prior is “approximated with stochastic gradient descent using all data excluding the current mini-batch.” Classical PAC-Bayes theory requires the prior to be chosen independently of training data. While modern PAC-Bayes extensions accommodate data-dependent priors, the paper does not formally invoke such extensions, leaving a gap in the theoretical justification.

[LLM-only, Methodology] “In Corollary 3.2, the preconditioning factor pj is derived under the assumption that gradients are independent across parameters. However, in practice, gradients are often highly correlated.”

Verifying this requires tracing from Equation (48) through (50) in Appendix C.2, where the factorization E(p ⊙ ε · ε′) = j pj · σ2(θj) implicitly assumes zero cross-parameter covariance — a standard but unverified assumption in deep networks with shared activations across layers.

[LLM-only, Clarity] “The paper frames the use of OSGR in the optimizer as novel, [but] it is not entirely clear how this differs from existing adaptive learning rate methods. For instance, Adam already adapts learning rates based on gradient magnitudes.”

Table 1 in the paper decomposes the proposed optimizer and Adam into convergence and alignment terms, but the structural similarity between the two requires parsing dense notation to appreciate the distinction — precisely the kind of close reading that time-constrained reviewers may skip.

What Humans catch that the LLM misses: claim–evidence calibration and field norms. Human reviewers identify 9 unique valid flaws that Reviewer2 entirely overlooks. These flaws target gaps between what the paper claims and what the evidence supports — a form of practical wisdom rooted in familiarity with community expectations.

[Human-only, Methodology] “The paper claims [the optimizer] promotes domain-invariant features, but it doesn’t directly evaluate this claim by examining feature representations. For example, some DG papers use metrics like center divergence between domain features.” (Human_2)

The paper provides qualitative visualizations showing class separation across domains, but no quantitative domain-invariance metric (e.g., MMD, center divergence) — a standard evaluation in the DG literature that the LLM does not demand.

[Human-only, Methodology] “The claim that uniformly distributed OSGR across parameters indicates better generalization...is stated as a conjecture rather than a theorem, and while intuitively supported, it’s not rigorously demonstrated.” (Human_2)

The paper transparently labels this as a “Conjecture” (§3.2) and provides partial support via Jensen’s inequality (Appendix C.3.2). However, only a human reviewer flags the intellectual-honesty gap

between a conjecture and a theorem — the LLM accepts the conjecture’s supporting evidence without questioning its formal status.

Where the LLM hallucinates: asserting absence of content that exists. Reviewer2 also generates several flaws that are directly contradicted by the paper. The most striking: the LLM claims “the paper does not adequately explain the computational cost relative to simpler optimizers,” despite the paper explicitly reporting training times across multiple iteration budgets (e.g., proposed method: 4,292s vs. Adam: 5,443s at 5K iterations). Similarly, the LLM asserts the paper “lacks qualitative insights (e.g., feature maps, attention weights),” overlooking multiple visualizations already present in the manuscript. These fabrications follow a consistent pattern: the LLM evaluates flaw claims in isolation without cross-referencing the manuscript’s actual content.

Key insight. This case demonstrates that LLM and human reviewers operate as complementary diagnostic instruments with near-zero overlap. The LLM excels at systematic, equation-level verification

— surfacing implicit assumptions (gradient independence, data-dependent priors) and theory-practice gaps that require cross-referencing proofs against algorithms. Human reviewers excel at claim– evidence calibration — demanding quantitative backing for qualitative claims and recognizing when a conjecture substitutes for a theorem. Neither perspective subsumes the other: the union of their flaw sets produces substantially broader diagnostic coverage than either alone, while both are susceptible to distinct failure modes (LLM: content hallucination; Human: limited equation-level scrutiny under time pressure).

- Case 2: The Diagnostic Volume Advantage — Broader Coverage Through Exhaustive Scanning. Paper aapUBU9U0D (ICLR 2025)

Context. This paper proposes an iterative data augmentation pipeline for fine-tuning LLMs on Operations Research (OR) tasks. The method generates synthetic optimization problems via an evolutionary strategy, validates them through four LLM-based checkers, and fine-tunes LLaMA-3-8B on the resulting corpus. Performance is evaluated on three OR benchmarks. The canonical flaw bank contains 28 entries, all valid upon verification, contributed equally by human reviewers (14 flaws) and Reviewer2 (14 flaws).

- Table 17: Flaw coverage and aspect distribution for aapUBU9U0D (ICLR 2025). Despite equal flaw counts, the two reviewer groups cover substantially different aspects of the paper, with Reviewer2 providing broader topical coverage across all four macro-categories.

Reviewer Valid Methodology Experimental Clarity Applicability

Human (4 rev.) 14 3 7 2 2 Reviewer2 (LLM) 14 4 4 1 5

Union 28 7 11 3 7

Human reviewers: deep but concentrated coverage. The four human reviewers collectively produce 14 valid flaws, with a strong concentration on Experimental Design (7 of 14, 50%). Their critiques are precise and field-specific, reflecting deep familiarity with OR benchmarking conventions:

[Human, Experimental Design] “No comparison with traditional OR solvers is provided. The paper claims to advance ‘automation of decision-making’ but never shows whether LLM-based modeling is competitive with established OR methods.”

[Human, Experimental Design] “The results should clarify whether COPT or GUROBI is used for evaluation, and whether baselines were given equal solver access.”

These flaws target the paper’s core experimental validity — precisely the type of domain-expert criticism that requires knowledge of solver ecosystems and OR benchmarking norms. However, human coverage leaves notable gaps: only 2 of 14 flaws address Applicability and Limitations, and no human reviewer questions the pipeline’s scalability to large-scale industrial problems or its dependence on English-only data.

Reviewer2: broader aspect coverage with systematic gap enumeration. Reviewer2 produces an equal number of valid flaws (14), but distributes them more evenly across aspect categories. Notably,

5 of its 14 flaws target Applicability and Limitations — a category that human reviewers largely neglect:

[LLM-only, Applicability] “The scalability of the pipeline to large-scale optimization problems with thousands of variables and constraints is not evaluated.”

[LLM-only, Applicability] “The pipeline’s applicability beyond linear and mixed-integer programming — for example, to nonlinear or stochastic optimization — remains unverified.”

[LLM-only, Methodology] “The four validation checkers are entirely LLM-prompt-based. No formal algorithmic specification, error bounds, or coverage guarantees are provided for the checking pipeline.”

These flaws arise from the LLM’s systematic scanning pattern: rather than evaluating only the paper’s explicit claims, Reviewer2 probes the boundaries of the contribution — what is not tested, not formalized, and not generalized. This “boundary probing” behavior is precisely what drives Reviewer2’s elevated recall at the macro level: by exhaustively questioning scope and applicability, the LLM surfaces valid concerns that time-constrained human reviewers deprioritize in favor of core experimental scrutiny.

The aspect complementarity pattern. When the two flaw sets are merged, the union covers all four macro-categories substantially: Methodology (7), Experimental Design (11), Clarity (3), and Applicability (7). Neither reviewer group alone achieves this breadth. Human reviewers anchor the evaluation in domain-specific experimental rigor, while Reviewer2 extends coverage into methodological formalization and scope limitations.

Key insight. This case concretizes the volume-to-coverage advantage observed at the aggregate level. Reviewer2’s exhaustive scanning style does not merely produce more flaws — it produces flaws in different diagnostic categories than human reviewers, systematically covering scope limitations and methodological formalization gaps that humans deprioritize. The resulting union of human and LLM flaw sets achieves broader aspect coverage than either alone, reinforcing the practical value of LLM-augmented peer review as a diagnostic broadening mechanism rather than a replacement for human expertise.

#### F.4 Multi-dimensional Constructiveness

Case: The Actionability Gap — From Diagnosis to Prescription. Paper: GenColor: A DiffusionBased Framework for Color Enhancement in Digital Photography (NeurIPS 2025)

Context. GenColor proposes a no-reference color enhancement pipeline consisting of three learned components: (1) a diffusion-based Color Generation Module, (2) a Texture Preservation Module, and (3) a post-processing Global Adjustment step. The system is trained on the authors’ proprietary ARTISAN dataset and evaluated via user studies and no-reference image quality metrics.

- Table 18: Per-system Constructiveness scores for GenColor (NeurIPS 2025). All dimensions are on a [0,2] raw scale.

System MCS D1:Act D2:Spec D3:Just D4:Sol D5:Tone ARCs Human 0.488 0.721 1.767 0.372 0.326 1.698 43 DeepReview 0.724 1.588 2.000 0.882 1.059 1.706 17 Reviewer2 0.800∗ 1.000 2.000 2.000 1.000 2.000 24 CycleReviewer 0.400 0.833 1.833 0.167 0.167 1.000 6 TreeReview 0.434 0.793 1.517 0.241 0.207 1.586 29

∗Reviewer2’s MCS is inflated by near-perfect D3 scores (verbose observation-only summaries; see text).

Human: Precise Diagnosis, Absent Prescription. Human reviewers produce 43 ARCs with respectable specificity (D2 = 1.767), demonstrating solid familiarity with the paper’s technical details. However, D4 (Solution) averages only 0.326: the majority of human ARCs identify a problem but stop short of prescribing a remedy. Representative human weakness ARCs illustrate this gatekeeping pattern:

[D1 =1, D4= 0] “The method’s near-deterministic nature raises concerns about user control and the ability to capture personalized styles.”

[D1=1, D4=0] “The proposed method has a relatively long runtime compared to lightweight comparison models.”

[D1=1, D4=0] “Previous methods trained on the proposed dataset perform worse, questioning the dataset’s general usefulness.”

Each statement correctly identifies a real limitation of GenColor:determinism, efficiency, dataset scope but offers no concrete path forward for the authors to address it. This is the hallmark of diagnostic gatekeeping: the review functions as a verdict, not a guide.

DeepReview: Prescriptive Constructiveness at Scale. DeepReview produces only 17 ARCs, yet achieves D1 = 1.588 and D4 = 1.059, both substantially above Human. Critically, six of its ARCs reach the maximum solution score D4=2 — meaning the feedback specifies not only what is missing but how to address it:

[D1=2, D4=2] “Include comprehensive ablation studies by systematically removing or modifying components to evaluate their individual contribution.”

[D1=2, D4=2] “Provide a detailed analysis of the computational cost of each component and the overall pipeline, including training time, inference time, and memory requirements.”

[D1=2, D4=2] “Clearly articulate the novelty of the specific combination and modifications of existing components, detailing unique aspects such as the training regime, degradation scheme, and weight blending strategy.”

Compared to the human weakness ARCs above, these comments address the same underlying concerns (ablation coverage, efficiency, novelty justification) but close the loop by telling the authors exactly what artifact to produce or section to revise.

Reviewer2: High MCS via Justification Inflation, Not Solutions. Reviewer2 achieves the highest raw MCS on this paper (0.800), driven entirely by near-perfect D3 scores (D3 = 2.000). Its 24 ARCs are detailed, well-grounded observations — but they are observations, not directives. D1 averages only 1.000 and D4 averages 1.000, because every ARC points to a general direction at best (“no systematic analysis is provided”, “no theoretical basis is provided”) without specifying an implementable fix. This reveals a limitation of MCS as a holistic score: high D3 can mask absent D4. The D4 sub-score surfaces this asymmetry directly.

CycleReviewer and TreeReview: Vagueness and Volume. CycleReviewer contributes 6 ARCs with D4 = 0.167 and D3 = 0.167 — vague, unsubstantiated comments such as:

[D1=1, D4=0] “The method might struggle with images exhibiting complex textures or patterns.”

TreeReview produces 29 ARCs but maintains D4 = 0.207; its highest-scoring feedback requests reproducibility details (training parameters, hyperparameter selection) rather than engaging with the paper’s core design choices.

Key insight. This case illustrates the central behavioral gap in constructiveness: even when human reviewers are technically perceptive (high D2, moderate D3), they default to problem identification without resolution. DeepReview’s architectural orientation toward explicit remediation — reflected in D4 exceeding human baseline by +0.733 and D1 by +0.867 — demonstrates that high constructiveness is not a matter of writing more, but of closing the feedback loop from critique to actionable prescription.

### G Limitations

While PRISM provides a rigorous, multi-dimensional benchmarking framework for automated peer review, we acknowledge several limitations that highlight avenues for future research.

Domain Generalization. Our dataset comprises 1,000 manuscripts exclusively from premier machine learning and representation learning venues (ICLR, ICML, NeurIPS). The structural norms, citation densities, and evaluation criteria in these venues differ from those in other scientific disciplines (e.g., clinical medicine, humanities, or pure mathematics). Consequently, the current instantiation of PRISM may require recalibration before deployment in non-ML domains.

LLM Dependency: Hallucination, Prompt Sensitivity, and Judge Bias. A foundational premise of PRISM is delegating complex tasks—such as text atomization, fact-finding, and scoring—to frontier LLMs. It is well-documented that LLMs are heavily susceptible to hallucinations (fabricating non-existent critiques or citations) and prompt sensitivity (where minor structural variations in instructions yield divergent outputs). To actively mitigate these vulnerabilities, PRISM strictly departs from monolithic, single-prompt evaluation. By decomposing the framework into constrained, multi-phase pipelines and enforcing deterministic decoding, we significantly restrict the generation space and effectively filter out hallucinated noise. Nevertheless, this dependency introduces residual bottlenecks: atomizing isolated sentences inherently risks context loss, fact-finding remains bounded by the coverage of external retrieval APIs, and models acting as judges may still retain subtle internal priors for specific rhetorical styles. In this work, our primary evaluation pipeline is instantiated using Gemini 2.5 Flash Lite. While we conducted preliminary robustness checks with an alternative model (Xiaomi MiMo V2.5 Pro) on a data subsample to confirm baseline metric stability, this single-judge dependency means we cannot fully rule out model-specific evaluation biases. Future work must not only develop robust uncertainty quantification to prevent edge-case extraction errors from cascading into downstream metrics, but also conduct comprehensive multi-judge studies across diverse LLM families to fully isolate and eliminate judge-specific priors.

