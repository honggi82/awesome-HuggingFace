# arXiv:2602.13964v3[cs.CL]27Feb2026

[Figure 1]

[Figure 2]

[Figure 3]

## HLE-Verified: A Systematic Verification and Structured Revision of Humanity’s Last Exam

Weiqi Zhai1∗, Zhihai Wang2∗, Jinghang Wang1, Boyu Yang1, Xiaogang Li1, Xander Xu1, Bohan Wang2, Peng Wang2, Xingzhe Wu2, Anfeng Li2, Qiyuan Feng1, Yuhao Zhou1, Shoulin Han1, Wenjie Luo1, Yiyuan Li1, Yaxuan Wang1, Ruixian Luo1, Guojie Lin1, Peiyao Xiao1, Chengliang Xu1, Ben Wang1, Zeyu Wang1, Zichao Chen1, Jianan Ye1, Yijie Hu1, Jialong Chen1, Zongwen Shen1, Yuliang Xu1, An Yang2, Bowen Yu2, Dayiheng Liu2, Junyang Lin2, Hu Wei1, Que Shen2†, Bing Zhao1†

1Alibaba Group 2Qwen Team, Alibaba Group

* Equal Contribution † Corresponding Authors: shenque.sq@alibaba-inc.com, xiongdao@alibaba-inc.com

### Abstract

Humanity’s Last Exam (HLE) has become a widely used benchmark for evaluating frontier large language models on challenging, multi-domain questions. However, subsequent community-led analyses have raised concerns that HLE contains a non-trivial number of noisy items (e.g., ambiguous statements, incorrect answers, or mismatched rationales), which can systematically bias evaluation results and distort cross-model comparisons. To address this challenge, we introduce HLE-Verified, a verified and revised version of HLE accompanied by a transparent, component-wise verification protocol and fine-grained error taxonomy. Our construction follows a two-stage validation-and-repair workflow resulting in a unified certified benchmark. In Stage I, each item is subjected to binary validation on the problem and final answer dimensions (with rationale used as an auxiliary consistency signal), combining domain-expert review and model-based cross-checks. This stage yields 668 items verified as correct. In Stage II, items identified as flawed but fixable are systematically revised under strict constraints that preserve the original evaluation intent, through dual independent expert repairs, model-assisted consistency auditing, and final expert adjudication, resulting in 1,143 revised-and-certified items. The remaining 689 items are released as a documented uncertain set with explicit uncertainty sources and required expertise tags for future community refinement. We compare the performance of eight state-of-the-art language models on HLE and HLEVerified, observing an average absolute accuracy gain of 7–10 percentage points on HLE-Verified. The improvement is particularly pronounced on items where the original HLE problem statement and/or reference answer is erroneous: on this subset, the models achieve an average accuracy increase of 30–40 percentage points. Moreover, our analyses indicate a strong association between model confidence and the presence of errors in the problem statement or reference answer, providing evidence for the effectiveness of our revisions. Overall, HLE-Verified improves HLE-style evaluations by reducing annotation noise and enabling more faithful measurements of model capabilities. Data is available at: https://huggingface.co/datasets/skylenage/HLE-Verified.

### 1 Introduction

As frontier language models (LMs) continue to advance rapidly, there is increasing demand for evaluation benchmarks that are simultaneously difficult, broad in disciplinary coverage, and resistant to saturation. High-difficulty benchmarks play a central role in tracking progress, validating capability claims, and shaping scientific discourse about model reasoning abilities. Against this backdrop, the Humanity’s Last Exam (HLE) has emerged as a widely adopted benchmark for evaluating model performance on challenging questions across mathematics, science, engineering, and the humanities(Center for AI Safety et al., 2026). Results on HLE have been cited to support claims regarding reasoning depth, generalization capacity, and reliability of state-of-the-art models.

However, as benchmark difficulty increases, so does the importance of annotation integrity. When many items lie near a model’s decision boundary, even minor inconsistencies in problem statements or

reference answers can disproportionately influence aggregate metrics. Recent independent audits and community-led reviews have highlighted substantial noise in the HLE dataset, including ambiguous or poorly defined question statements, erroneous reference answers, and inconsistencies between rationales and final answers (White, 2025; lhl, 2026; red, 2025). These analyses suggest that a portion of measured model performance on HLE may reflect annotation artifacts rather than genuine capability differences.

This issue highlights a broader methodological question: how should high-difficulty benchmarks be systematically verified and maintained after deployment? While benchmark construction traditionally emphasizes task design and data collection, comparatively less attention has been devoted to post-release verification, structured auditing, and transparent revision protocols. For benchmarks that inform scientific claims and cross-model comparisons, the absence of systematic validation can undermine interpretability, reproducibility, and measurement reliability.

To address this reliability challenge, we developed HLE-Verified—a rigorously validated and partially revised version of HLE built under a structured two-stage validation and revision protocol. This approach employs a component-level verification framework, treating problem statements and final answers as primary targets for correctness assessment while utilizing rationale as secondary signals to detect internal contradictions or underdefined specifications. Under this protocol, HLE-Verified is organized into three disjoint subsets: verified gold items, revised items corrected under preserved evaluation objectives, and uncertain items whose validity cannot be conclusively determined.

We position HLE-Verified as dataset infrastructure: not a new benchmark task, but a methodological reliability enhancement that enables more rigorous, interpretable, and reproducible model comparisons. Furthermore, we provide an empirical evaluation of how verification and revision affect downstream model measurements, and we release structured metadata that supports continued community-driven refinement.

#### Contributions.

- • We release HLE-Verified, a verified and revised version of HLE constructed via a transparent component-wise verification protocol and fine-grained error taxonomy, comprising 668 verified items, 1,143 revised-and-verified items, and a 689-item documented uncertain set for structured community refinement.
- • We introduce a systematic two-stage verification-and-revision framework for post-release benchmark auditing.
- • We benchmark eight state-of-the-art LLMs on HLE vs. HLE-Verified, demonstrating that verification materially alters measured performance (an average +7–10 accuracy points overall and

+30–40 points on items with erroneous problems/answers), and we analyze the relationship between model confidence and problem/answer errors.

### 2 Background

#### 2.1 Why benchmark errors matter

Benchmarks serve as the primary measurement interface between model development and scientific claims about capability. Let a benchmark contain a fraction ρ of flawed items. Even for small ρ, aggregate metrics such as accuracy or pass@k can be meaningfully distorted, particularly when flawed items are not randomly distributed. If such items are concentrated in specific domains, reasoning patterns, or difficulty strata, they may introduce systematic measurement bias (Liang et al., 2022; Gema et al., 2025), favoring certain model families while penalizing others due to differences in training exposure, reasoning style, or calibration behavior.

Beyond aggregate accuracy, benchmark flaws pose a deeper challenge to interpretability. Many analyses assume a well-defined correspondence between correctness and model confidence, particularly in calibration- or uncertainty-aware evaluations (Guo et al., 2017; Kadavath et al., 2022). When items are ill-posed, contain incorrect answer keys, or admit multiple valid interpretations, this correspondence becomes ill-defined: models may be penalized for producing valid but unanticipated answers, or rewarded for matching erroneous supervision. In such cases, apparent calibration error or ranking instability may reflect properties of benchmark noise rather than intrinsic model behavior (Chao et al., 2024).

In practice, benchmark flaws commonly manifest in several recurring forms: (1) underspecified or ambiguous problem statements Min et al. (2020); (2) incorrect or mismatched answer keys (e.g., unit or format inconsistencies); (3) internal inconsistencies between the provided rationale and the final answer; and (4) reliance on contested facts, implicit conventions, or unstated assumptions. Notably, these failure

modes often arise in different components of an evaluation item—the problem statement, the final answer, and the accompanying rationale—suggesting that item validity is multi-dimensional rather than binary.

#### 2.2 HLE as an evaluation substrate

Humanity’s Last Exam (HLE) has emerged as a widely used benchmark for evaluating frontier language models on challenging, multi-domain questions. In practice, HLE is often reduced to a single scalar performance metric, typically accuracy, occasionally augmented with rationale- or chain-of-thoughtbased prompting. Implicit in this usage is the assumption that annotation noise is sufficiently small and uniformly distributed such that aggregate comparisons remain meaningful.

However, prior work has demonstrated that large-scale benchmarks may contain systematic annotation errors or unstable evaluation artifacts that meaningfully affect model rankings (Gema et al., 2025; Prathifkumar et al., 2025). To date, such concerns have not been systematically quantified for HLE, nor has their impact on evaluation outcomes been rigorously characterized. Consequently, it remains unclear to what extent observed performance differences on HLE reflect genuine capability gaps versus sensitivity to benchmark defects.

This gap motivates a principled re-examination of HLE as an evaluation substrate. Rather than treating benchmark errors as unstructured noise, our work explicitly characterizes item validity, distinguishes different sources of error and uncertainty, and assesses how these factors affect evaluation results. HLEVerified operationalizes this perspective by making correctness, revision status, and epistemic uncertainty explicit at the dataset level, thereby enabling more reliable and interpretable model comparisons.

### 3 Dataset Verification Process and Methods

This section describes the construction protocol of HLE-Verified, including (i) the end-to-end item pipeline, (ii) the operational definitions of verified (gold), revised, and uncertain items, and (iii) the structured annotation schema used to record error types, revision actions, and item-level epistemic status. The protocol is designed to make benchmark validity explicit, auditable, and component-wise traceable, while preserving the original evaluation intent of HLE wherever possible.

#### 3.1 Overview of the pipeline

Starting from the original HLE collection (2,500 items), we construct a single consolidated benchmark release, HLE-Verified, through a structured two-stage process. Stage I performs componentwise binary verification and produces a highconfidence gold subset. Stage II conducts systematic repair for items judged flawed but repairable, followed by re-verification. Items that remain indeterminate after these procedures are retained as an explicitly documented uncertain subset rather than discarded.

Each item is decomposed into three annotatable components: (i) the problem (statement plus image, if present), (ii) the final answer, and (iii) the rationale (reference solution, when available).

The problem and final answer are primary objects of correctness assessment, defining the evaluand and grading target, while the rationale serves as diagnostic support for detecting inconsistencies, missing assumptions, or explanation defects.

[Figure 4]

Fig. 1: Structural composition of HLE-Verified.

The resulting release comprises three disjoint subsets (Fig 1):

- 1. Gold subset (668 items): validated without modification.
- 2. Revision subset (1,143 items): corrected and re-verified under preserved evaluation objectives.
- 3. Uncertain subset (689 items): items whose validity remains indeterminate under available expertise and evidence.

[Figure 5]

Fig. 2: HLE Revision Stage I. High-Difficulty Problem Validity Verification & Golden Subset Construction

Across both stages, we record structured item-level metadata including component-wise validity labels, error-type annotations, revision traces, adjudication notes, and uncertainty descriptors. These fields enable transparent accounting and downstream analyses stratified by defect type and epistemic status.

#### 3.2 Stage I: component-wise verification

- Goal. Stage I aims to determine whether an item is usable as originally released, without performing substantive content edits. The outcome of Stage I is a component-wise binary validity record and a high-confidence gold subset. Operational definition of validity. For each item component, validity is defined as follows:

- • Problem validity: the statement (and image, if present) is well-posed, self-consistent, and sufficiently specified for a unique or properly qualified solution; assumptions and conventions needed for solving are either explicit or standard within the domain.
- • Answer validity: the provided final answer is correct with respect to the problem specification, including units, format, and acceptable equivalence classes.
- • Rationale validity: the reference solution is mathematically/logically sound, internally consistent, and compatible with the final answer (when the rationale is intended as an authoritative reference).

Multi-source verification and adjudication. Stage I integrates three sources of evidence in a serial–hybrid workflow:

- 1. External domain-expert screening. Independent subject-matter reviewers assess problem, answer, and rationale, providing component-wise binary judgments and concise notes.
- 2. Model-assisted replication checks (pass@8). We invoke multiple frontier multimodal solvers under pass@8 sampling to generate independent solution attempts. Extracted final answers are normalized and compared against the reference answer under a fixed equivalence protocol (numeric tolerance, format normalization, semantic equivalence where applicable). This step provides reproducibility evidence and highlights items exhibiting extreme human–model disagreement.
- 3. Internal expert adjudication. Internal reviewers synthesize supplier judgments and modelassisted evidence to make conservative inclusion decisions. Items enter the gold subset only when both problem and answer are judged unproblematic and no high-risk ambiguity is identified.

[Figure 6]

Fig. 3: HLE Revision Stage II. Systematic Revision of Challenging Questions

- Stage I outcome. Stage I yields the gold (verified) subset of 668 items that are retained without modification. All remaining items are carried forward either to Stage II for repair (if flawed but repairable) or to the uncertain pool (if indeterminate or high-risk).

3.3 Stage II: systematic revision and re-verification

Goal. Stage II targets items that are judged flawed but repairable under the constraint that the original evaluation objective and reasoning target must be preserved. In other words, revisions are corrective rather than creative: they aim to restore well-posedness and correctness without altering what the item is intended to test.

Scope and boundary. To ensure that revision results are objectively reviewable, Stage II focuses on domains where correctness can be reliably adjudicated and independently reproduced (mathematics, physics, chemistry, biomedicine, and computer science in our current release). Items in domains with weak verification boundaries or subjective conventions are handled conservatively (routed to the uncertain subset).

Revision workflow. For each candidate item, Stage II generates independent repair proposals followed by expert convergence:

- 1. Independent supplier repairs (two-track). Two independent expert teams propose corrective edits following a standardized sequence: Problem Fix → Solution Fix → Answer Fix, each accompanied by change notes. Items requiring substantive alteration of evaluation intent are marked non-repairable.
- 2. Model-assisted auxiliary proposals. Multi-model sampling may generate additional repair candidates and stability checks. Model outputs serve as auxiliary evidence and do not replace expert adjudication.
- 3. Final expert adjudication. Internal experts select or synthesize a canonical repaired version under the principles of objective preservation, correctness, and minimal necessary edits. Items remaining ambiguous or unverifiable are routed to the uncertain subset.

- Stage II outcome. Stage II yields the revision subset of 1,143 items that are corrected and re-verified as suitable for evaluation. Each revised item includes structured revision metadata, including which components were changed, what error types were addressed, and brief adjudication notes sufficient for auditing.

[Figure 7]

Fig. 4: HLE Component-wise Defect Taxonomy

#### 3.4 Uncertain subset: epistemic status and documentation

After verification and revision, 689 items remain uncertain. Rather than discarding them, we retain these items as an explicit epistemic category, reflecting cases where validity cannot be established with sufficient confidence. An item is assigned to the uncertain subset when resolution would require non-standard assumptions, authoritative external references, unresolved expert disagreement, or domain knowledge beyond the current verification scope.

Each uncertain item includes structured documentation:

- • an uncertainty source label,
- • and a required expertise tag indicating the type of specialist input needed for resolution.

This design separates capability limits from benchmark indeterminacy and enables principled community refinement.

#### 3.5 Component-wise annotation framework and defect taxonomy

All HLE-Verified items are annotated at the component level to enable localized defect attribution, structured revision tracking, and reproducible auditing. Each item is decomposed into three semantically distinct components: the problem statement (Problem), the reference rationale/solution (Rationale), and the final answer (Answer). Defect labels and revision records are assigned to the specific component in which the issue originates, rather than treating item validity as a monolithic binary property.

Each defect category corresponds to a violation of component-level validity constraints, reflecting that item validity is structured and multi-dimensional. We define a taxonomy comprising 19 categories in total, organized as 5 problem-level errors, 10 rationale-level errors, and 4 answer-level errors. The taxonomy serves as the unified basis for (i) statistical reporting of defect prevalence, (ii) analysis of repair patterns in the revised subset, and (iii) public release of structured item-level metadata.

Problem-level defects. Problem-level defects arise from flaws in task specification that compromise interpretability, solvability, or semantic faithfulness.

- • (Q1) Semantic Error. The statement is ambiguous, contradictory, or underspecified, admitting multiple interpretations.
- • (Q2) Knowledge Error. The statement contains incorrect factual premises or misuse of established domain knowledge.

- • (Q3) Missing Information. Essential constraints or assumptions required for solvability or uniqueness are omitted.
- • (Q4) Theoretical Invalidity. The statement is invalid under accepted theory, supported by explicit corrective evidence.
- • (Q5) Format Semantic Error (Problem). Notation, LaTeX, or terminology defects that distort or obscure intended meaning.

Rationale-level defects. Rationale-level defects characterize failures in the reference reasoning chain provided by the benchmark. These categories apply to the authoritative solution text rather than to model-generated reasoning.

- • (S1) Non-Redundancy Violation. Redundant reasoning steps that reduce minimality or auditability.
- • (S2) Circular Reasoning. The conclusion depends on itself directly or indirectly.
- • (S3) Empirical Soundness Violation. Steps contradict established facts or accepted knowledge.
- • (S4) Step Inconsistency. Logical conflicts among intermediate steps.
- • (S5) Domain Misapplication. Misuse of rules or theorems outside their valid scope.
- • (S6) Overconfidence Bias. Incorrect content stated with unjustified certainty.
- • (S7) Missing Prerequisite. Critical assumptions or conditions omitted while proceeding with derivation.
- • (S8) Deceptive Similarity. Superficially plausible reasoning containing subtle structural corruption.
- • (S9) Multi-Solution Inconsistency. Inconsistency across legitimate solution paths or case analyses.
- • (S10) Format Semantic Error (Rationale). Symbol, LaTeX, unit, or terminology issues impairing verifiability or semantic alignment.

Answer-level defects. Answer-level defects concern correctness, completeness, and verifiability of the final output.

- • (A1) Incorrect Answer. The final answer is inconsistent with the correct solution (e.g., sign/value/boolean inversion).
- • (A2) Incomplete Answer. Required cases, qualifiers, or multi-part conclusions are missing.
- • (A3) Ambiguous / Ill-defined Answer. The answer is non-verifiable due to vagueness or mismatch with required output form.
- • (A4) Format Semantic Error (Answer). Expression-level defects (e.g., LaTeX errors, incorrect symbols, unit inconsistencies) that impair interpretability without necessarily altering conceptual content.

Revision and epistemic annotations. For items in the revised subset, we record component-level modification indicators (problem_fix, solution_fix, answer_fix) together with concise change notes. Revisions are performed under the constraint that the original evaluation objective and reasoning target are preserved; corrections restore validity and internal consistency without redefining the underlying capability being assessed.

Each item additionally includes an epistemic status label (verified, revised, or uncertain) and structured uncertainty descriptors where applicable. These fields make correction history, defect attribution, and indeterminacy explicit at the dataset level, enabling transparent auditing and stratified evaluation analyses.

### 4 Dataset Statistical Analysis

We report a statistical analysis of the problematic subset of HLE-Verified, defined as items that did not enter the gold subset in Stage I. Our analysis focuses on (i) component-wise annotation outcomes (Problem, Answer, Rationale), and (ii) cross-domain variation in defect patterns. We further summarize recurring failure modes to contextualize the distributional findings.

[Figure 8]

- Fig. 5: Overall annotation outcomes on the problematic data of HLE. We report the counts of three labels—valid (1), invalid (0), and uncertain—for each annotatable component: Problem, Answer, and Rationale. The results show clear component-wise differences, with substantially more invalid/uncertain cases in Answer and especially Rationale than in Problem.

[Figure 9]

- Fig. 6: Annotation outcome distribution across subject categories. For each subject category, we visualize the percentage of valid (1), invalid (0), and uncertain labels for the Problem, Answer, and Rationale components. The label composition varies by domain, while the Rationale component consistently exhibits the lowest validity and the highest uncertainty/invalidity across categories.

#### 4.1 Component-wise annotation outcomes

Each HLE instance is decomposed into three annotatable components—Problem, Answer, and Rationale—with labels valid (1), invalid (0), or uncertain. Figure 5 presents the aggregate distribution across the problematic subset.

Across components, the Problem field exhibits the highest reliability. The majority of problem statements are structurally valid, and explicit structural errors constitute only a small fraction of cases. This indicates that most instances are well-posed at the task-description level. In contrast, reliability decreases at the Answer level. Only about half of the answers are fully valid, with a substantial portion either incorrect or difficult to verify. This suggests that correctness mismatches and solution inconsistencies are a major source of dataset noise. The most pronounced degradation appears in the Rationale component.

Invalid rationales outnumber valid ones, indicating that reasoning chains frequently contain logical gaps, unsupported inferences, or incorrect derivations—even when the corresponding problem statement appears acceptable.

Importantly, the uncertain label constitutes a non-trivial fraction across all components. These cases typically arise from underspecified assumptions, ambiguous notation, incomplete derivations, or dependence on implicit conventions. The prevalence of uncertainty underscores that problematicness in HLE is not limited to outright incorrectness but often involves epistemic indeterminacy.

#### 4.2 Cross-domain variation

Figure 6 further disaggregates component-wise validity by subject category, revealing pronounced cross-domain differences.

For the Problem component, formal scientific domains exhibit markedly higher structural reliability. Math and Biology/Medicine both exceed 92% validity, while Chemistry remains comparatively strong at 73.3%. In contrast, Physics shows only 30.0% valid problem statements, and Engineering, Humanities/Social Science, and Other fall further to the 22–27% range. Notably, in these lower-validity domains, the majority of remaining cases are labeled uncertain (roughly 67–77%), rather than explicitly invalid, indicating verification ambiguity rather than structural error.

At the Answer level, domain gaps persist. Computer Science/AI (65.1%) and Chemistry (67.3%) achieve the highest answer validity rates. Math (59.6%) and Biology/Medicine (58.6%) remain above 50%, but both exhibit substantial invalid proportions (36.8% and 38.2%, respectively), suggesting that answer correctness in these domains is typically decidable and frequently incorrect rather than ambiguous. By contrast, Physics, Engineering, Humanities/Social Science, and Other remain below 30% valid, with uncertainty again dominating (approximately two-thirds to three-quarters of cases).

The divergence becomes most pronounced for the Rationale component. In Math and Biology/Medicine, invalid rationales dominate (65.9% and 66.4%, respectively), and Chemistry reaches an even higher invalid rate of 70.3%. These domains therefore display explicit logical or derivational errors even when problems are well-formed. In contrast, Engineering, Humanities/Social Science, and Other show extremely high uncertainty rates (approximately 73–77%) with virtually no explicit invalidation, reflecting verification difficulty rather than clear logical contradiction. Physics occupies an intermediate position, with high uncertainty (67.0%) but a non-trivial invalid share (10.9%).

Overall, these findings reinforce two structural observations. First, problematicness in HLE is rarely concentrated in problem statements alone; answer- and rationale-level defects account for the majority of reliability degradation. Second, domains differ not only in defect prevalence but in defect type: some disciplines exhibit predominantly incorrect content, while others exhibit epistemic indeterminacy due to verification complexity. These patterns justify a component-wise and epistemically explicit verification framework, as aggregate item-level labeling would obscure substantial heterogeneity in defect structure.

#### 4.3 Component-wise Defect Distribution Across Subjects

We next analyze the distribution of defect categories within the revised subset of HLE-Verified using the 19-category component-wise taxonomy. Rather than reporting raw counts alone, we examine the relative proportions of defect types within each component, both globally and across subjects (Figure S1).

Global component-level proportions. When aggregating across subjects, three structural patterns emerge.

Answer-level defects. Across all five domains, Incorrect Answer (type 1) is consistently the dominant answer-level defect. Its within-component proportion ranges from 69.4% (Chemistry) to 97.2% (Biology/Medicine), with all subjects exceeding 70%. This indicates that answer-level failures are primarily deterministic correctness violations rather than ambiguity, partial specification, or formatting issues.

Rationale-level defects. Rationale defects are more heterogeneous across subjects. In Mathematics (40.1%) and Biology/Medicine (45.0%), type 3 defects (structural incompleteness) dominate, whereas Chemistry (40.0%) and Computer Science (63.3%) are led by type 10 defects (format-induced semantic errors). Physics shows a flatter distribution, with type 3 accounting for 25.0% of rationale defects. Overall, rationale instability arises primarily from missing intermediate structure and representation misalignment rather than classical logical contradictions.

Problem-level defects. Problem-level defect patterns are domain-sensitive. Mathematics (40.0%), Chemistry (56.5%), and Computer Science (94.6%) are dominated by type 5 defects (format semantic errors),

Table 1: Dominant defect type (Top-1) by subject and component. Percentages denote within-component ratios.

Subject Problem (Top-1) Rationale (Top-1) Answer (Top-1)

Mathematics Format Semantic Error (40.0%) Information Missing (40.1%) Incorrect Answer (89.4%) Physics Semantic Error (57.1%) Information Missing (25.0%) Incorrect Answer (80.0%) Chemistry Format Semantic Error (56.5%) Format Semantic Error (40.0%) Incorrect Answer (71.4%) Biology/Medicine Knowledge Error (33.3%) Information Missing (45.0%) Incorrect Answer (97.2%) Computer Science Format Semantic Error (94.6%) Format Semantic Error (63.3%) Incorrect Answer (78.1%)

whereas Physics is led by type 1 (57.1%) and Biology/Medicine by type 2 (33.3%). The near absence of dominant theoretical-invalidity categories suggests that most flawed items originate from specification or representation-level imprecision rather than fundamentally incorrect task concepts.

Taken together, these proportions indicate that reliability degradation in HLE is primarily driven by answer-key instability and representation-level specification errors, with pronounced disciplinary heterogeneity in defect structure.

Structural implications. Across domains, several robust regularities emerge:

- 1. Incorrect final answers dominate answer-level instability. Deterministic answer-key errors constitute the principal failure mode across all subjects.
- 2. Rationale instability is primarily structural or representational. Missing intermediate steps and format-level semantic misalignment outweigh classical logical contradictions.
- 3. Representation sensitivity is discipline-dependent. Format semantic errors play a particularly large role in Chemistry and Computer Science.
- 4. Conceptual invalidity is not the dominant driver. Most defects arise from specification or answer-key instability rather than from fundamentally incorrect task concepts.

These findings suggest that defect patterns in HLE exhibit systematic structure rather than uniform random noise. Reliability risks are component-specific and discipline-sensitive, with answer correctness and rationale completeness constituting the dominant bottlenecks. This empirical structure justifies a component-wise verification protocol and cautions against treating benchmark noise as uniform or negligible.

- Table 1 summarizes the dominant defect type (Top-1) for each subject and component. The table highlights a consistent asymmetry: answer-level defects are overwhelmingly dominated by Incorrect Answer across all subjects, whereas problem- and rationale-level dominant defects vary by discipline. In particular, format-induced semantic errors dominate Computer Science, while structural incompleteness is most prevalent in Mathematics and Physics. These patterns reinforce the component-specific and domainsensitive nature of reliability degradation in HLE.

#### 4.4 Case Studies

To complement the quantitative findings, we present representative case studies illustrating structural defect patterns identified in HLE-Verified. These examples demonstrate that benchmark failures are not isolated anomalies but manifestations of systematic component-level weaknesses. We focus on three recurring families aligned with our component-wise analysis:

(1) Answer-level errors (incorrect conclusions, unit/format mismatches, inconsistency with the problem specification), (2) Rationale-level defects (missing prerequisites, circular reasoning, empirical violations, internally inconsistent derivations), and (3) Uncertainty-inducing ambiguity (underspecified assumptions or multiple valid interpretations).

These illustrate why component-wise auditing is essential: an item may present a well-formed problem statement yet remain unsuitable for evaluation due to an incorrect answer key or a structurally nonverifiable rationale. Below we present representative canonical cases selected by domain experts; the complete revision artifacts and detailed correction records are available in the open-sourced HLE-Verified benchmark.

#### Case 1: Theoretical–Implementation Confusion (Computer Science)

Defect Tags: S3 Empirical Soundness Violation + A1 Incorrect Answer Error Mechanism. A speculative decoding sanity-check asked for the expected acceptance rate when the same model serves as both draft and target. The original solution claimed the rate should be less than 1 due to GPU kernel differences (GEMV vs. GEMM) producing slightly different logits. This conflated implementation-level numerical variability with a theoretical property of the algorithm. Revision. Separating theory from hardware artifacts yields:

Ptarget(t) Pdraft(t)

r = min 1,

= 1.

The correct answer is exactly 1 under identical models. Significance. Prevents elevating engineering noise into theoretical reasoning—critical for ML benchmark integrity.

#### Case 2: Numerical Inconsistency in Stoichiometric Constraint (Chemistry)

Defect Tags: Q2 Knowledge Error + S3 Empirical Soundness Violation Error Mechanism. Incorrect molecular mass constants rendered the trimer-plus-lipid relation mathematically inconsistent, making the system unsatisfiable. Revision. Corrected constants restore internal coherence:

3 × protein + 3 × lipid = 101553Da.

The lipid mass (1501 Da) aligns with cardiolipin. Significance. Transforms an incoherent specification into a structurally valid biochemical reasoning task.

#### Case 3: Sign Inconsistency in Exciton Energy (Physics)

Defect Tags: S3 Empirical Soundness Violation + A1 Incorrect Answer Error Mechanism. Confusion among bandgap energy, resonance peak, and binding energy introduced a negative “Rydberg energy,” violating physical interpretation. Revision.

Eb(1) = 2 eV, Ry∗ = 0.5 eV, Eb(3) = 0.08 eV.

Significance. Restores physical coherence and eliminates brittle sign-convention errors.

Across domains, failures cluster in recurring structural patterns: theoretical–implementation confusion, constraint omission, numerical incoherence, sign inconsistency, invariant violation, and conceptual mislocalization. These are structural validity failures rather than cosmetic defects. Stage II revisions therefore restore logical soundness, theoretical alignment, and domain consistency, reinforcing that benchmark reliability depends on component-level integrity rather than superficial correctness.

### 5 Experimental Results

#### 5.1 Experimental Setup

Models. We evaluate the following frontier models: GPT-5.2-Thinking (OpenAI, 2025), Gemini3-ProPreview (Google DeepMind, 2025), Claude-Opus4.5 (Anthropic, 2025), Claude-Opus4.6 (Anthropic, 2026), Grok-4.1 (fast-reasoning) (xAI, 2025), DeepSeek-V3.2-Thinking (Liu et al., 2025), Qwen3-MaxThinking (Qwen, 2026b), Qwen3.5-Plus (Qwen, 2026a). All models were evaluated using the same system prompt recommended by the HLE official guidelines, along with each model’s own default recommended decoding configuration. To reduce variance from stochastic decoding, we run five independent rollouts per item and report avg5 accuracy, i.e., the average correctness across the five sampled completions (without additional post-hoc selection unless otherwise stated).

Metrics. We report (i) Accuracy (Acc) and (ii) Calibration Error (Cali Err). Calibration error is computed from the model’s self-reported confidence and the binary correctness label: we parse each response-level

Table 2: Results on Full Set and Revised Subset.

Full Set Revised Subset

Dataset / Model-Metrics Raw: HLE Full Revised: HLE-Verified Full ∆ Raw: HLE Subset Revised: HLE-Verified Subset ∆ Acc Cali Err Acc Cali Err ∆ Acc Cali Err Acc Cali Err ∆

Gemini3-pro 40.42 56 48.2 49 7.78 18.99 74 48.93 45 29.94 GPT-5.2-High 33.35 45 43.3 36 9.95 14.44 63 52.48 28 38.04

- Claude-Opus4.5 30.00 55 38.8 46 8.80 14.20 70 48.16 39 33.96 Grok 4.1-fast-reasoning 19.94 73 29.0 63 9.06 8.25 83 43.07 47 34.82
- Claude-Opus4.6 38.95 40 46.8 32 7.85 20.03 59 50.16 27 30.13 Deepseek-V3.2 24.90 56 36.4 46 11.50 7.87 70 47.45 28 39.58 Qwen3-Max-Thinking 30.00 66 38.2 57 8.20 14.2 79 48.48 44 34.28

confidence into c ∈ [0,1] and compute Cali Err = 100 × calib_err(c, y; p = 2, β = 100),

where y ∈ {0,1} indicates whether the item is answered correctly, and calib_err is a smoothed L2 miscalibration estimator with smoothing parameter β. Note that this calib_err estimator follows the official HLE evaluation code.

Evaluation Datasets. We report results on (i) the Full Set (Raw HLE Full vs. Revised HLE-Verified Full) to quantify the end-to-end impact on the benchmark, and (ii) a Revised Subset comparison (Raw HLE Subset vs. Revised HLE-Verified Subset) limited to items that were edited/flagged during our verification process, which isolates the effect of flawed items and their corrections. Note that, to ensure comprehensive revisions, we also corrected the original rationales and released them to the open-source community. However, under the standard HLE benchmark evaluation, items with errors only in the rationale do not affect the final scores, since mainstream evaluation uses only the problem and answer. Therefore, the Revised Subset in this experimental section includes only items for which at least one of the problem/answer fields was revised. Unless otherwise specified, all evaluation sets used in this section are the text-only subset of HLE.

#### 5.2 Main Evaluation: HLE vs. HLE-Verified

Table 2 summarizes performance on the original benchmark and on HLE-Verified. We focus on the accuracy shift

∆Acc = Acc(HLE-Verified) − Acc(HLE), computed either on the Full Set or on the Revised Subset.

Large accuracy gains on revised items. On the Revised Subset, all models exhibit substantial positive shifts, indicating that flawed items in the original benchmark systematically suppress measured performance. Specifically, accuracy increases by: Gemini-3-pro (+29.94), GPT-5.2 (+38.04), Claude-Opus4.5 (+32.94), Grok-4.1 fast-reasoning (+34.82), Claude-Opus4.6 (+30.13), and DeepSeek-V3.2 (+39.58) percentage points. The magnitude of these shifts implies that a non-trivial fraction of “errors” on raw HLE are attributable to benchmark issues rather than model capability.

Calibration improves after verification. Calibration error decreases consistently after revision on the same subset (e.g., GPT-5.2: 63→28; DeepSeek-V3.2: 70→28; Grok-4.1: 83→47), suggesting that flawed items can also distort confidence-based evaluation by inducing confidently incorrect (or otherwise miscalibrated) responses. In contrast, HLE-Verified yields a more faithful estimate of the confidence– correctness relationship.

Full-set results. On the Full Set, we also observe consistent gains after verification, though smaller in magnitude than on the Revised Subset, as expected since most items are unchanged. Specifically, accuracy increases by: Gemini-3-pro (+7.58), GPT-5.2 (+9.95), Claude-Opus4.5 (+8.68), Grok-4.1 fast-reasoning (+9.26), Claude-Opus4.6 (+7.75), DeepSeek-V3.2 (+10.79), and Qwen3-Max-Thinking (+8.92) percentage points. Calibration error likewise decreases across models (e.g., GPT-5.2: 45→36; DeepSeek-V3.2: 56→46; Grok-4.1: 73→63), indicating that HLE-Verified not only raises measured accuracy but also yields more reliable confidence estimates at benchmark scale.

#### 5.3 Comparison on Revised Subset across Subject Categories

Category-level gains are uneven but consistently positive. Figure 9 further breaks down the Revised Subset by subject category for several representative models. Across all categories, we observe a

###### HLE-Raw Leaderboard

HLE-Verified Leaderboard

40.4%

48.2%

Gemini 3 Pro

Gemini 3 Pro

39.0%

46.8%

Claude Opus 4.6

Claude Opus 4.6

33.3%

43.3%

GPT-5.2

GPT-5.2

30.0%

38.8%

Claude Opus 4.5

Claude Opus 4.5

30.0%

38.2%

Qwen3-Max-Thinking

Qwen3-Max-Thinking

28.7%

37.6%

Qwen3.5-Plus

Qwen3.5-Plus

24.9%

36.4%

DeepSeek-V3.2

DeepSeek-V3.2

19.9%

29.0%

Grok 4.1 (Fast)

Grok 4.1 (Fast)

0% 10% 20% 30% 40% 50% Average Accuracy (Pass@1)

0% 10% 20% 30% 40% 50% 60% Average Accuracy (Pass@1)

Fig. 7: HLE Raw LeaderBoard vs HLE-Verified LeaderBoard

HLE-Raw vs HLE-Verified on Revised Set

100%

Raw (IDs in Revised Set)

Verified (Revised Set)

80%

AverageAccuracy(Pass@1)

60%

40%

20%

0%

DeepSeek-V3.2 Gemini3Pro GPT-5.2 Grok4.1(Fast)Qwen3-Max-Thinking ClaudeOpus4.5 ClaudeOpus4.6 Qwen3.5-Plus

Fig. 8: HLE Raw vs HLE-Verified on Revised Set

consistent Raw→Verified accuracy increase, confirming that the benefits of verification are not confined to a single domain. However, the magnitude of improvement varies substantially by subject: the largest jumps appear in Physics and Biology/Medicine, where raw accuracies are particularly low but rise sharply after correction, suggesting that benchmark flaws in these categories more frequently affect the problem/answer fields and thus the final pass/fail outcome. In contrast, Chemistry and Computer Science/AI show smaller yet still clear gains, indicating fewer or less severe scoring-impacting issues among the revised items. Overall, this category-wise analysis highlights that raw HLE introduces nonuniform measurement noise across subjects, and HLE-Verified mitigates this distortion, yielding a more faithful cross-domain comparison of model capability.

#### 5.4 Confidence as a Diagnostic for Noisy Items

A natural question is whether models’ self-reported confidence is sensitive to noise in the benchmark itself. If noisy items systematically elicit lower confidence, confidence (and related calibration signals) may serve as a practical diagnostic for item flaws; conversely, if models remain highly confident on flawed items, confidence-based analyses may be unreliable. We therefore study the relationship between model confidence and item quality by comparing confidence statistics before and after verification and repair. To align with this goal, we focus on items whose problem statements are flagged as erroneous, and we compare each model’s confidence on the original version versus the repaired version of the same item. Intuitively, if an item is flawed, we expect models to express lower confidence on the raw statement due to missing conditions, ambiguities, or contradictions; after repair, the clarified statement should reduce uncertainty and lead to higher confidence.

Accordingly, for each model we compute the mean confidence shift

∆Conf = E[cVerified − cRaw] ,

reported both on the Full Set and on a Problem-Error Subset consisting only of items with statementlevel errors. Figure 10 shows that confidence increases consistently after repair on the Problem-Error Subset across all evaluated models, with gains ranging from roughly +1.83 to +11.08 confidence points (absolute). In contrast, the Full-Set shifts are near zero (and can even be slightly negative), which is expected because the majority of items are unchanged and thus dilute the effect.

Gemini 3 Pro: Raw vs Verified by Category (Revised Set)

GPT-5.2: Raw vs Verified by Category (Revised Set)

Claude Opus 4.6: Raw vs Verified by Category (Revised Set)

100%

100%

100%

Raw (IDs in Revised Set)

Raw (IDs in Revised Set)

Raw (IDs in Revised Set)

Verified (Revised Set)

Verified (Revised Set)

Verified (Revised Set)

80%

80%

80%

AverageAccuracy(Pass@1)

AverageAccuracy(Pass@1)

AverageAccuracy(Pass@1)

60%

60%

60%

40%

40%

40%

20%

20%

20%

0%

0%

0%

Physics Biology/Medicine Math Chemistry ComputerScience/AI

Physics Biology/Medicine Math Chemistry ComputerScience/AI

Physics Biology/Medicine Math Chemistry ComputerScience/AI

DeepSeek-V3.2: Raw vs Verified by Category (Revised Set)

qwen3-max: Raw vs Verified by Category (Revised Set)

Grok 4.1 (Fast): Raw vs Verified by Category (Revised Set)

100%

100%

100%

Raw (IDs in Revised Set)

Raw (IDs in Revised Set)

Raw (IDs in Revised Set)

Verified (Revised Set)

Verified (Revised Set)

Verified (Revised Set)

80%

80%

80%

AverageAccuracy(Pass@1)

AverageAccuracy(Pass@1)

AverageAccuracy(Pass@1)

60%

60%

60%

40%

40%

40%

20%

20%

20%

0%

0%

0%

Physics Biology/Medicine Math Chemistry ComputerScience/AI

Physics Biology/Medicine Math Chemistry ComputerScience/AI

Physics Biology/Medicine Math Chemistry ComputerScience/AI

Fig. 9: HLE Raw vs HLE-Verified on Revised Set across Subject Categories.

[Figure 10]

Fig. 10: Impact of item repair on model confidence. Mean confidence shift ∆Conf = E[cVerified − cRaw] computed on the Full Set (blue) and on the Problem-Error Subset containing items with statement-level errors (red). Confidence increases consistently after repair on the error subset, while shifts on the full set are near zero due to dilution by unchanged items.

These results suggest that statement-level noise in raw HLE may not only depress accuracy but also reduce model confidence in a systematic way. As a result, confidence (and related calibration signals) could be a useful diagnostic for flagging potentially noisy or ambiguous items, and HLE-Verified appears to mitigate one source of uncertainty that might otherwise distort confidence-based evaluation.

### 6 Conclusion

We introduced HLE-Verified, a verified and revised benchmark intended to strengthen the scientific reliability of HLE-based evaluations. Our work makes benchmark flaws measurable, provides a transparent correction pipeline, and quantifies how such flaws bias reported accuracy and calibration.

Beyond this release, the disputed set provides a roadmap for community-driven improvements. We expect that a continuously maintained verification process, with structured metadata and clear contribution guidelines, can make exam-style benchmarks more robust and more informative for tracking real progress in language model reasoning.

### References

Research: I forensically audited humanity’s last exam. https://www.reddit.com/r/LocalLLaMA/ comments/1qhz9e2/research_i_forensicaudited_humanitys_last_exam/, 2025. Reddit thread on r/LocalLLaMA.

- Anthropic. Claude opus 4.5. https://www.anthropic.com/news/claude-opus-4-5, 2025.
- Anthropic. Claude opus 4.6. https://www.anthropic.com/news/claude-opus-4-6, 2026.

Center for AI Safety, Scale AI, and HLE Contributors Consortium. A benchmark of expert-level academic questions to assess AI capabilities. Nature, 649:1139–1146, 2026. doi: 10.1038/s41586-025-09962-4.

Wen-Shuo Chao, Zhi Zheng, Hengshu Zhu, and Hao Liu. Make large language model a better ranker. arXiv preprint arXiv:2403.19181, 2024.

Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. Are we done with mmlu? In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5069–5096, 2025.

Google DeepMind. Gemini Pro. https://deepmind.google/models/gemini/pro/, 2025. Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks.

In International conference on machine learning, pages 1321–1330. PMLR, 2017.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

lhl. hle-gpqa-error-claims. https://github.com/lhl/hle-gpqa-error-claims, 2026. GitHub repository.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110, 2022.

Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. Ambigqa: Answering ambiguous open-domain questions. arXiv preprint arXiv:2004.10645, 2020.

OpenAI. Introducing gpt-5.2. https://openai.com/zh-Hans-CN/index/introducing-gpt-5-2/, 2025. Thanosan Prathifkumar, Noble Saji Mathews, and Meiyappan Nagappan. Does swe-bench-verified test

agent ability or model memory? arXiv preprint arXiv:2512.10218, 2025. Team Qwen. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026a. URL

https://qwen.ai/blog?id=qwen3.5.

Team Qwen. Pushing qwen3-max-thinking beyond its limits, January 2026b. URL https://qwen.ai/

blog?id=qwen3-max-thinking.

Andrew White. About 30% of humanity’s last exam chemistry/biology answers are likely wrong. https://www.futurehouse.org/research-announcements/hle-exam, July 2025. FutureHouse research announcement.

xAI. Grok 4.1 fast. https://x.ai/news/grok-4-1-fast, 2025.

### A Appendix

- A.1 LLM Judge Prompt

- A.1.1 LLM Judge Prompt in Stage I Stage I: Model Replication and Answer Verification Prompts

- (1) Solver Prompt.

Please solve the following problem and place the final answer inside \boxed{}. If there are multiple sub-questions, number them and include all answers inside \boxed{}.

{question}

Provide detailed reasoning steps and mark the final answer using \boxed{}.

Purpose. Standardizes solver outputs and enforces structured answer formatting to facilitate downstream extraction.

- (2) Answer Extraction Prompt. Extract the final answer from the following mathematical solution: {response} Requirements:

- 1. Extract the expression inside \boxed{} if present.
- 2. If no \boxed{}, extract the final numeric or declarative conclusion.
- 3. Remove formatting markers such as \boxed{}, $$, \(\).
- 4. If multiple answers exist, separate them with commas.
- 5. Preserve mathematical symbols such as \sqrt{}, \frac{}.

Return only the clean mathematical answer. Answer:

Purpose. Separates reasoning from final answer to reduce evaluation noise.

- (3) Answer Equivalence Judgment Prompt. Compare the following two answers for mathematical equivalence.

Problem: {question_text} Reference Answer: {standard_answer} Model Answer: {model_answer}

Comparison Rules:

- 1. Determine mathematical equivalence.
- 2. Consider alternative forms (e.g., 2\sqrt{3} vs 2*sqrt(3)).
- 3. Consider answer ordering for multi-part responses.
- 4. Ignore formatting differences. Respond with exactly one:

- - "Correct"
- - "Incorrect"
- - "Uncertain"

Role in Pipeline. Used to compute pass@k replication statistics. Outputs are treated as diagnostic signals rather than definitive correctness labels.

#### A.1.2 LLM Judge Prompt in Stage II Stage II: Structured Repair and Multi-Model Adjudication Prompts

- (1) Structured Extraction from Historical Responses. You are a strict information extractor. From the following solution text, extract:

- 1) Final answer (preferably from \boxed{}).
- 2) Core reasoning steps (retain key logical steps only).

If the final answer cannot be reliably extracted, set "empty_answer" to true.

Output JSON only: {

"empty_answer": true/false, "answer": "...", "rationale": "..."

} Solution text: {full_text}

- (2) Repair Prompt (Expert Correction Model). You are a rigorous problem-repair expert. Review:

- - Problem statement
- - Standard answer (if any)
- - Standard rationale (if any)
- - Historical model responses Your task is to output the correct answer and reasoning. Risk Control:
- - If uncertain, set "empty_answer" to true.
- - Do not guess.

Output JSON: {

"empty_answer": true/false, "answer": "...", "rationale": "...", "confidence": 0.0-1.0, "notes": "Brief explanation"

}

- (3) Final Adjudication Prompt. You are the final adjudicator.

Choose among A/B/C repair candidates, or choose "EMPTY" if none are reliable.

Criteria:

- - Self-consistent
- - Verifiable
- - Matches problem statement
- - Prefer EMPTY over fabrication

Output JSON: {

"final_choice": "A"|"B"|"C"|"EMPTY", "empty_answer": true/false, "answer": "...", "rationale": "...", "reason": "One-sentence justification"

}

Purpose. Implements multi-model consensus and conservative arbitration to minimize false corrections.

#### Stage II: Post-Repair Cross-Audit Prompt

You are a strict quality auditor. Evaluate the repaired item:

- (1) question_correct: Is the problem statement clear, self-consistent, and sufficient?
- (2) answer_correct: Is the repaired answer correct and consistent with the problem?
- (3) rationale_correct: Does the repaired reasoning logically derive the answer? If uncertain, choose false conservatively.

Output JSON only: {

"question_correct": true/false, "answer_correct": true/false, "rationale_correct": true/false, "reason_question": "...", "reason_answer": "...", "reason_rationale": "..."

}

Role in Pipeline. Provides structured post-repair validation and closes the verification loop under a conservative decision policy.

- A.2 Quality control and decision principles

We adopt conservative decision principles throughout both stages, reflecting the asymmetric risk inherent in benchmark release. Including a flawed item can introduce systematic evaluation bias and distort cross-model comparisons, whereas excluding a potentially valid item primarily reduces coverage without inducing measurement error. Consequently, our protocol prioritizes minimizing false inclusion over maximizing dataset size.

Conservative inclusion. Inclusion into the gold or revision subsets requires positive evidence of component-wise validity. Items are admitted only when the problem and final answer are judged well-posed, correct, and stable under expert scrutiny. The absence of detected defects is not considered sufficient; rather, validity must be affirmatively supported. Rationale-level defects alone do not automatically disqualify an item, since evaluation is typically answer-based. However, when a rationale defect signals deeper ambiguity, hidden assumptions, or incompatibility with the final answer, the item is escalated for re-audit or routed to revision.

Model-assisted checks as auxiliary evidence. Model-based replication outcomes (e.g., pass@8 success rates) are treated as diagnostic signals rather than adjudicative authority. Extreme patterns—such as systematic failure across multiple strong solvers—may trigger expert re-audit for underspecification, hidden conventions, or answer-key errors. Conversely, high solver agreement is not regarded as proof of correctness. Models serve as stochastic probes that expose potential instability, but final correctness judgments remain grounded in expert evaluation.

Expert recruitment and profile. The verification and revision process was conducted by domain experts recruited through two independent supplier teams and supported by internal adjudication specialists. The participating experts are predominantly Master’s- and Ph.D.-level researchers from research-intensive universities and institutes, with formal academic training in mathematics, physics, chemistry, biomedicine, and computer science. Their academic backgrounds align directly with the disciplinary scope of HLE-Verified, enabling subject-matter–grounded validation and revision across all covered domains. Independent acceptance reviewers with doctoral-level expertise further examined complex or contentious cases.

- A.3 Case Study

- Supplementary Case 1: Constraint Violation in BNLJ Cost Modeling (Database Systems)

Defect Tags: S7 Missing Prerequisite + A1 Incorrect Answer Error Mechanism. The solution reduced page counts using selection predicates while the query explicitly required execution without materialization. This omitted constraint altered the computational model. Revision. Applying the canonical BNLJ formula:

NR B − 2

Cost = NR +

NS,

yields the correct minimum of 465 I/Os. Significance. Demonstrates how omission of a single prerequisite invalidates answer correctness.

#### Supplementary Case 2: Structural Invariant Violation (Mathematics)

Defect Tags: A1 Incorrect Answer Error Mechanism. A closed-form “standard answer” contradicted invariants implied by the Euler sequence. Enumeration for small n exposed systematic divergence. Revision.

n(n + 1) 2

dim H0 Pn, Ω1Pn ⊗ O(2) =

.

Significance. Illustrates that symbolic plausibility does not guarantee theoretical correctness; invariant validation is essential.

#### Supplementary Case 3: Anatomical Mislocalization (Biology/Medicine)

Defect Tags: S3 Empirical Soundness Violation + A1 Incorrect Answer Error Mechanism. A complete oculomotor palsy case was incorrectly localized to the reticular formation, contradicting established neuroanatomy. Revision. Correct localization: midbrain, containing the oculomotor nuclear complex. Significance. Restores clinicopathological coherence and prevents domain-level misinformation in medical evaluation.

#### A.4 Component-wise Defect Distribution Across Subjects

[Figure 11]

Fig. S1: HLE Error Type Distribution by Subject

##### A.5 Cross-subject proportional differences. Figure S1 reveals substantial domain-level variation.

Mathematics. Answer-level defects are overwhelmingly dominated by Incorrect Answer (89.4%). At the rationale level, type 3 (40.1%) is the leading defect, reflecting structural incompleteness in derivations. Problem-level defects are more moderate in scale, with type 5 (40.0%) most frequent. This pattern suggests that mathematical items are generally conceptually stable but vulnerable to answer-key inaccuracies and incomplete reasoning articulation.

Physics. Physics exhibits relatively small defect counts overall. Answer-level errors remain dominated by type 1 (80.0%). Problem-level defects are led by type 1 (57.1%), while rationale defects show a flatter distribution with type 3 accounting for 25.0%. Compared to other domains, physics demonstrates comparatively less concentration in a single rationale category.

Chemistry. Chemistry shows a more balanced defect structure. Although incorrect answers (71.4%) remain the leading answer-level issue, both problem- (56.5%) and rationale-level (40.0%) defects are dominated by format semantic errors. This reflects the domain’s sensitivity to symbolic precision and representational alignment.

Biology/Medicine. Biology/Medicine displays the highest dominance of incorrect answers (97.2%). Rationale-level defects are primarily type 3 (45.0%), indicating incomplete explanatory structure. Problemlevel defects are more evenly distributed, with type 2 (33.3%) emerging as the most frequent category.

Computer Science. Computer Science exhibits the most pronounced representation sensitivity. Format semantic errors (type 5) dominate problem-level defects (94.6%), while type 10 dominates rationalelevel defects (63.3%). Answer-level incorrectness (78.1%) remains substantial but less extreme than in mathematics or biomedicine.

