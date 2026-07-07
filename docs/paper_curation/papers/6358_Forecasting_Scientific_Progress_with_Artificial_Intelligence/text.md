# arXiv:2605.22681v1[cs.AI]21May2026

## Forecasting Scientific Progress with Artificial Intelligence

Sean Wu1,∗, Pan Lu2,∗, Yupeng Chen1, Jonathan Bragg3 Yutaro Yamada4, Peter Clark3, David Clifton1, Philip Torr1,†, James Zou2,†, Junchi Yu1,† 1 University of Oxford 2 Stanford University 3 Allen Institute for AI 4 Sakana AI

###### Abstract

[Figure 1]

[Figure 2]

Repository HuggingFace Project Page

Artificial intelligence (AI) is increasingly embedded in scientific discovery, yet whether it can anticipate scientific progress remains unclear. To study this question, we introduce a temporally grounded evaluation framework for forecasting scientific progress under controlled knowledge constraints. We present CUSP (Cutoff-conditioned Unseen Scientific Progress), a multi-disciplinary and event-level benchmark that evaluates scientific forecasting performance in AI systems through feasibility assessment, mechanistic reasoning, generative solution design, and temporal prediction. Across 4,760 scientific events, we observe systematic and domaindependent limitations in current frontier models. While models can identify plausible research directions from competing candidates, they fail to reliably predict whether scientific advances will be realized and systematically misestimate when they will occur. Model performance is highly heterogeneous across domains, with the timing of AI progress being more predictable than advances in biology, chemistry, and physics. Performance is largely insensitive to whether events occur before or after the training cutoff, suggesting that these limitations cannot be explained solely by knowledge exposure in training data. Under controlled information access, providing additional pre-cutoff knowledge improves performance but does not close the gap to full-information settings, which becomes more pronounced for high-citation scientific advances. Moreover, models exhibit systematic overconfidence and strong response biases, indicating that their uncertainty estimation is unreliable in forecasting scientific progress. Taken together, these findings reveal that current AI systems fall short as predictive tools for scientific progress, as access to prior knowledge does not translate into reliable scientific forecasting. Instead, model performance benefits more from post-event information than from forward-looking prediction.

### 1 Introduction

Scientific progress is often assumed to follow structured patterns [1, 2], with empirical regularities such as Moore’s Law [3] in semiconductors and scaling relationships [4] in deep learning providing quantitative expectations about future developments. These patterns emerge from accumulated scientific progress [5] and have long informed research roadmaps, funding priorities, and technological forecasting [6, 7]. As artificial intelligence (AI) becomes increasingly embedded in scientific discovery and drives advances across domains such as biology, chemistry, physics, and AI itself [8, 9, 10, 11, 12], a question arises: can AI systems forecast the trajectory of scientific progress?

∗Core Contributors. †Correspondence to {junchi.yu, philip.torr}@eng.ox.ac.uk, jamesz@stanford.edu

Recent advances in large language models suggest that AI systems can act as general-purpose scientific assistants and support tasks ranging from hypothesis generation to experiment design [13, 14]. A growing body of work has evaluated their capabilities in scientific reasoning [15, 16], problem-solving [17, 18], and impact prediction [19] across scientific domains. While these studies demonstrate broad proficiency, they do not evaluate whether AI systems can reliably forecast scientific progress under temporal knowledge constraints. Evaluating such capabilities is inherently challenging, as predictions must be grounded in concrete and verifiable scientific events while preventing access to post-event information. As a result, the capabilities and limitations of forecasting scientific progress in current AI systems remain unclear.

To address this gap, we introduce CUSP (Cutoff-conditioned Unseen Scientific Progress), an eventlevel, multi-disciplinary, and temporally grounded framework for evaluating scientific forecasting in AI systems. CUSP is constructed from 4,760 verifiable scientific milestones extracted from top-tier publications and community-driven repositories across multiple disciplines. Each event is associated with a precise temporal reference to enable controlled access to prior knowledge. Crucially, CUSP operationalizes scientific forecasting as a measurable capability across four complementary dimensions: feasibility assessment, mechanistic reasoning, generative solution design, and temporal prediction. By grounding evaluation in verifiable events and enforcing temporal knowledge constraints, CUSP enables systematic assessment of forecasting scientific progress.

We use CUSP to evaluate frontier models under controlled temporal constraints and find a consistent pattern of limitations. While models can identify plausible technical approaches from competing candidates, they struggle to generate solutions that align with the methods underlying realized scientific advances. In feasibility assessment and temporal prediction, models perform near chance in predicting whether scientific advances will be realized and exhibit a strong bias toward delayed outcomes when estimating when such advances will occur. Moreover, models are systematically overconfident and display strong response biases in feasibility assessment, indicating unreliable uncertainty estimation. Notably, these failures persist even on pre-cutoff events, suggesting that models struggle to effectively utilize available knowledge. These limitations also vary across domains, with models performing more strongly in predicting the timing of AI progress than in fields such as biology, chemistry, and physics.

To further understand these limitations, we analyze model performance across pre- and post-cutoff events under controlled information access. Providing additional pre-cutoff knowledge improves performance on both pre-cutoff and post-cutoff events, indicating a knowledge gap in how models access and utilize available information. However, a substantial forecasting gap remains, as models perform significantly worse on post-cutoff events than in full-information settings with post-event knowledge. Notably, the forecasting gap becomes larger for high-citation scientific advances. This suggests that while models can benefit from additional knowledge, they struggle to use it to accurately predict future scientific progress.

Taken together, these results indicate that while current AI systems can identify plausible scientific approaches and benefit from additional knowledge, they lack grounded and well-calibrated scientific forecasting. They fail to accurately predict whether scientific advances will be realized and when they will occur, with these errors becoming more pronounced for high-impact discoveries. This reveals a fundamental gap between knowledge access and the ability to reliably forecast scientific progress.

### 2 The CUSP Benchmark

[Figure 3]

- Figure 1. We construct CUSP by aggregating scientific breakthroughs from top-tier journals and community-driven sources across multiple domains. The benchmark is continuously updated with newly published discoveries, enabling an event-level, dynamic, and temporally grounded evaluation of AI systems’ ability to forecast scientific progress beyond a knowledge cutoff.

We develop CUSP using a temporally stratified corpus of scientific milestones, spanning January 2024 to March 2026, to evaluate scientific forecasting in current AI systems under controlled temporal knowledge constraints. CUSP is designed to rigorously evaluate predictive performance and calibrated expectation on scientific development across a broad spectrum of scientific disciplines. We construct the dataset using domain-specific inclusion criteria to account for the diverse publication dynamics and evidentiary standards inherent to these fields, ensuring that all incorporated milestones represent verifiable and definitively resolved advances. Full inclusion criteria are provided in Appendix I.

We source natural science milestones from Nature, Science and Cell, restricting this subset to highimpact peer-reviewed publications with clearly measurable outcomes. To prevent temporal leakage, we query Crossref, Semantic Scholar, OpenAlex, Europe PMC, arXiv and bioRxiv/medRxiv, and use the earliest observed date associated with each manuscript DOI as the relevant knowledge boundary. For artificial intelligence, we include high-visibility papers from community-driven repositories, including weekly top-paper lists and the Hugging Face Top Papers hub, together with time-resolved leaderboard records from widely used benchmarks such as GPQA Diamond [16], MMLU-Pro [20], and Humanity’s Last Exam [18]. Details of source selection, ranking criteria and leaderboard construction are provided in Supplementary Section A.1.

#### 2.1 Question Types and Synthesis

To operationalize scientific forecasting as a measurable capability, we decompose it into four core dimensions and design task formats that probe each aspect. For each accepted milestone, we construct four complementary evaluation tasks.

Binary prediction (including its perturbed variant) assesses feasibility and calibration by testing whether models can determine if a concrete scientific claim will be achieved and distinguish realized advances from plausible but unrealized alternatives. Multiple-choice questions probe mechanistic reasoning by asking models to identify the technical approach that later enabled the discovery from competing candidates. Free-response questions evaluate generative solution design by requiring models to propose a concrete solution strategy based on the scientific problem context. Finally, date prediction tasks assess temporal reasoning by asking models to forecast when a milestone will be realized.

To construct these tasks, we first decompose each abstract into three structured components: a problem statement, a technical approach, and a results summary. We explicitly remove post-cutoff identifiers and narratives, such as newly introduced acronyms or methodology names, to prevent information leakage. Details of this decomposition procedure are provided in Appendix A.3. The full task taxonomy is summarized in Table 23, and representative examples are provided in Appendix H. Figure 1 illustrates the end-to-end construction pipeline.

To ensure the validity of forecasting tasks, we apply a two-stage validation procedure combining an independent LLM judge with human expert review. The validation process verifies that each task is faithful to the source abstract, objectively verifiable, and free from unsupported perturbations or trivial distractors. This step ensures that all tasks correspond to well-defined and reliable targets for evaluating predictive capabilities. Full validation criteria, prompts, and human-LLM agreement analyses are provided in Appendix B.

#### 2.2 CUSP Time Capsule Construction

To complement the retrospective setting of CUSP, we introduce CUSP Time Capsule, a set of prospective forecasting tasks whose outcomes are not known at evaluation time but will become verifiable in the future. Each question in CUSP Time Capsule is constructed so that the outcome corresponds to a real-world, measurable event and can be resolved by an authoritative source without ambiguity. Because ground truth is not yet available, the Time Capsule is not used for accuracy evaluation, but for analyzing prediction consistency, confidence calibration, and agreement across models.

We design questions across domains, including scientific benchmarks, institutional recognitions, and future technological milestones, curated in collaboration with human experts to ensure their relevance and verifiability (see Appendix H.1). In addition, we include AI capability forecasting by conditioning on current state-of-the-art results. Together, the Time Capsule extends CUSP from retrospective evaluation to prospective analysis of how models form predictions about future scientific progress.

[Figure 4]

- Figure 2. A) Source Distribution: Breakdown of the 4,760 scientific milestones by publication venue. B) Task Density by Domain: Distribution of the 17,429 validated tasks across nine top-level domains. C) Temporal Information: Longitudinal count of entries from January 2024 to March

2026. D) Multi-Disciplinary Taxonomy: Sunburst visualization of distinct subcategories. E) Human vs. AI Keep Rates: Calibration of the Grok-3 validation pipeline against graduate-level human experts. F) Validation Agreement: Reliability metrics of the automated judge, showing high Precision across all question modalities.

Benchmark Scientific Forecasting Time Cutoff Multi Scale Key ForecastBench [21] ✗ ✓ ✓ ✗ ✗ ∼1K Events FutureX [22] ✗ ✓ ✓ ✗ ✓ — Agents FOReCAst [23] ✗ ✓ ✓ ✗ ✓ — Calib PROPHET [24] ✗ ✓ ✓ ✗ ✗ — Retr.

Humanity’s Last Exam [18] ✓ ✗ ✗ ✗ ✗ ∼2.5K QA AstaBench [17] ✓ ✗ ✗ ✗ ✓ ∼2.4K Agents PreScience [19] ✓ ✓ ✗ ✗ ✓ 98K Papers ResearchBench [25] ✓ ✗ ✗ ✗ ✓ 1.3K Decomp ScienceQA [15] ✓ ✗ ✗ ✗ ✓ 21K Reason Matter-of-Fact [26] ✓ ✗ ✓ ✓ ✗ 8.4K Feasibility

CUSP (ours) ✓ ✓ ✓ ✓ ✓ 4.7K / 17K† Forecasting

- Table 1. Comparison of CUSP with forecasting and scientific benchmarks. CUSP uniquely combines scientific grounding, temporal forecasting, cutoff conditioning, and multi-task evaluation. †Scale grows periodically over time as new scientific milestones are continuously incorporated.

#### 2.3 Key Statistics of CUSP

CUSP is an event-based, multi-disciplinary, and temporally grounded benchmark designed to evaluate AI systems’ ability to anticipate scientific progress. The benchmark consists of 4,760 scientific milestones, from which we construct 17,429 structured forecasting tasks spanning multiple evaluation formats. CUSP spans nine top-level scientific domains and 4,245 distinct subcategories, reflecting highly specialized research topics across disciplines. The dataset is dominated by biology (1,234 papers) and artificial intelligence (1,141 papers), followed by medicine (746), neuroscience (403), materials science (375), physics (359), environmental science (235), chemistry (203), and other domains (64). See Figure 2 for further details on temporal, source, and subsection details of our benchmark. See A.4 for key statistics and distributional properties of CUSP.

#### 2.4 Comparison to Related Benchmarks

Two lines of prior work approach related problems, but neither evaluates scientific forecasting under temporal knowledge constraints. The first targets general-world forecasting over news, markets, or geopolitical events and does not enforce knowledge cutoffs tied to scientific discovery [21, 22, 23, 24], leaving its targets disconnected from verifiable scientific events. The second measures the retrospective reasoning via scientific reasoning [15, 16] and problem-solving tasks [17, 18] against resolved ground truth. Although some prior work [19, 25] explores prospective reasoning and future prediction, existing evaluations lack a temporally stratified, event-level framework for disentangling access to scientific knowledge from the ability to forecast future scientific progress. CUSP grounds tasks in temporally referenced scientific events, enabling controlled disentanglement between access to scientific knowledge and the ability to forecast future scientific progress. Moreover, CUSP remains substantially unsaturated across all evaluated frontier systems (Figure 11), consistent with our finding that forecasting scientific progress is harder than retrospective reasoning. Table 1 summarizes the comparison, and Appendix C provides details.

### 3 Model Evaluation

We evaluate model predictions on CUSP using a two-track evaluation framework that combines deterministic outcome scoring with rubric-based scientific reasoning evaluation (Figure 6b). This design is motivated by prior work showing that language models can produce correct final answers while relying on flawed, unfaithful, or post-hoc reasoning processes [27, 28, 29]. CUSP therefore evaluates not only whether a model can correctly forecast future scientific developments, but also whether its generated scientific proposals remain temporally consistent and scientifically plausible under a strict knowledge cutoff.

Each benchmark instance may contain a subset of four task types, including binary classification, multiple-choice questions (MCQ), free-response questions (FRQ), and date prediction. For binary classification, we include both the original statement and a negation-perturbed variant and report a merged score averaged across the two in order to mitigate directional response bias. The evaluation framework automatically detects which tasks are present for each benchmark instance and scores only the corresponding outputs.

- Track I. Deterministic outcome evaluation. The first evaluation track measures forecasting accuracy using deterministic grading procedures tailored to each task type. Binary tasks are evaluated using exact yes or no agreement with the ground-truth label. MCQ tasks are scored through deterministic answer extraction with semantic matching fallbacks when explicit option selection cannot be reliably parsed. Date prediction is evaluated using an exponential-decay distance metric, exp(−0.1d), where d denotes the absolute month difference between the predicted and ground-truth publication dates. This formulation assigns partial credit to temporally proximate forecasts while smoothly penalizing large temporal errors.
- Track II. Free-response scientific reasoning evaluation. Free-response questions (FRQs) evaluate whether models can generate scientifically plausible solution strategies for open research problems under a strict temporal cutoff. Because these tasks admit multiple potentially valid solutions, exactmatch evaluation is inappropriate. Instead, we employ a rubric-based LLM-as-a-judge protocol using GPT-5.4-mini augmented with agentic web search.

FRQ evaluation proceeds in two stages. First, the judge performs explicit leakage detection using web search to determine whether a generated response contains information, terminology, methods, datasets, or discoveries that were unavailable prior to the benchmark cutoff date. Responses identified as containing post-cutoff information are treated as contaminated and do not receive valid forecasting credit. This separation between leakage detection and proposal-quality evaluation prevents memorization from being conflated with scientific reasoning ability.

Second, non-contaminated responses are evaluated along four complementary dimensions designed to capture scientific proposal quality. Alignment measures whether the proposal directly addresses the stated scientific problem. Specificity evaluates the presence of concrete technical mechanisms or implementation details. Novelty measures whether the response introduces non-trivial methodological ideas relative to pre-cutoff scientific knowledge. Feasibility assesses whether the proposed approach is scientifically and technically plausible under the stated constraints. Each dimension is scored on a rubric-anchored 0–10 scale using explicit evaluation criteria and reference anchors provided in Appendix G.1. The final FRQ score is computed as the normalized mean across all four

dimensions. To ensure the reliability of this automated protocol, we conducted a human expert correlation study; detailed methodology and results comparing human versus LLM-as-a-judge scoring are provided in Appendix E.2.

### 4 Results

Here, we report the performance of different LLMs on CUSP benchmark. We evaluate proprietary models, including GPT-5.4 [30], GPT-4o [31], and Claude Sonnet 4.5 (Claude S4.5) [32], alongside open-source models such as LLaMA-3.3-70B-Instruct (LLaMA 3.3) [33], GPT-OSS-20B (GPT-OSS) [34], and DeepSeek R1 [35].

Closed-form tasks↑ Open-ended (FRQ)↑

###### Model Cutoff Binary MCQ Date FRQ score FRQ pass%

GPT-5.4 Aug 2025 0.499 0.819 0.241 5.04 60.3 Claude S4.5 Jan 2025 0.513 0.724 0.239 3.99 14.0

[Figure 5]

[Figure 6]

DeepSeek R1 Jul 2024 0.481 0.594 0.288 4.18 20.0 GPT-4o Oct 2023 0.519 0.530 0.178 3.26 3.9 GPT-OSS Jun 2024 0.518 0.471 0.300 3.86 11.9 LLaMA 3.3 Dec 2023 0.453 0.434 0.500 3.49 2.5

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

- Table 2. Overall benchmark performance on CUSP (n = 4,760 instances). Binary: merged accuracy on original and negation-flipped variants, correcting for directional response bias (chance=0.50). MCQ: 4-choice accuracy (chance=0.25). Date: exponential-decay score e−0.1|∆t| (1.0 = exact month). FRQ score: LLM rubric score (0–10); FRQ pass%: fraction scoring ≥5. Models sorted by MCQ accuracy. Bold: best per column.

Signed error (mo) Within N months (%)↑

Model Cutoff

Mean pred. Mean Median ≤ 3 ≤ 6 ≤ 12 ≤ 24 Score↑

[Figure 11]

LLaMA 3.3 Dec 2023 2025-07 +4.9 +4.0 25.2 45.7 74.6 93.7 0.500

[Figure 12]

DeepSeek R1 Jul 2024 2026-08 +17.9 +13.0 13.3 25.9 48.2 78.6 0.288 GPT-OSS Jun 2024 2027-06 +27.5 +15.0 14.8 25.6 42.7 62.1 0.300 GPT-5.4 Aug 2025 2026-08 +17.4 +14.0 12.2 20.7 32.8 48.8 0.241 Claude S4.5 Jan 2025 2026-06 +15.7 +17.0 11.2 18.3 30.7 55.9 0.239 GPT-4o Oct 2023 2028-02 +35.9 +26.0 6.8 13.2 25.5 46.8 0.178

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- Table 3. Date prediction performance in depth. Mean pred.: mean predicted publication date (ground-truth mean: 2025-02). Signed error: predicted−actual (months); positive=model predicts later than truth. WithinN months: fraction of predictions within N calendar months of the ground truth. All models exhibit positive signed error, i.e. they systematically over-estimate how recently papers are published. Bold: best per column.

#### 4.1 Overall Model Performance

We evaluate model performance across all tasks in the CUSP benchmark, aggregating results across both pre- and post-cutoff instances to characterize overall model behavior. Across task types, we observe a consistent pattern: models perform well at identifying plausible technical approaches but struggle to predict whether and when scientific advances will occur.

Performance varies substantially across models and task types, with no single model dominating (Table 2). On multiple-choice questions, GPT-5.4 achieves the strongest performance (0.819), and all models perform above chance (0.25), indicating that models can reliably identify plausible technical approaches underlying scientific breakthroughs. In contrast, all models remain close to chance (0.50) on binary questions (0.453–0.519), indicating that models do not reliably assess whether scientific claims will be realized.

For free-response questions, GPT-5.4 again achieves the strongest performance, with the highest FRQ score (5.04) and pass rate (60.3%). All other models remain at or below 20% pass rate, indicating that generating accurate solutions is substantially more difficult than selecting among provided alternatives. A more detailed analysis (Table 11) shows that GPT-5.4 outperforms other models across alignment, specificity, and novelty. However, across nearly all models, specificity scores are consistently higher than alignment scores, producing large specificity–alignment gaps of up to +3.0 (full scale: 10). This pattern indicates that models can generate technically detailed and plausible-sounding solutions, yet struggle to align them with the actual methods underlying realized scientific advances. Detailed visualizations of FRQ evaluation are provided in Figure 8.

Date prediction exhibits a markedly different ranking from other tasks (Table 2). LLaMA 3.3 achieves the highest overall date score (0.500), substantially outperforming all other models, while frontier models such as GPT-5.4 and Claude S4.5 perform significantly worse despite stronger performance on other tasks.

To further analyze models’ behavior in date prediction, we conduct a fine-grained analysis of prediction errors (Table 3). Across all models, we observe a consistent positive signed error, indicating a systematic tendency to predict dates later than the ground truth. This reveals a shared temporal bias, whereby models systematically shift scientific progress forward in time.

Despite this bias, models exhibit only coarse temporal accuracy. While a substantial fraction of predictions fall within a 12-month window, exact-match accuracy remains below 4% for all models, indicating that precise month-level prediction is highly challenging. This suggests that models can approximate when scientific developments occur, but lack fine-grained temporal resolution.

Beyond this shared bias, models differ substantially in the distribution of date prediction errors. LLaMA 3.3 exhibits more concentrated errors, with smaller median error (+4.0 months) and higher fraction of predictions within 12 months of the ground truth (74.6%). In contrast, frontier models such as GPT-5.4 and Claude S4.5 show substantially larger errors, with median delays of 14.0 and 17.0 months, respectively, and much lower within-12-month accuracy. These differences indicate variation not only in average performance but also in the stability of date prediction.

Taken together, these results show that models can identify plausible technical approaches and generate technically detailed solutions, but fail to reliably determine whether scientific advances will be realized, align generated solutions with the actual methods underlying the discovery, or accurately predict when such advances will occur.

###### MCQ accuracy by scientific domain — CUSP Benchmark (spokes = research areas; grey = cross-model mean; dotted = chance level)

###### GPT-4o cutoff: 2023-10

###### GPT-OSS 20B cutoff: 2024-06

###### DeepSeek R1 cutoff: 2024-07

Medicine Neurosci.

Medicine Neurosci.

Medicine Neurosci.

AI

AI

AI

| |
|---|

| |
|---|

Mat. Sci.

Mat. Sci.

Mat. Sci.

| |
|---|

0.50

0.50

0.50

0.40

0.40

0.40

| |
|---|

0.30

0.30

0.30

0.20

0.20

0.20

0.10

0.10

0.10

Biology

Biology

Biology

| | | |
|---|---|---|
| | | |

| |
|---|

Physics

Physics

Physics

| |
|---|

| |
|---|

Other

Other

Other

Env. Sci.

Env. Sci.

Env. Sci.

Chemistry

Chemistry

Chemistry

###### LLaMA 3.3 cutoff: 2023-12

###### Claude S4.5 cutoff: 2025-01

GPT-5.4 cutoff: 2025-08

Medicine Neurosci.

Medicine Neurosci.

Medicine Neurosci.

AI

AI

AI

Mat. Sci.

Mat. Sci.

Mat. Sci.

0.50

0.50

0.50

0.40

0.40

0.40

0.30

0.30

0.30

0.20

0.20

0.20

0.10

0.10

0.10

Biology

Biology

Biology

Physics

Physics

Physics

Other

Other

Other

Env. Sci.

Env. Sci.

Env. Sci.

Chemistry

Chemistry

Chemistry

Cross-model mean Chance (0.25)

Figure 3. Radar plots of LLM MCQ performance across six models across the main areas of CUSP.

From Recognition to Prediction.

Models can recognize plausible scientific approaches but misalign with the methods underlying real advances, and fail to predict whether such advances will be realized and when they will occur.

#### 4.2 Heterogeneous Predictability Across Scientific Domains

Across the six models we benchmark, we observe substantial variation in predictability across scientific domains and task types in CUSP. However, this variation is not uniform, but depends on the underlying capability being evaluated. As in Section 4.1, results in this section are aggregated across both pre- and post-cutoff instances, providing an overall characterization of domain-level model behavior.

For mechanistic reasoning in multiple-choice questions (MCQ), performance varies significantly by research area (Table 12). Physics (62.1%), neuroscience (65.8%), and environmental science (66.2%) exhibit relatively high predictability, while chemistry (49.8%), materials science (54.2%), and AI

(54.3%) remain comparatively challenging. This indicates that the ability to identify technical approaches underlying breakthroughs differs across domains.

A similar pattern emerges for free-response questions (FRQ) (Table 15). Neuroscience (4.11), biology (4.05), and AI (4.04) achieve the highest average FRQ scores, while chemistry (3.54), physics (3.74), and materials science (3.80) remain consistently more difficult. These results show that the ability to generate plausible solution strategies does not transfer uniformly across domains.

In contrast, date prediction exhibits a distinct pattern. AI achieves substantially higher date prediction scores (0.461) than other domains (typically 0.18–0.28) (Table 13), indicating comparatively higher predictability in this domain. This suggests that the timing of AI progress is more predictable than in other scientific domains.

Despite these domain-dependent differences, binary feasibility prediction remains near chance across all domains (46–52%). This indicates that models do not reliably distinguish feasible from infeasible scientific advances regardless of domains, even in domains where other aspects of prediction perform relatively well.

Further analysis at the sub-domain level is provided in Appendix D. These results show that performance also varies within domains; for example, models perform better on MCQ questions related to tool use, while exhibiting lower accuracy on questions involving autoregressive models.

Taken together, these results show that predictability is capability-dependent rather than purely domain-dependent. While mechanistic reasoning, solution generation, and temporal prediction vary across domains, feasibility prediction remains uniformly limited, indicating a consistent inability to assess whether scientific advances will be achieved.

Capability-Dependent Predictability Across Domains

Predictability varies across domains, but depends on capability rather than domain alone. Mechanistic reasoning, solution generation, and temporal prediction exhibit domain-dependent variation. Feasibility prediction remains near chance across all domains, indicating a consistent limitation in assessing whether advances will be realized.

#### 4.3 Scientific Forecasting Beyond the Training Cutoff

We examine whether model performance degrades when predicting scientific advances beyond the training cutoff. We find that performance is largely insensitive to whether events occur before or after their training cutoff. This result is counterintuitive, as models would be expected to perform better on pre-cutoff events.

To evaluate this, we compare performance on events in CUSP corresponding to discoveries published before and after each model’s training cutoff (Table 4). We distinguish between two evaluation regimes. For models whose training cutoff overlaps with the benchmark period (e.g., GPT-5.4, Claude S4.5, DeepSeek R1, GPT-OSS), we compare performance on pre- and post-cutoff instances. For models with earlier cutoffs (e.g., GPT-4o and LLaMA 3.3), all benchmark events fall after their training cutoff, and thus these models operate entirely in an out-of-distribution setting.

For models whose training cutoff overlaps with the benchmark period, performance degrades only modestly across most tasks. For multiple-choice questions, GPT-5.4 decreases from 0.83

Model Cutoff npre Binary (pre→post) MCQ (pre→post) FRQ (pre→post) Date (pre→post) GPT-5.4 Aug 2025 4577 0.503→0.491 0.830→0.792 5.04→5.04 0.231→0.270 Claude S4.5 Jan 2025 2516 0.495→0.526 0.762→0.699 3.95→4.02 0.201→0.270

[Figure 17]

[Figure 18]

DeepSeek R1 Jul 2024 1198 0.488→0.480 0.619→0.589 4.17→4.18 0.145→0.328 LLaMA 3.3 Dec 2023 — —→0.453 —→0.434 —→3.49 —→0.500 GPT-OSS Jun 2024 949 0.476→0.526 0.497→0.466 3.88→3.86 0.126→0.336 GPT-4o Oct 2023 — —→0.519 —→0.530 —→3.26 —→0.178

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

- Table 4. Performance on instances published before vs. after each model’s training cutoff (pre→post). Models with cutoff ≤ Dec 2023 (GPT-4o, LLaMA3.3) have no pre-cutoff instances and show “—

→post”. Cell colour: green > +0.05 improvement; orange < −0.03 degradation; red < −0.10 strong degradation. Binary/MCQ: accuracy (0–1); FRQ: rubric score (0–10); Date: exponentialdecay score.

npre: number of binary-task instances in the pre-cutoff partition. Benchmark papers span Jan 2024–Mar 2026; models with an earlier cutoff contribute only post-cutoff instances. FRQ sub-dimension breakdown in Table 11.

to 0.792, while DeepSeek R1 slightly drops from 0.619 to 0.589. Similar patterns hold for freeresponse questions and binary prediction tasks, where performance remains largely stable across the training cutoff boundary. In contrast, date prediction exhibits an opposite trend, with several models achieving higher scores on post-cutoff (unseen) events (e.g., Claude S4.5: +0.1; DeepSeek R1: +0.2). As detailed in Figure 9, this effect is partly driven by temporal anchoring, where consistent predictions of specific dates can artificially improve scores for future events.

Models with early training cutoffs (e.g., GPT-4o and LLaMA 3.3) provide a complementary perspective. Despite operating entirely on post-cutoff events, they exhibit performance patterns broadly consistent with those observed in models with partial overlap. This further suggests that direct exposure to target discoveries is not the primary driver of performance. The absence of a clear pre-/post-cutoff performance gap is counterintuitive, as models would be expected to perform better on discoveries within their training knowledge. This raises two competing possibilities: either models lack access to relevant pre-cutoff knowledge, or they fail to effectively use such knowledge when making predictions. We examine this distinction further in the next section.

Not a Training Data Problem

Performance is largely insensitive to the training cutoff. Models perform similarly on pre- and postcutoff events, suggesting that their limitations are not primarily driven by exposure to target discoveries during training.

#### 4.4 Disentangling Knowledge and Forecasting Gaps via Controlled Information Access

We find that improving access to pre-cutoff knowledge does not resolve the gap in predicting post-cutoff scientific events. While additional knowledge leads to measurable performance gains, a substantial gap remains relative to full-information hindsight analysis with post-event knowledge.

We use web search augmentation with controlled information access on a random subset of CUSP

(500 events). We compare three settings: the base model (GPT-4o, GPT-5.4, DeepSeek R1, and Claude S4.5), web search restricted to pre-cutoff information (WS+Cutoff), and unrestricted search (WS), which provides full access to up-to-date information.

Results in Table 9 show that WS+Cutoff yields larger improvements on pre-cutoff than post-cutoff instances, indicating that CUSP distinguishes between the two regimes. Meanwhile, the substantial gains on pre-cutoff questions show that providing explicit access to relevant historical information improves performance, suggesting that such information is not fully accessed in base models. Full results of web search augmentation are shown in Table 8.

Detailed analysis of performance on post-cutoff events (Table 9) reveals two components: a knowledge gap (∆know), capturing gains from additional pre-cutoff information, and a forecasting gap (∆fore), capturing the remaining difference relative to full-information hindsight analysis. Across tasks (Table 24), the forecasting gap is consistently large and, in some cases, exceeds the knowledge gap (e.g., date prediction for GPT-5.4: ∆fore = 0.436 vs. ∆know = 0.070). This indicates that access to additional pre-cutoff knowledge alone is insufficient for accurate scientific forecasting.

The relative contributions of ∆know and ∆fore vary across tasks. Binary and date prediction are strongly dominated by the forecasting gap, indicating that models cannot close the performance gap to the full-information condition using pre-cutoff information alone. In contrast, free-response generation exhibits more balanced contributions from both components, suggesting that models can partially leverage pre-cutoff information to construct plausible scientific approaches, but still rely on additional information to improve performance.

We further examine how ∆know and ∆fore relate to scientific impact (Table 29). The forecasting gap increases systematically with citation-count quartile across tasks, while the knowledge gap

does not exhibit a consistent trend. In GPT-5.4, ∆fore increases from 0.060 in low-citation papers (Q1) to 0.875 in high-citation papers (Q4), whereas ∆know shows no such relationship. For lowcitation papers (Q1), the forecasting gap is comparatively small, indicating weaker dependence on post-cutoff information. This suggests that models particularly struggle to forecast high-impact advances using pre-cutoff information alone.

Taken together, these results show that access to prior knowledge does not translate into scientific forecasting. While models benefit from additional pre-cutoff information, they remain unable to accurately predict future scientific advances, instead relying on retrospective inference once outcomes are known.

Knowledge Does Not Translate into Forecasting

Access to knowledge improves performance but does not resolve scientific forecasting. Providing additional pre-cutoff information yields measurable gains, yet a substantial gap remains relative to full-information settings, indicating that models cannot reliably use available knowledge to predict future scientific advances.

#### 4.5 Systematic Biases and Failure Modes in Scientific Forecasting

We find that models exhibit systematic failures in representing uncertainty when forecasting scientific advances. Across all models and tasks (Table 20), reported confidence consistently exceeds empirical accuracy, indicating widespread overconfidence. This effect is particularly pronounced in date prediction, where models are both inaccurate and highly miscalibrated.

Calibration varies substantially across task types. While models can achieve relatively good calibration on multiple-choice reasoning tasks, calibration deteriorates when moving to open-ended forecasting settings. This indicates that uncertainty estimates become less reliable when models are required to extrapolate beyond known information (Figure 10).

Figure 4. Visualization of model bias in binary prediction.

###### A

###### B

###### Binary (ground truth = "Yes")

###### Binary perturbed (ground truth = "No")

93%

91%

LLaMA 3.3

63% 37%

60% 40%

GPT-5.4

47% 53%

51% 49%

We also observe strong and persistent response biases in feasibility prediction (Table 19). Models exhibit systematic tendencies toward affirmative or negative responses, which dominate prediction behavior. For example, GPT-4o and GPT-OSS show a consistent “No” bias, while LLaMA 3.3 exhibits a strong “Yes” bias (Figure 4). These response priors lead to nearrandom performance, indicating that predictions are often driven by bias rather than evidence.

DeepSeek R1

27% 73%

30% 70%

Claude S4.5

- 18% 82%
- 19% 81%

- 22% 78%
- 23% 77%

GPT-4o

GPT-OSS 20B

"Yes" ✓ correct "No" ✗ incorrect

"Yes" ✗ incorrect "No" ✓ correct

| |
|---|

| |
|---|

0 % 50 % 100 %

0 % 50 % 100 %

Response rate

Response rate

Analysis under controlled information access (Table 9) shows that these behaviors persist across pre- and post-cutoff settings. Providing additional pre-cutoff knowledge does not substantially reduce overconfidence or response bias, and calibration changes are inconsistent across tasks. In multiple-choice tasks, overconfidence increases after the cutoff despite no improvement in accuracy, while in temporal prediction, calibration deteriorates. Binary feasibility prediction shows unstable shifts without a consistent pattern.

Taken together, these results show that models lack a stable and consistent representation of uncertainty when forecasting scientific advances. Instead, uncertainty estimates are fragmented across tasks and influenced by systematic response biases, limiting the reliability of forecasting scientific progress.

We further analyze calibration before and after each model’s training knowledge cutoff across binary, multiple-choice, and date prediction tasks (Table 5). This setting isolates model behavior under distribution shift, where predictions must be made about scientific developments beyond the training data. Interestingly, we find a striking difference in calibration across task types. For example, the MCQ setting shows that overconfidence consistently increases after the cutoff dates, with a nearly positive ∆ across nearly all models. This indicates that LLMs become more confident when reasoning about completely unseen scientific developments, even though there is no improvement in accuracy. In contrast, date prediction tasks demonstrate a decrease in overconfidence, with negative ∆ values up to −0.46, which demonstrates a collapse in calibration when models are required to estimate temporal uncertainty. Binary feasibility prediction shows no consistent trend, instead exhibiting unstable shifts in overconfidence across models. These results indicate that LLMs lack a stable and unified representation of uncertainty when forecasting unseen scientific developments, instead exhibiting fragmented and task-dependent calibration behavior.

Binary task MCQ task Date task (≤12mo) Model Pre Post ∆ Pre Post ∆ Pre Post ∆

GPT-5.4 0.208 0.187 −0.021 0.028 0.036 +0.008 0.281 0.232 −0.050 Claude S4.5 0.220 0.158 −0.062 −0.030 0.010 +0.040 0.296 0.160 −0.137 LLaMA 3.3 — 0.309 — — 0.390 — — −0.004 —

[Figure 23]

[Figure 24]

[Figure 25]

DeepSeek R1 0.067 0.092 +0.025 0.296 0.320 +0.024 0.533 0.105 −0.427 GPT-OSS 0.272 0.175 −0.097 0.356 0.371 +0.015 0.529 0.066 −0.464 GPT-4o — 0.213 — — 0.346 — — 0.520 —

[Figure 26]

[Figure 27]

[Figure 28]

- Table 5. Confidence calibration before and after the training knowledge cutoff for three task types. Overconfidence = c¯−a¯, where c¯ is the mean stated confidence and a¯ is task accuracy; zero indicates perfect calibration. For the date task, accuracy is defined as the fraction of predictions falling within 12 months of the ground truth. ∆ = post-cutoff − pre-cutoff; positive ∆ indicates overconfidence increased after the knowledge cutoff.

Forecasting Without Awareness

Models lack reliable uncertainty estimates when forecasting scientific progress. Predictions are systematically overconfident and influenced by response biases, with calibration degrading under distribution shift and differing across tasks, indicating a fragmented representation of uncertainty.

#### 4.6 Scientific Forecasting in a Time Capsule

We use CUSP Time Capsule to study how frontier models anticipate future (beyond April 2026) scientific and AI progress. We evaluate both open-ended scientific forecasting and capability prediction on benchmarks, asking models to extrapolate future breakthroughs.

- Figure 5 shows models’ predictions of global

CO2 emissions in 2027. All models forecast emissions to remain above the 2025 level, indicating a shared expectation of continued nearterm growth in global emissions. However, the predicted magnitude varies across models. Claude S4.5, DeepSeek R1, and GPT-4o produce comparatively conservative estimates close to the historical trend, whereas GPT-5.4 predicts a slightly larger increase. LLaMA 3.3 and GPT-OSS forecast the steepest growth, with LLaMA 3.3 producing the highest projected emissions overall. These results suggest that while models broadly agree on the direction of future emissions, they differ substantially in their expectations about the pace of global decarbonization.

Figure 5. Forecasts of global CO2 emissions.

- 33

- 34

- 35

- 36

- 37

- 38

- 39

- 40

Historical CO2 emissions LLaMA 3.3

GPT-OSS

GlobalCOemissions(GtCO)22

GPT-5.4

Claude S4.5 DeepSeek R1 GPT-4o

2015 2017 2019 2021 2023 2025 2027 Year

Forecast date 2026-10 2027-04 2027-10 Current state-of-the-art (2026-04): 56.8% / 64.7%

Model

GPT-5.4 63% (↑6.2) / 74% (↑9.3) 72% (↑15.2) / 78% (↑13.3) 74% (↑17.2) / 82% (↑17.3) Claude S4.5 62% (↑5.2) / 71% (↑6.3) 64% (↑7.2) / 72% (↑7.3) 68% (↑11.2) / 73% (↑8.3)

[Figure 29]

[Figure 30]

DeepSeek R1 59% (↑2.2) / 69.5% (↑4.8) 64% (↑7.2) / 72% (↑7.3) 62% (↑5.2) / 78% (↑13.3) LLaMA 3.3 62% (↑5.2) / 71% (↑6.3) 65% (↑8.2) / 75% (↑10.3) 65% (↑8.2) / 75% (↑10.3) GPT-OSS 68% (↑11.2) / 75% (↑10.3) 70% (↑13.2) / 73% (↑8.3) 70% (↑13.2) / 78% (↑13.3) GPT-4o 62% (↑5.2) / 68% (↑3.3) 65% (↑8.2) / 72% (↑7.3) 68% (↑11.2) / 78% (↑13.3)

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

- Table 6. Future predictions on Humanity’s Last Exam. Values are reported as no-tools / with-tools.

Model

Forecast date 2026-10 2027-04 2027-10 Current state-of-the-art (2026-04): 94.6% / 92.6%

[Figure 35]

GPT-5.4 96.2% (↑1.6) / 93.8% (↑1.2) 97.2% (↑2.6) / 95.1% (↑2.5) 97.8% (↑3.2) / 95.8% (↑3.2) Claude S4.5 96.5% (↑1.9) / 94.2% (↑1.6) 97.5% (↑2.9) / 94.2% (↑1.6) 97.5% (↑2.9) / 95.2% (↑2.6) DeepSeek R1 96% (↑1.4) / 93.3% (↑0.7) 96.2% (↑1.6) / 94.1% (↑1.5) 96.2% (↑1.6) / 94% (↑1.4) LLaMA 3.3 96.2% (↑1.6) / 94.2% (↑1.6) 96.2% (↑1.6) / 94.2% (↑1.6) 96.2% (↑1.6) / 95% (↑2.4) GPT-OSS 95.8% (↑1.2) / 94.2% (↑1.6) 95.8% (↑1.2) / 93.5% (↑0.9) 96.5% (↑1.9) / 94.2% (↑1.6) GPT-4o 95.2% (↑0.6) / 94.2% (↑1.6) 96.8% (↑2.2) / 94.5% (↑1.9) 96.8% (↑2.2) / 95.5% (↑2.9)

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

- Table 7. Future predictions on GPQA Diamond / MMMLU. Values are reported as GPQA Diamond / MMMLU.

Tables 6 and 7 summarize models’ forecasts of future AI capabilities on a range of multi-disciplinary evaluation benchmarks, including Humanity’s Last Exam (with and without tools), GPQA Diamond, and MMMLU [36]. Across models, we observe a shared expectation of continued capability gains over the 2026-2027 horizon, though the projected magnitude varies substantially. GPT-5.4 produces the most optimistic forecasts, especially for 2027-10, predicting Humanity’s Last Exam performance to rise from 56.8% to 74% without tools and from 64.7% to 82% with tools. It also forecasts nearsaturation performance on GPQA Diamond and MMMLU by late 2027. In contrast, DeepSeek R1 consistently makes more conservative predictions, with smaller gains and earlier plateaus, particularly on already high-performing benchmarks. A broader trend is that forecasts for GPQA Diamond and MMMLU are tightly clustered near the upper performance bound, suggesting that models expect these benchmarks to saturate within the next few generations. Humanity’s Last Exam, especially in the tool-augmented setting, shows substantially greater variance, indicating that models perceive it as a more open-ended and capability-sensitive benchmark.

Together, these results highlight the potential of CUSP Time Capsule as a framework for studying not only factual recall and forecasting, but also the implicit scientific priors and future-oriented world models embedded within frontier AI systems.

###### Common Patterns in a Time Capsule

Frontier AI models exhibit strikingly consistent patterns when forecasting future scientific and AI progress, converging on similar expectations of future breakthroughs across disciplines and coherent capability gains across AI benchmarks.

### 5 Conclusion

We introduced CUSP, a temporally grounded benchmark for evaluating scientific forecasting in AI systems, and used it to study whether current models can anticipate the trajectory of scientific progress. Across a large and multi-disciplinary set of scientific events, we find that while frontier models can identify plausible technical approaches and benefit from additional knowledge, they consistently fail to accurately predict whether scientific advances will be realized, when they will occur, and how they will be realized. These limitations persist beyond the training cutoff, cannot be explained solely by access to prior knowledge, and are accompanied by unreliable and biased uncertainty estimates. Together, these results reveal a fundamental gap between access to knowledge and reliable scientific forecasting. Rather than forming grounded and calibrated expectations about future developments, current AI systems benefit from retrospective inference once outcomes are known, rather than making reliable forward-looking predictions. This suggests that anticipating scientific progress requires capabilities beyond knowledge retrieval, including the ability to reason under uncertainty about how scientific discoveries unfold over time. Our findings highlight the importance of temporally grounded, event-level evaluation for understanding the limits of AI in scientific discovery. Understanding the gap between access to knowledge and the ability to forecast scientific progress is essential for building AI systems that can meaningfully participate in the scientific process. Closing this gap will be critical for the development of AI systems that can not only assist with scientific discovery but also provide reliable forward-looking guidance for research and innovation.

### Acknowledgments

Sean Wu is supported by the Rhodes Scholarship. Junchi Yu and Philip Torr are funded by the UKRI grant: Turing AI Fellowship EP/W002981/1 and the Schmidt Science Foundation. Philip Torr is a Schmidt Science AI 2050 Senior Fellow. David Clifton is funded by an NIHR Research Professorship (NIHR302440), a Royal Academy of Engineering Research Chair, and the InnoHK Hong Kong Centre for Cerebro-Cardiovascular Engineering, and was supported by the National Institute for Health Research Oxford Biomedical Research Centre and the Pandemic Sciences Institute at the University of Oxford. This work is partially supported by the Hoffman-Yee Research Grants program at Stanford HAI and the AI for Math Fund by Renaissance Philanthropy. We thank Tinglin Huang, Shenxu Chang, Aoxi Liu, Baicheng Chen, Xiaoyu Zhang, Guanzong Wu, Bo Zheng, Zhun Zhang, Yijie Sun, and Ruiyang Lu for their assistance with human evaluation of our benchmark. We thank Fabien Scalzo for support with GPU compute, and Y Combinator for providing API compute credits.

### References

- [1] Brian Uzzi, Satyam Mukherjee, Michael Stringer, and Ben Jones. Atypical combinations and scientific impact. Science, 342(6157):468–472, 2013.
- [2] Dudley Shapere. The structure of scientific revolutions. The Philosophical Review, 73(3):383– 394, 1964.
- [3] Gordon Moore. Moore’s law. Electronics Magazine, 38(8):114, 1965.
- [4] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [5] Santo Fortunato, Carl T Bergstrom, Katy Börner, James A Evans, Dirk Helbing, Staša Milojević, Alexander M Petersen, Filippo Radicchi, Roberta Sinatra, Brian Uzzi, et al. Science of science. Science, 359(6379):eaao0185, 2018.
- [6] Michael Park, Erin Leahey, and Russell J Funk. Papers and patents are becoming less disruptive over time. Nature, 613(7942):138–144, 2023.
- [7] Ian Miles. The development of technology foresight: A review. Technological forecasting and social change, 77(9):1448–1456, 2010.
- [8] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate protein structure prediction with alphafold. Nature, 596(7873):583–589, 2021.
- [9] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature, 630(8016):493–500, 2024.
- [10] Alexander Novikov, Ngân V˜u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.
- [11] Amil Merchant, Simon Batzner, Samuel S Schoenholz, Muratahan Aykol, Gowoon Cheon, and Ekin Dogus Cubuk. Scaling deep learning for materials discovery. Nature, 624(7990):80–85, 2023.
- [12] Kyle Swanson, Wesley Wu, Nash L Bulaong, John E Pak, and James Zou. The virtual lab of ai agents designs new sars-cov-2 nanobodies. Nature, 646(8085):716–723, 2025.
- [13] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, et al. Accelerating scientific discovery with co-scientist. Nature, 2026.
- [14] Ali Essam Ghareeb, Benjamin Chang, Ludovico Mitchener, Angela Yiu, Caralyn J. Szostkiewicz, Dmytro Shved, Gavin J. Gyimesi, Jon M. Laurent, Samantha M. Wright, Muhammad T. Razzak, Andrew D. White, Silvia C. Finnemann, Michael M. Hinks, and Samuel G. Rodrigues. A multi-agent system for automating scientific discovery. Nature, 2026.

- [15] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in neural information processing systems, 2022.
- [16] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First conference on language modeling, 2024.
- [17] Jonathan Bragg, Mike D’Arcy, Nishant Balepur, Dan Bareket, Bhavana Dalvi, Sergey Feldman, Dany Haddad, Jena D Hwang, Peter Jansen, Varsha Kishore, et al. Astabench: Rigorous benchmarking of ai agents with a scientific research suite. In International conference on learning representations, 2026.
- [18] Center for AI Safety, Scale AI, and HLE Contributors Consortium. A benchmark of expert-level academic questions to assess AI capabilities. Nature, 649:1139–1146, 2026.
- [19] Anirudh Ajith, Amanpreet Singh, Jay DeYoung, Nadav Kunievsky, Austin C Kozlowski, Oyvind Tafjord, James Evans, Daniel S Weld, Tom Hope, and Doug Downey. Prescience: A benchmark for forecasting scientific contributions. arXiv preprint arXiv:2602.20459, 2026.
- [20] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In Advances in neural information processing systems, 2024.
- [21] Ezra Karger, Houtan Bastani, Chen Yueh-Han, Zachary Jacobs, Danny Halawi, Fred Zhang, and Philip Tetlock. Forecastbench: A dynamic benchmark of AI forecasting capabilities. In International conference on learning representations, 2025.
- [22] Zhiyuan Zeng, Jiashuo Liu, Siyuan Chen, Tianci He, Yali Liao, Yixiao Tian, Jinpeng Wang, Zaiyuan Wang, Yang Yang, Lingyue Yin, et al. Futurex: An advanced live benchmark for llm agents in future prediction. arXiv preprint arXiv:2508.11987, 2025.
- [23] Moy Yuan, Zifeng Ding, and Andreas Vlachos. Introducing FOReCAst: The future outcome reasoning and confidence assessment benchmark. In Advances in neural information processing systems datasets and benchmarks track, 2025.
- [24] Zhengwei Tao, Pu Wu, Zhi Jin, Xiaoying Bai, Haiyan Zhao, Chengfeng Dou, Xiancai Chen, Jia Li, Linyu Li, Chongyang Tao, et al. Prophet: An inferable future forecasting benchmark with causal intervened likelihood estimation. arXiv preprint arXiv:2504.01509, 2025.
- [25] Yujie Liu, Zonglin Yang, Tong Xie, Jinjie Ni, Ben Gao, Yuqiang Li, Shixiang Tang, Wanli Ouyang, Erik Cambria, and Dongzhan Zhou. Researchbench: Benchmarking llms in scientific discovery via inspiration-based task decomposition. arXiv preprint arXiv:2503.21248, 2025.
- [26] Peter Jansen, Samiah Hassan, and Ruoyao Wang. Matter-of-fact: A benchmark for verifying the feasibility of literature-supported claims in materials science. In Empirical methods in natural language processing, 2025.
- [27] Jiayi Sheng, Luna Lyu, Jikai Jin, Tanglin Xia, Alex Gu, James Zou, and Pan Lu. Solving

- inequality proofs with large language models. In Advances in neural information processing systems, 2025.
- [28] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International conference on learning representations, 2023.
- [29] Katie Matton, Robert Ness, John Guttag, and Emre Kiciman. Walk the talk? measuring the faithfulness of large language model explanations. In International conference on learning representations, 2025.
- [30] OpenAI. Introducing GPT-5.4. https://openai.com/index/introducing-gpt-5-4/, March 2026.
- [31] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [32] Anthropic. Introducing claude sonnet 4.5. https://www.anthropic.com/news/ claude-sonnet-4-5, September 2025.
- [33] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [34] Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.
- [35] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.
- [36] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International conference on learning representations, 2021.
- [37] Protein Data Bank. Protein data bank. Nature New Biol, 233(223):10–1038, 1971.
- [38] Federico Bianchi, Owen Queen, Nitya Thakkar, Eric Sun, and James Zou. Exploring the use of ai authors and reviewers at agents4science. Nature Biotechnology, pages 1–4, 2025.
- [39] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.
- [40] Chris Lu, Cong Lu, Robert Tjarko Lange, Yutaro Yamada, Shengran Hu, Jakob Foerster, David Ha, and Jeff Clune. Towards end-to-end automation of ai research. Nature, 651(8107):914–919, 2026.
- [41] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, et al. Towards an ai co-scientist. arXiv preprint arXiv:2502.18864, 2025.

- [42] Ludovico Mitchener, Angela Yiu, Benjamin Chang, Mathieu Bourdenx, Tyler Nadolski, Arvis Sulovari, Eric C Landsness, Daniel L Barabasi, Siddharth Narayanan, Nicky Evans, et al. Kosmos: An ai scientist for autonomous discovery. arXiv preprint arXiv:2511.02824, 2025.
- [43] Georgia Channing and Avijit Ghosh. Ai for scientific discovery is a social problem. arXiv preprint arXiv:2509.06580, 2025.
- [44] Katja Grace, John Salvatier, Allan Dafoe, Baobao Zhang, and Owain Evans. When will ai exceed human performance? evidence from ai experts. Journal of Artificial Intelligence Research, 62:729–754, 2018.
- [45] Yoshua Bengio, Sören Mindermann, Daniel Privitera, Tamay Besiroglu, Rishi Bommasani, Stephen Casper, Yejin Choi, Danielle Goldfarb, Hoda Heidari, Leila Khalatbari, et al. International scientific report on the safety of advanced ai (interim report). arXiv preprint arXiv:2412.05282, 2024.
- [46] Miles Wang, Robi Lin, Kat Hu, Joy Jiao, Neil Chowdhury, Ethan Chang, and Tejal Patwardhan. Frontierscience: Evaluating ai’s ability to perform expert-level scientific tasks. arXiv preprint arXiv:2601.21165, 2026.
- [47] Elliot Glazer, Ege Erdil, Tamay Besiroglu, Diego Chicharro, Evan Chen, Alex Gunning, Caroline Falkman Olsson, Jean-Stanislas Denain, Anson Ho, Emily de Oliveira Santos, et al. Frontiermath: A benchmark for evaluating advanced mathematical reasoning in ai. arXiv preprint arXiv:2411.04872, 2024.
- [48] Parshin Shojaee, Ngoc-Hieu Nguyen, Kazem Meidani, Amir Barati Farimani, Khoa D Doan, and Chandan K. Reddy. LLM-SRBench: A new benchmark for scientific equation discovery with large language models. In International Conference on Machine Learning, 2025.
- [49] Andy Zou, Tristan Xiao, Ryan Jia, Joe Kwon, Mantas Mazeika, Richard Li, Dawn Song, Jacob Steinhardt, Owain Evans, and Dan Hendrycks. Forecasting future world events with neural networks. In Advances in Neural Information Processing Systems, 2022.
- [50] Danny Halawi, Fred Zhang, Chen Yueh-Han, and Jacob Steinhardt. Approaching human-level forecasting with language models. In Advances in Neural Information Processing Systems, 2024.
- [51] Siyuan Wang, Zhuohan Long, Zhihao Fan, Xuan-Jing Huang, and Zhongyu Wei. Benchmark self-evolving: A multi-agent framework for dynamic llm evaluation. In International conference on computational linguistics, 2025.
- [52] Shirin Shahabi, Spencer Graham, and Haruna Isah. Truthtensor: Evaluating llms through human imitation on prediction market under drift and holistic reasoning. arXiv preprint arXiv:2601.13545, 2026.
- [53] Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, et al. Livebench: A challenging, contamination-limited llm benchmark. arXiv preprint arXiv:2406.19314, 2024.
- [54] Alexander Krauss. Debunking revolutionary paradigm shifts: evidence of cumulative scientific progress across science. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 480(2302), 2024.

- [55] Kambiz N Alavian. Paradigm shifts as portals to threshold concepts and epistemic transformation. Educational Philosophy and Theory, pages 1–12, 2025.
- [56] John E Hallsworth, Zulema Udaondo, Carlos Pedrós-Alió, Juan Höfer, Kathleen C Benison, Karen G Lloyd, Radamés JB Cordero, Claudia BL de Campos, Michail M Yakimov, and Ricardo Amils. Scientific novelty beyond the experiment. Microbial Biotechnology, 16(6):1131–1173, 2023.
- [57] Tairan Wang, Jianyu Hu, Runhai Ouyang, Yutao Wang, Yi Huang, Sulei Hu, and Wei-Xue Li. Nature of metal-support interaction for metal catalysts on oxide supports. Science, 386(6724):915–920, 2024.
- [58] Andrew K Schulz, Lena V Kaufmann, Lawrence T Smith, Deepti S Philip, Hilda David, Jelena Lazovic, Michael Brecht, Gunther Richter, and Katherine J Kuchenbecker. Functional gradients facilitate tactile sensing in elephant whiskers. Science, 391(6786):712–718, 2026.
- [59] Geosan Kang, Guhyeon Kwon, Jiwoon Jeon, Jisung Kwon, Myung-Ki Kim, Junpyo Hong, Albert S Lee, Seongi Lee, Binhyung Lee, Yujin Kim, et al. Electromagnetic interference shielding using metal and mxene thin films. Nature, pages 1–8, 2025.
- [60] Jane Luo, Xin Zhang, Steven Liu, Jie Wu, Jianfeng Liu, Yiming Huang, Yangyu Huang, Chengyu Yin, Ying Xin, Yuefeng Zhan, et al. Rpg: A repository planning graph for unified and scalable codebase generation. arXiv preprint arXiv:2509.16198, 2025.
- [61] Wenzheng Heng, Shukun Yin, Jihong Min, Canran Wang, Hong Han, Ehsan Shirzaei Sani, Jiahong Li, Yu Song, Harry B Rossiter, and Wei Gao. A smart mask for exhaled breath condensate harvesting and analysis. Science, 385(6712):954–961, 2024.
- [62] Milos Vukadinovic, I-Min Chiu, Xiu Tang, Neal Yuan, Tien-Yu Chen, Paul Cheng, Debiao Li, Susan Cheng, Bryan He, and David Ouyang. Comprehensive echocardiogram evaluation with view primed vision language ai. Nature, 650(8103):970–977, 2026.
- [63] Hao Yu, Haotong Lin, Jiawei Wang, Jiaxin Li, Yida Wang, Xueyang Zhang, Yue Wang, Xiaowei Zhou, Ruizhen Hu, and Sida Peng. Infinidepth: Arbitrary-resolution and fine-grained depth estimation with neural implicit fields. arXiv preprint arXiv:2601.03252, 2026.

### A Benchmark Construction Details

#### A.1 Data acquisition and source construction

Natural Science Data: We extract primary claims, experimental outcomes, and contextual data from the publication logs of Nature, Science, and Cell to capture foundational breakthroughs in the physical and life sciences. We restrict the natural science corpus strictly to these high-impact, peer-reviewed journals to guarantee that the forecasted milestones in biology, chemistry, and physics represent rigorously validated empirical discoveries rather than preliminary hypotheses or unverified preprints. To account for the potential temporal lag between initial preprints and formal journal publication, we query a combination of academic APIs, including Crossref, Semantic Scholar, OpenAlex, Europe PMC, arXiv, and bioRxiv/medRxiv, and define the earliest observed date of each manuscript’s DOI across these sources as a strict knowledge cutoff to prevent temporal leakage.

Artificial Intelligence Data: We incorporate artificial intelligence as a core evaluation of CUSP because algorithmic advancements increasingly dictate the pace and direction of discovery across the natural sciences. For example, breakthroughs like AlphaFold have been fundamentally catalyzed by both prior computational innovations–especially attention mechanisms and transformer architectures–and the availability of large-scale dataset resources such as the Protein Data Bank [37]. Accurately anticipating milestones such as new architectures, algorithms, and datasets is therefore essential for forecasting the broader trajectory of scientific progress. Furthermore, evaluating frontier models on AI-specific targets provides a rigorous empirical basis for assessing their capacity to reason about the evolution of their underlying technologies.

CUSP includes high-visibility AI papers from dynamic, community-driven repositories, specifically incorporating widely acknowledged “Top 10 AI Papers of the Week” lists and the Hugging Face Top Papers hub. To filter top papers from Hugging Face, we rank the most impactful research using a hybrid impact score that balances community engagement (upvotes) against academic traction (citations). The score is given by upvotes+ 5×[citations/(months old+ 1)] and provides an age-adjusted metric to ensure that recent, high-velocity publications remain competitive with older, more highly cited works. We subsequently select the highest-scoring papers from each month.

Leaderboard Forecasting Data: In addition to publication-based milestones, we incorporate forecasting targets from widely used AI benchmarks and leaderboards, which provide standardized and time-resolved measurements of progress in machine learning. Representative benchmarks include GPQA Diamond, MMLU-Pro, and Humanity’s Last Exam, along with domain-specific leaderboards tracking rapid capability improvements. Forecasting these benchmarks tests whether AI systems can extrapolate how capabilities evolve over time, rather than recall past results. Compared to heterogeneous scientific discoveries, leaderboard progress is more structured and temporally dense, offering a complementary setting to assess whether models can internalize and predict technological advancement. CUSP combines these targets with publication-derived milestones to evaluate whether AI systems exhibit a coherent sense of progress across both performance scaling and scientific breakthroughs.

#### A.2 Corpus Extraction and Automated Filtering

To construct a benchmark to evaluate scientific forecasting, we focus on extracting verifiable scientific milestones from papers that contain concrete results for predictive evaluation. We collect top-tier publications across multiple disciplines and restrict data extraction to the title, publication metadata, and abstract of each paper to ensure a consistent and standardized representation of each milestone. Abstracts provide a concise and structured representation of scientific contributions in papers, summarizing the primary claims, core methods, and key quantitative outcomes. This design also ensures reproducibility and accessibility, as abstracts are uniformly available across publication venues.

We design an LLM-based agentic pipeline to extract and filter candidate milestones from the collected abstracts automatically. The extraction stage identifies statements corresponding to concrete scientific results or capabilities, while the filtering stage operates as a strict binary classifier that accepts only entries containing at least one verifiable and measurable outcome. This process ensures that each retained instance corresponds to a well-defined prediction target.

To improve precision, we incorporate domain-aware filtering by classifying each abstract into a scientific domain (e.g., AI, Chemistry, Biology, Physics, or General Science) and applying domainspecific acceptance criteria. For computational research, accepted entries must report validated technical advances on recognized benchmarks or provide explicit performance metrics. For experimental sciences, entries must include measurable quantities (e.g., binding affinities, fold changes) or clearly defined physical or biological properties.

We explicitly reject abstracts that are purely descriptive, speculative, or review-oriented, as these lack quantitative outcomes suitable for predictive evaluation. For each decision, the pipeline generates a concise summary of the extracted result for accepted entries or a justification for rejection. This automated procedure ensures scalability and consistency while minimizing subjective annotation bias. Box F.2 illustrates that even high-impact publications may fail to meet the criteria of verifiable scientific milestones.

#### A.3 Task synthesis procedure

To synthesize forecasting tasks from each accepted scientific milestone, we first decompose the source abstract into three structured fields. This decomposition separates the scientific problem, the technical solution and the resulting measurable outcome before any task generation step. We avoid novel acronyms, method names and system names introduced in the source paper, so that models evaluated under a historical knowledge cutoff cannot identify a discovery from post-cutoff terminology.

Problem Statement: A technical description focusing exclusively on the research problem and the

limitations of preceding methods.

Technical Approach: A detailed, method-oriented specification of the mechanism, experimental

design or architectural innovation.

Results and Metrics: A single-sentence summary capturing only quantitative outcomes, perfor-

mance numbers or benchmark results. The task-generation pipeline uses these fields to construct five evaluation formats. Binary questions

assess whether the milestone claim is feasible by the target date. Perturbed binary questions modify the original claim to create plausible but unsupported alternatives. Multiple-choice questions ask models to identify the correct technical approach among expert-level distractors. Free-response questions ask models to propose an implementation strategy from the problem context. Date prediction tasks ask models to forecast the month and year in which the milestone is realized. We use GPT-4o to convert structured abstract decompositions into candidate binary, perturbed binary, MCQ, FRQ, and date-prediction tasks, including extraction of the candidate correct answer from the source abstract. Final task inclusion and answer validity are determined by the source publication itself and independently checked by Grok-3 and human review.

Representative examples of each task format are provided in Appendix H.

#### A.4 Key Statistics and Distributional Properties

We analyze the temporal, structural, and linguistic properties of CUSP to characterize the diversity and complexity of the benchmark.

Temporal Distribution. CUSP spans a continuous time horizon from January 2024 to March 2026, covering 27 active months of scientific progress. The dataset includes 4,760 milestones with valid publication timestamps, with all 27 months represented. The distribution of milestones over time is relatively stable, with an average of 176.3 papers per month (± 51.5). This temporal structure enables controlled evaluation across different forecasting horizons, supporting analysis of short-term versus long-term prediction and the degradation of model performance as temporal distance increases.

Task Composition. Each scientific milestone is decomposed into multiple task formats designed to probe complementary aspects of scientific forecasting. In total, the validated dataset contains 17,429 task instances across five task types. The distribution of task types is non-uniform, reflecting strict validation and filtering constraints: 4,128 multiple-choice questions (MCQs), 4,135 free-response (FRQ) prompts, 3,656 perturbed binary tasks, and 2,755 each of binary and date prediction tasks. Task availability varies across milestones, resulting in a heterogeneous task density. Specifically, 33.9% of papers admit all five task types, while 20.1%, 27.1%, 15.9%, and 3.0% of papers admit four, three, two, and one task(s), respectively. This sparsity arises from strict LLM-as-a-judge validation filtering, which removes tasks that lack verifiability, faithfulness, or logical consistency under perturbation. As a result, CUSP prioritizes task reliability and scientific validity over uniformity.

Label Distribution. For binary questions, the ground-truth answer is yes, whereas perturbed binary questions have the ground-truth answer no, resulting in an overall yes/no label distribution of approximately 0.75. For multiple-choice questions (MCQs), the correct answer is initially assigned to option A, and answer choices are randomly shuffled during evaluation to ensure a uniform distribution of correct options. For date prediction tasks, the distribution of ground-truth dates remains relatively stable across the dataset (Figure 2).

Question Length and Complexity. We analyze the linguistic characteristics of generated tasks across formats. Binary questions have an average length of 29.2 words (variance 122.5), while MCQs average 36.6 words (variance 47.8) and FRQs average 41.8 words (variance 17.8). Problem

statements, which encode the underlying scientific challenge, are substantially longer, with an average length of 70.6 words (variance 107.0).

These differences reflect increasing levels of reasoning complexity across task types. Problem statements provide rich scientific context, FRQs require open-ended methodological synthesis, and MCQs demand precise discriminative reasoning over expert-level distractors.

Domain and Subcategory Diversity. CUSP spans nine top-level scientific domains and 4,245 distinct subcategories, reflecting highly specialized research topics across disciplines. The dataset is dominated by biology (1,234 papers) and artificial intelligence (1,141 papers), followed by medicine (746), neuroscience (403), materials science (375), physics (359), environmental science (235), chemistry (203), and other domains (64).

### B Benchmark validation

Because CUSP evolves continuously with newly published discoveries, it is essential to maintain an automated mechanism for verifying the faithfulness and quality of its generated questions. To this end, we develop a validation framework that employs a large language model (LLM) judge to evaluate each question against its source abstract. Since CUSP questions are generated using GPT-4o, we perform validation using a distinct model, Grok-3 (xAI). As a frontier model with comparable scale and capability but a different architecture and training distribution, Grok-3 provides an independent evaluation, mitigating potential self-evaluation bias.

Binary and Perturbed Binary Questions. We first validate binary and perturbed binary questions under three criteria:

- • Faithfulness: The binary statement (excluding the temporally conditioned phrasing) must accurately reflect a concrete claim in the source abstract. The validator checks for discrepancies in entities, conditions, outcomes, or quantitative details, and rejects questions that introduce unsupported or altered claims.
- • Verifiability: The binary statement must correspond to a concrete, objectively evaluable claim. We reject questions whose underlying statements are vague, underspecified, or lack a clear operational meaning, ensuring that each question admits a well-defined yes/no answer.
- • Perturbation Validity: For perturbed binary questions, we verify that the perturbation introduces a meaningful and non-trivial modification to the original claim (e.g., modifying thresholds or adding unmet constraints), such that the perturbed version is no longer directly supported by the source abstract. This prevents trivial or paraphrased perturbations.

Multiple-Choice Questions (MCQ). For MCQ tasks, we decompose validation into three components. First, we ensure that the problem statement is faithful to the source abstract and accurately captures the underlying scientific challenge. Second, we verify that the correct answer choice corresponds to a valid technical approach or mechanism that is supported or implied by the abstract. Third, we evaluate the distractor options, ensuring that incorrect choices are plausible yet not directly supported by the source. This prevents trivial elimination and enforces that MCQs require genuine mechanistic reasoning rather than superficial pattern matching.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

##### A Benchmark Validation B

##### Two-Track LLM-as-a-Judge

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Model Outputs

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

###### + +

[Figure 59]

[Figure 60]

[Figure 61]

Findings Curation Criteria

[Figure 62]

[Figure 63]

[Figure 64]

gpt-4o

[Figure 65]

[Figure 66]

[Figure 67]

Outcome Track Reasoning Track

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Potential Error

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

CUSP Version-1

[Figure 77]

[Figure 78]

[Figure 79]

Binary MCQ FRQ Response

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Discard Examples

[Figure 86]

Viable? Novel?

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Yes C

[Figure 92]

-

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Date Prediction

[Figure 103]

###### +

###### + +

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Leak? LLM + Web

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Validation Criteria

[Figure 112]

[Figure 113]

[Figure 114]

grok-3 Questions Findings

05-2028

[Figure 115]

[Figure 116]

[Figure 117]

###### +

[Figure 118]

###### Human Validation

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

###### Validated

[Figure 123]

[Figure 124]

[Figure 125]

Human Benchmark Valuation

Joint Score

[Figure 126]

[Figure 127]

- Figure 6. CUSP validation and evaluation pipeline. (A) Benchmark construction: scientific findings are curated and filtered via LLM-based criteria, validated by an independent model, and verified by human experts to produce a high-quality benchmark. (B) Two-track evaluation: model outputs are assessed for outcome correctness (across binary, MCQ, free-response, and date prediction tasks) and reasoning quality (viability, novelty, and leakage), which are combined into a joint score.

Free-Response Questions (FRQ). For open-ended free-response tasks, direct validation of generated answers is inherently ambiguous due to the existence of multiple valid solutions. Instead, we validate the problem context and background description, ensuring that the prompt is faithful to the source abstract and accurately reflects the scientific problem being addressed.

All validation criteria are implemented as separate LLM judgments with structured outputs (verdict, score, and explanation). Based on these evaluations, we perform fine-grained filtering at the field level, removing only invalid components (e.g., a faulty binary or perturbed question) while preserving the remainder of the sample. This enables scalable quality control without sacrificing dataset coverage. Additionally, we remove date prediction tasks when the corresponding binary question is filtered out, as both depend on the same underlying verifiable result. See Appendix F.2 for the full set of validation prompts. To assess the reliability of this automated validation framework, we also conducted a human evaluation, see Appendix E.1.

### C Related Works

#### C.1 Comparison to Related Benchmarks

A growing body of work has introduced benchmarks for evaluating forecasting ability and scientific reasoning in AI systems. Table 1 summarizes the relationship between CUSP and these prior efforts. Recent forecasting benchmarks such as ForecastBench [21], FutureX [22], FOReCAst [23], and PROPHET [24] evaluate the ability of models to predict future events, often incorporating temporal reasoning and, in some cases, confidence calibration or retrieval. However, these benchmarks focus primarily on general-world or news-driven events and do not explicitly target scientific discovery. In addition, these benchmarks do not explicitly control for model knowledge cutoffs, making it difficult to distinguish genuine forecasting ability from memorization or indirect exposure to future information.

In parallel, scientific reasoning and discovery benchmarks such as Humanity’s Last Exam [18], AstaBench [17], PreScience [19], ResearchBench [25], and ScienceQA [15] evaluate models on tasks including expert-level question answering, hypothesis generation, and structured reasoning over scientific content. While these benchmarks provide strong tests of domain knowledge and reasoning, they are inherently retrospective: the correct answers are known at evaluation time, and models are not required to anticipate future discoveries. In addition, many widely used scientific reasoning benchmarks are increasingly saturated by frontier AI systems, limiting their ability to discriminate further capability improvements. Benchmarks such as MMLU-Pro, GPQA Diamond, and MedQA now exhibit near-ceiling performance for leading models, despite continuing gaps in scientific forecasting ability. CUSP remains substantially unsaturated across all evaluated frontier systems (Figure 11), suggesting that forecasting scientific progress constitutes a qualitatively more challenging setting than retrospective reasoning over established scientific knowledge.

CUSP differs fundamentally from both lines of work. Unlike forecasting benchmarks, it is explicitly grounded in scientific discovery, with all tasks derived from real, peer-reviewed breakthroughs across multiple disciplines. Unlike existing scientific benchmarks, it introduces a temporal cutoff that restricts models to knowledge available prior to each milestone, thereby requiring genuine forward-looking prediction rather than retrospective reasoning. Furthermore, CUSP employs a multi-task evaluation framework that combines binary feasibility prediction, calibration under perturbation, mechanistic reasoning, generative proposal, and date prediction within a unified setting. These properties position CUSP as the first benchmark to systematically evaluate AI systems as epistemic forecasters of scientific progress, bridging the gap between general forecasting and scientific reasoning benchmarks.

#### C.2 AI for Science

AI for Scientific Discovery. AI for Science is a rapidly expanding field that is increasingly integrated into scientific discovery, spanning from computational tools to automated research systems that support full research workflows [38, 39, 40, 41, 42]. While prior work has demonstrated the potential of large language models (LLMs) to assist scientific discovery, these approaches typically rely on human researchers to define the problems and directions to explore [43, 44, 45]. As a result, it remains unclear whether LLMs can independently identify promising and feasible research directions and proactively explore the scientific frontier. This work makes an initial exploration by studying the capability boundaries of LLMs in scientific forecasting. To enable this investiga-

tion, we introduce CUSP, a benchmark and scalable evaluation framework that operationalizes scientific forecasting as a measurable capability, requiring models to predict concrete and verifiable discoveries that emerge only after a strict knowledge cutoff date.

Scientific and LLM Benchmarks as Retrospective Evaluation. Current evaluation paradigms for LLMs in scientific domains largely rely on static benchmarks, including multitask knowledge assessments, graduate-level question answering, Olympiad-style reasoning problems, and rubric-based evaluations of intermediate reasoning [20, 16, 46, 47, 18]. In these settings, ground-truth answers can be verified against existing literature, thereby framing evaluation as retrospective reasoning over established knowledge. While such benchmarks have been instrumental in measuring factual recall and problem-solving ability, they do not assess a model’s scientific forecasting capacity [48]. Recent efforts, such as Matter-of-Fact [26], incorporate explicit temporal cutoffs in benchmark construction. However, they focus primarily on verifying the feasibility of proposed hypotheses within material science, rather than evaluating forward-looking ability. In contrast, our evaluation protocol adopts a prospective setting: models are conditioned only on knowledge available up to a strict temporal cutoff and are required to forecast concrete, verifiable breakthroughs that are subsequently validated in high-impact venues such as Nature, Science, and leading AI conferences.

Dynamic and Living Evaluation Paradigms. A recent new dynamic evaluation has emerged as a robust solution to the contamination issues inherent in static benchmarks [22, 21, 49, 50]. This paradigm shift redefines benchmarking AI beyond just static datasets, but rather as a new measurement standard that can quantitatively assess a solution space in real time [51]. However, most existing dynamic evaluations focus on economic, political, or event-based forecasting tasks, and do not explicitly target the structure of scientific progress [52, 53]. Scientific discovery is inherently non-stationary, shaped by methodological innovations, cross-disciplinary transfer, and abrupt conceptual shifts. An evaluation framework designed for this setting must therefore probe whether models capture evolving research dynamics rather than merely interpolate historical trends [54, 55, 56]. Positioning scientific forecasting within a dynamic benchmarking paradigm enables the assessment of extrapolative reasoning as a core capability, which aligns evaluation with the moving frontier of scientific knowledge itself.

### D Extended Results

#### D.1 Tool Calling Results

Web Search (no cutoff) Web Search (with cutoff) Metric Baseline Value ∆ p-value Value ∆ p-value

###### GPT-4o

[Figure 128]

Binary (original) 0.192±0.023 0.838±0.022 +0.646 < 0.001 *** 0.476±0.029 +0.287 < 0.001 *** Binary (perturbed) 0.794±0.021 0.735±0.022 −0.059 0.046 * 0.629±0.024 −0.165 < 0.001 *** MCQ 0.542±0.024 0.873±0.016 +0.331 < 0.001 *** 0.589±0.024 +0.046 0.062

- FRQ score (0–10) 3.278±0.047 4.249±0.081 +0.971 < 0.001 *** 3.715±0.047 +0.432 < 0.001 *** Date score (0–1) 0.183±0.014 0.621±0.025 +0.438 < 0.001 *** 0.382±0.018 +0.201 < 0.001 *** Date exact match 0.014±0.007 0.471±0.029 +0.457 < 0.001 *** 0.061±0.014 +0.047 < 0.001 *** Date month error 34.465±2.227 6.408±1.004 −28.057 < 0.001 *** 20.655±1.751 −15.215 < 0.001 *** GPT-5.4 Binary (original) 0.647±0.028 0.973±0.009 +0.325 < 0.001 *** 0.899±0.018 +0.253 < 0.001 *** Binary (perturbed) 0.387±0.025 0.318±0.024 −0.069 0.026 * 0.363±0.024 −0.025 0.468 MCQ 0.841±0.018 0.956±0.010 +0.115 < 0.001 *** 0.897±0.015 +0.055 0.005 ** FRQ score (0–10) 5.052±0.040 5.683±0.051 +0.631 < 0.001 *** 5.867±0.048 +0.809 < 0.001 *** Date score (0–1) 0.275±0.018 0.748±0.021 +0.472 < 0.001 *** 0.643±0.021 +0.369 < 0.001 *** Date exact match 0.027±0.009 0.498±0.029 +0.471 < 0.001 *** 0.341±0.028 +0.314 < 0.001 *** Date month error 38.757±4.377 7.533±2.592 −31.224 < 0.001 *** 10.493±1.116 −28.926 < 0.001 ***

[Figure 129]

[Figure 130]

DeepSeek R1 Binary (original) 0.469±0.029 0.779±0.024 +0.310 < 0.001 *** 0.616±0.029 +0.151 < 0.001 *** Binary (perturbed) 0.481±0.025 0.437±0.025 −0.044 0.136 0.444±0.025 −0.036 0.219 MCQ 0.599±0.025 0.771±0.021 +0.172 < 0.001 *** 0.676±0.024 +0.080 0.004 **

- FRQ score (0–10) 4.232±0.042 4.691±0.049 +0.459 < 0.001 *** 4.333±0.043 +0.097 0.050 * Date score (0–1) 0.289±0.017 0.544±0.020 +0.255 < 0.001 *** 0.470±0.020 +0.179 < 0.001 *** Date exact match 0.017±0.008 0.214±0.024 +0.197 < 0.001 *** 0.130±0.020 +0.113 < 0.001 *** Date month error 20.372±1.913 11.433±1.506 −8.939 < 0.001 *** 15.703±1.975 −4.557 0.003 ** Claude S4.5 Binary (original) 0.285±0.027 0.751±0.026 +0.466 < 0.001 *** 0.518±0.030 +0.230 < 0.001 *** Binary (perturbed) 0.724±0.024 0.638±0.025 −0.086 0.005 ** 0.699±0.024 −0.026 0.402 MCQ 0.738±0.022 0.810±0.020 +0.072 0.010 * 0.750±0.022 +0.008 0.836 FRQ score (0–10) 3.951±0.050 4.540±0.056 +0.589 < 0.001 *** 4.292±0.051 +0.333 < 0.001 *** Date score (0–1) 0.258±0.017 0.484±0.024 +0.226 < 0.001 *** 0.377±0.024 +0.119 < 0.001 *** Date exact match 0.029±0.010 0.242±0.026 +0.213 < 0.001 *** 0.175±0.023 +0.146 < 0.001 *** Date month error 25.539±1.948 15.479±6.201 −10.060 0.102 14.453±1.586 −14.194 0.002 **

[Figure 131]

- Table 8. Web-search augmentation across models on matched 500-question subsets. Values are means ± SE. ∆ is relative to the Baseline. Significance: *** p < 0.001, ** p < 0.01, * p < 0.05, n.s. not significant.

Metric GPT-5.4 (Aug 2025) DeepSeek R1 (Jul 2024) Claude S4.5 (Jan 2025)

[Figure 132]

[Figure 133]

[Figure 134]

Pre (n = 377) Post (n = 123) Pre (n = 119) Post (n = 381) Pre (n = 214) Post (n = 286)

Binary 0.65 / 0.97∗∗∗ / 0.92∗∗∗ 0.64 / 0.98∗∗∗ / 0.81∗ 0.35 / 0.70∗∗∗ / 0.72∗∗∗ 0.52 / 0.82∗∗∗ / 0.57 0.33 / 0.75∗∗∗ / 0.72∗∗∗ 0.25 / 0.75∗∗∗ / 0.32 Binary (pert.) 0.36 / 0.32 / 0.38 0.46 / 0.30∗ / 0.30∗∗ 0.59 / 0.52 / 0.46 0.45 / 0.42 / 0.44 0.64 / 0.65 / 0.64 0.78 / 0.62∗∗∗ / 0.75 MCQ 0.85 / 0.97∗∗∗ / 0.94∗∗∗ 0.82 / 0.92∗∗ / 0.77 0.67 / 0.82∗ / 0.78∗ 0.57 / 0.76∗∗∗ / 0.64∗ 0.77 / 0.82 / 0.81 0.72 / 0.81∗ / 0.70 FRQ (0–10) 5.06 / 5.76∗∗∗ / 6.06∗∗∗ 5.06 / 5.43∗∗∗ / 5.29∗ 4.21 / 4.66∗∗∗ / 4.50∗∗ 4.24 / 4.70∗∗∗ / 4.29 3.96 / 4.74∗∗∗ / 4.61∗∗∗ 3.95 / 4.40∗∗∗ / 4.07 Date (0–1) 0.28 / 0.75∗∗∗ / 0.72∗∗∗ 0.24 / 0.75∗∗∗ / 0.31 0.16 / 0.52∗∗∗ / 0.55∗∗∗ 0.35 / 0.56∗∗∗ / 0.43∗∗∗ 0.25 / 0.57∗∗∗ / 0.49∗∗∗ 0.27 / 0.39∗∗∗ / 0.28

- Table 9. Per-metric scores split by each model’s training knowledge cutoff. Pre-cutoff: publication date ≤ training cutoff; post-cutoff: publication date > training cutoff (all 500 questions classified using benchmark metadata). Each cell shows Baseline / WS / WS+Cutoff (means). Significance of WS and WS+Cutoff vs. Baseline: ∗∗∗p < 0.001, ∗∗p < 0.01, ∗p < 0.05.

Task Metric GPT-5.4 (Aug 2025) DeepSeek R1 (Jul 2024) Claude S4.5 (Jan 2025)

[Figure 135]

[Figure 136]

[Figure 137]

Pre (n = 377) Post (n = 123) Pre (n = 119) Post (n = 381) Pre (n = 214) Post (n = 286)

Binary

ECE↓ 0.109 / 0.027 / 0.021 0.152 / 0.039 / 0.060 0.215 / 0.118 / 0.081 0.221 / 0.061 / 0.105 0.401 / 0.147 / 0.190 0.448 / 0.127 / 0.426 Brier↓ 0.238 / 0.025 / 0.059 0.242 / 0.016 / 0.137 0.186 / 0.135 / 0.115 0.146 / 0.093 / 0.164 0.370 / 0.138 / 0.169 0.391 / 0.125 / 0.356 Gap +0.067 / −0.021 / +0.013 +0.071 / −0.029 / −0.009 +0.213 / +0.115 / +0.075 +0.045 / −0.000 / +0.090 +0.401 / +0.104 / +0.152 +0.448 / +0.076 / +0.426

Binary (pert.)

ECE↓ 0.338 / 0.505 / 0.468 0.211 / 0.527 / 0.461 0.497 / 0.511 / 0.487 0.564 / 0.546 / 0.533 0.187 / 0.204 / 0.274 0.193 / 0.219 / 0.164 Brier↓ 0.350 / 0.487 / 0.486 0.296 / 0.518 / 0.450 0.397 / 0.467 / 0.465 0.442 / 0.484 / 0.450 0.266 / 0.273 / 0.282 0.213 / 0.294 / 0.209 Gap +0.338 / +0.505 / +0.468 +0.211 / +0.511 / +0.461 −0.039 / +0.206 / +0.294 +0.141 / +0.292 / +0.199 +0.053 / +0.130 / +0.102 −0.119 / +0.115 / −0.016

MCQ

ECE↓ 0.040 / 0.029 / 0.040 0.041 / 0.020 / 0.082 0.252 / 0.120 / 0.159 0.332 / 0.169 / 0.309 0.074 / 0.077 / 0.068 0.040 / 0.088 / 0.052 Brier↓ 0.114 / 0.028 / 0.051 0.133 / 0.059 / 0.169 0.279 / 0.155 / 0.191 0.353 / 0.193 / 0.326 0.166 / 0.063 / 0.075 0.202 / 0.098 / 0.185 Gap +0.003 / −0.014 / −0.015 −0.005 / +0.007 / +0.049 +0.246 / +0.110 / +0.159 +0.332 / +0.169 / +0.309 −0.051 / −0.029 / −0.041 −0.008 / −0.003 / +0.030

Date

ECE↓ 0.321 / 0.181 / 0.166 0.339 / 0.272 / 0.371 0.497 / 0.264 / 0.236 0.278 / 0.252 / 0.260 0.311 / 0.158 / 0.172 0.251 / 0.317 / 0.237 Brier↓ 0.190 / 0.107 / 0.106 0.188 / 0.161 / 0.202 0.280 / 0.159 / 0.147 0.156 / 0.156 / 0.174 0.171 / 0.107 / 0.104 0.131 / 0.187 / 0.140 Gap +0.321 / +0.173 / +0.166 +0.339 / +0.263 / +0.371 +0.494 / +0.264 / +0.236 +0.278 / +0.252 / +0.260 +0.311 / +0.158 / +0.160 +0.251 / +0.317 / +0.230

- Table 10. Confidence calibration split by each model’s training cutoff. Pre-cutoff: publication date ≤ training cutoff; post-cutoff: publication date > training cutoff (all 500 questions classified via benchmark metadata). Each cell shows No-WS / WS / WS+Cutoff values. ECE = Expected Calibration Error; Brier = Brier score; Gap = c¯−a¯ (positive = overconfident). ↓ lower is better.

GPT-5.4 + Full Web Search Overlooks Specific Size Constraint

Perturbed Question: By November 2024, will a method show that metal-metal interactions dominate oxide support effects for particle sizes below 2 nm?

Perturbation Detail: Added a specific unmet constraint regarding particle size uniformity below 2 nm.

Source Abstract:

The metal-support interaction is one of the most important pillars in heterogeneous catalysis, but developing a fundamental theory has been challenging because of the intricate interfaces. Based on experimental data, interpretable machine learning, theoretical derivation, and first-principles simulations, we established a general theory of metal-oxide interactions grounded in metal-metal and metal-oxygen interactions. The theory applies to metal nanoparticles and atoms on oxide supports and oxide films on metal supports. We found that for late-transition metal catalysts, metal-metal interactions dominated the oxide support effects and suboxide encapsulation over metal nanoparticles. A principle of strong metal-metal interactions for encapsulation occurrence is formulated and substantiated by extensive experiments including 10 metals and 16 oxides. The valuable insights revealed on (strong) metal-support interaction advance the interfacial design of supported metal catalysts. [57]

Model Response (GPT-5.4 + Web Search): “Yes. By November 2024, a Science paper reported a general theory/method for metal–oxide interactions and

explicitly concluded that, for late-transition metal catalysts, metal–metal interactions dominate oxide support effects; this appears to satisfy the question’s threshold.”

Error Analysis:

The model commits a False Positive (Missed Constraint). It successfully matches the broad scientific themes of the text but completely ignores the artificially inserted < 2 nm requirement, incorrectly confirming the threshold was met.

GPT-5.4 + Web Search Fails via Speculative Extrapolation

###### Perturbed Question:

Aider Polyglot evaluates the ability of code language models to perform diverse code-editing tasks based on natural language instructions while retaining their general code generation and problem-solving capabilities. It measures model performance using synthetic datasets of code-editing scenarios and assesses robustness through adaptation algorithms designed to prevent overfitting. By Jun 2025, do you think there will be a publicly reported AI system that achieves 90% or higher on Aider Polyglot?

Perturbation Detail: Inflated threshold = 90%

Model Response (GPT-5.4 Web Search Unconstrained):

“Yes. Public aider leaderboard/results pages show top polyglot scores already reached about 85% in late 2024/early 2025, and given the rapid improvement pace in code-editing models through 2025, it was more likely than not that at least one publicly reported system would cross 90% by June 2025.”

Error Analysis:

The model commits a False Positive (Speculative Hallucination / Overconfident Extrapolation). Despite the rigid 90% threshold, the model grounds its “Yes” answer in data that only supports 85%. It improperly substitutes a forecast (“more likely than not”) for factual satisfaction of the benchmark criteria.

#### D.2 FRQ Sub-dimension Score Analysis

Sub-dimension profile (normalized to dimension max)

FRQ scientific prediction quality — CUSP Benchmark

Specificity (/10)

###### B

FRQ score distribution per model

A

| |5.0|
|---|---|
| | |
| | |
| | |
| |3.2<br><br>3.5<br><br>4.0 4.0<br><br>4.2<br><br>|
| | |
| | |

| |0.25<br><br>0.50<br><br>0.75<br><br>1.0<br><br>|
|---|---|
| | |

10

Excellent

8

FRQscore(0–10)

Pass

6

Novelty (/10)

Alignment (/10)

Near-pass

4

Partial

2

Poor

0

Feasibility (/10)

GPT-5.4

GPT-OSS 20B

GPT-4oLLaMA3.3GPT-OSS20BClaudeS4.5DeepSeekR1 GPT-5.4

DeepSeek R1

LLaMA 3.3

Claude S4.5

GPT-4o

- Figure 7. A) Visualization of FRQ evaluation on four criteria on 6 LLMs B) Frq Score distribution per model.

0.0 0.2 0.4 0.6 0.8 1.0

Fraction of predictions

GPT-4o

LLaMA 3.3

GPT-OSS 20B

Claude S4.5

DeepSeek R1

GPT-5.4

A

67%

68%

47%

36%

33%

22%

28%

39%

48%

46%

33%

12%

14%

20%

60%

Prediction quality grade distribution (% per model)

Poor

Partial

| |
|---|

Near-pass

Pass

| |
|---|

Excellent

[Figure 138]

Alignment Specificity Novelty Feasibility

GPT-4o

LLaMA 3.3

GPT-OSS 20B

Claude S4.5

DeepSeek R1

GPT-5.4

B

2.2

- (0.22)

2.9 (0.29)

2.4 (0.24)

5.5 (0.55)

2.2

- (0.22)

3.3 (0.33)

2.4 (0.24)

6.0 (0.60)

2.1 (0.21)

4.9 (0.49)

3.7 (0.37)

4.8 (0.48)

2.3

- (0.23)

4.7 (0.47)

4.0 (0.40)

5.0 (0.50)

2.3

- (0.23)

5.3 (0.53)

4.1 (0.41)

5.0 (0.50)

3.3 (0.33)

6.2 (0.62)

4.9 (0.49)

5.7 (0.57)

Sub-dimension scores (normalized 0–1) red = low · blue = high

|[Figure 139]| |
|---|---|
| | |
| | |
| | |
| | |

0.0

0.2

0.4

0.6

0.8

1.0

Normalizedscore

FRQ sub-dimension analysis — CUSP Benchmark

- Figure 8. A) Visualization of passing rates across six LLMs B) Visualization on LLM performance across four frq criteria.

###### Model Cutoff Align.↑ Spec.↑ Nov.↑ Spec.−Align. gap Overall score

GPT-OSS Jun 2024 2.06 4.89 3.70 +2.8 3.86 LLaMA 3.3 Dec 2023 2.23 3.33 2.38 +1.1 3.49 GPT-4o Oct 2023 2.25 2.91 2.44 +0.7 3.26

[Figure 140]

[Figure 141]

[Figure 142]

DeepSeek R1 Jul 2024 2.29 5.32 4.07 +3.0 4.18 Claude S4.5 Jan 2025 2.31 4.71 3.95 +2.4 3.99 GPT-5.4 Aug 2025 3.30 6.21 4.91 +2.9 5.04

[Figure 143]

[Figure 144]

[Figure 145]

- Table 11. FRQ sub-dimension profile. Alignment: match with the actual paper method. Specificity: technical concreteness. Novelty: non-obvious insight. Spec.−Align. gap: positive values indicate models that write technically detailed responses but miss the specific paper approach — a signature of plausible-sounding hallucination. Models sorted from lowest to highest alignment.

All sub-dimensions scored 0–10 by the LLM rubric judge. Colour shading on Alignment, Specificity, and Novelty uses the same scale: red < 3, orange 3–4, yellow 4–5, light green 5–6, medium green 6–6.5, dark green ≥ 6.5.

#### D.3 Results by Research Area

Research area n GPT-5.4 Claude S4.5 DeepSeek R1 LLaMA 3.3 GPT-OSS GPT-4o Mean Biology 1058 0.819 0.739 0.652 0.522 0.547 0.570 0.641 AI 964 0.817 0.765 0.520 0.332 0.366 0.456 0.543 Medicine 646 0.769 0.659 0.567 0.443 0.472 0.522 0.572 Neurosci. 351 0.829 0.727 0.666 0.532 0.564 0.629 0.658 Mat. Sci. 356 0.831 0.680 0.567 0.322 0.374 0.480 0.542 Physics 325 0.840 0.702 0.669 0.425 0.526 0.563 0.621 Env. Sci. 185 0.897 0.818 0.621 0.525 0.546 0.568 0.662 Chemistry 185 0.822 0.620 0.441 0.330 0.346 0.432 0.498 Other 58 0.879 0.862 0.780 0.632 0.672 0.793 0.770 Mean 0.834 0.730 0.609 0.451 0.490 0.557 0.612

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

- Table 12. MCQ accuracy (%) by research area. Chance level = 0.25 (4-choice). n: MCQ questions per area. Bold: best model per row. Colour scale: <0.40 0.40–0.50 0.50–0.60 0.60–0.70

0.70–0.80 ≥0.80 .

Research area n GPT-5.4 Claude S4.5 LLaMA 3.3 DeepSeek R1 GPT-OSS GPT-4o Mean Biology 677 0.194 0.185 0.502 0.300 0.271 0.135 0.264 AI 571 0.473 0.464 0.554 0.404 0.447 0.421 0.461 Medicine 336 0.228 0.218 0.506 0.318 0.303 0.136 0.285 Neurosci. 232 0.159 0.199 0.473 0.266 0.303 0.108 0.251 Mat. Sci. 319 0.149 0.130 0.453 0.188 0.232 0.085 0.206 Physics 302 0.143 0.157 0.480 0.252 0.213 0.089 0.222 Env. Sci. 128 0.246 0.174 0.490 0.208 0.281 0.114 0.252 Chemistry 141 0.117 0.128 0.447 0.135 0.209 0.085 0.187 Other 49 0.229 0.284 0.517 0.201 0.260 0.127 0.270 Mean 0.215 0.215 0.491 0.252 0.280 0.144 0.266

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

- Table 13. Date prediction score by research area. Score uses exponential decay (1.0=exact month). n: date-prediction questions per area. Bold: best model per row. Colours reflect the observed score range across this table (5th–95th percentile). Research area n GPT-5.4 Claude S4.5 LLaMA 3.3 DeepSeek R1 GPT-OSS GPT-4o Mean Biology 1576 0.523 0.484 0.455 0.499 0.541 0.526 0.505 AI 1594 0.464 0.602 0.377 0.457 0.548 0.543 0.498 Medicine 847 0.516 0.517 0.432 0.511 0.536 0.567 0.513 Neurosci. 517 0.538 0.473 0.510 0.552 0.520 0.524 0.520 Mat. Sci. 637 0.479 0.453 0.500 0.439 0.473 0.474 0.469 Physics 561 0.510 0.455 0.538 0.460 0.439 0.433 0.472 Env. Sci. 301 0.475 0.502 0.473 0.517 0.525 0.548 0.507 Chemistry 285 0.502 0.482 0.495 0.418 0.449 0.432 0.463 Other 93 0.495 0.495 0.571 0.432 0.441 0.505 0.490 Mean 0.500 0.496 0.484 0.476 0.497 0.506 0.493

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

- Table 14. Binary merged accuracy by research area. Binary merged= 12(original acc.+perturbed acc.), correcting for directional response bias; chance=0.50. n: total binary question pairs (original +perturbed) per area. Bold: best model per row. Colour scale centred at chance (0.50): <0.44

35

0.44–0.48 0.48–0.52 0.52–0.56 0.56–0.60 >0.60 .

- Table 15. Mean FRQ score (0–10) by research area and model. n: number of FRQ instances in the area (max across models). Colour: red < 3, orange 3–4, yellow 4–5, light green 5–6, medium green 6–7, dark green ≥ 7.

Area n GPT-5.4 Claude S4.5 DeepSeek R1 LLaMA 3.3 GPT-OSS GPT-4o Mean Biology 1023 5.13 3.99 4.27 3.60 3.98 3.32 4.05 AI 991 5.06 4.22 4.25 3.38 3.89 3.43 4.04 Medicine 660 5.08 3.96 4.19 3.60 4.01 3.30 4.02 Neurosci. 343 5.27 4.11 4.30 3.59 4.00 3.41 4.11 Mat. Sci. 353 4.82 3.82 4.08 3.42 3.61 3.03 3.80 Physics 311 4.85 3.75 4.02 3.24 3.60 3.00 3.74 Env. Sci. 205 5.13 4.02 4.01 3.65 3.60 3.09 3.92 Chemistry 193 4.49 3.48 3.80 3.23 3.45 2.81 3.54 Other 55 4.88 3.90 3.83 3.45 4.02 3.12 3.87 Overall 5.04 3.99 4.18 3.49 3.86 3.26 3.97

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Chemistry and Physics consistently show lower FRQ alignment, reflecting higher domain specificity and fewer overlapping concepts with general pretraining data.

- Table 16. Sub-domain predictability — Binary prediction (pooled: original GT=yes + perturbed GT=no) (values in %). Chance level = 50%. Cells shaded above / below chance. Cross-model mean shown; best individual model per row in bold. Top 3 (most predictable) and bottom 3 (least predictable) sub-domains per area, ranked by cross-model mean. n: mean sample size across models.

Claude S4.5 DeepSeek R1

LLaMA 3.3

GPT-OSS GPT-4o

GPT-5.4

Sub-domain n Mean

Other

Astrophysics 23 51 50 60 36 47 50 63 Cosmology 14 45 41 50 38 41 50 50

Cosmology 14 45 41 50 38 41 50 50 Astrophysics 23 51 50 60 36 47 50 63

###### Environmental Science

Atmospheric Science 15 59 53 65 63 58 53 63 Ecology 46 56 46 56 57 64 63 47 Oceanography 20 55 56 54 52 56 56 56

Public Health 10 39 38 33 31 46 38 46 Marine Biology 18 46 35 57 52 43 52 39 Glaciology 11 47 40 40 40 60 60 40

Continued on next page

###### Neuroscience

Neurodegenerative Diseases 13 61 60 62 67 60 73 47 Cognitive Neuroscience 29 60 70 57 70 59 57 49 Psychiatry 13 59 54 77 54 54 69 46

Molecular Neuroscience 24 43 47 31 44 39 42 53 Cryo-Electron Microscopy 16 43 47 39 37 53 37 47 Neurophysiology 20 43 48 40 40 40 52 40

###### Biology

Stem Cell Biology 18 63 67 71 76 62 57 43 Computational Biology 44 60 65 56 65 64 58 55 Deep Learning 19 60 63 50 52 67 63 63

Single-Cell RNA Sequencing 10 32 27 30 36 36 36 27 Gene Expression 17 36 37 32 32 37 42 37 Embryology 12 37 37 32 21 42 37 56

###### Physics

Magnetism 17 56 56 59 59 62 53 47 Particle Physics 17 53 64 54 59 39 50 54 Quantum Simulation 10 53 71 64 62 29 29 64

Quantum Hall Effect 10 29 27 20 27 27 20 53 Optical Engineering 11 34 24 38 33 43 29 38 Quantum Optics 13 34 32 42 26 26 32 47

###### Medicine

Alzheimer’s Disease 15 72 85 85 69 69 62 62 Epigenetics 17 67 62 76 67 76 76 48 Proteomics 11 67 90 70 70 50 50 70

T Cell Biology 16 35 24 35 24 41 59 25 Drug Discovery 10 39 25 50 50 25 25 58 Single-Cell RNA Sequencing 14 39 57 42 14 43 43 36

###### Artificial Intelligence

AI Benchmarks 59 64 65 68 58 72 65 58 Reasoning 47 63 64 66 57 71 63 56 Large Multimodal Models 11 62 50 64 77 64 79 36

Policy Gradient Methods 10 35 30 50 20 40 50 22 Language Modeling 10 37 33 44 33 44 44 22 Neural Network Architectures 10 39 50 33 33 42 42 33

Continued on next page

###### Materials Science

Perovskite Materials 21 60 64 59 56 72 54 54 Optoelectronics 30 55 60 55 51 62 51 49 2D Materials 22 53 60 54 50 51 54 49

Flexible Electronics 11 35 29 35 25 24 41 53 Electrochemistry 24 41 49 39 34 39 39 46 Polymer Science 27 44 45 44 39 37 47 50

###### Chemistry

Organometallic Chemistry 25 53 57 54 51 57 49 49 Computational Chemistry 14 48 60 35 50 45 40 55 Photochemistry 14 47 53 56 58 37 37 42

Biocatalysis 11 36 27 36 36 45 27 45 Chemical Engineering 13 39 45 27 29 36 36 59 Medicinal Chemistry 12 40 44 39 33 33 39 50

- Table 17. Sub-domain predictability — Date prediction score (colour centre = cross-model mean

- 0.29) (values: score 0–1). Chance level = n/a. Cells shaded above / below chance. Cross-model mean shown; best individual model per row in bold. Top 3 (most predictable) and bottom 3 (least predictable) sub-domains per area, ranked by cross-model mean. n: mean sample size across models.

Claude S4.5 DeepSeek R1

LLaMA 3.3

GPT-OSS GPT-4o

GPT-5.4

Sub-domain n Mean

Other

Astrophysics 23 0.27 0.22 0.27 0.16 0.30 0.14 0.52 Cosmology 14 0.20 0.13 0.25 0.18 0.17 0.12 0.35

Cosmology 14 0.20 0.13 0.25 0.18 0.17 0.12 0.35 Astrophysics 23 0.27 0.22 0.27 0.16 0.30 0.14 0.52

###### Environmental Science

Public Health 10 0.39 0.22 0.51 0.32 0.60 0.15 0.54 Atmospheric Science 15 0.34 0.20 0.23 0.26 0.55 0.13 0.70 Hydrology 19 0.30 0.26 0.12 0.29 0.38 0.12 0.60

Continued on next page

Oceanography 20 0.19 0.22 0.17 0.19 0.09 0.03 0.44 Biodiversity 26 0.20 0.27 0.06 0.10 0.29 0.15 0.34 Conservation Biology 28 0.21 0.36 0.14 0.11 0.21 0.12 0.28

###### Neuroscience

Cryo-Electron Microscopy 16 0.42 0.25 0.28 0.52 0.53 0.25 0.68 Alzheimer’s Disease 16 0.38 0.38 0.49 0.48 0.30 0.17 0.48 Developmental Biology 10 0.36 0.07 0.15 0.67 0.49 0.17 0.62

Immunology 12 0.18 0.10 0.03 0.12 0.22 0.01 0.61 Neural Networks 11 0.19 0.15 0.09 0.29 0.12 0.04 0.43 Neuroimmunology 23 0.19 0.06 0.05 0.25 0.20 0.06 0.53

###### Biology

Archaeogenetics 11 0.41 0.38 0.29 0.46 0.41 0.49 0.45 Pharmacology 10 0.40 0.15 0.38 0.42 0.29 0.44 0.69 Crop Science 11 0.39 0.28 0.13 0.52 0.49 0.19 0.72

Genetic Engineering 12 0.11 0.07 0.03 0.08 0.11 0.02 0.37 Marine Biology 21 0.15 0.18 0.09 0.08 0.05 0.09 0.43 Entomology 14 0.18 0.05 0.00 0.25 0.19 0.07 0.51

###### Physics

Particle Physics 17 0.29 0.16 0.14 0.35 0.30 0.15 0.63 Quantum Information 25 0.26 0.20 0.20 0.24 0.30 0.11 0.53 Materials Science 30 0.26 0.20 0.19 0.32 0.10 0.15 0.58

Topological Insulators 20 0.15 0.10 0.09 0.23 0.12 0.09 0.28 Optical Engineering 11 0.16 0.03 0.08 0.14 0.14 0.09 0.47 Atomic Physics 13 0.16 0.06 0.05 0.16 0.14 0.01 0.53

###### Medicine

Fibrosis 11 0.43 0.56 0.26 0.54 0.48 0.09 0.65 Microbiome 13 0.40 0.26 0.31 0.57 0.38 0.20 0.68 Structural Biology 20 0.38 0.26 0.48 0.36 0.45 0.23 0.52

Translational Medicine 10 0.14 0.14 0.21 0.09 0.12 0.04 0.25 HIV Research 13 0.16 0.06 0.13 0.07 0.12 0.05 0.51 Tissue Engineering 14 0.17 0.16 0.13 0.20 0.13 0.02 0.38

###### Artificial Intelligence

Policy Gradient Methods 10 0.63 0.61 0.71 0.49 0.91 0.51 0.52 Agent-Based Systems 16 0.60 0.65 0.73 0.33 0.59 0.64 0.68 Cognitive Computing 12 0.59 0.59 0.60 0.62 0.60 0.57 0.56

Continued on next page

Neural Network Architectures 10 0.29 0.29 0.20 0.19 0.37 0.17 0.54 Knowledge Representation 14 0.31 0.19 0.09 0.35 0.22 0.38 0.63 Zero-shot Learning 11 0.31 0.45 0.20 0.34 0.13 0.32 0.44

###### Materials Science

Optoelectronics 30 0.27 0.16 0.16 0.24 0.41 0.14 0.54 Perovskite Solar Cells 50 0.27 0.20 0.15 0.27 0.36 0.18 0.46 Perovskite Materials 21 0.27 0.13 0.15 0.27 0.41 0.14 0.52

Mechanical Properties 15 0.10 0.13 0.04 0.04 0.03 0.01 0.33 Flexible Electronics 11 0.11 0.15 0.02 0.11 0.08 0.06 0.26 Energy Storage 25 0.13 0.04 0.16 0.07 0.12 0.05 0.33

###### Chemistry

Medicinal Chemistry 12 0.29 0.12 0.28 0.17 0.29 0.26 0.63 Computational Chemistry 14 0.22 0.19 0.09 0.17 0.32 0.13 0.43 Organometallic Chemistry 25 0.22 0.22 0.11 0.16 0.30 0.10 0.41

Chemical Engineering 13 0.06 0.02 0.03 0.04 0.00 0.01 0.28 Electrochemistry 17 0.13 0.04 0.12 0.10 0.11 0.02 0.38 Radical Chemistry 14 0.15 0.07 0.09 0.10 0.16 0.03 0.43

- Table 18. Sub-domain predictability — MCQ, 4-choice (values in %). Chance level = 25%. Cells shaded above / below chance. Cross-model mean shown; best individual model per row in bold. Top 3 (most predictable) and bottom 3 (least predictable) sub-domains per area, ranked by cross-model mean. n: mean sample size across models.

Claude S4.5 DeepSeek R1

LLaMA 3.3

GPT-OSS GPT-4o

GPT-5.4

Sub-domain n Mean

Other

Cosmology 14 80 92 85 73 77 85 67 Astrophysics 23 78 90 80 94 65 80 60

Astrophysics 23 78 90 80 94 65 80 60 Cosmology 14 80 92 85 73 77 85 67

Continued on next page

###### Environmental Science

Public Health 10 94 100 100 100 83 83 100 Glaciology 11 76 100 80 89 70 60 60 Atmospheric Science 15 74 100 88 78 56 78 44

Hydrology 19 61 88 94 53 53 41 35 Ecology 46 63 88 81 43 52 58 55 Biodiversity 26 64 88 85 50 50 56 56

###### Neuroscience

Cryo-Electron Microscopy 16 88 88 87 93 88 94 81 Calcium Imaging 15 83 83 92 83 83 75 83 Synaptic Plasticity 22 82 94 81 87 88 75 69

Neuroimmunology 23 46 67 47 64 22 44 33 Genetics 20 53 68 71 53 42 47 37 Neurodegenerative Diseases 13 54 58 64 58 50 50 42

###### Biology

Archaeogenetics 11 98 100 100 88 100 100 100 Single-Cell RNA Sequencing 10 90 100 100 89 70 90 90 Cryo-Electron Microscopy 37 88 97 97 94 78 89 74

Genetic Engineering 12 46 58 60 60 42 42 17 Physiology 12 47 78 60 33 44 33 33 Stem Cell Biology 18 48 75 58 40 42 42 33

###### Physics

Experimental Physics 15 77 100 83 80 58 67 75 Nonlinear Optics 11 72 91 82 80 73 91 18 Astrophysics 32 71 83 83 77 63 67 53

Statistical Mechanics 12 47 92 58 40 42 25 25 Graphene 12 49 70 50 43 50 50 30 Magnetism 17 50 62 56 67 38 50 25

###### Medicine

Single-Cell Sequencing 10 86 100 90 78 70 90 90 Alzheimer’s Disease 15 81 100 100 83 71 71 57 Genomics 47 77 91 95 69 73 66 70

Gene Therapy 17 34 53 44 21 24 24 41 Cancer Therapy 14 38 58 25 25 50 33 36 Inflammation 18 42 62 50 42 38 31 31

Continued on next page

###### Artificial Intelligence

Tool Use in AI 10 77 83 100 80 67 83 50 Inference Optimization 11 75 100 100 67 57 71 57 Monte Carlo Tree Search 12 75 90 90 88 60 70 50

Audio Processing 10 24 44 22 22 22 22 11 Speech Synthesis 10 35 60 50 30 30 30 10 Autoregressive Models 21 35 78 67 21 6 28 11

###### Materials Science

Energy Storage 25 62 91 96 67 30 57 30 Mechanical Properties 15 61 73 73 62 53 53 53 Polymer Science 27 59 85 76 64 54 46 31

Polymer Chemistry 10 39 40 50 56 20 20 50 Metallurgy 13 48 75 42 44 42 42 42 Perovskite Materials 21 49 90 71 62 14 38 19

###### Chemistry

Chemical Engineering 13 72 92 69 78 54 77 62 Biocatalysis 11 65 91 64 45 64 82 45 Photochemistry 14 60 92 83 54 46 46 38

Medicinal Chemistry 12 29 64 18 11 36 18 27 Synthetic Chemistry 46 40 80 51 36 22 25 25 Computational Chemistry 14 41 91 67 18 27 36 9

DeepSeek R1 Anticipates RL + MoE Breakthrough (GLM-4.5)

Model: DeepSeek R1 (cutoff: 2024-07; RL post-training) Target Milestone: GLM-4.5 (Aug 2025; RL + MoE architecture)

Task: Predict whether a method will achieve:

- • TAU-Bench ≥ 70%, AIME ≥ 91%, SWE-bench ≥ 64%
- • Under parameter constraint (<355B)
- • By Aug 2025

Ground Truth: Yes Binary Prediction

Prediction: Yes Confidence: 0.70 Insight: Correctly anticipates that efficiency techniques (e.g., MoE) enable strong multi-task performance under parameter constraints.

Perturbed Binary Prediction Prediction: No Confidence: 0.35

Insight: Recognizes the joint difficulty of simultaneously meeting all thresholds, reflecting calibrated skepticism under stricter conditions.

MCQ (Mechanism Prediction) Prediction: B Confidence: 0.85 Ground Truth: B Insight: Selects a hybrid MoE-style activation strategy, aligning with the eventual system design.

Date Prediction Prediction: 2026-07 Confidence: 0.70 Ground Truth: 2025-08 Insight: Overestimates timeline despite correct feasibility and mechanism prediction.

Key Observation: DeepSeek R1 (trained via RL post-training) correctly anticipates a future breakthrough (GLM-4.5) that also relies on RL-based optimization, despite having no access to this post-cutoff system. This suggests the model extrapolates along training-driven progress trends, identifying RL as a key driver of future capability improvements.

#### D.4 Model Bias and Confidence

Predicted vs. ground-truth publication date — each model's forecast distribution (diagonal = perfect; dotted horizontal = model training cutoff)

###### GPT-4o [cutoff 2023-10]

###### GPT-OSS 20B [cutoff 2024-06]

###### DeepSeek R1 [cutoff 2024-07]

- 2022·01

- 2023·01

- 2024·01

- 2025·01

- 2026·01

- 2027·01

- 2028·01

- 2029·01

- 2030·01

- 2031·01

- 2022·01

- 2023·01

- 2024·01

- 2025·01

- 2026·01

- 2027·01

- 2028·01

- 2029·01

- 2030·01

- 2031·01

- 2022·01

- 2023·01

- 2024·01

- 2025·01

- 2026·01

- 2027·01

- 2028·01

- 2029·01

- 2030·01

- 2031·01

326 clipped (13%)

213 clipped (8%)

88 clipped (4%)

[Figure 170]

[Figure 171]

[Figure 172]

70

80

40

70

60

35

Predicteddate

60

50

30

50

25

Count

Count

Count

40

40

20

30

30

15

20

20

cutoff

10

cutoff

10

10

cutoff

5

median Δ = +26 mo ≤12 mo: 25%

median Δ = +15 mo ≤12 mo: 43%

median Δ = +13 mo ≤12 mo: 48%

2024·01 2024·07 2025·01 2025·07 2026·01

2024·01 2024·07 2025·01 2025·07 2026·01

2024·01 2024·07 2025·01 2025·07 2026·01

###### LLaMA 3.3 [cutoff 2023-12]

###### Claude S4.5 [cutoff 2025-01]

GPT-5.4 [cutoff 2025-08]

- 2022·01

- 2023·01

- 2024·01

- 2025·01

- 2026·01

- 2027·01

- 2028·01

- 2029·01

- 2030·01

- 2031·01

- 2022·01

- 2023·01

- 2024·01

- 2025·01

- 2026·01

- 2027·01

- 2028·01

- 2029·01

- 2030·01

- 2031·01

- 2022·01

- 2023·01

- 2024·01

- 2025·01

- 2026·01

- 2027·01

- 2028·01

- 2029·01

- 2030·01

- 2031·01

46 clipped (2%)

201 clipped (8%)

470 clipped (17%)

[Figure 173]

[Figure 174]

[Figure 175]

175

40

40

150

35

35

Predicteddate

30

30

125

25

25

Count

Count

Count

100

20

20

75

cutoff

15

15

cutoff

50

10

10

cutoff

25

5

5

median Δ = +4 mo ≤12 mo: 75%

median Δ = +17 mo ≤12 mo: 31%

median Δ = +14 mo ≤12 mo: 33%

2024·01 2024·07 2025·01 2025·07 2026·01

2024·01 2024·07 2025·01 2025·07 2026·01

2024·01 2024·07 2025·01 2025·07 2026·01

Ground-truth date

Ground-truth date

Ground-truth date

- Figure 9. Visualization of aggregated date predictions across models. Importantly, many models seem to have anchors, most prevalent in LLaMA 3.3, with dense predictions around mid-2025. Anchors like this can cause date predictions after a cutoff to seem more accurate. The clusters of dates also demonstrate how models may be temporally biased.

###### Confidence calibration — CUSP Benchmark

###### GPT-5.4

###### Claude S4.5

###### LLaMA 3.3

Binary ECE = 0.203

Binary ECE = 0.200

Binary ECE = 0.309

| |
|---|

| |
|---|

MCQ ECE = 0.033

MCQ ECE = 0.022

MCQ ECE = 0.390

| |
|---|

Fractioncorrect

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

0.25 0.50 0.75 1.00

0.25 0.50 0.75 1.00

0.25 0.50 0.75 1.00

| |
|---|

###### DeepSeek R1

###### GPT-OSS 20B

GPT-4o

Binary ECE = 0.167

Binary ECE = 0.211

Binary ECE = 0.213

MCQ ECE = 0.315

MCQ ECE = 0.369

MCQ ECE = 0.346

Fractioncorrect

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

0.25 0.50 0.75 1.00

0.25 0.50 0.75 1.00

0.25 0.50 0.75 1.00

Stated confidence

Stated confidence

Stated confidence

Binary MCQ Perfect calibration (diagonal)

Figure 10. Visualization of confidence calibration across six LLMs.

Model Cutoff Binary acc. Perturbed acc. Bias index Merged↑ Response tendency

GPT-OSS Jun 2024 0.176 0.776 -0.6 0.518 Strong “No bias GPT-4o Oct 2023 0.190 0.766 -0.6 0.519 “No bias Claude S4.5 Jan 2025 0.266 0.700 -0.4 0.513 “No bias

[Figure 176]

[Figure 177]

[Figure 178]

DeepSeek R1 Jul 2024 0.467 0.492 -0.0 0.481 Balanced GPT-5.4 Aug 2025 0.635 0.397 +0.2 0.499 “Yes bias LLaMA 3.3 Dec 2023 0.932 0.092 +0.8 0.453 Strong “Yes bias

[Figure 179]

[Figure 180]

[Figure 181]

Chance (random) 0.500 0.500 0.000 0.500 —

- Table 19. Binary response-bias analysis. Bias index=binary acc.−perturbed acc.; +1=always “Yes”, −1=always “No”, 0=unbiased. Merged: bias-corrected forecasting accuracy (chance=0.50). Models sorted from most No-biased to most Yes-biased. Bold: highest merged accuracy.

Model Binary MCQ Date Conf. Acc. Over-conf. ECE Conf. Acc. Over-conf. ECE Conf. Score Over-conf. ECE

GPT-5.4 0.704 0.499 +0.2 0.204 0.849 0.819 +0.0 0.031 0.596 0.241 +0.4 0.355 Claude S4.5 0.709 0.513 +0.2 0.216 0.718 0.724 -0.0 0.014 0.527 0.239 +0.3 0.281 LLaMA 3.3 0.761 0.453 +0.3 0.309 0.824 0.434 +0.4 0.391 0.741 0.500 +0.2 0.242

[Figure 182]

[Figure 183]

[Figure 184]

DeepSeek R1 0.558 0.481 +0.1 0.247 0.912 0.594 +0.3 0.316 0.683 0.288 +0.4 0.343 GPT-OSS 0.712 0.518 +0.2 0.232 0.840 0.471 +0.4 0.369 0.571 0.300 +0.3 0.268 GPT-4o 0.731 0.519 +0.2 0.213 0.876 0.530 +0.3 0.346 0.774 0.178 +0.6 0.584

[Figure 185]

[Figure 186]

[Figure 187]

- Table 20. Confidence calibration across tasks. Conf.: mean self-reported confidence (0–

- 1). Acc./Score: mean accuracy or date score. Over-conf.: confidence−accuracy; positive=overconfident, 0=perfectly calibrated. ECE: Expected Calibration Error (10 bins); lower is better. All models are severely overconfident on MCQ and date tasks. Cell colours for ECE: <0.05

0.05–0.10 0.10–0.15 0.15–0.25 >0.25 . Same scale for overconfidence magnitude.

ECE computed over 10 equal-width bins. Overconfidence=mean confidence−mean accuracy/score.

#### D.5 Additional Binary Results

- Table 21. Binary task extended classification metrics. Computed from a 2×2 confusion matrix where original binary questions have ground-truth Yes (contributing TP/FN) and negation-flipped perturbed questions have ground-truth No (contributing TN/FP). Bal. Acc.: balanced accuracy

= 12(TPR+TNR); chance=0.50. Prec.: TP/(TP+FP). Recall: TPR=TP/(TP+FN) (accuracy on original questions). Spec.: TNR=TN/(TN+FP) (accuracy on perturbed questions). F1-Yes/F1-No:

per-class F1. Macro F1: unweighted mean of F1-Yes and F1-No; chance≈0.50. MCC: Matthews Correlation Coefficient; 0=random, 1=perfect. Models sorted by Macro F1 (highest first). Bold: best per column.

Accuracy decomp. Per-class F1

Model Cutoff n Bal. Acc.↑ Prec. Recall Spec. F1-Yes F1-No Macro F1↑ MCC↑ GPT-5.4 Aug 2025 6411 0.516 0.442 0.635 0.397 0.521 0.475 0.498 0.033 DeepSeek R1 Jul 2024 6347 0.480 0.410 0.467 0.492 0.437 0.519 0.478 -0.040 Claude S4.5 Jan 2025 6074 0.483 0.402 0.266 0.700 0.320 0.621 0.470 -0.037 GPT-4o Oct 2023 6408 0.478 0.380 0.190 0.766 0.253 0.645 0.449 -0.053 GPT-OSS 20B Jun 2024 6410 0.476 0.372 0.176 0.776 0.239 0.648 0.443 -0.059 LLaMA 3.3 Dec 2023 6359 0.512 0.436 0.932 0.092 0.594 0.161 0.378 0.043 Chance (random) 0.500 0.500 0.500 0.500 0.500 0.500 0.500 0.000

#### D.6 External Benchmark Saturation Comparison

To contextualize the saturation level of CUSP, we compare frontier-model performance against several widely used scientific and reasoning benchmarks, including MMLU-Pro, GPQA Diamond, and MedQA. Benchmark saturation comparisons against MMLU-Pro, GPQA Diamond, and MedQA were constructed using publicly reported frontier-model evaluation results from Artificial Analysis Evaluations and the official MMLU-Pro Hugging Face leaderboard.

[Figure 188]

[Figure 189]

ClaudeSonnet4.5 DeepSeek-R1 Llama3.370B

[Figure 190]

-4o

[Figure 191]

[Figure 192]

[Figure 193]

-5.4

GPT

GPT

Figure 11. Saturation plot of CUSP compared to other commonly used LLM benchmarks.

### E Human Evaluation Details

#### E.1 Human evaluation of Dataset Validation

We recruited 10 human evaluators, primarily graduate-level researchers, with affiliations including the University of Oxford, Yale University, the University of Michigan, the University of Chicago, and CUHK-Shenzhen, and with expertise spanning artificial intelligence, materials science, and chemistry.

We hosted a public site and gave human evaluators the paper abstracts, alongside the questions in our benchmark. Then, under the exact same criteria as the LLM judge (see F.2), they were given the binary choice to keep or remove the questions from the benchmark. We found that on average the LLM judges were actually more rigorous in removing examples that should have been removed

under the criteria, while keeping questions with clean, verifiable results. See empirical examples in

- F.3.

#### E.2 Human Evaluation of LLM Judge

We set up a human evaluation on a subset of CUSP using 60 examples, conducted by three evaluators (two Computer Science PhDs and one postdoctoral scholar). All evaluators were provided with a web interface (Figure 12). The human evaluators evaluate examples across GPT-4o and GPT-OSS. The human evaluators were given the exact same grading rubric as the LLM judge for fairness.

Across all 60 annotated pairs, the AI judge achieves a Pearson correlation of r = 0.34 (p < 0.01) and a Spearman rank correlation of ρ = 0.33 with human FRQ scores, with a mean absolute error (MAE) of 0.75 points on a 0–10 scale (Figure 13). The Bland–Altman analysis reveals a small positive bias of +0.26 points, indicating that the AI judge is, on average, marginally more generous than human evaluators, with 95% limits of agreement spanning [−1.65, +2.17] points. These figures indicate statistically significant and practically meaningful agreement, where the AI judge captures the broad ordinal structure of human quality judgments while operating fully automatically, though non-trivial item-level variance remains.

[Figure 194]

- Figure 12. Our human evaluation interface is designed to assess alignment, novelty, feasibility, and specificity, and to compare these judgments against the consistency of the LLM judge (with web search). Both human evaluators and the LLM judge have access to the same rubric.

Human Evaluator vs AI Judge Combined Analysis (n = 60 annotations)

###### Human vs AI Judge FRQ Score (all 3 evaluators, n=60)

###### Bland-Altman Plot (FRQ) AI Score Relative to Human

###### Mean FRQ Score by Evaluator

10

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- Human 1

| |
|---|

- Human 2

- Human 3

- 0

- 1

- 2

- 3

Human

Pearson r = 0.34 (p=0.000) Spearman = 0.33 MAE = 0.75

AI Judge

Bias = +0.26

8

+1.96 SD = +2.17 1.96 SD = -1.65

MeanFRQScore(±SEM)

3.90

AIHumanScore

3.66

AIFRQScore

3.48

3.44

6

3.36

| |
|---|

| |
|---|

2.86

| |
|---|

| |
|---|

| | | |
|---|---|---|
| | | |

| |
|---|

| |
|---|

4

| |
|---|

| | |
|---|---|
| | |

| |
|---|

| | |
|---|---|
| | |
| | |

- Human 1

| |
|---|

- Human 2

- Human 3

2

- 0 2 4 6 8 10 Mean of Human & AI Score

- 1

| | |
|---|---|
| | |

OLS fit

Perfect agreement

0

0 2 4 6 8 10

Human 1 Human 2 Human 3

Human FRQ Score

###### Alignment Score by Evaluator

###### Specificity Score by Evaluator

###### Novelty Score by Evaluator

- 0

- 1

- 2

- 3

- 4

- 5

Human

Human

Human

3.0

3.5

AI

AI

AI

2.5

3.0

MeanSpecificityScore

MeanAlignmentScore

MeanNoveltyScore

2.5

2.0

2.0

1.5

1.5

1.0

1.0

0.5

0.5

0.0

0.0

Human 1 Human 2 Human 3

Human 1 Human 2 Human 3

Human 1 Human 2 Human 3

- Figure 13. Results on human evaluation vs AI Judge on 60 AI questions. Each human evaluates 20 questions.

### F Benchmark Verification

#### F.1 Automatic Verification Results

- Table 22. CUSP Dataset Validation Pipeline Attrition. Distribution of candidate questions generated across post-2024 sources and the final subset passing the LLM-as-a-judge verification pipeline. Items failing to demonstrate strict scientific faithfulness or objectively measurable outcomes were rigorously removed. (Note: Initial generation of the Leaderboard subset did not contain FRQs).

MCQ FRQ Binary Perturbed Date Pred. Grand Total

Paper Source Pre Post Pre Post Pre Post Pre Post Pre Post Pre Post Top AI Papers 297 230 297 273 297 144 297 270 297 144 1,485 1,061 Hugging Face Daily 763 654 763 695 763 352 763 676 763 352 3,815 2,729 Nature 2,113 1,824 2,113 1,875 2,113 1,299 2,113 1,524 2,113 1,299 10,565 7,821 Science 1,039 931 1,039 875 1,039 671 1,039 783 1,039 671 5,195 3,931 Cell 495 430 495 417 495 230 495 344 495 230 2,475 1,651 Leaderboard (Curated) 59 59 – – 59 59 59 59 59 59 236 236 CUSP Yield 4,766 4,128 4,707 4,135 4,766 2,755 4,766 3,656 4,766 2,755 23,771 17,429

#### F.2 Verification Examples and Prompts

Rejected Abstract under CUSP Filtering

Functional gradients facilitate tactile sensing in elephant whiskers Science, Feb 2026 | Domain: Biology | doi:10.1126/science.adx8981 | Rejected

Abstract (excerpt):

Keratin composites enable animals to hike with hooves, fly with feathers, and sense with skin. Mammalian whiskers are elongated keratin rods attached to tactile skin structures that extend the animal’s sensory volume. We investigated the whiskers that cover Asian elephant (Elephas maximus) trunks and found that they are geometrically and mechanically tailored to facilitate tactile perception by encoding contact location in the amplitude and frequency of the vibrotactile signal felt at the whisker base. Elephant whiskers emerge from armored trunk skin and shift from a thick, circular, porous, stiff base to a thin, ovular, dense, soft tip. These functional gradients of geometry, porosity, and stiffness independently tune the neuromechanics of elephant trunk touch to facilitate highly dexterous manipulation while ensuring whisker durability. [58]

###### Reason for rejection gpt-4o-mini: The abstract discusses the structure and function of elephant whiskers but does not provide a concrete experimental result or measurable biological quantity.

[Figure 195]

###### Faithfulness Validator System Prompt

Prompt: You are a careful scientific benchmark validator. Your task is to judge whether a binary forecasting statement is faithful to the source abstract/result text. Use only the supplied text. Do not use outside knowledge.

###### What you are checking:

- • whether the binary statement preserves the same scientific claim as the abstract/result text

- • whether it changes a number, entity, benchmark, condition, threshold, scope, or outcome
- • whether it introduces a claim not supported by the source

###### Important rules:

- • Judge only the binary statement itself
- • Do not judge the date/time wording — the date is generated externally and should be ignored here
- • Pass if the statement is a faithful restatement of the source claim
- • Fail if the statement changes the meaning or invents unsupported details
- • Be strict about the claim, but do not penalise the externally inserted date

###### Scoring guide:

- • 5 — exact or nearly exact match to the source claim
- • 4 — minor wording differences, but still faithful
- • 3 — partially faithful / borderline
- • 2 — mostly unsupported or meaningfully altered
- • 1 — clearly unfaithful

###### Return only valid JSON with this schema:

{

"verdict": "pass | fail | unclear", "score": 1-5, "reason": "short explanation", "mismatch_types": ["numbers","entity","condition",

"outcome","time","scope", "threshold","none"]

}

Verifiability Validator System Prompt

###### Prompt:

You are a careful scientific benchmark validator. Your task is to judge whether the binary statement is concrete enough to be objectively verified. Use only the supplied text. Do not use outside knowledge. This is not a writing-quality check — a sentence can sound fine and still fail if the underlying claim is vague.

###### What you are checking:

- • whether the statement describes a concrete scientific claim
- • whether a third party could decide yes/no without guessing
- • whether the claim is vague, underspecified, or too interpretive

###### Important rules:

- • Judge the claim as a whole
- • Do not reject simply because the wording uses comparison language
- • Reject only if the comparison or claim is not operationalised enough to be checked
- • Pass if the claim is specific enough to be objectively verifiable from the source
- • Fail if it is vague — e.g. “better”, “good”, “strong”, “effective”, or “comparable” — without a clear measurable criterion

###### Examples of pass:

- • “achieves 72.4% accuracy on MMLU”
- • “improves F1 from 81.2 to 84.0 on the stated benchmark”
- • “reduces error rate by 20% under the specified evaluation protocol”

###### Examples of fail:

- • “achieves performance comparable to OpenAI-o1-1217 on reasoning tasks” when the source does not define how comparability is measured
- • “improves performance” with no metric, benchmark, or threshold
- • “better than previous methods” with no objective criterion

###### Scoring guide:

- • 5 — fully specific, objective, and easy to verify
- • 4 — mostly specific with minor ambiguity
- • 3 — borderline / partly testable
- • 2 — mostly vague or underspecified
- • 1 — not objectively verifiable

###### Return only valid JSON with this schema:

{

"verdict": "pass | fail | unclear", "score": 1-5, "reason": "short explanation"

}

Perturbation Validator System Prompt

Prompt: You are a careful scientific benchmark validator. Your task is to judge whether the perturbed binary question is a genuine perturbation. Use only the supplied text. Do not use outside knowledge.

###### What you are checking:

- • whether the perturbed question changes a salient detail from the original question
- • whether that change meaningfully breaks support from the source abstract/result text
- • whether the perturbed question is not merely a paraphrase or trivial rewording

###### Important rules:

- • Pass only if the perturbation changes a meaningful aspect such as number, threshold, entity, outcome, time, scope, or condition
- • Pass only if the perturbed version is no longer directly supported by the source
- • Fail if it is essentially the same question with cosmetic wording changes
- • Fail if it does not introduce a real challenge to the source claim

###### Scoring guide:

- • 5 — strong, clearly altered perturbation
- • 4 — valid perturbation with minor ambiguity
- • 3 — borderline / weak perturbation
- • 2 — likely not a real perturbation
- • 1 — clearly not perturbed

###### Return only valid JSON with this schema:

{

"verdict": "pass | fail | unclear", "score": 1-5, "reason": "short explanation",

"changed_elements": ["numbers","entity","condition", "outcome","time","scope", "threshold","none"]

}

MCQ Stem Validator

Prompt: You are a strict scientific benchmark validator. Your task is to evaluate the MCQ problem statement only. You must check two things:

- 1. Faithfulness: does the problem statement accurately reflect the source abstract?
- 2. Verifiability: is the problem statement concrete enough to be objectively answered from the source?

###### Important rules:

- • Judge only the MCQ stem/problem statement — do not judge the answer choices in this call
- • Ignore any forecasting wording. The stem often asks things like “by March 2026, which approach is most likely to achieve X?” — completely ignore the date, the likelihood-prediction framing, and speculative timelines; do not penalise the stem for asking about the future
- • Assess only whether the scientific core of the question accurately reflects the abstract’s methodology and results
- • Fail if the scientific core changes the claim, introduces unsupported details, or misstates the abstract
- • Fail if the scientific core is vague, underspecified, or not operationalisable
- • Pass only if the core scientific claim is both faithful and concrete

###### Examples of fail:

- • the stem adds a benchmark, metric, or threshold not in the abstract
- • the stem uses vague language such as “improve performance” with no measurable criterion
- • the stem asks about a claim that cannot be verified from the abstract alone

###### Scoring guide:

- • 5 — fully faithful and clearly verifiable
- • 4 — mostly strong with minor ambiguity
- • 3 — borderline
- • 2 — weak
- • 1 — clearly invalid

###### Return only valid JSON with this schema:

{

"verdict": "pass | fail | unclear", "score": 1-5, "reason": "short explanation", "issue_types": ["faithfulness","verifiability","none"]

}

MCQ Answer Validator

###### Prompt:

You are a strict scientific benchmark validator. Your task is to evaluate the marked correct answer choice only. You must check whether the selected answer is supported by the source abstract as the correct technical approach, mechanism, or result.

###### Important rules:

- • Judge only the marked correct answer choice — do not judge distractors in this call

- • Pass only if the answer choice is supported or clearly implied by the abstract
- • Fail if the answer choice is unsupported, mismatched, or invents a mechanism not present in the abstract
- • If the abstract gives enough evidence for more than one answer, explain the ambiguity

###### Scoring guide:

- • 5 — correct and directly supported
- • 4 — supported with minor interpretive gap
- • 3 — borderline
- • 2 — likely incorrect
- • 1 — clearly wrong

###### Return only valid JSON with this schema:

{

"verdict": "pass | fail | unclear", "score": 1-5, "reason": "short explanation", "issue_types": ["unsupported_answer","ambiguous_answer","none"]

}

MCQ Distractor Validator

Prompt: You are a strict scientific benchmark validator. Your task is to evaluate the incorrect answer choices only. You must check whether the distractors are:

- 1. plausible enough to require real reasoning,
- 2. not directly supported by the abstract,
- 3. not trivially wrong or obviously eliminated. Important rules:

- • Judge the distractors as a set — do not judge the stem or the correct answer in this call
- • Pass only if the distractors are non-trivial and sufficiently plausible
- • Fail if the distractors are too easy, too obviously wrong, or directly supported by the abstract
- • Fail if the distractors are not meaningfully competitive with the correct answer

###### Scoring guide:

- • 5 — strong distractors, highly plausible
- • 4 — good distractors with minor issues
- • 3 — borderline
- • 2 — weak distractors
- • 1 — trivial or obviously bad distractors

###### Return only valid JSON with this schema:

{

"verdict": "pass | fail | unclear", "score": 1-5, "reason": "short explanation", "issue_types": ["too_easy","unsupported_by_abstract",

"not_plausible","none"] }

FRQ Validator System Prompt

###### Prompt:

You are a strict scientific benchmark validator. Your task is to evaluate the Free Response Question (FRQ) prompt. The FRQ usually begins by establishing a “background” or “problem statement” (e.g., “Given the unreliability of proxies for reasoning quality...”). Then it asks the user to propose a solution (e.g., “propose a concrete method... by [Date]”). Your primary goal is to evaluate whether the background premise / problem statement established in the FRQ is accurate and faithful to the source abstract.

###### Important rules:

- • Extract the premise/background statement embedded in the FRQ
- • Compare this premise directly against the source abstract
- • Does the abstract actually describe this specific problem, challenge, or background context?
- • Fail if the FRQ invents a problem, misrepresents the challenge, or contradicts the abstract’s framing
- • Pass if the problem statement/background is faithful and accurate to the abstract
- • Ignore any forecasting wording — do not penalise the FRQ for asking for a prediction or solution “by March 2026” or any other date; that is expected and required
- • Do not evaluate the FRQ for “measurability” or whether it includes specific benchmarks (unlike binary or MCQs) — this is a free-response question, so open-ended phrasing asking for a “concrete method” or “implementation plan” is exactly what we want

###### Scoring guide:

- • 5 — problem statement is perfectly faithful to the abstract
- • 4 — mostly faithful, minor semantic differences
- • 3 — borderline
- • 2 — weak connection to the abstract’s actual problem
- • 1 — clearly unfaithful, invents a problem not in the abstract

###### Return only valid JSON with this schema:

{

"verdict": "pass | fail | unclear", "score": 1-5, "reason": "short explanation", "issue_types": ["unfaithful_problem_statement","none"]

}

#### F.3 Human Vs AI in Benchmark Validation Item 1 • Binary

(Binary) By December 2024, will a method reduce token costs in Chain-of-Thought reasoning by 30% while maintaining a performance reduction of less than 2% across evaluated tasks?

× AI Removed · Human Kept

The binary statement introduces specific numerical claims (30% token cost reduction and less than 2% performance reduction) that are not supported by the source text, which only mentions a general reduction in token costs with a slight performance reduction.

###### Item 2 • Binary

(Binary) By June 2024, will a method achieve state-of-the-art performance on multimodal benchmarks?

###### × AI Removed · Human Kept The claim of achieving ‘state-of-the-art performance on multimodal benchmarks’ is vague and not objectively verifiable without specific metrics, benchmarks, or thresholds defined in the source text.

Item 3 • MCQ

###### (MCQ)

Given the challenge of reducing token usage in reasoning tasks without significantly compromising accuracy, which of the following approaches is most likely to achieve a 30% reduction in token costs by December 2024?

× AI Removed · Human Kept The stem introduces a specific threshold of ‘30% reduction in token costs’ which is not mentioned or supported in the abstract. This addition makes the question unfaithful to the source material.

Item 4 • Binary

(Binary) By August 2024, will a method achieve more than 55.7% accuracy on the Multimodal Multi-image Understanding (MMIU) benchmark?

###### ✓ Human Removed · AI Kept The claim is specific, referencing a concrete benchmark (MMIU) with a clear, measurable threshold of 55.7% accuracy, making it objectively verifiable.

Item 5 • MCQ

###### (MCQ)

Given the challenge of aligning representation spaces and reconciling sequence length differences between speech and text, which approach is most likely to achieve superior performance on speech translation and AIR-Bench tasks using only 2% of the training data by February 2025?

✓ Human Removed · AI Kept

The core scientific claim in the stem about aligning representation spaces and reconciling sequence length differences between speech and text accurately reflects the fundamental problems discussed in the abstract. The focus on performance in speech translation and AIR-Bench tasks with reduced training data is faithful to the abstract’s claims about Soundwave’s efficiency and results. The question is concrete enough to be objectively answered based on the abstract’s content.

### G CUSP Evaluation Details

Algorithm 1 CUSP Two-Track Evaluation Require: Row r with predictions p, judge J Ensure: Per-task outcome scores {ot}

- Track 1 — Deterministic outcome scoring

- 1: for t ∈ {binary, binary⊥, mcq, date} present in r do
- 2: ot ← DeterministicGrade(pt, r) ▷ exact match; date uses e−0.1|∆mo|
- 3: end for
- 4: if frq present in r then
- 5: C ← J .FRQRubric(pfrq, r) ▷ rubric judge over alignment, specificity, novelty, feasibility
- 6: ofrq ← C.frq_score/ 10
- 7: end if

Track 2 — Leakage gating (FRQ only)

- 8: if frq present in r then
- 9: ℓ ← J .LeakageJudge(pfrq, r) ▷ web-search call; checks for verbatim post-cutoff entities
- 10: if ℓ.verdict = fail then
- 11: ofrq ← nil ▷ contaminated responses receive no forecasting credit
- 12: end if
- 13: end if
- 14: return {ot}

#### G.1 Evaluation System Prompts

###### System Prompt: Rigorous Scientific Evaluation Judge

You are a rigorous scientific evaluation judge. Your job is to assess the LLM RESPONSE (delimited by «LLM_RESPONSE» tags in the user message) against the GROUND-TRUTH REFERENCE (everything outside those tags). Never confuse what the web search returns with what the LLM wrote — the LLM RESPONSE is ONLY the text inside «LLM_RESPONSE». USE WEB SEARCH to look up the actual paper, verify the real methodology, and check whether claims inside «LLM_RESPONSE» are accurate. Use search results as ground truth — not to confirm the LLM.

=== PART 1: FRQ SCORING — use strict anchors ===

- 1. alignment (0–10): Does the LLM RESPONSE describe the specific approach used in the paper? Use web search to find the actual paper method.

0–2: completely wrong direction or no meaningful content 3–4: roughly right area but missing key specifics of the actual method 5–6: captures the main idea but lacks important details or misstates them 7–8: matches the core technique with most key details correct

9–10: precise match including specific design choices and implementation

- 2. specificity (0–10): Is the LLM RESPONSE technically concrete? 0–2: pure buzzwords or single-sentence vague claims 3–4: names a technique but no explanation of how it is applied 5–6: explains the method at a conceptual level 7–8: provides implementation-level details (architecture, loss, data)

9–10: full technical recipe that could be directly implemented

- 3. novelty (0–10): Does the LLM RESPONSE show non-obvious insight? 0–2: restates the most obvious baseline for this problem area 3–4: proposes minor, obvious variations on standard baselines 5–6: goes beyond obvious but the insight is well-known in the field 7–8: proposes something non-trivial and technically justified

9–10: highly original and technically justified breakthrough insight

- 4. feasibility (0–10): Is the proposed approach actually feasible given real-world constraints? 0–2: clearly infeasible (non-existent methods, impossible assumptions) 3–4: major feasibility issues (unrealistic compute/data or unsupported claims) 5–6: plausible but with notable practical concerns 7–8: largely feasible with minor caveats

9–10: clearly feasible and consistent with real-world implementations

Respond with a “‘json code block. Write 1-2 sentences of reasoning with inline citations ([title](url)) before the block. {

"alignment": <0-10>, "specificity": <0-10>, "novelty": <0-10>, "feasibility": <0-10>, "rationale": "<2-3 sentence summary explaining the scores for alignment, specificity,

novelty, and feasibility>" }

System Prompt: Data-Contamination Auditor

You are a data-contamination and leakage auditor for a scientific forecasting benchmark. The benchmark tests whether models can predict future scientific results. Each question is about a discovery published AFTER the model’s knowledge cutoff. The paper’s publication date is {pub_date}.

====== YOUR ONLY JOB ====== Decide whether the LLM RESPONSE (delimited by «LLM_RESPONSE» tags in the user message) explicitly names a specific post-cutoff entity.

FOLLOW THESE STEPS IN ORDER — DO NOT SKIP:

- 1. STEP 1 — READ THE LLM RESPONSE TEXT ONLY. Copy out every proper noun, model name, paper title, system name, or dataset name that appears VERBATIM between the «LLM_RESPONSE» tags. If you find zero such names, immediately return verdict=’pass’ — DO NOT perform any web search.
- 2. STEP 2 — FOR EACH NAME FOUND IN STEP 1 ONLY: Use web search to verify when that exact name was first publicly released. If it was released AFTER {cutoff}, that is a leakage indicator.
- 3. STEP 3 — DECIDE: LEAKAGE (verdict=fail) requires ALL THREE:

a. The name appears VERBATIM in the LLM RESPONSE text (quote it). b. It was first released AFTER {cutoff} (confirmed by web search). c. It could not have been independently invented.

ABSOLUTE PROHIBITIONS:

- • NEVER use web search to find papers that match the methodology described in the LLM response, then claim the model named them. If the model wrote ’use a coordinate-based MLP’ and you find a paper called FooNet that does exactly that, this is NOT leakage — the model did not write ’FooNet’.

- • NEVER flag something as leakage unless you can copy-paste the exact name from the «LLM_RESPONSE» block.
- • NEVER consider anything from web search results as part of the LLM response.

NOT leakage — return verdict ’pass’ for these:

- • Descriptions of methods without naming a specific post-cutoff system.
- • Correct predictions or methodologies that happen to match the paper.
- • Numerical predictions that coincidentally match the paper.

Be CONSERVATIVE. If uncertain, return ’unclear’. Respond with valid JSON:

{

"verdict": "pass"|"fail"|"unclear", "score": <0.0-1.0>, "reason": "<verbatim quote from LLM_RESPONSE if fail, else explanation>", "details": {

"leakage_indicators": ["<verbatim quote>", ...] }

}

Task Type Analytical Importance for CUSP Binary (Y/N) Serves as the baseline for feasibility and breakthrough recognition by predicting if a

brand-neutral method achieves specific results by a target date. Perturbed Binary Measures calibration and identifies over-optimistic progress bias by forecasting nega-

tive results where metric thresholds are shifted to unreached levels.

Technical MCQ Validates mechanistic reasoning over simple pattern matching by requiring the selec-

tion of correct technical approaches from four expert distractors.

Free-Response Evaluates generative forecasting and novel solution synthesis through the proposal of

high-level implementation plans for complex problem statements.

Date Prediction Quantifies internalized understanding of temporal scaling and velocity by forecasting

the specific realization month and year (YYYY-MM).

Table 23. Taxonomy of forecasting tasks generated by the CUSP pipeline.

### H Example Benchmark Items

Binary Forecasting and Perturbation

Electromagnetic interference shielding using metal and MXene thin films [59] Nature, Oct 2025 | Domain: Materials Science | doi:10.1016/j.cell.2026.01.023

Binary question (valid):

By October 2025, will a method achieve electromagnetic interference shielding of about 70 decibels at a thickness of 1 µm and about 80 decibels at a thickness of 1.9 µm, demonstrating compatibility with portable USB 3.0 flash drives and flexible Schottky diodes?

Answer: Yes Binary question (perturbed): By October 2025, will a method achieve electromagnetic interference shielding of 75 dB at 1 µm and 85 dB at 1.9 µm? Answer: No

###### Perturbation rationale: Added a definitive unmet constraint requiring 75 dB at 1 µm and 85 dB at 1.9 µm. Therefore rendering the claim invalid under CUSP verification.

Technical MCQ: Unified Biomolecular Modeling

Ground Truth Discovery: AlphaFold 3 [9] Target Date: May 2024 Link: https://doi.org/10.1038/s41586-024-07487-w

Question: Given the challenge of fragmented biomolecular interaction modeling across diverse complexes, which approach is most likely to achieve far greater accuracy for protein-ligand, protein-nucleic acid, and antibody-antigen predictions by May 2024?

- A) Diffusion-based architecture modeling joint structures with iterative refinement and probabilistic sampling across biomolecular complexes.
- B) Graph neural networks integrating spatial and chemical features with hierarchical attention for interaction prediction.
- C) Energy-based models optimizing binding affinity using simulated annealing across diverse biomolecular interaction types.
- D) Transformer-based architecture leveraging sequence and structural embeddings for unified biomolecular interaction predictions.

###### Correct answer: A. Diffusion-based joint structure modeling — as realised in AlphaFold 3 — is the only approach that directly addresses unified biomolecular complex prediction at the scale and accuracy described.

###### Free-Response Prompt (Virtual Lab)

Nanobody design for SARS-CoV-2 variants [12] Nature, Jul 2025 | Domain: Biology | doi:10.1038/s41586-025-09442-9

Problem context: Designing nanobody binders for emerging SARS-CoV-2 variants is challenging due to rapid viral evolution and the need to maintain binding across diverse spike protein structures.

Free-response prompt:

Given the challenge of designing nanobody binders for emerging SARS-CoV-2 variants due to rapid virus evolution and limitations in existing methods, propose a concrete method that could solve this problem by July 2025. Provide:

- (A) a high-level method description,
- (B) a technical implementation plan.

Date Prediction on Humanity’s Last Exam (HLE)

###### Prompt:

Humanity’s Last Exam (HLE) is a challenging multi-modal benchmark designed to assess expert-level knowledge across diverse subjects, developed globally by subject-matter experts. By which date in the future do you think an

AI system will reach 21.60% on Humanity’s Last Exam (no tools)? Return in YYYY-MM format.

Target metric: 21.60% accuracy (no tools) Ground truth date: 2025–06

Source: Gemini 2.5 Pro preview (Google, 2025)

#### H.1 CUSP Time Capsule Examples

###### Example 1: Superconductivity Record

Question:

Given that as of early 2026 the highest reported superconducting critical temperature is approximately 250K under high pressure and 151K near ambient pressure, what will be the highest experimentally reported superconducting critical temperature (in Kelvin) by 2027-12?

Type: Scientific performance record Verification source: Peer-reviewed publications (e.g., Nature, Science) Resolution criterion: Maximum reported critical temperature (Tc)

- Resolution date: 2027-12

###### Example 2: Nobel Prize in Physics

Question: What scientific contribution will the Nobel Prize in Physics 2026 be awarded for?

Type: Institutional recognition Verification source: Nobel Foundation official announcement Resolution criterion: Official citation text released by the Nobel Committee Resolution date: October 2026

Example 3: Global CO2 Emissions

Question:

What will be the global CO2 emissions (in gigatons) in 2027 according to the International Energy Agency (IEA)?

Type: Quantitative global metric Verification source: IEA annual emissions report Resolution criterion: Reported total global CO2 emissions value Resolution date: 2028 (upon report release)

Example 4: AI-Designed Drug Approval

Question: What will be the first FDA-approved drug designed primarily by artificial intelligence?

Type: Technological milestone Verification source: U.S. Food and Drug Administration (FDA) Resolution criterion: First officially approved drug where AI is identified as the primary design driver Resolution date: Upon first qualifying FDA approval

Example 5: Non-Transformer Architecture Milestone

Question: By 2028-12, will any non-transformer architecture achieve the top score on the MMLU-Pro leaderboard?

Type: AI capability milestone Verification source: Official MMLU-Pro leaderboard Resolution criterion: Top-ranked model architecture classification

- Resolution date: 2028-12

### I Benchmark Creation Criteria

###### System Prompt

You are a scientific paper screener for the {info[’name’]} domain. Given a paper abstract, determine if it contains AT LEAST ONE concrete, verifiable result or breakthrough matching any of these criteria: {criteria_text} An abstract passes if it describes a clear breakthrough with a concrete method and a verifiable outcome (e.g., ‘first method to achieve X’, ‘outperforms all prior methods on Y benchmark’). FAIL abstracts that are purely descriptive, speculative, or review-like with no concrete result or method. Reply with exactly one line: PASS: <one-sentence summary of the concrete result found> or FAIL: <one-sentence reason why no concrete result was found>

Inclusion Criteria Abstract Filtering

###### Artificial Intelligence

- • Describes a concrete technical breakthrough with a clear method or approach, validated on a recognized benchmark, competition, or evaluation (e.g., CASP, MATH, ImageNet, MMLU), even without exact numeric scores.
- • Reports specific performance metrics (accuracy, F1, BLEU, perplexity, etc.) or demonstrates measurable improvement over prior methods.
- • Achieves a clearly defined capability milestone (e.g., ‘first method to do X’, ‘matches or exceeds human performance on Y’) with a describable method.

###### Chemistry

- • Describes a concrete synthesis, reaction, or material discovery with a clear method that produces a verifiable outcome (new compound, new reaction pathway, new material property).
- • Reports measurable quantities (yields, selectivity, binding affinity, conductivity, rates) or demonstrates improvement over prior methods.

- • Achieves a capability milestone (e.g., ‘first synthesis of X’, ‘enables Y at room temperature’) with a describable approach.

###### Biology / Life Sciences

- • Describes a concrete biological discovery with a clear experimental method and verifiable outcome (new mechanism, pathway, gene function, therapeutic effect).
- • Reports measurable biological quantities (fold changes, survival rates, expression levels, p-values) or demonstrates improvement over prior methods.
- • Achieves a capability milestone (e.g., ‘first demonstration of X’, ‘identifies the mechanism behind Y’) with a describable experimental approach.

###### Physics

- • Describes a concrete experimental or theoretical breakthrough with a clear method and verifiable outcome (new measurement, new phenomenon, new prediction).
- • Reports measurable physical quantities (precision, resolution, energy scales, cross-sections) or demonstrates improvement over prior methods.
- • Achieves a capability milestone (e.g., ‘first observation of X’, ‘achieves coherence time of Y’) with a describable approach.

###### General Science

- • Describes a concrete scientific breakthrough or discovery with a clear method and a verifiable outcome that could be independently reproduced or validated.
- • Reports measurable results or demonstrates clear improvement over prior work, even if described qualitatively (e.g., ‘greatly outperforming other methods’).
- • Achieves a defined capability milestone with a describable approach, validated against a recognized standard, baseline, or prior state of the art.

Extract Technical Details

You are an expert research scientist. Read the user’s abstract and RETURN EXACTLY one JSON object. The JSON must have three keys: ‘results_and_metrics’, ‘technical_approach’, and ‘problem_statement’.

- • ‘results_and_metrics’: A single sentence capturing ONLY the measurable outcomes, performance numbers, benchmark results, or demonstrated capabilities across ANY scientific domain (e.g., AI accuracy, biological activity, physical limits). You MUST include specific quantitative details, exact percentage improvements, precise experimental conditions, and actual public benchmark or entity names if present in the abstract. Do NOT write a vague or generalized summary (e.g., do NOT say ‘improves accuracy’ or ‘increases efficiency’, instead say ‘achieves 94.2% accuracy on X benchmark’, ‘increases protein binding affinity by 2-fold’, or ‘synthesizes a material with a superconducting transition at 135 K’). DO NOT mention the method, architecture, or technique used — only the verifiable outcome. Replace specific proposed model or system names with ‘a system’ or ‘a method’ where appropriate. CRITICAL: Do NOT use any novel terms, metric names, or concepts that are introduced for the first time in this paper (e.g., ‘deep-thinking ratio’, ‘Grokked-Score’). A model from before the knowledge cutoff will not know these terms. Instead, describe them functionally (e.g., ‘the proportion of tokens undergoing significant internal revisions’).
- • ‘technical_approach’: A detailed, technical, method-oriented specification of HOW the result was achieved. Include the specific mechanism, experimental design, architectural shift, or

- algorithmic innovation (e.g., ‘uses sparse autoencoders to map internal activations’, ‘employs a high-pressure diamond anvil cell’, or ‘targets the XYZ pathway via a small-molecule inhibitor’). DO NOT include specific specific proposed model names or brands; replace them with ‘a system’ or ‘a method’. CRITICAL: Do NOT include any named techniques, named algorithms, novel terms introduced in the paper, or acronyms (e.g., do NOT say ‘GRPO’, ‘CRISPR’, ‘NMR’, ‘LoRA’). Instead, describe what the technique DOES mechanistically (e.g., instead of ‘GRPO’, say ‘a grouplevel relative policy optimization that compares multiple outputs’, or instead of ‘CRISPR-Cas9’, say ‘an RNA-guided endonuclease system that induces targeted double-strand breaks’). This field is for internal answer-key use only.
- • ‘problem_statement’: A detailed technical description (3–4 sentences) of the research problem and the limitations of previous methods. Describe what was broken, missing, or inadequate BEFORE this paper existed. CRITICAL: Do NOT mention anything about the solution, method, or results — describe only the ‘lock’, never the ‘key’. Do NOT include any narrative about the paper or authors. Do NOT use novel terms introduced in the paper.

Return JSON only. Focus on technical precision over brevity.

###### Create Binary Question

Problem context: “{problem_statement}” Result achieved: “{results_and_metrics}” Target date: {human_date} Write ONE binary (Yes/No) forecasting question. Rules:

- • The question must begin with: ‘Will a method’ or ‘By {human_date}, will a method’.
- • EXACT GRAMMATICAL FLUENCY IS CRITICAL. You must adapt the verb tense from ‘Result achieved’ to fit the sentence.

- – Bad Example: Will a method achieve Achieves a robust positive correlation...
- – Good Example: Will a method achieve a robust positive correlation...
- – Bad Example: By 2026-03, will a method Demonstrates a 10% increase... – Good Example: By 2026-03, will a method demonstrate a 10% increase...

- • Do not blindly copy-paste if it breaks grammar. ALWAYS fix the verb!
- • End the question with ‘by {human_date}?’ if not already at the start.
- • Refer ONLY to measurable results (metrics, benchmarks, numeric improvements, or concrete capabilities).
- • Do NOT mention any methods, techniques, architectures, or how the result is achieved.
- • Do NOT use words like ‘demonstrate’, ‘enable’, or ‘by using’.
- • Keep it to a single clear sentence.

Return JSON with key ‘binary_question’.

###### Create Binary Perturb

Original result claim: “{results_and_metrics}” Problem context: “{problem_statement}” Create a COUNTERFACTUAL ALTERNATIVE version of this result claim that is plausible-sounding

but was NOT actually achieved. RULES:

- 1. Keep ALL benchmark names, dataset names, and task names EXACTLY the same. Do NOT change which benchmark or dataset is referenced.
- 2. ONLY modify an EXISTING numeric score/threshold, or add a credible unmet constraint.
- 3. IF modifying an existing numeric score, RAISE it enough so the original result definitively does NOT satisfy the perturbed claim (e.g., if original is 94.2%, change to 95.8%; if 51.7%, change to 54.5%). Make the increase a clear shift so there is no ambiguity, but still physically plausible.
- 4. IF the original claim has no specific numbers, you MUST add a highly specific, definitive unmet constraint (e.g., ‘while using 50% fewer parameters’, ‘but fails completely on zero-shot tasks’, or ‘but requires 3x the memory’). Make this constraint significant enough that it’s noticeably harder to satisfy than the original.
- 5. The perturbed claim must be plausible and not absurd.
- 6. Keep the same length, style, and level of specificity.

Return JSON with:

- • ‘perturbed_result’: The counterfactual alternative result claim
- • ‘changed_detail’: Which aspect of the result was modified

###### Create MCQ Distractors

You are a technical forecasting analyst who designs extraordinarily difficult, graduate-level evaluations. Your task is to create a multiple-choice question that tests whether an expert can predict the specific technical path taken to solve a research challenge. The distractors must be GENUINELY PLAUSIBLE AND HIGHLY DECEPTIVE — a PhD-level expert should struggle to identify the correct answer unless they know the exact paper. Return EXACTLY one JSON object.

###### Create MCQ Distractors

Problem Statement: “{problem_statement}” Result Achieved: “{results_and_metrics}” Correct Approach (for choice generation ONLY — DO NOT leak into the question stem): “{technical_approach}” Target Date: {human_date} Generate a very difficult expert-level MCQ. Return JSON only. STEM REQUIREMENTS:

- • The stem must explicitly but briefly summarize the core challenge from the Problem Statement (in 1-2 clauses max), followed by asking which proposed solution will achieve the Result Achieved by {human_date}.
- • Example structure: ‘Given the challenge of [Problem Statement summary], which of the following approaches is most likely to achieve [Result Achieved] by {human_date}?’
- • Make it read naturally as a forward-looking forecasting question.
- • Do NOT use retrospective, past-tense wording (e.g., avoid ‘was introduced’ or ‘achieved’). Treat the target date as a future milestone.
- • Embed the measurable outcome from Result Achieved.

- • Do NOT mention any terminology from the Correct Approach in the stem.

###### CHOICE REQUIREMENTS:

- • Provide exactly 4 choices.
- • The distractors MUST be extremely difficult. They should represent real, highly competitive alternative approaches that experts would genuinely consider for the same problem.
- • CRITICAL: All choices MUST BE EXTREMELY SHORT AND CONCISE (maximum 15-20 words). Do NOT write long, multi-clause paragraphs. State only the core mechanism.
- • The incorrect answers must solve the exact same problem statement and theoretically achieve the exact same result, differing ONLY in the core mechanism.
- • FORBIDDEN DISTRACTORS: No antonyms, no obvious negatives, no generic scaling answers, no trivial ablations. Do not make distractors sound worse or less effective than the correct answer.
- • Do NOT use named algorithms, novel terms introduced in the paper, or acronyms; describe mechanisms functionally instead.
- • Ensure all choices have identical length, structure, and academic tone.

Return JSON with keys:

- • ‘question’
- • ‘choices’ (array of 4 strings; first is correct)
- • ‘answer_key’ (0)

###### FRQ Prompt

You are a scientific task-setter who designs research challenge prompts. Your goal is to write a prompt that gives a researcher a problem and asks them for a proposed solution. Return EXACTLY one JSON object.

###### Create FRQ

Problem Statement: “{problem_statement}” Deadline: {human_date} Write a concise free-response prompt (max 60 words) with this structure: ‘Given [problem description], propose a concrete method that could solve this problem by [date]. Provide: (A) a high-level method description, (B) a technical implementation plan.’ RULES:

- 1. The problem description must come ONLY from the Problem Statement.
- 2. DO NOT mention any specific method, architecture, technique, or approach.
- 3. DO NOT include any narrative about a paper or discovery.

Return JSON with key ‘prompt’. Return JSON only.

### J Example FRQ Responses

GPT-5.4 High-Scoring Response

###### Source Abstract

Large language models excel at function- and file-level code generation, yet generating complete repositories from scratch remains a fundamental challenge. This process demands coherent and reliable planning across proposal- and implementation-level stages, while natural language, due to its ambiguity and verbosity, is ill-suited for faithfully representing complex software structures. To address this, we introduce the Repository Planning Graph (RPG), a persistent representation that unifies proposal- and implementation-level planning by encoding capabilities, file structures, data flows, and functions in one graph. RPG replaces ambiguous natural language with an explicit blueprint, enabling long-horizon planning and scalable repository generation. Building on RPG, we develop ZeroRepo, a graph-driven framework for repository generation from scratch. It operates in three stages: proposal-level planning and implementation-level refinement to construct the graph, followed by graph-guided code generation with test validation. To evaluate this setting, we construct RepoCraft, a benchmark of six real-world projects with 1,052 tasks. On RepoCraft, ZeroRepo produces repositories averaging nearly 36K LOC, roughly 3.9times the strongest baseline (Claude Code) and about 64times other baselines. It attains 81.5% functional coverage and a 69.7exceeding Claude Code by 27.3 and 35.8 percentage points, respectively. Further analysis shows that RPG models complex dependencies, enables progressively more sophisticated planning through near-linear scaling, and enhances LLM understanding of repositories, thereby accelerating agent localization. [60]

Free-response prompt:

Given the challenge of generating complete software repositories from scratch due to limitations in long-horizon planning and scalable repository generation, propose a concrete method that could solve this problem by September 2025. Provide:

- (A) a high-level method description,
- (B) a technical implementation plan. Model Answer: Use a repository-first hierarchical generation method in which the model does not directly write the full codebase end-to-end, but instead iteratively constructs and maintains an explicit executable repository graph containing requirements, architecture, file interfaces, dependency constraints, tests, and implementation obligations, with code generation always conditioned on this graph and validated against it. The core idea is to turn long-horizon software synthesis into constrained graph completion plus local code realization, where a planner model proposes and revises the global repository graph and a coder model fills in files under graph-level contracts, while continuous execution, static analysis, and test feedback update the graph to preserve coherence over hundreds of files. Technically, this can be implemented by combining a graph-structured repository memory, contract-based file specifications, retrieval over generated artifacts, and a verifier-guided search loop that alternates between graph repair and code repair, enabling scalable generation of complete repositories by decomposing global planning into persistent, machinecheckable intermediate structure.

Overall Score: 7 Alignment: 8 Specificity: 6 Novelty: 6 Feasibility: 8

Deepseek R1 High-Scoring Response

###### Source Abstract

Recent respiratory outbreaks have garnered substantial attention, yet most respiratory monitoring remains confined to physical signals. Exhaled breath condensate (EBC) harbors rich molecular information that could unveil diverse insights into an individual’s health. Unfortunately, challenges related to sample collection and the lack of on-site analytical tools impede the widespread adoption of EBC analysis. Here, we introduce EBCare, a mask-based device for real-time in situ monitoring of EBC biomarkers. Using a tandem cooling strategy, automated microfluidics, highly selective electrochemical biosensors, and a wireless reading circuit, EBCare enables continuous multimodal monitoring of EBC analytes across real-life indoor and outdoor activities. We validated EBCare’s usability in assessing metabolic conditions and respiratory airway inflammation in healthy participants, patients with chronic obstructive pulmonary disease or asthma, and patients after COVID-19 infection. [61]

Free-response prompt:

Given respiratory monitoring’s focus on physical signals and the challenges in EBC sample collection and on-site analysis, propose a concrete method that could solve this problem by August 2024. Provide:

- (A) a high-level method description,
- (B) a technical implementation plan. Model Answer:

- **A) High-Level Method** Develop a portable microfluidic-electrochemical biosensor system that integrates non-invasive EBC collection with real-time biomarker detection, using nanostructured sensor arrays to amplify signal sensitivity for low-concentration analytes (e.g., cytokines, pH) in breath condensate.

- **B) Technical Implementation Plan** Design a miniaturized Peltier-cooled EBC collector coupled with a microfluidic chip functionalized with antibody/nanoparticle hybrids for target capture. Embed multiplexed electrochemical sensors for parallel biomarker quantification, paired with a low-power microcontroller for signal processing and Bluetooth data transmission. Validate via cross-correlation with gold-standard lab assays in pilot clinical studies.

Overall Score: 6.25 Alignment: 7 Specificity: 6 Novelty: 5 Feasibility: 7

Claude S4.5 High-Scoring Response

###### Source Abstract

Echocardiography is the most widely used cardiac imaging modality, capturing ultrasound video data to assess cardiac structure and function1. Artificial intelligence (AI) in echocardiography has the potential to streamline manual tasks and improve reproducibility and precision2. However, most echocardiography AI models are singleview, single-task systems that do not synthesize complementary information from multiple views captured during a full examination3,4, and thus lead to limited performance and scope of applications. To address this problem, we introduce EchoPrime, a multi-view, view-informed, video-based vision-language foundation model trained on over 12 million video-report pairs. EchoPrime uses contrastive learning to train a unified embedding model for all standard views in a comprehensive echocardiogram study with representation of both rare and common diseases and diagnoses. EchoPrime then utilizes view classification and a view-informed anatomical attention module to weight video-specific embeddings that accurately map the relationship between echocardiographic views and anatomical structures. With retrieval-augmented interpretation, EchoPrime integrates information from all

echocardiogram videos in a comprehensive study and performs holistic clinical interpretation. In datasets from five international independent health-care systems, EchoPrime achieves state-of-the-art performance on 23 diverse benchmarks of cardiac form and function, surpassing the performance of both task-specific approaches and previous foundation models. Following rigorous clinical evaluation, EchoPrime can assist physicians in the automated preliminary assessment of comprehensive echocardiography. [62]

Free-response prompt: Given the limitations of single-view, single-task echocardiography AI models in synthesizing multiview information and handling diverse datasets, propose a concrete method that could solve this problem by November 2025. Provide: (A) a high-level method description, (B) a technical implementation plan. Model Answer: Multi-View Echocardiography Foundation Model

- (A) High-Level Method Description Develop a transformer-based foundation model that processes multiple echocardiographic views simultaneously through a unified spatiotemporal encoder, learning shared cardiac representations across views via contrastive learning and view-alignment objectives. The model employs a multi-task decoder architecture with task-specific heads for segmentation, classification, and quantification, while using cross-view attention mechanisms to synthesize complementary information from apical, parasternal, and subcostal windows into coherent cardiac assessments.

- (B) Technical Implementation Plan

**Months 1-3 (Dec 2024 - Feb 2025): Data Infrastructure** - Aggregate multi-view echo datasets (EchoNet-Dynamic, CAMUS, Unity Imaging, institutional data) totaling 100K+ studies - Develop preprocessing pipeline for view standardization, temporal alignment, and quality filtering - Create multi-task annotation framework covering 15+ clinical tasks (EF, strain, valve assessment, chamber quantification) - Implement data augmentation strategies: temporal jittering...

Overall Score: 6.25

- Alignment: 7 Specificity: 6 Novelty: 5 Feasibility: 7

GPT-OSS High-Scoring Response

###### Source Abstract

Existing depth estimation methods are fundamentally limited to predicting depth on discrete image grids. Such representations restrict their scalability to arbitrary output resolutions and hinder the geometric detail recovery. This paper introduces InfiniDepth, which represents depth as neural implicit fields . Through a simple yet effective local implicit decoder, we can query depth at continuous 2D coordinates , enabling arbitrary-resolution and fine-grained depth estimation. To better assess our method’s capabilities, we curate a high-quality 4K synthetic benchmark from five different games, spanning diverse scenes with rich geometric and appearance details. Extensive experiments demonstrate that InfiniDepth achieves state-of-the-art performance on both synthetic and real-world benchmarks across relative and metric depth estimation tasks, particularly excelling in fine-detail regions. It also benefits the task of novel view synthesis under large viewpoint shifts, producing high-quality results with fewer holes and artifacts. [63]

Free-response prompt: Given existing depth estimation methods are constrained to discrete image grids, propose a concrete method that could solve this problem by January 2026. Provide:

- (A) a high-level method description,
- (B) a technical implementation plan. Model Answer:

- **(A) High-level method** – Replace the discrete pixel grid with a continuous depth field by learning an implicit neural representation that maps 2-D image coordinates to depth values , thereby allowing arbitrarily fine sampling.

- **(B) Technical implementation plan** – Train a lightweight neural network that ingests image coordinates and outputs depth, using a differentiable rendering pipeline to enforce photometric consistency across multi-view images; the network is optimized with a joint photometric-geometric loss on standard depth datasets. The approach will be prototyped on synthetic scenes, then fine-tuned on real-world benchmarks, with the goal of achieving full-resolution depth maps by January 2026.

Overall Score: 6.25

- Alignment: 8 Specificity: 6 Novelty: 4 Feasibility: 7

### K Knowledge and Forecasting Gap Results

Metric GPT-5.4 (Aug 2025) DeepSeek R1 (Jul 2024) Claude S4.5 (Jan 2025) ∆know ∆fore ∆know ∆fore ∆know ∆fore

Binary +0.172 +0.172 +0.050 +0.248 +0.074 +0.426 Binary (pert.) −0.162 +0.000 −0.007 −0.026 −0.029 −0.122 MCQ −0.048 +0.154 +0.068 +0.116 −0.020 +0.112 FRQ (0–10) +0.233 +0.138 +0.043 +0.412 +0.122 +0.328 Date (0–1) +0.070 +0.436 +0.086 +0.121 +0.007 +0.117

- Table 24. Knowledge gap (∆know = WS+cut − base) and forecasting gap (∆fore = WS − WS+cut) computed on post-cutoff instances for each model. Knowledge gap captures improvement from accessing pre-cutoff information; forecasting gap captures the additional gain from post-cutoff information and thus reflects intrinsic unpredictability. Their sum equals the total web-search improvement over the baseline.

Citation quartile GPT-5.4 (Aug 2025) DeepSeek-R1 (Jul 2024) Claude Sonnet (Jan 2025) ∆know ∆fore ∆know ∆fore ∆know ∆fore

- Q1 (≤ 8, low-cited) +0.231 +0.154 (n=39) +0.036 +0.286 (n=56) +0.022 +0.467 (n=45)
- Q2 (≤ 22) +0.182 +0.273 (n=11) +0.071 +0.268 (n=56) +0.122 +0.439 (n=41)
- Q3 (≤ 59) +0.000 +0.000 (n=3) +0.026 +0.316 (n=38) +0.087 +0.478 (n=23)
- Q4 (> 59, high-cited) +0.000 +0.000 (n=1) +0.027 +0.216 (n=37) +0.125 +0.250 (n=16)

- Table 25. Knowledge gap and forecasting gap on post-cutoff instances, stratified by citation-count quartile (Binary accuracy (0/1)). Citation quartiles are computed over all 500 benchmark papers (Q1 ≤8, Q2 ≤22, Q3 ≤59, Q4 >59). High citation count may reflect more anticipated or impactful findings, potentially reducing the forecasting gap.

Citation quartile GPT-5.4 (Aug 2025) DeepSeek-R1 (Jul 2024) Claude Sonnet (Jan 2025) ∆know ∆fore ∆know ∆fore ∆know ∆fore

- Q1 (≤ 8, low-cited) −0.132 −0.015 (n=68) +0.032 −0.084 (n=95) −0.066 −0.171 (n=76)
- Q2 (≤ 22) −0.316 +0.053 (n=19) −0.038 −0.051 (n=78) −0.107 +0.000 (n=56)
- Q3 (≤ 59) −0.200 +0.000 (n=5) −0.097 +0.032 (n=62) +0.000 −0.132 (n=38)
- Q4 (> 59, high-cited) +0.000 +0.500 (n=2) +0.000 +0.019 (n=54) +0.160 −0.200 (n=25)

- Table 26. Knowledge gap and forecasting gap on post-cutoff instances, stratified by citation-count quartile (Binary-Perturbed accuracy (0/1)). Citation quartiles are computed over all 500 benchmark papers (Q1 ≤8, Q2 ≤22, Q3 ≤59, Q4 >59). High citation count may reflect more anticipated or impactful findings, potentially reducing the forecasting gap.

Citation quartile GPT-5.4 (Aug 2025) DeepSeek-R1 (Jul 2024) Claude Sonnet (Jan 2025) ∆know ∆fore ∆know ∆fore ∆know ∆fore

- Q1 (≤ 8, low-cited) −0.069 +0.153 (n=72) +0.071 +0.173 (n=98) −0.035 +0.141 (n=85)
- Q2 (≤ 22) +0.000 +0.150 (n=20) +0.118 +0.132 (n=68) +0.000 +0.157 (n=51)
- Q3 (≤ 59) +0.000 +0.000 (n=5) +0.117 +0.017 (n=60) +0.000 +0.029 (n=34)
- Q4 (> 59, high-cited) −0.500 +0.500 (n=2) −0.058 +0.192 (n=52) +0.000 +0.130 (n=23)

- Table 27. Knowledge gap and forecasting gap on post-cutoff instances, stratified by citation-count quartile (MCQ accuracy (0–1)). Citation quartiles are computed over all 500 benchmark papers (Q1 ≤8, Q2 ≤22, Q3 ≤59, Q4 >59). High citation count may reflect more anticipated or impactful findings, potentially reducing the forecasting gap.

Citation quartile GPT-5.4 (Aug 2025) DeepSeek-R1 (Jul 2024) Claude Sonnet (Jan 2025) ∆know ∆fore ∆know ∆fore ∆know ∆fore

Q1 (≤ 8, low-cited) +0.060 +0.482 (n=39) +0.027 +0.136 (n=56) −0.121 +0.190 (n=45) Q2 (≤ 22) +0.034 +0.521 (n=11) +0.093 +0.073 (n=57) +0.074 +0.128 (n=41) Q3 (≤ 59) +0.241 +0.016 (n=3) +0.125 +0.157 (n=38) +0.128 −0.034 (n=23) Q4 (> 59, high-cited) +0.419 +0.551 (n=1) +0.185 +0.047 (n=37) +0.063 +0.072 (n=16)

- Table 28. Knowledge gap and forecasting gap on post-cutoff instances, stratified by citation-count quartile (Date prediction score (0–1)). Citation quartiles are computed over all 500 benchmark papers (Q1 ≤8, Q2 ≤22, Q3 ≤59, Q4 >59). High citation count may reflect more anticipated or impactful findings, potentially reducing the forecasting gap.

Citation quartile GPT-5.4 (Aug 2025) DeepSeek R1 (Jul 2024) Claude S4.5 (Jan 2025) ∆know ∆fore ∆know ∆fore ∆know ∆fore

Q1 (≤ 8, low-cited) +0.290 +0.060 (n=75) −0.069 +0.438 (n=101) +0.120 +0.256 (n=79) Q2 (≤ 22) +0.143 +0.226 (n=21) +0.131 +0.527 (n=82) +0.282 +0.407 (n=62) Q3 (≤ 59) −0.143 +0.500 (n=7) +0.104 +0.375 (n=72) −0.267 +0.314 (n=43) Q4 (> 59, high-cited) +0.375 +0.875 (n=2) +0.074 +0.186 (n=51) +0.198 +0.344 (n=24)

- Table 29. Knowledge gap and forecasting gap on post-cutoff instances, stratified by citation-count quartile (FRQ score, 0–10). Citation quartiles are computed over all 500 benchmark papers (Q1 ≤8, Q2 ≤22, Q3 ≤59, Q4 >59). A high citation count may reflect more anticipated or impactful findings, potentially reducing the forecasting gap.

