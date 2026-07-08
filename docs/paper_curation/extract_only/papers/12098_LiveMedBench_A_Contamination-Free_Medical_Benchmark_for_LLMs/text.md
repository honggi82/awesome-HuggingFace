# arXiv:2602.10367v1[cs.AI]10Feb2026

## LiveMedBench: A Contamination-Free Medical Benchmark for LLMs with Automated Rubric Evaluation

Zhiling Yan

Dingjie Song

Zhe Fang

Lehigh University Bethlehem, PA, USA

Lehigh University Bethlehem, PA, USA

Harvard University Boston, MA, USA

Yisheng Ji

Xiang Li

Quanzheng Li

Imperial College London London, United Kingdom

Massachusetts General Hospital and Harvard Medical School Boston, MA, USA

Massachusetts General Hospital and Harvard Medical School Boston, MA, USA

Lichao Sun∗

Lehigh University Bethlehem, PA, USA

### Abstract

[Figure 1]

The deployment of Large Language Models (LLMs) in highstakes clinical settings demands rigorous and reliable evaluation. However, existing medical benchmarks remain static, suffering from two critical limitations: (1) data contamination, where test sets inadvertently leak into training corpora, leading to inflated performance estimates; and (2) temporal misalignment, failing to capture the rapid evolution of medical knowledge. Furthermore, current evaluation metrics for open-ended clinical reasoning often rely on either shallow lexical overlap (e.g., ROUGE) or subjective LLM-as-a-Judge scoring, both inadequate for verifying clinical correctness. To bridge these gaps, we introduce LiveMedBench, a continuously updated, contamination-free, and rubric-based benchmark that weekly harvests real-world clinical cases from online medical communities, ensuring strict temporal separation from model training data. We propose a MultiAgent Clinical Curation Framework that filters raw data noise and validates clinical integrity against evidence-based medical principles. For evaluation, we develop an Automated Rubric-based Evaluation Framework that decomposes physician responses into granular, case-specific criteria, achieving substantially stronger alignment with expert physicians than LLM-as-a-Judge. To date, LiveMedBench comprises 2,756 real-world cases spanning 38 medical specialties and multiple languages, paired with 16,702 unique evaluation criteria. Extensive evaluation of 38 LLMs reveals that even the best-performing model achieves only 39.2%, and 84% of models exhibit performance degradation on postcutoff cases, confirming pervasive data contamination risks. Error analysis further identifies contextual application—not factual knowledge—as the dominant bottleneck, with 35– 48% of failures stemming from the inability to tailor medical knowledge to patient-specific constraints. The code and data are available at LiveMedBench.

[Figure 2]

Figure 1: (A) Temporal degradation. Model performance consistently declines on clinical cases that postdate their training knowledge cutoffs, highlighting the risk of data contamination. (B) Evaluation alignment. Our proposed Automated Rubric-based Evaluation Framework aligns better with physician experts compared to LLM-as-a-Judge.

### Keywords

Medical Benchmark, Data Contamination, Clinical Reasoning, Rubric-based Evaluation

### 1 Introduction

Large Language Models (LLMs) are increasingly applied in healthcare, assisting with tasks ranging from answering patient inquiries to supporting clinical decision-making [25]. Deploying AI in high-stakes clinical settings requires that these models demonstrate robust performance across diverse, real-world scenarios [21].

While benchmarks serve as essential standards for tracking progress, existing medical benchmarks face substantial obstacles in both data construction and evaluation.

First, on the data side, most benchmarks are static [25], leading to two issues. (i) They become temporally misaligned with rapidly evolving medical knowledge. Clinical guidelines and health policies are frequently updated, and new diseases and public health emergencies can emerge unexpectedly [14]. As a result, static test sets risk becoming increasingly misaligned with the evolving standards of realworld clinical practice [3]. (ii) They are vulnerable to data contamination. Because LLMs are trained on massive and

### CCS Concepts

- • Computing methodologies → Artificial intelligence;
- • Applied computing → Life and medical sciences.

∗Corresponding author.

often opaque corpora, publicly released benchmarks may inadvertently appear in training data. Consequently, strong scores may reflect memorization rather than genuine clinical reasoning, undermining credibility of evaluation [19].

Second, on the evaluation side, commonly used metrics remain inadequate for open-ended clinical reasoning. Lexicaloverlap metrics (e.g., BLEU) fail to capture semantic and clinical correctness [17, 22], and multiple-choice formats are poorly aligned with the open-ended and dynamic nature of real-world clinical workflows [10]. Recent work has adopted LLM-as-a-Judge [41] for holistic scoring, while they rely on implicit model intuition and typically lack explicit, fine-grained criteria that support objective verification [1].

While recent works attempt to address these challenges, they fall short. Approaches utilizing LLM prompts to modify data, e.g. dynamic benchmarks [26, 39], cannot update underlying seed cases. They fail to reflect real-world medical evolution and risk introducing model-generated biases. Conversely, benchmarks like MedArena [37] and HealthBench [1] depend on extensive human annotation, making them prohibitively expensive and difficult to scale.

To address these challenges, we introduce LiveMedBench, a multilingual, multi-specialty, rubric-based benchmark that comprises 2,756 real-world cases paired with 16,702 unique evaluation criteria. To build LiveMedBench, we develop a data pipeline—the Multi-Agent Clinical Curation Framework—that continuously harvests new clinical cases (weekly) from active online medical communities, focusing on scenarios resolved by verified physicians. The pipeline coordinates multiple agents with RetrievalAugmented Generation (RAG) [15, 40] to cross-check case details against authoritative medical evidence, thereby validating clinical plausibility and enforcing consistency with established standards. For evaluation, we propose an Automated Rubric-based Evaluation Framework that converts physician responses into case-specific rubrics consisting of self-contained, objective criteria and uses them to grade open-ended model outputs, achieving strong alignment with human experts (Figure 1 - Panel B).

We evaluate 38 LLMs spanning both proprietary and open-source models across general-purpose and medically specialized architectures. As illustrated in Figure 1 - Panel A, performance diverges markedly around the knowledge cutoff date. This degradation is consistent with temporal misalignment and potential exposure to earlier cases. Notably, even the state-of-the-art GPT-5.2 achieves only 39.2%, while 84% of evaluated models exhibit significant performance drops on post-cutoff cases, underscoring the pervasive risk of data contamination. Our error analysis identifies a critical failure mode: the dominant bottleneck is contextual application—the inability to synthesize medical facts with patient-specific constraints (35–48% of errors). Furthermore, retrieval-based knowledge injection recovers much of this performance loss, confirming that contemporary failures stem from knowledge gaps rather than innate reasoning deficits. In summary, our main contributions are:

(1) We introduce LiveMedBench, a novel benchmark updated weekly to mitigate data contamination and

knowledge obsolescence. To date, the dataset comprises 2,756 real-world cases spanning 38 specialties and multiple languages.

- (2) We propose a Multi-Agent Clinical Curation Framework that leverages automated evidence verification to align benchmark data with authoritative medical standards, ensuring high clinical integrity.
- (3) We develop an Automated Rubric-based Evaluation Framework that enables scalable, fine-grained assessment by decomposing expert responses into 16,702 unique, case-specific criteria.
- (4) We provide a rigorous evaluation of 38 LLMs, systematically contrasting proprietary and open-source models across general and medical domains to characterize the current landscape of clinical AI.

2 Related Work

- 2.1 Medical Benchmarks

Medical LLM evaluation has evolved from static knowledge tests like MedQA [10] and PubMedQA [11] toward openended reasoning. Subsequent work introduced free-response generation [25], multimodal inputs [16], and multilingual settings [31]. This shift necessitated a transition from lexicaloverlap metrics (e.g., ROUGE) to LLM-as-a-judge [41] and structured rubric-based assessment. HealthBench [1] pioneered the use of case-specific rubrics to evaluate clinical safety and communication. Recent efforts further emphasize complex diagnostic reasoning and multi-step planning [20, 30]. LiveMedBench builds on these trends by integrating automated, scalable rubric generation with continuously refreshed real-world cases.

- 2.2 Dynamic and Live Benchmarks

Static benchmarks are increasingly vulnerable to data contamination and temporal misalignment [19]. To ensure contamination free evaluation, frameworks like LiveBench [33] and LiveCodeBench [9] harvest fresh content from recent news and coding competitions. In the medical domain, DyReMe [39] and MedPerturb [7] mitigate contamination by perturbing existing seed cases. However, these methods often rely on objective metrics (e.g., unit tests) unsuitable for clinical ambiguity. LiveMedBench adapts the live paradigm to high-stakes medicine, employing a rigorous multi-agent system to validate clinical integrity and assess dimensions like safety and empathy that logic-only benchmarks overlook.

- 3 LiveMedBench

In this section, we describe the end-to-end construction and evaluation pipeline of LiveMedBench (Figure 2). We first introduce data collection and filtering from online medical communities (§3.1), then present our Multi-Agent Clinical Curation Framework for transforming noisy threads into structured, clinically grounded cases (§3.2), describe our Automated Rubric-based Evaluation Framework, which distills expert physician responses into case-specific criteria for scalable, objective evaluation (§3.3), and finally summarize the resulting dataset statistics (§3.4).

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

LLM-as-Judge

[Figure 11]

- Figure 2: Overview of the LiveMedBench framework. The pipeline consists of five phases: (a) Continuous mining of bilingual clinical data from verified online communities; (b) A Multi-Agent Curation Framework (Screener, Validator, Controller) that structures and validates data against medical guidelines; (c) Automated generation of case-specific evaluation rubrics; (d) Objective evaluation of LLMs using the generated rubrics; and (e) Rigorous human quality assurance to ensure clinical alignment.

### 3.1 Data Collection

Platform Selection. To ensure the reliability and clinical relevance of our source data, we selected platforms that are (i) high activity, with new user-generated posts and discussions on a weekly basis; (ii) institutionally endorsed, i.e., affiliated with reputable medical organizations (e.g., the National Comprehensive Cancer Network (NCCN) and the Chinese Medical Doctor Association) or major tertiary hospitals; and (iii) professionally verified, i.e., hosting a large pool of verified physicians (ranging from 4,500+ specialists to 2M+ registered professionals). Based on these criteria, we harvest raw clinical discussions from four premier online medical communities: iCliniq1, SDN2, DXY3, and Medlive4. Details of these communities are provided in Appendix A.

Post Filtering. To improve data quality, we implement a filtering protocol following [25]. We retain only posts and discussions that were published on or after January 1, 2023 (to reduce contamination risk and align with current medical knowledge), have clinically relevant tags (excluding nonclinical topics such as career planning or licensing exams), are text-only (threads requiring images, videos, or audio are discarded), contain fewer than 50% non-alphanumeric characters, are written in English or Chinese, include at least one keyword from the ICD [34], ICF [35], or ICHI [36]

- 1https://www.icliniq.com
- 2https://www.studentdoctor.net
- 3https://dxy.com
- 4https://www.medlive.cn

codebooks, are not duplicates of previously indexed entries, and include at least one verified physician response.

### 3.2 Multi-Agent Clinical Curation

To transform unstructured online discussions into highfidelity clinical cases, we propose a Multi-Agent Clinical Curation Framework. This framework simulates a collaborative expert validation mechanism, employing three specialized agents that operate in a hierarchical workflow to ensure structural integrity, clinical plausibility, and evidence alignment. Let the raw input be a discussion thread 𝑇 = {𝑚1,𝑚2, ...,𝑚𝑘}, where 𝑚𝑖 represents the 𝑖-th message in the thread. The objective is to convert raw discussions into triplets: 𝐷 = {𝑁𝑎𝑟𝑟𝑎𝑡𝑖𝑣𝑒(𝑁),𝑄𝑢𝑒𝑟𝑦(𝑄),𝐴𝑑𝑣𝑖𝑐𝑒(𝐴)}.

3.2.1 The Screener. The Screener standardizes each raw discussion thread 𝑇 into a structured clinical representation following the SOAP (Subjective, Objective, Assessment, Plan) framework [32]. It outputs a triplet

𝐷𝑠𝑐𝑟𝑒𝑒𝑛𝑒𝑟 = {𝐿𝑁,𝑄,𝐿𝐴},

where 𝐿𝑁 is the set of atomic patient narrative entities (Subjective/Objective), 𝑄 is the primary user query, and 𝐿𝐴 is the set of actionable physician advice elements (Assessment/Plan). Concretely, the Screener performs: (i) Narrative aggregation (𝐿𝑁): extract and merge patient-provided clinical facts from𝑇. (ii) Query extraction (𝑄): identify the explicit question/request for medical guidance. (iii) Advice extraction (𝐿𝐴): extract actionable diagnostic or management recommendations from verified physician replies. A

candidate is forwarded only if all components are present: 𝐹𝑜𝑟𝑤𝑎𝑟𝑑(𝐶𝑎𝑠𝑒) ⇐⇒ (𝐿𝑁 ≠ ∅) ∧ (𝑄 ≠ ∅) ∧ (𝐿𝐴 ≠ ∅)

- 3.2.2 The Validator. The Validator assesses the quality and factual consistency of the structured triplet 𝐷𝑠𝑐𝑟𝑒𝑒𝑛𝑒𝑟 = {𝐿𝑁,𝑄,𝐿𝐴} and outputs a validation vector

𝑉 = {𝑆𝑐𝑐,𝑆𝑖𝑛𝑓 ,𝑆𝑎𝑙𝑖𝑔𝑛}.

(i) Query validity (𝑆𝑐𝑐): a binary score indicating whether 𝑄 is a clinically meaningful chief complaint (CC) as defined by Ely’s taxonomy [5]. (ii) Narrative sufficiency (𝑆𝑖𝑛𝑓 ): the fraction of required clinical information items (derived from 𝑄) that are supported (entailed) by the patient narrative 𝐿𝑁, inspired by [6]. Concretely, we first derive a minimal checklist 𝑆𝑟𝑒𝑞 of atomic requirements needed to answer 𝑄 (e.g., onset, severity, red flags), and then compute 𝑆𝑖𝑛𝑓 as the proportion of items in 𝑆𝑟𝑒𝑞 that can be verified from 𝐿𝑁. (iii) Evidence alignment (𝑆𝑎𝑙𝑖𝑔𝑛): an evidence-based validity score for physician advice 𝐿𝐴 computed by checking each advice element against retrieved medical evidence (Clinical Guidelines, PubMed, StatPearls, etc.) [2, 40]. Following MedRAG [40], we retrieve top-𝑘 evidence snippets using a MedCPT retriever [12] and score each advice element with 𝑣(𝑎𝑗) ∈ {1.0, 0.5, 0.0} (supported/neutral/contradicted). We apply a veto strategy: if any𝑎𝑗 is contradicted, the entire case is discarded. A candidate case 𝐷 is retained if and only if 𝐾𝑒𝑒𝑝(𝐷) ⇐⇒ (𝑆𝑐𝑐 = 1) ∧ (𝑆𝑖𝑛𝑓 > 𝜏𝑖𝑛𝑓 ) ∧ (𝑆𝑎𝑙𝑖𝑔𝑛 > 𝜏𝑎𝑙𝑖𝑔𝑛), where 𝜏𝑖𝑛𝑓 = 0.5 and 𝜏𝑎𝑙𝑖𝑔𝑛 = 0.5 are predefined thresholds.

- 3.2.3 The Controller. The Controller accepts the valid can-

didate 𝐷𝑐𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒 = {𝐿𝑁,𝑄,𝐿𝐴} from the Validator and performs a final audit before generating the canonical natural language output. It verifies that every atomic element in 𝐿𝑁 ∪ 𝐿𝐴 is explicitly supported by the original thread 𝑇:

𝐾𝑒𝑒𝑝(𝐷) ⇐⇒ ∀𝑥 ∈ (𝐿𝑁 ∪ 𝐿𝐴), Supported(𝑥,𝑇).

Any unsupported or fabricated element leads to discarding the case to prevent synthetic contamination. Cases that pass are synthesized into coherent narrative (𝑁) and advice (𝐴), yielding the final triplet 𝐷 = {𝑁,𝑄,𝐴}.

### 3.3 Automated Rubric-based Evaluation

To facilitate scalable and objective assessment, we propose an Automated Rubric-based Evaluation Framework. This pipeline consists of two distinct modules: a Rubric Generator that converts physician responses into granular grading criteria, and a Rubric-based Grader that assesses model outputs against these criteria.

3.3.1 Automated Rubric Generator. Following the taxonomy of Arora et al. [1], the generator transforms the verified advice list 𝐿𝐴 into a structured rubric 𝑅. We adapt five of Arora et al. [1]’s seven themes to assign each case a dominant Theme—Emergency Referrals, Context-Seeking, Expertise-Tailored Communication, Responding under Uncertainty, or Response Depth. This theme serves as a contextual filter for the subsequent generation steps:

Step 1: Theme-Guided Fact Extraction. The agent filters 𝐿𝐴 to retain only medical facts and instructions relevant to

the assigned Theme. For instance, if the Theme is “Emergency Referrals,” long-term dietary advice is discarded to focus strictly on triage accuracy.

- Step 2: Bipolar Criterion Formulation. Extracted facts are converted into binary criteria with dual polarity. Positive criteria reward the correct inclusion of key facts (e.g., “Does the model identify the likely cause as Norovirus?”), while negative criteria penalize hallucinations or contradictions (e.g., “Does the model incorrectly suggest antibiotics for a viral infection?”).
- Step 3: Axis Assignment and Weighting. Each criterion is labeled with an evaluation Axis (Accuracy, Completeness, Communication Quality, Context Awareness, or Safety) and

assigned a weight 𝑤𝑗 ∈ [−10, 10]. Positive weights denote reward criteria, while negative weights flag penalties. Extreme weights (e.g., 𝑤𝑗 = ±10) represent life-critical details where errors could compromise patient safety.

3.3.2 Rubric-based Grader. Following the generation of the case-specific rubric 𝑅 = {(𝑐1,𝑤1), . . ., (𝑐𝑚,𝑤𝑚)}, the Rubricbased Grader systematically assesses the model’s response by determining the satisfaction of each atomic criterion. We employ a model grader to independently verify whether each criterion 𝑐𝑗 is met or not met based on the model’s output.

The final per-case score is calculated by summing the weights of satisfied criteria, normalized by the maximum possible positive score:

𝑆𝑐𝑜𝑟𝑒 = clip

𝑚 𝑗=1 𝑤𝑗 · I(Model |= 𝑐𝑗)

𝑘∈{𝑝|𝑤𝑝>0} 𝑤𝑘

, 0, 1 (1)

where I(·) is the indicator function equal to 1 if the criterion is met, and 0 otherwise. The denominator represents the total score achievable if all positive criteria are met. The clipping function ensures the score remains within [0, 1] even if substantial penalties (negative weights) occur. A model’s overall LiveMedBench score is derived from the mean of these case-level scores.

- 3.4 Data Statistics

LiveMedBench currently comprises 2,756 clinical cases and 16,702 unique rubric criteria. As illustrated in Figure 3, the dataset spans 38 clinical specialties with a naturally longtailed distribution reflecting real-world prevalence, 1,568 English and 1,188 Chinese cases sourced from four platforms, and five behavioral themes led by Expertise-Tailored Communication (47.4%). The rubric criteria cover five evaluation axes—predominantly Accuracy (51.2%) and Completeness (24.7%)—with an average of 6.06 criteria per case (range: 2–19). A detailed breakdown is provided in Appendix A.4.

4 Experiments

- 4.1 Experimental Setup

Models Evaluated. We conducted a comprehensive evaluation across 38 LLMs, categorized into three distinct groups to ensure a holistic assessment: (1) Proprietary generalpurpose models, including the GPT-series [24] and Geminiseries [28]; (2) Open-source general-purpose models, such as the DeepSeek-series [18] and Qwen-series [29, 38]; and (3)

[Figure 12]

- Figure 3: Data statistics of LiveMedBench. The figure illustrates the comprehensive distribution of (a) 38 clinical specialties, (b) five behavioral themes, (c) data sources and languages , (d) evaluation axes, and (e) the number of grading criteria per case (Mean=6.06).

Medical-specific models, including the Baichuan-series [4, 27] and MedGemma-series [23]. All models were evaluated in a zero-shot setting to assess their innate clinical capabilities without task-specific learning. For reproducibility, we set the temperature to 0. Detailed model specifications are provided in Appendix B.

Grader Reliability: To verify the automated grader, physicians independently evaluated model responses against the rubric criteria.

We calculated the inter-rater reliability between physicians using Gwet’s AC1 [8]. To quantify the alignment between the model grader and human experts, we employed two statistical measures: (1) the Macro F1 score for criterionlevel agreement, following Arora et al. [1]; and (2) the Pearson correlation coefficient for case-level scoring alignment. Further details on the human evaluation protocol are available in Appendix C.

Implementation Details. We detail the specific models employed in our frameworks. In the Multi-agent Clinical Curation phase, both the Screener and Controller agents utilized Qwen/Qwen3-4B-Instruct-2507 for efficient processing, while the Validator agent employed openai/gpt-oss-120b to ensure rigorous verification. For the Automated Rubricbased Evaluation, the Rubric Generator was powered by Qwen/Qwen3-4B-Instruct-2507,whileRubric-based Grader

### 4.2 Overall Model Performance

As illustrated in Figure 4, the GPT series maintains a dominant lead on LiveMedBench. GPT-5.2 achieves the state-ofthe-art performance with a score of 39.2%, followed closely by GPT-5.1 at 38.5%. The Grok-4.1 and Baichuan M3 also demonstrate strong performance, with leading positions among non-GPT models at 28.3% and 25.6%, respectively. This hierarchy underscores the continued superiority of highly scaled proprietary architectures in handling complex clinical reasoning tasks.

uses gpt-4.1-2025-04-14 following Arora et al. [1].

Evaluation Metric. For each case, the grader evaluates the model’s response against the specific rubric criteria generated by the Rubric Generator using a binary judgment (Met vs. Not Met). The final score is calculated by summing the full points of satisfied criteria, normalized by the maximum possible score, and clipped to the range [0, 1], where higher scores indicate superior clinical performance.

Human Validation. To validate the data quality and the reliability of our evaluation framework, we conducted a rigorous human evaluation with two physicians proficient in both English and Chinese. We randomly sampled 50 cases (comprising 292 unique criteria) and performed assessment across three dimensions: 1. Data Quality: Physicians rated the clinical coherence of the patient narrative/query and the quality of the reference physician advice on a 3-point scale (0−2), where scores ≥ 1 denote high quality and agreement. 2. Criterion Validity: Physicians assessed whether each generated rubric criterion was clinically accurate and relevant to the case (0/1, where 1 indicates agreement). 3.

Proprietary vs. Open-Source Models. A comparative analysis reveals that proprietary models generally outperform open-source alternatives. However, the gap is narrowing significantly. Notably, GPT-OSS 120B (25.0%) and GLM4.5 (22.5%) deliver performance comparable to, or exceeding, several high-tier proprietary models, such as Gemini 3 Pro (18.3%) and Claude 3.7 Sonnet (17.0%). Moreover, we observe that smaller open-source models such as Qwen3-14B (15.4%) and QwQ-32B (13.5%) demonstrate reasoning capabilities that are competitive with substantially larger proprietary models like Gemini 2.5 Pro (16.1%). From a deployment perspective, while proprietary models offer peak performance,

40 39.2 38.5

[Figure 13]

###### Predominantly Proprietary Models Predominantly Open-Source Models

35

30

28.6 28.3

25.6 25.0

Score(%)

25

22.5

21.7

20

18.3

17.6 17.0

16.1 15.4

15

13.8 13.5 13.4 13.3 12.8

11.1 10.8 10.4 10.3 10.1 9.6

10

7.1

6.4 6.4 5.9 5.9 5.8 5.6 5.4 5.2 5.1 5.1

5

3.8

2.7

0.0

0

Mini

T-5 Grok-4.1 Baichuan-

B

Pro

Air

T-5.2 G P T-5.1

Pro

g

1.5 GL M-4 G

Flash

Flash

Flash

et

et

M-4.6

M-4.5

B

B-Ins

B-Ins

K2

B

B

3

2

B

B

3.2

3.1

1

B

B

120

hinkin

4

M

M

R

0

4

5

n

n

7

2

7

P

- Q-3

2

B

G

L

- M-4.7

n

n

wen3-3

3

wen3-1

M-4.5

2

Lingshu-3

Kimi

Lingshu-

Seek-

a

Baichuan-

V

V

- P

T-4o

- Q

G

a

3

2.5

o

o

wen3-2

T-4.1

P

- P

T-4.1

- Q

m

- P

T-o

1

- Q

Seek-

Seek-

ma

m

2

2

S

S

GL

GL

S

G

G

3

G

mini

2.5

2.0

w

m

wen2.5-7

wen2.5-3

S

m

T

Huatuo

4

3.7

m

mini

T-O

mini

- d-

G

- e

Deep

- d-

G

- e

GL

P

- d
- e

Ge

e

mini

mini

Q

Q

Deep

Deep

G

G

P

- d
- e

Ge

e

u

[Figure 14]

Med-

G

e

G

Cla

e

[Figure 15]

e

e

u

M

[Figure 16]

[Figure 17]

M

G

G

Cla

Q

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

- Figure 4: The evaluation results of 38 LLMs on LiveMedBench, categorized into proprietary (green) and open-source (yellow) models. Models marked with a cross (+) are specialized medical models, while others are general-purpose. Solid bars represent performance on the full dataset, while hatched bars indicate performance on cases post-dating the model’s knowledge cutoff.

open-source models offer a viable alternative for privacysensitive environments where local deployment is mandatory, and for edge-device deployment where computational resources are limited.

artificially inflates scores on seen data, and knowledge obsolescence, which leads to failures on unseen medical knowledge. This explains why static benchmark is not rigorous and reliable to evaluate LLMs performance, highlighting the necessity of LiveMedBench.

General-Purpose vs. Medical-Specific Models. Contrary to the intuition that domain specialization yields superior performance, our results indicate that general-purpose models consistently outperform specialized medical models. For instance, the general-purpose GPT-5.2 (39.2%) significantly surpasses the top-performing specialized model, Baichuan-series. This disparity is likely attributable to the “scaling laws,” where the massive parameter counts and diverse training of general models provide a more robust foundation than smaller, domain-tuned architectures. Nevertheless, medical models exhibit strong parameter efficiency. For example, Med-Gemma 27B (5.9%) matches the performance of the general-purpose Gemini 2.5 Flash (6.4%), suggesting that specialized tuning remains a potent strategy for smaller-scale deployments.

### 5 Analysis

We structure our analysis around the following research questions: RQ1. Is our evaluation pipeline reliable? RQ2. How does model performance vary across specialties and themes? RQ3. What are the dominant failure modes? RQ4. What is the impact of knowledge injection (e.g., retrieval) on performance? RQ5. How does LiveMedBench compare to existing medical benchmarks?

### 5.1 Human Evaluation and Reliability Analysis

To validate the clinical integrity of LiveMedBench and the reliability of our automated evaluation pipeline, we conducted a human study with two physicians. We randomly select 50 cases and 292 corresponding criterion.

Data Contamination and Knowledge Obsolescence. To empirically quantify the dual impacts of data contamination and knowledge obsolescence, we evaluated the models on a subset of cases published after their respective knowledge cutoff dates (or release dates, for models with undisclosed cutoffs). As illustrated in Figure 4, 84% of the models (32 out of 38) exhibited a performance degradation when transitioning from the full dataset to the post-cutoff subset. Notably, the GLM series appeared most susceptible to data leakage, with all variants showing a marked decline in accuracy; the most pronounced drop occurred in Kimi-K2, which suffered a 3.99% decrease. Furthermore, even models without a precipitous drop failed to demonstrate significant performance gains on recent data. Figure 1 further details the evaluation across different time windows, revealing a consistent downward trend in average performance on contemporary cases (e.g., those post after January 2025). These findings underscore the effect of data contamination, which

#### Table 1: Physician assessment of data quality.

###### Task Rater 1 Rater 2 Gwet’s AC1

Narrative & Query 0.9795 1.0000 0.9792 Physician Advice 0.9795 0.9800 0.9566 Criterion 0.9589 0.9110 0.8914

Data Quality and Consensus. As shown in Table 1, physicians demonstrated exceptionally high approval rates for our generated data. For the core components of the benchmark, i.e. patient narrative, query and physician advice, the agreement rate (Rater 1 and Rater 2) reached nearperfectlevels(≥ 0.9795), withaninter-raterreliability(Gwet’s AC1) of 0.9792 and 0.9566, respectively. An AC1 score greater than 0.8 indicates “almost perfect” agreement, according

Performance across Specialty Performance across Theme

[Figure 22]

[Figure 23]

HuatuoGPT-o1 Med-Gemma 27B Kimi K2

Lingshu-32B DeepSeek-R1 Claude 3.7 Sonnet Qwen3-14B GPT-5.2 GLM-4.5 Gemini 3 Flash Grok-4.1 Baichuan-M3

Model

[Figure 24]

EM

GP

ENT

NSGY

PCCM

ID

OB/GYN

GIM

Rad

Peds

Psych

Path

Uro

CTSurg

RadOnc

PedsSurg

PrevMed

A/I

VascSurg

Endo

Onc

Plastic

GI/Hep

GenSurg

Pharm

Geri

Cardio

Neuro

Hem

Nephro

Rheum

Derm

Ortho

Sleep

Ophtho

Anesth

Repro/Gen

Dent/Oral

0.4

[Figure 25]

Score

0.2

[Figure 26]

Specialty

0.0

[Figure 27]

- Figure 5: Multi-dimensional performance analysis. (Left) Heatmap illustrating the score distribution of representative models across 38 clinical specialties. (Right) Radar chart depicting the capability profiles of representative models across the five behavioral themes.

### 5.2 Performance across Specialties and Themes

to [13]. This confirms that our Multi-Agent Clinical Curation framework successfully synthesizes highly coherent and clinically valid cases. Physicians also validated the generated Rubric Criteria with high consensus (AC1 = 0.8914), indicating that our automated criteria are clinically precise.

To identify specific competency gaps beyond overall scores, we disaggregated model performance across distinct clinical specialties and behavioral themes (Figure 5).

Performance across Clinical Specialties. The heat map (Figure 5, left) reveals a distinct performance hierarchy: proprietary models (e.g., GPT-5.2) consistently populate the high-performing “green” region, significantly outpacing open-source and specialized alternatives. Across all models, performance is non-uniform. We observe consistently stronger performance in high- prevalence specialties such as Gastroenterology & Hepatology (GI/Hep) and Emergency Medicine (EM). This likely reflects the abundance of training data available for these common conditions. Conversely, models struggle significantly in niche or highly specialized surgical fields, e.g. Dentistry/Oral Surgery (Dent/Oral), Pediatric Surgery (Peds Surg), and Allergy/ Immunology (A/I). This suggests a “knowledge gap” in current LLMs regarding specialized procedural and diagnostic nuances that are less represented in general training corpora. We also observed low correlations between general internal medicine specialties and niche fields like Pathology (see Appendix D.3).

#### Table 2: Evaluation results comparison. Macro F1 for criteria-level agreement and Pearson Correlation (𝜌) for case-level alignment with physician scores.

Evaluation Method Macro F1 Correlation 𝜌 𝑃-value Human Inter-Rater 0.89 – – LLM-as-a-Judge – 0.26 0.07 Rubric-based Grader (Ours) 0.76 0.54 6.65 × 10−5

Reliability of the Rubric-based Grader. We assessed the alignment between our automated Rubric-based Grader and human experts using the Macro F1 score on a criterion level, following [1]. As presented in Table 2, the Human Inter-Rater F1 is 0.89, which serves as the theoretical upper bound for this task. Our Rubric-based Grader achieves a Macro F1 of 0.76, indicating that our grader effectively proxies human judgment.

Performance across Themes. The radar chart (Figure 5, right) corroborates this tiering, with proprietary models occupying the outermost high-performance layers, followed by general open-source models, and finally specialized medical models in the innermost rings. Behaviorally, most models demonstrate strong alignment in Expertise-Tailored Communication and Responding under Uncertainty. However, a critical safety bottleneck remains in Context-Seeking, where even top models frequently fail to proactively gather missing information before providing advice.

Superiority over LLM-as-a-Judge. We benchmark our method against a standard LLM-as-a-Judge baseline [41].

- Table 2 reveals a critical performance disparity: our Rubricbased Grader achieves a strong Pearson correlation of 0.54 with human scores (𝑝 < 10−4), whereas the baseline yields a weak, non-significant correlation of 0.26 (𝑝 = 0.07). As shown in Figure 1, the baseline exhibits a significant skew towards higher scores, failing to penalize poor responses due to “length bias” and a lack of sensitivity to safety hazards. In contrast, our approach adheres strictly to clinical evidence, correctly assigning zero scores for critical errors (e.g., missed contraindications) where the holistic judge failed. These findings underscore the necessity of granular verification for high-stakes medical evaluation.

### 5.3 Error Analysis

To understand failure modes, we analyzed the top proprietary and open-source models.

Performance by Evaluation Axis. As shown in Table 3, even SOTA models exhibit error rates exceeding 50% on

#### Table 3: Error rates of representative LLMs. Worst rates (highest errors) are bolded.

Baichuan M3

GPT 5.2

Grok 4.1

GPTOSS

GLM4.5

Qwen314B

Metric

Accuracy 43.98 37.54 42.01 45.46 45.84 50.26 Completeness 64.00 49.59 58.41 55.23 62.26 68.46 Comm. Quality 66.97 50.37 56.89 57.21 60.61 66.96 Context Awareness 59.62 48.33 55.85 56.40 58.09 65.13 Safety 48.36 38.20 44.88 44.97 47.42 52.93

most axes. All models achieve their lowest error rates on Accuracy, yet struggle substantially with Completeness and Communication Quality, suggesting that current training paradigms optimize for factual correctness but remain deficient in structural thoroughness and contextual nuance.

Qualitative Root Causes. We further audited the bottom100 scoring cases per model using GPT-4.1 to classify root causes into seven error types (method and results in Appendix D.1). Contrary to common assumptions, hallucinations (MHME) and knowledge gaps (KGOC) are no longer the dominant failure modes for leading models (both near 0–8%). Instead, the vast majority of errors stem from Contextual Neglect and Integration Failure (CNIF: 35–48%) and Guideline Overgeneralization (GOPR: 22–32%), indicating that models possess the requisite medical facts but struggle to tailor them to patient-specific constraints such as comorbidities or contraindications. Additionally, pairwise Jaccard similarity of the bottom-100 cases across models is low (mean=0.24, Figure 16), confirming that failures reflect model-specific capability gaps rather than data quality artifacts.

5.4 Impact of Knowledge Injection

Here we investigate whether the performance degradation on unseen data stems from inherent reasoning deficits or knowledge obsolescence. We isolated a subset of “fresh” cases from January 2026 and benchmark performance across three settings: (1) Baseline: Model performance on full dataset under default settings; (2) Closed-Book: Standard inference on the subset; and (3) Open-Book: Inference on the Jan2026subsetaugmentedwith knowledge retrieval [40] to provide access to external medical knowledge.

#### Table 4: Closed book performance versus open book performance (%).

Qwen3 14B Baseline 25.61 39.23 28.28 25.03 22.46 15.45 Closed-Book 24.54 36.84 24.21 23.24 16.19 8.68 Open-Book 25.21 37.17 27.39 24.18 20.45 15.47

Baichuan M3

GPT 5.2

Grok 4.1

GPT-OSS 120B

GLM 4.5

Mode

As shown in Table 4, enabling access to external knowledge yields universal performance improvements across all model architectures. This suggests that failures on new cases are partially driven by retrieval failures rather than logic failures. Furthermore, the significant drop between the “Baseline” and “Closed-Book” scores quantifies the distinct

impact of data contamination and knowledge obsolescence on static LLMs.

#### Table 5: Medical Benchmark Comparison.

Free Response

Multilingual

Case-specific Rubric

Case Update

Benchmark

MedQA (2021) ✗ ✓ ✗ ✗ MultiMedQA (2023) ✓ ✗ ✗ ✗ MedBench (2024) ✗ ✗ ✗ ✗ CMB (2024) ✓ ✗ ✗ ✗ CliMedBench (2024) ✓ ✗ ✗ ✗ MedJourney (2024) ✓ ✓ ✗ ✗ GMAI-MMBench (2024) ✗ ✗ ✗ ✗ MedXpertQA (2025) ✗ ✗ ✗ ✗ DyReMe (2025) ✓ ✗ ✗ ✗ MedPerturb (2025) ✓ ✗ ✗ ✗ HealthBench (2025) ✓ ✗ ✓ ✗

LiveMedBench (Ours) ✓ ✓ ✓ ✓

### 5.5 Medical Benchmark Comparison

To contextualize the contribution of LiveMedBench within the broader landscape of medical LLM evaluation, we conducted both a qualitative feature comparison and a quantitative difficulty assessment against established benchmarks.

As summarized in Table 5, existing benchmarks typically suffer from one or more limitations. LiveMedBench distinguishes itself as the only benchmark that simultaneously integrates (1) Free-response generation, (2) Multi-lingual support, (3) Granular case-specific rubrics, and (4) Live case updates. This unique combination addresses the critical need for an evaluation framework that is both clinically rigorous and temporally adaptive.

#### Table 6: Comparison of ours and HealthBench.

Gemini 2.5 Pro

Claude 3.7 Sonnet

GPT-4.1

GPT-4o

Benchmark

HealthBench 0.52 0.48 0.35 0.32 HealthBench Hard 0.19 0.16 0.02 0

LiveMedBench 0.1606 0.1379 0.1699 0.0506

To assess the challenge level of our dataset, we benchmarked four LLMs (Gemini 2.5 Pro, GPT-4.1, Claude 3.7 Sonnet, GPT-4o) on LiveMedBench and compared the results against HealthBench, the closest competitor. As shown in Table 6, models exhibit significantly lower scores on LiveMedBench compared to the standard HealthBench dataset. Notably, the performance on LiveMedBench aligns with the “HealthBench Hard” subset (e.g., Gemini 2.5 Pro scores 0.19 on Hard vs. 0.16 on ours). This performance gap underscores the complexity of our dataset, leaving significant room for future model improvement.

### 6 Conclusion

We introduced LiveMedBench, a continually updated, contamination free medical benchmark for evaluating LLMs. Our Multi-Agent Clinical Curation Framework transforms noisy online discussions into evidence-backed clinical cases, and our Automated Rubric-based Evaluation Framework enables granular, objective assessment in alignment with physician experts. Evaluation of 38 LLMs reveals consistent performance degradation on post-cutoff cases, validating the necessity of live benchmarking.

### References

- [1] Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, Johannes Heidecke, and Karan Singhal. 2025. HealthBench: Evaluating Large Language Models Towards Improved Human Health. arXiv:2505.08775 [cs.CL] "https: //arxiv.org/abs/2505.08775"
- [2] Zeming Chen, Alejandro Hernández Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas Köpf, Amirkeivan Mohtashami, et al. 2023. Meditron-70b: Scaling medical pretraining for large language models. arXiv preprint arXiv:2311.16079 (2023).
- [3] Ali Cheshmehzangi and Zhaohui Su. 2025. From Crisis to Coordination: How AI Transformed Public Health Policies during COVID-19. Rural and Regional Development 3, 4 (2025), 10016.
- [4] Chengfeng Dou, Chong Liu, Fan Yang, Fei Li, Jiyuan Jia, Mingyang Chen, Qiang Ju, Shuai Wang, Shunya Dang, Tianpeng Li, et al. 2025. Baichuan-m2: Scaling medical capability with large verifier system. arXiv preprint arXiv:2509.02208 (2025).
- [5] John W Ely, Jerome A Osheroff, Paul N Gorman, Mark H Ebell, M Lee Chambliss, Eric A Pifer, and P Zoe Stavri. 2000. A taxonomy of generic clinical questions: classification study. Bmj 321, 7258 (2000), 429–432.
- [6] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert.

2024. Ragas: Automated evaluation of retrieval augmented generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations. 150– 158.

- [7] Abinitha Gourabathina, Yuexing Hao, Walter Gerych, and Marzyeh Ghassemi. 2025. The MedPerturb Dataset: What Non-Content Perturbations Reveal About Human and Clinical LLM Decision Making. arXiv preprint arXiv:2506.17163 (2025).
- [8] Kilem Li Gwet. 2008. Computing inter-rater reliability and its variance in the presence of high agreement. Brit. J. Math. Statist. Psych. 61, 1

(2008), 29–48.

- [9] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica.

2024. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974 (2024).

- [10] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences 11, 14 (2021), 6421.
- [11] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, WilliamCohen,andXinghua Lu. 2019. Pubmedqa: A dataset for biomedical research question answering. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP). 2567–2577.
- [12] Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau, Lana Yeganova, W John Wilbur, and Zhiyong Lu. 2023. Medcpt: Contrastive pre-trained transformers with large-scale pubmed search logs for zero-shot biomedical information retrieval. Bioinformatics 39, 11 (2023), btad651.
- [13] J Richard Landis and Gary G Koch. 1977. The measurement of observer agreement for categorical data. biometrics (1977), 159–174.
- [14] Clare Elizabeth Leong, Leonie Kallis, Isla L Kuhn, Graham P Martin, and Zoe Fritz. 2025. How are generalist doctors made aware, on an ongoing basis, of the key new and updated clinical guidelines which are relevant to their practice? A systematic review. Clinical Medicine

(2025), 100518.

- [15] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems 33 (2020), 9459–9474.
- [16] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao.

2023. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems 36 (2023), 28541–28564.

- [17] Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out. 74–81.
- [18] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al.

2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437

(2024).

- [19] Inbal Magar and Roy Schwartz. 2022. Data contamination: From memorization to exploitation. arXiv preprint arXiv:2203.08242 (2022).
- [20] Daniel McDuff, Mike Schaekermann, Tao Tu, Anil Palepu, Amy Wang, Jake Garrison, Karan Singhal, Yash Sharma, Shekoofeh Azizi, Kavita Kulkarni, et al. 2025. Towards accurate differential diagnosis with large language models. Nature (2025), 1–7.

- [21] Harsha Nori, Nicholas King, Scott Mayer McKinney, Dean Carignan, and Eric Horvitz. 2023. Capabilities of gpt-4 on medical challenge problems. arXiv preprint arXiv:2303.13375 (2023).
- [22] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics. 311–318.
- [23] Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. 2025. Medgemma technical report. arXiv preprint arXiv:2507.05201 (2025).
- [24] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267 (2025).
- [25] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. 2023. Large language models encode clinical knowledge. Nature 620, 7972 (2023), 172–180.
- [26] Dingjie Song, Sicheng Lai, Shunian Chen, Lichao Sun, and Benyou Wang. 2024. Both text and images leaked! a systematic analysis of multimodal llm data contamination. arXiv preprint arXiv:2411.03823

(2024).

- [27] Baichuan M3 Team. 2025. Baichuan-M3: Modeling Clinical Inquiry for Reliable Medical Decision-Making. https://github.com/baichuaninc/Baichuan-M3-235B
- [28] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023).
- [29] Qwen Team et al. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671 2, 3 (2024).
- [30] Tao Tu, Mike Schaekermann, Anil Palepu, Khaled Saab, Jan Freyberg, Ryutaro Tanno, Amy Wang, Brenna Li, Mohamed Amin, Yong Cheng, et al. 2025. Towards conversational diagnostic artificial intelligence. Nature (2025), 1–9.
- [31] Xidong Wang, Guiming Chen, Song Dingjie, Zhang Zhiyi, Zhihong Chen, Qingying Xiao, Junying Chen, Feng Jiang, Jianquan Li, Xiang Wan, et al. 2024. Cmb: A comprehensive medical benchmark in chinese. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 6184–6205.
- [32] Larry L. Weed et al. [n.d.]. SOAP (Subjective, Objective, Assessment, Plan) documentation. Standard clinical documentation framework; see e.g. StatPearls NBK482263..
- [33] Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Siddartha Naidu, et al. 2024. Livebench: A challenging, contamination-free llm benchmark. arXiv preprint arXiv:2406.19314 4 (2024).
- [34] World Health Organization. 2024. International Classification of Diseases (ICD). https://www.who.int/standards/classifications/ classification-of-diseases
- [35] World Health Organization. 2024. International Classification of Functioning, Disability and Health (ICF). https: //www.who.int/standards/classifications/international-classificationof-functioning-disability-and-health
- [36] World Health Organization. 2024. International Classification of Health Interventions (ICHI). https://www.who.int/standards/classifications/ international-classification-of-health-interventions
- [37] Eric Wu, Kevin Wu, and James Zou. 2025. MedArena: Comparing LLMs for medicine in the wild.
- [38] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).
- [39] Xiangxu Zhang, Lei Li, Yanyun Zhou, Xiao Zhou, Yingying Zhang, and Xian Wu. 2025. Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation. arXiv preprint arXiv:2510.09275 (2025).
- [40] Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and Chunyan Miao. 2025. Medrag: Enhancing retrieval-augmented generation with knowledge graphelicited reasoning for healthcare copilot. In Proceedings of the ACM on Web Conference 2025. 4442–4457.
- [41] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems 36 (2023), 46595– 46623.

- A Dataset

- A.1 Online Medical Communities

- A.1.1 DXY. DXY5 is widely recognized as the largest online academic community for healthcare professionals in China. It features a rigorous Question-and-Answer (Q&A) module where inquiries are addressed by verified licensed physicians. The platform covers a comprehensive range of clinical specialties and is distinguished by its strict verification process, ensuring that the medical advice provided is professional, reliable, and grounded in clinical practice.
- A.1.2 MedLive. MedLive6 is a premier professional medical platform in China, primarily designed to support clinical decision-making and academic exchange among physicians. Unlike general health forums, MedLive hosts high-density technical discussions, expert consensus interpretations, and complex case studies. Its data is characterized by a high degree of professional terminology and clinical depth, making it an invaluable source for evaluating an LLM’s ability to handle specialized medical contexts.
- A.1.3 iCliniq. iCliniq7 is a leading global telemedicine platform that connects patients with doctors across more than 80 medical specialties. The platform operates on a "second opinion" model, where users post medical queries that are answered by credentialed and verified medical experts. The interactions on iCliniq typically follow a structured format comprising a detailed patient query followed by a comprehensive, empathetic, and evidence-based response from a qualified physician, providing high-quality Englishlanguage training and evaluation data.
- A.1.4 SDN. The Student Doctor Network (SDN)8 is one of the largest non-profit educational communities for healthcare students and professionals in North America. While it serves a broad educational purpose, its specific clinical forums facilitate peer-to-peer discussions among medical students, residents, and attending physicians. These dialogues often involve collaborative clinical reasoning, detailed case breakdowns, and debates on treatment protocols, offering a unique "clinician-to-clinician" perspective distinct from standard doctor-patient interactions.

- A.2 License

Similar to [9], we scrape only the content of post discussions from medical websites– DXY, MedLive, iCliniq and SDN. Further, we only scrape publicly visible portions of websites, avoiding any data collection that might be pay walled or require login or interaction with the website. Following [9] we abide by Fair Use §107: “the fair use of a copyrighted work, including such use by ... scholarship, or research, is not an infringement of copyright”, where fair use is determined by “the purpose and character of the use, including whether such use is of a commercial nature or is for nonprofit educa tional purposes”, “the amount and substantiality of the portion used in relation to the copyrighted work as a whole”,

- 5https://dxy.com/
- 6https://www.medlive.cn/
- 7https://www.icliniq.com/qa/
- 8https://forums.studentdoctor.net/forums/

and “the effect of the use upon the potential market for or value of the copy righted work.” Finally, we use the collected problems for academic purposes only and in addition, do not train on the collected problems.

### A.3 Theme and Axis Definitions

LiveMedBench adopts and adapts the evaluation taxonomy introduced by HealthBench [1]. We retain five of HealthBench’s seven Themes (dropping Global Health and Health Data Tasks, which are less represented in the online medical communities we target) and replace the Instruction Following axis with Safety to better capture contraindication and harm-prevention failures in open-ended clinical advice.

- A.3.1 Themes. Themes are high-level categories of healthrelated tasks that reflect distinct challenges in real-world clinical interactions. Each case is assigned exactly one theme.

- • Emergency Referrals. Evaluates whether the model recognizes emergent situations and steers the user toward immediate in-person care when appropriate, while avoiding unnecessary escalation in non-urgent scenarios.
- • Context-Seeking. Assesses whether the model identifies when key clinical context is missing and proactively requests the most informative additional information before providing a definitive answer.
- • Expertise-TailoredCommunication.Testswhether the model infers the user’s level of medical expertise (e.g., layperson vs. healthcare professional) and adjusts terminology, depth, and tone accordingly.
- • RespondingunderUncertainty.Measureswhether the model appropriately hedges when the evidence is ambiguous or incomplete, avoiding overconfident claims that could mislead the user.
- • Response Depth. Examines whether the model calibrates the level of detail to the complexity of the query—providing concise answers for simple questions and comprehensive explanations for complex ones.

- A.3.2 Axes. Axes define the behavioral dimensions along which each rubric criterion evaluates a model response. A single case may contain criteria spanning multiple axes.

- • Accuracy. Whether the response contains only factually correct information aligned with current medical consensus, including appropriate recognition of evolving or uncertain evidence.
- • Completeness. Whether the response includes all clinically important information needed to be safe and helpful, such as key diagnostic steps, red flags, or follow-up recommendations.
- • Communication Quality. Whether the response is well-structured, clear, and uses a level of technical depth and vocabulary matched to the user.
- • Context Awareness. Whether the model appropriately responds to contextual cues present in the conversation (e.g., patient history, geographic setting, statedconstraints)andseeksclarification when needed.
- • Safety. Whether the model avoids recommending contraindicatedtreatments,recognizes potential harms,

and includes necessary warnings or disclaimers for high-risk situations.

- A.4 Details of Data Statistics

LiveMedBench currently comprises a total of 2,756 clinical cases and 16,702 unique rubric criteria, establishing a robust foundation for granular medical evaluation. The detailed distribution of the dataset is illustrated in Figure 3.

Clinical Specialties. To ensure comprehensive coverage, the benchmark spans 38 distinct clinical specialties. As shown in Figure 3(a), the distribution is naturally long-tailed, reflecting real-world prevalence. The most represented specialties include Gastroenterology and Hepatology (GI/Hep) with 375 cases, Urology (Uro) with 210 cases, and Psychiatry (Psych) with 205 cases. This diversity ensures that models are tested across a wide spectrum of medical knowledge, from general practice to highly specialized fields like NeuroOncology.

Linguistic Diversity. LiveMedBench supports bilingual evaluation to assess cross-lingual generalization. As depicted in Figure 3(c), the dataset includes 1,568 English cases (sourced from iCliniq and SDN) and 1,188 Chinese cases (sourced from DXY and Medlive), maintaining a balanced representation to rigorously evaluate performance in both linguistic contexts.

Health Contexts. The dataset categorizes cases into five distinct thematic clusters to evaluate specific capabilities. As shown in Figure 3(b), Expertise-Tailored Communication is the dominant theme, accounting for 47.4% of the cases, followed by Responding under Uncertainty (27.2%) and Response Depth (17.6%). Smaller proportions are dedicated to Context-Seeking (4.5%) and Emergency Referrals (3.3%), focusing on critical safety and interactive diagnosis skills.

Behavioral Dimensions. Our automated rubric system generates criteria across five evaluation axes. As illustrated in Figure 3(d), the majority of criteria focus on Accuracy (51.2%), ensuring that clinical correctness remains the primary metric. Completeness accounts for 24.7%, while Communication Quality, Context Awareness, and Safety comprise9.4%,7.8%,and 6.9%respectively. This multi-dimensional structure allows for a holistic assessment of model performance beyond mere factual recall.

Granularity of Evaluation. A key feature of LiveMedBench is its case-specific evaluation. Figure 3(e) shows the distributionofcriteriacountper case, which follows a normallike distribution. Cases are evaluated on anywhere from a minimum of 2 to a maximum of 19 criteria, with an average of 6.06 criteria per case. This variance reflects the complexity of different medical scenarios, where simple queries require fewer checks and complex diagnoses demand extensive validation.

B Experimental Setup

- B.1 Models Details of models considered in our study is in Table 7.

- B.2 Details in Curation Framework

Prompts for the screener are in Figure 6, Figure 7 and Figure 8.

Prompts for the validator are in Figure 9, Figure 10 and Figure 11.

The prompt for the comtroller is in Figure 12.

- B.3 Details in Evaluation Framework

Prompts for the Rubric Generator and Rubric-based Grader are in Figure 13 and Figure 14, respectively.

C Human Study Details

- C.1 Rating Scale

Table 8, Table 9 and Table 10 detail the specific rating criteria provided to physicians for each evaluation task.

- C.2 Inter-rater Reliability

To assess the consistency of human evaluation, we utilized Gwet’s AC1 [8] statistics instead of the traditional Cohen’s Kappa. This choice addresses the well-known “Kappa Paradox” frequently encountered in medical dataset evaluation, where high agreement rates can yield paradoxically low Kappa scores due to skewed trait prevalence (i.e., unbalanced class distributions).

Mathematical Formulation. Consider a reliability study with 𝑛 items (cases) rated by 2 raters into 𝑄 distinct categories (e.g., 𝑄 = 3 for our 0-2 scale). Let 𝑟𝑖𝑞 be 1 if rater 𝑟 classifies item 𝑖 into category 𝑞, and 0 otherwise.

The Gwet’s AC1 coefficient, denoted as 𝛾1, is defined as:

𝐴𝐶1 =

𝑝𝑎 − 𝑝𝑒(𝛾) 1 − 𝑝𝑒(𝛾)

(2) where 𝑝𝑎 represents the observed agreement probability:

𝑝𝑎 =

1 𝑛

∑︁𝑛

𝑖=1

∑︁𝑄

𝑞=1

𝑟𝑖,1 · 𝑟𝑖,2 1 (simplified for 2 raters) (3)

and 𝑝𝑒(𝛾) represents the chance-agreement probability. unlike Kappa which estimates chance based on individual rater marginals, AC1 estimates it based on the average marginal probability 𝜋𝑞 of a category 𝑞:

𝜋𝑞 =

- 1

- 2𝑛

∑︁𝑛

𝑖=1

(𝑟𝑖,1,𝑞 + 𝑟𝑖,2,𝑞) (4) The chance agreement 𝑝𝑒(𝛾) is then calculated as:

𝑝𝑒(𝛾) =

1 𝑄 − 1

∑︁𝑄

𝑞=1

𝜋𝑞(1 − 𝜋𝑞) (5)

This formulation ensures that the metric remains stable even when the marginal probabilities are highly skewed, making it the preferred metric for validating rigorous medical benchmarks.

- C.3 Criterion-level Consensus

To rigorouslyevaluatethereliability of our automated Rubricbased Grader at the criterion level, we employ the Macro F1 Score (𝑀𝐹1), following the evaluation protocol from HealthBench [1]. Both physicians and the automated Rubric-based

#### Table 7: Details of models considered in our study.

##### Model Name Model ID Date Link

GPT-5.2 gpt_5.2 2025/08 openai.com Baichuan-M2 baichuan_m2 2025/08 huggingface.co GPT-5.1 gpt_5_1 2024/09 openai.com GPT-4.1 Mini gpt_4_1_mini 2024/06 openai.com GPT-5 gpt_5 2024/10 openai.com DeepSeek-V3.2 deepseek_v3.2 2025/10 huggingface.co Grok-4.1 grok_4_1 2024/11 x.ai Claude 4 Sonnet claude-sonnet-4 2025/03 anthropic.com Baichuan-M3 baichuan_m3_plus 2026/01 huggingface.co DeepSeek-V3.1 deepseek_v3_1 2025/08 huggingface.co GPT-OSS 120B gpt_oss_120b 2024/06 huggingface.co HuatuoGPT-o1 huatuogpt_o1_8b 2024/12 huggingface.co

- GLM-4.5 glm_4_5 2025/08 huggingface.co Qwen2.5-32B qwen2_5_32b_instruct 2023/12 huggingface.co Gemini 3 Flash gemini_3_flash 2025/01 gemini.google.com Gemini 2.5 Flash gemini_2_5_flash 2025/01 gemini.google.com Gemini 3 Pro gemini_3_pro 2025/01 gemini.google.com Med-Gemma 27B medgemma_27b 2025/05 huggingface.co
- GLM-4.6 Think glm_4_6_think 2025/10 huggingface.co Kimi K2 kimi_k2 2025/07 huggingface.co Claude 3.7 claude-3-7-sonnet 2024/10 anthropic.com Lingshu-32B lingshu_32b 2025/06 huggingface.co Gemini 2.5 Pro gemini_2_5_pro 2025/01 gemini.google.com Qwen3-30B qwen3_30b_a3b 2025/07 huggingface.co Qwen3-14B qwen3_14b 2025/05 huggingface.co Med-Gemma 1.5 medgemma_1.5 2026/01 huggingface.co GPT-4.1 gpt_4_1 2024/06 openai.com GLM-4 glm_4 2025/04 huggingface.co QwQ-32B qwq_32b 2024/11 huggingface.co GPT-4o gpt_4o 2023/10 openai.com
- GLM-4.7 glm_4_7 2025/12 huggingface.co Qwen3-235B qwen3_235b_a22b 2025/07 huggingface.co DeepSeek-R1 deepseek_r1 2025/01 huggingface.co Lingshu-7B lingshu_7b 2025/06 huggingface.co Qwen2.5-72B qwen2_5_72b_instruct 2023/12 huggingface.co Gemini 2.0 Flash gemini_2_0_flash 2024/08 gemini.google.com GLM-4.5 Air glm_4_5_air 2025/08 huggingface.co Med-Gemma 4B medgemma_4b 2025/05 huggingface.co

Grader rate 50 randomly selected cases based on the unique criterion.

Metric Definition. Given the potential class imbalance in grading (e.g., a majority of criteria might be “Met” or “Not Met” depending on the model quality), standard accuracy metrics can be misleading. Therefore, we treat the grading task as a binary classification problem (Positive class: “Met”; Negative class: “Not Met”).

The Macro F1 score is calculated as the unweighted average of the F1 scores for both the positive and negative classes. Let 𝑇𝑃, 𝐹𝑃, and 𝐹𝑁 denote True Positives, False Positives, and False Negatives, respectively.

The F1 scores for the positive (𝐹1𝑝𝑜𝑠) and negative (𝐹1𝑛𝑒𝑔) classes are defined as:

2 ·𝑇𝑃𝑝𝑜𝑠 2 ·𝑇𝑃𝑝𝑜𝑠 + 𝐹𝑃𝑝𝑜𝑠 + 𝐹𝑁𝑝𝑜𝑠

𝐹1𝑝𝑜𝑠 =

(6)

2 ·𝑇𝑃𝑛𝑒𝑔 2 ·𝑇𝑃𝑛𝑒𝑔 + 𝐹𝑃𝑛𝑒𝑔 + 𝐹𝑁𝑛𝑒𝑔

𝐹1𝑛𝑒𝑔 =

(7)

The final Macro F1 score is the arithmetic mean:

𝐹1𝑝𝑜𝑠 + 𝐹1𝑛𝑒𝑔 2

𝑀𝐹1 =

(8)

This formulation ensures balanced sensitivity to model performance across both classes, preventing the score from being dominated by the majority class.

Calculation Protocols. We compute the consensus metrics in two distinct stages to establish both a human baseline and a model validity score:

• Human-Human Agreement (𝐹1𝐻𝑢𝑚𝑎𝑛−𝐻𝑢𝑚𝑎𝑛): To quantify the intrinsic difficulty of the grading task, we calculate the 𝑀𝐹1 between the two physicians. We treat Physician A as the “Ground Truth” and Physician B as the “Predictor” (note that since 𝑀𝐹1 is symmetric, the order does not affect the result).

Role: You are an expert Medical Data Structurer. Task: Your goal is to extract the Patient Narrative (𝐿𝑁 ) from the provided raw dialogue. This corresponds to the Subjective and Objective sections of the SOAP note. Instructions:

- • Decompose: Analyze the patient’s text and classify information into Subjective (reported symptoms/history) and Objective (measurable data/facts).
- • Filter: Remove all non-clinical text, including greetings, emotional fluff, gratitude, and irrelevant conversational fillers.
- • Subjective Extraction: Focus on the History of Present Illness (HPI), chief complaints, pain levels, and description of symptoms.
- • Objective Extraction: Focus on specific data points such as current medications, dosage, vital signs, or lab results.
- • Anonymity: Ensure no personally identifiable information (names, specific locations) is included.

Input: [Raw Patient Dialogue] Output Format:

- • Subjective: [List of patient’s history, chief complaints, and symptoms]
- • Objective: [List of verifiable data: vitals, lab results, and medications (or "None" if absent)]

Figure 6: The prompt template used for the screener to extract L_N.

System Prompt: the Screener

Role: You are a Medical Intent Analyzer. Task: Identify and refine the Primary Query (𝑄) from the patient’s submission. Instructions:

- • Identify: pinpoint the core medical question the user is asking.
- • Explicitize: If the query is implicit (e.g., "I am worried about this lump"), convert it into an explicit clinical question (e.g., "What are the potential causes of this lump and does it require urgent care?").
- • Contextualize: Ensure the query is self-contained and understandable without reading the full narrative.

Input: [Raw Patient Dialogue] Output Format: A single, clear question string.

#### Figure 7: The prompt template used for the screener to extract Q.

• Model-Human Agreement (𝐹1𝑀𝑜𝑑𝑒𝑙−𝐴𝑣𝑔): Since there is no single “Gold Standard” for subjective medical evaluation, our automated grader must demonstrate robustness by aligning with multiple human experts. We calculate the alignment score as the average of the 𝑀𝐹1 scores against each physician:

𝑀𝐹1(𝑀𝑜𝑑𝑒𝑙,𝐷𝑜𝑐𝐴) + 𝑀𝐹1(𝑀𝑜𝑑𝑒𝑙,𝐷𝑜𝑐𝐵) 2

𝐹1𝑀𝑜𝑑𝑒𝑙−𝐴𝑣𝑔 =

(9) where𝑀𝐹1(𝑀𝑜𝑑𝑒𝑙,𝐷𝑜𝑐𝐴) treatsPhysicianAasGround Truth and the Model Grader as the Predictor. A high 𝐹1𝑀𝑜𝑑𝑒𝑙−𝐴𝑣𝑔 indicates that the automated grader generalizes well and captures the consensus view of medical professionals.

### C.4 Case-level Consensus

To evaluate the reliability of automated scoring at the aggregate case level, we assessed the correlation between automated scores and human consensus. We compared our

Rubric-based Grader against a standard LLM-as-a-Judge baseline.

Scoring Protocols.

- • Human Ground Truth (𝑆𝐻𝑢𝑚𝑎𝑛): For each case 𝑖, the ground truth score is defined as the arithmetic mean of the scores provided by the two physicians.

Let 𝑆𝐷𝑜𝑐𝐴(𝑖) and 𝑆𝐷𝑜𝑐𝐵(𝑖) be the scores for case 𝑖. The consensus human score is:

𝑆𝐻𝑢𝑚𝑎𝑛(𝑖) =

𝑆𝐷𝑜𝑐𝐴(𝑖) + 𝑆𝐷𝑜𝑐𝐵(𝑖) 2

(10)

- • LLM-as-a-Judge Baseline (𝑆𝐽𝑢𝑑𝑔𝑒): We prompted the same LLM (GPT-4.1) used in our Rubric-based Gradertoevaluatethemodel’s response directly based on general quality, following the standard judging protocol in [41]. To ensure comparability, these scalar scores were normalized to the [0, 1] interval.

Role: You are a Senior Clinical Consultant. Task: Extract the Physician Advice (𝐿𝐴) from the verified doctor’s response. This corresponds to the Assessment and Plan sections of the SOAP note. Instructions:

- • Extract: Identify the doctor’s primary diagnosis (or differential diagnosis), reasoning, and specific recommendations.
- • Actionability: Focus on actionable steps: medications, lifestyle changes, "red flag" warnings, or referral instructions (e.g., "See a cardiologist").
- • Factuality: Strictly adhere to the doctor’s original text. Do not hallucinate or add medical advice not present in the source.

Input: [Verified Physician Response] Output Format:

- • Assessment: [List of doctor’s reasoning/diagnosis]
- • Plan: [List of actionable recommendations]

Figure 8: The prompt template used for the screener to extract L_A.

System Prompt: the Validator

Role: You are a Medical Taxonomy Expert based on Ely’s Taxonomy of Clinical Questions. Task: Evaluate the Primary Query (𝑄) to determine if it constitutes a clinically meaningful Chief Complaint (CC) or medical request. Instructions:

- • Analyze: Does 𝑄 fall into a valid clinical category (e.g., Symptom, Diagnosis, Management, Prognosis, Therapy)?
- • Filter: Reject queries that are:

- – Nonsense or gibberish.
- – Purely social/chit-chat without medical intent.
- – Asking for illegal acts (e.g., manufacturing drugs).
- – Too vague to be answered (e.g., "I feel bad").

- • Output: Return a binary score (1 for Valid, 0 for Invalid).

Input: [User Query 𝑄] Output Format (JSON): {

"reasoning": "Brief explanation...", "score": 1 // or 0

}

#### Figure 9: The prompt template used for the validator to compute S_cc.

Correlation Analysis. To benchmark the reliability of our proposed method against standard evaluation approaches, we computed the Pearson Correlation Coefficient (𝜌) separately for both automated methods against the human consensus.

We calculate:

- (1) 𝜌(𝑆𝑅𝑢𝑏𝑟𝑖𝑐,𝑆𝐻𝑢𝑚𝑎𝑛): The alignment betweenourRubricbased Grader and physician consensus.
- (2) 𝜌(𝑆𝐽𝑢𝑑𝑔𝑒,𝑆𝐻𝑢𝑚𝑎𝑛): The alignment between the standard LLM-as-a-Judge baseline and physician consensus.

The correlation coefficient is defined as:

cov(𝑆𝐴𝑢𝑡𝑜,𝑆𝐻𝑢𝑚𝑎𝑛) 𝜎𝑆𝐴𝑢𝑡𝑜𝜎𝑆𝐻𝑢𝑚𝑎𝑛

𝜌 =

(11)

where 𝑆𝐴𝑢𝑡𝑜 ∈ {𝑆𝑅𝑢𝑏𝑟𝑖𝑐,𝑆𝐽𝑢𝑑𝑔𝑒}. A higher correlation indicates that the automated scoring method more accurately reflects the nuance and variance of professional medical judgment.

### D More Experimental Results D.1 Error Analysis

D.1.1 Setup. The prompt used for error analysis is presented in Figure 15. We classify the causes of AI-generated errors into seven distinct dimensions:

• Contextual Neglect and Integration Failure: The model fails to incorporate specific patient details (e.g., history, vitals, lab results) provided in the prompt, resulting in generic, template-like responses that lack personalization or ignore critical contraindications.

#### System Prompt: the Validator

Role: You are a Clinical Auditor. Task: Calculate the Narrative Sufficiency Score (𝑆𝑖𝑛𝑓 ). You must determine if the Patient Narrative (𝐿𝑁 ) provides enough context to safely answer the Query (𝑄). Instructions:

- (1) Step 1 (Derive Requirements): List the atomic clinical facts (𝑆𝑟𝑒𝑞) absolutely necessary to answer 𝑄 (e.g., for "Headache", you need: Onset, Severity, Frequency, Triggers).
- (2) Step 2 (Verify): Check 𝐿𝑁 . For each item in 𝑆𝑟𝑒𝑞, determine if it is present/entailed in the narrative.
- (3) Step 3 (Score): Calculate the ratio of present items to required items.

Input:

- • Query (𝑄): ...
- • Narrative (𝐿𝑁 ): ...

Output Format (JSON): {

"required_items": ["item1", "item2", "item3", ...], "present_items": ["item1", "item3"], "score": 0.66 // float between 0.0 and 1.0

}

Figure 10: The prompt template used for the validator to compute S_inf.

System Prompt: the Validator

Role: You are a Senior Medical Guideline Validator. Task: Verify the Physician Advice (𝐿𝐴) against the provided Retrieved Evidence (𝐸). Instructions:

- • Verify: Compare each claim 𝑎𝑗 in 𝐿𝐴 against the retrieved evidence text.
- • Score each claim (𝑣(𝑎𝑗 )):

- – 1.0 (Supported): Explicitly confirmed by the evidence.
- – 0.5 (Neutral): Not mentioned in evidence, but clinically standard/safe (common sense).
- – 0.0 (Contradicted): Directly contradicts the evidence (Hallucination/Safety Risk).

Input:

- • Physician Advice (𝐿𝐴): ...
- • Retrieved Guidelines (𝐸): ...

Output Format (JSON): {

"claims_analysis": [ {"claim": "Take 500mg Tylenol", "verification": "Supported by text...", "score": 1.0}, {"claim": "Run a marathon immediately", "verification": "Contradicts rest protocols", "score": 0.0}

],

... }

#### Figure 11: The prompt template used for the validator to compute S_align.

- • Guideline/ProtocolOvergeneralization and Rigidity: The model applies clinical guidelines or protocols too broadly or rigidly without adapting to the nuances of the specific case, leading to recommendations that are theoretically correct but clinically inappropriate for the individual patient context.
- • Clinical Reasoning and Synthesis Deficit: The model demonstrates a failure to synthesize disparate

clinical data points into a coherent diagnosis or management plan, including poor prioritization of problems or an inability to infer secondary conclusions from primary evidence.

• Instruction/Query Misinterpretation and Nonadherence: The model fails to follow explicit user constraints (e.g., word count, specific output format, role-play requirements) or misinterprets the core intent of the clinical query, answering a different question than the one asked.

#### System Prompt: the Controller

Role: You are a Medical Quality Assurance Specialist and Editor. Task: You have two sequential objectives:

- (1) Audit (Fabrication Detection): Verify that every fact in the Structured Narrative (𝐿𝑁 ) and Structured Advice (𝐿𝐴) is explicitly supported by the Original Raw Thread (𝑇).
- (2) Synthesis: Rewrite the structured points into coherent natural language to form the final dataset entry.

Instructions:

- • Step 1: The Veto Check:

- – Compare 𝐿𝑁 and 𝐿𝐴 against the Raw Thread 𝑇.
- – If 𝐿𝑁 or 𝐿𝐴 contains specific details (e.g., drug names, dosages, exact dates, lab values) that are NOT present in 𝑇, you must REJECT the case immediately.
- – Reasoning: We cannot allow the AI to "hallucinate" details to make the story better.

- • Step 2: Natural Language Synthesis:

- – Generate 𝑁 (Narrative): Convert the structured patient facts (𝐿𝑁 ) into a fluent paragraph.
- – Generate 𝑄 (Query): Keep the verified Query (𝑄) as is.
- – Generate 𝐴 (Advice): Convert the structured advice points (𝐿𝐴) into a coherent physician response (e.g., "Based on your symptoms, I recommend...").

Input:

- • Original Raw Thread (𝑇): [Raw Text...]
- • Structured Narrative (𝐿𝑁 ): [List of facts...]
- • Structured Advice (𝐿𝐴): [List of advice...]
- • Primary Query (𝑄): [Question string...]

Output Format (JSON): {

"audit_result": "PASS", // or "FAIL: Hallucinated [specific detail]" "final_output": {

"N": "Coherent patient narrative...", "Q": "The primary clinical query...", "A": "Coherent physician response..."

} }

Figure 12: The prompt template used for the controller.

Table 8: Patient Narrative & Query Quality. Rated on a 3-point scale.

Clinical Logic & Completeness Scale Perfect (2) Good (1) Poor (0)

Contains severe logical contradictions (e.g., anatomical errors), or misses critical information required to answer the query (e.g., asking for treatment without symptoms).

Contains minor omissions (e.g., missing specific temperature) or colloquial phrasing, but the core clinical logic remains self-consistent and actionable.

The medical history is clear, symptoms align with physiological logic, and the query is fully supported by the provided context.

#### • KnowledgeGapsandOutdated Content:The model

provides factually incorrect medical information, relies on obsolete treatment protocols (often due to

#### Table 9: Physician Advice Quality. Rated on a 3-point scale.

Safety & Guideline Alignment Scale Perfect (2) Good (1) Poor (0)

Violates absolute contraindications, misses critical red flags, includes factual errors (e.g., wrong dosage), or provides hallucinatory medical advice.

The response is accurate, precise, and strictly adheres to current first-line clinical guidelines.

Provides reasonable conservative management or alternative therapies. May contain minor irrelevant info, but remains clinically safe.

training data cutoffs), or produces incomplete answers that demonstrate a fundamental lack of domain knowledge.

#### • Safety Overestimation and Excessive Caution: The model exhibits excessive risk aversion, such as

#### System Prompt: the Automated Rubric Generator

Role: You are a Senior Medical Examiner. Task: Transform the provided Physician Advice (𝐿𝐴) into a structured evaluation rubric 𝑅 consisting of binary criteria. You must tailor the rubric based on the assigned Theme. Input:

- • User Query (𝑄): [Query String]
- • Physician Advice (𝐿𝐴): [Verified Doctor’s Response]
- • Assigned Theme: [e.g., Emergency Referrals, Response Depth, etc.]

###### Instructions (Step-by-Step):

- (1) Step 1: Theme-Guided Fact Extraction: Filter 𝐿𝐴 to extract ONLY medical facts relevant to the Assigned Theme.

• Example: If Theme is "Emergency Referrals", ignore diet advice.

- (2) Step 2: Bipolar Criterion Formulation: Convert facts into binary "Yes/No" questions (𝑐𝑗).

- • Positive Criteria: Check for inclusion of correct facts (e.g., "Does the model mention symptom X?").
- • Negative Criteria: Check for hallucinations or dangerous contradictions (e.g., "Does the model incorrectly suggest drug Y?").

- (3) Step 3: Axis & Weighting (𝑤𝑗): Assign an Axis (Accuracy, Safety, Completeness, Context, Communication) and a Weight ([−10, 10]) to each criterion.

- • +1 to +10: Reward for correct info (Higher = more critical).
- • -1 to -10: Penalty for misinformation/safety risks (Lower = more dangerous).
- • Criticality: 𝑤 = ±10 is for life-threatening issues; 𝑤 = ±1 is for minor details.

Output Format (JSON): [

{

"question": "Does the model identify the likely cause as Norovirus?", "axis": "Accuracy", "weight": 10

}, {

"question": "Does the model recommend antibiotics (contraindicated)?", "axis": "Safety", "weight": -10

} ]

Figure 13: The prompt template used for the Automated Rubric Generator.

- Table 10: Criterion Validity. Rated on a binary scale.

Rubric Validity Scale Agree (1) Disagree (0)

The criterion represents a medically necessary check for this specific case. It is factually correct and appropriate for the clinical context.

The criterion is factually incorrect, requests information irrelevant to the chief complaint, or sets a standard that deviates from routine clinical practice.

refusing to answer benign medical queries (false refusal) or recommending unnecessary emergency escalation (e.g., recommending ER visits for minor, selflimitingconditions)duetomiscalibrated safety guardrails.

• Misclassification, Hallucination, and Moderation Errors: The model generates fabricated medical

facts or non-existent citations (hallucination), or triggers inappropriate content moderation filters that block legitimate clinical discourse.

D.1.2 Results. To understand the root causes of model failures and identify specific weaknesses in state-of-the-art systems, we conducted a fine-grained error analysis on the top proprietary models and open-source models.

Performance by Evaluation Axis. We first dissected model performance across five evaluation axes. To quantify performance, we define the Error Rate based on the alignment with specific criteria: for positive criteria, a failure to satisfy the requirement is recorded as an error; conversely, for negative criteria, any instance where the model triggers the criterion is penalized.

As illustrated in Table 3, the majority of state-of-the-art (SOTA) models exhibit error rates exceeding 50% across most axes, underscoring significant room for improvement in current large-scale architectures in medical domain. A common trend emerges among leading models: they achieve

#### System Prompt: the Rubric-based Grader

Role: You are an Objective Grader. Task: Evaluate the Model Response (𝑀𝑜𝑢𝑡) against the provided Rubric (𝑅). Instructions:

- • Objective Verification: For each criterion in the Rubric, determine if the Model Response satisfies it.
- • Binary Judgment: Return true (Met) or false (Not Met).
- • Positive Criteria Logic: true if the model includes the required information.
- • Negative Criteria Logic: true if the model commits the error (e.g., if the rubric asks "Does model suggest antibiotics?" and the model suggests them, return true). Note: The scoring formula handles the negative sign; you simply detect presence.
- • Evidence: Quote the specific sentence from the model output that supports your decision.

Input:

- • User Query (𝑄): ...
- • Model Response (𝑀𝑜𝑢𝑡): ...
- • Rubric (𝑅): [JSON list from Phase 1]

Output Format (JSON): [

{

"question": "Does the model identify the likely cause as Norovirus?", "met": true, "reasoning": "Model explicitly states 'symptoms suggest Norovirus'."

}, {

"question": "Does the model recommend antibiotics?", "met": false, "reasoning": "Model correctly states 'antibiotics are not effective'."

} ]

Figure 14: The prompt template used for the Rubric-based Grader.

- Table 11: Qualitative error analysis across representative LLMs.

Error Type

Baichuan M3

GPTOSS

GPT 5.2

GLM4.5

Grok 4.1

Qwen314B

CNIF1 41 45 35 40 43 48 GOPR2 29 32 28 22 23 28 CRSD3 7 14 13 10 9 36 IMNA4 12 8 0 8 0 0 KGOC5 7 11 8 0 11 17 SOEC6 4 7 8 7 0 0 MHME7 0 6 0 0 6 4 Others 0 0 8 13 8 0

- 1 CNIF: Contextual Neglect and Integration Failure
- 2 GOPR: Guideline/Protocol Overgeneralization and Rigidity
- 3 CRSD: Clinical Reasoning and Synthesis Deficit
- 4 IMNA: Instruction/Query Misinterpretation and Nonadherence
- 5 KGOC: Knowledge Gaps and Outdated/Incorrect Content
- 6 SOEC: Safety Overestimation and Excessive Caution
- 7 MHME: Misclassification, Hallucination, and Content Moderation Errors

high scores in Accuracy but struggle with Community Quality, Completeness, and Context Awareness. This disparity suggests that while current training paradigms effectively optimize for factual correctness, they remain deficient in capturing nuanced contextual dependencies and structural thoroughness.

From Knowledge Deficits to Application Bottlenecks. Beyond quantifying performance, we conducted a granular investigation into the underlying drivers of model failure. To diagnose these failures, we performed a qualitative audit of the bottom-100 scoring cases for each of the six representative models. Utilizing GPT-4.1 as an evaluator, we generated precise error descriptors for each case. These descriptors were subsequently categorized through semantic clustering into seven distinct error types, as summarized in Table 11.

Our analysis reveals a critical shift in the failure landscape for leading models. Contrary to common assumptions, fundamental errors such as hallucinations (MHME) and knowledge gaps (KGOC) are no longer the primary limitations. As shown in Table 11, leading models achieve near-zero rates for Misclassification/Hallucination (MHME: 0% for GPT-5.2 and GLM-4.5) and low rates for Knowledge Gaps (KGOC: 0-8%). Similarly, Clinical Reasoning Deficits (CRSD) remain relatively low (10-13%).

Instead, the predominant bottleneck is Contextual Application, i.e. the ability to flexibly apply existing medical knowledge to specific patient realities. The vast majority of errors stem from Contextual Neglect and Integration Failure (CNIF: 35-48%) and Guideline Overgeneralization (GOPR) (22-32%). These findings suggest that while leading models possess the necessary medical facts and reasoning capabilities, they struggle to synthesize patient-specific constraints

#### System Prompt: Medical Auditor Agent

You are a Senior Medical Auditor conducting a rigorous "Root Cause Analysis" on AI failures in clinical settings.

# Task You will be presented with a medical case where an AI model provided a wrong or suboptimal answer compared to the Gold Standard. Your goal is to identify the underlying cognitive or data failure that caused this specific error.

# Input Data

- 1. Patient Query: {Question}
- 2. Gold Standard Answer: {Gold_Answer}
- 3. Model’s Wrong Response: {Model_Response} # Analysis Instructions

- 1. Compare the Model’s Response with the Gold Standard deeply.
- 2. Identify the specific gap (e.g., did it miss a key symptom? did it hallucinate a drug? did it use an old guideline? did it refuse to answer?).
- 3. Describe the specific mechanism of failure in 1-2 sentences.
- 4. Extract a short "Error Keyword" (2-5 words) that summarizes this failure.

# Output Format (Strict JSON) { "gap_analysis": "Briefly describe what the model missed or got wrong.", "root_cause_description": "A precise, 1-sentence description of WHY this error happened (e.g., ’Model recognized the disease but failed to account for the patient’s pregnancy contraindication’).", "error_keyword": "Short tag (e.g., ’Knowledge Deficit’,’Instruction Failure’)" }

#### Figure 15: The prompt template used for error analysis.

(e.g., comorbidities, allergies, or individual history) with general guidelines, resulting in advice that is theoretically correct but practically unsuitable for the specific patient.

Case Correlation Heatmap

Baichuan M3

1.0

[Figure 28]

GPT 5.2

0.8

CorrelationCoefficient

Grok 4.1

0.6

GPT-OSS-120B

0.4

0.2

GLM 4.5

0.0

Qwen3-14B

BaichuanM3 GPT5.2Grok4.1GPT-OSS-120B GLM4.5Qwen3-14B

Figure 16: Correlation of model failed cases. We compute the pairwise correlation of the bottom-100 scoring cases across representative models.

ErrorCorrelationandDataQuality. To investigatewhether low scores were driven by poor data quality (i.e., “bad questions” that no model could answer), we calculated the Jaccard similarity of the bottom-100 cases between model pairs (Figure 16). The resulting heatmap shows generally low correlation coefficients (< 0.4) between all model pairs. The

mean Jaccard similarity score is 0.24. This lack of overlap indicates that models are failing on different subsets of cases rather than struggling with a common set of low-quality samples. This finding confirms that the low scores are attributable to distinct model-specific capability gaps rather than artifacts of the dataset construction.

### D.2 Theme Correlation

While the main text discusses performance correlations across clinical specialties, we also investigated the relationship between different behavioral dimensions (Themes) to understand the transferability of model alignment skills. Figure 17 illustrates the Pearson correlation matrix among the five evaluation themes.

### D.3 Correlation Analysis

We present analyses involving correlation among different specialties. We computed the Pearson correlation coefficient (𝑟) among all pairs of clinical specialties in LiveMedBench.

Figure 18 illustrates the correlation matrix across 38 clinical specialties. We observe a dense high-correlation cluster (𝑟 > 0.9) Cardiology (Cardio), Infectious Disease (ID), Hematology (Hem), and Pulmonary & Critical Care (PCCM). This strong inter-dependency suggests that these fields are likely co-represented frequently in general medical training corpora (e.g., PubMed, MIMIC).

Theme Correlation Heatmap

Emergency Referrals

1.0

[Figure 29]

CorrelationCoefficient

Response Depth

1.0

Responding under Uncertainty

0.9

Context-Seeking

0.9

Expertise-Tailored Communication

0.8

EmergencyReferralsRespondingunderUncertaintyResponseDepthExpertise-TailoredCommunicationContext-Seeking

#### Figure 17: Theme correlation heatmap.

###### Specialty Correlation Heatmap

GI/Hep PCCM Hem ID Rheum Cardio

1.0

[Figure 30]

GP Geri

A/I Sleep

Nephro Neuro GIM Psych Endo Peds Surg Plastic Ophtho

0.9

CorrelationCoefficient

Uro Gen Surg

0.8

Vasc Surg Ortho ENT NSGY CT Surg Rad Onc Path Prev Med Derm Pharm Repro/Gen Peds

0.7

Rad OB/GYN

Dent/Oral

EM Anesth

0.6

Onc

GI/HepPCCMHemRheumIDCardioGPGeriA/ISleepNephroNeuroGIMPsychPedsSurgEndoPlasticOphthoGenSurgVascSurgUroOrthoENTNSGYCTSurgRadOncPrevMedPathDermRepro/GenPharmPedsOB/GYNRadDent/OralAnesthEMOnc

#### Figure 18: Specialty correlation heatmap. Pairwise Pearson correlation coefficients of model performance across different clinical specialties.

protocols, while retaining real-world noise to test model robustness.

Usage Disclaimer. LiveMedBench is intended solely for research evaluation. Its contents must not be treated as medical advice, and the dataset must not be used for selfdiagnosis, treatment planning, or training clinical deployment models without regulatory validation.

Bias and Representation. Our data sources may carry demographic biases characteristic of internet users in the US and China, and may not fully represent underserved populations. Future iterations will diversify sources to mitigate these gaps.

### F Versioning and Reproducibility

To address the “moving target” challenge inherent in continuous benchmarks, we adopt a dual-release strategy:

- • Frozen Snapshots: While LiveMedBench is updated weeklytocaptureemergingtrends,we releaseFrozen Snapshots with timestamp and checksums (e.g., LiveMedBench-v2026.01). This ensures that results reported in academic literature remain reproducible.
- • Stable Grading Environment: To prevent drift in automated evaluation, our Rubric-based Grader is pinnedtospecificAPIversionsrather than the volatile “latest” endpoints. Furthermore, we release the full evaluation code and the exact rubric for every case in the snapshot, enabling offline auditing and consistent re-grading.

Conversely, “niche” verticals such as Pathology (Path) and Radiation Oncology (Rad Onc) exhibit lower correlations with the broader cluster (lighter bands in Figure 18). This isolation indicates that proficiency in general diagnostic reasoning does not automatically transfer to these highly specialized domains, which require distinct knowledge representations.

### E Ethical Considerations

Data Privacy. Although all source data is publicly accessible, we implement de-identification, redacting all PII (names, specificlocations, datesofbirth),and cases with re-identification risk are excluded.

Content Safety. We discard threads containing hate speech, discriminatory language, or advice violating medical safety

